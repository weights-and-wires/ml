---
title: "Out-of-bag error: a free test set"
slug: 046-out-of-bag-error
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Each tree never saw about a third of the rows, so scoring each row with only the trees that missed it gives a held-out error for free.
minutes: 35
gpu: false
objectives:
  - Explain why a row's score is honest when only the trees that never saw that row vote on it.
  - Compute the out-of-bag predictions and the out-of-bag error of a bagged ensemble by hand and in code.
  - Explain why the out-of-bag error is slightly pessimistic, and why a row can end up with no vote at all.
  - Say why the out-of-bag error tracks the test error while the training error does not.
prereqs:
  - 045-bootstrap-and-bagging
  - 002-honest-scores
exercise: 046-out-of-bag-error
---

# Out-of-bag error: a free test set

You trained 80 deep trees on all 200 of your rows. How well will the ensemble do on flats it has never seen? You have no flats left to test on, because every row went into training.

The obvious check is to score the ensemble on the rows you trained it with. That score lies. Each tree has seen most of those rows, so it can recite their answers, and the training error comes out far too low. The chapter "Honest scores: training error, test error, and holding data back" explained the lie, and the usual cure is to hold some rows back. That costs you data: hold back 20% and the trees train on 160 rows, not 200.

Bagging offers a cure that costs nothing. Each tree was trained on a bootstrap sample that left out about a third of the rows, so each tree already has rows it never saw. You can use those rows as a test set.

## What you already know

You already use these ideas:

- Training error flatters a model, because the model has seen the answers ("Honest scores: training error, test error, and holding data back").
- A bootstrap sample draws $n$ rows with replacement, and the rows it misses are out of the bag ("Bootstrap samples and bagging").
- A row is out of the bag for any one tree with chance about $0.368$.
- The mean squared error is the average of the squared misses.
- A bagged prediction is the average of the trees' predictions.

## Picture: each row asks only the trees that missed it

Take four rows, $x = [1, 2, 3, 4]$ and $y = [0, 10, 10, 0]$, and two stumps. Tree 1 was trained on the draw `[0, 0, 1, 1]`, so it saw rows 0 and 1, and it splits at $1.5$ with leaf values $0$ and $10$. Tree 2 was trained on `[2, 2, 3, 3]`, so it saw rows 2 and 3, and it splits at $3.5$ with leaf values $10$ and $0$. Figure 1 lays the four rows against the two trees.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A grid of four rows by two trees showing which cells are in the bag and which are out of the bag, with the held-out prediction of 10 in each out-of-bag cell and squared errors 100, 0, 0, 100 averaging 50">
<text x="15.0" y="22.0" class="fig-bold">Row</text>
<text x="205.0" y="22.0" text-anchor="middle" class="fig-bold">Tree 1</text>
<text x="205.0" y="38.0" text-anchor="middle" class="fig-small">draw [0, 0, 1, 1]</text>
<text x="325.0" y="22.0" text-anchor="middle" class="fig-bold">Tree 2</text>
<text x="325.0" y="38.0" text-anchor="middle" class="fig-small">draw [2, 2, 3, 3]</text>
<text x="437.0" y="22.0" text-anchor="middle" class="fig-bold">Held-out</text>
<text x="437.0" y="38.0" text-anchor="middle" class="fig-small">average</text>
<text x="537.0" y="22.0" text-anchor="middle" class="fig-bold">Squared</text>
<text x="537.0" y="38.0" text-anchor="middle" class="fig-small">error</text>
<text x="15.0" y="82.0" class="fig-small">row 0: x = 1, y = 0</text>
<rect x="150" y="52" width="110" height="44" class="fill-sunk stroke-muted dashed"/>
<text x="205.0" y="79.0" text-anchor="middle" class="fig-small">in bag: ignored</text>
<rect x="270" y="52" width="110" height="44" class="fill-soft-b stroke-b"/>
<text x="325.0" y="79.0" text-anchor="middle" class="fig-bold">predicts 10</text>
<text x="437.0" y="79.0" text-anchor="middle" class="fig-bold">10</text>
<text x="537.0" y="79.0" text-anchor="middle" class="fig-bold">100</text>
<text x="15.0" y="134.0" class="fig-small">row 1: x = 2, y = 10</text>
<rect x="150" y="104" width="110" height="44" class="fill-sunk stroke-muted dashed"/>
<text x="205.0" y="131.0" text-anchor="middle" class="fig-small">in bag: ignored</text>
<rect x="270" y="104" width="110" height="44" class="fill-soft-b stroke-b"/>
<text x="325.0" y="131.0" text-anchor="middle" class="fig-bold">predicts 10</text>
<text x="437.0" y="131.0" text-anchor="middle" class="fig-bold">10</text>
<text x="537.0" y="131.0" text-anchor="middle" class="fig-bold">0</text>
<text x="15.0" y="186.0" class="fig-small">row 2: x = 3, y = 10</text>
<rect x="150" y="156" width="110" height="44" class="fill-soft-b stroke-b"/>
<text x="205.0" y="183.0" text-anchor="middle" class="fig-bold">predicts 10</text>
<rect x="270" y="156" width="110" height="44" class="fill-sunk stroke-muted dashed"/>
<text x="325.0" y="183.0" text-anchor="middle" class="fig-small">in bag: ignored</text>
<text x="437.0" y="183.0" text-anchor="middle" class="fig-bold">10</text>
<text x="537.0" y="183.0" text-anchor="middle" class="fig-bold">0</text>
<text x="15.0" y="238.0" class="fig-small">row 3: x = 4, y = 0</text>
<rect x="150" y="208" width="110" height="44" class="fill-soft-b stroke-b"/>
<text x="205.0" y="235.0" text-anchor="middle" class="fig-bold">predicts 10</text>
<rect x="270" y="208" width="110" height="44" class="fill-sunk stroke-muted dashed"/>
<text x="325.0" y="235.0" text-anchor="middle" class="fig-small">in bag: ignored</text>
<text x="437.0" y="235.0" text-anchor="middle" class="fig-bold">10</text>
<text x="537.0" y="235.0" text-anchor="middle" class="fig-bold">100</text>
<line x1="395" y1="262" x2="585" y2="262" class="stroke-ink"/>
<text x="490.0" y="284.0" text-anchor="middle" class="fig-bold">mean = 50</text>
</svg>
Figure 1. Only the shaded orange cells count. Each row is scored by the one tree that never saw it, and the dashed grey cells are skipped. The four squared errors 100, 0, 0, and 100 average 50.
```

Read it row by row. Row 0 is in tree 1's bag, so tree 1 has seen its label and does not get a vote. Tree 2 never saw row 0, so tree 2 votes. Its prediction at $x = 1$ is 10, since $1 \le 3.5$ puts it in the left leaf. Row 1 is the same story. Rows 2 and 3 are in tree 2's bag, so tree 1 votes on them, and at $x = 3$ and $x = 4$ it predicts 10, since both are above 1.5.

The squared errors are $(10-0)^2 = 100$, $(10-10)^2 = 0$, $0$, and $100$, and their mean is 50. That is the **out-of-bag error**, usually shortened to OOB error.

This example shows the mechanism, and it is not a good model. Each stump saw two different $x$ values and guessed wildly for the others, so an error of 50 is worse than the 25 you would get by always predicting the mean of $y$, which is 5. Later in the chapter a larger run shows the method working.

## Why the score is honest

Think about what happens to row $i$. A tree that missed row $i$ never used its label $y_i$ during training, so its prediction at $x_i$ is a genuine guess at something it has not seen. That is exactly the position of a test row. If you average the guesses of all the trees that missed row $i$, you get a bagged prediction for a row that none of those trees saw.

Now compare that with the lie. Average all the trees on every row, and each row's score includes trees that memorised it. In the example above, the average of both stumps on every row is $[5, 10, 10, 5]$, with errors $25, 0, 0, 25$ and a mean of $12.5$. That is a training error. It is four times smaller than the honest 50, because half of its votes come from the tree that saw the answer.

In symbols, let $S$ be the rows that at least one tree missed. For each row $i$ in $S$, let $\hat y_i$ be the mean prediction of the trees that did not use row $i$. Then

$$
\text{OOB error} = \frac{1}{|S|} \sum_{i \in S} \left( y_i - \hat y_i \right)^2 .
$$

Read it aloud: score each row with its own set of honest voters, square the miss, and average over the rows that have any voter at all. In numpy, you keep a running total and a vote count for each row:

```python
total[missed] += _predict_stump(stump, X[missed])
count[missed] += 1
pred = total[count > 0] / count[count > 0]
```

The usual mistake is averaging every tree on every row. That computes the training error, and you will not see it fail, because the number comes out small and looks good.

> [!KEY]
> A row's out-of-bag prediction uses only the trees that never saw that row, so the out-of-bag error is a held-out error. It costs no extra data and no extra training.

## Does it match a real test error?

Figure 2 answers it with a larger run. It uses 200 noisy rows from the sine market of the earlier chapters, and trees grown to full depth, because deep trees memorise their rows, which makes the gap between training and honest error large. It also draws 3000 fresh rows as an independent test set.

```figure
<svg viewBox="0 0 600 295" role="img" aria-label="Training error, out-of-bag error, and test error of a bagged ensemble of full-depth trees as the number of trees grows from 1 to 80: training error stays near 0.02 while out-of-bag and test error both settle near 0.13">
<rect x="60.0" y="25.0" width="480.0" height="220.0" class="stroke-grid" fill="none"/>
<line x1="60.0" y1="245.0" x2="540.0" y2="245.0" class="stroke-grid" />
<text x="52.0" y="249.0" text-anchor="end" class="fig-small">0</text>
<line x1="60.0" y1="190.0" x2="540.0" y2="190.0" class="stroke-grid" />
<text x="52.0" y="194.0" text-anchor="end" class="fig-small">0.05</text>
<line x1="60.0" y1="135.0" x2="540.0" y2="135.0" class="stroke-grid" />
<text x="52.0" y="139.0" text-anchor="end" class="fig-small">0.1</text>
<line x1="60.0" y1="80.0" x2="540.0" y2="80.0" class="stroke-grid" />
<text x="52.0" y="84.0" text-anchor="end" class="fig-small">0.15</text>
<line x1="60.0" y1="25.0" x2="540.0" y2="25.0" class="stroke-grid" />
<text x="52.0" y="29.0" text-anchor="end" class="fig-small">0.2</text>
<line x1="60.0" y1="25.0" x2="60.0" y2="245.0" class="stroke-grid" />
<text x="60.0" y="262.0" text-anchor="middle" class="fig-small">1</text>
<line x1="312.2" y1="25.0" x2="312.2" y2="245.0" class="stroke-grid" />
<text x="312.2" y="262.0" text-anchor="middle" class="fig-small">10</text>
<line x1="540.0" y1="25.0" x2="540.0" y2="245.0" class="stroke-grid" />
<text x="540.0" y="262.0" text-anchor="middle" class="fig-small">80</text>
<text x="300.0" y="282.0" text-anchor="middle" class="fig-small">number of trees B (log scale)</text>
<line x1="60.0" y1="146.0" x2="540.0" y2="146.0" class="stroke-muted dashed" />
<text x="80.0" y="162.0" class="fig-small">noise floor 0.09</text>
<polyline points="60.0,187.8 135.9,199.7 180.3,206.9 211.9,206.2 236.3,212.8 256.3,215.2 273.2,215.6 287.8,215.8 300.7,217.0 312.2,218.3 322.7,218.6 332.2,219.3 341.0,219.9 349.1,220.2 356.6,219.9 363.7,220.5 370.3,220.7 376.6,221.1 382.5,221.8 388.1,222.5 393.5,222.6 398.6,222.8 403.5,222.6 408.1,222.4 412.6,222.6 416.9,222.9 421.0,222.8 425.0,222.9 428.8,223.1 432.6,223.1 436.2,223.3 439.6,223.7 443.0,223.7 446.3,223.8 449.4,223.5 452.5,223.6 455.5,223.8 458.5,224.0 461.3,224.2 464.1,224.2 466.8,224.3 469.4,224.6 472.0,224.4 474.5,224.5 477.0,224.5 479.4,224.5 481.7,224.8 484.0,225.0 486.3,225.0 488.5,225.1 490.7,225.0 492.8,224.9 494.9,225.0 496.9,225.1 499.0,225.1 500.9,225.1 502.9,225.1 504.8,225.0 506.6,225.0 508.5,225.1 510.3,225.2 512.1,225.1 513.8,225.1 515.6,225.0 517.3,224.9 518.9,225.0 520.6,225.0 522.2,224.8 523.8,224.8 525.4,224.8 526.9,224.8 528.5,224.9 530.0,224.8 531.5,224.9 532.9,224.9 534.4,224.9 535.8,224.9 537.2,225.0 538.6,225.1 540.0,225.2" fill="none" class="stroke-d" style="stroke-width:2.5"/>
<polyline points="60.0,35.9 135.9,58.6 180.3,65.6 211.9,70.1 236.3,80.7 256.3,86.2 273.2,89.1 287.8,90.3 300.7,91.7 312.2,92.2 322.7,93.3 332.2,94.6 341.0,96.8 349.1,99.2 356.6,99.0 363.7,99.7 370.3,100.0 376.6,100.5 382.5,100.3 388.1,100.5 393.5,100.8 398.6,100.8 403.5,100.3 408.1,100.9 412.6,100.3 416.9,100.1 421.0,99.9 425.0,100.5 428.8,100.2 432.6,100.5 436.2,100.9 439.6,100.9 443.0,101.0 446.3,101.0 449.4,101.4 452.5,101.1 455.5,101.0 458.5,100.8 461.3,101.1 464.1,101.0 466.8,100.9 469.4,100.7 472.0,101.1 474.5,101.2 477.0,101.3 479.4,101.3 481.7,100.9 484.0,100.9 486.3,100.9 488.5,100.6 490.7,101.0 492.8,101.2 494.9,100.9 496.9,100.8 499.0,100.8 500.9,100.9 502.9,100.9 504.8,101.3 506.6,101.8 508.5,101.5 510.3,101.3 512.1,101.3 513.8,101.3 515.6,101.8 517.3,101.9 518.9,101.9 520.6,101.7 522.2,101.9 523.8,102.1 525.4,102.3 526.9,102.3 528.5,102.1 530.0,102.2 531.5,102.2 532.9,102.3 534.4,102.4 535.8,102.6 537.2,102.6 538.6,102.4 540.0,102.5" fill="none" class="stroke-a" style="stroke-width:2.5" stroke-dasharray="6 4"/>
<polyline points="60.0,86.1 135.9,81.8 180.3,70.9 211.9,71.2 236.3,71.3 256.3,74.0 273.2,71.9 287.8,73.0 300.7,75.4 312.2,78.0 322.7,81.3 332.2,82.4 341.0,83.1 349.1,85.3 356.6,86.3 363.7,91.8 370.3,92.5 376.6,92.0 382.5,91.6 388.1,94.0 393.5,92.8 398.6,95.2 403.5,95.2 408.1,94.6 412.6,95.5 416.9,95.2 421.0,94.9 425.0,94.6 428.8,94.4 432.6,95.5 436.2,95.9 439.6,97.3 443.0,96.8 446.3,96.9 449.4,97.0 452.5,98.0 455.5,98.8 458.5,100.2 461.3,101.2 464.1,100.7 466.8,101.4 469.4,102.3 472.0,101.8 474.5,101.4 477.0,101.2 479.4,101.2 481.7,101.1 484.0,101.3 486.3,102.4 488.5,102.8 490.7,102.7 492.8,102.9 494.9,102.6 496.9,102.6 499.0,102.6 500.9,102.8 502.9,102.5 504.8,102.5 506.6,102.7 508.5,103.3 510.3,103.3 512.1,102.9 513.8,103.2 515.6,103.1 517.3,103.3 518.9,103.2 520.6,103.1 522.2,103.0 523.8,102.9 525.4,102.7 526.9,103.0 528.5,102.9 530.0,102.9 531.5,102.9 532.9,103.2 534.4,103.0 535.8,103.0 537.2,102.7 538.6,103.1 540.0,102.9" fill="none" class="stroke-b" style="stroke-width:2.5"/>
<text x="546.0" y="229.2" class="fig-small">training</text>
<text x="546.0" y="118.9" class="fig-small">OOB</text>
<text x="546.0" y="96.5" class="fig-small">test</text>
</svg>
Figure 2. Full-depth trees on 200 noisy rows. The purple training error stays low because each tree has memorised most of its own rows. The orange out-of-bag error, computed with no rows set aside, settles onto the dashed blue test error by about 50 trees, and sits a little above it before that. With only a few trees the two disagree and the orange line is jumpy, because each row has only a handful of held-out voters. Both sit above the noise floor of 0.09.
```

Compare the three curves at 80 trees. The training error is $0.018$, far below the noise floor of $0.09$ that the wobble of $0.3$ in every rent makes unavoidable, which means the ensemble has memorised its rows. The OOB error is $0.129$, and the test error on the 3000 fresh rows is $0.130$. The OOB error found the true error without a single row set aside. The exercise uses stumps, so that the code is short. Stumps do not memorise, and in the exercise the gap between training and OOB error is much smaller. The method is the same for any tree.

There is a small catch. A row is missed by about $0.368 \times B$ trees, so with 80 trees each row is scored by about 29 of them, not by all 80. A smaller ensemble is a little noisier, so the OOB error runs slightly above the error of the full ensemble. In the stump exercise the OOB error is $0.376$ against a test error of $0.366$. In the deep-tree run, an ensemble of 29 trees scores $0.132$ on the test set against $0.130$ for 80, so the effect is small.

## When a row has no vote

A row is in every one of $B$ bags with chance $0.632^B$, and then no tree can vote on it. Its `count` is 0, and dividing by zero gives `nan`. Figure 3 shows how quickly this goes away.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bars showing the share of rows with no out-of-bag tree, 63 percent for 1 tree, 25 percent for 3, 10 percent for 5, 1 percent for 10, and 0.01 percent for 20">
<line x1="60.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-muted" />
<line x1="60.0" y1="172.9" x2="540.0" y2="172.9" class="stroke-grid" />
<text x="52.0" y="176.9" text-anchor="end" class="fig-small">20%</text>
<line x1="60.0" y1="115.7" x2="540.0" y2="115.7" class="stroke-grid" />
<text x="52.0" y="119.7" text-anchor="end" class="fig-small">40%</text>
<line x1="60.0" y1="58.6" x2="540.0" y2="58.6" class="stroke-grid" />
<text x="52.0" y="62.6" text-anchor="end" class="fig-small">60%</text>
<rect x="78.0" y="49.4" width="60" height="180.6" class="fill-b"/>
<text x="108.0" y="42.4" text-anchor="middle" class="fig-bold">63%</text>
<text x="108.0" y="250.0" text-anchor="middle" class="fig-small">B = 1</text>
<rect x="174.0" y="157.8" width="60" height="72.2" class="fill-b"/>
<text x="204.0" y="150.8" text-anchor="middle" class="fig-bold">25%</text>
<text x="204.0" y="250.0" text-anchor="middle" class="fig-small">B = 3</text>
<rect x="270.0" y="201.2" width="60" height="28.8" class="fill-b"/>
<text x="300.0" y="194.2" text-anchor="middle" class="fig-bold">10%</text>
<text x="300.0" y="250.0" text-anchor="middle" class="fig-small">B = 5</text>
<rect x="366.0" y="227.1" width="60" height="2.9" class="fill-b"/>
<text x="396.0" y="220.1" text-anchor="middle" class="fig-bold">1.0%</text>
<text x="396.0" y="250.0" text-anchor="middle" class="fig-small">B = 10</text>
<rect x="462.0" y="228.5" width="60" height="1.5" class="fill-b"/>
<text x="492.0" y="221.5" text-anchor="middle" class="fig-bold">0.01%</text>
<text x="492.0" y="250.0" text-anchor="middle" class="fig-small">B = 20</text>
</svg>
Figure 3. The share of rows that sit in every bag, 0.632 raised to the power B, so no tree can score them. It falls fast: 10 trees leave about 1 row in 100 without a held-out vote, and 20 trees leave about 1 in 10,000.
```

With a single tree, 63% of the rows have no vote. With 10 trees about 1 row in 100 has none, and with 20 trees about 1 in 10,000. So you skip the rows with `count == 0`, and you raise an error in the case where none of the rows has a vote, which only happens with a handful of trees.

> [!WARNING]
> A row that sits in every bag has `count == 0`, and its prediction is `nan`. Average the squared error only over the rows with `count > 0`, or the whole error becomes `nan`.

> [!RECAP]
> - A tree never saw the rows its bootstrap sample missed, about 36.8% of them, so those rows are a test set for that tree.
> - Score each row with only the trees that missed it, then average the squared misses.
> - Using every tree on every row gives the training error, which flatters the model.
> - OOB error tracks the test error, and runs slightly high because each row is scored by about $B/3$ trees.
> - A row in every bag has no vote and must be skipped. This is rare once $B$ reaches 10 or more.

## Build it step by step

`_fit_stump`, `_predict_stump`, `bootstrap_indices`, `fit_bagged_stumps`, and `bagged_predict` are already written. You write three functions in this order.

1. **`oob_mask(idx, n)`** takes one draw `idx` and the number of rows `n`, and returns a boolean array of shape `(n,)`, True for each row the draw missed. Start from `in_bag = np.zeros(n, dtype=bool)`, set `in_bag[idx] = True`, and return `~in_bag`. A repeated index writes the same slot twice, which is harmless. The usual mistake is returning `in_bag`, which flips the meaning. Write it like this:

    ```python
    def oob_mask(idx, n):
        # True where the row never appears in idx.
    ```

2. **`oob_predictions(X, stumps, in_bag_indices)`** returns `(pred, count)`, both of shape `(n,)`. For each stump, get `missed = oob_mask(idx, n)`, add `_predict_stump(stump, X[missed])` into a running `total[missed]`, and add 1 into `count[missed]`. Then divide where `count > 0` and leave `np.nan` elsewhere. The usual mistake is predicting on all of `X` and adding it to every row. Write it like this:

    ```python
    def oob_predictions(X, stumps, in_bag_indices):
        # Accumulate total and count on the missed rows only. pred is nan where count == 0.
    ```

3. **`out_of_bag_error(X, y, stumps, in_bag_indices)`** returns a float. Call `oob_predictions`, keep the rows with `count > 0`, and average the squared error there. Raise `ValueError` if no row has a vote. The usual mistake is averaging over all rows, which returns `nan` the first time one row has no vote. Write it like this:

    ```python
    def out_of_bag_error(X, y, stumps, in_bag_indices):
        # Score only rows with count > 0. Raise ValueError when there are none.
    ```

## Where this breaks

- The OOB error measures the ensemble as you built it. If you try many settings and pick the one with the lowest OOB error, the winner's score becomes optimistic, for the same reason a test set becomes optimistic when you tune on it.
- It needs the bag to leave rows out. A method where every tree sees every row, such as the boosting in the next chapter, has no out-of-bag rows and no free test set.
- With very few trees, many rows have no vote and the estimate rests on the rest, which may not be typical.

## The exercise

You implement `oob_mask`, `oob_predictions`, and `out_of_bag_error`. The checks confirm that the mask marks exactly the rows missing from a draw, that `oob_predictions` averages only the stumps that missed each row and returns `nan` where none did, and that the hand example from Figure 1 gives an error of 50 and not the training error of 12.5. They also confirm that rows without a vote are skipped, that the function raises an error when no row has one, and that on a seeded dataset of 500 rows the OOB error lands within 10% of the error on 500 fresh rows.
