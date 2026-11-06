---
title: "Bagging, and the free test set it leaves behind"
slug: 042-bagging-and-out-of-bag-error
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Drawing n rows with replacement gives many different training sets to average over, and the rows each draw misses score the ensemble for free.
minutes: 45
gpu: false
objectives:
  - Explain why splitting or shuffling the data fails to give different trees, and why drawing with replacement works.
  - Draw a bootstrap sample, and show that it holds about 63.2% of the distinct rows.
  - Fit a bagged ensemble of stumps and average its predictions.
  - Compute the out-of-bag error by scoring each row with only the trees that missed it.
  - Explain why the out-of-bag error tracks the test error while the training error does not, and why a row can end up with no vote.
prereqs:
  - 041-averaging-models
  - 039-grow-and-predict
  - 002-honest-scores
exercise: 042-bagging-and-out-of-bag-error
---

# Bagging, and the free test set it leaves behind

The previous chapter, "Why averaging models cuts noise, and where it stops", showed that averaging a hundred noisy trees beats trusting one. It had a catch that is easy to miss: it averaged trees that each came from their own training set. You have exactly one dataset of 100 rows, and you cannot go back to the market for another 100. How do you get different training sets out of one, without throwing data away?

The obvious fix is to train the same tree a hundred times and average. That does nothing, because the same rows always give the same tree. A hundred identical trees have correlation $\rho = 1$, so the average has the variance of one tree. Models only differ when their training sets do.

This chapter gives a recipe for many different training sets from one dataset, called the bootstrap, and builds the ensemble from it, called bagging. Then it collects a second payoff: every training set leaves some rows out, and those rows test the ensemble without holding any data back.

## What you already know

You already use these ideas:

- A model's variance is how much its prediction moves when its training set is redrawn, and an average of $B$ models has variance $\rho\sigma^2 + (1-\rho)\sigma^2/B$ ("Why averaging models cuts noise, and where it stops").
- A stump is a tree with one split and two leaves, and each leaf predicts the mean of its training targets.
- Training error flatters a model, because the model has seen the answers ("Honest scores: training error, test error, and holding data back").
- A probability of "not" is one minus the probability of "yes", and independent chances multiply.

## Why not just split the data, or shuffle it?

Two ideas come to mind first, and both fail.

You could cut the 100 rows into 10 chunks of 10 and train one tree per chunk. A tree trained on 10 rows is far noisier than one trained on 100, so each model's $\sigma^2$ goes up and the average pays for it. You could instead shuffle the rows, but a tree does not care about row order, so it finds the identical split.

What you need is a copy with the same number of rows that really contains different rows. Sampling with replacement does that.

## Picture: drawing rows back out of a hat

Write each row on a ticket and put all the tickets in a hat. To build one new training set, pull a ticket, copy the row onto your list, and put the ticket back. Repeat until the list has $n$ rows, the same size as the original. Figure 1 shows three such draws from a dataset of ten rows.

```figure
<svg viewBox="0 0 600 385" role="img" aria-label="A dataset of ten numbered rows and three bootstrap draws of ten rows each, with repeated rows stacked and missing rows shown as empty dashed boxes">
<text x="10.0" y="28.0" class="fig-bold">Dataset</text>
<rect x="45.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="67.0" y="54.0" text-anchor="middle" class="fig-small">0</text>
<rect x="97.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="119.0" y="54.0" text-anchor="middle" class="fig-small">1</text>
<rect x="149.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="171.0" y="54.0" text-anchor="middle" class="fig-small">2</text>
<rect x="201.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="223.0" y="54.0" text-anchor="middle" class="fig-small">3</text>
<rect x="253.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="275.0" y="54.0" text-anchor="middle" class="fig-small">4</text>
<rect x="305.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="327.0" y="54.0" text-anchor="middle" class="fig-small">5</text>
<rect x="357.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="379.0" y="54.0" text-anchor="middle" class="fig-small">6</text>
<rect x="409.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="431.0" y="54.0" text-anchor="middle" class="fig-small">7</text>
<rect x="461.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="483.0" y="54.0" text-anchor="middle" class="fig-small">8</text>
<rect x="513.0" y="38" width="44" height="22" class="fill-soft-a stroke-a"/>
<text x="535.0" y="54.0" text-anchor="middle" class="fig-small">9</text>
<text x="10.0" y="92.0" class="fig-bold">Draw 1: 7 distinct rows in bag, 3 out of bag</text>
<rect x="45.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="45.0" y="128.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="45.0" y="106.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="67.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">0</text>
<rect x="97.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="119.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">1</text>
<rect x="149.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="171.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">2</text>
<rect x="201.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="223.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">3</text>
<rect x="253.0" y="150.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="275.0" y="166.0" text-anchor="middle" class="fig-small">4</text>
<rect x="305.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="327.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">5</text>
<rect x="357.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="379.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">6</text>
<rect x="409.0" y="150.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="431.0" y="166.0" text-anchor="middle" class="fig-small">7</text>
<rect x="461.0" y="150.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="461.0" y="128.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="483.0" y="166.0" text-anchor="middle" class="fig-small fill-paper">8</text>
<rect x="513.0" y="150.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="535.0" y="166.0" text-anchor="middle" class="fig-small">9</text>
<line x1="40" y1="172" x2="557" y2="172" class="stroke-muted"/>
<text x="10.0" y="192.0" class="fig-bold">Draw 2: 4 distinct rows in bag, 6 out of bag</text>
<rect x="45.0" y="250.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="67.0" y="266.0" text-anchor="middle" class="fig-small">0</text>
<rect x="97.0" y="250.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="119.0" y="266.0" text-anchor="middle" class="fig-small">1</text>
<rect x="149.0" y="250.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="171.0" y="266.0" text-anchor="middle" class="fig-small">2</text>
<rect x="201.0" y="250.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="223.0" y="266.0" text-anchor="middle" class="fig-small">3</text>
<rect x="253.0" y="250.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="275.0" y="266.0" text-anchor="middle" class="fig-small">4</text>
<rect x="305.0" y="250.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="305.0" y="228.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="305.0" y="206.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="327.0" y="266.0" text-anchor="middle" class="fig-small fill-paper">5</text>
<rect x="357.0" y="250.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="357.0" y="228.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="357.0" y="206.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="379.0" y="266.0" text-anchor="middle" class="fig-small fill-paper">6</text>
<rect x="409.0" y="250.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="431.0" y="266.0" text-anchor="middle" class="fig-small fill-paper">7</text>
<rect x="461.0" y="250.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="483.0" y="266.0" text-anchor="middle" class="fig-small">8</text>
<rect x="513.0" y="250.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="513.0" y="228.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="513.0" y="206.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="535.0" y="266.0" text-anchor="middle" class="fig-small fill-paper">9</text>
<line x1="40" y1="272" x2="557" y2="272" class="stroke-muted"/>
<text x="10.0" y="292.0" class="fig-bold">Draw 3: 7 distinct rows in bag, 3 out of bag</text>
<rect x="45.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="45.0" y="328.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="67.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">0</text>
<rect x="97.0" y="350.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="119.0" y="366.0" text-anchor="middle" class="fig-small">1</text>
<rect x="149.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="171.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">2</text>
<rect x="201.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="223.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">3</text>
<rect x="253.0" y="350.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="275.0" y="366.0" text-anchor="middle" class="fig-small">4</text>
<rect x="305.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="327.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">5</text>
<rect x="357.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="379.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">6</text>
<rect x="409.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="409.0" y="328.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="431.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">7</text>
<rect x="461.0" y="350.0" width="44" height="22" class="fill-a stroke-paper"/>
<rect x="461.0" y="328.0" width="44" height="22" class="fill-a stroke-paper"/>
<text x="483.0" y="366.0" text-anchor="middle" class="fig-small fill-paper">8</text>
<rect x="513.0" y="350.0" width="44" height="22" class="fill-sunk stroke-muted dashed"/>
<text x="535.0" y="366.0" text-anchor="middle" class="fig-small">9</text>
<line x1="40" y1="372" x2="557" y2="372" class="stroke-muted"/>
</svg>
Figure 1. Three bootstrap draws from the same ten rows. A stacked box is a row drawn more than once, and an empty dashed box is a row that draw never picked. Every draw has ten rows and still differs from the dataset and from the others.
```

Because each ticket goes back in the hat, a row can be drawn two or three times, and because only $n$ draws are made, other rows never come up. A tree trained on a draw sees some rows twice and some not at all, so it differs from a tree trained on another draw. This recipe is the **bootstrap**, and one draw is a **bootstrap sample**. The rows a draw contains are **in the bag**, and the rows it misses are **out of the bag**.

Take $n = 4$ rows and the draw `idx = [0, 0, 2, 3]`. Row 0 appears twice, rows 2 and 3 once each, and row 1 is out of the bag. In numpy, one draw is one line:

```python
idx = rng.integers(0, n, size=n)
```

Avoid `rng.choice(n, n, replace=False)`, which is the shuffle again.

## How many rows does a draw contain?

A given row is picked by any one draw with chance $1/n$, so it is missed with chance $1 - 1/n$. The $n$ draws are independent, so the chance that a row is missed by all of them is that number multiplied $n$ times:

$$
\left(1 - \frac{1}{n}\right)^n .
$$

Read it aloud: one draw's miss chance, raised to the number of draws. It is $0.316$ at $n = 4$ (that is $(3/4)^4$, so your draw `[0, 0, 2, 3]` missing a row is ordinary), $0.349$ at $n = 10$, $0.366$ at $n = 100$, and it settles near $0.368$, which is $1/e$.

Read the result the other way around. About $36.8\%$ of the rows are out of any one bag, and about $63.2\%$ of the distinct rows are in it. A bag holds $n$ draws but only about two thirds as many different rows, because the rest are repeats.

> [!KEY]
> A bootstrap sample has as many rows as the dataset, but only about 63.2% of the distinct rows. The repeats and the gaps make each sample, and so each tree, different. The same gaps are rows that tree never saw.

## Bagging: train on every draw, then average

**Bagging** is short for bootstrap aggregating. Draw $B$ bootstrap samples, train one model on each, and predict with the average of the $B$ models. This is the plan from the previous chapter, with the bootstrap supplying the different training sets.

The trees are still correlated, because two bags from the same rows share most of them, so $\rho$ is above zero and the floor $\rho\sigma^2$ is real. In the sine market of the previous chapter, three-level trees on eight noisy points have a prediction variance of about $0.26$ at $x = 0.5$. The average of 25 bagged trees has about $0.15$, which solves to $\rho \approx 0.58$. Bagging cut the variance by about 40%, and the shared part of the noise is what remains.

The exercise bags stumps because they are short to implement, but a stump has one split, so its error is mostly bias, and averaging cannot fix bias: 15 stumps sit at a squared distance of $0.119$ from the sine curve, against $0.130$ for one. Bagging pays off with deep trees, which have low bias and high variance, and the code is the same with a deeper tree in place of `_fit_stump`.

## Out of the bag: a free test set

Now return to the rows each bag missed. You trained 80 deep trees on all 200 of your rows. How well will the ensemble do on flats it has never seen? Scoring it on the training rows lies, because each tree has seen most of them and can recite their answers. Holding rows back fixes that but costs data: hold back 20% and the trees train on 160 rows, not 200. Bagging fixes it for free, because each tree already has about a third of the rows it never saw.

### Each row asks only the trees that missed it

Take four rows, $x = [1, 2, 3, 4]$ and $y = [0, 10, 10, 0]$, and two stumps. Tree 1 was trained on the draw `[0, 0, 1, 1]`, so it saw rows 0 and 1, and it splits at $1.5$ with leaf values $0$ and $10$. Tree 2 was trained on `[2, 2, 3, 3]`, so it saw rows 2 and 3, and it splits at $3.5$ with leaf values $10$ and $0$. Figure 2 lays the four rows against the two trees.

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
Figure 2. Only the shaded orange cells count. Each row is scored by the one tree that never saw it, and the dashed grey cells are skipped. The four squared errors 100, 0, 0, and 100 average 50.
```

Row 0 is in tree 1's bag, so tree 1 has seen its label and gets no vote. Tree 2 never saw row 0, so it votes: at $x = 1$ it predicts 10. Row 1 is the same. Rows 2 and 3 are in tree 2's bag, so tree 1 votes on them, and at $x = 3$ and $x = 4$ it predicts 10.

The squared errors are $(10-0)^2 = 100$, $(10-10)^2 = 0$, $0$, and $100$, and their mean is 50. That is the **out-of-bag error**, usually shortened to OOB error. This example shows the mechanism, not a good model: each stump saw two $x$ values and guessed wildly elsewhere, so 50 is worse than the 25 you get by always predicting the mean of $y$.

### Why the score is honest

A tree that missed row $i$ never used its label $y_i$, so its prediction at $x_i$ is a genuine guess, as for a test row. Average the guesses of the trees that missed row $i$, and you have a bagged prediction for a row none of them saw.

Compare the lie: averaging both stumps on every row gives $[5, 10, 10, 5]$, with errors $25, 0, 0, 25$ and a mean of $12.5$. That is the training error, four times smaller than the honest 50, because half its votes come from the tree that saw the answer.

Let $S$ be the rows that at least one tree missed, and $\hat y_i$ the mean prediction of the trees that did not use row $i$. Then

$$
\text{OOB error} = \frac{1}{|S|} \sum_{i \in S} \left( y_i - \hat y_i \right)^2 .
$$

Read it aloud: score each row with its own honest voters, square the miss, and average over the rows that have any voter. In numpy, keep a running total and a vote count for each row:

```python
total[missed] += _predict_stump(stump, X[missed])
count[missed] += 1
pred = total[count > 0] / count[count > 0]
```

The usual mistake is averaging every tree on every row. That computes the training error, and it never looks wrong, because the number is small.

> [!KEY]
> A row's out-of-bag prediction uses only the trees that never saw that row, so the out-of-bag error is a held-out error. It costs no extra data and no extra training.

### Does it match a real test error?

Figure 3 answers with a larger run: 200 noisy rows from the sine market, trees grown to full depth (so they memorise their rows), and 3000 fresh rows as an independent test set.

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
Figure 3. Full-depth trees on 200 noisy rows. The purple training error stays low because each tree has memorised most of its own rows. The orange out-of-bag error, computed with no rows set aside, settles onto the dashed blue test error by about 50 trees, and sits a little above it before that. With only a few trees the two disagree and the orange line is jumpy, because each row has only a handful of held-out voters. Both sit above the noise floor of 0.09.
```

At 80 trees the training error is $0.018$, far below the noise floor of $0.09$ that the wobble of $0.3$ in every rent makes unavoidable, so the ensemble has memorised its rows. The OOB error is $0.129$, and the test error on the 3000 fresh rows is $0.130$. The OOB error found the true error without a single row set aside. The exercise uses stumps, which do not memorise, so its training-to-OOB gap is much smaller, but the method is the same.

There is a small catch. Each row is missed by only about $0.368 \times B$ trees, about 29 of 80. A smaller ensemble is a little noisier, so the OOB error runs slightly above that of the full ensemble: in the stump exercise it is $0.376$ against a test error of $0.366$, and in the deep-tree run 29 trees score $0.132$ on the test set against $0.130$ for 80.

### When a row has no vote

A row is in every one of $B$ bags with chance $0.632^B$, and then no tree can vote on it. With one tree, 63% of the rows have no vote; with 10 trees about 1 row in 100; with 20 trees about 1 in 10,000. So you skip rows with `count == 0`, and raise an error when no row has a vote, which only happens with a handful of trees.

> [!WARNING]
> A row that sits in every bag has `count == 0`, and its prediction is `nan`. Average the squared error only over the rows with `count > 0`, or the whole error becomes `nan`.

> [!RECAP]
> - A bootstrap sample draws $n$ rows with replacement. A row is missed with chance $(1-1/n)^n \to 1/e \approx 0.368$, so about 63.2% of the distinct rows are in the bag.
> - Bagging trains one model per bootstrap sample and averages them, which cuts variance down to a floor set by how much the samples overlap.
> - Score each row with only the trees that missed it. Using every tree on every row gives the training error, which flatters the model.
> - OOB error tracks the test error, and runs slightly high because each row is scored by about $B/3$ trees. A row in every bag has no vote and must be skipped.

## Build it step by step

`_fit_stump`, `_predict_stump`, and `in_bag_fraction` are already written. You write six functions in this order.

1. **`bootstrap_indices(n, rng)`** takes the row count and a numpy generator, and returns an integer array of shape `(n,)`. Call `rng.integers(0, n, size=n)`. The usual mistake is sampling without replacement.
2. **`fit_bagged_stumps(X, y, n_estimators, rng)`** returns a list of stump dicts and a list of index arrays. For each estimator, draw `idx`, fit `_fit_stump(X[idx], y[idx])`, and keep both. The usual mistake is dropping the indices, which the out-of-bag steps need.
3. **`bagged_predict(stumps, X)`** returns shape `(n,)`. Stack the predictions into `(len(stumps), n)` and average over axis 0. The usual mistake is averaging over axis 1, which averages across rows.
4. **`oob_mask(idx, n)`** returns a boolean array of shape `(n,)`, True for each row the draw missed. Set `in_bag[idx] = True` on a zero boolean array and return `~in_bag`. The usual mistake is returning `in_bag`, which flips the meaning.
5. **`oob_predictions(X, stumps, in_bag_indices)`** returns `(pred, count)`, both of shape `(n,)`. For each stump, get `missed = oob_mask(idx, n)`, add `_predict_stump(stump, X[missed])` into `total[missed]`, and add 1 into `count[missed]`. Divide where `count > 0` and leave `np.nan` elsewhere. The usual mistake is predicting on all of `X` and adding it to every row.
6. **`out_of_bag_error(X, y, stumps, in_bag_indices)`** returns a float. Keep the rows with `count > 0` and average the squared error there. Raise `ValueError` if no row has a vote. The usual mistake is averaging over all rows, which gives `nan` the first time one row has no vote.

## Where this breaks

- The bootstrap creates no new information, and bagging leaves bias alone: a stiff model gives a stiff average.
- Tuning settings on the lowest OOB error makes the winner's score optimistic, as tuning on a test set does.
- A method where every tree sees every row, such as the boosting in the next chapter, has no out-of-bag rows and no free test set.

## The exercise

You implement `bootstrap_indices`, `fit_bagged_stumps`, `bagged_predict`, `oob_mask`, `oob_predictions`, and `out_of_bag_error`. The checks confirm that draws repeat rows and land near an in-bag fraction of $1 - 1/e$, that each stump matches a fit on its own draw, and that bagging lowers a stump's variance. They also confirm that the hand example from Figure 2 gives 50 and not 12.5, that rows without a vote are skipped, and that on a seeded dataset the OOB error lands within 10% of the error on 500 fresh rows.
