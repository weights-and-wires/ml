---
title: "From counts to a language model, and word vectors"
slug: 067-counts-to-a-language-model
part: "Part 6 — Transformers"
summary: "Build a next-word model from counts, match it with a softmax, and train vectors so a planted neighbour is nearby."
minutes: 75
gpu: false
objectives:
  - Build a bigram table and smooth it by adding k to every cell, including cells the tally never saw.
  - Compute the mean negative log likelihood and the perplexity, and show that a uniform model has perplexity equal to the vocabulary size.
  - Show that a softmax over one-hot inputs reproduces the smoothed table when each row is a log of counts.
  - Look up an embedding and send its gradient back with a scatter-add, so a repeated index accumulates.
  - Train a context-window network and a skip-gram with negative sampling so a planted neighbour is nearer than another word.
prereqs:
  - 005-gradient-descent-and-momentum
  - 034-multilayer-perceptrons
  - 037-softmax-and-whole-network-gradients
  - 047-im2col-and-convolution-backward
exercise: 067-counts-to-a-language-model
---

# From counts to a language model, and word vectors

You keep four short lines in a notebook:

```text
big dog runs
big hound runs
small cat sits
small kitten sits
```

You want the next word. After "big dog" the next word is "runs" on every line. After "big" it is "dog" on one line and "hound" on the other. A model of these lines, given the words so far, puts a probability on every word that could come next.

The obvious rule is a tally: count how often each word follows each other word, and divide. Two things go wrong. A word that has never followed "dog" gets probability 0, and the log of 0 is not a number, so one unseen word makes the line unscorable. And "dog" and "hound" are separate columns. They show up in the same company, between "big" and "runs", but a column of counts cannot treat them as the same kind of word.

This chapter adds a count to every cell, shows that a one-layer softmax is that table written as logarithms, and replaces the one-hot column with a short vector you can train. The [next chapter](/lessons/068-next-token-loss-and-safe-steps) puts the same loss on a transformer. A token here is a whole word. Subword pieces come later ([tokenisation](/lessons/070-tokenisation)).

## What you already know

- A probability sits between 0 and 1, and the next-word probabilities add to 1. Assigning probability $p$ to the word that happened costs $-\ln p$.
- A softmax turns a row of scores into probabilities. Subtracting a constant from the row changes nothing (see [softmax and cross-entropy](/lessons/037-softmax-and-whole-network-gradients)).
- A multilayer perceptron is an affine map, a bend such as tanh, and another affine map (see [multilayer perceptrons](/lessons/034-multilayer-perceptrons)).
- A central finite difference, $(f(x+h)-f(x-h))/(2h)$, estimates a gradient one entry at a time.
- Overlapping convolution windows add their gradients into the same cell (see [convolution backward](/lessons/047-im2col-and-convolution-backward)). An embedding row does the same when one word is looked up twice.

## A tally of the next word

Take the first two lines, and put a mark called `start` in front of each so the first word has a previous word:

```text
start big dog runs
start big hound runs
```

For this arithmetic the vocabulary is `start`, `big`, `dog`, `hound`, `runs`, in that order. Its size is $V = 5$. A *bigram* is one pair of neighbours. The six pairs are (`start`, `big`), (`big`, `dog`), (`dog`, `runs`), (`start`, `big`), (`big`, `hound`), and (`hound`, `runs`).

Figure 1 is the tally. The row is the previous word and the column is the next word. `start` is followed by `big` twice. `big` is followed once by `dog` and once by `hound`. `dog` and `hound` are each followed once by `runs`. The row for `runs` is empty, because `runs` is never a previous word.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Two grids. The left grid is bigram counts for start, big, dog, hound, and runs, with a 2 where start is followed by big and zeros elsewhere including an empty row for runs. The right grid is the same table after adding 1 to every cell and dividing, so the empty row is five copies of 1/5.">
<text x="160" y="22" text-anchor="middle" class="fig-bold">counts</text>
<text x="470" y="22" text-anchor="middle" class="fig-bold">add 1, then divide</text>
<text x="88" y="40" text-anchor="middle" class="fig-small">start</text>
<text x="124" y="40" text-anchor="middle" class="fig-small">big</text>
<text x="160" y="40" text-anchor="middle" class="fig-small">dog</text>
<text x="196" y="40" text-anchor="middle" class="fig-small">hound</text>
<text x="232" y="40" text-anchor="middle" class="fig-small">runs</text>
<text x="64" y="62" text-anchor="end" class="fig-small">start</text>
<text x="64" y="98" text-anchor="end" class="fig-small">big</text>
<text x="64" y="134" text-anchor="end" class="fig-small">dog</text>
<text x="64" y="170" text-anchor="end" class="fig-small">hound</text>
<text x="64" y="206" text-anchor="end" class="fig-small">runs</text>
<rect x="70" y="48" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="88" y="70" text-anchor="middle" class="fig-small">0</text>
<rect x="106" y="48" width="36" height="36" class="fill-soft-c stroke-grid"/><text x="124" y="70" text-anchor="middle" class="fig-small">2</text>
<rect x="142" y="48" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="160" y="70" text-anchor="middle" class="fig-small">0</text>
<rect x="178" y="48" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="196" y="70" text-anchor="middle" class="fig-small">0</text>
<rect x="214" y="48" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="232" y="70" text-anchor="middle" class="fig-small">0</text>
<rect x="70" y="84" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="88" y="106" text-anchor="middle" class="fig-small">0</text>
<rect x="106" y="84" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="124" y="106" text-anchor="middle" class="fig-small">0</text>
<rect x="142" y="84" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="160" y="106" text-anchor="middle" class="fig-small">1</text>
<rect x="178" y="84" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="196" y="106" text-anchor="middle" class="fig-small">1</text>
<rect x="214" y="84" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="232" y="106" text-anchor="middle" class="fig-small">0</text>
<rect x="70" y="120" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="88" y="142" text-anchor="middle" class="fig-small">0</text>
<rect x="106" y="120" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="124" y="142" text-anchor="middle" class="fig-small">0</text>
<rect x="142" y="120" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="160" y="142" text-anchor="middle" class="fig-small">0</text>
<rect x="178" y="120" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="196" y="142" text-anchor="middle" class="fig-small">0</text>
<rect x="214" y="120" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="232" y="142" text-anchor="middle" class="fig-small">1</text>
<rect x="70" y="156" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="88" y="178" text-anchor="middle" class="fig-small">0</text>
<rect x="106" y="156" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="124" y="178" text-anchor="middle" class="fig-small">0</text>
<rect x="142" y="156" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="160" y="178" text-anchor="middle" class="fig-small">0</text>
<rect x="178" y="156" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="196" y="178" text-anchor="middle" class="fig-small">0</text>
<rect x="214" y="156" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="232" y="178" text-anchor="middle" class="fig-small">1</text>
<rect x="70" y="192" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="88" y="214" text-anchor="middle" class="fig-small">0</text>
<rect x="106" y="192" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="124" y="214" text-anchor="middle" class="fig-small">0</text>
<rect x="142" y="192" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="160" y="214" text-anchor="middle" class="fig-small">0</text>
<rect x="178" y="192" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="196" y="214" text-anchor="middle" class="fig-small">0</text>
<rect x="214" y="192" width="36" height="36" class="fill-soft-a stroke-grid"/><text x="232" y="214" text-anchor="middle" class="fig-small">0</text>
<text x="398" y="40" text-anchor="middle" class="fig-small">start</text>
<text x="434" y="40" text-anchor="middle" class="fig-small">big</text>
<text x="470" y="40" text-anchor="middle" class="fig-small">dog</text>
<text x="506" y="40" text-anchor="middle" class="fig-small">hound</text>
<text x="542" y="40" text-anchor="middle" class="fig-small">runs</text>
<text x="374" y="62" text-anchor="end" class="fig-small">start</text>
<text x="374" y="98" text-anchor="end" class="fig-small">big</text>
<text x="374" y="134" text-anchor="end" class="fig-small">dog</text>
<text x="374" y="170" text-anchor="end" class="fig-small">hound</text>
<text x="374" y="206" text-anchor="end" class="fig-small">runs</text>
<rect x="380" y="48" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="398" y="70" text-anchor="middle" class="fig-small">1/7</text>
<rect x="416" y="48" width="36" height="36" class="fill-soft-c stroke-grid"/><text x="434" y="70" text-anchor="middle" class="fig-small">3/7</text>
<rect x="452" y="48" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="470" y="70" text-anchor="middle" class="fig-small">1/7</text>
<rect x="488" y="48" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="506" y="70" text-anchor="middle" class="fig-small">1/7</text>
<rect x="524" y="48" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="542" y="70" text-anchor="middle" class="fig-small">1/7</text>
<rect x="380" y="84" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="398" y="106" text-anchor="middle" class="fig-small">1/7</text>
<rect x="416" y="84" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="434" y="106" text-anchor="middle" class="fig-small">1/7</text>
<rect x="452" y="84" width="36" height="36" class="fill-soft-c stroke-grid"/><text x="470" y="106" text-anchor="middle" class="fig-small">2/7</text>
<rect x="488" y="84" width="36" height="36" class="fill-soft-c stroke-grid"/><text x="506" y="106" text-anchor="middle" class="fig-small">2/7</text>
<rect x="524" y="84" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="542" y="106" text-anchor="middle" class="fig-small">1/7</text>
<rect x="380" y="120" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="398" y="142" text-anchor="middle" class="fig-small">1/6</text>
<rect x="416" y="120" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="434" y="142" text-anchor="middle" class="fig-small">1/6</text>
<rect x="452" y="120" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="470" y="142" text-anchor="middle" class="fig-small">1/6</text>
<rect x="488" y="120" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="506" y="142" text-anchor="middle" class="fig-small">1/6</text>
<rect x="524" y="120" width="36" height="36" class="fill-soft-c stroke-grid"/><text x="542" y="142" text-anchor="middle" class="fig-small">1/3</text>
<rect x="380" y="156" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="398" y="178" text-anchor="middle" class="fig-small">1/6</text>
<rect x="416" y="156" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="434" y="178" text-anchor="middle" class="fig-small">1/6</text>
<rect x="452" y="156" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="470" y="178" text-anchor="middle" class="fig-small">1/6</text>
<rect x="488" y="156" width="36" height="36" class="fill-soft-b stroke-grid"/><text x="506" y="178" text-anchor="middle" class="fig-small">1/6</text>
<rect x="524" y="156" width="36" height="36" class="fill-soft-c stroke-grid"/><text x="542" y="178" text-anchor="middle" class="fig-small">1/3</text>
<rect x="380" y="192" width="36" height="36" class="fill-soft-d stroke-grid"/><text x="398" y="214" text-anchor="middle" class="fig-small">1/5</text>
<rect x="416" y="192" width="36" height="36" class="fill-soft-d stroke-grid"/><text x="434" y="214" text-anchor="middle" class="fig-small">1/5</text>
<rect x="452" y="192" width="36" height="36" class="fill-soft-d stroke-grid"/><text x="470" y="214" text-anchor="middle" class="fig-small">1/5</text>
<rect x="488" y="192" width="36" height="36" class="fill-soft-d stroke-grid"/><text x="506" y="214" text-anchor="middle" class="fig-small">1/5</text>
<rect x="524" y="192" width="36" height="36" class="fill-soft-d stroke-grid"/><text x="542" y="214" text-anchor="middle" class="fig-small">1/5</text>
<text x="160" y="250" text-anchor="middle" class="fig-small">previous down, next across</text>
<text x="470" y="250" text-anchor="middle" class="fig-small">empty row becomes uniform</text>
</svg>
Figure 1. The left grid is the tally of the two lines. Four rows contain zeros, and the row for runs is empty. The right grid adds 1 to every cell and divides by the new row sum, so dog can follow start, and a context that never occurred is a flat row of 1/5.
```

Write $C_{ij}$ for how often word $j$ follows word $i$. A repeated pair has to add. The pair (`start`, `big`) happens twice, so the cell is 2, not 1:

```python
counts = np.zeros((vocab_size, vocab_size))
np.add.at(counts, (prev, nxt), 1)
```

## Zeros are a prediction

Dividing a row by its sum looks like a probability, and it fails twice. The row for `runs` sums to 0, so there is nothing to divide by. A cell that is already 0 stays 0, so $P(\text{dog} \mid \text{start}) = 0$. The log of 0 is $-\infty$, and a training step has no finite direction to follow.

*Add-k smoothing* pretends you have already seen every next word $k$ extra times, in every context:

$$
P(j \mid i) = \frac{C_{ij} + k}{n_i + kV}
$$

$n_i$ is how often $i$ was a previous word, the sum of row $i$. The $kV$ term is $k$ once per word in the vocabulary. After this, every row is positive and adds to 1, including a row that was all zeros.

Take $k = 1$. The row for `start` has $n_i = 2$, so the denominator is $2 + 5 = 7$ and the numerators are $1, 3, 1, 1, 1$. Then $P(\text{big} \mid \text{start}) = 3/7$ and $P(\text{dog} \mid \text{start}) = 1/7$. The raw tally said dog could not follow start. The row for `big` has the same denominator and numerators $1, 1, 2, 2, 1$, so dog and hound each get $2/7$. The row for `dog` has denominator 6, and $P(\text{runs} \mid \text{dog}) = 2/6 = 1/3$. The row for `runs` has denominator 5, so every next word gets $1/5$. A context you have never seen is the uniform distribution.

> [!WARNING]
> Adding $k$ only where the count is already positive leaves unseen words at 0. On the row for `start` that bug turns $(0, 2, 0, 0, 0)$ into $(0, 1, 0, 0, 0)$ after dividing. $P(\text{dog} \mid \text{start})$ stays 0, and the empty row has no denominator.

> [!KEY]
> Add $k$ to every cell, seen or not, and grow the denominator by $kV$. A context that never occurred becomes the uniform distribution, probability $1/V$ on each next word.

`keepdims=True` holds the row sum in a shape that divides a row:

```python
smoothed = counts + k
probs = smoothed / smoothed.sum(axis=1, keepdims=True)
```

## Surprise, and a uniform model

The six smoothed probabilities, in pair order, are $3/7$, $2/7$, $1/3$, $3/7$, $2/7$, and $1/3$. One pair costs $-\ln p$.

The *negative log likelihood* is the mean of those six costs. They are two copies of three values, so

$$
L = -\frac{\ln(3/7) + \ln(2/7) + \ln(1/3)}{3} = 1.0662
$$

*Perplexity* undoes the log and reads the cost as a count of equally likely choices:

$$
\mathrm{PPL} = e^{L}
$$

Here that count is $2.904$. A model that sets every entry to $1/V$ pays $\ln V$ each time, so its perplexity is $e^{\ln V} = V$. For these five words that is 5, and $\ln 5 = 1.6094$. The [next chapter](/lessons/068-next-token-loss-and-safe-steps) uses this same pair on a transformer.

> [!KEY]
> Perplexity is $e$ raised to the mean negative log likelihood. A uniform distribution over $V$ words has perplexity $V$.

## The same table, as a softmax

A one-hot vector has a 1 in one position and 0 elsewhere. The one-hot for `big` is $(0, 1, 0, 0, 0)$.

Let $W$ be $V \times V$, with row $i$ the scores for what follows word $i$. `one_hot @ W` copies that row, and a softmax turns it into probabilities. Set the entries to logs of smoothed counts:

$$
W_{ij} = \ln(C_{ij} + k)
$$

Exponentiating a log returns the smoothed count, and dividing by the sum of the row is add-k smoothing. This needs $k > 0$, because $\ln 0$ is not a number.

Check `big`. The smoothed counts are $1, 1, 2, 2, 1$, and the logs are $0, 0, \ln 2, \ln 2, 0$ with $\ln 2 = 0.6931$. Exponentiating returns $1, 1, 2, 2, 1$, which add to 7, and dividing gives $1/7, 1/7, 2/7, 2/7, 1/7$, the add-1 row in Figure 1.

Softmax cancels a shift, so this count model needs no bias. Subtract the row maximum before the exponential anyway, so a row of large logs does not overflow:

```python
shifted = logits - logits.max(axis=1, keepdims=True)
probs = np.exp(shifted)
probs = probs / probs.sum(axis=1, keepdims=True)
```

> [!KEY]
> A one-layer softmax on a one-hot input is a count table stored as logarithms. Set each row to $\ln(\text{count} + k)$ and the probabilities match add-k smoothing.

`dog` and `hound` are still orthogonal, and the table still holds $V^2$ numbers. A short vector per word is the way out of both problems.

## A vector instead of a column

An *embedding table* $E$ has shape $(V, D)$. Row $i$ is the vector for word $i$. $D$ is a width you choose, 2 in the hand example and 8 in the training run. A lookup copies the row, which is the one-hot product with a narrower matrix:

```python
rows = table[indices]
```

Look up indices $(1, 0, 1)$ in a three-word table of width 2:

$$
E = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 3 & 1 \end{bmatrix}
\qquad
\text{lookup} = \begin{bmatrix} 0 & 2 \\ 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

Index 1 appears twice, so row $(0, 2)$ comes out twice. Nothing is summed yet.

## The gradient adds when a word repeats

Suppose the loss pulls on those three outputs with this gradient:

$$
\frac{\partial L}{\partial y} = \begin{bmatrix} 1 & 0 \\ 4 & 1 \\ 0 & 3 \end{bmatrix}
$$

Index 0 appears once, so row 0 receives $(4, 1)$. Index 1 appears twice, so row 1 receives $(1, 0) + (0, 3) = (1, 3)$. Index 2 is unused, so it stays $(0, 0)$:

$$
\frac{\partial L}{\partial E_{i}} = \sum_{t:\, i_t = i} \frac{\partial L}{\partial y_t}
$$

That sum is a scatter-add. `np.add.at` adds each output gradient into the row it came from, including when the index repeats.

> [!WARNING]
> An assignment keeps only the last visit. The second visit to index 1 replaces $(1, 0)$ with $(0, 3)$, and the first pull disappears. A finite difference of the lookup matches the sum $(1, 3)$, not the leftover row $(0, 3)$.

> [!KEY]
> Embedding backward is a scatter-add into the rows you looked up. A repeated index adds. It does not overwrite.

## Predict from a window

A bigram sees one previous word, so it can learn that `dog` is followed by `runs`. It cannot see both words of "big dog" at once.

A *context window* of width 2 looks up the two previous vectors and concatenates them. One hidden layer with a tanh bend, then a softmax, turns that concatenation into a next-word distribution. The input is a window of embeddings rather than a one-hot.

The training notebook uses all four lines and nine words:

| Index | Word |
|---|---|
| 0 | start |
| 1 | big |
| 2 | small |
| 3 | dog |
| 4 | hound |
| 5 | cat |
| 6 | kitten |
| 7 | runs |
| 8 | sits |

A window of 2 gives two rows per line. `(start, big)` is followed by `dog` or `hound`. `(big, dog)` is followed by `runs` every time. Half the windows are a coin flip between two animals, and half are a verb that does not vary. On a balanced copy, the best mean negative log likelihood is $(\ln 2)/2 = 0.3466$: cost 0 on the verbs, cost $\ln 2$ on the coin flips. An untrained softmax over 9 words sits near $\ln 9 = 2.197$, perplexity 9.

One forward pass, with $N$ windows, has these shapes:

```python
looked = table[context_ids]          # (N, window, D)
flat = looked.reshape(N, -1)         # (N, window * D)
hidden = np.tanh(flat @ weight_1 + bias_1)
logits = hidden @ weight_2 + bias_2  # (N, V)
```

The loss is the same mean $-\ln p$. The gradient on the logits is $(\text{probabilities} - \text{one hot}) / N$, and the embedding piece scatter-adds into the rows the window read. The slope of tanh is $1 - \tanh^2$.

Repeat each line 4 times. Use $D = 8$, hidden width 16, learning rate $0.5$, and 80 full-batch steps from `default_rng(0)`. Draw the embedding, then the first weight, then the second, from a normal of scale $0.1$, and start the biases at 0. The loss falls to $0.367$. Given `(big, dog)`, the probability of `runs` is $0.990$ and the probability of `sits` is $0.008$.

## Neighbours from shared company

The window network predicts a word. A *skip-gram* asks the vector of a center word to score the words beside it. In "big dog runs", with one word of window on each side, `dog` has context words `big` and `runs`. `hound` has the same two. The two animals share a company, and they occupy the same line.

A full softmax in the denominator sums all $V$ words. *Negative sampling* replaces that sum with one real context word and $K$ noise words:

$$
L = -\ln \sigma(c \cdot p) - \sum_{k=1}^{K} \ln \sigma(- c \cdot n_k)
$$

$c$ is the center vector, $p$ is the context vector of the real neighbour, and $n_k$ is the context vector of a noise word. $\sigma(s) = 1/(1+e^{-s})$. The center word reads one table. The context word and the noise words read a second table. A large dot product means the two vectors point the same way.

Draw the noise from the unigram $U(w)$ raised to the $3/4$ power and divided by the sum, so a very frequent word does not fill every noise slot.

The gradient on the center vector pushes $c$ toward $p$ and away from each $n_k$:

$$
\frac{\partial L}{\partial c} = \big(\sigma(c \cdot p) - 1\big)\, p + \sum_{k=1}^{K} \sigma(c \cdot n_k)\, n_k
$$

The gradient on $p$ is $(\sigma(c \cdot p) - 1)\, c$, and the gradient on $n_k$ is $\sigma(c \cdot n_k)\, c$. Because $\sigma(c \cdot p) - 1$ is negative, the real pair attracts. The noise coefficients are positive, so those pairs repel.

Try $c = (1, 0)$, $p = (1, 0)$, and one noise vector $n = (0, 1)$. Then $c \cdot p = 1$, $c \cdot n = 0$, $\sigma(1) = 0.7311$, and $\sigma(0) = 0.5$. The loss pieces $-\ln 0.7311 = 0.3133$ and $-\ln 0.5 = 0.6931$ add to $1.0064$. The center gradient is

$$
(0.7311 - 1)\,(1, 0) + 0.5\,(0, 1) = (-0.2689,\ 0.5)
$$

A central difference on these three vectors agrees. Compute the loss with softplus, $-\ln \sigma(s) = \mathrm{softplus}(-s)$ and $-\ln \sigma(-s) = \mathrm{softplus}(s)$, so a large score stays finite.

Train on the repeated notebook. Draw the center table from `default_rng(seed)` as a normal of scale $0.1$, then the context table. Each pass shuffles the in-line pairs, draws $K$ noise words from the $3/4$ unigram, and steps the rows that the pair touched.

Figure 2 uses dimension 8, learning rate $0.4$, $K = 4$, 20 passes, and seed 0. The cosine of `dog` with `hound` is $0.894$. The cosine with `sits` is $0.342$, and the cosine with `runs` is $0.312$, even though `runs` follows `dog` on the line. The step moves the center vector of `dog` toward the *context* vector of `runs`, not toward the center vector of `runs`. Words that share a context get similar center vectors, so `hound` is the near neighbour. The check asks the cosine with `hound` to beat the cosine with `sits` by more than $0.25$. This run leads by about $0.55$.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Four bars of cosine with dog after a seeded skip-gram. Hound reaches 0.89. Cat, runs, and sits sit near 0.31 to 0.34.">
<line x1="96" y1="40" x2="560" y2="40" class="stroke-grid"/>
<line x1="96" y1="144" x2="560" y2="144" class="stroke-grid"/>
<line x1="96" y1="248" x2="560" y2="248" class="stroke-muted"/>
<text x="88" y="44" text-anchor="end" class="fig-small">1</text>
<text x="88" y="148" text-anchor="end" class="fig-small">0.5</text>
<text x="88" y="252" text-anchor="end" class="fig-small">0</text>
<rect x="110" y="62.1" width="80" height="185.9" class="fill-b"/>
<rect x="226" y="183.6" width="80" height="64.4" class="fill-a"/>
<rect x="342" y="183.2" width="80" height="64.8" class="fill-a"/>
<rect x="458" y="176.9" width="80" height="71.1" class="fill-a"/>
<text x="150" y="56" text-anchor="middle" class="fig-small">0.89</text>
<text x="266" y="176" text-anchor="middle" class="fig-small">0.31</text>
<text x="382" y="176" text-anchor="middle" class="fig-small">0.31</text>
<text x="498" y="170" text-anchor="middle" class="fig-small">0.34</text>
<text x="150" y="270" text-anchor="middle" class="fig-small">hound</text>
<text x="266" y="270" text-anchor="middle" class="fig-small">cat</text>
<text x="382" y="270" text-anchor="middle" class="fig-small">runs</text>
<text x="498" y="270" text-anchor="middle" class="fig-small">sits</text>
<text x="320" y="294" text-anchor="middle" class="fig-small">cosine with dog, center table, seed 0</text>
</svg>
Figure 2. After 20 seeded passes, hound is nearer to dog than sits, cat, or runs. Runs follows dog on the line. Hound shares dog's company, which is the direction the center vector moves.
```

## Cost and scaling

A count table stores $V^2$ numbers, 2.5 billion cells at 50,000 words. An embedding stores $VD$ numbers and a lookup touches $D$ of them. The window network's last matrix is $(H, V)$, so its softmax still touches every word. Negative sampling touches $K+1$ rows instead, which is why it replaces that softmax for a large $V$. The gradient is still a scatter-add into those rows.

## Where this breaks

Add-k spends probability on impossible words, and it shares nothing between `dog` and `hound`. Vectors share that strength, and on a tiny notebook they can also tie two words together only because both sat next to a frequent word. A fixed window cannot see a word outside it. You have already built a recurrent cell and a [transformer block](/lessons/063-the-transformer-block) for longer context. This chapter does not rebuild either one. Negative sampling ranks a real context word above noise. It does not produce softmax probabilities, so read a neighbour with a cosine and do not read the skip-gram loss as a perplexity.

> [!RECAP]
> - $C_{ij}$ counts how often $j$ follows $i$. Add-k uses $(C_{ij}+k)/(n_i+kV)$, so an empty row is uniform and an unseen word is not probability 0.
> - The mean $-\ln p$ is the negative log likelihood. Perplexity is $e^{L}$, and a uniform model scores $V$.
> - $\mathrm{softmax}(\ln(C_{i}+k))$ is the add-k row. A one-hot times that matrix picks the row.
> - An embedding lookup copies rows. The backward pass scatter-adds, and a repeated index sums.
> - A window network predicts the next word. Skip-gram with negative sampling pulls a center vector toward the context vectors of the words it keeps company with.

## Build it step by step

The checks use the five-word tally first, then the nine-word notebook. Seed every generator with `np.random.default_rng(seed)`.

1. **`bigram_counts(prev, nxt, vocab_size)`** takes two integer arrays of equal length and returns a float array of shape `(vocab_size, vocab_size)`. Add 1 at each pair. The mistake is an assignment that leaves a repeated pair at 1.

2. **`add_k_probs(counts, k)`** returns a `(V, V)` probability array. Add $k$ to every cell, then divide by the new row sum. The mistake is adding $k$ only on positive cells, which leaves an unseen word at 0.

3. **`mean_nll(probs, prev, nxt)`** returns the mean of $-\ln p$ over the pairs, as a Python float. Index with `probs[prev, nxt]`. The mistake is a sum, or a log that is not natural.

4. **`perplexity(nll)`** returns $e$ raised to that mean. On a uniform model this equals the vocabulary size. The mistake is returning the loss unchanged.

5. **`row_softmax(logits)`** maps `(N, V)` scores to probabilities. Subtract the row maximum before the exponential. The mistake is `np.exp` on raw scores of 1000, which overflows.

6. **`embedding_forward(table, indices)`** copies rows of a `(V, D)` table at integer indices of shape `(N,)`, returning `(N, D)`. A repeated index repeats the row.

7. **`embedding_backward(grad_out, indices, vocab_size)`** scatter-adds `grad_out` of shape `(N, D)` into a `(vocab_size, D)` gradient. The mistake is assignment, which keeps the last write when an index repeats.

8. **`skipgram_loss_and_grad(center, positive, negatives)`** takes vectors of shape `(D,)`, `(D,)`, and `(K, D)`. Return the loss, the center gradient, the positive gradient, and the noise gradients. Use softplus for the loss and the sigmoid form for the derivatives. The mistake is a sign that pushes the real pair apart.

9. **`train_window_lm(sentences, vocab_size, window, dim, hidden, steps, lr, seed, query_context)`** trains on sentences of shape `(S, T)`. Draw the embedding, then both weights, from `default_rng(seed)` at scale $0.1$, biases at 0. Return the mean negative log likelihood after `steps` full-batch updates, and the `(vocab_size,)` probabilities of `query_context`. Divide the logit gradient by the number of windows.

10. **`train_skipgram(sentences, vocab_size, window, dim, epochs, lr, n_negative, seed)`** returns the center table, `(vocab_size, dim)`, drawn before the context table. Use both directions inside each line, skip offset 0, shuffle each epoch, and draw noise from the unigram to the power $0.75$, skipping the real context index. Copy a row before you update it.

## The exercise

You implement `bigram_counts`, `add_k_probs`, `mean_nll`, `perplexity`, `row_softmax`, `embedding_forward`, `embedding_backward`, `skipgram_loss_and_grad`, `train_window_lm`, and `train_skipgram`. The checks match the tally, the fractions in Figure 1, the mean $1.0662$, the perplexity $2.904$, and a uniform perplexity of 5. Softmax of $\ln(\text{count}+1)$ must match add-k, including a finite row of 1000s. A repeated embedding row must accumulate to $(1, 3)$, not the last write $(0, 3)$, and match a finite difference. The skip-gram loss $1.0064$ and its gradient must match a finite difference. The window model must prefer `runs` to `sits` after `(big, dog)` and land below $\ln 9$. The skip-gram cosine of `dog` with `hound` must beat `sits` by more than $0.25$.

## Further reading

The notes and lectures this chapter follows:

- For the bigram tally and the negative log likelihood, see [Lecture 2: The spelled-out intro to language modeling: building makemore](https://github.com/karpathy/nn-zero-to-hero) (Andrej Karpathy). For the windowed perceptron, see [Lecture 3: Building makemore Part 2: MLP](https://github.com/karpathy/nn-zero-to-hero) on that page. Those notebooks are character-level. This chapter uses whole words.
- The [makemore](https://github.com/karpathy/makemore) README points that MLP at Bengio, Ducharme, Vincent, and Jauvin, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) (JMLR, 2003).
- For skip-gram and negative sampling, see [Lecture 2: Word Vectors](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture02-wordvecs.pdf) (Diyi Yang, CS224N). The [course schedule](https://web.stanford.edu/class/cs224n/) lists that session as "Word Vectors". The noise distribution $U(w)^{3/4}/Z$ is equation (4) and section 2.2 of Mikolov, Sutskever, Chen, Corrado, and Dean, [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
- For n-gram counts, add-k, the fixed window, and perplexity, see [Lecture 4: Language Models and Recurrent Neural Networks](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture04-rnnlm.pdf) (Diyi Yang, CS224N). This chapter stops before the recurrent half.
- For next-word prediction as a fixed window, see the [Deep Sequence Modeling](https://introtodeeplearning.com/2019/materials/2019_6S191_L2.pdf) slides (Ava Soleimany, 6.S191, 2019). The [2019 schedule](https://introtodeeplearning.com/2019/) lists them as part 2 of session 1, then moves on to recurrent networks.
