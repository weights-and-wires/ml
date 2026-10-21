---
title: Decision trees
slug: 11-decision-trees
part: "Part 2 — Classical machine learning"
summary: Score impurity, search every split, grow a tree, predict from its leaves, and prune with a leaf penalty.
minutes: 60
gpu: false
objectives:
  - Compute Gini impurity, entropy, and squared-error impurity for a set of labels.
  - Find the best split of one feature in one sorted pass, using running sums.
  - Grow a tree by recursive splitting, and predict by walking each row to a leaf.
  - Prune a grown tree with a per-leaf penalty, and explain why a bigger penalty gives a smaller tree.
prereqs:
  - 01-the-learning-problem
exercise: 11-decision-trees
---

# Decision trees

Guess a house price by questions. First, is the floor area at most 80 square metres? The answer sends you left or right. A few questions later you reach a number.

A tree chops the rows into boxes and predicts one number in each box. It can follow a curve without new features.

Finding the best boxes exactly is too hard, so you pick one question at a time, then cut the tree back. You will score a mixed box, search every split, grow, predict, and prune.

## What you already know

These four ideas are the ones this lesson uses:

- Training picks the model with the smallest average loss on the training rows.
- The mean minimises squared error. For 1 and 3 the mean is 2, and the total squared error is 2.
- Two labels of 0 and two of 1 put probability 0.5 on class 1.
- Zero training error on four rows says little about a fifth row.

## Walk a row to a leaf

A root tests whether feature 0 is at most 0.5. The left leaf is $-1$ and the right leaf is $1$. Rows $0$, $0.5$, and $-3$ go left. Rows $1$ and $10$ go right.

An internal node stores a feature $j$ and a threshold $t$. A row goes left when $x_j \le t$. A leaf stores one number, the prediction for every row that lands there.

The leaf number is the mean of its training labels. Labels 1 and 3 have mean 2 and squared error 2. Predicting 0 costs $1+9 = 10$. A 0/1 mean is the class-1 fraction, so one rule covers both:

$$
\text{value} = \frac{1}{m}\sum_{i=1}^{m} y_i .
$$

$m$ is the number of training rows in the leaf. Labels 1 and 3 return 2.

Numpy is `float(y_leaf.mean())`. Walk until the node is a leaf:

```python
node = node.left if x[node.feature] <= node.threshold else node.right
```

The usual mistake is `<` instead of `<=`, which sends a row sitting on the threshold the wrong way.

> [!KEY]
> A leaf predicts the mean of the training labels that reach it. For a 0/1 label, that mean is the class-1 fraction.

## Score how mixed a node is

*Impurity* is 0 when every label matches, and larger when the labels mix.

Six labels $[0, 0, 1, 1, 1, 2]$ have fractions $2/6$, $3/6$, and $1/6$. Squares are $1/9$, $1/4$, and $1/36$, which are $4/36$, $9/36$, and $1/36$. They sum to $14/36$. Then $1 - 14/36 = 22/36 \approx 0.6111$.

That score is *Gini impurity*. With $p_k$ the fraction of class $k$,

$$
G = 1 - \sum_k p_k^2 .
$$

One pure class gives $G = 0$. These six labels return $22/36$. Numpy is `1 - np.sum((np.bincount(y.astype(int)) / y.size) ** 2)`.

The usual mistake is dividing by the number of classes instead of the number of labels.

> [!WARNING]
> `np.bincount` accepts only non-negative integers. Convert float labels with `astype(int)` first.

*Entropy* is a second mix score. The same fractions give $-\bigl(\tfrac13\log\tfrac13 + \tfrac12\log\tfrac12 + \tfrac16\log\tfrac16\bigr) \approx 0.3662 + 0.3466 + 0.2986 = 1.0114$.

$$
H = -\sum_k p_k \log p_k .
$$

The log is natural. Use $0\log 0 = 0$. Labels $[0, 1, 2, 3]$ give Gini $0.75$ and entropy $\log 4 \approx 1.3863$. Four copies of one label give 0. The six labels return $1.0114$. Numpy drops zero counts, then `-np.sum(p * np.log(p))`.

The usual mistake is `np.log` of a zero fraction, which returns `nan`.

For a 0/1 label, Gini is $2p(1-p)$, so $p = 1/2$ gives $1/2$. Entropy is $-p\log p - (1-p)\log(1-p)$. Both peak at $p = 1/2$.

For a numeric target, impurity is the variance. Parent labels $1, 3, 8, 10$ have mean $5.5$. Squared gaps $20.25, 6.25, 6.25, 20.25$ sum to 53, so the variance is $53/4 = 13.25$.

$$
V = \frac{1}{m}\sum_{i=1}^{m} (y_i - \bar y)^2 .
$$

Numpy is `np.mean((y - y.mean()) ** 2)`.

The usual mistake is a sum of squares with no division by $m$. Then 53 is not 13.25.

## Search every split on one feature

A split is one feature and one threshold. *Gain* is how much purer the children are:

$$
\text{gain} = I_{\text{parent}} - \frac{n_L}{n} I_L - \frac{n_R}{n} I_R .
$$

$I$ is impurity and $n_L / n$ is the left share. Without weights, one lucky row looks like a fair cut.

> [!KEY]
> The best split has the largest gain. Weight each child by its share of the rows.

Sort one feature and try the midpoint between two different neighbours. Equal neighbours give no split.

Recomputing impurity at every cut costs about $n^2$ steps per feature. Sort once and move one row at a time. Keep the running sum $S_L$ and the running sum of squares $Q_L$:

$$
V_L = \frac{Q_L}{n_L} - \left(\frac{S_L}{n_L}\right)^2 .
$$

That is the mean of squares minus the square of the mean. The right side uses the leftovers $S - S_L$ and $Q - Q_L$. A 0/1 label only needs a running count of ones.

On $x = [1, 2, 3, 4]$ and $y = [1, 3, 8, 10]$, the running sums are $[1, 4, 12, 22]$. The squared sums are $[1, 10, 74, 174]$. The parent variance is 13.25. Every candidate is:

| Threshold | $n_L$ | Left variance | Right variance | Gain |
|---|---|---|---|---|
| 1.5 | 1 | 0 | $8.667$ | 6.75 |
| 2.5 | 2 | 1 | 1 | 12.25 |
| 3.5 | 3 | $8.667$ | 0 | 6.75 |

At 2.5, the left variance is $10/2 - (4/2)^2 = 1$. The right side has sum 18 and sum of squares 164, so $164/2 - 9^2 = 1$. Gain is $13.25 - (2\cdot 1 + 2\cdot 1)/4 = 12.25$. That is the best cut. The threshold is the midpoint, not a data value. Running sums look like this:

```python
left_sum = np.cumsum(ys)[:-1]          # entry i covers i + 1 rows
left_var = left_sq / n_left - (left_sum / n_left) ** 2
```

The usual mistake is left counts `np.arange(n - 1)` instead of `np.arange(1, n)`.

> [!WARNING]
> Entry `i` of `cum_sum[:-1]` covers `i + 1` rows. An off-by-one shifts every variance.

> [!KEY]
> Sort a feature once. Running sums score every candidate in one pass.

> [!TIP]
> On a few rows, match the fast gains to a slow loop that calls your impurity function on both sides.

## Grow one question at a time

Take the best split and send `X[:, feature] <= threshold` left. Stop at impurity `1e-12`, at `max_depth`, when no split exists, or when the gain is not strictly positive.

The search is *greedy*, so two splits that help only together can be missed. With no depth limit, distinct inputs reach zero training error.

Store `value = y.mean()`, `n_samples`, and `impurity` on the leaf.

## Price each extra leaf

Grow a large tree, then drop splits that did not pay for their leaves. Each leaf costs $\alpha$. The cost of a tree is

$$
R_\alpha(T) = \sum_{\text{leaves}} n_\ell \, I_\ell \;+\; \alpha \times (\text{number of leaves}) .
$$

The sum is the training error, counted as rows times impurity. Then add $\alpha$ per leaf. At $\alpha = 0$ you keep every useful split. A huge $\alpha$ leaves only the root.

Prune both children first. Collapse when `n_samples * impurity + alpha` is no greater than the sum of those children's costs.

On the four-row split, collapsing costs $4 \times 13.25 + \alpha = 53 + \alpha$. Two children of impurity 1 cost $2 + 2 + 2\alpha = 4 + 2\alpha$. Keeping wins while $\alpha < 49$. A larger $\alpha$ never adds leaves, so the pruned trees nest.

> [!KEY]
> Each leaf costs $\alpha$. A split survives only when it cuts $\sum n_\ell I_\ell$ by more than $\alpha$ for each leaf it adds.

Keep these checks in mind:

> [!RECAP]
> - A row goes left when $x_j \le t$, and a leaf predicts the mean of its training labels.
> - Gini is $1 - \sum p_k^2$. Entropy skips zero fractions. Regression impurity is the variance.
> - Gain subtracts the children's impurities, weighted by $n_L / n$ and $n_R / n$.
> - Thresholds are midpoints of different sorted neighbours, scored with running sums.
> - Prune bottom-up. Collapse when `n_samples * impurity + alpha` is no more than the children.

## Build it step by step

Build the functions in the order the checks test them:

1. **`gini(y)`**. Takes integer labels of shape `(n,)` and returns a float. Use `np.bincount`, divide by `y.size`, then `1 - np.sum(p**2)`. The common mistake is dividing by the class count. The fractions have this shape:

```python
p = np.bincount(y.astype(int)) / y.size  # shape (K,)
```

2. **`entropy(y)`**. Same input, one float. Drop zero counts before `np.log`. The common mistake is `log` of 0, which returns `nan`. The kept fractions look like this:

```python
p = counts[counts > 0] / counts.sum()  # drops empty classes
```

3. **`predict_tree(tree, X)`**. Takes a root `Node` and `X` of shape `(n, d)`. Returns shape `(n,)`. Walk with `while not node.is_leaf` and `<=`. The common mistake is `<`. One row updates this way:

```python
node = node.left if row[node.feature] <= node.threshold else node.right
```

4. **`best_split(X, y, criterion, min_samples_leaf)`**. `X` is `(n, d)`. Returns `(feature, threshold, gain)` or `None`. For each feature, `np.argsort`, reorder `y`, and `np.cumsum`. A position is invalid when neighbours are equal or a side has fewer than `min_samples_leaf` rows. The threshold is `0.5 * (xs[i] + xs[i + 1])`. The common mistake is an unweighted impurity, or a threshold equal to a data value. Left counts look like this:

```python
n_left = np.arange(1, n)  # shape (n - 1,)
gain = parent - (n_left * left_imp + n_right * right_imp) / n
```

5. **`grow_tree(X, y, criterion, max_depth, min_samples_leaf)`**. Returns the root `Node`. An inner `build(X, y, depth)` stores `n_samples`, `impurity`, and `value = y.mean()`, then stops on the four rules above. Split with `X[:, feature] <= threshold`. The common mistake is storing a class label instead of the mean. A leaf starts like this:

```python
value = float(y.mean())  # one float, also for 0/1 labels
```

6. **`cost_complexity_prune(tree, alpha)`**. Returns a new tree and does not edit the input. A helper returns the pruned node and its cost. Compare `n_samples * impurity + alpha` with the sum of the already pruned children, and collapse when the leaf is no more expensive. The common mistake is scoring the children before they are pruned. One collapse looks like this:

```python
leaf_cost = node.n_samples * node.impurity + alpha  # one float
```

## Cost and scaling

Work grows with $n$ rows and $d$ features:

| Task | How the work grows | Why |
|---|---|---|
| Best split, one node | about $d \cdot n \log n$ | Sorting dominates the one pass. |
| Best split, slow way | about $n^2$ per feature | Impurity is recomputed at every cut. |
| Growing the tree | split cost times depth | Each level touches every row once. |
| One prediction | proportional to depth | One question per level. |
| Pruning | proportional to the node count | Each node is visited once. |

Memory is the data plus one node per split.

## Where this breaks

**A diagonal boundary.** Cuts follow one axis, so a diagonal edge needs a staircase. A feature $x_1 + x_2$ gives one diagonal question.

**A few rows near a threshold.** They can change the root, and everything under it. Averaging many trees steadies the prediction.

**A short-sighted search.** Greedy search misses two weak splits that are strong together, as in XOR.

**A feature with many distinct values.** It offers more thresholds, so it can win by luck. A depth limit and pruning reduce that.

**A fully grown tree.** It fits noise. Prune it, and choose $\alpha$ on held-out rows.

## The exercise

You implement `gini`, `entropy`, `predict_tree`, `best_split`, `grow_tree`, and `cost_complexity_prune`.

The checks match Gini and entropy at a pure node, a uniform node, and a mixed node. They walk `predict_tree` through a hand-built tree. They match `best_split` to a slow search under squared error and under Gini. An unlimited tree must reach zero training error, and each leaf must predict the mean of its training labels. After `cost_complexity_prune`, the leaf count never rises as `alpha` grows. It stays full at `alpha = 0`. It drops to one leaf for a very large `alpha`.
