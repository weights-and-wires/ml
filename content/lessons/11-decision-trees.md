---
title: Decision trees
slug: 11-decision-trees
part: "Part 2 — Classical machine learning"
summary: Build a decision tree from scratch, from impurity scores and fast split search to growing, predicting, and pruning.
minutes: 60
gpu: false
objectives:
  - Compute Gini impurity, entropy, and squared-error impurity for a set of labels.
  - Find the best split on a feature in one sorted pass using running sums.
  - Build a tree by recursive splitting, and predict by walking each row to a leaf.
  - Prune a grown tree with a per-leaf penalty, and explain why a bigger penalty gives a smaller tree.
prereqs:
  - 01-the-learning-problem
exercise: 11-decision-trees
---

# Decision trees

A decision tree is a game of twenty questions. To guess a house price, it asks "Is the floor area at most 80 square metres?" and then, depending on the answer, "Is it within 2 km of the centre?" Each answer sends you down a branch. After a few questions you reach a leaf, and the leaf holds the prediction.

Every model so far picked one shape in advance, such as a straight line or a sigmoid of a line. A tree does not. It chops the data into boxes and predicts one number inside each box. That lets it capture curves and interactions between features without any feature engineering.

The catch is that finding the best set of boxes is too hard to solve exactly. So in practice you build the tree greedily, one question at a time, and then cut it back so it does not memorise the training data. This lesson builds that whole pipeline.

## What you already know

- Training a model means choosing the one with the smallest average loss on the training data.
- The mean of a set of numbers is the constant that minimises squared error to those numbers.
- The fraction of labels in each class is the natural estimate of that class's probability.
- A model that scores perfectly on training data can still do badly on new data.

## How a tree predicts

A tree is made of nodes. An internal node holds two things: a feature index $j$ and a threshold $t$. A row $x$ goes left when $x_j \le t$ and right otherwise. A leaf holds one number, its value, and that value is the prediction for every row that lands there.

What number should a leaf predict? Use the mean of the training labels that reached it. For squared error, the mean is the best possible constant. For a 0/1 classification label, the mean of $y$ is the fraction of class-1 rows, which is also the best probability estimate. So one rule, "leaf value is the mean", serves both regression and classification.

> [!KEY]
> A leaf predicts the mean of the training labels that reach it. For a 0/1 label, that mean is the fraction of class-1 rows.

In numpy, a leaf's value is one line:

```python
value = float(y_leaf.mean())
```

To predict, you start at the root and follow the questions until you hit a leaf:

```python
node = root
while not node.is_leaf:
    node = node.left if x[node.feature] <= node.threshold else node.right
```

Take a tiny tree: the root asks "is $x_0 \le 0.5$?", the left leaf predicts $-1$, and the right leaf predicts $1$. The rows $0.0$, $0.5$, and $-3.0$ all go left and get $-1$. The rows $1.0$ and $10.0$ go right and get $1$. The row $0.5$ goes left because the test is "less than or equal".

## Impurity: how mixed is a node?

To pick a good question, you need a score for how mixed a group of labels is. That score is called **impurity**. It is zero when every label in the group is the same, and it grows as the group gets more mixed. A good split turns one mixed group into two purer groups.

### Gini impurity

Let $p_k$ be the fraction of labels in the group that equal class $k$. Gini impurity is

$$
G = 1 - \sum_k p_k^2 .
$$

Read it aloud: square each class fraction, add the squares, and subtract the total from one. If one class has every label, its fraction is 1, the sum of squares is 1, and $G = 0$. Another way to see it: $G$ is the chance that two labels drawn at random from the group, with replacement, disagree.

```python
p = np.bincount(y) / y.size
g = 1.0 - np.sum(p ** 2)
```

> [!WARNING]
> `np.bincount` accepts only non-negative integers. If your labels are stored as floats, convert them with `y.astype(int)` first, or numpy raises a `TypeError`.

### Entropy

Entropy is a second way to score mixing:

$$
H = -\sum_k p_k \log p_k .
$$

Read it aloud: for each class, multiply its fraction by the natural log of that fraction, add them up, and flip the sign. The log is the natural log, so the unit is nats. A class with $p_k = 0$ contributes nothing, by the convention $0 \log 0 = 0$. Entropy equals the average log loss the leaf pays when it predicts its own class fractions for its own rows.

```python
p = np.bincount(y) / y.size
p = p[p > 0]            # drop empty classes so log(0) never runs
h = -np.sum(p * np.log(p))
```

For a binary label with $p$ the fraction of class 1, Gini becomes $2p(1-p)$. Both Gini and entropy are zero at $p = 0$ and $p = 1$ and largest at $p = 0.5$. They almost always pick the same split.

### Squared-error impurity

For a regression target, impurity is the variance of the labels in the group:

$$
V = \frac{1}{m}\sum_{i=1}^{m} (y_i - \bar y)^2 ,
$$

where $m$ is the number of rows in the group and $\bar y$ is their mean. Read it aloud: take each label's distance from the group mean, square it, and average. This is the mean squared error the leaf makes when it predicts $\bar y$.

```python
v = np.mean((y - y.mean()) ** 2)
```

### A worked example

Take the six labels $[0, 0, 1, 1, 1, 2]$. The class fractions are $p_0 = 2/6 = 1/3$, $p_1 = 3/6 = 1/2$, and $p_2 = 1/6$.

- Gini: the squares are $1/9$, $1/4$, and $1/36$, which sum to $14/36$. So $G = 1 - 14/36 = 22/36 \approx 0.6111$.
- Entropy: $-(\tfrac13 \log \tfrac13 + \tfrac12 \log \tfrac12 + \tfrac16 \log \tfrac16) = 0.3662 + 0.3466 + 0.2986 \approx 1.0114$.

Two more values to check your code against: four labels $[0, 1, 2, 3]$ give Gini $0.75$ and entropy $\log 4 \approx 1.3863$. Four copies of the same label give zero for both.

## Choosing the best split

A split is a feature and a threshold. The best split is the one that makes the children purer than the parent by the largest amount. That amount is called the **gain**:

$$
\text{gain} = I_{\text{parent}} - \frac{n_L}{n} I_L - \frac{n_R}{n} I_R .
$$

Read it aloud: start with the parent's impurity. Subtract the left child's impurity weighted by the share of rows that went left, $n_L / n$. Subtract the right child's impurity weighted by its share, $n_R / n$. The weights matter. Without them, a split that peels off one lucky row would look as good as a split that cleanly halves the data.

> [!KEY]
> The best split has the largest gain: the parent's impurity minus the children's impurities, each weighted by its share of the rows.

```python
gain = parent_imp - (n_left * left_imp + n_right * right_imp) / n
```

### Which thresholds to try

Sort one feature's values. Any threshold between the same two neighbouring values gives the same split, so you only need one candidate per gap: the midpoint between two consecutive, different sorted values. If two neighbours are equal, there is no gap and no split there.

### Running sums make it fast

The slow way recomputes both children's impurity from scratch at every threshold. That costs about $n$ steps per threshold, and there are about $n$ thresholds, so about $n^2$ steps per feature.

The fast way sorts once and slides the threshold from left to right. Each step moves exactly one row from the right child to the left child. If you keep running totals, each step costs a constant amount of work.

For squared error, keep the running sum of labels $S_L$ and the running sum of squared labels $Q_L$ on the left. The left variance is then

$$
V_L = \frac{Q_L}{n_L} - \left(\frac{S_L}{n_L}\right)^2 .
$$

Read it aloud: the mean of the squares minus the square of the mean. The right side uses the same formula with $S_R = S_{\text{total}} - S_L$ and $Q_R = Q_{\text{total}} - Q_L$. In numpy, `np.cumsum` gives every running sum at once:

```python
cum_sum = np.cumsum(ys)        # ys is y reordered by the sorted feature
cum_sq = np.cumsum(ys ** 2)
left_sum, left_sq = cum_sum[:-1], cum_sq[:-1]
```

> [!WARNING]
> Entry `i` of `cum_sum[:-1]` covers `i + 1` rows, so the left counts are `np.arange(1, n)`, not `np.arange(n - 1)`. An off-by-one here shifts every variance.

For a 0/1 label under Gini or entropy, the running count of 1s on the left is enough. Divide it by $n_L$ to get $p_L$, and apply the binary formula.

> [!KEY]
> Sort a feature once, and running sums give you the impurity of every candidate split in a single pass.

### A worked split

Take one feature $x = [1, 2, 3, 4]$ with labels $y = [1, 3, 8, 10]$. The parent mean is $5.5$, and the parent variance is $(20.25 + 6.25 + 6.25 + 20.25)/4 = 13.25$.

The running sums are `cumsum(y) = [1, 4, 12, 22]` and `cumsum(y**2) = [1, 10, 74, 174]`. The following table shows every candidate threshold.

| Threshold | $n_L$ | $S_L$ | $Q_L$ | $V_L$ | $V_R$ | Gain |
|---|---|---|---|---|---|---|
| 1.5 | 1 | 1 | 1 | 0 | 8.667 | 6.75 |
| 2.5 | 2 | 4 | 10 | 1 | 1 | 12.25 |
| 3.5 | 3 | 12 | 74 | 8.667 | 0 | 6.75 |

At threshold 2.5, the left variance is $10/2 - (4/2)^2 = 5 - 4 = 1$. The right side has $S_R = 18$ and $Q_R = 164$, so $V_R = 164/2 - 9^2 = 1$. The gain is $13.25 - (2 \cdot 1 + 2 \cdot 1)/4 = 12.25$. That is the best split.

> [!TIP]
> Test your fast sweep against a slow loop that splits at each midpoint and calls your impurity function on both sides. On a few random rows, the two gains must agree up to rounding.

## Growing the tree

Growing is recursion. At a node, find the best split, send each row left or right, and repeat on each child. Stop and make a leaf when any of these holds:

- The node is already pure (impurity close to zero).
- The node is at `max_depth`.
- No valid split exists, because there are too few rows or every feature is constant.
- The best split has no positive gain.

This search is **greedy**: it takes the best question right now and never looks ahead. Finding the truly best tree is known to be computationally intractable for large data, so greedy search is the practical choice. The price is that it can miss a pair of splits that only help together.

A tree with no depth limit keeps splitting until every leaf is pure. On data with distinct inputs, that means zero training error, which usually means it has memorised noise.

## Pruning the tree back

Pruning grows a big tree first and then removes the splits that did not earn their keep. It gives each leaf a price $\alpha$ and asks whether the drop in error is worth the extra leaves.

The **cost** of a tree is

$$
R_\alpha(T) = \sum_{\text{leaves}} n_\ell \, I_\ell \;+\; \alpha \times (\text{number of leaves}) .
$$

Read it aloud: for each leaf, multiply its row count $n_\ell$ by its impurity $I_\ell$, and add them up. That is the total training error. Then add $\alpha$ for every leaf. With $\alpha = 0$, pruning keeps every useful split. With a huge $\alpha$, one leaf is cheapest, so the tree collapses to its root.

> [!KEY]
> Each leaf costs $\alpha$. A split survives pruning only when it lowers the total error $\sum n_\ell I_\ell$ by more than $\alpha$ for each leaf it adds.

You find the cheapest tree bottom-up. At each internal node, first prune both children. Then compare two costs:

- Collapse to a leaf: `n_samples * impurity + alpha`.
- Keep the split: the left child's cost plus the right child's cost.

Keep whichever is cheaper. Because each subtree's cost only grows faster than a single leaf's cost as $\alpha$ rises, a larger $\alpha$ never gives you more leaves. The pruned trees form a nested sequence: each is a cut-down version of the one before.

Using the worked split: the parent has 4 rows and impurity 13.25, so collapsing costs $53 + \alpha$. The two children each have 2 rows and impurity 1, so keeping costs $2 + 2 + 2\alpha = 4 + 2\alpha$. Keeping wins while $\alpha < 49$.

> [!RECAP]
> - A row goes left when $x_j \le t$, and a leaf predicts the mean of its training labels.
> - Gini is $1 - \sum_k p_k^2$, entropy is $-\sum_k p_k \log p_k$ over the nonzero $p_k$, and regression impurity is the variance.
> - Gain is the parent's impurity minus the children's impurities, weighted by $n_L / n$ and $n_R / n$.
> - Candidate thresholds are midpoints between consecutive, different sorted values, and running sums score them all at once.
> - Pruning works bottom-up: a node collapses when `n_samples * impurity + alpha` is no more than the total cost of its pruned children.

## Build it step by step

The checks test the functions in this order, so build them in this order too.

1. **`gini(y)`.** Takes an integer label array of shape `(n,)` and returns a float. Use `np.bincount`, divide by the total, then `1 - np.sum(p**2)`. Common mistake: dividing by the number of classes instead of the number of labels.
2. **`entropy(y)`.** Same input and output. Use `np.bincount`, keep only the nonzero counts, then `-np.sum(p * np.log(p))`. Common mistake: taking `np.log` of a zero fraction, which returns `nan`.
3. **`predict_tree(tree, X)`.** Takes a root `Node` and `X` of shape `(n, d)`, and returns an array of shape `(n,)`. Loop over rows, and walk each from the root with the `while not node.is_leaf` loop. Common mistake: using `<` instead of `<=`, which sends rows that sit exactly on the threshold the wrong way.
4. **`best_split(X, y, criterion, min_samples_leaf)`.** Returns `(feature, threshold, gain)` or `None`. For each feature, sort with `np.argsort`, reorder `y`, build running sums with `np.cumsum`, and compute the gain for all candidate positions at once. Mark a position invalid when neighbouring sorted values are equal or a side has fewer than `min_samples_leaf` rows, and use `np.argmax` over the valid gains. The threshold is the midpoint `0.5 * (xs[i] + xs[i + 1])`. Common mistake: returning a threshold equal to a data value, or forgetting the `n_L / n` weights.
5. **`grow_tree(X, y, criterion, max_depth, min_samples_leaf)`.** Returns the root `Node`. Write an inner function `build(X, y, depth)` that makes a leaf node with `n_samples`, `impurity`, and `value = y.mean()`, checks the stopping rules, and otherwise splits with a boolean mask `X[:, feature] <= threshold` and recurses. Common mistake: storing the class label instead of the mean as `value`.
6. **`cost_complexity_prune(tree, alpha)`.** Returns a new tree. Write a recursive helper that returns both the pruned node and its cost. Build new `Node` objects rather than editing the input. Common mistake: comparing against the children's unpruned costs.

## Cost and scaling

The following table shows how the work grows with $n$ rows and $d$ features.

| Task | How the work grows | Why |
|---|---|---|
| Best split at one node | About $n \log n$ per feature, so $d \cdot n \log n$ | Sorting dominates; the sweep is one pass. |
| Best split, slow way | About $n^2$ per feature | Recomputes impurity at every threshold. |
| Growing a whole tree | Split cost times depth, roughly | Each level of the tree touches every row once. |
| Predicting one row | Proportional to depth | One question per level. |
| Pruning | Proportional to the number of nodes | Each node is visited once. |

Memory is small: the data plus one node per split. Real libraries sort each feature once and pass sorted index arrays down the tree instead of re-sorting at every node.

## Where this breaks

- **Diagonal boundaries.** Every split cuts along one feature axis. A diagonal boundary needs a staircase of many small splits. Adding features such as $x_1 + x_2$ helps.
- **Instability.** A few rows near a threshold can change the root split, and the whole tree below it changes too. The next lesson fixes this by averaging many trees.
- **Short-sightedness.** Greedy search cannot see that two weak splits together are strong. An XOR pattern, where neither feature helps alone, can fool it.
- **Features with many values.** A feature with many distinct values offers more thresholds, so it is more likely to win by luck. Limiting depth and pruning reduce the damage.
- **Overfitting.** A fully grown tree memorises noise. Prune it, and choose $\alpha$ on held-out data.

## The exercise

You build a CART-style tree from the smallest pieces up. Implement `gini` and `entropy`, then `predict_tree`, then `best_split` with the sorted, running-sum sweep. Next, implement `grow_tree`, which recurses `best_split` into a full tree, and `cost_complexity_prune`, the bottom-up collapse-or-keep pass.

The checks confirm `gini` and `entropy` at a pure, a uniform, and a mixed node. They walk `predict_tree` through a hand-built tree. They compare `best_split` against a slow brute-force search under squared error and Gini. They confirm that an unlimited tree reaches zero training error and that each leaf predicts the mean of its training labels. Finally, they check that the leaf count after `cost_complexity_prune` never grows as `alpha` grows, stays full at `alpha = 0`, and drops to one leaf for a very large `alpha`.
