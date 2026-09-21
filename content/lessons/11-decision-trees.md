---
title: Decision trees
slug: 11-decision-trees
part: "Part 2 — Classical machine learning"
summary: Axis-aligned recursive partitioning, the impurity measures that drive it, and the pruning that keeps a single tree honest.
minutes: 35
gpu: false
objectives:
  - State the constant that minimises squared error and the constant that minimises log loss on a leaf.
  - State why exact optimal tree learning is NP-hard and what greedy recursive splitting gives up in exchange.
  - Implement the split search in O(n log n) per feature using sorted order and running sums.
  - State the weakest-link argument behind cost-complexity pruning and see why the pruned sequence is nested.
prereqs:
  - 01-the-learning-problem
exercise: 11-decision-trees
---

# Decision trees

Every hypothesis class you have fit so far commits to a global form before it sees the data: a hyperplane, a sigmoid of a hyperplane, a fixed set of basis functions. A decision tree commits to nothing global. It partitions the feature space into axis-aligned boxes and fits an independent constant inside each one, so its flexibility is exactly the flexibility of the partition you choose. That buys interaction effects and nonlinearity for free, with no feature engineering and no kernel, at the price of a combinatorial search: choosing the best partition of $n$ points into boxes is not a convex problem, and it is not a tractable one either.

This lesson builds the tree practitioners actually fit: a greedy, recursive approximation to the optimal partition, the impurity measures that decide each split, and the pruning procedure that stops the recursion from memorising the training set. The tree this procedure produces is unstable on its own: a small change to the data can flip an early split and rearrange everything beneath it. That instability is not a flaw to patch here. It is the property the next lesson exploits by averaging many trees together.

## What you already know

- Empirical risk minimisation: choose the hypothesis that minimises average loss over the training sample.
- The maximum likelihood estimate of a categorical distribution's parameters is the vector of observed class proportions.
- A held-out estimate of risk is the only honest way to compare models of different complexity.
- $NP$-hardness: a problem is $NP$-hard when no known algorithm solves every instance in polynomial time, and none is expected to exist.

## The hypothesis class

A tree of depth $D$ recursively splits $\mathbb R^d$ along coordinate axes. Each internal node holds a feature index $j$ and a threshold $\tau$, and sends a point $x$ left when $x_j \le \tau$ and right otherwise. Each leaf $\ell$ owns a box $R_\ell$, the intersection of the constraints along its root-to-leaf path, and predicts one constant $c_\ell$ for every point that lands there:

$$
f(x) = \sum_{\ell} c_\ell\, \mathbb 1[x \in R_\ell] .
$$

Fixing the partition decouples the leaves: each $c_\ell$ minimises loss only over the points inside $R_\ell$. Fitting a tree is two problems, not one. You choose a partition, then solve one small estimation problem per leaf.

### The leaf constant

For squared loss, $c_\ell$ minimises $\sum_{i\in\ell}(y_i-c)^2$; differentiating gives the within-leaf mean, $c_\ell = \frac1{n_\ell}\sum_{i\in\ell} y_i$. For a $K$-class leaf under log loss, the same style of argument, a Lagrange multiplier enforcing $\sum_k q_k=1$, gives $q_k = n_k/n_\ell$, the empirical class proportion. Both are the plug-in maximum-likelihood estimate of the label under the leaf's local distribution: Gaussian for squared loss, categorical for log loss. On a binary leaf, $q_1$ and the mean of $y$ coincide, so one leaf value serves regression and classification alike.

### Why the optimal partition is out of reach

Choosing the loss-minimising partition, subject to a leaf budget, means choosing which of exponentially many recursive splits to make at every node, each choice reshaping which choices are worth making beneath it. Deciding whether a training set admits a zero-error tree with at most $k$ leaves is $NP$-hard: it instantiates the general problem of finding a minimal boolean decision diagram consistent with a truth table.

What survives is a greedy relaxation: at each node, search every feature and threshold for the single split that most reduces an impurity criterion, apply it, and recurse into the children independently. This throws away any split that only pays off two levels down, which is exactly what makes the search tractable.

## Impurity measures

An impurity function $I(S)$ scores a set of labels: zero when pure, larger when mixed.

**Squared-error impurity** is the leaf's residual variance, $I_{\text{mse}}(S)=\frac1{|S|}\sum_{i\in S}(y_i-\bar y_S)^2$.

**Gini impurity**, for class proportions $p_k$, is $I_{\text{gini}}(S)=1-\sum_k p_k^2$. Read the second form, $\sum_k p_k(1-p_k)$, as an expected error rate: draw two labels independently from $S$'s own distribution, and that sum is the chance they disagree.

**Entropy** is $I_{\text{ent}}(S)=-\sum_k p_k\log p_k$. It equals the average log loss the leaf's own optimal constant achieves on its own points, not an ad hoc measure of disorder but the realised training loss under the criterion the tree is ultimately scored on.

For two classes with $p=p_1$, both reduce to concave functions of $p$ that vanish at $p\in\{0,1\}$: $I_{\text{gini}}=2p(1-p)$ and $I_{\text{ent}}=-p\log p-(1-p)\log(1-p)$. They share the same maximum at $p=\tfrac12$ and the same curvature there, so a split that reduces one substantially reduces the other by nearly the same proportion. The two criteria rarely disagree on which split to take; disagreements are confined to near-ties.

## The split-search algorithm

Evaluating every threshold directly costs $O(n)$ work per threshold, and there are $O(n)$ thresholds, so $O(n^2)$ per feature. Sorting removes the redundancy: sort feature $j$'s values once, in $O(n\log n)$, then sweep the threshold left to right, moving one point at a time from the right side to the left.

For squared error, maintain running sums $S_L=\sum_{i\in L} y_i$ and $Q_L=\sum_{i\in L}y_i^2$; the right side's sums follow by subtraction from the totals. The left variance is $Q_L/n_L-(S_L/n_L)^2$, updated in $O(1)$ per step. For Gini or entropy on a binary label, maintaining the running count of positive labels in the left partition is enough, since each impurity is a closed-form function of that count and $n_L$. One feature costs $O(n\log n)$ for the sort plus $O(n)$ for the sweep, so a node with $d$ features costs $O(nd\log n)$.

The criterion that picks a split is the reduction

$$
\Delta I = I(S) - \frac{n_L}{n}I(S_L) - \frac{n_R}{n}I(S_R) ,
$$

the parent's impurity minus the sample-weighted average of the children's. That weighting is what keeps $\Delta I \ge 0$ from being trivially achievable by putting every point on one side, so the search only accepts splits that genuinely help.

## Cost-complexity pruning

A tree grown until every leaf is pure drives training risk to zero and generalises poorly. Limiting depth in advance is one fix; pruning a fully grown tree back is a better one, because it judges which splits earned their keep only after seeing the whole tree.

Define the cost complexity of tree $T$ as

$$
R_\alpha(T) = R(T) + \alpha\,|\text{leaves}(T)|, \qquad R(T) = \sum_{\ell} n_\ell\, I(\ell) ,
$$

impurity across leaves plus a penalty $\alpha$ per leaf. $R(T)$ only falls as splits are added, so $\alpha$ is the exchange rate that makes growing costly enough to stop. Because $R_\alpha$ decomposes additively over leaves, comparing "collapse to a leaf" against "keep the subtree" at every node, bottom-up, produces the exact minimiser of $R_\alpha$ over every subtree of the fully grown tree.

The classical **weakest-link** algorithm computes the whole path at once. For an internal node $t$ with subtree $T_t$, define

$$
g(t) = \frac{R(t) - R(T_t)}{|\text{leaves}(T_t)| - 1} ,
$$

the per-leaf cost of keeping $t$'s subtree instead of collapsing it. Repeatedly collapsing the node with the smallest $g(t)$ produces a nested sequence $T_0 \supset T_1 \supset \cdots \supset \{\text{root}\}$, with nondecreasing collapsed $g$ values. Each $T_k$ is the optimal pruned tree for every $\alpha$ in $[g(t_k), g(t_{k+1}))$, so one sweep produces the optimal tree at every complexity level, rather than one bottom-up pass per candidate $\alpha$.

### Surrogate splits and missing values

CART handles a missing feature value at prediction time without imputing it. Alongside the primary split, it records surrogate splits: among the remaining features, the threshold whose partition agrees most often with the primary one. A row missing the primary feature is routed by the best surrogate it has, and to the majority branch if every surrogate is also missing. This degrades gracefully only when some other feature correlates with the primary one.

## Cost and scaling

| Operation | Cost | Notes |
|---|---|---|
| Best split at one node, $n$ rows, $d$ features | $O(nd \log n)$ | Sort per feature, one linear sweep with running sums. |
| Full tree, sorted order carried through recursion | $O(nd \log n + nd \cdot \text{depth})$ | Sort each feature once. |
| Full tree, re-sorting at every node | $O(n^2 d \log n)$ worst case | Depth can reach $n$ on adversarial data. |
| Prediction, one query | $O(\text{depth})$ | A root-to-leaf walk. |
| Cost-complexity pruning, whole path | $O(|T_0|)$ | One weakest-link sweep. |

The gap between the second and third rows is why production implementations pass sorted index arrays down the recursion instead of re-sorting at every node: that turns a tree that is linear in $n$, up to logarithms, into one that is quadratic.

## Where this breaks

**Axis-aligned bias.** A single split cuts perpendicular to one coordinate axis. A diagonal decision boundary needs a staircase of splits to approximate, each adding variance without adding much accuracy.

**Instability.** Because the split search is greedy and each split is a hard decision, a handful of points near a threshold can flip which feature wins at the root, and everything beneath it follows that choice. Two similar training sets can produce structurally different trees. The next lesson addresses this directly, by averaging many trees rather than stabilising any one of them.

**Myopia.** A greedy split search cannot see that two mediocre splits in sequence might beat the best single split available at the current step. Features that are only informative in combination, an XOR-like interaction, can look useless to every one-step split, regardless of depth budget.

**High-cardinality categorical features.** A feature with many distinct values offers more candidate thresholds, and so more chances for one to reduce impurity by chance on the training set, biasing the search toward it even without extra signal.

## The exercise

You build CART from its two smallest pieces up to a pruned tree. Implement `gini` and `entropy`, each taking an integer label array and returning its impurity, then `best_split(X, y, criterion)`, which returns the feature, threshold, and impurity reduction of the best axis-aligned split using sorted order and running sums rather than recomputing from scratch at every threshold.

Then implement `grow_tree`, which recurses `best_split` into a full tree subject to a maximum depth and a minimum leaf size, and `predict_tree`, which walks a fitted tree to a leaf for each query row. Finally implement `cost_complexity_prune(tree, alpha)`, the bottom-up collapse-or-keep pass derived above.

The checks confirm `gini` and `entropy` against their formulas at a pure node, a uniform node, and a mixed node; that `best_split` finds the same split, to numerical precision, as an $O(n^2)$ brute-force search over every threshold; that a tree grown without a depth limit drives training error to zero; that every leaf's prediction equals the mean of the training labels routed there; and that the number of leaves after `cost_complexity_prune` is nonincreasing as `alpha` grows, confirming the nested-sequence property the weakest-link argument predicts.
