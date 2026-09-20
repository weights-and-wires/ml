---
title: Decision trees
slug: 11-decision-trees
part: "Part 2 — Classical machine learning"
summary: Axis-aligned recursive partitioning, the impurity measures that drive it, and the pruning that keeps a single tree honest.
minutes: 50
gpu: false
objectives:
  - Derive the constant that minimises squared error and the constant that minimises log loss on a leaf.
  - State why exact optimal tree learning is NP-hard and what greedy recursive splitting gives up in exchange.
  - Implement the split search in O(n log n) per feature using sorted order and running sums.
  - Derive the weakest-link argument behind cost-complexity pruning and show the pruned sequence is nested.
prereqs:
  - 01-the-learning-problem
exercise: 11-decision-trees
---

# Decision trees

Every hypothesis class you have fit so far commits to a global functional form
before it sees the data: a hyperplane, a sigmoid of a hyperplane, a fixed set of
basis functions. A decision tree commits to nothing global. It partitions the
feature space into axis-aligned boxes and fits an independent constant inside
each one, so the hypothesis class is exactly as flexible as the partition you
choose. That flexibility buys interaction effects and nonlinearity for free —
no feature engineering, no kernel — at the price of a combinatorial search
problem: choosing the best partition of $n$ points into boxes is not a convex
problem, and it is not even a tractable one.

This lesson builds the tree that practitioners actually fit: a greedy,
recursive approximation to the optimal partition, together with the impurity
measures that decide each split and the pruning procedure that stops the
recursion from memorising the training set. The tree you get from this
procedure is an unstable, high-variance estimator on its own — small changes to
the data can flip an early split and rearrange everything beneath it. That
instability is not a flaw to patch here; it is the property the next lesson
exploits by averaging many trees together.

## What you already know

- Empirical risk minimisation: choose the hypothesis that minimises average
  loss over the training sample.
- The maximum likelihood estimate of a categorical distribution's parameters is
  the vector of observed class proportions.
- A held-out estimate of risk is the only honest way to compare models of
  different complexity.
- $NP$-hardness: a problem is $NP$-hard when no known algorithm solves every
  instance in time polynomial in the input size, and none is expected to exist.

## The hypothesis class: piecewise-constant partitions

A decision tree of depth $D$ recursively splits $\mathbb{R}^d$ along
coordinate axes. Each internal node holds a feature index $j$ and a threshold
$\tau$, and routes a point $x$ to its left child when $x_j \leq \tau$ and to
its right child otherwise. Each leaf $\ell$ owns a region
$R_\ell \subset \mathbb{R}^d$ — an axis-aligned box, being the intersection of
the half-space constraints along the root-to-leaf path — and predicts a single
constant $c_\ell$ for every point that lands in $R_\ell$. The tree computes

$$
f(x) = \sum_{\ell} c_\ell \, \mathbb{1}[x \in R_\ell] .
$$

Fitting a tree means choosing the partition $\{R_\ell\}$ and the constants
$\{c_\ell\}$ to minimise empirical risk. Fix the partition, and the constants
decouple: each $c_\ell$ minimises loss only over the points that fall in
$R_\ell$, independent of every other leaf. Fitting a tree is a two-level
problem: pick a partition, then solve $|\{\text{leaves}\}|$ independent
one-dimensional estimation problems.

### The leaf constant

For squared loss, the leaf constant solves
$\min_c \sum_{i \in \ell} (y_i - c)^2$. Differentiating and setting the result
to zero,

$$
-2 \sum_{i \in \ell} (y_i - c) = 0 \quad\Longrightarrow\quad
c_\ell = \frac{1}{n_\ell} \sum_{i \in \ell} y_i ,
$$

the within-leaf mean, and the second derivative $2 n_\ell > 0$ confirms it is a
minimum.

For a $K$-class leaf under log loss, the leaf predicts a distribution
$q \in \Delta^{K-1}$ and pays $-\sum_{i \in \ell} \log q_{y_i}$. Writing $n_k$
for the count of class $k$ in the leaf, the loss is $-\sum_k n_k \log q_k$,
minimised subject to $\sum_k q_k = 1$. The Lagrangian is
$-\sum_k n_k \log q_k + \lambda\left(\sum_k q_k - 1\right)$, and stationarity
in $q_k$ gives $q_k = n_k / \lambda$. Summing over $k$ fixes
$\lambda = \sum_k n_k = n_\ell$, so

$$
q_k = \frac{n_k}{n_\ell} ,
$$

the empirical class proportion. Both leaf constants are the same object: the
plug-in maximum likelihood estimate of the label under the leaf's local
distribution, Gaussian for squared loss and categorical for log loss. For a
binary leaf with $y \in \{0, 1\}$, $q_1$ and the mean of $y$ coincide, so a
single leaf value — the mean of the labels routed there — serves regression and
binary classification alike.

### Why the optimal partition is out of reach

Choosing the partition that minimises total loss, subject to a budget on the
number of leaves, means choosing which of the exponentially many recursive
splits to make at every node, with every choice affecting which choices are
worth making beneath it. Deciding whether a training set of $n$ binary
features admits a decision tree with at most $k$ leaves and zero training error
is $NP$-hard: an optimal tree of bounded size instantiates the general problem
of finding a minimal boolean decision diagram consistent with a truth table,
which is $NP$-complete. No known algorithm finds the globally optimal tree
faster than an exhaustive search over partitions, and none is believed to
exist.

What survives is the greedy relaxation: at each node, search over every
feature and every threshold for the single split that most reduces an impurity
criterion, apply it, and recurse into the two children independently. This
throws away any split that only pays off two or more levels down, which is
what makes the search tractable — the children of a split are never
revisited once chosen, so the algorithm never reasons about combinations of
splits jointly.

## Impurity measures

A split criterion needs an impurity function $I(S)$ that scores a set of
labels $S$: zero when $S$ is pure, larger when $S$ is mixed. Three choices
cover regression and classification.

**Squared-error impurity** is the residual variance of the leaf,
$I_{\text{mse}}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$, the
minimised value of the leaf objective derived above.

**Gini impurity**, for a $K$-class set with proportions $p_k$, is

$$
I_{\text{gini}}(S) = 1 - \sum_{k=1}^{K} p_k^2 = \sum_{k=1}^{K} p_k (1 - p_k) .
$$

The second form reads as an expected error rate: draw two labels
independently from the empirical distribution over $S$, and $p_k(1-p_k)$
summed over $k$ is the probability the second disagrees with a fixed first
draw of class $k$. Gini impurity is the misclassification rate of a classifier
that guesses labels at random according to the leaf's own distribution.

**Entropy** is $I_{\text{ent}}(S) = -\sum_{k=1}^{K} p_k \log p_k$, with the
convention $0 \log 0 = 0$. It is exactly the average log loss the leaf's
optimal constant achieves on its own training points: substitute
$q_k = p_k$ from the derivation above into $-\frac{1}{n_\ell}\sum_i \log
q_{y_i}$ and the sum becomes $-\sum_k p_k \log p_k$. Entropy is not an ad hoc
measure of disorder here; it is the realised training loss of a leaf under the
criterion the tree is ultimately scored on.

Both Gini and entropy restrict, for $K = 2$ with $p = p_1$, to symmetric
concave functions of $p$ that vanish at $p \in \{0, 1\}$:
$I_{\text{gini}}(p) = 2p(1-p)$ and
$I_{\text{ent}}(p) = -p \log p - (1-p)\log(1-p)$.

### Why Gini and entropy rarely disagree

Compare the two near their common maximum at $p = \tfrac{1}{2}$. Gini is exact
quadratic: $I_{\text{gini}}(p) = 2p - 2p^2$, so $I_{\text{gini}}''(p) = -4$
everywhere. For entropy, $I_{\text{ent}}'(p) = \log\frac{1-p}{p}$, which is zero
at $p = \tfrac12$, and $I_{\text{ent}}''(p) = -\frac{1}{p} - \frac{1}{1-p}$,
which equals $-4$ at $p = \tfrac12$ as well. The two impurities have identical
value ordering, identical zeros, and identical curvature at their shared
maximum; they differ only in the quartic and higher terms of their Taylor
expansions away from $p = \tfrac12$. A split that reduces one impurity
substantially reduces the other by nearly the same proportion, because a
weighted average of two concave functions with matching local shape ranks
candidate splits the same way almost everywhere. Disagreements are possible —
the two curves are not identical — but they are confined to the rare case
where two candidate splits achieve nearly equal impurity reduction under one
criterion and the tie-break flips under the other.

## The split-search algorithm

For a candidate feature $j$, evaluating every possible threshold naively costs
$O(n)$ work per threshold and there are $O(n)$ distinct thresholds, giving
$O(n^2)$ per feature. Sorting removes the redundancy. Sort the $n$ values of
feature $j$ once, in $O(n \log n)$, then sweep the threshold from left to
right, moving one point at a time from the right partition to the left one.

For the squared-error criterion, maintain running sums
$S_L = \sum_{i \in L} y_i$ and $Q_L = \sum_{i \in L} y_i^2$ for the left
partition; the right partition's sums follow by subtracting from the totals
$S$ and $Q$ computed once. The left variance is
$Q_L / n_L - (S_L / n_L)^2$, and moving a point from right to left updates
$S_L$ and $Q_L$ by one addition each — $O(1)$ per step, $O(n)$ for the sweep.
For Gini or entropy on a binary label, maintain the running count of positive
labels in the left partition; each impurity is a closed-form function of that
one count and $n_L$, again $O(1)$ per step. Either way, one feature costs
$O(n \log n)$ for the sort plus $O(n)$ for the sweep, and a node with $d$
features costs

$$
O(nd \log n) .
$$

Weighting matters: the criterion used to pick a split is not the raw impurity
of the two children but the reduction

$$
\Delta I = I(S) - \frac{n_L}{n} I(S_L) - \frac{n_R}{n} I(S_R) ,
$$

the parent's impurity minus the sample-weighted average of the children's.
Weighting by $n_L / n$ and $n_R / n$ is what makes $\Delta I \geq 0$ always
achievable by the trivial split that puts everything on one side, so the
search only accepts splits that do strictly better than not splitting.

## Cost-complexity pruning

A tree grown until every leaf is pure drives training risk to zero and
generalises poorly: every leaf ends up fit to the noise of the handful of
points that landed in it. Limiting depth in advance is one fix; pruning a
fully grown tree back is a better one, because it decides which splits earned
their keep only after seeing what the whole tree looks like.

Define the cost complexity of a tree $T$ with leaf set $\text{leaves}(T)$ as

$$
R_\alpha(T) = R(T) + \alpha \, |\text{leaves}(T)| , \qquad
R(T) = \sum_{\ell \in \text{leaves}(T)} n_\ell \, I(\ell) ,
$$

the total (unweighted) impurity across leaves plus a penalty $\alpha$ per leaf.
$R(T)$ only decreases as splits are added — a child's impurity times its count
never exceeds the parent's, by the nonnegativity of $\Delta I$ derived above —
so $\alpha$ is the exchange rate that makes growing the tree costly enough to
stop.

For a fixed $\alpha$, the subtree of the fully grown tree $T_0$ that minimises
$R_\alpha$ decomposes node by node. At an internal node $t$ with subtree
$T_t$, collapsing $T_t$ to a single leaf costs $R(t) + \alpha$, where $R(t)$ is
the impurity of $t$ treated as a leaf; keeping the subtree costs
$R_\alpha(T_t)$ computed recursively from the children. Because $R_\alpha$
decomposes additively over leaves, the cheaper of the two options at every
node is globally optimal — a single bottom-up pass, comparing collapse against
keep at each node, produces the exact minimiser of $R_\alpha(T)$ over every
subtree of $T_0$.

The classical **weakest-link** algorithm computes the whole path at once
rather than one $\alpha$ at a time. Define, for an internal node $t$,

$$
g(t) = \frac{R(t) - R(T_t)}{|\text{leaves}(T_t)| - 1} ,
$$

the per-leaf cost of keeping $t$'s subtree instead of collapsing it. The node
with the smallest $g(t)$ is the weakest link: it is the split that is
cheapest, per leaf saved, to remove. Collapsing it and repeating produces a
nested sequence $T_0 \supset T_1 \supset \cdots \supset \{\text{root}\}$, and
the sequence of collapsed $g(t)$ values is nondecreasing — collapsing the
globally weakest link first only ever needs recomputing $g$ locally, never
revisiting a decision already made. Breiman's theorem states that $T_k$ is the
minimiser of $R_\alpha$ for every $\alpha$ in
$[\,g(t_k),\, g(t_{k+1})\,)$, so this single pass produces the optimal pruned
tree at every complexity level in one sweep, rather than one bottom-up pass
per candidate $\alpha$. The two views agree: the bottom-up collapse-or-keep
computation at a fixed $\alpha$ always lands on some $T_k$ from this nested
sequence, which is why sweeping $\alpha$ upward can only shrink the tree,
never grow a pruned branch back.

### Surrogate splits and missing values

CART handles a missing feature value at prediction time without imputing it.
Alongside the primary split at a node, it records surrogate splits: among the
remaining features, the threshold whose induced partition of the
fully-observed rows agrees most often with the primary split's partition. A
row missing the primary feature is routed by the best surrogate whose feature
it does have, falling back through the ranked list, and to the majority
branch if every surrogate is also missing. This costs an extra $O(nd \log n)$
per node to rank the surrogates, and it degrades gracefully only when some
other feature is correlated with the primary one — a tree trained on features
that go missing together, uninformatively, has no signal left to fall back on.

## Cost and scaling

| Operation | Cost | Notes |
|---|---|---|
| Best split at one node, $n$ rows, $d$ features | $O(nd \log n)$ | Sort per feature, then one linear sweep with running sums. |
| Full tree, presorted feature order carried through recursion | $O(nd \log n + nd \cdot \text{depth})$ | Sort each feature once; partitioning at depth $k$ costs $O(n)$ total across that level. |
| Full tree, re-sorting at every node | $O(n^2 d \log n)$ worst case | Depth up to $n$ on adversarial or highly skewed data. |
| Prediction, one query | $O(\text{depth})$ | A root-to-leaf walk. |
| Cost-complexity pruning, whole path | $O(|T_0|)$ | One bottom-up pass, or one weakest-link sweep. |

The gap between the second and third rows is the entire reason production
implementations pass sorted index arrays down the recursion instead of calling
a general-purpose sort at every node: re-sorting turns a tree that is linear in
$n$ up to logarithms into one that is quadratic.

## Where this breaks

**Axis-aligned bias.** A single split only ever cuts perpendicular to a
coordinate axis. A decision boundary that runs diagonally through feature
space needs a staircase of many splits to approximate, each one adding
variance without adding much accuracy. Rotating or otherwise transforming
features before fitting can remove the staircase, but only if you know which
rotation to apply.

**Instability.** Because the split search is greedy and each split is a hard
decision, a handful of points near a threshold can flip which feature wins at
the root, and every split beneath the root is conditioned on that choice. Two
training sets differing by a small resample can produce structurally different
trees even when their predictions are similar. This is the high-variance,
low-bias regime that the next lesson addresses directly, by averaging many
such trees rather than trying to stabilise any one of them.

**Myopia.** The greedy search evaluates each split assuming every future split
will also be greedy; it cannot see that two mediocre splits in sequence might
beat the best single split available at this step. Features that are only informative
in combination — an XOR-like interaction — can look useless to every
one-step split and never get chosen, regardless of depth budget.

**High-cardinality categorical features.** A feature with many distinct
values offers more candidate thresholds and so more opportunities for one of
them to reduce impurity by chance on the training set, biasing the search
toward such features even when they carry no more signal than a coarser one.

## The exercise

You build CART from its two smallest pieces up to a pruned tree. Implement
`gini` and `entropy`, each taking an integer label array and returning its
impurity by the definitions above, then `best_split(X, y, criterion)`, which
returns the feature, threshold, and impurity reduction of the best axis-aligned
split, using sorted order and running sums rather than a threshold-by-threshold
recomputation.

Then implement `grow_tree`, which recurses `best_split` into a full tree
subject to a maximum depth and a minimum leaf size, and `predict_tree`, which
walks a fitted tree to a leaf for each query row. Finally implement
`cost_complexity_prune(tree, alpha)`, the bottom-up collapse-or-keep pass
derived above.

The checks confirm `gini` and `entropy` against their formulas at a pure node,
a uniform node, and an arbitrary label mix; that `best_split` finds the same
split, to numerical precision, as an $O(n^2)$ brute-force search that tries
every threshold directly; that a tree grown without a depth limit drives
training error to zero; that every leaf's prediction equals the mean of the
training labels actually routed there; and that the number of leaves after
`cost_complexity_prune` is nonincreasing as `alpha` grows, confirming the
nested-sequence property the weakest-link argument predicts.
