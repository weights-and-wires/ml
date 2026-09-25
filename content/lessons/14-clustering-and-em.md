---
title: Clustering and expectation maximisation
slug: 14-clustering-and-em
part: "Part 2 — Classical machine learning"
summary: Group unlabelled points with k-means and k-means++, then fit a Gaussian mixture with EM and see k-means inside it.
minutes: 60
gpu: false
objectives:
  - Compute the k-means objective and run Lloyd's assign-then-update loop.
  - Build k-means++ seeding and explain why it beats picking centers at random.
  - Compute E-step responsibilities and the log-likelihood without overflow, using log-sum-exp.
  - Build the M-step updates for a Gaussian mixture's weights, means, and covariances.
  - Explain why EM with tiny, equal covariances gives the same groups as k-means.
prereqs:
  - 13-pca-and-svd
exercise: 14-clustering-and-em
---

# Clustering and expectation maximisation

Every method so far came with labels: for each example, somebody told you the
right answer. Clustering works without labels. You hand it a pile of points
and a number $k$, and it splits the points into $k$ groups so that points in
the same group sit close together.

Think of sorting a bag of mixed coins by feel. You do not know the coin names,
but you can make piles of coins that are alike. k-means does exactly that with
distances. Expectation maximisation (EM) does a softer version: instead of
putting each coin in one pile, it says "70% this pile, 30% that pile".

You build both in this lesson. You also see that k-means is a special case of
EM, so the two ideas are really one idea.

## What you already know

- A data matrix `X` has shape `(n, d)`: $n$ examples, one per row, each with $d$ features.
- The squared distance between two points is the sum of squared differences of their coordinates.
- A mean of a group of rows is `X[mask].mean(axis=0)`.
- A Gaussian (normal) distribution has a mean and a variance, and its density is highest at the mean.
- Working with logs of probabilities avoids numbers that are too small for a computer to store.

## The k-means objective

k-means describes each group by one point, its **center**. A good set of
centers is one where every point lies close to the center of its group. You
measure "close" with squared distance and add it up over all points.

Write $c_1, \dots, c_k$ for the centers. Each point $x_i$ has an
**assignment** $a_i$, the index of its group. The objective is:

$$
J = \sum_{i=1}^{n} \lVert x_i - c_{a_i} \rVert^2
$$

Read it aloud: for each point $x_i$, look up its center $c_{a_i}$, take the
difference, square each coordinate, and add. Then add those numbers over all
$n$ points. $\lVert v \rVert^2$ means the squared length of a vector $v$, the
sum of its squared entries. Smaller $J$ means tighter groups.

In numpy, `centers[assignments]` picks the right center for every row at once:

```python
diffs = X - centers[assignments]      # (n, d)
J = np.sum(diffs * diffs)             # a single number
```

## Lloyd's algorithm

Finding the very best centers is too hard to do exactly. Lloyd's algorithm
finds a good answer by repeating two easy steps:

1. **Assign.** Put every point in the group of its nearest center.
2. **Update.** Move every center to the mean of the points in its group.

Each step can only lower $J$ or leave it the same. The assign step picks the
closest center for each point, which is the best choice for fixed centers. The
update step picks the mean, which is the point with the smallest total squared
distance to a group. So $J$ never goes up, and because there are finitely
many ways to split the points, the loop eventually stops changing.

> [!KEY]
> Each Lloyd step, assign then update, lowers $J$ or leaves it the same. That is why the history your code returns never goes up.

To find the nearest center fast, expand the squared distance:

$$
\lVert x - c \rVert^2 = \lVert x \rVert^2 - 2\, x \cdot c + \lVert c \rVert^2
$$

Read it aloud: the squared distance is the point's squared length, minus twice
the dot product of point and center, plus the center's squared length. The
first term is the same for every center, so it cannot change which center
wins. You can drop it and compare only the other two terms:

```python
scores = np.sum(centers**2, axis=1)[None, :] - 2 * X @ centers.T   # (n, k)
assignments = np.argmin(scores, axis=1)                             # (n,)
```

> [!WARNING]
> `scores` leaves out $\lVert x \rVert^2$, so it is not a squared distance and can be negative. Use it only to pick the `argmin`, and compute $J$ from real distances.

### Worked example by hand

Take four points on a line, $x = 0, 1, 4, 5$, with $k = 2$ and starting
centers $c_1 = 0$ and $c_2 = 1$. Groups are numbered 0 and 1, as in code.

| Step | Centers | Assignments | $J$ |
|---|---|---|---|
| Start | 0, 1 | 0, 1, 1, 1 | $0 + 0 + 9 + 16 = 25$ |
| Iteration 1 | 0, 3.333 | 0, 0, 1, 1 | $0 + 1 + 0.444 + 2.778 = 4.222$ |
| Iteration 2 | 0.5, 4.5 | 0, 0, 1, 1 | $0.25 \times 4 = 1.0$ |
| Iteration 3 | 0.5, 4.5 | 0, 0, 1, 1 | $1.0$, unchanged, so stop |

In iteration 1, the second center moves to $(1 + 4 + 5)/3 = 3.333$. Point 1
is then closer to 0 than to 3.333, so it switches groups. The history your
code returns for this example is `[25.0, 4.222, 1.0, 1.0]`.

## k-means++ seeding

Lloyd's algorithm only finds a local answer, and where it ends up depends on
where the centers start. If you pick starting centers at random, two of them
often land in the same true cluster, and one real cluster gets no center.

k-means++ fixes this by spreading the starting centers out. It picks the first
center as a random data point. Then, for each point, it measures $D(x)$, the
distance to the nearest center chosen so far. It picks the next center with
probability:

$$
p(x_i) = \frac{D(x_i)^2}{\sum_{j=1}^{n} D(x_j)^2}
$$

Read it aloud: a point's chance of being picked is its squared distance to the
nearest chosen center, divided by the total of those squared distances. Far
away points are much more likely to be picked. Points that are already centers
have $D = 0$, so they are never picked again.

This rule comes with a guarantee: on average, the objective after seeding is
within a factor of about $8(\ln k + 2)$ of the best possible objective. Random
seeding has no such promise.

In numpy, keep a running array of squared distances and shrink it with
`np.minimum` each time you add a center:

```python
probs = sq_dist / sq_dist.sum()
idx = rng.choice(n, p=probs)
sq_dist = np.minimum(sq_dist, np.sum((X - X[idx])**2, axis=1))
```

## Gaussian mixtures: soft clusters

k-means gives every point exactly one group. A **Gaussian mixture model**
(GMM) says instead that each group is a Gaussian "blob" with its own center,
shape, and size, and every point came from one blob that you did not see.

A mixture with $k$ components has three sets of parameters:

- **Weights** $\pi_j$: the fraction of points that come from blob $j$. They add up to 1.
- **Means** $\mu_j$: the center of blob $j$, shape `(d,)`.
- **Covariances** $\Sigma_j$: a `(d, d)` matrix that sets the blob's spread and tilt.

The density of one Gaussian at a point $x$, written in log form, is:

$$
\log \mathrm{N}(x; \mu, \Sigma) = -\tfrac{1}{2}\left( d \log(2\pi) + \log\det\Sigma + (x-\mu)^\top \Sigma^{-1} (x-\mu) \right)
$$

Read it aloud: start from a constant that depends on the dimension $d$, add the
log of the determinant of $\Sigma$ (a bigger blob has a bigger determinant and
a lower peak), and add the squared distance from $x$ to $\mu$, measured in the
blob's own units. Multiply the total by $-\frac{1}{2}$. The `solution.py` file
computes it with `np.linalg.slogdet` and `np.linalg.inv`.

```python
sign, logdet = np.linalg.slogdet(cov)
maha = np.einsum("ij,jk,ik->i", X - mean, np.linalg.inv(cov), X - mean)
log_pdf = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)       # (n,)
```

## The E-step: who owns each point

EM repeats two steps, like Lloyd's algorithm. The **E-step** (expectation)
works out, for every point, how likely it is that each blob produced it. These
numbers are the **responsibilities** $r_{ij}$. Each row adds up to 1.

$$
r_{ij} = \frac{\pi_j \, \mathrm{N}(x_i; \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \pi_l \, \mathrm{N}(x_i; \mu_l, \Sigma_l)}
$$

Read it aloud: take blob $j$'s weight times its density at $x_i$. Divide by the
same quantity added up over all blobs. The top is "how well blob $j$ explains
this point", and the bottom makes the row sum to 1.

> [!KEY]
> The E-step shares each point among the blobs in proportion to weight times density. Those shares are the responsibilities.

The bottom of that fraction is also the mixture's density at $x_i$. The
**log-likelihood** is the sum of its log over all points:

$$
\log L = \sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_j \, \mathrm{N}(x_i; \mu_j, \Sigma_j)
$$

Read it aloud: for each point, add up weight times density over all blobs,
take the log, and add over points. Higher means the mixture fits the data
better.

### Stay in log space with log-sum-exp

Densities can be astronomically small, such as $e^{-1000}$, which a computer
rounds to 0. Then you divide 0 by 0. The fix is the **log-sum-exp** trick.
Store $s_{ij} = \log \pi_j + \log \mathrm{N}(x_i; \mu_j, \Sigma_j)$. For each
row, subtract its largest value $m_i$ before exponentiating:

$$
\log \sum_j e^{s_{ij}} = m_i + \log \sum_j e^{s_{ij} - m_i}
$$

Read it aloud: pull the biggest term out front, then every shifted exponent is
at most 0, so the largest shifted term is exactly 1 and nothing overflows or
vanishes.

```python
m = s.max(axis=1, keepdims=True)          # (n, 1)
shifted = np.exp(s - m)                   # (n, k)
row_sums = shifted.sum(axis=1, keepdims=True)
resp = shifted / row_sums                 # rows sum to 1
log_lik = np.sum(m[:, 0] + np.log(row_sums[:, 0]))
```

> [!TIP]
> `keepdims=True` keeps `m` at shape `(n, 1)`, so `s - m` subtracts each row's own maximum. With shape `(n,)`, numpy raises an error, or, when $n = k$, silently subtracts the wrong numbers.

### Worked example by hand

One point $x = 1$ in one dimension. Two blobs with means 0 and 4, variance 1,
and weights 0.5 each. Here $\log(2\pi) = 1.8379$ and $\log 0.5 = -0.6931$.

- Blob 0: $\log\mathrm{N} = -0.5(1.8379 + 0 + 1^2) = -1.4189$, so $s_0 = -2.1121$.
- Blob 1: $\log\mathrm{N} = -0.5(1.8379 + 0 + 3^2) = -5.4189$, so $s_1 = -6.1121$.
- The row maximum is $m = -2.1121$. Shifted exponentials: $e^{0} = 1$ and $e^{-4} = 0.01832$. Their sum is $1.01832$.
- Responsibilities: $1/1.01832 = 0.9820$ and $0.01832/1.01832 = 0.0180$.
- Log-likelihood: $-2.1121 + \log 1.01832 = -2.1121 + 0.0182 = -2.0939$.

## The M-step: refit each blob

The **M-step** (maximisation) treats the responsibilities as fixed and refits
each blob. It is a weighted version of "average the points in the group".
Point $i$ counts as $r_{ij}$ of a point in blob $j$.

First count how many points each blob owns, in soft terms:
$N_j = \sum_i r_{ij}$. Then:

$$
\pi_j = \frac{N_j}{n}, \qquad
\mu_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, x_i, \qquad
\Sigma_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, (x_i - \mu_j)(x_i - \mu_j)^\top + \epsilon I
$$

Read it aloud, one piece at a time:

- The weight is the blob's share of the $n$ points.
- The mean is the responsibility-weighted average of the points.
- The covariance is the weighted average of each point's outer product with itself, after subtracting the blob's updated mean. $(v)(v)^\top$ is a `(d, d)` matrix, the **outer product**.
- $\epsilon I$ adds a small number `reg_covar` to the diagonal. It stops a blob from collapsing onto one point and becoming impossible to invert.

```python
n_k = resp.sum(axis=0)                            # (k,)
weights = n_k / n
means = (resp.T @ X) / n_k[:, None]               # (k, d)
diff = X - means[j]                               # (n, d)
cov_j = (resp[:, j][:, None] * diff).T @ diff / n_k[j] + reg_covar * np.eye(d)
```

### Worked example by hand

Three points $x = 0, 1, 4$ and blob 0 with responsibilities $r = 1, 0.5, 0$.
Set $\epsilon = 0$ for the example.

- $N_0 = 1 + 0.5 + 0 = 1.5$, so $\pi_0 = 1.5 / 3 = 0.5$.
- $\mu_0 = (1 \cdot 0 + 0.5 \cdot 1 + 0 \cdot 4) / 1.5 = 0.3333$.
- $\Sigma_0 = (1 \cdot 0.3333^2 + 0.5 \cdot 0.6667^2) / 1.5 = (0.1111 + 0.2222) / 1.5 = 0.2222$.

### Why EM keeps improving

Each full round of E-step then M-step never lowers the log-likelihood. In words:
the E-step builds a lower bound on the log-likelihood that touches it exactly
at the current parameters. The M-step pushes that bound up. Since the true
curve sits above the bound, it rises at least as much. Your `fit_gmm` history
should never decrease.

> [!KEY]
> A round of E-step then M-step never lowers the log-likelihood. If your history ever drops, you have a bug.

## k-means is EM with tiny blobs

Give every blob the same weight and the same covariance $\sigma^2 I$, a round
blob of variance $\sigma^2$. Then the responsibility of blob $j$ depends only
on the squared distance to $\mu_j$, divided by $2\sigma^2$. As $\sigma^2$
shrinks toward 0, the nearest blob's term dominates everything else, and its
responsibility goes to 1. Every point then belongs fully to its nearest
center, which is the k-means assign step. The M-step with 0-or-1
responsibilities is the k-means update step. One of the checks runs your
E-step with $\sigma^2 = 10^{-6}$ and compares the `argmax` of each row to
Lloyd's assignments.

> [!KEY]
> k-means is EM with equal weights and round blobs whose variance shrinks to 0. Soft responsibilities then become hard, nearest-center assignments.

> [!RECAP]
> - $J$ adds up the squared distance from each point to its own center, found with `centers[assignments]`.
> - Lloyd's algorithm alternates assign and update, keeps the old center for an empty group, and stops when $J$ stops changing.
> - k-means++ picks each new center with probability $D(x)^2 / \sum_j D(x_j)^2$, and shrinks `sq_dist` with `np.minimum`.
> - The E-step builds $s_{ij} = \log \pi_j + \log \mathrm{N}(x_i; \mu_j, \Sigma_j)$ and applies log-sum-exp to each row.
> - The M-step sets $\pi_j = N_j / n$, the weighted means, and the weighted covariances around the new means, plus `reg_covar` on the diagonal.

## Build it step by step

Build the functions in this order, which matches the checks.

1. **`kmeans_objective(X, centers, assignments)`.** Takes `X` `(n, d)`, `centers` `(k, d)`, and integer `assignments` `(n,)`. Returns a Python float. Index with `centers[assignments]`, subtract, square, and `np.sum`. Common mistake: returning the mean instead of the sum, or returning a numpy scalar instead of `float(...)`.
2. **`lloyd(X, k, init, iters)`.** Copy `init` so you do not change the caller's array. Assign once and record the objective. Then loop up to `iters` times: update each center to `X[assignments == j].mean(axis=0)`, reassign, append the objective, and `break` when the last two history values are equal. Return `(centers, assignments, history)`. Common mistake: averaging an empty group, which gives `NaN`. Skip the update when a group has no points, so it keeps its old center.
3. **`kmeans_plusplus_init(X, k, rng)`.** Pick the first index with `rng.integers(n)`. Keep `sq_dist` `(n,)` for the distance to the nearest chosen center. For each later center, normalise to `probs`, call `rng.choice(n, p=probs)`, and update `sq_dist` with `np.minimum`. Returns `(k, d)`. Common mistake: using plain distance instead of squared distance, or recomputing distances to all centers every time. If `sq_dist.sum()` is 0, fall back to uniform probabilities.
4. **`gmm_e_step(X, means, covs, weights)`.** Build `s` of shape `(n, k)` with one column per blob: `np.log(weights[j])` plus the log density. Apply log-sum-exp row by row. Return `(responsibilities, log_likelihood)`, with the log-likelihood as a float. Common mistake: exponentiating before subtracting the row maximum, which turns far-away blobs into `0 / 0`.
5. **`gmm_m_step(X, responsibilities, reg_covar)`.** Compute `n_k`, then weights `(k,)`, means `(k, d)`, and covariances `(k, d, d)` with a loop over `j`. Common mistake: subtracting the old mean instead of the updated one, or forgetting `reg_covar * np.eye(d)`.
6. **`fit_gmm(X, k, iters, rng, reg_covar)`.** Start the means with `kmeans_plusplus_init`. Start every covariance as `np.var(X, axis=0).mean() * np.eye(d)` and the weights as `np.full(k, 1/k)`. Run one E-step and record its log-likelihood. Then run `iters` rounds of M-step then E-step, recording each. The history has length `iters + 1`. Common mistake: running the M-step first on responsibilities that do not exist yet.

## Cost and scaling

Each iteration touches every point once for every cluster. The table shows
the cost of one iteration.

| Step | Time | Memory |
|---|---|---|
| k-means assign | about $n \cdot k \cdot d$ | the `(n, k)` score matrix |
| k-means update | about $n \cdot d$ | the `(k, d)` centers |
| k-means++ seeding (all $k$ centers) | about $n \cdot k \cdot d$ | one `(n,)` distance array |
| GMM E-step | about $n \cdot k \cdot d^2$, plus $k \cdot d^3$ to invert | the `(n, k)` responsibilities |
| GMM M-step | about $n \cdot k \cdot d^2$ | `k` covariance matrices, $k \cdot d^2$ numbers |

Doubling $n$ doubles the work. The GMM's full covariances get expensive as $d$
grows, because each one has $d^2$ entries. With many features, people use
diagonal covariances, which cost only $d$ numbers each.

## Where this breaks

**You must choose $k$.** Neither method tells you how many groups exist. Try
several values and compare, for example with the log-likelihood on held-out
data, or look for the value of $k$ where the k-means objective stops dropping
quickly.

**Local optima.** Both methods can get stuck in a poor answer. Run them from
several random seeds and keep the best result.

**k-means likes round, equal-sized groups.** It uses plain distance, so long,
thin, or very unequal clusters get cut in the wrong places. A GMM with full
covariances handles stretched blobs better.

**A GMM blob can collapse.** If one blob shrinks onto a single point, its
density there grows without limit and the log-likelihood shoots to infinity.
The `reg_covar` term on the diagonal prevents this.

**Features on different scales.** A feature measured in thousands dominates
one measured in fractions. Standardise each feature before clustering.

## The exercise

You implement `kmeans_objective`, `lloyd`, `kmeans_plusplus_init`,
`gmm_e_step`, `gmm_m_step`, and `fit_gmm` in `starter.py`.

The checks confirm that the objective sums squared distances, that Lloyd's
history never increases, and that k-means++ seeding beats uniform seeding on
planted clusters. They check that responsibilities sum to one per point, that
your log-likelihood matches a direct log-sum-exp, and that the M-step recovers
the mean and covariance. Finally, they confirm that EM's log-likelihood never
decreases and that EM with tiny, equal covariances assigns every point exactly
as Lloyd's algorithm does.
