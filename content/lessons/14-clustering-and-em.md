---
title: Clustering and expectation maximisation
slug: 14-clustering-and-em
part: "Part 2 — Classical machine learning"
summary: "Group unlabelled points with k-means, then fit a Gaussian mixture with EM and watch the likelihood rise."
minutes: 60
gpu: false
objectives:
  - Compute the k-means objective and run Lloyd's assign-then-update loop.
  - Seed centres with k-means++ and explain why far points get more chance.
  - Compute E-step responsibilities and the log-likelihood in log space, with log-sum-exp.
  - Update a Gaussian mixture's weights, means, and covariances in the M-step.
  - Explain why tiny equal covariances turn EM into the same groups as k-means.
prereqs:
  - 13-pca-and-svd
exercise: 14-clustering-and-em
---

# Clustering and expectation maximisation

Every method so far came with labels. Clustering has none. You give it points and a count of groups, and it puts nearby points together.

Sorting coins by feel is the same job. k-means makes hard piles. Expectation maximisation (EM) makes soft ones, such as 70% one pile and 30% another.

You can run both, and see k-means again as EM with tiny round blobs.

## What you already know

Four earlier ideas are enough to start:

- `X` has shape `(n, d)`, one row per point.
- Squared distance is the sum of squared coordinate gaps.
- A group's mean is `X[mask].mean(axis=0)`.
- A Gaussian peaks at its mean. At 0 with variance 1, the log density is about $-0.919$.

## Score a grouping

Points $0, 1, 4, 5$ and centres $0$ and $1$. Point 0 is closer to 0. Point 1 is closer to 1. Points 4 and 5 are closer to 1. The squared gaps you pay are $0$, $0$, $9$, and $16$. The sum is 25.

That sum is the **k-means objective**. In code the centres live in `centers`.

$$
J = \sum_{i=1}^{n} \lVert x_i - c_{a_i} \rVert^2 .
$$

Add each squared gap to its own centre. Plug in and you get 25 again:

```python
J = float(np.sum((X - centers[assignments]) ** 2))
```

The usual mistake is returning the mean, or a numpy scalar instead of a Python float.

## Alternate assign and update

**Lloyd's algorithm** assigns each point to the nearest centre, then moves each centre to its group's mean. Each step lowers $J$ or leaves it unchanged.

Centre 0 stays at 0. Centre 1 moves to $(1+4+5)/3 = 10/3$. Point 1 switches, because its squared gaps are 1 and $(7/3)^2$. Then $J = 1 + (2/3)^2 + (5/3)^2 = 38/9$.

The next means are $0.5$ and $4.5$. Assignments do not change. Each squared gap is $0.25$, so $J = 1$. Another pass does nothing. The history is $[25, 38/9, 1, 1]$.

Nearest-centre search can drop a shared term:

$$
\lVert x - c \rVert^2 = \lVert x \rVert^2 - 2\, x \cdot c + \lVert c \rVert^2 .
$$

$\lVert x \rVert^2$ does not change the winner:

```python
scores = np.sum(centers**2, axis=1) - 2 * X @ centers.T
assignments = np.argmin(scores, axis=1)
```

The usual mistake is the mean of an empty group, which is `nan`. Keep the old centre.

> [!WARNING]
> `scores` can be negative, because $\lVert x \rVert^2$ was dropped. Use it only in `argmin`. Compute $J$ from real gaps.

> [!KEY]
> Each Lloyd pass, assign then update, lowers $J$ or leaves it the same.

## Spread the starting centres

Points $0$, $1$, and $10$, with the first centre at $0$. Squared distances are $0$, $1$, and $100$. **k-means++** picks the next centre with probability proportional to those squares: $0$, $1/101$, and $100/101$. The point at 10 is about 100 times as likely as the point at 1.

$$
p(x_i) = \frac{D(x_i)^2}{\sum_{j=1}^{n} D(x_j)^2} .
$$

$D(x_i)$ is distance to the nearest centre already chosen. Squared distance over the sum of squared distances makes the chances add to 1. Plug in and you get $0$, $1/101$, and $100/101$.

On average the seeded objective stays within about $8(\ln k + 2)$ of the best. For two centres, $8(0.693+2) = 21.5$. That bound is an average. One draw can still be poor:

```python
idx = rng.choice(n, p=sq_dist / sq_dist.sum())
sq_dist = np.minimum(sq_dist, np.sum((X - X[idx]) ** 2, axis=1))
```

The usual mistake is plain distance. The chances become $0$, $1/11$, and $10/11$, only a factor of 10, not 100.

> [!KEY]
> k-means++ draws the next centre with probability proportional to squared distance from the centres already chosen.

## Share two points between two blobs

A **Gaussian mixture** hides which Gaussian blob produced each point. One round on two points comes before the general update.

Points $x = 0$ and $x = 2$. Blobs have means $0$ and $2$, variance $1$, and weights $0.5$. Use $\log(2\pi) = 1.837877$ and $\log 0.5 = -0.693147$.

Log density is $-\tfrac12$ times $(1.837877 + \text{squared gap})$. At point 0 the near gap is 0, so the log density is $-0.918939$. The far gap is 4, so the log density is $-2.918939$. Add $\log 0.5$ and the scores are $-1.612086$ and $-3.612086$, which differ by 2.

Subtract the larger score. Shifted exponentials are $1$ and $e^{-2} = 0.135335$, summing to $1.135335$. Shares are $0.880797$ and $0.119203$, and point 2 swaps them.

Those shares are the **responsibilities**. The log-likelihood of one point is $-1.612086 + \log 1.135335 = -1.485158$. Two points give $-2.970$.

Refit blob 0. The soft count is $N_0 = 1$, so the weight stays $0.5$. The mean is $0.119203 \times 2 = 0.238406$. Squared gaps are $0.056837$ and $3.103213$. Weighted, $0.880797 \times 0.056837 = 0.050062$ and $0.119203 \times 3.103213 = 0.369912$. The sum is $0.419974$. With $\epsilon = 0$, that is the variance. Blob 1 has mean $1.761594$.

Those two steps are EM. The matrix form is next.

## Write the general E-step

Blob $j$ has weight $\pi_j$, mean $\mu_j$, and covariance $\Sigma_j$. One Gaussian's log density is

$$
\log \mathrm{N}(x; \mu, \Sigma) = -\tfrac{1}{2}\left( d \log(2\pi) + \log\det\Sigma + (x-\mu)^\top \Sigma^{-1} (x-\mu) \right) .
$$

Add $d\log(2\pi)$, the log determinant, and the squared distance measured with $\Sigma^{-1}$. Multiply by $-\tfrac12$.

The responsibility and the log-likelihood are

$$
r_{ij} = \frac{\pi_j \, \mathrm{N}(x_i; \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \pi_l \, \mathrm{N}(x_i; \mu_l, \Sigma_l)} .
$$

$$
\log L = \sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_j \, \mathrm{N}(x_i; \mu_j, \Sigma_j) .
$$

Weight times density, divided so the row adds to 1. $\log L$ sums the log of the denominator. The two-point shares and $\log L = -2.970$ match.

Store $s_{ij} = \log\pi_j + \log\mathrm{N}$, and subtract the row max $m_i$ before the exponential:

$$
\log \sum_j e^{s_{ij}} = m_i + \log \sum_j e^{s_{ij} - m_i} .
$$

Every shifted exponent is at most 0, so the largest term is 1:

```python
m = s.max(axis=1, keepdims=True)
resp = np.exp(s - m)
resp /= resp.sum(axis=1, keepdims=True)
```

The usual mistake is exponentiating before the row maximum. Far blobs become `0 / 0`.

> [!TIP]
> `keepdims=True` keeps shape `(n, 1)`. Shape `(n,)` broadcasts the wrong way when $n = k$.

> [!KEY]
> The E-step shares each point in proportion to weight times density.

## Refit the blobs

Hold the responsibilities fixed. Let $N_j = \sum_i r_{ij}$.

$$
\pi_j = \frac{N_j}{n}, \qquad
\mu_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, x_i, \qquad
\Sigma_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, (x_i - \mu_j)(x_i - \mu_j)^\top + \epsilon I .
$$

Weight is a share of $n$. The mean is weighted. The covariance uses outer products around that mean, plus `reg_covar` on the diagonal. Blob 0 with $\epsilon = 0$ returns $0.5$, $0.238406$, and $0.419974$:

```python
means = (resp.T @ X) / resp.sum(axis=0)[:, None]
```

The usual mistake is the old mean inside $\Sigma_j$, or a missing `reg_covar * np.eye(d)`.

## See why the likelihood does not fall

After the M-step, $\log L$ rises from $-2.970$ to $-2.439$. A lower bound, the ELBO, explains why.

$$
\log p(x) = \mathrm{ELBO}(q) + \mathrm{KL}\big(q(z) \,\|\, p(z \mid x)\big) .
$$

The KL term is at least 0, so the ELBO sits on or below $\log p(x)$. For this mixture

$$
\mathrm{ELBO}(q) = \sum_i \sum_j r_{ij} \big( \log \pi_j + \log \mathrm{N}(x_i; \mu_j, \Sigma_j) - \log r_{ij} \big) .
$$

When $r_{ij}$ is the posterior, the KL term is 0 and the ELBO equals $\log L$.

Point 0, near blob: $-0.693147 - 0.918939 + 0.126928 = -1.485158$, times $0.880797$, is $-1.308123$. The far term is $-0.177035$. The other point matches, and the total is $-2.970$.

The M-step raises that bound with $r$ fixed, to $-2.626$. True $\log L$ is $-2.439$, above the bound. The next E-step puts the KL term back at 0, so a full round never lowers $\log L$.

The usual mistake is an M-step that reads a mean you have already overwritten. The sum in numpy is as follows:

```python
elbo = np.sum(resp * (log_joint - np.log(resp)))
```

> [!KEY]
> One E-step then one M-step never lowers the log-likelihood. A drop in the history is a bug.

## Recover k-means from tiny blobs

Equal weights and covariance $\sigma^2 I$ make responsibilities depend only on squared distance over $2\sigma^2$.

At $x = 1$, means $0$ and $4$, and $\sigma^2 = 1$, the shares are about $0.982$ and $0.018$. At $\sigma^2 = 10^{-6}$ the nearer share is 1, a hard assignment. Shares that are only 0 or 1 make the M-step a k-means update.

The check compares `argmax` at $\sigma^2 = 10^{-6}$ with Lloyd's assignment.

> [!RECAP]
> Keep these five facts in hand:
> - $J$ sums squared gaps from each point to `centers[assignments]`.
> - Lloyd assigns, then updates, keeps an empty centre, and stops when $J$ stops changing.
> - k-means++ draws with probability $D(x)^2 / \sum_j D(x_j)^2$, and shrinks `sq_dist` with `np.minimum`.
> - The E-step builds $\log\pi_j + \log\mathrm{N}$ and applies log-sum-exp on each row.
> - The M-step sets weights $N_j/n$, weighted means, and covariances around those means, plus `reg_covar` on the diagonal.

## Build it step by step

Build the six functions in this order:

1. **`kmeans_objective(X, centers, assignments)`** returns a Python float from shapes `(n, d)`, `(k, d)`, and `(n,)`. Sum gaps to `centers[assignments]`. The usual mistake is the mean. Write it like this:

    ```python
    def kmeans_objective(X, centers, assignments):
        # float sum of squared gaps. Not the mean.
    ```

2. **`lloyd(X, k, init, iters)`** copies `init`, records $J$ after the first assign, then updates non-empty centres up to `iters` times. Stop when $J$ repeats. The usual mistake is a `nan` mean. Write it like this:

    ```python
    def lloyd(X, k, init, iters):
        # Copy init. Skip the mean when a group is empty.
    ```

3. **`kmeans_plusplus_init(X, k, rng)`** returns `(k, d)`, each row a real row of `X`. First index from `rng.integers`. Later indices from `rng.choice(n, p=probs)`. If `sq_dist.sum()` is 0, use uniform probabilities. The usual mistake is plain distance. Write it like this:

    ```python
    def kmeans_plusplus_init(X, k, rng):
        # Probability proportional to squared distance. Then np.minimum.
    ```

4. **`gmm_e_step(X, means, covs, weights)`** returns responsibilities `(n, k)` and a float log-likelihood. Build log scores, subtract the row max, then exponentiate. The usual mistake is skipping that subtraction. Write it like this:

    ```python
    def gmm_e_step(X, means, covs, weights):
        # log weight plus log density, then log-sum-exp per row.
    ```

5. **`gmm_m_step(X, responsibilities, reg_covar)`** returns means `(k, d)`, covariances `(k, d, d)`, and weights `(k,)`. Covariance uses the updated mean. The usual mistake is the previous mean. Write it like this:

    ```python
    def gmm_m_step(X, responsibilities, reg_covar):
        # weights = n_k / n. Add reg_covar * eye on each covariance.
    ```

6. **`fit_gmm(X, k, iters, rng, reg_covar)`** returns a history of length `iters + 1`. Seed means with k-means++, covariances with `np.var(X, axis=0).mean() * np.eye(d)`, and weights with `1/k`. E-step before any M-step. The usual mistake is reversing that order. Write it like this:

    ```python
    def fit_gmm(X, k, iters, rng, reg_covar):
        # Record log L after the first E-step, then after every later E-step.
    ```

## Cost and scaling

One iteration touches every point once per cluster. The costs line up as follows:

| Step | Time | Memory |
|---|---|---|
| k-means assign | About $n \cdot k \cdot d$ | The `(n, k)` score matrix. |
| k-means update | About $n \cdot d$ | The `(k, d)` centres. |
| k-means++ , all $k$ centres | About $n \cdot k \cdot d$ | One `(n,)` distance array. |
| GMM E-step | About $n k d^2$, plus $k d^3$ to invert | Responsibilities `(n, k)`. |
| GMM M-step | About $n k d^2$ | $k$ covariances, $k d^2$ numbers. |

Doubling $n$ doubles the work. Diagonal covariances store $d$ numbers, not $d^2$.

## Where this breaks

Five failures show up in practice:

- You choose the group count. Compare held-out log-likelihood, or watch where $J$ flattens.
- Both methods stop locally. Keep the best of several seeds.
- k-means fits round groups of similar size. Full covariances fit stretched blobs.
- A blob on one point sends the log-likelihood to infinity. `reg_covar` blocks that.
- Standardise columns when units differ, or one feature dominates.

## The exercise

You implement `kmeans_objective`, `lloyd`, `kmeans_plusplus_init`, `gmm_e_step`, `gmm_m_step`, and `fit_gmm`.

Checks: $J$ sums squared distances, Lloyd's history never rises, and k-means++ beats uniform seeds. Responsibilities sum to 1. Log-likelihood matches log-sum-exp. The M-step recovers the weighted mean and covariance. EM's log-likelihood never falls. Tiny equal covariances match Lloyd under `argmax`.
