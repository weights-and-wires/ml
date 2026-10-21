---
title: PCA and the SVD
slug: 13-pca-and-svd
part: "Part 2 — Classical machine learning"
summary: "Find the directions a cloud spreads along, from the SVD, from power iteration, and from a noisy model fitted by EM."
minutes: 65
gpu: false
objectives:
  - Centre data and read principal components, scores, and explained variance from the SVD.
  - Reconstruct data from k components, and match the error to the singular values you dropped.
  - Find the top eigenvector with power iteration, and tie its speed to the eigenvalue gap.
  - Fit probabilistic PCA with EM, and check that the log-likelihood never falls.
prereqs:
  - 02-linear-regression
exercise: 13-pca-and-svd
---

# PCA and the SVD

So far every model had a label on each row. Principal component analysis (PCA) has only the points. It asks which directions the cloud spreads along, and which directions are thin enough to drop.

Picture a pancake in three dimensions. Two coordinates in its plane describe each point.

You can name the directions three ways, find the top one with a short loop, and fit a noisy copy of that plane.

## What you already know

Four earlier ideas are enough to start:

- $X = U S V^{\top}$, with orthonormal columns in $U$ and $V$, and $s_1 \ge s_2 \ge \cdots \ge 0$.
- An eigenvector obeys $Av = \lambda v$. Multiplication by $A$ stretches $v$ by $\lambda$.
- Variance is an average squared gap. For 0 and 2, with divisor $n-1 = 1$, the variance is 2.
- $\lVert v \rVert$ is length, `np.linalg.norm(v)`.

## Centre the cloud

Take the points $(2, 0)$, $(0, 2)$, and $(4, 4)$. Column means are $(2+0+4)/3 = 2$ and $(0+2+4)/3 = 2$. Subtract $(2, 2)$ from each row. You get $(0, -2)$, $(-2, 0)$, and $(2, 2)$.

That shift is **centring**. Spread is measured from the origin.

$$
X_c = X - \bar x .
$$

$\bar x$ is the vector of column means, subtracted from every row. The three points give the centred rows above:

```python
mean = X.mean(axis=0)
Xc = X - mean
```

The usual mistake is `axis=1`. Skip centring and the first direction points at the cloud's middle.

## Measure spread along a line

A line through the origin turns each centred point into one score. Try $u = (1, 1)/\sqrt{2}$. Dot products are $-2/\sqrt{2} = -\sqrt{2}$, $-\sqrt{2}$, and $4/\sqrt{2} = 2\sqrt{2}$. Squared scores are 2, 2, and 8, summing to 12. Divide by $n-1 = 2$ and the variance is 6.

Try $w = (1, -1)/\sqrt{2}$. Scores are $\sqrt{2}$, $-\sqrt{2}$, and 0. Squared scores sum to 4. The variance is $4/2 = 2$.

The winning direction is the **first principal component**. Component variances are

$$
\lambda_i = \frac{s_i^2}{n - 1} .
$$

Square singular value $i$ and divide by $n-1$. Here $s_1^2 = 12$ and $s_2^2 = 4$, so the variances are 6 and 2, the two numbers you computed:

```python
explained_variance = s[:k] ** 2 / (n - 1)
```

The usual mistake is dividing by $n$. The checks use $n-1$, as `np.cov` does.

> [!KEY]
> The first principal component is the direction whose scores have the largest variance.

## Match the shadows, not the points

Squared length equals squared shadow plus squared distance to the line. A longer shadow is a shorter distance.

Centred squared lengths are $0+4$, $4+0$, and $4+4$, totalling 16. Squared scores on $u$ already summed to 12. The leftover is 4.

A score $z$ comes back as $z u$. The scores $-\sqrt{2}$, $-\sqrt{2}$, and $2\sqrt{2}$ come back as $(-1,-1)$, $(-1,-1)$, and $(2,2)$. Add the mean $(2,2)$ and you get $(1,1)$, $(1,1)$, and $(4,4)$. Squared errors are 2, 2, and 0, totalling 4.

In matrix form,

$$
\hat X = \bar x + Z V_k^{\top} .
$$

Each row is the mean plus each score times its component. With $k = 1$ you rebuild $(1,1)$, $(1,1)$, and $(4,4)$:

```python
X_hat = mean + scores @ components.T
```

The usual mistake is forgetting the mean.

> [!KEY]
> The maximum-variance direction is also the one that keeps points closest to their shadows.

## Read the same directions from the covariance

**Covariance** is an average product of centred columns, divided by $n-1$. Column squares $0+4+4$ give variance $8/2 = 4$. The column product $0+0+4$ gives covariance $4/2 = 2$. The matrix has rows $(4, 2)$ and $(2, 4)$.

Then $(4-\lambda)^2 - 4 = 0$, so $\lambda$ is 6 or 2. For 6, $v_1 = v_2$, and the unit vector is $(1,1)/\sqrt{2}$. Same line as the maximum-variance direction.

PCA is those three descriptions at once: largest score variance, smallest reconstruction error, and the top covariance eigenvectors. The SVD of $X_c$ lists them. Components are the first $k$ columns of $V$, scores are $X_c V_k$, and $\lambda_i = s_i^2/(n-1)$.

Here $s_1 = \sqrt{12} \approx 3.464$ and $s_2 = 2$:

```python
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
components = Vt[:k].T
scores = Xc @ components
```

The usual mistake is `Vt[:, :k]`. The top direction is a row of `Vt`.

> [!WARNING]
> `np.linalg.svd` returns $V^{\top}$, largest singular value first. Components are rows of `Vt`.

> [!TIP]
> Two unit components can differ only by a sign. Compare `abs(u @ v)` with 1.

## Throw away the small singular values

The Eckart-Young theorem says the top $k$ terms are the best rank-$k$ approximation, and the error is exact:

$$
\sum_{\text{all entries}} (X - \hat X)^2 = \sum_{i > k} s_i^2 .
$$

The total squared error equals the sum of the squares of the singular values you dropped.

Plug in $k = 1$. You drop $s_2^2 = 4$, which matched the hand error.

Each component's share of the spread is

$$
r_i = \frac{s_i^2}{\sum_j s_j^2} .
$$

Divide $s_i^2$ by the sum of every $s_j^2$. The shares add to 1, and $n-1$ cancels.

Plug in 12 and 4. The shares are $12/16 = 0.75$ and $4/16 = 0.25$:

```python
ratio = s ** 2 / np.sum(s ** 2)
```

The usual mistake is keeping only the top $k$, so the shares no longer add to 1.

> [!KEY]
> No rank-$k$ approximation beats the top $k$ terms. The error is the sum of the dropped $s_i^2$.

## Repeat one matrix-vector product

**Power iteration** finds the top eigenvector of a symmetric matrix with only matrix-vector products. Multiply by $A$, then divide by the length:

$$
v \leftarrow \frac{A v}{\lVert A v \rVert} .
$$

Each multiply stretches direction $i$ by $\lambda_i$, so the largest eigenvalue wins. The **Rayleigh quotient** $v^{\top} A v$ equals $\lambda_1$ once $v$ is exact.

Let $A$ have rows $(2,0)$ and $(0,1)$, so $\lambda_1 = 2$, $\lambda_2 = 1$, and the top vector is $(1,0)$. Start at $(1,1)/\sqrt{2}$.

Step 1 gives $(2,1)/\sqrt{2} = (1.4142, 0.7071)$, entry ratio $1/2$. Divide by $\sqrt{2.5}$ to get $(0.8944, 0.4472)$. The Rayleigh quotient is $9/5 = 1.8$. Step 2 gives $(4,1)/\sqrt{5}$, entry ratio $1/4$, then $(0.9701, 0.2425)$ and Rayleigh quotient $33/17 = 1.9412$. Leftover error shrinks by about $|\lambda_2/\lambda_1|$ each step:

```python
v = A @ v / np.linalg.norm(A @ v)
eigenvalue = v @ A @ v
```

The usual mistake is scaling only after the loop. Entries can overflow.

> [!KEY]
> Power iteration turns a start vector towards the top eigenvector. The ratio $|\lambda_2 / \lambda_1|$ sets the speed.

## Fit a noisy line with EM

Probabilistic PCA maps a short hidden vector $z$ through a matrix $W$ and adds Gaussian noise of variance $\sigma^2$. The best $W$ spans the top $k$ components. An **expectation** is a weighted average: scores $1$ and $-1$, each half the time, have expectation 0.

Fit it by **expectation-maximisation** on centred data. On $(2,0)$ and $(-2,0)$, start with $W = (1,0)$ and $\sigma^2 = 1$. Then $M = 1+1 = 2$, and the guessed scores $X_c W / 2$ are $1$ and $-1$.

The E-step in general is

$$
E_z = X_c W M^{-1} ,
$$

$$
E_{zz} = n\,\sigma^2 M^{-1} + E_z^{\top} E_z .
$$

$M = W^{\top} W + \sigma^2 I$. The first line is the guess of $z$. The second is $n$ copies of the uncertainty, plus outer products of the guesses. Here $E_{zz} = 2\cdot 1\cdot 1/2 + 2 = 3$.

The M-step is

$$
W_{\text{new}} = X_c^{\top} E_z \, E_{zz}^{-1} .
$$

$$
\sigma^2_{\text{new}} = \frac{1}{nd}\Big( \textstyle\sum X_c^2 \;-\; 2 \sum \big(E_z \odot X_c W_{\text{new}}\big) \;+\; \operatorname{tr}\big(E_{zz} W_{\text{new}}^{\top} W_{\text{new}}\big) \Big) .
$$

$X_c^{\top} E_z = (4, 0)$, so $W_{\text{new}} = (4/3, 0)$. Squared entries sum to 8. Twice the elementwise product sums to $32/3$. The trace term is $16/3$. The numerator is $8 - 32/3 + 16/3 = 8/3$. Divide by $nd = 4$ and the noise is $2/3$:

```python
Ez = Xc @ W @ M_inv
W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz)
```

The usual mistake is the noise update from the old $W$. Both the product and the trace use $W_{\text{new}}$.

With $S = X_c^{\top} X_c / n$ and $C = WW^{\top} + \sigma^2 I$, the log-likelihood is

$$
\ell = -\frac{n}{2}\Big( d \log(2\pi) + \log\det C + \operatorname{tr}(C^{-1} S) \Big) .
$$

$d\log(2\pi)$ is constant, $\log\det C$ grows with a wide $C$, and the trace grows when the data miss $C$. Use `slogdet` and `solve`.

After the update, $\det C = 44/27$ and $\log\det C = 0.488$. The trace is $18/11 = 1.636$. Twice $\log(2\pi)$ is 3.676. The sum is $3.676 + 0.488 + 1.636 = 5.800$, so $\ell = -5.800$. Record it every iteration. It never falls.

> [!RECAP]
> Keep these five facts in hand:
> - Centre with `X.mean(axis=0)`, and add the mean back when you reconstruct.
> - Components are `Vt[:k].T`, scores are `Xc @ components`, and variances are `s[:k] ** 2 / (n - 1)`.
> - Variance ratios use every singular value. Eckart-Young error is the sum of the dropped $s_i^2$.
> - Power iteration multiplies by $A$, rescales to length 1, and reads $v^{\top} A v$.
> - PPCA updates $E_z$, $E_{zz}$, $W_{\text{new}}$, then $\sigma^2$ from $W_{\text{new}}$. The log-likelihood never falls.

## Build it step by step

Build the six functions in this order:

1. **`center(X)`** returns `(Xc, mean)`, shapes `(n, d)` and `(d,)`, from `X.mean(axis=0)`. The usual mistake is `axis=1`. Write it like this:

    ```python
    def center(X):
        # Subtract the column mean. Return Xc and that mean.
    ```

2. **`pca_svd(X, k)`** returns `(d, k)`, `(n, k)`, and `(k,)`. Centre, take `Vt[:k].T` from `svd(..., full_matrices=False)`, and set variances to `s[:k] ** 2 / (n - 1)`. The usual mistake is a column of `Vt`. Write it like this:

    ```python
    def pca_svd(X, k):
        # SVD of centred data. Scores are Xc times Vt[:k].T.
    ```

3. **`reconstruct(mean, components, scores)`** returns `(n, d)` from `mean + scores @ components.T`. The usual mistake is leaving the mean off. Write it like this:

    ```python
    def reconstruct(mean, components, scores):
        # Add the mean onto scores @ components.T.
    ```

4. **`explained_variance_ratio(X)`** returns every share, largest first, using `compute_uv=False`. The usual mistake is keeping only $k$ values. Write it like this:

    ```python
    def explained_variance_ratio(X):
        # s ** 2 / sum(s ** 2), using compute_uv=False.
    ```

5. **`power_iteration(A, iters)`** returns `(v, eigenvalue)`. Start from `np.ones(d) / np.sqrt(d)`, loop `iters` times, and finish with `v @ A @ v`. The usual mistake is scaling only at the end. Write it like this:

    ```python
    def power_iteration(A, iters):
        # Multiply by A and divide by the length, every step.
    ```

6. **`ppca_em(X, k, iters)`** returns `(W, sigma2, history)`. Seed `W` with `default_rng(0).normal(scale=0.1, size=(d, k))` and `sigma2 = 1`. Each round, E then M, append $\ell$, and clip `sigma2` at `1e-8`. The usual mistake is noise from the old `W`. Write it like this:

    ```python
    def ppca_em(X, k, iters):
        # Seed W, then Ez, Ezz, W_new, sigma2, and the log-likelihood.
    ```

## Cost and scaling

Routes for $n$ rows, $d$ features, and $k$ components line up as follows:

| Route | How the work grows | Notes |
|---|---|---|
| SVD of $X_c$ | About $n \cdot d \cdot \min(n, d)$ | Default. Most accurate. |
| Eigenvectors of $X_c^{\top} X_c$ | About $n d^2 + d^3$ | Forming the product loses precision. |
| Power iteration, $t$ steps | About $t \cdot d^2$ | Top vector only. Fast if $A$ is sparse. |
| One PPCA round | About $n d k$ | Cheap for small $k$. |

$X$ uses about $n d$ numbers. $U$ adds about $n \cdot \min(n, d)$.

## Where this breaks

Four failures show up in practice:

- Far points drag a component, because variance squares distance. Remove them, or use a robust variant.
- A spiral has no flat subspace. Kernel PCA handles curved clouds.
- If $\lambda_k \approx \lambda_{k+1}$, the kept direction can swap, and power iteration slows down.
- Grams and kilograms of the same feature get different variances. Standardise columns when units differ.

## The exercise

You compute PCA three ways. Write `center`, `pca_svd`, `reconstruct`, `explained_variance_ratio`, `power_iteration`, and `ppca_em`.

Checks: column means are zero, and adding the mean back recovers $X$. Components are orthonormal, match the covariance eigenvectors up to sign, and scores equal $X_c$ times them. Error equals $\sum_{i>k} s_i^2$. Ratios sum to 1. Power iteration tracks $|\lambda_2/\lambda_1|$. PPCA's log-likelihood never falls, and $W$ spans the PCA subspace.
