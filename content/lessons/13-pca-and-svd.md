---
title: PCA and the SVD
slug: 13-pca-and-svd
part: "Part 2 — Classical machine learning"
summary: Find the directions your data varies most along, using the SVD, power iteration, and a probabilistic version fitted by EM.
minutes: 65
gpu: false
objectives:
  - Centre data and compute principal components, scores, and explained variance from the SVD.
  - Reconstruct data from k components, and predict the reconstruction error from the singular values.
  - Build power iteration to find the top eigenvector, and explain why the eigenvalue gap sets its speed.
  - Fit probabilistic PCA with EM, and check that its log-likelihood never goes down.
prereqs:
  - 02-linear-regression
exercise: 13-pca-and-svd
---

# PCA and the SVD

So far every model had labels to learn from. Principal component analysis (PCA) works with the data alone. It asks: along which directions does the data actually spread out, and which directions are mostly noise you could throw away?

Think of a cloud of points shaped like a flat pancake floating in 3D. Almost all of the spread lies in the plane of the pancake. If you describe each point by its two coordinates in that plane, you lose very little. PCA finds that plane for you, in any number of dimensions.

The tool that does the work is the singular value decomposition (SVD), which you met when solving least squares. This lesson shows how to read PCA straight off the SVD, how to find the top direction with a simple loop called power iteration, and how a probabilistic version of PCA lands on the same answer.

## What you already know

- Any $n$-by-$d$ matrix $X$ factors as $X = U S V^{\top}$, the SVD. The columns of $U$ and $V$ are unit length and at right angles to each other, and $S$ is diagonal with nonnegative entries $s_1 \ge s_2 \ge \ldots$, the singular values.
- An eigenvector $v$ of a square matrix $A$ satisfies $Av = \lambda v$: multiplying by $A$ only stretches it, by the eigenvalue $\lambda$.
- The variance of a list of numbers is the average squared distance from their mean.
- The norm of a vector, $\lVert v \rVert$, is its length, `np.linalg.norm(v)`.

## Step 1: centre the data

PCA is about how the data spreads around its middle, so first move the middle to the origin. Subtract each column's mean from that column:

$$
X_c = X - \bar x ,
$$

where $\bar x$ is the length-$d$ vector of column means. Read it aloud: from every row, subtract the average row. Afterwards every column of $X_c$ has mean zero.

```python
mean = X.mean(axis=0)     # shape (d,)
Xc = X - mean             # broadcasting subtracts it from every row
```

Why bother? If you skip centring, the first "direction of spread" tends to point from the origin to the middle of the cloud. That tells you where the data sits, not how it varies.

## Step 2: the principal components

Picture a line through the origin. Project every centred point onto it, so each point becomes one number: its position along the line. Some lines give projections that are widely spread. Others squash everything together. The **first principal component** is the direction whose projections have the largest variance. The second is the best direction at right angles to the first, and so on.

There is a second, equivalent way to say this. Replace each point by its shadow on the line. The best line is the one that makes these shadows closest to the original points. The two definitions agree, because by Pythagoras, each point's squared length equals its squared shadow length plus its squared distance to the line. The total is fixed, so making the shadows long is the same as making the distances short.

### Reading PCA off the SVD

Take the SVD of the centred data, $X_c = U S V^{\top}$. Then:

- The **components** are the first $k$ columns of $V$. Each is a unit-length direction in feature space, with shape `(d,)`.
- The **scores** are the coordinates of each point along those directions: $X_c V_k$, with shape `(n, k)`.
- The **explained variance** of component $i$ is the variance of its scores:

$$
\lambda_i = \frac{s_i^2}{n - 1} .
$$

Read it aloud: square the $i$-th singular value and divide by $n - 1$. These $\lambda_i$ are also the eigenvalues of the sample covariance matrix $X_c^{\top} X_c / (n-1)$, and the components are its eigenvectors. The SVD gets them without ever forming that matrix, which is more accurate numerically.

```python
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
components = Vt[:k].T                  # shape (d, k), note the transpose
scores = Xc @ components               # shape (n, k)
explained_variance = s[:k] ** 2 / (n - 1)
```

`np.linalg.svd` returns $V^{\top}$, not $V$, and it sorts the singular values from largest to smallest.

## Step 3: reconstruct and measure the error

To go back from $k$ scores to the original $d$ features, walk along each component by its score and add the mean back:

$$
\hat X = \bar x + Z V_k^{\top} ,
$$

where $Z$ is the `(n, k)` score matrix. Read it aloud: each reconstructed row is the mean plus score 1 times component 1, plus score 2 times component 2, and so on.

```python
X_hat = mean + scores @ components.T   # shape (n, d)
```

How much do you lose? The Eckart–Young theorem says keeping the top $k$ SVD terms gives the best possible rank-$k$ approximation, and it tells you the error exactly:

$$
\sum_{\text{all entries}} (X - \hat X)^2 = \sum_{i > k} s_i^2 .
$$

Read it aloud: the total squared reconstruction error equals the sum of the squares of the singular values you threw away. You can predict the error before you reconstruct anything.

### Explained variance ratio

Each component's share of the total spread is its squared singular value over the sum of all of them:

$$
r_i = \frac{s_i^2}{\sum_j s_j^2} .
$$

Read it aloud: the fraction of total variance that component $i$ carries. The ratios add up to one. People plot them to decide how many components to keep. The $n - 1$ cancels, so you can skip it.

```python
s = np.linalg.svd(Xc, compute_uv=False)
ratio = s ** 2 / np.sum(s ** 2)
```

### A worked example

Take three points in 2D: $(2, 0)$, $(0, 2)$, and $(4, 4)$.

1. The mean is $(2, 2)$. The centred rows are $(0, -2)$, $(-2, 0)$, and $(2, 2)$.
2. The singular values of $X_c$ are $s_1 = \sqrt{12} \approx 3.4641$ and $s_2 = 2$. So $s_1^2 = 12$ and $s_2^2 = 4$.
3. With $n - 1 = 2$, the explained variances are $12/2 = 6$ and $4/2 = 2$. The ratios are $12/16 = 0.75$ and $4/16 = 0.25$.
4. The first component is $(1, 1)/\sqrt2 \approx (0.7071, 0.7071)$, up to sign.
5. The scores on it are $-\sqrt2$, $-\sqrt2$, and $2\sqrt2$.
6. Reconstructing with $k = 1$ gives $(1, 1)$, $(1, 1)$, and $(4, 4)$.
7. The squared errors are $1 + 1 = 2$, $1 + 1 = 2$, and $0$, for a total of $4 = s_2^2$. Eckart–Young checks out.

The SVD may flip the sign of a component. Then the scores flip too, and the reconstruction is unchanged.

## Step 4: power iteration

Sometimes you only need the top eigenvector of a symmetric matrix $A$, such as the covariance. **Power iteration** finds it with nothing but matrix-vector products. Start with any vector, multiply by $A$, rescale to length one, and repeat:

$$
v \leftarrow \frac{A v}{\lVert A v \rVert} .
$$

Read it aloud: multiply by $A$, then divide by the length so the vector does not blow up or vanish. Each multiplication stretches the part of $v$ along eigenvector $i$ by $\lambda_i$. The biggest eigenvalue wins, so $v$ turns toward the top eigenvector.

```python
v = np.ones(d) / np.sqrt(d)
for _ in range(iters):
    v = A @ v
    v = v / np.linalg.norm(v)
eigenvalue = v @ A @ v          # the Rayleigh quotient
```

The estimate of the eigenvalue is the **Rayleigh quotient** $v^{\top} A v$, which equals $\lambda_1$ when $v$ is the top eigenvector.

How fast does it converge? Each step shrinks the leftover error by about $|\lambda_2 / \lambda_1|$, the **eigenvalue gap** ratio. A ratio of 0.5 halves the error each step. A ratio of 0.99 crawls.

Worked example: let $A$ have rows $(2, 0)$ and $(0, 1)$, so $\lambda_1 = 2$, $\lambda_2 = 1$, and the top eigenvector is $(1, 0)$. Start from $(0.7071, 0.7071)$.

- Step 1: $Av = (1.4142, 0.7071)$, normalised to $(0.8944, 0.4472)$. Rayleigh quotient $1.8$.
- Step 2: $Av = (1.7889, 0.4472)$, normalised to $(0.9701, 0.2425)$. Rayleigh quotient $1.9412$.

The second entry roughly halves each step, matching the ratio $1/2$.

## Step 5: probabilistic PCA and EM

Probabilistic PCA (PPCA) tells a story about how the data was made. Each point starts as a short hidden vector $z$ of length $k$, drawn from a standard normal distribution. A $d$-by-$k$ matrix $W$ maps it into feature space, and Gaussian noise with variance $\sigma^2$ is added: $x = Wz + \text{noise}$. The best-fitting $W$ spans the same subspace as the top $k$ principal components.

You fit it with **expectation–maximisation (EM)**. EM alternates two steps. The E-step guesses the hidden $z$ for each point, given the current $W$ and $\sigma^2$. The M-step picks the $W$ and $\sigma^2$ that best explain the data, given those guesses. Every round is guaranteed not to lower the log-likelihood.

All the formulas below work on centred data $X_c$, one row per point.

**E-step.** Form the small $k$-by-$k$ matrix $M = W^{\top} W + \sigma^2 I$. The best guess of each point's hidden vector, stacked as rows, is

$$
E_z = X_c W M^{-1} ,
$$

a matrix of shape `(n, k)`. The E-step also needs the sum over points of $\mathbb{E}[z z^{\top}]$, which captures how uncertain those guesses are:

$$
E_{zz} = n\,\sigma^2 M^{-1} + E_z^{\top} E_z .
$$

Read it aloud: $n$ copies of the posterior uncertainty $\sigma^2 M^{-1}$, plus the outer products of the guesses. It has shape `(k, k)`.

**M-step.** Update $W$ by a least-squares-style solve:

$$
W_{\text{new}} = X_c^{\top} E_z \, E_{zz}^{-1} .
$$

Then update the noise variance as the average leftover error per entry:

$$
\sigma^2_{\text{new}} = \frac{1}{nd}\Big( \textstyle\sum X_c^2 \;-\; 2 \sum \big(E_z \odot X_c W_{\text{new}}\big) \;+\; \operatorname{tr}\big(E_{zz} W_{\text{new}}^{\top} W_{\text{new}}\big) \Big) .
$$

Read it aloud: take the total squared data, subtract twice the sum of the elementwise product $\odot$ of the guesses with $X_c W_{\text{new}}$, and add the trace (sum of the diagonal) of $E_{zz} W_{\text{new}}^{\top} W_{\text{new}}$. Divide by the number of entries $nd$.

```python
M_inv = np.linalg.inv(W.T @ W + sigma2 * np.eye(k))
Ez = Xc @ W @ M_inv
Ezz = n * sigma2 * M_inv + Ez.T @ Ez
W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz)
```

**Log-likelihood.** Under the model, each point is Gaussian with covariance $C = W W^{\top} + \sigma^2 I$. With $S = X_c^{\top} X_c / n$, the log-likelihood of all the data is

$$
\ell = -\frac{n}{2}\Big( d \log(2\pi) + \log\det C + \operatorname{tr}(C^{-1} S) \Big) .
$$

Read it aloud: a constant, plus a penalty for a large covariance ($\log\det C$), plus a penalty for data that does not fit it ($\operatorname{tr}(C^{-1}S)$), all scaled by $-n/2$. Compute $\log\det C$ with `np.linalg.slogdet` and $C^{-1} S$ with `np.linalg.solve(C, S)`. Record $\ell$ after every iteration. It should never go down.

## Build it step by step

The checks test the functions in this order.

1. **`center(X)`.** Takes shape `(n, d)`. Returns `(Xc, mean)` with shapes `(n, d)` and `(d,)`. Use `X.mean(axis=0)`. Common mistake: `axis=1`, which averages each row instead of each column.
2. **`pca_svd(X, k)`.** Returns `(components, scores, explained_variance)` with shapes `(d, k)`, `(n, k)`, and `(k,)`. Centre, call `np.linalg.svd(Xc, full_matrices=False)`, take `Vt[:k].T`, and use `s[:k] ** 2 / (n - 1)`. Common mistake: taking columns of `Vt` instead of rows, or running the SVD on uncentred `X`.
3. **`reconstruct(mean, components, scores)`.** Returns shape `(n, d)`: `mean + scores @ components.T`. Common mistake: forgetting to add the mean back.
4. **`explained_variance_ratio(X)`.** Returns every ratio, largest first. Centre, get the singular values with `compute_uv=False`, square, and divide by their sum. Common mistake: keeping only the top `k`, so the ratios do not sum to one.
5. **`power_iteration(A, iters)`.** Returns `(v, eigenvalue)`. Start from `np.ones(d) / np.sqrt(d)`, loop `iters` times, then return `v @ A @ v`. Common mistake: normalising only at the end, which can overflow.
6. **`ppca_em(X, k, iters)`.** Returns `(W, sigma2, history)`. Centre, set `W = np.random.default_rng(0).normal(scale=0.1, size=(d, k))` and `sigma2 = 1.0`, then loop: E-step, M-step, and append the log-likelihood under the new parameters. Clip `sigma2` to at least `1e-8`. Common mistake: computing the M-step for `sigma2` with the old `W` instead of `W_new`.

## Cost and scaling

The following table compares routes to the principal components, for $n$ rows, $d$ features, and $k$ components.

| Route | How the work grows | Notes |
|---|---|---|
| SVD of $X_c$ | About $n \cdot d \cdot \min(n, d)$ | Most accurate; the default. |
| Eigenvectors of $X_c^{\top} X_c$ | About $n d^2 + d^3$ | Forming $X_c^{\top} X_c$ loses precision. |
| Power iteration, $t$ steps | About $t \cdot d^2$ | Only the top vector; fast for sparse $A$. |
| One PPCA EM round | About $n d k$ | Cheap when $k$ is much smaller than $d$. |

Memory is dominated by $X$ itself, `n * d` numbers. The SVD also needs space for $U$, about `n * min(n, d)` numbers.

## Where this breaks

- **Outliers.** Variance squares distances, so a few far-away points can drag a component toward themselves. Remove outliers first, or use a robust variant.
- **Curved data.** A spiral or a sphere has no good flat subspace. Kernel PCA and nonlinear methods handle these.
- **Near-equal eigenvalues.** When $\lambda_k \approx \lambda_{k+1}$, a small change in the data can swap which direction is kept. Power iteration also becomes very slow.
- **Different units.** A feature measured in grams has a bigger variance than the same feature in kilograms, and PCA will favour it. Standardise each feature to unit variance first when units differ.

## The exercise

You compute PCA three ways and check that they agree. Write `center` and `pca_svd`, then `reconstruct` and `explained_variance_ratio`. Then write `power_iteration`, which finds the top eigenvector of a symmetric matrix, and `ppca_em`, which fits probabilistic PCA by EM and returns the log-likelihood after every iteration.

The checks confirm that `center` gives zero column means, that `pca_svd` returns orthonormal components with `scores == Xc @ components` and matches an eigendecomposition of the covariance up to sign, that `reconstruct` hits the Eckart–Young error $\sum_{i>k} s_i^2$, and that `explained_variance_ratio` sums to one. They check that `power_iteration` converges at the rate the eigenvalue gap predicts, that the log-likelihood from `ppca_em` never decreases, and that its fitted `W` spans the PCA subspace.
