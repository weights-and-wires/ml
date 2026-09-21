---
title: PCA and the SVD
slug: 13-pca-and-svd
part: "Part 2 — Classical machine learning"
summary: Maximum-variance projection, minimum reconstruction error, and probabilistic PCA are the same subspace, computed by the SVD.
minutes: 45
gpu: false
objectives:
  - State the two definitions of PCA and explain why they select the same subspace.
  - Compute PCA from the SVD of the centred data matrix, and convert singular values to eigenvalues.
  - Use the Eckart-Young-Mirsky theorem to predict reconstruction error without touching the data twice.
  - State the EM updates for probabilistic PCA and explain why the maximum likelihood solution recovers the PCA subspace.
prereqs:
  - 02-linear-regression
exercise: 13-pca-and-svd
---

# PCA and the SVD

Principal component analysis answers a question none of the supervised
methods so far can ask. Given only $X$, with no labels, which directions in
feature space carry the data's structure, and which are noise you can
discard? You can pose that question two ways: which projection keeps the
most variance, or which projection loses the least information. Both
questions have the same answer, and that answer is a subspace the singular
value decomposition already computes. The factorisation lesson 2 used to
solve least squares stably is, applied to the data matrix itself, principal
component analysis.

This lesson shows why the two definitions of PCA agree, ties the resulting
eigenproblem to the SVD, states the theorem that makes truncation provably
optimal, and extends the model to a probabilistic version that lands on the
same subspace from a different starting point.

## What you already know

- Every real $n \times d$ matrix has an SVD $X = U\Sigma V^{\top}$, with $U$
  and $V$ having orthonormal columns and $\Sigma$ diagonal and nonnegative.
- A symmetric matrix has a real eigendecomposition with orthonormal
  eigenvectors, by the spectral theorem.
- The trace of a matrix equals the sum of its eigenvalues, and is invariant
  under conjugation by an orthogonal matrix.
- Maximising a quadratic form over the unit sphere, $\max_{\lVert w \rVert = 1} w^{\top}Aw$
  for symmetric $A$, is solved by the top eigenvector of $A$.

## Two definitions, one subspace

Take centred data $x_1, \ldots, x_n \in \mathbb{R}^d$ (mean zero; centring is
addressed below), with sample covariance
$\hat{\Sigma} = \frac{1}{n-1}\sum_i x_i x_i^{\top}$. PCA looks for a
$k$-dimensional subspace, represented by a matrix $V \in \mathbb{R}^{d \times k}$
with orthonormal columns ($V^{\top}V = I_k$).

**Maximum variance** chooses $V$ to maximise the variance of the projected
data, $\frac{1}{n-1}\sum_i \lVert V^{\top}x_i \rVert^2 = \operatorname{tr}(V^{\top}\hat{\Sigma}V)$.
**Minimum reconstruction error** chooses $V$ to minimise the squared error of
approximating each $x_i$ by its projection back out of the subspace,
$\sum_i \lVert x_i - VV^{\top}x_i \rVert^2$.

These look like different objectives. They are the same one. Pythagoras
gives $\lVert x_i \rVert^2 = \lVert V^{\top}x_i \rVert^2 + \lVert x_i - VV^{\top}x_i \rVert^2$
for any orthonormal $V$, because $VV^{\top}x_i$ is the orthogonal projection
of $x_i$ onto $\operatorname{range}(V)$. Sum over $i$ and rearrange:

$$
\sum_i \lVert x_i - VV^{\top}x_i \rVert^2 = \sum_i \lVert x_i \rVert^2 - \sum_i \lVert V^{\top}x_i \rVert^2 .
$$

The first term on the right does not depend on $V$. Minimising reconstruction
error and maximising projected variance differ only by that constant, so
they pick the same $V$.

### Solving the optimisation

Maximising $\operatorname{tr}(V^{\top}\hat{\Sigma}V)$ over orthonormal
$V \in \mathbb{R}^{d \times k}$ is solved by taking the columns of $V$ to be
the top $k$ eigenvectors of $\hat{\Sigma}$. For $k=1$ this is the
quadratic-form maximisation you already know, giving the top eigenvector
$v_1$ with $\hat{\Sigma}v_1 = \lambda_1 v_1$. Ky Fan's theorem extends this
to general $k$: the maximum equals $\lambda_1 + \cdots + \lambda_k$,
achieved when $\operatorname{range}(V)$ is the span of the top $k$
eigenvectors. Those eigenvectors are the **principal components**, and
$XV \in \mathbb{R}^{n \times k}$ gives the **scores**.

## The SVD does this without forming the covariance

Let $X \in \mathbb{R}^{n \times d}$ be the centred data matrix (row $i$ is
$x_i^{\top}$), with SVD $X = U\Sigma V^{\top}$. Then

$$
\hat{\Sigma} = \frac{X^{\top}X}{n-1} = \frac{V\Sigma^{\top}\Sigma V^{\top}}{n-1} = V \left(\frac{\Sigma^{\top}\Sigma}{n-1}\right) V^{\top} ,
$$

which is an eigendecomposition of $\hat{\Sigma}$: the right singular vectors
of $X$ are the eigenvectors of the sample covariance, and the eigenvalues
satisfy

$$
\lambda_i = \frac{\sigma_i^2}{n-1} , \qquad \sigma_i^2 = (n-1)\lambda_i .
$$

Computing PCA through the SVD of $X$, rather than the eigendecomposition of
$X^{\top}X$, avoids the same trap as lesson 2's case against the normal
equations: forming $X^{\top}X$ squares the condition number of $X$ before
you compute anything with it.

### The Eckart-Young-Mirsky theorem

Truncating the SVD to its top $k$ terms, $X_k = U_{:,:k}\Sigma_{:k,:k}V_{:,:k}^{\top}$,
gives the best rank-$k$ approximation of $X$ in both the Frobenius and
spectral norms:

$$
X_k = \operatorname*{arg\,min}_{\operatorname{rank}(Y) \leq k} \lVert X - Y \rVert_F ,
\qquad
\lVert X - X_k \rVert_F^2 = \sum_{i > k} \sigma_i^2 .
$$

The reconstruction $VV^{\top}x_i$ from the previous section realises $X_k$
row by row. The theorem turns the reconstruction-error bound into an
equality you can predict without touching the data twice: sum the squares
of the discarded singular values, and that is the total squared
reconstruction error, exactly.

### Explained variance

The fraction of total variance kept by the top $k$ components is

$$
\frac{\lambda_1 + \cdots + \lambda_k}{\lambda_1 + \cdots + \lambda_d} = \frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_r^2} ,
$$

with $r = \operatorname{rank}(X)$. These **explained variance ratios** sum to
one across the full rank by construction, and reporting them alongside a
chosen $k$ is the standard way to justify how much structure a truncation
keeps.

### Why centring matters

Raw, uncentred data $z_1, \ldots, z_n$ with mean $\bar{z}$ has second-moment
matrix

$$
\sum_i z_i z_i^{\top} = \sum_i (z_i - \bar{z})(z_i - \bar{z})^{\top} + n\bar{z}\bar{z}^{\top} ,
$$

because the cross terms cancel. The uncentred second-moment matrix is the
covariance plus a rank-one term pointing along the mean. When
$\lVert \bar{z} \rVert$ is large relative to the data's spread, that term
dominates the spectrum, and the top "principal component" of uncentred data
becomes the mean direction itself — a fact about where the data sits, not
about how it varies. Centring removes that term before you solve the
eigenproblem.

### Power iteration

Given a symmetric $A$ with eigenvalues $\lvert \lambda_1 \rvert > \lvert \lambda_2 \rvert \geq \cdots$
and eigenvectors $u_1, u_2, \ldots$, power iteration repeats
$v \leftarrow Av / \lVert Av \rVert$ from an arbitrary starting vector
$v_0 = \sum_i c_i u_i$ with $c_1 \neq 0$. Each unnormalised step scales every
eigenvector's coefficient by its own $\lambda_i$, so after $t$ steps the
component along every eigenvector but the first shrinks relative to $c_1
u_1$ at rate $\lvert \lambda_2/\lambda_1 \rvert^t$ — the **eigenvalue gap**
raised to the iteration count. A large gap gives fast convergence to $u_1$;
a gap near one gives a rate near one and an iteration that crawls.
Deflation, which subtracts the found component and repeats, extracts
further components at the same kind of rate against the next gap,
$\lambda_3/\lambda_2$.

### Probabilistic PCA

Model each point as $x = Wz + \mu + \epsilon$, with latent
$z \sim \mathcal{N}(0, I_k)$, noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$,
and $W \in \mathbb{R}^{d \times k}$. Marginalising $z$ gives
$x \sim \mathcal{N}(\mu, C)$ with $C = WW^{\top} + \sigma^2 I_d$. Over
centred data ($\mu = 0$ without loss of generality), the log-likelihood is

$$
\ell(W, \sigma^2) = -\frac{n}{2}\left( d\log(2\pi) + \log\lvert C \rvert + \operatorname{tr}(C^{-1}S) \right) ,
\qquad S = \frac{1}{n}\sum_i x_i x_i^{\top} .
$$

A closed-form maximiser exists, but the EM route generalises to missing
data. Both steps are closed form because the complete-data log-likelihood,
given $z_i$, is quadratic in $W$ and $z_i$.

**E-step.** With $M = W^{\top}W + \sigma^2 I_k$, the latent posterior is
Gaussian:

$$
\mathbb{E}[z_i] = M^{-1}W^{\top}x_i , \qquad
\mathbb{E}[z_iz_i^{\top}] = \sigma^2 M^{-1} + \mathbb{E}[z_i]\mathbb{E}[z_i]^{\top} .
$$

**M-step.** Maximising the expected complete-data log-likelihood over $W$
and $\sigma^2$ gives

$$
W_{\text{new}} = \left( \sum_i x_i \mathbb{E}[z_i]^{\top} \right) \left( \sum_i \mathbb{E}[z_iz_i^{\top}] \right)^{-1} ,
$$

$$
\sigma^2_{\text{new}} = \frac{1}{nd} \sum_i \left( \lVert x_i \rVert^2 - 2\mathbb{E}[z_i]^{\top}W_{\text{new}}^{\top}x_i + \operatorname{tr}\!\big(\mathbb{E}[z_iz_i^{\top}]\, W_{\text{new}}^{\top}W_{\text{new}}\big) \right) .
$$

Each M-step maximises the same lower bound the E-step made tight, so the
marginal log-likelihood is nondecreasing across iterations. That guarantee
holds for every EM algorithm, not only this one.

Tipping and Bishop showed that at any stationary point,
$W = U_k(\Lambda_k - \sigma^2 I_k)^{1/2}R$ for an arbitrary orthogonal $R$,
where $U_k$ and $\Lambda_k$ are the top-$k$ eigenvectors and eigenvalues of
$S$, and the maximum likelihood noise variance is the average of the
discarded eigenvalues, $\sigma^2_{\text{ML}} = \frac{1}{d-k}\sum_{i>k}\lambda_i$.
The column space of $W$ is the PCA subspace at every such stationary point,
regardless of $\sigma^2$, and as $\sigma^2 \to 0$ the model's reconstruction
converges to ordinary PCA's.

### Whitening

Standardising each principal component to unit variance,
$x \mapsto \Lambda^{-1/2}V^{\top}(x - \mu)$ with $\Lambda$ the full
eigenvalue diagonal and $V$ the full eigenvector matrix, produces data with
identity covariance. Every direction carries the same scale, so a
distance-based or gradient-based method downstream stops responding to
units the original features happened to use.

## Cost and scaling

The following table compares routes to the same subspace.

| Route | Flops | Notes |
|---|---|---|
| Eigendecomposition of $X^{\top}X$ | $O(nd^2 + d^3)$ | Forms the Gram matrix; squares $\kappa_2(X)$ as in lesson 2. |
| Thin SVD of $X$ | $O(nd \min(n, d))$ | Never forms $X^{\top}X$; the numerically preferred route. |
| Power iteration, one eigenvector, $t$ steps | $O(td^2)$ dense, $O(t \cdot \text{nnz})$ sparse | Avoids factorising $A$ at all; pays per iteration instead. |
| One EM iteration for probabilistic PCA | $O(ndk + nk^2 + k^3)$ | The $k^2$ and $k^3$ terms come from forming and inverting $M$. |

When $k \ll d \ll n$, probabilistic PCA's EM iteration costs far less than a
full SVD of $X$, which is why it replaces the exact method on large,
high-dimensional data with only a handful of components to find.

## Where this breaks

**Outliers dominate the variance objective.** PCA maximises variance, and
variance is quadratic in each point's distance from the mean, so a small
number of outliers can rotate a principal component toward themselves.
Robust variants replace the squared reconstruction error with one that grows
more slowly.

**Nonlinear structure has no good linear subspace.** Data lying on a curved
manifold, such as a spiral or a sphere, has no low-dimensional linear
projection that preserves its structure. Kernel PCA and nonlinear
embeddings relax the "subspace" part of the definition to handle this case.

**Close eigenvalues make the subspace itself unstable, not only the iteration.**
When $\lambda_k \approx \lambda_{k+1}$, the top-$k$ subspace is barely better
supported by the data than the top-$(k-1)$ or top-$(k+1)$ subspace, and
small perturbations of the sample swap which eigenvector lands on which
side of the cut, even though the subspace spanned by the tied group stays
stable.

**Unscaled features bias the projection before it starts.** A feature
measured in different units carries a different variance for reasons
unrelated to importance, and PCA cannot distinguish large variance from
large scale. Standardising each feature to unit variance before fitting is
the usual response, at the cost of changing what "the principal directions"
mean.

## The exercise

You implement PCA three ways and connect them. Write `center`, which returns
centred data and the mean removed, and `pca_svd(X, k)`, which returns the top
$k$ components, scores, and explained variance through the SVD of the
centred data, not through an eigendecomposition of the covariance. Write
`reconstruct`, which inverts the projection back into the original space,
and `explained_variance_ratio`, the full-rank normalised spectrum.

Then write `power_iteration(A, iters)`, which finds the top eigenvector of a
symmetric matrix without factorising it, and `ppca_em(X, k, iters)`, which
fits probabilistic PCA with the EM updates derived above and returns the
log-likelihood at every iteration alongside the fitted parameters.

The checks confirm that `pca_svd`'s components and explained variance agree
with a direct eigendecomposition of the sample covariance up to sign, that
`reconstruct` applied to a truncated `pca_svd` matches the Eckart-Young
bound $\sum_{i>k}\sigma_i^2$ exactly, that `explained_variance_ratio` sums
to one across the full rank, that `power_iteration`'s convergence rate to
the top eigenvector tracks the eigenvalue gap, and that `ppca_em`'s
log-likelihood never decreases from one iteration to the next.
