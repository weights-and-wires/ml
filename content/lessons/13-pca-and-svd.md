---
title: PCA and the SVD
slug: 13-pca-and-svd
part: "Part 2 — Classical machine learning"
summary: Maximum-variance projection and minimum reconstruction error proved equivalent, computed by the SVD, and bounded by the Eckart-Young theorem.
minutes: 55
gpu: false
objectives:
  - Prove that maximum-variance projection and minimum reconstruction error select the same subspace.
  - Identify the eigendecomposition of the sample covariance with the SVD of the centred data matrix.
  - State the Eckart-Young-Mirsky theorem and use it to predict reconstruction error exactly.
  - Derive the EM updates for probabilistic PCA and show the maximum likelihood solution recovers the PCA subspace.
prereqs:
  - 02-linear-regression
exercise: 13-pca-and-svd
---

# PCA and the SVD

Principal component analysis answers a question none of the supervised
methods so far can ask: given only $X$, with no labels, which directions in
feature space carry the data's structure, and which are noise you can discard
without losing much? The answer turns out to be the same subspace whether you
pose the question as "which projection keeps the most variance" or "which
projection loses the least information," and that subspace is exactly the one
the singular value decomposition already computes — the factorisation lesson
2 used to solve least squares stably is, applied to the data matrix itself,
principal component analysis.

This lesson proves the two formulations of PCA coincide, ties the resulting
eigenproblem to the SVD with an exact conversion between singular values and
eigenvalues, states the theorem that makes truncation provably optimal, and
extends the model to a probabilistic version whose maximum likelihood
estimate reproduces the same subspace from an entirely different starting
point.

## What you already know

- Every real $n \times d$ matrix has an SVD $X = U\Sigma V^{\top}$, with $U$
  and $V$ having orthonormal columns and $\Sigma$ diagonal and nonnegative.
- A symmetric matrix has a real eigendecomposition with orthonormal
  eigenvectors, by the spectral theorem.
- The trace of a matrix equals the sum of its eigenvalues, and is invariant
  under conjugation by an orthogonal matrix.
- Maximising a quadratic form over the unit sphere, $\max_{\lVert w \rVert = 1} w^{\top}Aw$
  for symmetric $A$, is solved by the top eigenvector of $A$, by the Lagrange
  condition $Aw = \lambda w$.

## Two definitions, one subspace

Take centred data $x_1, \ldots, x_n \in \mathbb{R}^d$ (mean zero; centring is
addressed below), with sample covariance
$\hat{\Sigma} = \frac{1}{n-1}\sum_i x_i x_i^{\top}$. PCA looks for a
$k$-dimensional subspace, represented by a matrix $V \in \mathbb{R}^{d \times k}$
with orthonormal columns ($V^{\top}V = I_k$), by one of two criteria.

**Maximum variance.** Choose $V$ to maximise the variance of the projected
data, $\frac{1}{n-1}\sum_i \lVert V^{\top}x_i \rVert^2 = \operatorname{tr}(V^{\top}\hat{\Sigma}V)$.

**Minimum reconstruction error.** Choose $V$ to minimise the squared error of
approximating each $x_i$ by its projection back out of the subspace,
$\sum_i \lVert x_i - VV^{\top}x_i \rVert^2$.

These look like different objectives, but for any orthonormal $V$, Pythagoras
gives $\lVert x_i \rVert^2 = \lVert V^{\top}x_i \rVert^2 + \lVert x_i - VV^{\top}x_i \rVert^2$,
because $VV^{\top}x_i$ is the orthogonal projection of $x_i$ onto
$\operatorname{range}(V)$. Summing over $i$,

$$
\sum_i \lVert x_i - VV^{\top}x_i \rVert^2 = \sum_i \lVert x_i \rVert^2 - \sum_i \lVert V^{\top}x_i \rVert^2 ,
$$

and the first term on the right does not depend on $V$. Minimising
reconstruction error and maximising projected variance are the same
optimisation problem, differing only by the constant $\sum_i \lVert x_i \rVert^2$.
Whichever way you motivate PCA, it selects the same $V$.

### Solving the optimisation

Maximising $\operatorname{tr}(V^{\top}\hat{\Sigma}V)$ over orthonormal
$V \in \mathbb{R}^{d \times k}$ is solved by taking the columns of $V$ to be
the top $k$ eigenvectors of $\hat{\Sigma}$. For $k = 1$ this is exactly the
quadratic-form maximisation you already know, giving the top eigenvector
$v_1$ with $\hat{\Sigma}v_1 = \lambda_1 v_1$. For general $k$, Ky Fan's
theorem extends this: the maximum of $\operatorname{tr}(V^{\top}\hat{\Sigma}V)$
over orthonormal $V \in \mathbb{R}^{d \times k}$ equals
$\lambda_1 + \cdots + \lambda_k$, the sum of the $k$ largest eigenvalues,
achieved exactly when $\operatorname{range}(V)$ is the span of the
corresponding eigenvectors. The **principal components** are these
eigenvectors, and the **scores** are the projected coordinates
$X V \in \mathbb{R}^{n \times k}$.

## The SVD does this without forming the covariance

Let $X \in \mathbb{R}^{n \times d}$ be the centred data matrix (row $i$ is
$x_i^{\top}$), with SVD $X = U\Sigma V^{\top}$. Then

$$
\hat{\Sigma} = \frac{X^{\top}X}{n-1} = \frac{V\Sigma^{\top}\Sigma V^{\top}}{n-1} = V \left(\frac{\Sigma^{\top}\Sigma}{n-1}\right) V^{\top} ,
$$

which is exactly an eigendecomposition of $\hat{\Sigma}$: the right singular
vectors of $X$ are the eigenvectors of the sample covariance, and the
eigenvalues satisfy

$$
\lambda_i = \frac{\sigma_i^2}{n-1} , \qquad \sigma_i^2 = (n-1)\lambda_i .
$$

Computing PCA through the SVD of $X$ rather than the eigendecomposition of
$X^{\top}X$ is the same numerical argument as lesson 2's case against the
normal equations: forming $X^{\top}X$ squares the condition number of $X$
before you have done anything with it, and the SVD route never forms that
product.

### The Eckart-Young-Mirsky theorem

Truncating the SVD to its top $k$ terms, $X_k = U_{:,:k}\Sigma_{:k,:k}V_{:,:k}^{\top}$,
gives the best rank-$k$ approximation of $X$ in both the Frobenius and
spectral norms — the **Eckart-Young-Mirsky theorem**:

$$
X_k = \operatorname*{arg\,min}_{\operatorname{rank}(Y) \leq k} \lVert X - Y \rVert_F ,
\qquad
\lVert X - X_k \rVert_F^2 = \sum_{i > k} \sigma_i^2 .
$$

The reconstruction $VV^{\top}x_i$ from the previous section is exactly the
row-$i$ projection realising $X_k$, since $XV V^{\top}$ with $V$ the top-$k$
right singular vectors equals $U_{:,:k}\Sigma_{:k,:k}V_{:,:k}^{\top}$. The
theorem turns the reconstruction error bound into an equality you can predict
without touching the data twice: sum the squares of the singular values you
discarded, and that is the total squared reconstruction error you incur,
exactly, not merely as an upper bound.

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

Decompose the uncentred second moment matrix of raw data $z_1, \ldots, z_n$
with mean $\bar{z}$. Writing $z_i = (z_i - \bar{z}) + \bar{z}$ and expanding,

$$
\sum_i z_i z_i^{\top} = \sum_i (z_i - \bar{z})(z_i - \bar{z})^{\top} + n\bar{z}\bar{z}^{\top} ,
$$

because the cross terms $\sum_i (z_i - \bar{z})\bar{z}^{\top}$ vanish: they
factor as $\bar{z}^{\top}$ times $\sum_i (z_i - \bar{z}) = 0$. The uncentred
second moment matrix is the covariance plus a rank-one term
$n\bar{z}\bar{z}^{\top}$ pointing along the mean. Whenever
$\lVert \bar{z} \rVert$ is large relative to the spread of the data, that
rank-one term dominates the spectrum, and the top "principal component" of
uncentred data is approximately the mean direction itself — a fact about
where the data sits, not about how it varies. Centring subtracts exactly that
term before the eigenproblem is solved.

### Power iteration

Given a symmetric $A$ with eigenvalues $\lvert \lambda_1 \rvert > \lvert \lambda_2 \rvert \geq \cdots$
and eigenvectors $u_1, u_2, \ldots$, power iteration repeats
$v \leftarrow Av / \lVert Av \rVert$ from an arbitrary starting vector
$v_0 = \sum_i c_i u_i$ with $c_1 \neq 0$. After $t$ steps, before
renormalising,

$$
A^t v_0 = \sum_i c_i \lambda_i^t u_i = \lambda_1^t \left( c_1 u_1 + \sum_{i \geq 2} c_i \left(\frac{\lambda_i}{\lambda_1}\right)^{t} u_i \right) .
$$

The component along every eigenvector but the first shrinks geometrically
relative to $c_1 u_1$, at rate $\lvert \lambda_2 / \lambda_1 \rvert$ — the
**eigenvalue gap**. A large gap gives fast convergence to $u_1$; a gap near
one gives a rate near one and a power iteration that crawls. Extracting
further components by deflation (subtracting the found component's
contribution and repeating) inherits the same rate against the next gap,
$\lambda_3 / \lambda_2$, and so on.

### Probabilistic PCA

Model each point as $x = Wz + \mu + \epsilon$, with latent
$z \sim \mathcal{N}(0, I_k)$, noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$,
and $W \in \mathbb{R}^{d \times k}$. Marginalising $z$ gives
$x \sim \mathcal{N}(\mu, C)$ with $C = WW^{\top} + \sigma^2 I_d$, a Gaussian
model whose log-likelihood over centred data ($\mu = 0$ without loss of
generality) is

$$
\ell(W, \sigma^2) = -\frac{n}{2}\left( d\log(2\pi) + \log\lvert C \rvert + \operatorname{tr}(C^{-1}S) \right) ,
\qquad S = \frac{1}{n}\sum_i x_i x_i^{\top} .
$$

Direct maximisation is available in closed form, but the EM route generalises
to missing data and is the one worth deriving. The complete-data
log-likelihood, given $z_i$, is quadratic in $W$ and $z_i$, so both EM steps
are closed form.

**E-step.** With $M = W^{\top}W + \sigma^2 I_k$, the posterior over the
latent variable is Gaussian:

$$
\mathbb{E}[z_i] = M^{-1}W^{\top}x_i , \qquad
\mathbb{E}[z_iz_i^{\top}] = \sigma^2 M^{-1} + \mathbb{E}[z_i]\mathbb{E}[z_i]^{\top} .
$$

**M-step.** Maximising the expected complete-data log-likelihood over $W$ and
$\sigma^2$ gives

$$
W_{\text{new}} = \left( \sum_i x_i \mathbb{E}[z_i]^{\top} \right) \left( \sum_i \mathbb{E}[z_iz_i^{\top}] \right)^{-1} ,
$$

$$
\sigma^2_{\text{new}} = \frac{1}{nd} \sum_i \left( \lVert x_i \rVert^2 - 2\mathbb{E}[z_i]^{\top}W_{\text{new}}^{\top}x_i + \operatorname{tr}\!\big(\mathbb{E}[z_iz_i^{\top}]\, W_{\text{new}}^{\top}W_{\text{new}}\big) \right) .
$$

Each M-step maximises the same lower bound the E-step made tight, so the
marginal log-likelihood $\ell(W, \sigma^2)$ is nondecreasing across
iterations — the general guarantee behind every EM algorithm, not a
coincidence of this particular model.

Tipping and Bishop's result identifies the stationary points of $\ell$: at
any maximum, $W = U_k(\Lambda_k - \sigma^2 I_k)^{1/2}R$ for an arbitrary
orthogonal $R$, where $U_k$ and $\Lambda_k$ are the top-$k$ eigenvectors and
eigenvalues of $S$, and the maximum likelihood noise variance is the average
of the discarded eigenvalues, $\sigma^2_{\text{ML}} = \frac{1}{d-k}\sum_{i>k}\lambda_i$.
The column space of $W$ is the PCA subspace at every such stationary point,
regardless of $\sigma^2$; as $\sigma^2 \to 0$, the loading scale
$(\Lambda_k - \sigma^2 I_k)^{1/2} \to \Lambda_k^{1/2}$ and the model's
reconstruction converges to ordinary PCA's, recovering the deterministic
method as the zero-noise limit of the probabilistic one.

### Whitening

Standardising each principal component to unit variance,

$$
x \mapsto \Lambda^{-1/2}V^{\top}(x - \mu) ,
$$

with $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_d)$ and $V$ the
full eigenvector matrix, produces data with identity covariance: every
direction carries the same scale, and any subsequent distance-based or
gradient-based method stops being sensitive to the units the original
features happened to use. Whitening only the top $k$ components after
truncating discards the noise variance along the remaining directions rather
than rescaling it up to match the rest.

## Cost and scaling

| Route | Flops | Notes |
|---|---|---|
| Eigendecomposition of $X^{\top}X$ | $O(nd^2 + d^3)$ | Forms the Gram matrix; squares $\kappa_2(X)$ as in lesson 2. |
| Thin SVD of $X$ | $O(nd \min(n, d))$ | Never forms $X^{\top}X$; the numerically preferred route. |
| Power iteration, one eigenvector, $t$ steps | $O(td^2)$ dense, $O(t \cdot \text{nnz})$ sparse | Avoids factorising $A$ at all; pays per iteration instead. |
| One EM iteration for probabilistic PCA | $O(ndk + nk^2 + k^3)$ | $k^2$ and $k^3$ terms come from forming and inverting $M$. |

When $k \ll d \ll n$, probabilistic PCA's EM iteration is far cheaper than a
full SVD of $X$, which is why it displaces the exact method on large,
high-dimensional data where only a handful of components are wanted.

## Where this breaks

**Outliers dominate the variance objective.** PCA maximises variance, and
variance is quadratic in each point's distance from the mean, so a small
number of outliers can rotate a principal component toward themselves.
Robust variants replace the squared reconstruction error with one that grows
more slowly.

**Nonlinear structure has no good linear subspace.** Data lying on a curved
manifold — a spiral, a sphere — has no low-dimensional linear projection that
preserves its structure, however many components you keep; kernel PCA and
nonlinear embeddings exist precisely to relax the "subspace" part of the
definition.

**Close eigenvalues make the subspace, not just the iteration, unstable.**
When $\lambda_k \approx \lambda_{k+1}$, the top-$k$ subspace is barely better
supported by the data than the top-$(k-1)$ or top-$(k+1)$ subspace, and small
perturbations of the sample swap which eigenvector lands on which side of the
cut, even though the subspace spanned by the tied group stays stable.

**Unscaled features bias the projection before it starts.** A feature
measured in different units carries a different variance for reasons that
have nothing to do with importance, and PCA cannot distinguish "large
variance" from "large scale." Standardising each feature to unit variance
before fitting is the usual response, at the cost of changing what "the
principal directions" mean.

## The exercise

You implement PCA three ways and connect them. Write `center`, which returns
centred data and the mean removed, and `pca_svd(X, k)`, which returns the top
$k$ components, scores, and explained variance through the SVD of the
centred data — not through an eigendecomposition of the covariance. Write
`reconstruct`, which inverts the projection back into the original space, and
`explained_variance_ratio`, the full-rank normalised spectrum.

Then write `power_iteration(A, iters)`, which finds the top eigenvector of a
symmetric matrix without factorising it, and `ppca_em(X, k, iters)`, which
fits probabilistic PCA by the EM updates derived above and returns the
log-likelihood at every iteration alongside the fitted parameters.

The checks confirm that `pca_svd`'s components and explained variance agree
with a direct eigendecomposition of the sample covariance up to sign; that
`reconstruct` applied to a truncated `pca_svd` achieves reconstruction error
matching the Eckart-Young bound $\sum_{i>k}\sigma_i^2$ exactly; that
`explained_variance_ratio` sums to one across the full rank; that
`power_iteration`'s convergence rate to the top eigenvector tracks the
eigenvalue gap the analysis above predicts; and that `ppca_em`'s
log-likelihood never decreases from one EM iteration to the next.
