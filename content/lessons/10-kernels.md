---
title: Kernels and the representer theorem
slug: 10-kernels
part: "Part 2 — Classical machine learning"
summary: What a valid kernel is, why its optimum always lives in the span of the training data, and how to escape the O(n^2) wall that fact leaves behind.
minutes: 60
gpu: false
objectives:
  - State Mercer's condition and show that every kernel of the form <phi(x),phi(x')> gives a positive semidefinite Gram matrix, and conversely.
  - Construct the reproducing kernel Hilbert space a kernel induces and derive the reproducing property.
  - State the representer theorem's hypotheses and prove it by an orthogonal decomposition.
  - Derive kernel ridge regression's closed-form solution and show it recovers primal ridge regression exactly under a linear kernel.
  - Derive random Fourier features from Bochner's theorem and their O(1/sqrt(D)) concentration rate.
prereqs:
  - 09-support-vector-machines
  - 04-regularisation
exercise: 10-kernels
---

# Kernels and the representer theorem

The support vector machine's dual depends on the training data only through
inner products, $x_i^{\top}x_j$ for training pairs and $x^{\top}x_i$ at
prediction time. Nothing in the derivation used the fact that these were
literal Euclidean inner products of the raw features — only that they were
*some* symmetric, positive semidefinite measure of similarity between two
points. That observation generalises far past the SVM: any method whose
fitted solution can be written using training points solely through such
inner products can be handed a similarity measure it never has to unpack into
coordinates, and gains, for free, access to feature spaces too large to
write down.

This lesson makes that generalisation precise in two directions. First, what
exactly is a valid measure of similarity — a kernel — and what Hilbert space
does it implicitly define. Second, and more surprisingly, why an enormous
class of learning problems — not just SVMs, but ridge regression and every
other method built from a loss plus a norm penalty — always has a solution
expressible using only the training points, regardless of how large or
infinite-dimensional the space the kernel implies. That fact is the
representer theorem, and it is the reason the kernel trick is a method and
not a coincidence specific to margins.

## What you already know

- The SVM dual and the SVM's prediction rule depend on the data only through
  $x^{\top}x'$, never on an individual $x$.
- Ridge regression's closed form $\hat w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$,
  and that it minimises squared loss plus an $\ell_2$ penalty on $w$.
- A symmetric matrix is positive semidefinite exactly when every eigenvalue is
  non-negative, equivalently when $c^{\top}Mc \ge 0$ for every vector $c$.
- The Cauchy-Schwarz inequality, $|\langle f,g\rangle| \le \lVert f\rVert \lVert g \rVert$,
  and orthogonal decomposition: for a subspace $V$, every $f$ splits uniquely
  as $f = f_\parallel + f_\perp$ with $f_\parallel \in V$ and $f_\perp \perp V$,
  and $\lVert f \rVert^2 = \lVert f_\parallel \rVert^2 + \lVert f_\perp \rVert^2$.
- A concentration inequality (Hoeffding's, say) bounds how far the average of
  bounded i.i.d. random variables can stray from its mean, with the bound
  shrinking as $O(1/\sqrt{D})$ in the number of samples $D$.

## Positive semidefinite kernels and Mercer's condition

A **kernel** is a symmetric function $k : \mathcal X \times \mathcal X \to
\mathbb R$. Call it **positive semidefinite (PSD)** if, for every finite set
of points $x_1,\ldots,x_n \in \mathcal X$, the Gram matrix $K_{ij} =
k(x_i,x_j)$ is PSD as a matrix — the property that makes $k$ usable as a
similarity measure a learning algorithm can trust.

**PSD is necessary for $k$ to be an inner product of features.** Suppose
$k(x,x') = \langle \phi(x), \phi(x')\rangle$ for some map $\phi$ into an inner
product space. For any $c \in \mathbb R^n$ and any points $x_1,\ldots,x_n$,

$$
c^{\top}Kc = \sum_{i,j} c_ic_j \langle \phi(x_i),\phi(x_j)\rangle
= \Big\langle \sum_i c_i\phi(x_i),\, \sum_j c_j \phi(x_j) \Big\rangle
= \Big\lVert \sum_i c_i \phi(x_i) \Big\rVert^2 \ge 0 .
$$

**PSD is also sufficient**, which is the content of Mercer's theorem: for $k$
continuous and $\mathcal X$ compact, PSD-ness of $k$ guarantees an
eigenexpansion $k(x,x') = \sum_{m=1}^{\infty} \lambda_m \psi_m(x)\psi_m(x')$
with every $\lambda_m \ge 0$, so that
$\phi(x) = (\sqrt{\lambda_1}\psi_1(x), \sqrt{\lambda_2}\psi_2(x), \ldots)$ is
an explicit — if generally infinite-dimensional — feature map with
$k(x,x') = \langle \phi(x),\phi(x')\rangle_{\ell^2}$. Together, the two
directions say that PSD-ness exactly characterises which similarity functions
correspond to an inner product in some feature space, without ever
specifying what that space or that map need to look like.

## The RKHS and the reproducing property

A PSD kernel $k$ generates a specific Hilbert space of functions, built
directly from $k$ rather than from any particular $\phi$. Start with finite
sums

$$
f(\cdot) = \sum_{i=1}^n c_i \, k(\cdot, x_i) ,
$$

and define an inner product between two such functions,
$f = \sum_i c_i k(\cdot,x_i)$ and $g = \sum_j d_j k(\cdot, z_j)$, by

$$
\langle f, g\rangle_{\mathcal H} = \sum_{i,j} c_i d_j\, k(x_i, z_j) .
$$

This is well defined — independent of which coefficients happen to represent
$f$ or $g$ — and positive definite precisely because $k$ is PSD; completing
this space gives the **reproducing kernel Hilbert space (RKHS)** $\mathcal
H_k$. Its defining feature is the **reproducing property**: for any
$f \in \mathcal H_k$ and any point $x$,

$$
\langle f, k(\cdot, x) \rangle_{\mathcal H} = f(x) .
$$

This follows immediately from the construction: taking $g = k(\cdot,x)$ (the
single coefficient $d=1$ at the point $x$) in the inner product definition
gives $\langle f, k(\cdot,x)\rangle_{\mathcal H} = \sum_i c_i k(x_i,x) =
f(x)$. Evaluation at any point is an inner product against a single element
of the space — $k(\cdot,x)$ — which is exactly what it means for $k(\cdot,x)$
to represent the evaluation functional at $x$.

## The representer theorem

Consider fitting a function $f \in \mathcal H_k$ — not a parameter vector, a
function, possibly living in an infinite-dimensional space — by minimising

$$
J(f) = L\big(y_1, f(x_1), \ldots, y_n, f(x_n)\big) + \Omega\big(\lVert f \rVert_{\mathcal H}\big) ,
$$

where $L$ depends on $f$ only through its values at the $n$ training points,
and $\Omega : [0,\infty) \to \mathbb R$ is strictly increasing. Neither
assumption is restrictive: every loss function in this course depends on
predictions, not on $f$'s behaviour anywhere else, and every regulariser you
have seen — $\lVert w \rVert_2^2$, and its RKHS generalisation
$\lVert f \rVert_{\mathcal H}^2$ — is strictly increasing in the norm.

**Claim.** Any minimiser $f^*$ of $J$ can be written as
$f^*(\cdot) = \sum_{i=1}^n \alpha_i\, k(\cdot, x_i)$ — a combination of the
$n$ **kernel sections** $k(\cdot,x_i)$ at the training points, regardless of
the dimension of $\mathcal H_k$.

**Proof.** Take any $f \in \mathcal H_k$ and decompose it orthogonally
against the span of the training kernel sections,
$V = \operatorname{span}\{k(\cdot,x_1),\ldots,k(\cdot,x_n)\}$:

$$
f = f_\parallel + f_\perp, \qquad f_\parallel \in V, \qquad f_\perp \perp V .
$$

Because $f_\perp \perp V$, in particular $f_\perp \perp k(\cdot,x_i)$ for
every training point, so by the reproducing property
$f_\perp(x_i) = \langle f_\perp, k(\cdot,x_i)\rangle_{\mathcal H} = 0$ for
every $i$. Hence $f(x_i) = f_\parallel(x_i) + f_\perp(x_i) = f_\parallel(x_i)$
for every training point: **the loss term $L$ cannot distinguish $f$ from
$f_\parallel$**, since it only ever sees $f$ through its values at those $n$
points. By the Pythagorean identity for the orthogonal decomposition,

$$
\lVert f \rVert_{\mathcal H}^2 = \lVert f_\parallel \rVert_{\mathcal H}^2 + \lVert f_\perp \rVert_{\mathcal H}^2 \ \ge \ \lVert f_\parallel \rVert_{\mathcal H}^2 ,
$$

with equality only when $f_\perp = 0$. Since $\Omega$ is strictly increasing,
$\Omega(\lVert f\rVert_{\mathcal H}) \ge \Omega(\lVert f_\parallel
\rVert_{\mathcal H})$, again with equality only when $f_\perp = 0$. Combining
the two, $J(f) \ge J(f_\parallel)$, with equality only when $f_\perp = 0$.
Every minimiser of $J$ therefore has $f_\perp = 0$ and lies in $V$. $\blacksquare$

The exercise's `representer_check` verifies exactly the inequality this proof
establishes: it constructs a nonzero $f_\perp$ directly — a function built
from kernel sections *outside* the training set but engineered to vanish at
every training point — and confirms that adding any multiple of it to the
optimal $f_\parallel$ increases $J$ by precisely
$\Omega$'s contribution to $\lVert f_\perp \rVert_{\mathcal H}^2$, and by
nothing else, because the loss term genuinely cannot see it.

## Kernel ridge regression

Apply the representer theorem to ridge regression in $\mathcal H_k$:

$$
\min_{f \in \mathcal H_k} \ \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \lVert f \rVert_{\mathcal H}^2 .
$$

The theorem guarantees $f^* = \sum_i \alpha_i k(\cdot,x_i)$, so substitute.
Writing $K_{ij} = k(x_i,x_j)$, the predicted values are
$f(x_j) = \sum_i \alpha_i k(x_j,x_i) = (K\alpha)_j$, and the RKHS norm is,
directly from the inner product defined above,

$$
\lVert f \rVert_{\mathcal H}^2 = \Big\langle \sum_i \alpha_i k(\cdot,x_i), \sum_j \alpha_j k(\cdot,x_j)\Big\rangle_{\mathcal H} = \sum_{i,j}\alpha_i \alpha_j k(x_i,x_j) = \alpha^{\top}K\alpha .
$$

The problem becomes a finite-dimensional quadratic in $\alpha$:

$$
\min_{\alpha} \ \lVert y - K\alpha \rVert_2^2 + \lambda\, \alpha^{\top}K\alpha .
$$

Differentiating and setting the gradient to zero,
$-2K(y - K\alpha) + 2\lambda K\alpha = 0$, and since this holds in particular
when $K(K\alpha + \lambda\alpha - y) = 0$ is satisfied directly (not merely
in $K$'s range), the standard closed-form solution is

$$
\alpha = (K + \lambda I)^{-1} y .
$$

$K + \lambda I$ is invertible for any $\lambda > 0$ regardless of whether $K$
itself is, because $K$'s eigenvalues are non-negative (PSD) and adding
$\lambda I$ shifts every one of them up by $\lambda$.

### Recovering primal ridge regression

Take $k(x,x') = x^{\top}x'$, so $K = XX^{\top}$. The kernel solution predicts
through $w = X^{\top}\alpha = X^{\top}(XX^{\top}+\lambda I)^{-1}y$. Primal
ridge regression gives $\hat w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$. These
look different, but the identity

$$
(X^{\top}X + \lambda I)X^{\top} = X^{\top}XX^{\top} + \lambda X^{\top} = X^{\top}(XX^{\top}+\lambda I)
$$

rearranges, by left-multiplying by $(X^{\top}X+\lambda I)^{-1}$ and
right-multiplying by $(XX^{\top}+\lambda I)^{-1}$, to
$X^{\top}(XX^{\top}+\lambda I)^{-1} = (X^{\top}X+\lambda I)^{-1}X^{\top}$ —
exactly the equality of the two expressions for $w$. The kernel and primal
routes are the same computation, viewed through an $n \times n$ system in one
case and a $d \times d$ system in the other; the exercise verifies this
identity numerically rather than leaving it as algebra to trust.

## Kernels in practice

**Linear**, $k(x,x') = x^{\top}x'$, is ordinary ridge or SVM in the original
features, recovered as the special case above.

**Polynomial**, $k(x,x') = (x^{\top}x' + c)^p$ for $c \ge 0$, has an explicit
finite-dimensional feature map: expanding the power gives every monomial in
the coordinates of $x$ up to degree $p$, with $c$ controlling the relative
weight of lower-degree terms against the top one.

**RBF (Gaussian)**, $k(x,x') = \exp(-\gamma \lVert x - x'\rVert^2)$, is
infinite-dimensional. Expand the squared distance and factor:

$$
k(x,x') = e^{-\gamma \lVert x \rVert^2} \, e^{-\gamma \lVert x'\rVert^2} \, e^{2\gamma\, x^{\top}x'} ,
$$

and Taylor-expand the last factor, $e^{2\gamma\,x^{\top}x'} = \sum_{m=0}^{\infty}
\frac{(2\gamma)^m}{m!}(x^{\top}x')^m$. Each term $(x^{\top}x')^m$ is itself a
valid polynomial kernel with a finite-dimensional feature map, so the RBF
kernel is a non-negatively weighted infinite sum of valid PSD kernels — PSD
itself, by the same argument that a non-negative combination of PSD matrices
is PSD — whose implicit feature map concatenates the degree-$m$ monomial
features of every degree, each shifted by the Gaussian prefactor. No finite
set of explicit features ever produces this kernel exactly, which is both its
expressive power and the reason it needs the trick in the first place.

## The $O(n^3)$ and $O(n^2)$ walls

Kernel ridge regression needs $(K+\lambda I)^{-1}y$: a Cholesky factorisation
of an $n\times n$ matrix costs $O(n^3)$ time, and $K$ itself costs $O(n^2)$
memory regardless of how the solve is done. Prediction on a new point costs
$O(n)$ once $K$'s factorisation and $\alpha$ are cached, or $O(nd)$ if the
kernel row against every training point still needs computing. Both walls
are the same ones the SVM's dual meets, for the same reason: any method whose
representer-theorem solution needs every training point's kernel value pays
$O(n^2)$ to store them and $O(n^3)$ to solve the resulting linear system.

## Random Fourier features

**Bochner's theorem** says a continuous, shift-invariant kernel
$k(x,x') = \kappa(x-x')$ is PSD exactly when $\kappa$ is the Fourier
transform of a finite non-negative measure — after normalising
$\kappa(0)=1$, a probability distribution $p(\omega)$:

$$
\kappa(\delta) = \int p(\omega)\, e^{i\omega^{\top}\delta}\, d\omega = \mathbb E_{\omega \sim p}\big[\cos(\omega^{\top}\delta)\big] ,
$$

the last equality holding because $\kappa$ is real and $p$ is symmetric. For
the RBF kernel with bandwidth $\gamma$, matching Gaussian characteristic
functions gives $p(\omega) = \mathcal N(0, 2\gamma I)$: writing $\Sigma$ for
the covariance of $\omega$, $\mathbb E_\omega[e^{i\omega^{\top}\delta}] =
e^{-\frac12 \delta^{\top}\Sigma\delta}$, which matches
$e^{-\gamma\lVert\delta\rVert^2} = e^{-\delta^{\top}\delta/(1/\gamma)}$
exactly when $\Sigma = 2\gamma I$.

A single random frequency does not give an inner product, but a phase-shifted
pair does. Averaging over an independent phase $b \sim \operatorname{Uniform}(0,2\pi)$,
the product-to-sum identity gives
$\mathbb E_b\big[\cos(\omega^{\top}x+b)\cos(\omega^{\top}x'+b)\big] = \tfrac12
\cos(\omega^{\top}(x-x'))$, so defining
$z_\omega(x) = \sqrt2 \cos(\omega^{\top}x+b)$ gives
$\mathbb E_{\omega,b}[z_\omega(x)z_\omega(x')] = k(x,x')$ exactly. **Random
Fourier features** replace the expectation with a Monte Carlo average over
$D$ independent draws $(\omega_1,b_1),\ldots,(\omega_D,b_D)$:

$$
z(x) = \sqrt{\frac{2}{D}} \big(\cos(\omega_1^{\top}x+b_1), \ldots, \cos(\omega_D^{\top}x+b_D)\big) ,
\qquad
z(x)^{\top}z(x') \approx k(x,x') .
$$

Each of the $D$ summands in $z(x)^{\top}z(x')$ is bounded, since $\cos$ is
bounded by 1, so $z(x)^{\top}z(x')$ is an average of $D$ bounded i.i.d. terms
with mean $k(x,x')$. Hoeffding's inequality bounds its deviation from that
mean by $O(1/\sqrt D)$ with high probability, and a union bound over a
covering of a compact domain extends that to a uniform guarantee over every
pair $(x,x')$ at once, at the price of an extra factor logarithmic in the
covering number. Doubling $D$ shrinks the typical error by a factor of
$\sqrt 2$, not by half — the price of a Monte Carlo estimate rather than a
convergent series.

Once $z$ is fixed, $z(x)^{\top}z(x')$ is an ordinary inner product, so
whatever downstream method receives it runs as if the features were
$D$-dimensional and explicit: ridge regression costs $O(nD^2 + D^3)$ rather
than $O(n^3)$, a win whenever $D \ll n$.

## Nyström approximation

Random Fourier features approximate the *kernel function*, independent of the
data. The **Nyström method** instead approximates the *Gram matrix* directly,
using the data itself: sample $m \ll n$ landmark points, and approximate

$$
K \approx K_{n,m}\, K_{m,m}^{+}\, K_{m,n} ,
$$

where $K_{n,m}$ holds kernel values between all $n$ points and the $m$
landmarks, and $K_{m,m}^{+}$ is the pseudoinverse of the $m\times m$
landmark Gram matrix. This costs $O(nm^2 + m^3)$ rather than $O(n^3)$, and
because the landmarks come from the data's own distribution, Nyström adapts
to $K$'s actual spectrum — it can outperform random features sharply when
$K$'s effective rank is low — at the cost of needing access to the data
up front, where random features can be drawn without looking at it at all.

## Cost and scaling

| Method | Time | Memory |
|---|---|---|
| Exact kernel ridge | $O(n^3)$ | $O(n^2)$ |
| Random Fourier features, then ridge | $O(nD^2 + D^3)$ | $O(nD)$ |
| Nyström, then ridge | $O(nm^2 + m^3)$ | $O(nm)$ |

Both approximations turn the wall from quadratic-or-worse in $n$ into linear
in $n$, at the cost of a tunable approximation dimension ($D$ or $m$) that
has to grow with the desired accuracy and, for random features, with the
input dimension.

## Where this breaks

**Bandwidth and kernel choice.** $\gamma$, the polynomial degree, and the
choice of kernel family are hyperparameters exactly like $C$ in the SVM
lesson, tuned by the cross-validation of lesson 7, with no default that
transfers across problems.

**A regulariser that is not a function of $\lVert f\rVert_{\mathcal H}$.**
The representer theorem's proof used only that $\Omega$ is strictly
increasing in the RKHS norm. A different smoothness penalty — one not built
from $\langle \cdot,\cdot\rangle_{\mathcal H}$ — breaks the orthogonality
argument entirely, and the resulting optimal $f$ need not live in any
finite-dimensional span at all.

**Near-singular $K + \lambda I$.** Small $\lambda$, needed to approach
interpolation, combined with near-duplicate training points, reproduces
exactly the conditioning collapse of the normal equations in lesson 2. The
Cholesky-based solve catches the failure as a non-positive-definite matrix
rather than silently returning nonsense, but no factorisation recovers digits
that were never there.

**Random features in high dimensions.** The union bound behind the uniform
$O(1/\sqrt D)$ guarantee costs a covering number that grows with the input
dimension, so matching a fixed accuracy in high dimensions needs $D$ to grow
correspondingly — a real cost, not just a looser constant.

**The wall returns at deployment.** Both approximations move the wall rather
than removing the underlying trade: a kernel method with $D$ or $m$ in the
thousands is still doing thousands of multiplications per prediction, which
is why, once $n$ is large enough, practitioners increasingly prefer methods
that learn a finite feature map directly from data — the subject of Part 3.

## The exercise

You build the machinery of kernel methods and then verify the representer
theorem numerically rather than take the proof's word for it. Implement
`rbf_kernel` and `polynomial_kernel`; `is_psd`, which checks that a candidate
Gram matrix is symmetric and has no negative eigenvalue beyond tolerance;
`kernel_ridge_fit` and `kernel_ridge_predict`, solving the closed form
derived above through a Cholesky factorisation; `random_fourier_features`,
built from Bochner's theorem; and `representer_check`, which evaluates the
regularised objective for a candidate function given by its coefficients over
an arbitrary set of kernel sections.

The checks confirm both kernels produce PSD Gram matrices on random inputs
and that `is_psd` correctly rejects a matrix that is not one, that kernel
ridge regression with a linear kernel reproduces primal ridge regression's
weights exactly, and that random Fourier features approximate the exact RBF
Gram matrix with error that shrinks as $D$ grows. The last check is the
representer theorem made numerical: it builds a function supported on kernel
sections outside the training set but engineered, via a null-space
construction, to vanish at every training point, then confirms that adding
any multiple of it to the fitted solution increases the regularised objective
by exactly the amount the orthogonal-decomposition proof predicts — no more,
no less.
