---
title: Kernels and the representer theorem
slug: 10-kernels
part: "Part 2 — Classical machine learning"
summary: What a valid kernel is, why its optimum always lives in the span of the training data, and how to scale past it.
minutes: 45
gpu: false
objectives:
  - State Mercer's condition, that a kernel is valid exactly when its Gram matrix is always positive semidefinite.
  - State the reproducing property of the RKHS a kernel induces.
  - State the representer theorem and see why it follows from an orthogonal decomposition.
  - Derive kernel ridge regression's closed-form solution, and see that it recovers primal ridge regression under a linear kernel.
  - Compute random Fourier features and state their O(1/sqrt(D)) concentration rate.
prereqs:
  - 09-support-vector-machines
  - 04-regularisation
exercise: 10-kernels
---

# Kernels and the representer theorem

The SVM's dual depends on the training data only through inner products: $x_i^{\top}x_j$ for training pairs, $x^{\top}x_i$ at prediction time. Nothing in that derivation needed a literal Euclidean inner product, only a symmetric, positive measure of similarity between two points. Replace it with a richer similarity measure, and you gain access to feature spaces too large to write down, for free.

This lesson makes that idea precise in two steps. First, what counts as a valid similarity measure, a **kernel**, and what space of functions it defines. Second, why an enormous class of learning problems, not just the SVM, always has a solution built from finitely many training points, no matter how large or infinite-dimensional that space is. That second fact is the **representer theorem**, and it is the reason the kernel trick is a general method rather than a coincidence specific to margins.

## What you already know

- The SVM dual and its prediction rule depend on the data only through $x^{\top}x'$.
- Ridge regression's closed form $\hat w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$.
- A symmetric matrix is positive semidefinite exactly when $c^{\top}Mc \ge 0$ for every vector $c$.
- Orthogonal decomposition: for a subspace $V$, every $f$ splits uniquely as $f = f_\parallel + f_\perp$ with $f_\parallel \in V$ and $f_\perp \perp V$, and $\lVert f\rVert^2 = \lVert f_\parallel\rVert^2 + \lVert f_\perp\rVert^2$.
- A concentration inequality, Hoeffding's for instance, bounds how far the average of $D$ bounded i.i.d. random variables strays from its mean, shrinking as $O(1/\sqrt D)$.

## What makes a kernel valid

A **kernel** is a symmetric function $k : \mathcal X \times \mathcal X \to \mathbb R$. Call it **positive semidefinite (PSD)** when, for every finite set of points $x_1,\ldots,x_n$, the Gram matrix $K_{ij}=k(x_i,x_j)$ is PSD as a matrix.

PSD-ness is exactly the property of being an inner product somewhere. If $k(x,x')=\langle\phi(x),\phi(x')\rangle$ for some map $\phi$, then for any $c$,

$$
c^{\top}Kc = \Big\lVert \sum_i c_i \phi(x_i) \Big\rVert^2 \ge 0 ,
$$

so PSD is necessary. **Mercer's theorem** gives the converse: a continuous PSD kernel on a compact domain admits an eigenexpansion $k(x,x')=\sum_m \lambda_m \psi_m(x)\psi_m(x')$ with every $\lambda_m \ge 0$, so $\phi(x) = (\sqrt{\lambda_1}\psi_1(x), \ldots)$ is an explicit, if generally infinite-dimensional, feature map that reproduces $k$. PSD-ness is the whole test: it tells you which similarity functions correspond to some inner product, without pinning down what that inner product's space looks like.

## The RKHS and its reproducing property

A PSD kernel generates a Hilbert space of functions built directly from $k$. Start with finite sums $f(\cdot)=\sum_i c_i\,k(\cdot,x_i)$, and define an inner product between two such sums by $\langle f,g\rangle_{\mathcal H} = \sum_{i,j}c_id_j\,k(x_i,z_j)$. Completing this space gives the **reproducing kernel Hilbert space (RKHS)** $\mathcal H_k$, and it is positive definite precisely because $k$ is PSD.

Its defining feature is the **reproducing property**: for any $f\in\mathcal H_k$ and any point $x$,

$$
\langle f, k(\cdot,x)\rangle_{\mathcal H} = f(x) .
$$

Evaluating $f$ at $x$ is the same operation as taking an inner product against the single function $k(\cdot,x)$. That one identity is what the representer theorem's proof runs on.

## The representer theorem

Consider fitting a function $f\in\mathcal H_k$, not a parameter vector but a function that may live in an infinite-dimensional space, by minimising

$$
J(f) = L\big(y_1,f(x_1),\ldots,y_n,f(x_n)\big) + \Omega\big(\lVert f\rVert_{\mathcal H}\big) ,
$$

where $L$ depends on $f$ only through its values at the $n$ training points, and $\Omega$ is strictly increasing. Every loss and regulariser in this course fits that shape.

**Claim.** Any minimiser $f^*$ can be written $f^*(\cdot)=\sum_{i=1}^n \alpha_i\,k(\cdot,x_i)$, a combination of the $n$ training kernel sections, whatever the dimension of $\mathcal H_k$.

The proof decomposes any candidate $f$ against $V=\operatorname{span}\{k(\cdot,x_1),\ldots,k(\cdot,x_n)\}$: $f = f_\parallel + f_\perp$. The reproducing property gives $f_\perp(x_i) = \langle f_\perp, k(\cdot,x_i)\rangle_{\mathcal H}=0$ for every training point, so $L$ cannot distinguish $f$ from $f_\parallel$, since it only ever sees $f$ through those $n$ values. Meanwhile $\lVert f\rVert_{\mathcal H}^2 = \lVert f_\parallel\rVert^2 + \lVert f_\perp\rVert^2 \ge \lVert f_\parallel\rVert^2$, with equality only when $f_\perp=0$, and $\Omega$ being strictly increasing carries that inequality through to $J$. So $J(f)\ge J(f_\parallel)$, and every minimiser has $f_\perp = 0$.

The exercise's `representer_check` verifies exactly this inequality: it builds a nonzero $f_\perp$ from kernel sections outside the training set, engineered to vanish at every training point, and confirms that adding any multiple of it to the optimal $f_\parallel$ increases $J$ by precisely $\Omega$'s contribution to $\lVert f_\perp\rVert_{\mathcal H}^2$. Nothing else changes, because the loss term cannot see it.

## Kernel ridge regression

Apply the theorem to ridge regression in $\mathcal H_k$:

$$
\min_{f\in\mathcal H_k}\ \sum_{i=1}^n (y_i-f(x_i))^2 + \lambda\lVert f\rVert_{\mathcal H}^2 .
$$

The optimum is $f^*=\sum_i\alpha_i k(\cdot,x_i)$, so predictions are $f(x_j)=(K\alpha)_j$ and $\lVert f\rVert_{\mathcal H}^2 = \alpha^{\top}K\alpha$, where $K_{ij}=k(x_i,x_j)$. The problem becomes a finite quadratic in $\alpha$, and setting its gradient to zero gives the closed form

$$
\alpha = (K + \lambda I)^{-1} y .
$$

$K+\lambda I$ is always invertible for $\lambda>0$: $K$'s eigenvalues are non-negative, and adding $\lambda I$ shifts every one of them up.

Take $k(x,x')=x^{\top}x'$, so $K=XX^{\top}$. The kernel solution predicts through $w=X^{\top}\alpha=X^{\top}(XX^{\top}+\lambda I)^{-1}y$, and a short matrix identity shows this equals primal ridge regression's $\hat w=(X^{\top}X+\lambda I)^{-1}X^{\top}y$ exactly. Kernel and primal ridge are the same computation, solved as an $n\times n$ system in one case and a $d\times d$ system in the other.

## Kernels in practice

**Linear**, $k(x,x')=x^{\top}x'$, is ordinary ridge or SVM in the original features. **Polynomial**, $k(x,x')=(x^{\top}x'+c)^p$, has an explicit finite-dimensional feature map: every monomial in $x$'s coordinates up to degree $p$. **RBF (Gaussian)**, $k(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$, is infinite-dimensional. Expanding it gives a non-negatively weighted sum of every polynomial kernel's monomials, so it is PSD, and no finite set of explicit features reproduces it exactly.

## The $O(n^3)$ and $O(n^2)$ walls

Kernel ridge regression needs $(K+\lambda I)^{-1}y$: a Cholesky factorisation of an $n\times n$ matrix costs $O(n^3)$ time, and $K$ itself costs $O(n^2)$ memory regardless of how the solve is done. Both walls are the ones the SVM's dual meets, for the same reason: any method whose representer-theorem solution needs every training point's kernel value pays this cost to store and solve it.

## Random Fourier features

**Bochner's theorem** says a shift-invariant kernel $k(x,x')=\kappa(x-x')$ is PSD exactly when $\kappa$ is the Fourier transform of a probability distribution $p(\omega)$ over frequencies. For the RBF kernel with bandwidth $\gamma$, matching characteristic functions gives $p(\omega)=\mathcal N(0,2\gamma I)$.

A single random frequency does not give an inner product, but a phase-shifted pair does: averaging over $\omega\sim p$ and an independent phase $b\sim\text{Uniform}(0,2\pi)$,

$$
\mathbb E_{\omega,b}\big[z_\omega(x)z_\omega(x')\big] = k(x,x'), \qquad z_\omega(x) = \sqrt2\cos(\omega^{\top}x+b) .
$$

**Random Fourier features** replace that expectation with an average over $D$ independent draws:

$$
z(x) = \sqrt{\frac2D}\big(\cos(\omega_1^{\top}x+b_1),\ldots,\cos(\omega_D^{\top}x+b_D)\big), \qquad
z(x)^{\top}z(x') \approx k(x,x') .
$$

Each of the $D$ terms is bounded, so Hoeffding's inequality bounds the approximation error by $O(1/\sqrt D)$: doubling $D$ shrinks the typical error by a factor of $\sqrt2$, not by half. Once $z$ is fixed, downstream ridge regression runs on $D$-dimensional explicit features, costing $O(nD^2+D^3)$ instead of $O(n^3)$, a win whenever $D\ll n$.

## Nyström approximation

Random features approximate the kernel function itself, independent of the data. The **Nyström method** instead approximates the Gram matrix directly: sample $m\ll n$ landmark points and use $K\approx K_{n,m}K_{m,m}^{+}K_{m,n}$, costing $O(nm^2+m^3)$. Because the landmarks come from the data, Nyström adapts to $K$'s actual spectrum and can beat random features sharply when $K$'s effective rank is low, at the cost of needing the data up front.

## Cost and scaling

| Method | Time | Memory |
|---|---|---|
| Exact kernel ridge | $O(n^3)$ | $O(n^2)$ |
| Random Fourier features, then ridge | $O(nD^2 + D^3)$ | $O(nD)$ |
| Nyström, then ridge | $O(nm^2 + m^3)$ | $O(nm)$ |

Both approximations turn the wall from cubic or quadratic in $n$ into linear in $n$, at the cost of an approximation dimension, $D$ or $m$, that must grow with the accuracy you want.

## Where this breaks

**Bandwidth and kernel choice.** $\gamma$, the polynomial degree, and the kernel family are hyperparameters tuned by cross-validation, with no default that transfers across problems.

**A regulariser that is not a function of $\lVert f\rVert_{\mathcal H}$.** The representer theorem's proof used only that $\Omega$ is strictly increasing in the RKHS norm. A different smoothness penalty breaks the orthogonality argument, and the optimal $f$ need not live in any finite span at all.

**Near-singular $K+\lambda I$.** A small $\lambda$, needed to approach interpolation, combined with near-duplicate training points, reproduces the conditioning collapse of the normal equations from lesson 2.

**Random features in high dimensions.** A uniform accuracy guarantee across every pair $(x,x')$ needs $D$ to grow with the input dimension, a real cost rather than a looser constant.

**The wall returns at deployment.** Both approximations move the cost rather than remove it: a kernel method with $D$ or $m$ in the thousands still does thousands of multiplications per prediction, which is why practitioners increasingly prefer methods that learn a finite feature map directly from data once $n$ is large, the subject of Part 3.

## The exercise

You build the machinery of kernel methods and then verify the representer theorem numerically. Implement `rbf_kernel` and `polynomial_kernel`; `is_psd`, checking that a candidate Gram matrix is symmetric with no meaningfully negative eigenvalue; `kernel_ridge_fit` and `kernel_ridge_predict`, solving the closed form above through a Cholesky factorisation; `random_fourier_features`, built from Bochner's theorem; and `representer_check`, which evaluates the regularised objective for a candidate function given by its coefficients over an arbitrary set of kernel sections.

The checks confirm both kernels produce PSD Gram matrices, that kernel ridge regression with a linear kernel reproduces primal ridge regression's weights exactly, and that random Fourier features approximate the exact RBF Gram matrix with error that shrinks as $D$ grows. The last check makes the representer theorem numerical: it builds a function supported on kernel sections outside the training set, engineered to vanish at every training point, then confirms that adding any multiple of it to the fitted solution increases the objective by exactly the amount the orthogonal-decomposition proof predicts.
