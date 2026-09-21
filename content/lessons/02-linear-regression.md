---
title: Linear regression
slug: 02-linear-regression
part: "Part 1 — Foundations"
summary: Least squares as an orthogonal projection, three ways to compute it, and why the textbook formula loses digits the others don't.
minutes: 50
gpu: false
objectives:
  - State the normal equations, and read them as an orthogonality condition on the residual.
  - Explain why forming X^T X squares the condition number, and what that costs in digits.
  - Solve least squares through QR and through the SVD, and state when each is the right choice.
  - Describe the solution set when X is rank deficient, and identify the minimum-norm element.
prereqs:
  - 01-the-learning-problem
exercise: 02-linear-regression
---

# Linear regression

Least squares has a closed-form answer, and that answer is a projection. Almost every idea in the rest of this course responds to the fact that once the hypothesis class stops being a linear subspace, the projection is gone and you need something else.

The formula most textbooks print, $\hat{w} = (X^{\top}X)^{-1}X^{\top}y$, is also the one you must not implement. It throws away roughly twice as many digits as the problem itself requires. On the design matrices you meet in practice — correlated features, a column of ones, wildly different units — that difference decides whether your coefficients mean anything at all.

This lesson states the solution, reads it geometrically, and builds the two factorisations that compute it without the digit loss.

## What you already know

- Empirical risk minimisation, and that a held-out estimate is the only honest measure of risk.
- The gradient of a quadratic form: $\nabla_w \, w^{\top}Aw = (A + A^{\top})w$.
- Orthogonal matrices preserve the Euclidean norm, and $Q^{\top}Q = I$.
- Every real matrix has a singular value decomposition, and its singular values are the square roots of the eigenvalues of $X^{\top}X$.
- Floating-point arithmetic carries about 16 decimal digits in double precision.

## The normal equations

Take $X \in \mathbb{R}^{n \times d}$ with rows $x_i^{\top}$, and $y \in \mathbb{R}^n$. The empirical risk under squared loss is $L(w) = \tfrac{1}{2}\lVert Xw - y \rVert_2^2$, with gradient $\nabla L(w) = X^{\top}Xw - X^{\top}y$ and constant Hessian $X^{\top}X \succeq 0$. A constant, positive semidefinite Hessian means $L$ is convex, so every stationary point is a global minimum. Setting the gradient to zero gives the **normal equations**:

$$
X^{\top} X \hat{w} = X^{\top} y .
$$

They have a solution for every $X$ and $y$, because $X^{\top}y$ always lies in the range of $X^{\top}X$. When $X$ has full column rank, $X^{\top}X$ is positive definite and that solution is unique.

### The projection reading

Rewrite the normal equations as $X^{\top}(y - X\hat{w}) = 0$. The residual $r = y - X\hat{w}$ is orthogonal to every column of $X$, hence to the whole column space of $X$. That single fact is the geometry of least squares: $X\hat{w}$ is the orthogonal projection of $y$ onto the column space of $X$, and $\hat{w}$ is the coordinate vector of that projection in the basis the columns give you.

Take a two-column example to see it directly. If $X$'s two columns are already orthogonal, projecting $y$ onto their span decomposes into two independent one-dimensional projections, one per column, and $\hat w$ is just the pair of those two dot products divided by the column norms squared. Correlated columns lose that independence, which is exactly why the rest of this lesson exists.

This reading also explains a fact you can otherwise find surprising: adding a feature that is already a linear combination of existing features never improves the fit, because the column space does not grow, so the projection does not move.

## Why not to invert

Define the condition number of a full-rank $X$ as $\kappa_2(X) = \sigma_{\max}(X) / \sigma_{\min}(X)$, the ratio of its extreme singular values. It bounds how much a relative perturbation of the data can change the answer: a large $\kappa_2(X)$ means a design matrix where small measurement noise can swing the fitted coefficients by a lot.

Because the singular values of $X^{\top}X$ are $\sigma_i^2$,

$$
\kappa_2(X^{\top}X) = \kappa_2(X)^2 .
$$

Forming the Gram matrix $X^\top X$ squares the conditioning, before you have even started solving anything. In double precision you begin with about 16 digits, and solving through $X^{\top}X$ leaves roughly $16 - 2\log_{10}\kappa_2(X)$ of them. A design matrix with $\kappa_2(X) = 10^{8}$ — unremarkable when one feature is measured in metres and another in micrometres — leaves nothing at all. Two columns that differ only in the last representable digit become numerically identical inside $X^{\top}X$, and Cholesky factorisation can fail outright on a matrix that is mathematically positive definite.

The fix is to never form the Gram matrix, and both alternatives below work on $X$ directly.

### QR

Factor $X = QR$, the thin factorisation with $Q \in \mathbb{R}^{n \times d}$ having orthonormal columns and $R \in \mathbb{R}^{d \times d}$ upper triangular. Because $Q$ preserves norms on its range, minimising $\lVert Xw - y \rVert_2^2$ is the same as minimising $\lVert Rw - Q^{\top}y \rVert_2^2$ plus a constant that does not involve $w$. That means solving one triangular system by back substitution:

$$
R \hat{w} = Q^{\top} y .
$$

Back substitution costs $O(d^2)$. The Householder factorisation that produces $Q$ and $R$ costs $2nd^2 - \tfrac{2}{3}d^3$ flops, and it is backward stable: the computed $\hat{w}$ solves a nearby problem exactly, with error governed by $\kappa_2(X)$ itself rather than its square.

### The SVD

Factor $X = U \Sigma V^{\top}$, with $U \in \mathbb{R}^{n \times r}$ and $V \in \mathbb{R}^{d \times r}$ having orthonormal columns, $\Sigma$ diagonal and positive, and $r = \operatorname{rank}(X)$. The least squares solution is

$$
\hat{w} = V \Sigma^{-1} U^{\top} y = X^{+} y ,
$$

where $X^{+}$ is the Moore-Penrose pseudoinverse. Writing the sum out term by term makes the conditioning visible directly:

$$
\hat{w} = \sum_{i=1}^{r} \frac{u_i^{\top} y}{\sigma_i} \, v_i .
$$

Each direction $v_i$ carries a weight of $1/\sigma_i$. A small singular value multiplies whatever noise lands on $u_i$ by a large factor — the same instability the condition number predicts, and the instability that ridge regression will damp in lesson 4 by replacing $1/\sigma_i$ with $\sigma_i / (\sigma_i^2 + \lambda)$.

### Rank deficiency

When $\operatorname{rank}(X) = r < d$, the normal equations still hold, but they no longer pin down a unique $\hat{w}$. The solution set is an affine subspace of dimension $d - r$: one particular solution plus anything in $X$'s null space. Every element of that set gives the same fitted values and the same training loss, so the data cannot tell them apart.

The pseudoinverse picks one of them: $X^{+}y$ is the element of smallest Euclidean norm, because by construction it puts zero weight on every null-space direction. That is a choice, not a consequence forced by the data — and it is the same choice that gradient descent makes on its own when started from $w_0 = 0$, the first implicit-regularisation result you meet in this course.

Numerically, rank is a judgment call rather than a fact: singular values at or below $\max(n, d) \cdot \varepsilon \cdot \sigma_{\max}$ get treated as zero, and that threshold is a modelling decision wearing a numerical disguise.

## Cost and scaling

The three routes agree in exact arithmetic and differ in flops and in the digits they keep. The following table assumes $n \geq d$ and counts leading terms.

| Route | Flops | Extra memory | Digits lost |
|---|---|---|---|
| Normal equations with Cholesky | $nd^2 + \tfrac{1}{3}d^3$ | $d^2$ | $2\log_{10}\kappa_2(X)$ |
| Householder QR | $2nd^2 - \tfrac{2}{3}d^3$ | $d^2$ | $\log_{10}\kappa_2(X)$ |
| SVD | $\approx 6nd^2 + 20d^3$ | $nd$ | $\log_{10}\kappa_2(X)$ |

Cholesky on the Gram matrix is the cheapest route by about a factor of two, and the only one that never needs $X$ again once $X^\top X$ and $X^\top y$ are formed — useful when $n$ is large enough that $X$ does not fit in memory, since you can accumulate both in a single streaming pass using $O(d^2)$ space. That is why the Gram route survives in practice, for well-conditioned, low-dimensional problems where you also standardise the columns first.

For $n \gg d$, every route costs $O(nd^2)$, so the constant in front is what separates them. As $d$ approaches $n$, the $d^3$ terms start to dominate, and the SVD grows expensive enough that iterative methods take over — the subject of lesson 3.

## Where this breaks

**Correlated features.** Near-collinearity makes $\sigma_{\min}$ small without making it zero. The fit stays fine and the coefficients turn meaningless: their variance is $\operatorname{Var}(\hat{w}) = \sigma^2 (X^{\top}X)^{-1}$, with a trace of $\sigma^2 \sum_i \sigma_i^{-2}$. Two features that measure almost the same thing typically produce one large positive coefficient and one large negative one that cancel. Ridge regression, in lesson 4, is the standard response.

**Unscaled columns.** A column measured in different units changes $\kappa_2(X)$ without changing the model at all. Standardise before fitting, and exclude the intercept from any penalty you add later.

**More features than samples.** With $d > n$ the problem is always rank deficient, the training residual is zero, and the choice among the many exact-fitting solutions carries the entire modelling content. Modern models live in this regime routinely, and the minimum-norm solution is a large part of why gradient descent generalises there at all.

**Heavy-tailed noise.** Squared loss is the maximum-likelihood estimator under Gaussian noise, and only under Gaussian noise. A single outlier can move $\hat{w}$ without bound, because a point's influence on the fit grows linearly with its residual. Huber loss bounds that influence, at the cost of the closed form.

## The exercise

You implement least squares three times, and measure the difference the factorisation makes.

Write `fit_normal_equations`, `fit_qr`, and `fit_svd`, each taking a design matrix `X` and targets `y` and returning the coefficient vector. Build them from `numpy.linalg` primitives — `cholesky`, `qr`, `svd`, and a triangular solve — rather than calling `numpy.linalg.lstsq`, which would do the whole exercise for you. `fit_svd` takes an `rcond` tolerance, and must return the minimum-norm solution when the design matrix is rank deficient.

Then write `residual_orthogonality`, which returns $\lVert X^{\top}(y - X\hat{w}) \rVert_\infty$ for a fitted `w`, and `condition_number`, which returns $\kappa_2(X)$ computed from the singular values.

The checks verify that the three routes agree to high precision on a well-conditioned problem, that the residual is orthogonal to the column space, and that `fit_svd` returns the minimum-norm element on a deliberately rank-deficient design. The last check is the point of the lesson: on a Hilbert-like matrix with $\kappa_2 \approx 10^{9}$, it confirms that the normal-equations route loses the digits the table above predicts while QR and the SVD keep them — and it fails if your `fit_normal_equations` is secretly calling something stable.
