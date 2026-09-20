---
title: Linear regression
slug: 02-linear-regression
part: "Part 1 — Foundations"
summary: Least squares as an orthogonal projection, three ways to compute it, and the conditioning that decides which one you can trust.
minutes: 55
gpu: false
objectives:
  - Derive the normal equations from the first-order condition and read them as an orthogonality statement.
  - Explain why forming X^T X squares the condition number, and what that costs in digits.
  - Solve least squares through QR and through the SVD, and state when each is the right choice.
  - Characterise the solution set when X is rank deficient, and identify the minimum-norm element.
prereqs:
  - 01-the-learning-problem
exercise: 02-linear-regression
---

# Linear regression

Empirical risk minimisation over the squared loss and a linear hypothesis class
is the one nontrivial learning problem with a closed-form answer. That closed
form is a projection, and almost every idea in the rest of this course is a
response to the fact that the projection is not available once the hypothesis
class stops being a linear subspace.

The optimisation is settled in four lines. What takes the rest of this lesson is
the numerical question underneath it: the textbook formula
$\hat{w} = (X^{\top}X)^{-1}X^{\top}y$ is the one you must not implement. It
loses roughly twice as many digits as the problem itself requires, and on the
design matrices that turn up in practice — correlated features, a column of
ones, wildly different units — that difference decides whether your coefficients
mean anything.

This lesson derives the solution, reads it geometrically, then builds the two
factorisations that compute it without the squaring.

## What you already know

- Empirical risk minimisation, and that a held-out estimate is the only honest
  measure of risk.
- The gradient of a quadratic form: $\nabla_w \, w^{\top}Aw = (A + A^{\top})w$.
- Orthogonal matrices preserve the Euclidean norm, and $Q^{\top}Q = I$.
- Every real matrix has a singular value decomposition, and the singular values
  are the square roots of the eigenvalues of $X^{\top}X$.
- Floating-point arithmetic carries about 16 decimal digits in double precision.

## The normal equations

Take $X \in \mathbb{R}^{n \times d}$ with rows $x_i^{\top}$, and
$y \in \mathbb{R}^n$. The empirical risk under squared loss is

$$
L(w) = \frac{1}{2}\lVert Xw - y \rVert_2^2 .
$$

The factor of one half is bookkeeping; it cancels in the gradient. Expanding,

$$
L(w) = \tfrac{1}{2} w^{\top} X^{\top} X w - w^{\top} X^{\top} y + \tfrac{1}{2} y^{\top} y ,
$$

so $\nabla L(w) = X^{\top}Xw - X^{\top}y$ and
$\nabla^2 L(w) = X^{\top}X \succeq 0$. The Hessian is constant and positive
semidefinite, so $L$ is convex and every stationary point is a global minimum.
Setting the gradient to zero gives the **normal equations**:

$$
X^{\top} X \hat{w} = X^{\top} y .
$$

They are consistent for every $X$ and $y$, because $X^{\top}y$ always lies in the
range of $X^{\top}X$. When $X$ has full column rank, $X^{\top}X$ is positive
definite and the solution is unique.

### The projection reading

Rewrite the normal equations as $X^{\top}(y - X\hat{w}) = 0$. The residual
$r = y - X\hat{w}$ is orthogonal to every column of $X$, so it is orthogonal to
$\operatorname{range}(X)$. That is the whole content of least squares:
$X\hat{w}$ is the orthogonal projection of $y$ onto the column space of $X$, and
$\hat{w}$ is the coordinate vector of that projection in the basis given by the
columns.

The projection matrix is $P = X(X^{\top}X)^{-1}X^{\top}$, which satisfies
$P^2 = P$ and $P^{\top} = P$. Its trace is $d$, and $\operatorname{tr}(I - P)$
is $n - d$ — the degrees of freedom left in the residual, and the reason the
unbiased variance estimate divides by $n - d$ rather than $n$.

This reading explains why the fit cannot improve when you add a feature that is
already a linear combination of existing ones: the column space does not grow, so
the projection does not move.

## Why not to invert

Define the condition number of a full-rank $X$ as
$\kappa_2(X) = \sigma_{\max}(X) / \sigma_{\min}(X)$, the ratio of its extreme
singular values. It bounds how much a relative perturbation of the data can
change the answer.

Since the singular values of $X^{\top}X$ are $\sigma_i^2$,

$$
\kappa_2(X^{\top}X) = \kappa_2(X)^2 .
$$

Forming the Gram matrix squares the conditioning. In double precision you start
with about 16 digits; solving through $X^{\top}X$ leaves you roughly
$16 - 2\log_{10}\kappa_2(X)$. A design matrix with $\kappa_2(X) = 10^{8}$ —
unremarkable when one feature is in metres and another in micrometres — leaves
nothing at all.

Worse, the squaring happens before any solve. Two columns that differ in the
last representable digit become numerically identical in $X^{\top}X$, and the
Cholesky factorisation fails outright with a matrix that is mathematically
positive definite.

The fix is to never form the Gram matrix. Both alternatives work on $X$ directly.

### QR

Factor $X = QR$ with $Q \in \mathbb{R}^{n \times d}$ having orthonormal columns
and $R \in \mathbb{R}^{d \times d}$ upper triangular — the thin, or reduced,
factorisation. Because $Q$ preserves norms on its range,

$$
\lVert Xw - y \rVert_2^2 = \lVert QRw - y \rVert_2^2
= \lVert Rw - Q^{\top}y \rVert_2^2 + \lVert (I - QQ^{\top}) y \rVert_2^2 .
$$

The second term is free of $w$. Minimising the first means solving the triangular
system

$$
R \hat{w} = Q^{\top} y ,
$$

by back substitution, in $O(d^2)$. The factorisation itself, by Householder
reflections, costs $2nd^2 - \tfrac{2}{3}d^3$ flops and is backward stable: the
computed $\hat{w}$ is the exact solution of a nearby problem, with the error
governed by $\kappa_2(X)$ and not its square.

### The SVD

Factor $X = U \Sigma V^{\top}$ with $U \in \mathbb{R}^{n \times r}$,
$V \in \mathbb{R}^{d \times r}$ having orthonormal columns, $\Sigma$ diagonal and
positive, and $r = \operatorname{rank}(X)$. Then

$$
\hat{w} = V \Sigma^{-1} U^{\top} y = X^{+} y ,
$$

where $X^{+}$ is the Moore-Penrose pseudoinverse. Expanding the sum makes the
conditioning visible term by term:

$$
\hat{w} = \sum_{i=1}^{r} \frac{u_i^{\top} y}{\sigma_i} \, v_i .
$$

Each direction $v_i$ is weighted by $1/\sigma_i$. A small singular value
multiplies whatever noise happens to project onto $u_i$ by a large factor, which
is exactly the instability that ridge regression will damp in lesson 4 by
replacing $1/\sigma_i$ with $\sigma_i / (\sigma_i^2 + \lambda)$.

### Rank deficiency

When $\operatorname{rank}(X) = r < d$, the normal equations still hold but no
longer pin down $\hat{w}$. The solution set is an affine subspace

$$
\{\, \hat{w}_0 + v : v \in \operatorname{null}(X) \,\} ,
$$

of dimension $d - r$. Every element gives the same fitted values and the same
training loss, so the data cannot distinguish them.

The pseudoinverse selects one: $X^{+}y$ is the element of smallest Euclidean
norm, because it puts zero weight on every null-space direction by construction.
That is a choice, not a consequence — and it is the same choice that gradient
descent from $w_0 = 0$ makes on its own, which is the first implicit
regularisation result you will meet.

Numerically, rank is decided by a tolerance: singular values below
$\max(n, d) \cdot \varepsilon \cdot \sigma_{\max}$ are treated as zero. That
threshold is a modelling decision disguised as a numerical one.

## Cost and scaling

The three routes agree in exact arithmetic and differ in flops and in the digits
they keep. The following table assumes $n \geq d$ and counts leading terms.

| Route | Flops | Extra memory | Digits lost |
|---|---|---|---|
| Normal equations with Cholesky | $nd^2 + \tfrac{1}{3}d^3$ | $d^2$ | $2\log_{10}\kappa_2(X)$ |
| Householder QR | $2nd^2 - \tfrac{2}{3}d^3$ | $d^2$ | $\log_{10}\kappa_2(X)$ |
| SVD | $\approx 6nd^2 + 20d^3$ | $nd$ | $\log_{10}\kappa_2(X)$ |

Cholesky on the Gram matrix is the cheapest by a factor of two and the only one
that never needs to touch $X$ again, which matters when $n$ is large enough that
$X$ does not fit in memory: you can accumulate $X^{\top}X$ and $X^{\top}y$ in a
single streaming pass in $O(d^2)$ space. That is a real advantage, and it is why
the Gram route survives in practice for well-conditioned, low-dimensional
problems where you also standardise the columns first.

For $n \gg d$ every route is $O(nd^2)$, so the constant is what separates them.
When $d$ approaches $n$, the $d^3$ terms dominate and the SVD becomes expensive
enough that iterative methods take over — the subject of lesson 3.

## Where this breaks

**Correlated features.** Near-collinearity makes $\sigma_{\min}$ small without
making it zero. The fit is fine and the coefficients are meaningless: they have
enormous variance, $\operatorname{Var}(\hat{w}) = \sigma^2 (X^{\top}X)^{-1}$,
whose trace is $\sigma^2 \sum_i \sigma_i^{-2}$. Two features that measure nearly
the same thing produce a large positive coefficient and a large negative one that
cancel. Ridge regression is the standard response.

**Unscaled columns.** A column in different units changes $\kappa_2(X)$ without
changing the model at all. Standardise before fitting, and remember that the
intercept must be excluded from the penalty once you add one.

**More features than samples.** With $d > n$ the problem is always rank
deficient, the training residual is zero, and the choice among solutions is the
entire modelling content. This is the regime modern models live in, and the
minimum-norm solution is the reason gradient descent generalises there at all.

**Heavy-tailed noise.** Squared loss is the maximum likelihood estimator under
Gaussian noise, and nothing else. A single outlier moves $\hat{w}$ without bound,
because the influence of a point grows linearly in its residual. Huber loss
bounds that influence and gives up the closed form.

## The exercise

You implement least squares three times and measure the difference the
factorisation makes.

Write `fit_normal_equations`, `fit_qr`, and `fit_svd`, each taking a design
matrix `X` and targets `y` and returning the coefficient vector. Build them from
`numpy.linalg` primitives — `cholesky`, `qr`, `svd`, and the triangular solve —
rather than calling `numpy.linalg.lstsq`, which would do the whole exercise.
`fit_svd` takes an `rcond` tolerance and must return the minimum-norm solution
when the design matrix is rank deficient.

Then write `residual_orthogonality`, which returns
$\lVert X^{\top}(y - X\hat{w}) \rVert_\infty$ for a fitted `w`, and
`condition_number`, which returns $\kappa_2(X)$ from the singular values.

The checks verify the three routes agree to high precision on a well-conditioned
problem, that the residual is orthogonal to the column space, that `fit_svd`
returns the minimum-norm element on a deliberately rank-deficient design, and
that on a Hilbert-like matrix with $\kappa_2 \approx 10^{9}$ the normal-equations
route loses the digits the table predicts while QR and the SVD keep them. The
last check is the point of the lesson: it fails if your `fit_normal_equations` is
secretly calling something stable.
