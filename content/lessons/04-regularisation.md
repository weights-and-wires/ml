---
title: Ridge, lasso, and the proximal view
slug: 04-regularisation
part: "Part 1 — Foundations"
summary: Ridge as MAP shrinkage in the SVD basis, the lasso's subdifferential, and soft thresholding derived as a proximal operator.
minutes: 60
gpu: false
objectives:
  - Derive the ridge estimator as the MAP estimate under a Gaussian prior, and read its SVD-basis shrinkage factor and effective degrees of freedom.
  - Derive the subdifferential of the l1 norm and the KKT condition that makes the lasso solution exactly sparse.
  - Derive soft thresholding as the proximal operator of the l1 norm, and state the rates O(1/k) and O(1/k^2) for ISTA and FISTA.
  - Explain why the penalised and constrained forms of ridge and lasso are equivalent, and what each buys over unregularised least squares.
exercise: 04-regularisation
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
---

# Ridge, lasso, and the proximal view

Lesson 2 ended on the case $d > n$ or $X$ close to rank deficient, where least
squares stops pinning down an answer and the pseudoinverse's minimum-norm
choice is doing all the work silently. Regularisation makes that choice
explicit, and buys something even when $X$ has full rank: trading a
controlled amount of bias for a large reduction in variance, the exact
trade the bias-variance decomposition in lesson 1 quantified.

Ridge and the lasso add the same idea — penalise the size of $w$ — with two
different norms, and the difference is not cosmetic. Ridge stays a linear
system, solvable in the SVD basis lesson 2 already built. The lasso's penalty
is not differentiable at zero, which is exactly the point: it is what drives
coefficients to be exactly zero rather than merely small, and handling that
kink correctly needs a new operation, the proximal operator, and a new class
of algorithm built on it.

## What you already know

- The SVD $X = U\Sigma V^\top$ from lesson 2, and the pseudoinverse
  $\hat{w} = V\Sigma^{-1}U^\top y$ it gives for unregularised least squares.
- $L$-smoothness and the descent lemma from lesson 3:
  $f(v) \leq f(u) + \nabla f(u)^\top(v-u) + \tfrac{L}{2}\lVert v-u\rVert^2$.
- A convex function's subdifferential at $w$ is the set of vectors $g$ with
  $f(v) \geq f(w) + g^\top(v-w)$ for every $v$; at a differentiable point it
  is the singleton $\{\nabla f(w)\}$.
- The maximum a posteriori (MAP) estimate maximises
  $p(w \mid y) \propto p(y \mid w)\, p(w)$ over $w$.
- Lagrangian duality: minimising $f(w)$ subject to $g(w) \leq t$ is, for
  convex $f$ and $g$, equivalent for some $\lambda \geq 0$ to minimising
  $f(w) + \lambda g(w)$ unconstrained, with $\lambda$ and $t$ in one-to-one
  correspondence along the constraint's boundary.

## Ridge as MAP under a Gaussian prior

Ridge regression solves

$$
\hat{w}_{\text{ridge}} = \operatorname*{arg\,min}_{w} \; \tfrac{1}{2}\lVert Xw - y \rVert^2 + \frac{\lambda}{2}\lVert w \rVert^2 .
$$

This is a posterior mode. Model $y = Xw + \varepsilon$ with
$\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ and place a prior
$w \sim \mathcal{N}(0, \tau^2 I)$. The log posterior, up to a constant in $w$,
is

$$
\log p(w \mid y) = -\frac{1}{2\sigma^2}\lVert Xw - y \rVert^2 - \frac{1}{2\tau^2}\lVert w \rVert^2 + \text{const} .
$$

Maximising this is minimising
$\tfrac{1}{2}\lVert Xw-y\rVert^2 + \tfrac{\sigma^2}{2\tau^2}\lVert w\rVert^2$,
which is the ridge objective with $\lambda = \sigma^2/\tau^2$. A tighter prior
(smaller $\tau^2$, more confidence that $w$ is near zero) is a larger
$\lambda$. Ridge is not an ad hoc penalty; it is exact Bayesian inference
under a specific, symmetric belief about $w$, and $\lambda$ is a variance
ratio.

Setting the gradient of the ridge objective to zero gives
$(X^\top X + \lambda I)\hat{w}_{\text{ridge}} = X^\top y$. Unlike the
unregularised normal equations, $X^\top X + \lambda I$ is positive definite
for any $\lambda > 0$ regardless of the rank of $X$, so the solution is always
unique.

## The SVD-basis solution and shrinkage

Substitute $X = U\Sigma V^\top$ (thin SVD, $r = \operatorname{rank}(X)$ singular
values $\sigma_1, \dots, \sigma_r$) into the ridge normal equations. Using
$X^\top X = V\Sigma^2 V^\top$ and $V^\top V = I$,

$$
(V\Sigma^2 V^\top + \lambda I)\hat{w} = V\Sigma U^\top y
\;\Longrightarrow\;
V(\Sigma^2 + \lambda I)V^\top \hat{w} = V \Sigma U^\top y ,
$$

using $\lambda I = \lambda V V^\top$. Left-multiplying by $V^\top$ and solving
the resulting diagonal system for $V^\top \hat{w}$, then multiplying back by
$V$:

$$
\hat{w}_{\text{ridge}} = \sum_{i=1}^{r} \frac{\sigma_i}{\sigma_i^2 + \lambda} (u_i^\top y)\, v_i .
$$

Compare to the unregularised solution $\hat{w} = \sum_i \sigma_i^{-1}(u_i^\top
y) v_i$ from lesson 2: every coefficient along $v_i$ is shrunk by the factor
$\sigma_i^2/(\sigma_i^2+\lambda) \in (0, 1)$. A direction with a large singular
value ($\sigma_i \gg \sqrt{\lambda}$) is barely shrunk; a direction with a
small one ($\sigma_i \ll \sqrt{\lambda}$) is shrunk almost to zero. This is
the ridge fix for the instability lesson 2 identified: the directions that
made $1/\sigma_i$ explode are precisely the directions this factor damps.

Summing the shrinkage factors gives the **effective degrees of freedom**,

$$
\operatorname{df}(\lambda) = \sum_{i=1}^r \frac{\sigma_i^2}{\sigma_i^2+\lambda} ,
$$

which decreases continuously from $r$ at $\lambda=0$ (unregularised, one
degree of freedom per fitted direction) to $0$ as $\lambda \to \infty$
(everything shrunk to the constant zero fit). It plays the same role as the
integer parameter count $d$ did in lesson 1's approximation-estimation
trade-off, except now the model's effective complexity is continuous in
$\lambda$ and can be read directly off the singular values, with no
refitting.

## The lasso and its subdifferential

Replace the squared penalty with the $\ell_1$ norm:

$$
\hat{w}_{\text{lasso}} = \operatorname*{arg\,min}_{w} \; \tfrac{1}{2}\lVert Xw-y \rVert^2 + \lambda \lVert w \rVert_1 .
$$

$\lVert w \rVert_1 = \sum_j \lvert w_j \rvert$ is convex but not differentiable
at any $w_j = 0$. Its subdifferential at a scalar $w_j$ is

$$
\partial \lvert w_j \rvert =
\begin{cases}
\{\operatorname{sign}(w_j)\} & w_j \neq 0 \\
[-1, 1] & w_j = 0 .
\end{cases}
$$

The objective is convex (a sum of a convex quadratic and a convex penalty),
so $\hat{w}$ is a minimiser iff $0$ lies in the subdifferential of the whole
objective at $\hat{w}$:

$$
X_j^\top (X\hat{w} - y) + \lambda\, s_j = 0 \quad \text{for some } s_j \in \partial\lvert \hat{w}_j \rvert ,
$$

for every coordinate $j$, where $X_j$ is the $j$th column. Unpacking the two
cases: when $\hat{w}_j \neq 0$, $X_j^\top(X\hat{w}-y) = -\lambda\,
\operatorname{sign}(\hat{w}_j)$ exactly; when $\hat{w}_j = 0$, the condition
only requires $\lvert X_j^\top(X\hat{w}-y)\rvert \leq \lambda$, an inequality,
not an equality. That asymmetry is the entire mechanism behind sparsity: a
coordinate whose correlation with the residual has magnitude below $\lambda$
can sit at exactly zero and still satisfy optimality, because the
subdifferential of $\lvert \cdot \rvert$ at zero has room to absorb any
gradient of magnitude up to $\lambda$. Ridge's smooth penalty has no such
room — its stationarity condition is an equality at every coordinate, which
is why ridge shrinks every coefficient without ever zeroing one out exactly.

## Soft thresholding as a proximal operator

The lasso's objective is a sum of a smooth, $L$-smooth term
$g(w) = \tfrac{1}{2}\lVert Xw-y\rVert^2$ (with $L = \lambda_{\max}(X^\top X)$,
as in lesson 3) and a nonsmooth term $h(w) = \lambda \lVert w \rVert_1$. The
**proximal operator** of $h$ with step $\eta$ is

$$
\operatorname{prox}_{\eta h}(v) = \operatorname*{arg\,min}_{w} \; \frac{1}{2\eta}\lVert w - v \rVert^2 + h(w) .
$$

For $h(w) = \lambda \lVert w \rVert_1$ this separates coordinate by
coordinate. Fix $j$ and minimise
$\phi(w_j) = \tfrac{1}{2\eta}(w_j - v_j)^2 + \lambda \lvert w_j \rvert$. For
$w_j > 0$, $\phi'(w_j) = (w_j - v_j)/\eta + \lambda$, zero at
$w_j = v_j - \eta\lambda$, valid when $v_j > \eta\lambda$. By symmetry, for
$w_j < 0$ the minimiser is $v_j + \eta\lambda$, valid when $v_j <
-\eta\lambda$. At $w_j = 0$, $\phi$'s smooth part contributes the single
subgradient $-v_j/\eta$ and $\lambda\lvert w_j \rvert$ contributes the whole
interval $[-\lambda, \lambda]$, so $\partial\phi(0) = [-\lambda -
v_j/\eta,\, \lambda - v_j/\eta]$. Zero lies in that interval exactly when
$\lvert v_j \rvert \leq \eta\lambda$, so $w_j = 0$ is optimal on precisely
that range. Collecting the three cases:

$$
\operatorname{prox}_{\eta \lambda \lVert \cdot \rVert_1}(v)_j
= \operatorname{sign}(v_j) \max(\lvert v_j \rvert - \eta\lambda,\, 0) =: S_{\eta\lambda}(v_j) .
$$

$S_t$ is the **soft-thresholding operator**: it shifts every value toward
zero by $t$ and clips anything that crosses zero to exactly zero. It is the
lasso's exact analogue of ridge's shrinkage factor
$\sigma_i^2/(\sigma_i^2+\lambda)$ — both shrink, but soft thresholding also
zeros out, because it is a proximal operator of a norm with a corner at the
origin rather than a smooth quadratic penalty.

## ISTA and FISTA

Proximal gradient descent minimises $g(w) + h(w)$, with $g$ smooth and $h$
possibly nonsmooth, by alternating a gradient step on $g$ with a proximal
step on $h$:

$$
w_{k+1} = \operatorname{prox}_{\eta h}\big(w_k - \eta \nabla g(w_k)\big) .
$$

For the lasso this is exactly $w_{k+1} = S_{\eta\lambda}\big(w_k - \eta
X^\top(Xw_k - y)\big)$ — a gradient step on the smooth squared-error term
followed by soft thresholding — and is called **ISTA**, the iterative
shrinkage-thresholding algorithm. Because the gradient step alone satisfies
the descent lemma from lesson 3 whenever $\eta \leq 1/L$, and the proximal
step can only decrease $h$ relative to the point it is applied to, ISTA's
objective value never increases, and it inherits gradient descent's rate:
$O(1/k)$ convergence in objective value for a general convex, $L$-smooth $g$.

**FISTA** applies Nesterov's extrapolation from lesson 3 to the same update,
evaluating the gradient step at an extrapolated point $z_k$ rather than at
$w_k$:

$$
z_{k+1} = w_{k+1} + \frac{t_k - 1}{t_{k+1}}(w_{k+1} - w_k), \qquad
t_{k+1} = \frac{1 + \sqrt{1+4t_k^2}}{2}, \qquad t_0 = 1 ,
$$

with $w_{k+1} = S_{\eta\lambda}(z_k - \eta\nabla g(z_k))$. The sequence $t_k$
grows like $k/2$, and Nesterov's estimate-sequence argument — the same
argument sketched in lesson 3 for the smooth case, extended to the composite
objective $g + h$ — gives $O(1/k^2)$ convergence in objective value, matching
the improvement from linear to accelerated rates that momentum bought for
smooth strongly convex problems, now for a nonsmooth, only weakly convex one.
Unlike ISTA, FISTA's objective is not guaranteed to decrease every step; the
extrapolation can overshoot locally even while the overall rate is faster.

**Coordinate descent** offers a third route, specific to the lasso: cycle
through coordinates $j$, holding every other coordinate fixed, and solve the
one-dimensional problem in $w_j$ exactly. That one-dimensional problem is
itself a soft-thresholding operation, applied to the partial residual
$X_j^\top(y - X_{-j}w_{-j})$ rather than to a full gradient step. It has no
step size to tune and often converges in fewer passes than ISTA on problems
small enough that a full pass over coordinates is cheap, at the cost of being
inherently sequential rather than a single vectorised update.

## Constrained and penalised forms

The penalised form $\min_w \tfrac{1}{2}\lVert Xw-y\rVert^2 + \lambda \lVert w
\rVert_p$ and the constrained form $\min_{\lVert w \rVert_p \leq t}
\tfrac{1}{2}\lVert Xw-y\rVert^2$ (for $p \in \{1, 2\}$) trace out the same
curve of solutions as $\lambda$ or $t$ vary. Both objectives are convex, and
for the constrained problem the KKT conditions introduce a Lagrange
multiplier $\mu \geq 0$ on the constraint $\lVert w \rVert_p \leq t$, giving
the stationarity condition $\nabla(\tfrac{1}{2}\lVert Xw-y\rVert^2) + \mu\,
\partial\lVert w \rVert_p \ni 0$ — identical in form to the penalised
problem's stationarity condition with $\lambda = \mu$. Every $t$ on the
constraint's active boundary corresponds to some $\lambda \geq 0$ giving the
same minimiser, and conversely; the correspondence is monotonic but has no
closed form in general, which is the practical reason both parameterisations
are in use: $\lambda$ is the one an optimiser tunes, $t$ is the one that has
a direct interpretation as a budget on coefficient size.

## Cost and scaling

| Method | Per-iteration cost | Iterations for objective accuracy $\varepsilon$ |
|---|---|---|
| Ridge, closed form | One $d \times d$ solve, $O(nd^2 + d^3)$ | Exact, one solve |
| Ridge path over $m$ values of $\lambda$, from one SVD | $O(nd^2)$ once, then $O(dr)$ per $\lambda$ | Exact, $m$ cheap evaluations |
| ISTA | One matrix-vector product each way, $O(nd)$ | $O(1/\varepsilon)$ |
| FISTA | One matrix-vector product each way, $O(nd)$ | $O(1/\sqrt{\varepsilon})$ |
| Coordinate descent, one pass | $O(nd)$ for the full pass | Problem-dependent, often a small constant number of passes |

The ridge path row is the reason `ridge_svd_path` exists as a function
separate from `ridge_closed_form`: computing the SVD once costs the same
$O(nd^2)$ as a single closed-form solve, and every additional $\lambda$ after
that costs $O(dr)$ — a rank-one-per-singular-value rescaling — rather than a
fresh $O(nd^2 + d^3)$ solve. Cross-validating over fifty values of $\lambda$
is fifty cheap rescalings, not fifty solves.

## Where this breaks

**Correlated features split the credit arbitrarily.** When two columns of
$X$ are highly correlated, the lasso's optimum is not unique in which of the
two receives the nonzero coefficient — the KKT condition can be satisfied by
either, or by a split between them — while ridge, whose penalty is strictly
convex in $w$, spreads the coefficient evenly and predictably between them.
The elastic net, a mix of both penalties, is the standard response.

**The lasso's variable selection is not automatically consistent.** Recovering
the exact support of a sparse true $w$ requires an incoherence condition on
$X$ beyond what makes least squares well posed; without it, the lasso can
include irrelevant variables or exclude relevant ones even as $n \to \infty$.

**A fixed step size needs $L$.** Both ISTA and FISTA need $\eta \leq 1/L$
with $L = \lambda_{\max}(X^\top X)$; for large $d$ computing $L$ exactly costs
as much as the SVD path above, so practical implementations estimate it with
a few power-iteration steps rather than a full eigendecomposition.

**Neither penalty is scale-invariant.** Both $\lVert w \rVert_2^2$ and
$\lVert w \rVert_1$ treat every coordinate identically, so a feature's units
determine how much penalty it absorbs; standardising columns before fitting,
as lesson 2 already recommended for conditioning, is required here for the
penalty to mean the same thing across features.

## The exercise

You build both regularisers along the two axes this lesson opened: the ridge
path computed cheaply from one factorisation, and the lasso solved by
proximal gradient descent with and without acceleration.

Write `ridge_closed_form(X, y, lam)` as a direct solve, then
`ridge_svd_path(U, s, Vt, y, lambdas)`, which reuses one SVD to return the
ridge solution at every requested $\lambda$, and
`effective_degrees_of_freedom(s, lam)`. Write `soft_threshold(x, thresh)`,
the proximal operator derived above, then `ista(X, y, lam, step, iters, w0)`
and `fista(X, y, lam, step, iters, w0)`, each returning the final coefficient
vector alongside the objective value at every iteration.

The checks confirm `ridge_closed_form` against an independently constructed
augmented least-squares solve, and confirm `ridge_svd_path` reproduces it at
every $\lambda$ in a path rather than only at one. They confirm
`soft_threshold` is the exact proximal operator by comparing it against a
fine grid search over the scalar objective it minimises, and confirm ISTA's
objective never increases while FISTA's falls faster than ISTA's over the
same number of steps on an ill-conditioned design — the $O(1/k)$ versus
$O(1/k^2)$ gap made visible on one specific problem. The last check verifies
the point of the lasso section: that the coefficient vector `fista` returns
satisfies the KKT stationarity condition derived above, at every coordinate,
zero or not.
