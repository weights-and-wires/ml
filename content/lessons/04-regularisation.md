---
title: Ridge, lasso, and the proximal view
slug: 04-regularisation
part: "Part 1 — Foundations"
summary: Ridge as shrinkage in the SVD basis, the lasso's kink at zero, and soft thresholding as the operation that makes both fast to solve.
minutes: 55
gpu: false
objectives:
  - State the ridge estimator, read it as shrinkage in the SVD basis, and compute its effective degrees of freedom.
  - State the subdifferential of the l1 norm, and the KKT condition that makes the lasso solution exactly sparse.
  - State soft thresholding as the proximal operator of the l1 norm, and the rates O(1/k) and O(1/k^2) for ISTA and FISTA.
  - Explain why the penalised and constrained forms of ridge and lasso are equivalent, and what each buys over unregularised least squares.
exercise: 04-regularisation
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
---

# Ridge, lasso, and the proximal view

Lesson 2 ended on the case where $X$ is close to rank deficient, and least squares stops pinning down a unique answer — the pseudoinverse's minimum-norm choice was doing all the work silently. Regularisation makes that choice explicit, and it helps even when $X$ has full rank, by trading a controlled amount of bias for a large reduction in variance: the exact trade the bias-variance decomposition in lesson 1 quantified.

Ridge and the lasso both penalise the size of $w$, with two different norms, and the difference is not cosmetic. Ridge stays a linear system, solvable in the SVD basis lesson 2 already built. The lasso's penalty is not differentiable at zero — which is exactly the point, since that kink is what drives coefficients to land at exactly zero rather than merely small. Handling the kink correctly needs one new operation, the proximal operator, and a new family of algorithms built on it.

## What you already know

- The SVD $X = U\Sigma V^\top$ from lesson 2, and the pseudoinverse $\hat{w} = V\Sigma^{-1}U^\top y$ it gives for unregularised least squares.
- $L$-smoothness and the descent lemma from lesson 3: $f(v) \leq f(u) + \nabla f(u)^\top(v-u) + \tfrac{L}{2}\lVert v-u\rVert^2$.
- A convex function's subdifferential at $w$ is the set of vectors $g$ with $f(v) \geq f(w) + g^\top(v-w)$ for every $v$; at a differentiable point it is the singleton $\{\nabla f(w)\}$.
- The maximum a posteriori (MAP) estimate maximises $p(w \mid y) \propto p(y \mid w)\, p(w)$ over $w$.
- Minimising $f(w)$ subject to $g(w) \leq t$ is, for convex $f$ and $g$, equivalent for some $\lambda \geq 0$ to minimising $f(w) + \lambda g(w)$ unconstrained.

## Ridge as MAP under a Gaussian prior

Ridge regression solves

$$
\hat{w}_{\text{ridge}} = \operatorname*{arg\,min}_{w} \; \tfrac{1}{2}\lVert Xw - y \rVert^2 + \frac{\lambda}{2}\lVert w \rVert^2 .
$$

This is a posterior mode, not an arbitrary penalty. Model $y = Xw + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, and place a prior $w \sim \mathcal{N}(0, \tau^2 I)$. Maximising the log posterior over $w$ is minimising $\tfrac{1}{2}\lVert Xw-y\rVert^2 + \tfrac{\sigma^2}{2\tau^2}\lVert w\rVert^2$, the ridge objective with $\lambda = \sigma^2/\tau^2$. A tighter prior — smaller $\tau^2$, more confidence that $w$ sits near zero — is a larger $\lambda$, and $\lambda$ is a ratio of two variances, not a free knob.

Setting the gradient to zero gives $(X^\top X + \lambda I)\hat{w}_{\text{ridge}} = X^\top y$. Unlike the unregularised normal equations, $X^\top X + \lambda I$ is positive definite for any $\lambda > 0$ regardless of the rank of $X$, so the solution is always unique — the first thing ridge buys you.

## The SVD-basis solution and shrinkage

Substituting $X = U\Sigma V^\top$ into the ridge normal equations and solving the resulting diagonal system gives the solution directly in the SVD basis:

$$
\hat{w}_{\text{ridge}} = \sum_{i=1}^{r} \frac{\sigma_i}{\sigma_i^2 + \lambda} (u_i^\top y)\, v_i .
$$

Compare this to the unregularised $\hat{w} = \sum_i \sigma_i^{-1}(u_i^\top y) v_i$ from lesson 2: every coefficient along $v_i$ is shrunk by the factor $\sigma_i^2/(\sigma_i^2+\lambda) \in (0, 1)$. Take a direction with $\sigma_i = 10$ against $\lambda = 1$: the factor is $100/101 \approx 0.99$, barely any shrinkage. Take $\sigma_i = 0.1$ against the same $\lambda$: the factor drops to about $0.01$, almost all the way to zero. Ridge damps exactly the directions that made $1/\sigma_i$ explode in lesson 2.

Summing the shrinkage factors gives the **effective degrees of freedom**,

$$
\operatorname{df}(\lambda) = \sum_{i=1}^r \frac{\sigma_i^2}{\sigma_i^2+\lambda} ,
$$

which decreases continuously from $r$ at $\lambda=0$ to $0$ as $\lambda \to \infty$. It plays the role the integer parameter count $d$ played in lesson 1's bias-variance trade-off, except the model's effective complexity is now continuous in $\lambda$ and reads directly off the singular values, with no refitting needed.

## The lasso and its subdifferential

Replace the squared penalty with the $\ell_1$ norm, $\lVert w \rVert_1 = \sum_j \lvert w_j \rvert$:

$$
\hat{w}_{\text{lasso}} = \operatorname*{arg\,min}_{w} \; \tfrac{1}{2}\lVert Xw-y \rVert^2 + \lambda \lVert w \rVert_1 .
$$

$\lVert w \rVert_1$ is convex but not differentiable where any $w_j = 0$. Its subdifferential at a scalar $w_j$ is $\{\operatorname{sign}(w_j)\}$ when $w_j \neq 0$, and the whole interval $[-1, 1]$ when $w_j = 0$ — a flat corner has room for many supporting lines, not just one. The objective is convex, so $\hat{w}$ minimises it exactly when, for every coordinate $j$,

$$
X_j^\top (X\hat{w} - y) = -\lambda\, \operatorname{sign}(\hat{w}_j) \quad \text{if } \hat{w}_j \neq 0, \qquad
\lvert X_j^\top(X\hat{w}-y)\rvert \leq \lambda \quad \text{if } \hat{w}_j = 0 ,
$$

where $X_j$ is the $j$th column. That second line is an inequality, not an equality, and it is the entire mechanism behind sparsity: a coordinate whose correlation with the residual has magnitude below $\lambda$ can sit at exactly zero and still be optimal, because the subdifferential of $\lvert \cdot \rvert$ at zero has room to absorb any gradient up to $\lambda$. Ridge's smooth penalty has no such room, since its stationarity condition is an equality everywhere — which is why ridge shrinks every coefficient without ever zeroing one out exactly.

## Soft thresholding as a proximal operator

Split the lasso objective into a smooth term $g(w) = \tfrac{1}{2}\lVert Xw-y\rVert^2$, with $L = \lambda_{\max}(X^\top X)$ as in lesson 3, and a nonsmooth term $h(w) = \lambda \lVert w \rVert_1$. The **proximal operator** of $h$ with step $\eta$ is

$$
\operatorname{prox}_{\eta h}(v) = \operatorname*{arg\,min}_{w} \; \frac{1}{2\eta}\lVert w - v \rVert^2 + h(w) .
$$

For $h(w) = \lambda\lVert w\rVert_1$ this separates coordinate by coordinate, and balancing the pull of $(w_j - v_j)^2$ toward $v_j$ against the pull of $\lambda\lvert w_j\rvert$ toward zero gives

$$
\operatorname{prox}_{\eta \lambda \lVert \cdot \rVert_1}(v)_j
= \operatorname{sign}(v_j) \max(\lvert v_j \rvert - \eta\lambda,\, 0) =: S_{\eta\lambda}(v_j) .
$$

$S_t$ is the **soft-thresholding operator**: it moves every value toward zero by $t$, and clips anything that would cross zero to exactly zero. With $t = 1$, an input of $3$ maps to $2$, an input of $0.5$ maps to $0$, and an input of $-3$ maps to $-2$. It is the lasso's analogue of ridge's shrinkage factor $\sigma_i^2/(\sigma_i^2+\lambda)$: both shrink, but soft thresholding also zeros out, because it is the proximal operator of a norm with a corner at the origin rather than a smooth quadratic.

## ISTA and FISTA

Proximal gradient descent minimises $g(w) + h(w)$ by alternating a gradient step on the smooth part with a proximal step on the rest:

$$
w_{k+1} = \operatorname{prox}_{\eta h}\big(w_k - \eta \nabla g(w_k)\big) .
$$

For the lasso this is $w_{k+1} = S_{\eta\lambda}\big(w_k - \eta X^\top(Xw_k - y)\big)$, called **ISTA**, the iterative shrinkage-thresholding algorithm. Whenever $\eta \leq 1/L$, the gradient step alone cannot increase $g$ by more than the proximal step can decrease $h$, so ISTA's objective never increases, and it inherits gradient descent's $O(1/k)$ convergence in objective value.

**FISTA** applies Nesterov's extrapolation from lesson 3 to the same update, evaluating the gradient step at an extrapolated point $z_k$ instead of at $w_k$:

$$
z_{k+1} = w_{k+1} + \frac{t_k - 1}{t_{k+1}}(w_{k+1} - w_k), \qquad
t_{k+1} = \frac{1 + \sqrt{1+4t_k^2}}{2}, \qquad t_0 = 1 ,
$$

with $w_{k+1} = S_{\eta\lambda}(z_k - \eta\nabla g(z_k))$. The sequence $t_k$ grows like $k/2$, and Nesterov's estimate-sequence argument, extended from the smooth case to this composite objective, gives $O(1/k^2)$ convergence — the same jump from linear to accelerated rates that momentum bought in lesson 3, now for a nonsmooth objective. Unlike ISTA, FISTA is not guaranteed to decrease every single step; the extrapolation can overshoot locally even while the overall rate is faster.

**Coordinate descent** offers a third route specific to the lasso: cycle through coordinates, holding every other one fixed, and solve each one-dimensional problem exactly. That one-dimensional problem is itself a soft-thresholding operation, applied to a partial residual rather than a full gradient step, with no step size to tune.

## Constrained and penalised forms

The penalised problem $\min_w \tfrac{1}{2}\lVert Xw-y\rVert^2 + \lambda \lVert w \rVert_p$ and the constrained problem $\min_{\lVert w \rVert_p \leq t} \tfrac{1}{2}\lVert Xw-y\rVert^2$ trace out the same curve of solutions as $\lambda$ or $t$ vary, for $p \in \{1, 2\}$: every $t$ on the constraint boundary corresponds to some $\lambda \geq 0$ giving the same minimiser, and conversely, though the correspondence has no closed form. Both stay in use because $\lambda$ is what an optimiser tunes, while $t$ reads directly as a budget on coefficient size.

## Cost and scaling

| Method | Per-iteration cost | Iterations for objective accuracy $\varepsilon$ |
|---|---|---|
| Ridge, closed form | One $d \times d$ solve, $O(nd^2 + d^3)$ | Exact, one solve |
| Ridge path over $m$ values of $\lambda$, from one SVD | $O(nd^2)$ once, then $O(dr)$ per $\lambda$ | Exact, $m$ cheap evaluations |
| ISTA | One matrix-vector product each way, $O(nd)$ | $O(1/\varepsilon)$ |
| FISTA | One matrix-vector product each way, $O(nd)$ | $O(1/\sqrt{\varepsilon})$ |
| Coordinate descent, one pass | $O(nd)$ for the full pass | Problem-dependent, often a small constant number of passes |

`ridge_svd_path` exists separately from `ridge_closed_form` because of the second row: the SVD costs the same $O(nd^2)$ as one closed-form solve, and every additional $\lambda$ after that costs only $O(dr)$, a rescaling by singular value rather than a fresh solve. Cross-validating over fifty values of $\lambda$ is fifty cheap rescalings, not fifty solves.

## Where this breaks

**Correlated features split the credit arbitrarily.** When two columns of $X$ are highly correlated, the lasso's optimum is not unique in which one receives the nonzero coefficient — the KKT condition can be satisfied by either, or by a split. Ridge's strictly convex penalty spreads the coefficient evenly between them instead. The elastic net, a mix of both penalties, is the standard response.

**The lasso's variable selection is not automatically consistent.** Recovering the exact support of a sparse true $w$ needs an incoherence condition on $X$ beyond what makes least squares well posed. Without it, the lasso can include irrelevant variables or exclude relevant ones even as $n \to \infty$.

**A fixed step size needs $L$.** Both ISTA and FISTA need $\eta \leq 1/L$ with $L = \lambda_{\max}(X^\top X)$. Computing $L$ exactly for large $d$ costs as much as the SVD path above, so practical implementations estimate it with a few power-iteration steps instead.

**Neither penalty is scale-invariant.** Both $\lVert w \rVert_2^2$ and $\lVert w \rVert_1$ treat every coordinate identically, so a feature's units determine how much penalty it absorbs. Standardising columns before fitting, already necessary for conditioning in lesson 2, is required here too.

## The exercise

You build both regularisers along the two axes this lesson opened: the ridge path computed cheaply from one factorisation, and the lasso solved by proximal gradient descent with and without acceleration.

Write `ridge_closed_form(X, y, lam)` as a direct solve, then `ridge_svd_path(U, s, Vt, y, lambdas)`, which reuses one SVD to return the ridge solution at every requested $\lambda$, and `effective_degrees_of_freedom(s, lam)`. Write `soft_threshold(x, thresh)`, the proximal operator derived above, then `ista(X, y, lam, step, iters, w0)` and `fista(X, y, lam, step, iters, w0)`, each returning the final coefficient vector alongside the objective value at every iteration.

The checks confirm `ridge_closed_form` against an independently constructed augmented least-squares solve, and confirm `ridge_svd_path` reproduces it at every $\lambda$ in a path rather than only at one. They confirm `soft_threshold` is the exact proximal operator by comparing it against a fine grid search over the scalar objective it minimises, and confirm ISTA's objective never increases while FISTA's falls faster than ISTA's over the same number of steps on an ill-conditioned design — the $O(1/k)$ versus $O(1/k^2)$ gap made visible on one specific problem. The last check verifies the point of the lasso section: that the coefficient vector `fista` returns satisfies the KKT stationarity condition derived above, at every coordinate, zero or not.
