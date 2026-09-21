---
title: Logistic regression
slug: 05-logistic-regression
part: "Part 1 — Foundations"
summary: Fit a Bernoulli likelihood by gradient descent and Newton's method, and see why separable data breaks it.
minutes: 40
gpu: false
objectives:
  - State the logistic loss, its gradient, and its Hessian, and explain what each means for fitting.
  - Evaluate the loss through log-sum-exp and explain why the naive formula overflows.
  - Explain why Newton's method converges in far fewer steps than gradient descent on this loss.
  - Explain why linearly separable data sends the maximum likelihood estimate to infinity, and how a ridge penalty fixes it.
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
exercise: 05-logistic-regression
---

# Logistic regression

Linear regression assumes the target is a noisy linear function of the
features. That assumption breaks for a label in $\{0, 1\}$: nothing keeps
$Xw$ between $0$ and $1$, let alone at one of two points. Logistic regression
keeps the linear predictor $Xw$, but routes it through a link function
before treating it as a probability, and fits the result by maximum
likelihood instead of least squares.

The resulting loss is convex, so gradient descent from lesson 3 already
solves it. This lesson adds two things gradient descent alone doesn't give
you: Newton's method, which reaches the optimum in far fewer steps, and a
failure mode gradient descent shares but doesn't warn you about — on data a
linear boundary separates exactly, the maximum likelihood estimate doesn't
exist at all.

## What you already know

- A convex function's Hessian is positive semidefinite everywhere, and the
  function is strictly convex wherever the Hessian is positive definite.
- Newton's method already appears in lesson 3 as the step that removes
  gradient descent's dependence on the Hessian's condition number.
- $X^{\top}X \succeq 0$ for any $X$, and it's positive definite exactly when
  $X$ has full column rank.
- Floating-point arithmetic overflows to infinity above about $1.8 \times
  10^{308}$ in double precision.
- The chain rule applied to a scalar function of $Xw$ gives a gradient of
  the form $X^{\top}(\text{something evaluated at each row})$.

## The model and its loss

Logistic regression models the probability of the positive class as

$$
p(y_i = 1 \mid x_i; w) = \sigma(x_i^{\top} w), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}} .
$$

$\sigma$ is the sigmoid. It squashes any real number into $(0, 1)$, so the
log-odds of the positive class, $\log \frac{p}{1-p}$, equal $x_i^{\top}w$
exactly — the model is linear in $w$ even though $p$ is bounded. Each label
is an independent Bernoulli draw with that probability, and maximising the
resulting likelihood is equivalent to minimising this loss over $w$, with
$z = Xw$:

$$
L(w) = \sum_{i=1}^{n} \Bigl[ \operatorname{softplus}(z_i) - y_i z_i \Bigr],
\qquad \operatorname{softplus}(z) = \log(1 + e^z) .
$$

This is the logistic loss. You get here by taking the negative
log-likelihood of the Bernoulli model and simplifying with
$1 - \sigma(z) = \sigma(-z)$; the algebra is mechanical and doesn't teach you
anything the formula doesn't already show.

## Evaluating the loss without overflowing

Computed as `log(1 + exp(z))`, softplus overflows once $z$ is about $709$ or
larger, because `exp(z)` alone exceeds the largest representable double.
That isn't a rare edge case: a confidently correct prediction produces
exactly this kind of $z$, so a model that's working can overflow a naive
implementation while a broken one doesn't.

The fix subtracts the larger of $0$ and $z$ before exponentiating, so the
exponent is never positive:

$$
\operatorname{softplus}(z) = \max(z, 0) + \log\bigl(1 + e^{-|z|}\bigr) .
$$

The same branch-on-sign trick stabilises $\sigma$ itself: compute
$1/(1+e^{-z})$ for $z \geq 0$, and $e^{z}/(1+e^{z})$ for $z < 0$. Either
branch only exponentiates a number that's zero or negative.

## The gradient and the Hessian

Differentiating $L$ gives a gradient with the same shape as the linear
regression residual, with $\sigma(Xw)$ in place of the linear prediction:

$$
\nabla L(w) = X^{\top}\bigl(\sigma(Xw) - y\bigr) .
$$

Each term differentiates independently because
$\operatorname{softplus}'(z) = \sigma(z)$, and the chain rule through
$z_i = x_i^{\top}w$ supplies the $x_i$. Differentiating again, using
$\sigma'(z) = \sigma(z)(1-\sigma(z))$, gives the Hessian:

$$
\nabla^2 L(w) = X^{\top} D X, \qquad D = \operatorname{diag}\bigl(p_1(1-p_1),
\ldots, p_n(1-p_n)\bigr), \quad p_i = \sigma(x_i^{\top}w) .
$$

Every diagonal entry of $D$ lies in $[0, 1/4]$, so $D \succeq 0$, and
$v^{\top}X^{\top}DXv = \sum_i p_i(1-p_i)(x_i^{\top}v)^2 \geq 0$ for any $v$.
The Hessian is positive semidefinite everywhere, so $L$ is convex and every
stationary point is a global minimum. It's positive definite, and $L$
strictly convex, when $X$ has full column rank — the same condition as
ordinary least squares. Unlike the linear regression Hessian $X^{\top}X$,
this one depends on $w$ through $D$: the loss surface is convex, not
quadratic.

## Newton's method

Newton's method takes the step that exactly minimises the local quadratic
model of $L$:

$$
w \leftarrow w - \bigl(X^{\top} D X\bigr)^{-1} X^{\top} \bigl(\sigma(Xw) - y
\bigr) .
$$

Near a minimum with a well-behaved Hessian, this converges quadratically —
each step roughly squares the number of correct digits — which is why the
exercise's Newton run finishes in single digits of iterations against the
hundreds that fixed-step gradient descent needs on the same problem. The
price is an $O(d^3)$ solve at every step, against gradient descent's
$O(nd)$. Each Newton step is also a weighted least-squares fit of a
"working response" against $X$ with weights $D$, a view called iteratively
reweighted least squares; lesson 6 builds on this equivalence directly.

## Where separable data breaks the MLE

Suppose some $w_0$ separates the classes exactly: $x_i^{\top}w_0 > 0$ for
every positive label and $x_i^{\top}w_0 < 0$ for every negative one. Scaling
$w_0$ by $t$ and letting $t \to \infty$ drives every correctly classified
logit to $\pm\infty$ in the direction that sends its loss term to zero, so
$L(tw_0) \to 0$. The infimum of $L$ is zero, but no finite $w$ attains it —
$\sigma$ never reaches exactly $0$ or $1$ — so gradient descent and Newton's
method both chase a target forever, with $\lVert w_k \rVert \to \infty$
while the loss keeps falling. Along the way $D \to 0$ and $X^{\top}DX$
approaches singularity, which is what makes the unregularised Newton solve
unreliable on data that's separable or close to it.

A ridge penalty $\frac{\lambda}{2}\lVert w \rVert_2^2$ fixes this. It makes
the Hessian $X^{\top}DX + \lambda I \succ 0$ for every $w$, regardless of
rank or separability, and it grows without bound as
$\lVert w \rVert \to \infty$, so the penalised objective has a unique,
finite minimum. That's the numerical signature the exercise checks: the
unregularised coefficient norm keeps growing as training continues on
separable data, while the ridge-penalised run's norm stays bounded.

## Cost and scaling

Let $n$ be the sample count and $d$ the feature count.

| Quantity | Cost | Notes |
|---|---|---|
| Gradient $X^{\top}(\sigma(Xw) - y)$ | $O(nd)$ | One matrix-vector product, one transpose-vector product. |
| Hessian $X^{\top}DX$ | $O(nd^2)$ | A factor of $d$ more than the gradient. |
| Newton solve $(X^{\top}DX)^{-1}g$ | $O(d^3)$ once the Hessian is formed | Via Cholesky, since the Hessian is positive definite away from separability. |
| One Newton step overall | $O(nd^2 + d^3)$ | Dominated by forming the Hessian when $n \gg d$. |
| $k$ steps of gradient descent | $O(knd)$ | Cheaper per step, but $k$ scales with the Hessian's condition number. |

Newton's method wins whenever $d$ is small enough that forming and
factorising $X^{\top}DX$ is affordable. Gradient descent remains the only
option once $d$ is large enough that it isn't.

## Where this breaks

**Separable and near-separable data.** The unregularised MLE diverges, as
shown above. In practice, coefficients grow every epoch while the
validation metric stops improving. Ridge or early stopping are the standard
fixes — both bound $\lVert w \rVert$ instead of letting it chase an
unreachable infimum.

**Severe class imbalance.** When one class is rare, most $p_i(1-p_i)$ terms
are tiny, so $X^{\top}DX$ is poorly conditioned even without separability,
and Newton's step becomes fragile in the same way. Reweighting the loss by
inverse class frequency changes $D$ without changing what the model can
represent.

**Naive loss evaluation.** Skipping log-sum-exp doesn't fail loudly — it
returns `inf` or `nan` silently, and a training loop that doesn't check for
finiteness reports a broken model as one that failed to learn.

**Newton on wide problems.** When $d$ is comparable to or larger than $n$,
forming and factorising $X^{\top}DX$ costs more than gradient descent loses
by taking more steps. First-order or quasi-Newton methods take over.

## The exercise

You implement the loss and both fitting routines, then use them to
reproduce the lesson's two central claims numerically.

Write a numerically stable `sigmoid` and a numerically stable `log_loss`
matching $L(w) = \sum_i [\operatorname{softplus}(x_i^{\top}w) - y_i
x_i^{\top}w] + \frac{\text{ridge}}{2}\lVert w \rVert^2$, both finite for
logits as large in magnitude as $800$. Write `gradient` and `hessian` from
the formulas above, then `fit_gradient_descent` and `fit_newton`, each
returning the fitted coefficients and the loss recorded before every step.

The checks verify stability at $\pm 800$, check `gradient` and `hessian`
against finite differences, confirm the Hessian is positive semidefinite,
and confirm `fit_newton` reaches the same optimum as
`fit_gradient_descent` in under ten iterations where gradient descent needs
hundreds. The last check builds a linearly separable dataset and confirms
the unregularised coefficient norm keeps growing while the ridge-penalised
run's doesn't.
