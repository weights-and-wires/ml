---
title: Exponential families and GLMs
slug: 06-exponential-families
part: "Part 1 — Foundations"
summary: One log-partition function gives the mean, the variance, and the fitting algorithm for a whole family of models.
minutes: 40
gpu: false
objectives:
  - State the canonical exponential-family form and identify its log-partition function.
  - Read the mean and the variance of a family directly off the log-partition function's derivatives.
  - Explain why the canonical link keeps the GLM Hessian free of the link's own second derivative.
  - State the unified GLM gradient X^T(mu - y), and identify Fisher scoring with Newton's method under the canonical link.
prereqs:
  - 05-logistic-regression
exercise: 06-exponential-families
---

# Exponential families and GLMs

Lesson 5 derived the gradient and Hessian of logistic regression from
scratch: a sigmoid link, a Bernoulli likelihood, and a specific pair of
derivatives. Poisson regression, ordinary least squares, and gamma
regression each look like a separate derivation from that vantage point,
each needing its own gradient, its own Hessian, and its own convexity
argument.

They aren't separate. The Bernoulli, Gaussian, and Poisson distributions are
instances of one parametric family, written in a form where the mean, the
variance, and the maximum likelihood gradient all fall out of a single
scalar function's derivatives. This lesson states that form, reads logistic
regression as the special case it always was, and extends the same fitting
algorithm to any member of the family.

## What you already know

- The Bernoulli negative log-likelihood, its gradient
  $X^{\top}(\sigma(Xw) - y)$, and its Hessian $X^{\top}DX$ with
  $D = \operatorname{diag}(p(1-p))$.
- Newton's method and its local quadratic convergence rate.
- A covariance matrix is positive semidefinite, because
  $v^{\top}\operatorname{Cov}(T)v = \operatorname{Var}(v^{\top}T) \geq 0$.
- A convex function's Hessian is positive semidefinite everywhere.

## The canonical form

A scalar random variable $y$ belongs to a one-parameter exponential family
in canonical form when its density or mass function can be written

$$
p(y \mid \eta) = h(y) \exp\bigl(\eta \, T(y) - A(\eta)\bigr) ,
$$

for some base measure $h$, sufficient statistic $T$, and natural parameter
$\eta$. $A$ is the log-partition function: whatever normalises the density
to integrate to one. Different members of the family are distinguished by
their choice of $T$ and $h$; $A$ alone determines the mean, the variance,
and every derivative a fitting algorithm needs.

## The mean and variance are derivatives of A

Differentiating $A$ under the integral sign gives two facts that hold for
every member of the family, with no case-by-case argument required:

$$
A'(\eta) = \mathbb{E}_{\eta}[T(y)], \qquad
A''(\eta) = \operatorname{Var}_{\eta}\bigl[T(y)\bigr] .
$$

The mean of the sufficient statistic is the log-partition function's slope,
and its variance is the log-partition function's curvature. Because a
variance is never negative, $A'' \geq 0$ everywhere it's finite: the
log-partition function is convex for every member of the family. In several
natural parameters at once, the same identities read
$\nabla A(\eta) = \mathbb{E}[T(y)]$ and
$\nabla^2 A(\eta) = \operatorname{Cov}[T(y)] \succeq 0$.

## Three instances

Every family below has $T(y) = y$, so $A'(\eta)$ is directly the mean $\mu$
and $A''(\eta)$ is directly the variance. The canonical link is the map
from $\mu$ to $\eta$ — the functional inverse of $\mu = A'(\eta)$.

| Family | $\eta$ | $A(\eta)$ | Mean $A'(\eta)$ | Variance $A''(\eta)$ | Canonical link |
|---|---|---|---|---|---|
| Gaussian (unit variance) | $\mu$ | $\eta^2/2$ | $\eta$ | $1$ | identity |
| Bernoulli | $\log\frac{p}{1-p}$ | $\log(1+e^{\eta})$ | $\sigma(\eta)$ | $\sigma(\eta)(1-\sigma(\eta))$ | logit |
| Poisson | $\log \lambda$ | $e^{\eta}$ | $e^{\eta}$ | $e^{\eta}$ | log |

The Bernoulli row is lesson 5's `softplus` and $\sigma$, read as derivatives
of $A$ instead of derived from a likelihood directly. The Poisson row says
something sharper: its variance equals its mean, for every $\lambda$,
because $A(\eta) = e^{\eta}$ satisfies $A'' = A'$ automatically. Writing the
variance as a function of the mean gives the variance function $V(\mu)$,
which is $1$, $\mu(1-\mu)$, and $\mu$ down the table.

## GLMs and the canonical link

A generalised linear model ties the natural parameter of observation $i$ to
a linear predictor through a link function: $g(\mu_i) = x_i^{\top}w$.
Choosing $g$ to be the canonical link collapses this to
$\eta_i = x_i^{\top}w$ — the natural parameter equal to the linear
predictor, with nothing in between. That equality is what makes the
gradient and Hessian below match lesson 5's formulas exactly, for every
family.

## The gradient and the Hessian

Differentiating the log-likelihood under the canonical link, then summing
over the data and negating to work with a loss, gives one gradient for the
whole family:

$$
\nabla L(w) = X^{\top}(\mu - y), \qquad \mu = A'(Xw) \text{ elementwise} .
$$

Plug in $\mu = \sigma(Xw)$ and this is lesson 5's logistic gradient; plug in
$\mu = e^{Xw}$ and it's the Poisson regression gradient; plug in $\mu = Xw$
and it's the least-squares gradient of lesson 2. The differences between
fitting these three models live entirely inside $A$, never in the shape of
the update. Differentiating once more gives the Hessian,

$$
\nabla^2 L(w) = X^{\top}DX, \qquad D = \operatorname{diag}\bigl(V(\mu_1),
\ldots, V(\mu_n)\bigr) ,
$$

again lesson 5's formula, with $p(1-p)$ generalised to $V(\mu)$.

A non-canonical link — a probit link for binary data, for instance —
introduces an extra factor from the chain rule into both the gradient and
the Hessian, proportional to the link's own derivatives. The canonical link
sets $\eta_i = x_i^{\top}w$ directly, a linear relationship whose second
derivative is zero, so that extra term never appears. That's the concrete
sense in which the canonical link keeps the Hessian free of the link
function's own curvature, and it's also why the observed information
$-\nabla^2\ell(w)$ and the Fisher information
$\mathbb{E}[-\nabla^2\ell(w)]$ coincide exactly under this link: the
Hessian above doesn't depend on $y$ at all, so there's nothing left to
average over.

## Fisher scoring

Fisher scoring is Newton's method with the observed Hessian replaced by the
expected one:

$$
w \leftarrow w + I(w)^{-1}U(w), \qquad U(w) = X^{\top}(y - \mu), \quad
I(w) = X^{\top}DX .
$$

Because the two informations coincide under the canonical link, Fisher
scoring and Newton's method are the same iteration here. Logistic
regression's Newton step, Poisson regression's Fisher scoring step, and
ordinary least squares's one-shot normal equations are one algorithm,
evaluated at the family-specific $\mu$ and $D$. Fisher scoring inherits
Newton's local quadratic rate and its $O(nd^2 + d^3)$ per-step cost. For the
Poisson family, $D = \operatorname{diag}(e^{Xw})$ is strictly positive for
every finite $w$, so Poisson regression has no separability failure mode:
no finite linear predictor drives a Poisson mean outside what the family
can represent.

## Cost and scaling

Let $n$ be the number of observations and $d$ the number of features, as in
lesson 5.

| Quantity | Cost | Notes |
|---|---|---|
| $A(\eta)$, $A'(\eta)$, $A''(\eta)$ elementwise | $O(n)$ | A closed form per family; no iteration. |
| Score $X^{\top}(y-\mu)$ | $O(nd)$ | Identical shape across every family. |
| Fisher information $X^{\top}DX$ | $O(nd^2)$ | Family enters only through $D$. |
| One Fisher scoring step | $O(nd^2 + d^3)$ | Same as one Newton step in lesson 5. |

Nothing in this table changes across families: fitting a Poisson regression
costs the same, to leading order, as fitting a logistic regression on data
of the same shape.

## Where this breaks

**Overdispersion.** Real count data routinely has variance exceeding its
mean, which the Poisson's fixed $V(\mu) = \mu$ can't represent.
Practitioners move to a negative binomial family, whose extra parameter
frees the variance from the mean, or use a quasi-likelihood that scales $D$
by an estimated dispersion.

**Non-canonical links.** A probit link, or any link other than the
canonical one, brings back the extra chain-rule terms this lesson dropped.
The observed and expected information no longer coincide, and Fisher
scoring and Newton's method genuinely differ. Software that assumes the
canonical simplification computes the wrong Hessian for these models.

**Perfect prediction in bounded families.** The Bernoulli's separability
failure from lesson 5 is a statement about $V(\mu) \to 0$ as $\mu$
approaches the boundary of its support. Any family whose variance function
vanishes at a finite mean inherits the same divergence, and the same ridge
fix.

**Unmodelled dispersion in the Gaussian case.** Fixing the Gaussian
variance at $1$ throughout this lesson hides a free parameter that real
data doesn't share; a GLM fit that ignores dispersion reports standard
errors wrong by a factor of $\sqrt{\hat\sigma^2}$.

## The exercise

You implement the log-partition function and its first two derivatives for
three families, then the GLM machinery on top, assuming the canonical link
throughout.

Write `log_partition(family, eta)`, `mean_from_natural(family, eta)`, and
`variance_from_natural(family, eta)` for `"gaussian"`, `"bernoulli"`, and
`"poisson"`. Then write `glm_gradient`, `glm_fisher_information`, and
`fit_fisher_scoring`, which fits a canonical-link GLM starting from $w = 0$
and reports the coefficients and the loss recorded before each step.

The checks differentiate your `log_partition` numerically and compare
against your `mean_from_natural` and `variance_from_natural`, confirm the
Poisson variance function equals its mean function, verify `glm_gradient`
and `glm_fisher_information` against finite differences, confirm
`fit_fisher_scoring` recovers known coefficients on simulated Poisson
counts in under ten iterations, and confirm that for `family="bernoulli"`
the first Fisher scoring step matches, to machine precision, the manually
computed logistic Newton step from lesson 5.
