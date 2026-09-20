---
title: Exponential families and GLMs
slug: 06-exponential-families
part: "Part 1 — Foundations"
summary: The canonical exponential-family form, the log-partition function's derivatives as mean and variance, and Fisher scoring as the IRLS of the whole family.
minutes: 55
gpu: false
objectives:
  - Derive the mean and variance of a canonical exponential family from the log-partition function.
  - Tabulate the Gaussian, Bernoulli, Poisson, and exponential distributions as instances of one canonical form.
  - Explain why the canonical link makes the GLM Hessian free of the link's second derivative.
  - State the unified gradient X^T(mu - y) and identify Fisher scoring with IRLS.
prereqs:
  - 05-logistic-regression
exercise: 06-exponential-families
---

# Exponential families and GLMs

Lesson 5 derived the gradient and Hessian of logistic regression from scratch:
a sigmoid link, a Bernoulli likelihood, and a specific pair of derivatives.
Poisson regression, ordinary least squares, and gamma regression each look
like a separate derivation from that vantage point, and each would need its
own gradient, its own Hessian, and its own convexity argument.

They are not separate. The Bernoulli, Gaussian, and Poisson distributions —
along with the gamma, the exponential, and the multinomial — are instances of
one parametric family, written in a form where the mean, the variance, and
the maximum likelihood gradient all fall out of a single scalar function's
derivatives. This lesson derives that form, reads logistic regression as the
special case it always was, and extends the same fitting algorithm to any
member of the family through Fisher scoring.

## What you already know

- The Bernoulli negative log-likelihood, its gradient $X^{\top}(\sigma(Xw) -
  y)$, and its Hessian $X^{\top}DX$ with $D = \operatorname{diag}(p(1-p))$.
- Newton's method identified with iteratively reweighted least squares, and
  its local quadratic convergence rate.
- A covariance matrix is positive semidefinite, because
  $v^{\top}\operatorname{Cov}(T)v = \operatorname{Var}(v^{\top}T) \geq 0$.
- Differentiating under an integral sign is valid when the integrand and its
  derivatives are dominated by an integrable function, which holds for every
  family in this lesson.
- A convex function's Hessian is positive semidefinite everywhere.

## The canonical form

A scalar random variable $y$ belongs to a one-parameter **exponential
family** in canonical form when its density or mass function can be written

$$
p(y \mid \eta) = h(y) \exp\bigl(\eta \, T(y) - A(\eta)\bigr) ,
$$

for some base measure $h$, sufficient statistic $T$, and **natural
parameter** $\eta$. The function $A$ is the **log-partition function**: it is
whatever normalises the density to integrate to one,

$$
A(\eta) = \log \int h(y) \, e^{\eta T(y)} \, dy ,
$$

so $\eta$ ranges over the set where that integral converges. Different
members of the family are distinguished by their choice of $T$ and $h$; the
rest of this lesson shows that $A$ alone determines the mean, the variance,
and every derivative a fitting algorithm needs.

## The log-partition function's derivatives

Differentiate $A$ under the integral sign. Writing $Z(\eta) = e^{A(\eta)}$ for
the normalising constant,

$$
A'(\eta) = \frac{Z'(\eta)}{Z(\eta)}
= \frac{\int h(y) \, T(y) \, e^{\eta T(y)} \, dy}{\int h(y) \, e^{\eta T(y)} \, dy}
= \mathbb{E}_{\eta}[T(y)] .
$$

Differentiate again, using the quotient rule on the same ratio:

$$
A''(\eta) = \frac{\int h(y) T(y)^2 e^{\eta T(y)} dy}{Z(\eta)}
- \left( \frac{\int h(y) T(y) e^{\eta T(y)} dy}{Z(\eta)} \right)^2
= \mathbb{E}_{\eta}[T(y)^2] - \mathbb{E}_{\eta}[T(y)]^2
= \operatorname{Var}_{\eta}\bigl[T(y)\bigr] .
$$

The same computation in several natural parameters at once gives the vector
form: $\nabla A(\eta) = \mathbb{E}[T(y)]$ and $\nabla^2 A(\eta) =
\operatorname{Cov}[T(y)]$. A covariance matrix is always positive
semidefinite, so $\nabla^2 A(\eta) \succeq 0$ everywhere $A$ is finite: **the
log-partition function is convex**, for every member of the family, with no
case-by-case argument required.

## Four instances

Every family below has $T(y) = y$, so $A'(\eta)$ is directly the mean $\mu$
and $A''(\eta)$ is directly the variance. The **canonical link** is the map
from $\mu$ to $\eta$, that is, the functional inverse of $\mu = A'(\eta)$.

| Family | $\eta$ | $A(\eta)$ | $\mathbb{E}[y] = A'(\eta)$ | $\operatorname{Var}[y] = A''(\eta)$ | Canonical link |
|---|---|---|---|---|---|
| Gaussian (unit variance) | $\mu$ | $\eta^2/2$ | $\eta$ | $1$ | identity |
| Bernoulli | $\log\frac{p}{1-p}$ | $\log(1+e^{\eta})$ | $\sigma(\eta)$ | $\sigma(\eta)(1-\sigma(\eta))$ | logit |
| Poisson | $\log \lambda$ | $e^{\eta}$ | $e^{\eta}$ | $e^{\eta}$ | log |
| Exponential | $-\lambda$ | $-\log(-\eta)$ | $-1/\eta$ | $1/\eta^2$ | $\eta = -1/\mu$ |

The Bernoulli row is lesson 5 verbatim: $A(\eta) = \log(1+e^{\eta}) =
\operatorname{softplus}(\eta)$ is the stable function that lesson built, its
first derivative is $\sigma(\eta)$, and its second derivative is the $D$
matrix's diagonal entry. The Poisson row says something sharper: its variance
equals its mean, for every $\lambda$, a consequence of $A(\eta) = e^{\eta}$
satisfying $A'' = A'$ that no amount of curve-fitting on Poisson data could
tell you if you had not derived it. Writing $\mu = A'(\eta)$ and inverting
gives the variance directly as a function of the mean — the **variance
function** $V(\mu)$ — which is $1$, $\mu(1-\mu)$, $\mu$, and $\mu^2$ down the
table.

## GLMs and the canonical link

A **generalised linear model** ties the natural parameter of observation $i$
to a linear predictor through a link function $g$: $g(\mu_i) = x_i^{\top}w$.
Choosing $g$ to be the canonical link — the same function that maps $\mu$ to
$\eta$ inside the family itself — collapses this to

$$
\eta_i = x_i^{\top} w ,
$$

the natural parameter equal to the linear predictor with nothing in between.
That equality is what makes the next two derivatives short.

## The score and the unified gradient

The log-likelihood of one observation, natural parameters and canonical
link, is $\ell_i(w) = \eta_i y_i - A(\eta_i) + \log h(y_i)$ with
$\eta_i = x_i^{\top}w$. Differentiating,

$$
\frac{\partial \ell_i}{\partial w} = \bigl(y_i - A'(\eta_i)\bigr) x_i
= (y_i - \mu_i) \, x_i ,
$$

using $\partial \eta_i/\partial w = x_i$ and $A'(\eta_i) = \mu_i$. Summing
over the data and negating to work with the loss instead of the
log-likelihood,

$$
\nabla L(w) = X^{\top}(\mu - y) , \qquad \mu = A'(Xw) \text{ elementwise} ,
$$

the **score function** up to sign. This is the same formula for every family
in the table: plug in $\mu = \sigma(Xw)$ and it is lesson 5's logistic
gradient; plug in $\mu = e^{Xw}$ and it is the Poisson regression gradient;
plug in $\mu = Xw$ and it is the least-squares gradient of lesson 2. The
apparent differences between fitting a linear regression, a logistic
regression, and a Poisson regression are entirely inside $A$, and never in
the shape of the update.

## The Hessian, and why the canonical link is free of a term

Differentiate once more:

$$
\frac{\partial^2 \ell_i}{\partial w \, \partial w^{\top}}
= -A''(\eta_i) \, x_i x_i^{\top} = -V(\mu_i) \, x_i x_i^{\top} ,
$$

so the Hessian of the loss is $\nabla^2 L(w) = X^{\top} D X$ with $D =
\operatorname{diag}(V(\mu_1), \dots, V(\mu_n))$ — again lesson 5's formula,
with $p(1-p)$ generalised to $V(\mu)$.

Compare this with a non-canonical link, where $\eta_i = \psi(g^{-1}(x_i^{\top}
w))$ for $\psi = (A')^{-1}$ and a general $g$. The chain rule through
$g^{-1}$ introduces a factor $d\eta_i / d(x_i^{\top}w)$ into the gradient, and
differentiating that factor again introduces a second one, proportional to
the link's second derivative, into the Hessian. The canonical link sets
$\eta_i = x_i^{\top}w$ directly, so this term is identically zero: $d^2\eta_i
/ d(x_i^{\top}w)^2 = 0$ because the relationship is linear. That is the
concrete sense in which the canonical link is the link that makes the Hessian
depend on the data and the current fit alone, never on a derivative of the
link function itself.

A further consequence: $\nabla^2 L(w)$ above does not depend on $y$ at all,
only on $\mu$, which is a function of $w$ and $X$. The **observed
information** $-\nabla^2 \ell(w)$ and the **Fisher information**
$\mathbb{E}[-\nabla^2 \ell(w)]$ therefore coincide exactly under the
canonical link — there is nothing left to average over.

## Fisher scoring

**Fisher scoring** is Newton's method with the observed Hessian replaced by
the expected one:

$$
w \leftarrow w + I(w)^{-1} U(w) , \qquad
U(w) = X^{\top}(y - \mu), \quad I(w) = X^{\top} D X .
$$

Because the two informations coincide under the canonical link, Fisher
scoring and Newton's method are the same iteration here, and by the argument
in lesson 5, that iteration is IRLS: each step is a weighted least-squares
regression with weights $D$ and working response $z^{\ast} = Xw + D^{-1}(y -
\mu)$. Logistic regression's Newton step, Poisson regression's Fisher
scoring step, and ordinary least squares's one-shot normal equations are the
same algorithm, evaluated at the family-specific $\mu$ and $D$.

Fisher scoring inherits Newton's local quadratic rate under the same
conditions — a Lipschitz, nonsingular Hessian near the optimum — and the same
per-step cost, $O(nd^2)$ to form $X^{\top}DX$ plus $O(d^3)$ to solve it. For
the Poisson family, $D = \operatorname{diag}(e^{Xw})$ is strictly positive
for every finite $w$, since $V(\mu) = \mu > 0$ always: Poisson regression has
no separability failure mode, because no finite linear predictor drives a
Poisson mean to a value the family cannot represent.

## Cost and scaling

Let $n$ be the number of observations and $d$ the number of features, as in
lesson 5.

| Quantity | Cost | Notes |
|---|---|---|
| $A(\eta)$, $A'(\eta)$, $A''(\eta)$ elementwise | $O(n)$ | A closed form per family; no iteration. |
| Score $X^{\top}(y-\mu)$ | $O(nd)$ | Identical shape across every family. |
| Fisher information $X^{\top}DX$ | $O(nd^2)$ | Family enters only through $D$. |
| One Fisher scoring step | $O(nd^2 + d^3)$ | Same as one Newton step in lesson 5. |

Nothing in this table changes across families: the cost of fitting a Poisson
regression is, to leading order, the cost of fitting a logistic regression on
data of the same shape.

## Where this breaks

**Overdispersion.** Real count data routinely has variance exceeding its
mean, which the Poisson's fixed $V(\mu) = \mu$ cannot represent. Practitioners
either move to a negative binomial family, whose extra parameter frees the
variance from the mean, or use a quasi-likelihood that keeps the Poisson
score equation but scales $D$ by an estimated dispersion.

**Non-canonical links.** A practitioner sometimes wants a link other than the
canonical one — a probit link for binary data, for instance. The score
equation and Hessian both pick up the extra chain-rule terms this lesson
dropped, the observed and expected information no longer coincide, and
Fisher scoring and Newton's method genuinely differ. Software that assumes
the canonical link's simplification will silently compute the wrong Hessian
for these models.

**Perfect prediction in bounded families.** The Bernoulli's separability
failure from lesson 5 is really a statement about $V(\mu) \to 0$ as $\mu$
approaches the boundary of its support. Any family whose variance function
vanishes at a finite mean, not only the Bernoulli, inherits the same
divergence, and the same ridge fix.

**Unmodelled dispersion in the Gaussian case.** Fixing the Gaussian variance
at $1$ throughout this lesson hides a free parameter that real data does not
share; a GLM fit that ignores dispersion reports standard errors that are
wrong by a factor of $\sqrt{\hat\sigma^2}$.

## The exercise

You implement the log-partition function and its first two derivatives for
three families, then the GLM machinery on top, assuming the canonical link
throughout.

Write `log_partition(family, eta)`, `mean_from_natural(family, eta)`, and
`variance_from_natural(family, eta)` for `"gaussian"`, `"bernoulli"`, and
`"poisson"`. Then write `glm_gradient`, `glm_fisher_information`, and
`fit_fisher_scoring`, which fits a canonical-link GLM starting from $w = 0$
and reports the coefficients and the loss recorded before each step.

The checks differentiate your `log_partition` numerically and compare against
your `mean_from_natural` and `variance_from_natural`, confirm the Poisson
variance function equals its mean function, verify `glm_gradient` and
`glm_fisher_information` against finite differences of the Poisson
log-likelihood, confirm `fit_fisher_scoring` recovers known coefficients on
simulated Poisson counts in under ten iterations, and confirm that for
`family="bernoulli"` the very first Fisher scoring step matches, to machine
precision, the manually computed logistic Newton step from lesson 5 — the
numerical proof that the two lessons describe one algorithm.
