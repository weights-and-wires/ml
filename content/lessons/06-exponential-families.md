---
title: Exponential families and GLMs
slug: 06-exponential-families
part: "Part 1 — Foundations"
summary: One function per distribution gives you its mean, its variance, and a single fitting loop for many models.
minutes: 55
gpu: false
objectives:
  - Compute the log-partition function, the mean, and the variance for the Gaussian, Bernoulli, and Poisson families.
  - Explain how the mean and the variance come from the first and second derivatives of one function.
  - Build the shared GLM gradient and Fisher information, and fit a Poisson regression by Fisher scoring.
  - Explain why Fisher scoring on Bernoulli data is the same as the Newton step from lesson 5.
prereqs:
  - 05-logistic-regression
exercise: 06-exponential-families
---

# Exponential families and GLMs

In lesson 5 you built logistic regression for yes-or-no labels. Other
targets need other models. Counting how many customers arrive in an hour
calls for Poisson regression. Predicting a temperature calls for ordinary
least squares. At first each looks like a separate model with its own loss,
its own gradient, and its own Hessian.

They are the same model in disguise. The Gaussian, Bernoulli, and Poisson
distributions all fit one template. Once a distribution is in that template,
one small function tells you its mean, its variance, and how to fit it. The
gradient and Hessian you wrote in lesson 5 then work for every family, with
one line changed.

This lesson shows the template and uses it to write one fitting loop that
handles all three families. You fit a Poisson regression with it, and check
that on Bernoulli data it repeats lesson 5 exactly.

## What you already know

- Logistic regression predicts $p = \sigma(Xw)$ and has gradient $X^{\top}(p - y)$.
- Its Hessian is $X^{\top} D X$, where $D$ holds $p_i(1 - p_i)$ on the diagonal.
- Newton's method updates $w \leftarrow w - H^{-1} g$ and converges in a few steps.
- A variance measures how spread out a random quantity is, and it is never negative.
- A Poisson distribution models counts, such as arrivals per hour, with a single rate $\lambda$.

## One template for many distributions

Think of a universal phone charger with swappable tips. The charger is the
same for every phone. Only the tip changes. The *exponential family* is the
charger, and one function, called $A$, is the tip.

For the families in this lesson, the probability of seeing the value $y$ has
this form:

$$
p(y \mid \eta) = h(y) \exp\bigl(\eta \, y - A(\eta)\bigr)
$$

Read it aloud piece by piece:

- $\eta$ (the Greek letter eta) is the *natural parameter*. It is a single
  number that sets where the distribution sits.
- $h(y)$ depends only on $y$. It never matters for fitting, so you can ignore it.
- $\eta \, y$ is the parameter times the value.
- $A(\eta)$ is the *log-partition function*. Its job is to make all the
  probabilities add up to one. Each family has its own $A$.

In the general template, $y$ in $\eta \, y$ is replaced by a function $T(y)$
called the sufficient statistic. For all three families here, $T(y) = y$, so
the lesson writes $y$ directly.

## The mean and variance are slopes of A

The template has a useful property. Take $A$ and differentiate it. The
first derivative is the mean of $y$, and the second derivative is its
variance:

$$
\mu = A'(\eta), \qquad \operatorname{Var}(y) = A''(\eta)
$$

Read it aloud: the slope of $A$ at $\eta$ is the average value of $y$, which
this lesson calls $\mu$ (mu). The curvature of $A$ at $\eta$ is how spread
out $y$ is. Since a variance is never negative, $A''$ is never negative,
so $A$ curves upward everywhere. That upward curve is why every fit in this
lesson is a bowl with a single bottom.

You can check this numerically with a central difference. In numpy, for a
small step `h`:

```python
mean_estimate = (A(eta + h) - A(eta - h)) / (2 * h)
var_estimate = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2
```

The exercise checks do exactly this against your functions.

## The three families

The following table lists $A$ and its two derivatives for each family. The
last column names the *canonical link*, the function that turns the mean
$\mu$ back into $\eta$.

| Family | $A(\eta)$ | Mean $A'(\eta)$ | Variance $A''(\eta)$ | Canonical link |
|---|---|---|---|---|
| Gaussian, variance 1 | $\eta^2 / 2$ | $\eta$ | $1$ | identity: $\eta = \mu$ |
| Bernoulli | $\log(1 + e^{\eta})$ | $\sigma(\eta)$ | $\sigma(\eta)(1 - \sigma(\eta))$ | logit: $\eta = \log\frac{\mu}{1 - \mu}$ |
| Poisson | $e^{\eta}$ | $e^{\eta}$ | $e^{\eta}$ | log: $\eta = \log \mu$ |

Read each row aloud:

- **Gaussian.** $A$ is a parabola. Its slope is $\eta$, so the mean is
  $\eta$. Its curvature is 1, so the variance is always 1.
- **Bernoulli.** $A$ is the softplus from lesson 5. Its slope is the sigmoid,
  so the mean is the probability of a 1. Its curvature is $p(1 - p)$, the
  same weights that filled $D$ in lesson 5.
- **Poisson.** $A$ is $e^{\eta}$, which is its own derivative. So the mean
  and the variance are both $e^{\eta}$. A Poisson's variance always equals
  its mean.

In numpy, each row is one or two lines. For the Bernoulli family, reuse the
stable softplus and sigmoid from lesson 5:

```python
A = np.maximum(eta, 0) + np.log1p(np.exp(-np.abs(eta)))  # Bernoulli
mu = np.exp(eta)                                          # Poisson mean
```

Writing the variance as a function of the mean gives the *variance
function* $V(\mu)$. Down the table it is $1$, $\mu(1 - \mu)$, and $\mu$.

## Generalised linear models

A *generalised linear model*, or GLM, connects a family to your features. It
keeps the linear score from lesson 2 and uses the canonical link to set
the natural parameter equal to that score:

$$
\eta = Xw, \qquad \mu = A'(\eta)
$$

Read it aloud: compute one score per example with $Xw$. Treat each score as
that example's natural parameter. Pass it through $A'$ to get the predicted
mean. For Bernoulli, that is the sigmoid, and you get logistic regression.
For Poisson, that is the exponential, and you get Poisson regression.

```python
eta = X @ w
mu = mean_from_natural(family, eta)
```

## One gradient and one Hessian for every family

The loss is the negative log-likelihood. Dropping $h(y)$, which does not
depend on $w$, leaves this sum over the $n$ examples:

$$
L(w) = \sum_{i=1}^{n} \bigl[ A(\eta_i) - y_i \eta_i \bigr], \qquad \eta = Xw
$$

Read it aloud: for each example, take $A$ of its score and subtract the
label times the score, then add them up. For Bernoulli, this is exactly the
lesson 5 loss without the ridge term.

```python
loss = np.sum(log_partition(family, eta) - y * eta)
```

Its gradient has the same shape for every family:

$$
\nabla L(w) = X^{\top}(\mu - y)
$$

Read it aloud: subtract each label from its predicted mean, then let
$X^{\top}$ add up those errors, weighted by each feature. The result has
shape `(d,)`. The only thing that changes between families is how you get
$\mu$.

The Hessian also keeps its lesson 5 shape. Statisticians call it the
*Fisher information* $I(w)$:

$$
I(w) = X^{\top} D X, \qquad D = \operatorname{diag}\bigl(A''(\eta_1), \ldots, A''(\eta_n)\bigr)
$$

Read it aloud: give each example a weight equal to its variance, scale each
row of $X$ by that weight, and multiply by $X^{\top}$. The result has shape
`(d, d)`.

```python
grad = X.T @ (mu - y)
info = (X * var[:, None]).T @ X
```

This neat result depends on the canonical link. With the canonical link, the
score equals $\eta$ directly, so the chain rule adds no extra factors. With
any other link, such as the probit link for yes-or-no data, extra terms
appear in both the gradient and the Hessian.

## Fisher scoring

*Fisher scoring* is Newton's method with the Fisher information in place of
the Hessian. With the canonical link the two are the same matrix, so Fisher
scoring and Newton's method are the same algorithm here:

$$
w \leftarrow w - I(w)^{-1} \nabla L(w)
$$

Read it aloud: solve $I(w)\,\delta = \nabla L(w)$ for the step $\delta$ and
subtract it from $w$.

```python
delta = np.linalg.solve(info, grad)
w = w - delta
```

On Bernoulli data at $w = 0$, every mean is 0.5 and every variance is 0.25.
So the first step is $w_1 = -(0.25\,X^{\top}X)^{-1} X^{\top}(0.5 - y)$, the
same first Newton step as lesson 5. One of the checks tests exactly that.

## Work one example by hand

Fit a Poisson regression to three examples. The first column is all ones,
so its weight works as a bias:

$$
X = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \quad
y = \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}, \quad
w = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

Start at $w = 0$. Every score is $\eta_i = 0$, so every mean and every
variance is $e^0 = 1$.

- Loss: $\sum (e^0 - y_i \times 0) = 1 + 1 + 1 = 3$.
- Errors $\mu - y$: $[0, 0, -1]$.
- Gradient $X^{\top}(\mu - y)$: first entry $0 + 0 - 1 = -1$, second entry
  $0 \times 0 + 1 \times 0 + 2 \times (-1) = -2$. So $g = [-1, -2]$.
- Fisher information with all weights 1 is $X^{\top}X$:

$$
I = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}
$$

Solving $I \delta = g$ gives $\delta = [0.1667, -0.5]$, so
$w_1 = [-0.1667, 0.5]$.

At $w_1$ the scores are $[-0.1667, 0.3333, 0.8333]$ and the means are
$[0.8465, 1.3956, 2.3010]$. The loss drops to $2.7097$. One more step gives
$w_2 = [-0.1481, 0.3954]$ with loss $2.6599$. Use these values to check your
code.

## Build it step by step

Build the functions in the order the checks test them.

1. **`log_partition(family, eta)`**. Takes a family name and an array of any
   shape, and returns $A(\eta)$ with the same shape. Branch on `family` with
   `if` statements and raise `ValueError` for an unknown name. Use the stable
   softplus for Bernoulli. Common mistake: `np.log(1 + np.exp(eta))`, which
   overflows for large `eta`.
2. **`mean_from_natural(family, eta)`**. Returns $A'(\eta)$, same shape. Use
   `eta.copy()`, a stable sigmoid, or `np.exp(eta)`. Common mistake: writing a
   sigmoid that overflows for large negative scores.
3. **`variance_from_natural(family, eta)`**. Returns $A''(\eta)$, same shape.
   Use `np.ones_like(eta)`, `p * (1 - p)`, or `np.exp(eta)`. Common mistake:
   returning the scalar `1` for the Gaussian, which has the wrong shape.
4. **`glm_gradient(family, w, X, y)`**. Returns shape `(d,)`. Compute
   `eta = X @ w`, then `mu`, then `X.T @ (mu - y)`. Common mistake: flipping
   the sign to `y - mu`.
5. **`glm_fisher_information(family, w, X)`**. Returns shape `(d, d)`.
   Compute the variances and scale the rows of `X` with `[:, None]` before
   the product. Common mistake: building `np.diag(var)`, which works but
   uses an $n$-by-$n$ matrix.
6. **`fit_fisher_scoring(family, X, y, max_iter, tol)`**. Start from
   `w = np.zeros(d)`. In each loop pass, append
   `np.sum(log_partition(family, eta) - y * eta)` to `history`, compute the
   gradient, stop if its norm is below `tol`, and otherwise take the step
   with `np.linalg.solve`. Return `(w, history)`. Common mistake: recording
   the loss after the step instead of before it.

## Cost and scaling

The cost depends on $n$, the number of examples, and $d$, the number of
features. The following table shows how each piece grows:

| Piece | Work grows like | In plain words |
|---|---|---|
| $A$, mean, and variance | $n$ | One formula per score. |
| Gradient | $n \times d$ | One pass over the data. |
| Fisher information | $n \times d^2$ | $d$ times the gradient's work. |
| One Fisher scoring step | $n \times d^2 + d^3$ | Same as a lesson 5 Newton step. |

Nothing in the table depends on the family. A Poisson regression costs the
same to fit as a logistic regression on data of the same shape.

## Where this breaks

**Counts that vary more than their mean.** Real count data often has a
variance larger than its mean. This is called *overdispersion*. The Poisson
family forces the two to be equal, so it reports false confidence. Switch to
a negative binomial model, which has an extra parameter for the spread.

**Links other than the canonical one.** With a probit link, or any other
non-canonical link, the extra chain-rule terms come back. Fisher scoring and
Newton's method become different algorithms, and code that assumes the
simple formulas computes the wrong Hessian.

**Perfect prediction in the Bernoulli family.** The separable-data problem
from lesson 5 still applies. As a Bernoulli mean approaches 0 or 1, its
variance shrinks to 0 and the weights grow without limit. The ridge penalty
from lesson 5 still fixes it. The Poisson family has no such problem, since
its variance $e^{\eta}$ stays positive for every finite score.

**Fixing the Gaussian variance at 1.** Real data has its own noise level.
Ignoring it leaves the fitted weights unchanged, but any error bars you
compute are off by a factor equal to the true noise standard deviation.

## The exercise

You implement `log_partition`, `mean_from_natural`, and
`variance_from_natural` for `"gaussian"`, `"bernoulli"`, and `"poisson"`.
On top of them you build `glm_gradient`, `glm_fisher_information`, and
`fit_fisher_scoring`, all with the canonical link, so `eta = X @ w`.

The checks compare `log_partition` with the formulas in the table, and
compare `mean_from_natural` and `variance_from_natural` with numerical
derivatives of your `log_partition`. They confirm the Poisson variance
equals its mean, and test `glm_gradient` and `glm_fisher_information`
against finite differences. They confirm `fit_fisher_scoring` recovers known
Poisson weights in under ten steps with a falling loss. The last check
confirms the first Bernoulli step equals the lesson 5 Newton step to machine
precision.
