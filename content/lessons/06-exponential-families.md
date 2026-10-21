---
title: Exponential families and GLMs
slug: 06-exponential-families
part: "Part 1 — Foundations"
summary: Swap one function to get the mean, the variance, and one fit for temperature, yes-or-no, and counts.
minutes: 55
gpu: false
objectives:
  - Compute the log-partition, the mean, and the variance for the Gaussian, Bernoulli, and Poisson families.
  - Explain how the mean and the variance come from two derivatives of one function.
  - Build the shared gradient and Fisher information, and fit a Poisson model by Fisher scoring.
  - Explain why Bernoulli Fisher scoring repeats the Newton step from lesson 5.
prereqs:
  - 05-logistic-regression
exercise: 06-exponential-families
---

# Exponential families and GLMs

You meet three prediction jobs. A temperature sits on a line, and a bell curve of variance 1 fits it. A yes-or-no label is the coin flip from lesson 5. A count of arrivals is 0, 1, 2, or more in an hour.

Each job looks like its own loss and its own slope. You swap one function, called A. The mean, the variance, and the fitting step stay in place.

A is the piece that changes. Its slope is the mean you predict. Its bend is the variance, the spread around that mean.

## What you already know

Five facts from lesson 5 are the starting point:

- A loss is a score for one row. Weights are the numbers you adjust, such as a slope of 2.
- The gradient is the direction of steepest uphill. For coins it is $X^\top(p - y)$.
- The Hessian uses weights $p(1-p)$. Newton solves that matrix times a step, then subtracts the step.
- At zero weights on rows $(1, 0)$ and $(1, 2)$ with labels 1 and 0, Newton lands on 2 and $-2$.
- A variance is a spread and is never negative. A Poisson count of mean 4 also has variance 4.

## Differentiate one function on three jobs

Call the input of A eta, pronounced EE-tah. It slides the distribution along.

A temperature uses $A(\eta) = \eta^2 / 2$. At eta 2, $A = 2$. At eta 3, $A = 4.5$. At eta 1, $A = 0.5$. A step of 1 gives the slope $(4.5 - 0.5) / 2 = 2$, equal to eta. The bend is $4.5 - 2 \times 2 + 0.5 = 1$. The spread is 1 for every eta.

A yes-or-no uses the softplus from lesson 5, $A(\eta) = \log(1 + e^{\eta})$. At eta 0, $A = 0.69315$. A step of 0.1 gives $A(0.1) = 0.74440$ and $A(-0.1) = 0.64440$. The slope is $(0.74440 - 0.64440) / 0.2 = 0.5$. The bend is $(0.74440 + 0.64440 - 2 \times 0.69315) / 0.01 = 0.25$. The mean is 0.5 and the variance is $0.5 \times 0.5 = 0.25$.

A count uses $A(\eta) = e^{\eta}$. At eta 0, $A = 1$. A step of 0.1 gives $e^{0.1} = 1.1052$ and $e^{-0.1} = 0.9048$. The slope is $(1.1052 - 0.9048) / 0.2 = 1.002$. The bend is $(1.1052 + 0.9048 - 2) / 0.01 = 1.00$. The exact slope and the exact bend are both 1, so the mean equals the variance.

The slope of $A$ is the mean and the bend is the variance. $A$ is the *log-partition*. Eta is the *natural parameter*. Swap $A$, and every later line stays.

$$
\mu = A'(\eta), \qquad \operatorname{Var}(y) = A''(\eta)
$$

$\mu$, pronounced mu, is the mean. At eta 2 the formula returns mean 2 and variance 1. A variance cannot be negative, so $A$ bends upwards, and every fit is a bowl with one bottom.

The three families, with those derivatives, are as follows:

| Family | Log-partition $A(\eta)$ | Mean $A'(\eta)$ | Variance $A''(\eta)$ | Canonical link |
|---|---|---|---|---|
| Gaussian, variance 1 | $\eta^2 / 2$ | $\eta$ | $1$ | identity: $\eta = \mu$ |
| Bernoulli | $\log(1 + e^{\eta})$ | $\sigma(\eta)$ | $\sigma(\eta)(1 - \sigma(\eta))$ | logit: $\eta = \log\frac{\mu}{1 - \mu}$ |
| Poisson | $e^{\eta}$ | $e^{\eta}$ | $e^{\eta}$ | log: $\eta = \log \mu$ |

The link turns a mean back into eta. A Bernoulli mean of 0.8 has eta $\log(0.8 / 0.2) = \log 4 \approx 1.386$. A Poisson mean of 1 has eta 0. In terms of the mean, the variances are $1$, $\mu(1-\mu)$, and $\mu$.

> [!KEY]
> Differentiate $A$ once for the mean and twice for the variance. Only $A$ changes across the three families.

> [!KEY]
> A Poisson mean and a Poisson variance are the same number. At eta 0, both are 1.

The shared probability, which you do not code, comes after those derivatives:

$$
p(y \mid \eta) = h(y) \exp\bigl(\eta \, y - A(\eta)\bigr)
$$

$h(y)$ depends only on $y$. For a Poisson count, $h(y) = 1/y!$, so $h(2) = 1/2$. At eta 0 the exponent is $-1$, $e^{-1} = 0.3679$, and the probability is $0.5 \times 0.3679 = 0.1839$. That is two arrivals when the mean is 1.

A central difference checks both derivatives:

```python
mean_estimate = (A(eta + h) - A(eta - h)) / (2 * h)
var_estimate = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2
```

> [!TIP]
> Keep the second-difference nudge near `1e-4`. A much smaller nudge lets rounding swamp the bend.

`np.log(1 + np.exp(eta))` overflows at eta 800. The stable softplus stays finite.

## Fit three counts with one gradient

A generalised linear model, a GLM, sets eta equal to the linear score:

$$
\eta = Xw, \qquad \mu = A'(\eta)
$$

For a coin, $A'$ is the sigmoid, and you recover logistic regression. For counts, $A'$ is $e^{\eta}$.

The rows are $(1, 0)$, $(1, 1)$, and $(1, 2)$, with counts 1, 1, and 2. At weights 0 and 0, every eta is 0 and every mean is $e^{0} = 1$.

Each row scores $1 - 0 = 1$, so the loss is 3. Errors are 0, 0, and $1 - 2 = -1$. The first weight sums them to $-1$. The second does $2 \times (-1) = -2$. The gradient is $[-1,\ -2]$.

The loss and the gradient have one shape for every family:

$$
L(w) = \sum_{i=1}^{n} \bigl[ A(\eta_i) - y_i \eta_i \bigr], \qquad \eta = Xw
$$

$$
\nabla L(w) = X^{\top}(\mu - y)
$$

Add $A$ of the score and subtract the label times the score. On coins that is the lesson 5 loss without ridge. Here the formula returns $[-1,\ -2]$, shape `(d,)`.

Two lines have this shape:

```python
loss = np.sum(log_partition(family, eta) - y * eta)
grad = X.T @ (mu - y)  # shape (d,)
```

`y - mu` flips the gradient to $[1,\ 2]$, and every later step walks the wrong way.

> [!WARNING]
> This loss drops $h(y)$. A Poisson value from `scipy.stats` is smaller by $\sum \log (y_i!)$. At a count of 2 that gap is $\log 2 = 0.693$.

> [!KEY]
> With the canonical link, every family has gradient $X^\top(\mu - y)$. Only the line that builds $\mu$ changes.

## Scale each row by its variance

Every variance is 1, so the Fisher information is $X^\top X$. Column $(1, 1, 1)$ dotted with itself is 3, and dotted with $(0, 1, 2)$ is 3. Column $(0, 1, 2)$ dotted with itself is $0 + 1 + 4 = 5$. The matrix is $[[3,\ 3],\ [3,\ 5]]$.

$$
I(w) = X^{\top} D X, \qquad D = \operatorname{diag}\bigl(A''(\eta_1), \ldots, A''(\eta_n)\bigr)
$$

$D$ holds one variance per row. Every scale is 1, so you recover $[[3,\ 3],\ [3,\ 5]]$, shape `(d, d)`. Build it like this:

```python
info = (X * var[:, None]).T @ X
```

`np.diag(var)` spends an $n \times n$ matrix on the same product.

Fisher scoring is Newton with this matrix. On the canonical link the two matrices match:

$$
w \leftarrow w - I(w)^{-1} \nabla L(w)
$$

Solve $I\delta = g$ and subtract. The equations are $3\delta_1 + 3\delta_2 = -1$ and $3\delta_1 + 5\delta_2 = -2$. Their difference is $2\delta_2 = -1$, so $\delta_2 = -0.5$. Then $3\delta_1 - 1.5 = -1$, so $3\delta_1 = 0.5$ and $\delta_1 = 1/6$. Subtract that step from zero. The weights are $-1/6$ and $0.5$.

The scores are $-0.1667$, $0.3333$, and $0.8333$. The means are about 0.8465, 1.3956, and 2.3010. The loss falls from 3 to about 2.7097. A second step reaches about $-0.1481$ and $0.3954$, with loss about 2.6599. The update has this shape:

```python
w = w - np.linalg.solve(info, grad)
```

On the lesson 5 rows, Bernoulli means are 0.5 and variances are 0.25. The gradient is $[0,\ 1]$ and the information is $[[0.5,\ 0.5],\ [0.5,\ 1]]$. The weights become $[2,\ -2]$.

> [!KEY]
> Bernoulli Fisher scoring repeats the lesson 5 Newton step. From zero weights it lands on $[2,\ -2]$.

A probit link adds chain-rule factors, so Fisher scoring and Newton then differ. On a Gaussian identity with labels 3 and 1, the errors at zero weights are $-3$ and $-1$. One step lands on 3 and 1, which is least squares.

> [!RECAP]
> Hold these five facts before you write code:
>
> - The mean is $A'(\eta)$ and the variance is $A''(\eta)$. Eta is pronounced EE-tah.
> - $A$ is $\eta^2/2$, stable softplus, or $e^{\eta}$ for the Gaussian, Bernoulli, and Poisson.
> - A canonical GLM sets $\eta = Xw$ and predicts $\mu = A'(\eta)$.
> - The gradient is $X^\top(\mu - y)$. Fisher scoring scales the rows of $X$ by the variances.
> - On coins, the first step from zero weights matches the Newton step in lesson 5.

## Build it step by step

Build the six functions in the order the checks call them:

1. `log_partition(family, eta)` returns $A(\eta)$ in the shape of `eta`. Branch on `"gaussian"`, `"bernoulli"`, and `"poisson"`. Raise `ValueError` for any other name. Common mistake: `np.log(1 + np.exp(eta))` overflows for a large positive eta.

The three expressions have this shape:

```python
0.5 * eta ** 2
np.maximum(eta, 0) + np.log1p(np.exp(-np.abs(eta)))
np.exp(eta)
```

2. `mean_from_natural(family, eta)` returns $A'(\eta)$, same shape. Use `eta.copy()`, a stable sigmoid, or `np.exp(eta)`. Common mistake: `np.exp(-eta)` overflows on a large negative eta.

3. `variance_from_natural(family, eta)` returns $A''(\eta)$, same shape. Use `np.ones_like(eta)`, `p * (1 - p)`, or `np.exp(eta)`. Common mistake: the bare number `1` has the wrong shape.

4. `glm_gradient(family, w, X, y)` returns shape `(d,)`. Set `eta = X @ w`, then return `X.T @ (mu - y)`. Common mistake: `y - mu` flips the sign.

5. `glm_fisher_information(family, w, X)` returns shape `(d, d)`. Scale rows with `var[:, None]`. Common mistake: `np.diag(var)` builds an $n \times n$ matrix.

6. `fit_fisher_scoring(family, X, y, max_iter, tol)` starts at `w = np.zeros(d)`. Append `np.sum(log_partition(family, eta) - y * eta)` before the step. Stop, without stepping, when the gradient norm is under `tol`. Otherwise subtract `np.linalg.solve(info, grad)`. Return `(w, history)`. Common mistake: recording after the step, so index 0 is not the loss at zero.

The step has this shape:

```python
w = w - np.linalg.solve(info, grad)
```

## Cost and scaling

For $n$ rows and $d$ weights, the work grows as follows:

| Piece | Work |
|---|---|
| $A$, the mean, and the variance | About $n$, one formula per score |
| Gradient | About $nd$, one pass |
| Fisher information | About $nd^2$ |
| One Fisher scoring step | About $nd^2 + d^3$, the same as one Newton step |

Nothing in that table depends on the family. A Poisson fit costs the same as a logistic fit of the same shape.

## Where this breaks

The shared loop fails in the following ways:

- **Counts that spread more than their mean.** This is *overdispersion*. Poisson forces variance to equal the mean and reports a false confidence. A negative binomial model adds a parameter for the extra spread.
- **A link other than the canonical one.** A probit link brings the chain-rule terms back. Fisher scoring and Newton diverge, and the short formulas give the wrong Hessian.
- **A perfect Bernoulli split.** As a coin probability nears 0 or 1, its variance shrinks towards 0 and the weights grow without a bound. The ridge fee from lesson 5 still stops them. Poisson variance $e^{\eta}$ stays positive.
- **A Gaussian variance fixed at 1.** The fitted weights do not change. Error bars are off by a factor equal to the true noise standard deviation.

## The exercise

You implement `log_partition`, `mean_from_natural`, and `variance_from_natural` for `"gaussian"`, `"bernoulli"`, and `"poisson"`. You also implement `glm_gradient`, `glm_fisher_information`, and `fit_fisher_scoring`. Eta equals $X @ w$.

The checks compare `log_partition` with $\eta^2/2$, the stable softplus, and $e^{\eta}$ on $-1$, $0$, $1$, and $2$. The mean must match a numerical derivative of $A$, and the variance must match a second derivative. Poisson variance must equal the Poisson mean. `glm_gradient` and `glm_fisher_information` must match finite differences, and the information must be symmetric. Poisson `fit_fisher_scoring` must recover known weights to within 0.15 in under ten steps, and the loss must fall. The first Bernoulli step must match $-(0.25\, X^\top X)^{-1} X^\top(0.5 - y)$. The full coin run must finish in under ten steps with a tiny gradient.
