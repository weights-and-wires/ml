---
title: Logistic regression
slug: 05-logistic-regression
part: "Part 1 — Foundations"
summary: Turn a yes-or-no score into a finite probability, then fit it by small steps and by Newton.
minutes: 60
gpu: false
objectives:
  - Compute the sigmoid and the log loss without overflow at a score of 800.
  - Build the gradient and the Hessian, and check both against finite differences.
  - Fit with gradient descent and with Newton's method, and compare how many steps each takes.
  - Explain why a perfect split makes the weights grow, and how a ridge penalty stops that growth.
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
exercise: 05-logistic-regression
---

# Logistic regression

You give each example one score. A score of 2 leans towards yes. A score of -3 leans towards no. A yes-or-no label needs a probability between 0 and 1, not a raw score.

A score of 800 breaks the direct recipe. You ask the machine to store e raised to 800, and that value overflows to infinity. The probability fails exactly when the model is most sure.

You build a probability that stays finite, then a loss, then two fitting loops. Small steps follow the slope. Newton steps also use the curvature. A perfect split of the labels makes the weights grow without a bound, and a ridge fee stops them.

## What you already know

Five facts are already in hand:

- A score is one number per row. Weights are the numbers you adjust. A slope of 2 is a weight.
- The gradient is the direction of steepest uphill. Descent steps against it.
- A loss is a score for one row. You add one term per row.
- Ridge adds half of lambda times the squared length of the weights.
- `np.exp(800)` is `inf`.

## Turn huge scores into probabilities

Take the scores $-800$, $0$, and $800$. At 0, $1/(1+1) = 0.5$. At 800, raise e only to $-800$, which is 0 in practice, so the probability is 1. At $-800$, use that same tiny power in $e^{z}/(1+e^{z})$, and the probability is 0.

This map is the *sigmoid*. A nonnegative score uses $1/(1+e^{-z})$. A negative score uses $e^{z}/(1+e^{z})$.

The single expression, before that split, is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It returns 0.5 at 0, and the stable branches return 1 and 0 at $\pm 800$. A score of $-20$ is about $2.06 \times 10^{-9}$, not 0. Do not clamp every large score.

The two branches have this shape:

```python
out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
```

Call `np.exp(-z)` on the whole array and the entry $-800$ asks for e to the 800. The output becomes `nan`.

> [!WARNING]
> `np.where` still evaluates both branches on every entry. A score of $-800$ overflows before the mask can pick a side.

> [!KEY]
> Raise e only to a number that is 0 or negative. Then a score of 800 stays finite, and the probability is 1.

## Score two rows with log loss

Use rows $(1, 0)$ and $(1, 2)$, labels 1 and 0, and weights 0.5 and $-1$. A loss is a score for one row. You add the rows.

Row scores are $0.5$ and $0.5 - 2 = -1.5$. Stable softplus of 0.5 keeps 0.5 and adds $\log(1 + 0.6065) = 0.4741$, so it equals 0.9741. Subtract the label times the score, $1 \times 0.5$. The row scores 0.4741.

For $-1.5$ the positive part is 0, and $\log(1 + 0.2231) = 0.2014$. The label is 0, so the row scores 0.2014. The total is 0.6755. Lambda is 0, so the ridge fee adds 0.

Ten rows at zero weights each score $\log 2 = 0.693$, so the total is 6.931. Weights 3 and $-4$ with ridge 2 have squared length 25, and half of $2 \times 25$ is 25. Drop the half and the fee is 50.

The loss sums those row scores and adds the fee:

$$
L(w) = \sum_{i=1}^{n} \Bigl[ \operatorname{softplus}(z_i) - y_i z_i \Bigr]
+ \frac{\lambda}{2} \lVert w \rVert^2,
\qquad \operatorname{softplus}(z) = \log(1 + e^{z})
$$

$z_i$ is the row score. You subtract $y_i z_i$, then add the ridge fee. The sum, not a mean, returns $0.4741 + 0.2014 = 0.6755$ here.

Keep softplus finite with the same trick:

$$
\operatorname{softplus}(z) = \max(z, 0) + \log\bigl(1 + e^{-|z|}\bigr)
$$

At score 800 and label 1, softplus is 800, and $800 - 800 = 0$. The direct $\log(1 + e^{800})$ is infinity.

The loss has this shape:

```python
softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
loss = np.sum(softplus - y * z) + 0.5 * ridge * np.dot(w, w)
```

Use the mean instead of the sum and the two rows score about 0.338 instead of 0.6755.

## Build the gradient

The probabilities on these rows are $1 / 1.6065 = 0.6225$ and $0.2231 / 1.2231 = 0.1824$. Errors are probability minus label: $-0.3775$ and $0.1824$.

The first weight sees a 1 in both rows. $-0.3775 + 0.1824 = -0.1951$. The second weight sees a 0 and a 2. $2 \times 0.1824 = 0.3648$, or 0.3649 if you keep one more digit of the probability. Use $g = [-0.1951,\ 0.3649]$.

The gradient, the steepest uphill direction, is that vector plus the ridge term:

$$
\nabla L(w) = X^{\top}\bigl(\sigma(Xw) - y\bigr) + \lambda w
$$

Lambda is 0, so the formula returns $[-0.1951,\ 0.3649]$, with shape `(d,)`.

The line has this shape:

```python
grad = X.T @ (sigmoid(X @ w) - y) + ridge * w
```

Write `y - p` and the first entry flips to $+0.1951$. Descent then walks uphill.

> [!KEY]
> The logistic gradient is $X^\top(p - y)$ plus lambda times $w$.

## Build the Hessian

Each row also has a curvature weight $p(1 - p)$. Row 1 gives $0.6225 \times 0.3775 = 0.2350$. Row 2 gives $0.1824 \times 0.8176 = 0.1491$. The largest possible weight is $0.5 \times 0.5 = 0.25$.

Add the weights into $X^\top X$. The entries are $0.2350 + 0.1491 = 0.3841$, $2 \times 0.1491 = 0.2982$, and $4 \times 0.1491 = 0.5964$. Unrounded, that is about $0.3842$, $0.2983$, and $0.5966$.

The Hessian puts those weights on a diagonal $D$ and adds lambda:

$$
\nabla^2 L(w) = X^{\top} D X + \lambda I, \qquad
D = \operatorname{diag}\bigl(p_1(1-p_1), \ldots, p_n(1-p_n)\bigr)
$$

$D$ is diagonal, so skip the $n \times n$ matrix. Lambda 0.5 adds 0.5 on the diagonal, and the corners become about 0.8842 and 1.0966.

The product has this shape:

```python
H = (X * (p * (1.0 - p))[:, None]).T @ X + ridge * np.eye(d)
```

Drop `[:, None]` and numpy tries to scale columns instead of rows. The shape breaks, or the entries come out wrong.

> [!TIP]
> On these two rows, `X * s[:, None]` matches `np.diag(s) @ X` and skips the big diagonal matrix.

Every curvature weight is at least 0, so the loss is a bowl. A zero gradient is the bottom.

> [!KEY]
> Curvature weights such as 0.235 and 0.149 are never negative. A zero gradient is the bottom of the bowl.

## Compare a gradient step with a Newton step

Stay on the same two rows and the same weights, 0.5 and $-1$. The loss is 0.6755 and the gradient is about $[-0.1951,\ 0.3649]$.

One gradient step of size 1 subtracts that gradient. The weights become $0.5 - (-0.1951) = 0.6951$ and $-1 - 0.3649 = -1.3649$. The loss falls only to about 0.528.

Newton fits a bowl to the curvature and jumps to its bottom:

$$
w \leftarrow w - H^{-1} g, \qquad g = \nabla L(w), \quad H = \nabla^2 L(w)
$$

Solve $H \delta = g$ and subtract $\delta$. Here $\delta$ is about $[-1.6065,\ 1.4148]$. The first row checks out: $0.3842 \times (-1.6065) + 0.2983 \times 1.4148 = -0.1952$. The weights become $2.1065$ and $-2.4148$, and the loss is about 0.178. Newton moved the second weight by 1.41. The plain step moved it by 0.36.

Start at zero on these rows. Probabilities are 0.5 and curvature weights are 0.25. The gradient is $[0,\ 1]$. The Hessian is $[[0.5,\ 0.5],\ [0.5,\ 1]]$. From $0.5\delta_1 + 0.5\delta_2 = 0$ you get $\delta_1 = -\delta_2$. Then $0.5\delta_2 = 1$, so the step is $[-2,\ 2]$ and the weights are $[2,\ -2]$. The loss falls from 1.386 to about 0.254. A plain step only reaches $[0,\ -1]$, with loss about 0.820. The update has this shape:

```python
w = w - np.linalg.solve(H, g)
```

`np.linalg.inv` makes a noisy step when the bowl is flat. A huge gradient step makes the loss rise.

## Let a perfect split grow the weights

Put a label 0 at input $-1$ and a label 1 at input 1. Any positive weight already separates them. The loss of this pair is $2 \log(1 + e^{-w})$.

At $w = 1$, $2 \log(1.3679) = 0.627$. At 5 and 10 the loss is about 0.013 and 0.00009. The sigmoid never returns 0 or 1, so the loss never hits 0, and descent keeps lengthening $w$.

Ridge strength 1 adds $\tfrac{1}{2} w^2$. Totals at 1, 5, and 10 are about 1.13, 12.5, and 50. The fee wins, and it also puts lambda on the Hessian diagonal. `fit_newton` therefore defaults `ridge` to `1e-8`.

> [!KEY]
> On a perfect split the unpenalised weights grow without a bound. A ridge fee gives them a finite size.

> [!RECAP]
> Hold these five facts before you write code:
>
> - Stable sigmoid maps $-800$, $0$, $800$ to $0$, $0.5$, $1$. Softplus is $\max(z,0)+\log(1+e^{-|z|})$.
> - The loss sums softplus minus $y_i z_i$, plus half lambda times the squared length.
> - Gradient $X^\top(p-y)+\lambda w$ has shape `(d,)`. Hessian $X^\top DX+\lambda I$ has shape `(d, d)`.
> - Newton solves $H\delta=g$. Record the loss first, so index 0 is the loss at zero weights.
> - A perfect split grows unpenalised weights. Ridge stops that growth.

## Build it step by step

Build the six functions in this order. Each one uses the ones before it:

1. `sigmoid(z)` returns the shape of `z`. Mask with `z >= 0`, fill each branch only on its own side, and use `np.empty_like`. Common mistake: `np.exp(-z)` on every entry overflows at $-800$.

2. `log_loss(w, X, y, ridge)` takes shapes `(d,)`, `(n, d)`, and `(n,)`, and returns one float. Use the stable softplus, sum `softplus - y * z`, and add `0.5 * ridge * np.dot(w, w)`. Common mistake: a mean instead of a sum turns 0.6755 into about 0.338.

3. `gradient(w, X, y, ridge)` returns shape `(d,)`. Use your `sigmoid`, then `X.T @ (p - y) + ridge * w`. Common mistake: `y - p` sends every later step uphill.

4. `hessian(w, X, y, ridge)` returns shape `(d, d)`. Scale rows with `X * s[:, None]`, multiply by `X`, and add `ridge * np.eye(d)`. Common mistake: `X * s` without `[:, None]` scales the wrong axis.

5. `fit_gradient_descent(X, y, lr, max_iter, ridge, tol)` starts at zeros. Append the loss, stop without stepping when the gradient norm is below `tol`, otherwise set `w = w - lr * g`. Common mistake: recording after the update, so index 0 is not the loss at zero.

6. `fit_newton(X, y, max_iter, ridge, tol)` uses that same history. Default `ridge` is `1e-8`. Update with `w - np.linalg.solve(H, g)`. Common mistake: `np.linalg.inv` returns a noisy step.

## Cost and scaling

For $n$ rows and $d$ weights, the work grows as follows:

| Piece | Work |
|---|---|
| Loss or gradient | About $nd$, one pass |
| Hessian | About $nd^2$ |
| Newton solve | About $d^3$ |
| Hessian memory | $d^2$ numbers, so 10,000 weights need 100 million |

A few Newton steps beat hundreds of cheap steps when $d$ is small. Large $d$ belongs to gradient descent.

## Where this breaks

The fit fails in the following ways:

- **A perfect split.** Weights grow after the held-out score has stalled. Add ridge, or stop early.
- **A rare class.** If 99 of 100 labels are 0, most weights $p(1-p)$ are tiny and Newton gets fragile. Weight a row by the inverse of its class frequency.
- **Hidden overflow.** A naive loss returns `inf`. Check `np.isfinite`, and use the stable formulas.
- **Too many weights.** Near $d \approx n$, the Hessian costs more than it saves. Use gradient descent, or L-BFGS, which stores a curvature estimate instead of $H$.

## The exercise

You implement `sigmoid`, `log_loss`, `gradient`, `hessian`, `fit_gradient_descent`, and `fit_newton`. Both fitting loops start at $w = 0$ and record the loss before every step.

`sigmoid` must be finite and correct from $-800$ to $800$, including $-20$ and $20$. `log_loss` must stay finite at $\pm 800$. At zero weights it equals $n \log 2$, and ridge 2 on weights 3 and $-4$ adds 25. `gradient` and `hessian` must match finite differences. The Hessian must be symmetric and positive semidefinite. Newton must finish in under ten steps. Gradient descent must need hundreds, and the two answers must agree. On a perfect split the unpenalised norm must keep growing, and the ridge loss must stay higher.
