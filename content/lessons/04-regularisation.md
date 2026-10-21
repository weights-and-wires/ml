---
title: Ridge, lasso, and the proximal view
slug: 04-regularisation
part: "Part 1 — Foundations"
summary: Shrink huge weights with a ridge penalty and a lasso penalty, and fit the lasso by soft thresholding.
minutes: 65
gpu: false
objectives:
  - Solve ridge regression with one linear system, and for many penalties from one SVD.
  - Compute the effective degrees of freedom and say what that number measures.
  - Build soft thresholding and explain why the lasso sets some weights to exactly zero.
  - Build ISTA and FISTA, and compare how fast the lasso loss falls.
exercise: 04-regularisation
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
---

# Ridge, lasso, and the proximal view

You fit a straight line through two points. One point has input 0 and label 0. The other has input 0.1 and label 1. The slope is 10. A weight is a number you adjust. On an input of 1, the line predicts 10. That line fits the noise.

Regularisation adds a fee for large weights on top of each miss. The line can miss the noisy point a little, and it no longer predicts 10.

Ridge charges squared size and still has a formula. The lasso charges absolute size and can set a weight to 0.

## What you already know

Five ideas from the prerequisite lessons matter here:

- A loss is a score for one example, so a squared miss of 9 contributes 4.5.
- The gradient is the direction of steepest uphill. For least squares it is $X^\top(Xw - y)$.
- The SVD writes $X$ using stretches $s_i$, the singular values.
- A safe step is $1/L$, with $L$ the largest stretch of $X^\top X$.
- Nesterov momentum reads the slope at a look-ahead point. Overfitting chases noise.

## Add the squared size to the score

Use the $2 \times 2$ identity, so the predictions are the weights. Labels are 3 and 1. Lambda is 1.

Weights 3 and 1 miss by 0. Squared size is $9 + 1 = 10$. The fee is half of 10, which is 5. The score is 5.

Weights 1.5 and 0.5 miss by $-1.5$ and $-0.5$. Half of $2.25 + 0.25$ is 1.25. Half of squared size 2.5 is 1.25. The score is 2.5.

This fee is *ridge regression*. Lambda 0 leaves least squares.

$$
L(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2
$$

Half the squared misses, plus half of lambda times the squared length, scores $1.25 + 1.25 = 2.5$ here.

The score has this shape:

```python
score = 0.5 * np.sum((X @ w - y) ** 2) + 0.5 * lam * np.dot(w, w)
```

Drop the half on the fee and the score is 3.75.

> [!KEY]
> Ridge adds half of lambda times the squared length.

## Solve ridge with one linear system

The gradient is the uphill direction. Misses $-1.5$ and $-0.5$, plus lambda times the weights, sum to $(0, 0)$. This point is the bottom.

$$
(X^\top X + \lambda I)\, w = X^\top y
$$

$X^\top X$ and $I$ are the identity, and $X^\top y = (3, 1)$. Lambda lifts the diagonal to 2. Then $2 w_1 = 3$ and $2 w_2 = 1$. The weights are 1.5 and 0.5.

The solve has this shape:

```python
w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
```

Omit lambda and the weights jump back to 3 and 1. The added diagonal also handles a repeated column.

## Read every penalty from one SVD

Both singular values of the identity are 1. The scale $1 / (1 + 1)$ is 0.5. Labels 3 and 1 become weights 1.5 and 0.5.

For stretch $s_i = 0.1$ and lambda 1, least squares uses $1 / 0.1 = 10$. Ridge uses $0.1 / (0.01 + 1) = 0.099$.

One SVD covers every lambda when $X = U \Sigma V^\top$:

$$
w = V \, \operatorname{diag}\!\left(\frac{s_i}{s_i^2 + \lambda}\right) U^\top y
$$

Rotate $y$ by $U^\top$, scale by $s_i / (s_i^2 + \lambda)$, and rotate back by $V$. A small $s_i$ goes towards 0, not towards $1/s_i$. These numbers return 1.5 and 0.5.

The path has this shape:

```python
shrink = s[None, :] / (s[None, :] ** 2 + lambdas[:, None])
path = (shrink * (U.T @ y)[None, :]) @ Vt
```

Divide by `s` alone and 0.1 returns 10, not 0.099.

> [!KEY]
> Ridge shrinks a weak direction. Singular value 0.1 gives about 0.099, not 10.

## Count the effective degrees of freedom

Each direction contributes a share from 0 to 1. Stretches 1 and 1 at lambda 1 give $0.5 + 0.5 = 1$. At lambda 0 the total is 2, the rank. A zero stretch adds nothing, so the total counts nonzero stretches.

Stretch 0.1 at lambda 1 contributes $0.01 / 1.01 = 0.0099$. Stretch 1 at lambda 1000 contributes about 0.001. A huge lambda drives the total towards 0.

$$
\text{df}(\lambda) = \sum_i \frac{s_i^2}{s_i^2 + \lambda}
$$

On the identity example the sum is 1.

The sum has this shape:

```python
dof = np.sum(s ** 2 / (s ** 2 + lam))
```

Using $s_i / (s_i^2 + \lambda)$ hides when $s_i = 1$, since both factors equal 0.5. At $s_i = 0.1$ you get 0.099 instead of 0.0099.

## Charge for the absolute size

A weight of 0.1 and lambda 1 costs $\tfrac{1}{2}(0.1)^2 = 0.005$ under ridge. The lasso costs 0.1. The charge is lambda on the first step off zero, so a weak pull leaves the weight at 0. That corner has no slope.

$$
L(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2 + \lambda \lVert w\rVert_1
$$

$\lVert w\rVert_1$ sums absolute values, with no extra half, so 0.1 contributes 0.1.

The loss has this shape:

```python
objective = 0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w))
```

Halve that fee and the contribution is 0.05.

Let $g_i$ be entry $i$ of $X^\top(Xw - y)$. Each best weight obeys one of the following rules:

- If $w_i \neq 0$, then $g_i + \lambda \, \mathrm{sign}(w_i) = 0$.
- If $w_i = 0$, then $|g_i| \le \lambda$.

The column example meets the first KKT rule exactly.

## Shrink one number past a threshold

Stay near 1.2 and pay 0.5 per unit. Staying costs $0.5 \times 1.2 = 0.6$. Moving to 0.7 costs half of $0.5^2$, which is 0.125, plus $0.5 \times 0.7 = 0.35$, total 0.475. Zero costs $0.5 \times 1.44 = 0.72$. So 0.7 wins.

From 0.3, staying costs 0.15 and zero costs 0.045, so zero wins.

This map is *soft thresholding*, the proximal operator of $t|z|$. It minimises $\tfrac{1}{2}(z - x)^2 + t|z|$.

$$
S_t(x) = \text{sign}(x) \, \max(|x| - t, \, 0)
$$

The sign stays and the result stops at 0. Threshold 0.5 sends 1.2 to 0.7, 0.3 to 0, and $-1.2$ to $-0.7$. Threshold 1 sends 3, $-3$, 0.5, $-0.5$, 0, 2 to 2, $-2$, 0, 0, 0, 1.

The map has this shape:

```python
z = np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
```

Drop the max with 0 and 0.3 becomes $-0.2$.

> [!TIP]
> Test `soft_threshold` on 1.2 and on 0.3 before any loop.

> [!KEY]
> The lasso leaves a weight at 0 unless the pull beats lambda. Soft thresholding is that rule for one number.

## Take a downhill step, then threshold

ISTA steps on the squared error, then thresholds at the step size times lambda.

Use the column $(1, 2)$ and labels $(1, 2)$. Then $X^\top X = 5$, so the step is $1/5 = 0.2$. Lambda is 1. Start at 0.

Misses $-1$ and $-2$ score 2.5. The gradient is $-5$. The step $0 - 0.2 \times (-5)$ lands on 1, and threshold 0.2 returns 0.8.

Predictions 0.8 and 1.6 miss by $-0.2$ and $-0.4$. Half of $0.04 + 0.16$ is 0.1, the fee is 0.8, and the loss is 0.9. The gradient is $-1$, and $-1 + 1 = 0$. At step $1/L$ the loss does not rise.

$$
w_{k+1} = S_{\alpha\lambda}\!\left(w_k - \alpha \, X^\top(Xw_k - y)\right)
$$

It returns 0.8. The step has this shape:

```python
grad = X.T @ (X @ w - y)  # shape (d,)
w = soft_threshold(w - step * grad, step * lam)
```

Threshold with `lam` and 1 becomes 0, not 0.8.

> [!KEY]
> ISTA steps on the squared error, then thresholds at `step * lam`. That second move writes the zeros.

After $k$ steps the gap shrinks like $1/k$.

## Look ahead with FISTA

FISTA takes that step at a look-ahead point $z$. A counter $t$ starts at 1.

$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \qquad z_{k+1} = w_{k+1} + \frac{t_k - 1}{t_{k+1}}\,(w_{k+1} - w_k)
$$

On the column, $t$ becomes $(1 + \sqrt{5}) / 2 = 1.618$, because $\sqrt{5} \approx 2.236$. The fraction $(1 - 1) / 1.618$ is 0, so $z$ and the loss stay at 0.8 and 0.9.

The momentum lines have this shape:

```python
t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
z = w_new + ((t - 1) / t_new) * (w_new - w)
```

A gradient at `w` still gives 0.8 on pass 1, then loses the $1/k^2$ rate. FISTA's loss can tick up. ISTA's does not, at step $1/L$.

> [!WARNING]
> Update $w$ and $t$ only after you form $z$. The fraction needs the old weight and the old $t$.

> [!RECAP]
> Hold these five facts before you write code:
>
> - Ridge scores 2.5 at weights 1.5 and 0.5, using $(X^\top X + \lambda I)\, w = X^\top y$.
> - The SVD scales direction $i$ by $s_i / (s_i^2 + \lambda)$. Degrees of freedom sum $s_i^2 / (s_i^2 + \lambda)$.
> - Soft thresholding turns 1.2 at threshold 0.5 into 0.7, and it turns 0.3 into 0.
> - ISTA steps by $1/L$, then thresholds at `step * lam`. The column goes from 0 to 0.8.
> - FISTA steps from $z$, starts $t$ at 1, and records the loss at $w$.

## Build it step by step

Build the six functions in this order:

1. `soft_threshold(x, thresh)` returns the shape of `x` from `np.sign`, `np.abs`, and `np.maximum`. Common mistake: `np.max` returns one number, so use `np.maximum`.

2. `ridge_closed_form(X, y, lam)` returns `(d,)`. Solve `X.T @ X + lam * np.eye(d)` with `np.linalg.solve`. Common mistake: `np.linalg.inv` can shift a weight.

3. `effective_degrees_of_freedom(s, lam)` returns a float, or `(m,)` for many lambdas. Branch on `np.asarray(lam, dtype=float).ndim == 0` and sum axis `-1`. Common mistake: the other axis yields one share per stretch.

4. `ridge_svd_path(U, s, Vt, y, lambdas)` returns `(m, d)`. Reuse `U.T @ y`, then `@ Vt`. Common mistake: no `+ lambda` turns 0.1 into 10.

5. `ista(X, y, lam, step, iters, w0=None)` returns `w` shaped `(d,)` and a history of length `iters + 1`, with the start loss at index 0. Use zeros if `w0` is omitted. Common mistake: threshold `lam`, and 0.8 becomes 0.

6. `fista` matches those shapes. Start `t` at 1, copy `w` into `z`, and record the loss at `w`. Common mistake: a gradient at `w` stalls after pass 1.

Form $z$ before you overwrite $w$ and $t$:

```python
# old w and old t
z = w_new + ((t - 1) / t_new) * (w_new - w)
```

## Cost and scaling

For $n$ rows and $d$ columns, the work grows as follows:

| Task | Rough work |
|---|---|
| Ridge, one lambda | About $nd^2$ to form $X^\top X$, plus $d^3$ to solve |
| Ridge, $m$ lambdas | One SVD, a few times $nd^2$, then about $m \cdot d \cdot r$ |
| One ISTA or FISTA step | About $2nd$ for the gradient |
| Soft thresholding | One pass over the $d$ weights |

The SVD pays after a handful of penalties. ISTA and FISTA never form $X^\top X$. FISTA needs far fewer steps on correlated columns.

## Where this breaks

The methods fail in the following ways:

- **Unequal scales.** Standardise columns. A tiny-unit column otherwise grows a huge weight.
- **A penalised intercept.** Centre $X$ and $y$, or leave the bias out of the fee.
- **A poor lambda.** Too small still fits noise. Too large sets every weight to 0. Choose lambda on held-out data.
- **Copied columns.** The lasso keeps one copy and zeros the rest. Ridge shares the weight.
- **A step above $1/L$.** The ISTA loss can rise. Read $L$ from `np.linalg.eigvalsh(X.T @ X).max()`.

## The exercise

You build `soft_threshold`, `ridge_closed_form`, `effective_degrees_of_freedom`, `ridge_svd_path`, `ista`, and `fista`.

`soft_threshold` must send 3, $-3$, 0.5, $-0.5$, 0, and 2, at threshold 1, to 2, $-2$, 0, 0, 0, and 1. It must also minimise $\tfrac{1}{2}(z - x)^2 + t|z|$.

`ridge_closed_form` must match least squares with $\sqrt{\lambda}\, I$ under $X$ and zeros under $y$. `ridge_svd_path` must match it at every lambda. Degrees of freedom must equal the rank at lambda 0 and sit near 0 when lambda is huge. At step $1/L$ the ISTA loss must not rise. After 80 steps on correlated columns, the FISTA loss must be under 0.7 times the ISTA loss, and both KKT rules must hold.
