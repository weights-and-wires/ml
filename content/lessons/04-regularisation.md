---
title: Ridge, lasso, and the proximal view
slug: 04-regularisation
part: "Part 1 — Foundations"
summary: Stop weights from growing wild with ridge and lasso penalties, and solve the lasso with soft thresholding.
minutes: 65
gpu: false
objectives:
  - Compute ridge regression with one linear solve, and for many penalties at once from one SVD.
  - Compute the effective degrees of freedom and explain what it measures.
  - Build soft thresholding and explain why the lasso sets some weights to exactly zero.
  - Build ISTA and FISTA, and compare how fast they reduce the lasso loss.
exercise: 04-regularisation
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
---

# Ridge, lasso, and the proximal view

Lesson 2 ended with a warning. When features repeat each other, least squares weights become huge and swing wildly with tiny changes in the data. The model fits the training set well and predicts new data badly. That is overfitting, which you met in lesson 1.

Regularisation fixes this by charging a fee for large weights. You still minimise the squared error, but you add a penalty that grows with the size of $w$. The model now has to balance fitting the data against keeping its weights small. It gives up a little accuracy on the training set and gains a lot of stability.

You build two penalties. Ridge charges for the squared size of the weights and still has an exact formula. The lasso charges for their absolute size, which pushes some weights to exactly zero. That makes the lasso a feature selector, but it needs a new tool, soft thresholding, and a gradient-style solver built on it.

## What you already know

- The least squares loss $\tfrac{1}{2}\lVert Xw - y\rVert^2$ and its gradient $X^\top(Xw - y)$ (lesson 2).
- The SVD $X = U \Sigma V^\top$, with singular values in a vector `s` (lesson 2).
- Gradient descent with step $1/L$, where $L$ is the largest eigenvalue of $X^\top X$ (lesson 3).
- Nesterov's look-ahead momentum (lesson 3).
- Overfitting: a model that is too flexible fits noise (lesson 1).

## Ridge regression

Think of each weight as held near zero by a spring. The data pulls the weight toward the value that fits best. The spring pulls it back. A stiffer spring keeps weights smaller. Ridge adds that spring to the loss:

$$L(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2$$

Read it aloud: the first term is the usual squared error. The second is the penalty. $\lVert w\rVert^2$ is the sum of the squared weights, and $\lambda$ (lambda) is a number you choose, at least 0, that sets how stiff the spring is. $\lambda = 0$ gives back plain least squares.

Setting the gradient to zero gives a linear system, like the normal equations with one extra term:

$$(X^\top X + \lambda I)\, w = X^\top y$$

Read it aloud: add $\lambda$ to every diagonal entry of $X^\top X$, then solve. $I$ is the $d \times d$ identity matrix. Adding $\lambda$ to the diagonal also makes the system solvable even when features repeat.

```python
w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
```

### Ridge for many penalties at once

You rarely know the right $\lambda$, so you try many. Re-solving each time is wasteful. The SVD gives every answer cheaply. With $X = U \Sigma V^\top$, the ridge weights are

$$w = V \, \operatorname{diag}\!\left(\frac{s_i}{s_i^2 + \lambda}\right) U^\top y$$

Read it aloud from right to left: rotate $y$ into the $U$ coordinates, which you do once. Multiply coordinate $i$ by $s_i/(s_i^2 + \lambda)$. Rotate back with $V$. Only the middle step depends on $\lambda$.

Compare this with plain least squares, which multiplies by $1/s_i$. When $s_i$ is large, $\lambda$ barely matters. When $s_i$ is small, the factor shrinks toward zero instead of exploding. Ridge shrinks most in the directions the data pins down least.

```python
u_dot_y = U.T @ y                                   # shape (r,), once
shrink = s[None, :] / (s[None, :] ** 2 + lambdas[:, None])   # shape (m, r)
path = (shrink * u_dot_y[None, :]) @ Vt             # shape (m, d)
```

The `[None, :]` and `[:, None]` add an axis so numpy broadcasts: each row of `shrink` holds the factors for one $\lambda$.

### Effective degrees of freedom

How flexible is a ridge model? Plain least squares with $r$ independent features has $r$ degrees of freedom, one per weight. Ridge has fewer, because the spring holds the weights back. The effective degrees of freedom counts this:

$$\text{df}(\lambda) = \sum_i \frac{s_i^2}{s_i^2 + \lambda}$$

Read it aloud: each singular direction contributes a number between 0 and 1. It is close to 1 when $s_i^2$ is much bigger than $\lambda$, and close to 0 when it is much smaller. At $\lambda = 0$, df equals the number of nonzero singular values. As $\lambda$ grows huge, df falls to 0.

```python
dof = np.sum(s ** 2 / (s ** 2 + lam))
```

## The lasso

The lasso swaps the squared penalty for the sum of absolute values:

$$L(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2 + \lambda \lVert w\rVert_1$$

Read it aloud: squared error, plus $\lambda$ times $\lVert w\rVert_1$, the sum of the absolute values of the weights. This is called the $\ell_1$ norm.

```python
objective = 0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w))
```

Why does this zero out weights? Look at the cost of moving one weight away from zero. With ridge, the penalty $\tfrac{\lambda}{2}w^2$ is almost flat near zero, so a tiny weight costs almost nothing, and weights rarely land at exactly zero. With the lasso, the penalty $\lambda|w|$ has a sharp corner at zero. Moving off zero costs $\lambda$ per unit right away. If the data can't pay that price, the weight stays at exactly zero.

That corner is also the trouble. $|w|$ has no derivative at zero, so plain gradient descent doesn't apply.

### The condition at the answer

At the lasso answer, one rule holds for each weight $w_i$. Let $g_i$ be entry $i$ of the squared-error gradient $X^\top(Xw - y)$.

- If $w_i$ is not zero, then $g_i + \lambda \, \text{sign}(w_i) = 0$. The data's pull exactly balances the penalty.
- If $w_i$ is zero, then $|g_i| \leq \lambda$. The data's pull is too weak to pay the price of moving off zero.

These are called the KKT conditions. You can test any lasso solver against them.

## Soft thresholding

Soft thresholding is the one-number version of the lasso. Given a value $x$ and a threshold $t \geq 0$, it finds the $z$ that minimises $\tfrac{1}{2}(z - x)^2 + t|z|$. In words: stay close to $x$, but pay $t$ per unit of size. The answer is

$$S_t(x) = \text{sign}(x) \, \max(|x| - t, \, 0)$$

Read it aloud: move $x$ toward zero by $t$. If that would cross zero, stop at zero. Values with $|x| \leq t$ become exactly zero. This operation is called the proximal operator of $t|\cdot|$.

```python
z = np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
```

With $t = 1$: $3 \to 2$, $-3 \to -2$, $0.5 \to 0$, $-0.5 \to 0$, $0 \to 0$, and $2 \to 1$.

## ISTA: gradient step, then threshold

ISTA handles the two parts of the lasso one at a time. It takes an ordinary gradient step on the squared error, then applies soft thresholding for the penalty:

$$w_{k+1} = S_{\alpha\lambda}\!\left(w_k - \alpha \, X^\top(Xw_k - y)\right)$$

Read it aloud: compute the gradient of the squared error, step downhill by $\alpha$ times it, then soft threshold every entry with threshold $\alpha\lambda$. Use step $\alpha = 1/L$, with $L$ the largest eigenvalue of $X^\top X$. With that step, the lasso loss never goes up.

```python
grad = X.T @ (X @ w - y)
w = soft_threshold(w - step * grad, step * lam)
```

ISTA stands for iterative shrinkage-thresholding algorithm.

## FISTA: ISTA with momentum

FISTA adds Nesterov's look-ahead from lesson 3. It keeps a look-ahead point $z$, takes the ISTA step from $z$, and updates a momentum counter $t$ that starts at $t_0 = 1$:

$$t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \qquad z_{k+1} = w_{k+1} + \frac{t_k - 1}{t_{k+1}}\,(w_{k+1} - w_k)$$

Read it aloud: $t$ grows by about one half each step. The fraction $(t_k - 1)/t_{k+1}$ is the momentum. It starts at 0 and creeps toward 1. The look-ahead point $z$ is the new point plus that fraction of the last move. The ISTA step at each iteration uses the gradient at $z$, not at $w$. Start with $z_0 = w_0$.

```python
w_new = soft_threshold(z - step * X.T @ (X @ z - y), step * lam)
t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
z = w_new + ((t - 1) / t_new) * (w_new - w)
```

After $k$ steps, ISTA's gap to the best loss shrinks like $1/k$. FISTA's shrinks like $1/k^2$, which is much faster. Unlike ISTA, FISTA's loss can tick up now and then.

## A worked example by hand

Use one feature and two examples: $X = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $y = (1, 2)$. Then $X^\top X = 5$ and $X^\top y = 5$, and the least squares weight is $1$.

**Ridge with $\lambda = 1$.** Solve $(5 + 1)\,w = 5$, so $w = 5/6 \approx 0.8333$. The SVD route agrees: $s = \sqrt{5}$, $U^\top y = 5/\sqrt{5} = \sqrt{5}$, and $\sqrt{5} \cdot \sqrt{5}/(5 + 1) = 5/6$. The effective degrees of freedom is $5/(5 + 1) \approx 0.8333$.

**One ISTA step with $\lambda = 1$.** Here $L = 5$, so the step is $\alpha = 0.2$. Start at $w_0 = 0$.

1. The loss at $w_0$ is $\tfrac{1}{2}(1^2 + 2^2) + 0 = 2.5$.
2. The gradient is $X^\top(Xw_0 - y) = 1 \cdot (-1) + 2 \cdot (-2) = -5$.
3. The gradient step gives $0 - 0.2 \cdot (-5) = 1$.
4. Soft threshold with $\alpha\lambda = 0.2$ gives $w_1 = 0.8$.
5. The residuals are $(0.8 - 1, 1.6 - 2) = (-0.2, -0.4)$, so the loss is $\tfrac{1}{2}(0.04 + 0.16) + 0.8 = 0.9$.

Check the KKT rule: the gradient at $0.8$ is $1(-0.2) + 2(-0.4) = -1$, and $-1 + 1 \cdot \text{sign}(0.8) = 0$. So $0.8$ is the exact lasso answer. For FISTA's first step, $t_1 = (1 + \sqrt{5})/2 \approx 1.618$ and the momentum is $(1 - 1)/1.618 = 0$, so it matches ISTA.

## Build it step by step

The checks test the functions in this order.

1. **`soft_threshold(x, thresh)`** returns an array the same shape as `x`. Use `np.sign`, `np.abs`, and `np.maximum`. Common mistake: using `np.max`, which returns one number for the whole array.
2. **`ridge_closed_form(X, y, lam)`** returns `w` with shape `(d,)`. Build `X.T @ X + lam * np.eye(d)` and call `np.linalg.solve`. Common mistake: using `np.linalg.inv`, which is slower and less accurate.
3. **`effective_degrees_of_freedom(s, lam)`** returns a float when `lam` is a single number, and an array of shape `(m,)` when it is an array. Convert with `np.asarray(lam, dtype=float)` and branch on `.ndim == 0`. For the array case, broadcast to shape `(m, r)` and `.sum(axis=-1)`. Common mistake: summing over the wrong axis, which gives one value per singular value.
4. **`ridge_svd_path(U, s, Vt, y, lambdas)`** returns an array of shape `(m, d)`, one row of weights per penalty. Compute `U.T @ y` once, build the `(m, r)` shrinkage factors by broadcasting, multiply, then `@ Vt`. Common mistake: dividing by `s` alone and forgetting the `+ lambda`.
5. **`ista(X, y, lam, step, iters, w0=None)`** returns `(w, history)`. `w` has shape `(d,)`. `history` has shape `(iters + 1,)`, where `history[k]` is the lasso loss after `k` steps and `history[0]` is the loss at the start. Start from `np.zeros(d)` when `w0` is `None`. Common mistake: thresholding with `lam` instead of `step * lam`.
6. **`fista(X, y, lam, step, iters, w0=None)`** returns `(w, history)` in the same format. Keep `w`, `z`, and `t`. Record the loss at `w`, not at `z`. Common mistake: taking the gradient at `w` instead of `z`.

## Cost and scaling

The table summarises the work each piece needs, for $X$ of shape $n \times d$.

| Task | Rough work |
|---|---|
| Ridge, one $\lambda$ | $nd^2$ to form $X^\top X$, plus $d^3$ to solve |
| Ridge, $m$ values of $\lambda$ | one SVD (a few times $nd^2$), plus about $m \cdot d \cdot r$ |
| One ISTA or FISTA step | about $2nd$ for the gradient |
| Soft thresholding | $d$, one pass over the weights |

The SVD path pays off as soon as you try more than a handful of penalties. ISTA and FISTA never form $X^\top X$, so they scale to large $d$. FISTA needs far fewer steps when the features are correlated.

## Where this breaks

- **Features on different scales.** The penalty treats every weight the same, so a feature in small units gets punished unfairly. Standardise every column first.
- **Penalising the intercept.** You usually don't want to shrink the bias. Centre $X$ and $y$ so the model needs no intercept, or leave it out of the penalty.
- **Choosing $\lambda$.** Too small and you overfit, too large and every weight goes to zero. Pick it on held-out data, as in lesson 1.
- **Correlated features and the lasso.** Among several near-copies, the lasso tends to keep one at random and zero the rest. Ridge spreads weight across all of them.
- **Step too big.** ISTA with a step above $1/L$ can let the loss rise. Estimate $L$ with `np.linalg.eigvalsh(X.T @ X).max()`.

## The exercise

You build six functions. `ridge_closed_form`, `ridge_svd_path`, and `effective_degrees_of_freedom` cover ridge. `soft_threshold`, `ista`, and `fista` cover the lasso.

The checks confirm that `soft_threshold` gives the values in the example above and truly minimises $\tfrac{1}{2}(z - x)^2 + t|z|$. `ridge_closed_form` must match an independent solve, and `ridge_svd_path` must match `ridge_closed_form` at every $\lambda$. `effective_degrees_of_freedom` must equal the rank at $\lambda = 0$ and fall near 0 for huge $\lambda$. ISTA's loss must never go up. FISTA must reach a clearly lower loss than ISTA in 80 steps on correlated data, and its answer must satisfy the KKT conditions.
