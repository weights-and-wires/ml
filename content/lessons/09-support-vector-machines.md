---
title: Support vector machines
slug: 09-support-vector-machines
part: "Part 2 — Classical machine learning"
summary: Choose the widest gap between classes, then train it with hinge loss and a two-at-a-time solver.
minutes: 70
gpu: false
objectives:
  - Compute the hinge loss and a subgradient for a linear classifier.
  - Explain the margin, and what the slack penalty C trades off.
  - Sort each training point by its alpha, and measure how far it breaks its rule.
  - Build one clipped two-variable SMO update, and repeat it until the rules hold.
  - Explain why the SVM uses only inner products, which is what makes kernels possible.
prereqs:
  - 03-gradient-descent
  - 05-logistic-regression
exercise: 09-support-vector-machines
---

# Support vector machines

On data a straight line can separate, logistic regression does not pick one line. Its weights grow without limit, as in lesson 5.

Picture houses on two sides of a street. You want the widest road that still leaves every house on its own side. That road is the support vector machine, or SVM.

You will write its hinge loss, test whether a solution is optimal, and update two multipliers at a time.

## What you already know

These five ideas are the ones this lesson uses:

- A linear classifier scores a point with $w^{\top}x + b$ and predicts the sign. A score of 2 predicts $+1$.
- Gradient descent steps against the slope of a loss.
- $\lVert w \rVert$ is the length of $w$. The vector $[3, 4]$ has length 5, and $\lVert w \rVert^2 = w^{\top}w$.
- Adding $\lVert w \rVert^2$, as ridge does, keeps weights small.
- The matrix of pairwise inner products $x_i^{\top}x_j$ is `X @ X.T`.

## Charge a point that falls short of the road

Take $w = [1, 0]$, $b = 0$, and labels in $\{-1,+1\}$.

Point $[2, 0]$ with label $+1$ scores 2, so the product is 2. Point $[0.5, 0]$ with label $+1$ scores 0.5. Point $[-1, 0]$ with label $-1$ scores $-1$, and $(-1)\times(-1) = 1$. The product is

$$
z_i = y_i(w^{\top}x_i + b) .
$$

These points give $z = 2$, $0.5$, and $1$. Numpy is `z = y * (X @ w + b)`.

The usual mistake is labels in $\{0, 1\}$. Point 3 then scores $-1$, not $1$.

Ask for $z_i \ge 1$. Here $\lVert w \rVert = 1$, so the *margin*, the half-width, is 1. Shrinking $w$ widens the road.

> [!KEY]
> The road's half-width is $1 / \lVert w \rVert$. A smaller $w$ makes a wider road.

With $C = 1$, the shortfalls are $\max(0, 1-2) = 0$, $\max(0, 1-0.5) = 0.5$, and $\max(0, 1-1) = 0$. Half of $\lVert w \rVert^2$ is $0.5$. Add $C$ times the total shortfall: $0.5 + 0.5 = 1$.

That shortfall is the *hinge loss*. The whole objective is the *soft-margin* SVM:

$$
J(w, b) = \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\max(0,\, 1 - z_i) .
$$

The square rewards a wide road. The sum, not a mean, punishes $z_i < 1$. These points return $J = 1$. Large $C$ spends margin to fit points.

Numpy is `0.5 * w @ w + C * np.sum(np.maximum(0.0, 1.0 - z))`.

The usual mistake is `np.mean`. With three points the hinge part becomes $0.5 / 3$, and the loss is about $0.667$ instead of $1$.

> [!WARNING]
> The hinge is a sum over points, not a mean. `np.mean` makes the loss and the slope $n$ times too small.

> [!KEY]
> The objective trades a small $\lVert w \rVert$ against the total hinge. $C$ is the exchange rate.

A *subgradient* is one slope at the corner $z = 1$. *Active* means $z_i < 1$. Treat $z_i = 1$ as inactive.

$$
\frac{\partial J}{\partial w} = w - C\sum_{i:\, z_i < 1} y_i x_i, \qquad
\frac{\partial J}{\partial b} = -C\sum_{i:\, z_i < 1} y_i .
$$

Only point 2 is active. The slope in $w$ is $[1, 0] - 1 \cdot [0.5, 0] = [0.5, 0]$. The slope in $b$ is $-1$.

Numpy uses `grad_w = w - C * (y[z < 1] @ X[z < 1])`.

The usual mistake is `z <= 1`. Point 3 then joins, and the $w$ slope becomes $[-0.5, 0]$.

## Write one multiplier per point

The same problem has a *dual*: one multiplier $\alpha_i \ge 0$ per point, how hard it pushes. On $x = 1$ and $x = -1$, with labels $+1$ and $-1$ and $\alpha = [0.5, 0.5]$,

$$
w = 0.5\cdot 1\cdot 1 + 0.5\cdot(-1)\cdot(-1) = 1.
$$

In general

$$
w = \sum_{i=1}^{n}\alpha_i y_i x_i .
$$

The formula returns $w = 1$ here.

The dual maximises

$$
D(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i}\sum_{j}\alpha_i\alpha_j y_i y_j K_{ij} .
$$

$K$ is the *Gram matrix*, $K_{ij} = x_i^{\top}x_j$. Here the diagonal is 1 and the off-diagonal is $-1$. With $v = \alpha y = [0.5, -0.5]$, $v^{\top}Kv = 1$, so $D = 0.5$.

The *box rule* is $0 \le \alpha_i \le C$. The *balance rule* is $\sum_i \alpha_i y_i = 0$. Here $0.5 - 0.5 = 0$, inside $[0, 10]$.

Numpy is `alpha.sum() - 0.5 * (alpha * y) @ K @ (alpha * y)`.

The usual mistake is dropping the labels in the quadratic term, so $v^{\top}Kv$ is not $1$.

Scores of the training points are

$$
f_i = \sum_{k}\alpha_k y_k K_{ik} + b .
$$

With $b = 0$ both scores have absolute value 1. Numpy is `(alpha * y) @ K + b`.

The data appear only as entries of $K$. Another kernel in place of $K$ fits a curved boundary.

> [!KEY]
> In the dual, the data appear only through $K$. The same solver works for any kernel.

> [!TIP]
> One row is enough for one score: `(alpha * y) @ K[i] + b`. An update needs only two scores.

## Test three groups of points

At the best $\alpha$, write $g_i = y_i f_i - 1$. On the pair above, $g_1 = g_2 = 0$. At $\alpha = 0$ and $b = 0$, both scores are 0, so both $g$ values are $-1$. A point stuck at 0 is then short by $\max(0, -g_i) = 1$.

Each point falls in one group. These are the *KKT conditions*:

| $\alpha_i$ | Place | Rule | Violation |
|---|---|---|---|
| $0$ | Outside, not pushing | $g_i \ge 0$ | $\max(0, -g_i)$ |
| between $0$ and $C$ | On the edge, a support vector | $g_i = 0$ | $\lvert g_i \rvert$ |
| $C$ | Inside, at the cap | $g_i \le 0$ | $\max(0, g_i)$ |

The worst of those violations is the distance from optimal. Within `tol` of 0 or $C$ counts as the boundary. Violation is 0 at $\alpha = 0.5$ and 1 at $\alpha = 0$.

Numpy is `g = y * f - 1` with `alpha <= tol` and `alpha >= C - tol`.

The usual mistake is `alpha == 0`. A stored `1e-16` is then not at the boundary.

## Update two multipliers at a time

*Sequential minimal optimisation* (SMO) changes a pair so the balance rule still holds.

From $\alpha = [0, 0]$, $b = 0$, and $C = 10$, update the pair. Formula indices 1 and 2 are code positions 0 and 1.

Scores are 0, so the errors are $E_1 = 0 - 1 = -1$ and $E_2 = 0 - (-1) = 1$. Then $1 + 1 - 2(-1) = 4$. Labels differ, so the lower clip is $\max(0, 0) = 0$ and the upper clip is $\min(10, 10) = 10$. The new second multiplier is $0 + (-1)(-1 - 1) / 4 = 0.5$. The first becomes $0 + (1)(-1)(0 - 0.5) = 0.5$.

That pair of moves is one SMO step:

$$
\eta = K_{ii} + K_{jj} - 2K_{ij}, \qquad
\alpha_j^{\text{new}} = \alpha_j + \frac{y_j(E_i - E_j)}{\eta} .
$$

$\eta$ is the squared distance between the points, and $E_i = f_i - y_i$ uses the old $\alpha$. If $\eta \le 10^{-12}$, return unchanged. The formula returns $\eta = 4$ and $\alpha_2 = 0.5$.

Clip into $[L, H]$ using the old alphas. Different labels use $L = \max(0, \alpha_j - \alpha_i)$ and $H = \min(C, C + \alpha_j - \alpha_i)$. Matching labels use $L = \max(0, \alpha_i + \alpha_j - C)$ and $H = \min(C, \alpha_i + \alpha_j)$. Here $L = 0$ and $H = 10$.

If $L \ge H$, return unchanged. Otherwise recover $\alpha_i$:

$$
\alpha_i^{\text{new}} = \alpha_i + y_i y_j(\alpha_j - \alpha_j^{\text{new}}) .
$$

The same numbers give $\alpha_1 = 0.5$. Write $\Delta$ for each change. Two candidate biases land a point on the edge:

$$
b_1 = b - E_i - y_i\Delta_i K_{ii} - y_j\Delta_j K_{ij}, \qquad
b_2 = b - E_j - y_i\Delta_i K_{ij} - y_j\Delta_j K_{jj} .
$$

Use $b_1$ if the new $\alpha_i$ is strictly inside $(0, C)$, else $b_2$ if $\alpha_j$ is, else $(b_1 + b_2) / 2$. Here $\Delta_1 = \Delta_2 = 0.5$, so $b_1 = 0 - (-1) - 0.5 - 0.5 = 0$. The new bias is 0.

Read `K[i]` and `K[j]` before either write. The usual mistake is reading $E_j$ after $\alpha_i$ changes.

Afterwards $w = 1$ and both $z$ values are 1. $D$ rises from 0 to 0.5, and a step never lowers it.

> [!TIP]
> After a step, every alpha must sit in $[0, C]$, and `alpha @ y` must be about 0.

Keep these checks in mind:

> [!RECAP]
> - $z_i = y_i(w^{\top}x_i + b)$, and the hinge is 0 once $z_i \ge 1$.
> - Only $z_i < 1$ enters the subgradient, and the hinge is a sum.
> - $g_i = y_i f_i - 1$ tests the KKT group of $\alpha_i$.
> - SMO adds $y_j(E_i - E_j)/\eta$ to $\alpha_j$, clips to $[L, H]$, and moves $\alpha_i$ back.
> - The new bias is $b_1$, $b_2$, or their average.

## Build it step by step

Build the four functions in the order the checks test them:

1. **`hinge_loss_and_subgradient(w, b, X, y, C)`**. `w` is `(d,)`, `X` is `(n, d)`, and labels are in $\{-1,+1\}$. Returns `(loss, grad_w, grad_b)`. Mask with `z < 1`, not `z <= 1`. The arrays look like this:

```python
z = y * (X @ w + b)                 # shape (n,)
grad_w = w - C * (y[z < 1] @ X[z < 1])  # shape (d,)
```

2. **`kkt_violation(alpha, y, K, b, C, tol)`**. Returns the worst violation, one float. Build `f` and `g`, and use `tol` on the box edges. The common mistake is `alpha == 0`. Scores have this shape:

```python
f = (alpha * y) @ K + b  # shape (n,)
g = y * f - 1.0          # shape (n,)
```

3. **`smo_step(alpha, y, K, b, C, i, j)`**. Returns `(new_alpha, new_b)`. Copy `alpha`. Return unchanged if `i == j`, if `eta <= 1e-12`, or if `L >= H`. Compute both errors before any write. The common mistake is an error read after the first write. The copy looks like this:

```python
alpha = alpha.copy()  # shape (n,)
eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
```

4. **`fit_smo(K, y, C, max_iter, tol)`**. Returns `(alpha, b)` from zeros. Scan `i`. Update `i` only when $y_i E_i < -\mathrm{tol}$ with $\alpha_i < C$, or $y_i E_i > \mathrm{tol}$ with $\alpha_i > 0$. Then scan `j` from 0 and keep the first move of `alpha[j]` past `1e-7`. Stop on small `kkt_violation`, a pass with no change, or `max_iter`. The common mistake is missing the no-change stop. Start from this array:

```python
alpha = np.zeros(len(y))  # shape (n,)
b = 0.0
```

## Cost and scaling

The Gram matrix has $n^2$ entries. The steps compare as follows:

| Step | Time | Memory |
|---|---|---|
| Build $K$ | grows like $n^2 d$, once | $n^2$ numbers |
| One SMO step | grows like $n$ | nothing beyond $K$ |
| One full pass | up to about $n^2$ dots | nothing beyond $K$ |
| Predict one new point | support vectors times $d$ | the support vectors |

Doubling $n$ makes $K$ four times bigger. At 50,000 points, $K$ is about 2.5 billion numbers, near 20 GB. Only $\alpha_i > 0$ is needed to predict.

## Where this breaks

**Choosing $C$.** A huge $C$ lets one noisy point bend the road. A tiny $C$ ignores the data. Pick $C$ by cross-validation.

**Too many points.** Past about 100,000 points, $K$ does not fit. Use hinge subgradient steps for a linear SVM, or approximate the kernel.

**A fixed scan.** Index order takes many passes. A solver such as libsvm picks the pair that breaks the rules most.

**No probabilities.** You get a score and a sign. Platt scaling fits a one-feature logistic regression on those scores.

## The exercise

You implement `hinge_loss_and_subgradient`, `kkt_violation`, `smo_step`, and `fit_smo`.

The checks match the loss to a direct value and the slope to finite differences off the corner. `kkt_violation` is near 0 at an optimum and large at $\alpha = 0$. `smo_step` stays in $[0, C]$, keeps $\sum_i \alpha_i y_i = 0$, and does not lower the dual. `fit_smo` drives the violation under tolerance and matches a reference $w$.
