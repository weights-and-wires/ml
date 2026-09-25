---
title: Support vector machines
slug: 09-support-vector-machines
part: "Part 2 — Classical machine learning"
summary: Pick the boundary with the widest gap between classes, and train it with the hinge loss and a two-at-a-time solver.
minutes: 70
gpu: false
objectives:
  - Compute the hinge loss objective and its subgradient for a linear classifier.
  - Explain the margin and what the slack penalty C trades off.
  - Sort every training point into one of three groups from its alpha value, and measure how far it breaks its rule.
  - Build one clipped two-variable SMO update and a loop that repeats it until the rules hold.
  - Explain why the SVM only needs inner products between points, which is what makes kernels possible.
prereqs:
  - 03-gradient-descent
  - 05-logistic-regression
exercise: 09-support-vector-machines
---

# Support vector machines

When two classes can be separated by a straight line, there are usually infinitely many lines that do it. Logistic regression does not pick one. On perfectly separable data, its weights keep growing forever, as you saw in lesson 5.

A support vector machine (SVM) makes a clear choice. It picks the line that leaves the widest possible gap to the nearest point of either class. Picture a road between two groups of houses: the SVM builds the widest road it can without knocking any houses down.

This lesson shows you the SVM's loss, the hinge loss, and a second way of writing the same problem in terms of one number per training point. You then build SMO, the classic algorithm that solves that second form by changing two numbers at a time.

## What you already know

- A linear classifier scores a point with $w^{\top}x + b$ and predicts the sign of the score.
- Gradient descent lowers a loss by stepping against its gradient.
- The norm $\lVert w \rVert$ is the length of a vector, and $\lVert w \rVert^2 = w^{\top}w$.
- Adding $\lVert w \rVert^2$ to a loss, as ridge regression does, keeps the weights small.
- A matrix of all pairwise inner products $x_i^{\top}x_j$ is `X @ X.T`.

## Labels, scores, and the margin

SVMs use labels $y_i \in \{-1, +1\}$ instead of $\{0, 1\}$. That makes one quantity very handy:

$$
z_i = y_i(w^{\top}x_i + b) .
$$

Read it aloud: take point $i$'s score and multiply by its label. If the point is on the correct side, the score and the label have the same sign, so $z_i$ is positive. If it is on the wrong side, $z_i$ is negative. The bigger $z_i$ is, the more comfortably correct the point is.

```python
z = y * (X @ w + b)    # shape (n,)
```

The SVM asks every point to be correct with room to spare: $z_i \ge 1$. The points with $z_i = 1$ sit exactly on the edge of the road. The road's half-width works out to $1 / \lVert w \rVert$. So making $\lVert w \rVert$ small makes the road wide.

> [!KEY]
> The road's half-width is $1 / \lVert w \rVert$. To widen the road, you shrink $w$.

Real data is messy, and some points will not fit. So the SVM lets points fall short of 1, and charges a penalty for how far short they fall. The constant $C > 0$ sets how expensive that penalty is.

## The hinge loss

The penalty for point $i$ is the **hinge loss**, $\max(0, 1 - z_i)$. It is zero for a point that clears the edge of the road. It grows in a straight line as the point drifts inside the road or onto the wrong side. The whole objective is

$$
J(w, b) = \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\max(0,\, 1 - z_i) .
$$

Read it aloud: half the squared length of $w$, which rewards a wide road, plus $C$ times the total hinge loss, which punishes points that break the rule. The sum is over all points, not an average. A large $C$ says "fit the training points at almost any cost". A small $C$ says "a wide road matters more than a few mistakes".

> [!KEY]
> The SVM objective trades a wide road, from a small $\lVert w \rVert$, against a small total hinge loss. $C$ sets the exchange rate between the two.

```python
loss = 0.5 * w @ w + C * np.sum(np.maximum(0.0, 1.0 - z))
```

> [!WARNING]
> The objective sums the hinge losses over the points. It does not average them. If you write `np.mean`, the hinge part of your loss and your slope comes out $n$ times too small.

### Its slope

The hinge has a corner at $z = 1$, so strictly it has no gradient there. Away from the corner, it has a slope, and you can use that slope like a gradient. That is called a **subgradient**. A point with $z_i < 1$ is **active**: it contributes to the slope. A point with $z_i \ge 1$ contributes nothing. At exactly $z_i = 1$, treat it as inactive.

$$
\frac{\partial J}{\partial w} = w - C\sum_{i:\, z_i < 1} y_i x_i, \qquad
\frac{\partial J}{\partial b} = -C\sum_{i:\, z_i < 1} y_i .
$$

Read it aloud: the slope in $w$ is $w$ itself, from the $\frac{1}{2}\lVert w\rVert^2$ term, minus $C$ times the sum of $y_i x_i$ over the active points only. The slope in $b$ is minus $C$ times the sum of the active labels.

```python
active = z < 1.0
grad_w = w - C * (y[active] @ X[active])
grad_b = -C * np.sum(y[active])
```

### A worked example

Take $w = [1, 0]$, $b = 0$, and $C = 1$, with three points:

| $i$ | $x_i$ | $y_i$ | score $w^{\top}x_i + b$ | $z_i$ | hinge $\max(0, 1 - z_i)$ | active? |
|---|---|---|---|---|---|---|
| 1 | $[2, 0]$ | $+1$ | $2$ | $2$ | $0$ | no |
| 2 | $[0.5, 0]$ | $+1$ | $0.5$ | $0.5$ | $0.5$ | yes |
| 3 | $[-1, 0]$ | $-1$ | $-1$ | $1$ | $0$ | no, exactly on the edge |

- Loss: $\frac{1}{2}(1^2 + 0^2) + 1 \cdot (0 + 0.5 + 0) = 1.0$.
- Slope in $w$: $[1, 0] - 1 \cdot (+1)[0.5, 0] = [0.5, 0]$.
- Slope in $b$: $-1 \cdot (+1) = -1$.

## The dual: one number per point

There is a second way to write the same problem, called the **dual**. Instead of searching over $w$ and $b$, you search over one number $\alpha_i$ for each training point. Think of $\alpha_i$ as how hard point $i$ pushes on the road. The weights are then a weighted sum of the training points:

$$
w = \sum_{i=1}^{n}\alpha_i y_i x_i .
$$

Read it aloud: every point pushes in the direction of its own position, signed by its label, with strength $\alpha_i$. Points with $\alpha_i = 0$ do not push at all.

The dual picks the $\alpha$ values that make this quantity as large as possible:

$$
D(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i}\sum_{j}\alpha_i\alpha_j y_i y_j K_{ij} .
$$

Here $K$ is the $n \times n$ **Gram matrix** of inner products, $K_{ij} = x_i^{\top}x_j$, which is `X @ X.T`. The $\alpha$ values must obey two rules:

- The **box rule**: $0 \le \alpha_i \le C$ for every point.
- The **balance rule**: $\sum_i \alpha_i y_i = 0$. The pushes from the $+1$ side and the $-1$ side must cancel.

```python
v = alpha * y
dual = alpha.sum() - 0.5 * v @ K @ v
```

In terms of $\alpha$, the score of any training point $i$ is

$$
f_i = \sum_{k}\alpha_k y_k K_{ik} + b .
$$

```python
f = (alpha * y) @ K + b    # shape (n,), score of every training point
```

> [!TIP]
> When you need the score of one point only, use one row of the Gram matrix: `(alpha * y) @ K[i] + b`. An SMO step needs just two scores, so this saves you from computing all $n$.

### Why this form matters

Look at $D(\alpha)$ and $f_i$. The data only shows up through $K$, the inner products between points. You never need the points themselves. The next lesson swaps $K$ for other functions, called kernels, that act like inner products in a much richer space. That trick only works in the dual.

> [!KEY]
> In the dual, the data appears only through the inner products in $K$. Swap $K$ for another kernel, and the same solver fits a nonlinear boundary.

## Three kinds of point

At the best $\alpha$, every training point lands in one of three groups. Write $g_i = y_i f_i - 1$. This is $z_i - 1$, how far point $i$ is past the edge of the road.

| $\alpha_i$ | What the point is | Rule it must obey |
|---|---|---|
| $\alpha_i = 0$ | Safely outside the road; it does not push | $g_i \ge 0$ |
| $0 < \alpha_i < C$ | Exactly on the edge; a **support vector** | $g_i = 0$ |
| $\alpha_i = C$ | Inside the road or on the wrong side; pushing as hard as allowed | $g_i \le 0$ |

These are the **KKT conditions**, named after Karush, Kuhn, and Tucker. When they all hold, $\alpha$ is optimal. That gives you a stopping test. For each point, measure how badly it breaks its rule:

- If $\alpha_i$ is at 0, the violation is $\max(0, -g_i)$.
- If $\alpha_i$ is at $C$, the violation is $\max(0, g_i)$.
- Otherwise, the violation is $|g_i|$.

The worst violation over all points tells you how far from optimal you are. Floating-point numbers rarely land exactly on 0 or $C$, so treat "within `tol` of 0" as at 0, and the same for $C$.

```python
g = y * f - 1.0
at_lower = alpha <= tol
at_upper = alpha >= C - tol
```

## SMO: change two at a time

You could try to improve one $\alpha_i$ at a time. But changing one alone breaks the balance rule. So Sequential Minimal Optimisation (SMO) changes a pair, $\alpha_i$ and $\alpha_j$, together, in a way that keeps the balance. With only two numbers moving, the best step has a closed form.

First compute each point's error, its score minus its label: $E_i = f_i - y_i$ and $E_j = f_j - y_j$. Use the current $\alpha$ and $b$ for both, before changing anything. Then:

$$
\eta = K_{ii} + K_{jj} - 2K_{ij}, \qquad
\alpha_j^{\text{new}} = \alpha_j + \frac{y_j(E_i - E_j)}{\eta} .
$$

Read it aloud: $\eta$ (eta) is the squared distance between points $i$ and $j$. The new $\alpha_j$ moves by the difference in errors, signed by $y_j$, divided by that distance. If $\eta$ is zero or tiny, the two points are the same and there is no useful step, so return unchanged.

### Stay inside the box

The unclipped step can leave the box $[0, C]$. Keeping the balance rule fixes $\alpha_i y_i + \alpha_j y_j$, which confines $\alpha_j$ to a range $[L, H]$:

- If $y_i \ne y_j$: $L = \max(0, \alpha_j - \alpha_i)$ and $H = \min(C, C + \alpha_j - \alpha_i)$.
- If $y_i = y_j$: $L = \max(0, \alpha_i + \alpha_j - C)$ and $H = \min(C, \alpha_i + \alpha_j)$.

Here $\alpha_i$ and $\alpha_j$ are the old values. If $L \ge H$, there is no room to move, so return unchanged. Otherwise clip $\alpha_j^{\text{new}}$ into $[L, H]$, then move $\alpha_i$ by the opposite amount so the balance holds:

$$
\alpha_i^{\text{new}} = \alpha_i + y_i y_j(\alpha_j - \alpha_j^{\text{new}}) .
$$

### Update the bias

Write $\Delta_i = \alpha_i^{\text{new}} - \alpha_i$ and $\Delta_j = \alpha_j^{\text{new}} - \alpha_j$. Two candidate biases make point $i$ or point $j$ land exactly on the edge:

$$
b_1 = b - E_i - y_i\Delta_i K_{ii} - y_j\Delta_j K_{ij}, \qquad
b_2 = b - E_j - y_i\Delta_i K_{ij} - y_j\Delta_j K_{jj} .
$$

Use $b_1$ if $0 < \alpha_i^{\text{new}} < C$. Otherwise use $b_2$ if $0 < \alpha_j^{\text{new}} < C$. Otherwise use their average $\frac{1}{2}(b_1 + b_2)$.

### A worked SMO step

Take two one-dimensional points, $x_1 = 1$ with $y_1 = +1$ and $x_2 = -1$ with $y_2 = -1$. With a linear kernel, $K = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$. Start from $\alpha = [0, 0]$, $b = 0$, and $C = 10$. Update the pair $(i, j) = (1, 2)$.

- Scores: $f_1 = f_2 = 0$, so errors $E_1 = 0 - 1 = -1$ and $E_2 = 0 - (-1) = 1$.
- $\eta = 1 + 1 - 2(-1) = 4$.
- Labels differ, so $L = \max(0, 0) = 0$ and $H = \min(10, 10) = 10$.
- $\alpha_2^{\text{new}} = 0 + (-1)(-1 - 1)/4 = 0.5$. It is inside $[0, 10]$, so no clipping.
- $\alpha_1^{\text{new}} = 0 + (1)(-1)(0 - 0.5) = 0.5$.
- $\Delta_1 = \Delta_2 = 0.5$, so $b_1 = 0 + 1 - 0.5 \cdot 1 - (-1)(0.5)(-1) = 0$. Since $0 < 0.5 < 10$, the new $b$ is 0.

The result is $\alpha = [0.5, 0.5]$ and $b = 0$. Then $w = 0.5 \cdot 1 \cdot 1 + 0.5 \cdot (-1)(-1) = 1$, and both points have $z = 1$: they sit exactly on the edge. Each SMO step solves its two-variable problem exactly, so the dual objective never goes down.

> [!TIP]
> After each step, test `np.all((alpha >= 0) & (alpha <= C))` and `abs(alpha @ y) < 1e-9`. If either fails, look at your clipping or your update of $\alpha_i$.

> [!RECAP]
> - $z_i = y_i(w^{\top}x_i + b)$, and the hinge loss $\max(0, 1 - z_i)$ is zero once $z_i \ge 1$.
> - Only points with $z_i < 1$ add to the subgradient.
> - In the dual, `f = (alpha * y) @ K + b`, and $g_i = y_i f_i - 1$ tells you whether point $i$ breaks the rule for its group.
> - An SMO step moves $\alpha_j$ by $y_j(E_i - E_j)/\eta$, clips it to $[L, H]$, then moves $\alpha_i$ the opposite way.
> - The new bias is $b_1$, $b_2$, or their average, depending on which new $\alpha$ lies strictly inside $(0, C)$.

## Build it step by step

Work through the functions in the order the checks test them.

1. **`hinge_loss_and_subgradient(w, b, X, y, C)`.** Takes `w` `(d,)`, a float `b`, `X` `(n, d)`, labels `y` `(n,)` in $\{-1, +1\}$, and `C`. Returns `(loss, grad_w, grad_b)`, with `grad_w` of shape `(d,)` and the others floats. Compute `z`, then use `np.maximum` for the loss and a boolean mask `z < 1.0` for the slope. The common mistake is using `<=`, which counts points exactly on the edge as active.
2. **`kkt_violation(alpha, y, K, b, C, tol)`.** Returns one float, the worst violation. Compute `f` and `g`, build the three masks, and fill a violation array with `np.maximum` and `np.abs`. Return its `.max()`. The common mistake is testing `alpha == 0` exactly instead of within `tol`.
3. **`smo_step(alpha, y, K, b, C, i, j)`.** Returns `(new_alpha, new_b)`. Start with `alpha = alpha.copy()` so you never change the caller's array. Return unchanged if `i == j`, if `eta <= 1e-12`, or if `L >= H`. Otherwise follow the steps above in order: errors, `eta`, box, clip with `min(max(...))`, recover `alpha_i`, then the bias. The common mistake is computing `E_j` after you have already changed `alpha[i]`.
4. **`fit_smo(K, y, C, max_iter, tol)`.** Returns `(alpha, b)`. Start from zeros. In each outer pass, loop over `i`. Skip `i` unless it breaks its rule by more than `tol`: $y_i E_i < -\text{tol}$ with $\alpha_i < C$, or $y_i E_i > \text{tol}$ with $\alpha_i > 0$. For a breaking `i`, scan `j` from 0, skip `j == i`, and keep the first `smo_step` that changes `alpha[j]` by more than `1e-7`. Stop when `kkt_violation` drops below `tol`, when a pass changes nothing, or after `max_iter` passes. The common mistake is forgetting the "nothing changed" exit, which makes the loop run all `max_iter` passes.

## Cost and scaling

The Gram matrix is the main cost. It has $n^2$ entries, one for every pair of training points.

| Step | Time | Memory |
|---|---|---|
| Build $K$ | grows with $n^2 d$, once | $n^2$ numbers |
| One SMO step | grows with $n$ (two scores) | nothing beyond $K$ |
| One full pass over the data | up to $n^2$ steps' worth of scores | nothing beyond $K$ |
| Predict one new point | grows with (number of support vectors) $\times d$ | the support vectors |

Doubling $n$ makes $K$ four times as big. At 50,000 points, $K$ holds 2.5 billion numbers, about 20 GB. That is the practical wall. Prediction stays cheap, because only points with $\alpha_i > 0$ matter, and that is usually a small fraction of them.

## Where this breaks

**Choosing $C$.** Too large a $C$ lets one noisy point bend the road. Too small a $C$ ignores the data. Pick $C$ with the cross-validation from lesson 7.

**Too many points.** The $n \times n$ Gram matrix stops fitting in memory past roughly 100,000 points. For linear SVMs, run subgradient descent on the hinge loss instead. It needs no Gram matrix. For kernel SVMs, approximate the kernel, as the next lesson shows.

**Slow convergence.** Scanning pairs in a fixed order, as the exercise does, takes many passes. Production libraries such as libsvm pick the pair that breaks the rules the most, and converge far faster.

**No probabilities.** An SVM gives you a score and a sign, not a probability. To get probabilities, fit a one-feature logistic regression on the SVM's scores. This is called Platt scaling.

## The exercise

You build the pieces of a soft-margin SVM solver. Implement `hinge_loss_and_subgradient` to return the objective and its slope at a given `w` and `b`. Implement `kkt_violation` to sort points into the three groups and return the worst violation. Implement `smo_step` for one clipped two-variable update, and `fit_smo` to repeat that update from $\alpha = 0$ until the rules hold.

The checks compare your loss against a direct computation and your slope against finite differences away from the corner. They confirm that `kkt_violation` is near zero at a known optimum and large at $\alpha = 0$. They check that `smo_step` keeps every $\alpha$ inside $[0, C]$, keeps $\sum_i \alpha_i y_i = 0$, and never lowers the dual objective. The last two checks run `fit_smo` on a small separable problem and confirm that the violation falls below tolerance and that $w = \sum_i \alpha_i y_i x_i$ matches a reference solution.
