---
title: Gradient descent
slug: 03-gradient-descent
part: "Part 1 — Foundations"
summary: Walk downhill on a bowl, keep each step at or under 2/L, and use momentum to take fewer steps.
minutes: 60
gpu: false
objectives:
  - Compute a quadratic loss and its gradient, and check the gradient with a small nudge.
  - Run fixed-step gradient descent and explain why a step above 2/L blows up.
  - Compute the best step along the gradient in closed form.
  - Run heavy-ball momentum and Nesterov's method, and measure how much faster they are.
exercise: 03-gradient-descent
prereqs:
  - 02-linear-regression
---

# Gradient descent

You stand on a hillside in fog. You cannot see the valley. You can feel the slope under your feet, so you step downhill, feel the slope again, and repeat.

Lesson 2 found the bottom of a squared-error bowl with one formula. Logistic regression and a neural network have no such formula. You walk.

By the end you can choose a safe step, and you can add momentum so the walk takes far fewer steps.

## What you already know

Five facts you can check with a pencil:

- Half the sum of squared misses is the lesson 2 loss. On three points, one bad line had loss 0.5.
- Lesson 1 averaged four squared losses, 0.25, 0.25, 1, and 1, and got 0.625. That average is the empirical risk.
- The bowl `(w - 3) ** 2` equals 0 at $w=3$ and 1 at $w=4$. The slope at the bottom is 0.
- Lesson 2 solved a 2 by 2 system. Multiplying that matrix by a vector is `A @ w`.
- The condition number there was about 6.79, the big stretch divided by the small stretch.

## Score a tiny bowl

The two entries of $w$ are the weights. In lesson 1, weights 2 and 1 on 3 and 5 made 11.

Stand at $(w_1,w_2)$. The height is half of $w_1^2+10w_2^2$, minus $w_1+10w_2$.

At $(0,0)$ every term is 0, so the height is 0. At $(1,1)$ the half is $0.5\times(1+10)=5.5$, and you subtract $1+10=11$. The height is $5.5-11=-5.5$.

This shape is a quadratic. The loss here is lesson 2's half-sum, not lesson 1's average of 0.625.

In symbols:

$$
f(w)=\tfrac12 w^\top Aw-b^\top w
$$

$A$ is the diagonal matrix with entries 1 and 10, and $b$ is $(1,10)$. At $(1,1)$, $A$ times $(1,1)$ is $(1,10)$. The dot with $(1,1)$ is 11, and $0.5\times11-11=-5.5$ again.

The bottom is $(1,1)$, because $A$ times $(1,1)$ equals $b$. Call that point $w^\star$. Lesson 2 found it with the normal equations.

Least squares is this bowl with $A=X^\top X$ and $b=X^\top y$. Those rows are the training set, 15 of 20 in lesson 1. Write the height as follows:

```python
loss = 0.5 * (w @ A @ w) - b @ w  # then wrap it as a Python float
```

Drop the 0.5 and the height at $(0,1)$ is 0. The true height there is $5-10=-5$.

## Feel the slope

Nudge the second weight by 1. The heights at $(0,1)$ and $(0,-1)$ are $-5$ and $5-(-10)=15$, so the slope is $(-5-15)/2=-10$.

Nudge the first weight by 1. The heights at $(1,0)$ and $(-1,0)$ are $0.5-1=-0.5$ and $0.5-(-1)=1.5$, so the slope is $(-0.5-1.5)/2=-1$.

The slope at $(0,0)$ is $(-1,-10)$. This vector is the gradient, written $\nabla f$ and read "grad". It points uphill.

In symbols:

$$
\nabla f(w)=Aw-b
$$

At $(0,0)$ that is $-b$, which is $(-1,-10)$, the same vector. It is zero at $w^\star$, where $Aw^\star=b$. Compute the slope as follows:

```python
grad = A @ w - b  # shape (d,)
```

Drop the minus on $b$ and the slope at $(0,0)$ is $(1,10)$. You then walk uphill.

> [!TIP]
> For a check, use `(f(w + h * e) - f(w - h * e)) / (2 * h)`. Here `e` is 0 except a 1 in one entry, and `h` is about `1e-6`.

## Take one downhill step

From $(0,0)$, subtract $0.1$ times $(-1,-10)$. You land at $(0.1,1.0)$.

The height there is $0.5\times(0.01+10)-(0.1+10)=5.005-10.1=-5.095$.

This walk is gradient descent. The step size is $\alpha$ (alpha, said "al-fa"), here $0.1$, also called the learning rate. $w_0$ is $(0,0)$, and $w_k$ is the point after $k$ steps.

In symbols:

$$
w_{k+1}=w_k-\alpha\nabla f(w_k)
$$

Plug in $\alpha=0.1$ and the slope $(-1,-10)$. You get $(0.1,1.0)$ again. Take the step as follows:

```python
w = w - step * grad(w)  # shape (d,)
```

Add the slope instead of subtracting it, and you step to $(-0.1,-1)$, which is uphill.

## See a step that blows up

Try step $0.3$ from $(0,0)$. You land at $(0.3,3.0)$. The steep answer is 1, so the miss grew from 1 to 2.

The slope there is $(0.3-1,\,30-10)=(-0.7,20)$. Subtract $0.3$ times that slope and you land at $(0.51,-3)$, a steep miss of 4.

The steep misses go 1, 2, 4. They double.

An eigenvalue is a stretch along one axis. Here they are 1 and 10, real numbers along the two axes, because $A$ is symmetric. The largest is $L=10$, and the smallest is $\mu$ (mu, said "mew") $=1$.

A step above $2/L$ blows up. Here $2/L=0.2$, so $0.3$ is past it, and the misses double because $|1-0.3\times10|=2$. The safe step is $1/L=0.1$.

> [!KEY]
> The steepest eigenvalue $L$ limits the step. A fixed step above $2/L$ blows up, and $1/L$ is the safe choice.

Take the safe step $0.1$. The slope at $(0.1,1)$ is $(0.1-1,\,0)=(-0.9,0)$. Subtract $0.1$ times that slope and you land at $(0.19,1.0)$.

The flat answer is 1. The flat misses are 1, then 0.9, then 0.81. Each factor is 0.9, which is $1-\mu/L$.

Using $2L=20$ as the limit, instead of $2/L$, picks a step that explodes at once.

## Count the steps

The ratio $L/\mu=10/1=10$ is the condition number, written $\kappa$ (kappa, said "kap-pa"). It is the big stretch divided by the small one. Lesson 2's matrix had condition number about 6.79.

In symbols:

$$
\kappa=L/\mu
$$

Plug in 10 and 1 and you get 10. With step $1/L$, the flat miss shrinks by $1-\mu/L$ each step. When $\kappa=100$, that factor is 0.99.

The number of steps is proportional to $\kappa$. One step size must suit the steep axis and the flat axis, so you zigzag.

> [!KEY]
> Plain gradient descent needs a number of steps proportional to $\kappa=L/\mu$. One step size has to suit both axes.

## Pick the best step on the line

At $(0,0)$ the slope $g$ is $(-1,-10)$. Its squared length is $1+100=101$. $A$ times $g$ is $(-1,-100)$, and the dot with $g$ is $1+1000=1001$.

The best step on that line is $101/1001$, about 0.1009. This choice is exact line search. It minimises the height along the downhill direction.

In symbols:

$$
\alpha=\frac{g^\top g}{g^\top Ag}
$$

Plug in 101 and 1001 and you get about 0.1009 again. A sharp direction gets a short step. Compute the step as follows:

```python
alpha = (g @ g) / (g @ A @ g)
```

Flip the fraction and the step is about 9.9, far above 0.2, so the walk blows up.

## Add the last step

With step 0.1, the first step has no past, so you still land at $(0.1,1.0)$. The plain second step lands at $(0.19,1.0)$.

Add half of the step you took, half of $(0.1,1.0)$, which is $(0.05,0.5)$. You land at $(0.24,1.5)$.

That fraction is the momentum $\beta$ (beta, said "bay-ta"). Here $\beta=0.5$, a sample value. It speeds up on the flat axis.

In symbols, and with $w_{-1}$ treated as $w_0$:

$$
w_{k+1}=w_k-\alpha\nabla f(w_k)+\beta(w_k-w_{k-1})
$$

Plug in $(0.1,1)$, slope $(-0.9,0)$, $\alpha=0.1$, and $\beta=0.5$. You get $(0.24,1.5)$ again. Update the point as follows:

```python
w_next = w - step * grad(w) + momentum * (w - w_prev)
```

Set `w_prev = w` first and the extra term is 0, so you get the plain point $(0.19,1.0)$.

> [!WARNING]
> Compute `w_next` before you change `w_prev`. The line `w_prev, w = w, w_next` updates both in the right order.

The best settings use $L=10$ and $\mu=1$. $\sqrt{10}\approx3.162$, the sum with 1 is about 4.162, and $4/4.162^2$ is about 0.231. The gap $2.162/4.162$ is about 0.519, and that square is about 0.270.

In symbols:

$$
\alpha=\frac{4}{(\sqrt{L}+\sqrt{\mu})^2},\quad\beta=\left(\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}\right)^2
$$

Plug in $L=10$ and $\mu=1$ and you get about 0.231 and 0.270 again. The same step is $(2/(\sqrt{L}+\sqrt{\mu}))^2$, because $2^2=4$.

> [!KEY]
> Momentum keeps the last step. It speeds up on the flat axis and cancels the steep zigzag, cutting the step count from about $\kappa$ to about $\sqrt{\kappa}$.

## Look ahead, then step

From $(0.1,1.0)$, add half of $(0.1,1.0)$. You look ahead to $(0.15,1.5)$. The slope there is $(0.15-1,\,15-10)=(-0.85,5)$.

Subtract $0.1$ times that slope from $(0.15,1.5)$. You land at $(0.235,1.0)$.

This is Nesterov's method. You measure the slope at the look-ahead point.

In symbols, again with $w_{-1}=w_0$:

$$
y_k=w_k+\beta(w_k-w_{k-1}),\qquad w_{k+1}=y_k-\alpha\nabla f(y_k)
$$

Plug in the look-ahead $(0.15,1.5)$ and the slope $(-0.85,5)$. You get $(0.235,1.0)$ again. Step from the look-ahead as follows:

```python
y = w + momentum * (w - w_prev)
w_next = y - step * grad(y)
```

Take the slope at $w$ instead of at $y$, and you get the heavy-ball point $(0.24,1.5)$.

A usual choice is step $1/L=0.1$ and momentum $(\sqrt{L}-\sqrt{\mu})/(\sqrt{L}+\sqrt{\mu})$, about 0.519, not squared. It also takes about $\sqrt{\kappa}$ steps, and it behaves better when the loss is not a bowl.

> [!RECAP]
> - The loss is $\tfrac12 w^\top Aw-b^\top w$, and the gradient is $Aw-b$.
> - A step is $w-\alpha\nabla f(w)$. It is safe at $1/L$ and blows up above $2/L$.
> - Exact line search uses $(g^\top g)/(g^\top Ag)$.
> - Heavy ball adds $\beta$ times the last step. Nesterov takes the slope at the look-ahead point.
> - Every path has shape `(iters + 1, d)`, with `w0` in row 0.

## Build it step by step

The checks call the functions in this order:

1. **`quadratic_loss_and_grad(A, b, w)`** takes $A$ of shape `(d, d)`, and $b$ and $w$ of shape `(d,)`. It returns a float loss and a gradient of shape `(d,)`:

   ```python
   # loss = 0.5 * (w @ A @ w) - b @ w, then float(...)
   # grad = A @ w - b
   ```

   Dropping the 0.5, or returning a numpy scalar, fails the check.

2. **`exact_line_search_step(A, grad_vec)`** returns one float, $(g^\top g)/(g^\top Ag)$:

   ```python
   # (grad_vec @ grad_vec) / (grad_vec @ A @ grad_vec)
   ```

   Flipping the fraction returns a step that is far too long.

3. **`gradient_descent(grad, w0, step, iters)`** returns the path. `grad` is a function you call as `grad(w)`:

   ```python
   path = np.empty((iters + 1, len(w0)))
   # Row 0 is w0.copy(). Each later row is w - step * grad(w).
   ```

   Returning only the last point, or `iters` rows, fails the shape check.

4. **`heavy_ball(grad, w0, step, momentum, iters)`** returns the path. Keep `w` and `w_prev`, both copies of `w0`:

   ```python
   # w_next = w - step * grad(w) + momentum * (w - w_prev)
   # Then w_prev, w = w, w_next
   ```

   Updating `w_prev` first makes the momentum term 0.

5. **`nesterov(grad, w0, step, momentum, iters)`** returns the path. Same two copies, plus a look-ahead point `y`:

   ```python
   # y = w + momentum * (w - w_prev)
   # w_next = y - step * grad(y)
   ```

   The gradient at `w`, instead of at `y`, follows the heavy-ball rule.

## Cost and scaling

Each step costs one gradient. For this bowl that is one product $A w$, about $d^2$ operations, where $d$ is the number of weights. For least squares the gradient $X^\top(Xw-y)$ costs about $2nd$, with $n$ training rows.

The path stores $(\text{iters}+1)\times d$ numbers. Real training keeps only the current point. When $\kappa=10000$, plain descent takes about 10000 steps and momentum takes about 100.

The three walks compare as follows:

|Method|Steps for a fixed gain|Extra memory|
|---|---|---|
|Gradient descent, step $1/L$|about $\kappa$|none|
|Heavy ball|about $\sqrt{\kappa}$|one previous point|
|Nesterov|about $\sqrt{\kappa}$|one previous point|

## Where this breaks

**Step too big.** Above $2/L$ the loss grows every step. If the loss goes up, halve the step.

**Step too small.** The walk crawls. Take a larger step, or estimate $L$ from the largest eigenvalue of $A$.

**Badly scaled features.** They make $\kappa$ large, so every method slows down. Standardise each feature first.

**The loss is not a bowl.** Many valleys break the step counts, and you can stop in a poor one. Neural networks live here.

**Momentum overshoots.** Too much momentum makes the path swing past the bottom. Lower $\beta$ if the loss oscillates.

## The exercise

You build `quadratic_loss_and_grad`, `exact_line_search_step`, `gradient_descent`, `heavy_ball`, and `nesterov`.

The loss must equal $0.5\,w^\top Aw-b^\top w$, and the gradient must match a finite difference. The line-search step must be the closed form, and a local minimiser along that line.

A step of $2.1/L$ must make the error grow, and the path must have one more row than the number of steps. With step $1/L$, the flat error must shrink by exactly $1-\mu/L$ each step.

`heavy_ball` and `nesterov` must follow their updates exactly, with the first step equal to a plain step. With the best heavy-ball step $(2/(\sqrt{L}+\sqrt{\mu}))^2$ and the squared momentum, the heavy ball must beat plain descent by about $\sqrt{\kappa}$.
