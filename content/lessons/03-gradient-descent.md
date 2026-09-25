---
title: Gradient descent
slug: 03-gradient-descent
part: "Part 1 — Foundations"
summary: Walk downhill on a loss one step at a time, pick a safe step size, and use momentum to get there far faster.
minutes: 60
gpu: false
objectives:
  - Compute a quadratic loss and its gradient, and check the gradient numerically.
  - Build fixed-step gradient descent and explain why a step above 2 / L blows up.
  - Compute the best step size along the gradient in closed form.
  - Build heavy-ball momentum and Nesterov's method, and measure their speedup.
exercise: 03-gradient-descent
prereqs:
  - 02-linear-regression
---

# Gradient descent

In lesson 2 you solved least squares in one go, with a formula. Most losses in machine learning have no such formula. Logistic regression and neural networks are two examples. For those, you need a method that finds the minimum step by step.

Gradient descent is that method. Picture yourself on a foggy hillside, trying to reach the valley floor. You can't see the valley, but you can feel which way the ground slopes under your feet. So you take a step downhill, feel the slope again, and repeat.

This lesson answers two practical questions. How big should each step be? And how many steps does it take? You also build momentum, a small change that often cuts the number of steps from thousands to tens.

## What you already know

- Least squares has the loss $\tfrac{1}{2}\lVert Xw - y\rVert^2$ and gradient $X^\top(Xw - y)$ (lesson 2).
- A derivative tells you how fast a function changes as you nudge its input.
- A symmetric matrix has real eigenvalues, and its eigenvectors point along its main axes.
- The condition number is the ratio of the largest to the smallest stretch factor (lesson 2).

## The loss you practise on

You test every method on one simple bowl-shaped function, a quadratic:

$$
f(w) = \tfrac{1}{2} w^\top A w - b^\top w
$$

Read it aloud: $w$ is a vector of length $d$. $A$ is a $d \times d$ symmetric matrix with all eigenvalues positive, which makes $f$ a bowl with one lowest point. $b$ is a vector of length $d$. The first term is a curved bowl, and the second term tilts it.

Least squares is exactly this shape, with $A = X^\top X$ and $b = X^\top y$, so everything here carries over.

```python
loss = 0.5 * w @ A @ w - b @ w
```

The gradient is the vector of partial derivatives, one per entry of $w$. It points in the direction of steepest uphill. For this function it is

$$
\nabla f(w) = Aw - b
$$

Read it aloud: multiply $A$ by $w$ and subtract $b$. The gradient is zero exactly at the bottom of the bowl, $w^\star$, where $Aw^\star = b$.

```python
grad = A @ w - b
```

### Check a gradient numerically

A wrong gradient is the most common bug in optimisation code. You can catch it with a finite difference. Nudge one entry of $w$ by a tiny amount $h$, see how much $f$ changes, and divide by $h$. That approximates one entry of the gradient. The checks do this for you, so a sign error or a missing $\tfrac{1}{2}$ shows up at once.

> [!TIP]
> To check one entry by hand, use a two-sided difference: `(f(w + h * e) - f(w - h * e)) / (2 * h)`. Here `e` is all zeros except a 1 at that entry, and `h` is about `1e-6`. It is more accurate than a one-sided nudge.

## Fixed-step gradient descent

Each step moves a little way downhill:

$$
w_{k+1} = w_k - \alpha \, \nabla f(w_k)
$$

Read it aloud: the next point $w_{k+1}$ is the current point $w_k$ minus the step size $\alpha$ times the gradient at $w_k$. The minus sign turns "uphill" into "downhill". The step size is also called the learning rate.

```python
for k in range(iters):
    w = w - step * grad(w)
```

### How big a step is safe

The bowl has a steepest direction and a flattest direction. Its steepness in each direction is an eigenvalue of $A$. Call the largest eigenvalue $L$ and the smallest $\mu$ (the Greek letter mu).

Too big a step overshoots the valley and lands higher up the opposite side. Each overshoot is bigger than the last, and the method blows up. For a quadratic, this happens when the step exceeds $2/L$.

> [!KEY]
> The steepest direction, with eigenvalue $L$, limits the step. A fixed step above $2/L$ blows up, and $1/L$ is the safe, standard choice.

### How many steps it takes

With step $1/L$, the error along the flattest direction shrinks by a factor of $1 - \mu/L$ every step. The error is the distance from $w_k$ to $w^\star$.

The ratio $\kappa = L/\mu$ is the condition number of $A$. When $\kappa = 100$, the error shrinks by a factor of $0.99$ per step. That is slow: you need several hundred steps to gain a few digits. As a rule, fixed-step gradient descent needs a number of steps proportional to $\kappa$.

The reason is that one step size must serve every direction. It has to be small enough for the steep direction, which makes it tiny for the flat one. You zigzag down a narrow valley.

> [!KEY]
> Plain gradient descent needs a number of steps proportional to $\kappa = L/\mu$. One step size has to suit both the steepest and the flattest direction.

## A worked example by hand

Take $A = \begin{bmatrix} 1 & 0 \\ 0 & 10 \end{bmatrix}$ and $b = (1, 10)$. The minimum is at $w^\star = (1, 1)$. Here $\mu = 1$ and $L = 10$. Start at $w_0 = (0, 0)$ with step $\alpha = 0.1$, which is $1/L$.

1. The gradient at $w_0$ is $A w_0 - b = (-1, -10)$.
2. The first step gives $w_1 = (0, 0) - 0.1 \cdot (-1, -10) = (0.1, 1.0)$.
3. The loss there is $f(w_1) = \tfrac{1}{2}(0.1^2 + 10 \cdot 1^2) - (0.1 + 10) = 5.005 - 10.1 = -5.095$.
4. The gradient at $w_1$ is $(0.1 - 1, 10 - 10) = (-0.9, 0)$.
5. The second step gives $w_2 = (0.1 + 0.09, 1.0) = (0.19, 1.0)$.

The steep second coordinate reached its answer in one step. The flat first coordinate crawls: its error goes $1, 0.9, 0.81$, shrinking by $1 - \mu/L = 0.9$ each time.

## The best step along the gradient

Instead of a fixed step, you can choose the step that goes furthest downhill along the current gradient direction $g$. This is called exact line search. For a quadratic it has a closed form:

$$
\alpha = \frac{g^\top g}{g^\top A g}
$$

Read it aloud: the top is the squared length of the gradient. The bottom measures how curved the bowl is along that direction. A sharply curved direction gets a short step.

```python
alpha = (g @ g) / (g @ A @ g)
```

In the worked example, at $w_0$ the gradient is $g = (-1, -10)$. Then $g^\top g = 101$ and $g^\top A g = 1 + 1000 = 1001$, so $\alpha = 101/1001 \approx 0.1009$.

## Momentum: the heavy ball

Imagine rolling a heavy ball down the valley instead of walking. The ball builds up speed along the flat direction, where the slope keeps pointing the same way. In the steep direction, the back-and-forth pushes cancel out. The heavy-ball method adds exactly that memory of the last step:

$$
w_{k+1} = w_k - \alpha \, \nabla f(w_k) + \beta \, (w_k - w_{k-1})
$$

Read it aloud: take a normal gradient step, then add $\beta$ times the step you took last time. $\beta$ is the momentum, a number between 0 and 1. For the very first step there is no previous step, so treat $w_{-1}$ as equal to $w_0$.

```python
w_next = w - step * grad(w) + momentum * (w - w_prev)
w_prev, w = w, w_next
```

> [!WARNING]
> Compute `w_next` before you touch `w_prev`. If you set `w_prev = w` first, then `w - w_prev` is zero and the momentum term disappears. The tuple assignment updates both at once, in the right order.

With well-chosen values, the number of steps drops from about $\kappa$ to about $\sqrt{\kappa}$. When $\kappa = 10{,}000$, that is the gap between about 10,000 steps and about 100. For a quadratic, the best settings are

$$
\alpha = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \quad \beta = \left(\frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}\right)^2
$$

```python
step = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
momentum = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
```

> [!KEY]
> Momentum remembers the last step, so it speeds up along the flat direction and cancels the zigzag in the steep one. That cuts the step count from about $\kappa$ to about $\sqrt{\kappa}$.

Continue the worked example with $\alpha = 0.1$ and $\beta = 0.5$. The first step equals plain gradient descent, $w_1 = (0.1, 1.0)$. The second step is $w_2 = (0.1, 1.0) - 0.1 \cdot (-0.9, 0) + 0.5 \cdot (0.1, 1.0) = (0.24, 1.5)$.

## Nesterov's method: look before you leap

Nesterov's method changes one thing. It first jumps ahead along the momentum, and only then measures the slope at that look-ahead point:

$$
y_k = w_k + \beta \, (w_k - w_{k-1}), \qquad w_{k+1} = y_k - \alpha \, \nabla f(y_k)
$$

Read it aloud: $y_k$ is where momentum alone would carry you. Take the gradient step from $y_k$, using the gradient at $y_k$, not at $w_k$. Again, treat $w_{-1}$ as $w_0$.

```python
y = w + momentum * (w - w_prev)
w_next = y - step * grad(y)
```

A common setting is $\alpha = 1/L$ and $\beta = (\sqrt{L} - \sqrt{\mu})/(\sqrt{L} + \sqrt{\mu})$. It also needs about $\sqrt{\kappa}$ steps, and it behaves better than the heavy ball on functions that are not quadratics.

In the worked example with $\alpha = 0.1$ and $\beta = 0.5$, $w_1 = (0.1, 1.0)$ again. Then $y_1 = (0.1, 1.0) + 0.5 \cdot (0.1, 1.0) = (0.15, 1.5)$. The gradient there is $(-0.85, 5)$, so $w_2 = (0.15 + 0.085, 1.5 - 0.5) = (0.235, 1.0)$.

> [!RECAP]
> - For the quadratic, the loss is $\tfrac{1}{2} w^\top A w - b^\top w$ and the gradient is $Aw - b$.
> - A gradient step is $w - \alpha \nabla f(w)$. It is safe at $\alpha = 1/L$ and blows up above $2/L$.
> - Exact line search uses $\alpha = g^\top g / g^\top A g$.
> - The heavy ball adds $\beta (w_k - w_{k-1})$. Nesterov takes its gradient at the look-ahead point $y_k$. Both start with $w_{-1} = w_0$.
> - Every trajectory has shape `(iters + 1, d)`, with `w0` in row 0.

## Build it step by step

The checks test the functions in this order. Every trajectory is an array of shape `(iters + 1, d)`, with row 0 equal to `w0` and row `k` the point after `k` steps.

1. **`quadratic_loss_and_grad(A, b, w)`** returns a tuple `(loss, grad)`: a float and an array of shape `(d,)`. Use `w @ A @ w`, `b @ w`, and `A @ w - b`. Common mistake: forgetting the `0.5` in the loss, or returning a numpy scalar instead of `float(loss)`.
2. **`exact_line_search_step(A, grad_vec)`** returns a float, the closed-form step above. Two dot products and a division. Common mistake: flipping the fraction.
3. **`gradient_descent(grad, w0, step, iters)`** returns the trajectory. `grad` is a function you call as `grad(w)`. Allocate with `np.empty((iters + 1, d))`, store `w0` in row 0, then loop. Common mistake: returning only the last point, or storing `iters` rows instead of `iters + 1`.
4. **`heavy_ball(grad, w0, step, momentum, iters)`** returns the trajectory. Keep two variables, `w` and `w_prev`, both starting as copies of `w0`. Common mistake: updating `w_prev` before you use it, which makes the momentum term zero.
5. **`nesterov(grad, w0, step, momentum, iters)`** returns the trajectory. Same bookkeeping as the heavy ball, plus the look-ahead point `y`. Common mistake: evaluating the gradient at `w` instead of `y`.

Use `w0.copy()` so you never change the caller's array.

## Cost and scaling

Each step costs one gradient evaluation. For the quadratic, that is one matrix-vector product with $A$, about $d^2$ operations. For least squares from data, the gradient $X^\top(Xw - y)$ costs about $2nd$. The number of steps is what differs.

| Method | Steps to gain a fixed number of digits | Extra memory |
|---|---|---|
| Gradient descent, step $1/L$ | about $\kappa$ | none |
| Heavy ball | about $\sqrt{\kappa}$ | one previous point |
| Nesterov | about $\sqrt{\kappa}$ | one previous point |

The trajectory array costs $(\text{iters} + 1) \times d$ numbers. Real training code keeps only the current point.

## Where this breaks

- **Step too big.** Above $2/L$ the loss grows every step and reaches infinity. If your loss goes up, halve the step.
- **Step too small.** The method crawls. Try a larger step, or estimate $L$ from the largest eigenvalue of $A$.
- **Badly scaled features.** They make $\kappa$ large, which makes every method slow. Standardise your features first.
- **The loss is not a bowl.** On functions with many valleys, the step counts above no longer hold, and you may stop in a poor valley. Neural networks live here.
- **Momentum overshoots.** Too much momentum makes the path spiral past the minimum. Lower $\beta$ if the loss oscillates.

## The exercise

You build five functions: `quadratic_loss_and_grad`, `exact_line_search_step`, `gradient_descent`, `heavy_ball`, and `nesterov`. Each method returns its whole path, so the checks can inspect every step.

The checks confirm that your gradient matches a finite difference and that your line-search step is the closed form above. They run gradient descent with a step of $2.1/L$ and expect it to blow up. With step $1/L$, the error along the flat direction must shrink by exactly $1 - \mu/L$ per step. The heavy ball and Nesterov must follow their update rules exactly. Finally, with the best settings, the heavy ball must reach a target accuracy roughly $\sqrt{\kappa}$ times faster than plain gradient descent.
