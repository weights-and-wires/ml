---
title: Optimisers for deep networks
slug: 17-optimisers
part: "Part 3 — Neural networks"
summary: "Build SGD, momentum, Adam, and AdamW behind one interface, plus a warmup and cosine learning-rate schedule."
minutes: 60
gpu: false
objectives:
  - Build SGD and momentum update steps, and compute momentum's effective step size.
  - Build Adam with bias correction, and explain why the correction is needed early in training.
  - Explain how AdamW's weight decay differs from adding an L2 term to the gradient.
  - Compute a learning rate from a linear warmup followed by cosine decay.
prereqs:
  - 03-gradient-descent
  - 16-backpropagation
exercise: 17-optimisers
---

# Optimisers for deep networks

You can already point downhill. The gradient is the list of partial derivatives: how much the loss changes when you nudge one weight. It does not choose a distance, and a batch gradient jitters from step to step.

One step size fits a narrow valley badly. Steep walls bounce, and a gentle floor barely moves.

You build SGD, momentum, Adam, and AdamW, plus a schedule for the step size.

## What you already know

You already have the following:

- Gradient descent subtracts a step along the gradient.
- The learning rate is that step size. One number scales every weight.
- A gradient from a random batch stands in for the full-data gradient, with noise.
- Backpropagation returns that gradient for every weight.
- L2 regularisation adds a penalty that pulls weights toward zero.

## Subtract a scaled gradient

Take a weight of 1.0, a gradient of 0.1, and a step size of 0.1. Subtract 0.1 times 0.1. The weight becomes 0.99.

That step is **stochastic gradient descent** (SGD) when the gradient comes from a random batch. New weights equal old weights minus the learning rate times the gradient:

$$
w_{\text{new}} = w - \eta \, g
$$

Eta, $\eta$, is the learning rate. The symbol $g$ is the gradient. The same numbers give 0.99:

```python
new_w = w - lr * grad
```

The usual mistake is to return `new_w` without `state`.

Return `(new_w, new_state)`. The dict `state` persists across steps. SGD passes it through, and a missing key means zeros. `run_optimiser` calls any of the four.

> [!KEY]
> SGD uses one step size for every weight, so a steep direction and a flat direction fight over that number.

## Let speed build on a slope

Picture a ball on a hillside. Speed builds where the slope keeps its sign. Sideways bumps cancel.

Keep 90 percent of the old speed, with the gradient stuck at 2, starting from speed 0. The first update is 0 plus 2, which is 2.

That speed is **momentum**, the heavy-ball method. New velocity is $\beta$ times the old velocity, plus the gradient. The weights then step by the learning rate times that velocity:

$$
v_{\text{new}} = \beta \, v + g, \qquad w_{\text{new}} = w - \eta \, v_{\text{new}}
$$

There is no $(1 - \beta)$ on $g$, so $v$ is a fading sum, not an average.

The same gradient continues as follows:

| Step | Calculation | Velocity |
|---|---|---|
| 1 | $0.9 \times 0 + 2$ | 2.0 |
| 2 | $0.9 \times 2.0 + 2$ | 3.8 |
| 3 | $0.9 \times 3.8 + 2$ | 5.42 |
| many | $v = 0.9 v + 2$ | 20.0 |

It settles at $g / (1 - \beta) = 20$. On a steady slope the step is

$$
\frac{\eta}{1 - \beta} \, g
$$

With $\beta = 0.9$ that factor is 10. Sign flips cancel in $v$, and a steady slope adds:

```python
v = beta * state.get("v", np.zeros_like(w)) + grad
new_w = w - lr * v
```

The usual mistake is to multiply `grad` by `(1 - beta)`.

> [!KEY]
> Momentum adds gradients that agree and cancels gradients that flip. A steady slope then steps $1 / (1 - \beta)$ times harder than SGD.

## Scale each weight by its own gradient

One weight has gradient 3. Another has gradient 3000. One step size crawls or leaps.

Start at 0, with gradient 3. The average gradient becomes $0.1 \times 3 = 0.3$. The average square becomes $0.001 \times 9 = 0.009$.

Those memories are **Adam**. An exponential moving average fades old values by a fixed factor. $m$ averages the gradient. $v$ averages the squared gradient, entry by entry:

$$
m_{\text{new}} = \beta_1 m + (1 - \beta_1)\, g, \qquad
v_{\text{new}} = \beta_2 v + (1 - \beta_2)\, g^2
$$

The first step is $m = 0.3$ and $v = 0.009$:

```python
m = beta1 * m_prev + (1.0 - beta1) * grad
v = beta2 * v_prev + (1.0 - beta2) * grad**2
```

The usual mistake is to collapse `grad**2` with `np.sum`, so one $v$ serves every weight.

Both averages start at 0, so the first $m$ is $0.1 g$, not $g$. After $t$ constant steps, $m = (1 - \beta_1^t)\, g$. Dividing that factor back out is **bias correction**, with $t$ counted from 1:

$$
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}
$$

$\hat{m}$ is $m$ divided by $1 - \beta_1^t$. Early on the denominator is small, so the division lifts the average. On this first step, $\hat{m} = 3$ and $\hat{v} = 9$:

```python
t = state.get("t", 0) + 1
m_hat = m / (1.0 - beta1**t)
v_hat = v / (1.0 - beta2**t)
```

> [!WARNING]
> Start $t$ at 1, not 0. Otherwise $1 - \beta_1^0$ is zero and the first step divides by zero.

Adam then divides the corrected direction by the corrected typical size. Epsilon, $\epsilon$, such as $10^{-8}$, keeps the denominator off zero:

$$
w_{\text{new}} = w - \eta \, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

The fraction is a typical direction over a typical size, and $\eta$ sets the length. With $\eta = 0.1$ and $w = 0$, the step is about 0.1, so the new weight is about $-0.1$. Gradient 3000 still steps about 0.1, because $\sqrt{\hat{v}}$ grows with $\hat{m}$. SGD would grow by 1,000. The update is:

```python
new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
```

The usual mistake is to store $\hat{m}$ and $\hat{v}$, so the next call corrects an average twice. Store raw $m$, raw $v$, and $t$.

> [!KEY]
> Adam divides each average gradient by its typical size, so a step stays about $\eta$ long at any gradient scale.

## Decay the weights after the Adam step

Weight decay shrinks every weight by the same fraction. For plain SGD, the penalty $\tfrac{\lambda}{2}\|w\|^2$ has gradient $\lambda w$, and times $\eta$ it matches that shrink.

Adam does not. Add $\lambda w$ to the gradient and the division by $\sqrt{\hat{v}}$ shrinks a quiet weight hard and a noisy weight barely at all.

Take $w = 1$, $\eta = 0.1$, and decay 0.01. The decay piece is $0.001$. Subtract it after Adam, using the old $w$.

That split is **AdamW**. Moments see only the raw gradient:

$$
w_{\text{new}} = w_{\text{Adam}} - \eta \, \lambda \, w
$$

Lambda, $\lambda$, is the decay rate. Every weight loses the fraction $\eta \lambda$. The same numbers subtract 0.001. Decay 0 makes AdamW match Adam:

```python
new_w, new_state = adam_step(w, grad, state, lr, beta1, beta2, eps)
new_w = new_w - lr * weight_decay * w
```

The usual mistake is to add `weight_decay * w` onto `grad` before the moments.

> [!KEY]
> AdamW subtracts $\eta \lambda w$ after the Adam step, so every weight shrinks by the same fraction.

## Warm the step size, then let it fall

Early Adam averages rest on one or two batches, and the weights are random. A full step can make the loss `nan`. Late on, a fixed step lets noise keep kicking the weights.

Use 100 warmup steps, 1,000 steps in all, and peak rate 0.01. Step 0 gives 0. Step 50 gives $0.01 \times 50 / 100 = 0.005$. Step 100 hits the peak.

The rise is **linear warmup**. The fall is **cosine decay**. The fraction $p$ runs from 0 at the end of warmup to 1 at the last step:

$$
p = \frac{\text{step} - \text{warmup}}{\text{total} - \text{warmup}}, \qquad
\eta(\text{step}) = \eta_{\text{peak}} \cdot \tfrac{1}{2}\big(1 + \cos(\pi p)\big)
$$

Warmup multiplies the peak by $\text{step} / \text{warmup}$. Decay takes half of one plus a cosine, from the peak at $p = 0$ down to 0 at $p = 1$.

The same schedule hits these values:

| Step | Phase | Calculation | Learning rate |
|---|---|---|---|
| 0 | warmup | $0.01 \times 0/100$ | 0.0 |
| 50 | warmup | $0.01 \times 50/100$ | 0.005 |
| 100 | decay, $p = 0$ | $0.01 \times 0.5 \times 2$ | 0.01 |
| 325 | decay, $p = 0.25$ | $0.01 \times 0.5 \times (1 + 0.7071)$ | 0.0085355 |
| 550 | decay, $p = 0.5$ | $0.01 \times 0.5 \times 1$ | 0.005 |
| 1000 | decay, $p = 1$ | $0.01 \times 0.5 \times 0$ | 0.0 |

Clamp the fraction, then apply the cosine:

```python
p = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
lr = peak * 0.5 * (1.0 + np.cos(np.pi * p))
```

The usual mistake is to measure $p$ from step 0. If `warmup >= total`, return `peak` and do not divide.

> [!TIP]
> Check step 0, the end of warmup, halfway through decay, and the last step. The rates are 0, the peak, half the peak, and 0.

> [!RECAP]
> Hold these five points before you write code:
>
> - Each update takes `(w, grad, state)` and returns `(new_w, new_state)`. A missing entry means zeros.
> - Momentum uses `v = beta * v + grad`, with no $(1 - \beta)$ on the gradient.
> - Adam stores raw $m$, raw $v$, and $t$ from 1. Correct $m$ and $v$ only while building the step.
> - AdamW runs Adam, then subtracts `lr * weight_decay * w` with the old `w`.
> - Warmup is a straight line up. Cosine decay is half a cosine from the peak down to 0.

## Build it step by step

The checks call the functions in this order:

1. **`sgd_step(w, grad, state, lr)`.** Arrays share a shape. Return `(new_w, state)`. The usual mistake is dropping the tuple. The body is `w - lr * grad`.
2. **`momentum_step(w, grad, state, lr, beta)`.** Default `"v"` with `np.zeros_like(w)`. Set `v = beta * v + grad`, then subtract `lr * v`. Return `{"v": v}`. The usual mistake is a `(1 - beta)` factor.
3. **`adam_step(w, grad, state, lr, beta1, beta2, eps)`.** Use `t = state.get("t", 0) + 1`. Step with the corrected moments. Store raw `m`, raw `v`, and `t`. The usual mistake is storing the corrected copies.
4. **`adamw_step(w, grad, state, lr, beta1, beta2, eps, weight_decay)`.** Call `adam_step` on the raw gradient, then subtract `lr * weight_decay * w` with the incoming `w`. The usual mistake is folding decay into `grad`.
5. **`cosine_schedule(step, warmup, total, peak)`.** Return a float. If `warmup >= total`, return `peak`. Then the straight line, then the cosine. The usual mistake is starting $p$ at step 0.
6. **`run_optimiser(grad_fn, init, step_fn, steps)`.** Start history as a copy of float `init` and append a copy after each call, for `steps + 1` entries. The usual mistake is appending `w` without `.copy()`.

## Cost and scaling

Time per step grows in a straight line with the number of weights. Memory is the extra arrays the size of `w`.

Extra storage for one billion 32-bit weights is as follows:

| Optimiser | Extra arrays | Extra memory |
|---|---|---|
| SGD | none | 0 GB |
| Momentum | one velocity | 4 GB |
| Adam | $m$ and $v$ | 8 GB |
| AdamW | $m$ and $v$ | 8 GB |

An optimiser step is small next to the forward and backward passes.

## Where this breaks

Each of these failures has a direct change:

- Too high a learning rate sends the loss to infinity or to `nan`. Cut it by three to ten, or add warmup.
- Momentum at $\beta = 0.9$, with the old learning rate, makes steps about ten times larger. Lower $\eta$.
- A tiny $\hat{v}$ turns a rare gradient into a huge Adam step. Raise $\epsilon$.
- An L2 term inside Adam decays weights unevenly. Use AdamW.
- A fixed rate at the end lets noise keep jittering the weights. Decay the schedule toward zero.
- On some vision tasks, SGD with momentum reaches a slightly lower test error than Adam. Train both.

## The exercise

You implement `sgd_step`, `momentum_step`, `adam_step`, `adamw_step`, `cosine_schedule`, and `run_optimiser`. Each step returns `(new_w, new_state)`, so `run_optimiser` drives any of them with `functools.partial`.

The checks cover the SGD step, momentum's velocity, and the factor $\eta / (1 - \beta)$. They cover Adam's moments and the AdamW gap of `lr * weight_decay * w`. They check scale invariance and the schedule at 0, the peak, half the peak, and 0. All four must reach a low loss on a quadratic 100 times steeper in one coordinate.
