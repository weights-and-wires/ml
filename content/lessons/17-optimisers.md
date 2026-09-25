---
title: Optimisers for deep networks
slug: 17-optimisers
part: "Part 3 — Neural networks"
summary: Build SGD, momentum, Adam, and AdamW behind one interface, plus the warmup-and-cosine learning-rate schedule.
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

In lesson 16 you learned how to compute the gradient of a network's loss. The
gradient tells you which way is downhill. It doesn't tell you how big a step to
take, or how to cope when the gradient is noisy or points in an awkward
direction. That is the job of an optimiser.

Plain gradient descent works on small, tidy problems. Deep networks are not
tidy. You compute each gradient from a small random batch, so it is noisy. The
loss surface is often steep in some directions and flat in others. Both
problems make a single fixed learning rate a poor fit.

In this lesson you build four optimisers that share one interface: SGD,
momentum, Adam, and AdamW. You also build a learning-rate schedule that ramps
up and then decays. These are the tools that almost every deep learning
project uses.

## What you already know

- Gradient descent moves the weights a small step against the gradient.
- The learning rate controls how big that step is.
- A gradient computed on a random batch is a noisy estimate of the full gradient.
- Backpropagation gives you the gradient of the loss for every weight.
- L2 regularisation adds a penalty that pulls weights toward zero.

## One interface for every optimiser

Every optimiser in this lesson is a function with the same shape. It takes the
current weights, the gradient, and a small dictionary called `state`. It
returns the new weights and the new state.

```python
new_w, new_state = step_fn(w, grad, state, lr=0.01)
```

The `state` dictionary is the optimiser's memory. Plain SGD remembers nothing,
so its state stays empty. Momentum remembers a running velocity. Adam remembers
two running averages and a step counter. On the first call, `state` is `{}`,
and each optimiser treats a missing entry as zeros.

This shared shape means one training loop can drive any optimiser. You write
that loop as `run_optimiser`.

## Plain SGD

Stochastic gradient descent (SGD) is gradient descent where each gradient comes
from a random batch rather than the whole dataset. The update is the same one
you already know:

$$
w_{\text{new}} = w - \eta \, g
$$

Read it aloud: the new weights equal the old weights minus the learning rate
$\eta$ (the Greek letter eta) times the gradient $g$. Every weight moves
against its own gradient, scaled by the same $\eta$.

```python
new_w = w - lr * grad
```

For example, with $w = 1.0$, $g = 0.1$, and $\eta = 0.1$, the new weight is
$1.0 - 0.1 \times 0.1 = 0.99$.

SGD has two weaknesses. First, noise: each batch gradient points in a slightly
different direction, so the weights jitter. Second, uneven curvature. Picture
a long, narrow valley. Across the valley the walls are steep, and along it the
floor slopes gently. A learning rate small enough to avoid bouncing between the
steep walls is far too small to make progress along the floor.

## Momentum

Momentum fixes some of that. Think of a ball rolling downhill. It doesn't stop
and restart at every bump. It builds up speed in the direction it keeps
going, and bumps from side to side cancel out.

Momentum keeps a velocity $v$, a running total of past gradients where older
ones fade away:

$$
v_{\text{new}} = \beta \, v + g, \qquad w_{\text{new}} = w - \eta \, v_{\text{new}}
$$

Read it aloud: the new velocity is $\beta$ (beta, a number such as 0.9) times
the old velocity, plus the current gradient. Then the weights move by the
learning rate times the new velocity. A $\beta$ of 0.9 means each past
gradient keeps 90% of its influence at every step.

```python
v = beta * state.get("v", np.zeros_like(w)) + grad
new_w = w - lr * v
```

This version has no $(1 - \beta)$ factor in front of $g$. That matters for the
next result.

### Worked example: velocity builds up

Suppose the gradient is a constant $g = 2$ and $\beta = 0.9$. Start with
$v = 0$:

| Step | Calculation | Velocity $v$ |
|---|---|---|
| 1 | $0.9 \times 0 + 2$ | 2.0 |
| 2 | $0.9 \times 2.0 + 2$ | 3.8 |
| 3 | $0.9 \times 3.8 + 2$ | 5.42 |
| many | settles where $v = 0.9v + 2$ | 20.0 |

The velocity settles at $g / (1 - \beta) = 2 / 0.1 = 20$. So once momentum has
warmed up on a steady slope, each step has size

$$
\frac{\eta}{1 - \beta} \, g
$$

Read it aloud: momentum behaves like plain SGD with a learning rate
$1 / (1 - \beta)$ times larger. With $\beta = 0.9$, that is ten times larger.
This is why people lower $\eta$ when they turn momentum on.

On a narrow valley, the gradients across the valley flip sign each step and
mostly cancel inside $v$. The gradients along the floor all point the same
way and add up. The ball speeds up where it should.

Momentum does not remove the noise from stochastic gradients. It averages the
noise over roughly $1 / (1 - \beta)$ steps, but the effective step also grows by
the same factor. To settle at the bottom, you still need to shrink the learning
rate over time. That is what a schedule does, later in this lesson.

## Adam

Momentum uses one learning rate for every weight. Adam gives each weight its
own step size. The idea: if a weight's gradient is usually large, take smaller
steps for it. If it is usually small, take larger steps. Adam measures "usually"
with running averages.

Adam keeps two running averages for every weight:

- $m$, the average gradient. This is like momentum's velocity.
- $v$, the average squared gradient. Its square root measures how large the
  gradient typically is.

An exponential moving average (EMA) is a running average where older values
fade by a factor each step. Adam updates both averages like this:

$$
m_{\text{new}} = \beta_1 m + (1 - \beta_1)\, g, \qquad
v_{\text{new}} = \beta_2 v + (1 - \beta_2)\, g^2
$$

Read it aloud: the new $m$ is 90% old $m$ plus 10% of today's gradient
(with $\beta_1 = 0.9$). The new $v$ is 99.9% old $v$ plus 0.1% of today's
squared gradient (with $\beta_2 = 0.999$). Here $g^2$ squares each entry of
the gradient separately. Unlike momentum, these averages do have the
$(1 - \beta)$ factor, so they are true averages.

```python
m = beta1 * m_prev + (1 - beta1) * grad
v = beta2 * v_prev + (1 - beta2) * grad**2
```

### Bias correction

Both averages start at zero. That drags them toward zero for the first few
steps. After one step with gradient $g$, the average $m$ equals $0.1 g$, not
$g$. After $t$ steps with a constant gradient, $m = (1 - \beta_1^t)\, g$.

The fix is to divide that shrinkage back out. Let $t$ be the number of steps
taken so far, counting this one:

$$
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}
$$

Read it aloud: $\hat{m}$ ("m hat") is $m$ divided by one minus $\beta_1$ to
the power $t$. Early on, $1 - \beta_1^t$ is small, so the division scales $m$
back up. After many steps, $\beta_1^t$ is close to zero, and the correction
fades to nothing.

```python
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
```

### The Adam update

Adam then moves each weight by its corrected average gradient, divided by
the square root of its corrected average squared gradient:

$$
w_{\text{new}} = w - \eta \, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Read it aloud: the step is the learning rate times "typical direction" over
"typical size". The small number $\epsilon$ (epsilon, such as $10^{-8}$)
prevents division by zero.

```python
new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
```

### Worked example: the first Adam step

Take one weight with $w = 0$, gradient $g = 3$, $\eta = 0.1$,
$\beta_1 = 0.9$, $\beta_2 = 0.999$, and $t = 1$:

1. $m = 0.9 \times 0 + 0.1 \times 3 = 0.3$.
2. $v = 0.999 \times 0 + 0.001 \times 9 = 0.009$.
3. $\hat{m} = 0.3 / (1 - 0.9) = 3$.
4. $\hat{v} = 0.009 / (1 - 0.999) = 9$.
5. The step is $0.1 \times 3 / (\sqrt{9} + 10^{-8}) \approx 0.1$.
6. The new weight is $w \approx -0.1$.

If the gradient had been 3000 instead of 3, the step would still be about 0.1.
Adam's step size depends on the learning rate, not on the scale of the
gradient. SGD's step would have been a thousand times larger. The checks test
this directly.

## AdamW: weight decay done right

Weight decay shrinks every weight a little on each step, which keeps weights
small and helps the model generalise. With plain SGD, adding an L2 penalty
$\tfrac{\lambda}{2}\|w\|^2$ to the loss gives exactly the same result as weight
decay, because its gradient $\lambda w$ gets scaled by the same $\eta$.

Adam breaks that. If you add $\lambda w$ to the gradient, Adam divides it by
$\sqrt{\hat{v}}$ like everything else. Weights with large gradients get almost
no decay, and weights with small gradients get a lot. The decay no longer does
what you asked for.

AdamW keeps the two apart. It runs a normal Adam step using the raw gradient,
then subtracts the decay separately:

$$
w_{\text{new}} = w_{\text{Adam}} - \eta \, \lambda \, w
$$

Read it aloud: take the weights Adam would produce, then subtract the learning
rate times the weight decay rate $\lambda$ (lambda) times the old weights, the
ones from before this step. The decay never passes through Adam's averages.

```python
new_w, new_state = adam_step(w, grad, state, lr, beta1, beta2, eps)
new_w = new_w - lr * weight_decay * w
```

With `weight_decay=0`, AdamW gives exactly the same result as Adam.

## Learning-rate schedules

A fixed learning rate is a compromise. Early in training you want to move fast.
Late in training you want small steps so the noise doesn't keep kicking the
weights away from a good spot. A schedule changes the learning rate over time.

The most common schedule has two parts:

1. **Warmup.** For the first `warmup` steps, the rate rises in a straight line
   from 0 to a peak value. At the start, Adam's averages are based on only a
   few gradients, and the weights are random. Big steps then can throw the
   model somewhere it never recovers from. Warmup starts gently.
2. **Cosine decay.** From `warmup` to `total`, the rate follows half a cosine
   wave from the peak down to 0. It drops slowly at first, faster in the
   middle, and slowly again at the end.

Here is the formula for the decay part. Let $p$ be the fraction of the decay
phase that is done:

$$
p = \frac{\text{step} - \text{warmup}}{\text{total} - \text{warmup}}, \qquad
\eta(\text{step}) = \eta_{\text{peak}} \cdot \tfrac{1}{2}\big(1 + \cos(\pi p)\big)
$$

Read it aloud: $p$ goes from 0 at the end of warmup to 1 at the final step.
At $p = 0$, $\cos(0) = 1$, so the rate is the full peak. At $p = 1$,
$\cos(\pi) = -1$, so the rate is 0. At $p = 0.5$, the rate is half the peak.
During warmup, the rate is $\eta_{\text{peak}} \cdot \text{step} / \text{warmup}$.

```python
p = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
lr = peak * 0.5 * (1.0 + np.cos(np.pi * p))
```

### Worked example: a schedule

Take `warmup=100`, `total=1000`, and `peak=0.01`:

| Step | Phase | Calculation | Learning rate |
|---|---|---|---|
| 0 | warmup | $0.01 \times 0/100$ | 0.0 |
| 50 | warmup | $0.01 \times 50/100$ | 0.005 |
| 100 | decay, $p = 0$ | $0.01 \times 0.5 \times 2$ | 0.01 |
| 325 | decay, $p = 0.25$ | $0.01 \times 0.5 \times (1 + 0.7071)$ | 0.0085355 |
| 550 | decay, $p = 0.5$ | $0.01 \times 0.5 \times 1$ | 0.005 |
| 1000 | decay, $p = 1$ | $0.01 \times 0.5 \times 0$ | 0.0 |

If `warmup` is greater than or equal to `total`, there is no decay phase, and
dividing by `total - warmup` would fail. In that case, return `peak`.

## Build it step by step

The checks test the functions in this order. Each step builds on the one
before.

1. **`sgd_step(w, grad, state, lr)`.** Takes arrays `w` and `grad` of the same
   shape. Returns `(new_w, state)`. One line of arithmetic. The common mistake
   is forgetting to return a tuple.
2. **`momentum_step(w, grad, state, lr, beta)`.** Read the old velocity with
   `state.get("v", np.zeros_like(w))`. Compute the new velocity, move `w`, and
   return `{"v": v}` as the new state. The common mistake is adding a
   `(1 - beta)` factor. The checks expect `v = beta * v + grad`, which is what
   gives the $\eta / (1 - \beta)$ step size.
3. **`adam_step(w, grad, state, lr, beta1, beta2, eps)`.** Read `t` with
   `state.get("t", 0) + 1`, and read `m` and `v` with zero defaults. Update both
   averages, bias-correct them, and take the step. Return
   `{"m": m, "v": v, "t": t}`. Store the raw `m` and `v`, not the corrected
   ones. The common mistake is storing $\hat{m}$, which compounds the
   correction on every call.
4. **`adamw_step(..., weight_decay)`.** Call `adam_step`, then subtract
   `lr * weight_decay * w` using the `w` that came in. The common mistake is
   adding `weight_decay * w` to `grad`, which is plain L2, not AdamW.
5. **`cosine_schedule(step, warmup, total, peak)`.** Three branches: guard
   `warmup >= total`, the linear warmup, and the cosine. Use `np.cos` and
   `np.pi`. The common mistake is measuring $p$ from step 0 instead of from
   the end of warmup.
6. **`run_optimiser(grad_fn, init, step_fn, steps)`.** Start with
   `w = np.array(init, dtype=float)`, an empty state, and `history = [w.copy()]`.
   Loop `steps` times: compute the gradient, call `step_fn(w, grad, state)`,
   and append a copy. Return `(w, history)`. The history has `steps + 1`
   entries. The common mistake is appending `w` without `.copy()`.

## Cost and scaling

Every optimiser here works on each weight separately, so the time per step
grows in a straight line with the number of weights. The difference is memory:
how many extra arrays of the same size as `w` each one keeps.

| Optimiser | Extra arrays stored | Memory for a 1-billion-weight model (32-bit) |
|---|---|---|
| SGD | 0 | 0 GB |
| Momentum | 1 ($v$) | 4 GB |
| Adam | 2 ($m$, $v$) | 8 GB |
| AdamW | 2 ($m$, $v$) | 8 GB |

For large models, Adam's two extra arrays are often the biggest memory cost
after the weights themselves. The time for an optimiser step is small next to
the forward and backward pass.

## Where this breaks

- **Learning rate too high.** The loss blows up or turns into `nan`. Lower the
  rate by a factor of three to ten, or add warmup.
- **Momentum with an unchanged learning rate.** Turning on momentum with
  $\beta = 0.9$ makes steps about ten times larger. Lower $\eta$ to compensate.
- **Tiny $\hat{v}$.** When a weight's gradients are almost always zero, Adam's
  denominator is tiny and a rare nonzero gradient causes a huge step.
  A larger $\epsilon$ tames this.
- **L2 inside Adam.** Adding the penalty to the gradient gives uneven decay.
  Use AdamW's separate decay instead.
- **No decay at the end.** With a fixed rate, noisy gradients keep the weights
  jittering around the minimum. A schedule that decays to zero lets them
  settle.
- **Adam generalising worse than SGD.** On some vision tasks, well-tuned SGD
  with momentum reaches slightly better test accuracy. People often try both.

## The exercise

You implement `sgd_step`, `momentum_step`, `adam_step`, `adamw_step`,
`cosine_schedule`, and `run_optimiser`. Every step function returns
`(new_w, new_state)`, so `run_optimiser` can drive any of them through
`functools.partial`.

The checks verify SGD's update, momentum's velocity rule and its $\eta / (1 -
\beta)$ steady-state step, Adam's raw and bias-corrected averages, the exact
difference between Adam and AdamW, Adam's near-indifference to gradient scale,
and the schedule's values at key steps. The last check runs all four
optimisers on a narrow valley, a quadratic 100 times steeper in one direction
than the other, and expects each to reach a low loss.
