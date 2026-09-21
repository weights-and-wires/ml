---
title: Optimisers for deep networks
slug: 17-optimisers
part: "Part 3 — Neural networks"
summary: How stochastic gradients change the update rule, and the momentum, Adam, AdamW, and schedule tricks that make training stable.
minutes: 45
gpu: false
objectives:
  - Explain why a stochastic gradient forces a decaying or noise-tolerant step size, and why momentum cannot fix that on its own.
  - Implement Adam's bias correction and explain what problem it solves.
  - Explain why weight decay and L2 regularisation diverge once the gradient is preconditioned, and what AdamW changes to reconcile them.
  - Build a cosine learning-rate schedule with linear warmup, and explain what the warmup phase protects against.
prereqs:
  - 03-gradient-descent
  - 16-backpropagation
exercise: 17-optimisers
---

# Optimisers for deep networks

Lesson 3 treated gradient descent as an iteration on the exact gradient of a
fixed, convex objective. Training a deep network breaks both assumptions: you
can only afford the gradient of a random minibatch, not the full dataset, and
a nonlinear network's loss surface is not convex. A minibatch gradient's
expectation is the true gradient, but its variance never goes away, no matter
how far training runs.

This lesson builds the optimiser stack deep learning runs on from that one
fact. Gradient noise forces a smaller or smarter step than plain gradient
descent uses, which is why momentum alone cannot recover the convergence
rate lesson 3 derived for the noiseless case. Rescaling the step per
parameter — the idea behind Adam — answers loss surfaces whose curvature
differs by orders of magnitude across directions, a problem the next lesson
covers in more depth. The lesson closes with two choices every training run
makes before it starts: how weight decay interacts with an adaptive update,
and how the learning rate should move over training.

## What you already know

- Backpropagation computes an exact gradient of the loss on a chosen batch, at
  a cost proportional to one forward pass.
- An unbiased estimator's expectation equals the quantity it estimates, and
  averaging `n` independent copies divides its variance by `n`.
- Gradient descent converges under smoothness and convexity, and momentum
  speeds up that convergence when the gradient is exact.
- A surface's condition number is the ratio of its largest to its smallest
  curvature; a large condition number makes a single global step size hard to
  choose well.
- An exponential moving average $x_k = \beta x_{k-1} + (1-\beta)y_k$ weights
  the past geometrically, with an effective window of about $1/(1-\beta)$
  steps.

## Stochastic gradients need a different step

Write the empirical risk as $L(w) = \frac{1}{n}\sum_{i=1}^n \ell_i(w)$ and a
minibatch gradient over a random batch $\mathcal B$ of size $B$ as $g(w) =
\frac{1}{B}\sum_{i\in\mathcal B}\nabla\ell_i(w)$. This gradient is unbiased,
$\mathbb E[g(w)] = \nabla L(w)$, but its variance falls only as $1/B$:

$$
\operatorname{Var}(g(w)) = \frac{\sigma^2}{B} .
$$

Unbiasedness alone does not make gradient descent converge to a point. At the
true minimum $\nabla L(w) = 0$, but $g(w)$ is not, so a fixed step keeps the
iterate jittering forever. The step has to shrink over time for training to
settle — a schedule like $\eta_k = \eta_0/k$ shrinks fast enough to control
that noise while still summing to infinity, so an initial error can always be
corrected given enough steps.

In the noiseless case, momentum turns gradient descent's $O(1/K)$ convergence
into $O(1/K^2)$. Once the gradient carries noise of variance $\sigma^2$, no
first-order method can localise the minimiser faster than $O(1/\sqrt K)$
using $K$ noisy gradients, because two minimisers close enough together are
indistinguishable given that much noisy data. Momentum still helps in
practice — it damps oscillation and speeds up early training — but it does
not change this asymptotic rate the way it does with an exact gradient.

## Minibatching, warmup, and batch size

Quartering the batch variance $\sigma^2/B$ takes sixteen times the data per
step, and a larger batch also costs more compute per step. A bigger batch
pays off only when a wider GPU or more parallel workers absorb that extra
work at no added wall-clock cost.

If you scale the batch size by a factor $k$, scale the learning rate by
roughly the same factor — the linear-scaling heuristic. It follows from
noticing that $k$ small steps at $(B, \eta)$ land close to one step at
$(kB, k\eta)$, provided the gradient barely changes across those $k$ steps.
That assumption fails at the start of training, when weights move fast and
consecutive gradients disagree. **Warmup** ramps the learning rate up from a
small value over the first few hundred steps, so training spends the unstable
early phase at a safer rate and only reaches the fully scaled rate once the
approximation above holds.

## Momentum as a moving average of gradients

Heavy-ball momentum replaces the raw gradient with a running accumulator:

$$
v_k = \beta v_{k-1} + g_k, \qquad w_{k+1} = w_k - \eta v_k ,
$$

starting from $v_0 = 0$. Under a gradient $g$ that stays constant, $v_k$
approaches $g/(1-\beta)$, so the eventual step per iteration is
$\eta v_k \approx \frac{\eta}{1-\beta}g$. With $\beta = 0.9$, momentum
eventually moves ten times faster than plain gradient descent at the same
$\eta$, along a direction the gradient keeps pointing toward. Along a
direction where the gradient's sign flips every step — exactly the
oscillation ill-conditioned problems produce — the accumulated terms cancel
instead of adding up, so the effective step there stays close to $\eta$.
Amplifying a persistent signal while damping an alternating one is the whole
mechanism.

## Adaptive per-parameter steps

**AdaGrad** and **RMSProp** rescale each parameter's step by its own recent
gradient magnitude, so a coordinate with large gradients gets a smaller step
and a coordinate with small gradients gets a larger one. AdaGrad accumulates
the sum of every squared gradient seen so far, which only grows, so its step
eventually shrinks to zero even on parameters that still need to move.
RMSProp uses an exponential moving average of the squared gradient instead
of a running sum, so the denominator can grow or shrink as the local
gradient scale changes.

## Adam

Adam keeps an exponential moving average of the gradient and one of its
square:

$$
m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k, \qquad
v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2 ,
$$

starting from $m_0 = v_0 = 0$. Starting at zero biases both estimates toward
zero in early training: under a gradient that has been roughly constant for
$k$ steps, $m_k \approx g(1-\beta_1^k)$, so at $k=1$ with $\beta_1=0.9$ the raw
estimate carries only a tenth of the true gradient. Dividing out the known
factor corrects this exactly under that assumption:

$$
\hat m_k = \frac{m_k}{1-\beta_1^k}, \qquad \hat v_k = \frac{v_k}{1-\beta_2^k} ,
$$

and the update is

$$
w_{k+1} = w_k - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} .
$$

Without this correction, early steps would be far smaller than intended,
throttling exactly the phase warmup is otherwise built to protect.

The ratio $\hat m_k/\sqrt{\hat v_k}$ barely changes if you multiply every
gradient by a constant $c$: multiplying $g$ by $c$ multiplies $m_k$ by $c$ and
$\sqrt{v_k}$ by $|c|$, so the ratio cancels the constant almost entirely. Plain
SGD has no such cancellation — its step $\eta g_k$ scales with $c$ directly.

## Weight decay and AdamW

L2 regularisation adds $\frac{\lambda}{2}\lVert w\rVert^2$ to the loss, which
adds $\lambda w$ to the gradient. Under plain SGD this produces the same
update as shrinking the weight directly, $w_{k+1} = w_k(1-\eta\lambda) - \eta
g_k$ — "L2 regularisation" and "weight decay" name the same thing.

They stop being the same once the gradient is preconditioned. Fold $\lambda
w$ into $g$ before computing Adam's moments, and the decay term gets divided
by $\sqrt{\hat v_k}+\epsilon$ along with the rest of the gradient: a parameter
with a large, noisy gradient has its decay shrunk, while a stationary
parameter keeps a decay close to $\eta\lambda w$. AdamW decouples the two
terms instead: it computes $m_k$ and $v_k$ from the raw gradient only, and
applies the shrinkage afterward, outside the adaptive step:

$$
w_{k+1} = w_k - \eta\left(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}
+ \lambda w_k\right) .
$$

The two updates agree exactly when $\lambda = 0$, and otherwise differ by the
term $\eta\lambda w_k$ that AdamW adds back unpreconditioned — a uniform
shrinkage rate that does not depend on a parameter's gradient history.

## Learning-rate schedules

A **linear warmup** ramps $\eta$ from a small value to a peak $\eta_{\max}$
over the first $w$ steps, $\eta_k = \eta_{\max}\cdot k/w$ for $k \le w$, for
the reasons already covered above: early steps carry unreliable signal, so
the schedule keeps the step small until the rest of training's assumptions
hold.

**Cosine decay** takes over after warmup and brings the rate smoothly down to
zero by a target step $T$:

$$
\eta_k = \eta_{\max}\cdot\frac{1}{2}\left(1+\cos\!\Big(\pi\,\frac{k-w}{T-w}\Big)\right) , \qquad w \le k \le T .
$$

At $k=w$ this equals $\eta_{\max}$, and at $k=T$ it equals $0$, with no jump
at any point in between. That smoothness is the main advantage over a step
schedule that divides $\eta$ by a constant at a few chosen epochs: cosine
decay needs no guess about which epochs matter.

## Cost and scaling

Every optimiser here costs $O(d)$ flops per step beyond the gradient itself,
for $d$ parameters. They differ in how much extra state they keep per
parameter.

| Optimiser | Extra state | Extra memory (float32) |
|---|---|---|
| SGD | none | $0$ |
| Momentum | one array, $v$ | $4d$ bytes |
| AdaGrad / RMSProp | one array | $4d$ bytes |
| Adam / AdamW | two arrays, $m$ and $v$ | $8d$ bytes |

Adam doubles the optimiser state a plain SGD run needs, which is why training
a model with $d$ parameters treats optimiser memory as a real cost, alongside
the $4d$ bytes the parameters themselves occupy, before gradients or
activations are counted.

## Where this breaks

**AdaGrad on long, non-convex runs.** Its accumulator only grows, so the step
eventually stops moving every coordinate. RMSProp and Adam replaced it for
deep networks.

**Adam without decoupled decay.** Folding L2 into the gradient under an
adaptive optimiser tends to generalise worse than SGD with momentum at a
matched decay rate. AdamW closes most of that gap.

**Skipping warmup after scaling the batch size or peak rate.** The
linear-scaling heuristic only holds locally, so runs that skip warmup after a
large batch or a high peak rate diverge often in the first few hundred steps.

**Treating $O(1/\sqrt K)$ as a wall-clock guarantee.** It bounds the number of
gradient evaluations needed, not seconds. Halving the batch size to take more
steps in the same wall-clock time does not beat the bound once noise, not
step count, limits progress.

## The exercise

You implement four optimiser steps sharing one interface —
`step_fn(w, grad, state, **hyperparameters) -> (new_w, new_state)` — so one
loop can drive any of them. Write `sgd_step`, `momentum_step`, and `adam_step`
with the bias correction derived above, then `adamw_step`, which reuses
`adam_step`'s moment update and adds the decoupled decay term separately.
Write `cosine_schedule(step, warmup, total, peak)`, returning the learning
rate at a given step, and `run_optimiser(grad_fn, init, step_fn, steps)`, a
loop that starts at `init`, calls `step_fn` `steps` times, and returns the
final iterate and the full trajectory.

The checks build from the smallest piece up: that `sgd_step` and
`momentum_step`'s velocity recursion match direct arithmetic, and that
momentum's steady-state step size under a constant gradient converges to
$\eta/(1-\beta)$ times that gradient. Another check confirms `adam_step`'s
raw and bias-corrected moments match the closed-form expectation derived
above at small `t`, which catches a missing or wrong correction term
directly. A further check compares `adam_step` and `adamw_step` on identical
inputs and requires their outputs to differ by exactly $\eta\lambda w$, and
another rescales the gradient by a large constant to confirm Adam's update
barely moves while SGD's scales with it directly. `cosine_schedule` is
checked at the end of warmup, at the total step count, and at the midpoint of
the decay. The last check runs `run_optimiser` with each of the four step
functions, using hyperparameters you choose, to reach a target loss on a
quadratic whose two curvatures differ by a factor of a hundred — the
anisotropic problem no single global learning rate handles well in every
direction at once.
