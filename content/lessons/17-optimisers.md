---
title: Optimisers for deep networks
slug: 17-optimisers
part: "Part 3 — Neural networks"
summary: Why stochastic gradients force a decaying step, how momentum, Adam, and AdamW follow from that, and the schedules that stabilise training.
minutes: 55
gpu: false
objectives:
  - Derive the O(1/sqrt(K)) convergence bound for stochastic gradient descent, and explain why momentum improves the deterministic rate but not this one.
  - Derive Adam's bias correction from the closed-form expectation of its exponential moving averages, and implement the resulting update.
  - Explain why weight decay and L2 regularisation diverge once the gradient is preconditioned, and what AdamW changes to reconcile them.
  - Build a cosine learning-rate schedule with linear warmup, and explain what the warmup phase protects against.
prereqs:
  - 03-gradient-descent
  - 16-backpropagation
exercise: 17-optimisers
---

# Optimisers for deep networks

Lesson 3 analysed gradient descent as an iteration on the exact gradient of a
fixed, convex objective, and lesson 16 showed how to compute that gradient for a
computation graph of any depth. Training a deep network violates both premises
at once: the objective is an empirical risk over a dataset too large to
differentiate exactly at every step, and the surface a nonlinear network
induces is not convex. What you can afford is the gradient of a random
minibatch, a quantity whose expectation is the true gradient but whose variance
does not vanish as training proceeds.

This lesson builds the optimiser stack deep learning runs on from that single
fact. A stochastic gradient's variance forces a decaying or noise-tolerant step
size, which is why plain stochastic gradient descent converges at a rate that
momentum, unlike in the noiseless case, cannot improve. Rescaling that step
per parameter — AdaGrad, then RMSProp, then Adam — answers loss surfaces whose
curvature differs by orders of magnitude across directions, a situation the
next lesson's discussion of initialisation will make precise. The lesson closes
with the two settings every training run fixes before it starts: how weight
decay interacts with a preconditioned update, and how the learning rate should
move over the course of training.

## What you already know

- Gradient descent's $O(1/k)$ rate under smoothness and convexity, and the
  $O(1/k^2)$ improvement heavy-ball and Nesterov momentum give in the
  noiseless, full-batch case.
- Backpropagation computes an exact gradient of the loss on a chosen batch, at
  a cost proportional to one forward pass.
- An unbiased estimator's expectation equals the quantity it estimates, and
  averaging $n$ independent copies divides its variance by $n$.
- A convex loss has a positive semidefinite Hessian everywhere, and a
  surface's condition number is the ratio of its extreme curvatures.
- The exponential moving average $x_k = \beta x_{k-1} + (1-\beta)y_k$ weights
  the past geometrically, with an effective window of about $1/(1-\beta)$
  steps.

## Stochastic gradients

Write the empirical risk as $L(w) = \frac{1}{n}\sum_{i=1}^n \ell_i(w)$ and a
minibatch gradient over a batch $\mathcal B$ of size $B$, drawn uniformly at
random, as $g(w) = \frac{1}{B}\sum_{i\in\mathcal B}\nabla\ell_i(w)$. Each term
$\nabla\ell_i(w)$ has the same distribution as a uniformly chosen index, so
$\mathbb E[g(w)] = \nabla L(w)$: the stochastic gradient is unbiased. Writing
$\nabla\ell_i(w) = \nabla L(w) + \xi_i$ with $\mathbb E[\xi_i]=0$ and
$\mathbb E\lVert\xi_i\rVert^2 \le \sigma^2$, independence across the batch gives

$$
\operatorname{Var}(g(w)) = \frac{\sigma^2}{B} .
$$

Unbiasedness is not enough on its own. At a stationary point $\nabla L(w)=0$
but $g(w)$ is not, so a fixed step size keeps the iterate moving by an amount
set by $\eta^2\sigma^2$ forever. The step has to shrink for the iterate to
settle, and the Robbins-Monro conditions state exactly how much:

$$
\sum_{k=1}^{\infty}\eta_k = \infty, \qquad \sum_{k=1}^{\infty}\eta_k^2 < \infty .
$$

The first condition says the steps, summed, must be able to travel arbitrarily
far, so no schedule that decays too fast can correct an initial error however
long you run it. The second says the accumulated noise the steps inject must
stay finite, so the resulting random walk has bounded variance and can settle
rather than wander indefinitely. The schedule $\eta_k = \eta_0/k$ satisfies
both — the harmonic series diverges, and $\sum 1/k^2 = \pi^2/6$ converges.

### Why momentum does not change the rate

Run SGD with a constant step $\eta$ for $K$ iterations and average the
iterates, $\bar w_K = \frac{1}{K}\sum_{k=0}^{K-1}w_k$. One step gives

$$
\lVert w_{k+1}-w^\star\rVert^2 = \lVert w_k-w^\star\rVert^2
- 2\eta\, g_k^\top(w_k-w^\star) + \eta^2\lVert g_k\rVert^2 .
$$

Taking expectation conditional on $w_k$, using $\mathbb E[g_k\mid w_k] =
\nabla L(w_k)$, convexity ($\nabla L(w_k)^\top(w_k-w^\star)\ge L(w_k)-L^\star$),
and a bound $\mathbb E\lVert g_k\rVert^2 \le M^2$ that folds together the
gradient's own magnitude and its variance $\sigma^2$,

$$
\mathbb E\lVert w_{k+1}-w^\star\rVert^2 \le \mathbb E\lVert w_k-w^\star\rVert^2
- 2\eta\big(\mathbb E[L(w_k)]-L^\star\big) + \eta^2 M^2 .
$$

Summing over $k=0,\dots,K-1$, telescoping the left side, and dividing by
$2\eta K$ gives, after applying Jensen's inequality to the average iterate,

$$
\mathbb E[L(\bar w_K)] - L^\star \;\le\;
\frac{\lVert w_0-w^\star\rVert^2}{2\eta K} + \frac{\eta M^2}{2} .
$$

Setting $\eta = D/(M\sqrt K)$ with $D=\lVert w_0-w^\star\rVert$ balances the two
terms, each becoming $DM/(2\sqrt K)$, for a total rate of $O(1/\sqrt K)$.

Split $M^2$ back into its two sources. The part coming from the gradient's own
magnitude is the deterministic term acceleration was built to shrink: Nesterov
momentum, run with no noise, turns the $O(1/K)$ dependence of plain gradient
descent into $O(1/K^2)$. The part coming from $\sigma^2$ is a noise floor: no
first-order method computing $K$ noisy gradients can localise the minimiser
faster than $D\sigma/\sqrt K$, because two minimisers separated by less than
the statistical error of $K$ averaged samples of variance $\sigma^2$ are not
distinguishable from the data available. Once $\sigma>0$, this floor dominates
the bound as $K\to\infty$ regardless of what the deterministic term does, so a
stochastic accelerated method still lands at $O(1/\sqrt K)$ overall: momentum
improves a term noise has already made irrelevant, the opposite of the
deterministic case, where there is no floor and acceleration changes the
exponent outright.

## Minibatching

Batch variance $\sigma^2/B$ falls off only as $1/B$: quartering it costs
sixteen times the samples per step. Compute per step also grows linearly in
$B$, so a larger batch is not free on a single device — it becomes attractive
once $B$ additional samples cost no extra wall-clock time, which is what a
wider GPU or more parallel workers provide.

Consider $k$ consecutive steps at batch size $B$ and learning rate $\eta$,
starting from a common point $w_0$ and approximating the gradient as roughly
constant across the window, $g_j(w_j)\approx g_j(w_0)$:

$$
w_k = w_0 - \eta\sum_{j=1}^k g_j(w_0)
= w_0 - (k\eta)\cdot\underbrace{\frac{1}{k}\sum_{j=1}^k g_j(w_0)}_{\text{average over } kB \text{ samples}} .
$$

Under that approximation, $k$ steps at $(B,\eta)$ land at the same place as one
step at $(kB, k\eta)$: scaling the batch size by $k$ calls for scaling the
learning rate by $k$ too — the linear-scaling heuristic.

The approximation $g_j(w_j)\approx g_j(w_0)$ is exactly what breaks at the
start of training, when the weights move fast relative to the window and
consecutive gradients disagree in both direction and magnitude. Applying the
fully scaled rate $k\eta$ from step one risks a divergence the "few large steps
equal one big step" argument no longer justifies. Warmup — ramping the rate
from a small value up to $k\eta$ over the first few hundred steps — spends the
non-linear early phase at a safer step and reaches the linear-scaled rate only
once the trajectory sits where the approximation holds.

## Momentum as a moving average of gradients

Heavy-ball momentum replaces the raw gradient with an accumulator,

$$
v_k = \beta v_{k-1} + g_k, \qquad w_{k+1} = w_k - \eta v_k ,
$$

with $v_0=0$. Unrolling gives $v_k = \sum_{i=1}^k \beta^{k-i}g_i$: a moving
average of the gradient history with geometric weights that do not sum to one.
Under a persistent gradient $g$, the weights sum to
$(1-\beta^k)/(1-\beta)\to 1/(1-\beta)$, so $v_k\to g/(1-\beta)$ and the step
per iteration approaches $\eta v_k = \frac{\eta}{1-\beta}g$: with $\beta=0.9$,
momentum eventually moves ten times faster than plain gradient descent at the
same $\eta$, in the direction of a signal that keeps pointing the same way.
Along a direction where the gradient's sign flips every step — the oscillating,
ill-conditioned directions momentum is meant to damp — the geometric sum
partly cancels instead of accumulating, so the effective step there stays
close to $\eta$. Amplifying the persistent signal while damping the
alternating one is the entire mechanism.

## Adaptive per-parameter steps

**AdaGrad** accumulates the sum of squared gradients coordinatewise,
$G_k = G_{k-1} + g_k^2$ with $G_0=0$, and updates
$w_{k+1} = w_k - \eta\, g_k/(\sqrt{G_k}+\epsilon)$. A coordinate that has seen
large gradients accumulates a large $G_k$ and gets a shrinking step, which is
what makes AdaGrad effective on sparse features. But $G_k$ sums nonnegative
terms, so it is monotone non-decreasing in every coordinate, and the step
$\eta/\sqrt{G_k}$ shrinks to zero even on a parameter that still needs to move
late in training — on a long run this stops the optimiser outright.

**RMSProp** replaces the running sum with an exponential moving average,

$$
s_k = \beta_2 s_{k-1} + (1-\beta_2)g_k^2 ,
$$

with the same update rule. Since $s_k$ remembers roughly the last
$1/(1-\beta_2)$ steps, the denominator can grow or shrink as the local gradient
scale changes rather than only growing, which removes AdaGrad's forced decay
while keeping the coordinatewise rescaling: a high-curvature direction divides
by a large $\sqrt{s_k}$, a flat one by a small one, so the effective step is
closer to curvature-independent than any single global $\eta$ can be.

## Adam

Adam keeps an exponential moving average of the gradient and one of its
square,

$$
m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k, \qquad
v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2 ,
$$

with $m_0=v_0=0$. Starting both accumulators at zero biases them toward zero
early in training. Unroll $m_k$ under a locally constant gradient $g$:

$$
m_k = (1-\beta_1)\sum_{i=1}^k \beta_1^{k-i} g
= g(1-\beta_1)\cdot\frac{1-\beta_1^k}{1-\beta_1}
= g\,(1-\beta_1^k) ,
$$

so $\mathbb E[m_k] = (1-\beta_1^k)\,\mathbb E[g]$ under a stationary gradient —
at $k=1$ with $\beta_1=0.9$, the raw estimate carries a tenth of its true
magnitude. Dividing out the known factor corrects it exactly under that
assumption:

$$
\hat m_k = \frac{m_k}{1-\beta_1^k}, \qquad \hat v_k = \frac{v_k}{1-\beta_2^k} ,
$$

and the update is

$$
w_{k+1} = w_k - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} .
$$

Without the correction, early steps would be smaller by a factor of roughly
$\sqrt{1-\beta_2}/(1-\beta_1)$ than intended, throttling exactly the phase a
warmup schedule is otherwise built to protect.

The ratio $\hat m_k/\sqrt{\hat v_k}$ is close to scale-invariant: multiplying
the loss, and hence every gradient, by a constant $c$ scales $m_k$ by $c$ and
$\sqrt{v_k}$ by $|c|$ (the moving average is linear, and the square root of a
quadratic average scales linearly), so the ratio is unchanged up to the
$\epsilon$ term. Plain SGD has no such cancellation; its step $\eta g_k$ scales
with $c$ directly.

## Weight decay and AdamW

L2 regularisation adds $\frac{\lambda}{2}\lVert w\rVert^2$ to the loss, adding
$\lambda w$ to the gradient. Under plain SGD this is identical to shrinking the
weight directly, $w_{k+1} = w_k(1-\eta\lambda) - \eta g_k$ — "L2
regularisation" and "weight decay" name the same update.

They stop naming the same update once the gradient is preconditioned. Fold
$\lambda w$ into $g$ before computing Adam's moments, and the decay term is
divided by $\sqrt{\hat v_k}+\epsilon$ along with the rest of the gradient: a
parameter with a large, noisy gradient accumulates a large $v_k$ and has its
decay shrunk by the same factor, while a nearly stationary parameter keeps a
decay close to $\eta\lambda w$. Preconditioning, whose purpose is to equalise
progress across coordinates, ends up equalising decay in the wrong direction.

AdamW decouples the two terms instead of folding one into the other: compute
$m_k$ and $v_k$ from the raw gradient only, and apply the shrinkage afterward,
outside the adaptive step,

$$
w_{k+1} = w_k - \eta\left(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}
+ \lambda w_k\right) .
$$

The coupled update (decay folded into $g$) and the decoupled one agree exactly
when $\lambda=0$, and for any other $\lambda$ they differ by precisely the term
$\eta\lambda w_k$ that AdamW adds back unpreconditioned. Because that term no
longer runs through $\sqrt{\hat v_k}$, weight decay under AdamW is a uniform
shrinkage rate independent of a parameter's gradient history, which is the
property the name promises.

## Learning-rate schedules

A **step schedule** holds $\eta$ fixed and divides it by a constant at chosen
epochs — cheap, and dependent on knowing good epochs in advance. A **linear
warmup** ramps $\eta$ from a small value to a peak $\eta_{\max}$ over the first
$w$ steps, $\eta_k = \eta_{\max}\cdot k/w$ for $k\le w$, for the reason both
minibatching and Adam surfaced above: early steps carry unreliable signal, so
the schedule keeps the nominal step small until the rest of the run's
assumptions hold.

**Cosine decay** replaces a schedule of milestones with one smooth curve.
After warmup, for $w \le k \le T$,

$$
\eta_k = \eta_{\max}\cdot\frac{1}{2}\left(1+\cos\!\Big(\pi\,\frac{k-w}{T-w}\Big)\right) .
$$

At $k=w$ the cosine equals $1$ and $\eta_k=\eta_{\max}$; at $k=T$ it equals
$-1$ and $\eta_k=0$: the schedule interpolates smoothly between the endpoints
warmup and the training budget fix, with no discontinuity at any epoch
boundary.

## Cost and scaling

Every optimiser here costs $O(d)$ flops per step beyond computing the gradient
itself; they differ in how much state they keep per parameter.

| Optimiser | Extra state per parameter | Extra memory (float32) |
|---|---|---|
| SGD | none | $0$ |
| Momentum | one array, $v$ | $4d$ bytes |
| AdaGrad / RMSProp | one array, $G$ or $s$ | $4d$ bytes |
| Adam / AdamW | two arrays, $m$ and $v$ | $8d$ bytes |

Adam doubles the optimiser state relative to plain SGD, which is why training
a $d$-parameter model tracks optimiser memory as a first-class cost alongside
the $4d$ bytes the parameters themselves occupy, before gradients or
activations are counted.

## Where this breaks

**AdaGrad on long, non-convex runs.** The monotone accumulator eventually
stops every coordinate, which is why RMSProp and Adam displaced it for deep
networks and it survives mainly on sparse, near-convex problems.

**Adam without decoupled decay.** Folding L2 into the gradient under an
adaptive optimiser produces models that generalise worse than SGD with
momentum at a matched nominal decay rate — a gap several empirical studies tie
directly to the preconditioning argument above. AdamW closes most of it.

**Skipping warmup after scaling the batch size or the peak learning rate.**
The linear-scaling approximation holds locally; without warmup, large-batch or
high-peak-rate runs diverge often enough in the first few hundred steps that
warmup length is treated as a hyperparameter of its own, not a convenience.

**Reading $O(1/\sqrt K)$ as a wall-clock guarantee.** It bounds the number of
gradient evaluations needed, not seconds. Halving the batch size to take more
steps in the same wall-clock time does not beat the bound once the noise
floor, not the step count, is what limits progress.

## The exercise

You implement four optimiser steps sharing one interface —
`step_fn(w, grad, state, **hyperparameters) -> (new_w, new_state)` — so a
single loop can drive any of them. Write `sgd_step`, `momentum_step`, and
`adam_step` with the bias correction derived above, then `adamw_step`, which
reuses `adam_step`'s moment update and adds the decoupled decay term
separately. Write `cosine_schedule(step, warmup, total, peak)`, returning the
learning rate at a given step, and `run_optimiser(grad_fn, init, step_fn,
steps)`, a loop that starts at `init`, calls `step_fn` `steps` times, and
returns the final iterate and the full trajectory.

The checks build from the smallest piece up: that `sgd_step` and
`momentum_step`'s velocity recursion match direct arithmetic, that momentum's
steady-state step size under a constant gradient converges to $\eta/(1-\beta)$
times that gradient, and that `adam_step`'s raw and bias-corrected moments
match the closed-form expectation derived above at small $t$ — the check that
catches a missing or wrong correction term outright. Another compares
`adam_step` and `adamw_step` on identical inputs and requires their outputs to
differ by exactly $\eta\lambda w$, and another rescales the gradient by a large
constant to confirm Adam's update barely moves while SGD's scales with it
directly. `cosine_schedule` is checked at the end of warmup, at the total
step count, and at the midpoint of the decay. The exercise closes on the
result the lesson is about: `run_optimiser` driving each of the four step
functions, with hyperparameters you choose, below a target loss on a
quadratic whose two curvatures differ by a factor of a hundred — the
anisotropic problem no single global learning rate handles well in every
direction at once.
