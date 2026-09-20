---
title: Recurrent networks and BPTT
slug: 22-recurrent-networks
part: "Part 4 — Vision and sequences"
summary: Backpropagation through time derived as a sum of chain rules, the spectral radius that decides whether it vanishes or explodes, and why the recursion caps parallelism.
minutes: 55
gpu: false
objectives:
  - Derive backpropagation through time as a sum over time steps of the chain rule, with the repeated Jacobian product it implies.
  - State precisely, via the spectral radius of the recurrent weight matrix, when that product vanishes and when it explodes.
  - Explain the bias truncated BPTT introduces, and why clipping by global norm preserves the update's direction.
  - Explain why the recurrent state equation caps parallelism in a way convolutions and dense layers do not.
prereqs:
  - 16-backpropagation
exercise: 22-recurrent-networks
---

# Recurrent networks and BPTT

A convolution shares one small set of weights across every spatial position
because it assumes the same local pattern can appear anywhere in an image. A
recurrent network makes the same bet about time: the same weights that turned
$h_{t-1}$ into $h_t$ should turn $h_t$ into $h_{t+1}$, because whatever
dynamics govern a sequence — a sentence, a time series, an audio waveform —
are assumed to be the same dynamics at every step. That single assumption
turns a sequence of arbitrary length into a fixed number of parameters, and it
turns training into an instance of the reverse-mode automatic differentiation
from lesson 16 applied to a computation graph shaped like a very long, very
thin chain.

The thinness of that chain is the whole story of this lesson. A gradient that
has to cross $T$ identical steps to reach an early input is a product of $T$
near-identical Jacobians, and unlike the additive identity term a residual
connection supplies, a plain recurrence has no such floor: the product either
shrinks geometrically, grows geometrically, or, on a knife's edge, does
neither. This lesson derives backpropagation through time as an explicit sum
of chain rules, states exactly when that sum vanishes or explodes in terms of
one number — the spectral radius of the recurrent weight matrix — and ends
with the property that has nothing to do with gradients at all: the
recursion itself, $h_t$ depending on the *value* of $h_{t-1}$, forces every
step to wait for the one before it, which is the opening the transformer
walks through by removing the recursion altogether.

## What you already know

- Reverse-mode automatic differentiation computes a gradient by propagating
  vector-Jacobian products backward through a computation graph, and a
  parameter used at multiple points in the graph receives the sum of the
  gradients from every use.
- For $A \in \mathbb{R}^{d \times d}$, the spectral radius $\rho(A)$ is the
  largest magnitude among its (possibly complex) eigenvalues, and Gelfand's
  formula states $\lim_{n \to \infty} \lVert A^n \rVert^{1/n} = \rho(A)$.
- $\tanh$ has derivative $\phi'(z) = 1 - \tanh^2(z) \in (0, 1]$, equal to $1$
  only at $z = 0$ and shrinking toward $0$ as $|z|$ grows.
- `mlcheck.finite_difference` approximates a gradient by central differences,
  accurate to $O(\varepsilon^2)$ per coordinate.
- A matrix and its scalar multiples share eigenvectors, and scaling a vector
  by a positive number changes its norm but never its direction.

## The recurrent state equation

A vanilla recurrent cell maps a state and an input to a new state:

$$
h_t = \phi(z_t), \qquad z_t = W_{hh} h_{t-1} + W_{xh} x_t + b ,
$$

with $\phi = \tanh$, $h_t \in \mathbb{R}^{d_h}$, $x_t \in \mathbb{R}^{d_x}$,
and the same $W_{hh} \in \mathbb{R}^{d_h \times d_h}$, $W_{xh} \in
\mathbb{R}^{d_h \times d_x}$, $b \in \mathbb{R}^{d_h}$ reused at every $t$.
Unrolled over $T$ steps this is an ordinary computation graph — $T$ copies of
the same cell, wired in a chain, with tied weights — so nothing about
computing its gradient is new in kind. What is new is that a single set of
weights now appears $T$ times, so the multivariable chain rule's rule for a
repeated variable, "sum the gradient from every occurrence," is not a detail;
it is most of the computation.

## Backpropagation through time, derived

Suppose the loss accumulates one term per step, $\ell = \sum_{t=1}^{T}
\ell_t(h_t)$; a loss attached only to the final state is the special case
$\ell_t \equiv 0$ for $t < T$. Differentiating the state equation,

$$
\frac{\partial h_{t+1}}{\partial h_t} = \operatorname{diag}\big(\phi'(z_{t+1})\big)\, W_{hh} ,
$$

so the total gradient reaching $h_t$ — from its own loss term and from every
later step through the chain of these Jacobians — obeys the backward
recursion

$$
\frac{\partial \ell}{\partial h_t} = \frac{\partial \ell_t}{\partial h_t} +
\left(\frac{\partial \ell}{\partial h_{t+1}}\right)^{\!\top}
\operatorname{diag}\big(\phi'(z_{t+1})\big)\, W_{hh} ,
$$

read as a row-vector recursion running from $t = T$ down to $t = 1$. Unrolling
it once gives the closed form this lesson is named for: the gradient reaching
step $t$ is a **sum over every later step**, each term the local loss
gradient at that step carried back through a **product of repeated
Jacobians**,

$$
\frac{\partial \ell}{\partial h_t} = \sum_{t'=t}^{T}
\frac{\partial \ell_{t'}}{\partial h_{t'}}
\prod_{s=t+1}^{t'} \operatorname{diag}\big(\phi'(z_s)\big)\, W_{hh} .
$$

Every term with $t' > t$ is a chain of $t' - t$ copies of the same kind of
factor, $\operatorname{diag}(\phi'(z_s))\,W_{hh}$ — the repeated Jacobian
product that decides everything in the next section. The parameter gradients
follow the same pattern as a convolution's: define $\delta_t =
\operatorname{diag}(\phi'(z_t))\, (\partial \ell / \partial h_t)^{\top}$ (a
column vector), and

$$
\frac{\partial \ell}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\, h_{t-1}^{\top}, \qquad
\frac{\partial \ell}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t\, x_t^{\top}, \qquad
\frac{\partial \ell}{\partial b} = \sum_{t=1}^{T} \delta_t ,
$$

sums over every time step because the same weight is reused at every one of
them — the direct consequence of weight sharing across time, exactly as
weight sharing across space made a convolution's weight gradient a sum over
every spatial position it touched.

## Vanishing and exploding gradients: the spectral radius test

Consider a stretch of steps where the local slope stays roughly constant,
$\phi'(z_s) \approx \kappa$ for $s = t+1, \dots, t'$ — the linearised regime
near a fixed point, or simply small pre-activations. The repeated Jacobian
product becomes $(\kappa W_{hh})^{\,t' - t}$ up to the reordering that
diag-times-matrix products in general prevent, and Gelfand's formula gives
the precise asymptotic behaviour of its norm:

$$
\lVert (\kappa W_{hh})^n \rVert \sim \big(\kappa\, \rho(W_{hh})\big)^n
\quad \text{as } n \to \infty ,
$$

up to a polynomial correction in $n$ from non-trivial Jordan blocks. Three
regimes follow, and they are exhaustive:

- $\kappa\, \rho(W_{hh}) < 1$: the product decays exponentially in the number
  of steps crossed — the **vanishing gradient**.
- $\kappa\, \rho(W_{hh}) > 1$: it grows exponentially — the **exploding
  gradient**.
- $\kappa\, \rho(W_{hh}) = 1$: neither, a critical point some initialisation
  schemes target deliberately, and one that any drift in the parameters
  during training immediately leaves.

For $\tanh$, $\kappa = \phi'(z) \le 1$ always, so $\rho(W_{hh}) < 1$ is
**sufficient for eventual vanishing, unconditionally** — no choice of data or
activation pattern can prevent it, because the bound $\kappa \le 1$ holds
everywhere. The converse is not unconditional: $\rho(W_{hh}) > 1$ makes
explosion *possible*, but only while the trajectory stays where $\kappa$ is
close to its bound. $\tanh$ caps the forward state itself to
$(-1, 1)^{d_h}$, so a state that grows is pushed toward $|z|$ large, where
$\phi'(z) \to 0$ throttles $\kappa$ back down — the forward pass cannot
diverge the way a plain linear recursion can. The gradient can still spike
sharply before that saturation engages, which is exactly the transient
instability gradient clipping is built to survive rather than prevent.

## Truncated BPTT and the bias it introduces

Storing and differentiating through every step of a long sequence costs
memory linear in $T$, and gradient contributions that have crossed enough
vanishing-regime steps are negligible anyway. **Truncated BPTT** cuts the
backward pass at a fixed window $k$: split the sequence into chunks of $k$
steps, run the forward recursion continuously across the whole sequence, but
backpropagate independently within each chunk, treating the hidden state
carried in from the previous chunk as a constant with no gradient attached.

This is a bias, not noise. The full sum derived above includes, for a loss at
step $t'$ near the end of a chunk, contributions from every earlier step back
to $t = 1$; truncation deletes every term whose path crosses a chunk
boundary, every time, in the same direction — toward zero contribution from
anything more than $k$ steps back — regardless of whether that contribution
was large or small. When $\rho(W_{hh}) < 1$ and the true dependency decays
within $k$ steps anyway, the deleted terms were negligible and the bias is
harmless. When a task genuinely depends on structure more than $k$ steps
back, truncated BPTT cannot learn it, no matter how long training runs,
because the gradient that would teach it is discarded before it is computed.

## Gradient clipping by global norm

Collect every parameter's gradient into one list $g_1, \dots, g_m$ and define
the **global norm** $G = \big(\sum_i \lVert g_i \rVert_F^2\big)^{1/2}$ — the
norm of the concatenation of every $g_i$ into a single vector $g$. Clipping
by global norm rescales every $g_i$ by the same scalar,

$$
g_i \leftarrow \min\!\left(1, \frac{\text{max\_norm}}{G}\right) g_i .
$$

Every entry of the concatenated vector is multiplied by the same positive
number $c$, so the clipped vector is $cg$: a positive scalar multiple of $g$,
with $\cos(cg, g) = 1$ exactly, and norm exactly $cG = \min(G,
\text{max\_norm})$. Clipping never changes the direction the optimiser steps
in; it only caps how far it steps, which is why it is the standard response
to the transient spikes the exploding-gradient analysis above predicts,
rather than a fix that competes with better-conditioned architectures — it
buys stability without touching what the gradient is trying to say.

## Teacher forcing and exposure bias

Training a sequence model to predict $x_{t+1}$ from its own history can feed
back either the ground-truth $x_t$ or the model's own (possibly wrong)
prediction as the next step's input. **Teacher forcing** always feeds the
ground truth, so every step's recurrence starts from a state that reflects
the correct history rather than one contaminated by an earlier mistake; this
converges faster and avoids compounding early errors during training. It also
never trains the model on the states its own mistakes produce, so at
inference — where there is no ground truth left to feed back — a single wrong
prediction can push the hidden state into a region training never visited,
and errors compound in a way the training loss never penalised. This gap
between training-time and inference-time input distributions is **exposure
bias**, and it is worse for exactly the long, free-running generations that
matter most in practice.

## Why sequential computation caps parallelism

$h_t$ depends on the *value* of $h_{t-1}$, not merely on shared parameters —
a genuine data dependency between steps, not a notational one. No hardware
budget removes it: computing $h_t$ cannot begin until $h_{t-1}$ exists, so a
sequence of length $T$ forces $T$ sequential steps forward and $T$ sequential
steps backward, however many accelerator cores sit idle waiting. A
convolution or a dense layer has no such dependency between output
positions — every position is an independent function of the input, computed
in parallel — and this is precisely the property self-attention restores for
sequences: every position attends to every other position through matrix
multiplies with no data dependency between positions within a layer. The
recurrence that makes an RNN's parameter count independent of sequence length
is the same recurrence that makes its wall-clock training time scale with
sequence length no matter the hardware, and that trade is what the next part
of this course revisits.

## Cost and scaling

| Quantity | Cost |
|---|---|
| Parameters | $d_h(d_h + d_x + 1)$, independent of $T$ |
| Flops, one step | $2N(d_h^2 + d_h d_x)$ |
| Flops, full sequence | $2NT(d_h^2 + d_h d_x)$ |
| Activation memory, full BPTT | $O(NTd_h)$ |
| Activation memory, truncated at $k$ | $O(Nkd_h)$ |
| Sequential steps, forward or backward | $T$, regardless of parallel hardware |

Parameter count is flat in $T$ because of weight sharing; every other row
grows with it, and the memory row is the concrete reason truncated BPTT
exists even on tasks where the bias it introduces is not the primary
concern.

## Where this breaks

**Long-range dependencies are structurally hard to learn.** Whenever
$\rho(W_{hh}) < 1$ — the common case, since larger values risk the exploding
regime — every gradient contribution from more than a few dozen steps back is
exponentially small before training even sees it, whether or not the task
needs that dependency.

**Exploding gradients still happen inside the safe-looking regime.** Even a
network with $\rho(W_{hh})$ near one can spike sharply on particular
sequences where activations linger near zero for many consecutive steps,
which is why clipping is applied unconditionally in practice rather than
reserved for networks with $\rho(W_{hh}) > 1$.

**Truncated BPTT's window is a guess.** Choosing $k$ requires knowing how far
back a task's dependencies reach, which is rarely known in advance; too small
silently caps what the model can ever learn, and too large defeats the memory
saving that motivated truncation.

**Exposure bias grows with generation length.** Teacher forcing's convergence
advantage is largest exactly where its blind spot is worst: long,
autoregressive outputs are where a single early mistake has the most steps
left in which to compound.

**The architecture itself resists parallel hardware.** No amount of
gradient-flow engineering changes the fact that $T$ sequential dependencies
take $T$ sequential steps; this is a ceiling on throughput that better
initialisation, clipping, or gating (next lesson) cannot lift, because it is
a property of the recursion, not of the numbers flowing through it.

## The exercise

You implement a vanilla RNN's forward pass and its backpropagation through
time by hand, then measure the vanishing and exploding regimes this lesson
derived, rather than taking the spectral radius argument on faith.

Write `rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b)` for one step and
`rnn_forward(x, h0, W_xh, W_hh, b)` for a full sequence. Write `bptt(dH, x,
h0, W_xh, W_hh, b)`, returning the gradient with respect to the input, both
weight matrices, the bias, and the initial hidden state, by implementing the
backward recursion derived above. Write `truncated_bptt(dH, x, h0, W_xh,
W_hh, b, k)`, which must reduce to full `bptt` when $k$ equals the sequence
length. Write `clip_by_global_norm(grads, max_norm)`, and
`gradient_scale_over_time(W, phi_prime_bound, T)`, which returns the
Gelfand-formula prediction $(\text{phi\_prime\_bound} \cdot \rho(W))^T$ for
the size of a $T$-step Jacobian product.

The checks verify `rnn_cell_forward` and `rnn_forward` against direct
computation, `bptt` against `mlcheck.finite_difference` on a short sequence,
that `truncated_bptt` with $k$ equal to the sequence length matches full
`bptt` exactly, that `clip_by_global_norm` leaves an under-threshold gradient
untouched and rescales an over-threshold one to exactly the threshold norm
without changing its direction, that `gradient_scale_over_time` matches an
independent eigenvalue computation, and that the measured gradient magnitude
after 100 steps of a near-linear recurrence tracks the spectral-radius
prediction for both a contracting and an expanding recurrent weight matrix.
