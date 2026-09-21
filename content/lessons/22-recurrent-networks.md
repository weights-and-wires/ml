---
title: Recurrent networks and BPTT
slug: 22-recurrent-networks
part: "Part 4 — Vision and sequences"
summary: "Backpropagation through time as a chain of repeated Jacobians, the spectral radius that predicts vanishing or exploding, and its parallelism ceiling."
minutes: 45
gpu: false
objectives:
  - Derive backpropagation through time as a sum over time steps of the chain rule.
  - State, using the spectral radius of the recurrent weight matrix, when the resulting product vanishes and when it explodes.
  - Explain the bias truncated BPTT introduces, and why clipping by global norm preserves the update's direction.
  - Explain why the recurrent state equation caps parallelism in a way convolutions and dense layers don't.
prereqs:
  - 16-backpropagation
exercise: 22-recurrent-networks
---

# Recurrent networks and BPTT

A convolution shares one small set of weights across every spatial position,
betting that the same local pattern can appear anywhere in an image. A
recurrent network makes the same bet about time: the weights that turn
$h_{t-1}$ into $h_t$ should turn $h_t$ into $h_{t+1}$, because a sentence, a
time series, or an audio waveform is assumed to follow the same dynamics at
every step. That single assumption turns a sequence of any length into a
fixed number of parameters, and it turns training into reverse-mode automatic
differentiation applied to a computation graph shaped like a long, thin
chain.

The result this lesson builds toward: a gradient that has to cross $T$
identical steps to reach an early input is a product of $T$ near-identical
Jacobians, and unlike the additive identity term a residual connection
supplies, a plain recurrence has no floor under that product. It either
shrinks geometrically, grows geometrically, or, on a knife's edge, does
neither, and one number — the spectral radius of the recurrent weight
matrix — tells you which. This lesson derives that result, then covers what
practitioners do about it, and ends with a property that has nothing to do
with gradients: $h_t$ depends on the *value* of $h_{t-1}$, so every step has
to wait for the one before it. That dependency is the opening the transformer
walks through by removing the recurrence entirely.

## What you already know

- Reverse-mode automatic differentiation propagates vector-Jacobian products
  backward through a computation graph, and a parameter used at multiple
  points receives the sum of the gradients from every use.
- For $A \in \mathbb{R}^{d \times d}$, the spectral radius $\rho(A)$ is the
  largest magnitude among its eigenvalues, and Gelfand's formula states
  $\lim_{n \to \infty} \lVert A^n \rVert^{1/n} = \rho(A)$.
- $\tanh$ has derivative $\phi'(z) = 1 - \tanh^2(z) \in (0, 1]$, equal to $1$
  only at $z = 0$.
- `mlcheck.finite_difference` approximates a gradient by central differences.

## The recurrent state equation

A vanilla recurrent cell maps a state and an input to a new state:

$$
h_t = \phi(z_t), \qquad z_t = W_{hh} h_{t-1} + W_{xh} x_t + b ,
$$

with $\phi = \tanh$, $h_t \in \mathbb{R}^{d_h}$, and the same $W_{hh}$,
$W_{xh}$, and $b$ reused at every $t$. Unrolled over $T$ steps, this is $T$
copies of the same cell wired in a chain with tied weights. The one thing
that's new is that a single set of weights now appears $T$ times, so the
multivariable chain rule's rule for a repeated variable — sum the gradient
from every occurrence — is most of the computation.

## Backpropagation through time

Suppose the loss accumulates one term per step, $\ell = \sum_{t=1}^{T}
\ell_t(h_t)$. Differentiating the state equation gives the local Jacobian

$$
\frac{\partial h_{t+1}}{\partial h_t} = \operatorname{diag}\big(\phi'(z_{t+1})\big)\, W_{hh} ,
$$

so the gradient reaching $h_t$ — from its own loss term and from every later
step — obeys the backward recursion

$$
\frac{\partial \ell}{\partial h_t} = \frac{\partial \ell_t}{\partial h_t} +
\left(\frac{\partial \ell}{\partial h_{t+1}}\right)^{\!\top}
\operatorname{diag}\big(\phi'(z_{t+1})\big)\, W_{hh} ,
$$

running from $t = T$ down to $t = 1$. Unrolling it gives the closed form this
lesson is named for: the gradient reaching step $t$ is a sum over every later
step, each term carried back through a product of repeated Jacobians,

$$
\frac{\partial \ell}{\partial h_t} = \sum_{t'=t}^{T}
\frac{\partial \ell_{t'}}{\partial h_{t'}}
\prod_{s=t+1}^{t'} \operatorname{diag}\big(\phi'(z_s)\big)\, W_{hh} .
$$

Every term with $t' > t$ chains together $t' - t$ copies of the same kind of
factor — the repeated Jacobian product the next section studies. The
parameter gradients follow the same pattern as a convolution's shared
weight: with $\delta_t = \operatorname{diag}(\phi'(z_t))\, (\partial \ell /
\partial h_t)^{\top}$,

$$
\frac{\partial \ell}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_t\, h_{t-1}^{\top}, \qquad
\frac{\partial \ell}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_t\, x_t^{\top}, \qquad
\frac{\partial \ell}{\partial b} = \sum_{t=1}^{T} \delta_t .
$$

Each is a sum over every time step, because the same weight is reused at
every one of them.

## The spectral radius test

Consider a stretch of steps where the local slope stays roughly constant,
$\phi'(z_s) \approx \kappa$. The repeated Jacobian product then behaves like
$(\kappa W_{hh})^{n}$ for $n$ steps, and Gelfand's formula gives its
asymptotic size:

$$
\lVert (\kappa W_{hh})^n \rVert \sim \big(\kappa\, \rho(W_{hh})\big)^n
\quad \text{as } n \to \infty .
$$

Three regimes follow, and they're exhaustive: $\kappa\, \rho(W_{hh}) < 1$
decays exponentially in the number of steps crossed, the **vanishing
gradient**; $\kappa\, \rho(W_{hh}) > 1$ grows exponentially, the **exploding
gradient**; and $\kappa\, \rho(W_{hh}) = 1$ is a critical point some
initializations target deliberately, one that any drift in the parameters
during training immediately leaves.

For $\tanh$, $\kappa \le 1$ always, so $\rho(W_{hh}) < 1$ guarantees eventual
vanishing — no choice of data or activation pattern avoids it. The converse
isn't unconditional: $\rho(W_{hh}) > 1$ makes explosion possible, but only
while the trajectory stays where $\kappa$ is close to its bound. $\tanh$ caps
the forward state to $(-1, 1)^{d_h}$, so a growing state gets pushed toward
$|z|$ large, where $\phi'(z) \to 0$ throttles $\kappa$ back down. The gradient
can still spike sharply before that saturation kicks in, which is the
transient instability gradient clipping exists to survive.

## Truncated BPTT and the bias it introduces

Differentiating through every step of a long sequence costs memory linear in
$T$, and contributions that have crossed many vanishing-regime steps are
negligible anyway. **Truncated BPTT** cuts the backward pass at a fixed
window $k$: run the forward recursion continuously across the sequence, but
backpropagate independently within chunks of $k$ steps, treating the hidden
state carried in from the previous chunk as a constant.

This is a bias, not noise: truncation deletes every gradient term whose path
crosses a chunk boundary, every time, in the same direction. When the true
dependency decays within $k$ steps anyway, the deleted terms were negligible.
When a task genuinely depends on structure more than $k$ steps back,
truncated BPTT can't learn it, because the gradient that would teach it is
discarded before it's computed.

## Gradient clipping by global norm

Collect every parameter's gradient into one vector $g$, and define the
**global norm** $G = \lVert g \rVert$. Clipping by global norm rescales every
gradient tensor by the same scalar,

$$
g \leftarrow \min\!\left(1, \frac{\text{max\_norm}}{G}\right) g .
$$

The clipped vector is $cg$ for some positive $c$: a scalar multiple of $g$,
with the same direction and norm exactly $\min(G, \text{max\_norm})$.
Clipping never changes the direction the optimizer steps in; it only caps how
far it steps, which is why it's the standard response to the transient
spikes the exploding-gradient analysis predicts.

## Teacher forcing and exposure bias

A sequence model predicting $x_{t+1}$ from its own history can feed back
either the ground-truth $x_t$ or its own prediction as the next step's input.
**Teacher forcing** always feeds the ground truth, so training converges
faster and avoids compounding early errors. But at inference, where there's
no ground truth left to feed back, a single wrong prediction can push the
hidden state somewhere training never visited, and errors compound in a way
the training loss never penalized. This gap between training-time and
inference-time inputs is **exposure bias**, worst for exactly the long,
free-running generations that matter most in practice.

## Why sequential computation caps parallelism

$h_t$ depends on the *value* of $h_{t-1}$, a genuine data dependency between
steps rather than a notational one. Computing $h_t$ can't begin until
$h_{t-1}$ exists, so a sequence of length $T$ forces $T$ sequential steps
forward and $T$ sequential steps backward, however many accelerator cores sit
idle. A convolution or a dense layer has no such dependency between output
positions — self-attention restores exactly that property for sequences,
letting every position attend to every other position through matrix
multiplies with no dependency between positions within a layer. The
recurrence that keeps an RNN's parameter count independent of sequence length
is the same recurrence that makes its training time scale with sequence
length no matter the hardware.

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
exists even on tasks where the bias it introduces isn't a concern.

## Where this breaks

**Long-range dependencies are structurally hard to learn.** Whenever
$\rho(W_{hh}) < 1$, the common case since larger values risk exploding, every
gradient contribution from more than a few dozen steps back is exponentially
small before training even sees it. A network with $\rho(W_{hh})$ near one
can still spike sharply on sequences where activations linger near zero for
many steps, which is why clipping is applied unconditionally rather than
reserved for networks with $\rho(W_{hh}) > 1$.

**Truncated BPTT's window is a guess.** Choosing $k$ requires knowing how far
back a task's dependencies reach, which is rarely known in advance; too small
caps what the model can learn, and too large defeats the memory saving that
motivated truncation.

**Exposure bias grows with generation length.** Teacher forcing's convergence
advantage is largest exactly where its blind spot is worst: long outputs are
where a single early mistake has the most steps left to compound.

**The architecture resists parallel hardware.** No amount of gradient-flow
engineering changes the fact that $T$ sequential dependencies take $T$
sequential steps — a ceiling that better initialization, clipping, or gating
(next lesson) can't lift, because it's a property of the recursion, not the
numbers flowing through it.

## The exercise

You implement a vanilla RNN's forward pass and its backpropagation through
time by hand, then measure the vanishing and exploding regimes this lesson
derived instead of taking the spectral radius argument on faith.

Write `rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b)` for one step and
`rnn_forward(x, h0, W_xh, W_hh, b)` for a full sequence. Write `bptt(dH, x,
h0, W_xh, W_hh, b)`, returning the gradient with respect to the input, both
weight matrices, the bias, and the initial hidden state. Write
`truncated_bptt(dH, x, h0, W_xh, W_hh, b, k)`, which must reduce to full
`bptt` when $k$ equals the sequence length. Write `clip_by_global_norm(grads,
max_norm)`, and `gradient_scale_over_time(W, phi_prime_bound, T)`, which
returns the Gelfand-formula prediction $(\text{phi\_prime\_bound} \cdot
\rho(W))^T$ for the size of a $T$-step Jacobian product.

The checks verify `rnn_cell_forward` and `rnn_forward` against direct
computation, `bptt` against `mlcheck.finite_difference`, that
`truncated_bptt` with $k$ equal to the sequence length matches full `bptt`
exactly, that `clip_by_global_norm` leaves an under-threshold gradient
untouched and rescales an over-threshold one to exactly the threshold norm
without changing its direction, that `gradient_scale_over_time` matches an
independent eigenvalue computation, and that the measured gradient magnitude
after 100 steps of a near-linear recurrence tracks the spectral-radius
prediction for both a contracting and an expanding recurrent weight matrix.
