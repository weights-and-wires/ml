---
title: Recurrent networks and BPTT
slug: 22-recurrent-networks
part: "Part 4 — Vision and sequences"
summary: "Build an RNN that reads a sequence one step at a time, train it with backpropagation through time, and see why gradients fade."
minutes: 70
gpu: false
objectives:
  - Build a vanilla RNN cell and run it over a whole sequence in numpy.
  - Compute backpropagation through time with one backward loop over the steps.
  - Explain how the largest eigenvalue size of the recurrent weights predicts vanishing or exploding gradients.
  - Build truncated backpropagation and gradient clipping, and explain what each one trades away.
prereqs:
  - 16-backpropagation
exercise: 22-recurrent-networks
---

# Recurrent networks and BPTT

A sentence, a stock price, and an audio clip are all sequences. Their length
changes from one example to the next, and the order matters. A normal
feed-forward network wants a fixed-size input, so it has no clean way to read
them.

A **recurrent neural network** (RNN) reads a sequence one step at a time, the
way you read a sentence one word at a time. It keeps a running summary called
the **hidden state**. At each step it mixes the new input with the old summary
to make a new summary. It uses the same weights at every step, so one small
set of weights handles a sequence of any length.

In this lesson you build that cell, run it over a sequence, and train it with
**backpropagation through time** (BPTT). You then see the catch: a gradient
that travels back many steps gets multiplied by the same matrix again and
again, so it fades to nothing or blows up. You measure that effect and build
the two standard tools for living with it.

## What you already know

- Backpropagation applies the chain rule from the loss back to each weight.
- When one weight is used in several places, its gradient is the sum of the
  gradients from every place.
- The derivative of $\tanh(z)$ is $1 - \tanh(z)^2$, which is between 0 and 1.
- An **eigenvalue** $\lambda$ of a square matrix $W$ is a number for which
  some vector $v$ satisfies $W v = \lambda v$.
- The **norm** of a vector is its length, `np.linalg.norm(v)`.

## The recurrent cell

Picture someone listening to a story and keeping a one-line summary in their
head. Each new sentence updates the summary. The summary is the hidden state
$h$, and the update rule is the cell.

Sequences in this exercise are **time-major**: `x` has shape `(T, n, d_x)`,
with $T$ steps, $n$ examples in a batch, and $d_x$ input features per step.
The hidden state has $d_h$ numbers per example. One step of the cell is

$$
h_t = \tanh\left(x_t W_{xh}^{\top} + h_{t-1} W_{hh}^{\top} + b\right)
$$

Read it aloud: take this step's input $x_t$ and multiply by the
input-to-hidden weights $W_{xh}$, shape `(d_h, d_x)`. Take the previous hidden
state $h_{t-1}$ and multiply by the hidden-to-hidden weights $W_{hh}$, shape
`(d_h, d_h)`. Add the two and a bias $b$, shape `(d_h,)`. Squash the result
with $\tanh$ so every entry stays between $-1$ and $1$. In numpy:

```python
h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)
```

To run a whole sequence, start from an initial state $h_0$ and apply the cell
once per step, storing each new state. The output has shape `(T, n, d_h)`.
Row 0 holds $h_1$, the state after the first input; $h_0$ itself isn't stored.

### Work a forward pass by hand

Use one example, one input feature, and one hidden unit, so every quantity is
a single number. Let $W_{xh} = 0.5$, $W_{hh} = 0.8$, $b = 0$, $h_0 = 0$, and
inputs $x_1 = 1$, $x_2 = 0.5$.

- Step 1: $z_1 = 0.5 \times 1 + 0.8 \times 0 = 0.5$, so
  $h_1 = \tanh(0.5) = 0.4621$.
- Step 2: $z_2 = 0.5 \times 0.5 + 0.8 \times 0.4621 = 0.6197$, so
  $h_2 = \tanh(0.6197) = 0.5509$.

Here $z_t$ is the value inside the $\tanh$, called the **pre-activation**.

## Backpropagation through time

Unroll the loop and you get a long chain of identical cells, one per step.
BPTT is ordinary backpropagation on that chain. You start at the last step and
walk backward to the first.

The gradient reaching a hidden state $h_t$ comes from two places: the loss
attached directly to $h_t$, which the exercise passes in as `dH[t]`, and
everything later in the sequence, which flows back through $h_{t+1}$. Call the
second part the **carry**. At each step, going backward:

$$
dh_t = dH_t + \text{carry}, \qquad
dz_t = dh_t \odot \left(1 - h_t^2\right), \qquad
\text{carry} \leftarrow dz_t\, W_{hh}
$$

Read it aloud: add the direct gradient to what came back from the future. Pass
it through the $\tanh$ by multiplying element by element with $1 - h_t^2$,
which is the $\tanh$ derivative written using the output you already stored.
Then send it one step further back by multiplying by $W_{hh}$. The symbol
$\odot$ means element-by-element multiplication.

Every weight is reused at every step, so its gradient is a sum over steps:

$$
dW_{xh} = \sum_t dz_t^{\top} x_t, \qquad
dW_{hh} = \sum_t dz_t^{\top} h_{t-1}, \qquad
db = \sum_t \textstyle\sum_{\text{rows}} dz_t
$$

Read it aloud: at each step, the upstream gradient transposed times that
step's input gives a contribution, and you add them all up. The input gradient
at each step is $dx_t = dz_t W_{xh}$. When the loop ends, whatever is left in
the carry is the gradient for $h_0$. In numpy, inside a loop over
`reversed(range(T))`:

```python
dz = (dH[t] + carry) * (1.0 - h_seq[t] ** 2)
dW_hh += dz.T @ h_prev
carry = dz @ W_hh
```

### Work the backward pass by hand

Continue the example. Put a loss gradient of 1 on $h_2$ and 0 on $h_1$.

- Step 2: $dh_2 = 1$. $dz_2 = 1 \times (1 - 0.5509^2) = 0.6965$.
  Carry $= 0.6965 \times 0.8 = 0.5572$.
- Step 1: $dh_1 = 0 + 0.5572$. $dz_1 = 0.5572 \times (1 - 0.4621^2) = 0.4382$.
  Carry $= 0.4382 \times 0.8 = 0.3506$.

The sums give:

- $dW_{xh} = 0.6965 \times 0.5 + 0.4382 \times 1 = 0.7865$.
- $dW_{hh} = 0.6965 \times 0.4621 + 0.4382 \times 0 = 0.3219$.
- $db = 0.6965 + 0.4382 = 1.1347$.
- $dx_1 = 0.4382 \times 0.5 = 0.2191$ and $dx_2 = 0.6965 \times 0.5 = 0.3482$.
- $dh_0 = 0.3506$.

## Why gradients vanish or explode

Look at what happens to the carry. Every step backward multiplies it by
$(1 - h^2)$ and by $W_{hh}$. Over many steps, you multiply by nearly the same
thing many times. It's like compound interest: a rate a little below one
shrinks your money to zero over enough years, and a rate a little above one
makes it explode.

For a matrix, the "rate" is its **spectral radius** $\rho(W)$, the largest
absolute value among its eigenvalues. If the $\tanh$ slope along the path is
about $\kappa$ (a number between 0 and 1), then after $T$ steps the gradient
size is roughly

$$
\left(\kappa \, \rho(W_{hh})\right)^T
$$

Read it aloud: multiply the slope by the spectral radius, then raise it to the
number of steps. Below one, it shrinks to zero; that's a **vanishing
gradient**. Above one, it grows without limit; that's an **exploding
gradient**. In numpy:

```python
rho = np.max(np.abs(np.linalg.eigvals(W)))
scale = (kappa * rho) ** T
```

Eigenvalues can be complex numbers, so take `np.abs` before `np.max`. For
example, with $W = \operatorname{diag}(0.5, 0.9)$, the eigenvalues are 0.5 and
0.9, so $\rho = 0.9$. With $\kappa = 1$ and $T = 10$, the prediction is
$0.9^{10} = 0.349$. Over 100 steps, $0.9^{100} \approx 2.7 \times 10^{-5}$,
while $1.05^{100} \approx 131.5$. Small changes in $\rho$ make huge
differences.

Because $\kappa \le 1$ for $\tanh$, a spectral radius below one always leads
to vanishing. A spectral radius above one can explode, but $\tanh$ saturates
as the state grows, which pulls $\kappa$ back down. That leaves sudden spikes
rather than steady growth.

## Truncated BPTT

Storing every hidden state of a very long sequence costs a lot of memory.
**Truncated BPTT** runs the forward pass over the whole sequence as usual, but
runs the backward pass separately on chunks of $k$ steps. At the start of each
chunk, it treats the incoming hidden state as a fixed constant, so no gradient
crosses from one chunk into the one before it.

The cost is a **bias**: the gradient always misses any link longer than the
chunk. If your task depends on something 50 steps back and $k = 20$, the model
never gets the signal it needs to learn it. When $k$ is at least the sequence
length, truncated BPTT gives exactly the full BPTT answer.

## Gradient clipping by global norm

Clipping stops a sudden spike from wrecking your weights. Treat all the
gradient arrays as one long vector $g$ and measure its length $G$, the
**global norm**. If $G$ is larger than a limit, shrink every array by the
same factor:

$$
g \leftarrow \min\left(1, \frac{\text{max\_norm}}{G}\right) g
$$

Read it aloud: if the gradient is already short enough, multiply by 1 and
leave it alone. Otherwise multiply by $\text{max\_norm}/G$, which makes its
length exactly $\text{max\_norm}$. Every array gets the same factor, so the
direction doesn't change; only the step size does. In numpy:

```python
G = np.sqrt(sum(np.sum(g ** 2) for g in grads))
scale = min(1.0, max_norm / G)
```

For example, two gradients $[3]$ and $[4]$ have global norm 5. With a limit of
1, the scale is $0.2$, giving $[0.6]$ and $[0.8]$, whose global norm is
exactly 1.

## Teacher forcing and why RNNs are slow

When an RNN learns to predict the next item, you can feed it the true previous
item during training. That's **teacher forcing**. It trains fast, but at test
time the model sees its own guesses, which may be wrong in ways it never saw
in training. This mismatch is called **exposure bias**.

An RNN also can't compute step $t$ until step $t-1$ is done. A sequence of
length $T$ takes $T$ steps in a row, however many processor cores you have.
Later lessons on attention remove that limit.

## Build it step by step

The checks test the functions in this order.

1. **`rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b)`.** Takes `x_t` of shape
   `(n, d_x)` and `h_prev` of shape `(n, d_h)`. Returns the new state, shape
   `(n, d_h)`. Use `@`, `.T`, and `np.tanh`. Common mistake: writing
   `W_xh @ x_t`, which has the wrong shapes for batched rows.
2. **`rnn_forward(x, h0, W_xh, W_hh, b)`.** Takes `x` of shape `(T, n, d_x)`.
   Returns all states, shape `(T, n, d_h)`. Allocate with `np.empty`, loop over
   `t`, and call your cell. Common mistake: storing `h0` in row 0.
3. **`bptt(dH, x, h0, W_xh, W_hh, b)`.** Rerun `rnn_forward` to get the
   states. Start the carry at zeros and loop `for t in reversed(range(T))`.
   Use `h_seq[t - 1]` as the previous state, or `h0` when `t == 0`. Return
   `(dx, dW_xh, dW_hh, db, dh0)`. Use `+=` for the weight sums and
   `.sum(axis=0)` for the bias. Common mistake: using `h_seq[t]` instead of
   the previous state in `dW_hh`.
4. **`truncated_bptt(dH, x, h0, W_xh, W_hh, b, k)`.** Run the forward pass
   once. For each chunk `start` in `range(0, T, k)`, call your `bptt` on the
   slice, with the true state just before the chunk as its `h0`. Add the weight
   gradients, copy the chunk's `dx`, and throw away its `dh0`. Return
   `(dx, dW_xh, dW_hh, db)`. Common mistake: restarting each chunk from the
   original `h0`.
5. **`clip_by_global_norm(grads, max_norm)`.** Return a new list with every
   array times one shared scale. Guard against a global norm of zero. Common
   mistake: clipping each array by its own norm, which changes the direction.
6. **`gradient_scale_over_time(W, phi_prime_bound, T)`.** Return
   `(phi_prime_bound * rho) ** T` as a float. Common mistake: taking the
   largest eigenvalue without `np.abs`.

## Cost and scaling

The table uses $n$ examples, $T$ steps, input size $d_x$, and hidden size
$d_h$.

| Quantity | How it grows |
|---|---|
| Weights | $d_h (d_h + d_x + 1)$, the same for any $T$ |
| Work for one step | about $2 n (d_h^2 + d_h d_x)$ operations |
| Work for the sequence | $T$ times one step |
| Memory for full BPTT | grows with $n T d_h$ |
| Memory for truncated BPTT | grows with $n k d_h$ |
| Steps that must run in order | $T$, on any hardware |

The weight count doesn't depend on sequence length. Everything else grows in
a straight line with $T$.

## Where this breaks

**Long-range links are hard to learn.** With $\rho(W_{hh}) < 1$, a gradient
from 50 steps back is tiny before training even starts. Gated cells in the
next lesson are the standard fix.

**Spikes happen anyway.** Even a well-behaved network can produce a sudden
huge gradient. People clip every step, not only when $\rho$ is above one.

**The truncation window is a guess.** Too small a $k$ hides long links; too
large a $k$ loses the memory saving.

**Exposure bias grows with output length.** Long generated outputs give one
early mistake many steps to compound.

**Training is sequential.** No trick changes the fact that $T$ dependent steps
take $T$ steps of time.

## The exercise

You build `rnn_cell_forward` and `rnn_forward`, then `bptt` for the full
backward pass and `truncated_bptt` for the chunked version. You add
`clip_by_global_norm` and `gradient_scale_over_time`, the spectral-radius
prediction.

The checks compare your forward pass with a direct computation and your
`bptt` with finite differences for every input. They confirm `truncated_bptt`
matches `bptt` when `k` equals the sequence length and differs when `k` is 2.
They confirm clipping leaves a small gradient alone and scales a large one to
exactly the limit without turning it. Finally, they run 100 steps with
$\rho = 0.9$ and $\rho = 1.05$ and confirm your measured `dh0` is within a
factor of ten of your prediction.
