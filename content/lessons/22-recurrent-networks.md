---
title: Recurrent networks and BPTT
slug: 22-recurrent-networks
part: "Part 4 — Vision and sequences"
summary: "Build an RNN that reads one step at a time, train it with backpropagation through time, and see gradients fade."
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

A sentence, a price series, and a clip of audio are sequences. The length changes from one example to the next, and the order matters. A feed-forward network takes a fixed-size input, so a changing length does not fit.

You read a sentence one word at a time and keep a one-line summary in your head. Each new word updates that summary. A recurrent network does the same job with one hidden state and one set of weights, reused at every step.

You build that cell, train it with backpropagation through time, and measure when the gradient fades or blows up.

## What you already know

You already have the following:

- Backpropagation applies the chain rule from the loss back to each weight.
- When one weight is used in several places, its gradient is the sum of those places.
- The derivative of $\tanh(z)$ is $1 - \tanh(z)^2$, a number between 0 and 1.
- An eigenvalue $\lambda$ of a square matrix $W$ satisfies $Wv = \lambda v$ for some vector $v$.
- The norm of a vector is its length, `np.linalg.norm`.

## Read one word, update the summary

Use one example, one input feature, and one hidden unit, so every quantity is a single number. Let $W_{xh} = 0.5$, $W_{hh} = 0.8$, $b = 0$, and $h_0 = 0$. The first input is $1$. The value inside the squash is $0.5 \times 1 + 0.8 \times 0 = 0.5$, and $\tanh(0.5) = 0.4621$.

That update is a **vanilla recurrent cell**. Sequences here are time-major: `x` has shape `(T, n, d_x)`, with $T$ steps and $n$ examples. The hidden state has $d_h$ numbers per example. One step is

$$
h_t = \tanh\left(x_t W_{xh}^{\top} + h_{t-1} W_{hh}^{\top} + b\right)
$$

$W_{xh}$ has shape `(d_h, d_x)` and $W_{hh}$ has shape `(d_h, d_h)`. The bias has shape `(d_h,)`. Tanh keeps every entry between $-1$ and $1$. The same second input, $0.5$, gives $0.5 \times 0.5 + 0.8 \times 0.4621 = 0.6197$, and $h_2 = \tanh(0.6197) = 0.5509$. The value inside tanh is the pre-activation.

Rows of the batch are examples, so the multiplies use a transpose on the weights:

```python
h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)
```

The usual mistake is `W_xh @ x_t`, which has the wrong shapes for a batch of rows.

The output is `(T, n, d_h)`. Row 0 is $h_1$. The initial state $h_0$ is not stored.

> [!KEY]
> One cell, with the same weights, runs at every step. The hidden state carries a summary of the inputs so far.

## Walk the chain backward

Unroll the loop and you get a chain of identical cells. **Backpropagation through time** is ordinary backpropagation on that chain. Start at the last step and walk back to the first.

Continue the same numbers. Put a loss gradient of 1 on $h_2$ and 0 on $h_1$. At step 2 the local gradient is 1. Multiply by the tanh derivative $1 - 0.5509^2$ and you get $0.6965$. Send that back through $W_{hh}$: $0.6965 \times 0.8 = 0.5572$.

The gradient at a hidden state comes from two places. `dH[t]` is the loss attached directly to that state. The **carry** is everything later in the sequence, flowing back through the next cell. Going backward:

$$
dh_t = dH_t + \text{carry}, \qquad
dz_t = dh_t \odot \left(1 - h_t^2\right), \qquad
\text{carry} \leftarrow dz_t\, W_{hh}
$$

Add the direct gradient to the carry, multiply by the tanh slope $1 - h_t^2$, then multiply by $W_{hh}$ to step one further back.

On step 1 the direct gradient is 0, so $dh_1 = 0.5572$. Then $dz_1 = 0.5572 \times (1 - 0.4621^2) = 0.4382$, and the carry becomes $0.4382 \times 0.8 = 0.3506$.

Every weight is reused at every step, so its gradient is a sum:

$$
dW_{xh} = \sum_t dz_t^{\top} x_t, \qquad
dW_{hh} = \sum_t dz_t^{\top} h_{t-1}, \qquad
db = \sum_t \textstyle\sum_{\text{rows}} dz_t
$$

Sum each upstream gradient, transposed, times that step's input. $dx_t = dz_t W_{xh}$, and the final carry is the gradient of $h_0$.

The same numbers add up as follows:

- $dW_{xh} = 0.6965 \times 0.5 + 0.4382 \times 1 = 0.7865$.
- $dW_{hh} = 0.6965 \times 0.4621 + 0.4382 \times 0 = 0.3219$.
- $db = 0.6965 + 0.4382 = 1.1347$.
- $dx_1 = 0.2191$, $dx_2 = 0.3482$, and $dh_0 = 0.3506$.

Inside a loop over `reversed(range(T))`, the numpy has this shape:

```python
dz = (dH[t] + carry) * (1.0 - h_seq[t] ** 2)
dW_hh += dz.T @ h_prev
carry = dz @ W_hh
```

The usual mistake is to use `h_seq[t]` as the previous state. Row `t` stores $h_{t+1}$.

> [!WARNING]
> Row `t` of the stored states is $h_{t+1}$. The previous state is `h_seq[t - 1]`, or `h0` when `t` is 0.

> [!KEY]
> Backpropagation through time is the chain rule on the unrolled cells. Each weight's gradient is a sum over steps.

> [!TIP]
> Test this scalar case first. Give `x` and `dH` shape `(2, 1, 1)`, both weights and `h0` shape `(1, 1)`, and `b` shape `(1,)`.

## A rate a little off from one

Each step back multiplies the carry by $(1 - h^2)$ and by $W_{hh}$. A factor a little below one shrinks to zero. A factor a little above one blows up. That is compound interest.

For a matrix, the rate is its **spectral radius** $\rho(W)$, the largest absolute eigenvalue. If the tanh slope along the path is about $\kappa$, a number from 0 to 1, then after $T$ steps the gradient size is about

$$
\left(\kappa \, \rho(W_{hh})\right)^T
$$

Multiply the slope by the spectral radius, then raise it to the number of steps. Below one, the gradient vanishes. Above one, it can explode.

Take $W = \mathrm{diag}(0.5, 0.9)$. The eigenvalues are 0.5 and 0.9, so $\rho = 0.9$. With $\kappa = 1$, ten steps give $0.9^{10} = 0.349$. Over 100 steps, $0.9^{100} \approx 2.7 \times 10^{-5}$ and $1.05^{100} \approx 131.5$:

```python
rho = np.max(np.abs(np.linalg.eigvals(W)))
scale = (kappa * rho) ** T
```

The usual mistake is to take the largest eigenvalue without `np.abs`. Eigenvalues can be complex.

For tanh, $\kappa \le 1$, so a spectral radius below one vanishes. Above one, tanh flattens as the state grows and pulls $\kappa$ down, leaving a spike rather than a steady climb.

> [!KEY]
> Each step back multiplies the gradient by about $\kappa$ times $\rho(W_{hh})$. Below one it vanishes over many steps. Above one it can explode.

## Stop the backward pass at a window

**Truncated backpropagation through time** keeps the full forward pass, then runs the backward pass on chunks of $k$ steps. The state entering a chunk is a constant, so no gradient crosses into the chunk before it.

Take $T = 6$ and $k = 2$. The chunks are steps 0 to 1, 2 to 3, and 4 to 5. A loss on the last chunk does not reach the state that entered it. Links longer than $k$ are missing. If the task needs something 50 steps back and $k = 20$, that signal never arrives. When $k$ is at least the sequence length, the result matches full backpropagation through time.

The usual mistake is to restart each chunk from the original $h_0$. The state entering a chunk is the true state from the step before it.

## Shrink a spike, keep its direction

A sudden huge gradient can wreck the weights. Treat every gradient array as one long vector and measure its length $G$, the global norm. If $G$ is larger than a limit, shrink every array by the same factor:

$$
g \leftarrow \min\left(1, \frac{\text{max\_norm}}{G}\right) g
$$

A short gradient is left alone. A long one is scaled by $\text{max\_norm}/G$, so its length becomes the limit and its direction stays.

Two gradients $[3]$ and $[4]$ have global norm 5. A limit of 1 scales them by $0.2$, to $[0.6]$ and $[0.8]$:

```python
G = np.sqrt(sum(np.sum(g ** 2) for g in grads))
scale = min(1.0, max_norm / G)
```

The usual mistake is clipping each array by its own norm.

> [!KEY]
> Global-norm clipping multiplies every array by one shared factor, so the direction stays and only the length is capped.

Training can feed the true previous item when the task is to predict the next one. That is **teacher forcing**. At test time the inputs are the network's own outputs. The mismatch is **exposure bias**, and it grows with output length. Step $t$ also waits on step $t - 1$, so a length of $T$ takes $T$ steps in a row.

> [!RECAP]
> Hold these five points before you write code:
>
> - The cell is `np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)`. Store $h_1$ through $h_T$, not $h_0$.
> - Walk backward. Add `dH[t]` to the carry, multiply by $1 - h_t^2$, add into the weight sums, then send `dz @ W_hh` back.
> - Truncated backpropagation runs on chunks of `k` steps, each starting from the true state before that chunk.
> - Clipping multiplies every array by $\min(1, \text{max\_norm} / G)$.
> - The predicted scale is $(\kappa \rho)^T$, with $\rho$ the largest absolute eigenvalue.

## Build it step by step

The checks call the functions in this order:

1. **`rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b)`.** `x_t` is `(n, d_x)` and `h_prev` is `(n, d_h)`. Return the new state, `(n, d_h)`. Use `@`, `.T`, and `np.tanh`. The usual mistake is `W_xh @ x_t`.
2. **`rnn_forward(x, h0, W_xh, W_hh, b)`.** `x` is `(T, n, d_x)`. Return `(T, n, d_h)`. Allocate with `np.empty` and call the cell. The usual mistake is storing `h0` in row 0.
3. **`bptt(dH, x, h0, W_xh, W_hh, b)`.** Rerun the forward pass. Loop `for t in reversed(range(T))` with the carry starting at zeros. The previous state is `h_seq[t - 1]`, or `h0` at `t == 0`. Return `(dx, dW_xh, dW_hh, db, dh0)`, using `+=` and `.sum(axis=0)`. The usual mistake is `h_seq[t]` inside `dW_hh`.
4. **`truncated_bptt(dH, x, h0, W_xh, W_hh, b, k)`.** Forward once. For each chunk `range(0, T, k)`, call `bptt` with the true state before the chunk, add the weight gradients, copy `dx`, and drop `dh0`. Return `(dx, dW_xh, dW_hh, db)`. The usual mistake is restarting from the original `h0`.
5. **`clip_by_global_norm(grads, max_norm)`.** Return a new list, every array times one shared scale. Guard a global norm of zero. The usual mistake is clipping each array on its own.
6. **`gradient_scale_over_time(W, phi_prime_bound, T)`.** Return `(phi_prime_bound * rho) ** T` as a float. The usual mistake is skipping `np.abs`.

## Cost and scaling

With $n$ examples, $T$ steps, input size $d_x$, and hidden size $d_h$, the costs grow as follows:

| Quantity | How it grows |
|---|---|
| Weights | $d_h(d_h + d_x + 1)$, the same for any $T$ |
| Work for one step | about $2n(d_h^2 + d_h d_x)$ operations |
| Work for the sequence | $T$ times one step |
| Memory for full backpropagation | grows with $n T d_h$ |
| Memory for a truncated window | grows with $n k d_h$ |
| Steps that must run in order | $T$, on any hardware |

The weight count does not depend on the sequence length. The work and the full backward memory grow in a straight line with $T$.

## Where this breaks

Each of these failures has a direct change:

- With $\rho(W_{hh}) < 1$, a gradient from 50 steps back is already tiny. Gated cells are the usual fix.
- Clip every step. A sudden huge gradient still shows up when $\rho$ is below one.
- Too small a window $k$ hides long links. Too large a window spends the memory you meant to save.
- Exposure bias grows with output length. One early mistake then has many steps in which to compound.
- The $T$ dependent steps still take $T$ steps of time.

## The exercise

You build `rnn_cell_forward` and `rnn_forward`, then `bptt` and `truncated_bptt`. You add `clip_by_global_norm` and `gradient_scale_over_time`.

The checks match the forward pass to a direct computation, and `bptt` to finite differences for the input, both weights, the bias, and $h_0$. Truncation with `k` equal to the length matches full `bptt`. Truncation with `k` of 2 does not. Clipping leaves a short gradient alone and scales a long one to the limit without turning it. Over 100 steps, with $\rho = 0.9$ and $\rho = 1.05$, the measured `dh0` stays within a factor of ten of the spectral-radius prediction.
