---
title: Self-attention
slug: 25-self-attention
part: "Part 5 — Transformers"
summary: Scaled dot-product attention, multi-head projection, and causal masking, with the reasoning behind each.
minutes: 45
gpu: false
objectives:
  - Explain why dividing by the square root of the key dimension keeps attention scores from saturating the softmax.
  - Write multi-head attention as per-head projection and concatenation, and explain why splitting into heads leaves the flop count unchanged.
  - Explain why causal masking must happen before the softmax rather than after.
  - State the time and memory cost of self-attention and why it treats a sequence as an unordered set.
prereqs:
  - 24-seq2seq-attention
exercise: 25-self-attention
---

# Self-attention

The previous lesson let a decoder query a separate encoder's states. Self-attention
removes that separation: the query, the keys, and the values all come from the
same sequence, so every position can look at every other position, including
itself, using the same machinery. That single change removes the sequential
dependency of a recurrent cell. A recurrent cell's state at time $t$ needs the
state at time $t-1$ to exist first, but self-attention computes every
position's output from the same fixed set of keys and values, so every
position's output is computable in parallel once the projections are done.

Running attention over a whole sequence exposes two problems. The score
$q_i^\top k_j$ grows with the dimension of $q$ and $k$, and an oversized score
pushes the softmax toward a near one-hot output with no useful gradient — one
rescaling fixes it. And a self-attending sequence carries no notion of order:
permute the input and the output permutes the same way, a limitation the
mechanism itself cannot fix. This lesson derives the scale, builds multi-head
attention on top of single-head attention, and works out what masking future
positions costs.

## What you already know

- Attention as a query, keys, and values: scores between a query and every
  key, a softmax over those scores, and a context vector as the resulting
  weighted average of the values.
- For independent, mean-zero random variables $X$ and $Y$,
  $\operatorname{Var}(XY) = \mathbb{E}[X^2]\mathbb{E}[Y^2]$.
- The softmax Jacobian is $\operatorname{diag}(p) - pp^{\top}$ for
  $p = \operatorname{softmax}(z)$.
- A matrix product $AB$ with $A \in \mathbb{R}^{p \times q}$,
  $B \in \mathbb{R}^{q \times r}$ costs $2pqr$ flops.
- Concatenating vectors is linear whenever each concatenated block's own
  transform is linear.

## Scaled dot-product attention

For a query $q \in \mathbb{R}^{d_k}$ against keys $k_1, \dots, k_T \in
\mathbb{R}^{d_k}$ and values $v_1, \dots, v_T \in \mathbb{R}^{d_v}$, stack the
keys and values as rows of $K \in \mathbb{R}^{T \times d_k}$ and
$V \in \mathbb{R}^{T \times d_v}$, and define

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(
\frac{QK^{\top}}{\sqrt{d_k}}
\right) V ,
$$

applying the softmax along each row so every query's weights sum to one.
Self-attention sets $Q$, $K$, and $V$ to three learned linear projections of
the same sequence $X \in \mathbb{R}^{T \times d}$: $Q = XW_Q$, $K = XW_K$,
$V = XW_V$.

The $\sqrt{d_k}$ divides out a size effect, not a modelling choice. Suppose
the entries of $q$ and $k$ are independent, mean zero, and unit variance, a
reasonable model for freshly initialised projections of normalised inputs.
Each of the $d_k$ terms in $q^{\top}k = \sum_l q_l k_l$ then has variance one,
so the dot product itself has variance $d_k$, and its typical magnitude grows
with $\sqrt{d_k}$. Feed scores that large into a softmax and the output
saturates near a one-hot vector, where the softmax Jacobian
$\operatorname{diag}(p) - pp^{\top}$ is close to zero and gradients stop
flowing. Dividing every score by $\sqrt{d_k}$ restores unit variance
regardless of $d_k$, which keeps the softmax's gradient away from zero no
matter how wide the projections are.

## Multi-head attention

A single attention operation computes one weighted average per query, using
one notion of similarity. Multi-head attention runs $h$ independent copies of
scaled dot-product attention on lower-dimensional projections, letting each
head specialise:

$$
\begin{aligned}
\text{head}_i &= \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}), \\
\operatorname{MultiHead}(X) &= \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O ,
\end{aligned}
$$

with each $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_k}$ using
$d_k = d/h$, so the concatenated heads have total width $h d_k = d$ again,
and $W_O \in \mathbb{R}^{d \times d}$ mixes the heads back together. An
implementation folds the $h$ per-role projections into one $(d, d)$ matrix and
reshapes the result into $h$ blocks of width $d_k$, which is the identical
arithmetic to running the heads separately.

Splitting one wide head into $h$ narrow ones changes nothing about the
compute budget: $h \cdot d \cdot d_k = d^2$ regardless of $h$, so the four
projections still cost $8Td^2$ flops between them, and the per-head score
computation still totals $4T^2d$ flops summed over every head. Doubling the
head count buys each head a narrower, more specialised notion of similarity
at no extra cost — only the representation changes, never the flop count.

## Causal masking

Autoregressive models must not let position $t$ see positions
$t+1, \dots, T$, because those positions do not exist yet at generation time.
Add a mask to the scores before the softmax:

$$
\operatorname{Attention}(Q,K,V,\text{mask}) = \operatorname{softmax}\!\left(
\frac{QK^{\top}}{\sqrt{d_k}} + M
\right) V, \qquad
M_{t,j} = \begin{cases} 0 & j \le t \\ -\infty & j > t \end{cases} .
$$

Adding $-\infty$ before the exponential sends $e^{-\infty} = 0$, so the
softmax renormalises over only the allowed positions. Masking has to happen
before the softmax, not after: the softmax's denominator would otherwise sum
over every position, including the ones a later step zeros out, leaving the
surviving weights summing to less than one and understating every attended
position's true share.

## Order and permutation equivariance

Attention has no notion of order built in. Permute the rows of $X$ and the
same permutation $P$ carries through to $Q$, $K$, and $V$, because each is a
row-wise linear map. A row-wise softmax commutes with a row permutation, so
$\operatorname{Attention}(PQ, PK, PV) = P \cdot \operatorname{Attention}(Q,K,V)$:
self-attention applied to a reordered sequence returns the same outputs,
reordered the same way, with no trace of the original order surviving
anywhere in the computation. A mechanism with this property cannot recover
which input arrived at which position on its own, which is why every
practical use of self-attention adds positional information to $X$ before
projecting, the subject of the next lesson.

## Cost and scaling

For a sequence of length $T$ and model width $d$ split across $h$ heads, the
following costs apply.

| Quantity | Cost | Where it shows up |
|---|---|---|
| $Q$, $K$, $V$, $O$ projections | $8Td^2$ flops | Linear in sequence length, quadratic in width |
| Score matrix and weighted sum | $4T^2 d$ flops | Quadratic in sequence length, linear in width |
| Score matrix storage | $O(hT^2)$ memory | One $T \times T$ matrix per head, kept for the backward pass |
| Activations for backpropagation | $O(Td)$ memory | $Q$, $K$, $V$, and the output, per layer |

The $4T^2 d$ term dominates once $T$ exceeds $2d$, so doubling the sequence
length quadruples the dominant cost while doubling the model width only
doubles it — the reverse of a recurrent cell, whose cost is linear in $T$ but
whose sequential dependency was the real bottleneck. Self-attention trades an
$O(T^2)$ compute bill for direct, one-step access between every pair of
positions and full parallelism across time.

## Where this breaks

**Quadratic cost at long context.** The $T^2$ term makes attention the
dominant cost once sequences reach the thousands of tokens, both in compute
and in the memory needed to hold every head's $T \times T$ score matrix for
the backward pass. Sparse and linear-time approximations to attention trade
exactness for sub-quadratic scaling.

**No sense of order without help.** Permutation equivariance is exact, so a
sequence with no positional signal added is indistinguishable from any
reordering of it. Every architecture built on self-attention injects
position from outside the attention mechanism itself.

**One layer, one hop.** A single layer computes one convex combination of
the value vectors per position, chosen by a content-based similarity to that
position's query. It copies, averages, and routes information between any
two positions in one step, but it cannot compose two such lookups on its own:
attending to "the token before the token this position matches" needs the
result of one lookup to feed a second, which is exactly what stacking layers
provides.

## The exercise

You implement scaled dot-product attention, multi-head attention, and the
supporting pieces from this lesson. Write `softmax(x, axis=-1)`, a numerically
stable softmax along an arbitrary axis, and
`scaled_dot_product_attention(q, k, v, mask=None)`, applying an additive mask
before the softmax when one is given. Write `multi_head_attention(x, Wq, Wk,
Wv, Wo, heads, mask=None)` for a single sequence `x`, splitting the projected
`Q`, `K`, `V` into `heads` equal blocks along their last axis before running
attention per head and recombining. Write `causal_mask(T)`, returning the
$T \times T$ additive mask from this lesson, and `attention_flops(T, d,
heads)`, returning the hand-computed flop count $8Td^2 + 4T^2d$. Finally write
`attention_backward(dout, q, k, v, mask=None)` for the single-head case,
returning the gradient with respect to `q`, `k`, and `v`.

The checks verify that `softmax` stays finite and sums to one at logits large
enough to overflow a naive implementation, that the causal mask makes a given
position's output unchanged when a later position in the input is perturbed,
that one-head multi-head attention exactly reproduces single-head attention,
that permuting the rows of the input permutes the rows of the output
identically, that `attention_backward` matches `mlcheck.finite_difference`, and
that `attention_flops` matches the hand-computed value and stays unchanged
across different head counts at fixed `T` and `d`.
