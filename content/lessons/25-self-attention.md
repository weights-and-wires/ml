---
title: Self-attention
slug: 25-self-attention
part: "Part 5 — Transformers"
summary: Scaled dot-product attention derived from the variance of a random dot product, multi-head projection, and causal masking.
minutes: 60
gpu: false
objectives:
  - Derive the 1/sqrt(d_k) scale from the variance of a dot product of independent unit-variance entries.
  - Write multi-head attention as per-head projection and concatenation, and show its flop count does not depend on the number of heads.
  - Explain why causal masking has to happen before the softmax rather than after.
  - State the time and memory cost of self-attention and why it is permutation equivariant.
prereqs:
  - 24-seq2seq-attention
exercise: 25-self-attention
---

# Self-attention

The previous lesson let a decoder query a separate encoder's states. Self-
attention is the special case where the query, the keys, and the values all
come from the same sequence: every position looks at every other position,
including itself, using the same query-key-value machinery, with no encoder or
decoder distinction left at all. That single simplification is what removes
the sequential dependency the last two lessons could not: computing the
attention output for every position uses the same fixed set of keys and
values, so every position's output is computable in parallel once the
projections are done, unlike a recurrent cell whose state at time $t$ requires
the state at time $t-1$ to already exist.

Two problems appear immediately once attention is asked to do this job at
scale. The score $q_i^{\top}k_j$ grows with the dimension of $q$ and $k$ in a
way that pushes the softmax into a regime where it stops being useful, which is
fixed by a single scalar rescaling derived below. And a self-attending sequence
has no notion of order at all — permuting the input permutes the output the
same way — which the mechanism itself cannot fix and later lessons address with
explicit position information. This lesson derives both facts, builds
multi-head attention on top of the single-head mechanism, and works out
exactly what masking future positions costs and why it has to happen before
the softmax rather than after.

## What you already know

- Attention as a query, keys, and values: scores between a query and every key,
  a softmax over those scores, and a context vector as the resulting weighted
  average of the values.
- $\operatorname{Var}(XY) = \mathbb{E}[X^2]\mathbb{E}[Y^2] - \mathbb{E}[X]^2\mathbb{E}[Y]^2$
  for independent $X, Y$, which is $\mathbb{E}[X^2]\mathbb{E}[Y^2]$ when both
  have mean zero.
- The softmax Jacobian is $\operatorname{diag}(p) - pp^{\top}$ for
  $p = \operatorname{softmax}(z)$, with diagonal entries $p_k(1-p_k)$.
- A matrix product $AB$ with $A \in \mathbb{R}^{p \times q}$,
  $B \in \mathbb{R}^{q \times r}$ costs $2pqr$ flops.
- Concatenating vectors is linear: a function applied independently to each
  concatenated block and then stacked is linear in the full input exactly when
  each block's function is linear in that block alone.

## Scaled dot-product attention

For a single query $q \in \mathbb{R}^{d_k}$ against keys $k_1, \dots, k_T \in
\mathbb{R}^{d_k}$ and values $v_1, \dots, v_T \in \mathbb{R}^{d_v}$, stack the
keys and values as rows of $K \in \mathbb{R}^{T \times d_k}$ and
$V \in \mathbb{R}^{T \times d_v}$, and define

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(
\frac{QK^{\top}}{\sqrt{d_k}}
\right) V ,
$$

applying the softmax along each row of $QK^{\top}/\sqrt{d_k}$ so every query's
weights sum to one over the source positions, exactly as in the additive and
multiplicative forms from the previous lesson. Self-attention sets $Q, K, V$ to
three learned linear projections of the same sequence $X \in \mathbb{R}^{T
\times d}$: $Q = XW_Q$, $K = XW_K$, $V = XW_V$.

### Why $1/\sqrt{d_k}$

Take one entry of $QK^{\top}$ before scaling: $q^{\top}k = \sum_{l=1}^{d_k}
q_l k_l$, and suppose the entries of $q$ and $k$ are independent, mean zero,
and unit variance — a reasonable model for freshly initialised projections
of normalised inputs. Each term $q_l k_l$ has

$$
\mathbb{E}[q_l k_l] = \mathbb{E}[q_l]\mathbb{E}[k_l] = 0, \qquad
\operatorname{Var}(q_l k_l) = \mathbb{E}[q_l^2]\mathbb{E}[k_l^2] - 0 = 1 ,
$$

using independence of $q_l$ and $k_l$ for the first equality and their unit
variance for the second. Summing $d_k$ independent, identically distributed
terms adds variances:

$$
\operatorname{Var}(q^{\top}k) = \sum_{l=1}^{d_k} \operatorname{Var}(q_l k_l) = d_k .
$$

The typical magnitude of an unscaled score grows like $\sqrt{d_k}$. Feed scores
of that size into a softmax and it saturates: for two scores separated by a gap
of order $\sqrt{d_k}$, the softmax output is close to one-hot, because
$\operatorname{softmax}$ is shift-invariant but not scale-invariant, and
multiplying every logit by a growing constant sharpens the distribution toward
a point mass. A saturated softmax has Jacobian $\operatorname{diag}(p) -
pp^{\top}$ with every $p_k$ near zero or one, so every entry $p_k(1-p_k)$ is
near zero — the gradient flowing back through the softmax vanishes exactly
where the largest logit differences are, which is precisely the regime large
$d_k$ pushes the network into. Dividing every score by $\sqrt{d_k}$ restores
unit variance regardless of $d_k$,

$$
\operatorname{Var}\!\left(\frac{q^{\top}k}{\sqrt{d_k}}\right)
= \frac{1}{d_k}\operatorname{Var}(q^{\top}k) = 1 ,
$$

keeping the softmax in the regime where its gradient does not vanish by
construction, independent of how wide the attention projections are.

## Multi-head attention

A single attention operation computes one weighted average per query, using
one notion of similarity. Multi-head attention runs $h$ independent copies of
scaled dot-product attention in parallel, each on a lower-dimensional
projection, and lets each head specialise in a different notion of relevance:

$$
\begin{aligned}
\text{head}_i &= \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}), \\
\operatorname{MultiHead}(X) &= \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O ,
\end{aligned}
$$

with each $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_k}$ using
$d_k = d/h$, so the concatenated heads have total width $h \cdot d_k = d$
again, and $W_O \in \mathbb{R}^{d \times d}$ mixes the heads back together.
Implementations fold the $h$ separate projection matrices per role into one
matrix of shape $(d, d)$ and reshape the result into $h$ blocks of width
$d_k$, which is arithmetically identical to running $h$ separate smaller
projections.

### Heads do not change the flop count

Counting the projections: $Q = XW_Q$ costs $2Td^2$ flops regardless of whether
$W_Q$ is one $(d,d)$ matrix or $h$ separate $(d, d_k)$ blocks, because
$h \cdot d \cdot d_k = d \cdot (h \cdot d_k) = d \cdot d = d^2$ either way — the
same identity that keeps the LSTM-versus-GRU parameter ratio exact in lesson
23's arithmetic, applied here to heads instead of gates. The same
holds for $K$, $V$, and the output projection $W_O$: four projections at
$2Td^2$ flops each. Inside the heads, each of the $h$ heads computes
$QK^{\top}$ at cost $2T^2 d_k$ and the weighted sum $\cdot V$ at cost $2T^2
d_k$; summed over $h$ heads with $d_k = d/h$, that totals $h \cdot 4T^2(d/h) =
4T^2 d$, independent of $h$ again. The total flop count,
$8Td^2 + 4T^2 d$, is the same whether attention uses one head or sixteen. What
changes with $h$ is not the compute budget but the representational structure:
sixteen heads of width $d/16$ can each specialise in a narrow notion of
similarity that a single head of width $d$ would have to represent as one
mixture, at identical cost.

## Causal masking

Autoregressive models must not let position $t$ see positions $t+1, \dots, T$,
because at generation time those positions do not exist yet. Enforce this by
adding a mask to the scores before the softmax:

$$
\operatorname{Attention}(Q,K,V,\text{mask}) = \operatorname{softmax}\!\left(
\frac{QK^{\top}}{\sqrt{d_k}} + M
\right) V, \qquad
M_{t,j} = \begin{cases} 0 & j \le t \\ -\infty & j > t \end{cases} .
$$

Adding $-\infty$ before the exponential sends $e^{-\infty} = 0$, so masked
positions receive exactly zero weight and the softmax renormalises over only
the allowed positions $j \le t$, exactly reproducing what attention restricted
to $\{1, \dots, t\}$ from the start would compute. Masking after the softmax
instead — zeroing the disallowed weights post hoc — computes the softmax over
all $T$ positions first, so the denominator $\sum_j e^{z_j}$ already includes
the masked entries; zeroing some weights afterward leaves the surviving weights
summing to less than one, systematically under-weighting every position
$t$ attends to and breaking the promise that attention output is a convex
combination of the values it is allowed to see. Masking has to happen inside
the normalisation, not after it.

## Permutation equivariance

Let $P$ be a $T \times T$ permutation matrix. Replacing $X$ with $PX$ permutes
$Q, K, V$ by the same $P$, since each is a fixed linear map applied row-wise:
$PX W_Q = P(XW_Q)$. Then

$$
(PQ)(PK)^{\top} = PQK^{\top}P^{\top} ,
$$

which permutes both the rows and the columns of the score matrix — row $t$ of
the scores is now what used to be row $P(t)$, scored against keys reordered the
same way. A row-wise softmax commutes with this: permuting a row's entries and
then normalising gives the same distribution, permuted. The final product with
$PV$ then gives $\operatorname{Attention}(PQ, PK, PV) = P\,
\operatorname{Attention}(Q,K,V)$: self-attention applied to a permuted sequence
returns the same outputs, permuted the same way, with no dependence on the
original ordering surviving anywhere in the computation. A set-processing
operation with this property cannot recover which input arrived at which
position, which is why every practical use of self-attention adds positional
information to $X$ before projecting — a fix external to attention itself,
covered in the next lesson.

## What one layer of attention can and cannot express

One layer of self-attention computes, for every position, a single convex
combination of the sequence's value vectors, chosen by a content-based
similarity to that position's query. It can copy, average, and route
information between any two positions in one step, with no distance penalty —
the $O(T^2)$ cost pays for direct access to every pair, not only nearby ones.
It cannot, in one layer, compose two such lookups: attending to "the token two
positions before the token this position matches" requires a lookup whose
result feeds a second lookup, which is exactly what stacking layers is for. Nor
can a single head express a hard, discrete choice — the output is always a
weighted average, never a selection — so tasks that need to count, index by
exact position, or apply strict logic over attended content need either
several layers, positional structure to exploit, or both.

## Cost and scaling

For a sequence of length $T$ and model width $d$ split across $h$ heads:

| Quantity | Cost | Where it shows up |
|---|---|---|
| $Q$, $K$, $V$, $O$ projections | $8Td^2$ flops | Linear in sequence length, quadratic in width |
| Score matrix and weighted sum | $4T^2 d$ flops | Quadratic in sequence length, linear in width |
| Score matrix storage | $O(hT^2)$ memory | One $T \times T$ matrix per head, kept for the backward pass |
| Activations for backpropagation | $O(Td)$ memory | $Q$, $K$, $V$, and the output, per layer |

The $4T^2 d$ term dominates once $T$ exceeds $2d$, so doubling the sequence
length quadruples the dominant cost while doubling the model width only
doubles it — the reverse of a recurrent cell, whose cost is linear in $T$ and
where the sequential dependency, not the flop count, was the binding
constraint. Self-attention pays $O(T^2)$ up front to buy $O(1)$-depth access to
every pair of positions and full parallelism across time.

## Where this breaks

**Quadratic cost at long context.** $T^2$ scaling makes attention the dominant
cost once sequences reach the thousands of tokens, both in compute and in the
memory needed to hold every head's $T \times T$ score matrix for the backward
pass — the reason sparse and linear-time approximations to attention exist,
trading exactness for a computation that scales sub-quadratically.

**No sense of order without help.** Permutation equivariance is exact, so a
bag of positions with no positional signal added is indistinguishable from any
reordering of it. Every architecture built on self-attention has to inject
position from outside the attention mechanism itself.

**One layer, one hop.** Reasoning that requires chaining two attention lookups
needs two layers at minimum, and tasks with a natural sequential or recursive
structure can need a number of layers that scales with the structure's depth,
not with $T$.

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
