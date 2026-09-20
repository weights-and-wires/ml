---
title: The transformer block
slug: 26-the-transformer-block
part: "Part 5 — Transformers"
summary: The residual-and-normalisation wrapper that turns attention into a stackable layer, and four ways to tell a token where it sits.
minutes: 60
gpu: false
objectives:
  - Derive why a SwiGLU feed-forward network scales its hidden width by two-thirds to match an ungated network's parameter count.
  - Explain the gradient argument for pre-norm over post-norm at depth, and the residual-stream variance growth pre-norm accepts in exchange.
  - Derive how rotating queries and keys by a position-dependent angle makes the attention score a function of the offset alone.
  - Compute the parameter count of a transformer block from its width and expansion ratio.
prereqs:
  - 25-self-attention
  - 18-initialisation-and-norm
exercise: 26-the-transformer-block
---

# The transformer block

Self-attention mixes information across positions, but on its own it is a
linear operation in the values: drop the softmax's nonlinearity and attention
collapses to a fixed convex combination of value vectors, which is far too weak
a hypothesis class to model language. The transformer block fixes this by
pairing attention with a second sublayer that transforms each position on its
own, and wrapping both in the machinery — residual connections and
normalisation — that lets you stack dozens of these pairs without the
optimisation collapsing.

None of the three ingredients is specific to attention. The feed-forward
network is the multilayer perceptron from lesson 15. The residual connection is
the identity shortcut you have not yet met by that name but will recognise
from lesson 21's argument for gradient flow. The normalisation is the layer
norm derived in lesson 18. What is new here is how they compose, why the
composition order matters more at large depth than it looks like it should,
and how the model learns a notion of order at all, since attention by itself
is permutation equivariant and blind to it.

## What you already know

- Scaled dot-product attention, multi-head attention, and causal masking
  (lesson 25).
- Layer norm's forward pass: subtract the mean, divide by the standard
  deviation, then apply a learned per-feature gain and bias (lesson 18).
- A residual connection adds its sublayer's output to its input rather than
  replacing it.
- The chain rule for a composition of functions, and how a Jacobian near the
  identity keeps gradients from vanishing across many compositions.
- A matrix multiply of shapes $(m, k)$ by $(k, n)$ costs $2mnk$ flops.

## The block

Given a sequence represented as $X \in \mathbb{R}^{T \times d_{\text{model}}}$,
one block computes

$$
\begin{aligned}
X' &= X + \operatorname{Attn}(\operatorname{Norm}(X)) \\
X'' &= X' + \operatorname{FFN}(\operatorname{Norm}(X'))
\end{aligned}
$$

This is the **pre-norm** arrangement: normalise, transform, add back. The
feed-forward network acts identically and independently on every row of its
input,

$$
\operatorname{FFN}(x) = \sigma(x W_1) W_2, \qquad x \in \mathbb{R}^{d_{\text{model}}},
$$

with $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $\sigma$ an
elementwise nonlinearity. It is "position-wise" precisely because $W_1$ and
$W_2$ do not depend on the row index: whatever mixing across positions the
block performs, attention already did it, and the feed-forward network is
free to spend its capacity per position.

### The expansion ratio

Every widely used architecture sets $d_{\text{ff}} = 4 d_{\text{model}}$. Attention's
four square projections — query, key, value, and output — cost
$4 d_{\text{model}}^2$ parameters (ignoring biases, which later architectures
drop). The feed-forward network costs $2 d_{\text{model}} d_{\text{ff}}$. With
the ratio of four that is $8 d_{\text{model}}^2$, so the block totals

$$
4 d_{\text{model}}^2 + 8 d_{\text{model}}^2 = 12 d_{\text{model}}^2,
$$

and the feed-forward network holds $8/12 = 2/3$ of it. The forward-pass flop
count tracks the parameter count almost exactly: each parameter enters one
multiply and one add per token it touches, so a matmul of $P$ parameters costs
$2P$ flops per token, plus attention's context term. Summing both sublayers,

$$
\underbrace{8 d_{\text{model}}^2}_{\text{attention proj.}}
+ \underbrace{O(T d_{\text{model}})}_{\text{attention scores}}
+ \underbrace{16 d_{\text{model}}^2}_{\text{feed-forward}}
\;=\; 24 d_{\text{model}}^2 + O(T d_{\text{model}})
$$

flops per token — twice the block's parameter count, plus a term that only
matters once the context length $T$ rivals $d_{\text{model}}$. That factor of
two per parameter is the same rule lesson 27 extends to a full training run.

### Nonlinearities: ReLU, GELU, SwiGLU

The original feed-forward network used $\sigma = \operatorname{ReLU}$. GELU,
$x \Phi(x)$ for $\Phi$ the standard normal CDF, replaces the hard cutoff with a
smooth one and is the more common default; both are single-input
nonlinearities applied after one matrix.

**SwiGLU** is different in kind: it gates one linear projection with the
silu ($x \cdot \operatorname{sigmoid}(x)$) of another, and needs a third matrix
for the output,

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big) W_{\text{down}},
$$

with $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_h}$
and $W_{\text{down}} \in \mathbb{R}^{d_h \times d_{\text{model}}}$. Three
matrices of width $d_h$ cost $3 d_{\text{model}} d_h$ parameters, against
$2 d_{\text{model}} d_{\text{ff}}$ for the two-matrix design. Matching that
budget means solving $3 d_h = 2 d_{\text{ff}}$, giving

$$
d_h = \tfrac{2}{3} d_{\text{ff}} .
$$

Shrink the gated network's hidden width by two-thirds and it costs the same
parameters, and — because flops track parameters linearly for a matmul — the
same flops, as the plain two-matrix network it replaces. This is why a
SwiGLU model's hidden dimension in configuration files looks like an odd
number: it is $\tfrac{2}{3}$ of $4 d_{\text{model}}$, occasionally rounded to a
multiple of a hardware-friendly constant.

## Pre-norm against post-norm

The original transformer normalised after the residual add:
$X' = \operatorname{Norm}(X + \operatorname{Attn}(X))$. Follow the gradient of
the final loss $L$ backward through $L$ stacked post-norm blocks and every
layer's contribution passes through that block's normalisation Jacobian before
it reaches the one below. Layer norm's backward pass divides by the input's
standard deviation and subtracts a mean-and-projection term; its Jacobian has
no reason to sit near the identity, and stacking $L$ of them multiplies $L$
such factors together. Nothing guarantees the product does not shrink or grow
geometrically in $L$.

Pre-norm removes the normalisation from the shortcut entirely. From
$X' = X + \operatorname{Attn}(\operatorname{Norm}(X))$,

$$
\frac{\partial X'}{\partial X} = I + \frac{\partial \operatorname{Attn}(\operatorname{Norm}(X))}{\partial X} .
$$

The identity term is exact and depth-independent: whatever the sublayer's
Jacobian does, the gradient flowing backward always has an unobstructed
identity path back to every earlier layer, because $\partial X' / \partial X$
contains $I$ as an additive term rather than as one factor in a product. Stack
$L$ blocks and the gradient with respect to $X_0$ still contains $I$ as one
term in its expansion — it cannot vanish with depth by construction. This is
the same shortcut argument lesson 21 makes for residual networks, and it is
why every transformer built past a few dozen layers uses pre-norm.

The trade is in the forward pass. Normalisation no longer touches the residual
stream itself, only the copy each sublayer reads. If each sublayer's output
$\operatorname{Attn}(\operatorname{Norm}(X_l))$ contributes roughly independent,
zero-mean noise of variance $v$ to the stream regardless of the stream's
current scale — plausible, since layer norm resets the scale of what the
sublayer reads every time — then

$$
\operatorname{Var}(X_l) \approx \operatorname{Var}(X_0) + l v ,
$$

growing linearly in depth $l$. Post-norm resets the variance to one at every
block by construction and never accumulates it. Pre-norm accepts that growth
because a linearly growing stream is a solvable problem — a final layer norm
before the output head undoes it in one step — while a gradient that vanishes
geometrically with depth is not solvable by rescaling anything downstream.

## The residual stream as a shared channel

Write the pre-norm recursion as $X_l = X_0 + \sum_{i<l} f_i(\operatorname{Norm}(X_i))$
for the $i$-th sublayer's function $f_i$. Every sublayer, attention or
feed-forward, reads the entire accumulated stream through its own normalisation
and writes an additive update back into the same $d_{\text{model}}$-dimensional
space that every other sublayer, at every other depth, also reads and writes.
No sublayer owns a private subspace; the split into "what attention wrote" and
"what the feed-forward network wrote" is not visible in $X_l$ itself, only in
the sum. This is why interpretability work treats the residual stream as a
communication bus of fixed bandwidth $d_{\text{model}}$ shared across
$2L$ writers (two sublayers per block, $L$ blocks): a wider stream is not just
a more expressive representation, it is more channel capacity for every later
layer to read what an earlier one wrote.

## Telling the model where a token is

Attention computes $\operatorname{softmax}(QK^\top / \sqrt{d_k}) V$ from sets of
query, key, and value vectors. Permute the rows of $Q$, $K$, and $V$ together
and the output permutes the same way — attention has no notion of order unless
you give it one.

### Sinusoidal encodings

The original scheme adds a fixed vector to each token embedding, built from
sinusoids at geometrically spaced frequencies. With
$\omega_i = \theta^{-2i/d}$ for $i = 0, \dots, d/2 - 1$ and a base $\theta$
(conventionally $10^4$), position $p$ gets

$$
\operatorname{PE}(p)_{2i} = \sin(p\, \omega_i), \qquad
\operatorname{PE}(p)_{2i+1} = \cos(p\, \omega_i).
$$

Fix an offset $k$ and expand with the angle-addition formulas:

$$
\begin{pmatrix} \sin((p+k)\omega_i) \\ \cos((p+k)\omega_i) \end{pmatrix}
=
\begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix}
\begin{pmatrix} \sin(p\omega_i) \\ \cos(p\omega_i) \end{pmatrix} .
$$

Each frequency band of $\operatorname{PE}(p+k)$ is a rotation of the
corresponding band of $\operatorname{PE}(p)$ by an angle $k \omega_i$ that does
not depend on $p$. Stacking the bands, $\operatorname{PE}(p+k) = M_k \operatorname{PE}(p)$
for a block-diagonal orthogonal matrix $M_k$ fixed across all $p$ — a linear
map from any position's encoding to the encoding $k$ steps later, which is
exactly the property that lets a linear layer learn to attend by relative
offset from an absolute encoding.

### Learned absolute embeddings

Replace the fixed sinusoids with a trainable table
$E \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}$ and add row $p$ to the
embedding at position $p$. This costs $T_{\max} d_{\text{model}}$ parameters,
has no built-in relative structure, and produces embeddings for positions past
$T_{\max}$ only if you extend the table and retrain — it does not extrapolate.

### Rotary position embeddings

RoPE applies the sinusoidal idea directly to the query and key vectors instead
of to the input embedding. Split a query $q \in \mathbb{R}^d$ into $d/2$ pairs
$(q_i, q_{i+d/2})$ and rotate pair $i$ by angle $m \omega_i$ at position $m$:

$$
\operatorname{RoPE}(q, m)_i =
\begin{pmatrix} q_i \\ q_{i + d/2} \end{pmatrix}
\text{ rotated by } m\omega_i
=
\begin{pmatrix} q_i \cos(m\omega_i) - q_{i+d/2}\sin(m\omega_i) \\
q_{i+d/2}\cos(m\omega_i) + q_i \sin(m\omega_i) \end{pmatrix} .
$$

Write this per-pair rotation as an orthogonal block $R_i(m) \in \mathbb{R}^{2\times 2}$
and the whole vector's transform as the block-diagonal $R(m)$. Two facts follow
immediately from $R(m)$ being a rotation. First, $R(m)^\top R(m) = I$, so
$\lVert \operatorname{RoPE}(q, m) \rVert = \lVert q \rVert$: rotation cannot
change a vector's norm. Second, rotations compose by adding angles and
transpose by negating them, so $R(m)^\top R(n) = R(n - m)$, and the attention
score between a rotated query at position $m$ and a rotated key at position
$n$ is

$$
\operatorname{RoPE}(q, m)^\top \operatorname{RoPE}(k, n)
= q^\top R(m)^\top R(n)\, k = q^\top R(n - m)\, k,
$$

a function of the offset $n - m$ alone, for any $m$ and $n$. Absolute position
never appears in the score; only the distance between the two tokens does,
recovered without ever materialising a relative-position table. This is the
derivation the exercise's checks verify directly.

### ALiBi

ALiBi drops position vectors entirely and instead subtracts a penalty from the
raw attention score before the softmax, linear in the distance between
positions: $\operatorname{score}_{ij} \mathrel{-}= m_h \, (i - j)$ for a
causal query at $i$ attending to key $j \le i$, with a head-specific slope
$m_h$ fixed at initialisation (typically a geometric sequence over heads). It
adds no parameters, needs no rotation, and — because the penalty is linear
in distance rather than encoding an absolute position — extrapolates cleanly
to sequences longer than any seen in training, which is the property RoPE
lacks without further modification.

## Cost and scaling

For a block of width $d_{\text{model}}$ and feed-forward ratio $4$, ignoring
the low-order normalisation parameters:

| Sublayer | Parameters | Forward flops per token |
|---|---|---|
| Attention (4 projections) | $4 d_{\text{model}}^2$ | $8 d_{\text{model}}^2 + O(T d_{\text{model}})$ |
| Feed-forward (ratio 4, gated or not) | $8 d_{\text{model}}^2$ | $16 d_{\text{model}}^2$ |
| Block total | $12 d_{\text{model}}^2$ | $24 d_{\text{model}}^2 + O(T d_{\text{model}})$ |

A full model of $L$ blocks and vocabulary $V$, tying the input embedding to
the output projection, costs

$$
N \approx V d_{\text{model}} + 12 L d_{\text{model}}^2 ,
$$

parameters — the embedding once, plus twelve times the block count times the
width squared. For $d_{\text{model}} = 4096$, $L = 32$, $V = 50000$: the
embedding is about $0.2$ billion parameters and the blocks are
$12 \times 32 \times 4096^2 \approx 6.4$ billion, so the blocks dominate by more
than thirty to one, which is why scaling a transformer is overwhelmingly a
question of $L$ and $d_{\text{model}}$ and barely one of vocabulary size.

## Where this breaks

**Deep post-norm without care.** Very deep post-norm stacks are trainable only
with a long learning-rate warmup and careful initial scaling of the residual
branches; without it the gradient product across layers is unstable in either
direction. Pre-norm needs far less of this and is the default past a few dozen
layers for exactly that reason.

**RoPE and context extension.** The relative-offset guarantee holds for any
$m, n$, but the frequencies $\omega_i$ are fixed at training time. Evaluate at
offsets much larger than anything seen in training and the lowest frequencies
have barely completed a fraction of a cycle, so the model has never learned
what that portion of the rotation means — the cause of RoPE's well-documented
difficulty extrapolating to context lengths far past training, addressed by
frequency rescaling methods that lie outside this lesson.

**Learned absolute embeddings and length.** A learned table has no value to
return for a position past $T_{\max}$; extending context length means growing
the table and retraining those new rows from nothing.

**Residual-stream capacity.** As $L$ grows for fixed $d_{\text{model}}$, more
sublayers write into the same fixed-width channel. Interpretability studies of
large models find directions in the stream carrying multiple superposed
features precisely because the channel is narrower than the number of
distinct things every layer would like to write there.

## The exercise

You build one full pre-norm transformer block from its parts. Implement
`sinusoidal_encoding(T, d)`, `rope_frequencies(d, theta)`, and
`apply_rope(x, cos, sin)` for the rotate-half convention, then `swiglu(x,
W_gate, W_up, W_down)`. Combine an attention implementation you write in the
same file with these pieces in `transformer_block_forward(x, params, mask)`,
which applies RoPE inside attention and SwiGLU inside the feed-forward
sublayer, both in the pre-norm arrangement. Finish with
`block_parameter_count(d_model, d_ff, heads)`, returning the exact parameter
count for a block whose feed-forward network is SwiGLU with the width adjusted
to match a plain two-matrix network of hidden size `d_ff`.

The checks verify each piece before the whole: that the sinusoidal encoding at
offset $k$ is the fixed linear map of the encoding at $p$ derived above, that
`apply_rope` preserves vector norms and makes the query-key score depend on
$m - n$ alone across several pairs at the same offset, that
`transformer_block_forward` preserves its input's shape, that zeroing the
attention and feed-forward output projections makes the block return its input
exactly, and that `block_parameter_count` matches a hand computation.
