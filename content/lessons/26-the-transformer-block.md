---
title: The transformer block
slug: 26-the-transformer-block
part: "Part 5 — Transformers"
summary: The residual-and-normalisation wrapper that turns attention into a stackable layer, plus four ways to encode position.
minutes: 50
gpu: false
objectives:
  - Explain why a SwiGLU feed-forward network shrinks its hidden width to two-thirds of a plain network's to match its parameter count.
  - Explain the gradient argument for pre-norm over post-norm at depth, and the trade-off it accepts in the residual stream's variance.
  - Derive how rotating queries and keys by a position-dependent angle makes the attention score depend on their offset alone.
  - Compute the parameter count of a transformer block from its width and expansion ratio.
prereqs:
  - 25-self-attention
  - 18-initialisation-and-norm
exercise: 26-the-transformer-block
---

# The transformer block

Self-attention only mixes information across positions. Left on its own it is
linear in the values: drop the softmax's nonlinearity and attention collapses
to a fixed convex combination of value vectors, too weak a hypothesis class to
model language. The transformer block pairs attention with a second sublayer
that transforms each position on its own, then wraps both in residual
connections and normalisation, the machinery that lets you stack dozens of
these pairs without the optimisation collapsing.

None of the three ingredients is specific to attention: the feed-forward
network is the multilayer perceptron from lesson 15, the residual connection
is the identity shortcut behind lesson 21's argument for gradient flow, and
the normalisation is the layer norm from lesson 18. This lesson covers how
they compose, why the composition order matters more at large depth than it
looks like it should, and how the model learns a notion of position, since
attention by itself treats a sequence as an unordered set.

## What you already know

- Scaled dot-product attention, multi-head attention, and causal masking
  (lesson 25).
- Layer norm's forward pass: subtract the mean, divide by the standard
  deviation, then apply a learned per-feature gain and bias (lesson 18).
- A residual connection adds its sublayer's output to its input rather than
  replacing it.
- A Jacobian near the identity keeps gradients from vanishing across many
  compositions.
- A matrix multiply of shapes $(m, k)$ by $(k, n)$ costs $2mnk$ flops.

## The block

Given a sequence $X \in \mathbb{R}^{T \times d_{\text{model}}}$, one block
computes

$$
\begin{aligned}
X' &= X + \operatorname{Attn}(\operatorname{Norm}(X)) \\
X'' &= X' + \operatorname{FFN}(\operatorname{Norm}(X'))
\end{aligned}
$$

This is the **pre-norm** arrangement: normalise, transform, add back. The
feed-forward network acts identically on every row of its input,

$$
\operatorname{FFN}(x) = \sigma(x W_1) W_2, \qquad x \in \mathbb{R}^{d_{\text{model}}},
$$

for an elementwise nonlinearity $\sigma$. It is called position-wise because
$W_1$ and $W_2$ do not depend on the row index: attention already mixed
information across positions, so the feed-forward network spends its capacity
per position instead.

Every widely used architecture sets $d_{\text{ff}} = 4 d_{\text{model}}$.
Attention's four square projections cost $4 d_{\text{model}}^2$ parameters,
and a plain two-matrix feed-forward network at that ratio costs
$2 d_{\text{model}} d_{\text{ff}} = 8 d_{\text{model}}^2$, for a block total
of $12 d_{\text{model}}^2$ — the feed-forward network holds two-thirds of it.
Forward-pass flops track parameters closely, at about two flops per parameter
per token, so the block costs about $24 d_{\text{model}}^2$ flops per token,
twice its parameter count.

### SwiGLU

**SwiGLU** replaces the single nonlinearity with a gate. It multiplies the
silu ($x \cdot \operatorname{sigmoid}(x)$) of one projection by a second
projection, then applies a third matrix to the result,

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big) W_{\text{down}},
$$

with $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_h}$
and $W_{\text{down}} \in \mathbb{R}^{d_h \times d_{\text{model}}}$. Three
matrices of width $d_h$ cost $3 d_{\text{model}} d_h$ parameters, against
$2 d_{\text{model}} d_{\text{ff}}$ for the plain two-matrix network. Matching
that budget means solving $3 d_h = 2 d_{\text{ff}}$, so

$$
d_h = \tfrac{2}{3} d_{\text{ff}} .
$$

Shrink the gated network's hidden width to two-thirds of the plain network's
and it costs the same parameters and, because flops track parameters
linearly, the same flops. This is why a SwiGLU model's hidden dimension in a
configuration file looks like an odd number: it is two-thirds of
$4 d_{\text{model}}$, occasionally rounded to a hardware-friendly multiple.

## Pre-norm against post-norm

The original transformer normalised after the residual add:
$X' = \operatorname{Norm}(X + \operatorname{Attn}(X))$. Backpropagating the
loss through $L$ stacked post-norm blocks passes every layer's gradient
through that block's normalisation Jacobian first, and layer norm's Jacobian
has no reason to sit near the identity. Stack $L$ of them and nothing
guarantees the product of those Jacobians does not shrink or grow
geometrically with depth.

Pre-norm removes normalisation from the shortcut path entirely.
Differentiating $X' = X + \operatorname{Attn}(\operatorname{Norm}(X))$ gives

$$
\frac{\partial X'}{\partial X} = I + \frac{\partial \operatorname{Attn}(\operatorname{Norm}(X))}{\partial X} .
$$

The identity term is exact and depth-independent: whatever the sublayer's
Jacobian does, an unobstructed path back to every earlier layer survives,
because $I$ appears as an additive term rather than as one factor in a
product. Stack $L$ blocks and the gradient with respect to the input still
contains $I$ as one term in its expansion, so it cannot vanish with depth by
construction. This is the same shortcut argument lesson 21 makes for residual
networks, and it is why every transformer built past a few dozen layers uses
pre-norm.

The trade shows up in the forward pass. Normalisation no longer touches the
residual stream itself, only the copy each sublayer reads, so if each
sublayer's output adds roughly independent, zero-mean noise of variance $v$ to
the stream, that variance accumulates linearly with depth:
$\operatorname{Var}(X_l) \approx \operatorname{Var}(X_0) + l v$. Post-norm
resets the variance to one at every block and never accumulates it. Pre-norm
accepts the growth because it is solvable — a final layer norm before the
output head undoes it in one step — while a gradient that vanishes
geometrically with depth is not solvable by rescaling anything downstream.

## Telling the model where a token is

Attention computes $\operatorname{softmax}(QK^\top / \sqrt{d_k}) V$ from sets
of vectors. Permute the rows of $Q$, $K$, and $V$ together and the output
permutes the same way, so attention has no notion of order unless you add
one.

**Sinusoidal encodings.** The original scheme adds a fixed vector to each
token embedding, built from sinusoids at geometrically spaced frequencies
$\omega_i = \theta^{-2i/d}$ for $i = 0, \dots, d/2 - 1$ and a base $\theta$
(conventionally $10^4$):

$$
\operatorname{PE}(p)_{2i} = \sin(p\, \omega_i), \qquad
\operatorname{PE}(p)_{2i+1} = \cos(p\, \omega_i).
$$

For a fixed offset $k$, $\operatorname{PE}(p+k)$ works out to a fixed rotation
of $\operatorname{PE}(p)$ by an angle $k \omega_i$ in each frequency band, the
same rotation for every $p$. That gives a linear map from any position's
encoding to the encoding $k$ steps later, which is what lets a linear layer
learn to attend by relative offset from an absolute encoding.

**Learned absolute embeddings.** Replace the fixed sinusoids with a trainable
table $E \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}$ and add row $p$ to
the embedding at position $p$. This costs $T_{\max} d_{\text{model}}$
parameters, carries no relative structure, and produces nothing for positions
past $T_{\max}$ without extending the table and retraining.

**Rotary position embeddings.** RoPE applies the sinusoidal idea to the query
and key vectors instead of the input embedding. Split a query $q \in
\mathbb{R}^d$ into $d/2$ pairs and rotate pair $i$ by angle $m \omega_i$ at
position $m$, using the same $\omega_i$ as above. Write that per-pair
rotation as an orthogonal block $R_i(m)$ and the whole transform as the
block-diagonal $R(m)$. Rotation matrices compose by adding angles and
transpose by negating them, so $R(m)^\top R(n) = R(n - m)$, and the score
between a rotated query at $m$ and a rotated key at $n$ works out to

$$
\operatorname{RoPE}(q, m)^\top \operatorname{RoPE}(k, n)
= q^\top R(m)^\top R(n)\, k = q^\top R(n - m)\, k,
$$

a function of the offset $n - m$ alone, for any $m$ and $n$. Absolute position
never reaches the score; only the distance between the two tokens does, and
because rotation cannot change a vector's norm, RoPE leaves $\lVert q \rVert$
and $\lVert k \rVert$ untouched. This is the derivation the exercise's checks
verify directly.

**ALiBi.** ALiBi drops position vectors and instead subtracts a distance
penalty from the raw score before the softmax:
$\operatorname{score}_{ij} \mathrel{-}= m_h \, (i - j)$ for a causal query at
$i$ attending to key $j \le i$, with a head-specific slope $m_h$ fixed at
initialisation. It adds no parameters, and because the penalty is linear in
distance rather than in an absolute position, it extrapolates cleanly to
sequences longer than training ever saw, unlike RoPE.

## Cost and scaling

For a block of width $d_{\text{model}}$ and feed-forward ratio 4, ignoring
the smaller normalisation parameters, the following costs apply.

| Sublayer | Parameters | Forward flops per token |
|---|---|---|
| Attention (4 projections) | $4 d_{\text{model}}^2$ | $8 d_{\text{model}}^2 + O(T d_{\text{model}})$ |
| Feed-forward (ratio 4, gated or not) | $8 d_{\text{model}}^2$ | $16 d_{\text{model}}^2$ |
| Block total | $12 d_{\text{model}}^2$ | $24 d_{\text{model}}^2 + O(T d_{\text{model}})$ |

A full model of $L$ blocks and vocabulary $V$, tying the input embedding to
the output projection, costs about

$$
N \approx V d_{\text{model}} + 12 L d_{\text{model}}^2 ,
$$

parameters — the embedding once, plus twelve times the block count times the
width squared. For $d_{\text{model}} = 4096$, $L = 32$, and $V = 50000$, the
embedding is about $0.2$ billion parameters against
$12 \times 32 \times 4096^2 \approx 6.4$ billion for the blocks, so scaling a
transformer is overwhelmingly a question of $L$ and $d_{\text{model}}$.

## Where this breaks

**Deep post-norm without care.** Very deep post-norm stacks need a long
learning-rate warmup and careful initial scaling of the residual branches to
train at all. Pre-norm needs far less of that, which is why it is the default
past a few dozen layers.

**RoPE and context extension.** The relative-offset guarantee holds for any
$m$ and $n$, but the frequencies $\omega_i$ are fixed at training time.
Evaluate at offsets much larger than training ever used and the lowest
frequencies have barely completed a fraction of a cycle, so RoPE struggles to
extrapolate to context lengths far past training — a problem frequency-rescaling
methods address outside this lesson.

**Learned absolute embeddings and length.** A learned table has no value to
return past $T_{\max}$; extending context length means growing the table and
retraining those rows from nothing.

**Residual-stream capacity.** Every sublayer, at every depth, reads and
writes the same fixed-width $d_{\text{model}}$ channel. As $L$ grows for
fixed $d_{\text{model}}$, more sublayers compete for that channel, and
interpretability studies find individual directions in the stream carrying
more than one feature as a result.

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
