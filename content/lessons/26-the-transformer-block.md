---
title: The transformer block
slug: 26-the-transformer-block
part: "Part 5 — Transformers"
summary: Wrap attention and a gated feed-forward network in residuals and layer norm, and teach the block where each token sits.
minutes: 70
gpu: false
objectives:
  - Build a pre-norm transformer block from layer norm, attention, SwiGLU, and two residual adds.
  - Compute sinusoidal position encodings and apply rotary position embeddings to queries and keys.
  - Explain why rotary embeddings make the attention score depend only on the distance between two tokens.
  - Compute the parameter count of a block, including why SwiGLU uses two-thirds of the usual hidden width.
prereqs:
  - 25-self-attention
  - 18-initialisation-and-norm
exercise: 26-the-transformer-block
---

# The transformer block

Self-attention blends tokens. A block also runs a small network on each token alone.

A residual add keeps a path around each sublayer, so a gradient, the slope of the loss, can travel straight back. Layer normalisation holds each token at a steady scale.

Attention still has no order. You mark where each token sits, in two ways, and you build the block.

## What you already know

You already have these tools:

- Multi-head attention with a causal mask, from the previous lesson.
- A residual connection returns `x + f(x)`.
- Layer norm recentres each row and rescales it, from the norm lesson.
- A dot product multiplies matching entries and adds them.
- A softmax turns scores into positive weights that add up to one.

## One row, then a skip

Take the row [1, 2, 3, 6]. The mean is (1 + 2 + 3 + 6) / 4 = 3. The squared gaps are 4, 1, 0, and 9. Their average, the variance, is 14 / 4 = 3.5. The Greek letter epsilon, written ε, is 10^{−5}. The root of 3.5 + ε is 1.8708.

The centred entries divided by that root are (1 − 3) / 1.8708 = −1.0690 and (2 − 3) / 1.8708 = −0.5345. The other two are 0 and (6 − 3) / 1.8708 = 1.6036. With gain 1 and bias 0, that is the normalised row.

This is layer normalisation. The Greek letter mu, written μ, is the row mean. Sigma-squared, σ², is the variance. The gain g and the bias b are learned vectors of length d:

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot g + b
$$

Subtract the mean, divide by the root of variance plus epsilon, then apply g and b. With g = 1 and b = 0 you get the row you computed.

```python
mu = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)
out = (x - mu) / np.sqrt(var + 1e-5) * g + b
```

The usual mistake is dividing the variance by the width minus one. Numpy's default divides by the width.

Suppose attention returns [0.1, 0, 0, −0.1]. Added to the raw row, that is [1.1, 2, 3, 5.9]. The block norms, runs the sublayer, then adds. That order is pre-norm:

$$
X' = X + \operatorname{Attn}(\operatorname{LN}_1(X)), \qquad
X'' = X' + \operatorname{FFN}(\operatorname{LN}_2(X'))
$$

Normalise, run attention, and add to raw X. Normalise again, run the feed-forward net, and add. The residual plug-in is [1.1, 2, 3, 5.9]. If both output matrices are zero, the block returns X.

```python
x1 = x + attention(layer_norm(x, g1, b1))
x2 = x1 + swiglu(layer_norm(x1, g2, b2), W_gate, W_up, W_down)
```

The usual mistake is adding the sublayer to the normalised copy. The skip then disappears.

> [!WARNING]
> Add each sublayer to the raw `x`, not to `layer_norm(x)`. The skip has to bypass the norm.

> [!KEY]
> In a pre-norm block the plain `x +` path carries the gradient straight back, which is what lets you stack many blocks.

## A faucet on the hidden units

A plain feed-forward net is two matrices with a bend between them. SwiGLU adds a third matrix, a faucet on each hidden unit.

The bend is SiLU. The sigmoid squashes a number into 0 to 1, and SiLU multiplies the number by that sigmoid. For 1, e^{−1} = 0.3679, so the sigmoid is 1 / 1.3679 = 0.7311 and SiLU(1) = 0.7311.

Take x = [1, 1], a gate column [1, 0], an up column [0, 2], and a down row [1, −1]. The gate sum is 1 × 1 + 1 × 0 = 1, and SiLU(1) = 0.7311. The up sum is 0 × 1 + 2 × 1 = 2. The product is 0.7311 × 2 = 1.4622. Times [1, −1] the output is [1.4622, −1.4622].

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}}
$$

W_gate and W_up are d by d_h. SiLU hits only the gate. W_down is d_h by d. The same numbers give [1.4622, −1.4622].

```python
gate = x @ W_gate
hidden = gate / (1.0 + np.exp(-gate)) * (x @ W_up)
out = hidden @ W_down
```

The usual mistake is SiLU on the up projection as well as the gate.

A plain net holds 2 d d_ff weights. SwiGLU holds 3 d d_h. Set them equal and d_h is two-thirds of d_ff. With d = 2 and d_ff = 6, both hold 24 weights: 2 × 2 × 6 = 24 and 3 × 2 × 4 = 24. At d_ff = 4d and d = 64, d_h = round(512 / 3) = 171.

$$
d_h = \tfrac{2}{3} d_{\text{ff}}
$$

In the exercise, `d_h = round(2 * d_ff / 3)`.

> [!KEY]
> SwiGLU has three matrices. A hidden width of two-thirds d_ff keeps the same weight count as a plain feed-forward net.

## A clock for each position

The first position mark is a fixed fingerprint added to each token. At position 0 and width 4, every angle is 0, so the row is [0, 1, 0, 1].

The fast frequency is 1. The slow one is 10000^{−1/2} = 0.01. Position 1 is [sin 1, cos 1, sin 0.01, cos 0.01] = [0.8415, 0.5403, 0.0100, 0.99995]. One step turns the fast pair by 1 radian and the slow pair by 0.01, at every position.

The Greek letter omega, written ω, is that frequency. This fingerprint is a sinusoidal position encoding:

$$
\omega_i = 10000^{-2i/d}, \qquad
\operatorname{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad
\operatorname{PE}(p)_{2i+1} = \cos(p\,\omega_i)
$$

For i from 0 to d/2 − 1, ω starts at 1 and shrinks. Even columns are sines and odd columns are cosines. Those two rows are the plug-in.

```python
omega = 10000.0 ** (-2 * np.arange(d // 2) / d)
angles = np.arange(T)[:, None] * omega[None, :]
pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
```

The usual mistake is sines in the first half and cosines in the second half. That layout is for RoPE.

> [!WARNING]
> Sinusoidal encodings interleave sine and cosine. RoPE splits the vector into two halves. Do not mix the layouts.

> [!KEY]
> A shift of k positions turns every sine and cosine pair by k times its frequency, so one linear map works at every position.

## A turn of the query and the key

RoPE turns the queries and the keys with the same clock. The Greek letter theta, written θ, is the base, 10000. Frequencies are θ^{−2j/d}, with d the head width.

Width 2 has one frequency, 1. Turn q = [1, 0] at position 1. cos 1 = 0.5403 and sin 1 = 0.8415, repeated on both halves. Rotate-half of [1, 0] is [0, 1]. The output is [0.5403, 0] + [0, 0.8415] = [0.5403, 0.8415]. The length is 0.5403² + 0.8415² = 1.000.

At position 0 the same vector stays [1, 0], and its dot with the position-1 vector is 0.5403. At position 2 the key is [cos 2, sin 2] = [−0.4161, 0.9093]. Then 0.5403 × (−0.4161) + 0.8415 × 0.9093 = −0.2248 + 0.7652 = 0.5404. The score matches the distance 1.

$$
\operatorname{RoPE}(x) = x \odot \cos + \operatorname{rotate\_half}(x) \odot \sin,
\qquad \operatorname{rotate\_half}(x) = [-x_2,\; x_1]
$$

Each frequency is stored twice, so the tables match the width of x. Turn queries and keys only.

```python
half = x.shape[-1] // 2
rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
out = x * cos + rot * sin
```

The usual mistake is `[x2, −x1]`, which turns the other way and breaks the distance match.

> [!TIP]
> Build the `(T, d_head)` cosine and sine tables once. They broadcast over a `(heads, T, d_head)` stack, so one call turns every head.

> [!KEY]
> RoPE turns each query and each key by its own position. Their dot product then depends only on the distance between the tokens.

A learned table stops at its length. ALiBi stores no vectors and subtracts a penalty that grows with distance.

## How many numbers a block holds

Four attention matrices hold 4d². Two layer norms hold 4d. SwiGLU holds 3 d d_h with d_h = round(2 d_ff / 3). For d = 64 and d_ff = 192, d_h = 128. Then 4 × 4096 + 3 × 64 × 128 + 4 × 64 = 16384 + 24576 + 256 = 41216.

> [!RECAP]
> Carry these facts into the exercise:
> - Pre-norm is `x1 = x + Attn(LN1(x))`, then `x2 = x1 + FFN(LN2(x1))`. Add to the raw `x`.
> - Layer norm runs on the last axis with epsilon 10^{−5}, then applies the gain and the bias.
> - SwiGLU is `(silu(x @ W_gate) * (x @ W_up)) @ W_down`, with SiLU on the gate only.
> - Sinusoidal encodings interleave sine and cosine. RoPE uses rotate-half `[−x2, x1]` on queries and keys only.
> - A block holds 4d² + 3 d d_h + 4d parameters, with d_h rounded from 2 d_ff / 3.

## Build it step by step

The checks test the functions in this order:

1. **`sinusoidal_encoding(T, d)`.** Return `(T, d)`. Build `omega` with `np.arange(d // 2)`, take the outer product with positions, and write `np.sin` into columns `0::2` and `np.cos` into columns `1::2`. The usual mistake is sines in the first half and cosines in the second.
2. **`rope_frequencies(d, theta=10000.0)`.** Return `(d/2,)` as `theta ** (-2 * i / d)`. The usual mistake is dividing by `d // 2` instead of `d`.
3. **`apply_rope(x, cos, sin)`.** Return the same shape as `x`. Split the last axis, build rotate-half as `[−x2, x1]`, and combine. Index with `x[..., :half]` so any leading shape works. The usual mistake is `[x2, −x1]`, which turns the wrong way.
4. **`swiglu(x, W_gate, W_up, W_down)`.** Map `(..., d)` to `(..., d)`. Two products, SiLU on the gate only, an entry-wise product, then `@ W_down`. The usual mistake is SiLU on the up branch too.
5. **`transformer_block_forward(x, params, mask)`.** Map `(T, d)` to `(T, d)`. Norm, project `Wq`, `Wk`, and `Wv`, then `.reshape(T, heads, d_head).transpose(1, 0, 2)`. RoPE on `q` and `k`. Divide by `np.sqrt(d_head)`, add `mask`, softmax, times `v`. Undo the reshape, times `Wo`, add to raw `x`. Second norm, `swiglu`, second add. Use `params.get("theta", 10000.0)`. The usual mistake is the residual on the normalised input.
6. **`block_parameter_count(d_model, d_ff, heads)`.** Return an int: 4d² for attention, 3 d d_h with `d_h = round(2 * d_ff / 3)`, and 4d for the two norms. For d = 64 and d_ff = 192 the total is 41216. The usual mistake is forgetting the norm gain and bias. `heads` must divide d, and it does not otherwise change the count.

## Cost and scaling

Each weight is used once per token in a multiply and an add, so the forward pass costs about two flops per weight per token. With d_ff = 4d the counts are:

| Part | Weights | Forward flops per token |
|---|---|---|
| Attention projections | 4d² | 8d², plus a term that grows with T |
| Feed-forward network | 8d² | 16d² |
| Whole block | 12d² | 24d², plus a term that grows with T |

L blocks and a vocabulary of V tokens hold about V d + 12 L d² weights. For d = 4096, L = 32, and V = 50000, the blocks hold about 6.4 billion and the embedding about 0.2 billion.

## Where this breaks

The block fails in four familiar ways:

- Post-norm puts a norm on the main path, so a deep stack needs a slow start to the learning rate. Pre-norm avoids that.
- In pre-norm the stream grows with depth, because nothing rescales the adds. A final layer norm before the output fixes it.
- RoPE's slow angles barely move in training. A much longer sequence shows angles the block did not train on. Rescale the frequencies to extend context.
- A learned position table has no row past its length. Grow the table and train the extra rows.

## The exercise

You build `sinusoidal_encoding`, `rope_frequencies`, `apply_rope`, `swiglu`, `transformer_block_forward`, and `block_parameter_count`. The block uses RoPE inside attention and SwiGLU as its feed-forward network.

The checks confirm the sine and cosine layout, the fixed-shift linear map, and the RoPE frequencies. They check that `apply_rope` keeps lengths and that the query-key score depends only on distance. They compare `swiglu` with a direct product, check the block's shape, check that zero `Wo` and `W_down` make an identity, and check the parameter count.
