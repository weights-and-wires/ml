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

Self-attention lets tokens share information. On its own, though, it can only
blend value vectors together. It never transforms any single token in a rich,
nonlinear way. A transformer block pairs attention with a small neural network
that works on each token separately. Think of a meeting: first everyone talks
(attention), then each person goes back to their desk and thinks (the
feed-forward network).

Two more pieces make these blocks stackable. A residual connection adds each
step's output to its input, so information and gradients have a direct path
through the stack. Layer norm keeps each token's numbers at a steady scale.
With these, you can stack dozens of blocks and still train them.

The last piece is position. The previous lesson showed that attention can't
tell word order. This lesson gives you two ways to fix that, and you build
both.

## What you already know

- Multi-head attention with a causal mask, from lesson 25.
- Layer norm subtracts each row's mean, divides by its standard deviation, and
  applies a learned gain and bias, from lesson 18.
- A residual connection computes `x + f(x)` instead of `f(x)`, from lesson 21.
- The sigmoid squashes any number into the range 0 to 1.
- A rotation turns a 2D vector by an angle without changing its length.

## The block

Here is the whole block in two lines. $X$ is a $T \times d$ matrix, one row per
token, where $T$ is the sequence length and $d$ is the model width:

$$
X' = X + \operatorname{Attn}(\operatorname{LN}_1(X)), \qquad
X'' = X' + \operatorname{FFN}(\operatorname{LN}_2(X'))
$$

Read aloud: normalise $X$, run attention on it, and add the result back to
$X$. Then normalise that, run the feed-forward network, and add again. This
order is called **pre-norm**, because the norm comes before each sublayer.

```python
x1 = x + attention(layer_norm(x, g1, b1)) @ Wo
x2 = x1 + swiglu(layer_norm(x1, g2, b2), W_gate, W_up, W_down)
```

Layer norm works on each row by itself:

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot g + b
$$

Read aloud: $\mu$ is the mean of the row, $\sigma^2$ is its variance (the
average squared distance from the mean), and $\epsilon = 10^{-5}$ stops a
divide by zero. The gain $g$ and bias $b$ are learned vectors of length $d$.

```python
mu = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)
out = (x - mu) / np.sqrt(var + 1e-5) * g + b
```

For the row $[1, 2, 3, 6]$, the mean is 3 and the variance is 3.5. The
normalised row is about $[-1.069, -0.535, 0, 1.604]$.

### Why pre-norm

The original transformer put the norm after the add. That puts a layer norm
on the main path between every pair of blocks, and a gradient flowing back
through 50 of them can shrink or blow up. Pre-norm keeps the main path clean:
$X' = X + (\text{something})$. The "$X +$" part passes the gradient straight
back, untouched, however deep the stack is.

There's a useful side effect. If both output projections, $W_O$ and
$W_{\text{down}}$, are all zeros, each sublayer adds zero, and the block
returns its input exactly. Your exercise checks this.

## The SwiGLU feed-forward network

A plain feed-forward network is two matrices with a nonlinearity between
them. SwiGLU adds a gate: one projection decides how much of a second
projection gets through, like a dimmer switch on each hidden unit.

First, the nonlinearity. **SiLU** is $\operatorname{silu}(z) = z \cdot
\operatorname{sigmoid}(z)$, where $\operatorname{sigmoid}(z) = 1 / (1 +
e^{-z})$. It's close to $z$ for large positive $z$ and close to 0 for large
negative $z$. For example, $\operatorname{silu}(1) = 0.7311$ and
$\operatorname{silu}(-1) = -0.2689$.

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}}
$$

Read aloud: project $x$ two ways, with $W_{\text{gate}}$ and $W_{\text{up}}$,
each $d \times d_h$. Apply SiLU to the gate. Multiply the two results entry by
entry (that's what $\odot$ means). Then project back down with
$W_{\text{down}}$, which is $d_h \times d$.

```python
gate = x @ W_gate
silu = gate / (1 + np.exp(-gate))
out = (silu * (x @ W_up)) @ W_down
```

A worked example with $d = 2$ and $d_h = 1$: let $x = [1, 1]$,
$W_{\text{gate}} = [1, 0]^{\top}$, $W_{\text{up}} = [0, 2]^{\top}$, and
$W_{\text{down}} = [1, -1]$. The gate is 1, and $\operatorname{silu}(1) =
0.7311$. The up projection is 2. Their product is 1.4621. The output is
$[1.4621, -1.4621]$.

### Why two-thirds

A plain network with hidden size $d_{\text{ff}}$ has two matrices, so
$2 d \, d_{\text{ff}}$ weights. SwiGLU has three matrices, so $3 d \, d_h$. To
keep the same number of weights, set them equal: $3 d_h = 2 d_{\text{ff}}$, so

$$
d_h = \tfrac{2}{3} d_{\text{ff}}
$$

With the usual $d_{\text{ff}} = 4d$, a model with $d = 64$ has
$d_{\text{ff}} = 256$ and $d_h \approx 171$. That's why SwiGLU configs have
odd-looking hidden sizes.

## Telling the model where a token is

### Sinusoidal encodings

The original fix adds a fixed "position fingerprint" vector to each token's
embedding. The fingerprint is a set of sine and cosine waves at different
speeds, like the hands of a clock: a fast hand for fine detail and slow hands
for the big picture.

$$
\omega_i = 10000^{-2i/d}, \qquad
\operatorname{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad
\operatorname{PE}(p)_{2i+1} = \cos(p\,\omega_i)
$$

Read aloud: for $i = 0, 1, \dots, d/2 - 1$, frequency $\omega_i$ starts at 1
and shrinks geometrically. At position $p$, even column $2i$ holds the sine
of $p$ times that frequency, and odd column $2i+1$ holds the cosine.

```python
omega = 10000.0 ** (-2 * np.arange(d // 2) / d)   # (d/2,)
angles = np.arange(T)[:, None] * omega[None, :]   # (T, d/2)
pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
```

Try $T = 2$ and $d = 4$. The frequencies are $\omega_0 = 1$ and $\omega_1 =
10000^{-1/2} = 0.01$. Row 0 is $[0, 1, 0, 1]$. Row 1 is
$[\sin 1, \cos 1, \sin 0.01, \cos 0.01] = [0.8415, 0.5403, 0.0100, 0.99995]$.

Each sine and cosine pair is a point on a circle. Moving $k$ positions turns
every point by a fixed angle $k\,\omega_i$. So the encoding at $p + k$ is the
same rotation of the encoding at $p$, for every $p$. A linear layer can learn
that rotation, which lets the model reason about "three tokens back".

### Rotary position embeddings

Rotary embeddings (RoPE) use the same clock idea, but they rotate the queries
and keys instead of adding to the input. Here's why that's clever. If you turn
the query by the angle for position $m$ and the key by the angle for position
$n$, their dot product only feels the difference between the two angles. The
score depends on the distance $n - m$, not on where the pair sits.

The code uses the "rotate-half" layout. Split a vector's last axis into two
halves $x_1$ and $x_2$. Entry $j$ of $x_1$ pairs with entry $j$ of $x_2$, and
each pair is rotated by its own angle:

$$
\operatorname{RoPE}(x) = x \odot \cos + \operatorname{rotate\_half}(x) \odot \sin,
\qquad \operatorname{rotate\_half}(x) = [-x_2,\; x_1]
$$

Read aloud: multiply $x$ entry by entry by a table of cosines, then add the
rotated-half vector times a table of sines. The cosine and sine tables hold
angle $m\,\omega_j$ for position $m$, and each frequency appears twice, once
for each half, so the tables have the same width as $x$. The RoPE frequencies
are $\omega_j = \theta^{-2j/d}$ with $\theta = 10000$, the same formula as
above, where $d$ is the head width.

```python
half = x.shape[-1] // 2
rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
out = x * cos + rot * sin
```

A worked example with $d = 2$: the only frequency is $\omega_0 = 1$. Rotate
$q = [1, 0]$ at position $m = 1$. The tables are $\cos = [0.5403, 0.5403]$ and
$\sin = [0.8415, 0.8415]$. $\operatorname{rotate\_half}(q) = [0, 1]$. So the
output is $[0.5403, 0] + [0, 0.8415] = [0.5403, 0.8415]$. Its length is
still 1, as a rotation must keep it.

Inside the block, you build the tables for every position once:

```python
angles = np.arange(T)[:, None] * rope_frequencies(d_head)[None, :]  # (T, d_head/2)
cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)      # (T, d_head)
```

Then apply RoPE to each head's queries and keys, never to the values.

### Two other options, in words

- **Learned position embeddings** keep a trainable table with one row per
  position. They're flexible, but they have no row for positions beyond the
  table's length.
- **ALiBi** adds no vectors at all. It subtracts a penalty from each score
  that grows with the distance between the two tokens, so far-away tokens get
  less attention.

## Build it step by step

The checks test your functions in this order.

1. **`sinusoidal_encoding(T, d)`** returns a $(T, d)$ array. Build `omega`
   with `np.arange(d // 2)`, take the outer product with positions, and write
   `np.sin` into columns `0::2` and `np.cos` into columns `1::2`. Common
   mistake: putting sines in the first half and cosines in the second half.
   That's the RoPE layout, not this one.
2. **`rope_frequencies(d, theta=10000.0)`** returns a $(d/2,)$ array,
   `theta ** (-2 * i / d)`. Common mistake: dividing by `d // 2` instead of
   `d`.
3. **`apply_rope(x, cos, sin)`** returns an array shaped like `x`. Split the
   last axis in half, build `rotate_half`, and combine. It must work for any
   leading shape, so index with `x[..., :half]`. Common mistake: writing
   `[x2, -x1]`, which rotates the wrong way and breaks the distance property.
4. **`swiglu(x, W_gate, W_up, W_down)`** maps $(\dots, d)$ to $(\dots, d)$.
   Two matrix products, SiLU on the gate, an elementwise product, then
   `@ W_down`. Common mistake: applying SiLU to the up projection too.
5. **`transformer_block_forward(x, params, mask)`** maps $(T, d)$ to $(T, d)$.
   Write a small layer norm helper. Project with `params["Wq"]` and friends,
   reshape to heads with `.reshape(T, heads, d_head).transpose(1, 0, 2)`,
   apply RoPE to `q` and `k`, compute scores divided by `np.sqrt(d_head)`,
   add `mask`, softmax, multiply by `v`, undo the reshape, multiply by
   `params["Wo"]`, and add to `x`. Then the second norm, `swiglu`, and the
   second add. Use `params.get("theta", 10000.0)`. Common mistake: adding
   the residual to the normalised input instead of the raw `x`.
6. **`block_parameter_count(d_model, d_ff, heads)`** returns an integer:
   $4d^2$ for the four attention matrices, plus $3 d\, d_h$ with
   `d_h = round(2 * d_ff / 3)`, plus $4d$ for two layer norms' gains and
   biases. For $d = 64$ and $d_{\text{ff}} = 192$: $d_h = 128$, and the total
   is $16384 + 24576 + 256 = 41216$. Common mistake: forgetting the layer
   norm parameters.

## Cost and scaling

Each weight is used once per token in a multiply and an add, so the forward
pass costs about two flops per weight per token. With $d_{\text{ff}} = 4d$,
the counts are these.

| Part | Weights | Forward flops per token |
|---|---|---|
| Attention projections | $4d^2$ | $8d^2$, plus a part that grows with $T$ |
| Feed-forward network | $8d^2$ | $16d^2$ |
| Whole block | $12d^2$ | $24d^2$, plus a part that grows with $T$ |

A model with $L$ blocks and a vocabulary of $V$ tokens has about $V d + 12 L
d^2$ weights. For $d = 4096$, $L = 32$, and $V = 50000$, the blocks hold about
6.4 billion weights and the embedding about 0.2 billion. Model size is mostly
a question of depth and width.

## Where this breaks

- **Deep post-norm stacks.** They need a long, slow start to the learning
  rate to train at all. Pre-norm is the default for deep models.
- **Growing residual stream.** In pre-norm, each sublayer adds to the stream
  and nothing rescales it, so its size grows with depth. A final layer norm
  before the output layer fixes this.
- **RoPE past its training length.** The slowest clock hands barely move
  during training. At much longer sequences, the model sees angles it never
  learned. People rescale the frequencies to extend context.
- **Learned embeddings past their table.** There's no row for position
  $T_{\max} + 1$. You have to grow the table and train it.

## The exercise

You build a full pre-norm transformer block in one file:
`sinusoidal_encoding`, `rope_frequencies`, `apply_rope`, `swiglu`,
`transformer_block_forward`, and `block_parameter_count`. The block uses RoPE
inside attention and SwiGLU as its feed-forward network.

The checks confirm the sine and cosine layout, that shifting a sinusoidal
encoding by a fixed offset is the same linear map at every position, and that
the RoPE frequencies are the right geometric sequence. They check that
`apply_rope` keeps vector lengths and makes the query-key score depend only on
the distance between positions. Then they compare `swiglu` with a direct
computation, check that the block keeps its input shape, check that zeroing
`Wo` and `W_down` turns the block into an exact identity, and check your
parameter count against a hand computation.
