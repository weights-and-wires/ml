---
title: LSTMs and GRUs
slug: 23-gated-recurrence
part: "Part 4 — Vision and sequences"
summary: "Add gates and a memory cell to a recurrent network so gradients can travel many steps back without fading."
minutes: 70
gpu: false
objectives:
  - Build one step of an LSTM cell, forward and backward, in numpy.
  - Build one step of a GRU cell and explain what its two gates do.
  - Explain why the LSTM cell state keeps the gradient alive when the forget gate stays near one.
  - Compute the parameter count of an LSTM and a GRU for any input and hidden size.
prereqs:
  - 22-recurrent-networks
exercise: 23-gated-recurrence
---

# LSTMs and GRUs

In the previous lesson, the plain RNN passed its whole memory through a weight
matrix at every step. Going backward, the gradient got multiplied by that
matrix again and again. It faded to nothing or blew up, depending on the
matrix's spectral radius. No single matrix works for every sequence length.

Gated cells fix this with a different kind of memory. Picture a conveyor belt
running alongside the network. Items ride along the belt untouched unless a
gate decides to remove some or add some. Because nothing multiplies the belt
by a weight matrix, information and gradients can ride it a long way.

The two gated cells people use are the **LSTM** (long short-term memory) and
the **GRU** (gated recurrent unit). In this lesson you build one step of each,
write the LSTM's backward pass, and measure how far a gradient survives on the
belt compared with a plain RNN.

## What you already know

- A plain RNN cell computes $h_t = \tanh(x_t W_{xh}^{\top} + h_{t-1} W_{hh}^{\top} + b)$.
- Backpropagating through many plain RNN steps multiplies the gradient by the
  same matrix many times, so it vanishes or explodes.
- The **sigmoid** $\sigma(z) = 1/(1+e^{-z})$ squashes any number into the
  range 0 to 1. Its derivative is $\sigma(z)(1 - \sigma(z))$.
- The $\tanh$ function squashes into $-1$ to $1$, with derivative $1 - \tanh^2$.
- The symbol $\odot$ means element-by-element multiplication, `*` in numpy.

## Gates

A **gate** is a vector of numbers between 0 and 1 that you multiply element by
element with another vector. A 0 blocks that entry; a 1 lets it through
completely; 0.5 lets half through. Think of a row of dimmer switches, one per
entry.

Every gate is computed the same way, from the current input and the previous
hidden state:

$$
\text{gate} = \sigma\left(x W + h U + b\right)
$$

Read it aloud: multiply the input $x$ by a weight matrix $W$, multiply the
previous hidden state $h$ by another matrix $U$, add a bias $b$, and squash
with a sigmoid so the result lies between 0 and 1. In this exercise $x$ has
shape `(n, d_x)` for $n$ examples with $d_x$ input features, and $h$ has shape
`(n, d_h)`. Each $W$ has shape `(d_x, d_h)`, each $U$ has shape `(d_h, d_h)`,
and each $b$ has shape `(d_h,)`. There's no transpose in this lesson's
convention. In numpy:

```python
f = sigmoid(x @ params["Wf"] + h @ params["Uf"] + params["bf"])
```

The starter file gives you a `_sigmoid` that doesn't overflow for large
negative inputs.

## The LSTM cell

An LSTM keeps two states: the hidden state $h$, which the rest of the network
reads, and the **cell state** $c$, which is the conveyor belt. Each step
computes three gates and one candidate:

$$
\begin{aligned}
f &= \sigma(x W_f + h U_f + b_f) && \text{forget gate} \\
i &= \sigma(x W_i + h U_i + b_i) && \text{input gate} \\
o &= \sigma(x W_o + h U_o + b_o) && \text{output gate} \\
g &= \tanh(x W_g + h U_g + b_g) && \text{candidate}
\end{aligned}
$$

Then it updates the belt and reads from it:

$$
c_{\text{new}} = f \odot c + i \odot g, \qquad
h_{\text{new}} = o \odot \tanh(c_{\text{new}})
$$

Read it aloud. The forget gate $f$ decides how much of the old belt $c$ to
keep. The candidate $g$ is new content, between $-1$ and $1$. The input gate
$i$ decides how much of it to add. The new belt is "kept old stuff plus added
new stuff". The output gate $o$ decides how much of the squashed belt to show
as the new hidden state. In numpy:

```python
c_new = f * c + i * g
h_new = o * np.tanh(c_new)
```

### Work one LSTM step by hand

Use one example and one hidden unit. Let $x = 1$, $h = 0$, $c = 0.5$. Set every
$U$ to zero, so $h$ plays no part. Set $W_f = 1$, $b_f = 1$, $W_g = 0.5$, and
every other weight and bias to zero.

- $f = \sigma(1 \times 1 + 1) = \sigma(2) = 0.8808$.
- $i = \sigma(0) = 0.5$ and $o = \sigma(0) = 0.5$.
- $g = \tanh(0.5 \times 1) = 0.4621$.
- $c_{\text{new}} = 0.8808 \times 0.5 + 0.5 \times 0.4621 = 0.4404 + 0.2311 = 0.6715$.
- $h_{\text{new}} = 0.5 \times \tanh(0.6715) = 0.5 \times 0.5859 = 0.2930$.

## Why the belt keeps gradients alive

Look at how $c$ enters $c_{\text{new}}$: it's multiplied element by element
by $f$, and nothing else. No weight matrix touches it. So going backward, the
gradient on the belt gets multiplied by $f$ at each step:

$$
dc = dc_{\text{new}} \odot f
$$

Over many steps, the gradient that reaches the start of the belt is the
running product of all the forget gates:

$$
f_1 \odot f_2 \odot \cdots \odot f_t
$$

Read it aloud: multiply the forget gates of every step together, one entry at
a time. If every forget gate sits near 1, this product stays near 1. This idea
is called the **constant error carousel**. In numpy, for an array of gates
with shape `(T, d_h)`:

```python
products = np.cumprod(forget_gates, axis=0)
```

Row $t$ of the result is the product of rows 0 through $t$. Compare some
numbers over 200 steps:

| Per-step factor | After 200 steps |
|---|---|
| LSTM forget gate 0.995 | $0.995^{200} = 0.367$ |
| Plain RNN, $\tanh$ slope 0.5 times $\rho$ 0.9 | $0.45^{200} \approx 4 \times 10^{-70}$ |

The LSTM keeps about a third of its gradient. The plain RNN keeps nothing.

The gates themselves depend on $h$, so the full gradient also has other paths.
The belt path is the one a plain RNN doesn't have, and it's the one that
matters for long memory.

**Start the forget gate open.** At the start of training the weights are
small, so $f \approx \sigma(b_f)$. With $b_f = 0$, $f \approx 0.5$ and the belt
halves every step. Setting $b_f$ to 1 gives $\sigma(1) = 0.73$, and 2 gives
$\sigma(2) = 0.88$. So people start $b_f$ at 1 or 2, and training can close
the gate later where it needs to.

## The LSTM backward pass

The backward pass receives two gradients: $dh_{\text{new}}$ and
$dc_{\text{new}}$, both shape `(n, d_h)`. It walks the forward equations in
reverse.

First, $c_{\text{new}}$ gets gradient from two places: directly, and through
$h_{\text{new}} = o \odot \tanh(c_{\text{new}})$:

$$
dc_{\text{total}} = dc_{\text{new}} + dh_{\text{new}} \odot o \odot \left(1 - \tanh^2(c_{\text{new}})\right)
$$

Read it aloud: take the gradient that arrived on the belt, and add the
gradient from the hidden state after passing it back through $o$ and the
$\tanh$.

Next, each quantity in the forward equations gets its share:

$$
do = dh_{\text{new}} \odot \tanh(c_{\text{new}}), \quad
df = dc_{\text{total}} \odot c, \quad
di = dc_{\text{total}} \odot g, \quad
dg = dc_{\text{total}} \odot i, \quad
dc_{\text{prev}} = dc_{\text{total}} \odot f
$$

Read it aloud: in a product, each factor's gradient is the upstream gradient
times the other factor. So $f$'s gradient uses $c$, $i$'s uses $g$, and so on.

Then pass each through its own squashing function. Call the value inside the
sigmoid or $\tanh$ the **pre-activation**, $a$:

$$
da_f = df \odot f \odot (1 - f), \qquad
da_g = dg \odot (1 - g^2)
$$

Read it aloud: sigmoid gates multiply by $f(1-f)$, and the $\tanh$ candidate
multiplies by $1 - g^2$. The input and output gates follow the same sigmoid
rule as $f$.

Last, the weight and input gradients. For each of the four gates:

$$
dW = x^{\top} da, \qquad dU = h^{\top} da, \qquad db = \textstyle\sum_{\text{rows}} da
$$

$$
dx = \sum_{\text{gates}} da \, W^{\top}, \qquad dh_{\text{prev}} = \sum_{\text{gates}} da \, U^{\top}
$$

Read it aloud: each gate's weight gradient is its input transposed times its
pre-activation gradient. The bias gradient sums over examples. The input and
hidden-state gradients add up one contribution from each of the four gates.
In numpy, for the forget gate:

```python
d_pre_f = df * f * (1.0 - f)
dparams["Wf"] = x.T @ d_pre_f
dx_part = d_pre_f @ params["Wf"].T
```

## The GRU cell

The GRU is a slimmer design. It drops the separate belt and uses only $h$,
with two gates and one candidate:

$$
\begin{aligned}
z &= \sigma(x W_z + h U_z + b_z) && \text{update gate} \\
r &= \sigma(x W_r + h U_r + b_r) && \text{reset gate} \\
\tilde{h} &= \tanh\left(x W_h + (r \odot h) U_h + b_h\right) && \text{candidate} \\
h_{\text{new}} &= (1 - z) \odot h + z \odot \tilde{h}
\end{aligned}
$$

Read it aloud. The update gate $z$ blends old and new: $1 - z$ of the old
state plus $z$ of the candidate. One gate does both the forget and the input
job. The reset gate $r$ decides how much of the old state the candidate is
allowed to look at. With $r$ near zero, the candidate starts fresh from the
input alone. In numpy:

```python
cand = np.tanh(x @ Wh + (r * h) @ Uh + bh)
h_new = (1.0 - z) * h + z * cand
```

For a hand example, take one unit with $x = 1$, $h = 0.5$, $W_z = 1$,
$W_h = 1$, $U_h = 1$, and all else zero. Then $z = \sigma(1) = 0.7311$,
$r = \sigma(0) = 0.5$, and $\tilde{h} = \tanh(1 + 0.5 \times 0.5) =
\tanh(1.25) = 0.8483$. The new state is $0.2689 \times 0.5 + 0.7311 \times
0.8483 = 0.7546$.

## Counting parameters

Every gate or candidate has one $W$ with $d_x d_h$ numbers, one $U$ with
$d_h^2$, and one $b$ with $d_h$. The LSTM has four of these (f, i, o, g) and
the GRU has three (z, r, candidate):

$$
\text{LSTM: } 4\left(d_x d_h + d_h^2 + d_h\right), \qquad
\text{GRU: } 3\left(d_x d_h + d_h^2 + d_h\right)
$$

Read it aloud: count one gate, then multiply by four or three. With
$d_x = d_h = 512$, one gate has $262{,}144 + 262{,}144 + 512 = 524{,}800$
numbers. The LSTM has $2{,}099{,}200$ and the GRU has $1{,}574{,}400$, a
quarter fewer.

## Build it step by step

The checks test the functions in this order.

1. **`lstm_cell_forward(x, h, c, params)`.** Takes `x` of shape `(n, d_x)`
   and `h`, `c` of shape `(n, d_h)`. Returns `(h_new, c_new, cache)`. Compute
   the four gates with `@` and `_sigmoid` or `np.tanh`, then the two update
   lines. Store `x`, `h`, `c`, `f`, `i`, `o`, `g`, `c_new`, `tanh_c_new`, and
   `params` in the cache dict. Common mistake: using `tanh` on the three gates
   or `sigmoid` on the candidate.
2. **`gru_cell_forward(x, h, params)`.** Returns `h_new`, shape `(n, d_h)`.
   Common mistake: multiplying the candidate's `h @ Uh` by `r` after the
   product instead of computing `(r * h) @ Uh`. Both give the same
   shape, but only the second is correct.
3. **`lstm_cell_backward(dh_new, dc_new, cache)`.** Returns
   `(dx, dh_prev, dc_prev, dparams)`, with `dparams` holding all twelve keys.
   Follow the order above: `dc_total`, the five products, the four
   pre-activation gradients, then the weights. Use `.T @` and `.sum(axis=0)`.
   Common mistake: forgetting the $dh_{\text{new}}$ term inside `dc_total`.
4. **`forget_gate_gradient_product(forget_gates)`.** Takes shape
   `(T, d_h)` and returns the running product, same shape. Use `np.cumprod`
   along axis 0. Common mistake: using `np.prod`, which returns only the last
   row.
5. **`parameter_count(kind, input_size, hidden_size)`.** Returns an `int` for
   `"lstm"` or `"gru"`. Common mistake: counting only one bias for the whole
   cell.

## Cost and scaling

Each gate's matrix products cost about $2 d_x d_h + 2 d_h^2$ operations per
example. The element-by-element steps cost far less.

| Quantity | LSTM | GRU |
|---|---|---|
| Parameters | $4(d_x d_h + d_h^2 + d_h)$ | $3(d_x d_h + d_h^2 + d_h)$ |
| Work per step, per example | about $8 d_x d_h + 8 d_h^2$ | about $6 d_x d_h + 6 d_h^2$ |
| Values stored per step for backprop | four gates plus the cell | three gates plus the candidate |

Both grow in a straight line with sequence length, and both must run their
steps in order.

## Where this breaks

**Gating slows the fading; it doesn't stop it.** A forget gate of 0.99 at
every step still gives $0.99^{1000} \approx 4.3 \times 10^{-5}$ after a
thousand steps. Very long memory is still hard.

**Steps still run in order.** Each $c_{\text{new}}$ needs the previous $c$.
Parallel hardware can't speed up that chain. Attention, later in the course,
removes it.

**Variants rarely matter.** Tweaks such as "peephole" connections seldom beat
a plain LSTM whose forget bias starts at 1 or 2.

**The belt can still grow.** If the input gate stays open and $g$ keeps
pushing the same way, $c$ can grow large. People still clip gradients.

## The exercise

You build `lstm_cell_forward` and `lstm_cell_backward` for one LSTM step,
`gru_cell_forward` for one GRU step, `forget_gate_gradient_product` for the
belt's running product, and `parameter_count` for both designs.

The checks confirm your LSTM output shapes and gate ranges and compare both
cells with the gate equations. They compare `lstm_cell_backward` with finite
differences for `x`, `h`, `c`, and all twelve parameters. They confirm your
running product matches a loop, that a forget gate pinned at 0.995 keeps the
product above 0.1 after 200 steps, and that `parameter_count` matches the
formulas.
