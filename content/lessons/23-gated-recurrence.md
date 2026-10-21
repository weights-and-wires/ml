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

A plain recurrent net multiplies its memory by a weight matrix every step. Going backward, you multiply the gradient, the slope of the loss, by that matrix again and again. The slope fades, or it blows up.

A gated cell adds a conveyor belt beside that path. A gate between 0 and 1 is a faucet on the number riding the belt. No weight matrix multiplies the belt, so a value can travel a long way.

You build one LSTM step and one GRU step, then send a gradient back and count the stored numbers. LSTM means long short-term memory. GRU means gated recurrent unit.

## What you already know

You already have these tools from the recurrent-net lesson:

- A plain cell mixes the input and the previous hidden state with tanh.
- Many plain steps multiply the gradient by one matrix each step.
- The sigmoid squashes any real number into the range 0 to 1.
- The tanh squashes any real number into the range −1 to 1.
- The symbol ⊙ means entry-by-entry multiplication, which is `*` in numpy.

## Four small gates and one cell

Picture one hidden unit. The forget faucet is 0.9, almost open. The input faucet is 0.2. The output faucet is 0.5. The old cell is 0.4. The candidate, fresh content between −1 and 1, is 0.6.

Keep most of the cell: 0.9 × 0.4 = 0.36. Let in a little candidate: 0.2 × 0.6 = 0.12. The updated cell is 0.36 + 0.12 = 0.48. Then tanh(0.48) = 0.4462, and half of that is 0.5 × 0.4462 = 0.2231.

This cell is an LSTM. The belt is the cell state. The faucets are the forget gate, the input gate, and the output gate. A vector does this once per entry.

$$
c_{\text{new}} = f \odot c + i \odot g, \qquad
h_{\text{new}} = o \odot \tanh(c_{\text{new}})
$$

Here c_new keeps a share of the old cell and adds a share of the candidate. The same numbers give 0.48 and 0.2231.

You write the update as follows:

```python
c_new = f * c + i * g
h_new = o * np.tanh(c_new)
```

The usual mistake is tanh on the gates, or a sigmoid on the candidate.

> [!KEY]
> The cell update keeps a share of the old belt and adds fresh content. No weight matrix multiplies the cell.

## A gate from one input

The Greek letter sigma, written σ, is the sigmoid. Take x = 1, h = 0, input weight 1, hidden weight 0, and bias 1. The sum is 1 × 1 + 0 + 1 = 2. Then σ(2) = 1 / (1 + e^{−2}). Because e^{−2} = 0.1353, the gate is 1 / 1.1353 = 0.8808. Name it a gate:

$$
\text{gate} = \sigma\left(x W + h U + b\right)
$$

The candidate uses tanh, not σ. Here `x` is `(n, d_x)`, `h` is `(n, d_h)`, each `W` is `(d_x, d_h)`, each `U` is `(d_h, d_h)`, and each bias is `(d_h,)`.

$$
\begin{aligned}
f &= \sigma(x W_f + h U_f + b_f) && \text{forget gate} \\
i &= \sigma(x W_i + h U_i + b_i) && \text{input gate} \\
o &= \sigma(x W_o + h U_o + b_o) && \text{output gate} \\
g &= \tanh(x W_g + h U_g + b_g) && \text{candidate}
\end{aligned}
$$

Each line is `x @ W + h @ U + b`, then a squash. The forget line plugs in as σ(2) = 0.8808. One gate in numpy is:

```python
f = _sigmoid(x @ params["Wf"] + h @ params["Uf"] + params["bf"])
```

The usual mistake is a transpose on `W`.

> [!WARNING]
> Lesson 22 wrote `x @ W_xh.T`. Write `x @ W` here. Each `W` is `(d_x, d_h)`.

Bias 0 gives σ(0) = 0.5, so the belt halves every step. Bias 1 gives 0.7311. Bias 2 gives 0.8808. Start the forget bias at 1 or 2.

> [!TIP]
> Start the forget bias at 1 or 2 so the belt begins open.

## Three forget gates in a row

Go back three steps with forget gate 0.9 each time. Multiply 0.9 × 0.9 = 0.81, then 0.81 × 0.9 = 0.729. After three steps, 73% of the cell gradient remains.

This running product is the constant error carousel:

$$
dc = dc_{\text{new}} \odot f
$$

$$
f_1 \odot f_2 \odot \cdots \odot f_t
$$

The old-cell gradient is the updated gradient times f. Three gates of 0.9 again give 0.729. For shape `(T, d_h)`, row t multiplies rows 0 through t:

```python
products = np.cumprod(forget_gates, axis=0)
```

The usual mistake is `np.prod`, which returns only the last row.

The 200-step comparison is:

| Per-step factor | Value after 200 steps |
|---|---|
| Forget gate 0.995 | $0.995^{200} = 0.367$ |
| Plain net, slope 0.5 times radius 0.9 | $0.45^{200} \approx 4 \times 10^{-70}$ |

> [!KEY]
> Going backward, the belt multiplies the gradient only by the forget gate. Near 1, the product survives for hundreds of steps.

## One gradient on the belt

Use the forward numbers. The cell is 0.48 and tanh(0.48) = 0.4462. Let the hidden gradient be 1 and the cell gradient be 0.2.

The hidden state was the output gate times tanh of the cell, so the cell gets slope from there too. The tanh slope is 1 − 0.4462² = 0.8009. Times the output gate: 0.5 × 0.8009 = 0.4005. Add the belt gradient: 0.2 + 0.4005 = 0.6005. That is dc_total. The forget share is 0.6005 × 0.4 = 0.2402, then 0.2402 × 0.09 = 0.0216.

$$
dc_{\text{total}} = dc_{\text{new}} + dh_{\text{new}} \odot o \odot \left(1 - \tanh^2(c_{\text{new}})\right)
$$

dc_total adds the hidden path to the belt gradient. The factors then split:

$$
do = dh_{\text{new}} \odot \tanh(c_{\text{new}}), \quad
df = dc_{\text{total}} \odot c, \quad
di = dc_{\text{total}} \odot g, \quad
dg = dc_{\text{total}} \odot i, \quad
dc_{\text{prev}} = dc_{\text{total}} \odot f
$$

Each factor's gradient is the upstream gradient times the other factor. Squash slopes come next:

$$
da_f = df \odot f \odot (1 - f), \qquad
da_g = dg \odot (1 - g^2)
$$

A sigmoid multiplies by f(1 − f), and the candidate by 1 − g². The forget plug-in is da_f = 0.0216.

Weight gradients use a batch sum on the bias:

$$
dW = x^{\top} da, \qquad dU = h^{\top} da, \qquad db = \textstyle\sum_{\text{rows}} da
$$

$$
dx = \sum_{\text{gates}} da \, W^{\top}, \qquad dh_{\text{prev}} = \sum_{\text{gates}} da \, U^{\top}
$$

A weight gradient is the transposed input times da. Add one term per gate:

```python
d_pre_f = df * f * (1.0 - f)
dparams["Wf"] = x.T @ d_pre_f
dx_part = d_pre_f @ params["Wf"].T
```

The usual mistake is dropping the hidden gradient inside dc_total.

> [!TIP]
> Key the pre-activation gradients by `"f"`, `"i"`, `"o"`, and `"g"`, then loop.

> [!KEY]
> Backward starts at dc_total, the belt slope plus the slope that arrives through the hidden state.

## An update gate instead of a belt

A GRU has no separate belt. Take x = 1, h = 0.5, biases 0, and an update weight of 1. The update gate is σ(1) = 0.7311. The reset gate is σ(0) = 0.5.

The candidate sees 0.5 × 0.5 = 0.25 of the old state. With both candidate weights at 1, tanh(1.25) = 0.8483. Keep 0.2689 × 0.5 = 0.1345. Add 0.7311 × 0.8483 = 0.6202. The sum 0.7547 is 0.7546 at full precision.

The update gate does the forget job and the input job. The reset gate scales the old state the candidate sees.

$$
\begin{aligned}
z &= \sigma(x W_z + h U_z + b_z) && \text{update gate} \\
r &= \sigma(x W_r + h U_r + b_r) && \text{reset gate} \\
\tilde{h} &= \tanh\left(x W_h + (r \odot h) U_h + b_h\right) && \text{candidate} \\
h_{\text{new}} &= (1 - z) \odot h + z \odot \tilde{h}
\end{aligned}
$$

The blend keeps 1 − z of the old state and adds z of the candidate. The same numbers give 0.7546.

You write it as follows:

```python
cand = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
h_new = (1.0 - z) * h + z * cand
```

The usual mistake multiplies by r after `h @ Uh`. With h = [1, 0], r = [1, 0], and a swap matrix, `(r * h) @ Uh` is [0, 1] and `r * (h @ Uh)` is [0, 0].

> [!KEY]
> The update gate keeps 1 − z of the old state and adds z of the candidate.

## Ten numbers per gate

Both widths are 2. One gate holds 2 × 2 + 2 × 2 + 2 = 10 numbers. Four LSTM gates hold 40. Three GRU gates hold 30.

$$
\text{LSTM: } 4\left(d_x d_h + d_h^2 + d_h\right), \qquad
\text{GRU: } 3\left(d_x d_h + d_h^2 + d_h\right)
$$

Widths of 2 give 40 and 30. You return an int:

```python
per_gate = input_size * hidden_size + hidden_size**2 + hidden_size
# Multiply per_gate by 4 for "lstm" and by 3 for "gru".
```

The usual mistake is one bias for the whole cell.

> [!RECAP]
> Carry these facts into the exercise:
> - A gate is σ(xW + hU + b), with `W` shaped `(d_x, d_h)` and no transpose. The candidate uses tanh.
> - The LSTM writes `c_new = f * c + i * g` and `h_new = o * tanh(c_new)`.
> - Backward starts at dc_total, the belt slope plus the hidden-state path.
> - The GRU candidate uses `(r * h) @ Uh`, then `(1 - z) * h + z * candidate`.
> - One gate holds d_x d_h + d_h² + d_h numbers. An LSTM has four gates and a GRU has three.

## Build it step by step

The checks test the functions in this order:

1. **`lstm_cell_forward(x, h, c, params)`.** `x` is `(n, d_x)`. `h` and `c` are `(n, d_h)`. Return `(h_new, c_new, cache)`. Build the gates with `@`, `_sigmoid`, and `np.tanh`, then the two update lines. Store `x`, `h`, `c`, `f`, `i`, `o`, `g`, `c_new`, `tanh_c_new`, and `params`. The usual mistake is tanh on a gate, or a sigmoid on the candidate.
2. **`gru_cell_forward(x, h, params)`.** Return `h_new` with shape `(n, d_h)`. Reset first: `(r * h) @ Uh`. The usual mistake is `r * (h @ Uh)`. Shapes match either way. Values do not.
3. **`lstm_cell_backward(dh_new, dc_new, cache)`.** Return `(dx, dh_prev, dc_prev, dparams)` with twelve keys. Form dc_total, the five products, the four pre-activation gradients, then `.T @` and `.sum(axis=0)`. The usual mistake is dropping `dh_new` inside dc_total.
4. **`forget_gate_gradient_product(forget_gates)`.** Input `(T, d_h)`, same output shape. Use `np.cumprod` on axis 0. The usual mistake is `np.prod`, which keeps only the last row.
5. **`parameter_count(kind, input_size, hidden_size)`.** Return an `int` for `"lstm"` or `"gru"`. The usual mistake is one bias for the whole cell.

## Cost and scaling

With both widths at 512, one gate holds 262,144 + 262,144 + 512 = 524,800 numbers. The LSTM holds 2,099,200. The GRU holds 1,574,400.

The two cells scale as follows:

| Quantity | LSTM | GRU |
|---|---|---|
| Parameters | 4(d_x d_h + d_h² + d_h) | 3(d_x d_h + d_h² + d_h) |
| Work per step, per example | about 8 d_x d_h + 8 d_h² | about 6 d_x d_h + 6 d_h² |
| Stored for the backward pass | four gates plus the cell | three gates plus the candidate |

Both grow in a line with length. Step t waits for step t − 1.

## Where this breaks

Gating slows the fade. It does not remove it. The usual failures are:

- A forget gate of 0.99 gives about 4.3 × 10^{−5} after 1,000 steps. Very long memory stays hard.
- Each cell needs the previous cell, so the steps stay in order. Attention removes that chain.
- Peephole links from the cell into the gates seldom beat a forget bias of 1 or 2.
- An open input gate and a candidate of one sign let the cell grow. You still clip gradients.

## The exercise

You build `lstm_cell_forward`, `lstm_cell_backward`, `gru_cell_forward`, `forget_gate_gradient_product`, and `parameter_count`.

The checks match both cells to the gate equations, and the backward pass to finite differences on `x`, `h`, `c`, and all twelve parameters. They match the running product to a loop, require a gate of 0.995 to stay above 0.1 after 200 steps, and match `parameter_count` to the formulas.
