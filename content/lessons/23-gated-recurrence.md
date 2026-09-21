---
title: LSTMs and GRUs
slug: 23-gated-recurrence
part: "Part 4 — Vision and sequences"
summary: "The additive cell update that turns the backward pass through time into a product of gates instead of a product of Jacobians."
minutes: 45
gpu: false
objectives:
  - Derive the constant error carousel and show the cell-state Jacobian is a gate, not a weight matrix.
  - Write the full LSTM forward equations and justify initializing the forget-gate bias near one.
  - Derive the GRU's update and reset gates and compare its parameter count with the LSTM's.
  - State the LSTM backward pass through the cell state, and what gating does and doesn't fix.
prereqs:
  - 22-recurrent-networks
exercise: 23-gated-recurrence
---

# LSTMs and GRUs

The vanilla RNN cell carries its entire memory of the past inside $h_{t-1}$,
and every step multiplies that memory by a weight matrix before a
nonlinearity squashes it. The previous lesson showed what this costs: the
gradient of a loss at step $T$ with respect to $h_0$ is a product of $T$
Jacobians, governed by the spectral radius of $W$. Push the radius below one
and the gradient decays geometrically; push it above one and it explodes. No
setting of $W$ avoids both failure modes for every sequence length, because
the same matrix has to serve every step.

The fix here isn't a better matrix. It replaces the multiplicative recurrence
with an additive one, carried by a second piece of state that a weight
matrix never touches directly. That state is called the cell, the mechanism
is called gating, and the two instances that matter in practice are the LSTM
and the GRU. Both separate what gets remembered from how it gets computed,
and let the network learn the first part per dimension and per time step,
rather than fixing it in a matrix every step must share.

## What you already know

- The vanilla RNN cell $h_t = \tanh(Wx_t + Uh_{t-1} + b)$ and its backward
  pass through time.
- Backpropagation through $T$ steps of a vanilla RNN multiplies $T$
  Jacobians, and the spectral radius of $W$ decides whether that product
  vanishes or explodes.
- The sigmoid $\sigma(z) = 1/(1+e^{-z})$ maps $\mathbb{R}$ to $(0,1)$ with
  derivative $\sigma(z)(1-\sigma(z)) \in (0, 0.25]$.
- $\odot$ denotes the elementwise (Hadamard) product.

## The constant error carousel

Introduce a second state vector $c_t \in \mathbb{R}^m$, the cell, updated by

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t ,
$$

where $f_t, i_t \in (0,1)^m$ are gates and $g_t \in (-1,1)^m$ is a candidate
update, all computed from $x_t$ and $h_{t-1}$. The defining feature of this
update is that $c_{t-1}$ enters through addition and an elementwise product,
never a matrix multiplication, so differentiating it with the gates held
fixed gives

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) .
$$

The Jacobian of the cell-state recurrence is a diagonal matrix of gate
values, not a dense matrix of learned weights. Because diagonal matrices
commute and multiply elementwise, composing this across $T$ steps gives

$$
\frac{\partial c_T}{\partial c_0} = \operatorname{diag}\!\left( \prod_{t=1}^{T} f_t \right) .
$$

This is the **constant error carousel**: pin every entry of $f_t$ near one
and the product stays near one for every $T$, with no eigenvalue condition
on any matrix to satisfy, because there's no matrix in the recurrence to
begin with. A vanilla RNN can't make this promise, because its single shared
$W$ has to support both remembering and computing at once, and those pull
its spectral radius in opposite directions.

The gates still depend on $h_{t-1}$, so the full gradient carries a second,
indirect path alongside $\operatorname{diag}(f_t)$, which the backward pass
later in this lesson accounts for. The carousel argument isolates the direct
path deliberately, because it's the term a vanilla RNN has no analogue of.

## The LSTM cell

With input $x_t \in \mathbb{R}^n$, previous hidden state $h_{t-1} \in
\mathbb{R}^m$, and previous cell state $c_{t-1} \in \mathbb{R}^m$, the LSTM
computes four affine maps, each followed by a nonlinearity:

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{forget gate} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{input gate} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{output gate} \\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) && \text{candidate} \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

Reading left to right: the forget gate decides what fraction of the old cell
survives, the input gate decides how much of the candidate to add, the
candidate is what would be written if the input gate were fully open, and
the output gate decides how much of the cell to expose as the hidden state.
The cell is the only path along which memory travels unmultiplied by a
matrix.

**The forget-gate bias.** At initialization, weights are small, so $f_t
\approx \sigma(b_f)$. With $b_f = 0$, every unit forgets roughly half its
cell content per step, and $\mathbb{E}[\prod_{t=1}^{T} f_t] \approx 0.5^{T}$
vanishes before a single gradient update has told the network that
long-range memory is worth keeping. Setting $b_f = 1$ or $2$ gives $\sigma(b_f)
\approx 0.73$ or $0.88$, starting training with a carousel that already
preserves most of its content and leaving learning to decide where to close
the gate. Nothing stops a trained forget gate from closing to zero later —
this is only an initialization choice.

**Two variants.** The peephole variant adds a term $v_{\bullet} \odot
c_{t-1}$ to each gate's pre-activation, so gating can depend on the exact
memory content rather than only $h_{t-1}$'s summary of it. The coupled
variant sets $i_t = 1 - f_t$, tying input and forget gates so that whatever
fraction of the cell is discarded is exactly the fraction replaced.

## The GRU

The GRU drops the separate cell and works entirely on $h_t$, using two gates
instead of four:

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) && \text{update gate} \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) && \text{reset gate} \\
\tilde{h}_t &= \tanh\big(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\big) &&
  \text{candidate} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

The update equation is the coupled-gate LSTM idea by design: $1-z_t$ plays
the forget role and $z_t$ plays the input role, one gate doing both jobs.
There's no output gate, because there's no separate cell to expose — $h_t$
already is the quantity the rest of the network reads. The reset gate $r_t$
has no LSTM analogue: it controls how much of the previous state the
candidate sees before it's computed, and a reset near zero lets the
candidate start fresh from $x_t$ alone.

**Parameter count.** Each gate of either architecture costs the same: a map
from $x_t$ ($nm$ parameters), a map from $h_{t-1}$ ($m^2$ parameters), and a
bias ($m$ parameters), for $nm + m^2 + m$ per gate. The LSTM has four gates
and the GRU has three:

$$
\text{LSTM: } 4(nm + m^2 + m), \qquad \text{GRU: } 3(nm + m^2 + m) ,
$$

a ratio of exactly $3/4$ for every choice of $n$ and $m$, since every gate is
priced the same. At $n=m=512$, that's $2{,}099{,}200$ parameters for the LSTM
against $1{,}574{,}400$ for the GRU — a quarter fewer weights for a model
that gives up one gate's worth of control over what gets exposed.

## The LSTM backward pass through the cell state

Let $\bar\delta h_t = \partial L/\partial h_t$ and $\bar\delta c_t = \partial
L/\partial c_t$ be the gradients arriving from every use of $h_t$ and $c_t$
at step $t$ and later. Because $h_t = o_t \odot \tanh(c_t)$, the cell also
receives gradient through the hidden state, so the total incoming gradient
to $c_t$ is

$$
\delta c_t = \bar\delta c_t + \bar\delta h_t \odot o_t \odot \big(1 - \tanh^2(c_t)\big) .
$$

Differentiating $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ and $h_t = o_t
\odot \tanh(c_t)$ gives the gradient with respect to every quantity that
produced them:

$$
\delta o_t = \bar\delta h_t \odot \tanh(c_t), \qquad
\delta f_t = \delta c_t \odot c_{t-1}, \qquad
\delta i_t = \delta c_t \odot g_t, \qquad
\delta g_t = \delta c_t \odot i_t ,
$$

with $\delta c_{t-1} = \delta c_t \odot f_t$ — the carousel gradient from
earlier, now one term inside a complete backward pass. Each gate gradient
passes back through its own nonlinearity, for example
$\delta(\text{pre-}f_t) = \delta f_t \odot f_t \odot (1 - f_t)$, and likewise
for the other three. Every parameter gradient is then an outer product,
$\delta W_f = \delta(\text{pre-}f_t)\, x_t^{\top}$ and $\delta U_f =
\delta(\text{pre-}f_t)\,h_{t-1}^{\top}$ with $\delta b_f =
\delta(\text{pre-}f_t)$, the same pattern for the other three gates. The
gradient handed back to the previous step sums every gate's contribution:

$$
\delta h_{t-1} = \sum_{\bullet \in \{f,i,o,g\}} U_{\bullet}^{\top}\,
\delta(\text{pre-}\bullet_t), \qquad
\delta x_t = \sum_{\bullet \in \{f,i,o,g\}} W_{\bullet}^{\top}\,
\delta(\text{pre-}\bullet_t) .
$$

This is the recursion `lstm_cell_backward` implements, and it's what the
exercise checks against finite differences: every one of the eight weight
matrices and four biases, plus $x_t$, $h_{t-1}$, and $c_{t-1}$.

## Cost and scaling

Counting a matrix-vector product of shape $(m,n)$ as $2mn$ flops, each gate's
pre-activation costs $2nm + 2m^2$ flops, and the elementwise combination that
follows is lower order.

| Quantity | LSTM | GRU |
|---|---|---|
| Parameters | $4(nm+m^2+m)$ | $3(nm+m^2+m)$ |
| Flops per step (forward) | $\approx 8nm + 8m^2$ | $\approx 6nm + 6m^2$ |
| Activations stored per step for BPTT | $O(n+m)$ per gate, four gates | $O(n+m)$ per gate, three gates |

Both scale linearly in sequence length $T$, and neither parallelizes across
time, because $c_t$ depends on $c_{t-1}$ by construction. That dependency,
not the flop count, is the strongest argument for the attention-based
architectures ahead: they trade the carousel for a mechanism that inspects
every past position directly and in parallel.

## Where this breaks

**Gating slows the decay; it doesn't remove it.** The carousel gradient is
$\prod_{t=1}^{T} f_t$, a product of numbers strictly less than one whenever
any gate is less than fully open. Even at $f_t = 0.99$ for every unit and
step, $0.99^{1000} \approx 4.3 \times 10^{-5}$: still vanishing over a
thousand steps, just far slower than a vanilla RNN's.

**The architecture is still sequential.** Every step of both cells must wait
for the previous one, because the recurrence is the entire point. Training
throughput on long sequences is bounded by $T$ sequential matrix
multiplications no matter how much parallel hardware is available.

**Variants mostly don't help.** Peephole connections, coupled gates, and
similar changes to the gating equations rarely beat a plainly implemented
LSTM once its forget-gate bias is initialized sensibly. The gain comes almost
entirely from the additive cell update and from starting with the carousel
open.

**The candidate path can still explode.** If the input gate stays open and
$g_t$ repeatedly pushes the same direction, $c_t$ can grow without bound even
while the forget gate behaves well, which is why clipping remains standard
practice for gated recurrent networks too.

## The exercise

You implement one step of both cells to see the carousel argument in numbers
instead of only in algebra. Write `lstm_cell_forward(x, h, c, params)`,
returning the new hidden state, the new cell state, and a cache for the
backward pass, then `lstm_cell_backward(dh, dc, cache)`, returning the
gradient with respect to the input, the previous hidden state, the previous
cell state, and every one of the eight weight matrices and four biases in
`params`. Write `gru_cell_forward(x, h, params)` for the three-gate cell,
`forget_gate_gradient_product(forget_gates)`, which returns the cumulative
elementwise product $\prod_{t=1}^{t'} f_t$ at every $t' \le T$, and
`parameter_count(kind, input_size, hidden_size)` for `kind` equal to
`"lstm"` or `"gru"`.

The checks verify both cells' output shapes and confirm every gate lands in
its correct range, check `lstm_cell_backward` against
`mlcheck.finite_difference` for the input, both previous states, and all
twelve parameter tensors, and confirm that pinning the forget gate near one
keeps `forget_gate_gradient_product` within an order of magnitude of one
after 200 steps while a comparably deep plain-RNN-style product of
derivatives collapses to numerical zero over the same span. A final check
confirms `parameter_count` against the hand-computed
$4(nm+m^2+m)$ and $3(nm+m^2+m)$.
