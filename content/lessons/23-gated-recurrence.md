---
title: LSTMs and GRUs
slug: 23-gated-recurrence
part: "Part 4 — Vision and sequences"
summary: The additive cell update that turns the backward pass through time into a product of gates instead of a product of Jacobians.
minutes: 55
gpu: false
objectives:
  - Derive the constant error carousel and show the cell-state Jacobian is the forget gate, not a weight matrix.
  - Write the full LSTM forward equations and justify initialising the forget-gate bias near one.
  - Derive the GRU's update and reset gates and compare its parameter count with the LSTM's.
  - Derive the LSTM backward pass through the cell state and state what gating does and does not fix.
prereqs:
  - 22-recurrent-networks
exercise: 23-gated-recurrence
---

# LSTMs and GRUs

The vanilla RNN cell carries its entire memory of the past inside $h_{t-1}$, and
every step multiplies that memory by a weight matrix before a nonlinearity
squashes it. The previous lesson showed what this costs: the gradient of a loss
at step $T$ with respect to $h_0$ is a product of $T$ Jacobians, each one
$\operatorname{diag}(1-h_t^2)W^{\top}$, and the norm of that product is governed
by the spectral radius of $W$. Push the radius below one and the gradient decays
geometrically; push it above one and it explodes. There is no setting of $W$ that
avoids both failure modes for every sequence length, because the same matrix has
to serve every step.

The fix in this lesson is not a better matrix. It replaces the multiplicative
recurrence with an additive one, carried by a second piece of state that a
weight matrix never touches directly. That state is called the cell, the
mechanism is called gating, and the two instances that matter in practice are
the LSTM and the GRU. Both reduce to the same design principle: separate what
gets remembered from how it gets computed, and let the network learn the first
part per dimension and per time step, rather than fixing it in a matrix that
every step must share.

## What you already know

- The vanilla RNN cell $h_t = \tanh(Wx_t + Uh_{t-1} + b)$ and its backward pass
  through time.
- Backpropagation through $T$ steps of a vanilla RNN multiplies $T$ Jacobians,
  $\prod_{t=1}^{T} \operatorname{diag}(1-h_t^2)W^{\top}$, and the spectral radius
  of $W$ decides whether that product vanishes or explodes.
- The sigmoid $\sigma(z) = 1/(1+e^{-z})$ maps $\mathbb{R}$ to $(0,1)$ with
  derivative $\sigma(z)(1-\sigma(z)) \in (0, 0.25]$.
- The chain rule for a composition of vector-valued functions is a product of
  Jacobians, applied in reverse order.
- $\odot$ denotes the elementwise (Hadamard) product.

## The constant error carousel

Introduce a second state vector $c_t \in \mathbb{R}^m$, the cell, updated by

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t ,
$$

where $f_t, i_t \in (0,1)^m$ are gates and $g_t \in (-1,1)^m$ is a candidate
update, all three computed from $x_t$ and $h_{t-1}$ alone. The defining feature
of this update is that $c_{t-1}$ enters through addition and an elementwise
product, never through a matrix multiplication. Holding the gates fixed and
differentiating,

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) .
$$

The Jacobian of the cell-state recurrence is a diagonal matrix of gate values,
not a dense matrix built from learned weights. Composing this across $T$ steps,

$$
\frac{\partial c_T}{\partial c_0}
= \prod_{t=1}^{T} \operatorname{diag}(f_t)
= \operatorname{diag}\!\left( \prod_{t=1}^{T} f_t \right) ,
$$

because diagonal matrices commute and multiplying them multiplies their
diagonals elementwise. The gradient carried across $T$ steps of the cell state
is the elementwise product of $T$ gate vectors, one per unit, rather than a
repeated matrix product. This is the constant error carousel: pin every entry of
$f_t$ near one and $\prod_t f_t$ stays near one for every $T$, with no
eigenvalue condition on any matrix to satisfy, because there is no matrix in the
recurrence to begin with. A vanilla RNN cannot make this promise, because its
single shared $W$ has to support both remembering and computing at once, and
those pull the spectral radius in opposite directions.

The gates themselves still depend on $h_{t-1}$, and $h_{t-1}$ depends on
$c_{t-1}$ through the output gate below, so the full gradient carries a second,
indirect path alongside $\operatorname{diag}(f_t)$. That path is exactly what
the backward pass in a later section accounts for in full. The carousel argument
isolates the direct path deliberately: it is the term that a vanilla RNN has no
analogue of, and the one the forget gate controls without needing the indirect
term to cooperate.

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

with $W_{\bullet} \in \mathbb{R}^{m \times n}$, $U_{\bullet} \in
\mathbb{R}^{m \times m}$, and $b_{\bullet} \in \mathbb{R}^m$ for each of the four
gates. Reading the equations left to right: the forget gate decides what
fraction of the old cell survives, the input gate decides how much of the new
candidate to add, the candidate is what would be written if the input gate were
fully open, and the output gate decides how much of the cell to expose as the
hidden state that feeds the next layer and the next step. The hidden state is
now a filtered, squashed view of the cell, and the cell is the only path along
which memory travels unmultiplied by a matrix.

### The forget-gate bias

At initialisation, weights are small and $x_t, h_{t-1}$ have no reason to
correlate with any particular gate value, so $W_f x_t + U_f h_{t-1} \approx 0$
and $f_t \approx \sigma(b_f)$. With $b_f = 0$, every unit forgets roughly half
of its cell content at every step, and $\mathbb{E}[\prod_{t=1}^{T} f_t] \approx
0.5^{T}$ vanishes before a single gradient update has told the network that
long-range memory is worth keeping. Setting $b_f = 1$ gives $\sigma(1) \approx
0.73$, and $b_f = 2$ gives $\sigma(2) \approx 0.88$; both start training with a
carousel that already preserves most of its content, leaving learning to decide
where to *close* the gate rather than to discover, against a vanishing gradient,
that it should open one. This is an initialisation choice, not a constraint on
the trained network: nothing stops a trained forget gate from closing to zero
where the task calls for it.

### Two variants, one sentence each

The **peephole** variant lets each gate look at the cell directly by adding a
diagonal term $v_{\bullet} \odot c_{t-1}$ (or $c_t$, for the output gate) to its
pre-activation, so gating can depend on the exact memory content and not only on
$h_{t-1}$'s squashed summary of it. The **coupled** variant sets $i_t = 1 - f_t$,
tying input and forget gates so that whatever fraction of the cell is discarded
is exactly the fraction replaced by the candidate, removing one entire gate's
parameters at the cost of a design that can no longer forget and stay silent at
the same time.

## The GRU

The GRU drops the separate cell and works entirely on $h_t$, using two gates
instead of four maps:

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) && \text{update gate} \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) && \text{reset gate} \\
\tilde{h}_t &= \tanh\big(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\big) &&
  \text{candidate} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

The update equation is the coupled-gate LSTM variant applied by design: $1-z_t$
plays the forget role and $z_t$ plays the input role, sharing one gate between
the two jobs. There is no output gate, because there is no separate cell to
selectively expose — $h_t$ already is the quantity every other part of the
network reads. The only gate with no LSTM analogue is the reset gate $r_t$,
which controls how much of the previous state the candidate itself is allowed
to see before it is computed; a reset near zero lets the candidate ignore the
past entirely and start a new summary from $x_t$ alone.

### Parameter count

Each gate of either architecture costs the same: a map from $x_t$
($nm$ parameters), a map from $h_{t-1}$ ($m^2$ parameters), and a bias ($m$
parameters), for $nm + m^2 + m$ per gate. The LSTM has four gates and the GRU
has three, so

$$
\text{LSTM: } 4(nm + m^2 + m), \qquad \text{GRU: } 3(nm + m^2 + m) ,
$$

and the ratio is exactly $3/4$ for every choice of $n$ and $m$, because every
gate is priced identically. At $n=m=512$ that is $2{,}099{,}200$ parameters for
the LSTM against $1{,}574{,}400$ for the GRU — a quarter fewer weights for a
model that gives up one gate's worth of control over what gets exposed to the
rest of the network.

## The LSTM backward pass through the cell state

Fix one time step and let $\bar\delta h_t = \partial L/\partial h_t$ and
$\bar\delta c_t = \partial L/\partial c_t$ denote the gradients of the loss that
arrive from every use of $h_t$ and $c_t$ at step $t$ and later — exactly the
quantities backpropagation through time hands to this step from the future.
Because $h_t = o_t \odot \tanh(c_t)$, the cell state also receives gradient
through the hidden state, so the total incoming gradient to $c_t$ is

$$
\delta c_t = \bar\delta c_t + \bar\delta h_t \odot o_t \odot \big(1 - \tanh^2(c_t)\big) .
$$

The output gate's gradient comes directly from $h_t$:

$$
\delta o_t = \bar\delta h_t \odot \tanh(c_t) .
$$

Differentiating $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ gives the gradient
with respect to every quantity that produced it:

$$
\delta f_t = \delta c_t \odot c_{t-1}, \quad
\delta i_t = \delta c_t \odot g_t, \quad
\delta g_t = \delta c_t \odot i_t, \quad
\delta c_{t-1} = \delta c_t \odot f_t .
$$

The last of these is the carousel gradient from the previous section, now
appearing as one term inside a complete backward pass rather than in isolation.
Each gate gradient still has to pass back through its nonlinearity before it
reaches the weights:

$$
\delta(\text{pre-}f_t) = \delta f_t \odot f_t \odot (1 - f_t), \qquad
\delta(\text{pre-}g_t) = \delta g_t \odot (1 - g_t^2) ,
$$

and likewise for $i_t$ and $o_t$ through the sigmoid derivative. From there, the
parameter gradients are outer products — $\delta W_f = \delta(\text{pre-}f_t)\,
x_t^{\top}$ and $\delta U_f = \delta(\text{pre-}f_t)\,h_{t-1}^{\top}$, with
$\delta b_f = \delta(\text{pre-}f_t)$, and the same pattern for the other three
gates — and the gradient handed back to the previous step sums every gate's
contribution:

$$
\delta h_{t-1} = \sum_{\bullet \in \{f,i,o,g\}} U_{\bullet}^{\top}\,
\delta(\text{pre-}\bullet_t), \qquad
\delta x_t = \sum_{\bullet \in \{f,i,o,g\}} W_{\bullet}^{\top}\,
\delta(\text{pre-}\bullet_t) .
$$

This is the complete recursion one step of `lstm_cell_backward` implements, and
it is what the exercise checks against finite differences: every one of the
eight weight matrices and four biases, plus $x_t$, $h_{t-1}$, and $c_{t-1}$.

## Cost and scaling

Counting a matrix-vector product of shape $(m,n)$ against a vector as $2mn$
flops, each gate's pre-activation costs $2nm + 2m^2$ flops, and the elementwise
combination that follows costs $O(m)$ and is lower order.

| Quantity | LSTM | GRU |
|---|---|---|
| Parameters | $4(nm+m^2+m)$ | $3(nm+m^2+m)$ |
| Flops per step (forward) | $\approx 8nm + 8m^2$ | $\approx 6nm + 6m^2$ |
| Activations stored per step for BPTT | $O(n+m)$ per gate, four gates | $O(n+m)$ per gate, three gates |

Both scale linearly in sequence length $T$ for a full pass — $T$ times the
per-step cost — and neither can be parallelised across time, because $c_t$
depends on $c_{t-1}$ by construction. That sequential dependency, not the
per-step flop count, is what later becomes the strongest argument for the
attention-based architectures in the next two lessons: they trade the constant
error carousel for a mechanism that inspects every past position directly and
computes every position's representation in parallel.

## Where this breaks

**Gating slows the decay; it does not remove it.** The carousel gradient is
$\prod_{t=1}^{T} f_t$, a product of numbers strictly less than one whenever any
gate is less than fully open. Even at $f_t = 0.99$ for every unit and every
step, $0.99^{1000} \approx 4.3 \times 10^{-5}$: still a vanishing gradient over
a thousand steps, just a far slower one than a vanilla RNN's. Dependencies
spanning tens of thousands of steps remain out of reach for the same reason
they were out of reach before, at a longer horizon.

**The architecture is still sequential.** Every step of both cells must wait
for the previous one to finish, because the recurrence is the entire point.
Training throughput on long sequences is bounded by $T$ sequential matrix
multiplications no matter how much parallel hardware is available, which is a
cost gating never addresses because gating is a fix for gradient flow, not for
the dependency structure of the computation.

**Variants mostly do not help.** Peephole connections, coupled gates, and other
architectural changes to the gating equations have been tried extensively, and
broad empirical comparisons across tasks find that none of them reliably beats
a plainly implemented LSTM once its forget-gate bias is initialised sensibly.
The gain from gating comes almost entirely from the additive cell update itself
and from starting training with the carousel open, not from the exact shape of
the gates around it.

**The candidate path can still explode.** If the input gate stays open and the
candidate $g_t$ repeatedly pushes in the same direction, $c_t$ can grow without
bound even while the forget gate behaves well; gradient clipping remains
standard practice for gated recurrent networks for exactly this reason.

## The exercise

You implement one step of both cells and the machinery to see the carousel
argument in numbers instead of only in algebra. Write `lstm_cell_forward(x, h,
c, params)`, returning the new hidden state, the new cell state, and a cache for
the backward pass, then `lstm_cell_backward(dh, dc, cache)`, returning the
gradient with respect to the input, the previous hidden state, the previous
cell state, and every one of the eight weight matrices and four biases packed
into `params`. Write `gru_cell_forward(x, h, params)` for the three-gate cell.
Then write `forget_gate_gradient_product(forget_gates)`, which takes an array of
forget-gate values across $T$ steps and returns the cumulative elementwise
product $\prod_{t=1}^{t'} f_t$ at every $t' \le T$ — the exact quantity this
lesson derived for $\partial c_T/\partial c_0$ — and `parameter_count(kind,
input_size, hidden_size)`, which returns the parameter count for `kind` equal to
`"lstm"` or `"gru"`.

The checks verify both cells' output shapes and confirm every gate lands in its
correct range, check `lstm_cell_backward` against `mlcheck.finite_difference`
for the input, both previous states, and all twelve parameter tensors, and
confirm that pinning the forget gate near one keeps
`forget_gate_gradient_product` within an order of magnitude of one after 200
steps while a comparably deep plain-RNN-style product of derivatives collapses
to numerical zero over the same span. A final check confirms `parameter_count`
against the hand-computed $4(nm+m^2+m)$ and $3(nm+m^2+m)$.
