---
title: Backpropagation through the LSTM, and the GRU
slug: 052-lstm-backward-and-the-gru
part: "Part 5 — Vision and sequences"
summary: Send the gradient back through one LSTM step by hand, then merge its forget and input gates into the GRU and count what each cell costs.
minutes: 75
gpu: false
objectives:
  - Explain why the LSTM cell state receives gradient from two paths, and why its total gradient is their sum.
  - Work the backward pass of one LSTM unit by hand, and write lstm_cell_backward in numpy that matches finite differences.
  - Explain why one update gate can replace the forget and input gates, and what the GRU gives up by doing so.
  - Write the GRU forward step with the reset gate applied before the recurrent matrix, and count the parameters of a plain RNN, a GRU, and an LSTM.
prereqs:
  - 034-backpropagation
  - 051-the-lstm
exercise: 052-lstm-backward-and-the-gru
---

# Backpropagation through the LSTM, and the GRU

The loss has looked at the output of one LSTM step and found it wanting. It says: the shown value $h_{\text{new}}$ should move, with slope 1, and the new cell $c_{\text{new}}$ should move too, with slope 0.2, because the next step reads it. Which way should each of the three gates nudge? How much should each weight change, and how much of this error belongs to the previous cell and the previous shown value, so it can be passed on to the step before?

You could answer by brute force: nudge one weight a little, rerun the cell, and see how the loss moves. A cell with 512 hidden units and 512 inputs holds about two million weights, so that is four million forward passes for a single gradient. Backpropagation answers every question in one backward sweep, because each operation in the cell needs only its own small slope. The new idea is one place where gradient arrives from two directions, and the sum that follows.

Once the sweep works, a second question follows. That LSTM holds 2,099,200 numbers, all stored, all updated by the optimiser, and all multiplied again at every step. If a cell with a quarter fewer numbers keeps the belt's benefit, you train it faster and run it in less memory. The gated recurrent unit, or GRU, is that cell: it ties the forget and input gates together and drops the separate cell.

## What you already know

- The forward step of an LSTM cell, with its three gates, its candidate, and its two outputs, and the forget bias that keeps the belt open ([the LSTM](/lessons/051-the-lstm)).
- Backpropagation sends a gradient backward through a graph one operation at a time, and a value used in two places collects the sum of the gradients from both ([backpropagation](/lessons/034-backpropagation)).
- Each array operation has a small backward rule that keeps the gradient the same shape as the value.
- The sigmoid has slope $\sigma(1 - \sigma)$ and tanh has slope $1 - \tanh^2$.
- Along the belt, the gradient across one step is multiplied by the forget gate.

## Two words and two rules

Two terms keep the rest short. The **upstream gradient** at a node is the gradient that arrives from the loss side: how much the loss changes per unit change in that node's output. The **local derivative** is the slope of the node's output with respect to one of its inputs, with the other inputs held still. Backward through a node, the gradient on an input is the upstream gradient times the local derivative.

Two node types cover the whole LSTM. A **sum** node, $y = a + b$, has a local derivative of 1 for each input, so it hands its upstream gradient to both inputs unchanged. A **product** node, $y = a \cdot b$, has local derivative $b$ for input $a$ and $a$ for input $b$, because nudging $a$ by a small amount changes $y$ by $b$ times that amount. So the gradient on one input of a product is the upstream gradient times the other input. Figure 1 draws the cell as a graph, with the forward values in blue and the backward gradients in orange.

```figure
<svg viewBox="0 0 700 335" role="img" aria-label="A computation graph of one LSTM step. Forward arrows run from c, f, i, g through two products and a plus to the new cell, then through tanh and a product with o to the hidden output. Backward arrows run the other way, and the new cell receives gradient from two places">
<defs><marker id="c040-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c040-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="69" y="38" width="52" height="28" rx="6" class="fill-soft-a stroke-a"/>
<text x="95" y="57" text-anchor="middle">c</text>
<rect x="69" y="98" width="52" height="28" rx="6" class="fill-soft-a stroke-a"/>
<text x="95" y="117" text-anchor="middle">f</text>
<rect x="69" y="178" width="52" height="28" rx="6" class="fill-soft-a stroke-a"/>
<text x="95" y="197" text-anchor="middle">i</text>
<rect x="69" y="238" width="52" height="28" rx="6" class="fill-soft-c stroke-c"/>
<text x="95" y="257" text-anchor="middle">g</text>
<rect x="348" y="137" width="84" height="30" rx="6" class="fill-soft-d stroke-d"/>
<text x="390" y="157" text-anchor="middle">c_new</text>
<rect x="348" y="214" width="84" height="30" rx="6" class="fill-soft-a stroke-a"/>
<text x="390" y="234" text-anchor="middle">tanh</text>
<rect x="484" y="291" width="52" height="28" rx="6" class="fill-soft-a stroke-a"/>
<text x="510" y="310" text-anchor="middle">o</text>
<rect x="558" y="214" width="64" height="30" rx="6" class="fill-soft-b stroke-b"/>
<text x="590" y="234" text-anchor="middle">h_new</text>
<circle cx="200" cy="82" r="16" class="fill-paper stroke-ink"/>
<text x="200" y="88" text-anchor="middle" class="fig-bold">×</text>
<circle cx="200" cy="222" r="16" class="fill-paper stroke-ink"/>
<text x="200" y="228" text-anchor="middle" class="fig-bold">×</text>
<circle cx="300" cy="152" r="16" class="fill-paper stroke-ink"/>
<text x="300" y="158" text-anchor="middle" class="fig-bold">+</text>
<circle cx="510" cy="229" r="16" class="fill-paper stroke-ink"/>
<text x="510" y="235" text-anchor="middle" class="fig-bold">×</text>
<line x1="121.2" y1="54.8" x2="185.9" y2="73.3" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="183.4" y1="81.9" x2="118.8" y2="63.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="118.8" y1="100.5" x2="183.4" y2="82.1" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="185.9" y1="90.7" x2="121.2" y2="109.2" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="215.7" y1="87.5" x2="289.5" y2="139.1" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="284.3" y1="146.5" x2="210.5" y2="94.9" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="121.2" y1="194.8" x2="185.9" y2="213.3" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="183.4" y1="221.9" x2="118.8" y2="203.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="118.8" y1="240.5" x2="183.4" y2="222.1" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="185.9" y1="230.7" x2="121.2" y2="249.2" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="210.5" y1="209.1" x2="284.3" y2="157.5" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="289.5" y1="164.9" x2="215.7" y2="216.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="316" y1="147.5" x2="348" y2="147.5" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="348" y1="156.5" x2="316" y2="156.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="394.5" y1="167" x2="394.5" y2="214" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="385.5" y1="214" x2="385.5" y2="167" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="432" y1="224.5" x2="494" y2="224.5" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="494" y1="233.5" x2="432" y2="233.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="505.5" y1="291" x2="505.5" y2="245" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="514.5" y1="245" x2="514.5" y2="291" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="526" y1="224.5" x2="558" y2="224.5" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="558" y1="233.5" x2="526" y2="233.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<line x1="432" y1="147.5" x2="674" y2="147.5" class="stroke-a" marker-end="url(#c040-arrow-a)"/>
<line x1="674" y1="156.5" x2="432" y2="156.5" class="stroke-b" marker-end="url(#c040-arrow-b)"/>
<text x="450" y="138" class="fig-small">to the next step's cell</text>
<text x="63" y="57" text-anchor="end" class="fig-small fig-bold">dc_prev</text>
<text x="63" y="117" text-anchor="end" class="fig-small fig-bold">df</text>
<text x="63" y="197" text-anchor="end" class="fig-small fig-bold">di</text>
<text x="63" y="257" text-anchor="end" class="fig-small fig-bold">dg</text>
<text x="542" y="309" class="fig-small fig-bold">do</text>
<text x="626" y="233" class="fig-small fig-bold">dh_new</text>
<text x="450" y="172" class="fig-small fig-bold">dc_new</text>
<text x="390" y="122" text-anchor="middle" class="fig-small fig-bold">dc_total</text>
<text x="40" y="20" class="fig-small">forward values in blue, gradients flowing back in orange</text>
</svg>
Figure 1. One LSTM step as a graph. Blue arrows carry the forward values, and orange arrows carry the gradient back. Notice that c_new has two outgoing blue arrows, one to the next step and one down through tanh to h_new, so two orange arrows arrive there and their gradients add to make dc_total.
```

## Why the cell state gets two paths

Look at $c_{\text{new}}$ in Figure 1. It is used in two places: it is the cell that the next step will read, and it is also squashed by tanh and multiplied by the output gate to make $h_{\text{new}}$. A value used in two places collects the sum of the gradients from both, because nudging it changes the loss through both routes at once.

The first route's gradient is given: $dc_{\text{new}} = 0.2$. The second route runs back through the product with $o$ and then through tanh. Use the numbers of the one-step example from [the LSTM](/lessons/051-the-lstm): $c = 0.4$, $g = 0.6$, $f = 0.9$, $i = 0.2$, $o = 0.5$, and $c_{\text{new}} = 0.48$. Start with $dh_{\text{new}} = 1$. The product with $o = 0.5$ passes back $1 \times 0.5 = 0.5$, the upstream times the other factor. The tanh then multiplies by its slope, $1 - \tanh^2(0.48) = 1 - 0.4462^2 = 0.8009$, so the second route delivers $0.5 \times 0.8009 = 0.4004$. The total gradient on the new cell is

$$
dc_{\text{total}} = dc_{\text{new}} + dh_{\text{new}} \odot o \odot \left(1 - \tanh^2(c_{\text{new}})\right)
$$

Read it aloud: the total gradient on the new cell is the gradient from the next step plus the gradient from the shown value, which is $dh_{\text{new}}$ times the output gate times the tanh slope. With the numbers, $0.2 + 0.4004 = 0.6004$. In numpy:

```python
dc_total = dc_new + dh_new * o * (1.0 - tanh_c_new ** 2)
```

The most common mistake is dropping the $dh_{\text{new}}$ term, which treats the cell as feeding only the next step.

## Down through the products, the squashes, and the weights

Now $dc_{\text{total}}$ flows back through the sum and the two products that built $c_{\text{new}} = f \odot c + i \odot g$. The sum hands $dc_{\text{total}}$ to both products unchanged, and each product gives each input the upstream gradient times the other input:

$$
df = dc_{\text{total}} \odot c, \quad
di = dc_{\text{total}} \odot g, \quad
dg = dc_{\text{total}} \odot i, \quad
dc_{\text{prev}} = dc_{\text{total}} \odot f
$$

The last one is the belt. The gradient reaching the previous cell is the total times the forget gate, so $dc_{\text{prev}} = 0.6004 \times 0.9 = 0.5404$. Inside a single step this is exact, because $c$ enters $c_{\text{new}}$ only through the product $f \odot c$. The output gate gets its own share from the other product, $do = dh_{\text{new}} \odot \tanh(c_{\text{new}}) = 0.4462$. The other three come out as $df = 0.2402$, $di = 0.3603$, and $dg = 0.1201$. Figure 2 follows the numbers along the belt.

```figure
<svg viewBox="0 0 660 345" role="img" aria-label="The worked example as a chain from c to h_new. Each box shows its forward value and its gradient. The gradient of the new cell is 0.2 plus 0.4004, which is 0.6004, and the gradient of the previous cell is 0.5404">
<defs><marker id="c040-fig2-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c040-fig2-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="10" y="30" width="112" height="100" rx="6" class="fill-soft-a stroke-a"/>
<text x="66" y="50" text-anchor="middle" class="fig-bold">c</text>
<text x="66" y="72" text-anchor="middle" class="fig-mono">fwd 0.4</text>
<text x="66" y="94" text-anchor="middle" class="fig-mono">grad 0.5404</text>
<text x="66" y="116" text-anchor="middle" class="fig-small">= dc_total × f</text>
<rect x="142" y="30" width="112" height="100" rx="6" class="fill-soft-a stroke-a"/>
<text x="198" y="50" text-anchor="middle" class="fig-bold">f · c</text>
<text x="198" y="72" text-anchor="middle" class="fig-mono">fwd 0.36</text>
<text x="198" y="94" text-anchor="middle" class="fig-mono">grad 0.6004</text>
<text x="198" y="116" text-anchor="middle" class="fig-small"></text>
<rect x="274" y="30" width="112" height="100" rx="6" class="fill-soft-a stroke-a"/>
<text x="330" y="50" text-anchor="middle" class="fig-bold">c_new</text>
<text x="330" y="72" text-anchor="middle" class="fig-mono">fwd 0.48</text>
<text x="330" y="94" text-anchor="middle" class="fig-mono">grad 0.6004</text>
<text x="330" y="116" text-anchor="middle" class="fig-small">0.2 + 0.4004</text>
<rect x="406" y="30" width="112" height="100" rx="6" class="fill-soft-a stroke-a"/>
<text x="462" y="50" text-anchor="middle" class="fig-bold">tanh(c_new)</text>
<text x="462" y="72" text-anchor="middle" class="fig-mono">fwd 0.4462</text>
<text x="462" y="94" text-anchor="middle" class="fig-mono">grad 0.5</text>
<text x="462" y="116" text-anchor="middle" class="fig-small">= 1 × o</text>
<rect x="538" y="30" width="112" height="100" rx="6" class="fill-soft-a stroke-a"/>
<text x="594" y="50" text-anchor="middle" class="fig-bold">h_new</text>
<text x="594" y="72" text-anchor="middle" class="fig-mono">fwd 0.2231</text>
<text x="594" y="94" text-anchor="middle" class="fig-mono">grad 1</text>
<text x="594" y="116" text-anchor="middle" class="fig-small"></text>
<rect x="274" y="190" width="112" height="70" rx="6" class="fill-soft-a stroke-a"/>
<text x="330" y="210" text-anchor="middle" class="fig-bold">i · g</text>
<text x="330" y="232" text-anchor="middle" class="fig-mono">fwd 0.12</text>
<text x="330" y="254" text-anchor="middle" class="fig-mono">grad 0.6004</text>
<line x1="140" y1="82" x2="124" y2="82" class="stroke-b" marker-end="url(#c040-fig2-arrow-b)"/>
<line x1="272" y1="82" x2="256" y2="82" class="stroke-b" marker-end="url(#c040-fig2-arrow-b)"/>
<line x1="404" y1="82" x2="388" y2="82" class="stroke-b" marker-end="url(#c040-fig2-arrow-b)"/>
<line x1="536" y1="82" x2="520" y2="82" class="stroke-b" marker-end="url(#c040-fig2-arrow-b)"/>
<text x="136" y="150" text-anchor="middle" class="fig-small">× f</text>
<text x="268" y="150" text-anchor="middle" class="fig-small">× 1</text>
<text x="400" y="150" text-anchor="middle" class="fig-small">× 0.8009</text>
<text x="532" y="150" text-anchor="middle" class="fig-small">× o</text>
<line x1="330" y1="188" x2="330" y2="134" class="stroke-b" marker-end="url(#c040-fig2-arrow-b)"/>
<text x="340" y="160" class="fig-small">× 1</text>
<text x="10" y="292" class="fig-small">df = 0.6004 × c = 0.2402, then da_f = 0.2402 × 0.9 × 0.1 = 0.0216</text>
<text x="10" y="312" class="fig-small">di = 0.6004 × g = 0.3603, and dg = 0.6004 × i = 0.1201</text>
<text x="10" y="332" class="fig-small">do = dh_new × tanh(c_new) = 0.4462</text>
<text x="10" y="14" class="fig-small">Orange arrows carry the gradient back; each label is the local slope used on that edge</text>
</svg>
Figure 2. The worked example as a chain, with forward values and gradients in each box. Notice that the new cell's gradient is 0.2 plus 0.4004, that the gradient across the plus sign is unchanged, and that the previous cell ends up with 0.5404, which is the total times the forget gate of 0.9.
```

Across many steps the gates also read the previous shown value, which depends on the previous cell, so a second, smaller route runs from one cell to the next. Backpropagating a whole sequence includes it, and `lstm_cell_backward` hands it to the step before through `dh_prev`.

A gate is a sigmoid of a weighted sum. Call that weighted sum $x W + h U + b$ the **pre-activation**, because it comes before the squash. The weights see the pre-activation, not the gate, so the gradient must pass backward through the squash first. A sigmoid gate has local slope $f(1 - f)$ and the candidate has slope $1 - g^2$. Write $da$ for the gradient on a pre-activation:

$$
da_f = df \odot f \odot (1 - f), \qquad
da_g = dg \odot (1 - g^2)
$$

and likewise $da_i = di \odot i \odot (1 - i)$ and $da_o = do \odot o \odot (1 - o)$. Read it aloud: the gradient on the pre-activation is the gradient on the gate times the gate's own slope. For the forget gate, $da_f = 0.2402 \times 0.9 \times 0.1 = 0.0216$. The other three come out as $da_i = 0.0576$, $da_o = 0.1116$, and $da_g = 0.1201 \times (1 - 0.6^2) = 0.0769$. A gate near 0 or 1 has a slope near 0, so a saturated gate passes back very little.

Each pre-activation is $a = x W + h U + b$, so the rules for array operations apply to each gate, once per gate:

$$
dW = x^{\top} da, \qquad dU = h^{\top} da, \qquad db = \textstyle\sum_{\text{rows}} da
$$

$$
dx = \sum_{\text{gates}} da \, W^{\top}, \qquad dh_{\text{prev}} = \sum_{\text{gates}} da \, U^{\top}
$$

Read it aloud: a weight's gradient is the input that fed it, transposed, times the gradient on the pre-activation. The bias gradient adds up the rows of the batch, because one bias was added to every row. The input $x$ and the previous shown value $h$ each feed all four gates, so their gradients are the sum of one term per gate. In numpy, once for every gate:

```python
dW[name] = x.T @ da[name]
dx += da[name] @ params["W" + name].T
```

> [!TIP]
> Keep the four pre-activation gradients in a dict keyed by `"f"`, `"i"`, `"o"`, and `"g"`, and loop. The four gates then share one block of code, and a typo in one cannot hide.

You can test the whole backward pass without trusting any derivation. Nudge one entry of `x` by a small $\epsilon$ up and down, rerun the forward step, and divide the change in loss by $2\epsilon$. That finite-difference gradient must match the backward pass. On the exercise's test case, the largest gap over the twelve entries of $dx$ is about $2 \times 10^{-11}$.

> [!KEY]
> The cell state has two outgoing paths, so its gradient is their sum, $dc_{\text{total}}$. Each product then passes back the upstream times the other factor, each squash multiplies by its own slope, and the weights, the input, and the previous state read off what they need. The previous cell receives $dc_{\text{total}} \odot f$, the belt gradient.

## One gate that blends

Look back at the store-hold-read example from [the LSTM](/lessons/051-the-lstm). When the forget gate was 0, the input gate was 1, and when the forget gate was 1, the input gate was 0. The two gates moved in opposite directions. If they usually do, one gate could do both jobs, and the cell would need one weight set fewer.

The GRU has no separate cell. It keeps a single state $h$, which is both its memory and the value it shows, so there is no output gate either. Its update is a blend of the old state and a candidate $\tilde{h}$, using one **update gate** $z$:

$$
h_{\text{new}} = (1 - z) \odot h + z \odot \tilde{h}
$$

Read it aloud: the new state keeps the fraction $1 - z$ of the old state and takes the fraction $z$ of the candidate. The two shares always add to 1, so the new state sits between the old state and the candidate, never beyond. A gate of $z = 0$ keeps the old state exactly, and $z = 1$ replaces it. This is the LSTM's forget gate and input gate tied together as $1 - z$ and $z$. In numpy, `(1 - z) * h + z * candidate`. It is also why the belt survives. The old state reaches the new one through the factor $1 - z$, so when the net sets $z$ near 0 to hold a memory, the gradient crosses that step almost unchanged, exactly like a forget gate near 1.

Take one unit with every weight written out. The input is $x = 1$ and the old state is $h = 0.5$. All biases are 0. The update gate has $W_z = 1$ and $U_z = 0$, so $z = \sigma(1) = 0.7311$. The reset gate, defined next, has $W_r = 0$ and $U_r = 0$, so $r = \sigma(0) = 0.5$. The candidate has $W_h = 1$ and $U_h = 1$ and sees the old state scaled by $r$, which is $0.5 \times 0.5 = 0.25$, so the candidate is $\tanh(1 \times 1 + 0.25 \times 1) = \tanh(1.25) = 0.8483$. The new state keeps $0.2689 \times 0.5 = 0.1345$ of the old state and writes $0.7311 \times 0.8483 = 0.6201$ of the candidate, and the sum is $0.7546$.

What does the GRU give up? In an LSTM the forget and input gates are independent, so a cell can keep its whole old value and add a full candidate on top ($f = 1$ and $i = 1$), and the cell grows. In a GRU the shares always sum to 1, so keeping more of the past means writing less of the present. For many tasks that is a fair trade for a quarter fewer numbers, but it is a trade, not a free saving.

## The reset gate

The candidate is a proposal computed from the input and the old state. Sometimes the old state would mislead it. Suppose a sentence has just ended and a fresh clause begins. The state still describes the old clause, and a candidate computed from it would carry that description into the next one. The **reset gate** $r$ lets the net hide the past from the candidate by scaling the old state before the candidate sees it. When $r$ is near 0, the candidate is a fresh function of the input alone, and when $r$ is near 1, it sees the whole state.

$$
\begin{aligned}
z &= \sigma(x W_z + h U_z + b_z) && \text{update gate} \\
r &= \sigma(x W_r + h U_r + b_r) && \text{reset gate} \\
\tilde{h} &= \tanh\left(x W_h + (r \odot h) U_h + b_h\right) && \text{candidate} \\
h_{\text{new}} &= (1 - z) \odot h + z \odot \tilde{h}
\end{aligned}
$$

Read the third line aloud: the candidate sees the input and the old state scaled entry by entry by $r$, each through its own matrix. In numpy:

```python
cand = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
```

The order matters. Take $h = [1, 0]$, $r = [1, 0]$, and a swap matrix $U_h$ that exchanges the two entries. Then $(r \odot h) U_h = [1, 0] U_h = [0, 1]$, but $r \odot (h U_h) = [1, 0] \odot [0, 1] = [0, 0]$. Both versions have the same shape, so only the numbers reveal the mistake. Figure 3 puts the whole cell next to the LSTM.

```figure
<svg viewBox="0 0 620 345" role="img" aria-label="Two cell layouts side by side. The LSTM has a forget gate, an input gate, an output gate, a candidate, a cell, and a hidden value. The GRU merges the forget and input gates into one update gate, adds a reset gate, and has only a hidden state.">
<defs><marker id="c040-fig3-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20" y="26" class="fig-bold">LSTM: four weight sets</text>
<text x="350" y="26" class="fig-bold">GRU: three weight sets</text>
<rect x="20" y="50" width="200" height="38" rx="6" class="fill-soft-b stroke-b"/>
<text x="120" y="74" text-anchor="middle">forget gate f</text>
<rect x="20" y="98" width="200" height="38" rx="6" class="fill-soft-b stroke-b"/>
<text x="120" y="122" text-anchor="middle">input gate i</text>
<rect x="20" y="156" width="200" height="38" rx="6" class="fill-soft-a stroke-a"/>
<text x="120" y="180" text-anchor="middle">output gate o</text>
<rect x="20" y="204" width="200" height="38" rx="6" class="fill-soft-c stroke-c"/>
<text x="120" y="228" text-anchor="middle">candidate g (tanh)</text>
<rect x="20" y="266" width="90" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="65" y="291" text-anchor="middle">cell c</text>
<rect x="130" y="266" width="90" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="175" y="291" text-anchor="middle">hidden h</text>
<polyline points="226,52 234,52 234,134 226,134" fill="none" class="stroke-b"/>
<line x1="234" y1="93" x2="340" y2="93" class="stroke-b" marker-end="url(#c040-fig3-arrow)"/>
<text x="238" y="82" class="fig-small">merged</text>
<rect x="350" y="50" width="220" height="86" rx="6" class="fill-soft-b stroke-b"/>
<text x="460" y="98" text-anchor="middle">update gate z</text>
<rect x="350" y="156" width="220" height="38" rx="6" class="fill-soft-d stroke-d"/>
<text x="460" y="180" text-anchor="middle">reset gate r</text>
<rect x="350" y="204" width="220" height="38" rx="6" class="fill-soft-c stroke-c"/>
<text x="460" y="228" text-anchor="middle">candidate (tanh)</text>
<rect x="350" y="266" width="220" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="460" y="291" text-anchor="middle">hidden h (the only state)</text>
<text x="350" y="66"></text>
<text x="460" y="118" text-anchor="middle" class="fig-small">keeps 1 - z, writes z</text>
<text x="20" y="330" class="fig-small">No output gate in a GRU: the state is shown as it is.</text>
</svg>
Figure 3. The two cells with matching colours for matching roles. The forget and input gates, in orange, merge into the single update gate z. The GRU adds a reset gate, in purple, and drops both the separate cell and the output gate. Notice that the candidate, in green, is the same idea in both cells.
```

## What each cell costs

A weight set holds three things: `W` of shape $(d_x, d_h)$, `U` of shape $(d_h, d_h)$, and `b` of shape $(d_h,)$. With input width $d_x$ and hidden width $d_h$, one set holds $d_x d_h + d_h^2 + d_h$ numbers. Take both widths as 2: $4 + 4 + 2 = 10$. A plain recurrent cell has one such set. A GRU has three, for $z$, $r$, and the candidate, so 30. An LSTM has four, for $f$, $i$, $o$, and the candidate, so 40:

$$
\text{GRU: } 3\left(d_x d_h + d_h^2 + d_h\right), \qquad
\text{LSTM: } 4\left(d_x d_h + d_h^2 + d_h\right)
$$

Read it aloud: count one weight set, then multiply by the number of gates plus candidates. At both widths 512, one set holds $262{,}144 + 262{,}144 + 512 = 524{,}800$ numbers. A plain RNN holds 524,800, a GRU holds 1,574,400, and an LSTM holds 2,099,200. Gating costs three or four times as much as a plain cell, and the GRU saves exactly a quarter of the LSTM. Figure 4 shows these counts and how they grow.

```figure
<svg viewBox="0 0 650 285" role="img" aria-label="Left, bars of parameter counts at width 512: plain RNN 0.52 million, GRU 1.57 million, LSTM 2.10 million. Right, lines of parameter counts for hidden sizes 64 to 1024, growing with the square of the size.">
<text x="20" y="24" class="fig-bold">Hidden and input size both 512</text>
<rect x="40" y="202.5" width="64" height="47.5" class="fill-a"/>
<text x="72" y="194.5" text-anchor="middle" class="fig-small fig-bold">0.52M</text>
<text x="72" y="268" text-anchor="middle" class="fig-small">plain RNN</text>
<rect x="130" y="107.5" width="64" height="142.5" class="fill-c"/>
<text x="162" y="99.5" text-anchor="middle" class="fig-small fig-bold">1.57M</text>
<text x="162" y="268" text-anchor="middle" class="fig-small">GRU</text>
<rect x="220" y="60" width="64" height="190" class="fill-b"/>
<text x="252" y="52" text-anchor="middle" class="fig-small fig-bold">2.10M</text>
<text x="252" y="268" text-anchor="middle" class="fig-small">LSTM</text>
<line x1="30" y1="250" x2="310" y2="250" class="stroke-muted"/>
<text x="340" y="24" class="fig-bold">Counts as the width grows (d_x = d_h)</text>
<line x1="380" y1="250" x2="570" y2="250" class="stroke-grid"/>
<text x="372" y="254" text-anchor="end" class="fig-small">0M</text>
<line x1="380" y1="202.4" x2="570" y2="202.4" class="stroke-grid"/>
<text x="372" y="206.4" text-anchor="end" class="fig-small">2M</text>
<line x1="380" y1="154.8" x2="570" y2="154.8" class="stroke-grid"/>
<text x="372" y="158.8" text-anchor="end" class="fig-small">4M</text>
<line x1="380" y1="107.1" x2="570" y2="107.1" class="stroke-grid"/>
<text x="372" y="111.1" text-anchor="end" class="fig-small">6M</text>
<line x1="380" y1="59.5" x2="570" y2="59.5" class="stroke-grid"/>
<text x="372" y="63.5" text-anchor="end" class="fig-small">8M</text>
<text x="380" y="268" text-anchor="middle" class="fig-small">64</text>
<text x="427.5" y="268" text-anchor="middle" class="fig-small">128</text>
<text x="475" y="268" text-anchor="middle" class="fig-small">256</text>
<text x="522.5" y="268" text-anchor="middle" class="fig-small">512</text>
<text x="570" y="268" text-anchor="middle" class="fig-small">1024</text>
<text x="475" y="286" text-anchor="middle" class="fig-small">hidden size</text>
<polyline points="380,249.2 427.5,246.9 475,237.5 522.5,200 570,50.2" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="380" cy="249.2" r="3.5" class="fill-b"/>
<circle cx="427.5" cy="246.9" r="3.5" class="fill-b"/>
<circle cx="475" cy="237.5" r="3.5" class="fill-b"/>
<circle cx="522.5" cy="200" r="3.5" class="fill-b"/>
<circle cx="570" cy="50.2" r="3.5" class="fill-b"/>
<text x="578" y="54.2" class="fig-small">LSTM 8.4M</text>
<polyline points="380,249.4 427.5,247.7 475,240.6 522.5,212.5 570,100.1" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="380" cy="249.4" r="3.5" class="fill-c"/>
<circle cx="427.5" cy="247.7" r="3.5" class="fill-c"/>
<circle cx="475" cy="240.6" r="3.5" class="fill-c"/>
<circle cx="522.5" cy="212.5" r="3.5" class="fill-c"/>
<circle cx="570" cy="100.1" r="3.5" class="fill-c"/>
<text x="578" y="104.1" class="fig-small">GRU 6.3M</text>
<polyline points="380,249.8 427.5,249.2 475,246.9 522.5,237.5 570,200" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="380" cy="249.8" r="3.5" class="fill-a"/>
<circle cx="427.5" cy="249.2" r="3.5" class="fill-a"/>
<circle cx="475" cy="246.9" r="3.5" class="fill-a"/>
<circle cx="522.5" cy="237.5" r="3.5" class="fill-a"/>
<circle cx="570" cy="200" r="3.5" class="fill-a"/>
<text x="578" y="204" class="fig-small">RNN 2.1M</text>
</svg>
Figure 4. Left, parameter counts at width 512 for the three cells. Right, the same counts as the width doubles from 64 to 1024. Notice that each doubling multiplies the counts by almost 4, because the square of the width dominates, so width is expensive.
```

Each weight is used once per step per example, so counting 2 floating-point operations per multiply-add, a GRU step costs about $6 (d_x d_h + d_h^2)$ operations per example and an LSTM step about $8 (d_x d_h + d_h^2)$. The backward pass stores six arrays of shape $(n, d_h)$ per LSTM step and four per GRU step, and step $t$ waits for step $t - 1$.

> [!KEY]
> A GRU keeps the belt's benefit with a simpler cell: one state and one update gate, whose shares $1 - z$ and $z$ blend old and new so the gradient survives. A reset gate hides the past from the candidate when it would mislead. The cell needs three weight sets instead of four, which saves a quarter of the LSTM's numbers, at the price of tying forgetting to writing.

> [!RECAP]
> - The LSTM cell state feeds the next step and the hidden output, so $dc_{\text{total}} = dc_{\text{new}} + dh_{\text{new}} \odot o \odot (1 - \tanh^2)$, and $dc_{\text{prev}} = dc_{\text{total}} \odot f$ inside one step.
> - Each gate's gradient passes through its squash slope to the pre-activation. Weight gradients are $x^{\top} da$ and $h^{\top} da$, and $dx$ and $dh_{\text{prev}}$ sum one term per gate.
> - The GRU update is $(1 - z) \odot h + z \odot \tilde{h}$, a blend whose two shares add to 1.
> - The reset gate scales the old state before the candidate's matrix sees it, so you write `(r * h) @ Uh`.
> - A plain RNN has one weight set of $d_x d_h + d_h^2 + d_h$ numbers, a GRU three, and an LSTM four.

## Build it step by step

The starter gives you `lstm_cell_forward` and `_sigmoid`, already written. The checks test five functions in this order.

1. **`cell_state_gradient(dh_new, dc_new, o, tanh_c_new)`.** Takes the gradients on both outputs, the output gate, and $\tanh(c_{\text{new}})$, all of shape `(n, d_h)`, and returns $dc_{\text{total}}$ in the same shape. It is one line of arithmetic on arrays. The most common mistake is dropping the $dh_{\text{new}}$ term, or using $\tanh$ in place of its slope $1 - \tanh^2$.
2. **`lstm_cell_backward(dh_new, dc_new, cache)`.** Takes the two gradients and the cache from the forward pass, and returns `(dx, dh_prev, dc_prev, dparams)`. Call `cell_state_gradient`, form `do`, `df`, `di`, `dg`, and `dc_prev`, then the four pre-activation gradients, then loop over the gates for the weights, the biases, `dx`, and `dh_prev`. The most common mistake is using the gate where the slope is needed, so that $da_f$ is $df \odot f$ and not $df \odot f \odot (1 - f)$.
3. **`gru_update_blend(h, candidate, z)`.** Takes the old state, a candidate, and the update gate, all of shape `(n, d_h)`, and returns the blend in the same shape. Write `(1 - z) * h + z * candidate`. The most common mistake is swapping `z` and `1 - z`, which makes $z = 1$ keep the old state.
4. **`gru_cell_forward(x, h, params)`.** Takes the input `(n, d_x)`, the old state `(n, d_h)`, and the nine parameter arrays, and returns the new state. Compute `z` and `r` with `_sigmoid`, the candidate with `(r * h) @ Uh`, and finish with `gru_update_blend`. The most common mistake is `r * (h @ Uh)`.
5. **`parameter_count(kind, input_size, hidden_size)`.** Takes `"rnn"`, `"gru"`, or `"lstm"` and returns an `int`. Compute one weight set, then multiply by 1, 3, or 4, and raise `ValueError` for any other kind. The most common mistake is one bias for the whole cell instead of one per weight set.

## Where this breaks

Everything in the backward half is the gradient through one step. A sequence chains many steps, each step's `dh_prev` and `dc_prev` becoming the next `dh_new` and `dc_new`, and the belt carries `dc_prev` along. The path through `dh_prev` still passes through weight matrices and squashes, so it can fade or blow up, which is why [clipping](/lessons/050-vanishing-gradients-and-clipping) remains useful for both cells. A gate that saturates near 0 or 1 passes back almost nothing, which is the price of a decisive gate. Gating slows the fade without removing it, so very long memories stay hard for both cells, and neither removes the sequential chain: step $t$ needs the state from step $t - 1$, so long sequences cannot be computed in parallel. [Attention](/lessons/053-attention-as-soft-lookup) takes up both problems by letting a step read any earlier state directly.

## The exercise

You build `cell_state_gradient`, `lstm_cell_backward`, `gru_update_blend`, `gru_cell_forward`, and `parameter_count`. The checks confirm that $dc_{\text{total}}$ is 0.6004 on the worked example and adds both paths, that the backward pass returns twelve parameter gradients of the right shapes and gives $dc_{\text{prev}} = 0.5404$ and a forget-bias gradient of 0.0216, and that every gradient matches finite differences. They confirm that with no gradient on $h_{\text{new}}$ the belt passes back exactly $dc_{\text{new}} \odot f$. For the GRU they check that the blend returns the old state at $z = 0$ and the candidate at $z = 1$, that the worked step gives 0.7546, that the reset gate acts before the matrix on a swap-matrix example, and that the counts are 10, 30, and 40 at width 2 and 524,800, 1,574,400, and 2,099,200 at width 512.
