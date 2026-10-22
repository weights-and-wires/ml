---
title: Backpropagation through one LSTM step
slug: 091-lstm-backward-step
part: "Part 5 — Vision and sequences"
summary: The cell state feeds two places, so its gradient is a sum of two paths, and every product node then passes back the upstream gradient times the other factor.
minutes: 40
gpu: false
objectives:
  - Define upstream gradient and local derivative, and state the backward rules for a sum node and a product node.
  - Explain why the cell state receives gradient from two paths and why its total gradient is their sum.
  - Work the backward pass of one LSTM unit by hand and show that the gradient reaching the previous cell is the total times the forget gate.
  - Write lstm_cell_backward in numpy and check it against finite differences.
prereqs:
  - 061-backprop-the-graph-and-reverse-order
  - 062-backprop-array-operations
  - 089-why-rnns-forget-the-belt
  - 090-gates-and-the-lstm-update
exercise: 091-lstm-backward-step
---

# Backpropagation through one LSTM step

The loss has looked at the output of one LSTM step and found it wanting. It says: the shown value $h_{\text{new}}$ should move, with slope 1, and the new cell $c_{\text{new}}$ should move too, with slope 0.2, because the next step reads it. Which way should each of the three gates nudge? How much should each weight change, and how much of this error belongs to the previous cell and the previous shown value, so it can be passed on to the step before?

You could answer by brute force: nudge one weight a little, rerun the cell, and see how the loss moves. A cell with 512 hidden units and 512 inputs holds about two million weights, so that is four million forward passes for a single gradient. Backpropagation answers every question in one backward sweep, because each operation in the cell needs only its own small slope.

This chapter walks that sweep through the cell from the last chapter, with the same numbers: a gradient of 1 on $h_{\text{new}}$ and 0.2 on $c_{\text{new}}$. The idea this chapter adds is one place where gradient arrives from two directions, and the sum that follows.

## What you already know

Five ideas carry this chapter:

- The forward step of an LSTM cell, with its three gates, its candidate, and its two outputs (see "Gates and the LSTM memory update").
- Backpropagation sends a gradient backward through a graph one operation at a time, and a value used in two places collects the sum of the gradients from both (see "Walking slopes backwards through a calculation").
- Each array operation has a small backward rule that keeps the gradient the same shape as the value (see "The backward rule for each array operation").
- The sigmoid has slope $\sigma(1 - \sigma)$ and tanh has slope $1 - \tanh^2$.
- Along the belt, the gradient across one step is multiplied by the forget gate (see "Why a recurrent net forgets, and the belt that fixes it").

## Two words and two rules

Two terms keep the rest short. The **upstream gradient** at a node is the gradient that arrives from the loss side: how much the loss changes per unit change in that node's output. The **local derivative** is the slope of the node's output with respect to one of its inputs, with the other inputs held still. Backward through a node, the gradient on an input is the upstream gradient times the local derivative.

Two node types cover the whole LSTM. A **sum** node, $y = a + b$, has a local derivative of 1 for each input, so it hands its upstream gradient to both inputs unchanged. A **product** node, $y = a \cdot b$, has local derivative $b$ for input $a$ and $a$ for input $b$, because nudging $a$ by a small amount changes $y$ by $b$ times that amount. So the gradient on one input of a product is the upstream gradient times the other input. Figure 1 draws the cell as a graph, with the forward values in blue and the backward gradients in orange.

```figure
<svg viewBox="0 0 700 335" role="img" aria-label="A computation graph of one LSTM step. Forward arrows run from c, f, i, g through two products and a plus to the new cell, then through tanh and a product with o to the hidden output. Backward arrows run the other way, and the new cell receives gradient from two places">
<defs><marker id="c091-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c091-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="121.2" y1="54.8" x2="185.9" y2="73.3" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="183.4" y1="81.9" x2="118.8" y2="63.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="118.8" y1="100.5" x2="183.4" y2="82.1" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="185.9" y1="90.7" x2="121.2" y2="109.2" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="215.7" y1="87.5" x2="289.5" y2="139.1" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="284.3" y1="146.5" x2="210.5" y2="94.9" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="121.2" y1="194.8" x2="185.9" y2="213.3" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="183.4" y1="221.9" x2="118.8" y2="203.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="118.8" y1="240.5" x2="183.4" y2="222.1" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="185.9" y1="230.7" x2="121.2" y2="249.2" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="210.5" y1="209.1" x2="284.3" y2="157.5" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="289.5" y1="164.9" x2="215.7" y2="216.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="316" y1="147.5" x2="348" y2="147.5" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="348" y1="156.5" x2="316" y2="156.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="394.5" y1="167" x2="394.5" y2="214" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="385.5" y1="214" x2="385.5" y2="167" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="432" y1="224.5" x2="494" y2="224.5" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="494" y1="233.5" x2="432" y2="233.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="505.5" y1="291" x2="505.5" y2="245" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="514.5" y1="245" x2="514.5" y2="291" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="526" y1="224.5" x2="558" y2="224.5" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="558" y1="233.5" x2="526" y2="233.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="432" y1="147.5" x2="674" y2="147.5" class="stroke-a" marker-end="url(#c091-arrow-a)"/>
<line x1="674" y1="156.5" x2="432" y2="156.5" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
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

The first route's gradient is given: $dc_{\text{new}} = 0.2$. The second route runs back through the product with $o$ and then through tanh. Start with $dh_{\text{new}} = 1$. The product with $o = 0.5$ passes back $1 \times 0.5 = 0.5$, the upstream times the other factor. The tanh then multiplies by its slope, $1 - \tanh^2(0.48) = 1 - 0.4462^2 = 0.8009$, so the second route delivers $0.5 \times 0.8009 = 0.4004$. The total gradient on the new cell is

$$
dc_{\text{total}} = dc_{\text{new}} + dh_{\text{new}} \odot o \odot \left(1 - \tanh^2(c_{\text{new}})\right)
$$

Read it aloud: the total gradient on the new cell is the gradient from the next step plus the gradient from the shown value, which is $dh_{\text{new}}$ times the output gate times the tanh slope. With the numbers, $0.2 + 0.4004 = 0.6004$. In numpy:

```python
dc_total = dc_new + dh_new * o * (1.0 - tanh_c_new ** 2)
```

The most common mistake is dropping the $dh_{\text{new}}$ term, which treats the cell as feeding only the next step.

## Down through the products

Now $dc_{\text{total}}$ flows back through the sum and the two products that built $c_{\text{new}} = f \odot c + i \odot g$. The sum hands $dc_{\text{total}}$ to both products unchanged. Each product then gives each input the upstream gradient times the other input. With $c = 0.4$, $g = 0.6$, $f = 0.9$, and $i = 0.2$:

$$
df = dc_{\text{total}} \odot c, \quad
di = dc_{\text{total}} \odot g, \quad
dg = dc_{\text{total}} \odot i, \quad
dc_{\text{prev}} = dc_{\text{total}} \odot f
$$

The last one is the belt from the previous chapter. The gradient reaching the previous cell is the total times the forget gate, so $dc_{\text{prev}} = 0.6004 \times 0.9 = 0.5404$. Inside a single step this is exact, because $c$ enters $c_{\text{new}}$ only through the product $f \odot c$. The output gate gets its own share from the other product, $do = dh_{\text{new}} \odot \tanh(c_{\text{new}}) = 1 \times 0.4462 = 0.4462$. The other three come out as $df = 0.6004 \times 0.4 = 0.2402$, $di = 0.6004 \times 0.6 = 0.3603$, and $dg = 0.6004 \times 0.2 = 0.1201$. Figure 2 follows all of these numbers along the belt.

```figure
<svg viewBox="0 0 660 345" role="img" aria-label="The worked example as a chain from c to h_new. Each box shows its forward value and its gradient. The gradient of the new cell is 0.2 plus 0.4004, which is 0.6004, and the gradient of the previous cell is 0.5404">
<defs><marker id="c091-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c091-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="140" y1="82" x2="124" y2="82" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="272" y1="82" x2="256" y2="82" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="404" y1="82" x2="388" y2="82" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<line x1="536" y1="82" x2="520" y2="82" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<text x="136" y="150" text-anchor="middle" class="fig-small">× f</text>
<text x="268" y="150" text-anchor="middle" class="fig-small">× 1</text>
<text x="400" y="150" text-anchor="middle" class="fig-small">× 0.8009</text>
<text x="532" y="150" text-anchor="middle" class="fig-small">× o</text>
<line x1="330" y1="188" x2="330" y2="134" class="stroke-b" marker-end="url(#c091-arrow-b)"/>
<text x="340" y="160" class="fig-small">× 1</text>
<text x="10" y="292" class="fig-small">df = 0.6004 × c = 0.2402, then da_f = 0.2402 × 0.9 × 0.1 = 0.0216</text>
<text x="10" y="312" class="fig-small">di = 0.6004 × g = 0.3603, and dg = 0.6004 × i = 0.1201</text>
<text x="10" y="332" class="fig-small">do = dh_new × tanh(c_new) = 0.4462</text>
<text x="10" y="14" class="fig-small">Orange arrows carry the gradient back; each label is the local slope used on that edge</text>
</svg>
Figure 2. The worked example as a chain, with forward values and gradients in each box. Notice that the new cell's gradient is 0.2 plus 0.4004, that the gradient across the plus sign is unchanged, and that the previous cell ends up with 0.5404, which is the total times the forget gate of 0.9.
```

A caution about reach. Inside one step the factor on the belt is exactly $f$. Across many steps, the gates also read the previous shown value, which depends on the previous cell, so there is a second, smaller route from one cell to the next. Backpropagating the whole sequence includes it, and `lstm_cell_backward` hands it to the step before through `dh_prev`.

## Through the squashes

A gate is a sigmoid of a weighted sum. Call that weighted sum $x W + h U + b$ the **pre-activation**, because it comes before the squash. The weights see the pre-activation, not the gate, so the gradient must pass backward through the squash first. A sigmoid gate has local slope $f(1 - f)$ and the candidate has slope $1 - g^2$. Write $da$ for the gradient on a pre-activation:

$$
da_f = df \odot f \odot (1 - f), \qquad
da_g = dg \odot (1 - g^2)
$$

and likewise $da_i = di \odot i \odot (1 - i)$ and $da_o = do \odot o \odot (1 - o)$. Read it aloud: the gradient on the pre-activation is the gradient on the gate times the gate's own slope. For the forget gate, $da_f = 0.2402 \times 0.9 \times 0.1 = 0.0216$. The other three come out as $da_i = 0.0576$, $da_o = 0.1116$, and $da_g = 0.1201 \times (1 - 0.6^2) = 0.0769$. A gate near 0 or 1 has a slope near 0, so a saturated gate passes back very little, and the gradient that reaches the weights through that gate is small.

## Down to the weights and the inputs

Each pre-activation is $a = x W + h U + b$, so the rules from "The backward rule for each array operation" apply to each gate, once per gate:

$$
dW = x^{\top} da, \qquad dU = h^{\top} da, \qquad db = \textstyle\sum_{\text{rows}} da
$$

$$
dx = \sum_{\text{gates}} da \, W^{\top}, \qquad dh_{\text{prev}} = \sum_{\text{gates}} da \, U^{\top}
$$

Read it aloud: a weight's gradient is the input that fed it, transposed, times the gradient on the pre-activation, so a weight is blamed in proportion to how active its input was. The bias gradient adds up the rows of the batch, because one bias was added to every row. The input $x$ and the previous shown value $h$ each feed all four gates, so their gradients are the sum of one term per gate. In numpy, once for every gate:

```python
dW[name] = x.T @ da[name]
dx += da[name] @ params["W" + name].T
```

> [!TIP]
> Keep the four pre-activation gradients in a dict keyed by `"f"`, `"i"`, `"o"`, and `"g"`, and loop. The four gates then share one block of code, and a typo in one cannot hide.

## Check by finite differences

You can test the whole backward pass without trusting any derivation. Nudge one entry of `x` by a small $\epsilon$ up and down, rerun the forward step, and divide the change in loss by $2\epsilon$. That is the finite-difference gradient, and it must match the backward pass. Figure 3 plots the twelve entries of $dx$ on the test case, one point per entry.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Scatter plot of the analytic gradient of x against the finite-difference gradient for twelve entries, all lying on the diagonal">
<rect x="70" y="30" width="260" height="260" rx="0" class="fill-paper stroke-grid"/>
<text x="70" y="308" text-anchor="middle" class="fig-small">-0.1</text>
<text x="62" y="294" text-anchor="end" class="fig-small">-0.1</text>
<text x="200" y="308" text-anchor="middle" class="fig-small">0</text>
<text x="62" y="164" text-anchor="end" class="fig-small">0</text>
<text x="330" y="308" text-anchor="middle" class="fig-small">0.1</text>
<text x="62" y="34" text-anchor="end" class="fig-small">0.1</text>
<line x1="70" y1="290" x2="330" y2="30" class="stroke-muted dashed"/>
<circle cx="211.6" cy="148.4" r="5" class="fill-b"/>
<circle cx="236.4" cy="123.6" r="5" class="fill-b"/>
<circle cx="223.9" cy="136.1" r="5" class="fill-b"/>
<circle cx="181.6" cy="178.4" r="5" class="fill-b"/>
<circle cx="173.6" cy="186.4" r="5" class="fill-b"/>
<circle cx="131" cy="229" r="5" class="fill-b"/>
<circle cx="195.6" cy="164.4" r="5" class="fill-b"/>
<circle cx="309.8" cy="50.2" r="5" class="fill-b"/>
<circle cx="191.1" cy="168.9" r="5" class="fill-b"/>
<circle cx="178.6" cy="181.4" r="5" class="fill-b"/>
<circle cx="132.1" cy="227.9" r="5" class="fill-b"/>
<circle cx="191.6" cy="168.4" r="5" class="fill-b"/>
<text x="200" y="330" text-anchor="middle" class="fig-small">gradient from lstm_cell_backward</text>
<text x="62" y="20" class="fig-small">gradient from finite differences</text>
<text x="360" y="80" class="fig-bold">12 entries of dx on the test case</text>
<text x="360" y="104" class="fig-small">dashed line: the two agree exactly</text>
<text x="360" y="126" class="fig-small">largest gap: 1.9e-11</text>
</svg>
Figure 3. The backward-pass gradient of x against the finite-difference gradient, for the twelve entries of dx on one test case. Notice that every point sits on the dashed diagonal, with the largest gap near 2e-11, which is why the check passes.
```

> [!KEY]
> The loss reached $h_{\text{new}}$ and $c_{\text{new}}$, and the backward pass spreads it in a fixed order. The cell state has two outgoing paths, so its gradient is their sum, $dc_{\text{total}}$. Each product then passes back the upstream times the other factor, each squash multiplies by its own slope, and the weights, the input, and the previous state read off what they need. The previous cell receives $dc_{\text{total}} \odot f$, the belt gradient from "Why a recurrent net forgets, and the belt that fixes it".

> [!RECAP]
> - Upstream gradient arrives from the loss side, and the local derivative is a node's own slope.
> - A sum node passes its upstream gradient to both inputs, and a product node passes the upstream times the other input.
> - The cell state feeds the next step and the hidden output, so $dc_{\text{total}} = dc_{\text{new}} + dh_{\text{new}} \odot o \odot (1 - \tanh^2)$.
> - $dc_{\text{prev}} = dc_{\text{total}} \odot f$ inside one step, and each gate's gradient passes through its squash slope to the pre-activation.
> - Weight gradients are $x^{\top} da$ and $h^{\top} da$, and $dx$ and $dh_{\text{prev}}$ sum one term per gate.

## Build it step by step

The starter gives you `lstm_cell_forward` from the last chapter, already written. The checks test two functions.

1. **`cell_state_gradient(dh_new, dc_new, o, tanh_c_new)`.** Takes the gradients on both outputs, the output gate, and $\tanh(c_{\text{new}})$, all of shape `(n, d_h)`, and returns $dc_{\text{total}}$ in the same shape. It is one line of arithmetic on arrays. The most common mistake is dropping the $dh_{\text{new}}$ term, or using $\tanh$ in place of its slope $1 - \tanh^2$.
2. **`lstm_cell_backward(dh_new, dc_new, cache)`.** Takes the two gradients and the cache from the forward pass, and returns `(dx, dh_prev, dc_prev, dparams)`. Call `cell_state_gradient`, form `do`, `df`, `di`, `dg`, and `dc_prev`, then the four pre-activation gradients, then loop over the gates for the weights, the biases, `dx`, and `dh_prev`. The most common mistake is using the gate where the slope is needed, so that $da_f$ is $df \odot f$ and not $df \odot f \odot (1 - f)$.

## Where this breaks

Everything here is the gradient through one step. A sequence chains many steps, each step's `dh_prev` and `dc_prev` becoming the next `dh_new` and `dc_new`, and the belt then carries the `dc_prev` along as in "Why a recurrent net forgets, and the belt that fixes it". The path through `dh_prev` still passes through weight matrices and squashes, so it can fade or blow up, and it is the reason clipping remains useful. A gate that saturates near 0 or 1 passes back almost nothing, which is the price of a decisive gate.

## The exercise

You build `cell_state_gradient` and `lstm_cell_backward`. The checks confirm that $dc_{\text{total}}$ is 0.6004 on the worked example and adds both paths, that the backward pass returns twelve parameter gradients of the right shapes, that it gives $dc_{\text{prev}} = 0.5404$ and a forget-bias gradient of 0.0216 on the worked numbers, and that every gradient matches finite differences. A last check confirms that with no gradient on $h_{\text{new}}$ the belt passes back exactly $dc_{\text{new}} \odot f$.
