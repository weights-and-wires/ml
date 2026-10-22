---
title: Count the parameters, flops, and memory of a network
slug: 060-count-parameters-and-flops
part: "Part 4 — Neural networks"
summary: "Read a network's cost off its layer sizes: parameters for memory, flops for time, and stored activations for training."
minutes: 35
gpu: false
objectives:
  - Count the parameters of a network from its layer sizes, and say what each term is.
  - Count flops with the convention of 2 per multiply-add, 1 per ReLU output, and 9 per GELU output.
  - Count the activations a network must store for one input, and say why training needs them.
  - Explain why doubling the width roughly quadruples the cost while doubling the depth roughly doubles it.
prereqs:
  - 057-why-layers-need-a-bend
  - 058-forward-pass-and-gelu
  - 059-width-adds-depth-multiplies
exercise: 060-count-parameters-and-flops
---

# Count the parameters, flops, and memory of a network

You have a network with layers of size 4, 8, 8, and 3. A friend builds one with the hidden layers twice as wide, 4, 16, 16, and 3. Both fit on your laptop, but the second one is almost three times as expensive, and the gap grows toward four times as the layers get wider. How can you tell that from the layer sizes alone, before you run anything?

Without a way to count, cost is a surprise. You find out a model is too big when the memory runs out or the job takes all night, and then you do not know which layer to shrink. This chapter gives you the counting rules, so you can budget a network on paper. You will count three things: parameters, which set how much memory the weights take, flops, which set how long one prediction takes, and stored activations, which set the extra memory that training needs.

## What you already know

- A layer holds a weight matrix `W` of shape `(out, in)` and a bias vector `b` of length `out`, and a unit is one row of `W` ("Why a stack of layers needs a bend").
- A forward pass applies each layer, then its bend, in order ("Run a network forward, with a sharp or smooth bend").
- ReLU is one comparison, and GELU needs a cube, a tanh, and several more operations.
- Stacking layers can multiply the number of pieces a network makes ("Width adds bends and depth multiplies them").
- A list of layers is a list of dicts with `"W"`, `"b"`, and `"activation"`.

## Parameters: the numbers training can change

A **parameter** is one number the network learns: every entry of every `W` and every entry of every `b`. Count them because they set the memory the weights occupy, and as a rough guide, how much the network can learn. A layer with `W` of shape `(out, in)` has `out * in` weights and `out` biases.

Take the 4-8-8-3 network. Its three layers have shapes $(8, 4)$, $(8, 8)$, and $(3, 8)$, so

$$
(32 + 8) + (64 + 8) + (24 + 3) = 40 + 72 + 27 = 139 .
$$

Read it aloud: "weights plus biases, layer by layer, added up." The same network in the doubled width, 4-16-16-3, has $(64+16) + (256+16) + (48+3) = 403$. In code:

```python
sum(layer["W"].size + layer["b"].size for layer in params)
```

## Flops: the arithmetic of one prediction

A **flop** is one floating-point operation, such as one addition or one multiplication. Counting them tells you how long a prediction takes on a given chip. Each of the `in * out` weights in a matrix product takes part in one multiplication and one addition, so the product costs $2 \cdot \text{in} \cdot \text{out}$ flops. This is a counting convention: a real chip fuses and reorders operations, so treat the totals as comparisons, not stopwatch readings. Then add `out` for the bias additions, and the cost of the bend on each output.

The bend costs depend on what it does. ReLU makes one comparison per output, so charge 1. GELU, written out, takes: `x**3` as two multiplications, one multiplication by 0.044715, one addition, one multiplication by $\sqrt{2/\pi}$, one tanh (counted as a single flop here), one addition of 1, one multiplication by `x`, and one by 0.5. That is 9. Identity does nothing, so charge 0.

For the same 4-8-8-3 network with ReLU, ReLU, then identity, tally each layer as matrix product, plus bias, plus bend:

- Layer 1: $2\cdot 4\cdot 8 + 8 + 8 = 64 + 8 + 8 = 80$.
- Layer 2: $2\cdot 8\cdot 8 + 8 + 8 = 128 + 8 + 8 = 144$.
- Layer 3: $2\cdot 8\cdot 3 + 3 + 0 = 48 + 3 + 0 = 51$.

The total is $80 + 144 + 51 = 275$. As a formula for one layer:

$$
\text{flops} = 2\,\text{in}\cdot\text{out} + \text{out} + c_{\text{act}}\cdot\text{out} ,\qquad c_{\text{relu}} = 1,\; c_{\text{gelu}} = 9,\; c_{\text{identity}} = 0 .
$$

Figure 1 draws both counts, layer by layer.

```figure
<svg viewBox="0 0 600 258" role="img" aria-label="Two stacked bar charts for the 4-8-8-3 network. Parameters per layer are 40, 72 and 27, split into weights and biases. Flops per layer are 80, 144 and 51, split into the matrix product, biases and ReLU.">

<text x="155.0" y="22.0" text-anchor="middle" class="fig-bold">Parameters</text>
<line x1="30.0" y1="190.0" x2="280.0" y2="190.0" class="stroke-muted"/>
<rect x="30.0" y="160.1" width="50.0" height="29.9" class="fill-a"/>
<text x="55.0" y="179.1" text-anchor="middle" class="fig-small">32</text>
<rect x="30.0" y="152.7" width="50.0" height="7.5" class="fill-d"/>
<text x="55.0" y="146.7" text-anchor="middle" class="fig-bold">total 40</text>
<text x="55.0" y="206.0" text-anchor="middle" class="fig-small">layer 1</text>
<rect x="115.0" y="130.3" width="50.0" height="59.7" class="fill-a"/>
<text x="140.0" y="164.1" text-anchor="middle" class="fig-small">64</text>
<rect x="115.0" y="122.8" width="50.0" height="7.5" class="fill-d"/>
<text x="140.0" y="116.8" text-anchor="middle" class="fig-bold">total 72</text>
<text x="140.0" y="206.0" text-anchor="middle" class="fig-small">layer 2</text>
<rect x="200.0" y="167.6" width="50.0" height="22.4" class="fill-a"/>
<text x="225.0" y="182.8" text-anchor="middle" class="fig-small">24</text>
<rect x="200.0" y="164.8" width="50.0" height="2.8" class="fill-d"/>
<text x="225.0" y="158.8" text-anchor="middle" class="fig-bold">total 27</text>
<text x="225.0" y="206.0" text-anchor="middle" class="fig-small">layer 3</text>
<text x="445.0" y="22.0" text-anchor="middle" class="fig-bold">Flops</text>
<line x1="320.0" y1="190.0" x2="570.0" y2="190.0" class="stroke-muted"/>
<rect x="320.0" y="130.3" width="50.0" height="59.7" class="fill-a"/>
<text x="345.0" y="164.1" text-anchor="middle" class="fig-small">64</text>
<rect x="320.0" y="122.8" width="50.0" height="7.5" class="fill-d"/>
<rect x="320.0" y="115.3" width="50.0" height="7.5" class="fill-b"/>
<text x="345.0" y="109.3" text-anchor="middle" class="fig-bold">total 80</text>
<text x="345.0" y="206.0" text-anchor="middle" class="fig-small">layer 1</text>
<rect x="405.0" y="70.5" width="50.0" height="119.5" class="fill-a"/>
<text x="430.0" y="134.3" text-anchor="middle" class="fig-small">128</text>
<rect x="405.0" y="63.1" width="50.0" height="7.5" class="fill-d"/>
<rect x="405.0" y="55.6" width="50.0" height="7.5" class="fill-b"/>
<text x="430.0" y="49.6" text-anchor="middle" class="fig-bold">total 144</text>
<text x="430.0" y="206.0" text-anchor="middle" class="fig-small">layer 2</text>
<rect x="490.0" y="145.2" width="50.0" height="44.8" class="fill-a"/>
<text x="515.0" y="171.6" text-anchor="middle" class="fig-small">48</text>
<rect x="490.0" y="142.4" width="50.0" height="2.8" class="fill-d"/>
<text x="515.0" y="136.4" text-anchor="middle" class="fig-bold">total 51</text>
<text x="515.0" y="206.0" text-anchor="middle" class="fig-small">layer 3</text>
<rect x="30.0" y="236.0" width="12.0" height="12.0" class="fill-a"/>
<text x="48.0" y="246.0" text-anchor="start" class="fig-small">weights, or the matrix product</text>
<rect x="250.0" y="236.0" width="12.0" height="12.0" class="fill-d"/>
<text x="268.0" y="246.0" text-anchor="start" class="fig-small">biases</text>
<rect x="340.0" y="236.0" width="12.0" height="12.0" class="fill-b"/>
<text x="358.0" y="246.0" text-anchor="start" class="fig-small">ReLU</text>
</svg>
Figure 1. Where the 139 parameters and 275 flops of the 4-8-8-3 network sit. The middle layer is the biggest, because both its input and output are 8 wide. Notice that the matrix product, not the bias or the ReLU, is most of the flops in every layer.
```

The matrix product is most of the flops in every layer, and the middle layer dominates because both of its sides are 8 wide. In code, read the shape as `out_dim, in_dim = layer["W"].shape`. The name `in` is a Python keyword, so call it `in_dim`.

> [!WARNING]
> `W` has shape `(out, in)`. If you read the shape backwards, the product term $2 \cdot \text{in} \cdot \text{out}$ still comes out right, because it is symmetric, but the bias and bend terms use `out` and come out wrong.

## Activations: what training must keep

A third cost appears when you train. To adjust the weights, a later chapter walks the network in reverse, in what is called the **backward pass**, and that walk needs the values each layer produced on the way forward. So training keeps the input and every layer's output instead of throwing each one away after the next layer has used it. For one input, the 4-8-8-3 network holds $4 + 8 + 8 + 3 = 23$ numbers. For a batch of 100 inputs it holds 2300. Real training code often also keeps each pre-activation, which roughly doubles the hidden part, so treat 23 as a floor and as a counting convention. Parameters, by contrast, do not depend on the batch.

```python
in_dim + sum(layer["W"].shape[0] for layer in params)
```

## Why width is expensive and depth is cheap

Look at the weights as areas. A layer with $m$ inputs and $m$ outputs has an $m \times m$ matrix, so doubling $m$ quadruples its area, as in Figure 2. Doubling the depth only adds more layers of the same size, so cost doubles.

```figure
<svg viewBox="0 0 600 238" role="img" aria-label="Three squares of side 4, 8 and 16 cells, with 16, 64 and 256 cells inside, showing that the number of weights in a square layer grows with the square of its width.">

<rect x="50.0" y="158.0" width="32.0" height="32.0" class="fill-soft-a stroke-a"/>
<line x1="58.0" y1="158.0" x2="58.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="50.0" y1="166.0" x2="82.0" y2="166.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="66.0" y1="158.0" x2="66.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="50.0" y1="174.0" x2="82.0" y2="174.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="74.0" y1="158.0" x2="74.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="50.0" y1="182.0" x2="82.0" y2="182.0" class="stroke-grid" stroke-width="0.6"/>
<rect x="50.0" y="158.0" width="32.0" height="32.0" class="stroke-a" fill="none" stroke-width="1.6"/>
<text x="66.0" y="208.0" text-anchor="middle" class="fig-bold">4 × 4</text>
<text x="66.0" y="226.0" text-anchor="middle" class="fig-small">16 weights</text>
<rect x="150.0" y="126.0" width="64.0" height="64.0" class="fill-soft-a stroke-a"/>
<line x1="158.0" y1="126.0" x2="158.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="134.0" x2="214.0" y2="134.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="166.0" y1="126.0" x2="166.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="142.0" x2="214.0" y2="142.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="174.0" y1="126.0" x2="174.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="150.0" x2="214.0" y2="150.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="182.0" y1="126.0" x2="182.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="158.0" x2="214.0" y2="158.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="190.0" y1="126.0" x2="190.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="166.0" x2="214.0" y2="166.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="198.0" y1="126.0" x2="198.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="174.0" x2="214.0" y2="174.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="206.0" y1="126.0" x2="206.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="150.0" y1="182.0" x2="214.0" y2="182.0" class="stroke-grid" stroke-width="0.6"/>
<rect x="150.0" y="126.0" width="64.0" height="64.0" class="stroke-a" fill="none" stroke-width="1.6"/>
<text x="182.0" y="208.0" text-anchor="middle" class="fig-bold">8 × 8</text>
<text x="182.0" y="226.0" text-anchor="middle" class="fig-small">64 weights</text>
<rect x="280.0" y="62.0" width="128.0" height="128.0" class="fill-soft-a stroke-a"/>
<line x1="288.0" y1="62.0" x2="288.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="70.0" x2="408.0" y2="70.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="296.0" y1="62.0" x2="296.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="78.0" x2="408.0" y2="78.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="304.0" y1="62.0" x2="304.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="86.0" x2="408.0" y2="86.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="312.0" y1="62.0" x2="312.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="94.0" x2="408.0" y2="94.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="320.0" y1="62.0" x2="320.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="102.0" x2="408.0" y2="102.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="328.0" y1="62.0" x2="328.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="110.0" x2="408.0" y2="110.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="336.0" y1="62.0" x2="336.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="118.0" x2="408.0" y2="118.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="344.0" y1="62.0" x2="344.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="126.0" x2="408.0" y2="126.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="352.0" y1="62.0" x2="352.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="134.0" x2="408.0" y2="134.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="360.0" y1="62.0" x2="360.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="142.0" x2="408.0" y2="142.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="368.0" y1="62.0" x2="368.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="150.0" x2="408.0" y2="150.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="376.0" y1="62.0" x2="376.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="158.0" x2="408.0" y2="158.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="384.0" y1="62.0" x2="384.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="166.0" x2="408.0" y2="166.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="392.0" y1="62.0" x2="392.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="174.0" x2="408.0" y2="174.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="400.0" y1="62.0" x2="400.0" y2="190.0" class="stroke-grid" stroke-width="0.6"/>
<line x1="280.0" y1="182.0" x2="408.0" y2="182.0" class="stroke-grid" stroke-width="0.6"/>
<rect x="280.0" y="62.0" width="128.0" height="128.0" class="stroke-a" fill="none" stroke-width="1.6"/>
<text x="344.0" y="208.0" text-anchor="middle" class="fig-bold">16 × 16</text>
<text x="344.0" y="226.0" text-anchor="middle" class="fig-small">256 weights</text>
<text x="450.0" y="95.0" text-anchor="start" class="fig-small">Each side doubles,</text>
<text x="450.0" y="113.0" text-anchor="start" class="fig-small">the area quadruples:</text>
<text x="450.0" y="131.0" text-anchor="start" class="fig-small">16 → 64 → 256</text>
<text x="450.0" y="149.0" text-anchor="start" class="fig-small">weights.</text>
<text x="300.0" y="18.0" text-anchor="middle" class="fig-bold">One layer's weight matrix, out rows by in columns, at three widths</text>
</svg>
Figure 2. A layer with out units and in inputs has out × in weights, which is the area of its matrix. Notice that the 16 × 16 square holds four times the cells of the 8 × 8 square even though its side is only twice as long.
```

For a stack of $L$ square layers of width $m$, large against the input and output sizes, the costs change like this:

| Change | Parameters | Flops per input | Flops for a whole batch |
|---|---|---|---|
| Double the width $m$ | About 4 times | About 4 times | About 4 times |
| Double the depth $L$ | About 2 times | About 2 times | About 2 times |
| Double the batch size | Unchanged | Unchanged | 2 times |

For a check, take layers 4-64-64-3 and 4-128-128-3. They have 4675 and 17539 parameters, a ratio of 3.75, and 9347 and 35075 flops, also 3.75. The ratio sits below 4 because the 4 inputs and 3 outputs do not grow, and it approaches 4 as the width grows.

This is the quantitative case for the last chapter's sawtooth. At depth $k$ it has $4 + 6(k-1) + 3 = 6k + 1$ parameters. A one-hidden-layer network with the $2^k$ units it needs has $2^k$ weights in, $2^k$ constants, $2^k$ weights out, and one output constant, so $3\cdot 2^k + 1$. They match at $k = 1$ (7) and $k = 2$ (13), and then part ways: at $k = 10$ it is 61 against 3073, shown in Figure 3. Depth reuses each unit in a chain, and width pays for every bend separately.

```figure
<svg viewBox="0 0 600 246" role="img" aria-label="Two lines on a log scale for k from 1 to 10. The depth-k sawtooth network has 6k plus 1 parameters, reaching 61. One hidden layer with 2 to the k units has 3 times 2 to the k plus 1 parameters, reaching 3073.">

<line x1="70.0" y1="183.9" x2="560.0" y2="183.9" class="stroke-grid"/>
<text x="62.0" y="187.9" text-anchor="end" class="fig-small">10</text>
<line x1="70.0" y1="130.6" x2="560.0" y2="130.6" class="stroke-grid"/>
<text x="62.0" y="134.6" text-anchor="end" class="fig-small">100</text>
<line x1="70.0" y1="77.3" x2="560.0" y2="77.3" class="stroke-grid"/>
<text x="62.0" y="81.3" text-anchor="end" class="fig-small">1000</text>
<line x1="70.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted"/>
<text x="70.0" y="216.0" text-anchor="middle" class="fig-small">1</text>
<text x="124.4" y="216.0" text-anchor="middle" class="fig-small">2</text>
<text x="178.9" y="216.0" text-anchor="middle" class="fig-small">3</text>
<text x="233.3" y="216.0" text-anchor="middle" class="fig-small">4</text>
<text x="287.8" y="216.0" text-anchor="middle" class="fig-small">5</text>
<text x="342.2" y="216.0" text-anchor="middle" class="fig-small">6</text>
<text x="396.7" y="216.0" text-anchor="middle" class="fig-small">7</text>
<text x="451.1" y="216.0" text-anchor="middle" class="fig-small">8</text>
<text x="505.6" y="216.0" text-anchor="middle" class="fig-small">9</text>
<text x="560.0" y="216.0" text-anchor="middle" class="fig-small">10</text>
<text x="315.0" y="234.0" text-anchor="middle" class="fig-small">k = number of folds (so 2^k pieces)</text>
<path d="M70.0,192.2 L124.4,177.9 L178.9,162.7 L233.3,147.1 L287.8,131.3 L342.2,115.4 L396.7,99.4 L451.1,83.4 L505.6,67.3 L560.0,51.3" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M70.0,192.2 L124.4,177.9 L178.9,169.1 L233.3,162.7 L287.8,157.7 L342.2,153.6 L396.7,150.2 L451.1,147.1 L505.6,144.5 L560.0,142.1" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="70.0" cy="192.2" r="3.5" class="fill-a"/>
<circle cx="124.4" cy="177.9" r="3.5" class="fill-a"/>
<circle cx="178.9" cy="162.7" r="3.5" class="fill-a"/>
<circle cx="233.3" cy="147.1" r="3.5" class="fill-a"/>
<circle cx="287.8" cy="131.3" r="3.5" class="fill-a"/>
<circle cx="342.2" cy="115.4" r="3.5" class="fill-a"/>
<circle cx="396.7" cy="99.4" r="3.5" class="fill-a"/>
<circle cx="451.1" cy="83.4" r="3.5" class="fill-a"/>
<circle cx="505.6" cy="67.3" r="3.5" class="fill-a"/>
<circle cx="560.0" cy="51.3" r="3.5" class="fill-a"/>
<circle cx="70.0" cy="192.2" r="3.5" class="fill-b"/>
<circle cx="124.4" cy="177.9" r="3.5" class="fill-b"/>
<circle cx="178.9" cy="169.1" r="3.5" class="fill-b"/>
<circle cx="233.3" cy="162.7" r="3.5" class="fill-b"/>
<circle cx="287.8" cy="157.7" r="3.5" class="fill-b"/>
<circle cx="342.2" cy="153.6" r="3.5" class="fill-b"/>
<circle cx="396.7" cy="150.2" r="3.5" class="fill-b"/>
<circle cx="451.1" cy="147.1" r="3.5" class="fill-b"/>
<circle cx="505.6" cy="144.5" r="3.5" class="fill-b"/>
<circle cx="560.0" cy="142.1" r="3.5" class="fill-b"/>
<text x="552.0" y="43.3" text-anchor="end" class="fig-small">3073</text>
<text x="552.0" y="134.1" text-anchor="end" class="fig-small">61</text>
<text x="91.8" y="67.9" text-anchor="start" class="fig-small">one hidden layer: 3·2^k + 1 (blue)</text>
<text x="287.8" y="179.7" text-anchor="start" class="fig-small">depth k: 6k + 1 (orange)</text>
<text x="300.0" y="18.0" text-anchor="middle" class="fig-bold">Parameters needed for 2^k pieces, log scale</text>
</svg>
Figure 3. Parameters of the sawtooth network against a one-hidden-layer network with the 2^k units it needs, both making 2^k pieces. The two are equal at k = 1 and 2 (7 and 13). Notice that the one-layer line climbs in a straight line on this log scale, which means it grows by the same factor with every fold, while the depth line flattens out, ending at 61 against 3073.
```

> [!KEY]
> A network's cost follows from its layer shapes alone: parameters are `out * in + out` per layer, flops are $2\,\text{in}\cdot\text{out} + \text{out} + c\cdot\text{out}$, and training keeps the input and every layer's output. Doubling the width roughly quadruples the cost, so the 4-8-8-3 network and its doubled twin differ by about 3 times already, and by nearly 4 times at larger widths.

> [!RECAP]
> - Parameters per layer are `out * in + out`, and the network total is their sum.
> - Flops per layer are `2 * in * out`, plus `out` for the bias, plus `out` times 1 for ReLU, 9 for GELU, or 0 for identity.
> - Read the shape as `out_dim, in_dim = W.shape`, and charge nothing for identity.
> - Training also stores the input and every layer's output: `in_dim` plus the sum of the `out` sizes.
> - The sawtooth has $6k + 1$ parameters and one hidden layer needs $3\cdot 2^k + 1$.

## Build it step by step

`sawtooth_network` from the last chapter is already in the starter, so the checks can use it. Build three functions.

1. **`parameter_count(params)`** returns the total number of parameters as an integer. Add the size of `W` and the size of `b` for every layer. A common mistake is to count only the weights and forget the biases. Write it like this:

    ```python
    def parameter_count(params):
        # Add both arrays on every layer.
    ```

2. **`flop_count(params)`** returns the total flops as an integer. For each layer read `out_dim, in_dim = W.shape`, add the product, the bias, and the bend cost per output, and sum over layers. Look up the bend cost by activation name: 1, 9, or 0. A common mistake is to charge identity a flop or to swap `in_dim` and `out_dim`. Write it like this:

    ```python
    def flop_count(params):
        # 2 * in * out + out + bend_cost * out, summed over layers.
    ```

3. **`activation_memory(params, in_dim)`** returns the number of floats held for one input: the input plus every layer's output. It needs only the shapes. A common mistake is to leave out the input. Write it like this:

    ```python
    def activation_memory(params, in_dim):
        # in_dim plus the output size of every layer.
    ```

## Where this breaks

These counts are a map, not a stopwatch. Real speed also depends on how fast memory moves, and a very wide layer can run out of memory long before it runs out of time. Cost is also not the only danger of depth: signals can shrink or grow as they pass through many layers, and a ReLU unit can get stuck at 0. Later chapters on starting values and normalisation deal with both.

## The exercise

You implement `parameter_count`, `flop_count`, and `activation_memory`. The checks count the 4-8-8-3 network by hand, with 139 parameters, 275 flops, and 23 stored activations. They check the sawtooth at $6k+1$ parameters, a network with a GELU layer, a layer that is not square so a swapped shape is caught, and that the flops of two stacked networks equal the sum of their parts.
