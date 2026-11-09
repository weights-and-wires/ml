---
title: "Convolution: slide a filter, count the weights, count the view"
slug: 036-convolution-and-receptive-field
part: "Part 5 — Vision and sequences"
summary: Reuse one small filter at every position, then count a layer's weights, its output size, and how many input pixels each unit can see.
minutes: 75
gpu: false
objectives:
  - Slide a small filter over an image by hand, and say why one filter finds the same pattern at every position.
  - Count the weights of a dense layer, a convolution layer, and a depthwise separable layer, and explain the gap.
  - Write a convolution with plain loops, for one channel and for a batch of multi-channel images.
  - Count the output length of a convolution with padding, stride, and dilation, and choose the padding that keeps the size.
  - Compute the receptive field of every layer in a stack, and apply max pooling to a grid.
prereqs:
  - 028-multilayer-perceptrons
  - 030-softmax-and-whole-network-gradients
  - 035-dropout-and-label-smoothing
exercise: 036-convolution-and-receptive-field
---

# Convolution: slide a filter, count the weights, count the view

A colour photo, 224 pixels on a side with three colour channels, holds 224 × 224 × 3 = 150,528 numbers. Suppose you want a first layer with 1,000 units that reads that photo. A dense layer gives every unit one weight for every pixel, so the layer holds 150,528 × 1,000 + 1,000 = 150,529,000 numbers to learn. Memory is the smaller problem.

The larger problem is that a dense layer has no idea that the top-left corner and the bottom-right corner of a photo are the same kind of place. A cat ear at the top left is detected by some weights. A cat ear at the bottom right is detected by different weights, which start out random and must learn the same thing again from scratch. Shrinking the photo cuts the weights, but it throws away the detail that tells an ear from a whisker. A better question: can one tiny detector slide over the whole photo and report where it matches, using a few thousand weights in total?

It can, and the answer is a **convolution**. Two practical questions follow, and this chapter answers both. How large is the map after each layer, so that you know the shapes your code must handle? And does a unit in the last layer see enough of the picture to recognise a whole object? Both answers come from counting.

## What you already know

- A dense layer gives every output a weight for every input, plus one bias, so it has `n_in * n_out + n_out` numbers to learn ([multilayer perceptrons](/lessons/028-multilayer-perceptrons)).
- A dot product multiplies two lists entry by entry and adds the results.
- A grayscale image is a grid of numbers. A colour image is three such grids, called channels, for red, green, and blue.
- In numpy, `x[i:i + 3, j:j + 3]` cuts out a 3 by 3 block of a grid, and integer division `a // b` rounds down.
- A layer's output grid becomes the next layer's input grid.

## Look for the same pattern everywhere

Two facts about pictures make the dense layer wasteful. The first is that a useful pattern, such as an edge, a corner, or an ear, is small. To decide whether an edge is present, a unit needs to look at a few neighbouring pixels, not all 150,528. The second is that a pattern means the same thing wherever it appears, so one detector for it should be enough.

A **filter** (also called a **kernel**) is a small grid of weights that acts as that detector. You lay it on a patch of the image, called a **window**, multiply each weight by the pixel under it, and add everything up. The sum is large when the patch looks like the pattern the weights describe. Then you move the window one pixel and repeat. Figure 1 shows the idea on a 5 by 5 image.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="A 3 by 3 filter slides over a 5 by 5 image and fills a 3 by 3 output">
<defs>
<marker id="c034-arrow-1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<text x="24" y="24" class="fig-bold">Image (5 by 5)</text>
<rect x="24" y="58" width="34" height="34" class="fill-soft-c stroke-grid"/>
<text x="41" y="79.5" text-anchor="middle">0</text>
<rect x="58" y="58" width="34" height="34" class="fill-soft-c stroke-grid"/>
<text x="75" y="79.5" text-anchor="middle">0</text>
<rect x="92" y="58" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="109" y="79.5" text-anchor="middle">1</text>
<rect x="126" y="58" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="143" y="79.5" text-anchor="middle">1</text>
<rect x="160" y="58" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="177" y="79.5" text-anchor="middle">1</text>
<rect x="24" y="92" width="34" height="34" class="fill-soft-c stroke-grid"/>
<text x="41" y="113.5" text-anchor="middle">0</text>
<rect x="58" y="92" width="34" height="34" class="fill-soft-c stroke-grid"/>
<text x="75" y="113.5" text-anchor="middle">0</text>
<rect x="92" y="92" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="109" y="113.5" text-anchor="middle">1</text>
<rect x="126" y="92" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="143" y="113.5" text-anchor="middle">1</text>
<rect x="160" y="92" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="177" y="113.5" text-anchor="middle">1</text>
<rect x="24" y="126" width="34" height="34" class="fill-soft-c stroke-grid"/>
<text x="41" y="147.5" text-anchor="middle">0</text>
<rect x="58" y="126" width="34" height="34" class="fill-soft-c stroke-grid"/>
<text x="75" y="147.5" text-anchor="middle">0</text>
<rect x="92" y="126" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="109" y="147.5" text-anchor="middle">1</text>
<rect x="126" y="126" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="143" y="147.5" text-anchor="middle">1</text>
<rect x="160" y="126" width="34" height="34" class="fill-soft-d stroke-grid"/>
<text x="177" y="147.5" text-anchor="middle">1</text>
<rect x="24" y="160" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="41" y="181.5" text-anchor="middle">0</text>
<rect x="58" y="160" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="75" y="181.5" text-anchor="middle">0</text>
<rect x="92" y="160" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="109" y="181.5" text-anchor="middle">1</text>
<rect x="126" y="160" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="143" y="181.5" text-anchor="middle">1</text>
<rect x="160" y="160" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="177" y="181.5" text-anchor="middle">1</text>
<rect x="24" y="194" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="41" y="215.5" text-anchor="middle">0</text>
<rect x="58" y="194" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="75" y="215.5" text-anchor="middle">0</text>
<rect x="92" y="194" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="109" y="215.5" text-anchor="middle">1</text>
<rect x="126" y="194" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="143" y="215.5" text-anchor="middle">1</text>
<rect x="160" y="194" width="34" height="34" class="fill-paper stroke-grid"/>
<text x="177" y="215.5" text-anchor="middle">1</text>
<rect x="24" y="58" width="102" height="102" class="stroke-c"/>
<rect x="92" y="58" width="102" height="102" class="stroke-d dashed"/>
<text x="24" y="250" class="fig-small">Window 1 (green), window 2 (purple, dashed)</text>
<line x1="75" y1="53" x2="143" y2="53" class="stroke-ink" marker-end="url(#c034-arrow-1)"/>
<text x="109" y="46" text-anchor="middle" class="fig-small">slide two pixels right</text>
<text x="252" y="30" class="fig-bold">Kernel (3 by 3)</text>
<rect x="252" y="58" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="266" y="76.5" text-anchor="middle">-1</text>
<rect x="280" y="58" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="294" y="76.5" text-anchor="middle">0</text>
<rect x="308" y="58" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="322" y="76.5" text-anchor="middle">1</text>
<rect x="252" y="86" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="266" y="104.5" text-anchor="middle">-1</text>
<rect x="280" y="86" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="294" y="104.5" text-anchor="middle">0</text>
<rect x="308" y="86" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="322" y="104.5" text-anchor="middle">1</text>
<rect x="252" y="114" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="266" y="132.5" text-anchor="middle">-1</text>
<rect x="280" y="114" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="294" y="132.5" text-anchor="middle">0</text>
<rect x="308" y="114" width="28" height="28" class="fill-paper stroke-grid"/>
<text x="322" y="132.5" text-anchor="middle">1</text>
<text x="252" y="156" class="fig-small">right minus left:</text>
<text x="252" y="172" class="fig-small">a vertical-edge filter</text>
<text x="410" y="30" class="fig-bold">Output (3 by 3)</text>
<rect x="410" y="58" width="44" height="44" class="fill-soft-c stroke-grid"/>
<text x="432" y="84.5" text-anchor="middle">3</text>
<rect x="454" y="58" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="476" y="84.5" text-anchor="middle">3</text>
<rect x="498" y="58" width="44" height="44" class="fill-soft-d stroke-grid"/>
<text x="520" y="84.5" text-anchor="middle">0</text>
<rect x="410" y="102" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="432" y="128.5" text-anchor="middle">3</text>
<rect x="454" y="102" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="476" y="128.5" text-anchor="middle">3</text>
<rect x="498" y="102" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="520" y="128.5" text-anchor="middle">0</text>
<rect x="410" y="146" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="432" y="172.5" text-anchor="middle">3</text>
<rect x="454" y="146" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="476" y="172.5" text-anchor="middle">3</text>
<rect x="498" y="146" width="44" height="44" class="fill-paper stroke-grid"/>
<text x="520" y="172.5" text-anchor="middle">0</text>
<rect x="410" y="58" width="44" height="44" class="stroke-c"/>
<rect x="498" y="58" width="44" height="44" class="stroke-d dashed"/>
<text x="410" y="212" class="fig-small">Window 1 gives 3, window 2 gives 0</text>
<text x="410" y="228" class="fig-small">Same three weights, two positions</text>
<text x="225" y="106" text-anchor="middle" class="fig-bold">×</text>
<text x="384" y="106" text-anchor="middle" class="fig-bold">=</text>
</svg>
Figure 1. One filter, two positions. In window 1 each row reads 0, 0, 1, so the filter sums (1 − 0) three times and outputs 3. In window 2 every pixel is bright, each row gives 1 − 1 = 0, and the output is 0. Notice that the same nine weights produced both numbers.
```

Take the image in Figure 1. Its left two columns are dark (0) and its right three are bright (1). The filter has −1 in its left column, 0 in its middle column, and 1 in its right column, so it computes "right side minus left side". On window 1, every row of the patch reads 0, 0, 1. Each row contributes (−1)(0) + (0)(0) + (1)(1) = 1, and the three rows add up to 3. On window 2, all nine pixels are 1. Each row contributes −1 + 0 + 1 = 0, so the total is 0. The filter reports a big number where the picture goes from dark to bright and zero where it is flat, so it is a vertical-edge detector. Sliding it over all nine positions gives [3, 3, 0] in every row.

In symbols, for a filter with weights $w$ of size $k$ by $k$, a bias $b$, and an image $x$, the output at row $i$ and column $j$ is

$$
y_{i,j} = b + \sum_{a=0}^{k-1}\sum_{c=0}^{k-1} w_{a,c}\; x_{i+a,\,j+c}
$$

Read it aloud: put the filter's top-left corner on pixel $(i, j)$, multiply each weight $w_{a,c}$ by the pixel $a$ rows down and $c$ columns right of that corner, add all $k^2$ products, and add the bias. In numpy, one output value is a single line:

```python
y[i, j] = b + np.sum(w * x[i:i + k, j:j + k])
```

> [!WARNING]
> Textbook convolution flips the filter before sliding it. The operation above does not flip it, and it is what libraries such as PyTorch compute under the name convolution. Signal-processing texts call it cross-correlation. The flip does not matter for learning, because a network learns whichever weights it needs. Your code must not flip the filter, so that your answers match the libraries.

## Give the filter depth, and give the layer many filters

A colour image has three channels, so a window is a 3 by 3 patch in each of them, a block of 27 pixels, and a filter needs a weight for each of the 27. One filter therefore has shape `(C_in, k, k)`, like a small stack of three stamps. You add the 27 products into one number and add the bias, so one filter still produces one output grid.

One filter finds one pattern. A layer holds many filters, `C_out` of them, and each produces its own output grid, called a **channel** or **feature map**. One channel might light up on vertical edges, another on horizontal edges. The layer's weights have shape `(C_out, C_in, k, k)`, its biases have shape `(C_out,)`, and a batch of images has shape `(N, C_in, H, W)`.

Count the weights for the photo. A layer of 64 filters, each 3 by 3 and reading 3 channels, has 3 × 3 × 3 × 64 = 1,728 weights plus 64 biases, which is 1,792 numbers. The dense layer needed 150,529,000, so the convolution layer is about 84,000 times smaller, and the count does not grow if the photo gets bigger. Each filter is reused at 222 × 222 positions on this photo.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A dense layer with 24 separate wires beside a convolution that reuses three weights">
<text x="20" y="24" class="fig-bold">Dense layer: every pixel to every output</text>
<line x1="30" y1="88" x2="76" y2="192" class="stroke-grid"/>
<line x1="76" y1="88" x2="76" y2="192" class="stroke-grid"/>
<line x1="122" y1="88" x2="76" y2="192" class="stroke-grid"/>
<line x1="168" y1="88" x2="76" y2="192" class="stroke-grid"/>
<line x1="214" y1="88" x2="76" y2="192" class="stroke-grid"/>
<line x1="260" y1="88" x2="76" y2="192" class="stroke-grid"/>
<line x1="30" y1="88" x2="122" y2="192" class="stroke-grid"/>
<line x1="76" y1="88" x2="122" y2="192" class="stroke-grid"/>
<line x1="122" y1="88" x2="122" y2="192" class="stroke-grid"/>
<line x1="168" y1="88" x2="122" y2="192" class="stroke-grid"/>
<line x1="214" y1="88" x2="122" y2="192" class="stroke-grid"/>
<line x1="260" y1="88" x2="122" y2="192" class="stroke-grid"/>
<line x1="30" y1="88" x2="168" y2="192" class="stroke-grid"/>
<line x1="76" y1="88" x2="168" y2="192" class="stroke-grid"/>
<line x1="122" y1="88" x2="168" y2="192" class="stroke-grid"/>
<line x1="168" y1="88" x2="168" y2="192" class="stroke-grid"/>
<line x1="214" y1="88" x2="168" y2="192" class="stroke-grid"/>
<line x1="260" y1="88" x2="168" y2="192" class="stroke-grid"/>
<line x1="30" y1="88" x2="214" y2="192" class="stroke-grid"/>
<line x1="76" y1="88" x2="214" y2="192" class="stroke-grid"/>
<line x1="122" y1="88" x2="214" y2="192" class="stroke-grid"/>
<line x1="168" y1="88" x2="214" y2="192" class="stroke-grid"/>
<line x1="214" y1="88" x2="214" y2="192" class="stroke-grid"/>
<line x1="260" y1="88" x2="214" y2="192" class="stroke-grid"/>
<circle cx="30" cy="80" r="8" class="fill-a"/>
<circle cx="76" cy="80" r="8" class="fill-a"/>
<circle cx="122" cy="80" r="8" class="fill-a"/>
<circle cx="168" cy="80" r="8" class="fill-a"/>
<circle cx="214" cy="80" r="8" class="fill-a"/>
<circle cx="260" cy="80" r="8" class="fill-a"/>
<circle cx="76" cy="200" r="8" class="fill-b"/>
<circle cx="122" cy="200" r="8" class="fill-b"/>
<circle cx="168" cy="200" r="8" class="fill-b"/>
<circle cx="214" cy="200" r="8" class="fill-b"/>
<text x="20" y="245" class="fig-bold">6 pixels × 4 outputs = 24 weights</text>
<text x="20" y="264" class="fig-small">each wire has its own weight</text>
<text x="300" y="24" class="fig-bold">Convolution: one stamp, reused</text>
<line x1="310" y1="88" x2="356" y2="192" class="stroke-a"/>
<line x1="356" y1="88" x2="356" y2="192" class="stroke-b"/>
<line x1="402" y1="88" x2="356" y2="192" class="stroke-c"/>
<line x1="356" y1="88" x2="402" y2="192" class="stroke-a"/>
<line x1="402" y1="88" x2="402" y2="192" class="stroke-b"/>
<line x1="448" y1="88" x2="402" y2="192" class="stroke-c"/>
<line x1="402" y1="88" x2="448" y2="192" class="stroke-a"/>
<line x1="448" y1="88" x2="448" y2="192" class="stroke-b"/>
<line x1="494" y1="88" x2="448" y2="192" class="stroke-c"/>
<line x1="448" y1="88" x2="494" y2="192" class="stroke-a"/>
<line x1="494" y1="88" x2="494" y2="192" class="stroke-b"/>
<line x1="540" y1="88" x2="494" y2="192" class="stroke-c"/>
<circle cx="310" cy="80" r="8" class="fill-a"/>
<circle cx="356" cy="80" r="8" class="fill-a"/>
<circle cx="402" cy="80" r="8" class="fill-a"/>
<circle cx="448" cy="80" r="8" class="fill-a"/>
<circle cx="494" cy="80" r="8" class="fill-a"/>
<circle cx="540" cy="80" r="8" class="fill-a"/>
<circle cx="356" cy="200" r="8" class="fill-b"/>
<circle cx="402" cy="200" r="8" class="fill-b"/>
<circle cx="448" cy="200" r="8" class="fill-b"/>
<circle cx="494" cy="200" r="8" class="fill-b"/>
<text x="300" y="245" class="fig-bold">3 weights, used at all 4 positions</text>
<text x="300" y="264" class="fig-small">left wire blue, middle orange, right green</text>
<text x="20" y="66" class="fig-small">pixels</text>
<text x="20" y="222" class="fig-small">outputs</text>
<text x="20" y="300" class="fig-small">On the 224 by 224 colour photo with 1,000 dense units: 150,529,000 weights.</text>
<text x="20" y="318" class="fig-small">A 3 by 3 convolution with 64 filters: 1,792 weights.</text>
</svg>
Figure 2. The same six pixels and four outputs, wired two ways. Dense: 24 wires, 24 weights. Convolution: 12 wires, but only 3 weights, because the blue, orange, and green wires repeat at every position. Notice that adding more pixels adds wires but no weights.
```

Figure 2 shows why the count falls so far. In the dense layer, every wire is a separate weight. In the convolution, the wires still exist, but the left wire of every window shares one weight, the middle wire shares another, and the right wire shares a third. Sharing is the assumption "a pattern means the same thing everywhere", written as a wiring diagram.

Sharing has a visible consequence. Moving the object in the image moves the response, and nothing else changes. This property is called **translation equivariance**: shift the input, and the output shifts by the same amount. It holds away from the border, because shifting brings zeros in on one side and the output cells that read them have no partner. It is not invariance: an equivariant layer says "the edge is here", and if the edge moves, the answer moves. Pooling, covered below, is the first step that throws position away.

A cheaper filter bank exists when channels are many. A 3 by 3 layer with 64 channels in and 64 out costs 64 × 64 × 9 = 36,864 weights, plus 64 biases. The **depthwise separable** layer splits each filter's two jobs. It looks in space with one 3 by 3 filter per input channel, which costs 64 × 9 = 576 weights. Then it mixes channels at each pixel with a **1 by 1 convolution**, which costs 64 × 64 = 4,096 weights. The total is 4,672 weights plus 64 biases, about eight times fewer, at the price that every output channel builds on the same 64 spatial patterns.

> [!KEY]
> A dense layer cannot share what it learns between positions. A convolution reuses one small filter at every window, so it needs only `C_out * C_in * k * k + C_out` numbers (1,792 for the photo instead of 150 million) and a shifted picture gives a shifted answer.

## Count where the window fits

So far the window moved one pixel at a time and never left the image. Stacking layers needs exact shapes, so count the positions. Start in one dimension, with a strip of 7 cells and a window of width 3. The window's left edge can start at cell 0, 1, 2, 3, and 4. At cell 5 it would hang off the end, because it would need cells 5, 6, and 7, and the strip stops at 6. So there are 5 positions, and the output has length 5. The last legal start is `size - k`, the starts run from 0 to that value, and counting them gives `size - k + 1`. Panel A of Figure 3 draws the five windows as bars.

```figure
<svg viewBox="0 0 600 452" role="img" aria-label="Window positions along a strip of 7 cells for four settings of padding and stride">
<text x="20" y="18" class="fig-bold">A. Kernel 3, stride 1, no padding</text>
<rect x="20" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="34" y="44.5" text-anchor="middle" class="fig-small">0</text>
<rect x="48" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="62" y="44.5" text-anchor="middle" class="fig-small">1</text>
<rect x="76" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="90" y="44.5" text-anchor="middle" class="fig-small">2</text>
<rect x="104" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="118" y="44.5" text-anchor="middle" class="fig-small">3</text>
<rect x="132" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="146" y="44.5" text-anchor="middle" class="fig-small">4</text>
<rect x="160" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="174" y="44.5" text-anchor="middle" class="fig-small">5</text>
<rect x="188" y="26" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="202" y="44.5" text-anchor="middle" class="fig-small">6</text>
<line x1="23" y1="63" x2="101" y2="63" class="stroke-a"/>
<line x1="51" y1="72" x2="129" y2="72" class="stroke-b"/>
<line x1="79" y1="81" x2="157" y2="81" class="stroke-c"/>
<line x1="107" y1="63" x2="185" y2="63" class="stroke-a"/>
<line x1="135" y1="72" x2="213" y2="72" class="stroke-b"/>
<text x="310" y="40" class="fig-bold">5 windows, output length 5</text>
<text x="310" y="60" class="fig-mono fig-small">floor((7 + 0 − 3) / 1) + 1 = 5</text>
<text x="20" y="128" class="fig-bold">B. Add padding 1 on each side</text>
<rect x="20" y="136" width="28" height="28" class="fill-sunk stroke-muted dashed"/>
<text x="34" y="154.5" text-anchor="middle" class="fig-small">0</text>
<rect x="48" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="62" y="154.5" text-anchor="middle" class="fig-small">0</text>
<rect x="76" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="90" y="154.5" text-anchor="middle" class="fig-small">1</text>
<rect x="104" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="118" y="154.5" text-anchor="middle" class="fig-small">2</text>
<rect x="132" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="146" y="154.5" text-anchor="middle" class="fig-small">3</text>
<rect x="160" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="174" y="154.5" text-anchor="middle" class="fig-small">4</text>
<rect x="188" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="202" y="154.5" text-anchor="middle" class="fig-small">5</text>
<rect x="216" y="136" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="230" y="154.5" text-anchor="middle" class="fig-small">6</text>
<rect x="244" y="136" width="28" height="28" class="fill-sunk stroke-muted dashed"/>
<text x="258" y="154.5" text-anchor="middle" class="fig-small">0</text>
<line x1="23" y1="173" x2="101" y2="173" class="stroke-a"/>
<line x1="51" y1="182" x2="129" y2="182" class="stroke-b"/>
<line x1="79" y1="191" x2="157" y2="191" class="stroke-c"/>
<line x1="107" y1="173" x2="185" y2="173" class="stroke-a"/>
<line x1="135" y1="182" x2="213" y2="182" class="stroke-b"/>
<line x1="163" y1="191" x2="241" y2="191" class="stroke-c"/>
<line x1="191" y1="173" x2="269" y2="173" class="stroke-a"/>
<text x="310" y="150" class="fig-bold">7 windows, output length 7</text>
<text x="310" y="170" class="fig-mono fig-small">floor((7 + 2 − 3) / 1) + 1 = 7</text>
<text x="310" y="190" class="fig-small">dashed cells are zero padding</text>
<text x="20" y="238" class="fig-bold">C. Padding 1 and stride 2</text>
<rect x="20" y="246" width="28" height="28" class="fill-sunk stroke-muted dashed"/>
<text x="34" y="264.5" text-anchor="middle" class="fig-small">0</text>
<rect x="48" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="62" y="264.5" text-anchor="middle" class="fig-small">0</text>
<rect x="76" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="90" y="264.5" text-anchor="middle" class="fig-small">1</text>
<rect x="104" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="118" y="264.5" text-anchor="middle" class="fig-small">2</text>
<rect x="132" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="146" y="264.5" text-anchor="middle" class="fig-small">3</text>
<rect x="160" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="174" y="264.5" text-anchor="middle" class="fig-small">4</text>
<rect x="188" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="202" y="264.5" text-anchor="middle" class="fig-small">5</text>
<rect x="216" y="246" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="230" y="264.5" text-anchor="middle" class="fig-small">6</text>
<rect x="244" y="246" width="28" height="28" class="fill-sunk stroke-muted dashed"/>
<text x="258" y="264.5" text-anchor="middle" class="fig-small">0</text>
<line x1="23" y1="283" x2="101" y2="283" class="stroke-a"/>
<line x1="79" y1="292" x2="157" y2="292" class="stroke-b"/>
<line x1="135" y1="301" x2="213" y2="301" class="stroke-c"/>
<line x1="191" y1="283" x2="269" y2="283" class="stroke-a"/>
<text x="310" y="260" class="fig-bold">4 windows, output length 4</text>
<text x="310" y="280" class="fig-mono fig-small">floor((7 + 2 − 3) / 2) + 1 = 4</text>
<text x="310" y="300" class="fig-small">dashed cells are zero padding</text>
<text x="20" y="348" class="fig-bold">D. Stride 3, no padding</text>
<rect x="20" y="356" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="34" y="374.5" text-anchor="middle" class="fig-small">0</text>
<rect x="48" y="356" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="62" y="374.5" text-anchor="middle" class="fig-small">1</text>
<rect x="76" y="356" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="90" y="374.5" text-anchor="middle" class="fig-small">2</text>
<rect x="104" y="356" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="118" y="374.5" text-anchor="middle" class="fig-small">3</text>
<rect x="132" y="356" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="146" y="374.5" text-anchor="middle" class="fig-small">4</text>
<rect x="160" y="356" width="28" height="28" class="fill-soft-a stroke-grid"/>
<text x="174" y="374.5" text-anchor="middle" class="fig-small">5</text>
<rect x="188" y="356" width="28" height="28" class="fill-paper stroke-bad"/>
<text x="202" y="374.5" text-anchor="middle" class="fig-small">6</text>
<line x1="23" y1="393" x2="101" y2="393" class="stroke-a"/>
<line x1="107" y1="402" x2="185" y2="402" class="stroke-b"/>
<text x="310" y="370" class="fig-bold">2 windows, output length 2</text>
<text x="310" y="390" class="fig-mono fig-small">floor((7 + 0 − 3) / 3) + 1 = 2</text>
<text x="310" y="410" class="fig-small">cell 6 is never read (dropped)</text>
</svg>
Figure 3. A strip of 7 cells and a window of width 3. Each coloured bar is one window position. Padding adds zero cells (dashed) and gives more positions; a larger stride skips positions. In panel D the red cell is never read, because the leftover cells do not fit a whole window.
```

Two things go wrong with this count. The strip shrinks: after one layer 7 becomes 5, after the next 3, and after the next 1, so a deep stack runs out of picture. And the cells are not treated equally: in panel A, cell 0 sits under one window and cell 3 sits under three, so the border is seen poorly.

**Padding** fixes both. You add $p$ zero cells to each end of the strip before sliding. The strip becomes $7 + 2p$ cells long, so the count becomes $7 + 2p - 3 + 1$. With $p = 1$ that is 7 windows (panel B). To keep the size unchanged for an odd kernel at stride 1, you need $\text{size} + 2p - k + 1 = \text{size}$, which gives $p = (k-1)/2$. For $k = 3$ that is $p = 1$, and for $k = 5$ it is $p = 2$.

**Stride** is the number of cells the window moves between positions. Stride 2 evaluates every second position, which shrinks the output and cuts the work of every later layer. With stride $s$ the starts are $0, s, 2s, \dots$, so you divide the travel by $s$ and round down. Panel C uses padding 1 and stride 2 on the 7-cell strip, which is 9 cells once padded: the starts are 0, 2, 4, and 6, so the output has 4 cells. Rounding down matters. In panel D the stride is 3, the starts are 0 and 3, and cell 6 is never read, because the next window would hang off the end. Leftover cells that no whole window reaches are dropped.

One more knob is **dilation**: spacing the taps of the kernel $d$ cells apart. A width-3 kernel with $d = 2$ reads cells $i$, $i + 2$, and $i + 4$, so it reaches 5 cells while still using 3 weights. Call the distance the kernel spans its **reach**, which is $d(k-1) + 1$. Putting all four knobs together,

$$
\text{out} = \left\lfloor \frac{\text{size} + 2p - \big(d\,(k - 1) + 1\big)}{s} \right\rfloor + 1
$$

Read it aloud: pad both ends, subtract the kernel's reach to get the distance the window can travel, divide by the stride and round down to count the jumps, and add one for the starting position. In Python, the same sentence is two lines:

```python
reach = dilation * (kernel_size - 1) + 1
out = (size + 2 * padding - reach) // stride + 1
```

The following table lists the cases from Figure 3, plus the dilated one. Each row follows the formula.

| size | $k$ | $s$ | $p$ | $d$ | output |
|---|---|---|---|---|---|
| 7 | 3 | 1 | 0 | 1 | 5 |
| 7 | 3 | 1 | 1 | 1 | 7 |
| 7 | 3 | 2 | 1 | 1 | 4 |
| 7 | 3 | 3 | 0 | 1 | 2 |
| 7 | 3 | 1 | 0 | 2 | 3 |

The formula applies to the height and the width separately. Try a stack of three layers on a 32 by 32 image, each with a 3 by 3 filter and padding 1. The first two have stride 2 and the last has stride 1. The first layer gives $\lfloor (32 + 2 - 3)/2 \rfloor + 1 = 16$, the second gives $\lfloor (16 + 2 - 3)/2 \rfloor + 1 = 8$, and the third gives $8 + 2 - 3 + 1 = 8$. The maps are 16 by 16, 8 by 8, and 8 by 8.

> [!KEY]
> Counting window positions gives the output size: pad, subtract the reach, divide by the stride, round down, add one. Padding of $(k-1)/2$ keeps the size at stride 1, and a stride larger than 1 can drop leftover cells.

## Count how far back a unit looks

The **receptive field** of a unit is the set of input pixels it depends on. Along one axis, it is a count of pixels. For a unit in the first layer it equals the kernel size. For a deeper unit, you trace back through the layers.

Take two 3 by 3 layers at stride 1. A unit in the second layer reads 3 neighbouring first-layer units, say units 4, 5, and 6. Unit 4 reads pixels 4 to 6, and unit 6 reads pixels 6 to 8, so together they cover pixels 4 to 8: the second-layer unit sees 5 pixels.

A stride changes the picture. If the first layer has stride 2, neighbouring first-layer units sit two pixels apart, not one. The second layer's three taps are then two pixels apart, and each tap beyond the first adds 2 pixels to the view, not 1. Define the **jump** of a layer as the distance, in input pixels, between neighbouring units of that layer: the product of the strides of all layers up to and including it. A layer with kernel $k$ adds $k - 1$ extra taps, each one `jump` pixels further out, where `jump` is the spacing of the layer it reads:

$$
r_{\text{new}} = r + (k - 1)\cdot \text{jump}, \qquad \text{jump}_{\text{new}} = \text{jump}\cdot s
$$

Read it aloud: the field grows by the kernel's extra taps times the spacing of the units it reads, and then this layer's stride multiplies the spacing for the layers after it. The order matters, because the layer itself reads its input at the old spacing. In Python, update `r` first:

```python
r = r + (kernel_size - 1) * jump
jump = jump * stride
```

For the stack above, with (kernel, stride) pairs (3, 2), (3, 2), and (3, 1), start from $r = 1$ and jump 1. Layer 1 gives $r = 1 + 2 \cdot 1 = 3$ and jump 2. Layer 2 gives $r = 3 + 2 \cdot 2 = 7$ and jump 4. Layer 3 gives $r = 7 + 2 \cdot 4 = 15$, and the jump stays at 4. If you update the jump first, you get 21, which is wrong. Figure 4 draws the three layers.

```figure
<svg viewBox="0 0 640 410" role="img" aria-label="A three-layer stack seen from the top unit down to 15 input pixels, with the field and jump of each layer">
<rect x="22" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="36" y="354" text-anchor="middle" class="fig-small">0</text>
<rect x="50" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="64" y="354" text-anchor="middle" class="fig-small">1</text>
<rect x="78" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="92" y="354" text-anchor="middle" class="fig-small">2</text>
<rect x="106" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="120" y="354" text-anchor="middle" class="fig-small">3</text>
<rect x="134" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="148" y="354" text-anchor="middle" class="fig-small">4</text>
<rect x="162" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="176" y="354" text-anchor="middle" class="fig-small">5</text>
<rect x="190" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="204" y="354" text-anchor="middle" class="fig-small">6</text>
<rect x="218" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="232" y="354" text-anchor="middle" class="fig-small">7</text>
<rect x="246" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="260" y="354" text-anchor="middle" class="fig-small">8</text>
<rect x="274" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="288" y="354" text-anchor="middle" class="fig-small">9</text>
<rect x="302" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="316" y="354" text-anchor="middle" class="fig-small">10</text>
<rect x="330" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="344" y="354" text-anchor="middle" class="fig-small">11</text>
<rect x="358" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="372" y="354" text-anchor="middle" class="fig-small">12</text>
<rect x="386" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="400" y="354" text-anchor="middle" class="fig-small">13</text>
<rect x="414" y="338" width="28" height="24" class="fill-soft-a stroke-grid"/>
<text x="428" y="354" text-anchor="middle" class="fig-small">14</text>
<line x1="36" y1="338" x2="64" y2="259" class="stroke-a"/>
<line x1="64" y1="338" x2="64" y2="259" class="stroke-a"/>
<line x1="92" y1="338" x2="64" y2="259" class="stroke-a"/>
<line x1="92" y1="338" x2="120" y2="259" class="stroke-a"/>
<line x1="120" y1="338" x2="120" y2="259" class="stroke-a"/>
<line x1="148" y1="338" x2="120" y2="259" class="stroke-a"/>
<line x1="148" y1="338" x2="176" y2="259" class="stroke-a"/>
<line x1="176" y1="338" x2="176" y2="259" class="stroke-a"/>
<line x1="204" y1="338" x2="176" y2="259" class="stroke-a"/>
<line x1="204" y1="338" x2="232" y2="259" class="stroke-a"/>
<line x1="232" y1="338" x2="232" y2="259" class="stroke-a"/>
<line x1="260" y1="338" x2="232" y2="259" class="stroke-a"/>
<line x1="260" y1="338" x2="288" y2="259" class="stroke-a"/>
<line x1="288" y1="338" x2="288" y2="259" class="stroke-a"/>
<line x1="316" y1="338" x2="288" y2="259" class="stroke-a"/>
<line x1="316" y1="338" x2="344" y2="259" class="stroke-a"/>
<line x1="344" y1="338" x2="344" y2="259" class="stroke-a"/>
<line x1="372" y1="338" x2="344" y2="259" class="stroke-a"/>
<line x1="372" y1="338" x2="400" y2="259" class="stroke-a"/>
<line x1="400" y1="338" x2="400" y2="259" class="stroke-a"/>
<line x1="428" y1="338" x2="400" y2="259" class="stroke-a"/>
<line x1="64" y1="241" x2="120" y2="169" class="stroke-b"/>
<line x1="120" y1="241" x2="120" y2="169" class="stroke-b"/>
<line x1="176" y1="241" x2="120" y2="169" class="stroke-b"/>
<line x1="176" y1="241" x2="232" y2="169" class="stroke-b"/>
<line x1="232" y1="241" x2="232" y2="169" class="stroke-b"/>
<line x1="288" y1="241" x2="232" y2="169" class="stroke-b"/>
<line x1="288" y1="241" x2="344" y2="169" class="stroke-b"/>
<line x1="344" y1="241" x2="344" y2="169" class="stroke-b"/>
<line x1="400" y1="241" x2="344" y2="169" class="stroke-b"/>
<line x1="120" y1="151" x2="232" y2="79" class="stroke-d"/>
<line x1="232" y1="151" x2="232" y2="79" class="stroke-d"/>
<line x1="344" y1="151" x2="232" y2="79" class="stroke-d"/>
<circle cx="64" cy="250" r="9" class="fill-a"/>
<circle cx="120" cy="250" r="9" class="fill-a"/>
<circle cx="176" cy="250" r="9" class="fill-a"/>
<circle cx="232" cy="250" r="9" class="fill-a"/>
<circle cx="288" cy="250" r="9" class="fill-a"/>
<circle cx="344" cy="250" r="9" class="fill-a"/>
<circle cx="400" cy="250" r="9" class="fill-a"/>
<circle cx="120" cy="160" r="9" class="fill-b"/>
<circle cx="232" cy="160" r="9" class="fill-b"/>
<circle cx="344" cy="160" r="9" class="fill-b"/>
<circle cx="232" cy="70" r="9" class="fill-d"/>
<line x1="22" y1="376" x2="442" y2="376" class="stroke-ink"/>
<line x1="22" y1="371" x2="22" y2="381" class="stroke-ink"/>
<line x1="442" y1="371" x2="442" y2="381" class="stroke-ink"/>
<text x="232" y="394" text-anchor="middle" class="fig-small">the top unit depends on all 15 pixels</text>
<text x="466" y="348" class="fig-bold">field 1, jump 1</text>
<text x="466" y="364" class="fig-small">pixels</text>
<text x="466" y="248" class="fig-bold">field 3, jump 2</text>
<text x="466" y="264" class="fig-small">layer 1: kernel 3, stride 2</text>
<text x="466" y="158" class="fig-bold">field 7, jump 4</text>
<text x="466" y="174" class="fig-small">layer 2: kernel 3, stride 2</text>
<text x="466" y="68" class="fig-bold">field 15, jump 4</text>
<text x="466" y="84" class="fig-small">layer 3: kernel 3, stride 1</text>
</svg>
Figure 4. The top unit reads 3 units of layer 2, which read 3 units each of layer 1 (neighbours 2 pixels apart), which read 3 pixels each. Overlaps count once, so the top unit sees pixels 0 to 14: a field of 15. Notice that the dots get further apart (jump 1, 2, 4) after each stride of 2.
```

Padding does not appear in the recursion, because zeros are not input pixels. With dilation, put the layer's reach in place of $k$. So on a 32 by 32 image, the last unit sees at most 15 pixels per side. If the object you want to name is 20 pixels wide, no unit sees it whole, and you fix that with more layers, larger strides, dilation, or pooling.

**Max pooling** is a layer with no weights that slides a window over each channel and keeps only the largest value in each window. On a 4 by 4 grid with 2 by 2 windows and stride 2, the windows hold (1, 3, 4, 9), (2, 5, 1, 0), (6, 2, 1, 1), and (7, 1, 3, 2), so the pooled values are 9, 5, 6, and 7. Pooling does three jobs. It shrinks the map, so later layers cost less. It widens the receptive field, because for size and field counts a pooling window of size $k$ and stride $s$ behaves exactly like a convolution layer with that kernel and stride and padding 0: a 2 by 2 pool with stride 2 between two 3 by 3 convolutions gives fields 3, then $3 + 1 \cdot 1 = 4$ with jump 2, then $4 + 2 \cdot 2 = 8$, where two plain 3 by 3 layers give only 5. And it tolerates some small shifts: move the picture one pixel left and the top row of pooled values (9 and 5) is unchanged, because both maxima stay inside their windows, while the bottom row changes (6, 7 becomes 7, 2), because a maximum crosses a window boundary.

> [!KEY]
> A convolution slides one small filter over the picture, so the weights stay few and the output size comes from counting window positions. A unit's receptive field grows by $(k-1)$ times the spacing of the units it reads, and that spacing is the product of the strides before it. The formula gives a ceiling: pixels near the centre of the field reach the unit through many paths and pixels near the edge through few, so the region that matters in practice is smaller than 15.

> [!RECAP]
> Hold these points before you write code:
>
> - A filter is a small grid of weights. The output at a window is the dot product of the filter with that window, plus a bias. The filter is not flipped, and one filter reads all `C_in` channels at once.
> - The weight count is `C_out * C_in * k * k + C_out`, independent of the image size. A depthwise separable layer uses `C_in * k * k + C_in * C_out + C_out` numbers.
> - A shifted input gives a shifted output away from the border.
> - Output length is $\lfloor(\text{size}+2p-(d(k-1)+1))/s\rfloor+1$ for each spatial axis. Padding $(k-1)/2$ keeps the size for odd $k$ at stride 1.
> - The receptive field starts at 1 and grows by $(k-1)\cdot\text{jump}$ before the jump is multiplied by the stride.
> - Max pooling keeps the largest value of each window and has the same size arithmetic as a convolution.

## Build it step by step

The checks call the functions in this order.

1. **`conv2d_single(x, w, b)`.** `x` has shape `(H, W)` and `w` has shape `(k, k)`. Return a grid of shape `(H - k + 1, W - k + 1)`. Loop over output positions and use the one-line formula above. The most common mistake is flipping `w`.
2. **`conv2d_naive(x, w, b, stride, padding)`.** `x` is `(N, C_in, H, W)` and `w` is `(C_out, C_in, k, k)`. The output has `(H + 2 * padding - k) // stride + 1` rows and likewise columns. Pad the image with `np.pad`, then loop over images, output channels, rows, and columns. At each position, multiply `w[o]` by the patch across all channels and add `b[o]`. The patch starts at `i * stride`, not at `i`. The most common mistake is summing over one input channel instead of all of them.
3. **`param_counts(n_in, n_out, c_in, c_out, k)`.** Return a dict with the keys `"dense"`, `"conv"`, and `"separable"`, each an integer that includes biases. The most common mistake is forgetting the biases.
4. **`conv_output_shape(size, kernel_size, stride, padding, dilation)`.** Return an integer. Compute the reach `dilation * (kernel_size - 1) + 1`, subtract it from `size + 2 * padding`, and use `//` by the stride. The most common mistake is using `kernel_size` where the reach belongs, which drops the dilation.
5. **`receptive_field_per_layer(layers)`.** `layers` is a list of `(kernel_size, stride)` pairs. Return a list with the field after each layer. Keep `r` and `jump`, grow `r` first, append it, then multiply `jump`. The most common mistake is the reverse order, which gives 21 instead of 15.
6. **`max_pool_forward(x, pool_size, stride)`.** `x` has shape `(N, C, H, W)`. Return `(N, C, H_out, W_out)`, with sizes from `conv_output_shape`. For each output position, take `window.max(axis=(2, 3))` so that every channel keeps its own maximum. The most common mistake is a maximum that mixes channels or images.

## Cost and scaling

Parameters do not grow with the image, but work does. Each output value of a 3 by 3 layer with 64 input channels costs 576 multiplies, and a 224 by 224 image has about 50,000 positions per channel. The loop version is far too slow for real networks. [Turning a convolution into one matrix multiply](/lessons/037-im2col-and-convolution-backward) replaces it, and you keep the loop version as the reference answer.

## The exercise

You implement `conv2d_single`, `conv2d_naive`, `param_counts`, `conv_output_shape`, `receptive_field_per_layer`, and `max_pool_forward`.

The checks run your single-channel slide on the edge image from Figure 1 and on an asymmetric filter, which catches a flipped kernel. They compare `conv2d_naive` with an independent reference for several strides and paddings, confirm that a shifted input gives a shifted output in the interior, and confirm the weight counts. Then they compare `conv_output_shape` with the table and the drop-the-leftover case, confirm the fields 3, 7, 15 for the chapter's stack, and run max pooling against the 4 by 4 grid and against a direct windowed maximum.
