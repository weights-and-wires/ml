---
title: Sliding one small filter over a picture
slug: 069-sliding-a-filter
part: "Part 5 — Vision and sequences"
summary: Reuse one small set of weights at every position to find a pattern anywhere, with far fewer weights than a dense layer.
minutes: 40
gpu: false
objectives:
  - Slide a small filter over an image by hand, and say why one filter finds the same pattern at every position.
  - Count the weights of a dense layer, a convolution layer, and a depthwise separable layer, and explain the gap.
  - Explain why a shifted input gives a shifted output, and where that statement stops being true.
  - Write a convolution with plain loops, for one channel and for a batch of multi-channel images.
prereqs:
  - 053-width-depth-and-cost
  - 057-backprop-mlp-and-gradient-check
  - 068-label-smoothing
exercise: 069-sliding-a-filter
---

# Sliding one small filter over a picture

A colour photo, 224 pixels on a side with three colour channels, holds 224 × 224 × 3 = 150,528 numbers. Suppose you want a first layer with 1,000 units that reads that photo. A dense layer gives every unit one weight for every pixel, so the layer holds 150,528 × 1,000 + 1,000 = 150,529,000 numbers to learn. Memory is the smaller problem.

The larger problem is that a dense layer has no idea that the top-left corner and the bottom-right corner of a photo are the same kind of place. A cat ear at the top left is detected by some weights. A cat ear at the bottom right is detected by different weights, which start out random and must learn the same thing again from scratch. You would need photos with ears in every position before the layer learned ears everywhere.

Why not shrink the photo to 28 by 28 and keep the dense layer? A smaller photo cuts the weights, but it throws away the detail that tells an ear from a whisker, and the layer still relearns every pattern at every position. A better question: can one tiny detector slide over the whole photo and report where it matches, using a few thousand weights in total?

## What you already know

You already use these ideas:

- A dense layer gives every output a weight for every input, plus one bias, so it has `n_in * n_out + n_out` numbers to learn.
- A dot product multiplies two lists entry by entry and adds the results.
- A grayscale image is a grid of numbers. A colour image is three such grids, called channels, for red, green, and blue.
- In numpy, `x[i:i + 3, j:j + 3]` cuts out a 3 by 3 block of a grid.
- A weight is a number the training loop adjusts to reduce the loss.

## Look for the same pattern everywhere

Two facts about pictures make the dense layer wasteful. The first is that a useful pattern, such as an edge, a corner, or an ear, is small. To decide whether an edge is present, a unit needs to look at a few neighbouring pixels, not all 150,528. The second is that a pattern means the same thing wherever it appears. A vertical edge in the top left is the same kind of edge as one in the bottom right, so one detector for it should be enough.

A **filter** (also called a **kernel**) is a small grid of weights that acts as that detector. You lay it on a patch of the image, called a **window**, multiply each weight by the pixel under it, and add everything up. The sum is large when the patch looks like the pattern the weights describe. Then you move the window one pixel and repeat. Figure 1 shows the idea on a 5 by 5 image.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="A 3 by 3 filter slides over a 5 by 5 image and fills a 3 by 3 output">
<defs>
<marker id="c077-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
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
<line x1="75" y1="53" x2="143" y2="53" class="stroke-ink" marker-end="url(#c077-arrow)"/>
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

Take the image in Figure 1. Its left two columns are dark (0) and its right three are bright (1). The filter has −1 in its left column, 0 in its middle column, and 1 in its right column, so it computes "right side minus left side". On window 1, every row of the patch reads 0, 0, 1. Each row contributes (−1)(0) + (0)(0) + (1)(1) = 1, and the three rows add up to 3. On window 2, all nine pixels are 1. Each row contributes −1 + 0 + 1 = 0, so the total is 0. The filter reports a big number where the picture goes from dark to bright, and zero where the picture is flat. That makes it a vertical-edge detector. Sliding it over all nine window positions gives the output grid [3, 3, 0] in every row.

In symbols, for a filter with weights $w$ of size $k$ by $k$, a bias $b$, and an image $x$, the output at row $i$ and column $j$ is

$$
y_{i,j} = b + \sum_{a=0}^{k-1}\sum_{c=0}^{k-1} w_{a,c}\; x_{i+a,\,j+c}
$$

Read it aloud: put the filter's top-left corner on pixel $(i, j)$, multiply each weight $w_{a,c}$ by the pixel $a$ rows down and $c$ columns right of that corner, add all $k^2$ products, and add the bias. In numpy, one output value is a single line:

```python
y[i, j] = b + np.sum(w * x[i:i + k, j:j + k])
```

The output grid is smaller than the image. A 3 by 3 window fits in an image of height 5 at only 3 vertical positions, so the output has $5 - 3 + 1 = 3$ rows. The chapter "How big is the output and how far can a unit see" explains this counting in detail.

Two options change how the window moves, and the exercise needs both. **Stride** is how many pixels the window jumps between positions. Stride 2 evaluates every second position, which shrinks the output and cuts the work. **Padding** adds a border of zeros around the image before sliding, so that a window can sit on the edge pixels and the output can keep the image's size. You do not need their theory yet. You only need to carry them through the loop: after padding, the window for output $(i, j)$ starts at pixel $(i \cdot \text{stride},\, j \cdot \text{stride})$ of the padded image.

> [!WARNING]
> Textbook convolution flips the filter before sliding it. The operation above does not flip it, and it is what libraries such as PyTorch compute under the name convolution. Signal-processing texts call it cross-correlation. The flip does not matter for learning: a network that wants a flipped filter learns the flipped weights. Your code must not flip the filter, so that your answers match the libraries.

## Give the filter depth, and give the layer many filters

A colour image has three channels, so a window is a 3 by 3 patch in each of them, a block of 27 pixels, and a filter that looks at it needs a weight for each of the 27. One filter therefore has shape `(C_in, k, k)`: a 3 by 3 patch of weights for each input channel, like a small stack of three stamps. You multiply every weight by the pixel beneath it, add the 27 products into one number, and add the bias. Because the sum runs across the channels, one filter can ask "is there an edge in red that is bright in blue?" One filter still produces one output grid.

One filter finds one pattern. A layer holds many filters, `C_out` of them, and each produces its own output grid, called a **channel** or **feature map**. One channel might light up on vertical edges, another on horizontal edges, another on a particular colour change. The layer's weights have shape `(C_out, C_in, k, k)`, its biases have shape `(C_out,)`, and a batch of images has shape `(N, C_in, H, W)`.

Count the weights for the photo. A layer of 64 filters, each 3 by 3 and reading 3 channels, has 3 × 3 × 3 × 64 = 1,728 weights plus 64 biases, which is 1,792 numbers. The dense layer needed 150,529,000. The convolution layer is about 84,000 times smaller, and the count does not grow if the photo gets bigger. Each filter is reused at 222 × 222 positions on this photo (a 3 by 3 window fits $224 - 3 + 1 = 222$ times along each side), so every weight earns its keep about 49,000 times.

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

## Shift the picture and the output shifts too

Sharing has a visible consequence. Because one filter sees every window, moving the object in the image moves the response, and nothing else changes. This property is called **translation equivariance**: shift the input, and the output shifts by the same amount.

Figure 3 shows it. The block in the top image produces a column of positive numbers on its left edge and negative numbers on its right edge. Shift the block one pixel right, and the same numbers appear one column further right.

```figure
<svg viewBox="0 0 600 460" role="img" aria-label="An edge filter's output map shifts by one column when the input block shifts by one column">
<defs>
<marker id="c077-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<text x="20" y="24" class="fig-bold">Image</text>
<text x="290" y="24" class="fig-bold">Output of the vertical-edge filter</text>
<rect x="20" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="44" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="65" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="86" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="86" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="86" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="83" y="86" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="104" y="86" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="125" y="86" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="86" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="86" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="107" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="107" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="107" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="83" y="107" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="104" y="107" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="125" y="107" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="107" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="107" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="128" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="128" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="128" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="83" y="128" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="104" y="128" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="125" y="128" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="128" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="128" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="149" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="170" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="191" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="290" y="40" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="303" y="57.5" text-anchor="middle" class="fig-small">1</text>
<rect x="316" y="40" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="57.5" text-anchor="middle" class="fig-small">1</text>
<rect x="342" y="40" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="57.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="40" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="381" y="57.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="394" y="40" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="57.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="420" y="40" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="57.5" text-anchor="middle" class="fig-small">0</text>
<rect x="290" y="66" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="303" y="83.5" text-anchor="middle" class="fig-small">2</text>
<rect x="316" y="66" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="83.5" text-anchor="middle" class="fig-small">2</text>
<rect x="342" y="66" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="83.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="66" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="381" y="83.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="394" y="66" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="83.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="420" y="66" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="83.5" text-anchor="middle" class="fig-small">0</text>
<rect x="290" y="92" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="303" y="109.5" text-anchor="middle" class="fig-small">3</text>
<rect x="316" y="92" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="109.5" text-anchor="middle" class="fig-small">3</text>
<rect x="342" y="92" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="109.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="92" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="381" y="109.5" text-anchor="middle" class="fig-small">-3</text>
<rect x="394" y="92" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="109.5" text-anchor="middle" class="fig-small">-3</text>
<rect x="420" y="92" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="109.5" text-anchor="middle" class="fig-small">0</text>
<rect x="290" y="118" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="303" y="135.5" text-anchor="middle" class="fig-small">2</text>
<rect x="316" y="118" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="135.5" text-anchor="middle" class="fig-small">2</text>
<rect x="342" y="118" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="135.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="118" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="381" y="135.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="394" y="118" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="135.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="420" y="118" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="135.5" text-anchor="middle" class="fig-small">0</text>
<rect x="290" y="144" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="303" y="161.5" text-anchor="middle" class="fig-small">1</text>
<rect x="316" y="144" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="161.5" text-anchor="middle" class="fig-small">1</text>
<rect x="342" y="144" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="161.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="144" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="381" y="161.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="394" y="144" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="161.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="420" y="144" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="161.5" text-anchor="middle" class="fig-small">0</text>
<rect x="290" y="170" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="187.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="170" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="329" y="187.5" text-anchor="middle" class="fig-small">0</text>
<rect x="342" y="170" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="187.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="170" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="187.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="170" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="407" y="187.5" text-anchor="middle" class="fig-small">0</text>
<rect x="420" y="170" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="187.5" text-anchor="middle" class="fig-small">0</text>
<rect x="20" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="262" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="283" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="304" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="304" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="304" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="304" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="104" y="304" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="125" y="304" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="146" y="304" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="304" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="325" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="325" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="325" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="325" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="104" y="325" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="125" y="325" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="146" y="325" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="325" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="346" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="346" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="346" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="346" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="104" y="346" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="125" y="346" width="21" height="21" class="fill-soft-a stroke-grid"/>
<rect x="146" y="346" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="346" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="367" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="388" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="20" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="41" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="62" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="83" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="104" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="125" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="146" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="167" y="409" width="21" height="21" class="fill-paper stroke-grid"/>
<rect x="290" y="258" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="275.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="258" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="275.5" text-anchor="middle" class="fig-small">1</text>
<rect x="342" y="258" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="355" y="275.5" text-anchor="middle" class="fig-small">1</text>
<rect x="368" y="258" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="275.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="258" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="275.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="420" y="258" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="433" y="275.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="290" y="284" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="301.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="284" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="301.5" text-anchor="middle" class="fig-small">2</text>
<rect x="342" y="284" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="355" y="301.5" text-anchor="middle" class="fig-small">2</text>
<rect x="368" y="284" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="301.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="284" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="301.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="420" y="284" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="433" y="301.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="290" y="310" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="327.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="310" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="327.5" text-anchor="middle" class="fig-small">3</text>
<rect x="342" y="310" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="355" y="327.5" text-anchor="middle" class="fig-small">3</text>
<rect x="368" y="310" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="327.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="310" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="327.5" text-anchor="middle" class="fig-small">-3</text>
<rect x="420" y="310" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="433" y="327.5" text-anchor="middle" class="fig-small">-3</text>
<rect x="290" y="336" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="353.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="336" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="353.5" text-anchor="middle" class="fig-small">2</text>
<rect x="342" y="336" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="355" y="353.5" text-anchor="middle" class="fig-small">2</text>
<rect x="368" y="336" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="353.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="336" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="353.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="420" y="336" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="433" y="353.5" text-anchor="middle" class="fig-small">-2</text>
<rect x="290" y="362" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="379.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="362" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="329" y="379.5" text-anchor="middle" class="fig-small">1</text>
<rect x="342" y="362" width="26" height="26" class="fill-soft-b stroke-grid"/>
<text x="355" y="379.5" text-anchor="middle" class="fig-small">1</text>
<rect x="368" y="362" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="379.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="362" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="407" y="379.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="420" y="362" width="26" height="26" class="fill-soft-d stroke-grid"/>
<text x="433" y="379.5" text-anchor="middle" class="fig-small">-1</text>
<rect x="290" y="388" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="303" y="405.5" text-anchor="middle" class="fig-small">0</text>
<rect x="316" y="388" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="329" y="405.5" text-anchor="middle" class="fig-small">0</text>
<rect x="342" y="388" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="355" y="405.5" text-anchor="middle" class="fig-small">0</text>
<rect x="368" y="388" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="381" y="405.5" text-anchor="middle" class="fig-small">0</text>
<rect x="394" y="388" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="407" y="405.5" text-anchor="middle" class="fig-small">0</text>
<rect x="420" y="388" width="26" height="26" class="fill-paper stroke-grid"/>
<text x="433" y="405.5" text-anchor="middle" class="fig-small">0</text>
<text x="20" y="230" class="fig-small">bright block</text>
<text x="20" y="448" class="fig-small">same block, shifted one pixel right</text>
<line x1="202" y1="128" x2="280" y2="128" class="stroke-ink" marker-end="url(#c077-arrow2)"/>
<line x1="202" y1="346" x2="280" y2="346" class="stroke-ink" marker-end="url(#c077-arrow2)"/>
<text x="224" y="122" text-anchor="middle" class="fig-small">filter</text>
<text x="224" y="340" text-anchor="middle" class="fig-small">filter</text>
<line x1="160" y1="240" x2="160" y2="254" class="stroke-ink" marker-end="url(#c077-arrow2)"/>
<text x="168" y="262" class="fig-small">shift the picture</text>
<rect x="420" y="40" width="26" height="156" class="stroke-bad dashed"/>
<rect x="290" y="258" width="26" height="156" class="stroke-bad dashed"/>
<text x="290" y="212" class="fig-small">dashed: no partner in the other map</text>
<text x="290" y="430" class="fig-small">orange: positive, purple: negative</text>
</svg>
Figure 3. Shift the block one pixel right and the output pattern moves one column right, with the same numbers. Only the dashed columns have no partner: the last column of the top map falls off, and the first column of the bottom map is fresh. Notice that equivariance holds away from the border.
```

The property has honest limits. It holds away from the border, because shifting pushes pixels off one side and brings zeros in on the other, and the output cells that read those pixels have no partner in the other map (the dashed columns). With a stride larger than 1, it holds only for shifts that are a multiple of the stride. The exercise checks the interior with a stride of 1.

Equivariance is not the same as invariance. An equivariant layer says "the edge is here"; if the edge moves, the answer moves. To answer "is there an edge somewhere?" you need a later step that throws position away. Pooling, covered in "How big is the output and how far can a unit see", is the first such step.

## A cheaper filter bank

A 3 by 3 layer with 64 channels in and 64 out costs 64 × 64 × 9 = 36,864 weights, plus 64 biases. The **depthwise separable** layer splits each filter's two jobs. First, it looks in space: one 3 by 3 filter per input channel, which costs 64 × 9 = 576 weights. Second, it mixes channels at each pixel with a **1 by 1 convolution**, a filter of size 1 that reads all 64 channels of a single pixel, which costs 64 × 64 = 4,096 weights. The total is 4,672 weights plus 64 biases, about eight times fewer. The price is flexibility: every output channel must build on the same 64 spatial patterns, instead of owning a 3 by 3 pattern for each input channel.

> [!KEY]
> A dense layer cannot share what it learns between positions. A convolution reuses one small filter at every window, so it needs only `C_out * C_in * k * k + C_out` numbers (1,792 for the photo instead of 150 million) and a shifted picture gives a shifted answer.

> [!RECAP]
> Hold these points before you write code:
>
> - A filter is a small grid of weights. The output at a window is the dot product of the filter with that window, plus a bias.
> - The filter is not flipped. A filter with `C_in` channels reads all of them at once, and one layer holds `C_out` filters.
> - The weight count is `C_out * C_in * k * k + C_out`, independent of the image size.
> - A shifted input gives a shifted output away from the border.
> - A depthwise separable layer uses `C_in * k * k + C_in * C_out + C_out` numbers.

## Build it step by step

The checks call the functions in this order.

1. **`conv2d_single(x, w, b)`.** `x` has shape `(H, W)` and `w` has shape `(k, k)`. Return a grid of shape `(H - k + 1, W - k + 1)`. Loop over output positions and use the one-line formula above. The most common mistake is flipping `w`.
2. **`conv2d_naive(x, w, b, stride, padding)`.** `x` is `(N, C_in, H, W)` and `w` is `(C_out, C_in, k, k)`. The output has `(H + 2 * padding - k) // stride + 1` rows and likewise columns. Pad the image with `np.pad`, then loop over images, output channels, rows, and columns. At each position, multiply `w[o]` by the patch across all channels and add `b[o]`. The patch starts at `i * stride`, not at `i`. The most common mistake is summing over one input channel instead of all of them.
3. **`param_counts(n_in, n_out, c_in, c_out, k)`.** Return a dict with the keys `"dense"`, `"conv"`, and `"separable"`, each an integer that includes biases. The most common mistake is forgetting the biases.

## Cost and scaling

Parameters do not grow with the image, but work does. Each output value of a 3 by 3 layer with 64 input channels costs 3 × 3 × 64 = 576 multiplies, and a 224 by 224 image has about 50,000 output positions per channel. The loop version is far too slow for real networks. In "Turning a convolution into one matrix multiply" you replace it, and you keep the loop version as the reference answer.

## The exercise

You implement `conv2d_single`, `conv2d_naive`, and `param_counts` in `starter.py`. The checks run your single-channel slide on the edge image from Figure 1 and on an asymmetric filter, which catches a flipped kernel. They compare `conv2d_naive` with an independent reference for several channels, images, strides, and paddings, confirm that a shifted input gives a shifted output in the interior, and confirm the weight counts from this chapter.
