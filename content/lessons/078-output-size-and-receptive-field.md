---
title: How big is the output and how far can a unit see
slug: 078-output-size-and-receptive-field
part: "Part 5 — Vision and sequences"
summary: Count where a window fits to get a layer's output size, and track how many input pixels each unit depends on.
minutes: 40
gpu: false
objectives:
  - Count the output length of a convolution by counting window positions, and add padding, stride, and dilation one at a time.
  - Choose the padding that keeps the size unchanged for an odd kernel at stride 1.
  - Compute the receptive field of every layer in a stack, and explain why the update uses the product of earlier strides.
  - Apply max pooling to a grid, and say what it keeps and what it throws away.
prereqs:
  - 077-sliding-a-filter
exercise: 078-output-size-and-receptive-field
---

# How big is the output and how far can a unit see

You stack three convolution layers on a 32 by 32 image. Each layer slides a 3 by 3 filter. Each pads the border with one ring of zeros. The first two move the window two pixels at a time, and the last moves it one pixel at a time. Before you build anything, you want two answers. How large is the map after each layer, so that you know the shapes your code must handle? And does a unit in the last layer see enough of the picture to recognise a whole object?

Without a way to count, you find out the hard way: a shape error deep in the network, or a last unit that sees 15 pixels when the object it must name is 20 pixels wide. No unit sees the whole object, so none can recognise it. Both answers come from counting, and this chapter builds each count from the smallest case.

## What you already know

You already use these ideas:

- A filter slides over an image, and each window position gives one output number.
- Integer division `a // b` rounds down, so `7 // 2` is 3.
- A slice `x[i:i + 3]` cuts out three neighbouring cells.
- A layer's output grid becomes the next layer's input grid.
- Zero adds nothing to a sum, because multiplying by it gives zero.

## Count where the window fits

Start in one dimension, with a strip of 7 cells and a window of width 3. The window's left edge can start at cell 0. It can start at cell 1, 2, 3, and 4. At cell 5 it would hang off the end, because it would need cells 5, 6, and 7, and the strip stops at 6. So there are 5 positions, and the output has length 5. The rule is that the last legal start is `size - k`, the starts run from 0 to that value, and counting them gives `size - k + 1`. Panel A of Figure 1 draws the five windows as bars.

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
Figure 1. A strip of 7 cells and a window of width 3. Each coloured bar is one window position. Padding adds zero cells (dashed) and gives more positions; a larger stride skips positions. In panel D the red cell is never read, because the leftover cells do not fit a whole window.
```

Two things go wrong with this count, and each leads to one knob. The first is that the strip shrinks. After one layer 7 becomes 5, after the next 3, and after the next 1. A deep stack would run out of picture. The second is that the cells are not treated equally. In panel A, cell 0 sits under one window, and cell 3 sits under three. The border pixels count for less, so anything that lives at the border is seen poorly.

**Padding** fixes both. You add $p$ zero cells to each end of the strip before sliding. The strip becomes $7 + 2p$ cells long, so the count becomes $7 + 2p - 3 + 1$. With $p = 1$ that is 7 windows (panel B). To keep the size unchanged for an odd kernel at stride 1, you need $\text{size} + 2p - k + 1 = \text{size}$, which gives $p = (k-1)/2$. For $k = 3$ that is $p = 1$, and for $k = 5$ it is $p = 2$. The zeros add nothing to a sum, so they let the window overlap the border without inventing data, although a border window still sees fewer real pixels than an interior one.

**Stride** is the number of cells the window moves between positions. With stride $s$ the starts are $0, s, 2s, \dots$, so you divide the travel by $s$ and round down. Panel C uses padding 1 and stride 2 on the 7-cell strip, which is 9 cells once padded: the starts are 0, 2, 4, and 6, so the output has 4 cells. Rounding down matters. In panel D the stride is 3 on the 7-cell strip, the starts are 0 and 3, and cell 6 is never read, because the next window would hang off the end. Leftover cells that no whole window reaches are dropped. You choose a stride above 1 on purpose, to shrink the map and to cut the work of every later layer.

One more knob is **dilation**: spacing the taps of the kernel $d$ cells apart. A width-3 kernel with $d = 2$ reads cells $i$, $i + 2$, and $i + 4$, so it reaches 5 cells while still using 3 weights. Call the distance the kernel spans its **reach**, which is $d(k-1) + 1$. Putting all four knobs together,

$$
\text{out} = \left\lfloor \frac{\text{size} + 2p - \big(d\,(k - 1) + 1\big)}{s} \right\rfloor + 1
$$

Read it aloud: pad both ends, subtract the kernel's reach to get the distance the window can travel, divide by the stride and round down to count the jumps, and add one for the starting position. Every knob you added earlier is a term in that sentence. In Python, the same sentence is two lines:

```python
reach = dilation * (kernel_size - 1) + 1
out = (size + 2 * padding - reach) // stride + 1
```

The following table lists the cases from Figure 1, plus the dilated one. Each row follows the formula.

| size | $k$ | $s$ | $p$ | $d$ | output |
|---|---|---|---|---|---|
| 7 | 3 | 1 | 0 | 1 | 5 |
| 7 | 3 | 1 | 1 | 1 | 7 |
| 7 | 3 | 2 | 1 | 1 | 4 |
| 7 | 3 | 3 | 0 | 1 | 2 |
| 7 | 3 | 1 | 0 | 2 | 3 |

The formula applies to the height and the width separately. For the opening stack, with padding 1 on every layer, the first layer gives $\lfloor (32 + 2 - 3)/2 \rfloor + 1 = 16$, the second gives $\lfloor (16 + 2 - 3)/2 \rfloor + 1 = 8$, and the third, at stride 1, gives $8 + 2 - 3 + 1 = 8$. The maps are 16 by 16, 8 by 8, and 8 by 8.

> [!KEY]
> Counting window positions gives the output size: pad, subtract the reach, divide by the stride, round down, add one. Padding of $(k-1)/2$ keeps the size at stride 1, and a stride larger than 1 can drop leftover cells.

## Count how far back a unit looks

The **receptive field** of a unit is the set of input pixels it depends on. Along one axis, it is a count of pixels. For a unit in the first layer it equals the kernel size. For a deeper unit, you trace back through the layers.

Take two 3 by 3 layers at stride 1. A unit in the second layer reads 3 neighbouring first-layer units, say units 4, 5, and 6. Unit 4 reads pixels 4 to 6, and unit 6 reads pixels 6 to 8, so together they cover pixels 4 to 8: the second-layer unit sees 5 pixels. That is the view of one 5 by 5 filter, but with 2 × 9 = 18 weights per pair of channels instead of 25.

A stride changes the picture. If the first layer has stride 2, neighbouring first-layer units sit two pixels apart, not one. The second layer's three taps are then two pixels apart, and each tap beyond the first adds 2 pixels to the view, not 1. Define the **jump** of a layer as the distance, in input pixels, between neighbouring units of that layer. It is the product of the strides of all the layers up to and including that one, because every stride multiplies the spacing. Then a layer with kernel $k$ adds $k - 1$ extra taps, each one `jump` pixels further out, where `jump` is the spacing of the layer it reads:

$$
r_{\text{new}} = r + (k - 1)\cdot \text{jump}, \qquad \text{jump}_{\text{new}} = \text{jump}\cdot s
$$

Read it aloud: the field grows by the kernel's extra taps times the spacing of the units it reads, and then this layer's stride multiplies the spacing for the layers after it. The order matters for that reason. The stride of a layer changes how far apart its own outputs are, and the layer itself reads its input at the old spacing. In Python, update `r` first:

```python
r = r + (kernel_size - 1) * jump
jump = jump * stride
```

For the opening stack, with (kernel, stride) pairs (3, 2), (3, 2), and (3, 1), start from $r = 1$ and jump 1. Layer 1 gives $r = 1 + 2 \cdot 1 = 3$ and jump 2. Layer 2 gives $r = 3 + 2 \cdot 2 = 7$ and jump 4. Layer 3 gives $r = 7 + 2 \cdot 4 = 15$, and the jump stays at 4. If you update the jump first, you get 21, which is wrong. Figure 2 draws the three layers.

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
Figure 2. The top unit reads 3 units of layer 2, which read 3 units each of layer 1 (neighbours 2 pixels apart), which read 3 pixels each. Overlaps count once, so the top unit sees pixels 0 to 14: a field of 15. Notice that the dots get further apart (jump 1, 2, 4) after each stride of 2.
```

Padding does not appear in the recursion, because zeros are not input pixels. With dilation, put the layer's reach in place of $k$, because the reach is the span of its taps.

The opening question has an answer. On a 32 by 32 image, the last unit sees at most 15 pixels per side. If the object you want to name is 20 pixels wide, no unit sees it whole, and the network cannot recognise it by looking at one unit. You fix that with more layers, larger strides, dilation, or pooling, which is the next idea.

> [!KEY]
> A unit's receptive field grows by $(k-1)$ times the spacing of the units it reads, and that spacing is the product of the strides before it. Early strides make later layers see further for the same kernel size.

## Shrink on purpose with max pooling

**Max pooling** is a layer with no weights. It slides a window over each channel and keeps only the largest value in each window. Figure 3 shows a 4 by 4 grid with 2 by 2 windows and stride 2. The windows hold (1, 3, 4, 9), (2, 5, 1, 0), (6, 2, 1, 1), and (7, 1, 3, 2), so the pooled values are 9, 5, 6, and 7.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="A 4 by 4 grid pooled with 2 by 2 windows, and the same grid shifted one pixel left">
<defs>
<marker id="c078-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<text x="16" y="44" class="fig-bold">Image</text>
<rect x="16" y="60" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="36" y="84.5" text-anchor="middle">1</text>
<rect x="56" y="60" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="76" y="84.5" text-anchor="middle">3</text>
<rect x="96" y="60" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="116" y="84.5" text-anchor="middle">2</text>
<rect x="136" y="60" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="156" y="84.5" text-anchor="middle">5</text>
<rect x="16" y="100" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="36" y="124.5" text-anchor="middle">4</text>
<rect x="56" y="100" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="76" y="124.5" text-anchor="middle">9</text>
<rect x="96" y="100" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="116" y="124.5" text-anchor="middle">1</text>
<rect x="136" y="100" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="156" y="124.5" text-anchor="middle">0</text>
<rect x="16" y="140" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="36" y="164.5" text-anchor="middle">6</text>
<rect x="56" y="140" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="76" y="164.5" text-anchor="middle">2</text>
<rect x="96" y="140" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="116" y="164.5" text-anchor="middle">7</text>
<rect x="136" y="140" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="156" y="164.5" text-anchor="middle">1</text>
<rect x="16" y="180" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="36" y="204.5" text-anchor="middle">1</text>
<rect x="56" y="180" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="76" y="204.5" text-anchor="middle">1</text>
<rect x="96" y="180" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="116" y="204.5" text-anchor="middle">3</text>
<rect x="136" y="180" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="156" y="204.5" text-anchor="middle">2</text>
<circle cx="76" cy="120" r="15" class="stroke-ink"/>
<circle cx="156" cy="80" r="15" class="stroke-ink"/>
<circle cx="36" cy="160" r="15" class="stroke-ink"/>
<circle cx="116" cy="160" r="15" class="stroke-ink"/>
<line x1="184" y1="140" x2="220" y2="140" class="stroke-ink" marker-end="url(#c078-arrow)"/>
<text x="202" y="132" text-anchor="middle" class="fig-small">max</text>
<text x="232" y="44" class="fig-bold">Pooled</text>
<rect x="232" y="100" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="252" y="124.5" text-anchor="middle" class="fig-bold">9</text>
<rect x="272" y="100" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="292" y="124.5" text-anchor="middle" class="fig-bold">5</text>
<rect x="232" y="140" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="252" y="164.5" text-anchor="middle" class="fig-bold">6</text>
<rect x="272" y="140" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="292" y="164.5" text-anchor="middle" class="fig-bold">7</text>
<text x="336" y="44" class="fig-bold">Picture moved one pixel left</text>
<rect x="336" y="60" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="356" y="84.5" text-anchor="middle">3</text>
<rect x="376" y="60" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="396" y="84.5" text-anchor="middle">2</text>
<rect x="416" y="60" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="436" y="84.5" text-anchor="middle">5</text>
<rect x="456" y="60" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="476" y="84.5" text-anchor="middle">0</text>
<rect x="336" y="100" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="356" y="124.5" text-anchor="middle">9</text>
<rect x="376" y="100" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="396" y="124.5" text-anchor="middle">1</text>
<rect x="416" y="100" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="436" y="124.5" text-anchor="middle">0</text>
<rect x="456" y="100" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="476" y="124.5" text-anchor="middle">0</text>
<rect x="336" y="140" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="356" y="164.5" text-anchor="middle">2</text>
<rect x="376" y="140" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="396" y="164.5" text-anchor="middle">7</text>
<rect x="416" y="140" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="436" y="164.5" text-anchor="middle">1</text>
<rect x="456" y="140" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="476" y="164.5" text-anchor="middle">0</text>
<rect x="336" y="180" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="356" y="204.5" text-anchor="middle">1</text>
<rect x="376" y="180" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="396" y="204.5" text-anchor="middle">3</text>
<rect x="416" y="180" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="436" y="204.5" text-anchor="middle">2</text>
<rect x="456" y="180" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="476" y="204.5" text-anchor="middle">0</text>
<circle cx="356" cy="120" r="15" class="stroke-ink"/>
<circle cx="436" cy="80" r="15" class="stroke-ink"/>
<circle cx="396" cy="160" r="15" class="stroke-ink"/>
<circle cx="436" cy="200" r="15" class="stroke-ink"/>
<line x1="504" y1="140" x2="540" y2="140" class="stroke-ink" marker-end="url(#c078-arrow)"/>
<text x="522" y="132" text-anchor="middle" class="fig-small">max</text>
<text x="552" y="44" class="fig-bold">Pooled</text>
<rect x="552" y="100" width="40" height="40" class="fill-soft-a stroke-grid"/>
<text x="572" y="124.5" text-anchor="middle" class="fig-bold">9</text>
<rect x="592" y="100" width="40" height="40" class="fill-soft-b stroke-grid"/>
<text x="612" y="124.5" text-anchor="middle" class="fig-bold">5</text>
<rect x="552" y="140" width="40" height="40" class="fill-soft-c stroke-grid"/>
<text x="572" y="164.5" text-anchor="middle" class="fig-bold">7</text>
<rect x="592" y="140" width="40" height="40" class="fill-soft-d stroke-grid"/>
<text x="612" y="164.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="552" y="100" width="40" height="40" class="stroke-c"/>
<rect x="592" y="100" width="40" height="40" class="stroke-c"/>
<rect x="552" y="140" width="40" height="40" class="stroke-bad dashed"/>
<rect x="592" y="140" width="40" height="40" class="stroke-bad dashed"/>
<text x="16" y="244" class="fig-small">Circles mark each window's maximum.</text>
<text x="336" y="244" class="fig-small">Green outline: same value. Red dashed: changed.</text>
<text x="336" y="262" class="fig-small">The left column fell off; a zero column came in.</text>
</svg>
Figure 3. Max pooling keeps the largest value in each 2 by 2 window. Move the picture one pixel left and the top row of pooled values (9 and 5) is unchanged, because both maxima stayed inside their windows. The bottom row changed (6, 7 became 7, 2), because a maximum crossed a window boundary. Notice that pooling tolerates some shifts, not all.
```

Pooling does three jobs. It shrinks the map, so later layers cost less. It widens the receptive field, because for the size and field counts a pooling window of size $k$ and stride $s$ behaves exactly like a convolution layer with that kernel and stride, and padding 0. Put a 2 by 2 pool with stride 2 after a 3 by 3 convolution and before another one: the fields are 3, then $3 + 1 \cdot 1 = 4$ with jump 2, then $4 + 2 \cdot 2 = 8$, where two plain 3 by 3 layers would give only 5. And it gives some tolerance to small shifts. In Figure 3, a one-pixel shift left leaves the top row of pooled values unchanged, because both maxima stay inside their windows, and changes the bottom row, because a maximum crosses a window boundary. The tolerance is partial.

One caution about the count. The receptive field says which pixels a unit can depend on, not how much it does. Pixels near the centre of the patch reach the unit through many paths and pixels near the edge through few, so the region that matters in practice, the effective receptive field, is smaller than the formula says. Treat 15 as a ceiling.

> [!RECAP]
> Hold these points before you write code:
>
> - Output length is $\lfloor(\text{size}+2p-(d(k-1)+1))/s\rfloor+1$ for each spatial axis.
> - Padding $(k-1)/2$ keeps the size for odd $k$ at stride 1, and leftover cells are dropped when the stride does not divide the travel.
> - The receptive field starts at 1 and grows by $(k-1)\cdot\text{jump}$ before the jump is multiplied by the stride.
> - Max pooling keeps the largest value of each window and has the same size arithmetic as a convolution.

## Build it step by step

The checks call the functions in this order.

1. **`conv_output_shape(size, kernel_size, stride, padding, dilation)`.** Return an integer. Compute the reach `dilation * (kernel_size - 1) + 1`, subtract it from `size + 2 * padding`, and use `//` by the stride. The most common mistake is using `kernel_size` where the reach belongs, which drops the dilation.
2. **`receptive_field_per_layer(layers)`.** `layers` is a list of `(kernel_size, stride)` pairs. Return a list with the field after each layer. Keep `r` and `jump`, grow `r` first, append it, then multiply `jump`. The most common mistake is the reverse order, which gives 21 instead of 15.
3. **`max_pool_forward(x, pool_size, stride)`.** `x` has shape `(N, C, H, W)`. Return `(N, C, H_out, W_out)`, with sizes from `conv_output_shape`. For each output position, take `window.max(axis=(2, 3))` so that every channel keeps its own maximum. The most common mistake is a maximum that mixes channels or images.

## The exercise

You implement `conv_output_shape`, `receptive_field_per_layer`, and `max_pool_forward` in `starter.py`. The checks compare the output size with the table and the drop-the-leftover case, confirm that padding $(k-1)/2$ preserves the size, and confirm the fields 3, 7, 15 for the chapter's stack and a stack where stride order matters. They check max pooling against the grid in Figure 3, the shifted grid, and a direct windowed maximum with overlapping windows.
