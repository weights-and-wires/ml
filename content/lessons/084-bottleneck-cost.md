---
title: "Doing the wide work at a narrow width: bottleneck cost"
slug: 084-bottleneck-cost
part: "Part 5 — Vision and sequences"
summary: A convolution's cost grows with the square of its channel count, so a bottleneck does its 3×3 work at a quarter of the width, about 17 times cheaper.
minutes: 35
gpu: false
objectives:
  - Count the multiply-adds of a convolution from its channels, kernel size, and image size, and explain why the count is a product.
  - Explain why a 1×1 convolution has no factor of 9 and why shrinking the channels cuts the cost of a 3×3 convolution quadratically.
  - Compute the cost of a bottleneck block and show that it is about 17 times cheaper than two full-width 3×3 convolutions.
  - Say what a projection shortcut is, when it is needed, and why memory grows linearly with depth.
prereqs:
  - 060-count-parameters-and-flops
  - 077-sliding-a-filter
  - 083-gradient-through-deep-stacks
exercise: 084-bottleneck-cost
---

# Doing the wide work at a narrow width: bottleneck cost

A residual network can be deep because the shortcut keeps the gradient alive. Depth costs compute, though. Take a block of two 3×3 convolutions on 256 channels, applied to a 14×14 picture. Each picture takes 231,211,008 multiply-adds to pass through that one block. A hundred such blocks take over 23 billion multiply-adds for the forward pass of a single picture, before any backward pass.

You can cut that bill by using fewer channels, but the channels are how a layer holds what it has seen at each position, and a narrow stack holds less. So the question is how to keep the wide path between blocks and still make each block cheap. This chapter answers with the bottleneck: shrink the channels with a cheap convolution, do the expensive 3×3 work at the narrow width, and grow back. You count the cost exactly, and see why the saving is about 17 times.

## What you already know

Four ideas carry this chapter:

- A convolution slides a small filter over a picture, and one filter produces one output channel (see "Sliding one small filter over a picture").
- A **channel** is one layer of numbers at every pixel, so a picture with 256 channels has 256 numbers at each pixel.
- A **multiply-add** is one multiplication followed by one addition, the unit used to count a layer's work (see "Count the parameters, flops, and memory of a network").
- A residual block adds its input to its output, so the two must have the same shape (see "Why deeper networks got worse, and the residual block").

## Pack, work, unpack

Picture a job that has to be done on a crowded workbench. You could spread the whole toolbox over the bench and work among it, which is slow. Or you could pack the few tools you need into a small case, work in the small space, and unpack afterward. A bottleneck block does the second thing. It squeezes 256 channels into 64, runs the 3×3 convolution on those 64, and expands back to 256. Figure 1 draws the widths of one block to scale.

```figure
<svg viewBox="0 0 640 340" role="img" aria-label="Channel widths through a bottleneck block: 256 channels in, 64 after a one-by-one convolution, 64 after a three-by-three convolution, 256 after a one-by-one convolution, with the shortcut passing over the top at full width">
<defs><marker id="c084-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<defs><marker id="c084-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<rect x="60" y="119.6" width="44" height="140.8" class="fill-soft-a stroke-a"/>
<text x="82.0" y="282" text-anchor="middle" class="fig-bold">256</text>
<text x="82.0" y="300" text-anchor="middle" class="fig-small">input</text>
<rect x="200" y="172.4" width="44" height="35.2" class="fill-soft-a stroke-a"/>
<text x="222.0" y="282" text-anchor="middle" class="fig-bold">64</text>
<text x="222.0" y="300" text-anchor="middle" class="fig-small">after 1×1</text>
<rect x="340" y="172.4" width="44" height="35.2" class="fill-soft-a stroke-a"/>
<text x="362.0" y="282" text-anchor="middle" class="fig-bold">64</text>
<text x="362.0" y="300" text-anchor="middle" class="fig-small">after 3×3</text>
<rect x="480" y="119.6" width="44" height="140.8" class="fill-soft-a stroke-a"/>
<text x="502.0" y="282" text-anchor="middle" class="fig-bold">256</text>
<text x="502.0" y="300" text-anchor="middle" class="fig-small">after 1×1</text>
<line x1="110" y1="190" x2="192" y2="190" class="stroke-ink" marker-end="url(#c084-arrow)"/>
<text x="151" y="180" text-anchor="middle" class="fig-small">1×1 conv: shrink</text>
<line x1="250" y1="190" x2="332" y2="190" class="stroke-ink" marker-end="url(#c084-arrow)"/>
<text x="291" y="180" text-anchor="middle" class="fig-small">3×3 conv: work</text>
<line x1="390" y1="190" x2="472" y2="190" class="stroke-ink" marker-end="url(#c084-arrow)"/>
<text x="431" y="180" text-anchor="middle" class="fig-small">1×1 conv: grow</text>
<path d="M82.0,119.6 L82.0,95.6 L585,95.6 L585,174" fill="none" class="stroke-a" stroke-width="2.5" marker-end="url(#c084-arrow-a)"/>
<text x="322" y="88" text-anchor="middle" class="fig-small">shortcut at full width: 256 channels, no cost</text>
<line x1="524" y1="190" x2="568" y2="190" class="stroke-ink" marker-end="url(#c084-arrow)"/>
<circle cx="585" cy="190" r="15" class="fill-paper stroke-ink"/><text x="585" y="196" text-anchor="middle" class="fig-bold">+</text>
</svg>
Figure 1. The channel widths of one bottleneck block, with heights drawn in proportion to the channel counts. The block shrinks 256 channels to 64 with a cheap 1×1 convolution, does its 3×3 work at that narrow width, and grows back to 256. The shortcut skips all of it at full width, so the plus sign adds two arrays of the same shape.
```

The shortcut skips all three steps at full width, so the block's input and output still have 256 channels and the plus sign works. The narrow part is hidden inside the branch.

Is squeezing to 64 channels safe? Two things make it so. The first 1×1 convolution is learned, so it picks which 64 mixtures of the 256 channels are worth keeping. And the shortcut still carries all 256 channels past the block untouched, so the branch only has to write a correction, not hold the whole picture. A narrow branch is a small pencil, and the page it edits stays full size.

Why is that cheaper? To answer it, count what a convolution costs.

## What a convolution costs

Take a convolution that keeps the image the same size, with stride 1, and ignore biases. Each output number is a sum of products. It reads a window of `k × k` pixels in all `c_in` input channels, which is $k^2 c_{\text{in}}$ products. There is one output number for every pixel and every output channel, which is $H \cdot W \cdot c_{\text{out}}$ of them. Multiply the two counts:

$$
\text{cost} = H \, W \, c_{\text{in}} \, c_{\text{out}} \, k^2
$$

Read it aloud: the cost is the number of output numbers, height times width times output channels, times the number of products each one needs, input channels times kernel area. A tiny case makes it concrete. Take a 2×2 picture with 4 channels in and 4 out. A 1×1 convolution costs $2 \cdot 2 \cdot 4 \cdot 4 \cdot 1 = 64$. A 3×3 convolution costs $2 \cdot 2 \cdot 4 \cdot 4 \cdot 9 = 576$, nine times as much, because each output reads nine times as many inputs.

A 1×1 convolution therefore has no factor of 9. It mixes the channels at each pixel and never looks at a neighbour, so it is a small matrix multiply applied at every pixel. In numpy:

```python
flops = h * w * c_in * c_out * k * k
```

Two facts follow from the formula. Cost is a product of both channel counts, so doubling every channel multiplies the cost by four, and halving them divides it by four. And a 3×3 costs nine times a 1×1 at the same channels. A bottleneck uses both facts.

## Count one bottleneck

The block has three steps: a 1×1 from $c_{\text{in}}$ to $c_{\text{mid}}$ channels, a 3×3 from $c_{\text{mid}}$ to $c_{\text{mid}}$, and a 1×1 from $c_{\text{mid}}$ to $c_{\text{out}}$. Add the three costs:

$$
\text{flops} = H W \left( c_{\text{in}} c_{\text{mid}} + 9\, c_{\text{mid}}^2 + c_{\text{mid}} c_{\text{out}} \right)
$$

Read it aloud: at every pixel, the shrink step costs input times middle, the 3×3 step costs nine times middle squared, and the grow step costs middle times output. The shortcut is a plain addition and is not in the count. Work it with $c_{\text{in}} = c_{\text{out}} = 256$, $c_{\text{mid}} = 64$, at 14×14:

1. The shrink is $256 \times 64 = 16{,}384$ per pixel.
2. The 3×3 is $9 \times 64 \times 64 = 36{,}864$ per pixel.
3. The grow is $64 \times 256 = 16{,}384$ per pixel.
4. Together that is $69{,}632$ per pixel. Over $14 \times 14 = 196$ pixels, the block costs $13{,}647{,}872$.

Two full-width 3×3 convolutions at 256 channels cost $2 \times 9 \times 256 \times 256 = 1{,}179{,}648$ per pixel, which is $231{,}211{,}008$ over the picture. The ratio is $231{,}211{,}008 / 13{,}647{,}872 \approx 16.9$. Figure 2 draws the two per-pixel costs to one scale.

```figure
<svg viewBox="0 0 640 260" role="img" aria-label="Bars drawn to one scale: two three-by-three convolutions at 256 channels cost 589,824 multiply-adds each per position, while the whole bottleneck costs 69,632 in three small segments">
<text x="20" y="30" class="fig-bold">Multiply-adds per position, to one scale</text>
<text x="185" y="83.0" text-anchor="end">two 3×3 convs</text>
<rect x="195" y="60" width="210.0" height="36" class="fill-b"/>
<rect x="405.0" y="60" width="210.0" height="36" class="fill-soft-b stroke-b"/>
<text x="300" y="83.0" text-anchor="middle" class="fig-small fig-mono fill-paper">589,824</text>
<text x="510" y="83.0" text-anchor="middle" class="fig-small fig-mono">589,824</text>
<text x="615" y="114" text-anchor="end" class="fig-small">total 1,179,648</text>
<text x="185" y="173.0" text-anchor="end">bottleneck</text>
<rect x="195.0" y="150" width="5.83" height="36" class="fill-a"/>
<rect x="200.8" y="150" width="13.12" height="36" class="fill-b"/>
<rect x="214.0" y="150" width="5.83" height="36" class="fill-a"/>
<text x="195" y="204" class="fig-small">16,384 + 36,864 + 16,384 = 69,632</text>
<text x="230" y="173.0" class="fig-small fig-bold">17 times shorter</text>
<rect x="195" y="215" width="14" height="14" class="fill-a"/><text x="215" y="227" class="fig-small">1×1 convolution</text>
<rect x="345" y="215" width="14" height="14" class="fill-b"/><text x="365" y="227" class="fig-small">3×3 convolution</text>
</svg>
Figure 2. Multiply-adds per position at 256 channels, drawn to one scale. The two full-width 3×3 convolutions take 1,179,648. The whole bottleneck takes 69,632, a bar about one-seventeenth as long. The orange 3×3 step is the largest part of the bottleneck, but at 64 channels it is small.
```

The ratio does not depend on the picture size or on the exact width, as long as the middle is a quarter of the outer width $C$. Per pixel, the bottleneck's weights add to $\tfrac{1}{4} C^2 + \tfrac{9}{16} C^2 + \tfrac{1}{4} C^2 = \tfrac{17}{16} C^2$, while two 3×3 convolutions have $18 C^2$. The ratio is $18 / (17/16) = 288/17 \approx 16.9$.

The middle width is a dial, and Figure 3 shows what it does. At 64 channels the block costs 69,632 per pixel. As the middle width grows, the 3×3 term grows with its square, and the cost climbs.

```figure
<svg viewBox="0 0 640 350" role="img" aria-label="Cost per position of a 256-channel bottleneck block as the middle width grows from 16 to 256, curving upward, against the flat cost of two full-width three-by-three convolutions">
<line x1="90" y1="290" x2="500" y2="290" class="stroke-muted"/><line x1="90" y1="290" x2="90" y2="40" class="stroke-muted"/>
<line x1="90" y1="290.0" x2="500" y2="290.0" class="stroke-grid"/><text x="82" y="294.0" text-anchor="end" class="fig-small">0.0M</text>
<line x1="90" y1="213.1" x2="500" y2="213.1" class="stroke-grid"/><text x="82" y="217.1" text-anchor="end" class="fig-small">0.4M</text>
<line x1="90" y1="136.2" x2="500" y2="136.2" class="stroke-grid"/><text x="82" y="140.2" text-anchor="end" class="fig-small">0.8M</text>
<line x1="90" y1="59.2" x2="500" y2="59.2" class="stroke-grid"/><text x="82" y="63.2" text-anchor="end" class="fig-small">1.2M</text>
<line x1="90.0" y1="290" x2="90.0" y2="295" class="stroke-muted"/><text x="90.0" y="310" text-anchor="middle" class="fig-small">16</text>
<line x1="172.0" y1="290" x2="172.0" y2="295" class="stroke-muted"/><text x="172.0" y="310" text-anchor="middle" class="fig-small">64</text>
<line x1="281.3" y1="290" x2="281.3" y2="295" class="stroke-muted"/><text x="281.3" y="310" text-anchor="middle" class="fig-small">128</text>
<line x1="390.7" y1="290" x2="390.7" y2="295" class="stroke-muted"/><text x="390.7" y="310" text-anchor="middle" class="fig-small">192</text>
<line x1="500.0" y1="290" x2="500.0" y2="295" class="stroke-muted"/><text x="500.0" y="310" text-anchor="middle" class="fig-small">256</text>
<text x="295.0" y="332" text-anchor="middle" class="fig-small">middle width C_mid (channels)</text>
<text x="90" y="26" class="fig-small">multiply-adds per position</text>
<polyline points="90.0,288.0 96.9,287.3 103.9,286.6 110.8,285.8 117.8,285.0 124.7,284.1 131.7,283.2 138.6,282.2 145.6,281.1 152.5,280.0 159.5,278.9 166.4,277.6 173.4,276.3 180.3,275.0 187.3,273.6 194.2,272.2 201.2,270.6 208.1,269.1 215.1,267.4 222.0,265.8 229.0,264.0 235.9,262.2 242.9,260.4 249.8,258.4 256.8,256.5 263.7,254.4 270.7,252.4 277.6,250.2 284.6,248.0 291.5,245.7 298.5,243.4 305.4,241.1 312.4,238.6 319.3,236.1 326.3,233.6 333.2,231.0 340.2,228.3 347.1,225.6 354.1,222.8 361.0,220.0 368.0,217.1 374.9,214.2 381.9,211.2 388.8,208.1 395.8,205.0 402.7,201.8 409.7,198.6 416.6,195.3 423.6,192.0 430.5,188.6 437.5,185.1 444.4,181.6 451.4,178.0 458.3,174.4 465.3,170.7 472.2,166.9 479.2,163.1 486.1,159.3 493.1,155.3 500.0,151.4" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="90" y1="63.1" x2="500" y2="63.1" class="stroke-b dashed" stroke-width="2"/>
<text x="98" y="55.1" class="fig-small">two 3×3 at 256 channels: 1,179,648</text>
<circle cx="172.0" cy="276.6" r="5" class="fill-a"/>
<text x="112.0" y="242.6" text-anchor="start" class="fig-small fig-bold">C_mid = 64: 69,632</text>
<circle cx="500.0" cy="151.4" r="5" class="fill-a"/>
<text x="492.0" y="139.4" text-anchor="end" class="fig-small fig-bold">C_mid = 256: 720,896</text>
</svg>
Figure 3. Cost per position of a 256-channel bottleneck as its middle width grows. The curve bends upward because the 3×3 term grows with the square of the middle width. At 64 channels the block costs 69,632. Even at 256 it costs 720,896, below two full-width 3×3 convolutions, because one of the three steps is a cheap 1×1.
```

> [!KEY]
> A convolution's cost is the product of its channel counts, so shrinking the channels with a cheap 1×1 makes the 3×3 work cost a quarter of the width squared. A bottleneck keeps the wide path between blocks and spends about 17 times fewer multiply-adds, which is how a network affords fifty or a hundred blocks.

> [!RECAP]
> Hold these points before you write code:
>
> - A convolution costs $H W c_{\text{in}} c_{\text{out}} k^2$ multiply-adds, which is one product per window entry for every output number.
> - A 1×1 convolution has $k^2 = 1$, so it has no factor of 9.
> - A bottleneck costs $HW(c_{\text{in}} c_{\text{mid}} + 9 c_{\text{mid}}^2 + c_{\text{mid}} c_{\text{out}})$.
> - For 256, 64, 256 at 14×14 the cost is 13,647,872, against 231,211,008 for two full-width 3×3 convolutions.
> - With the middle at a quarter of the outer width, the ratio is $288/17 \approx 16.9$ at any picture size.

## Build it step by step

The checks call the functions in this order:

1. **`conv_flops(c_in, c_out, k, h, w)`.** Return the multiply-add count of one convolution as an `int`, from the formula above. The usual mistake is to leave out the kernel area, which makes every convolution look like a 1×1.
2. **`bottleneck_flops(c_in, c_mid, c_out, h, w)`.** Return an `int`. Call `conv_flops` three times: a 1×1 from `c_in` to `c_mid`, a 3×3 from `c_mid` to `c_mid`, and a 1×1 from `c_mid` to `c_out`. The usual mistake is to forget the 9 on the middle step by using `k = 1` for all three.
3. **`bottleneck_savings(c, h, w)`.** Return a `float`: the cost of two 3×3 convolutions at `c` channels divided by `bottleneck_flops(c, c // 4, c, h, w)`. Build the first from `conv_flops` twice. The usual mistake is to divide the other way, which gives about 0.06 and not about 17.

## Cost and scaling

The shortcut is one addition per number, so it is free in this count. When a block changes the number of channels, though, the input and output no longer have the same shape and the plus sign breaks. A **projection shortcut** fixes that with a learned 1×1 convolution on the shortcut, which maps $c_{\text{in}}$ channels to $c_{\text{out}}$. It costs $H W c_{\text{in}} c_{\text{out}}$, which for 256 to 512 channels at 14×14 is 25,690,112. Networks use one only in the first block of a stage, where the width changes, and keep the plain shortcut elsewhere.

The two blocks compare as follows, for $C$ channels at height $H$ and width $W$:

| Block | Weights, ignoring biases | Multiply-adds |
|---|---|---|
| Two 3×3 at width $C$ | $18 C^2$ | $18 H W C^2$ |
| Bottleneck, $C \to C/4 \to C/4 \to C$ | $\tfrac{17}{16} C^2$ | $\tfrac{17}{16} H W C^2$ |

Memory behaves differently from multiply-adds. Backpropagation keeps every block's activations until the backward pass reaches that block, so memory grows in a straight line with depth. Twice the blocks need twice the stored activations, however cheap each block is to compute.

## Where this breaks

Three limits keep the saving honest:

- A multiply-add count is a guide and not a stopwatch. A layer's running time also depends on how much data it moves, so a block that is 17 times cheaper to count is not always 17 times faster to run.
- The count leaves out biases, normalisation, ReLUs, and the shortcut's addition. Those are small beside the convolutions, but they are not zero.
- The comparison holds the outer width fixed. Networks often use bottleneck blocks at a wider outer width than basic blocks, which spends the saving on width instead of keeping it as lower cost. A quarter is also a design choice, not a law: a narrower middle is cheaper and sees fewer channels.

## The exercise

You build `conv_flops`, `bottleneck_flops`, and `bottleneck_savings`. The checks match `conv_flops` on the 2×2 example above and on larger sizes, and return plain `int` values. They match `bottleneck_flops` on three hand-computed cases, including one where the input and output widths differ. They confirm the 256, 64, 256 case at 14×14 and the ratio near 16.9, which must hold at several widths and picture sizes.
