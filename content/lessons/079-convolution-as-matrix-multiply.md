---
title: Turning a convolution into one matrix multiply
slug: 079-convolution-as-matrix-multiply
part: "Part 5 — Vision and sequences"
summary: Lay every window out as a column so a whole convolution becomes one matrix multiply, and see what that costs in memory.
minutes: 40
gpu: false
objectives:
  - Explain why a loop over output positions is too slow, and why a matrix multiply is the fast alternative.
  - Build the im2col array with one strided slice per kernel offset, and state the order in which each patch is flattened.
  - Write the forward pass of a convolution as one matrix multiply, and compare it with the loop version.
  - Estimate the extra memory im2col needs, and write a 1 by 1 convolution as a plain matrix multiply.
prereqs:
  - 077-sliding-a-filter
  - 078-output-size-and-receptive-field
exercise: 079-convolution-as-matrix-multiply
---

# Turning a convolution into one matrix multiply

Take one realistic layer: 64 channels in, 64 channels out, 3 by 3 filters, on a 56 by 56 feature map. The loop version from "Sliding one small filter over a picture" visits every output channel and every output position, which is 64 × 56 × 56 = 200,704 steps for one image. Each step cuts out a patch, multiplies it by a kernel, and adds. A training run processes millions of images, so the loop spends its life on Python bookkeeping, and the arithmetic itself is a small part of the time.

Hardware does one kind of arithmetic extremely well: the matrix multiply. Processors and GPUs ship with libraries tuned for years to run it at close to their peak speed. You could write the loop in a faster language, and libraries do for some cases, but a faster loop is still a loop over 200,704 small jobs. The better move is to rearrange the data so the whole job is one big matrix multiply, which is the one thing the library already does well. This chapter builds that rearrangement, called im2col, and shows what it costs.

## What you already know

You already use these ideas:

- A matrix product has entry $(i, j)$ equal to the dot product of row $i$ of the left matrix with column $j$ of the right one.
- `a.reshape(...)` reads the numbers in order, last axis fastest, and cuts them into a new shape without changing the order.
- A slice with a step, `a[start:stop:step]`, keeps every `step`-th entry.
- A convolution output is a dot product of a kernel with a patch, plus a bias, as in "Sliding one small filter over a picture".
- The output length formula from "How big is the output and how far can a unit see".

## Flatten the patch, flatten the kernel

One output value is a dot product of the kernel with a patch. A dot product does not care about shape: flatten the kernel into a row of numbers and the patch into a column of numbers, pair them up, and add. Next, take every patch, flatten each into a column, and set the columns side by side. A row times that matrix is the dot product of the row with every column, which is every output value in one product. Figure 1 shows this on a tiny case.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Four 2 by 2 patches of a 3 by 3 image laid out as columns and multiplied by the kernel row">
<defs>
<marker id="c079-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<text x="250" y="20" class="fig-bold">Each patch, cut out of the image</text>
<rect x="252" y="34" width="16" height="16" class="fill-soft-a stroke-grid"/>
<text x="260" y="46.5" text-anchor="middle" class="fig-small">1</text>
<rect x="268" y="34" width="16" height="16" class="fill-soft-a stroke-grid"/>
<text x="276" y="46.5" text-anchor="middle" class="fig-small">2</text>
<rect x="284" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="292" y="46.5" text-anchor="middle" class="fig-small">3</text>
<rect x="252" y="50" width="16" height="16" class="fill-soft-a stroke-grid"/>
<text x="260" y="62.5" text-anchor="middle" class="fig-small">4</text>
<rect x="268" y="50" width="16" height="16" class="fill-soft-a stroke-grid"/>
<text x="276" y="62.5" text-anchor="middle" class="fig-small">5</text>
<rect x="284" y="50" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="292" y="62.5" text-anchor="middle" class="fig-small">6</text>
<rect x="252" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="260" y="78.5" text-anchor="middle" class="fig-small">7</text>
<rect x="268" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="276" y="78.5" text-anchor="middle" class="fig-small">8</text>
<rect x="284" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="292" y="78.5" text-anchor="middle" class="fig-small">9</text>
<text x="276" y="96" text-anchor="middle" class="fig-small">patch 1</text>
<line x1="276" y1="102" x2="276" y2="124" class="stroke-muted" marker-end="url(#c079-arrow)"/>
<rect x="304" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="312" y="46.5" text-anchor="middle" class="fig-small">1</text>
<rect x="320" y="34" width="16" height="16" class="fill-soft-b stroke-grid"/>
<text x="328" y="46.5" text-anchor="middle" class="fig-small">2</text>
<rect x="336" y="34" width="16" height="16" class="fill-soft-b stroke-grid"/>
<text x="344" y="46.5" text-anchor="middle" class="fig-small">3</text>
<rect x="304" y="50" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="312" y="62.5" text-anchor="middle" class="fig-small">4</text>
<rect x="320" y="50" width="16" height="16" class="fill-soft-b stroke-grid"/>
<text x="328" y="62.5" text-anchor="middle" class="fig-small">5</text>
<rect x="336" y="50" width="16" height="16" class="fill-soft-b stroke-grid"/>
<text x="344" y="62.5" text-anchor="middle" class="fig-small">6</text>
<rect x="304" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="312" y="78.5" text-anchor="middle" class="fig-small">7</text>
<rect x="320" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="328" y="78.5" text-anchor="middle" class="fig-small">8</text>
<rect x="336" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="344" y="78.5" text-anchor="middle" class="fig-small">9</text>
<text x="328" y="96" text-anchor="middle" class="fig-small">patch 2</text>
<line x1="328" y1="102" x2="328" y2="124" class="stroke-muted" marker-end="url(#c079-arrow)"/>
<rect x="356" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="364" y="46.5" text-anchor="middle" class="fig-small">1</text>
<rect x="372" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="380" y="46.5" text-anchor="middle" class="fig-small">2</text>
<rect x="388" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="396" y="46.5" text-anchor="middle" class="fig-small">3</text>
<rect x="356" y="50" width="16" height="16" class="fill-soft-c stroke-grid"/>
<text x="364" y="62.5" text-anchor="middle" class="fig-small">4</text>
<rect x="372" y="50" width="16" height="16" class="fill-soft-c stroke-grid"/>
<text x="380" y="62.5" text-anchor="middle" class="fig-small">5</text>
<rect x="388" y="50" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="396" y="62.5" text-anchor="middle" class="fig-small">6</text>
<rect x="356" y="66" width="16" height="16" class="fill-soft-c stroke-grid"/>
<text x="364" y="78.5" text-anchor="middle" class="fig-small">7</text>
<rect x="372" y="66" width="16" height="16" class="fill-soft-c stroke-grid"/>
<text x="380" y="78.5" text-anchor="middle" class="fig-small">8</text>
<rect x="388" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="396" y="78.5" text-anchor="middle" class="fig-small">9</text>
<text x="380" y="96" text-anchor="middle" class="fig-small">patch 3</text>
<line x1="380" y1="102" x2="380" y2="124" class="stroke-muted" marker-end="url(#c079-arrow)"/>
<rect x="408" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="416" y="46.5" text-anchor="middle" class="fig-small">1</text>
<rect x="424" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="432" y="46.5" text-anchor="middle" class="fig-small">2</text>
<rect x="440" y="34" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="448" y="46.5" text-anchor="middle" class="fig-small">3</text>
<rect x="408" y="50" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="416" y="62.5" text-anchor="middle" class="fig-small">4</text>
<rect x="424" y="50" width="16" height="16" class="fill-soft-d stroke-grid"/>
<text x="432" y="62.5" text-anchor="middle" class="fig-small">5</text>
<rect x="440" y="50" width="16" height="16" class="fill-soft-d stroke-grid"/>
<text x="448" y="62.5" text-anchor="middle" class="fig-small">6</text>
<rect x="408" y="66" width="16" height="16" class="fill-paper stroke-grid"/>
<text x="416" y="78.5" text-anchor="middle" class="fig-small">7</text>
<rect x="424" y="66" width="16" height="16" class="fill-soft-d stroke-grid"/>
<text x="432" y="78.5" text-anchor="middle" class="fig-small">8</text>
<rect x="440" y="66" width="16" height="16" class="fill-soft-d stroke-grid"/>
<text x="448" y="78.5" text-anchor="middle" class="fig-small">9</text>
<text x="432" y="96" text-anchor="middle" class="fig-small">patch 4</text>
<line x1="432" y1="102" x2="432" y2="124" class="stroke-muted" marker-end="url(#c079-arrow)"/>
<rect x="250" y="130" width="52" height="30" class="fill-soft-a stroke-grid"/>
<text x="276" y="149.5" text-anchor="middle">1</text>
<rect x="302" y="130" width="52" height="30" class="fill-soft-b stroke-grid"/>
<text x="328" y="149.5" text-anchor="middle">2</text>
<rect x="354" y="130" width="52" height="30" class="fill-soft-c stroke-grid"/>
<text x="380" y="149.5" text-anchor="middle">4</text>
<rect x="406" y="130" width="52" height="30" class="fill-soft-d stroke-grid"/>
<text x="432" y="149.5" text-anchor="middle">5</text>
<rect x="250" y="160" width="52" height="30" class="fill-soft-a stroke-grid"/>
<text x="276" y="179.5" text-anchor="middle">2</text>
<rect x="302" y="160" width="52" height="30" class="fill-soft-b stroke-grid"/>
<text x="328" y="179.5" text-anchor="middle">3</text>
<rect x="354" y="160" width="52" height="30" class="fill-soft-c stroke-grid"/>
<text x="380" y="179.5" text-anchor="middle">5</text>
<rect x="406" y="160" width="52" height="30" class="fill-soft-d stroke-grid"/>
<text x="432" y="179.5" text-anchor="middle">6</text>
<rect x="250" y="190" width="52" height="30" class="fill-soft-a stroke-grid"/>
<text x="276" y="209.5" text-anchor="middle">4</text>
<rect x="302" y="190" width="52" height="30" class="fill-soft-b stroke-grid"/>
<text x="328" y="209.5" text-anchor="middle">5</text>
<rect x="354" y="190" width="52" height="30" class="fill-soft-c stroke-grid"/>
<text x="380" y="209.5" text-anchor="middle">7</text>
<rect x="406" y="190" width="52" height="30" class="fill-soft-d stroke-grid"/>
<text x="432" y="209.5" text-anchor="middle">8</text>
<rect x="250" y="220" width="52" height="30" class="fill-soft-a stroke-grid"/>
<text x="276" y="239.5" text-anchor="middle">5</text>
<rect x="302" y="220" width="52" height="30" class="fill-soft-b stroke-grid"/>
<text x="328" y="239.5" text-anchor="middle">6</text>
<rect x="354" y="220" width="52" height="30" class="fill-soft-c stroke-grid"/>
<text x="380" y="239.5" text-anchor="middle">8</text>
<rect x="406" y="220" width="52" height="30" class="fill-soft-d stroke-grid"/>
<text x="432" y="239.5" text-anchor="middle">9</text>
<text x="468" y="146" class="fig-small">one column per patch,</text>
<text x="468" y="160" class="fig-small">flattened row by row</text>
<text x="16" y="276" class="fig-small">kernel as a row</text>
<rect x="16" y="284" width="30" height="30" class="fill-paper stroke-grid"/>
<text x="31" y="303.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="46" y="284" width="30" height="30" class="fill-paper stroke-grid"/>
<text x="61" y="303.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="76" y="284" width="30" height="30" class="fill-paper stroke-grid"/>
<text x="91" y="303.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="106" y="284" width="30" height="30" class="fill-paper stroke-grid"/>
<text x="121" y="303.5" text-anchor="middle" class="fig-bold">4</text>
<text x="190" y="304" text-anchor="middle" class="fig-bold">× matrix =</text>
<rect x="250" y="284" width="52" height="30" class="fill-soft-a stroke-grid"/>
<text x="276" y="303.5" text-anchor="middle" class="fig-bold">37</text>
<rect x="302" y="284" width="52" height="30" class="fill-soft-b stroke-grid"/>
<text x="328" y="303.5" text-anchor="middle" class="fig-bold">47</text>
<rect x="354" y="284" width="52" height="30" class="fill-soft-c stroke-grid"/>
<text x="380" y="303.5" text-anchor="middle" class="fig-bold">67</text>
<rect x="406" y="284" width="52" height="30" class="fill-soft-d stroke-grid"/>
<text x="432" y="303.5" text-anchor="middle" class="fig-bold">77</text>
<text x="468" y="304" class="fig-small">4 outputs</text>
</svg>
Figure 1. The 3 by 3 image (1 to 9) holds four 2 by 2 patches, drawn in four colours. Each patch becomes a column of four numbers, and the kernel (1, 2, 3, 4) is a row. The row times the matrix gives all four outputs at once: 37, 47, 67, and 77. Notice that pixel 5 appears in every column, because every patch contains it.
```

Take the 3 by 3 image holding 1 to 9 and the 2 by 2 kernel with rows (1, 2) and (3, 4). The top-left patch has pixels 1, 2, 4, 5, and the kernel flattened row by row is (1, 2, 3, 4). The dot product is 1 + 4 + 12 + 20 = 37. The other three patches give 47, 67, and 77, and they are exactly what the sliding filter would produce. If the layer has several filters, you stack their flattened kernels as the rows of a matrix $W_{\text{mat}}$, and one product gives every output channel at every position:

$$
Y_{\text{mat}} = W_{\text{mat}}\,\text{col} + b
$$

Read it aloud: each row of $W_{\text{mat}}$ is one flattened kernel, each column of `col` is one flattened patch, entry $(o, l)$ of the product is the score of kernel $o$ on patch $l$, and the bias $b_o$ is added to every entry in row $o$. The "col" in the name is the point of **im2col**, short for "image to columns".

## Keep the flatten orders the same

One detail decides whether the rearrangement is right. The dot product pairs the $t$-th entry of the kernel row with the $t$-th entry of the patch column. If you flatten the kernel in one order and the patch in another, the pairs are wrong. For the patch (1, 2, 4, 5) read column by column instead of row by row, you get (1, 4, 2, 5), and the dot product with (1, 2, 3, 4) is 1 + 8 + 6 + 20 = 35, not 37. No error message warns you. The output has the wrong numbers, and nothing else tells you.

The kernel array `w` has shape `(C_out, C_in, k, k)`, and `w.reshape(C_out, -1)` flattens each kernel in the order channel, kernel row, kernel column. The patch must use the same order. Figure 2 shows the complete `col` array for an image with two channels, where the second channel is the first plus 9. Each column lists the four taps of channel 0, then the four taps of channel 1. Each column also belongs to one output position: column $l$ is the patch at position $(l \,//\, W_{\text{out}},\; l \bmod W_{\text{out}})$, because the positions are numbered row by row.

```figure
<svg viewBox="0 0 640 330" role="img" aria-label="The im2col array for a two-channel 3 by 3 image and a 2 by 2 kernel: eight rows, four columns">
<text x="20" y="20" class="fig-bold">col has shape (N, C·k·k, L) = (1, 8, 4) here</text>
<text x="178" y="48" text-anchor="middle" class="fig-small">l = 0</text>
<text x="178" y="62" text-anchor="middle" class="fig-small fig-mono">(0, 0)</text>
<text x="234" y="48" text-anchor="middle" class="fig-small">l = 1</text>
<text x="234" y="62" text-anchor="middle" class="fig-small fig-mono">(0, 1)</text>
<text x="290" y="48" text-anchor="middle" class="fig-small">l = 2</text>
<text x="290" y="62" text-anchor="middle" class="fig-small fig-mono">(1, 0)</text>
<text x="346" y="48" text-anchor="middle" class="fig-small">l = 3</text>
<text x="346" y="62" text-anchor="middle" class="fig-small fig-mono">(1, 1)</text>
<rect x="150" y="70" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="178" y="86.5" text-anchor="middle">1</text>
<rect x="206" y="70" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="234" y="86.5" text-anchor="middle">2</text>
<rect x="262" y="70" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="290" y="86.5" text-anchor="middle">4</text>
<rect x="318" y="70" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="346" y="86.5" text-anchor="middle">5</text>
<rect x="150" y="94" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="178" y="110.5" text-anchor="middle">2</text>
<rect x="206" y="94" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="234" y="110.5" text-anchor="middle">3</text>
<rect x="262" y="94" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="290" y="110.5" text-anchor="middle">5</text>
<rect x="318" y="94" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="346" y="110.5" text-anchor="middle">6</text>
<rect x="150" y="118" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="178" y="134.5" text-anchor="middle">4</text>
<rect x="206" y="118" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="234" y="134.5" text-anchor="middle">5</text>
<rect x="262" y="118" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="290" y="134.5" text-anchor="middle">7</text>
<rect x="318" y="118" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="346" y="134.5" text-anchor="middle">8</text>
<rect x="150" y="142" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="178" y="158.5" text-anchor="middle">5</text>
<rect x="206" y="142" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="234" y="158.5" text-anchor="middle">6</text>
<rect x="262" y="142" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="290" y="158.5" text-anchor="middle">8</text>
<rect x="318" y="142" width="56" height="24" class="fill-soft-a stroke-grid"/>
<text x="346" y="158.5" text-anchor="middle">9</text>
<rect x="150" y="166" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="178" y="182.5" text-anchor="middle">10</text>
<rect x="206" y="166" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="234" y="182.5" text-anchor="middle">11</text>
<rect x="262" y="166" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="290" y="182.5" text-anchor="middle">13</text>
<rect x="318" y="166" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="346" y="182.5" text-anchor="middle">14</text>
<rect x="150" y="190" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="178" y="206.5" text-anchor="middle">11</text>
<rect x="206" y="190" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="234" y="206.5" text-anchor="middle">12</text>
<rect x="262" y="190" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="290" y="206.5" text-anchor="middle">14</text>
<rect x="318" y="190" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="346" y="206.5" text-anchor="middle">15</text>
<rect x="150" y="214" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="178" y="230.5" text-anchor="middle">13</text>
<rect x="206" y="214" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="234" y="230.5" text-anchor="middle">14</text>
<rect x="262" y="214" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="290" y="230.5" text-anchor="middle">16</text>
<rect x="318" y="214" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="346" y="230.5" text-anchor="middle">17</text>
<rect x="150" y="238" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="178" y="254.5" text-anchor="middle">14</text>
<rect x="206" y="238" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="234" y="254.5" text-anchor="middle">15</text>
<rect x="262" y="238" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="290" y="254.5" text-anchor="middle">17</text>
<rect x="318" y="238" width="56" height="24" class="fill-soft-c stroke-grid"/>
<text x="346" y="254.5" text-anchor="middle">18</text>
<text x="140" y="86" text-anchor="end" class="fig-small fig-mono">ch0 r0 c0</text>
<text x="140" y="110" text-anchor="end" class="fig-small fig-mono">ch0 r0 c1</text>
<text x="140" y="134" text-anchor="end" class="fig-small fig-mono">ch0 r1 c0</text>
<text x="140" y="158" text-anchor="end" class="fig-small fig-mono">ch0 r1 c1</text>
<text x="140" y="182" text-anchor="end" class="fig-small fig-mono">ch1 r0 c0</text>
<text x="140" y="206" text-anchor="end" class="fig-small fig-mono">ch1 r0 c1</text>
<text x="140" y="230" text-anchor="end" class="fig-small fig-mono">ch1 r1 c0</text>
<text x="140" y="254" text-anchor="end" class="fig-small fig-mono">ch1 r1 c1</text>
<line x1="150" y1="166" x2="374" y2="166" class="stroke-ink"/>
<text x="388" y="122" class="fig-bold">channel 0 (rows 0 to 3)</text>
<text x="388" y="218" class="fig-bold">channel 1 (rows 4 to 7)</text>
<text x="388" y="140" class="fig-small">taps in order: row 0 col 0,</text>
<text x="388" y="154" class="fig-small">row 0 col 1, row 1 col 0, row 1 col 1</text>
<text x="20" y="288" class="fig-small">Row labels: channel, kernel row, kernel column (ch, r, c).</text>
<text x="20" y="304" class="fig-small">Column labels: l, then the output position (l // W_out, l mod W_out), with W_out = 2.</text>
</svg>
Figure 2. The columns of im2col for two channels (the second channel is the first plus 9). Each column is one patch: all four taps of channel 0, then all four taps of channel 1. The kernel matrix must be flattened in the same order, so that row r of the kernel lines up with row r of every column.
```

For a batch, `col` has shape `(N, C * k * k, H_out * W_out)`: one such matrix per image.

## Build the columns without a loop over positions

A loop over output positions would bring back the slowness you are trying to leave. Instead, loop over the $k^2$ kernel offsets, and for each offset gather that tap's pixel for every output position at once. One slice is enough, for this reason. Output position $(i, j)$ reads pixel $(i \cdot s + d_h,\; j \cdot s + d_w)$ at kernel tap $(d_h, d_w)$, where $s$ is the stride. As $i$ and $j$ run over all the outputs, those pixels are every $s$-th pixel of the image starting at $(d_h, d_w)$, which is a strided slice:

```python
x_padded[:, :, dh:dh + stride * h_out:stride, dw:dw + stride * w_out:stride]
```

That slice has shape `(N, C, H_out, W_out)`: the tap $(d_h, d_w)$ of every patch. Store the $k^2$ slices in an array of shape `(N, C, k, k, H_out, W_out)`, at position `[:, :, dh, dw]`. Then one `reshape` to `(N, C * k * k, H_out * W_out)` merges `C, k, k` into the row axis, in the order channel, kernel row, kernel column, and merges `H_out, W_out` into the column axis, in the order of Figure 2. Use `np.pad` to add the zero border first, and take `H_out` and `W_out` from `conv_output_shape`.

> [!WARNING]
> The slice stops at `dh + stride * h_out`, not at `h`. If you stop at `h`, the slice holds the wrong number of entries (at stride 1, `h - dh` instead of `h_out`), and the assignment fails with a shape error.

## Multiply

With `col` built, the forward pass is one line. The weights flatten to `w_mat = w.reshape(C_out, -1)`, and the multiply is a batched matrix product: for each image, `w_mat` of shape `(C_out, C_in * k * k)` times that image's `col` of shape `(C_in * k * k, L)`.

You can write it with `w_mat @ col`, and you can write it with `np.einsum`. In an einsum string, each letter names an axis. A letter that appears on the left but not on the right is summed over. The string `"oc,ncl->nol"` says: take `w_mat` with axes `(o, c)` and `col` with axes `(n, c, l)`, sum over `c`, and return an array with axes `(n, o, l)`. That is the matrix product, for every image `n` at once:

```python
out = np.einsum("oc,ncl->nol", w_mat, col) + b[None, :, None]
out = out.reshape(N, C_out, H_out, W_out)
```

The bias has one entry per output channel, so `b[None, :, None]` stretches it across the batch and across the positions. The reshape at the end turns `L` back into rows and columns of the output map.

> [!KEY]
> Flatten each kernel into a row of `w_mat`, flatten each window into a column of `col` in the same order (channel, kernel row, kernel column), and one matrix multiply computes every output value of the layer. The loop over 200,704 positions becomes a loop over 9 kernel offsets.

## What the rearrangement costs

Every pixel lands in several columns, because every window that contains it copies it. With a 3 by 3 kernel at stride 1 and padding 1, each pixel is copied 9 times, so `col` holds about nine times as many numbers as the image. For the 64-channel, 56 by 56 layer, the image has 64 × 56 × 56 = 200,704 numbers and `col` has 64 × 9 × 56 × 56 = 1,806,336. A stride of $s$ divides the count of columns by $s^2$, because there are that many fewer windows. This is the trade: you spend memory to get speed.

The arithmetic is the same in either version. The layer does $C_{\text{out}} \cdot C_{\text{in}} k^2$ multiply-adds per output position, which for this layer is 64 × 576 × 3,136 = 115,605,504 per image. Figure 3 puts the two costs side by side. When memory gets tight, libraries build the columns in pieces, or use other algorithms. The idea in this chapter is the clearest one, and the one you can check against a loop.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Memory used by the input and by the im2col array, and the work done by a loop and by one matrix multiply">
<text x="20" y="24" class="fig-bold">Memory: numbers held for one 56 by 56 image, 64 channels</text>
<rect x="20" y="40" width="44.4" height="26" class="fill-soft-a stroke-a"/>
<text x="74.4" y="58">input: 200,704</text>
<rect x="20" y="78" width="400" height="26" class="fill-soft-b stroke-b"/>
<text x="28" y="96">col, for a 3 by 3 kernel: 1,806,336 (9 times as many)</text>
<text x="20" y="150" class="fig-bold">Work: the same 115,605,504 multiply-adds either way</text>
<rect x="20" y="166" width="560" height="44" class="fill-soft-a stroke-a"/>
<text x="30" y="184">Loop: 200,704 Python steps per image,</text>
<text x="30" y="202" class="fig-small">each a 576-term dot product, one output value at a time</text>
<rect x="20" y="222" width="560" height="44" class="fill-soft-b stroke-b"/>
<text x="30" y="240">im2col: 9 slice copies, then one matrix multiply</text>
<text x="30" y="258" class="fig-small">(64 by 576) times (576 by 3,136), handed to optimised code</text>
</svg>
Figure 3. The price of the rearrangement. The im2col array holds each pixel about k² times (here 9), so memory rises ninefold. The arithmetic does not change. What changes is who does it: a Python loop of 200,704 steps, or one matrix multiply that tuned library code runs at full speed.
```

The same view explains the 1 by 1 convolution from "Sliding one small filter over a picture". With $k = 1$, a window is a single pixel, so `col` is the image with its pixels laid out as columns, and it needs no extra memory. The whole layer is `w.reshape(C_out, C_in) @ x.reshape(N, C_in, H * W)`: a dense layer applied to the channels of every pixel.

> [!RECAP]
> Hold these points before you write code:
>
> - One output is a dot product of a flattened kernel with a flattened patch, so many outputs are a matrix product.
> - `col` has shape `(N, C * k * k, H_out * W_out)`, and column $l$ is the patch at $(l // W_{\text{out}}, l \bmod W_{\text{out}})$.
> - Patch and kernel use the same flatten order: channel, kernel row, kernel column.
> - One strided slice per kernel offset fills `col`, and the forward pass is `w.reshape(C_out, -1)` times `col`, plus the bias.
> - `col` holds about $k^2$ times as many numbers as the image, and a 1 by 1 convolution needs no `col` at all.

## Build it step by step

The checks call the functions in this order. The starter already contains the loop version `conv2d_naive` and `conv_output_shape`, solved, from the earlier chapters.

1. **`im2col(x, kernel_size, stride, padding)`.** `x` has shape `(N, C, H, W)`. Return `(N, C * k * k, H_out * W_out)`. Pad with `np.pad`, fill an `(N, C, k, k, H_out, W_out)` array with one strided slice per kernel offset, and reshape. The most common mistake is the slice end, which must be `dh + stride * h_out`.
2. **`conv2d_forward(x, w, b, stride, padding)`.** Return `(N, C_out, H_out, W_out)`. Build `col`, multiply by `w.reshape(C_out, -1)`, add `b[None, :, None]`, and reshape. The most common mistake is adding the bias along the wrong axis.
3. **`conv1x1_as_matmul(x, w, b)`.** `w` has shape `(C_out, C_in, 1, 1)`. Reshape `x` to `(N, C_in, H * W)`, multiply by `w.reshape(C_out, C_in)`, add the bias, and reshape back. Do not call `im2col`. The most common mistake is flattening the image as `(N, H * W, C_in)`, which pairs pixels with channels in the wrong order.

## The exercise

You implement `im2col`, `conv2d_forward`, and `conv1x1_as_matmul` in `starter.py`. The checks compare `im2col` with the windows cut out by hand, including the channel-row-column order, and with stride, padding, and a non-square image. They compare `conv2d_forward` with an independent reference built from numpy's sliding windows, with several strides and paddings, and with the loop version from "Sliding one small filter over a picture". The last check confirms that `conv1x1_as_matmul` equals a convolution with a 1 by 1 kernel.
