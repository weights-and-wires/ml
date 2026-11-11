---
title: Convolution as a matrix multiply, forward and backward
slug: 045-im2col-and-convolution-backward
part: "Part 5 — Vision and sequences"
summary: Lay every window out as a column so a convolution is one matrix multiply, then send the gradient back with col2im.
minutes: 75
gpu: false
objectives:
  - Build the im2col array with one strided slice per kernel offset, and state the order in which each patch is flattened.
  - Write the forward pass of a convolution as one matrix multiply, and estimate the extra memory it needs.
  - Explain why a shared weight's gradient adds up over positions, and a pixel's gradient adds up over the windows that read it.
  - Write col2im as a scatter-add, and test it against im2col with the adjoint identity.
  - Check dx, dw, and db against finite differences.
prereqs:
  - 044-convolution-and-receptive-field
  - 035-backpropagation
exercise: 045-im2col-and-convolution-backward
---

# Convolution as a matrix multiply, forward and backward

Take one realistic layer: 64 channels in, 64 channels out, 3 by 3 filters, on a 56 by 56 feature map. The loop version from [sliding one small filter over a picture](/lessons/044-convolution-and-receptive-field) visits every output channel and every output position, which is 64 × 56 × 56 = 200,704 steps for one image. Each step cuts out a patch, multiplies it by a kernel, and adds. A training run processes millions of images, so the loop spends its life on Python bookkeeping, and the arithmetic is a small part of the time.

Hardware does one kind of arithmetic extremely well: the matrix multiply. A faster loop is still a loop over 200,704 small jobs. The better move is to rearrange the data so the whole job is one big matrix multiply, which tuned libraries already run close to peak speed. That rearrangement is called im2col, and the first half of this chapter builds it.

Training then asks a second question. When the loss says the output was too low, whom does it blame? Each of the nine weights in a filter influenced thousands of outputs, and each pixel was read by up to nine windows. Nudging one weight and rerunning the forward pass would cost 36,864 passes for this layer, one per weight. The fast way follows the blame backward through the same steps as the forward pass, and the answer is one rule: where something was shared, the blame adds up. The second half builds that backward pass.

## What you already know

- A matrix product has entry $(i, j)$ equal to the dot product of row $i$ of the left matrix with column $j$ of the right one.
- `a.reshape(...)` reads the numbers in order, last axis fastest, and cuts them into a new shape. A slice `a[start:stop:step]` keeps every `step`-th entry.
- A convolution output is a dot product of a kernel with a patch, plus a bias, and the output length follows the formula from [the receptive field chapter](/lessons/044-convolution-and-receptive-field).
- For a matrix product $Y = WX$, the gradients are $dW = dY\,X^\top$ and $dX = W^\top dY$, and a value that feeds several places collects the sum of the gradients from each place ([backpropagation](/lessons/035-backpropagation)).
- A finite difference, $(L(x + \epsilon) - L(x - \epsilon)) / 2\epsilon$, estimates a gradient numerically.

## Flatten the patch, flatten the kernel

One output value is a dot product of the kernel with a patch. A dot product does not care about shape: flatten the kernel into a row of numbers and the patch into a column of numbers, pair them up, and add. Next, take every patch, flatten each into a column, and set the columns side by side. A row times that matrix is the dot product of the row with every column, which is every output value in one product. Figure 1 shows this on a tiny case.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Four 2 by 2 patches of a 3 by 3 image laid out as columns and multiplied by the kernel row">
<defs>
<marker id="c035-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
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
<line x1="276" y1="102" x2="276" y2="124" class="stroke-muted" marker-end="url(#c035-arrow)"/>
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
<line x1="328" y1="102" x2="328" y2="124" class="stroke-muted" marker-end="url(#c035-arrow)"/>
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
<line x1="380" y1="102" x2="380" y2="124" class="stroke-muted" marker-end="url(#c035-arrow)"/>
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
<line x1="432" y1="102" x2="432" y2="124" class="stroke-muted" marker-end="url(#c035-arrow)"/>
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

Take the 3 by 3 image holding 1 to 9 and the 2 by 2 kernel with rows (1, 2) and (3, 4). The top-left patch has pixels 1, 2, 4, 5, and the kernel flattened row by row is (1, 2, 3, 4). The dot product is 1 + 4 + 12 + 20 = 37. The other three patches give 47, 67, and 77, which is exactly what the sliding filter produces. If the layer has several filters, you stack their flattened kernels as the rows of a matrix $W_{\text{mat}}$, and one product gives every output channel at every position:

$$
Y_{\text{mat}} = W_{\text{mat}}\,\text{col} + b
$$

Read it aloud: each row of $W_{\text{mat}}$ is one flattened kernel, each column of `col` is one flattened patch, entry $(o, l)$ of the product is the score of kernel $o$ on patch $l$, and the bias $b_o$ is added to every entry in row $o$. The name **im2col** is short for "image to columns".

### Keep the flatten orders the same

One detail decides whether the rearrangement is right. The dot product pairs the $t$-th entry of the kernel row with the $t$-th entry of the patch column. If you flatten the kernel in one order and the patch in another, the pairs are wrong. For the patch (1, 2, 4, 5) read column by column instead of row by row, you get (1, 4, 2, 5), and the dot product with (1, 2, 3, 4) is 1 + 8 + 6 + 20 = 35, not 37. No error message warns you.

The kernel array `w` has shape `(C_out, C_in, k, k)`, and `w.reshape(C_out, -1)` flattens each kernel in the order channel, kernel row, kernel column. Each patch must use the same order: all the taps of channel 0, then all the taps of channel 1, and so on. Each column also belongs to one output position. Column $l$ is the patch at position $(l \,//\, W_{\text{out}},\; l \bmod W_{\text{out}})$, because positions are numbered row by row. For a batch, `col` has shape `(N, C * k * k, H_out * W_out)`: one such matrix per image.

### Build the columns without a loop over positions

A loop over output positions would bring back the slowness you are leaving. Instead, loop over the $k^2$ kernel offsets, and for each offset gather that tap's pixel for every output position at once. Output position $(i, j)$ reads pixel $(i \cdot s + d_h,\; j \cdot s + d_w)$ at kernel tap $(d_h, d_w)$, where $s$ is the stride. As $i$ and $j$ run over all the outputs, those pixels are every $s$-th pixel of the image starting at $(d_h, d_w)$, which is a strided slice:

```python
x_padded[:, :, dh:dh + stride * h_out:stride, dw:dw + stride * w_out:stride]
```

That slice has shape `(N, C, H_out, W_out)`: the tap $(d_h, d_w)$ of every patch. Store the $k^2$ slices in an array of shape `(N, C, k, k, H_out, W_out)`, at position `[:, :, dh, dw]`. One `reshape` to `(N, C * k * k, H_out * W_out)` then merges `C, k, k` into the row axis, in the order channel, kernel row, kernel column, and merges `H_out, W_out` into the column axis. Use `np.pad` to add the zero border first.

> [!WARNING]
> The slice stops at `dh + stride * h_out`, not at `h`. If you stop at `h`, the slice holds the wrong number of entries (at stride 1, `h - dh` instead of `h_out`), and the assignment fails with a shape error.

### Multiply

With `col` built, the forward pass is one line. The weights flatten to `w_mat = w.reshape(C_out, -1)`, and the multiply is a batched matrix product. You can write it with `np.einsum`. In an einsum string, each letter names an axis, and a letter that appears on the left but not on the right is summed over. The string `"oc,ncl->nol"` takes `w_mat` with axes `(o, c)` and `col` with axes `(n, c, l)`, sums over `c`, and returns axes `(n, o, l)`:

```python
out = np.einsum("oc,ncl->nol", w_mat, col) + b[None, :, None]
out = out.reshape(N, C_out, H_out, W_out)
```

The bias has one entry per output channel, so `b[None, :, None]` stretches it across the batch and the positions. The final reshape turns `L` back into rows and columns.

> [!KEY]
> Flatten each kernel into a row of `w_mat`, flatten each window into a column of `col` in the same order (channel, kernel row, kernel column), and one matrix multiply computes every output value of the layer. The loop over 200,704 positions becomes a loop over 9 kernel offsets.

### A 1 by 1 convolution needs no columns

With $k = 1$, a window is a single pixel, so `col` is the image with its pixels laid out as columns, and it needs no extra memory. The whole layer is `w.reshape(C_out, C_in) @ x.reshape(N, C_in, H * W)`: a dense layer applied to the channels of every pixel.

## What the rearrangement costs

Every pixel lands in several columns, because every window that contains it copies it. With a 3 by 3 kernel at stride 1 and padding 1, each pixel is copied 9 times, so `col` holds about nine times as many numbers as the image. For the 64-channel, 56 by 56 layer, the image has 64 × 56 × 56 = 200,704 numbers and `col` has 64 × 9 × 56 × 56 = 1,806,336. A stride of $s$ divides the count of columns by $s^2$, because there are that many fewer windows. You spend memory to get speed.

The arithmetic is the same in either version: $C_{\text{out}} \cdot C_{\text{in}} k^2$ multiply-adds per output position, which for this layer is 64 × 576 × 3,136 = 115,605,504 per image. Figure 2 puts the two costs side by side. When memory gets tight, libraries build the columns in pieces, or use other algorithms.

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
Figure 2. The price of the rearrangement. The im2col array holds each pixel about k² times (here 9), so memory rises ninefold. The arithmetic does not change. What changes is who does it: a Python loop of 200,704 steps, or one matrix multiply that tuned library code runs at full speed.
```

## A shared weight collects blame from every position

Reuse the small example: the 3 by 3 image holding 1 to 9, a 2 by 2 kernel with rows (1, 2) and (3, 4), and stride 1. The output $Y$ is 2 by 2. Suppose the loss tells you that raising each of the four outputs by 1 raises the loss by 1, so the incoming gradient $dY$ is all ones.

Take the kernel weight in the top-left tap, $w_{00}$. It was used four times: window 1 multiplied it by pixel 1, window 2 by pixel 2, window 3 by pixel 4, and window 4 by pixel 5. Raise $w_{00}$ by a tiny amount $\epsilon$ and the four outputs rise by $\epsilon \cdot 1$, $\epsilon \cdot 2$, $\epsilon \cdot 4$, and $\epsilon \cdot 5$. The loss rises by $\epsilon$ times 1 + 2 + 4 + 5 = 12, so $dw_{00} = 12$. The other three taps give 16, 24, and 28, each a sum over the four windows of the pixel that tap met.

That is the general rule for a weight: its gradient is the sum, over every output position that used it, of the upstream gradient there times the pixel it met. With a batch, every image counts too, because the same weight is used on every image. The bias is the easiest case. It is added once to every output of its channel, so $db = 4$, the sum of $dY$ over the positions, and with several images the sum also runs over the batch.

## A pixel collects blame from every window that read it

Ask the same question for a pixel. Pixel 1, the top-left corner, sits in one window, at tap $(0, 0)$, so its gradient is $1 \times w_{00} = 1$. The middle pixel, 5, sits in all four windows and meets a different tap each time: tap $(1, 1)$ with weight 4, then $(1, 0)$ with weight 3, then $(0, 1)$ with weight 2, then $(0, 0)$ with weight 1. Each window pushes some blame onto it, so its gradient is $4 + 3 + 2 + 1 = 10$. The whole $dx$ for this example is

$$
dx = \begin{bmatrix} 1 & 3 & 2 \\ 4 & 10 & 6 \\ 3 & 7 & 4 \end{bmatrix}
$$

If every kernel weight were 1, $dx$ would equal the number of windows that cover each pixel: 1, 2, 1 on the first row, 2, 4, 2 on the second, and 1, 2, 1 on the third. With real weights, each count becomes a weighted sum.

### One rule in matrix form

The two sums are what the product rule gives for $Y_{\text{mat}} = W_{\text{mat}}\,\text{col} + b$. Let $dY_{\text{mat}}$ be $dY$ reshaped to `(N, C_out, L)`, where $L = H_{\text{out}} W_{\text{out}}$. Then

$$
db = \sum_{n,\,l} dY_{\text{mat}}, \qquad
dW_{\text{mat}} = \sum_{n} dY_{\text{mat}}\;\text{col}^\top, \qquad
d\,\text{col} = W_{\text{mat}}^\top\, dY_{\text{mat}}
$$

Read it aloud. The bias gradient adds $dY$ over images and positions. The weight gradient multiplies the upstream gradient at each position by the patch that was there, and adds over images. The gradient of the columns sends each output's blame back through the kernel that made it. In numpy, these are three lines:

```python
db = dY_mat.sum(axis=(0, 2))
dw_mat = np.einsum("nol,ncl->oc", dY_mat, col)
dcol = np.einsum("oc,nol->ncl", w_mat, dY_mat)
```

The weight gradient sums over `n` and `l` because those letters are missing from the output of the einsum string, which is the "shared weight" rule written as notation. Reshape `dw_mat` to the shape of `w`. Figure 3 shows how the three products connect.

```figure
<svg viewBox="0 0 600 352" role="img" aria-label="Flow of the backward pass: dY splits into db, dw, and dcol, and dcol returns to image shape through col2im">
<defs>
<marker id="c035-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<rect x="4" y="142" width="150" height="56" class="fill-soft-a stroke-a"/>
<text x="14" y="162" class="fig-bold">dY</text>
<text x="14" y="180" class="fig-small">(N, C_out, H_out, W_out)</text>
<rect x="4" y="222" width="150" height="40" class="fill-paper stroke-grid"/>
<text x="14" y="242" class="fig-bold">dY_mat</text>
<text x="14" y="260" class="fig-small">(N, C_out, L)</text>
<line x1="79" y1="198" x2="79" y2="222" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<text x="86" y="214" class="fig-small">reshape</text>
<rect x="190" y="30" width="200" height="56" class="fill-soft-c stroke-c"/>
<text x="200" y="50" class="fig-bold">db = sum of dY_mat</text>
<text x="200" y="68" class="fig-small">over n and l: shape (C_out,)</text>
<rect x="190" y="142" width="200" height="56" class="fill-soft-b stroke-b"/>
<text x="200" y="162" class="fig-bold">dw_mat = dY_mat · colᵀ</text>
<text x="200" y="180" class="fig-small">summed over n: (C_out, C_in·k·k)</text>
<rect x="190" y="254" width="200" height="56" class="fill-soft-d stroke-d"/>
<text x="200" y="274" class="fig-bold">dcol = W_matᵀ · dY_mat</text>
<text x="200" y="292" class="fig-small">shape (N, C_in·k·k, L)</text>
<line x1="154" y1="242" x2="188" y2="58" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<line x1="154" y1="242" x2="188" y2="170" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<line x1="154" y1="242" x2="188" y2="282" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<rect x="436" y="30" width="160" height="56" class="fill-paper stroke-grid"/>
<text x="446" y="50" class="fig-bold">db</text>
<text x="446" y="68" class="fig-small">(C_out,)</text>
<rect x="436" y="142" width="160" height="56" class="fill-paper stroke-grid"/>
<text x="446" y="162" class="fig-bold">dw</text>
<text x="446" y="180" class="fig-small">reshape to (C_out, C_in, k, k)</text>
<rect x="436" y="254" width="160" height="56" class="fill-paper stroke-grid"/>
<text x="446" y="274" class="fig-bold">dx</text>
<text x="446" y="292" class="fig-small">(N, C_in, H, W)</text>
<line x1="390" y1="58" x2="434" y2="58" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<line x1="390" y1="170" x2="434" y2="170" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<line x1="390" y1="282" x2="434" y2="282" class="stroke-ink" marker-end="url(#c035-flow-arrow)"/>
<text x="392" y="276" class="fig-small fig-bold">col2im</text>
<text x="436" y="328" class="fig-small">col2im adds with +=</text>
<text x="190" y="216" class="fig-small">needs col, saved from the forward pass</text>
<text x="190" y="328" class="fig-small">needs the flattened weights W_mat</text>
</svg>
Figure 3. The backward pass of a convolution. The incoming gradient dY (reshaped to dY_mat) feeds three products. Two of them (db and dw) add over the batch and over positions, because weights and biases are shared. The third (dcol) lives in the column layout and returns to image shape through col2im, which adds overlapping entries.
```

### col2im adds the copies back together

The third line leaves one job. $d\,\text{col}$ is the gradient for every copy of every pixel, laid out like `col`, and you need the gradient for each pixel itself. During the forward pass, `im2col` copied each pixel into every window that contains it. Copying forward means adding backward: a pixel that was copied into four columns gets the sum of the four gradients. That step is **col2im**. It takes each kernel offset's slice of $d\,\text{col}$ and adds it into the matching strided slice of a zero image:

```python
dx_padded[:, :, dh:dh + stride * h_out:stride, dw:dw + stride * w_out:stride] += dcol_r[:, :, dh, dw]
```

Figure 4 shows the result on the example. With $dY$ all ones, each window sends the kernel's own numbers back, and adding the four windows into place produces the $dx$ above.

```figure
<svg viewBox="0 0 600 384" role="img" aria-label="Four 2 by 2 patches added into a 3 by 3 canvas with plus-equals, beside the wrong result of overwriting">
<text x="20" y="22" class="fig-bold">With dY all ones, each window sends the kernel's own numbers back</text>
<rect x="30" y="42" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="45" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="60" y="42" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="75" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="30" y="72" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="45" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="60" y="72" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="75" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="60" y="118" text-anchor="middle" class="fig-small">window 1</text>
<text x="60" y="132" text-anchor="middle" class="fig-small">corner (0, 0)</text>
<rect x="160" y="42" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="175" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="190" y="42" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="205" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="160" y="72" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="175" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="190" y="72" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="205" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="190" y="118" text-anchor="middle" class="fig-small">window 2</text>
<text x="190" y="132" text-anchor="middle" class="fig-small">corner (0, 1)</text>
<rect x="290" y="42" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="305" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="320" y="42" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="335" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="290" y="72" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="305" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="320" y="72" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="335" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="320" y="118" text-anchor="middle" class="fig-small">window 3</text>
<text x="320" y="132" text-anchor="middle" class="fig-small">corner (1, 0)</text>
<rect x="420" y="42" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="435" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="450" y="42" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="465" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="420" y="72" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="435" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="450" y="72" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="465" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="450" y="118" text-anchor="middle" class="fig-small">window 4</text>
<text x="450" y="132" text-anchor="middle" class="fig-small">corner (1, 1)</text>
<text x="30" y="178" class="fig-bold">Add them into place (+=)</text>
<rect x="30" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="56" y="224.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="82" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="108" y="224.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="134" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="160" y="224.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="30" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="56" y="276.5" text-anchor="middle" class="fig-bold">4</text>
<rect x="82" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="108" y="276.5" text-anchor="middle" class="fig-bold">10</text>
<rect x="134" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="160" y="276.5" text-anchor="middle" class="fig-bold">6</text>
<rect x="30" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="56" y="328.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="82" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="108" y="328.5" text-anchor="middle" class="fig-bold">7</text>
<rect x="134" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="160" y="328.5" text-anchor="middle" class="fig-bold">4</text>
<text x="30" y="368" class="fig-small">this is dx</text>
<text x="330" y="178" class="fig-bold">Overwrite instead (=): last window wins</text>
<rect x="330" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="356" y="224.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="382" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="408" y="224.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="434" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="460" y="224.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="330" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="356" y="276.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="382" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="408" y="276.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="434" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="460" y="276.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="330" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="356" y="328.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="382" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="408" y="328.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="434" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="460" y="328.5" text-anchor="middle" class="fig-bold">4</text>
<rect x="382" y="194" width="52" height="52" class="stroke-bad"/>
<rect x="330" y="246" width="52" height="52" class="stroke-bad"/>
<rect x="382" y="246" width="52" height="52" class="stroke-bad"/>
<rect x="434" y="246" width="52" height="52" class="stroke-bad"/>
<rect x="382" y="298" width="52" height="52" class="stroke-bad"/>
<text x="330" y="368" class="fig-small">red cells are wrong: overlaps lost</text>
</svg>
Figure 4. col2im on the chapter's example. Adding the four patches gives dx = (1, 3, 2 / 4, 10, 6 / 3, 7, 4). Overwriting keeps only the last patch that touched each pixel, so the five pixels read by two or more windows are wrong (red). Notice that the four corner pixels, each read by one window, survive either way.
```

The `+=` is essential. Writing `=` keeps only the last window that touched each pixel, and the red cells in Figure 4 are wrong for exactly that reason. After the loop, cut the padding off the canvas. When the padding is 0, do not write `canvas[:, :, 0:-0]`, because `0:-0` is an empty slice. Return the whole canvas instead.

`col2im` is the **adjoint**, or transpose, of `im2col`: `im2col` is a linear map from images to columns, and `col2im` is the matrix transpose of that map. The defining test moves the map across a dot product:

$$
\langle \text{im2col}(x),\, c\rangle = \langle x,\, \text{col2im}(c)\rangle
$$

Read it aloud: take any image $x$ and any array $c$ shaped like `col`. Copy the image into columns and take the dot product with $c$, and you get the same number as scattering $c$ back and taking the dot product with $x$. The exercise tests this identity with random $x$ and $c$, which catches a wrong slice, a wrong stride, or a mishandled padding. It is not an inverse: `col2im(im2col(x))` multiplies each pixel by its coverage count instead of returning $x$. That count, the number of windows that read a pixel, is `col2im` applied to ones.

> [!KEY]
> A shared weight and bias collect gradient from every position and image, so $dW$ and $db$ are sums. A pixel read by several windows collects gradient from each, so $dx$ is a scatter-add of $d\,\text{col}$. Both sums come from one fact: the forward pass reused the same number in many places.

### Trust the gradient only after you check it

A hand derivation can be wrong in a way that still looks right: a missed sum, a transposed axis. The check from [whole-network gradients](/lessons/036-softmax-and-whole-network-gradients) works here. Pick a scalar loss, $L = \sum dY \odot Y$, which has $dL/dY = dY$ by construction. Nudge each entry of $x$, $w$, or $b$ by $\pm\epsilon$, rerun the forward pass, and compare $(L_+ - L_-)/2\epsilon$ with your analytic gradient. That needs one forward pass per entry, which is far too slow for training and fast enough for a 5 by 5 image. The exercise does this for all three gradients, with a stride of 2 and padding of 1 so that overlaps, skipped pixels, and the padding cut all get tested.

> [!RECAP]
> Hold these points before you build:
>
> - `col` has shape `(N, C * k * k, H_out * W_out)`, and patch and kernel use the same flatten order: channel, kernel row, kernel column.
> - One strided slice per kernel offset fills `col`, and the forward pass is `w.reshape(C_out, -1)` times `col`, plus the bias. `col` holds about $k^2$ times as many numbers as the image.
> - $db$ sums $dY$ over the batch and positions, $dW_{\text{mat}}$ is $dY_{\text{mat}}\,\text{col}^\top$ summed over the batch, and $d\,\text{col} = W_{\text{mat}}^\top\, dY_{\text{mat}}$.
> - `col2im` adds each entry back to the pixel it was copied from with `+=`, and it is the transpose of `im2col`.
> - The number of windows covering a pixel is `col2im` applied to ones.

## Build it step by step

The checks call the functions in this order. The starter already contains the loop version `conv2d_naive` and `conv_output_shape`, solved, from the earlier chapter.

1. **`im2col(x, kernel_size, stride, padding)`.** `x` has shape `(N, C, H, W)`. Return `(N, C * k * k, H_out * W_out)`. Pad with `np.pad`, fill an `(N, C, k, k, H_out, W_out)` array with one strided slice per kernel offset, and reshape. The most common mistake is the slice end, which must be `dh + stride * h_out`.
2. **`conv2d_forward(x, w, b, stride, padding)`.** Return `(N, C_out, H_out, W_out)`. Build `col`, multiply by `w.reshape(C_out, -1)`, add `b[None, :, None]`, and reshape. The most common mistake is adding the bias along the wrong axis.
3. **`conv1x1_as_matmul(x, w, b)`.** `w` has shape `(C_out, C_in, 1, 1)`. Reshape `x` to `(N, C_in, H * W)`, multiply by `w.reshape(C_out, C_in)`, add the bias, and reshape back. Do not call `im2col`. The most common mistake is flattening the image as `(N, H * W, C_in)`, which pairs pixels with channels in the wrong order.
4. **`col2im(dcol, x_shape, kernel_size, stride, padding)`.** `dcol` has the shape that `im2col` returns, and `x_shape` is `(N, C, H, W)`. Reshape `dcol` to `(N, C, k, k, H_out, W_out)`, add each offset's slice into a zero canvas of the padded size with `+=`, then cut off the padding. The most common mistake is `=` instead of `+=`, or `padding:-padding` when the padding is 0.
5. **`window_coverage(x_shape, kernel_size, stride, padding)`.** Return an `(H, W)` array with the number of windows that read each pixel. Apply `col2im` to an array of ones with the shape `im2col` would give for a single one-channel image. The most common mistake is returning the padded size, because the cut-off in `col2im` already removes the padding.
6. **`conv2d_backward(dY, x, w, b, stride, padding)`.** Return `(dx, dw, db)` with the shapes of `x`, `w`, and `b`. Rebuild `col` with `im2col`, form the three gradients above, and call `col2im` for `dx`. The most common mistake is to skip the sum over the batch in `dw`.

## Where this breaks

- A stride larger than the kernel leaves some pixels in no window. Their coverage is 0, so their gradient is 0: the network cannot learn from pixels it never reads. Try `window_coverage((1, 1, 2, 6), 2, 3, 0)`.
- Padding cells take part in the forward pass, but they are not pixels. Their gradient is computed and then thrown away by the cut-off.
- `dcol` is as large as `col`, about $k^2$ times the image. The backward pass needs that memory at the same time as the forward pass's saved `col`, which is why large networks recompute it or build it in pieces.

## The exercise

You implement `im2col`, `conv2d_forward`, `conv1x1_as_matmul`, `col2im`, `window_coverage`, and `conv2d_backward` in `starter.py`. The checks compare `im2col` with windows cut out by hand, including the channel-row-column order, stride, padding, and a non-square image. They compare `conv2d_forward` with an independent reference built from numpy's sliding windows, and with the loop version, and confirm that `conv1x1_as_matmul` equals a convolution with a 1 by 1 kernel. They confirm that `col2im` adds overlaps and satisfies the adjoint identity, that `window_coverage` returns 1, 2, 1 / 2, 4, 2 / 1, 2, 1 on the example and 0 for pixels no window reads, and that `conv2d_backward` matches finite differences for `dx`, `dw`, and `db`. A last check reproduces the hand example.
