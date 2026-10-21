---
title: Convolutional networks
slug: 20-convolutions
part: "Part 4 — Vision and sequences"
summary: "Build a 2-D convolution as one matrix multiply, with backward, pooling, output sizes, and receptive fields."
minutes: 70
gpu: false
objectives:
  - Explain how a convolution shares weights across positions, and why a shifted input gives a shifted output.
  - Compute the output size and receptive field of a stack of convolutional layers.
  - Build the forward and backward pass of a convolution with im2col and matrix multiplies.
  - Compare the number of weights in a convolutional layer, a dense layer, and a depthwise separable layer.
prereqs:
  - 16-backpropagation
exercise: 20-convolutions
---

# Convolutional networks

A colour photo, 224 pixels on a side with three colour channels, holds 150,528 numbers. Feed that to a dense layer with 1,000 units and the first layer alone needs about 150 million weights. A digit in the top-left corner is still that digit in the bottom-right, and a dense layer has to relearn it.

You slide one small filter across the photo and reuse a few thousand weights. You build that slide as a matrix multiply, forward and backward, and you compute the output size and how much of the photo each unit can see.

## What you already know

You already have the following:

- A dense layer multiplies its input by a weight matrix and adds a bias.
- Backpropagation carries the gradient of the loss back through each layer.
- The product $AB$ sums along the shared dimension.
- For $Y = WX$, the gradients are $dW = dY\, X^\top$ and $dX = W^\top dY$.
- A slice `a[start:stop:step]` keeps every `step`-th entry.

## Slide one small stamp across the photo

A stamp for the digit 7, pressed on every patch, fills the output grid.

Bias 0. The stamp fits in two positions across and two down:

$$
x = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}, \qquad
w = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

The top-left score is $1\cdot 1 + 0\cdot 2 + 0\cdot 4 + (-1)\cdot 5 = -4$.

That stamp-and-sum is a **convolution**. The kernel is not flipped. For one channel and a $k \times k$ kernel, the score at $(i, j)$ is

$$
y_{i,j} = b + \sum_{a=0}^{k-1}\sum_{c=0}^{k-1} w_{a,c}\; x_{i+a,\,j+c}
$$

Place the kernel's top-left on pixel $(i, j)$, multiply, add the bias $b$. The other three scores are $2 + (-6)$, $4 + (-8)$, and $5 + (-9)$, each $-4$:

```python
y[i, j] = b + np.sum(w * x[i:i + k, j:j + k])
```

The usual mistake is to flip the kernel before the multiply.

One kernel has shape `(C_in, k, k)`. A layer of `C_out` kernels has `w` of shape `(C_out, C_in, k, k)` and bias `(C_out,)`. The input is `(N, C_in, H, W)`.

Move the input one pixel right and the output moves one pixel right. That is translation equivariance.

> [!KEY]
> One small set of weights slides over every position. Shift the input and the output shifts by the same amount.

## Count how many times the stamp fits

Kernel size $k$, stride $s$, padding $p$, and dilation $d$ set the length. At $d = 2$ a width-3 kernel covers 5 pixels.

On a length of 7, with $k = 3$, stride 1, no padding, and dilation 1, the kernel covers 3. Travel is $7 - 3 = 4$, plus the start, so the output length is 5.

The length along one axis is

$$
\text{out} = \left\lfloor \frac{\text{size} + 2p - \big(d\,(k - 1) + 1\big)}{s} \right\rfloor + 1
$$

Subtract the reach from the padded length, divide by the stride, round down, and add 1.

These four cases follow that rule:

| size | $k$ | $s$ | $p$ | $d$ | Output |
|---|---|---|---|---|---|
| 7 | 3 | 1 | 0 | 1 | 5 |
| 7 | 3 | 1 | 1 | 1 | 7 |
| 7 | 3 | 2 | 1 | 1 | 4 |
| 7 | 3 | 1 | 0 | 2 | 3 |

Integer division gives those lengths:

```python
reach = dilation * (kernel_size - 1) + 1
out = (size + 2 * padding - reach) // stride + 1
```

The usual mistake is to use `kernel_size` as the reach, which drops the dilation. At $d = 2$ the reach is 5, not 3.

## See how far back one unit looks

One 3 by 3 layer covers 3 input pixels. Stack a second 3 by 3 and each of its units covers 3 units of the first layer, which together cover 5 pixels.

After a stride of 2, neighbours are 2 input pixels apart, so each extra tap adds 2 pixels. Keep a jump, the product of earlier strides. Start from $r = 1$ and jump $1$:

$$
r_{\text{new}} = r + (k - 1) \cdot \text{jump}, \qquad \text{jump}_{\text{new}} = \text{jump} \cdot s
$$

Add $(k-1)$ times the current jump, then multiply the jump by this layer's stride. That stride affects only later layers.

Three layers, $(k, s) = (3, 2)$, then $(3, 2)$, then $(3, 1)$, grow as follows:

| Layer | $k$ | $s$ | Jump used | Field after | Jump after |
|---|---|---|---|---|---|
| start | | | | 1 | 1 |
| 1 | 3 | 2 | 1 | $1 + 2 \times 1 = 3$ | 2 |
| 2 | 3 | 2 | 2 | $3 + 2 \times 2 = 7$ | 4 |
| 3 | 3 | 1 | 4 | $7 + 2 \times 4 = 15$ | 4 |

Update the field, then the jump:

```python
r = r + (kernel_size - 1) * jump
jump = jump * stride
```

The usual mistake is to update the jump first, which gives 21 on this stack rather than 15.

> [!KEY]
> Each layer widens the field by $k - 1$ times the product of earlier strides. Early strides make it grow fastest.

## Cut every patch into a column

**im2col** (image to columns) flattens every patch into a column. The convolution becomes one matrix multiply.

For the 3 by 3 picture and the 2 by 2 kernel there are 4 patches, and each patch has 4 numbers. The top-left patch, flattened in row-then-column order, is $[1, 2, 4, 5]$. Dot that with the flattened kernel $[1, 0, 0, -1]$ and you get $1 - 5 = -4$, the same score as before.

In a batch, the column array has shape `(N, C * k * k, H_out * W_out)`. Column $l$ is the patch at output position $(l // W_{\text{out}},\, l \bmod W_{\text{out}})$, flattened as channel, then kernel row, then kernel column. Flatten the weights to $W_{\text{mat}}$ of shape `(C_out, C_in * k * k)`, one row per output channel:

$$
Y_{\text{mat}} = W_{\text{mat}} \, \text{col} + b
$$

Each row of $W_{\text{mat}}$ is one kernel. Each column of `col` is one patch. Add the bias per output channel, then reshape to `(C_out, H_out, W_out)`:

```python
w_mat = w.reshape(C_out, -1)
out = np.einsum("oc,ncl->nol", w_mat, col) + b[None, :, None]
```

The usual mistake is a flatten order other than channel, then kernel row, then kernel column.

Loop over the $k^2$ offsets. One strided slice gives that offset at every output position:

```python
x_padded[:, :, dh:dh + stride * H_out:stride, dw:dw + stride * W_out:stride]
```

Park the slices in `(N, C, k, k, H_out, W_out)`, then reshape. `np.pad` adds the zero border.

> [!KEY]
> im2col turns a convolution into one matrix multiply. Each row of $W_{\text{mat}}$ is a kernel, and each column of `col` is a patch.

> [!TIP]
> In an einsum string, the letters on the left that are missing on the right are the summed axes.

## Scatter the gradient back onto the photo

`dY` has the output shape. Reshape it to `(N, C_out, L)`. The forward pass was a product $Y = W\,\mathrm{col}$, so the gradients reuse the product rules.

`db` sums `dY` over the batch and over space. `dw` is `dY` times `col` transposed, summed over the batch, then reshaped to `(C_out, C_in, k, k)`. `dcol` is $W_{\text{mat}}^\top$ times `dY`, shape `(N, C_in k^2, L)`:

```python
db = dY_mat.sum(axis=(0, 2))
dw_mat = np.einsum("nol,ncl->oc", dY_mat, col)
dcol = np.einsum("oc,nol->ncl", w_mat, dY_mat)
```

`col2im` adds each slice of `dcol` back into the padded image with `+=`. Patches overlap, so one pixel's gradient is a sum. Then drop the padding. If padding is 0, do not write `padding:-padding`, because `0:-0` is empty.

The usual mistake is `=` instead of `+=`, or forgetting to sum `dw` over the batch.

Max pooling keeps the largest value in each window, per channel. The output length uses `conv_output_shape` at padding 0 and dilation 1. A small shift inside a window often leaves that max unchanged, until the max crosses into the next window.

> [!RECAP]
> Hold these five points before you write code:
>
> - Output length is $\lfloor(\mathrm{size}+2p-(d(k-1)+1))/s\rfloor+1$.
> - `im2col` returns `(N, C*k*k, H_out*W_out)`, one strided slice per kernel offset.
> - The forward pass multiplies `w.reshape(C_out, -1)` by those columns and adds `b[None, :, None]`.
> - Backward reuses the product rules. `col2im` adds overlapping patches with `+=`.
> - The receptive field starts at 1 and grows by `(k - 1) * jump` before the jump changes.

## Build it step by step

The checks call the functions in this order:

1. **`conv_output_shape(size, kernel_size, stride, padding, dilation)`.** Return an int. The reach is `dilation * (kernel_size - 1) + 1`. The usual mistake is to ignore dilation.
2. **`im2col(x, kernel_size, stride, padding)`.** `x` is `(N, C, H, W)`. Return `(N, C * k * k, H_out * W_out)`. Pad, fill one strided slice per kernel offset, and reshape. The usual mistake is a slice end off by one.
3. **`col2im(dcol, x_shape, kernel_size, stride, padding)`.** Scatter-add with `+=` into shape `x_shape`. The checks call it through `conv2d_backward`. The usual mistake is `padding:-padding` when padding is 0.
4. **`conv2d_forward(x, w, b, stride, padding)`.** Return `(N, C_out, H_out, W_out)`. Multiply `w.reshape(C_out, -1)` by the columns, add `b[None, :, None]`, and reshape. The usual mistake is adding `b` on the wrong axis.
5. **`conv2d_backward(dY, x, w, b, stride, padding)`.** Return `(dx, dw, db)` in the shapes of `x`, `w`, and `b`. Rebuild the columns, form the three gradients, and call `col2im` for `dx`. The usual mistake is to skip the sum of `dw` over the batch.
6. **`max_pool_forward(x, pool_size, stride)`.** Return `(N, C, H_out, W_out)`. Take `window.max(axis=(2, 3))` at each output position. The usual mistake is a max that mixes channels.
7. **`receptive_field(layers)`.** `layers` is a list of `(kernel_size, stride)` pairs. Start from 1 and grow the field before you update the jump. The usual mistake is the reverse order, which gives 21 instead of 15.

## Cost and scaling

A convolution has $C_{\text{out}} \cdot C_{\text{in}} \cdot k^2 + C_{\text{out}}$ weights. That count does not depend on the image size. The work does. Each output pixel costs $C_{\text{in}} k^2$ multiplies for every output channel.

A 3 by 3 layer, 64 channels in and out, on a 56 by 56 image, costs the following:

| Layer | Weights | Multiplies per image |
|---|---|---|
| Dense, every pixel to every pixel | about $4 \times 10^{10}$ | about $4 \times 10^{10}$ |
| Convolution, $k = 3$ | $64 \cdot 64 \cdot 9 = 36{,}864$ | about $1.2 \times 10^{8}$ |
| Depthwise separable, $k = 3$ | $64 \cdot 9 + 64 \cdot 64 = 4{,}672$ | about $1.5 \times 10^{7}$ |

A depthwise separable layer runs one $k \times k$ filter per channel, then a 1 by 1 mix. Here that is about eight times fewer weights. im2col stores each pixel about $k^2$ times, about nine times the input for a 3 by 3 kernel.

## Where this breaks

Each of these failures has a direct change:

- No padding shrinks the image, and edge pixels appear in fewer windows. Padding $(k-1)/2$ keeps the size for odd $k$ and stride 1.
- Shifts are handled. Rotations and scale changes are not. Rotated and resized training copies, data augmentation, cover some of that.
- A stride skips pixels, so a small shift can change the output sharply. Some layers blur before they downsample.
- A field smaller than the object has no path from the whole object to the output. Add layers, stride, or dilation.
- im2col can run out of memory. Other algorithms, or columns built in pieces, avoid storing every patch at once.

## The exercise

You implement `conv_output_shape`, `im2col`, `col2im`, `conv2d_forward`, `conv2d_backward`, `max_pool_forward`, and `receptive_field`. The forward and backward passes go through `im2col` and matrix multiplies, not a Python loop over every output pixel.

The checks match the output size, including dilation, and match `im2col` to direct slices. They match the forward pass to a slow loop, the three gradients to finite differences, and a shifted input to a shifted output. They match max pooling to a windowed maximum, and `receptive_field` to 15.
