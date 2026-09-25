---
title: Convolutional networks
slug: 20-convolutions
part: "Part 4 — Vision and sequences"
summary: Build a 2-D convolution as one matrix multiply, with its backward pass, pooling, output sizes, and receptive fields.
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

A colour photo of 224 by 224 pixels holds $224 \times 224 \times 3 = 150{,}528$
numbers. Feed that to a multilayer perceptron with 1000 hidden units, and the
first layer alone needs about 150 million weights. That is slow, it needs a
huge amount of data, and it ignores a basic fact about images: a cat in the
top-left corner is still a cat in the bottom-right.

A convolutional network builds that fact in. It uses a small filter, say 3 by
3 pixels, and slides it across the whole image. The same few weights look for
the same pattern everywhere. A first layer might have a few thousand weights
instead of 150 million.

In this lesson you build a convolution layer from scratch, forward and
backward. You turn it into one big matrix multiply with a trick called
im2col, which is how fast libraries do it. You also compute output sizes and
receptive fields, which tell you how much of the image each unit can see.

## What you already know

- A dense layer multiplies its input by a weight matrix and adds a bias.
- Backpropagation passes the gradient of the loss back through each layer.
- A matrix multiply $A B$ sums products along the shared dimension.
- The gradient of $Y = W X$ is $dW = dY \, X^\top$ and $dX = W^\top dY$.
- numpy slicing like `a[start:stop:step]` picks every `step`-th element.

## What a convolution does

Think of a magnifying glass sliding over a page, looking for the letter "e".
At every position, you ask the same question: how much does this patch look
like an "e"? You write the answer down in a grid. That grid is the output.

A convolution does this with numbers. The filter, also called the kernel, is a
small grid of weights. At each position, you multiply the filter by the patch
of image under it, entry by entry, and add everything up. Then you add a bias.

For one input channel and a $k \times k$ kernel $w$, the output at row $i$ and
column $j$ is

$$
y_{i,j} = b + \sum_{a=0}^{k-1}\sum_{c=0}^{k-1} w_{a,c}\; x_{i+a,\,j+c}
$$

Read it aloud: to get output pixel $(i, j)$, place the kernel's top-left
corner on input pixel $(i, j)$. Multiply each kernel weight $w_{a,c}$ by the
input pixel under it, add all $k \times k$ products, and add the bias $b$.

```python
y[i, j] = b + np.sum(w * x[i:i+k, j:j+k])
```

Deep learning libraries call this a convolution, although mathematicians would
call it a cross-correlation, because the kernel is not flipped. The exercise
follows the libraries.

Real images have several channels, such as red, green, and blue. A kernel then
has shape `(C_in, k, k)`: one $k \times k$ slice per input channel, all summed
together. A layer has `C_out` such kernels, so its weights have shape
`(C_out, C_in, k, k)` and its bias has shape `(C_out,)`. The input has shape
`(N, C_in, H, W)`: $N$ images, $C_{\text{in}}$ channels, height $H$, and width
$W$.

### Worked example: a 2-by-2 kernel

Take this 3-by-3 input and 2-by-2 kernel, with bias 0:

$$
x = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}, \qquad
w = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

The kernel fits in 2 positions across and 2 down, so the output is 2 by 2:

- $y_{0,0} = 1 \cdot 1 + 0 \cdot 2 + 0 \cdot 4 + (-1) \cdot 5 = -4$
- $y_{0,1} = 1 \cdot 2 + 0 \cdot 3 + 0 \cdot 5 + (-1) \cdot 6 = -4$
- $y_{1,0} = 1 \cdot 4 + 0 \cdot 5 + 0 \cdot 7 + (-1) \cdot 8 = -4$
- $y_{1,1} = 1 \cdot 5 + 0 \cdot 6 + 0 \cdot 8 + (-1) \cdot 9 = -4$

This kernel measures the difference between a pixel and its lower-right
neighbour. The input rises evenly, so every answer is the same.

### Shift in, shift out

Because the same weights are used at every position, moving the input one
pixel to the right moves the output one pixel to the right. Nothing else
changes. This property is called translation equivariance. It is why a
convolution finds a cat wherever the cat is.

Pooling takes this a step further. Max pooling slides a window over each
channel and keeps only the largest value in each window. Small shifts inside a
window then often leave the output unchanged. That is only approximate: a
shift that moves the largest value into the next window does change the
output.

## Output size

Four settings decide how big the output is:

- **Kernel size $k$**: how many pixels wide the filter is.
- **Stride $s$**: how many pixels the filter moves between positions.
- **Padding $p$**: how many rows and columns of zeros you add on each side.
- **Dilation $d$**: the spacing between kernel taps. With $d = 2$, a 3-wide
  kernel skips every other pixel and covers 5 pixels.

The output length along one axis is

$$
\text{out} = \left\lfloor \frac{\text{size} + 2p - \big(d\,(k - 1) + 1\big)}{s} \right\rfloor + 1
$$

Read it aloud: the padded input has length $\text{size} + 2p$. The kernel
covers $d(k-1) + 1$ pixels. Subtract to get how far the kernel can travel,
divide by the stride and round down (the $\lfloor \cdot \rfloor$ brackets), then
add 1 for the starting position.

```python
effective_kernel = dilation * (kernel_size - 1) + 1
out = (size + 2 * padding - effective_kernel) // stride + 1
```

Check it on some cases:

| size | $k$ | $s$ | $p$ | $d$ | Calculation | Output |
|---|---|---|---|---|---|---|
| 7 | 3 | 1 | 0 | 1 | $(7 - 3) // 1 + 1$ | 5 |
| 7 | 3 | 1 | 1 | 1 | $(9 - 3) // 1 + 1$ | 7 |
| 7 | 3 | 2 | 1 | 1 | $(9 - 3) // 2 + 1$ | 4 |
| 7 | 3 | 1 | 0 | 2 | $(7 - 5) // 1 + 1$ | 3 |

## Receptive field

The receptive field of a unit is the patch of the original image it can see.
One 3-by-3 layer sees 3 pixels across. Stack a second 3-by-3 layer on top, and
each of its units sees 3 units of the first layer, which together cover 5
pixels. Deeper units see more of the image.

Stride makes the field grow faster. After a layer with stride 2, neighbouring
units are 2 pixels apart in the input, so each extra kernel tap adds 2 pixels
of view. Keep a running "jump", the product of every earlier stride:

$$
r_{\text{new}} = r + (k - 1) \cdot \text{jump}, \qquad \text{jump}_{\text{new}} = \text{jump} \cdot s
$$

Read it aloud: start with $r = 1$ and $\text{jump} = 1$. For each layer in
order, grow the field by the kernel size minus 1, times the current jump. Then
multiply the jump by this layer's stride, ready for the next layer.

```python
r = r + (kernel_size - 1) * jump
jump *= stride
```

### Worked example: receptive field

Take three layers with $(k, s)$ equal to $(3, 2)$, $(3, 2)$, and $(3, 1)$:

| Layer | $k$ | $s$ | jump used | $r$ after | jump after |
|---|---|---|---|---|---|
| start | | | | 1 | 1 |
| 1 | 3 | 2 | 1 | $1 + 2 \times 1 = 3$ | 2 |
| 2 | 3 | 2 | 2 | $3 + 2 \times 2 = 7$ | 4 |
| 3 | 3 | 1 | 4 | $7 + 2 \times 4 = 15$ | 4 |

Each unit at the top sees a 15-by-15 patch of the input.

## Convolution as one matrix multiply

Looping over every output pixel in Python is slow. The im2col trick ("image to
columns") rearranges the input so that the whole convolution becomes one
matrix multiply, which numpy runs fast.

The idea: cut out every patch the kernel will visit, flatten each one into a
column, and stack the columns side by side. For one image with $C_{\text{in}}$
channels, each patch has $C_{\text{in}} k^2$ numbers. There are
$L = H_{\text{out}} \times W_{\text{out}}$ patches. So the column matrix has
shape $(C_{\text{in}} k^2, L)$.

Flatten the weights the same way, one row per output channel, into
$W_{\text{mat}}$ of shape $(C_{\text{out}}, C_{\text{in}} k^2)$. Now

$$
Y_{\text{mat}} = W_{\text{mat}} \, \text{col} + b
$$

Read it aloud: each row of $W_{\text{mat}}$ is one kernel, and each column of
$\text{col}$ is one patch. Their product is every kernel applied to every
patch at once. Add each output channel's bias to its whole row. The result
has shape $(C_{\text{out}}, L)$, which you reshape to
$(C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$.

```python
w_mat = w.reshape(C_out, -1)
out = np.einsum("oc,ncl->nol", w_mat, col) + b[None, :, None]
```

Here `np.einsum` does the matrix multiply for every image `n` in the batch.
You can also loop over the batch and use `@`.

### Building the columns

You don't need to loop over every output position. Loop only over the $k^2$
kernel offsets $(d_h, d_w)$. For each offset, one strided slice of the padded
input gives the value at that offset for every output position at once:

```python
x_padded[:, :, dh : dh + stride * H_out : stride, dw : dw + stride * W_out : stride]
```

Store each slice in an array of shape `(N, C, k, k, H_out, W_out)`, then
reshape to `(N, C * k * k, H_out * W_out)`. That order, channel then kernel row
then kernel column, matches `w.reshape(C_out, -1)`. Use `np.pad` to add the
zero border.

## The backward pass

You get the incoming gradient `dY` with the output's shape, and you need the
gradients for the bias, the weights, and the input. Because the forward pass is
a matrix multiply, you reuse the rules for $Y = W X$. Reshape `dY` to
$dY_{\text{mat}}$ of shape $(N, C_{\text{out}}, L)$.

- **Bias.** Each bias is added to every position of every image, so its
  gradient is the sum of `dY` over the batch and over space:
  `dY_mat.sum(axis=(0, 2))`.
- **Weights.** $dW_{\text{mat}} = dY_{\text{mat}} \, \text{col}^\top$, summed
  over the batch. Reshape it back to `(C_out, C_in, k, k)`.
- **Columns.** $d\text{col} = W_{\text{mat}}^\top \, dY_{\text{mat}}$, with shape
  $(N, C_{\text{in}} k^2, L)$.

```python
dw_mat = np.einsum("nol,ncl->oc", dY_mat, col)
dcol = np.einsum("oc,nol->ncl", w_mat, dY_mat)
```

The last step turns $d\text{col}$ back into an image-shaped gradient, with
`col2im`. It runs im2col in reverse. For each kernel offset, add the slice of
$d\text{col}$ back into the same strided positions of a zero array the size of
the padded input. Use `+=`, not `=`, because patches overlap and one input
pixel feeds several patches. Its gradient is the sum of all of them. Finally,
cut off the padding.

## Build it step by step

The checks test the functions in this order.

1. **`conv_output_shape(size, kernel_size, stride, padding, dilation)`.**
   Returns an integer. One line with `//`. The common mistake is forgetting the
   dilation, so the kernel's reach is `kernel_size` instead of
   `dilation * (kernel_size - 1) + 1`.
2. **`im2col(x, kernel_size, stride, padding)`.** Takes `(N, C, H, W)`. Returns
   `(N, C * k * k, H_out * W_out)`. Pad with `np.pad`, fill a six-axis array
   with one strided slice per kernel offset, then reshape. The common mistake
   is a slice end that is off by one, which gives the wrong number of output
   positions.
3. **`col2im(dcol, x_shape, kernel_size, stride, padding)`.** The reverse of
   `im2col`, with `+=`. It isn't tested on its own, but `conv2d_backward` needs
   it. When `padding` is 0, don't slice with `padding:-padding`, because `0:-0`
   is empty.
4. **`conv2d_forward(x, w, b, stride, padding)`.** Returns
   `(N, C_out, H_out, W_out)`. Call `im2col`, multiply by `w.reshape(C_out, -1)`,
   add the bias, and reshape. The common mistake is adding `b` without
   `b[None, :, None]`, so it lines up with the wrong axis.
5. **`conv2d_backward(dY, x, w, b, stride, padding)`.** Returns `(dx, dw, db)`
   with the same shapes as `x`, `w`, and `b`. Rebuild `col`, compute the three
   gradients, and call `col2im` for `dx`. The common mistake is forgetting to
   sum `dw` over the batch.
6. **`max_pool_forward(x, pool_size, stride)`.** Returns
   `(N, C, H_out, W_out)`, with sizes from `conv_output_shape` at padding 0.
   Loop over output positions and take `window.max(axis=(2, 3))`.
7. **`receptive_field(layers)`.** Takes a list of `(kernel_size, stride)`
   pairs. Returns an integer. The common mistake is updating the jump before
   using it, which gives 21 instead of 15 in the worked example.

## Cost and scaling

The number of weights in a convolution layer is $C_{\text{out}} \cdot
C_{\text{in}} \cdot k^2 + C_{\text{out}}$. It doesn't depend on the image
size at all. The work does: each output pixel costs $C_{\text{in}} k^2$
multiplies per output channel.

Here is the cost for a 3-by-3 layer with 64 channels in and out, on a
56-by-56 image:

| Layer | Weights | Multiplies per image |
|---|---|---|
| Dense layer, every pixel to every pixel | about $4 \times 10^{10}$ | about $4 \times 10^{10}$ |
| Convolution, $k = 3$ | $64 \cdot 64 \cdot 9 = 36{,}864$ | about $1.2 \times 10^8$ |
| Depthwise separable, $k = 3$ | $64 \cdot 9 + 64 \cdot 64 = 4{,}672$ | about $1.5 \times 10^7$ |

A depthwise separable layer splits the work in two. First, one $k \times k$
filter per channel, with no mixing between channels. Then a 1-by-1
convolution that mixes the channels. It needs about eight times fewer weights
here, and phone-sized networks rely on it.

im2col has a memory cost. The column matrix stores every pixel about $k^2$
times, once for each patch it appears in. For a 3-by-3 kernel, that is about
nine times the input's memory.

## Where this breaks

- **Borders.** Without padding, each layer shrinks the image, and pixels near
  the edge are seen less. Padding of $(k - 1)/2$ keeps the size the same for
  odd $k$ and stride 1.
- **Shift isn't everything.** A convolution handles shifts well, but not
  rotations or changes of scale. People add rotated and resized copies of
  training images, which is called data augmentation.
- **Stride and pooling lose detail.** Striding skips pixels, so a small shift
  can change the output more than you'd expect. Some networks blur before
  downsampling to reduce this.
- **Too small a receptive field.** If the top units can't see a whole object,
  the network can't recognise it. Add layers, strides, or dilation.
- **im2col memory.** Large images and batches can run out of memory. Fast
  libraries use other methods for large kernels, or compute the columns in
  pieces.

## The exercise

You implement `conv_output_shape`, `im2col`, `col2im`, `conv2d_forward`,
`conv2d_backward`, `max_pool_forward`, and `receptive_field`. The forward and
backward pass must go through `im2col` and matrix multiplies, not through a
loop over every output pixel.

The checks verify the output-size formula, including dilation, that `im2col`
gathers the same patches as direct slicing, and that `conv2d_forward` matches
a slow loop reference with and without stride and padding. They check all
three gradients from `conv2d_backward` against finite differences, and that
shifting the input shifts the output. The last checks test max pooling
against a direct windowed maximum and `receptive_field` against the 15 from
the worked example.
