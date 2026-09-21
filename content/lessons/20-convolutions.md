---
title: Convolutional networks
slug: 20-convolutions
part: "Part 4 — Vision and sequences"
summary: Convolution as weight sharing and locality, the equivariance that follows from it, and the matrix multiply that makes it fast.
minutes: 50
gpu: false
objectives:
  - Show that convolution commutes with translation, and explain why pooling only approximates invariance.
  - Compute output size and receptive field for a stack of convolutional layers.
  - Implement the forward and backward pass of a convolution through im2col.
  - Compare the parameter and flop cost of a conv layer against a dense layer, and against a depthwise separable factorisation.
prereqs:
  - 16-backpropagation
exercise: 20-convolutions
---

# Convolutional networks

A multilayer perceptron applied to a 224-by-224 RGB image flattens it into a
vector of 150,528 numbers and multiplies by a dense weight matrix that has no
notion of which pixel sits next to which. Learning a detector for some
pattern this way means learning it separately at every location it might
appear, because nothing in the architecture says a shape in the top-left
corner is the same shape in the bottom-right. Backpropagation tells you how
to differentiate that dense matrix. It has nothing to say about why the
matrix should be dense in the first place.

Convolution replaces the dense matrix with two constraints built into the
weights: locality, so each output depends on a small neighbourhood of the
input, and weight sharing, so the same small set of weights is reused at
every neighbourhood. Neither constraint follows from the data. You impose
both because natural images are approximately stationary — a texture looks
the same wherever it appears — and an object's identity does not depend on
its pixel coordinates. This lesson makes those two constraints precise: what
they buy in parameter count, the equivariance they prove, the backward pass
they imply, and the matrix multiply, `im2col`, that makes the operation fast
on hardware built for dense linear algebra.

## What you already know

- Reverse-mode automatic differentiation computes a full gradient in a small
  constant multiple of the forward pass's cost.
- The backward pass of any linear map is multiplication by its transpose.
- A matrix product of shapes $(m, k)$ and $(k, n)$ costs $2mkn$ flops.
- `mlcheck.finite_difference` approximates a gradient with central
  differences.
- Composing linear maps composes their matrices, so the chain rule for a
  stack of linear layers is a chain of matrix products.

## Convolution as a constrained linear map

Fix a single input channel and a 1-D signal. A dense layer maps $x \in
\mathbb{R}^n$ to $y \in \mathbb{R}^n$ using an arbitrary matrix with $n^2$
free parameters. A convolution with a kernel $w \in \mathbb{R}^k$, $k \ll n$,
computes instead

$$
y[i] = \sum_{d=0}^{k-1} w[d]\, x[i + d] ,
$$

which is still linear in $x$, so it is still a matrix multiply, but that
matrix has only $k$ free parameters instead of $n^2$: nonzero only near the
diagonal (locality), and identical along every diagonal (weight sharing).
That drop from $n^2$ to $k$, independent of $n$, is the entire saving a
convolution offers, and it costs you exactly what you would expect: a
convolution can only represent a map that is local and shift-invariant.

In two dimensions, with $C_{\text{in}}$ input and $C_{\text{out}}$ output
channels, a kernel $w \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}
\times k \times k}$, stride $s$, and padding $p$ compute

$$
y[c_o, i, j] = b[c_o] + \sum_{c_i=1}^{C_{\text{in}}} \sum_{d_h=0}^{k-1}
\sum_{d_w=0}^{k-1} w[c_o, c_i, d_h, d_w]\, x_p[c_i,\, s i + d_h,\, s j + d_w] ,
$$

where $x_p$ is $x$ zero-padded by $p$ on every spatial side. Every deep
learning framework calls this operation "convolution," although it is
technically a cross-correlation, because the sign of the flip never matters
once the kernel is learned.

## Translation equivariance, proved

Convolution commutes with a shift of the input: shifting $x$ by $a$ positions
shifts the output $w * x$ by exactly $a$ positions too, for every kernel
$w$, because sliding the input past a fixed kernel gives the same overlaps
in a shifted order. This property is **equivariance**, not invariance — the
output moves with the input rather than staying fixed — and it is a theorem
about the operator, true before training touches the kernel at all. It holds
exactly only away from a boundary. Zero padding breaks it near the edges,
because a shift that pulls fresh zeros into a boundary output's receptive
field changes what that output sees.

## From equivariance to invariance: pooling

Equivariance means the feature map slides when the input slides. It does not
by itself make anything invariant. A pooling layer trades resolution for
approximate invariance: max pooling with window and stride $q$ computes

$$
y[i] = \max_{0 \le d < q} x[qi + d] ,
$$

which stays unchanged under a shift only as long as the same input pixel
remains the maximiser and still lands in the same window — true for small
shifts away from a window boundary, and false the moment a shift moves the
maximiser into a different window. Pooling also throws away any detail finer
than the sampling resolution it leaves behind, so it buys robustness to
small jitter at the cost of losing anything that varies faster than the
stride resolves. Unlike
equivariance, this robustness is a statistical tendency, not a guarantee: it
degrades gradually as the shift grows.

## Output size, stride, padding, and dilation

A dilated kernel inserts $\delta - 1$ zeros between taps, reaching an
effective span of $k' = \delta(k-1) + 1$ input positions. Sliding that span
with stride $s$ across a padded input of length $n + 2p$ gives

$$
n_{\text{out}} = \left\lfloor \frac{n + 2p - \delta(k-1) - 1}{s} \right\rfloor + 1 .
$$

Each spatial dimension applies this formula independently.

## Receptive fields

A unit's receptive field is the set of input positions that can affect it.
Let $r_l$ be that size at layer $l$, and let $j_l$ be the **jump**: the
spacing, in input pixels, between adjacent units of layer $l$. A unit reads
$k_l$ adjacent units from the previous layer, each $j_{l-1}$ input pixels
apart, so

$$
r_l = r_{l-1} + (k_l - 1)\, j_{l-1} , \qquad j_l = j_{l-1}\, s_l , \qquad r_0 = j_0 = 1 .
$$

A worked stack makes this concrete: three layers with kernel size 3 and
strides 2, 2, 1 give $r_0=1$, $r_1=3$, $r_2=7$, $r_3=15$. The same three
kernels run at stride 1 throughout would reach only $r_3=7$. Downsampling
early is the cheapest way to grow the receptive field, because every later
layer inherits the compounded jump.

## Parameters and flops: conv versus dense

A conv layer with kernel $k \times k$ has $C_{\text{out}}(C_{\text{in}}k^2 +
1)$ parameters, independent of the spatial size, while a dense layer
connecting the same flattened input and output needs a parameter for every
input-output pair. For a $32 \times 32$ input and output with $k=3$, the
dense layer needs roughly $32^4/9 \approx 1.16 \times 10^5$ times more
parameters than the conv layer to represent a map that, if it happens to be
local and shift-invariant, the conv layer represents exactly. That
compression costs expressive power: a conv layer cannot represent a map that
depends on distant pixels without stacking enough layers to reach them.

## Convolution as a matrix multiply: `im2col`

Hardware is fastest at one thing: dense matrix multiplication. `im2col`
turns a convolution into exactly that. For every output position, gather the
$C_{\text{in}}k^2$ input values its receptive field touches into one column
of a matrix $X_{\text{col}}$, and reshape the kernel into a matrix
$W_{\text{mat}}$. Then

$$
Y_{\text{mat}} = W_{\text{mat}} X_{\text{col}} + b \mathbf{1}^{\top}
$$

computes exactly the convolution sum above, reshaped back to the output's
shape. `im2col` does no arithmetic of its own — it is a gather that
duplicates every pixel touched by more than one patch — but it turns the
convolution into one call to a matrix-multiply routine, which is worth the
$k^2$-fold memory cost on hardware built for dense linear algebra.

## The backward pass

The bias gradient sums $\partial L/\partial y$ over every spatial position
that uses it. The weight gradient at kernel tap $(d_h, d_w)$ is a
correlation of the upstream gradient with the input: slide the gradient map
across the padded input at the same stride the forward pass used, and sum
the products. In the `im2col` picture, both gradients are one line of matrix
calculus:

$$
\frac{\partial L}{\partial W_{\text{mat}}} = \frac{\partial L}{\partial Y_{\text{mat}}} X_{\text{col}}^{\top}, \qquad
\frac{\partial L}{\partial X_{\text{col}}} = W_{\text{mat}}^{\top} \frac{\partial L}{\partial Y_{\text{mat}}} .
$$

The input gradient needs one more step, because `im2col`'s gather is itself
a linear map, $X_{\text{col}} = Sx$ for a 0/1 selection matrix $S$. Its
backward pass is $S^\top$, usually called `col2im`: it scatters each column
entry back to the input position it came from and sums the contributions
where patches overlapped,

$$
\frac{\partial L}{\partial x} = S^{\top} \frac{\partial L}{\partial X_{\text{col}}} .
$$

Indexed out directly rather than through `im2col`, this same gradient is a
full correlation of the flipped kernel with the upstream gradient map — the
classical description of a convolution's backward pass, and the same
numbers this matrix form computes.

## $1 \times 1$ convolutions and depthwise separable convolutions

Setting $k=1$ removes the spatial sum entirely: $y[c_o,i,j] = b[c_o] +
\sum_{c_i} w[c_o,c_i]\, x[c_i,i,j]$ mixes channels at every position
independently, with no spatial receptive field at all. It is the cheapest
way to change channel count, at $HWC_{\text{in}}C_{\text{out}}$
multiply-adds.

A depthwise separable convolution factors a $k \times k$ conv into a
**depthwise** stage, which convolves each channel with its own $k \times k$
filter and mixes no channels, followed by a $1\times1$ **pointwise** stage
that mixes channels. The ratio of its cost to a standard conv's is

$$
\frac{1}{C_{\text{out}}} + \frac{1}{k^2} .
$$

At $k=3$ and $C_{\text{out}}=256$, a typical mid-network layer, this comes
to about $0.115$ — roughly a ninefold reduction in flops, at the cost of
being unable to represent a kernel whose weights genuinely couple a spatial
offset to a specific pair of channels.

## Cost and scaling

| Operation | Parameters | Multiply-adds |
|---|---|---|
| Dense layer, matched input and output | $C_{\text{in}}HW \cdot C_{\text{out}}H_{\text{out}}W_{\text{out}}$ | same |
| Standard conv, $k \times k$ | $C_{\text{out}}(C_{\text{in}}k^2+1)$ | $H_{\text{out}}W_{\text{out}}C_{\text{out}}C_{\text{in}}k^2$ |
| $1\times1$ conv | $C_{\text{out}}(C_{\text{in}}+1)$ | $HWC_{\text{in}}C_{\text{out}}$ |
| Depthwise separable, $k \times k$ | $C_{\text{in}}(k^2+1) + C_{\text{out}}(C_{\text{in}}+1)$ | $H_{\text{out}}W_{\text{out}}C_{\text{in}}(k^2+C_{\text{out}})$ |

`im2col` trades memory for that matmul speedup: $X_{\text{col}}$ holds $k^2$
copies of every interior pixel, so its footprint is $k^2$ times the input's.
That is the standing objection to it on memory-constrained hardware, and the
reason production kernels favour direct, Winograd, or FFT-based convolution
algorithms that avoid materialising it in full.

## Where this breaks

**Non-stationary signals.** Locality and weight sharing are correct priors
exactly when a signal's statistics do not depend on position. Tabular data
with unrelated columns, or a task where absolute position carries meaning,
gets little from convolution.

**Pooling degrades continuously, not categorically.** Invariance from
subsampling holds only for shifts smaller than the stride, away from window
boundaries, so a network's robustness to shift is measured empirically, not
guaranteed the way equivariance is. Strided convolutions used for
downsampling share the same weakness for the same reason.

**Zero padding is not translation-equivariant.** The proof above assumed no
boundary. Every network with zero padding has border artefacts that deep
stacks compound, which is why some architectures use reflection or circular
padding instead when the border matters.

**Fixed receptive fields cap long-range reasoning.** Stacking small kernels
can grow a receptive field to cover a whole image, but each layer added is
still local at every step. A task that needs one output to depend on two
far-apart, dynamically determined positions is exactly what motivates
dropping the locality constraint, which Part 5 covers with self-attention.

## The exercise

You implement the forward and backward pass of a 2-D convolution through
`im2col`, plus the two supporting measurements this lesson derived: output
size and receptive field.

Write `conv_output_shape(size, kernel_size, stride, padding, dilation=1)`,
returning the formula derived above for one spatial axis; `im2col(x,
kernel_size, stride, padding)`, which gathers every patch of the batched
input `x` into the column matrix; `conv2d_forward(x, w, b, stride, padding)`,
built on top of `im2col` as a single matrix multiply; and
`conv2d_backward(dY, x, w, b, stride, padding)`, returning the gradient with
respect to the input, the weights, and the bias, built on `im2col`'s adjoint.
Also write `max_pool_forward(x, pool_size, stride)` and
`receptive_field(layers)`, which takes a list of `(kernel_size, stride)`
pairs from the first layer of a stack to the last and returns the final
receptive field using the recursion above.

The checks confirm `conv_output_shape` against known stride-and-padding
cases, `im2col` against direct slicing on a small tensor, and
`conv2d_forward` against an independent quadruple-loop reference. All three
outputs of `conv2d_backward` are checked against `mlcheck.finite_difference`.
A further check shifts the input and confirms the output shifts by the same
amount away from the border, `max_pool_forward` is checked against a direct
windowed maximum, and `receptive_field` is checked against the three-layer
stack worked out above.
