---
title: Convolutional networks
slug: 20-convolutions
part: "Part 4 — Vision and sequences"
summary: Convolution as weight sharing and locality made precise, with the equivariance it proves, the backward pass it implies, and the matrix multiply it reduces to.
minutes: 60
gpu: false
objectives:
  - Prove that convolution commutes with translation, and explain why pooling only approximates invariance.
  - Derive the output-size formula and the receptive-field recursion from first principles.
  - Derive the backward pass of a convolution as a correlation with a flipped kernel, and implement it through im2col.
  - Compare the parameter and flop cost of a conv layer against the equivalent dense layer, and against a depthwise separable factorisation.
prereqs:
  - 16-backpropagation
exercise: 20-convolutions
---

# Convolutional networks

A multilayer perceptron applied to a 224-by-224 RGB image flattens it into a
vector of 150,528 numbers and multiplies by a dense weight matrix that has no
notion of which pixel sits next to which. Learning a detector for some pattern
this way means learning it independently at every location the pattern might
appear, because nothing in the architecture says that a shape found in the
top-left corner is the same shape when it turns up in the bottom-right.
Backpropagation, from the previous lesson, tells you how to differentiate that
dense matrix. It has nothing to say about why the matrix should be dense.

Convolution replaces the dense matrix with two constraints written directly
into the weights: locality, so each output depends on a small neighbourhood of
the input, and weight sharing, so the same small set of weights is reused at
every neighbourhood. Neither is a consequence of the data. Both are priors you
impose because natural images are approximately stationary — a texture looks
the same statistics wherever it appears — and because an object's identity does
not depend on its pixel coordinates. This lesson makes those two constraints
precise: what they buy in parameter count, the equivariance they prove, the
backward pass they imply, and the single matrix multiply, `im2col`, that makes
the whole operation fast on hardware built for dense linear algebra.

## What you already know

- Reverse-mode automatic differentiation computes a full gradient in a small
  constant multiple of the forward pass, by propagating vector-Jacobian
  products backward through the computation graph.
- The backward pass of any linear map is multiplication by its transpose:
  $\langle Ax, y \rangle = \langle x, A^{\top} y \rangle$ for every $x, y$.
- A matrix product of shapes $(m, k)$ and $(k, n)$ costs $2mkn$ flops.
- `mlcheck.finite_difference` approximates a gradient with central differences,
  accurate to $O(\varepsilon^2)$ per coordinate.
- Composing linear maps composes their matrices; the chain rule for a stack of
  linear layers is a chain of matrix products.

## Convolution as a constrained linear map

Fix a single input channel and a 1-D signal for a moment. A dense layer maps
$x \in \mathbb{R}^n$ to $y \in \mathbb{R}^n$ by $y = Wx$ with $W \in
\mathbb{R}^{n \times n}$ arbitrary: $n^2$ free parameters. A convolution with a
kernel $w \in \mathbb{R}^k$, $k \ll n$, computes instead

$$
y[i] = \sum_{d=0}^{k-1} w[d]\, x[i + d] ,
$$

which is still linear in $x$, so it is still $y = Wx$ for some matrix — but
now $W$ is banded (row $i$ is zero outside columns $i, \dots, i+k-1$, which is
locality) and every diagonal is constant (row $i$ and row $i+1$ carry the same
$k$ numbers shifted by one, which is weight sharing). Locality and weight
sharing are exactly the statement that $W$ is banded and Toeplitz. The number
of free parameters drops from $n^2$ to $k$, independent of $n$: the entire
saving of a convolution is the claim that this restricted class of matrices is
the right hypothesis class for images and audio.

In two dimensions, with $C_{\text{in}}$ input channels and $C_{\text{out}}$
output channels, the forward map with kernel $w \in \mathbb{R}^{C_{\text{out}}
\times C_{\text{in}} \times k \times k}$, stride $s$, and padding $p$ is

$$
y[c_o, i, j] = b[c_o] + \sum_{c_i=1}^{C_{\text{in}}} \sum_{d_h=0}^{k-1}
\sum_{d_w=0}^{k-1} w[c_o, c_i, d_h, d_w]\, x_p[c_i,\, s i + d_h,\, s j + d_w] ,
$$

where $x_p$ is $x$ zero-padded by $p$ on every spatial side. This is the
cross-correlation, not the flipped convolution of signal processing; every
deep learning framework calls it "convolution" and this lesson follows that
convention, because the sign of the flip never matters when the kernel is
learned.

## Translation equivariance, proved

Define the shift operator $(T_a x)[i] = x[i - a]$ on an infinite (or
circular) signal. Convolution commutes with it:

$$
(w * T_a x)[i] = \sum_d w[d]\, (T_a x)[i + d] = \sum_d w[d]\, x[i + d - a]
= (w * x)[i - a] = (T_a (w * x))[i] .
$$

So $w * (T_a x) = T_a (w * x)$: shifting the input shifts the output by the
same amount, exactly. This is **equivariance**, not invariance — the output
moves with the input rather than staying fixed — and it is a theorem about the
operator, true for every kernel $w$, not a property that training has to
discover. It is also exact only for signals with no boundary, or with circular
padding. Zero padding breaks it near the edges: a shift that pulls fresh zeros
into the receptive field of a boundary output produces a different value than
the shift theorem predicts, because $x_p$ is not literally a shifted copy of
itself there. The theorem is exact in the interior and approximate within $k$
pixels of the border.

## From equivariance to invariance: pooling

Equivariance means the feature map slides when the input slides; it does not
by itself make anything invariant. A pooling layer trades resolution for
approximate invariance: max pooling with window and stride $q$ computes

$$
y[i] = \max_{0 \le d < q} x[qi + d] ,
$$

which is invariant to a shift of the input by $a$ as long as the same $q$
input pixels — after accounting for the shift — remain the maximiser and
land in the same pooling window. That holds for shifts $|a| < q$ only when the
maximiser does not fall near a window boundary, and it fails outright when a
shift moves the maximiser into a different window. Subsampling by $q$ also
throws away every frequency component above the new Nyquist limit, so pooling
buys invariance to sub-pixel jitter within a window at the cost of aliasing
anything that varies faster than the stride resolves. This is why invariance
from pooling is a statistical tendency measured over many images, not a
theorem like equivariance is: it degrades continuously with the size of the
shift, and it degrades faster for high-frequency content.

## Output size, stride, padding, and dilation

A dilated kernel inserts $\delta - 1$ zeros between taps, so a kernel of
$k$ taps and dilation $\delta$ reaches an effective span of
$k' = \delta(k - 1) + 1$ input positions. Sliding that span across a padded
input of length $n + 2p$ with stride $s$, the number of positions it fits
without running off the end is

$$
n_{\text{out}} = \left\lfloor \frac{n + 2p - k'}{s} \right\rfloor + 1
= \left\lfloor \frac{n + 2p - \delta(k-1) - 1}{s} \right\rfloor + 1 .
$$

Each spatial dimension obeys this formula independently, so a rectangular
input and a rectangular kernel apply it once per axis.

## Receptive fields

The receptive field of a unit is the set of input positions that can affect
it. Let $r_l$ be the receptive field size (in input pixels) of a unit in layer
$l$, and let $j_l = \prod_{i \le l} s_i$ be the **jump**: the spacing, in
input pixels, between adjacent units of layer $l$. A unit in layer $l$ reads
$k_l$ adjacent units of layer $l-1$, each $j_{l-1}$ input pixels apart, so
moving across the kernel widens the receptive field by $(k_l - 1)$ steps of
size $j_{l-1}$:

$$
r_l = r_{l-1} + (k_l - 1) \prod_{i < l} s_i , \qquad r_0 = 1 .
$$

The product $\prod_{i<l} s_i$ is the jump of the *previous* layer,
$j_{l-1}$, which is why the recursion needs both quantities: $r_l$ tracks
size, $j_l$ tracks spacing, and only spacing compounds multiplicatively.
Three stride-2, kernel-3 layers make this concrete:

| Layer | $k_l$ | $s_l$ | $r_l$ | $j_l$ |
|---|---|---|---|---|
| 0 (input) | — | — | 1 | 1 |
| 1 | 3 | 2 | $1 + 2 \cdot 1 = 3$ | 2 |
| 2 | 3 | 2 | $3 + 2 \cdot 2 = 7$ | 4 |
| 3 | 3 | 1 | $7 + 2 \cdot 4 = 15$ | 4 |

Three $3\times3$ layers with stride 1 throughout would reach only $r_3 = 7$;
the two stride-2 layers here more than double that for the same kernel sizes,
because every later layer inherits the compounded jump. This is the
architectural reason early downsampling is standard: it is the cheapest way
to grow the receptive field.

## Parameters and flops: conv versus dense

A conv layer with kernel $k \times k$, $C_{\text{in}}$ input and
$C_{\text{out}}$ output channels has

$$
\#\text{params} = C_{\text{out}} (C_{\text{in}} k^2 + 1),
$$

independent of the spatial size $H \times W$. Producing an output of shape
$C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$ costs
$H_{\text{out}} W_{\text{out}} C_{\text{out}} C_{\text{in}} k^2$
multiply-adds, so $2 H_{\text{out}} W_{\text{out}} C_{\text{out}}
C_{\text{in}} k^2$ flops.

A dense layer connecting the same flattened input to the same flattened output
has $C_{\text{in}} H W \cdot C_{\text{out}} H_{\text{out}} W_{\text{out}}$
parameters, and the same number of multiply-adds. The ratio of dense to conv
parameters is

$$
\frac{\#\text{params}_{\text{dense}}}{\#\text{params}_{\text{conv}}}
\approx \frac{H W H_{\text{out}} W_{\text{out}}}{k^2} .
$$

For a $32 \times 32$ input, a $32 \times 32$ output, and $k=3$, that ratio is
$32^4 / 9 \approx 1.16 \times 10^5$: the dense layer needs on the order of a
hundred thousand times more parameters to represent a map that, if it happens
to be local and shift-invariant, the conv layer represents exactly. The cost
of that compression is expressive power: a conv layer cannot represent a map
that genuinely depends on distant pixels without stacking enough layers to
grow the receptive field there.

## Convolution as a matrix multiply: `im2col`

Hardware is fast at one thing above all others: dense matrix multiplication.
`im2col` turns a convolution into exactly that. For every output position
$(i, j)$, gather the $C_{\text{in}} k^2$ input values its receptive field
touches into one column of a matrix $X_{\text{col}} \in \mathbb{R}^{(C_{\text{in}}
k^2) \times L}$, where $L = H_{\text{out}} W_{\text{out}}$. Reshape the kernel
to $W_{\text{mat}} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} k^2)}$.
Then

$$
Y_{\text{mat}} = W_{\text{mat}} X_{\text{col}} + b \mathbf{1}^{\top} ,
$$

reshaped back to $C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$,
computes exactly the sum in the definition of $y[c_o, i, j]$ above, term for
term. `im2col` itself does no arithmetic — it is a gather, a linear map given
by a 0/1 selection matrix $S$ with $X_{\text{col}} = Sx$ — but it turns
$C_{\text{in}} k^2 \cdot L \cdot C_{\text{out}}$ multiply-adds into one call to
a matrix-multiply routine, which is worth the $k^2$ memory blow-up from
duplicating every pixel that participates in $k^2$ overlapping patches.

## The backward pass, derived

Because $y[c_o,i,j]$ is an explicit sum, every gradient follows from
differentiating it directly.

**Bias.** $\partial y[c_o,i,j] / \partial b[c_o] = 1$ for every $(i,j)$, so

$$
\frac{\partial L}{\partial b[c_o]} = \sum_{i,j} \frac{\partial L}{\partial y[c_o,i,j]} .
$$

**Weights.** $\partial y[c_o,i,j] / \partial w[c_o,c_i,d_h,d_w] = x_p[c_i, si+d_h,
sj+d_w]$, so summing over every output position that uses a given weight,

$$
\frac{\partial L}{\partial w[c_o,c_i,d_h,d_w]}
= \sum_{i,j} \frac{\partial L}{\partial y[c_o,i,j]}\, x_p[c_i,\, si+d_h,\, sj+d_w] .
$$

This is a correlation: slide the $H_{\text{out}} \times W_{\text{out}}$
gradient map, with stride $s$, over the padded input, and read off one number
per kernel tap — the same sliding-window sum that computed the forward pass,
with the roles of "input" and "kernel" swapped between $x_p$ and $\partial
L/\partial y$.

**Input.** Every $x_p[c_i, m, n]$ contributes to $y[c_o,i,j]$ whenever
$m = si + d_h$ and $n = sj + d_w$ for some valid kernel offset, so by the
multivariable chain rule,

$$
\frac{\partial L}{\partial x_p[c_i,m,n]}
= \sum_{c_o} \sum_{\substack{d_h, d_w \\ i = (m-d_h)/s \\ j = (n-d_w)/s}}
w[c_o,c_i,d_h,d_w]\, \frac{\partial L}{\partial y[c_o,i,j]} .
$$

For stride $s=1$, substitute $d_h' = k-1-d_h$ (and likewise for $d_w$) to flip
the kernel: the sum becomes a correlation of the flipped kernel with the
gradient map, padded by $k-1-p$ — the **full correlation with the flipped
kernel** that gives this section its name. For $s > 1$, the same identity
holds after inserting $s-1$ zeros between consecutive entries of $\partial
L/\partial y$ along each spatial axis, which is the operation this course
calls a transposed convolution when it appears as a layer in its own right.

In the `im2col` picture this is one line of matrix calculus. Since $Y_{\text{mat}}
= W_{\text{mat}} X_{\text{col}} + b\mathbf{1}^{\top}$ and $X_{\text{col}} = Sx$,

$$
\frac{\partial L}{\partial W_{\text{mat}}} = \frac{\partial L}{\partial Y_{\text{mat}}} X_{\text{col}}^{\top}, \qquad
\frac{\partial L}{\partial X_{\text{col}}} = W_{\text{mat}}^{\top} \frac{\partial L}{\partial Y_{\text{mat}}}, \qquad
\frac{\partial L}{\partial x} = S^{\top} \frac{\partial L}{\partial X_{\text{col}}} .
$$

$S^{\top}$ scatters each entry of $\partial L / \partial X_{\text{col}}$ back
to the input position it was copied from and sums the contributions where
patches overlapped — the operation usually named `col2im`, and precisely the
adjoint of the gather that `im2col` performed. The index-chasing derivation
above and this one-line matrix derivation compute the same numbers; the first
tells you what the answer means, the second tells you how to implement it.

## $1 \times 1$ convolutions and depthwise separable convolutions

Set $k = 1$: the sum over $d_h, d_w$ disappears and $y[c_o,i,j] =
b[c_o] + \sum_{c_i} w[c_o,c_i]\, x[c_i,i,j]$ is a single linear map applied
independently at every spatial position — pure channel mixing, with no
spatial receptive field at all. It costs $H W C_{\text{in}} C_{\text{out}}$
multiply-adds and is the cheapest way to change channel count, which is why
bottleneck blocks (next lesson) use it to compress and expand.

A depthwise separable convolution factors a $k \times k$ conv into two
cheaper stages: a **depthwise** stage that convolves each of the
$C_{\text{in}}$ channels with its own $k \times k$ filter and no mixing across
channels, at $H_{\text{out}} W_{\text{out}} C_{\text{in}} k^2$ multiply-adds,
followed by a $1 \times 1$ **pointwise** stage that mixes channels, at
$H_{\text{out}} W_{\text{out}} C_{\text{in}} C_{\text{out}}$ multiply-adds.
The ratio of separable cost to standard cost is

$$
\frac{H_{\text{out}}W_{\text{out}} C_{\text{in}} (k^2 + C_{\text{out}})}
{H_{\text{out}}W_{\text{out}} C_{\text{out}} C_{\text{in}} k^2}
= \frac{1}{C_{\text{out}}} + \frac{1}{k^2} .
$$

At $k=3$, $C_{\text{out}}=256$ — an ordinary mid-network layer — this is
$1/256 + 1/9 \approx 0.115$: about a ninefold reduction in flops for a
representation that can express any standard conv whose spatial and
channel-mixing behaviour happen to factor, and cannot express one whose
weights genuinely couple a spatial offset to a specific pair of channels.

## Cost and scaling

| Operation | Params | Multiply-adds |
|---|---|---|
| Dense layer, same input/output size | $C_{\text{in}}HW \cdot C_{\text{out}}H_{\text{out}}W_{\text{out}}$ | same |
| Standard conv, $k \times k$ | $C_{\text{out}}(C_{\text{in}}k^2+1)$ | $H_{\text{out}}W_{\text{out}}C_{\text{out}}C_{\text{in}}k^2$ |
| $1\times1$ conv | $C_{\text{out}}(C_{\text{in}}+1)$ | $HWC_{\text{in}}C_{\text{out}}$ |
| Depthwise separable, $k \times k$ | $C_{\text{in}}(k^2+1) + C_{\text{out}}(C_{\text{in}}+1)$ | $H_{\text{out}}W_{\text{out}}C_{\text{in}}(k^2+C_{\text{out}})$ |

Flops follow by doubling every entry in the right column, since each
multiply-add is two floating-point operations; the ratios in this lesson are
unaffected by that constant factor, so the rest of the discussion works in
multiply-adds directly.

`im2col` trades memory for the matmul speedup: $X_{\text{col}}$ holds $k^2$
copies of every interior pixel, so its footprint is $k^2$ times the input's,
which is the standing objection to it on memory-constrained hardware and the
reason production kernels (Winograd, FFT-based, or direct convolution
algorithms) avoid materialising it in full.

## Where this breaks

**Non-stationary signals.** Locality and weight sharing are correct priors
exactly when the statistics of the signal do not depend on position. Tabular
data with unrelated columns, or an image task where absolute position carries
meaning (a digit's position on a cheque, not the digit itself), gets nothing
from convolution and sometimes actively suffers from it; those tasks add
explicit positional inputs or drop the constraint back to a dense layer.

**Pooling degrades continuously, not categorically.** Because invariance from
subsampling holds only for shifts smaller than the stride and only away from
window boundaries, a network's actual robustness to shift is an empirical,
graded property, not the exact guarantee that equivariance is. Strided
convolutions used for downsampling inherit the same aliasing weakness pooling
has, for the same reason: subsampling any signal above its new Nyquist rate
loses information no amount of training recovers.

**Zero padding is not translation-equivariant.** The proof of equivariance
assumed no boundary. Every network with zero padding has border artefacts —
features near the edge behave differently from features in the interior — and
deep stacks compound this near the border more than the interior, which is
why some architectures replace zero padding with reflection or circular
padding when the border matters.

**Fixed receptive fields cap long-range reasoning.** Growing the receptive
field to cover a whole image by stacking small kernels costs depth
logarithmic in the image size with aggressive striding, but each layer added
this way is still local at every individual step. Tasks that need one output
to depend on two far-apart, dynamically-determined input positions — the kind
of dependency language modelling needs constantly — are exactly what motivates
dropping the locality constraint altogether, which is the opening self-attention
walks through in Part 5.

## The exercise

You implement the forward and backward pass of a 2-D convolution through
`im2col`, plus the two supporting measurements this lesson derived: output
size and receptive field.

Write `conv_output_shape(size, kernel_size, stride, padding, dilation=1)`
returning the formula derived above for one spatial axis; `im2col(x,
kernel_size, stride, padding)`, which gathers every patch of the batched input
`x` into the column matrix $X_{\text{col}}$; `conv2d_forward(x, w, b, stride,
padding)`, built on top of `im2col` as a single matrix multiply; and
`conv2d_backward(dY, x, w, b, stride, padding)`, returning the gradient with
respect to the input, the weights, and the bias, built on `im2col`'s adjoint.
Also write `max_pool_forward(x, pool_size, stride)` and `receptive_field(layers)`,
which takes a list of `(kernel_size, stride)` pairs from the first layer of a
stack to the last and returns the final receptive field by the recursion
above.

The checks confirm `conv_output_shape` against known stride-and-padding cases,
`im2col` against direct slicing on a small tensor, `conv2d_forward` against an
independent quadruple-loop reference, all three outputs of `conv2d_backward`
against `mlcheck.finite_difference`, that shifting the input shifts the
output by the same amount away from the border, that `max_pool_forward`
matches a direct windowed maximum, and that `receptive_field` reproduces the
three-layer stack worked out in the table above.
