---
title: Residual networks
slug: 21-residual-networks
part: "Part 4 — Vision and sequences"
summary: "Add a shortcut around each block so gradients survive a deep stack, and see why the bottleneck block is cheap."
minutes: 60
gpu: true
objectives:
  - Build the forward and backward pass of a residual block by hand in numpy.
  - Explain why the shortcut keeps the gradient from shrinking in a deep stack.
  - Measure the gradient that reaches the input of a 50-layer residual stack and a 50-layer plain stack.
  - Compute the multiply-add cost of a bottleneck block and compare it with two full-width convolutions.
prereqs:
  - 20-convolutions
  - 18-initialisation-and-norm
exercise: 21-residual-networks
---

# Residual networks

You might expect a deeper network to always do at least as well as a shallow
one. In practice it doesn't. Researchers found that a 56-layer plain
convolutional network had higher *training* error than a 20-layer one, with
the same data and the same optimiser. That isn't overfitting, because
overfitting means training error is low and test error is high. The deep
network could in principle copy the shallow one and leave its extra layers
doing nothing. Gradient descent couldn't find that solution.

This is called the **degradation problem**. The fix is small and powerful. Add
each block's input back to its output, so the block only has to learn a
correction. If the correction is zero, the block passes its input through
untouched. That one addition, the **shortcut**, keeps the training signal
strong however deep the network gets.

In this lesson you build a residual block by hand, with its backward pass.
Then you measure how much gradient survives 50 layers with and without the
shortcut. You also count the cost of the bottleneck block that makes very deep
networks affordable.

## What you already know

- Backpropagation applies the chain rule layer by layer, from the loss back to
  the input.
- A linear layer computes `x @ W.T + b`, and ReLU sets negative values to zero.
- Gradients shrink or grow when you multiply them by many numbers smaller or
  larger than one.
- Batch norm ends with a learned scale $\gamma$ and shift $\beta$ per channel.
- A convolution's cost grows with output height, output width, input channels,
  output channels, and kernel area.

## The residual block

Think of editing an essay. A plain layer rewrites the whole essay from
scratch. A residual block keeps the essay and writes suggested edits in the
margin. If there's nothing to fix, it writes no edits, and the essay comes
through unchanged.

A residual block computes

$$
y = x + F(x)
$$

Read it aloud: the output $y$ is the input $x$ plus a correction $F(x)$. The
function $F$ is called the **residual branch**. It can be any stack of layers,
as long as its output has the same shape as $x$, so that you can add the two
element by element. The plain path from $x$ to the plus sign is the
**shortcut**.

In this exercise, $x$ has shape `(n, d)`: $n$ examples, each with $d$
features. The branch is two linear layers with a ReLU between them:

$$
h_1 = x W_1^{\top} + b_1, \qquad a_1 = \operatorname{ReLU}(h_1), \qquad
h_2 = a_1 W_2^{\top} + b_2, \qquad y = x + h_2
$$

Read it aloud: multiply the input by the first weight matrix and add a bias to
get $h_1$. Zero out the negative entries to get $a_1$. Multiply by the second
weight matrix and add a bias to get $h_2$. Add the original input back. Both
$W_1$ and $W_2$ have shape `(d, d)`, and both biases have shape `(d,)`. In
numpy:

```python
h1 = x @ W1.T + b1
a1 = np.maximum(h1, 0.0)
h2 = a1 @ W2.T + b2
y = x + h2
```

If you set $W_2$ and $b_2$ to zero, then $h_2$ is zero for every input, and
$y = x$ exactly. The block starts as a perfect "do nothing" layer. A plain
layer $y = F(x)$ has no such easy setting: its weights must learn to copy the
input, which is hard.

> [!KEY]
> A residual block learns only a correction to its input. With a zero branch, it
> passes the input through exactly.

### Work one example by hand

Take two examples with two features each:

$$
x = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}, \quad
W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
b_1 = [0, -1], \quad
W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}, \quad
b_2 = [0.1, 0]
$$

Follow the four steps:

1. $h_1 = x W_1^{\top} + b_1 = \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$.
   $W_1$ is the identity, so you only subtract 1 from the second column.
2. $a_1 = \operatorname{ReLU}(h_1) = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$.
   The $-1$ becomes $0$.
3. $h_2 = a_1 W_2^{\top} + b_2 = \begin{bmatrix} 2.1 & 0.5 \\ 0.1 & 1.0 \end{bmatrix}$.
   The first column doubles and gains $0.1$. The second column halves.
4. $y = x + h_2 = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.

The first check in the exercise uses exactly these numbers.

## The backward pass

The backward pass takes $dY$, the gradient of the loss with respect to $y$,
with shape `(n, d)`. It returns $dx$ and the gradient for each parameter.

The key idea is the plus sign. When you add two things, the gradient flows
into both of them unchanged. So $dY$ goes two ways: straight down the
shortcut into $dx$, and into the branch. The branch then sends back its own
contribution, which you add to $dx$.

Walk backward through the branch one line at a time:

$$
dh_2 = dY, \qquad dW_2 = dh_2^{\top} a_1, \qquad db_2 = \textstyle\sum_{\text{rows}} dh_2
$$

$$
da_1 = dh_2 W_2, \qquad dh_1 = da_1 \odot [h_1 > 0], \qquad
dW_1 = dh_1^{\top} x, \qquad db_1 = \textstyle\sum_{\text{rows}} dh_1
$$

$$
dx = dY + dh_1 W_1
$$

Read it aloud. The gradient reaching $h_2$ equals $dY$, because $y = x + h_2$.
A weight gradient is "upstream gradient, transposed, times the layer's input".
A bias gradient sums the upstream gradient over the $n$ rows, because the same
bias is added to every example. ReLU passes gradient only where its input was
positive; $\odot$ means element-by-element multiplication and $[h_1 > 0]$ is
a mask of ones and zeros. Last, $dx$ is the shortcut's $dY$ **plus** the
branch's $dh_1 W_1$. In numpy:

```python
dW2 = dY.T @ a1
dh1 = (dY @ W2) * (h1 > 0)
dx = dY + dh1 @ W1
```

> [!WARNING]
> The forward pass uses `W1.T`, but the backward pass multiplies by `W1` with no
> transpose. With square weights, the wrong choice still runs and gives wrong
> numbers.

With $dY$ set to all ones in the example above, you get
$dW_2 = \begin{bmatrix} 1 & 3 \\ 1 & 3 \end{bmatrix}$,
$dh_1 = \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix}$, and
$dx = \begin{bmatrix} 3 & 1.5 \\ 1 & 1.5 \end{bmatrix}$.
The ones from the shortcut are visible inside $dx$.

> [!KEY]
> A plus sign copies the incoming gradient into both of its inputs. So $dx$ is
> always $dY$ from the shortcut plus whatever the branch sends back.

> [!TIP]
> After a backward pass, check that each gradient has its parameter's shape:
> `assert grads[k].shape == params[k].shape` for every key.

## Why gradients survive depth

Imagine passing a message down a line of 50 people. In a plain stack, each
person repeats only a tenth of what they heard. After 50 people the message is
$0.1^{50}$ of its original size, which is effectively nothing. In a residual
stack, each person repeats the whole message **and** adds their own small
note. The original message always reaches the end.

In symbols, when a plain layer scales the gradient by a factor $J$, a residual
layer scales it by $1 + J$. The "1" comes from the shortcut. Multiply 50 of
these factors together:

$$
\text{plain: } J^{50}, \qquad \text{residual: } (1 + J)^{50}
$$

Read it aloud: the plain stack multiplies 50 small numbers, which collapses to
zero. The residual stack multiplies 50 numbers close to one, which stays close
to one.

For matrices the same idea holds: each residual layer turns the gradient $g$
into $g + g W$ instead of $g W$. The exercise builds each $W$ as $0.1$ times a
random **orthogonal matrix**, which is a matrix that rotates a vector without
changing its length. So each plain layer shrinks the gradient's length by
exactly $0.1$. In numpy:

```python
g = g + g @ W   # residual layer
g = g @ W       # plain layer
```

Starting from a gradient of length one, the numbers come out like this:

| Depth | Residual gradient length | Plain gradient length |
|---|---|---|
| 10 | about 0.93 | $10^{-10}$ |
| 50 | about 0.53 | $10^{-50}$ |

The residual stack keeps about half its signal after 50 layers. The plain
stack keeps none.

> [!KEY]
> The shortcut adds 1 to each layer's gradient factor. The gradient no longer
> has to pass through a long product of small numbers, so it reaches the early
> layers.

Two design choices keep the shortcut clean. First, put batch norm and ReLU
*inside* the branch, before the layers, rather than after the addition. This
is called **pre-activation**, and it means nothing touches the shortcut path.
Second, start the last batch-norm scale $\gamma$ of each branch at zero. Then
every branch outputs zero at the start, and the whole network begins as the
identity.

## The bottleneck block

A basic residual block uses two $3\times3$ convolutions at full width $C$.
That gets expensive. A **bottleneck** block does three cheaper steps instead:

1. A $1\times1$ convolution shrinks $C_{\text{in}}$ channels to $C_{\text{mid}}$.
2. A $3\times3$ convolution works at the small width $C_{\text{mid}}$.
3. A $1\times1$ convolution grows back to $C_{\text{out}}$ channels.

Each convolution costs (height) times (width) times (input channels) times
(output channels) times (kernel area) multiply-adds. Add the three:

$$
\text{flops} = H W \left( C_{\text{in}} C_{\text{mid}} + 9\, C_{\text{mid}}^2 + C_{\text{mid}} C_{\text{out}} \right)
$$

Read it aloud: for each of the $H \times W$ output positions, pay
$C_{\text{in}} C_{\text{mid}}$ for the shrink, $9 C_{\text{mid}}^2$ for the
$3\times3$ step (9 is the kernel area), and $C_{\text{mid}} C_{\text{out}}$
for the grow. In Python:

```python
flops = h * w * (c_in * c_mid + 9 * c_mid * c_mid + c_mid * c_out)
```

Try $C_{\text{in}} = C_{\text{out}} = 256$, $C_{\text{mid}} = 64$, at
$14 \times 14$:

- Shrink: $256 \times 64 = 16{,}384$.
- Middle: $9 \times 64 \times 64 = 36{,}864$.
- Grow: $64 \times 256 = 16{,}384$.
- Sum per position: $69{,}632$. Times $196$ positions: $13{,}647{,}872$.

Two full-width $3\times3$ convolutions cost $196 \times 2 \times 9 \times 256
\times 256 = 231{,}211{,}008$. The bottleneck is about 17 times cheaper. With
$C_{\text{mid}} = C/4$ in general, the ratio is $\tfrac{17}{16}$ against $18$,
so ResNets with 50 or more layers use bottlenecks.

> [!KEY]
> A bottleneck block does its $3\times3$ work at a quarter of the width. That
> makes it about 17 times cheaper than two full-width $3\times3$ convolutions.

> [!RECAP]
> - The block computes `h1 = x @ W1.T + b1`, a ReLU, `h2 = a1 @ W2.T + b2`, and
>   `y = x + h2`, with no ReLU after the addition.
> - The cache holds what the backward pass needs: `x`, `W1`, `W2`, `h1`, and
>   `a1`.
> - Weight gradients are "upstream, transposed, times the layer's input". Bias
>   gradients sum over rows. The ReLU mask is `h1 > 0`.
> - `dx` is `dY + dh1 @ W1`: the shortcut term plus the branch term.
> - The bottleneck cost is $HW(C_{\text{in}} C_{\text{mid}} + 9 C_{\text{mid}}^2 +
>   C_{\text{mid}} C_{\text{out}})$.

## Build it step by step

The checks test the functions in this order.

1. **`residual_block_forward(x, params)`.** Takes `x` of shape `(n, d)` and a
   dict with `W1`, `b1`, `W2`, `b2`. Returns `(y, cache)`, with `y` of shape
   `(n, d)`. Use `@`, `.T`, and `np.maximum(h1, 0.0)`. Put everything the
   backward pass needs in `cache`: at least `x`, `W1`, `W2`, `h1`, and `a1`.
   Common mistake: writing `W1 @ x` instead of `x @ W1.T`, which fails on
   non-square shapes.
2. **Zero branch.** No new function. With `W2` and `b2` set to zero, your
   forward pass must return `x` exactly. If it doesn't, you probably applied a
   ReLU after the addition.
3. **`residual_block_backward(dY, cache)`.** Takes `dY` of shape `(n, d)` and
   the cache. Returns `(dx, grads)`, where `grads` has keys `W1`, `b1`, `W2`,
   `b2` with the same shapes as the parameters. Use `.T @`, `.sum(axis=0)` for
   biases, and `(h1 > 0)` for the ReLU mask. Common mistake: returning only the
   branch term for `dx` and forgetting to add `dY` from the shortcut.
4. **`gradient_norm_through_stack(depth, residual)`.** Uses $d = 16$ and
   $\rho = 0.1$. Seed `np.random.default_rng(0)`. For each layer, take the Q
   factor from `np.linalg.qr` of a Gaussian `(d, d)` matrix, and multiply it by
   $\rho$. Start from `g = np.ones(d) / np.sqrt(d)`, loop over the layers in
   reverse, update `g`, and return `float(np.linalg.norm(g))`. Common mistake:
   using a plain Gaussian matrix instead of an orthogonal one, which makes the
   plain stack's shrinkage unpredictable.
5. **`bottleneck_flops(c_in, c_mid, c_out, h, w)`.** Returns an `int` from the
   formula above. Common mistake: forgetting the factor 9 on the middle
   convolution.

## Cost and scaling

The shortcut itself is nearly free: one addition per number. The table
compares the blocks for $C$ channels at $H \times W$.

| Block | Weights (ignoring biases) | Multiply-adds |
|---|---|---|
| Basic, two $3\times3$ at width $C$ | $18 C^2$ | $18 H W C^2$ |
| Bottleneck, $C \to C/4 \to C/4 \to C$ | $\tfrac{17}{16} C^2$ | $\tfrac{17}{16} H W C^2$ |
| Projection shortcut, $1\times1$ | $C_{\text{in}} C_{\text{out}}$ | $H W C_{\text{in}} C_{\text{out}}$ |

Memory is the real limit. Backpropagation needs every block's activations, so
memory grows in a straight line with depth. Doubling the number of blocks
doubles the memory, however cheap each block is.

## Where this breaks

**Shape changes need a projection.** You can only add $x$ and $F(x)$ when they
have the same shape. When a stage changes the number of channels or the image
size, the shortcut becomes a learned $1\times1$ convolution. That path is no
longer a clean copy. Networks keep it to one block per stage.

**Zero start isn't permanent.** A zero $\gamma$ makes the network the identity
only at step zero. Training moves it away. Networks without batch norm need
other initialisation tricks to get the same start.

**Depth still runs out.** Past a few hundred layers, adding blocks stops
helping. People then add width or data instead.

**A clean shortcut doesn't fix a bad branch.** The shortcut keeps the gradient
alive. It says nothing about whether the branch itself trains well. The branch
still needs good initialisation and normalisation.

## The exercise

You build `residual_block_forward` and `residual_block_backward` for a block
whose branch is two linear layers with a ReLU. You then build
`gradient_norm_through_stack` to measure the gradient length after 10 and 50
layers, with and without the shortcut. Last, you write `bottleneck_flops`.

The checks compare your forward pass with the worked example, confirm a zeroed
branch gives an exact identity, and compare your backward pass with finite
differences. They confirm the residual gradient at depth 50 stays between 0.1
and 10 while the plain one falls below $10^{-6}$, and check the flop count and
the more-than-fifteen-times saving. The `__main__` block trains a 20-block
residual network on synthetic data and prints its loss.
