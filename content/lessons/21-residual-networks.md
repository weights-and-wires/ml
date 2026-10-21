---
title: Residual networks
slug: 21-residual-networks
part: "Part 4 — Vision and sequences"
summary: "Add a shortcut around each block so a deep stack can train, and count the cost of a bottleneck."
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

A 56-layer plain network reaches a higher training error than a 20-layer one, on the same data and the same optimiser. That is not overfitting. Overfitting means the training error is low and the error on new data is high. Here the training error itself is worse.

The deep stack could copy the shallow one and leave the extra layers doing nothing. Gradient descent does not find that setting. This is the degradation problem.

The fix adds each block's input back onto its output, so the block only adds a correction. You build that block, measure the gradient that survives 50 layers, and count the cost of a bottleneck.

## What you already know

You already have the following:

- Backpropagation applies the chain rule from the loss back to the input.
- A linear layer computes `x @ W.T + b`, and ReLU sets negatives to zero.
- A long product of numbers smaller than one shrinks toward zero.
- Batch norm ends with a learned scale and shift on each channel.
- A convolution's cost grows with height, width, channels, and kernel area.

## Keep the essay and write in the margin

Keep the essay, and write edits in the margin. No edits, and the essay is unchanged.

Take two examples with two features, and these weights:

$$
x = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}, \quad
W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
b_1 = [0, -1]
$$

$W_1$ is the identity, so the first layer only subtracts 1 from the second column. That gives $h_1 = \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$.

The block that does this is a **residual block**. The output is the input plus a correction:

$$
y = x + F(x)
$$

$F$ is the residual branch. The plain path into the plus sign is the shortcut. Here the branch is two linear layers with a ReLU between them, and no ReLU after the addition:

$$
h_1 = x W_1^{\top} + b_1, \qquad a_1 = \operatorname{ReLU}(h_1), \qquad
h_2 = a_1 W_2^{\top} + b_2, \qquad y = x + h_2
$$

Both weights are `(d, d)` and both biases are `(d,)`, with `x` shaped `(n, d)`.

Continue the same numbers. $W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ and $b_2 = [0.1, 0]$. ReLU turns the $-1$ into 0, so $a_1 = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$. The second layer gives $h_2 = \begin{bmatrix} 2.1 & 0.5 \\ 0.1 & 1.0 \end{bmatrix}$. Adding the input back gives $y = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.

The forward lines use a transpose on the weights, because each example is a row:

```python
h1 = x @ W1.T + b1
a1 = np.maximum(h1, 0.0)
h2 = a1 @ W2.T + b2
y = x + h2
```

The usual mistake is `W1 @ x`, which fails when the shapes are not square, or a ReLU after the addition.

Set $W_2$ and $b_2$ to zero and $h_2$ is zero, so $y = x$. A plain layer $y = F(x)$ has no such setting. Its weights would have to copy the input.

> [!KEY]
> A residual block outputs a correction. A zero branch passes the input through unchanged.

## The plus sign copies the gradient

The loss sends back `dY`, shape `(n, d)`. A plus sign passes that gradient down both paths. The shortcut contributes `dY` itself. The branch contributes whatever comes back through the two layers. You add those two pieces.

Walk the branch backward. On the example above, take `dY` as all ones. Then $dW_2 = dY^\top a_1 = \begin{bmatrix} 1 & 3 \\ 1 & 3 \end{bmatrix}$.

The three lines of the backward pass are:

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

A weight gradient is the upstream gradient, transposed, times that layer's input. A bias gradient sums over the $n$ rows. ReLU passes gradient only where its input was positive. $\odot$ is entrywise multiplication, and $[h_1 > 0]$ is a mask of ones and zeros. Because $W_1$ is the identity, $dh_1 = \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix}$ and $dx = \begin{bmatrix} 3 & 1.5 \\ 1 & 1.5 \end{bmatrix}$. The ones from the shortcut are sitting inside $dx$.

The forward pass multiplies by `W.T`. The backward pass multiplies by `W` with no extra transpose:

```python
dW2 = dY.T @ a1
dh1 = (dY @ W2) * (h1 > 0)
dx = dY + dh1 @ W1
```

The usual mistake is to multiply by `W1.T` on the way back. Square weights still run, and the numbers are wrong.

> [!WARNING]
> Forward uses `W.T`. Backward multiplies by `W` itself. The wrong choice still runs on square weights.

> [!KEY]
> A plus sign copies the incoming gradient onto both inputs. `dx` is `dY` plus the branch term.

> [!TIP]
> After a backward pass, check `grads[k].shape == params[k].shape` for every key.

## A message passed down fifty people

Picture 50 people passing a message. A plain line repeats a tenth each time, so $0.1^{50}$ is left. A residual line repeats the whole message and adds a short note.

If a plain layer multiplies the gradient by a number $J$, a residual layer multiplies it by $1 + J$. The 1 is the shortcut. Fifty layers give

$$
\text{plain: } J^{50}, \qquad \text{residual: } (1 + J)^{50}
$$

The plain product collapses. The residual product does not have to pass through a chain of factors below one. That scalar picture assumes each $J$ points the same way as the gradient. The exercise is the matrix version, which does not.

Each weight is $0.1$ times a random orthogonal matrix, a matrix that rotates a vector and keeps its length. A plain layer then shortens the gradient by exactly $0.1$. A residual layer does $g + gW$, not $(1 + 0.1)g$, because $gW$ points somewhere new. From a starting length of 1, the lengths come out as follows:

| Depth | Residual length | Plain length |
|---|---|---|
| 10 | about 0.93 | $10^{-10}$ |
| 50 | about 0.53 | $10^{-50}$ |

Use width 16, scale $0.1$, and seed 0. Start from `np.ones(d) / np.sqrt(d)`. Take Q from `np.linalg.qr`, then multiply by $0.1$:

```python
g = g + g @ W  # residual layer
g = g @ W      # plain layer
```

The usual mistake is a plain Gaussian matrix instead of an orthogonal one. The plain stack's shrink is then no longer exactly $0.1$ per layer.

> [!KEY]
> The shortcut adds 1 to each layer's gradient factor, so the gradient does not have to survive a long product of small numbers.

Put batch norm and ReLU inside the branch, before the addition. That is pre-activation, so nothing sits on the shortcut. Start the last batch-norm scale at zero and every branch outputs zero at step zero. The network starts as the identity. Training moves it off that start.

## Do the wide work at a narrow width

Two $3 \times 3$ convolutions at full width $C$ get expensive. A **bottleneck** does three cheaper steps. A $1 \times 1$ shrinks $C_{\text{in}}$ channels to $C_{\text{mid}}$. A $3 \times 3$ works at that narrow width. A $1 \times 1$ grows back to $C_{\text{out}}$.

Each convolution costs height times width times input channels times output channels times kernel area. Add the three:

$$
\text{flops} = H W \left( C_{\text{in}} C_{\text{mid}} + 9\, C_{\text{mid}}^2 + C_{\text{mid}} C_{\text{out}} \right)
$$

The 9 is the area of the $3 \times 3$ kernel. The shortcut is not in this count.

Try $C_{\text{in}} = C_{\text{out}} = 256$, $C_{\text{mid}} = 64$, at $14 \times 14$. The shrink is $256 \times 64 = 16{,}384$. The middle is $9 \times 64 \times 64 = 36{,}864$. The grow is $64 \times 256 = 16{,}384$. Per position that is $69{,}632$. Times $196$ positions that is $13{,}647{,}872$.

Two full-width $3 \times 3$ convolutions cost $196 \times 2 \times 9 \times 256 \times 256 = 231{,}211{,}008$, about 17 times more. With $C_{\text{mid}} = C/4$, the weight ratio is $17/16$ against $18$:

```python
flops = h * w * (c_in * c_mid + 9 * c_mid * c_mid + c_mid * c_out)
```

The usual mistake is to forget the 9 on the middle convolution.

> [!KEY]
> A bottleneck does its $3 \times 3$ work at a quarter of the width, about 17 times cheaper than two full-width $3 \times 3$ layers.

> [!RECAP]
> Hold these five points before you write code:
>
> - Forward is `h1 = x @ W1.T + b1`, a ReLU, `h2 = a1 @ W2.T + b2`, and `y = x + h2`.
> - The cache needs `x`, `W1`, `W2`, `h1`, and `a1`.
> - Weight gradients are upstream, transposed, times the layer input. Bias gradients sum over rows. The ReLU mask is `h1 > 0`.
> - `dx` is `dY + dh1 @ W1`.
> - Bottleneck cost is $HW(C_{\text{in}} C_{\text{mid}} + 9 C_{\text{mid}}^2 + C_{\text{mid}} C_{\text{out}})$.

## Build it step by step

The checks call the functions in this order:

1. **`residual_block_forward(x, params)`.** `x` is `(n, d)`. `params` holds `W1`, `b1`, `W2`, `b2`. Return `(y, cache)` with `y` shaped `(n, d)`. Use `@`, `.T`, and `np.maximum(h1, 0.0)`. Cache at least `x`, `W1`, `W2`, `h1`, and `a1`. The usual mistake is `W1 @ x`, or a ReLU after the sum. With `W2` and `b2` at zero, `y` must equal `x`.
2. **`residual_block_backward(dY, cache)`.** `dY` is `(n, d)`. Return `(dx, grads)` with keys `W1`, `b1`, `W2`, `b2`. Use `.T @`, `.sum(axis=0)` for biases, and `(h1 > 0)` for the ReLU. The usual mistake is returning only the branch term and forgetting to add `dY`.
3. **`gradient_norm_through_stack(depth, residual)`.** Use width 16 and scale $0.1$. Seed `np.random.default_rng(0)`. Build each layer as $0.1$ times the Q factor of `np.linalg.qr`. Start from `np.ones(d) / np.sqrt(d)`, walk the layers in reverse, and return `float(np.linalg.norm(g))`. The usual mistake is a Gaussian matrix that is not orthogonal.
4. **`bottleneck_flops(c_in, c_mid, c_out, h, w)`.** Return an `int` from the formula above. The usual mistake is dropping the factor 9.

## Cost and scaling

The shortcut is one addition per number. For $C$ channels at height $H$ and width $W$, the blocks compare as follows:

| Block | Weights, ignoring biases | Multiply-adds |
|---|---|---|
| Basic, two $3 \times 3$ at width $C$ | $18 C^2$ | $18 H W C^2$ |
| Bottleneck, $C \to C/4 \to C/4 \to C$ | $17/16\, C^2$ | $17/16\, H W C^2$ |
| Projection shortcut, $1 \times 1$ | $C_{\text{in}} C_{\text{out}}$ | $H W C_{\text{in}} C_{\text{out}}$ |

Memory is the limit. Backpropagation stores every block's activations, so memory grows in a straight line with depth. Twice the blocks means twice the memory, however cheap each block is.

## Where this breaks

Each of these failures has a direct change:

- You can add $x$ and $F(x)$ only when the shapes match. A stage that changes channels or image size puts a learned $1 \times 1$ on the shortcut. That path is no longer a clean copy. Keep it to one block per stage.
- A zero scale makes the network the identity only at step zero. Training moves it. Networks without batch norm need another initialisation to get the same start.
- Past a few hundred layers, more blocks stop helping. Add width or data instead.
- A clean shortcut keeps the gradient alive. It does not make a badly scaled branch train. The branch still needs a good initialisation and a norm.

## The exercise

You build `residual_block_forward` and `residual_block_backward` for two linear layers with a ReLU between them. You build `gradient_norm_through_stack` for depths 10 and 50, with and without the shortcut. You write `bottleneck_flops`.

The checks match the forward pass to the worked numbers, and a zero branch to the identity. They match the backward pass to finite differences. At depth 50 the residual gradient stays between 0.1 and 10, and the plain one falls below $10^{-6}$. The gap widens from depth 10 to depth 50. The flop count matches, and the bottleneck is more than fifteen times cheaper.
