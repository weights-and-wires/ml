---
title: Residual networks
slug: 21-residual-networks
part: "Part 4 — Vision and sequences"
summary: "The shortcut connection that keeps gradients alive at any depth, and why the bottleneck block made deep networks affordable."
minutes: 45
gpu: true
objectives:
  - State the degradation problem precisely enough to rule out overfitting and capacity as its cause.
  - Explain why the shortcut in a residual block gives the gradient an identity term that survives any depth.
  - Explain why pre-activation ordering and a zero-initialized final batch-norm scale make that identity path exact.
  - Compute the flop cost of a bottleneck block and compare it with a stack of full-width convolutions.
prereqs:
  - 20-convolutions
  - 18-initialisation-and-norm
exercise: 21-residual-networks
---

# Residual networks

Stack enough plain convolutional layers and training error goes up, not down.
He et al. reported a 56-layer plain network reaching higher *training* error
than its 20-layer counterpart on CIFAR-10, using the same optimizer, the same
number of steps, and the same data. That rules out overfitting, because
overfitting means training error keeps falling while held-out error rises.
It also rules out a capacity argument: a 56-layer network can copy the
20-layer solution and set the extra 36 layers to the identity map, so it can
represent everything the shallow network can and more. If a strictly more
expressive model trained the same way reaches worse training loss, the
failure sits in the optimization landscape, not in what the function class
can express. Depth itself was making the identity map hard for gradient
descent to find.

This is the **degradation problem**. The fix is the residual block: instead
of asking a stack of nonlinear layers to learn the identity when that's the
useful thing to do, make the identity free, and let every layer add a small
correction on top of it. The result this lesson builds toward is one line:
the gradient of a residual stack always contains an exact, unattenuated
identity term, no matter how many blocks you stack. A plain stack has no such
floor.

## What you already know

- The chain rule for a composition of vector-valued functions is a product of
  Jacobians, and reverse-mode automatic differentiation computes it as a
  sequence of vector-Jacobian products from the output back to the input.
- Batch normalization standardizes its input per channel, then applies a
  learned affine map $\gamma \odot \hat{x} + \beta$. At $\gamma = 0$, that map
  is identically zero.
- Vanishing and exploding gradients come from repeatedly multiplying by
  Jacobians whose norm stays below or above one across many layers.
- A convolution's flop cost scales as
  $H_{\text{out}}W_{\text{out}}C_{\text{out}}C_{\text{in}}k^2$.

## The residual block

A residual block computes

$$
y = x + F(x) ,
$$

where $F$, the **residual branch**, is one or more convolution and
nonlinearity stages that produce an output the same shape as $x$, so the
addition is defined elementwise. Setting $F \equiv 0$ makes the block compute
$y = x$ exactly, with no layer having to learn a single weight to reach it.
Compare that with a plain layer $y = F(x)$: reaching the identity there means
finding weights that make a nonlinear function behave like the identity map,
which the degradation problem shows is hard once the network gets deep.

## Why gradients survive depth

Stack $L$ residual blocks, $x_l = x_{l-1} + F_l(x_{l-1})$, and let $J_l =
\partial F_l(x_{l-1}) / \partial x_{l-1}$ be the Jacobian of the $l$-th
branch. Differentiating the recursion gives

$$
\frac{\partial x_l}{\partial x_{l-1}} = I + J_l .
$$

That identity matrix is the whole result: every step's local derivative
contains a term equal to $I$, regardless of what the branch computes. Chain
two blocks together and multiply out the product:

$$
(I + J_2)(I + J_1) = I + J_1 + J_2 + J_2 J_1 .
$$

The bare $I$ term never disappears, however small or large $J_1$ and $J_2$
are. The same thing happens at any depth: multiplying out $L$ factors of
$(I + J_l)$ always leaves a lone $I$ term standing alongside every product of
one or more Jacobians. That term is the **identity path** — an exact copy of
the loss gradient at the top of the stack, reaching the bottom no matter what
the branches do, even if every $J_l$ is singular.

A plain stack has no such term. Its gradient is a single product, $J_L
J_{L-1} \cdots J_1$, with nothing added, so if every $\lVert J_l \rVert \le
\rho < 1$ the whole product is bounded by $\rho^L$ and decays geometrically
with depth. The residual sum doesn't stop the higher-order terms from
vanishing or growing, but their sum sits on top of a term of size exactly
$\lVert \partial\ell/\partial x_L \rVert$, so the total can't fall below what
that floor permits. Near initialization the branches start small, so a
residual stack's gradient starts close to that floor and only moves away from
it as training makes the branches contribute something real.

## Making the identity path exact

Two design choices decide whether that identity term is a literal $I$ or
something weaker.

**Ordering.** The original design applies a nonlinearity *after* the
addition, so the next block receives $\operatorname{ReLU}(x_{l-1} +
F_l(x_{l-1}))$, not $x_{l-1} + F_l(x_{l-1})$ directly. That ReLU multiplies
the next step's Jacobian by a 0/1 mask, and masks compound across blocks the
same way any other repeated factor does. Pre-activation moves the
normalization and nonlinearity to the front of $F_l$ instead, so nothing sits
on the route between one block's sum and the next block's input. The
shortcut then carries a literal, unmodified $I$ all the way from the bottom
of the stack to the top.

**Initialization.** A branch that ends in batch norm computes $\gamma_l
\odot \hat{h} + \beta_l$ as its last step. Setting $\gamma_l = 0$ makes
$F_l(x) \equiv 0$ exactly, for every input, on the first forward pass, so the
entire stack starts as the identity function regardless of depth. Training
then has to move $\gamma_l$ away from zero before the branch contributes
anything, which is a gentler starting point than asking every weight in
$F_l$ to be well scaled from the start.

## The bottleneck block

A **bottleneck** replaces two full-width $3\times3$ convolutions with a
$1\times1$ reduction to $C_{\text{mid}}$ channels, a $3\times3$ convolution at
that reduced width, and a $1\times1$ expansion back to $C_{\text{out}}$
channels. For an output of spatial size $H \times W$, its multiply-add cost is

$$
\text{flops}_{\text{bottleneck}} = HW\big(C_{\text{in}}C_{\text{mid}} +
9\,C_{\text{mid}}^2 + C_{\text{mid}}C_{\text{out}}\big) .
$$

With the standard choice $C_{\text{mid}} = C/4$ and $C_{\text{in}} =
C_{\text{out}} = C$, that works out to $\tfrac{17}{16} HWC^2$, against
$18HWC^2$ for two full-width $3\times3$ convolutions on the same channel
count — more than a fifteenfold reduction, for a block with one extra layer
and the same input and output shape. That's why every ResNet deep enough to
matter, 50 layers and up, uses bottlenecks instead of full-width
convolutions.

## Cost and scaling

The identity shortcut needs $x_{l-1}$ and $F_l(x_{l-1})$ to share a shape.
Whenever a stage changes spatial resolution or channel count, the shortcut is
replaced by a $1\times1$ convolution with matching stride — a **projection
shortcut** — which is a learned map, not the identity, so the argument above
applies only to the plain shortcuts inside a stage.

| Component | Params | Multiply-adds |
|---|---|---|
| Basic block (two $3\times3$, $C\!\to\!C$) | $2(9C^2 + C)$ | $18HWC^2$ |
| Bottleneck ($C\!\to\!C/4\!\to\!C/4\!\to\!C$) | $\tfrac{17}{16}C^2 + 2C_{\text{mid}} + C$ | $\tfrac{17}{16}HWC^2$ |
| Projection shortcut ($1\times1$, stride $s$) | $C_{\text{in}}C_{\text{out}}$ | $\frac{H}{s}\frac{W}{s}C_{\text{in}}C_{\text{out}}$ |

Backward-pass memory doesn't benefit from any of this: every activation
needed for the backward pass of every block, residual or not, has to stay in
memory until its gradient is computed. Memory scales linearly with depth
regardless of how cheap each block is to compute, and it's what limits how
many blocks fit on a fixed accelerator.

## Where this breaks

**Shape changes break the exact identity.** The gradient argument depends on
the shortcut being a literal identity. A projection shortcut is a learned
linear map with no guaranteed floor on its Jacobian, so networks keep every
stage's internal blocks on plain shortcuts and pay the projection cost only
once per stage, where resolution or width actually changes.

**Zero initialization is a starting point, not a lasting guarantee.**
Zero-initializing the last batch-norm scale gives an exact identity at step
zero. Nothing stops later gradient steps from moving every branch far from
zero, which is why normalization-free architectures that drop batch norm
need more delicate initializations (Fixup, SkipInit) to reconstruct the same
starting point without an affine parameter that's exactly zero.

**Depth still saturates.** Beyond a few hundred layers, accuracy gains from
adding blocks flatten and can reverse. Practitioners reach for width, better
data, or a different architecture, not more residual blocks, once that
happens.

**A clean path through the stack doesn't fix a badly conditioned branch.**
The identity term guarantees the sum doesn't vanish; it says nothing about
whether $F_l$ trains well internally. A branch with its own exploding
activation still needs the initialization and normalization tools from the
earlier lesson applied to its own layers.

## The exercise

You implement a residual block's forward and backward pass by hand, then
measure the gradient-flow claim this lesson derived instead of taking it on
faith.

Write `residual_block_forward(x, params)` and `residual_block_backward(dY, x,
params, cache)` for a block whose branch is two small linear-and-ReLU
stages, small enough to run on a CPU. Then write
`gradient_norm_through_stack(depth, residual)`, which builds a stack of the
given depth — with or without the residual addition, controlled by the
`residual` flag — and returns the norm of the gradient that reaches the input
from a fixed upstream gradient at the output. Write `bottleneck_flops(c_in,
c_mid, c_out, h, w)`, returning the multiply-add count derived above.

The checks verify the backward pass against `mlcheck.finite_difference` on a
small block, confirm that a zero-initialized residual branch makes the block
an exact identity to floating-point precision, confirm that
`gradient_norm_through_stack` at depth 50 keeps the residual stack's gradient
within an order of magnitude of one while the plain stack's collapses by many
orders of magnitude, and check `bottleneck_flops` against the formula above.
The `__main__` block trains a small residual network on synthetic data and
prints its loss curve, sized to run quickly even on the CPU the automated
checks use.
