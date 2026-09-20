---
title: Residual networks
slug: 21-residual-networks
part: "Part 4 — Vision and sequences"
summary: The additive identity term that keeps gradients alive at depth 100, and why residual networks are still the baseline every architecture is measured against.
minutes: 55
gpu: true
objectives:
  - State the degradation problem precisely enough to rule out overfitting as its cause.
  - Derive the additive identity term in the gradient of a residual stack, and explain why it prevents total vanishing at any depth.
  - Explain why pre-activation ordering and a zero-initialised final batch-norm scale make the identity path exact rather than approximate.
  - Account for the flop cost of a bottleneck block and compare it against the equivalent stack of full-width convolutions.
prereqs:
  - 20-convolutions
  - 18-initialisation-and-norm
exercise: 21-residual-networks
---

# Residual networks

Stack enough plain convolutional layers and training error goes up, not down.
He et al. reported a 56-layer plain network reaching higher *training* error
than its 20-layer counterpart on CIFAR-10 — not held-out error, training
error, on the same optimiser, the same number of steps, the same data. That
single fact rules out overfitting as the explanation, because overfitting is
by definition a training error that keeps falling while held-out error rises.
It also rules out the naive appeal to capacity: a 56-layer network can
represent everything a 20-layer network can, by copying the shallow solution
and setting the extra 36 layers to the identity map. If a strictly more
expressive model trained with the same procedure reaches worse training loss,
the failure is in the optimisation landscape, not in what the function class
can express. Something about depth itself was making the identity map, and
everything near it, hard for gradient descent to reach.

This is the **degradation problem**, and the residual block is the fix: don't
ask a stack of nonlinear layers to learn the identity when useful, force the
identity to be free, and let every layer be a small correction on top of it.
Everything else in this lesson is one derivation and its consequences — the
gradient of a residual stack contains an exact, unattenuated identity term
that no amount of depth can remove.

## What you already know

- Chain rule through a composition of vector-valued functions is a product of
  Jacobians, and reverse-mode automatic differentiation computes it as a
  sequence of vector-Jacobian products from the output back to the input.
- Batch normalisation standardises its input to zero mean and unit variance
  per channel, then applies a learned affine map $\gamma \odot \hat{x} +
  \beta$; at $\gamma = 0$ that affine map is identically zero.
  Vanishing and exploding gradients arise from repeated multiplication by
  Jacobians whose spectral norm is persistently below or above one.
- Xavier and He initialisation choose weight variances so a plain layer's
  forward and backward signal neither shrinks nor grows in expectation, layer
  by layer, but say nothing about what happens when the number of layers runs
  into the hundreds.
- A convolution's parameter and flop cost scale with kernel size and channel
  count as $C_{\text{out}}(C_{\text{in}}k^2 + 1)$ and
  $H_{\text{out}}W_{\text{out}}C_{\text{out}}C_{\text{in}}k^2$ respectively.

## The residual block

A residual block computes

$$
y = x + F(x) ,
$$

where $F$, the **residual branch**, is one or more convolution–normalisation–
nonlinearity stages with the same output shape as its input, so the addition
is defined elementwise. Nothing about this forces $F$ to do anything: at
$F \equiv 0$, the block computes $y = x$ exactly, for free, with no layer
having to learn a single weight to reach it. Contrast a plain layer $y =
F(x)$, where reaching the identity means solving for weights that make a
generally nonlinear $F$ behave like the identity map on the data distribution
— possible in principle, apparently hard in practice at depth, since that is
what the degradation problem reports.

## The gradient of a residual stack

Stack $L$ residual blocks, $x_l = x_{l-1} + F_l(x_{l-1})$ for $l = 1,
\dots, L$, and let $\ell$ denote the scalar loss at the end of the stack.
Write $J_l = \partial F_l(x_{l-1}) / \partial x_{l-1}$ for the Jacobian of the
$l$-th residual branch, evaluated at the forward activations. Differentiating
the recursion,

$$
\frac{\partial x_l}{\partial x_{l-1}} = I + J_l ,
$$

so the chain rule back to the input of the stack is a product of these
Jacobians:

$$
\frac{\partial \ell}{\partial x_0} = \frac{\partial \ell}{\partial x_L}
\prod_{l=L}^{1} (I + J_l) .
$$

Expand the product. Each factor contributes either $I$ or $J_l$ to a term, so
multiplying out $\prod_{l=1}^{L}(I + J_l)$ produces one term for every subset
$S \subseteq \{1, \dots, L\}$ — a sum of $2^L$ terms, with the factors in each
term kept in their original order:

$$
\prod_{l=1}^{L} (I + J_l) = I + \sum_{l=1}^{L} J_l + \sum_{l < m} J_m J_l +
\cdots + J_L J_{L-1} \cdots J_1 .
$$

The subset $S = \emptyset$ contributes exactly $I$, so

$$
\frac{\partial \ell}{\partial x_0} = \frac{\partial \ell}{\partial x_L} +
\frac{\partial \ell}{\partial x_L}\left( \sum_l J_l + \sum_{l<m} J_m J_l +
\cdots \right) .
$$

The first term is the **pure identity path**: an exact, unattenuated copy of
the loss gradient at the top of the stack, reaching the bottom regardless of
what any $J_l$ is. It survives even if every $J_l$ is singular, or has
spectral norm zero, or has spectral norm ten — the identity term never
depends on the branches at all. A plain stack has no such term: its gradient
is $\partial\ell/\partial x_0 = (\partial\ell/\partial x_L) J_L J_{L-1} \cdots
J_1$, a single product with nothing added to it, so if every $\lVert J_l
\rVert \le \rho < 1$ the product is bounded by $\rho^L$ and decays
geometrically in depth; if every $\lVert J_l \rVert \ge \rho > 1$ it grows the
same way. The residual sum does not stop the higher-order terms from
vanishing or exploding — but their sum is added to a term of magnitude
exactly $\lVert \partial\ell/\partial x_L \rVert$, so the total can never fall
below what cancellation among the remaining $2^L - 1$ terms permits, and
generically does not.

Near initialisation the branches are small: with $\lVert J_l \rVert \approx
\varepsilon$ for every $l$, a term built from $k$ of the $J_l$'s is $O(\varepsilon^k)$,
so the $\binom{L}{1} = L$ single-branch terms dominate every higher-order
term once $\varepsilon$ is small enough that $L\varepsilon^2 \ll 1$, and

$$
\frac{\partial \ell}{\partial x_0} \approx \frac{\partial \ell}{\partial x_L}
\left( I + \sum_{l=1}^{L} J_l \right) ,
$$

a gradient within $O(L\varepsilon)$ of the identity term itself. This is the
precise sense in which a 100-layer residual stack keeps gradients within an
order of magnitude of their value at the top, while a 100-layer plain stack
with the same per-layer Jacobian norms collapses or blows up exponentially:
the residual sum has an $O(1)$ floor that never depends on $L$, and the plain
product has none.

## Pre-activation: making the identity path exactly clean

The block above hides an ordering choice: where do normalisation and the
nonlinearity sit? The original design computes $y_l = x_{l-1} + F_l(x_{l-1})$
with $F_l$ ending in batch norm, then applies the nonlinearity *after* the
addition, so the next block actually receives $x_l = \operatorname{ReLU}(y_l)$.
The shortcut addition is still exact, but the value handed to the next layer
is not $x_{l-1}$ plus a correction — it is a ReLU of that sum, so
$\partial x_l/\partial x_{l-1}$ picks up a further multiplication by
$\operatorname{diag}(\mathbb{1}[y_l > 0])$, a 0/1 mask. The identity term
derived above still exists inside $\partial y_l/\partial x_{l-1}$, but by the
time the signal reaches $x_l$ it has been gated, and gating compounds across
$L$ blocks exactly like any other repeated Jacobian.

Pre-activation moves batch norm and the nonlinearity to the front of $F_l$:
$F_l(x) = \text{Conv}(\operatorname{ReLU}(\text{BN}(\text{Conv}(\operatorname{ReLU}(\text{BN}(x))))))$,
and applies nothing to the sum $x_{l-1} + F_l(x_{l-1})$ before passing it on
as $x_l$. Now $x_l$ *is* $x_{l-1}$ plus a correction, with no function of any
kind interposed on the shortcut route between consecutive blocks. The
identity term in the gradient derivation above is not merely present inside
one step — it is a literal, unmodified $I$ connecting the input of the stack
to its output, because the forward recursion itself contains an unmodified
copy of $x_{l-1}$ at every step. This is why pre-activation ordering, not the
residual addition alone, is what the gradient argument needs to hold exactly
rather than up to a per-layer gating factor.

## Zero-initialising the last batch norm

A residual branch $F_l$ that ends in batch norm computes
$\gamma_l \odot \hat{h} + \beta_l$ as its last step, where $\hat{h}$ is
already normalised to zero mean and unit variance per channel. Initialising
$\gamma_l = 0$ makes $F_l(x) \equiv 0$ exactly, for every input, on the first
forward pass — not approximately small, as ordinary weight initialisation
gives you, but identically zero. The block starts as an exact identity map,
so the whole stack of $L$ blocks starts as the identity function on its
input, however large $L$ is. Training then has to move $\gamma_l$ away from
zero before the branch contributes anything, which is a gentler starting
point than asking every weight in $F_l$ to be jointly well-scaled: the
network begins at the one point in parameter space where the degradation
problem cannot appear, because there is nothing yet for depth to degrade.

## The ensemble-of-shallow-paths reading

The subset expansion of $\prod_l (I + J_l)$ has a forward-pass mirror. Because
$x_l = x_{l-1} + F_l(x_{l-1})$, unrolling the recursion gives an explicit sum,
$x_L = x_0 + \sum_{l=1}^L F_l(x_{l-1})$, rather than the nested composition
$F_L(F_{L-1}(\cdots F_1(x_0)))$ a plain stack computes. Combined with the
$2^L$-term gradient expansion, this is Veit et al.'s **unravelled view**: a
residual network of depth $L$ behaves like an implicit ensemble of $2^L$
paths of length $0$ to $L$, one path for every subset of blocks whose branch
is used rather than skipped, and the gradient of the whole network is the sum
of the gradients along every path. Path lengths concentrate near $L/2$ by the
binomial distribution over subset sizes, and the few paths of length close to
$L$ carry a vanishingly small share of the gradient, because they are
built from the most factors. This is why deleting a single block from a
trained ResNet barely changes its accuracy — a property plain networks do
not share, because removing one layer from a nested composition removes the
only path through it — and why "how deep is this network really" is, for a
ResNet, a question about the effective depth of the dominant paths, not
about $L$.

## The bottleneck block

A **bottleneck** replaces two full-width $3\times3$ convolutions with a
$1\times1$ reduction to $C_{\text{mid}}$ channels, a $3\times3$ convolution at
that reduced width, and a $1\times1$ expansion back to $C_{\text{out}}$
channels — trading one extra layer for a much smaller channel count on the
expensive spatial convolution. For an output of spatial size $H \times W$,

$$
\text{flops}_{\text{bottleneck}} = HW\big(C_{\text{in}}C_{\text{mid}} +
9\,C_{\text{mid}}^2 + C_{\text{mid}}C_{\text{out}}\big) .
$$

With the standard choice $C_{\text{mid}} = C_{\text{out}}/4$ and
$C_{\text{in}} = C_{\text{out}} = C$, this is $HW C^2 (\tfrac14 + \tfrac{9}{16}
+ \tfrac14) = \tfrac{17}{16} HW C^2$ — against $2 \cdot 9 HWC^2 = 18HWC^2$ for
two full-width $3\times3$ convolutions on the same channel count, a
$\tfrac{17/16}{18} \approx 0.059$ ratio, better than a seventeenfold
reduction, for a block with one more layer and the same input and output
shape. This is why every ResNet deep enough to matter — 50 layers and up —
uses bottlenecks rather than stacking full-width convolutions.

## Cost and scaling

The identity shortcut needs $x_{l-1}$ and $F_l(x_{l-1})$ to share a shape.
Whenever a stage changes spatial resolution or channel count, the shortcut is
replaced by a $1\times1$ convolution with matching stride — a **projection
shortcut** — which has its own flop and parameter cost and, critically, is
not the identity: the additive-identity argument above applies only to the
blocks within a stage, where the plain shortcut is used.

| Component | Params | Multiply-adds |
|---|---|---|
| Basic block (two $3\times3$, $C\!\to\!C$) | $2(9C^2 + C)$ | $18HWC^2$ |
| Bottleneck ($C\!\to\!C/4\!\to\!C/4\!\to\!C$) | $\tfrac{17}{16}C^2 + 2C_{\text{mid}} + C$ | $\tfrac{17}{16}HWC^2$ |
| Projection shortcut ($1\times1$, stride $s$) | $C_{\text{in}}C_{\text{out}}$ | $\frac{H}{s}\frac{W}{s}C_{\text{in}}C_{\text{out}}$ |

Backward-pass memory is unaffected by any of this: every activation needed
for the backward pass of every block, residual or not, has to be kept until
its gradient is computed, so memory still scales linearly in depth, and it is
what limits how many blocks fit in a fixed accelerator regardless of how
cheap each one is to compute.

## Where this breaks

**Shape changes break the exact identity.** The gradient argument depends on
the shortcut being a literal identity; a projection shortcut is a learned
linear map like any other layer, with no guaranteed floor on its Jacobian.
Networks manage this by keeping every stage's internal blocks on plain
shortcuts and paying the projection cost only once per stage, where it
changes resolution or width.

**The clean-identity argument is an initialisation and training-dynamics
claim, not a static guarantee.** Zero-initialising the last batch norm scale
gives you an exact identity at step zero; nothing stops later gradient steps
from moving every branch far from zero and reintroducing a landscape closer
to a plain network's. Normalisation-free residual architectures that drop
batch norm entirely need alternative, more delicate initialisations (Fixup,
SkipInit) to reconstruct the same near-identity starting point, because they
no longer have an affine parameter that is exactly zero by construction.

**Depth still saturates.** Beyond a few hundred layers, accuracy gains from
adding more blocks flatten and can reverse, because the ensemble of paths
described above is already wide enough that additional blocks mostly add
redundant, rarely-used paths rather than new effective capacity. The fix
practitioners reach for is width, better data, or a different architecture,
not more residual blocks.

**A clean path through the stack does not fix a badly conditioned branch.**
The identity term guarantees the *sum* does not vanish; it says nothing about
whether $F_l$ itself trains well internally. A branch with its own vanishing-
gradient pathology, or an exploding activation from a badly scaled
convolution, still needs the initialisation and normalisation lesson's tools
applied to its own layers.

## The exercise

You implement a residual block's forward and backward pass by hand, then
measure the gradient-flow claim this lesson derived directly, rather than
taking it on faith.

Write `residual_block_forward(x, params)` and `residual_block_backward(dY, x,
params, cache)` for a block whose branch is two small linear-and-ReLU
stages, so the whole derivation stays checkable on tensors small enough to
run on a CPU. Then write `gradient_norm_through_stack(depth, residual)`,
which builds a stack of the given depth — with or without the residual
addition, controlled by the `residual` flag — and returns the norm of the
gradient that reaches the input from a fixed upstream gradient at the output,
and `bottleneck_flops(c_in, c_mid, c_out, h, w)`, returning the multiply-add
count derived above.

The checks verify the backward pass against `mlcheck.finite_difference` on a
small block, that a zero-initialised residual branch makes the block an exact
identity to floating-point precision, that `gradient_norm_through_stack` at
depth 50 keeps the residual stack's gradient within an order of magnitude of
one while the plain stack's collapses by many orders of magnitude, and that
`bottleneck_flops` matches the hand-computed value from the table above. The
`__main__` block trains a small residual network on synthetic data and prints
its loss curve — sized to be quick on a GPU and merely slow, not impossible,
on the CPU the automated checks run on.
