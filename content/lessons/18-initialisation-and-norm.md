---
title: Initialisation and normalisation
slug: 18-initialisation-and-norm
part: "Part 3 — Neural networks"
summary: The scale that keeps a deep network's signal from vanishing or exploding, and the normalisation layers that stop relying on getting it right.
minutes: 50
gpu: false
objectives:
  - Derive Xavier and He initialisation from the variance a deep linear stack needs to stay flat.
  - Quantify what a wrong initialisation scale does to signal magnitude after fifty layers.
  - Implement the forward and backward pass of batch normalisation and layer normalisation.
  - Explain why layer normalisation removes the batch-size dependence batch norm has, and why residual connections give gradients a path that bypasses depth entirely.
prereqs:
  - 15-multilayer-perceptrons
  - 16-backpropagation
exercise: 18-initialisation-and-norm
---

# Initialisation and normalisation

A network's weights start at random values before a single gradient step
runs. A forward pass through fifty or a hundred layers is then a product of
fifty or a hundred random matrices. Lesson 15 built that forward pass and
lesson 16 the gradient that trains it, but neither says what happens to a
signal repeatedly multiplied by matrices whose scale you chose. Get that
scale wrong, and depth stops being an asset: activations vanish or blow up
before training starts, and the gradient that would fix them vanishes or
blows up right along with them.

This lesson derives the initialisation scale that keeps a signal's variance
flat across depth, first for a linear layer and then for a network with
ReLU. That scale is fragile: it depends on assumptions that training breaks
within a few steps. Batch normalisation and layer normalisation respond by
renormalising the signal at every layer instead, so survival no longer
depends on choosing the right scale at all. The lesson ends with residual
connections, which solve the same problem for the gradient directly, by
giving it a path through the network that skips every layer's Jacobian.

## What you already know

- A linear layer computes $h = Wx$, and backpropagation needs
  $\partial h/\partial W$ and $\partial h/\partial x$ (lesson 16).
- For independent, zero-mean random variables, the variance of a sum is the
  sum of the variances.
- ReLU is $\max(0, z)$, and for $z$ symmetric about zero, half its mass sits
  on each side.
- A Jacobian composes multiplicatively along a computation path: the
  gradient at an early layer is a product of every later layer's Jacobian.
- A multilayer perceptron stacks affine layers and pointwise nonlinearities
  (lesson 15).

## The variance a deep stack needs

Consider a stack of layers $h^{(l)} = W^{(l)} h^{(l-1)}$, where each
$W^{(l)}$ has entries drawn independently from a zero-mean distribution, and
$h^{(l-1)}$'s components are independent of each other and of $W^{(l)}$. Each
output unit sums $n_{l-1}$ independent products, so its variance scales with
the number of inputs it sums over:

$$
\operatorname{Var}\big(h^{(l)}\big) = n_{l-1}\,\operatorname{Var}(W)\,\operatorname{Var}\big(h^{(l-1)}\big) .
$$

The signal's variance stays flat across depth exactly when
$n_{l-1}\operatorname{Var}(W) = 1$; any other value compounds it
geometrically, layer after layer. The gradient obeys a mirror recursion
running backward, which wants $n_l\operatorname{Var}(W) = 1$ instead. The two
targets agree only when every layer has the same width. **Xavier
initialisation** (Glorot and Bengio) accepts that both cannot hold exactly
and compromises:

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} .
$$

### He initialisation

Xavier's derivation assumes a linear layer. A ReLU changes the recursion,
because it zeroes roughly half of every pre-activation: for a pre-activation
symmetric about zero, $\mathbb E[\max(0,z)^2] = \tfrac12\mathbb E[z^2]$, so a
ReLU layer's mean-square activation is half of what the linear recursion
above predicts. Preserving variance forward through a ReLU network needs
double the weight variance a linear network would use:

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}}} .
$$

This is **He initialisation**. Using Xavier's scale with a ReLU network
leaves every layer's variance at half the previous one's.

### What the wrong scale costs

Write the per-layer variance factor as $c$. After $L$ layers, the signal's
standard deviation has scaled by $c^{L/2}$. At $L=50$, a $10\%$ error in $c$
compounds badly: $c=1.1$ leaves activations roughly ten times too large, and
$c=0.9$ leaves less than a tenth of the signal standing. The same factor
multiplies the backward pass, so a poorly scaled network starts training
with both an unusable forward signal and a gradient that has vanished or
exploded in the same direction. Errors like this do not average out across
layers — they compound.

## Batch normalisation

Batch normalisation sidesteps the whole scale problem by renormalising
activations directly, rather than hoping the weight scale keeps them in
range. Given a batch of $N$ activations for one feature,

$$
\mu = \frac1N\sum_{i=1}^N x_i, \qquad
\sigma^2 = \frac1N\sum_{i=1}^N (x_i-\mu)^2, \qquad
\hat x_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}, \qquad
y_i = \gamma\hat x_i + \beta ,
$$

with $\gamma$ and $\beta$ learned per feature. By construction, $\hat x$ has
zero mean and unit variance across the batch, whatever scale the layer's
input arrived at, and $\gamma$, $\beta$ let the network undo that
normalisation when it helps.

The backward pass is harder than it looks, because $\hat x_i$ depends on
every $x_j$ in the batch through $\mu$ and $\sigma^2$, not only through
$x_i$. Writing $\text{istd} = (\sigma^2+\epsilon)^{-1/2}$ and
$d\hat x_i = \partial L/\partial y_i \cdot \gamma$, the gradient with respect
to one input is

$$
\frac{\partial L}{\partial x_j} = \frac{\text{istd}}{N}\left(
N\,d\hat x_j \;-\; \sum_{i} d\hat x_i \;-\; \hat x_j\sum_i d\hat x_i\hat x_i
\right) .
$$

The two correction terms beyond a naive $\text{istd}\cdot d\hat x_j$ account
for $\mu$ and $\sigma^2$ both depending on $x_j$: the middle term corrects
for every output shifting when the mean shifts, and the last term corrects
for every output rescaling when the variance shifts. Drop either one, and the
gradient is wrong for every input, not only the one you differentiated.

At training time, $\mu$ and $\sigma^2$ come from the current minibatch, so a
single example's normalised value depends on which other examples share its
batch — noise that acts as a mild regulariser but makes the layer compute a
slightly different function on every batch. At test time, batch normalisation
uses a running average of $\mu$ and $\sigma^2$ collected during training
instead, because a batch of size one has no variance to normalise by. That
makes training and inference genuinely different computations, and the gap
between them widens whenever the running average came from too few or too
unrepresentative batches.

## Layer normalisation

Layer normalisation applies the same transform, but computes $\mu$ and
$\sigma^2$ across one example's features instead of across the batch. Its
backward pass is the batch-norm formula above with the batch size $N$
replaced by the feature count and the sums taken over features. Normalising
one example never touches any other example, so there is no batch statistic
to estimate, no running average, and no gap between training and test — the
function computed is identical either way, even at a batch size of one.
That is why sequence models almost always use layer normalisation: a batch
of sequences padded to different lengths gives batch normalisation no clean
per-position statistic, while layer normalisation, working within a single
token's own features, is unaffected by anything else in the batch.

**RMSNorm** drops the mean subtraction and normalises only by the root mean
square, $y = \gamma\, x / \sqrt{\tfrac1D\sum_d x_d^2 + \epsilon}$. It saves a
reduction and a subtraction, and at transformer scale it matches layer
normalisation's effect on training stability, since re-centring turns out to
matter less than rescaling does.

## Residual connections

A residual block computes $y = x + F(x)$ instead of $y = F(x)$, so its
Jacobian is $I + \partial F/\partial x$ and the upstream gradient splits into
two paths: one that passes through $F$'s Jacobian, and one, the identity
term, that does not. Stack $L$ residual blocks, and that identity term
survives every one of them, so the gradient reaching the input is never
smaller than the gradient leaving the last block — the guarantee a plain
deep stack does not have, since every layer's Jacobian there can shrink the
gradient further.

**Post-norm**, $y = \operatorname{LayerNorm}(x+F(x))$, renormalises the sum
at every block, which routes the identity path through the normalisation
layer's Jacobian too and brings back the depth-dependent attenuation
residual connections were meant to remove. **Pre-norm**,
$y = x + F(\operatorname{LayerNorm}(x))$, normalises only the input to $F$,
leaving the running sum untouched, so the identity path stays exact. The
trade-off shows up going forward instead: each block's contribution adds to
the running variance, so later blocks see a much larger signal than earlier
ones do.

## Cost and scaling

| Layer | Extra parameters | Reductions per call | Batch dependence |
|---|---|---|---|
| Batch norm | $2D$ ($\gamma$, $\beta$) | 2, over the batch | Needs $N>1$, and a running estimate at test time |
| Layer norm | $2D$ | 2, over the features | None |
| RMSNorm | $D$ ($\gamma$ only) | 1, over the features | None |

All three cost $O(ND)$ flops for a batch of $N$ examples and $D$ features,
dominated by the reductions; the difference between them is which axis they
reduce over.

## Where this breaks

**No normalisation layer, and the wrong scale.** A plain deep network with no
norm layer is exactly as fragile as the "what the wrong scale costs"
calculation above predicts — the practical reason normalisation layers
became the default rather than a backup to careful initialisation.

**Batch norm with small or correlated batches.** A batch drawn with
correlation, such as every example from one class, gives $\mu$ and
$\sigma^2$ that reflect the batch's makeup rather than the data
distribution, and the running average carries that bias into every later
prediction.

**Normalising across the wrong axis in a sequence model.** Batch statistics
computed across variable-length, padded sequences mix real tokens with
padding unless the code masks it out. Layer normalisation cannot have this
bug, because it never looks outside one token's own features.

**Residual streams with no variance control.** Pre-norm's variance growth,
left unmanaged across enough layers, pushes the final layers' input far
outside the range earlier layers were tuned for, which is why several large
model recipes rescale residual branches by a depth-dependent factor.

## The exercise

You implement the two initialisation scales and confirm what they do to a
signal over depth, then the forward and backward passes of both
normalisation layers.

Write `xavier_scale(n_in, n_out)` and `he_scale(n_in)`, returning the
standard deviation to draw weights from. Write `forward_variance(depth,
widths, scale, activation)`, which simulates a random stack of the given
widths and activation (`"linear"` or `"relu"`) using weights of that scale,
and returns the empirical variance of the activation at every layer,
including the input. Then write `batchnorm_forward`, `batchnorm_backward`,
`layernorm_forward`, and `layernorm_backward`, following the derivations
above; each forward function returns the normalised output together with
whatever cache its backward needs.

The checks confirm the forward passes first, that batch norm's output has
zero mean and unit variance per feature and that layer norm's has zero mean
and unit variance per example, then verify both backward passes against
`mlcheck.finite_difference`, with respect to the input, $\gamma$, and $\beta$
separately, since a correct forward pass and a wrong backward pass are a
common way to fail silently. A further check shifts one whole example by a
constant and confirms layer norm's output for that example does not move
while batch norm's does. The last check is the point of the lesson:
`forward_variance` run for fifty ReLU layers shows He's scale holding the
activation variance flat, while Xavier's decays it by almost exactly the
one-half per layer the derivation predicts.
