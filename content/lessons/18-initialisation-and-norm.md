---
title: Initialisation and normalisation
slug: 18-initialisation-and-norm
part: "Part 3 — Neural networks"
summary: The variance recursion that decides whether a deep network's signal survives its own depth, and the normalisation layers that stop relying on getting it right.
minutes: 60
gpu: false
objectives:
  - Derive the forward and backward variance recursions for a deep linear stack, and Xavier and He initialisation from them.
  - Quantify what a wrong initialisation scale does to signal and gradient magnitude after fifty layers.
  - Derive batch normalisation's backward pass through the batch statistics, including both correction terms.
  - Explain why layer normalisation removes the batch-size dependence batch norm has, and why residual connections give gradients a path that bypasses depth entirely.
prereqs:
  - 15-multilayer-perceptrons
  - 16-backpropagation
exercise: 18-initialisation-and-norm
---

# Initialisation and normalisation

A network's weights start at random values before a single gradient step runs,
and the forward pass through fifty or a hundred layers of a random linear map
is a product of fifty or a hundred random matrices. Lesson 15 built the
forward pass and lesson 16 the gradient that trains it; neither says what
happens to a signal, or a gradient, repeatedly multiplied by matrices whose
scale you chose. Get that scale wrong and depth stops being an asset: the
forward activations vanish or blow up before training starts, and the
gradient that would fix them has already vanished or blown up along the same
path, in the same direction.

This lesson derives the initialisation scale that keeps a signal's variance
from drifting with depth, first for a linear layer and then for the nonlinearity
that halves it. Getting the scale exactly right is fragile and depends on
assumptions — a fixed activation, independent weights, no correlation between
layers — that training itself breaks within a few steps. Batch normalisation
and layer normalisation are the response: instead of choosing a scale and
hoping it survives, renormalise the signal at every layer and make survival
unconditional. The lesson ends with residual connections, which solve the same
problem for the gradient directly, by giving it a path through the network
that does not go through any layer's Jacobian at all.

## What you already know

- A linear layer computes $h = Wx$ with $W \in \mathbb{R}^{n_{\text{out}}
  \times n_{\text{in}}}$, and backpropagation needs $\partial h/\partial W$
  and $\partial h/\partial x$ (lesson 16).
- For independent, zero-mean random variables, the variance of a sum is the
  sum of the variances, and the variance of a product of independent zero-mean
  factors is the product of their variances plus cross terms that vanish at
  zero mean.
- ReLU is $\max(0, z)$, and for $z$ symmetric about zero, $\Pr(z>0)=1/2$.
- A Jacobian composes multiplicatively along a computational path: the
  gradient with respect to an early layer is a product of every later layer's
  Jacobian.
- A multilayer perceptron stacks affine layers and pointwise nonlinearities
  (lesson 15).

## Signal propagation through a deep linear stack

Consider a stack of layers $h^{(l)} = W^{(l)} h^{(l-1)}$, $W^{(l)} \in
\mathbb{R}^{n_l \times n_{l-1}}$, with entries drawn i.i.d. from a zero-mean
distribution of variance $\operatorname{Var}(W)$, independent of the input
$h^{(l-1)}$. For one output unit $h^{(l)}_i = \sum_{j=1}^{n_{l-1}} W^{(l)}_{ij}
h^{(l-1)}_j$, and assuming the components of $h^{(l-1)}$ are independent of
each other and of $W^{(l)}$ with common variance $\operatorname{Var}(h^{(l-1)})$,

$$
\operatorname{Var}\big(h^{(l)}_i\big)
= \sum_{j=1}^{n_{l-1}} \operatorname{Var}\big(W^{(l)}_{ij}\big)\operatorname{Var}\big(h^{(l-1)}_j\big)
= n_{l-1}\,\operatorname{Var}(W)\,\operatorname{Var}\big(h^{(l-1)}\big) ,
$$

using $\operatorname{Var}(W_{ij}h_j) = \operatorname{Var}(W_{ij})\operatorname{Var}(h_j)$
for independent zero-mean factors and summing $n_{l-1}$ independent terms. This
is the **forward variance recursion**. Iterating it from an input of variance
$\operatorname{Var}(h^{(0)})$ gives

$$
\operatorname{Var}\big(h^{(L)}\big) = \operatorname{Var}\big(h^{(0)}\big)
\prod_{l=1}^{L} n_{l-1}\operatorname{Var}(W) .
$$

The variance is flat across depth exactly when every factor
$n_{l-1}\operatorname{Var}(W)$ equals one — any other value compounds
geometrically.

The backward pass obeys the mirror recursion. Backpropagation computes
$\delta^{(l-1)} = W^{(l)\top}\delta^{(l)}$, where $\delta^{(l)} =
\partial L/\partial h^{(l)}$. By the same argument, with $W^{(l)\top}$'s
entries having the same variance as $W^{(l)}$'s and summing over the $n_l$
components of $\delta^{(l)}$ that a fixed input coordinate feeds into,

$$
\operatorname{Var}\big(\delta^{(l-1)}\big) = n_l\,\operatorname{Var}(W)\,\operatorname{Var}\big(\delta^{(l)}\big) .
$$

Forward wants $n_{l-1}\operatorname{Var}(W)=1$; backward wants
$n_l\operatorname{Var}(W)=1$. The two agree only when $n_{l-1}=n_l$ — a network
whose width never changes. Xavier initialisation (Glorot and Bengio) accepts
that the two cannot both be satisfied exactly and compromises with the
harmonic-mean-like choice

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}} ,
$$

which reduces to the exact solution $1/n$ when $n_{\text{in}}=n_{\text{out}}=n$
and interpolates between the forward and backward targets otherwise.

### He initialisation

Xavier's derivation assumes a linear layer. Insert a ReLU,
$h^{(l)} = \max(0, W^{(l)}h^{(l-1)})$, and the recursion changes: for a
pre-activation $z$ symmetric about zero (true when $W^{(l)}h^{(l-1)}$ is a sum
of many independent zero-mean terms, by the central limit theorem),
$\mathbb E[\max(0,z)^2] = \tfrac12 \mathbb E[z^2]$, because the integral over
the negative half is exactly zero and the two halves of a symmetric
distribution contribute equally to $\mathbb E[z^2]$. Since $z$ is zero-mean,
$\mathbb E[z^2]=\operatorname{Var}(z)$, but $h^{(l)}=\max(0,z)$ is not
zero-mean, so from here $\operatorname{Var}(h^{(l)})$ is shorthand for
$\mathbb E[(h^{(l)})^2]$ — the mean-square activation, which is exactly the
quantity the next layer's zero-mean pre-activation variance depends on,
whether or not $h^{(l)}$ itself is centred. So

$$
\operatorname{Var}\big(h^{(l)}\big) = \tfrac{1}{2}\, n_{l-1}\operatorname{Var}(W)\,\operatorname{Var}\big(h^{(l-1)}\big) ,
$$

and preserving variance forward requires $\operatorname{Var}(W) =
2/n_{\text{in}}$ — He initialisation. Using Xavier's scale with a ReLU network
leaves every layer's variance at half the previous one's, a factor the
simulation in the exercise measures directly.

### What the wrong scale costs

Write the per-layer factor as $c = n_{l-1}\operatorname{Var}(W)$ for a linear
stack (or $c/2$ under ReLU with He's normalisation omitted). After $L$ layers
the signal's standard deviation has scaled by $c^{L/2}$. At $L=50$, a $10\%$
error in the variance factor ($c=1.1$ instead of $1$) leaves
$1.1^{25} \approx 10.8$ on the standard-deviation scale — activations an order
of magnitude too large — while $c=0.9$ leaves $0.9^{25}\approx 0.072$, most of
the signal gone. The same factor multiplies the backward pass, so a network
initialised with the wrong scale starts training with both an unusable forward
signal and a gradient that is unusable in the same direction: variance errors
do not average out across layers, they compound.

## Batch normalisation

Batch normalisation removes the dependence on getting $\operatorname{Var}(W)$
right by renormalising activations directly. Given a batch of $N$ activations
for one feature, $x_1,\dots,x_N$,

$$
\mu = \frac1N\sum_{i=1}^N x_i, \qquad
\sigma^2 = \frac1N\sum_{i=1}^N (x_i-\mu)^2, \qquad
\hat x_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}, \qquad
y_i = \gamma\hat x_i + \beta ,
$$

with $\gamma,\beta$ learned per feature. By construction $\hat x$ has zero mean
and unit variance across the batch regardless of the layer's input scale, and
$\gamma,\beta$ let the network undo the normalisation when that is the better
choice.

### The backward pass

Given $\partial L/\partial y_i = dy_i$, the parameter gradients follow directly
from $y_i = \gamma\hat x_i+\beta$:

$$
\frac{\partial L}{\partial\beta} = \sum_i dy_i, \qquad
\frac{\partial L}{\partial\gamma} = \sum_i dy_i\,\hat x_i, \qquad
d\hat x_i = dy_i\,\gamma .
$$

The harder term is $\partial L/\partial x_j$, because $\hat x_i$ depends on
every $x_j$ through $\mu$ and $\sigma^2$, not only through $x_i$. Write
$\text{istd}=(\sigma^2+\epsilon)^{-1/2}$. Differentiating $\mu$ and $\sigma^2$
with respect to $x_j$ gives $\partial\mu/\partial x_j = 1/N$ and, using
$\sum_k(x_k-\mu)=0$, $\partial\sigma^2/\partial x_j = \tfrac2N(x_j-\mu)$, so

$$
\frac{\partial\hat x_i}{\partial x_j}
= \text{istd}\left(\delta_{ij} - \frac1N - \frac1N\hat x_i\hat x_j\right) ,
$$

after substituting $(x_i-\mu)=\hat x_i/\text{istd}$ and simplifying the
$\text{istd}^3$ term that comes from differentiating through $\sigma^2$. Sum
over $i$ against $d\hat x_i$ to get $\partial L/\partial x_j = \sum_i
d\hat x_i\,\partial\hat x_i/\partial x_j$, which collapses to

$$
\frac{\partial L}{\partial x_j} = \frac{\text{istd}}{N}\left(
N\,d\hat x_j \;-\; \sum_{i} d\hat x_i \;-\; \hat x_j\sum_i d\hat x_i\hat x_i
\right) .
$$

The two correction terms beyond the naive $\text{istd}\cdot d\hat x_j$ are
exactly $\mu$'s and $\sigma^2$'s dependence on $x_j$: $\sum_i d\hat x_i$
corrects for every output shifting when the mean shifts, and $\hat
x_j\sum_i d\hat x_i\hat x_i$ corrects for every output rescaling when the
variance shifts. Drop either term and the gradient is wrong for every $x_j$,
not only the one you differentiated with respect to.

### Train, test, and batch size

At training time $\mu$ and $\sigma^2$ are the current minibatch's statistics,
which makes a single example's normalised value depend on which other
examples happen to share its batch — a source of noise that acts as a mild
regulariser but means the layer computes a different function on every batch.
At test time, a batch of size one has no variance to normalise by, so batch
normalisation instead uses a running average of $\mu$ and $\sigma^2$
accumulated during training, fixed after training ends. This makes the
train-time and test-time computations genuinely different functions of the
input, not merely the same function evaluated on different data, and the gap
between them widens whenever the running average was estimated from too few
or too unrepresentative batches. The batch statistics also get noisier as $N$
shrinks — $\operatorname{Var}(\hat\mu) = \sigma^2/N$ for the mean estimate
alone — so batch normalisation degrades on the small batches that memory
constraints force on large models.

## Layer normalisation

Layer normalisation computes the same $\hat x = (x-\mu)/\sqrt{\sigma^2+\epsilon}$
transform, but with $\mu$ and $\sigma^2$ taken across one example's features
rather than across the batch:

$$
\mu_n = \frac1D\sum_{d=1}^D x_{n,d}, \qquad
\sigma_n^2 = \frac1D\sum_{d=1}^D (x_{n,d}-\mu_n)^2 .
$$

Every statistic is a function of a single row of the input, so the backward
pass is the batch-norm formula above with $N$ replaced by $D$ and the sums
taken over features instead of over the batch — and, critically, normalising
example $n$ never touches example $m$. There is no batch statistic to
estimate, no running average, and no discrepancy between training and test:
the function computed is identical either way, and it is well defined even at
a batch size of one. That is why sequence models use layer normalisation
almost exclusively — a batch of sequences padded to different lengths does not
give batch normalisation a well-defined per-position statistic to normalise
by, while layer normalisation, working within a single token's feature
vector, is unaffected by anything about the rest of the batch.

**RMSNorm** drops the mean subtraction entirely, normalising only by the root
mean square, $\text{rms}(x)=\sqrt{\tfrac1D\sum_d x_d^2+\epsilon}$, and computing
$y=\gamma\, x/\text{rms}(x)$. It saves one reduction and one subtraction per
call, and empirically it matches layer normalisation's effect on training
stability at transformer scale, since re-centring turns out to contribute less
to the effect than rescaling does.

## Residual connections

A residual block computes $y = x + F(x)$ instead of $y=F(x)$. Its Jacobian is
$\partial y/\partial x = I + \partial F/\partial x$, so the upstream gradient
splits as

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}
+ \frac{\partial L}{\partial y}\frac{\partial F}{\partial x} .
$$

The first term carries the gradient through unchanged, regardless of what
$\partial F/\partial x$ is. Stack $L$ residual blocks and the identity term
survives every one of them: the gradient reaching the input is at least
$\partial L/\partial y^{(L)}$ in magnitude before any block's Jacobian
multiplies anything, which is exactly the guarantee a plain deep stack does
not have — there, every layer's Jacobian multiplies the gradient, and a
Jacobian with singular values below one shrinks it geometrically, the same
compounding this lesson opened with.

**Post-norm**, $y = \operatorname{LayerNorm}(x+F(x))$, renormalises the sum at
every block, which puts the identity path through the normalisation layer's
Jacobian too and reintroduces exactly the depth-dependent attenuation residual
connections were meant to remove — deep post-norm networks are correspondingly
hard to train without careful warmup. **Pre-norm**, $y = x +
F(\operatorname{LayerNorm}(x))$, only normalises the input to $F$, leaving the
running sum — the residual stream — untouched, so the identity path is exact.
The cost shows up in the forward direction instead: with each block
contributing a roughly independent increment, $\operatorname{Var}(x^{(l)})
\approx \operatorname{Var}(x^{(0)}) + \sum_{i=1}^l \operatorname{Var}(F_i)$
grows with depth, so later blocks see a residual stream with a far larger norm
than earlier ones do, which is the trade-off pre-norm architectures manage
rather than avoid.

## Cost and scaling

| Layer | Extra parameters | Extra reductions per call | Batch dependence |
|---|---|---|---|
| Batch norm | $2D$ ($\gamma,\beta$) | 2 (mean, variance over $N$) | Yes: needs $N>1$, and a running estimate at test time |
| Layer norm | $2D$ | 2 (mean, variance over $D$) | None |
| RMSNorm | $D$ ($\gamma$ only) | 1 (mean square over $D$) | None |

All three cost $O(ND)$ flops, dominated by the reductions; the difference
between them is entirely about which axis they reduce over and what that axis
costs to depend on.

## Where this breaks

**Getting the scale wrong without a norm layer.** A plain deep network with no
normalisation is exactly as fragile as the "what the wrong scale costs"
calculation above predicts, and this is the practical reason normalisation
layers, not tighter initialisation, became the default rather than a backup.

**Batch norm with small or correlated batches.** Batches drawn with
correlation — all examples from one class, or from one video's consecutive
frames — give $\mu$ and $\sigma^2$ that reflect the batch's composition, not
the data distribution, and the running average absorbs that bias into every
inference the model makes afterward.

**Normalising across the wrong axis in a sequence model.** Computing batch
statistics across the batch dimension in a model that sees variable-length,
padded sequences mixes real tokens with padding unless the implementation
masks it out — a bug that layer normalisation's per-token statistics cannot
have, because it never looks outside one token's own features.

**Residual streams without any variance control.** Pre-norm's additive
variance growth, left unmanaged across enough layers, pushes the input to the
final layers far outside the range earlier layers were tuned for; several
large-model recipes rescale residual branches by a depth-dependent factor for
exactly this reason.

## The exercise

You implement the two initialisation scales and confirm what they do to a
signal over depth, then the forward and backward passes of both normalisation
layers.

Write `xavier_scale(n_in, n_out)` and `he_scale(n_in)`, returning the standard
deviation to draw weights from. Write `forward_variance(depth, widths, scale,
activation)`, which simulates a random stack of the given widths and
activation (`"linear"` or `"relu"`) using i.i.d. weights of that scale, and
returns the empirical variance of the activation at every layer, including the
input. Then write `batchnorm_forward`, `batchnorm_backward`,
`layernorm_forward`, and `layernorm_backward`, following the derivations
above; each forward function returns the normalised output together with
whatever cache its backward needs.

The checks confirm the forward passes first — that batch norm's output has
zero mean and unit variance per feature, and that layer norm's has zero mean
and unit variance per example — then verify both backward passes against
`mlcheck.finite_difference`, with respect to the input and to $\gamma$ and
$\beta$ separately, since a correct forward pass and an incorrect backward
pass are a common way to fail silently. A further check shifts one whole
example by a constant and confirms layer norm's output for that example does
not move while batch norm's does. The last check is the point of the lesson:
`forward_variance` run for fifty ReLU layers shows He's scale holding the
activation variance flat while Xavier's decays it by almost exactly the
one-half per layer the derivation predicts.
