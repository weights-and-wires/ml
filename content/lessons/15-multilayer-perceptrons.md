---
title: Multilayer perceptrons
slug: 15-multilayer-perceptrons
part: "Part 3 — Neural networks"
summary: Why depth needs nonlinearity, what the universal approximation theorem promises, and why depth beats width on the sawtooth.
minutes: 40
gpu: false
objectives:
  - State why a composition of affine maps is affine, and what that forces onto a network with no nonlinearity.
  - State the universal approximation theorem precisely, including what it does not say about width or learnability.
  - Give the width a two-layer ReLU network needs to approximate a Lipschitz function to a stated error.
  - State the exponential-in-depth bound on the linear regions of a scalar ReLU network, and apply it to the sawtooth.
prereqs:
  - 05-logistic-regression
exercise: 15-multilayer-perceptrons
---

# Multilayer perceptrons

Logistic regression is a single affine map followed by a fixed
nonlinearity, and every derivation you have done so far has depended on
that map being the whole model. A multilayer perceptron stacks several such
maps. This lesson answers what stacking buys you, and it is not "more
capacity" as a slogan. It is a specific theorem about which functions
become representable, and a specific construction that hits a stated
error with a stated width. It is also a specific function that depth
represents efficiently while width alone cannot.

The nonlinearity between the affine maps is the entire reason any of this
works. Remove it, and depth becomes provably useless — the cleanest way to
see why the nonlinearity has to be there.

## What you already know

- An affine map is $x \mapsto Wx + b$; the composition of two affine maps is
  affine, with $W = W_2 W_1$ and $b = W_2 b_1 + b_2$.
- The rank of a product of matrices is at most the minimum of the factors'
  ranks: $\operatorname{rank}(AB) \leq \min(\operatorname{rank}(A),
  \operatorname{rank}(B))$.
- A function is Lipschitz with constant $L$ if $|f(x) - f(y)| \leq L|x - y|$
  for all $x, y$ in its domain.
- Logistic regression's map $x \mapsto \sigma(w^\top x + b)$, where $\sigma$
  is a fixed, non-affine nonlinearity applied after the affine part.
- A continuous function on a compact set is bounded and attains its
  extrema.

## The MLP as alternating affine maps and nonlinearities

A multilayer perceptron with $L$ layers computes

$$
h^{(0)} = x, \qquad h^{(\ell)} = \phi_\ell\big(W_\ell h^{(\ell-1)} + b_\ell\big)
\ \text{ for } \ell = 1, \dots, L,
$$

where $W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$, $b_\ell \in
\mathbb{R}^{n_\ell}$, and $\phi_\ell$ is a fixed nonlinearity applied
elementwise — typically the identity at the final layer, when the output is
a score or a set of logits rather than a probability. The $n_\ell$ are the
widths, and $L$ is the depth.

### Why the nonlinearity cannot be dropped

Suppose every $\phi_\ell$ is the identity. Then $h^{(\ell)} = W_\ell
h^{(\ell-1)} + b_\ell$ is affine in $h^{(\ell-1)}$, and by induction the
whole network is affine in $x$: composing two affine maps gives $(W_2 W_1)
x + (W_2 b_1 + b_2)$, and the argument extends to any number of layers. An
$L$-layer network with no nonlinearity computes $x \mapsto \bar{W} x +
\bar{b}$ for a single $\bar{W} = W_L \cdots W_1$ — the hypothesis class of
linear regression, regardless of $L$.

Depth without nonlinearity still changes what is easy to reach by gradient
descent. A hidden layer narrower than the input or output forces a rank
constraint, $\operatorname{rank}(\bar{W}) \leq \min_\ell n_\ell$. A network
with a width-$r$ bottleneck represents only affine maps of rank at most
$r$, however many wide layers surround it. This is the entire content of linear
autoencoders, and one reason PCA appears as a special case of a linear
network with a bottleneck. But such a network can never represent a
function that is not affine. Every other claim in this lesson depends on
restoring the nonlinearity between layers.

## Activations and their derivatives

Three nonlinearities recur through the rest of the course.

**ReLU.** $\operatorname{relu}(z) = \max(z, 0)$, with derivative $1$ for
$z > 0$ and $0$ for $z < 0$ — undefined at $z=0$, where implementations
pick a value in $[0,1]$ without practical consequence. ReLU is piecewise
linear, unbounded above, and exactly zero on a whole half-line. That is
both its main failure mode (a unit that lands in the zero half-line for
every training example stops learning, the "dying ReLU" problem) and the
reason it costs almost nothing to compute or differentiate.

**tanh.** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, with derivative
$1 - \tanh^2(z)$. It is bounded in $(-1,1)$, smooth, and saturates: for
large $|z|$ the derivative is exponentially small, which drives vanishing
gradients in deep or recurrent networks built from it.

**GELU.** $\operatorname{gelu}(z) = z\,\Phi(z)$, where $\Phi$ is the
standard Gaussian CDF. Since $\Phi$ has no elementary closed form, practice
uses the tanh approximation

$$
\operatorname{gelu}(z) \approx \tfrac{1}{2} z \left(1 + \tanh\!\left(
\sqrt{2/\pi}\,\big(z + 0.044715\, z^3\big)\right)\right),
$$

whose derivative follows from the product and chain rules. GELU is smooth
and non-monotone: it dips slightly negative for $z$ a little below zero
before rising, a property that is empirically useful in transformer
feedforward blocks.

## The universal approximation theorem

**Theorem (Cybenko 1989; Hornik 1991).** Let $\phi$ be any continuous,
non-polynomial function, and let $K \subset \mathbb{R}^d$ be compact. For
every continuous $f : K \to \mathbb{R}$ and every $\varepsilon > 0$, there
exist an integer $m$, weights $w_1, \dots, w_m \in \mathbb{R}^d$, biases
$b_1, \dots, b_m$, and coefficients $c_1, \dots, c_m$ such that

$$
\sup_{x \in K} \left| f(x) - \sum_{i=1}^{m} c_i\, \phi(w_i^\top x + b_i)
\right| < \varepsilon .
$$

Read this for exactly what it promises. A **two-layer** network — one
hidden layer of width $m$, one linear output — approximates any continuous
function on a compact set to any accuracy, for essentially any
nonlinearity. It says nothing about how large $m$ must be: the theorem is
purely existential. For many functions, $m$ must grow exponentially in
$d$ to hit a fixed $\varepsilon$ — an instance of the curse of
dimensionality the theorem stays silent about. It says nothing about how to
find the weights: no training procedure is guaranteed to locate a good
approximation, only that one exists. And it says nothing about depth: a
function that needs exponential width at depth two can need only
polynomial width at depth three or four, the subject of the next section.

## An explicit two-layer construction

The theorem's proof is nonconstructive in general, but for a single input
dimension you can build the approximating network by hand.

Let $f : [a,b] \to \mathbb{R}$ be $L$-Lipschitz, and let $f_m$ be the
piecewise-linear interpolant of $f$ at $m+1$ equally spaced breakpoints,
spacing $h = (b-a)/m$. On each interval, $f_m$'s convex-combination
structure bounds the gap to $f$ by the Lipschitz constant: the interpolation
error is maximised at the midpoint of an interval, where it equals
$\tfrac{Lh}{2}$, so $\sup_x |f(x) - f_m(x)| \leq \tfrac{Lh}{2}$ on all of
$[a,b]$. Setting $\tfrac{Lh}{2} \leq \varepsilon$ and solving for $m$ gives

$$
m \geq \frac{L(b-a)}{2\varepsilon} .
$$

Any continuous piecewise-linear function with interior breakpoints $t_1 <
\cdots < t_{m-1}$ can be written exactly, with no approximation, as a sum
of ReLUs:

$$
f_m(x) = f(t_0) + k_0 (x - t_0) + \sum_{j=1}^{m-1} \Delta k_j \,
\operatorname{relu}(x - t_j),
$$

where $k_0$ is the slope on the first interval and $\Delta k_j$ is the
change in slope at $t_j$. Each $\operatorname{relu}(x-t_j)$ contributes
nothing before $t_j$ and a ramp of slope $1$ after it, so summing them with
coefficients equal to the slope changes reproduces $f_m$ exactly. Writing
the leading affine term itself as a difference of two ReLUs turns the whole
sum into a genuine two-layer ReLU network. Substituting $m \approx
L(b-a)/(2\varepsilon)$ gives the headline result: a two-layer ReLU network
needs $\Theta(1/\varepsilon)$ hidden units to approximate an $L$-Lipschitz
function on $[a,b]$ to error $\varepsilon$. This bound is tight up to
constants — no piecewise-linear approximation with fewer pieces can achieve
error $\varepsilon$ against every $L$-Lipschitz function.

## Depth separation: the sawtooth

The construction above is one-dimensional, and its width is linear in
$1/\varepsilon$. In higher dimension, the same style of argument makes
width grow exponentially in $d$ for generic Lipschitz functions — the
curse of dimensionality the universal approximation theorem hides. Depth
changes the picture for specific, structured functions: some functions a
narrow, deep network computes exactly require exponentially wider shallow
networks to approximate at all.

The canonical example is the **tent map** $T(x) = 2\min(x, 1-x)$ on $[0,1]$,
a "V" folded in half. Two ReLU units compute it exactly:
$T(x) = 2\operatorname{relu}(x) - 4\operatorname{relu}(x - \tfrac12)$.
Define the depth-$k$ **sawtooth** $S_k = T \circ T \circ \cdots \circ T$
($k$ compositions). $T$ maps $[0,1]$ onto $[0,1]$ two-to-one, and
does so on each piece the previous composition already created. As a
result, $S_k$ has exactly $2^k$ monotone linear pieces on $[0,1]$: each
additional layer of $T$ doubles the piece count by folding the range back
over the whole domain. A ReLU network with $k$ hidden layers of width $2$ computes $S_k$
exactly, by feeding each tent map's output into the next tent map's two
probes — depth $k$, width $O(1)$.

The matching lower bound is a counting fact about linear regions. A ReLU
network taking a scalar input, with hidden layers of widths $n_1, \dots,
n_L$ and a linear output layer, computes a function with at most
$\prod_{l=1}^{L} (n_l+1)$ linear pieces on any interval. That bound holds
because each of the $n_l$ units in layer $l$ can cross zero at most once
inside a piece the previous layer already created, splitting it into at
most $n_l+1$ sub-pieces. A **shallow** competitor, one hidden layer of
width $n$, has at most $n+1$ pieces. Matching the sawtooth's $2^k$ pieces
therefore forces $n \geq 2^k-1$ — width exponential in $k$. The deep
network reaches the same count with $2k$ hidden units, linear in $k$. More
generally, a network of depth $L$ with width bounded by a constant $w$
computes at most $(w+1)^L$ pieces. Matching the sawtooth at fixed width
therefore forces depth $\Omega(k/\log w)$. What depth $\Theta(k)$ computes
with $O(1)$ width per layer, $O(1)$ depth needs exponential width to match
(Telgarsky; Eldan and Shamir, for related constructions).

## Cost and scaling

For a layer with input dimension $n_{\text{in}}$ and output dimension
$n_{\text{out}}$, the affine part costs $2\, n_{\text{in}} n_{\text{out}}$
flops for the matrix-vector product plus $n_{\text{out}}$ for the bias, and
the nonlinearity costs $O(n_{\text{out}})$ more. None of this depends on
depth directly; it depends on the widths, as the following table shows.

| Quantity | Formula | Notes |
|---|---|---|
| Parameters, one layer | $n_{\text{in}} n_{\text{out}} + n_{\text{out}}$ | Weights plus biases. |
| Parameters, $L$ layers | $\sum_{\ell=1}^{L} (n_{\ell-1} n_\ell + n_\ell)$ | Dominated by the widest adjacent pair. |
| Flops, one layer, ReLU | $2 n_{\text{in}} n_{\text{out}} + 2 n_{\text{out}}$ | Matmul, bias, one comparison per unit. |
| Flops, forward pass | $\sum_{\ell} \big(2 n_{\ell-1} n_\ell + O(n_\ell)\big)$ | Linear in depth, quadratic in width per layer. |

A network with $L$ layers of equal width $n$ has $O(Ln^2)$ parameters and
flops. Doubling the depth doubles the cost, while doubling the width
quadruples it. That is one reason practical architectures favor moderate
width and enough depth to reach the function class they need. The sawtooth
section is the sharpest illustration: reaching $2^k$ linear pieces costs
$O(k)$ parameters at width $2$ and depth $k$, against $O(2^k)$ parameters
at depth $2$.

## Where this breaks

**The theorem does not tell you the width you need.** Cybenko and Hornik's
result is an existence statement. Outside hand-built cases like the
Lipschitz interpolant above, the required $m$ for a target $\varepsilon$ is
unknown and can be astronomically large. Treating "a two-layer network can
approximate anything" as a design principle ignores that the "anything" is
frequently infeasible in practice.

**Optimization is not part of the guarantee.** Nothing above says gradient
descent, from a random initialization, finds weights anywhere near the
approximation the theorem promises exists. Expressivity and learnability
are different questions, and a network can be expressive enough for a task
while remaining hard to train on it.

**Piecewise-linear networks are not smooth.** A ReLU network's output has
discontinuous derivatives at every breakpoint. This is invisible in the
loss during training but matters for uses that need a smooth function,
such as physical simulators or gradient-based optimization of the input
rather than the parameters, where kinks make second-order methods behave
badly.

**Linear-region counts do not measure practical expressivity.** The bound
above is an upper bound on the pieces a network *can* produce, achieved by
adversarial weights. A network trained by gradient descent on real data
tends to realize far fewer effective regions than its architecture permits,
and region counts correlate only loosely with generalization.

## The exercise

You implement `relu`, `relu_grad`, `gelu`, and `gelu_grad`, then
`mlp_forward(params, x)` for a network given as a list of layers, each a
dict of a weight matrix, a bias, and an activation name. Then you build
`sawtooth_network(depth)`, producing the exact weights derived above for a
depth-`k` ReLU network computing the sawtooth, and `count_linear_regions_1d`,
which counts a scalar ReLU network's linear pieces exactly by propagating
breakpoints through the layers rather than by sampling. Finally,
`parameter_count` and `flop_count` implement the formulas from the table
above for an arbitrary network.

The checks verify each activation and its derivative against
`mlcheck.finite_difference`. They also verify that a two-layer network
built from two `"identity"` layers collapses to the single affine map
their product predicts, and inherits its rank bound. They verify that
`sawtooth_network(k)` produces exactly $2^k$ linear pieces for several
values of $k$ and matches the closed-form sawtooth to machine precision.
Finally, they verify that `parameter_count` and `flop_count` match
hand-worked totals for a stated four-layer architecture.
