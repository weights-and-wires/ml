---
title: Multilayer perceptrons
slug: 15-multilayer-perceptrons
part: "Part 3 — Neural networks"
summary: Why depth needs nonlinearity to matter, what the universal approximation theorem does and does not promise, and the sawtooth that separates depth from width.
minutes: 45
gpu: false
objectives:
  - Prove that a composition of affine maps is affine, and state what that forces onto a network with no nonlinearity.
  - State the universal approximation theorem precisely, including what it does not say about width or learnability.
  - Construct a two-layer ReLU network that approximates a Lipschitz function to a stated error, and give the width it needs.
  - Derive the exponential-in-depth bound on the linear regions of a ReLU network with scalar input, and apply it to the sawtooth.
prereqs:
  - 05-logistic-regression
exercise: 15-multilayer-perceptrons
---

# Multilayer perceptrons

Logistic regression is a single affine map followed by a fixed nonlinearity,
and every derivation you have done so far has depended on that map being the
whole model. A multilayer perceptron stacks several such maps, and the
question this lesson answers is what stacking buys you, exactly: not "more
capacity" as a slogan, but a specific theorem about which functions become
representable, a specific construction that achieves a stated error with a
stated width, and a specific example of a function that depth represents
efficiently and width alone cannot.

The nonlinearity between the affine maps is the entire reason any of this
works. Remove it and depth is provably useless — a fact this lesson proves
first, because it is the cleanest way to see why the nonlinearity has to be
there at all.

## What you already know

- An affine map is $x \mapsto Wx + b$; the composition of two affine maps is
  affine, with $W = W_2 W_1$ and $b = W_2 b_1 + b_2$.
- The rank of a product of matrices is at most the minimum of the factors'
  ranks: $\operatorname{rank}(AB) \leq \min(\operatorname{rank}(A),
  \operatorname{rank}(B))$.
- A function is Lipschitz with constant $L$ if $|f(x) - f(y)| \leq L|x - y|$
  for all $x, y$ in its domain.
- Logistic regression's map $x \mapsto \sigma(w^\top x + b)$, and that
  $\sigma$ is a fixed, non-affine nonlinearity applied after the affine part.
- A continuous function on a compact set is bounded and attains its extrema.

## The MLP as alternating affine maps and nonlinearities

A multilayer perceptron with $L$ layers computes

$$
h^{(0)} = x, \qquad h^{(\ell)} = \phi_\ell\big(W_\ell h^{(\ell-1)} + b_\ell\big)
\ \text{ for } \ell = 1, \dots, L,
$$

where $W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$, $b_\ell \in
\mathbb{R}^{n_\ell}$, and $\phi_\ell$ is a fixed nonlinearity applied
elementwise — typically identity at the final layer, when the output is a
score or a set of logits rather than a probability or a class label. The
$n_\ell$ are the widths, and $L$ is the depth.

### Why the nonlinearity cannot be dropped

Suppose every $\phi_\ell$ is the identity. Then $h^{(\ell)} = W_\ell
h^{(\ell-1)} + b_\ell$ is affine in $h^{(\ell-1)}$, and by induction the whole
network is affine in $x$: composing $h^{(1)} = W_1 x + b_1$ with $h^{(2)} =
W_2 h^{(1)} + b_2$ gives $h^{(2)} = (W_2 W_1) x + (W_2 b_1 + b_2)$, an affine
map with weight matrix $W_2 W_1$ and bias $W_2 b_1 + b_2$, and the same
argument extends to any number of layers. An $L$-layer network with no
nonlinearity computes $x \mapsto \bar{W} x + \bar{b}$ for some single
$\bar{W} = W_L W_{L-1} \cdots W_1$ and $\bar{b}$ built the same way — exactly
the hypothesis class of linear regression, regardless of $L$.

Depth without nonlinearity is not wasted in the sense of computing nothing;
it changes what is *easy to reach by gradient descent* and, when a hidden
layer is narrower than the input or output, it forces a rank constraint:
$\operatorname{rank}(\bar{W}) \leq \min_\ell n_\ell$, so a network with a
width-$r$ bottleneck can only represent affine maps of rank at most $r$,
however many wide layers surround it. This is the entire content of linear
autoencoders and one reason PCA appears as a special case of a linear
network with a bottleneck. But it can never represent a function that is not
affine. Every other claim in this lesson is downstream of restoring the
nonlinearity between layers.

## Activations and their derivatives

Three nonlinearities recur throughout the rest of the course.

**ReLU.** $\operatorname{relu}(z) = \max(z, 0)$, with derivative $1$ for $z >
0$ and $0$ for $z < 0$ — undefined at $z = 0$, where any value in $[0, 1]$ is
a valid subgradient, and implementations pick one (commonly $0$) without
consequence in practice. ReLU is piecewise linear, unbounded above, and
exactly zero on a whole half-line, which is both its main failure mode (a
unit that lands in the zero half-line for every training example stops
learning, the "dying ReLU" problem) and the reason it is nearly free to
compute and to differentiate.

**tanh.** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, with derivative $1 -
\tanh^2(z)$. It is bounded in $(-1, 1)$, smooth, and saturates: for $|z|$
large the derivative is exponentially small, which is the mechanism behind
vanishing gradients in deep or recurrent networks built from it.

**GELU.** $\operatorname{gelu}(z) = z \, \Phi(z)$, where $\Phi$ is the
standard Gaussian CDF — the expected value of $z$ under a Bernoulli gate that
keeps $z$ with probability $\Phi(z)$. The CDF has no elementary closed form,
so practice uses the tanh approximation

$$
\operatorname{gelu}(z) \approx \tfrac{1}{2} z \left(1 + \tanh\!\left(
\sqrt{2/\pi}\,\big(z + 0.044715\, z^3\big)\right)\right),
$$

whose derivative follows from the product and chain rules applied to that
closed form: writing $u(z) = \sqrt{2/\pi}(z + 0.044715 z^3)$ and $t = \tanh
u$,

$$
\frac{d}{dz}\operatorname{gelu}(z) \approx \tfrac{1}{2}(1 + t) +
\tfrac{1}{2} z (1 - t^2) \, u'(z), \qquad u'(z) = \sqrt{2/\pi}\big(1 +
3 \times 0.044715\, z^2\big).
$$

Unlike ReLU, GELU is smooth and non-monotone — it dips slightly negative for
$z$ a little below zero before rising — which is a property empirically
useful in transformer feedforward blocks and one this lesson does not
attempt to explain from first principles.

## The universal approximation theorem

**Theorem (Cybenko 1989; Hornik 1991).** Let $\phi$ be any continuous,
non-polynomial function, and let $K \subset \mathbb{R}^d$ be compact. For
every continuous $f : K \to \mathbb{R}$ and every $\varepsilon > 0$, there
exist an integer $m$, weights $w_1, \dots, w_m \in \mathbb{R}^d$, biases
$b_1, \dots, b_m \in \mathbb{R}$, and coefficients $c_1, \dots, c_m \in
\mathbb{R}$ such that

$$
\sup_{x \in K} \left| f(x) - \sum_{i=1}^{m} c_i\, \phi(w_i^\top x + b_i)
\right| < \varepsilon .
$$

Read this precisely for what it promises and refuses to promise. It says a
**two-layer** network — one hidden layer of width $m$, one linear output —
suffices to approximate any continuous function on a compact set to any
accuracy, for essentially any nonlinearity. It says nothing about how large
$m$ must be: the theorem is purely existential, and for many functions of
interest $m$ must grow exponentially in $d$ to hit a fixed $\varepsilon$, an
instance of the curse of dimensionality that the theorem is silent about. It
says nothing about how to find the weights: gradient descent is not
mentioned, and the theorem gives no guarantee that any particular training
procedure locates a good approximation, only that one exists. And it says
nothing about depth: the theorem is a statement about width alone, and a
function that needs exponential width at depth two can need only polynomial
width at depth three or four — the subject of the next two sections.

## An explicit two-layer construction

The theorem's proof is nonconstructive in general, but for a single input
dimension it is possible to build the approximating network by hand and read
off the width $\varepsilon$ costs.

Let $f : [a, b] \to \mathbb{R}$ be $L$-Lipschitz. Partition $[a, b]$ into $m$
equal intervals with breakpoints $t_0 = a < t_1 < \cdots < t_m = b$, spacing
$h = (b-a)/m$, and let $f_m$ be the piecewise-linear interpolant agreeing
with $f$ at every $t_j$. Fix an interval $[t_j, t_{j+1}]$ and write $x = t_j
+ s$ with $0 \leq s \leq h$. Because $f_m$ interpolates linearly between the
endpoints, $f_m(x) = (1 - s/h) f(t_j) + (s/h) f(t_{j+1})$ is a convex
combination of the two endpoint values, so

$$
f(x) - f_m(x) = \Big(1 - \frac{s}{h}\Big)\big(f(x) - f(t_j)\big) +
\frac{s}{h}\big(f(x) - f(t_{j+1})\big) .
$$

Bounding each term by the Lipschitz constant, $|f(x)-f(t_j)| \leq Ls$ and
$|f(x)-f(t_{j+1})| \leq L(h-s)$, and combining with the triangle inequality,

$$
|f(x) - f_m(x)| \leq \Big(1-\frac{s}{h}\Big) Ls + \frac{s}{h} L(h-s) =
\frac{2Ls(h-s)}{h} .
$$

The right side is maximised over $s \in [0,h]$ at $s = h/2$, where it equals
$\tfrac{Lh}{2}$. So $\sup_x |f(x) - f_m(x)| \leq \tfrac{Lh}{2}$ on every
interval and hence on all of $[a,b]$ — the interpolation error of a
Lipschitz function is at most half the Lipschitz constant times the mesh
width, with equality approached by a function that moves away from the
chord at the constant rate $L$ from both endpoints toward the interval's
midpoint. Setting $\tfrac{Lh}{2} \leq \varepsilon$ and solving for $m =
(b-a)/h$ gives

$$
m \geq \frac{L(b-a)}{2\varepsilon} .
$$

Now realise $f_m$ exactly with ReLUs. Any continuous piecewise-linear
function with breakpoints $t_1 < \cdots < t_{m-1}$ (interior breakpoints
only) can be written as

$$
f_m(x) = f(t_0) + k_0 (x - t_0) + \sum_{j=1}^{m-1} \Delta k_j \,
\operatorname{relu}(x - t_j),
$$

where $k_0$ is the slope on the first interval and $\Delta k_j$ is the
*change* in slope at $t_j$ — the slope on $[t_j, t_{j+1}]$ minus the slope on
$[t_{j-1}, t_j]$. Each term $\operatorname{relu}(x - t_j)$ contributes
nothing before $t_j$ and adds a ramp of slope $1$ after it, so summing them
with coefficients equal to the slope changes reproduces $f_m$ exactly: this
is the ReLU analogue of the Newton divided-difference form, and it is exact,
not an approximation. Writing the leading affine term $f(t_0) + k_0(x -
t_0)$ itself as $c_+ \operatorname{relu}(x - t_0) - c_- \operatorname{relu}(t_0
- x)$ for suitable $c_+, c_-$ turns the whole thing into a genuine two-layer
ReLU network — one hidden layer, one linear combination — using $m + 1$
hidden units for $m-1$ interior breakpoints plus the two units for the
leading term.

Substituting $m \approx L(b-a)/(2\varepsilon)$ gives the headline result: a
two-layer ReLU network needs $\Theta(1/\varepsilon)$ hidden units to
approximate an $L$-Lipschitz function on $[a,b]$ to sup-norm error
$\varepsilon$, with the constant fixed by $L$ and the domain length. This is
also tight up to constants: no continuous piecewise-linear approximation
with fewer than about $L(b-a)/(2\varepsilon)$ pieces can achieve error
$\varepsilon$ against every $L$-Lipschitz function, because such a function
can zigzag at rate $L$ inside every interval regardless of how the
breakpoints are placed.

## Depth separation: the sawtooth

The construction above is one-dimensional and its width is linear in
$1/\varepsilon$. In higher dimension the same style of argument makes width
requirements grow exponentially in $d$ for generic Lipschitz functions —
this is the curse of dimensionality the universal approximation theorem
hides. Depth changes the picture for specific, structured functions: some
functions computable by a narrow, deep network require exponentially wider
shallow networks to approximate at all.

The canonical example is built from the **tent map** $T : [0, 1] \to [0,
1]$, $T(x) = 2\min(x, 1-x)$ — a "V" folded in half, mapping $[0,
\tfrac12]$ onto $[0,1]$ with slope $2$ and $[\tfrac12, 1]$ onto $[0,1]$ with
slope $-2$. $T$ is computed by two ReLU units: $T(x) = 2\operatorname{relu}(x)
- 4\operatorname{relu}(x - \tfrac12)$ for $x \in [0,1]$, as direct
substitution on each half confirms. Define the depth-$k$ **sawtooth**
$S_k = T \circ T \circ \cdots \circ T$ ($k$ compositions). Because $T$ maps
$[0,1]$ onto $[0,1]$ two-to-one, and does so on each of the pieces the
previous composition already created, $S_k$ has exactly $2^k$ monotone
linear pieces on $[0,1]$, each mapped affinely onto $[0,1]$: every additional
layer of $T$ doubles the piece count by folding the range back over the
whole domain. $S_k$ is computed **exactly** by a ReLU network with $k$
hidden layers of width $2$ — depth $k$, width $O(1)$ — by feeding each tent
map's output into the next tent map's two probes.

The matching lower bound is a counting argument on linear regions.

**Claim.** A ReLU network taking a scalar input, with hidden layers of
widths $n_1, \dots, n_L$ and a linear output layer, computes a function with
at most $\prod_{l=1}^{L} (n_l + 1)$ linear pieces on any interval.

**Proof.** Let $P_{l}$ be the number of pieces (maximal intervals of
affinity) of the vector-valued function computed by the units of layer $l$,
as a function of the scalar input, with $P_0 = 1$ (the input itself is
affine everywhere). Within one of the $P_{l-1}$ pieces of layer $l-1$'s
output, that output is a single affine function of $x$, so the pre-activation
of layer $l$, $W_l h^{(l-1)} + b_l$, is also affine in $x$ on that piece —
one affine function per unit. Each of the $n_l$ units can cross zero at most
once within that single piece, because a nonconstant affine function has one
root, so applying $\operatorname{relu}$ can introduce at most $n_l$ new
breakpoints inside that one piece, splitting it into at most $n_l + 1$
sub-pieces. Summing over the $P_{l-1}$ existing pieces gives $P_l \leq
P_{l-1}(n_l+1)$, and unrolling the recursion from $P_0 = 1$ gives $P_L \leq
\prod_{l=1}^L (n_l+1)$. The final linear layer recombines the last hidden
layer's units without introducing new breakpoints, so the network's output
has at most this many pieces. $\blacksquare$

Apply this to a **shallow** competitor: a depth-2 network (one hidden layer)
with width $n$ has at most $n+1$ pieces. Matching the sawtooth's $2^k$
pieces forces $n \geq 2^k - 1$ — width exponential in $k$ — while the deep
network achieves the same piece count with $2k$ total hidden units, linear
in $k$. This is a real separation, not merely a matter of the specific
construction: any network of depth $L$ with width bounded by a constant $w$
computes at most $(w+1)^L$ pieces by the claim above, so approximating the
sawtooth to within a constant fraction of its own pieces at fixed width
forces depth $L = \Omega(k / \log w)$ — the number of layers needed grows
with $k$ whenever the width is held fixed, which is the qualitative content
of the depth-separation results of this kind (Telgarsky, and Eldan and
Shamir for related constructions): what a network of depth $\Theta(k)$
computes with $O(1)$ width per layer, a network with only $O(1)$ depth needs
width exponential in $k$ to match.

## Cost and scaling

For a layer with input dimension $n_{\text{in}}$ and output dimension
$n_{\text{out}}$, the affine part costs $2\, n_{\text{in}} n_{\text{out}}$
flops for the matrix-vector product (one multiply and one add per term of
the sum) plus $n_{\text{out}}$ for the bias, and the nonlinearity costs
$O(n_{\text{out}})$ more — one comparison per unit for ReLU, a handful of
elementary operations per unit for the GELU approximation. None of this
depends on depth directly; it depends on the widths.

| Quantity | Formula | Notes |
|---|---|---|
| Parameters, one layer | $n_{\text{in}} n_{\text{out}} + n_{\text{out}}$ | Weights plus biases. |
| Parameters, $L$ layers | $\sum_{\ell=1}^{L} (n_{\ell-1} n_\ell + n_\ell)$ | Dominated by the widest adjacent pair. |
| Flops, one layer, ReLU | $2 n_{\text{in}} n_{\text{out}} + 2 n_{\text{out}}$ | Matmul, bias, one comparison per unit. |
| Flops, forward pass | $\sum_{\ell} \big(2 n_{\ell-1} n_\ell + O(n_\ell)\big)$ | Linear in depth, quadratic in width per layer. |

A network with $L$ layers of equal width $n$ has $O(Ln^2)$ parameters and
flops, so doubling the depth doubles the cost while doubling the width
quadruples it — a large part of why practical architectures favor
moderate width and enough depth to reach the function class they need,
rather than the reverse. The sawtooth section is the sharpest illustration:
reaching $2^k$ linear pieces costs $O(k)$ parameters at width $2$ and depth
$k$, against $O(2^k)$ parameters at depth $2$.

## Where this breaks

**The theorem does not tell you the width you need.** Cybenko and Hornik's
result is an existence statement; outside of hand-built cases like the
Lipschitz interpolant above, the required $m$ for a target $\varepsilon$ is
unknown and can be astronomically large. Treating "a two-layer network can
approximate anything" as a design principle ignores that the "anything" is
frequently infeasible in practice.

**Optimization is not part of the guarantee.** Nothing above says gradient
descent, from a random initialization, finds weights anywhere near the
approximation the theorem promises exists. Expressivity and learnability are
different questions, and a network can be expressive enough for a task while
remaining hard to train on it.

**Piecewise-linear networks are not smooth.** A ReLU network's output has
discontinuous derivatives at every breakpoint; this is invisible in the loss
during training but matters for downstream uses that need a smooth function
— certain physical simulators, or gradient-based optimization *of the
input* rather than the parameters, such as adversarial example search, where
kinks make second-order methods behave badly.

**Linear-region counts do not measure practical expressivity.** The
exponential bound above is an upper bound on the pieces a network *can*
produce, achieved by adversarially constructed weights; a network trained by
gradient descent on real data tends to realize vastly fewer effective
regions than its architecture permits, and linear-region counts correlate
only loosely with generalization or with any single practical metric of
"how complex is this function."

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
`mlcheck.finite_difference`, that a two-layer network built from two
`"identity"` layers collapses to the single affine map their product
predicts and inherits its rank bound, that `sawtooth_network(k)` produces
exactly $2^k$ linear pieces for several values of $k$ and matches the
closed-form sawtooth to machine precision rather than only approximately,
and that `parameter_count` and `flop_count` match hand-worked totals for a
stated four-layer architecture.
