---
title: Backpropagation as reverse-mode AD
slug: 16-backpropagation
part: "Part 3 — Neural networks"
summary: Backpropagation as reverse-mode automatic differentiation, with every vector-Jacobian product an MLP needs worked out.
minutes: 55
gpu: false
objectives:
  - Explain forward-mode and reverse-mode automatic differentiation as Jacobian-vector and vector-Jacobian products, and state their relative cost.
  - State the vector-Jacobian product of an affine layer, a pointwise nonlinearity, a matrix product, and softmax-cross-entropy.
  - State the memory-versus-recompute trade that checkpointing makes, and why it exists.
  - State the error floor of central-difference numerical differentiation and explain why it checks a gradient rather than computes one.
prereqs:
  - 15-multilayer-perceptrons
exercise: 16-backpropagation
---

# Backpropagation as reverse-mode AD

Every gradient this course has derived so far — the normal equations, the
logistic loss, the M-step of a Gaussian mixture — was worked out by hand,
once, for one specific function. Training a network with millions of
parameters and an architecture that changes weekly cannot work that way.
What makes it unnecessary is a fact about the chain rule that has nothing
to do with neural networks specifically. The chain rule is a product of
Jacobians, and matrix multiplication is associative. One particular way of
parenthesising that product computes the gradient of a scalar with respect
to every input at once, for the cost of a single extra pass over the
computation. That fact is backpropagation. This lesson derives it from the
chain rule directly, then works out the specific vector-Jacobian products
an MLP needs, so that the exercise's autodiff engine has a hand-checked
derivation behind every operation it implements.

## What you already know

- The multivariable chain rule: for $y = f(g(x))$, the Jacobian is
  $J_y = J_f(g(x))\, J_g(x)$, a product of Jacobians that extends to any
  number of compositions.
- The Jacobian of an affine map $x \mapsto Wx + b$ is $W$, constant in $x$.
- A multilayer perceptron is an alternating composition of affine maps and
  pointwise nonlinearities, from the previous lesson.
- Matrix multiplication is associative but not commutative: $(AB)C = A(BC)$,
  and the two sides can cost very different amounts of arithmetic despite
  being equal.
- Floating-point numbers carry a relative rounding error of about
  $\varepsilon_{\text{machine}} \approx 2.2 \times 10^{-16}$ in double
  precision, on every arithmetic operation.

## The computational graph

Any expression built from primitive operations — addition, multiplication,
matrix products, elementwise nonlinearities — is a directed acyclic graph:
one node per intermediate value, one edge from each value to every value
computed from it. Evaluating the expression is a forward pass over the
graph in topological order. Every automatic differentiation method operates
on this graph rather than on the symbolic expression, which lets it handle
control flow and data-dependent structure that a purely symbolic
differentiator would struggle with.

Write the graph as a chain of vector-valued stages for the purpose of
deriving cost, $x_0 \to x_1 \to \cdots \to x_L$, with $x_\ell =
f_\ell(x_{\ell-1})$ and Jacobian $J_\ell = \partial x_\ell / \partial
x_{\ell-1}$. A real graph is not a chain — nodes can have multiple parents
and children — but the chain captures the cost argument exactly, and a node
with multiple children accumulates one gradient contribution from each.

## Forward mode and reverse mode

The end-to-end Jacobian from $x_0$ to $x_L$ is the chain rule's product,

$$
J = J_L J_{L-1} \cdots J_1 ,
$$

and every automatic differentiation method contracts this product against a
vector without ever forming any $J_\ell$ densely. The two modes differ in
which vector they carry and which side of the product they multiply from.

**Forward mode** fixes a direction $v \in \mathbb{R}^{n_0}$ and propagates
$Jv$ from the right, $Jv = J_L(J_{L-1}(\cdots(J_1 v)\cdots))$. Each step is
a **Jacobian-vector product** (JVP), $u_\ell = J_\ell u_{\ell-1}$ with
$u_0 = v$, computed alongside the forward pass: for an affine layer the JVP
is $W_\ell u_{\ell-1}$, and for a pointwise nonlinearity it is
$f'(x_{\ell-1}) \odot u_{\ell-1}$. One sweep produces $Jv$ — one column of
$J$ — at a cost within a constant factor of the forward pass. Recovering
the full Jacobian this way costs $n_0$ sweeps, one per input dimension.

**Reverse mode** fixes a cotangent $u \in \mathbb{R}^{n_L}$, typically
$u=1$ when $x_L$ is a scalar loss, and propagates $u^\top J$ from the left,
$u^\top J = (((u^\top J_L)\, J_{L-1}) \cdots) J_1$. Each step is a
**vector-Jacobian product** (VJP), $\bar{x}_{\ell-1} = \bar{x}_\ell J_\ell$,
computed backward through the graph once every $x_\ell$ is known. One
sweep produces the *entire* gradient of a scalar $x_L$ with respect to
every coordinate of $x_0$, at a cost within a constant factor of the
forward pass.

**The cost argument.** These are the same associative product, contracted
from opposite ends, and the asymmetry in what one sweep buys is the entire
reason backpropagation is reverse mode. To get the full gradient of a
scalar loss with respect to $n_0$ inputs, forward mode needs $n_0$ sweeps —
one per basis direction — while reverse mode needs exactly one, because the
output is already scalar. A network with millions of parameters and one
scalar loss is the extreme case this asymmetry was built for: forward mode
would cost millions of forward passes, reverse mode costs one backward
pass. The asymmetry runs the other way for a low-dimensional input and a
high-dimensional output. That is why both modes exist in a mature autodiff
system: the choice between them depends on the function's input and output
dimensions, not a universal preference.

## The vector-Jacobian products an MLP needs

Backpropagation through a specific architecture is registering the VJP of
each primitive operation, then letting the chain of VJPs run itself. Four
suffice for an MLP trained with a softmax classifier.

### Affine layer

For $y = Wx + b$ with $x \in \mathbb{R}^n$, $W \in \mathbb{R}^{m \times n}$,
and a cotangent $\bar y = \partial L/\partial y$ for some scalar $L$
downstream, differentiating $y_i = \sum_j W_{ij}x_j + b_i$ gives

$$
\bar x = W^\top \bar y, \qquad \bar W = \bar y\, x^\top, \qquad \bar b =
\bar y .
$$

Every VJP here costs the same order of flops as the forward matrix-vector
product that produced $y$: an affine layer's backward pass is not free, but
it is not more expensive in order of magnitude either.

### Pointwise nonlinearity

For $y = f(x)$ applied elementwise, the Jacobian is diagonal
($\partial y_i/\partial x_j = f'(x_i)$ for $i=j$, zero otherwise), so

$$
\bar x = \bar y \odot f'(x) ,
$$

an elementwise product rather than a matrix-vector product. A diagonal
Jacobian turns the general VJP formula into an $O(n)$ operation instead of
$O(n^2)$. This is why activations are cheap relative to the affine layers
around them, regardless of how expensive $f$ itself is to evaluate.

### Matrix product

For $C = AB$ with $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k
\times n}$, and cotangent $\bar C$, differentiating $C_{ij} = \sum_l
A_{il}B_{lj}$ gives

$$
\bar A = \bar C B^\top, \qquad \bar B = A^\top \bar C .
$$

This is the batched generalisation of the affine layer's VJP: a batch of
inputs stacked into a matrix multiplied by a shared weight matrix is
exactly this operation. It costs two matrix products the same shape as the
forward one, the origin of the folklore that a backward pass costs about
twice a forward pass for a matmul-dominated network.

### Softmax combined with cross-entropy

Consider logits $z \in \mathbb{R}^c$, softmax probabilities $p_k = e^{z_k} /
\sum_{k'} e^{z_{k'}}$, and a one-hot label $y$ with cross-entropy loss
$L = -\sum_k y_k \log p_k$. The two Jacobians involved are individually
messy: softmax's Jacobian $\partial p_k/\partial z_j = p_k(\delta_{kj} -
p_j)$ is a full matrix, not diagonal. Their product collapses regardless. Chaining
them and using $\sum_k y_k = 1$ (because $y$ is one-hot) gives

$$
\bar z = p - y .
$$

The softmax Jacobian's off-diagonal terms and cross-entropy's division by
$p_k$ both vanish in the combination, leaving the cheapest possible
formula: predicted probabilities minus the one-hot target, with no division
anywhere. Implementing softmax and cross-entropy as two separate
differentiable operations, and letting a generic chain rule combine their
VJPs, would reintroduce the division by $p_k$. Recovering this formula
would then require a cancellation that floating point does not perform.
Without it, the computation can lose precision, or divide by a probability
that underflowed to zero. Deriving and
implementing the combined VJP directly, as the exercise does, is the
numerically correct choice for exactly this reason.

## Checkpointing

Reverse mode needs every intermediate $x_\ell$ from the forward pass
available when its VJP runs, because the pointwise nonlinearity's VJP needs
$f'(x_{\ell-1})$ and the matmul VJPs need the original operands. Storing
every activation in a network with $L$ layers of width $n$ and batch size
$B$ costs $O(LnB)$ memory, linear in depth. For a deep enough network, or a
long enough sequence, this becomes the binding constraint before compute
does.

**Checkpointing** trades memory for recomputation. Store only a subset of
the activations, for instance one in every $\sqrt{L}$ layers. When the
backward pass needs one that was not stored, recompute it by re-running the
forward pass from the nearest checkpoint. Storing $O(\sqrt{L})$ checkpoints
reduces memory from $O(L)$ to $O(\sqrt{L})$ at the cost of one extra forward
pass overall, roughly a 33% increase in total compute for a network
otherwise dominated by one forward and one backward pass. This is a real
trade, not a free lunch: the right number of checkpoints depends on whether
the model is memory-bound or compute-bound on the hardware it runs on.

## Why numerical differentiation is a check, not a method

A central-difference approximation to a partial derivative,

$$
\frac{\partial L}{\partial x_i} \approx \frac{L(x + h e_i) - L(x - h e_i)}
{2h} ,
$$

costs two evaluations of $L$ per coordinate of $x$: $O(n)$ full forward
passes to approximate a gradient with $n$ entries, against reverse mode's
single backward pass for all $n$ entries at once. For a network with
millions of parameters this is a different complexity class. That is why
finite differences are never used to train anything, but are used
constantly to check a hand-derived or hand-implemented gradient on a small
example. This lesson's exercise checks every gradient exactly that way.

The approximation also has an accuracy floor that no choice of $h$ can
beat, from two errors that trade off against each other. The **truncation
error** is $O(h^2 |L'''|)$ — central differences cancel the $O(h)$ term a
one-sided difference would leave, which is why they are preferred at the
same step size. Evaluating $L$ in floating point carries an absolute error
of about $\varepsilon_{\text{machine}}|L|$, and dividing by $2h$ amplifies
it to a **rounding error** of $O(\varepsilon_{\text{machine}}/h)$. Balancing
the two, $h^2 \propto \varepsilon_{\text{machine}}/h$, gives $h^\star
\propto \varepsilon_{\text{machine}}^{1/3}$ and an irreducible error floor
of

$$
O\big(\varepsilon_{\text{machine}}^{2/3}\big)
$$

at the optimal step, about $10^{-11}$ in double precision. A one-sided
forward difference has truncation error $O(h)$ rather than $O(h^2)$;
balancing it against rounding gives $h^\star \propto
\varepsilon_{\text{machine}}^{1/2}$ and a worse floor of
$O(\varepsilon_{\text{machine}}^{1/2}) \approx 10^{-8}$. Central differences
buy roughly three more digits of accuracy for the same reason QR beats the
normal equations in lesson 2: cancelling a lower-order error term before it
competes with rounding. No finite-difference scheme reaches machine
precision, which is why the exercise's checks compare analytic gradients to
finite differences with a tolerance like $10^{-5}$ rather than $10^{-12}$.

## Cost and scaling

Let $L$ be the number of layers, $n$ the largest layer width, and $B$ the
batch size.

| Quantity | Forward pass | Backward pass |
|---|---|---|
| Flops, one affine layer | $2n^2 B$ | $\approx 4n^2 B$ (two matmuls of the same size) |
| Flops, one pointwise nonlinearity | $O(nB)$ | $O(nB)$ |
| Memory for activations | $O(LnB)$ stored for reuse | Same, unless checkpointed |
| Sweeps for a full gradient, reverse mode | — | $1$ |
| Sweeps for a full gradient, forward mode | — | $n_0$ (the input dimension) |

A full training step costs about three times the forward pass alone — one
forward, roughly two forward-equivalents of backward — the standard rule of
thumb for estimating training compute from a model's parameter count.

## Where this breaks

**Non-differentiable operations.** An argmax, a hard threshold, or a
sampling step has no useful Jacobian: it is piecewise constant with a
derivative of zero almost everywhere, or its derivative is undefined
outright. Straight-through estimators, the Gumbel-softmax relaxation, and
REINFORCE-style score-function estimators manufacture a usable gradient
signal through an operation that does not have a true one, each with its
own bias-variance trade-off.

**Vanishing and exploding products.** The chain rule's product $J_L \cdots
J_1$ can shrink or grow geometrically in the number of factors when the
$J_\ell$ are not close to orthogonal. This is the mechanism behind
vanishing and exploding gradients in deep and recurrent networks. Reverse mode computes
an exact product of many matrices, and that kind of quantity is
numerically fragile by nature.

**Second-order information is not free.** Reverse mode gives the gradient
at the cost of roughly one forward pass, but the Hessian is a different
object. Computing it densely costs $O(n_0)$ reverse-mode sweeps, one per
Hessian-vector product. An application that needs only a single
Hessian-vector product is the exception: Pearlmutter's trick computes that
with two extra passes. Assuming second-order methods inherit backpropagation's cheapness
is a common and costly mistake.

**Memory, not compute, is often the binding constraint.** Storing
activations for backward passes scales with depth and batch size in a way
compute does not have to. That is why checkpointing exists, and why the
largest models are frequently limited by accelerator memory before they
are limited by flops available.

## The exercise

You build a small reverse-mode autodiff engine around a `Tensor` class that
holds a value, a gradient, and a backward closure. It implements `__add__`,
`__mul__`, `__matmul__`, `relu`, `sum`, and `softmax_cross_entropy`. The
last of these uses the combined VJP derived above, rather than composing
separate softmax and cross-entropy nodes. `backward()` runs a topological sort over
the graph reachable through each tensor's recorded children, then calls
every node's backward closure in reverse order. A tensor used more than
once accumulates every contribution to its gradient before that gradient
is used further back. `mlp_loss_and_grad` then builds an MLP's forward pass
out of these operations and reads its gradients back off the parameter
tensors after one call to `backward()`.

The checks verify every operation's gradient against
`mlcheck.finite_difference`, including a broadcast bias in `__add__`. They
verify that a tensor used twice within a single operation accumulates
rather than overwrites its gradient, and that a diamond-shaped graph — one
tensor feeding two paths that later recombine — backpropagates correctly
through both paths. They also verify that `softmax_cross_entropy`'s
gradient matches the closed-form $p - y$ derived above, and that
`mlp_loss_and_grad` matches finite differences end to end on a small
two-layer network.
