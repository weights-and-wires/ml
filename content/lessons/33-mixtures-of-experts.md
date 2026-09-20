---
title: Mixtures of experts
slug: 33-mixtures-of-experts
part: "Part 6 — Large language models"
summary: Sparse routing that decouples parameter count from flops per token, the load-balancing loss that keeps it from collapsing, and what dropping tokens costs.
minutes: 60
gpu: false
objectives:
  - Explain how sparse MoE decouples total parameter count from compute per token, and compute the active-versus-total parameter ratio for a given configuration.
  - Derive the router's gradient, and show why the gate weight must multiply the expert output for the router to receive a gradient at all.
  - Derive the load-balancing auxiliary loss and show it is minimised by uniform routing and maximised by a collapsed one.
  - Describe capacity factors, token dropping, expert parallelism's communication pattern, and where sparse routing hurts memory, batching, and fine-tuning.
prereqs:
  - 26-the-transformer-block
exercise: 33-mixtures-of-experts
---

# Mixtures of experts

The feed-forward sublayer in a transformer block spends the same flops on
every token: each position is multiplied by the same two weight matrices,
regardless of whether the token is a common function word or a rare technical
term that needs specialised processing. Making the model more capable by
widening that sublayer buys more parameters and more flops together, in
lockstep, because every added parameter sits on the path every token takes.

A sparse mixture of experts breaks that lockstep. Replace the single
feed-forward network with $N$ copies of it, add a router that picks a small
subset of $k$ of them per token, and run only the chosen experts on that
token. Total parameter count grows with $N$; compute per token grows with
$k$. Once $k \ll N$, a model can hold far more parameters — and, empirically,
far more knowledge — than a dense model trained with the same compute budget
per token, at the cost of a routing decision that has to be learned, kept
balanced across experts, and paid for in communication once the experts live
on different devices. This lesson derives the router's gradient, the loss
that keeps its assignments from collapsing onto a handful of experts, and the
capacity mechanics that turn a ragged, data-dependent routing pattern into
something you can batch on fixed-size hardware.

## What you already know

- The transformer block's feed-forward sublayer is a per-token function,
  applied identically and independently to every position (lesson 26).
- The softmax Jacobian: for $s = \operatorname{softmax}(z)$,
  $\partial s_i / \partial z_j = s_i(\delta_{ij} - s_j)$.
- A matrix multiply of shapes $(m, k)$ by $(k, n)$ costs $2mnk$ flops.
- An indicator function of a continuous argument — such as "is this the
  largest of these $N$ numbers" — is piecewise constant and has zero
  derivative everywhere it is differentiable at all.
- Convexity of $x \mapsto x^2$ and the Cauchy-Schwarz inequality.

## The sparse MoE layer

Let $x \in \mathbb{R}^{d}$ be one token's representation at the point where a
dense block would apply its feed-forward network. A router with weights
$W_{\text{gate}} \in \mathbb{R}^{d \times N}$ produces logits $z =
x^{\top} W_{\text{gate}} \in \mathbb{R}^N$ over $N$ experts $E_1, \dots, E_N$,
each an ordinary feed-forward network of the kind lesson 26 built. Convert
the logits to probabilities with a softmax, $s = \operatorname{softmax}(z)$,
select the indices $\mathcal{K}(x) \subset \{1, \dots, N\}$ of the $k$
largest entries of $s$, and renormalise the probabilities on that subset:

$$
g_i(x) = \frac{s_i}{\sum_{j \in \mathcal{K}(x)} s_j} \quad \text{for } i \in \mathcal{K}(x), \qquad g_i(x) = 0 \text{ otherwise.}
$$

The layer's output is the gate-weighted sum of the selected experts' outputs:

$$
y(x) = \sum_{i \in \mathcal{K}(x)} g_i(x) \, E_i(x) .
$$

Every expert not in $\mathcal{K}(x)$ costs nothing for this token: it is
never evaluated. This is the whole mechanism. Everything else in this lesson
is about making the router learnable, the routing balanced, and the ragged
per-expert batches tractable on hardware built for fixed shapes.

## Why the gate must multiply the output

The router's parameters, $W_{\text{gate}}$, influence the output only through
$g_i(x)$ — the choice of $\mathcal{K}(x)$ and the weights on it. Suppose
instead the layer used hard, unweighted routing, $y(x) = \sum_{i \in
\mathcal{K}(x)} E_i(x)$, dropping $g_i$ entirely. Membership in
$\mathcal{K}(x)$ is decided by comparing logits, a discrete, piecewise
constant function of $z$: for almost every $x$, a small perturbation of
$W_{\text{gate}}$ leaves $\mathcal{K}(x)$ unchanged, so $y(x)$ is locally
constant in $W_{\text{gate}}$ and $\partial y / \partial W_{\text{gate}} = 0$
almost everywhere. Backpropagation would deliver an exact zero gradient to
the router. Multiplying by the continuous gate weight $g_i(x)$ is what gives
the router anything to learn from through ordinary backpropagation, and the
gradient this buys is worth deriving in full, because it explains the
load-balancing failure in the next section.

Fix $x$ and treat the top-$k$ membership $\mathcal{K}(x)$ as locally constant
(true away from a measure-zero set of ties). On $\mathcal{K}(x)$, $g$ is a
softmax over the restricted set of logits $\{z_i : i \in \mathcal{K}(x)\}$, so
the same Jacobian you already know applies within the subset:

$$
\frac{\partial g_i}{\partial z_j} = g_i(\delta_{ij} - g_j) \quad \text{for } i, j \in \mathcal{K}(x), \qquad \frac{\partial g_i}{\partial z_j} = 0 \text{ for } j \notin \mathcal{K}(x) .
$$

That second case matters as much as the first: a logit for an expert outside
the current top-$k$ has no effect on this token's output at all, so it gets
no gradient from this token, from any route, hard or soft. Let $\bar y =
\partial \mathcal{L} / \partial y \in \mathbb{R}^{d}$ be the loss gradient
flowing back into this layer's output, and define $c_i = \bar y^{\top}
E_i(x)$ for $i \in \mathcal{K}(x)$ — how much the loss would fall if $g_i$
increased, holding everything else fixed. Since $\partial y/\partial g_i =
E_i(x)$, the chain rule through the softmax gives, for $j \in \mathcal{K}(x)$,

$$
\frac{\partial \mathcal{L}}{\partial z_j}
= \sum_{i \in \mathcal{K}(x)} c_i \frac{\partial g_i}{\partial z_j}
= \sum_{i \in \mathcal{K}(x)} c_i \, g_i(\delta_{ij} - g_j)
= g_j \Big( c_j - \sum_{i \in \mathcal{K}(x)} g_i c_i \Big) ,
$$

and $\partial \mathcal{L}/\partial z_j = 0$ for $j \notin \mathcal{K}(x)$.
Writing $\bar c = \sum_i g_i c_i$ for the gate-weighted average of $c$ over
the selected experts, the router logit for expert $j$ moves toward
increasing $g_j$ exactly when expert $j$'s output correlates with reducing
the loss more than that average, $c_j > \bar c$, and away from it otherwise.
The gradient with respect to $W_{\text{gate}}$ follows from $z = x^{\top}
W_{\text{gate}}$: column $j$ of $\partial \mathcal{L}/\partial W_{\text{gate}}$
is $x \cdot \partial \mathcal{L}/\partial z_j$, zero for every expert this
token did not select.

## Load balancing and the collapse it prevents

That last fact — an unselected expert gets no gradient from a token, ever —
is also the mechanism behind a well-documented failure. If, early in
training, one expert is even slightly more likely to be selected, it
receives more gradient updates from more tokens, which tends to make its
outputs more useful, which increases $c_i$ for that expert on future tokens
and makes it more likely to be selected again. The other $N-1$ experts,
selected rarely, receive gradient from few tokens, improve slowly if at all,
and become still less likely to be chosen. Nothing in the objective so far
resists this: the layer's loss only cares about the output $y(x)$, and a
handful of experts doing all the work produces a perfectly good $y(x)$. Left
alone, routing collapses onto a small subset of experts and the rest of the
model's parameters go unused, defeating the entire point of paying for $N$
experts.

The standard fix is an auxiliary loss added to the training objective,
computed per batch of $T$ tokens:

$$
\mathcal{L}_{\text{aux}} = \alpha N \sum_{i=1}^{N} f_i P_i ,
\qquad
f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[i \in \mathcal{K}(x_t)],
\qquad
P_i = \frac{1}{T} \sum_{t=1}^{T} s_i(x_t) .
$$

$f_i$ is the empirical fraction of tokens actually routed to expert $i$ — a
hard count, built from the same indicator function that made hard routing
non-differentiable, so $\partial f_i / \partial W_{\text{gate}} = 0$ almost
everywhere, exactly as before. $P_i$ is the mean of the full softmax
probability the router assigns to expert $i$, before any top-$k$ cutoff — a
smooth function of $W_{\text{gate}}$ with a well-defined gradient. The
product form uses each term for what it can do: $f_i$ is treated as a fixed
importance weight (detached from the computation graph) at the value it
happens to take in this batch, and $P_i$ is the only factor differentiated
through, so

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial W_{\text{gate}}} = \alpha N \sum_{i=1}^{N} f_i \, \frac{\partial P_i}{\partial W_{\text{gate}}} ,
$$

a gradient that pushes down $P_i$ in proportion to how much traffic $f_i$
that expert is already getting — exactly the corrective pressure the
self-reinforcing collapse needs, applied through the one factor that can
carry it.

Both $f$ and $P$ are distributions over the $N$ experts, each summing to 1,
and in practice they track each other closely, since $f_i$ is the frequency
with which expert $i$ has the largest $s_i$. Treating them as one
distribution $p$ makes the two extremes easy to pin down. By Cauchy-Schwarz,
$\big(\sum_i p_i\big)^2 \leq N \sum_i p_i^2$, so $\sum_i p_i^2 \geq 1/N$,
with equality exactly when $p$ is uniform — the case $f_i = P_i = 1/N$ for
every expert. At the other extreme, a collapsed distribution with $p_i = 1$
for one expert and 0 elsewhere gives $\sum_i p_i^2 = 1$, the maximum a sum of
squared probabilities summing to 1 can reach, since $\sum_i p_i^2 \leq
\big(\max_i p_i\big) \sum_i p_i \leq 1$. So $\sum_i f_i P_i$ ranges over
$[1/N, 1]$, and

$$
\mathcal{L}_{\text{aux}} \in [\alpha, \, \alpha N] ,
$$

minimised at $\alpha$ by uniform routing and maximised at $\alpha N$ by total
collapse onto one expert. The factor of $N$ in front is not decorative: it
cancels the $1/N$ that a perfectly balanced router would otherwise contribute,
so a well-balanced layer costs the fixed amount $\alpha$ regardless of how
many experts it has, and the auxiliary loss keeps the same regularising
strength as $N$ grows instead of shrinking toward zero.

## Capacity factors and token dropping

Training and inference batch tokens into fixed-size arrays per expert, but
routing produces a data-dependent, ragged number of tokens per expert. Fixing
that requires a capacity: the maximum number of tokens an expert accepts in
one batch. With $T$ tokens, $k$ slots each, and $N$ experts, the load a
perfectly uniform router would produce per expert is $Tk/N$, so a capacity
factor $\gamma$ sets

$$
\text{capacity} = \Big\lceil \gamma \, \frac{Tk}{N} \Big\rceil .
$$

$\gamma = 1$ gives an expert exactly the slots it would need under perfectly
even routing; real routing is never perfectly even within one finite batch,
so even a well-balanced router drops some tokens at $\gamma = 1$. Tokens
routed to an expert already at capacity are dropped from that expert: their
slot contributes nothing to $y(x)$. Since the sublayer sits inside a residual
connection, $y(x) = x + \text{FFN}(x)$ at the block level, a token dropped
from every one of its $k$ slots does not error or stall — it passes through
the block as the identity, silently skipping the sublayer's transformation
for that token at that layer. That is a real capacity cost, not a graceful
approximation: a token that needed the expert computation does not get it,
and nothing downstream is told this happened. Raising $\gamma$ gives every
expert more slack and strictly reduces how many tokens spill over, at the
cost of the extra compute and memory needed to size every expert's buffer for
the larger capacity, whether or not it is filled.

## Expert parallelism

Once $N$ is large enough that all experts cannot fit on one device, experts
are sharded across devices — expert parallelism — and each token's hidden
state must travel to wherever its selected experts live. This is an
**all-to-all** communication step: every device sends each token to the
device holding the token's assigned expert, that device runs the expert
computation, and a second all-to-all sends the results back to the token's
original device to be combined into $y(x)$. The volume moved scales with the
number of routed tokens times $d_{\text{model}}$ in each direction, and
unlike the matmuls in the rest of the block, this traffic does not overlap
well with compute on the same device, so small batches or low bandwidth
between devices turn expert parallelism into a communication-bound stage
that can dominate wall-clock time even though the flops per token stayed
small by design.

## Router z-loss and shared experts

Two further devices are common. The router **z-loss** penalises the
magnitude of the logits themselves,

$$
\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \Big(\log \sum_{i=1}^N \exp(z_i(x_t))\Big)^2 ,
$$

added with a small coefficient. Large logits make the softmax's exponentials
overflow more readily in reduced precision and make the router's decisions
oversensitive to small weight changes; penalising the log-sum-exp keeps the
logits, and the softmax built on them, well conditioned without changing
which expert wins. A **shared expert** — one feed-forward network applied to
every token in addition to whichever $k$ routed experts are selected — gives
every token a fixed, router-independent computation to fall back on. It
absorbs features common across the whole data distribution so the routed
experts specialise instead of re-learning shared structure, and it makes the
output less sensitive to a routing mistake, at the fixed cost of that one
expert's flops on every token, whether the layer sparsifies or not.

## Active-versus-total parameters, a worked count

With $N$ experts of $P_{\text{expert}}$ parameters each, $k$ selected per
token, and $P_{\text{shared}}$ parameters in always-on experts,

$$
P_{\text{total}} = N P_{\text{expert}} + P_{\text{shared}}, \qquad
P_{\text{active}} = k P_{\text{expert}} + P_{\text{shared}} ,
$$

ignoring the router's $d_{\text{model}} \times N$ weights, which are
negligible next to the experts themselves. Take $N = 8$ experts at
$P_{\text{expert}} = 100$ million parameters, $k = 2$, and one shared expert
of the same size: $P_{\text{total}} = 8 \times 100\text{M} + 100\text{M} =
900$ million, while $P_{\text{active}} = 2 \times 100\text{M} + 100\text{M} =
300$ million. A token's forward pass costs the flops of a 300-million
parameter dense feed-forward sublayer, while the model carries the capacity
of a 900-million parameter one — an active-to-total ratio of $1/3$ that
improves as $k/N$ shrinks.

## Cost and scaling

| Quantity | Dense feed-forward | Sparse MoE |
|---|---|---|
| Parameters | $\Theta(d_{\text{model}} \, d_{\text{ff}})$ | $\Theta(N \, d_{\text{model}} \, d_{\text{ff}})$ |
| Flops per token | $\Theta(d_{\text{model}} \, d_{\text{ff}})$ | $\Theta(k \, d_{\text{model}} \, d_{\text{ff}})$ |
| Memory to serve one token | one expert's weights | all $N$ experts' weights, unless offloaded |
| Cross-device traffic | none beyond the block's own activations | one all-to-all per direction, per MoE layer |

The parameter row and the flops row are the entire trade the layer makes:
parameters scale with $N$, flops with $k$, and the gap between them is the
reason to use sparse routing at all. Nothing in that trade reduces the
memory needed to hold the model, which scales with $N$ regardless of $k$.

## Where this breaks

**Memory.** Every expert must be reachable — resident in accelerator memory
or one hop away — even though a given token touches only $k$ of them, so a
sparse model's memory footprint tracks its total parameter count, not its
active count. A model with the inference flops of a modest dense model can
still need the memory of a much larger one.

**Batching.** Each expert only ever sees the tokens routed to it, so at small
global batch sizes an individual expert's local batch can be too small to
use accelerator matrix units efficiently, and the imbalance between experts
under real, non-uniform routing widens that gap further: the busiest expert
runs a full batch while others run nearly empty ones, and the layer runs as
slow as its slowest expert's batch.

**Fine-tuning instability.** A router trained for a broad pretraining
distribution can shift sharply on a narrow fine-tuning set, concentrating
traffic on the experts that happen to fit the new domain and reintroducing
the collapse the load-balancing loss controlled during pretraining. Because
top-$k$ membership is a discrete decision, a small parameter update near a
decision boundary can flip which experts a token uses, which changes the
loss surface discontinuously and can destabilise fine-tuning runs that would
be smooth for a dense model. Common responses include freezing the router,
fine-tuning only a subset of experts, or dropping the auxiliary loss
altogether once the routing pattern has settled.

Mixtures of experts attack one axis of the cost of scaling a transformer:
parameter count against flops per token, at a fixed sequence length. State
space models and linear attention attack the other axis — the growth of
attention's cost with sequence length itself — by replacing the softmax
attention sublayer with a recurrence or a kernel form whose per-token
inference cost stays constant as context grows, instead of a mixture of
per-token feed-forward networks. The two are complementary rather than
competing: a model can route sparsely through its feed-forward sublayers and
mix tokens sub-quadratically through its sequence sublayers in the same
architecture, and production systems increasingly do both.

## The exercise

You implement the router, the auxiliary loss, and the capacity mechanics
this lesson derives. Write `router(x, W_gate, k)`, returning the top-$k$
expert indices per token, their renormalised gate weights, and the full
softmax probabilities before the top-$k$ cutoff. Write
`load_balancing_loss(expert_indices, gate_probs, n_experts)`, computing $N
\sum_i f_i P_i$ from the router's own outputs. Write `moe_forward(x, W_gate,
experts, k, capacity_factor)`, which runs the full sparse layer with a token
capacity per expert and reports how many token-expert assignments it
dropped, and `active_parameters(n_experts, k, expert_params, shared_params)`
for the worked count above. Finally, write `router_backward`, the gradient
this lesson derives for the router's logits, given the downstream gradient
and every expert's output for every token.

The checks verify that the router's gate weights are non-negative and sum to
one per token; that `moe_forward` with $k$ equal to the number of experts and
no capacity limit reproduces a dense, softmax-weighted combination of every
expert exactly, which is the sparse layer's behaviour at the point where it
stops being sparse; that `load_balancing_loss` matches a hand computation of
$f$ and $P$, and that it is minimised by uniform routing and maximised by a
collapsed one, both checked numerically rather than only algebraically; that
raising the capacity factor monotonically reduces the number of dropped
assignments; that a token dropped from its only selected expert contributes
nothing to the output; that `active_parameters` matches a hand-computed
value; and that `router_backward` matches `mlcheck.finite_difference` on the
scalar objective $\sum_t \langle \bar y_t, y(x_t) \rangle$ this lesson's
derivation is built from.
