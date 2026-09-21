---
title: Mixtures of experts
slug: 33-mixtures-of-experts
part: "Part 6 — Large language models"
summary: Sparse routing that decouples parameter count from flops per token, the load-balancing loss that keeps it from collapsing, and what dropping tokens costs.
minutes: 50
gpu: false
objectives:
  - Explain how sparse MoE decouples total parameter count from compute per token, and compute the active-versus-total parameter ratio for a given configuration.
  - Derive the router's gradient, and show why the gate weight must multiply the expert output for the router to receive a gradient at all.
  - State the load-balancing auxiliary loss, and show it is minimised by uniform routing and maximised by a collapsed one.
  - Describe capacity factors, token dropping, and expert parallelism's communication pattern, and where sparse routing hurts memory, batching, and fine-tuning.
prereqs:
  - 26-the-transformer-block
exercise: 33-mixtures-of-experts
---

# Mixtures of experts

The feed-forward sublayer in a transformer block spends the same flops on
every token. Each position runs through the same two weight matrices,
whether it's a common function word or a rare technical term. Widening that
sublayer to make the model more capable buys more parameters and more flops
together, in lockstep, because every added parameter sits on the path every
token takes.

A sparse mixture of experts breaks that lockstep. Replace the single
feed-forward network with $N$ copies of it. Add a router that picks a small
subset of $k$ of them per token, and run only the chosen experts on that
token. Total parameter count grows with $N$; compute per token grows with
$k$. Once $k \ll N$, a model can hold far more parameters, and empirically
far more knowledge, than a dense model trained with the same compute budget
per token. That comes at the cost of a routing decision that has to be
learned, kept balanced across experts, and paid for in communication once
the experts live on different devices. This lesson derives the router's
gradient and the loss that keeps its assignments from collapsing onto a
handful of experts. It also derives the capacity mechanics that turn a
ragged, data-dependent routing pattern into something you can batch on
fixed-size hardware.

## What you already know

- The transformer block's feed-forward sublayer is a per-token function,
  applied identically and independently to every position (lesson 26).
- The softmax Jacobian: for $s = \operatorname{softmax}(z)$,
  $\partial s_i / \partial z_j = s_i(\delta_{ij} - s_j)$.
- A matrix multiply of shapes $(m, k)$ by $(k, n)$ costs $2mnk$ flops.
- An indicator function of a continuous argument, such as "is this the
  largest of these $N$ numbers," is piecewise constant and has zero
  derivative everywhere it's differentiable at all.
- Convexity of $x \mapsto x^2$, and the Cauchy-Schwarz inequality.

## The sparse MoE layer

Let $x \in \mathbb{R}^{d}$ be one token's representation at the point where a
dense block would apply its feed-forward network. A router with weights
$W_{\text{gate}} \in \mathbb{R}^{d \times N}$ produces logits $z =
x^{\top} W_{\text{gate}} \in \mathbb{R}^N$ over $N$ experts, and converts them
to probabilities with a softmax $s = \operatorname{softmax}(z)$. It then
selects the indices $\mathcal{K}(x)$ of the $k$ largest entries of $s$, and
renormalizes the probabilities on that subset:

$$
g_i(x) = \frac{s_i}{\sum_{j \in \mathcal{K}(x)} s_j} \quad \text{for } i \in \mathcal{K}(x), \qquad g_i(x) = 0 \text{ otherwise.}
$$

The layer's output is the gate-weighted sum of the selected experts' outputs,
$y(x) = \sum_{i \in \mathcal{K}(x)} g_i(x) \, E_i(x)$. Every expert not in
$\mathcal{K}(x)$ costs nothing for this token — it's never evaluated.

## Why the gate must multiply the output

The router's parameters influence the output only through $g_i(x)$: the
choice of $\mathcal{K}(x)$ and the weights on it. If the layer instead used
hard, unweighted routing, $y(x) = \sum_{i \in \mathcal{K}(x)} E_i(x)$,
membership in $\mathcal{K}(x)$ would be a discrete, piecewise constant
function of $z$. For almost every $x$, a small perturbation of
$W_{\text{gate}}$ leaves $\mathcal{K}(x)$ unchanged. So $y(x)$ is locally
constant in $W_{\text{gate}}$, and backpropagation delivers an exact zero
gradient to the router. Multiplying by the continuous gate weight $g_i(x)$
is what gives the router anything to learn from.

Fix $x$ and treat $\mathcal{K}(x)$ as locally constant. Let $\bar y =
\partial \mathcal{L} / \partial y$ be the loss gradient flowing into this
layer's output, and define $c_i = \bar y^{\top} E_i(x)$ for $i \in
\mathcal{K}(x)$: how much the loss would fall if $g_i$ increased. Since $g$
is a softmax over the restricted logits on $\mathcal{K}(x)$, the same
Jacobian above applies within the subset, and the chain rule gives, for $j
\in \mathcal{K}(x)$,

$$
\frac{\partial \mathcal{L}}{\partial z_j} = g_j \Big( c_j - \sum_{i \in \mathcal{K}(x)} g_i c_i \Big) ,
$$

and $\partial \mathcal{L}/\partial z_j = 0$ for $j \notin \mathcal{K}(x)$: an
expert outside the current top-$k$ gets no gradient from this token, from
any routing rule. Writing $\bar c$ for the gate-weighted average of $c$ over
the selected experts, the router logit for expert $j$ moves toward
increasing $g_j$ exactly when $c_j > \bar c$, and away from it otherwise.

## Load balancing and the collapse it prevents

That an unselected expert never gets gradient from a token is also the
mechanism behind a well-documented failure. If, early in training, one
expert is even slightly more likely to be selected, it receives more
gradient from more tokens. That improves its outputs, which raises $c_i$
for that expert on future tokens and makes it more likely to be selected
again. The other experts, chosen rarely, improve slowly if at all and become
still less likely to be picked. Left alone, routing collapses onto a handful
of experts and the rest of the model's parameters go unused.

The standard fix is an auxiliary loss computed per batch of $T$ tokens,

$$
\mathcal{L}_{\text{aux}} = \alpha N \sum_{i=1}^{N} f_i P_i ,
\qquad
f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[i \in \mathcal{K}(x_t)],
\qquad
P_i = \frac{1}{T} \sum_{t=1}^{T} s_i(x_t) .
$$

$f_i$ is the fraction of tokens actually routed to expert $i$, treated as a
fixed importance weight. $P_i$ is the mean full softmax probability assigned
to expert $i$, the only factor differentiated through. The gradient pushes
down $P_i$ in proportion to how much traffic $f_i$ that expert already gets
— the corrective pressure the collapse needs.

Both $f$ and $P$ are distributions over $N$ experts. By Cauchy-Schwarz,
$\sum_i p_i^2 \geq 1/N$ for any such distribution $p$, with equality exactly
at uniform routing, and $\sum_i p_i^2 \leq 1$, with equality exactly at total
collapse onto one expert. So $\sum_i f_i P_i$ ranges over $[1/N, 1]$, and

$$
\mathcal{L}_{\text{aux}} \in [\alpha, \, \alpha N] ,
$$

minimized at $\alpha$ by uniform routing and maximized at $\alpha N$ by
total collapse. The factor of $N$ cancels the $1/N$ a balanced router would
otherwise contribute, so the auxiliary loss keeps the same strength as $N$
grows instead of shrinking toward zero.

## Capacity factors and token dropping

Training and inference batch tokens into fixed-size arrays per expert, but
routing produces a ragged, data-dependent number of tokens per expert.
Fixing that needs a capacity, the maximum number of tokens an expert accepts
per batch. With $T$ tokens, $k$ slots each, and $N$ experts, a perfectly
uniform router sends $Tk/N$ tokens to each expert, so a capacity factor
$\gamma$ sets $\text{capacity} = \lceil \gamma \, Tk/N \rceil$. Tokens routed
to an expert already at capacity are dropped: their slot contributes nothing
to $y(x)$. Because the sublayer sits inside a residual connection, a token
dropped from every one of its $k$ slots passes through the block as the
identity. That is a real capacity cost, not a graceful approximation. Raising
$\gamma$ reduces how many tokens spill over, at the cost of sizing every
expert's buffer for the larger capacity whether or not it's filled.

## Expert parallelism, z-loss, and shared experts

Once $N$ is large enough that all experts can't fit on one device, experts
are sharded across devices. Each token's hidden state must then travel to
wherever its selected experts live. This all-to-all step sends each token to
the device holding its assigned expert and sends the results back to be
combined into $y(x)$. That traffic scales with the number of routed tokens
times $d_{\text{model}}$ in each direction, and doesn't overlap well with
compute. So small batches or low inter-device bandwidth can make expert
parallelism a communication-bound stage, even though the flops per token
stayed small by design.

Two further devices are common. The router **z-loss** penalizes the
log-sum-exp of the router logits, keeping them well conditioned in reduced
precision without changing which expert wins. A **shared expert** is one
feed-forward network applied to every token in addition to whichever $k$
routed experts are selected. It gives every token a fixed computation to fall
back on, letting the routed experts specialize instead of relearning shared
structure. This comes at the fixed cost of that expert's flops on every
token.

## Active-versus-total parameters, a worked count

With $N$ experts of $P_{\text{expert}}$ parameters each, $k$ selected per
token, and $P_{\text{shared}}$ parameters in always-on experts,

$$
P_{\text{total}} = N P_{\text{expert}} + P_{\text{shared}}, \qquad
P_{\text{active}} = k P_{\text{expert}} + P_{\text{shared}} ,
$$

ignoring the router's negligible $d_{\text{model}} \times N$ weights. Take
$N = 8$ experts at $P_{\text{expert}} = 100$ million parameters, $k = 2$,
and one shared expert of the same size: $P_{\text{total}} = 900$ million,
while $P_{\text{active}} = 300$ million. A token's forward pass costs the
flops of a 300-million-parameter dense sublayer, while the model carries the
capacity of a 900-million-parameter one. That is an active-to-total ratio of
$1/3$ that improves as $k/N$ shrinks.

## Cost and scaling

| Quantity | Dense feed-forward | Sparse MoE |
|---|---|---|
| Parameters | $\Theta(d_{\text{model}} \, d_{\text{ff}})$ | $\Theta(N \, d_{\text{model}} \, d_{\text{ff}})$ |
| Flops per token | $\Theta(d_{\text{model}} \, d_{\text{ff}})$ | $\Theta(k \, d_{\text{model}} \, d_{\text{ff}})$ |
| Memory to serve one token | one expert's weights | all $N$ experts' weights, unless offloaded |
| Cross-device traffic | none beyond the block's own activations | one all-to-all per direction, per MoE layer |

The parameter row and the flops row are the entire trade the layer makes.
Parameters scale with $N$, flops with $k$, and the gap between them is the
reason to use sparse routing at all. Nothing in that trade reduces the
memory needed to hold the model, which scales with $N$ regardless of $k$.

## Where this breaks

**Memory.** Every expert must stay reachable, resident in accelerator memory
or one hop away, even though a given token touches only $k$ of them. So a
sparse model's memory footprint tracks its total parameter count, not its
active count.

**Batching.** Each expert sees only the tokens routed to it. So at small
batch sizes an individual expert's local batch can be too small to use
accelerator matrix units efficiently. Real, non-uniform routing widens
that gap further: the layer runs as slow as its busiest expert's batch.

**Fine-tuning instability.** A router trained for a broad pretraining
distribution can shift sharply on a narrow fine-tuning set. Doing so
concentrates traffic on the experts that fit the new domain and reintroduces
the collapse the load-balancing loss controlled during pretraining. Because
top-$k$ membership is discrete, a small parameter update near a decision
boundary can flip which experts a token uses, changing the loss surface
discontinuously. Common responses include freezing the router, fine-tuning
only a subset of experts, or dropping the auxiliary loss once routing has
settled.

Mixtures of experts attack one axis of the cost of scaling a transformer:
parameter count against flops per token, at a fixed sequence length. State
space models and linear attention attack the other axis, the growth of
attention's cost with sequence length, and the two are complementary rather
than competing.

## The exercise

You implement the router, the auxiliary loss, and the capacity mechanics
this lesson derives. Write `router(x, W_gate, k)`, returning the top-$k$
expert indices per token, their renormalized gate weights, and the full
softmax probabilities before the top-$k$ cutoff. Write
`load_balancing_loss(expert_indices, gate_probs, n_experts)`, computing $N
\sum_i f_i P_i$ from the router's own outputs. Write `moe_forward(x, W_gate,
experts, k, capacity_factor)`, which runs the full sparse layer with a token
capacity per expert and reports how many token-expert assignments it
dropped. Write `active_parameters(n_experts, k, expert_params, shared_params)`
for the worked count above. Finally, write `router_backward`, the gradient
this lesson derives for the router's logits, given the downstream gradient
and every expert's output for every token.

The checks verify that the router's gate weights are non-negative and sum to
one per token. They confirm that `moe_forward` with $k$ equal to the number
of experts and no capacity limit reproduces a dense, softmax-weighted
combination of every expert exactly, the sparse layer's behavior at the
point where it stops being sparse. They check that `load_balancing_loss`
matches a hand computation of $f$ and $P$, and is minimized by uniform
routing and maximized by a collapsed one, both checked numerically rather
than only algebraically. They verify that raising the capacity factor
monotonically reduces the number of dropped assignments, that a token
dropped from its only selected expert contributes nothing to the output,
and that `active_parameters` matches a hand-computed value. Finally, they
check that `router_backward` matches `mlcheck.finite_difference` on the
scalar objective $\sum_t \langle \bar y_t, y(x_t) \rangle$ this lesson's
derivation is built from.
