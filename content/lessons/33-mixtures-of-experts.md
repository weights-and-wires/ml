---
title: Mixtures of experts
slug: 33-mixtures-of-experts
part: "Part 6 — Large language models"
summary: Route each token to a few experts, keep the load balanced, and handle full experts by dropping tokens.
minutes: 55
gpu: false
objectives:
  - Build a top-k router and compute each token's gate weights.
  - Compute active and total parameter counts, and explain why they differ.
  - Compute the load-balancing loss, and explain why uniform routing minimises it.
  - Build a capacity-limited forward pass that drops tokens when an expert is full.
  - Compute the router's gradient, and explain why the gate weight must multiply the expert output.
prereqs:
  - 26-the-transformer-block
exercise: 33-mixtures-of-experts
---

# Mixtures of experts

In a normal transformer block, every token goes through the same
feed-forward network. A common word like "the" gets exactly as much compute
as a rare technical term. To make the network smarter you widen it, and then
every token pays for the extra width.

A mixture of experts, or MoE, breaks that link. Think of a hospital. Instead
of one doctor who sees every patient, you have many specialists and a
receptionist who sends each patient to the two most suitable ones. The
hospital employs many doctors, but each patient sees only a couple. In the
model, the specialists are copies of the feed-forward network called
*experts*, and the receptionist is a small *router*.

The model can then hold far more parameters than it spends compute on. The
price is a routing decision that has to be learned, kept balanced, and
squeezed into fixed-size hardware buffers. You build each of those pieces in
this lesson.

## What you already know

- The feed-forward sublayer acts on each token separately.
- A softmax turns a vector of scores into probabilities that sum to one.
- `np.argsort` returns the indices that would sort an array.
- The sublayer sits inside a residual connection, so a zero output leaves the
  token unchanged.
- A gradient says how much the loss changes when you nudge a number.

## Route each token

You have $n$ tokens, each a vector of $d$ numbers, stacked in `x` of shape
`(n, d)`. You have $E$ experts. The router is a weight matrix $W_{\text{gate}}$
of shape `(d, E)`. For each token it works in three steps:

1. **Score every expert.** The logits are $z = x W_{\text{gate}}$, shape
   `(n, E)`. One score per token per expert.
2. **Turn scores into probabilities.** Apply a softmax to each row. Call the
   result $s$, the full probabilities.
3. **Keep the top $k$ and renormalise.** Pick the $k$ experts with the
   highest scores, then rescale their probabilities to sum to one. These are
   the *gate weights* $g$.

The gate weight for a chosen expert $i$ is

$$
g_i = \frac{s_i}{\sum_{j \in \text{top-}k} s_j}
$$

Read it as: the expert's probability divided by the total probability of the
chosen experts. Experts outside the top $k$ get weight 0 and never run.

The layer's output for a token is the gate-weighted sum of the chosen
experts' outputs:

$$
\hat{y} = \sum_{i \in \text{top-}k} g_i \cdot \text{expert}_i(x)
$$

In the exercise each expert is one matrix of shape `(d, d_out)`, so
$\text{expert}_i(x)$ is `x[t] @ experts[i]`. In numpy, the router is:

```python
logits = x @ W_gate                              # (n, E)
top = np.argsort(-logits, axis=-1)[:, :k]        # (n, k)
chosen = np.take_along_axis(full_probs, top, axis=-1)
```

### Worked example

One token with logits $[2, 1, 0, -1]$ over 4 experts, and $k = 2$:

- Exponentials: $[7.389, 2.718, 1.000, 0.368]$, total $11.475$
- Full probabilities: $s = [0.6439, 0.2369, 0.0871, 0.0321]$
- Top 2 are experts 0 and 1, with total $0.6439 + 0.2369 = 0.8808$
- Gate weights: $0.6439 / 0.8808 = 0.7311$ and $0.2369 / 0.8808 = 0.2689$

The output is $0.7311 \cdot \text{expert}_0(x) + 0.2689 \cdot
\text{expert}_1(x)$. Experts 2 and 3 cost nothing for this token.

## Count active and total parameters

With $E$ experts of $P_{\text{expert}}$ parameters each, $k$ chosen per token,
and $P_{\text{shared}}$ parameters every token always uses:

$$
P_{\text{total}} = E \cdot P_{\text{expert}} + P_{\text{shared}}, \qquad
P_{\text{active}} = k \cdot P_{\text{expert}} + P_{\text{shared}}
$$

Read it as: the model stores all $E$ experts but runs only $k$ of them per
token. The shared part counts in both. Compute per token follows the active
count. Memory follows the total count.

For $E = 8$ experts of 100 million parameters, $k = 2$, and 100 million shared
parameters: total is $8 \times 100 + 100 = 900$ million, active is $2 \times
100 + 100 = 300$ million. Each token costs as much as a 300-million-parameter
model, while the model holds 900 million. When $k = E$, active equals total
and the layer is dense again.

## Why the gate must multiply the output

Picking the top $k$ is a yes-or-no decision. Nudge the router weights a tiny
bit, and almost always the same experts win. If the output were a plain sum
of the chosen experts, it wouldn't change at all, and the router would get a
gradient of exactly zero. It could never learn.

Multiplying by $g_i$ fixes that. The gate weights change smoothly as the
logits change, so the loss has something to pass back to the router.

Here's the gradient. Let $\bar{y}$ be the gradient of the loss with respect
to the token's output, a vector of length `d_out`. For each chosen expert,
define $c_i = \bar{y} \cdot \text{expert}_i(x)$, a single number that says how
much the loss would change if expert $i$ got more weight. Then for each
chosen expert $j$:

$$
\frac{\partial L}{\partial z_j} = g_j \left(c_j - \bar{c}\right), \qquad \bar{c} = \sum_{i \in \text{top-}k} g_i c_i
$$

Read it as: $\bar{c}$ is the gate-weighted average of the $c$ values. Each
expert's logit gradient is its gate weight times how far its $c$ sits from
that average. Experts not in the top $k$ get a gradient of 0 from this token.
Gradient descent lowers logits with positive gradient, so experts that reduce
the loss more than average get pushed up.

The gradient for $W_{\text{gate}}$ follows from $z = x W_{\text{gate}}$: stack
every token's logit gradients into `dz` of shape `(n, E)`, then compute
`x.T @ dz`, shape `(d, E)`.

### Worked example

Gate weights $g = [0.7311, 0.2689]$ for the two chosen experts, and $c = [1,
3]$:

- $\bar{c} = 0.7311 \times 1 + 0.2689 \times 3 = 1.5379$
- Expert 0: $0.7311 \times (1 - 1.5379) = -0.3932$
- Expert 1: $0.2689 \times (3 - 1.5379) = 0.3932$

The two values add to zero. That always happens, because making one chosen
expert heavier makes the others lighter.

## Keep the load balanced

Routing has a rich-get-richer problem. If one expert is chosen slightly more
often early on, it gets more gradient, improves faster, and gets chosen even
more. The others starve. Left alone, routing *collapses* onto a few experts
and most of the model sits unused.

The fix is an extra loss, added to the main one, that penalises lopsided
routing. For a batch of $n$ tokens, measure two things per expert $i$:

- $f_i$: the fraction of all routing slots that went to expert $i$. There are
  $n \cdot k$ slots in total.
- $P_i$: the average full probability the router gave expert $i$, over all
  tokens.

The load-balancing loss is

$$
L_{\text{aux}} = E \sum_{i=1}^{E} f_i P_i
$$

Read it as: multiply each expert's traffic share by its average probability,
add them up, and scale by the number of experts. In numpy:

```python
f = np.bincount(expert_indices.ravel(), minlength=E) / (n * k)
P = gate_probs.mean(axis=0)
```

Both $f$ and $P$ sum to one. If routing is perfectly even, each is $1/E$, so
the sum is $E \times (1/E)^2 = 1/E$ and the loss is exactly 1. If everything
goes to one expert, both are 1 there and 0 elsewhere, and the loss is $E$.
Every other routing lands between 1 and $E$. The factor $E$ keeps the loss at
the same strength however many experts you use. In practice it's multiplied
by a small weight, often around 0.01, before being added to the main loss.

Only $P$ carries gradient, because $f$ comes from counting. The gradient
lowers the probability of each expert in proportion to how much traffic it
already gets.

### Worked example

Two tokens, two experts, $k = 1$. Both tokens go to expert 0. Full
probabilities are $[0.9, 0.1]$ and $[0.6, 0.4]$.

- $f = [2/2, 0/2] = [1, 0]$
- $P = [(0.9 + 0.6)/2, (0.1 + 0.4)/2] = [0.75, 0.25]$
- $L_{\text{aux}} = 2 \times (1 \times 0.75 + 0 \times 0.25) = 1.5$

## Drop tokens at capacity

Hardware wants fixed-size arrays. Routing gives each expert a different,
unpredictable number of tokens. So each expert gets a fixed *capacity*: the
most routing slots it accepts per batch. If routing were perfectly even, each
expert would get $nk/E$ slots. A *capacity factor* $\gamma$ adds headroom:

$$
\text{capacity} = \left\lceil \gamma \cdot \frac{n k}{E} \right\rceil
$$

Read it as: the even share, times the headroom factor, rounded up. In code:
`int(np.ceil(capacity_factor * n * k / E))`, and at least 1.

Walk through the tokens in order. For each of a token's $k$ choices, if that
expert still has room, count the slot and add its weighted output. If the
expert is full, the slot is *dropped*: it adds nothing, and you count it. A
token dropped from all its experts gets a zero output, and the residual
connection passes it through unchanged. That's a real loss of computation for
that token.

For example, with $n = 6$ tokens, $k = 2$, $E = 3$, and $\gamma = 1$, the
capacity is $\lceil 1 \times 12 / 3 \rceil = 4$ slots per expert. With
$\gamma = 0.5$ it's 2. A bigger factor drops fewer tokens but reserves more
memory, whether it's used or not.

## Spread experts across devices

Big MoE models put different experts on different accelerators. Each token
must then be sent to the device holding its experts, and the results sent
back. This exchange is called an *all-to-all*. It's pure communication, and
with small batches or slow links it can take longer than the expert math.

Two more tricks are common. A *router z-loss* keeps the router's logits from
growing huge, which matters in 16-bit arithmetic. A *shared expert* runs on
every token alongside the routed ones, so the routed experts can specialise.

## Build it step by step

The checks test the functions in this order.

1. **`router(x, W_gate, k)`.** Takes `x` of shape `(n, d)`. Returns
   `(topk_indices, gate_weights, full_probs)` with shapes `(n, k)`, `(n, k)`,
   and `(n, E)`. Compute logits, a row-wise stable softmax (subtract each
   row's max, `axis=-1, keepdims=True`), the top-`k` indices with
   `np.argsort(-logits, axis=-1)[:, :k]`, then gather and renormalise with
   `np.take_along_axis`. Common mistake: sorting ascending and keeping the
   lowest scores.
2. **`moe_forward(x, W_gate, experts, k, capacity_factor=None)`.** Returns
   `(output, n_dropped)`, where `output` is `(n, d_out)`. Call your router,
   set the capacity (with `None`, use `n * k`, so nothing drops), and loop
   over tokens and slots with a per-expert counter. Common mistake: adding
   the expert output without the gate weight. With `k = E` and no limit, the
   checks expect the dense softmax-weighted sum.
3. **`load_balancing_loss(expert_indices, gate_probs, n_experts)`.** Returns
   a float. Use `np.bincount(..., minlength=n_experts)` and `.mean(axis=0)`.
   Common mistake: dividing the counts by `n` instead of `n * k`.
4. **Capacity behaviour.** No new function, but the checks confirm that a
   larger `capacity_factor` never drops more, and that a token dropped from
   its only expert gets an all-zero output row. Process tokens in row order,
   so the first tokens claim the slots.
5. **`active_parameters(n_experts, k, expert_params, shared_params)`.**
   Returns `(active, total)` as integers. Common mistake: leaving
   `shared_params` out of one of them.
6. **`router_backward(x, W_gate, k, expert_outputs, grad_output)`.**
   `expert_outputs` has shape `(n, E, d_out)` and `grad_output` has shape
   `(n, d_out)`. Returns the gradient for `W_gate`, shape `(d, E)`. For each
   token, compute `c` for its chosen experts, then $\bar{c}$, then fill those
   entries of `dz`. Finish with `x.T @ dz`. Common mistake: filling gradients
   for experts outside the top `k`. They must stay zero.

## Cost and scaling

The whole trade fits in one table. Here $d_{\text{ff}}$ is each expert's
hidden width.

| Quantity | Dense feed-forward | Mixture of experts |
|---|---|---|
| Parameters | about $d \cdot d_{\text{ff}}$ | about $E \cdot d \cdot d_{\text{ff}}$ |
| Compute per token | about $d \cdot d_{\text{ff}}$ | about $k \cdot d \cdot d_{\text{ff}}$ |
| Memory to serve | one network | all $E$ experts |
| Device traffic | none extra | one all-to-all each way, per MoE layer |

Parameters grow with $E$ and compute grows with $k$. That gap is the reason
to use MoE. Memory still grows with $E$. The router itself is tiny: one
`(d, E)` matrix and a sort of $E$ numbers per token.

## Where this breaks

**Memory.** Every expert must stay loaded, even though each token uses only
$k$ of them. A sparse model needs memory for its total parameters, not its
active ones. People offload rarely used experts or spread them across many
devices.

**Small batches.** Each expert sees only its share of the tokens. With a
small batch, that share is too small to keep the hardware busy, and the
layer runs as slow as its busiest expert.

**Fine-tuning.** A narrow fine-tuning dataset can pull traffic onto a few
experts and bring collapse back. Because top-$k$ is a yes-or-no choice, a
small update can suddenly switch a token's experts. People freeze the router,
tune only some experts, or keep the balancing loss on.

**Dropped tokens.** A low capacity factor silently skips computation for
some tokens. People raise the factor, or use the balancing loss to keep the
load even enough that few tokens overflow.

## The exercise

You build the router, the balancing loss, and the capacity rules. Write
`router(x, W_gate, k)`, which returns the top-$k$ expert indices, their
renormalised gate weights, and the full softmax probabilities. Write
`moe_forward(x, W_gate, experts, k, capacity_factor)`, which returns the
layer's output and the number of dropped slots, and
`load_balancing_loss(expert_indices, gate_probs, n_experts)`, which computes
$E \sum_i f_i P_i$. Finish with `active_parameters(n_experts, k,
expert_params, shared_params)` and `router_backward(x, W_gate, k,
expert_outputs, grad_output)`.

The checks verify that the router picks the true top-$k$ experts and that
gate weights sum to one, that `moe_forward` with $k = E$ and no capacity
limit equals the dense weighted sum, and that the balancing loss matches a
hand computation, equals 1 for uniform routing and $E$ for collapse, and
lands between them otherwise. They check that a larger capacity factor never
drops more, that a fully dropped token outputs zeros, the active and total
counts, and `router_backward` against finite differences.
