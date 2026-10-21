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

In a normal transformer block, every token goes through the same feed-forward network. A common word and a rare term cost the same compute. Widen the network, and every token pays for the extra width.

A mixture of experts keeps many copies of that network, and a small router sends each token to a few of them. The model stores more parameters than one token uses.

You build the router, the loss that stops one expert from taking every token, and the rule that drops a token when an expert's buffer is full. You also write the router's gradient.

## What you already know

This lesson uses five ideas from earlier work:

- The feed-forward sublayer acts on each token on its own.
- A softmax turns a row of scores into positive numbers that sum to one.
- `np.argsort` returns the indices that would sort an array.
- A residual connection leaves a token unchanged when the sublayer outputs zero.
- A matrix product sends a gradient from outputs back to weights.

## Route one token to two of three experts

One token has three experts and scores `[2, 1, 0]`. Keep $k = 2$. A softmax exponentiates each score and divides by the row total. The exponentials are 7.389, 2.718, and 1, totalling 11.107. The probabilities are $7.389 / 11.107 = 0.6652$, $2.718 / 11.107 = 0.2447$, and $1 / 11.107 = 0.0900$.

Experts 0 and 1 are the top two. Their probabilities sum to 0.9099, so the gate weights are $0.6652 / 0.9099 = 0.7311$ and $0.2447 / 0.9099 = 0.2689$. Expert 2 gets weight 0 and does not run. If the chosen experts return 4 and 10, the mix is $0.7311 \times 4 + 0.2689 \times 10 = 5.613$.

That mix is a top-k router, and the rescaled probabilities are the gate weights. In general, `x` has shape `(n, d)` and $W_{\text{gate}}$ has shape `(d, E)`. Scores are $z = x W_{\text{gate}}$. The gate weight of a chosen expert is

$$
g_i = \frac{s_i}{\sum_{j \in \text{top-}k} s_j}
$$

Divide the expert's probability by the total probability of the chosen experts. The token output is

$$
\hat{y} = \sum_{i \in \text{top-}k} g_i \cdot \text{expert}_i(x)
$$

Here expert $i$ is a matrix of shape `(d, d_out)`, and its output is `x[t] @ experts[i]`. The three scores give gate weights 0.7311 and 0.2689 again, and the same mix is 5.613. Sketch the gates as follows:

```python
logits = x @ W_gate                        # (n, E)
top = np.argsort(-logits, axis=-1)[:, :k]  # (n, k)
```

> [!TIP]
> `np.take_along_axis` picks different columns per row. With `top` of shape `(n, k)`, it returns those probabilities.

The usual mistake sorts ascending and keeps the smallest scores.

> [!KEY]
> Each token runs only its top $k$ experts, mixed by gate weights that sum to one. Every other expert costs that token nothing.

## Count what you store and what you run

Take 8 experts of 100 million parameters, $k = 2$, and 100 million shared parameters on every token. Storage is $8 \times 100 + 100 = 900$ million. Compute per token is $2 \times 100 + 100 = 300$ million.

$$
P_{\text{total}} = E \cdot P_{\text{expert}} + P_{\text{shared}}, \qquad
P_{\text{active}} = k \cdot P_{\text{expert}} + P_{\text{shared}}
$$

Shared parameters sit in both counts. These figures give 900 million and 300 million again. When $k = E$, the layer is dense. Count them as follows:

```python
active = k * expert_params + shared_params
total = n_experts * expert_params + shared_params
```

The usual mistake leaves `shared_params` out of one count.

> [!KEY]
> Compute per token follows the active count. Memory follows the total count, all $E$ experts.

## Multiply by the gate

A gradient says how much the loss changes when you nudge one number. Top-$k$ is a yes-or-no choice. A tiny nudge usually keeps the same experts. If the output is a plain sum of those experts, that nudge leaves the output the same, and the router gradient is 0.

The gate weight moves smoothly with the logits, so the loss has a path back. Let $\bar{y}$ be the output gradient, length `d_out`. For a chosen expert, $c_i = \bar{y} \cdot \text{expert}_i(x)$. Then

$$
\frac{\partial L}{\partial z_j} = g_j \left(c_j - \bar{c}\right), \qquad \bar{c} = \sum_{i \in \text{top-}k} g_i c_i
$$

$\bar{c}$ is the gate-weighted average of the $c$ values. Each chosen logit gets its gate weight times the gap from that average. Other experts get 0 from this token. A downhill step lowers a positive gradient, so an expert that cuts the loss by more than average is pushed up.

With gates 0.7311 and 0.2689 and $c = [1, 3]$, the average is $0.7311 + 0.8067 = 1.5378$. Expert 0 gets $0.7311 \times (1 - 1.5378) = -0.3932$. Expert 1 gets $0.2689 \times (3 - 1.5378) = 0.3932$. The two values sum to 0, because a heavier chosen expert makes the other lighter. The formula returns those two numbers.

Stack the logit gradients as `dz` of shape `(n, E)`, then `x.T @ dz` is the gradient of $W_{\text{gate}}$. The usual mistake fills experts outside the top $k$.

> [!KEY]
> The top-$k$ choice has no gradient. The gate weight on each expert output is the path back to the router.

## Penalise a lopsided router

An expert chosen a bit more often gets more gradient, improves faster, and is chosen more. Routing then collapses, and most experts sit unused.

For $n$ tokens, $f_i$ is expert $i$'s share of the $n \cdot k$ routing slots, and $P_i$ is its average probability over tokens. The load-balancing loss is

$$
L_{\text{aux}} = E \sum_{i=1}^{E} f_i P_i
$$

Multiply traffic share by average probability, add, and scale by the number of experts. Two tokens, two experts, $k = 1$, both sent to expert 0, with probabilities `[0.9, 0.1]` and `[0.6, 0.4]`: $f = [1, 0]$ and $P = [0.75, 0.25]$. The loss is $2 \times 0.75 = 1.5$. The fractions have this shape:

```python
f = np.bincount(expert_indices.ravel(), minlength=E) / (n * k)
P = gate_probs.mean(axis=0)
```

> [!WARNING]
> Divide the counts by $n \cdot k$ slots. With $k = 2$, dividing by $n$ makes $f$ sum to 2.

Even routing sets $f_i = P_i = 1/E$, and the loss is 1. Total collapse makes the loss $E$. Every other routing lands between 1 and $E$. The factor $E$ holds that scale fixed. In practice a small weight, often about 0.01, multiplies the loss before it is added to the main one. Only $P$ has a gradient, because $f$ is a count. The usual mistake uses the wrong denominator.

> [!KEY]
> The balancing loss is 1 when routing is even and $E$ when it collapses. The gradient pushes probability off the busiest experts.

## Drop a token when the expert is full

Each expert accepts a fixed number of slots. Even routing would send $nk / E$ slots each. A capacity factor $\gamma$ (gamma, pronounced GAM-uh) adds headroom:

$$
\text{capacity} = \left\lceil \gamma \cdot \frac{n k}{E} \right\rceil
$$

Round the even share times $\gamma$ up. For $n = 6$, $k = 2$, $E = 3$, and $\gamma = 1$, capacity is $\lceil 12 / 3 \rceil = 4$. For $\gamma = 0.5$ it is 2. A larger factor drops fewer tokens and reserves more memory. Use at least 1. With `capacity_factor is None`, set capacity to $n \cdot k$ so nothing drops.

Walk tokens in row order. If the expert has room, add the weighted output. If it is full, drop the slot and count it. A token dropped from every choice gets a zero row, and the residual connection passes the original token through. The usual mistake lets later rows claim slots first.

## Send tokens to the device that holds the expert

Experts can sit on different accelerators. Tokens travel there and results travel back, an all-to-all that can outlast the expert math on a small batch. A router z-loss keeps logits from growing huge in 16-bit arithmetic. A shared expert runs on every token beside the routed ones. The usual mistake forgets that exchange.

> [!RECAP]
> Keep these five facts in front of you while you code:
>
> - Softmax `x @ W_gate` per row, keep the top $k$ logits, and renormalise those probabilities.
> - A token output is the gate-weighted sum of `x[t] @ experts[i]` on the chosen experts.
> - Active parameters are $k P_{\text{expert}} + P_{\text{shared}}$. Total parameters use $E$ in place of $k$.
> - The balancing loss is $E \sum_i f_i P_i$, with $f$ counted over $n k$ slots.
> - Capacity is $\lceil \gamma \, n k / E \rceil$, at least 1. Earlier rows claim slots, and a full expert drops the slot.

## Build it step by step

The checks call your functions in this order:

1. `router(x, W_gate, k)` takes `(n, d)` and returns indices, gate weights, and full probabilities, shapes `(n, k)`, `(n, k)`, and `(n, E)`. Subtract the row max, then softmax. The usual mistake sorts the wrong way:

```python
def router(x, W_gate, k):
    # top-k of x @ W_gate, then renormalise those probabilities
    return topk_indices, gate_weights, full_probs
```

2. `moe_forward(x, W_gate, experts, k, capacity_factor=None)` returns `(output, n_dropped)` with `output` shaped `(n, d_out)`. `None` means capacity `n * k`. Otherwise use the ceiling, at least 1. The usual mistake adds an expert output without its gate weight. With $k = E$ and no limit, the result is the dense weighted sum:

```python
def moe_forward(x, W_gate, experts, k, capacity_factor=None):
    # add g * (x[t] @ experts[i]) while the expert has room
    return output, n_dropped
```

3. `load_balancing_loss(expert_indices, gate_probs, n_experts)` returns $E \sum f_i P_i$. Use `np.bincount(..., minlength=n_experts)` and divide by `n * k`. The usual mistake divides by `n`:

```python
def load_balancing_loss(expert_indices, gate_probs, n_experts):
    # n_experts * sum(f * P)
    return loss
```

4. The capacity checks call `moe_forward` again. A larger `capacity_factor` never drops more. A token dropped from its only expert has an all-zero row. Earlier rows claim the slots.

5. `active_parameters(n_experts, k, expert_params, shared_params)` returns `(active, total)`. The usual mistake omits `shared_params` from one of them:

```python
def active_parameters(n_experts, k, expert_params, shared_params):
    # k * expert + shared, and n_experts * expert + shared
    return active, total
```

6. `router_backward(x, W_gate, k, expert_outputs, grad_output)` returns shape `(d, E)`. `expert_outputs` is `(n, E, d_out)` and `grad_output` is `(n, d_out)`. Write $g (c - \bar{c})$ on the chosen experts and finish with `x.T @ dz`. The usual mistake fills the other experts:

```python
def router_backward(x, W_gate, k, expert_outputs, grad_output):
    # dz only on the chosen experts, then x.T @ dz
    return grad_W
```

## Cost and scaling

Here $d_{\text{ff}}$ is each expert's hidden width. A dense layer and a mixture compare as follows:

| Quantity | Dense feed-forward | Mixture of experts |
|---|---|---|
| Parameters | about $d \cdot d_{\text{ff}}$ | about $E \cdot d \cdot d_{\text{ff}}$ |
| Compute per token | about $d \cdot d_{\text{ff}}$ | about $k \cdot d \cdot d_{\text{ff}}$ |
| Memory to serve | one network | all $E$ experts |
| Device traffic | none extra | one all-to-all each way, per layer |

Parameters grow with $E$ and compute grows with $k$. Memory still grows with $E$.

## Where this breaks

Every expert stays loaded, so memory follows the total count. Rare experts are offloaded or spread across devices. A small batch leaves each expert with too little work, and the layer runs as slow as its busiest expert.

A narrow fine-tuning set can collapse routing again. Top-$k$ can also switch a token's experts after a small update. People freeze the router, tune some experts, or keep the balancing loss on. A low capacity factor skips tokens quietly. Raising the factor, or balancing the load, leaves fewer overflows.

## The exercise

`router(x, W_gate, k)` returns top-$k$ indices, renormalised gate weights, and the full softmax. `moe_forward(x, W_gate, experts, k, capacity_factor)` returns the output and the dropped-slot count. `load_balancing_loss(expert_indices, gate_probs, n_experts)` computes $E \sum_i f_i P_i$. `active_parameters(n_experts, k, expert_params, shared_params)` returns active and total counts. `router_backward(x, W_gate, k, expert_outputs, grad_output)` returns the router gradient.

The checks confirm the true top-$k$ experts and gate weights that sum to one. With $k = E$ and no limit, `moe_forward` matches the dense weighted sum. The balancing loss matches a hand computation, equals 1 for uniform routing, equals $E$ for collapse, and lands between them otherwise. A larger capacity factor never drops more, a fully dropped token is zeros, and `router_backward` matches finite differences.
