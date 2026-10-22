---
title: "When an expert is full: capacity and dropping"
slug: 132-capacity-and-dropping
part: "Part 7 — Large language models"
summary: Each expert accepts a fixed number of tokens per batch, and the overflow is dropped while the residual connection carries the token through.
minutes: 40
gpu: false
objectives:
  - Explain why an expert has a fixed number of seats per batch, and compute the capacity from the capacity factor.
  - Run a capacity-limited forward pass that drops overflow assignments in row order, and count them.
  - Explain why a dropped assignment is survivable, using the residual connection, and what a token with two choices loses when one is dropped.
  - Read a table of capacity factor against dropped share, and say what the extra memory buys.
prereqs:
  - 131-load-balancing
  - 129-top-k-routing
  - 104-the-residual-block
exercise: 132-capacity-and-dropping
---

# When an expert is full: capacity and dropping

A batch of eight tokens arrives at a layer with four experts, and each expert has room for four tokens. The router sends six tokens to expert 0 and two to expert 1. Expert 0 has four seats for six tokens, and experts 2 and 3 have eight seats between them and nothing to do. What happens to the two extra tokens, and how much does losing them hurt?

The room limit exists for a practical reason. Accelerators run fastest on tensors of a fixed shape, so each expert is given a buffer of fixed size, decided before the batch is routed. A router cannot promise to fill every buffer exactly, and the balancing loss from the last chapter only nudges the average, not every batch. So some experts overflow while others sit half empty.

This chapter shows what the layer does with the overflow, why the answer is less harmful than it sounds, and what it costs to avoid it.

## What you already know

- The router gives each token $k$ experts and gate weights that sum to one, from "Send each token to a few experts".
- A routing slot is one (token, expert) choice, so a batch has $n k$ slots, as in "Keep the load balanced".
- Balanced routing is the aim, but any one batch is lumpy.
- A block adds a sublayer's output to the token, as in "Stacking it up: residuals, layer norm, and the pre-norm block".
- A matrix product `x[t] @ expert` is one expert's output for one token.

## The picture: seats on a bus

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Four experts with four seats each. Expert 0 receives six tokens, so two are dropped; expert 1 receives two; experts 2 and 3 are empty">
<defs><marker id="c132-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="300.0" y="20.0" text-anchor="middle" class="fig-bold">capacity: 4 seats per expert per batch</text>
<text x="140.0" y="46.0" text-anchor="middle" class="fig-bold">expert 0</text>
<rect x="110.0" y="60.0" width="60.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="140.0" y="82.0" text-anchor="middle" class="fig-bold">t1</text>
<rect x="110.0" y="104.0" width="60.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="140.0" y="126.0" text-anchor="middle" class="fig-bold">t2</text>
<rect x="110.0" y="148.0" width="60.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="140.0" y="170.0" text-anchor="middle" class="fig-bold">t3</text>
<rect x="110.0" y="192.0" width="60.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="140.0" y="214.0" text-anchor="middle" class="fig-bold">t4</text>
<text x="260.0" y="46.0" text-anchor="middle" class="fig-bold">expert 1</text>
<rect x="230.0" y="60.0" width="60.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="260.0" y="82.0" text-anchor="middle" class="fig-bold">t7</text>
<rect x="230.0" y="104.0" width="60.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="260.0" y="126.0" text-anchor="middle" class="fig-bold">t8</text>
<rect x="230.0" y="148.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="260.0" y="170.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="230.0" y="192.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="260.0" y="214.0" text-anchor="middle" class="fig-small">empty</text>
<text x="380.0" y="46.0" text-anchor="middle" class="fig-bold">expert 2</text>
<rect x="350.0" y="60.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="380.0" y="82.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="350.0" y="104.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="380.0" y="126.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="350.0" y="148.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="380.0" y="170.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="350.0" y="192.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="380.0" y="214.0" text-anchor="middle" class="fig-small">empty</text>
<text x="500.0" y="46.0" text-anchor="middle" class="fig-bold">expert 3</text>
<rect x="470.0" y="60.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="500.0" y="82.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="470.0" y="104.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="500.0" y="126.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="470.0" y="148.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="500.0" y="170.0" text-anchor="middle" class="fig-small">empty</text>
<rect x="470.0" y="192.0" width="60.0" height="34.0" rx="4" class="fill-sunk stroke-grid dashed"/>
<text x="500.0" y="214.0" text-anchor="middle" class="fig-small">empty</text>
<circle cx="40.0" cy="100.0" r="20" class="fill-paper stroke-bad"/>
<text x="40.0" y="105.0" text-anchor="middle" class="fig-bold">t5</text>
<line x1="62.0" y1="100.0" x2="104.0" y2="100.0" class="stroke-bad dashed" marker-end="url(#c132-arrow)"/>
<line x1="76.0" y1="93.0" x2="90.0" y2="107.0" class="stroke-bad"/>
<line x1="76.0" y1="107.0" x2="90.0" y2="93.0" class="stroke-bad"/>
<circle cx="40.0" cy="160.0" r="20" class="fill-paper stroke-bad"/>
<text x="40.0" y="165.0" text-anchor="middle" class="fig-bold">t6</text>
<line x1="62.0" y1="160.0" x2="104.0" y2="160.0" class="stroke-bad dashed" marker-end="url(#c132-arrow)"/>
<line x1="76.0" y1="153.0" x2="90.0" y2="167.0" class="stroke-bad"/>
<line x1="76.0" y1="167.0" x2="90.0" y2="153.0" class="stroke-bad"/>
<text x="40.0" y="60.0" text-anchor="middle" class="fig-small">dropped</text>
<text x="300.0" y="276.0" text-anchor="middle" class="fig-small">Tokens t1 to t8 arrive in order. Six want expert 0, two want expert 1.</text>
<text x="300.0" y="296.0" text-anchor="middle" class="fig-small">Four seats fill, two tokens turn away, and ten seats elsewhere stay empty.</text>
</svg>
Figure 1. Overflow and waste at the same time. Expert 0 has more tokens than seats, so t5 and t6 are dropped, while experts 2 and 3 have no tokens at all. Tokens arrive in row order, so the early ones get the seats.
```

Think of each expert as a bus with a fixed number of seats, here four. Tokens board in the order they arrive. In Figure 1, tokens t1 to t4 want expert 0 and fill its four seats. Tokens t5 and t6 also want expert 0, find no seat, and are turned away, which is what dropping means. Tokens t7 and t8 take two seats on expert 1. Experts 2 and 3 carry nobody. The layer shows both problems at once, overflow where traffic is heavy and waste where it is light.

## How many seats

If routing were perfectly even, each expert would receive $n k / E$ slots, so even share is the natural number of seats. You then multiply by a capacity factor $\gamma$ (the Greek letter gamma), which buys extra seats to absorb lumpiness:

$$
\text{capacity} = \left\lceil \gamma \, \frac{n k}{E} \right\rceil
$$

Read aloud: the even share of slots per expert, times $\gamma$, rounded up to a whole number. For the bus in Figure 1, $n = 8$, $k = 1$, $E = 4$, and the even share is 2. With $\gamma = 2$ the capacity is $\lceil 2 \times 2 \rceil = 4$, the four seats in the picture. Take another example with $n = 6$, $k = 2$, $E = 3$: the even share is $12 / 3 = 4$, so $\gamma = 1$ gives 4 seats and $\gamma = 0.5$ gives 2.

The ceiling guarantees a whole number, and the capacity is never allowed to fall below 1, so every expert can take at least one slot. That "at least 1" applies to the capacity, not to $\gamma$. A factor below 1 is allowed, as the $\gamma = 0.5$ example shows. It buys fewer seats than an even share, so with a factor of 0.5 slots are dropped even when the routing is perfectly even. When there is no limit, set the capacity to $n k$, a number no expert can exceed.

```python
capacity = max(1, math.ceil(gamma * n * k / E))
```

## Who gets a seat

The simplest rule is first come, first served. Walk the tokens in row order, and within a token walk its $k$ choices in order. Keep a count of seats used per expert. If the expert's count is below its capacity, the token takes a seat and the expert's weighted output is added to the token. If not, the slot is dropped and a counter goes up.

```python
if seats_used[e] < capacity:
    seats_used[e] += 1
    out[t] += g[t, slot] * (x[t] @ experts[e])
else:
    n_dropped += 1
```

The rule is cheap but not fair. It is arbitrary, because row order has nothing to do with which token needs the expert most, and it is biased, because tokens late in the batch lose every time. Some systems rank the claims by gate weight, so the token that trusts the expert most boards first, and others shuffle the order. The exercise uses row order so that the result is checkable.

## What dropping does to a token

Dropping removes a slot, not a token. With $k = 2$, a token dropped by one expert still gets the other. The other's gate weight is not rescaled, so the token's output shrinks: if the gates were 0.73 and 0.27 and the second expert is full, the token receives only $0.73 \times \text{expert}_0(x)$. A token dropped from both of its experts gets a row of zeros from the layer.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Top row: a kept token gives x plus the expert output. Bottom row: a dropped token gives x plus zero, which is x">
<defs><marker id="c132c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="300.0" y="22.0" text-anchor="middle" class="fig-bold">kept token</text>
<rect x="20.0" y="50.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="75.0" y="75.0" text-anchor="middle" class="fig-small">x = [0.5, −1.0]</text>
<line x1="130.0" y1="70.0" x2="200.0" y2="70.0" class="stroke-muted" marker-end="url(#c132c-arrow)"/>
<rect x="205.0" y="50.0" width="150.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="280.0" y="75.0" text-anchor="middle" class="fig-small">MoE(x) = [0.3, 0.2]</text>
<line x1="355.0" y1="70.0" x2="402.0" y2="70.0" class="stroke-muted" marker-end="url(#c132c-arrow)"/>
<circle cx="420.0" cy="70.0" r="18" class="fill-paper stroke-ink"/>
<text x="420.0" y="75.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="438.0" y1="70.0" x2="468.0" y2="70.0" class="stroke-muted" marker-end="url(#c132c-arrow)"/>
<rect x="472.0" y="50.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="527.0" y="75.0" text-anchor="middle" class="fig-small">[0.8, −0.8]</text>
<line x1="75.0" y1="50.0" x2="75.0" y2="36.0" class="stroke-a"/>
<line x1="75.0" y1="36.0" x2="420.0" y2="36.0" class="stroke-a"/>
<line x1="420.0" y1="36.0" x2="420.0" y2="52.0" class="stroke-a" marker-end="url(#c132c-arrow)"/>
<text x="300.0" y="152.0" text-anchor="middle" class="fig-bold">dropped token</text>
<rect x="20.0" y="180.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="75.0" y="205.0" text-anchor="middle" class="fig-small">x = [0.5, −1.0]</text>
<line x1="130.0" y1="200.0" x2="200.0" y2="200.0" class="stroke-muted" marker-end="url(#c132c-arrow)"/>
<rect x="205.0" y="180.0" width="150.0" height="40.0" rx="6" class="fill-sunk stroke-bad dashed"/>
<text x="280.0" y="205.0" text-anchor="middle" class="fig-small">dropped: MoE(x) = 0</text>
<line x1="355.0" y1="200.0" x2="402.0" y2="200.0" class="stroke-muted" marker-end="url(#c132c-arrow)"/>
<circle cx="420.0" cy="200.0" r="18" class="fill-paper stroke-ink"/>
<text x="420.0" y="205.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="438.0" y1="200.0" x2="468.0" y2="200.0" class="stroke-muted" marker-end="url(#c132c-arrow)"/>
<rect x="472.0" y="180.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="527.0" y="205.0" text-anchor="middle" class="fig-small">[0.5, −1.0]</text>
<line x1="75.0" y1="180.0" x2="75.0" y2="166.0" class="stroke-a"/>
<line x1="75.0" y1="166.0" x2="420.0" y2="166.0" class="stroke-a"/>
<line x1="420.0" y1="166.0" x2="420.0" y2="182.0" class="stroke-a" marker-end="url(#c132c-arrow)"/>
<text x="300.0" y="255.0" text-anchor="middle" class="fig-small">The skip line carries x around the expert, so the sum still contains x.</text>
</svg>
Figure 2. The skip connection (the line over the top) adds x back after the expert. When the expert is skipped its output is zero, so the sum is x itself. The token loses the expert's update, not its identity.
```

That sounds like losing the token, but it is not, because of the sum that wraps every sublayer. A block does not replace the token with the layer's output. It adds the output to the token, $x + \text{MoE}(x)$. Figure 2 shows a kept token with $x = [0.5, -1.0]$ and an expert output of $[0.3, 0.2]$, which sum to $[0.8, -0.8]$. For a dropped token, the expert output is zero, and the sum is $x + 0 = [0.5, -1.0]$. The skip line carries the token around the layer untouched. A dropped token behaves as if the layer were absent for it, so it loses an update, not its identity. A few drops are a small loss in quality. Many drops mean many tokens skip the layer, which is real damage.

## What the extra seats cost

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Dropped assignments out of 120 fall from 88 at capacity factor 0.25 to zero near 1.75">
<defs><marker id="c132b-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="50.0" class="stroke-muted"/>
<line x1="114.5" y1="230.0" x2="114.5" y2="234.0" class="stroke-muted"/>
<text x="114.5" y="248.0" text-anchor="middle" class="fig-small">0.5</text>
<line x1="203.6" y1="230.0" x2="203.6" y2="234.0" class="stroke-muted"/>
<text x="203.6" y="248.0" text-anchor="middle" class="fig-small">1</text>
<line x1="292.7" y1="230.0" x2="292.7" y2="234.0" class="stroke-muted"/>
<text x="292.7" y="248.0" text-anchor="middle" class="fig-small">1.5</text>
<line x1="381.8" y1="230.0" x2="381.8" y2="234.0" class="stroke-muted"/>
<text x="381.8" y="248.0" text-anchor="middle" class="fig-small">2</text>
<line x1="470.9" y1="230.0" x2="470.9" y2="234.0" class="stroke-muted"/>
<text x="470.9" y="248.0" text-anchor="middle" class="fig-small">2.5</text>
<line x1="560.0" y1="230.0" x2="560.0" y2="234.0" class="stroke-muted"/>
<text x="560.0" y="248.0" text-anchor="middle" class="fig-small">3</text>
<line x1="66.0" y1="230.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<text x="62.0" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="66.0" y1="170.0" x2="70.0" y2="170.0" class="stroke-muted"/>
<text x="62.0" y="174.0" text-anchor="end" class="fig-small">30</text>
<line x1="66.0" y1="110.0" x2="70.0" y2="110.0" class="stroke-muted"/>
<text x="62.0" y="114.0" text-anchor="end" class="fig-small">60</text>
<line x1="66.0" y1="50.0" x2="70.0" y2="50.0" class="stroke-muted"/>
<text x="62.0" y="54.0" text-anchor="end" class="fig-small">90</text>
<polyline points="70.0,54.0 78.9,70.0 87.8,84.0 96.7,84.0 105.6,98.0 114.5,112.0 123.5,124.0 132.4,124.0 141.3,136.0 150.2,146.0 159.1,154.0 168.0,154.0 176.9,162.0 185.8,170.0 194.7,176.0 203.6,176.0 212.5,182.0 221.5,188.0 230.4,194.0 239.3,194.0 248.2,200.0 257.1,206.0 266.0,212.0 274.9,212.0 283.8,218.0 292.7,224.0 301.6,228.0 310.5,228.0 319.5,230.0 328.4,230.0 337.3,230.0 346.2,230.0 355.1,230.0 364.0,230.0 372.9,230.0 381.8,230.0 390.7,230.0 399.6,230.0 408.5,230.0 417.5,230.0 426.4,230.0 435.3,230.0 444.2,230.0 453.1,230.0 462.0,230.0 470.9,230.0 479.8,230.0 488.7,230.0 497.6,230.0 506.5,230.0 515.5,230.0 524.4,230.0 533.3,230.0 542.2,230.0 551.1,230.0 560.0,230.0" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="114.5" cy="112.0" r="4" class="fill-a"/>
<text x="122.5" y="104.0" text-anchor="start" class="fig-small">59</text>
<circle cx="203.6" cy="176.0" r="4" class="fill-a"/>
<text x="211.6" y="168.0" text-anchor="start" class="fig-small">27</text>
<circle cx="292.7" cy="224.0" r="4" class="fill-a"/>
<text x="300.7" y="216.0" text-anchor="start" class="fig-small">3</text>
<circle cx="381.8" cy="230.0" r="4" class="fill-a"/>
<text x="389.8" y="222.0" text-anchor="start" class="fig-small">0</text>
<text x="315.0" y="262.0" text-anchor="middle" class="fig-small">capacity factor γ</text>
<text x="70.0" y="36.0" text-anchor="start" class="fig-small">dropped assignments (of 120)</text>
</svg>
Figure 3. One fixed routing of 60 tokens, k = 2, eight experts. Drops fall steadily as the buffer grows, and reach zero once every expert can hold its busiest share. Each step right costs more memory in every expert, busy or not.
```

Figure 3 fixes one lumpy routing of 60 tokens with $k = 2$ over eight experts, 120 slots in all, and varies only $\gamma$. The next table reads some points off the curve:

| $\gamma$ | Seats per expert | Dropped slots | Share dropped |
|---|---|---|---|
| 0.5 | 8 | 59 | 49% |
| 1.0 | 15 | 27 | 22.5% |
| 1.5 | 23 | 3 | 2.5% |
| 2.0 | 30 | 0 | 0% |

Even at $\gamma = 1$, the even share, more than a fifth of the slots are dropped, because the busiest expert received 25 slots against 15 seats. Drops reach zero only when every expert's buffer holds its busiest share, here 25 seats, which is a $\gamma$ of about 1.65. (For the curious, the eight experts received 25, 24, 23, 14, 11, 10, 8, and 5 slots, which add to 120. At $\gamma = 1$ the three busiest overflow their 15 seats by 10, 9, and 8, which is the 27 in the table.) The cost of the buffer is paid in every expert, busy or not. Doubling $\gamma$ doubles the memory reserved for the buffers and the compute spent on padding the empty seats. So the choice is a trade: a smaller factor saves memory and time but silently skips work for some tokens.

> [!KEY]
> Each expert has a fixed number of seats per batch, $\lceil \gamma n k / E \rceil$. When routing is lumpy, the overflow slots are dropped, first come first served. A dropped slot contributes zero, and because the block computes $x + \text{MoE}(x)$, a token that loses all its slots passes through unchanged. So eight tokens with six wanting one expert lose two assignments, and the two tokens skip that layer rather than disappear.

> [!RECAP]
> Keep these points in front of you while you code:
>
> - Capacity is `max(1, ceil(gamma * n * k / E))`, and `n * k` when there is no limit.
> - Walk tokens in row order and each token's slots in order, and count seats used per expert.
> - A full expert adds nothing and adds one to the dropped count.
> - A dropped slot does not rescale the surviving gate weight.
> - A token dropped from every choice has an all-zero row.

## Build it step by step

The router from "Send each token to a few experts" is already written for you. The checks call your functions in this order:

1. `expert_capacity(n_tokens, k, n_experts, capacity_factor)` returns an integer. `None` means `n_tokens * k`. Otherwise round up and keep at least 1. The usual mistake is reading "at least 1" as a rule about the factor.

```python
def expert_capacity(n_tokens, k, n_experts, capacity_factor):
    # None -> n_tokens * k, else max(1, ceil(...))
    return capacity
```

2. `moe_forward(x, W_gate, experts, k, capacity_factor=None)` returns `(output, n_dropped)`. It extends the version from "Send each token to a few experts" with seats. The usual mistake is letting later rows claim seats first.

```python
def moe_forward(x, W_gate, experts, k, capacity_factor=None):
    # route, then walk tokens and slots in order, counting seats
    return output, n_dropped
```

3. `dropped_fraction(n_dropped, n_tokens, k)` returns the share of all `n_tokens * k` slots that were dropped. The usual mistake is dividing by the number of tokens.

```python
def dropped_fraction(n_dropped, n_tokens, k):
    # dropped slots over all slots
    return fraction
```

## Where this breaks

A small batch gives each expert too few tokens to fill its buffer, so most seats go unused and the layer wastes compute. Experts that sit on different accelerators need an all-to-all exchange, in which every device sends each token to the device that holds its expert and receives the results back, and the whole layer waits for the busiest device. Narrow fine-tuning data can bring collapse back, which raises the dropped share. Raising $\gamma$ or balancing the load reduces drops. Some designs add a shared expert that every token runs, so that a dropped token still gets some computation from it.

## The exercise

You write `expert_capacity(n_tokens, k, n_experts, capacity_factor)`, `moe_forward(x, W_gate, experts, k, capacity_factor=None)`, and `dropped_fraction(n_dropped, n_tokens, k)`. The checks confirm the capacities 4 and 2 on the worked example, that the capacity rounds up and never falls below 1, and that `dropped_fraction` divides by $n k$. They also confirm that `moe_forward` without a limit matches the plain mix, drops exactly the overflow when an expert is oversubscribed, gives zero rows to tokens that lose their only choice, keeps the surviving gate weight when one of two slots is dropped, and never drops more as $\gamma$ grows.
