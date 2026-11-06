---
title: "Keeping the experts balanced"
slug: 117-balancing-the-experts
part: "Part 7 — Large language models"
summary: An auxiliary loss on traffic share times average probability keeps a few experts from taking every token, and a capacity limit drops what still overflows.
minutes: 55
gpu: false
objectives:
  - Explain how a small head start lets one expert take most tokens, and why the others then never learn.
  - Define f and P, compute the load-balancing loss for even, skewed, and collapsed routing, and read the gradient it sends to the router.
  - Compute an expert's capacity from the capacity factor, and run a forward pass that drops overflow assignments in row order.
  - Explain why a dropped assignment is survivable, using the residual connection, and read the cost of extra seats in memory.
prereqs:
  - 115-top-k-routing
  - 116-router-gradient
  - 094-the-residual-block
exercise: 117-balancing-the-experts
---

# Keeping the experts balanced

Two restaurants open on the same street. One gets a few more customers on the first night, by luck. Its kitchen gets more practice, its reviews improve, and next week it gets even more customers. A month later the other restaurant is empty, though nothing in the first was better at the start. The gap fed itself.

A mixture of experts has the same loop. If one expert is chosen a little more often, it sees more tokens, trains more, gets better, and the router learns to score it higher. Then it is chosen even more. Left alone, most tokens end up at one or two experts, and the others sit idle, never trained. You pay memory for eight experts and get the quality of two.

This chapter has two halves. First, a small extra loss pushes the router toward sharing the work, and the question is what to measure so the push can be trained. Second, even a balanced router makes lumpy batches, and every expert has a fixed number of seats. What happens to the tokens that find no seat, and how much does losing them hurt?

## What you already know

- The router gives each token a softmax probability for every expert, and the top $k$ run, with gate weights that sum to one.
- The top-$k$ choice is a staircase with no slope, so only the soft probabilities carry a gradient.
- A routing slot is one (token, expert) choice, so a batch of $n$ tokens has $n k$ slots.
- Training adds a small extra loss to the main loss, and gradients add.
- A block adds a sublayer's output to the token, as in "Stacking it up: residuals, layer norm, and the pre-norm block".

## The loop in numbers

Figure 1 runs the loop on four experts with a toy rule. Each expert's share of the tokens is the softmax of its router score, and each round every score rises by three times the share just received, because an expert that sees more tokens trains more. Expert A starts at 30 percent, only five points ahead of B and C. After one round A has 34 percent, after two 42, and after three 58, while D has 7. The rule is a model, not a measurement, but the shape is the one practitioners report: do nothing, and routing drifts toward collapse.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Four panels of token shares for four experts over four rounds; expert A grows from 30 to 58 percent">
<defs><marker id="c131-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="80.0" y="24.0" text-anchor="middle" class="fig-bold">round 0</text>
<line x1="21.0" y1="215.0" x2="137.0" y2="215.0" class="stroke-muted"/>
<rect x="25.0" y="125.0" width="22.0" height="90.0" class="fill-b"/>
<text x="36.0" y="120.0" text-anchor="middle" class="fig-small">30</text>
<text x="36.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="53.0" y="140.0" width="22.0" height="75.0" class="fill-soft-a stroke-a"/>
<text x="64.0" y="135.0" text-anchor="middle" class="fig-small">25</text>
<text x="64.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="81.0" y="140.0" width="22.0" height="75.0" class="fill-soft-a stroke-a"/>
<text x="92.0" y="135.0" text-anchor="middle" class="fig-small">25</text>
<text x="92.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="109.0" y="155.0" width="22.0" height="60.0" class="fill-soft-a stroke-a"/>
<text x="120.0" y="150.0" text-anchor="middle" class="fig-small">20</text>
<text x="120.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="225.0" y="24.0" text-anchor="middle" class="fig-bold">round 1</text>
<line x1="166.0" y1="215.0" x2="282.0" y2="215.0" class="stroke-muted"/>
<rect x="170.0" y="112.6" width="22.0" height="102.4" class="fill-b"/>
<text x="181.0" y="107.6" text-anchor="middle" class="fig-small">34</text>
<text x="181.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="198.0" y="141.5" width="22.0" height="73.5" class="fill-soft-a stroke-a"/>
<text x="209.0" y="136.5" text-anchor="middle" class="fig-small">24</text>
<text x="209.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="226.0" y="141.5" width="22.0" height="73.5" class="fill-soft-a stroke-a"/>
<text x="237.0" y="136.5" text-anchor="middle" class="fig-small">24</text>
<text x="237.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="254.0" y="164.4" width="22.0" height="50.6" class="fill-soft-a stroke-a"/>
<text x="265.0" y="159.4" text-anchor="middle" class="fig-small">17</text>
<text x="265.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="370.0" y="24.0" text-anchor="middle" class="fig-bold">round 2</text>
<line x1="311.0" y1="215.0" x2="427.0" y2="215.0" class="stroke-muted"/>
<rect x="315.0" y="88.3" width="22.0" height="126.7" class="fill-b"/>
<text x="326.0" y="83.3" text-anchor="middle" class="fig-small">42</text>
<text x="326.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="343.0" y="147.0" width="22.0" height="68.0" class="fill-soft-a stroke-a"/>
<text x="354.0" y="142.0" text-anchor="middle" class="fig-small">23</text>
<text x="354.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="371.0" y="147.0" width="22.0" height="68.0" class="fill-soft-a stroke-a"/>
<text x="382.0" y="142.0" text-anchor="middle" class="fig-small">23</text>
<text x="382.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="399.0" y="177.7" width="22.0" height="37.3" class="fill-soft-a stroke-a"/>
<text x="410.0" y="172.7" text-anchor="middle" class="fig-small">12</text>
<text x="410.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="515.0" y="24.0" text-anchor="middle" class="fig-bold">round 3</text>
<line x1="456.0" y1="215.0" x2="572.0" y2="215.0" class="stroke-muted"/>
<rect x="460.0" y="40.3" width="22.0" height="174.7" class="fill-b"/>
<text x="471.0" y="35.3" text-anchor="middle" class="fig-small">58</text>
<text x="471.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="488.0" y="162.8" width="22.0" height="52.2" class="fill-soft-a stroke-a"/>
<text x="499.0" y="157.8" text-anchor="middle" class="fig-small">17</text>
<text x="499.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="516.0" y="162.8" width="22.0" height="52.2" class="fill-soft-a stroke-a"/>
<text x="527.0" y="157.8" text-anchor="middle" class="fig-small">17</text>
<text x="527.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="544.0" y="194.0" width="22.0" height="21.0" class="fill-soft-a stroke-a"/>
<text x="555.0" y="189.0" text-anchor="middle" class="fig-small">7</text>
<text x="555.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="300.0" y="250.0" text-anchor="middle" class="fig-small">percent of tokens sent to each expert, A to D</text>
</svg>
Figure 1. A toy run in which each round an expert's score rises in proportion to the tokens it received. Expert A starts only 5 points ahead of B and C, yet by round 3 it holds 58 percent while D has 7 percent. The rule is a simplified model, not a measurement.
```

## Why not penalise uneven traffic directly

The obvious fix is a penalty on how uneven the traffic is. But traffic share is a count divided by a total, and a count changes in steps. As the last chapter showed, a step has no slope: nudge a logit slightly and no token changes expert, so the gradient is zero.

The other obvious fix is a penalty on the probabilities themselves. That has a gradient but controls the wrong thing: every token could give probabilities of 0.26, 0.25, 0.25, and 0.24, which are almost even, and still send every token to expert 0, because 0.26 is the largest. The probabilities look balanced and the traffic is collapsed. You want a loss that measures the real traffic but passes its gradient through the probabilities. Multiplying the two does that.

## Two quantities: f and P

Let $n$ be the number of tokens, $k$ the experts per token, and $E$ the number of experts. Define:

- $f_i$, the share of all $n k$ slots that went to expert $i$: its count of slots divided by $n k$.
- $P_i$, the average over tokens of the router's softmax probability for expert $i$.

Both sum to one: $f$ is what happened and $P$ is what the router leaned toward. The loss is

$$
L_{\text{aux}} = E \sum_{i=1}^{E} f_i \, P_i
$$

Read aloud: for each expert, multiply its share of the traffic by its average probability, add the products, and scale by the number of experts.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Three routing situations with bars for token share f and average probability P, and the computed loss under each">
<defs><marker id="c131b-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="105.0" y="24.0" text-anchor="middle" class="fig-bold">balanced</text>
<line x1="21.0" y1="215.0" x2="189.0" y2="215.0" class="stroke-muted"/>
<rect x="25.0" y="177.5" width="15.0" height="37.5" class="fill-a"/>
<rect x="41.0" y="177.5" width="15.0" height="37.5" class="fill-b"/>
<text x="40.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="65.0" y="177.5" width="15.0" height="37.5" class="fill-a"/>
<rect x="81.0" y="177.5" width="15.0" height="37.5" class="fill-b"/>
<text x="80.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="105.0" y="177.5" width="15.0" height="37.5" class="fill-a"/>
<rect x="121.0" y="177.5" width="15.0" height="37.5" class="fill-b"/>
<text x="120.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="145.0" y="177.5" width="15.0" height="37.5" class="fill-a"/>
<rect x="161.0" y="177.5" width="15.0" height="37.5" class="fill-b"/>
<text x="160.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="105.0" y="252.0" text-anchor="middle" class="fig-bold">loss = 4 × 0.250 = 1.00</text>
<text x="300.0" y="24.0" text-anchor="middle" class="fig-bold">mild skew</text>
<line x1="216.0" y1="215.0" x2="384.0" y2="215.0" class="stroke-muted"/>
<rect x="220.0" y="140.0" width="15.0" height="75.0" class="fill-a"/>
<rect x="236.0" y="155.0" width="15.0" height="60.0" class="fill-b"/>
<text x="235.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="260.0" y="177.5" width="15.0" height="37.5" class="fill-a"/>
<rect x="276.0" y="170.0" width="15.0" height="45.0" class="fill-b"/>
<text x="275.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="300.0" y="192.5" width="15.0" height="22.5" class="fill-a"/>
<rect x="316.0" y="185.0" width="15.0" height="30.0" class="fill-b"/>
<text x="315.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="340.0" y="200.0" width="15.0" height="15.0" class="fill-a"/>
<rect x="356.0" y="200.0" width="15.0" height="15.0" class="fill-b"/>
<text x="355.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="300.0" y="252.0" text-anchor="middle" class="fig-bold">loss = 4 × 0.315 = 1.26</text>
<text x="495.0" y="24.0" text-anchor="middle" class="fig-bold">collapsed</text>
<line x1="411.0" y1="215.0" x2="579.0" y2="215.0" class="stroke-muted"/>
<rect x="415.0" y="65.0" width="15.0" height="150.0" class="fill-a"/>
<rect x="431.0" y="80.0" width="15.0" height="135.0" class="fill-b"/>
<text x="430.0" y="231.0" text-anchor="middle" class="fig-small">A</text>
<rect x="455.0" y="215.0" width="15.0" height="0.0" class="fill-a"/>
<rect x="471.0" y="209.0" width="15.0" height="6.0" class="fill-b"/>
<text x="470.0" y="231.0" text-anchor="middle" class="fig-small">B</text>
<rect x="495.0" y="215.0" width="15.0" height="0.0" class="fill-a"/>
<rect x="511.0" y="210.5" width="15.0" height="4.5" class="fill-b"/>
<text x="510.0" y="231.0" text-anchor="middle" class="fig-small">C</text>
<rect x="535.0" y="215.0" width="15.0" height="0.0" class="fill-a"/>
<rect x="551.0" y="210.5" width="15.0" height="4.5" class="fill-b"/>
<text x="550.0" y="231.0" text-anchor="middle" class="fig-small">D</text>
<text x="495.0" y="252.0" text-anchor="middle" class="fig-bold">loss = 4 × 0.900 = 3.60</text>
<rect x="30.0" y="282.0" width="14.0" height="14.0" class="fill-a"/>
<text x="50.0" y="294.0" text-anchor="start" class="fig-small">f: share of tokens</text>
<rect x="210.0" y="282.0" width="14.0" height="14.0" class="fill-b"/>
<text x="230.0" y="294.0" text-anchor="start" class="fig-small">P: average probability</text>
</svg>
Figure 2. Even bars give exactly 1. As the blue bars (f) and the orange bars (P) grow tall on the same expert, their product grows, and the loss climbs toward 4 when every token goes to one expert.
```

Figure 2 shows three situations for four experts. When routing is balanced, every bar is 0.25 and the loss is $4 \times (4 \times 0.25 \times 0.25) = 1$. In the middle group $f = [0.5, 0.25, 0.15, 0.1]$ and $P = [0.4, 0.3, 0.2, 0.1]$. The products add to $0.2 + 0.075 + 0.03 + 0.01 = 0.315$, and the loss is $4 \times 0.315 = 1.26$. On the right every token goes to expert A and the router's probabilities back it, so the one big product $1 \times 0.9$ dominates and the loss is 3.6.

## Why the product works

Take the derivative with respect to $P_i$. The loss is $E \sum f_i P_i$ and $f_i$ is a fixed number during the backward pass, so $\partial L / \partial P_i = E \, f_i$. Each expert's average probability is pushed down in proportion to how busy that expert already is, and an idle expert gets no push. The gradient travels through the softmax into the router, and because each token's probabilities sum to one, pushing the busy expert down lifts the others. So $f$ has no gradient but says who is overloaded, and $P$ has a gradient but does not say who is overloaded. The product uses $f$ to decide where to push and $P$ to carry the push.

## A worked example

Take two tokens, two experts, $k = 1$. The router gives token 1 probabilities $[0.9, 0.1]$ and token 2 probabilities $[0.6, 0.4]$. Both tokens choose expert 0. There are $n k = 2$ slots, and expert 0 took both, so $f = [1, 0]$. The average probabilities are $P = [(0.9 + 0.6)/2, (0.1 + 0.4)/2] = [0.75, 0.25]$. Then

$$
L_{\text{aux}} = 2 \times (1 \times 0.75 + 0 \times 0.25) = 1.5
$$

The gradient on $P$ is $E f = [2, 0]$: push expert 0 down, leave expert 1 alone. In numpy:

```python
f = np.bincount(expert_indices.ravel(), minlength=E) / expert_indices.size
P = gate_probs.mean(axis=0)
loss = E * np.sum(f * P)
```

> [!WARNING]
> Divide the counts by $n k$, the number of slots, not by $n$. With $k = 2$ every token fills two slots, and dividing by $n$ makes $f$ sum to 2.

## What the number means

Why the factor $E$? If routing is perfectly even, every $f_i$ and $P_i$ equals $1/E$, so $\sum f_i P_i = 1/E$. The factor $E$ turns that into exactly 1 for any number of experts.

How high can it go? Each token fills at most one slot per expert, so $f_i \leq 1/k$, and since $\sum P_i = 1$ the sum $\sum f_i P_i$ is at most $1/k$. The loss is at most $E/k$. For $k = 1$ the worst case $E$ needs every token to go to one expert with probability 1, so the collapse in Figure 2 scores 3.6, below 4.

The floor is less tidy. With $E = 2$, $k = 1$, and three tokens with probabilities $[0.51, 0.49]$, $[0.51, 0.49]$, and $[0.1, 0.9]$, the first two go to expert 0 and the third to expert 1, so $f = [2/3, 1/3]$ and $P = [0.373, 0.627]$. The loss is $2 \times (0.249 + 0.209) = 0.92$, below 1, because the busier expert has the lower average probability. Treat 1 as the mark of balance, not as a floor. The loss is added to the main loss with a small weight, often around 0.01, and $f$ counts the router's own choices, before any capacity limit.

## When an expert is full anyway

Balancing nudges the average, not every batch. Eight tokens may arrive at four experts with six wanting expert 0 and two wanting expert 1. Accelerators run fastest on tensors of a fixed shape, so each expert has a fixed-size buffer. Think of each expert as a bus with a fixed number of seats, here four, and tokens boarding in arrival order. In Figure 3, t1 to t4 fill expert 0's seats, and t5 and t6 find none and are turned away, which is what dropping means. Experts 2 and 3 carry nobody, so the layer shows overflow where traffic is heavy and waste where it is light.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Four experts with four seats each. Expert 0 receives six tokens, so two are dropped; expert 1 receives two; experts 2 and 3 are empty">
<defs><marker id="c131c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="62.0" y1="100.0" x2="104.0" y2="100.0" class="stroke-bad dashed" marker-end="url(#c131c-arrow)"/>
<line x1="76.0" y1="93.0" x2="90.0" y2="107.0" class="stroke-bad"/>
<line x1="76.0" y1="107.0" x2="90.0" y2="93.0" class="stroke-bad"/>
<circle cx="40.0" cy="160.0" r="20" class="fill-paper stroke-bad"/>
<text x="40.0" y="165.0" text-anchor="middle" class="fig-bold">t6</text>
<line x1="62.0" y1="160.0" x2="104.0" y2="160.0" class="stroke-bad dashed" marker-end="url(#c131c-arrow)"/>
<line x1="76.0" y1="153.0" x2="90.0" y2="167.0" class="stroke-bad"/>
<line x1="76.0" y1="167.0" x2="90.0" y2="153.0" class="stroke-bad"/>
<text x="40.0" y="60.0" text-anchor="middle" class="fig-small">dropped</text>
<text x="300.0" y="276.0" text-anchor="middle" class="fig-small">Tokens t1 to t8 arrive in order. Six want expert 0, two want expert 1.</text>
<text x="300.0" y="296.0" text-anchor="middle" class="fig-small">Four seats fill, two tokens turn away, and ten seats elsewhere stay empty.</text>
</svg>
Figure 3. Overflow and waste at the same time. Expert 0 has more tokens than seats, so t5 and t6 are dropped, while experts 2 and 3 have no tokens at all. Tokens arrive in row order, so the early ones get the seats.
```

If routing were perfectly even, each expert would receive $n k / E$ slots. You multiply that even share by a capacity factor $\gamma$ (gamma), which buys extra seats to absorb lumpiness:

$$
\text{capacity} = \left\lceil \gamma \, \frac{n k}{E} \right\rceil
$$

Read aloud: the even share of slots per expert, times $\gamma$, rounded up to a whole number. For Figure 3, $n = 8$, $k = 1$, $E = 4$, so the even share is 2, and $\gamma = 2$ gives 4 seats. With $n = 6$, $k = 2$, $E = 3$ the even share is 4, so $\gamma = 1$ gives 4 seats and $\gamma = 0.5$ gives 2. The capacity never falls below 1, so every expert can take at least one slot. That floor is on the capacity, not on $\gamma$: a factor below 1 is allowed, and it drops slots even when routing is perfectly even. With no limit, the capacity is $n k$.

```python
capacity = max(1, math.ceil(gamma * n * k / E))
```

The simplest rule for who gets a seat is first come, first served: walk the tokens in row order, and each token's $k$ choices in order:

```python
if seats_used[e] < capacity:
    seats_used[e] += 1
    out[t] += g[t, slot] * (x[t] @ experts[e])
else:
    n_dropped += 1
```

The rule is cheap but arbitrary, and it is biased against late tokens. The exercise uses row order so that the result is checkable.

## What dropping does to a token

Dropping removes a slot, not a token. With $k = 2$, a token dropped by one expert still gets the other, with its gate weight not rescaled: if the gates were 0.73 and 0.27 and the second expert is full, the token receives only $0.73 \times \text{expert}_0(x)$. A token dropped from both gets a row of zeros.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Top row: a kept token gives x plus the expert output. Bottom row: a dropped token gives x plus zero, which is x">
<defs><marker id="c131d-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="300.0" y="22.0" text-anchor="middle" class="fig-bold">kept token</text>
<rect x="20.0" y="50.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="75.0" y="75.0" text-anchor="middle" class="fig-small">x = [0.5, −1.0]</text>
<line x1="130.0" y1="70.0" x2="200.0" y2="70.0" class="stroke-muted" marker-end="url(#c131d-arrow)"/>
<rect x="205.0" y="50.0" width="150.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="280.0" y="75.0" text-anchor="middle" class="fig-small">MoE(x) = [0.3, 0.2]</text>
<line x1="355.0" y1="70.0" x2="402.0" y2="70.0" class="stroke-muted" marker-end="url(#c131d-arrow)"/>
<circle cx="420.0" cy="70.0" r="18" class="fill-paper stroke-ink"/>
<text x="420.0" y="75.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="438.0" y1="70.0" x2="468.0" y2="70.0" class="stroke-muted" marker-end="url(#c131d-arrow)"/>
<rect x="472.0" y="50.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="527.0" y="75.0" text-anchor="middle" class="fig-small">[0.8, −0.8]</text>
<line x1="75.0" y1="50.0" x2="75.0" y2="36.0" class="stroke-a"/>
<line x1="75.0" y1="36.0" x2="420.0" y2="36.0" class="stroke-a"/>
<line x1="420.0" y1="36.0" x2="420.0" y2="52.0" class="stroke-a" marker-end="url(#c131d-arrow)"/>
<text x="300.0" y="152.0" text-anchor="middle" class="fig-bold">dropped token</text>
<rect x="20.0" y="180.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="75.0" y="205.0" text-anchor="middle" class="fig-small">x = [0.5, −1.0]</text>
<line x1="130.0" y1="200.0" x2="200.0" y2="200.0" class="stroke-muted" marker-end="url(#c131d-arrow)"/>
<rect x="205.0" y="180.0" width="150.0" height="40.0" rx="6" class="fill-sunk stroke-bad dashed"/>
<text x="280.0" y="205.0" text-anchor="middle" class="fig-small">dropped: MoE(x) = 0</text>
<line x1="355.0" y1="200.0" x2="402.0" y2="200.0" class="stroke-muted" marker-end="url(#c131d-arrow)"/>
<circle cx="420.0" cy="200.0" r="18" class="fill-paper stroke-ink"/>
<text x="420.0" y="205.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="438.0" y1="200.0" x2="468.0" y2="200.0" class="stroke-muted" marker-end="url(#c131d-arrow)"/>
<rect x="472.0" y="180.0" width="110.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="527.0" y="205.0" text-anchor="middle" class="fig-small">[0.5, −1.0]</text>
<line x1="75.0" y1="180.0" x2="75.0" y2="166.0" class="stroke-a"/>
<line x1="75.0" y1="166.0" x2="420.0" y2="166.0" class="stroke-a"/>
<line x1="420.0" y1="166.0" x2="420.0" y2="182.0" class="stroke-a" marker-end="url(#c131d-arrow)"/>
<text x="300.0" y="255.0" text-anchor="middle" class="fig-small">The skip line carries x around the expert, so the sum still contains x.</text>
</svg>
Figure 4. The skip connection (the line over the top) adds x back after the expert. When the expert is skipped its output is zero, so the sum is x itself. The token loses the expert's update, not its identity.
```

That sounds like losing the token, but a block adds the layer's output to the token, $x + \text{MoE}(x)$, instead of replacing it. In Figure 4, a kept token with $x = [0.5, -1.0]$ and an expert output of $[0.3, 0.2]$ gives $[0.8, -0.8]$. For a dropped token the expert output is zero and the sum is $x + 0 = [0.5, -1.0]$. A dropped token loses an update, not its identity. A few drops cost little quality, but many drops mean many tokens skip the layer.

## What the extra seats cost

Take one lumpy routing of 60 tokens with $k = 2$ over eight experts, 120 slots in all, and vary only $\gamma$:

| $\gamma$ | Seats per expert | Dropped slots | Share dropped |
|---|---|---|---|
| 0.5 | 8 | 59 | 49% |
| 1.0 | 15 | 27 | 22.5% |
| 1.5 | 23 | 3 | 2.5% |
| 2.0 | 30 | 0 | 0% |

At $\gamma = 1$, more than a fifth of the slots are dropped, because the busiest expert received 25 slots against 15 seats. Drops reach zero only when every buffer holds the busiest share, here 25 seats, a $\gamma$ of about 1.65. The cost is paid in every expert, busy or not: doubling $\gamma$ doubles the memory reserved for buffers and the compute spent on empty seats. A smaller factor saves memory but silently skips work for some tokens.

> [!KEY]
> A small head start lets one expert train more and be chosen more. The loss $E \sum_i f_i P_i$ stops this because $f$ measures the real traffic and $P$ carries the gradient, so the busiest expert is pushed down hardest. It equals 1 for even routing and is at most $E/k$.

> [!KEY]
> Each expert has $\lceil \gamma n k / E \rceil$ seats per batch. Overflow slots are dropped, first come first served, and because the block computes $x + \text{MoE}(x)$, a token that loses all its slots passes through unchanged. More seats cost memory in every expert.

> [!RECAP]
> Keep these points in front of you while you code:
>
> - $f_i$ is expert $i$'s slot count divided by $n k$, $P_i$ is the mean softmax probability, and the loss is $E \sum_i f_i P_i$.
> - Capacity is `max(1, ceil(gamma * n * k / E))`, and `n * k` when there is no limit.
> - Walk tokens in row order and each token's slots in order, counting seats used per expert.
> - A full expert adds nothing and adds one to the dropped count, and the surviving gate weight is not rescaled.

## Build it step by step

The router from "Send each token to a few experts" is already written for you. The checks call your functions in this order:

1. **`expert_load(expert_indices, n_experts)`.** It takes an integer array of shape `(n, k)` and returns $f$ of shape `(n_experts,)`. Use `np.bincount(..., minlength=n_experts)` and divide by `expert_indices.size`. The usual mistake is dividing by the number of tokens.
2. **`mean_gate_probability(gate_probs)`.** It averages the `(n, E)` softmax over the token axis.
3. **`load_balancing_loss(expert_indices, gate_probs, n_experts)`.** It returns the float $E \sum f_i P_i$ using your two helpers. The usual mistake is a sum over tokens instead of over experts.
4. **`expert_capacity(n_tokens, k, n_experts, capacity_factor)`.** It returns an integer. `None` means `n_tokens * k`. Otherwise round up and keep at least 1. The usual mistake is reading "at least 1" as a rule about the factor.
5. **`moe_forward(x, W_gate, experts, k, capacity_factor=None)`.** It returns `(output, n_dropped)`. The usual mistake is letting later rows board first.
6. **`dropped_fraction(n_dropped, n_tokens, k)`.** It returns the share of all `n_tokens * k` slots that were dropped. The usual mistake is dividing by the number of tokens.

## Where this breaks

A balancing weight that is too large forces even routing and can stop experts from specialising, since specialising means unequal traffic. One that is too small leaves the loop in Figure 1 running. Narrow fine-tuning data can bring collapse back, because all the tokens look alike, so many teams keep the loss on during fine-tuning. A small batch leaves most seats unused.

## The exercise

You write the six functions above. The checks confirm that `expert_load` divides by $n k$, that the loss gives 1.5 on the worked example, 1 for uniform routing, and $E$ or $E/k$ under collapse, and that `expert_capacity` rounds up with a floor of 1. They also confirm that `moe_forward` drops exactly the overflow, gives zero rows to tokens that lose their only choice, keeps the surviving gate weight, and never drops more as $\gamma$ grows.
