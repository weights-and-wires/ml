---
title: Keep the load balanced
slug: 131-load-balancing
part: "Part 7 — Large language models"
summary: Left alone, a few experts take every token, so an auxiliary loss on traffic share times average probability pushes routing back toward even.
minutes: 40
gpu: false
objectives:
  - Explain how a small head start lets one expert take most tokens, and why the others then never learn.
  - Define f, an expert's share of the routing slots, and P, its average router probability, and compute both by hand.
  - Explain why the loss multiplies f by P instead of using either alone, and read the gradient it sends to the router.
  - Compute the load-balancing loss for even, skewed, and collapsed routing, and state honestly what range it can take.
prereqs:
  - 129-top-k-routing
  - 130-router-gradient
exercise: 131-load-balancing
---

# Keep the load balanced

Two restaurants open on the same street. One gets a few more customers on the first night, by luck. Its kitchen gets more practice, its reviews get better, and next week it gets even more customers. A month later the other restaurant is empty. Nothing in the first restaurant was better at the start. The gap fed itself.

A mixture of experts has the same loop. If one expert is chosen a little more often, it sees more tokens, trains more, gets better, and the router learns to score it higher. Then it is chosen even more. Left alone, most tokens end up at one or two experts, and the others sit idle, never trained. You pay memory for eight experts and get the quality of two.

This chapter adds a small extra loss that pushes the router toward sharing the work. The question is what to measure so that the push can be trained.

## What you already know

- The router gives each token a softmax probability for every expert, and the top $k$ run.
- The top-$k$ choice is a staircase with no slope, so only the soft probabilities can carry a gradient.
- An average over $n$ tokens adds up $n$ numbers and divides by $n$.
- A fraction of counts sums to one when you divide by the total count.
- Training adds a small extra loss to the main loss, and the gradient of the sum is the sum of the gradients.

## The loop in numbers

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

Figure 1 runs the loop on four experts with a toy rule. Each expert has a router score, and its share of the tokens is the softmax of the scores, so a higher score means more tokens. Each round, every expert's score rises by three times the share it just received, because an expert that sees more tokens trains more and so earns a higher score. Expert A starts at 30 percent, only five points ahead of B and C. After one round A has 34 percent. After two, 42. After three, 58, while D has 7. The head start grew because a larger share earns a larger push. The rule is a model of the loop, not a measurement of a real network, and real runs differ in speed. But the shape is the one practitioners report: if you do nothing, routing drifts toward collapse.

## Why not penalise uneven traffic directly

The obvious fix is a penalty on how uneven the traffic is, such as the spread of the traffic shares. The traffic share of expert $i$ is a count: how many of the tokens it received, divided by the total. A count changes in steps, and the last chapter showed why a step has no slope. Nudge a logit slightly and no token changes expert, so the penalty does not move and the gradient is zero. A penalty on the count cannot be trained.

The other obvious fix is a penalty on the probabilities themselves, such as the sum of squared average probabilities. That one has a gradient. But it controls the wrong thing. Every token could give probabilities of 0.26, 0.25, 0.25, and 0.24, which are almost even, and still send every token to expert 0, because 0.26 is the largest. The probabilities look balanced and the traffic is collapsed.

What you want is a loss that measures the real traffic but passes its gradient through the probabilities. Multiplying the two does that.

## Two quantities: f and P

Let $n$ be the number of tokens, $k$ the experts per token, and $E$ the number of experts. There are $n \times k$ routing slots in a batch, one for each choice a token makes. Define:

- $f_i$, the share of all slots that went to expert $i$: the count of slots for expert $i$ divided by $n k$.
- $P_i$, the average over tokens of the router's softmax probability for expert $i$.

The $f_i$ sum to one, and so do the $P_i$. The first is what actually happened and the second is what the router leaned toward. The loss is

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

Figure 2 shows three situations for four experts. Each group draws the traffic shares $f$ in blue and the average probabilities $P$ in orange. When routing is balanced, every bar is 0.25 and the loss is $4 \times (4 \times 0.25 \times 0.25) = 1$. In the middle group the shares are $f = [0.5, 0.25, 0.15, 0.1]$ and the probabilities are $P = [0.4, 0.3, 0.2, 0.1]$. The bars lean on the same experts, the products add to $0.2 + 0.075 + 0.03 + 0.01 = 0.315$, and the loss is $4 \times 0.315 = 1.26$. On the right every token goes to expert A and the router's probabilities back it, so the one big product $1 \times 0.9$ dominates (the other experts add only $0 \times 0.04$ and so on) and the loss is 3.6. The loss rises as the two kinds of bar line up on the same expert.

## Why the product works

Take the derivative with respect to $P_i$. The loss is $E \sum f_i P_i$ and $f_i$ is a fixed number during the backward pass, so $\partial L / \partial P_i = E \, f_i$. Each expert's average probability gets pushed down in proportion to how busy that expert already is. The busiest expert has the largest $f$ and so gets the largest push. An expert with no traffic gets no push at all. The gradient then travels through the softmax into the router, using the machinery from the last chapter. Because each token's probabilities sum to one, pushing the busy expert's probability down lifts the others, which is the shift you want.

So the product splits the job between the two quantities. $f$ is hard and has no gradient, but it says who is overloaded. $P$ is soft and has a gradient, but it does not say who is overloaded. The product uses $f$ to decide where to push and $P$ to carry the push.

## A worked example

Take two tokens, two experts, $k = 1$. The router gives token 1 probabilities $[0.9, 0.1]$ and token 2 probabilities $[0.6, 0.4]$. Both tokens choose expert 0, the larger probability.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two tokens both sent to expert 0 give f = 1, 0 and P = 0.75, 0.25, and loss 1.5">
<defs><marker id="c131c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="100.0" y="22.0" text-anchor="middle" class="fig-bold">two tokens</text>
<text x="330.0" y="22.0" text-anchor="middle" class="fig-bold">two experts</text>
<text x="520.0" y="22.0" text-anchor="middle" class="fig-bold">the two vectors</text>
<rect x="20.0" y="46.0" width="160.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="100.0" y="66.0" text-anchor="middle" class="fig-bold">token 1</text>
<text x="100.0" y="84.0" text-anchor="middle" class="fig-small">probs 0.9, 0.1</text>
<line x1="180.0" y1="70.0" x2="285.0" y2="100.0" class="stroke-b" marker-end="url(#c131c-arrow)"/>
<rect x="20.0" y="126.0" width="160.0" height="48.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="100.0" y="146.0" text-anchor="middle" class="fig-bold">token 2</text>
<text x="100.0" y="164.0" text-anchor="middle" class="fig-small">probs 0.6, 0.4</text>
<line x1="180.0" y1="150.0" x2="285.0" y2="100.0" class="stroke-b" marker-end="url(#c131c-arrow)"/>
<rect x="290.0" y="76.0" width="90.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="335.0" y="96.0" text-anchor="middle" class="fig-bold">expert 0</text>
<text x="335.0" y="114.0" text-anchor="middle" class="fig-small">2 tokens</text>
<rect x="290.0" y="176.0" width="90.0" height="48.0" rx="6" class="fill-sunk stroke-grid"/>
<text x="335.0" y="196.0" text-anchor="middle" class="fig-small">expert 1</text>
<text x="335.0" y="214.0" text-anchor="middle" class="fig-small">0 tokens</text>
<text x="500.0" y="70.0" text-anchor="middle" class="fig-mono">f = [1, 0]</text>
<text x="500.0" y="92.0" text-anchor="middle" class="fig-small">2 of 2 slots, 0 of 2</text>
<text x="500.0" y="140.0" text-anchor="middle" class="fig-mono">P = [0.75, 0.25]</text>
<text x="500.0" y="162.0" text-anchor="middle" class="fig-small">mean of the two rows</text>
<text x="300.0" y="262.0" text-anchor="middle" class="fig-bold">loss = E × (f0·P0 + f1·P1) = 2 × (1×0.75 + 0×0.25) = 1.5</text>
</svg>
Figure 3. Each number in the formula comes from a picture you have already seen. Both tokens went to expert 0, so f is [1, 0]. The mean of the two probability rows gives P. Only expert 0 has f above zero, so only P0 counts.
```

There are $n k = 2$ slots, and expert 0 took both, so $f = [2/2, 0/2] = [1, 0]$. The average probabilities are $P = [(0.9 + 0.6)/2, (0.1 + 0.4)/2] = [0.75, 0.25]$. Then

$$
L_{\text{aux}} = 2 \times (1 \times 0.75 + 0 \times 0.25) = 1.5
$$

The gradient on $P$ is $E f = [2, 0]$: push expert 0's probability down, leave expert 1's alone. Perfectly even routing would score 1, so 1.5 says the routing leans.

In numpy, the two vectors are short:

```python
f = np.bincount(expert_indices.ravel(), minlength=E) / expert_indices.size
P = gate_probs.mean(axis=0)
loss = E * np.sum(f * P)
```

> [!WARNING]
> Divide the counts by $n k$, the number of slots, not by $n$. With $k = 2$ every token fills two slots, and dividing by $n$ makes $f$ sum to 2.

## What the number means

Why the factor $E$? If routing is perfectly even, every $f_i$ and $P_i$ equals $1/E$, so $\sum f_i P_i = E \times (1/E)^2 = 1/E$. The factor $E$ turns that into exactly 1 for any number of experts, so a value near 1 always means balanced, whether you have 4 experts or 64.

How high can it go? Each token fills at most one slot per expert, so $f_i \leq 1/k$, and since $\sum P_i = 1$ the sum $\sum f_i P_i$ can be at most $1/k$. The loss is at most $E/k$. For $k = 1$, the worst case $E$ happens when every token goes to one expert with probability 1. In Figure 2 the collapse scores 3.6, below the limit of 4, because the probabilities are not quite one-hot. For $k = 2$ the worst case is $E/2$.

The floor is less tidy. Uniform routing gives 1, but 1 is not a proven minimum for every $f$ and $P$. Take $E = 2$, $k = 1$, three tokens with probabilities $[0.51, 0.49]$, $[0.51, 0.49]$, and $[0.1, 0.9]$. The first two go to expert 0 and the third to expert 1, so $f = [2/3, 1/3]$ and $P = [0.373, 0.627]$, and the loss is $2 \times (0.249 + 0.209) = 0.92$. This is slightly below 1. It happens because the expert with more traffic has the lower average probability. So treat 1 as the mark of balance, not as a floor that no routing can pass.

In training the loss is added to the main loss with a small weight, often around 0.01, so that balance nudges the router without overriding what it learned to do. The counts $f$ come from the router's own choices, before any limit on how many tokens an expert can accept. The next chapter covers that limit.

> [!KEY]
> If one expert is chosen a little more, it trains more and is chosen more, until a few experts take every token. The loss $E \sum_i f_i P_i$ stops this because $f$ measures the real traffic and $P$ carries the gradient: the busiest expert's probability is pushed down hardest. It equals 1 for even routing and is at most $E/k$.

> [!RECAP]
> Keep these points in front of you while you code:
>
> - $f_i$ is expert $i$'s count of slots divided by $n k$, so $f$ sums to one.
> - $P_i$ is the mean of the softmax probability of expert $i$ over tokens, so $P$ sums to one.
> - The loss is $E \sum_i f_i P_i$, which is 1 for even routing and at most $E/k$.
> - The gradient on $P_i$ is $E f_i$: busy experts are pushed down.
> - Use `np.bincount` and divide by `n * k`.

## Build it step by step

The checks call your functions in this order:

1. `expert_load(expert_indices, n_experts)` takes an integer array of shape `(n, k)` and returns $f$ of shape `(n_experts,)`. Count with `np.bincount(expert_indices.ravel(), minlength=n_experts)` and divide by `expert_indices.size`. The usual mistake is dividing by the number of tokens.

```python
def expert_load(expert_indices, n_experts):
    # slots per expert, divided by n * k
    return f
```

2. `mean_gate_probability(gate_probs)` takes the full softmax of shape `(n, E)` and returns $P$ of shape `(E,)`. Average over the token axis.

```python
def mean_gate_probability(gate_probs):
    # mean over axis 0
    return P
```

3. `load_balancing_loss(expert_indices, gate_probs, n_experts)` returns the float $E \sum f_i P_i$. Call your two helpers. The usual mistake is a sum over tokens instead of over experts.

```python
def load_balancing_loss(expert_indices, gate_probs, n_experts):
    # n_experts * sum(f * P)
    return loss
```

## Where this breaks

A weight that is too large forces even routing and can stop experts from specialising, since specialising means unequal traffic for tokens that differ. A weight that is too small leaves the loop in Figure 1 running. Fine-tuning on a narrow set of text can also bring collapse back, because all the tokens then look alike and the router sends them to the same few experts. Many teams therefore keep the balancing loss on during fine-tuning.

## The exercise

You write `expert_load(expert_indices, n_experts)`, `mean_gate_probability(gate_probs)`, and `load_balancing_loss(expert_indices, gate_probs, n_experts)`. The checks confirm that `expert_load` divides by $n k$ and sums to one, that `mean_gate_probability` averages over tokens, and that the loss gives 1.5 on the worked example and exactly 1 for uniform routing. They also confirm the collapse values $E$ for $k = 1$ and $E/k$ for larger $k$, and that random routings never exceed $E/k$.
