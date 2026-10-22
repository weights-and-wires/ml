---
title: Send each token to a few experts
slug: 129-top-k-routing
part: "Part 7 — Large language models"
summary: A router scores the experts for each token, keeps the top k, and mixes only those, so a layer stores many experts but runs few.
minutes: 40
gpu: false
objectives:
  - Explain why a router sends each token to only k experts, and why sending every token to every expert is the dense layer again.
  - Turn a token's router scores into gate weights by softmax, top-k, and renormalising, and check them by hand on three scores.
  - Mix the chosen experts' outputs with the gate weights, and explain why the unchosen experts cost nothing.
  - Count active and total parameters for a layer, and explain why memory follows the total while compute follows the active count.
prereqs:
  - 103-gated-feed-forward
  - 104-the-residual-block
  - 063-softmax-cross-entropy-gradient
exercise: 129-top-k-routing
---

# Send each token to a few experts

A hospital has eight specialists and a triage nurse. A patient who saw all eight would wait all day and cost the hospital eight consultations, when two would have answered the question. So the nurse looks at the patient, picks two specialists, and the hospital combines their two opinions into one answer. The hospital employs eight people but each patient uses two.

A transformer has the same problem. After attention, every token passes through a feed-forward network, which is where much of the model's knowledge is stored. To make the model know more, you widen that network, and then every token pays for the extra width, the common word "the" as much as a rare technical term. Compute grows with knowledge, one for one.

This chapter builds the nurse. You keep many feed-forward networks, called experts, and add a small router that picks a few for each token. The layer stores a lot and each token runs a little.

## What you already know

- A feed-forward network acts on each token on its own, as in "The feed-forward network and the gate: SwiGLU".
- A softmax turns a row of scores into positive numbers that sum to one, as in "Softmax and cross-entropy: a loss with a gradient you can read".
- A matrix product `x @ W` scores a vector against every column of `W` at once.
- `np.argsort(-a)` lists the positions of a row from the largest value to the smallest.
- A block adds each sublayer's output to the token, as in "Stacking it up: residuals, layer norm, and the pre-norm block".

## The picture: a nurse who picks two

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="One token scored by a router over eight experts; the top two run and are mixed with weights 0.73 and 0.27">
<defs><marker id="c129-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="14.0" y="128.0" width="62.0" height="52.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="45.0" y="150.0" text-anchor="middle" class="fig-bold">token</text>
<text x="45.0" y="168.0" text-anchor="middle" class="fig-bold">x</text>
<line x1="76.0" y1="154.0" x2="116.0" y2="154.0" class="stroke-muted" marker-end="url(#c129-arrow)"/>
<text x="215.0" y="16.0" text-anchor="middle" class="fig-bold">router scores</text>
<text x="370.0" y="16.0" text-anchor="middle" class="fig-bold">experts</text>
<rect x="120.0" y="29.0" width="154.0" height="22.0" class="fill-a"/>
<text x="280.0" y="44.0" text-anchor="start" class="fig-small">2</text>
<rect x="330.0" y="26.0" width="80.0" height="28.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="370.0" y="45.0" text-anchor="middle" class="fig-bold">expert 0</text>
<rect x="120.0" y="65.0" width="126.0" height="22.0" class="fill-a"/>
<text x="252.0" y="80.0" text-anchor="start" class="fig-small">1</text>
<rect x="330.0" y="62.0" width="80.0" height="28.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="370.0" y="81.0" text-anchor="middle" class="fig-bold">expert 1</text>
<rect x="120.0" y="101.0" width="70.0" height="22.0" class="fill-soft-a"/>
<text x="196.0" y="116.0" text-anchor="start" class="fig-small">-1</text>
<rect x="330.0" y="98.0" width="80.0" height="28.0" rx="4" class="fill-sunk stroke-grid"/>
<text x="370.0" y="117.0" text-anchor="middle" class="fig-small">expert 2</text>
<rect x="120.0" y="137.0" width="42.0" height="22.0" class="fill-soft-a"/>
<text x="168.0" y="152.0" text-anchor="start" class="fig-small">-2</text>
<rect x="330.0" y="134.0" width="80.0" height="28.0" rx="4" class="fill-sunk stroke-grid"/>
<text x="370.0" y="153.0" text-anchor="middle" class="fig-small">expert 3</text>
<rect x="120.0" y="173.0" width="98.0" height="22.0" class="fill-soft-a"/>
<text x="224.0" y="188.0" text-anchor="start" class="fig-small">0</text>
<rect x="330.0" y="170.0" width="80.0" height="28.0" rx="4" class="fill-sunk stroke-grid"/>
<text x="370.0" y="189.0" text-anchor="middle" class="fig-small">expert 4</text>
<rect x="120.0" y="209.0" width="14.0" height="22.0" class="fill-soft-a"/>
<text x="140.0" y="224.0" text-anchor="start" class="fig-small">-3</text>
<rect x="330.0" y="206.0" width="80.0" height="28.0" rx="4" class="fill-sunk stroke-grid"/>
<text x="370.0" y="225.0" text-anchor="middle" class="fig-small">expert 5</text>
<rect x="120.0" y="245.0" width="56.0" height="22.0" class="fill-soft-a"/>
<text x="182.0" y="260.0" text-anchor="start" class="fig-small">-1.5</text>
<rect x="330.0" y="242.0" width="80.0" height="28.0" rx="4" class="fill-sunk stroke-grid"/>
<text x="370.0" y="261.0" text-anchor="middle" class="fig-small">expert 6</text>
<rect x="120.0" y="281.0" width="84.0" height="22.0" class="fill-soft-a"/>
<text x="210.0" y="296.0" text-anchor="start" class="fig-small">-0.5</text>
<rect x="330.0" y="278.0" width="80.0" height="28.0" rx="4" class="fill-sunk stroke-grid"/>
<text x="370.0" y="297.0" text-anchor="middle" class="fig-small">expert 7</text>
<circle cx="500.0" cy="58.0" r="20" class="fill-paper stroke-b"/>
<text x="500.0" y="63.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="410.0" y1="40.0" x2="480.0" y2="53.0" class="stroke-b" marker-end="url(#c129-arrow)"/>
<text x="455.0" y="32.0" text-anchor="middle" class="fig-small">×0.73</text>
<line x1="410.0" y1="76.0" x2="480.0" y2="63.0" class="stroke-b" marker-end="url(#c129-arrow)"/>
<text x="455.0" y="92.0" text-anchor="middle" class="fig-small">×0.27</text>
<text x="500.0" y="100.0" text-anchor="middle" class="fig-small">output</text>
<text x="500.0" y="250.0" text-anchor="middle" class="fig-small">six experts do no</text>
<text x="500.0" y="266.0" text-anchor="middle" class="fig-small">work for this token</text>
<line x1="118.0" y1="22.0" x2="118.0" y2="306.0" class="stroke-muted"/>
<text x="14.0" y="200.0" class="fig-small">one score per</text>
<text x="14.0" y="216.0" class="fig-small">expert →</text>
<text x="215.0" y="316.0" text-anchor="middle" class="fig-small">bars start at -3.5, so every bar has some length</text>
</svg>
Figure 1. The router gives every expert a score, but only the two highest run. Their weights, 0.73 and 0.27, depend only on those two scores, so the six grey experts cost this token nothing. Bars are drawn from a baseline of -3.5 so that negative scores still show.
```

Figure 1 follows one token through the layer. On the left is the token. Next to it, the router gives each of eight experts a score, drawn as a bar (the same token is scored against all eight). The two experts with the highest scores, 2 and 1, light up. Their outputs are scaled by weights, 0.73 and 0.27, and added together into the layer's output. The other six experts never see the token.

Why does this help? Nobody assigns topics to experts. Because each expert is trained only on the tokens sent to it, experts can drift toward different kinds of tokens, which is how a hospital with specialists beats one with eight identical doctors. And because only two of eight run, a token costs about a quarter of what it would cost to run all eight.

## What the router is

The router is a tiny classifier. It has one weight vector per expert, stored as a column of a matrix $W_{\text{gate}}$ with shape `(d, E)`, where `d` is the token's width and `E` is the number of experts. The score of expert $i$ for a token $x$ is the dot product of $x$ with column $i$, so a large score means the token looks like what that expert has learned to handle. Stacking all scores gives the logits, the raw scores before any softmax:

$$
z = x \, W_{\text{gate}}
$$

Read aloud: the logits are the token times the router matrix, one number per expert. To keep the numbers small, take a token with a single feature, $x = 1$, and three experts with router weights $2$, $1$, and $0$. The logits are $z = [2, 1, 0]$. The router has no hidden layers and no memory. It is one matrix multiply, which is why it adds almost nothing to the cost of the layer.

## From scores to weights

Scores are not yet weights. You want the weights to be positive and to sum to one, so that the layer's output stays on the same scale as a single expert's output. A softmax does that. Figure 2 follows the three scores through every step.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Three panels: raw scores 2, 1, 0; softmax probabilities; and the top-2 gate weights 0.73 and 0.27">
<defs><marker id="c129b-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="115.0" y="24.0" text-anchor="middle" class="fig-bold">1. scores</text>
<text x="305.0" y="24.0" text-anchor="middle" class="fig-bold">2. softmax probabilities</text>
<text x="495.0" y="24.0" text-anchor="middle" class="fig-bold">3. keep two, divide by 0.910</text>
<line x1="36.0" y1="230.0" x2="190.0" y2="230.0" class="stroke-muted"/>
<rect x="45.0" y="110.0" width="34.0" height="120.0" class="fill-a"/>
<text x="62.0" y="104.0" text-anchor="middle" class="fig-small">2</text>
<text x="62.0" y="246.0" text-anchor="middle" class="fig-small">e0</text>
<rect x="93.0" y="170.0" width="34.0" height="60.0" class="fill-a"/>
<text x="110.0" y="164.0" text-anchor="middle" class="fig-small">1</text>
<text x="110.0" y="246.0" text-anchor="middle" class="fig-small">e1</text>
<rect x="141.0" y="230.0" width="34.0" height="0.0" class="fill-a"/>
<text x="158.0" y="224.0" text-anchor="middle" class="fig-small">0</text>
<text x="158.0" y="246.0" text-anchor="middle" class="fig-small">e2</text>
<line x1="226.0" y1="230.0" x2="380.0" y2="230.0" class="stroke-muted"/>
<rect x="235.0" y="130.2" width="34.0" height="99.8" class="fill-a"/>
<text x="252.0" y="124.2" text-anchor="middle" class="fig-small">0.665</text>
<text x="252.0" y="246.0" text-anchor="middle" class="fig-small">e0</text>
<rect x="283.0" y="193.3" width="34.0" height="36.7" class="fill-a"/>
<text x="300.0" y="187.3" text-anchor="middle" class="fig-small">0.245</text>
<text x="300.0" y="246.0" text-anchor="middle" class="fig-small">e1</text>
<rect x="331.0" y="216.5" width="34.0" height="13.5" class="fill-a"/>
<text x="348.0" y="210.5" text-anchor="middle" class="fig-small">0.090</text>
<text x="348.0" y="246.0" text-anchor="middle" class="fig-small">e2</text>
<line x1="416.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-muted"/>
<rect x="425.0" y="120.3" width="34.0" height="109.7" class="fill-a"/>
<text x="442.0" y="114.3" text-anchor="middle" class="fig-small">0.731</text>
<text x="442.0" y="246.0" text-anchor="middle" class="fig-small">e0</text>
<rect x="473.0" y="189.7" width="34.0" height="40.3" class="fill-a"/>
<text x="490.0" y="183.7" text-anchor="middle" class="fig-small">0.269</text>
<text x="490.0" y="246.0" text-anchor="middle" class="fig-small">e1</text>
<rect x="521.0" y="227.0" width="34.0" height="3.0" class="fill-muted"/>
<text x="538.0" y="220.0" text-anchor="middle" class="fig-small">0</text>
<text x="538.0" y="246.0" text-anchor="middle" class="fig-small">e2</text>
<line x1="235.0" y1="104.2" x2="317.0" y2="104.2" class="stroke-b"/>
<text x="276.0" y="96.2" text-anchor="middle" class="fig-small">0.665 + 0.245 = 0.910</text>
<line x1="185.0" y1="120.0" x2="215.0" y2="120.0" class="stroke-muted" marker-end="url(#c129b-arrow)"/>
<text x="200.0" y="110.0" text-anchor="middle" class="fig-small">softmax</text>
<line x1="375.0" y1="120.0" x2="405.0" y2="120.0" class="stroke-muted" marker-end="url(#c129b-arrow)"/>
<text x="390.0" y="110.0" text-anchor="middle" class="fig-small">top 2</text>
<text x="300.0" y="272.0" text-anchor="middle" class="fig-small">e0, e1, e2 are the three experts. Expert 2 is cut, so its weight is 0.</text>
</svg>
Figure 2. The same three experts at three stages. The kept probabilities 0.665 and 0.245 sum to only 0.910, so dividing each by 0.910 gives gate weights 0.731 and 0.269 that sum to one.
```

The exponentials of $[2, 1, 0]$ are $7.389$, $2.718$, and $1$, and they total $11.107$. Dividing each by the total gives the probabilities $0.665$, $0.245$, and $0.090$.

With $k = 2$ you keep the two highest, experts 0 and 1, and the third expert is cut. But the kept probabilities sum to $0.665 + 0.245 = 0.910$, not to one. The missing $0.090$ belonged to the expert you cut. If you used them as they are, the output would shrink whenever the cut experts held a lot of probability, and the layer's scale would drift from token to token. So divide each kept probability by their total:

$$
g_i = \frac{s_i}{\sum_{j \,\in\, \text{kept}} s_j}
$$

Here $s_i$ is the softmax probability of expert $i$, and the sum runs over the $k$ kept experts. Read aloud: the gate weight is the expert's share of the kept probability. For the example, $g_0 = 0.665 / 0.910 = 0.731$ and $g_1 = 0.245 / 0.910 = 0.269$, and these sum to one.

You could have chosen the top two first and applied a softmax to the two kept logits alone. You get the same answer, because the softmax's denominator is the same number in every probability, so it cancels in the division. The logits $[2, 1]$ give $0.731$ and $0.269$ directly. It also explains Figure 1: the weights 0.73 and 0.27 depend only on the two kept scores, 2 and 1, so the six cut experts could have any lower scores and the weights would not change.

In numpy, the first two steps are two lines:

```python
logits = x @ W_gate                          # (n, E)
top = np.argsort(-logits, axis=-1)[:, :k]    # (n, k), best first
```

Negating the logits makes the largest come first in the sort. A frequent mistake is to sort `logits` ascending and keep the smallest scores. After this, `np.take_along_axis(probs, top, axis=-1)` picks the kept probabilities, and dividing by their row sum gives the gate weights.

> [!WARNING]
> Subtract each row's maximum logit before you exponentiate. Logits of 1000 overflow `np.exp` to infinity, and the result is not a number. The shift changes no probability, because it multiplies the top and bottom of every fraction by the same factor.

## Mix the chosen experts

An expert is a function that takes a token and returns an output. To keep every check about routing, an expert here is a single matrix of shape `(d, d_out)`, and its output for token $x$ is `x @ expert`. A real expert is a small two-layer network, but the router and the mixing rule are the same.

Suppose the chosen experts return $4$ for expert 0 and $10$ for expert 1. (Expert 2, the one that was cut, would have returned 100, which is why it matters that it contributes nothing.) The layer's output is the weighted sum:

$$
\hat{y} = \sum_{i \,\in\, \text{kept}} g_i \cdot \text{expert}_i(x)
$$

Read aloud: the output is each chosen expert's answer, scaled by its gate weight, added up. For the example, $0.731 \times 4 + 0.269 \times 10 = 2.924 + 2.690 = 5.614$ with the rounded weights (the unrounded weights give 5.6136, which also rounds to 5.614). The answer sits between the two expert outputs, closer to expert 0's 4 because expert 0 holds more of the weight. In numpy, one token is one loop over its chosen experts:

```python
for slot in range(k):
    e = top[t, slot]
    out[t] += gate_weights[t, slot] * (x[t] @ experts[e])
```

Why not send every token to every expert and weight them by the softmax? Run that and you get a soft mixture, which gives every expert some weight and so needs every expert's output. That is the dense layer again, and it costs as much as running all eight. Setting $k = E$ shows the link: with $k = E$ no expert is cut, nothing is renormalised, and the sparse layer becomes the dense weighted sum. Top-$k$ routing is that dense sum with the small terms deleted, and deleting them is where the savings come from.

## Stored versus used

Take eight experts of 100 million parameters each, $k = 2$, and 100 million parameters that every token uses regardless, such as attention weights, which we call shared. The layer stores $8 \times 100 + 100 = 900$ million parameters. A token runs only $2 \times 100 + 100 = 300$ million.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Two stacked bars: 900 million parameters stored, 300 million used per token">
<defs><marker id="c129c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="110.0" y="253.3" width="120.0" height="26.7" class="fill-soft-c stroke-paper"/>
<text x="170.0" y="270.7" text-anchor="middle" class="fig-small">shared 100M</text>
<rect x="110.0" y="226.7" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="244.0" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="200.0" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="217.3" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="173.3" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="190.7" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="146.7" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="164.0" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="120.0" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="137.3" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="93.3" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="110.7" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="66.7" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="84.0" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="110.0" y="40.0" width="120.0" height="26.7" class="fill-soft-a stroke-paper"/>
<text x="170.0" y="57.3" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="340.0" y="253.3" width="120.0" height="26.7" class="fill-soft-c stroke-paper"/>
<text x="400.0" y="270.7" text-anchor="middle" class="fig-small">shared 100M</text>
<rect x="340.0" y="226.7" width="120.0" height="26.7" class="fill-soft-b stroke-paper"/>
<text x="400.0" y="244.0" text-anchor="middle" class="fig-small">expert 100M</text>
<rect x="340.0" y="200.0" width="120.0" height="26.7" class="fill-soft-b stroke-paper"/>
<text x="400.0" y="217.3" text-anchor="middle" class="fig-small">expert 100M</text>
<text x="170.0" y="300.0" text-anchor="middle" class="fig-bold">stored (memory)</text>
<text x="400.0" y="300.0" text-anchor="middle" class="fig-bold">run per token (compute)</text>
<text x="170.0" y="32.0" text-anchor="middle" class="fig-bold">900M</text>
<text x="400.0" y="192.0" text-anchor="middle" class="fig-bold">300M</text>

</svg>
Figure 3. All eight experts and the shared part must sit in memory, the tall bar. A token runs only two experts and the shared part, the short bar, so compute is one third of the memory bill.
```

$$
P_{\text{total}} = E \cdot P_{\text{expert}} + P_{\text{shared}}, \qquad
P_{\text{active}} = k \cdot P_{\text{expert}} + P_{\text{shared}}
$$

Read aloud: stored parameters count every expert, and active parameters count only the $k$ the token visits, with the shared part in both. Think of a library and the books you read in one afternoon. Memory has to hold the whole library, so it follows the tall bar in Figure 3. The work you do follows the books you actually walked to, so compute per token follows the short bar. As $k$ approaches $E$ the two bars meet, and at $k = E$ they are equal.

```python
active = k * expert_params + shared_params
total = n_experts * expert_params + shared_params
```

> [!KEY]
> A router scores every expert, keeps the top $k$, divides the kept probabilities by their total so the weights sum to one, and mixes only those experts. The triage question has an answer: pick two, then combine them with weights that add up to one. The layer stores all $E$ experts but each token pays for $k$.

> [!RECAP]
> Keep these points in front of you while you code:
>
> - The router is one matrix multiply, `x @ W_gate`, with one logit per expert.
> - Softmax over all experts, keep the top $k$ by logit, and divide the kept probabilities by their sum.
> - The output is the gate-weighted sum of `x[t] @ experts[e]` over the chosen experts only.
> - Active parameters are $k P_{\text{expert}} + P_{\text{shared}}$, stored parameters use $E$, and $k = E$ is the dense layer.
> - Subtract the row maximum before the softmax.

## Build it step by step

The checks call your functions in this order:

1. `router(x, W_gate, k)` takes `x` of shape `(n, d)` and returns `topk_indices` `(n, k)`, `gate_weights` `(n, k)`, and `full_probs` `(n, E)`. Use `np.argsort(-logits, axis=-1)[:, :k]` for the indices, so the best expert comes first. The usual mistake is dividing by the sum of all `E` probabilities, which gives the probabilities back instead of gate weights.

```python
def router(x, W_gate, k):
    # logits, softmax with the max subtracted, top-k, then renormalise
    return topk_indices, gate_weights, full_probs
```

2. `moe_forward(x, W_gate, experts, k)` returns an array of shape `(n, d_out)`. Call `router` once, then loop over tokens and slots. The usual mistake is adding an expert's output without its gate weight.

```python
def moe_forward(x, W_gate, experts, k):
    # out[t] += gate_weights[t, slot] * (x[t] @ experts[e])
    return output
```

3. `active_parameters(n_experts, k, expert_params, shared_params)` returns the pair `(active, total)`. The usual mistake is leaving `shared_params` out of one of the two counts.

```python
def active_parameters(n_experts, k, expert_params, shared_params):
    # k experts plus shared, and all experts plus shared
    return active, total
```

## Where this breaks

With $k = 1$ the single gate weight is always exactly 1, so the weight carries no information. The next chapter shows why that matters for training the router. Two experts with equal logits also tie, and `argsort` then picks whichever it lists first, so results at exact ties depend on the sort. And the stored count never shrinks: every expert must sit in memory even if a given batch uses only a few.

## The exercise

You write `router(x, W_gate, k)`, `moe_forward(x, W_gate, experts, k)`, and `active_parameters(n_experts, k, expert_params, shared_params)`. The checks confirm the true top-$k$ experts come back best first, the worked example gives weights 0.7311 and 0.2689, the gate weights sum to one and equal a softmax over the kept logits, and huge logits do not overflow. They also confirm `moe_forward` reproduces the hand mix 5.614, equals the dense weighted sum when $k = E$, ignores experts a token did not choose, and that `active_parameters` returns 300 million and 900 million for the example.
