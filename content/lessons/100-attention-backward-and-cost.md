---
title: "The gradient and the cost of attention"
slug: 100-attention-backward-and-cost
part: "Part 6 — Transformers"
summary: Backpropagate through attention in three steps, V, softmax, then the scaled product, and see why the T by T score table makes cost grow with T squared.
minutes: 40
gpu: false
objectives:
  - Derive the softmax backward rule from the fact that the weights in a row sum to 1, and say why each score gradient row sums to 0.
  - Backpropagate through attention in three steps and give the shape of every gradient.
  - Verify a hand-written gradient against finite differences.
  - Count the flops of one attention layer, find where the score table overtakes the projections, and explain the quadratic growth.
prereqs:
  - 095-attention-gradients
  - 099-heads-and-causal-mask
exercise: 100-attention-backward-and-cost
---

# The gradient and the cost of attention

Attention is a chain of three operations: scores from queries and keys, a softmax, and a blend of values. Training needs the gradient of the loss with respect to every input, and the interesting inputs are the queries and keys, because they decide who reads whom. They sit at the start of the chain and the loss sits at the end, so the gradient must find its way back through the blend and the softmax first. How does it get there? The chapter "Backpropagation through attention" did this for the additive scorer; here it is the dot-product attention of the last three chapters, for one head. Several heads repeat the same steps head by head.

There is a second question, about size. The weights form a table with one entry for every pair of tokens. For a sentence of 100 tokens that is 10,000 entries. What happens to the cost when the sentence is ten times longer? This chapter answers both, starting with the gradient.

## What you already know

- Backpropagation goes through a chain one step at a time, each step multiplying by its own local rule.
- A weight matrix product `Y = A @ B` sends back `dA = dY @ B.T` and `dB = A.T @ dY`.
- The attention weights in each row are positive and sum to 1.
- A flop is one multiply or one add, so a `(p, q)` by `(q, r)` matrix product costs `2 * p * q * r`.

## The chain, and the three steps back

```figure
<svg viewBox="0 0 600 248" role="img" aria-label="Forward chain Q Kᵀ, division by the square root of d_k, softmax, and product with V, with the backward chain beneath it running right to left and showing the worked numbers">
<defs><marker id="c100-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<defs><marker id="c100-arrowb" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="300.0" y="18.0" text-anchor="middle" class="fig-bold">Forward</text>
<rect x="20.0" y="26.0" width="120.0" height="44.0" rx="4" class="fill-soft-a stroke-a" />
<text x="80.0" y="46.0" text-anchor="middle" class="fig-bold">Q Kᵀ</text>
<text x="80.0" y="62.0" text-anchor="middle" class="fig-small">→ scores</text>
<rect x="160.0" y="26.0" width="120.0" height="44.0" rx="4" class="fill-soft-a stroke-a" />
<text x="220.0" y="46.0" text-anchor="middle" class="fig-bold">÷ √d_k</text>
<text x="220.0" y="62.0" text-anchor="middle" class="fig-small">→ scaled</text>
<rect x="300.0" y="26.0" width="120.0" height="44.0" rx="4" class="fill-soft-a stroke-a" />
<text x="360.0" y="46.0" text-anchor="middle" class="fig-bold">softmax</text>
<text x="360.0" y="62.0" text-anchor="middle" class="fig-small">→ weights</text>
<rect x="440.0" y="26.0" width="120.0" height="44.0" rx="4" class="fill-soft-a stroke-a" />
<text x="500.0" y="46.0" text-anchor="middle" class="fig-bold">weights · V</text>
<text x="500.0" y="62.0" text-anchor="middle" class="fig-small">→ output</text>
<line x1="140.0" y1="48.0" x2="160.0" y2="48.0" class="stroke-muted" marker-end="url(#c100-arrow)"/>
<line x1="280.0" y1="48.0" x2="300.0" y2="48.0" class="stroke-muted" marker-end="url(#c100-arrow)"/>
<line x1="420.0" y1="48.0" x2="440.0" y2="48.0" class="stroke-muted" marker-end="url(#c100-arrow)"/>
<text x="300.0" y="112.0" text-anchor="middle" class="fig-bold">Backward, right to left</text>
<rect x="20.0" y="122.0" width="120.0" height="44.0" rx="4" class="fill-soft-b stroke-b" />
<rect x="160.0" y="122.0" width="120.0" height="44.0" rx="4" class="fill-soft-b stroke-b" />
<rect x="300.0" y="122.0" width="120.0" height="44.0" rx="4" class="fill-soft-b stroke-b" />
<rect x="440.0" y="122.0" width="120.0" height="44.0" rx="4" class="fill-soft-b stroke-b" />
<text x="500.0" y="142.0" text-anchor="middle" class="fig-bold">dout</text>
<text x="500.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">[1, −1]</text>
<text x="360.0" y="142.0" text-anchor="middle" class="fig-bold">dV, dweights</text>
<text x="360.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">dweights [1, −1]</text>
<text x="220.0" y="142.0" text-anchor="middle" class="fig-bold">dscores</text>
<text x="220.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">[0.3932, −0.3932]</text>
<text x="80.0" y="142.0" text-anchor="middle" class="fig-bold">dQ, dK</text>
<text x="80.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">× 1/√d_k = 0.5</text>
<line x1="440.0" y1="144.0" x2="420.0" y2="144.0" class="stroke-b" marker-end="url(#c100-arrowb)"/>
<line x1="300.0" y1="144.0" x2="280.0" y2="144.0" class="stroke-b" marker-end="url(#c100-arrowb)"/>
<line x1="160.0" y1="144.0" x2="140.0" y2="144.0" class="stroke-b" marker-end="url(#c100-arrowb)"/>
<text x="430.0" y="186.0" text-anchor="middle" class="fig-small">V step</text>
<text x="290.0" y="186.0" text-anchor="middle" class="fig-small">softmax step</text>
<text x="150.0" y="186.0" text-anchor="middle" class="fig-small">scale step</text>
<text x="300.0" y="216.0" text-anchor="middle" class="fig-small fig-mono">dV = weightsᵀ · dout, dweights = dout · Vᵀ</text>
<text x="300.0" y="234.0" text-anchor="middle" class="fig-small fig-mono">dscores = weights × (dweights − rowsum), dQ = dscores · K / √d_k</text>
</svg>
Figure 1. The forward chain on top and the gradient flowing back beneath it. The numbers are the worked example below: the softmax step turns the weight gradient [1, −1] into the score gradient [0.3932, −0.3932], and the scale step multiplies by 1/√d_k = 0.5 on the way to dQ and dK.
```

Read Figure 1 from right to left. The loss hands us `dout`, the gradient with respect to the output. We pass it back through three steps, each with its own rule, and each rule needs only that step's inputs and outputs.

**Step 1: the blend.** The output is `weights @ V`. The rule for a matrix product gives `dV = weights.T @ dout` and `dweights = dout @ V.T`. The shapes check themselves: `dout` is `(T_q, d_v)` and `V` is `(T_k, d_v)`, so `dout @ V.T` is `(T_q, T_k)`, the same shape as the weights.

**Step 2: the softmax.** This is the step to understand, because it is not a plain matrix product. Suppose you raise the score of "cat" a little. Its weight goes up, but the weights in the row must still add to 1, so the weight of "sat" goes down. One score affects every weight in its row. To find the effect of score $s_j$ on the loss, we add up its influence on every weight $w_i$. The softmax slope is $\partial w_i / \partial s_j = w_i(\delta_{ij} - w_j)$, where $\delta_{ij}$ is 1 when $i = j$ and 0 otherwise: it is $w_j(1 - w_j)$ for the weight's own score and $-w_i w_j$ for the others. (Why: $w_i = e^{s_i}/Z$, where $Z$ is the sum of the exponentials. Raising $s_j$ grows $Z$ by a share $w_j$ of itself. For $i = j$ the top grows too, which leaves $w_j - w_j^2$. For $i \ne j$ only the bottom grows, which leaves $-w_i w_j$.) Summing over $i$ gives, for one row:

$$
\frac{\partial L}{\partial s_j} = w_j\,\Big(\frac{\partial L}{\partial w_j} - \sum_i w_i \frac{\partial L}{\partial w_i}\Big)
$$

Read aloud: the gradient for a score is its weight times how much its weight's gradient beats the weight-averaged gradient of the whole row. The subtracted sum is the correction for the coupling. If every weight in the row had the same gradient, the result would be zero, which is right, because the weights cannot all rise together. A consequence you can check by eye is that the gradients of one row sum to 0. Add them up: $\sum_j w_j \, g_j$ minus (the same weighted sum times $\sum_j w_j = 1$) is 0. It matches a fact from the previous chapters: adding the same number to every score changes no weight, so the loss cannot react to such a shift, and the slopes must cancel.

**Step 3: the scaled product.** The scores are `Q @ K.T / sqrt(d_k)`, so the matrix-product rule applies with the factor carried along: `dQ = dscores @ K / sqrt(d_k)` and `dK = dscores.T @ Q / sqrt(d_k)`. The shapes work out again: `(T_q, T_k)` times `(T_k, d_k)` is `(T_q, d_k)`.

> [!KEY]
> The gradient reaches the queries and keys by going back through three local rules: the blend, the softmax with its row-sum correction, and the scaled product. The softmax rule is the one that is not a plain product, because the weights in a row are coupled by summing to 1.

## A worked example

Take one query $q = [1, 1, 1, 1]$, two keys $[1, 1, 1, 1]$ and $[0.5, 0.5, 0.5, 0.5]$, and identity values, so $d_k = 4$. The raw scores are $[4, 2]$, the scaled scores are $[4, 2]/2 = [2, 1]$, and the weights are $[0.7311, 0.2689]$. The loss gradient on the output is $\text{dout} = [1, -1]$.

- **Blend.** Because $V$ is the identity, `dweights` is `dout`, so $[1, -1]$. And `dV = weights.T @ dout` is $[[0.7311, -0.7311], [0.2689, -0.2689]]$.
- **Softmax.** The weighted row sum is $0.7311 \times 1 + 0.2689 \times (-1) = 0.4621$. Then $0.7311 \times (1 - 0.4621) = 0.3932$ and $0.2689 \times (-1 - 0.4621) = -0.3932$. The row sums to 0, as promised.
- **Scale.** `dscores @ K` is $0.3932 \times 1 - 0.3932 \times 0.5 = 0.1966$ in every column, and dividing by $\sqrt{4} = 2$ gives `dQ` $= [0.0983, 0.0983, 0.0983, 0.0983]$. For the keys, $0.3932 \times 1 / 2 = 0.1966$ for the "cat" key and $-0.1966$ for the "sat" key.

The positive score gradient on "cat" says: raising the "cat" score raises the loss for this $\text{dout}$, so gradient descent will lower it. In code:

```python
dv = weights.T @ dout
dweights = dout @ v.T
dscores = weights * (dweights - np.sum(weights * dweights, axis=-1, keepdims=True))
dq = dscores @ k / np.sqrt(d_k)
dk = dscores.T @ q / np.sqrt(d_k)
```

A masked position has weight 0, so its score gradient is $w_j(\dots) = 0$ and nothing flows to masked keys. That is why one backward function handles the mask with no extra code.

> [!TIP]
> Never trust a hand-written gradient until you have checked it numerically. Nudge one input by $\pm 10^{-6}$, measure how the scalar loss changes, and divide by $2 \times 10^{-6}$. The tests do this for every entry of `dq`, `dk`, and `dv`.

> [!WARNING]
> Keep the $1/\sqrt{d_k}$ on both `dq` and `dk`. Dropping it still produces gradients of the right shape and the right direction, but too large by a factor $\sqrt{d_k}$, so only a numerical check catches it.

## Cost: how many flops, and where they go

Counting flops answers the second question. A matrix product of a $(p, q)$ matrix and a $(q, r)$ matrix costs $2pqr$: $pqr$ multiplies and about as many adds. One layer, with $T$ tokens and width $d$, has two kinds of work.

**Projections.** There are four products of a $(T, d)$ matrix by a $(d, d)$ matrix: $W_Q$, $W_K$, $W_V$, and $W_O$. Each costs $2Td^2$, so together they cost $8Td^2$.

**The score table.** $Q K^{\top}$ is $(T, d)$ by $(d, T)$, which costs $2T^2 d$. Blending, `weights @ V`, is $(T, T)$ by $(T, d)$, which costs another $2T^2 d$. Together they cost $4T^2 d$.

$$
\text{flops} = 8 T d^2 + 4 T^2 d
$$

For $T = 16$ and $d = 4$ that is $8 \times 16 \times 16 + 4 \times 256 \times 4 = 2048 + 4096 = 6144$. The number of heads does not appear: $h$ heads each cost $2T^2 (d/h)$ for the scores, and $h \times 2T^2(d/h) = 2T^2 d$.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Line chart of flops against sequence length for d equal 64: a straight line for the projections and a rising curve for the score table, crossing at length 128">
<line x1="70.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted" />
<line x1="70.0" y1="200.0" x2="70.0" y2="30.0" class="stroke-muted" />
<line x1="70.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-grid" />
<text x="62.0" y="204.0" text-anchor="end" class="fig-small">0M</text>
<line x1="70.0" y1="160.0" x2="560.0" y2="160.0" class="stroke-grid" />
<text x="62.0" y="164.0" text-anchor="end" class="fig-small">4M</text>
<line x1="70.0" y1="120.0" x2="560.0" y2="120.0" class="stroke-grid" />
<text x="62.0" y="124.0" text-anchor="end" class="fig-small">8M</text>
<line x1="70.0" y1="80.0" x2="560.0" y2="80.0" class="stroke-grid" />
<text x="62.0" y="84.0" text-anchor="end" class="fig-small">12M</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid" />
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">16M</text>
<text x="70.0" y="218.0" text-anchor="middle" class="fig-small">0</text>
<text x="192.5" y="218.0" text-anchor="middle" class="fig-small">64</text>
<text x="315.0" y="218.0" text-anchor="middle" class="fig-small">128</text>
<text x="437.5" y="218.0" text-anchor="middle" class="fig-small">192</text>
<text x="560.0" y="218.0" text-anchor="middle" class="fig-small">256</text>
<text x="315.0" y="238.0" text-anchor="middle" class="fig-small">sequence length T (d = 64)</text>
<polyline points="70.0,200.0 85.3,197.4 100.6,194.8 115.9,192.1 131.2,189.5 146.6,186.9 161.9,184.3 177.2,181.6 192.5,179.0 207.8,176.4 223.1,173.8 238.4,171.2 253.8,168.5 269.1,165.9 284.4,163.3 299.7,160.7 315.0,158.1 330.3,155.4 345.6,152.8 360.9,150.2 376.2,147.6 391.6,144.9 406.9,142.3 422.2,139.7 437.5,137.1 452.8,134.5 468.1,131.8 483.4,129.2 498.8,126.6 514.1,124.0 529.4,121.4 544.7,118.7 560.0,116.1" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="70.0,200.0 85.3,199.8 100.6,199.3 115.9,198.5 131.2,197.4 146.6,195.9 161.9,194.1 177.2,192.0 192.5,189.5 207.8,186.7 223.1,183.6 238.4,180.2 253.8,176.4 269.1,172.3 284.4,167.9 299.7,163.1 315.0,158.1 330.3,152.7 345.6,146.9 360.9,140.9 376.2,134.5 391.6,127.7 406.9,120.7 422.2,113.3 437.5,105.6 452.8,97.6 468.1,89.2 483.4,80.6 498.8,71.5 514.1,62.2 529.4,52.5 544.7,42.5 560.0,32.2" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="315.0" cy="158.1" r="5" class="fill-ink"/>
<text x="307.0" y="146.1" text-anchor="end" class="fig-small">T = 2d = 128: both 4.2M</text>
<rect x="86.0" y="36.0" width="10.0" height="10.0" rx="4" class="fill-a" />
<text x="102.0" y="45.0" text-anchor="start" class="fig-small">projections, 8 T d²</text>
<rect x="86.0" y="54.0" width="10.0" height="10.0" rx="4" class="fill-b" />
<text x="102.0" y="63.0" text-anchor="start" class="fig-small">score table and mix, 4 T² d</text>
<text transform="rotate(-90 24 115)" x="24.0" y="115.0" text-anchor="middle" class="fig-small">flops (millions)</text>
</svg>
Figure 2. For d = 64, the projections (blue) grow in a straight line and the score table (orange) grows as a curve. They cross at T = 128 = 2d, where both cost 4.2 million flops. Beyond that point the table is the bigger bill.
```

The two terms cross where $4T^2 d = 8Td^2$, which simplifies to $T = 2d$. Below it, the projections dominate. Above it, the table does, and doubling $T$ roughly quadruples the cost. With $d = 64$, a sentence ten times longer than 128 tokens costs 10 times more in projections and 100 times more in the table.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Three square score tables for sequence lengths 4, 8, and 16 with 16, 64, and 256 cells">
<rect x="30.0" y="40.0" width="40.0" height="40.0" rx="4" class="fill-soft-b stroke-b" />
<line x1="40.0" y1="40.0" x2="40.0" y2="80.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="30.0" y1="50.0" x2="70.0" y2="50.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="50.0" y1="40.0" x2="50.0" y2="80.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="30.0" y1="60.0" x2="70.0" y2="60.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="60.0" y1="40.0" x2="60.0" y2="80.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="30.0" y1="70.0" x2="70.0" y2="70.0" class="stroke-muted" stroke-width="0.5"/>
<text x="50.0" y="22.0" text-anchor="middle" class="fig-bold">T = 4</text>
<text x="50.0" y="98.0" text-anchor="middle" class="fig-small">16 cells</text>
<rect x="130.0" y="40.0" width="80.0" height="80.0" rx="4" class="fill-soft-b stroke-b" />
<line x1="140.0" y1="40.0" x2="140.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="50.0" x2="210.0" y2="50.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="150.0" y1="40.0" x2="150.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="60.0" x2="210.0" y2="60.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="160.0" y1="40.0" x2="160.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="70.0" x2="210.0" y2="70.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="170.0" y1="40.0" x2="170.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="80.0" x2="210.0" y2="80.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="180.0" y1="40.0" x2="180.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="90.0" x2="210.0" y2="90.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="190.0" y1="40.0" x2="190.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="100.0" x2="210.0" y2="100.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="200.0" y1="40.0" x2="200.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="130.0" y1="110.0" x2="210.0" y2="110.0" class="stroke-muted" stroke-width="0.5"/>
<text x="170.0" y="22.0" text-anchor="middle" class="fig-bold">T = 8</text>
<text x="170.0" y="138.0" text-anchor="middle" class="fig-small">64 cells</text>
<rect x="270.0" y="40.0" width="160.0" height="160.0" rx="4" class="fill-soft-b stroke-b" />
<line x1="280.0" y1="40.0" x2="280.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="50.0" x2="430.0" y2="50.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="290.0" y1="40.0" x2="290.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="60.0" x2="430.0" y2="60.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="300.0" y1="40.0" x2="300.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="70.0" x2="430.0" y2="70.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="310.0" y1="40.0" x2="310.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="80.0" x2="430.0" y2="80.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="320.0" y1="40.0" x2="320.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="90.0" x2="430.0" y2="90.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="330.0" y1="40.0" x2="330.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="100.0" x2="430.0" y2="100.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="340.0" y1="40.0" x2="340.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="110.0" x2="430.0" y2="110.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="350.0" y1="40.0" x2="350.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="120.0" x2="430.0" y2="120.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="360.0" y1="40.0" x2="360.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="130.0" x2="430.0" y2="130.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="370.0" y1="40.0" x2="370.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="140.0" x2="430.0" y2="140.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="380.0" y1="40.0" x2="380.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="150.0" x2="430.0" y2="150.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="390.0" y1="40.0" x2="390.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="160.0" x2="430.0" y2="160.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="400.0" y1="40.0" x2="400.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="170.0" x2="430.0" y2="170.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="410.0" y1="40.0" x2="410.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="180.0" x2="430.0" y2="180.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="420.0" y1="40.0" x2="420.0" y2="200.0" class="stroke-muted" stroke-width="0.5"/>
<line x1="270.0" y1="190.0" x2="430.0" y2="190.0" class="stroke-muted" stroke-width="0.5"/>
<text x="350.0" y="22.0" text-anchor="middle" class="fig-bold">T = 16</text>
<text x="350.0" y="218.0" text-anchor="middle" class="fig-small">256 cells</text>
<text x="470.0" y="70.0" text-anchor="start" class="fig-bold">Double T,</text>
<text x="470.0" y="90.0" text-anchor="start" class="fig-bold">four times the cells.</text>
<text x="470.0" y="115.0" text-anchor="start" class="fig-small">4 → 8 → 16 is</text>
<text x="470.0" y="132.0" text-anchor="start" class="fig-small">16 → 64 → 256.</text>
</svg>
Figure 3. The score table for T = 4, 8, and 16. Doubling the length doubles both sides of the square, so the cell count goes 16, 64, 256: four times each step.
```

Memory follows the same square. Every head keeps its own $T \times T$ table of weights for the backward pass. At $T = 4096$ with 16 heads, that is $16 \times 4096^2 = 268$ million numbers per layer, about 1 GB at 4 bytes each.

> [!RECAP]
> Before you build:
> - Backward through attention: `dV = weights.T @ dout`, `dweights = dout @ V.T`, then the softmax rule, then `dQ = dscores @ K / sqrt(d_k)` and `dK = dscores.T @ Q / sqrt(d_k)`.
> - The softmax rule is `weights * (dweights - rowsum)`, with `rowsum = sum(weights * dweights)` along the row, and each result row sums to 0.
> - One layer costs $8Td^2 + 4T^2 d$ flops, independent of the head count.
> - The score table overtakes the projections at $T = 2d$.

## Build it step by step

1. **`softmax_backward(weights, dweights)`.** Take two `(T_q, T_k)` arrays and return the score gradient of the same shape. Compute `rowsum` with `keepdims=True`, then `weights * (dweights - rowsum)`. The usual mistake is dropping `keepdims`, so the subtraction broadcasts across the wrong axis.
2. **`attention_backward(dout, q, k, v, mask=None)`.** Recompute `scores` and `weights` (adding the mask), then apply the three steps. Return `(dq, dk, dv)` with the shapes of `q`, `k`, `v`. The usual mistake is forgetting `1 / sqrt(d_k)` on `dq` and `dk`.
3. **`attention_flops(T, d, heads)`.** Return the integer `8*T*d**2 + 4*T**2*d`. The `heads` argument is accepted and unused. The usual mistake is multiplying by the number of heads.

## Where this breaks

The quadratic table is the main limit. At tens of thousands of tokens it dominates both time and memory. **Tiled attention** computes the table one block at a time with a running softmax, so the full $T \times T$ table is never stored, and recomputes blocks in the backward pass. That brings memory down, but the flop count stays quadratic. Separately, a single layer reads each token's neighbours once. A chain of lookups, where the answer to one is the question for the next, needs stacked layers.

## The exercise

You write `softmax_backward`, `attention_backward`, and `attention_flops`. The earlier `softmax` and `scaled_dot_product_attention` are already in the file. The checks confirm the worked score gradient $[0.3932, -0.3932]$, that each softmax gradient row sums to 0 and matches finite differences, the worked `dq`, `dk`, and `dv`, finite differences for all three gradients with and without a causal mask, that `dq` keeps its $1/\sqrt{d_k}$ factor, the flop formula for every head count, and the crossover at $T = 2d$.
