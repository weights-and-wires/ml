---
title: "Mixtures of experts: routing, the router's gradient, and balance"
slug: 066-mixtures-of-experts
part: "Part 7 — Large language models"
summary: A router sends each token to a few experts. It learns through the gate weights, and a balancing loss and a seat limit keep the load even.
minutes: 80
gpu: false
objectives:
  - Turn a token's router scores into gate weights by softmax, top-k, and renormalising, and mix only the chosen experts.
  - Explain why a top-k choice has zero slope, and why the gate weight is the router's only path to a gradient.
  - Compute the gradient on each gate logit by hand, and assemble the router's weight gradient with a matrix product.
  - Compute the load-balancing loss for even, skewed, and collapsed routing, and explain why it multiplies traffic share by average probability.
  - Compute an expert's capacity from the capacity factor, and run a forward pass that drops overflow slots in row order.
prereqs:
  - 056-the-transformer-block
  - 032-softmax-and-whole-network-gradients
exercise: 066-mixtures-of-experts
---

# Mixtures of experts: routing, the router's gradient, and balance

A hospital has eight specialists and a triage nurse. A patient who saw all eight would wait all day, when two would have answered the question. So the nurse picks two specialists, and the hospital combines their two opinions into one answer. The hospital employs eight people, but each patient uses two.

A transformer has the same problem. Every token passes through a feed-forward network, where much of the model's knowledge is stored, and widening that network makes every token pay for the extra width. This chapter builds the nurse: many feed-forward networks, called experts, and a small router that picks a few for each token.

Three questions follow. How does the router turn scores into a choice, and how does the layer mix the chosen experts? A choice like "send the patient to Dr. A" cannot be nudged a little, so how does the router learn at all? And what stops the nurse from sending every patient to the same two doctors?

## What you already know

- A feed-forward network acts on each token on its own, and a block adds its output to the token, as in [the transformer block](/lessons/056-the-transformer-block).
- A softmax turns a row of scores into positive numbers that sum to one, and its slope is $g_i(1 - g_i)$ on the diagonal and $-g_i g_j$ off it, as in [softmax and whole-network gradients](/lessons/032-softmax-and-whole-network-gradients).
- The chain rule multiplies slopes along a path, and `x.T @ dz` collects a gradient on outputs into a gradient on weights.
- `np.argsort(-a)` lists the positions of a row from the largest value to the smallest.

## The picture: a nurse who picks two

Figure 1 follows one token. The router gives each of eight experts a score, drawn as a bar. The two experts with the highest scores, 2 and 1, run, and their outputs are scaled by weights 0.73 and 0.27 and added. The other six never see the token, so it costs about a quarter of running all eight. Nobody assigns topics to experts: each trains only on the tokens sent to it, so experts can drift toward different kinds of tokens.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="One token scored by a router over eight experts; the top two run and are mixed with weights 0.73 and 0.27">
<defs><marker id="c055-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="14.0" y="128.0" width="62.0" height="52.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="45.0" y="150.0" text-anchor="middle" class="fig-bold">token</text>
<text x="45.0" y="168.0" text-anchor="middle" class="fig-bold">x</text>
<line x1="76.0" y1="154.0" x2="116.0" y2="154.0" class="stroke-muted" marker-end="url(#c055-arrow)"/>
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
<line x1="410.0" y1="40.0" x2="480.0" y2="53.0" class="stroke-b" marker-end="url(#c055-arrow)"/>
<text x="455.0" y="32.0" text-anchor="middle" class="fig-small">×0.73</text>
<line x1="410.0" y1="76.0" x2="480.0" y2="63.0" class="stroke-b" marker-end="url(#c055-arrow)"/>
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

## From scores to weights

The router is a tiny classifier: a matrix $W_{\text{gate}}$ of shape `(d, E)`, one column per expert. The score of expert $i$ for a token $x$ is the dot product of $x$ with column $i$, so a large score means the token looks like what that expert handles. The logits, the raw scores before any softmax, are

$$
z = x \, W_{\text{gate}}
$$

Read it aloud: the token times the router matrix, one number per expert, at almost no cost. Take a token $x = 1$ and three experts with router weights $2$, $1$, and $0$, so $z = [2, 1, 0]$.

A softmax turns scores into positive weights that sum to one. The exponentials of $[2, 1, 0]$ are $7.389$, $2.718$, and $1$, which total $11.107$, so the probabilities are $0.665$, $0.245$, and $0.090$. With $k = 2$ you keep experts 0 and 1 and cut the third. But the kept probabilities sum to $0.910$, not one, and using them as they are would shrink the output whenever the cut experts held a lot of probability. So divide each kept probability by their total:

$$
g_i = \frac{s_i}{\sum_{j \,\in\, \text{kept}} s_j}
$$

Here $s_i$ is the softmax probability of expert $i$. Read it aloud: the gate weight is the expert's share of the kept probability. For the example, $g_0 = 0.665 / 0.910 = 0.731$ and $g_1 = 0.245 / 0.910 = 0.269$. The same weights come from a softmax over the two kept logits alone, because the softmax denominator is common to every probability and cancels in the division. That is why Figure 1's weights depend only on the two kept scores. In numpy, the first two steps are:

```python
logits = x @ W_gate                          # (n, E)
top = np.argsort(-logits, axis=-1)[:, :k]    # (n, k), best first
```

Then `np.take_along_axis(probs, top, axis=-1)` picks the kept probabilities, and dividing by their row sum gives the gate weights.

> [!WARNING]
> Subtract each row's maximum logit before you exponentiate. Logits of 1000 overflow `np.exp` to infinity, and the result is not a number. The shift changes no probability, because it multiplies the top and bottom of every fraction by the same factor.

## Mix the chosen experts

Here an expert is a single matrix of shape `(d, d_out)`, so its output is `x @ expert`. A real expert is a small network, but the mixing rule is the same. The layer's output is the weighted sum over the chosen experts:

$$
\hat{y} = \sum_{i \,\in\, \text{kept}} g_i \cdot \text{expert}_i(x)
$$

Suppose expert 0 returns $4$ and expert 1 returns $10$. The output is $0.731 \times 4 + 0.269 \times 10 = 5.614$, closer to expert 0's 4 because it holds more of the weight. In numpy, one token is one loop over its chosen slots:

```python
for slot in range(k):
    e = top[t, slot]
    out[t] += gate_weights[t, slot] * (x[t] @ experts[e])
```

Why not weight every expert by the softmax? That needs every expert's output, which is the dense layer at full cost. At $k = E$ nothing is cut, and the sparse layer is the dense weighted sum, so top-$k$ routing is that sum with the small terms deleted.

The savings are in compute, not memory. With eight experts of 100 million parameters, $k = 2$, and 100 million shared parameters, the layer stores 900 million but a token runs $2 \times 100 + 100 = 300$ million.

> [!KEY]
> A router scores every expert, keeps the top $k$, divides the kept probabilities by their total so the weights sum to one, and mixes only those experts. The layer stores all $E$ experts, but each token pays for $k$.

## Why the choice has no slope

The patient did badly, and "pick a different pair" is not something you can nudge a little. A router chooses with an `argsort`, a hard yes-or-no. Nudge one score slightly and the same experts still win, so the output and the loss do not change. Training follows the slope of the loss, so a flat choice leaves the router with nothing to follow, and every token would be routed by chance.

Figure 2 slides the logit of expert 0 from $-2$ to $2$ with expert 1 held at 0. On the left the winner is a staircase with one step at the tie, so nudging the logit from $-1$ to $-0.6$ changes nothing. On the right, the weight expert 0 gets when the two are mixed rises smoothly, and the same nudge moves it from 0.27 to 0.35.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Left: which expert wins is a step function of logit 0. Right: the softmax weight on expert 0 is a smooth curve">
<defs><marker id="c055b-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="150.0" y="24.0" text-anchor="middle" class="fig-bold">which expert wins</text>
<line x1="40.0" y1="210.0" x2="260.0" y2="210.0" class="stroke-muted"/>
<line x1="40.0" y1="50.0" x2="40.0" y2="210.0" class="stroke-muted"/>
<line x1="40.0" y1="210.0" x2="40.0" y2="214.0" class="stroke-muted"/>
<text x="40.0" y="228.0" text-anchor="middle" class="fig-small">-2</text>
<line x1="95.0" y1="210.0" x2="95.0" y2="214.0" class="stroke-muted"/>
<text x="95.0" y="228.0" text-anchor="middle" class="fig-small">-1</text>
<line x1="150.0" y1="210.0" x2="150.0" y2="214.0" class="stroke-muted"/>
<text x="150.0" y="228.0" text-anchor="middle" class="fig-small">0</text>
<line x1="205.0" y1="210.0" x2="205.0" y2="214.0" class="stroke-muted"/>
<text x="205.0" y="228.0" text-anchor="middle" class="fig-small">1</text>
<line x1="260.0" y1="210.0" x2="260.0" y2="214.0" class="stroke-muted"/>
<text x="260.0" y="228.0" text-anchor="middle" class="fig-small">2</text>
<text x="150.0" y="248.0" text-anchor="middle" class="fig-small">logit of expert 0 (expert 1 fixed at 0)</text>
<text x="450.0" y="24.0" text-anchor="middle" class="fig-bold">weight on expert 0</text>
<line x1="340.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-muted"/>
<line x1="340.0" y1="50.0" x2="340.0" y2="210.0" class="stroke-muted"/>
<line x1="340.0" y1="210.0" x2="340.0" y2="214.0" class="stroke-muted"/>
<text x="340.0" y="228.0" text-anchor="middle" class="fig-small">-2</text>
<line x1="395.0" y1="210.0" x2="395.0" y2="214.0" class="stroke-muted"/>
<text x="395.0" y="228.0" text-anchor="middle" class="fig-small">-1</text>
<line x1="450.0" y1="210.0" x2="450.0" y2="214.0" class="stroke-muted"/>
<text x="450.0" y="228.0" text-anchor="middle" class="fig-small">0</text>
<line x1="505.0" y1="210.0" x2="505.0" y2="214.0" class="stroke-muted"/>
<text x="505.0" y="228.0" text-anchor="middle" class="fig-small">1</text>
<line x1="560.0" y1="210.0" x2="560.0" y2="214.0" class="stroke-muted"/>
<text x="560.0" y="228.0" text-anchor="middle" class="fig-small">2</text>
<text x="450.0" y="248.0" text-anchor="middle" class="fig-small">logit of expert 0 (expert 1 fixed at 0)</text>
<line x1="40.0" y1="190.0" x2="150.0" y2="190.0" class="stroke-b"/>
<line x1="150.0" y1="190.0" x2="150.0" y2="70.0" class="stroke-b dashed"/>
<line x1="150.0" y1="70.0" x2="260.0" y2="70.0" class="stroke-b"/>
<text x="48.0" y="205.0" text-anchor="start" class="fig-small">expert 1</text>
<text x="252.0" y="62.0" text-anchor="end" class="fig-small">expert 0</text>
<polyline points="340.0,190.9 342.8,190.1 345.5,189.2 348.2,188.3 351.0,187.3 353.8,186.3 356.5,185.3 359.2,184.2 362.0,183.1 364.8,182.0 367.5,180.8 370.2,179.6 373.0,178.3 375.8,177.1 378.5,175.7 381.2,174.4 384.0,173.0 386.8,171.5 389.5,170.0 392.2,168.5 395.0,167.0 397.8,165.4 400.5,163.8 403.2,162.1 406.0,160.4 408.8,158.7 411.5,156.9 414.2,155.1 417.0,153.3 419.8,151.5 422.5,149.6 425.2,147.7 428.0,145.8 430.8,143.9 433.5,141.9 436.2,139.9 439.0,138.0 441.8,136.0 444.5,134.0 447.2,132.0 450.0,130.0 452.8,128.0 455.5,126.0 458.2,124.0 461.0,122.0 463.8,120.1 466.5,118.1 469.2,116.1 472.0,114.2 474.8,112.3 477.5,110.4 480.2,108.5 483.0,106.7 485.8,104.9 488.5,103.1 491.2,101.3 494.0,99.6 496.8,97.9 499.5,96.2 502.2,94.6 505.0,93.0 507.8,91.5 510.5,90.0 513.2,88.5 516.0,87.0 518.8,85.6 521.5,84.3 524.2,82.9 527.0,81.7 529.8,80.4 532.5,79.2 535.2,78.0 538.0,76.9 540.8,75.8 543.5,74.7 546.2,73.7 549.0,72.7 551.8,71.7 554.5,70.8 557.2,69.9 560.0,69.1" fill="none" class="stroke-b" stroke-width="2"/>
<text x="334.0" y="214.0" text-anchor="end" class="fig-small">0</text>
<text x="334.0" y="134.0" text-anchor="end" class="fig-small">0.5</text>
<text x="334.0" y="54.0" text-anchor="end" class="fig-small">1</text>
<circle cx="395.0" cy="167.0" r="4" class="fill-a"/>
<circle cx="417.0" cy="153.3" r="4" class="fill-a"/>
<text x="406.0" y="188.4" text-anchor="middle" class="fig-small">a nudge of 0.4</text>
<text x="406.0" y="203.4" text-anchor="middle" class="fig-small">moves it 0.09</text>
<circle cx="95.0" cy="190.0" r="4" class="fill-a"/>
<circle cx="117.0" cy="190.0" r="4" class="fill-a"/>
<text x="106.0" y="160.0" text-anchor="middle" class="fig-small">same nudge,</text>
<text x="106.0" y="176.0" text-anchor="middle" class="fig-small">no change</text>
</svg>
Figure 2. Left: as the logit of expert 0 rises, the winner flips once and stays flat on both sides, so the slope is zero almost everywhere. Right: the weight on expert 0 rises smoothly over the same range, so a nudge always changes something.
```

Making the choice smooth would be the dense layer again. So the router keeps the hard choice and learns through the smooth weight on top of it. The same logits decide the choice and set the trust, so tuning the trust moves the choice over many steps.

## The loss slope per unit of an expert

The layer's output is $\hat{y} = \sum_i g_i \, e_i$, where $e_i$ is expert $i$'s output vector. Backpropagation hands the layer $\bar{y}$, the slope of the loss with respect to $\hat{y}$, so the slope with respect to a gate weight $g_i$ is

$$
c_i = \bar{y} \cdot e_i
$$

Read it aloud: $c_i$ is how much the loss moves per unit of weight on expert $i$. It is positive when the expert's output points along the loss gradient, so more weight on it makes things worse, and negative when more weight helps. With $c = [1, 3]$ and gates 0.73 and 0.27, the mixed score $\sum g_i c_i$ is $1.54$. Moving 0.1 of the weight toward expert 1 raises it to 1.74, and moving 0.1 toward expert 0 lowers it to 1.34, so the router should move weight away from the expert with the larger $c$.

## From weights back to logits

The router sets logits $z$, and a softmax turns them into weights, so the weights cannot move independently: raising one lowers the others. Adding up the effect of logit $z_j$ on every weight, using the softmax slopes above, gives

$$
\frac{\partial L}{\partial z_j} = g_j \left( c_j - \bar{c} \right), \qquad \bar{c} = \sum_{i} g_i \, c_i
$$

The algebra: a nudge to $z_j$ moves its own weight by $g_j(1-g_j)$ and every other weight $g_i$ by $-g_i g_j$, each times its $c$, and pulling out $g_j$ leaves $g_j (c_j - \bar{c})$. Read it aloud: the gradient on a logit is its gate weight times the gap between its $c$ and the weighted average $\bar{c}$. Only the gap matters, because a logit that rises takes weight from the others.

With gates 0.7311 and 0.2689 and $c = [1, 3]$, $\bar{c} = 1.5378$. Expert 0 gets $0.7311 \times (1 - 1.5378) = -0.3932$ and expert 1 gets $+0.3932$. They sum to zero, because a gap below average for one expert is above average for the other. Training subtracts the gradient, so expert 0's logit rises (it adds less loss than average, so the router trusts it more) and expert 1's falls. An unchosen expert has weight 0 whatever its logit does, so it gets zero gradient from this token. In code:

```python
c_bar = np.sum(g * c)
dz_chosen = g * (c - c_bar)
```

The logits come from $z = x \, W_{\text{gate}}$, so the router's gradient follows the rule of every linear layer. Stack each token's logit gradient as a row of `dz` with shape `(n, E)`, zero at the unchosen experts. Then

$$
\frac{\partial L}{\partial W_{\text{gate}}} = x^{\top} \, dz
$$

Read it aloud: each router weight collects the logit gradients of every token, scaled by that token's feature.

```python
dz = np.zeros((n, E))
dz[t, chosen] = g * (c - c_bar)   # for each token t; other columns stay zero
grad_W = x.T @ dz
```

The checks confirm this gradient with finite differences of a surrogate loss: the sum over tokens and chosen experts of $g \cdot c$, with expert outputs and $\bar{y}$ held fixed.

> [!KEY]
> The top-$k$ choice is a staircase and has no slope. The router learns through the gate weights: each chosen logit gets its weight times how far that expert's loss slope sits from the weighted average, and the gradients sum to zero. Nudging a logit never changes who wins, so the router learns by changing how much it trusts the winners.

## The loop that empties experts

Two restaurants open on one street. One gets a few more customers on the first night, by luck, so it gets more practice and better reviews, and next week it gets even more. A month later the other is empty.

A mixture of experts has the same loop: an expert chosen a little more often sees more tokens, trains more, and the router learns to score it higher. In a toy rule on four experts, each expert's share of tokens is the softmax of its score, and each round every score rises by three times the share just received. Expert A starts at 30 percent, five points ahead of B and C, and holds 34 percent after one round, 42 after two, and 58 after three, while D has 7. The rule is a model, not a measurement, but the shape matches what practitioners report: left alone, routing drifts toward collapse, and you pay memory for eight experts and get the quality of two.

## Why not penalise uneven traffic directly

The obvious fix is a penalty on uneven traffic. But traffic share is a count divided by a total, and a count changes in steps, so its gradient is zero, as with the staircase. A penalty on the router's probabilities has a gradient but controls the wrong thing: every token could give probabilities 0.26, 0.25, 0.25, and 0.24, almost even, and still send every token to expert 0. You want a loss that measures the real traffic and passes its gradient through the probabilities. Multiplying the two does that.

Let $n$ be the number of tokens, $k$ the experts per token, and $E$ the number of experts. A slot is one (token, expert) choice, so there are $n k$ slots. Define two quantities:

- $f_i$, the share of all slots that went to expert $i$.
- $P_i$, the average over tokens of the router's softmax probability for expert $i$.

Both sum to one: $f$ is what happened and $P$ is what the router leaned toward. The loss is

$$
L_{\text{aux}} = E \sum_{i=1}^{E} f_i \, P_i
$$

Read it aloud: for each expert, multiply its traffic share by its average probability, add the products, and scale by $E$. In Figure 3, balanced routing has every bar at 0.25, so the loss is $4 \times (4 \times 0.25 \times 0.25) = 1$. In the middle group $f = [0.5, 0.25, 0.15, 0.1]$ and $P = [0.4, 0.3, 0.2, 0.1]$, so the loss is $4 \times 0.315 = 1.26$. On the right every token goes to expert A and the probabilities back it, so the one big product $1 \times 0.9$ dominates and the loss is 3.6.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Three routing situations with bars for token share f and average probability P, and the computed loss under each">
<defs><marker id="c055c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
Figure 3. Even bars give exactly 1. As the blue bars (f) and the orange bars (P) grow tall on the same expert, their product grows, and the loss climbs toward 4 when every token goes to one expert.
```

Why does the product work? In the backward pass $f_i$ is a fixed number, so $\partial L / \partial P_i = E \, f_i$: each expert's average probability is pushed down in proportion to how busy it already is, and an idle expert gets no push. Because each token's probabilities sum to one, pushing the busy expert down lifts the others. So $f$ says who is overloaded but has no gradient, and $P$ carries a gradient but cannot say who is overloaded.

Work a small case: two tokens, two experts, $k = 1$, with probabilities $[0.9, 0.1]$ and $[0.6, 0.4]$, both choosing expert 0. Then $f = [1, 0]$, $P = [0.75, 0.25]$, and $L_{\text{aux}} = 2 \times 0.75 = 1.5$. The gradient on $P$ is $E f = [2, 0]$: push expert 0 down only.

```python
f = np.bincount(expert_indices.ravel(), minlength=E) / expert_indices.size
P = gate_probs.mean(axis=0)
loss = E * np.sum(f * P)
```

> [!WARNING]
> Divide the counts by $n k$, the number of slots, not by $n$. With $k = 2$ every token fills two slots, and dividing by $n$ makes $f$ sum to 2.

The factor $E$ makes even routing score exactly 1 for any number of experts, and the loss is at most $E/k$ because each token fills at most one slot per expert. Treat 1 as the mark of balance, not as a floor. The loss joins the main loss with a small weight, often around 0.01.

> [!KEY]
> A small head start lets one expert train more and be chosen more. The loss $E \sum_i f_i P_i$ stops this because $f$ measures the real traffic and $P$ carries the gradient, so the busiest expert is pushed down hardest. It equals 1 for even routing and is at most $E/k$.

## When an expert is full anyway

Balancing nudges the average, not every batch. Accelerators run fastest on tensors of a fixed shape, so each expert has a fixed number of seats, like a bus, and tokens board in arrival order. In Figure 4, six of eight tokens want expert 0, t1 to t4 fill its four seats, and t5 and t6 are dropped, while experts 2 and 3 carry nobody.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Four experts with four seats each. Expert 0 receives six tokens, so two are dropped; expert 1 receives two; experts 2 and 3 are empty">
<defs><marker id="c055d-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="62.0" y1="100.0" x2="104.0" y2="100.0" class="stroke-bad dashed" marker-end="url(#c055d-arrow)"/>
<line x1="76.0" y1="93.0" x2="90.0" y2="107.0" class="stroke-bad"/>
<line x1="76.0" y1="107.0" x2="90.0" y2="93.0" class="stroke-bad"/>
<circle cx="40.0" cy="160.0" r="20" class="fill-paper stroke-bad"/>
<text x="40.0" y="165.0" text-anchor="middle" class="fig-bold">t6</text>
<line x1="62.0" y1="160.0" x2="104.0" y2="160.0" class="stroke-bad dashed" marker-end="url(#c055d-arrow)"/>
<line x1="76.0" y1="153.0" x2="90.0" y2="167.0" class="stroke-bad"/>
<line x1="76.0" y1="167.0" x2="90.0" y2="153.0" class="stroke-bad"/>
<text x="40.0" y="60.0" text-anchor="middle" class="fig-small">dropped</text>
<text x="300.0" y="276.0" text-anchor="middle" class="fig-small">Tokens t1 to t8 arrive in order. Six want expert 0, two want expert 1.</text>
<text x="300.0" y="296.0" text-anchor="middle" class="fig-small">Four seats fill, two tokens turn away, and ten seats elsewhere stay empty.</text>
</svg>
Figure 4. Overflow and waste at the same time. Expert 0 has more tokens than seats, so t5 and t6 are dropped, while experts 2 and 3 have no tokens at all. Tokens arrive in row order, so the early ones get the seats.
```

If routing were even, each expert would receive $n k / E$ slots. A capacity factor $\gamma$ (gamma) buys extra seats to absorb lumpiness:

$$
\text{capacity} = \left\lceil \gamma \, \frac{n k}{E} \right\rceil
$$

Read it aloud: the even share of slots per expert, times $\gamma$, rounded up. For Figure 4, the even share is $8 \times 1 / 4 = 2$, so $\gamma = 2$ gives 4 seats. With $n = 6$, $k = 2$, $E = 3$ the even share is 4, so $\gamma = 1$ gives 4 seats and $\gamma = 0.5$ gives 2. The capacity never falls below 1, so every expert can take at least one slot. That floor is on the capacity, not on $\gamma$: a factor below 1 is allowed, and it drops slots even when routing is even. With no limit, the capacity is $n k$. The seat rule is first come, first served, walking tokens in row order and each token's choices in order:

```python
if seats_used[e] < capacity:
    seats_used[e] += 1
    out[t] += g[t, slot] * (x[t] @ experts[e])
else:
    n_dropped += 1
```

The rule is cheap but biased against late tokens. The exercise uses it so the result is checkable.

Dropping removes a slot, not a token. With $k = 2$, a token dropped by one expert still gets the other, with its gate weight not rescaled, so with gates 0.73 and 0.27 and the second expert full, it receives only $0.73 \times \text{expert}_0(x)$. A token dropped from both gets a row of zeros, but a block computes $x + \text{MoE}(x)$, so its sum is $x$ itself. It loses an update, not its identity.

Extra seats cost memory in every expert, busy or not. In one lumpy routing of 60 tokens with $k = 2$ over eight experts, $\gamma = 1$ drops 27 of 120 slots, and drops reach zero only at a $\gamma$ of about 1.65. A smaller factor saves memory but silently skips work.

> [!KEY]
> Each expert has $\lceil \gamma n k / E \rceil$ seats per batch. Overflow slots are dropped, first come first served, and because the block computes $x + \text{MoE}(x)$, a token that loses all its slots passes through unchanged. More seats cost memory in every expert.

> [!KEY]
> The nurse's three questions have answers. Softmax, top-$k$, and renormalising pick and weight the experts. The gate weights carry the router's gradient, because the choice is flat. A balancing loss and a seat limit keep the traffic even.

> [!RECAP]
> Keep these points in front of you while you code:
>
> - Softmax with the row maximum subtracted, keep the top $k$, and divide the kept probabilities by their sum.
> - The logit gradient is $g_j (c_j - \bar{c})$, it sums to zero, and the router gradient is `x.T @ dz`.
> - The loss is $E \sum_i f_i P_i$, with $f_i$ the slot count over $n k$ and $P_i$ the mean probability.
> - Capacity is `max(1, ceil(gamma * n * k / E))`, or `n * k` with no limit.
> - Walk tokens and slots in row order. A full expert adds nothing, and the surviving gate weight is not rescaled.

## Build it step by step

The checks call your functions in this order:

1. `router(x, W_gate, k)` takes `x` of shape `(n, d)` and returns `topk_indices` `(n, k)`, `gate_weights` `(n, k)`, and `full_probs` `(n, E)`. Use `np.argsort(-logits, axis=-1)[:, :k]` for the indices. The usual mistake is dividing by the sum of all `E` probabilities, which returns the probabilities instead of gate weights.

```python
def router(x, W_gate, k):
    # logits, softmax with the max subtracted, top-k, then renormalise
    return topk_indices, gate_weights, full_probs
```

2. `gate_logit_grad(g, c)` takes two `(k,)` arrays and returns `g * (c - c_bar)`. The usual mistake is to subtract the plain mean of `c` instead of the gate-weighted average.

3. `router_backward(x, W_gate, k, expert_outputs, grad_output)` returns an array of shape `(d, E)`. `expert_outputs` has shape `(n, E, d_out)` and `grad_output` has shape `(n, d_out)`. For each token, build `c` for its chosen experts, call `gate_logit_grad`, write the result into `dz` at those columns only, and finish with `x.T @ dz`.

4. `expert_load(expert_indices, n_experts)` takes an integer array of shape `(n, k)` and returns $f$ of shape `(n_experts,)`. Use `np.bincount(..., minlength=n_experts)` and divide by `expert_indices.size`. The usual mistake is dividing by the number of tokens.

5. `load_balancing_loss(expert_indices, gate_probs, n_experts)` returns the float $E \sum f_i P_i$ from `expert_load` and the mean of `gate_probs` over tokens. The usual mistake is a sum over tokens instead of over experts.

6. `expert_capacity(n_tokens, k, n_experts, capacity_factor)` returns an integer: `n_tokens * k` for `None`, otherwise round up and keep at least 1. The usual mistake is reading "at least 1" as a rule about the factor.

7. `moe_forward(x, W_gate, experts, k, capacity_factor=None)` returns `(output, n_dropped)` with `output` of shape `(n, d_out)`. Call `expert_capacity` and `router` once, then loop over tokens and slots. The usual mistakes are adding an expert's output without its gate weight and letting later rows board first.

## Where this breaks

With $k = 1$ the renormalised gate is always exactly 1, so $c - \bar{c}$ is zero and this chapter's gradient gives the router nothing to learn from. That is a design choice: routers that send each token to one expert, such as the Switch-style ones, keep the raw softmax probability as the gate, which stays below one and carries a signal. The gradient also says nothing about a choice you did not make.

A balancing weight that is too large forces even routing and can stop experts from specialising, since specialising means unequal traffic. One that is too small leaves the loop running. Narrow fine-tuning data can bring collapse back. And the stored count never shrinks: every expert must sit in memory even if a batch uses only a few.

## The exercise

You write the seven functions above. The checks confirm that the true top-$k$ experts come back best first, that the worked weights are 0.7311 and 0.2689, and that huge logits do not overflow. They confirm the hand gradients $-0.3932$ and $+0.3932$ and that `router_backward` matches finite differences. They also confirm that `expert_load` divides by $n k$, that the loss gives 1.5 on the worked example, 1 for uniform routing, and $E$ or $E/k$ under collapse, and that `expert_capacity` rounds up with a floor of 1. Last, `moe_forward` reproduces the hand mix 5.614, equals the dense sum when $k = E$, drops exactly the overflow, and never drops more as $\gamma$ grows.
