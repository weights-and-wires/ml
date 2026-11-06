---
title: Why the gate weight lets the router learn
slug: 116-router-gradient
part: "Part 7 — Large language models"
summary: Choosing the top k experts has no slope, so the router learns only through the gate weights, with a gradient you can write in one line.
minutes: 40
gpu: false
objectives:
  - Explain why a top-k choice has zero slope almost everywhere, and why the gate weight is the router's only path to a gradient.
  - Define c, the loss slope per unit of an expert's weight, and say which sign of the gap from the weighted average means an expert should get more trust.
  - Compute the gradient on each gate logit by hand on two experts, and show that the gradients sum to zero.
  - Assemble the router's weight gradient with a matrix product, and confirm it against finite differences.
prereqs:
  - 115-top-k-routing
  - 056-softmax-cross-entropy-gradient
exercise: 116-router-gradient
---

# Why the gate weight lets the router learn

The nurse in the last chapter picked two specialists for a patient, and the patient did badly. Who is to blame, and what should the nurse change? Perhaps the nurse should have picked a different pair. But "pick a different pair" is not something you can nudge a little. You either send the patient to Dr. A or you do not.

A router has the same trouble. It chooses experts with an `argsort`, a hard yes-or-no. If you nudge one score by a tiny amount, the same two experts still win, the output does not change, and the loss does not change. Training works by following the slope of the loss, so a flat choice leaves the router with nothing to follow. It would stay at its random starting weights forever, and every token would be routed by chance.

This chapter shows where the slope comes from. It does not come from the choice. It comes from how much each chosen expert is trusted, and that part is smooth.

## What you already know

- A gradient is the slope of the loss with respect to one number: how much the loss changes per unit nudge.
- The chain rule multiplies slopes along a path, from the loss back to the weight you want to change.
- A softmax gives positive weights that sum to one, so raising one weight must lower the others.
- A matrix product `x.T @ dz` collects a gradient on outputs into a gradient on weights.
- The router in "Send each token to a few experts" keeps the top $k$ logits and renormalises their probabilities into gate weights.

## The staircase and the slope

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Left: which expert wins is a step function of logit 0. Right: the softmax weight on expert 0 is a smooth curve">
<defs><marker id="c130-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
Figure 1. Left: as the logit of expert 0 rises, the winner flips once and stays flat on both sides, so the slope is zero almost everywhere. Right: the weight on expert 0 rises smoothly over the same range, so a nudge always changes something.
```

Take two experts, with the logit of expert 1 held at 0, and slide the logit of expert 0 from $-2$ to $2$. The left panel of Figure 1 plots which expert wins. For every logit below 0, expert 1 wins. For every logit above 0, expert 0 wins. The curve is a staircase with one step at the tie. Nudge the logit from $-1$ to $-0.6$ and the winner does not change, so the output does not change, so the slope is zero. At the single point of the tie the curve jumps, and a jump has no usable slope either.

The right panel plots something else: the weight that expert 0 gets when the two experts are mixed, $g_0 = \text{softmax}$ of the two logits. It rises smoothly over the same range. The same nudge, $-1$ to $-0.6$, moves it from 0.27 to 0.35. Here a nudge changes the output, so there is a slope to follow.

Why not make the choice smooth too, and send every token to every expert with its softmax weight? You can, but it is the dense layer from the last chapter, with the savings gone. The fix keeps the hard choice and uses the smooth weight on top of it. The router never learns which experts to pick by being told "that choice was wrong". It learns by turning the trust it gives each chosen expert up or down, and the logits that decide the choice are the same logits that set the trust, so tuning the trust also moves the choice over many steps.

## The loss slope per unit of an expert

The layer's output is $\hat{y} = \sum_i g_i \, e_i$, where $e_i$ is expert $i$'s output vector for this token. Upstream, backpropagation hands the layer the vector $\bar{y}$, the slope of the loss with respect to $\hat{y}$. The slope of the loss with respect to a gate weight $g_i$ is then the dot product

$$
c_i = \bar{y} \cdot e_i
$$

Read aloud: $c_i$ is how much the loss moves per unit of weight put on expert $i$. It is large and positive when expert $i$'s output points the same way as the loss gradient, which is the direction that makes the loss bigger. More weight on that expert makes things worse. When $c_i$ is small or negative, the expert's output points away from the loss gradient, and more weight on it helps.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three split bars show gate weights 0.73 and 0.27, then 0.63 and 0.37, then 0.83 and 0.17, with mixed scores 1.54, 1.74, and 1.34">
<defs><marker id="c130b-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="150.0" y="22.0" text-anchor="middle" class="fig-bold">gate weights</text>
<text x="500.0" y="22.0" text-anchor="middle" class="fig-bold">mixed score g0·1 + g1·3</text>
<text x="20.0" y="42.0" text-anchor="start" class="fig-small">start</text>
<rect x="20.0" y="50.0" width="219.3" height="32.0" class="fill-soft-a stroke-a"/>
<rect x="239.3" y="50.0" width="80.7" height="32.0" class="fill-soft-b stroke-b"/>
<text x="129.7" y="71.0" text-anchor="middle" class="fig-bold">0.73</text>
<text x="279.7" y="71.0" text-anchor="middle" class="fig-bold">0.27</text>
<text x="500.0" y="72.0" text-anchor="middle" class="fig-bold">1.54</text>
<text x="20.0" y="112.0" text-anchor="start" class="fig-small">move 0.1 toward expert 1</text>
<rect x="20.0" y="120.0" width="189.3" height="32.0" class="fill-soft-a stroke-a"/>
<rect x="209.3" y="120.0" width="110.7" height="32.0" class="fill-soft-b stroke-b"/>
<text x="114.7" y="141.0" text-anchor="middle" class="fig-bold">0.63</text>
<text x="264.7" y="141.0" text-anchor="middle" class="fig-bold">0.37</text>
<text x="500.0" y="142.0" text-anchor="middle" class="fig-bold">1.74</text>
<text x="20.0" y="182.0" text-anchor="start" class="fig-small">move 0.1 toward expert 0</text>
<rect x="20.0" y="190.0" width="249.3" height="32.0" class="fill-soft-a stroke-a"/>
<rect x="269.3" y="190.0" width="50.7" height="32.0" class="fill-soft-b stroke-b"/>
<text x="144.7" y="211.0" text-anchor="middle" class="fig-bold">0.83</text>
<text x="294.7" y="211.0" text-anchor="middle" class="fig-bold">0.17</text>
<text x="500.0" y="212.0" text-anchor="middle" class="fig-bold">1.34</text>
<text x="20.0" y="262.0" text-anchor="start" class="fig-small">Blue: expert 0, c = 1. Orange: expert 1, c = 3. A larger mixed score means a larger loss.</text>
<line x1="335.0" y1="66.0" x2="335.0" y2="66.0" class="stroke-muted"/>
</svg>
Figure 2. Moving weight toward expert 1, whose c is larger, raises the mixed score from 1.54 to 1.74. Moving weight toward expert 0 lowers it to 1.34. The router should therefore move weight away from the expert with the larger c.
```

Figure 2 works an example with two experts, $c_0 = 1$ and $c_1 = 3$, and gate weights 0.73 and 0.27. (To make that concrete, suppose the output has one number, $\bar{y} = 1$, and the experts return 1 and 3.) The mixed score $g_0 c_0 + g_1 c_1$ is $0.73 \times 1 + 0.27 \times 3 = 1.54$, and it measures how much the loss moves because of this mix. Move 0.1 of the weight toward expert 1, which has the larger $c$, and the score rises to 1.74. Move 0.1 toward expert 0 and it falls to 1.34. So the router should move weight away from the expert with the larger $c$.

## From weights back to logits

The router does not set gate weights directly. It sets logits $z$, and a softmax turns them into weights. Because the weights sum to one, they cannot move independently: raising one lowers the others. Work out what that does to the gradient. The slope of weight $g_i$ with respect to logit $z_j$ is $g_i(1 - g_i)$ when $i = j$ and $-g_i g_j$ otherwise, which is the softmax rule from "Softmax and cross-entropy: a loss with a gradient you can read". Apply the chain rule, adding up the effect of logit $z_j$ on every weight:

$$
\frac{\partial L}{\partial z_j} = g_j \left( c_j - \bar{c} \right), \qquad \bar{c} = \sum_{i} g_i \, c_i
$$

Here is the algebra. A nudge to $z_j$ moves its own weight by $g_j(1-g_j)$ and every other weight $g_i$ by $-g_i g_j$, and each weight's change is multiplied by its $c$. So the total is $c_j g_j (1 - g_j) - \sum_{i \neq j} c_i g_i g_j$. Pull $g_j$ out and write $-g_j \sum_{i} c_i g_i$ for what is left to get $g_j (c_j - \bar{c})$.

Read aloud: the gradient on a logit is its gate weight times the gap between its $c$ and the weighted average $\bar{c}$ of all the chosen experts' $c$ values. Two things in it make sense. First, only the gap matters, because the weights compete: a logit that rises takes weight from every other expert, so what counts is how this expert compares with the average it is taking weight from. Second, the sum over experts is the same softmax over the $k$ kept logits that the last chapter showed gives the renormalised gates, so the formula uses exactly the gate weights you computed.

Use the worked example. The gates are 0.7311 and 0.2689 and $c = [1, 3]$. The average is $\bar{c} = 0.7311 \times 1 + 0.2689 \times 3 = 1.5378$. Then

- expert 0: $0.7311 \times (1 - 1.5378) = -0.3932$
- expert 1: $0.2689 \times (3 - 1.5378) = +0.3932$

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Left: c values 1 and 3 with the weighted average 1.54 as a dashed line. Right: the two logit gradients, -0.39 and +0.39">
<defs><marker id="c130c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="150.0" y="24.0" text-anchor="middle" class="fig-bold">c: how much loss each expert adds</text>
<text x="450.0" y="24.0" text-anchor="middle" class="fig-bold">gradient on each logit</text>
<line x1="40.0" y1="230.0" x2="260.0" y2="230.0" class="stroke-muted"/>
<rect x="80.0" y="180.0" width="50.0" height="50.0" class="fill-a"/>
<text x="105.0" y="174.0" text-anchor="middle" class="fig-small">1</text>
<text x="105.0" y="246.0" text-anchor="middle" class="fig-small">expert 0</text>
<rect x="170.0" y="80.0" width="50.0" height="150.0" class="fill-a"/>
<text x="195.0" y="74.0" text-anchor="middle" class="fig-small">3</text>
<text x="195.0" y="246.0" text-anchor="middle" class="fig-small">expert 1</text>
<line x1="40.0" y1="153.1" x2="260.0" y2="153.1" class="stroke-b dashed"/>
<text x="262.0" y="157.1" text-anchor="start" class="fig-small">c̄ = 1.54</text>
<line x1="340.0" y1="140.0" x2="560.0" y2="140.0" class="stroke-muted"/>
<rect x="380.0" y="140.0" width="50.0" height="78.6" class="fill-b"/>
<text x="405.0" y="234.6" text-anchor="middle" class="fig-small">-0.39</text>
<text x="405.0" y="264.0" text-anchor="middle" class="fig-small">expert 0</text>
<rect x="470.0" y="61.4" width="50.0" height="78.6" class="fill-b"/>
<text x="495.0" y="55.4" text-anchor="middle" class="fig-small">+0.39</text>
<text x="495.0" y="264.0" text-anchor="middle" class="fig-small">expert 1</text>
<text x="570.0" y="144.0" text-anchor="end" class="fig-small">0</text>
</svg>
Figure 3. Expert 0 adds less than the weighted average (1 against 1.54), so its logit gradient is negative and a downhill step raises its logit. Expert 1 adds more, so its logit falls. The two bars are mirror images and sum to zero.
```

The two numbers sum to zero, as Figure 3 shows, because a gap that is below average for one expert is above average for the other. Read the signs with a downhill step in mind: training subtracts the gradient. The gradient on expert 0 is negative, so subtracting it raises expert 0's logit. Expert 0 adds less loss than the average, so the router trusts it more. Expert 1 adds more loss than average and its logit falls.

An expert the token did not choose is not in the sum at all, since its weight is 0 whatever its logit does. Its logit gets a gradient of zero from this token. In code:

```python
c_bar = np.sum(g * c)
dz_chosen = g * (c - c_bar)
```

> [!WARNING]
> A common slip is to write the gradient into every expert's column. Only the $k$ chosen experts get a nonzero value. The others stay at zero.

## From logits to router weights

The logits come from $z = x \, W_{\text{gate}}$, so the gradient on the router matrix follows the matrix rule you used for every linear layer. Stack each token's logit gradient as a row of an array `dz` of shape `(n, E)`, with zeros at the unchosen experts. Then the gradient on $W_{\text{gate}}$, of shape `(d, E)`, is

$$
\frac{\partial L}{\partial W_{\text{gate}}} = x^{\top} \, dz
$$

Read aloud: each router weight collects the logit gradients of every token, scaled by that token's feature. Column $j$ of the result moves the router's score for expert $j$ toward tokens that deserve more trust in it.

```python
dz = np.zeros((n, E))
dz[t, chosen] = g * (c - c_bar)   # for each token t
grad_W = x.T @ dz
```

## Trust the formula only after you check it

A hand-derived gradient is easy to get wrong in a sign or an index. The check is finite differences: nudge one weight up and down by a tiny $\epsilon$, rerun the forward computation, and divide the change in the loss by $2\epsilon$. If the formula is right, the two numbers agree to several digits. To use it you need a scalar to differentiate. Define the surrogate loss as the sum, over tokens and chosen experts, of $g \cdot c$. It holds the expert outputs and $\bar{y}$ fixed, and so measures only the router's path. Its gradient with respect to $W_{\text{gate}}$ is exactly what you are building, and you will check it that way.

> [!KEY]
> The top-$k$ choice is a staircase and has no slope. The router learns through the gate weights: each chosen logit gets its weight times how far that expert's loss slope sits from the weighted average, and the gradients sum to zero. That answers the opening question. Nudging a logit never changes who wins, so the router learns by changing how much it trusts the winners.

> [!RECAP]
> Keep these points in front of you while you code:
>
> - $c_i = \bar{y} \cdot e_i$ is the loss slope per unit of weight on expert $i$.
> - The logit gradient for a chosen expert is $g_j (c_j - \bar{c})$ with $\bar{c} = \sum_i g_i c_i$.
> - The gradients of one token sum to zero, and unchosen experts get zero.
> - Collect them in `dz` of shape `(n, E)`, and the router gradient is `x.T @ dz`.
> - With $k = 1$ the gate is always 1, so every gradient is zero.

## Build it step by step

The router from the last chapter is already written for you. The checks call your functions in this order:

1. `gate_logit_grad(g, c)` takes two arrays of shape `(k,)` and returns the array `g * (c - c_bar)`. The usual mistake is to forget the weights in `c_bar` and subtract the plain mean of `c`.

```python
def gate_logit_grad(g, c):
    # c_bar is the gate-weighted average, not the plain mean
    return g * (c - c_bar)
```

2. `surrogate_loss(x, W_gate, k, expert_outputs, grad_output)` returns one float. `expert_outputs` has shape `(n, E, d_out)` and `grad_output` has shape `(n, d_out)`. Call `router`, then add `gate_weights[t, slot] * (grad_output[t] @ expert_outputs[t, expert])` for every token and slot. The usual mistake is to include experts that were not chosen.

```python
def surrogate_loss(x, W_gate, k, expert_outputs, grad_output):
    # sum over tokens and chosen experts of g * c
    return total
```

3. `router_backward(x, W_gate, k, expert_outputs, grad_output)` returns an array of shape `(d, E)`. For each token, build `c` for its chosen experts, call `gate_logit_grad`, and write the result into `dz` at those columns only. Finish with `x.T @ dz`.

```python
def router_backward(x, W_gate, k, expert_outputs, grad_output):
    # dz[t, chosen] = gate_logit_grad(g, c), then x.T @ dz
    return grad_W
```

## Where this breaks

With $k = 1$ the renormalised gate is always exactly 1. Then $c - \bar{c}$ is zero for the single expert, so this formula gives the router no gradient and it never learns. This is a design choice of the renormalised gate, not a law. Routers that send each token to one expert, such as the Switch-style ones, keep the raw softmax probability as the gate, which stays below one and so carries a signal. This chapter's gradient needs $k \geq 2$.

The gradient also says nothing about a choice you did not make. If a token would have been better served by an expert outside its top $k$, the formula has no term for it. That expert's logit rises only when it grows into the top $k$ through changes in other tokens' training, or through a separate pressure such as the balancing loss in the next chapter.

## The exercise

You write `gate_logit_grad(g, c)`, `surrogate_loss(x, W_gate, k, expert_outputs, grad_output)`, and `router_backward(x, W_gate, k, expert_outputs, grad_output)`. The checks confirm the hand example $-0.3932$ and $+0.3932$, that the gradients of one token sum to zero and vanish when every $c$ is equal or $k = 1$, and that `gate_logit_grad` agrees with finite differences through a softmax. They also confirm `surrogate_loss` reproduces the hand mix and ignores unchosen experts, and that `router_backward` matches finite differences of the surrogate loss.
