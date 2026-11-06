---
title: Backpropagation through attention
slug: 086-attention-gradients
part: "Part 5 — Vision and sequences"
summary: Run attention backward one local rule at a time, and see why the softmax couples the weights and why copies in the forward pass become sums.
minutes: 45
gpu: false
objectives:
  - Walk the slope of the loss backward through the blend, the softmax, the dot with v, the tanh, and the two projections.
  - Explain why the softmax's slopes always sum to zero, and why a value copied into every row collects the sum of the row slopes.
  - Say why saturated tanh units pass little slope backward, using the worked numbers.
  - Build the three backward functions in numpy and check each against finite differences.
prereqs:
  - 054-backprop-the-graph-and-reverse-order
  - 055-backprop-array-operations
  - 056-softmax-cross-entropy-gradient
  - 057-backprop-mlp-and-gradient-check
  - 085-additive-attention
exercise: 086-attention-gradients
---

# Backpropagation through attention

During training the decoder writes a word, a loss says it was wrong, and the loss hands back one piece of information: how the loss would change if the context vector moved a little. That is a slope on the context. Attention has many numbers behind that context, namely $W_1$, $W_2$, $v$, the query, the keys, and the values, and each one needs its own slope. Without them, you could only learn by nudging every number and rerunning the whole forward pass, once per number.

Backpropagation avoids the reruns by walking the forward pass backward. Each step of the forward pass is small and has a local rule: if my input moves a little, my output moves by this much. Walking backward means taking the slope that arrives at a step's output, applying that step's rule, and passing the result to the step's input. Do this from the context all the way back, and every number gets its slope in one sweep. "Walking slopes backwards through a calculation" explains the idea in general. This chapter applies it to the additive attention you built, step by step, with real numbers.

Two ideas make attention's backward pass different from a plain stack of layers. The softmax couples the weights, because they share a budget of 1. And the query is copied into every row in the forward pass, so its slope is a sum in the backward pass. Figure 1 shows the whole path.

```figure
<svg viewBox="0 0 640 240" role="img" aria-label="A forward chain from the inputs through u, tanh, scores, softmax, and the blend to the context, with orange arrows running backward and labelled with the local rule and the worked numbers">
<defs><marker id="c095-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c095-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="10.0" y="56.0" width="88.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="54.0" y="76.0" text-anchor="middle" class="fig-bold">u</text>
<text x="54.0" y="94.0" text-anchor="middle" class="fig-small">W1 q + W2 k</text>
<rect x="132.0" y="56.0" width="88.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="176.0" y="76.0" text-anchor="middle" class="fig-bold">t</text>
<text x="176.0" y="94.0" text-anchor="middle" class="fig-small">tanh(u)</text>
<rect x="254.0" y="56.0" width="88.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="298.0" y="76.0" text-anchor="middle" class="fig-bold">scores</text>
<text x="298.0" y="94.0" text-anchor="middle" class="fig-small">t · v</text>
<rect x="376.0" y="56.0" width="88.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="420.0" y="76.0" text-anchor="middle" class="fig-bold">weights</text>
<text x="420.0" y="94.0" text-anchor="middle" class="fig-small">softmax</text>
<rect x="498.0" y="56.0" width="88.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="542.0" y="76.0" text-anchor="middle" class="fig-bold">context</text>
<text x="542.0" y="94.0" text-anchor="middle" class="fig-small">weights · values</text>
<line x1="98.0" y1="81.0" x2="131.0" y2="81.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<line x1="220.0" y1="81.0" x2="253.0" y2="81.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<line x1="342.0" y1="81.0" x2="375.0" y2="81.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<line x1="464.0" y1="81.0" x2="497.0" y2="81.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<rect x="268.0" y="2.0" width="60.0" height="26.0" rx="6" class="fill-soft-d stroke-d"/>
<text x="298.0" y="20.0" text-anchor="middle">v</text>
<line x1="288.0" y1="28.0" x2="288.0" y2="54.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<path d="M308.0,55 L308.0,30" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="315.0" y="44.0" class="fig-small fig-bold">dv</text>
<rect x="512.0" y="2.0" width="60.0" height="26.0" rx="6" class="fill-soft-d stroke-d"/>
<text x="542.0" y="20.0" text-anchor="middle">values</text>
<line x1="532.0" y1="28.0" x2="532.0" y2="54.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<path d="M552.0,55 L552.0,30" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="559.0" y="44.0" class="fig-small fig-bold">dvalues</text>
<path d="M520.0,108 L520.0,128 L442.0,128 L442.0,110" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="481.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">dweights</text>
<text x="481.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">[10, 0]</text>
<path d="M398.0,108 L398.0,128 L320.0,128 L320.0,110" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="359.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">dscores</text>
<text x="359.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">[2.31, -2.31]</text>
<path d="M276.0,108 L276.0,128 L198.0,128 L198.0,110" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="237.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">dt</text>
<text x="237.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">outer(dscores, v)</text>
<path d="M154.0,108 L154.0,128 L76.0,128 L76.0,110" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="115.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">du</text>
<text x="115.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">dt * (1 - t²)</text>
<path d="M32.0,108 L32.0,168" fill="none" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<text x="12.0" y="186.0" class="fig-small fig-bold">dW1, dW2, dquery, dkeys</text>
<text x="12.0" y="215.0" class="fig-small">dW1, dquery: sum du over the rows.   dW2, dkeys: one row of du per key.</text>
</svg>
Figure 1. The forward pass runs left to right (black). The backward pass runs right to left (orange), and each arrow applies one local rule to the slope it receives. Notice that the numbers on the arrows are the chapter's running example, and that the rules never reach back further than one step.
```

## What you already know

Five ideas carry this chapter:

- A slope of the loss with respect to a number says how much the loss moves when that number moves a little. The usual word for it is gradient, and this chapter says slope. A name that starts with `d`, such as `dweights`, holds the slope with respect to the thing that follows.
- In the backward pass, each step multiplies the incoming slope by its local rule, and a value used in several places collects the sum of the slopes from all of them (see "Walking slopes backwards through a calculation").
- The attention forward pass is `u = W1 @ query + keys @ W2.T`, then `t = tanh(u)`, `scores = t @ v`, `weights = softmax(scores)`, and `context = weights @ values` (see "Additive attention: let a small network score the match").
- The slope of tanh at $u$ is $1 - \tanh(u)^2$.
- A finite difference nudges one input by a tiny $\epsilon$ in each direction and divides the change in the loss by $2\epsilon$, which approximates the slope (see "Putting it together: gradients for a whole network").

## Pick a loss, then walk backward

To check slopes, you need a loss. Let the loss be $L = g \cdot \text{context}$ for a fixed vector $g$. Then the slope of $L$ with respect to the context is exactly $g$, and the exercise names it `dcontext`. In a real model, `dcontext` would come from the layers after attention, and here you supply it yourself. This loss also lets you check every other slope with finite differences.

Use the running example from the last two chapters: query $[1, 0]$, keys $[1, 0]$ and $[0, 1]$, values $[10, 0]$ and $[0, 10]$, identity $W_1$ and $W_2$, and $v = [1, 1]$. The forward pass gave scores $0.9640$ and $1.5232$, and weights $0.3637$ and $0.6363$. Take `dcontext` to be $[1, 0]$, which says that only the first entry of the context affects the loss.

### Step 1: the blend

The context is $\sum_i a_i V_i$, a weighted sum of value rows. Each value row is scaled by its weight, so the slope on row $i$ is its weight times `dcontext`, which gives rows $[0.3637, 0]$ and $[0.6363, 0]$. Each weight multiplies a whole value row, so the slope on weight $a_i$ is how much the loss moves per unit of weight, which is $V_i \cdot \text{dcontext}$. The values are $[10, 0]$ and $[0, 10]$, so the weight slopes are $10$ and $0$. Raising weight 1 by 0.1 adds $0.1 \cdot [10, 0]$ to the context, so the loss rises by 1.

### Step 2: the softmax

The weights are not independent, because they sum to 1. Raising the score of position 1 raises its weight and, to keep the total at 1, lowers the other weights. So a weight's slope cannot be passed back on its own. To see what the rule must be, nudge score $j$ up by a small $\epsilon$ and watch every weight. Weight $j$ rises by $a_j(1 - a_j)\epsilon$, and each other weight $a_i$ falls by $a_i a_j \epsilon$, so that the total stays 1. On the example, nudging score 1 raises weight 1 by $0.3637 \cdot 0.6363 = 0.2314$ per unit and lowers weight 2 by the same amount. The loss then moves by $10 \cdot 0.2314 + 0 \cdot (-0.2314) = 2.314$ per unit. Adding up these effects for each score, the slope on a score compares its weight's slope with the average slope, weighted by the weights:

$$
\text{dscores}_i = a_i \left( \text{dweights}_i - \sum_j a_j \, \text{dweights}_j \right)
$$

Read it aloud: take the slope on weight $i$, subtract the weighted average of all the weight slopes, and multiply by the weight. A weight whose slope is above the average should have its score raised, and one below the average should have it lowered. The factor $a_i$ shrinks the answer for a weight near 0 or near 1, which barely moves when its score does.

On the example, the average is $0.3637 \cdot 10 + 0.6363 \cdot 0 = 3.637$. The differences are $6.363$ and $-3.637$, and multiplying by the weights gives $2.314$ and $-2.314$. Figure 2 draws it. These two numbers sum to zero, which is a sanity check you can apply to any softmax backward pass. Adding the same number to every score leaves the weights unchanged, so the loss cannot depend on such a shift, and the slopes must cancel.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three bar panels: the slopes on the weights, 10 and 0, then the same minus their weighted average, then multiplied by the weights, giving 2.314 and minus 2.314">
<defs><marker id="c095-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="24.0" class="fig-bold">1. slope on each weight</text>
<line x1="20.0" y1="150.0" x2="190.0" y2="150.0" class="stroke-muted"/>
<rect x="44.0" y="50.0" width="44.0" height="100.0" class="fill-a"/>
<text x="66.0" y="44.0" text-anchor="middle" class="fig-small fig-mono">10.000</text>
<text x="66.0" y="238.0" text-anchor="middle" class="fig-small">key 1</text>
<text x="142.0" y="144.0" text-anchor="middle" class="fig-small fig-mono">0.000</text>
<text x="142.0" y="238.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="215.0" y="24.0" class="fig-bold">2. minus the average</text>
<line x1="215.0" y1="150.0" x2="385.0" y2="150.0" class="stroke-muted"/>
<rect x="239.0" y="86.4" width="44.0" height="63.6" class="fill-a"/>
<text x="261.0" y="80.4" text-anchor="middle" class="fig-small fig-mono">6.363</text>
<text x="261.0" y="238.0" text-anchor="middle" class="fig-small">key 1</text>
<rect x="315.0" y="150.0" width="44.0" height="36.4" class="fill-soft-a stroke-a"/>
<text x="337.0" y="201.4" text-anchor="middle" class="fig-small fig-mono">-3.637</text>
<text x="337.0" y="238.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="410.0" y="24.0" class="fig-bold">3. times the weight</text>
<line x1="410.0" y1="150.0" x2="580.0" y2="150.0" class="stroke-muted"/>
<rect x="434.0" y="126.9" width="44.0" height="23.1" class="fill-a"/>
<text x="456.0" y="120.9" text-anchor="middle" class="fig-small fig-mono">2.314</text>
<text x="456.0" y="238.0" text-anchor="middle" class="fig-small">key 1</text>
<rect x="510.0" y="150.0" width="44.0" height="23.1" class="fill-soft-a stroke-a"/>
<text x="532.0" y="188.1" text-anchor="middle" class="fig-small fig-mono">-2.314</text>
<text x="532.0" y="238.0" text-anchor="middle" class="fig-small">key 2</text>
<line x1="20.0" y1="113.6" x2="190.0" y2="113.6" class="stroke-b dashed"/>
<text x="96.0" y="91.6" class="fig-small">weighted average</text>
<text x="96.0" y="106.6" class="fig-small fig-mono">3.637</text>
<text x="500.0" y="262.0" text-anchor="middle" class="fig-bold fig-mono">sum = 0</text>
<line x1="194.0" y1="100.0" x2="211.0" y2="100.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<line x1="389.0" y1="100.0" x2="406.0" y2="100.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
</svg>
Figure 2. The softmax step compares each weight's slope with the average slope, then scales by the weight. Notice that the two answers in panel 3 are equal and opposite, because the weights share a budget of 1.
```

### Step 3: the dot with $v$, and the tanh

Each score is $t_i \cdot v$, a dot product of row $i$ of $t = \tanh(u)$ with $v$. So row $i$ of the slope on $t$ is $\text{dscores}_i \cdot v$. The vector $v$ is used for every key, so its slope collects a share from each: $\text{d}v = \sum_i \text{dscores}_i \, t_i$, which is `t.T @ dscores`. Here it is $[0.4685, -1.7626]$.

The tanh step multiplies by its slope, $1 - t^2$:

$$
\text{du} = \text{dt} \odot (1 - t^2)
$$

The symbol $\odot$ means entry-by-entry multiplication. In the example, $t$ is $[0.9640, 0]$ for key 1 and $[0.7616, 0.7616]$ for key 2. The slope $1 - t^2$ is $[0.0707, 1]$ for key 1 and $[0.42, 0.42]$ for key 2. The slope arriving at key 1 is $[2.314, 2.314]$, and after the tanh step it is $[0.1635, 2.3143]$. The saturated unit, where $u = 2$, passes only 7% of its slope, and the unit at $u = 0$ passes all of it. This is the cost of saturation: a unit sitting on a flat part of tanh barely learns.

### Step 4: the projections and the sum

The matrix $u$ has a row for each key, and row $i$ is $W_1 q + W_2 k_i$. The vector $W_1 q$ was the same in every row, so in the forward pass it was copied $T$ times. A copy in the forward pass becomes a sum in the backward pass, because the query influenced the loss through every row. So you add the row slopes first, then go back through $W_1$ (Figure 3):

$$
\text{dW}_1 = \Big(\sum_i \text{du}_i\Big) q^{\top}, \qquad
\text{dquery} = W_1^{\top} \sum_i \text{du}_i
$$

For the example the sum is $[0.1635, 2.3143] + [-0.972, -0.972] = [-0.8085, 1.3424]$, which is `dquery` because $W_1$ is the identity. The same rule explains the other gradients. The matrices $W_1$, $W_2$, and $v$ are all shared by every key, so each of their slopes is a sum over keys. Each key row is used once, so it keeps its own row:

$$
\text{dW}_2 = \text{du}^{\top} \, \text{keys}, \qquad \text{dkeys} = \text{du} \, W_2
$$

```figure
<svg viewBox="0 0 600 296" role="img" aria-label="Top: the query projection is copied into two rows of u. Bottom: the two rows of du are added into one vector">
<defs><marker id="c095-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c095-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="20.0" class="fig-bold">Forward: one query, copied into every row</text>
<rect x="14.0" y="38.0" width="120.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="74.0" y="58.0" text-anchor="middle" class="fig-bold">W1 q</text>
<text x="74.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">[1, 0]</text>
<rect x="250.0" y="30.0" width="150.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="325.0" y="48.0" text-anchor="middle" class="fig-small">row 1 of u</text>
<text x="325.0" y="66.0" text-anchor="middle" class="fig-small fig-mono">[2, 0]</text>
<line x1="134.0" y1="62.0" x2="248.0" y2="52.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<rect x="250.0" y="90.0" width="150.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="325.0" y="108.0" text-anchor="middle" class="fig-small">row 2 of u</text>
<text x="325.0" y="126.0" text-anchor="middle" class="fig-small fig-mono">[1, 1]</text>
<line x1="134.0" y1="62.0" x2="248.0" y2="112.0" class="stroke-ink" marker-end="url(#c095-arrow-ink)"/>
<text x="420.0" y="78.0" class="fig-small">plus its own W2 key</text>
<line x1="14.0" y1="150.0" x2="586.0" y2="150.0" class="stroke-grid"/>
<text x="14.0" y="176.0" class="fig-bold">Backward: the row slopes are added into one</text>
<rect x="250.0" y="190.0" width="150.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="325.0" y="208.0" text-anchor="middle" class="fig-small">row 1 of du</text>
<text x="325.0" y="226.0" text-anchor="middle" class="fig-small fig-mono">[0.1635, 2.3143]</text>
<line x1="248.0" y1="212.0" x2="136.0" y2="232.0" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<rect x="250.0" y="246.0" width="150.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="325.0" y="264.0" text-anchor="middle" class="fig-small">row 2 of du</text>
<text x="325.0" y="282.0" text-anchor="middle" class="fig-small fig-mono">[-0.972, -0.972]</text>
<line x1="248.0" y1="268.0" x2="136.0" y2="238.0" class="stroke-b" marker-end="url(#c095-arrow-b)"/>
<rect x="14.0" y="214.0" width="120.0" height="48.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="74.0" y="234.0" text-anchor="middle" class="fig-small">sum of rows</text>
<text x="74.0" y="252.0" text-anchor="middle" class="fig-small fig-mono">[-0.8085, 1.3424]</text>
<text x="420.0" y="232.0" class="fig-small">this one vector is dW1's factor</text>
<text x="420.0" y="250.0" class="fig-small">(through W1 it gives dquery)</text>
</svg>
Figure 3. Copying a value into every row in the forward pass (top) means adding the row slopes in the backward pass (bottom). Notice that the sum is one vector, not two: the query is one number set used many times.
```

> [!WARNING]
> Sum `du` over axis 0 before calling `np.outer`. A `(T, h)` array passed to `np.outer` is flattened into one long vector, and the gradient comes out with the wrong shape.

In numpy, the last three steps are:

```python
du = np.outer(dscores, v) * (1.0 - t ** 2)     # (T, h)
du_sum = du.sum(axis=0)                        # (h,): the copies collect here
dW1 = np.outer(du_sum, query)                  # (h, d_s)
```

> [!KEY]
> The backward pass is the forward pass in reverse, one local rule per step. The softmax compares each slope with the average, so its answers sum to zero. A tanh passes less slope where it is flat. A value copied in the forward pass collects the sum of the slopes in the backward pass. Together these send the loss's blame from the context back to every number that shaped it.

> [!RECAP]
> Before you build:
> - Blend: `dweights = values @ dcontext` and `dvalues = np.outer(weights, dcontext)`.
> - Softmax: `dscores = weights * (dweights - np.sum(weights * dweights))`, and the result sums to zero.
> - Dot with `v`: `dv = t.T @ dscores` and `dt = np.outer(dscores, v)`. Tanh: `du = dt * (1 - t**2)`.
> - Projections: `du.sum(axis=0)` goes to `dW1` and `dquery`, and `dW2 = du.T @ keys`, `dkeys = du @ W2`.
> - A finite difference confirms every one of these.

## Build it step by step

The starter holds `softmax`, `attend`, `additive_scores`, and `additive_attention` from the earlier chapters. The checks test these functions in order:

1. **`softmax_backward(weights, dweights)`.** Takes both of shape `(T,)` and returns `dscores` of shape `(T,)`. Use the formula in Step 2. The usual mistake is dropping the subtraction of the weighted average, which gives `weights * dweights`.
2. **`context_backward(dcontext, weights, values)`.** Returns `(dweights, dvalues)` with shapes `(T,)` and `(T, d_v)`. The usual mistake is swapping the two outputs.
3. **`attention_backward(dcontext, query, keys, values, W1, W2, v)`.** Reruns the forward pass, keeps `t`, calls your two functions, then applies the rules from Steps 3 and 4. It returns `(dquery, dkeys, dvalues, dW1, dW2, dv)`, each shaped like its input. The usual mistake is forgetting the factor $1 - t^2$, or forgetting to sum `du` before the outer product.

## Where this breaks

A saturated tanh passes almost no slope, as Step 3 showed, so a scorer whose hidden units sit on the flat parts learns slowly. A hand-written backward pass can also be wrong while still returning arrays of the right shape, and only a finite-difference check catches that. This is why the checks nudge every input by a tiny amount and compare.

## The exercise

You build `softmax_backward`, `context_backward`, and `attention_backward`. The checks confirm that the softmax slopes on the worked example are $2.314$ and $-2.314$, that they sum to zero, and that they match finite differences. They also check that `context_backward` gives weight slopes $[10, 0]$ on the example, and that all six gradients of `attention_backward` match finite differences. A final check passes a single key, where the weight is always 1, and requires every gradient except `dvalues` to be zero.
