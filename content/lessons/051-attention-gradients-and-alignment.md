---
title: "Attention gradients and the alignment matrix"
slug: 051-attention-gradients-and-alignment
part: "Part 5 — Vision and sequences"
summary: Run additive attention backward one local rule at a time, then stack the weights into an alignment matrix and hand-set a copy decoder.
minutes: 70
gpu: false
objectives:
  - Walk the slope of the loss backward through the blend, the softmax, the dot with v, the tanh, and the two projections.
  - Explain why the softmax's slopes always sum to zero, and why a value copied into every row collects the sum of the row slopes.
  - Read an alignment matrix, and say what the identity matrix means for a copy task.
  - Explain why the hand-built copy decoder works, using scores of -10 and -20, and why flipping the sign of v breaks it.
  - Build the three backward functions, the alignment matrix, an order-blind summary, and the copy decoder in numpy.
prereqs:
  - 031-backpropagation
  - 032-softmax-and-whole-network-gradients
  - 050-attention-as-soft-lookup
exercise: 051-attention-gradients-and-alignment
---

# Attention gradients and the alignment matrix

During training the decoder writes a word, a loss says it was wrong, and the loss hands back one piece of information: how the loss would change if the context vector moved a little. That is a slope on the context. Attention has many numbers behind that context, namely $W_1$, $W_2$, $v$, the query, the keys, and the values, and each one needs its own slope. Without them, you could only learn by nudging every number and rerunning the whole forward pass, once per number.

Backpropagation avoids the reruns by walking the forward pass backward. Each step of the forward pass is small and has a local rule: if my input moves a little, my output moves by this much. Walking backward means taking the slope that arrives at a step's output, applying that step's rule, and passing the result to the step's input. This chapter applies the idea from [backpropagation](/lessons/031-backpropagation) to the additive attention you built, with real numbers. Two ideas make the backward pass different from a plain stack of layers. The softmax couples the weights, because they share a budget of 1. And the query is copied into every row in the forward pass, so its slope is a sum in the backward pass. Figure 1 shows the whole path.

```figure
<svg viewBox="0 0 640 240" role="img" aria-label="A forward chain from the inputs through u, tanh, scores, softmax, and the blend to the context, with orange arrows running backward and labelled with the local rule and the worked numbers">
<defs><marker id="c042-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c042-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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
<line x1="98.0" y1="81.0" x2="131.0" y2="81.0" class="stroke-ink" marker-end="url(#c042-arrow-ink)"/>
<line x1="220.0" y1="81.0" x2="253.0" y2="81.0" class="stroke-ink" marker-end="url(#c042-arrow-ink)"/>
<line x1="342.0" y1="81.0" x2="375.0" y2="81.0" class="stroke-ink" marker-end="url(#c042-arrow-ink)"/>
<line x1="464.0" y1="81.0" x2="497.0" y2="81.0" class="stroke-ink" marker-end="url(#c042-arrow-ink)"/>
<rect x="268.0" y="2.0" width="60.0" height="26.0" rx="6" class="fill-soft-d stroke-d"/>
<text x="298.0" y="20.0" text-anchor="middle">v</text>
<line x1="288.0" y1="28.0" x2="288.0" y2="54.0" class="stroke-ink" marker-end="url(#c042-arrow-ink)"/>
<path d="M308.0,55 L308.0,30" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="315.0" y="44.0" class="fig-small fig-bold">dv</text>
<rect x="512.0" y="2.0" width="60.0" height="26.0" rx="6" class="fill-soft-d stroke-d"/>
<text x="542.0" y="20.0" text-anchor="middle">values</text>
<line x1="532.0" y1="28.0" x2="532.0" y2="54.0" class="stroke-ink" marker-end="url(#c042-arrow-ink)"/>
<path d="M552.0,55 L552.0,30" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="559.0" y="44.0" class="fig-small fig-bold">dvalues</text>
<path d="M520.0,108 L520.0,128 L442.0,128 L442.0,110" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="481.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">dweights</text>
<text x="481.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">[10, 0]</text>
<path d="M398.0,108 L398.0,128 L320.0,128 L320.0,110" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="359.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">dscores</text>
<text x="359.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">[2.31, -2.31]</text>
<path d="M276.0,108 L276.0,128 L198.0,128 L198.0,110" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="237.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">dt</text>
<text x="237.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">outer(dscores, v)</text>
<path d="M154.0,108 L154.0,128 L76.0,128 L76.0,110" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="115.0" y="146.0" text-anchor="middle" class="fig-small fig-bold">du</text>
<text x="115.0" y="161.0" text-anchor="middle" class="fig-small fig-mono">dt * (1 - t²)</text>
<path d="M32.0,108 L32.0,168" fill="none" class="stroke-b" marker-end="url(#c042-arrow-b)"/>
<text x="12.0" y="186.0" class="fig-small fig-bold">dW1, dW2, dquery, dkeys</text>
<text x="12.0" y="215.0" class="fig-small">dW1, dquery: sum du over the rows.   dW2, dkeys: one row of du per key.</text>
</svg>
Figure 1. The forward pass runs left to right (black). The backward pass runs right to left (orange), and each arrow applies one local rule to the slope it receives. Notice that the numbers on the arrows are the chapter's running example, and that the rules never reach back further than one step.
```

Slopes tell the scorer how to change. They do not say what a good scorer looks like. So the second half of the chapter asks a sharper question. Collect the weights from every decoder step and you get a table with one row per output step and one column per source position, the **alignment matrix**. Can you set a scorer by hand, with no training, so that every row of that table reads exactly the right position? Building that shows what attention weights have to look like for attention to be right.

## What you already know

Five ideas carry this chapter:

- A slope of the loss with respect to a number says how much the loss moves when that number moves a little. The usual word for it is gradient, and this chapter says slope. A name that starts with `d`, such as `dweights`, holds the slope with respect to the thing that follows.
- In the backward pass, each step multiplies the incoming slope by its local rule, and a value used in several places collects the sum of the slopes from all of them.
- The additive attention forward pass is `u = W1 @ query + keys @ W2.T`, then `t = tanh(u)`, `scores = t @ v`, `weights = softmax(scores)`, and `context = weights @ values` (see [attention as a soft lookup](/lessons/050-attention-as-soft-lookup)). The slope of tanh at $u$ is $1 - \tanh(u)^2$.
- A finite difference nudges one input by a tiny $\epsilon$ in each direction and divides the change in the loss by $2\epsilon$, which approximates the slope (see [softmax and whole-network gradients](/lessons/032-softmax-and-whole-network-gradients)).
- A one-hot row has a single 1 and zeros elsewhere, `np.eye(n)[tokens]` stacks one row per token, and `np.argmax` returns the index of the largest entry.

## Pick a loss, then walk backward

To check slopes, you need a loss. Let the loss be $L = g \cdot \text{context}$ for a fixed vector $g$. Then the slope of $L$ with respect to the context is exactly $g$, and the exercise names it `dcontext`. In a real model, `dcontext` would come from the layers after attention, and here you supply it yourself. This loss also lets you check every other slope with finite differences.

Use the running example from the previous chapter: query $[1, 0]$, keys $[1, 0]$ and $[0, 1]$, values $[10, 0]$ and $[0, 10]$, identity $W_1$ and $W_2$, and $v = [1, 1]$. The forward pass gave scores $0.9640$ and $1.5232$, and weights $0.3637$ and $0.6363$. Take `dcontext` to be $[1, 0]$, which says that only the first entry of the context affects the loss.

### Step 1: the blend

The context is $\sum_i a_i V_i$, a weighted sum of value rows. Each value row is scaled by its weight, so the slope on row $i$ is its weight times `dcontext`, which gives rows $[0.3637, 0]$ and $[0.6363, 0]$. Each weight multiplies a whole value row, so the slope on weight $a_i$ is how much the loss moves per unit of weight, which is $V_i \cdot \text{dcontext}$. The values are $[10, 0]$ and $[0, 10]$, so the weight slopes are $10$ and $0$.

### Step 2: the softmax

The weights are not independent, because they sum to 1. Raising the score of position 1 raises its weight and, to keep the total at 1, lowers the other weights. So a weight's slope cannot be passed back on its own. To see what the rule must be, nudge score $j$ up by a small $\epsilon$ and watch every weight. Weight $j$ rises by $a_j(1 - a_j)\epsilon$, and each other weight $a_i$ falls by $a_i a_j \epsilon$, so that the total stays 1. On the example, nudging score 1 raises weight 1 by $0.3637 \cdot 0.6363 = 0.2314$ per unit and lowers weight 2 by the same amount. The loss then moves by $10 \cdot 0.2314 + 0 \cdot (-0.2314) = 2.314$ per unit. Adding up these effects for each score, the slope on a score compares its weight's slope with the average slope, weighted by the weights:

$$
\text{dscores}_i = a_i \left( \text{dweights}_i - \sum_j a_j \, \text{dweights}_j \right)
$$

Read it aloud: take the slope on weight $i$, subtract the weighted average of all the weight slopes, and multiply by the weight. A weight whose slope is above the average should have its score raised, and one below the average should have it lowered. The factor $a_i$ shrinks the answer for a weight near 0 or near 1, which barely moves when its score does.

On the example, the average is $0.3637 \cdot 10 + 0.6363 \cdot 0 = 3.637$. The differences are $6.363$ and $-3.637$, and multiplying by the weights gives $2.314$ and $-2.314$. Figure 2 draws it. These two numbers sum to zero, which is a sanity check you can apply to any softmax backward pass. Adding the same number to every score leaves the weights unchanged, so the loss cannot depend on such a shift, and the slopes must cancel.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three bar panels: the slopes on the weights, 10 and 0, then the same minus their weighted average, then multiplied by the weights, giving 2.314 and minus 2.314">
<defs><marker id="c042-arrow-ink2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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
<line x1="194.0" y1="100.0" x2="211.0" y2="100.0" class="stroke-ink" marker-end="url(#c042-arrow-ink2)"/>
<line x1="389.0" y1="100.0" x2="406.0" y2="100.0" class="stroke-ink" marker-end="url(#c042-arrow-ink2)"/>
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

The matrix $u$ has a row for each key, and row $i$ is $W_1 q + W_2 k_i$. The vector $W_1 q$ was the same in every row, so in the forward pass it was copied $T$ times. A copy in the forward pass becomes a sum in the backward pass, because the query influenced the loss through every row. So you add the row slopes first, then go back through $W_1$:

$$
\text{dW}_1 = \Big(\sum_i \text{du}_i\Big) q^{\top}, \qquad
\text{dquery} = W_1^{\top} \sum_i \text{du}_i
$$

For the example the sum is $[0.1635, 2.3143] + [-0.972, -0.972] = [-0.8085, 1.3424]$, which is `dquery` because $W_1$ is the identity. The same rule explains the other gradients. The matrices $W_1$, $W_2$, and $v$ are all shared by every key, so each of their slopes is a sum over keys. Each key row is used once, so it keeps its own row:

$$
\text{dW}_2 = \text{du}^{\top} \, \text{keys}, \qquad \text{dkeys} = \text{du} \, W_2
$$

> [!WARNING]
> Sum `du` over axis 0 before calling `np.outer`. A `(T, h)` array passed to `np.outer` is flattened into one long vector, and the gradient comes out with the wrong shape.

In numpy, the last three steps are:

```python
du = np.outer(dscores, v) * (1.0 - t ** 2)     # (T, h)
du_sum = du.sum(axis=0)                        # (h,): the copies collect here
dW1 = np.outer(du_sum, query)                  # (h, d_s)
```

> [!KEY]
> The backward pass is the forward pass in reverse, one local rule per step. The softmax compares each slope with the average, so its answers sum to zero. A tanh passes less slope where it is flat. A value copied in the forward pass collects the sum of the slopes in the backward pass.

## The alignment matrix

At output step 3, which source word is the decoder reading? One call to attention answers that for one step: its weights say how much each source position counts. Stack the weight vector from each step, one per row:

$$
A_{t,i} = \text{weight on source position } i \text{ at output step } t
$$

Row $t$ is where the decoder looked while it wrote token $t$. Each row sums to 1, because each row is a softmax. A cell near 1 means step $t$ read almost only position $i$. You can read the matrix like a picture, and it needs no labels. A trained translation model never receives a list of which source word goes with which output word, yet its matrix often shows those pairs. For translation the dark cells form word pairs, and they cross wherever the two languages order words differently, as in the sketch on the right of Figure 3.

The loop behind the matrix has four moves at each step $t$. The decoder's current state becomes a query. Attention scores that query against every encoder state and returns a context. The decoder uses the context to write token $t$. Its updated state becomes the query for step $t+1$. The encoder states stay fixed, so the same keys and values are read at every step, and what changes is the query, which is why the context changes. The exercise leaves out the recurrent decoder, the part that updates a state from step to step, and hands you the queries directly. That isolates the look-back so you can test it on its own.

For a copy task the target is the identity: step 1 reads position 1, step 2 reads position 2, and so on. A trained model finds weights like these with the backward pass above. Here you build them by hand.

```figure
<svg viewBox="0 0 600 316" role="img" aria-label="Left: a five by five alignment matrix from the copy decoder, dark on the diagonal. Right: a three by three sketch of a translation alignment with a crossing">
<text x="14.0" y="22.0" class="fig-bold">Copy decoder, computed</text>
<text x="172.0" y="50.0" text-anchor="middle" class="fig-small">source position (token)</text>
<text x="84.0" y="66.0" text-anchor="middle" class="fig-small">1 (3)</text>
<text x="128.0" y="66.0" text-anchor="middle" class="fig-small">2 (0)</text>
<text x="172.0" y="66.0" text-anchor="middle" class="fig-small">3 (4)</text>
<text x="216.0" y="66.0" text-anchor="middle" class="fig-small">4 (1)</text>
<text x="260.0" y="66.0" text-anchor="middle" class="fig-small">5 (2)</text>
<text x="56.0" y="96.0" text-anchor="end" class="fig-small">step 1</text>
<rect x="62.0" y="70.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="84.0" y="96.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="106.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="140.0" text-anchor="end" class="fig-small">step 2</text>
<rect x="62.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="114.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="128.0" y="140.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="150.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="184.0" text-anchor="end" class="fig-small">step 3</text>
<rect x="62.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="158.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="172.0" y="184.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="194.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="228.0" text-anchor="end" class="fig-small">step 4</text>
<rect x="62.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="202.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="216.0" y="228.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="238.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="272.0" text-anchor="end" class="fig-small">step 5</text>
<rect x="62.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="246.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="260.0" y="272.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<text x="172.0" y="312.0" text-anchor="middle" class="fig-small">cells below 0.00005 show as 0.00</text>
<text x="340.0" y="22.0" class="fig-bold">Translation, sketched</text>
<text x="406.0" y="90.0" text-anchor="middle" class="fig-small">the</text>
<text x="458.0" y="90.0" text-anchor="middle" class="fig-small">white</text>
<text x="510.0" y="90.0" text-anchor="middle" class="fig-small">house</text>
<text x="374.0" y="130.0" text-anchor="end" class="fig-small">la</text>
<rect x="380.0" y="100.0" width="52.0" height="52.0" class="fill-b stroke-paper"/>
<rect x="432.0" y="100.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="484.0" y="100.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<text x="374.0" y="182.0" text-anchor="end" class="fig-small">maison</text>
<rect x="380.0" y="152.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="432.0" y="152.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="484.0" y="152.0" width="52.0" height="52.0" class="fill-b stroke-paper"/>
<text x="374.0" y="234.0" text-anchor="end" class="fig-small">blanche</text>
<rect x="380.0" y="204.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="432.0" y="204.0" width="52.0" height="52.0" class="fill-b stroke-paper"/>
<rect x="484.0" y="204.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<text x="458.0" y="278.0" text-anchor="middle" class="fig-small">not model output: an illustration</text>
<text x="458.0" y="298.0" text-anchor="middle" class="fig-small">the highlighted cells cross where the</text>
<text x="458.0" y="314.0" text-anchor="middle" class="fig-small">word order differs</text>
</svg>
Figure 3. Each row is the attention weights of one decoder step. In the copy decoder (left, computed by decode_with_attention) every step looks almost entirely at the same position, so the matrix is nearly the identity. Notice the crossing in the sketch: a translation reads positions out of order.
```

## Setting the scorer by hand

You need every query $t$ to give a high score to key $t$ and low scores to the rest, using the additive scorer. Here is a design with reasons for each choice.

- **Queries and keys are $10\, e_t$.** Position $t$ gets a direction of its own, the $t$-th basis vector $e_t$ (a 1 in slot $t$ and zeros elsewhere), scaled by 10. Query $t$ and key $t$ point the same way, and every other pair is perpendicular.
- **$W_1$ and $W_2$ are the identity.** The sum for query $t$ and key $i$ is then $u = 10\,e_t + 10\,e_i$.
- **The values are one-hot rows of the source tokens.** A context that is nearly one of those rows tells you which token was read.

Trace the sum. When $i = t$, the two terms land in the same slot, so $u = 20\,e_t$, which is one active entry. When $i \ne t$, they land in different slots, so $u$ has two active entries of 10. The scale of 10 is what makes this clean, because $\tanh(10)$ and $\tanh(20)$ are both within $10^{-8}$ of 1. Every active entry of $\tanh(u)$ is exactly 1 for practical purposes, so $\tanh(u)$ counts the active entries.

The vector $v$ then turns that count into a score. With $v = -10$ on every entry, the score is $-10$ times the number of active entries. The matching key has one, so it scores $-10$. Every other key has two, so it scores $-20$. A gap of 10 between scores sends a weight of $e^{-10} \approx 4.5 \times 10^{-5}$ to each wrong position. With five source tokens, four wrong positions hold $1.8 \times 10^{-4}$ in total, and the right position gets $0.99982$. Figure 4 shows step 3.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two bar panels for decoder step 3: with v equal to minus 10, the matching position gets nearly all the weight; with v equal to plus 10, it gets almost none">
<text x="20.0" y="24.0" class="fig-bold">v = -10 on every entry</text>
<text x="20.0" y="44.0" class="fig-small">scores:</text>
<line x1="20.0" y1="200.0" x2="286.0" y2="200.0" class="stroke-muted"/>
<line x1="30.0" y1="200.0" x2="68.0" y2="200.0" class="stroke-b"/>
<text x="49.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="49.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="49.0" y="218.0" text-anchor="middle" class="fig-small">pos 1</text>
<line x1="82.0" y1="200.0" x2="120.0" y2="200.0" class="stroke-b"/>
<text x="101.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="101.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="101.0" y="218.0" text-anchor="middle" class="fig-small">pos 2</text>
<rect x="134.0" y="95.0" width="38.0" height="105.0" class="fill-b"/>
<text x="153.0" y="89.0" text-anchor="middle" class="fig-small fig-mono">0.99982</text>
<text x="153.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-10</text>
<text x="153.0" y="218.0" text-anchor="middle" class="fig-small">pos 3</text>
<line x1="186.0" y1="200.0" x2="224.0" y2="200.0" class="stroke-b"/>
<text x="205.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="205.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="205.0" y="218.0" text-anchor="middle" class="fig-small">pos 4</text>
<line x1="238.0" y1="200.0" x2="276.0" y2="200.0" class="stroke-b"/>
<text x="257.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="257.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="257.0" y="218.0" text-anchor="middle" class="fig-small">pos 5</text>
<text x="320.0" y="24.0" class="fig-bold">v = +10 on every entry</text>
<text x="320.0" y="44.0" class="fig-small">scores:</text>
<line x1="320.0" y1="200.0" x2="586.0" y2="200.0" class="stroke-muted"/>
<rect x="330.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="349.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="349.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="349.0" y="218.0" text-anchor="middle" class="fig-small">pos 1</text>
<rect x="382.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="401.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="401.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="401.0" y="218.0" text-anchor="middle" class="fig-small">pos 2</text>
<line x1="434.0" y1="200.0" x2="472.0" y2="200.0" class="stroke-b"/>
<text x="453.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">1.1e-5</text>
<text x="453.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">10</text>
<text x="453.0" y="218.0" text-anchor="middle" class="fig-small">pos 3</text>
<rect x="486.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="505.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="505.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="505.0" y="218.0" text-anchor="middle" class="fig-small">pos 4</text>
<rect x="538.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="557.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="557.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="557.0" y="218.0" text-anchor="middle" class="fig-small">pos 5</text>
<text x="14.0" y="250.0" class="fig-small">Step 3 matches position 3 (darker bar). With v = -10 a match activates one entry and scores -10;</text>
<text x="14.0" y="268.0" class="fig-small">the others activate two entries and score -20. With v = +10 the order flips, so the match loses.</text>
</svg>
Figure 4. Same queries, same keys, opposite sign of v. Notice how few numbers decide the outcome: scores of -10 against -20 put 0.99982 on the right position, and the flipped sign sends the right position's weight down to 1e-5.
```

The sign of $v$ matters. Flip it to $+10$, and the match scores $+10$ while every other key scores $+20$, which is the same "active entries" count in the other direction. The non-matching positions win. The diagonal falls to about $1.1 \times 10^{-5}$, and each of the four other positions gets about $0.25$.

> [!WARNING]
> A positive `v` does not give a weaker copy decoder, it gives the opposite one. With `v_sign=+1`, the decoded tokens no longer equal the source.

The decoder reads the token off the context with `np.argmax(context)`. Because the context is nearly the one-hot row of the token at position $t$, the argmax is that token. In the exercise, `decode_with_attention([3, 0, 4, 1, 2], 5)` returns `[3, 0, 4, 1, 2]`.

### Closing the loop with the order-blind summary

The order-blind summary from the start of [the previous chapter](/lessons/050-attention-as-soft-lookup) is the mean of the one-hot rows:

```python
one_hot = np.eye(vocab_size)[tokens]     # (T, vocab_size)
summary = one_hot.mean(axis=0)           # (vocab_size,)
```

For `[0, 1, 2]` and `[2, 1, 0]` it is $[1/3, 1/3, 1/3]$ both times, so a decoder reading it cannot recover the order. The attention decoder reads the same two sources and returns `[0, 1, 2]` and `[2, 1, 0]`. Order is kept because it lives in which position each step reads, and the weights say that out loud.

Be fair about what this shows. The hand-built queries $10\,e_t$ already name the position to read at each step, so the decoder had the order handed to it. What the comparison proves is that attention can carry order through, because the encoder keeps one vector per position. It does not prove that a decoder finds the right queries by itself. Training has to teach the decoder to produce them from its own state, and the backward pass in the first half of this chapter is how.

> [!KEY]
> Attention gradients flow one local rule at a time, and a decoder that attends does not need the encoder to squeeze the source into one vector. Each step gets its own query, reads its own blend of the encoder states, and the alignment matrix shows where it looked. In the copy task, scores of $-10$ against $-20$ give a nearly perfect identity matrix, and the order that an average throws away is still there to read, as long as the queries ask for it.

> [!RECAP]
> Before you build:
> - Blend: `dweights = values @ dcontext` and `dvalues = np.outer(weights, dcontext)`.
> - Softmax: `dscores = weights * (dweights - np.sum(weights * dweights))`, and the result sums to zero.
> - Dot with `v`: `dv = t.T @ dscores` and `dt = np.outer(dscores, v)`. Tanh: `du = dt * (1 - t**2)`. Projections: `du.sum(axis=0)` goes to `dW1` and `dquery`, and `dW2 = du.T @ keys`, `dkeys = du @ W2`.
> - `alignment_matrix` calls `additive_attention` once per query and stacks the contexts and the weights, so each row sums to 1.
> - The copy decoder uses queries and keys `10 * np.eye(T)`, identity `W1` and `W2`, one-hot values, and `v = -10 * np.ones(T)`. `mean_summary` is the same for every ordering of the tokens.

## Build it step by step

The starter already holds `softmax`, `attend`, `additive_scores`, and `additive_attention` from the previous chapter. The checks test these functions in order:

1. **`softmax_backward(weights, dweights)`.** Takes both of shape `(T,)` and returns `dscores` of shape `(T,)`. Use the formula in Step 2. The usual mistake is dropping the subtraction of the weighted average, which gives `weights * dweights`.
2. **`context_backward(dcontext, weights, values)`.** Returns `(dweights, dvalues)` with shapes `(T,)` and `(T, d_v)`. The usual mistake is swapping the two outputs.
3. **`attention_backward(dcontext, query, keys, values, W1, W2, v)`.** Reruns the forward pass, keeps `t`, calls your two functions, then applies the rules from Steps 3 and 4. It returns `(dquery, dkeys, dvalues, dW1, dW2, dv)`, each shaped like its input. The usual mistake is forgetting the factor $1 - t^2$, or forgetting to sum `du` before the outer product.
4. **`alignment_matrix(queries, keys, values, W1, W2, v)`.** Takes `queries` of shape `(T_y, d_s)` and returns `(contexts, A)` of shapes `(T_y, d_v)` and `(T_y, T_x)`. Loop over the queries, call `additive_attention`, and stack the results with `np.stack`. The usual mistake is passing the whole `queries` array in one call, which treats it as a single query.
5. **`mean_summary(source_tokens, vocab_size)`.** Returns the mean of the one-hot rows, shape `(vocab_size,)`. The usual mistake is averaging over axis 1 instead of axis 0.
6. **`decode_with_attention(source_tokens, vocab_size, v_sign=-1.0)`.** Builds the hand-set parameters described above, calls `alignment_matrix`, and reads each token with `np.argmax`. It returns `(decoded, alignment)`. The usual mistake is a positive `v`.

## Where this breaks

A saturated tanh passes almost no slope, as Step 3 showed, so a scorer whose hidden units sit on the flat parts learns slowly. A hand-written backward pass can also be wrong while still returning arrays of the right shape, and only a finite-difference check catches that. This is why the checks nudge every input by a tiny amount and compare.

Attention fixes what the decoder can see, and it still fails in familiar ways:

- The decoder can read the same source word at many steps, and the output repeats a phrase. A fix called coverage keeps a running total of the weight each position has received and feeds it to the scorer, so the decoder can learn to move on.
- The encoder still reads one step at a time, and attention does not speed that chain up. The next part replaces it with attention itself.
- The work grows with `T_x * T_y`, because every output step looks at every source position. Compute `W2 @ keys` once and reuse it at every step, since it does not depend on the decoder's state.

## The exercise

You build `softmax_backward`, `context_backward`, `attention_backward`, `alignment_matrix`, `mean_summary`, and `decode_with_attention`. The checks confirm that the softmax slopes on the worked example are $2.314$ and $-2.314$, that they sum to zero, and that `softmax_backward`, `context_backward`, and all six gradients of `attention_backward` match finite differences. A single-key case must give zero for every gradient except `dvalues`. For the alignment side, the checks require that every row of the alignment matrix sums to 1 and equals a stack of single `additive_attention` calls, and that `decode_with_attention` copies `[3, 0, 4, 1, 2]` exactly with an alignment matrix within 0.01 of the identity. They also require that `v_sign=+1.0` pushes the diagonal below 0.01, and that the attention decoder tells `[0, 1, 2]` from `[2, 1, 0]` while the mean summary cannot.
