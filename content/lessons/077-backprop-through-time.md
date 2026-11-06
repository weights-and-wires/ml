---
title: "Walk the chain backward: backpropagation through time"
slug: 077-backprop-through-time
part: "Part 5 — Vision and sequences"
summary: Unroll the cell into a chain and run ordinary backprop on it, so each shared weight collects a sum over steps.
minutes: 40
gpu: false
objectives:
  - Say why unrolling a recurrent cell turns training into ordinary backpropagation on a deep chain.
  - Work one backward step by hand, and say which values flow into the gradient of each weight.
  - Explain why a shared weight's gradient is a sum over steps, and why the previous state matters in that sum.
  - Write the gradient of tanh, the gradient of one step, and the backward loop, and check them against finite differences.
prereqs:
  - 055-backprop-array-operations
  - 072-convolution-backward
  - 076-recurrent-cell
exercise: 077-backprop-through-time
---

# Walk the chain backward: backpropagation through time

The sensor model from the last chapter read a run of readings and made a prediction at the end, and the prediction was wrong. The loss is a single number. Three weights, $W_{xh}$, $W_{hh}$, and the bias $b$, were used at every step to produce that prediction. Which of them should change, and by how much?

One way is to nudge each weight, rerun the whole sequence, and see whether the loss went down. That needs one full pass per weight, which is hopeless for a cell with 400 weights, and it does not say which step the blame belongs to. A second way is to apply the chain rule you already use for dense layers. The trouble is that the same weight sits at every step, so it looks as if there is no clean place to apply it.

There is. Draw the cell once per step, as in the last chapter, and the recurrent network becomes a long ordinary network whose layers happen to share weights. Run backpropagation on it, and the chain rule does the rest. This chapter walks that backward pass on two steps.

## What you already know

You already use these ideas:

- Backpropagation starts at the loss and walks backward, multiplying by the local slope at each operation.
- When a value is used in several places, its gradient is the sum of the gradients from each place.
- For $z = x W^{\top}$, the gradient on $W$ is the upstream gradient transposed times $x$, and the gradient on $x$ is the upstream gradient times $W$.
- If $h = \tanh(z)$, the slope of tanh at $z$ is $1 - h^2$, which you can compute from $h$ alone.
- You can check a gradient by nudging a value by a tiny amount and seeing how the loss moves.

## Unroll the chain

Use the running example again: $W_{xh} = 0.5$, $W_{hh} = 0.8$, $b = 0$, and readings $x_1 = 1$ and $x_2 = 0.5$. This time start from $h_0 = 0.3$ instead of zero, so the starting state has a nonzero value for $W_{hh}$ to multiply. The forward pass gives $h_1 = \tanh(0.5 + 0.8 \times 0.3) = \tanh(0.74) = 0.6291$ and $h_2 = \tanh(0.25 + 0.8 \times 0.6291) = \tanh(0.7533) = 0.6371$.

To keep the numbers clean, pretend the loss pulls on $h_2$ with a strength of 1. In other words, raising $h_2$ by a small amount raises the loss by that same amount. A real loss gives a different strength, but every gradient below is proportional to it, so the pattern holds. Figure 1 shows the forward pass in blue, and the gradient walking back through it in orange.

```figure
<svg viewBox="0 0 620 285" role="img" aria-label="Forward values flow right in blue along two cells; gradients flow left in orange, multiplied by the tanh slope and by the recurrent weight at each step">
<defs><marker id="c086-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c086-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="62" class="fig-bold">Forward pass</text>
<rect x="140" y="95" width="80" height="50" rx="6" class="fill-soft-a stroke-a"/>
<text x="180" y="115" text-anchor="middle">step 1</text>
<text x="180" y="133" text-anchor="middle" class="fig-small">x1 = 1</text>
<rect x="380" y="95" width="80" height="50" rx="6" class="fill-soft-a stroke-a"/>
<text x="420" y="115" text-anchor="middle">step 2</text>
<text x="420" y="133" text-anchor="middle" class="fig-small">x2 = 0.5</text>
<line x1="20" y1="120" x2="138" y2="120" class="stroke-a" marker-end="url(#c086-arrow-a)"/>
<text x="80" y="112" text-anchor="middle" class="fig-small">h0 = 0.3</text>
<line x1="222" y1="120" x2="378" y2="120" class="stroke-a" marker-end="url(#c086-arrow-a)"/>
<text x="300" y="112" text-anchor="middle" class="fig-small">h1 = 0.6291</text>
<line x1="462" y1="120" x2="600" y2="120" class="stroke-a" marker-end="url(#c086-arrow-a)"/>
<text x="535" y="112" text-anchor="middle" class="fig-small">h2 = 0.6371</text>
<text x="20" y="188" class="fig-bold">Backward pass</text>
<rect x="528" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="560" y="236" text-anchor="middle">1</text>
<text x="560" y="264" text-anchor="middle" class="fig-small">dh2</text>
<rect x="388" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="420" y="236" text-anchor="middle">0.5941</text>
<text x="420" y="264" text-anchor="middle" class="fig-small">dz2</text>
<rect x="268" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="300" y="236" text-anchor="middle">0.4753</text>
<text x="300" y="264" text-anchor="middle" class="fig-small">dh1</text>
<rect x="148" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="180" y="236" text-anchor="middle">0.2871</text>
<text x="180" y="264" text-anchor="middle" class="fig-small">dz1</text>
<rect x="28" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="60" y="236" text-anchor="middle">0.2297</text>
<text x="60" y="264" text-anchor="middle" class="fig-small">dh0</text>
<line x1="526" y1="231" x2="454" y2="231" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<text x="490" y="209" text-anchor="middle" class="fig-small fill-b">× 0.5941</text>
<text x="490" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="386" y1="231" x2="334" y2="231" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<text x="360" y="209" text-anchor="middle" class="fig-small fill-b">× 0.8</text>
<text x="360" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="266" y1="231" x2="214" y2="231" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<text x="240" y="209" text-anchor="middle" class="fig-small fill-b">× 0.6042</text>
<text x="240" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="146" y1="231" x2="94" y2="231" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<text x="120" y="209" text-anchor="middle" class="fig-small fill-b">× 0.8</text>
<text x="120" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="180" y1="147" x2="180" y2="215" class="stroke-muted dashed"/>
<line x1="420" y1="147" x2="420" y2="215" class="stroke-muted dashed"/>
</svg>
Figure 1. A pull of 1 on h2 travels back through two cells. Each cell contributes two multiplications: the tanh slope (0.5941 at step 2, 0.6042 at step 1) and the weight 0.8. Notice that the value shrinks at every multiplication, from 1 down to 0.2297 at h0.
```

Read the orange row from right to left. Two things happen at every cell. First, the gradient passes through tanh, which scales it by the slope $1 - h^2$ at that step, because a flat part of tanh passes less of a nudge than a steep part. Second, it passes through the recurrent weight, which scales it by 0.8, because the previous state entered the sum multiplied by 0.8. The 1 on $h_2$ becomes $0.5941$ at the cell, then $0.4753$ on $h_1$, then $0.2871$, then $0.2297$ on $h_0$.

## Where the loss lives

The gradient at a state has two sources, which is why the loop below adds two things. The first is the loss attached directly to that state, written $dH_t$. In the last chapter the readout looked at the last state only, so $dH_t$ was 0 everywhere except the last step. If the task scores every step, as in labelling every word of a sentence, each state has its own $dH_t$. The second source is everything later in the sequence, which reaches $h_t$ by flowing back through the next cell. Call that the **carry**. The full gradient on a state is

$$
dh_t = dH_t + \text{carry}
$$

In the example, $dH_2 = 1$ and the carry into step 2 is 0, because nothing comes after it. At step 1, $dH_1 = 0$ and the carry is $0.4753$. So $dh_1 = 0.4753$ comes only from the future.

## One step backward

Take step 2 alone. The state came out of $h_2 = \tanh(z_2)$, so the gradient on the sum inside tanh is $dz_2 = dh_2 \times (1 - h_2^2) = 1 \times 0.5941 = 0.5941$. You do not need $z_2$, because the slope can be computed from the stored output $h_2$.

The sum $z_2 = x_2 W_{xh}^{\top} + h_1 W_{hh}^{\top} + b$ has three terms, so $dz_2$ flows to each of them. Each gets the rule from the dense layer: the gradient on a weight is $dz$ times the value that weight multiplied, and the gradient on a value is $dz$ times the weight. Figure 2 lays out the five results.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="A box diagram of one backward step: the incoming gradient passes through the tanh gate and splits into five outputs">
<defs><marker id="c086-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="10" y="115" width="110" height="56" rx="6" class="fill-soft-b stroke-b"/>
<text x="65" y="138" text-anchor="middle" class="fig-bold">dh_t</text>
<text x="65" y="156" text-anchor="middle" class="fig-small">(n, d_h)</text>
<text x="65" y="108" text-anchor="middle" class="fig-small">dH[t] + carry</text>
<line x1="122" y1="143" x2="148" y2="143" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<rect x="150" y="115" width="110" height="56" rx="6" class="fill-soft-d stroke-d"/>
<text x="205" y="138" text-anchor="middle" class="fig-bold">tanh gate</text>
<text x="205" y="156" text-anchor="middle" class="fig-small">× (1 − h_t²)</text>
<line x1="262" y1="143" x2="288" y2="143" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<rect x="290" y="115" width="90" height="56" rx="6" class="fill-soft-b stroke-b"/>
<text x="335" y="138" text-anchor="middle" class="fig-bold">dz</text>
<text x="335" y="156" text-anchor="middle" class="fig-small">(n, d_h)</text>
<rect x="420" y="14" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="33">dx_t = dz W_xh</text>
<text x="430" y="49" class="fig-small">(n, d_x)</text>
<line x1="382" y1="143" x2="418" y2="35" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<rect x="420" y="68" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="87">dW_xh = dzᵀ x_t</text>
<text x="430" y="103" class="fig-small">(d_h, d_x)</text>
<line x1="382" y1="143" x2="418" y2="89" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<rect x="420" y="122" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="141">dW_hh = dzᵀ h_prev</text>
<text x="430" y="157" class="fig-small">(d_h, d_h)</text>
<line x1="382" y1="143" x2="418" y2="143" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<rect x="420" y="176" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="195">db = sum of dz over n</text>
<text x="430" y="211" class="fig-small">(d_h,)</text>
<line x1="382" y1="143" x2="418" y2="197" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<rect x="420" y="230" width="190" height="42" rx="6" class="fill-soft-d stroke-d"/>
<text x="430" y="249">dh_prev = dz W_hh</text>
<text x="430" y="265" class="fig-small">(n, d_h)</text>
<line x1="382" y1="143" x2="418" y2="251" class="stroke-b" marker-end="url(#c086-arrow-b)"/>
<text x="500" y="288" text-anchor="middle" class="fig-small fill-d">dh_prev is the carry for step t − 1</text>
</svg>
Figure 2. One step of the backward loop. The tanh gate turns dh_t into dz, and dz alone produces every other gradient. Four outputs are collected into sums or written to dx, and the purple one, dh_prev, becomes the carry for the previous step.
```

At step 2 these are $dW_{xh} = 0.5941 \times 0.5 = 0.2970$, $dW_{hh} = 0.5941 \times 0.6291 = 0.3738$, $db = 0.5941$, $dx_2 = 0.5941 \times 0.5 = 0.2970$, and the part sent to the previous state, $dh_1 = 0.5941 \times 0.8 = 0.4753$. That last value is the carry. Written for a batch whose rows are examples, the five lines are

```python
dz = dh_t * (1.0 - h_t ** 2)
dx_t, dh_prev = dz @ W_xh, dz @ W_hh
dW_xh, dW_hh, db = dz.T @ x_t, dz.T @ h_prev, dz.sum(axis=0)
```

Here `dz.sum(axis=0)` adds up the rows of the batch, because the bias was added to every example.

## Each copy of a weight gets a nudge

Next, step 1, with $dh_1 = 0 + 0.4753$. The slope at step 1 is $1 - 0.6291^2 = 0.6042$, so $dz_1 = 0.4753 \times 0.6042 = 0.2871$. The weights at step 1 get $dW_{xh} = 0.2871 \times 1 = 0.2871$, $dW_{hh} = 0.2871 \times 0.3 = 0.0861$, and $db = 0.2871$. The carry that leaves step 1 is $0.2871 \times 0.8 = 0.2297$, the gradient on $h_0$.

Think of $W_{hh}$ as being photocopied once per step. Each copy was used once and receives its own nudge, and the weight itself is one object that gets all of them. So its gradient is the sum, $0.0861 + 0.3738 = 0.4599$. In the same way $dW_{xh} = 0.2871 + 0.2970 = 0.5842$ and $db = 0.2871 + 0.5941 = 0.8812$. Figure 3 shows the sums.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bars showing the gradient of W_hh and W_xh as a sum of one contribution from each step">
<defs></defs>
<text x="160" y="32" text-anchor="middle" class="fig-bold">Gradient of W_hh</text>
<line x1="55" y1="215" x2="275" y2="215" class="stroke-muted"/>
<rect x="60" y="193.5" width="44" height="21.5" class="fill-soft-b stroke-b"/>
<text x="82" y="186.5" text-anchor="middle" class="fig-small">0.0861</text>
<text x="82" y="233" text-anchor="middle" class="fig-small">step 1 copy</text>
<rect x="135" y="121.6" width="44" height="93.4" class="fill-soft-b stroke-b"/>
<text x="157" y="114.6" text-anchor="middle" class="fig-small">0.3738</text>
<text x="157" y="233" text-anchor="middle" class="fig-small">step 2 copy</text>
<rect x="210" y="100" width="44" height="115" class="fill-b"/>
<text x="232" y="93" text-anchor="middle" class="fig-small">0.4599</text>
<text x="232" y="233" text-anchor="middle" class="fig-small">sum</text>
<text x="440" y="32" text-anchor="middle" class="fig-bold">Gradient of W_xh</text>
<line x1="335" y1="215" x2="555" y2="215" class="stroke-muted"/>
<rect x="340" y="143.2" width="44" height="71.8" class="fill-soft-b stroke-b"/>
<text x="362" y="136.2" text-anchor="middle" class="fig-small">0.2871</text>
<text x="362" y="233" text-anchor="middle" class="fig-small">step 1 copy</text>
<rect x="415" y="140.7" width="44" height="74.3" class="fill-soft-b stroke-b"/>
<text x="437" y="133.7" text-anchor="middle" class="fig-small">0.2970</text>
<text x="437" y="233" text-anchor="middle" class="fig-small">step 2 copy</text>
<rect x="490" y="69" width="44" height="146" class="fill-b"/>
<text x="512" y="62" text-anchor="middle" class="fig-small">0.5842</text>
<text x="512" y="233" text-anchor="middle" class="fig-small">sum</text>
<text x="300" y="250" text-anchor="middle" class="fig-small">The step 1 copy of W_hh multiplies the starting state 0.3, so its bar is small but not zero.</text>
</svg>
Figure 3. One shared weight is several copies. Each copy gets its own nudge, and the weight's gradient is their sum: 0.0861 + 0.3738 = 0.4599 for W_hh, and 0.2871 + 0.2970 = 0.5842 for W_xh.
```

The full backward loop is the same two moves repeated from the last step to the first. Add $dH_t$ to the carry, take one step backward, add that step's weight gradients to the running sums, and keep the carry it returns. After step 1, the carry is $dh_0$.

$$
dh_t = dH_t + \text{carry}, \qquad
dW = \sum_t dW^{(t)}, \qquad
\text{carry} \leftarrow dz_t\, W_{hh}
$$

Read it aloud: at each step the gradient on the state is its own loss plus what came back from the future, each weight's gradient is the total of its per-step nudges, and the carry is the gradient that goes to the previous state. In code, inside `for t in reversed(range(T))`, it looks like

```python
h_prev = h_seq[t - 1] if t > 0 else h0
dx[t], carry, dW_xh_t, dW_hh_t, db_t = step_backward(dH[t] + carry, h_seq[t], h_prev, x[t], W_xh, W_hh)
dW_hh += dW_hh_t
```

> [!WARNING]
> Row `t` of the stored states is $h_{t+1}$, so the state before step `t` is `h_seq[t - 1]`, or `h0` when `t` is 0. Using `h_seq[t]` there gives $dW_{hh} = 0.5592$ in the example instead of $0.4599$, and with $h_0 = 0$ the mistake at step 1 would be hidden.

## Trust the numbers by checking them

You can test the whole loop with the nudge idea from the opening, which serves here as a check instead of the method. Add $10^{-6}$ to $W_{hh}$ and subtract $10^{-6}$ from it, rerun the forward pass each time, and divide the difference in $h_2$ by $2 \times 10^{-6}$. The result is $0.4599$, matching the sum above. The exercise checks all five outputs this way: if a gradient in the prose and the code disagree, the finite difference tells you which one is wrong.

> [!KEY]
> Backpropagation through time is ordinary backpropagation on the unrolled chain. The gradient on each state is its own loss plus a carry from the future, and each shared weight's gradient is the sum of the nudges from every copy. That answers the opening question: the loss blames each weight in proportion to how much each copy of it mattered.

> [!RECAP]
> - Unrolled, a recurrent net is a chain of identical cells, so the chain rule applies as usual.
> - The gradient on a state is $dH_t$ plus the carry from later steps. The carry starts at zero.
> - At each step, multiply by $1 - h_t^2$ to get $dz$, then use $dz$ to produce $dx$, the carry $dz\,W_{hh}$, and this step's weight gradients.
> - Each shared weight's gradient is the sum of its per-step gradients, and the previous state at step 0 is $h_0$.
> - Check every output against finite differences.

## Build it step by step

The checks call the functions in this order. The cell, the loop that runs it, and the readout from the last chapter are already written for you.

1. **`tanh_backward(dh, h)`.** `h` is the output of tanh, and `dh` is the gradient with respect to it. Return `dh * (1 - h**2)`. The usual mistake is applying tanh to `h` a second time, which gives $1 - \tanh(h)^2$.
2. **`step_backward(dh_t, h_t, h_prev, x_t, W_xh, W_hh)`.** Call your tanh gradient to get `dz`, then return `(dx_t, dh_prev, dW_xh, dW_hh, db)`. The shapes match `x_t`, `h_prev`, `W_xh`, `W_hh`, and the bias. The usual mistake is a missing transpose in the weight gradients.
3. **`bptt(dH, x, h0, W_xh, W_hh, b)`.** Run `rnn_forward` once, then loop from the last step to the first, with the carry starting at zeros. Pass `dH[t] + carry` to `step_backward`, add the weight gradients into running sums, and copy `dx[t]`. Return `(dx, dW_xh, dW_hh, db, dh0)`, where `dh0` is the final carry. The usual mistake is using `h_seq[t]` as the previous state.

## Cost and scaling

The backward pass visits each step once and does the same size of work as the forward pass, so its time grows in a straight line with $T$. It needs every state from the forward pass, which is why the memory for training grows with $n T d_h$ even though the weights do not grow at all. The chapter "Taming long sequences: truncation and clipping" shows how to cap that memory.

## Where this breaks

- If you forget the sum over steps and keep only the last step's weight gradient, the weights train on a fraction of the signal and learn slowly or not at all.
- The carry passes through a multiplication at every step. The next chapter asks what that does when there are many steps.

## The exercise

You build `tanh_backward`, `step_backward`, and `bptt`. The checks compare `step_backward` and `bptt` to the hand numbers above, including $dW_{hh} = 0.4599$ with the nonzero start. They compare every output to finite differences, for the input, both weights, the bias, and $h_0$. They also check that a loss on every step gives the same gradients as the losses of each step added up, which is the sum rule at work.
