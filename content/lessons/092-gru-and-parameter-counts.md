---
title: "The GRU: one gate that forgets and writes, and what each cell costs"
slug: 092-gru-and-parameter-counts
part: "Part 5 — Vision and sequences"
summary: A GRU merges the LSTM's forget and input gates into one update gate that blends old and new, adds a reset gate, and so needs three weight sets instead of four.
minutes: 35
gpu: false
objectives:
  - Explain why one update gate can replace the forget and input gates, and what the cell gives up by doing so.
  - Work one GRU step by hand from fully specified weights, and say when the reset gate matters.
  - Write the GRU forward step in numpy, with the reset gate applied before the recurrent matrix.
  - Count the parameters of a plain RNN, a GRU, and an LSTM for any input and hidden size.
prereqs:
  - 089-why-rnns-forget-the-belt
  - 090-gates-and-the-lstm-update
  - 091-lstm-backward-step
exercise: 092-gru-and-parameter-counts
---

# The GRU: one gate that forgets and writes, and what each cell costs

An LSTM with 512 inputs and 512 hidden units holds 2,099,200 numbers. All of them are stored, all of them are updated by the optimiser, and all of them are multiplied again at every step of every sequence. If a cell with a quarter fewer numbers learns the same task, you train it faster and run it in less memory. The question for this chapter is whether the belt's benefit survives in a simpler cell, and what the simpler cell saves.

Look back at the store-hold-read example from "Gates and the LSTM memory update". When the forget gate was 0, the input gate was 1, and when the forget gate was 1, the input gate was 0. The two gates moved in opposite directions. If they usually do, one gate could do both jobs, and the cell would need one weight set fewer.

That is the idea behind the gated recurrent unit, or GRU. It uses one update gate $z$ to decide how much of the old state to keep and how much of a fresh proposal to write, and it adds a second gate, the reset gate, that decides how much of the past the proposal gets to see. You build the blend, the full step, and a parameter count for three kinds of cell.

## What you already know

Five ideas carry this chapter:

- A gate is $\sigma(xW + hU + b)$, a fraction between 0 and 1 for every hidden unit (see "Gates and the LSTM memory update").
- The candidate is a tanh of a weighted sum of $x$ and $h$, a proposal for what to write.
- An LSTM keeps a private cell and a shown value, and its forget and input gates scale the old cell and the candidate separately.
- A belt gate near one lets the gradient survive many steps (see "Why a recurrent net forgets, and the belt that fixes it").
- A weight set for one gate has a matrix `W` for the input, a matrix `U` for the previous state, and a bias `b`.

## One gate that blends

The GRU has no separate cell. It keeps a single state $h$, which is both its memory and the value it shows. With no separate cell there is no output gate either, since nothing is hidden from the next layer. Its update is a blend of the old state and a candidate $\tilde{h}$:

$$
h_{\text{new}} = (1 - z) \odot h + z \odot \tilde{h}
$$

Read it aloud: the new state keeps the fraction $1 - z$ of the old state and takes the fraction $z$ of the candidate. The two shares always add to 1, so the new state sits between the old state and the candidate, never beyond. A gate of $z = 0$ keeps the old state exactly, and $z = 1$ replaces it. This is the store-hold-read pattern from the last chapter, with the forget gate and the input gate tied together as $1 - z$ and $z$. In numpy, `(1 - z) * h + z * candidate`. This is also why the belt survives. The old state reaches the new one through the factor $1 - z$, so when the net sets $z$ near 0 to hold a memory, the gradient crosses that step almost unchanged, exactly like a forget gate near 1.

Figure 1 shows the blend with the numbers used in this chapter: an old state of 0.5, a candidate of 0.8483, and four settings of $z$.

```figure
<svg viewBox="0 0 620 320" role="img" aria-label="Four rows showing a GRU blend of an old state 0.5 and a candidate 0.8483 on a number line. As the update gate z grows from 0 to 1, the new state slides from the old state to the candidate. At z = 0.73 it is 0.7546.">
<circle cx="120" cy="22" r="6" class="fill-a"/>
<text x="132" y="27" class="fig-small">old state h = 0.5</text>
<circle cx="300" cy="22" r="6" class="fill-c"/>
<text x="312" y="27" class="fig-small">candidate 0.8483</text>
<circle cx="470" cy="22" r="7" class="fill-b"/>
<text x="482" y="27" class="fig-small">new state</text>
<text x="20" y="80" class="fig-mono">z = 0</text>
<line x1="110" y1="75" x2="570" y2="75" class="stroke-grid"/>
<line x1="202" y1="75" x2="522.4" y2="75" class="stroke-muted"/>
<circle cx="202" cy="75" r="6" class="fill-a"/>
<circle cx="522.4" cy="75" r="6" class="fill-c"/>
<circle cx="202" cy="75" r="8" class="fill-b"/>
<text x="202" y="61" text-anchor="middle" class="fig-small">0.50</text>
<text x="20" y="135" class="fig-mono">z = 0.27</text>
<line x1="110" y1="130" x2="570" y2="130" class="stroke-grid"/>
<line x1="202" y1="130" x2="522.4" y2="130" class="stroke-muted"/>
<circle cx="202" cy="130" r="6" class="fill-a"/>
<circle cx="522.4" cy="130" r="6" class="fill-c"/>
<circle cx="288.2" cy="130" r="8" class="fill-b"/>
<text x="288.2" y="116" text-anchor="middle" class="fig-small">0.59</text>
<rect x="10" y="161" width="600" height="48" rx="6" class="fill-soft-b"/>
<text x="20" y="190" class="fig-mono">z = 0.73</text>
<line x1="110" y1="185" x2="570" y2="185" class="stroke-grid"/>
<line x1="202" y1="185" x2="522.4" y2="185" class="stroke-muted"/>
<circle cx="202" cy="185" r="6" class="fill-a"/>
<circle cx="522.4" cy="185" r="6" class="fill-c"/>
<circle cx="436.2" cy="185" r="8" class="fill-b"/>
<text x="436.2" y="171" text-anchor="middle" class="fig-small fig-bold">0.7546</text>
<text x="20" y="245" class="fig-mono">z = 1</text>
<line x1="110" y1="240" x2="570" y2="240" class="stroke-grid"/>
<line x1="202" y1="240" x2="522.4" y2="240" class="stroke-muted"/>
<circle cx="202" cy="240" r="6" class="fill-a"/>
<circle cx="522.4" cy="240" r="6" class="fill-c"/>
<circle cx="522.4" cy="240" r="8" class="fill-b"/>
<text x="522.4" y="226" text-anchor="middle" class="fig-small">0.85</text>
<line x1="110" y1="270" x2="570" y2="270" class="stroke-muted"/>
<line x1="110" y1="270" x2="110" y2="275" class="stroke-muted"/>
<text x="110" y="290" text-anchor="middle" class="fig-small">0.4</text>
<line x1="202" y1="270" x2="202" y2="275" class="stroke-muted"/>
<text x="202" y="290" text-anchor="middle" class="fig-small">0.5</text>
<line x1="294" y1="270" x2="294" y2="275" class="stroke-muted"/>
<text x="294" y="290" text-anchor="middle" class="fig-small">0.6</text>
<line x1="386" y1="270" x2="386" y2="275" class="stroke-muted"/>
<text x="386" y="290" text-anchor="middle" class="fig-small">0.7</text>
<line x1="478" y1="270" x2="478" y2="275" class="stroke-muted"/>
<text x="478" y="290" text-anchor="middle" class="fig-small">0.8</text>
<line x1="570" y1="270" x2="570" y2="275" class="stroke-muted"/>
<text x="570" y="290" text-anchor="middle" class="fig-small">0.9</text>
</svg>
Figure 1. The new state is a point between the old state (blue) and the candidate (green), and the update gate z slides it from one to the other. Notice that z = 0 leaves the state untouched, z = 1 replaces it, and the highlighted row, z = 0.73, gives 0.7546.
```

Take one unit with every weight written out, so that you can reproduce the numbers. The input is $x = 1$ and the old state is $h = 0.5$. All biases are 0. The update gate has $W_z = 1$ and $U_z = 0$, so $z = \sigma(1) = 0.7311$. The reset gate has $W_r = 0$ and $U_r = 0$, so $r = \sigma(0) = 0.5$. The candidate has $W_h = 1$ and $U_h = 1$, and sees the old state scaled by $r$, which is $0.5 \times 0.5 = 0.25$, so the candidate is $\tanh(1 \times 1 + 0.25 \times 1) = \tanh(1.25) = 0.8483$. The new state keeps $0.2689 \times 0.5 = 0.1345$ of the old state and writes $0.7311 \times 0.8483 = 0.6201$ of the candidate, and the sum is $0.7546$.

What does the GRU give up? In an LSTM the forget and input gates are independent, so a cell can keep its whole old value and add a full candidate on top ($f = 1$ and $i = 1$), and the cell grows. In a GRU the shares always sum to 1, so keeping more of the past means writing less of the present. For many tasks that is a fair trade for a quarter fewer numbers, and it is a trade, not a free saving.

## The reset gate: hiding the past from the candidate

The candidate is a proposal computed from the input and the old state. Sometimes the old state would mislead it. Suppose a sentence has just ended and a fresh clause begins. The state still describes the old clause, and a candidate computed from it would carry that description into the next one. The **reset gate** $r$ lets the net hide the past from the candidate: it scales the old state before the candidate sees it. When $r$ is near 0, the candidate is a fresh function of the input alone, and when $r$ is near 1, it sees the whole state.

$$
\begin{aligned}
z &= \sigma(x W_z + h U_z + b_z) && \text{update gate} \\
r &= \sigma(x W_r + h U_r + b_r) && \text{reset gate} \\
\tilde{h} &= \tanh\left(x W_h + (r \odot h) U_h + b_h\right) && \text{candidate} \\
h_{\text{new}} &= (1 - z) \odot h + z \odot \tilde{h}
\end{aligned}
$$

Read the third line aloud: the candidate sees the input and the old state scaled entry by entry by $r$, each through its own matrix. In numpy:

```python
cand = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
```

The order matters. Take $h = [1, 0]$, $r = [1, 0]$, and a swap matrix $U_h$ that exchanges the two entries. Then $(r \odot h) U_h = [1, 0] U_h = [0, 1]$, but $r \odot (h U_h) = [1, 0] \odot [0, 1] = [0, 0]$. Both versions have the same shape, so only the numbers reveal the mistake. Figure 2 puts the whole cell next to the LSTM.

```figure
<svg viewBox="0 0 620 345" role="img" aria-label="Two cell layouts side by side. The LSTM has a forget gate, an input gate, an output gate, a candidate, a cell, and a hidden value. The GRU merges the forget and input gates into one update gate, adds a reset gate, and has only a hidden state.">
<defs><marker id="c092-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20" y="26" class="fig-bold">LSTM: four weight sets</text>
<text x="350" y="26" class="fig-bold">GRU: three weight sets</text>
<rect x="20" y="50" width="200" height="38" rx="6" class="fill-soft-b stroke-b"/>
<text x="120" y="74" text-anchor="middle">forget gate f</text>
<rect x="20" y="98" width="200" height="38" rx="6" class="fill-soft-b stroke-b"/>
<text x="120" y="122" text-anchor="middle">input gate i</text>
<rect x="20" y="156" width="200" height="38" rx="6" class="fill-soft-a stroke-a"/>
<text x="120" y="180" text-anchor="middle">output gate o</text>
<rect x="20" y="204" width="200" height="38" rx="6" class="fill-soft-c stroke-c"/>
<text x="120" y="228" text-anchor="middle">candidate g (tanh)</text>
<rect x="20" y="266" width="90" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="65" y="291" text-anchor="middle">cell c</text>
<rect x="130" y="266" width="90" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="175" y="291" text-anchor="middle">hidden h</text>
<polyline points="226,52 234,52 234,134 226,134" fill="none" class="stroke-b"/>
<line x1="234" y1="93" x2="340" y2="93" class="stroke-b" marker-end="url(#c092-arrow)"/>
<text x="238" y="82" class="fig-small">merged</text>
<rect x="350" y="50" width="220" height="86" rx="6" class="fill-soft-b stroke-b"/>
<text x="460" y="98" text-anchor="middle">update gate z</text>
<rect x="350" y="156" width="220" height="38" rx="6" class="fill-soft-d stroke-d"/>
<text x="460" y="180" text-anchor="middle">reset gate r</text>
<rect x="350" y="204" width="220" height="38" rx="6" class="fill-soft-c stroke-c"/>
<text x="460" y="228" text-anchor="middle">candidate (tanh)</text>
<rect x="350" y="266" width="220" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="460" y="291" text-anchor="middle">hidden h (the only state)</text>
<text x="350" y="66"></text>
<text x="460" y="118" text-anchor="middle" class="fig-small">keeps 1 - z, writes z</text>
<text x="20" y="330" class="fig-small">No output gate in a GRU: the state is shown as it is.</text>
</svg>
Figure 2. The two cells with matching colours for matching roles. The forget and input gates, in orange, merge into the single update gate z. The GRU adds a reset gate, in purple, and drops both the separate cell and the output gate. Notice that the candidate, in green, is the same idea in both cells.
```

## What each cell costs

A weight set holds three things: `W` of shape $(d_x, d_h)$, `U` of shape $(d_h, d_h)$, and `b` of shape $(d_h,)$. With input width $d_x$ and hidden width $d_h$, one set holds $d_x d_h + d_h^2 + d_h$ numbers. Take both widths as 2: $4 + 4 + 2 = 10$. A plain recurrent cell has one such set. A GRU has three, for $z$, $r$, and the candidate, so 30. An LSTM has four, for $f$, $i$, $o$, and the candidate, so 40. The factors 3 and 4 come from one weight set per gate, with each gate reading the same $x$ and $h$:

$$
\text{GRU: } 3\left(d_x d_h + d_h^2 + d_h\right), \qquad
\text{LSTM: } 4\left(d_x d_h + d_h^2 + d_h\right)
$$

Read it aloud: count one weight set, then multiply by the number of gates plus candidates. At both widths 512, one set holds $262{,}144 + 262{,}144 + 512 = 524{,}800$ numbers. A plain RNN holds 524,800, a GRU holds 1,574,400, and an LSTM holds 2,099,200. Gating costs three or four times as much as a plain cell, and the GRU saves exactly a quarter of the LSTM. Figure 3 shows these counts and how they grow.

```figure
<svg viewBox="0 0 650 285" role="img" aria-label="Left, bars of parameter counts at width 512: plain RNN 0.52 million, GRU 1.57 million, LSTM 2.10 million. Right, lines of parameter counts for hidden sizes 64 to 1024, growing with the square of the size.">
<text x="20" y="24" class="fig-bold">Hidden and input size both 512</text>
<rect x="40" y="202.5" width="64" height="47.5" class="fill-a"/>
<text x="72" y="194.5" text-anchor="middle" class="fig-small fig-bold">0.52M</text>
<text x="72" y="268" text-anchor="middle" class="fig-small">plain RNN</text>
<rect x="130" y="107.5" width="64" height="142.5" class="fill-c"/>
<text x="162" y="99.5" text-anchor="middle" class="fig-small fig-bold">1.57M</text>
<text x="162" y="268" text-anchor="middle" class="fig-small">GRU</text>
<rect x="220" y="60" width="64" height="190" class="fill-b"/>
<text x="252" y="52" text-anchor="middle" class="fig-small fig-bold">2.10M</text>
<text x="252" y="268" text-anchor="middle" class="fig-small">LSTM</text>
<line x1="30" y1="250" x2="310" y2="250" class="stroke-muted"/>
<text x="340" y="24" class="fig-bold">Counts as the width grows (d_x = d_h)</text>
<line x1="380" y1="250" x2="570" y2="250" class="stroke-grid"/>
<text x="372" y="254" text-anchor="end" class="fig-small">0M</text>
<line x1="380" y1="202.4" x2="570" y2="202.4" class="stroke-grid"/>
<text x="372" y="206.4" text-anchor="end" class="fig-small">2M</text>
<line x1="380" y1="154.8" x2="570" y2="154.8" class="stroke-grid"/>
<text x="372" y="158.8" text-anchor="end" class="fig-small">4M</text>
<line x1="380" y1="107.1" x2="570" y2="107.1" class="stroke-grid"/>
<text x="372" y="111.1" text-anchor="end" class="fig-small">6M</text>
<line x1="380" y1="59.5" x2="570" y2="59.5" class="stroke-grid"/>
<text x="372" y="63.5" text-anchor="end" class="fig-small">8M</text>
<text x="380" y="268" text-anchor="middle" class="fig-small">64</text>
<text x="427.5" y="268" text-anchor="middle" class="fig-small">128</text>
<text x="475" y="268" text-anchor="middle" class="fig-small">256</text>
<text x="522.5" y="268" text-anchor="middle" class="fig-small">512</text>
<text x="570" y="268" text-anchor="middle" class="fig-small">1024</text>
<text x="475" y="286" text-anchor="middle" class="fig-small">hidden size</text>
<polyline points="380,249.2 427.5,246.9 475,237.5 522.5,200 570,50.2" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="380" cy="249.2" r="3.5" class="fill-b"/>
<circle cx="427.5" cy="246.9" r="3.5" class="fill-b"/>
<circle cx="475" cy="237.5" r="3.5" class="fill-b"/>
<circle cx="522.5" cy="200" r="3.5" class="fill-b"/>
<circle cx="570" cy="50.2" r="3.5" class="fill-b"/>
<text x="578" y="54.2" class="fig-small">LSTM 8.4M</text>
<polyline points="380,249.4 427.5,247.7 475,240.6 522.5,212.5 570,100.1" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="380" cy="249.4" r="3.5" class="fill-c"/>
<circle cx="427.5" cy="247.7" r="3.5" class="fill-c"/>
<circle cx="475" cy="240.6" r="3.5" class="fill-c"/>
<circle cx="522.5" cy="212.5" r="3.5" class="fill-c"/>
<circle cx="570" cy="100.1" r="3.5" class="fill-c"/>
<text x="578" y="104.1" class="fig-small">GRU 6.3M</text>
<polyline points="380,249.8 427.5,249.2 475,246.9 522.5,237.5 570,200" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="380" cy="249.8" r="3.5" class="fill-a"/>
<circle cx="427.5" cy="249.2" r="3.5" class="fill-a"/>
<circle cx="475" cy="246.9" r="3.5" class="fill-a"/>
<circle cx="522.5" cy="237.5" r="3.5" class="fill-a"/>
<circle cx="570" cy="200" r="3.5" class="fill-a"/>
<text x="578" y="204" class="fig-small">RNN 2.1M</text>
</svg>
Figure 3. Left, parameter counts at width 512 for the three cells. Right, the same counts as the width doubles from 64 to 1024. Notice that each doubling multiplies the counts by almost 4, because the square of the width dominates, so width is expensive.
```

## Cost and scaling

Each weight is used once per step per example, so a step costs about one multiply and one add for every weight in a matrix. Counting 2 floating-point operations per multiply-add, a GRU step costs about $6 (d_x d_h + d_h^2)$ operations per example and an LSTM step about $8 (d_x d_h + d_h^2)$, because each of the 3 or 4 weight sets contributes $d_x d_h + d_h^2$ multiply-adds. For the backward pass you store more than the outputs. An LSTM keeps $f$, $i$, $o$, $g$, $c_{\text{new}}$, and $\tanh(c_{\text{new}})$, six arrays of shape $(n, d_h)$. A GRU keeps $z$, $r$, the candidate, and $r \odot h$, four such arrays. Both grow linearly with the sequence length, and step $t$ waits for step $t - 1$.

> [!KEY]
> Can we keep the belt's benefit with a simpler cell? Yes. A GRU keeps one state and one update gate, so the shares $1 - z$ and $z$ blend old and new and the belt survives. A reset gate hides the past from the candidate when it would mislead. The cell needs three weight sets instead of four, which saves a quarter of the LSTM's numbers, at the price of tying forgetting to writing.

> [!RECAP]
> - The GRU update is $(1 - z) \odot h + z \odot \tilde{h}$, a blend whose two shares add to 1.
> - The reset gate scales the old state before the candidate's matrix sees it, so you write `(r * h) @ Uh`.
> - One weight set holds $d_x d_h + d_h^2 + d_h$ numbers, and a plain RNN has one, a GRU three, and an LSTM four.
> - At width 512 the counts are 524,800, 1,574,400, and 2,099,200, and they grow with $d_h^2$.
> - Tying forget to write saves a quarter of the LSTM's numbers but removes independent control.

## Build it step by step

The checks test three functions in this order.

1. **`gru_update_blend(h, candidate, z)`.** Takes the old state, a candidate, and the update gate, all of shape `(n, d_h)`, and returns the blend in the same shape. Write `(1 - z) * h + z * candidate`. The most common mistake is swapping `z` and `1 - z`, which makes $z = 1$ keep the old state.
2. **`gru_cell_forward(x, h, params)`.** Takes the input `(n, d_x)`, the old state `(n, d_h)`, and the nine parameter arrays, and returns the new state. Compute `z` and `r` with a sigmoid, the candidate with `(r * h) @ Uh`, and finish with `gru_update_blend`. The most common mistake is `r * (h @ Uh)`.
3. **`parameter_count(kind, input_size, hidden_size)`.** Takes `"rnn"`, `"gru"`, or `"lstm"` and returns an `int`. Compute one weight set, then multiply by 1, 3, or 4, and raise `ValueError` for any other kind. The most common mistake is one bias for the whole cell instead of one per weight set.

## Where this breaks

Gating slows the fade and does not remove it. As the chapter "Why a recurrent net forgets, and the belt that fixes it" showed with a gate of 0.99, very long memories stay hard for both cells. Neither cell removes the sequential chain, because step $t$ needs the state from step $t - 1$, so long sequences cannot be computed in parallel. The chapter "Why one vector is not enough: attention as a soft lookup" takes up both problems by letting a step read any earlier state directly. Gradients through the previous state can still blow up, so "Taming long sequences: truncation and clipping" still applies to a GRU.

## The exercise

You build `gru_update_blend`, `gru_cell_forward`, and `parameter_count`. The checks confirm that the blend returns the old state at $z = 0$ and the candidate at $z = 1$, that the worked step gives 0.7546, that the reset gate acts before the matrix on a swap-matrix example, and that the counts are 10, 30, and 40 at width 2 and 524,800, 1,574,400, and 2,099,200 at width 512.
