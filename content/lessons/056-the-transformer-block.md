---
title: "The transformer block: SwiGLU, residuals, and layer norm"
slug: 056-the-transformer-block
part: "Part 6 — Transformers"
summary: Attention mixes tokens, a gated feed-forward network computes on each one, and residuals with layer norm let dozens of blocks stack.
minutes: 80
gpu: false
objectives:
  - Explain the division of labour in a transformer layer, where attention moves information between tokens and the feed-forward network computes on each token alone.
  - Compute SiLU and a SwiGLU output by hand, and choose a gated hidden width that matches the weight count of a plain network.
  - Show with numbers why a plain stack of layers loses its gradient, and why x + f(x) keeps a direct road back.
  - Compute layer norm by hand, and explain pre-norm against post-norm.
  - Assemble a block from layer norm, rotary attention, SwiGLU, and two residual adds, and count its parameters.
prereqs:
  - 030-multilayer-perceptrons
  - 035-normalisation-layers
  - 042-residual-networks
  - 052-scaled-attention-and-heads
  - 055-position-encodings
exercise: 056-the-transformer-block
---

# The transformer block: SwiGLU, residuals, and layer norm

After attention, the word "bit" in "dog bit man" holds a blend of what "dog" and "man" had to say. What happens to that blend? The token still has to decide that this is a biting event, that the subject is a dog, and that the object is a man. Attention only gathers and averages. A stack of nothing but attention layers would keep passing averaged values along and never compute a feature from a token's own vector. So each token needs a small network of its own.

A language model then stacks attention and that network 32 or more times, and a deep stack of ordinary layers fails in a specific way. The gradient, which is the signal that tells the first layers how to change, shrinks as it travels back through each layer. Later in this chapter, a 32-layer stack loses a factor of tens of billions on the way back. What small change in the wiring lets the same layers stack 32 deep and still train?

This chapter builds the whole block in three steps: a gated per-token network called SwiGLU, a residual stream that gives the gradient a direct road back, and layer norm that keeps the stream's scale steady. It ends by assembling the pieces and counting the parameters. Figure 1 previews the finished block.

```figure
<svg viewBox="0 0 600 384" role="img" aria-label="Flow of one pre-norm block: the stream runs straight down, and two side loops, each a layer norm then a sublayer, add their result back onto the stream">
<defs><marker id="c046-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c046-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="155.0" y="14.0" width="150" height="28" rx="6" class="stroke-ink fill-paper"/>
<text x="230" y="32" text-anchor="middle" class="fig-bold fig-small">x (token vectors)</text>
<line x1="230" y1="42" x2="230" y2="148" class="stroke-b" stroke-width="4"/>
<line x1="230" y1="182" x2="230" y2="278" class="stroke-b" stroke-width="4"/>
<line x1="230" y1="312" x2="230" y2="334" class="stroke-b" stroke-width="4" marker-end="url(#c046-arrow-b)"/>
<line x1="230" y1="95" x2="280" y2="95" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<rect x="280.0" y="80.0" width="80" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="320" y="99" text-anchor="middle" class="fig-small">layer norm 1</text>
<line x1="360" y1="95" x2="388" y2="95" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<rect x="388.0" y="80.0" width="124" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="450" y="99" text-anchor="middle" class="fig-small">attention + RoPE</text>
<polyline points="512,95 550,95 550,165 249,165" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<circle cx="230" cy="165" r="17" class="stroke-ink fill-paper"/><text x="230" y="170" text-anchor="middle" class="fig-bold">+</text>
<text x="216" y="201" text-anchor="end" class="fig-bold">x′</text>
<line x1="230" y1="225" x2="280" y2="225" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<rect x="280.0" y="210.0" width="80" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="320" y="229" text-anchor="middle" class="fig-small">layer norm 2</text>
<line x1="360" y1="225" x2="388" y2="225" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<rect x="388.0" y="210.0" width="124" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="450" y="229" text-anchor="middle" class="fig-small">SwiGLU</text>
<polyline points="512,225 550,225 550,295 249,295" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<circle cx="230" cy="295" r="17" class="stroke-ink fill-paper"/><text x="230" y="300" text-anchor="middle" class="fig-bold">+</text>
<text x="216" y="81" text-anchor="end" class="fig-small">the raw stream skips the norm</text>
<text x="216" y="334" text-anchor="end" class="fig-bold">x″ = block output</text>
<text x="300" y="372" text-anchor="middle" class="fig-small">Orange: the stream. Each side loop reads a normed copy and writes a correction back with +.</text>
</svg>
Figure 1. One pre-norm block. Notice that the orange stream passes by the layer norms on its way down: each norm sits on a side loop, and only the correction travels back through a plus.
```

## What you already know

You already have these ideas:

- Attention scores each query against each key, divides by $\sqrt{d_{\text{head}}}$, adds a mask, and applies a softmax over the keys, with several narrow heads running in parallel. Rotary embeddings turn queries and keys by their positions, and the values stay untouched (see [position encodings](/lessons/055-position-encodings)).
- A layer multiplies a vector by a weight matrix, and a nonlinear function between layers lets a network compute more than one matrix could. ReLU sets negative numbers to zero.
- The sigmoid, $\sigma(z) = 1/(1 + e^{-z})$, squashes any number into the range 0 to 1.
- A residual connection returns $x + f(x)$.
- Layer norm recentres a row of numbers and rescales it, and the gradient reaching early layers is a product of one slope per layer.

## Attention talks, the feed-forward network thinks

A transformer layer splits the work in two. Attention is communication: each token reads from the other tokens. The feed-forward network is computation: each token takes what it collected and transforms it, alone. The same weights are applied to every token's row, and no row sees another, so this step never mixes tokens.

The plain version has two matrices. $W_1$ has shape $(d, d_{\text{ff}})$ and expands a token from $d$ numbers to $d_{\text{ff}}$ numbers, usually about $4d$. $W_2$ has shape $(d_{\text{ff}}, d)$ and compresses it back. In between sits ReLU:

$$
\operatorname{FFN}(x) = \operatorname{ReLU}(x W_1)\, W_2
$$

Read it aloud: expand the token, switch off the negative entries, and compress back. Each entry of $xW_1$ is a hidden unit, one weighted sum of the token's numbers. The weakness shows when you look at one unit. It computes $\operatorname{ReLU}(w \cdot x)$, so one number has to do two jobs: say how strongly a feature is present, and say whether to pass anything on. A gate separates the two jobs.

## SiLU and the gate

First swap the ReLU for a smoother bend. ReLU has a corner at 0, and it throws away every negative input. SiLU keeps a small negative response and has no corner:

$$
\operatorname{silu}(z) = z \cdot \sigma(z)
$$

Read it aloud: multiply the number by the sigmoid of the number, so the sigmoid acts as a volume knob that runs from 0 to 1. For $z = 1$ the sigmoid is $0.7311$, so $\operatorname{silu}(1) = 0.7311$. For $z = -1$ it is $0.2689$, so $\operatorname{silu}(-1) = -0.2689$. For a large positive $z$ the knob is fully open and $\operatorname{silu}(z) \approx z$. For a large negative $z$ it is shut and $\operatorname{silu}(z) \approx 0$.

A gated network uses two projections of the same token instead of one. The gate branch computes $xW_{\text{gate}}$ and passes it through SiLU. The up branch computes $xW_{\text{up}}$ and applies no bend. The hidden vector is their entry-by-entry product, and a third matrix $W_{\text{down}}$ compresses it back:

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}}
$$

Read it aloud: one branch decides how much of each hidden unit passes, the other branch supplies what passes, and the product goes on to $W_{\text{down}}$. $W_{\text{gate}}$ and $W_{\text{up}}$ both have shape $(d, d_h)$, and $W_{\text{down}}$ has shape $(d_h, d)$. The letters $d_h$ name the hidden width of this gated network. SiLU goes on the gate branch only.

Work one token by hand, with $d = 2$ and a hidden width of 2. Take $x = [1, 2]$ and these matrices, written as rows:

$$
W_{\text{gate}} = \begin{bmatrix} -3 & 1 \\ 0 & 1 \end{bmatrix}, \quad
W_{\text{up}} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W_{\text{down}} = \begin{bmatrix} 1 & 0 \\ 1 & -1 \end{bmatrix}
$$

The gate branch gives $xW_{\text{gate}} = [-3, 3]$. SiLU turns that into $[-0.142, 2.858]$, because $\operatorname{silu}(-3) = -3 \times 0.0474 = -0.142$ and $\operatorname{silu}(3) = 3 \times 0.9526 = 2.858$. The gate has nearly shut unit 1 and opened unit 2. The up branch gives $xW_{\text{up}} = [1, 2]$. Multiply entry by entry to get the hidden vector $[-0.142, 5.715]$. Finally $W_{\text{down}}$ gives $[-0.142 + 5.715,\ -5.715] = [5.573, -5.715]$. Figure 2 follows these numbers through the network.

```figure
<svg viewBox="0 0 600 372" role="img" aria-label="Data flow through a gated feed-forward network for the token [1, 2]: a gate branch through SiLU and an up branch multiply entry by entry, then a down projection gives the output">
<defs><marker id="c046-ffn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="205.0" y="20.0" width="190" height="32" rx="6" class="stroke-ink fill-paper"/>
<text x="300" y="40" text-anchor="middle" class="fig-bold fig-small">token x = [1, 2]</text>
<rect x="55.0" y="92.0" width="230" height="40" rx="6" class="stroke-ink fill-paper"/>
<text x="170" y="108" text-anchor="middle" class="fig-bold fig-small">gate: x W_gate</text>
<text x="170" y="124" text-anchor="middle" class="fig-small">[−3, 3]</text>
<rect x="315.0" y="92.0" width="230" height="40" rx="6" class="stroke-ink fill-paper"/>
<text x="430" y="108" text-anchor="middle" class="fig-bold fig-small">up: x W_up</text>
<text x="430" y="124" text-anchor="middle" class="fig-small">[1, 2]</text>
<line x1="270" y1="52" x2="200" y2="90" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
<line x1="330" y1="52" x2="400" y2="90" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
<rect x="55.0" y="170.0" width="230" height="40" rx="6" class="stroke-ink fill-paper"/>
<text x="170" y="186" text-anchor="middle" class="fig-bold fig-small">SiLU of the gate</text>
<text x="170" y="202" text-anchor="middle" class="fig-small">[−0.142, 2.858]</text>
<line x1="170" y1="132" x2="170" y2="168" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
<text x="292" y="186" class="fig-small">unit 1: nearly shut</text><text x="292" y="202" class="fig-small">unit 2: open</text>
<circle cx="300" cy="252" r="15" class="stroke-ink fill-paper"/><text x="300" y="258" text-anchor="middle" class="fig-bold">×</text>
<polyline points="170,210 170,252 283,252" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
<polyline points="430,132 430,252 317,252" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
<text x="320" y="282" class="fig-small">hidden = [−0.142, 5.715]</text>
<line x1="300" y1="267" x2="300" y2="296" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
<rect x="215.0" y="299.0" width="170" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="300" y="318" text-anchor="middle" class="fig-bold fig-small">× W_down</text>
<text x="300" y="362" text-anchor="middle" class="fig-bold fig-small">output = [5.573, −5.715]</text>
<line x1="300" y1="329" x2="300" y2="346" class="stroke-ink" marker-end="url(#c046-ffn-arrow)"/>
</svg>
Figure 2. The token [1, 2] through a SwiGLU network with two hidden units. Notice that the gate branch nearly shuts unit 1 (−0.142) and opens unit 2 (2.858), so unit 2 passes the up value 2 magnified 2.858 times, while unit 1 passes its up value 1 scaled down to −0.142.
```

The gate depends on the token, not only on the weights. Feed the same network $x = [-1, 2]$. The gate branch gives $[3, 1]$, and SiLU turns that into $[2.858, 0.731]$. The up branch gives $[-1, 2]$, so the hidden vector is $[-2.858, 1.462]$. Now unit 1 is wide open and unit 2 is partly open. Different tokens switch different hidden units on, and the up branch decides what flows through each unit that is on. Experiments that compare gated and plain feed-forward networks of the same size find that the gated ones reach a lower loss. The "how much" and "what" picture is a way to think about that result, not a proof.

In numpy, with `z * sigmoid(z)` written as a stable expression:

```python
gate = x @ W_gate
silu = gate * 0.5 * (1.0 + np.tanh(gate / 2.0))   # gate * sigmoid(gate), safe for large negative gate
hidden = silu * (x @ W_up)
out = hidden @ W_down
```

The expression `0.5 * (1 + tanh(z / 2))` equals the sigmoid, and it never overflows. The more obvious `1 / (1 + np.exp(-z))` overflows for a gate value near $-800$, because $e^{800}$ is larger than a float can hold.

## Same budget, three matrices

A plain network holds two matrices of $d \times d_{\text{ff}}$ weights, so $2\,d\,d_{\text{ff}}$ in all. SwiGLU holds three of $d \times d_h$, so $3\,d\,d_h$. At the same width the gated network would have 50 percent more weights, and a comparison would mix "gating helps" with "a bigger network helps". To compare fairly, set the two counts equal:

$$
3\,d\,d_h = 2\,d\,d_{\text{ff}} \quad\Longrightarrow\quad d_h = \tfrac{2}{3}\, d_{\text{ff}}
$$

Read it aloud: with three matrices instead of two, each matrix gets two-thirds of the width. A check with $d = 2$ and $d_{\text{ff}} = 6$: the plain network holds $2 \times 2 \times 6 = 24$ weights and the gated one holds $3 \times 2 \times 4 = 24$. When $d_{\text{ff}}$ is not a multiple of 3, you round. At $d = 64$ and $d_{\text{ff}} = 256$, two-thirds is 170.67, so $d_h = 171$. The plain network holds $32768$ weights and the gated one holds $3 \times 64 \times 171 = 32832$, a little over 0.2 percent more. This chapter leaves out bias vectors, which most gated networks do not use.

> [!KEY]
> Attention moves information between tokens, and the feed-forward network computes on each token alone. SwiGLU computes with a gate: one branch decides how much of each hidden unit passes and the other supplies what passes, and a hidden width of two-thirds of $d_{\text{ff}}$ keeps the weight count equal to a plain network.

## Why a plain deep stack loses its gradient

The gradient at layer 1 is a product, one slope for each layer it passes back through. Take three layers that each have slope 0.1. The product is $0.1^3 = 0.001$. At 32 layers it is $0.1^{32}$, far too small to move anything.

Now let each layer return $x + f(x)$ instead of $f(x)$. The slope of $x + f(x)$ is $1 + f'(x)$. With the same slope 0.1 in each of three layers, the product is $1.1^3 = 1.331$, and it expands as $1 + 0.3 + 0.03 + 0.001$. The first term, 1, is the road that skips all three layers. It is always there, whatever the layers do, so the gradient always has a direct way back.

Figure 3 checks this on a real stack: 32 layers of width 16, each computing $\tanh(Wh)$ with random weights of standard deviation $0.5/\sqrt{16}$, run once as a plain stack and once with the $x +$ added. The plain stack passes only 0.0178 of the gradient back through 4 layers, $3 \times 10^{-6}$ through 16, and $2.6 \times 10^{-11}$ through 32. The residual stack passes 1.155, 2.648, and 6.445. The residual gradient grows by a factor of six, not shrinks by a factor of tens of billions. This is one random draw of the weights, so the exact numbers vary, but the gap of ten orders of magnitude does not. The growth is the cost of letting corrections pile up, and layer norm keeps it from running away.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Gradient size against the number of layers it has passed back through, on a log scale: the plain stack collapses toward zero, the residual stack stays near one">
<line x1="70" y1="250.0" x2="560" y2="250.0" class="stroke-grid"/>
<text x="62" y="254.0" text-anchor="end" class="fig-small">1e−12</text>
<line x1="70" y1="201.5" x2="560" y2="201.5" class="stroke-grid"/>
<text x="62" y="205.5" text-anchor="end" class="fig-small">1e−9</text>
<line x1="70" y1="153.1" x2="560" y2="153.1" class="stroke-grid"/>
<text x="62" y="157.1" text-anchor="end" class="fig-small">1e−6</text>
<line x1="70" y1="104.6" x2="560" y2="104.6" class="stroke-grid"/>
<text x="62" y="108.6" text-anchor="end" class="fig-small">1e−3</text>
<line x1="70" y1="56.2" x2="560" y2="56.2" class="stroke-muted dashed"/>
<text x="62" y="60.2" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="268" text-anchor="middle" class="fig-small">0</text>
<text x="131.2" y="268" text-anchor="middle" class="fig-small">4</text>
<text x="192.5" y="268" text-anchor="middle" class="fig-small">8</text>
<text x="315.0" y="268" text-anchor="middle" class="fig-small">16</text>
<text x="437.5" y="268" text-anchor="middle" class="fig-small">24</text>
<text x="560.0" y="268" text-anchor="middle" class="fig-small">32</text>
<text x="315.0" y="288" text-anchor="middle" class="fig-small">layers the gradient has passed back through</text>
<text x="14" y="26" class="fig-small">gradient size</text>
<polyline points="70.0,56.2 85.3,65.7 100.6,70.9 115.9,78.4 131.2,84.4 146.6,91.0 161.9,95.4 177.2,99.3 192.5,103.3 207.8,109.2 223.1,115.0 238.4,122.1 253.8,126.0 269.1,130.6 284.4,134.7 299.7,141.7 315.0,145.4 330.3,149.4 345.6,155.7 360.9,163.2 376.2,167.9 391.6,172.0 406.9,176.4 422.2,181.7 437.5,187.2 452.8,191.5 468.1,196.3 483.4,200.6 498.8,205.4 514.1,210.6 529.4,215.7 544.7,221.7 560.0,227.3" class="stroke-a"/>
<polyline points="70.0,56.2 85.3,55.6 100.6,55.6 115.9,55.6 131.2,55.1 146.6,54.7 161.9,54.4 177.2,54.3 192.5,52.9 207.8,53.0 223.1,52.3 238.4,51.2 253.8,50.7 269.1,50.1 284.4,49.3 299.7,50.1 315.0,49.3 330.3,48.8 345.6,48.1 360.9,47.5 376.2,46.2 391.6,45.9 406.9,45.0 422.2,45.3 437.5,44.7 452.8,44.3 468.1,44.2 483.4,44.4 498.8,44.4 514.1,44.2 529.4,43.4 544.7,43.7 560.0,43.1" class="stroke-b"/>
<text x="554.0" y="249.3" text-anchor="end" class="fig-small">plain stack: 2.6e-11</text>
<text x="554.0" y="33.1" text-anchor="end" class="fig-small">with x + f(x): 6.4</text>
<text x="300" y="316" text-anchor="middle" class="fig-small">32 layers of width 16, f(h) = tanh(W h) with random W. Same weights in both stacks.</text>
</svg>
Figure 3. Gradient size against the number of layers it has passed back through, for the same 32 layers run two ways. Notice the log scale: the plain stack loses more than ten orders of magnitude, while the stack with x + f(x) stays within one order of magnitude of 1.
```

Think of the vector for each token as a draft that every layer reads and edits. A layer reads the draft, works out a small correction, and writes it by adding it on. The running sum is called the residual stream. A block whose layers have nothing to say adds zero and returns its input exactly, so each layer only has to learn a change.

## Layer norm keeps the scale steady

Every correction added to the stream changes its size, and over many layers the size drifts. The attention scores grow with the size of the vectors, so large vectors make the softmax too sharp and small ones make it too flat. Layer norm resets each token to a steady scale before a layer reads it.

Work one token by hand: $[1, 2, 3, 6]$. The mean is $(1 + 2 + 3 + 6)/4 = 3$. The gaps from the mean are $-2, -1, 0, 3$, and their squares are $4, 1, 0, 9$. The average of the squares, the variance, is $14/4 = 3.5$. Add a tiny $\epsilon = 10^{-5}$ and take the square root to get $1.8708$. Dividing each gap by that gives $[-1.0690, -0.5345, 0, 1.6036]$. The ten-times-larger token $[10, 20, 30, 60]$ gives exactly the same row, so the output no longer depends on the token's overall size. The formula is

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot g + b
$$

Read it aloud: subtract the token's mean $\mu$, divide by its standard deviation $\sigma$ (made safe by $\epsilon$), then scale by a learned gain $g$ and shift by a learned bias $b$. Figure 4 shows three tokens before and after.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Three token vectors as bars before layer norm, with very different scales, and after layer norm, where tokens A and B are identical">
<text x="160.0" y="26" text-anchor="middle" class="fig-bold">Before layer norm</text>
<line x1="50" y1="223.3" x2="270" y2="223.3" class="stroke-muted"/>
<text x="44" y="227.3" text-anchor="end" class="fig-small">0</text>
<line x1="50" y1="170.0" x2="270" y2="170.0" class="stroke-grid"/>
<text x="44" y="174.0" text-anchor="end" class="fig-small">20</text>
<line x1="50" y1="116.7" x2="270" y2="116.7" class="stroke-grid"/>
<text x="44" y="120.7" text-anchor="end" class="fig-small">40</text>
<line x1="50" y1="63.3" x2="270" y2="63.3" class="stroke-grid"/>
<text x="44" y="67.3" text-anchor="end" class="fig-small">60</text>
<rect x="57.3" y="220.7" width="13.3" height="2.7" class="fill-a"/>
<rect x="73.3" y="218.0" width="13.3" height="5.3" class="fill-a"/>
<rect x="89.3" y="215.3" width="13.3" height="8.0" class="fill-a"/>
<rect x="105.3" y="207.3" width="13.3" height="16.0" class="fill-a"/>
<text x="86.7" y="268" text-anchor="middle" class="fig-small">A</text>
<rect x="130.7" y="196.7" width="13.3" height="26.7" class="fill-b"/>
<rect x="146.7" y="170.0" width="13.3" height="53.3" class="fill-b"/>
<rect x="162.7" y="143.3" width="13.3" height="80.0" class="fill-b"/>
<rect x="178.7" y="63.3" width="13.3" height="160.0" class="fill-b"/>
<text x="160.0" y="268" text-anchor="middle" class="fig-small">B = 10 A</text>
<rect x="204.0" y="223.3" width="13.3" height="8.0" class="fill-c"/>
<rect x="220.0" y="212.7" width="13.3" height="10.7" class="fill-c"/>
<rect x="236.0" y="223.3" width="13.3" height="2.7" class="fill-c"/>
<rect x="252.0" y="202.0" width="13.3" height="21.3" class="fill-c"/>
<text x="233.3" y="268" text-anchor="middle" class="fig-small">C</text>
<text x="450.0" y="26" text-anchor="middle" class="fig-bold">After layer norm</text>
<line x1="340" y1="250.0" x2="560" y2="250.0" class="stroke-grid"/>
<text x="334" y="254.0" text-anchor="end" class="fig-small">-2</text>
<line x1="340" y1="200.0" x2="560" y2="200.0" class="stroke-grid"/>
<text x="334" y="204.0" text-anchor="end" class="fig-small">-1</text>
<line x1="340" y1="150.0" x2="560" y2="150.0" class="stroke-muted"/>
<text x="334" y="154.0" text-anchor="end" class="fig-small">0</text>
<line x1="340" y1="100.0" x2="560" y2="100.0" class="stroke-grid"/>
<text x="334" y="104.0" text-anchor="end" class="fig-small">1</text>
<line x1="340" y1="50.0" x2="560" y2="50.0" class="stroke-grid"/>
<text x="334" y="54.0" text-anchor="end" class="fig-small">2</text>
<rect x="347.3" y="150.0" width="13.3" height="53.5" class="fill-a"/>
<rect x="363.3" y="150.0" width="13.3" height="26.7" class="fill-a"/>
<rect x="379.3" y="150.0" width="13.3" height="0.6" class="fill-a"/>
<rect x="395.3" y="69.8" width="13.3" height="80.2" class="fill-a"/>
<text x="376.7" y="268" text-anchor="middle" class="fig-small">A</text>
<rect x="420.7" y="150.0" width="13.3" height="53.5" class="fill-b"/>
<rect x="436.7" y="150.0" width="13.3" height="26.7" class="fill-b"/>
<rect x="452.7" y="150.0" width="13.3" height="0.6" class="fill-b"/>
<rect x="468.7" y="69.8" width="13.3" height="80.2" class="fill-b"/>
<text x="450.0" y="268" text-anchor="middle" class="fig-small">B = 10 A</text>
<rect x="494.0" y="150.0" width="13.3" height="58.1" class="fill-c"/>
<rect x="510.0" y="126.8" width="13.3" height="23.2" class="fill-c"/>
<rect x="526.0" y="150.0" width="13.3" height="34.9" class="fill-c"/>
<rect x="542.0" y="80.3" width="13.3" height="69.7" class="fill-c"/>
<text x="523.3" y="268" text-anchor="middle" class="fig-small">C</text>
<text x="300" y="296" text-anchor="middle" class="fig-small">After the norm A and B match exactly: [−1.069, −0.535, 0, 1.604]. C keeps its own shape.</text>
<text x="300" y="316" text-anchor="middle" class="fig-small">Each token has four entries, drawn as four bars. Every token ends with mean 0 and variance 1.</text>
</svg>
Figure 4. Three tokens before and after layer norm. Notice that tokens A and B have very different sizes but become identical, and that every token ends at mean 0 and variance 1.
```

Three design choices are worth a reason. The statistics come from each token's own entries, not from the whole batch, because sequences have different lengths and a token's norm should not depend on its neighbours. The $\epsilon$ is there because a token whose entries are all equal has variance 0, and dividing by zero would break the model. The gain and bias are learned so that the network can choose a scale other than 1 if it needs one. With $g = 2$ and $b = 1$, the example row becomes $[-1.138, -0.069, 1, 4.207]$.

```python
mu = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)           # divides by d, numpy's default
out = (x - mu) / np.sqrt(var + eps) * gain + bias
```

## Pre-norm, and why the skip must stay raw

The norm can go in two places. Post-norm applies it after the add, $\operatorname{LN}(x + f(x))$, which puts the norm on the main road. Every block then rescales the stream, so the clean "1" road is interrupted at each block, and very deep post-norm stacks usually need a slow, careful learning-rate warm-up. Pre-norm applies it to the copy that the sublayer reads, $x + f(\operatorname{LN}(x))$, which puts the norm on the side road. The stream itself is never rescaled. The trade-off is that the stream's size still grows with depth, so pre-norm models add one last layer norm right before the output. This course uses pre-norm.

> [!WARNING]
> Add each sublayer's output to the raw stream `x`, not to `layer_norm(x)`. Adding to the normed copy puts the norm back on the main road and removes the skip.

## Assemble the block

The block has two sublayers, each reading a normed copy and adding its result to the raw stream:

$$
X' = X + \operatorname{Attn}\big(\operatorname{LN}_1(X)\big), \qquad
X'' = X' + \operatorname{SwiGLU}\big(\operatorname{LN}_2(X')\big)
$$

Attention is the part with the most steps, and each step has a reason:

1. Project the normed rows with $W_q$, $W_k$, and $W_v$, then reshape each result into `heads` heads of width $d_{\text{head}} = d/\text{heads}$, so each head asks its own question.
2. Turn the queries and keys, not the values, with the rotary tables. The tables have shape `(T, d_head)` and broadcast over the heads.
3. Compute the scores $q k^\top / \sqrt{d_{\text{head}}}$. A dot product of $d_{\text{head}}$ entries of ordinary size has a spread that grows like $\sqrt{d_{\text{head}}}$, and without the division the softmax saturates.
4. Add the mask before the softmax: 0 allows a position and $-10^9$ forbids it, so a forbidden position ends with weight 0.
5. Take the softmax over the key axis, multiply by $v$, merge the heads, and multiply by $W_o$ to mix the heads back into one width-$d$ vector.

```python
x1 = x + attention(layer_norm(x, g1, b1))
x2 = x1 + swiglu(layer_norm(x1, g2, b2), W_gate, W_up, W_down)
```

> [!KEY]
> A stack of 32 blocks trains because each sublayer writes a small correction onto a stream that flows straight through, so the gradient always has a direct road back. Layer norm resets the scale before each sublayer reads the stream, and it sits on the side road so that the skip stays clean. Attention mixes the tokens, and SwiGLU computes on each one.

## Where the parameters live

Count the weights. Attention holds four matrices of $d \times d$, which is $4d^2$. Splitting into more heads does not change this, because the width $d$ is divided among the heads, not enlarged. SwiGLU holds $3\,d\,d_h$. The two layer norms hold a gain and a bias of length $d$ each, which is $4d$:

$$
4d^2 + 3\,d\,d_h + 4d, \qquad d_h = \operatorname{round}\!\big(\tfrac{2}{3}\, d_{\text{ff}}\big)
$$

For $d = 64$ and $d_{\text{ff}} = 192$, $d_h = 128$ and the total is $16384 + 24576 + 256 = 41216$. With the usual $d_{\text{ff}} = 4d$, a block holds about $4d^2$ in attention and $8d^2$ in the feed-forward network, so the feed-forward part holds two-thirds of the weights. Each weight takes part in one multiply and one add per token, so a forward pass costs about $24d^2$ floating-point operations per token, plus the cost of comparing every query with every key, which grows with the sequence length. For $d = 4096$ and 32 blocks, the blocks hold about $12 \times 32 \times 4096^2 \approx 6.4$ billion weights.

## Where this breaks

**SiLU belongs on the gate only.** With SiLU on both branches, doubling the up branch no longer doubles the output. A network trained with one definition gives wrong answers with the other.

**The two widths are easy to confuse.** The hidden width $d_h$ of the feed-forward network is not the width $d_{\text{head}}$ of one attention head. Many models also round $d_h$ up to a multiple of 64 or 256 so matrix multiplies run efficiently, which drifts the count a little from the exact budget.

**The stream grows with depth, and long inputs are untested territory.** Nothing rescales the stream in a pre-norm block, so its entries get larger the deeper you go, and the final layer norm handles that. The block's weights do not depend on the sequence length, but the rotary angles of a very long input go beyond those seen in training, and attention costs more as the sequence grows.

> [!RECAP]
> Carry these facts into the exercise:
> - The feed-forward network works on each token's row alone. `silu(z) = z * sigmoid(z)`, written with a stable sigmoid. `swiglu(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down`, with SiLU on the gate only.
> - A plain network holds `2 * d * d_ff` weights. A gated one holds `3 * d * d_h` with `d_h = round(2 * d_ff / 3)`.
> - The slope of $x + f(x)$ is $1 + f'(x)$, so the gradient always has a direct road back.
> - Layer norm runs over the last axis with $\epsilon = 10^{-5}$ and a variance that divides by $d$, then applies the gain and the bias.
> - The block is `x1 = x + attention(LN1(x))`, then `x2 = x1 + swiglu(LN2(x1))`, with adds to the raw stream. Attention scales by $\sqrt{d_{\text{head}}}$, adds the mask before the softmax, and turns only queries and keys.
> - A block holds $4d^2 + 3\,d\,d_h + 4d$ parameters, whatever the number of heads.

## Build it step by step

The starter file provides `rope_tables` and `apply_rope`, solved, from the chapter on position encodings. You write six functions, in the order the checks test them.

1. **`silu(z)`** takes an array of any shape and returns `z * sigmoid(z)` with the same shape. Write the sigmoid as `0.5 * (1 + np.tanh(z / 2))`. The common mistake is `1 / (1 + np.exp(-z))`, which overflows for large negative `z`, and the check turns that overflow into a failure.
2. **`swiglu(x, W_gate, W_up, W_down)`** maps `(..., d)` to `(..., d)`. Compute the two products, apply your `silu` to the gate product, multiply entry by entry, and finish with `@ W_down`. The common mistake is SiLU on the up branch too.
3. **`ffn_parameter_count(d_model, d_ff, gated)`** returns an `int`. For a plain network it is `2 * d_model * d_ff`. For a gated one, compute `d_h = round(2 * d_ff / 3)` and return `3 * d_model * d_h`. The common mistake is to use `d_ff` as the gated width, which gives 50 percent too many.
4. **`layer_norm(x, gain, bias, eps=1e-5)`** takes `x` of shape `(..., d)` and returns the same shape. Use `axis=-1` and `keepdims=True` for the mean and the variance, divide by `np.sqrt(var + eps)`, then multiply by `gain` and add `bias`. The common mistake is to apply the gain and bias before dividing.
5. **`transformer_block_forward(x, params, mask)`** maps `(T, d)` to `(T, d)`. Norm, project, reshape with `.reshape(T, heads, d_head).transpose(1, 0, 2)`, turn `q` and `k`, scale, add `mask`, softmax over the last axis, multiply by `v`, merge, multiply by `Wo`, and add to the raw `x`. Then norm again, call `swiglu`, and add again. Read the base with `params.get("theta", 10000.0)`. Subtract the row maximum before `np.exp`, so the softmax cannot overflow. The common mistake is the residual on the normed input.
6. **`block_parameter_count(d_model, d_ff, heads)`** returns an `int`: `4 * d_model**2`, plus `3 * d_model * d_h` with `d_h = round(2 * d_ff / 3)`, plus `4 * d_model`. Raise `ValueError` when `heads` does not divide `d_model`. The common mistake is to forget the norm's gain and bias.

## The exercise

You write `silu`, `swiglu`, `ffn_parameter_count`, `layer_norm`, `transformer_block_forward`, and `block_parameter_count`. The checks confirm `silu(1) = 0.7311`, the identity $\operatorname{silu}(z) - \operatorname{silu}(-z) = z$, and that large negative inputs return a finite number near 0. They compare `swiglu` with the two-unit hand example and a direct computation, and confirm the counts 24 and 24 at $d = 2$ and $d_{\text{ff}} = 6$. They verify `layer_norm` on the $[1, 2, 3, 6]$ example, its zero mean and unit variance, and the order of the gain and bias. They check the block's shape, that zeroing `Wo` and `W_down` makes it an exact identity, and that under a causal mask changing a later token leaves earlier outputs unchanged. A loop-based reference catches a missing scale, a wrong softmax axis, or a residual on the normed copy. The last check confirms the count 41216 for $(64, 192, 8)$ and that changing `heads` leaves the count alone.
