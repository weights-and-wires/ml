---
title: "Stacking it up: residuals, layer norm, and the pre-norm block"
slug: 104-the-residual-block
part: "Part 6 — Transformers"
summary: Each sublayer writes a small correction onto a stream that flows straight through, with layer norm keeping its scale steady, so dozens of blocks can stack.
minutes: 45
gpu: false
objectives:
  - Show with numbers why a plain stack of layers loses its gradient, and why x + f(x) keeps a direct road back.
  - Compute layer norm by hand, and explain why it runs on each token and why its gain and bias are learned.
  - Explain pre-norm against post-norm, and why each sublayer adds to the raw stream and not to the normed copy.
  - Assemble a block from layer norm, rotary attention, SwiGLU, and two residual adds, and count its parameters.
prereqs:
  - 071-batch-norm-and-layer-norm
  - 083-gradient-through-deep-stacks
  - 099-heads-and-causal-mask
  - 102-rotary-embeddings
  - 103-gated-feed-forward
exercise: 104-the-residual-block
---

# Stacking it up: residuals, layer norm, and the pre-norm block

You have the two working parts of a transformer layer: attention, where tokens read from each other, and a gated feed-forward network, where each token computes alone. A language model stacks these pairs 32 or more times. Why not just stack them as they are? Because a deep stack of ordinary layers fails in a specific way. The gradient, which is the signal that tells the first layers how to change, shrinks as it travels back through each layer, and by the time it reaches layer 1 it is too small to teach anything. Later in this chapter, a 32-layer stack loses a factor of tens of billions on the way back.

What small change in the wiring lets the same layers stack 32 deep and still train? The answer has two parts. Let each layer add a small correction to a stream that flows straight through, instead of replacing it. And rescale each token before each layer reads it, so the stream's size cannot drift. This chapter builds both, assembles the full block, and counts its parameters. Figure 1 previews the finished block.

```figure
<svg viewBox="0 0 600 384" role="img" aria-label="Flow of one pre-norm block: the stream runs straight down, and two side loops, each a layer norm then a sublayer, add their result back onto the stream">
<defs><marker id="c104-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c104-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="155.0" y="14.0" width="150" height="28" rx="6" class="stroke-ink fill-paper"/>
<text x="230" y="32" text-anchor="middle" class="fig-bold fig-small">x (token vectors)</text>
<line x1="230" y1="42" x2="230" y2="148" class="stroke-b" stroke-width="4"/>
<line x1="230" y1="182" x2="230" y2="278" class="stroke-b" stroke-width="4"/>
<line x1="230" y1="312" x2="230" y2="334" class="stroke-b" stroke-width="4" marker-end="url(#c104-arrow-b)"/>
<line x1="230" y1="95" x2="280" y2="95" class="stroke-ink" marker-end="url(#c104-arrow)"/>
<rect x="280.0" y="80.0" width="80" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="320" y="99" text-anchor="middle" class="fig-small">layer norm 1</text>
<line x1="360" y1="95" x2="388" y2="95" class="stroke-ink" marker-end="url(#c104-arrow)"/>
<rect x="388.0" y="80.0" width="124" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="450" y="99" text-anchor="middle" class="fig-small">attention + RoPE</text>
<polyline points="512,95 550,95 550,165 249,165" class="stroke-ink" marker-end="url(#c104-arrow)"/>
<circle cx="230" cy="165" r="17" class="stroke-ink fill-paper"/><text x="230" y="170" text-anchor="middle" class="fig-bold">+</text>
<text x="216" y="201" text-anchor="end" class="fig-bold">x′</text>
<line x1="230" y1="225" x2="280" y2="225" class="stroke-ink" marker-end="url(#c104-arrow)"/>
<rect x="280.0" y="210.0" width="80" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="320" y="229" text-anchor="middle" class="fig-small">layer norm 2</text>
<line x1="360" y1="225" x2="388" y2="225" class="stroke-ink" marker-end="url(#c104-arrow)"/>
<rect x="388.0" y="210.0" width="124" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="450" y="229" text-anchor="middle" class="fig-small">SwiGLU</text>
<polyline points="512,225 550,225 550,295 249,295" class="stroke-ink" marker-end="url(#c104-arrow)"/>
<circle cx="230" cy="295" r="17" class="stroke-ink fill-paper"/><text x="230" y="300" text-anchor="middle" class="fig-bold">+</text>
<text x="216" y="81" text-anchor="end" class="fig-small">the raw stream skips the norm</text>
<text x="216" y="334" text-anchor="end" class="fig-bold">x″ = block output</text>
<text x="300" y="372" text-anchor="middle" class="fig-small">Orange: the stream. Each side loop reads a normed copy and writes a correction back with +.</text>
</svg>
Figure 1. One pre-norm block. Notice that the orange stream passes by the layer norms on its way down: each norm sits on a side loop, and only the correction travels back through a plus.
```

## What you already know

You already have these ideas:

- Attention scores each query against each key, divides by $\sqrt{d_{\text{head}}}$, adds a mask, and applies a softmax over the keys, with several narrow heads running in parallel.
- Rotary embeddings turn queries and keys by their positions, and the values stay untouched.
- SwiGLU is a per-token network built from a gate branch, an up branch, and a down matrix.
- A residual connection returns $x + f(x)$.
- Layer norm recentres a row of numbers and rescales it, and the gradient reaching early layers is a product of one slope per layer.

## Why a plain deep stack loses its gradient

The gradient at layer 1 is a product, one slope for each layer it passes back through. Take three layers that each have slope 0.1. The product is $0.1^3 = 0.001$. At 32 layers it is $0.1^{32}$, far too small to move anything.

Now let each layer return $x + f(x)$ instead of $f(x)$. The slope of $x + f(x)$ is $1 + f'(x)$. With the same slope 0.1 in each of three layers, the product is $1.1^3 = 1.331$, and it expands as $1 + 0.3 + 0.03 + 0.001$. The first term, 1, is the road that skips all three layers. It is always there, whatever the layers do, so the gradient always has a direct way back.

Figure 2 checks this on a real stack: 32 layers of width 16, each computing $\tanh(Wh)$ with random weights of standard deviation $0.5/\sqrt{16}$, run once as a plain stack and once with the $x +$ added. The plain stack passes only 0.0178 of the gradient back through 4 layers, $3 \times 10^{-6}$ through 16, and $2.6 \times 10^{-11}$ through 32. The residual stack passes 1.155, 2.648, and 6.445. The residual gradient does not stay frozen at 1, because the corrections add up and the size creeps upward. It grows by a factor of six, though, not shrinks by a factor of tens of billions. This is one random draw of the weights, so the exact numbers vary, but the gap of ten orders of magnitude does not. The growth is the cost of letting corrections pile up, and the layer norm and the final norm below keep it from running away.

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
Figure 2. Gradient size against the number of layers it has passed back through, for the same 32 layers run two ways. Notice the log scale: the plain stack loses more than ten orders of magnitude, while the stack with x + f(x) stays within one order of magnitude of 1.
```

## The stream is a shared draft

Think of the vector for each token as a draft that every layer reads and edits. A layer reads the draft, works out a small correction, and writes it by adding it on. The running sum is called the residual stream. A block whose layers have nothing to say adds zero and returns its input exactly. Passing the input through is the default, and each layer only has to learn a change.

## Layer norm keeps the scale steady

Every correction added to the stream changes its size, and over many layers the size drifts. A drifting size causes two kinds of trouble. The attention scores grow with the size of the vectors, so large vectors make the softmax too sharp and small ones make it too flat. And each layer would have to cope with whatever scale reached it. Layer norm resets each token to a steady scale before a layer reads it.

Work one token by hand: $[1, 2, 3, 6]$. The mean is $(1 + 2 + 3 + 6)/4 = 3$. The gaps from the mean are $-2, -1, 0, 3$, and their squares are $4, 1, 0, 9$. The average of the squares, the variance, is $14/4 = 3.5$. Add a tiny $\epsilon = 10^{-5}$ and take the square root to get $1.8708$. Dividing each gap by that gives $[-1.0690, -0.5345, 0, 1.6036]$. The ten-times-larger token $[10, 20, 30, 60]$ gives exactly the same row, so the output no longer depends on the token's overall size. The formula is

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot g + b
$$

Read it aloud: subtract the token's mean $\mu$, divide by its standard deviation $\sigma$ (made safe by $\epsilon$), then scale by a learned gain $g$ and shift by a learned bias $b$. Figure 3 shows three tokens before and after.

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
Figure 3. Three tokens before and after layer norm. Notice that tokens A and B have very different sizes but become identical, and that every token ends at mean 0 and variance 1.
```

Three design choices are worth a reason. The statistics come from each token's own entries, not from the whole batch, because sequences have different lengths and batch statistics are unreliable, and a token's norm should not depend on its neighbours. The $\epsilon$ is there because a token whose entries are all equal has variance 0, and dividing by zero would break the model. In the example, with variance 3.5, it changes nothing visible. The gain and bias are learned so that the network can choose a scale other than 1 if it needs one. With $g = 2$ and $b = 1$, the example row becomes $[-1.138, -0.069, 1, 4.207]$. The normalisation takes away the scale, and $g$ and $b$ hand back whatever scale training finds useful.

```python
mu = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)           # divides by d, numpy's default
out = (x - mu) / np.sqrt(var + eps) * gain + bias
```

## Pre-norm, and why the skip must stay raw

The norm can go in two places. Post-norm applies it after the add, $\operatorname{LN}(x + f(x))$, which puts the norm on the main road. Every block then rescales the stream, so the clean "1" road from the last section is interrupted at each block, and very deep post-norm stacks usually need a slow, careful learning-rate warm-up. Pre-norm applies it to the copy that the sublayer reads, $x + f(\operatorname{LN}(x))$, which puts the norm on the side road. The stream itself is never rescaled.

The trade-off is that nothing rescales the stream, so its size still grows with depth. Pre-norm models add one last layer norm right before the output to compensate. This course uses pre-norm.

> [!WARNING]
> Add each sublayer's output to the raw stream `x`, not to `layer_norm(x)`. Adding to the normed copy puts the norm back on the main road and removes the skip.

## Assemble the block

The block has two sublayers, each reading a normed copy and adding its result to the raw stream. Figure 1 shows the flow again.

$$
X' = X + \operatorname{Attn}\big(\operatorname{LN}_1(X)\big), \qquad
X'' = X' + \operatorname{SwiGLU}\big(\operatorname{LN}_2(X')\big)
$$

Read it aloud: each sublayer reads a normed copy of the stream and adds its answer onto the raw stream. Attention is the part with the most steps, and each step has a reason:

1. Project the normed rows with $W_q$, $W_k$, and $W_v$, then reshape each result into `heads` heads of width $d_{\text{head}} = d/\text{heads}$, so each head asks its own question.
2. Turn the queries and keys, not the values, with the rotary tables. The tables have shape `(T, d_head)` and broadcast over the heads.
3. Compute the scores $q k^\top / \sqrt{d_{\text{head}}}$. The division is there because a dot product of $d_{\text{head}}$ entries of ordinary size has a spread that grows like $\sqrt{d_{\text{head}}}$, and without it the softmax saturates, as in the scaling chapter.
4. Add the mask before the softmax: 0 allows a position and $-10^9$ forbids it, so a forbidden position ends with weight 0.
5. Take the softmax over the key axis, multiply by $v$, merge the heads, and multiply by $W_o$ to mix the heads back into one width-$d$ vector.

```python
x1 = x + attention(layer_norm(x, g1, b1))
x2 = x1 + swiglu(layer_norm(x1, g2, b2), W_gate, W_up, W_down)
```

> [!KEY]
> A stack of 32 blocks trains because each sublayer writes a small correction onto a stream that flows straight through, so the gradient always has a direct road back. Layer norm resets the scale before each sublayer reads the stream, and the norm sits on the side road so that the skip stays clean. Both fixes together answer the opening question.

## Where the parameters live

Count the weights. Attention holds four matrices of $d \times d$, which is $4d^2$. Splitting into more heads does not change this, because the width $d$ is divided among the heads, not enlarged. SwiGLU holds $3\,d\,d_h$. The two layer norms hold a gain and a bias of length $d$ each, which is $4d$:

$$
4d^2 + 3\,d\,d_h + 4d, \qquad d_h = \operatorname{round}\!\big(\tfrac{2}{3}\, d_{\text{ff}}\big)
$$

For $d = 64$ and $d_{\text{ff}} = 192$, $d_h = 128$ and the total is $16384 + 24576 + 256 = 41216$. Figure 4 shows how the weights split. With the usual $d_{\text{ff}} = 4d$, a block holds about $4d^2$ in attention and $8d^2$ in the feed-forward network, so the feed-forward part holds two-thirds of the weights.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Stacked bars of where a block holds its parameters at d equals 64, for two feed-forward widths: attention holds about a third to two fifths, the feed-forward network the rest, and the norms under one percent">
<text x="300" y="22" text-anchor="middle" class="fig-bold">Where a block keeps its weights (d = 64)</text>
<text x="40" y="60" class="fig-small fig-bold">d_ff = 192 (the worked example): 41,216 weights</text>
<rect x="40.0" y="70" width="206.7" height="34" class="fill-a"/>
<text x="143.4" y="86" text-anchor="middle" class="fig-small fig-bold">attention 4d²</text>
<text x="143.4" y="100" text-anchor="middle" class="fig-small">16,384 (39.8%)</text>
<rect x="246.7" y="70" width="310.1" height="34" class="fill-b"/>
<text x="401.7" y="86" text-anchor="middle" class="fig-small fig-bold">SwiGLU 3·d·d_h</text>
<text x="401.7" y="100" text-anchor="middle" class="fig-small">24,576 (59.6%)</text>
<rect x="556.8" y="70" width="3.2" height="34" class="fill-d"/>
<line x1="558.4" y1="104" x2="558.4" y2="116" class="stroke-ink"/>
<text x="560.0" y="128" text-anchor="end" class="fig-small">norms 4d: 256 (0.6%)</text>
<text x="40" y="150" class="fig-small fig-bold">d_ff = 4d = 256: 49,472 weights</text>
<rect x="40.0" y="160" width="172.2" height="34" class="fill-a"/>
<text x="126.1" y="176" text-anchor="middle" class="fig-small fig-bold">attention 4d²</text>
<text x="126.1" y="190" text-anchor="middle" class="fig-small">16,384 (33.1%)</text>
<rect x="212.2" y="160" width="345.1" height="34" class="fill-b"/>
<text x="384.8" y="176" text-anchor="middle" class="fig-small fig-bold">SwiGLU 3·d·d_h</text>
<text x="384.8" y="190" text-anchor="middle" class="fig-small">32,832 (66.4%)</text>
<rect x="557.3" y="160" width="2.7" height="34" class="fill-d"/>
<line x1="558.7" y1="194" x2="558.7" y2="206" class="stroke-ink"/>
<text x="560.0" y="218" text-anchor="end" class="fig-small">norms 4d: 256 (0.5%)</text>
</svg>
Figure 4. Where the weights of one block sit at d = 64. Notice that the norms hold under one percent, and that at the usual d_ff = 4d the feed-forward network holds about two-thirds of the block.
```

## Cost and scaling

A block holds about $12d^2$ weights, and each weight takes part in one multiply and one add per token, so a forward pass costs about $24d^2$ floating-point operations per token. On top of that comes the cost of comparing every query with every key, which grows with the sequence length. For $d = 4096$ and 32 blocks, the blocks hold $12 \times 32 \times 4096^2 \approx 6.4$ billion weights. A vocabulary of 50000 tokens adds about 0.2 billion in the embedding table.

## Where this breaks

**The stream grows with depth.** Nothing rescales the stream in a pre-norm block, so its entries get larger the deeper you go. The final layer norm before the output handles this.

**Long inputs are untested territory.** The block's weights do not depend on the sequence length, but the rotary angles of a very long input go beyond those seen in training, as "Rotating queries and keys: rotary position embeddings" explained, and attention costs more as the sequence grows.

> [!RECAP]
> Carry these facts into the exercise:
> - The slope of $x + f(x)$ is $1 + f'(x)$, so the gradient always has a direct road back.
> - Layer norm runs over the last axis with $\epsilon = 10^{-5}$ and a variance that divides by $d$, then applies the gain and the bias.
> - The block is `x1 = x + attention(LN1(x))`, then `x2 = x1 + swiglu(LN2(x1))`, with adds to the raw stream.
> - Attention scales by $\sqrt{d_{\text{head}}}$, adds the mask before the softmax, and turns only queries and keys.
> - A block holds $4d^2 + 3\,d\,d_h + 4d$ parameters, whatever the number of heads.

## Build it step by step

The starter file provides `rope_tables`, `apply_rope`, and `swiglu`, solved, from the earlier chapters. You write three functions, in the order the checks test them.

1. **`layer_norm(x, gain, bias, eps=1e-5)`** takes `x` of shape `(..., d)` and returns the same shape. Use `axis=-1` and `keepdims=True` for the mean and the variance, divide by `np.sqrt(var + eps)`, then multiply by `gain` and add `bias`. The common mistake is to apply the gain and bias before dividing.
2. **`transformer_block_forward(x, params, mask)`** maps `(T, d)` to `(T, d)`. Norm, project, reshape with `.reshape(T, heads, d_head).transpose(1, 0, 2)`, turn `q` and `k`, scale, add `mask`, softmax over the last axis, multiply by `v`, merge, multiply by `Wo`, and add to the raw `x`. Then norm again, call `swiglu`, and add again. Read the base with `params.get("theta", 10000.0)`. Subtract the row maximum before `np.exp`, so the softmax cannot overflow. The common mistake is the residual on the normed input.
3. **`block_parameter_count(d_model, d_ff, heads)`** returns an `int`: `4 * d_model**2`, plus `3 * d_model * d_h` with `d_h = round(2 * d_ff / 3)`, plus `4 * d_model`. Raise `ValueError` when `heads` does not divide `d_model`. The common mistake is to forget the norm's gain and bias.

## The exercise

You write `layer_norm`, `transformer_block_forward`, and `block_parameter_count`. The checks verify `layer_norm` on the $[1, 2, 3, 6]$ example, its zero mean and unit variance, its scale independence, and the order of the gain and bias. They check the block's shape, that zeroing `Wo` and `W_down` makes it an exact identity, and that under a causal mask changing a later token leaves earlier outputs unchanged. They compare the block with a loop-based reference, so a missing scale, a wrong softmax axis, or a residual on the normed copy fails. The last check confirms the count 41216 for $(64, 192, 8)$, a second configuration, and that changing `heads` leaves the count alone.
