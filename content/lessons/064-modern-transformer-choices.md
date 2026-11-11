---
title: "Modern transformer choices: RMSNorm, grouped-query attention, and parameter budgets"
slug: 064-modern-transformer-choices
part: "Part 6 — Transformers"
summary: Why current decoder models use RMSNorm, fewer key-value heads, and tied embeddings, and how to count their weights and cache.
minutes: 75
gpu: false
objectives:
  - Compute RMSNorm and its gradient by hand, and say what it drops from layer norm and why the loss rarely misses it.
  - Measure how pre-norm and post-norm change the stream's size and the gradient's size across 32 layers.
  - Build grouped-query attention, and work out the weight and key-value memory it saves.
  - Count the parameters of a decoder from its config, with and without tied embeddings, and match two real models.
  - Apply the sizing rules for the feed-forward width, the head width, and the depth.
prereqs:
  - 040-normalisation-layers
  - 059-scaled-attention-and-heads
  - 063-the-transformer-block
exercise: 064-modern-transformer-choices
---

# Modern transformer choices: RMSNorm, grouped-query attention, and parameter budgets

Open the config file of a released 8-billion-parameter model and read the lines that describe its blocks. You find `num_attention_heads: 32` next to `num_key_value_heads: 8`, an `rms_norm_eps` where you expected a layer norm, `intermediate_size: 14336` for a model whose width is 4096, and `tie_word_embeddings: false`. The block from the previous chapter has none of these four choices. Each one is a small engineering decision, and each has a reason that you can compute.

Without the reasons, a config is a list of numbers to copy. With them, you can read any decoder config and say what it costs in weights and memory, and why a model called "8 billion" has 8,030,261,248 weights.

This chapter takes the choices in turn: a norm that drops the mean, a measurement of where the norm goes, attention that shares keys and values, and the arithmetic that adds it all up. Figure 1 previews the attention change.

```figure
<svg viewBox="0 0 600 268" role="img" aria-label="Eight query heads and the key-value heads they read, for 8 groups, 2 groups, and 1 group, with the key-value bytes per token per layer">
<text x="14" y="53" class="fig-bold fig-small">multi-head</text>
<rect x="150" y="34" width="36" height="26" rx="4" class="fill-a"/><text x="168" y="52" text-anchor="middle" class="fig-small fill-paper">q0</text>
<rect x="194" y="34" width="36" height="26" rx="4" class="fill-b"/><text x="212" y="52" text-anchor="middle" class="fig-small fill-paper">q1</text>
<rect x="238" y="34" width="36" height="26" rx="4" class="fill-c"/><text x="256" y="52" text-anchor="middle" class="fig-small fill-paper">q2</text>
<rect x="282" y="34" width="36" height="26" rx="4" class="fill-d"/><text x="300" y="52" text-anchor="middle" class="fig-small fill-paper">q3</text>
<rect x="326" y="34" width="36" height="26" rx="4" class="fill-a"/><text x="344" y="52" text-anchor="middle" class="fig-small fill-paper">q4</text>
<rect x="370" y="34" width="36" height="26" rx="4" class="fill-b"/><text x="388" y="52" text-anchor="middle" class="fig-small fill-paper">q5</text>
<rect x="414" y="34" width="36" height="26" rx="4" class="fill-c"/><text x="432" y="52" text-anchor="middle" class="fig-small fill-paper">q6</text>
<rect x="458" y="34" width="36" height="26" rx="4" class="fill-d"/><text x="476" y="52" text-anchor="middle" class="fig-small fill-paper">q7</text>
<rect x="150" y="74" width="36" height="22" rx="4" class="fill-soft-a stroke-a"/><text x="168.0" y="90" text-anchor="middle" class="fig-small">k, v 0</text>
<line x1="168" y1="60" x2="168" y2="74" class="stroke-a"/>
<rect x="194" y="74" width="36" height="22" rx="4" class="fill-soft-b stroke-b"/><text x="212.0" y="90" text-anchor="middle" class="fig-small">k, v 1</text>
<line x1="212" y1="60" x2="212" y2="74" class="stroke-b"/>
<rect x="238" y="74" width="36" height="22" rx="4" class="fill-soft-c stroke-c"/><text x="256.0" y="90" text-anchor="middle" class="fig-small">k, v 2</text>
<line x1="256" y1="60" x2="256" y2="74" class="stroke-c"/>
<rect x="282" y="74" width="36" height="22" rx="4" class="fill-soft-d stroke-d"/><text x="300.0" y="90" text-anchor="middle" class="fig-small">k, v 3</text>
<line x1="300" y1="60" x2="300" y2="74" class="stroke-d"/>
<rect x="326" y="74" width="36" height="22" rx="4" class="fill-soft-a stroke-a"/><text x="344.0" y="90" text-anchor="middle" class="fig-small">k, v 4</text>
<line x1="344" y1="60" x2="344" y2="74" class="stroke-a"/>
<rect x="370" y="74" width="36" height="22" rx="4" class="fill-soft-b stroke-b"/><text x="388.0" y="90" text-anchor="middle" class="fig-small">k, v 5</text>
<line x1="388" y1="60" x2="388" y2="74" class="stroke-b"/>
<rect x="414" y="74" width="36" height="22" rx="4" class="fill-soft-c stroke-c"/><text x="432.0" y="90" text-anchor="middle" class="fig-small">k, v 6</text>
<line x1="432" y1="60" x2="432" y2="74" class="stroke-c"/>
<rect x="458" y="74" width="36" height="22" rx="4" class="fill-soft-d stroke-d"/><text x="476.0" y="90" text-anchor="middle" class="fig-small">k, v 7</text>
<line x1="476" y1="60" x2="476" y2="74" class="stroke-d"/>
<text x="510" y="54" class="fig-small">4096 B</text><text x="510" y="90" class="fig-small">per token</text>
<text x="14" y="135" class="fig-bold fig-small">grouped, 2 groups</text>
<rect x="150" y="116" width="36" height="26" rx="4" class="fill-a"/><text x="168" y="134" text-anchor="middle" class="fig-small fill-paper">q0</text>
<rect x="194" y="116" width="36" height="26" rx="4" class="fill-a"/><text x="212" y="134" text-anchor="middle" class="fig-small fill-paper">q1</text>
<rect x="238" y="116" width="36" height="26" rx="4" class="fill-a"/><text x="256" y="134" text-anchor="middle" class="fig-small fill-paper">q2</text>
<rect x="282" y="116" width="36" height="26" rx="4" class="fill-a"/><text x="300" y="134" text-anchor="middle" class="fig-small fill-paper">q3</text>
<rect x="326" y="116" width="36" height="26" rx="4" class="fill-b"/><text x="344" y="134" text-anchor="middle" class="fig-small fill-paper">q4</text>
<rect x="370" y="116" width="36" height="26" rx="4" class="fill-b"/><text x="388" y="134" text-anchor="middle" class="fig-small fill-paper">q5</text>
<rect x="414" y="116" width="36" height="26" rx="4" class="fill-b"/><text x="432" y="134" text-anchor="middle" class="fig-small fill-paper">q6</text>
<rect x="458" y="116" width="36" height="26" rx="4" class="fill-b"/><text x="476" y="134" text-anchor="middle" class="fig-small fill-paper">q7</text>
<rect x="150" y="156" width="168" height="22" rx="4" class="fill-soft-a stroke-a"/><text x="234.0" y="172" text-anchor="middle" class="fig-small">k, v 0</text>
<line x1="168" y1="142" x2="168" y2="156" class="stroke-a"/>
<line x1="212" y1="142" x2="212" y2="156" class="stroke-a"/>
<line x1="256" y1="142" x2="256" y2="156" class="stroke-a"/>
<line x1="300" y1="142" x2="300" y2="156" class="stroke-a"/>
<rect x="326" y="156" width="168" height="22" rx="4" class="fill-soft-b stroke-b"/><text x="410.0" y="172" text-anchor="middle" class="fig-small">k, v 1</text>
<line x1="344" y1="142" x2="344" y2="156" class="stroke-b"/>
<line x1="388" y1="142" x2="388" y2="156" class="stroke-b"/>
<line x1="432" y1="142" x2="432" y2="156" class="stroke-b"/>
<line x1="476" y1="142" x2="476" y2="156" class="stroke-b"/>
<text x="510" y="136" class="fig-small">1024 B</text><text x="510" y="172" class="fig-small">per token</text>
<text x="14" y="217" class="fig-bold fig-small">multi-query</text>
<rect x="150" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="168" y="216" text-anchor="middle" class="fig-small fill-paper">q0</text>
<rect x="194" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="212" y="216" text-anchor="middle" class="fig-small fill-paper">q1</text>
<rect x="238" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="256" y="216" text-anchor="middle" class="fig-small fill-paper">q2</text>
<rect x="282" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="300" y="216" text-anchor="middle" class="fig-small fill-paper">q3</text>
<rect x="326" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="344" y="216" text-anchor="middle" class="fig-small fill-paper">q4</text>
<rect x="370" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="388" y="216" text-anchor="middle" class="fig-small fill-paper">q5</text>
<rect x="414" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="432" y="216" text-anchor="middle" class="fig-small fill-paper">q6</text>
<rect x="458" y="198" width="36" height="26" rx="4" class="fill-a"/><text x="476" y="216" text-anchor="middle" class="fig-small fill-paper">q7</text>
<rect x="150" y="238" width="344" height="22" rx="4" class="fill-soft-a stroke-a"/><text x="322.0" y="254" text-anchor="middle" class="fig-small">k, v 0</text>
<line x1="168" y1="224" x2="168" y2="238" class="stroke-a"/>
<line x1="212" y1="224" x2="212" y2="238" class="stroke-a"/>
<line x1="256" y1="224" x2="256" y2="238" class="stroke-a"/>
<line x1="300" y1="224" x2="300" y2="238" class="stroke-a"/>
<line x1="344" y1="224" x2="344" y2="238" class="stroke-a"/>
<line x1="388" y1="224" x2="388" y2="238" class="stroke-a"/>
<line x1="432" y1="224" x2="432" y2="238" class="stroke-a"/>
<line x1="476" y1="224" x2="476" y2="238" class="stroke-a"/>
<text x="510" y="218" class="fig-small">512 B</text><text x="510" y="254" class="fig-small">per token</text>
<text x="300" y="262" text-anchor="middle" class="fig-small">Cache bytes: 2 (k and v) × groups × 128 (d_head) × 2 (16-bit), per layer.</text>
</svg>
Figure 1. Eight query heads and the key-value heads they read. Notice that the number of queries never changes, and only the stored keys and values shrink, from 4096 bytes per token per layer to 512.
```

## What you already know

You already have these ideas:

- Layer norm subtracts a token's mean, divides by its standard deviation, then applies a gain and a bias (see [normalisation layers](/lessons/040-normalisation-layers)).
- Multi-head attention gives each of $H$ heads its own query, key, and value of width $d_{\text{head}}$, and a causal mask hides the future (see [scaled attention and heads](/lessons/059-scaled-attention-and-heads)).
- A pre-norm block adds each sublayer's output to a raw stream, and a SwiGLU feed-forward network has three matrices (see [the transformer block](/lessons/063-the-transformer-block)).
- A softmax is unchanged when you add the same number to every score.

## RMSNorm: layer norm without the mean

Layer norm does two jobs. It recentres a token so its entries average 0, and it rescales so their spread is 1. RMSNorm keeps the rescaling and drops the recentring. It divides the token by its root mean square, the square root of the average of the squared entries:

$$
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}} \cdot g
$$

Read it aloud: measure how large the token is, divide that size out, then multiply by a learned gain $g$ per entry. There is no bias, because a bias is what lets a norm shift the output, and nothing here shifts.

Work the token $[1, 2, 3, 6]$ by hand. The squares are $1, 4, 9, 36$, and their average is $12.5$. With $\epsilon = 10^{-5}$ the root is $\sqrt{12.50001} = 3.5355$. Dividing gives $[0.2828, 0.5657, 0.8485, 1.6971]$. The squares of that row average exactly 1, so the row has unit size. Its mean is $0.8485$, not 0. Layer norm turned the same token into $[-1.0690, -0.5345, 0, 1.6036]$, and that row has mean 0. The two rows differ because layer norm subtracted 3 first.

Why is losing the recentring acceptable? First, the mean is one direction out of $d$. If a token's entries are independent and centred, the share of its squared length along the all-ones direction is about $1/d$, which is 0.02 percent at $d = 4096$. A token with a large shared offset carries more, and RMSNorm keeps that offset for the next matrix to use like any other direction.

Second, Zhang and Sennrich argue that the rescaling is what makes layer norm help, and report quality close to layer norm at lower cost. Treat that as evidence from their experiments, not a proof. The saving is plain: one reduction instead of two, and $d$ fewer parameters per norm.

```python
inv_rms = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
y = x * inv_rms * gain
```

### The backward pass

Write $r = 1/\sqrt{\operatorname{mean}(x^2) + \epsilon}$, so $y_i = g_i\, x_i\, r$. Entry $x_j$ reaches the output twice: directly through $x_j$, and through $r$, because every entry feeds the mean square. The slope of $r$ with respect to $x_j$ is $-r^3 x_j / d$. Collect both routes. With $u = g \odot dy$ and $\hat{x} = x\, r$:

$$
dx = r\,\big(u - \hat{x}\cdot \operatorname{mean}(u \odot \hat{x})\big), \qquad dg = \sum_{\text{rows}} dy \odot \hat{x}
$$

Read it aloud: pass the gradient through the scale $r$, then subtract the part that points along the output $\hat{x}$. That part is removed because multiplying $x$ by any positive number leaves $y$ unchanged, so the loss cannot depend on how far you move along $x$. Layer norm's backward pass has one more subtracted term, $\operatorname{mean}(u)$, for the recentring.

Check it on $x = [1, 2, 3, 6]$, $g = 1$, and $dy = [1, 0, 0, 0]$. Here $r = 0.28284$, and the mean of $u \odot \hat{x}$ is $0.2828 / 4 = 0.0707$, so $u - 0.0707\,\hat{x} = [0.98, -0.04, -0.06, -0.12]$. Multiply by $r$ to get $dx = [0.2772, -0.0113, -0.0170, -0.0339]$. As the scale argument predicts, $dx \cdot x = 0.2772 - 0.0226 - 0.0510 - 0.2034 = 0$, up to rounding.

> [!KEY]
> RMSNorm divides a token by its root mean square and nothing else. It keeps the rescaling that layer norm provides, skips the recentring that removes about $1/d$ of a token's energy, and has a backward pass with one term fewer.

## Pre-norm and post-norm, measured

The previous chapter said that post-norm puts the norm on the main road and pre-norm puts it on a side road. This section measures what that does. The setup is 32 sublayers, each a ReLU feed-forward network of width $4d$ with $d = 64$ and random weights, plus an RMSNorm with gain 1. The stack runs as $x + f(\operatorname{norm}(x))$ for pre-norm and $\operatorname{norm}(x + f(x))$ for post-norm, and a random gradient enters at the output. The experiment records the root mean square of the stream after each layer, and the size of the gradient reaching that layer's input divided by the size of the upstream gradient. Figure 2 plots the average of five seeds.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Two panels over 32 layers: the stream rms grows from 1.4 to 5.6 for pre-norm and stays at 1 for post-norm, and the input gradient falls from 6.0 to 1.0 across layers for pre-norm and stays between 0.9 and 1.4 for post-norm">
<text x="155" y="18" text-anchor="middle" class="fig-bold fig-small">stream rms after each layer</text>
<line x1="50" y1="210.0" x2="260" y2="210.0" class="stroke-grid"/><text x="44" y="214.0" text-anchor="end" class="fig-small">0</text>
<line x1="50" y1="150.0" x2="260" y2="150.0" class="stroke-grid"/><text x="44" y="154.0" text-anchor="end" class="fig-small">2</text>
<line x1="50" y1="90.0" x2="260" y2="90.0" class="stroke-grid"/><text x="44" y="94.0" text-anchor="end" class="fig-small">4</text>
<line x1="50" y1="30.0" x2="260" y2="30.0" class="stroke-grid"/><text x="44" y="34.0" text-anchor="end" class="fig-small">6</text>
<line x1="50" y1="210" x2="260" y2="210" class="stroke-muted"/>
<text x="50.0" y="228" text-anchor="middle" class="fig-small">1</text>
<text x="97.4" y="228" text-anchor="middle" class="fig-small">8</text>
<text x="151.6" y="228" text-anchor="middle" class="fig-small">16</text>
<text x="205.8" y="228" text-anchor="middle" class="fig-small">24</text>
<text x="260.0" y="228" text-anchor="middle" class="fig-small">32</text>
<polyline points="50.0,168.0 56.8,159.2 63.5,151.1 70.3,142.7 77.1,136.0 83.9,129.3 90.6,124.1 97.4,118.9 104.2,111.8 111.0,107.5 117.7,105.6 124.5,100.9 131.3,99.1 138.1,97.0 144.8,92.8 151.6,88.8 158.4,86.6 165.2,83.4 171.9,79.1 178.7,75.0 185.5,70.8 192.3,66.2 199.0,63.5 205.8,60.3 212.6,57.6 219.4,54.4 226.1,51.7 232.9,48.5 239.7,46.2 246.5,44.0 253.2,42.5 260.0,41.3" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="50.0,180.0 56.8,180.0 63.5,180.0 70.3,180.0 77.1,180.0 83.9,180.0 90.6,180.0 97.4,180.0 104.2,180.0 111.0,180.0 117.7,180.0 124.5,180.0 131.3,180.0 138.1,180.0 144.8,180.0 151.6,180.0 158.4,180.0 165.2,180.0 171.9,180.0 178.7,180.0 185.5,180.0 192.3,180.0 199.0,180.0 205.8,180.0 212.6,180.0 219.4,180.0 226.1,180.0 232.9,180.0 239.7,180.0 246.5,180.0 253.2,180.0 260.0,180.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<text x="445" y="18" text-anchor="middle" class="fig-bold fig-small">gradient at the layer input</text>
<line x1="340" y1="210.0" x2="550" y2="210.0" class="stroke-grid"/><text x="334" y="214.0" text-anchor="end" class="fig-small">0</text>
<line x1="340" y1="150.0" x2="550" y2="150.0" class="stroke-grid"/><text x="334" y="154.0" text-anchor="end" class="fig-small">2</text>
<line x1="340" y1="90.0" x2="550" y2="90.0" class="stroke-grid"/><text x="334" y="94.0" text-anchor="end" class="fig-small">4</text>
<line x1="340" y1="30.0" x2="550" y2="30.0" class="stroke-grid"/><text x="334" y="34.0" text-anchor="end" class="fig-small">6</text>
<line x1="340" y1="210" x2="550" y2="210" class="stroke-muted"/>
<text x="340.0" y="228" text-anchor="middle" class="fig-small">1</text>
<text x="387.4" y="228" text-anchor="middle" class="fig-small">8</text>
<text x="441.6" y="228" text-anchor="middle" class="fig-small">16</text>
<text x="495.8" y="228" text-anchor="middle" class="fig-small">24</text>
<text x="550.0" y="228" text-anchor="middle" class="fig-small">32</text>
<polyline points="340.0,30.6 346.8,82.4 353.5,108.0 360.3,121.8 367.1,132.3 373.9,139.1 380.6,144.5 387.4,148.2 394.2,152.3 401.0,155.1 407.7,157.4 414.5,159.7 421.3,161.7 428.1,163.9 434.8,165.3 441.6,167.1 448.4,168.4 455.2,169.3 461.9,170.6 468.7,171.6 475.5,172.6 482.3,173.5 489.0,174.3 495.8,174.9 502.6,175.7 509.4,176.4 516.1,176.9 522.9,177.6 529.7,178.1 536.5,178.5 543.2,179.0 550.0,179.6" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="340.0,173.2 346.8,173.8 353.5,175.3 360.3,176.0 367.1,174.1 373.9,175.2 380.6,174.0 387.4,172.1 394.2,169.1 401.0,169.6 407.7,170.2 414.5,172.5 421.3,171.3 428.1,173.5 434.8,174.1 441.6,172.4 448.4,173.0 455.2,171.8 461.9,175.2 468.7,176.6 475.5,176.0 482.3,175.7 489.0,177.6 495.8,176.0 502.6,176.3 509.4,176.0 516.1,178.1 522.9,179.8 529.7,181.8 536.5,179.9 543.2,179.8 550.0,180.5" fill="none" class="stroke-a" stroke-width="2.5"/>
<text x="155" y="250" text-anchor="middle" class="fig-small">layer</text><text x="445" y="250" text-anchor="middle" class="fig-small">layer</text>
<line x1="170" y1="270" x2="196" y2="270" class="stroke-b" stroke-width="2.5"/><text x="202" y="274" class="fig-small">pre-norm</text><line x1="320" y1="270" x2="346" y2="270" class="stroke-a" stroke-width="2.5"/><text x="352" y="274" class="fig-small">post-norm</text>
</svg>
Figure 2. Stream size and input gradient across 32 layers. Notice that post-norm holds both near 1 at every layer, while pre-norm lets the stream grow to 5.6 and the gradient at layer 1 grow to 6.0 times the gradient at the output.
```

Post-norm pins the stream at exactly 1, because the last operation of every layer is the norm, and the gradient stays between 0.94 and 1.36 at all 32 layers. Pre-norm never rescales the stream, so each layer's write adds to it, and the size grows from 1.40 after layer 1 to 5.62 after layer 32. The gradient mirrors that growth. The gradient at a layer's input is the straight road (worth 1) plus one branch for every later layer, and independent branches add their squares, so the size grows like the square root of the number of branches. Table 1 shows the gradient at layer 1 against $\sqrt{L}$.

| Layers $L$ | pre-norm | post-norm | $\sqrt{L}$ |
|---|---|---|---|
| 8 | 2.94 | 0.90 | 2.83 |
| 16 | 4.28 | 1.05 | 4.00 |
| 32 | 5.98 | 1.23 | 5.66 |
| 64 | 8.05 | 1.21 | 8.00 |

Table 1. First-layer gradient divided by output gradient, averaged over five seeds.

The price of the clean skip road is a stream that gets large, so a later sublayer's write is a smaller fraction of it, and a final norm must divide the growth back out.

> [!WARNING]
> This experiment does not show post-norm failing. At initialisation both wirings keep the gradient within a factor of 6 over 32 layers. Xiong et al. report that post-norm transformers need a learning-rate warm-up and pre-norm ones do not, and they analyse real attention layers. This toy has no attention and no training, so it cannot test that.

## Fewer key-value heads

Generation reads the keys and values of every earlier token at each step, and it keeps them in memory so it does not recompute them. A layer stores $2 \times H \times d_{\text{head}}$ numbers per token, one key and one value for each head. The queries are not stored, because only the newest token's queries are used. So the stored bytes grow with the number of key-value heads, and the number of queries does not matter to the cache.

That suggests letting several query heads read the same key and value. Shazeer's multi-query attention takes this to the limit: one key head and one value head, shared by all $H$ query heads. Ainslie et al. add a dial between the extremes. Grouped-query attention uses $G$ key-value heads, called groups. With $G = H$ it is ordinary multi-head attention, and with $G = 1$ it is multi-query attention.

The queries are split into consecutive blocks. Query head $h$ reads group $h \,/\!/\, (H/G)$, where $/\!/$ is integer division. With $H = 8$ and $G = 2$, heads 0 to 3 read group 0 and heads 4 to 7 read group 1, as in the middle row of Figure 1. The attention computation does not change. Each head still takes a scaled dot product, applies the mask, and takes a softmax. Only the source of the keys and values changes.

```python
per_group = H // n_groups
k_full = np.repeat(k, per_group, axis=0)    # k0 k0 k0 k0 k1 k1 k1 k1
v_full = np.repeat(v, per_group, axis=0)
scores = q @ k_full.transpose(0, 2, 1) / np.sqrt(d_head)
```

The `np.repeat` call repeats each head in place. `np.tile` would give $k_0\,k_1\,k_0\,k_1$, so query head 1 would read group 1. Both run without error, but a model trained with consecutive blocks computes a different function under the interleaved order.

### The arithmetic

Grouping changes two costs. The attention weights are $W_q$ at $d \times H d_{\text{head}}$, $W_k$ and $W_v$ at $d \times G d_{\text{head}}$ each, and $W_o$ at $H d_{\text{head}} \times d$. For $d = 4096$, $H = 32$, $d_{\text{head}} = 128$, and $G = 8$, that is $16{,}777{,}216 + 2 \times 4{,}194{,}304 + 16{,}777{,}216 = 41{,}943{,}040$ per layer. Multi-head attention at the same width holds $67{,}108{,}864$. Grouping saves $25{,}165{,}824$ weights per layer and $805{,}306{,}368$ over 32 layers.

The cache per token per layer holds $2 \times G \times d_{\text{head}}$ numbers, which is $2 G d_{\text{head}} \times 2$ bytes in 16-bit. At $d_{\text{head}} = 128$ the rows of Figure 1 are 4096, 1024, and 512 bytes for $G = 8$, 2, and 1. Across 32 layers, a model with 32 key-value heads holds $524{,}288$ bytes per token, which is 2 GiB at 4096 tokens. The model with $G = 8$ holds $131{,}072$ bytes per token, so 1 GiB covers 8192 tokens. How a server uses the cache belongs to the [inference course](https://qwen.ifkash.dev/). Here it is arithmetic only.

Ainslie et al. report that grouped-query attention with a few groups stays close to multi-head quality, and that multi-query attention gives up a little more.

> [!KEY]
> Grouped-query attention keeps all $H$ query heads and shares $G$ key-value heads, with consecutive query heads sharing one group. It cuts the cache by $H/G$ and the attention weights by a smaller factor, because the queries and output projection stay full size.

## Weight tying and the parameter budget

The embedding table $E$ has shape $(V, d)$ for a vocabulary of $V$ tokens. It turns a token id into a vector. The last step of the model does the opposite: it turns the final vector $h$ into $V$ scores, normally with a separate matrix of shape $(d, V)$. Weight tying reuses the table, so the scores are $h E^{\top}$. Each score is a dot product between the final vector and one token's embedding, which asks how similar the output is to that token. The saving is $V d$ weights.

The saving matters most when the table is a large share of the model. GPT-2 small has $V = 50257$ and $d = 768$, so one table holds $38{,}597{,}376$ weights, about 31 percent of the model. A model with $V = 128256$ and $d = 4096$ holds $525{,}336{,}576$ per table, and two tables are 13 percent of an 8-billion-parameter model. Large models often keep the tables separate, because the saving is smaller.

Count a whole decoder by adding up its parts. With $L$ blocks, $d_{\text{head}} = d/H$, and a feed-forward width $d_{\text{ff}}$ for each matrix:

$$
P = Vd + \underbrace{L\,\big(A + F + 2n\big)}_{\text{blocks}} + n + \underbrace{[\text{untied}]\,Vd}_{\text{output table}}
$$

Here $A = d H d_{\text{head}} + 2\,d\,G\,d_{\text{head}} + H d_{\text{head}}\,d$ is the attention weights, $F$ is $2\,d\,d_{\text{ff}}$ for a plain network and $3\,d\,d_{\text{ff}}$ for a gated one, and $n$ is the weights in one norm, $d$ for RMSNorm and $2d$ for layer norm. The extra $n$ is the final norm. Learned position tables add $T_{\max} d$. Biases add $Hd_{\text{head}} + 2Gd_{\text{head}} + d$ to $A$ and, for a plain network, $d_{\text{ff}} + d$ to $F$. Read the formula aloud: embeddings, then every block, then the last norm, then the output table when it is separate.

Work a toy count with $d = 4$, $H = 2$, $G = 1$, $d_{\text{head}} = 2$, $d_{\text{ff}} = 8$, one plain block, RMSNorm, no biases, and $V = 10$. Attention holds $4 \times 4 + 2 \times 4 \times 2 + 4 \times 4 = 48$. The feed-forward network holds $2 \times 4 \times 8 = 64$. Two norms hold 8, so the block holds 120. Add the final norm (4) and the embedding table (40) to get 164 with tied embeddings. A separate output table adds 40 and gives 204.

Figure 3 applies the formula to three real configs. The totals are 124,439,808 for GPT-2 small, 6,738,415,616 for Llama 2 7B, and 8,030,261,248 for Llama 3 8B, and they match the counts that libraries report for those models.

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="Share of parameters in embeddings, attention, and feed-forward layers for GPT-2 small, Llama 2 7B, and Llama 3 8B">
<text x="14" y="52" class="fig-bold fig-small">GPT-2 small</text>
<rect x="130.0" y="30" width="126.6" height="32" class="fill-d"/>
<text x="193.3" y="51" text-anchor="middle" class="fig-small fill-paper">31.6%</text>
<rect x="256.6" y="30" width="91.1" height="32" class="fill-a"/>
<text x="302.2" y="51" text-anchor="middle" class="fig-small fill-paper">22.8%</text>
<rect x="347.7" y="30" width="182.3" height="32" class="fill-b"/>
<text x="438.9" y="51" text-anchor="middle" class="fig-small fill-paper">45.6%</text>
<text x="14" y="108" class="fig-bold fig-small">Llama 2 7B</text>
<rect x="130.0" y="86" width="15.6" height="32" class="fill-d"/>
<rect x="145.6" y="86" width="127.5" height="32" class="fill-a"/>
<text x="209.3" y="107" text-anchor="middle" class="fig-small fill-paper">31.9%</text>
<rect x="273.0" y="86" width="257.0" height="32" class="fill-b"/>
<text x="401.5" y="107" text-anchor="middle" class="fig-small fill-paper">64.2%</text>
<text x="14" y="164" class="fig-bold fig-small">Llama 3 8B</text>
<rect x="130.0" y="142" width="52.3" height="32" class="fill-d"/>
<text x="156.2" y="163" text-anchor="middle" class="fig-small fill-paper">13.1%</text>
<rect x="182.3" y="142" width="66.9" height="32" class="fill-a"/>
<text x="215.8" y="163" text-anchor="middle" class="fig-small fill-paper">16.7%</text>
<rect x="249.2" y="142" width="280.8" height="32" class="fill-b"/>
<text x="389.6" y="163" text-anchor="middle" class="fig-small fill-paper">70.2%</text>
<rect x="130" y="204" width="14" height="14" class="fill-d"/><text x="150" y="216" class="fig-small">embeddings and head</text><rect x="290" y="204" width="14" height="14" class="fill-a"/><text x="310" y="216" class="fig-small">attention</text><rect x="400" y="204" width="14" height="14" class="fill-b"/><text x="420" y="216" class="fig-small">feed-forward and norms</text>
</svg>
Figure 3. Where the weights live. Notice that the feed-forward share grows from 46 percent to 70 percent across the three models, and that the attention share drops from 32 percent in Llama 2 7B to 17 percent in Llama 3 8B once grouping shrinks $W_k$ and $W_v$.
```

## Two stabilisers for large runs

**z-loss.** A softmax does not change when you add a constant to every logit, so nothing in the cross-entropy loss stops the logits from drifting upward together. Large logits lose precision in 16-bit arithmetic. The z-loss adds a penalty on the log of the softmax's denominator, $\log Z = \log\sum_i e^{\text{logit}_i}$:

$$
L_z = \lambda\,(\log Z)^2
$$

Take the logits $[10, 12, 11]$. Then $\log Z = 12.4076$, its square is $153.95$, and with $\lambda = 10^{-4}$ the penalty is $0.0154$. The gradient with respect to the logits is $2\lambda \log Z$ times the softmax probabilities, which is $[0.00022, 0.00165, 0.00061]$. It pulls every logit down, in proportion to its probability, and does not change which token wins. A value of $\lambda$ near $10^{-4}$ is common.

```python
log_z = np.log(np.sum(np.exp(logits - logits.max()), axis=-1)) + logits.max(axis=-1)
loss = cross_entropy + 1e-4 * np.mean(log_z ** 2)
```

**QK-norm.** Attention scores are dot products of queries and keys, and nothing bounds them. If the queries and keys grow, the softmax saturates into a near one-hot choice, and training can become unstable. QK-norm applies an RMSNorm to each head's queries and keys before the dot product. With unit root mean square, each vector has length $\sqrt{d_{\text{head}}}$, so Cauchy-Schwarz bounds the scaled score by $\sqrt{d_{\text{head}}}$, which is 11.3 at $d_{\text{head}} = 128$. Learned gains move the bound, but the scores can no longer grow without limit through the sizes of $q$ and $k$ alone.

## Sizing rules of thumb

Table 2 lists the shape numbers of the three models of Figure 3.

| Model | $d$ | Layers | Heads | $d_{\text{head}}$ | $d_{\text{ff}}$ | $d_{\text{ff}}/d$ | $d$ / layers |
|---|---|---|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 12 | 64 | 3072 | 4.00 | 64 |
| Llama 2 7B | 4096 | 32 | 32 | 128 | 11008 | 2.69 | 128 |
| Llama 3 8B | 4096 | 32 | 32 | 128 | 14336 | 3.50 | 128 |

Table 2. Shape numbers of three released models.

- **Feed-forward width.** A plain network uses $d_{\text{ff}} = 4d$. A gated network uses about $\tfrac{8}{3}d$, which is the two-thirds rule of the previous chapter, so the weights match a plain network. Round it up to a multiple of 256 for the hardware: $\tfrac{8}{3} \times 4096 = 10922.67$ rounds up to 11008, which is the Llama 2 7B width. Llama 3 8B chose 3.5, a wider network, and spends 70 percent of its weights there.
- **Head width.** Models grow by adding heads and keep the head width near 64 or 128, which fits the hardware.
- **Aspect ratio.** Width divided by depth is 64 in the small model and 128 in the two large ones. Kaplan et al. report that loss at a fixed parameter count depends only weakly on this shape over a wide range, so the ratio is a soft choice.

> [!TIP]
> Pick $d_{\text{head}}$ and the ratios from the table, then call your parameter counter and adjust $d_{\text{ff}}$ until the total lands where you want it.

## Where this breaks

**The group count must divide the head count.** With 32 heads, $G$ can be 1, 2, 4, 8, 16, or 32. Any other count leaves some query heads with no group, so the code must raise an error.

**Counts disagree by small, specific amounts.** A missing final norm is off by $d$, and tied embeddings counted twice are off by $Vd$. When your count differs from a library's, the difference usually names the part you forgot.

**The stabilisers treat a symptom.** A z-loss or QK-norm that fixes a run is holding down growing logits. If they grow because the learning rate is too high, lowering it is the better fix.

> [!RECAP]
> Carry these facts into the exercise:
> - `rmsnorm(x) = x / sqrt(mean(x * x) + eps) * gain`, with no subtraction. The backward pass is `dx = r * (u - xhat * mean(u * xhat))` with `u = dy * gain`, and `dgain` sums `dy * xhat` over every row.
> - Grouped-query attention sends query head `h` to group `h // (H // G)`. Use `np.repeat`, not `np.tile`. `G = H` is multi-head and `G = 1` is multi-query.
> - The cache holds `2 * n_layers * n_kv_heads * d_head * seq_len * bytes` bytes.
> - A parameter count adds the embedding table, every block with two norms, a final norm, and a second table only when the embeddings are untied.
> - Pre-norm is `x + f(norm(x))` and post-norm is `norm(x + f(x))`. Going backward, pre-norm gives `dx = dout + norm_back(f_back(dout))`, and post-norm gives `dz = norm_back(dout)` then `dx = dz + f_back(dz)`.

## Build it step by step

The starter provides `mlp_forward`, `mlp_backward`, `make_stack`, and `depth_profile`. You write six functions, in the order the checks test them.

1. **`rmsnorm(x, gain, eps=1e-5)`** takes `x` of shape `(..., d)` and returns the same shape. Use `axis=-1` and `keepdims=True`. The common mistake is to subtract the mean first, which is layer norm and fails the $[1, 2, 3, 6]$ check.
2. **`rmsnorm_backward(dy, x, gain, eps=1e-5)`** returns `(dx, dgain)`. Compute `xhat` and `u = dy * gain`, then apply the formula from the chapter. The common mistake is to forget that `dgain` has shape `(d,)` and must sum over every leading axis, not only the last one.
3. **`gqa_attention(q, k, v, n_groups, causal=True)`** takes `q` of shape `(H, T, d_head)` and `k`, `v` of shape `(n_groups, T, d_head)`. Repeat the groups with `np.repeat`, add the causal mask, subtract the row maximum before `np.exp`, and raise `ValueError` when `n_groups` does not divide `H`. The common mistake is `np.tile`.
4. **`kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per_value=2)`** returns an `int`. The common mistake is to leave out the factor 2 for keys and values.
5. **`count_params(cfg)`** takes the config dict named in its docstring and returns an `int`. Build up `attn`, `ffn`, and `norm`, then add the embedding, the optional position table, the final norm, and the optional output table. The common mistake is to count tied embeddings twice.
6. **`residual_stack(x, layers, kind, dout)`** returns `(out, grads)`. Store the raw input for pre-norm and the sum for post-norm, because each backward pass needs it. Chain your `rmsnorm_backward` and the provided `mlp_backward`. The common mistake is the residual on the normed copy in pre-norm, which fails the zero-weight identity check.

## The exercise

You write `rmsnorm`, `rmsnorm_backward`, `gqa_attention`, `kv_cache_bytes`, `count_params`, and `residual_stack`. The checks confirm that `rmsnorm` turns $[1, 2, 3, 6]$ into the row above and leaves a nonzero mean, and that `rmsnorm_backward` matches finite differences. They compare `gqa_attention` with a loop reference at $G = H$, $G = 1$, and $G = 2$ with six heads, which catches `np.tile`. They verify `kv_cache_bytes` against the Llama figures, and `count_params` against GPT-2 small, Llama 3 8B, and a real torch module's parameter count across seven option combinations. The stack checks confirm the zero-weight identities, finite-difference gradients for both wirings, and the Figure 2 shape of the depth profile.

## Further reading

Lecture numbers appear only where the page prints them.

- [CS336 lecture 3, Architectures, hyperparameters](https://cs336.stanford.edu/) (Spring 2026). The page prints "Lecture 3" for Mon April 6.
- [CS336 lecture 4, Attention alternatives and mixture of experts](https://cs336.stanford.edu/). The page prints "Lecture 4" for Wed April 8.
- [CS224N, Transformers](https://web.stanford.edu/class/cs224n/). The Winter 2026 schedule lists "Transformers" on Tue Jan 20 and prints no lecture number.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang and Sennrich), for RMSNorm.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer), for multi-query attention.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al.), for grouped-query attention.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (Xiong et al.), for pre-norm against post-norm.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.), for the weak dependence of loss on shape.
- The [inference course](https://qwen.ifkash.dev/), for the key-value cache in serving.
