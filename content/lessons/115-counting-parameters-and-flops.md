---
title: "Counting a transformer's parameters and flops"
slug: 115-counting-parameters-and-flops
part: "Part 7 — Large language models"
summary: Training costs about 6 flops per weight per token, and only the weights that multiply count, not the embedding tables.
minutes: 35
gpu: false
objectives:
  - Count a transformer's weights by component, and separate the ones that multiply from the embedding tables.
  - Explain why a lookup costs no multiply-adds and so stays out of the compute formula.
  - Derive the six in C = 6ND from one weight, one input, and one gradient.
  - Turn a parameter count and a token count into flops and into days on one accelerator.
prereqs:
  - 114-loss-from-size-and-data
  - 060-count-parameters-and-flops
  - 104-the-residual-block
  - 108-cost-of-training
exercise: 115-counting-parameters-and-flops
---

# Counting a transformer's parameters and flops

Someone tells you a model has 1.5 billion parameters and was trained on 300 billion tokens. You want to know how much arithmetic that was, in a number you can set beside a chip's speed, because a chip rated at 200 trillion operations per second will take a certain number of days, and a lab pays for every one of them. Two numbers in the sentence, a count of weights and a count of tokens, have to turn into one number of operations.

Without a rule, you would run the training and time it, which is a weeks-long way to learn the price. Counting by hand fails in a different way. A transformer has attention matrices, feed-forward matrices, and big lookup tables, and if you count all of them as equal you will overstate the cost of a small model by half. Which weights do arithmetic, and how much arithmetic does each one do per token?

This chapter answers both questions. First you count the weights that multiply, then you work out what one weight costs per token, and the product is the training bill.

## What you already know

- A flop is one floating-point operation, one multiply or one add, and a multiply-add pair costs 2.
- A network's cost follows from its weight shapes, as in the chapter on counting the parameters, flops, and memory of a network.
- A transformer block is attention followed by a feed-forward network, each added back onto a residual stream, as in the chapter on stacking it up with residuals.
- A gradient is how much a result changes when you nudge one input, and backpropagation computes one for every weight.
- The last chapter used $N$, the parameters that matter, and $D$, the tokens, without saying how to count $N$.

## The picture: which boxes multiply

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A toy transformer with two blocks: four attention matrices and two feed-forward matrices per block count toward N, while the token and position tables are lookups that do not">
<text x="20.0" y="22.0" text-anchor="start" class="fig-bold">One block (counted in N)</text>
<rect x="14.0" y="30.0" width="360.0" height="208.0" rx="6" class="fill-sunk stroke-muted dashed"/>
<text x="24.0" y="50.0" text-anchor="start" class="fig-small fig-bold">attention: four 8×8 matrices</text>
<rect x="24.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="38.0" y="76.0" text-anchor="middle" class="fig-bold">Q</text>
<rect x="66.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="80.0" y="76.0" text-anchor="middle" class="fig-bold">K</text>
<rect x="108.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="122.0" y="76.0" text-anchor="middle" class="fig-bold">V</text>
<rect x="150.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="164.0" y="76.0" text-anchor="middle" class="fig-bold">O</text>
<text x="182.0" y="76.0" text-anchor="start" class="fig-small fig-bold">4 × 64 = 256</text>
<text x="24.0" y="126.0" text-anchor="start" class="fig-small fig-bold">feed-forward: an 8×32 matrix, then a 32×8 matrix</text>
<rect x="24.0" y="134.0" width="112.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="80.0" y="152.0" text-anchor="middle" class="fig-small">up: 8 × 32 = 256</text>
<rect x="24.0" y="172.0" width="112.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="80.0" y="190.0" text-anchor="middle" class="fig-small">down: 32 × 8 = 256</text>
<text x="148.0" y="166.0" text-anchor="start" class="fig-small fig-bold">2 × 256 = 512</text>
<text x="24.0" y="226.0" text-anchor="start" class="fig-small fig-bold">per block 256 + 512 = 768, and 2 blocks make 1,536</text>
<text x="410.0" y="22.0" text-anchor="start" class="fig-bold">Lookup tables (not in N)</text>
<rect x="404.0" y="30.0" width="180.0" height="208.0" rx="6" class="fill-sunk stroke-muted dashed"/>
<rect x="414.0" y="48.0" width="160.0" height="52.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="494.0" y="70.0" text-anchor="middle" class="fig-bold">token table</text>
<text x="494.0" y="88.0" text-anchor="middle" class="fig-small">100 × 8 = 800</text>
<rect x="414.0" y="112.0" width="160.0" height="52.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="494.0" y="134.0" text-anchor="middle" class="fig-bold">position table</text>
<text x="494.0" y="152.0" text-anchor="middle" class="fig-small">16 × 8 = 128</text>
<text x="494.0" y="186.0" text-anchor="middle" class="fig-small">a token id picks one row:</text>
<text x="494.0" y="202.0" text-anchor="middle" class="fig-small">no multiply-adds</text>
<text x="494.0" y="226.0" text-anchor="middle" class="fig-small fig-bold">800 + 128 = 928</text>
</svg>
Figure 1. The toy model from the chapter, 2 blocks, width 8, feed-forward width 32, vocabulary 100, and 16 positions. The orange boxes are weights that multiply, and they make N = 1,536. The green tables only return a row, so their 928 numbers stay out of N. Box areas match the sizes of the matrices.
```

Figure 1 draws a toy model with two blocks, width $d = 8$, and a feed-forward width of $d_{\text{ff}} = 32$. Each block has four square matrices in its attention, called $Q$, $K$, $V$, and $O$ for the query, key, value, and output projections, each $d \times d$. The feed-forward network has two matrices, one that widens from $d$ to $d_{\text{ff}}$ and one that narrows back. On the right are two lookup tables, one with a row per token in the vocabulary and one with a row per position.

The two groups behave differently. Every token passes through every matrix on the left and gets multiplied by each, so each of those weights does arithmetic for each token. A lookup table only returns a row: token number 57 fetches row 57 and nothing is multiplied. That is why the tables stay outside $N$ in a compute formula.

## Counting the toy model

Attention holds $4 \times 8^2 = 256$ weights per block, and the feed-forward network holds $2 \times 8 \times 32 = 512$. A block holds $768$, and two blocks hold $1{,}536$. The token table is $100 \times 8 = 800$, and the position table is $16 \times 8 = 128$, so the tables hold $928$. The output layer reuses the token table, a trick called weight tying, so it adds nothing. The total is $1{,}536 + 928 = 2{,}464$, and $N = 1{,}536$.

In symbols, with $L$ blocks:

$$
N = L\,\bigl(4 d^{2} + 2\, d\, d_{\text{ff}}\bigr),
\qquad \text{embeddings} = (\text{vocab} + \text{positions})\, d
$$

Read it aloud: each block has four $d \times d$ attention matrices and two $d \times d_{\text{ff}}$ feed-forward matrices, and the tables hold one row of width $d$ per token and per position. The embeddings appear once, and the blocks repeat $L$ times, so the usual mistake is to multiply the embedding count by $L$. When $d_{\text{ff}} = 4d$ a block holds $4d^2 + 8d^2 = 12 d^2$ weights, so $N = 12 L d^2$.

```python
attention = layers * 4 * d_model**2
feedforward = layers * 2 * d_model * d_ff
embedding = (vocab + seq_len) * d_model
```

The count leaves out biases and layer-norm scales. Each adds about $d$ numbers per matrix, against $d^2$ for the matrix itself, so they are smaller by a factor of $d$. A gated feed-forward network, from the chapter on SwiGLU, has three matrices instead of two and would change the count, so the formula above assumes the plain two-matrix kind.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Embedding share of total parameters for four model sizes on a log scale, falling from 38 percent for the toy model to 0.37 percent for a 96-block model">
<line x1="210.0" y1="200.0" x2="540.0" y2="200.0" class="stroke-muted"/><line x1="210.0" y1="30.0" x2="210.0" y2="200.0" class="stroke-muted"/><line x1="210.0" y1="30.0" x2="210.0" y2="200.0" class="stroke-grid"/><text x="210.0" y="216.0" text-anchor="middle" class="fig-small">0.1%</text><line x1="320.0" y1="30.0" x2="320.0" y2="200.0" class="stroke-grid"/><text x="320.0" y="216.0" text-anchor="middle" class="fig-small">1%</text><line x1="430.0" y1="30.0" x2="430.0" y2="200.0" class="stroke-grid"/><text x="430.0" y="216.0" text-anchor="middle" class="fig-small">10%</text><line x1="540.0" y1="30.0" x2="540.0" y2="200.0" class="stroke-grid"/><text x="540.0" y="216.0" text-anchor="middle" class="fig-small">100%</text>
<text x="200.0" y="56.0" text-anchor="end" class="fig-small">toy: 2 blocks, width 8</text>
<line x1="210.0" y1="52.0" x2="493.3" y2="52.0" class="stroke-c" stroke-width="3"/>
<circle cx="493.3" cy="52.0" r="6" class="fill-c"/>
<text x="505.3" y="56.0" text-anchor="start" class="fig-small fig-bold">37.7%</text>
<text x="200.0" y="96.0" text-anchor="end" class="fig-small">12 blocks, width 768</text>
<line x1="210.0" y1="92.0" x2="485.1" y2="92.0" class="stroke-c" stroke-width="3"/>
<circle cx="485.1" cy="92.0" r="6" class="fill-c"/>
<text x="497.1" y="96.0" text-anchor="start" class="fig-small fig-bold">31.7%</text>
<text x="200.0" y="136.0" text-anchor="end" class="fig-small">48 blocks, width 1600</text>
<line x1="210.0" y1="132.0" x2="399.4" y2="132.0" class="stroke-c" stroke-width="3"/>
<circle cx="399.4" cy="132.0" r="6" class="fill-c"/>
<text x="411.4" y="136.0" text-anchor="start" class="fig-small fig-bold">5.3%</text>
<text x="200.0" y="176.0" text-anchor="end" class="fig-small">96 blocks, width 12288</text>
<line x1="210.0" y1="172.0" x2="272.3" y2="172.0" class="stroke-c" stroke-width="3"/>
<circle cx="272.3" cy="172.0" r="6" class="fill-c"/>
<text x="284.3" y="176.0" text-anchor="start" class="fig-small fig-bold">0.37%</text>
<text x="375.0" y="236.0" text-anchor="middle" class="fig-small">embedding parameters as a share of all parameters (log scale)</text>
</svg>
Figure 2. The share of parameters that sit in the embedding tables, for four sizes with a 50,257-token vocabulary (100 for the toy). The tables grow with width, the weights that multiply grow with width squared and with depth, so the share falls from over a third to a fraction of a percent. For a big model the choice barely changes N, and for a small one it matters.
```

Does leaving the tables out matter? Figure 2 says it depends on size. For the toy model the tables hold 38 percent of all parameters, and for a model with 12 blocks and width 768 they still hold 32 percent, so including or excluding them changes $N$ by a large share. For a 48-block model of width 1600 the share is 5.3 percent, and for a 96-block model of width 12288 it is 0.37 percent. The tables grow with $d$ and the matrices grow with $d^2$ and with depth, so large models are almost all matrices.

> [!KEY]
> $N$ counts the weights that multiply, which are the attention and feed-forward matrices. The embedding tables are lookups, so they stay out of the compute formula, and they are counted once, not once per block.

## One weight costs six flops per token

Next, the cost of one weight. Figure 3 follows a single weight $w = 0.5$ that sees an input $x = 3$. Forward, it multiplies, $0.5 \times 3 = 1.5$, and the product is added into a running sum, so one multiply and one add make 2 flops. Every weight in a matrix does exactly this once for each token, so a pass over $D$ tokens costs $2ND$ flops.

```figure
<svg viewBox="0 0 600 284" role="img" aria-label="One weight with input 3 and weight 0.5: the forward pass costs two flops, and the backward pass sends gradients to the weight and to the input at two flops each">
<defs><marker id="c115-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="12.0" y="20.0" text-anchor="start" class="fig-bold">Forward: one weight, one input</text>
<rect x="30.0" y="40.0" width="70.0" height="40.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="65.0" y="58.0" text-anchor="middle" class="fig-small">input x</text>
<text x="65.0" y="74.0" text-anchor="middle" class="fig-bold">3</text>
<line x1="100.0" y1="60.0" x2="150.0" y2="60.0" class="stroke-muted" marker-end="url(#c115-arrow)"/>
<rect x="150.0" y="40.0" width="100.0" height="40.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="200.0" y="58.0" text-anchor="middle" class="fig-small">weight w</text>
<text x="200.0" y="74.0" text-anchor="middle" class="fig-bold">0.5</text>
<line x1="250.0" y1="60.0" x2="300.0" y2="60.0" class="stroke-muted" marker-end="url(#c115-arrow)"/>
<text x="275.0" y="52.0" text-anchor="middle" class="fig-small">multiply</text>
<rect x="300.0" y="40.0" width="100.0" height="40.0" rx="3" class="fill-sunk stroke-muted"/>
<text x="350.0" y="58.0" text-anchor="middle" class="fig-small">product</text>
<text x="350.0" y="74.0" text-anchor="middle" class="fig-bold">1.5</text>
<line x1="400.0" y1="60.0" x2="450.0" y2="60.0" class="stroke-muted" marker-end="url(#c115-arrow)"/>
<text x="425.0" y="52.0" text-anchor="middle" class="fig-small">add</text>
<rect x="450.0" y="40.0" width="120.0" height="40.0" rx="3" class="fill-sunk stroke-muted"/>
<text x="510.0" y="58.0" text-anchor="middle" class="fig-small">running sum</text>
<text x="510.0" y="74.0" text-anchor="middle" class="fig-bold">+ 1.5</text>
<text x="300.0" y="102.0" text-anchor="middle" class="fig-bold">1 multiply + 1 add = 2 flops for this weight and this token</text>
<text x="12.0" y="138.0" text-anchor="start" class="fig-bold">Backward: the gradient g = 2 arrives at the sum</text>
<line x1="510.0" y1="150.0" x2="510.0" y2="176.0" class="stroke-muted" />
<line x1="510.0" y1="176.0" x2="200.0" y2="176.0" class="stroke-muted" marker-end="url(#c115-arrow)"/>
<text x="355.0" y="170.0" text-anchor="middle" class="fig-small">to the weight: g × x = 2 × 3 = 6</text>
<text x="355.0" y="194.0" text-anchor="middle" class="fig-small fig-bold">1 multiply + 1 add = 2 flops</text>
<line x1="510.0" y1="150.0" x2="510.0" y2="224.0" class="stroke-muted" />
<line x1="510.0" y1="224.0" x2="65.0" y2="224.0" class="stroke-muted" marker-end="url(#c115-arrow)"/>
<text x="290.0" y="218.0" text-anchor="middle" class="fig-small">to the input: g × w = 2 × 0.5 = 1</text>
<text x="290.0" y="242.0" text-anchor="middle" class="fig-small fig-bold">1 multiply + 1 add = 2 flops</text>
<text x="300.0" y="270.0" text-anchor="middle" class="fig-bold">forward 2 + backward 4 = 6 flops per weight per token</text>
</svg>
Figure 3. One weight, one token. Forward, the weight multiplies the input and the product is added to a running sum, which is 2 flops. Backward, the same weight is asked for two gradients, one for itself and one for the input it saw, each a multiply and an add. Two plus four is six.
```

Training also needs gradients. Suppose the gradient $g = 2$ arrives at the sum, meaning the loss rises by 2 for every unit the sum rises. The weight needs two answers. To update itself it needs the gradient with respect to $w$, which is $g \times x = 6$, a multiply and an add once the results are summed over tokens. To pass the signal to the layer before it, it must hand back the gradient with respect to its input, $g \times w = 1$, another multiply and add. Each is a matrix multiply the same size as the forward one, so together the backward pass costs $4$ flops per weight per token, and training costs $2 + 4 = 6$.

$$
C = 6\,N\,D
$$

Read it aloud: compute is six times the weights that multiply, times the tokens. In code it is one line, and one detail matters:

```python
flops = 6.0 * float(params) * float(tokens)
```

The product of $6 \times 7 \times 10^{9} \times 3 \times 10^{11}$ is $1.26 \times 10^{22}$. A NumPy 64-bit integer overflows above about $9.2 \times 10^{18}$, so an expression like `6 * params * tokens` can wrap around to a wrong number when `params` and `tokens` arrive as `np.int64` values. Plain Python integers do not overflow, but converting to float first is safe for both.

The formula is an estimate. It leaves out the attention scores, the products of queries with keys and of weights with values, which cost about $T/(6d)$ of the matrix work for a sequence length $T$. For $d = 12288$ and $T = 2048$ that is under 3 percent, and for $d = 768$ and $T = 1024$ it is about a fifth, so the six is least accurate for models that are narrow next to their sequence length.

## From flops to days

Back to the opening. The 1.5-billion-parameter model, taken as 1,474,560,000 weights that multiply (48 blocks, width 1600), on 300 billion tokens costs $6 \times 1.4746 \times 10^{9} \times 3 \times 10^{11} = 2.65 \times 10^{21}$ flops. The round numbers 1.5 billion and 300 billion give $2.7 \times 10^{21}$. A chip that sustains $2 \times 10^{14}$ flops per second does that in $1.33 \times 10^{7}$ seconds, which is 154 days. A run of $10^{20}$ flops takes about 5.8 days on the same chip, and one of $10^{22}$ takes about 1.6 years. That is why large runs share the work among thousands of chips.

Memory is a separate bill. Each weight needs 4 bytes of its own, 4 for its gradient, and 8 for the two running averages that Adam keeps, which is 16 bytes in all, as the chapter on the cost of a run explained. A 1.5-billion-parameter model therefore needs about 24 gigabytes before it stores a single activation.

> [!WARNING]
> The six counts the weights that multiply. Using the total parameter count, tables included, overstates the compute of a 12-block model of width 768 by almost half (124 million weights instead of 85 million) and changes a 96-block model's by under half a percent.

> [!RECAP]
> - Count $N = L(4d^2 + 2 d\, d_{\text{ff}})$ for the weights that multiply. Count the embeddings, $(\text{vocab} + \text{positions})\, d$, once.
> - A lookup returns a row and multiplies nothing, so the tables stay out of $N$.
> - One weight does a multiply and an add forward, and two more such pairs backward, one for itself and one for its input, so 6 flops per weight per token.
> - $C = 6ND$, computed in floats. Divide by the chip's flops per second and by 86,400 for days.
> - The estimate skips attention scores, about $T/(6d)$ of the matrix work.

## Build it step by step

You write three functions, and the checks call them in this order.

1. `transformer_params(layers, d_model, d_ff, vocab, seq_len)` returns a dict of integers with the keys `"attention"`, `"feedforward"`, `"embedding"`, `"non_embedding"`, and `"total"`. Attention and feed-forward are summed over all the blocks. The toy sizes should give 512, 1024, 928, 1536, and 2464.
2. `training_flops(params, tokens)` returns `6.0 * params * tokens` as a float. Call `float()` on each input first, because the mistake that fails the overflow check is multiplying `np.int64` values.
3. `training_days(flops, flops_per_second=2e14)` divides flops by the speed to get seconds, then by 86,400 to get days. At the default speed $10^{20}$ flops should give 5.79 days.

The shape of the first function:

```python
def transformer_params(layers, d_model, d_ff, vocab, seq_len):
    # attention and feed-forward per block times layers; embeddings once
    return {"attention": ..., "feedforward": ..., "embedding": ...,
            "non_embedding": ..., "total": ...}
```

> [!TIP]
> Use `non_embedding` as the `params` argument of `training_flops`. If your count for 48 blocks, width 1600, and feed-forward width 6400 is not 1,474,560,000, the bug is in the first function.

## The exercise

You implement `transformer_params`, `training_flops`, and `training_days` in `starter.py`. The checks verify the toy model's hand count, that embeddings are counted once while the weights that multiply scale with depth, that doubling `d_model` and `d_ff` quadruples them and that $d_{\text{ff}} = 4d$ gives $12 L d^2$, that `training_flops` returns $6ND$ and survives `np.int64` inputs, that `training_days` gives 5.79 days for $10^{20}$ flops, and that a 48-block configuration runs through all three functions to 154 days.
