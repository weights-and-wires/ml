---
title: "Spending a compute budget well"
slug: 104-compute-optimal-models
part: "Part 7 — Large language models"
summary: Count the weights that multiply, price training at six flops per weight per token, then split a fixed budget between model size and data.
minutes: 55
gpu: false
objectives:
  - Count a transformer's weights by component, and separate the ones that multiply from the embedding tables.
  - Derive the six in C = 6ND from one weight, one input, and one gradient, and turn a budget into days on one accelerator.
  - Explain why, at a fixed budget, the loss along the model-versus-data trade-off is a U, and derive its bottom by equating the marginal gains.
  - Say when the best ratio of tokens to parameters is constant, and why heavy serving pushes toward a smaller model trained on more tokens.
prereqs:
  - 103-loss-from-size-and-data
  - 053-width-depth-and-cost
  - 094-the-residual-block
  - 098-cost-of-training
exercise: 104-compute-optimal-models
---

# Spending a compute budget well

You have $10^{20}$ flops, which is about six days on a chip that does 200 trillion per second. You can train a huge model for a short time, a tiny model for a long time, or something in between. The loss law from the last chapter says a bigger model helps, and more text helps too, but it never said how to choose when the budget is fixed.

First you need the price of a model. A transformer has attention matrices, feed-forward matrices, and big lookup tables, and if you count them all as equal you overstate the cost of a small model by half. Then comes the split. The obvious plan is to build the biggest model the budget allows, but compute is the product of model size and tokens, so a bigger model at fixed compute leaves fewer tokens, and the data penalty rises faster than the size penalty falls. A small model on mountains of text fails from the other side.

This chapter counts the weights that do arithmetic, prices one weight per token, and finds the one split where the loss is lowest. For these constants it comes out at about 20 tokens per parameter.

## What you already know

- A flop is one floating-point operation, one multiply or one add, so a multiply-add pair costs 2.
- A transformer block is attention followed by a feed-forward network, each added back onto a residual stream.
- A gradient is how much a result changes when you nudge one input, and backpropagation computes one per weight.
- The loss is $E + A/N^{\alpha} + B/D^{\beta}$, a floor plus a size penalty and a data penalty, where $N$ counts the parameters that matter and $D$ the tokens.

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
Figure 1. The toy model: 2 blocks, width 8, feed-forward width 32, vocabulary 100, and 16 positions. The orange boxes are weights that multiply, and they make N = 1,536. The green tables only return a row, so their 928 numbers stay out of N. Box areas match the sizes of the matrices.
```

Figure 1 draws a toy model with two blocks, width $d = 8$, and a feed-forward width of $d_{\text{ff}} = 32$. Each block has four $d \times d$ attention matrices (query, key, value, and output) and two feed-forward matrices, one that widens from $d$ to $d_{\text{ff}}$ and one that narrows back. On the right are two lookup tables, one with a row per token in the vocabulary and one with a row per position.

The two groups behave differently. Every token passes through every matrix on the left and gets multiplied by each, so each of those weights does arithmetic for each token. A lookup table only returns a row: token number 57 fetches row 57 and nothing is multiplied. That is why the tables stay outside $N$.

Count the toy model. Attention holds $4 \times 8^2 = 256$ weights per block, and the feed-forward network holds $2 \times 8 \times 32 = 512$. A block holds $768$, and two blocks hold $1{,}536$. The token table is $100 \times 8 = 800$, and the position table is $16 \times 8 = 128$, so the tables hold $928$. The output layer reuses the token table, a trick called weight tying, so it adds nothing. The total is $2{,}464$, and $N = 1{,}536$. In symbols, with $L$ blocks:

$$
N = L\,\bigl(4 d^{2} + 2\, d\, d_{\text{ff}}\bigr),
\qquad \text{embeddings} = (\text{vocab} + \text{positions})\, d
$$

Read it aloud: each block has four $d \times d$ attention matrices and two $d \times d_{\text{ff}}$ feed-forward matrices, and the tables hold one row of width $d$ per token and per position. The embeddings appear once, the blocks $L$ times. When $d_{\text{ff}} = 4d$ a block holds $12 d^2$ weights, so $N = 12 L d^2$. The count leaves out biases and layer-norm scales, which are smaller by a factor of $d$, and assumes a plain two-matrix feed-forward network.

Leaving the tables out matters at small sizes: they hold 38 percent of the toy model's parameters and 32 percent of a 12-block, width-768 model, but only 0.37 percent of a 96-block, width-12288 model, because the tables grow with $d$ while the matrices grow with $d^2$ and with depth.

> [!KEY]
> $N$ counts the weights that multiply, which are the attention and feed-forward matrices. The embedding tables are lookups, so they stay out of the compute formula, and they are counted once, not once per block.

## One weight costs six flops per token

Figure 2 follows a single weight $w = 0.5$ that sees an input $x = 3$. Forward, it multiplies, $0.5 \times 3 = 1.5$, and the product is added into a running sum, so one multiply and one add make 2 flops. Every weight does this once for each token, so a pass over $D$ tokens costs $2ND$ flops.

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
Figure 2. One weight, one token. Forward, the weight multiplies the input and the product is added to a running sum, which is 2 flops. Backward, the same weight is asked for two gradients, one for itself and one for the input it saw, each a multiply and an add. Two plus four is six.
```

Training also needs gradients. Suppose the gradient $g = 2$ arrives at the sum. To update itself, the weight needs the gradient with respect to $w$, which is $g \times x = 6$. To pass the signal to the layer before it, it must hand back the gradient with respect to its input, $g \times w = 1$. Each is a multiply and an add of the same size as the forward one, so the backward pass costs $4$ flops per weight per token, and training costs $2 + 4 = 6$.

$$
C = 6\,N\,D
$$

Read it aloud: compute is six times the weights that multiply, times the tokens. In code, one detail matters:

```python
flops = 6.0 * float(params) * float(tokens)
```

A NumPy 64-bit integer overflows above about $9.2 \times 10^{18}$, so `6 * params * tokens` can wrap around to a wrong number when the inputs arrive as `np.int64`. Converting to float first is safe. The formula is an estimate that skips the attention scores, about $T/(6d)$ of the matrix work for a sequence length $T$.

Divide by the chip's speed to get time. A model with 1,474,560,000 weights that multiply (48 blocks, width 1600) on 300 billion tokens costs $6 \times 1.4746 \times 10^{9} \times 3 \times 10^{11} = 2.65 \times 10^{21}$ flops. A chip that sustains $2 \times 10^{14}$ flops per second does that in $1.33 \times 10^{7}$ seconds, which is 154 days. A run of $10^{20}$ flops takes 5.8 days on the same chip.

> [!WARNING]
> The six counts the weights that multiply. Using the total parameter count, tables included, overstates the compute of a 12-block model of width 768 by almost half (124 million weights instead of 85 million).

## Spending the budget: a U along the budget line

Now spend the budget. Because $D = C/(6N)$, a larger model means fewer tokens. Figure 3 fixes the budget at $1.2 \times 10^{20}$ flops, which is 1 billion parameters on 20 billion tokens, and slides along it.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Size penalty falling, data penalty rising, and their sum making a U shape along a line of fixed compute, with the lowest point at 1 billion parameters">
<line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-muted"/><line x1="70.0" y1="36.0" x2="70.0" y2="236.0" class="stroke-muted"/><line x1="120.9" y1="36.0" x2="120.9" y2="236.0" class="stroke-grid"/><text x="120.9" y="252.0" text-anchor="middle" class="fig-small">100M</text><line x1="290.0" y1="36.0" x2="290.0" y2="236.0" class="stroke-grid"/><text x="290.0" y="252.0" text-anchor="middle" class="fig-small">1B</text><line x1="459.1" y1="36.0" x2="459.1" y2="236.0" class="stroke-grid"/><text x="459.1" y="252.0" text-anchor="middle" class="fig-small">10B</text><line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-grid"/><text x="64.0" y="240.0" text-anchor="end" class="fig-small">0</text><line x1="70.0" y1="156.0" x2="510.0" y2="156.0" class="stroke-grid"/><text x="64.0" y="160.0" text-anchor="end" class="fig-small">0.5</text><line x1="70.0" y1="76.0" x2="510.0" y2="76.0" class="stroke-grid"/><text x="64.0" y="80.0" text-anchor="end" class="fig-small">1</text>
<polyline points="70.0,81.6 71.5,82.7 72.9,83.7 74.4,84.8 75.9,85.8 77.4,86.8 78.8,87.8 80.3,88.8 81.8,89.8 83.2,90.8 84.7,91.8 86.2,92.8 87.7,93.8 89.1,94.7 90.6,95.7 92.1,96.6 93.5,97.6 95.0,98.5 96.5,99.5 98.0,100.4 99.4,101.3 100.9,102.2 102.4,103.1 103.8,104.0 105.3,104.9 106.8,105.8 108.3,106.7 109.7,107.6 111.2,108.4 112.7,109.3 114.1,110.2 115.6,111.0 117.1,111.9 118.6,112.7 120.0,113.6 121.5,114.4 123.0,115.2 124.4,116.0 125.9,116.9 127.4,117.7 128.9,118.5 130.3,119.3 131.8,120.1 133.3,120.8 134.7,121.6 136.2,122.4 137.7,123.2 139.2,123.9 140.6,124.7 142.1,125.5 143.6,126.2 145.1,126.9 146.5,127.7 148.0,128.4 149.5,129.2 150.9,129.9 152.4,130.6 153.9,131.3 155.4,132.0 156.8,132.7 158.3,133.4 159.8,134.1 161.2,134.8 162.7,135.5 164.2,136.2 165.7,136.9 167.1,137.5 168.6,138.2 170.1,138.9 171.5,139.5 173.0,140.2 174.5,140.8 176.0,141.5 177.4,142.1 178.9,142.8 180.4,143.4 181.8,144.0 183.3,144.7 184.8,145.3 186.3,145.9 187.7,146.5 189.2,147.1 190.7,147.7 192.1,148.3 193.6,148.9 195.1,149.5 196.6,150.1 198.0,150.7 199.5,151.2 201.0,151.8 202.4,152.4 203.9,153.0 205.4,153.5 206.9,154.1 208.3,154.6 209.8,155.2 211.3,155.7 212.7,156.3 214.2,156.8 215.7,157.4 217.2,157.9 218.6,158.4 220.1,159.0 221.6,159.5 223.0,160.0 224.5,160.5 226.0,161.0 227.5,161.5 228.9,162.0 230.4,162.5 231.9,163.0 233.3,163.5 234.8,164.0 236.3,164.5 237.8,165.0 239.2,165.5 240.7,166.0 242.2,166.4 243.6,166.9 245.1,167.4 246.6,167.8 248.1,168.3 249.5,168.8 251.0,169.2 252.5,169.7 253.9,170.1 255.4,170.6 256.9,171.0 258.4,171.5 259.8,171.9 261.3,172.3 262.8,172.8 264.2,173.2 265.7,173.6 267.2,174.0 268.7,174.5 270.1,174.9 271.6,175.3 273.1,175.7 274.5,176.1 276.0,176.5 277.5,176.9 279.0,177.3 280.4,177.7 281.9,178.1 283.4,178.5 284.8,178.9 286.3,179.3 287.8,179.7 289.3,180.1 290.7,180.4 292.2,180.8 293.7,181.2 295.2,181.6 296.6,181.9 298.1,182.3 299.6,182.7 301.0,183.0 302.5,183.4 304.0,183.8 305.5,184.1 306.9,184.5 308.4,184.8 309.9,185.2 311.3,185.5 312.8,185.8 314.3,186.2 315.8,186.5 317.2,186.9 318.7,187.2 320.2,187.5 321.6,187.9 323.1,188.2 324.6,188.5 326.1,188.8 327.5,189.1 329.0,189.5 330.5,189.8 331.9,190.1 333.4,190.4 334.9,190.7 336.4,191.0 337.8,191.3 339.3,191.6 340.8,191.9 342.2,192.2 343.7,192.5 345.2,192.8 346.7,193.1 348.1,193.4 349.6,193.7 351.1,194.0 352.5,194.3 354.0,194.6 355.5,194.8 357.0,195.1 358.4,195.4 359.9,195.7 361.4,195.9 362.8,196.2 364.3,196.5 365.8,196.8 367.3,197.0 368.7,197.3 370.2,197.5 371.7,197.8 373.1,198.1 374.6,198.3 376.1,198.6 377.6,198.8 379.0,199.1 380.5,199.3 382.0,199.6 383.4,199.8 384.9,200.1 386.4,200.3 387.9,200.6 389.3,200.8 390.8,201.0 392.3,201.3 393.7,201.5 395.2,201.8 396.7,202.0 398.2,202.2 399.6,202.4 401.1,202.7 402.6,202.9 404.0,203.1 405.5,203.3 407.0,203.6 408.5,203.8 409.9,204.0 411.4,204.2 412.9,204.4 414.3,204.7 415.8,204.9 417.3,205.1 418.8,205.3 420.2,205.5 421.7,205.7 423.2,205.9 424.6,206.1 426.1,206.3 427.6,206.5 429.1,206.7 430.5,206.9 432.0,207.1 433.5,207.3 434.9,207.5 436.4,207.7 437.9,207.9 439.4,208.1 440.8,208.3 442.3,208.5 443.8,208.6 445.3,208.8 446.7,209.0 448.2,209.2 449.7,209.4 451.1,209.6 452.6,209.7 454.1,209.9 455.6,210.1 457.0,210.3 458.5,210.5 460.0,210.6 461.4,210.8 462.9,211.0 464.4,211.1 465.9,211.3 467.3,211.5 468.8,211.6 470.3,211.8 471.7,212.0 473.2,212.1 474.7,212.3 476.2,212.5 477.6,212.6 479.1,212.8 480.6,212.9 482.0,213.1 483.5,213.2 485.0,213.4 486.5,213.6 487.9,213.7 489.4,213.9 490.9,214.0 492.3,214.2 493.8,214.3 495.3,214.5 496.8,214.6 498.2,214.7 499.7,214.9 501.2,215.0 502.6,215.2 504.1,215.3 505.6,215.5 507.1,215.6 508.5,215.7 510.0,215.9" fill="none" class="stroke-a" stroke-width="2"/>
<polyline points="70.0,216.0 71.5,215.9 72.9,215.7 74.4,215.6 75.9,215.5 77.4,215.3 78.8,215.2 80.3,215.0 81.8,214.9 83.2,214.7 84.7,214.6 86.2,214.5 87.7,214.3 89.1,214.2 90.6,214.0 92.1,213.9 93.5,213.7 95.0,213.6 96.5,213.4 98.0,213.2 99.4,213.1 100.9,212.9 102.4,212.8 103.8,212.6 105.3,212.5 106.8,212.3 108.3,212.1 109.7,212.0 111.2,211.8 112.7,211.6 114.1,211.5 115.6,211.3 117.1,211.1 118.6,211.0 120.0,210.8 121.5,210.6 123.0,210.5 124.4,210.3 125.9,210.1 127.4,209.9 128.9,209.7 130.3,209.6 131.8,209.4 133.3,209.2 134.7,209.0 136.2,208.8 137.7,208.7 139.2,208.5 140.6,208.3 142.1,208.1 143.6,207.9 145.1,207.7 146.5,207.5 148.0,207.3 149.5,207.1 150.9,206.9 152.4,206.7 153.9,206.5 155.4,206.3 156.8,206.1 158.3,205.9 159.8,205.7 161.2,205.5 162.7,205.3 164.2,205.1 165.7,204.9 167.1,204.7 168.6,204.4 170.1,204.2 171.5,204.0 173.0,203.8 174.5,203.6 176.0,203.4 177.4,203.1 178.9,202.9 180.4,202.7 181.8,202.5 183.3,202.2 184.8,202.0 186.3,201.8 187.7,201.5 189.2,201.3 190.7,201.1 192.1,200.8 193.6,200.6 195.1,200.3 196.6,200.1 198.0,199.8 199.5,199.6 201.0,199.3 202.4,199.1 203.9,198.8 205.4,198.6 206.9,198.3 208.3,198.1 209.8,197.8 211.3,197.6 212.7,197.3 214.2,197.0 215.7,196.8 217.2,196.5 218.6,196.2 220.1,195.9 221.6,195.7 223.0,195.4 224.5,195.1 226.0,194.8 227.5,194.6 228.9,194.3 230.4,194.0 231.9,193.7 233.3,193.4 234.8,193.1 236.3,192.8 237.8,192.5 239.2,192.2 240.7,191.9 242.2,191.6 243.6,191.3 245.1,191.0 246.6,190.7 248.1,190.4 249.5,190.1 251.0,189.8 252.5,189.5 253.9,189.2 255.4,188.8 256.9,188.5 258.4,188.2 259.8,187.9 261.3,187.5 262.8,187.2 264.2,186.9 265.7,186.5 267.2,186.2 268.7,185.9 270.1,185.5 271.6,185.2 273.1,184.8 274.5,184.5 276.0,184.1 277.5,183.8 279.0,183.4 280.4,183.0 281.9,182.7 283.4,182.3 284.8,181.9 286.3,181.6 287.8,181.2 289.3,180.8 290.7,180.5 292.2,180.1 293.7,179.7 295.2,179.3 296.6,178.9 298.1,178.5 299.6,178.1 301.0,177.7 302.5,177.3 304.0,176.9 305.5,176.5 306.9,176.1 308.4,175.7 309.9,175.3 311.3,174.9 312.8,174.5 314.3,174.1 315.8,173.6 317.2,173.2 318.7,172.8 320.2,172.3 321.6,171.9 323.1,171.5 324.6,171.0 326.1,170.6 327.5,170.1 329.0,169.7 330.5,169.2 331.9,168.8 333.4,168.3 334.9,167.9 336.4,167.4 337.8,166.9 339.3,166.5 340.8,166.0 342.2,165.5 343.7,165.0 345.2,164.5 346.7,164.0 348.1,163.6 349.6,163.1 351.1,162.6 352.5,162.1 354.0,161.5 355.5,161.0 357.0,160.5 358.4,160.0 359.9,159.5 361.4,159.0 362.8,158.4 364.3,157.9 365.8,157.4 367.3,156.8 368.7,156.3 370.2,155.8 371.7,155.2 373.1,154.7 374.6,154.1 376.1,153.5 377.6,153.0 379.0,152.4 380.5,151.8 382.0,151.3 383.4,150.7 384.9,150.1 386.4,149.5 387.9,148.9 389.3,148.3 390.8,147.7 392.3,147.1 393.7,146.5 395.2,145.9 396.7,145.3 398.2,144.7 399.6,144.0 401.1,143.4 402.6,142.8 404.0,142.1 405.5,141.5 407.0,140.9 408.5,140.2 409.9,139.5 411.4,138.9 412.9,138.2 414.3,137.6 415.8,136.9 417.3,136.2 418.8,135.5 420.2,134.8 421.7,134.1 423.2,133.4 424.6,132.7 426.1,132.0 427.6,131.3 429.1,130.6 430.5,129.9 432.0,129.2 433.5,128.4 434.9,127.7 436.4,127.0 437.9,126.2 439.4,125.5 440.8,124.7 442.3,124.0 443.8,123.2 445.3,122.4 446.7,121.6 448.2,120.9 449.7,120.1 451.1,119.3 452.6,118.5 454.1,117.7 455.6,116.9 457.0,116.1 458.5,115.2 460.0,114.4 461.4,113.6 462.9,112.7 464.4,111.9 465.9,111.0 467.3,110.2 468.8,109.3 470.3,108.5 471.7,107.6 473.2,106.7 474.7,105.8 476.2,104.9 477.6,104.0 479.1,103.1 480.6,102.2 482.0,101.3 483.5,100.4 485.0,99.5 486.5,98.5 487.9,97.6 489.4,96.7 490.9,95.7 492.3,94.7 493.8,93.8 495.3,92.8 496.8,91.8 498.2,90.8 499.7,89.8 501.2,88.9 502.6,87.8 504.1,86.8 505.6,85.8 507.1,84.8 508.5,83.8 510.0,82.7" fill="none" class="stroke-d" stroke-width="2"/>
<polyline points="70.0,61.7 71.5,62.6 72.9,63.5 74.4,64.4 75.9,65.2 77.4,66.1 78.8,67.0 80.3,67.9 81.8,68.7 83.2,69.6 84.7,70.4 86.2,71.2 87.7,72.1 89.1,72.9 90.6,73.7 92.1,74.5 93.5,75.3 95.0,76.1 96.5,76.9 98.0,77.6 99.4,78.4 100.9,79.2 102.4,79.9 103.8,80.6 105.3,81.4 106.8,82.1 108.3,82.8 109.7,83.6 111.2,84.3 112.7,85.0 114.1,85.7 115.6,86.3 117.1,87.0 118.6,87.7 120.0,88.4 121.5,89.0 123.0,89.7 124.4,90.3 125.9,91.0 127.4,91.6 128.9,92.2 130.3,92.8 131.8,93.4 133.3,94.0 134.7,94.6 136.2,95.2 137.7,95.8 139.2,96.4 140.6,97.0 142.1,97.5 143.6,98.1 145.1,98.7 146.5,99.2 148.0,99.7 149.5,100.3 150.9,100.8 152.4,101.3 153.9,101.8 155.4,102.4 156.8,102.9 158.3,103.4 159.8,103.8 161.2,104.3 162.7,104.8 164.2,105.3 165.7,105.7 167.1,106.2 168.6,106.7 170.1,107.1 171.5,107.5 173.0,108.0 174.5,108.4 176.0,108.8 177.4,109.3 178.9,109.7 180.4,110.1 181.8,110.5 183.3,110.9 184.8,111.3 186.3,111.6 187.7,112.0 189.2,112.4 190.7,112.8 192.1,113.1 193.6,113.5 195.1,113.8 196.6,114.2 198.0,114.5 199.5,114.8 201.0,115.2 202.4,115.5 203.9,115.8 205.4,116.1 206.9,116.4 208.3,116.7 209.8,117.0 211.3,117.3 212.7,117.6 214.2,117.9 215.7,118.1 217.2,118.4 218.6,118.7 220.1,118.9 221.6,119.2 223.0,119.4 224.5,119.6 226.0,119.9 227.5,120.1 228.9,120.3 230.4,120.5 231.9,120.8 233.3,121.0 234.8,121.2 236.3,121.4 237.8,121.5 239.2,121.7 240.7,121.9 242.2,122.1 243.6,122.3 245.1,122.4 246.6,122.6 248.1,122.7 249.5,122.9 251.0,123.0 252.5,123.2 253.9,123.3 255.4,123.4 256.9,123.5 258.4,123.7 259.8,123.8 261.3,123.9 262.8,124.0 264.2,124.1 265.7,124.2 267.2,124.2 268.7,124.3 270.1,124.4 271.6,124.5 273.1,124.5 274.5,124.6 276.0,124.6 277.5,124.7 279.0,124.7 280.4,124.8 281.9,124.8 283.4,124.8 284.8,124.9 286.3,124.9 287.8,124.9 289.3,124.9 290.7,124.9 292.2,124.9 293.7,124.9 295.2,124.9 296.6,124.9 298.1,124.8 299.6,124.8 301.0,124.8 302.5,124.7 304.0,124.7 305.5,124.6 306.9,124.6 308.4,124.5 309.9,124.5 311.3,124.4 312.8,124.3 314.3,124.2 315.8,124.2 317.2,124.1 318.7,124.0 320.2,123.9 321.6,123.8 323.1,123.7 324.6,123.5 326.1,123.4 327.5,123.3 329.0,123.2 330.5,123.0 331.9,122.9 333.4,122.7 334.9,122.6 336.4,122.4 337.8,122.3 339.3,122.1 340.8,121.9 342.2,121.7 343.7,121.5 345.2,121.4 346.7,121.2 348.1,121.0 349.6,120.8 351.1,120.5 352.5,120.3 354.0,120.1 355.5,119.9 357.0,119.6 358.4,119.4 359.9,119.2 361.4,118.9 362.8,118.7 364.3,118.4 365.8,118.1 367.3,117.9 368.7,117.6 370.2,117.3 371.7,117.0 373.1,116.7 374.6,116.4 376.1,116.1 377.6,115.8 379.0,115.5 380.5,115.2 382.0,114.8 383.4,114.5 384.9,114.2 386.4,113.8 387.9,113.5 389.3,113.1 390.8,112.8 392.3,112.4 393.7,112.0 395.2,111.7 396.7,111.3 398.2,110.9 399.6,110.5 401.1,110.1 402.6,109.7 404.0,109.3 405.5,108.8 407.0,108.4 408.5,108.0 409.9,107.6 411.4,107.1 412.9,106.7 414.3,106.2 415.8,105.8 417.3,105.3 418.8,104.8 420.2,104.3 421.7,103.9 423.2,103.4 424.6,102.9 426.1,102.4 427.6,101.9 429.1,101.3 430.5,100.8 432.0,100.3 433.5,99.8 434.9,99.2 436.4,98.7 437.9,98.1 439.4,97.6 440.8,97.0 442.3,96.4 443.8,95.8 445.3,95.2 446.7,94.7 448.2,94.1 449.7,93.5 451.1,92.8 452.6,92.2 454.1,91.6 455.6,91.0 457.0,90.3 458.5,89.7 460.0,89.0 461.4,88.4 462.9,87.7 464.4,87.0 465.9,86.4 467.3,85.7 468.8,85.0 470.3,84.3 471.7,83.6 473.2,82.8 474.7,82.1 476.2,81.4 477.6,80.7 479.1,79.9 480.6,79.2 482.0,78.4 483.5,77.6 485.0,76.9 486.5,76.1 487.9,75.3 489.4,74.5 490.9,73.7 492.3,72.9 493.8,72.1 495.3,71.3 496.8,70.4 498.2,69.6 499.7,68.7 501.2,67.9 502.6,67.0 504.1,66.1 505.6,65.3 507.1,64.4 508.5,63.5 510.0,62.6" fill="none" class="stroke-b" stroke-width="3"/>
<circle cx="290.8" cy="124.9" r="6" class="fill-b"/>
<line x1="290.8" y1="36.0" x2="290.8" y2="236.0" class="stroke-muted dashed" />
<text x="290.8" y="30.0" text-anchor="middle" class="fig-small fig-bold">best split: N* = 1.01B, D* = 19.8B</text>
<text x="78.3" y="48.8" text-anchor="start" class="fig-bold fig-small">model too small</text>
<text x="506.0" y="48.8" text-anchor="end" class="fig-bold fig-small">too few tokens</text>
<text x="83.4" y="152.8" text-anchor="start" class="fig-small fig-bold fill-a">size penalty</text>
<text x="508.1" y="152.8" text-anchor="end" class="fig-small fig-bold fill-d">data penalty</text>
<text x="300.8" y="103.6" text-anchor="start" class="fig-small fig-bold fill-b">sum 0.694</text>
<text x="290.0" y="272.0" text-anchor="middle" class="fig-small">model size N, with D = compute / 6N so that 6ND = 1.2e20</text>
<text x="12.0" y="20.0" text-anchor="start" class="fig-small">loss above the floor</text>
</svg>
Figure 3. Fix the budget at 1.2e20 flops and slide along it. A larger model leaves fewer tokens, so the size penalty (blue) falls while the data penalty (purple) rises. The sum (orange) is a U. The bottom of the U is where the two penalties are equal, because the exponents are equal here.
```

On the left the model is small and the data is plentiful, so the size penalty (blue) is big and the data penalty (purple) is small. Moving right, the blue line falls and the purple line rises, so the sum above the floor (orange) falls, bottoms out, and climbs again. The left arm is a model that is too small, and the right arm is a model that sees too few tokens. The bottom of the U sits at $N^{*} = 1.01$ billion parameters and $D^{*} = 19.8$ billion tokens, where the two penalties cross at 0.347 each.

Find it exactly. Growing $N$ by 1 percent means shrinking $D$ by about 1 percent, so one side pays exactly what the other receives. The size penalty falls by about $\alpha$ percent of itself, and the data penalty rises by about $\beta$ percent of itself. If the fall is bigger, keep sliding toward a larger model, and if the rise is bigger, slide back. You stop where the two are equal:

$$
\frac{\alpha A}{N^{\alpha}} = \frac{\beta B}{D^{\beta}}
$$

Read it aloud: the loss that 1 percent more model buys equals the loss that 1 percent more data buys, so no bargain is left on either side. Spend the whole budget by putting $D = C/(6N)$ on the right side, so $\beta B / D^{\beta} = \beta B\,(6N/C)^{\beta}$, and collect the powers of $N$:

$$
N^{\alpha + \beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}
$$

Take the $(\alpha+\beta)$-th root. Writing $a = \beta/(\alpha+\beta)$ and $G = (\alpha A/\beta B)^{1/(\alpha+\beta)}$, the best size and the matching tokens are

$$
N^{*} = G \cdot 6^{-a} \cdot C^{a}, \qquad D^{*} = \frac{C}{6N^{*}}
$$

Now the numbers at $10^{20}$ flops, with $A = 400$, $B = 1100$, and $\alpha = \beta = 0.34$. Then $a = 0.5$ and $G = (400/1100)^{1/0.68} = 0.2259$. With $6^{-0.5} = 0.4082$, $N^{*} = 0.2259 \times 0.4082 \times 10^{10} = 9.22 \times 10^{8}$ and $D^{*} = 10^{20} / (6 \times 9.22 \times 10^{8}) = 1.81 \times 10^{10}$. In code:

```python
a = beta / (alpha + beta)
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
n_star = G * 6.0 ** (-a) * compute ** a
d_star = compute / (6.0 * n_star)
```

A search can check the algebra: try many sizes $N$, set $D = C/(6N)$ for each, and keep the lowest loss. At $10^{20}$ flops, a grid of 2000 sizes lands within 0.2 percent of $9.22 \times 10^{8}$.

> [!KEY]
> At a fixed budget the loss along the size-versus-data line is a U, and its bottom is where 1 percent more model and 1 percent more data are worth the same. For these constants that is $N^{*} = G\,6^{-a}\,C^{a}$.

## What the optimum looks like

For these constants $D^{*}/N^{*} = 19.6$, about 20 tokens for every parameter, at every budget. That holds only because $\alpha = \beta$. Then $a = 0.5$, both $N^{*}$ and $D^{*}$ grow as $\sqrt{C}$, and the ratio $1/G^{2}$ does not change. When the exponents differ, $N^{*}$ grows as $C^{a}$ and $D^{*}$ as $C^{1-a}$, so the ratio drifts as $C^{1-2a}$. With a data exponent of 0.28 in place of 0.34, $a$ is about 0.45, and the ratio grows by about a quarter for each tenfold increase in compute.

The balanced split is worth the trouble. At $1.2 \times 10^{20}$ flops it has loss 2.3844, while a model 4 times larger on a quarter of the tokens, with the same compute, has loss $1.69 + 0.2175 + 0.5542 = 2.4617$.

Published fits disagree on the exponents, so a scaling law is a fit to experiments, not a law of physics. Still, the Chinchilla study's 70-billion-parameter model on 1.4 trillion tokens beat a 280-billion-parameter model on 300 billion tokens at about the same compute.

## The cost of serving

The best split minimises training compute for a given loss. It ignores what happens after training. Each token a deployed model reads or writes costs about $2N$ flops, the forward pass alone.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Total flops against tokens served for a balanced model and a model half its size trained to the same loss, with the lines crossing at 28.6 billion tokens">
<line x1="80.0" y1="226.0" x2="500.0" y2="226.0" class="stroke-muted"/><line x1="80.0" y1="36.0" x2="80.0" y2="226.0" class="stroke-muted"/><line x1="80.0" y1="36.0" x2="80.0" y2="226.0" class="stroke-grid"/><text x="80.0" y="242.0" text-anchor="middle" class="fig-small">0</text><line x1="164.0" y1="36.0" x2="164.0" y2="226.0" class="stroke-grid"/><text x="164.0" y="242.0" text-anchor="middle" class="fig-small">20B</text><line x1="248.0" y1="36.0" x2="248.0" y2="226.0" class="stroke-grid"/><text x="248.0" y="242.0" text-anchor="middle" class="fig-small">40B</text><line x1="332.0" y1="36.0" x2="332.0" y2="226.0" class="stroke-grid"/><text x="332.0" y="242.0" text-anchor="middle" class="fig-small">60B</text><line x1="416.0" y1="36.0" x2="416.0" y2="226.0" class="stroke-grid"/><text x="416.0" y="242.0" text-anchor="middle" class="fig-small">80B</text><line x1="500.0" y1="36.0" x2="500.0" y2="226.0" class="stroke-grid"/><text x="500.0" y="242.0" text-anchor="middle" class="fig-small">100B</text><line x1="80.0" y1="178.5" x2="500.0" y2="178.5" class="stroke-grid"/><text x="74.0" y="182.5" text-anchor="end" class="fig-small">1e20</text><line x1="80.0" y1="131.0" x2="500.0" y2="131.0" class="stroke-grid"/><text x="74.0" y="135.0" text-anchor="end" class="fig-small">2e20</text><line x1="80.0" y1="83.5" x2="500.0" y2="83.5" class="stroke-grid"/><text x="74.0" y="87.5" text-anchor="end" class="fig-small">3e20</text><line x1="80.0" y1="36.0" x2="500.0" y2="36.0" class="stroke-grid"/><text x="74.0" y="40.0" text-anchor="end" class="fig-small">4e20</text>
<polyline points="80.0,169.0 500.0,73.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="80.0,155.3 500.0,107.3" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="199.9" cy="141.6" r="5" class="fill-ink"/>
<line x1="199.9" y1="141.6" x2="199.9" y2="226.0" class="stroke-muted dashed" />
<text x="207.9" y="218.0" text-anchor="start" class="fig-small fig-bold">break-even: 28.6B served tokens</text>
<text x="496.0" y="154.8" text-anchor="end" class="fig-small fig-bold fill-a">balanced: N = 1.01B, D = 20B</text>
<text x="496.0" y="170.8" text-anchor="end" class="fig-small fig-bold fill-b">half size: N = 0.51B, D = 49B</text>
<text x="290.0" y="262.0" text-anchor="middle" class="fig-small">tokens served after training</text>
<text x="12.0" y="20.0" text-anchor="start" class="fig-small">flops, training plus serving</text>
</svg>
Figure 4. Both models reach the same loss. The half-size model starts higher, because it needed 49 billion training tokens, 24% more compute than the balanced model. Each token served costs it half as much, so its line is flatter and crosses the other at 28.6 billion served tokens. Beyond that, it is the cheaper model.
```

Take the balanced model at $1.2 \times 10^{20}$ flops, with $N^{*} = 1.01 \times 10^{9}$. A model half the size can reach the same loss of 2.3844 if you train it on enough text: solving the loss law for $D$ gives $4.9 \times 10^{10}$ tokens, which is 97 tokens per parameter. Training it costs 24 percent more than the balanced model, but serving costs half as much per token, so the lifetime totals cross:

$$
\text{total} = 6\,N\,D_{\text{train}} + 2\,N\,D_{\text{served}}
$$

Read it aloud: training at six flops per weight per token, plus serving at two. The break-even is $2.86 \times 10^{10}$ served tokens. After a trillion served tokens, the totals are $2.1 \times 10^{21}$ flops for the balanced model and $1.2 \times 10^{21}$ for the half-size one, so a heavily used model is often trained far beyond 20 tokens per parameter.

> [!RECAP]
> Before you build:
>
> - Count $N = L(4d^2 + 2 d\, d_{\text{ff}})$ for the weights that multiply, and the embeddings once. Training costs $C = 6ND$, computed in floats.
> - At a fixed budget, $D = C/(6N)$, so a larger model means fewer tokens, and the loss traces a U.
> - The bottom of the U is where $\alpha A/N^{\alpha} = \beta B/D^{\beta}$, which gives $N^{*} = G\,6^{-a}\,C^{a}$ and $D^{*} = C/(6N^{*})$. With equal exponents the ratio $D^{*}/N^{*}$ is 19.6.
> - Serving costs $2N$ per token, which pushes a team to smaller models trained longer.

## Build it step by step

The checks call six functions, in this order. `predict_loss` from the last chapter is already in the starter file.

1. `transformer_params(layers, d_model, d_ff, vocab, seq_len)` returns a dict of integers with the keys `"attention"`, `"feedforward"`, `"embedding"`, `"non_embedding"`, and `"total"`. The toy sizes should give 512, 1024, 928, 1536, and 2464, and the embedding count must not change with the number of blocks.
2. `training_flops(params, tokens)` returns `6.0 * params * tokens` as a float. Call `float()` on each input first, because the mistake that fails the overflow check is multiplying `np.int64` values.
3. `training_days(flops, flops_per_second=2e14)` divides flops by the speed to get seconds, then by 86,400 to get days. At the default speed $10^{20}$ flops should give 5.79 days.
4. `chinchilla_optimal(compute)` returns `(n_star, d_star)` from the closed form. Compute `a`, then `G`, then `n_star`, then `d_star` from $C = 6ND$. The usual mistake is dropping the $6^{-a}$, which pushes the ratio far from 19.6.
5. `grid_optimal(compute, n_points=2000)` builds a log-spaced grid of sizes from $\sqrt{C/6}/100$ to $100\sqrt{C/6}$, sets `D = compute / (6 * N)` for each, evaluates `predict_loss` on the arrays, and returns the `(N, D)` with the lowest loss. Use `np.geomspace` and `np.argmin`. If it disagrees with `chinchilla_optimal` by more than a percent, widen the range before you blame the formula.
6. `lifetime_flops(n_params, train_tokens, served_tokens)` returns $6ND_{\text{train}} + 2ND_{\text{served}}$ as a float. Convert each input to `float` first, so numpy integers cannot overflow.

## Where this breaks

The data has to exist: at $10^{24}$ flops, $D^{*}$ asks for 1.8 trillion tokens of distinct text, and repeated data helps less than fresh data. A small error in an exponent also grows with distance, so predicting the best split three decades beyond your experiments is a guess.

## The exercise

You implement `transformer_params`, `training_flops`, `training_days`, `chinchilla_optimal`, `grid_optimal`, and `lifetime_flops` in `starter.py`. The checks verify the toy model's hand count, that embeddings are counted once, that `training_flops` returns $6ND$ and survives `np.int64` inputs, that 48 blocks reach 154 days, that the closed form spends the budget, sits at the bottom of the loss along the budget line, and gives a ratio of 19.6 at every budget, that the grid agrees within 1 percent, and that `lifetime_flops` reproduces the break-even at $2.86 \times 10^{10}$ served tokens.
