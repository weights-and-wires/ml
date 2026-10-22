---
title: "A low-rank update: build the LoRA layer"
slug: 122-lora-layer
part: "Part 7 — Large language models"
summary: Describe the change fine-tuning makes to a weight as a tall matrix times a wide one, so you train a few thousand numbers instead of millions.
minutes: 40
gpu: false
objectives:
  - Explain why a column times a row fills a whole matrix, and why a sum of r such products has rank at most r.
  - Compute a LoRA forward pass by hand on a 2 by 2 layer and write it in numpy.
  - Explain why B starts at zero and A starts random, and why starting both at zero learns nothing.
  - Count the trained parameters as r(d_in + d_out) and compare with d_in * d_out.
prereqs:
  - 121-fine-tuning-memory-bill
  - 008-qr-and-svd
exercise: 122-lora-layer
---

# A low-rank update: build the LoRA layer

Fine-tuning changes the weights of a model. Take one layer of a 7-billion-parameter model, a weight $W$ with 4096 rows and 4096 columns. That is 16,777,216 numbers, and full fine-tuning changes all of them by some amount, which we call the update. The previous chapter, "Why fine-tuning a big model runs out of memory", showed that every number you train costs 16 bytes, so training all of them is the expensive path.

Here is the question that opens a cheaper one. Suppose you want the model to answer in a legal tone. That is a small change of behaviour, and it would be strange if it took 16.7 million independent adjustments to express. If the update is simple, you should be able to describe it with far fewer numbers and train only those.

Without that idea you have two blunt options: train every number, or freeze whole layers and hope the ones you left free are the ones that matter. This chapter builds a third option, low-rank adaptation, called LoRA, in which every layer can change but only in a restricted shape that is cheap to describe.

## What you already know

- A linear layer computes `y = x @ W.T`, where `W` has shape `(d_out, d_in)`: one row of `W` per output.
- An outer product of a column and a row is a matrix with one row for each entry of the column and one column for each entry of the row.
- Matrix shapes must line up: `(a, b) @ (b, c)` gives `(a, c)`.
- Training a parameter costs 16 bytes, so the memory bill is proportional to how many parameters you train.
- The singular value decomposition, from the earlier chapter on it, writes any matrix as a sum of one-layer pieces, which is the idea behind rank.

## The bet: the update is simpler than the weight

Think of $W$ as a thick book and fine-tuning as a sheet of corrections stapled to it. The sheet is short because the book already knows most of what you need. LoRA bets that the update behaves like that sheet, so it can be written compactly.

You may ask why not train a small matrix directly. A small matrix cannot be added to $W$, because the shapes would not match, and shrinking $W$ itself would change the layer you started from. The update must be a full-size $d_{\text{out}} \times d_{\text{in}}$ matrix that is built from few numbers. A column times a row does exactly that.

## The picture: a column times a row

Take a column with 4 entries and a row with 4 entries and multiply them. Every cell of the result is one column entry times one row entry, so 8 numbers fill all 16 cells. Notice what that forces: every row of the result is the same row, scaled by that row's column entry, so all the rows point the same way. Figure 1 draws this twice, with a dot for each cell, then adds the two results.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="A tall column times a wide row makes a full 4 by 4 grid of rank 1; two such products added make rank 2">
<text x="20" y="32" class="fig-bold">Layer 1: column B1 times row A1</text>
<circle cx="30" cy="50" r="3.0" class="fill-a"/>
<circle cx="30" cy="70" r="4.2" class="fill-a"/>
<circle cx="30" cy="90" r="3.0" class="fill-b"/>
<circle cx="30" cy="110" r="5.2" class="fill-a"/>
<rect x="20" y="40" width="20" height="80" class="stroke-muted" fill="none"/>
<text x="48" y="85" class="fig-bold">×</text>
<circle cx="80" cy="80" r="4.2" class="fill-a"/>
<circle cx="100" cy="80" r="3.0" class="fill-a"/>
<circle cx="120" cy="80" r="3.0" class="fill-b"/>
<circle cx="140" cy="80" r="3.0" class="fill-a"/>
<rect x="70" y="70" width="80" height="20" class="stroke-muted" fill="none"/>
<text x="158" y="85" class="fig-bold">=</text>
<circle cx="190" cy="50" r="2.7" class="fill-a"/>
<circle cx="210" cy="50" r="1.9" class="fill-a"/>
<circle cx="230" cy="50" r="1.9" class="fill-b"/>
<circle cx="250" cy="50" r="1.9" class="fill-a"/>
<circle cx="190" cy="70" r="3.8" class="fill-a"/>
<circle cx="210" cy="70" r="2.7" class="fill-a"/>
<circle cx="230" cy="70" r="2.7" class="fill-b"/>
<circle cx="250" cy="70" r="2.7" class="fill-a"/>
<circle cx="190" cy="90" r="2.7" class="fill-b"/>
<circle cx="210" cy="90" r="1.9" class="fill-b"/>
<circle cx="230" cy="90" r="1.9" class="fill-a"/>
<circle cx="250" cy="90" r="1.9" class="fill-b"/>
<circle cx="190" cy="110" r="4.7" class="fill-a"/>
<circle cx="210" cy="110" r="3.3" class="fill-a"/>
<circle cx="230" cy="110" r="3.3" class="fill-b"/>
<circle cx="250" cy="110" r="3.3" class="fill-a"/>
<rect x="180" y="40" width="80" height="80" class="stroke-muted" fill="none"/>
<text x="20" y="152" class="fig-bold">Layer 2: column B2 times row A2</text>
<circle cx="30" cy="170" r="3.0" class="fill-a"/>
<circle cx="30" cy="190" r="4.2" class="fill-b"/>
<circle cx="30" cy="210" r="4.2" class="fill-a"/>
<circle cx="30" cy="230" r="3.0" class="fill-a"/>
<rect x="20" y="160" width="20" height="80" class="stroke-muted" fill="none"/>
<text x="48" y="205" class="fig-bold">×</text>
<circle cx="80" cy="200" r="3.0" class="fill-b"/>
<circle cx="100" cy="200" r="4.2" class="fill-a"/>
<circle cx="120" cy="200" r="3.0" class="fill-a"/>
<circle cx="140" cy="200" r="4.2" class="fill-b"/>
<rect x="70" y="190" width="80" height="20" class="stroke-muted" fill="none"/>
<text x="158" y="205" class="fig-bold">=</text>
<circle cx="190" cy="170" r="1.9" class="fill-b"/>
<circle cx="210" cy="170" r="2.7" class="fill-a"/>
<circle cx="230" cy="170" r="1.9" class="fill-a"/>
<circle cx="250" cy="170" r="2.7" class="fill-b"/>
<circle cx="190" cy="190" r="2.7" class="fill-a"/>
<circle cx="210" cy="190" r="3.8" class="fill-b"/>
<circle cx="230" cy="190" r="2.7" class="fill-b"/>
<circle cx="250" cy="190" r="3.8" class="fill-a"/>
<circle cx="190" cy="210" r="2.7" class="fill-b"/>
<circle cx="210" cy="210" r="3.8" class="fill-a"/>
<circle cx="230" cy="210" r="2.7" class="fill-a"/>
<circle cx="250" cy="210" r="3.8" class="fill-b"/>
<circle cx="190" cy="230" r="1.9" class="fill-b"/>
<circle cx="210" cy="230" r="2.7" class="fill-a"/>
<circle cx="230" cy="230" r="1.9" class="fill-a"/>
<circle cx="250" cy="230" r="2.7" class="fill-b"/>
<rect x="180" y="160" width="80" height="80" class="stroke-muted" fill="none"/>
<text x="380" y="32" class="fig-bold">Layer 1 + layer 2: rank 2</text>
<circle cx="395" cy="55" r="2.6" class="fill-a"/>
<circle cx="425" cy="55" r="4.5" class="fill-a"/>
<circle cx="485" cy="55" r="2.6" class="fill-b"/>
<circle cx="395" cy="85" r="6.4" class="fill-a"/>
<circle cx="425" cy="85" r="3.7" class="fill-b"/>
<circle cx="455" cy="85" r="5.2" class="fill-b"/>
<circle cx="485" cy="85" r="6.4" class="fill-a"/>
<circle cx="395" cy="115" r="5.2" class="fill-b"/>
<circle cx="425" cy="115" r="4.5" class="fill-a"/>
<circle cx="455" cy="115" r="4.5" class="fill-a"/>
<circle cx="485" cy="115" r="5.8" class="fill-b"/>
<circle cx="395" cy="145" r="5.8" class="fill-a"/>
<circle cx="425" cy="145" r="5.8" class="fill-a"/>
<circle cx="455" cy="145" r="3.7" class="fill-b"/>
<circle cx="485" cy="145" r="2.6" class="fill-a"/>
<rect x="380" y="40" width="120" height="120" class="stroke-muted" fill="none"/>
<text x="380" y="190" class="fig-small">The sum needs 2 × (4 + 4) = 16</text>
<text x="380" y="206" class="fig-small">numbers, as many as it has</text>
<text x="380" y="222" class="fig-small">cells: no saving at rank 2.</text>
<text x="20" y="258" class="fig-small">Blue dots are positive cells, orange dots negative, and area grows with size. Empty means 0.</text>
</svg>
Figure 1. Each layer is a column times a row, and it fills a whole 4 by 4 grid from 8 numbers. Adding two layers gives a matrix that needs two columns and two rows.
```

A matrix made from one column and one row is called rank 1. Adding $r$ such layers gives a matrix of rank at most $r$, and we call $r$ the rank: the number of one-layer pieces you need. Look at the right-hand grid. It needs two columns and two rows, 16 numbers, and has 16 cells, so on a 4 by 4 grid rank 2 saves nothing. The saving appears only when $r$ is small compared with the grid's width, as it will be when the grid is 4096 wide.

Stack the $r$ columns side by side and you get a tall matrix $B$ with $d_{\text{out}}$ rows and $r$ columns. Stack the $r$ rows and you get a wide matrix $A$ with $r$ rows and $d_{\text{in}}$ columns. The product $BA$ adds up all $r$ layers in one step and has the shape of $W$.

## A 2 by 2 layer, by hand

Use the smallest case that has all the parts. The input width is 2, the output width is 2, and the rank is $r = 1$. Let $W$ be the identity, $A$ the row $[1, 2]$, and $B$ the column $[1, 0]$. A setting called $\alpha$ (alpha), which you choose, is 2, and the update is multiplied by $\alpha / r = 2$. We feed in $x = [1, 1]$. Figure 2 follows the numbers.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="The input [1, 1] goes through the frozen weight to [1, 1], and through A to 3, then B to [3, 0], scaled by 2 to [6, 0], and the two paths add to [7, 1]">
<defs><marker id="c122-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="20.0" y="111.0" width="34" height="26" class="fill-soft-a stroke-ink"/>
<text x="37.0" y="128.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="20.0" y="137.0" width="34" height="26" class="fill-soft-a stroke-ink"/>
<text x="37.0" y="154.5" text-anchor="middle" class="fig-bold">1</text>
<text x="37" y="187" text-anchor="middle" class="fig-small">x = [1, 1]</text>
<line x1="58.0" y1="129.0" x2="100.0" y2="74.0" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<rect x="104" y="50" width="110" height="40" rx="6" class="fill-sunk stroke-ink"/>
<text x="159.0" y="74.5" text-anchor="middle" class="fig-bold">W (frozen)</text>
<line x1="214" y1="70" x2="470" y2="70" class="stroke-ink"/>
<text x="330" y="62" text-anchor="middle" class="fig-small">x W^T = [1, 1]</text>
<line x1="58.0" y1="145.0" x2="100.0" y2="201.0" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<rect x="104" y="185" width="70" height="40" rx="6" class="fill-soft-c stroke-ink"/>
<text x="139.0" y="209.5" text-anchor="middle" class="fig-bold">A</text>
<line x1="174.0" y1="205.0" x2="206.0" y2="205.0" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<rect x="210.0" y="192.0" width="34" height="26" class="fill-soft-c stroke-ink"/>
<text x="227.0" y="209.5" text-anchor="middle" class="fig-bold">3</text>
<text x="227" y="239" text-anchor="middle" class="fig-small">r = 1</text>
<line x1="246.0" y1="205.0" x2="278.0" y2="205.0" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<rect x="282" y="185" width="70" height="40" rx="6" class="fill-soft-c stroke-ink"/>
<text x="317.0" y="209.5" text-anchor="middle" class="fig-bold">B</text>
<line x1="352.0" y1="205.0" x2="384.0" y2="205.0" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<rect x="388.0" y="179.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="405.0" y="196.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="388.0" y="205.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="405.0" y="222.5" text-anchor="middle" class="fig-bold">0</text>
<text x="405" y="249" text-anchor="middle" class="fig-small">[3, 0]</text>
<line x1="426" y1="205" x2="470" y2="205" class="stroke-ink"/>
<text x="240" y="175" text-anchor="middle" class="fig-small">squeeze to r numbers, expand back</text>
<text x="448" y="197" text-anchor="middle" class="fig-small">× 2</text>
<circle cx="470" cy="137" r="14" class="fill-paper stroke-ink"/><text x="470" y="142" text-anchor="middle" class="fig-bold">+</text>
<line x1="470" y1="70" x2="470" y2="122" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<line x1="470" y1="205" x2="470" y2="152" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<line x1="484.0" y1="137.0" x2="522.0" y2="137.0" class="stroke-ink" marker-end="url(#c122-arrow)"/>
<rect x="526.0" y="111.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="543.0" y="128.5" text-anchor="middle" class="fig-bold">7</text>
<rect x="526.0" y="137.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="543.0" y="154.5" text-anchor="middle" class="fig-bold">1</text>
<text x="543" y="187" text-anchor="middle" class="fig-small">y = [7, 1]</text>
<text x="470" y="227" text-anchor="middle" class="fig-small">[6, 0]</text>
</svg>
Figure 2. The input splits into two paths. The top path is the frozen layer. The bottom path squeezes the input down to r = 1 number, expands it back, and scales it. The two paths add to give the output.
```

The top path is the frozen layer: $x W^{\top} = [1, 1]$. The bottom path first multiplies by $A$, which squeezes the two input numbers down to $1 \cdot 1 + 2 \cdot 1 = 3$, a single number because $r = 1$. Then $B$ expands it back to two numbers, $[3, 0]$. Multiplying by the scale 2 gives $[6, 0]$. Adding the paths gives $[1, 1] + [6, 0] = [7, 1]$.

The narrow waist in the middle is the whole trick. However wide the layer is, the correction can only pass information through $r$ numbers, and that limit is what keeps the number of trained parameters small.

## The formula

Name what you computed. Call the scale $s = \alpha / r$. The adapted layer acts as if its weight were

$$
W_{\text{eff}} = W + s \, BA, \qquad s = \frac{\alpha}{r}
$$

Read it aloud: the effective weight is the frozen weight plus the rank-$r$ update, scaled by $s$. For a batch $x$ with one row per example, the forward pass applies the pieces separately:

$$
\hat{y} = x W^{\top} + s \, (x A^{\top}) B^{\top}
$$

Read it aloud: the frozen layer's output, plus the squeeze by $A$, the expand by $B$, and the scale. In numpy:

```python
base = x @ W.T                     # (n, d_out)
update = (x @ A.T) @ B.T           # (n, r) then (n, d_out)
y = base + (alpha / r) * update
```

Check the 2 by 2 numbers: $BA$ is the matrix with first row $[1, 2]$ and second row $[0, 0]$, so $W_{\text{eff}} = [[3, 4], [0, 1]]$, and $x W_{\text{eff}}^{\top} = [1 \cdot 3 + 1 \cdot 4,\; 1 \cdot 0 + 1 \cdot 1] = [7, 1]$. Both routes agree.

Why divide by $r$? The update $BA$ is a sum of $r$ layers, so with more layers it tends to grow. Dividing by $r$ keeps its size roughly steady when you change the rank, so one learning rate works for several ranks. This is a convention from the LoRA paper, not a law, and it holds only roughly, since the real size also depends on how $A$ and $B$ are initialised and trained.

> [!WARNING]
> `W` has shape `(d_out, d_in)`, so the layer is `x @ W.T`. When `W` is square, `x @ W` still runs and returns different numbers with no error.

## Where B and A start

You train $A$ and $B$ from some starting values, and the choice matters. Start $B$ at zero. Then $BA = 0$ and $W_{\text{eff}} = W$, so before the first step the adapted layer is exactly the pretrained one. Training moves away from a model that already works, instead of from noise.

Why not start $A$ at zero as well? Let $G$ be the gradient of the loss with respect to $W_{\text{eff}}$, a $d_{\text{out}} \times d_{\text{in}}$ matrix. The chain rule through $BA$, which the chapter "What a rank-r update can reach, and how it is trained" derives, gives

$$
\frac{\partial L}{\partial B} = s \, G A^{\top}, \qquad \frac{\partial L}{\partial A} = s \, B^{\top} G
$$

The gradient for $B$ is built from $A$, and the gradient for $A$ is built from $B$. If both are zero, both gradients are zero and nothing ever moves. Try it on the 2 by 2 layer with $G$ equal to the identity and $s = 2$. With $A = [1, 2]$ and $B = [0, 0]^{\top}$, the gradient for $B$ is $2 \cdot G A^{\top} = [2, 4]^{\top}$, which is not zero, while the gradient for $A$ is $2 B^{\top} G = [0, 0]$. So $B$ moves first, and after it moves, $A$'s gradient is no longer zero. With $A = 0$ as well, both gradients are $0$ and training is stuck at step zero. This is why $B$ starts at zero and $A$ starts random. The randomness matters too: if every row of $A$ were the same, the $r$ layers of the update would be copies of one another, and you would pay for rank $r$ while getting rank 1.

## Counting the numbers you train

The update has $r$ columns of length $d_{\text{out}}$ and $r$ rows of length $d_{\text{in}}$, so

$$
\text{LoRA parameters} = r \, (d_{\text{in}} + d_{\text{out}})
$$

Read it aloud: $r$ times the sum of the two widths, against the $d_{\text{in}} d_{\text{out}}$ of the full matrix. The toy 2 by 2 layer trains $1 \times (2 + 2) = 4$ numbers, the same as the full matrix, which is why the saving needs $r$ far below the width. For the $4096 \times 4096$ weight at $r = 8$, you train $8 \times 8192 = 65{,}536$ numbers instead of 16,777,216, which is 256 times fewer. Figure 3 shows the sliver.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A 4096 by 4096 weight drawn as 16 by 16 cells; LoRA at rank 8 trains one cell, 65,536 of 16,777,216 numbers">
<text x="60" y="26" class="fig-bold">The weight W: 4096 × 4096 = 16,777,216 numbers</text>
<rect x="60" y="50" width="14" height="14" class="fill-b stroke-grid"/>
<rect x="74" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="50" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="64" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="78" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="92" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="106" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="120" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="134" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="148" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="162" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="176" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="190" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="204" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="218" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="232" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="246" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="74" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="88" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="102" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="116" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="130" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="144" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="158" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="172" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="186" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="200" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="214" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="228" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="242" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="256" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="270" y="260" width="14" height="14" class="fill-soft-a stroke-grid"/>
<rect x="60.0" y="50.0" width="224.0" height="224.0" class="stroke-ink" fill="none"/>
<rect x="52" y="50" width="3" height="224" class="fill-c"/>
<rect x="60" y="42" width="224" height="3" class="fill-c"/>
<text x="48" y="162.0" text-anchor="end" class="fig-small">B</text>
<text x="292" y="47" class="fig-small">A</text>
<text x="334" y="64" class="fig-small">Each cell is 65,536 numbers,</text>
<text x="334" y="84" class="fig-small">one 256th of the weight.</text>
<text x="334" y="104" class="fig-small"></text>
<text x="334" y="124" class="fig-small">Green strips: A (8 × 4096) and B</text>
<text x="334" y="144" class="fig-small">(4096 × 8), 65,536 numbers together,</text>
<text x="334" y="164" class="fig-small">exactly the size of the orange cell.</text>
<text x="334" y="184" class="fig-small"></text>
<text x="334" y="204" class="fig-small">Strips are drawn about 8 times too</text>
<text x="334" y="224" class="fig-small">thick; at true scale they are hairlines.</text>
<text x="60" y="300" class="fig-small">Full training: 16,777,216 numbers. LoRA at r = 8: 8 × (4096 + 4096) = 65,536, which is 256 times fewer.</text>
</svg>
Figure 3. The weight as 256 cells. The factors A and B together hold the numbers of one cell, so LoRA trains a 256th of what full fine-tuning trains.
```

At 16 bytes per trained parameter, the memory for those numbers falls from 256 MiB to 1 MiB for this one matrix. The frozen $W$ still sits in memory at 2 bytes per parameter, so the saving applies to the trained part only, as the previous chapter explained.

> [!KEY]
> Fine-tuning needs an update the size of $W$, but a column times a row fills a whole matrix from a few numbers, so LoRA writes the update as $BA$ with $B$ tall and $A$ wide. You train $r(d_{\text{in}} + d_{\text{out}})$ numbers, 256 times fewer for a 4096 by 4096 layer at rank 8, and starting $B$ at zero keeps step zero identical to the pretrained model.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The update is $s \, BA$ with $s = \alpha / r$, where $B$ is `(d_out, r)` and $A$ is `(r, d_in)`, so $BA$ has the shape of $W$.
> - The forward pass is `x @ W.T` plus `(alpha / r) * (x @ A.T) @ B.T`.
> - $B$ starts at zero and $A$ starts random, because each one's gradient is built from the other.
> - LoRA trains $r(d_{\text{in}} + d_{\text{out}})$ numbers.

## Build it step by step

`x` is `(n, d_in)`, `W` is `(d_out, d_in)`, `A` is `(r, d_in)`, and `B` is `(d_out, r)`.

1. `lora_parameter_count(d_in, d_out, r)` returns $r (d_{\text{in}} + d_{\text{out}})$ as an int. The usual mistake is to return `d_in * d_out`:

```python
def lora_parameter_count(d_in, d_out, r):
    # r * (d_in + d_out)
    return count
```

2. `lora_init(d_in, d_out, r, seed)` returns the pair `(A, B)`. Draw `A` from a normal distribution with standard deviation `1 / sqrt(d_in)`, using `np.random.default_rng(seed)`, and make `B` all zeros. The scale keeps `x @ A.T` near the size of `x` when the entries of `x` are about 1. The usual mistake is to start both at zero:

```python
def lora_init(d_in, d_out, r, seed):
    rng = np.random.default_rng(seed)
    # A: (r, d_in) normal, scale 1 / sqrt(d_in); B: (d_out, r) zeros
    return A, B
```

3. `lora_forward(x, W, A, B, alpha, r)` returns an array of shape `(n, d_out)`. Keep the two paths separate and add them. The usual mistake is to write `x @ W`:

```python
def lora_forward(x, W, A, B, alpha, r):
    # x @ W.T + (alpha / r) * (x @ A.T) @ B.T
    return y
```

## Where this breaks

A rank-$r$ update cannot express every change. If the update that full fine-tuning would make needs many independent layers, a small $r$ forces LoRA to approximate it and some of the change is lost. The chapter "What a rank-r update can reach, and how it is trained" measures how much, and shows when low rank is a good bet.

## The exercise

You write `lora_parameter_count`, `lora_init`, and `lora_forward`. The checks confirm the parameter counts for several sizes, that `lora_init` returns zeros for `B` and a seeded random `A` of the right scale, that a zero `B` reproduces the frozen layer exactly, that the 2 by 2 example gives $[7, 1]$, that the forward pass matches the explicit effective weight, and that the update the layer adds never has rank above $r$.
