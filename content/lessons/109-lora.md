---
title: "LoRA: train a low-rank update, then merge it"
slug: 109-lora
part: "Part 7 — Large language models"
summary: Write a fine-tuning update as a tall matrix times a wide one, train a few thousand numbers, then fold the result into the weight for serving.
minutes: 55
gpu: false
objectives:
  - Explain why a column times a row fills a whole matrix, and why a sum of r such products has rank at most r.
  - Compute a LoRA forward pass by hand on a 2 by 2 layer and write it in numpy, with B at zero and A random for a reason you can state.
  - Count the trained parameters as r(d_in + d_out) and compare with d_in * d_out.
  - Merge the adapter into the frozen weight as W + (alpha / r) B A, and explain why the outputs match.
  - Say when to merge and when to keep adapters separate, with the memory numbers for each.
prereqs:
  - 108-fine-tuning-memory-bill
  - 007-stable-least-squares
exercise: 109-lora
---

# LoRA: train a low-rank update, then merge it

Fine-tuning changes the weights of a model. Take one layer of a 7-billion-parameter model, a weight $W$ with 4096 rows and 4096 columns. That is 16,777,216 numbers, and full fine-tuning changes all of them by some amount, which we call the update. The previous chapter, "Why fine-tuning a big model runs out of memory", showed that every number you train costs 16 bytes, so training all of them is the expensive path.

Suppose you want the model to answer in a legal tone. That is a small change of behaviour, and it would be strange if it took 16.7 million independent adjustments to express. If the update is simple, you should be able to describe it with far fewer numbers and train only those. The blunt alternatives are to train every number, or to freeze whole layers and hope the free ones are the ones that matter. This chapter builds a third option, low-rank adaptation, called LoRA, in which every layer can change but only in a restricted shape that is cheap to describe.

A second question comes after training. Each request takes a detour through the adapter, and you ship the base model and the adapter as a pair. Can the adapter disappear, leaving a model of the original size and speed? One line of algebra says yes, and its cost shows when it is the wrong move.

## What you already know

- A linear layer computes `y = x @ W.T`, where `W` has shape `(d_out, d_in)`: one row of `W` per output.
- An outer product of a column and a row has one row per column entry and one column per row entry.
- Matrix shapes must line up: `(a, b) @ (b, c)` gives `(a, c)`.
- Training a parameter costs 16 bytes, so the memory bill is proportional to how many parameters you train.
- The singular value decomposition writes any matrix as a sum of one-layer pieces, the idea behind rank.

## The bet: the update is simpler than the weight

Think of $W$ as a thick book and fine-tuning as a sheet of corrections stapled to it. The sheet is short because the book already knows most of what you need. LoRA bets that the update is like that sheet.

Why not train a small matrix directly? It cannot be added to $W$, because the shapes would not match. The update must be a full-size $d_{\text{out}} \times d_{\text{in}}$ matrix built from few numbers, and a column times a row does exactly that.

## The picture: a column times a row

Take a column with 4 entries and a row with 4 entries and multiply them. Every cell of the result is one column entry times one row entry, so 8 numbers fill all 16 cells. Every row of the result is the same row, scaled by that row's column entry. Figure 1 draws this twice, then adds the two results.

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

A matrix made from one column and one row is called rank 1. Adding $r$ such layers gives a matrix of rank at most $r$, and we call $r$ the rank. On the right-hand grid, rank 2 needs 16 numbers for 16 cells and saves nothing. The saving appears only when $r$ is small compared with the grid's width, as it will be when the grid is 4096 wide.

Stack the $r$ columns into a tall matrix $B$ with $d_{\text{out}}$ rows and $r$ columns, and the $r$ rows into a wide matrix $A$ with $r$ rows and $d_{\text{in}}$ columns. The product $BA$ adds up all $r$ layers in one step and has the shape of $W$.

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

The top path is the frozen layer: $x W^{\top} = [1, 1]$. The bottom path first multiplies by $A$, which squeezes the two input numbers down to $1 \cdot 1 + 2 \cdot 1 = 3$, a single number because $r = 1$. Then $B$ expands it back to two numbers, $[3, 0]$. Multiplying by the scale 2 gives $[6, 0]$. Adding the paths gives $[1, 1] + [6, 0] = [7, 1]$. The narrow waist is the whole trick: the correction passes through only $r$ numbers.

## The formula

Call the scale $s = \alpha / r$. The adapted layer acts as if its weight were

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

Check the 2 by 2 numbers: $BA$ has first row $[1, 2]$ and second row $[0, 0]$, so $W_{\text{eff}} = [[3, 4], [0, 1]]$, and $x W_{\text{eff}}^{\top} = [1 \cdot 3 + 1 \cdot 4,\; 1 \cdot 0 + 1 \cdot 1] = [7, 1]$. Both routes agree. Dividing by $r$ keeps the update's size roughly steady across ranks, so one learning rate works for several. That is a convention, not a law.

> [!WARNING]
> `W` has shape `(d_out, d_in)`, so the layer is `x @ W.T`. When `W` is square, `x @ W` still runs and returns different numbers with no error.

## Where B and A start

Start $B$ at zero. Then $BA = 0$ and $W_{\text{eff}} = W$, so before the first step the adapted layer is exactly the pretrained one. Training starts from a model that already works.

Why not start $A$ at zero as well? Let $G$ be the gradient of the loss with respect to $W_{\text{eff}}$. The chain rule through $BA$, derived in "What a rank-r update can reach, and how it is trained", gives

$$
\frac{\partial L}{\partial B} = s \, G A^{\top}, \qquad \frac{\partial L}{\partial A} = s \, B^{\top} G
$$

The gradient for $B$ is built from $A$, and the gradient for $A$ is built from $B$. Try the 2 by 2 layer with $G$ equal to the identity and $s = 2$. With $A = [1, 2]$ and $B = [0, 0]^{\top}$, the gradient for $B$ is $2 \cdot G A^{\top} = [2, 4]^{\top}$, which is not zero, while the gradient for $A$ is $2 B^{\top} G = [0, 0]$. So $B$ moves first, and then $A$'s gradient is no longer zero. With $A = 0$ as well, both gradients are zero and training is stuck. If every row of $A$ were the same, the $r$ layers would be copies, and you would pay for rank $r$ while getting rank 1.

## Counting the numbers you train

The update has $r$ columns of length $d_{\text{out}}$ and $r$ rows of length $d_{\text{in}}$, so

$$
\text{LoRA parameters} = r \, (d_{\text{in}} + d_{\text{out}})
$$

Read it aloud: $r$ times the sum of the two widths, against the $d_{\text{in}} d_{\text{out}}$ of the full matrix. The 2 by 2 layer trains $1 \times (2 + 2) = 4$ numbers, the same as the full matrix, so the saving needs a small $r$. For the $4096 \times 4096$ weight at $r = 8$, you train $8 \times 8192 = 65{,}536$ numbers instead of 16,777,216, which is 256 times fewer. At 16 bytes each, the memory for those numbers falls from 256 MiB to 1 MiB. The frozen $W$ still takes 2 bytes per parameter.

## Merging the adapter back

After training, the adapter path and the frozen path both multiply $x$ by a matrix, and adding two such results equals multiplying $x$ by the sum of the matrices. So you can add the matrices once, ahead of time. Figure 3 draws the layer before and after.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Before merging a layer has a frozen path and an adapter path; after merging it has one matrix, and the adapter's extra multiplies are gone">
<defs><marker id="c122-merge-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20" y="22" class="fig-bold">Before: two paths, 16,842,752 multiplies per example</text>
<text x="40" y="78" class="fig-small">x</text>
<line x1="52.0" y1="74.0" x2="110.0" y2="52.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<rect x="114" y="28" width="200" height="48" rx="6" class="fill-sunk stroke-ink"/>
<text x="214.0" y="52.0" text-anchor="middle" class="fig-bold">W (frozen)</text>
<text x="214.0" y="68.0" text-anchor="middle" class="fig-small">16,777,216 multiplies</text>
<line x1="52.0" y1="80.0" x2="110.0" y2="112.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<rect x="114" y="96" width="90" height="48" rx="6" class="fill-soft-c stroke-ink"/>
<text x="159.0" y="120.0" text-anchor="middle" class="fig-bold">A</text>
<text x="159.0" y="136.0" text-anchor="middle" class="fig-small">32,768</text>
<line x1="204.0" y1="120.0" x2="226.0" y2="120.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<rect x="230" y="96" width="90" height="48" rx="6" class="fill-soft-c stroke-ink"/>
<text x="275.0" y="120.0" text-anchor="middle" class="fig-bold">B</text>
<text x="275.0" y="136.0" text-anchor="middle" class="fig-small">32,768</text>
<line x1="314.0" y1="52.0" x2="430.0" y2="84.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<line x1="320.0" y1="120.0" x2="430.0" y2="92.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<circle cx="444" cy="88" r="14" class="fill-paper stroke-ink"/><text x="444" y="93" text-anchor="middle" class="fig-bold">+</text>
<line x1="458.0" y1="88.0" x2="500.0" y2="88.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<text x="504" y="93" class="fig-bold">y</text>
<text x="340" y="146" class="fig-small">extra path: 65,536 = 0.39% of the base</text>
<text x="20" y="196" class="fig-bold">After: one matrix, 16,777,216 multiplies per example</text>
<text x="40" y="250" class="fig-small">x</text>
<line x1="52.0" y1="246.0" x2="110.0" y2="246.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<rect x="114" y="222" width="250" height="48" rx="6" class="fill-soft-b stroke-ink"/>
<text x="239.0" y="246.0" text-anchor="middle" class="fig-bold">W + (α/r) B A</text>
<text x="239.0" y="262.0" text-anchor="middle" class="fig-small">16,777,216 multiplies</text>
<line x1="364.0" y1="246.0" x2="430.0" y2="246.0" class="stroke-ink" marker-end="url(#c122-merge-arrow)"/>
<text x="440" y="251" class="fig-bold">y</text>
<text x="20" y="306" class="fig-small">Both layers return the same y. The merged layer has nothing left to run except W.</text>
</svg>
Figure 3. Before merging, the adapter path adds 65,536 multiplies to the frozen layer's 16,777,216. After merging, only the one matrix is left and the output is the same.
```

The merged weight is the $W_{\text{eff}}$ you already computed:

```python
W_merged = W + (alpha / r) * (B @ A)   # (d_out, d_in)
y = x @ W_merged.T
```

On the 2 by 2 example the merged matrix is $[[3, 4], [0, 1]]$, and it gives $[7, 1]$, the same as the split path. Why `B @ A` and not `A @ B`? `B` is `(d_out, r)` and `A` is `(r, d_in)`, so `B @ A` is `(d_out, d_in)`, the shape of `W`. `A @ B` needs `d_in` to equal `d_out`, and even then it returns an `(r, r)` matrix, which cannot be added to `W`.

> [!WARNING]
> On a square layer both `B @ A` and `A @ B` run without an error. Only `B @ A` has the shape of `W`, so check the shape of what you add.

Per example, the adapter path costs $r \, d_{\text{in}}$ multiplies for $A$ and $d_{\text{out}} \, r$ for $B$, one for each trained parameter. For the 4096 by 4096 layer at rank 8 that is 65,536 against 16,777,216, or 0.39 percent. So the saving is not mainly arithmetic. The merged layer runs one multiplication where the split layer runs three, needs no adapter-aware server code, and reads exactly the base model's bytes per token, which is what decoding waits on, as in "Why decoding waits on memory, and the KV cache".

One caution: if the base weights are stored in 16 bits, a very small update can round away when merged, so compute the merge in 32 bits and round once at the end. If the base is stored in 4 bits, as in QLoRA, merging is only an approximation.

## When not to merge

Suppose you serve three customers, each with an adapter, from one 7-billion-parameter base. Merging makes one full model per customer. Figure 4 counts the memory.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three merged 7 billion parameter copies need 42 GB; one base plus three adapters needs about 14 GB">
<text x="20" y="24" class="fig-bold">Serving three customers from a 7-billion-parameter base</text>
<text x="20" y="62" class="fig-small">Merge each:</text>
<rect x="60.0" y="70" width="160.0" height="40" class="fill-soft-b stroke-ink"/>
<text x="140.0" y="95" text-anchor="middle" class="fig-bold">14 GB</text>
<rect x="220.0" y="70" width="160.0" height="40" class="fill-soft-b stroke-ink"/>
<text x="300.0" y="95" text-anchor="middle" class="fig-bold">14 GB</text>
<rect x="380.0" y="70" width="160.0" height="40" class="fill-soft-b stroke-ink"/>
<text x="460.0" y="95" text-anchor="middle" class="fig-bold">14 GB</text>
<text x="540.0" y="128" text-anchor="end" class="fig-bold">42 GB: three full copies</text>
<text x="20" y="168" class="fig-small">Keep the adapters separate:</text>
<rect x="60" y="176" width="160.0" height="40" class="fill-soft-a stroke-ink"/>
<text x="140.0" y="201" text-anchor="middle" class="fig-bold">14 GB base</text>
<rect x="220.0" y="176" width="2" height="40" class="fill-b"/>
<text x="234.0" y="199" class="fig-small">three adapters: 48 MiB</text>
<text x="234.0" y="216" class="fig-small">(adapters: 4 matrices per layer, 32 layers, rank 8)</text>
</svg>
Figure 4. Three merged copies take 42 GB, because each is a full 14 GB model. One base with the adapters kept apart takes about 14 GB, since the adapters together are only 48 MiB.
```

At 2 bytes per number, one merged copy is 14 GB, so three copies are 42 GB. An adapter on four 4096 by 4096 matrices in each of 32 layers at rank 8 holds $128 \times 65{,}536 = 8{,}388{,}608$ numbers, or 16 MiB, so three adapters are 48 MiB. A server that keeps them separate loads the base once and adds the small extra path for whichever customer's request arrives. A server with one adapter that takes almost all the traffic should merge.

LoRA folds in because its update is linear. Older adapters insert small extra layers with a nonlinearity, so they cannot be merged away.

> [!KEY]
> Fine-tuning needs an update the size of $W$, but a column times a row fills a whole matrix from a few numbers, so LoRA writes the update as $BA$ with $B$ tall and $A$ wide. You train $r(d_{\text{in}} + d_{\text{out}})$ numbers, 256 times fewer for a 4096 by 4096 layer at rank 8, and starting $B$ at zero keeps step zero identical to the pretrained model.

> [!KEY]
> The update is linear in the input, so $x W^{\top} + s (x A^{\top}) B^{\top}$ equals $x (W + s B A)^{\top}$ and you can fold the adapter into one ordinary matrix. The merged layer matches the split outputs with no extra path, but each merged copy is a full model, so a server with many adapters keeps them separate.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The forward pass is `x @ W.T` plus `(alpha / r) * (x @ A.T) @ B.T`, with $B$ `(d_out, r)` starting at zero and $A$ `(r, d_in)` starting random.
> - LoRA trains $r(d_{\text{in}} + d_{\text{out}})$ numbers, and the unmerged path adds the same count of multiplies per example.
> - The merged weight is `W + (alpha / r) * (B @ A)`, in that order.
> - Merge when one adapter serves most traffic, and keep adapters separate when many share a base.

## Build it step by step

`x` is `(n, d_in)`, `W` is `(d_out, d_in)`, `A` is `(r, d_in)`, and `B` is `(d_out, r)`.

1. **`lora_parameter_count(d_in, d_out, r)`.** It returns $r (d_{\text{in}} + d_{\text{out}})$ as an int. The usual mistake is to return `d_in * d_out`.
2. **`lora_init(d_in, d_out, r, seed)`.** It returns the pair `(A, B)`. Draw `A` from a normal distribution with standard deviation `1 / sqrt(d_in)`, using `np.random.default_rng(seed)`, and make `B` all zeros. The usual mistake is to start both at zero.
3. **`lora_forward(x, W, A, B, alpha, r)`.** It returns shape `(n, d_out)`. Keep the two paths separate and add them. The usual mistake is to write `x @ W`.
4. **`merge_lora(W, A, B, alpha, r)`.** It returns the merged weight with shape `(d_out, d_in)`. The usual mistake is to write `A @ B`.
5. **`extra_multiplies(d_in, d_out, r)`.** It returns the multiplies the unmerged adapter adds per example, as an int. The usual mistake is to count the base layer's multiplies too.
6. **`merged_matches_split(x, W, A, B, alpha, r)`.** It returns the largest absolute difference between `x @ merged.T` and `lora_forward(x, W, A, B, alpha, r)`, calling your `merge_lora`. The usual mistake is to compare against the base layer instead of the split path.

## Where this breaks

A rank-$r$ update cannot express every change. If full fine-tuning would need many independent layers, a small $r$ loses some of the change. The chapter "What a rank-r update can reach, and how it is trained" measures how much, and shows when low rank is a good bet.

## The exercise

You write the six functions above. The checks confirm the parameter counts, that `lora_init` gives zeros for `B` and a seeded `A` of the right scale, that a zero `B` reproduces the frozen layer, and that the 2 by 2 example gives $[7, 1]$ through `lora_forward` and $[[3, 4], [0, 1]]$ through `merge_lora`. They also check that the update has rank $r$, that `extra_multiplies` returns 65,536 at 4096 by 4096 and rank 8, and that merged and split outputs match to machine precision.
