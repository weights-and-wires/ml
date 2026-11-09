---
title: Fine-tuning on a budget with LoRA
slug: 054-fine-tuning-and-lora
part: "Part 7 — Large language models"
summary: Training a parameter costs 16 bytes, so LoRA trains a low-rank update, merges it for serving, and loses only the singular values it drops.
minutes: 75
gpu: false
objectives:
  - Name the five pieces that make a trained parameter cost 16 bytes, and compute the bill when most parameters are frozen.
  - Compute a LoRA forward pass by hand and in numpy, with B at zero and A random for a reason you can state.
  - Count the trained parameters as r(d_in + d_out) and merge the adapter as W + (alpha / r) B A.
  - Compute the best rank-r approximation with the SVD, and read from a spectrum how much a low-rank update loses.
  - Derive the gradients of A and B by the chain rule, and say which part of the full gradient a LoRA step follows.
prereqs:
  - 031-sgd-adam-and-schedules
  - 053-decoding-and-sampling
  - 004-stable-least-squares
  - 024-principal-components
exercise: 054-fine-tuning-and-lora
---

# Fine-tuning on a budget with LoRA

You have one GPU with 24 GB of memory and a language model with 7 billion parameters. You want to fine-tune it, which means training it a little more on your own data. The weights are 16-bit numbers, two bytes each, so they take 14 GB, which fits. You start the training script, and it stops with an out-of-memory error before the first step finishes.

The model fits for answering, but training holds far more than the weights. This chapter adds up what training holds for every parameter and finds the one lever that shrinks the bill: train fewer parameters.

Which few? Freezing whole layers is cheap but blunt. Low-rank adaptation, called LoRA, lets every layer change, but only in a restricted shape that is cheap to describe. The chapter builds that shape, folds it into the weights for serving, and asks what the restriction gives up.

## What you already know

- Training computes one gradient for each trained parameter at every step.
- Adam keeps two running averages per weight, as in the chapter on [SGD, Adam, and schedules](/lessons/031-sgd-adam-and-schedules).
- A 16-bit number takes 2 bytes and a 32-bit number takes 4 bytes.
- A linear layer computes `y = x @ W.T`, where `W` has shape `(d_out, d_in)`, and shapes must line up: `(a, b) @ (b, c)` gives `(a, c)`.
- The singular value decomposition, SVD, writes a matrix as a sum of one-layer pieces ordered by size, as in the chapter on [principal components](/lessons/024-principal-components).

## The picture: one parameter, five pieces

To answer a question, the model needs one copy of each weight, 2 bytes in 16-bit form. To train it, the computer holds four more things next to that copy. Figure 1 draws them for one parameter on top and for all 7 billion below.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="One parameter costs 16 bytes in five pieces; a 7 billion parameter model needs 112 GB, far above a 24 GB GPU">
<text x="20" y="24" class="fig-bold">One parameter: 16 bytes in five pieces</text>
<rect x="20.0" y="36" width="60.0" height="30" class="fill-soft-a stroke-ink"/>
<text x="50.0" y="56" text-anchor="middle" class="fig-small fig-bold">2 B</text>
<text x="50.0" y="84" text-anchor="middle" class="fig-small">weight</text>
<rect x="80.0" y="36" width="60.0" height="30" class="fill-soft-d stroke-ink"/>
<text x="110.0" y="56" text-anchor="middle" class="fig-small fig-bold">2 B</text>
<text x="110.0" y="84" text-anchor="middle" class="fig-small">gradient</text>
<rect x="140.0" y="36" width="120.0" height="30" class="fill-soft-c stroke-ink"/>
<text x="200.0" y="56" text-anchor="middle" class="fig-small fig-bold">4 B</text>
<text x="200.0" y="84" text-anchor="middle" class="fig-small">master copy</text>
<rect x="260.0" y="36" width="120.0" height="30" class="fill-soft-b stroke-ink"/>
<text x="320.0" y="56" text-anchor="middle" class="fig-small fig-bold">4 B</text>
<text x="320.0" y="84" text-anchor="middle" class="fig-small">average m</text>
<rect x="380.0" y="36" width="120.0" height="30" class="fill-sunk stroke-ink"/>
<text x="440.0" y="56" text-anchor="middle" class="fig-small fig-bold">4 B</text>
<text x="440.0" y="84" text-anchor="middle" class="fig-small">average v</text>
<text x="20" y="130" class="fig-bold">Seven billion parameters: the same pieces times 7 billion</text>
<rect x="20.0" y="142" width="65.3" height="34" class="fill-soft-a stroke-ink"/>
<text x="52.7" y="164" text-anchor="middle" class="fig-small fig-bold">14</text>
<rect x="85.3" y="142" width="65.3" height="34" class="fill-soft-d stroke-ink"/>
<text x="118.0" y="164" text-anchor="middle" class="fig-small fig-bold">14</text>
<rect x="150.7" y="142" width="130.7" height="34" class="fill-soft-c stroke-ink"/>
<text x="216.0" y="164" text-anchor="middle" class="fig-small fig-bold">28</text>
<rect x="281.3" y="142" width="130.7" height="34" class="fill-soft-b stroke-ink"/>
<text x="346.7" y="164" text-anchor="middle" class="fig-small fig-bold">28</text>
<rect x="412.0" y="142" width="130.7" height="34" class="fill-sunk stroke-ink"/>
<text x="477.3" y="164" text-anchor="middle" class="fig-small fig-bold">28</text>
<text x="542.7" y="196" text-anchor="end" class="fig-bold">112 GB in total</text>
<line x1="132.0" y1="132" x2="132.0" y2="236" class="stroke-bad dashed"/>
<text x="137.0" y="216" class="fig-small">24 GB GPU</text>
<line x1="393.3" y1="132" x2="393.3" y2="236" class="stroke-bad dashed"/>
<text x="398.3" y="216" class="fig-small">80 GB GPU</text>
<line x1="20" y1="236" x2="580.0" y2="236" class="stroke-muted"/>
<line x1="20.0" y1="232" x2="20.0" y2="240" class="stroke-muted"/>
<text x="20.0" y="252" text-anchor="middle" class="fig-small">0</text>
<line x1="206.7" y1="232" x2="206.7" y2="240" class="stroke-muted"/>
<text x="206.7" y="252" text-anchor="middle" class="fig-small">40</text>
<line x1="393.3" y1="232" x2="393.3" y2="240" class="stroke-muted"/>
<text x="393.3" y="252" text-anchor="middle" class="fig-small">80</text>
<line x1="580.0" y1="232" x2="580.0" y2="240" class="stroke-muted"/>
<text x="580.0" y="252" text-anchor="middle" class="fig-small">120</text>
<text x="20" y="312" class="fig-small">GB, decimal. The first slice, the 16-bit weight, is the only part inference needs: 14 GB.</text>
</svg>
Figure 1. The weight is the thin first slice. Training adds four more pieces per parameter, so the bill is 8 times the weights and lands far above a 24 GB GPU.
```

The top strip adds up to 2 + 2 + 4 + 4 + 4 = 16 bytes. The bottom bar multiplies each piece by 7 billion: 14 GB for the weights and 14 GB for the gradients, then 28 GB each for the master copy and Adam's two averages, which are 32-bit. The total is 112 GB, so even an 80 GB GPU is too small.

Each piece has a reason to exist.

- The **weight** is the number itself, in 16 bits because 16-bit arithmetic is fast and halves the memory.
- The **gradient** says which way to move the weight. The backward pass produces all of them before the update starts, so they sit in memory together.
- The **first average** m is a running average of the gradient, which smooths a noisy step into a direction you can trust.
- The **second average** v is a running average of the squared gradient, so its square root is the weight's typical gradient size. Dividing m by it gives weights with tiny and huge gradients comparable steps. Both averages must stay precise, so they are 32-bit and cost 8 bytes together.
- The **master copy** exists because of rounding. In 16 bits, the spacing between numbers near 1.0 is about 0.001, so adding a typical step of 0.0001 to 1.0 rounds back to 1.0 and the update disappears. In 32 bits the sum gives 1.0001. Mixed-precision training therefore accumulates updates in a 32-bit master copy and rounds it to 16 bits for the fast arithmetic. That costs 4 more bytes.

## Add it up, then freeze most of the model

If you train $P$ parameters, the bill is

$$
\text{bytes} = 16 \times P
$$

Read it aloud: every trained parameter costs 16 bytes, whatever the model's shape. For the running example, $7 \times 10^9 \times 16 = 112 \times 10^9$ bytes, which is 112 GB. The figure is a floor, because training also keeps activations, the numbers the forward pass saves for the backward pass. A smaller batch shrinks those, but not the 112 GB, which depends only on $P$. Fewer bits per number trades away precision and still leaves a bill proportional to $P$.

So the lever is $P$. A frozen parameter keeps a fixed value, so it needs no gradient and no averages. It still has to sit in memory, because the forward pass reads it, so it costs 2 bytes, not zero. Train 1 percent of the 7 billion and freeze the rest. Trained: $7 \times 10^7$ parameters at 16 bytes is 1.12 GB. Frozen: $6.93 \times 10^9$ parameters at 2 bytes is 13.86 GB. The sum is 14.98 GB, which fits the 24 GB card with about 9 GB left for activations. With $N$ parameters in total and $P$ of them trained,

$$
\text{bytes} = 16P + 2(N - P)
$$

Read it aloud: pay 16 for each trained parameter and 2 for each frozen one. In numpy:

```python
bytes_needed = 16 * trained + 2 * (total - trained)
```

The 14 GB of frozen weights is a floor under the bill, and freezing saves less time than memory, because the backward pass still passes gradients through the frozen layers. The open question is which 1 percent to train.

> [!KEY]
> Training a parameter costs 16 bytes, not the 2 that storing it costs, because the gradient, the master copy, and Adam's two averages sit beside it. Freezing a parameter drops its cost back to 2 bytes, so the way to fit is to train fewer parameters.

## The bet: the update is simpler than the weight

Take one layer of the model, a weight $W$ with 4096 rows and 4096 columns, which is 16,777,216 numbers. Full fine-tuning changes all of them by some amount, which we call the update. A change as small as answering in a legal tone should not need 16.7 million independent adjustments. LoRA bets that the update is like a short sheet of corrections stapled to a thick book.

A small matrix trained directly cannot be added to $W$, because the shapes would not match. The update must be a full-size matrix built from few numbers, and a column times a row does exactly that. Multiply a column of 4 entries by a row of 4 entries and every cell of the result is one column entry times one row entry, so 8 numbers fill all 16 cells. A matrix made from one column and one row is called rank 1. Adding $r$ such layers gives a matrix of rank at most $r$, and we call $r$ the rank.

Stack the $r$ columns into a tall matrix $B$ with $d_{\text{out}}$ rows and $r$ columns, and the $r$ rows into a wide matrix $A$ with $r$ rows and $d_{\text{in}}$ columns. The product $BA$ adds up all $r$ layers in one step and has the shape of $W$.

## A 2 by 2 layer, by hand

The input width is 2, the output width is 2, and the rank is $r = 1$. Let $W$ be the identity, $A$ the row $[1, 2]$, and $B$ the column $[1, 0]$. A setting called $\alpha$ (alpha), which you choose, is 2, so the update is multiplied by $\alpha / r = 2$. Feed in $x = [1, 1]$. Figure 2 follows the numbers.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="The input [1, 1] goes through the frozen weight to [1, 1], and through A to 3, then B to [3, 0], scaled by 2 to [6, 0], and the two paths add to [7, 1]">
<defs><marker id="c052-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="20.0" y="111.0" width="34" height="26" class="fill-soft-a stroke-ink"/>
<text x="37.0" y="128.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="20.0" y="137.0" width="34" height="26" class="fill-soft-a stroke-ink"/>
<text x="37.0" y="154.5" text-anchor="middle" class="fig-bold">1</text>
<text x="37" y="187" text-anchor="middle" class="fig-small">x = [1, 1]</text>
<line x1="58.0" y1="129.0" x2="100.0" y2="74.0" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<rect x="104" y="50" width="110" height="40" rx="6" class="fill-sunk stroke-ink"/>
<text x="159.0" y="74.5" text-anchor="middle" class="fig-bold">W (frozen)</text>
<line x1="214" y1="70" x2="470" y2="70" class="stroke-ink"/>
<text x="330" y="62" text-anchor="middle" class="fig-small">x W^T = [1, 1]</text>
<line x1="58.0" y1="145.0" x2="100.0" y2="201.0" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<rect x="104" y="185" width="70" height="40" rx="6" class="fill-soft-c stroke-ink"/>
<text x="139.0" y="209.5" text-anchor="middle" class="fig-bold">A</text>
<line x1="174.0" y1="205.0" x2="206.0" y2="205.0" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<rect x="210.0" y="192.0" width="34" height="26" class="fill-soft-c stroke-ink"/>
<text x="227.0" y="209.5" text-anchor="middle" class="fig-bold">3</text>
<text x="227" y="239" text-anchor="middle" class="fig-small">r = 1</text>
<line x1="246.0" y1="205.0" x2="278.0" y2="205.0" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<rect x="282" y="185" width="70" height="40" rx="6" class="fill-soft-c stroke-ink"/>
<text x="317.0" y="209.5" text-anchor="middle" class="fig-bold">B</text>
<line x1="352.0" y1="205.0" x2="384.0" y2="205.0" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<rect x="388.0" y="179.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="405.0" y="196.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="388.0" y="205.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="405.0" y="222.5" text-anchor="middle" class="fig-bold">0</text>
<text x="405" y="249" text-anchor="middle" class="fig-small">[3, 0]</text>
<line x1="426" y1="205" x2="470" y2="205" class="stroke-ink"/>
<text x="240" y="175" text-anchor="middle" class="fig-small">squeeze to r numbers, expand back</text>
<text x="448" y="197" text-anchor="middle" class="fig-small">× 2</text>
<circle cx="470" cy="137" r="14" class="fill-paper stroke-ink"/><text x="470" y="142" text-anchor="middle" class="fig-bold">+</text>
<line x1="470" y1="70" x2="470" y2="122" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<line x1="470" y1="205" x2="470" y2="152" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<line x1="484.0" y1="137.0" x2="522.0" y2="137.0" class="stroke-ink" marker-end="url(#c052-arrow)"/>
<rect x="526.0" y="111.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="543.0" y="128.5" text-anchor="middle" class="fig-bold">7</text>
<rect x="526.0" y="137.0" width="34" height="26" class="fill-soft-b stroke-ink"/>
<text x="543.0" y="154.5" text-anchor="middle" class="fig-bold">1</text>
<text x="543" y="187" text-anchor="middle" class="fig-small">y = [7, 1]</text>
<text x="470" y="227" text-anchor="middle" class="fig-small">[6, 0]</text>
</svg>
Figure 2. The input splits into two paths. The top path is the frozen layer. The bottom path squeezes the input down to r = 1 number, expands it back, and scales it. The two paths add to give the output.
```

The top path is the frozen layer: $x W^{\top} = [1, 1]$. The bottom path multiplies by $A$, which squeezes the input down to $1 \cdot 1 + 2 \cdot 1 = 3$, a single number because $r = 1$. Then $B$ expands it to $[3, 0]$, and the scale 2 gives $[6, 0]$. Adding the paths gives $[1, 1] + [6, 0] = [7, 1]$. The narrow waist is the whole trick: the correction passes through only $r$ numbers.

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

Check the 2 by 2 numbers: $BA$ has first row $[1, 2]$ and second row $[0, 0]$, so $W_{\text{eff}} = [[3, 4], [0, 1]]$ and $x W_{\text{eff}}^{\top} = [7, 1]$. Both routes agree. Dividing by $r$ keeps the update's size roughly steady across ranks, which is a convention, not a law.

> [!WARNING]
> `W` has shape `(d_out, d_in)`, so the layer is `x @ W.T`. When `W` is square, `x @ W` still runs and returns different numbers with no error. The same holds for `B @ A` against `A @ B`: only `B @ A` has the shape of `W`.

### Where B and A start

Start $B$ at zero. Then $BA = 0$ and $W_{\text{eff}} = W$, so before the first step the adapted layer is exactly the pretrained one. Training starts from a model that already works.

Why not start $A$ at zero as well? Let $G$ be the gradient of the loss with respect to $W_{\text{eff}}$. The chain rule through $BA$, derived later in this chapter, gives $\partial L / \partial B = s \, G A^{\top}$ and $\partial L / \partial A = s \, B^{\top} G$, so each factor's gradient is built from the other factor. With $B = 0$ and $A$ nonzero, the gradient for $B$ is not zero but the gradient for $A$ is, so $B$ moves first and then $A$'s gradient wakes up. With $A = 0$ as well, both gradients are zero and training is stuck. If every row of $A$ were the same, the $r$ layers would be copies and rank $r$ would act like rank 1.

### Counting the numbers you train

The update has $r$ columns of length $d_{\text{out}}$ and $r$ rows of length $d_{\text{in}}$, so

$$
\text{LoRA parameters} = r \, (d_{\text{in}} + d_{\text{out}})
$$

Read it aloud: $r$ times the sum of the two widths, against the $d_{\text{in}} d_{\text{out}}$ of the full matrix. For the $4096 \times 4096$ weight at $r = 8$, you train $8 \times 8192 = 65{,}536$ numbers instead of 16,777,216, which is 256 times fewer. At 16 bytes each, their memory falls from 256 MiB to 1 MiB, and the frozen $W$ still takes 2 bytes per parameter.

> [!KEY]
> LoRA writes the update as $BA$ with $B$ tall and $A$ wide, so you train $r(d_{\text{in}} + d_{\text{out}})$ numbers, 256 times fewer for a 4096 by 4096 layer at rank 8, and starting $B$ at zero keeps step zero identical to the pretrained model.

## Merging the adapter back

After training, both paths multiply $x$ by a matrix, and adding two such results equals multiplying $x$ by the sum of the matrices. So you can add the matrices once, ahead of time. Figure 3 draws the layer before and after.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Before merging a layer has a frozen path and an adapter path; after merging it has one matrix, and the adapter's extra multiplies are gone">
<defs><marker id="c052-merge-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20" y="22" class="fig-bold">Before: two paths, 16,842,752 multiplies per example</text>
<text x="40" y="78" class="fig-small">x</text>
<line x1="52.0" y1="74.0" x2="110.0" y2="52.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<rect x="114" y="28" width="200" height="48" rx="6" class="fill-sunk stroke-ink"/>
<text x="214.0" y="52.0" text-anchor="middle" class="fig-bold">W (frozen)</text>
<text x="214.0" y="68.0" text-anchor="middle" class="fig-small">16,777,216 multiplies</text>
<line x1="52.0" y1="80.0" x2="110.0" y2="112.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<rect x="114" y="96" width="90" height="48" rx="6" class="fill-soft-c stroke-ink"/>
<text x="159.0" y="120.0" text-anchor="middle" class="fig-bold">A</text>
<text x="159.0" y="136.0" text-anchor="middle" class="fig-small">32,768</text>
<line x1="204.0" y1="120.0" x2="226.0" y2="120.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<rect x="230" y="96" width="90" height="48" rx="6" class="fill-soft-c stroke-ink"/>
<text x="275.0" y="120.0" text-anchor="middle" class="fig-bold">B</text>
<text x="275.0" y="136.0" text-anchor="middle" class="fig-small">32,768</text>
<line x1="314.0" y1="52.0" x2="430.0" y2="84.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<line x1="320.0" y1="120.0" x2="430.0" y2="92.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<circle cx="444" cy="88" r="14" class="fill-paper stroke-ink"/><text x="444" y="93" text-anchor="middle" class="fig-bold">+</text>
<line x1="458.0" y1="88.0" x2="500.0" y2="88.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<text x="504" y="93" class="fig-bold">y</text>
<text x="340" y="146" class="fig-small">extra path: 65,536 = 0.39% of the base</text>
<text x="20" y="196" class="fig-bold">After: one matrix, 16,777,216 multiplies per example</text>
<text x="40" y="250" class="fig-small">x</text>
<line x1="52.0" y1="246.0" x2="110.0" y2="246.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
<rect x="114" y="222" width="250" height="48" rx="6" class="fill-soft-b stroke-ink"/>
<text x="239.0" y="246.0" text-anchor="middle" class="fig-bold">W + (α/r) B A</text>
<text x="239.0" y="262.0" text-anchor="middle" class="fig-small">16,777,216 multiplies</text>
<line x1="364.0" y1="246.0" x2="430.0" y2="246.0" class="stroke-ink" marker-end="url(#c052-merge-arrow)"/>
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

On the 2 by 2 example the merged matrix is $[[3, 4], [0, 1]]$, and it gives $[7, 1]$, the same as the split path.

Per example, the adapter path costs one multiply for each trained parameter, 65,536 against 16,777,216 for the 4096 by 4096 layer at rank 8, or 0.39 percent. So the gain from merging is not mainly arithmetic. The merged layer needs no adapter-aware server code and reads only the base model's bytes per token, which is what decoding waits on, as in the chapter on [decoding and the KV cache](/lessons/053-decoding-and-sampling). Compute the merge in 32 bits and round once at the end, because a very small update can round away in 16 bits.

Merging is not always right. Suppose three customers each have an adapter on one 7-billion-parameter base. Three merged copies take 42 GB, while one base plus three adapters of 16 MiB each takes about 14 GB. A server with many adapters keeps them separate and adds the small extra path for each request. A server whose one adapter takes almost all the traffic should merge.

> [!KEY]
> The update is linear in the input, so $x W^{\top} + s (x A^{\top}) B^{\top}$ equals $x (W + s B A)^{\top}$ and you can fold the adapter into one ordinary matrix.

## What a rank-r update can reach

Suppose full fine-tuning would change a weight by a matrix $\Delta W$. A rank-$r$ update $s \, BA$ cannot equal it in general, so how close can it get?

The SVD splits a matrix into rank-1 layers, each weighted by a singular value, the largest first. Keeping the first $r$ layers gives a rank-$r$ matrix, and dropping the smallest layers loses the least. The loss is measured by the Frobenius norm, the square root of the sum of the squares of all the entries, which is the matrix version of a vector's length. The error of the best rank-$r$ approximation depends only on what you dropped:

$$
\text{error} = \sqrt{s_{r+1}^2 + s_{r+2}^2 + \cdots}
$$

Read it aloud: square the singular values you threw away, add them, and take the square root. This is the Eckart-Young result, and no rank-$r$ matrix is closer to $\Delta W$, so it is a ceiling on what any LoRA update of rank $r$ can match.

Take $M = [[2, 1], [1, 2]]$, whose singular values are 3 and 1. The best rank-1 matrix keeps the layer with value 3, which is $[[1.5, 1.5], [1.5, 1.5]]$. The difference from $M$ has four entries of size 0.5, so its squared entries add to $4 \times 0.25 = 1$ and the error is 1, as the formula gives from the dropped value alone. The energy is the share of the squared singular values you keep: $(s_1^2 + \dots + s_r^2) / (s_1^2 + s_2^2 + \dots)$. For $M$ at $r = 1$ it is $9 / 10 = 0.9$. In numpy:

```python
U, s, Vt = np.linalg.svd(delta_w, full_matrices=False)
approx = (U[:, :r] * s[:r]) @ Vt[:r]           # first r layers
energy = (s[:r] ** 2).sum() / (s ** 2).sum()
```

Whether low rank is a good bet depends on the shape of the spectrum. Figure 4 compares two updates.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two spectra of singular values: one decays fast and rank 2 keeps 94 percent of the energy, the other is flat and rank 2 keeps 41 percent">
<text x="20" y="24" class="fig-bold">Fast decay</text>
<line x1="20" y1="190" x2="270" y2="190" class="stroke-muted"/>
<rect x="30" y="60.0" width="28" height="130.0" class="fill-a"/>
<text x="44" y="55.0" text-anchor="middle" class="fig-small">6.00</text>
<text x="44" y="206" text-anchor="middle" class="fig-small">1</text>
<text x="44" y="226" text-anchor="middle" class="fig-small fig-bold">75%</text>
<rect x="70" y="125.0" width="28" height="65.0" class="fill-a"/>
<text x="84" y="120.0" text-anchor="middle" class="fig-small">3.00</text>
<text x="84" y="206" text-anchor="middle" class="fig-small">2</text>
<text x="84" y="226" text-anchor="middle" class="fig-small fig-bold">94%</text>
<rect x="110" y="157.5" width="28" height="32.5" class="fill-a"/>
<text x="124" y="152.5" text-anchor="middle" class="fig-small">1.50</text>
<text x="124" y="206" text-anchor="middle" class="fig-small">3</text>
<text x="124" y="226" text-anchor="middle" class="fig-small fig-bold">98%</text>
<rect x="150" y="173.8" width="28" height="16.2" class="fill-a"/>
<text x="164" y="168.8" text-anchor="middle" class="fig-small">0.75</text>
<text x="164" y="206" text-anchor="middle" class="fig-small">4</text>
<text x="164" y="226" text-anchor="middle" class="fig-small fig-bold">100%</text>
<rect x="190" y="181.9" width="28" height="8.1" class="fill-a"/>
<text x="204" y="176.9" text-anchor="middle" class="fig-small">0.38</text>
<text x="204" y="206" text-anchor="middle" class="fig-small">5</text>
<text x="204" y="226" text-anchor="middle" class="fig-small fig-bold">100%</text>
<rect x="230" y="185.9" width="28" height="4.1" class="fill-a"/>
<text x="244" y="180.9" text-anchor="middle" class="fig-small">0.19</text>
<text x="244" y="206" text-anchor="middle" class="fig-small">6</text>
<text x="244" y="226" text-anchor="middle" class="fig-small fig-bold">100%</text>
<line x1="104" y1="50" x2="104" y2="194" class="stroke-bad dashed"/>
<text x="20" y="246" class="fig-small">layer number, then energy kept up to it</text>
<text x="320" y="24" class="fig-bold">Flat</text>
<line x1="320" y1="190" x2="570" y2="190" class="stroke-muted"/>
<rect x="330" y="146.7" width="28" height="43.3" class="fill-d"/>
<text x="344" y="141.7" text-anchor="middle" class="fig-small">2.00</text>
<text x="344" y="206" text-anchor="middle" class="fig-small">1</text>
<text x="344" y="226" text-anchor="middle" class="fig-small fig-bold">22%</text>
<rect x="370" y="148.8" width="28" height="41.2" class="fill-d"/>
<text x="384" y="143.8" text-anchor="middle" class="fig-small">1.90</text>
<text x="384" y="206" text-anchor="middle" class="fig-small">2</text>
<text x="384" y="226" text-anchor="middle" class="fig-small fig-bold">41%</text>
<rect x="410" y="151.0" width="28" height="39.0" class="fill-d"/>
<text x="424" y="146.0" text-anchor="middle" class="fig-small">1.80</text>
<text x="424" y="206" text-anchor="middle" class="fig-small">3</text>
<text x="424" y="226" text-anchor="middle" class="fig-small fig-bold">58%</text>
<rect x="450" y="153.2" width="28" height="36.8" class="fill-d"/>
<text x="464" y="148.2" text-anchor="middle" class="fig-small">1.70</text>
<text x="464" y="206" text-anchor="middle" class="fig-small">4</text>
<text x="464" y="226" text-anchor="middle" class="fig-small fig-bold">74%</text>
<rect x="490" y="155.3" width="28" height="34.7" class="fill-d"/>
<text x="504" y="150.3" text-anchor="middle" class="fig-small">1.60</text>
<text x="504" y="206" text-anchor="middle" class="fig-small">5</text>
<text x="504" y="226" text-anchor="middle" class="fig-small fig-bold">88%</text>
<rect x="530" y="157.5" width="28" height="32.5" class="fill-d"/>
<text x="544" y="152.5" text-anchor="middle" class="fig-small">1.50</text>
<text x="544" y="206" text-anchor="middle" class="fig-small">6</text>
<text x="544" y="226" text-anchor="middle" class="fig-small fig-bold">100%</text>
<line x1="404" y1="50" x2="404" y2="194" class="stroke-bad dashed"/>
<text x="320" y="246" class="fig-small">layer number, then energy kept up to it</text>
<text x="20" y="280" class="fig-small">Dashed line: the cut after r = 2. Energy is the share of the squared singular values.</text>
</svg>
Figure 4. Left: the singular values halve each time, so two layers hold 94 percent of the energy. Right: the values are nearly equal, so two layers hold only 41 percent, and every layer matters.
```

If the singular values fall quickly, a small $r$ keeps almost everything, and the bet pays off. If they are flat, every layer carries real information, and a small $r$ throws a lot away. Reports on adapting language models usually find that quality rises with $r$ and then levels off, which fits the left panel, but where it levels off depends on the task, so test it.

## How training moves A and B

Does training find the best rank-$r$ update, and which directions can each step move? Let $L$ be the loss and $G$ its gradient with respect to $W_{\text{eff}}$, the matrix full fine-tuning would use. A small change $dB$ moves $W_{\text{eff}}$ by $s \, (dB) A$, so the loss changes by $G$ paired entry by entry with that change, which equals $s \, G A^{\top}$ paired with $dB$. Doing the same for $A$ gives

$$
\frac{\partial L}{\partial B} = s \, G A^{\top}, \qquad \frac{\partial L}{\partial A} = s \, B^{\top} G
$$

Read it aloud: the gradient for $B$ is $G$ seen through $A$, and the gradient for $A$ is $G$ seen through $B$, each scaled by $s$. The shapes agree with those of $B$ and $A$. In numpy:

```python
dA = (alpha / r) * (B.T @ G)   # (r, d_in)
dB = (alpha / r) * (G @ A.T)   # (d_out, r)
```

Use the layer from the 2 by 2 example: $W$ the identity, $A = [1, 2]$, $B = [1, 0]^{\top}$, $s = 2$, so $W_{\text{eff}} = [[3, 4], [0, 1]]$. Say the target is $[[1, 1], [2, 0]]$ and the loss is $L = \tfrac{1}{2} \lVert W_{\text{eff}} - \text{target} \rVert^2$. Then $G = W_{\text{eff}} - \text{target} = [[2, 3], [-2, 1]]$ and $L = 9$. The gradients are $dA = 2 \cdot [2, 3] = [4, 6]$ and $dB = 2 \cdot [8, 0]^{\top} = [16, 0]^{\top}$. 

Take one step of size $\eta = 0.01$. Then $A$ becomes $[0.96, 1.94]$ and $B$ becomes $[0.84, 0]^{\top}$, and $W_{\text{eff}}$ changes by $[[-0.3872, -0.7408], [0, 0]]$. The loss falls from 9 to about 6.35. Only the top row moved, so the step is not $-\eta G$, which would have moved both rows.

Matching the target means making $s\,BA$ equal to $\Delta W = \text{target} - W$, and the best a rank-$r$ update can do is the SVD answer. For this squared loss, every local minimum of the factorisation reaches it, though progress can stall on flat patches. For a real language model's loss there is no such guarantee, so treat the SVD error as the best case.

Each factor moves, so to first order in $\eta$ the weight changes by

$$
\Delta W_{\text{eff}} \approx -\eta \, s^2 \left( G A^{\top} A + B B^{\top} G \right)
$$

Read it aloud: the step is $G$, filtered by $A^{\top} A$ on the input side and by $B B^{\top}$ on the output side. With the example numbers, $G A^{\top} A + B B^{\top} G = [[10, 19], [0, 0]]$, so the prediction is $[[-0.4, -0.76], [0, 0]]$, close to the $[[-0.3872, -0.7408]]$ above, and the gap shrinks with $\eta$.

The filter $A^{\top} A$ is a true projection onto the rows of $A$ only when those rows have length 1 and are perpendicular. Otherwise it keeps only the directions $A$ spans, but weights each by its squared length: $A = [1, 2]$ has length $\sqrt{5}$, so $A^{\top} A$ is five times a projection, and the same holds for $B B^{\top}$. A LoRA step therefore follows the full gradient only inside the directions $A$ and $B$ already cover, so which directions they cover matters as much as how many.

> [!KEY]
> A rank-$r$ update can reach no closer to a full update than the dropped singular values allow, and training moves $W_{\text{eff}}$ along the full gradient only through the directions $A$ and $B$ already cover. Both limits are mild when the update's spectrum decays quickly, and severe when it is flat.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The bill is $16P + 2(N - P)$ bytes for $P$ trained parameters out of $N$.
> - The forward pass is `x @ W.T` plus `(alpha / r) * (x @ A.T) @ B.T`, with $B$ starting at zero and $A$ random. LoRA trains $r(d_{\text{in}} + d_{\text{out}})$ numbers, and the merge is `W + (alpha / r) * (B @ A)`.
> - The best rank-$r$ approximation keeps the first $r$ SVD layers, and energy kept is the first $r$ squared singular values over all of them.
> - The gradients are $dA = s B^{\top} G$ and $dB = s G A^{\top}$, with $s = \alpha / r$ and $G = W_{\text{eff}} - \text{target}$.

## Build it step by step

`x` is `(n, d_in)`, `W` is `(d_out, d_in)`, `A` is `(r, d_in)`, and `B` is `(d_out, r)`.

1. `adam_bytes_breakdown()` returns a dictionary with the keys `"weight"`, `"gradient"`, `"master_weight"`, `"first_average"`, and `"second_average"`, each mapped to its byte count. Write the five sizes, not the total.
2. `trainable_fraction_memory(total_params, trained_params)` returns the bytes when trained parameters cost 16 and frozen ones cost 2. Raise `ValueError` when `trained_params` is larger than `total_params`. The usual mistake is to charge the frozen parameters zero.
3. `lora_parameter_count(d_in, d_out, r)` returns $r (d_{\text{in}} + d_{\text{out}})$ as an int. The usual mistake is to return `d_in * d_out`.
4. `lora_init(d_in, d_out, r, seed)` returns the pair `(A, B)`. Draw `A` from a normal distribution with standard deviation `1 / sqrt(d_in)`, using `np.random.default_rng(seed)`, and make `B` all zeros. The usual mistake is to start both at zero.
5. `lora_forward(x, W, A, B, alpha, r)` returns shape `(n, d_out)`. Keep the two paths separate and add them. The usual mistake is to write `x @ W`.
6. `merge_lora(W, A, B, alpha, r)` returns the merged weight with shape `(d_out, d_in)`. The usual mistake is to write `A @ B`.
7. `best_rank_approximation(delta_w, r)` returns a matrix of the same shape as `delta_w`. Call `np.linalg.svd` with `full_matrices=False` and keep the first $r$ layers. The usual mistake is to keep the smallest singular values.
8. `energy_captured(delta_w, r)` returns the fraction of the squared singular values kept at rank `r`, a float between 0 and 1. The usual mistake is to forget to square the singular values.
9. `lora_gradients(W, A, B, alpha, r, target)` returns `(dA, dB)` for the loss $\tfrac{1}{2} \lVert W_{\text{eff}} - \text{target} \rVert^2$, where $W_{\text{eff}}$ comes from your `merge_lora`. The usual mistake is to drop $\alpha / r$ from one of the two.

## Where this breaks

A small $r$ cannot reach an update that is full rank. A matrix that permutes 64 outputs has all 64 singular values equal to 1, so rank 8 keeps $8 / 64 = 12.5$ percent of the energy, with an error of $\sqrt{56} \approx 7.48$ against a matrix of norm 8. Real updates are rarely that bad, but the right panel of Figure 4 is the shape to watch for.

Few trainable parameters limit how far the weights move, but they do not choose which skill moves, so mix in general data when you tune on one narrow task.

## The exercise

You write the nine functions above. The checks confirm that the five pieces sum to 16 and that the frozen-model bill is 14.98 GB for the 1 percent example. They check `lora_init`, the 2 by 2 example through `lora_forward` and `merge_lora`, and that merged and split outputs match. They also check the Eckart-Young error and the energy, and that `lora_gradients` matches the worked example, finite differences, and the first-order step formula.
