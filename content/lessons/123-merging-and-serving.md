---
title: Merging an adapter back into the weight
slug: 123-merging-and-serving
part: "Part 7 — Large language models"
summary: Fold the trained low-rank update into one ordinary matrix so the adapter costs nothing at inference, and see when keeping it separate is better.
minutes: 35
gpu: false
objectives:
  - Merge a trained adapter into the frozen weight, and explain why the merged layer gives the same outputs.
  - Explain why the product is B @ A and not A @ B, using shapes.
  - Count the extra multiplies an unmerged adapter adds per example.
  - Say when to merge and when to keep adapters separate, with the memory numbers for each.
prereqs:
  - 122-lora-layer
  - 121-fine-tuning-memory-bill
exercise: 123-merging-and-serving
---

# Merging an adapter back into the weight

Your adapter is trained. On the legal-tone task it does what you wanted, and you are ready to serve it. Every request takes a detour: each adapted layer runs the frozen weight, then squeezes the input through $A$, expands it through $B$, scales the result, and adds it back. You also have to ship two things, the base model and the adapter, and keep them paired.

The question is whether the adapter can disappear after training, leaving a model that is the same size as the original, runs at its speed, and needs no special code to load. If it cannot, every server that touches your model has to know about LoRA.

This chapter answers that with one line of algebra and then asks the honest follow-up: what changes when you serve many adapters, not one? The answer decides when merging is the right move and when it is the wrong one.

## What you already know

- The adapted layer computes `x @ W.T` plus `(alpha / r) * (x @ A.T) @ B.T`, from the chapter "A low-rank update: build the LoRA layer".
- `B` has shape `(d_out, r)` and `A` has shape `(r, d_in)`, so `B @ A` has the shape of `W`.
- Matrix multiplication distributes over addition: $x(P + Q) = xP + xQ$.
- A multiply-add costs one unit of arithmetic, and a layer's cost per example is about its number of weights.
- A model's file size is its parameter count times the bytes per number.

## The picture: two paths become one

Figure 1 draws the layer before and after. Before, the input takes two paths and the results add. After, there is one path through one matrix.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Before merging a layer has a frozen path and an adapter path; after merging it has one matrix, and the adapter's extra multiplies are gone">
<defs><marker id="c123-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20" y="22" class="fig-bold">Before: two paths, 16,842,752 multiplies per example</text>
<text x="40" y="78" class="fig-small">x</text>
<line x1="52.0" y1="74.0" x2="110.0" y2="52.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<rect x="114" y="28" width="200" height="48" rx="6" class="fill-sunk stroke-ink"/>
<text x="214.0" y="52.0" text-anchor="middle" class="fig-bold">W (frozen)</text>
<text x="214.0" y="68.0" text-anchor="middle" class="fig-small">16,777,216 multiplies</text>
<line x1="52.0" y1="80.0" x2="110.0" y2="112.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<rect x="114" y="96" width="90" height="48" rx="6" class="fill-soft-c stroke-ink"/>
<text x="159.0" y="120.0" text-anchor="middle" class="fig-bold">A</text>
<text x="159.0" y="136.0" text-anchor="middle" class="fig-small">32,768</text>
<line x1="204.0" y1="120.0" x2="226.0" y2="120.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<rect x="230" y="96" width="90" height="48" rx="6" class="fill-soft-c stroke-ink"/>
<text x="275.0" y="120.0" text-anchor="middle" class="fig-bold">B</text>
<text x="275.0" y="136.0" text-anchor="middle" class="fig-small">32,768</text>
<line x1="314.0" y1="52.0" x2="430.0" y2="84.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<line x1="320.0" y1="120.0" x2="430.0" y2="92.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<circle cx="444" cy="88" r="14" class="fill-paper stroke-ink"/><text x="444" y="93" text-anchor="middle" class="fig-bold">+</text>
<line x1="458.0" y1="88.0" x2="500.0" y2="88.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<text x="504" y="93" class="fig-bold">y</text>
<text x="340" y="146" class="fig-small">extra path: 65,536 = 0.39% of the base</text>
<text x="20" y="196" class="fig-bold">After: one matrix, 16,777,216 multiplies per example</text>
<text x="40" y="250" class="fig-small">x</text>
<line x1="52.0" y1="246.0" x2="110.0" y2="246.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<rect x="114" y="222" width="250" height="48" rx="6" class="fill-soft-b stroke-ink"/>
<text x="239.0" y="246.0" text-anchor="middle" class="fig-bold">W + (α/r) B A</text>
<text x="239.0" y="262.0" text-anchor="middle" class="fig-small">16,777,216 multiplies</text>
<line x1="364.0" y1="246.0" x2="430.0" y2="246.0" class="stroke-ink" marker-end="url(#c123-arrow)"/>
<text x="440" y="251" class="fig-bold">y</text>
<text x="20" y="306" class="fig-small">Both layers return the same y. The merged layer has nothing left to run except W.</text>
</svg>
Figure 1. Before merging, the adapter path adds 65,536 multiplies to the frozen layer's 16,777,216. After merging, only the one matrix is left and the output is the same.
```

The reason this works is that both paths are linear in $x$. Each path multiplies $x$ by a matrix, and adding two such results equals multiplying $x$ by the sum of the matrices. So you can add the matrices once, ahead of time, and multiply by the sum at every request.

## A 2 by 2 example, again

Take the layer from the previous chapter: $W$ the identity, $A = [1, 2]$, $B = [1, 0]^{\top}$, and $\alpha / r = 2$. The product $BA$ has first row $[1, 2]$ and second row $[0, 0]$, so $2BA$ is $[[2, 4], [0, 0]]$, and adding the identity gives the merged weight $[[3, 4], [0, 1]]$. Figure 2 shows the arithmetic.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The identity plus 2 times B A gives the merged matrix 3, 4, 0, 1, and the input 1, 1 gives 7, 1 through the split path or the merged one">
<rect x="20" y="50" width="40" height="40" class="fill-sunk stroke-ink"/>
<text x="40.0" y="75.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="60" y="50" width="40" height="40" class="fill-sunk stroke-ink"/>
<text x="80.0" y="75.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="20" y="90" width="40" height="40" class="fill-sunk stroke-ink"/>
<text x="40.0" y="115.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="60" y="90" width="40" height="40" class="fill-sunk stroke-ink"/>
<text x="80.0" y="115.0" text-anchor="middle" class="fig-bold">1</text>
<text x="60" y="38" text-anchor="middle" class="fig-small">W</text>
<text x="123" y="96" text-anchor="middle" class="fig-bold">+  2 ×</text>
<rect x="160" y="50" width="40" height="40" class="fill-soft-c stroke-ink"/>
<text x="180.0" y="75.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="200" y="50" width="40" height="40" class="fill-soft-c stroke-ink"/>
<text x="220.0" y="75.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="160" y="90" width="40" height="40" class="fill-soft-c stroke-ink"/>
<text x="180.0" y="115.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="200" y="90" width="40" height="40" class="fill-soft-c stroke-ink"/>
<text x="220.0" y="115.0" text-anchor="middle" class="fig-bold">0</text>
<text x="200" y="38" text-anchor="middle" class="fig-small">B A</text>
<text x="266" y="96" text-anchor="middle" class="fig-bold">=</text>
<rect x="290" y="50" width="40" height="40" class="fill-soft-b stroke-ink"/>
<text x="310.0" y="75.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="330" y="50" width="40" height="40" class="fill-soft-b stroke-ink"/>
<text x="350.0" y="75.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="290" y="90" width="40" height="40" class="fill-soft-b stroke-ink"/>
<text x="310.0" y="115.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="330" y="90" width="40" height="40" class="fill-soft-b stroke-ink"/>
<text x="350.0" y="115.0" text-anchor="middle" class="fig-bold">1</text>
<text x="330" y="38" text-anchor="middle" class="fig-small">merged weight</text>
<text x="400" y="80" class="fig-small">Row 1 of BA is [1, 2], row 2 is [0, 0],</text>
<text x="400" y="98" class="fig-small">because B = [1, 0] and A = [1, 2].</text>
<text x="20" y="168" class="fig-bold">Feed x = [1, 1] through both ways</text>
<text x="20" y="194" class="fig-small">Split:  x W^T + 2 (x A^T) B^T = [1, 1] + [6, 0] = [7, 1]</text>
<text x="20" y="216" class="fig-small">Merged:  x [[3, 4], [0, 1]]^T = [1·3 + 1·4, 1·0 + 1·1] = [7, 1]</text>
</svg>
Figure 2. Adding 2 B A to the identity gives the merged weight. The input [1, 1] gives [7, 1] through the split path and through the merged matrix, as the algebra says it must.
```

Both routes land on $[7, 1]$. That is not a coincidence of these numbers. It is the distributive law applied to the picture above.

## The formula

Write the merged weight and the layer's output as

$$
W_{\text{merged}} = W + \frac{\alpha}{r} \, BA, \qquad \hat{y} = x \, W_{\text{merged}}^{\top}
$$

Read it aloud: compute the product of the two thin factors, scale it, add it to the frozen weight once, and use the result as an ordinary weight from then on. In numpy:

```python
W_merged = W + (alpha / r) * (B @ A)   # (d_out, d_in)
y = x @ W_merged.T
```

Why `B @ A` and not `A @ B`? Look at the shapes. `B` is `(d_out, r)` and `A` is `(r, d_in)`, so `B @ A` is `(d_out, d_in)`, the shape of `W`, and the sum is defined. `A @ B` needs `d_in` to equal `d_out`, and even when it runs, on a square layer, it returns an `(r, r)` matrix, which cannot be added to `W`.

> [!WARNING]
> On a square layer both `B @ A` and `A @ B` run without an error. Only `B @ A` has the shape of `W`, so check the shape of what you add.

## What merging saves

Per example, the adapter path multiplies the input by $A$, which costs $r \, d_{\text{in}}$ multiplies, and then by $B$, which costs $d_{\text{out}} \, r$ more. The total is

$$
\text{extra multiplies} = r \, (d_{\text{in}} + d_{\text{out}})
$$

Read it aloud: the extra cost is the same count as the number of trained parameters, because each parameter is used once per example. For the 4096 by 4096 layer at rank 8 that is 65,536 against the base layer's 16,777,216, or 0.39 percent. In code:

```python
extra = r * (d_in + d_out)
```

That is a small share of the arithmetic, so be honest about why merging is still worth doing. The merged layer runs one matrix multiplication where the split layer runs three, and it needs no adapter-aware code in the server. Generating text reads every weight from memory once per token, as the chapter "Why generating text waits on memory" explained, so a merged model reads exactly the base model's bytes and nothing else. The saving is simplicity and a fixed, known speed, more than a large cut in arithmetic.

One caution about precision. If the base weights are stored in 16 bits, the merged weights are rounded to 16 bits, and a very small update can round away, for the same reason a 16-bit weight could not absorb a step of 0.0001 in the memory chapter. Compute the merge in 32 bits and round once at the end. When the base is stored in 4 bits, which is the QLoRA setup described below, merging is an approximation rather than an exact identity.

## When not to merge

Suppose you serve three customers, each with their own adapter, from one 7-billion-parameter base. Merging produces one full model per customer. Figure 3 counts the memory.

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
Figure 3. Three merged copies take 42 GB, because each is a full 14 GB model. One base with the adapters kept apart takes about 14 GB, since the adapters together are only 48 MiB.
```

At 2 bytes per number, one merged 7-billion-parameter copy is 14 GB, so three copies are 42 GB. The adapters are tiny by comparison. If each one covers four attention matrices of size 4096 by 4096 in each of 32 layers, at rank 8, that is $128 \times 65{,}536 = 8{,}388{,}608$ numbers, or 16 MiB, so three adapters are 48 MiB. A server that keeps the adapters separate loads the base once and adds the small extra path for whichever customer's request arrives. A server with one adapter that takes almost all the traffic should merge, because the extra path would be pure overhead.

Three relatives of LoRA make different choices. QLoRA stores the frozen base in 4 bits, half a byte per parameter, so a 7-billion-parameter base takes 3.5 GB, and trains a LoRA adapter on top. Adapters, in the older sense, insert small extra layers with a nonlinearity inside each block, so they stay in the forward pass at inference and cannot be merged away. Prefix tuning learns extra vectors that are added to the attention keys and values, so it spends context length on every request. LoRA is popular because its update is linear and folds in.

> [!KEY]
> A LoRA update is linear in the input, so $x W^{\top} + s (x A^{\top}) B^{\top}$ equals $x (W + s B A)^{\top}$ and you can fold the adapter into one ordinary matrix after training. The merged layer matches the split outputs and runs with no extra path, but each merged copy is a full model, so a server with many adapters keeps them separate.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The merged weight is $W + \frac{\alpha}{r} BA$, with `B @ A` in that order so the shape matches `W`.
> - Merging works because both paths are linear: the sum of two matrices applied to $x$ equals the two results added.
> - The unmerged path adds $r(d_{\text{in}} + d_{\text{out}})$ multiplies per example.
> - Merge when one adapter serves most traffic, and keep adapters separate when many share a base.

## Build it step by step

The starter already contains `lora_forward` from the previous chapter, solved. You write three functions.

1. `merge_lora(W, A, B, alpha, r)` returns the merged weight with shape `(d_out, d_in)`. The usual mistake is to write `A @ B`:

```python
def merge_lora(W, A, B, alpha, r):
    # W + (alpha / r) * (B @ A)
    return merged
```

2. `extra_multiplies(d_in, d_out, r)` returns the multiplies the unmerged adapter adds per example, as an int. The usual mistake is to count the base layer's multiplies too:

```python
def extra_multiplies(d_in, d_out, r):
    # r * d_in for A, plus d_out * r for B
    return count
```

3. `merged_matches_split(x, W, A, B, alpha, r)` returns the largest absolute difference between `x @ merged.T` and `lora_forward(x, W, A, B, alpha, r)`. It calls your `merge_lora`. The usual mistake is to compare against the base layer instead of the split path:

```python
def merged_matches_split(x, W, A, B, alpha, r):
    # np.abs(x @ merge_lora(...).T - lora_forward(...)).max()
    return gap
```

## The exercise

You write `merge_lora`, `extra_multiplies`, and `merged_matches_split`. The checks confirm that the merge gives $[[3, 4], [0, 1]]$ on the chapter's example, that its shape is right on a non-square layer, that merging changes $W$ by a matrix of rank at most $r$, that `extra_multiplies` returns 65,536 for the 4096 by 4096 layer at rank 8, and that the merged outputs match `lora_forward` to machine precision.
