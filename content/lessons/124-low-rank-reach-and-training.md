---
title: "What a rank-r update can reach, and how it is trained"
slug: 124-low-rank-reach-and-training
part: "Part 7 — Large language models"
summary: The SVD says how close a rank-r matrix can get to a full update, and the chain rule says where training moves A and B.
minutes: 45
gpu: false
objectives:
  - Compute the best rank-r approximation of a matrix with the SVD, and its error from the dropped singular values.
  - Explain from a spectrum when a low-rank update loses little and when it loses a lot.
  - Derive the gradients of A and B by the chain rule, and check them with finite differences.
  - Say which part of the full gradient a LoRA step follows, and under what condition that part is a projection.
prereqs:
  - 123-merging-and-serving
  - 122-lora-layer
  - 050-compress-and-know-what-you-lost
exercise: 124-low-rank-reach-and-training
---

# What a rank-r update can reach, and how it is trained

A LoRA layer trains a few thousand numbers where full fine-tuning trains millions, so something must be given up. The first question is what. Suppose full fine-tuning would have changed a weight by a matrix $\Delta W$. A rank-$r$ update $s \, BA$ cannot equal that matrix in general, so how close can it get?

The second question is about training. You start from $B = 0$ and move $A$ and $B$ by gradient descent. Does that find the best rank-$r$ update, and which directions can each step move the weight in?

This chapter answers the first with the singular value decomposition, which writes a matrix as a stack of one-layer pieces ordered by size, and the second with the chain rule. Together they say what low rank can reach, and how training moves inside it.

## What you already know

- A rank-1 matrix is a column times a row, and a rank-$r$ matrix is a sum of $r$ of them, from the chapter "A low-rank update: build the LoRA layer".
- The adapted weight is $W_{\text{eff}} = W + s \, BA$ with $s = \alpha / r$, and `merge_lora` computes it.
- The singular value decomposition, SVD, writes a matrix as $U \,\text{diag}(s)\, V^{\top}$ with the singular values $s$ sorted largest first, as in the chapter "Compress, and know what you lost".
- The chain rule carries a derivative back through a product.
- Gradient descent moves each parameter by minus a step size times its gradient.

## The picture: a matrix as stacked layers

The SVD splits a matrix into rank-1 layers, each weighted by a singular value, the largest layer first. Keeping the first $r$ layers gives a rank-$r$ matrix. Figure 1 takes a 6 by 6 update and shows the best rank-1, rank-2, and rank-3 versions.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A 6 by 6 update and its best approximations of rank 1, 2, and 3, with the error under each">
<text x="20" y="28" class="fig-bold">update ΔW</text>
<circle cx="29.0" cy="49.0" r="6.9" class="fill-a"/>
<circle cx="47.0" cy="49.0" r="1.1" class="fill-b"/>
<circle cx="65.0" cy="49.0" r="3.6" class="fill-b"/>
<circle cx="83.0" cy="49.0" r="4.1" class="fill-b"/>
<circle cx="101.0" cy="49.0" r="4.4" class="fill-a"/>
<circle cx="119.0" cy="49.0" r="3.4" class="fill-a"/>
<circle cx="29.0" cy="67.0" r="2.4" class="fill-a"/>
<circle cx="47.0" cy="67.0" r="5.5" class="fill-a"/>
<circle cx="65.0" cy="67.0" r="1.2" class="fill-a"/>
<circle cx="83.0" cy="67.0" r="5.0" class="fill-a"/>
<circle cx="101.0" cy="67.0" r="0.9" class="fill-a"/>
<circle cx="119.0" cy="67.0" r="1.1" class="fill-b"/>
<circle cx="29.0" cy="85.0" r="2.0" class="fill-a"/>
<circle cx="47.0" cy="85.0" r="4.8" class="fill-a"/>
<circle cx="65.0" cy="85.0" r="4.2" class="fill-a"/>
<circle cx="83.0" cy="85.0" r="5.1" class="fill-a"/>
<circle cx="101.0" cy="85.0" r="3.7" class="fill-b"/>
<circle cx="119.0" cy="85.0" r="3.2" class="fill-a"/>
<circle cx="29.0" cy="103.0" r="4.9" class="fill-a"/>
<circle cx="47.0" cy="103.0" r="5.2" class="fill-b"/>
<circle cx="65.0" cy="103.0" r="3.1" class="fill-a"/>
<circle cx="83.0" cy="103.0" r="2.3" class="fill-b"/>
<circle cx="101.0" cy="103.0" r="3.1" class="fill-a"/>
<circle cx="119.0" cy="103.0" r="2.5" class="fill-a"/>
<circle cx="29.0" cy="121.0" r="6.9" class="fill-b"/>
<circle cx="47.0" cy="121.0" r="8.3" class="fill-b"/>
<circle cx="65.0" cy="121.0" r="5.4" class="fill-b"/>
<circle cx="83.0" cy="121.0" r="7.5" class="fill-b"/>
<circle cx="101.0" cy="121.0" r="3.7" class="fill-a"/>
<circle cx="119.0" cy="121.0" r="6.7" class="fill-b"/>
<circle cx="29.0" cy="139.0" r="0.6" class="fill-a"/>
<circle cx="47.0" cy="139.0" r="5.5" class="fill-a"/>
<circle cx="65.0" cy="139.0" r="5.4" class="fill-a"/>
<circle cx="83.0" cy="139.0" r="5.8" class="fill-a"/>
<circle cx="101.0" cy="139.0" r="3.2" class="fill-b"/>
<circle cx="119.0" cy="139.0" r="4.1" class="fill-a"/>
<rect x="20" y="40" width="108" height="108" class="stroke-muted" fill="none"/>
<text x="20" y="172" class="fig-small">singular values:</text>
<text x="20" y="189" class="fig-small">5.00, 2.50, 1.20,</text>
<text x="20" y="206" class="fig-small">0.50, 0.25, 0.10</text>
<text x="168" y="28" class="fig-bold">rank 1</text>
<circle cx="177.0" cy="49.0" r="0.8" class="fill-a"/>
<circle cx="195.0" cy="49.0" r="1.1" class="fill-a"/>
<circle cx="213.0" cy="49.0" r="0.8" class="fill-a"/>
<circle cx="231.0" cy="49.0" r="1.0" class="fill-a"/>
<circle cx="249.0" cy="49.0" r="0.5" class="fill-b"/>
<circle cx="267.0" cy="49.0" r="0.8" class="fill-a"/>
<circle cx="177.0" cy="67.0" r="3.2" class="fill-a"/>
<circle cx="195.0" cy="67.0" r="4.6" class="fill-a"/>
<circle cx="213.0" cy="67.0" r="3.2" class="fill-a"/>
<circle cx="231.0" cy="67.0" r="4.3" class="fill-a"/>
<circle cx="249.0" cy="67.0" r="2.2" class="fill-b"/>
<circle cx="267.0" cy="67.0" r="3.4" class="fill-a"/>
<circle cx="177.0" cy="85.0" r="3.5" class="fill-a"/>
<circle cx="195.0" cy="85.0" r="5.0" class="fill-a"/>
<circle cx="213.0" cy="85.0" r="3.5" class="fill-a"/>
<circle cx="231.0" cy="85.0" r="4.7" class="fill-a"/>
<circle cx="249.0" cy="85.0" r="2.4" class="fill-b"/>
<circle cx="267.0" cy="85.0" r="3.7" class="fill-a"/>
<circle cx="177.0" cy="103.0" r="1.7" class="fill-b"/>
<circle cx="195.0" cy="103.0" r="2.4" class="fill-b"/>
<circle cx="213.0" cy="103.0" r="1.6" class="fill-b"/>
<circle cx="231.0" cy="103.0" r="2.2" class="fill-b"/>
<circle cx="249.0" cy="103.0" r="1.1" class="fill-a"/>
<circle cx="267.0" cy="103.0" r="1.8" class="fill-b"/>
<circle cx="177.0" cy="121.0" r="5.9" class="fill-b"/>
<circle cx="195.0" cy="121.0" r="8.3" class="fill-b"/>
<circle cx="213.0" cy="121.0" r="5.8" class="fill-b"/>
<circle cx="231.0" cy="121.0" r="7.8" class="fill-b"/>
<circle cx="249.0" cy="121.0" r="4.0" class="fill-a"/>
<circle cx="267.0" cy="121.0" r="6.2" class="fill-b"/>
<circle cx="177.0" cy="139.0" r="4.1" class="fill-a"/>
<circle cx="195.0" cy="139.0" r="5.8" class="fill-a"/>
<circle cx="213.0" cy="139.0" r="4.0" class="fill-a"/>
<circle cx="231.0" cy="139.0" r="5.4" class="fill-a"/>
<circle cx="249.0" cy="139.0" r="2.7" class="fill-b"/>
<circle cx="267.0" cy="139.0" r="4.2" class="fill-a"/>
<rect x="168" y="40" width="108" height="108" class="stroke-muted" fill="none"/>
<text x="168" y="172" class="fig-small">error 2.83</text>
<text x="168" y="189" class="fig-small">from the dropped values</text>
<text x="168" y="206" class="fig-small">energy kept 75.7%</text>
<text x="316" y="28" class="fig-bold">rank 2</text>
<circle cx="325.0" cy="49.0" r="6.9" class="fill-a"/>
<circle cx="343.0" cy="49.0" r="2.9" class="fill-b"/>
<circle cx="361.0" cy="49.0" r="2.8" class="fill-b"/>
<circle cx="379.0" cy="49.0" r="3.7" class="fill-b"/>
<circle cx="397.0" cy="49.0" r="4.1" class="fill-a"/>
<circle cx="415.0" cy="49.0" r="3.7" class="fill-a"/>
<circle cx="325.0" cy="67.0" r="2.0" class="fill-a"/>
<circle cx="343.0" cy="67.0" r="4.7" class="fill-a"/>
<circle cx="361.0" cy="67.0" r="3.3" class="fill-a"/>
<circle cx="379.0" cy="67.0" r="4.5" class="fill-a"/>
<circle cx="397.0" cy="67.0" r="2.7" class="fill-b"/>
<circle cx="415.0" cy="67.0" r="3.1" class="fill-a"/>
<circle cx="325.0" cy="85.0" r="1.6" class="fill-a"/>
<circle cx="343.0" cy="85.0" r="5.2" class="fill-a"/>
<circle cx="361.0" cy="85.0" r="3.7" class="fill-a"/>
<circle cx="379.0" cy="85.0" r="5.0" class="fill-a"/>
<circle cx="397.0" cy="85.0" r="3.1" class="fill-b"/>
<circle cx="415.0" cy="85.0" r="3.3" class="fill-a"/>
<circle cx="325.0" cy="103.0" r="4.7" class="fill-a"/>
<circle cx="343.0" cy="103.0" r="3.3" class="fill-b"/>
<circle cx="361.0" cy="103.0" r="2.7" class="fill-b"/>
<circle cx="379.0" cy="103.0" r="3.6" class="fill-b"/>
<circle cx="397.0" cy="103.0" r="3.2" class="fill-a"/>
<circle cx="415.0" cy="103.0" r="2.0" class="fill-a"/>
<circle cx="325.0" cy="121.0" r="7.0" class="fill-b"/>
<circle cx="343.0" cy="121.0" r="8.3" class="fill-b"/>
<circle cx="361.0" cy="121.0" r="5.6" class="fill-b"/>
<circle cx="379.0" cy="121.0" r="7.6" class="fill-b"/>
<circle cx="397.0" cy="121.0" r="3.3" class="fill-a"/>
<circle cx="415.0" cy="121.0" r="6.5" class="fill-b"/>
<circle cx="325.0" cy="139.0" r="1.2" class="fill-a"/>
<circle cx="343.0" cy="139.0" r="6.0" class="fill-a"/>
<circle cx="361.0" cy="139.0" r="4.3" class="fill-a"/>
<circle cx="379.0" cy="139.0" r="5.8" class="fill-a"/>
<circle cx="397.0" cy="139.0" r="3.6" class="fill-b"/>
<circle cx="415.0" cy="139.0" r="3.7" class="fill-a"/>
<rect x="316" y="40" width="108" height="108" class="stroke-muted" fill="none"/>
<text x="316" y="172" class="fig-small">error 1.33</text>
<text x="316" y="189" class="fig-small">from the dropped values</text>
<text x="316" y="206" class="fig-small">energy kept 94.7%</text>
<text x="464" y="28" class="fig-bold">rank 3</text>
<circle cx="473.0" cy="49.0" r="6.9" class="fill-a"/>
<circle cx="491.0" cy="49.0" r="1.5" class="fill-b"/>
<circle cx="509.0" cy="49.0" r="3.8" class="fill-b"/>
<circle cx="527.0" cy="49.0" r="3.9" class="fill-b"/>
<circle cx="545.0" cy="49.0" r="4.3" class="fill-a"/>
<circle cx="563.0" cy="49.0" r="3.4" class="fill-a"/>
<circle cx="473.0" cy="67.0" r="2.0" class="fill-a"/>
<circle cx="491.0" cy="67.0" r="5.6" class="fill-a"/>
<circle cx="509.0" cy="67.0" r="0.7" class="fill-a"/>
<circle cx="527.0" cy="67.0" r="4.3" class="fill-a"/>
<circle cx="545.0" cy="67.0" r="2.2" class="fill-b"/>
<circle cx="563.0" cy="67.0" r="2.4" class="fill-a"/>
<circle cx="473.0" cy="85.0" r="1.6" class="fill-a"/>
<circle cx="491.0" cy="85.0" r="4.8" class="fill-a"/>
<circle cx="509.0" cy="85.0" r="4.2" class="fill-a"/>
<circle cx="527.0" cy="85.0" r="5.1" class="fill-a"/>
<circle cx="545.0" cy="85.0" r="3.2" class="fill-b"/>
<circle cx="563.0" cy="85.0" r="3.5" class="fill-a"/>
<circle cx="473.0" cy="103.0" r="4.7" class="fill-a"/>
<circle cx="491.0" cy="103.0" r="5.1" class="fill-b"/>
<circle cx="509.0" cy="103.0" r="3.1" class="fill-a"/>
<circle cx="527.0" cy="103.0" r="3.1" class="fill-b"/>
<circle cx="545.0" cy="103.0" r="2.7" class="fill-a"/>
<circle cx="563.0" cy="103.0" r="3.2" class="fill-a"/>
<circle cx="473.0" cy="121.0" r="7.0" class="fill-b"/>
<circle cx="491.0" cy="121.0" r="8.3" class="fill-b"/>
<circle cx="509.0" cy="121.0" r="5.6" class="fill-b"/>
<circle cx="527.0" cy="121.0" r="7.6" class="fill-b"/>
<circle cx="545.0" cy="121.0" r="3.3" class="fill-a"/>
<circle cx="563.0" cy="121.0" r="6.5" class="fill-b"/>
<circle cx="473.0" cy="139.0" r="1.3" class="fill-a"/>
<circle cx="491.0" cy="139.0" r="5.4" class="fill-a"/>
<circle cx="509.0" cy="139.0" r="5.2" class="fill-a"/>
<circle cx="527.0" cy="139.0" r="5.9" class="fill-a"/>
<circle cx="545.0" cy="139.0" r="3.8" class="fill-b"/>
<circle cx="563.0" cy="139.0" r="4.1" class="fill-a"/>
<rect x="464" y="40" width="108" height="108" class="stroke-muted" fill="none"/>
<text x="464" y="172" class="fig-small">error 0.57</text>
<text x="464" y="189" class="fig-small">from the dropped values</text>
<text x="464" y="206" class="fig-small">energy kept 99.0%</text>
<text x="20" y="236" class="fig-small">Blue dots are positive, orange negative, area grows with size. Each rank adds one layer to the one before.</text>
</svg>
Figure 1. Each added layer fixes less than the last, because the singular values fall from 5.00 to 0.10. By rank 3 the picture is close, and the error has dropped from 5.75 with nothing kept to 0.57.
```

Why keep the biggest layers? Each layer's contribution to the matrix is measured by its singular value, so dropping the smallest ones loses the least. The size of the loss is measured by the Frobenius norm, the square root of the sum of the squares of all the entries, which is the matrix version of a vector's length. The error of the best rank-$r$ approximation depends only on what you dropped:

$$
\text{error} = \sqrt{s_{r+1}^2 + s_{r+2}^2 + \cdots}
$$

Read it aloud: square the singular values you threw away, add them, and take the square root. This is the Eckart-Young result, and it also says no rank-$r$ matrix is closer to $\Delta W$ than this one, so it is a ceiling on what any LoRA update of rank $r$ can match.

## A 2 by 2 example

Take $M = [[2, 1], [1, 2]]$. Its singular values are 3 and 1. The best rank-1 matrix keeps the layer with value 3, which is $[[1.5, 1.5], [1.5, 1.5]]$. The difference from $M$ is $[[0.5, -0.5], [-0.5, 0.5]]$, whose squared entries add to $4 \times 0.25 = 1$, so the error is $\sqrt{1} = 1$. The formula gives the same, $\sqrt{1^2} = 1$, using only the dropped value.

A second number says how much was kept. The energy is the share of the squared singular values you keep: $(s_1^2 + \dots + s_r^2) / (s_1^2 + s_2^2 + \dots)$. For $M$ at $r = 1$ it is $9 / 10 = 0.9$. In numpy:

```python
U, s, Vt = np.linalg.svd(delta_w, full_matrices=False)
approx = (U[:, :r] * s[:r]) @ Vt[:r]           # first r layers
energy = (s[:r] ** 2).sum() / (s ** 2).sum()
```

> [!TIP]
> `U[:, :r] * s[:r]` scales each column by its singular value through broadcasting, so you do not need `np.diag(s)`.

## When low rank loses little

Whether low rank is a good bet depends on the shape of the spectrum. Figure 2 compares two updates.

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
Figure 2. Left: the singular values halve each time, so two layers hold 94 percent of the energy. Right: the values are nearly equal, so two layers hold only 41 percent, and every layer matters.
```

If the singular values fall quickly, a small $r$ keeps almost everything, and the bet that the update is simple pays off. If they are flat, every layer carries real information, and a small $r$ throws a lot away. Reports on adapting language models usually find that quality rises as $r$ grows from 1 and then levels off, often within the low tens, which fits updates that look like the left panel. How early it levels off depends on the task, so treat that as a habit to test rather than a guarantee.

> [!KEY]
> The best rank-$r$ matrix keeps the $r$ largest singular values, and its error is the square root of the squared values it drops. LoRA works when a full update's singular values fall quickly, and loses a lot when they stay flat.

## How training moves A and B

The second question is about the training step. Let $L$ be the loss and $G$ the gradient of $L$ with respect to $W_{\text{eff}}$, the matrix full fine-tuning would use. It has the shape of $W$. To get the gradients for $A$ and $B$, change one factor a little and see what happens. A small change $dB$ moves $W_{\text{eff}}$ by $s \, (dB) A$, and the loss changes by the sum of the entries of $G$ times that change. Rearranging the sum, that equals $s \, G A^{\top}$ paired with $dB$. Doing the same for $A$ gives

$$
\frac{\partial L}{\partial B} = s \, G A^{\top}, \qquad \frac{\partial L}{\partial A} = s \, B^{\top} G
$$

Read it aloud: the gradient for $B$ is $G$ seen through $A$, and the gradient for $A$ is $G$ seen through $B$, each scaled by $s$. The shapes agree: $G A^{\top}$ is `(d_out, r)`, the shape of $B$, and $B^{\top} G$ is `(r, d_in)`, the shape of $A$. In numpy:

```python
dA = (alpha / r) * (B.T @ G)   # (r, d_in)
dB = (alpha / r) * (G @ A.T)   # (d_out, r)
```

Use the layer from the last two chapters: $W$ the identity, $A = [1, 2]$, $B = [1, 0]^{\top}$, $s = 2$, so $W_{\text{eff}} = [[3, 4], [0, 1]]$. Say the target is $[[1, 1], [2, 0]]$ and the loss is $L = \tfrac{1}{2} \lVert W_{\text{eff}} - \text{target} \rVert^2$. Then $G = W_{\text{eff}} - \text{target} = [[2, 3], [-2, 1]]$ and $L = 9$. The gradients are $dA = 2 \cdot [2, 3] = [4, 6]$ and $dB = 2 \cdot [8, 0]^{\top} = [16, 0]^{\top}$. A finite-difference check in the exercise confirms these formulas.

Take one step of size $\eta = 0.01$. Then $A$ becomes $[0.96, 1.94]$ and $B$ becomes $[0.84, 0]^{\top}$, and $W_{\text{eff}}$ changes by $[[-0.3872, -0.7408], [0, 0]]$. The loss falls from 9 to about 6.35. Only the top row of $W_{\text{eff}}$ moved, and the step is not $-\eta G$, which would have moved both rows.

This loss ties the two halves of the chapter together. Since $W_{\text{eff}} = W + s\,BA$, matching the target means making $s\,BA$ equal to $\Delta W = \text{target} - W$, and the best a rank-$r$ update can do is the SVD answer from the first half. So does gradient descent find it? For this squared loss, factorisations like $BA$ are known to have no bad resting places: every local minimum reaches the best rank-$r$ error, although progress can stall for a while on flat patches. For the loss of a real language model there is no such guarantee, so treat the SVD error as the best case and not as a promise.

## Which direction the step follows

The change in $W_{\text{eff}}$ has two parts, one from each factor, because $W_{\text{eff}}$ depends on both. To first order in $\eta$, substituting the gradients gives

$$
\Delta W_{\text{eff}} \approx -\eta \, s^2 \left( G A^{\top} A + B B^{\top} G \right)
$$

Read it aloud: the step is $G$, but filtered by $A^{\top} A$ on the input side and by $B B^{\top}$ on the output side, with a size set by $\eta$ and $s^2$. The factor $s^2$ appears because each factor's gradient carries one $s$ and the weight change carries another. With the example numbers, $G A^{\top} A + B B^{\top} G = [[10, 19], [0, 0]]$, so the prediction is $-0.04 \times [[10, 19], [0, 0]] = [[-0.4, -0.76], [0, 0]]$, close to the $[[-0.3872, -0.7408]]$ you got above. The gap shrinks as $\eta$ shrinks.

Figure 3 shows what the filter does in the smallest case, one output and two inputs, with $B = 0$ so that only the $A$ side acts.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The full gradient G is an arrow; the LoRA step follows only its part along the row of A">
<defs><marker id="c124-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="40" y1="250" x2="260" y2="250" class="stroke-grid"/><line x1="60" y1="270" x2="60" y2="30" class="stroke-grid"/>
<text x="264" y="254" class="fig-small">input 1</text><text x="66" y="28" class="fig-small">input 2</text>
<line x1="32.0" y1="271.0" x2="228.0" y2="124.0" class="stroke-c dashed"/>
<text x="236.0" y="128.0" class="fig-small">the row of A</text>
<line x1="165.0" y1="110.0" x2="194.4" y2="149.2" class="stroke-muted dashed"/>
<line x1="60" y1="250" x2="165.0" y2="110.0" class="stroke-a" stroke-width="2.5" marker-end="url(#c124-arrow)"/>
<line x1="60" y1="250" x2="194.4" y2="149.2" class="stroke-b" stroke-width="2.5" marker-end="url(#c124-arrow)"/>
<text x="153.0" y="108.0" text-anchor="end" class="fig-bold">G = [1.5, 2.0]</text>
<text x="204.4" y="167.2" class="fig-bold">G A^T A = [1.92, 1.44]</text>
<text x="385" y="50" class="fig-small">A = [0.8, 0.6] has length 1,</text>
<text x="385" y="69" class="fig-small">so A^T A keeps the part of G</text>
<text x="385" y="88" class="fig-small">along the row of A:</text>
<text x="385" y="107" class="fig-small">G · A = 1.5·0.8 + 2.0·0.6 = 2.4</text>
<text x="385" y="126" class="fig-small">2.4 × [0.8, 0.6] = [1.92, 1.44]</text>
<text x="385" y="145" class="fig-small"></text>
<text x="385" y="164" class="fig-small">Grey dashes: the part of G the</text>
<text x="385" y="183" class="fig-small">step cannot reach.</text>
</svg>
Figure 3. The full gradient G points one way, but the step follows only the part of G along the row of A. The grey dashed piece is the part the step leaves out.
```

The filter $A^{\top} A$ is a true projection, which keeps exactly the part of $G$ along the rows of $A$, only when the rows of $A$ have length 1 and are perpendicular. Otherwise it is a stretched version: it still keeps only the directions $A$ spans, but weights each by its squared length. In the example above, $A = [1, 2]$ has length $\sqrt{5}$, so $A^{\top} A$ is five times a projection. The same is true of $B B^{\top}$ on the output side. This answers the second question: a LoRA step follows the full gradient only inside the directions $A$ and $B$ already cover, so which directions they cover matters as much as how many.

## Where this breaks

If the update full fine-tuning needs is full rank, a small $r$ cannot reach it. Consider an update that permutes 64 outputs, a permutation matrix. All 64 singular values equal 1, the spectrum is perfectly flat, and rank 8 keeps $8 / 64 = 12.5$ percent of the energy, with an error of $\sqrt{56} \approx 7.48$ against a matrix of norm 8. Real updates are rarely that bad, but the example shows the failure mode, and Figure 2's right panel is the shape to watch for.

Few trainable parameters limit how far the weights move, but they do not choose which skill moves. Fine-tuning on one narrow task can overwrite something the data never tests, and mixing in some general data helps.

> [!KEY]
> A rank-$r$ update can reach no closer to a full update than the dropped singular values allow, and training moves $W_{\text{eff}}$ along the full gradient only through the directions $A$ and $B$ already cover. Both limits are mild when the update's spectrum decays quickly, and severe when it is flat.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The best rank-$r$ approximation keeps the first $r$ layers of the SVD, and its Frobenius error is the square root of the sum of the dropped squared singular values.
> - Energy kept at rank $r$ is the sum of the first $r$ squared singular values over the sum of all of them.
> - The gradients are $dA = s B^{\top} G$ and $dB = s G A^{\top}$, with $s = \alpha / r$ and $G = W_{\text{eff}} - \text{target}$ for a squared loss.
> - One step changes $W_{\text{eff}}$ by about $-\eta s^2 (G A^{\top} A + B B^{\top} G)$.

## Build it step by step

The starter already contains `merge_lora` from the previous chapter, solved. Use it inside `lora_gradients`.

1. `best_rank_approximation(delta_w, r)` returns a matrix of the same shape as `delta_w`. Call `np.linalg.svd` with `full_matrices=False` and keep the first $r$ layers. The usual mistake is to keep the smallest singular values:

```python
def best_rank_approximation(delta_w, r):
    # (U[:, :r] * s[:r]) @ Vt[:r]
    return approx
```

2. `energy_captured(delta_w, r)` returns the fraction of the squared singular values kept at rank `r`, a float between 0 and 1. The usual mistake is to forget to square the singular values:

```python
def energy_captured(delta_w, r):
    # (s[:r] ** 2).sum() / (s ** 2).sum()
    return fraction
```

3. `lora_gradients(W, A, B, alpha, r, target)` returns `(dA, dB)` for the loss $\tfrac{1}{2} \lVert W_{\text{eff}} - \text{target} \rVert^2$, where $W_{\text{eff}}$ comes from `merge_lora`. The usual mistake is to drop $\alpha / r$ from one of the two:

```python
def lora_gradients(W, A, B, alpha, r, target):
    # G = merge_lora(...) - target; dA = s * B.T @ G; dB = s * G @ A.T
    return dA, dB
```

## The exercise

You write `best_rank_approximation`, `energy_captured`, and `lora_gradients`. The checks confirm the rank-1 answer on $[[2, 1], [1, 2]]$, that the approximation has rank $r$ and the Eckart-Young error, that it beats other rank-$r$ matrices, that the energy matches the error, that the gradients match the worked example and finite differences, that $B = 0$ gives a zero gradient for $A$, and that one small step moves $W_{\text{eff}}$ as the formula predicts.
