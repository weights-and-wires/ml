---
title: "The cost of a run: flops, memory, and number formats"
slug: 098-cost-of-training
part: "Part 6 — Transformers"
summary: Estimate a run's compute as 6 flops per weight per token and its memory as four model copies, and pick a format that keeps tiny gradients.
minutes: 40
gpu: false
objectives:
  - Estimate the flops of a training run with the 6ND rule, and explain where the 2N forward and 4N backward costs come from.
  - Estimate the memory of weights, gradients, and the two Adam averages, and say what the estimate leaves out.
  - Explain with a number line why fp16 loses small gradients while bf16 keeps them.
  - Show with the chain rule why loss scaling restores a gradient exactly, and simulate it.
prereqs:
  - 053-width-depth-and-cost
  - 059-adam-and-adamw
  - 097-the-training-loop
exercise: 098-cost-of-training
---

# The cost of a run: flops, memory, and number formats

You have 10 billion tokens of text, a model with 100 million weights, and one GPU. Before you start, you want three answers. How long will the run take? Will the model fit in the GPU's memory? And why do some people train in 16-bit numbers for speed, while others who try it find their gradients have vanished and no error was raised?

The first two answers come from arithmetic you can do on the back of an envelope, and doing it first saves you from launching a run that would take a month or crash on the first step. The third needs a look inside a number. This chapter works through all three, using the training step from the previous chapter as the thing being counted.

## What you already know

Five ideas carry this chapter:

- A flop is one floating-point operation, one addition or one multiplication of two numbers (see "Count the parameters, flops, and memory of a network").
- A weight is one number the model learns, and a matrix of $a \times b$ weights multiplies a vector of $b$ numbers in $a \times b$ multiplications and about as many additions.
- Adam keeps two extra numbers for every weight, $m$ and $v$ (see "Adam: a step size for every weight").
- A training step runs forward, loss, backward, clip, and update, and the backward pass fills a gradient for every weight (see "One training step and the loop around it").
- A gradient is a slope, and slopes near a good solution are often tiny.

## Count the work

Start with one token and one weight. The weight multiplies a number that arrives, and the product is added into a running total: one multiplication and one addition, two flops. A model with $N$ weights does this for every weight, so the forward pass costs about $2N$ flops per token.

The backward pass has twice the work, for a reason you can see in the same picture. Every weight sits between an input $a$ and an output, and the loss sends back a gradient $dy$ for that output. Two things are needed. The first is a gradient for the input $a$, which is $w \cdot dy$ and has to be passed further back. The second is a gradient for the weight, which is $a \cdot dy$, to update it. Each is one multiplication and one addition per weight, so each costs about $2N$, and backward costs $4N$ in all. Figure 1 draws the three jobs.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Bars showing forward cost 2N, backward cost 4N split into two halves, and the total 6N flops per token">
<text x="118" y="86" text-anchor="end" class="fig-bold">forward</text>
<rect x="130" y="60" width="160" height="40" class="fill-soft-a stroke-a"/>
<text x="210" y="85" text-anchor="middle" class="fig-bold">2N</text>
<text x="118" y="146" text-anchor="end" class="fig-bold">backward</text>
<rect x="130" y="120" width="160" height="40" class="fill-soft-b stroke-b"/>
<text x="210" y="137" text-anchor="middle" class="fig-bold">2N</text>
<text x="210" y="153" text-anchor="middle" class="fig-small">gradient for the inputs</text>
<rect x="290" y="120" width="160" height="40" class="fill-soft-c stroke-c"/>
<text x="370" y="137" text-anchor="middle" class="fig-bold">2N</text>
<text x="370" y="153" text-anchor="middle" class="fig-small">gradient for the weights</text>
<text x="118" y="216" text-anchor="end" class="fig-bold">one token</text>
<rect x="130" y="190" width="160" height="40" class="fill-soft-a stroke-a"/>
<text x="210" y="215" text-anchor="middle" class="fig-bold">2N</text>
<rect x="290" y="190" width="160" height="40" class="fill-soft-b stroke-b"/>
<text x="370" y="215" text-anchor="middle" class="fig-bold">2N</text>
<rect x="450" y="190" width="160" height="40" class="fill-soft-c stroke-c"/>
<text x="530" y="215" text-anchor="middle" class="fig-bold">2N</text>
<text x="370" y="254" text-anchor="middle" class="fig-small">6N flops in all: three jobs the size of one forward pass</text>
<text x="370" y="30" text-anchor="middle" class="fig-small">flops for one token, N weights (drawn for N = 1 million, so 2N = 2 million)</text>
</svg>
Figure 1. The flops one token costs for a model with N weights. The forward pass costs 2N. The backward pass has to produce a gradient for the inputs of every layer, to keep sending back, and a gradient for the weights, to update them, and each costs about as much as the forward pass. The total is 6N, three forward-sized jobs.
```

One token costs $2N + 4N = 6N$ flops, so a run over $D$ tokens costs:

$$
C \approx 6\,N\,D
$$

Read it aloud: six flops, times the number of weights, times the number of tokens. As a check, take a toy model with a million weights and ten thousand tokens: $6 \times 10^6 \times 10^4 = 6 \times 10^{10}$ flops. In code it is one line, `6 * params * tokens`.

Now the opening question. A model with $10^8$ weights on $10^{10}$ tokens costs $6 \times 10^{8} \times 10^{10} = 6 \times 10^{18}$ flops. Suppose your GPU sustains $10^{14}$ flops per second, a round number for a datacenter GPU doing 16-bit matrix work at a reasonable share of its peak. Check the figure for yours. The run takes $6 \times 10^{18} / 10^{14} = 6 \times 10^{4}$ seconds, about 16.7 hours. A model ten times larger, 1 billion weights, trained on 20 billion tokens, costs $6 \times 10^9 \times 2 \times 10^{10} = 1.2 \times 10^{20}$ flops, which is $1.2 \times 10^{6}$ seconds, about 14 days on the same GPU.

The rule is an estimate, not an exact count. It ignores the attention score table and the softmax, which add a little for long sequences. It also counts every weight, although a lookup in the embedding table does no multiplication. This chapter counts every weight, so the estimate is a little high for models with a large vocabulary.

## Count the memory

Time is half the budget. The other half is memory, and the weights are only one of the things you must keep. During training you hold four copies of the model's shape:

| Copy | Why you need it |
|---|---|
| Weights | The model itself. |
| Gradients | One per weight, filled by the backward pass. |
| $m$ | Adam's average of the gradient. |
| $v$ | Adam's average of the squared gradient. |

That is $4N$ numbers. If every number takes 4 bytes, as in fp32, the memory is $16N$ bytes. For $10^8$ weights that is 1.6 GB, which fits. For $10^9$ weights it is 16 GB, and the GPU still needs room for the activations, the intermediate values the backward pass reads, which grow with the batch size and the sequence length. Plain gradient descent needs only two copies, weights and gradients.

If a batch is too big for the activations, gradient accumulation helps: run several small batches, add their gradients, and step once. It shrinks the activations, but not the four copies. And halving every number to 2 bytes does not halve the whole bill in practice, because common recipes keep an fp32 copy of the weights as well. The next sections explain why 16 bits is worth the trouble, and what it risks.

## Look inside a number

A floating-point number is scientific notation in base 2: a sign, a size, and some digits, as in $1.5 \times 2^{k}$. The **exponent** is the $k$. Its bits decide how large and how small a number can be, which is the range. The **mantissa** is the $1.5$. Its bits decide how many digits you keep, which is the precision. Three formats matter here:

| Format | Exponent bits | Mantissa bits | Largest | Smallest normal | Gap between neighbours |
|---|---|---|---|---|---|
| fp32 | 8 | 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | about $10^{-7}$ of the value |
| fp16 | 5 | 10 | 65,504 | $6.1 \times 10^{-5}$ | about $10^{-3}$ of the value |
| bf16 | 8 | 7 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | about $8 \times 10^{-3}$ of the value |

The last column is the coarseness of the grid of numbers the format can hold. In fp16, the next number above 1000 is about 1 higher, so a gradient is only kept to about three digits. Figure 2 draws the ranges.

```figure
<svg viewBox="0 0 620 285" role="img" aria-label="Ranges of fp32, bf16 and fp16 on a logarithmic axis, with a gradient of 1e-8 marked, inside the first two and outside fp16">
<line x1="72.4" y1="70" x2="72.4" y2="230" class="stroke-grid"/>
<line x1="72.4" y1="230" x2="72.4" y2="236" class="stroke-muted"/>
<text x="72.4" y="252" text-anchor="middle" class="fig-small">1e-40</text>
<line x1="137.1" y1="70" x2="137.1" y2="230" class="stroke-grid"/>
<line x1="137.1" y1="230" x2="137.1" y2="236" class="stroke-muted"/>
<text x="137.1" y="252" text-anchor="middle" class="fig-small">1e-30</text>
<line x1="201.8" y1="70" x2="201.8" y2="230" class="stroke-grid"/>
<line x1="201.8" y1="230" x2="201.8" y2="236" class="stroke-muted"/>
<text x="201.8" y="252" text-anchor="middle" class="fig-small">1e-20</text>
<line x1="266.5" y1="70" x2="266.5" y2="230" class="stroke-grid"/>
<line x1="266.5" y1="230" x2="266.5" y2="236" class="stroke-muted"/>
<text x="266.5" y="252" text-anchor="middle" class="fig-small">1e-10</text>
<line x1="331.2" y1="70" x2="331.2" y2="230" class="stroke-grid"/>
<line x1="331.2" y1="230" x2="331.2" y2="236" class="stroke-muted"/>
<text x="331.2" y="252" text-anchor="middle" class="fig-small">1</text>
<line x1="395.9" y1="70" x2="395.9" y2="230" class="stroke-grid"/>
<line x1="395.9" y1="230" x2="395.9" y2="236" class="stroke-muted"/>
<text x="395.9" y="252" text-anchor="middle" class="fig-small">1e10</text>
<line x1="460.6" y1="70" x2="460.6" y2="230" class="stroke-grid"/>
<line x1="460.6" y1="230" x2="460.6" y2="236" class="stroke-muted"/>
<text x="460.6" y="252" text-anchor="middle" class="fig-small">1e20</text>
<line x1="525.3" y1="70" x2="525.3" y2="230" class="stroke-grid"/>
<line x1="525.3" y1="230" x2="525.3" y2="236" class="stroke-muted"/>
<text x="525.3" y="252" text-anchor="middle" class="fig-small">1e30</text>
<line x1="590" y1="70" x2="590" y2="230" class="stroke-grid"/>
<line x1="590" y1="230" x2="590" y2="236" class="stroke-muted"/>
<text x="590" y="252" text-anchor="middle" class="fig-small">1e40</text>
<line x1="40" y1="230" x2="590" y2="230" class="stroke-muted"/>
<text x="315" y="274" text-anchor="middle">size of the number, on a log scale (each tick is a factor of 10^10)</text>

<rect x="85.7" y="80" width="494.8" height="30" class="fill-soft-c stroke-c"/>

<rect x="85.7" y="125" width="494.7" height="30" class="fill-soft-d stroke-d"/>

<rect x="303.9" y="170" width="58.4" height="30" class="fill-soft-b stroke-b"/>
<rect x="284.4" y="179" width="19.5" height="12" class="fill-sunk stroke-muted"/>
<text x="44" y="76" text-anchor="middle" class="fig-small"></text>

<text x="460.6" y="100" text-anchor="middle" class="fig-small">fp32: exponent 8 bits, mantissa 23</text>
<text x="460.6" y="145" text-anchor="middle" class="fig-small">bf16: exponent 8 bits, mantissa 7</text>
<text x="279.4" y="190" text-anchor="middle" class="fig-small"></text>
<text x="460.6" y="189" text-anchor="middle" class="fig-small">fp16: exponent 5 bits, mantissa 10</text>
<text x="292.4" y="216" text-anchor="start" class="fig-small">thin strip: lower precision</text>
<line x1="279.4" y1="60" x2="279.4" y2="230" class="stroke-bad dashed" stroke-width="2"/>
<text x="279.4" y="52" text-anchor="middle" class="fig-bold">a gradient of 1e-8</text>
</svg>
Figure 2. The range of positive numbers each format can hold, on a log scale. fp32 and bf16 both reach from about 1e-38 to 1e38, because both spend 8 bits on the exponent. fp16 spends 5, so it reaches only from 6e-5 to 65,504, plus a thin strip of low-precision numbers down to 6e-8. The dashed gradient, 1e-8, sits inside the range of the first two and to the left of fp16.
```

Notice what happens to a small number in fp16. Below $6.1 \times 10^{-5}$, fp16 keeps going with a thin strip of subnormal numbers, which have lower precision, down to $6 \times 10^{-8}$. So a gradient of $10^{-6}$ is stored as $1.013 \times 10^{-6}$, about 1.3 percent off. A gradient of $10^{-7}$ is stored as $1.19 \times 10^{-7}$, 19 percent off. A gradient of $10^{-8}$, below half of the smallest subnormal, rounds to exactly 0. Precision collapses first, and then the number disappears with no error. bf16 trades the other way: its 8 exponent bits give it fp32's range, so $10^{-8}$ survives, but with only 7 mantissa bits it keeps about two to three digits.

## Rescue small gradients by scaling the loss

Why not stay in fp32? Because 16-bit numbers halve the memory traffic and run faster on GPUs built for them. So fp16 users look for a way to keep the gradients in range. The trick is to multiply the loss by a number $s$ before the backward pass, as Figure 3 shows.

```figure
<svg viewBox="0 0 620 275" role="img" aria-label="A gradient of 1e-8 lies left of the fp16 range, multiplying by 1024 moves it inside the range, and dividing by 1024 afterwards restores it">
<defs><marker id="c108-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="50" y1="110" x2="50" y2="210" class="stroke-grid"/>
<line x1="50" y1="210" x2="50" y2="216" class="stroke-muted"/>
<text x="50" y="232" text-anchor="middle" class="fig-small">1e-10</text>
<line x1="116.2" y1="110" x2="116.2" y2="210" class="stroke-grid"/>
<line x1="116.2" y1="210" x2="116.2" y2="216" class="stroke-muted"/>
<text x="116.2" y="232" text-anchor="middle" class="fig-small">1e-8</text>
<line x1="182.5" y1="110" x2="182.5" y2="210" class="stroke-grid"/>
<line x1="182.5" y1="210" x2="182.5" y2="216" class="stroke-muted"/>
<text x="182.5" y="232" text-anchor="middle" class="fig-small">1e-6</text>
<line x1="248.8" y1="110" x2="248.8" y2="210" class="stroke-grid"/>
<line x1="248.8" y1="210" x2="248.8" y2="216" class="stroke-muted"/>
<text x="248.8" y="232" text-anchor="middle" class="fig-small">1e-4</text>
<line x1="315" y1="110" x2="315" y2="210" class="stroke-grid"/>
<line x1="315" y1="210" x2="315" y2="216" class="stroke-muted"/>
<text x="315" y="232" text-anchor="middle" class="fig-small">1e-2</text>
<line x1="381.2" y1="110" x2="381.2" y2="210" class="stroke-grid"/>
<line x1="381.2" y1="210" x2="381.2" y2="216" class="stroke-muted"/>
<text x="381.2" y="232" text-anchor="middle" class="fig-small">1</text>
<line x1="447.5" y1="110" x2="447.5" y2="210" class="stroke-grid"/>
<line x1="447.5" y1="210" x2="447.5" y2="216" class="stroke-muted"/>
<text x="447.5" y="232" text-anchor="middle" class="fig-small">1e2</text>
<line x1="513.8" y1="110" x2="513.8" y2="210" class="stroke-grid"/>
<line x1="513.8" y1="210" x2="513.8" y2="216" class="stroke-muted"/>
<text x="513.8" y="232" text-anchor="middle" class="fig-small">1e4</text>
<line x1="580" y1="110" x2="580" y2="210" class="stroke-grid"/>
<line x1="580" y1="210" x2="580" y2="216" class="stroke-muted"/>
<text x="580" y="232" text-anchor="middle" class="fig-small">1e6</text>
<line x1="50" y1="210" x2="580" y2="210" class="stroke-muted"/>
<rect x="241.6" y="150" width="299.1" height="40" class="fill-soft-b stroke-b"/>
<rect x="141.9" y="160" width="99.7" height="20" class="fill-sunk stroke-muted"/>
<text x="381.2" y="175" text-anchor="middle" class="fig-bold">fp16 can hold these</text>
<circle cx="116.2" cy="170" r="6" class="fill-bad"/>
<text x="138.2" y="124" text-anchor="end" class="fig-small">gradient 1e-8</text>
<text x="138.2" y="138" text-anchor="end" class="fig-small">stored as 0</text>
<circle cx="216" cy="170" r="6" class="fill-a"/>
<text x="194" y="124" text-anchor="start" class="fig-small">× 1024 = 1.02e-5</text>
<text x="194" y="138" text-anchor="start" class="fig-small">stored as 1.0252e-5</text>
<path d="M124.2,100 L208,100" fill="none" class="stroke-ink" stroke-width="1.5" marker-end="url(#c108-arrow)"/>
<text x="206.1" y="92" text-anchor="middle" class="fig-small">multiply the loss by 1024</text>

<text x="315" y="262" text-anchor="middle" class="fig-small">Then divide the gradient by 1024 in fp32: 1.0012e-8, within 0.2 percent of 1e-8.</text>
</svg>
Figure 3. Loss scaling moves a gradient across the axis without changing its meaning. In fp16, 1e-8 rounds to zero. Multiplying the loss by 1024 multiplies every gradient by 1024, which lifts 1e-8 to about 1e-5, where fp16 can store it. Dividing by 1024 in fp32 afterwards gives back 1.0012e-8.
```

This is harmless because of the chain rule. The slope of $s \times L$ with respect to a weight $w$ is $s$ times the slope of $L$, so every gradient in the model is multiplied by exactly $s$ and nothing else changes:

$$
\frac{\partial (s\,L)}{\partial w} = s\,\frac{\partial L}{\partial w}
$$

Read it aloud: scale the loss by $s$ and every gradient scales by $s$. Take $s = 1024$. A gradient of $10^{-8}$ becomes $1.024 \times 10^{-5}$, which fp16 stores as $1.0252 \times 10^{-5}$, a rounding error of 0.1 percent. After the backward pass, convert the gradients to fp32 and divide by 1024 to get $1.0012 \times 10^{-8}$, within 0.2 percent of the truth. Without scaling, it was 0. The simulation in code is two lines:

```python
scaled = fp16_roundtrip(true_grad * scale)
recovered = scaled / scale
```

The division has to happen in fp32. If you divided in fp16, you would push the value back out of range.

A scale that is too large fails the other way. A gradient of 1.0 times $10^5$ exceeds fp16's largest number, 65,504, and becomes infinity. Training libraries watch for infinities, skip that step, and lower the scale, which is called dynamic loss scaling. bf16 needs none of this, because its range already holds the small gradients.

## The whole run on one page

Return to the training step from the previous chapter. Per token, the forward pass costs $2N$ and the backward pass $4N$. The clip and the update touch each weight a few times per step, not per token, so with thousands of tokens in a batch they are a rounding error next to $6N$ per token. The memory is four copies of the weights plus activations. And the number format decides whether the gradients in the backward pass arrive intact.

> [!KEY]
> Before a run, estimate it. Compute is about $6ND$ flops, two forward-sized jobs for the backward pass on top of the forward one. Memory is four copies of the model: weights, gradients, $m$, and $v$, or $16N$ bytes in fp32, plus activations. The number format decides whether tiny gradients survive: fp16 has a small range and loses them unless you scale the loss, and bf16 has fp32's range and does not. That answers the opening question: about 17 hours and 1.6 GB, with 16-bit numbers worth it as long as small gradients stay in range.

> [!RECAP]
> - Forward costs $2N$ flops per token, backward costs $4N$, and a run costs $6ND$.
> - Training holds four copies of the model, weights, gradients, $m$, and $v$: $16N$ bytes in fp32, and activations come on top.
> - The exponent sets the range and the mantissa sets the precision. fp16 has a small range, and bf16 has fp32's.
> - A gradient too small for fp16 loses precision, then rounds to zero without an error.
> - Multiplying the loss by $s$ multiplies every gradient by $s$. Divide in fp32 afterwards.

## Build it step by step

The checks test three functions, in this order. The file also holds `fp16_roundtrip`, already written, which casts a number to fp16 and back so you can see what survives.

1. **`training_flops(params, tokens)`.** Return `6 * params * tokens`. The usual mistake is to count only the forward pass, which reports $2ND$.
2. **`training_memory_bytes(params, bytes_per_number=4, copies=4)`.** Return the bytes for `copies` numbers per weight. The defaults describe fp32 weights, gradients, $m$, and $v$, so a million weights need 16 million bytes.
3. **`scaled_grad_survives(true_grad, scale)`.** Multiply `true_grad` by `scale`, pass the result through `fp16_roundtrip`, divide by `scale`, and return the recovered gradient. It should work on a number or an array. A scale of 1 loses $10^{-8}$, and a scale that is too large returns infinity.

## Where this breaks

- fp16 without loss scaling can turn small gradients into zeros or poorly rounded numbers, and raise no error. Check for it, or use bf16.
- A loss scale that is too large overflows to infinity. The step must be skipped, not applied.
- A first loss far from $\ln V$ usually points to the logits or the targets, not the number format. Check those first, as in "How a language model is scored: next-token loss and perplexity".
- The $6ND$ rule and the $16N$-byte rule are estimates. They leave out attention's extra flops, the activations, and any extra fp32 copies.

## The exercise

You write `training_flops`, `training_memory_bytes`, and `scaled_grad_survives`. The checks compare the flops with $6ND$ on the worked example and at the scale of a billion weights, and the memory with 16 bytes per weight and its variants. They then check that a gradient of $10^{-8}$ vanishes at scale 1 and returns within 1 percent at scale 1024, that an ordinary gradient is left alone, and that a scale that is too large gives infinity. Run prints the opening estimate and a loss-scaling table.
