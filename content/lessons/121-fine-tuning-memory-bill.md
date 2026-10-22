---
title: Why fine-tuning a big model runs out of memory
slug: 121-fine-tuning-memory-bill
part: "Part 7 — Large language models"
summary: Training one parameter costs 16 bytes, not 2, so a 7-billion-parameter model needs 112 GB to fine-tune and training fewer parameters is the lever.
minutes: 35
gpu: false
objectives:
  - Explain why training a parameter costs 16 bytes when storing it costs 2, naming each of the five pieces and why it exists.
  - Compute the memory bill for fully fine-tuning a model of any size, and compare it with a GPU's capacity.
  - Compute the bill when most parameters are frozen, and explain why frozen weights still cost memory.
  - Say what the 16-byte rule leaves out, and why the real bill is larger.
prereqs:
  - 066-adam
  - 120-speculative-decoding
exercise: 121-fine-tuning-memory-bill
---

# Why fine-tuning a big model runs out of memory

You have one GPU with 24 GB of memory and a language model with 7 billion parameters. You want to fine-tune it, which means training it a little more on your own data so it answers in your style. The model's weights are stored as 16-bit numbers, two bytes each, so they take 14 GB. That fits with room to spare. You start the training script, and it stops with an out-of-memory error before the first step finishes.

The error is confusing because the model fits. Running it to generate text works fine. The gap is that training holds far more than the weights. If you do not know what else is in memory, you can only guess at fixes: a smaller batch, a shorter text, a bigger GPU. Some of those help and most do not, because they leave the biggest cost untouched.

This chapter shows what training holds for every parameter, adds it up for a 7-billion-parameter model, and finds the one lever that shrinks the bill. The chapters after it use that lever to build a method called LoRA.

## What you already know

- A parameter is one number inside the model, and a 7-billion-parameter model has 7 billion of them.
- A gradient is the amount the loss changes when you nudge one parameter, so training computes one gradient per parameter at every step.
- Adam, from the earlier chapter on it, gives every weight its own step size by keeping running averages of that weight's gradient.
- A 16-bit number takes 2 bytes and a 32-bit number takes 4 bytes, so a 32-bit number has more digits to spare.
- Fine-tuning means continuing to train a pretrained model on your own data.

## The picture: one parameter, five pieces

Take a single weight. To answer a question, the model needs one copy of it, 2 bytes in 16-bit form. To train it, the computer holds four more things next to that copy. Figure 1 draws them for one parameter on top and for all 7 billion below.

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

The top strip adds up to 2 + 2 + 4 + 4 + 4 = 16 bytes. The bottom bar multiplies each piece by 7 billion: 14 GB for the weights and 14 GB for the gradients, then 28 GB each for the master copy and Adam's two averages, which are 32-bit, so each costs twice as much. The total is 112 GB, and the 24 GB line cuts through the second slice. Even an 80 GB GPU, the next size up, is too small.

## Why every piece has to exist

A list of five sizes is easy to forget, so here is the reason for each one.

The **weight** is the number itself, held in 16 bits because 16-bit arithmetic is fast and halves the memory. The **gradient** is the number that says which way to move this weight. The optimiser needs it at the moment of the update, and the backward pass produces gradients for every weight before the update starts, so all of them sit in memory together.

The optimiser is the rule that turns gradients into weight updates, and the one used here is Adam. Plain gradient descent needs nothing beyond the gradient, but Adam keeps two extra numbers per weight, shown in Figure 2.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Noisy gradients with their smoothed running average m, and two weights whose step sizes become comparable once divided by the typical gradient size">
<text x="30" y="22" class="fig-bold">One weight over 40 steps</text>
<line x1="30" y1="120" x2="330" y2="120" class="stroke-grid"/>
<text x="30" y="134" class="fig-small">gradient 0</text>
<circle cx="30.0" cy="61.9" r="3" class="fill-a"/>
<circle cx="37.7" cy="163.0" r="3" class="fill-a"/>
<circle cx="45.4" cy="97.6" r="3" class="fill-a"/>
<circle cx="53.1" cy="119.3" r="3" class="fill-a"/>
<circle cx="60.8" cy="116.8" r="3" class="fill-a"/>
<circle cx="68.5" cy="111.5" r="3" class="fill-a"/>
<circle cx="76.2" cy="151.2" r="3" class="fill-a"/>
<circle cx="83.8" cy="111.9" r="3" class="fill-a"/>
<circle cx="91.5" cy="125.8" r="3" class="fill-a"/>
<circle cx="99.2" cy="33.7" r="3" class="fill-a"/>
<circle cx="106.9" cy="101.8" r="3" class="fill-a"/>
<circle cx="114.6" cy="114.6" r="3" class="fill-a"/>
<circle cx="122.3" cy="113.0" r="3" class="fill-a"/>
<circle cx="130.0" cy="121.5" r="3" class="fill-a"/>
<circle cx="137.7" cy="130.0" r="3" class="fill-a"/>
<circle cx="145.4" cy="115.4" r="3" class="fill-a"/>
<circle cx="153.1" cy="96.2" r="3" class="fill-a"/>
<circle cx="160.8" cy="112.0" r="3" class="fill-a"/>
<circle cx="168.5" cy="85.7" r="3" class="fill-a"/>
<circle cx="176.2" cy="111.2" r="3" class="fill-a"/>
<circle cx="183.8" cy="106.3" r="3" class="fill-a"/>
<circle cx="191.5" cy="72.8" r="3" class="fill-a"/>
<circle cx="199.2" cy="94.8" r="3" class="fill-a"/>
<circle cx="206.9" cy="117.9" r="3" class="fill-a"/>
<circle cx="214.6" cy="110.8" r="3" class="fill-a"/>
<circle cx="222.3" cy="94.9" r="3" class="fill-a"/>
<circle cx="230.0" cy="64.2" r="3" class="fill-a"/>
<circle cx="237.7" cy="112.7" r="3" class="fill-a"/>
<circle cx="245.4" cy="112.2" r="3" class="fill-a"/>
<circle cx="253.1" cy="84.7" r="3" class="fill-a"/>
<circle cx="260.8" cy="126.3" r="3" class="fill-a"/>
<circle cx="268.5" cy="113.2" r="3" class="fill-a"/>
<circle cx="276.2" cy="87.4" r="3" class="fill-a"/>
<circle cx="283.8" cy="94.0" r="3" class="fill-a"/>
<circle cx="291.5" cy="104.8" r="3" class="fill-a"/>
<circle cx="299.2" cy="92.1" r="3" class="fill-a"/>
<circle cx="306.9" cy="169.0" r="3" class="fill-a"/>
<circle cx="314.6" cy="84.3" r="3" class="fill-a"/>
<circle cx="322.3" cy="127.9" r="3" class="fill-a"/>
<circle cx="330.0" cy="143.5" r="3" class="fill-a"/>
<polyline points="30.0,61.9 37.7,115.1 45.4,108.7 53.1,111.7 60.8,113.0 68.5,112.7 76.2,120.1 83.8,118.6 91.5,119.8 99.2,106.6 106.9,105.9 114.6,107.1 122.3,107.9 130.0,109.7 137.7,112.2 145.4,112.6 153.1,110.6 160.8,110.8 168.5,107.9 176.2,108.3 183.8,108.1 191.5,104.1 199.2,103.1 206.9,104.7 214.6,105.4 222.3,104.3 230.0,100.0 237.7,101.4 245.4,102.5 253.1,100.6 260.8,103.3 268.5,104.3 276.2,102.6 283.8,101.7 291.5,102.0 299.2,101.0 306.9,107.9 314.6,105.5 322.3,107.8 330.0,111.4" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="30" y="205" class="fig-small">Blue dots: each step's gradient.</text>
<text x="30" y="222" class="fig-small">Orange line: m, the running average.</text>
<text x="30" y="239" class="fig-small">m leans the way most dots lean.</text>
<text x="370" y="22" class="fig-bold">Two weights, same learning rate</text>
<text x="370" y="42" class="fig-small">bar height = size of the step</text>
<rect x="380" y="90.0" width="34" height="100.0" class="fill-a"/>
<rect x="420" y="189.0" width="34" height="1.0" class="fill-c"/>
<text x="417" y="208" text-anchor="middle" class="fig-small">m only</text>
<rect x="490" y="126.8" width="34" height="63.2" class="fill-a"/>
<rect x="530" y="126.8" width="34" height="63.2" class="fill-c"/>
<text x="527" y="208" text-anchor="middle" class="fig-small">m / sqrt(v)</text>
<line x1="372" y1="190" x2="590" y2="190" class="stroke-muted"/>
<text x="372" y="242" class="fig-small">P (blue) and Q (green): Q's gradients</text>
<text x="372" y="259" class="fig-small">are 100 times smaller. Dividing by</text>
<text x="372" y="276" class="fig-small">sqrt(v) gives them the same step.</text>
</svg>
Figure 2. Left: a weight's gradients are noisy, but their running average m leans the way they lean. Right: a weight with 100 times smaller gradients would move 100 times more slowly on m alone, and dividing by sqrt(v) gives both the same step.
```

The first, the **first average** m, is a running average of the gradient. One step's gradient is noisy, and the average smooths it into a direction you can trust. The second, the **second average** v, is a running average of the squared gradient, so its square root is the weight's typical gradient size. Dividing m by that size gives a weight with tiny gradients and a weight with huge gradients comparable steps. Adam needs both, for every weight, and they must stay precise, so they are 32-bit. Together they cost 8 bytes.

The last piece, the **master copy**, exists because of a rounding problem. Each step changes a weight by a small amount, often around 0.0001. In 16 bits, the spacing between neighbouring numbers near 1.0 is about 0.001, so adding 0.0001 to 1.0 rounds back to 1.0 and the update disappears. In 32 bits the same sum keeps its change and gives 1.0001. Mixed-precision training therefore keeps a 32-bit master copy where updates accumulate, and rounds it to a 16-bit working copy for the fast arithmetic. The master copy costs 4 more bytes.

> [!WARNING]
> Full 32-bit training lands on 16 bytes too, by a different route: a 4-byte weight, a 4-byte gradient, and two 4-byte averages. Counting 2 bytes per parameter because the model file is small is the mistake behind the out-of-memory error.

## Add it up for 7 billion parameters

Here is the whole bill as one rule. If you train $P$ parameters,

$$
\text{bytes} = 16 \times P
$$

Read it aloud: every trained parameter costs 16 bytes, whatever the model's shape. For the running example, $7 \times 10^9 \times 16 = 112 \times 10^9$ bytes, which is 112 GB in decimal units. In numpy it is one multiplication:

```python
bytes_needed = 16 * params
```

This is a floor, not the full bill. Training also keeps activations, the intermediate numbers the forward pass saves so the backward pass can use them, and these grow with the batch size and the text length. You can shrink activations with a smaller batch. You cannot shrink the 112 GB that way, because it depends only on how many parameters you train. The function you write is named `optimiser_memory_bytes` after the optimiser's two averages, the largest share. It counts all five pieces, so read it as the bill for the trained parameters.

## Why not buy a bigger GPU or shrink the batch?

Three escapes come to mind, and each fails or costs more than it saves.

A bigger GPU works: two 80 GB cards hold 160 GB. It also means renting twice the hardware and splitting the model across cards, for every fine-tune you run.

A smaller batch lowers activations only. The 112 GB stays.

Fewer bits per number does lower the bill, and practitioners use it, but it trades away precision and still leaves a bill proportional to $P$.

That last sentence points to the lever. The bill is $16P$, so the only way to shrink it without touching precision is to make $P$ smaller: train fewer parameters.

## Freeze most of the model

A frozen parameter keeps its fixed value during training. It needs no gradient and no averages, because nothing updates it. It still has to sit in memory, because the forward pass reads it, so it costs 2 bytes, not zero. Suppose you train 1 percent of the 7 billion and freeze the rest. Figure 3 compares the two bills.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Training every parameter needs 112 GB; freezing 99 percent and training 1 percent needs 14.98 GB">
<line x1="40" y1="270" x2="580" y2="270" class="stroke-muted"/>
<line x1="40" y1="230.0" x2="580" y2="230.0" class="stroke-bad dashed"/>
<text x="230" y="224.0" class="fig-small">24 GB GPU</text>
<rect x="90" y="83.3" width="110" height="186.7" class="fill-soft-b stroke-ink"/>
<text x="145" y="75.3" text-anchor="middle" class="fig-bold">112 GB</text>
<text x="145" y="288.0" text-anchor="middle" class="fig-small">train all</text>
<rect x="330" y="246.9" width="110" height="23.1" class="fill-sunk stroke-ink"/>
<rect x="330" y="245.0" width="110" height="1.9" class="fill-b stroke-ink"/>
<text x="385" y="222.0" text-anchor="middle" class="fig-bold">14.98 GB</text>
<text x="385" y="288.0" text-anchor="middle" class="fig-small">freeze 99%, train 1%</text>
<text x="450" y="262.4" class="fig-small">frozen: 13.86 GB</text>
<text x="470" y="239.0" class="fig-small">trained: 1.12 GB</text>
<line x1="466" y1="235.0" x2="442" y2="246.0" class="stroke-b"/>
</svg>
Figure 3. Freezing 99 percent removes the gradient and Adam pieces for those parameters but not their 2-byte weights. The bill drops from 112 GB to under 15 GB, and the trained slice at the top is a thin sliver.
```

Count it. Trained: $0.01 \times 7 \times 10^9 = 7 \times 10^7$ parameters at 16 bytes is 1.12 GB. Frozen: $6.93 \times 10^9$ parameters at 2 bytes is 13.86 GB. The sum is 14.98 GB, which fits on the 24 GB card with about 9 GB left for activations. As a formula, with $N$ parameters in total and $P$ of them trained,

$$
\text{bytes} = 16P + 2(N - P)
$$

Read it aloud: pay 16 for each trained parameter and 2 for each frozen one. In numpy:

```python
bytes_needed = 16 * trained + 2 * (total - trained)
```

Two honest limits apply. The 14 GB of frozen weights does not shrink, so there is a floor under the bill that no amount of freezing removes. And freezing saves less time than memory, because the backward pass still has to pass gradients through the frozen layers to reach the trained ones. The open question is which 1 percent to train. Training a few whole layers and freezing the rest is cheap, but blunt, because the frozen layers can no longer adapt. The next chapter builds a smarter choice.

> [!KEY]
> Training a parameter costs 16 bytes, not the 2 that storing it costs, because the gradient, the master copy, and Adam's two averages all live in memory beside it. That is why a 7-billion-parameter model fits on a 24 GB GPU for answering but needs 112 GB to train. Freezing a parameter drops its cost back to 2 bytes, so the way to fit is to train fewer parameters.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - One trained parameter holds five pieces: a 2-byte weight, a 2-byte gradient, a 4-byte master copy, and two 4-byte Adam averages. The sum is 16 bytes.
> - The bill is $16P$ for $P$ trained parameters, and $7 \times 10^9$ gives 112 GB.
> - A frozen parameter costs 2 bytes, so the bill is $16P + 2(N - P)$.
> - The bill leaves out activations, so the real number is higher.

## Build it step by step

Each step is one function, in the order the checks test them.

1. `adam_bytes_breakdown()` takes nothing and returns a dictionary with the keys `"weight"`, `"gradient"`, `"master_weight"`, `"first_average"`, and `"second_average"`, each mapped to its byte count. The checks sum the values, so write the sizes you reasoned out above, not the total:

```python
def adam_bytes_breakdown():
    # one entry per piece in Figure 1
    return {"weight": ..., ...}
```

2. `optimiser_memory_bytes(params, mode)` returns the bytes for `params` trained parameters. `mode` is `"full"` or `"lora"`, and both use the same rate, because Adam charges 16 bytes whichever parameters it trains. Raise `ValueError` for any other mode. The usual mistake is to charge the `"lora"` mode a different rate:

```python
def optimiser_memory_bytes(params, mode):
    # reject unknown modes, then 16 * params
    return nbytes
```

3. `trainable_fraction_memory(total_params, trained_params)` returns the bytes when trained parameters cost 16 and frozen ones cost 2. Raise `ValueError` when `trained_params` is larger than `total_params`. The usual mistake is to charge the frozen parameters zero:

```python
def trainable_fraction_memory(total_params, trained_params):
    # 16 * trained + 2 * (total - trained)
    return nbytes
```

## The exercise

You write `adam_bytes_breakdown`, `optimiser_memory_bytes`, and `trainable_fraction_memory`. The checks confirm that the five pieces sum to 16, that 7 billion parameters cost 112 GB in both modes, that an unknown mode raises `ValueError`, and that the frozen-model bill lands at 14.98 GB for the 1 percent example. They also check that training everything gives the same answer as `optimiser_memory_bytes`, and that training nothing leaves only the 2-byte weights.
