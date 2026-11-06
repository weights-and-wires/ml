---
title: "Why decoding waits on memory, and the KV cache"
slug: 105-decoding-and-the-kv-cache
part: "Part 7 — Large language models"
summary: Writing one token reads every weight once, so generation waits on memory, and a KV cache of earlier keys and values adds a second read per user.
minutes: 50
gpu: false
objectives:
  - Explain why a model writing one token at a time is limited by memory bandwidth and not by arithmetic, and why a larger batch helps.
  - Compute flops per byte for a weight matrix times a batch, the ridge point, and the seconds per token of a 7-billion-parameter model.
  - Explain why the keys and values of earlier tokens never change, and what a cache saves and what it does not.
  - Compute the size of a KV cache, and explain how sharing key-value heads shrinks it.
  - Build a cache that appends one row per token and matches a full recompute.
prereqs:
  - 089-heads-and-causal-mask
  - 098-cost-of-training
exercise: 105-decoding-and-the-kv-cache
---

# Why decoding waits on memory, and the KV cache

Picture a 7-billion-parameter language model on a data-centre chip that can do 312 trillion arithmetic operations every second. The answer appears one token at a time, where a token is a word or a piece of a word. Writing one token takes about 14 billion operations, which the chip could finish in 0.045 milliseconds, or over 20,000 tokens per second. In practice a single user sees at most about 140 per second. Where does the other 99 percent of the time go?

It goes into waiting. The chip is a chef who cooks a dish in a blink, but the ingredients sit in a pantry down a long corridor, and each dish needs a fresh trip. A faster stove does nothing, and neither does a bigger recipe, which needs more ingredients down the same corridor.

The weights are not the only thing in the pantry. By token 1001 of a story, the model needs a key and a value from each of the 1000 tokens before it. It computed those while writing them, so why compute them again? Keeping them is the obvious fix, and the worry is that keeping them is another trip down the corridor. This chapter counts both trips.

## What you already know

- A flop is one arithmetic operation on a number, such as one multiply or one add.
- A forward pass costs about 2 flops per weight per token, and a weight in bf16 takes 2 bytes, as in "The cost of a run: flops, memory, and number formats".
- Attention scores a query against every earlier key, divides by the square root of the head width, and blends the values.
- A causal mask means token $t$ reads only tokens up to $t$, and each head has its own queries, keys, and values, as in "Many heads and the causal mask".

## The picture: one weight, one trip

When the model writes a token, each weight is fetched from memory, multiplied by one number, and added into a running sum. Figure 1 shows this on the left: a grid of weights meets one vector, and every cell does one multiply and one add. That is 2 flops for each 2-byte weight, so the chip does 1 flop for each byte it fetches.

```figure
<svg viewBox="0 0 600 306" role="img" aria-label="A grid of weights multiplied by one vector, then by four vectors: the same weight read feeds one pair of operations, then four">
<line x1="300" y1="14" x2="300" y2="284" class="stroke-grid"/>
<text x="119" y="24" text-anchor="middle" class="fig-bold">Batch 1: one vector</text>
<rect x="52" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="72" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="92" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="112" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="132" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="52" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="72" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="92" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="112" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="132" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="52" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="72" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="92" y="96" width="18" height="18" rx="2" class="fill-b"/>
<rect x="112" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="132" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="52" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="72" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="92" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="112" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="132" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="52" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="72" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="92" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="112" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="132" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<text x="159" y="111" text-anchor="middle">×</text>
<rect x="166" y="56" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="166" y="76" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="166" y="96" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="166" y="116" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="166" y="136" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<text x="102" y="172" text-anchor="middle" class="fig-small">weights</text>
<text x="176" y="172" text-anchor="middle" class="fig-small">vector</text>
<text x="449" y="24" text-anchor="middle" class="fig-bold">Batch 4: four vectors, one read</text>
<rect x="352" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="372" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="392" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="412" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="432" y="56" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="352" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="372" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="392" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="412" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="432" y="76" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="352" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="372" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="392" y="96" width="18" height="18" rx="2" class="fill-b"/>
<rect x="412" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="432" y="96" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="352" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="372" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="392" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="412" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="432" y="116" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="352" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="372" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="392" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="412" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<rect x="432" y="136" width="18" height="18" rx="2" class="fill-soft-a stroke-a"/>
<text x="459" y="111" text-anchor="middle">×</text>
<rect x="466" y="56" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="486" y="56" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="506" y="56" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="526" y="56" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="466" y="76" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="486" y="76" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="506" y="76" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="526" y="76" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="466" y="96" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="486" y="96" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="506" y="96" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="526" y="96" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="466" y="116" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="486" y="116" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="506" y="116" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="526" y="116" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="466" y="136" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="486" y="136" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="506" y="136" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<rect x="526" y="136" width="18" height="18" rx="2" class="fill-soft-d stroke-d"/>
<text x="402" y="172" text-anchor="middle" class="fig-small">weights</text>
<text x="506" y="172" text-anchor="middle" class="fig-small">4 vectors</text>
<text x="150" y="204" text-anchor="middle">each weight is read once: 2 bytes</text>
<text x="150" y="224" text-anchor="middle">and used once: 1 multiply, 1 add</text>
<text x="150" y="244" text-anchor="middle" class="fig-bold">2 flops per 2 bytes</text>
<text x="150" y="264" text-anchor="middle" class="fig-bold">= 1 flop per byte</text>
<text x="450" y="204" text-anchor="middle">the same 2-byte read now feeds</text>
<text x="450" y="224" text-anchor="middle">4 multiplies and 4 adds</text>
<text x="450" y="244" text-anchor="middle" class="fig-bold">8 flops per 2 bytes</text>
<text x="450" y="264" text-anchor="middle" class="fig-bold">= 4 flops per byte</text>
<text x="300" y="298" text-anchor="middle" class="fig-small">The orange cell is one weight in each panel.</text>
</svg>
Figure 1. One weight read from memory costs 2 bytes. With one vector it does a single multiply and add. With four vectors it does four, so the same read buys four times the work.
```

Why can't the weight do more work while it is there? Because each weight has only one number of the vector to multiply. The right side of Figure 1 shows the escape: give the same read more vectors. With four tokens processed together, the weight is fetched once but does four multiplies and four adds. We call the number of tokens that share one read of the weights the batch size $B$. They can come from different users, or from positions in one long prompt. They cannot come from one user's own future tokens, because each depends on the token before it. That is why a single user is stuck at batch 1.

## A matrix of 4096 by 4096, by hand

Take one 4096-by-4096 weight matrix in bf16, on a chip with 312 trillion flops per second and 2 trillion bytes per second of memory bandwidth. It holds 16,777,216 weights, so reading it moves 33,554,432 bytes, and one vector does 2 flops per weight, which is 33,554,432 flops. The ratio is exactly 1 flop per byte. Reading takes 16.8 microseconds at 2 trillion bytes per second. The arithmetic takes 0.108 microseconds at 312 trillion flops per second. The chip is idle for 99.4 percent of the time. Double the width to 8192 and both flops and bytes grow fourfold, so the ratio is still 1. Width cancels, which is why a bigger model does not escape the wait.

## The formula, read aloud

The ratio of flops to bytes has a name: arithmetic intensity. For a $d \times d$ matrix, a batch of $B$ vectors, and $b$ bytes per weight, the matrix does $2Bd^2$ flops and reads $bd^2$ bytes:

$$
I = \frac{2Bd^2}{b\,d^2} = \frac{2B}{b}
$$

Read it aloud: intensity is twice the batch size, divided by the bytes in one weight. With $B = 1$ and $b = 2$ you get 1, and with $B = 4$ you get 4, which matches Figure 1. In numpy that is `2 * batch / bytes_per_elt`.

## Where the chip stops waiting

How much intensity is enough? A chip does at most `peak_flops` per second and fetches at most `bandwidth` bytes per second. A workload with intensity $I$ gets at most $I \times$ bandwidth flops each second from memory, and the speed you get is the smaller of the two limits. They are equal at the ridge point:

$$
I_{\text{ridge}} = \frac{\text{peak flops}}{\text{bandwidth}} = \frac{312 \times 10^{12}}{2 \times 10^{12}} = 156
$$

Below 156 flops per byte, the workload is memory bound and the chip waits for data. Above it, the workload is compute bound and the chip runs out of arithmetic. In Figure 2 the sloped line is the memory limit, the flat line is the arithmetic limit, and they meet at the ridge.

```figure
<svg viewBox="0 0 600 328" role="img" aria-label="Roofline plot on log axes: a rising line up to a ridge point at 156 flops per byte, then a flat roof at 312 teraflops, with decode at batch 1 far down the slope">
<line x1="111.8" y1="70" x2="111.8" y2="270" class="stroke-grid"/>
<text x="111.8" y="286" text-anchor="middle" class="fig-small">1</text>
<line x1="250.6" y1="70" x2="250.6" y2="270" class="stroke-grid"/>
<text x="250.6" y="286" text-anchor="middle" class="fig-small">10</text>
<line x1="389.4" y1="70" x2="389.4" y2="270" class="stroke-grid"/>
<text x="389.4" y="286" text-anchor="middle" class="fig-small">100</text>
<line x1="528.2" y1="70" x2="528.2" y2="270" class="stroke-grid"/>
<text x="528.2" y="286" text-anchor="middle" class="fig-small">1000</text>
<line x1="70" y1="250.4" x2="570" y2="250.4" class="stroke-grid"/>
<text x="62" y="254.4" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="185.5" x2="570" y2="185.5" class="stroke-grid"/>
<text x="62" y="189.5" text-anchor="end" class="fig-small">10</text>
<line x1="70" y1="120.5" x2="570" y2="120.5" class="stroke-grid"/>
<text x="62" y="124.5" text-anchor="end" class="fig-small">100</text>
<line x1="70" y1="270" x2="570" y2="270" class="stroke-muted"/>
<line x1="70" y1="70" x2="70" y2="270" class="stroke-muted"/>
<path d="M70,250.4 L416.2,88.4 L570,88.4" fill="none" class="stroke-a" stroke-width="3"/>
<text x="74" y="30" class="fig-small">memory bound: speed = 2 TB/s × intensity</text>
<circle cx="111.8" cy="230.9" r="5" class="fill-b"/>
<text x="125.8" y="238.9" class="fig-bold">decode, batch 1: 2 of 312 TFLOPs</text>
<text x="125.8" y="255.9">about 0.6 percent of peak</text>
<circle cx="416.2" cy="88.4" r="5" class="fill-c"/>
<text x="404.2" y="78.4" text-anchor="end" class="fig-bold">ridge point: 156 flops per byte</text>
<circle cx="528.2" cy="88.4" r="5" class="fill-d"/>
<text x="534.2" y="110.4" text-anchor="end">prompt of 1000 tokens</text>
<text x="570" y="30" text-anchor="end" class="fig-small">compute bound: capped at 312 TFLOPs</text>
<text x="320" y="308" text-anchor="middle">flops per byte read (log scale)</text>
<text x="70" y="58" class="fig-small">achieved TFLOPs (log scale)</text>
</svg>
Figure 2. Speed is the lower of two limits. Left of the ridge the memory limit rules, and decode at batch 1 sits at 1 flop per byte, far below the roof. A long prompt reads each weight for many tokens at once, so it reaches the roof.
```

Decode, the step that writes one token, sits at intensity 1, so the memory limit allows 2 of the chip's 312 trillion flops, about 0.6 percent. Prefill, the step that reads a whole prompt, is different. A prompt of 1000 tokens is a batch of 1000 against one read of the weights, so it sits on the flat roof. Prefill is compute bound, so it is fast per token. Decode is not.

## Time per token, and batching

If decode waits on memory, the time per token is the time to fetch every weight once. A 7-billion-parameter model in bf16 is $7 \times 10^9 \times 2 = 14 \times 10^9$ bytes. At $2 \times 10^{12}$ bytes per second:

$$
t = \frac{\text{parameters} \times \text{bytes per weight}}{\text{bandwidth}} = \frac{14 \times 10^9}{2 \times 10^{12}} = 7\ \text{ms}
$$

Seven milliseconds per token is 143 tokens per second, and 156 times the arithmetic time, which is the ridge point again. Storing each weight in 1 byte halves it.

A server cannot make one user's token cheaper, but it can serve many users with one trip. With 156 users in a batch, every step still takes 7 ms but writes 156 tokens, about 22,000 per second in total, while each user still sees 143. Past 156 the chip runs out of arithmetic, so the step time grows with the batch and the total stops rising. Batching has a ceiling.

Batching shares the weights but not everything. Every user also has a private history of earlier tokens, and reading that history is a second trip that nobody else shares.

## The second trip: remembering earlier tokens

Without a cache, step $t$ runs the network over all $t$ tokens so far, so writing 1000 tokens costs $1 + 2 + \dots + 1000 = 500{,}500$ token-passes where 1000 would do. In Figure 3, each row is a decoding step and each cell is a token whose key and value that step needs. The left triangle computes every cell. The right triangle computes only the last cell of each row and reads the rest from storage.

```figure
<svg viewBox="0 0 600 274" role="img" aria-label="Two triangles of cells, one cell per token per step: without a cache every cell is computed again, with a cache only the last cell of each row is computed">
<line x1="300" y1="14" x2="300" y2="260" class="stroke-grid"/>
<text x="142" y="22" text-anchor="middle" class="fig-bold">Without a cache</text>
<text x="91" y="46" text-anchor="middle" class="fig-small">t1</text>
<text x="115" y="46" text-anchor="middle" class="fig-small">t2</text>
<text x="139" y="46" text-anchor="middle" class="fig-small">t3</text>
<text x="163" y="46" text-anchor="middle" class="fig-small">t4</text>
<text x="187" y="46" text-anchor="middle" class="fig-small">t5</text>
<text x="211" y="46" text-anchor="middle" class="fig-small">t6</text>
<text x="72" y="68" text-anchor="end" class="fig-small">step 1</text>
<rect x="80" y="52" width="22" height="22" rx="2" class="fill-b"/>
<text x="72" y="92" text-anchor="end" class="fig-small">step 2</text>
<rect x="80" y="76" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="104" y="76" width="22" height="22" rx="2" class="fill-b"/>
<text x="72" y="116" text-anchor="end" class="fig-small">step 3</text>
<rect x="80" y="100" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="104" y="100" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="128" y="100" width="22" height="22" rx="2" class="fill-b"/>
<text x="72" y="140" text-anchor="end" class="fig-small">step 4</text>
<rect x="80" y="124" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="104" y="124" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="128" y="124" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="152" y="124" width="22" height="22" rx="2" class="fill-b"/>
<text x="72" y="164" text-anchor="end" class="fig-small">step 5</text>
<rect x="80" y="148" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="104" y="148" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="128" y="148" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="152" y="148" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="176" y="148" width="22" height="22" rx="2" class="fill-b"/>
<text x="72" y="188" text-anchor="end" class="fig-small">step 6</text>
<rect x="80" y="172" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="104" y="172" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="128" y="172" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="152" y="172" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="176" y="172" width="22" height="22" rx="2" class="fill-soft-b stroke-b"/>
<rect x="200" y="172" width="22" height="22" rx="2" class="fill-b"/>
<text x="142" y="216" text-anchor="middle" class="fig-bold">21 key-value computations</text>
<text x="142" y="234" text-anchor="middle" class="fig-small">1 + 2 + 3 + 4 + 5 + 6</text>
<text x="442" y="22" text-anchor="middle" class="fig-bold">With a cache</text>
<text x="391" y="46" text-anchor="middle" class="fig-small">t1</text>
<text x="415" y="46" text-anchor="middle" class="fig-small">t2</text>
<text x="439" y="46" text-anchor="middle" class="fig-small">t3</text>
<text x="463" y="46" text-anchor="middle" class="fig-small">t4</text>
<text x="487" y="46" text-anchor="middle" class="fig-small">t5</text>
<text x="511" y="46" text-anchor="middle" class="fig-small">t6</text>
<text x="372" y="68" text-anchor="end" class="fig-small">step 1</text>
<rect x="380" y="52" width="22" height="22" rx="2" class="fill-b"/>
<text x="372" y="92" text-anchor="end" class="fig-small">step 2</text>
<rect x="380" y="76" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="404" y="76" width="22" height="22" rx="2" class="fill-b"/>
<text x="372" y="116" text-anchor="end" class="fig-small">step 3</text>
<rect x="380" y="100" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="404" y="100" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="428" y="100" width="22" height="22" rx="2" class="fill-b"/>
<text x="372" y="140" text-anchor="end" class="fig-small">step 4</text>
<rect x="380" y="124" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="404" y="124" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="428" y="124" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="452" y="124" width="22" height="22" rx="2" class="fill-b"/>
<text x="372" y="164" text-anchor="end" class="fig-small">step 5</text>
<rect x="380" y="148" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="404" y="148" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="428" y="148" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="452" y="148" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="476" y="148" width="22" height="22" rx="2" class="fill-b"/>
<text x="372" y="188" text-anchor="end" class="fig-small">step 6</text>
<rect x="380" y="172" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="404" y="172" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="428" y="172" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="452" y="172" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="476" y="172" width="22" height="22" rx="2" class="fill-sunk stroke-grid"/>
<rect x="500" y="172" width="22" height="22" rx="2" class="fill-b"/>
<text x="442" y="216" text-anchor="middle" class="fig-bold">6 key-value computations</text>
<text x="442" y="234" text-anchor="middle" class="fig-small">one new row per step</text>
<rect x="110" y="252" width="12" height="12" rx="2" class="fill-b"/>
<text x="128" y="262" class="fig-small">computed this step</text>
<rect x="240" y="252" width="12" height="12" rx="2" class="fill-soft-b stroke-b"/>
<text x="258" y="262" class="fig-small">recomputed</text>
<rect x="350" y="252" width="12" height="12" rx="2" class="fill-sunk stroke-grid"/>
<text x="368" y="262" class="fig-small">read from the cache</text>
</svg>
Figure 3. Each row is a decoding step and each cell a token whose key and value that step needs. Without a cache the triangle is computed again every step, 21 cells for six tokens. With a cache only the diagonal is computed. The grey cells are still read, so the cache saves arithmetic, not memory traffic.
```

Why is reading the old cells safe? Because of the causal mask. Token 2 never looks at token 3, so token 3 cannot change token 2's hidden state at any layer. Token 2's key and value are functions of that state, so they are the same at step 3 as at step 2.

The cache removes the repeated computation, not the reading. Per-token work such as the projections and feed-forward layers drops from growing with $T^2$ to growing with $T$, but attention still compares the new query with all $t$ cached keys.

Take one head of width 2 and two cached tokens. The keys are $[1, 0]$ and $[0, 1]$, the values are $[1, 2]$ and $[3, 4]$, and the new query is $q = [2, 0]$. The scores are each key dotted with $q$, divided by $\sqrt{2}$:

$$
s = \frac{1}{\sqrt{2}}\,[2,\ 0] = [1.4142,\ 0]
$$

A softmax turns them into weights. The exponentials are $e^{1.4142} = 4.1133$ and $e^0 = 1$, which add to 5.1133, so the weights are $[0.8044, 0.1956]$. The output blends the two value rows:

$$
0.8044 \cdot [1, 2] + 0.1956 \cdot [3, 4] = [1.3911,\ 2.3911]
$$

Read aloud: score each stored key against the new query, turn the scores into shares, and add up the stored values in those shares. In numpy:

```python
scores = K @ q / np.sqrt(d_head)            # (T,)
weights = softmax(scores)                   # subtract the max first
out = weights @ V                           # (d_head,)
```

## Counting the bytes

Every layer stores one key and one value per key-value head. Take 24 layers, 8 key-value heads, and head width 64. One token adds $2 \times 24 \times 8 \times 64 = 24{,}576$ numbers, where the 2 is one key plus one value. In bf16 that is 48 KiB, and 2048 tokens make 96 MiB.

$$
\text{bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot \text{bytes per number}
$$

Read it aloud: two copies, for every layer, key-value head, position in the head, and token, each as many bytes as one number takes. In code:

```python
size = 2 * n_layer * n_kv * d_head * seq_len * bytes_per_elt
```

Only key-value heads are stored, however many query heads the model has. 96 MiB looks harmless, but with 32 layers, 8 key-value heads, head width 128, and 32,768 tokens, one sequence needs 4 GiB. A 7-billion-parameter model in bf16 holds about 13 GiB of weights, so an 80 GiB card has 67 GiB left, which fits only 16 such sequences. The cache also costs time. Every sequence's cache is a trip of its own, so four sequences read 16 GiB of cache per step, more than the 13 GiB of weights.

## Sharing key-value heads

In multi-head attention every head keeps its own notebook of keys and values, and only the notebooks are stored, because queries are computed fresh for each token. So let several query heads read from one shared notebook. Each query head still asks its own question, because it still has its own query vector, but they all look into the same stored keys and values. Grouped-query attention puts the query heads into groups that each share one key-value head. Multi-query attention is the extreme where all share one.

```figure
<svg viewBox="0 0 600 252" role="img" aria-label="Three panels, each with eight query heads on top: eight key-value heads, two, and one, with lines showing which query heads read which key-value head">
<text x="106" y="20" text-anchor="middle" class="fig-bold">Multi-head</text>
<rect x="12" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="36" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="60" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="84" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="108" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="132" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="156" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="180" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="12" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="36" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="60" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="84" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="108" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="132" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="156" y="150" width="20" height="24" rx="2" class="fill-a"/>
<rect x="180" y="150" width="20" height="24" rx="2" class="fill-a"/>
<line x1="22" y1="64" x2="22" y2="150" class="stroke-muted"/>
<line x1="46" y1="64" x2="46" y2="150" class="stroke-muted"/>
<line x1="70" y1="64" x2="70" y2="150" class="stroke-muted"/>
<line x1="94" y1="64" x2="94" y2="150" class="stroke-muted"/>
<line x1="118" y1="64" x2="118" y2="150" class="stroke-muted"/>
<line x1="142" y1="64" x2="142" y2="150" class="stroke-muted"/>
<line x1="166" y1="64" x2="166" y2="150" class="stroke-muted"/>
<line x1="190" y1="64" x2="190" y2="150" class="stroke-muted"/>
<text x="106" y="196" text-anchor="middle">8 key-value heads</text>
<text x="106" y="216" text-anchor="middle" class="fig-bold">cache: 1× size</text>
<text x="304" y="20" text-anchor="middle" class="fig-bold">Grouped-query</text>
<rect x="210" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="234" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="258" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="282" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="306" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="330" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="354" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="378" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="214" y="150" width="88" height="24" rx="2" class="fill-a"/>
<rect x="310" y="150" width="88" height="24" rx="2" class="fill-a"/>
<line x1="220" y1="64" x2="258" y2="150" class="stroke-muted"/>
<line x1="244" y1="64" x2="258" y2="150" class="stroke-muted"/>
<line x1="268" y1="64" x2="258" y2="150" class="stroke-muted"/>
<line x1="292" y1="64" x2="258" y2="150" class="stroke-muted"/>
<line x1="316" y1="64" x2="354" y2="150" class="stroke-muted"/>
<line x1="340" y1="64" x2="354" y2="150" class="stroke-muted"/>
<line x1="364" y1="64" x2="354" y2="150" class="stroke-muted"/>
<line x1="388" y1="64" x2="354" y2="150" class="stroke-muted"/>
<text x="304" y="196" text-anchor="middle">2 key-value heads</text>
<text x="304" y="216" text-anchor="middle" class="fig-bold">cache: 4× smaller</text>
<text x="502" y="20" text-anchor="middle" class="fig-bold">Multi-query</text>
<rect x="408" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="432" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="456" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="480" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="504" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="528" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="552" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="576" y="44" width="20" height="20" rx="2" class="fill-soft-d stroke-d"/>
<rect x="412" y="150" width="184" height="24" rx="2" class="fill-a"/>
<line x1="418" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="442" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="466" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="490" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="514" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="538" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="562" y1="64" x2="504" y2="150" class="stroke-muted"/>
<line x1="586" y1="64" x2="504" y2="150" class="stroke-muted"/>
<text x="502" y="196" text-anchor="middle">1 key-value head</text>
<text x="502" y="216" text-anchor="middle" class="fig-bold">cache: 8× smaller</text>
<text x="300" y="242" text-anchor="middle" class="fig-small">Purple squares are query heads and blue boxes are the key-value heads the cache stores.</text>
</svg>
Figure 4. Eight query heads in every panel. Count the blue boxes: 8, 2, and 1. Every box is stored for every token, so fewer boxes mean a cache 1, 4, or 8 times smaller, while each query head still asks its own question.
```

With 8 query heads, going from 8 key-value heads to 2 shrinks the cache by 4, and going to 1 shrinks it by 8. For the 4 GiB sequence above, one key-value head gives 0.5 GiB, and the card holds 133 sequences instead of 16. Models trained with shared heads lose little quality.

> [!KEY]
> A single user gets 140 tokens per second instead of 20,000 because each token needs every weight fetched once, and each fetched weight does only 2 flops. That is 1 flop per byte, against the 156 a chip needs to stay busy. A wider model does not change the ratio, and sharing one read among more tokens does.

> [!KEY]
> The causal mask guarantees that earlier keys and values never change, so the model stores them instead of recomputing them. That trades memory for arithmetic, and sharing key-value heads is the main way to shrink the bill.

> [!RECAP]
> Before you build:
> - Arithmetic intensity is flops divided by bytes, and for a weight matrix times a batch $B$ it is $2B/b$.
> - The ridge point is peak flops divided by bandwidth, and the smallest batch that reaches it is $B = \lceil I_{\text{ridge}} \cdot b / 2 \rceil$.
> - One decode step at batch 1 takes parameters times bytes per weight, divided by bandwidth.
> - The cache size is two copies times layers, key-value heads, head width, tokens, and bytes per number.
> - A cached step scores the query against all stored keys, softmaxes, and blends the stored values.

## Build it step by step

1. **`arithmetic_intensity(batch, bytes_per_elt)`.** It returns twice the batch, divided by the bytes per weight. Batch 1 with 2 bytes gives 1. The usual mistake is to include the width, which cancels.
2. **`decode_seconds_per_token(n_params, bytes_per_elt, bandwidth)`.** It returns the bytes of weights divided by the bandwidth in bytes per second. The usual mistake is to mix units, such as gigabytes with single numbers.
3. **`min_batch_to_saturate(bytes_per_elt, peak_flops, bandwidth)`.** Compute the ridge point, then solve $2B/b \ge$ ridge for $B$ and round up with `math.ceil`. It returns an `int`. The usual mistake is to round down, which lands just under the ridge, where the chip still waits.
4. **`kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt=2)`.** It returns the byte count as an int. The usual mistake is to forget the two copies.
5. **`KVCache`.** `__init__` creates one list of keys and one of values per layer. `append(layer, k, v)` adds one `(n_kv, d_head)` array to that layer's lists. `get(layer)` stacks each list on axis 0 with `np.stack` and returns `(K, V)`, or zero rows of shape `(0, n_kv, d_head)` for an empty layer, because `np.stack` fails on an empty list. The usual mistake is `np.concatenate`, which merges the token axis into the head axis.
6. **`cached_attention_step(q, K, V)`.** Take `q` of shape `(d_head,)` and `K`, `V` of shape `(T, d_head)`. Score, divide by the square root of the width, softmax with the maximum subtracted, and blend the value rows. The usual mistake is to forget the division by the square root.

## Where this breaks

The cache grows with every token, so a long conversation eventually does not fit beside the weights. Servers drop old entries, which forgets the start of the text, store the cache in fewer bits, which adds rounding error, or cap the context length, which turns long requests away. Sharing key-value heads keeps every token, but it has to be built into the model before training.

## The exercise

You write `arithmetic_intensity`, `decode_seconds_per_token`, `min_batch_to_saturate`, `kv_cache_bytes`, `KVCache`, and `cached_attention_step`. The checks confirm 1 flop per byte at batch 1, 7 ms for 7 billion bf16 parameters at 2 TB/s, a batch of 156 for the 312 TFLOPs chip, and the 96 MiB hand count. They check `get` shapes and order, one cached step against a manual softmax, and that decoding six tokens through the cache matches a full causal recompute.
