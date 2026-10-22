---
title: "Why generating text waits on memory"
slug: 117-why-decoding-waits-on-memory
part: "Part 7 — Large language models"
summary: Writing one token reads every weight once and does two flops with it, so generation waits on memory bandwidth, and only a larger batch changes that.
minutes: 35
gpu: false
objectives:
  - Explain why a model writing one token at a time is limited by memory bandwidth and not by arithmetic.
  - Compute flops per byte for a weight matrix times a batch, and the ridge point where a chip stops waiting.
  - Estimate the seconds per token of a 7-billion-parameter model from its size and the memory bandwidth.
  - Explain why a wider model does not fix the wait, and why a larger batch does.
prereqs:
  - 099-heads-and-causal-mask
  - 108-cost-of-training
exercise: 117-why-decoding-waits-on-memory
---

# Why generating text waits on memory

Picture a 7-billion-parameter language model on a data-centre chip that can do 312 trillion arithmetic operations every second. You ask a question, and the answer appears one token at a time, where a token is a word or a piece of a word. Writing one token takes about 14 billion operations, which the chip could finish in 0.045 milliseconds. At that rate the answer would stream out at over 20,000 tokens per second. In practice a single user sees at most about 140 per second. Where does the other 99 percent of the time go?

It goes into waiting. The chip is a chef who cooks a dish in a blink, but the ingredients sit in a pantry down a long corridor, and each dish needs a fresh trip. Buying a faster stove does nothing, because the stove was already idle. Making the recipe bigger does nothing either, because a bigger recipe needs more ingredients carried down the same corridor. The speed is set by the corridor.

This chapter turns that story into numbers. You count the work done per byte carried, find the point where a chip stops waiting, and see why feeding the chip several users at once is the one change that helps. The next chapters build on it: each one is a way to deal with the wait.

## What you already know

- A flop is one arithmetic operation on a number, such as one multiply or one add.
- A language model writes one token at a time, and each token uses every weight in the model once.
- A forward pass costs about 2 flops per weight per token, as in "The cost of a run: flops, memory, and number formats".
- A weight stored in bf16 takes 2 bytes, and it must travel from memory to the chip before it is used.
- A matrix-vector product multiplies each weight by one entry of the vector and adds up the results.

## The picture: one weight, one trip

Look at what happens to a single weight when the model writes a token. The weight is fetched from memory, multiplied by one number, and added into a running sum. Then it is not needed again until the next token. Figure 1 shows this on the left: a grid of weights meets one vector, and every cell does exactly one multiply and one add. That is 2 flops for each 2-byte weight, so the chip does 1 flop for each byte it fetches.

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

Why can't the weight do more work while it is there? Because the vector is a single token's numbers, and each weight has only one of those numbers to multiply. The right side of Figure 1 shows the escape: give the same read more vectors to multiply. If four tokens are processed together, the weight is still fetched once, but it now does four multiplies and four adds. The trip is the same and the work is four times larger. We call the number of tokens that share one read of the weights the batch size $B$. The tokens can come from four different users, or from four positions in one long prompt. They cannot come from one user's own future tokens, because each of those depends on the token before it, which does not exist yet. That is why a single user is stuck at batch 1.

## A matrix of 4096 by 4096, by hand

Take one weight matrix with 4096 rows and 4096 columns, stored in bf16, on a chip with 312 trillion flops per second and a memory that delivers 2 trillion bytes per second. It holds $4096 \times 4096 = 16{,}777{,}216$ weights, so reading it moves 33,554,432 bytes. Multiplying it by one vector does 2 flops per weight, which is also 33,554,432 flops. The ratio is exactly 1 flop per byte.

Now compare the two times. Reading 33,554,432 bytes at 2 trillion bytes per second takes 16.8 microseconds. Doing 33,554,432 flops at 312 trillion per second takes 0.108 microseconds. The arithmetic needs under 1 percent of the time the read takes, so the chip sits idle for the other 99.4 percent. Double the width to 8192 and the matrix has four times the weights. It needs four times the flops and four times the bytes, so the ratio is still 1. Width cancels, which is why a bigger model does not escape the wait.

## The formula, read aloud

The ratio of flops to bytes has a name: arithmetic intensity. For a $d \times d$ matrix, a batch of $B$ vectors, and $b$ bytes per weight, the matrix does $2Bd^2$ flops and reads $bd^2$ bytes:

$$
I = \frac{2Bd^2}{b\,d^2} = \frac{2B}{b}
$$

Read it aloud: intensity is twice the batch size, divided by the bytes in one weight. The width $d$ cancels, and that is the arithmetic behind the paragraph above. With $B = 1$ and $b = 2$ you get 1, and with $B = 4$ you get 4, which matches Figure 1. This ignores the cost of reading the vectors themselves, which is small while $B$ is much smaller than $d$. In numpy it is one line:

```python
flops_per_byte = 2 * batch / bytes_per_elt
```

## Where the chip stops waiting

How much intensity is enough? A chip can do at most `peak_flops` per second and fetch at most `bandwidth` bytes per second. A workload with intensity $I$ asks for $I$ flops for each byte, so the memory can feed at most $I \times$ bandwidth flops each second. The speed you get is the smaller of the two limits. They are equal at the ridge point:

$$
I_{\text{ridge}} = \frac{\text{peak flops}}{\text{bandwidth}} = \frac{312 \times 10^{12}}{2 \times 10^{12}} = 156
$$

Below 156 flops per byte, the workload is memory bound: the chip waits for data. Above it, the workload is compute bound: the chip runs out of arithmetic. Figure 2 draws both limits on log axes. The sloped line is the memory limit, the flat line is the arithmetic limit, and they meet at the ridge.

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

Decode, the step that writes one token, sits at intensity 1. At that point the memory limit allows $1 \times 2$ trillion, which is 2 of the chip's 312 trillion flops, about 0.6 percent. Prefill, the step that reads a whole prompt, is different. A prompt of 1000 tokens is a batch of 1000 against one read of the weights, so its intensity is 1000 and it sits on the flat roof. Prefill is fast per token because it is compute bound. Decode is slow per token because it is not.

## Time per token

If decode waits on memory, the time per token is the time to fetch every weight once. A 7-billion-parameter model in bf16 is $7 \times 10^9 \times 2 = 14 \times 10^9$ bytes. At $2 \times 10^{12}$ bytes per second:

$$
t = \frac{\text{parameters} \times \text{bytes per weight}}{\text{bandwidth}} = \frac{14 \times 10^9}{2 \times 10^{12}} = 7\ \text{ms}
$$

Seven milliseconds per token is 143 tokens per second. Compare the arithmetic: $2 \times 7 \times 10^9 = 14$ billion flops at 312 trillion per second is 0.045 ms. The read is 156 times slower than the arithmetic, which is the ridge point again. This estimate is a lower bound on time, because it ignores the cache of earlier tokens, which the next chapter adds, and the overhead of launching work. It also shows the other lever: store each weight in 1 byte and the time halves, because there are half the bytes to carry.

## Batching and prefill

A server cannot make one user's token cheaper, but it can serve many users with one trip. Put 156 users in a batch. Every step still reads the weights once, and still takes 7 ms, but now it writes 156 tokens. Figure 3 follows the step time for larger and larger batches, using the memory limit and the flop limit together.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Bars of decode step time for batch sizes 1 to 1200: flat at 7 milliseconds up to batch 156, then rising">
<text x="300" y="18" text-anchor="middle" class="fig-bold">Time for one decode step, by batch size</text>
<rect x="150" y="30" width="10" height="10" rx="2" class="fill-a"/>
<text x="166" y="39" class="fig-small">waiting on memory</text>
<rect x="300" y="30" width="10" height="10" rx="2" class="fill-b"/>
<text x="316" y="39" class="fig-small">waiting on arithmetic</text>
<line x1="110" y1="196" x2="590" y2="196" class="stroke-muted"/>
<rect x="130" y="176.9" width="40" height="19.1" rx="2" class="fill-a"/>
<text x="150" y="170.9" text-anchor="middle" class="fig-small">7.0 ms</text>
<text x="150" y="212" text-anchor="middle" class="fig-bold">1</text>
<text x="150" y="234" text-anchor="middle" class="fig-small">143</text>
<text x="150" y="254" text-anchor="middle" class="fig-small">143</text>
<rect x="198" y="176.9" width="40" height="19.1" rx="2" class="fill-a"/>
<text x="218" y="170.9" text-anchor="middle" class="fig-small">7.0 ms</text>
<text x="218" y="212" text-anchor="middle" class="fig-bold">8</text>
<text x="218" y="234" text-anchor="middle" class="fig-small">143</text>
<text x="218" y="254" text-anchor="middle" class="fig-small">1,143</text>
<rect x="266" y="176.9" width="40" height="19.1" rx="2" class="fill-a"/>
<text x="286" y="170.9" text-anchor="middle" class="fig-small">7.0 ms</text>
<text x="286" y="212" text-anchor="middle" class="fig-bold">32</text>
<text x="286" y="234" text-anchor="middle" class="fig-small">143</text>
<text x="286" y="254" text-anchor="middle" class="fig-small">4,571</text>
<rect x="334" y="176.9" width="40" height="19.1" rx="2" class="fill-a"/>
<text x="354" y="170.9" text-anchor="middle" class="fig-small">7.0 ms</text>
<text x="354" y="212" text-anchor="middle" class="fig-bold">156</text>
<text x="354" y="234" text-anchor="middle" class="fig-small">143</text>
<text x="354" y="254" text-anchor="middle" class="fig-small">22,286</text>
<rect x="402" y="159.3" width="40" height="36.7" rx="2" class="fill-b"/>
<text x="422" y="153.3" text-anchor="middle" class="fig-small">13.5 ms</text>
<text x="422" y="212" text-anchor="middle" class="fig-bold">300</text>
<text x="422" y="234" text-anchor="middle" class="fig-small">74</text>
<text x="422" y="254" text-anchor="middle" class="fig-small">22,286</text>
<rect x="470" y="122.6" width="40" height="73.4" rx="2" class="fill-b"/>
<text x="490" y="116.6" text-anchor="middle" class="fig-small">26.9 ms</text>
<text x="490" y="212" text-anchor="middle" class="fig-bold">600</text>
<text x="490" y="234" text-anchor="middle" class="fig-small">37</text>
<text x="490" y="254" text-anchor="middle" class="fig-small">22,286</text>
<rect x="538" y="49.1" width="40" height="146.9" rx="2" class="fill-b"/>
<text x="558" y="43.1" text-anchor="middle" class="fig-small">53.8 ms</text>
<text x="558" y="212" text-anchor="middle" class="fig-bold">1200</text>
<text x="558" y="234" text-anchor="middle" class="fig-small">19</text>
<text x="558" y="254" text-anchor="middle" class="fig-small">22,286</text>
<text x="8" y="212" class="fig-small">batch size</text>
<text x="8" y="234" class="fig-small">tokens/s per user</text>
<text x="8" y="254" class="fig-small">tokens/s in total</text>
<line x1="388" y1="48" x2="388" y2="196" class="stroke-muted dashed"/>
<text x="394" y="62" class="fig-small">ridge: B = 156</text>
</svg>
Figure 3. Up to batch 156 the step takes the same 7 ms, so total tokens per second grows with the batch while each user's speed holds. Past the ridge the step grows with the batch, and the total stops rising.
```

Up to the ridge, the step time is flat, so the total tokens per second grows in step with the batch, from 143 to about 22,000. Each user still waits 7 ms per token, so each user sees the same speed. Past the ridge the chip is out of arithmetic, so the step time grows with the batch and the total stops rising. That is the answer to the question "why not batch a thousand users?". Batching has a ceiling, and each user's wait gets worse once you hit it.

Batching shares the weights but not everything. Every user also has a private history of earlier tokens, and reading that history is another trip that no one else shares. At long contexts that trip can cost more than the weights. "The KV cache: remember, don't recompute" works out how much.

> [!KEY]
> A single user gets 140 tokens per second instead of 20,000 because each token needs every weight fetched once, and each fetched weight does only 2 flops. That is 1 flop per byte, against the 156 a chip needs to stay busy. Making the model wider does not change the ratio. Sharing one read among more tokens does.

> [!RECAP]
> Before you build:
> - Arithmetic intensity is flops divided by bytes, and for a weight matrix times a batch $B$ it is $2B/b$.
> - The ridge point is peak flops divided by bandwidth. Below it the chip waits on memory, and above it the chip waits on arithmetic.
> - One decode step at batch 1 reads every weight once, so its time is parameters times bytes per weight, divided by bandwidth.
> - The smallest batch that reaches the ridge is $B = \lceil I_{\text{ridge}} \cdot b / 2 \rceil$, where $\lceil\,\rceil$ rounds up.

## Build it step by step

1. **`arithmetic_intensity(batch, bytes_per_elt)`.** It takes two numbers and returns a float: twice the batch, divided by the bytes per weight. Check it against Figure 1: batch 1 with 2 bytes gives 1, and batch 4 gives 4. The usual mistake is to include the width of the matrix, which cancels.
2. **`decode_seconds_per_token(n_params, bytes_per_elt, bandwidth)`.** It returns the bytes of weights divided by the bandwidth in bytes per second. The usual mistake is to mix units, for example giving the bandwidth in gigabytes and the parameters in single numbers.
3. **`min_batch_to_saturate(bytes_per_elt, peak_flops, bandwidth)`.** Compute the ridge point first, then solve $2B/b \ge$ ridge for $B$ and round up with `math.ceil`. It returns an `int`. The usual mistake is to round down, which lands just under the ridge, where the chip still waits.

## Where this breaks

The estimates here are idealised. Real decode steps also read the cache of earlier tokens and pay launch overhead, so measured speeds sit below the bandwidth limit. The ridge point depends on the number format as well, because fewer bytes per weight lowers the batch you need. A single step can also be spread over several chips, which adds time spent communicating. The shape of the argument still holds in each case: count the bytes moved, count the flops done, and compare their ratio with the chip's ridge.

## The exercise

You write `arithmetic_intensity`, `decode_seconds_per_token`, and `min_batch_to_saturate`. The checks confirm that batch 1 in bf16 gives 1 flop per byte and that the intensity doubles with the batch. They confirm that 7 billion bf16 parameters at 2 TB/s take 7 ms. They confirm that the 312 TFLOPs, 2 TB/s chip needs a batch of 156 in bf16, rounds a fractional ridge up, and that for random chips the batch you return reaches the ridge while one fewer does not.
