---
title: "Decoding and sampling: cache, temperature, speculation"
slug: 069-decoding-and-sampling
part: "Part 7 — Large language models"
summary: Why decoding waits on memory, how a KV cache, temperature sampling, and beam search work, and how a draft model speeds it up without changing the output.
minutes: 80
gpu: false
objectives:
  - Explain why a model writing one token at a time is limited by memory bandwidth, compute flops per byte and seconds per token, and find the batch that saturates a chip.
  - Compute the size of a KV cache, build one, and show a cached decode matches a full recompute.
  - Turn scores into a token with temperature, top-k, top-p, and a seeded draw.
  - Run beam search with a length penalty and finished hypotheses, and show on a small example that greedy decoding can miss the most probable sequence.
  - Show with a small example that accept-or-replace speculative decoding gives exactly the target model's distribution, and estimate its speedup.
prereqs:
  - 059-scaled-attention-and-heads
  - 066-training-loop-and-cost
  - 065-next-token-loss-and-safe-steps
exercise: 069-decoding-and-sampling
---

# Decoding and sampling: cache, temperature, speculation

Picture a 7-billion-parameter language model on a chip that can do 312 trillion arithmetic operations every second. The answer appears one token at a time, where a token is a word or a piece of a word. Writing one token takes about 14 billion operations, so the chip could write over 20,000 tokens per second. A single user sees at most about 140. Where does the other 99 percent of the time go?

It goes into waiting. The chip is a chef who cooks a dish in a blink, but the ingredients sit in a pantry down a long corridor, and each dish needs a fresh trip. By token 1001 of a story, the model also needs a key and a value from each of the 1000 earlier tokens, which is another trip. How do you then pick a token without repeating yourself or writing nonsense, and can the waiting be put to work?

This chapter counts the trips, builds the cache, turns scores into a token, searches for the best whole sequence with a beam, and ends with speculative decoding, where a cheap model drafts and the large one checks.

## What you already know

- A flop is one multiply or one add. A forward pass costs about 2 flops per weight per token, and a weight in bf16 takes 2 bytes, as in [the cost of a training run](/lessons/066-training-loop-and-cost).
- Attention scores a query against earlier keys, divides by the square root of the head width, and blends the values, and a causal mask hides later tokens, as in [scaled attention and heads](/lessons/059-scaled-attention-and-heads).
- A softmax turns scores into positive numbers that add to 1, and subtracting the largest first avoids overflow.
- A language model gives a probability to every possible next token, as in [next-token loss](/lessons/065-next-token-loss-and-safe-steps), and a token you pick is fed back in.

## Decoding waits on memory

When the model writes a token, each weight is fetched from memory, multiplied by one number, and added into a running sum. That is 2 flops for each 2-byte weight, so the chip does 1 flop for each byte it fetches. Take one 4096-by-4096 matrix on a chip with 312 trillion flops per second and 2 trillion bytes per second of bandwidth. Reading its 16,777,216 weights moves 33.6 million bytes, which takes 16.8 microseconds, and the arithmetic takes 0.108 microseconds. The chip idles 99.4 percent of the time. 

The escape is to give each read more work. Process $B$ tokens together, the batch size, and each weight is fetched once but multiplies $B$ numbers. The tokens can come from different users or from one long prompt, but not from one user's own future tokens. The ratio of flops to bytes is the arithmetic intensity. For a $d \times d$ matrix, $B$ vectors, and $b$ bytes per weight, the matrix does $2Bd^2$ flops and reads $bd^2$ bytes:

$$
I = \frac{2Bd^2}{b\,d^2} = \frac{2B}{b}
$$

Read it aloud: intensity is twice the batch size over the bytes in one weight. In numpy that is `2 * batch / bytes_per_elt`. Speed is the lower of two limits: the memory limit $I \times$ bandwidth and the chip's peak flops. They meet at the ridge point:

$$
I_{\text{ridge}} = \frac{\text{peak flops}}{\text{bandwidth}} = \frac{312 \times 10^{12}}{2 \times 10^{12}} = 156
$$

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
Figure 1. Speed is the lower of two limits. Left of the ridge the memory limit rules, and decode at batch 1 sits at 1 flop per byte, far below the roof. A long prompt reads each weight for many tokens at once, so it reaches the roof.
```

In Figure 1, the sloped line is the memory limit and the flat line the arithmetic limit. Below 156 flops per byte a workload is memory bound. Decode at batch 1 sits at intensity 1, so it gets about 0.6 percent of the chip, while prefill, which reads a whole prompt, is a batch of 1000 and sits on the roof.

If decode waits on memory, the time per token is the time to fetch every weight once. A 7-billion-parameter model in bf16 holds $14 \times 10^9$ bytes, so at $2 \times 10^{12}$ bytes per second:

$$
t = \frac{\text{parameters} \times \text{bytes per weight}}{\text{bandwidth}} = \frac{14 \times 10^9}{2 \times 10^{12}} = 7\ \text{ms}
$$

That is 143 tokens per second. With 156 users in a batch, each step still takes 7 ms and writes 156 tokens, about 22,000 per second in total. The smallest batch that reaches the ridge solves $2B/b \ge I_{\text{ridge}}$, so $B = \lceil I_{\text{ridge}}\, b / 2 \rceil$.

> [!KEY]
> A single user gets 140 tokens per second instead of 20,000 because each token needs every weight fetched once, and each fetched weight does only 2 flops. That is 1 flop per byte against the 156 a chip needs.

## The second trip: the key-value cache

Batching shares the weights but not each user's private history. Without a cache, step $t$ runs the network over all $t$ tokens so far, so writing 1000 tokens costs $1 + 2 + \dots + 1000 = 500{,}500$ token-passes where 1000 would do.

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
Figure 2. Each row is a decoding step and each cell a token whose key and value that step needs. Without a cache the triangle is computed again every step, 21 cells for six tokens. With a cache only the diagonal is computed. The grey cells are still read, so the cache saves arithmetic, not memory traffic.
```

Reading old keys and values is safe because of the causal mask. Token 2 never looks at token 3, so token 2's key and value are the same at step 3 as at step 2.

Take one head of width 2 with two cached tokens. The keys are $[1, 0]$ and $[0, 1]$, the values are $[1, 2]$ and $[3, 4]$, and the query is $q = [2, 0]$. The scores are each key dotted with $q$, divided by $\sqrt{2}$, giving $[1.4142, 0]$. The exponentials 4.1133 and 1 add to 5.1133, so the weights are $[0.8044, 0.1956]$ and the output is $0.8044\,[1, 2] + 0.1956\,[3, 4] = [1.3911, 2.3911]$. In numpy:

```python
scores = K @ q / np.sqrt(d_head)            # (T,)
weights = softmax(scores)                   # subtract the max first
out = weights @ V                           # (d_head,)
```

Counting the bytes: every layer stores one key and one value per key-value head, so

$$
\text{bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot \text{bytes per number}
$$

Read it aloud: two copies, for every layer, key-value head, head position, and token. With 24 layers, 8 key-value heads, head width 64, and 2048 tokens in bf16, one token adds $2 \times 24 \times 8 \times 64 \times 2 = 49{,}152$ bytes, and 2048 tokens make 96 MiB. With 32 layers, 8 key-value heads, head width 128, and 32,768 tokens, one sequence needs 4 GiB, so an 80 GiB card with 13 GiB of weights fits only 16.

Only key-value heads are stored, so let several query heads share one. Grouped-query attention shares one key-value head per group of query heads, and multi-query attention shares one across all. With 8 query heads, 2 key-value heads shrink the cache by 4 and 1 shrinks it by 8, so the card holds 133 sequences instead of 16.

> [!KEY]
> The causal mask guarantees that earlier keys and values never change, so the model stores them instead of recomputing them. That trades memory for arithmetic, and sharing key-value heads is the main way to shrink the bill.

## From scores to a token

The last layer gives one score per token. Take five tokens with scores `[0.5, -2.0, 3.5, 3.4, -1.0]`. Greedy decoding returns `int(np.argmax(logits))`, which is 2. It is dull, because every run is identical, and it loops, because a phrase that was the top continuation once looks likelier after it repeats. Drawing from the model's own probabilities fixes both. Subtract the largest score, exponentiate, and divide: the probabilities are `[0.0253, 0.0021, 0.5077, 0.4594, 0.0056]`, so tokens 2 and 3 both appear. A real vocabulary has thousands of junk tokens, though, and 10,000 of them at 0.00001 each add up to 10 percent, so one draw in ten is nonsense. A temperature dial and a tail filter handle this.

The dial divides every score by a temperature $\tau$ before the softmax. Dividing by $\tau$ above 1 shrinks the gaps between scores and flattens the distribution, and below 1 it stretches them so the leader dominates.

```figure
<svg viewBox="0 0 600 252" role="img" aria-label="Three bar charts of the same five scores at temperature 0.5, 1, and 2: the junk tail shrinks when cold and grows when hot">
<text x="106" y="20" text-anchor="middle" class="fig-bold">τ = 0.5, sharper</text>
<line x1="14" y1="176" x2="198" y2="176" class="stroke-muted"/>
<line x1="14" y1="138" x2="198" y2="138" class="stroke-muted dashed"/>
<rect x="22" y="175.7" width="28" height="0.3" rx="2" class="fill-muted"/>
<text x="36" y="170.7" text-anchor="middle" class="fig-small">0.001</text>
<text x="36" y="192" text-anchor="middle" class="fig-small">0</text>
<rect x="57" y="176" width="28" height="0" rx="2" class="fill-muted"/>
<text x="71" y="171" text-anchor="middle" class="fig-small">0.000</text>
<text x="71" y="192" text-anchor="middle" class="fig-small">1</text>
<rect x="92" y="71.7" width="28" height="104.3" rx="2" class="fill-b"/>
<text x="106" y="66.7" text-anchor="middle" class="fig-small">0.549</text>
<text x="106" y="192" text-anchor="middle" class="fig-small">2</text>
<rect x="127" y="90.6" width="28" height="85.4" rx="2" class="fill-b"/>
<text x="141" y="85.6" text-anchor="middle" class="fig-small">0.450</text>
<text x="141" y="192" text-anchor="middle" class="fig-small">3</text>
<rect x="162" y="176" width="28" height="0" rx="2" class="fill-muted"/>
<text x="176" y="171" text-anchor="middle" class="fig-small">0.000</text>
<text x="176" y="192" text-anchor="middle" class="fig-small">4</text>
<text x="20" y="134" class="fig-small">0.2</text>
<text x="302" y="20" text-anchor="middle" class="fig-bold">τ = 1</text>
<line x1="210" y1="176" x2="394" y2="176" class="stroke-muted"/>
<line x1="210" y1="138" x2="394" y2="138" class="stroke-muted dashed"/>
<rect x="218" y="171.2" width="28" height="4.8" rx="2" class="fill-muted"/>
<text x="232" y="166.2" text-anchor="middle" class="fig-small">0.025</text>
<text x="232" y="192" text-anchor="middle" class="fig-small">0</text>
<rect x="253" y="175.6" width="28" height="0.4" rx="2" class="fill-muted"/>
<text x="267" y="170.6" text-anchor="middle" class="fig-small">0.002</text>
<text x="267" y="192" text-anchor="middle" class="fig-small">1</text>
<rect x="288" y="79.5" width="28" height="96.5" rx="2" class="fill-b"/>
<text x="302" y="74.5" text-anchor="middle" class="fig-small">0.508</text>
<text x="302" y="192" text-anchor="middle" class="fig-small">2</text>
<rect x="323" y="88.7" width="28" height="87.3" rx="2" class="fill-b"/>
<text x="337" y="83.7" text-anchor="middle" class="fig-small">0.459</text>
<text x="337" y="192" text-anchor="middle" class="fig-small">3</text>
<rect x="358" y="174.9" width="28" height="1.1" rx="2" class="fill-muted"/>
<text x="372" y="169.9" text-anchor="middle" class="fig-small">0.006</text>
<text x="372" y="192" text-anchor="middle" class="fig-small">4</text>
<text x="216" y="134" class="fig-small">0.2</text>
<text x="498" y="20" text-anchor="middle" class="fig-bold">τ = 2, flatter</text>
<line x1="406" y1="176" x2="590" y2="176" class="stroke-muted"/>
<line x1="406" y1="138" x2="590" y2="138" class="stroke-muted dashed"/>
<rect x="414" y="157.9" width="28" height="18.1" rx="2" class="fill-muted"/>
<text x="428" y="152.9" text-anchor="middle" class="fig-small">0.095</text>
<text x="428" y="192" text-anchor="middle" class="fig-small">0</text>
<rect x="449" y="170.8" width="28" height="5.2" rx="2" class="fill-muted"/>
<text x="463" y="165.8" text-anchor="middle" class="fig-small">0.027</text>
<text x="463" y="192" text-anchor="middle" class="fig-small">1</text>
<rect x="484" y="94.9" width="28" height="81.1" rx="2" class="fill-b"/>
<text x="498" y="89.9" text-anchor="middle" class="fig-small">0.427</text>
<text x="498" y="192" text-anchor="middle" class="fig-small">2</text>
<rect x="519" y="98.9" width="28" height="77.1" rx="2" class="fill-b"/>
<text x="533" y="93.9" text-anchor="middle" class="fig-small">0.406</text>
<text x="533" y="192" text-anchor="middle" class="fig-small">3</text>
<rect x="554" y="167.5" width="28" height="8.5" rx="2" class="fill-muted"/>
<text x="568" y="162.5" text-anchor="middle" class="fig-small">0.045</text>
<text x="568" y="192" text-anchor="middle" class="fig-small">4</text>
<text x="412" y="134" class="fig-small">0.2</text>
<text x="300" y="218" text-anchor="middle" class="fig-small">token id under each bar; dashed line is 0.2, where all five tokens would tie</text>
<rect x="150" y="232" width="10" height="10" rx="2" class="fill-b"/>
<text x="166" y="241" class="fig-small">the two leaders</text>
<rect x="300" y="232" width="10" height="10" rx="2" class="fill-muted"/>
<text x="316" y="241" class="fig-small">the junk tail</text>
</svg>
Figure 3. The same scores at three temperatures. Cold (left), the two leaders take almost everything and the tail all but vanishes, yet the near tie between tokens 2 and 3 stays a near tie. Hot (right), the tail grows to 0.167 in total. Temperature reshapes the whole distribution, but a gap of 0.1 only decides the draw at very low temperatures.
```

At $\tau = 2$ in Figure 3 the scaled scores are `[0.25, -1, 1.75, 1.7, -0.5]`. Subtract 1.75 and exponentiate to get `[0.2231, 0.0639, 1, 0.9512, 0.1054]`, which add to 2.3437, so the probabilities are `[0.0952, 0.0273, 0.4267, 0.4059, 0.0450]`.

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

Read it aloud: divide each score by $\tau$, exponentiate, and take its share of the total. As $\tau$ shrinks the leader's share goes to 1, which is greedy decoding, and as $\tau$ grows every share goes to $1/5$. Code defines $\tau = 0$ as greedy. In numpy:

```python
s = logits / temperature
probs = np.exp(s - s.max())
probs = probs / probs.sum()
```

> [!WARNING]
> Keep the subtraction of `s.max()`. At $\tau = 0.001$ the largest scaled score is 3500, and `np.exp(3500)` overflows to `inf`, which turns the result into `nan`.

Temperature shrinks the tail but never removes it, so a filter sets the unlikely tokens to zero and divides by what remains. Top-k keeps the $k$ largest probabilities. At $\tau = 2$ with $k = 2$, tokens 2 and 3 hold 0.4267 and 0.4059, which total 0.8326, so they become 0.5125 and 0.4875. A fixed $k$ ignores the shape of the distribution: it keeps barely believed tokens when the model is confident and cuts fair answers when it is unsure.

Top-p, or nucleus sampling, makes the count adapt. Sort the tokens from most to least likely and keep the shortest prefix whose total reaches $p$. At $\tau = 2$ and $p = 0.9$ the sorted probabilities are 0.4267 (token 2), 0.4059 (token 3), 0.0952 (token 0), and so on, with running totals 0.4267, 0.8325, 0.9278. The total first reaches 0.9 at the third token, so keep tokens 2, 3, and 0 and divide by 0.9278 to get `[0.1026, 0, 0.4599, 0.4375, 0]`. The token that crosses $p$ must be kept: stopping just before it leaves less than $p$ of the mass, and with $p = 0.3$ it would keep nothing. In numpy, use `np.argsort(-probs)`, `np.cumsum`, and `np.searchsorted(cumulative, p, side="left")`. Ten tokens of 0.1 add to 0.9999999999999999, so `p = 1.0` returns index 10: clamp it to `len(probs) - 1`.

The whole step runs in order: temperature and softmax, top-k, top-p on the renormalised result, then one draw with `rng.choice` on the generator you were given, so that a seed reproduces the text.

> [!KEY]
> Greedy decoding is dull, and drawing from raw probabilities lets the junk tail through. Temperature sets how much weight the tail gets, and a filter removes it. Top-p adapts to how sure the model is, which a fixed $k$ cannot.

## Beam search: keeping several candidates

Greedy decoding takes the best token at each step, but that can lead to a worse sequence. Take tokens `a`, `b`, and an end-of-sequence token `eos`. After the start token, a toy model gives `a` 0.5, `b` 0.4, and `eos` 0.1. After `a` it gives `a` 0.3, `b` 0.3, and `eos` 0.4. After `b` it gives `a` 0.05, `b` 0.05, and `eos` 0.9. Greedy takes `a` and then `eos`, a sequence with probability $0.5 \times 0.4 = 0.20$. The path through `b` has probability $0.4 \times 0.9 = 0.36$, and greedy never looks at it.

Beam search keeps $B$ prefixes alive, where $B$ is the beam width. At each step, extend every kept prefix with every token, score all the extensions, and keep the $B$ best of them together. A prefix scores the sum of the log-probabilities of its tokens, so a long product becomes a sum that cannot underflow:

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log p(y_i \mid y_{<i})
$$

Read it aloud: add the log-probability of each token given the ones before it, and a sum closer to 0 is better. With $B = 2$ on the toy model, step 1 scores `a` at $-0.693$, `b` at $-0.916$, and `eos` at $-2.303$, so `a` and `b` stay. Step 2 scores `b eos` at $-1.022$, `a eos` at $-1.609$, and `a a` and `a b` at $-1.897$. The top two are `b eos` and `a eos`, so the best sequence is `b eos`, with probability 0.36. A beam of 1 is greedy.

The cut is global: keep the $B$ best of all $B \times V$ extensions, not $B$ per prefix, so a prefix with two strong children can take two slots. A kept extension that ends in `eos` is finished: it leaves the beam and joins a finished list, and the search continues with the rest until `max_len`. The answer is the best finished hypothesis.

Every log-probability is at most 0, so each added token lowers the score, and without a correction beam search favors short outputs. Wu et al. (2016, section 7) divide the score by a length penalty $\mathrm{lp}(Y) = \bigl((5 + |Y|)/6\bigr)^{\alpha}$, where $\alpha = 0$ is plain beam search and 0.6 to 0.7 usually worked best. In a second model, `a eos` has probability 0.30 and length 2, and `b c eos` has 0.27 and length 3. Raw scores are $\ln 0.30 = -1.204$ and $\ln 0.27 = -1.309$, so the short one wins. With $\alpha = 1$ the penalties are $7/6$ and $8/6$, giving $-1.032$ and $-0.982$, and the long one wins.

Beam search is not the default for open-ended writing. Holtzman et al. (2020, section 4.3) show that human text does not stay in a high-probability zone, so the most probable text is generic and repetitive. Sampling with top-p avoids that, and beam search suits translation, where the input scopes the output.

> [!KEY]
> Greedy can miss the most probable sequence, and beam search keeps the $B$ best prefixes by summed log-probability to find it. Finished hypotheses leave the beam, and a length penalty stops short outputs from winning on length alone.

## Speculative decoding: guess fast, verify once

Decode leaves the chip idle, so suppose a model a tenth the cost guessed the next few tokens. Call the large model the target, with next-token distribution $p$, and the small one the draft, with $q$. The draft writes $k = 3$ tokens. Then the target reads the last written token plus all three in a single pass, so four tokens share one read of the weights. Because of the causal mask, that pass gives the target's own distribution after each position. Compare from left to right: keep each draft token the target agrees with, and at the first rejection write the target's own pick and discard the rest. If all are kept, the last position gives a bonus token, so a round yields between 1 and $k + 1$ tokens.

"Agrees with" needs care. Accepting every draft token makes the output follow $q$, and accepting only the target's top choice never keeps a token the target would draw 20 percent of the time. We want the output distributed exactly as if sampled from $p$.

Take $p = [0.7, 0.2, 0.1]$ and a deliberately bad draft $q = [0.1, 0.1, 0.8]$. The draft proposes $x$ with probability $q(x)$, and we accept it with probability $\min(1, p(x)/q(x))$:

- $x = 0$ and $x = 1$ each come up with probability 0.1, the ratios are 7 and 2, so both are always accepted and add 0.1 to tokens 0 and 1.
- $x = 2$ comes up with probability 0.8, the ratio is $0.1/0.8 = 0.125$, so it is accepted 12.5 percent of the time. That adds $0.8 \times 0.125 = 0.1$ to token 2, and the other 0.7 is rejected.

A rejection takes a replacement from what the target wants and the draft did not supply, $\max(p - q, 0) = [0.6, 0.1, 0]$. It sums to 0.7, so dividing gives the replacement distribution $[0.857, 0.143, 0]$. Multiply by the rejection chance 0.7 to get $[0.6, 0.1, 0]$, and add the two sources:

$$
[0.1,\ 0.1,\ 0.1] + [0.6,\ 0.1,\ 0] = [0.7,\ 0.2,\ 0.1] = p
$$

```figure
<svg viewBox="0 0 600 312" role="img" aria-label="For each of three tokens, a pair of bars for the target p and the draft q with their overlap in green, the target's leftover and the draft's excess">
<line x1="30" y1="210" x2="450" y2="210" class="stroke-muted"/>
<rect x="54" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="54" y="70" width="42" height="120" rx="0" class="fill-soft-b stroke-b"/>
<rect x="104" y="190" width="42" height="20" rx="0" class="fill-c"/>
<text x="75" y="64" text-anchor="middle" class="fig-bold">0.7</text>
<text x="125" y="184" text-anchor="middle" class="fig-bold">0.1</text>
<text x="75" y="226" text-anchor="middle" class="fig-small">p</text>
<text x="125" y="226" text-anchor="middle" class="fig-small">q</text>
<text x="100" y="244" text-anchor="middle" class="fig-bold">token 0</text>
<text x="100" y="262" text-anchor="middle" class="fig-small">min 0.1, left 0.6</text>
<rect x="194" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="194" y="170" width="42" height="20" rx="0" class="fill-soft-b stroke-b"/>
<rect x="244" y="190" width="42" height="20" rx="0" class="fill-c"/>
<text x="215" y="164" text-anchor="middle" class="fig-bold">0.2</text>
<text x="265" y="184" text-anchor="middle" class="fig-bold">0.1</text>
<text x="215" y="226" text-anchor="middle" class="fig-small">p</text>
<text x="265" y="226" text-anchor="middle" class="fig-small">q</text>
<text x="240" y="244" text-anchor="middle" class="fig-bold">token 1</text>
<text x="240" y="262" text-anchor="middle" class="fig-small">min 0.1, left 0.1</text>
<rect x="334" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="384" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="384" y="50" width="42" height="140" rx="0" class="fill-sunk stroke-muted dashed"/>
<text x="355" y="184" text-anchor="middle" class="fig-bold">0.1</text>
<text x="405" y="44" text-anchor="middle" class="fig-bold">0.8</text>
<text x="355" y="226" text-anchor="middle" class="fig-small">p</text>
<text x="405" y="226" text-anchor="middle" class="fig-small">q</text>
<text x="380" y="244" text-anchor="middle" class="fig-bold">token 2</text>
<text x="380" y="262" text-anchor="middle" class="fig-small">min 0.1, left 0.0</text>
<rect x="462" y="40" width="14" height="14" rx="0" class="fill-c"/>
<text x="482" y="52" class="fig-small">overlap: min(p, q)</text>
<rect x="462" y="66" width="14" height="14" rx="0" class="fill-soft-b stroke-b"/>
<text x="482" y="78" class="fig-small">target leftover</text>
<rect x="462" y="92" width="14" height="14" rx="0" class="fill-sunk stroke-muted dashed"/>
<text x="482" y="104" class="fig-small">draft excess</text>
<text x="300" y="284" text-anchor="middle" class="fig-small">Overlap: 0.1 + 0.1 + 0.1 = 0.3, the acceptance rate.</text>
<text x="300" y="302" text-anchor="middle" class="fig-small">Target leftover: 0.6 + 0.1 + 0 = 0.7, the rejection chance.</text>
</svg>
Figure 4. Green is the part of each token that both models agree on, and it adds up to 0.3. The orange leftover is what the target wants and the draft did not supply, and it adds up to 0.7. Green plus orange gives p for every token.
```

The output is exactly $p$ even though the draft was badly wrong. In Figure 4, the green overlap $\min(p, q)$ is what the draft supplies and the orange leftover $\max(p - q, 0)$ is what the replacement supplies. This holds for any $p$ and $q$. Token $x$ comes out by acceptance with probability $q(x)\min(1, p(x)/q(x)) = \min(p(x), q(x))$. The rejection chance $1 - \sum_x \min(p, q)$ equals $\sum_x \max(p - q, 0)$, the total that normalises the leftover, so rejection times the normalised leftover is $\max(p(x) - q(x), 0)$. Adding the acceptance part gives $p(x)$. In code, use the generator for both draws:

```python
accept = rng.uniform() < min(1.0, p[x] / q[x])
leftover = np.maximum(p - q, 0.0)
replacement = leftover / leftover.sum()
```

The draft token came from $q$, so $q(x) > 0$ and the ratio is defined. If $p$ equals $q$ the leftover is all zeros, but no token is ever rejected, so return $p$ as a fallback to avoid dividing by zero. The overlap $\sum \min(p, q)$ is the acceptance rate $\alpha$, 0.3 here.

Suppose each draft token is accepted with the same chance $\alpha$, independently. A round yields one token, a second with probability $\alpha$, a third with $\alpha^2$, up to $\alpha^k$:

$$
\text{tokens per pass} = 1 + \alpha + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

Read it aloud: one guaranteed token, plus the chance of each further one surviving. At $\alpha = 1$ the sum is $k + 1$. With $k = 3$ and $\alpha = 0.8$ you get $1 + 0.8 + 0.64 + 0.512 = 2.952$, and with $\alpha = 0.3$ you get 1.417. A draft step at 0.1 of a target step makes a round take 1.3 target steps, so the speedup is $2.952/1.3 = 2.27$ at $\alpha = 0.8$ and $1.417/1.3 = 1.09$ at $\alpha = 0.3$.

> [!KEY]
> Decoding waits on memory, so a big model can check several drafted tokens for about the price of one. Accepting with probability $\min(1, p/q)$ and replacing from $\max(p - q, 0)$ makes the output follow $p$ exactly, so a bad draft costs only speed.

> [!RECAP]
> - Arithmetic intensity is $2B/b$, the ridge point is peak flops over bandwidth, and a batch-1 decode step takes weight bytes over bandwidth.
> - The cache holds two copies for every layer, key-value head, head width, and token.
> - Temperature divides the logits before a stable softmax, top-k then top-p filter and renormalise, and the crossing token is kept.
> - Beam search keeps the $B$ best prefixes by summed log-probability, drops finished ones from the beam, and divides by a length penalty.
> - A draft token is kept with probability $\min(1, p/q)$, a rejection draws from $\max(p - q, 0)$ normalised, and a round yields $(1 - \alpha^{k+1})/(1 - \alpha)$ tokens on average.

## Build it step by step

You write thirteen functions, and the checks call them in this order.

1. `arithmetic_intensity(batch, bytes_per_elt)` returns twice the batch over the bytes per weight. Batch 1 with 2 bytes gives 1.
2. `decode_seconds_per_token(n_params, bytes_per_elt, bandwidth)` returns the bytes of weights over the bandwidth in bytes per second.
3. `min_batch_to_saturate(bytes_per_elt, peak_flops, bandwidth)` computes the ridge point, solves $2B/b \ge$ ridge, and rounds up with `math.ceil`, returning an `int`.
4. `kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt=2)` returns the byte count as an int.
5. `KVCache` keeps one list of keys and one of values per layer. `append(layer, k, v)` adds one `(n_kv, d_head)` array. `get(layer)` stacks each list on axis 0 with `np.stack` and returns `(K, V)`, or zero rows of shape `(0, n_kv, d_head)` for an empty layer.
6. `cached_attention_step(q, K, V)` takes `q` of shape `(d_head,)` and `K`, `V` of shape `(T, d_head)`. Score, divide by the square root of the width, softmax, and blend the value rows.
7. `apply_temperature(logits, temperature)` divides, subtracts the maximum, exponentiates, and divides by the sum.
8. `filter_probs(probs, top_k=None, top_p=None)` returns a copy and never changes its input. Run top-k first, then top-p with `np.cumsum` and a clamped `np.searchsorted`.
9. `sample_token(logits, rng, temperature=1.0, top_k=None, top_p=None)` returns `int(np.argmax(logits))` at temperature 0, and otherwise chains `apply_temperature`, `filter_probs`, and `rng.choice(len(probs), p=probs)`, returning a plain `int`. Use `rng.choice`, not `np.random.choice`, which ignores your seed.
10. `expected_accept_rate(p, q)` returns the sum of the elementwise minimum as a float, 0.3 on the worked example.
11. `residual_distribution(p, q)` returns `np.maximum(p - q, 0)` over its sum, or a copy of `p` when the sum is 0.
12. `speculative_accept(p, q, draft_token, rng)` draws `rng.uniform()` and compares it with `min(1, p[draft_token] / q[draft_token])`. If smaller it returns `int(draft_token)`, otherwise a draw from the residual with `rng.choice`.
13. `beam_search(log_probs, bos, eos, beam_width, max_len, alpha=0.0)` takes a transition matrix whose entry `[i, j]` is the log-probability of token `j` after token `i`. Extend every live hypothesis with every token, sort all extensions by score with a stable sort, and keep the first `beam_width`. Move kept extensions that end in `eos` to a finished list, and rank finished hypotheses by the score over `((5 + L) / 6) ** alpha`. The usual mistake is to keep finished hypotheses in the beam.

## Where this breaks

The cache grows with every token, so a long conversation eventually does not fit beside the weights, and servers drop old entries, store the cache in fewer bits, or cap the context length. No sampling dial repairs a wrong model. Speculation needs a memory-bound chip, because at a large batch checking extra positions is no longer free, a draft that agrees with the target often, and the target's vocabulary. It also cannot use a large $k$, because the $j$-th draft token survives with chance about $\alpha^j$.

## The exercise

You write `arithmetic_intensity`, `decode_seconds_per_token`, `min_batch_to_saturate`, `kv_cache_bytes`, `KVCache`, `cached_attention_step`, `apply_temperature`, `filter_probs`, `sample_token`, `expected_accept_rate`, `residual_distribution`, `speculative_accept`, and `beam_search`. The checks confirm 7 ms per token, a batch of 156, and 96 MiB, that decoding six tokens through the cache matches a full recompute, the hand-worked temperature, top-k, and top-p values, that seeded draws match the filtered distribution, and that 20,000 seeded accept-or-replace rounds give the target distribution. For `beam_search`, a wide beam must equal exhaustive enumeration, a beam of 1 must equal greedy decoding, a beam of 2 must find `b eos` where greedy does not, length normalization must flip the winner on a planted case, and finished hypotheses must leave the beam.

## Further reading

The beam search section follows these two papers:

- For length normalization, the length penalty, and the report that $\alpha$ of 0.6 to 0.7 usually worked best, see section 7, "Decoder", of [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) (Wu et al., 2016). The paper also adds a coverage penalty, which this chapter leaves out.
- For why high-probability text is degenerate in open-ended generation, see section 4.3, "Natural language does not maximize probability", of [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., ICLR 2020). Section 2 separates directed generation from open-ended generation.
