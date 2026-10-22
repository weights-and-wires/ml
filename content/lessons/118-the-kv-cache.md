---
title: "The KV cache: remember, don't recompute"
slug: 118-the-kv-cache
part: "Part 7 — Large language models"
summary: Keys and values of earlier tokens never change, so a cache stores them once and appends a row per step, trading memory for arithmetic.
minutes: 40
gpu: false
objectives:
  - Explain why the keys and values of earlier tokens never change, and what a cache saves and what it does not.
  - Compute the size of a KV cache from layers, key-value heads, head width, tokens, and bytes per number.
  - Explain how sharing key-value heads shrinks the cache and why each query head still asks its own question.
  - Build a cache that appends one row per token and check that decoding through it matches a full recompute.
prereqs:
  - 099-heads-and-causal-mask
  - 117-why-decoding-waits-on-memory
exercise: 118-the-kv-cache
---

# The KV cache: remember, don't recompute

A model has written the first 1000 tokens of a story, and it is about to write token 1001. To do that, the new token asks a question of every earlier token, and each earlier token answers with a key and a value. The model already computed those 1000 keys and 1000 values while it was writing them. So why would it compute them again for token 1001, and again for token 1002?

Without a cache it does exactly that. Each step runs the network over the whole text so far, so step $t$ handles $t$ tokens. Writing 1000 tokens costs $1 + 2 + \dots + 1000 = 500{,}500$ token-passes where 1000 would do. The obvious fix is to keep the answers, and the obvious worry is what keeping them costs. This chapter shows why keeping them is safe, counts the bytes, and then shows the trick that makes the bill smaller.

## What you already know

- Attention scores a query against every earlier key, then blends the values, as in "Letting tokens look at each other: scores and softmax weights".
- The scores are divided by the square root of the head width, as in "Why scores are divided by the square root of the width".
- A causal mask means token $t$ reads only tokens up to $t$, and each head has its own queries, keys, and values, as in "Many heads and the causal mask".
- Writing a token is limited by bytes moved, not flops, as in "Why generating text waits on memory".
- A number in bf16 takes 2 bytes.

## The picture: the same rows, again and again

Look at Figure 1. Each row is a decoding step, and each cell is a token whose key and value that step needs. Step 3 needs tokens 1, 2, and 3, so row 3 has three cells. The left triangle computes every cell in every row. The right triangle computes only the last cell of each row and reads the rest from storage.

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
Figure 1. Each row is a decoding step and each cell a token whose key and value that step needs. Without a cache the triangle is computed again every step, 21 cells for six tokens. With a cache only the diagonal is computed. The grey cells are still read, so the cache saves arithmetic, not memory traffic.
```

Why is reading the old cells safe? Because of the causal mask. Token 2 never looks at token 3, so nothing that token 3 brings can change what token 2's hidden state is, at any layer. Token 2's key and value are functions of that hidden state, so they are the same at step 3 as they were at step 2. The old cells are constants. Recomputing a constant gives the same answer at a higher price.

The price difference is the triangle against the diagonal. Six tokens cost $1 + 2 + 3 + 4 + 5 + 6 = 21$ computations without a cache and 6 with one. For 1000 tokens it is 500,500 against 1000. Be precise about what is saved. The per-token work, the key and value projections and the feed-forward layers, goes from growing with $T^2$ to growing with $T$. The attention step itself still compares the new query with all $t$ cached keys, so it grows with $t$ at every step. The cache removes the repeated computation, not the reading.

## One cached step, by hand

To make the reading concrete, take one head of width 2 and two cached tokens. The keys are $[1, 0]$ and $[0, 1]$, the values are $[1, 2]$ and $[3, 4]$, and the new query is $q = [2, 0]$. The scores are each key dotted with $q$, divided by $\sqrt{2}$:

$$
s = \frac{1}{\sqrt{2}}\,[2,\ 0] = [1.4142,\ 0]
$$

A softmax turns them into weights. The exponentials are $e^{1.4142} = 4.1133$ and $e^0 = 1$, which add to 5.1133, so the weights are $[0.8044, 0.1956]$. The output blends the two value rows:

$$
0.8044 \cdot [1, 2] + 0.1956 \cdot [3, 4] = [1.3911,\ 2.3911]
$$

Read aloud: score each stored key against the new query, turn the scores into shares, and add up the stored values in those shares. In numpy, with `K` and `V` of shape `(T, d_head)` and `q` of shape `(d_head,)`:

```python
scores = K @ q / np.sqrt(d_head)            # (T,)
weights = softmax(scores)                   # subtract the max first
out = weights @ V                           # (d_head,)
```

Nothing here recomputes a key or a value. They sit in `K` and `V`, put there by earlier steps.

## Counting the bytes

Storage is the price of the cache. Count what one token adds. Every layer stores one key and one value for every key-value head, and each of those is a vector of the head width. Take 24 layers, 8 key-value heads, and head width 64. One token adds $2 \times 24 \times 8 \times 64 = 24{,}576$ numbers, where the 2 is one key plus one value. In bf16 that is 49,152 bytes, or 48 KiB. A sequence of 2048 tokens adds that 2048 times: $48\ \text{KiB} \times 2048 = 96\ \text{MiB}$.

$$
\text{bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot \text{bytes per number}
$$

Read it aloud: two copies (keys and values), for every layer, every key-value head, every position in the head, every token, each as many bytes as one number takes. As code, it is a single product:

```python
size = 2 * n_layer * n_kv * d_head * seq_len * bytes_per_elt
```

The factor to watch is $n_{\text{kv}}$, the number of key-value heads. The model may have more query heads than that, but only key-value heads are stored.

## Why the cache gets expensive

96 MiB looks harmless. Scale the model up to 32 layers, 8 key-value heads, and head width 128, and let the context grow to 32,768 tokens:

| Model shape | Tokens | Cache for one sequence |
|---|---|---|
| 24 layers, 8 kv heads, width 64 | 2,048 | 96 MiB |
| 32 layers, 8 kv heads, width 128 | 32,768 | 4 GiB |

Now put it next to the weights. A 7-billion-parameter model in bf16 holds about 13 GiB of weights, and an 80 GiB card has $80 - 13 = 67$ GiB left, which fits only 16 sequences of 4 GiB. A hundred users would need 400 GiB of cache. The cache also costs time. In "Why generating text waits on memory" the weights were one shared trip per step. Every sequence's cache is a trip of its own that nobody shares, so four sequences already read 16 GiB of cache per step, more than the 13 GiB of weights.

## Sharing key-value heads

If the cache is the problem, shrink the part we multiply by. In multi-head attention every head has its own queries, keys, and values, so each head looks for something different and keeps its own notebook of keys and values. But only the notebooks are stored. Queries are computed fresh for each token and never cached.

That suggests a trade. Let several query heads read from one shared notebook. Each query head still asks its own question, because it still has its own query vector, so the heads still look for different things. They look into the same stored keys and values. Grouped-query attention puts the query heads into groups, and each group shares one key-value head. Multi-query attention is the extreme where all query heads share one.

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
Figure 2. Eight query heads in every panel. Count the blue boxes: 8, 2, and 1. Every box is stored for every token, so fewer boxes mean a cache 1, 4, or 8 times smaller, while each query head still asks its own question.
```

Figure 2 counts the stored boxes in each case.

With 8 query heads, going from 8 key-value heads to 2 shrinks the cache by 4, and going to 1 shrinks it by 8. Models trained with shared heads lose little quality, which is why many large models use grouped-query attention. For the 4 GiB sequence above, one key-value head gives 0.5 GiB, and the card from before holds 133 sequences instead of 16, as Figure 3 shows:

```figure
<svg viewBox="0 0 600 256" role="img" aria-label="Two bars for an 80 GiB card: weights, then 16 stripes of 4 GiB caches, or 133 thin stripes of 0.5 GiB caches">
<text x="300" y="20" text-anchor="middle" class="fig-bold">An 80 GiB card holding a 7B model and 32,768-token sequences</text>
<text x="40" y="56">8 key-value heads: 4 GiB per sequence</text>
<rect x="40" y="66" width="520" height="34" rx="2" class="fill-paper stroke-muted"/>
<rect x="40" y="66" width="84.8" height="34" rx="2" class="fill-sunk stroke-muted"/>
<text x="82.4" y="88" text-anchor="middle" class="fig-small">weights</text>
<rect x="124.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="150.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="176.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="202.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="228.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="254.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="280.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="306.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="332.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="358.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="384.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="410.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="436.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="462.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<rect x="488.8" y="66" width="26" height="34" rx="0" class="fill-a"/>
<rect x="514.8" y="66" width="26" height="34" rx="0" class="fill-soft-a"/>
<line x1="540.8" y1="62" x2="540.8" y2="104" class="stroke-bad dashed"/>
<text x="300" y="124" text-anchor="middle" class="fig-bold">16 sequences fit</text>
<text x="40" y="154">1 key-value head: 0.5 GiB per sequence</text>
<rect x="40" y="164" width="520" height="34" rx="2" class="fill-paper stroke-muted"/>
<rect x="40" y="164" width="84.8" height="34" rx="2" class="fill-sunk stroke-muted"/>
<text x="82.4" y="186" text-anchor="middle" class="fig-small">weights</text>
<rect x="124.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="128" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="131.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="134.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="137.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="141" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="144.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="147.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="150.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="154" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="157.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="160.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="163.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="167" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="170.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="173.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="176.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="180" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="183.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="186.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="189.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="193" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="196.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="199.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="202.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="206" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="209.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="212.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="215.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="219" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="222.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="225.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="228.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="232" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="235.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="238.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="241.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="245" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="248.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="251.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="254.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="258" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="261.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="264.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="267.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="271" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="274.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="277.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="280.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="284" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="287.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="290.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="293.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="297" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="300.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="303.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="306.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="310" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="313.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="316.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="319.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="323" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="326.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="329.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="332.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="336" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="339.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="342.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="345.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="349" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="352.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="355.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="358.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="362" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="365.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="368.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="371.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="375" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="378.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="381.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="384.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="388" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="391.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="394.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="397.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="401" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="404.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="407.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="410.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="414" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="417.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="420.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="423.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="427" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="430.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="433.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="436.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="440" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="443.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="446.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="449.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="453" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="456.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="459.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="462.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="466" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="469.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="472.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="475.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="479" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="482.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="485.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="488.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="492" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="495.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="498.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="501.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="505" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="508.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="511.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="514.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="518" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="521.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="524.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="527.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="531" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="534.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="537.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="540.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="544" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="547.3" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<rect x="550.5" y="164" width="3.2" height="34" rx="0" class="fill-soft-a"/>
<rect x="553.8" y="164" width="3.2" height="34" rx="0" class="fill-a"/>
<line x1="557" y1="160" x2="557" y2="202" class="stroke-bad dashed"/>
<text x="300" y="222" text-anchor="middle" class="fig-bold">133 sequences fit</text>
<text x="40" y="218" text-anchor="middle" class="fig-small">0</text>
<text x="560" y="218" text-anchor="end" class="fig-small">80 GiB</text>
<text x="124.8" y="218" text-anchor="middle" class="fig-small">13 GiB</text>
<line x1="124.8" y1="202" x2="124.8" y2="208" class="stroke-muted"/>
<text x="300" y="246" text-anchor="middle" class="fig-small">Each stripe is one sequence's cache. The dashed line is where the next one no longer fits.</text>
</svg>
Figure 3. The same card and the same model, with only the number of key-value heads changed. Eight heads leave room for 16 sequences. One head leaves room for 133, because each sequence's cache is eight times smaller.
```

> [!KEY]
> The model computes each token's keys and values once, and the causal mask guarantees they never change, so it stores them instead of recomputing them. That trades memory for arithmetic. The memory bill is $2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T$ numbers, and sharing key-value heads is the main way to shrink it.

> [!RECAP]
> Before you build:
> - A key and a value of an earlier token never change, so the cache appends one row per token per layer.
> - The size is two copies times layers, key-value heads, head width, tokens, and bytes per number.
> - `get` returns arrays of shape `(T, n_kv, d_head)`, stacked on axis 0, and a layer with nothing stored returns zero rows.
> - A cached step scores the new query against all stored keys, softmaxes, and blends the stored values.

## Build it step by step

1. **`kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt=2)`.** It takes five integers and returns the byte count as an int. Check it against the 48 KiB per token above. The usual mistake is to forget that keys and values are two copies.
2. **`KVCache`.** `__init__` creates one list of keys and one list of values for each layer. `append(layer, k, v)` adds one array of shape `(n_kv, d_head)` to that layer's lists. `get(layer)` stacks each list on axis 0 with `np.stack` and returns `(K, V)`. Return zero rows of shape `(0, n_kv, d_head)` for an empty layer, because `np.stack` raises an error on an empty list. The usual mistake is `np.concatenate`, which merges the token axis into the head axis.
3. **`cached_attention_step(q, K, V)`.** Take `q` of shape `(d_head,)` and cached `K`, `V` of shape `(T, d_head)`. Score, divide by the square root of the width, softmax with the maximum subtracted, and blend the value rows. It returns shape `(d_head,)`. The usual mistake is to forget the division by the square root.

## Where this breaks

The cache grows with every token, so a long conversation eventually does not fit beside the weights. Servers respond in three ways, each with a price. They drop old entries, which forgets the start of the text. They store the cache in fewer bits, which adds rounding error. They cap the context length, which turns long requests away. Sharing key-value heads is gentler, since it keeps all the tokens, but it has to be built into the model before training.

## The exercise

You write `kv_cache_bytes`, `KVCache`, and `cached_attention_step`. The checks match the byte count to the hand count of 96 MiB, confirm that going from 8 key-value heads to 1 shrinks 4 GiB to 0.5 GiB, and confirm that `get` returns the appended rows in order with shape `(T, n_kv, d_head)`, including zero rows for an empty layer. They check one step against a manual softmax, and they check that decoding six tokens one at a time through `KVCache` and `cached_attention_step` gives the same output as a full causal recompute.
