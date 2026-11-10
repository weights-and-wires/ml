---
title: "Online softmax and tiled attention"
slug: 051-online-softmax-and-tiled-attention
part: "Part 6 — Transformers"
summary: Compute softmax in one streaming pass, merge partial results, and tile attention so the T by T table is never stored, with a backward pass that recomputes it.
minutes: 60
gpu: false
objectives:
  - Compute the max and the sum of a softmax in one pass, and say why the running sum must be rescaled when the max changes.
  - Merge two partial softmax states and explain why the merge is associative, so blocks can arrive in any grouping.
  - Write block-wise attention that holds only a block by block table of scores, and skip the blocks wholly above the causal diagonal.
  - Recompute the attention weights in the backward pass from one saved number per row, and count the memory and flops that trade against.
prereqs:
  - 050-attention-backward-and-cost
exercise: 051-online-softmax-and-tiled-attention
---

# Online softmax and tiled attention

The last chapter ended with a bill. One attention head at 4,096 tokens keeps a table of 4,096 times 4,096 scores, and a second table of weights after the softmax. With 16 heads in float32 that is 2 GiB for one layer, and the backward pass reads both tables back. Doubling the sequence length multiplies the bill by four. Before the table gets too big to hold, there is a question to answer: does the output of attention need the table at all?

It looks as if it does. A softmax divides every entry of a row by the sum of the whole row, so you cannot write down the first weight until you have seen the last score.

This chapter shows that the whole row is not needed. A softmax can be computed as a stream, with two numbers of memory per row, and two partial streams can be glued together exactly. So you can cut the table into small tiles, process one at a time, and get the dense output to rounding error. The backward pass rebuilds each tile instead of reading it from memory.

## What you already know

- The softmax of a row is `exp(x - max) / sum(exp(x - max))`, and subtracting the max stops overflow without changing the answer.
- Attention is `softmax(Q @ K.T / sqrt(d)) @ V`, and a causal mask sets the scores above the diagonal to minus infinity.
- The softmax backward rule is `weights * (dweights - rowsum)`, where `rowsum` is `sum(weights * dweights)` along the row.
- One attention layer costs `8*T*d**2 + 4*T**2*d` flops, and its `T` by `T` tables are activations saved for the backward pass.

## The softmax in one pass

Start with one row of four scores, `[2, 1, 5, 3]`. The stable softmax makes three trips over it. The first trip finds the max, 5. The second adds up `exp(x - 5)` to get the sum. The third divides each term by that sum. Three trips means the row stays in memory the whole time.

Now try to do the first two trips together. Read the scores one at a time and keep two numbers: `m`, the largest score so far, and `l`, the sum of `exp(x - m)` over the scores so far. When a new score `x` arrives, there are two cases.

- If `x` is not larger than `m`, the max stays, and you add `exp(x - m)` to `l`.
- If `x` is larger, the max moves to `x`, and the old sum is stale. It was measured against the old max.

The second case needs a repair, and the repair is exact. Every term in the old sum has the form `exp(s - m_old)`. The same term measured against the new max is `exp(s - m_new)`, which equals `exp(s - m_old) * exp(m_old - m_new)`. Every term gets the same factor, so the whole sum gets it too. In symbols, with `m_new = max(m, x)`:

$$
l_{\text{new}} = l \cdot e^{\,m - m_{\text{new}}} + e^{\,x - m_{\text{new}}}
$$

Read it aloud: shrink the old sum to the new max, then add the new score measured against the new max. When the max does not move, `m - m_new` is 0, the factor is 1, and the rule reduces to the plain case. When the max moves, the factor `exp(m - m_new)` is less than 1, because `m` is smaller than `m_new`, so the stale sum shrinks and never overflows.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Streaming softmax over the scores 2, 1, 5, 3: the running max, the factor applied to the old sum, and the running sum after each score">
<text x="20" y="48" class="fig-small fig-bold">score x</text>
<text x="20" y="96" class="fig-small fig-bold">running max m</text>
<text x="20" y="144" class="fig-small fig-bold">old sum times</text>
<text x="20" y="192" class="fig-small fig-bold">running sum l</text>
<rect x="105" y="26" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="150" y="49" text-anchor="middle" class="fig-mono">2</text>
<rect x="105" y="74" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="150" y="97" text-anchor="middle" class="fig-mono">2</text>
<rect x="105" y="122" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="150" y="145" text-anchor="middle" class="fig-mono">(empty)</text>
<rect x="105" y="170" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="150" y="193" text-anchor="middle" class="fig-mono">1.0000</text>
<line x1="195" y1="44" x2="225" y2="44" class="stroke-muted"/>
<line x1="195" y1="92" x2="225" y2="92" class="stroke-muted"/>
<line x1="195" y1="140" x2="225" y2="140" class="stroke-muted"/>
<line x1="195" y1="188" x2="225" y2="188" class="stroke-muted"/>
<rect x="225" y="26" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="270" y="49" text-anchor="middle" class="fig-mono">1</text>
<rect x="225" y="74" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="270" y="97" text-anchor="middle" class="fig-mono">2</text>
<rect x="225" y="122" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="270" y="145" text-anchor="middle" class="fig-mono">1.0000</text>
<rect x="225" y="170" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="270" y="193" text-anchor="middle" class="fig-mono">1.3679</text>
<line x1="315" y1="44" x2="345" y2="44" class="stroke-muted"/>
<line x1="315" y1="92" x2="345" y2="92" class="stroke-muted"/>
<line x1="315" y1="140" x2="345" y2="140" class="stroke-muted"/>
<line x1="315" y1="188" x2="345" y2="188" class="stroke-muted"/>
<rect x="345" y="26" width="90" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="390" y="49" text-anchor="middle" class="fig-mono">5</text>
<rect x="345" y="74" width="90" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="390" y="97" text-anchor="middle" class="fig-mono">5</text>
<rect x="345" y="122" width="90" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="390" y="145" text-anchor="middle" class="fig-mono">0.0498</text>
<rect x="345" y="170" width="90" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="390" y="193" text-anchor="middle" class="fig-mono">1.0681</text>
<line x1="435" y1="44" x2="465" y2="44" class="stroke-muted"/>
<line x1="435" y1="92" x2="465" y2="92" class="stroke-muted"/>
<line x1="435" y1="140" x2="465" y2="140" class="stroke-muted"/>
<line x1="435" y1="188" x2="465" y2="188" class="stroke-muted"/>
<rect x="465" y="26" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="510" y="49" text-anchor="middle" class="fig-mono">3</text>
<rect x="465" y="74" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="510" y="97" text-anchor="middle" class="fig-mono">5</text>
<rect x="465" y="122" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="510" y="145" text-anchor="middle" class="fig-mono">1.0000</text>
<rect x="465" y="170" width="90" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="510" y="193" text-anchor="middle" class="fig-mono">1.2034</text>
<text x="390" y="226" text-anchor="middle" class="fig-small fig-bold">the max jumps from 2 to 5, so the old sum shrinks by e^(2-5) = 0.0498</text>
<text x="390" y="244" text-anchor="middle" class="fig-small"></text>
</svg>
Figure 1. The streaming softmax over [2, 1, 5, 3]. The max holds at 2 for the second score. At the third score it jumps to 5, so the old sum 1.3679 shrinks by 0.0498 before the new term 1 is added. The result, 1.0681, then reaches 1.2034 after the last score.
```

Figure 1 is the worked example, step by step.

1. `x = 2`. Nothing is stored yet, so `m = 2` and `l = exp(0) = 1`.
2. `x = 1`. The max holds. `l = 1 + exp(1 - 2) = 1.3679`.
3. `x = 5`. The max moves from 2 to 5. The old sum shrinks by `exp(2 - 5) = 0.0498`, giving 0.0681, and the new term `exp(0) = 1` is added, so `l = 1.0681`.
4. `x = 3`. The max holds. `l = 1.0681 + exp(3 - 5) = 1.2034`.

The final pair is `m = 5` and `l = 1.2034`, the same pair the three-trip method finds. The probabilities are `exp(x - 5) / 1.2034`, which is `[0.0414, 0.0152, 0.8310, 0.1125]`. Milakov and Gimelshein derive this recurrence as the "online normalizer" for softmax.

> [!KEY]
> A softmax needs only two running numbers per row, the max `m` and the sum `l`. When the max moves, multiply the old sum by `exp(m_old - m_new)`. That one factor makes the streaming result equal the three-trip result.

Writing out the probabilities still needs a second look at the scores, because each weight is `exp(x - m) / l`. The stream pays off when you want a blend of values, as in attention, because the blend can be accumulated at the same time.

## Carrying the blend along

In attention, a row of scores is paired with a row of values, and the output is the weighted blend of those values. Suppose the row arrives in two blocks. Block 1 holds scores `[2, 1]` with values `[10, 20]`, and block 2 holds scores `[5, 3]` with values `[30, 40]`. For each block, keep three numbers relative to that block's own max. Call the triple a **state**.

- `m`, the block's max score.
- `l`, the sum of `exp(score - m)` over the block.
- `o`, the unnormalised output: the sum of `exp(score - m) * value`.

For block 1, `m = 2`, `l = 1 + 0.3679 = 1.3679`, and `o = 1 * 10 + 0.3679 * 20 = 17.3576`. For block 2, `m = 5`, `l = 1 + 0.1353 = 1.1353`, and `o = 1 * 30 + 0.1353 * 40 = 35.4134`.

To merge the two states, move both to the larger max, 5. Block 2 is already there. Block 1 shrinks by `alpha = exp(2 - 5) = 0.0498`. The factor multiplies `l` and `o` alike, because both were measured against the max 2, and every term inside them gets the same factor:

$$
m = \max(m_a, m_b), \qquad l = l_a\, e^{m_a - m} + l_b\, e^{m_b - m}, \qquad o = o_a\, e^{m_a - m} + o_b\, e^{m_b - m}
$$

Here `l = 1.3679 * 0.0498 + 1.1353 = 1.2034`, and `o = 17.3576 * 0.0498 + 35.4134 = 36.2776`. The output is `o / l = 36.2776 / 1.2034 = 30.145`. A dense computation with the weights `[0.0414, 0.0152, 0.8310, 0.1125]` gives `0.414 + 0.304 + 24.93 + 4.50 = 30.145`. The two agree.

The merge has two properties that carry the rest of the chapter. First, it is **associative**: merging `(a, b)` and then `c` gives the same state as merging `a` and then `(b, c)`. Each state describes a set of keys, and the merged state describes the union, so the grouping cannot matter. So you can process the blocks left to right, as a tree, or in parallel, and get the same answer. Second, the empty state `m = -inf`, `l = 0`, `o = 0` is the identity, so a loop can start from it.

### The mistake the rescale prevents

Suppose the merge adds `o` without the factor, so it computes `o = o_a + o_b * beta`. On the worked example, `o = 17.3576 + 35.4134 = 52.771`, and the output is `52.771 / 1.2034 = 43.85` instead of 30.145. The bug passes any test in which the largest score sits in the first block, because then `alpha` is 1 for the later blocks and the missing factor does nothing. On random data it shows up as soon as the max arrives late, so the exercise tests exactly that.

> [!WARNING]
> Rescale `o` by the same factor as `l`, every time the max changes. Forgetting it gives the right answer when the largest score comes first and a wrong answer otherwise, so a test needs the max to arrive late.

## Tiling the attention table

Now put the state to work. Cut the queries into blocks of `B` rows and the keys and values into blocks of `B` rows. Take one block of queries. Start from the empty state for each of its rows. For each key block in turn, compute the `B` by `B` tile of scores, turn it into a state with `block_state`, and merge it into the running state. When the last key block has been merged, divide `o` by `l`. That is the output for the `B` query rows. Move to the next query block.

```python
for i in range(0, T, B):                      # one block of query rows
    state = empty_state(rows=B)               # m = -inf, l = 0, o = 0
    for j in range(0, T, B):                  # one block of keys and values
        s = q[i:i+B] @ k[j:j+B].T / sqrt(d)   # a B by B tile, the only table held
        state = merge_states(state, block_state(s, v[j:j+B]))
    m, l, o = state
    out[i:i+B] = o / l[:, None]
```

The only table in memory at any moment is the `B` by `B` tile `s`. The full `T` by `T` table never exists. The extra memory is the tile plus the three state arrays, which is `B * B + B * (d + 2)` numbers, and it does not grow with `T`. Over a whole layer the extra memory is linear in `T`, because of the output. Rabe and Staats make the same point: the quadratic table comes from the order of the loops, not from the maths.

What this buys is memory, not flops. Every tile still does a `B` by `d` times `d` by `B` product and a `B` by `B` times `B` by `d` product. The total is the same `4 T² d` as in the last chapter, plus the cost of rescaling, which grows with `T²/B`. Tiling reorders the work so that it fits in small memory.

### The causal mask with blocks

A causal mask sets every score above the diagonal to minus infinity. In a tiled layout it falls into three cases, each handled differently.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A four by four grid of tiles for a causal table with T equal 8 and block 2: 4 diagonal tiles partly masked, 6 full tiles below, 6 skipped tiles above">
<text x="48" y="67.0" text-anchor="end" class="fig-small">rows 0-1</text>
<text x="89.0" y="24" text-anchor="middle" class="fig-small">keys 0-1</text>
<rect x="60" y="34" width="58" height="58" class="fill-soft-b stroke-b"/>
<text x="89.0" y="67.0" text-anchor="middle" class="fig-small">mask</text>
<rect x="118" y="34" width="58" height="58" class="fill-sunk stroke-grid dashed"/>
<text x="147.0" y="67.0" text-anchor="middle" class="fig-small">skip</text>
<rect x="176" y="34" width="58" height="58" class="fill-sunk stroke-grid dashed"/>
<text x="205.0" y="67.0" text-anchor="middle" class="fig-small">skip</text>
<rect x="234" y="34" width="58" height="58" class="fill-sunk stroke-grid dashed"/>
<text x="263.0" y="67.0" text-anchor="middle" class="fig-small">skip</text>
<text x="48" y="125.0" text-anchor="end" class="fig-small">rows 2-3</text>
<text x="147.0" y="24" text-anchor="middle" class="fig-small">keys 2-3</text>
<rect x="60" y="92" width="58" height="58" class="fill-soft-a stroke-a"/>
<text x="89.0" y="125.0" text-anchor="middle" class="fig-small">full</text>
<rect x="118" y="92" width="58" height="58" class="fill-soft-b stroke-b"/>
<text x="147.0" y="125.0" text-anchor="middle" class="fig-small">mask</text>
<rect x="176" y="92" width="58" height="58" class="fill-sunk stroke-grid dashed"/>
<text x="205.0" y="125.0" text-anchor="middle" class="fig-small">skip</text>
<rect x="234" y="92" width="58" height="58" class="fill-sunk stroke-grid dashed"/>
<text x="263.0" y="125.0" text-anchor="middle" class="fig-small">skip</text>
<text x="48" y="183.0" text-anchor="end" class="fig-small">rows 4-5</text>
<text x="205.0" y="24" text-anchor="middle" class="fig-small">keys 4-5</text>
<rect x="60" y="150" width="58" height="58" class="fill-soft-a stroke-a"/>
<text x="89.0" y="183.0" text-anchor="middle" class="fig-small">full</text>
<rect x="118" y="150" width="58" height="58" class="fill-soft-a stroke-a"/>
<text x="147.0" y="183.0" text-anchor="middle" class="fig-small">full</text>
<rect x="176" y="150" width="58" height="58" class="fill-soft-b stroke-b"/>
<text x="205.0" y="183.0" text-anchor="middle" class="fig-small">mask</text>
<rect x="234" y="150" width="58" height="58" class="fill-sunk stroke-grid dashed"/>
<text x="263.0" y="183.0" text-anchor="middle" class="fig-small">skip</text>
<text x="48" y="241.0" text-anchor="end" class="fig-small">rows 6-7</text>
<text x="263.0" y="24" text-anchor="middle" class="fig-small">keys 6-7</text>
<rect x="60" y="208" width="58" height="58" class="fill-soft-a stroke-a"/>
<text x="89.0" y="241.0" text-anchor="middle" class="fig-small">full</text>
<rect x="118" y="208" width="58" height="58" class="fill-soft-a stroke-a"/>
<text x="147.0" y="241.0" text-anchor="middle" class="fig-small">full</text>
<rect x="176" y="208" width="58" height="58" class="fill-soft-a stroke-a"/>
<text x="205.0" y="241.0" text-anchor="middle" class="fig-small">full</text>
<rect x="234" y="208" width="58" height="58" class="fill-soft-b stroke-b"/>
<text x="263.0" y="241.0" text-anchor="middle" class="fig-small">mask</text>
<rect x="332" y="50" width="14" height="14" class="fill-soft-b stroke-b"/><text x="354" y="62" class="fig-small">4 diagonal tiles: compute, then mask</text>
<rect x="332" y="84" width="14" height="14" class="fill-soft-a stroke-a"/><text x="354" y="96" class="fig-small">6 tiles below: compute, no mask</text>
<rect x="332" y="118" width="14" height="14" class="fill-sunk stroke-grid dashed"/><text x="354" y="130" class="fig-small">6 tiles above: never computed</text>
<text x="332" y="170" class="fig-small fig-bold">10 of 16 tiles run</text>
<text x="332" y="190" class="fig-small">at T = 4096, block = 128:</text>
<text x="332" y="208" class="fig-small">528 of 1024, or 51.6 percent</text>
</svg>
Figure 2. The 4 by 4 grid of tiles for T = 8 and block = 2 under a causal mask. Tiles below the diagonal are full, tiles on the diagonal are computed and then masked, and tiles above the diagonal are never computed. Ten of 16 tiles run.
```

- A tile wholly below the diagonal has every key at or before every query. It needs no mask.
- A tile on the diagonal has some keys after some queries. Compute it, then set the scores where the key index is larger than the query index to minus infinity.
- A tile wholly above the diagonal has every score masked. **Skip it.** Computing it wastes flops, and there is a second reason that the next warning explains.

With `nb = ceil(T / B)` tiles per side, the tiles on or below the diagonal number `nb * (nb + 1) / 2`. For `T = 8` and `B = 2`, `nb = 4`, so 10 of 16 tiles run and 6 are skipped. At `T = 4096` and `B = 128`, 528 of 1,024 tiles run, which is 51.6 percent. As `T` grows, the share tends to one half, which is the saving a causal mask promises.

> [!WARNING]
> A tile with every score at minus infinity has a row max of minus infinity, and `exp(-inf - (-inf))` is not a number. It is `nan`, and one `nan` spreads through every state it merges with. Skipping the tile avoids it. On a diagonal tile, the diagonal entry of each row stays unmasked, so each row's max is finite.

## The backward pass without the table

Dense attention saves the scores `S` and the weights `P` for the backward pass, so each head stores `2 T²` numbers. Tiled attention has no table to save. It has two choices: save every tile, which brings back the quadratic memory, or **recompute** each tile from `Q`, `K`, and one number per row.

That number is the **log-sum-exp** of the row, `L = m + log(l)`, which the forward pass already has when it finishes the last block. The reason it is enough: the weight for a score `s` is `exp(s - m) / l`, which equals `exp(s - m - log l) = exp(s - L)`. So the backward pass rebuilds any weight with one subtraction and one `exp`:

$$
P_{ij} = e^{\,s_{ij} - L_i}
$$

No sum is needed, so it works tile by tile, and the rows sum to 1 because `L` is the log of that row's own sum.

The softmax rule from the last chapter needs `rowsum = sum(P * dP)` along the row, where `dP = dout @ V.T`. Computing it needs the whole row of `dP`, which is another table. There is a shortcut. Substituting `dP_ij = dout_i · v_j` gives

$$
\sum_j P_{ij}\, dP_{ij} = \sum_j P_{ij}\, (dout_i \cdot v_j) = dout_i \cdot \sum_j P_{ij}\, v_j = dout_i \cdot out_i
$$

Read it aloud: the row correction is the dot product of the output gradient with the output, both of which have `d` numbers per row. Call it `D_i`. The gradient then follows the same steps as before, with `P` rebuilt from `L` and `rowsum` replaced by `D`:

```python
p = np.exp(q @ k.T / np.sqrt(d) - lse[:, None])     # rebuilt, not saved
big_d = np.sum(dout * out, axis=-1, keepdims=True)  # one number per row
dv = p.T @ dout
ds = p * (dout @ v.T - big_d)
dq = ds @ k / np.sqrt(d)
dk = ds.T @ q / np.sqrt(d)
```

The exercise writes this untiled to keep the focus on what is saved. A kernel runs the same lines on one tile at a time.

> [!KEY]
> The forward pass saves one number per row, the log-sum-exp `L`, and the output it has to produce anyway. The backward pass rebuilds each weight as `exp(s - L)` and gets the softmax row correction from `dout · out`. No table is stored, and the gradients match the dense ones.

## What the trade costs

Memory first, for one layer with `T = 4096` and 16 heads in float32, counting what is saved beyond the inputs and the output.

| Method | Saved per head | All 16 heads | In bytes |
|---|---|---|---|
| Dense: scores `S` and weights `P` | 2 * 4096² = 33,554,432 | 536,870,912 | 2.0 GiB |
| Tiled: log-sum-exp `L` | 4,096 | 65,536 | 256 KiB |

A 128 by 128 tile in flight is 16,384 numbers, which is 64 KiB per head. The saved memory goes from quadratic to linear in `T`, a factor of 8,192 here.

Flops next, with the layer width `d = 1024` and the same `T`. Each product of a `T` by `d` matrix with a `d` by `T` matrix (or the reverse) costs `2 T² d`.

| Pass | Products | Flops | At T = 4096, d = 1024 |
|---|---|---|---|
| Forward | `Q Kᵀ`, `P V` | 4 T² d | 68.7 billion |
| Backward, `P` saved | `dV`, `dP`, `dQ`, `dK` | 8 T² d | 137.4 billion |
| Backward, `P` recomputed | the four above, plus `Q Kᵀ` | 10 T² d | 171.8 billion |

A training step with a saved table costs `4 + 8 = 12 T² d`, and with recomputation it costs `4 + 10 = 14 T² d`. That is a rise of one sixth, about 17 percent, on the table products. In exchange the table's memory disappears. A causal mask cuts every row of the table by about half.

Bytes moved matter too. Dense attention writes `S`, reads it for the softmax, writes `P`, and reads it for the blend: `4 T²` numbers per head between slow and fast memory. A tiled kernel reads each key and value block again for every query block, which is `2 T² d_h / B` numbers per head, where `d_h` is the head width. With `d_h = 64` and `B = 128` the ratio is `2 B / d_h = 4`. This is a model, not a measurement.

> [!TIP]
> To check a tiled implementation, compare it with dense attention for several block sizes, including 1, a size that does not divide `T`, and a size larger than `T`. The edge blocks are where an off-by-one index hides.

## Where the rest of the story lives

This chapter owns the algorithm: the streaming softmax, the merge, the tile loop, and the recompute. Making it fast on a GPU belongs to two other courses. The inference course covers [FlashAttention](https://qwen.ifkash.dev/c/14-flash-attention), then [split-KV decode attention](https://qwen.ifkash.dev/c/14a-split-kv-decode-attention) for one query against a long cache, and [paged attention](https://qwen.ifkash.dev/c/15-paged-attention) for a cache stored in pages. The GPU course writes the [FlashAttention kernel](https://gpu.ifkash.dev/c/16-flash-attention) in Triton, forward and backward. Linear attention and state-space models change the maths of the layer instead of the schedule, so they are outside this chapter.

> [!RECAP]
> Before you build:
> - Stream a softmax with `m` and `l`: `m_new = max(m, x)` and `l = l * exp(m - m_new) + exp(x - m_new)`.
> - A state `(m, l, o)` summarises a block of keys. Merge two states by moving both to the larger max and adding. The merge is associative, and `o` is rescaled exactly like `l`.
> - Tiled attention loops over query blocks and key blocks, holds one `B` by `B` tile, and skips tiles wholly above the causal diagonal. `ceil(T/B) * (ceil(T/B) + 1) / 2` tiles run.
> - Save `L = m + log(l)` per row. The backward pass rebuilds `P = exp(s - L)` and uses `D = sum(dout * out)` for the softmax correction.
> - Recomputing costs one more `Q Kᵀ` product, `10 T² d` against `8 T² d` backward, and cuts the saved memory from `2 T²` to `T` per head.

## Build it step by step

`dense_attention` and `block_state` are provided in the starter. `block_state(scores, v)` turns one block of scores and values into the state `(m, l, o)`.

1. **`online_stats(stream)`.** Take any iterable of floats, which the tests pass as a one-shot generator, and return `(m, l)`. Start from `m = -inf` and `l = 0`, and update both with the rule above. The usual mistake is to compute `max(stream)` first, which consumes the generator.
2. **`online_softmax(x)`.** Call `online_stats`, then return `exp(x - m) / l`. The tests include scores near 1,000, where an unshifted `exp` overflows.
3. **`merge_states(a, b)`.** Each state is `(m, l, o)` with shapes `(n,)`, `(n,)`, and `(n, d_v)`. Move both to `m = maximum(ma, mb)` and add. The usual mistake is to forget the factor on `o`, or to forget `[:, None]` so that the factor lines up with the rows of `o`.
4. **`tiled_attention(q, k, v, block, causal=False)`.** Loop over query blocks and key blocks. Scale the tile by `1/sqrt(d)`, add the causal mask with minus infinity, call `block_state`, and merge. Return `(out, lse, peak)`, where `lse = m + log(l)` and `peak` is the largest tile size held. Break out of the key loop once a block lies wholly above the diagonal. The usual mistake is the missing rescale, and the test that catches it puts the largest score in a late block.
5. **`count_causal_blocks(T, block)`.** Return `(computed, skipped)` with `nb = ceil(T / block)`. The usual mistake is to use `T // block`, which drops the short last tile.
6. **`recompute_probs(q, k, lse, causal=False)`.** Return `exp(scores - lse[:, None])` with the masked entries set to 0. It must not call a softmax.
7. **`backward_from_lse(dout, q, k, v, out, lse, causal=False)`.** Rebuild `P`, compute `D = sum(dout * out)` with `keepdims=True`, and apply the five lines above. The usual mistake is to take `D` from `P * dP` after all, which needs the table the method is trying to avoid.
8. **`extra_saved(T, tiled)` and `score_table_flops(T, d, mode)`.** Return `2*T*T` or `T` numbers, and `4`, `8`, or `10` times `T*T*d` for the three modes.

## Where this breaks

Tiling saves memory and data movement. It does not cut the flops, so a very long sequence is still quadratic in time. The merge is exact in real arithmetic, but every rescale multiplies by a number below 1, and in half precision the rounding errors add up, so kernels keep `m`, `l`, and `o` in float32. A block size of 1 is correct and slow, because loop overhead dominates.

## The exercise

You write `online_stats`, `online_softmax`, `merge_states`, `tiled_attention`, `count_causal_blocks`, `recompute_probs`, `backward_from_lse`, `extra_saved`, and `score_table_flops`. The checks confirm the worked pair `m = 5`, `l = 1.2034` from a one-shot stream, and that `online_softmax` equals the dense softmax to 1e-12, including scores near 1,000. They confirm that `merge_states` is associative, that `tiled_attention` equals dense attention for several block sizes, including when the largest score arrives in a late block, and, with the causal mask, that `peak` is `block * block` and that memory measured at `T = 1024` stays far below the dense table. The last checks cover the tile counts, the weights rebuilt exactly from `lse`, the gradients against finite differences, and the memory and flop counts.

## Further reading

- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). The one-pass recurrence for the max and the sum.
- Rabe and Staats, [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682). Attention computed in chunks with a running softmax.
- Dao, Fu, Ermon, Rudra, and Ré, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). The tiled forward and backward passes, and the account of memory traffic.
- Stanford CS336, [Spring 2025 schedule](http://cs336.stanford.edu/spring2025/). Lecture 5, "GPUs", and Lecture 6, "Kernels, Triton", place these tiles on real hardware.
