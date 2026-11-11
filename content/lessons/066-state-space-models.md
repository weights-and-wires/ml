---
title: "State-space models and linear attention"
slug: 066-state-space-models
part: "Part 6 — Transformers"
summary: Turn attention into a recurrence, read a state-space model as a recurrence or a convolution, and run recurrences in parallel with a scan.
minutes: 65
gpu: false
objectives:
  - Explain why a transformer's cost per generated token grows with the length of its context, and what a fixed-size state changes.
  - Write causal linear attention as a quadratic masked form and as a running sum, and show that the two agree.
  - Unroll a linear state-space model into a convolution with the kernel C A^j B, and discretise a continuous system with a zero-order hold.
  - State the stability condition for a diagonal state matrix and say what an input-dependent (selective) model gives up and keeps.
  - Compose recurrence steps with an associative rule, and run a linear recurrence in a logarithmic number of parallel rounds.
prereqs:
  - 063-the-transformer-block
exercise: 066-state-space-models
---

# State-space models and linear attention

A chat model has produced 4,000 tokens, and you ask for the 4,001st. Attention compares the current token's query against the keys of all 4,000 earlier tokens and reads all 4,000 values. The 4,002nd token reads 4,001 of each. The work per token grows with every token already written, so a long answer costs far more than a short one, and the cache grows with it.

A person reading a book does not behave like that. They carry a summary and update it with each sentence, so the cost of a sentence does not depend on how many came before. This chapter asks whether a layer can work that way and still be a good sequence model: keep a fixed-size state, fold each token into it, and read the answer out of it.

It can, in two forms that are close relatives. Linear attention swaps the softmax for a feature map, and the attention sum becomes a running sum. A state-space model starts from a recurrence with a fixed rule, which you can read as a recurrence or as a convolution. You will build both, check that each pair of views agrees, and then see how a recurrence trains in parallel with a scan.

## What you already know

- Attention is `softmax(Q @ K.T / sqrt(d)) @ V`, and a causal mask stops each position from reading later ones.
- One attention layer costs `8*T*d**2 + 4*T**2*d` flops, and a decoder keeps a key-value cache that grows by one row per token.
- Tiled attention saves memory but not flops.
- A convolution computes each output as a weighted sum of nearby inputs, with the same weights everywhere.

## Why attention gets more expensive with every token

[The cost chapter](/lessons/060-attention-backward-and-cost) counted the training cost: the `T` by `T` table makes a layer's flops grow as `T²`. Generation has the same shape in a different place. With a key-value cache, the model does not recompute earlier keys and values. The current token still has to read all of them. At position `t` with head width `d`, that is `2·t·d` numbers per head per layer: `t` keys and `t` values.

Figure 1 plots that against a fixed-size alternative that this chapter builds. For `d = 64`, the fixed state holds `64·64 + 64 = 4,160` numbers. Attention reads fewer numbers than that for the first 32 tokens. At token 4,096 it reads 524,288, which is 126 times as many, and the gap keeps widening.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Values read per generated token against the number of earlier tokens: attention grows in a straight line, linear attention stays flat, and the lines cross near token 32">
<line x1="60" y1="200" x2="560" y2="200" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="200" class="stroke-muted"/>
<line x1="60" y1="200" x2="60" y2="205" class="stroke-muted"/><text x="60" y="222" text-anchor="middle" class="fig-small">0</text>
<line x1="180" y1="200" x2="180" y2="205" class="stroke-muted"/><text x="180" y="222" text-anchor="middle" class="fig-small">64</text>
<line x1="300" y1="200" x2="300" y2="205" class="stroke-muted"/><text x="300" y="222" text-anchor="middle" class="fig-small">128</text>
<line x1="420" y1="200" x2="420" y2="205" class="stroke-muted"/><text x="420" y="222" text-anchor="middle" class="fig-small">192</text>
<line x1="540" y1="200" x2="540" y2="205" class="stroke-muted"/><text x="540" y="222" text-anchor="middle" class="fig-small">256</text>
<line x1="55" y1="200" x2="60" y2="200" class="stroke-muted"/><text x="50" y="204" text-anchor="end" class="fig-small">0</text>
<line x1="55" y1="115" x2="60" y2="115" class="stroke-muted"/><text x="50" y="119" text-anchor="end" class="fig-small">16,384</text>
<line x1="55" y1="30" x2="60" y2="30" class="stroke-muted"/><text x="50" y="34" text-anchor="end" class="fig-small">32,768</text>
<line x1="60.0" y1="200.0" x2="540.0" y2="30.0" class="stroke-a" stroke-width="2.5"/>
<line x1="60.0" y1="178.4" x2="540.0" y2="178.4" class="stroke-b" stroke-width="2.5"/>
<circle cx="120.9" cy="178.4" r="5" class="fill-ink"/><text x="131" y="166" class="fig-small">cross near token 32</text>
<text x="420" y="61" class="fig-small fig-bold">attention: 2 t d</text>
<text x="330" y="170" class="fig-small fig-bold">linear attention: d d + d = 4,160</text>
<text x="310" y="244" text-anchor="middle" class="fig-small">earlier tokens t</text></svg>
Figure 1. Numbers read to produce one token, with head width 64. Attention reads the whole cache, so its line climbs. A fixed state reads the same 4,160 numbers at every position. The lines cross near token 32, and by token 4,096 attention reads 126 times as many.
```

The [online softmax chapter](/lessons/061-online-softmax-and-tiled-attention) made exact attention cheaper to move through memory. It did not change what the layer computes, so every query still reads every earlier key. This chapter changes the layer.

## Linear attention: replace the exponential with a feature map

Attention's weight between a query `q` and a key `k` is `exp(q · k)`, normalised over the row. That exponential is why the table is needed. It does not split into a function of `q` times a function of `k`. If it did, you could group the sums by key and never form the table.

Replace it with something that does split. Pick a **feature map** `phi` that sends a vector to a vector of positive numbers, and define the weight between query `t` and key `s` as the dot product `phi(q_t) · phi(k_s)`. A common choice is `phi(x) = elu(x) + 1`, which equals `x + 1` for positive `x` and `exp(x)` for the rest, so it is never negative. Positive features keep every weight positive, so the row normalisation divides by a positive sum.

The causal output for position `t` is a weighted average of the values at positions `0` to `t`:

$$
y_t = \frac{\sum_{s \le t} \left(\phi(q_t) \cdot \phi(k_s)\right) v_s}{\sum_{s \le t} \phi(q_t) \cdot \phi(k_s)}
$$

Read it aloud: score each earlier position by the dot product of features, then average the values with those scores as weights. This is the **quadratic form**, because it builds the `T` by `T` table of scores and masks the upper triangle, exactly as ordinary attention does.

Pull `phi(q_t)` out of both sums. The dot product `phi(q_t) · phi(k_s)` is a sum over the `d` feature entries, so the numerator is `phi(q_t)` times the matrix `S_t = sum of phi(k_s) v_s^T`, and the denominator is `phi(q_t)` times the vector `z_t = sum of phi(k_s)`:

$$
S_t = S_{t-1} + \phi(k_t)\, v_t^{\top}, \qquad z_t = z_{t-1} + \phi(k_t), \qquad y_t = \frac{\phi(q_t)^{\top} S_t}{\phi(q_t)^{\top} z_t}
$$

Read it aloud: keep a `d` by `dv` matrix `S` that adds one outer product per token, keep a vector `z` that adds the key features, and answer each query with one vector-matrix product. This is the **recurrent form**, an RNN whose hidden state is a matrix. Its cost per token is `d·dv`, however long the context.

Here is a three-token example you can check by hand, with `d = 2` and one value channel. The features are given directly: `phi(Q) = [[1,0],[1,1],[0,1]]`, `phi(K) = [[1,0],[0,1],[1,1]]`, and `V = [2, 4, 6]`.

| Step `t` | Added to `S` | `S` | `z` | Numerator | Denominator | `y_t` |
|---|---|---|---|---|---|---|
| 1 | `[1,0]·2` | `[2, 0]` | `[1, 0]` | 2 | 1 | 2 |
| 2 | `[0,1]·4` | `[2, 4]` | `[1, 1]` | 6 | 2 | 3 |
| 3 | `[1,1]·6` | `[8, 10]` | `[2, 2]` | 10 | 2 | 5 |

The quadratic form agrees. The masked scores are `[[1,0,0],[1,1,0],[0,1,1]]`. Row 2 averages values 2 and 4 with equal weights to get 3, and row 3 averages 4 and 6 to get 5.

> [!WARNING]
> Update the state before you read it. A loop that reads `S` first leaves out the current token, so row 1 divides 0 by 0 and every later row misses its own step. The output looks plausible and is wrong by exactly one token.

The price is exact lookup. A softmax can put nearly all its weight on one earlier token, but the state `S` holds only `d·dv` numbers. It can store about `d` unrelated key-value pairs without interference, and a longer context blurs.

> [!KEY]
> Replace `exp(q · k)` with `phi(q) · phi(k)` and causal attention equals a running sum. Train with the parallel table, generate with the running sum. They give the same numbers.

## A state-space model: a recurrence with a fixed rule

Linear attention starts from attention and ends at a recurrence. A **state-space model** starts at the recurrence. Keep a hidden state `h` of `n` numbers. At each step the model multiplies the state by a matrix `A`, adds the input scaled by a vector `B`, and reads the output with a vector `C`:

$$
h_t = A\, h_{t-1} + B\, x_t, \qquad y_t = C\, h_t
$$

Read it aloud: the state decays or rotates according to `A`, each input pushes the state along `B`, and `C` reports one number from the state. Here `x_t` is one number per step, and a real layer runs many such systems side by side, one for each channel of the model's width. The rule is **linear** and **time-invariant**: `A`, `B`, and `C` are the same at every step, and the state enters linearly.

Take the smallest case: one state, `A = 0.5`, `B = 1`, `C = 1`, and the inputs `x = [1, 2, 3]`. Start with `h = 0`. Step 1 gives `h = 0.5·0 + 1 = 1`. Step 2 gives `0.5·1 + 2 = 2.5`. Step 3 gives `0.5·2.5 + 3 = 4.25`. So `y = [1, 2.5, 4.25]`. Each input fades by half at every step after it arrives, and the state is the sum of those fading echoes.

## The same model as a convolution

That sum of echoes is the key. Unroll the recurrence. The state after step `t` is

$$
h_t = \sum_{j=0}^{t} A^{j} B\, x_{t-j}, \qquad y_t = \sum_{j=0}^{t} K_j\, x_{t-j}, \qquad K_j = C\, A^{j} B
$$

Read it aloud: an input that arrived `j` steps ago has been multiplied by `A` a total of `j` times, and the output weighs it by `K_j`. That weighted sum over lags is a causal **convolution** with the **kernel** `K`. The kernel is the model's response to a single 1 followed by zeros, called its impulse response.

For the example, `K = [1, 0.5, 0.25]`. Then `y_3 = 1·x_3 + 0.5·x_2 + 0.25·x_1 = 3 + 1 + 0.25 = 4.25`, which matches the recurrence. This is the **recurrence-convolution duality**: one model, two ways to evaluate it.

The two views suit different jobs. Generation uses the recurrence: each token costs one update of an `n`-number state, with no cache. Training can use the convolution: all `T` outputs depend only on the inputs and the kernel, so they come at once, and a fast Fourier transform brings the cost from `T²` to `T log T`. S4 (Gu, Goel, and Ré, 2021) builds on this.

## Stability and the diagonal matrix

A kernel `K_j = C A^j B` must not blow up, and that depends on the powers of `A`. Take the simplest case, a **diagonal** matrix with entries `λ_1` to `λ_n`. Its power is `diag(λ_1^j, ..., λ_n^j)`, and the kernel becomes a sum of `n` geometric sequences:

$$
K_j = \sum_{i=1}^{n} C_i\, \lambda_i^{\,j}\, B_i
$$

Each term decays when `|λ_i| < 1`, stays flat when `|λ_i| = 1`, and grows when `|λ_i| > 1`. Figure 2 plots `a^j` for three values of one entry.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The kernel a to the power j for a equal to 0.5, 0.9, and 1.05: the first dies in a few steps, the second fades slowly, and the third grows">
<line x1="60" y1="200" x2="560" y2="200" class="stroke-muted"/>
<line x1="60" y1="20" x2="60" y2="200" class="stroke-muted"/>
<line x1="55" y1="200" x2="60" y2="200" class="stroke-muted"/><text x="50" y="204" text-anchor="end" class="fig-small">0</text>
<line x1="55" y1="100" x2="60" y2="100" class="stroke-muted"/><text x="50" y="104" text-anchor="end" class="fig-small">1</text>
<line x1="60" y1="100" x2="560" y2="100" class="stroke-grid dashed"/>
<text x="70" y="218" text-anchor="middle" class="fig-small">0</text>
<text x="190" y="218" text-anchor="middle" class="fig-small">3</text>
<text x="310" y="218" text-anchor="middle" class="fig-small">6</text>
<text x="430" y="218" text-anchor="middle" class="fig-small">9</text>
<text x="510" y="218" text-anchor="middle" class="fig-small">11</text>
<polyline points="70,100.0 110,150.0 150,175.0 190,187.5 230,193.8 270,196.9 310,198.4 350,199.2 390,199.6 430,199.8 470,199.9 510,200.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="70" cy="100.0" r="3" class="fill-a"/>
<circle cx="110" cy="150.0" r="3" class="fill-a"/>
<circle cx="150" cy="175.0" r="3" class="fill-a"/>
<circle cx="190" cy="187.5" r="3" class="fill-a"/>
<circle cx="230" cy="193.8" r="3" class="fill-a"/>
<circle cx="270" cy="196.9" r="3" class="fill-a"/>
<circle cx="310" cy="198.4" r="3" class="fill-a"/>
<circle cx="350" cy="199.2" r="3" class="fill-a"/>
<circle cx="390" cy="199.6" r="3" class="fill-a"/>
<circle cx="430" cy="199.8" r="3" class="fill-a"/>
<circle cx="470" cy="199.9" r="3" class="fill-a"/>
<circle cx="510" cy="200.0" r="3" class="fill-a"/>
<polyline points="70,100.0 110,110.0 150,119.0 190,127.1 230,134.4 270,141.0 310,146.9 350,152.2 390,157.0 430,161.3 470,165.1 510,168.6" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="70" cy="100.0" r="3" class="fill-c"/>
<circle cx="110" cy="110.0" r="3" class="fill-c"/>
<circle cx="150" cy="119.0" r="3" class="fill-c"/>
<circle cx="190" cy="127.1" r="3" class="fill-c"/>
<circle cx="230" cy="134.4" r="3" class="fill-c"/>
<circle cx="270" cy="141.0" r="3" class="fill-c"/>
<circle cx="310" cy="146.9" r="3" class="fill-c"/>
<circle cx="350" cy="152.2" r="3" class="fill-c"/>
<circle cx="390" cy="157.0" r="3" class="fill-c"/>
<circle cx="430" cy="161.3" r="3" class="fill-c"/>
<circle cx="470" cy="165.1" r="3" class="fill-c"/>
<circle cx="510" cy="168.6" r="3" class="fill-c"/>
<polyline points="70,100.0 110,95.0 150,89.8 190,84.2 230,78.4 270,72.4 310,66.0 350,59.3 390,52.3 430,44.9 470,37.1 510,29.0" fill="none" class="stroke-bad" stroke-width="2.5"/>
<circle cx="70" cy="100.0" r="3" class="fill-bad"/>
<circle cx="110" cy="95.0" r="3" class="fill-bad"/>
<circle cx="150" cy="89.8" r="3" class="fill-bad"/>
<circle cx="190" cy="84.2" r="3" class="fill-bad"/>
<circle cx="230" cy="78.4" r="3" class="fill-bad"/>
<circle cx="270" cy="72.4" r="3" class="fill-bad"/>
<circle cx="310" cy="66.0" r="3" class="fill-bad"/>
<circle cx="350" cy="59.3" r="3" class="fill-bad"/>
<circle cx="390" cy="52.3" r="3" class="fill-bad"/>
<circle cx="430" cy="44.9" r="3" class="fill-bad"/>
<circle cx="470" cy="37.1" r="3" class="fill-bad"/>
<circle cx="510" cy="29.0" r="3" class="fill-bad"/>
<text x="500" y="21" text-anchor="end" class="fig-small fig-bold">a = 1.05: 1.71, grows</text>
<text x="390" y="149" class="fig-small fig-bold">a = 0.9: 0.31 at j = 11</text>
<text x="156" y="169" class="fig-small fig-bold">a = 0.5: 0.0005 at j = 11</text>
<text x="310" y="244" text-anchor="middle" class="fig-small">lag j</text></svg>
Figure 2. The kernel entry a^j for a = 0.5, 0.9, and 1.05. At 0.5 the memory is gone in a few steps. At 0.9 an input still counts for 0.31 after 11 steps. At 1.05 the response grows, so an old input matters more than a recent one and the model is unstable.
```

So a diagonal model is stable when every `|λ_i| < 1`. A value near 1 gives a long memory and a value near 0 a short one. For a general `A` the same holds for its eigenvalues. A diagonal `A` also makes `A h` an elementwise product costing `n` operations instead of `n²`.

## Discretising a continuous system

Most state-space papers start from a system in continuous time, `h'(τ) = a h(τ) + b x(τ)`, and a sequence is a set of samples of it. To get a recurrence, you choose a step size `Δ` and hold the input constant between samples. That is the **zero-order hold**. With a constant input `x_t` over one step, the differential equation has an exact solution, and for a diagonal `a` it gives

$$
\bar a = e^{\Delta a}, \qquad \bar b = \frac{e^{\Delta a} - 1}{a}\, b, \qquad h_t = \bar a\, h_{t-1} + \bar b\, x_t
$$

Read it aloud: after `Δ` units of time the old state has decayed by `e^{Δ a}`, and the held input has had time to push the state up by `(e^{Δa} − 1)/a` times `b`. Check it with `a = -1`, `b = 1`, `Δ = 1`. The state starts at 0 and relaxes toward 1 under `h' = -h + 1`, so after one unit it reaches `1 − e^{-1} = 0.6321`. The formula gives `ā = e^{-1} = 0.3679` and `b̄ = (0.3679 − 1)/(−1) = 0.6321`. They agree.

Two properties matter. If `a < 0`, then `ā` lies strictly between 0 and 1 for every `Δ > 0`, so the discrete model is stable for any step size. And `Δ` sets the memory: a small `Δ` gives `ā` near 1, so the state keeps its past, and a large `Δ` gives `ā` near 0, so it forgets. When `a` is 0 the formula divides by zero, and the limit is `b̄ = Δ b`, so code needs a branch.

## Selective models: let the rule depend on the input

A time-invariant model has a weakness: its weight on an input depends only on how long ago the input arrived. It cannot say "ignore this token" or "forget everything before this token". A model that reads text needs to hold on to a name and drop filler words.

A **selective** state-space model, the idea behind Mamba (Gu and Dao, 2023), lets the rule depend on the input. The step size `Δ_t`, and the vectors `B_t` and `C_t`, are computed from `x_t` with small learned projections. With a diagonal `a`, each step becomes

$$
h_t = \bar a_t \odot h_{t-1} + \bar b_t\, x_t, \qquad \bar a_t = e^{\Delta_t a}, \qquad y_t = C_t \cdot h_t
$$

where `⊙` multiplies entry by entry. Read it aloud: a large `Δ_t` makes `ā_t` near 0, which wipes the state before the token is written. A small `Δ_t` makes `ā_t` near 1, which keeps the state and lets the token pass without leaving a mark. The model chooses per token.

Take a state of 3. A token with `ā = 0` and `b̄ x = 5` leaves `0·3 + 5 = 5`, a complete reset. A token with `ā = 1` and `b̄ x = 0` leaves 3, untouched.

This flexibility has a price: **the convolution view breaks.** In the unrolled sum the weight on input `s` at output `t` is `C_t` times the product of `ā_r` for `r` from `s+1` to `t`, times `b̄_s`. That weight depends on `s` and `t` separately, and on the data between them. It is no longer one kernel indexed by the lag `t − s`, so there is no single kernel to precompute and no fast Fourier transform. The **recurrence survives**, so generation is as cheap as before. Training needs another way to run a recurrence in parallel, which is the scan.

> [!KEY]
> A time-invariant model can be a recurrence or a convolution. A selective model gives up the convolution and keeps the recurrence, and in return it can reset, skip, and remember depending on the token.

## The parallel scan

A recurrence looks sequential: `h_t` needs `h_{t-1}`. For `h_t = a_t h_{t-1} + b_t` that dependence is only apparent, because it has an algebraic structure you can exploit.

Treat each step as a function that maps a state to a state: step `t` is `h → a_t h + b_t`, which you can store as the pair `(a_t, b_t)`. Applying step 1 and then step 2 gives `a_2 (a_1 h + b_1) + b_2`, which is again of the form `a h + b`, with

$$
(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2)
$$

Read it aloud: the first step's offset `b_1` is carried through the second step's decay `a_2` before the second offset is added. Composing functions is associative, so this rule is too. Associativity lets you group the steps in any tree, and so you can combine many of them at once. The pair for steps 1 to `t`, applied to `h = 0`, has `b` equal to `h_t`.

Take `a = [0.5, 2, 3]` and `b = [1, 1, 1]`. Step by step, `h = 1, 3, 10`. Combining steps 1 and 2 gives `(0.5·2, 2·1 + 1) = (1, 3)`, and then step 3 gives `(1·3, 3·3 + 1) = (3, 10)`, so `b = 10` is `h_3`. Combining steps 2 and 3 first gives `(6, 4)`, and then `(0.5, 1)` with `(6, 4)` gives `(3, 6·1 + 4) = (3, 10)`. The grouping does not matter.

> [!WARNING]
> Compose as `(a_1 a_2, a_2 b_1 + b_2)`. The tempting `(a_1 a_2, b_1 + b_2)` forgets that `b_1` passes through the second step. On the example it gives `h_2 = 2` instead of 3, and the error compounds along the sequence.

A **parallel scan** uses this rule to compute every prefix at once. In the Hillis-Steele version, round `r` combines each element with the one `2^r` places earlier. After round 0 each cell holds a window of 1 step, after round 1 a window of 2, then 4, then 8. A cell is finished when its window reaches back to step 1. Figure 3 shows 8 steps finishing in 3 rounds.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three rounds of a Hillis-Steele scan over 8 steps: after each round a cell holds the combined window of the last 1, 2, 4, then 8 steps, and finished cells are shaded">
<text x="20" y="22" class="fig-small fig-bold">step t</text>
<text x="138" y="22" text-anchor="middle" class="fig-small fig-bold">1</text>
<text x="194" y="22" text-anchor="middle" class="fig-small fig-bold">2</text>
<text x="250" y="22" text-anchor="middle" class="fig-small fig-bold">3</text>
<text x="306" y="22" text-anchor="middle" class="fig-small fig-bold">4</text>
<text x="362" y="22" text-anchor="middle" class="fig-small fig-bold">5</text>
<text x="418" y="22" text-anchor="middle" class="fig-small fig-bold">6</text>
<text x="474" y="22" text-anchor="middle" class="fig-small fig-bold">7</text>
<text x="530" y="22" text-anchor="middle" class="fig-small fig-bold">8</text>
<text x="20" y="56" class="fig-small fig-bold">round 0</text><text x="20" y="72" class="fig-small">window 1</text>
<rect x="112" y="34" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="138" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="168" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="194" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="224" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="250" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="280" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="306" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="336" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="362" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="392" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="418" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="448" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="474" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="504" y="34" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="530" y="59" text-anchor="middle" class="fig-mono">1.000</text>
<text x="20" y="112" class="fig-small fig-bold">round 1</text><text x="20" y="128" class="fig-small">window 2</text>
<rect x="112" y="90" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="138" y="115" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="168" y="90" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="194" y="115" text-anchor="middle" class="fig-mono">1.900</text>
<rect x="224" y="90" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="250" y="115" text-anchor="middle" class="fig-mono">1.800</text>
<rect x="280" y="90" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="306" y="115" text-anchor="middle" class="fig-mono">1.700</text>
<rect x="336" y="90" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="362" y="115" text-anchor="middle" class="fig-mono">1.900</text>
<rect x="392" y="90" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="418" y="115" text-anchor="middle" class="fig-mono">1.600</text>
<rect x="448" y="90" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="474" y="115" text-anchor="middle" class="fig-mono">1.800</text>
<rect x="504" y="90" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="530" y="115" text-anchor="middle" class="fig-mono">1.500</text>
<text x="20" y="168" class="fig-small fig-bold">round 2</text><text x="20" y="184" class="fig-small">window 4</text>
<rect x="112" y="146" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="138" y="171" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="168" y="146" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="194" y="171" text-anchor="middle" class="fig-mono">1.900</text>
<rect x="224" y="146" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="250" y="171" text-anchor="middle" class="fig-mono">2.520</text>
<rect x="280" y="146" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="306" y="171" text-anchor="middle" class="fig-mono">2.764</text>
<rect x="336" y="146" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="362" y="171" text-anchor="middle" class="fig-mono">3.034</text>
<rect x="392" y="146" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="418" y="171" text-anchor="middle" class="fig-mono">2.518</text>
<rect x="448" y="146" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="474" y="171" text-anchor="middle" class="fig-mono">2.712</text>
<rect x="504" y="146" width="52" height="40" rx="4" class="fill-soft-a stroke-a"/><text x="530" y="171" text-anchor="middle" class="fig-mono">2.140</text>
<text x="20" y="224" class="fig-small fig-bold">round 3</text><text x="20" y="240" class="fig-small">window 8</text>
<rect x="112" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="138" y="227" text-anchor="middle" class="fig-mono">1.000</text>
<rect x="168" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="194" y="227" text-anchor="middle" class="fig-mono">1.900</text>
<rect x="224" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="250" y="227" text-anchor="middle" class="fig-mono">2.520</text>
<rect x="280" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="306" y="227" text-anchor="middle" class="fig-mono">2.764</text>
<rect x="336" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="362" y="227" text-anchor="middle" class="fig-mono">3.488</text>
<rect x="392" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="418" y="227" text-anchor="middle" class="fig-mono">3.093</text>
<rect x="448" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="474" y="227" text-anchor="middle" class="fig-mono">3.474</text>
<rect x="504" y="202" width="52" height="40" rx="4" class="fill-soft-c stroke-c"/><text x="530" y="227" text-anchor="middle" class="fig-mono">2.737</text>

</svg>
Figure 3. A scan over 8 steps with a = [0.5, 0.9, 0.8, 0.7, 0.9, 0.6, 0.8, 0.5] and b all 1. Each cell shows its current b. Cells turn green when their window reaches step 1, and the last row equals the h from a step-by-step loop.
```

For `T` steps this takes `ceil(log2 T)` rounds, so the depth is 12 for 4,096 steps and 20 for a million, where a loop has depth `T`. Each round is an elementwise operation that a GPU does in parallel. Hillis-Steele does `O(T log T)` work. Blelloch's scan (1990) uses an up-sweep and a down-sweep to bring the work back to `O(T)`, at about twice the depth. For a diagonal `a_t` each combine costs `n` operations. For a full matrix `A_t` it would need a matrix product, which is one more reason selective models use diagonal states.

The backward pass is a scan too. Suppose the loss has gradient `dh_t` at each state. The state `h_t` feeds the loss directly and also feeds `h_{t+1} = a_{t+1} h_t + b_{t+1}`, so the total gradient `g_t` at `h_t` is

$$
g_t = dh_t + a_{t+1}\, g_{t+1}, \qquad \frac{\partial L}{\partial b_t} = g_t, \qquad \frac{\partial L}{\partial a_t} = g_t\, h_{t-1}
$$

Read it aloud: gradients flow backward through the same decays, and the gradient on `a_t` is the incoming gradient times the state it multiplied. The first equation is a linear recurrence run from the end, so it can use the same scan.

> [!KEY]
> A linear recurrence is a prefix of associative compositions. That makes its depth logarithmic in `T` instead of linear, and its gradient is another linear recurrence. Linear attention is the special case `a_t = 1`.

## Where the kernels live

This chapter owns the maths. Making the scan fast belongs to the GPU course: its chapter on [scan and histogram](https://gpu.ifkash.dev/c/07-scan-and-histogram) builds the block scans of Kogge-Stone and Blelloch that fast kernels start from, and the [GPU course](https://gpu.ifkash.dev/) is the place for a chunked scan. State-space layers in speech encoders belong to the speech course, listed at [courses.ifkash.dev](https://courses.ifkash.dev). Mamba-2 (Dao and Gu, 2024) shows that with one scalar decay per step, a state-space model equals a masked attention whose mask holds products of the decays.

> [!RECAP]
> - Attention reads every cached key and value for each token, so generation costs more as the context grows. A fixed-size state costs the same everywhere.
> - Causal linear attention uses `phi(q) · phi(k)` for the weight, which equals a running state `S_t = S_{t-1} + phi(k_t) v_t^T` read by `phi(q_t)`.
> - A time-invariant state-space model is a recurrence and a causal convolution with kernel `K_j = C A^j B`. A diagonal `A` is stable when every `|λ| < 1`.
> - A selective model makes `Δ`, `B`, and `C` depend on the input. It loses the convolution and keeps the recurrence.
> - The step `(a, b)` composes as `(a_1 a_2, a_2 b_1 + b_2)`, which is associative, so a scan finds all `T` states in `ceil(log2 T)` rounds.

## Build it step by step

The functions are in the order the checks test them.

1. **`feature_map(x)`** returns `elu(x) + 1`. Clip the argument of `exp` before `np.where` picks a branch, because `np.where` computes both branches and a large positive `x` overflows the unused one.
2. **`causal_linear_attention_quadratic(phi_q, phi_k, v)`** takes features `(T, d)` and values `(T, dv)`. Build `phi_q @ phi_k.T`, keep the lower triangle with `np.tril`, divide each row by its sum, and multiply by `v`. The worked example gives `[2, 3, 5]`.
3. **`causal_linear_attention_recurrent(phi_q, phi_k, v)`** returns the same output from `S` and `z`. The usual mistake is reading the state before adding the current step.
4. **`ssm_kernel(A, B, C, T)`** returns `K_j = C A^j B`. Carry a vector `u`, starting at `B`, record `C @ u`, and multiply `u` by `A`. Do not build `A**j` for every `j`.
5. **`ssm_recurrent(A, B, C, x)`** and **`ssm_convolution(A, B, C, x)`** return the same `y` of shape `(T,)`. The second is `np.convolve(x, K)[:T]`. People forget the slice, and the full convolution has `2T − 1` values.
6. **`discretise_zoh(a_diag, b, dt)`** returns `(a_bar, b_bar)`. Use `np.expm1`, and treat `a` near 0 as the limit.
7. **`scan_combine(e1, e2)`** returns the pair for `e1` then `e2`. Write `h → a_2 (a_1 h + b_1) + b_2` before you code it.
8. **`linear_recurrence(a, b)`** and **`parallel_scan(a, b)`** take `(T, n)` arrays and return `h`. For the scan, double a shift each round, combine `x[t - shift]` with `x[t]` for `t >= shift` from the old arrays, and keep the first `shift` entries. It must work when `T` is not a power of two.
9. **`linear_recurrence_backward(a, b, dh)`** returns `(da, db)`. Sweep from the end with `g[t] = dh[t] + a[t+1] g[t+1]`. The usual mistake is `h[t]` where `da` needs `h[t-1]`.
10. **`selective_ssm(x, a, b, c)`** runs `parallel_scan` on `a` and `b * x`, then sums `c * h` over the state axis.

## Where this breaks

- **Recall.** A fixed state forgets. Copying an exact string from far back, or looking up a fact by its key, is harder for it than for attention, which can point at any cached token. Many production models mix attention layers with state-space layers for this reason.
- **A weak feature map.** `elu(x) + 1` cannot make one weight dominate the way a softmax does, so linear attention is often less sharp.
- **Numerical drift.** A product of 1,000 decays near 0.9 is below `1e-45`, which underflows float32, so fast kernels work in logs or in chunks. A scan and a loop also add in different orders, so they agree to rounding error, and the checks use a tolerance.

## The exercise

You implement the ten functions above. The checks cover the feature map, the worked example `[2, 3, 5]` with no leak from the future, the recurrent form against the quadratic form to `1e-10`, the kernel, the recurrence against the convolution, the zero-order hold against a fine simulation of the differential equation, the composition order and associativity of `scan_combine`, the scan against the loop for `T` of 1, 8, and 13, the backward pass against finite differences, and a selective model against a plain loop.

## Further reading

- Katharopoulos, Vyas, Pappas, and Fleuret, [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (2020). The feature map, the running sum, and the equivalence with the quadratic form.
- Gu, Goel, and Ré, [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) (2021). The S4 model, and the convolution view of a state-space layer.
- Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (2023). Input-dependent `Δ`, `B`, and `C`, and training them with a scan.
- Dao and Gu, [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (2024). The link between state-space models and masked attention.
- Blelloch, Prefix sums and their applications, Carnegie Mellon technical report CMU-CS-90-190 (1990). The up-sweep and down-sweep scan.
- Stanford CS336, [Spring 2026 schedule](http://cs336.stanford.edu/). Lecture 4, "Attention alternatives and mixture of experts".
