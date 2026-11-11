---
title: "Encoders, decoders, and cross-attention"
slug: 065-encoders-and-cross-attention
part: "Part 6 — Transformers"
summary: Bidirectional and padding masks, BERT-style masked-token training, cross-attention from decoder to encoder, and T5 span corruption.
minutes: 75
gpu: false
objectives:
  - Tell the bidirectional, causal, and padding masks apart, and say which axis of the score table each one blocks.
  - Corrupt tokens with the 15 percent, 80/10/10 rule, and compute a loss only on the chosen positions.
  - Compute cross-attention by hand, with queries from the decoder and keys and values from the encoder, and show that padded encoder positions cannot change the output.
  - Turn a sentence into a span-corruption input and target with sentinel tokens.
  - Count how many positions give a training signal under causal, masked, and span-corruption objectives.
prereqs:
  - 059-scaled-attention-and-heads
  - 063-the-transformer-block
exercise: 065-encoders-and-cross-attention
---

# Encoders, decoders, and cross-attention

Read "The ___ sat on the mat" and you fill the blank from both sides. The words after the gap matter as much as the words before it. A model that reads only left to right, as the language model in the previous chapters does, can't use "sat on the mat" when it guesses the first word. For tasks that read a whole text and then decide something about it, such as tagging, search, or classification, that is a handicap.

Translation raises a second problem. The input is one sentence and the output is a different sentence, often of a different length. The model that writes the output has to look at the input while it writes, and the input isn't part of the output's own history.

Two questions follow. How do you train a model that sees both sides of a word without letting it copy the answer? And how does a decoder read a sequence that isn't its own? This chapter answers the first with masked-token training, as in BERT, and the second with cross-attention, as in the original transformer and in T5. It ends by counting how much each training recipe learns from one sequence. Figure 1 previews the masks that the chapter builds on.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three attention masks over five positions where the last position is padding: encoder self-attention allows every real key, decoder self-attention allows only keys at or before the query, and cross-attention lets three decoder queries read every real encoder key">
<text x="75" y="20" text-anchor="middle" class="fig-bold fig-small">encoder self-attention</text>
<rect x="20" y="40" width="20" height="20" class="fill-a"/>
<rect x="42" y="40" width="20" height="20" class="fill-a"/>
<rect x="64" y="40" width="20" height="20" class="fill-a"/>
<rect x="86" y="40" width="20" height="20" class="fill-a"/>
<rect x="108" y="40" width="20" height="20" class="fill-sunk"/>
<rect x="20" y="62" width="20" height="20" class="fill-a"/>
<rect x="42" y="62" width="20" height="20" class="fill-a"/>
<rect x="64" y="62" width="20" height="20" class="fill-a"/>
<rect x="86" y="62" width="20" height="20" class="fill-a"/>
<rect x="108" y="62" width="20" height="20" class="fill-sunk"/>
<rect x="20" y="84" width="20" height="20" class="fill-a"/>
<rect x="42" y="84" width="20" height="20" class="fill-a"/>
<rect x="64" y="84" width="20" height="20" class="fill-a"/>
<rect x="86" y="84" width="20" height="20" class="fill-a"/>
<rect x="108" y="84" width="20" height="20" class="fill-sunk"/>
<rect x="20" y="106" width="20" height="20" class="fill-a"/>
<rect x="42" y="106" width="20" height="20" class="fill-a"/>
<rect x="64" y="106" width="20" height="20" class="fill-a"/>
<rect x="86" y="106" width="20" height="20" class="fill-a"/>
<rect x="108" y="106" width="20" height="20" class="fill-sunk"/>
<rect x="20" y="128" width="20" height="20" class="fill-a"/>
<rect x="42" y="128" width="20" height="20" class="fill-a"/>
<rect x="64" y="128" width="20" height="20" class="fill-a"/>
<rect x="86" y="128" width="20" height="20" class="fill-a"/>
<rect x="108" y="128" width="20" height="20" class="fill-sunk"/>
<text x="117" y="164" text-anchor="middle" class="fig-small">pad</text>
<text x="75" y="182" text-anchor="middle" class="fig-small">columns are keys</text>
<text x="285" y="20" text-anchor="middle" class="fig-bold fig-small">decoder self-attention</text>
<rect x="230" y="40" width="20" height="20" class="fill-a"/>
<rect x="252" y="40" width="20" height="20" class="fill-sunk"/>
<rect x="274" y="40" width="20" height="20" class="fill-sunk"/>
<rect x="296" y="40" width="20" height="20" class="fill-sunk"/>
<rect x="318" y="40" width="20" height="20" class="fill-sunk"/>
<rect x="230" y="62" width="20" height="20" class="fill-a"/>
<rect x="252" y="62" width="20" height="20" class="fill-a"/>
<rect x="274" y="62" width="20" height="20" class="fill-sunk"/>
<rect x="296" y="62" width="20" height="20" class="fill-sunk"/>
<rect x="318" y="62" width="20" height="20" class="fill-sunk"/>
<rect x="230" y="84" width="20" height="20" class="fill-a"/>
<rect x="252" y="84" width="20" height="20" class="fill-a"/>
<rect x="274" y="84" width="20" height="20" class="fill-a"/>
<rect x="296" y="84" width="20" height="20" class="fill-sunk"/>
<rect x="318" y="84" width="20" height="20" class="fill-sunk"/>
<rect x="230" y="106" width="20" height="20" class="fill-a"/>
<rect x="252" y="106" width="20" height="20" class="fill-a"/>
<rect x="274" y="106" width="20" height="20" class="fill-a"/>
<rect x="296" y="106" width="20" height="20" class="fill-a"/>
<rect x="318" y="106" width="20" height="20" class="fill-sunk"/>
<rect x="230" y="128" width="20" height="20" class="fill-a"/>
<rect x="252" y="128" width="20" height="20" class="fill-a"/>
<rect x="274" y="128" width="20" height="20" class="fill-a"/>
<rect x="296" y="128" width="20" height="20" class="fill-a"/>
<rect x="318" y="128" width="20" height="20" class="fill-sunk"/>
<text x="327" y="164" text-anchor="middle" class="fig-small">pad</text>
<text x="285" y="182" text-anchor="middle" class="fig-small">columns are keys</text>
<text x="495" y="20" text-anchor="middle" class="fig-bold fig-small">cross-attention</text>
<rect x="440" y="40" width="20" height="20" class="fill-a"/>
<rect x="462" y="40" width="20" height="20" class="fill-a"/>
<rect x="484" y="40" width="20" height="20" class="fill-a"/>
<rect x="506" y="40" width="20" height="20" class="fill-a"/>
<rect x="528" y="40" width="20" height="20" class="fill-sunk"/>
<rect x="440" y="62" width="20" height="20" class="fill-a"/>
<rect x="462" y="62" width="20" height="20" class="fill-a"/>
<rect x="484" y="62" width="20" height="20" class="fill-a"/>
<rect x="506" y="62" width="20" height="20" class="fill-a"/>
<rect x="528" y="62" width="20" height="20" class="fill-sunk"/>
<rect x="440" y="84" width="20" height="20" class="fill-a"/>
<rect x="462" y="84" width="20" height="20" class="fill-a"/>
<rect x="484" y="84" width="20" height="20" class="fill-a"/>
<rect x="506" y="84" width="20" height="20" class="fill-a"/>
<rect x="528" y="84" width="20" height="20" class="fill-sunk"/>
<text x="537" y="120" text-anchor="middle" class="fig-small">pad</text>
<text x="495" y="182" text-anchor="middle" class="fig-small">columns are keys</text>
<text x="300" y="238" text-anchor="middle" class="fig-small">Blue: attention allowed. Grey: blocked. Rows are queries.</text>
</svg>
Figure 1. Three masks over a 5-token sequence whose last token is padding. Notice that the padding column is grey in all three, and that the decoder's own mask is a staircase on top of it.
```

## What you already know

You already have these ideas:

- Scaled attention scores each query against each key, divides by $\sqrt{d}$, adds a mask, and applies a softmax over the keys (see [scaled attention and heads](/lessons/059-scaled-attention-and-heads)).
- A causal mask blocks every key that sits after the query, so a language model can't read the token it must predict.
- The weights in each row of an attention table are non-negative and sum to 1, so each output is a blend of value vectors.
- Cross-entropy loss is $-\log p$ of the correct token, and its gradient with respect to the logits is the predicted probabilities minus a one-hot vector.
- A transformer block adds an attention sublayer and a feed-forward sublayer to a residual stream (see [the transformer block](/lessons/063-the-transformer-block)).

## Who may look at whom

Attention computes a table of scores with one row per query and one column per key. A **mask** decides which cells of that table may take weight. Blocked cells get a score of $-\infty$ before the softmax, so they end up with weight exactly 0.

Three masks cover the whole chapter.

- A **bidirectional mask** blocks nothing. Every position reads every other position, left and right. A stack of transformer blocks with this mask is an **encoder**.
- A **causal mask** blocks every key after the query. A stack with this mask is a **decoder**, and it is what a left-to-right language model uses.
- A **padding mask** blocks keys that aren't real tokens. Sequences in a batch have different lengths, so shorter ones are filled out with a filler token up to the longest length. Padding is meant to be invisible, so no query may read it.

Take a batch with two sequences of lengths 3 and 2, padded to 4. The padding mask marks real tokens with 1: `[1, 1, 1, 0]` and `[1, 1, 0, 0]`. For the first sequence, the encoder's table allows columns 0, 1, and 2 in every row. The decoder's table allows only column 0 in row 0, and columns 0, 1, and 2 in rows 2 and 3.

The padding mask goes on the key axis, which means on columns. This is the part people reverse. A padded query produces an output that nobody reads, so it does no harm. A padded key would get weight and leak a meaningless filler vector into every real token's output. If you block the rows instead, a padded query's row is entirely $-\infty$, the softmax divides 0 by 0, and the table fills with `nan`.

> [!KEY]
> A mask answers one question for each cell: may this query read this key? Bidirectional allows everything, causal allows keys at or before the query, and padding removes every key that isn't a real token. Padding always acts on columns.

Why does the language model need the causal mask, while an encoder can go without? Next-token training asks the model to predict token $t+1$ from tokens up to $t$. If position $t$ could read token $t+1$, the model would copy it and learn nothing. An encoder is trained on a different task, one where the answer has been taken out of the input, and the next section shows how.

## Masked-token training

Suppose you want an encoder to learn language without labels. You can't use next-token prediction, for the reason above. Instead, hide some of the tokens and ask the encoder to recover them from the rest. This is **masked language modelling**, the recipe behind BERT.

Here is the recipe in words. Walk over the positions. Choose each one with probability 0.15. Corrupt the chosen ones and leave the rest. Then run the corrupted sequence through the encoder, and compute a loss only at the chosen positions, comparing the prediction with the original token.

The word "corrupt" hides a detail. If every chosen token became a special `[MASK]` token, the model would only ever see `[MASK]` at the places where it must make a prediction. But the `[MASK]` token never appears in the text you later use the model on. So BERT splits the chosen positions three ways:

- 80 percent become the mask token.
- 10 percent become a random token drawn from the vocabulary.
- 10 percent stay as they are.

Read it as a defence against shortcuts. If chosen tokens always became `[MASK]`, the model could trust every visible token. With the random and unchanged cases, any visible token might be a chosen one, so the model has to build a good representation for every position and check each token against its context.

### The loss counts only the chosen positions

At a position that wasn't chosen, the input already shows the right token. Predicting it is trivial, so it carries no information about language. If you averaged the loss over every position, 85 percent of the terms would be near zero and would dilute the signal from the 15 percent that matter. So the loss averages over the chosen set $M$:

$$
L = -\frac{1}{|M|} \sum_{i \in M} \log p\left(x_i \mid \text{corrupted input}\right)
$$

Read it aloud: for each chosen position, take the log-probability the model gave to the original token, average those, and flip the sign. Here $|M|$ is the number of chosen positions and $x_i$ is the original token at position $i$.

Work three positions of a vocabulary of 3 by hand. The logits are rows of a table, and the correct tokens are 0, 0, and 2:

| Position | Logits | Probability of the correct token | Loss $-\log p$ | Chosen |
|---|---|---|---|---|
| 0 | [2.0, 0.5, −1.0] | 0.7856 | 0.2413 | yes |
| 1 | [0.0, 0.0, 4.0] | 0.0177 | 4.0360 | no |
| 2 | [1.0, 1.0, 1.0] | 0.3333 | 1.0986 | yes |

The masked loss is $(0.2413 + 1.0986)/2 = 0.6700$. Averaging over all three positions gives $1.7920$, which is bigger only because position 1, a position the loss doesn't count, is badly wrong.

> [!WARNING]
> Averaging over all positions and averaging over the chosen positions differ by more than a scale factor. The first one pushes gradient into positions that were never corrupted. The check in the exercise compares your loss with 0.6700, not 1.7920.

The gradient follows from the cross-entropy rule you know. At a chosen row it is the softmax of the logits minus a one-hot vector for the target, divided by $|M|$. At an unchosen row it is exactly 0. For position 0 above, the row is $[-0.1072, 0.0876, 0.0196]$, and position 1's row is all zeros.

In numpy, the whole corruption step is a few lines of array logic:

```python
chosen = rng.random(ids.shape) < rate
roll = rng.random(ids.shape)
corrupted = np.where(chosen & (roll < 0.8), mask_id, ids)
corrupted = np.where(chosen & (roll >= 0.8) & (roll < 0.9), random_ids, corrupted)
```

The `roll` array decides which of the three treatments each chosen position gets. The last 10 percent need no line, because they keep their original ids.

> [!KEY]
> Masked-token training hides the answer instead of hiding the future. That lets every position read both sides. The 80/10/10 split keeps the model from relying on `[MASK]`, and the loss counts only the chosen positions because the rest are already visible.

## Cross-attention: a decoder that reads another sequence

Masked-token training gives you an encoder, which turns a sequence into one vector per position. To translate or summarise, you pair it with a decoder that writes the output one token at a time. The encoder runs once over the source. The decoder runs over the target and, in every layer, reads the encoder's output.

In **self-attention**, queries, keys, and values all come from one sequence. In **cross-attention**, they come from two:

- The **queries** come from the decoder's own state, one per decoder position. Each asks, "which parts of the source matter for the token I'm writing?"
- The **keys** and **values** come from the encoder's output, one per source position.

If $H$ is the encoder output with shape $(T_{\text{enc}}, d)$ and $S$ is the decoder state with shape $(T_{\text{dec}}, d)$, then:

$$
Q = S W_q, \qquad K = H W_k, \qquad V = H W_v, \qquad
\operatorname{out} = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{d}} + \text{mask}\right) V
$$

Read it aloud: each decoder position scores every source position, turns the scores into weights, and blends the source values by those weights. The score table has shape $(T_{\text{dec}}, T_{\text{enc}})$. It's rectangular, because the two sequences have different lengths. This is the alignment matrix from [attention gradients and the alignment matrix](/lessons/057-attention-gradients-and-alignment), computed with scaled dot products and many heads.

Two things change from the decoder's self-attention. There is no causal mask, because the decoder may read the whole source, which is finished before generation begins. And the only mask is the padding mask on the encoder's positions, which are the keys.

Work one query by hand. Let $d = 2$ and take a decoder query $q = [1, 0]$. The encoder has three positions, and the third is padding:

| Source position | Key | Value | Real? |
|---|---|---|---|
| 0 | [1, 0] | [10, 0] | yes |
| 1 | [0, 1] | [0, 10] | yes |
| 2 | [1, 1] | [5, 5] | no, padding |

The scaled scores are $q \cdot k / \sqrt{2}$, which are $0.7071$, $0$, and $0.7071$. Block position 2. The softmax over the first two scores uses $e^{0.7071} = 2.0281$ and $e^{0} = 1$, so the weights are $2.0281/3.0281 = 0.6698$ and $1/3.0281 = 0.3302$. The output is $0.6698 \times [10, 0] + 0.3302 \times [0, 10] = [6.698, 3.302]$.

Without the mask, position 2 would score as high as position 0, the weights would be $[0.4011, 0.1978, 0.4011]$, and the output would be $[6.017, 3.983]$. The padding token's filler vector would have pulled the answer away by about 0.7 in each coordinate.

The padding mask has a testable promise: whatever sits in a padded position, the output is the same. Replace the padded key and value with huge random numbers and the output must not move. The exercise runs that experiment.

```python
scores = q_dec @ k_enc.T / np.sqrt(d)
weights = masked_softmax(scores, pad_mask[None, :])   # pad_mask has one entry per encoder position
out = weights @ v_enc
```

> [!TIP]
> To test any mask, change the data that the mask hides and check that the output stays the same. Replace the padded keys and values with large random numbers. If the output moves, the mask leaks. This test finds mistakes that a hand check of one example can miss.

> [!WARNING]
> Zeroing the weights of padded keys after the softmax is a different operation from blocking them before it. The remaining weights no longer sum to 1, so the output shrinks. Put $-\infty$ in the scores, then take the softmax.

A full encoder-decoder layer stacks three sublayers on the residual stream: causal self-attention over the target so far, cross-attention to the encoder output, and the feed-forward network. Because the source doesn't change while the decoder writes, a generating decoder can compute the keys and values $K$ and $V$ for the source once and reuse them at every step.

## T5 span corruption

An encoder-decoder model has a decoder that writes a whole sequence, so it can recover more than single tokens. T5 uses that with **span corruption**.

Pick spans of consecutive tokens. Replace each span in the input with a single **sentinel** token, a special id that marks a gap, numbered in order. The target is the list of dropped spans, each introduced by its sentinel, with one extra sentinel at the end to mark that the list is over.

Take the sentence "Thank you for inviting me to your party last week" and drop the span "for inviting" and the span "last":

- Input: `Thank you <X> me to your party <Y> week`
- Target: `<X> for inviting <Y> last <Z>`

The input has 9 tokens, and the target has 6. The sentinels are named in order, so `<X>` in the target tells the decoder which gap to fill with the words that follow it. A sentinel is a one-token stand-in for a gap of any size, which is why the input gets shorter, not the same length.

The encoder reads about 85 percent of the text plus a few sentinels, and the decoder writes only the missing part, so the target is short. The T5 paper used a corruption rate of 15 percent and a mean span length of 3, and the exercise does as well.

The lengths follow from three numbers. With $T$ tokens, a noise density of 0.15, and a mean span length of 3:

$$
\text{noise} = \operatorname{round}(0.15\,T), \qquad
\text{spans} = \operatorname{round}(\text{noise}/3), \qquad
\text{input} = T - \text{noise} + \text{spans}, \qquad
\text{target} = \text{noise} + \text{spans} + 1
$$

Read it aloud: the input loses the corrupted tokens but gains one sentinel per span, and the target holds the corrupted tokens, one sentinel per span, and the closing sentinel. For $T = 512$ the noise is 77 tokens in 26 spans, so the input has $512 - 77 + 26 = 461$ tokens and the target has $77 + 26 + 1 = 104$.

## Three objectives, counted

You can compare the three recipes by how much of one sequence the loss touches. Figure 2 counts the targets in a 512-token sequence.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Bars for the number of target positions per 512-token sequence: causal language model 511, masked language model 77, span corruption 104">
<text x="140" y="42" text-anchor="end" class="fig-bold fig-small">causal LM</text>
<rect x="150" y="24" width="339.3" height="26" rx="3" class="fill-a"/>
<text x="497.3" y="42" class="fig-small">511 targets</text>
<text x="140" y="90" text-anchor="end" class="fig-bold fig-small">masked LM</text>
<rect x="150" y="72" width="51.1" height="26" rx="3" class="fill-b"/>
<text x="209.1" y="90" class="fig-small">77 targets</text>
<text x="140" y="138" text-anchor="end" class="fig-bold fig-small">span corruption</text>
<rect x="150" y="120" width="69.1" height="26" rx="3" class="fill-c"/>
<text x="227.1" y="138" class="fig-small">104 targets</text>
<text x="300" y="178" text-anchor="middle" class="fig-small">Targets the loss is computed on, per sequence of 512 tokens.</text>
</svg>
Figure 2. Positions that carry a loss in one 512-token sequence. A causal language model learns from almost every position, while masked-token training learns from 15 percent.
```

- **Causal language modelling** predicts token $t+1$ at every position, so a sequence of 512 gives 511 targets. Each prediction sees only the left context.
- **Masked-token training** gives about $0.15 \times 512 = 77$ targets. Each prediction sees both sides, but the sequence offers about a seventh of the signal per pass.
- **Span corruption** gives a target of 104 tokens. Of those, 77 are real corrupted tokens, and 27 are sentinels that mark structure.

The trade is between the quality of the context and the number of lessons per sequence. Bidirectional context makes each prediction easier and each representation richer, but only a fraction of the positions teach anything. How much that costs in total training is an empirical question that depends on the model and the data, and this chapter doesn't claim a number. The fact you can verify is the count.

The recipes lead to different tools. A causal model generates text. A masked-token encoder produces a vector for every position that reads both sides, which suits classification and search. An encoder-decoder reads the input in full and generates conditioned on it. Whisper, a speech recognition model, uses this layout with an audio encoder and a text decoder.

> [!KEY]
> The three objectives differ in what the model may see and in what it is graded on. Causal training grades 511 of 512 positions and reads left only. Masked-token training grades 77 and reads both sides. Span corruption grades 104 target tokens, reads both sides in the encoder, and writes the answer with a causal decoder.

Fine-tuning any of these models on labelled data, including how to mask the loss on a prompt, belongs to the reinforcement learning course. See [supervised fine-tuning and the post-training pipeline](https://rl.ifkash.dev/lessons/050-sft-and-the-post-training-pipeline).

> [!RECAP]
> - A mask decides which cells of the score table may take weight. Bidirectional blocks nothing, causal blocks later keys, and padding blocks non-real keys, on the column axis.
> - Masked-token training chooses 15 percent of positions, turns them into the mask token (80 percent), a random token (10 percent), or leaves them (10 percent), and averages the loss over the chosen positions only.
> - Cross-attention takes queries from the decoder and keys and values from the encoder. The table is rectangular, the only mask is the padding mask on keys, and padded positions can't change the output.
> - Span corruption swaps each dropped span for a numbered sentinel in the input, and the target lists each sentinel and its span, with one closing sentinel.
> - Per 512-token sequence, causal training grades 511 positions, masked-token training about 77, and span corruption a target of 104 tokens.

## Build it step by step

You write eight functions in the order the checks test them. Masks are boolean arrays, where `True` means the query may read the key.

1. **`padding_mask(lengths, T)`** takes the real lengths and the padded length, and returns a `(B, T)` boolean array. Compare `np.arange(T)[None, :]` with `lengths[:, None]`. The common mistake is an off-by-one that includes position `lengths[b]`.
2. **`attention_allowed(pad, causal)`** takes the `(B, T)` mask and returns `(B, T, T)`, indexed `[batch, query, key]`. Broadcast `pad[:, None, :]` over the query axis, and when `causal` is true, combine it with `np.tril(np.ones((T, T), dtype=bool))`. The common mistake is `pad[:, :, None]`, which blocks the queries.
3. **`masked_softmax(scores, allowed)`** returns weights of the same shape. Use `np.where(allowed, scores, -np.inf)`, subtract the row maximum, exponentiate, and divide by the row sum. The common mistake is to subtract the maximum of all the scores, including the blocked ones, which can underflow every allowed entry.
4. **`cross_attention(q_dec, k_enc, v_enc, pad_mask)`** returns `(output, weights)` with shapes `(T_dec, d_v)` and `(T_dec, T_enc)`. Divide by `np.sqrt(d)` with `d = q_dec.shape[-1]`, broadcast `pad_mask[None, :]` to the score shape, and call your `masked_softmax`. The common mistake is to zero the padded weights after the softmax.
5. **`mask_tokens(ids, rng, mask_id, vocab, rate=0.15)`** returns `(corrupted, chosen)`. Draw `chosen`, `roll`, and `random_ids` from the same `rng`, and combine them with `np.where`. The common mistake is to turn every chosen token into `mask_id`, which the proportion checks catch.
6. **`masked_lm_loss(logits, targets, chosen)`** returns `(loss, dlogits)`. Compute a stable log-softmax, average the negative log-probability of the target over the chosen positions, and build the gradient as softmax minus one-hot, scaled by `chosen / n_chosen`. Return `0.0` and zeros when nothing is chosen. The common mistake is to divide by `T`.
7. **`span_corrupt(ids, spans, sentinel_start)`** returns `(inputs, targets)` as integer arrays. Walk the spans in order and keep a pointer to the end of the previous one. Sentinel $k$ has id `sentinel_start + k`. The common mistake is to forget the closing sentinel.
8. **`span_corruption_lengths(T, noise_density, mean_span)`** returns `(input_length, target_length, n_spans)`. Use Python's `round` on the two products. The common mistake is to leave out the closing sentinel in the target length.

## Where this breaks

The `[MASK]` token appears only in training. The 10 percent random and 10 percent unchanged cases reduce that gap and don't remove it.

Masked-token training predicts each chosen token independently, given the corrupted input. When two neighbouring tokens are both hidden, the model can't use the first guess to help with the second. Span corruption writes the missing tokens in order, so its decoder can.

An encoder trained this way doesn't generate text. Use a causal model or an encoder-decoder when you need generation.

A mask row that is entirely blocked gives `nan`. A padding mask on keys never produces one unless a sequence has length 0.

## The exercise

You write `padding_mask`, `attention_allowed`, `masked_softmax`, `cross_attention`, `mask_tokens`, `masked_lm_loss`, `span_corrupt`, and `span_corruption_lengths`. The checks confirm the masks on a batch with lengths 3 and 2, and compare `masked_softmax` with a per-row loop, including scores of 1000 and 5000. They verify `cross_attention` against the worked example and a loop version, and confirm that replacing padded keys and values with large random numbers leaves the output unchanged. A seeded run of `mask_tokens` over 200,000 positions checks the 15 percent rate and the 80, 10, and 10 split within tight tolerances. They check `masked_lm_loss` against 0.6700 on the hand example, and its gradient against `finite_difference`. The last two checks confirm the input and target of the inviting-and-last example and the lengths 461 and 104 at $T = 512$.

## Further reading

The sources below are the ones this chapter follows:

- For the encoder-decoder transformer and cross-attention, see Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- For masked-token training and the 80/10/10 rule, see Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018).
- For span corruption with sentinel tokens, see Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (2019).
- For lecture treatments, see the Stanford CS224N Winter 2026 [schedule](https://web.stanford.edu/class/cs224n/), whose entries "Transformers" (20 January) and "Pretraining" (27 January) cover this material.
