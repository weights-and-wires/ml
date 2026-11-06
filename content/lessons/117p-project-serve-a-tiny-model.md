---
title: "Serve a small language model"
slug: 117p-project-serve-a-tiny-model
part: "Part 7 — Large language models"
kind: project
summary: Ship a tiny transformer with a tokeniser, KV cache, sampling, speculative decoding, and a LoRA adapter, proving each stage against its slow twin.
minutes: 200
gpu: false
objectives:
  - Train and use a byte-pair tokeniser whose decode inverts encode exactly, and measure what it costs on text it never saw.
  - Generate with a KV cache and prove it matches full recomputation, token for token.
  - Sample with temperature, top-k, and top-p, and use speculative decoding without changing the output distribution.
  - Serve a customer's LoRA adapter on the fly and merged, with identical outputs both ways.
prereqs:
  - 100-byte-pair-encoding
  - 101-fertility-and-fairness
  - 105-decoding-and-the-kv-cache
  - 106-temperature-and-sampling
  - 107-speculative-decoding
  - 109-lora
exercise: 117p-project-serve-a-tiny-model
---

# Serve a small language model

A colleague hands you a model and says, "Ship it by Friday." It is a small decoder-only transformer, and its weights are finished: nobody will train it again. A shop-and-delivery company wants to send it text and read text back. They also have a customer, a regional courier, who owns a LoRA adapter that changes how the model behaves, and the courier's requests must run with that adapter on. Your job is everything around the weights.

The obvious way to serve the model is to call its forward pass on the whole text, once per token, and to take the highest score every time. It works, and it is slow, repetitive, and wasteful. Every token recomputes every earlier token. Greedy choices write the same text for the same prompt forever. The tokeniser is an afterthought until a customer sends German or Hindi and pays five to eleven times as much for the same sentence. The adapter costs extra multiplications on every token, in every layer, for as long as the service runs.

Each of these problems has a fix from this part, and each fix has a way to go quietly wrong. A cache with an off-by-one position gives fluent, slightly different text. A speculative decoder with a wrong replacement rule gives text that looks fine and follows the wrong distribution. A merge with the wrong scale gives a model that is subtly someone else's. So in this project every stage ships with a proof: you compare it against the slow, obvious version, and the numbers must agree.

## The brief

You deliver a serving pipeline with these pieces:

- A byte-pair tokeniser trained on a small corpus. `decode(encode(s))` returns `s` exactly on strings it never saw, and you report its fertility inside and outside the training domain.
- Incremental decoding with a KV cache. Its logits match the full recompute to within `1e-9`, and cached generation matches uncached generation token for token.
- A sampler with temperature, top-k, and top-p, driven by a seeded generator.
- Speculative decoding with a smaller draft model. Its output follows the target's distribution, and it reports the accepted-token rate.
- The customer's LoRA adapter, applied on the fly and merged into the weights, with identical logits both ways.

A good answer clears four end-to-end tests. Cached and uncached greedy generation are the same list of ids. Over thousands of seeded runs, speculative sampling gives next-token frequencies within 0.05 total variation of the target's (0.06 on the real model). Merged and unmerged adapters agree to `1e-8` in logits and exactly in greedy text. And every unseen string survives the round trip through the tokeniser.

## What you will use

- [Byte-pair encoding](/lessons/100-byte-pair-encoding) gives you the loop that learns merges from pair counts, with ties broken by the smaller pair, the replay of merges in learned order, and the byte-level decode that makes the round trip exact.
- [Fertility and fairness](/lessons/101-fertility-and-fairness) gives you tokens per word as the price a language pays.
- [The KV cache](/lessons/105-decoding-and-the-kv-cache) gives you the stored keys and values that make one step cost one token.
- [Temperature and sampling](/lessons/106-temperature-and-sampling) gives you the sampler: temperature, top-k, top-p, and a seeded draw.
- [Speculative decoding](/lessons/107-speculative-decoding) gives you the accept-or-replace rule that keeps the target's distribution.
- [LoRA and merging](/lessons/109-lora) give you the low-rank update, on the fly and folded into the weight.

## Meet the data

Nothing is downloaded, and `make_data(seed)` builds everything in the starter. It returns four things.

- `corpus` is 500 short sentences, about 24,000 characters, from a small shop-and-delivery grammar such as "the driver shipped the parcel on monday." About a third end with an order number like "under order 4821". Words repeat heavily, which is the pattern your tokeniser learns from.
- `params` is the target model: two layers, two attention heads, width 32, and a vocabulary of 400 ids (256 bytes plus 144 merges). Its weights are random and fixed by the seed. It is given, like a checkpoint a colleague hands over, and you never train it.
- `draft` is the same model with only its first layer. It shares the vocabulary and the embeddings, so its guesses often agree with the target's.
- `adapter` is the courier's LoRA: rank 4, `alpha` 8, on `Wq` and `Wv` in both layers, with a nonzero `B`, so it changes the logits visibly.

Because the weights are random, the model's text is not fluent, and it is not meant to be. Every check compares your pipeline with the model's own numbers, so a correct pipeline is one that reproduces what the slow path computes. The planted pattern in the corpus is its vocabulary: ` the`, ` driver`, ` returned`, ` refund`, and the digits of order numbers. Figure 1 shows what that does to the cost of text. The in-domain sentence is cheap, a number-and-date string is dearer, and text in a language the tokeniser never saw falls back to one id per byte.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="Bar chart of tokens per word: 1.14 for in-domain English, 4.75 for an order number and date, 5.50 for German, 12.80 for Hindi">
<text x="10" y="22" class="fig-bold">Tokens per word after training on the shop corpus</text>
<line x1="260" y1="40" x2="260" y2="210" class="stroke-muted"/>
<line x1="364" y1="40" x2="364" y2="210" class="stroke-grid"/>
<line x1="468" y1="40" x2="468" y2="210" class="stroke-grid"/>
<line x1="572" y1="40" x2="572" y2="210" class="stroke-grid"/>
<text x="364" y="228" text-anchor="middle" class="fig-small">4</text>
<text x="468" y="228" text-anchor="middle" class="fig-small">8</text>
<text x="572" y="228" text-anchor="middle" class="fig-small">12</text>
<text x="250" y="68" text-anchor="end">in-domain English</text>
<rect x="260" y="50" width="30" height="26" class="fill-c"/>
<text x="298" y="68" class="fig-small fig-bold">1.14</text>
<text x="250" y="108" text-anchor="end">order number and date</text>
<rect x="260" y="90" width="123" height="26" class="fill-a"/>
<text x="391" y="108" class="fig-small fig-bold">4.75</text>
<text x="250" y="148" text-anchor="end">German</text>
<rect x="260" y="130" width="143" height="26" class="fill-a"/>
<text x="411" y="148" class="fig-small fig-bold">5.50</text>
<text x="250" y="188" text-anchor="end">Hindi</text>
<rect x="260" y="170" width="333" height="26" class="fill-bad"/>
<text x="601" y="188" class="fig-small fig-bold">12.80</text>
</svg>
Figure 1. The corpus-trained tokeniser spends 1.14 tokens per word on text like its training data. A string of numbers costs 4.75, German 5.50, and Hindi 12.80, which is one id for every byte of the script.
```

## Plan the build

Figure 2 shows the whole service. Text goes in on the left and text comes out on the right, with the transformer in the middle. The stages run in the order a request flows, and each one is checked before the next relies on it.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Pipeline: text, encode, transformer with KV cache, sampler, decode back to text; a LoRA adapter feeds the transformer and a draft model feeds the sampler's speculative check; the sampled token loops back in">
<defs><marker id="p132p-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="10" y="80" width="80" height="48" rx="3" class="fill-soft-a stroke-a"/>
<text x="50" y="108" text-anchor="middle">text in</text>
<rect x="112" y="80" width="90" height="48" rx="3" class="fill-soft-a stroke-a"/>
<text x="157" y="101" text-anchor="middle">encode</text>
<text x="157" y="118" text-anchor="middle" class="fig-small">BPE, step 1 to 4</text>
<rect x="224" y="80" width="140" height="48" rx="3" class="fill-soft-b stroke-b"/>
<text x="294" y="101" text-anchor="middle">transformer</text>
<text x="294" y="118" text-anchor="middle" class="fig-small">KV cache, step 5</text>
<rect x="386" y="80" width="110" height="48" rx="3" class="fill-soft-b stroke-b"/>
<text x="441" y="101" text-anchor="middle">sample</text>
<text x="441" y="118" text-anchor="middle" class="fig-small">step 6 to 9</text>
<rect x="518" y="80" width="112" height="48" rx="3" class="fill-soft-a stroke-a"/>
<text x="574" y="101" text-anchor="middle">decode</text>
<text x="574" y="118" text-anchor="middle" class="fig-small">text out, step 3</text>
<line x1="90" y1="104" x2="108" y2="104" class="stroke-muted" marker-end="url(#p132p-arrow)"/>
<line x1="202" y1="104" x2="220" y2="104" class="stroke-muted" marker-end="url(#p132p-arrow)"/>
<line x1="364" y1="104" x2="382" y2="104" class="stroke-muted" marker-end="url(#p132p-arrow)"/>
<line x1="496" y1="104" x2="514" y2="104" class="stroke-muted" marker-end="url(#p132p-arrow)"/>
<text x="373" y="72" text-anchor="middle" class="fig-small">logits</text>
<path d="M441,76 L441,36 L294,36 L294,76" fill="none" class="stroke-muted dashed" marker-end="url(#p132p-arrow)"/>
<text x="367" y="28" text-anchor="middle" class="fig-small">the chosen token is fed back in</text>
<rect x="224" y="196" width="140" height="48" rx="3" class="fill-soft-d stroke-d"/>
<text x="294" y="217" text-anchor="middle">LoRA adapter</text>
<text x="294" y="234" text-anchor="middle" class="fig-small">on the fly or merged, 10 to 11</text>
<line x1="294" y1="196" x2="294" y2="132" class="stroke-d" marker-end="url(#p132p-arrow)"/>
<rect x="386" y="196" width="244" height="48" rx="3" class="fill-soft-c stroke-c"/>
<text x="508" y="217" text-anchor="middle">draft model proposes, target checks</text>
<text x="508" y="234" text-anchor="middle" class="fig-small">speculative decoding, steps 7 to 9</text>
<line x1="441" y1="196" x2="441" y2="132" class="stroke-c" marker-end="url(#p132p-arrow)"/>
<text x="10" y="280" class="fig-small">Step 12, complete, wires every box together. Each box has a check that compares it with a slower version.</text>
</svg>
Figure 2. A request enters as text, becomes ids, runs through the cached transformer, and leaves as a sampled token that goes back in. The adapter plugs into the transformer's weights, and the draft model plugs into the sampler.
```

The order matters because each stage is the reference for the next. The tokeniser comes first, since every other test needs real prompts. The cache comes next, and its yardstick is the given `forward`. Sampling comes before speculation, because speculation reuses the sampler's filtered probabilities as its `p` and `q`. LoRA comes last because it only needs a working transformer to compare against, and `complete` ties everything into one call.

## Build it step by step

> [!RECAP]
> Before you build:
> - A merge list is an ordered recipe. Ids 0 to 255 are bytes, merge `i` creates id `256 + i`, and ties in pair counts go to the smaller pair.
> - A cached step must equal the last row of the full forward pass. The next token's position is the cache length before you append.
> - A rejected draft token is replaced from `max(p - q, 0)` normalised, never from `p`. The rate you report is accepted over checked.
> - LoRA computes `x @ W.T + (alpha / r) * (x @ A.T) @ B.T`, and the merged weight is `W + (alpha / r) * B @ A`. Merging returns a copy.

The starter gives you `make_data`, the transformer `forward`, `KVCache`, `new_cache`, `plain_linear`, `get_pair_counts`, `merge_pair`, and `make_prob_fn`. You write the twelve steps below, in this order.

1. **`train_bpe(corpus, vocab_size)`.** It pre-tokenises with `_pretokenize`, turns each piece into byte values, and merges the most frequent pair until the vocabulary is full. It returns `merges` and a `vocab` from every id to its bytes. This is [training BPE](/lessons/100-byte-pair-encoding) with one change: the corpus is a few thousand pieces, so a plain loop is fast enough. The usual mistake is counting pairs across a word boundary, which glues the end of one word to the start of the next. On `"aaabdaaabac"` with `vocab_size=259`, expect merges `[(97, 97), (97, 98), (256, 257)]`. Counts tie between `(97, 98)` and `(256, 97)` after the first merge, and the smaller pair wins.
2. **`encode(text, merges)`.** For each piece, start from its UTF-8 bytes and replay merge 0, then merge 1, and so on, with `merge_pair`. Join the pieces' ids. This is [encoding and decoding](/lessons/100-byte-pair-encoding). The usual mistake is replaying the merges in the wrong order, or checking every merge once per piece instead of once per merge in sequence. `encode("", merges)` is `[]`.
3. **`decode(ids, vocab)`.** Join `vocab[i]` for each id and decode the bytes as UTF-8 with `errors="replace"`. The replacement matters here and not in the chapter, because a model can write any ids in any order, and a lone continuation byte would crash a strict decode in the middle of a response.
4. **`fertility(text, merges)`.** Tokens per whitespace-separated word, from [fertility and fairness](/lessons/101-fertility-and-fairness). It is one line, and the check wants it to be higher outside the training domain, as in Figure 1.
5. **`decode_step(params, token, cache, linear)`.** Run one token through the transformer, reading everything earlier from the cache. Take the position from `cache.length` before you append. For each layer, compute `q`, `k`, and `v` for the one token, reshape each to `(n_head, 16)`, append `k` and `v`, fetch the whole cache, and attend. No mask is needed, because the cache holds only the past and the present. Every weight goes through `linear(x, W, (layer, name))`, because step 11 swaps that hook. This is [the KV cache](/lessons/105-decoding-and-the-kv-cache) with several heads. The usual mistake is a position embedding of 0 for every token, which gives plausible but wrong logits.
6. **`next_token_probs(logits, temperature, top_k, top_p)`.** Divide by the temperature, softmax with the maximum subtracted, keep the top `k`, keep the nucleus up to and including the token that crosses `top_p`, and renormalise. Temperature 0 returns a one-hot vector at the argmax. This is [temperature and sampling](/lessons/106-temperature-and-sampling). Speculative decoding needs the probabilities themselves, so write this function and let step 7 build on it.
7. **`sample_token(logits, rng, ...)` and `generate(...)`.** `sample_token` returns the argmax at temperature 0 and otherwise draws with `rng.choice(len(probs), p=probs)`. `generate` runs the prompt through `decode_step`, then repeats: sample, append, feed the token back. With `use_cache=False` it calls `forward` on the whole sequence each step. The usual mistake is feeding the prompt's last token twice, or never feeding the sampled token back, which freezes the output on one token.
8. **`residual_distribution(p, q)` and `speculative_round(target_fn, draft_fn, context, k, rng)`.** The residual is `max(p - q, 0)` divided by its sum, with a copy of `p` when the sum is 0. In a round, the draft proposes `k` tokens and remembers each row as `q`. The target is called once on `context + drafted`, and its row `len(context) - 1 + i` is `p` for draft token `i`. Accept token `x` when `rng.uniform() < min(1, p[x] / q[x])`. On the first rejection, draw from the residual and stop. If all `k` pass, draw a bonus token from the last row. Return `(tokens, n_accepted, n_checked)`. The usual mistake is replacing from `p`, which looks right on one example and fails over thousands. This is [speculative decoding](/lessons/107-speculative-decoding).
9. **`speculative_generate(target_fn, draft_fn, prompt, n_new, k, rng)`.** Call `speculative_round` until `n_new` tokens exist, cut to exactly `n_new`, and return `(tokens, stats)` with the keys `accepted`, `checked`, `target_passes`, and `acceptance_rate`. Divide accepted by checked, and count a checked token whether or not it passed. The usual mistake is dividing by `k` per round, which counts draft tokens that were thrown away after an earlier rejection and understates the rate.
10. **`lora_linear(x, W, A, B, alpha, r)` and `merge_lora(W, A, B, alpha, r)`.** The first computes `x @ W.T + (alpha / r) * (x @ A.T) @ B.T` without ever forming the update. The second returns `W + (alpha / r) * B @ A`. These are [the LoRA chapter](/lessons/109-lora). The usual mistake is scaling by `alpha` alone. The chapter's 2 by 2 example gives `[[7, 1]]` and `[[3, 4], [0, 1]]`.
11. **`make_lora_linear(adapter)` and `merge_adapter(params, adapter)`.** The first returns a hook that applies the adapter for keys it holds and a plain `x @ W.T` for the rest. The second deep-copies `params` and replaces each adapted weight with its merge. The usual mistake is writing into `params` in place, which silently adapts the base model for every later request.
12. **`complete(prompt, merges, vocab, params, n_new, rng, ...)`.** Encode, pick the merged or hooked model, generate with the cache, and decode the prompt plus the written ids. The prompt text comes back unchanged at the front.

> [!TIP]
> Before you trust a stage, run it against the slow path it replaces. Step 5 against `forward(params, ids)[-1]`, and step 11 against step 10, are each one line, and they catch almost every bug in this project.

## Read your results

Run the file. With the reference solution, the output has these numbers, and yours should land close to them.

**The tokeniser.** The 39-byte sentence "the driver returned a refund on monday." encodes to 8 tokens, and decodes back exactly. Over the whole corpus, 24,488 bytes become 4,774 tokens, a fertility of 1.14 tokens per word. The German sentence costs 5.50 tokens per word and the Hindi one 12.80, as in Figure 1. If your in-domain number is above 2, your merges probably cross piece boundaries or ignore the tie rule. If round trips fail only on emoji or Hindi, you are probably decoding ids as single characters instead of joining bytes first.

**The cache.** The largest gap between cached logits and the full forward pass is about `3e-15`, which is rounding. A gap of `1e-3` or larger points to the wrong position or a wrong head reshape. The cache also changes the cost. For an 8-token prompt and 40 written tokens, uncached generation pushes 1,100 token positions through the model, since every step recomputes the whole sequence, while the cached version pushes 48. That is 23 times fewer, and the saving grows with length.

**The sampler.** On the logits after "the driver returned a refund", the top probability is 0.69 at temperature 0.5, 0.17 at temperature 1, and 0.03 at temperature 2. The entropy rises from 1.56 to 4.45 to 5.70 nats, so the distribution flattens. At temperature 1, `top_k=40` keeps exactly 40 tokens, and `top_p=0.9` keeps 174 of the 400, which is how much of the tail the nucleus needs on a model this uncertain.

**The adapter.** The adapter has 1,024 numbers, against 4,096 in the four matrices it adapts and 16,384 in the whole model. On the prompt above, it moves the largest logit by about 2.6, so the customer's model is visibly different from the base. Merged and on-the-fly logits agree to about `3e-15`, and the greedy completions are the same string. A gap of `1e-2` or larger usually means the scale is `alpha` instead of `alpha / r`. A difference in text but not in logits means the generator was seeded differently.

**Speculative decoding.** With top-5 filtering on both models, the accepted-token rate is 0.76 at both `k = 2` and `k = 4`, and the target is called once per 2.33 and 3.03 tokens. Figure 3 sweeps `k`. The prediction `(1 - a^(k+1)) / (1 - a)` for `k = 4` at `a = 0.76` is 3.10, close to the 3.03 you measure, because acceptance is not exactly independent across positions. The last bar is a draft with unrelated weights: it is accepted 1 time in 100, and you get 1.01 tokens per pass. That is the point of the rule. A bad draft costs speed, and the output is still the target's.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Bar chart of tokens per target pass: 1.72 for k equal to 1, 2.33 for 2, 2.86 for 3, 3.03 for 4, 3.45 for 6, and 1.01 for a bad draft at k equal to 4">
<text x="10" y="22" class="fig-bold">Tokens written per target pass, 100 tokens, top-5 filtering</text>
<line x1="60" y1="250" x2="620" y2="250" class="stroke-muted"/>
<line x1="60" y1="190" x2="620" y2="190" class="stroke-grid"/>
<line x1="60" y1="130" x2="620" y2="130" class="stroke-grid"/>
<line x1="60" y1="70" x2="620" y2="70" class="stroke-grid"/>
<text x="52" y="254" text-anchor="end" class="fig-small">0</text>
<text x="52" y="194" text-anchor="end" class="fig-small">1</text>
<text x="52" y="134" text-anchor="end" class="fig-small">2</text>
<text x="52" y="74" text-anchor="end" class="fig-small">3</text>
<rect x="80" y="147" width="60" height="103" class="fill-b"/>
<text x="110" y="141" text-anchor="middle" class="fig-small fig-bold">1.72</text>
<text x="110" y="270" text-anchor="middle" class="fig-small">k = 1</text>
<rect x="170" y="110" width="60" height="140" class="fill-b"/>
<text x="200" y="104" text-anchor="middle" class="fig-small fig-bold">2.33</text>
<text x="200" y="270" text-anchor="middle" class="fig-small">k = 2</text>
<rect x="260" y="78" width="60" height="172" class="fill-b"/>
<text x="290" y="72" text-anchor="middle" class="fig-small fig-bold">2.86</text>
<text x="290" y="270" text-anchor="middle" class="fig-small">k = 3</text>
<rect x="350" y="68" width="60" height="182" class="fill-b"/>
<text x="380" y="62" text-anchor="middle" class="fig-small fig-bold">3.03</text>
<text x="380" y="270" text-anchor="middle" class="fig-small">k = 4</text>
<rect x="440" y="43" width="60" height="207" class="fill-b"/>
<text x="470" y="37" text-anchor="middle" class="fig-small fig-bold">3.45</text>
<text x="470" y="270" text-anchor="middle" class="fig-small">k = 6</text>
<rect x="530" y="189" width="60" height="61" class="fill-bad"/>
<text x="560" y="183" text-anchor="middle" class="fig-small fig-bold">1.01</text>
<text x="560" y="270" text-anchor="middle" class="fig-small">bad draft</text>
<text x="560" y="288" text-anchor="middle" class="fig-small">k = 4</text>
</svg>
Figure 3. A draft that agrees with the target, with acceptance near 0.76, writes about three tokens per pass at k = 4, and the gain flattens as k grows. A draft with unrelated weights writes one token per pass, the same as no speculation.
```

> [!KEY]
> Every stage in this project has a slow twin: the full forward pass for the cache, the unmerged layer for the merge, and the target's own distribution for speculation. A fast path is correct when it reproduces its twin. Fluent text is not the test, and with a random model it could not be.

The speculative numbers depend on the order in which your code draws random numbers, so a correct solution can differ from these by about 0.1 tokens per pass and a few points of acceptance. The distribution check, not the exact rate, is what tells right from wrong.

## Take it further

1. Replace the loop in `generate` so that the draft model also uses a KV cache inside `speculative_round`. You will see that the cached draft makes each proposal cost one step, and that you must discard cache entries for rejected tokens.
2. Train the tokeniser on a corpus where German makes up a third of the text and measure the fertility of all four strings again. You will see how the vocabulary budget moves between languages.
3. Sweep the adapter's `alpha` from 0 to 16 and plot how far the logits move from the base model. You will see that the update is linear in `alpha`, which is why folding it into the weight is exact.
4. Add a per-request adapter store that merges on first use and caches the merged model, and measure memory against the on-the-fly hook. You will see where the break-even between many small adapters and few hot ones lies.
5. Replace top-5 filtering in the speculative check with no filtering and compare the accepted-token rate. You will see whether filtering raises or lowers agreement, and you will have to reason about which distribution speculation promises to preserve.

## The exercise

You write `train_bpe`, `encode`, `decode`, `fertility`, `decode_step`, `next_token_probs`, `sample_token`, `generate`, `residual_distribution`, `speculative_round`, `speculative_generate`, `lora_linear`, `merge_lora`, `make_lora_linear`, `merge_adapter`, and `complete`. The checks run from the first stage to the last. They verify the hand-worked merges, the learned `' the'` token, and that no token spans a piece boundary; `encode` against explicit merge lists; `decode(encode(s))` on unseen strings with emoji, Hindi, tabs, and double spaces, plus the no-raise decode of invalid bytes; and the exact fertility of a hand example, higher outside the domain.

They then compare `decode_step` with `forward` at every position of a prompt to `1e-9`, check the sampler against hand-computed temperature, top-k, and top-p distributions with seeded frequencies, and compare greedy and seeded sampled generation with and without the cache, token for token. They test the residual and the round on identical and hopeless drafts, and check speculative sampling by total variation over 6,000 seeded runs on a five-token vocabulary, plus an acceptance rate equal to the sum of `min(p, q)`. The LoRA checks compare the 2 by 2 example, merged against split on random layers, and merged against on-the-fly logits through the transformer. The last check serves the adapter merged, on the fly, and not at all, and runs speculative decoding on the real model against its top-3 distribution.
