---
title: "Train a tiny transformer to reverse digits"
slug: 050p-project-train-a-tiny-transformer
part: "Part 6 — Transformers"
summary: Build every piece of a small decoder-only transformer, train it from scratch, and prove it reverses digit strings it has never seen.
minutes: 150
gpu: false
kind: project
objectives:
  - Assemble masked multi-head attention, positions, pre-norm residual blocks, and a SwiGLU feed-forward into one working language model.
  - Score only the answer positions of a planted task, and train with warmup, cosine decay, clipping, and AdamW.
  - Judge the model with checks that catch a leaking mask, then count its weights and flops from formulas.
prereqs:
  - 045-scaled-attention-and-heads
  - 047-position-encodings
  - 048-the-transformer-block
  - 049-next-token-loss-and-safe-steps
  - 050-training-loop-and-cost
exercise: 050p-project-train-a-tiny-transformer
---

# Train a tiny transformer to reverse digits

A teammate asks a fair question: "Does the transformer you keep drawing on the whiteboard actually work, or does it only look good in diagrams?" You have spent a whole part building its pieces one at a time, each with a small test of its own. A pile of passing unit tests is not a model that learns, though. Wires cross when you assemble parts, and the usual fault is silent: the loss still falls, and the model is still wrong.

So you will build the whole thing and give it a job that only attention can do. The job is to reverse a string of six digits. To write the answer's first digit, the model must look back and find the input's last digit. A model with no attention can't do that, and a model whose mask lets it peek at the answer can fake it during training and fail when it has to write the answer unaided.

You will write every stage with torch tensors and autograd, train on a laptop CPU in a few seconds, and judge the result on strings the model has never seen.

## The brief

You deliver a working language model and the evidence that it works:

- A decoder-only transformer with two layers, four heads, and width 32, built from your own functions.
- A training run that uses a warmup-then-cosine learning rate, gradient clipping, and AdamW, and scores only the answer tokens.
- A greedy decoder and an exact-match score on held-out strings.
- A weight count and a flop count that come from formulas, not from adding up tensors after the fact.

A good answer meets four numbers. The average answer loss over the last 20 steps is below 0.05. Greedy decoding reproduces at least 95% of 300 held-out strings exactly. The output at any position doesn't change when you rewrite later tokens. The formula count equals the number of weights the initialiser creates.

## What you will use

- [Attention scores, and why we scale them](/lessons/045-scaled-attention-and-heads) for the query-key score and the `1/sqrt(d_head)` scale.
- [Heads and the causal mask](/lessons/045-scaled-attention-and-heads) for splitting the width into heads and blocking the future.
- [Position clocks](/lessons/047-position-encodings) for the sinusoidal table that tells each token where it stands.
- [The gated feed-forward](/lessons/048-the-transformer-block) for SwiGLU, the per-position network inside each block.
- [The residual block](/lessons/048-the-transformer-block) for the pre-norm layout, where each branch is normalised and added back to an untouched stream.
- [Next-token loss](/lessons/049-next-token-loss-and-safe-steps) for scoring a prediction at every position.
- [Safe steps](/lessons/049-next-token-loss-and-safe-steps) and [the training loop](/lessons/050-training-loop-and-cost) for the schedule, the clip, and AdamW.
- [The cost of a run](/lessons/050-training-loop-and-cost) for the `6 N D` flop estimate.

## Meet the data

`make_data(seed)` builds sequences of 13 tokens. The vocabulary has 11 tokens: the digits 0 to 9 are their own ids, and `>` has id 10. A sequence holds six random digits, then `>`, then the same six digits in reverse. For example, `865230>032568`.

The planted pattern is exact. Answer digit `j` (counting from 0 after the separator) equals input digit `5 - j`. Nothing else in the sequence carries information about the answer, so the only way to lower the loss is to learn that lookup. The function builds 3,000 training strings and 300 test strings, and it removes any repeat, so no test string appears in training. There are a million possible digit strings and the model sees 3,000, which means memorising can't pass the test.

```figure
<svg viewBox="0 0 560 190" role="img" aria-label="A 13-token sequence with arrows from each input digit to the answer position that copies it">
  <defs>
    <marker id="p108-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" class="fill-b"/>
    </marker>
  </defs>
  <path d="M243 108 Q285 76 327 107" fill="none" class="stroke-b" stroke-width="1.6" marker-end="url(#p108-arrow)"/>
  <path d="M201 108 Q285 44 369 107" fill="none" class="stroke-b" stroke-width="1.6" marker-end="url(#p108-arrow)"/>
  <path d="M159 108 Q285 12 411 107" fill="none" class="stroke-b" stroke-width="1.6" marker-end="url(#p108-arrow)"/>
  <path d="M117 108 Q285 -20 453 107" fill="none" class="stroke-b" stroke-width="1.6" marker-end="url(#p108-arrow)"/>
  <path d="M75 108 Q285 -52 495 107" fill="none" class="stroke-b" stroke-width="1.6" marker-end="url(#p108-arrow)"/>
  <path d="M33 108 Q285 -84 537 107" fill="none" class="stroke-b" stroke-width="1.6" marker-end="url(#p108-arrow)"/>
  <rect x="14" y="112" width="38" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="33" y="137" text-anchor="middle" class="fig-mono fig-bold">8</text>
  <text x="33" y="166" text-anchor="middle" class="fig-small">0</text>
  <rect x="56" y="112" width="38" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="75" y="137" text-anchor="middle" class="fig-mono fig-bold">6</text>
  <text x="75" y="166" text-anchor="middle" class="fig-small">1</text>
  <rect x="98" y="112" width="38" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="117" y="137" text-anchor="middle" class="fig-mono fig-bold">5</text>
  <text x="117" y="166" text-anchor="middle" class="fig-small">2</text>
  <rect x="140" y="112" width="38" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="159" y="137" text-anchor="middle" class="fig-mono fig-bold">2</text>
  <text x="159" y="166" text-anchor="middle" class="fig-small">3</text>
  <rect x="182" y="112" width="38" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="201" y="137" text-anchor="middle" class="fig-mono fig-bold">3</text>
  <text x="201" y="166" text-anchor="middle" class="fig-small">4</text>
  <rect x="224" y="112" width="38" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="243" y="137" text-anchor="middle" class="fig-mono fig-bold">0</text>
  <text x="243" y="166" text-anchor="middle" class="fig-small">5</text>
  <rect x="266" y="112" width="38" height="38" rx="4" class="fill-sunk stroke-muted"/>
  <text x="285" y="137" text-anchor="middle" class="fig-mono fig-bold">></text>
  <text x="285" y="166" text-anchor="middle" class="fig-small">6</text>
  <rect x="308" y="112" width="38" height="38" rx="4" class="fill-soft-b stroke-b"/>
  <text x="327" y="137" text-anchor="middle" class="fig-mono fig-bold">0</text>
  <text x="327" y="166" text-anchor="middle" class="fig-small">7</text>
  <rect x="350" y="112" width="38" height="38" rx="4" class="fill-soft-b stroke-b"/>
  <text x="369" y="137" text-anchor="middle" class="fig-mono fig-bold">3</text>
  <text x="369" y="166" text-anchor="middle" class="fig-small">8</text>
  <rect x="392" y="112" width="38" height="38" rx="4" class="fill-soft-b stroke-b"/>
  <text x="411" y="137" text-anchor="middle" class="fig-mono fig-bold">2</text>
  <text x="411" y="166" text-anchor="middle" class="fig-small">9</text>
  <rect x="434" y="112" width="38" height="38" rx="4" class="fill-soft-b stroke-b"/>
  <text x="453" y="137" text-anchor="middle" class="fig-mono fig-bold">5</text>
  <text x="453" y="166" text-anchor="middle" class="fig-small">10</text>
  <rect x="476" y="112" width="38" height="38" rx="4" class="fill-soft-b stroke-b"/>
  <text x="495" y="137" text-anchor="middle" class="fig-mono fig-bold">6</text>
  <text x="495" y="166" text-anchor="middle" class="fig-small">11</text>
  <rect x="518" y="112" width="38" height="38" rx="4" class="fill-soft-b stroke-b"/>
  <text x="537" y="137" text-anchor="middle" class="fig-mono fig-bold">8</text>
  <text x="537" y="166" text-anchor="middle" class="fig-small">12</text>
  <text x="14" y="186" class="fig-small">inputs (positions 0-5)</text>
  <text x="318" y="186" class="fig-small">answer, scored (positions 7-12)</text>
</svg>
Figure 1. One training row, `865230>032568`. Each arrow carries an input digit to the answer slot that repeats it, so the arrows nest like rainbows. The first answer digit copies the last input digit, and the last answer digit copies the first one, twelve positions earlier.
```

The training loop feeds the model the first 12 tokens and asks it to predict tokens 1 to 12. Targets at positions 0 to 4 are the next random input digit, which no model can predict, and the target at position 5 is the separator, which is free. Only the last six targets are answers. You score only those, so the loss measures the skill you care about rather than a floor of unlearnable noise.

## Plan the build

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Pipeline from data through embedding, blocks, and logits to the loss, the training step, and greedy decoding">
  <defs>
    <marker id="p108-flow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" class="fill-muted"/>
    </marker>
  </defs>
  <rect x="10" y="30" width="130" height="56" rx="6" class="fill-soft-a stroke-a"/>
  <text x="75" y="54" text-anchor="middle" class="fig-bold">make_data</text>
  <text x="75" y="72" text-anchor="middle" class="fig-small">digits &gt; reversed</text>
  <rect x="160" y="30" width="130" height="56" rx="6" class="fill-sunk stroke-muted"/>
  <text x="225" y="54" text-anchor="middle" class="fig-bold">embed + position</text>
  <text x="225" y="72" text-anchor="middle" class="fig-small">sinusoidal table</text>
  <rect x="310" y="30" width="130" height="56" rx="6" class="fill-soft-c stroke-c"/>
  <text x="375" y="54" text-anchor="middle" class="fig-bold">2 blocks</text>
  <text x="375" y="72" text-anchor="middle" class="fig-small">mask, attention, SwiGLU</text>
  <rect x="460" y="30" width="130" height="56" rx="6" class="fill-soft-b stroke-b"/>
  <text x="525" y="54" text-anchor="middle" class="fig-bold">norm + unembed</text>
  <text x="525" y="72" text-anchor="middle" class="fig-small">logits (B, T, 11)</text>
  <line x1="140" y1="58" x2="158" y2="58" class="stroke-muted" marker-end="url(#p108-flow)"/>
  <line x1="290" y1="58" x2="308" y2="58" class="stroke-muted" marker-end="url(#p108-flow)"/>
  <line x1="440" y1="58" x2="458" y2="58" class="stroke-muted" marker-end="url(#p108-flow)"/>
  <rect x="460" y="160" width="130" height="56" rx="6" class="fill-soft-b stroke-b"/>
  <text x="525" y="184" text-anchor="middle" class="fig-bold">answer_loss</text>
  <text x="525" y="202" text-anchor="middle" class="fig-small">answer slots only</text>
  <rect x="310" y="160" width="130" height="56" rx="6" class="fill-sunk stroke-muted"/>
  <text x="375" y="184" text-anchor="middle" class="fig-bold">train step</text>
  <text x="375" y="202" text-anchor="middle" class="fig-small">clip, schedule, AdamW</text>
  <rect x="10" y="160" width="290" height="56" rx="6" class="fill-soft-a stroke-a"/>
  <text x="155" y="184" text-anchor="middle" class="fig-bold">greedy_decode, exact_match</text>
  <text x="155" y="202" text-anchor="middle" class="fig-small">held-out strings, no peeking</text>
  <line x1="525" y1="86" x2="525" y2="158" class="stroke-muted" marker-end="url(#p108-flow)"/>
  <line x1="460" y1="188" x2="442" y2="188" class="stroke-muted" marker-end="url(#p108-flow)"/>
  <line x1="310" y1="188" x2="302" y2="188" class="stroke-muted" marker-end="url(#p108-flow)"/>
  <line x1="375" y1="158" x2="375" y2="88" class="stroke-muted dashed" marker-end="url(#p108-flow)"/>
  <text x="383" y="128" class="fig-small">updated weights</text>
</svg>
Figure 2. The forward path runs along the top, the loss and the update run underneath, and the dashed arrow returns the updated weights to the blocks. Decoding uses the trained weights and never sees the answer.
```

The stages come in the order data has to flow. Positions go in before the blocks because attention on its own can't tell order apart: without them, `1 2 3` and `3 2 1` look the same to the model. The mask lives inside attention, because a block can't be trusted to hide the future if the attention below it leaks. The loss comes after the logits and before the update, because the update needs a gradient to follow. Decoding comes last because it is the only honest test: the model has to produce the answer from the prompt alone.

I chose sinusoidal positions rather than rotary ones. This task needs absolute positions, because answer slot `j` always copies position `5 - j`. Rotary encodings give the model relative offsets, which here change with every slot. Both can work, and you can compare them in the extensions.

## Build it step by step

> [!RECAP]
> - The mask adds `-inf` above the diagonal of the scores, so softmax gives those positions zero weight.
> - Scores are divided by `sqrt(d_head)`, each head attends on its own slice of the width, and `Wo` mixes the heads.
> - A pre-norm block adds `attention(norm(x))` and then `feed_forward(norm(x))` to a stream that nothing normalises in place.
> - The loss averages `per_token_loss` over answer positions only.
> - A step is forward, loss, backward, clip, schedule, update, in that order.

`make_data`, `init_params`, `per_token_loss`, `layer_norm`, `warmup_cosine_lr`, `clip_global_norm`, and `adamw_update` are given, because you wrote each one in an earlier chapter. Everything else is yours.

1. **`sinusoidal_positions(T, d)`** returns a `(T, d)` table. Build the speeds `10000 ** (-2 * i / d)` for `i` in `range(d // 2)`, multiply by the position column, and write sine into even columns and cosine into odd ones. The common mistake is to write sine into the first half of the columns and cosine into the second half. The checks expect the interleaved layout from [position clocks](/lessons/047-position-encodings).

2. **`causal_mask(T)`** returns a `(T, T)` tensor with `-inf` where the column is greater than the row and `0` elsewhere. For `T = 3` the rows are `[0, -inf, -inf]`, `[0, 0, -inf]`, and `[0, 0, 0]`. Don't build it by multiplying by zero, because `0 * -inf` is NaN.

3. **`causal_self_attention(x, Wq, Wk, Wv, Wo, n_heads)`** takes `x` of shape `(B, T, d)`. Project, reshape to `(B, T, n_heads, d_head)`, move the head axis next to the batch axis, compute `q @ k.transpose(-1, -2) / sqrt(d_head)`, add the mask, softmax over the last axis, multiply by `v`, then move the heads back and apply `Wo`. The common mistake is to reshape the heads back without transposing first, which scrambles the width silently. Shapes still agree, so only the value check catches it.

4. **`transformer_block(x, p, prefix, n_heads)`** reads its weights from `p[prefix + "Wq"]` and so on. Normalise, attend, add. Normalise again, run `silu(h @ Wg) * (h @ Wu)` through `Wd`, add. The common mistake is to write `x = layer_norm(x) + attention(x)`, which normalises the stream itself and loses the clean residual path the check tests with zeroed output matrices.

5. **`forward(params, ids, cfg)`** looks up `params["embed"][ids]`, multiplies by `sqrt(d_model)` so the token signal is on the same scale as the position table, adds the positions, runs the blocks named `l0.`, `l1.`, applies the final norm, and multiplies by `params["embed"].T`. Tying the unembedding to the embedding saves a matrix, and it is why `parameter_count` counts the embedding once.

6. **`answer_loss(logits, targets, answer_mask)`** computes `per_token_loss`, zeroes the positions where the mask is false, and divides the sum by the number of true positions. The common mistake is to divide by all positions, which shrinks the loss and the gradient by half.

7. **`parameter_count(cfg)`** and **`training_flops(cfg, n_tokens)`** are formulas. The embedding has `V * d` weights. A block has four `d x d` attention matrices, three SwiGLU matrices of size `d x f`, `f x d`, and `d x f`, and two layer norms with a gain and a bias each:

   $$
   N = V d + L\,(4 d^2 + 3 d f + 4 d) + 2 d
   $$

   Read it aloud: the embedding, plus `L` blocks of attention, feed-forward, and two norms, plus the final norm. By hand, `V = 3`, `d = 2`, `f = 4`, `L = 1` gives `6 + (16 + 24 + 8) + 4 = 58`. The flop estimate is `6 * N * n_tokens`, from [the cost of a run](/lessons/050-training-loop-and-cost).

8. **`train(params, data, cfg, tcfg)`** runs the loop. Each step draws a batch with a seeded `torch.Generator`, splits rows into inputs `[:, :-1]` and targets `[:, 1:]`, builds a mask that is true from target position `N_DIGITS` on, and computes the loss. Call `backward()`, clip all gradients together, ask `warmup_cosine_lr` for this step's rate, and update every weight inside `torch.no_grad()`. Skip weight decay for tensors with fewer than two dimensions, since decaying a norm gain pulls it away from 1 for no benefit. Store each result as `theta.detach().requires_grad_()`, so the next step's gradients start clean. Count steps from 1, because bias correction divides by `1 - beta ** step`.

9. **`greedy_decode(params, prompts, n_new, cfg)`** runs `forward` on everything so far, takes the argmax at the last position, appends it, and repeats `n_new` times. **`exact_match(params, data, cfg)`** feeds the first seven tokens (six digits and the separator), decodes six tokens, and returns the fraction of rows that match all 13 tokens.

## Read your results

Run the file. With the reference solution and `seed = 0`, you should see these numbers, and yours should be close.

| Quantity | Value |
|---|---|
| Weights | 21,152 (352 in the embedding, 20,736 in two blocks, 64 in the final norm) |
| Training flops, 450 steps of 64 rows of 12 tokens | 4.39e10 |
| Loss at step 1 | 2.63 |
| Loss at steps 50, 100, 200 | 1.73, 0.26, 0.009 |
| Loss at step 450 | 0.001 |
| Exact match, untrained model | 0.00 |
| Exact match, trained, 300 held-out strings | 1.00 |

Across seeds 0 to 4, the reference's average loss over the last 20 steps stays between 0.0008 and 0.0012, and exact match is 1.00 on every seed. Training takes about 2 seconds.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Training loss against step: a plateau near 2.5, a sharp drop between steps 40 and 150, then a floor near zero">
  <line x1="70" y1="200" x2="560" y2="200" class="stroke-muted"/>
  <line x1="70" y1="30" x2="70" y2="200" class="stroke-muted"/>
  <line x1="70" y1="55" x2="560" y2="55" class="stroke-grid dashed"/>
  <text x="62" y="59" text-anchor="end" class="fig-small">2.3</text>
  <line x1="70" y1="197" x2="560" y2="197" class="stroke-grid dashed"/>
  <text x="62" y="186" text-anchor="end" class="fig-small">0.05</text>
  <text x="62" y="204" text-anchor="end" class="fig-small">0</text>
  <text x="70" y="218" text-anchor="middle" class="fig-small">0</text>
  <text x="123" y="218" text-anchor="middle" class="fig-small">50</text>
  <text x="177" y="218" text-anchor="middle" class="fig-small">100</text>
  <text x="283" y="218" text-anchor="middle" class="fig-small">200</text>
  <text x="550" y="218" text-anchor="middle" class="fig-small">450</text>
  <text x="315" y="236" text-anchor="middle" class="fig-small">training step</text>
  <polyline points="70.0,34.7 80.7,44.1 96.7,59.2 112.7,76.8 123.3,91.0 150.0,141.4 176.7,183.8 230.0,197.5 283.3,199.4 390.0,199.8 550.0,199.9" fill="none" class="stroke-b" stroke-width="2"/>
  <circle cx="70" cy="34.7" r="3.5" class="fill-b"/>
  <circle cx="123.3" cy="91" r="3.5" class="fill-b"/>
  <circle cx="176.7" cy="183.8" r="3.5" class="fill-b"/>
  <text x="80" y="28" class="fig-small">2.63 at step 1</text>
  <text x="190" y="170" class="fig-small">0.26 at step 100</text>
  <text x="300" y="60" class="fig-small">2.3 = ln 10, a guess among ten digits</text>
</svg>
Figure 3. The loss falls slowly through the 40 warmup steps, then collapses once attention finds the right source position. The dashed line near zero marks the 0.05 threshold in the checks.
```

Here is what each number tells your teammate. A loss near 2.3 means the model guesses among ten digits, so it hasn't found the lookup. A loss under 0.05 means the model is nearly certain of each answer digit, and exact match above 0.95 confirms the certainty is deserved on strings it never trained on.

Wrong numbers usually point to one cause each:

- **Loss stays near 2.3 for the whole run.** The model can't locate source positions. Check that `sinusoidal_positions` is added in `forward`, and that the head split and the transposes match.
- **Loss is near zero but exact match is far below 1.** The model has seen the future. In training, the input at position `t + 1` is the target at `t`, so a missing mask lets it read the answer it is scored on. At decode time that answer doesn't exist. In my runs with the mask removed, exact match ranged from 0.003 to 1.00 across five seeds, so a pass on one seed isn't evidence. The causal checks catch it deterministically.
- **Loss falls and then turns to NaN.** Look at the clip and at the mask: a row with every score at `-inf` gives NaN. The mask here never blocks a position's own column.
- **The weight count is off by 64 or 352.** You forgot the final norm, or you counted the tied embedding twice.

## Take it further

- Swap the sinusoidal table for rotary embeddings on the queries and keys. This shows whether relative offsets are enough for a task that wants absolute positions.
- Train on four-digit strings and test on six-digit strings. This shows how far a learned lookup reaches beyond the lengths it saw.
- Remove the mask from both training and decoding and measure exact match on five seeds. This shows why a low loss alone can't certify a language model.
- Change the task to add two 2-digit numbers, with the answer's digits written least significant first. This shows that carries make a task need more layers or more steps than copying does.
- Halve the width, then double it, and plot exact match against `parameter_count`. This links the formula to how capacity buys accuracy.

## The exercise

You fill in `sinusoidal_positions`, `causal_mask`, `causal_self_attention`, `transformer_block`, `forward`, `answer_loss`, `parameter_count`, `training_flops`, `train`, `greedy_decode`, and `exact_match`. The checks run from the smallest piece to the whole model:

- The position table matches sine and cosine values, and the mask has `-inf` exactly above the diagonal.
- Attention matches a head-by-head reference, and its output at position `t` doesn't change when later tokens change.
- A block with zeroed output matrices returns its input, and `forward` never looks ahead and does use order.
- `answer_loss` equals the mean over scored positions and gives no gradient to the rest.
- `parameter_count` equals the size of the initialised weights for two configurations and the hand example, and `training_flops` equals `6 N D`.
- The trained model's late-step loss is below 0.05, and greedy decoding reproduces at least 95% of held-out strings exactly.
