---
title: Training a transformer
slug: 27-training-a-transformer
part: "Part 5 — Transformers"
summary: "Train a tiny language model: the next-token loss, a warmup-cosine schedule, gradient clipping, and one AdamW step."
minutes: 75
gpu: true
objectives:
  - Compute the next-token cross-entropy loss and perplexity, and check that an untrained model scores log V.
  - Build a learning-rate schedule that warms up linearly and then decays along a cosine.
  - Clip gradients by their global norm and apply one AdamW update by hand.
  - Estimate the compute of a training run with the 6ND rule, and explain why bf16 needs no loss scaling.
prereqs:
  - 26-the-transformer-block
  - 17-optimisers
exercise: 27-training-a-transformer
---

# Training a transformer

You can build a transformer block. Training scores how well a stack of blocks guesses the next token. A token is a character or a word fragment. A gradient is the slope of the loss, and each step moves the weights along it.

A bad learning rate can waste a week of compute. The wrong number format can zero out tiny gradients and raise no error.

## What you already know

You already have these tools:

- Cross-entropy is minus the log of the correct token's probability.
- A softmax turns scores into positive numbers that add to one.
- AdamW averages the gradient and its square, and shrinks large weights.
- A forward pass costs about two flops per weight per token.
- A vector's norm is the square root of its squared entries, added up.

## Two guesses, then the loss

Take two positions and a vocabulary of three tokens. The scores, called logits, are [2, 0, 0] with target 0, and [0, 3, 0] with target 1.

For the first row, e² = 7.389. Add 1 and 1 to get 9.389. The log is 2.2395, so the loss is 2.2395 − 2 = 0.2395 and e^{−0.2395} = 0.787.

For the second row, e³ = 20.086. Add 1 and 1 to get 22.086. The log is 3.0949, so the loss is 3.0949 − 3 = 0.0949. The mean is (0.2395 + 0.0949) / 2 = 0.1672.

This loss is next-token cross-entropy. N is the number of predictions. You average the log of the probability of the true next token, then flip the sign:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \log p_i[\text{target}_i]
$$

Teacher forcing feeds the true text so far, so one forward pass scores every position. The targets are the inputs shifted by one place.

Dividing e to each score by the sum is the softmax. Subtract the log of that sum to get the log probability. `torch.logsumexp` does this without overflow.

Compute the loss from logits and integer targets:

```python
logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
loss = nll.mean()
```

The usual mistake is summing the token losses. The log V check needs the mean.

> [!TIP]
> `logp` is `(..., V)` and `targets.unsqueeze(-1)` is `(..., 1)`. After `squeeze(-1)` the result matches `targets`.

Logits near zero give probability 1/V and loss log V, for any target. V = 11 gives 2.398. V = 50000 gives 10.82. A first loss far from log V is a bug.

> [!KEY]
> An untrained model must score a loss close to log V.

Perplexity is e raised to the loss: how many tokens stay plausible. Here e^{0.1672} = 1.182. A flat model scores V, and a loss of 0 scores 1.

Turn a loss into a perplexity:

```python
ppl = math.exp(float(loss))
```

The usual mistake is reporting the loss itself as the perplexity.

## A ramp, then a glide

The first gradients are noisy, so a large step can throw the weights off. The rate climbs in a straight line. That climb is the warmup. A cosine then glides to a floor.

Try warmup 10, total 50, peak 2, and floor 0.1. Step 5 is 2 × 5 / 10 = 1.0. Step 10 is the peak, 2. Step 30 has progress (30 − 10) / (50 − 10) = 0.5. The Greek letter pi, written π, is the circle constant, so cos(π × 0.5) = 0 and the rate is 0.1 + 0.5 × 1.9 = 1.05. Step 50 is the floor, 0.1.

This curve is a warmup-cosine schedule:

$$
\text{lr}(s) =
\begin{cases}
\text{peak} \cdot \dfrac{s}{\text{warmup}} & s < \text{warmup} \\[2ex]
\text{floor} + \tfrac{1}{2}(\text{peak} - \text{floor})\big(1 + \cos(\pi \cdot \text{progress})\big) & s \ge \text{warmup}
\end{cases}
$$

Progress is $(s - \text{warmup}) / (\text{total} - \text{warmup})$. Warmup climbs from 0 to the peak. Progress 0 stays at the peak, and progress 1 lands on the floor.

Compute one learning rate:

```python
if step < warmup:
    return peak * step / warmup
progress = (step - warmup) / (total - warmup)
return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
```

The usual mistake is dividing progress by `total` alone. The drop then misses the floor.

## One speed limit on every gradient

A huge gradient can undo hours of training. Measure the length of every gradient at once. Past the limit, shrink every array by the same factor.

Take gradients [3, 4] and [0]. Squares 9 + 16 + 0 = 25, and the root is 5. A limit of 10 changes nothing. A limit of 2.5 multiplies by 2.5 / 5 = 0.5, so [3, 4] becomes [1.5, 2].

This limit is clipping by the global norm:

$$
g = \sqrt{\sum_{\text{arrays}} \sum_{\text{entries}} \text{grad}^2},
\qquad
\text{if } g > \text{max\_norm}: \ \text{grad} \leftarrow \text{grad} \cdot \frac{\text{max\_norm}}{g}
$$

Square every entry, add them, and take the square root. That length is g. Over the limit, multiply every array by the limit divided by g. The example returns 5.

Clip a list of gradient arrays:

```python
total = math.sqrt(sum(float((g**2).sum()) for g in grads))
if total > max_norm:
    for g in grads:
        g *= max_norm / total  # in place
```

The usual mistake is `g = g * scale`. That builds another array and leaves the caller's gradients alone.

> [!WARNING]
> Scale with `g *= ...`, in place. `g = g * ...` does not change the arrays in the list.

> [!KEY]
> Global-norm clipping shrinks every gradient by the same factor. The step keeps its direction and only gets shorter.

## One AdamW step

The Greek letter theta, written θ, is one weight. Beta, written β, is the blend on a running average. Lambda, written λ, is the weight decay. Epsilon, written ε, is 10^{−8} in the denominator.

Start from θ = 1, g = 0.5, step 1, learning rate 0.1, and λ = 0.1, with averages at 0. Use β₁ = 0.9 and β₂ = 0.95. Then m = (1 − 0.9) × 0.5 = 0.05 and v = (1 − 0.95) × 0.25 = 0.0125. Divide by 1 − βᵗ: m-hat = 0.05 / 0.1 = 0.5 and v-hat = 0.0125 / 0.05 = 0.25. The root is 0.5, so m-hat / 0.5 = 1. Rate times λ is 0.1 × 0.1 = 0.01, so the weight is 1 × (1 − 0.01) − 0.1 = 0.89.

AdamW is the optimiser. At step t, counting from 1:

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1 - \beta_1)\, g, &
v &\leftarrow \beta_2 v + (1 - \beta_2)\, g^2, \\
\hat{m} &= \frac{m}{1 - \beta_1^{t}}, &
\hat{v} &= \frac{v}{1 - \beta_2^{t}}, \\
\theta &\leftarrow \theta\,(1 - \text{lr}\cdot\lambda) - \text{lr}\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\end{aligned}
$$

Blend g into m and into v, then divide by 1 − βᵗ, because the averages start at zero. Shrink θ by λ and step by m-hat over the root of v-hat. These numbers give 0.89. With β₂ = 0.95, 1 / 0.05 = 20, so v covers about 20 steps.

Apply one update:

```python
m = b1 * m + (1 - b1) * g
v = b2 * v + (1 - b2) * g**2
m_hat = m / (1 - b1**step)
v_hat = v / (1 - b2**step)
theta = theta * (1 - lr * wd) - lr * m_hat / (torch.sqrt(v_hat) + eps)
```

The usual mistake is step 0. Then 1 − β⁰ = 0, and the correction divides by zero.

> [!WARNING]
> Count t from 1. On the first update, t = 1. With t = 0 the bias correction divides by zero.

Weight decay pulls a big matrix toward zero. Layer-norm gains and biases get λ = 0, listed in `no_decay`, as does any tensor with fewer than two dimensions.

One matrix can embed tokens and, transposed, score them. That saves V × d weights. The demo uses `params["embed"].T`.

## Sixteen-bit numbers

Two 16-bit formats are faster than 32-bit floats, and they fail in different ways. Exponent bits set the range. Mantissa bits set the precision.

The three formats differ like this:

| Format | Exponent bits | Mantissa bits | Smallest normal number |
|---|---|---|---|
| fp32 | 8 | 23 | about 10^{−38} |
| fp16 | 5 | 10 | about 6 × 10^{−5} |
| bf16 | 8 | 7 | about 10^{−38} |

A gradient of 10^{−6} is below fp16's smallest normal number, about 6 × 10^{−5}, so it becomes 0. Multiply the loss by s, then divide the gradients by s. bf16 has fp32's range, so it needs no scaling.

> [!KEY]
> fp16 has a small range, so tiny gradients round to zero unless you scale the loss. bf16 has the range of fp32, so it needs no loss scaling.

## Six times weights times tokens

Forward costs about 2N flops per token. Backward costs about 4N, so D tokens cost about 6ND. A million weights and ten thousand tokens take 6 × 10⁶ × 10⁴ = 6 × 10¹⁰ flops.

$$
C \approx 6 N D
$$

That product is six times the weights times the tokens.

Estimate the flops of a run:

```python
flops = 6 * params * tokens
```

The usual mistake is counting only the forward pass, which reports 2ND.

> [!KEY]
> Training costs about 6ND flops: 2N per token forward and 4N backward.

> [!RECAP]
> Carry these facts into the exercise:
> - The loss averages minus the log of each target probability, using `torch.logsumexp` on `dim=-1`.
> - An untrained model scores about log V. Perplexity is e raised to the loss.
> - Warmup climbs straight to the peak, then a cosine falls to the floor.
> - One global norm scales every gradient in place. Return the norm from before the clip.
> - AdamW starts at step 1, corrects m and v, and skips names in `no_decay`.

## Build it step by step

The checks test the functions in this order:

1. **`causal_lm_loss(logits, targets)`.** Logits are `(..., V)` and targets are `(...)`. Return a scalar from `torch.logsumexp`, `gather`, and `.mean()` on `dim=-1`. The usual mistake is a sum, which breaks the log V check.
2. **`perplexity(loss)`.** Return `math.exp(float(loss))`.
3. **`warmup_cosine_lr(step, warmup, total, peak, floor)`.** Return a float. For `step < warmup`, return `peak * step / warmup`. After warmup, divide progress by `total - warmup`. The usual mistake is dividing by `total`.
4. **`clip_global_norm(grads, max_norm)`.** Return the norm before clipping, and scale with `g *= scale`. The usual mistake is `g = g * scale`.
5. **`training_flops(params, tokens)`.** Return `6 * params * tokens`.
6. **`train_step(model_fn, params, batch, opt_state, config)`.** Return parameters, optimiser state, and a float loss. Run `model_fn(params, batch["inputs"])`, `causal_lm_loss` on `batch["targets"]`, and `loss.backward()`. Clip each `params[name].grad` with `config["max_grad_norm"]`. Set `step = opt_state["step"] + 1` before bias correction, and read `peak_lr`, `floor_lr`, `warmup_steps`, and `total_steps`. Inside `torch.no_grad()`, update every name and use decay 0 for `config["no_decay"]`. Defaults: `beta1` 0.9, `beta2` 0.95, `eps` 1e-8, `weight_decay` 0. Store `updated.detach().requires_grad_()`. The usual mistake is bias correction at step 0.

## Cost and scaling

The main costs of a run grow like this:

| Quantity | Rough size |
|---|---|
| Forward flops per token | 2N |
| Backward flops per token | 4N |
| Total training flops | 6ND |
| Memory for AdamW | 4N numbers: weights, gradients, m, and v |

AdamW also stores m and v, so memory is about 4N and often fills the GPU. Gradient accumulation sums small batches, then takes one step.

## Where this breaks

Training fails in four familiar ways:

- A first loss far from log V means the vocabulary, the output scale, or the target shift is wrong. Check before training.
- fp16 without loss scaling turns small gradients into zeros, with no error. Use bf16 or scale the loss.
- A limit that clips every step throws away the gradient's size. Set it to catch rare spikes.
- A decaying layer-norm gain weakens its sublayer. Keep gains and biases in `no_decay`.

## The exercise

You write `causal_lm_loss`, `perplexity`, `warmup_cosine_lr`, `clip_global_norm`, `training_flops`, and `train_step`. The file already holds a tiny character-level transformer. Once the functions work, Run trains it on a repeated line and prints a sample.

The checks match log V on uniform logits, with and without a batch, plus a hand loss, perplexity, and the schedule at 0, the peak, and the floor. They check the pre-clip norm, 6ND, one AdamW step, and a loss below log V after twenty steps.
