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

You can build a transformer block. Now you'll teach a stack of them to write.
The task is always the same: read some text and guess the next token. A token
is a small piece of text, such as a character or a word fragment. Do that
guess well enough, billions of times, and the model learns grammar, facts, and
style along the way.

Training a big model is expensive, so small mistakes cost real money. A bad
learning rate can waste a week of compute. The wrong number format can quietly
zero out gradients without any error. This lesson walks through the handful of
choices that almost every transformer training run shares, and why each one
is there.

You'll finish by writing one complete training step in PyTorch and watching a
tiny model memorise a string.

## What you already know

- Cross-entropy loss is minus the log of the probability the model gave the
  right answer.
- AdamW keeps running averages of the gradient and its square, and shrinks
  weights a little each step, from lesson 17.
- A transformer block's forward pass costs about two flops per weight per
  token, from lesson 26.
- The softmax turns scores, called logits, into probabilities that sum to one.
- The norm of a vector is its length: the square root of the sum of its
  squared entries.

## The objective

A language model reads tokens $x_1, x_2, \dots$ and at each position predicts
a probability for every token in its vocabulary of size $V$. The loss at each
position is the cross-entropy against the token that actually came next. You
average over all positions:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \log p_i[\text{target}_i]
$$

Read aloud: $N$ is the number of predictions (every position of every
sequence in the batch). For prediction $i$, look up the probability $p_i$
the model gave to the true next token, take its log, and average. The minus
sign turns "high probability" into "low loss".

In training, the model always sees the true text so far, never its own
guesses. This is called **teacher forcing**. It means one forward pass gives
predictions at every position at once. The targets are the inputs shifted by
one: if the inputs are "t, o, space" the targets are "o, space, b".

### Compute it safely

You get log-probabilities from logits with the log-softmax: subtract the
log-sum-exp of the row. The log-sum-exp is $\log \sum_j e^{z_j}$, and
`torch.logsumexp` computes it without overflow.

```python
logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)   # (..., V)
nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)       # (...)
loss = nll.mean()
```

`gather` picks, at each position, the entry whose index is the target.

### A worked example

Take two positions with $V = 3$. The logits are $[2, 0, 0]$ with target 0, and
$[0, 3, 0]$ with target 1.

1. Row 1: $\log(e^2 + 1 + 1) = \log 9.389 = 2.2395$. The loss is
   $2.2395 - 2 = 0.2395$. The model gave the target probability 0.787.
2. Row 2: $\log(1 + e^3 + 1) = \log 22.086 = 3.0949$. The loss is
   $3.0949 - 3 = 0.0949$.
3. The mean is $(0.2395 + 0.0949) / 2 = 0.1672$.

### The sanity check: log V

An untrained model has logits near zero, so it spreads probability evenly:
$1/V$ for every token. Its loss is $-\log(1/V) = \log V$, whatever the target.
With $V = 11$, that's 2.398. With $V = 50000$, it's 10.82. If your very first
loss is far from $\log V$, you have a bug. Find it before spending any compute.

### Perplexity

**Perplexity** is $e^L$, the loss taken back out of log space.

```python
ppl = math.exp(loss)
```

It reads as "how many tokens the model is torn between, on average". The
untrained model has perplexity $e^{\log V} = V$: it's guessing among all $V$
tokens. The worked example has perplexity $e^{0.1672} = 1.18$, almost
certain. A loss of 0 gives perplexity 1.

## The learning-rate schedule

Early in training, AdamW's running averages are built from only a few noisy
gradients. Big steps at that point can knock the model somewhere bad. So you
start with a tiny learning rate and ramp it up, like easing a car out of a
parking space before you hit the motorway. That's the **warmup**. After it,
you slowly lower the rate so the model settles into a good spot. A cosine
curve gives a smooth glide down.

$$
\text{lr}(s) =
\begin{cases}
\text{peak} \cdot \dfrac{s}{\text{warmup}} & s < \text{warmup} \\[2ex]
\text{floor} + \tfrac{1}{2}(\text{peak} - \text{floor})\big(1 + \cos(\pi \cdot \text{progress})\big) & s \ge \text{warmup}
\end{cases}
$$

where $\text{progress} = (s - \text{warmup}) / (\text{total} - \text{warmup})$.

Read aloud: during warmup, the rate climbs in a straight line from 0 to the
peak. After warmup, progress runs from 0 to 1. At progress 0, the cosine is 1,
so the bracket is 2 and the rate is exactly the peak. At progress 1, the
cosine is $-1$, the bracket is 0, and the rate is exactly the floor.

```python
if step < warmup:
    return peak * step / warmup
progress = (step - warmup) / (total - warmup)
return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
```

Try warmup 10, total 50, peak 2, and floor 0.1. At step 5, the rate is
$2 \cdot 5/10 = 1.0$. At step 10, it's the peak, 2. At step 30, progress is
0.5, the cosine is 0, and the rate is $0.1 + 0.95 = 1.05$. At step 50, it's
the floor, 0.1.

## Gradient clipping

Now and then, one batch produces a huge gradient. One huge step can undo hours
of training. Clipping puts a speed limit on the step. You measure the length
of the whole gradient, across every parameter at once, and if it's over the
limit, you shrink every gradient by the same factor.

$$
g = \sqrt{\sum_{\text{arrays}} \sum_{\text{entries}} \text{grad}^2},
\qquad
\text{if } g > \text{max\_norm}: \ \text{grad} \leftarrow \text{grad} \cdot \frac{\text{max\_norm}}{g}
$$

Read aloud: square every gradient entry in every array, add them all up, and
take the square root. That's the global norm $g$. If it's over the limit,
multiply every array by the limit divided by $g$. Afterwards the global norm
is exactly the limit. Shrinking everything by the same factor keeps the
direction of the step and only changes its length.

```python
total = math.sqrt(sum(float((g**2).sum()) for g in grads))
if total > max_norm:
    for g in grads:
        g *= max_norm / total   # in place
```

For the gradients $[3, 4]$ and $[0]$, the global norm is 5. With a limit of
10, nothing changes. With a limit of 2.5, both arrays are halved, and
$[3, 4]$ becomes $[1.5, 2]$. The function returns 5, the norm before
clipping.

## One AdamW step

AdamW is the optimiser. For every parameter $\theta$ with gradient $g$, it
keeps two running averages: $m$, the average gradient, and $v$, the average
squared gradient. At step $t$ (counting from 1):

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1 - \beta_1)\, g, &
v &\leftarrow \beta_2 v + (1 - \beta_2)\, g^2, \\
\hat{m} &= \frac{m}{1 - \beta_1^{t}}, &
\hat{v} &= \frac{v}{1 - \beta_2^{t}}, \\
\theta &\leftarrow \theta\,(1 - \text{lr}\cdot\lambda) - \text{lr}\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\end{aligned}
$$

Read aloud: blend the new gradient into each running average. Both averages
start at zero, so early on they're too small. Dividing by $1 - \beta^t$ fixes
that, which is called bias correction. Then shrink the weight a little by
$\lambda$, the weight decay, and step against the gradient, scaled by its
typical size. Transformers use $\beta_1 = 0.9$ and $\beta_2 = 0.95$. The
smaller $\beta_2$ means $v$ remembers only about the last 20 steps.

```python
m = b1 * m + (1 - b1) * g
v = b2 * v + (1 - b2) * g**2
m_hat, v_hat = m / (1 - b1**step), v / (1 - b2**step)
theta = theta * (1 - lr * wd) - lr * m_hat / (torch.sqrt(v_hat) + eps)
```

A worked example: $\theta = 1$, $g = 0.5$, step 1, lr 0.1, $\lambda = 0.1$.
Then $m = 0.05$ and $v = 0.0125$. After bias correction, $\hat{m} = 0.5$ and
$\hat{v} = 0.25$, so $\hat{m} / \sqrt{\hat{v}} = 1$. The new weight is
$1 \cdot (1 - 0.01) - 0.1 \cdot 1 = 0.89$.

### Which weights not to decay

Weight decay pulls weights toward zero. That's good for big weight matrices.
It's bad for layer norm gains, which start at 1 on purpose, and pointless for
biases. So those get $\lambda = 0$. In code, the config lists their names in
`no_decay`. A common rule is to skip decay for any tensor with fewer than two
dimensions.

### Weight tying

The input embedding maps tokens to vectors. The output layer maps vectors back
to token scores. Many models use the same matrix for both, transposed on the
output side. That saves $V \times d$ weights. The `__main__` demo does this:
it multiplies by `params["embed"].T` to get logits.

## Number formats

Training in 32-bit floats (fp32) is safe but slow. Two 16-bit formats are
faster, and they fail differently. The exponent bits set how tiny or huge a
number can be. The mantissa bits set how many digits of precision you get.

| Format | Exponent bits | Mantissa bits | Smallest normal number |
|---|---|---|---|
| fp32 | 8 | 23 | about $10^{-38}$ |
| fp16 | 5 | 10 | about $6 \times 10^{-5}$ |
| bf16 | 8 | 7 | about $10^{-38}$ |

Gradients are often smaller than $10^{-5}$. In fp16, those round to exactly
zero, and learning silently stops. The fix is **loss scaling**: multiply the
loss by a big number $s$ before the backward pass, which multiplies every
gradient by $s$ too. Divide the gradients by $s$ before the optimiser step.
bf16 has the same range as fp32, so it needs no scaling. It's less precise,
but it doesn't lose small numbers.

## How much compute a run takes

The forward pass costs about $2N$ flops per token for a model with $N$
weights: one multiply and one add per weight. The backward pass does two
matrix products of the same size for every forward one: one for the gradient
of the inputs and one for the gradient of the weights. So it costs about $4N$.
For $D$ training tokens:

$$
C \approx 6 N D
$$

Read aloud: total training flops are six times the number of weights times
the number of tokens.

```python
flops = 6 * params * tokens
```

A model with a million weights trained on ten thousand tokens takes
$6 \times 10^{6} \times 10^{4} = 6 \times 10^{10}$ flops. Lesson 29 uses this
rule to choose model sizes.

## Build it step by step

The checks test your functions in this order.

1. **`causal_lm_loss(logits, targets)`** takes logits of shape $(\dots, V)$
   and integer targets of shape $(\dots)$, and returns a scalar tensor. Use
   `torch.logsumexp`, `gather`, and `.mean()`. It must work with or without a
   batch axis, so work on `dim=-1`. Common mistake: summing instead of taking
   the mean, which breaks the $\log V$ check.
2. **`perplexity(loss)`** returns `math.exp` of the loss as a float. Wrap the
   input in `float()` so a tensor works too.
3. **`warmup_cosine_lr(step, warmup, total, peak, floor)`** returns a float.
   Two branches, as above. Common mistake: using `step <= warmup` for the
   warmup branch. Both branches give the peak at the boundary, so either works,
   but make sure the cosine branch divides by `total - warmup`, not `total`.
4. **`clip_global_norm(grads, max_norm)`** returns the norm before clipping
   and scales the arrays in place. Use `g *= scale`, which works for numpy
   arrays and torch tensors. Common mistake: `g = g * scale`, which makes a
   new array and leaves the caller's list untouched.
5. **`training_flops(params, tokens)`** returns `6 * params * tokens`.
6. **`train_step(model_fn, params, batch, opt_state, config)`** returns
   `(new_params, new_opt_state, loss_as_float)`. Steps:
   - Compute logits with `model_fn(params, batch["inputs"])`, the loss with
     `causal_lm_loss`, and call `loss.backward()`.
   - Collect every `params[name].grad` and pass the list to
     `clip_global_norm` with `config["max_grad_norm"]`.
   - Set `step = opt_state["step"] + 1`, then get the learning rate for that
     step from `warmup_cosine_lr` with the config's schedule fields.
   - Inside `torch.no_grad()`, apply the AdamW update to each name, using 0
     decay for names in `config["no_decay"]`. Read `beta1`, `beta2`, `eps`,
     and `weight_decay` with `config.get` and defaults 0.9, 0.95, `1e-8`, and
     0.
   - Store each new weight as `updated.detach().requires_grad_()`, and return
     the new `m`, `v`, and step in a fresh `opt_state` dict.

   Common mistake: using the old step count, 0, in the bias correction. That
   divides by $1 - \beta^0 = 0$.

## Cost and scaling

The main costs of a run grow like this.

| Quantity | Rough size |
|---|---|
| Forward flops per token | $2N$ |
| Backward flops per token | $4N$ |
| Total training flops | $6ND$ |
| Memory for AdamW | $4N$ numbers: weights, gradients, $m$, and $v$ |

The memory row surprises people. AdamW stores two extra numbers per weight, so
the optimiser state is often the first thing that runs out of room on a GPU.
**Gradient accumulation** helps with batch size: add up the gradients of
several small batches before one step. It behaves like one big batch, for
the price of more time instead of more memory.

## Where this breaks

- **First loss far from log V.** Something is wrong: the vocabulary size, the
  output layer's starting scale, or the target shift. Check before training.
- **fp16 without loss scaling.** Small gradients become zero, and the earliest
  layers stop learning, with no error message. Use bf16 or loss scaling.
- **Clip limit set too low.** If every step is clipped, every step has the same
  length, and you lose the information in the gradient's size. The limit
  should only catch rare spikes.
- **Decaying norm gains.** A layer norm gain shrinking toward zero weakens its
  sublayer every step. The symptom looks like an unrelated instability. Keep
  gains and biases in `no_decay`.

## The exercise

You write six functions: `causal_lm_loss`, `perplexity`, `warmup_cosine_lr`,
`clip_global_norm`, `training_flops`, and `train_step`. The file already
contains a tiny character-level transformer, so once your functions work, the
**Run** button trains it on a repeated line of text and prints a sample.

The checks confirm that uniform logits give a loss of exactly $\log V$, with
and without a batch axis, and that the loss matches a hand computation. They
check that perplexity is $e^L$, that the schedule starts at 0, hits the peak
at the warmup boundary without a jump, and ends at the floor. They check that
clipping reports the pre-clip norm, leaves small gradients alone, and trims
large ones to exactly the limit. Then they check the $6ND$ formula, one
`train_step` against an AdamW update computed by hand, and that twenty steps
drive the loss on a repeating pattern well below $\log V$.
