---
title: "Alignment: reward models, PPO, and DPO"
slug: 32-alignment
part: "Part 6 — Large language models"
summary: Turn human preferences into a training signal with a reward model, then train a policy with PPO or DPO.
minutes: 60
gpu: false
objectives:
  - Compute the Bradley-Terry reward loss and its gradient from pairs of preferred and rejected answers.
  - Compute the best policy for a KL-limited reward objective, and explain what beta controls.
  - Describe PPO's rollout loop, value head, and clipped objective, without implementing them.
  - Build the DPO loss from implicit rewards, and explain why its gradient acts as a per-example weight.
prereqs:
  - 31-parameter-efficient-tuning
exercise: 32-alignment
---

# Alignment: reward models, PPO, and DPO

Pretraining teaches a model to predict the next word of internet text.
Supervised fine-tuning then teaches it to copy example answers that people
wrote. Both copy a target. Neither ever learns that one answer is *better*
than another. A thorough answer and a curt one that start the same way look
alike to a copying objective.

People struggle to write the perfect answer, but they're good at comparing
two. "Answer A is better than answer B" is cheap to collect and fairly
reliable. This lesson turns comparisons like that into a training signal.

You build three pieces. A reward model learns a score from comparisons. A
closed-form rule says what the best policy looks like if you want high
reward without drifting far from the original model. DPO then trains toward
that best policy directly, with no sampling loop. PPO, the older method, is
described in words.

## What you already know

- Supervised fine-tuning minimises cross-entropy against a fixed target
  answer.
- The sigmoid $\sigma(z) = 1 / (1 + e^{-z})$ squashes any number into a
  probability between 0 and 1.
- A log-probability is the logarithm of a probability. It's always zero or
  negative.
- A softmax turns a vector of scores into probabilities that sum to one.
- LoRA makes it cheap to keep several fine-tuned copies of one model in
  memory.

In this lesson, a *policy* is the language model viewed as a rule for picking
answers. $\pi(y \mid x)$ is the probability the policy gives answer $y$ to
prompt $x$.

## Learn a reward from comparisons

Think of chess ratings. You never see a player's "true skill", only who beat
whom. A rating system assigns each player a number so that the gap between
two numbers predicts who wins. The Bradley-Terry model does the same for
answers.

Suppose a reward $r$ scores each answer. The data is pairs: for each prompt, a
*chosen* answer $y_w$ that a person preferred, and a *rejected* answer $y_l$.
The model says the chance a person prefers the chosen one is

$$
p(\text{chosen wins}) = \sigma(r_w - r_l)
$$

Read it as: take the reward gap, chosen minus rejected, and squash it with a
sigmoid. A gap of 0 gives even odds, 0.5. A gap of 3 gives about 0.95.

To train the reward, maximise the probability of the preferences you
observed. As a loss to minimise, that's the average of the negative log over
$n$ pairs:

$$
\text{loss} = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(z_i), \qquad z_i = r_{w,i} - r_{l,i}
$$

Read it as: for each pair, find the gap $z$, take minus the log of its
sigmoid, and average. A big correct gap costs almost nothing. A big wrong gap
costs about the size of the gap. Two numpy tricks keep this stable:
$-\log \sigma(z) = \log(1 + e^{-z})$, and `np.log1p(x)` computes $\log(1 +
x)$ accurately.

```python
z = reward_chosen - reward_rejected
loss = np.mean(np.log1p(np.exp(-z)))   # fine for moderate z; see step 1
```

### The gradient

The *gradient* tells you how the loss changes when you nudge each input. The
derivative of $-\log \sigma(z)$ with respect to $z$ is $\sigma(z) - 1$. The
loss is an average, so each pair's share carries a $1/n$. And $z$ goes up when
$r_w$ goes up and down when $r_l$ goes up. So:

$$
\frac{\partial\, \text{loss}}{\partial r_{w,i}} = \frac{\sigma(z_i) - 1}{n}, \qquad
\frac{\partial\, \text{loss}}{\partial r_{l,i}} = -\frac{\sigma(z_i) - 1}{n}
$$

Read it as: $\sigma(z) - 1$ is always negative, so the loss falls when you
raise the chosen reward. It's close to $-1$ when the pair is ranked wrong and
close to 0 when it's already ranked right by a wide margin. Pairs the model
already gets right barely move it.

### Worked example

Two pairs: rewards chosen $[2, 0]$, rejected $[1, 1]$.

- Gaps: $z = [1, -1]$
- Sigmoids: $\sigma(1) = 0.7311$, $\sigma(-1) = 0.2689$
- Per-pair loss: $-\log 0.7311 = 0.3133$ and $-\log 0.2689 = 1.3133$
- Mean loss: $(0.3133 + 1.3133) / 2 = 0.8133$
- Gradient for chosen: $[(0.7311 - 1)/2, (0.2689 - 1)/2] = [-0.1345,
  -0.3655]$. For rejected, the same values with the sign flipped.

The second pair, ranked wrong, gets the bigger push.

### Only gaps matter

The loss sees rewards only through differences. Add the same number to both
answers for one prompt, and nothing changes. So the reward is fixed only up
to a constant per prompt. That slack looks like a flaw here. DPO turns it into
a feature.

## Find the best policy on a leash

A reward model alone isn't enough. If you push a policy to maximise it with
no limits, the policy finds tricks that score well and read badly. This is
called *reward hacking*. The fix is a leash: stay close to a *reference
policy* $\pi_{\text{ref}}$, usually the supervised fine-tuned model, while
chasing reward.

The leash is measured with the KL divergence. For two probability lists $p$
and $q$ over the same answers,

$$
\text{KL}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

Read it as: for each answer, weight the log of the ratio by $p$'s probability,
and add up. It's 0 when $p$ and $q$ match and positive otherwise. Terms with
$p_i = 0$ count as 0. In numpy: `np.sum(np.where(p > 0, p * np.log(p / q),
0.0), axis=-1)`. For $p = [0.2, 0.3, 0.5]$ and $q = [0.1, 0.4, 0.5]$:
$0.2 \log 2 + 0.3 \log 0.75 + 0 = 0.1386 - 0.0863 = 0.0523$.

The goal is to maximise average reward minus $\beta$ times this KL, where
$\beta$ is a positive number you pick. Solving that with calculus gives an
exact best policy:

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(r(x, y) / \beta)}{Z(x)}
$$

Read it piece by piece:

- Start from the reference probability of each answer.
- Multiply by $\exp(\text{reward} / \beta)$, which boosts high-reward answers.
- Divide by $Z(x)$, the total of those products over all answers for this
  prompt, so the result sums to one.

With log-probabilities, that's a softmax of `ref_logprobs + rewards / beta`:

```python
logits = ref_logprobs + rewards / beta
logits = logits - logits.max(axis=-1, keepdims=True)
```

$\beta$ is the length of the leash. When $\beta$ is huge, $r / \beta$ is near
0 and $\pi^*$ equals the reference: the policy never moves. When $\beta$ is
tiny, the best answer's boost dwarfs the rest and $\pi^*$ puts everything on
the highest-reward answer: no leash at all.

### Worked example

Three answers with reference probabilities $[0.5, 0.3, 0.2]$, rewards $[1, 0,
0]$, and $\beta = 1$:

- Boosts: $[e^1, e^0, e^0] = [2.7183, 1, 1]$
- Products: $[1.3591, 0.3, 0.2]$, total $Z = 1.8591$
- $\pi^* = [0.7311, 0.1614, 0.1076]$

A real model can't compute $Z(x)$, because it would need to sum over every
possible sentence. PPO and DPO are two ways around that.

## PPO in words

Proximal Policy Optimisation treats the problem as reinforcement learning. It
needs several moving parts:

- **A rollout loop.** Generate full answers from the current policy, one
  token at a time, and score each with the frozen reward model. This is slow,
  and it repeats after every update, because PPO learns from its own fresh
  samples.
- **A value head.** A second output predicts how much reward a partial answer
  will end up with. It makes the learning signal less noisy.
- **Advantages.** For each token, how much better it turned out than the
  value head expected. This is what the update pushes on.
- **A clipped objective.** PPO reuses each batch of samples for a few steps.
  To stop one batch from moving the policy too far, it caps how much each
  token's probability ratio against the sampling policy can change the
  objective, to a range like 0.8 to 1.2.

The KL leash enters as a per-token penalty on the reward. PPO needs four
models in memory: the policy, the value head, the reward model, and the
reference.

## DPO: skip the sampling

Direct Preference Optimisation starts from the best-policy formula and runs
it backwards. Solve it for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Read it as: the reward of an answer is $\beta$ times how much more likely the
best policy makes it than the reference, plus a term that depends only on the
prompt.

Now plug that into the Bradley-Terry gap. Both answers share the prompt, so
the $\beta \log Z(x)$ term appears in both rewards and cancels. That's the
"only gaps matter" slack from earlier. What's left uses only
log-probabilities the model can compute. Define the *implicit reward* of an
answer under the policy being trained:

$$
\hat{r} = \beta \left(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\right)
$$

In numpy that's `beta * (policy_logps - ref_logps)`. The DPO loss is the
Bradley-Terry loss applied to these implicit rewards:

$$
\text{DPO loss} = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(\hat{r}_{w,i} - \hat{r}_{l,i})
$$

Read it as: for each pair, compute how much the policy has raised the chosen
answer relative to the reference, minus the same for the rejected answer,
then apply the reward-model loss. There's no sampling, no reward model during
training, and no value head. You need only the policy and a frozen copy of the
reference.

### The gradient

The chain rule adds one factor. $\hat{r}$ changes by $\beta$ for every unit
change in the policy log-probability. So the gradient with respect to the
policy's log-probabilities is $\beta$ times the Bradley-Terry gradient,
evaluated at the implicit rewards:

$$
\frac{\partial\, \text{loss}}{\partial \log \pi_\theta(y_{w,i})} = \beta \, \frac{\sigma(u_i) - 1}{n}, \qquad u_i = \hat{r}_{w,i} - \hat{r}_{l,i}
$$

The rejected answer gets the same value with the sign flipped. Read it as a
per-example weight: pairs the policy still ranks wrong get a push near
$\beta / n$. Pairs it already ranks right by a wide margin get almost none.
Going downhill raises the chosen answer's log-probability and lowers the
rejected one's.

### Worked example

One pair, $\beta = 0.1$. Policy log-probs: chosen $-1$, rejected $-3$.
Reference log-probs: both $-2$.

- Implicit rewards: $\hat{r}_w = 0.1 \times (-1 + 2) = 0.1$, and $\hat{r}_l =
  0.1 \times (-3 + 2) = -0.1$
- Gap: $u = 0.2$, and $\sigma(0.2) = 0.5498$
- Loss: $-\log 0.5498 = 0.5981$
- Gradient for chosen: $0.1 \times (0.5498 - 1) = -0.0450$. For rejected:
  $+0.0450$

## Build it step by step

The checks test the functions in this order.

1. **`bradley_terry_loss(reward_chosen, reward_rejected)`.** Takes two `(n,)`
   arrays, returns a float. Compute `z`, then $\log(1 + e^{-z})$. For large
   negative `z`, `np.exp(-z)` overflows, so use `np.where` to switch to $-z +
   \log(1 + e^{z})$ when `z < 0`. Common mistake: summing instead of
   averaging. The checks expect $\log 2$ for tied rewards.
2. **`bradley_terry_grad(reward_chosen, reward_rejected)`.** Returns two `(n,)`
   arrays. Write your own stable sigmoid: for `z < 0` use `exp(z) / (1 +
   exp(z))`. Common mistake: forgetting the `1 / n` from the mean. Finite
   differences catch it.
3. **`optimal_policy(ref_logprobs, rewards, beta)`.** Takes arrays of the same
   shape, often `(prompts, answers)`. Returns probabilities of that shape,
   each row summing to one. Use `axis=-1` and `keepdims=True` for the max and
   the sum. Common mistake: normalising over the whole array instead of each
   row.
4. **`kl_divergence(p, q)`.** Sum over the last axis. Common mistake: letting
   `0 * log(0)` produce `nan`. Mask it with `np.where`.
5. **`implicit_reward(policy_logps, ref_logps, beta)`.** One line,
   elementwise.
6. **`dpo_loss(...)`.** Build both implicit rewards and pass them to your
   `bradley_terry_loss`. Reusing it guarantees the first DPO check passes and
   gives you the $Z(x)$ cancellation for free.
7. **`dpo_grad(...)`.** Call `bradley_terry_grad` on the implicit rewards, then
   multiply both results by `beta`. Common mistake: dropping the `beta`
   factor. The last check runs 50 gradient steps and expects the loss to fall.

## Cost and scaling

The methods differ in how many models they keep in memory and whether they
generate text during training.

| Stage | Models in memory | Generates text? | Work per training pair |
|---|---|---|---|
| Reward model | 1 | No | One forward pass per answer |
| PPO | 4: policy, value head, reward model, reference | Yes, every round | Token-by-token generation, plus scoring |
| DPO | 2: policy and reference | No | One forward and backward pass per answer |

PPO spends most of its time generating, which is slow because it's
sequential. DPO costs about the same as supervised fine-tuning on twice as
many answers. The exercise functions are all a few array operations over the
batch, so they cost time proportional to $n$.

## Where this breaks

**Reward hacking.** A reward model trained on a finite set of comparisons is
wrong somewhere. PPO actively searches for those errors and finds answers
that score well but read badly: repetitive, flattering, or gaming the format.
The KL leash slows this but doesn't stop it. People retrain the reward model
on fresh comparisons and stop training early.

**Bias in the comparisons.** Both methods learn whatever the people comparing
answers preferred, including their blind spots. Neither knows anything about
prompts unlike the ones they saw.

**Collapse at small beta.** The best-policy formula shows it: a short leash
makes the policy nearly deterministic. Output becomes less varied and more
sensitive to reward-model mistakes.

**DPO lowering both answers.** DPO only needs the gap to grow. It can grow
the gap by lowering the chosen answer's probability a little and the rejected
one's a lot. Since rejected answers are often fluent, this can pull
probability away from other good answers. People add a small supervised loss
on the chosen answers to hold them up.

## The exercise

You build the losses and the best-policy rule from this lesson. Write
`bradley_terry_loss` and `bradley_terry_grad` for the reward-model loss and
its gradient, `optimal_policy(ref_logprobs, rewards, beta)` for $\pi^*$ over a
short list of answers per prompt, and `kl_divergence(p, q)`. Then write
`implicit_reward(policy_logps, ref_logps, beta)`, and `dpo_loss` and
`dpo_grad` from the policy and reference log-probabilities of the chosen and
rejected answers.

The checks verify the Bradley-Terry loss at tied and widely separated
rewards, both gradients against finite differences, and that `optimal_policy`
matches the formula and reaches the reference for huge $\beta$ and the top
answer for tiny $\beta$. They check KL against a hand computation, that
`dpo_loss` equals `bradley_terry_loss` on the implicit rewards and ignores a
shared shift of the reference log-probabilities, that `implicit_reward`
recovers a planted reward up to a per-prompt constant, and that 50 DPO steps
widen the gap and lower the loss.
