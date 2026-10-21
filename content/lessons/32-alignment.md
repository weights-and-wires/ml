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

Pretraining predicts the next word. Supervised fine-tuning copies answers people wrote. Both copy a target. A careful answer and a short one that start the same way look alike to that loss.

People struggle to write one perfect answer. They can say which of two answers they prefer. "B is better than A" is cheap to collect.

A policy is the model used as a rule that gives each answer a probability. You turn preferences into a reward, then into a loss. You compute one comparison probability, the best policy on a leash, and direct preference optimisation (DPO), which trains that policy with no sampling loop. Proximal policy optimisation (PPO) is the older loop, and it stays in words.

## What you already know

This lesson uses five ideas from earlier work:

- Supervised fine-tuning minimises cross-entropy against a fixed answer.
- A sigmoid squashes any real number into a value between 0 and 1.
- A softmax turns scores into probabilities that sum to one.
- LoRA keeps several fine-tuned copies of one model cheaply in memory.
- A loss you can differentiate gives a direction for each trained number.

## Start from one preference

A person reads two answers and says B is better than A. Give B reward 2 and A reward 0. The gap is 2.

The sigmoid $\sigma$ (sigma, pronounced SIG-muh) is $\sigma(z) = 1 / (1 + e^{-z})$. Here $e^{-2} \approx 0.1353$, so $1 + 0.1353 = 1.1353$ and $1 / 1.1353 \approx 0.8808$. The chance the preference picks B is about 0.881. A gap of 0 gives 0.5.

This rule is the Bradley-Terry model. With $r_w$ the preferred reward and $r_l$ the other one,

$$
p(\text{chosen wins}) = \sigma(r_w - r_l)
$$

The probability is a sigmoid of the gap, preferred minus rejected. Rewards 2 and 0 give $\sigma(2) \approx 0.8808$ again. The usual mistake treats the raw gap as a probability.

## Average the negative log

Training uses many pairs. Preferred rewards `[2, 0]` and rejected rewards `[1, 1]` have gaps `[1, -1]`. Then $\sigma(1) = 0.7311$ and $\sigma(-1) = 0.2689$. The negative logs are 0.3133 and 1.3133. The mean is $(0.3133 + 1.3133) / 2 = 0.8133$. The wrong-way pair costs much more.

A log probability is the logarithm of a probability, so it is zero or negative. The loss is the average of those negative logs:

$$
\text{loss} = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(z_i), \qquad z_i = r_{w,i} - r_{l,i}
$$

Average $-\log \sigma(z)$ over the pairs. A large correct gap costs almost nothing, and a large wrong gap costs about the size of the gap. Use `np.log1p(np.exp(-z))`. The two pairs give 0.8133 again.

> [!WARNING]
> When $z$ is large and negative, `np.exp(-z)` overflows. For `z < 0`, use $-z + \log(1 + e^{z})$.

The usual mistake sums the pairs. Tied rewards, gap 0, give exactly $\log 2 \approx 0.6931$.

> [!TIP]
> Test equal rewards first. The loss on a tie is $\log 2$.

## Push the chosen reward up

A gradient says how much the loss changes when you nudge one input. The derivative of $-\log \sigma(z)$ in $z$ is $\sigma(z) - 1$, and the mean puts a $1/n$ on each pair:

$$
\frac{\partial\, \text{loss}}{\partial r_{w,i}} = \frac{\sigma(z_i) - 1}{n}, \qquad
\frac{\partial\, \text{loss}}{\partial r_{l,i}} = -\frac{\sigma(z_i) - 1}{n}
$$

$\sigma(z) - 1$ is negative or zero, so raising the preferred reward lowers the loss. It is near $-1$ when the pair is ranked wrong, and near 0 when the margin is already wide. On the two pairs the preferred gradients are $[(0.7311 - 1) / 2,\ (0.2689 - 1) / 2] = [-0.1345,\ -0.3655]$. The rejected gradients flip the sign. The wrong pair gets the bigger push. The usual mistake forgets the $1/n$.

The loss sees only differences. Add one constant to both rewards for a prompt and nothing changes.

> [!KEY]
> The Bradley-Terry loss sees only the gap $r_w - r_l$. One constant added to both rewards for a prompt changes nothing.

## Reweight the reference policy

Pushing reward with no limit raises answers that score well and read badly. That failure is reward hacking. The leash is a reference policy $\pi_{\text{ref}}$, usually the supervised model. The Greek letter $\pi$ (pi, pronounced PIE) names a policy. Distance from it is a KL divergence:

$$
\text{KL}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

Add $p_i \log(p_i / q_i)$. A term with $p_i = 0$ counts as 0. For $p = [0.2, 0.3, 0.5]$ and $q = [0.1, 0.4, 0.5]$, the sum is $0.2 \log 2 + 0.3 \log 0.75 = 0.0523$. Mask with `np.where(p > 0, ...)`.

Maximise average reward minus $\beta$ times this KL. The Greek letter $\beta$ (beta, pronounced BEE-tuh) is a positive number you choose. The best policy is

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(r(x, y) / \beta)}{Z(x)}
$$

Multiply the reference probability by $\exp(r / \beta)$ and divide by $Z(x)$, the prompt's total. A softmax exponentiates scores and divides by their total. In logs this is a softmax of `ref_logprobs + rewards / beta`.

Reference probabilities `[0.5, 0.3, 0.2]`, rewards `[1, 0, 0]`, and $\beta = 1$ give boosts `[2.7183, 1, 1]` and products `[1.3591, 0.3, 0.2]`. Then $Z = 1.8591$ and $\pi^* = [0.7311,\ 0.1614,\ 0.1076]$. A huge $\beta$ keeps $\pi^*$ on the reference. A tiny $\beta$ puts the mass on the highest reward. The usual mistake normalises the whole array instead of `axis=-1`. Summing $Z(x)$ over every sentence is out of reach, which is why PPO and DPO exist.

> [!KEY]
> The best policy reweights the reference by $\exp(r / \beta)$. A large $\beta$ stays near the reference. A small $\beta$ chases reward.

> [!WARNING]
> A term $0 \log 0$ in the KL sum is `nan` in numpy. Count that term as 0.

## Describe PPO without building it

PPO is reinforcement learning on the leash. The exercise leaves it in words.

A rollout writes answers from the current policy and a frozen reward model scores them. It repeats after every update, on samples from that policy. A value head predicts the score of a partial answer. The advantage is how much better a token turned out than that prediction, and the update pushes on it.

The clipped objective reuses one batch and caps each token's probability ratio, often to 0.8 through 1.2. The KL leash is a per-token penalty. Four models sit in memory: policy, value head, reward model, and reference. The usual mistake treats the clip as a decoding rule. It limits the training update.

## Train from the implicit reward

In words, the implicit reward of an answer is $\beta$ times how far the training policy's log probability sits above the reference log probability. The same prompt adds one extra piece to every answer, and that piece cancels in a comparison. That is why Bradley-Terry only needed gaps.

Solving the best-policy formula for the reward gives

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

The second term depends only on the prompt, so the gap drops it. The implicit reward is

$$
\hat{r} = \beta \left(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\right)
$$

In code, `beta * (policy_logps - ref_logps)`. DPO is Bradley-Terry on these rewards:

$$
\text{DPO loss} = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(\hat{r}_{w,i} - \hat{r}_{l,i})
$$

Training keeps the policy and a frozen reference. It does not sample, score with a reward model, or fit a value head. Because $\hat{r}$ moves by $\beta$ per unit of log probability, the gradient is

$$
\frac{\partial\, \text{loss}}{\partial \log \pi_\theta(y_{w,i})} = \beta \, \frac{\sigma(u_i) - 1}{n}, \qquad u_i = \hat{r}_{w,i} - \hat{r}_{l,i}
$$

The rejected answer flips the sign. That factor is a per-example weight: near $\beta / n$ when the pair is still wrong, and near 0 when the margin is already wide.

Take $\beta = 0.1$, policy logs $-1$ and $-3$, and reference logs $-2$ for both. Implicit rewards are $0.1$ and $-0.1$. The gap is $0.2$, $\sigma(0.2) = 0.5498$, and the loss is $0.5981$. The preferred gradient is $0.1 \times (0.5498 - 1) = -0.0450$. The usual mistake drops $\beta$.

> [!KEY]
> DPO is Bradley-Terry on implicit rewards, $\beta$ times a difference of log probabilities. The unknown $Z(x)$ cancels, so training skips sampling and skips a reward model.

> [!KEY]
> The DPO gradient is a per-example weight. A pair still ranked wrong is pushed by about $\beta / n$. A pair already right by a wide margin barely moves.

> [!RECAP]
> Keep these five facts in front of you while you code:
>
> - The Bradley-Terry loss is the mean of $-\log \sigma(r_w - r_l)$. Tied rewards give $\log 2$.
> - The preferred gradient is $(\sigma(z) - 1) / n$. The rejected gradient flips that sign.
> - The best policy is a softmax of `ref_logprobs + rewards / beta` on each row.
> - KL sums $p \log(p / q)$ on the last axis, and a term with $p_i = 0$ counts as 0.
> - DPO runs Bradley-Terry on the implicit rewards, and the gradient carries an extra $\beta$.

## Build it step by step

The checks call your functions in this order:

1. `bradley_terry_loss(reward_chosen, reward_rejected)` takes two `(n,)` arrays and returns a float. For `z < 0`, switch to $-z + \log(1 + e^{z})$. The usual mistake sums. Ties must give $\log 2$:

```python
def bradley_terry_loss(reward_chosen, reward_rejected):
    # mean of the stable -log sigmoid of the gap
    return loss
```

2. `bradley_terry_grad(reward_chosen, reward_rejected)` returns two `(n,)` arrays. For `z < 0`, the sigmoid is `exp(z) / (1 + exp(z))`. The helper `_sigmoid` already does this. The usual mistake forgets `1 / n`:

```python
def bradley_terry_grad(reward_chosen, reward_rejected):
    # (sigmoid(z) - 1) / n, and the negation
    return grad_chosen, grad_rejected
```

3. `optimal_policy(ref_logprobs, rewards, beta)` returns row probabilities. Subtract the max and sum with `axis=-1` and `keepdims=True`. The usual mistake normalises the whole array:

```python
def optimal_policy(ref_logprobs, rewards, beta):
    # softmax of ref_logprobs + rewards / beta
    return probs
```

4. `kl_divergence(p, q)` sums on the last axis. The usual mistake lets `0 * log(0)` become `nan`:

```python
def kl_divergence(p, q):
    # mask p == 0, then sum p * log(p / q)
    return kl
```

5. `implicit_reward(policy_logps, ref_logps, beta)` is elementwise:

```python
def implicit_reward(policy_logps, ref_logps, beta):
    # beta * (policy_logps - ref_logps)
    return reward
```

6. `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` passes the two implicit rewards to `bradley_terry_loss`:

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    # bradley_terry_loss on the implicit rewards
    return loss
```

7. `dpo_grad` takes the same arguments and returns two `(n,)` arrays. Multiply `bradley_terry_grad` of the implicit rewards by `beta`. The usual mistake drops `beta`. Fifty steps must lower the loss:

```python
def dpo_grad(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    # beta times the Bradley-Terry gradient
    return grad_chosen, grad_rejected
```

## Cost and scaling

The three stages compare as follows:

| Stage | Models in memory | Generates text? | Work per pair |
|---|---|---|---|
| Reward model | 1 | No | One forward pass per answer |
| PPO | 4 | Yes, every round | Generation, then scoring |
| DPO | 2 | No | One forward and backward pass per answer |

Generation is sequential, so PPO spends most of its time there. DPO costs about as much as supervised fine-tuning on twice as many answers.

## Where this breaks

A reward model is wrong somewhere. PPO raises answers that score well and read badly: repetitive, flattering, or aimed at the format. The KL leash slows that, and training can still find it. People retrain the reward model on fresh comparisons and stop early.

Both methods learn the preferences they were given, including blind spots, and prompts outside that set have no signal. A small $\beta$ makes output nearly deterministic. DPO can grow the gap by lowering the preferred answer a little and the rejected answer a lot. A small supervised loss on the preferred answers holds them up.

## The exercise

`bradley_terry_loss` and `bradley_terry_grad` are the reward loss and its gradient. `optimal_policy(ref_logprobs, rewards, beta)` is $\pi^*$ on a short list of answers. `kl_divergence(p, q)` is the leash. `implicit_reward(policy_logps, ref_logps, beta)` is the DPO reward. `dpo_loss` and `dpo_grad` take policy and reference log probabilities for the preferred and rejected answers.

The checks cover tied rewards, wide gaps, finite-difference gradients, and the $\beta$ limits of `optimal_policy`. A shared shift of the reference logs leaves `dpo_loss` unchanged. Fifty DPO steps widen the gap and lower the loss.
