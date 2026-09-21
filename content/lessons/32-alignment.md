---
title: "Alignment: reward models, PPO, and DPO"
slug: 32-alignment
part: "Part 6 — Large language models"
summary: The Bradley-Terry reward loss, the KL-regularised policy optimum, PPO's machinery, and DPO's route to the same optimum without it.
minutes: 60
gpu: false
objectives:
  - Derive the reward-model loss from the Bradley-Terry model of pairwise preference, and state why the reward is identified only up to a per-prompt constant.
  - State the closed-form optimum of the KL-regularised policy objective, and explain the role each term plays.
  - Describe PPO's clipped surrogate, value head, and rollout loop, without implementing them.
  - Derive the DPO loss by substituting the optimal policy's implied reward into the Bradley-Terry loss, and read its gradient as a per-example weight.
prereqs:
  - 31-parameter-efficient-tuning
exercise: 32-alignment
---

# Alignment: reward models, PPO, and DPO

Pretraining fits $p(x_{t+1} \mid x_{\leq t})$ to the distribution of a text
corpus, and supervised fine-tuning repeats the same objective on a narrower
set of instruction-response pairs. Both stages maximize the likelihood of the
next token under a fixed target. Neither ever sees a preference. Likelihood
can't tell a comprehensive answer from a curt one that starts the same way,
because a maximum-likelihood target is a single string per prompt, weighted
only by how often something like it appeared in training.

Supervised fine-tuning narrows the target to whatever a human demonstrator
wrote, which helps, but imitation inherits every limit of the demonstrations:
a demonstrator can show you one good answer, not that it's better than a
hundred others the model might generate, or by how much. What's easier for a
human is a different task: given two responses to the same prompt, say which
one is better. That comparison is cheaper to collect and more reliable, and,
at scale, is the empirical basis for the rest of this lesson, which builds
the reward model that turns comparisons into a scalar objective, the
KL-regularized policy problem that objective poses, and two algorithms — PPO
and DPO — that solve it.

## What you already know

- Supervised fine-tuning minimizes token-level cross-entropy against a fixed
  target sequence (lesson 27).
- The KL divergence $\mathrm{KL}(p \Vert q) = \mathbb{E}_p[\log(p/q)]$ is
  non-negative, and zero only when $p = q$ almost everywhere.
- LoRA and other parameter-efficient methods make it cheap to hold several
  fine-tuned variants of one base model in memory at once (lesson 31).
- The logistic function $\sigma(z) = 1/(1 + e^{-z})$ has derivative
  $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
- A constrained maximum can be found by setting the gradient of its
  Lagrangian to zero.

## The Bradley-Terry reward model

Assume a scalar reward function $r(x, y)$ scoring how good a response $y$ is
to a prompt $x$, and a dataset of comparisons: for each $x$, a chosen
response $y_w$ that a human preferred over a rejected response $y_l$. The
Bradley-Terry model turns a reward into a probability of that comparison:

$$
p(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big) .
$$

A reward difference of zero gives even odds; a gap of a few units makes the
preference near certain. Fitting $r_\theta$ by maximum likelihood gives the
**reward-model loss**, $\mathcal{L}_{\text{RM}}(\theta) = -\log
\sigma(z)$ for $z = r_\theta(x, y_w) - r_\theta(x, y_l)$. The chain rule
through $-\log \sigma(z)$ gives $d\mathcal{L}_{\text{RM}}/dz = \sigma(z) - 1
= -\sigma(-z)$, so

$$
\nabla_\theta \mathcal{L}_{\text{RM}} = -\sigma(-z)\big(\nabla_\theta r_\theta(x, y_w) - \nabla_\theta r_\theta(x, y_l)\big) .
$$

The weight $\sigma(-z)$ is large exactly when the model still ranks the pair
wrong or is unsure, and shrinks toward zero once $r_\theta$ already separates
them by a wide margin.

Because the loss depends on $r_\theta$ only through the difference
$r_\theta(x, y_w) - r_\theta(x, y_l)$, adding any function $c(x)$ of the
prompt alone to $r_\theta$ leaves every training loss unchanged. The reward
model is identified only up to a per-prompt constant: nothing in the
comparisons pins down the absolute scale of $r$ for a given prompt, only the
ordering and the relative gaps. That slack is not a defect — it's exactly
the freedom DPO exploits later in this lesson.

## The KL-regularised objective

A reward model alone isn't a policy. Optimizing $r_\theta$ directly over $y$
throws away everything the base language model knows about producing fluent
text, and a reward model trained on finitely many comparisons is trivial to
fool once you search hard enough over its input space. The fix keeps the
policy close to a reference policy $\pi_{\text{ref}}$ (the supervised
fine-tuned model) while nudging it toward higher reward:

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] - \beta \, \mathrm{KL}\big(\pi(\cdot \mid x) \Vert \pi_{\text{ref}}(\cdot \mid x)\big) .
$$

The KL term is a leash: without it the optimal $\pi$ puts all its mass on
$\operatorname*{arg\,max}_y r(x, y)$, the reward-hacking failure the
reference model is meant to prevent. $\beta$ trades reward against fidelity
to $\pi_{\text{ref}}$.

Per prompt, this objective is concave in $\pi$ — the expected reward is
linear and $-\beta\,\mathrm{KL}$ is concave — so its unique stationary point
is the global maximizer. Setting the gradient of its Lagrangian to zero and
solving for $\pi(y)$ gives

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right),
\qquad
Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right) .
$$

Two limits are worth checking against the formula. As $\beta \to \infty$,
$r(x,y)/\beta \to 0$ for every $y$, so $\pi^* \to \pi_{\text{ref}}$: an
infinite leash means the policy never moves. As $\beta \to 0^{+}$, the
$\exp(r(y)/\beta)$ factor for the largest reward dominates every other term
by a diverging ratio, so $\pi^*$ collapses onto
$\operatorname*{arg\,max}_y r(x, y)$: no leash recovers pure reward
maximization.

$\pi^*$ isn't directly usable — computing $Z(x)$ means summing over every
possible response, which a language model can't enumerate. Two algorithms
turn this closed form into something trainable. PPO optimizes the original
objective by sampling. DPO uses the closed form itself.

## PPO: reinforcement learning against a reward model

Proximal Policy Optimization treats the KL-regularized objective as a
reinforcement learning problem, folding the KL penalty into a per-token
reward: the sequence-level reward at the final token, plus a running
per-token cost $-\beta \log(\pi_\theta / \pi_{\text{ref}})$. This needs
machinery a closed-form update doesn't:

- **A rollout loop.** Sample full completions from the current policy
  autoregressively, then score each with the frozen reward model. This is
  expensive: generation is sequential, and every policy update needs fresh
  samples from the updated policy, because PPO is on-policy.
- **A value head.** A second output head predicts the expected future return
  from a partial sequence, trained by regression toward observed returns, to
  reduce the variance of the gradient estimator.
- **Advantage estimation.** The advantage measures how much better a sampled
  token was than the value head expected, and it's what multiplies the
  policy gradient.
- **A clipped surrogate.** With $\rho_t(\theta) = \pi_\theta(y_t \mid \cdot) /
  \pi_{\theta_{\text{old}}}(y_t \mid \cdot)$ the probability ratio against the
  policy that generated the rollout,

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta) A_t, \;\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t\big)\Big] .
$$

The clip keeps one batch of rollouts from moving $\pi_\theta$ so far that the
sampled ratios stop being informative, which matters because rollouts are
reused for several gradient steps before the next, expensive sampling round.
Every piece here exists to make on-policy sampling tractable and stable —
none of it is needed to state the objective, only to optimize it by sampling
from a policy that keeps changing.

## DPO: solving for the reward instead of sampling for it

The closed-form optimum gives a second route. Solve the $\pi^*$ equation for
$r$ instead of $\pi^*$:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) .
$$

Every optimal reward function is, up to the additive $\beta \log Z(x)$, the
log-ratio of the aligned policy to the reference policy. Substitute this
expression, evaluated at the policy $\pi_\theta$ being fit, into the
Bradley-Terry difference: $Z(x)$ depends on the prompt alone, and both
completions share the prompt, so it cancels exactly. That cancellation is
precisely the per-prompt identifiability slack from the reward model.
Plugging the surviving difference into $\mathcal{L}_{\text{RM}}$ gives the
**DPO loss**:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right) .
$$

This is a supervised loss on log-probabilities the policy already computes
by teacher forcing, with no sampling, no reward model at training time, and
no value network — training it directly optimizes the same $\pi^*$ that PPO
chases by sampling.

Write $\hat r_\theta(x, y) = \beta \log(\pi_\theta(y\mid x) /
\pi_{\text{ref}}(y \mid x))$ for the **implicit reward**, and $u = \hat
r_\theta(x, y_w) - \hat r_\theta(x, y_l)$. By the same chain rule as the
reward model,

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big) \Big[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\Big] .
$$

Descending this gradient raises $\log \pi_\theta(y_w \mid x)$ and lowers
$\log \pi_\theta(y_l \mid x)$. The per-example weight is large when the
implicit reward still has the pair backwards or too close to call, and
shrinks once the policy already prefers $y_w$ by enough margin. DPO never
computes a reward at inference time — the trained policy is the deliverable.

## Cost and scaling

Both methods start from a supervised fine-tuned model and a frozen copy of
it as $\pi_{\text{ref}}$. The difference is how many model instances a
training step needs, and whether it requires generation.

| Stage | Models in memory | Needs generation | Per-pair-or-token cost |
|---|---|---|---|
| Reward model training | 1 (reward model) | No | One forward pass per response |
| PPO | 4: policy, value head, reward model, reference | Yes, every rollout | Sequential decoding, plus reward and value forward passes |
| DPO | 2: policy, reference | No | One forward-and-backward pass per pair, for each of $y_w$ and $y_l$ |

PPO's dominant cost is generation: producing a completion token by token is
sequential and can't be parallelized the way a forward pass over a fixed
sequence can, so a PPO step spends most of its wall-clock time sampling, not
updating weights. DPO's cost per pair is two teacher-forced
forward-and-backward passes — comparable to supervised fine-tuning at twice
the batch work, independent of how long the compared responses are to
generate.

## Where this breaks

**Reward hacking.** A reward model trained on finitely many comparisons is
wrong somewhere, and PPO's optimization actively searches for whichever
policy exploits that error — repetitive, sycophantic, or format-gaming
output that scores well and reads badly. The KL penalty slows this; it
doesn't stop it, because a large enough gap in $r$ still beats the penalty
at fixed $\beta$.

**Miscalibration outside the training distribution.** Both approaches
inherit whatever bias sits in the pairwise comparisons, and neither has any
signal about a response's absolute quality outside the prompts they were
trained on.

**Collapse at low $\beta$.** The closed-form $\pi^*$ shows this directly: as
$\beta$ shrinks, the optimum sharpens toward a near-deterministic
arg max policy, showing up as reduced output diversity and increased
sensitivity to reward-model error, for either training method.

**Likelihood displacement in DPO.** DPO's gradient doesn't distinguish
between raising $y_w$'s probability and lowering $y_l$'s, so gradient
descent is free to prefer whichever is locally cheaper — which can mean
driving down the chosen response's probability along with the rejected one,
as long as the gap grows. Because $y_l$ is a real, fluent response, this can
pull down probability mass from other good responses that share its
structure, a failure mode PPO's on-policy rollouts and value baseline make
less likely, at the cost of a rollout loop, a value network, and roughly
twice the memory.

## The exercise

You implement the two losses this lesson derives, the closed-form optimal
policy, and the identity linking them. Write `bradley_terry_loss(reward_chosen,
reward_rejected)`, returning the mean of $-\log\sigma(r_w - r_l)$ over a batch
of comparisons, and `bradley_terry_grad`, its analytic gradient with respect
to both reward arguments. Write `optimal_policy(ref_logprobs, rewards,
beta)`, returning the closed-form $\pi^*$ over a finite, enumerated set of
completions per prompt, normalized along the last axis, together with
`kl_divergence(p, q)` for two discrete distributions. Write `dpo_loss` and
`dpo_grad` for the policy and reference log-probabilities of chosen and
rejected completions, and `implicit_reward(policy_logps, ref_logps, beta)`
for $\hat r_\theta$.

The checks verify each gradient against `mlcheck.finite_difference`; that
`optimal_policy` normalizes to a valid distribution and reduces to
$\pi_{\text{ref}}$ as $\beta \to \infty$ and to the reward's arg max as
$\beta \to 0$; that `dpo_loss` equals `bradley_terry_loss` applied to the two
completions' implicit rewards, the substitution this lesson derives, made
executable; that `dpo_loss` is unchanged by adding any constant to both
completions' reference log-probabilities for a shared prompt, the $Z(x)$
cancellation, checked numerically rather than taken on faith; that
`implicit_reward` recovers a planted reward up to the per-prompt constant the
Bradley-Terry model leaves undetermined; and that a handful of gradient steps
on `dpo_loss`, starting from the policy equal to the reference, raises the
chosen completion's log-probability relative to the rejected one.
