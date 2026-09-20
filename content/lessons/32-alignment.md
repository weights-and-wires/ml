---
title: "Alignment: reward models, PPO, and DPO"
slug: 32-alignment
part: "Part 6 — Large language models"
summary: The Bradley-Terry reward loss, the KL-regularised optimum derived in full, PPO's machinery, and DPO's route to the same optimum without it.
minutes: 70
gpu: false
objectives:
  - Derive the reward-model loss from the Bradley-Terry model of pairwise preference, and state why the reward is identified only up to a per-prompt constant.
  - Derive the closed-form optimum of the KL-regularised policy objective by pointwise maximisation of its Lagrangian.
  - Describe PPO's clipped surrogate and the value head, advantage estimates, and rollout loop it needs, without implementing them.
  - Derive the DPO loss by inverting the optimal policy for the reward and substituting into the Bradley-Terry loss, and read its gradient as a per-example weight.
prereqs:
  - 31-parameter-efficient-tuning
exercise: 32-alignment
---

# Alignment: reward models, PPO, and DPO

Pretraining fits $p(x_{t+1} \mid x_{\leq t})$ to the distribution of some
enormous text corpus, and supervised fine-tuning repeats the same objective on
a narrower set of instruction-response pairs. Both stages maximise the
likelihood of the next token under a fixed target. Neither objective ever sees
a preference. Likelihood cannot tell a comprehensive answer from a curt one
that happens to start the same way, or a helpful refusal from a lazy one,
because a maximum-likelihood target is a single string per prompt, weighted
only by how often something like it appeared in training. Two responses can
both be plausible continuations of a prompt while one is clearly better, and
the next-token objective has no term that expresses that.

Supervised fine-tuning narrows the target distribution to whatever a human
demonstrator wrote, which helps: the model now imitates instruction-following
behaviour instead of generic web text. But imitation inherits every limit of
the demonstrations. A demonstrator can show you one good answer per prompt;
they cannot show you that it is better than a hundred other answers the model
might generate, or by how much, and the fine-tuned model is bounded by
demonstrator skill and coverage. What is comparatively easy for a human is a
different task: given two responses to the same prompt, say which one is
better. That comparison is cheaper to collect, more reliable, and, applied at
scale, is the entire empirical basis for the rest of this lesson. This lesson
builds the reward model that turns comparisons into a scalar objective, the
KL-regularised policy optimisation problem that objective poses, and two
algorithms — PPO and DPO — that solve it.

## What you already know

- Supervised fine-tuning minimises token-level cross-entropy against a fixed
  target sequence (lesson 27).
- The KL divergence $\mathrm{KL}(p \Vert q) = \mathbb{E}_p[\log(p/q)]$ is
  non-negative and zero only when $p = q$ almost everywhere.
- LoRA and other parameter-efficient methods make it cheap to hold several
  fine-tuned variants of one base model in memory at once (lesson 31).
- The logistic function $\sigma(z) = 1/(1 + e^{-z})$ has derivative
  $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
- A constrained maximisation is solved by setting the gradient of its
  Lagrangian to zero and reading off the stationarity condition.

## The Bradley-Terry reward model

Assume a scalar reward function $r(x, y)$ scoring how good a response $y$ is
to a prompt $x$, and a dataset of comparisons: for each $x$, a chosen response
$y_w$ that a human preferred over a rejected response $y_l$. The Bradley-Terry
model turns a reward into a probability of that comparison:

$$
p(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big) .
$$

A reward difference of zero gives even odds; a gap of a few units makes the
preference near-certain. Fitting $r_\theta$ by maximum likelihood on the
observed comparisons gives the **reward-model loss**:

$$
\mathcal{L}_{\text{RM}}(\theta) = -\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big) .
$$

This is logistic regression on reward differences, and its gradient has the
familiar shape. Writing $z = r_\theta(x, y_w) - r_\theta(x, y_l)$, the chain
rule through $-\log\sigma(z)$ gives $d\mathcal{L}_{\text{RM}}/dz =
\sigma(z) - 1 = -\sigma(-z)$, so

$$
\nabla_\theta \mathcal{L}_{\text{RM}} = -\sigma(-z)\big(\nabla_\theta r_\theta(x, y_w) - \nabla_\theta r_\theta(x, y_l)\big) .
$$

The weight $\sigma(-z)$ is large exactly when the model still ranks the
pair wrong or is unsure, and shrinks toward zero once $r_\theta$ already
separates them by a wide margin — a comparison the model already gets right
contributes almost nothing further.

Because the loss depends on $r_\theta$ only through the difference
$r_\theta(x, y_w) - r_\theta(x, y_l)$, adding any function $c(x)$ of the
prompt alone to $r_\theta$ leaves every training loss unchanged: $r'(x, y) =
r(x, y) + c(x)$ satisfies $r'(x,y_w) - r'(x,y_l) = r(x,y_w) - r(x,y_l)$
identically. The reward model is identified only up to a per-prompt constant.
Nothing in the comparisons pins down the absolute scale of $r$ for a given
prompt, only the ordering and the relative gaps between responses to it. That
slack is not a defect to patch; it is exactly the freedom DPO will exploit
later in this lesson.

## The KL-regularised objective

A reward model alone is not a policy. Optimising $r_\theta$ directly over
$y$ — take the argmax response the reward model prefers — throws away
everything the base language model knows about producing fluent, on-topic
text, and a reward model trained on a finite comparison set is trivial to
fool once you search hard enough over its input space; it has no signal
outside the region the comparisons covered. The fix is to keep the policy
close to a reference policy $\pi_{\text{ref}}$ (the supervised fine-tuned
model) while nudging it toward higher reward:

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] - \beta \, \mathrm{KL}\big(\pi(\cdot \mid x) \Vert \pi_{\text{ref}}(\cdot \mid x)\big) .
$$

The KL term is a leash, not an afterthought: without it the optimal $\pi$
puts all its mass on $\operatorname*{arg\,max}_y r(x, y)$, which is precisely
the reward-hacking failure mode the reference model is meant to prevent. The
coefficient $\beta$ trades off reward against fidelity to $\pi_{\text{ref}}$,
and it is the only knob in this objective.

### Solving it in closed form

Fix a prompt $x$ and treat $\pi(\cdot \mid x)$ as a free distribution over the
response space, subject to $\sum_y \pi(y \mid x) = 1$. Writing out the
objective per prompt,

$$
J(\pi) = \sum_y \pi(y) \, r(y) - \beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} ,
$$

dropping the $x$-conditioning from the notation. Introduce a Lagrange
multiplier $\lambda$ for the normalisation constraint and differentiate
pointwise with respect to $\pi(y)$ for a fixed $y$:

$$
\frac{\partial}{\partial \pi(y)} \left[ J(\pi) - \lambda \Big(\sum_{y'} \pi(y') - 1\Big) \right]
= r(y) - \beta \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} - \beta - \lambda = 0 .
$$

Solving for $\pi(y)$,

$$
\log \frac{\pi(y)}{\pi_{\text{ref}}(y)} = \frac{r(y)}{\beta} - 1 - \frac{\lambda}{\beta}
\quad\Longrightarrow\quad
\pi(y) = \pi_{\text{ref}}(y) \exp\!\left(\frac{r(y)}{\beta}\right) \exp\!\left(-1 - \frac{\lambda}{\beta}\right) .
$$

The last factor does not depend on $y$, so call it $1/Z(x)$ and fix it by the
normalisation constraint. The stationary point is

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right),
\qquad
Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right) .
$$

$J$ is concave in $\pi$ (the KL term is convex, so $-\beta\,\mathrm{KL}$ is
concave, and the expected reward is linear in $\pi$), so this stationary
point is the global maximiser. The same answer falls out without Lagrange
multipliers by completing the square inside the KL: $J(\pi) = -\beta\,
\mathrm{KL}(\pi \Vert q) + \beta \log Z(x)$ for $q(y) = \pi_{\text{ref}}(y)
\exp(r(y)/\beta)/Z(x)$, and $\mathrm{KL}(\pi \Vert q) \geq 0$ with equality
iff $\pi = q$, so $\pi^* = q$ maximises $J$.

Two limits are worth checking directly against the formula. As $\beta \to
\infty$, $r(x,y)/\beta \to 0$ for every $y$, so $\pi^* \to \pi_{\text{ref}}$:
an infinite leash means the policy never moves. As $\beta \to 0^+$, the
$\exp(r(y)/\beta)$ factor for the largest reward dominates every other term
by a ratio that diverges, so $\pi^*$ collapses onto
$\operatorname*{arg\,max}_y r(x, y)$: no leash recovers pure reward
maximisation, and the reward-hacking failure returns.

$\pi^*$ is not directly usable: computing $Z(x)$ means summing over every
possible response, which is exactly what a language model cannot enumerate.
Two algorithms turn this closed form into something trainable. PPO optimises
the original objective by sampling. DPO uses the closed form itself.

## PPO: reinforcement learning against a reward model

Proximal Policy Optimization treats the KL-regularised objective as a
reinforcement learning problem, folding the KL penalty into a per-token
reward: $\tilde r(x, y_t) = r(x, y) \mathbb{1}[t = T] - \beta \log
\big(\pi_\theta(y_t \mid x, y_{<t}) / \pi_{\text{ref}}(y_t \mid x,
y_{<t})\big)$, the sequence-level reward at the final token plus a running
per-token KL cost. This needs machinery that a closed-form policy update does
not:

- **A rollout loop.** Sample full completions from the current policy
  $\pi_\theta$, autoregressively, one token at a time, then score each
  completed sequence with the frozen reward model. This is the expensive
  part: generation is sequential, and every policy update needs fresh
  samples from the updated policy, because PPO is an on-policy method.
- **A value head.** A second output head, usually sharing the transformer
  trunk, predicts $V_\phi(s_t)$, the expected future return from a partial
  sequence. It is trained by regression toward the observed returns and
  serves only to reduce the variance of what comes next.
- **Advantage estimation.** The advantage $A_t = \tilde r(x, y_t) + \gamma
  V_\phi(s_{t+1}) - V_\phi(s_t)$, typically smoothed across a whole
  trajectory by generalised advantage estimation, measures how much better
  the sampled token was than the value head expected, and it is what
  actually multiplies the policy gradient.
- **The clipped surrogate.** With $\rho_t(\theta) = \pi_\theta(y_t \mid
  \cdot) / \pi_{\theta_{\text{old}}}(y_t \mid \cdot)$ the probability ratio
  against the policy that generated the rollout,

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta) A_t, \;\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t\big)\Big] .
$$

The clip keeps a single batch of rollouts from producing an update so large
that $\pi_\theta$ leaves the trust region where the sampled ratios are still
informative, which matters because rollouts are reused for several gradient
steps before the next, expensive sampling round.

Every one of these pieces exists to make on-policy sampling tractable and
stable: the value head and advantage estimate reduce the variance of a
policy-gradient estimator that would otherwise be too noisy to use, and the
clip bounds the damage from stale importance weights. None of it is needed to
state the objective — only to optimise it by sampling from a policy that
keeps changing.

## DPO: solving for the reward instead of sampling for it

The closed-form optimum gives a second route. Solve the $\pi^*$ equation for
$r$ instead of for $\pi^*$:

$$
\exp\!\left(\frac{r(x,y)}{\beta}\right) = \frac{\pi^*(y\mid x)\, Z(x)}{\pi_{\text{ref}}(y \mid x)}
\quad\Longrightarrow\quad
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) .
$$

Every optimal reward function is, up to the additive constant $\beta \log
Z(x)$, the log-ratio of the aligned policy to the reference policy. Substitute
this expression for $r$, evaluated at the optimal policy $\pi_\theta$ you are
now fitting directly, into the Bradley-Terry difference:

$$
r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{=\,0} .
$$

$Z(x)$ depends on the prompt alone, and both completions share the same
prompt, so it cancels exactly between them — the per-prompt identifiability
slack from the reward model is precisely what makes this substitution work
without ever computing $Z(x)$. Plugging the surviving difference into
$\mathcal{L}_{\text{RM}}$ gives the **DPO loss**:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right) .
$$

This is a supervised loss on log-probabilities the policy already computes by
teacher forcing over $y_w$ and $y_l$, with no sampling, no reward model at
training time, and no value network. Training it directly optimises the same
$\pi^*$ that PPO chases by sampling.

### The gradient

Write $\hat r_\theta(x, y) = \beta \log\big(\pi_\theta(y\mid x) /
\pi_{\text{ref}}(y \mid x)\big)$ for the **implicit reward** — the quantity
DPO trains without ever naming it as one — and $u = \hat r_\theta(x, y_w) -
\hat r_\theta(x, y_l)$. The loss is $-\log\sigma(u)$, so by the same chain
rule used for the reward model, $d\mathcal{L}_{\text{DPO}}/du = -\sigma(-u)$,
and since $\hat r_\theta$ is the only $\theta$-dependent piece,

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big) \Big[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\Big] .
$$

Descending this gradient increases $\log \pi_\theta(y_w \mid x)$ and decreases
$\log \pi_\theta(y_l \mid x)$, exactly as the reward model's gradient pushed
$r_\theta(y_w)$ up and $r_\theta(y_l)$ down. The per-example weight
$\sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$ reads the same way too:
it is large — near one — when the implicit reward still has the pair
backwards or too close to call, and it shrinks toward zero once the policy
already assigns $y_w$ enough more probability mass than $y_l$, relative to
the reference model, to be confident. DPO never computes a reward at
inference time; the trained policy $\pi_\theta$ is the deliverable, and
$\hat r_\theta$ is only ever used through this gradient during training.

### What DPO gives up

DPO trades PPO's machinery for a fixed, offline dataset of comparisons, and
that trade has a cost. PPO samples from the current policy at every step, so
the reward model is queried exactly where the policy puts its
probability mass, and the policy's own drift is what generates the next
batch of training signal. DPO's comparisons are fixed once collected; if
several rounds of gradient steps move $\pi_\theta$ far from whatever
generated $y_w$ and $y_l$, the loss keeps optimising against a fixed
snapshot that no longer describes the policy's actual output distribution,
and nothing in the objective notices. This shows up in practice as
**likelihood displacement**: because the loss only depends on the difference
$\log\pi_\theta(y_w) - \log\pi_\theta(y_l)$, it is satisfied by raising
$y_w$'s probability, by lowering $y_l$'s, or by some mixture, and gradient
descent is free to prefer whichever direction is locally cheaper — which can
mean driving down the probability of the chosen response along with the
rejected one, as long as the gap grows. PPO's on-policy rollouts and value
baseline make this failure mode less likely, at the cost of a rollout loop, a
value network, and roughly twice the memory: PPO holds the policy, the value
head, the reward model, and the reference model simultaneously, where DPO
holds only the policy and the reference model.

## Cost and scaling

Both methods start from a supervised fine-tuned model and a fixed
$\pi_{\text{ref}}$, a frozen copy of it. The difference is how many model
instances a training step needs, and whether that step requires generation.

| Stage | Models in memory | Needs generation | Per-pair-or-token cost |
|---|---|---|---|
| Reward model training | 1 (reward model) | No | One forward pass per response |
| PPO | 4: policy, value head, reward model, reference | Yes, every rollout | Sequential decoding per rollout, plus forward passes for reward and value |
| DPO | 2: policy, reference | No | One forward-and-backward pass per pair, for each of $y_w$ and $y_l$ |

PPO's dominant cost is generation: producing a completion token by token is
inherently sequential and cannot be parallelised across positions the way a
forward pass over a fixed sequence can, so a single PPO step spends most of
its wall-clock time sampling, not updating weights. DPO never generates
during training, so its cost per pair is two teacher-forced forward-and-backward
passes — comparable to ordinary supervised fine-tuning at twice the batch
work — independent of how long the responses being compared happen to be to
generate.

## Where this breaks

**Reward hacking.** A reward model trained on finitely many comparisons is
exact nowhere and wrong somewhere, and PPO's optimisation actively searches
for whichever policy exploits that error, producing responses that score
well and read badly — repetitive, sycophantic, or exploiting length or
formatting artifacts in the reward model's training data. The KL penalty
slows this; it does not stop it, because a large enough gap in $r$ still
beats the penalty at fixed $\beta$.

**Reward and implicit-reward miscalibration.** Both approaches inherit
whatever bias sits in the pairwise comparisons, and neither has any signal
about the absolute quality of a response outside the training distribution
of prompts; extrapolation is unconstrained by the objective.

**Collapse at low $\beta$.** The closed-form $\pi^*$ shows this directly: as
$\beta$ shrinks, the optimum sharpens toward a near-deterministic
arg max policy, which shows up as reduced output diversity and increased
sensitivity to reward-model error, for either training method.

**Likelihood displacement in DPO.** As above, DPO's gradient does not
distinguish between raising $y_w$'s probability and lowering $y_l$'s, and
because $y_l$ is a real, fluent, on-distribution response, driving its
probability toward zero can pull down probability mass from other good
responses that share structure with it — a failure mode with no analogue in
PPO's per-token KL-regularised rollout.

## The exercise

You implement the two losses this lesson derives, the closed-form optimal
policy, and the identity linking them. Write `bradley_terry_loss(reward_chosen,
reward_rejected)`, returning the mean of $-\log\sigma(r_w - r_l)$ over a batch
of comparisons, and `bradley_terry_grad`, its analytic gradient with respect
to both reward arguments. Write `optimal_policy(ref_logprobs, rewards, beta)`,
which returns the closed-form $\pi^*$ over a finite, enumerated set of
completions per prompt, normalised along the last axis, together with
`kl_divergence(p, q)` for two discrete distributions. Write `dpo_loss` and
`dpo_grad` for the policy and reference log-probabilities of chosen and
rejected completions, and `implicit_reward(policy_logps, ref_logps, beta)` for
$\hat r_\theta$.

The checks verify each gradient against `mlcheck.finite_difference`; that
`optimal_policy` normalises to a valid distribution and reduces to
$\pi_{\text{ref}}$ as $\beta \to \infty$ and to the reward's arg max as
$\beta \to 0$; that `dpo_loss` equals `bradley_terry_loss` applied to the two
completions' implicit rewards, which is the substitution this lesson derives,
made executable; that `dpo_loss` is unchanged by adding any constant to both
completions' reference log-probabilities for a shared prompt, which is the
$Z(x)$ cancellation, checked numerically rather than taken on faith; that
`implicit_reward` recovers a planted reward up to the per-prompt constant the
Bradley-Terry model leaves undetermined; and that a handful of gradient steps
on `dpo_loss`, starting from the policy equal to the reference, raises the
chosen completion's log-probability relative to the rejected one.
