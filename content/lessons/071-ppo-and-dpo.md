---
title: "PPO and DPO: two ways to reach the leashed optimum"
slug: 071-ppo-and-dpo
part: "Part 7 — Large language models"
summary: PPO reaches the best leashed policy by sampling, an advantage, and a clip. DPO reads a reward off the policy and trains on pairs directly.
minutes: 80
gpu: false
objectives:
  - Walk through one round of PPO, from sample and score to baseline, clipped update, and repeat.
  - Compute an advantage and a clipped objective by hand, and explain what the clip does for positive and negative advantages.
  - Solve the best-policy formula for the reward, and show with numbers why the unknown normaliser cancels in a comparison.
  - Compute the DPO implicit reward, loss, and gradient, and read the gradient as a per-pair weight that fades as a pair is ranked right.
  - State how the per-answer advantage relates to generalised advantage estimation, and show that with one terminal reward and gamma = lambda = 1 it is the reward minus the value guess at every token.
  - Compare the two methods by the models they keep in memory and whether they generate text during training.
prereqs:
  - 070-reward-models-and-the-kl-leash
  - 064-next-token-loss-and-safe-steps
exercise: 071-ppo-and-dpo
---

# PPO and DPO: two ways to reach the leashed optimum

The [previous chapter on reward models and the KL leash](/lessons/070-reward-models-and-the-kl-leash) found the best policy for reward minus a leash: the reference model, reweighted by $e^{r/\beta}$ and divided by a total $Z$. You cannot build that policy directly, because $Z$ adds up every possible answer, and for a real language model there are about $10^{470}$ answers of 100 tokens. You do have two things, though: a reward model, and preference pairs such as "the helpful mortgage answer beats the padded one".

So the question is how to move toward the best policy with only those. Two obvious tries break. Fine-tune on the highest-scoring sample, and you ignore the losers and ignore how much better the winner was. Push up the reward of every sampled answer by plain gradient ascent, and one lucky sample with a huge score drags the whole model along a direction that was true only for that sample.

This chapter gives two answers. Proximal policy optimisation (PPO) samples answers, scores them, and nudges the policy toward the better-than-expected ones, with a clip that limits each step. Direct preference optimisation (DPO) asks whether the sampling can be skipped entirely: if a reward gives you the best policy, the best policy should give you the reward back. Both aim at the same target, and they differ in what they cost.

## What you already know

- A policy gives every answer a probability, and a reference policy is the supervised model it must stay near.
- A reward model gives each answer one number, and only gaps between rewards matter.
- The Bradley-Terry loss is the mean of $-\log \sigma(r_w - r_l)$, and its gradient with respect to the chosen reward is $(\sigma(z) - 1)/n$.
- The best policy on a leash is the reference multiplied by $e^{r/\beta}$ and divided by $Z(x)$.
- A log probability of an answer is the sum of the log probabilities of its tokens.

## Route one: PPO, sample and compare

Figure 1 shows one round. The policy writes answers to a batch of prompts, which is called a rollout. The frozen reward model scores each answer, and the score is lowered by a leash penalty. Then each answer is compared with what the policy usually earns for that prompt, and the policy is nudged toward answers that did better than usual and away from answers that did worse. The loop repeats from fresh samples, because the nudge changed the policy and answers written by the old policy no longer describe it.

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="One round of PPO: prompt, policy writes answers, reward model scores them, advantage compares with a baseline, clipped update, repeat">
<defs><marker id="c054-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="10.0" y="84.0" width="96.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="58.0" y="113.0" text-anchor="middle" class="fig-small">1 prompt</text>
<rect x="130.0" y="84.0" width="96.0" height="50.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="178.0" y="105.5" text-anchor="middle" class="fig-small">2 policy</text>
<text x="178.0" y="120.5" text-anchor="middle" class="fig-small">writes answers</text>
<rect x="250.0" y="84.0" width="96.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="298.0" y="105.5" text-anchor="middle" class="fig-small">3 reward model</text>
<text x="298.0" y="120.5" text-anchor="middle" class="fig-small">scores them</text>
<rect x="370.0" y="84.0" width="96.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="418.0" y="98.0" text-anchor="middle" class="fig-small">4 advantage:</text>
<text x="418.0" y="113.0" text-anchor="middle" class="fig-small">better than</text>
<text x="418.0" y="128.0" text-anchor="middle" class="fig-small">expected?</text>
<rect x="490.0" y="84.0" width="96.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="538.0" y="105.5" text-anchor="middle" class="fig-small">5 clipped</text>
<text x="538.0" y="120.5" text-anchor="middle" class="fig-small">update</text>
<line x1="108.0" y1="109.0" x2="127.0" y2="109.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<line x1="228.0" y1="109.0" x2="247.0" y2="109.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<line x1="348.0" y1="109.0" x2="367.0" y2="109.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<line x1="468.0" y1="109.0" x2="487.0" y2="109.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<rect x="250.0" y="190.0" width="96.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="298.0" y="211.5" text-anchor="middle" class="fig-small">reference</text>
<text x="298.0" y="226.5" text-anchor="middle" class="fig-small">model</text>
<line x1="298.0" y1="189.0" x2="298.0" y2="137.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<text x="306.0" y="168.0" text-anchor="start" class="fig-small">KL penalty</text>
<rect x="370.0" y="190.0" width="96.0" height="50.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="418.0" y="211.5" text-anchor="middle" class="fig-small">value head</text>
<text x="418.0" y="226.5" text-anchor="middle" class="fig-small">(baseline)</text>
<line x1="418.0" y1="189.0" x2="418.0" y2="137.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<text x="426.0" y="168.0" text-anchor="start" class="fig-small">expected reward</text>
<polyline points="538.0,82.0 538.0,58.0 178.0,58.0" fill="none" stroke-width="2" class="stroke-muted"/>
<line x1="178.0" y1="58.0" x2="178.0" y2="81.0" class="stroke-muted" marker-end="url(#c054-arrow)" stroke-width="2"/>
<text x="310.0" y="50.0" text-anchor="middle" class="fig-small">repeat with the updated policy</text>
<rect x="20.0" y="262.0" width="14.0" height="14.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="40.0" y="274.0" text-anchor="start" class="fig-small">model that is trained</text>
<rect x="190.0" y="262.0" width="14.0" height="14.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="210.0" y="274.0" text-anchor="start" class="fig-small">model that stays frozen</text>
</svg>
Figure 1. One round of PPO. The policy writes answers, the frozen reward model scores them, the reference model adds a per-token KL penalty, and the value head supplies the expected reward so that the advantage asks "better than expected?". The clipped update changes the policy a little, and the loop repeats with fresh samples from the updated policy.
```

Each stage fixes a specific failure, and the next sections take them in turn.

### The leash, one token at a time

The KL leash is an average over the policy's own answers, and the rollout supplies exactly those answers. So PPO charges every sampled token for how much more likely the policy finds it than the reference does: the token's reward drops by $\beta\,(\log \pi(\text{token}) - \log \pi_{\text{ref}}(\text{token}))$.

Suppose the policy gives a token probability 0.28 and the reference gives 0.10. The log ratio is $\ln 2.8 = 1.03$, so with $\beta = 0.1$ the token pays $0.103$. A token both models agree on pays nothing. Averaged over many answers, this charge estimates the KL divergence from the reference.

### Better than expected: the advantage

Take four sampled answers to the mortgage prompt, scored by the reward model:

| Sample | Reward | Value head's guess | Reward − guess |
|---|---|---|---|
| 1 | 3.3 | 2.8 | +0.5 |
| 2 | 3.1 | 2.6 | +0.5 |
| 3 | 2.4 | 2.9 | −0.5 |
| 4 | 2.6 | 3.1 | −0.5 |

All four rewards are positive, so pushing up each sampled answer in proportion to its reward lifts the bad ones along with the good ones. The update needs to know whether an answer was better than expected. The expectation comes from a value head, a small output on top of the policy's network that learns to predict the reward the policy earns. It trains alongside the policy against the rewards the rollouts actually earn, so its guess tracks the policy as it changes. Subtracting the guess leaves the last column: samples 1 and 2 beat it, and samples 3 and 4 fall short.

Because the reward scale is arbitrary, the gaps are rescaled to zero mean and unit standard deviation. Here the mean is 0 and the standard deviation is 0.5, so the advantages are $[1, 1, -1, -1]$:

$$
A_i = \frac{(r_i - v_i) - \text{mean}}{\text{std}}
$$

Read it aloud: reward minus the value head's guess, then recentred and rescaled over the batch. If every sample matches its guess, the standard deviation is 0 and the advantages must be 0, not `nan`, so add a tiny constant to the denominator. In numpy: `raw = rewards - values`, then `(raw - raw.mean()) / (raw.std() + 1e-8)`.

### How this relates to a full PPO trainer

This chapter treats the advantage as given for each answer. A real PPO trainer fits the value head by regression: it nudges the head's guess toward the return the rollouts earned. It also spreads credit over tokens with *generalised advantage estimation* (GAE) instead of giving one number to a whole answer. The reinforcement learning course teaches both.

Here is how the two forms connect. Take a three-token answer with one reward, 3.5, at the last token, and value guesses $[2.0, 2.5, 3.0]$ for the three tokens. The value after the last token is 0. Set $\gamma = 1$ and $\lambda = 1$. GAE sums one-step errors $\delta_t = r_t + \gamma V_{t+1} - V_t$, which here are $0.5$, $0.5$, and $0.5$. Summing from each token to the end gives advantages $[1.5, 1.0, 0.5]$. Each one is $3.5$ minus that token's guess, which is the per-answer form applied at every token. With $\gamma$ or $\lambda$ below 1, the later errors count for less, and the advantages move away from that simple gap.

To go deeper, read these lessons in the RL course:

- [Baselines and actor-critic](https://rl.ifkash.dev/lessons/031-baselines-and-actor-critic)
- [Generalized advantage estimation](https://rl.ifkash.dev/lessons/032-generalized-advantage-estimation)
- [Proximal policy optimization](https://rl.ifkash.dev/lessons/034-proximal-policy-optimization)
- [RLHF with PPO](https://rl.ifkash.dev/lessons/053-rlhf-with-ppo)

The exercise has a small `terminal_reward_gae` function that reproduces the numbers above.

### Nudge, but not too far

To raise the probability of a token with a positive advantage, the update uses the probability ratio, the policy's current probability of the token divided by its probability when the answer was written:

$$
\rho = \frac{\pi_{\text{new}}(\text{token})}{\pi_{\text{old}}(\text{token})} = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})
$$

Read it aloud: how much more or less likely the token has become since sampling. At the start of a round $\rho = 1$, and the gradient of $\rho \cdot A$ there is exactly "raise this token's log probability in proportion to its advantage". The ratio lets the same samples serve several updates, which is cheap, because generation is usually the slow part.

Reuse has a limit. The advantage was measured on answers from the old policy, so once the policy has moved far, it tells you little. The update therefore clips the ratio into the band from $1 - \epsilon$ to $1 + \epsilon$, with $\epsilon = 0.2$, and takes the lower of the clipped and unclipped terms:

$$
L = \frac{1}{n}\sum_{i=1}^{n} \min\bigl(\rho_i A_i,\ \text{clip}(\rho_i,\ 1 - \epsilon,\ 1 + \epsilon)\, A_i\bigr)
$$

Read it aloud: for each token, multiply the ratio by the advantage, do the same with the ratio forced into the band, keep the smaller value, and average. PPO moves the policy to make $L$ larger.

Work the four samples. Their ratios are $[1.1, 1.4, 0.7, 1.3]$ and their advantages are $[1, 1, -1, -1]$:

| Sample | $\rho A$ | Clipped $\rho$ times $A$ | Smaller (the term in $L$) |
|---|---|---|---|
| 1 | 1.1 | 1.1 | 1.1 |
| 2 | 1.4 | 1.2 | 1.2 |
| 3 | −0.7 | −0.8 | −0.8 |
| 4 | −1.3 | −1.2 | −1.3 |

The mean of the last column is $L = 0.05$. Without the clip it would be $0.125$. Figure 2 shows both cases. For a good answer, the term stops growing once $\rho$ passes 1.2, so sample 2 has nothing left to gain and gives no push. For a bad answer, lowering $\rho$ helps until it drops below 0.8, so sample 3, already at 0.7, gives no push either. But sample 4 is a bad answer whose probability rose by 30 percent. Taking the smaller value keeps its penalty at $-1.3$, and its gradient still pulls the probability back. The clip never hides a mistake.

```figure
<svg viewBox="0 0 600 272" role="img" aria-label="The clipped objective as a function of the probability ratio, for a positive and for a negative advantage, with the four worked samples marked">
<text x="170.0" y="24.0" text-anchor="middle" class="fig-bold">advantage +1</text>
<rect x="122.0" y="50.0" width="96.0" height="175.0" class="fill-soft-c"/>
<line x1="50.0" y1="225.0" x2="290.0" y2="225.0" class="stroke-grid"/>
<text x="44.0" y="229.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="50.0" y1="137.5" x2="290.0" y2="137.5" class="stroke-grid"/>
<text x="44.0" y="141.5" text-anchor="end" class="fig-small">1</text>
<line x1="50.0" y1="50.0" x2="290.0" y2="50.0" class="stroke-grid"/>
<text x="44.0" y="54.0" text-anchor="end" class="fig-small">1.5</text>
<text x="50.0" y="243.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="122.0" y="243.0" text-anchor="middle" class="fig-small">0.8</text>
<text x="170.0" y="243.0" text-anchor="middle" class="fig-small">1</text>
<text x="218.0" y="243.0" text-anchor="middle" class="fig-small">1.2</text>
<text x="290.0" y="243.0" text-anchor="middle" class="fig-small">1.5</text>
<line x1="50.0" y1="225.0" x2="290.0" y2="225.0" class="stroke-muted"/>
<line x1="50.0" y1="50.0" x2="50.0" y2="225.0" class="stroke-muted"/>
<polyline points="50.0,225.0 52.4,223.2 54.8,221.5 57.2,219.8 59.6,218.0 62.0,216.2 64.4,214.5 66.8,212.8 69.2,211.0 71.6,209.2 74.0,207.5 76.4,205.8 78.8,204.0 81.2,202.2 83.6,200.5 86.0,198.8 88.4,197.0 90.8,195.2 93.2,193.5 95.6,191.8 98.0,190.0 100.4,188.2 102.8,186.5 105.2,184.8 107.6,183.0 110.0,181.2 112.4,179.5 114.8,177.8 117.2,176.0 119.6,174.2 122.0,172.5 124.4,170.8 126.8,169.0 129.2,167.2 131.6,165.5 134.0,163.8 136.4,162.0 138.8,160.2 141.2,158.5 143.6,156.8 146.0,155.0 148.4,153.2 150.8,151.5 153.2,149.8 155.6,148.0 158.0,146.2 160.4,144.5 162.8,142.8 165.2,141.0 167.6,139.2 170.0,137.5 172.4,135.8 174.8,134.0 177.2,132.2 179.6,130.5 182.0,128.8 184.4,127.0 186.8,125.2 189.2,123.5 191.6,121.8 194.0,120.0 196.4,118.3 198.8,116.5 201.2,114.8 203.6,113.0 206.0,111.3 208.4,109.5 210.8,107.8 213.2,106.0 215.6,104.3 218.0,102.5 220.4,100.8 222.8,99.0 225.2,97.2 227.6,95.5 230.0,93.8 232.4,92.0 234.8,90.2 237.2,88.5 239.6,86.8 242.0,85.0 244.4,83.2 246.8,81.5 249.2,79.8 251.6,78.0 254.0,76.2 256.4,74.5 258.8,72.7 261.2,71.0 263.6,69.2 266.0,67.5 268.4,65.7 270.8,64.0 273.2,62.2 275.6,60.5 278.0,58.7 280.4,57.0 282.8,55.2 285.2,53.5 287.6,51.8 290.0,50.0" fill="none" class="stroke-muted dashed"/>
<polyline points="50.0,225.0 52.4,223.2 54.8,221.5 57.2,219.8 59.6,218.0 62.0,216.2 64.4,214.5 66.8,212.8 69.2,211.0 71.6,209.2 74.0,207.5 76.4,205.8 78.8,204.0 81.2,202.2 83.6,200.5 86.0,198.8 88.4,197.0 90.8,195.2 93.2,193.5 95.6,191.8 98.0,190.0 100.4,188.2 102.8,186.5 105.2,184.8 107.6,183.0 110.0,181.2 112.4,179.5 114.8,177.8 117.2,176.0 119.6,174.2 122.0,172.5 124.4,170.8 126.8,169.0 129.2,167.2 131.6,165.5 134.0,163.8 136.4,162.0 138.8,160.2 141.2,158.5 143.6,156.8 146.0,155.0 148.4,153.2 150.8,151.5 153.2,149.8 155.6,148.0 158.0,146.2 160.4,144.5 162.8,142.8 165.2,141.0 167.6,139.2 170.0,137.5 172.4,135.8 174.8,134.0 177.2,132.2 179.6,130.5 182.0,128.8 184.4,127.0 186.8,125.2 189.2,123.5 191.6,121.8 194.0,120.0 196.4,118.3 198.8,116.5 201.2,114.8 203.6,113.0 206.0,111.3 208.4,109.5 210.8,107.8 213.2,106.0 215.6,104.3 218.0,102.5 220.4,102.5 222.8,102.5 225.2,102.5 227.6,102.5 230.0,102.5 232.4,102.5 234.8,102.5 237.2,102.5 239.6,102.5 242.0,102.5 244.4,102.5 246.8,102.5 249.2,102.5 251.6,102.5 254.0,102.5 256.4,102.5 258.8,102.5 261.2,102.5 263.6,102.5 266.0,102.5 268.4,102.5 270.8,102.5 273.2,102.5 275.6,102.5 278.0,102.5 280.4,102.5 282.8,102.5 285.2,102.5 287.6,102.5 290.0,102.5" fill="none" stroke-width="3" class="stroke-b"/>
<circle cx="194.0" cy="120.0" r="7" class="fill-a"/>
<circle cx="266.0" cy="102.5" r="7" class="fill-a"/>
<text x="460.0" y="24.0" text-anchor="middle" class="fig-bold">advantage −1</text>
<rect x="412.0" y="50.0" width="96.0" height="175.0" class="fill-soft-c"/>
<line x1="340.0" y1="225.0" x2="580.0" y2="225.0" class="stroke-grid"/>
<text x="334.0" y="229.0" text-anchor="end" class="fig-small">−1.5</text>
<line x1="340.0" y1="137.5" x2="580.0" y2="137.5" class="stroke-grid"/>
<text x="334.0" y="141.5" text-anchor="end" class="fig-small">−1</text>
<line x1="340.0" y1="50.0" x2="580.0" y2="50.0" class="stroke-grid"/>
<text x="334.0" y="54.0" text-anchor="end" class="fig-small">−0.5</text>
<text x="340.0" y="243.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="412.0" y="243.0" text-anchor="middle" class="fig-small">0.8</text>
<text x="460.0" y="243.0" text-anchor="middle" class="fig-small">1</text>
<text x="508.0" y="243.0" text-anchor="middle" class="fig-small">1.2</text>
<text x="580.0" y="243.0" text-anchor="middle" class="fig-small">1.5</text>
<line x1="340.0" y1="225.0" x2="580.0" y2="225.0" class="stroke-muted"/>
<line x1="340.0" y1="50.0" x2="340.0" y2="225.0" class="stroke-muted"/>
<polyline points="340.0,50.0 342.4,51.8 344.8,53.5 347.2,55.2 349.6,57.0 352.0,58.8 354.4,60.5 356.8,62.2 359.2,64.0 361.6,65.8 364.0,67.5 366.4,69.2 368.8,71.0 371.2,72.8 373.6,74.5 376.0,76.2 378.4,78.0 380.8,79.8 383.2,81.5 385.6,83.2 388.0,85.0 390.4,86.8 392.8,88.5 395.2,90.2 397.6,92.0 400.0,93.8 402.4,95.5 404.8,97.2 407.2,99.0 409.6,100.8 412.0,102.5 414.4,104.3 416.8,106.0 419.2,107.8 421.6,109.5 424.0,111.3 426.4,113.0 428.8,114.8 431.2,116.5 433.6,118.2 436.0,120.0 438.4,121.8 440.8,123.5 443.2,125.2 445.6,127.0 448.0,128.8 450.4,130.5 452.8,132.2 455.2,134.0 457.6,135.8 460.0,137.5 462.4,139.2 464.8,141.0 467.2,142.8 469.6,144.5 472.0,146.2 474.4,148.0 476.8,149.8 479.2,151.5 481.6,153.2 484.0,155.0 486.4,156.7 488.8,158.5 491.2,160.2 493.6,162.0 496.0,163.8 498.4,165.5 500.8,167.2 503.2,169.0 505.6,170.8 508.0,172.5 510.4,174.2 512.8,176.0 515.2,177.8 517.6,179.5 520.0,181.2 522.4,183.0 524.8,184.8 527.2,186.5 529.6,188.2 532.0,190.0 534.4,191.8 536.8,193.5 539.2,195.2 541.6,197.0 544.0,198.8 546.4,200.5 548.8,202.3 551.2,204.0 553.6,205.8 556.0,207.5 558.4,209.3 560.8,211.0 563.2,212.8 565.6,214.5 568.0,216.3 570.4,218.0 572.8,219.8 575.2,221.5 577.6,223.2 580.0,225.0" fill="none" class="stroke-muted dashed"/>
<polyline points="340.0,102.5 342.4,102.5 344.8,102.5 347.2,102.5 349.6,102.5 352.0,102.5 354.4,102.5 356.8,102.5 359.2,102.5 361.6,102.5 364.0,102.5 366.4,102.5 368.8,102.5 371.2,102.5 373.6,102.5 376.0,102.5 378.4,102.5 380.8,102.5 383.2,102.5 385.6,102.5 388.0,102.5 390.4,102.5 392.8,102.5 395.2,102.5 397.6,102.5 400.0,102.5 402.4,102.5 404.8,102.5 407.2,102.5 409.6,102.5 412.0,102.5 414.4,104.3 416.8,106.0 419.2,107.8 421.6,109.5 424.0,111.3 426.4,113.0 428.8,114.8 431.2,116.5 433.6,118.2 436.0,120.0 438.4,121.8 440.8,123.5 443.2,125.2 445.6,127.0 448.0,128.8 450.4,130.5 452.8,132.2 455.2,134.0 457.6,135.8 460.0,137.5 462.4,139.2 464.8,141.0 467.2,142.8 469.6,144.5 472.0,146.2 474.4,148.0 476.8,149.8 479.2,151.5 481.6,153.2 484.0,155.0 486.4,156.7 488.8,158.5 491.2,160.2 493.6,162.0 496.0,163.8 498.4,165.5 500.8,167.2 503.2,169.0 505.6,170.8 508.0,172.5 510.4,174.2 512.8,176.0 515.2,177.8 517.6,179.5 520.0,181.2 522.4,183.0 524.8,184.8 527.2,186.5 529.6,188.2 532.0,190.0 534.4,191.8 536.8,193.5 539.2,195.2 541.6,197.0 544.0,198.8 546.4,200.5 548.8,202.3 551.2,204.0 553.6,205.8 556.0,207.5 558.4,209.3 560.8,211.0 563.2,212.8 565.6,214.5 568.0,216.3 570.4,218.0 572.8,219.8 575.2,221.5 577.6,223.2 580.0,225.0" fill="none" stroke-width="3" class="stroke-b"/>
<circle cx="388.0" cy="102.5" r="7" class="fill-a"/>
<circle cx="532.0" cy="190.0" r="7" class="fill-a"/>
<text x="170.0" y="163.8" text-anchor="start" class="fig-small">flat above 1.2:</text>
<text x="170.0" y="183.0" text-anchor="start" class="fig-small">nothing more to gain</text>
<text x="346.0" y="155.0" text-anchor="start" class="fig-small">flat below 0.8:</text>
<text x="346.0" y="174.2" text-anchor="start" class="fig-small">nothing more to gain</text>
<text x="580.0" y="71.0" text-anchor="end" class="fig-small">above 1.2 it keeps falling,</text>
<text x="580.0" y="90.2" text-anchor="end" class="fig-small">so a bad step is punished</text>
<text x="194.0" y="108.0" text-anchor="middle" class="fig-small">sample 1</text>
<text x="266.0" y="90.5" text-anchor="middle" class="fig-small">sample 2</text>
<text x="388.0" y="90.5" text-anchor="middle" class="fig-small">sample 3</text>
<text x="532.0" y="178.0" text-anchor="middle" class="fig-small">sample 4</text>
<text x="170.0" y="262.0" text-anchor="middle" class="fig-small">probability ratio ρ = new / old</text>
<text x="460.0" y="262.0" text-anchor="middle" class="fig-small">probability ratio ρ = new / old</text>
</svg>
Figure 2. The clipped objective (orange) against the probability ratio, with ε = 0.2. The dashed line is the unclipped ρ·A, and the green band is where the ratio stays within 0.8 to 1.2. For a positive advantage (left) the curve goes flat above 1.2, so a lucky sample stops pulling. For a negative advantage (right) it goes flat below 0.8, but above 1.2 it keeps falling, so a bad step is still punished. Dots are the four samples worked in the text.
```

> [!WARNING]
> The clip limits the training update. It does not change which tokens the model writes, and it has nothing to do with decoding. It lives inside the loss and disappears once training is over.

### Why PPO is heavy

A round of PPO keeps four models in memory: the policy, the reference, the frozen reward model, and the value head. Two of them are trained, and each trained model also needs gradients and optimiser state. A round starts by generating whole answers one token after another, and generation is sequential, so most of the wall-clock time usually goes there. The loop also has knobs that interact: $\beta$, $\epsilon$, the learning rates, and the value head's accuracy. A weak value head gives noisy advantages, and noisy advantages give noisy updates.

> [!KEY]
> PPO moves toward the leashed optimum from samples. It subtracts a baseline so a sample is pushed up only if it beat expectations, and it clips the probability ratio so one batch cannot pull the policy far. The price is four models and a generation loop.

## Route two: DPO, read the reward off the policy

You have preference pairs for the mortgage prompt. The route through PPO has two stages: fit a reward model on the pairs, then run PPO against it. Each stage can go wrong, and errors in the first carry into the second. Can the middle be skipped?

Why not fine-tune on the chosen answers alone? That raises the chosen answer's probability but never lowers the rejected one, and it knows nothing about the reference or the leash. The pairs say more: the chosen answer is better than the rejected one, by some margin, relative to where the policy started. Figure 3 puts the two pipelines side by side. DPO deletes the reward model, the sampling, and the value head, and keeps a loss that reads the log probabilities of the chosen and rejected answers under the policy being trained and under the frozen reference.

```figure
<svg viewBox="0 0 600 356" role="img" aria-label="The PPO pipeline of five stages next to the DPO pipeline of three stages after the preference pairs">
<defs><marker id="c054b-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="24.0" text-anchor="start" class="fig-bold">PPO pipeline</text>
<text x="330.0" y="24.0" text-anchor="start" class="fig-bold">DPO pipeline</text>
<rect x="20.0" y="40.0" width="240.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="140.0" y="66.0" text-anchor="middle" class="fig-small">preference pairs</text>
<rect x="20.0" y="100.0" width="240.0" height="44.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="140.0" y="126.0" text-anchor="middle" class="fig-small">train a reward model</text>
<rect x="20.0" y="160.0" width="240.0" height="44.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="140.0" y="186.0" text-anchor="middle" class="fig-small">sample answers from the policy</text>
<rect x="20.0" y="220.0" width="240.0" height="44.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="140.0" y="246.0" text-anchor="middle" class="fig-small">score, then compare with a value head</text>
<rect x="20.0" y="280.0" width="240.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="140.0" y="306.0" text-anchor="middle" class="fig-small">clipped update of the policy</text>
<line x1="140.0" y1="85.0" x2="140.0" y2="97.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<line x1="140.0" y1="145.0" x2="140.0" y2="157.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<line x1="140.0" y1="205.0" x2="140.0" y2="217.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<line x1="140.0" y1="265.0" x2="140.0" y2="277.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<rect x="330.0" y="40.0" width="240.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="450.0" y="66.0" text-anchor="middle" class="fig-small">preference pairs</text>
<rect x="330.0" y="100.0" width="240.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="450.0" y="118.5" text-anchor="middle" class="fig-small">log probabilities of both answers,</text>
<text x="450.0" y="133.5" text-anchor="middle" class="fig-small">under the policy and the reference</text>
<rect x="330.0" y="160.0" width="240.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="450.0" y="178.5" text-anchor="middle" class="fig-small">one loss: Bradley-Terry on</text>
<text x="450.0" y="193.5" text-anchor="middle" class="fig-small">implicit rewards</text>
<rect x="330.0" y="220.0" width="240.0" height="44.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="450.0" y="246.0" text-anchor="middle" class="fig-small">gradient update of the policy</text>
<line x1="450.0" y1="85.0" x2="450.0" y2="97.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<line x1="450.0" y1="145.0" x2="450.0" y2="157.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<line x1="450.0" y1="205.0" x2="450.0" y2="217.0" class="stroke-muted" marker-end="url(#c054b-arrow)" stroke-width="2"/>
<text x="330.0" y="290.0" text-anchor="start" class="fig-small">no reward model, no sampling,</text>
<text x="330.0" y="306.0" text-anchor="start" class="fig-small">no value head: two models in memory</text>
<rect x="20.0" y="332.0" width="14.0" height="14.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="40.0" y="344.0" text-anchor="start" class="fig-small">stage DPO removes</text>
<rect x="200.0" y="332.0" width="14.0" height="14.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="220.0" y="344.0" text-anchor="start" class="fig-small">stage that stays</text>
</svg>
Figure 3. The two pipelines side by side. Orange boxes are the stages PPO needs and DPO does not: fitting a reward model, sampling answers, and scoring them against a value head. DPO goes from preference pairs straight to one loss on the policy's and the reference's log probabilities.
```

### Run the formula backwards

Start from the best policy:

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}}{Z(x)}
$$

Take the log of both sides, since the log of a product is a sum and the log of a division is a difference, then solve for $r$:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Read it aloud: the reward of an answer is $\beta$ times the log of how much the best policy boosted it, plus a number that depends on the prompt and not on the answer. Check it on three answers with $\beta = 1$. The best policy is $[0.731, 0.161, 0.108]$ against the reference $[0.5, 0.3, 0.2]$. The log ratios are $[0.380, -0.620, -0.620]$. Add $\log Z = 0.620$ and you recover the rewards $[1, 0, 0]$. A policy is a reward model in disguise.

### The constant cancels

The term $\beta \log Z(x)$ is the one thing you cannot compute, because $Z(x)$ sums over every answer. But it is the same for every answer to the same prompt. A pair compares two answers to one prompt, and the Bradley-Terry loss sees only the gap, so the term drops out:

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

In the example, the log ratios $0.380$ and $-0.620$ differ by $1.000$, and the rewards $1$ and $0$ differ by $1$. PPO got around $Z$ by sampling, and DPO gets around it by never needing it.

### The implicit reward and the DPO loss

Replace the best policy with the policy $\pi_\theta$ you are training. Any policy has an implicit reward, the reward for which it would be the best policy:

$$
\hat r(x, y) = \beta \left(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\right)
$$

Read it aloud: $\beta$ times how much more log probability the trained policy gives the answer than the reference does. In numpy: `beta * (policy_logps - ref_logps)`. The DPO loss is Bradley-Terry applied to these rewards, so training adjusts $\theta$ until the policy's own implicit reward ranks the pairs as people did:

$$
\text{DPO loss} = \frac{1}{n}\sum_{i=1}^{n} -\log \sigma(u_i), \qquad u_i = \hat r_{w,i} - \hat r_{l,i}
$$

The margin $u_i$ is the implicit reward of the chosen answer minus that of the rejected one. This is the same problem PPO solves, because the best policy for the reward the pairs imply is the policy whose implicit reward agrees with the pairs, and Bradley-Terry is the loss that makes a reward agree with the pairs. Nothing here needs a sample, a reward model, or $Z$.

Take $\beta = 1$ and one pair, helpful over padded, with reference log probabilities $\ln 0.5 = -0.693$ and $\ln 0.2 = -1.609$. At the start the policy equals the reference, so both implicit rewards are 0, the margin is 0, and the loss is $\log 2 = 0.693$, the price of a tie.

### What the gradient does

The loss depends on the policy only through the two log probabilities, and $\hat r$ moves by $\beta$ per unit of log probability. So the chain rule puts a factor $\beta$ on the Bradley-Terry gradient:

$$
\frac{\partial\, \text{loss}}{\partial \log \pi_\theta(y_{w,i})} = \beta\,\frac{\sigma(u_i) - 1}{n}, \qquad
\frac{\partial\, \text{loss}}{\partial \log \pi_\theta(y_{l,i})} = -\beta\,\frac{\sigma(u_i) - 1}{n}
$$

Read it aloud: raise the chosen log probability and lower the rejected one, each by a weight $\beta(1 - \sigma(u))$ that is largest when the pair is ranked wrong and shrinks as it becomes ranked right. The weight is always below $\beta$. It equals $\beta/2$ at $u = 0$ and approaches $\beta$ only when the pair is very wrong. Figure 4 plots it.

```figure
<svg viewBox="0 0 600 284" role="img" aria-label="The DPO weight of a pair as a function of its margin, with the margins 0 and 0.81 marked">
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="182.5" x2="560.0" y2="182.5" class="stroke-grid"/>
<text x="62.0" y="186.5" text-anchor="end" class="fig-small">0.25</text>
<line x1="70.0" y1="135.0" x2="560.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70.0" y1="87.5" x2="560.0" y2="87.5" class="stroke-grid"/>
<text x="62.0" y="91.5" text-anchor="end" class="fig-small">0.75</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="248.0" text-anchor="middle" class="fig-small">−4</text>
<text x="168.0" y="248.0" text-anchor="middle" class="fig-small">−2</text>
<text x="266.0" y="248.0" text-anchor="middle" class="fig-small">0</text>
<text x="364.0" y="248.0" text-anchor="middle" class="fig-small">2</text>
<text x="462.0" y="248.0" text-anchor="middle" class="fig-small">4</text>
<text x="560.0" y="248.0" text-anchor="middle" class="fig-small">6</text>
<rect x="413.0" y="40.0" width="147.0" height="190.0" class="fill-soft-c"/>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="40.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<polyline points="70.0,43.4 72.4,43.6 74.9,43.8 77.3,44.0 79.8,44.2 82.2,44.4 84.7,44.6 87.2,44.8 89.6,45.1 92.1,45.3 94.5,45.6 96.9,45.8 99.4,46.1 101.8,46.4 104.3,46.8 106.8,47.1 109.2,47.4 111.7,47.8 114.1,48.2 116.6,48.6 119.0,49.0 121.4,49.4 123.9,49.9 126.4,50.4 128.8,50.9 131.2,51.4 133.7,52.0 136.2,52.5 138.6,53.1 141.1,53.8 143.5,54.4 145.9,55.1 148.4,55.8 150.9,56.5 153.3,57.3 155.8,58.1 158.2,59.0 160.7,59.8 163.1,60.7 165.6,61.7 168.0,62.6 170.4,63.7 172.9,64.7 175.3,65.8 177.8,67.0 180.2,68.1 182.7,69.3 185.2,70.6 187.6,71.9 190.1,73.3 192.5,74.7 194.9,76.1 197.4,77.6 199.8,79.1 202.3,80.7 204.8,82.3 207.2,84.0 209.7,85.7 212.1,87.5 214.6,89.3 217.0,91.1 219.5,93.0 221.9,94.9 224.4,96.9 226.8,98.9 229.2,101.0 231.7,103.0 234.2,105.2 236.6,107.3 239.1,109.5 241.5,111.7 244.0,114.0 246.4,116.2 248.9,118.5 251.3,120.9 253.8,123.2 256.2,125.5 258.6,127.9 261.1,130.3 263.6,132.6 266.0,135.0 268.4,137.4 270.9,139.7 273.4,142.1 275.8,144.5 278.2,146.8 280.7,149.1 283.2,151.5 285.6,153.8 288.1,156.0 290.5,158.3 292.9,160.5 295.4,162.7 297.9,164.8 300.3,167.0 302.8,169.0 305.2,171.1 307.7,173.1 310.1,175.1 312.6,177.0 315.0,178.9 317.5,180.7 319.9,182.5 322.4,184.3 324.8,186.0 327.2,187.7 329.7,189.3 332.2,190.9 334.6,192.4 337.1,193.9 339.5,195.3 342.0,196.7 344.4,198.1 346.9,199.4 349.3,200.7 351.8,201.9 354.2,203.0 356.7,204.2 359.1,205.3 361.6,206.3 364.0,207.4 366.5,208.3 368.9,209.3 371.4,210.2 373.8,211.0 376.2,211.9 378.7,212.7 381.1,213.5 383.6,214.2 386.1,214.9 388.5,215.6 390.9,216.2 393.4,216.9 395.9,217.5 398.3,218.0 400.8,218.6 403.2,219.1 405.7,219.6 408.1,220.1 410.6,220.6 413.0,221.0 415.5,221.4 417.9,221.8 420.4,222.2 422.8,222.6 425.2,222.9 427.7,223.2 430.2,223.6 432.6,223.9 435.1,224.2 437.5,224.4 440.0,224.7 442.4,224.9 444.9,225.2 447.3,225.4 449.8,225.6 452.2,225.8 454.7,226.0 457.1,226.2 459.6,226.4 462.0,226.6 464.5,226.7 466.9,226.9 469.4,227.1 471.8,227.2 474.2,227.3 476.7,227.5 479.1,227.6 481.6,227.7 484.1,227.8 486.5,227.9 489.0,228.0 491.4,228.1 493.9,228.2 496.3,228.3 498.8,228.4 501.2,228.4 503.6,228.5 506.1,228.6 508.6,228.7 511.0,228.7 513.5,228.8 515.9,228.8 518.4,228.9 520.8,229.0 523.2,229.0 525.7,229.1 528.1,229.1 530.6,229.1 533.0,229.2 535.5,229.2 538.0,229.3 540.4,229.3 542.9,229.3 545.3,229.4 547.8,229.4 550.2,229.4 552.7,229.5 555.1,229.5 557.6,229.5 560.0,229.5" fill="none" stroke-width="3" class="stroke-b"/>
<circle cx="266.0" cy="135.0" r="6" class="fill-a"/>
<text x="278.0" y="123.0" text-anchor="start" class="fig-bold fig-small">u = 0: weight 0.5</text>
<circle cx="305.6" cy="171.4" r="6" class="fill-a"/>
<text x="317.6" y="159.4" text-anchor="start" class="fig-bold fig-small">u = 0.81: weight 0.308</text>
<text x="421.0" y="116.0" text-anchor="start" class="fig-small">pair already right:</text>
<text x="421.0" y="130.0" text-anchor="start" class="fig-small">barely moves</text>
<text x="74.9" y="173.0" text-anchor="start" class="fig-small">pair ranked wrong:</text>
<text x="74.9" y="187.0" text-anchor="start" class="fig-small">weight near 1</text>
<text x="315.0" y="272.0" text-anchor="middle" class="fig-small">margin u = implicit reward of chosen minus implicit reward of rejected</text>
<text x="70.0" y="24.0" text-anchor="start" class="fig-small">weight β(1 − σ(u)), with β = 1</text>
</svg>
Figure 4. The weight DPO gives each pair, β(1 − σ(u)), as a function of the margin u. At u = 0 the weight is half its maximum. It falls as the pair is ranked more and more right, and it stays above 0 but is tiny in the green region. A pair ranked wrong gets a weight close to β.
```

Do one step by hand with learning rate 0.1. At $u = 0$ the chosen gradient is $-0.5$, so descent raises the chosen log probability by $0.05$ and lowers the rejected one by $0.05$. The margin becomes $u = 0.1$ and the loss falls from $0.693$ to $0.644$. After ten steps the chosen log probability is $-0.290$ and the rejected one is $-2.013$. The margin is $0.807$, the loss is $0.369$, and the weight has dropped from $0.5$ to $0.308$. The weight is why DPO does not overdo it: a pair already ranked right gets a small push, and the pairs still wrong get most of the effort.

> [!WARNING]
> The exercise treats the two log probabilities as free numbers that you move directly. A real policy is a network, and the log probabilities of different answers are tied together through shared weights. In a real run, pushing the rejected answer down can drag the chosen answer down too, so both fall while their gap grows. This failure is sometimes called likelihood displacement.

> [!KEY]
> A policy hides a reward: $\beta$ times its log ratio to the reference, plus a prompt-only constant. The constant cancels in a comparison, so Bradley-Terry on these implicit rewards trains the policy straight from preference pairs. That reaches PPO's goal with two models in memory, no sampling, and no reward model.

> [!KEY]
> You have two routes to the same leashed optimum. PPO samples from the policy, so it can score answers that no human ever compared, at the price of four models and a generation loop. DPO trains on fixed pairs with two models, so it is cheaper and steadier, but it learns only what the pairs show.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The probability ratio is `np.exp(new_logps - old_logps)`, and it is 1 when nothing has changed.
> - The advantage is the reward minus the value guess, recentred and divided by its standard deviation, and it is 0 when every sample matches its guess.
> - The clipped term is `min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)`, averaged. A positive advantage stops pulling above $1 + \epsilon$. A negative one stops pulling below $1 - \epsilon$ and keeps pulling above $1 + \epsilon$.
> - The implicit reward is `beta * (policy_logps - ref_logps)`, and `dpo_loss` is the Bradley-Terry loss of the two implicit rewards, equal to $\log 2$ when the policy equals the reference.
> - `dpo_grad` is $\beta$ times the Bradley-Terry gradient of the implicit rewards, and shifting both reference log probabilities of a prompt by one constant changes nothing.

## Build it step by step

The starter file already holds `preference_probability`, `bradley_terry_loss`, `bradley_terry_grad`, and `optimal_policy` from the previous chapter. Use them. The checks call your functions in this order:

1. `policy_ratio(new_logps, old_logps)` takes two arrays of log probabilities of the same shape and returns their ratio of probabilities. Subtract the logs before exponentiating. The common mistake is to exponentiate each first and divide, which loses precision when log probabilities are very negative:

```python
def policy_ratio(new_logps, old_logps):
    return ...  # exp of the difference of logs
```

2. `advantage(rewards, values)` takes two `(n,)` arrays and returns the recentred, rescaled difference. Add `1e-8` to the standard deviation. The common mistake is to normalise the rewards first and subtract the value afterwards, which mixes two scales:

```python
def advantage(rewards, values):
    raw = rewards - values
    return ...  # (raw - mean) / (std + 1e-8)
```

3. `clipped_objective(ratio, advantage, eps)` returns the mean of the smaller term. Use `np.clip` and `np.minimum`, not `np.maximum`. The common mistake is to clip the advantage instead of the ratio:

```python
def clipped_objective(ratio, advantage, eps):
    unclipped = ratio * advantage
    clipped = ...  # np.clip(ratio, 1 - eps, 1 + eps) * advantage
    return np.mean(np.minimum(unclipped, clipped))
```

4. `implicit_reward(policy_logps, ref_logps, beta)` takes two arrays of the same shape and returns $\beta$ times their difference. The common mistake is to forget the reference and return `beta * policy_logps`:

```python
def implicit_reward(policy_logps, ref_logps, beta):
    return ...  # beta * (policy_logps - ref_logps)
```

5. `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` returns a float. Compute the two implicit rewards and pass them to `bradley_terry_loss`. The common mistake is to pass the raw log probabilities:

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    r_w = implicit_reward(policy_chosen_logps, ref_chosen_logps, beta)
    r_l = ...  # implicit reward of the rejected answers
    return bradley_terry_loss(r_w, r_l)
```

6. `dpo_grad` takes the same arguments and returns two `(n,)` arrays, the gradients with respect to the policy's chosen and rejected log probabilities. Multiply the gradients from `bradley_terry_grad` by `beta`. The common mistake is to drop $\beta$, which is correct only when $\beta = 1$:

```python
def dpo_grad(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    r_w = ...  # implicit rewards, as above
    r_l = ...
    grad_w, grad_l = bradley_terry_grad(r_w, r_l)
    return ..., ...  # multiply each by beta
```

## Where this breaks

A weak reward model is still the weakest link for PPO. The leash and the clip slow down a policy that chases its blind spots, and neither removes them. The value head must learn at the pace of the policy, and when it lags the advantages mislead. The advantage in this chapter is also incomplete without a trained value function: a real trainer fits the value head by regression and spreads credit over tokens with GAE. Because each step is small, PPO needs many rounds, and each round needs fresh generation.

DPO learns the preferences in its pairs, blind spots included, and it has no signal for prompts the pairs do not cover. The pairs were usually written by some other model, so the policy trains on answers it might never write itself. A small $\beta$ lets the policy move far from the reference before the weights fade, and a large $\beta$ keeps it close. Practitioners often add a small supervised loss on the chosen answers to keep their probability from falling.

## The exercise

You build `policy_ratio`, `advantage`, `clipped_objective`, `implicit_reward`, `dpo_loss`, `dpo_grad`, and a short pointer to the full method, `terminal_reward_gae`. The PPO checks verify the hand-worked ratios, the advantages $[1, 1, -1, -1]$ and the zero-variance case, and the objective value $0.05$. Finite differences confirm that the objective's gradient is 0 in the flat regions and nonzero for a bad answer whose ratio went above $1 + \epsilon$, and a gradient ascent on one ratio stops near $1 + \epsilon$.

The DPO checks verify the implicit reward and the planted-reward recovery from `optimal_policy`, the loss against Bradley-Terry on the implicit rewards, the value $\log 2$ at the start, and the claim that shifting both reference log probabilities by one constant changes nothing. They also check the gradient against finite differences, check that its size is $\beta(1 - \sigma(u))/n$, and run fifty gradient steps to confirm that the loss falls and the chosen answer gains on the rejected one.

The `terminal_reward_gae` checks compare a hand-worked answer with $\gamma = 0.9$ and $\lambda = 0.8$, and confirm that with $\gamma = \lambda = 1$ every token's advantage is the reward minus its value guess.

## Further reading

- Schulman, Moritz, Levine, Jordan, and Abbeel, ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438) (ICLR 2016). Section 3 defines the one-step error $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ and the estimator as the $\gamma\lambda$-discounted sum of those errors (Equation 16). Equation 18 shows that $\lambda = 1$ gives the discounted return minus $V(s_t)$, which is the per-answer form this chapter uses.
