---
title: "PPO: the sampling loop, in words and one clip"
slug: 127-ppo-the-sampling-loop
part: "Part 7 — Large language models"
summary: PPO reaches the leashed optimum by sampling answers, scoring them, and nudging the policy toward better-than-expected ones, with a clip that limits each step.
minutes: 40
gpu: false
objectives:
  - Explain why PPO works from samples, and walk through one round of sample, score, compare with a baseline, update, and repeat.
  - Compute an advantage from rewards and value predictions, and explain why a baseline and a normalisation are used.
  - Compute the probability ratio and the clipped objective by hand, and explain what the clip does for positive and negative advantages.
  - Explain why PPO keeps four models resident and why that makes it heavy and fragile.
prereqs:
  - 126-kl-leash-and-optimal-policy
  - 125-reward-from-preferences
  - 105-next-token-loss
exercise: 127-ppo-the-sampling-loop
---

# PPO: the sampling loop, in words and one clip

The last chapter found the best policy for reward minus a leash: the reference model, reweighted by $e^{r/\beta}$ and divided by a total $Z$. You cannot build that policy, because $Z$ adds up every possible answer, and for a real language model there are about $10^{470}$ answers of 100 tokens. What you can do is ask the policy to answer. A model you already have writes an answer to the mortgage prompt in a few seconds, and the reward model scores it.

So the question is how to move toward the best policy using only samples from the model you have. The obvious tries break in two ways. Fine-tune on the highest-scoring sample, and you copy the winners but ignore what the losers say, and you ignore how much better the winner was. Push up the reward of every sampled answer by plain gradient ascent, and a single lucky sample with a huge score drags the whole model along, in a direction that was true only for that sample.

Proximal policy optimisation (PPO) answers both. It compares each sample with what you would normally expect, so a good answer goes up and a bad one goes down. And it limits each step, so one sample cannot pull the policy far. This chapter builds those two pieces, the advantage and the clip. The rest of the loop stays in words, because a full run needs a real language model.

## What you already know

- A policy gives every answer a probability, and a reference policy is the supervised model it must stay near (see the chapter on the KL leash).
- A reward model gives each answer one number, and only gaps between rewards matter.
- The KL divergence is the average, over the policy's own answers, of the log ratio of policy to reference.
- A log probability of an answer is the sum of the log probabilities of its tokens.
- A gradient step changes the weights a little, which changes every probability the model gives.

## The picture: sample, score, compare, nudge, repeat

Figure 1 shows one round. The policy writes answers to a batch of prompts, which is called a rollout. The frozen reward model scores each answer. The score is lowered by a leash penalty, a charge for drifting from the reference model. Then each answer is compared with what the policy usually earns for that prompt, and the policy is nudged toward answers that did better than usual and away from answers that did worse. The loop then repeats from fresh samples, because the nudge changed the policy, and answers written by the old policy no longer describe it.

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="One round of PPO: prompt, policy writes answers, reward model scores them, advantage compares with a baseline, clipped update, repeat">
<defs><marker id="c127-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="108.0" y1="109.0" x2="127.0" y2="109.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<line x1="228.0" y1="109.0" x2="247.0" y2="109.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<line x1="348.0" y1="109.0" x2="367.0" y2="109.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<line x1="468.0" y1="109.0" x2="487.0" y2="109.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<rect x="250.0" y="190.0" width="96.0" height="50.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="298.0" y="211.5" text-anchor="middle" class="fig-small">reference</text>
<text x="298.0" y="226.5" text-anchor="middle" class="fig-small">model</text>
<line x1="298.0" y1="189.0" x2="298.0" y2="137.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<text x="306.0" y="168.0" text-anchor="start" class="fig-small">KL penalty</text>
<rect x="370.0" y="190.0" width="96.0" height="50.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="418.0" y="211.5" text-anchor="middle" class="fig-small">value head</text>
<text x="418.0" y="226.5" text-anchor="middle" class="fig-small">(baseline)</text>
<line x1="418.0" y1="189.0" x2="418.0" y2="137.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<text x="426.0" y="168.0" text-anchor="start" class="fig-small">expected reward</text>
<polyline points="538.0,82.0 538.0,58.0 178.0,58.0" fill="none" stroke-width="2" class="stroke-muted"/>
<line x1="178.0" y1="58.0" x2="178.0" y2="81.0" class="stroke-muted" marker-end="url(#c127-arrow)" stroke-width="2"/>
<text x="310.0" y="50.0" text-anchor="middle" class="fig-small">repeat with the updated policy</text>
<rect x="20.0" y="262.0" width="14.0" height="14.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="40.0" y="274.0" text-anchor="start" class="fig-small">model that is trained</text>
<rect x="190.0" y="262.0" width="14.0" height="14.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="210.0" y="274.0" text-anchor="start" class="fig-small">model that stays frozen</text>
</svg>
Figure 1. One round of PPO. The policy writes answers, the frozen reward model scores them, the reference model adds a per-token KL penalty, and the value head supplies the expected reward so that the advantage asks "better than expected?". The clipped update changes the policy a little, and the loop repeats with fresh samples from the updated policy.
```

Each stage exists to fix a specific failure, and the next sections take them in turn.

## The leash, one token at a time

The KL leash from the last chapter is an average over the policy's answers, and the rollout gives exactly that: answers written by the policy. So PPO charges every sampled token for how much more likely the policy finds it than the reference does. The token's reward is lowered by $\beta\,(\log \pi(\text{token}) - \log \pi_{\text{ref}}(\text{token}))$.

Suppose the policy gives some token probability 0.28 and the reference gives it 0.10. The log ratio is $\ln 2.8 = 1.03$, so with $\beta = 0.1$ the token pays $0.103$. A token the two models agree on pays nothing. Averaged over many sampled answers, this log ratio estimates $\text{KL}(\pi \,\|\, \pi_{\text{ref}})$, the leash from the last chapter, because that was defined as an average of that same log ratio over the policy's own answers.

## Better than expected: the advantage

Take four sampled answers to the mortgage prompt, scored by the reward model:

| Sample | Reward | Value head's guess | Reward − guess |
|---|---|---|---|
| 1 | 3.3 | 2.8 | +0.5 |
| 2 | 3.1 | 2.6 | +0.5 |
| 3 | 2.4 | 2.9 | −0.5 |
| 4 | 2.6 | 3.1 | −0.5 |

All four rewards are positive. If you push up the log probability of every sampled answer in proportion to its reward, you push up all four, and the bad ones get lifted along with the good ones. What the update needs to know is whether an answer was better than expected. The expectation comes from a value head, a small output on top of the policy's network that learns to predict the reward the policy will earn. It is trained alongside the policy with a squared-error loss against the rewards the rollouts actually earned, so its guess tracks the policy as it changes. Subtracting its guess leaves the last column: samples 1 and 2 beat the guess, and samples 3 and 4 fall short.

The reward scale is arbitrary, since the earlier chapter showed that only gaps matter, so the gaps are rescaled to zero mean and unit standard deviation. Here the mean is 0 and the standard deviation is 0.5, so the advantages are $[1, 1, -1, -1]$:

$$
A_i = \frac{(r_i - v_i) - \text{mean}}{\text{std}}
$$

Read it aloud: reward minus the value head's guess, then recentred and rescaled over the batch. If every sample matches its guess, the standard deviation is 0 and the advantages must be 0, not `nan`, so add a tiny constant to the denominator. In numpy: `a = rewards - values`, then `(a - a.mean()) / (a.std() + 1e-8)`.

## Nudge, but not too far

To raise the probability of a token with a positive advantage, the update uses the probability ratio, the policy's current probability of the token divided by its probability when the answer was written:

$$
\rho = \frac{\pi_{\text{new}}(\text{token})}{\pi_{\text{old}}(\text{token})} = \exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})
$$

Read it aloud: how much more or less likely the token has become since sampling. At the start of a round $\rho = 1$, and the gradient of $\rho \cdot A$ there is exactly "raise the log probability of this token in proportion to its advantage". The ratio lets the same samples be reused for several updates, which is cheap, because generation is usually the slow part.

Reuse has a limit. The advantage was measured on answers from the old policy, and it describes the old policy. Once the new policy has moved far from the old one, the advantage tells you little. So the update clips the ratio into the band from $1 - \epsilon$ to $1 + \epsilon$, with $\epsilon = 0.2$, and takes the lower of the clipped and unclipped terms:

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

The mean of the last column is $L = 0.05$. Without the clip it would be $0.125$. Figure 2 shows both cases of the rule. For a good answer (advantage above 0), the term stops growing once $\rho$ passes 1.2, so sample 2 has nothing left to gain and gives no push. For a bad answer, lowering $\rho$ helps until it drops below 0.8 and then stops helping, so sample 3, already at 0.7, gives no push either. But sample 4 is a bad answer whose probability went up by 30 percent. Taking the smaller value keeps its penalty at $-1.3$, and its gradient still pulls the probability back. The clip never hides a mistake.

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

## Why this is heavy

A round of PPO needs four models in memory: the policy, the reference, the frozen reward model, and the value head. Two of them are trained, and each trained model also needs gradients and optimiser state. A round starts by generating whole answers, one token after another, and generation is sequential, so most of the wall-clock time usually goes there. The loop has many knobs that interact: $\beta$, $\epsilon$, the learning rates, and the value head's accuracy. A weak value head gives noisy advantages, and noisy advantages give noisy updates.

```figure
<svg viewBox="0 0 600 272" role="img" aria-label="Four models kept in memory by PPO, against two kept by DPO">
<text x="20.0" y="34.0" text-anchor="start" class="fig-bold">PPO keeps four models resident</text>
<rect x="20.0" y="46.0" width="108.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="74.0" y="73.0" text-anchor="middle" class="fig-small">policy</text>
<rect x="140.0" y="46.0" width="108.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="194.0" y="73.0" text-anchor="middle" class="fig-small">reference</text>
<rect x="260.0" y="46.0" width="108.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="314.0" y="73.0" text-anchor="middle" class="fig-small">reward model</text>
<rect x="380.0" y="46.0" width="108.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="434.0" y="73.0" text-anchor="middle" class="fig-small">value head</text>
<text x="20.0" y="124.0" text-anchor="start" class="fig-small">and generates text, one token after another, for every round</text>
<text x="20.0" y="168.0" text-anchor="start" class="fig-bold">DPO (the next chapter) keeps two</text>
<rect x="20.0" y="180.0" width="108.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="74.0" y="207.0" text-anchor="middle" class="fig-small">policy</text>
<rect x="140.0" y="180.0" width="108.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="194.0" y="207.0" text-anchor="middle" class="fig-small">reference</text>
<text x="20.0" y="258.0" text-anchor="start" class="fig-small">and never generates text during training</text>
<rect x="330.0" y="200.0" width="14.0" height="14.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="350.0" y="212.0" text-anchor="start" class="fig-small">trained</text>
<rect x="330.0" y="224.0" width="14.0" height="14.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="350.0" y="236.0" text-anchor="start" class="fig-small">frozen</text>
</svg>
Figure 3. The models PPO keeps in memory, against the two that DPO keeps. If each model were the same size, PPO would hold twice as many models as DPO. The larger cost is time: every round starts by generating whole answers one token at a time. In a real system the value head often shares the policy's body, so the counts are a guide.
```

That cost is why the next chapter asks a different question: can you reach the same best policy without a reward model, a value head, or a sampling loop?

> [!KEY]
> PPO moves toward the leashed optimum from samples. It subtracts a baseline so a sample is pushed up only if it beat expectations, and it clips the probability ratio so one batch cannot pull the policy far. The price is four models and a generation loop.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The probability ratio is `np.exp(new_logps - old_logps)`, and it is 1 when nothing has changed.
> - The advantage is the reward minus the value guess, recentred and divided by its standard deviation, and it is 0 when every sample matches its guess.
> - The clipped term is `min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)`, averaged over the batch.
> - A positive advantage stops pulling above $1 + \epsilon$. A negative advantage stops pulling below $1 - \epsilon$, and keeps pulling above $1 + \epsilon$.

## Build it step by step

The checks call your functions in this order:

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

## Where this breaks

A weak reward model is still the weakest link. The leash and the clip slow down a policy that chases its blind spots, and neither removes them. The value head must learn at the pace of the policy, and when it lags the advantages mislead. Because each step is small, PPO needs many rounds, and each round needs fresh generation. Most of the cost sits in the loop, not in the three functions you write.

## The exercise

You build `policy_ratio`, `advantage`, and `clipped_objective`. The checks verify the hand-worked ratios, the advantages $[1, 1, -1, -1]$ and the zero-variance case, and the objective value $0.05$ on the four samples. Two checks use finite differences on your objective: they confirm that the gradient is 0 in the flat regions and nonzero for a bad answer whose ratio went above $1 + \epsilon$. A last check runs gradient ascent on a single ratio and confirms it stops near $1 + \epsilon$.
