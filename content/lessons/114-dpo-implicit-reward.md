---
title: "DPO: training on the implicit reward"
slug: 114-dpo-implicit-reward
part: "Part 7 — Large language models"
summary: Run the best-policy formula backwards to read a reward off any policy, then train the policy directly on preference pairs, with no reward model and no sampling.
minutes: 40
gpu: false
objectives:
  - Solve the best-policy formula for the reward, and explain why a policy's log ratio to the reference is a reward in disguise.
  - Show with numbers why the unknown normaliser Z(x) cancels in a comparison, and say why that is what makes direct training possible.
  - Compute the implicit reward, the DPO loss, and the gradient with respect to the policy's log probabilities.
  - Explain the gradient as a per-pair weight that fades as a pair becomes ranked right, and say what the exercise's free-variable model leaves out.
prereqs:
  - 113-ppo-the-sampling-loop
  - 112-kl-leash-and-optimal-policy
  - 111-reward-from-preferences
exercise: 114-dpo-implicit-reward
---

# DPO: training on the implicit reward

You have pairs for the mortgage prompt. In each, a person preferred the helpful answer to a padded one. The route from the last two chapters has two stages. First, fit a reward model on the pairs. Second, run PPO against that reward model, with a leash to the reference. The second stage needs four models in memory, a value head, and a loop that generates whole answers every round. Each stage has its own ways to go wrong, and errors in the first carry into the second.

This chapter asks whether the middle can be skipped. The best policy for a reward is the reference reweighted by $e^{r/\beta}$. If a reward gives you the best policy, then the best policy should give you the reward back, and then you could train the policy straight from the pairs.

Why not fine-tune on the chosen answers alone? Because that raises the chosen answer's probability and never lowers the rejected one, and it knows nothing about the reference or the leash. The pairs say more: the chosen answer is better than the rejected one, by some margin, relative to where the policy started. The rest of the chapter shows how to use that margin directly.

## What you already know

- The Bradley-Terry loss is the mean of $-\log \sigma(r_w - r_l)$, and it depends only on the gap between the two rewards.
- The best policy on a leash is the reference multiplied by $e^{r/\beta}$ and divided by a total $Z(x)$ that sums over every possible answer.
- A log probability is the logarithm of a probability, and $\log(a/b) = \log a - \log b$.
- PPO needs a reward model, a value head, and a sampling loop.
- The gradient of the mean Bradley-Terry loss with respect to the chosen reward is $(\sigma(z) - 1)/n$.

## The picture: remove the stages in the middle

Figure 1 puts the two pipelines side by side. DPO, direct preference optimisation, deletes the reward model, the sampling, and the value head. What remains is a loss that reads the log probabilities of the chosen and rejected answers under the policy being trained and under the frozen reference.

```figure
<svg viewBox="0 0 600 356" role="img" aria-label="The PPO pipeline of five stages next to the DPO pipeline of three stages after the preference pairs">
<defs><marker id="c128-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="140.0" y1="85.0" x2="140.0" y2="97.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
<line x1="140.0" y1="145.0" x2="140.0" y2="157.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
<line x1="140.0" y1="205.0" x2="140.0" y2="217.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
<line x1="140.0" y1="265.0" x2="140.0" y2="277.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
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
<line x1="450.0" y1="85.0" x2="450.0" y2="97.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
<line x1="450.0" y1="145.0" x2="450.0" y2="157.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
<line x1="450.0" y1="205.0" x2="450.0" y2="217.0" class="stroke-muted" marker-end="url(#c128-arrow)" stroke-width="2"/>
<text x="330.0" y="290.0" text-anchor="start" class="fig-small">no reward model, no sampling,</text>
<text x="330.0" y="306.0" text-anchor="start" class="fig-small">no value head: two models in memory</text>
<rect x="20.0" y="332.0" width="14.0" height="14.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="40.0" y="344.0" text-anchor="start" class="fig-small">stage DPO removes</text>
<rect x="200.0" y="332.0" width="14.0" height="14.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="220.0" y="344.0" text-anchor="start" class="fig-small">stage that stays</text>
</svg>
Figure 1. The two pipelines side by side. Orange boxes are the stages PPO needs and DPO does not: fitting a reward model, sampling answers, and scoring them against a value head. DPO goes from preference pairs straight to one loss on the policy's and the reference's log probabilities.
```

## Run the formula backwards

Start from the best policy of the last chapter:

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}}{Z(x)}
$$

Take the log of both sides. The log of a product is a sum, and the log of a division is a difference:

$$
\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{r(x, y)}{\beta} - \log Z(x)
$$

Solve for $r$:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Read it aloud: the reward of an answer is $\beta$ times the log of how much the best policy boosted it, plus a number that depends on the prompt and not on the answer. Check it on the three answers of the last chapter, with $\beta = 1$. The best policy was $[0.731, 0.161, 0.108]$ against the reference $[0.5, 0.3, 0.2]$. The log ratios are $[0.380, -0.620, -0.620]$. Add $\log Z = 0.620$ and you recover the rewards $[1, 0, 0]$. A policy is a reward model in disguise.

## The constant cancels

The extra term $\beta \log Z(x)$ is the one thing you cannot compute, because $Z(x)$ sums over every answer. It is, however, the same for every answer to the same prompt. A pair compares two answers to one prompt, and the Bradley-Terry loss only sees the gap, so the term drops out:

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

Check the example. The log ratios $0.380$ and $-0.620$ differ by $1.000$, and the rewards $1$ and $0$ differ by $1$. This is the second time that only gaps matter, and it does real work: PPO got around $Z$ by sampling, and DPO gets around it by never needing it.

## The implicit reward and the DPO loss

Next, replace the best policy with the policy $\pi_\theta$ you are training. Any policy has an implicit reward, the reward for which it would be the best policy:

$$
\hat r(x, y) = \beta \left(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\right)
$$

Read it aloud: $\beta$ times how much more log probability the trained policy gives the answer than the reference does. In numpy: `beta * (policy_logps - ref_logps)`. The DPO loss is Bradley-Terry applied to these rewards, and training adjusts $\theta$ so that the policy's own implicit reward ranks the pairs as people did:

$$
\text{DPO loss} = \frac{1}{n}\sum_{i=1}^{n} -\log \sigma(u_i), \qquad u_i = \hat r_{w,i} - \hat r_{l,i}
$$

The margin $u_i$ is the implicit reward of the chosen answer minus that of the rejected one. Why is this the same problem PPO solves? Because the best policy for the reward the pairs imply is the policy whose implicit reward agrees with the pairs, and the Bradley-Terry loss is the loss that makes the reward agree with the pairs. Nothing here needs a sample, a reward model, or $Z$.

Take $\beta = 1$ and one pair, helpful over padded. The reference log probabilities are $\ln 0.5 = -0.693$ and $\ln 0.2 = -1.609$. At the start the policy equals the reference, so both implicit rewards are 0, the margin is 0, and the loss is $\log 2 = 0.693$, the price of a tie. Training starts with a tie and ranks the pair right.

## What the gradient does

The loss depends on the policy only through the two log probabilities, and $\hat r$ moves by $\beta$ per unit of log probability. So the chain rule puts a factor $\beta$ on the Bradley-Terry gradient:

$$
\frac{\partial\, \text{loss}}{\partial \log \pi_\theta(y_{w,i})} = \beta\,\frac{\sigma(u_i) - 1}{n}, \qquad
\frac{\partial\, \text{loss}}{\partial \log \pi_\theta(y_{l,i})} = -\beta\,\frac{\sigma(u_i) - 1}{n}
$$

Read it aloud: raise the chosen log probability and lower the rejected one, each by a weight $\beta(1 - \sigma(u))$ that is largest when the pair is ranked wrong and shrinks as it becomes ranked right. The weight is always below $\beta$. It equals $\beta/2$ at $u = 0$, and it approaches $\beta$ only when the pair is very wrong. Figure 2 plots it.

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
Figure 2. The weight DPO gives each pair, β(1 − σ(u)), as a function of the margin u. At u = 0 the weight is half its maximum. It falls as the pair is ranked more and more right, and it stays above 0 but is tiny in the green region. A pair ranked wrong gets a weight close to β.
```

Do one step by hand, with learning rate 0.1. At $u = 0$ the chosen gradient is $-0.5$, so descent raises the chosen log probability by $0.05$ and lowers the rejected one by $0.05$. The margin becomes $u = 0.1$ and the loss falls from $0.693$ to $0.644$. Repeat ten times and the chosen log probability reaches $-0.290$ while the rejected one falls to $-2.013$ (Figure 3). The margin is $0.807$, the loss is $0.369$, and the weight has dropped from $0.5$ to $0.308$.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Log probabilities of a chosen and a rejected answer under the reference and after ten DPO steps">
<defs><marker id="c128-arrow-c" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-c"/></marker><marker id="c128-arrow-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-muted"/>
<line x1="87.9" y1="230.0" x2="87.9" y2="236.0" class="stroke-muted"/>
<text x="87.9" y="252.0" text-anchor="middle" class="fig-small">−2.5</text>
<line x1="177.1" y1="230.0" x2="177.1" y2="236.0" class="stroke-muted"/>
<text x="177.1" y="252.0" text-anchor="middle" class="fig-small">−2</text>
<line x1="266.4" y1="230.0" x2="266.4" y2="236.0" class="stroke-muted"/>
<text x="266.4" y="252.0" text-anchor="middle" class="fig-small">−1.5</text>
<line x1="355.7" y1="230.0" x2="355.7" y2="236.0" class="stroke-muted"/>
<text x="355.7" y="252.0" text-anchor="middle" class="fig-small">−1</text>
<line x1="445.0" y1="230.0" x2="445.0" y2="236.0" class="stroke-muted"/>
<text x="445.0" y="252.0" text-anchor="middle" class="fig-small">−0.5</text>
<line x1="534.3" y1="230.0" x2="534.3" y2="236.0" class="stroke-muted"/>
<text x="534.3" y="252.0" text-anchor="middle" class="fig-small">0</text>
<text x="320.0" y="276.0" text-anchor="middle" class="fig-small">log probability of the answer</text>
<text x="70.0" y="60.0" text-anchor="start" class="fig-bold">chosen (helpful)</text>
<line x1="70.0" y1="90.0" x2="570.0" y2="90.0" class="stroke-grid"/>
<circle cx="410.5" cy="90.0" r="7" class="fill-paper stroke-c "/>
<line x1="418.5" y1="90.0" x2="473.6" y2="90.0" class="stroke-c" marker-end="url(#c128-arrow-c)" stroke-width="3"/>
<circle cx="482.6" cy="90.0" r="7" class="fill-c"/>
<text x="410.5" y="76.0" text-anchor="middle" class="fig-small">reference −0.693</text>
<text x="482.6" y="114.0" text-anchor="middle" class="fig-small fig-bold">policy −0.290</text>
<text x="70.0" y="140.0" text-anchor="start" class="fig-bold">rejected (padded)</text>
<line x1="70.0" y1="170.0" x2="570.0" y2="170.0" class="stroke-grid"/>
<circle cx="246.9" cy="170.0" r="7" class="fill-paper stroke-b "/>
<line x1="238.9" y1="170.0" x2="183.8" y2="170.0" class="stroke-b" marker-end="url(#c128-arrow-b)" stroke-width="3"/>
<circle cx="174.8" cy="170.0" r="7" class="fill-b"/>
<text x="246.9" y="156.0" text-anchor="middle" class="fig-small">reference −1.609</text>
<text x="174.8" y="194.0" text-anchor="middle" class="fig-small fig-bold">policy −2.013</text>
<text x="70.0" y="24.0" text-anchor="start" class="fig-small">after 10 DPO steps on one pair, β = 1, learning rate 0.1</text>
<text x="320.0" y="206.0" text-anchor="start" class="fig-small"></text>
</svg>
Figure 3. One pair, ten steps. The chosen answer's log probability rises from −0.693 to −0.290, and the rejected answer's falls from −1.609 to −2.013. Each moves by 0.404, so the implicit rewards are +0.404 and −0.404, and the margin is u = 0.807.
```

The weight is why DPO does not overdo it. A pair that is already ranked right gets a small push, and the pairs that are still wrong get most of the effort. Fifty steps would bring the loss down to $0.106$ and the weight to $0.100$, because the weight never reaches 0.

> [!WARNING]
> The exercise treats the two log probabilities as free numbers that you move directly. A real policy is a network, and the log probabilities of different answers are tied together through shared weights. Free numbers can also drift above 0 in the exercise, which a real log probability never can. In a real run, pushing the rejected answer down can drag the chosen answer down too, so both log probabilities fall while their gap grows. This failure is sometimes called likelihood displacement.

> [!KEY]
> A policy hides a reward: $\beta$ times its log ratio to the reference, plus a prompt-only constant. The constant cancels in a comparison, so Bradley-Terry on these implicit rewards trains the policy straight from preference pairs. That gives PPO's goal with two models, no sampling, and no reward model.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The implicit reward is `beta * (policy_logps - ref_logps)`, elementwise.
> - `dpo_loss` is the Bradley-Terry loss of the two implicit rewards, and it is $\log 2$ when the policy equals the reference.
> - `dpo_grad` is $\beta$ times the Bradley-Terry gradient of the implicit rewards, one array for the chosen and one for the rejected log probabilities.
> - Shifting both reference log probabilities of a prompt by one constant changes nothing.

## Build it step by step

The starter file already holds the functions you wrote in the earlier chapters: `preference_probability`, `bradley_terry_loss`, `bradley_terry_grad`, and `optimal_policy`. Use them. The checks call your functions in this order:

1. `implicit_reward(policy_logps, ref_logps, beta)` takes two arrays of the same shape and returns $\beta$ times their difference. The common mistake is to forget the reference and return `beta * policy_logps`:

```python
def implicit_reward(policy_logps, ref_logps, beta):
    return ...  # beta * (policy_logps - ref_logps)
```

2. `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)` returns a float. Compute the two implicit rewards and pass them to `bradley_terry_loss`. The common mistake is to pass the raw log probabilities:

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    r_w = implicit_reward(policy_chosen_logps, ref_chosen_logps, beta)
    r_l = ...  # implicit reward of the rejected answers
    return bradley_terry_loss(r_w, r_l)
```

3. `dpo_grad` takes the same arguments and returns two `(n,)` arrays, the gradients with respect to the policy's chosen and rejected log probabilities. Multiply the gradients from `bradley_terry_grad` by `beta`. The common mistake is to drop $\beta$, which is correct only when $\beta = 1$:

```python
def dpo_grad(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    r_w = ...  # implicit rewards, as above
    r_l = ...
    grad_w, grad_l = bradley_terry_grad(r_w, r_l)
    return ..., ...  # multiply each by beta
```

## Where this breaks

DPO learns the preferences in its pairs, blind spots included, and it has no signal for prompts the pairs do not cover. The pairs were usually written by some other model, so the policy trains on answers it might never write itself. A small $\beta$ lets the policy move far from the reference before the weights fade, and a large $\beta$ keeps it close. Practitioners often add a small supervised loss on the chosen answers to keep their probability from falling.

## The exercise

You write `implicit_reward`, `dpo_loss`, and `dpo_grad`. The checks verify the implicit reward and the planted-reward recovery from `optimal_policy`, the loss against the Bradley-Terry loss of the implicit rewards, the value $\log 2$ at the start, and the claim that shifting both reference log probabilities by one constant changes nothing. They also check the gradient against finite differences, check that the gradient's size is $\beta(1 - \sigma(u))/n$, and run fifty gradient steps to confirm that the loss falls and the chosen answer gains on the rejected one.
