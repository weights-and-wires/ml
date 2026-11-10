---
title: Reward models and the KL leash
slug: 063-reward-models-and-the-kl-leash
part: "Part 7 — Large language models"
summary: Train a scorer from "B beats A" with the Bradley-Terry loss, then keep a model near its reference with a KL leash and a closed-form best policy.
minutes: 80
gpu: false
objectives:
  - Turn a reward gap into the chance that the chosen answer wins with a sigmoid, and compute the Bradley-Terry loss and its gradient without overflow.
  - Explain why a pair that is ranked wrong gets the bigger push, and why only reward gaps matter.
  - Compute the KL divergence between two small distributions by hand and with numpy, including the zero-probability case.
  - Write the leashed objective, expected reward minus beta times KL, and compute the best policy for three answers.
  - Explain what beta controls, and why the normaliser Z(x) cannot be computed over all sentences.
prereqs:
  - 062-fine-tuning-and-lora
  - 008-sigmoid-and-log-loss
  - 031-softmax-and-whole-network-gradients
  - 057-next-token-loss-and-safe-steps
exercise: 063-reward-models-and-the-kl-leash
---

# Reward models and the KL leash

A user asks an assistant, "Explain a mortgage to me." Two answers come back. Answer A says, "A mortgage is a loan for a house." Answer B says what the loan is, how each monthly payment splits into interest and repayment, and what happens if you stop paying. A reader picks B in two seconds.

Supervised fine-tuning cannot make that choice. It copies one written answer, token by token, so it knows only what the writer wrote and cannot say that B beats A. To use it, a person would have to write the best answer to thousands of prompts, which is slow even for an expert. Picking the better of two answers takes seconds, so people can make millions of those choices.

This chapter answers two questions. First, how do you turn "B is better than A" into a number a model can train on? Give every answer a score, turn the gap between two scores into the chance that a person picks the higher one, and reward the scorer for giving a high chance to the choice people made. Second, what happens when a language model chases that scorer? The scorer learned from a finite pile of comparisons, so it has blind spots, and a model that climbs its score freely finds them. The fix is a leash to a reference model, and the best policy on that leash has a closed form.

## What you already know

- A sigmoid squashes any real number into a value between 0 and 1, and the input 0 gives $1/2$, as in the chapter on [sigmoid and log loss](/lessons/008-sigmoid-and-log-loss).
- A softmax turns a list of scores into probabilities that add to 1, and adding a constant to every score changes nothing.
- The loss $-\log p$ scores how much probability a model gave to what actually happened, and multiplying probabilities adds their logs.
- A gradient says how much the loss changes when you nudge one number, and gradient descent moves each number against it.
- A mean over $n$ examples puts a factor $1/n$ on each example's gradient.

## The picture: a rating for each answer

Why not ask people to score each answer from 1 to 10? Because raters disagree about what a 7 means, while two answers side by side get a much steadier verdict. So we keep the comparison, and we still want a number per answer, because a model that outputs one number is easy to train.

Chess shows how to join the two. Every player has a rating, and the chance that one player beats another depends only on the gap between their ratings. A big gap means a likely win, equal ratings mean even odds, and nobody's win chance is exactly 0 or 1. Give every answer a rating in the same way and call it the reward, written $r$. A reward network reads the prompt and an answer and outputs one number, and a higher number means a better answer. Because only gaps matter, the numbers have no meaning on their own, as a rating of 1600 means nothing without someone to compare against.

Figure 1 draws the rule. The horizontal axis is the gap between the chosen answer's reward and the rejected answer's reward. The vertical axis is the chance that a person picks the chosen answer.

```figure
<svg viewBox="0 0 600 284" role="img" aria-label="The sigmoid curve of the reward gap, with the gaps 0, 2 and minus 1 marked">
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="135.0" x2="560.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<line x1="315.0" y1="40.0" x2="315.0" y2="230.0" class="stroke-grid"/>
<text x="119.0" y="248.0" text-anchor="middle" class="fig-small">−4</text>
<text x="217.0" y="248.0" text-anchor="middle" class="fig-small">−2</text>
<text x="315.0" y="248.0" text-anchor="middle" class="fig-small">0</text>
<text x="413.0" y="248.0" text-anchor="middle" class="fig-small">2</text>
<text x="511.0" y="248.0" text-anchor="middle" class="fig-small">4</text>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="40.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<polyline points="70.0,228.7 74.9,228.6 79.8,228.4 84.7,228.3 89.6,228.1 94.5,227.9 99.4,227.7 104.3,227.5 109.2,227.2 114.1,226.9 119.0,226.6 123.9,226.2 128.8,225.8 133.7,225.4 138.6,224.9 143.5,224.4 148.4,223.9 153.3,223.2 158.2,222.6 163.1,221.8 168.0,221.0 172.9,220.1 177.8,219.1 182.7,218.0 187.6,216.9 192.5,215.6 197.4,214.2 202.3,212.7 207.2,211.0 212.1,209.3 217.0,207.4 221.9,205.3 226.8,203.0 231.7,200.7 236.6,198.1 241.5,195.3 246.4,192.4 251.3,189.3 256.2,186.0 261.1,182.5 266.0,178.9 270.9,175.1 275.8,171.1 280.7,167.0 285.6,162.7 290.5,158.3 295.4,153.8 300.3,149.1 305.2,144.5 310.1,139.7 315.0,135.0 319.9,130.3 324.8,125.5 329.7,120.9 334.6,116.2 339.5,111.7 344.4,107.3 349.3,103.0 354.2,98.9 359.1,94.9 364.0,91.1 368.9,87.5 373.8,84.0 378.7,80.7 383.6,77.6 388.5,74.7 393.4,71.9 398.3,69.3 403.2,67.0 408.1,64.7 413.0,62.6 417.9,60.7 422.8,59.0 427.7,57.3 432.6,55.8 437.5,54.4 442.4,53.1 447.3,52.0 452.2,50.9 457.1,49.9 462.0,49.0 466.9,48.2 471.8,47.4 476.7,46.8 481.6,46.1 486.5,45.6 491.4,45.1 496.3,44.6 501.2,44.2 506.1,43.8 511.0,43.4 515.9,43.1 520.8,42.8 525.7,42.5 530.6,42.3 535.5,42.1 540.4,41.9 545.3,41.7 550.2,41.6 555.1,41.4 560.0,41.3" fill="none" stroke-width="2.5" class="stroke-b"/>
<line x1="315.0" y1="135.0" x2="315.0" y2="230.0" class="stroke-muted dashed"/>
<circle cx="315.0" cy="135.0" r="6" class="fill-a"/>
<text x="329.0" y="157.0" text-anchor="start" class="fig-bold">gap 0 → 0.5</text>
<line x1="413.0" y1="62.6" x2="413.0" y2="230.0" class="stroke-muted dashed"/>
<circle cx="413.0" cy="62.6" r="6" class="fill-a"/>
<text x="425.0" y="84.6" text-anchor="start" class="fig-bold">gap 2 → 0.881</text>
<line x1="266.0" y1="178.9" x2="266.0" y2="230.0" class="stroke-muted dashed"/>
<circle cx="266.0" cy="178.9" r="6" class="fill-a"/>
<text x="254.0" y="168.0" text-anchor="end" class="fig-bold">gap −1 → 0.269</text>
<text x="315.0" y="272.0" text-anchor="middle" class="fig-small">reward gap: chosen reward minus rejected reward</text>
<text x="70.0" y="24.0" text-anchor="start" class="fig-small">chance the person picks the chosen answer</text>
</svg>
Figure 1. The chance that a person picks the chosen answer, as a function of the reward gap. A gap of 0 gives even odds, 0.5. A gap of 2 gives 0.881. The curve never reaches 0 or 1, and a gap of −1 still leaves a 0.269 chance for the chosen answer.
```

## From a gap to a chance, and then to a loss

Four things must hold for the curve in Figure 1. Any gap must give a number between 0 and 1. A gap of 0 must give even odds. A bigger gap must give more confidence. Swapping the two answers must swap the chance $p$ for $1 - p$. The sigmoid $\sigma(z) = 1/(1 + e^{-z})$ does all four, and it is a softmax over two answers, as the next formula shows. This rule is the Bradley-Terry model.

Take the mortgage prompt, with reward 2 for B and 0 for A. The gap is 2, $e^{-2} \approx 0.1353$, and $\sigma(2) = 1/1.1353 \approx 0.8808$, so the model says a person picks B about 88 times in 100. With $r_w$ for the chosen reward and $r_l$ for the rejected one,

$$
p(\text{chosen wins}) = \sigma(r_w - r_l) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}}
$$

Read it aloud: the chance that the chosen answer wins is the sigmoid of the reward gap, which is also a softmax over the two answers. Check it: $e^{2}/(e^{2} + e^{0}) = 7.389/8.389 = 0.8808$. In numpy the sigmoid must survive large gaps, and `np.exp(-np.logaddexp(0, -z))` does, because `np.logaddexp(0, -z)` computes $\log(1 + e^{-z})$ without forming the huge number $e^{-z}$.

Next, score the scorer. A person chose B, and the model said B wins with chance 0.88, which is good. At 0.27 it would be bad. The number that is small in the first case and large in the second is $-\log p$, with $p$ the chance the model gave to what the person chose. The table works it out for five gaps:

| Gap $z$ | Chance $\sigma(z)$ | Loss $-\log \sigma(z)$ |
|---|---|---|
| 3 | 0.9526 | 0.0486 |
| 1 | 0.7311 | 0.3133 |
| 0 | 0.5000 | 0.6931 |
| −1 | 0.2689 | 1.3133 |
| −3 | 0.0474 | 3.0486 |

On the right the loss is almost flat, so a pair that is already ranked right costs almost nothing. On the left it is close to a straight line of slope −1, so a pair ranked confidently wrong costs about as much as the gap is wrong. A tie costs $\log 2 \approx 0.6931$, the price of a coin flip. A hard cutoff, "the chosen reward must beat the rejected one", would have no slope and give no signal about how to fix a wrong pair. The smooth curve gives a push everywhere, and the push fades only when the pair is already right.

With $n$ pairs, average the loss. Take two prompts with chosen rewards $[2, 0]$ and rejected rewards $[1, 1]$, so the gaps are $[1, -1]$. The losses are $0.3133$ and $1.3133$, and their mean is $0.8133$:

$$
\text{loss} = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(z_i), \qquad z_i = r_{w,i} - r_{l,i}
$$

Read it aloud: for each pair, take the gap, take the negative log of the sigmoid, and average. In numpy: `np.mean(np.logaddexp(0, -z))`.

> [!WARNING]
> The tempting version, `np.log(1 + np.exp(-z))`, overflows when $z$ is very negative: `np.exp(800)` is `inf`. Use `np.logaddexp(0, -z)`, which returns the right answer, 800, for $z = -800$.

## Push the chosen reward up and the rejected one down

The derivative of $-\log \sigma(z)$ with respect to $z$ is $\sigma(z) - 1$, because the sigmoid's own slope is $\sigma(z)(1 - \sigma(z))$ and dividing that by $\sigma(z)$ leaves $1 - \sigma(z)$, with a minus sign from the $-\log$. This is minus the chance the model gave to the wrong outcome. The gap is $r_w - r_l$, so the chosen reward gets $\sigma(z) - 1$, the rejected reward gets the opposite sign, and the mean puts $1/n$ on both:

$$
\frac{\partial\, \text{loss}}{\partial r_{w,i}} = \frac{\sigma(z_i) - 1}{n}, \qquad
\frac{\partial\, \text{loss}}{\partial r_{l,i}} = -\frac{\sigma(z_i) - 1}{n}
$$

Read it aloud: the push on a reward is the chance the model gave to the wrong outcome, divided by $n$. On the two pairs, $\sigma(1) - 1 = -0.2689$ and $\sigma(-1) - 1 = -0.7311$, so after dividing by $n = 2$ the chosen gradients are $[-0.1345, -0.3655]$ and the rejected gradients are $[+0.1345, +0.3655]$. Gradient descent raises a reward with a negative gradient and lowers one with a positive gradient. In numpy: `(preference_probability(rc, rr) - 1) / n`, and its negation.

```figure
<svg viewBox="0 0 600 348" role="img" aria-label="One gradient step on two pairs of rewards, with arrows for how far each reward moves">
<defs><marker id="c053-arrow-c" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-c"/></marker><marker id="c053-arrow-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<line x1="70.0" y1="40.0" x2="70.0" y2="300.0" class="stroke-muted"/>
<line x1="70.0" y1="262.9" x2="560.0" y2="262.9" class="stroke-grid"/>
<text x="62.0" y="266.9" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="188.6" x2="560.0" y2="188.6" class="stroke-grid"/>
<text x="62.0" y="192.6" text-anchor="end" class="fig-small">1</text>
<line x1="70.0" y1="114.3" x2="560.0" y2="114.3" class="stroke-grid"/>
<text x="62.0" y="118.3" text-anchor="end" class="fig-small">2</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">3</text>
<text x="70.0" y="24.0" text-anchor="start" class="fig-small">reward</text>
<text x="190.0" y="334.0" text-anchor="middle" class="fig-bold">pair 1: gap +1</text>
<circle cx="135.0" cy="114.3" r="6" class="fill-c "/>
<line x1="135.0" y1="114.3" x2="135.0" y2="88.3" class="stroke-c" marker-end="url(#c053-arrow-c)" stroke-width="2.5"/>
<circle cx="135.0" cy="94.3" r="4" class="fill-sunk stroke-c"/>
<text x="125.0" y="118.3" text-anchor="end" class="fig-small">2</text>
<text x="125.0" y="98.3" text-anchor="end" class="fig-bold fig-small">2.27</text>
<circle cx="245.0" cy="188.6" r="6" class="fill-b "/>
<line x1="245.0" y1="188.6" x2="245.0" y2="214.5" class="stroke-b" marker-end="url(#c053-arrow-b)" stroke-width="2.5"/>
<circle cx="245.0" cy="208.5" r="4" class="fill-sunk stroke-b"/>
<text x="255.0" y="192.6" text-anchor="start" class="fig-small">1</text>
<text x="255.0" y="212.5" text-anchor="start" class="fig-bold fig-small">0.73</text>
<text x="135.0" y="300.0" text-anchor="middle" class="fig-small">chosen</text>
<text x="245.0" y="300.0" text-anchor="middle" class="fig-small">rejected</text>
<text x="430.0" y="334.0" text-anchor="middle" class="fig-bold">pair 2: gap −1</text>
<circle cx="375.0" cy="262.9" r="6" class="fill-c "/>
<line x1="375.0" y1="262.9" x2="375.0" y2="202.5" class="stroke-c" marker-end="url(#c053-arrow-c)" stroke-width="2.5"/>
<circle cx="375.0" cy="208.5" r="4" class="fill-sunk stroke-c"/>
<text x="365.0" y="266.9" text-anchor="end" class="fig-small">0</text>
<text x="365.0" y="212.5" text-anchor="end" class="fig-bold fig-small">0.73</text>
<circle cx="485.0" cy="188.6" r="6" class="fill-b "/>
<line x1="485.0" y1="188.6" x2="485.0" y2="248.9" class="stroke-b" marker-end="url(#c053-arrow-b)" stroke-width="2.5"/>
<circle cx="485.0" cy="242.9" r="4" class="fill-sunk stroke-b"/>
<text x="495.0" y="192.6" text-anchor="start" class="fig-small">1</text>
<text x="495.0" y="246.9" text-anchor="start" class="fig-bold fig-small">0.27</text>
<text x="375.0" y="300.0" text-anchor="middle" class="fig-small">chosen</text>
<text x="485.0" y="300.0" text-anchor="middle" class="fig-small">rejected</text>
</svg>
Figure 2. One gradient step with learning rate 2. Filled dots are the rewards before the step, ringed dots after it, and the numbers beside them are the values. Pair 2 starts out ranked wrong, so its rewards move 0.73 each, nearly three times as far as the rewards of pair 1, which move 0.27 each. After one step both pairs are ranked right.
```

The pair that is ranked wrong gets almost three times the push, 0.3655 against 0.1345. Figure 2 shows one step with learning rate 2, large on purpose so the arrows are readable. The loss falls from 0.8133 to 0.3416, and pair 2 is no longer ranked wrong. Adding the same constant to every reward changes nothing, because $z$ is a difference. In a real system the rewards come from a network and its weights get this gradient by backpropagation. In the exercise the rewards are free numbers that you nudge directly.

> [!KEY]
> A person's "B beats A" becomes training data in three moves. A reward network scores each answer, the sigmoid of the gap gives the chance that the chosen answer wins, and the negative log of that chance is the loss. Its gradient pushes the chosen reward up and the rejected reward down, hardest when the pair is wrong. Only gaps matter.

## Why a model on a reward needs a leash

The obvious next move is to fine-tune the language model to make the reward as high as possible. Suppose the reward model likes answers that begin "Great question!" A model that climbs the score freely will find that habit within a few updates and say it to every prompt. The score goes up, and the answers get worse.

Two quick fixes fail. Stopping after a few updates gives no principled stopping point, and the right number changes with every reward model. Penalising how far the weights move fails too, because the same weight movement can change the answers a lot in one place and not at all in another. What we care about is how different the answers are, not how different the weights are. So we leash the model to a reference, the supervised model you started from, and measure the length of the leash in answers.

Call the prompt $x$ and an answer $y$. A policy, written $\pi$, is the language model viewed as a rule that gives every answer a probability, $\pi(y \mid x)$. The reference policy $\pi_{\text{ref}}$ is the supervised model. To keep the numbers small, let one prompt have three possible answers:

| Answer | Reference probability | Reward |
|---|---|---|
| helpful | 0.5 | 1 |
| terse | 0.3 | 0 |
| padded ("Great question!" and more) | 0.2 | 0 |

## How far has the policy drifted?

Why not subtract the two lists of probabilities? A difference of 0.01 is a tiny change for an answer at 0.50 and a big one for an answer at 0.02, because it halves that answer's chance. Probabilities should be compared by ratios, and the logarithm of the ratio is exactly the extra surprise. The policy writes answer $y$ with probability $\pi(y)$ and the reference would write it with $\pi_{\text{ref}}(y)$, so the extra surprise is $\log(\pi(y) / \pi_{\text{ref}}(y))$. Average that over the answers the policy actually writes, weighting each by $\pi(y)$, and you have the KL divergence (Kullback-Leibler divergence):

$$
\text{KL}(\pi \,\|\, \pi_{\text{ref}}) = \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)}
$$

Read it aloud: for each answer, take the policy's probability times the log of the ratio of policy to reference, and add them up. The unit is the nat, because the logarithm is natural.

Try $\pi = [0.7, 0.2, 0.1]$ against the reference $[0.5, 0.3, 0.2]$. The three terms are $0.7 \ln 1.4 = 0.2355$, $0.2 \ln(2/3) = -0.0811$, and $0.1 \ln 0.5 = -0.0693$, which sum to $0.0851$ nats. In numpy: `np.sum(p * np.log(p / q), axis=-1)`.

KL is 0 only when the two distributions are identical, and it is never negative. The order matters: swapping the arguments here gives $0.0920$, not $0.0851$. The policy goes first because it is the one writing answers, so the average runs over its answers. And KL punishes the policy for putting weight where the reference puts little: half the probability on an answer the reference rates at 0.01 costs $0.5 \ln 50 = 1.96$ nats from that one term.

> [!WARNING]
> An answer the policy never writes has $\pi(y) = 0$, and the term $0 \cdot \log 0$ is `nan` in numpy. The correct value is 0, because an answer that never happens adds no surprise. Mask those terms, for example with `np.where(p > 0, p * np.log(p / q), 0.0)`.

## What best on a leash means

Average reward is $\sum_y \pi(y)\, r(y)$. The leash is $\beta$ times the KL divergence, where $\beta$ (beta) is a positive number you choose, the price in reward of each nat of drift:

$$
J(\pi) = \sum_y \pi(y)\, r(y) \;-\; \beta\, \text{KL}(\pi \,\|\, \pi_{\text{ref}})
$$

Read it aloud: the score of a policy is the reward it earns minus the price of how far it moved. With $\beta = 1$, the table shows the trade:

| Policy | Expected reward | KL | $J$ |
|---|---|---|---|
| reference $[0.5, 0.3, 0.2]$ | 0.500 | 0.000 | 0.500 |
| $[0.6, 0.25, 0.15]$ | 0.600 | 0.021 | 0.579 |
| $[0.8, 0.1, 0.1]$ | 0.800 | 0.197 | 0.603 |
| $[0.7, 0.2, 0.1]$ | 0.700 | 0.085 | 0.615 |
| $[1, 0, 0]$ | 1.000 | 0.693 | 0.307 |

Moving toward the helpful answer pays until the leash costs more than it earns. The all-in policy $[1, 0, 0]$ has the most reward and the lowest score. Figure 3 shows the same trade for a policy with two answers.

```figure
<svg viewBox="0 0 600 284" role="img" aria-label="Expected reward, leash cost and their difference as the policy's probability of the helpful answer changes">
<line x1="70.0" y1="198.3" x2="560.0" y2="198.3" class="stroke-grid"/>
<text x="62.0" y="202.3" text-anchor="end" class="fig-small">−0.5</text>
<line x1="70.0" y1="145.6" x2="560.0" y2="145.6" class="stroke-grid"/>
<text x="62.0" y="149.6" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="92.8" x2="560.0" y2="92.8" class="stroke-grid"/>
<text x="62.0" y="96.8" text-anchor="end" class="fig-small">0.5</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="248.0" text-anchor="middle" class="fig-small">0</text>
<text x="192.5" y="248.0" text-anchor="middle" class="fig-small">0.25</text>
<text x="315.0" y="248.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="437.5" y="248.0" text-anchor="middle" class="fig-small">0.75</text>
<text x="560.0" y="248.0" text-anchor="middle" class="fig-small">1</text>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="40.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<polyline points="71.0,145.3 72.6,145.0 74.2,144.6 75.9,144.3 77.5,143.9 79.1,143.6 80.8,143.2 82.4,142.9 84.0,142.5 85.7,142.2 87.3,141.8 88.9,141.5 90.6,141.1 92.2,140.8 93.8,140.4 95.5,140.1 97.1,139.7 98.7,139.4 100.4,139.0 102.0,138.7 103.6,138.3 105.3,138.0 106.9,137.6 108.5,137.3 110.2,136.9 111.8,136.6 113.4,136.2 115.1,135.9 116.7,135.5 118.3,135.1 119.9,134.8 121.6,134.4 123.2,134.1 124.8,133.7 126.5,133.4 128.1,133.0 129.7,132.7 131.4,132.3 133.0,132.0 134.6,131.6 136.3,131.3 137.9,130.9 139.5,130.6 141.2,130.2 142.8,129.9 144.4,129.5 146.1,129.2 147.7,128.8 149.3,128.5 151.0,128.1 152.6,127.8 154.2,127.4 155.9,127.1 157.5,126.7 159.1,126.4 160.8,126.0 162.4,125.7 164.0,125.3 165.6,125.0 167.3,124.6 168.9,124.2 170.5,123.9 172.2,123.5 173.8,123.2 175.4,122.8 177.1,122.5 178.7,122.1 180.3,121.8 182.0,121.4 183.6,121.1 185.2,120.7 186.9,120.4 188.5,120.0 190.1,119.7 191.8,119.3 193.4,119.0 195.0,118.6 196.7,118.3 198.3,117.9 199.9,117.6 201.6,117.2 203.2,116.9 204.8,116.5 206.5,116.2 208.1,115.8 209.7,115.5 211.4,115.1 213.0,114.8 214.6,114.4 216.2,114.1 217.9,113.7 219.5,113.3 221.1,113.0 222.8,112.6 224.4,112.3 226.0,111.9 227.7,111.6 229.3,111.2 230.9,110.9 232.6,110.5 234.2,110.2 235.8,109.8 237.5,109.5 239.1,109.1 240.7,108.8 242.4,108.4 244.0,108.1 245.6,107.7 247.3,107.4 248.9,107.0 250.5,106.7 252.2,106.3 253.8,106.0 255.4,105.6 257.1,105.3 258.7,104.9 260.3,104.6 262.0,104.2 263.6,103.9 265.2,103.5 266.8,103.2 268.5,102.8 270.1,102.4 271.7,102.1 273.4,101.7 275.0,101.4 276.6,101.0 278.3,100.7 279.9,100.3 281.5,100.0 283.2,99.6 284.8,99.3 286.4,98.9 288.1,98.6 289.7,98.2 291.3,97.9 293.0,97.5 294.6,97.2 296.2,96.8 297.9,96.5 299.5,96.1 301.1,95.8 302.8,95.4 304.4,95.1 306.0,94.7 307.7,94.4 309.3,94.0 310.9,93.7 312.6,93.3 314.2,93.0 315.8,92.6 317.4,92.3 319.1,91.9 320.7,91.5 322.3,91.2 324.0,90.8 325.6,90.5 327.2,90.1 328.9,89.8 330.5,89.4 332.1,89.1 333.8,88.7 335.4,88.4 337.0,88.0 338.7,87.7 340.3,87.3 341.9,87.0 343.6,86.6 345.2,86.3 346.8,85.9 348.5,85.6 350.1,85.2 351.7,84.9 353.4,84.5 355.0,84.2 356.6,83.8 358.3,83.5 359.9,83.1 361.5,82.8 363.2,82.4 364.8,82.1 366.4,81.7 368.0,81.4 369.7,81.0 371.3,80.6 372.9,80.3 374.6,79.9 376.2,79.6 377.8,79.2 379.5,78.9 381.1,78.5 382.7,78.2 384.4,77.8 386.0,77.5 387.6,77.1 389.3,76.8 390.9,76.4 392.5,76.1 394.2,75.7 395.8,75.4 397.4,75.0 399.1,74.7 400.7,74.3 402.3,74.0 404.0,73.6 405.6,73.3 407.2,72.9 408.9,72.6 410.5,72.2 412.1,71.9 413.8,71.5 415.4,71.2 417.0,70.8 418.6,70.5 420.3,70.1 421.9,69.7 423.5,69.4 425.2,69.0 426.8,68.7 428.4,68.3 430.1,68.0 431.7,67.6 433.3,67.3 435.0,66.9 436.6,66.6 438.2,66.2 439.9,65.9 441.5,65.5 443.1,65.2 444.8,64.8 446.4,64.5 448.0,64.1 449.7,63.8 451.3,63.4 452.9,63.1 454.6,62.7 456.2,62.4 457.8,62.0 459.5,61.7 461.1,61.3 462.7,61.0 464.4,60.6 466.0,60.3 467.6,59.9 469.2,59.6 470.9,59.2 472.5,58.8 474.1,58.5 475.8,58.1 477.4,57.8 479.0,57.4 480.7,57.1 482.3,56.7 483.9,56.4 485.6,56.0 487.2,55.7 488.8,55.3 490.5,55.0 492.1,54.6 493.7,54.3 495.4,53.9 497.0,53.6 498.6,53.2 500.3,52.9 501.9,52.5 503.5,52.2 505.2,51.8 506.8,51.5 508.4,51.1 510.1,50.8 511.7,50.4 513.3,50.1 514.9,49.7 516.6,49.4 518.2,49.0 519.8,48.6 521.5,48.3 523.1,47.9 524.7,47.6 526.4,47.2 528.0,46.9 529.6,46.5 531.3,46.2 532.9,45.8 534.5,45.5 536.2,45.1 537.8,44.8 539.4,44.4 541.1,44.1 542.7,43.7 544.3,43.4 546.0,43.0 547.6,42.7 549.2,42.3 550.9,42.0 552.5,41.6 554.1,41.3 555.8,40.9 557.4,40.6 559.0,40.2" fill="none" stroke-width="2" class="stroke-a"/>
<polyline points="71.0,73.9 72.6,75.9 74.2,77.6 75.9,79.2 77.5,80.8 79.1,82.2 80.8,83.5 82.4,84.9 84.0,86.1 85.7,87.3 87.3,88.5 88.9,89.7 90.6,90.8 92.2,91.9 93.8,92.9 95.5,94.0 97.1,95.0 98.7,95.9 100.4,96.9 102.0,97.9 103.6,98.8 105.3,99.7 106.9,100.6 108.5,101.5 110.2,102.3 111.8,103.2 113.4,104.0 115.1,104.8 116.7,105.6 118.3,106.4 119.9,107.2 121.6,107.9 123.2,108.7 124.8,109.4 126.5,110.1 128.1,110.8 129.7,111.5 131.4,112.2 133.0,112.9 134.6,113.6 136.3,114.2 137.9,114.9 139.5,115.5 141.2,116.1 142.8,116.7 144.4,117.4 146.1,118.0 147.7,118.5 149.3,119.1 151.0,119.7 152.6,120.3 154.2,120.8 155.9,121.4 157.5,121.9 159.1,122.4 160.8,123.0 162.4,123.5 164.0,124.0 165.6,124.5 167.3,125.0 168.9,125.5 170.5,126.0 172.2,126.4 173.8,126.9 175.4,127.4 177.1,127.8 178.7,128.3 180.3,128.7 182.0,129.1 183.6,129.5 185.2,130.0 186.9,130.4 188.5,130.8 190.1,131.2 191.8,131.6 193.4,132.0 195.0,132.3 196.7,132.7 198.3,133.1 199.9,133.4 201.6,133.8 203.2,134.1 204.8,134.5 206.5,134.8 208.1,135.2 209.7,135.5 211.4,135.8 213.0,136.1 214.6,136.4 216.2,136.7 217.9,137.0 219.5,137.3 221.1,137.6 222.8,137.9 224.4,138.2 226.0,138.4 227.7,138.7 229.3,139.0 230.9,139.2 232.6,139.5 234.2,139.7 235.8,139.9 237.5,140.2 239.1,140.4 240.7,140.6 242.4,140.8 244.0,141.1 245.6,141.3 247.3,141.5 248.9,141.7 250.5,141.9 252.2,142.0 253.8,142.2 255.4,142.4 257.1,142.6 258.7,142.7 260.3,142.9 262.0,143.1 263.6,143.2 265.2,143.4 266.8,143.5 268.5,143.6 270.1,143.8 271.7,143.9 273.4,144.0 275.0,144.1 276.6,144.3 278.3,144.4 279.9,144.5 281.5,144.6 283.2,144.7 284.8,144.8 286.4,144.8 288.1,144.9 289.7,145.0 291.3,145.1 293.0,145.1 294.6,145.2 296.2,145.2 297.9,145.3 299.5,145.3 301.1,145.4 302.8,145.4 304.4,145.5 306.0,145.5 307.7,145.5 309.3,145.5 310.9,145.5 312.6,145.6 314.2,145.6 315.8,145.6 317.4,145.6 319.1,145.5 320.7,145.5 322.3,145.5 324.0,145.5 325.6,145.5 327.2,145.4 328.9,145.4 330.5,145.3 332.1,145.3 333.8,145.2 335.4,145.2 337.0,145.1 338.7,145.1 340.3,145.0 341.9,144.9 343.6,144.8 345.2,144.8 346.8,144.7 348.5,144.6 350.1,144.5 351.7,144.4 353.4,144.3 355.0,144.1 356.6,144.0 358.3,143.9 359.9,143.8 361.5,143.6 363.2,143.5 364.8,143.4 366.4,143.2 368.0,143.1 369.7,142.9 371.3,142.7 372.9,142.6 374.6,142.4 376.2,142.2 377.8,142.0 379.5,141.9 381.1,141.7 382.7,141.5 384.4,141.3 386.0,141.1 387.6,140.8 389.3,140.6 390.9,140.4 392.5,140.2 394.2,139.9 395.8,139.7 397.4,139.5 399.1,139.2 400.7,139.0 402.3,138.7 404.0,138.4 405.6,138.2 407.2,137.9 408.9,137.6 410.5,137.3 412.1,137.0 413.8,136.7 415.4,136.4 417.0,136.1 418.6,135.8 420.3,135.5 421.9,135.2 423.5,134.8 425.2,134.5 426.8,134.1 428.4,133.8 430.1,133.4 431.7,133.1 433.3,132.7 435.0,132.3 436.6,132.0 438.2,131.6 439.9,131.2 441.5,130.8 443.1,130.4 444.8,130.0 446.4,129.5 448.0,129.1 449.7,128.7 451.3,128.3 452.9,127.8 454.6,127.4 456.2,126.9 457.8,126.4 459.5,126.0 461.1,125.5 462.7,125.0 464.4,124.5 466.0,124.0 467.6,123.5 469.2,123.0 470.9,122.4 472.5,121.9 474.1,121.4 475.8,120.8 477.4,120.3 479.0,119.7 480.7,119.1 482.3,118.5 483.9,118.0 485.6,117.4 487.2,116.7 488.8,116.1 490.5,115.5 492.1,114.9 493.7,114.2 495.4,113.6 497.0,112.9 498.6,112.2 500.3,111.5 501.9,110.8 503.5,110.1 505.2,109.4 506.8,108.7 508.4,107.9 510.1,107.2 511.7,106.4 513.3,105.6 514.9,104.8 516.6,104.0 518.2,103.2 519.8,102.3 521.5,101.5 523.1,100.6 524.7,99.7 526.4,98.8 528.0,97.9 529.6,96.9 531.3,95.9 532.9,95.0 534.5,94.0 536.2,92.9 537.8,91.9 539.4,90.8 541.1,89.7 542.7,88.5 544.3,87.3 546.0,86.1 547.6,84.9 549.2,83.5 550.9,82.2 552.5,80.8 554.1,79.2 555.8,77.6 557.4,75.9 559.0,73.9" fill="none" stroke-width="2" class="stroke-d"/>
<polyline points="71.0,217.0 72.6,214.7 74.2,212.6 75.9,210.6 77.5,208.7 79.1,207.0 80.8,205.2 82.4,203.6 84.0,202.0 85.7,200.4 87.3,198.9 88.9,197.4 90.6,195.9 92.2,194.5 93.8,193.1 95.5,191.7 97.1,190.3 98.7,189.0 100.4,187.7 102.0,186.4 103.6,185.1 105.3,183.8 106.9,182.6 108.5,181.4 110.2,180.1 111.8,179.0 113.4,177.8 115.1,176.6 116.7,175.5 118.3,174.3 119.9,173.2 121.6,172.1 123.2,171.0 124.8,169.9 126.5,168.8 128.1,167.8 129.7,166.7 131.4,165.7 133.0,164.6 134.6,163.6 136.3,162.6 137.9,161.6 139.5,160.6 141.2,159.7 142.8,158.7 144.4,157.7 146.1,156.8 147.7,155.8 149.3,154.9 151.0,154.0 152.6,153.1 154.2,152.1 155.9,151.2 157.5,150.3 159.1,149.5 160.8,148.6 162.4,147.7 164.0,146.9 165.6,146.0 167.3,145.2 168.9,144.3 170.5,143.5 172.2,142.7 173.8,141.8 175.4,141.0 177.1,140.2 178.7,139.4 180.3,138.6 182.0,137.9 183.6,137.1 185.2,136.3 186.9,135.6 188.5,134.8 190.1,134.1 191.8,133.3 193.4,132.6 195.0,131.8 196.7,131.1 198.3,130.4 199.9,129.7 201.6,129.0 203.2,128.3 204.8,127.6 206.5,126.9 208.1,126.2 209.7,125.5 211.4,124.9 213.0,124.2 214.6,123.5 216.2,122.9 217.9,122.2 219.5,121.6 221.1,120.9 222.8,120.3 224.4,119.7 226.0,119.1 227.7,118.4 229.3,117.8 230.9,117.2 232.6,116.6 234.2,116.0 235.8,115.4 237.5,114.9 239.1,114.3 240.7,113.7 242.4,113.1 244.0,112.6 245.6,112.0 247.3,111.5 248.9,110.9 250.5,110.4 252.2,109.8 253.8,109.3 255.4,108.8 257.1,108.2 258.7,107.7 260.3,107.2 262.0,106.7 263.6,106.2 265.2,105.7 266.8,105.2 268.5,104.7 270.1,104.2 271.7,103.7 273.4,103.3 275.0,102.8 276.6,102.3 278.3,101.9 279.9,101.4 281.5,101.0 283.2,100.5 284.8,100.1 286.4,99.7 288.1,99.2 289.7,98.8 291.3,98.4 293.0,98.0 294.6,97.5 296.2,97.1 297.9,96.7 299.5,96.3 301.1,95.9 302.8,95.5 304.4,95.2 306.0,94.8 307.7,94.4 309.3,94.0 310.9,93.7 312.6,93.3 314.2,93.0 315.8,92.6 317.4,92.3 319.1,91.9 320.7,91.6 322.3,91.2 324.0,90.9 325.6,90.6 327.2,90.3 328.9,90.0 330.5,89.6 332.1,89.3 333.8,89.0 335.4,88.7 337.0,88.5 338.7,88.2 340.3,87.9 341.9,87.6 343.6,87.3 345.2,87.1 346.8,86.8 348.5,86.6 350.1,86.3 351.7,86.1 353.4,85.8 355.0,85.6 356.6,85.3 358.3,85.1 359.9,84.9 361.5,84.7 363.2,84.5 364.8,84.2 366.4,84.0 368.0,83.8 369.7,83.6 371.3,83.5 372.9,83.3 374.6,83.1 376.2,82.9 377.8,82.8 379.5,82.6 381.1,82.4 382.7,82.3 384.4,82.1 386.0,82.0 387.6,81.8 389.3,81.7 390.9,81.6 392.5,81.5 394.2,81.3 395.8,81.2 397.4,81.1 399.1,81.0 400.7,80.9 402.3,80.8 404.0,80.7 405.6,80.7 407.2,80.6 408.9,80.5 410.5,80.4 412.1,80.4 413.8,80.3 415.4,80.3 417.0,80.2 418.6,80.2 420.3,80.2 421.9,80.1 423.5,80.1 425.2,80.1 426.8,80.1 428.4,80.1 430.1,80.1 431.7,80.1 433.3,80.1 435.0,80.2 436.6,80.2 438.2,80.2 439.9,80.3 441.5,80.3 443.1,80.4 444.8,80.4 446.4,80.5 448.0,80.6 449.7,80.6 451.3,80.7 452.9,80.8 454.6,80.9 456.2,81.0 457.8,81.1 459.5,81.3 461.1,81.4 462.7,81.5 464.4,81.7 466.0,81.8 467.6,82.0 469.2,82.1 470.9,82.3 472.5,82.5 474.1,82.7 475.8,82.9 477.4,83.1 479.0,83.3 480.7,83.5 482.3,83.7 483.9,84.0 485.6,84.2 487.2,84.5 488.8,84.8 490.5,85.0 492.1,85.3 493.7,85.6 495.4,85.9 497.0,86.2 498.6,86.6 500.3,86.9 501.9,87.3 503.5,87.6 505.2,88.0 506.8,88.4 508.4,88.8 510.1,89.2 511.7,89.6 513.3,90.0 514.9,90.5 516.6,90.9 518.2,91.4 519.8,91.9 521.5,92.4 523.1,92.9 524.7,93.5 526.4,94.0 528.0,94.6 529.6,95.2 531.3,95.8 532.9,96.4 534.5,97.1 536.2,97.8 537.8,98.5 539.4,99.2 541.1,100.0 542.7,100.8 544.3,101.6 546.0,102.5 547.6,103.4 549.2,104.3 550.9,105.3 552.5,106.4 554.1,107.6 555.8,108.8 557.4,110.2 559.0,111.9" fill="none" stroke-width="3" class="stroke-b"/>
<line x1="428.2" y1="80.1" x2="428.2" y2="230.0" class="stroke-muted dashed"/>
<circle cx="428.2" cy="80.1" r="6" class="fill-bad"/>
<line x1="428.2" y1="56.9" x2="428.2" y2="80.1" class="stroke-muted dashed"/>
<text x="418.0" y="50.9" text-anchor="end" class="fig-bold">best: p = 0.731, score 0.620</text>
<line x1="246.4" y1="177.2" x2="280.7" y2="177.2" class="stroke-a" stroke-width="3"/>
<text x="290.5" y="181.2" text-anchor="start" class="fig-small">expected reward</text>
<line x1="246.4" y1="198.3" x2="280.7" y2="198.3" class="stroke-d" stroke-width="3"/>
<text x="290.5" y="202.3" text-anchor="start" class="fig-small">leash cost β·KL</text>
<line x1="246.4" y1="219.4" x2="280.7" y2="219.4" class="stroke-b" stroke-width="3"/>
<text x="290.5" y="223.4" text-anchor="start" class="fig-small">score = reward − leash</text>
<text x="315.0" y="272.0" text-anchor="middle" class="fig-small">p: probability the policy gives the helpful answer</text>
<text x="70.0" y="24.0" text-anchor="start" class="fig-small">score</text>
</svg>
Figure 3. A policy over two answers, helpful (reference 0.5) and everything else (reference 0.5), with β = 1. Expected reward (blue) climbs in a straight line, the leash cost (purple) climbs steeply near the ends, and their difference (orange) peaks at p = 0.731, the value the closed form predicts. Chasing reward alone would choose p = 1, where the score is only 0.307.
```

## The best policy

The best policy has a closed form, and a short rearrangement shows why. Write the reward inside the log, $r(y) = \beta \log e^{r(y)/\beta}$, so that

$$
J(\pi) = \beta \sum_y \pi(y) \log \frac{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}{\pi(y)}
$$

The numerator, $\pi_{\text{ref}}(y)\, e^{r(y)/\beta}$, is the reference reweighted by the reward, and it does not add to 1. Divide it by its total $Z = \sum_y \pi_{\text{ref}}(y)\, e^{r(y)/\beta}$ and call the result $\pi^*$:

$$
\pi^*(y) = \frac{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}{Z}
$$

Read it aloud: multiply each reference probability by $e^{r/\beta}$, then divide by the total so the probabilities add to 1. Substituting back, the log of the numerator splits into $\log \pi^*(y) + \log Z$, and the sum becomes $J(\pi) = \beta \log Z - \beta\, \text{KL}(\pi \,\|\, \pi^*)$. The first term does not depend on $\pi$. The second is never negative and is 0 only when $\pi = \pi^*$. So no policy beats $\pi^*$, and its score is exactly $\beta \log Z$.

Check it with $\beta = 1$. The boosts are $e^{1} = 2.718$, $1$, and $1$, so the reweighted numbers are $1.359$, $0.3$, and $0.2$. They total $Z = 1.859$, so $\pi^* = [0.731, 0.161, 0.108]$, and the score is $\ln 1.859 = 0.620$, higher than every row of the table. The helpful answer rises from 0.5 to 0.731, and the other two give up the mass it took. In log space, $\log \pi^* = \log \pi_{\text{ref}} + r/\beta - \log Z$: the reward adds to the log probability, which multiplies the probability. In code this is a softmax of `ref_logprobs + rewards / beta`.

## The dial: what beta does

The reward appears only as $r/\beta$, so $\beta$ sets the scale on which rewards count. A large $\beta$ shrinks every boost toward 1 and $\pi^*$ toward the reference. A small $\beta$ blows the boosts up until the top-reward answer takes everything. Figure 4 shows both ends: at $\beta = 0.1$ the helpful answer gets 1.00, and at $\beta = 100$ the policy has not moved from the reference by more than 0.003.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The best policy for five values of beta, sliding from all on the top reward answer back to the reference">
<line x1="20.0" y1="190.0" x2="120.0" y2="190.0" class="stroke-muted"/>
<rect x="24.0" y="60.0" width="24.0" height="130.0" class="fill-b"/>
<text x="36.0" y="55.0" text-anchor="middle" class="fig-small">1.00</text>
<line x1="21.0" y1="125.0" x2="51.0" y2="125.0" class="stroke-ink dashed"/>
<rect x="54.0" y="190.0" width="24.0" height="0.0" class="fill-b"/>
<text x="66.0" y="185.0" text-anchor="middle" class="fig-small">0.00</text>
<line x1="51.0" y1="151.0" x2="81.0" y2="151.0" class="stroke-ink dashed"/>
<rect x="84.0" y="190.0" width="24.0" height="0.0" class="fill-b"/>
<text x="96.0" y="185.0" text-anchor="middle" class="fig-small">0.00</text>
<line x1="81.0" y1="164.0" x2="111.0" y2="164.0" class="stroke-ink dashed"/>
<text x="67.0" y="212.0" text-anchor="middle" class="fig-bold">β = 0.1</text>
<line x1="132.0" y1="190.0" x2="232.0" y2="190.0" class="stroke-muted"/>
<rect x="136.0" y="75.5" width="24.0" height="114.5" class="fill-b"/>
<text x="148.0" y="70.5" text-anchor="middle" class="fig-small">0.88</text>
<line x1="133.0" y1="125.0" x2="163.0" y2="125.0" class="stroke-ink dashed"/>
<rect x="166.0" y="180.7" width="24.0" height="9.3" class="fill-b"/>
<text x="178.0" y="175.7" text-anchor="middle" class="fig-small">0.07</text>
<line x1="163.0" y1="151.0" x2="193.0" y2="151.0" class="stroke-ink dashed"/>
<rect x="196.0" y="183.8" width="24.0" height="6.2" class="fill-b"/>
<text x="208.0" y="178.8" text-anchor="middle" class="fig-small">0.05</text>
<line x1="193.0" y1="164.0" x2="223.0" y2="164.0" class="stroke-ink dashed"/>
<text x="179.0" y="212.0" text-anchor="middle" class="fig-bold">β = 0.5</text>
<line x1="244.0" y1="190.0" x2="344.0" y2="190.0" class="stroke-muted"/>
<rect x="248.0" y="95.0" width="24.0" height="95.0" class="fill-b"/>
<text x="260.0" y="90.0" text-anchor="middle" class="fig-small">0.73</text>
<line x1="245.0" y1="125.0" x2="275.0" y2="125.0" class="stroke-ink dashed"/>
<rect x="278.0" y="169.0" width="24.0" height="21.0" class="fill-b"/>
<text x="290.0" y="164.0" text-anchor="middle" class="fig-small">0.16</text>
<line x1="275.0" y1="151.0" x2="305.0" y2="151.0" class="stroke-ink dashed"/>
<rect x="308.0" y="176.0" width="24.0" height="14.0" class="fill-b"/>
<text x="320.0" y="159.0" text-anchor="middle" class="fig-small">0.11</text>
<line x1="305.0" y1="164.0" x2="335.0" y2="164.0" class="stroke-ink dashed"/>
<text x="291.0" y="212.0" text-anchor="middle" class="fig-bold">β = 1</text>
<line x1="356.0" y1="190.0" x2="456.0" y2="190.0" class="stroke-muted"/>
<rect x="360.0" y="118.5" width="24.0" height="71.5" class="fill-b"/>
<text x="372.0" y="113.5" text-anchor="middle" class="fig-small">0.55</text>
<line x1="357.0" y1="125.0" x2="387.0" y2="125.0" class="stroke-ink dashed"/>
<rect x="390.0" y="154.9" width="24.0" height="35.1" class="fill-b"/>
<text x="402.0" y="146.0" text-anchor="middle" class="fig-small">0.27</text>
<line x1="387.0" y1="151.0" x2="417.0" y2="151.0" class="stroke-ink dashed"/>
<rect x="420.0" y="166.6" width="24.0" height="23.4" class="fill-b"/>
<text x="432.0" y="159.0" text-anchor="middle" class="fig-small">0.18</text>
<line x1="417.0" y1="164.0" x2="447.0" y2="164.0" class="stroke-ink dashed"/>
<text x="403.0" y="212.0" text-anchor="middle" class="fig-bold">β = 5</text>
<line x1="468.0" y1="190.0" x2="568.0" y2="190.0" class="stroke-muted"/>
<rect x="472.0" y="124.7" width="24.0" height="65.3" class="fill-b"/>
<text x="484.0" y="119.7" text-anchor="middle" class="fig-small">0.50</text>
<line x1="469.0" y1="125.0" x2="499.0" y2="125.0" class="stroke-ink dashed"/>
<rect x="502.0" y="151.2" width="24.0" height="38.8" class="fill-b"/>
<text x="514.0" y="146.2" text-anchor="middle" class="fig-small">0.30</text>
<line x1="499.0" y1="151.0" x2="529.0" y2="151.0" class="stroke-ink dashed"/>
<rect x="532.0" y="164.1" width="24.0" height="25.9" class="fill-b"/>
<text x="544.0" y="159.1" text-anchor="middle" class="fig-small">0.20</text>
<line x1="529.0" y1="164.0" x2="559.0" y2="164.0" class="stroke-ink dashed"/>
<text x="515.0" y="212.0" text-anchor="middle" class="fig-bold">β = 100</text>
<text x="24.0" y="24.0" text-anchor="start" class="fig-small">probability of [helpful, terse, padded] under the best policy</text>
<text x="24.0" y="238.0" text-anchor="start" class="fig-small">dashed ticks mark the reference probabilities 0.5, 0.3, 0.2</text>
</svg>
Figure 4. The best policy for the same rewards at five values of β. A small β gives the helpful answer almost everything. As β grows, each bar slides back to its dashed reference tick, and at β = 100 the policy is the reference to within 0.003.
```

The dial also shows what the leash cannot do. If the reward model scored the padded answer 3 (and helpful 1, terse 0), the same formula at $\beta = 1$ would give the padded answer 0.708, and at $\beta = 5$ it gets 0.286. A leash does not know which reward is right. It limits how fast you can chase one, so $\beta$ has to be chosen against the size of the rewards you trust.

$Z$ sums over every possible answer. With three answers that is 1.859, but a real answer is a sequence of tokens, and with 50,000 tokens and 100 positions there are about $10^{470}$ sequences. The best policy exists, and you cannot write it down by summing. A later chapter on [PPO and DPO](/lessons/064-ppo-and-dpo) reaches it by sampling and then rewrites the problem so that $Z$ cancels.

> [!KEY]
> Without a limit, a policy exploits the reward model's blind spots. The fix is to maximise reward minus $\beta$ times the KL distance from the reference. The winner is the reference reweighted by $e^{r/\beta}$, divided by the total $Z$, and $\beta$ decides how far it may move.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The chance that the chosen answer wins is $\sigma(r_w - r_l)$, computed as `np.exp(-np.logaddexp(0, -z))`. The loss is the mean of `np.logaddexp(0, -z)`, and a tie gives $\log 2$.
> - The chosen gradient is $(\sigma(z) - 1)/n$, and the rejected gradient is its negation.
> - KL is $\sum p \log(p/q)$ on the last axis, with the policy first, and a term with $p = 0$ counts as 0.
> - The leashed score is expected reward minus $\beta$ times KL.
> - The best policy is a softmax of `ref_logprobs + rewards / beta` along the last axis, and its score is $\beta \log Z$.

## Build it step by step

The checks call your functions in this order:

1. `preference_probability(reward_chosen, reward_rejected)` takes two arrays of the same shape and returns the chance that the chosen answer wins, elementwise. The common mistake is `1 / (1 + np.exp(-z))`, which overflows for large negative gaps.
2. `bradley_terry_loss(reward_chosen, reward_rejected)` takes two `(n,)` arrays and returns one float: the mean of $-\log \sigma(z)$. The common mistake is to sum instead of average.
3. `bradley_terry_grad(reward_chosen, reward_rejected)` returns two `(n,)` arrays, the gradients with respect to each argument. Reuse `preference_probability`. The common mistake is to forget the $1/n$, or to return the same sign for both arrays.
4. `kl_divergence(p, q)` takes two distributions with the same shape `(..., n)` and returns the KL on the last axis. Mask terms where `p == 0`, and keep the masked branch from evaluating `log(0)` into a warning that hides a bug. The common mistake is to sum over all axes instead of the last:

```python
def kl_divergence(p, q):
    safe_p = np.where(p > 0, p, 1.0)
    terms = ...  # p * log(safe_p / q), zero where p == 0
    return np.sum(terms, axis=-1)
```

5. `optimal_policy(ref_logprobs, rewards, beta)` takes log reference probabilities and rewards of the same shape and returns probabilities. Form `ref_logprobs + rewards / beta`, subtract the row maximum, exponentiate, and divide by the sum with `axis=-1` and `keepdims=True`. The common mistake is to normalise the whole array instead of each row.
6. `penalised_objective(probs, ref_probs, rewards, beta)` returns expected reward minus $\beta$ times KL, on the last axis. Reuse `kl_divergence`. The common mistake is to put the arguments in the wrong order, which gives the other KL.

## Where this breaks

A reward model learns from a finite set of comparisons. Where people disagree, or where no one compared answers, its scores are guesses. Accuracy, the share of pairs where the chosen reward is above the rejected one, hides this: in the two-pair example it is 50 percent before the step and 100 percent after, but the loss also says how confident the scorer is.

The leash keeps the policy close to the reference only on average, and the reward model can still be wrong inside that region. In a real system $\text{KL}(\pi \,\|\, \pi_{\text{ref}})$ is estimated from the answers the policy writes, not summed over all answers. Picking $\beta$ is a trial: too small and the model exploits the reward model, too large and it learns nothing.

## The exercise

You build the six functions above. The checks cover the hand-worked values, ties and huge gaps without overflow, and the gradient against finite differences. They also verify the KL by hand and its zero-probability case, the closed form on the three-answer example, and the limits of large and tiny $\beta$. Two checks test the derivation: no random distribution beats `optimal_policy`, and the penalised objective of `optimal_policy` equals $\beta \log Z$, while every other policy scores $\beta \log Z$ minus $\beta$ times its KL from the best one.
