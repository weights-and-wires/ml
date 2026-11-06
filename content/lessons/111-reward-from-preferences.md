---
title: Learning a reward from human preferences
slug: 111-reward-from-preferences
part: "Part 7 — Large language models"
summary: Turn "B is better than A" into a training signal, by giving each answer a score and rewarding the scorer for ranking answers as people do.
minutes: 40
gpu: false
objectives:
  - Explain why a person's choice between two answers is easier to collect than a written perfect answer, and why a gap between scores is the natural way to model it.
  - Turn a reward gap into the chance that the chosen answer wins, with a sigmoid, and compute it by hand and with numpy.
  - Compute the Bradley-Terry loss, explain why it is the negative log of that chance, and compute it without overflow.
  - Derive the gradient of the loss with respect to both rewards, and explain why a pair that is ranked wrong gets the bigger push.
prereqs:
  - 110-low-rank-reach-and-training
  - 015-sigmoid-and-probability
  - 056-softmax-cross-entropy-gradient
  - 095-next-token-loss
exercise: 111-reward-from-preferences
---

# Learning a reward from human preferences

A user asks an assistant, "Explain a mortgage to me." Two answers come back. Answer A says, "A mortgage is a loan for a house." Answer B says what the loan is, how each monthly payment splits into interest and repayment, and what happens if you stop paying. A reader picks B in two seconds.

Supervised fine-tuning cannot make that choice. It copies one written answer, token by token, and scores a model by how much probability it gave that text. It knows only what the writer wrote, so it cannot say that B beats A. To use it, a person would have to write the best answer to thousands of prompts, and writing the best answer is slow even for an expert. Picking the better of two answers takes seconds, so people can make millions of those choices.

This chapter answers one question: how do you turn "B is better than A" into a number that a model can train on? The answer has three parts. Give every answer a score. Turn the gap between two scores into the chance that a person picks the higher one. Then reward the scorer for giving a high chance to the choice people made. The later chapters use that scorer to steer a language model.

## What you already know

- A sigmoid squashes any real number into a value between 0 and 1, and the input 0 gives $1/2$ (see the chapter on turning a score into a probability).
- A softmax turns a list of scores into probabilities that add to 1.
- A log probability is the logarithm of a probability. The loss $-\log p$ scores how much probability a model gave to what actually happened.
- A gradient says how much the loss changes when you nudge one number, and gradient descent moves each number against it.
- A mean over $n$ examples puts a factor $1/n$ on each example's gradient.

## The picture: a rating for each answer

Why not ask people to score each answer from 1 to 10? Because raters disagree about what a 7 means, while two answers side by side get a much steadier verdict. So we keep the comparison, and we still want a number per answer, because a model that outputs one number is easy to train.

Chess shows how to join the two. Every player has a rating, and the chance that one player beats another depends only on the gap between their ratings. A big gap means a likely win, equal ratings mean even odds, and nobody's win chance is exactly 0 or 1. Give every answer a rating in the same way. Call it the reward, written $r$. A reward network reads the prompt and an answer and outputs one number, and a higher number means a better answer. Because only gaps matter, the numbers have no meaning on their own, as a rating of 1600 means nothing without someone to compare against.

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

## From a gap to a chance

Four things must hold for the curve in Figure 1. Any gap, positive or negative, must give a number between 0 and 1. A gap of 0 must give even odds. A bigger gap in favour of an answer must give more confidence. Swapping the two answers must swap the chance $p$ for $1 - p$. The sigmoid $\sigma(z) = 1/(1 + e^{-z})$ does all four, so we use it. Other curves would also pass these tests, but the sigmoid has a bonus that the later chapters lean on: it is a softmax over two answers, as the next formula shows. This rule is the Bradley-Terry model.

Take the mortgage prompt, with reward 2 for B and 0 for A. The gap is 2. Then $e^{-2} \approx 0.1353$, so $\sigma(2) = 1/1.1353 \approx 0.8808$. The model says a person picks B about 88 times in 100. With $r_w$ for the chosen reward and $r_l$ for the rejected one, the rule reads:

$$
p(\text{chosen wins}) = \sigma(r_w - r_l) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}}
$$

Read it aloud: the chance that the chosen answer wins is the sigmoid of the chosen reward minus the rejected reward. The second form shows that it is a softmax over two answers. Check it on the example: $e^{2}/(e^{2} + e^{0}) = 7.389/8.389 = 0.8808$. In numpy the sigmoid of the gap is one line. It must also survive large gaps, and `np.exp(-np.logaddexp(0, -z))` does, because `np.logaddexp(0, -z)` computes $\log(1 + e^{-z})$ without ever forming the huge number $e^{-z}$.

In a real system the rewards come from a network, and its weights get the gradient by backpropagation. In the exercise the rewards are free numbers that you nudge directly. The signal you compute is what a real reward network would receive.

## Score the scorer: the negative log

A person chose B, and the model said B wins with chance 0.88. That is a good model. If it had said 0.27, it would be a bad one. We need one number that is small for the first case and large for the second. The number that does it is $-\log p$, the loss from the chapter on how a language model is scored, with $p$ the chance the model gave to what the person chose. Training to make the chosen outcome likely is called maximum likelihood.

The table works it out for five gaps:

| Gap $z$ | Chance $\sigma(z)$ | Loss $-\log \sigma(z)$ |
|---|---|---|
| 3 | 0.9526 | 0.0486 |
| 1 | 0.7311 | 0.3133 |
| 0 | 0.5000 | 0.6931 |
| −1 | 0.2689 | 1.3133 |
| −3 | 0.0474 | 3.0486 |

```figure
<svg viewBox="0 0 600 284" role="img" aria-label="The loss minus log sigmoid of the gap and the size of its push, with gaps 1 and minus 1 marked">
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="184.8" x2="560.0" y2="184.8" class="stroke-grid"/>
<text x="62.0" y="188.8" text-anchor="end" class="fig-small">1</text>
<line x1="70.0" y1="139.5" x2="560.0" y2="139.5" class="stroke-grid"/>
<text x="62.0" y="143.5" text-anchor="end" class="fig-small">2</text>
<line x1="70.0" y1="94.3" x2="560.0" y2="94.3" class="stroke-grid"/>
<text x="62.0" y="98.3" text-anchor="end" class="fig-small">3</text>
<line x1="70.0" y1="49.0" x2="560.0" y2="49.0" class="stroke-grid"/>
<text x="62.0" y="53.0" text-anchor="end" class="fig-small">4</text>
<text x="70.0" y="248.0" text-anchor="middle" class="fig-small">−4</text>
<text x="192.5" y="248.0" text-anchor="middle" class="fig-small">−2</text>
<text x="315.0" y="248.0" text-anchor="middle" class="fig-small">0</text>
<text x="437.5" y="248.0" text-anchor="middle" class="fig-small">2</text>
<text x="560.0" y="248.0" text-anchor="middle" class="fig-small">4</text>
<line x1="70.0" y1="230.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="40.0" x2="70.0" y2="230.0" class="stroke-muted"/>
<polyline points="70.0,48.2 73.1,50.4 76.1,52.7 79.2,54.9 82.3,57.1 85.3,59.3 88.4,61.5 91.4,63.7 94.5,65.9 97.6,68.1 100.6,70.3 103.7,72.5 106.8,74.7 109.8,76.9 112.9,79.1 115.9,81.3 119.0,83.4 122.1,85.6 125.1,87.8 128.2,89.9 131.2,92.1 134.3,94.2 137.4,96.4 140.4,98.5 143.5,100.7 146.6,102.8 149.6,104.9 152.7,107.0 155.8,109.1 158.8,111.2 161.9,113.3 164.9,115.4 168.0,117.5 171.1,119.6 174.1,121.6 177.2,123.7 180.2,125.7 183.3,127.8 186.4,129.8 189.4,131.8 192.5,133.8 195.6,135.8 198.6,137.7 201.7,139.7 204.8,141.7 207.8,143.6 210.9,145.5 213.9,147.4 217.0,149.3 220.1,151.2 223.1,153.0 226.2,154.9 229.2,156.7 232.3,158.5 235.4,160.3 238.4,162.1 241.5,163.8 244.6,165.5 247.6,167.2 250.7,168.9 253.8,170.6 256.8,172.2 259.9,173.9 262.9,175.4 266.0,177.0 269.1,178.6 272.1,180.1 275.2,181.6 278.2,183.1 281.3,184.5 284.4,185.9 287.4,187.3 290.5,188.7 293.6,190.0 296.6,191.4 299.7,192.6 302.8,193.9 305.8,195.1 308.9,196.3 311.9,197.5 315.0,198.6 318.1,199.8 321.1,200.8 324.2,201.9 327.2,202.9 330.3,203.9 333.4,204.9 336.4,205.9 339.5,206.8 342.6,207.7 345.6,208.6 348.7,209.4 351.8,210.2 354.8,211.0 357.9,211.8 360.9,212.5 364.0,213.2 367.1,213.9 370.1,214.6 373.2,215.2 376.2,215.8 379.3,216.4 382.4,217.0 385.4,217.6 388.5,218.1 391.6,218.6 394.6,219.1 397.7,219.6 400.8,220.0 403.8,220.5 406.9,220.9 409.9,221.3 413.0,221.7 416.1,222.1 419.1,222.4 422.2,222.8 425.3,223.1 428.3,223.4 431.4,223.7 434.4,224.0 437.5,224.3 440.6,224.5 443.6,224.8 446.7,225.0 449.8,225.2 452.8,225.5 455.9,225.7 458.9,225.9 462.0,226.1 465.1,226.3 468.1,226.4 471.2,226.6 474.3,226.8 477.3,226.9 480.4,227.1 483.4,227.2 486.5,227.3 489.6,227.5 492.6,227.6 495.7,227.7 498.8,227.8 501.8,227.9 504.9,228.0 507.9,228.1 511.0,228.2 514.1,228.3 517.1,228.4 520.2,228.4 523.2,228.5 526.3,228.6 529.4,228.7 532.4,228.7 535.5,228.8 538.6,228.8 541.6,228.9 544.7,228.9 547.8,229.0 550.8,229.0 553.9,229.1 556.9,229.1 560.0,229.2" fill="none" stroke-width="2.5" class="stroke-b"/>
<polyline points="70.0,185.6 73.1,185.6 76.1,185.7 79.2,185.7 82.3,185.8 85.3,185.8 88.4,185.9 91.4,185.9 94.5,186.0 97.6,186.0 100.6,186.1 103.7,186.2 106.8,186.2 109.8,186.3 112.9,186.4 115.9,186.5 119.0,186.5 122.1,186.6 125.1,186.7 128.2,186.8 131.2,186.9 134.3,187.0 137.4,187.1 140.4,187.2 143.5,187.4 146.6,187.5 149.6,187.6 152.7,187.7 155.8,187.9 158.8,188.0 161.9,188.2 164.9,188.4 168.0,188.5 171.1,188.7 174.1,188.9 177.2,189.1 180.2,189.3 183.3,189.5 186.4,189.7 189.4,189.9 192.5,190.2 195.6,190.4 198.6,190.6 201.7,190.9 204.8,191.2 207.8,191.5 210.9,191.7 213.9,192.1 217.0,192.4 220.1,192.7 223.1,193.0 226.2,193.4 229.2,193.7 232.3,194.1 235.4,194.5 238.4,194.8 241.5,195.2 244.6,195.6 247.6,196.1 250.7,196.5 253.8,196.9 256.8,197.4 259.9,197.8 262.9,198.3 266.0,198.8 269.1,199.3 272.1,199.8 275.2,200.3 278.2,200.8 281.3,201.3 284.4,201.8 287.4,202.4 290.5,202.9 293.6,203.5 296.6,204.0 299.7,204.6 302.8,205.1 305.8,205.7 308.9,206.3 311.9,206.8 315.0,207.4 318.1,207.9 321.1,208.5 324.2,209.1 327.2,209.6 330.3,210.2 333.4,210.7 336.4,211.3 339.5,211.8 342.6,212.4 345.6,212.9 348.7,213.4 351.8,214.0 354.8,214.5 357.9,215.0 360.9,215.5 364.0,216.0 367.1,216.5 370.1,216.9 373.2,217.4 376.2,217.8 379.3,218.3 382.4,218.7 385.4,219.1 388.5,219.5 391.6,219.9 394.6,220.3 397.7,220.7 400.8,221.1 403.8,221.4 406.9,221.7 409.9,222.1 413.0,222.4 416.1,222.7 419.1,223.0 422.2,223.3 425.3,223.6 428.3,223.9 431.4,224.1 434.4,224.4 437.5,224.6 440.6,224.8 443.6,225.1 446.7,225.3 449.8,225.5 452.8,225.7 455.9,225.9 458.9,226.1 462.0,226.2 465.1,226.4 468.1,226.6 471.2,226.7 474.3,226.9 477.3,227.0 480.4,227.2 483.4,227.3 486.5,227.4 489.6,227.5 492.6,227.6 495.7,227.8 498.8,227.9 501.8,228.0 504.9,228.0 507.9,228.1 511.0,228.2 514.1,228.3 517.1,228.4 520.2,228.5 523.2,228.5 526.3,228.6 529.4,228.7 532.4,228.7 535.5,228.8 538.6,228.9 541.6,228.9 544.7,229.0 547.8,229.0 550.8,229.1 553.9,229.1 556.9,229.1 560.0,229.2" fill="none" stroke-width="1.8" class="stroke-d"/>
<circle cx="376.2" cy="215.8" r="6" class="fill-a"/>
<text x="388.2" y="199.8" text-anchor="start" class="fig-bold">gap +1 → loss 0.3133</text>
<circle cx="253.8" cy="170.6" r="6" class="fill-a"/>
<text x="267.8" y="156.6" text-anchor="start" class="fig-bold">gap −1 → loss 1.3133</text>
<text x="278.2" y="107.9" text-anchor="start" class="fig-bold">loss −log σ(z)</text>
<text x="76.1" y="139.5" text-anchor="start" class="fig-small">slope about −1:</text>
<text x="76.1" y="153.1" text-anchor="start" class="fig-small">confidently wrong</text>
<text x="431.4" y="162.1" text-anchor="start" class="fig-small">flat: already right</text>
<text x="76.1" y="205.1" text-anchor="start" class="fig-small">push size 1 − σ(z)</text>
<text x="315.0" y="272.0" text-anchor="middle" class="fig-small">reward gap z: chosen reward minus rejected reward</text>
<text x="70.0" y="24.0" text-anchor="start" class="fig-small">loss for one pair</text>
</svg>
Figure 2. The loss for one pair (orange) and the size of its push, 1 − σ(z) (purple). On the right the loss curve is flat, so a pair that is already right costs almost nothing and gets almost no push. On the left it runs close to a straight line of slope −1, so the cost grows about as fast as the gap is wrong, and the push stays near 1.
```

Figure 2 plots the loss. On the right it is almost flat, so a pair that is already ranked right costs almost nothing. On the left it is close to a straight line of slope −1, so a pair ranked confidently wrong costs about as much as the gap is wrong. A tie, gap 0, costs $\log 2 \approx 0.6931$, the price of a coin flip.

Why not require only that the chosen reward beats the rejected one, with a hard cutoff? A hard cutoff has no slope, so it gives no signal about how to fix a wrong pair. The smooth curve gives a push everywhere, and the push fades only when the pair is already right.

With $n$ pairs, average the loss. Take two prompts. Chosen rewards are $[2, 0]$ and rejected rewards are $[1, 1]$, so the gaps are $[1, -1]$. The losses are $0.3133$ and $1.3133$, and their mean is $0.8133$:

$$
\text{loss} = \frac{1}{n} \sum_{i=1}^{n} -\log \sigma(z_i), \qquad z_i = r_{w,i} - r_{l,i}
$$

Read it aloud: for each pair, take the gap, take the negative log of the sigmoid, and average. In numpy: `np.mean(np.logaddexp(0, -z))`.

> [!WARNING]
> The tempting version, `np.log(1 + np.exp(-z))`, overflows when $z$ is very negative: `np.exp(800)` is `inf`. Use `np.logaddexp(0, -z)`, which returns the right answer, 800, for $z = -800$.

## Push the chosen reward up and the rejected one down

To train, ask how the loss changes when you nudge each reward. The derivative of $-\log \sigma(z)$ with respect to $z$ is $\sigma(z) - 1$, because the sigmoid's own slope is $\sigma(z)(1 - \sigma(z))$ and dividing that by $\sigma(z)$ leaves $1 - \sigma(z)$, with a minus sign from the $-\log$. This is minus the chance the model gave to the wrong outcome. The gap $z$ is $r_w - r_l$, so the chosen reward gets $\sigma(z) - 1$, the rejected reward gets the opposite sign, and the mean puts $1/n$ on both:

$$
\frac{\partial\, \text{loss}}{\partial r_{w,i}} = \frac{\sigma(z_i) - 1}{n}, \qquad
\frac{\partial\, \text{loss}}{\partial r_{l,i}} = -\frac{\sigma(z_i) - 1}{n}
$$

Read it aloud: the push on a reward is the chance the model gave to the wrong outcome, divided by $n$. On the two pairs, $\sigma(1) - 1 = -0.2689$ and $\sigma(-1) - 1 = -0.7311$. After dividing by $n = 2$ the chosen gradients are $[-0.1345, -0.3655]$ and the rejected gradients are $[+0.1345, +0.3655]$. A negative gradient on the chosen reward means gradient descent raises it. A positive gradient on the rejected reward means descent lowers it. In numpy: `(preference_probability(rc, rr) - 1) / n`, and its negation.

```figure
<svg viewBox="0 0 600 348" role="img" aria-label="One gradient step on two pairs of rewards, with arrows for how far each reward moves">
<defs><marker id="c125-arrow-c" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-c"/></marker><marker id="c125-arrow-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="135.0" y1="114.3" x2="135.0" y2="88.3" class="stroke-c" marker-end="url(#c125-arrow-c)" stroke-width="2.5"/>
<circle cx="135.0" cy="94.3" r="4" class="fill-sunk stroke-c"/>
<text x="125.0" y="118.3" text-anchor="end" class="fig-small">2</text>
<text x="125.0" y="98.3" text-anchor="end" class="fig-bold fig-small">2.27</text>
<circle cx="245.0" cy="188.6" r="6" class="fill-b "/>
<line x1="245.0" y1="188.6" x2="245.0" y2="214.5" class="stroke-b" marker-end="url(#c125-arrow-b)" stroke-width="2.5"/>
<circle cx="245.0" cy="208.5" r="4" class="fill-sunk stroke-b"/>
<text x="255.0" y="192.6" text-anchor="start" class="fig-small">1</text>
<text x="255.0" y="212.5" text-anchor="start" class="fig-bold fig-small">0.73</text>
<text x="135.0" y="300.0" text-anchor="middle" class="fig-small">chosen</text>
<text x="245.0" y="300.0" text-anchor="middle" class="fig-small">rejected</text>
<text x="430.0" y="334.0" text-anchor="middle" class="fig-bold">pair 2: gap −1</text>
<circle cx="375.0" cy="262.9" r="6" class="fill-c "/>
<line x1="375.0" y1="262.9" x2="375.0" y2="202.5" class="stroke-c" marker-end="url(#c125-arrow-c)" stroke-width="2.5"/>
<circle cx="375.0" cy="208.5" r="4" class="fill-sunk stroke-c"/>
<text x="365.0" y="266.9" text-anchor="end" class="fig-small">0</text>
<text x="365.0" y="212.5" text-anchor="end" class="fig-bold fig-small">0.73</text>
<circle cx="485.0" cy="188.6" r="6" class="fill-b "/>
<line x1="485.0" y1="188.6" x2="485.0" y2="248.9" class="stroke-b" marker-end="url(#c125-arrow-b)" stroke-width="2.5"/>
<circle cx="485.0" cy="242.9" r="4" class="fill-sunk stroke-b"/>
<text x="495.0" y="192.6" text-anchor="start" class="fig-small">1</text>
<text x="495.0" y="246.9" text-anchor="start" class="fig-bold fig-small">0.27</text>
<text x="375.0" y="300.0" text-anchor="middle" class="fig-small">chosen</text>
<text x="485.0" y="300.0" text-anchor="middle" class="fig-small">rejected</text>
</svg>
Figure 3. One gradient step with learning rate 2. Filled dots are the rewards before the step, ringed dots after it, and the numbers beside them are the values. Pair 2 starts out ranked wrong, so its rewards move 0.73 each, nearly three times as far as the rewards of pair 1, which move 0.27 each. After one step both pairs are ranked right.
```

The pair that is ranked wrong gets almost three times the push, 0.3655 against 0.1345. Figure 3 shows one step with learning rate 2, which is large on purpose so the arrows are readable. The loss falls from 0.8133 to 0.3416, and pair 2 is no longer ranked wrong.

Add the same constant to every reward and nothing changes, because $z$ is a difference. That is why the chapter said that only gaps matter.

> [!KEY]
> A person's "B beats A" becomes training data in three moves. A reward network scores each answer. The sigmoid of the gap gives the chance that the chosen answer wins. The negative log of that chance is the loss, and its gradient pushes the chosen reward up and the rejected reward down, hardest when the pair is wrong. Only gaps matter.

> [!RECAP]
> Keep these facts in front of you while you code:
>
> - The chance that the chosen answer wins is $\sigma(r_w - r_l)$, and it is $1/2$ on a tie.
> - Compute it with `np.exp(-np.logaddexp(0, -z))` so that large gaps do not overflow.
> - The loss is the mean of $\log(1 + e^{-z})$, written `np.logaddexp(0, -z)`, and a tie gives $\log 2$.
> - The chosen gradient is $(\sigma(z) - 1)/n$, and the rejected gradient is its negation.

## Build it step by step

The checks call your functions in this order:

1. `preference_probability(reward_chosen, reward_rejected)` takes two arrays of the same shape and returns the chance that the chosen answer wins, elementwise. Compute the gap and pass it through the stable sigmoid. The common mistake is `1 / (1 + np.exp(-z))`, which overflows for large negative gaps:

```python
def preference_probability(reward_chosen, reward_rejected):
    z = reward_chosen - reward_rejected
    return ...  # stable sigmoid of z
```

2. `bradley_terry_loss(reward_chosen, reward_rejected)` takes two `(n,)` arrays and returns one float: the mean of $-\log \sigma(z)$. Use `np.logaddexp(0, -z)` on each gap. The common mistake is to sum instead of average:

```python
def bradley_terry_loss(reward_chosen, reward_rejected):
    z = reward_chosen - reward_rejected
    return ...  # mean of the per-pair losses
```

3. `bradley_terry_grad(reward_chosen, reward_rejected)` returns two `(n,)` arrays, the gradients with respect to each argument. Reuse `preference_probability`. The common mistake is to forget the $1/n$, or to return the same sign for both arrays:

```python
def bradley_terry_grad(reward_chosen, reward_rejected):
    p = preference_probability(reward_chosen, reward_rejected)
    return ..., ...  # (p - 1) / n and its negation
```

## Where this breaks

A reward model learns from a finite set of comparisons. Where people disagree, or where no one compared answers, its scores are guesses. Accuracy, the share of pairs where the chosen reward is above the rejected one, hides this. In the two-pair example it is 50 percent before the step and 100 percent after, but the loss also says how confident the scorer is. The next chapter shows what happens when a language model is trained to chase such a scorer without limit.

## The exercise

You build the three functions above. `preference_probability` turns a reward gap into the chance that the chosen answer wins. `bradley_terry_loss` is the average negative log of that chance. `bradley_terry_grad` is the push on each reward.

The checks cover the hand-worked values, ties and huge gaps without overflow, the fact that adding one constant to all rewards changes nothing, the gradient against finite differences, and a few gradient steps that lower the loss and fix a pair that starts out wrong.
