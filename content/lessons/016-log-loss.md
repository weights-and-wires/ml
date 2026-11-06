---
title: How wrong is a probability? The log loss
slug: 016-log-loss
part: "Part 1 — Foundations"
summary: "Score a probability by how surprised it was by the truth, then compute that log loss without overflow as softplus(z) − y z."
minutes: 35
gpu: false
objectives:
  - Explain why a confident wrong answer should cost far more than a mild one, and why squared error does not do that.
  - Compute the loss of a row as minus the log of the probability given to what happened, by hand.
  - Rewrite the loss as softplus(z) − y z and evaluate it without overflow at a score of 800.
  - Add the ridge fee to the loss and sanity-check the total at zero weights.
prereqs:
  - 015-sigmoid-and-probability
exercise: 016-log-loss
---

# How wrong is a probability? The log loss

Two classifiers examine a patient who has the disease. Both say "no", but they differ in how they say it. The first gives the disease a probability of 0.4. The second gives it 0.001. Both are wrong, yet the second was far more sure, so it should lose more.

A fitting method can only improve a model if it has a number that says how bad each prediction was. Without that number, a mildly wrong model and a confidently wrong one look alike, and the fit has no reason to fix the confident mistake first. The obvious score, squared error, hardly separates the two: it gives 0.36 and 0.998.

This chapter builds a score that does separate them, the log loss. You will see it as a measure of surprise, check it by hand on two emails, and then rewrite it so it stays finite when the model is very sure.

## What you already know

- The sigmoid turns a score $z$ into a probability (see From a score to a probability).
- A loss is one number per example, and you add the numbers up (see Scoring a guess: loss and empirical risk).
- The probability that independent events all happen is the product of their probabilities.
- The logarithm undoes the exponential, and $\ln p$ is negative for $p$ between 0 and 1.
- The slope of a loss says which way is downhill, and fitting follows that slope (see Walking downhill: slope, gradient, and one step).

## Measure surprise

Ask the question in the natural way: how surprised was the model by what really happened? If the model gave the true outcome probability 1, it was not surprised at all. If it gave the true outcome probability close to 0, it was very surprised. A score for that is $-\ln$ of the probability the model gave to the truth.

| Probability given to the truth | Surprise, $-\ln p$ |
|---|---|
| 1 | 0 |
| 0.5 | 0.693 |
| 0.4 | 0.916 |
| 0.1 | 2.303 |
| 0.001 | 6.908 |

The curve is gentle when the model is right and climbs without limit as the probability goes to 0. For the patient, the truth is "yes". The first classifier gave yes 0.4, so its loss is 0.916. The second gave 0.001, so its loss is 6.908, which is 7.5 times larger.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Loss against the probability given to a yes: surprise minus log p climbs steeply near zero, while squared error stays below 1">
<line x1="70.0" y1="270.0" x2="560.0" y2="270.0" class="stroke-grid" />
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="206.0" x2="560.0" y2="206.0" class="stroke-grid" />
<text x="62.0" y="210.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="142.0" x2="560.0" y2="142.0" class="stroke-grid" />
<text x="62.0" y="146.0" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="78.0" x2="560.0" y2="78.0" class="stroke-grid" />
<text x="62.0" y="82.0" class="fig-small" text-anchor="end">6</text>
<line x1="70.0" y1="270.0" x2="560.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted" />
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="168.0" y1="270.0" x2="168.0" y2="275.0" class="stroke-muted" />
<text x="168.0" y="290.0" class="fig-small" text-anchor="middle">0.2</text>
<line x1="266.0" y1="270.0" x2="266.0" y2="275.0" class="stroke-muted" />
<text x="266.0" y="290.0" class="fig-small" text-anchor="middle">0.4</text>
<line x1="364.0" y1="270.0" x2="364.0" y2="275.0" class="stroke-muted" />
<text x="364.0" y="290.0" class="fig-small" text-anchor="middle">0.6</text>
<line x1="462.0" y1="270.0" x2="462.0" y2="275.0" class="stroke-muted" />
<text x="462.0" y="290.0" class="fig-small" text-anchor="middle">0.8</text>
<line x1="560.0" y1="270.0" x2="560.0" y2="275.0" class="stroke-muted" />
<text x="560.0" y="290.0" class="fig-small" text-anchor="middle">1</text>
<polyline points="70.5,49.0 71.0,71.1 72.6,102.5 74.2,118.1 75.9,128.5 77.5,136.3 79.1,142.6 80.8,147.9 82.4,152.4 84.1,156.3 85.7,159.9 87.3,163.0 89.0,165.9 90.6,168.6 92.2,171.0 93.9,173.3 95.5,175.4 97.1,177.4 98.8,179.3 100.4,181.0 102.0,182.7 103.7,184.3 105.3,185.8 106.9,187.3 108.6,188.6 110.2,190.0 111.8,191.3 113.5,192.5 115.1,193.7 116.7,194.8 118.4,195.9 120.0,197.0 121.6,198.0 123.3,199.0 124.9,200.0 126.5,200.9 128.2,201.8 129.8,202.7 131.4,203.6 133.1,204.4 134.7,205.2 136.3,206.0 138.0,206.8 139.6,207.5 141.2,208.3 142.9,209.0 144.5,209.7 146.1,210.4 147.8,211.1 149.4,211.8 151.0,212.4 152.7,213.1 154.3,213.7 155.9,214.3 157.6,214.9 159.2,215.5 160.8,216.1 162.5,216.6 164.1,217.2 165.7,217.8 167.4,218.3 169.0,218.8 170.6,219.4 172.3,219.9 173.9,220.4 175.5,220.9 177.2,221.4 178.8,221.8 180.4,222.3 182.1,222.8 183.7,223.3 185.4,223.7 187.0,224.2 188.6,224.6 190.3,225.0 191.9,225.5 193.5,225.9 195.2,226.3 196.8,226.7 198.4,227.1 200.1,227.6 201.7,228.0 203.3,228.3 205.0,228.7 206.6,229.1 208.2,229.5 209.9,229.9 211.5,230.3 213.1,230.6 214.8,231.0 216.4,231.3 218.0,231.7 219.7,232.0 221.3,232.4 222.9,232.7 224.6,233.1 226.2,233.4 227.8,233.7 229.5,234.1 231.1,234.4 232.7,234.7 234.4,235.0 236.0,235.4 237.6,235.7 239.3,236.0 240.9,236.3 242.5,236.6 244.2,236.9 245.8,237.2 247.4,237.5 249.1,237.8 250.7,238.1 252.3,238.4 254.0,238.7 255.6,238.9 257.2,239.2 258.9,239.5 260.5,239.8 262.1,240.0 263.8,240.3 265.4,240.6 267.0,240.8 268.7,241.1 270.3,241.4 271.9,241.6 273.6,241.9 275.2,242.1 276.8,242.4 278.5,242.7 280.1,242.9 281.8,243.2 283.4,243.4 285.0,243.6 286.7,243.9 288.3,244.1 289.9,244.4 291.6,244.6 293.2,244.8 294.8,245.1 296.5,245.3 298.1,245.5 299.7,245.8 301.4,246.0 303.0,246.2 304.6,246.4 306.3,246.7 307.9,246.9 309.5,247.1 311.2,247.3 312.8,247.5 314.4,247.7 316.1,248.0 317.7,248.2 319.3,248.4 321.0,248.6 322.6,248.8 324.2,249.0 325.9,249.2 327.5,249.4 329.1,249.6 330.8,249.8 332.4,250.0 334.0,250.2 335.7,250.4 337.3,250.6 338.9,250.8 340.6,251.0 342.2,251.2 343.8,251.4 345.5,251.6 347.1,251.8 348.7,251.9 350.4,252.1 352.0,252.3 353.6,252.5 355.3,252.7 356.9,252.9 358.5,253.1 360.2,253.2 361.8,253.4 363.4,253.6 365.1,253.8 366.7,253.9 368.3,254.1 370.0,254.3 371.6,254.5 373.2,254.6 374.9,254.8 376.5,255.0 378.1,255.2 379.8,255.3 381.4,255.5 383.1,255.7 384.7,255.8 386.3,256.0 388.0,256.2 389.6,256.3 391.2,256.5 392.9,256.6 394.5,256.8 396.1,257.0 397.8,257.1 399.4,257.3 401.0,257.4 402.7,257.6 404.3,257.8 405.9,257.9 407.6,258.1 409.2,258.2 410.8,258.4 412.5,258.5 414.1,258.7 415.7,258.8 417.4,259.0 419.0,259.1 420.6,259.3 422.3,259.4 423.9,259.6 425.5,259.7 427.2,259.9 428.8,260.0 430.4,260.2 432.1,260.3 433.7,260.5 435.3,260.6 437.0,260.7 438.6,260.9 440.2,261.0 441.9,261.2 443.5,261.3 445.1,261.5 446.8,261.6 448.4,261.7 450.0,261.9 451.7,262.0 453.3,262.1 454.9,262.3 456.6,262.4 458.2,262.5 459.8,262.7 461.5,262.8 463.1,262.9 464.7,263.1 466.4,263.2 468.0,263.3 469.6,263.5 471.3,263.6 472.9,263.7 474.5,263.9 476.2,264.0 477.8,264.1 479.4,264.3 481.1,264.4 482.7,264.5 484.4,264.6 486.0,264.8 487.6,264.9 489.3,265.0 490.9,265.1 492.5,265.3 494.2,265.4 495.8,265.5 497.4,265.6 499.1,265.7 500.7,265.9 502.3,266.0 504.0,266.1 505.6,266.2 507.2,266.4 508.9,266.5 510.5,266.6 512.1,266.7 513.8,266.8 515.4,266.9 517.0,267.1 518.7,267.2 520.3,267.3 521.9,267.4 523.6,267.5 525.2,267.6 526.8,267.8 528.5,267.9 530.1,268.0 531.7,268.1 533.4,268.2 535.0,268.3 536.6,268.4 538.3,268.5 539.9,268.7 541.5,268.8 543.2,268.9 544.8,269.0 546.4,269.1 548.1,269.2 549.7,269.3 551.3,269.4 553.0,269.5 554.6,269.6 556.2,269.8 557.9,269.9 559.5,270.0" class="stroke-b" fill="none"/>
<polyline points="70.0,238.0 74.9,238.6 79.9,239.3 84.8,239.9 89.8,240.5 94.7,241.2 99.7,241.8 104.6,242.4 109.6,243.0 114.5,243.6 119.5,244.1 124.4,244.7 129.4,245.3 134.3,245.9 139.3,246.4 144.2,247.0 149.2,247.5 154.1,248.0 159.1,248.6 164.0,249.1 169.0,249.6 173.9,250.1 178.9,250.6 183.8,251.1 188.8,251.6 193.7,252.1 198.7,252.6 203.6,253.1 208.6,253.5 213.5,254.0 218.5,254.5 223.4,254.9 228.4,255.3 233.3,255.8 238.3,256.2 243.2,256.6 248.2,257.0 253.1,257.4 258.1,257.9 263.0,258.2 268.0,258.6 272.9,259.0 277.9,259.4 282.8,259.8 287.8,260.1 292.7,260.5 297.7,260.8 302.6,261.2 307.6,261.5 312.5,261.8 317.5,262.2 322.4,262.5 327.4,262.8 332.3,263.1 337.3,263.4 342.2,263.7 347.2,264.0 352.1,264.2 357.1,264.5 362.0,264.8 367.0,265.0 371.9,265.3 376.9,265.5 381.8,265.8 386.8,266.0 391.7,266.2 396.7,266.4 401.6,266.7 406.6,266.9 411.5,267.1 416.5,267.3 421.4,267.4 426.4,267.6 431.3,267.8 436.3,268.0 441.2,268.1 446.2,268.3 451.1,268.4 456.1,268.6 461.0,268.7 466.0,268.8 470.9,268.9 475.9,269.1 480.8,269.2 485.8,269.3 490.7,269.4 495.7,269.4 500.6,269.5 505.6,269.6 510.5,269.7 515.5,269.7 520.4,269.8 525.4,269.8 530.3,269.9 535.3,269.9 540.2,269.9 545.2,270.0 550.1,270.0 555.1,270.0 560.0,270.0" class="stroke-c dashed" fill="none"/>
<circle cx="266.0" cy="240.7" r="5" class="fill-b"/>
<circle cx="266.0" cy="258.5" r="5" class="fill-c"/>
<circle cx="70.5" cy="49.0" r="5" class="fill-b"/>
<circle cx="70.5" cy="238.1" r="5" class="fill-c"/>
<text x="276.0" y="226.7" class="fig-small" text-anchor="start">p = 0.4: loss 0.92 (squared error 0.36)</text>
<text x="82.5" y="52.9" class="fig-small" text-anchor="start">p = 0.001: loss 6.91 (squared error 1.00)</text>
<line x1="324.8" y1="87.6" x2="364.0" y2="87.6" class="stroke-b" />
<text x="373.8" y="91.6" class="fig-small" text-anchor="start">surprise: −ln p</text>
<line x1="324.8" y1="113.2" x2="364.0" y2="113.2" class="stroke-c dashed" />
<text x="373.8" y="117.2" class="fig-small" text-anchor="start">squared error: (1 − p)²</text>
<text x="315.0" y="312.0" text-anchor="middle">probability the model gave to yes (the true label is yes)</text>
<text x="62.0" y="22.0" text-anchor="end">loss</text>
</svg>
Figure 1. The solid curve is the surprise −ln p, the dashed curve is squared error (1 − p)². Notice that the two classifiers differ by a factor of 2.8 under squared error and by 7.5 under the log loss, and that the log curve keeps climbing where squared error has stopped at 1.
```

## Reject squared error

You already know squared error from linear regression, so why not use $(y - p)^2$? The two numbers above already hint at the problem: 0.36 against 0.998 hardly distinguishes a mild mistake from a confident one.

The deeper problem is the slope. A fit changes the score $z$ by following the loss downhill, so what matters is how much the loss moves when you nudge $z$. For squared error that slope is $-2(y - p)\,p(1-p)$. At $p = 0.001$ with $y = 1$ it is $-0.002$, because the factor $p(1-p)$ is almost 0. For the log loss the slope is $p - y = -0.999$, as the next chapter derives. At the most wrong point the log loss pushes about 500 times harder, and squared error nearly stops. A model that is sure and wrong would barely be corrected.

## Score two emails by hand

Use the two emails from the last chapter. Email A is spam, so $y = 1$, and the model gave it $p = 0.622$. Email B is not spam, so $y = 0$, and the model gave spam $p = 0.182$, which means it gave the truth, "not spam", $1 - 0.182 = 0.818$.

The surprise for A is $-\ln 0.622 = 0.474$. For B it is $-\ln 0.818 = 0.201$. The total is $0.474 + 0.201 = 0.6755$.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Two emails: the probability each model gave to the truth, the surprise that follows, and the total of 0.6755">
<text x="20.0" y="28.0" class="fig-bold" text-anchor="start">What happened</text>
<text x="170.0" y="28.0" class="fig-bold" text-anchor="start">Probability given to it</text>
<text x="430.0" y="28.0" class="fig-bold" text-anchor="start">Surprise −ln(that)</text>
<text x="20.0" y="74.0" class="fig-bold" text-anchor="start">Email A</text>
<text x="20.0" y="92.0" class="fig-small" text-anchor="start">spam, p = 0.622</text>
<rect x="170.0" y="60.0" width="190.0" height="28.0" class="fill-sunk" />
<rect x="170.0" y="60.0" width="118.3" height="28.0" class="fill-soft-a" />
<rect x="170.0" y="60.0" width="190.0" height="28.0" class="stroke-muted" />
<text x="282.3" y="79.0" class="fig-small" text-anchor="end">0.622</text>
<rect x="430.0" y="60.0" width="80.6" height="28.0" class="fill-b" />
<text x="518.6" y="79.0" class="fig-small" text-anchor="start">0.474</text>
<text x="20.0" y="152.0" class="fig-bold" text-anchor="start">Email B</text>
<text x="20.0" y="170.0" class="fig-small" text-anchor="start">not spam, p = 0.182</text>
<rect x="170.0" y="138.0" width="190.0" height="28.0" class="fill-sunk" />
<rect x="170.0" y="138.0" width="155.3" height="28.0" class="fill-soft-a" />
<rect x="170.0" y="138.0" width="190.0" height="28.0" class="stroke-muted" />
<text x="319.3" y="157.0" class="fig-small" text-anchor="end">0.818</text>
<rect x="430.0" y="138.0" width="34.2" height="28.0" class="fill-d" />
<text x="472.2" y="157.0" class="fig-small" text-anchor="start">0.201</text>
<line x1="20.0" y1="204.0" x2="580.0" y2="204.0" class="stroke-grid" />
<text x="20.0" y="235.0" class="fig-bold" text-anchor="start">Total loss</text>
<rect x="430.0" y="216.0" width="80.6" height="28.0" class="fill-b" />
<rect x="510.6" y="216.0" width="34.2" height="28.0" class="fill-d" />
<text x="552.8" y="235.0" class="fig-small fig-bold" text-anchor="start">0.6755</text>
<text x="170.0" y="235.0" class="fig-small" text-anchor="start">0.622 × 0.818 = 0.509</text>
</svg>
Figure 2. Each row shows the probability the model gave to what happened, and the surprise that follows. Notice that A's surprise is larger because the model was less sure about it.
```

There is a reason to add the surprises. The model gave the two true labels probabilities 0.622 and 0.818, so, if the emails are independent, it gave the pair of labels together $0.622 \times 0.818 = 0.509$. Taking $-\ln$ of a product gives the sum of the $-\ln$ terms, and $-\ln 0.509 = 0.6755$, the same total. So minimising the summed surprise is the same as choosing the weights that make the observed labels as probable as possible. This is called *maximum likelihood*.

To write this as one formula for both labels, let $y$ pick the right term:

$$
\ell(p, y) = -\bigl[\,y \ln p + (1 - y)\ln(1 - p)\,\bigr]
$$

Read it aloud: when $y = 1$ the second term vanishes and you get $-\ln p$. When $y = 0$ the first term vanishes and you get $-\ln(1-p)$. In numpy:

```python
loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
```

## Rewrite it so it stays finite

This code has a flaw. For an email that is not spam ($y = 0$) with score $z = 40$, the sigmoid is $1 - 4 \times 10^{-18}$, and a 64-bit float rounds that to exactly 1.0. Then $1 - p = 0$, `np.log(0)` is minus infinity, and the loss is `inf`, when the true loss is 40. The problem is forming $p$ at all.

Skip $p$ and write the loss directly in the score. Define the *softplus*, $\operatorname{softplus}(z) = \ln(1 + e^{z})$, a smooth version of $\max(z, 0)$. Substitute $p = 1/(1+e^{-z})$ into each term:

- When $y = 1$: $-\ln p = \ln(1 + e^{-z}) = \operatorname{softplus}(-z)$.
- When $y = 0$: $-\ln(1-p) = \ln(1 + e^{z}) = \operatorname{softplus}(z)$.

The two cases differ by one identity, $\operatorname{softplus}(z) - \operatorname{softplus}(-z) = \ln\frac{1+e^{z}}{1+e^{-z}} = \ln e^{z} = z$, which gives $\operatorname{softplus}(-z) = \operatorname{softplus}(z) - z$. Both labels then fit one formula:

$$
\ell(z, y) = \operatorname{softplus}(z) - y\,z
$$

Check it on the emails. For A, $\operatorname{softplus}(0.5) = 0.974$, and $0.974 - 1 \times 0.5 = 0.474$. For B, $\operatorname{softplus}(-1.5) = 0.201$ and $y = 0$, so the loss is 0.201. These match the table above.

The softplus can still overflow, since $e^{800}$ is infinite. Factor out the larger of the terms $1$ and $e^{z}$:

$$
\operatorname{softplus}(z) = \max(z, 0) + \ln\bigl(1 + e^{-|z|}\bigr)
$$

The exponent $-|z|$ is never positive, so nothing overflows. At $z = 800$ this gives 800, so a spam email with $y = 1$ costs $800 - 800 = 0$, and a non-spam email with $y = 0$ costs 800. The extra term is the gap between the smooth curve and $\max(z, 0)$, and it is largest, $\ln 2 = 0.693$, at $z = 0$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Softplus against max(z, 0): the smooth curve sits above the hinge, with the largest gap of 0.693 at z = 0">
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-grid" />
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="182.3" x2="560.0" y2="182.3" class="stroke-grid" />
<text x="62.0" y="186.3" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="114.6" x2="560.0" y2="114.6" class="stroke-grid" />
<text x="62.0" y="118.6" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="46.9" x2="560.0" y2="46.9" class="stroke-grid" />
<text x="62.0" y="50.9" class="fig-small" text-anchor="end">6</text>
<line x1="315.0" y1="250.0" x2="315.0" y2="30.0" class="stroke-grid" />
<polygon points="70.0,249.9 72.0,249.9 74.1,249.9 76.1,249.9 78.2,249.9 80.2,249.9 82.2,249.9 84.3,249.9 86.3,249.9 88.4,249.9 90.4,249.9 92.5,249.9 94.5,249.8 96.5,249.8 98.6,249.8 100.6,249.8 102.7,249.8 104.7,249.8 106.8,249.8 108.8,249.8 110.8,249.8 112.9,249.8 114.9,249.7 117.0,249.7 119.0,249.7 121.0,249.7 123.1,249.7 125.1,249.7 127.2,249.7 129.2,249.6 131.2,249.6 133.3,249.6 135.3,249.6 137.4,249.6 139.4,249.5 141.5,249.5 143.5,249.5 145.5,249.5 147.6,249.4 149.6,249.4 151.7,249.4 153.7,249.4 155.8,249.3 157.8,249.3 159.8,249.3 161.9,249.2 163.9,249.2 166.0,249.1 168.0,249.1 170.0,249.0 172.1,249.0 174.1,248.9 176.2,248.9 178.2,248.8 180.2,248.8 182.3,248.7 184.3,248.6 186.4,248.6 188.4,248.5 190.5,248.4 192.5,248.4 194.5,248.3 196.6,248.2 198.6,248.1 200.7,248.0 202.7,247.9 204.8,247.8 206.8,247.7 208.8,247.6 210.9,247.5 212.9,247.3 215.0,247.2 217.0,247.1 219.0,246.9 221.1,246.8 223.1,246.6 225.2,246.4 227.2,246.3 229.2,246.1 231.3,245.9 233.3,245.7 235.4,245.5 237.4,245.3 239.5,245.1 241.5,244.8 243.5,244.6 245.6,244.3 247.6,244.1 249.7,243.8 251.7,243.5 253.8,243.2 255.8,242.9 257.8,242.5 259.9,242.2 261.9,241.8 264.0,241.5 266.0,241.1 268.0,240.7 270.1,240.3 272.1,239.8 274.2,239.4 276.2,238.9 278.2,238.5 280.3,238.0 282.3,237.4 284.4,236.9 286.4,236.4 288.5,235.8 290.5,235.2 292.5,234.6 294.6,234.0 296.6,233.3 298.7,232.6 300.7,231.9 302.8,231.2 304.8,230.5 306.8,229.8 308.9,229.0 310.9,228.2 313.0,227.4 315.0,226.5 317.0,225.7 319.1,224.8 321.1,223.9 323.2,223.0 325.2,222.0 327.2,221.1 329.3,220.1 331.3,219.1 333.4,218.1 335.4,217.0 337.5,216.0 339.5,214.9 341.5,213.8 343.6,212.7 345.6,211.5 347.7,210.4 349.7,209.2 351.8,208.0 353.8,206.8 355.8,205.6 357.9,204.3 359.9,203.0 362.0,201.8 364.0,200.5 366.0,199.2 368.1,197.8 370.1,196.5 372.2,195.2 374.2,193.8 376.2,192.4 378.3,191.0 380.3,189.6 382.4,188.2 384.4,186.8 386.5,185.3 388.5,183.9 390.5,182.4 392.6,181.0 394.6,179.5 396.7,178.0 398.7,176.5 400.7,175.0 402.8,173.5 404.8,172.0 406.9,170.5 408.9,168.9 411.0,167.4 413.0,165.8 415.0,164.3 417.1,162.7 419.1,161.1 421.2,159.6 423.2,158.0 425.3,156.4 427.3,154.8 429.3,153.2 431.4,151.6 433.4,150.0 435.5,148.4 437.5,146.8 439.5,145.2 441.6,143.6 443.6,142.0 445.7,140.3 447.7,138.7 449.8,137.1 451.8,135.4 453.8,133.8 455.9,132.2 457.9,130.5 460.0,128.9 462.0,127.2 464.0,125.6 466.1,123.9 468.1,122.3 470.2,120.6 472.2,119.0 474.3,117.3 476.3,115.7 478.3,114.0 480.4,112.3 482.4,110.7 484.5,109.0 486.5,107.3 488.5,105.7 490.6,104.0 492.6,102.3 494.7,100.7 496.7,99.0 498.8,97.3 500.8,95.6 502.8,94.0 504.9,92.3 506.9,90.6 509.0,88.9 511.0,87.3 513.0,85.6 515.1,83.9 517.1,82.2 519.2,80.5 521.2,78.9 523.2,77.2 525.3,75.5 527.3,73.8 529.4,72.1 531.4,70.4 533.5,68.8 535.5,67.1 537.5,65.4 539.6,63.7 541.6,62.0 543.7,60.3 545.7,58.7 547.8,57.0 549.8,55.3 551.8,53.6 553.9,51.9 555.9,50.2 558.0,48.5 560.0,46.8 560.0,46.9 558.0,48.6 555.9,50.3 553.9,52.0 551.8,53.7 549.8,55.4 547.8,57.1 545.7,58.8 543.7,60.5 541.6,62.2 539.6,63.8 537.5,65.5 535.5,67.2 533.5,68.9 531.4,70.6 529.4,72.3 527.3,74.0 525.3,75.7 523.2,77.4 521.2,79.1 519.2,80.8 517.1,82.5 515.1,84.2 513.0,85.8 511.0,87.5 509.0,89.2 506.9,90.9 504.9,92.6 502.8,94.3 500.8,96.0 498.8,97.7 496.7,99.4 494.7,101.1 492.6,102.8 490.6,104.5 488.5,106.2 486.5,107.8 484.5,109.5 482.4,111.2 480.4,112.9 478.3,114.6 476.3,116.3 474.3,118.0 472.2,119.7 470.2,121.4 468.1,123.1 466.1,124.8 464.0,126.5 462.0,128.2 460.0,129.8 457.9,131.5 455.9,133.2 453.8,134.9 451.8,136.6 449.8,138.3 447.7,140.0 445.7,141.7 443.6,143.4 441.6,145.1 439.5,146.8 437.5,148.5 435.5,150.2 433.4,151.8 431.4,153.5 429.3,155.2 427.3,156.9 425.3,158.6 423.2,160.3 421.2,162.0 419.1,163.7 417.1,165.4 415.0,167.1 413.0,168.8 411.0,170.5 408.9,172.2 406.9,173.8 404.8,175.5 402.8,177.2 400.7,178.9 398.7,180.6 396.7,182.3 394.6,184.0 392.6,185.7 390.5,187.4 388.5,189.1 386.5,190.8 384.4,192.5 382.4,194.2 380.3,195.8 378.3,197.5 376.2,199.2 374.2,200.9 372.2,202.6 370.1,204.3 368.1,206.0 366.0,207.7 364.0,209.4 362.0,211.1 359.9,212.8 357.9,214.5 355.8,216.2 353.8,217.8 351.8,219.5 349.7,221.2 347.7,222.9 345.6,224.6 343.6,226.3 341.5,228.0 339.5,229.7 337.5,231.4 335.4,233.1 333.4,234.8 331.3,236.5 329.3,238.2 327.2,239.8 325.2,241.5 323.2,243.2 321.1,244.9 319.1,246.6 317.0,248.3 315.0,250.0 313.0,250.0 310.9,250.0 308.9,250.0 306.8,250.0 304.8,250.0 302.8,250.0 300.7,250.0 298.7,250.0 296.6,250.0 294.6,250.0 292.5,250.0 290.5,250.0 288.5,250.0 286.4,250.0 284.4,250.0 282.3,250.0 280.3,250.0 278.2,250.0 276.2,250.0 274.2,250.0 272.1,250.0 270.1,250.0 268.0,250.0 266.0,250.0 264.0,250.0 261.9,250.0 259.9,250.0 257.8,250.0 255.8,250.0 253.8,250.0 251.7,250.0 249.7,250.0 247.6,250.0 245.6,250.0 243.5,250.0 241.5,250.0 239.5,250.0 237.4,250.0 235.4,250.0 233.3,250.0 231.3,250.0 229.2,250.0 227.2,250.0 225.2,250.0 223.1,250.0 221.1,250.0 219.0,250.0 217.0,250.0 215.0,250.0 212.9,250.0 210.9,250.0 208.8,250.0 206.8,250.0 204.8,250.0 202.7,250.0 200.7,250.0 198.6,250.0 196.6,250.0 194.5,250.0 192.5,250.0 190.5,250.0 188.4,250.0 186.4,250.0 184.3,250.0 182.3,250.0 180.2,250.0 178.2,250.0 176.2,250.0 174.1,250.0 172.1,250.0 170.0,250.0 168.0,250.0 166.0,250.0 163.9,250.0 161.9,250.0 159.8,250.0 157.8,250.0 155.8,250.0 153.7,250.0 151.7,250.0 149.6,250.0 147.6,250.0 145.5,250.0 143.5,250.0 141.5,250.0 139.4,250.0 137.4,250.0 135.3,250.0 133.3,250.0 131.2,250.0 129.2,250.0 127.2,250.0 125.1,250.0 123.1,250.0 121.0,250.0 119.0,250.0 117.0,250.0 114.9,250.0 112.9,250.0 110.8,250.0 108.8,250.0 106.8,250.0 104.7,250.0 102.7,250.0 100.6,250.0 98.6,250.0 96.5,250.0 94.5,250.0 92.5,250.0 90.4,250.0 88.4,250.0 86.3,250.0 84.3,250.0 82.2,250.0 80.2,250.0 78.2,250.0 76.1,250.0 74.1,250.0 72.0,250.0 70.0,250.0" class="fill-soft-b"/>
<polyline points="70.0,250.0 72.0,250.0 74.1,250.0 76.1,250.0 78.2,250.0 80.2,250.0 82.2,250.0 84.3,250.0 86.3,250.0 88.4,250.0 90.4,250.0 92.5,250.0 94.5,250.0 96.5,250.0 98.6,250.0 100.6,250.0 102.7,250.0 104.7,250.0 106.8,250.0 108.8,250.0 110.8,250.0 112.9,250.0 114.9,250.0 117.0,250.0 119.0,250.0 121.0,250.0 123.1,250.0 125.1,250.0 127.2,250.0 129.2,250.0 131.2,250.0 133.3,250.0 135.3,250.0 137.4,250.0 139.4,250.0 141.5,250.0 143.5,250.0 145.5,250.0 147.6,250.0 149.6,250.0 151.7,250.0 153.7,250.0 155.8,250.0 157.8,250.0 159.8,250.0 161.9,250.0 163.9,250.0 166.0,250.0 168.0,250.0 170.0,250.0 172.1,250.0 174.1,250.0 176.2,250.0 178.2,250.0 180.2,250.0 182.3,250.0 184.3,250.0 186.4,250.0 188.4,250.0 190.5,250.0 192.5,250.0 194.5,250.0 196.6,250.0 198.6,250.0 200.7,250.0 202.7,250.0 204.8,250.0 206.8,250.0 208.8,250.0 210.9,250.0 212.9,250.0 215.0,250.0 217.0,250.0 219.0,250.0 221.1,250.0 223.1,250.0 225.2,250.0 227.2,250.0 229.2,250.0 231.3,250.0 233.3,250.0 235.4,250.0 237.4,250.0 239.5,250.0 241.5,250.0 243.5,250.0 245.6,250.0 247.6,250.0 249.7,250.0 251.7,250.0 253.8,250.0 255.8,250.0 257.8,250.0 259.9,250.0 261.9,250.0 264.0,250.0 266.0,250.0 268.0,250.0 270.1,250.0 272.1,250.0 274.2,250.0 276.2,250.0 278.2,250.0 280.3,250.0 282.3,250.0 284.4,250.0 286.4,250.0 288.5,250.0 290.5,250.0 292.5,250.0 294.6,250.0 296.6,250.0 298.7,250.0 300.7,250.0 302.8,250.0 304.8,250.0 306.8,250.0 308.9,250.0 310.9,250.0 313.0,250.0 315.0,250.0 317.0,248.3 319.1,246.6 321.1,244.9 323.2,243.2 325.2,241.5 327.2,239.8 329.3,238.2 331.3,236.5 333.4,234.8 335.4,233.1 337.5,231.4 339.5,229.7 341.5,228.0 343.6,226.3 345.6,224.6 347.7,222.9 349.7,221.2 351.8,219.5 353.8,217.8 355.8,216.2 357.9,214.5 359.9,212.8 362.0,211.1 364.0,209.4 366.0,207.7 368.1,206.0 370.1,204.3 372.2,202.6 374.2,200.9 376.2,199.2 378.3,197.5 380.3,195.8 382.4,194.2 384.4,192.5 386.5,190.8 388.5,189.1 390.5,187.4 392.6,185.7 394.6,184.0 396.7,182.3 398.7,180.6 400.7,178.9 402.8,177.2 404.8,175.5 406.9,173.8 408.9,172.2 411.0,170.5 413.0,168.8 415.0,167.1 417.1,165.4 419.1,163.7 421.2,162.0 423.2,160.3 425.3,158.6 427.3,156.9 429.3,155.2 431.4,153.5 433.4,151.8 435.5,150.2 437.5,148.5 439.5,146.8 441.6,145.1 443.6,143.4 445.7,141.7 447.7,140.0 449.8,138.3 451.8,136.6 453.8,134.9 455.9,133.2 457.9,131.5 460.0,129.8 462.0,128.2 464.0,126.5 466.1,124.8 468.1,123.1 470.2,121.4 472.2,119.7 474.3,118.0 476.3,116.3 478.3,114.6 480.4,112.9 482.4,111.2 484.5,109.5 486.5,107.8 488.5,106.2 490.6,104.5 492.6,102.8 494.7,101.1 496.7,99.4 498.8,97.7 500.8,96.0 502.8,94.3 504.9,92.6 506.9,90.9 509.0,89.2 511.0,87.5 513.0,85.8 515.1,84.2 517.1,82.5 519.2,80.8 521.2,79.1 523.2,77.4 525.3,75.7 527.3,74.0 529.4,72.3 531.4,70.6 533.5,68.9 535.5,67.2 537.5,65.5 539.6,63.8 541.6,62.2 543.7,60.5 545.7,58.8 547.8,57.1 549.8,55.4 551.8,53.7 553.9,52.0 555.9,50.3 558.0,48.6 560.0,46.9" class="stroke-muted dashed" fill="none"/>
<polyline points="70.0,249.9 72.0,249.9 74.1,249.9 76.1,249.9 78.2,249.9 80.2,249.9 82.2,249.9 84.3,249.9 86.3,249.9 88.4,249.9 90.4,249.9 92.5,249.9 94.5,249.8 96.5,249.8 98.6,249.8 100.6,249.8 102.7,249.8 104.7,249.8 106.8,249.8 108.8,249.8 110.8,249.8 112.9,249.8 114.9,249.7 117.0,249.7 119.0,249.7 121.0,249.7 123.1,249.7 125.1,249.7 127.2,249.7 129.2,249.6 131.2,249.6 133.3,249.6 135.3,249.6 137.4,249.6 139.4,249.5 141.5,249.5 143.5,249.5 145.5,249.5 147.6,249.4 149.6,249.4 151.7,249.4 153.7,249.4 155.8,249.3 157.8,249.3 159.8,249.3 161.9,249.2 163.9,249.2 166.0,249.1 168.0,249.1 170.0,249.0 172.1,249.0 174.1,248.9 176.2,248.9 178.2,248.8 180.2,248.8 182.3,248.7 184.3,248.6 186.4,248.6 188.4,248.5 190.5,248.4 192.5,248.4 194.5,248.3 196.6,248.2 198.6,248.1 200.7,248.0 202.7,247.9 204.8,247.8 206.8,247.7 208.8,247.6 210.9,247.5 212.9,247.3 215.0,247.2 217.0,247.1 219.0,246.9 221.1,246.8 223.1,246.6 225.2,246.4 227.2,246.3 229.2,246.1 231.3,245.9 233.3,245.7 235.4,245.5 237.4,245.3 239.5,245.1 241.5,244.8 243.5,244.6 245.6,244.3 247.6,244.1 249.7,243.8 251.7,243.5 253.8,243.2 255.8,242.9 257.8,242.5 259.9,242.2 261.9,241.8 264.0,241.5 266.0,241.1 268.0,240.7 270.1,240.3 272.1,239.8 274.2,239.4 276.2,238.9 278.2,238.5 280.3,238.0 282.3,237.4 284.4,236.9 286.4,236.4 288.5,235.8 290.5,235.2 292.5,234.6 294.6,234.0 296.6,233.3 298.7,232.6 300.7,231.9 302.8,231.2 304.8,230.5 306.8,229.8 308.9,229.0 310.9,228.2 313.0,227.4 315.0,226.5 317.0,225.7 319.1,224.8 321.1,223.9 323.2,223.0 325.2,222.0 327.2,221.1 329.3,220.1 331.3,219.1 333.4,218.1 335.4,217.0 337.5,216.0 339.5,214.9 341.5,213.8 343.6,212.7 345.6,211.5 347.7,210.4 349.7,209.2 351.8,208.0 353.8,206.8 355.8,205.6 357.9,204.3 359.9,203.0 362.0,201.8 364.0,200.5 366.0,199.2 368.1,197.8 370.1,196.5 372.2,195.2 374.2,193.8 376.2,192.4 378.3,191.0 380.3,189.6 382.4,188.2 384.4,186.8 386.5,185.3 388.5,183.9 390.5,182.4 392.6,181.0 394.6,179.5 396.7,178.0 398.7,176.5 400.7,175.0 402.8,173.5 404.8,172.0 406.9,170.5 408.9,168.9 411.0,167.4 413.0,165.8 415.0,164.3 417.1,162.7 419.1,161.1 421.2,159.6 423.2,158.0 425.3,156.4 427.3,154.8 429.3,153.2 431.4,151.6 433.4,150.0 435.5,148.4 437.5,146.8 439.5,145.2 441.6,143.6 443.6,142.0 445.7,140.3 447.7,138.7 449.8,137.1 451.8,135.4 453.8,133.8 455.9,132.2 457.9,130.5 460.0,128.9 462.0,127.2 464.0,125.6 466.1,123.9 468.1,122.3 470.2,120.6 472.2,119.0 474.3,117.3 476.3,115.7 478.3,114.0 480.4,112.3 482.4,110.7 484.5,109.0 486.5,107.3 488.5,105.7 490.6,104.0 492.6,102.3 494.7,100.7 496.7,99.0 498.8,97.3 500.8,95.6 502.8,94.0 504.9,92.3 506.9,90.6 509.0,88.9 511.0,87.3 513.0,85.6 515.1,83.9 517.1,82.2 519.2,80.5 521.2,78.9 523.2,77.2 525.3,75.5 527.3,73.8 529.4,72.1 531.4,70.4 533.5,68.8 535.5,67.1 537.5,65.4 539.6,63.7 541.6,62.0 543.7,60.3 545.7,58.7 547.8,57.0 549.8,55.3 551.8,53.6 553.9,51.9 555.9,50.2 558.0,48.5 560.0,46.8" class="stroke-b" fill="none"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted" />
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">-6</text>
<line x1="151.7" y1="250.0" x2="151.7" y2="255.0" class="stroke-muted" />
<text x="151.7" y="270.0" class="fig-small" text-anchor="middle">-4</text>
<line x1="233.3" y1="250.0" x2="233.3" y2="255.0" class="stroke-muted" />
<text x="233.3" y="270.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="315.0" y1="250.0" x2="315.0" y2="255.0" class="stroke-muted" />
<text x="315.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="396.7" y1="250.0" x2="396.7" y2="255.0" class="stroke-muted" />
<text x="396.7" y="270.0" class="fig-small" text-anchor="middle">2</text>
<line x1="478.3" y1="250.0" x2="478.3" y2="255.0" class="stroke-muted" />
<text x="478.3" y="270.0" class="fig-small" text-anchor="middle">4</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted" />
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">6</text>
<text x="315.0" y="290.0" text-anchor="middle">score z</text>
<circle cx="315.0" cy="226.5" r="5" class="fill-b"/>
<text x="303.0" y="212.5" class="fig-small" text-anchor="end">gap 0.693 at z = 0</text>
<text x="445.7" y="60.5" class="fig-bold" text-anchor="start">softplus(z)</text>
<text x="413.0" y="219.5" class="fig-small" text-anchor="start">dashed: max(z, 0)</text>
<text x="74.1" y="192.5" class="fig-small" text-anchor="start">shaded gap: ln(1 + exp(−|z|))</text>
</svg>
Figure 3. Softplus hugs max(z, 0) far from zero, and rounds the corner near zero. Notice that the shaded gap is largest at z = 0 and shrinks to nothing on both sides, so the second term only matters near the corner and can never overflow.
```

```python
softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
loss = np.sum(softplus - y * z)
```

## Add the ridge fee and check the total

The full loss adds an optional fee on large weights, which keeps the fit from chasing extreme probabilities (the next chapter shows a case where this matters), the *ridge fee* from Ridge: a fee on large weights, with strength $\lambda$ (`ridge` in code). With $\lambda = 0$ there is no fee:

$$
L(w) = \sum_{i=1}^{n}\Bigl[\operatorname{softplus}(z_i) - y_i z_i\Bigr] + \frac{\lambda}{2}\lVert w\rVert^2, \qquad z = Xw
$$

This sums over the rows, so adding a row can only add loss. A mean would divide the two-email total by $n = 2$ and give 0.338, which is the same minimum but changes the size of the gradient you meet in the next chapter.

You can check any implementation without a calculator. At $w = 0$ every score is 0, every probability is 0.5, and every row costs $\ln 2 = 0.693$, whatever its label. Ten rows give $10 \ln 2 = 6.931$. The ridge fee for weights $3$ and $-4$ at $\lambda = 2$ is $\tfrac{2}{2}(9 + 16) = 25$.

> [!KEY]
> The log loss is the surprise of the true label: minus the log of the probability the model gave to it. A confident wrong answer costs a lot, as the patient's 6.9 shows, and the slope stays large exactly where the model is most wrong. The stable form $\operatorname{softplus}(z) - yz$ computes it from the score, so no probability ever rounds to 1.

> [!RECAP]
> - Surprise, $-\ln p$ for the probability given to the truth, is 0 when right and sure, and grows without limit as the truth gets less likely.
> - Squared error gives a mild and a confident mistake similar scores, and its slope vanishes where the model is most wrong.
> - Summing the surprises is the same as maximising the likelihood of the labels.
> - The loss for one row is $\operatorname{softplus}(z) - yz$, with $\operatorname{softplus}(z) = \max(z, 0) + \ln(1 + e^{-|z|})$.
> - At zero weights the loss is $n \ln 2$.

## Build it step by step

Build three functions. The sigmoid from the last chapter is already provided.

1. `softplus(z)` takes an array of any shape and returns the same shape. Use `np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))`. The common mistake is `np.log(1 + np.exp(z))`, which overflows at 710.

2. `log_loss(w, X, y, ridge)` takes weights `(d,)`, a matrix `(n, d)`, labels `(n,)` of 0 and 1, and a ridge strength, and returns one float. Compute `z = X @ w`, sum `softplus(z) - y * z`, and add `0.5 * ridge * np.dot(w, w)`. The common mistake is a mean in place of the sum.

3. `naive_log_loss(p, y)` takes probabilities and labels, and returns $-\sum[y\ln p + (1-y)\ln(1-p)]$. It exists so that you can see `log_loss` agree with the textbook formula at moderate scores, and disagree where the textbook formula breaks.

## Where this breaks

- **Labels other than 0 and 1.** The formula `softplus(z) - y * z` assumes $y \in \{0, 1\}$. With labels $-1$ and $+1$ the code runs and returns wrong numbers, so convert the labels first.
- **A rare class.** If 99 of 100 rows are 0, a model that always says "no" earns a low loss, since every row counts equally. Check the loss of each class, not only the total.

## The exercise

You implement `softplus`, `log_loss`, and `naive_log_loss`. The checks run `softplus` at $\pm 800$ and compare it with `np.log1p(np.exp(z))` at moderate scores. They check that `log_loss` agrees with `naive_log_loss` on moderate data, stays finite at $\pm 800$, equals $n \ln 2$ at zero weights, adds exactly $(\lambda/2)\lVert w\rVert^2$ for the ridge fee, and returns 0.6755 on the two emails.
