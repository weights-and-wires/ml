---
title: From a score to a probability
slug: 015-sigmoid-and-probability
part: "Part 1 — Foundations"
summary: "The sigmoid squashes any score into a probability: score 0 is a coin flip, and each +1 multiplies the odds by e."
minutes: 35
gpu: false
objectives:
  - Explain why a raw score, a clipped score, and a hard threshold each fail as a probability.
  - Read a score as log-odds, and say what a weight of 0.5 or -1 does to the odds.
  - Compute the sigmoid and its inverse, the logit, without overflow at a score of 800.
  - Turn probabilities into class labels with a threshold, and say when to move the threshold.
prereqs:
  - 005-least-squares-line
  - 011-ridge-penalty-and-closed-form
exercise: 015-sigmoid-and-probability
---

# From a score to a probability

A spam filter reads an email and adds up evidence. Each suspicious word pushes a number up, and each sign of a real sender pushes it down. For one email the total comes out as 3.2. Is that 3.2 percent spam, 3.2 times as likely as not, or something else? What should the filter say when the total is -50, or 800?

You cannot answer yet, because a total like this can be any number on the whole line, and a probability must sit between 0 and 1. Without a rule that connects the two, you cannot say "I am 90 percent sure", you cannot compare two emails fairly, and you cannot ask later chapters to fit the weights by measuring how wrong a probability was.

This chapter builds that rule from the question "what does a score mean?" The answer is the sigmoid, an S-shaped curve. You also get a way to compute it that stays finite when the model is very sure.

## What you already know

- A linear model multiplies each feature by a weight and adds the results, which gives one number per example (see Fit a line by minimising squared misses).
- A data matrix $X$ has one row per example and one column per feature, so `X @ w` scores every row at once.
- A probability is a number from 0 to 1, and the probabilities of "spam" and "not spam" add to 1.
- `np.exp(x)` grows very fast, and a 64-bit float cannot hold it once $x$ passes about 709.
- The logarithm undoes the exponential, so $\ln(e^{z}) = z$.

## Score each email

Take two emails and one feature: how many times you have replied to that sender before. Email A has 0 replies and is spam. Email B has 2 replies and is not spam. You also want a constant starting point, so add a column of ones, called the *bias* column. A bias is a weight that does not depend on any feature. Each row is then $(1, \text{replies})$:

$$
X = \begin{bmatrix} 1 & 0 \\ 1 & 2 \end{bmatrix}, \qquad
w = \begin{bmatrix} 0.5 \\ -1 \end{bmatrix}, \qquad
z = Xw = \begin{bmatrix} 0.5 \\ -1.5 \end{bmatrix}
$$

Here $n = 2$ is the number of rows and $d = 2$ is the number of weights. The score $z$ is one number per row. Weight $-1$ means each earlier reply pulls the spam score down by 1. This is the pair of emails that the next chapters reuse.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The sigmoid S-curve: score on the horizontal axis, probability on the vertical axis, with the points at -2, 0, and 2 and the two example emails">
<line x1="70.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-grid" />
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="145.0" x2="570.0" y2="145.0" class="stroke-grid" />
<text x="62.0" y="149.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid" />
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<rect x="70.0" y="40.0" width="93.8" height="210.0" class="fill-soft-a" />
<rect x="476.2" y="40.0" width="93.8" height="210.0" class="fill-soft-a" />
<text x="116.9" y="200.0" class="fig-small" text-anchor="middle">p barely moves</text>
<text x="523.1" y="80.0" class="fig-small" text-anchor="middle">p barely moves</text>
<line x1="320.0" y1="250.0" x2="320.0" y2="40.0" class="stroke-grid" />
<line x1="70.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted" />
<polyline points="70.0,249.9 73.1,249.9 76.2,249.9 79.4,249.9 82.5,249.9 85.6,249.9 88.7,249.9 91.9,249.9 95.0,249.8 98.1,249.8 101.2,249.8 104.4,249.8 107.5,249.8 110.6,249.7 113.8,249.7 116.9,249.7 120.0,249.7 123.1,249.6 126.2,249.6 129.4,249.5 132.5,249.5 135.6,249.4 138.8,249.4 141.9,249.3 145.0,249.2 148.1,249.1 151.2,249.1 154.4,249.0 157.5,248.8 160.6,248.7 163.8,248.6 166.9,248.4 170.0,248.3 173.1,248.1 176.2,247.9 179.4,247.7 182.5,247.5 185.6,247.2 188.8,246.9 191.9,246.6 195.0,246.2 198.1,245.8 201.2,245.4 204.4,244.9 207.5,244.4 210.6,243.8 213.8,243.2 216.9,242.5 220.0,241.8 223.1,240.9 226.2,240.0 229.4,239.0 232.5,238.0 235.6,236.8 238.8,235.5 241.9,234.1 245.0,232.5 248.1,230.9 251.3,229.1 254.4,227.1 257.5,225.0 260.6,222.7 263.8,220.2 266.9,217.6 270.0,214.7 273.1,211.7 276.2,208.5 279.4,205.0 282.5,201.4 285.6,197.6 288.8,193.5 291.9,189.3 295.0,184.9 298.1,180.3 301.2,175.6 304.4,170.7 307.5,165.7 310.6,160.6 313.8,155.5 316.9,150.2 320.0,145.0 323.1,139.8 326.3,134.5 329.4,129.4 332.5,124.3 335.6,119.3 338.8,114.4 341.9,109.7 345.0,105.1 348.1,100.7 351.2,96.5 354.4,92.4 357.5,88.6 360.6,85.0 363.8,81.5 366.9,78.3 370.0,75.3 373.1,72.4 376.2,69.8 379.4,67.3 382.5,65.0 385.6,62.9 388.8,60.9 391.9,59.1 395.0,57.5 398.1,55.9 401.3,54.5 404.4,53.2 407.5,52.0 410.6,51.0 413.8,50.0 416.9,49.1 420.0,48.2 423.1,47.5 426.2,46.8 429.4,46.2 432.5,45.6 435.6,45.1 438.8,44.6 441.9,44.2 445.0,43.8 448.1,43.4 451.3,43.1 454.4,42.8 457.5,42.5 460.6,42.3 463.8,42.1 466.9,41.9 470.0,41.7 473.1,41.6 476.2,41.4 479.4,41.3 482.5,41.2 485.6,41.0 488.8,40.9 491.9,40.9 495.0,40.8 498.1,40.7 501.2,40.6 504.4,40.6 507.5,40.5 510.6,40.5 513.8,40.4 516.9,40.4 520.0,40.3 523.1,40.3 526.2,40.3 529.4,40.3 532.5,40.2 535.6,40.2 538.8,40.2 541.9,40.2 545.0,40.2 548.1,40.1 551.2,40.1 554.4,40.1 557.5,40.1 560.6,40.1 563.8,40.1 566.9,40.1 570.0,40.1" class="stroke-b" fill="none"/>
<line x1="132.5" y1="250.0" x2="132.5" y2="255.0" class="stroke-muted" />
<text x="132.5" y="270.0" class="fig-small" text-anchor="middle">-6</text>
<line x1="195.0" y1="250.0" x2="195.0" y2="255.0" class="stroke-muted" />
<text x="195.0" y="270.0" class="fig-small" text-anchor="middle">-4</text>
<line x1="257.5" y1="250.0" x2="257.5" y2="255.0" class="stroke-muted" />
<text x="257.5" y="270.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="320.0" y1="250.0" x2="320.0" y2="255.0" class="stroke-muted" />
<text x="320.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="382.5" y1="250.0" x2="382.5" y2="255.0" class="stroke-muted" />
<text x="382.5" y="270.0" class="fig-small" text-anchor="middle">2</text>
<line x1="445.0" y1="250.0" x2="445.0" y2="255.0" class="stroke-muted" />
<text x="445.0" y="270.0" class="fig-small" text-anchor="middle">4</text>
<line x1="507.5" y1="250.0" x2="507.5" y2="255.0" class="stroke-muted" />
<text x="507.5" y="270.0" class="fig-small" text-anchor="middle">6</text>
<text x="320.0" y="290.0" text-anchor="middle">score z</text>
<text x="70.0" y="24.0" text-anchor="start">probability</text>
<circle cx="257.5" cy="225.0" r="5" class="fill-b"/>
<text x="247.5" y="217.0" class="fig-small" text-anchor="end">z = -2: 0.12</text>
<circle cx="320.0" cy="145.0" r="5" class="fill-b"/>
<text x="310.0" y="137.0" class="fig-small" text-anchor="end">z = 0: 0.50</text>
<circle cx="382.5" cy="65.0" r="5" class="fill-b"/>
<text x="392.5" y="83.0" class="fig-small" text-anchor="start">z = 2: 0.88</text>
<circle cx="335.6" cy="119.3" r="4" class="fill-a"/>
<circle cx="273.1" cy="211.7" r="4" class="fill-a"/>
<text x="343.6" y="113.3" class="fig-bold" text-anchor="start">A</text>
<text x="278.1" y="227.7" class="fig-bold" text-anchor="start">B</text>
</svg>
Figure 1. The sigmoid turns any score into a probability. Notice that it crosses 0.5 at score 0, and that past about ±5 the curve is nearly flat, so a more extreme score adds almost nothing. Emails A and B from this section sit at 0.62 and 0.18.
```

## Reject the three obvious fixes

You might try to use the score as the probability. A score of 3.2 would read as 320 percent, and $-50$ as a negative chance, so this fails before you start.

You might then clip the score into the range 0 to 1. Every score above 1 then reads as exactly 1, which means a score of 1.5 and a score of 800 look the same. The slope is also 0 in the clipped region, and later chapters learn by following slopes, so a clipped model gets no signal exactly when it is wrong and sure.

Last, you might compare the score with 0 and answer yes or no. That gives a probability of 0 or 1 and nothing between, so 0.51 and 0.99 look identical, and again the slope is 0 everywhere except at one point.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three ways to turn a score into a probability: the raw score leaves the 0 to 1 band, a hard step has no confidence, and the sigmoid is smooth and bounded">
<rect x="20.0" y="67.8" width="170.0" height="94.4" class="fill-soft-a" />
<line x1="20.0" y1="162.2" x2="190.0" y2="162.2" class="stroke-grid" />
<line x1="20.0" y1="67.8" x2="190.0" y2="67.8" class="stroke-grid" />
<text x="17.0" y="166.2" class="fig-small" text-anchor="end">0</text>
<text x="17.0" y="71.8" class="fig-small" text-anchor="end">1</text>
<polyline points="73.1,162.2 74.2,160.6 75.2,159.1 76.3,157.5 77.4,155.9 78.4,154.4 79.5,152.8 80.6,151.2 81.6,149.6 82.7,148.1 83.8,146.5 84.8,144.9 85.9,143.3 86.9,141.8 88.0,140.2 89.1,138.6 90.1,137.0 91.2,135.5 92.3,133.9 93.3,132.3 94.4,130.7 95.4,129.2 96.5,127.6 97.6,126.0 98.6,124.4 99.7,122.9 100.8,121.3 101.8,119.7 102.9,118.1 103.9,116.6 105.0,115.0 106.1,113.4 107.1,111.9 108.2,110.3 109.2,108.7 110.3,107.1 111.4,105.6 112.4,104.0 113.5,102.4 114.6,100.8 115.6,99.3 116.7,97.7 117.8,96.1 118.8,94.5 119.9,93.0 120.9,91.4 122.0,89.8 123.1,88.2 124.1,86.7 125.2,85.1 126.2,83.5 127.3,81.9 128.4,80.4 129.4,78.8 130.5,77.2 131.6,75.6 132.6,74.1 133.7,72.5 134.8,70.9 135.8,69.4 136.9,67.8" class="stroke-b" fill="none"/>
<polyline points="136.9,67.8 162.4,30.0" class="stroke-bad" fill="none"/>
<polyline points="73.1,162.2 47.6,200.0" class="stroke-bad" fill="none"/>
<line x1="105.0" y1="200.0" x2="105.0" y2="30.0" class="stroke-grid" />
<text x="105.0" y="18.0" class="fig-bold" text-anchor="middle">Raw score</text>
<text x="105.0" y="222.0" class="fig-small" text-anchor="middle">probabilities below 0</text>
<text x="105.0" y="238.0" class="fig-small" text-anchor="middle">and above 1</text>
<rect x="215.0" y="67.8" width="170.0" height="94.4" class="fill-soft-a" />
<line x1="215.0" y1="162.2" x2="385.0" y2="162.2" class="stroke-grid" />
<line x1="215.0" y1="67.8" x2="385.0" y2="67.8" class="stroke-grid" />
<text x="212.0" y="166.2" class="fig-small" text-anchor="end">0</text>
<text x="212.0" y="71.8" class="fig-small" text-anchor="end">1</text>
<polyline points="215.0,162.2 300.0,162.2 300.0,67.8 385.0,67.8" class="stroke-b" fill="none"/>
<line x1="300.0" y1="200.0" x2="300.0" y2="30.0" class="stroke-grid" />
<text x="300.0" y="18.0" class="fig-bold" text-anchor="middle">Hard threshold</text>
<text x="300.0" y="222.0" class="fig-small" text-anchor="middle">0.51 and 0.99 look the same;</text>
<text x="300.0" y="238.0" class="fig-small" text-anchor="middle">no slope to learn from</text>
<rect x="410.0" y="67.8" width="170.0" height="94.4" class="fill-soft-a" />
<line x1="410.0" y1="162.2" x2="580.0" y2="162.2" class="stroke-grid" />
<line x1="410.0" y1="67.8" x2="580.0" y2="67.8" class="stroke-grid" />
<text x="407.0" y="166.2" class="fig-small" text-anchor="end">0</text>
<text x="407.0" y="71.8" class="fig-small" text-anchor="end">1</text>
<polyline points="410.0,160.5 411.1,160.4 412.1,160.3 413.2,160.3 414.2,160.2 415.3,160.1 416.4,159.9 417.4,159.8 418.5,159.7 419.6,159.6 420.6,159.5 421.7,159.3 422.8,159.2 423.8,159.0 424.9,158.9 425.9,158.7 427.0,158.5 428.1,158.3 429.1,158.2 430.2,158.0 431.2,157.7 432.3,157.5 433.4,157.3 434.4,157.1 435.5,156.8 436.6,156.5 437.6,156.3 438.7,156.0 439.8,155.7 440.8,155.4 441.9,155.1 442.9,154.7 444.0,154.4 445.1,154.0 446.1,153.6 447.2,153.2 448.2,152.8 449.3,152.4 450.4,151.9 451.4,151.5 452.5,151.0 453.6,150.5 454.6,149.9 455.7,149.4 456.8,148.8 457.8,148.2 458.9,147.6 459.9,147.0 461.0,146.4 462.1,145.7 463.1,145.0 464.2,144.3 465.2,143.5 466.3,142.8 467.4,142.0 468.4,141.2 469.5,140.4 470.6,139.5 471.6,138.6 472.7,137.7 473.8,136.8 474.8,135.9 475.9,134.9 476.9,133.9 478.0,132.9 479.1,131.9 480.1,130.9 481.2,129.8 482.2,128.8 483.3,127.7 484.4,126.6 485.4,125.4 486.5,124.3 487.6,123.2 488.6,122.0 489.7,120.9 490.8,119.7 491.8,118.5 492.9,117.4 493.9,116.2 495.0,115.0 496.1,113.8 497.1,112.6 498.2,111.5 499.2,110.3 500.3,109.1 501.4,108.0 502.4,106.8 503.5,105.7 504.6,104.6 505.6,103.4 506.7,102.3 507.8,101.2 508.8,100.2 509.9,99.1 510.9,98.1 512.0,97.1 513.1,96.1 514.1,95.1 515.2,94.1 516.2,93.2 517.3,92.3 518.4,91.4 519.4,90.5 520.5,89.6 521.6,88.8 522.6,88.0 523.7,87.2 524.8,86.5 525.8,85.7 526.9,85.0 527.9,84.3 529.0,83.6 530.1,83.0 531.1,82.4 532.2,81.8 533.2,81.2 534.3,80.6 535.4,80.1 536.4,79.5 537.5,79.0 538.6,78.5 539.6,78.1 540.7,77.6 541.8,77.2 542.8,76.8 543.9,76.4 544.9,76.0 546.0,75.6 547.1,75.3 548.1,74.9 549.2,74.6 550.2,74.3 551.3,74.0 552.4,73.7 553.4,73.5 554.5,73.2 555.6,72.9 556.6,72.7 557.7,72.5 558.8,72.3 559.8,72.0 560.9,71.8 561.9,71.7 563.0,71.5 564.1,71.3 565.1,71.1 566.2,71.0 567.2,70.8 568.3,70.7 569.4,70.5 570.4,70.4 571.5,70.3 572.6,70.2 573.6,70.1 574.7,69.9 575.8,69.8 576.8,69.7 577.9,69.7 578.9,69.6 580.0,69.5" class="stroke-b" fill="none"/>
<line x1="495.0" y1="200.0" x2="495.0" y2="30.0" class="stroke-grid" />
<text x="495.0" y="18.0" class="fig-bold" text-anchor="middle">Sigmoid</text>
<text x="495.0" y="222.0" class="fig-small" text-anchor="middle">smooth and bounded,</text>
<text x="495.0" y="238.0" class="fig-small" text-anchor="middle">a slope everywhere</text>
</svg>
Figure 2. Each panel plots probability against score from −4 to 4. Notice that only the sigmoid stays inside the shaded 0-to-1 band and still has a slope at every score.
```

You want a curve that is bounded, smooth, and equal to 0.5 at a score of 0, because a score of 0 should mean "no evidence either way". Its tails should flatten gently, since being very sure should not need an infinite score.

## Read the score as log-odds

*Odds* compare the two outcomes: if spam has probability $p$, the odds are $p/(1-p)$. At $p = 0.8$ the odds are 4, or "4 to 1 for spam". Odds run from 0 to infinity, so they cannot be a score yet. Their logarithm, the *log-odds*, runs over the whole line: it is 0 at even odds, positive when spam is more likely, and negative otherwise.

Why log-odds, and not some other stretch of the 0-to-1 range? Think of each clue as a vote that multiplies the odds. A suspicious word might double them, and a reply from a known sender might halve them, and it does not matter whether you were at 1 to 1 or 20 to 1 when the clue arrived. Multiplying odds is adding logs, and adding is exactly what a linear model does with its weighted features.

So make the one modelling choice that this chapter rests on: *the score is the log-odds of spam*. A score of $z$ then means odds of $e^{z}$. At $z = 2$ the odds are $e^2 = 7.39$, so $p = 7.39 / 8.39 = 0.881$. At $z = 3.2$ the answer to the opening question is $p = 0.961$. Each step of +1 in the score multiplies the odds by $e \approx 2.72$.

This also reads the weights. Weight $-1$ on replies multiplies the spam odds by $e^{-1} = 0.37$ per reply, so two replies cut the odds to about 14 percent of where they were. For our emails, A has score 0.5, odds $e^{0.5} = 1.65$, and $p = 1.65/2.65 = 0.622$. B has score $-1.5$, odds $0.223$, and $p = 0.223/1.223 = 0.182$.

To get a formula, turn odds back into a probability. If the odds are $e^{z}$, then $p = e^{z}/(1+e^{z})$. Multiply top and bottom by $e^{-z}$ and you get the usual form:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad
\operatorname{logit}(p) = \ln\frac{p}{1-p}
$$

Read it aloud: $\sigma$, the sigmoid, takes a score and returns the probability whose log-odds equal that score. The logit goes the other way, from probability back to score, so $\operatorname{logit}(\sigma(z)) = z$. In numpy:

```python
p = 1.0 / (1.0 + np.exp(-z))
z = np.log(p / (1.0 - p))
```

## Turn probabilities into labels

A probability is not yet a yes or no. To label an email, pick a *threshold* and say "spam" when $p$ is at or above it. With a threshold of 0.5, the label is spam exactly when $z \geq 0$, because $\sigma(0) = 0.5$ and the curve only rises. Email A (0.622) is labelled spam and B (0.182) is not.

The threshold is a choice about costs, not about the model. If sending real mail to the spam folder hurts more than letting spam through, use 0.9. Then A, at 0.622, stays in your inbox. The probability is the model's belief, and the threshold is your decision rule.

## Compute it without overflow

The one-line formula has a trap. At $z = -800$ it computes `np.exp(800)`, which is infinity, because a float tops out near $10^{308}$. The answer comes out as 0.0, which is right, but numpy prints an overflow warning on the way. The other form, $e^{z}/(1+e^{z})$, fails worse: at $z = 800$ it divides infinity by infinity and returns `nan`.

The fix is to exponentiate only a number that is zero or negative. For $z \geq 0$, use $1/(1+e^{-z})$. For $z < 0$, use $e^{z}/(1+e^{z})$, which is the same number as the first form, by the multiplication above, and which only calls the exponential on a negative number. Neither branch can overflow, so a score of $\pm 800$ gives 1 and 0, and a score of $-20$ gives $2.06 \times 10^{-9}$, not 0.

```figure
<svg viewBox="0 0 600 200" role="img" aria-label="A number line of scores from -800 to 800: the one-line formula overflows for scores below -709, the two-branch form never does">
<rect x="40.0" y="66.0" width="29.6" height="28.0" class="fill-soft-a" />
<rect x="69.6" y="66.0" width="490.4" height="28.0" class="fill-soft-c" />
<line x1="40.0" y1="80.0" x2="560.0" y2="80.0" class="stroke-muted" />
<line x1="40.0" y1="62.0" x2="40.0" y2="98.0" class="stroke-muted" />
<text x="30.0" y="116.0" class="fig-small" text-anchor="start">-800</text>
<line x1="69.6" y1="62.0" x2="69.6" y2="98.0" class="stroke-muted" />
<text x="69.6" y="132.0" class="fig-small" text-anchor="middle">-709</text>
<line x1="300.0" y1="62.0" x2="300.0" y2="98.0" class="stroke-muted" />
<text x="300.0" y="116.0" class="fig-small" text-anchor="middle">0</text>
<line x1="560.0" y1="62.0" x2="560.0" y2="98.0" class="stroke-muted" />
<text x="560.0" y="116.0" class="fig-small" text-anchor="middle">800</text>
<text x="30.0" y="56.0" class="fig-small" text-anchor="start">exp(−z) overflows below −709</text>
<text x="332.5" y="56.0" class="fig-small" text-anchor="middle">1/(1+exp(−z)) is safe</text>
<rect x="40.0" y="136.0" width="260.0" height="28.0" class="fill-soft-b" />
<rect x="300.0" y="136.0" width="260.0" height="28.0" class="fill-soft-c" />
<text x="170.0" y="154.0" class="fig-small" text-anchor="middle">z < 0: exp(z)/(1+exp(z))</text>
<text x="430.0" y="154.0" class="fig-small" text-anchor="middle">z ≥ 0: 1/(1+exp(−z))</text>
<line x1="300.0" y1="132.0" x2="300.0" y2="168.0" class="stroke-ink" />
<text x="300.0" y="186.0" class="fig-small" text-anchor="middle">both branches give 0.5</text>
</svg>
Figure 3. The top strip shows where the one-line formula overflows. The bottom strip shows the two-branch rule that avoids it, with both branches meeting at 0.5 at the split.
```

The two branches have this shape:

```python
out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
```

> [!WARNING]
> `np.where(z >= 0, branch_a, branch_b)` computes both branches on every entry before it picks one, so the unused branch still overflows and warns. Index with a mask instead.

> [!KEY]
> A score is log-odds: 0 is a coin flip, and each +1 multiplies the odds by $e$. The sigmoid turns that score into a probability, so 3.2 means about 96 percent, $-50$ means about $2 \times 10^{-22}$, and 800 means 1.0, as long as you only exponentiate a number that is zero or negative.

> [!RECAP]
> - A raw score, a clipped score, and a hard threshold each fail as a probability: they leave the 0-to-1 range, go flat, or have no confidence.
> - The score is log-odds, so odds are $e^{z}$ and $p = 1/(1+e^{-z})$.
> - The logit, $\ln(p/(1-p))$, undoes the sigmoid.
> - A threshold turns a probability into a label, and you move it when one mistake costs more than the other.
> - Use one branch for $z \geq 0$ and the other for $z < 0$, so no exponent is positive.

## Build it step by step

Build three functions, in this order:

1. `sigmoid(z)` takes an array of any shape and returns the same shape. Make a mask for `z >= 0`, allocate the output with `np.empty_like`, and fill each part with the branch that only exponentiates a non-positive number. The common mistake is `np.exp(-z)` on every entry.

2. `logit(p)` takes probabilities strictly between 0 and 1 and returns scores. Compute `np.log(p / (1 - p))`. The check is a round trip, `logit(sigmoid(z))`, which must return `z`.

3. `predict_proba(w, X)` takes weights of shape `(d,)` and a matrix of shape `(n, d)`, and returns the probabilities as shape `(n,)`. Score with `X @ w` and pass the result to your `sigmoid`. The common mistake is to write `X * w`, which multiplies without summing.

## Where this breaks

The sigmoid only turns a score into a probability. It does not make that probability true. If the real log-odds are not close to a linear function of your features, the model can be confidently wrong, and its 90 percent will not match what you see in the data. Holding data back, as in Honest scores: training error, test error, and holding data back, is how you find out whether the numbers are trustworthy.

## The exercise

You implement `sigmoid`, `logit`, and `predict_proba`. Applying the threshold is a one-line comparison, so it is not a separate function here, and `train_accuracy` in the later chapter "Learning the weights: gradient and descent" uses it. The checks compare `sigmoid` with known values, run it with numpy's overflow errors turned on at scores of $\pm 800$, and confirm that $\sigma(-z) = 1 - \sigma(z)$. They also check that `logit` inverts it, that a +1 step in the score multiplies the odds by $e$, and that `predict_proba` reproduces the 0.622 and 0.182 of the running example.
