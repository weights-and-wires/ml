---
title: "Ridge: a fee on large weights"
slug: 013-ridge-penalty-and-closed-form
part: "Part 1 — Foundations"
summary: A fee on squared weights stops a fit from leaning on noise, and because the score stays a smooth bowl, one linear solve finds its bottom.
minutes: 40
gpu: false
objectives:
  - Explain why a very large weight signals that a fit is chasing noise, and how a fee on squared weights discourages it.
  - Score a set of weights with the ridge objective, and write its gradient.
  - Solve for the bottom of the ridge bowl with one linear system, and say why the fee also fixes copied columns.
prereqs:
  - 004-bias-variance-noise
  - 005-least-squares-line
  - 012-momentum
exercise: 013-ridge-penalty-and-closed-form
---

# Ridge: a fee on large weights

You have two measurements. At input 0 the label is 0. At input 0.1 the label is 1. You fit a line through the origin, so there is one weight to find: the slope. The line that hits both points exactly has slope 10, and it predicts 10 at input 1. That is ten times the largest label you have ever seen.

Figure 1 shows what went wrong. The input is only 0.1, so the only way to reach a label of 1 from there is a huge slope. A large weight means a small change in the input swings the prediction by a lot. Two points cannot tell you whether the label of 1 is a real effect or noise. If it is noise, the slope of 10 is a disaster at input 1, and a flatter line such as slope 5 gives up half of that one label (it predicts 0.5 at input 0.1) to stay sensible everywhere else. When the data cannot tell you whether a big swing is real, you want the model to hold back.

This chapter builds the standard way to make a model hold back: charge a fee for large weights, and let the fit decide how much weight each unit of fee is worth. Without a fee, least squares chases every wobble in the labels. With one, it settles for a slightly worse fit on the training data in return for weights it can trust.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Left: two points and three lines through the origin with slopes 10, 5 and 0.91. Right: the fitted slope falls smoothly from 10 towards 0 as the fee grows.">

<line x1="45.0" y1="215.0" x2="235.0" y2="215.0" class="stroke-muted" />
<line x1="45.0" y1="215.0" x2="45.0" y2="62.0" class="stroke-muted" />
<line x1="45.0" y1="215.0" x2="45.0" y2="219.0" class="stroke-muted" />
<text x="45.0" y="231.0" class="fig-small" text-anchor="middle">0</text>
<line x1="140.0" y1="215.0" x2="140.0" y2="219.0" class="stroke-muted" />
<text x="140.0" y="231.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="235.0" y1="215.0" x2="235.0" y2="219.0" class="stroke-muted" />
<text x="235.0" y="231.0" class="fig-small" text-anchor="middle">1</text>
<line x1="45.0" y1="215.0" x2="235.0" y2="215.0" class="stroke-grid" />
<line x1="41.0" y1="215.0" x2="45.0" y2="215.0" class="stroke-muted" />
<text x="37.0" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="45.0" y1="138.5" x2="235.0" y2="138.5" class="stroke-grid" />
<line x1="41.0" y1="138.5" x2="45.0" y2="138.5" class="stroke-muted" />
<text x="37.0" y="142.5" class="fig-small" text-anchor="end">5</text>
<line x1="45.0" y1="62.0" x2="235.0" y2="62.0" class="stroke-grid" />
<line x1="41.0" y1="62.0" x2="45.0" y2="62.0" class="stroke-muted" />
<text x="37.0" y="66.0" class="fig-small" text-anchor="end">10</text>
<text x="140.0" y="250.0" class="fig-small" text-anchor="middle">input x</text>
<text x="45.0" y="22.0" class="fig-bold">Two points, three lines</text>
<text x="45.0" y="40.0" class="fig-small">points (0, 0) and (0.1, 1)</text>
<line x1="45.0" y1="215.0" x2="235.0" y2="62.0" class="stroke-bad" stroke-width="2.5"/>
<line x1="45.0" y1="215.0" x2="235.0" y2="138.5" class="stroke-b" stroke-width="2.5"/>
<line x1="45.0" y1="215.0" x2="235.0" y2="201.1" class="stroke-d" stroke-width="2.5"/>
<text x="241.0" y="66.0" class="fig-small">slope 10</text>
<text x="241.0" y="142.5" class="fig-small">slope 5</text>
<text x="241.0" y="205.1" class="fig-small">slope 0.91</text>
<circle cx="45.0" cy="215.0" r="5.5" class="fill-a"/>
<circle cx="64.0" cy="199.7" r="5.5" class="fill-a"/>
<line x1="350.0" y1="215.0" x2="570.0" y2="215.0" class="stroke-muted" />
<line x1="350.0" y1="215.0" x2="350.0" y2="62.0" class="stroke-muted" />
<line x1="350.0" y1="215.0" x2="350.0" y2="219.0" class="stroke-muted" />
<text x="350.0" y="231.0" class="fig-small" text-anchor="middle">0.001</text>
<line x1="405.0" y1="215.0" x2="405.0" y2="219.0" class="stroke-muted" />
<text x="405.0" y="231.0" class="fig-small" text-anchor="middle">0.01</text>
<line x1="460.0" y1="215.0" x2="460.0" y2="219.0" class="stroke-muted" />
<text x="460.0" y="231.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="515.0" y1="215.0" x2="515.0" y2="219.0" class="stroke-muted" />
<text x="515.0" y="231.0" class="fig-small" text-anchor="middle">1</text>
<line x1="570.0" y1="215.0" x2="570.0" y2="219.0" class="stroke-muted" />
<text x="570.0" y="231.0" class="fig-small" text-anchor="middle">10</text>
<line x1="350.0" y1="215.0" x2="570.0" y2="215.0" class="stroke-grid" />
<line x1="346.0" y1="215.0" x2="350.0" y2="215.0" class="stroke-muted" />
<text x="342.0" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="350.0" y1="138.5" x2="570.0" y2="138.5" class="stroke-grid" />
<line x1="346.0" y1="138.5" x2="350.0" y2="138.5" class="stroke-muted" />
<text x="342.0" y="142.5" class="fig-small" text-anchor="end">5</text>
<line x1="350.0" y1="62.0" x2="570.0" y2="62.0" class="stroke-grid" />
<line x1="346.0" y1="62.0" x2="350.0" y2="62.0" class="stroke-muted" />
<text x="342.0" y="66.0" class="fig-small" text-anchor="end">10</text>
<text x="460.0" y="250.0" class="fig-small" text-anchor="middle">fee λ (log scale)</text>
<text x="350.0" y="22.0" class="fig-bold">Fitted slope as the fee grows</text>
<polyline points="350.0,75.9 351.8,76.9 353.7,78.0 355.5,79.1 357.4,80.4 359.2,81.6 361.1,83.0 362.9,84.4 364.8,86.0 366.6,87.6 368.5,89.3 370.3,91.0 372.2,92.9 374.0,94.9 375.9,96.9 377.7,99.0 379.6,101.2 381.4,103.5 383.3,105.9 385.1,108.4 387.0,110.9 388.8,113.5 390.7,116.2 392.5,119.0 394.4,121.8 396.2,124.6 398.1,127.5 399.9,130.4 401.8,133.3 403.6,136.3 405.5,139.2 407.3,142.2 409.2,145.1 411.0,148.1 412.9,151.0 414.7,153.8 416.6,156.7 418.4,159.4 420.3,162.1 422.1,164.8 423.9,167.3 425.8,169.8 427.6,172.3 429.5,174.6 431.3,176.9 433.2,179.0 435.0,181.1 436.9,183.1 438.7,185.0 440.6,186.9 442.4,188.6 444.3,190.2 446.1,191.8 448.0,193.3 449.8,194.7 451.7,196.0 453.5,197.3 455.4,198.4 457.2,199.6 459.1,200.6 460.9,201.6 462.8,202.5 464.6,203.4 466.5,204.2 468.3,204.9 470.2,205.6 472.0,206.3 473.9,206.9 475.7,207.5 477.6,208.0 479.4,208.5 481.3,209.0 483.1,209.4 485.0,209.8 486.8,210.2 488.7,210.5 490.5,210.8 492.4,211.2 494.2,211.4 496.1,211.7 497.9,211.9 499.7,212.2 501.6,212.4 503.4,212.6 505.3,212.7 507.1,212.9 509.0,213.1 510.8,213.2 512.7,213.3 514.5,213.5 516.4,213.6 518.2,213.7 520.1,213.8 521.9,213.9 523.8,213.9 525.6,214.0 527.5,214.1 529.3,214.2 531.2,214.2 533.0,214.3 534.9,214.3 536.7,214.4 538.6,214.4 540.4,214.5 542.3,214.5 544.1,214.5 546.0,214.6 547.8,214.6 549.7,214.6 551.5,214.7 553.4,214.7 555.2,214.7 557.1,214.7 558.9,214.8 560.8,214.8 562.6,214.8 564.5,214.8 566.3,214.8 568.2,214.8 570.0,214.8" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="405.0" cy="138.5" r="4.5" class="fill-b"/>
<text x="415.0" y="130.5" class="fig-small">5</text>
<circle cx="460.0" cy="201.1" r="4.5" class="fill-b"/>
<text x="470.0" y="193.1" class="fig-small">0.91</text>
<circle cx="515.0" cy="213.5" r="4.5" class="fill-b"/>
<text x="515.0" y="203.5" class="fig-small" text-anchor="middle">0.099</text>
</svg>
Figure 1. Left: the exact fit (red) predicts 10 at input 1, and each larger fee tilts the line flatter (orange for fee 0.01, then purple for fee 0.1) while it misses the second point by more. Right: the fitted slope falls smoothly as the fee grows, so the fee works as a dial. At fee 0 the slope is 10, which a log axis cannot draw.
```

## What you already know

- Half the sum of squared misses scores a fit, as in the chapter *Fit a line by minimising squared misses*.
- A rule that matches the noise in its training labels does worse on data it has not seen, as in *Bias, variance, and noise*.
- The gradient points uphill, and at the bottom of a bowl it is zero, as in *Walking downhill*.
- The best least-squares weights solve $X^\top X\, w = X^\top y$.
- You solve a linear system with a solver rather than an inverse, because the solver is faster and loses fewer digits. The chapter *Why the textbook formula loses digits* showed how many.

## Why not cap the weights?

The first idea is a wall: allow no weight above 3. It stops the slope of 10, but it treats a weight of 2.9 and a weight of 0.1 the same, and it picks 3 for no reason the data gave. A fit that lands on the wall is stuck there, and a wall breaks the smooth bowl that the downhill methods of earlier chapters rely on.

A fee is softer. Every bit of weight costs something, a little at first and more as the weight grows. The data pulls the weight up to fit, the fee pulls it down, and the answer is wherever the two pulls balance. Nothing is cut off, so the score stays smooth.

## Add the fee to the score

Take the two-point example and charge $\tfrac{1}{2}\lambda w^2$, where $\lambda$ (lambda) is the fee rate. You choose it. Try $\lambda = 0.01$. The score of a slope $w$ is the half-squared misses plus the fee.

| Slope $w$ | Half-squared misses | Fee | Score |
|---|---|---|---|
| 0 | 0.5 | 0 | 0.5 |
| 1 | 0.405 | 0.005 | 0.41 |
| 5 | 0.125 | 0.125 | 0.25 |
| 10 | 0 | 0.5 | 0.5 |

The perfect fit at slope 10 is no longer the best. Slope 5 scores 0.25, half as much. Figure 2 shows why: the misses form a bowl with its bottom at 10, the fee forms a bowl with its bottom at 0, and the sum of two bowls is a bowl with its bottom in between.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Three curves against the slope w: the miss bowl with its bottom at 10, the fee bowl with its bottom at 0, and their sum with its bottom at 5.">

<line x1="60.0" y1="215.0" x2="560.0" y2="215.0" class="stroke-muted" />
<line x1="60.0" y1="215.0" x2="60.0" y2="36.0" class="stroke-muted" />
<line x1="60.0" y1="215.0" x2="60.0" y2="219.0" class="stroke-muted" />
<text x="60.0" y="231.0" class="fig-small" text-anchor="middle">0</text>
<line x1="143.3" y1="215.0" x2="143.3" y2="219.0" class="stroke-muted" />
<text x="143.3" y="231.0" class="fig-small" text-anchor="middle">2</text>
<line x1="226.7" y1="215.0" x2="226.7" y2="219.0" class="stroke-muted" />
<text x="226.7" y="231.0" class="fig-small" text-anchor="middle">4</text>
<line x1="268.3" y1="215.0" x2="268.3" y2="219.0" class="stroke-muted" />
<text x="268.3" y="231.0" class="fig-small" text-anchor="middle">5</text>
<line x1="310.0" y1="215.0" x2="310.0" y2="219.0" class="stroke-muted" />
<text x="310.0" y="231.0" class="fig-small" text-anchor="middle">6</text>
<line x1="393.3" y1="215.0" x2="393.3" y2="219.0" class="stroke-muted" />
<text x="393.3" y="231.0" class="fig-small" text-anchor="middle">8</text>
<line x1="476.7" y1="215.0" x2="476.7" y2="219.0" class="stroke-muted" />
<text x="476.7" y="231.0" class="fig-small" text-anchor="middle">10</text>
<line x1="560.0" y1="215.0" x2="560.0" y2="219.0" class="stroke-muted" />
<text x="560.0" y="231.0" class="fig-small" text-anchor="middle">12</text>
<line x1="60.0" y1="215.0" x2="560.0" y2="215.0" class="stroke-grid" />
<line x1="56.0" y1="215.0" x2="60.0" y2="215.0" class="stroke-muted" />
<text x="52.0" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="60.0" y1="155.3" x2="560.0" y2="155.3" class="stroke-grid" />
<line x1="56.0" y1="155.3" x2="60.0" y2="155.3" class="stroke-muted" />
<text x="52.0" y="159.3" class="fig-small" text-anchor="end">0.25</text>
<line x1="60.0" y1="95.7" x2="560.0" y2="95.7" class="stroke-grid" />
<line x1="56.0" y1="95.7" x2="60.0" y2="95.7" class="stroke-muted" />
<text x="52.0" y="99.7" class="fig-small" text-anchor="end">0.5</text>
<line x1="60.0" y1="36.0" x2="560.0" y2="36.0" class="stroke-grid" />
<line x1="56.0" y1="36.0" x2="60.0" y2="36.0" class="stroke-muted" />
<text x="52.0" y="40.0" class="fig-small" text-anchor="end">0.75</text>
<text x="310.0" y="250.0" class="fig-small" text-anchor="middle">slope w</text>
<polyline points="60.0,95.7 62.5,97.1 65.0,98.5 67.5,99.9 70.1,101.4 72.6,102.8 75.1,104.1 77.6,105.5 80.1,106.9 82.6,108.3 85.1,109.6 87.6,111.0 90.2,112.3 92.7,113.6 95.2,115.0 97.7,116.3 100.2,117.6 102.7,118.9 105.2,120.2 107.7,121.4 110.3,122.7 112.8,124.0 115.3,125.2 117.8,126.5 120.3,127.7 122.8,128.9 125.3,130.2 127.8,131.4 130.4,132.6 132.9,133.8 135.4,134.9 137.9,136.1 140.4,137.3 142.9,138.4 145.4,139.6 147.9,140.7 150.5,141.9 153.0,143.0 155.5,144.1 158.0,145.2 160.5,146.3 163.0,147.4 165.5,148.5 168.0,149.5 170.6,150.6 173.1,151.6 175.6,152.7 178.1,153.7 180.6,154.8 183.1,155.8 185.6,156.8 188.1,157.8 190.7,158.8 193.2,159.8 195.7,160.7 198.2,161.7 200.7,162.7 203.2,163.6 205.7,164.5 208.2,165.5 210.8,166.4 213.3,167.3 215.8,168.2 218.3,169.1 220.8,170.0 223.3,170.9 225.8,171.8 228.3,172.6 230.9,173.5 233.4,174.3 235.9,175.1 238.4,176.0 240.9,176.8 243.4,177.6 245.9,178.4 248.4,179.2 251.0,180.0 253.5,180.8 256.0,181.5 258.5,182.3 261.0,183.0 263.5,183.8 266.0,184.5 268.5,185.2 271.1,185.9 273.6,186.6 276.1,187.3 278.6,188.0 281.1,188.7 283.6,189.4 286.1,190.0 288.6,190.7 291.2,191.3 293.7,192.0 296.2,192.6 298.7,193.2 301.2,193.8 303.7,194.4 306.2,195.0 308.7,195.6 311.3,196.2 313.8,196.8 316.3,197.3 318.8,197.9 321.3,198.4 323.8,198.9 326.3,199.5 328.8,200.0 331.4,200.5 333.9,201.0 336.4,201.5 338.9,202.0 341.4,202.4 343.9,202.9 346.4,203.3 348.9,203.8 351.5,204.2 354.0,204.7 356.5,205.1 359.0,205.5 361.5,205.9 364.0,206.3 366.5,206.7 369.0,207.0 371.6,207.4 374.1,207.8 376.6,208.1 379.1,208.5 381.6,208.8 384.1,209.1 386.6,209.4 389.1,209.7 391.7,210.0 394.2,210.3 396.7,210.6 399.2,210.9 401.7,211.1 404.2,211.4 406.7,211.6 409.2,211.9 411.8,212.1 414.3,212.3 416.8,212.5 419.3,212.7 421.8,212.9 424.3,213.1 426.8,213.3 429.3,213.5 431.9,213.6 434.4,213.8 436.9,213.9 439.4,214.0 441.9,214.2 444.4,214.3 446.9,214.4 449.4,214.5 452.0,214.6 454.5,214.7 457.0,214.7 459.5,214.8 462.0,214.9 464.5,214.9 467.0,214.9 469.5,215.0 472.1,215.0 474.6,215.0 477.1,215.0 479.6,215.0 482.1,215.0 484.6,215.0 487.1,214.9 489.6,214.9 492.2,214.8 494.7,214.8 497.2,214.7 499.7,214.6 502.2,214.6 504.7,214.5 507.2,214.4 509.7,214.2 512.3,214.1 514.8,214.0 517.3,213.9 519.8,213.7 522.3,213.6 524.8,213.4 527.3,213.2 529.8,213.1 532.4,212.9 534.9,212.7 537.4,212.5 539.9,212.3 542.4,212.0 544.9,211.8 547.4,211.6 549.9,211.3 552.5,211.1 555.0,210.8 557.5,210.5 560.0,210.2" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="60.0,215.0 62.5,215.0 65.0,215.0 67.5,215.0 70.1,214.9 72.6,214.9 75.1,214.8 77.6,214.8 80.1,214.7 82.6,214.6 85.1,214.6 87.6,214.5 90.2,214.4 92.7,214.3 95.2,214.1 97.7,214.0 100.2,213.9 102.7,213.7 105.2,213.6 107.7,213.4 110.3,213.3 112.8,213.1 115.3,212.9 117.8,212.7 120.3,212.5 122.8,212.3 125.3,212.1 127.8,211.8 130.4,211.6 132.9,211.4 135.4,211.1 137.9,210.8 140.4,210.6 142.9,210.3 145.4,210.0 147.9,209.7 150.5,209.4 153.0,209.1 155.5,208.7 158.0,208.4 160.5,208.1 163.0,207.7 165.5,207.3 168.0,207.0 170.6,206.6 173.1,206.2 175.6,205.8 178.1,205.4 180.6,205.0 183.1,204.6 185.6,204.2 188.1,203.7 190.7,203.3 193.2,202.8 195.7,202.3 198.2,201.9 200.7,201.4 203.2,200.9 205.7,200.4 208.2,199.9 210.8,199.4 213.3,198.9 215.8,198.3 218.3,197.8 220.8,197.2 223.3,196.7 225.8,196.1 228.3,195.5 230.9,194.9 233.4,194.3 235.9,193.7 238.4,193.1 240.9,192.5 243.4,191.9 245.9,191.2 248.4,190.6 251.0,189.9 253.5,189.3 256.0,188.6 258.5,187.9 261.0,187.2 263.5,186.5 266.0,185.8 268.5,185.1 271.1,184.4 273.6,183.6 276.1,182.9 278.6,182.2 281.1,181.4 283.6,180.6 286.1,179.9 288.6,179.1 291.2,178.3 293.7,177.5 296.2,176.7 298.7,175.8 301.2,175.0 303.7,174.2 306.2,173.3 308.7,172.5 311.3,171.6 313.8,170.7 316.3,169.9 318.8,169.0 321.3,168.1 323.8,167.2 326.3,166.2 328.8,165.3 331.4,164.4 333.9,163.4 336.4,162.5 338.9,161.5 341.4,160.6 343.9,159.6 346.4,158.6 348.9,157.6 351.5,156.6 354.0,155.6 356.5,154.6 359.0,153.6 361.5,152.5 364.0,151.5 366.5,150.4 369.0,149.4 371.6,148.3 374.1,147.2 376.6,146.1 379.1,145.0 381.6,143.9 384.1,142.8 386.6,141.7 389.1,140.5 391.7,139.4 394.2,138.2 396.7,137.1 399.2,135.9 401.7,134.7 404.2,133.6 406.7,132.4 409.2,131.2 411.8,130.0 414.3,128.7 416.8,127.5 419.3,126.3 421.8,125.0 424.3,123.8 426.8,122.5 429.3,121.2 431.9,120.0 434.4,118.7 436.9,117.4 439.4,116.1 441.9,114.7 444.4,113.4 446.9,112.1 449.4,110.7 452.0,109.4 454.5,108.0 457.0,106.7 459.5,105.3 462.0,103.9 464.5,102.5 467.0,101.1 469.5,99.7 472.1,98.3 474.6,96.9 477.1,95.4 479.6,94.0 482.1,92.5 484.6,91.1 487.1,89.6 489.6,88.1 492.2,86.6 494.7,85.1 497.2,83.6 499.7,82.1 502.2,80.6 504.7,79.1 507.2,77.5 509.7,76.0 512.3,74.4 514.8,72.8 517.3,71.3 519.8,69.7 522.3,68.1 524.8,66.5 527.3,64.9 529.8,63.3 532.4,61.6 534.9,60.0 537.4,58.4 539.9,56.7 542.4,55.0 544.9,53.4 547.4,51.7 549.9,50.0 552.5,48.3 555.0,46.6 557.5,44.9 560.0,43.2" fill="none" class="stroke-d" stroke-width="2.5"/>
<polyline points="60.0,95.7 62.5,97.1 65.0,98.5 67.5,99.9 70.1,101.3 72.6,102.6 75.1,104.0 77.6,105.3 80.1,106.6 82.6,107.9 85.1,109.2 87.6,110.4 90.2,111.7 92.7,112.9 95.2,114.1 97.7,115.3 100.2,116.5 102.7,117.6 105.2,118.8 107.7,119.9 110.3,121.0 112.8,122.1 115.3,123.1 117.8,124.2 120.3,125.2 122.8,126.2 125.3,127.2 127.8,128.2 130.4,129.2 132.9,130.1 135.4,131.0 137.9,131.9 140.4,132.8 142.9,133.7 145.4,134.6 147.9,135.4 150.5,136.2 153.0,137.0 155.5,137.8 158.0,138.6 160.5,139.3 163.0,140.1 165.5,140.8 168.0,141.5 170.6,142.2 173.1,142.9 175.6,143.5 178.1,144.1 180.6,144.8 183.1,145.4 185.6,145.9 188.1,146.5 190.7,147.0 193.2,147.6 195.7,148.1 198.2,148.6 200.7,149.0 203.2,149.5 205.7,149.9 208.2,150.4 210.8,150.8 213.3,151.2 215.8,151.5 218.3,151.9 220.8,152.2 223.3,152.5 225.8,152.8 228.3,153.1 230.9,153.4 233.4,153.7 235.9,153.9 238.4,154.1 240.9,154.3 243.4,154.5 245.9,154.6 248.4,154.8 251.0,154.9 253.5,155.0 256.0,155.1 258.5,155.2 261.0,155.3 263.5,155.3 266.0,155.3 268.5,155.3 271.1,155.3 273.6,155.3 276.1,155.3 278.6,155.2 281.1,155.1 283.6,155.0 286.1,154.9 288.6,154.8 291.2,154.6 293.7,154.5 296.2,154.3 298.7,154.1 301.2,153.8 303.7,153.6 306.2,153.4 308.7,153.1 311.3,152.8 313.8,152.5 316.3,152.2 318.8,151.8 321.3,151.5 323.8,151.1 326.3,150.7 328.8,150.3 331.4,149.9 333.9,149.4 336.4,149.0 338.9,148.5 341.4,148.0 343.9,147.5 346.4,146.9 348.9,146.4 351.5,145.8 354.0,145.3 356.5,144.7 359.0,144.0 361.5,143.4 364.0,142.7 366.5,142.1 369.0,141.4 371.6,140.7 374.1,140.0 376.6,139.2 379.1,138.5 381.6,137.7 384.1,136.9 386.6,136.1 389.1,135.3 391.7,134.4 394.2,133.6 396.7,132.7 399.2,131.8 401.7,130.9 404.2,129.9 406.7,129.0 409.2,128.0 411.8,127.1 414.3,126.1 416.8,125.0 419.3,124.0 421.8,123.0 424.3,121.9 426.8,120.8 429.3,119.7 431.9,118.6 434.4,117.4 436.9,116.3 439.4,115.1 441.9,113.9 444.4,112.7 446.9,111.5 449.4,110.2 452.0,109.0 454.5,107.7 457.0,106.4 459.5,105.1 462.0,103.8 464.5,102.4 467.0,101.1 469.5,99.7 472.1,98.3 474.6,96.9 477.1,95.4 479.6,94.0 482.1,92.5 484.6,91.0 487.1,89.5 489.6,88.0 492.2,86.5 494.7,84.9 497.2,83.3 499.7,81.7 502.2,80.1 504.7,78.5 507.2,76.9 509.7,75.2 512.3,73.5 514.8,71.8 517.3,70.1 519.8,68.4 522.3,66.7 524.8,64.9 527.3,63.1 529.8,61.3 532.4,59.5 534.9,57.7 537.4,55.8 539.9,54.0 542.4,52.1 544.9,50.2 547.4,48.2 549.9,46.3 552.5,44.4 555.0,42.4 557.5,40.4 560.0,38.4" fill="none" class="stroke-b" stroke-width="3"/>
<line x1="150.0" y1="14.0" x2="178.0" y2="14.0" class="stroke-a" stroke-width="3"/>
<text x="184.0" y="18.0" class="fig-small">misses</text>
<line x1="300.0" y1="14.0" x2="328.0" y2="14.0" class="stroke-d" stroke-width="3"/>
<text x="334.0" y="18.0" class="fig-small">fee, 0.005 w²</text>
<line x1="450.0" y1="14.0" x2="478.0" y2="14.0" class="stroke-b" stroke-width="3"/>
<text x="484.0" y="18.0" class="fig-small">misses + fee</text>
<circle cx="476.7" cy="95.7" r="4.5" class="fill-bad"/>
<text x="470.7" y="85.7" class="fig-small" text-anchor="end">w = 10: sum 0.5</text>
<circle cx="476.7" cy="215.0" r="5" class="fill-a"/>
<text x="476.7" y="205.0" class="fig-small" text-anchor="middle">misses alone bottom out</text>
<circle cx="268.3" cy="155.3" r="6" class="fill-b"/>
<line x1="268.3" y1="155.3" x2="268.3" y2="215.0" class="stroke-b dashed" />
<text x="278.3" y="177.3" class="fig-small">bottom of the sum: w = 5, score 0.25</text>
</svg>
Figure 2. The misses (blue) bottom out at slope 10, the fee (purple) at slope 0. Their sum (orange) bottoms out between them, at slope 5, where the two pulls balance.
```

You can find that bottom by hand. The slope of the score is $0.1(0.1w - 1) + \lambda w$. Set it to zero and collect the terms: $(0.01 + \lambda)\,w = 0.1$. So the best slope is $0.1 / (0.01 + \lambda)$.

| Fee $\lambda$ | Best slope |
|---|---|
| 0 | 10 |
| 0.01 | 5 |
| 0.1 | 0.91 |
| 1 | 0.099 |

This is the right panel of Figure 1. The fee is a dial: at 0 you get plain least squares, and as it grows the weight shrinks smoothly towards 0. No setting is correct in general. You choose it by holding data back and keeping the fee whose fit scores best on that data.

## Read the ridge score aloud

With many weights, the fee charges the squared length of the weight vector $w$. The whole score is

$$
L(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2
$$

Read it as: half the squared misses, plus half of lambda times the squared length of the weights. The first half is the least-squares score from earlier chapters. Both halves are there on purpose. When you take the slope, the 2 in each square cancels its half, which leaves a gradient with no stray factors. Charging the *square* of each weight matters too, because it keeps the score a smooth bowl.

Check it on the smallest two-weight case, where $X$ is the identity, so the predictions are the weights, and $y = (3, 1)$. Take $\lambda = 1$. The weights $(3, 1)$ miss by 0, and their fee is $\tfrac12(9 + 1) = 5$, so they score 5. The weights $(1.5, 0.5)$ miss by $-1.5$ and $-0.5$, which costs $\tfrac12(2.25 + 0.25) = 1.25$, and their fee is $\tfrac12(2.25 + 0.25) = 1.25$. They score 2.5, half as much.

```python
misses = X @ w - y                       # shape (n,)
score = 0.5 * np.sum(misses ** 2) + 0.5 * lam * np.dot(w, w)
```

## Find the bottom with one linear solve

The gradient of the first half is $X^\top(Xw - y)$. The gradient of the fee is $\lambda w$, one pull per weight, proportional to the weight itself. So

$$
\nabla L(w) = X^\top(Xw - y) + \lambda w
$$

This is the data's pull plus the fee's pull. At the bottom of a bowl the gradient is zero, and because the score is a bowl there is exactly one such point. Set it to zero, $X^\top X w - X^\top y + \lambda w = 0$, and collect the terms that multiply $w$. You get one linear system:

$$
(X^\top X + \lambda I)\, w = X^\top y
$$

Read it as: the least-squares equations, with $\lambda$ added to the diagonal. That addition is why the system is always solvable. $X^\top X$ can be singular, but its eigenvalues are never negative. Adding $\lambda$ to the diagonal adds $\lambda$ to every eigenvalue, so all of them are at least $\lambda > 0$, and a matrix with no zero eigenvalue has an inverse. Check it on the one-weight example, where $X^\top X = 0.01$ and $X^\top y = 0.1$. The system is $(0.01 + \lambda)\,w = 0.1$, the equation you found above. On the identity example, $(I + I)\,w = (3, 1)$ gives $w = (1.5, 0.5)$, the better score from a moment ago.

```python
d = X.shape[1]
w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)   # no np.linalg.inv
```

## What the fee does to copied columns

Suppose two columns of $X$ are identical, say the same measurement recorded twice. Take $X$ with rows $(1, 1)$ and $(2, 2)$, and labels $(1, 2)$. Then $X^\top X$ is the matrix with every entry 5. Any weights with $w_1 + w_2 = 1$ fit the labels exactly, so least squares has infinitely many answers (the blue line in Figure 3), and the system has no unique solution.

Add $\lambda = 1$ to the diagonal and the system becomes $\begin{bmatrix} 6 & 5 \\ 5 & 6 \end{bmatrix} w = (5, 5)$. It has one answer, $w = (5/11, 5/11) \approx (0.4545, 0.4545)$. The fee breaks the tie by preferring the pair of weights with the smallest squared length, and by symmetry that is the even split. As $\lambda$ shrinks, the answer tends towards $(0.5, 0.5)$, the exact fit nearest to 0.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="The plane of two weights for two identical columns: a blue line of exactly fitting weights, a purple fee circle touching it at the even split, and the ridge answer on the diagonal.">

<line x1="40.0" y1="157.4" x2="270.0" y2="157.4" class="stroke-muted" />
<line x1="127.6" y1="245.0" x2="127.6" y2="15.0" class="stroke-muted" />
<text x="270.0" y="173.4" class="fig-small" text-anchor="end">w1</text>
<text x="133.6" y="25.0" class="fig-small">w2</text>
<line x1="29.0" y1="15.0" x2="259.0" y2="245.0" class="stroke-grid" />
<line x1="160.5" y1="15.0" x2="390.5" y2="245.0" class="stroke-grid" />
<line x1="94.8" y1="15.0" x2="324.8" y2="245.0" class="stroke-a" stroke-width="3"/>
<polyline points="205.1,157.4 205.0,153.3 204.6,149.2 204.1,145.2 203.3,141.1 202.4,137.2 201.2,133.3 199.8,129.4 198.3,125.6 196.5,121.9 194.5,118.4 192.4,114.9 190.0,111.5 187.5,108.3 184.9,105.2 182.0,102.3 179.0,99.5 175.9,96.8 172.6,94.4 169.3,92.1 165.7,90.0 162.1,88.1 158.4,86.3 154.6,84.8 150.8,83.5 146.8,82.4 142.9,81.4 138.8,80.8 134.8,80.3 130.7,80.0 126.6,79.9 122.5,80.1 118.4,80.5 114.4,81.1 110.4,81.9 106.4,82.9 102.5,84.1 98.7,85.5 94.9,87.2 91.3,89.0 87.7,91.0 84.3,93.2 80.9,95.6 77.8,98.1 74.7,100.8 71.8,103.7 69.0,106.7 66.4,109.9 64.0,113.2 61.8,116.6 59.7,120.1 57.8,123.8 56.2,127.5 54.7,131.3 53.4,135.2 52.3,139.2 51.5,143.2 50.8,147.2 50.4,151.3 50.2,155.3 50.2,159.4 50.4,163.5 50.8,167.6 51.5,171.6 52.3,175.6 53.4,179.6 54.7,183.4 56.2,187.3 57.8,191.0 59.7,194.6 61.8,198.2 64.0,201.6 66.4,204.9 69.0,208.0 71.8,211.0 74.7,213.9 77.8,216.6 80.9,219.2 84.3,221.6 87.7,223.8 91.3,225.8 94.9,227.6 98.7,229.2 102.5,230.6 106.4,231.9 110.4,232.9 114.4,233.7 118.4,234.3 122.5,234.7 126.6,234.8 130.7,234.8 134.8,234.5 138.8,234.0 142.9,233.3 146.8,232.4 150.8,231.3 154.6,230.0 158.4,228.4 162.1,226.7 165.7,224.8 169.3,222.7 172.6,220.4 175.9,217.9 179.0,215.3 182.0,212.5 184.9,209.6 187.5,206.5 190.0,203.2 192.4,199.9 194.5,196.4 196.5,192.8 198.3,189.1 199.8,185.4 201.2,181.5 202.4,177.6 203.3,173.6 204.1,169.6 204.6,165.5 205.0,161.5 205.1,157.4" fill="none" class="stroke-d dashed" />
<circle cx="182.4" cy="102.6" r="4" class="fill-d"/>
<circle cx="177.4" cy="107.6" r="5" class="fill-b"/>
<circle cx="127.6" cy="157.4" r="3.5" class="fill-ink"/>
<text x="119.6" y="173.4" class="fig-small" text-anchor="end">0</text>
<text x="300.0" y="84.0" class="fig-small">Blue line: every point on it fits both labels exactly,</text>
<text x="300.0" y="102.0" class="fig-small">so least squares has no single answer.</text>
<text x="300.0" y="134.0" class="fig-small">Purple circle: all weights with the same fee, radius 0.71.</text>
<text x="300.0" y="152.0" class="fig-small">It first touches the blue line at (0.5, 0.5).</text>
<text x="300.0" y="184.0" class="fig-small">Orange dot: ridge at λ = 1, at (0.4545, 0.4545),</text>
<text x="300.0" y="202.0" class="fig-small">a little closer to 0 than the purple touch point.</text>
</svg>
Figure 3. With two identical columns, every weight pair on the blue line fits perfectly. The fee prefers the pair nearest 0, so ridge lands on the diagonal, at the even split.
```

> [!KEY]
> A model leans on large weights when the data cannot tell signal from noise, like the slope of 10 from two points. Ridge adds half of lambda times the squared weights to the score, so the fit must pay for every bit of lean. The score is still a smooth bowl, which means its bottom is where the gradient $X^\top(Xw - y) + \lambda w$ is zero. That is one linear system, and lambda is the dial between fitting the data and staying small.

> [!RECAP]
> - A large weight lets a tiny change in the input swing the prediction, so it is a warning sign when data is thin or noisy.
> - Ridge scores a fit as half the squared misses plus half of lambda times the squared weights. On the identity example, $(1.5, 0.5)$ scores 2.5 and $(3, 1)$ scores 5.
> - The gradient is $X^\top(Xw - y) + \lambda w$, and setting it to zero gives $(X^\top X + \lambda I)\, w = X^\top y$.
> - You solve that system with `np.linalg.solve`, never with an inverse.
> - Adding lambda to the diagonal makes the system solvable even for copied columns, and splits the weight evenly between them.

## Build it step by step

You write three functions, in the order the checks test them.

1. `ridge_objective(X, y, w, lam)` returns a float. Compute the misses `X @ w - y`, then half their squared sum plus `0.5 * lam * np.dot(w, w)`. The most common mistake is to drop the half on the fee, which turns 2.5 into 3.75 on the identity example.

2. `ridge_gradient(X, y, w, lam)` returns an array of shape `(d,)`. It is `X.T @ misses + lam * w`. The common mistake is to write `lam * w.sum()`, a single number, where each weight needs its own pull. The checks compare your answer with finite differences of `ridge_objective`, so the formula on the page and the code have to agree.

3. `ridge_closed_form(X, y, lam)` returns an array of shape `(d,)`. Build `X.T @ X + lam * np.eye(d)` and hand it to `np.linalg.solve` with `X.T @ y`. The common mistake is to call `np.linalg.inv`, which does more work and keeps fewer digits for the same answer.

## Where this breaks

- **Columns on different scales.** The fee charges every weight in the same currency, so a column measured in tiny units needs a huge weight and is punished for it. Divide each column by its spread before you fit, so each weight means the same thing.
- **A fee on the intercept.** A constant offset in the labels should not be shrunk, because shifting every label by the same amount would then change the fit. Subtract the mean from $y$ and from each column first, or leave the bias out of the fee.
- **Choosing lambda.** Too small and the fit still chases noise. Too large and every weight collapses towards 0. Fit on training data for a grid of lambdas such as 0.001, 0.01, 0.1, 1, and 10, score each fit on held-out data, and keep the lambda that scores best.

## The exercise

You build `ridge_objective`, `ridge_gradient`, and `ridge_closed_form`. The checks confirm the hand value 2.5 at $(1.5, 0.5)$, compare your gradient with finite differences, and confirm that your solution has zero gradient and matches least squares on stacked data. They also check that two identical columns get the even split $(5/11, 5/11)$, and that the weight length falls as lambda rises.
