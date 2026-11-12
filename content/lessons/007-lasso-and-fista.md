---
title: "The lasso: exact zeros, soft thresholding, and FISTA"
slug: 007-lasso-and-fista
part: "Part 1 — Foundations"
summary: A fee on absolute weights zeroes weak features exactly. Soft thresholding solves one number, and ISTA and FISTA fit any design.
minutes: 75
gpu: false
objectives:
  - Explain why a fee on squared weights shrinks them but never zeroes them, and why a fee on absolute weights does.
  - Build soft thresholding, and solve the lasso exactly when the design is the identity.
  - Build ISTA, which steps on the squared error and then soft-thresholds, and FISTA, which adds momentum.
  - Test whether a set of weights is a lasso answer with the KKT rules.
prereqs:
  - 005-gradient-descent-and-momentum
  - 006-ridge-regression
exercise: 007-lasso-and-fista
---

# The lasso: exact zeros, soft thresholding, and FISTA

You have 40 features and you suspect only 5 of them matter. You fit ridge. It shrinks all 40 weights, and none of them reaches zero. You are left with 40 small but nonzero weights. You cannot say which features matter, and every prediction still needs all 40 measurements.

Rounding the small ridge weights down to 0 afterwards would use an arbitrary cut-off, and the surviving weights were fit on the assumption that the others were present. A better plan is to change the fee itself, so that a feature with a weak signal ends up at exactly 0 as part of the fit. That fee is the absolute size of each weight, and the model it produces is the *lasso*.

The change raises a second problem. For the identity design the lasso has a one-move solution, but for a real data matrix there is no formula, because the fee has a corner at 0. You need a method that walks to the bottom, lands on exact zeros, and tells you when it has arrived. This chapter builds both the fee and the walk.

## What you already know

- Ridge charges $\tfrac12\lambda w^2$ per weight, and its answer sits where the data's pull and the fee's pull balance, as in [ridge regression](/lessons/006-ridge-regression). The pull of a fee on a weight is its slope.
- When the design is the identity, the predictions are the weights, so each miss is $w_i - y_i$.
- The absolute value $|w|$ is the distance from 0. Its slope is $-1$ to the left of 0 and $+1$ to the right, so its graph has a corner at 0.
- A downhill step must stay under $2/L$, where $L$ is the largest curvature, and for the squared error $L$ is the largest eigenvalue of $X^\top X$. A long narrow valley is slow, and momentum speeds up a walk along it, as in [gradient descent and momentum](/lessons/005-gradient-descent-and-momentum).

## Why ridge cannot reach zero

Take a single weight. The data says the weight should be $x = 0.5$, so the squared miss is $\tfrac12(w - 0.5)^2$. Charge a fee with $\lambda = 1$. The ridge fee $\tfrac12 w^2$ has slope $w$, which is tiny for small $w$, so a weight near 0 is nearly free to keep and stays a little above 0. The lasso fee $|w|$ has slope 1 at every positive $w$, however close to 0. Every unit of weight costs the same all the way down.

| Weight $w$ | Ridge score | Lasso score |
|---|---|---|
| 0 | 0.125 | 0.125 |
| 0.1 | 0.085 | 0.18 |
| 0.25 | 0.0625 | 0.28125 |

Ridge improves as the weight leaves 0, and its best weight is 0.25, half of the data's 0.5. The lasso score gets worse at once, so its best weight is exactly 0. Figure 1 shows the tug of war: the data's pull on this weight never exceeds 0.5, and the lasso fee pulls back with 1.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Left: the ridge fee w squared over two and the lasso fee absolute w from minus 2 to 2. Right: the data's pull falls from 0.5 to 0, the ridge pull rises as w, and the lasso pull stays at 1.">

<line x1="45.0" y1="215.0" x2="250.0" y2="215.0" class="stroke-muted" />
<line x1="45.0" y1="215.0" x2="45.0" y2="50.0" class="stroke-muted" />
<line x1="45.0" y1="215.0" x2="45.0" y2="219.0" class="stroke-muted" />
<text x="45.0" y="231.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="96.2" y1="215.0" x2="96.2" y2="219.0" class="stroke-muted" />
<text x="96.2" y="231.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="147.5" y1="215.0" x2="147.5" y2="219.0" class="stroke-muted" />
<text x="147.5" y="231.0" class="fig-small" text-anchor="middle">0</text>
<line x1="198.8" y1="215.0" x2="198.8" y2="219.0" class="stroke-muted" />
<text x="198.8" y="231.0" class="fig-small" text-anchor="middle">1</text>
<line x1="250.0" y1="215.0" x2="250.0" y2="219.0" class="stroke-muted" />
<text x="250.0" y="231.0" class="fig-small" text-anchor="middle">2</text>
<line x1="45.0" y1="215.0" x2="250.0" y2="215.0" class="stroke-grid" />
<line x1="41.0" y1="215.0" x2="45.0" y2="215.0" class="stroke-muted" />
<text x="37.0" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="45.0" y1="132.5" x2="250.0" y2="132.5" class="stroke-grid" />
<line x1="41.0" y1="132.5" x2="45.0" y2="132.5" class="stroke-muted" />
<text x="37.0" y="136.5" class="fig-small" text-anchor="end">1</text>
<line x1="45.0" y1="50.0" x2="250.0" y2="50.0" class="stroke-grid" />
<line x1="41.0" y1="50.0" x2="45.0" y2="50.0" class="stroke-muted" />
<text x="37.0" y="54.0" class="fig-small" text-anchor="end">2</text>
<text x="147.0" y="250.0" class="fig-small" text-anchor="middle">weight w</text>
<text x="45.0" y="24.0" class="fig-bold">What each fee costs</text>
<polyline points="45.0,50.0 46.0,53.3 47.1,56.6 48.1,59.8 49.1,63.0 50.2,66.2 51.2,69.3 52.2,72.4 53.2,75.5 54.3,78.5 55.3,81.5 56.3,84.5 57.4,87.4 58.4,90.3 59.4,93.2 60.5,96.0 61.5,98.8 62.5,101.6 63.5,104.3 64.6,107.0 65.6,109.7 66.6,112.3 67.7,114.9 68.7,117.5 69.7,120.0 70.8,122.5 71.8,125.0 72.8,127.4 73.8,129.8 74.9,132.2 75.9,134.5 76.9,136.8 78.0,139.1 79.0,141.3 80.0,143.5 81.1,145.7 82.1,147.8 83.1,149.9 84.1,152.0 85.2,154.0 86.2,156.0 87.2,158.0 88.3,159.9 89.3,161.8 90.3,163.7 91.4,165.5 92.4,167.3 93.4,169.1 94.4,170.8 95.5,172.5 96.5,174.2 97.5,175.8 98.6,177.4 99.6,179.0 100.6,180.5 101.7,182.0 102.7,183.5 103.7,184.9 104.7,186.3 105.8,187.7 106.8,189.0 107.8,190.3 108.9,191.6 109.9,192.8 110.9,194.0 112.0,195.2 113.0,196.3 114.0,197.4 115.1,198.5 116.1,199.5 117.1,200.5 118.1,201.5 119.2,202.4 120.2,203.3 121.2,204.2 122.3,205.0 123.3,205.8 124.3,206.6 125.4,207.3 126.4,208.0 127.4,208.7 128.4,209.3 129.5,209.9 130.5,210.5 131.5,211.0 132.6,211.5 133.6,212.0 134.6,212.4 135.7,212.8 136.7,213.2 137.7,213.5 138.7,213.8 139.8,214.1 140.8,214.3 141.8,214.5 142.9,214.7 143.9,214.8 144.9,214.9 146.0,215.0 147.0,215.0 148.0,215.0 149.0,215.0 150.1,214.9 151.1,214.8 152.1,214.7 153.2,214.5 154.2,214.3 155.2,214.1 156.3,213.8 157.3,213.5 158.3,213.2 159.3,212.8 160.4,212.4 161.4,212.0 162.4,211.5 163.5,211.0 164.5,210.5 165.5,209.9 166.6,209.3 167.6,208.7 168.6,208.0 169.6,207.3 170.7,206.6 171.7,205.8 172.7,205.0 173.8,204.2 174.8,203.3 175.8,202.4 176.9,201.5 177.9,200.5 178.9,199.5 179.9,198.5 181.0,197.4 182.0,196.3 183.0,195.2 184.1,194.0 185.1,192.8 186.1,191.6 187.2,190.3 188.2,189.0 189.2,187.7 190.3,186.3 191.3,184.9 192.3,183.5 193.3,182.0 194.4,180.5 195.4,179.0 196.4,177.4 197.5,175.8 198.5,174.2 199.5,172.5 200.6,170.8 201.6,169.1 202.6,167.3 203.6,165.5 204.7,163.7 205.7,161.8 206.7,159.9 207.8,158.0 208.8,156.0 209.8,154.0 210.9,152.0 211.9,149.9 212.9,147.8 213.9,145.7 215.0,143.5 216.0,141.3 217.0,139.1 218.1,136.8 219.1,134.5 220.1,132.2 221.2,129.8 222.2,127.4 223.2,125.0 224.2,122.5 225.3,120.0 226.3,117.5 227.3,114.9 228.4,112.3 229.4,109.7 230.4,107.0 231.5,104.3 232.5,101.6 233.5,98.8 234.5,96.0 235.6,93.2 236.6,90.3 237.6,87.4 238.7,84.5 239.7,81.5 240.7,78.5 241.8,75.5 242.8,72.4 243.8,69.3 244.8,66.2 245.9,63.0 246.9,59.8 247.9,56.6 249.0,53.3 250.0,50.0" fill="none" class="stroke-d" stroke-width="3"/>
<polyline points="45.0,50.0 46.0,51.7 47.1,53.3 48.1,55.0 49.1,56.6 50.2,58.3 51.2,59.9 52.2,61.6 53.2,63.3 54.3,64.9 55.3,66.6 56.3,68.2 57.4,69.9 58.4,71.6 59.4,73.2 60.5,74.9 61.5,76.5 62.5,78.2 63.5,79.8 64.6,81.5 65.6,83.2 66.6,84.8 67.7,86.5 68.7,88.1 69.7,89.8 70.8,91.5 71.8,93.1 72.8,94.8 73.8,96.4 74.9,98.1 75.9,99.7 76.9,101.4 78.0,103.1 79.0,104.7 80.0,106.4 81.1,108.0 82.1,109.7 83.1,111.4 84.1,113.0 85.2,114.7 86.2,116.3 87.2,118.0 88.3,119.6 89.3,121.3 90.3,123.0 91.4,124.6 92.4,126.3 93.4,127.9 94.4,129.6 95.5,131.3 96.5,132.9 97.5,134.6 98.6,136.2 99.6,137.9 100.6,139.5 101.7,141.2 102.7,142.9 103.7,144.5 104.7,146.2 105.8,147.8 106.8,149.5 107.8,151.2 108.9,152.8 109.9,154.5 110.9,156.1 112.0,157.8 113.0,159.4 114.0,161.1 115.1,162.8 116.1,164.4 117.1,166.1 118.1,167.7 119.2,169.4 120.2,171.1 121.2,172.7 122.3,174.4 123.3,176.0 124.3,177.7 125.4,179.3 126.4,181.0 127.4,182.7 128.4,184.3 129.5,186.0 130.5,187.6 131.5,189.3 132.6,191.0 133.6,192.6 134.6,194.3 135.7,195.9 136.7,197.6 137.7,199.2 138.7,200.9 139.8,202.6 140.8,204.2 141.8,205.9 142.9,207.5 143.9,209.2 144.9,210.9 146.0,212.5 147.0,214.2 148.0,214.2 149.0,212.5 150.1,210.9 151.1,209.2 152.1,207.5 153.2,205.9 154.2,204.2 155.2,202.6 156.3,200.9 157.3,199.2 158.3,197.6 159.3,195.9 160.4,194.3 161.4,192.6 162.4,191.0 163.5,189.3 164.5,187.6 165.5,186.0 166.6,184.3 167.6,182.7 168.6,181.0 169.6,179.3 170.7,177.7 171.7,176.0 172.7,174.4 173.8,172.7 174.8,171.1 175.8,169.4 176.9,167.7 177.9,166.1 178.9,164.4 179.9,162.8 181.0,161.1 182.0,159.4 183.0,157.8 184.1,156.1 185.1,154.5 186.1,152.8 187.2,151.2 188.2,149.5 189.2,147.8 190.3,146.2 191.3,144.5 192.3,142.9 193.3,141.2 194.4,139.5 195.4,137.9 196.4,136.2 197.5,134.6 198.5,132.9 199.5,131.3 200.6,129.6 201.6,127.9 202.6,126.3 203.6,124.6 204.7,123.0 205.7,121.3 206.7,119.6 207.8,118.0 208.8,116.3 209.8,114.7 210.9,113.0 211.9,111.4 212.9,109.7 213.9,108.0 215.0,106.4 216.0,104.7 217.0,103.1 218.1,101.4 219.1,99.7 220.1,98.1 221.2,96.4 222.2,94.8 223.2,93.1 224.2,91.5 225.3,89.8 226.3,88.1 227.3,86.5 228.4,84.8 229.4,83.2 230.4,81.5 231.5,79.8 232.5,78.2 233.5,76.5 234.5,74.9 235.6,73.2 236.6,71.6 237.6,69.9 238.7,68.2 239.7,66.6 240.7,64.9 241.8,63.3 242.8,61.6 243.8,59.9 244.8,58.3 245.9,56.6 246.9,55.0 247.9,53.3 249.0,51.7 250.0,50.0" fill="none" class="stroke-c" stroke-width="3"/>
<text x="193.6" y="95.4" class="fig-small" text-anchor="end">lasso |w|</text>
<text x="216.7" y="157.8" class="fig-small">ridge w²/2</text>
<line x1="340.0" y1="215.0" x2="570.0" y2="215.0" class="stroke-muted" />
<line x1="340.0" y1="215.0" x2="340.0" y2="50.0" class="stroke-muted" />
<line x1="340.0" y1="215.0" x2="340.0" y2="219.0" class="stroke-muted" />
<text x="340.0" y="231.0" class="fig-small" text-anchor="middle">0</text>
<line x1="397.5" y1="215.0" x2="397.5" y2="219.0" class="stroke-muted" />
<text x="397.5" y="231.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="455.0" y1="215.0" x2="455.0" y2="219.0" class="stroke-muted" />
<text x="455.0" y="231.0" class="fig-small" text-anchor="middle">1</text>
<line x1="512.5" y1="215.0" x2="512.5" y2="219.0" class="stroke-muted" />
<text x="512.5" y="231.0" class="fig-small" text-anchor="middle">1.5</text>
<line x1="570.0" y1="215.0" x2="570.0" y2="219.0" class="stroke-muted" />
<text x="570.0" y="231.0" class="fig-small" text-anchor="middle">2</text>
<line x1="340.0" y1="215.0" x2="570.0" y2="215.0" class="stroke-grid" />
<line x1="336.0" y1="215.0" x2="340.0" y2="215.0" class="stroke-muted" />
<text x="332.0" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="340.0" y1="156.1" x2="570.0" y2="156.1" class="stroke-grid" />
<line x1="336.0" y1="156.1" x2="340.0" y2="156.1" class="stroke-muted" />
<text x="332.0" y="160.1" class="fig-small" text-anchor="end">0.5</text>
<line x1="340.0" y1="97.1" x2="570.0" y2="97.1" class="stroke-grid" />
<line x1="336.0" y1="97.1" x2="340.0" y2="97.1" class="stroke-muted" />
<text x="332.0" y="101.1" class="fig-small" text-anchor="end">1</text>
<text x="455.0" y="250.0" class="fig-small" text-anchor="middle">weight w</text>
<text x="300.0" y="24.0" class="fig-bold">Tug of war on one weight (w ≥ 0)</text>
<polyline points="340.0,156.1 342.0,158.1 344.0,160.1 345.9,162.2 347.9,164.2 349.9,166.2 351.9,168.3 353.9,170.3 355.9,172.3 357.8,174.4 359.8,176.4 361.8,178.4 363.8,180.5 365.8,182.5 367.8,184.5 369.7,186.6 371.7,188.6 373.7,190.6 375.7,192.6 377.7,194.7 379.7,196.7 381.6,198.7 383.6,200.8 385.6,202.8 387.6,204.8 389.6,206.9 391.6,208.9 393.5,210.9 395.5,213.0 397.5,215.0" fill="none" class="stroke-a" stroke-width="3"/>
<polyline points="340.0,215.0 343.3,211.6 346.6,208.3 349.9,204.9 353.1,201.5 356.4,198.2 359.7,194.8 363.0,191.4 366.3,188.1 369.6,184.7 372.9,181.3 376.1,178.0 379.4,174.6 382.7,171.2 386.0,167.9 389.3,164.5 392.6,161.1 395.9,157.8 399.1,154.4 402.4,151.0 405.7,147.7 409.0,144.3 412.3,140.9 415.6,137.6 418.9,134.2 422.1,130.8 425.4,127.4 428.7,124.1 432.0,120.7 435.3,117.3 438.6,114.0 441.9,110.6 445.1,107.2 448.4,103.9 451.7,100.5 455.0,97.1 458.3,93.8 461.6,90.4 464.9,87.0 468.1,83.7 471.4,80.3 474.7,76.9 478.0,73.6 481.3,70.2 484.6,66.8 487.9,63.5 491.1,60.1 494.4,56.7 497.7,53.4 501.0,50.0" fill="none" class="stroke-d" stroke-width="3"/>
<polyline points="340.0,97.1 342.3,97.1 344.6,97.1 347.0,97.1 349.3,97.1 351.6,97.1 353.9,97.1 356.3,97.1 358.6,97.1 360.9,97.1 363.2,97.1 365.6,97.1 367.9,97.1 370.2,97.1 372.5,97.1 374.8,97.1 377.2,97.1 379.5,97.1 381.8,97.1 384.1,97.1 386.5,97.1 388.8,97.1 391.1,97.1 393.4,97.1 395.8,97.1 398.1,97.1 400.4,97.1 402.7,97.1 405.1,97.1 407.4,97.1 409.7,97.1 412.0,97.1 414.3,97.1 416.7,97.1 419.0,97.1 421.3,97.1 423.6,97.1 426.0,97.1 428.3,97.1 430.6,97.1 432.9,97.1 435.3,97.1 437.6,97.1 439.9,97.1 442.2,97.1 444.5,97.1 446.9,97.1 449.2,97.1 451.5,97.1 453.8,97.1 456.2,97.1 458.5,97.1 460.8,97.1 463.1,97.1 465.5,97.1 467.8,97.1 470.1,97.1 472.4,97.1 474.7,97.1 477.1,97.1 479.4,97.1 481.7,97.1 484.0,97.1 486.4,97.1 488.7,97.1 491.0,97.1 493.3,97.1 495.7,97.1 498.0,97.1 500.3,97.1 502.6,97.1 504.9,97.1 507.3,97.1 509.6,97.1 511.9,97.1 514.2,97.1 516.6,97.1 518.9,97.1 521.2,97.1 523.5,97.1 525.9,97.1 528.2,97.1 530.5,97.1 532.8,97.1 535.2,97.1 537.5,97.1 539.8,97.1 542.1,97.1 544.4,97.1 546.8,97.1 549.1,97.1 551.4,97.1 553.7,97.1 556.1,97.1 558.4,97.1 560.7,97.1 563.0,97.1 565.4,97.1 567.7,97.1 570.0,97.1" fill="none" class="stroke-c" stroke-width="3"/>
<text x="404.4" y="207.9" class="fig-small">data's pull 0.5 - w</text>
<text x="507.0" y="64.0" class="fig-small">ridge pull w</text>
<text x="351.5" y="87.1" class="fig-small">lasso pull 1, always</text>
<circle cx="368.8" cy="185.5" r="5" class="fill-d"/>
<text x="363.0" y="141.9" class="fig-small">ridge balances at 0.25</text>
<circle cx="340.0" cy="97.1" r="5" class="fill-c"/>
</svg>
Figure 1. Left: the two fees. Right: the pulls on one weight, for a feature whose data pull at w = 0 is 0.5 and falls as w grows. The ridge pull w fades to nothing at 0, so it balances the data at 0.25. The lasso pull stays at 1, above the data's pull everywhere, so nothing balances it except w = 0.
```

So the lasso zeroes a weight whenever the data's pull at 0 is no larger than $\lambda$. Ridge zeroes a weight only if the data's pull at 0 is exactly 0.

## Solve one number: soft thresholding

When the design is the identity, the lasso score splits into one separate problem for each weight. For one number $x$ and fee $t$, you want the $z$ that minimises $\tfrac12(z - x)^2 + t\,|z|$: stay close to $x$, but pay $t$ for every unit of distance from 0.

Try $t = 1$. For $x = 3$, staying at 3 costs $0 + 3 = 3$, moving to 2 costs $\tfrac12(1)^2 + 2 = 2.5$, and moving to 0 costs $\tfrac12(9) = 4.5$. So 2 wins, which is $3 - 1$. For $x = 1$, staying costs 1, moving to 0.5 costs $0.125 + 0.5 = 0.625$, and 0 costs $0.5$. So 0 wins.

The pattern is: move $x$ towards 0 by $t$, and if that would cross 0, stop at 0. For $z > 0$ the slope of the cost is $(z - x) + t$, which is zero at $z = x - t$. That point is valid only when it is positive, so when $x \le t$ the slope is positive at every $z > 0$ and the cost is lowest at the corner, $z = 0$. As a formula, this is *soft thresholding*:

$$
S_t(x) = \mathrm{sign}(x)\,\max(|x| - t,\; 0)
$$

Read it as: take the size of $x$, subtract $t$, floor the result at 0, and put the sign back. At $t = 1$ it sends $3, -3, 0.5, -0.5, 0, 2$ to $2, -2, 0, 0, 0, 1$. Figure 2 draws it: a line parallel to the diagonal but one unit closer to 0, with a flat stretch at 0 where small inputs are killed.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="The soft-threshold graph for threshold 1: a flat zero stretch between -1 and 1, then lines that run parallel to the diagonal but one unit closer to zero.">
<defs><marker id="c007-soft-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="60.0" y1="125.0" x2="290.0" y2="125.0" class="stroke-muted" />
<line x1="175.0" y1="235.0" x2="175.0" y2="15.0" class="stroke-muted" />
<text x="290.0" y="141.0" class="fig-small" text-anchor="end">x</text>
<text x="181.0" y="25.0" class="fig-small">S(x)</text>
<rect x="146.2" y="15.0" width="57.5" height="220.0" class="fill-soft-b"/>
<line x1="60.0" y1="125.0" x2="290.0" y2="125.0" class="stroke-muted" />
<line x1="175.0" y1="235.0" x2="175.0" y2="15.0" class="stroke-muted" />
<line x1="60.0" y1="235.0" x2="290.0" y2="15.0" class="stroke-grid dashed" />
<polyline points="60.0,207.5 60.6,206.9 61.1,206.4 61.7,205.8 62.3,205.3 62.9,204.8 63.5,204.2 64.0,203.7 64.6,203.1 65.2,202.6 65.8,202.0 66.3,201.4 66.9,200.9 67.5,200.4 68.1,199.8 68.6,199.2 69.2,198.7 69.8,198.2 70.3,197.6 70.9,197.1 71.5,196.5 72.1,195.9 72.7,195.4 73.2,194.8 73.8,194.3 74.4,193.8 75.0,193.2 75.5,192.7 76.1,192.1 76.7,191.6 77.2,191.0 77.8,190.4 78.4,189.9 79.0,189.3 79.6,188.8 80.1,188.2 80.7,187.7 81.3,187.1 81.8,186.6 82.4,186.0 83.0,185.5 83.6,184.9 84.1,184.4 84.7,183.8 85.3,183.3 85.9,182.8 86.5,182.2 87.0,181.7 87.6,181.1 88.2,180.6 88.8,180.0 89.3,179.4 89.9,178.9 90.5,178.3 91.0,177.8 91.6,177.2 92.2,176.7 92.8,176.2 93.3,175.6 93.9,175.1 94.5,174.5 95.1,174.0 95.7,173.4 96.2,172.9 96.8,172.3 97.4,171.8 98.0,171.2 98.5,170.7 99.1,170.1 99.7,169.6 100.2,169.0 100.8,168.4 101.4,167.9 102.0,167.3 102.5,166.8 103.1,166.2 103.7,165.7 104.3,165.2 104.8,164.6 105.4,164.1 106.0,163.5 106.6,162.9 107.2,162.4 107.7,161.8 108.3,161.3 108.9,160.8 109.4,160.2 110.0,159.6 110.6,159.1 111.2,158.6 111.8,158.0 112.3,157.4 112.9,156.9 113.5,156.3 114.0,155.8 114.6,155.2 115.2,154.7 115.8,154.2 116.3,153.6 116.9,153.1 117.5,152.5 118.1,151.9 118.7,151.4 119.2,150.8 119.8,150.3 120.4,149.8 121.0,149.2 121.5,148.6 122.1,148.1 122.7,147.6 123.2,147.0 123.8,146.4 124.4,145.9 125.0,145.3 125.6,144.8 126.1,144.2 126.7,143.7 127.3,143.2 127.8,142.6 128.4,142.1 129.0,141.5 129.6,140.9 130.1,140.4 130.7,139.8 131.3,139.3 131.9,138.8 132.4,138.2 133.0,137.7 133.6,137.1 134.2,136.6 134.8,136.0 135.3,135.4 135.9,134.9 136.5,134.3 137.1,133.8 137.6,133.2 138.2,132.7 138.8,132.1 139.4,131.6 139.9,131.1 140.5,130.5 141.1,129.9 141.6,129.4 142.2,128.9 142.8,128.3 143.4,127.8 143.9,127.2 144.5,126.7 145.1,126.1 145.7,125.5 146.2,125.0 146.8,125.0 147.4,125.0 148.0,125.0 148.6,125.0 149.1,125.0 149.7,125.0 150.3,125.0 150.9,125.0 151.4,125.0 152.0,125.0 152.6,125.0 153.2,125.0 153.7,125.0 154.3,125.0 154.9,125.0 155.4,125.0 156.0,125.0 156.6,125.0 157.2,125.0 157.8,125.0 158.3,125.0 158.9,125.0 159.5,125.0 160.1,125.0 160.6,125.0 161.2,125.0 161.8,125.0 162.4,125.0 162.9,125.0 163.5,125.0 164.1,125.0 164.7,125.0 165.2,125.0 165.8,125.0 166.4,125.0 166.9,125.0 167.5,125.0 168.1,125.0 168.7,125.0 169.2,125.0 169.8,125.0 170.4,125.0 171.0,125.0 171.6,125.0 172.1,125.0 172.7,125.0 173.3,125.0 173.8,125.0 174.4,125.0 175.0,125.0 175.6,125.0 176.2,125.0 176.7,125.0 177.3,125.0 177.9,125.0 178.4,125.0 179.0,125.0 179.6,125.0 180.2,125.0 180.8,125.0 181.3,125.0 181.9,125.0 182.5,125.0 183.1,125.0 183.6,125.0 184.2,125.0 184.8,125.0 185.4,125.0 185.9,125.0 186.5,125.0 187.1,125.0 187.7,125.0 188.2,125.0 188.8,125.0 189.4,125.0 190.0,125.0 190.5,125.0 191.1,125.0 191.7,125.0 192.3,125.0 192.8,125.0 193.4,125.0 194.0,125.0 194.5,125.0 195.1,125.0 195.7,125.0 196.3,125.0 196.8,125.0 197.4,125.0 198.0,125.0 198.6,125.0 199.2,125.0 199.7,125.0 200.3,125.0 200.9,125.0 201.4,125.0 202.0,125.0 202.6,125.0 203.2,125.0 203.8,125.0 204.3,124.4 204.9,123.9 205.5,123.3 206.1,122.8 206.6,122.2 207.2,121.7 207.8,121.2 208.3,120.6 208.9,120.1 209.5,119.5 210.1,119.0 210.7,118.4 211.2,117.9 211.8,117.3 212.4,116.8 213.0,116.2 213.5,115.7 214.1,115.1 214.7,114.5 215.2,114.0 215.8,113.5 216.4,112.9 217.0,112.3 217.6,111.8 218.1,111.2 218.7,110.7 219.3,110.2 219.9,109.6 220.4,109.0 221.0,108.5 221.6,108.0 222.1,107.4 222.7,106.8 223.3,106.3 223.9,105.8 224.4,105.2 225.0,104.7 225.6,104.1 226.2,103.5 226.8,103.0 227.3,102.4 227.9,101.9 228.5,101.3 229.0,100.8 229.6,100.2 230.2,99.7 230.8,99.1 231.3,98.6 231.9,98.0 232.5,97.5 233.1,96.9 233.7,96.4 234.2,95.8 234.8,95.3 235.4,94.7 236.0,94.2 236.5,93.6 237.1,93.1 237.7,92.6 238.2,92.0 238.8,91.5 239.4,90.9 240.0,90.3 240.6,89.8 241.1,89.2 241.7,88.7 242.3,88.2 242.9,87.6 243.4,87.1 244.0,86.5 244.6,85.9 245.2,85.4 245.7,84.8 246.3,84.3 246.9,83.8 247.5,83.2 248.0,82.7 248.6,82.1 249.2,81.6 249.8,81.0 250.3,80.4 250.9,79.9 251.5,79.3 252.0,78.8 252.6,78.2 253.2,77.7 253.8,77.2 254.3,76.6 254.9,76.0 255.5,75.5 256.1,74.9 256.6,74.4 257.2,73.8 257.8,73.3 258.4,72.8 258.9,72.2 259.5,71.6 260.1,71.1 260.7,70.5 261.2,70.0 261.8,69.4 262.4,68.9 263.0,68.3 263.6,67.8 264.1,67.2 264.7,66.7 265.3,66.1 265.9,65.6 266.4,65.1 267.0,64.5 267.6,64.0 268.1,63.4 268.7,62.8 269.3,62.3 269.9,61.8 270.5,61.2 271.0,60.7 271.6,60.1 272.2,59.6 272.8,59.0 273.3,58.4 273.9,57.9 274.5,57.3 275.1,56.8 275.6,56.2 276.2,55.7 276.8,55.2 277.4,54.6 277.9,54.1 278.5,53.5 279.1,52.9 279.6,52.4 280.2,51.8 280.8,51.3 281.4,50.8 281.9,50.2 282.5,49.7 283.1,49.1 283.7,48.5 284.2,48.0 284.8,47.4 285.4,46.9 286.0,46.3 286.5,45.8 287.1,45.2 287.7,44.7 288.3,44.1 288.9,43.6 289.4,43.0 290.0,42.5" fill="none" class="stroke-b" stroke-width="3.5"/>
<line x1="261.2" y1="42.5" x2="261.2" y2="63.0" class="stroke-ink" stroke-width="1.8" marker-end="url(#c007-soft-arrow)"/>
<circle cx="261.2" cy="42.5" r="4" class="fill-a"/>
<circle cx="261.2" cy="70.0" r="4.5" class="fill-b"/>
<line x1="88.8" y1="207.5" x2="88.8" y2="187.0" class="stroke-ink" stroke-width="1.8" marker-end="url(#c007-soft-arrow)"/>
<circle cx="88.8" cy="207.5" r="4" class="fill-a"/>
<circle cx="88.8" cy="180.0" r="4.5" class="fill-b"/>
<line x1="198.0" y1="103.0" x2="198.0" y2="118.0" class="stroke-ink" stroke-width="1.8" marker-end="url(#c007-soft-arrow)"/>
<circle cx="198.0" cy="103.0" r="4" class="fill-a"/>
<circle cx="198.0" cy="125.0" r="4.5" class="fill-b"/>
<text x="320.0" y="80.0" class="fig-small fig-bold">Threshold t = 1.</text>
<text x="320.0" y="100.0" class="fig-small">Dashed line: the input, S(x) = x.</text>
<text x="320.0" y="120.0" class="fig-small">Orange: the output. It is the dashed line</text>
<text x="320.0" y="140.0" class="fig-small">moved 1 closer to 0.</text>
<text x="320.0" y="160.0" class="fig-small">Shaded strip: every input with |x| ≤ 1</text>
<text x="320.0" y="180.0" class="fig-small">is sent to exactly 0.</text>
<text x="320.0" y="200.0" class="fig-small fig-mono">3 → 2   0.8 → 0   -3 → -2</text>
</svg>
Figure 2. Soft thresholding at t = 1. Each blue dot is an input and each orange dot is the output. Inputs outside the shaded strip move 1 towards 0. Inputs inside it land exactly on 0, which is where the lasso's exact zeros come from.
```

```python
z = np.sign(x) * np.maximum(np.abs(x) - t, 0.0)   # np.maximum works entry by entry
```

## The lasso when the design is the identity

Write the whole lasso score:

$$
L(w) = \tfrac12\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert_1
$$

Read it as: half the squared misses, plus lambda times the sum of the absolute weights. There is no half on the fee, so that the threshold in the one-number problem is exactly $\lambda$.

With $X = I$ the misses are $w_i - y_i$, so the score is a sum of one-number problems and the answer is $w = S_\lambda(y)$. On the example $y = (3, 1)$ and $\lambda = 1$, the lasso gives $(2, 0)$ and ridge gives $(1.5, 0.5)$. The lasso scores $\tfrac12(1 + 1) + 2 = 3$ at its own answer, but $1.25 + 2 = 3.25$ at ridge's answer. The second feature's data pull at 0 is 1, which is no larger than $\lambda = 1$, so it is killed. The first feature's pull is 3, which beats the fee, so it survives with 1 unit of weight taken off. A fee of $\lambda = 3$ or more would zero both weights.

```python
misses = X @ w - y
score = 0.5 * np.sum(misses ** 2) + lam * np.sum(np.abs(w))
```

Copied columns behave differently from ridge. Take $X$ with rows $(1, 1)$ and $(2, 2)$, labels $(1, 2)$, and $\lambda = 1$. Ridge split the weight evenly. For the lasso, only the sum $w_1 + w_2$ affects the misses, and among non-negative weights the fee also depends only on the sum, so $(0.8, 0)$, $(0.4, 0.4)$, and $(0, 0.8)$ all score 0.9. The lasso does not care how copies share a signal, so which split you get depends on the solver.

> [!KEY]
> Ridge's pull on a weight fades as the weight nears 0, so a weak signal leaves a small nonzero weight. The lasso charges a constant pull of $\lambda$ per unit of weight, so any feature whose data pull at 0 is at most $\lambda$ stays at exactly 0. For one number this is soft thresholding: move towards 0 by $\lambda$, and stop at 0.

## Fit any design: why not set the slope to zero, or walk anyway?

For a general $X$ the squared error ties the weights together, and the corner means weights should land on exactly 0. Ridge had a formula because its score is smooth, so you set the slope to zero. The lasso fee has slope $-\lambda$ on the left of 0 and $+\lambda$ on the right, with no single slope in between, so you cannot solve "slope equals zero" at a weight that sits on the corner, and sitting there is what you want.

You might walk anyway, using the slope $\lambda\,\mathrm{sign}(w)$ for the fee. Take $X$ with one column $(1, 2)$, labels $y = (1, 2)$, and $\lambda = 6$. The best weight is 0, because the data's pull at 0 is 5, which is below 6. Start at $w = 0.1$ with step 0.2. The slope of the squared error is $5w - 5$, so the total slope is $-4.5 + 6 = 1.5$ and the walk goes to $0.1 - 0.2 \times 1.5 = -0.2$. There the slope is $-6 - 6 = -12$, so it jumps to 2.2, then back to $-0.2$, and so on forever. Each step jumps over 0 and never lands on it.

The method that works treats the two parts differently. It walks on the smooth part, and for the corner it asks a direct question: given where the walk landed, what is the best compromise between staying there and paying the fee? You answered that for one number above. It is soft thresholding.

### Step downhill, then threshold

One ISTA step takes a plain gradient step on the squared error, then applies soft thresholding. Use the column $(1, 2)$ with labels $(1, 2)$ and $\lambda = 1$, and start at $w = 0$. The misses are $-1$ and $-2$, so the score is $\tfrac12(1 + 4) = 2.5$. The gradient of the squared error is $X^\top(Xw - y) = 1(-1) + 2(-2) = -5$. Here $X^\top X = 5$, so $L = 5$ and the step is $1/L = 0.2$. The gradient step goes to $0 - 0.2 \times (-5) = 1$. The fee gets the same step size, so you threshold at $0.2 \times 1 = 0.2$, and $1$ becomes $0.8$. The misses are $-0.2$ and $-0.4$, and the score is $\tfrac12(0.04 + 0.16) + 0.8 = 0.9$. In symbols, with step $\alpha$:

$$
w_{k+1} = S_{\alpha\lambda}\!\big(w_k - \alpha\, X^\top(Xw_k - y)\big)
$$

Read it as: step against the gradient of the squared error, then soft-threshold at the step size times lambda. The threshold is step times lambda because the best compromise between staying near the gradient step's landing point $v$ and paying the fee is the one-number problem $\tfrac12(z - v)^2 + \alpha\lambda|z|$, with $t = \alpha\lambda$. A step of $1/L$ is safe because the squared error is a bowl whose steepest curvature is $L$. For a convex bowl and a step of at most $1/L$, the score never rises from one ISTA step to the next.

```python
grad = X.T @ (X @ w - y)                          # shape (d,)
w = soft_threshold(w - step * grad, step * lam)   # threshold is step * lam, not lam
```

Return to the $\lambda = 6$ case where the plain walk bounced. From $w = 0.1$ the gradient step goes to $0.1 - 0.2 \times (-4.5) = 1.0$, and the threshold is $0.2 \times 6 = 1.2$. Since $1.0 < 1.2$, the weight lands on exactly 0 and stays there. When $X$ is the identity, $L = 1$ and the step is 1, so the gradient step lands on $y$ whatever $w$ was, and the threshold then gives $S_\lambda(y)$. One ISTA step is the exact solve you built above.

### Know when you have finished: the KKT rules

At the bottom of a smooth bowl the slope is zero. The lasso needs a test that also covers weights on the corner. Write $g = X^\top(Xw - y)$, the gradient of the squared error, and read $-g_i$ as the data's pull on weight $i$. The fee's pull is $-\lambda\,\mathrm{sign}(w_i)$. The *KKT rules*, named after Karush, Kuhn, and Tucker, say each weight must obey one of these:

- If $w_i \ne 0$, then $g_i + \lambda\,\mathrm{sign}(w_i) = 0$. The data's pull and the fee's pull cancel.
- If $w_i = 0$, then $|g_i| \le \lambda$. The data's pull is too weak to move the weight off 0.

On the column example at $w = 0.8$ we have $g = -1$, so $g + \lambda = 0$ and the first rule holds, which confirms that 0.8 is the answer. At $w = 0$ we have $g = -5$, and $5 > 1$ breaks the second rule by $5 - 1 = 4$, so you have not finished. Define the *violation* as the largest breach over all weights. It is 0 at a lasso answer and positive elsewhere, which makes it the test for "am I done?"

```python
g = X.T @ (X @ w - y)
breach = np.where(w != 0, np.abs(g + lam * np.sign(w)), np.maximum(np.abs(g) - lam, 0.0))
```

### Add momentum: FISTA

Correlated columns make a long, narrow valley, and ISTA's step is set by the steepest wall, so along the nearly flat floor each step covers little. In Figure 3 the curvatures are about 78 and 0.34, a ratio of 231, so a step covers about 1/231 of the remaining distance. Momentum is the cure, as in [gradient descent and momentum](/lessons/005-gradient-descent-and-momentum).

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="Contours of a long narrow valley in the plane of two weights, with the ISTA path creeping along the valley floor and the FISTA path moving along it faster, both towards the best weights.">

<polyline points="301.8,195.5 301.2,194.9 300.7,194.4 300.2,193.9 299.8,193.5 299.4,193.0 299.0,192.6 298.6,192.2 298.2,191.9 297.8,191.5 297.5,191.2 297.1,190.8 296.8,190.5 296.5,190.2 296.2,189.9 295.9,189.6 295.6,189.3 295.3,189.0 295.0,188.7 294.7,188.4 294.4,188.1 294.1,187.9 293.9,187.6 293.6,187.3 293.3,187.0 293.0,186.7 292.7,186.5 292.4,186.2 292.2,185.9 291.9,185.6 291.6,185.3 291.3,185.0 290.9,184.7 290.6,184.4 290.3,184.1 290.0,183.7 289.6,183.4 289.3,183.0 288.9,182.7 288.5,182.3 288.1,181.9 287.7,181.5 287.2,181.0 286.7,180.6 286.2,180.1 285.7,179.5 285.1,179.0 284.5,178.4 283.8,177.7 283.1,177.0 282.3,176.2 281.4,175.4 280.5,174.4 279.4,173.3 278.1,172.1 276.7,170.8 275.1,169.2 273.2,167.3 270.9,165.2 268.2,162.5 264.8,159.3 260.5,155.2 255.0,149.9 247.4,142.8 236.8,132.9 227.7,126.3 223.8,126.7 221.5,129.0 221.0,133.0 222.2,138.2 224.4,144.0 227.1,149.7 230.1,155.1 238.0,163.3 243.8,169.3 248.2,173.8 251.7,177.4 254.6,180.3 256.9,182.6 258.8,184.6 260.5,186.3 261.9,187.7 263.2,189.0 264.3,190.1 265.3,191.1 266.2,192.0 267.0,192.8 267.8,193.6 268.4,194.2 269.1,194.9 269.6,195.5 270.2,196.0 270.7,196.5 271.2,197.0 271.6,197.4 272.1,197.9 272.5,198.3 272.9,198.7 273.2,199.0 273.6,199.4 273.9,199.7 274.3,200.1 274.6,200.4 274.9,200.7 275.2,201.0 275.5,201.3 275.8,201.6 276.1,201.9 276.4,202.2 276.7,202.5 277.0,202.8 277.3,203.0 277.6,203.3 277.8,203.6 278.1,203.9 278.4,204.2 278.7,204.4 279.0,204.7 279.3,205.0 279.6,205.3 279.9,205.6 280.2,205.9 280.5,206.2 280.8,206.5 281.1,206.8 281.4,207.2 281.8,207.5 282.1,207.9 282.5,208.2 282.9,208.6 283.3,209.0 283.7,209.4 284.2,209.9 284.7,210.3 285.2,210.8 285.7,211.4 286.3,211.9 286.9,212.5 287.6,213.2 288.3,213.9 289.1,214.7 290.0,215.5 291.0,216.5 292.0,217.6 293.3,218.8 294.7,220.1 296.3,221.7 298.2,223.6 300.5,225.8 303.2,228.4 306.6,231.6 310.9,235.7 316.5,241.0 324.0,248.2 334.6,258.1" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="341.3,235.8 333.4,227.6 327.6,221.6 323.2,217.1 319.7,213.5 316.9,210.6 314.5,208.3 312.6,206.3 310.9,204.6 309.5,203.2 308.2,201.9 307.1,200.8 306.1,199.8 305.2,198.9 304.4,198.1 303.7,197.3 303.0,196.7 302.4,196.0 301.8,195.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="317.8,195.5 316.8,194.4 315.7,193.4 314.8,192.4 313.9,191.5 313.0,190.6 312.2,189.8 311.4,189.0 310.7,188.3 309.9,187.6 309.2,186.9 308.6,186.2 307.9,185.6 307.3,184.9 306.7,184.3 306.0,183.7 305.4,183.1 304.9,182.5 304.3,182.0 303.7,181.4 303.1,180.8 302.6,180.3 302.0,179.7 301.4,179.2 300.9,178.6 300.3,178.0 299.8,177.5 299.2,176.9 298.6,176.3 298.0,175.8 297.4,175.2 296.8,174.6 296.2,174.0 295.6,173.3 294.9,172.7 294.2,172.0 293.5,171.3 292.8,170.6 292.1,169.9 291.3,169.1 290.5,168.3 289.6,167.5 288.7,166.6 287.8,165.7 286.8,164.7 285.7,163.6 284.6,162.5 283.3,161.3 282.0,160.0 280.5,158.5 278.9,157.0 277.2,155.3 275.2,153.4 273.0,151.2 270.6,148.8 267.7,146.1 264.5,142.9 260.7,139.2 256.1,134.8 250.7,129.6 243.9,123.1 235.4,114.9 226.4,107.6 218.5,103.0 210.0,98.5 201.4,95.0 193.9,93.5 189.1,95.4 188.0,101.1 190.5,109.7 195.1,119.4 200.5,128.8 205.8,137.4 210.8,144.9 215.3,151.5 219.3,157.1 222.8,162.0 226.0,166.3 228.8,170.1 232.0,173.7 235.3,177.1 238.2,180.0 240.7,182.5 242.9,184.8 244.9,186.8 246.7,188.6 248.3,190.2 249.8,191.7 251.2,193.0 252.4,194.3 253.6,195.5 254.7,196.5 255.7,197.6 256.6,198.5 257.5,199.4 258.4,200.3 259.2,201.1 260.0,201.9 260.7,202.6 261.5,203.3 262.2,204.0 262.8,204.7 263.5,205.3 264.1,206.0 264.8,206.6 265.4,207.2 266.0,207.8 266.6,208.4 267.1,208.9 267.7,209.5 268.3,210.1 268.8,210.6 269.4,211.2 270.0,211.8 270.5,212.3 271.1,212.9 271.7,213.4 272.2,214.0 272.8,214.6 273.4,215.2 274.0,215.7 274.6,216.3 275.2,217.0 275.9,217.6 276.5,218.2 277.2,218.9 277.9,219.6 278.6,220.3 279.3,221.0 280.1,221.8 280.9,222.6 281.8,223.4 282.7,224.3 283.6,225.2 284.6,226.2 285.7,227.3 286.9,228.4 288.1,229.6 289.4,230.9 290.9,232.4 292.5,233.9 294.2,235.6 296.2,237.5 298.4,239.7 300.9,242.1 303.7,244.8 306.9,248.0 310.7,251.7 315.3,256.1" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="348.0,225.8 343.4,221.1 339.5,217.2 336.1,213.8 333.2,210.9 330.7,208.4 328.5,206.1 326.5,204.1 324.7,202.3 323.1,200.7 321.6,199.2 320.3,197.9 319.0,196.6 317.8,195.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="341.4,195.5 339.5,193.6 337.7,191.8 336.1,190.2 334.5,188.6 333.0,187.1 331.6,185.7 330.2,184.4 328.9,183.1 327.7,181.8 326.5,180.6 325.3,179.5 324.2,178.3 323.1,177.2 322.0,176.2 320.9,175.1 319.9,174.1 318.9,173.1 317.9,172.1 316.9,171.1 315.9,170.1 314.9,169.1 313.9,168.2 313.0,167.2 312.0,166.3 311.0,165.3 310.0,164.3 309.0,163.3 308.0,162.3 307.0,161.3 306.0,160.3 304.9,159.3 303.9,158.2 302.8,157.1 301.6,156.0 300.5,154.9 299.3,153.7 298.0,152.5 296.7,151.2 295.4,149.9 294.0,148.5 292.5,147.0 291.0,145.5 289.3,143.9 287.6,142.2 285.7,140.3 283.7,138.4 281.6,136.3 279.2,134.0 276.7,131.5 274.0,128.8 270.9,125.8 267.5,122.5 263.7,118.8 259.5,114.7 254.6,109.9 248.9,104.5 242.3,98.1 234.5,90.5 225.9,82.9 216.6,75.8 205.8,67.7 193.2,58.3 178.3,47.6 161.1,36.0 142.1,24.3" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="107.0,22.8 114.1,40.9 126.5,61.9 139.9,81.5 152.2,98.4 162.9,112.6 172.0,124.4 179.8,134.3 186.5,142.7 192.3,149.9 197.4,156.1 201.8,161.6 205.8,166.4 209.3,170.6 212.4,174.4 215.3,177.9 217.9,181.0 220.3,183.9 222.6,186.6 224.6,189.0 226.6,191.3 228.3,193.4 230.0,195.5 231.9,197.3 233.7,199.1 235.3,200.7 236.9,202.3 238.4,203.8 239.8,205.2 241.2,206.6 242.5,207.8 243.7,209.1 244.9,210.3 246.1,211.5 247.2,212.6 248.3,213.7 249.4,214.7 250.5,215.8 251.5,216.8 252.5,217.8 253.5,218.8 254.5,219.8 255.5,220.8 256.5,221.8 257.5,222.7 258.4,223.7 259.4,224.7 260.4,225.6 261.4,226.6 262.4,227.6 263.4,228.6 264.4,229.6 265.4,230.6 266.5,231.6 267.5,232.7 268.6,233.8 269.8,234.9 270.9,236.0 272.1,237.2 273.4,238.4 274.7,239.7 276.0,241.0 277.4,242.4 278.9,243.9 280.5,245.4 282.1,247.0 283.8,248.7 285.7,250.6 287.7,252.5 289.8,254.6 292.2,256.9 294.7,259.4" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="347.9,202.0 345.6,199.6 343.4,197.5 341.4,195.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="349.8,174.6 348.0,172.8 346.2,171.0 344.5,169.3 342.8,167.6 341.1,166.0 339.5,164.4 337.9,162.8 336.4,161.3 334.9,159.7 333.3,158.2 331.8,156.8 330.3,155.3 328.8,153.8 327.4,152.3 325.9,150.8 324.4,149.4 322.9,147.9 321.4,146.4 319.8,144.9 318.3,143.3 316.7,141.8 315.1,140.2 313.4,138.6 311.8,136.9 310.0,135.2 308.3,133.5 306.4,131.7 304.5,129.8 302.6,127.8 300.5,125.8 298.4,123.7 296.1,121.5 293.7,119.1 291.2,116.7 288.5,114.1 285.7,111.3 282.7,108.3 279.4,105.1 275.8,101.6 272.0,97.8 267.8,93.7 263.1,89.1 257.9,84.1 252.2,78.4 245.6,72.1 238.2,64.8 229.6,56.5 220.4,48.7 209.8,39.8 197.4,29.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="69.3,38.3 90.1,63.5 107.1,83.8 121.1,100.4 132.9,114.2 142.8,125.8 151.4,135.6 158.8,144.2 165.3,151.6 171.0,158.2 176.2,164.0 180.8,169.3 185.0,174.0 188.8,178.4 192.3,182.3 195.5,186.0 198.5,189.4 201.3,192.5 203.9,195.5 206.4,198.2 208.7,200.8 210.9,203.3 213.0,205.7 215.0,207.9 216.9,210.1 218.8,212.1 220.6,214.1 222.3,216.1 224.0,217.9 225.6,219.7 227.2,221.5 228.7,223.2 230.3,224.9 231.9,226.5 233.5,228.1 235.0,229.6 236.6,231.2 238.1,232.7 239.6,234.2 241.1,235.6 242.6,237.1 244.1,238.6 245.5,240.1 247.0,241.5 248.5,243.0 250.1,244.5 251.6,246.0 253.1,247.6 254.7,249.1 256.3,250.7 258.0,252.3 259.6,254.0 261.4,255.7 263.1,257.4 265.0,259.3" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="349.2,152.6 347.3,150.7 345.4,148.8 343.5,146.9 341.6,145.1 339.8,143.2 337.9,141.4 336.1,139.5 334.2,137.7 332.3,135.8 330.4,133.9 328.5,132.0 326.5,130.1 324.6,128.2 322.5,126.2 320.5,124.2 318.4,122.1 316.2,119.9 314.0,117.7 311.7,115.5 309.3,113.1 306.8,110.7 304.3,108.2 301.6,105.5 298.7,102.7 295.8,99.8 292.6,96.7 289.3,93.4 285.7,89.9 281.9,86.2 277.8,82.1 273.3,77.8 268.5,73.1 263.2,67.9 257.4,62.2 250.9,55.8 243.6,48.8 235.5,40.8 226.3,32.2 216.2,23.4" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="32.4,24.6 55.9,51.8 75.1,73.8 90.9,91.9 104.2,106.9 115.5,119.7 125.3,130.6 133.8,140.2 141.3,148.5 148.0,156.0 154.0,162.6 159.4,168.6 164.3,174.0 168.8,179.0 172.9,183.6 176.8,187.8 180.4,191.8 183.7,195.5 186.8,198.9 189.8,202.2 192.6,205.2 195.3,208.2 197.8,210.9 200.3,213.6 202.6,216.2 204.8,218.6 207.0,221.0 209.1,223.3 211.2,225.6 213.2,227.7 215.1,229.9 217.0,232.0 218.9,234.0 220.8,236.0 222.6,238.0 224.4,240.0 226.2,241.9 228.0,243.9 229.8,245.8 231.6,247.7 233.5,249.5 235.4,251.4 237.2,253.2 239.1,255.1 241.0,257.0 242.9,258.9" fill="none" class="stroke-muted" stroke-width="1"/>
<rect x="30.0" y="20.0" width="320.0" height="240.0" fill="none" class="stroke-muted"/>
<text x="34.0" y="276.0" class="fig-small">w1</text>
<text x="24.0" y="32.0" class="fig-small" text-anchor="end">w2</text>
<polyline points="80.0,80.0 129.9,36.1 133.2,39.4 136.5,42.6 139.7,45.8 143.0,49.0 146.2,52.2 149.4,55.4 152.6,58.6 155.8,61.7 158.9,64.9 162.1,68.0 165.2,71.1 168.4,74.2 171.5,77.3 174.6,80.3 177.7,83.4 180.7,86.4 183.8,89.5 186.8,92.5 189.9,95.5 192.9,98.5 195.9,101.4 198.9,104.4 201.9,107.4 204.8,110.3" fill="none" class="stroke-d" stroke-width="6"/>
<polyline points="80.0,80.0 129.9,36.1 133.2,39.4 137.4,43.5 142.5,48.5 148.4,54.4 155.1,61.0 162.6,68.5 170.9,76.7 179.9,85.6 189.6,95.2 200.1,105.6 211.2,116.6 222.9,128.2 230.1,140.4 238.4,148.6 245.4,155.6 251.5,161.6 256.8,166.8 261.4,171.4 265.5,175.4 269.1,179.0 272.4,182.2 275.2,185.1 277.8,187.7 280.2,190.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="143.0" cy="49.0" r="4.5" class="fill-d"/>
<text x="132.0" y="53.0" class="fig-small fig-bold" text-anchor="end">5</text>
<circle cx="158.9" cy="64.9" r="4.5" class="fill-d"/>
<text x="147.9" y="68.9" class="fig-small fig-bold" text-anchor="end">10</text>
<circle cx="174.6" cy="80.3" r="4.5" class="fill-d"/>
<text x="163.6" y="84.3" class="fig-small fig-bold" text-anchor="end">15</text>
<circle cx="189.9" cy="95.5" r="4.5" class="fill-d"/>
<text x="178.9" y="99.5" class="fig-small fig-bold" text-anchor="end">20</text>
<circle cx="204.8" cy="110.3" r="4.5" class="fill-d"/>
<text x="193.8" y="114.3" class="fig-small fig-bold" text-anchor="end">25</text>
<circle cx="148.4" cy="54.4" r="4.5" class="fill-b"/>
<text x="159.4" y="52.4" class="fig-small fig-bold" text-anchor="start">5</text>
<circle cx="189.6" cy="95.2" r="4.5" class="fill-b"/>
<text x="200.6" y="93.2" class="fig-small fig-bold" text-anchor="start">10</text>
<circle cx="238.4" cy="148.6" r="4.5" class="fill-b"/>
<text x="249.4" y="146.6" class="fig-small fig-bold" text-anchor="start">15</text>
<circle cx="265.5" cy="175.4" r="4.5" class="fill-b"/>
<text x="276.5" y="173.4" class="fig-small fig-bold" text-anchor="start">20</text>
<circle cx="280.2" cy="190.0" r="4.5" class="fill-b"/>
<text x="291.2" y="188.0" class="fig-small fig-bold" text-anchor="start">25</text>
<circle cx="285.7" cy="195.5" r="5" class="fill-ink"/>
<circle cx="80.0" cy="80.0" r="5.5" class="fill-a"/>
<text x="90.0" y="84.0" class="fig-small">start</text>
<text x="295.7" y="217.5" class="fig-small">best weights</text>
<line x1="372.0" y1="40.0" x2="400.0" y2="40.0" class="stroke-d" stroke-width="3"/>
<text x="408.0" y="44.0" class="fig-small fig-bold">ISTA</text>
<line x1="372.0" y1="64.0" x2="400.0" y2="64.0" class="stroke-b" stroke-width="3"/>
<text x="408.0" y="68.0" class="fig-small fig-bold">FISTA</text>
<text x="372.0" y="100.0" class="fig-small">Numbers are step counts: ISTA on the left</text>
<text x="372.0" y="117.0" class="fig-small">of the path, FISTA on the right.</text>
<text x="372.0" y="134.0" class="fig-small">Gray lines: the score is 0.5, 2, 6, 14,</text>
<text x="372.0" y="151.0" class="fig-small">and 22 above its best value.</text>
<text x="372.0" y="185.0" class="fig-small">ISTA drops into the valley in one step,</text>
<text x="372.0" y="202.0" class="fig-small">then creeps along its floor.</text>
<text x="372.0" y="219.0" class="fig-small">FISTA gathers speed along the floor.</text>
<text x="372.0" y="253.0" class="fig-small">Score above best after 25 steps:</text>
<text x="372.0" y="270.0" class="fig-small">ISTA 1.29, FISTA 0.001</text>
</svg>
Figure 3. Contours of the lasso score for two correlated columns, with the paths of ISTA and FISTA from the same start. Both drop into the valley on the first step and run along the same floor, so the purple ISTA line hides under the orange one. Compare the dots: after 25 steps ISTA is still far from the best weights, while FISTA is almost there.
```

FISTA takes each step from a look-ahead point $z$, found by extending the last move. A counter $t$ starts at 1 and sets how far to extend:

$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \qquad z_{k+1} = w_{k+1} + \frac{t_k - 1}{t_{k+1}}\,(w_{k+1} - w_k)
$$

Read it as: take the ISTA step from $z$ to get $w_{k+1}$, then push $z$ past $w_{k+1}$ in the direction you just moved, by a fraction $(t_k - 1)/t_{k+1}$ of that move. The fraction starts at 0 and rises towards 1 (0, 0.28, 0.43, 0.53, and so on), so momentum builds as the walk keeps going in a consistent direction. The exact recurrence for $t$ is what makes the $1/k^2$ guarantee below work.

The column example converges in one step at $1/L$, so use a smaller step of 0.1 to see momentum work. ISTA moves 0, 0.4, 0.6, 0.7, 0.75. FISTA moves 0, 0.4, 0.6, 0.728, 0.792. The first two steps are equal because the fraction is 0 at step 1. By step 3 the look-ahead has pushed FISTA ahead, towards the answer 0.8, and by step 5 it has overshot to 0.813. FISTA's score goes 0.9129, 0.90016, then 0.90041 at step 5, so it rose. ISTA's score cannot rise, but FISTA's can, because momentum overshoots. That is why you test the finish with the KKT rules, not with "the score stopped falling".

```python
t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
z = w_new + ((t - 1) / t_new) * (w_new - w)     # uses the old w and the old t
```

Figure 4 compares the two on the exercise's 40-feature design, with $\lambda = 3$. After 80 steps ISTA's score is still 22.6 above its best, and FISTA's is 0.28 above. The guarantees agree: for a convex score and a step of at most $1/L$, the gap shrinks like $1/k$ for ISTA and $1/k^2$ for FISTA. For a gap ten times smaller, ISTA needs about ten times more steps and FISTA about three times more.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Score above its best value against step number on a log scale for ISTA and FISTA on a design with correlated columns. FISTA ends well below ISTA, with a few small bumps.">

<line x1="70.0" y1="220.0" x2="450.0" y2="220.0" class="stroke-muted" />
<line x1="70.0" y1="220.0" x2="70.0" y2="45.0" class="stroke-muted" />
<line x1="70.0" y1="220.0" x2="70.0" y2="224.0" class="stroke-muted" />
<text x="70.0" y="236.0" class="fig-small" text-anchor="middle">0</text>
<line x1="165.0" y1="220.0" x2="165.0" y2="224.0" class="stroke-muted" />
<text x="165.0" y="236.0" class="fig-small" text-anchor="middle">20</text>
<line x1="260.0" y1="220.0" x2="260.0" y2="224.0" class="stroke-muted" />
<text x="260.0" y="236.0" class="fig-small" text-anchor="middle">40</text>
<line x1="355.0" y1="220.0" x2="355.0" y2="224.0" class="stroke-muted" />
<text x="355.0" y="236.0" class="fig-small" text-anchor="middle">60</text>
<line x1="450.0" y1="220.0" x2="450.0" y2="224.0" class="stroke-muted" />
<text x="450.0" y="236.0" class="fig-small" text-anchor="middle">80</text>
<line x1="70.0" y1="220.0" x2="450.0" y2="220.0" class="stroke-grid" />
<line x1="66.0" y1="220.0" x2="70.0" y2="220.0" class="stroke-muted" />
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="70.0" y1="185.0" x2="450.0" y2="185.0" class="stroke-grid" />
<line x1="66.0" y1="185.0" x2="70.0" y2="185.0" class="stroke-muted" />
<text x="62.0" y="189.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="150.0" x2="450.0" y2="150.0" class="stroke-grid" />
<line x1="66.0" y1="150.0" x2="70.0" y2="150.0" class="stroke-muted" />
<text x="62.0" y="154.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="115.0" x2="450.0" y2="115.0" class="stroke-grid" />
<line x1="66.0" y1="115.0" x2="70.0" y2="115.0" class="stroke-muted" />
<text x="62.0" y="119.0" class="fig-small" text-anchor="end">100</text>
<line x1="70.0" y1="80.0" x2="450.0" y2="80.0" class="stroke-grid" />
<line x1="66.0" y1="80.0" x2="70.0" y2="80.0" class="stroke-muted" />
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">1000</text>
<line x1="70.0" y1="45.0" x2="450.0" y2="45.0" class="stroke-grid" />
<line x1="66.0" y1="45.0" x2="70.0" y2="45.0" class="stroke-muted" />
<text x="62.0" y="49.0" class="fig-small" text-anchor="end">10000</text>
<text x="260.0" y="252.0" class="fig-small" text-anchor="middle">step</text>
<text x="70.0" y="24.0" class="fig-bold">Score above its best value (log scale)</text>
<polyline points="70.0,74.7 74.8,100.0 79.5,115.1 84.2,122.4 89.0,125.3 93.8,126.6 98.5,127.1 103.2,127.4 108.0,127.7 112.8,127.9 117.5,128.0 122.2,128.2 127.0,128.3 131.8,128.5 136.5,128.7 141.2,128.8 146.0,129.0 150.8,129.1 155.5,129.3 160.2,129.4 165.0,129.6 169.8,129.7 174.5,129.9 179.2,130.0 184.0,130.2 188.8,130.3 193.5,130.5 198.2,130.6 203.0,130.8 207.8,130.9 212.5,131.1 217.2,131.2 222.0,131.4 226.8,131.5 231.5,131.7 236.2,131.8 241.0,131.9 245.8,132.1 250.5,132.2 255.2,132.3 260.0,132.5 264.8,132.6 269.5,132.7 274.2,132.9 279.0,133.0 283.8,133.1 288.5,133.3 293.2,133.4 298.0,133.5 302.8,133.7 307.5,133.8 312.2,133.9 317.0,134.1 321.8,134.2 326.5,134.3 331.2,134.5 336.0,134.6 340.8,134.7 345.5,134.8 350.2,135.0 355.0,135.1 359.8,135.2 364.5,135.4 369.2,135.5 374.0,135.6 378.8,135.7 383.5,135.9 388.2,136.0 393.0,136.1 397.8,136.3 402.5,136.4 407.2,136.5 412.0,136.6 416.8,136.8 421.5,136.9 426.2,137.0 431.0,137.1 435.8,137.2 440.5,137.4 445.2,137.5 450.0,137.6" fill="none" class="stroke-d" stroke-width="3"/>
<polyline points="70.0,74.7 74.8,100.0 79.5,115.1 84.2,124.0 89.0,126.8 93.8,127.4 98.5,127.8 103.2,128.2 108.0,128.6 112.8,129.1 117.5,129.6 122.2,130.1 127.0,130.7 131.8,131.3 136.5,131.9 141.2,132.5 146.0,133.2 150.8,133.9 155.5,134.6 160.2,135.4 165.0,136.2 169.8,137.0 174.5,137.9 179.2,138.7 184.0,139.6 188.8,140.6 193.5,141.5 198.2,142.5 203.0,143.4 207.8,144.4 212.5,145.4 217.2,146.3 222.0,147.3 226.8,148.3 231.5,149.4 236.2,150.5 241.0,151.6 245.8,152.7 250.5,153.8 255.2,155.0 260.0,156.1 264.8,157.2 269.5,158.4 274.2,159.6 279.0,160.9 283.8,162.1 288.5,163.4 293.2,164.7 298.0,166.2 302.8,167.5 307.5,168.6 312.2,169.7 317.0,170.9 321.8,172.1 326.5,173.3 331.2,174.6 336.0,176.2 340.8,178.0 345.5,180.0 350.2,181.8 355.0,183.6 359.8,185.4 364.5,187.3 369.2,189.4 374.0,191.3 378.8,193.5 383.5,196.1 388.2,198.6 393.0,199.2 397.8,199.4 402.5,199.4 407.2,198.8 412.0,198.6 416.8,198.7 421.5,199.3 426.2,200.8 431.0,203.2 435.8,206.8 440.5,209.5 445.2,207.1 450.0,204.1" fill="none" class="stroke-b" stroke-width="3"/>
<text x="460.0" y="131.6" class="fig-small fig-bold">ISTA</text>
<text x="460.0" y="146.6" class="fig-small">22.6 above best</text>
<text x="460.0" y="161.6" class="fig-small">(score 37.95)</text>
<text x="460.0" y="180.1" class="fig-small fig-bold">FISTA</text>
<text x="460.0" y="195.1" class="fig-small">0.28 above best</text>
<text x="460.0" y="210.1" class="fig-small">(score 15.66)</text>
<text x="98.5" y="199.0" class="fig-small">Small bumps near steps 71 and 79:</text>
<text x="98.5" y="209.5" class="fig-small">momentum can make the score tick up.</text>
</svg>
Figure 4. How far each method's score is above its best value, on 40 correlated features with lambda 3. After 80 steps FISTA's score is 15.66 and ISTA's is 37.95, a ratio of 0.41. The small bumps on FISTA's curve come from momentum overshooting.
```

> [!KEY]
> The lasso's corner stops you setting the slope to zero, and a plain walk bounces around 0. ISTA splits the job: step down the smooth squared error, then soft-threshold at step times lambda, which writes exact zeros. The KKT rules say when you are done. FISTA adds a growing look-ahead that speeds up the walk along narrow valleys, at the cost of a score that can tick up.

> [!RECAP]
> - The pull of a fee is its slope. Ridge's slope $w$ vanishes at 0, and the lasso's slope $\lambda$ does not.
> - Soft thresholding $S_t(x) = \mathrm{sign}(x)\max(|x| - t, 0)$ exactly minimises $\tfrac12(z-x)^2 + t|z|$. The lasso score is $\tfrac12\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert_1$, with no half on the fee, and for $X = I$ the answer is $S_\lambda(y)$.
> - ISTA: $w \leftarrow S_{\alpha\lambda}(w - \alpha X^\top(Xw - y))$ with $\alpha = 1/L$. On the column example it goes from 0 to 0.8, with score 0.9.
> - KKT: $g_i + \lambda\,\mathrm{sign}(w_i) = 0$ for nonzero weights, and $|g_i| \le \lambda$ for zero weights. The violation is the largest breach, and it is 0 at the answer.
> - FISTA takes the same step from a look-ahead $z$, with the counter $t$ starting at 1. Its score may rise.

## Build it step by step

The starter provides `lipschitz_constant`, which returns the largest eigenvalue of $X^\top X$. You write six functions, in the order the checks test them.

1. **`soft_threshold(x, thresh)`** returns an array with the shape of `x`. Use `np.sign`, `np.abs`, and `np.maximum`. The common mistake is `np.max`, which collapses the array to one number instead of comparing entry by entry.
2. **`lasso_objective(X, y, w, lam)`** returns a float. Compute half the squared misses plus `lam * np.sum(np.abs(w))`. Putting a half on the fee turns 3 into 2.5 on the identity example.
3. **`lasso_identity(y, lam)`** returns an array of the same shape as `y`, with no loop. Dividing `y` by `1 + lam`, as ridge does, gives $(1.5, 0.5)$ and not $(2, 0)$.
4. **`ista(X, y, lam, step, iters, w0=None)`** returns `(w, history)`. `w` has shape `(d,)`, and `history` has length `iters + 1` with the starting score at index 0. Copy `w0`, or use zeros, so you never change the caller's array. Thresholding at `lam` turns 0.8 into 0 on the column example. The threshold is `step * lam`.
5. **`kkt_violation(X, y, w, lam)`** returns a float. Compute `g`, choose between the two breaches with `np.where(w != 0, ..., ...)`, and return the largest. Testing only `|g| <= lam` accepts a nonzero weight whose pull has not cancelled.
6. **`fista(X, y, lam, step, iters, w0=None)`** returns `(w, history)` with the same shapes. Start `t` at 1 and `z` at a copy of `w`. Take the gradient at `z`, not at `w`, record the score at `w`, and form `z` before you overwrite `w` and `t`, or `w_new - w` becomes 0 and momentum disappears.

## Where this breaks

- **A fee that is too large.** Any $\lambda$ at least $\max|X^\top y|$ sets every weight to 0, and both methods return it. For the identity design that is $\max|y_i|$, and on the column example it is 5. Pick $\lambda$ on held-out data, as with ridge.
- **Columns on different scales.** The fee charges all weights in the same currency, so a column in tiny units needs a huge weight and is the first to be killed. Standardise the columns first.
- **Correlated columns.** When two columns carry the same signal, the lasso may keep either one, and a small change in the data can swap which one survives. Do not read a zero as proof that a feature is useless.
- **A step above $1/L$ or a score-based stop.** The guarantee that ISTA's score never rises needs a step of at most $1/L$, and above $2/L$ the walk can blow up. Compute $L$ from the largest eigenvalue of $X^\top X$. FISTA's score can rise, so stop when the KKT violation is small.

A duality gap that tells FISTA when to stop is in [convex duality](/lessons/019-convex-duality).

## The exercise

You build `soft_threshold`, `lasso_objective`, `lasso_identity`, `ista`, `kkt_violation`, and `fista`.

The checks confirm the hand values for `soft_threshold`, that it is odd and returns exact zeros inside the threshold, and that it matches a grid search of $\tfrac12(z-x)^2 + t|z|$. They score the identity example at 4, 3.25, and 3 and confirm that no nearby weights beat `lasso_identity`. One ISTA step must go from 0 to 0.8 and the score from 2.5 to 0.9, the score must never rise at step $1/L$, and thresholding must write exact zeros. `kkt_violation` must give the hand values 0 and 4. For FISTA, three steps at step 0.1 must reach 0.728, its score after 80 steps must be under 0.7 times ISTA's, and 800 steps must end with a tiny KKT violation.
