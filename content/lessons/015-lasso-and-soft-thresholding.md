---
title: "Lasso and soft thresholding: why some weights become exactly zero"
slug: 015-lasso-and-soft-thresholding
part: "Part 1 — Foundations"
summary: Charging the absolute size of a weight, not its square, keeps the pull constant down to zero, so weak features switch off exactly.
minutes: 40
gpu: false
objectives:
  - Explain why a fee on squared weights shrinks them but never zeroes them, and why a fee on absolute weights does.
  - Build soft thresholding and say what it computes in plain words.
  - Solve the lasso exactly when the design is the identity, and compare it with ridge.
prereqs:
  - 009-walking-downhill
  - 013-ridge-penalty-and-closed-form
  - 014-ridge-svd-path-and-degrees-of-freedom
exercise: 015-lasso-and-soft-thresholding
---

# Lasso and soft thresholding: why some weights become exactly zero

You have 40 features and you suspect only 5 of them matter. You fit ridge. It shrinks all 40 weights, and none of them reaches zero. You are left with 40 small but nonzero weights. You cannot say which features matter, and every prediction still needs all 40 measurements.

You might try rounding the small ridge weights down to 0 afterwards. The cut-off would be arbitrary, and the surviving weights were fit on the assumption that the others were present, so they would need refitting. A better plan is to change the fee itself, so that a feature with a weak signal ends up at exactly 0 as part of the fit.

That fee is the absolute size of each weight, and the model it produces is the *lasso*. The change looks small, and Figure 1 shows why it matters so much: ridge's pull on a weight fades as the weight approaches 0, while the lasso's pull does not.

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

## What you already know

- Ridge charges $\tfrac12\lambda w^2$ per weight, as in the chapter *Ridge: a fee on large weights*. The answer sits where the data's pull and the fee's pull balance.
- The pull of a fee on a weight is its slope, as in *Walking downhill*.
- When the design is the identity, the predictions are the weights, so each miss is $w_i - y_i$.
- The absolute value $|w|$ is the distance from 0. Its slope is $-1$ to the left of 0 and $+1$ to the right, so its graph has a corner at 0.
- You choose the fee with held-out data, as in *Honest scores*.

## Why ridge cannot reach zero

Take a single weight. The data says the weight should be $x = 0.5$, so the squared miss is $\tfrac12(w - 0.5)^2$. Charge a fee with $\lambda = 1$. The ridge fee $\tfrac12 w^2$ has slope $w$, which is tiny for small $w$. A weight near 0 is nearly free to keep, so it stays a little above 0. The lasso fee $|w|$ has slope 1 at every positive $w$, however close to 0. Every unit of weight costs the same all the way down.

Compare the scores:

| Weight $w$ | Ridge score | Lasso score |
|---|---|---|
| 0 | 0.125 | 0.125 |
| 0.1 | 0.085 | 0.18 |
| 0.25 | 0.0625 | 0.28125 |

Ridge improves as the weight leaves 0, and its best weight is 0.25, half of the data's 0.5. The lasso score gets worse at once, so its best weight is exactly 0. The reason is the tug of war in Figure 1. The data's pull on this weight never exceeds 0.5, and the lasso fee pulls back with 1. A fee-pull of 1 against a data-pull of at most 0.5 can only end at 0.

The rule: the lasso zeroes a weight whenever the data's pull at 0 is no larger than $\lambda$. Ridge zeroes a weight only if the data's pull at 0 is exactly 0.

## Solve one number: soft thresholding

When the design is the identity, the lasso score splits into one separate problem for each weight. For one number $x$ and fee $t$, you want the $z$ that minimises $\tfrac12(z - x)^2 + t\,|z|$: stay close to $x$, but pay $t$ for every unit of distance from 0. The answer to that question is called the *proximal operator* of the fee, and for the absolute value it has a simple form.

Try $t = 1$. For $x = 3$ the candidates cost the following. Staying at 3 costs $0 + 3 = 3$. Moving to 2 costs $\tfrac12(1)^2 + 2 = 2.5$. Moving to 0 costs $\tfrac12(9) + 0 = 4.5$. So 2 wins, which is $3 - 1$. For $x = 1$, staying costs $0 + 1 = 1$, moving to 0.5 costs $0.125 + 0.5 = 0.625$, and 0 costs $0.5$. So 0 wins.

The pattern is: move $x$ towards 0 by $t$, and if that would cross 0, stop at 0. Here is the reason. For $z > 0$ the slope of the cost is $(z - x) + t$, which is zero at $z = x - t$. That point is only valid when it is positive, so when $x \le t$ the slope is still positive at every $z > 0$ and the cost is lowest at the corner, $z = 0$. As a formula, this is *soft thresholding*:

$$
S_t(x) = \mathrm{sign}(x)\,\max(|x| - t,\; 0)
$$

Read it as: take the size of $x$, subtract $t$, floor the result at 0, and put the sign back. At $t = 1$ it sends $3, -3, 0.5, -0.5, 0, 2$ to $2, -2, 0, 0, 0, 1$. Figure 2 draws it: a line parallel to the diagonal but one unit closer to 0, with a flat stretch at 0 where small inputs are killed.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="The soft-threshold graph for threshold 1: a flat zero stretch between -1 and 1, then lines that run parallel to the diagonal but one unit closer to zero.">
<defs><marker id="c015-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="60.0" y1="125.0" x2="290.0" y2="125.0" class="stroke-muted" />
<line x1="175.0" y1="235.0" x2="175.0" y2="15.0" class="stroke-muted" />
<text x="290.0" y="141.0" class="fig-small" text-anchor="end">x</text>
<text x="181.0" y="25.0" class="fig-small">S(x)</text>
<rect x="146.2" y="15.0" width="57.5" height="220.0" class="fill-soft-b"/>
<line x1="60.0" y1="125.0" x2="290.0" y2="125.0" class="stroke-muted" />
<line x1="175.0" y1="235.0" x2="175.0" y2="15.0" class="stroke-muted" />
<line x1="60.0" y1="235.0" x2="290.0" y2="15.0" class="stroke-grid dashed" />
<polyline points="60.0,207.5 60.6,206.9 61.1,206.4 61.7,205.8 62.3,205.3 62.9,204.8 63.5,204.2 64.0,203.7 64.6,203.1 65.2,202.6 65.8,202.0 66.3,201.4 66.9,200.9 67.5,200.4 68.1,199.8 68.6,199.2 69.2,198.7 69.8,198.2 70.3,197.6 70.9,197.1 71.5,196.5 72.1,195.9 72.7,195.4 73.2,194.8 73.8,194.3 74.4,193.8 75.0,193.2 75.5,192.7 76.1,192.1 76.7,191.6 77.2,191.0 77.8,190.4 78.4,189.9 79.0,189.3 79.6,188.8 80.1,188.2 80.7,187.7 81.3,187.1 81.8,186.6 82.4,186.0 83.0,185.5 83.6,184.9 84.1,184.4 84.7,183.8 85.3,183.3 85.9,182.8 86.5,182.2 87.0,181.7 87.6,181.1 88.2,180.6 88.8,180.0 89.3,179.4 89.9,178.9 90.5,178.3 91.0,177.8 91.6,177.2 92.2,176.7 92.8,176.2 93.3,175.6 93.9,175.1 94.5,174.5 95.1,174.0 95.7,173.4 96.2,172.9 96.8,172.3 97.4,171.8 98.0,171.2 98.5,170.7 99.1,170.1 99.7,169.6 100.2,169.0 100.8,168.4 101.4,167.9 102.0,167.3 102.5,166.8 103.1,166.2 103.7,165.7 104.3,165.2 104.8,164.6 105.4,164.1 106.0,163.5 106.6,162.9 107.2,162.4 107.7,161.8 108.3,161.3 108.9,160.8 109.4,160.2 110.0,159.6 110.6,159.1 111.2,158.6 111.8,158.0 112.3,157.4 112.9,156.9 113.5,156.3 114.0,155.8 114.6,155.2 115.2,154.7 115.8,154.2 116.3,153.6 116.9,153.1 117.5,152.5 118.1,151.9 118.7,151.4 119.2,150.8 119.8,150.3 120.4,149.8 121.0,149.2 121.5,148.6 122.1,148.1 122.7,147.6 123.2,147.0 123.8,146.4 124.4,145.9 125.0,145.3 125.6,144.8 126.1,144.2 126.7,143.7 127.3,143.2 127.8,142.6 128.4,142.1 129.0,141.5 129.6,140.9 130.1,140.4 130.7,139.8 131.3,139.3 131.9,138.8 132.4,138.2 133.0,137.7 133.6,137.1 134.2,136.6 134.8,136.0 135.3,135.4 135.9,134.9 136.5,134.3 137.1,133.8 137.6,133.2 138.2,132.7 138.8,132.1 139.4,131.6 139.9,131.1 140.5,130.5 141.1,129.9 141.6,129.4 142.2,128.9 142.8,128.3 143.4,127.8 143.9,127.2 144.5,126.7 145.1,126.1 145.7,125.5 146.2,125.0 146.8,125.0 147.4,125.0 148.0,125.0 148.6,125.0 149.1,125.0 149.7,125.0 150.3,125.0 150.9,125.0 151.4,125.0 152.0,125.0 152.6,125.0 153.2,125.0 153.7,125.0 154.3,125.0 154.9,125.0 155.4,125.0 156.0,125.0 156.6,125.0 157.2,125.0 157.8,125.0 158.3,125.0 158.9,125.0 159.5,125.0 160.1,125.0 160.6,125.0 161.2,125.0 161.8,125.0 162.4,125.0 162.9,125.0 163.5,125.0 164.1,125.0 164.7,125.0 165.2,125.0 165.8,125.0 166.4,125.0 166.9,125.0 167.5,125.0 168.1,125.0 168.7,125.0 169.2,125.0 169.8,125.0 170.4,125.0 171.0,125.0 171.6,125.0 172.1,125.0 172.7,125.0 173.3,125.0 173.8,125.0 174.4,125.0 175.0,125.0 175.6,125.0 176.2,125.0 176.7,125.0 177.3,125.0 177.9,125.0 178.4,125.0 179.0,125.0 179.6,125.0 180.2,125.0 180.8,125.0 181.3,125.0 181.9,125.0 182.5,125.0 183.1,125.0 183.6,125.0 184.2,125.0 184.8,125.0 185.4,125.0 185.9,125.0 186.5,125.0 187.1,125.0 187.7,125.0 188.2,125.0 188.8,125.0 189.4,125.0 190.0,125.0 190.5,125.0 191.1,125.0 191.7,125.0 192.3,125.0 192.8,125.0 193.4,125.0 194.0,125.0 194.5,125.0 195.1,125.0 195.7,125.0 196.3,125.0 196.8,125.0 197.4,125.0 198.0,125.0 198.6,125.0 199.2,125.0 199.7,125.0 200.3,125.0 200.9,125.0 201.4,125.0 202.0,125.0 202.6,125.0 203.2,125.0 203.8,125.0 204.3,124.4 204.9,123.9 205.5,123.3 206.1,122.8 206.6,122.2 207.2,121.7 207.8,121.2 208.3,120.6 208.9,120.1 209.5,119.5 210.1,119.0 210.7,118.4 211.2,117.9 211.8,117.3 212.4,116.8 213.0,116.2 213.5,115.7 214.1,115.1 214.7,114.5 215.2,114.0 215.8,113.5 216.4,112.9 217.0,112.3 217.6,111.8 218.1,111.2 218.7,110.7 219.3,110.2 219.9,109.6 220.4,109.0 221.0,108.5 221.6,108.0 222.1,107.4 222.7,106.8 223.3,106.3 223.9,105.8 224.4,105.2 225.0,104.7 225.6,104.1 226.2,103.5 226.8,103.0 227.3,102.4 227.9,101.9 228.5,101.3 229.0,100.8 229.6,100.2 230.2,99.7 230.8,99.1 231.3,98.6 231.9,98.0 232.5,97.5 233.1,96.9 233.7,96.4 234.2,95.8 234.8,95.3 235.4,94.7 236.0,94.2 236.5,93.6 237.1,93.1 237.7,92.6 238.2,92.0 238.8,91.5 239.4,90.9 240.0,90.3 240.6,89.8 241.1,89.2 241.7,88.7 242.3,88.2 242.9,87.6 243.4,87.1 244.0,86.5 244.6,85.9 245.2,85.4 245.7,84.8 246.3,84.3 246.9,83.8 247.5,83.2 248.0,82.7 248.6,82.1 249.2,81.6 249.8,81.0 250.3,80.4 250.9,79.9 251.5,79.3 252.0,78.8 252.6,78.2 253.2,77.7 253.8,77.2 254.3,76.6 254.9,76.0 255.5,75.5 256.1,74.9 256.6,74.4 257.2,73.8 257.8,73.3 258.4,72.8 258.9,72.2 259.5,71.6 260.1,71.1 260.7,70.5 261.2,70.0 261.8,69.4 262.4,68.9 263.0,68.3 263.6,67.8 264.1,67.2 264.7,66.7 265.3,66.1 265.9,65.6 266.4,65.1 267.0,64.5 267.6,64.0 268.1,63.4 268.7,62.8 269.3,62.3 269.9,61.8 270.5,61.2 271.0,60.7 271.6,60.1 272.2,59.6 272.8,59.0 273.3,58.4 273.9,57.9 274.5,57.3 275.1,56.8 275.6,56.2 276.2,55.7 276.8,55.2 277.4,54.6 277.9,54.1 278.5,53.5 279.1,52.9 279.6,52.4 280.2,51.8 280.8,51.3 281.4,50.8 281.9,50.2 282.5,49.7 283.1,49.1 283.7,48.5 284.2,48.0 284.8,47.4 285.4,46.9 286.0,46.3 286.5,45.8 287.1,45.2 287.7,44.7 288.3,44.1 288.9,43.6 289.4,43.0 290.0,42.5" fill="none" class="stroke-b" stroke-width="3.5"/>
<line x1="261.2" y1="42.5" x2="261.2" y2="63.0" class="stroke-ink" stroke-width="1.8" marker-end="url(#c015-arrow)"/>
<circle cx="261.2" cy="42.5" r="4" class="fill-a"/>
<circle cx="261.2" cy="70.0" r="4.5" class="fill-b"/>
<line x1="88.8" y1="207.5" x2="88.8" y2="187.0" class="stroke-ink" stroke-width="1.8" marker-end="url(#c015-arrow)"/>
<circle cx="88.8" cy="207.5" r="4" class="fill-a"/>
<circle cx="88.8" cy="180.0" r="4.5" class="fill-b"/>
<line x1="198.0" y1="103.0" x2="198.0" y2="118.0" class="stroke-ink" stroke-width="1.8" marker-end="url(#c015-arrow)"/>
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

Read it as: half the squared misses, plus lambda times the sum of the absolute weights. There is no half on the fee here, so that the threshold in the one-number problem is exactly $\lambda$.

With $X = I$ the misses are $w_i - y_i$, so the score is a sum of one-number problems and the answer is $w = S_\lambda(y)$. On the running example $y = (3, 1)$ and $\lambda = 1$, the lasso gives $(2, 0)$ and ridge gives $(1.5, 0.5)$. The lasso scores $\tfrac12(1 + 1) + 2 = 3$ at its own answer, but $1.25 + 2 = 3.25$ at ridge's answer. Figure 3 shows the same two answers in the plane of the weights. The blue circles hold the weights that miss $y$ by the same amount, so the best answer is where the smallest blue circle just touches the fee's shape: a diamond for the lasso (weights whose absolute sizes add to 2) and a circle for ridge (weights of one fixed length).

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="In the plane of two weights, circles around the unconstrained answer (3, 1), a green diamond for the lasso fee whose corner (2, 0) is on an axis, and a purple circle for the ridge fee touching at (1.5, 0.5).">

<line x1="-1.2" y1="135.0" x2="311.6" y2="135.0" class="stroke-muted" />
<line x1="100.0" y1="236.2" x2="100.0" y2="29.2" class="stroke-muted" />
<text x="311.6" y="151.0" class="fig-small" text-anchor="end">w1</text>
<text x="106.0" y="39.2" class="fig-small">w2</text>
<polyline points="303.1,89.0 303.0,86.4 302.9,83.9 302.6,81.3 302.2,78.8 301.8,76.2 301.2,73.7 300.6,71.2 299.8,68.8 299.0,66.3 298.0,64.0 297.0,61.6 295.9,59.3 294.7,57.0 293.3,54.8 292.0,52.7 290.5,50.6 288.9,48.5 287.3,46.5 285.6,44.6 283.8,42.8 281.9,41.0 280.0,39.3 278.0,37.7 275.9,36.1 273.8,34.7 271.6,33.3 269.4,32.0 267.1,30.8 264.8,29.7 262.4,28.7 260.0,27.8 257.6,27.0 255.1,26.2 252.7,25.6 250.1,25.1 247.6,24.7 245.1,24.3 242.5,24.1 239.9,24.0 237.4,23.9 234.8,24.0 232.2,24.2 229.7,24.5 227.1,24.9 224.6,25.3 222.1,25.9 219.6,26.6 217.2,27.4 214.7,28.2 212.4,29.2 210.0,30.3 207.7,31.4 205.5,32.7 203.3,34.0 201.1,35.4 199.0,36.9 197.0,38.5 195.0,40.1 193.2,41.9 191.3,43.7 189.6,45.6 187.9,47.5 186.3,49.5 184.8,51.6 183.3,53.7 182.0,55.9 180.7,58.2 179.5,60.4 178.5,62.8 177.5,65.1 176.6,67.6 175.8,70.0 175.1,72.5 174.5,75.0 174.0,77.5 173.6,80.0 173.3,82.6 173.1,85.1 173.0,87.7 173.0,90.3 173.1,92.9 173.3,95.4 173.6,98.0 174.0,100.5 174.5,103.0 175.1,105.5 175.8,108.0 176.6,110.4 177.5,112.9 178.5,115.2 179.5,117.6 180.7,119.8 182.0,122.1 183.3,124.3 184.8,126.4 186.3,128.5 187.9,130.5 189.6,132.4 191.3,134.3 193.2,136.1 195.0,137.9 197.0,139.5 199.0,141.1 201.1,142.6 203.3,144.0 205.5,145.3 207.7,146.6 210.0,147.7 212.4,148.8 214.7,149.8 217.2,150.6 219.6,151.4 222.1,152.1 224.6,152.7 227.1,153.1 229.7,153.5 232.2,153.8 234.8,154.0 237.4,154.1 239.9,154.0 242.5,153.9 245.1,153.7 247.6,153.3 250.1,152.9 252.7,152.4 255.1,151.8 257.6,151.0 260.0,150.2 262.4,149.3 264.8,148.3 267.1,147.2 269.4,146.0 271.6,144.7 273.8,143.3 275.9,141.9 278.0,140.3 280.0,138.7 281.9,137.0 283.8,135.2 285.6,133.4 287.3,131.5 288.9,129.5 290.5,127.4 292.0,125.3 293.3,123.2 294.7,121.0 295.9,118.7 297.0,116.4 298.0,114.0 299.0,111.7 299.8,109.2 300.6,106.8 301.2,104.3 301.8,101.8 302.2,99.2 302.6,96.7 302.9,94.1 303.0,91.6 303.1,89.0" fill="none" class="stroke-a" stroke-width="2"/>
<polyline points="310.7,89.0 310.7,86.1 310.5,83.3 310.2,80.4 309.8,77.6 309.3,74.7 308.7,71.9 308.0,69.1 307.1,66.4 306.2,63.7 305.1,61.0 304.0,58.4 302.7,55.8 301.3,53.3 299.9,50.8 298.3,48.4 296.7,46.0 294.9,43.7 293.1,41.5 291.2,39.4 289.2,37.3 287.1,35.3 284.9,33.4 282.7,31.6 280.4,29.9 278.0,28.3 275.6,26.7 273.1,25.3 270.6,24.0 268.0,22.7 265.3,21.6 262.7,20.6 259.9,19.7 257.2,18.8 254.4,18.1 251.6,17.5 248.7,17.1 245.9,16.7 243.0,16.4 240.2,16.3 237.3,16.3 234.4,16.4 231.5,16.6 228.7,16.9 225.8,17.3 223.0,17.8 220.2,18.5 217.4,19.2 214.7,20.1 212.0,21.1 209.3,22.2 206.7,23.3 204.2,24.6 201.6,26.0 199.2,27.5 196.8,29.1 194.4,30.8 192.2,32.5 190.0,34.4 187.9,36.3 185.8,38.3 183.9,40.4 182.0,42.6 180.2,44.9 178.5,47.2 176.9,49.6 175.4,52.0 174.0,54.5 172.6,57.1 171.4,59.7 170.3,62.3 169.3,65.0 168.4,67.8 167.7,70.5 167.0,73.3 166.4,76.1 166.0,79.0 165.6,81.8 165.4,84.7 165.3,87.6 165.3,90.4 165.4,93.3 165.6,96.2 166.0,99.0 166.4,101.9 167.0,104.7 167.7,107.5 168.4,110.2 169.3,113.0 170.3,115.7 171.4,118.3 172.6,120.9 174.0,123.5 175.4,126.0 176.9,128.4 178.5,130.8 180.2,133.1 182.0,135.4 183.9,137.6 185.8,139.7 187.9,141.7 190.0,143.6 192.2,145.5 194.4,147.2 196.8,148.9 199.2,150.5 201.6,152.0 204.2,153.4 206.7,154.7 209.3,155.8 212.0,156.9 214.7,157.9 217.4,158.8 220.2,159.5 223.0,160.2 225.8,160.7 228.7,161.1 231.5,161.4 234.4,161.6 237.3,161.7 240.2,161.7 243.0,161.6 245.9,161.3 248.7,160.9 251.6,160.5 254.4,159.9 257.2,159.2 259.9,158.3 262.7,157.4 265.3,156.4 268.0,155.3 270.6,154.0 273.1,152.7 275.6,151.3 278.0,149.7 280.4,148.1 282.7,146.4 284.9,144.6 287.1,142.7 289.2,140.7 291.2,138.6 293.1,136.5 294.9,134.3 296.7,132.0 298.3,129.6 299.9,127.2 301.3,124.7 302.7,122.2 304.0,119.6 305.1,117.0 306.2,114.3 307.1,111.6 308.0,108.9 308.7,106.1 309.3,103.3 309.8,100.4 310.2,97.6 310.5,94.7 310.7,91.9 310.7,89.0" fill="none" class="stroke-a dashed" stroke-width="2"/>
<polyline points="172.7,135.0 172.7,132.1 172.5,129.3 172.2,126.4 171.8,123.6 171.3,120.7 170.7,117.9 170.0,115.1 169.1,112.4 168.2,109.7 167.1,107.0 166.0,104.4 164.7,101.8 163.3,99.3 161.9,96.8 160.3,94.4 158.7,92.0 156.9,89.7 155.1,87.5 153.2,85.4 151.2,83.3 149.1,81.3 146.9,79.4 144.7,77.6 142.4,75.9 140.0,74.3 137.6,72.7 135.1,71.3 132.6,70.0 130.0,68.7 127.3,67.6 124.7,66.6 121.9,65.7 119.2,64.8 116.4,64.1 113.6,63.5 110.7,63.1 107.9,62.7 105.0,62.4 102.2,62.3 99.3,62.3 96.4,62.4 93.5,62.6 90.7,62.9 87.8,63.3 85.0,63.8 82.2,64.5 79.4,65.2 76.7,66.1 74.0,67.1 71.3,68.2 68.7,69.3 66.2,70.6 63.6,72.0 61.2,73.5 58.8,75.1 56.4,76.8 54.2,78.5 52.0,80.4 49.9,82.3 47.8,84.3 45.9,86.4 44.0,88.6 42.2,90.9 40.5,93.2 38.9,95.6 37.4,98.0 36.0,100.5 34.6,103.1 33.4,105.7 32.3,108.3 31.3,111.0 30.4,113.8 29.7,116.5 29.0,119.3 28.4,122.1 28.0,125.0 27.6,127.8 27.4,130.7 27.3,133.6 27.3,136.4 27.4,139.3 27.6,142.2 28.0,145.0 28.4,147.9 29.0,150.7 29.7,153.5 30.4,156.2 31.3,159.0 32.3,161.7 33.4,164.3 34.6,166.9 36.0,169.5 37.4,172.0 38.9,174.4 40.5,176.8 42.2,179.1 44.0,181.4 45.9,183.6 47.8,185.7 49.9,187.7 52.0,189.6 54.2,191.5 56.4,193.2 58.8,194.9 61.2,196.5 63.6,198.0 66.2,199.4 68.7,200.7 71.3,201.8 74.0,202.9 76.7,203.9 79.4,204.8 82.2,205.5 85.0,206.2 87.8,206.7 90.7,207.1 93.5,207.4 96.4,207.6 99.3,207.7 102.2,207.7 105.0,207.6 107.9,207.3 110.7,206.9 113.6,206.5 116.4,205.9 119.2,205.2 121.9,204.3 124.7,203.4 127.3,202.4 130.0,201.3 132.6,200.0 135.1,198.7 137.6,197.3 140.0,195.7 142.4,194.1 144.7,192.4 146.9,190.6 149.1,188.7 151.2,186.7 153.2,184.6 155.1,182.5 156.9,180.3 158.7,178.0 160.3,175.6 161.9,173.2 163.3,170.7 164.7,168.2 166.0,165.6 167.1,163.0 168.2,160.3 169.1,157.6 170.0,154.9 170.7,152.1 171.3,149.3 171.8,146.4 172.2,143.6 172.5,140.7 172.7,137.9 172.7,135.0" fill="none" class="stroke-d" stroke-width="2.5"/>
<polyline points="192.0,135.0 100.0,43.0 8.0,135.0 100.0,227.0 192.0,135.0" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="238.0" cy="89.0" r="5" class="fill-a"/>
<text x="238.0" y="77.0" class="fig-small" text-anchor="middle">(3, 1)</text>
<circle cx="192.0" cy="135.0" r="5.5" class="fill-c"/>
<line x1="192.0" y1="142.0" x2="192.0" y2="175.0" class="stroke-muted" />
<text x="192.0" y="189.0" class="fig-small" text-anchor="middle">lasso (2, 0)</text>
<circle cx="169.0" cy="112.0" r="5.5" class="fill-d"/>
<text x="158.0" y="106.0" class="fig-small" text-anchor="end">ridge</text>
<text x="330.0" y="40.0" class="fig-small">Blue circles: weights that miss (3, 1)</text>
<text x="330.0" y="58.0" class="fig-small">by the same amount.</text>
<text x="330.0" y="88.0" class="fig-small">Green diamond: all weights whose absolute</text>
<text x="330.0" y="106.0" class="fig-small">sizes add to 2. Corners lie on the axes.</text>
<text x="330.0" y="136.0" class="fig-small">Purple circle: all weights of length 1.58.</text>
<text x="330.0" y="166.0" class="fig-small">The diamond's corner sticks out towards</text>
<text x="330.0" y="184.0" class="fig-small">the data, so the nearest point is the</text>
<text x="330.0" y="202.0" class="fig-small">corner, where w2 = 0. The circle has no</text>
<text x="330.0" y="220.0" class="fig-small">corner, so ridge stops at an interior</text>
<text x="330.0" y="238.0" class="fig-small">point, with both weights nonzero.</text>
</svg>
Figure 3. The lasso answer (2, 0) lies on the corner of the diamond, so the second weight is exactly 0. Ridge's answer (1.5, 0.5) lies at a smooth point of the circle, where both weights are nonzero. Both are the point of their shape that is nearest to (3, 1).
```

The second feature's data pull at 0 is 1, which is no larger than $\lambda = 1$, so it is killed. The first feature's pull is 3, which beats the fee, so it survives with 1 unit of weight taken off. A fee of $\lambda = 3$ or more would zero both weights.

```python
misses = X @ w - y
score = 0.5 * np.sum(misses ** 2) + lam * np.sum(np.abs(w))
```

For a general $X$ there is no formula, because the corner at 0 stops you from setting the gradient to zero. The next chapter fits that case. What you have here is the one-number step it will repeat.

## Copied columns, again

Take the copied columns of the ridge chapter, with $X$ rows $(1, 1)$ and $(2, 2)$, labels $(1, 2)$, and $\lambda = 1$. Ridge split the weight evenly. For the lasso, only the sum $w_1 + w_2$ affects the misses, and among non-negative weights the fee also depends only on the sum. So every non-negative split of the same sum scores the same, and $(0.8, 0)$, $(0.4, 0.4)$, and $(0, 0.8)$ all score 0.9. The lasso does not care how copies share a signal, so which split you get depends on the solver. Ridge always shares evenly.

> [!KEY]
> Ridge's pull on a weight fades as the weight nears 0, so a weak signal leaves a small nonzero weight. The lasso charges a constant pull of $\lambda$ per unit of weight, so any feature whose data pull at 0 is at most $\lambda$ stays at exactly 0. For one number this is soft thresholding: move towards 0 by $\lambda$, and stop at 0. That is how the lasso switches features off, and why 40 features with 5 that matter can give a model with only a handful of nonzero weights.

> [!RECAP]
> - The pull of a fee is its slope. Ridge's slope $w$ vanishes at 0, and the lasso's slope $\lambda$ does not.
> - Soft thresholding $S_t(x) = \mathrm{sign}(x)\max(|x| - t, 0)$ moves $x$ towards 0 by $t$ and stops at 0. At $t = 1$ it sends 3 to 2 and 0.5 to 0.
> - It exactly minimises $\tfrac12(z-x)^2 + t|z|$.
> - The lasso score is $\tfrac12\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert_1$, with no half on the fee.
> - When $X$ is the identity, the lasso answer is $S_\lambda(y)$. For $y = (3, 1)$ and $\lambda = 1$ that is $(2, 0)$, against ridge's $(1.5, 0.5)$.

## Build it step by step

You write three functions, in the order the checks test them.

1. `soft_threshold(x, thresh)` returns an array with the shape of `x`. Use `np.sign`, `np.abs`, and `np.maximum`. The common mistake is `np.max`, which collapses the array to one number instead of comparing entry by entry.

2. `lasso_objective(X, y, w, lam)` returns a float. Compute half the squared misses plus `lam * np.sum(np.abs(w))`. The common mistake is to put a half on the fee, which turns 3 into 2.5 on the running example.

3. `lasso_identity(y, lam)` returns an array of the same shape as `y`. You need no loop, since the problem splits into one number per weight. The common mistake is to solve it like ridge, by dividing `y` by `1 + lam`, which gives $(1.5, 0.5)$ and not $(2, 0)$.

## Where this breaks

- **A fee that is too large.** For the identity design, any $\lambda$ at least as big as $\max|y_i|$ sets every weight to 0. Pick $\lambda$ on held-out data, as with ridge.
- **Columns on different scales.** The fee charges all weights in the same currency, so a column in tiny units needs a huge weight and is the first to be killed. Standardise the columns first.
- **Correlated columns.** When two columns carry the same signal, the lasso may keep either one, as the copied-column example shows. A small change in the data can swap which one survives, so do not read a zero as proof that a feature is useless.

## The exercise

You build `soft_threshold`, `lasso_objective`, and `lasso_identity`. The checks confirm the hand values for `soft_threshold`, that it is odd and returns exact zeros inside the threshold, and that it minimises $\tfrac12(z-x)^2 + t|z|$ against a grid search. They score the running example at 4, 3.25, and 3, check the formula with no half on the fee, and confirm that no nearby weights beat `lasso_identity`.
