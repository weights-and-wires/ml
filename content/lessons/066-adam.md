---
title: "Adam: a step size for every weight"
slug: 066-adam
part: "Part 4 — Neural networks"
summary: Divide each weight's average gradient by its own typical size, so every weight moves about one learning rate per step.
minutes: 40
gpu: false
objectives:
  - Explain why one learning rate suits a weight with gradient 3 and a weight with gradient 3000 badly.
  - Build an exponential moving average, and say how many past steps it remembers.
  - Explain why the average starts too small, and correct it with a factor that fades.
  - Take an Adam step, storing the raw averages and the step count.
prereqs:
  - 012-momentum
  - 064-backprop-mlp-and-gradient-check
  - 065-stochastic-gradients
exercise: 066-adam
---

# Adam: a step size for every weight

A network has weights of very different kinds. A bias might see a gradient of 3, while a weight attached to a feature with huge values sees a gradient of 3000. Take a learning rate of 0.1. The first weight moves 0.3, which is sensible. The second moves 300, which throws the model across the map.

Lower the rate to suit the second weight and you get the opposite trouble. At 0.0001 the second weight moves 0.3, but the first moves 0.0003, and after a thousand steps it has barely left home. This is the long-valley problem from earlier chapters, with one learning rate squeezed between a steep axis and a flat one. How can the optimiser measure each weight's own scale and use it?

This chapter builds Adam, the optimiser that does that. It first gives every weight a moving average of its gradient and of its squared gradient. It then divides one by the other.

## What you already know

Five ideas carry this chapter:

- A step is the learning rate times the gradient, so a steep axis gets long steps and a flat axis gets short ones (see "Noisy gradients: stochastic gradient descent and mini-batches").
- One learning rate must stay below $2/L$, where $L$ is the steepest curvature (see "How big a step: curvature and the 2/L limit").
- Momentum keeps a running sum of past gradients and steps along it (see "Momentum: a heavy ball and a look-ahead").
- A mini-batch gradient is right on average and noisy at each step.
- Each weight has its own gradient, so every number below applies to each weight on its own.

## Give each weight its own ruler

Think of a thermostat in each room instead of one for the whole house. A weight with gradients near 3000 is a hot room, and one with gradients near 3 is a cold one. Measure each gradient in units of its own typical size and every weight speaks the same language. A gradient of 3 in the cold room and a gradient of 3000 in the hot room both read "about 1".

The step is then the learning rate times that unit-free number, so every weight moves about one learning rate per step, whatever its scale.

Why not divide each gradient by its own absolute value? That throws away the information you need. A weight whose gradient flips between +3 and −3 would move a full step each time, back and forth, although it has no direction. A better ruler divides the *average* gradient by the *typical size* of the gradient. A steady slope gives a ratio near ±1 and a long step. Noise that flips sign gives an average near 0 and a short step.

So Adam needs two memories per weight: an average of the gradient, and an average of its square.

## Remember with a moving average

An **exponential moving average** keeps a running value and, at each step, blends in the newest number. With decay $\beta$:

$$
\text{avg} \leftarrow \beta\,\text{avg} + (1 - \beta)\,x
$$

Read it aloud: keep a fraction $\beta$ of the old average and take $1 - \beta$ of the new number. Unrolled, the newest number has weight $1 - \beta$, the one before has $(1 - \beta)\beta$, and the one $k$ steps ago has $(1 - \beta)\beta^k$. The weights fade by the same factor each step and add up to nearly 1. After $1/(1 - \beta)$ steps the weight has fallen to about a third of the newest, so call that the **window**. Figure 1 shows both windows.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Weights that an exponential moving average gives to past gradients for beta 0.9 and beta 0.999">
<defs></defs>
<line x1="55.0" y1="230.0" x2="275.0" y2="230.0" class="stroke-muted"/>
<line x1="55.0" y1="40.0" x2="55.0" y2="230.0" class="stroke-muted"/>
<line x1="62.1" y1="230.0" x2="62.1" y2="235.0" class="stroke-muted"/>
<text x="62.1" y="249.0" class="fig-small" text-anchor="middle">0</text>
<line x1="133.1" y1="230.0" x2="133.1" y2="235.0" class="stroke-muted"/>
<text x="133.1" y="249.0" class="fig-small" text-anchor="middle">10</text>
<line x1="204.0" y1="230.0" x2="204.0" y2="235.0" class="stroke-muted"/>
<text x="204.0" y="249.0" class="fig-small" text-anchor="middle">20</text>
<line x1="275.0" y1="230.0" x2="275.0" y2="235.0" class="stroke-muted"/>
<text x="275.0" y="249.0" class="fig-small" text-anchor="middle">30</text>
<line x1="55.0" y1="230.0" x2="275.0" y2="230.0" class="stroke-grid"/>
<line x1="50.0" y1="230.0" x2="55.0" y2="230.0" class="stroke-muted"/>
<text x="46.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="55.0" y1="143.6" x2="275.0" y2="143.6" class="stroke-grid"/>
<line x1="50.0" y1="143.6" x2="55.0" y2="143.6" class="stroke-muted"/>
<text x="46.0" y="147.6" class="fig-small" text-anchor="end">0.05</text>
<line x1="55.0" y1="57.3" x2="275.0" y2="57.3" class="stroke-grid"/>
<line x1="50.0" y1="57.3" x2="55.0" y2="57.3" class="stroke-muted"/>
<text x="46.0" y="61.3" class="fig-small" text-anchor="end">0.1</text>
<text x="165.0" y="268.0" class="fig-small" text-anchor="middle">steps ago</text>
<rect x="59.6" y="57.3" width="5.0" height="172.7" class="fill-a"/>
<rect x="66.7" y="74.5" width="5.0" height="155.5" class="fill-a"/>
<rect x="73.8" y="90.1" width="5.0" height="139.9" class="fill-a"/>
<rect x="80.9" y="104.1" width="5.0" height="125.9" class="fill-a"/>
<rect x="88.0" y="116.7" width="5.0" height="113.3" class="fill-a"/>
<rect x="95.1" y="128.0" width="5.0" height="102.0" class="fill-a"/>
<rect x="102.2" y="138.2" width="5.0" height="91.8" class="fill-a"/>
<rect x="109.3" y="147.4" width="5.0" height="82.6" class="fill-a"/>
<rect x="116.4" y="155.6" width="5.0" height="74.4" class="fill-a"/>
<rect x="123.5" y="163.1" width="5.0" height="66.9" class="fill-a"/>
<rect x="130.6" y="169.8" width="5.0" height="60.2" class="fill-a"/>
<rect x="137.7" y="175.8" width="5.0" height="54.2" class="fill-a"/>
<rect x="144.8" y="181.2" width="5.0" height="48.8" class="fill-a"/>
<rect x="151.9" y="186.1" width="5.0" height="43.9" class="fill-a"/>
<rect x="159.0" y="190.5" width="5.0" height="39.5" class="fill-a"/>
<rect x="166.1" y="194.4" width="5.0" height="35.6" class="fill-a"/>
<rect x="173.2" y="198.0" width="5.0" height="32.0" class="fill-a"/>
<rect x="180.3" y="201.2" width="5.0" height="28.8" class="fill-a"/>
<rect x="187.4" y="204.1" width="5.0" height="25.9" class="fill-a"/>
<rect x="194.5" y="206.7" width="5.0" height="23.3" class="fill-a"/>
<rect x="201.5" y="209.0" width="5.0" height="21.0" class="fill-a"/>
<rect x="208.6" y="211.1" width="5.0" height="18.9" class="fill-a"/>
<rect x="215.7" y="213.0" width="5.0" height="17.0" class="fill-a"/>
<rect x="222.8" y="214.7" width="5.0" height="15.3" class="fill-a"/>
<rect x="229.9" y="216.2" width="5.0" height="13.8" class="fill-a"/>
<rect x="237.0" y="217.6" width="5.0" height="12.4" class="fill-a"/>
<rect x="244.1" y="218.8" width="5.0" height="11.2" class="fill-a"/>
<rect x="251.2" y="220.0" width="5.0" height="10.0" class="fill-a"/>
<rect x="258.3" y="221.0" width="5.0" height="9.0" class="fill-a"/>
<rect x="265.4" y="221.9" width="5.0" height="8.1" class="fill-a"/>
<line x1="133.1" y1="60.0" x2="133.1" y2="230.0" class="stroke-muted dashed"/>
<text x="139.1" y="70.0" class="fig-small" text-anchor="start">window: 10</text>
<text x="165.0" y="22.0" class="fig-bold" text-anchor="middle">beta = 0.9</text>
<line x1="340.0" y1="230.0" x2="575.0" y2="230.0" class="stroke-muted"/>
<line x1="340.0" y1="40.0" x2="340.0" y2="230.0" class="stroke-muted"/>
<line x1="340.0" y1="230.0" x2="340.0" y2="235.0" class="stroke-muted"/>
<text x="340.0" y="249.0" class="fig-small" text-anchor="middle">0</text>
<line x1="418.3" y1="230.0" x2="418.3" y2="235.0" class="stroke-muted"/>
<text x="418.3" y="249.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="496.7" y1="230.0" x2="496.7" y2="235.0" class="stroke-muted"/>
<text x="496.7" y="249.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="575.0" y1="230.0" x2="575.0" y2="235.0" class="stroke-muted"/>
<text x="575.0" y="249.0" class="fig-small" text-anchor="middle">3000</text>
<line x1="340.0" y1="230.0" x2="575.0" y2="230.0" class="stroke-grid"/>
<line x1="335.0" y1="230.0" x2="340.0" y2="230.0" class="stroke-muted"/>
<text x="331.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="340.0" y1="147.4" x2="575.0" y2="147.4" class="stroke-grid"/>
<line x1="335.0" y1="147.4" x2="340.0" y2="147.4" class="stroke-muted"/>
<text x="331.0" y="151.4" class="fig-small" text-anchor="end">0.0005</text>
<line x1="340.0" y1="64.8" x2="575.0" y2="64.8" class="stroke-grid"/>
<line x1="335.0" y1="64.8" x2="340.0" y2="64.8" class="stroke-muted"/>
<text x="331.0" y="68.8" class="fig-small" text-anchor="end">0.001</text>
<text x="457.5" y="268.0" class="fig-small" text-anchor="middle">steps ago</text>
<polyline points="340.0,64.8 341.6,68.1 343.1,71.3 344.7,74.4 346.3,77.5 347.8,80.5 349.4,83.5 351.0,86.4 352.5,89.2 354.1,92.0 355.7,94.7 357.2,97.4 358.8,100.1 360.4,102.6 361.9,105.1 363.5,107.6 365.1,110.0 366.6,112.4 368.2,114.8 369.8,117.0 371.3,119.3 372.9,121.5 374.5,123.6 376.0,125.7 377.6,127.8 379.2,129.8 380.7,131.8 382.3,133.7 383.9,135.7 385.4,137.5 387.0,139.4 388.6,141.1 390.1,142.9 391.7,144.6 393.3,146.3 394.8,148.0 396.4,149.6 398.0,151.2 399.5,152.8 401.1,154.3 402.7,155.8 404.2,157.3 405.8,158.7 407.4,160.1 408.9,161.5 410.5,162.9 412.1,164.2 413.6,165.5 415.2,166.8 416.8,168.0 418.3,169.3 419.9,170.5 421.5,171.6 423.0,172.8 424.6,173.9 426.2,175.0 427.7,176.1 429.3,177.2 430.9,178.2 432.4,179.3 434.0,180.3 435.6,181.3 437.1,182.2 438.7,183.2 440.3,184.1 441.8,185.0 443.4,185.9 445.0,186.8 446.5,187.6 448.1,188.5 449.7,189.3 451.2,190.1 452.8,190.9 454.4,191.7 455.9,192.4 457.5,193.2 459.1,193.9 460.6,194.6 462.2,195.3 463.8,196.0 465.3,196.7 466.9,197.3 468.5,198.0 470.0,198.6 471.6,199.2 473.2,199.8 474.7,200.4 476.3,201.0 477.9,201.6 479.4,202.2 481.0,202.7 482.6,203.3 484.1,203.8 485.7,204.3 487.3,204.8 488.8,205.3 490.4,205.8 492.0,206.3 493.5,206.8 495.1,207.2 496.7,207.7 498.2,208.1 499.8,208.5 501.4,209.0 502.9,209.4 504.5,209.8 506.1,210.2 507.6,210.6 509.2,211.0 510.8,211.3 512.3,211.7 513.9,212.1 515.5,212.4 517.0,212.8 518.6,213.1 520.2,213.5 521.7,213.8 523.3,214.1 524.9,214.4 526.4,214.7 528.0,215.0 529.6,215.3 531.1,215.6 532.7,215.9 534.3,216.2 535.8,216.5 537.4,216.7 539.0,217.0 540.5,217.2 542.1,217.5 543.7,217.7 545.2,218.0 546.8,218.2 548.4,218.5 549.9,218.7 551.5,218.9 553.1,219.1 554.6,219.3 556.2,219.6 557.8,219.8 559.3,220.0 560.9,220.2 562.5,220.4 564.0,220.6 565.6,220.7 567.2,220.9 568.7,221.1 570.3,221.3 571.9,221.5 573.4,221.6 575.0,221.8" fill="none" class="stroke-b"/>
<line x1="418.3" y1="60.0" x2="418.3" y2="230.0" class="stroke-muted dashed"/>
<text x="424.3" y="70.0" class="fig-small" text-anchor="start">window: 1000</text>
<text x="457.5" y="22.0" class="fig-bold" text-anchor="middle">beta = 0.999</text>
</svg>
Figure 1. How much each past gradient counts in the average, from the newest on the left. With beta 0.9 (left) the weights fall to about a third of the newest weight after 10 steps, so the average remembers roughly the last 10 gradients. With beta 0.999 (right) the same fall takes 1,000 steps. The two vertical scales differ, because the weights in each panel add up to nearly 1.
```

In numpy, with `prev` the old average and `x` the new value:

```python
avg = beta * prev + (1.0 - beta) * x
```

Adam keeps two of these for every weight. The first, $m$, averages the gradient $g$, so $m$ is the typical direction. It uses $\beta_1 = 0.9$, a window of 10 steps. The second, $v$, averages the squared gradient $g^2$ entry by entry, so $\sqrt{v}$ is the typical size, a root-mean-square. It squares the gradient so that a gradient of +3 and one of −3 both count as 9 and cannot cancel, and the square root afterwards returns the answer to the gradient's own units. It uses $\beta_2 = 0.999$, a window of 1,000 steps, because a size needs a longer memory than a direction. Statisticians call these the first and second moments.

## Correct the averages that start at zero

Both averages start at 0, and 0 is a bad guess. Take a gradient of 3, steady. After one step $m = 0.1 \times 3 = 0.3$, a tenth of the truth. After two steps $m = 0.57$, and after three $m = 0.813$. The average has only seen $t$ numbers, and the unseen ones count as zeros. For a constant gradient $g$, the raw average is exactly $(1 - \beta_1^{\,t})\,g$.

The fix is to divide that fraction out. This is **bias correction**, where bias means a systematic error in one direction. Count $t$ from 1, the number of updates so far including this one:

$$
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}
$$

Read it aloud: divide each average by the share of the window that has filled so far. For the gradient 3 at $t = 1$, $\hat{m} = 0.3 / 0.1 = 3$ and $\hat{v} = 0.009 / 0.001 = 9$. Later the divisor drifts toward 1 and the correction fades away. Figure 2 shows both fading.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Raw and corrected moving averages for a constant gradient">
<defs></defs>
<line x1="55.0" y1="230.0" x2="275.0" y2="230.0" class="stroke-muted"/>
<line x1="55.0" y1="40.0" x2="55.0" y2="230.0" class="stroke-muted"/>
<line x1="55.0" y1="230.0" x2="55.0" y2="235.0" class="stroke-muted"/>
<text x="55.0" y="249.0" class="fig-small" text-anchor="middle">0</text>
<line x1="126.0" y1="230.0" x2="126.0" y2="235.0" class="stroke-muted"/>
<text x="126.0" y="249.0" class="fig-small" text-anchor="middle">10</text>
<line x1="196.9" y1="230.0" x2="196.9" y2="235.0" class="stroke-muted"/>
<text x="196.9" y="249.0" class="fig-small" text-anchor="middle">20</text>
<line x1="267.9" y1="230.0" x2="267.9" y2="235.0" class="stroke-muted"/>
<text x="267.9" y="249.0" class="fig-small" text-anchor="middle">30</text>
<line x1="55.0" y1="230.0" x2="275.0" y2="230.0" class="stroke-grid"/>
<line x1="50.0" y1="230.0" x2="55.0" y2="230.0" class="stroke-muted"/>
<text x="46.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="55.0" y1="174.1" x2="275.0" y2="174.1" class="stroke-grid"/>
<line x1="50.0" y1="174.1" x2="55.0" y2="174.1" class="stroke-muted"/>
<text x="46.0" y="178.1" class="fig-small" text-anchor="end">1</text>
<line x1="55.0" y1="118.2" x2="275.0" y2="118.2" class="stroke-grid"/>
<line x1="50.0" y1="118.2" x2="55.0" y2="118.2" class="stroke-muted"/>
<text x="46.0" y="122.2" class="fig-small" text-anchor="end">2</text>
<line x1="55.0" y1="62.4" x2="275.0" y2="62.4" class="stroke-grid"/>
<line x1="50.0" y1="62.4" x2="55.0" y2="62.4" class="stroke-muted"/>
<text x="46.0" y="66.4" class="fig-small" text-anchor="end">3</text>
<text x="165.0" y="268.0" class="fig-small" text-anchor="middle">step t</text>
<line x1="55.0" y1="62.4" x2="275.0" y2="62.4" class="stroke-a"/>
<polyline points="62.1,213.2 69.2,198.1 76.3,184.6 83.4,172.3 90.5,161.3 97.6,151.4 104.7,142.5 111.8,134.5 118.9,127.3 126.0,120.8 133.1,115.0 140.2,109.7 147.3,105.0 154.4,100.7 161.5,96.9 168.5,93.4 175.6,90.3 182.7,87.5 189.8,85.0 196.9,82.7 204.0,80.7 211.1,78.9 218.2,77.2 225.3,75.7 232.4,74.4 239.5,73.2 246.6,72.1 253.7,71.1 260.8,70.2 267.9,69.5" fill="none" class="stroke-b"/>
<circle cx="62.1" cy="213.2" r="3.5" class="fill-b"/>
<circle cx="69.2" cy="198.1" r="3.5" class="fill-b"/>
<circle cx="76.3" cy="184.6" r="3.5" class="fill-b"/>
<text x="275.0" y="54.4" class="fig-small" text-anchor="end">corrected m: always 3</text>
<text x="97.6" y="162.9" class="fig-small" text-anchor="start">raw m: 0.3, 0.57, 0.81, ...</text>
<text x="165.0" y="22.0" class="fig-bold" text-anchor="middle">first average, gradient 3</text>
<line x1="340.0" y1="230.0" x2="575.0" y2="230.0" class="stroke-muted"/>
<line x1="340.0" y1="40.0" x2="340.0" y2="230.0" class="stroke-muted"/>
<line x1="340.0" y1="230.0" x2="340.0" y2="235.0" class="stroke-muted"/>
<text x="340.0" y="249.0" class="fig-small" text-anchor="middle">0</text>
<line x1="418.3" y1="230.0" x2="418.3" y2="235.0" class="stroke-muted"/>
<text x="418.3" y="249.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="496.7" y1="230.0" x2="496.7" y2="235.0" class="stroke-muted"/>
<text x="496.7" y="249.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="575.0" y1="230.0" x2="575.0" y2="235.0" class="stroke-muted"/>
<text x="575.0" y="249.0" class="fig-small" text-anchor="middle">3000</text>
<line x1="340.0" y1="230.0" x2="575.0" y2="230.0" class="stroke-grid"/>
<line x1="335.0" y1="230.0" x2="340.0" y2="230.0" class="stroke-muted"/>
<text x="331.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="340.0" y1="143.6" x2="575.0" y2="143.6" class="stroke-grid"/>
<line x1="335.0" y1="143.6" x2="340.0" y2="143.6" class="stroke-muted"/>
<text x="331.0" y="147.6" class="fig-small" text-anchor="end">0.5</text>
<line x1="340.0" y1="57.3" x2="575.0" y2="57.3" class="stroke-grid"/>
<line x1="335.0" y1="57.3" x2="340.0" y2="57.3" class="stroke-muted"/>
<text x="331.0" y="61.3" class="fig-small" text-anchor="end">1</text>
<text x="457.5" y="268.0" class="fig-small" text-anchor="middle">step t</text>
<polyline points="340.1,224.5 340.2,222.3 340.2,220.5 340.3,219.1 340.4,217.8 340.5,216.6 340.5,215.6 340.6,214.6 340.7,213.6 340.8,212.8 340.9,211.9 340.9,211.1 341.0,210.4 341.1,209.6 341.2,208.9 341.3,208.2 341.3,207.6 341.4,206.9 341.5,206.3 341.6,205.7 341.6,205.1 341.7,204.5 341.8,203.9 341.9,203.4 342.0,202.9 342.0,202.3 342.1,201.8 342.2,201.3 342.3,200.8 342.4,200.3 342.4,199.8 342.5,199.3 342.6,198.9 342.7,198.4 342.7,198.0 342.8,197.5 342.9,197.1 343.0,196.6 343.1,196.2 343.1,195.8 343.2,195.4 343.3,195.0 343.4,194.6 343.4,194.2 343.5,193.8 343.6,193.4 343.7,193.0 343.8,192.6 343.8,192.2 343.9,191.8 344.0,191.5 344.1,191.1 344.2,190.7 344.2,190.4 344.3,190.0 344.4,189.7 344.5,189.3 344.5,189.0 344.6,188.6 344.7,188.3 346.3,182.1 347.8,176.7 349.4,171.9 351.0,167.6 352.5,163.6 354.1,159.9 355.7,156.4 357.2,153.2 358.8,150.2 360.4,147.3 361.9,144.6 363.5,142.0 365.1,139.6 366.6,137.2 368.2,135.0 369.8,132.9 371.3,130.8 372.9,128.8 374.5,126.9 376.0,125.1 377.6,123.3 379.2,121.6 380.7,120.0 382.3,118.4 383.9,116.9 385.4,115.4 387.0,114.0 388.6,112.6 390.1,111.2 391.7,109.9 393.3,108.7 394.8,107.4 396.4,106.2 398.0,105.1 399.5,104.0 401.1,102.9 402.7,101.8 404.2,100.8 405.8,99.8 407.4,98.8 408.9,97.8 410.5,96.9 412.1,96.0 413.6,95.1 415.2,94.3 416.8,93.5 418.3,92.7 419.9,91.9 421.5,91.1 423.0,90.3 424.6,89.6 426.2,88.9 427.7,88.2 429.3,87.5 430.9,86.9 432.4,86.2 434.0,85.6 435.6,85.0 437.1,84.4 438.7,83.8 440.3,83.2 441.8,82.7 443.4,82.1 445.0,81.6 446.5,81.1 448.1,80.6 449.7,80.1 451.2,79.6 452.8,79.1 454.4,78.6 455.9,78.2 457.5,77.7 459.1,77.3 460.6,76.9 462.2,76.5 463.8,76.1 465.3,75.7 466.9,75.3 468.5,74.9 470.0,74.5 471.6,74.2 473.2,73.8 474.7,73.5 476.3,73.1 477.9,72.8 479.4,72.5 481.0,72.2 482.6,71.9 484.1,71.6 485.7,71.3 487.3,71.0 488.8,70.7 490.4,70.4 492.0,70.2 493.5,69.9 495.1,69.6 496.7,69.4 498.2,69.1 499.8,68.9 501.4,68.6 502.9,68.4 504.5,68.2 506.1,68.0 507.6,67.7 509.2,67.5 510.8,67.3 512.3,67.1 513.9,66.9 515.5,66.7 517.0,66.5 518.6,66.3 520.2,66.1 521.7,66.0 523.3,65.8 524.9,65.6 526.4,65.4 528.0,65.3 529.6,65.1 531.1,65.0 532.7,64.8 534.3,64.7 535.8,64.5 537.4,64.4 539.0,64.2 540.5,64.1 542.1,63.9 543.7,63.8 545.2,63.7 546.8,63.5 548.4,63.4 549.9,63.3 551.5,63.2 553.1,63.1 554.6,62.9 556.2,62.8 557.8,62.7 559.3,62.6 560.9,62.5 562.5,62.4 564.0,62.3 565.6,62.2 567.2,62.1 568.7,62.0 570.3,61.9 571.9,61.8 573.4,61.7 575.0,61.6" fill="none" class="stroke-d"/>
<circle cx="340.1" cy="224.5" r="4" class="fill-d"/>
<circle cx="418.3" cy="92.7" r="4" class="fill-d"/>
<text x="350.1" y="218.8" class="fig-small" text-anchor="start">0.03 at t = 1</text>
<text x="422.3" y="112.7" class="fig-small" text-anchor="start">0.80 at t = 1000</text>
<text x="457.5" y="22.0" class="fig-bold" text-anchor="middle">raw sqrt(v) over its true size</text>
</svg>
Figure 2. Left: because the average starts at 0, the raw average of a constant gradient 3 begins at 0.3 and only creeps toward 3 (orange). Dividing by 1 - 0.9^t gives exactly 3 at every step (blue). Right: with beta 0.999 the raw square root of the second average starts at 3 percent of its true size and needs about 1,000 steps to reach 80 percent, so its correction matters for much longer.
```

Why not start $m$ and $v$ at the first gradient? That fixes step 1 only, and it trusts a single batch. With zeros and a correction, $\hat{m}$ is a proper weighted average of the $t$ gradients seen so far, at every step. The second moment needs this most: its raw square root is 3 percent of the truth at step 1 and takes about 1,000 steps to reach 80 percent.

> [!KEY]
> A moving average that starts at zero is too small early on. Dividing by $1 - \beta^t$ restores it, and the correction fades as $t$ grows.

> [!WARNING]
> Start $t$ at 1, not 0. At $t = 0$ the divisor $1 - \beta_1^{\,0}$ is zero, and the first step divides by zero.

In numpy:

```python
m_hat = m / (1.0 - beta1**t)
v_hat = v / (1.0 - beta2**t)
```

## Divide direction by size

Divide the corrected direction by the corrected size. A small $\epsilon$, such as $10^{-8}$, keeps the denominator off zero for a weight whose gradient has always been zero:

$$
w \leftarrow w - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Read it aloud: the learning rate times the average direction measured in typical sizes. In numpy:

```python
new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
```

Work the hot room and the cold room, both at $t = 1$ with $\text{lr} = 0.1$. For the gradient 3, $\hat{m} = 3$ and $\sqrt{\hat{v}} = 3$, so the step is $0.1 \times 3/3 = 0.1$. For the gradient 3000, the corrected averages are 3000 and $\sqrt{9{,}000{,}000} = 3000$, so the step is again 0.1. Plain SGD at the same rate moved 0.3 and 300. At $t = 1$ the ratio $\hat{m}/\sqrt{\hat{v}}$ is exactly $g/|g| = \pm 1$, so Adam's first step is one full learning rate in every coordinate, whatever the gradient says. Remember that, because a later chapter, "Learning-rate schedules: warmup and cosine decay", leans on it. Figure 3 shows the same effect in the valley.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Length of the first step on a flat and a steep coordinate for SGD and for Adam">
<defs></defs>
<line x1="80.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="40.0" x2="80.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-grid"/>
<line x1="75.0" y1="250.0" x2="80.0" y2="250.0" class="stroke-muted"/>
<text x="71.0" y="254.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="186.4" x2="570.0" y2="186.4" class="stroke-grid"/>
<line x1="75.0" y1="186.4" x2="80.0" y2="186.4" class="stroke-muted"/>
<text x="71.0" y="190.4" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="122.7" x2="570.0" y2="122.7" class="stroke-grid"/>
<line x1="75.0" y1="122.7" x2="80.0" y2="122.7" class="stroke-muted"/>
<text x="71.0" y="126.7" class="fig-small" text-anchor="end">1e-1</text>
<line x1="80.0" y1="59.1" x2="570.0" y2="59.1" class="stroke-grid"/>
<line x1="75.0" y1="59.1" x2="80.0" y2="59.1" class="stroke-muted"/>
<text x="71.0" y="63.1" class="fig-small" text-anchor="end">1</text>
<text x="80.0" y="30.0" class="fig-small" text-anchor="start">length of the first step (log scale)</text>
<rect x="111.2" y="187.8" width="60.0" height="62.2" class="fill-b"/>
<text x="141.2" y="181.8" class="fig-small" text-anchor="middle">0.0095</text>
<rect x="233.8" y="60.5" width="60.0" height="189.5" class="fill-b"/>
<text x="263.8" y="54.5" class="fig-small" text-anchor="middle">0.95</text>
<rect x="356.2" y="141.9" width="60.0" height="108.1" class="fill-a"/>
<text x="386.2" y="135.9" class="fig-small" text-anchor="middle">0.05</text>
<rect x="478.8" y="141.9" width="60.0" height="108.1" class="fill-a"/>
<text x="508.8" y="135.9" class="fig-small" text-anchor="middle">0.05</text>
<text x="141.2" y="268.0" class="fig-small" text-anchor="middle">flat axis</text>
<text x="263.8" y="268.0" class="fig-small" text-anchor="middle">steep axis</text>
<text x="386.2" y="268.0" class="fig-small" text-anchor="middle">flat axis</text>
<text x="508.8" y="268.0" class="fig-small" text-anchor="middle">steep axis</text>
<text x="202.5" y="288.0" class="fig-bold" text-anchor="middle">SGD, learning rate 0.0095</text>
<text x="447.5" y="288.0" class="fig-bold" text-anchor="middle">Adam, learning rate 0.05</text>
</svg>
Figure 3. At the point (1, 1) of the valley the gradients are 1 on the flat axis and 100 on the steep axis. SGD's first steps differ by a factor of 100, since a step is learning rate times gradient. Adam divides each average gradient by its own typical size, so both steps are the learning rate, 0.05. Bars start at 1e-3.
```

Skip the correction and the first step is wrong. Raw $m$ is 0.3 and raw $\sqrt{v}$ is 0.095, so the step is $0.1 \times 0.3 / 0.095 = 0.316$, about 3.2 times too long. The two raw averages start at different fractions, 0.1 and 0.001, and the correction repairs both.

Epsilon matters only when a gradient is tiny. A gradient of exactly 0 gives $0/(0 + \epsilon) = 0$, with no division by zero. A gradient of $10^{-12}$ with $\sqrt{\hat v} = 10^{-12}$ gives a step of $10^{-4}$ learning rates, because $\epsilon = 10^{-8}$ dominates the denominator. A large $\epsilon$ makes Adam behave more like SGD with momentum.

One detail about state: store the raw $m$, the raw $v$, and $t$, never the corrected copies. If you store $\hat{m}$, the next step corrects an average that is already corrected.

## Make the valley steeper

The claim to test is that Adam's step does not depend on how steep an axis is. Take the valley from before, with curvatures 1 and 100, and start at $(1, 1)$. Give SGD and momentum a rate set from the steepest curvature $L$, just under their stability limits: $1.95/L$ for SGD and $1.6/L$ for momentum. Give Adam $\text{lr} = 0.05$. Count the steps until the loss falls below $10^{-4}$ and stays there. SGD needs 217, momentum 127, and Adam 123. On this valley momentum and Adam tie.

Make the steep axis 100 times sharper, curvatures 1 and 10,000, and rescale the rates for SGD and momentum. SGD needs 21,837 steps, because its rate must shrink with $L$ and the flat axis crawls. Momentum needs 2,630. Adam keeps the same learning rate and needs 167. Figure 4 shows both valleys.

```figure
<svg viewBox="0 0 600 350" role="img" aria-label="Steps needed to settle below a loss of 1e-4 for SGD, momentum, and Adam on two valleys">
<defs></defs>
<line x1="70.0" y1="270.0" x2="580.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="580.0" y2="270.0" class="stroke-grid"/>
<line x1="65.0" y1="270.0" x2="70.0" y2="270.0" class="stroke-muted"/>
<text x="61.0" y="274.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="210.0" x2="580.0" y2="210.0" class="stroke-grid"/>
<line x1="65.0" y1="210.0" x2="70.0" y2="210.0" class="stroke-muted"/>
<text x="61.0" y="214.0" class="fig-small" text-anchor="end">1e2</text>
<line x1="70.0" y1="150.0" x2="580.0" y2="150.0" class="stroke-grid"/>
<line x1="65.0" y1="150.0" x2="70.0" y2="150.0" class="stroke-muted"/>
<text x="61.0" y="154.0" class="fig-small" text-anchor="end">1e3</text>
<line x1="70.0" y1="90.0" x2="580.0" y2="90.0" class="stroke-grid"/>
<line x1="65.0" y1="90.0" x2="70.0" y2="90.0" class="stroke-muted"/>
<text x="61.0" y="94.0" class="fig-small" text-anchor="end">1e4</text>
<line x1="70.0" y1="30.0" x2="580.0" y2="30.0" class="stroke-grid"/>
<line x1="65.0" y1="30.0" x2="70.0" y2="30.0" class="stroke-muted"/>
<text x="61.0" y="34.0" class="fig-small" text-anchor="end">1e5</text>
<text x="70.0" y="20.0" class="fig-small" text-anchor="start">steps to settle (log scale)</text>
<rect x="111.5" y="189.8" width="52.0" height="80.2" class="fill-b"/>
<text x="137.5" y="183.8" class="fig-small" text-anchor="middle">217</text>
<rect x="171.5" y="203.8" width="52.0" height="66.2" class="fill-d"/>
<text x="197.5" y="197.8" class="fig-small" text-anchor="middle">127</text>
<rect x="231.5" y="204.6" width="52.0" height="65.4" class="fill-a"/>
<text x="257.5" y="198.6" class="fig-small" text-anchor="middle">123</text>
<rect x="366.5" y="69.6" width="52.0" height="200.4" class="fill-b"/>
<text x="392.5" y="63.6" class="fig-small" text-anchor="middle">21,837</text>
<rect x="426.5" y="124.8" width="52.0" height="145.2" class="fill-d"/>
<text x="452.5" y="118.8" class="fig-small" text-anchor="middle">2,630</text>
<rect x="486.5" y="196.6" width="52.0" height="73.4" class="fill-a"/>
<text x="512.5" y="190.6" class="fig-small" text-anchor="middle">167</text>
<text x="197.5" y="290.0" class="fig-bold" text-anchor="middle">curvatures 1 and 100</text>
<text x="452.5" y="290.0" class="fig-bold" text-anchor="middle">curvatures 1 and 10,000</text>
<rect x="110.0" y="325.0" width="12.0" height="12.0" class="fill-b"/>
<text x="128.0" y="336.0" class="fig-small" text-anchor="start">SGD</text>
<rect x="230.0" y="325.0" width="12.0" height="12.0" class="fill-d"/>
<text x="248.0" y="336.0" class="fig-small" text-anchor="start">momentum</text>
<rect x="380.0" y="325.0" width="12.0" height="12.0" class="fill-a"/>
<text x="398.0" y="336.0" class="fig-small" text-anchor="start">Adam</text>
</svg>
Figure 4. Steps until the loss falls below 1e-4 and stays there, from the start (1, 1). SGD and momentum use a rate set from the steep curvature, 1.95 / L and 1.6 / L. Adam keeps lr 0.05 on both valleys. When the steep axis gets 100 times sharper, SGD needs 101 times as many steps and momentum 21 times, but Adam needs only 1.4 times as many.
```

> [!KEY]
> Adam keeps an average gradient and an average squared gradient for every weight, corrects both for starting at zero, and divides one by the other. Each weight then moves about one learning rate per step, whether its gradient is 3 or 3000.

> [!RECAP]
> Hold these five points before you write code:
>
> - `ema_update(prev, x, beta)` is `beta * prev + (1 - beta) * x`, with a window of about `1 / (1 - beta)` steps.
> - `bias_correct(avg, beta, t)` divides by `1 - beta**t`, with `t` counted from 1.
> - Adam's step is `lr * m_hat / (sqrt(v_hat) + eps)`.
> - The state holds the raw `m`, the raw `v`, and `t`. A missing entry means zeros.
> - Each weight moves about `lr` per step, whatever the size of its gradient.

## Build it step by step

The starter gives you `sgd_step`, `momentum_step`, `run_optimiser`, and `steps_to_settle`, which counts the steps until a loss falls below a target and stays there. You write three functions, in the order the checks test them:

1. **`ema_update(prev, x, beta)`.** It takes arrays of one shape and returns `beta * prev + (1 - beta) * x`. The usual mistake is swapping the two weights, so that the new value gets $\beta$.
2. **`bias_correct(avg, beta, t)`.** It returns `avg / (1 - beta**t)`, and raises `ValueError` when `t < 1`. The usual mistake is counting $t$ from 0.
3. **`adam_step(w, grad, state, lr, beta1, beta2, eps)`.** Read `m`, `v`, and `t` from `state`, with zeros and 0 as defaults, and add 1 to `t`. Use `ema_update` for $m$ and for $v$ with `grad**2`. Correct both, and return `(new_w, {"m": m, "v": v, "t": t})`. The usual mistake is storing the corrected averages. Another is collapsing `grad**2` with `np.sum`, so that one $v$ serves every weight.

## Cost and scaling

Adam keeps two extra arrays the size of the weights, $m$ and $v$. For one billion 32-bit weights that is 8 GB, against 4 GB for momentum and none for SGD. The time per step is a few operations per weight, small next to the forward and backward passes.

## Where this breaks

Each of these has a direct remedy:

- A tiny $\hat{v}$ turns a rare gradient into a huge step. Raise $\epsilon$, for instance to $10^{-6}$.
- Adam rescales each axis on its own, so it helps most when the steep and flat directions line up with the weights. A valley tilted across the axes still slows it.
- A large learning rate still makes the loss spike in the first steps, when $\hat{v}$ rests on one or two batches. A later chapter, "Learning-rate schedules: warmup and cosine decay", handles that.

## The exercise

You implement `ema_update`, `bias_correct`, and `adam_step`. The checks verify the average against its closed form, the corrected average, and Adam's first step for gradients 3 and 3000. They check that a zero gradient gives a zero step and that your step matches a reference built from the formulas. Finally they race the three optimisers on the steeper valley.
