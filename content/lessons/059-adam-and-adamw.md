---
title: "Adam, and why AdamW decouples the decay"
slug: 059-adam-and-adamw
part: "Part 4 — Neural networks"
summary: Adam divides each weight's average gradient by its own typical size, and AdamW applies weight decay after that division so every weight shrinks equally.
minutes: 55
gpu: false
objectives:
  - Explain why one learning rate suits a weight with gradient 3 and a weight with gradient 3000 badly.
  - Build an exponential moving average, correct it for starting at zero, and take an Adam step.
  - Show that an L2 penalty and decoupled decay are the same for plain SGD, and why they differ inside Adam.
  - Take an AdamW step, and measure how much shrink decay alone causes.
prereqs:
  - 010-momentum
  - 011-ridge-penalty-and-closed-form
  - 057-backprop-mlp-and-gradient-check
  - 058-stochastic-gradients
exercise: 059-adam-and-adamw
---

# Adam, and why AdamW decouples the decay

A network has weights of very different kinds. A bias might see a gradient of 3, while a weight attached to a feature with huge values sees a gradient of 3000. Take a learning rate of 0.1. The first weight moves 0.3, which is sensible. The second moves 300, which throws the model across the map. Lower the rate to suit the second weight and the first moves 0.0003 per step, so after a thousand steps it has barely left home. How can the optimiser measure each weight's own scale and use it?

This chapter builds Adam, which keeps a moving average of each weight's gradient and of its squared gradient, then divides one by the other. That division has a side effect. Put the usual fee on large weights, an L2 penalty, into Adam's gradient and a weight that always saw tiny gradients collapses to zero, while one that saw huge gradients barely moves. The second half shows why, and the one-line change, called AdamW, that fixes it.

## What you already know

Three ideas carry this chapter:

- A step is the learning rate times the gradient, so a steep axis gets long steps and a flat axis gets short ones ("Noisy gradients: stochastic gradient descent and mini-batches").
- Momentum keeps a running sum of past gradients and steps along it ("Momentum: a heavy ball and a look-ahead").
- A ridge penalty charges a fee on the squared size of the weights, and its gradient pulls each weight toward zero ("Ridge: a fee on large weights").

## Give each weight its own ruler

Think of a thermostat in each room instead of one for the whole house. A weight with gradients near 3000 is a hot room, and one with gradients near 3 is a cold one. Measure each gradient in units of its own typical size and every weight speaks the same language: a gradient of 3 in the cold room and 3000 in the hot room both read "about 1". The step is then the learning rate times that unit-free number, so every weight moves about one learning rate per step, whatever its scale.

Why not divide each gradient by its own absolute value? A weight whose gradient flips between +3 and −3 would move a full step each time, back and forth, although it has no direction. A better ruler divides the *average* gradient by the *typical size* of the gradient. A steady slope gives a ratio near ±1 and a long step. Noise that flips sign gives an average near 0 and a short step. So Adam needs two memories per weight: an average of the gradient, and an average of its square.

## Remember with a moving average

An **exponential moving average** keeps a running value and, at each step, blends in the newest number. With decay $\beta$:

$$
\text{avg} \leftarrow \beta\,\text{avg} + (1 - \beta)\,x
$$

Read it aloud: keep a fraction $\beta$ of the old average and take $1 - \beta$ of the new number. Unrolled, the number $k$ steps ago has weight $(1 - \beta)\beta^k$, so the weights fade by the same factor each step. After $1/(1 - \beta)$ steps the weight has fallen to about a third of the newest, so call that the **window**. Figure 1 shows two windows. In numpy, with `prev` the old average:

```python
avg = beta * prev + (1.0 - beta) * x
```

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

Adam keeps two of these for every weight. The first, $m$, averages the gradient $g$, so $m$ is the typical direction. It uses $\beta_1 = 0.9$, a window of 10 steps. The second, $v$, averages the squared gradient $g^2$ entry by entry, so $\sqrt{v}$ is the typical size, a root-mean-square. Squaring stops +3 and −3 from cancelling, and the square root returns the answer to the gradient's own units. It uses $\beta_2 = 0.999$, a window of 1,000 steps, because a size needs a longer memory than a direction.

## Correct the averages that start at zero

Both averages start at 0, and 0 is a bad guess. Take a steady gradient of 3. After one step $m = 0.1 \times 3 = 0.3$, a tenth of the truth. After two steps $m = 0.57$, and after three $m = 0.813$. The average has only seen $t$ numbers, and the unseen ones count as zeros. For a constant gradient $g$, the raw average is exactly $(1 - \beta_1^{\,t})\,g$.

The fix is to divide that fraction out. This is **bias correction**, where bias means a systematic error in one direction. Count $t$ from 1, the number of updates so far including this one:

$$
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}
$$

Read it aloud: divide each average by the share of the window that has filled so far. For the gradient 3 at $t = 1$, $\hat{m} = 0.3 / 0.1 = 3$ and $\hat{v} = 0.009 / 0.001 = 9$. Later the divisor drifts toward 1 and the correction fades away. In numpy:

```python
m_hat = m / (1.0 - beta1**t)
v_hat = v / (1.0 - beta2**t)
```

> [!WARNING]
> Start $t$ at 1, not 0. At $t = 0$ the divisor $1 - \beta_1^{\,0}$ is zero, and the first step divides by zero.

## Divide direction by size

Divide the corrected direction by the corrected size. A small $\epsilon$, such as $10^{-8}$, keeps the denominator off zero for a weight whose gradient has always been zero:

$$
w \leftarrow w - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Read it aloud: the learning rate times the average direction measured in typical sizes. Work the hot room and the cold room, both at $t = 1$ with $\text{lr} = 0.1$. For the gradient 3, $\hat{m} = 3$ and $\sqrt{\hat{v}} = 3$, so the step is $0.1 \times 3/3 = 0.1$. For the gradient 3000, the corrected averages are 3000 and $\sqrt{9{,}000{,}000} = 3000$, so the step is again 0.1. Plain SGD at the same rate moved 0.3 and 300. At $t = 1$ the ratio $\hat{m}/\sqrt{\hat{v}}$ is exactly $g/|g| = \pm 1$, so Adam's first step is one full learning rate in every coordinate. Figure 2 shows the same effect in a long valley, and in numpy the step is:

```python
new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
```

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
Figure 2. At the point (1, 1) of the valley the gradients are 1 on the flat axis and 100 on the steep axis. SGD's first steps differ by a factor of 100, since a step is learning rate times gradient. Adam divides each average gradient by its own typical size, so both steps are the learning rate, 0.05. Bars start at 1e-3.
```

Skip the correction and the first step is about 3.2 times too long. Store the raw $m$, the raw $v$, and $t$, never the corrected copies, or the next step corrects an average that is already corrected.

The payoff shows on a valley with curvatures 1 and 10,000, started at $(1, 1)$. To settle below $10^{-4}$, SGD at its stability limit needs 21,837 steps, momentum 2,630, and Adam at its usual learning rate 167.

> [!KEY]
> Adam keeps an average gradient and an average squared gradient for every weight, corrects both for starting at zero, and divides one by the other. Each weight then moves about one learning rate per step, whether its gradient is 3 or 3000.

## Shrink every weight by the same fraction

Now the penalty. **Weight decay** is a simple rule: at each step, take a small fraction of every weight away. With learning rate $\eta$ and decay rate $\lambda$, each weight loses the fraction $\eta\lambda$ of itself:

$$
w \leftarrow w - \eta\,\lambda\,w
$$

Read it aloud: subtract the learning rate times the decay rate times the weight. Take $w = 1$, $\eta = 0.1$, and $\lambda = 0.01$. The weight loses $0.1 \times 0.01 \times 1 = 0.001$. Every weight loses the same fraction, and that uniformity is the point: decay should not favour any weight.

Where does the rule come from? A penalty $\tfrac{\lambda}{2}\lVert w\rVert^2$ added to the loss has gradient $\lambda w$. Put that into a gradient step and you subtract $\eta\lambda w$. For plain SGD the two views give the same number. Take $w = 1$, $g = 0.5$, $\eta = 0.1$, and $\lambda = 0.01$. With the penalty in the gradient, the gradient is $0.51$ and the weight becomes $1 - 0.051 = 0.949$. With decay applied separately, the plain step gives $0.95$ and the decay removes $0.001$ more, so the weight is $0.949$. Same answer, because SGD does nothing to the gradient except scale it.

## See where Adam breaks the match

Adam does something to the gradient: it divides by $\sqrt{\hat{v}}$. Put the penalty into the gradient and the sum $g + \lambda w$ goes through the averages and then through that division, as the top of Figure 3 shows.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Two update pipelines: where the decay term joins when it is added to the gradient and when it is applied after Adam">
<defs><marker id="c066-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c066-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="300.0" y="20.0" class="fig-bold" text-anchor="middle">Adam with L2 in the gradient</text>
<rect x="2.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="2" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="54.0" y="65.0" class="fig-small" text-anchor="middle">gradient g</text>
<rect x="125.0" y="34.0" width="104.0" height="54.0" class="fill-soft-b"/>
<rect x="125" y="34" width="104" height="54" fill="none" class="stroke-bad"/>
<text x="177.0" y="57.0" class="fig-small" text-anchor="middle">add decay</text>
<text x="177.0" y="73.0" class="fig-small" text-anchor="middle">g + lambda w</text>
<rect x="248.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="248" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="300.0" y="57.0" class="fig-small" text-anchor="middle">averages</text>
<text x="300.0" y="73.0" class="fig-small" text-anchor="middle">m and v</text>
<rect x="371.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="371" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="423.0" y="57.0" class="fig-small" text-anchor="middle">divide</text>
<text x="423.0" y="73.0" class="fig-small" text-anchor="middle">m_hat / sqrt(v_hat)</text>
<rect x="494.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="494" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="546.0" y="65.0" class="fig-small" text-anchor="middle">new weights</text>
<line x1="106.0" y1="61.0" x2="125.0" y2="61.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<line x1="229.0" y1="61.0" x2="248.0" y2="61.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<line x1="352.0" y1="61.0" x2="371.0" y2="61.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<line x1="475.0" y1="61.0" x2="494.0" y2="61.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<text x="300.0" y="110.0" class="fig-small" text-anchor="middle">The decay term goes through the averages and gets divided by sqrt(v_hat)</text>
<text x="300.0" y="156.0" class="fig-bold" text-anchor="middle">AdamW</text>
<rect x="2.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="2" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="54.0" y="201.0" class="fig-small" text-anchor="middle">gradient g</text>
<rect x="125.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="125" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="177.0" y="193.0" class="fig-small" text-anchor="middle">averages</text>
<text x="177.0" y="209.0" class="fig-small" text-anchor="middle">m and v</text>
<rect x="248.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="248" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="300.0" y="193.0" class="fig-small" text-anchor="middle">divide</text>
<text x="300.0" y="209.0" class="fig-small" text-anchor="middle">m_hat / sqrt(v_hat)</text>
<rect x="371.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="371" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="423.0" y="193.0" class="fig-small" text-anchor="middle">subtract</text>
<text x="423.0" y="209.0" class="fig-small" text-anchor="middle">lr x step</text>
<rect x="494.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="494" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="546.0" y="201.0" class="fig-small" text-anchor="middle">new weights</text>
<line x1="106.0" y1="197.0" x2="125.0" y2="197.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<line x1="229.0" y1="197.0" x2="248.0" y2="197.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<line x1="352.0" y1="197.0" x2="371.0" y2="197.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<line x1="475.0" y1="197.0" x2="494.0" y2="197.0" class="stroke-ink" marker-end="url(#c066-arrow-ink)"/>
<rect x="331.0" y="260.0" width="184.0" height="40.0" class="fill-soft-a"/>
<rect x="331" y="260" width="184" height="40" fill="none" class="stroke-a"/>
<text x="423.0" y="284.0" class="fig-small" text-anchor="middle">decay: lr x lambda x w</text>
<line x1="423.0" y1="260.0" x2="423.0" y2="224.0" class="stroke-a" marker-end="url(#c066-arrow-a)"/>
<text x="300.0" y="322.0" class="fig-small" text-anchor="middle">The decay term skips the averages and the division, and joins at the end</text>
</svg>
Figure 3. Top: with L2 in the gradient, the decay term enters at the start, so Adam treats it as part of the gradient and divides it by the weight's typical gradient size. Bottom: AdamW keeps the decay out of the averages and subtracts lr x lambda x w at the end, using the weight from before the step.
```

Consider a quiet weight, with gradient noise of typical size 0.01, and a noisy weight, with typical size 10. Both equal 1. The pull $\lambda w = 0.01$ is as big as the quiet weight's whole gradient, so Adam takes a long step. For the noisy weight it is a thousandth of the gradient size, so Adam sees almost nothing. Adam turns a push into a step of $\eta$ times the push, divided by the weight's typical gradient size. For the noisy weight that is $0.1 \times 0.01 / 10 = 0.0001$, ten times less than the intended 0.001. For the quiet weight, adding the pull makes the typical size about $\sqrt{0.01^2 + 0.01^2} = 0.014$, so the step is about $0.1 \times 0.01 / 0.014 = 0.07$, roughly seventy times more. To measure this, freeze the weights at 1, feed Adam 200 noisy gradients of those two sizes, and compare the next step with decay and without it. Figure 4 shows the result. One number $\lambda$ cannot fix two weights that are off in opposite directions.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Shrink of two weights in one Adam step caused by decay alone, coupled to the gradient versus decoupled">
<defs></defs>
<line x1="80.0" y1="250.0" x2="580.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="40.0" x2="80.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="233.4" x2="580.0" y2="233.4" class="stroke-grid"/>
<line x1="75.0" y1="233.4" x2="80.0" y2="233.4" class="stroke-muted"/>
<text x="71.0" y="237.4" class="fig-small" text-anchor="end">1e-4</text>
<line x1="80.0" y1="178.2" x2="580.0" y2="178.2" class="stroke-grid"/>
<line x1="75.0" y1="178.2" x2="80.0" y2="178.2" class="stroke-muted"/>
<text x="71.0" y="182.2" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="122.9" x2="580.0" y2="122.9" class="stroke-grid"/>
<line x1="75.0" y1="122.9" x2="80.0" y2="122.9" class="stroke-muted"/>
<text x="71.0" y="126.9" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="67.6" x2="580.0" y2="67.6" class="stroke-grid"/>
<line x1="75.0" y1="67.6" x2="80.0" y2="67.6" class="stroke-muted"/>
<text x="71.0" y="71.6" class="fig-small" text-anchor="end">1e-1</text>
<text x="80.0" y="30.0" class="fig-small" text-anchor="start">shrink caused by decay (log scale)</text>
<rect x="94.5" y="75.0" width="96.0" height="175.0" class="fill-b"/>
<text x="142.5" y="69.0" class="fig-small" text-anchor="middle">0.0735</text>
<rect x="219.5" y="233.5" width="96.0" height="16.5" class="fill-b"/>
<text x="267.5" y="227.5" class="fig-small" text-anchor="middle">0.00010</text>
<rect x="344.5" y="178.2" width="96.0" height="71.8" class="fill-a"/>
<text x="392.5" y="172.2" class="fig-small" text-anchor="middle">0.0010</text>
<rect x="469.5" y="178.2" width="96.0" height="71.8" class="fill-a"/>
<text x="517.5" y="172.2" class="fig-small" text-anchor="middle">0.0010</text>
<text x="142.5" y="268.0" class="fig-small" text-anchor="middle">quiet weight</text>
<text x="267.5" y="268.0" class="fig-small" text-anchor="middle">noisy weight</text>
<text x="392.5" y="268.0" class="fig-small" text-anchor="middle">quiet weight</text>
<text x="517.5" y="268.0" class="fig-small" text-anchor="middle">noisy weight</text>
<text x="205.0" y="288.0" class="fig-bold" text-anchor="middle">Adam with L2 in the gradient</text>
<text x="455.0" y="288.0" class="fig-bold" text-anchor="middle">AdamW</text>
<line x1="80.0" y1="178.2" x2="580.0" y2="178.2" class="stroke-muted dashed"/>
<text x="267.5" y="172.2" class="fig-small" text-anchor="middle">intended: 0.001</text>
</svg>
Figure 4. Both weights equal 1, the learning rate is 0.1, and the decay is 0.01, so decay should shrink each weight by 0.001 (dashed line). The quiet weight sees gradients of typical size 0.01 and the noisy one sees size 10. With L2 inside Adam, the quiet weight shrinks by 0.074, about 74 times too much, and the noisy weight by 0.00010, about ten times too little. AdamW shrinks both by exactly 0.001.
```

> [!KEY]
> Adam divides everything in the gradient by the weight's typical gradient size, including the decay pull. So an L2 penalty inside Adam shrinks quiet weights too much and noisy weights too little.

## Apply the decay after Adam

The fix keeps the decay out of the pipeline. Run Adam on the raw gradient, so the averages see only the loss. Then subtract the decay:

$$
w_{\text{new}} = w_{\text{Adam}} - \eta\,\lambda\,w
$$

Read it aloud: take Adam's result and remove the learning rate times the decay rate times the weight you started the step with. This is **AdamW**, where the W stands for weight decay. With $\lambda = 0$ it is Adam. In numpy:

```python
new_w, new_state = adam_step(w, grad, state, lr, beta1, beta2, eps)
new_w = new_w - lr * weight_decay * w
```

Run the frozen test again on AdamW. Both weights shrink by exactly 0.001, whatever their gradient sizes, and the state is the same as Adam's because the averages never see the decay.

> [!KEY]
> AdamW subtracts $\eta\lambda w$ after the Adam step, so every weight shrinks by the same fraction, and the averages see only the loss.

> [!RECAP]
> Hold these points before you write code:
>
> - `ema_update(prev, x, beta)` is `beta * prev + (1 - beta) * x`, and `bias_correct(avg, beta, t)` divides by `1 - beta**t`, with `t` counted from 1.
> - Adam's step is `lr * m_hat / (sqrt(v_hat) + eps)`, and the state holds the raw `m`, the raw `v`, and `t`.
> - In plain SGD an L2 penalty in the gradient and decay after the step are the same.
> - In Adam, `grad + weight_decay * w` goes through the division, so quiet weights shrink too much and noisy weights too little.
> - AdamW calls Adam on the raw gradient, then subtracts `lr * weight_decay * w` using the incoming `w`.

## Build it step by step

The starter gives you `sgd_step`, `momentum_step`, `run_optimiser`, and `steps_to_settle`. You write six functions, in the order the checks test them:

1. **`ema_update(prev, x, beta)`** returns `beta * prev + (1 - beta) * x`. The usual mistake is swapping the two weights, so the new value gets $\beta$.
2. **`bias_correct(avg, beta, t)`** returns `avg / (1 - beta**t)`, and raises `ValueError` when `t < 1`. The usual mistake is counting $t$ from 0.
3. **`adam_step(w, grad, state, lr, beta1, beta2, eps)`** reads `m`, `v`, and `t` from `state`, with zeros and 0 as defaults, and adds 1 to `t`. Use `ema_update` for $m$ and for $v$ with `grad**2`, correct both, and return `(new_w, {"m": m, "v": v, "t": t})`. The usual mistakes are storing the corrected averages and collapsing `grad**2` with `np.sum`, so one $v$ serves every weight.
4. **`adam_l2_step(..., weight_decay)`** is the coupled version, kept for comparison: call `adam_step` on `grad + weight_decay * w`. The usual mistake is applying the decay after the step.
5. **`adamw_step(..., weight_decay)`** calls `adam_step` on the raw `grad`, then subtracts `lr * weight_decay * w` with the incoming `w`. The usual mistake is using the new weight in the decay term.
6. **`decay_amount(step_fn, w, grads, lr, weight_decay)`** keeps `w` frozen and runs `step_fn(w, g, state, lr=lr, weight_decay=d)` over every row of `grads`, threading the state, once with `d = 0` and once with `d = weight_decay`. It returns the last weights of the first run minus those of the second. The usual mistake is updating `w` between calls, which mixes the decay into later gradients.

## Where this breaks

Each of these has a direct remedy:

- A tiny $\hat{v}$ turns a rare gradient into a huge step. Raise $\epsilon$, for instance to $10^{-6}$.
- A large learning rate still makes the loss spike in the first steps, when $\hat{v}$ rests on one or two batches. A later chapter handles that with warmup.
- A decay rate that is too large pulls the weights to zero faster than the loss can push back, so lower $\lambda$. Common practice leaves biases and normalisation scales out of the decay.
- Some libraries name their coupled penalty `weight_decay` on a plain Adam. It is the L2 version from this chapter, so read the documentation before trusting the name.

## The exercise

You implement `ema_update`, `bias_correct`, `adam_step`, `adam_l2_step`, `adamw_step`, and `decay_amount`. The checks verify the average and its correction, Adam's first step for gradients 3 and 3000, and a race against SGD and momentum. They check that the L2 version is Adam on the penalised gradient and that AdamW differs from Adam by exactly $\eta\lambda w$. Finally they measure the shrink from decay alone: equal for both weights under AdamW and very uneven under L2.
