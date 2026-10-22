---
title: "Spread, shadows, and the first direction"
slug: 048-spread-shadows-first-direction
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: The best single line through a cloud is the one with the longest shadows, and that is also the line that loses the least.
minutes: 40
gpu: false
objectives:
  - Centre a cloud of points, and say why spread is measured from its middle.
  - Turn a line into one number per point (a score, or shadow), and compute the variance of those scores.
  - Sweep a line through every angle, and find the angle with the largest variance.
  - Split each point's squared length into kept and lost parts, and explain why most kept means least lost.
prereqs:
  - 006-residuals-and-projection
  - 047-gradient-boosting
exercise: 048-spread-shadows-first-direction
---

# Spread, shadows, and the first direction

Every model so far had an answer to learn from. This chapter has none. You hold a table of points and nothing else, and you want to describe it with fewer numbers. Picture thousands of dust grains lying in a thin pancake of flour on a tilted tray. Each grain has three coordinates, but the pancake is almost flat, so two numbers (where on the pancake) say nearly everything. Fewer numbers means smaller files, plots you can look at, less noise, and faster models.

Which two numbers? You could drop one of the three coordinates. If the tray is tilted, though, every coordinate axis cuts across the pancake, and dropping any one of them throws away real information. You want the pancake's own sheet, whatever its tilt.

You build that sheet one line at a time, and this chapter finds the first line. Two goals sound different: keep the points as spread out as possible, or lose as little of them as possible. By the end you see why they pick the same line.

## What you already know

Four ideas carry this chapter:

- A row of a table is a point, and `X.mean(axis=0)` averages down each column.
- The dot product of a point with a unit vector (a vector of length 1) is one number, how far along that vector the point reaches. "Check a fit with right angles" drew it as a shadow.
- Variance is the average of squared gaps from the mean.
- In a right triangle, the two short sides squared add to the long side squared.

## Centre the cloud

The running example has three points with two readings each: $(2, 0)$, $(0, 2)$, and $(4, 4)$. Spread means how far points sit from the middle of the cloud, so the middle has to be our reference. The column means are $(2+0+4)/3 = 2$ and $(0+2+4)/3 = 2$, so the middle is $(2, 2)$. Subtract it from every row and the points become $(0, -2)$, $(-2, 0)$, and $(2, 2)$.

That shift is called **centring**. It moves the middle of the cloud to the origin, so a line through the origin is a line through the middle.

Without it, the measurement goes wrong. Measure along $(1,1)/\sqrt{2}$ from the origin and the raw points give squared readings of 2, 2, and 32. Their sum is 36, and dividing by $n-1 = 2$, as the next section does, gives 18. Most of that number records how far the whole cloud sits from the origin, not how it spreads. After centring, the same measurement gives 6.

$$
X_c = X - \bar{x}
$$

$\bar{x}$ is the row of column means, and it is subtracted from every row of $X$. In numpy:

```python
mean = X.mean(axis=0)
Xc = X - mean
```

## A score is a shadow

Hold a line over the table and shine a light straight down onto it, at right angles. Each point casts a shadow on the line. Its **score** is the signed distance of that shadow from the middle, positive on one side and negative on the other.

Figure 1 shows the picture for two lines through the middle of the example. In words, a score says "how far along this line does the point reach". For a unit vector $u$ pointing along the line, the score of a point is its dot product with $u$.

```figure
<svg viewBox="0 0 600 316" role="img" aria-label="Three points with their shadows on two lines through the mean: line u gets long shadows and short gaps, line w the reverse">
<rect x="10.0" y="12.0" width="270.0" height="270.0" class="stroke-grid" fill="none"/>
<line x1="52.2" y1="282.0" x2="52.2" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="239.8" x2="280.0" y2="239.8" class="stroke-grid" />
<line x1="94.4" y1="282.0" x2="94.4" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="197.6" x2="280.0" y2="197.6" class="stroke-grid" />
<line x1="136.6" y1="282.0" x2="136.6" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="155.4" x2="280.0" y2="155.4" class="stroke-grid" />
<line x1="178.8" y1="282.0" x2="178.8" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="113.3" x2="280.0" y2="113.3" class="stroke-grid" />
<line x1="220.9" y1="282.0" x2="220.9" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="71.1" x2="280.0" y2="71.1" class="stroke-grid" />
<line x1="263.1" y1="282.0" x2="263.1" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="28.9" x2="280.0" y2="28.9" class="stroke-grid" />
<clipPath id="c048-clip0"><rect x="10" y="12" width="270" height="270"/></clipPath>
<g clip-path="url(#c048-clip0)">
<line x1="17.2" y1="274.8" x2="255.9" y2="36.1" class="stroke-b" stroke-width="2.5"/>
<line x1="136.6" y1="239.8" x2="94.4" y2="197.6" class="stroke-muted dashed" />
<line x1="52.2" y1="155.4" x2="94.4" y2="197.6" class="stroke-muted dashed" />
<line x1="220.9" y1="71.1" x2="220.9" y2="71.1" class="stroke-muted dashed" />
</g>
<circle cx="94.4" cy="197.6" r="4.5" class="fill-b"/>
<circle cx="94.4" cy="197.6" r="4.5" class="fill-b"/>
<circle cx="220.9" cy="71.1" r="4.5" class="fill-b"/>
<circle cx="136.6" cy="239.8" r="5.5" class="fill-a"/>
<circle cx="52.2" cy="155.4" r="5.5" class="fill-a"/>
<circle cx="220.9" cy="71.1" r="5.5" class="fill-a"/>
<circle cx="136.6" cy="155.4" r="4" class="fill-ink"/>
<text x="145.0" y="304.0" class="fig-bold" text-anchor="middle">line u: kept 12, lost 4</text>
<text x="128.6" y="257.8" class="fig-small" text-anchor="end">(2, 0)</text>
<text x="48.2" y="145.4" class="fig-small" text-anchor="end">(0, 2)</text>
<text x="214.9" y="59.1" class="fig-small" text-anchor="end">(4, 4)</text>
<text x="144.6" y="177.4" class="fig-small" text-anchor="start">mean (2, 2)</text>
<text x="86.0" y="214.0" class="fig-small" text-anchor="end">score −1.41</text>
<text x="226.9" y="89.1" class="fig-small" text-anchor="start">score 2.83</text>
<rect x="310.0" y="12.0" width="270.0" height="270.0" class="stroke-grid" fill="none"/>
<line x1="352.2" y1="282.0" x2="352.2" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="239.8" x2="580.0" y2="239.8" class="stroke-grid" />
<line x1="394.4" y1="282.0" x2="394.4" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="197.6" x2="580.0" y2="197.6" class="stroke-grid" />
<line x1="436.6" y1="282.0" x2="436.6" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="155.4" x2="580.0" y2="155.4" class="stroke-grid" />
<line x1="478.8" y1="282.0" x2="478.8" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="113.3" x2="580.0" y2="113.3" class="stroke-grid" />
<line x1="520.9" y1="282.0" x2="520.9" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="71.1" x2="580.0" y2="71.1" class="stroke-grid" />
<line x1="563.1" y1="282.0" x2="563.1" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="28.9" x2="580.0" y2="28.9" class="stroke-grid" />
<clipPath id="c048-clip1"><rect x="310" y="12" width="270" height="270"/></clipPath>
<g clip-path="url(#c048-clip1)">
<line x1="317.2" y1="36.1" x2="555.9" y2="274.8" class="stroke-b" stroke-width="2.5"/>
<line x1="436.6" y1="239.8" x2="478.8" y2="197.6" class="stroke-muted dashed" />
<line x1="352.2" y1="155.4" x2="394.4" y2="113.3" class="stroke-muted dashed" />
<line x1="520.9" y1="71.1" x2="436.6" y2="155.4" class="stroke-muted dashed" />
</g>
<circle cx="478.8" cy="197.6" r="4.5" class="fill-b"/>
<circle cx="394.4" cy="113.3" r="4.5" class="fill-b"/>
<circle cx="436.6" cy="155.4" r="4.5" class="fill-b"/>
<circle cx="436.6" cy="239.8" r="5.5" class="fill-a"/>
<circle cx="352.2" cy="155.4" r="5.5" class="fill-a"/>
<circle cx="520.9" cy="71.1" r="5.5" class="fill-a"/>
<circle cx="436.6" cy="155.4" r="4" class="fill-ink"/>
<text x="445.0" y="304.0" class="fig-bold" text-anchor="middle">line w: kept 4, lost 12</text>
<text x="446.6" y="235.8" class="fig-small" text-anchor="start">(2, 0)</text>
<text x="348.2" y="145.4" class="fig-small" text-anchor="end">(0, 2)</text>
<text x="512.9" y="61.1" class="fig-small" text-anchor="end">(4, 4)</text>
<text x="446.6" y="147.4" class="fig-small" text-anchor="start">mean, score 0</text>
<text x="486.8" y="215.6" class="fig-small" text-anchor="start">score 1.41</text>
<text x="386.4" y="105.3" class="fig-small" text-anchor="end">score −1.41</text>
</svg>
Figure 1. The same three points and two lines through the middle (dark dot). The orange dots are the shadows, and the dashed segments are the gaps. On line u the shadows are far apart and the gaps are short (the two lower points share one shadow, and (4, 4) sits on its own shadow). On line w the shadows bunch up and the gaps are long.
```

Take $u = (1, 1)/\sqrt{2}$. The scores of the three centred points are $-2/\sqrt{2} = -1.41$, $-1.41$, and $4/\sqrt{2} = 2.83$. Take $w = (1, -1)/\sqrt{2}$, at right angles to $u$. The scores are $1.41$, $-1.41$, and $0$. Line $u$ spreads the shadows far apart, and line $w$ bunches them together.

The **variance** of the scores puts a number on that. The scores average to zero because the data is centred, so the variance is the sum of squared scores divided by $n-1$. For $u$ the squares are 2, 2, and 8, which sum to 12, and $12/2 = 6$. For $w$ they are 2, 2, and 0, which sum to 4, and $4/2 = 2$. You divide by $n-1$ rather than $n$ because the mean came from the same points, which makes the gaps look a little small. One lone point shows it: its gap from its own mean is always 0, so one point says nothing about spread, and $n-1 = 0$ refuses to pretend otherwise.

$$
\operatorname{var}(u) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i \cdot u)^2
$$

Read it aloud as "take each centred point's score on $u$, square it, add them up, and divide by $n-1$". Both the squaring and the sum run over the scores, so one matrix product does it:

```python
scores = Xc @ u                 # shape (n,), one shadow per point
variance = scores @ scores / (n - 1)
```

## Try every direction

Which line is best? In two dimensions you can try them all. A line at angle $\theta$ from the x axis has unit vector $(\cos\theta, \sin\theta)$. On the example, the centred points $(0,-2)$, $(-2,0)$, and $(2,2)$ have scores $-2\sin\theta$, $-2\cos\theta$, and $2\cos\theta + 2\sin\theta$. Squaring and adding gives $8 + 8\sin\theta\cos\theta = 8 + 4\sin 2\theta$, and dividing by $n-1 = 2$ leaves a variance of $4 + 2\sin 2\theta$. Figure 2 plots it.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Variance of the scores against the angle of the line, peaking at 45 degrees and dipping at 135 degrees">
<line x1="70.0" y1="227.0" x2="550.0" y2="227.0" class="stroke-grid" />
<text x="60.0" y="231.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="181.0" x2="550.0" y2="181.0" class="stroke-grid" />
<text x="60.0" y="185.0" class="fig-small" text-anchor="end">3</text>
<line x1="70.0" y1="135.0" x2="550.0" y2="135.0" class="stroke-grid" />
<text x="60.0" y="139.0" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="89.0" x2="550.0" y2="89.0" class="stroke-grid" />
<text x="60.0" y="93.0" class="fig-small" text-anchor="end">5</text>
<line x1="70.0" y1="43.0" x2="550.0" y2="43.0" class="stroke-grid" />
<text x="60.0" y="47.0" class="fig-small" text-anchor="end">6</text>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted" />
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0°</text>
<line x1="190.0" y1="250.0" x2="190.0" y2="255.0" class="stroke-muted" />
<text x="190.0" y="270.0" class="fig-small" text-anchor="middle">45°</text>
<line x1="310.0" y1="250.0" x2="310.0" y2="255.0" class="stroke-muted" />
<text x="310.0" y="270.0" class="fig-small" text-anchor="middle">90°</text>
<line x1="430.0" y1="250.0" x2="430.0" y2="255.0" class="stroke-muted" />
<text x="430.0" y="270.0" class="fig-small" text-anchor="middle">135°</text>
<line x1="550.0" y1="250.0" x2="550.0" y2="255.0" class="stroke-muted" />
<text x="550.0" y="270.0" class="fig-small" text-anchor="middle">180°</text>
<line x1="70.0" y1="250.0" x2="550.0" y2="250.0" class="stroke-muted" />
<polyline points="70.0,135.0 72.7,131.8 75.3,128.6 78.0,125.4 80.7,122.2 83.3,119.0 86.0,115.9 88.7,112.7 91.3,109.6 94.0,106.6 96.7,103.5 99.3,100.5 102.0,97.6 104.7,94.7 107.3,91.8 110.0,89.0 112.7,86.2 115.3,83.6 118.0,80.9 120.7,78.4 123.3,75.9 126.0,73.4 128.7,71.1 131.3,68.8 134.0,66.6 136.7,64.5 139.3,62.5 142.0,60.6 144.7,58.7 147.3,57.0 150.0,55.3 152.7,53.8 155.3,52.3 158.0,51.0 160.7,49.7 163.3,48.5 166.0,47.5 168.7,46.6 171.3,45.7 174.0,45.0 176.7,44.4 179.3,43.9 182.0,43.5 184.7,43.2 187.3,43.1 190.0,43.0 192.7,43.1 195.3,43.2 198.0,43.5 200.7,43.9 203.3,44.4 206.0,45.0 208.7,45.7 211.3,46.6 214.0,47.5 216.7,48.5 219.3,49.7 222.0,51.0 224.7,52.3 227.3,53.8 230.0,55.3 232.7,57.0 235.3,58.7 238.0,60.6 240.7,62.5 243.3,64.5 246.0,66.6 248.7,68.8 251.3,71.1 254.0,73.4 256.7,75.9 259.3,78.4 262.0,80.9 264.7,83.6 267.3,86.2 270.0,89.0 272.7,91.8 275.3,94.7 278.0,97.6 280.7,100.5 283.3,103.5 286.0,106.6 288.7,109.6 291.3,112.7 294.0,115.9 296.7,119.0 299.3,122.2 302.0,125.4 304.7,128.6 307.3,131.8 310.0,135.0 312.7,138.2 315.3,141.4 318.0,144.6 320.7,147.8 323.3,151.0 326.0,154.1 328.7,157.3 331.3,160.4 334.0,163.4 336.7,166.5 339.3,169.5 342.0,172.4 344.7,175.3 347.3,178.2 350.0,181.0 352.7,183.8 355.3,186.4 358.0,189.1 360.7,191.6 363.3,194.1 366.0,196.6 368.7,198.9 371.3,201.2 374.0,203.4 376.7,205.5 379.3,207.5 382.0,209.4 384.7,211.3 387.3,213.0 390.0,214.7 392.7,216.2 395.3,217.7 398.0,219.0 400.7,220.3 403.3,221.5 406.0,222.5 408.7,223.4 411.3,224.3 414.0,225.0 416.7,225.6 419.3,226.1 422.0,226.5 424.7,226.8 427.3,226.9 430.0,227.0 432.7,226.9 435.3,226.8 438.0,226.5 440.7,226.1 443.3,225.6 446.0,225.0 448.7,224.3 451.3,223.4 454.0,222.5 456.7,221.5 459.3,220.3 462.0,219.0 464.7,217.7 467.3,216.2 470.0,214.7 472.7,213.0 475.3,211.3 478.0,209.4 480.7,207.5 483.3,205.5 486.0,203.4 488.7,201.2 491.3,198.9 494.0,196.6 496.7,194.1 499.3,191.6 502.0,189.1 504.7,186.4 507.3,183.8 510.0,181.0 512.7,178.2 515.3,175.3 518.0,172.4 520.7,169.5 523.3,166.5 526.0,163.4 528.7,160.4 531.3,157.3 534.0,154.1 536.7,151.0 539.3,147.8 542.0,144.6 544.7,141.4 547.3,138.2 550.0,135.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="190.0" cy="43.0" r="6" class="fill-b"/>
<text x="202.0" y="35.0" class="fig-bold" text-anchor="start">best: 45°, variance 6</text>
<circle cx="430.0" cy="227.0" r="6" class="fill-b"/>
<text x="442.0" y="247.0" class="fig-bold" text-anchor="start">worst: 135°, variance 2</text>
<circle cx="70.0" cy="135.0" r="5" class="fill-a"/>
<circle cx="310.0" cy="135.0" r="5" class="fill-a"/>
<text x="80.0" y="159.0" class="fig-small" text-anchor="start">x axis: 4</text>
<text x="300.0" y="159.0" class="fig-small" text-anchor="end">y axis: 4</text>
<text x="310.0" y="285.0" text-anchor="middle">angle of the line, measured from the x axis</text>
<text transform="rotate(-90 14 135)" x="14.0" y="135.0" text-anchor="middle">variance</text>
</svg>
Figure 2. The variance of the scores as the line turns. The peak at 45° is the first principal component, and the dip at 135° sits at a right angle to it. The two coordinate axes (blue dots) both get only 4.
```

The curve rises to 6 at $45^\circ$ and falls to 2 at $135^\circ$, and the best and worst lines sit at right angles. The question from the opening has an answer. Dropping a coordinate means keeping the line at $0^\circ$ or $90^\circ$, where the variance is 4. Both columns of the example have variance 4, and the tilted line at $45^\circ$ gets 6. Because the cloud leans, neither axis lines up with it. The total variance is the same at every angle: $6 + 2 = 4 + 4 = 8$. Turning the line only moves variance between a line and the one at right angles to it.

> [!KEY]
> The **first principal component** is the direction whose scores have the largest variance. On the example it is $(1,1)/\sqrt{2}$, with variance 6.

## Shadows or gaps, the same goal

A second goal sounds different. Instead of asking for long shadows, ask for a line that loses as little of each point as possible. When you replace a point by its shadow, the **gap** is the distance from the point to the line.

Each centred point, its shadow, and its gap form a right triangle, with the point's distance from the middle as the long side. Pythagoras gives

$$
\underbrace{\lVert x_i \rVert^2}_{\text{squared length}} = \underbrace{(x_i \cdot u)^2}_{\text{shadow squared}} + \underbrace{\lVert x_i - (x_i \cdot u)\,u \rVert^2}_{\text{gap squared}} .
$$

Read it aloud as "the squared distance from the middle is the squared shadow plus the squared gap". On the example the squared lengths are 4, 4, and 8, which total 16. Figure 3 splits each one for line $u$: shadows squared are 2, 2, and 8, so 12 are kept, and gaps squared are 2, 2, and 0, so 4 are lost.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Stacked bars: each point's squared length from the mean is its squared shadow plus its squared gap to the line">
<text x="300.0" y="22.0" class="fig-bold" text-anchor="middle">squared length from the mean, split into kept and lost</text>
<text x="138.0" y="72.0" text-anchor="end">(2, 0)</text>
<rect x="150.0" y="50.0" width="76.0" height="34.0" class="fill-b"/>
<rect x="226.0" y="50.0" width="76.0" height="34.0" class="fill-a"/>
<text x="188.0" y="72.0" class="fig-bold fill-paper" text-anchor="middle">2</text>
<text x="264.0" y="72.0" class="fig-bold fill-paper" text-anchor="middle">2</text>
<text x="312.0" y="72.0" class="fig-small" text-anchor="start">= 4</text>
<text x="138.0" y="124.0" text-anchor="end">(0, 2)</text>
<rect x="150.0" y="102.0" width="76.0" height="34.0" class="fill-b"/>
<rect x="226.0" y="102.0" width="76.0" height="34.0" class="fill-a"/>
<text x="188.0" y="124.0" class="fig-bold fill-paper" text-anchor="middle">2</text>
<text x="264.0" y="124.0" class="fig-bold fill-paper" text-anchor="middle">2</text>
<text x="312.0" y="124.0" class="fig-small" text-anchor="start">= 4</text>
<text x="138.0" y="176.0" text-anchor="end">(4, 4)</text>
<rect x="150.0" y="154.0" width="304.0" height="34.0" class="fill-b"/>
<text x="302.0" y="176.0" class="fig-bold fill-paper" text-anchor="middle">8</text>
<text x="464.0" y="176.0" class="fig-small" text-anchor="start">= 8</text>
<line x1="90.0" y1="198.0" x2="560.0" y2="198.0" class="stroke-muted" />
<text x="138.0" y="228.0" class="fig-bold" text-anchor="end">all three</text>
<rect x="150.0" y="206.0" width="228.0" height="34.0" class="fill-b"/>
<rect x="378.0" y="206.0" width="76.0" height="34.0" class="fill-a"/>
<text x="264.0" y="228.0" class="fig-bold fill-paper" text-anchor="middle">12</text>
<text x="416.0" y="228.0" class="fig-bold fill-paper" text-anchor="middle">4</text>
<text x="464.0" y="228.0" class="fig-bold" text-anchor="start">= 16</text>
<rect x="70.0" y="260.0" width="14.0" height="14.0" class="fill-b"/>
<text x="90.0" y="272.0" class="fig-small" text-anchor="start">kept (shadow squared)</text>
<rect x="290.0" y="260.0" width="14.0" height="14.0" class="fill-a"/>
<text x="310.0" y="272.0" class="fig-small" text-anchor="start">lost (gap squared)</text>
</svg>
Figure 3. For line u, each point's squared length (orange plus blue) is fixed. The kept part (orange) totals 12 and the lost part (blue) totals 4. Whatever the line keeps is taken away from what it loses.
```

Here the two goals meet. The squared lengths do not depend on the line, so the total stays 16 whatever you choose. A line that keeps more must lose less, and the line with the longest shadows is the line with the shortest gaps. Line $w$ confirms it: it keeps 4 and loses 12.

You can also rebuild each point from its score alone. The score $z$ puts the point at $z\,u$ from the middle, and adding the middle back gives a point in the original table. The scores $-1.41$, $-1.41$, and $2.83$ rebuild $(1,1)$, $(1,1)$, and $(4,4)$. Their squared distances from the true points are 2, 2, and 0, so the error is the 4 that was lost. In numpy:

```python
Xhat = mean + np.outer(scores, u)   # shape (n, d)
```

The covariance matrix keeps all of this in one small table. It holds each column's variance on the diagonal and how each pair of columns move together off the diagonal. For the example it is $C = X_c^{\top} X_c / (n-1)$, which has rows $(4, 2)$ and $(2, 4)$. The variance along any unit $u$ is then $u^{\top} C u$. For $u = (1,1)/\sqrt{2}$ that is $\tfrac{1}{2}(4 + 2 + 2 + 4) = 6$, the number you found by hand. The next chapter, "The SVD hands you every direction at once", reads the best directions straight from this table.

> [!KEY]
> The pancake needs the line through the cloud that keeps the most. Longest shadows and shortest gaps are one goal, because each point's squared length splits into the two and the total never changes. Centre first, and the best line is the first principal component.

> [!RECAP]
> - Centre with `X - X.mean(axis=0)`, so spread is measured from the middle of the cloud.
> - A score is a shadow: `Xc @ u` for a unit vector `u`.
> - The variance along `u` is `scores @ scores / (n - 1)`, and it depends on the angle.
> - Squared length equals shadow squared plus gap squared, so most kept means least lost.
> - The best line has the largest variance, and the covariance table gives it as $u^{\top} C u$.

## Build it step by step

Build three functions, in this order. A fourth, `sweep_directions`, is written for you in the starter, because it only calls `variance_along` in a loop.

1. **`center(X)`** takes an `(n, d)` array and returns `(Xc, mean)`, shapes `(n, d)` and `(d,)`. Use `X.mean(axis=0)`. The usual mistake is `axis=1`, which averages each row. The shape of the code:

    ```python
    def center(X):
        # Subtract the column mean. Return Xc and that mean.
    ```

2. **`variance_along(Xc, u)`** takes centred data and a unit vector, and returns one float. Take the scores `Xc @ u`, square them, add them up, and divide by `n - 1`. The usual mistake is dividing by `n`. The shape of the code:

    ```python
    def variance_along(Xc, u):
        # Scores, then their sum of squares over n - 1.
    ```

3. **`shadow_split(Xc, u)`** returns `(kept, lost)`, both summed over all points. Compute `kept` from the scores. Compute `lost` from the points themselves, as the squared length of what remains after you subtract each shadow, `Xc - np.outer(scores, u)`. Do not compute it as total minus kept, because the check tests that the three add up. The usual mistake is to forget the sum over points. The shape of the code:

    ```python
    def shadow_split(Xc, u):
        # kept from the scores; lost from the leftover vectors.
    ```

## Where this breaks

- A line describes a flat sheet, so a curved cloud, such as a banana, has no good first direction.
- Variance squares distances, so a few far-off points can drag the line toward themselves.
- Variance depends on units: the same column measured in grams instead of kilograms has a million times the variance. "Compress, and know what you lost" shows how to handle it.

## The exercise

You write `center`, `variance_along`, and `shadow_split`, and the starter gives you `sweep_directions`. The checks confirm the hand numbers: centred means of zero, variance 6 and 2 along $u$ and $w$, and kept 12 and lost 4. They compare `variance_along` with `np.var` on random data, check that the sweep over angles peaks at $45^\circ$ on the example, and confirm that kept plus lost equals the total squared length. The last check sweeps many angles on a random cloud and confirms that the angle with the most kept is the angle with the least lost.
