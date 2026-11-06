---
title: "Principal directions: where the data spreads"
slug: 044-principal-directions
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: The best line through a cloud has the longest shadows and loses the least, and the SVD of the centred data lists every such direction in order.
minutes: 50
gpu: false
objectives:
  - Centre a cloud, turn a line into one score per point (a shadow), and compute the variance of those scores.
  - Explain why the line with the largest score variance is also the line that loses the least.
  - Read components, scores, and explained variances from the SVD of the centred data.
  - Show that the covariance table has the same directions, and compute them with an eigendecomposition.
  - Pick the number of directions to keep from the explained variance ratios.
prereqs:
  - 006-residuals-and-projection
  - 043-gradient-boosting
  - 007-stable-least-squares
exercise: 044-principal-directions
---

# Principal directions: where the data spreads

Every model so far had an answer to learn from. This chapter has none. You hold a table of points and nothing else, and you want to describe it with fewer numbers. Picture thousands of dust grains lying in a thin pancake of flour on a tilted tray. Each grain has three coordinates, but the pancake is almost flat, so two numbers (where on the pancake) say nearly everything. Fewer numbers means smaller files, plots you can look at, less noise, and faster models.

Which two numbers? Dropping a coordinate fails: on a tilted tray every axis cuts across the pancake, so dropping one throws away real information. You want the pancake's own sheet.

You build that sheet one line at a time. First you find the best single line, and see that "keep the points spread out" and "lose as little of them as possible" pick the same line. Then you see that one call to the SVD hands you every direction in order, with no search.

## What you already know

Four ideas carry this chapter:

- A row of a table is a point, and `X.mean(axis=0)` averages down each column.
- The dot product of a point with a unit vector (a vector of length 1) is how far along that vector the point reaches. "Check a fit with right angles" drew it as a shadow.
- Variance is the average of squared gaps from the mean, and in a right triangle the two short sides squared add to the long side squared.
- "Lost digits, and how QR and the SVD keep them" split a matrix as $X = U S V^{\top}$, with orthonormal columns in $U$ and $V$ and singular values $s_1 \ge s_2 \ge \cdots \ge 0$ on the diagonal of $S$.

## Centre the cloud

The running example has three points with two readings each: $(2, 0)$, $(0, 2)$, and $(4, 4)$. Spread means how far points sit from the middle of the cloud, so the middle is our reference. The column means are $(2+0+4)/3 = 2$ and $(0+2+4)/3 = 2$, so the middle is $(2, 2)$. Subtract it from every row and the points become $(0, -2)$, $(-2, 0)$, and $(2, 2)$. That shift is called **centring**. It moves the middle of the cloud to the origin, so a line through the origin is a line through the middle.

Without it, the measurement goes wrong. Along $(1,1)/\sqrt{2}$ the raw points give squared readings of 2, 2, and 32, summing to 36, and most of that records how far the cloud sits from the origin. After centring, the sum is 12.

```python
mean = X.mean(axis=0)
Xc = X - mean
```

## A score is a shadow

Shine a light at right angles onto a line through the middle, and each point casts a shadow on it. Its **score** is the signed distance of that shadow from the middle, positive on one side and negative on the other. For a unit vector $u$ pointing along the line, the score of a point is its dot product with $u$. Figure 1 shows two lines through the middle of the example.

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

The **variance** of the scores puts a number on that. The scores average to zero because the data is centred, so the variance is the sum of squared scores divided by $n-1$. For $u$ the squares are 2, 2, and 8, which sum to 12, and $12/2 = 6$. For $w$ they are 2, 2, and 0, so the variance is $4/2 = 2$. You divide by $n-1$ rather than $n$ because the mean came from the same points, which makes the gaps look a little small: one lone point has a gap of 0 from its own mean, and $n-1 = 0$ refuses to pretend that means no spread.

$$
\operatorname{var}(u) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i \cdot u)^2
$$

Read it aloud as "square each centred point's score on $u$, add them up, and divide by $n-1$":

```python
scores = Xc @ u                 # shape (n,), one shadow per point
variance = scores @ scores / (n - 1)
```

## Try every direction

A line at angle $\theta$ has unit vector $(\cos\theta, \sin\theta)$, so the centred points have scores $-2\sin\theta$, $-2\cos\theta$, and $2\cos\theta + 2\sin\theta$. Squaring, adding, and dividing by $n-1 = 2$ leaves a variance of $4 + 2\sin 2\theta$. Figure 2 plots it.

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

The curve rises to 6 at $45^\circ$ and falls to 2 at $135^\circ$, and the best and worst lines sit at right angles. Dropping a coordinate keeps the line at $0^\circ$ or $90^\circ$, where the variance is only 4, because the cloud leans. The total is the same at every angle, $6 + 2 = 4 + 4 = 8$: turning the line only moves variance between a line and the one at right angles to it.

> [!KEY]
> The **first principal component** is the direction whose scores have the largest variance. On the example it is $(1,1)/\sqrt{2}$, with variance 6.

## Shadows or gaps, the same goal

A second goal sounds different: lose as little of each point as possible. When you replace a point by its shadow, the **gap** is the distance from the point to the line. Each centred point, its shadow, and its gap form a right triangle, with the point's distance from the middle as the long side. Pythagoras gives

$$
\underbrace{\lVert x_i \rVert^2}_{\text{squared length}} = \underbrace{(x_i \cdot u)^2}_{\text{shadow squared}} + \underbrace{\lVert x_i - (x_i \cdot u)\,u \rVert^2}_{\text{gap squared}} .
$$

Read it aloud as "the squared distance from the middle is the squared shadow plus the squared gap". On the example the squared lengths are 4, 4, and 8, which total 16. For line $u$ the squared shadows are 2, 2, and 8, so 12 are kept, and the squared gaps are 2, 2, and 0, so 4 are lost. The squared lengths do not depend on the line, so the total stays 16 whatever you choose. A line that keeps more must lose less, so the line with the longest shadows has the shortest gaps. Line $w$ confirms it: it keeps 4 and loses 12.

## Why searching stops working

Suppose each row has 100 readings. A direction in 100 dimensions needs 99 angles, so a grid with 180 steps per angle has about $10^{223}$ points. You do not have to search.

Multiply the centred data matrix $X_c$ by a unit direction $u$ and you get the scores. Their sum of squares is $(n-1)$ times the variance along $u$, so "how far does $X_c$ stretch $u$?" and "how spread out are the shadows on $u$?" are the same question. Feed every unit direction through $X_c$ and the outputs trace an ellipse, which Figure 3 draws, and the long axis is the direction of greatest spread. The SVD finds the axes. The stretchiest direction is $v_1$, with stretch $s_1$. Among directions at right angles to $v_1$, the stretchiest is $v_2$, with stretch $s_2$, and so on down the list.

```figure
<svg viewBox="0 0 600 296" role="img" aria-label="A unit circle of input directions on the left, and on the right the ellipse that the data matrix turns it into, with axis lengths 3.46 and 2.00">
<defs><marker id="c048-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c048-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<circle cx="120.0" cy="150.0" r="85" class="stroke-muted" fill="none"/>
<line x1="0.0" y1="150.0" x2="240.0" y2="150.0" class="stroke-grid" />
<line x1="120.0" y1="30.0" x2="120.0" y2="270.0" class="stroke-grid" />
<line x1="120.0" y1="150.0" x2="180.1" y2="89.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c048-arrow)"/>
<text x="188.1" y="83.9" class="fig-bold" text-anchor="start">v1</text>
<line x1="120.0" y1="150.0" x2="59.9" y2="89.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c048-arrow)"/>
<text x="51.9" y="83.9" class="fig-bold" text-anchor="end">v2</text>
<line x1="120.0" y1="150.0" x2="155.9" y2="73.0" class="stroke-a" stroke-width="2" marker-end="url(#c048-arrow-a)"/>
<circle cx="155.9" cy="73.0" r="4.5" class="fill-a"/>
<text x="166.0" y="60.0" class="fig-small" text-anchor="start">any unit u</text>
<text x="120.0" y="282.0" class="fig-bold" text-anchor="middle">inputs: every unit direction</text>
<polyline points="568.6,150.0 568.4,145.8 567.8,141.6 566.9,137.5 565.5,133.4 563.8,129.3 561.8,125.3 559.4,121.3 556.6,117.5 553.5,113.7 550.0,110.0 546.2,106.4 542.1,103.0 537.7,99.7 533.0,96.5 528.0,93.4 522.7,90.5 517.2,87.8 511.4,85.3 505.5,82.9 499.3,80.7 492.9,78.7 486.4,76.9 479.7,75.3 472.8,73.9 465.9,72.7 458.8,71.7 451.7,71.0 444.5,70.4 437.3,70.1 430.0,70.0 422.7,70.1 415.5,70.4 408.3,71.0 401.2,71.7 394.1,72.7 387.2,73.9 380.3,75.3 373.6,76.9 367.1,78.7 360.7,80.7 354.5,82.9 348.6,85.3 342.8,87.8 337.3,90.5 332.0,93.4 327.0,96.5 322.3,99.7 317.9,103.0 313.8,106.4 310.0,110.0 306.5,113.7 303.4,117.5 300.6,121.3 298.2,125.3 296.2,129.3 294.5,133.4 293.1,137.5 292.2,141.6 291.6,145.8 291.4,150.0 291.6,154.2 292.2,158.4 293.1,162.5 294.5,166.6 296.2,170.7 298.2,174.7 300.6,178.7 303.4,182.5 306.5,186.3 310.0,190.0 313.8,193.6 317.9,197.0 322.3,200.3 327.0,203.5 332.0,206.6 337.3,209.5 342.8,212.2 348.6,214.7 354.5,217.1 360.7,219.3 367.1,221.3 373.6,223.1 380.3,224.7 387.2,226.1 394.1,227.3 401.2,228.3 408.3,229.0 415.5,229.6 422.7,229.9 430.0,230.0 437.3,229.9 444.5,229.6 451.7,229.0 458.8,228.3 465.9,227.3 472.8,226.1 479.7,224.7 486.4,223.1 492.9,221.3 499.3,219.3 505.5,217.1 511.4,214.7 517.2,212.2 522.7,209.5 528.0,206.6 533.0,203.5 537.7,200.3 542.1,197.0 546.2,193.6 550.0,190.0 553.5,186.3 556.6,182.5 559.4,178.7 561.8,174.7 563.8,170.7 565.5,166.6 566.9,162.5 567.8,158.4 568.4,154.2 568.6,150.0" fill="none" class="stroke-muted"/>
<line x1="270.0" y1="150.0" x2="590.0" y2="150.0" class="stroke-grid" />
<line x1="430.0" y1="30.0" x2="430.0" y2="270.0" class="stroke-grid" />
<line x1="430.0" y1="150.0" x2="567.6" y2="150.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c048-arrow)"/>
<line x1="430.0" y1="150.0" x2="430.0" y2="71.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c048-arrow)"/>
<text x="562.6" y="170.0" class="fig-bold" text-anchor="end">s1 = 3.46</text>
<text x="440.0" y="58.0" class="fig-bold" text-anchor="start">s2 = 2.00</text>
<line x1="430.0" y1="150.0" x2="560.2" y2="122.6" class="stroke-a" stroke-width="2"/>
<circle cx="560.2" cy="122.6" r="4.5" class="fill-a"/>
<text x="584.0" y="102.0" class="fig-small" text-anchor="middle">Xc u</text>
<text x="430.0" y="282.0" class="fig-bold" text-anchor="middle">outputs: the scores Xc u</text>
<text x="248.0" y="140.0" class="fig-bold" text-anchor="middle">Xc</text>
<line x1="222.0" y1="150.0" x2="274.0" y2="150.0" class="stroke-muted" marker-end="url(#c048-arrow-a)"/>
</svg>
Figure 3. Left: every unit direction u, with the two special directions v1 and v2 at right angles. Right: the data matrix turns that circle into an ellipse, whose long and short axes have lengths s1 and s2. The directions v1 and v2 are exactly the ones that become the axes, so the ellipse is longest where the spread is largest.
```

On the example the stretches are $s_1 = \sqrt{12} = 3.46$ and $s_2 = 2$. Squared and divided by $n-1 = 2$ they give 6 and 2, the variances you found by sweeping.

$$
X_c = U S V^{\top}, \qquad \lambda_i = \frac{s_i^2}{n-1}
$$

Read it aloud as "the data splits into a rotation, a stretch, and a rotation, and direction $i$ holds its stretch squared over $n-1$". The directions are the columns of $V$, so numpy hands them back as the rows of `Vt`:

```python
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
components = Vt[:k].T                   # shape (d, k)
scores = Xc @ components                # shape (n, k)
explained_variance = s[:k] ** 2 / (n - 1)
```

On the example, numpy returns the first row of `Vt` as $(-0.71, -0.71)$, not $(0.71, 0.71)$. Both describe the same line, and the sign carries no meaning, so compare directions with `abs(u @ v)` equal to 1.

> [!WARNING]
> `np.linalg.svd` returns $V^{\top}$, not $V$. The directions are the rows of `Vt`, so the top $k$ are `Vt[:k]`. The columns of `Vt` mean nothing.

## The covariance table finds the same directions

The SVD is not the only route. The covariance table $C = X_c^{\top} X_c/(n-1)$ holds each column's variance on the diagonal and how each pair of columns move together off it. For the example its rows are $(4, 2)$ and $(2, 4)$, and the variance along any unit $u$ is $u^{\top} C u$. For $u = (1,1)/\sqrt{2}$ that is $\tfrac{1}{2}(4 + 2 + 2 + 4) = 6$, the number you found by hand.

Substitute $X_c = U S V^{\top}$ into $X_c^{\top} X_c$. The middle $U^{\top} U$ is the identity, because the columns of $U$ are orthonormal, and what remains is

$$
X_c^{\top} X_c = V S^2 V^{\top} .
$$

An **eigenvector** of a matrix is a direction it only stretches, never turns, and the stretch factor is the **eigenvalue**. So the columns of $V$ are the eigenvectors of $X_c^{\top} X_c$, with eigenvalues $s_i^2$, and of $C$, with eigenvalues $s_i^2/(n-1)$.

Check it on the example. $Cv = \lambda v$ has a nonzero solution only when $C - \lambda I$ has determinant zero: $(4 - \lambda)^2 - 4 = 0$, so $\lambda = 6$ or $2$, the two variances again. For $\lambda = 6$ the rows of $C - 6I$ say $v_1 = v_2$, and the unit vector is $(1,1)/\sqrt{2}$, the peak of the sweep. That is no accident: write any unit direction as $u = a\,v_1 + b\,v_2$ with $a^2 + b^2 = 1$, and its variance is $6a^2 + 2b^2$, a weighted average of 6 and 2, largest when all the weight sits on 6.

Why call the SVD at all, when `np.linalg.eigh` handles $C$? Forming $X_c^{\top} X_c$ squares every singular value. Stretches of 1 and $10^{-9}$ become 1 and $10^{-18}$, and double precision keeps only about 16 digits, so the small one drowns. The SVD works on $X_c$ directly. Also, `eigh` returns eigenvalues in ascending order, so reverse them:

```python
C = Xc.T @ Xc / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(C)    # smallest first
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
```

## How many directions to keep

Each direction holds a share of the total variance, and the shares add to 1:

$$
r_i = \frac{s_i^2}{\sum_j s_j^2} .
$$

The $n-1$ cancels, so the example has $r = (12/16,\ 4/16) = (0.75,\ 0.25)$. There are $\min(n, d)$ shares, and when $n \le d$ the last is zero, because $n$ centred points span at most $n-1$ directions.

```python
ratios = s ** 2 / np.sum(s ** 2)
```

Figure 4 shows a **scree plot** for six-dimensional data built from two strong directions plus a little noise. Two bars tower and four are tiny. The cumulative shares are 78% after one and 99% after two. A common habit is to keep the smallest $k$ whose cumulative share reaches 90%, as `int(np.searchsorted(np.cumsum(ratios), 0.90) + 1)`. The 90% is a habit, not a theorem: use a higher target when you need to reconstruct, and look for the elbow when the share creeps up slowly.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Bars of the variance of each of six components, two tall and four tiny, with a cumulative line crossing 90 percent at the second">
<line x1="80.0" y1="230.0" x2="520.0" y2="230.0" class="stroke-grid" />
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="190.0" x2="520.0" y2="190.0" class="stroke-grid" />
<text x="72.0" y="194.0" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="150.0" x2="520.0" y2="150.0" class="stroke-grid" />
<text x="72.0" y="154.0" class="fig-small" text-anchor="end">4</text>
<line x1="80.0" y1="110.0" x2="520.0" y2="110.0" class="stroke-grid" />
<text x="72.0" y="114.0" class="fig-small" text-anchor="end">6</text>
<line x1="80.0" y1="70.0" x2="520.0" y2="70.0" class="stroke-grid" />
<text x="72.0" y="74.0" class="fig-small" text-anchor="end">8</text>
<line x1="80.0" y1="30.0" x2="520.0" y2="30.0" class="stroke-grid" />
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">10</text>
<text x="528.0" y="234.0" class="fig-small" text-anchor="start">0%</text>
<text x="528.0" y="134.0" class="fig-small" text-anchor="start">50%</text>
<text x="528.0" y="34.0" class="fig-small" text-anchor="start">100%</text>
<rect x="94.7" y="50.9" width="44.0" height="179.1" class="fill-a"/>
<text x="116.7" y="64.0" class="fig-small fill-paper" text-anchor="middle">8.96</text>
<text x="116.7" y="248.0" class="fig-small" text-anchor="middle">1</text>
<rect x="168.0" y="183.3" width="44.0" height="46.7" class="fill-a"/>
<text x="190.0" y="177.3" class="fig-small" text-anchor="middle">2.34</text>
<text x="190.0" y="248.0" class="fig-small" text-anchor="middle">2</text>
<rect x="241.3" y="229.1" width="44.0" height="0.9" class="fill-a"/>
<text x="263.3" y="223.1" class="fig-small" text-anchor="middle">0.04</text>
<text x="263.3" y="248.0" class="fig-small" text-anchor="middle">3</text>
<rect x="314.7" y="229.2" width="44.0" height="0.8" class="fill-a"/>
<text x="336.7" y="223.2" class="fig-small" text-anchor="middle">0.04</text>
<text x="336.7" y="248.0" class="fig-small" text-anchor="middle">4</text>
<rect x="388.0" y="229.2" width="44.0" height="0.8" class="fill-a"/>
<text x="410.0" y="223.2" class="fig-small" text-anchor="middle">0.04</text>
<text x="410.0" y="248.0" class="fig-small" text-anchor="middle">5</text>
<rect x="461.3" y="229.4" width="44.0" height="0.6" class="fill-a"/>
<text x="483.3" y="223.4" class="fig-small" text-anchor="middle">0.03</text>
<text x="483.3" y="248.0" class="fig-small" text-anchor="middle">6</text>
<polyline points="116.7,73.6 190.0,32.8 263.3,32.0 336.7,31.3 410.0,30.6 483.3,30.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="116.7" cy="73.6" r="4.5" class="fill-b"/>
<circle cx="190.0" cy="32.8" r="4.5" class="fill-b"/>
<circle cx="263.3" cy="32.0" r="4.5" class="fill-b"/>
<circle cx="336.7" cy="31.3" r="4.5" class="fill-b"/>
<circle cx="410.0" cy="30.6" r="4.5" class="fill-b"/>
<circle cx="483.3" cy="30.0" r="4.5" class="fill-b"/>
<line x1="80.0" y1="50.0" x2="520.0" y2="50.0" class="stroke-b dashed" />
<text x="516.0" y="68.0" class="fig-small" text-anchor="end">90% of the variance</text>
<text x="126.7" y="91.6" class="fig-bold" text-anchor="start">78%</text>
<text x="200.0" y="22.8" class="fig-bold" text-anchor="start">99%</text>
<text x="300.0" y="272.0" text-anchor="middle">component number</text>
<text transform="rotate(-90 14 130)" x="14.0" y="130.0" text-anchor="middle">variance of each component</text>
</svg>
Figure 4. The variance of each of six components on generated data, with a cumulative line. Two bars tower over four small ones, and the line passes 90% at the second bar, so keep two.
```

> [!KEY]
> You never search for the best direction. The SVD of the centred data lists all of them, in order, and $s_i^2/(n-1)$ is the variance each one holds. Longest shadows and shortest gaps are one goal, because each point's squared length splits into the two and the total never changes, so these are also the directions that lose the least.

> [!RECAP]
> - Centre first. A score is a shadow, `Xc @ u`, and its variance is `scores @ scores / (n - 1)`.
> - Squared length equals shadow squared plus gap squared, so most kept means least lost.
> - Components are `Vt[:k].T`, and variances are `s[:k] ** 2 / (n - 1)`. The covariance table has the same directions.
> - Shares are $s_i^2/\sum_j s_j^2$, and you keep enough directions to reach a target such as 90%.

## Build it step by step

`sweep_directions` is written for you, because it only calls `variance_along` in a loop. You write six functions, in this order.

1. **`center(X)`** takes an `(n, d)` array and returns `(Xc, mean)`, shapes `(n, d)` and `(d,)`. The usual mistake is `axis=1`, which averages each row.
2. **`variance_along(Xc, u)`** takes centred data and a unit vector, and returns one float: the scores `Xc @ u`, squared, summed, and divided by `n - 1`. The usual mistake is dividing by `n`.
3. **`shadow_split(Xc, u)`** returns `(kept, lost)`, both summed over all points. Compute `kept` from the scores and `lost` from the leftover vectors `Xc - np.outer(scores, u)`, not as total minus kept. The usual mistake is to forget the sum over points.
4. **`pca_svd(X, k)`** returns `components` `(d, k)`, `scores` `(n, k)`, and `explained_variance` `(k,)`. Centre, call `np.linalg.svd(Xc, full_matrices=False)`. The usual mistake is `Vt[:, :k]`, which slices columns of $V^{\top}$.
5. **`covariance_eigh(X, k)`** returns the top-$k$ eigenvalues `(k,)` and eigenvectors `(d, k)` of the sample covariance, largest first. Use `np.linalg.eigh` and flip the order. The usual mistake is to leave them ascending.
6. **`explained_variance_ratio(X)`** returns all `min(n, d)` shares, largest first, from `np.linalg.svd(Xc, compute_uv=False)`. The usual mistake is to keep only the top $k$, so the shares no longer add to 1.

## Where this breaks

- A line describes a flat sheet, so a curved cloud, such as a banana, has no good first direction.
- Variance squares distances, so far-off points drag the line toward them, and it depends on units: grams have a million times the variance of kilograms. "Compress, and know what you lost" shows how to handle it.
- The 90% rule says nothing about whether the dropped directions matter to your task. They might hold the signal.

## The exercise

You write `center`, `variance_along`, `shadow_split`, `pca_svd`, `covariance_eigh`, and `explained_variance_ratio`. The checks confirm the hand numbers (variance 6 and 2, kept 12 and lost 4, shares 0.75 and 0.25), that kept plus lost equals the total squared length, that the components are orthonormal, and that no random direction beats the first one. They also confirm that the eigenvectors satisfy $Cv = \lambda v$ and match the SVD up to sign.
