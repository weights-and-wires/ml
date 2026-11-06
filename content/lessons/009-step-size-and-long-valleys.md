---
title: "How big a step? Step size and long valleys"
slug: 009-step-size-and-long-valleys
part: "Part 1 — Foundations"
summary: Each step multiplies your distance to the bottom by a factor per axis, so the sharpest axis caps the step and the ratio L/mu sets the speed.
minutes: 55
gpu: false
objectives:
  - Explain why a step multiplies the distance to the bottom by 1 minus the step size times the curvature on each axis, and find the longest safe step, 2/L.
  - Compute the condition number L/mu and the shrink factor 1 - 1/kappa at step 1/L.
  - Count the steps to cut the miss by a factor, and see that the count grows in proportion to kappa.
  - Compute the best step along a downhill line, and connect a stretched bowl to unscaled features.
prereqs:
  - 008-walking-downhill
exercise: 009-step-size-and-long-valleys
---

# How big a step? Step size and long valleys

You walk the bowl from the last chapter, and each step costs you the same effort whether it is short or long. So why not take a huge step and get there faster? You can try it: replace the step 0.1 with 0.3 and walk again.

The loss does not fall. It goes from 0 to 14.7 to 74.6 to 314.6. Each step lands you higher on the opposite wall than where you started, and the walk flies apart. A step that is too small wastes time, and a step that is too big wrecks the walk.

Even a safe step is not the whole story. On one bowl the safe step, $1/L$, cuts the miss to a thousandth in 66 steps. On another bowl it needs 688. Both walks end at the bottom, so why is the second ten times slower? This chapter answers both questions by looking at one axis at a time.

## What you already know

- Gradient descent moves from $w$ to $w-\alpha\nabla f(w)$, where $\alpha$ (alpha) is the step size.
- The running bowl has height $\tfrac12(w_1-1)^2+5(w_2-1)^2-5.5$, with its bottom at $(1,1)$.
- The *miss* of a coordinate is its distance from the bottom value. The $w_2$ axis is steep and the $w_1$ axis is flat.
- With step 0.1 the steep coordinate landed on its bottom value in one step, and the flat miss went $1,\,0.9,\,0.81$.
- `np.linalg.eigvalsh(A)` returns the eigenvalues of a symmetric matrix, in ascending order.

## One axis at a time

Look at a single coordinate of the walk. On the steep axis the height is $5(w_2-1)^2$, whose slope is $10(w_2-1)$, which is 10 times the miss. A step moves you by $\alpha$ times that slope, backwards. So the next miss is the old miss minus $10\alpha$ times the old miss:

$$
\text{miss}_{\text{next}} = (1-10\,\alpha)\;\text{miss}
$$

Read it aloud as "each step multiplies the miss by one minus ten times the step size." On the flat axis the same argument gives $1-1\cdot\alpha$. The matrix $A$ is diagonal, so the axes do not mix: each coordinate follows its own multiplication.

Try $\alpha=0.3$. The steep factor is $1-3=-2$. Count the miss as positive when you are short of the bottom, so $w_2=0$ starts at a miss of 1. The misses go $1,\,-2,\,4,\,-8$. The sign flips and the size doubles, which is the blow-up you saw. The flat factor is $1-0.3=0.7$, so the flat misses shrink: $1,\,0.7,\,0.49,\,0.343$. One axis converges and the other explodes, and the explosion wins.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Distance from the bottom along the steep axis over four steps for four step sizes">
<defs></defs>
<line x1="70.0" y1="250.0" x2="330.0" y2="250.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="250.0" class="stroke-grid" />
<line x1="70.0" y1="140.0" x2="330.0" y2="140.0" class="stroke-muted dashed" />
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted" />
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="156.7" y1="250.0" x2="156.7" y2="255.0" class="stroke-muted" />
<text x="156.7" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="243.3" y1="250.0" x2="243.3" y2="255.0" class="stroke-muted" />
<text x="243.3" y="270.0" class="fig-small" text-anchor="middle">2</text>
<line x1="330.0" y1="250.0" x2="330.0" y2="255.0" class="stroke-muted" />
<text x="330.0" y="270.0" class="fig-small" text-anchor="middle">3</text>
<text x="62.0" y="241.8" class="fig-small" text-anchor="end">-8</text>
<line x1="70.0" y1="237.8" x2="330.0" y2="237.8" class="stroke-grid" />
<text x="62.0" y="192.9" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="188.9" x2="330.0" y2="188.9" class="stroke-grid" />
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">0</text>
<text x="62.0" y="95.1" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="91.1" x2="330.0" y2="91.1" class="stroke-grid" />
<text x="62.0" y="46.2" class="fig-small" text-anchor="end">8</text>
<line x1="70.0" y1="42.2" x2="330.0" y2="42.2" class="stroke-grid" />
<text x="200.0" y="292.0" class="fig-small" text-anchor="middle">step number</text>
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">signed distance from the bottom, steep axis</text>
<polyline points="70.0,127.8 156.7,133.9 243.3,136.9 330.0,138.5" fill="none" class="stroke-c" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-c"/>
<circle cx="156.7" cy="133.9" r="3.5" class="fill-c"/>
<circle cx="243.3" cy="136.9" r="3.5" class="fill-c"/>
<circle cx="330.0" cy="138.5" r="3.5" class="fill-c"/>
<line x1="360.0" y1="80.0" x2="386.0" y2="80.0" class="stroke-c" />
<circle cx="373.0" cy="80.0" r="3.5" class="fill-c"/>
<text x="394.0" y="84.0" class="fig-small" text-anchor="start">step 0.05: creeps</text>
<polyline points="70.0,127.8 156.7,140.0 243.3,140.0 330.0,140.0" fill="none" class="stroke-a" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-a"/>
<circle cx="156.7" cy="140.0" r="3.5" class="fill-a"/>
<circle cx="243.3" cy="140.0" r="3.5" class="fill-a"/>
<circle cx="330.0" cy="140.0" r="3.5" class="fill-a"/>
<line x1="360.0" y1="110.0" x2="386.0" y2="110.0" class="stroke-a" />
<circle cx="373.0" cy="110.0" r="3.5" class="fill-a"/>
<text x="394.0" y="114.0" class="fig-small" text-anchor="start">step 0.1: lands at 0</text>
<polyline points="70.0,127.8 156.7,149.8 243.3,132.2 330.0,146.3" fill="none" class="stroke-d" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-d"/>
<circle cx="156.7" cy="149.8" r="3.5" class="fill-d"/>
<circle cx="243.3" cy="132.2" r="3.5" class="fill-d"/>
<circle cx="330.0" cy="146.3" r="3.5" class="fill-d"/>
<line x1="360.0" y1="140.0" x2="386.0" y2="140.0" class="stroke-d" />
<circle cx="373.0" cy="140.0" r="3.5" class="fill-d"/>
<text x="394.0" y="144.0" class="fig-small" text-anchor="start">step 0.18: overshoots, settles</text>
<polyline points="70.0,127.8 156.7,164.4 243.3,91.1 330.0,237.8" fill="none" class="stroke-bad" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-bad"/>
<circle cx="156.7" cy="164.4" r="3.5" class="fill-bad"/>
<circle cx="243.3" cy="91.1" r="3.5" class="fill-bad"/>
<circle cx="330.0" cy="237.8" r="3.5" class="fill-bad"/>
<line x1="360.0" y1="170.0" x2="386.0" y2="170.0" class="stroke-bad" />
<circle cx="373.0" cy="170.0" r="3.5" class="fill-bad"/>
<text x="394.0" y="174.0" class="fig-small" text-anchor="start">step 0.3: swings double</text>
</svg>
Figure 1. The same start, distance 1 from the bottom, with four step sizes. A step of 0.05 creeps (green). A step of 0.1 lands exactly on the bottom (blue). A step of 0.18 overshoots and flips sides each time, but the swings shrink (purple). A step of 0.3 overshoots by more than it started with, so the swings double: 1, -2, 4, -8 (red).
```

The number that matters on each axis is its *curvature*, how fast the slope itself changes: the slope $10(w_2-1)$ grows by 10 for every unit you move, and the flat slope $1(w_1-1)$ grows by 1. So the curvatures here are 10 and 1. For a general symmetric $A$, the curvatures along the bowl's own axes are its *eigenvalues*, written $\lambda$ (lambda). The largest is $L$ and the smallest is $\mu$ (mu). Here $L=10$ and $\mu=1$. If $A$ is not diagonal, the bowl's axes are tilted, and the same argument holds along each tilted axis. You read $L$ and $\mu$ from `np.linalg.eigvalsh(A)`.

The factor for one axis is $1-\alpha\lambda$. Figure 2 plots its size against the step size. Read the steep axis, in orange, from left to right:

- **Below $1/L=0.1$**: the factor is between 0 and 1, so you creep toward the bottom.
- **At $1/L$**: the factor is 0, so you land on the bottom value in one step.
- **Between $1/L$ and $2/L$**: the factor is between $-1$ and 0, so you overshoot to the other side, but the size still shrinks.
- **At $2/L=0.2$**: the factor is $-1$, so you swing between two points forever.
- **Above $2/L$**: the size of the factor exceeds 1, so the misses grow.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Shrink factor per step against step size for a steep axis and a flat axis">
<defs></defs>
<line x1="70.0" y1="260.0" x2="560.0" y2="260.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="260.0" class="stroke-grid" />
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted" />
<text x="70.0" y="280.0" class="fig-small" text-anchor="middle">0</text>
<line x1="233.3" y1="260.0" x2="233.3" y2="265.0" class="stroke-muted" />
<text x="233.3" y="280.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="396.7" y1="260.0" x2="396.7" y2="265.0" class="stroke-muted" />
<text x="396.7" y="280.0" class="fig-small" text-anchor="middle">0.2</text>
<line x1="560.0" y1="260.0" x2="560.0" y2="265.0" class="stroke-muted" />
<text x="560.0" y="280.0" class="fig-small" text-anchor="middle">0.3</text>
<text x="62.0" y="264.0" class="fig-small" text-anchor="end">0</text>
<text x="62.0" y="209.2" class="fig-small" text-anchor="end">0.5</text>
<text x="62.0" y="154.5" class="fig-small" text-anchor="end">1</text>
<text x="62.0" y="99.7" class="fig-small" text-anchor="end">1.5</text>
<text x="62.0" y="45.0" class="fig-small" text-anchor="end">2</text>
<text x="315.0" y="322.0" class="fig-small" text-anchor="middle">step size</text>
<line x1="70.0" y1="150.5" x2="560.0" y2="150.5" class="stroke-muted dashed" />
<polyline points="70.0,150.5 74.1,153.2 78.2,156.0 82.2,158.7 86.3,161.4 90.4,164.2 94.5,166.9 98.6,169.6 102.7,172.4 106.8,175.1 110.8,177.9 114.9,180.6 119.0,183.3 123.1,186.1 127.2,188.8 131.2,191.5 135.3,194.3 139.4,197.0 143.5,199.8 147.6,202.5 151.7,205.2 155.8,208.0 159.8,210.7 163.9,213.5 168.0,216.2 172.1,218.9 176.2,221.7 180.2,224.4 184.3,227.1 188.4,229.9 192.5,232.6 196.6,235.4 200.7,238.1 204.8,240.8 208.8,243.6 212.9,246.3 217.0,249.0 221.1,251.8 225.2,254.5 229.2,257.3 233.3,260.0 237.4,257.3 241.5,254.5 245.6,251.8 249.7,249.0 253.8,246.3 257.8,243.6 261.9,240.8 266.0,238.1 270.1,235.4 274.2,232.6 278.2,229.9 282.3,227.1 286.4,224.4 290.5,221.7 294.6,218.9 298.7,216.2 302.8,213.5 306.8,210.7 310.9,208.0 315.0,205.2 319.1,202.5 323.2,199.8 327.2,197.0 331.3,194.3 335.4,191.5 339.5,188.8 343.6,186.1 347.7,183.3 351.8,180.6 355.8,177.9 359.9,175.1 364.0,172.4 368.1,169.6 372.2,166.9 376.2,164.2 380.3,161.4 384.4,158.7 388.5,156.0 392.6,153.2 396.7,150.5 400.8,147.7 404.8,145.0 408.9,142.3 413.0,139.5 417.1,136.8 421.2,134.0 425.2,131.3 429.3,128.6 433.4,125.8 437.5,123.1 441.6,120.4 445.7,117.6 449.8,114.9 453.8,112.1 457.9,109.4 462.0,106.7 466.1,103.9 470.2,101.2 474.3,98.5 478.3,95.7 482.4,93.0 486.5,90.2 490.6,87.5 494.7,84.8 498.8,82.0 502.8,79.3 506.9,76.5 511.0,73.8 515.1,71.1 519.2,68.3 523.2,65.6 527.3,62.9 531.4,60.1 535.5,57.4 539.6,54.6 543.7,51.9 547.8,49.2 551.8,46.4 555.9,43.7 560.0,41.0" fill="none" class="stroke-b" />
<polyline points="70.0,150.5 74.1,150.8 78.2,151.0 82.2,151.3 86.3,151.6 90.4,151.8 94.5,152.1 98.6,152.4 102.7,152.7 106.8,152.9 110.8,153.2 114.9,153.5 119.0,153.8 123.1,154.0 127.2,154.3 131.2,154.6 135.3,154.9 139.4,155.1 143.5,155.4 147.6,155.7 151.7,156.0 155.8,156.2 159.8,156.5 163.9,156.8 168.0,157.0 172.1,157.3 176.2,157.6 180.2,157.9 184.3,158.1 188.4,158.4 192.5,158.7 196.6,159.0 200.7,159.2 204.8,159.5 208.8,159.8 212.9,160.1 217.0,160.3 221.1,160.6 225.2,160.9 229.2,161.2 233.3,161.4 237.4,161.7 241.5,162.0 245.6,162.2 249.7,162.5 253.8,162.8 257.8,163.1 261.9,163.3 266.0,163.6 270.1,163.9 274.2,164.2 278.2,164.4 282.3,164.7 286.4,165.0 290.5,165.3 294.6,165.5 298.7,165.8 302.8,166.1 306.8,166.4 310.9,166.6 315.0,166.9 319.1,167.2 323.2,167.5 327.2,167.7 331.3,168.0 335.4,168.3 339.5,168.5 343.6,168.8 347.7,169.1 351.8,169.4 355.8,169.6 359.9,169.9 364.0,170.2 368.1,170.5 372.2,170.7 376.2,171.0 380.3,171.3 384.4,171.6 388.5,171.8 392.6,172.1 396.7,172.4 400.8,172.7 404.8,172.9 408.9,173.2 413.0,173.5 417.1,173.8 421.2,174.0 425.2,174.3 429.3,174.6 433.4,174.8 437.5,175.1 441.6,175.4 445.7,175.7 449.8,175.9 453.8,176.2 457.9,176.5 462.0,176.8 466.1,177.0 470.2,177.3 474.3,177.6 478.3,177.9 482.4,178.1 486.5,178.4 490.6,178.7 494.7,179.0 498.8,179.2 502.8,179.5 506.9,179.8 511.0,180.0 515.1,180.3 519.2,180.6 523.2,180.9 527.3,181.1 531.4,181.4 535.5,181.7 539.6,182.0 543.7,182.2 547.8,182.5 551.8,182.8 555.9,183.1 560.0,183.3" fill="none" class="stroke-a" />
<circle cx="233.3" cy="260.0" r="4.5" class="fill-b"/>
<text x="233.3" y="300.0" class="fig-small" text-anchor="middle">factor 0: lands on the bottom</text>
<circle cx="396.7" cy="150.5" r="4.5" class="fill-b"/>
<text x="386.7" y="140.5" class="fig-small" text-anchor="end">2/L = 0.2: factor 1, swings forever</text>
<circle cx="233.3" cy="161.4" r="4.5" class="fill-a"/>
<line x1="100.0" y1="48.0" x2="130.0" y2="48.0" class="stroke-b" />
<text x="138.0" y="52.0" class="fig-small" text-anchor="start">steep axis, curvature 10</text>
<line x1="100.0" y1="68.0" x2="130.0" y2="68.0" class="stroke-a" />
<text x="138.0" y="72.0" class="fig-small" text-anchor="start">flat axis, curvature 1</text>
<line x1="100.0" y1="88.0" x2="130.0" y2="88.0" class="stroke-muted dashed" />
<text x="138.0" y="92.0" class="fig-small" text-anchor="start">factor 1: the distance stops shrinking</text>
</svg>
Figure 2. The size of the factor against the step size, for the steep axis (orange) and the flat axis (blue). The orange line crosses 1 at a step of 0.2, which is 2/L with L = 10, so that is where one axis stops converging. At a step of 0.1 the orange factor is 0 and the blue factor is 0.9. The blue line stays below 1 in this range, but only just, so the flat axis shrinks slowly.
```

The flat axis never gets in trouble first. Its factor $1-\alpha\mu$ stays between 0.8 and 1 for every step below $0.2$. The steepest axis blows up first, so plain gradient descent on a bowl converges from every start when

$$
0 < \alpha < \frac{2}{L}
$$

Read it aloud as "the step must be positive and below two over the largest curvature." The common safe choice is $1/L$, because it stays inside the limit and zeroes the steepest axis. In numpy, the factors for all axes at once are:

```python
factors = np.abs(1 - step * eigenvalues)  # one per axis
```

If any entry is above 1, that axis grows. For a loss without a matrix you cannot read $L$ off, so people start with a guess and halve the step whenever the loss rises. That is the $2/L$ rule applied by hand.

> [!KEY]
> On each axis a step multiplies your distance from the bottom by $1-\alpha\lambda$. The axis with the largest curvature $L$ blows up first, so the step must stay below $2/L$, and $1/L$ is the usual safe choice.

## The slow axis sets the pace

With step $1/L$, the steep axis has factor $1-L/L=0$ and vanishes after one step. Everything after that is the flat axis, with factor $1-\mu/L$. On the running bowl that is $1-1/10=0.9$.

To compare walks, count the steps needed to cut the miss to one thousandth of its start. Start off only along the flat axis, at $(0,1)$, so the starting miss is 1. The miss after $k$ steps is $0.9^k$, and $0.9^{66}\approx0.00096$ is the first power under 0.001. That is the 66 steps.

Stretch the bowl by making the steep axis ten times as curved: $A=\text{diag}(1,100)$, $b=(1,100)$, with the same bottom $(1,1)$. Here $L=100$ and the safe step is $0.01$. The flat factor is $1-1/100=0.99$, and $0.99^{688}\approx0.00099$ is the first power under 0.001. That is the 688 steps.

Could a longer step help? The longest allowed step is just under $2/L$, and it only changes the flat factor from $1-\mu/L$ to about $1-2\mu/L$. That roughly halves the count and does not close a gap of ten times. The valley floor is nearly flat, and the steep walls stop you from pushing harder.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two bowls viewed in the same window: round contours and a long thin valley, each with its walk">
<defs><marker id="c010-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><clipPath id="c010-clip0"><rect x="20" y="5" width="270" height="270"/></clipPath><clipPath id="c010-clip1"><rect x="310" y="5" width="270" height="270"/></clipPath></defs>
<rect x="20" y="5" width="270" height="270" class="stroke-grid" fill="none"/>
<g clip-path="url(#c010-clip0)">
<ellipse cx="155.0" cy="140.0" rx="42.2" ry="42.2" class="stroke-grid" fill="none"/>
<ellipse cx="155.0" cy="140.0" rx="84.4" ry="84.4" class="stroke-grid" fill="none"/>
<ellipse cx="155.0" cy="140.0" rx="119.3" ry="119.3" class="stroke-a" fill="none"/>
</g>
<line x1="70.6" y1="224.4" x2="149.1" y2="145.9" class="stroke-b" marker-end="url(#c010-arrow-b)"/>
<circle cx="70.6" cy="224.4" r="4" class="fill-b"/>
<circle cx="155.0" cy="140.0" r="4" class="fill-ink"/>
<text x="155.0" y="297.0" class="fig-small" text-anchor="middle">circle (condition number 1)</text>
<text x="163.0" y="132.0" class="fig-small" text-anchor="start">bottom, one step</text>
<text x="64.6" y="240.4" class="fig-small" text-anchor="end">start</text>
<rect x="310" y="5" width="270" height="270" class="stroke-grid" fill="none"/>
<g clip-path="url(#c010-clip1)">
<ellipse cx="445.0" cy="140.0" rx="848.0" ry="84.8" class="stroke-a" fill="none"/>
<ellipse cx="445.0" cy="140.0" rx="533.6" ry="53.4" class="stroke-grid" fill="none"/>
<ellipse cx="445.0" cy="140.0" rx="266.8" ry="26.7" class="stroke-grid" fill="none"/>
<ellipse cx="445.0" cy="140.0" rx="119.3" ry="11.9" class="stroke-grid" fill="none"/>
</g>
<polyline points="360.6,224.4 361.5,140.0" fill="none" class="stroke-b" />
<polyline points="361.5,140.0 362.3,140.0 363.1,140.0 363.9,140.0 364.8,140.0 365.6,140.0 366.4,140.0 367.1,140.0 367.9,140.0 368.7,140.0 369.5,140.0 370.2,140.0 371.0,140.0 371.7,140.0 372.4,140.0 373.2,140.0 373.9,140.0 374.6,140.0 375.3,140.0 376.0,140.0 376.7,140.0 377.4,140.0 378.0,140.0 378.7,140.0 379.4,140.0 380.0,140.0 380.7,140.0 381.3,140.0 382.0,140.0 382.6,140.0 383.2,140.0 383.8,140.0 384.4,140.0 385.0,140.0 385.6,140.0 386.2,140.0 386.8,140.0 387.4,140.0 388.0,140.0 388.6,140.0 389.1,140.0 389.7,140.0 390.2,140.0 390.8,140.0 391.3,140.0 391.9,140.0 392.4,140.0 392.9,140.0 393.4,140.0 394.0,140.0 394.5,140.0 395.0,140.0 395.5,140.0 396.0,140.0 396.5,140.0 396.9,140.0 397.4,140.0 397.9,140.0 398.4,140.0 398.8,140.0 399.3,140.0 399.8,140.0 400.2,140.0 400.7,140.0 401.1,140.0 401.5,140.0 402.0,140.0 402.4,140.0 402.8,140.0 403.2,140.0 403.7,140.0 404.1,140.0 404.5,140.0 404.9,140.0 405.3,140.0 405.7,140.0 406.1,140.0 406.5,140.0 406.9,140.0 407.2,140.0 407.6,140.0 408.0,140.0 408.4,140.0 408.7,140.0 409.1,140.0 409.4,140.0 409.8,140.0 410.2,140.0 410.5,140.0 410.9,140.0 411.2,140.0 411.5,140.0 411.9,140.0 412.2,140.0 412.5,140.0 412.8,140.0 413.2,140.0 413.5,140.0 413.8,140.0 414.1,140.0 414.4,140.0 414.7,140.0 415.0,140.0 415.3,140.0 415.6,140.0 415.9,140.0 416.2,140.0 416.5,140.0 416.8,140.0 417.1,140.0 417.3,140.0 417.6,140.0 417.9,140.0 418.2,140.0 418.4,140.0 418.7,140.0 419.0,140.0 419.2,140.0 419.5,140.0 419.7,140.0 420.0,140.0 420.2,140.0 420.5,140.0 420.7,140.0 421.0,140.0 421.2,140.0 421.5,140.0 421.7,140.0 421.9,140.0 422.2,140.0 422.4,140.0 422.6,140.0 422.8,140.0 423.1,140.0 423.3,140.0 423.5,140.0 423.7,140.0 423.9,140.0 424.1,140.0 424.3,140.0 424.5,140.0 424.8,140.0 425.0,140.0 425.2,140.0 425.4,140.0 425.5,140.0 425.7,140.0 425.9,140.0 426.1,140.0 426.3,140.0 426.5,140.0 426.7,140.0 426.9,140.0 427.1,140.0 427.2,140.0 427.4,140.0 427.6,140.0 427.8,140.0 427.9,140.0 428.1,140.0 428.3,140.0 428.4,140.0 428.6,140.0 428.8,140.0 428.9,140.0 429.1,140.0 429.2,140.0 429.4,140.0 429.6,140.0 429.7,140.0 429.9,140.0 430.0,140.0 430.2,140.0 430.3,140.0 430.5,140.0 430.6,140.0 430.8,140.0 430.9,140.0 431.0,140.0 431.2,140.0 431.3,140.0 431.5,140.0 431.6,140.0 431.7,140.0 431.9,140.0 432.0,140.0 432.1,140.0 432.2,140.0 432.4,140.0 432.5,140.0 432.6,140.0 432.7,140.0 432.9,140.0 433.0,140.0 433.1,140.0 433.2,140.0 433.3,140.0 433.5,140.0 433.6,140.0 433.7,140.0 433.8,140.0 433.9,140.0 434.0,140.0 434.1,140.0 434.2,140.0 434.4,140.0 434.5,140.0 434.6,140.0 434.7,140.0 434.8,140.0 434.9,140.0 435.0,140.0 435.1,140.0 435.2,140.0 435.3,140.0 435.4,140.0 435.5,140.0 435.6,140.0 435.7,140.0 435.8,140.0 435.8,140.0 435.9,140.0 436.0,140.0 436.1,140.0 436.2,140.0 436.3,140.0 436.4,140.0 436.5,140.0 436.6,140.0 436.6,140.0 436.7,140.0 436.8,140.0 436.9,140.0 437.0,140.0 437.0,140.0 437.1,140.0 437.2,140.0 437.3,140.0 437.4,140.0 437.4,140.0 437.5,140.0 437.6,140.0 437.7,140.0 437.7,140.0 437.8,140.0 437.9,140.0 438.0,140.0 438.0,140.0 438.1,140.0 438.2,140.0 438.2,140.0 438.3,140.0 438.4,140.0 438.4,140.0 438.5,140.0 438.6,140.0 438.6,140.0 438.7,140.0 438.8,140.0 438.8,140.0 438.9,140.0 438.9,140.0 439.0,140.0 439.1,140.0 439.1,140.0 439.2,140.0 439.2,140.0 439.3,140.0 439.3,140.0 439.4,140.0 439.5,140.0 439.5,140.0 439.6,140.0 439.6,140.0 439.7,140.0 439.7,140.0 439.8,140.0 439.8,140.0 439.9,140.0 439.9,140.0 440.0,140.0 440.0,140.0 440.1,140.0 440.1,140.0 440.2,140.0 440.2,140.0 440.3,140.0 440.3,140.0 440.4,140.0 440.4,140.0 440.5,140.0 440.5,140.0 440.6,140.0 440.6,140.0 440.6,140.0 440.7,140.0 440.7,140.0 440.8,140.0 440.8,140.0 440.9,140.0 440.9,140.0 440.9,140.0 441.0,140.0 441.0,140.0 441.1,140.0 441.1,140.0 441.1,140.0 441.2,140.0 441.2,140.0 441.3,140.0 441.3,140.0 441.3,140.0 441.4,140.0 441.4,140.0 441.4,140.0 441.5,140.0 441.5,140.0 441.5,140.0 441.6,140.0 441.6,140.0 441.6,140.0 441.7,140.0 441.7,140.0 441.7,140.0 441.8,140.0 441.8,140.0 441.8,140.0 441.9,140.0 441.9,140.0 441.9,140.0 442.0,140.0 442.0,140.0 442.0,140.0 442.1,140.0 442.1,140.0 442.1,140.0 442.1,140.0 442.2,140.0 442.2,140.0 442.2,140.0 442.3,140.0 442.3,140.0 442.3,140.0 442.3,140.0 442.4,140.0 442.4,140.0 442.4,140.0 442.4,140.0 442.5,140.0 442.5,140.0 442.5,140.0 442.5,140.0 442.6,140.0 442.6,140.0 442.6,140.0 442.6,140.0 442.7,140.0 442.7,140.0 442.7,140.0 442.7,140.0 442.8,140.0 442.8,140.0 442.8,140.0 442.8,140.0 442.8,140.0 442.9,140.0 442.9,140.0 442.9,140.0 442.9,140.0 443.0,140.0 443.0,140.0 443.0,140.0 443.0,140.0 443.0,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.2,140.0 443.2,140.0 443.2,140.0 443.2,140.0 443.2,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0" fill="none" class="stroke-b" />
<circle cx="361.5" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="414.1" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="436.6" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="444.2" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="360.6" cy="224.4" r="4" class="fill-b"/>
<circle cx="445.0" cy="140.0" r="4" class="fill-ink"/>
<text x="445.0" y="297.0" class="fig-small" text-anchor="middle">long valley (condition number 100)</text>
<text x="414.1" y="116.0" class="fig-small" text-anchor="middle">step 100</text>
<text x="436.6" y="170.0" class="fig-small" text-anchor="middle">step 230</text>
<text x="353.5" y="130.0" class="fig-small" text-anchor="end">step 1</text>
<text x="354.6" y="240.4" class="fig-small" text-anchor="end">start</text>
</svg>
Figure 3. Both panels show the same window of the plane, with the same start (0, 0) and the same bottom (1, 1). On the left the contours are circles and one step reaches the bottom. On the right the contours are so stretched that, in this window, they look like parallel lines: the first step climbs the steep wall to the floor, and then the walk crawls along the floor. Step 100 is still 0.37 short of the bottom along the floor, step 230 is still 0.1 short, and step 460 is almost on top of it.
```

The ratio of the largest to the smallest curvature is the *condition number*, written $\kappa$ (kappa):

$$
\kappa = \frac{L}{\mu}
$$

Read it aloud as "the sharpest curvature divided by the flattest." With step $1/L$ the flat factor is $1-1/\kappa$. For the two bowls, $\kappa$ is 10 and 100. In numpy, `kappa = L / mu` from the eigenvalue range. The condition number is also the bowl's shape: the lines of equal height are ellipses whose long side is $\sqrt{\kappa}$ times the short side. With $\kappa=1$ they are circles, and one step reaches the bottom, because every direction points at it.

## The count grows with kappa

Each step multiplies the flat miss by the same factor, so after $k$ steps the miss is $(1-1/\kappa)^k$ times the start. On a log scale that is a straight line, and the steepness of the line is how fast you win. Figure 4 draws the two bowls.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Distance from the bottom on a log scale for two valleys, with the step count to reach one thousandth">
<defs></defs>
<line x1="70.0" y1="260.0" x2="560.0" y2="260.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="260.0" class="stroke-grid" />
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted" />
<text x="70.0" y="280.0" class="fig-small" text-anchor="middle">0</text>
<line x1="140.0" y1="260.0" x2="140.0" y2="265.0" class="stroke-muted" />
<text x="140.0" y="280.0" class="fig-small" text-anchor="middle">100</text>
<line x1="210.0" y1="260.0" x2="210.0" y2="265.0" class="stroke-muted" />
<text x="210.0" y="280.0" class="fig-small" text-anchor="middle">200</text>
<line x1="280.0" y1="260.0" x2="280.0" y2="265.0" class="stroke-muted" />
<text x="280.0" y="280.0" class="fig-small" text-anchor="middle">300</text>
<line x1="350.0" y1="260.0" x2="350.0" y2="265.0" class="stroke-muted" />
<text x="350.0" y="280.0" class="fig-small" text-anchor="middle">400</text>
<line x1="420.0" y1="260.0" x2="420.0" y2="265.0" class="stroke-muted" />
<text x="420.0" y="280.0" class="fig-small" text-anchor="middle">500</text>
<line x1="490.0" y1="260.0" x2="490.0" y2="265.0" class="stroke-muted" />
<text x="490.0" y="280.0" class="fig-small" text-anchor="middle">600</text>
<line x1="560.0" y1="260.0" x2="560.0" y2="265.0" class="stroke-muted" />
<text x="560.0" y="280.0" class="fig-small" text-anchor="middle">700</text>
<line x1="70.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid" />
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="95.7" x2="560.0" y2="95.7" class="stroke-grid" />
<text x="62.0" y="99.7" class="fig-small" text-anchor="end">10^-1</text>
<line x1="70.0" y1="161.4" x2="560.0" y2="161.4" class="stroke-grid" />
<text x="62.0" y="165.4" class="fig-small" text-anchor="end">10^-2</text>
<line x1="70.0" y1="227.1" x2="560.0" y2="227.1" class="stroke-grid" />
<text x="62.0" y="231.1" class="fig-small" text-anchor="end">10^-3</text>
<text x="315.0" y="300.0" class="fig-small" text-anchor="middle">step number</text>
<polyline points="70.0,30.0 70.7,33.0 71.4,36.0 72.1,39.0 72.8,42.0 73.5,45.0 74.2,48.0 74.9,51.0 75.6,54.1 76.3,57.1 77.0,60.1 77.7,63.1 78.4,66.1 79.1,69.1 79.8,72.1 80.5,75.1 81.2,78.1 81.9,81.1 82.6,84.1 83.3,87.1 84.0,90.1 84.7,93.1 85.4,96.2 86.1,99.2 86.8,102.2 87.5,105.2 88.2,108.2 88.9,111.2 89.6,114.2 90.3,117.2 91.0,120.2 91.7,123.2 92.4,126.2 93.1,129.2 93.8,132.2 94.5,135.2 95.2,138.2 95.9,141.3 96.6,144.3 97.3,147.3 98.0,150.3 98.7,153.3 99.4,156.3 100.1,159.3 100.8,162.3 101.5,165.3 102.2,168.3 102.9,171.3 103.6,174.3 104.3,177.3 105.0,180.3 105.7,183.4 106.4,186.4 107.1,189.4 107.8,192.4 108.5,195.4 109.2,198.4 109.9,201.4 110.6,204.4 111.3,207.4 112.0,210.4 112.7,213.4 113.4,216.4 114.1,219.4 114.8,222.4 115.5,225.4 116.2,227.1" fill="none" class="stroke-b" />
<polyline points="70.0,30.0 70.7,30.3 71.4,30.6 72.1,30.9 72.8,31.1 73.5,31.4 74.2,31.7 74.9,32.0 75.6,32.3 76.3,32.6 77.0,32.9 77.7,33.2 78.4,33.4 79.1,33.7 79.8,34.0 80.5,34.3 81.2,34.6 81.9,34.9 82.6,35.2 83.3,35.4 84.0,35.7 84.7,36.0 85.4,36.3 86.1,36.6 86.8,36.9 87.5,37.2 88.2,37.5 88.9,37.7 89.6,38.0 90.3,38.3 91.0,38.6 91.7,38.9 92.4,39.2 93.1,39.5 93.8,39.8 94.5,40.0 95.2,40.3 95.9,40.6 96.6,40.9 97.3,41.2 98.0,41.5 98.7,41.8 99.4,42.0 100.1,42.3 100.8,42.6 101.5,42.9 102.2,43.2 102.9,43.5 103.6,43.8 104.3,44.1 105.0,44.3 105.7,44.6 106.4,44.9 107.1,45.2 107.8,45.5 108.5,45.8 109.2,46.1 109.9,46.3 110.6,46.6 111.3,46.9 112.0,47.2 112.7,47.5 113.4,47.8 114.1,48.1 114.8,48.4 115.5,48.6 116.2,48.9 116.9,49.2 117.6,49.5 118.3,49.8 119.0,50.1 119.7,50.4 120.4,50.7 121.1,50.9 121.8,51.2 122.5,51.5 123.2,51.8 123.9,52.1 124.6,52.4 125.3,52.7 126.0,52.9 126.7,53.2 127.4,53.5 128.1,53.8 128.8,54.1 129.5,54.4 130.2,54.7 130.9,55.0 131.6,55.2 132.3,55.5 133.0,55.8 133.7,56.1 134.4,56.4 135.1,56.7 135.8,57.0 136.5,57.2 137.2,57.5 137.9,57.8 138.6,58.1 139.3,58.4 140.0,58.7 140.7,59.0 141.4,59.3 142.1,59.5 142.8,59.8 143.5,60.1 144.2,60.4 144.9,60.7 145.6,61.0 146.3,61.3 147.0,61.6 147.7,61.8 148.4,62.1 149.1,62.4 149.8,62.7 150.5,63.0 151.2,63.3 151.9,63.6 152.6,63.8 153.3,64.1 154.0,64.4 154.7,64.7 155.4,65.0 156.1,65.3 156.8,65.6 157.5,65.9 158.2,66.1 158.9,66.4 159.6,66.7 160.3,67.0 161.0,67.3 161.7,67.6 162.4,67.9 163.1,68.1 163.8,68.4 164.5,68.7 165.2,69.0 165.9,69.3 166.6,69.6 167.3,69.9 168.0,70.2 168.7,70.4 169.4,70.7 170.1,71.0 170.8,71.3 171.5,71.6 172.2,71.9 172.9,72.2 173.6,72.5 174.3,72.7 175.0,73.0 175.7,73.3 176.4,73.6 177.1,73.9 177.8,74.2 178.5,74.5 179.2,74.7 179.9,75.0 180.6,75.3 181.3,75.6 182.0,75.9 182.7,76.2 183.4,76.5 184.1,76.8 184.8,77.0 185.5,77.3 186.2,77.6 186.9,77.9 187.6,78.2 188.3,78.5 189.0,78.8 189.7,79.0 190.4,79.3 191.1,79.6 191.8,79.9 192.5,80.2 193.2,80.5 193.9,80.8 194.6,81.1 195.3,81.3 196.0,81.6 196.7,81.9 197.4,82.2 198.1,82.5 198.8,82.8 199.5,83.1 200.2,83.4 200.9,83.6 201.6,83.9 202.3,84.2 203.0,84.5 203.7,84.8 204.4,85.1 205.1,85.4 205.8,85.6 206.5,85.9 207.2,86.2 207.9,86.5 208.6,86.8 209.3,87.1 210.0,87.4 210.7,87.7 211.4,87.9 212.1,88.2 212.8,88.5 213.5,88.8 214.2,89.1 214.9,89.4 215.6,89.7 216.3,89.9 217.0,90.2 217.7,90.5 218.4,90.8 219.1,91.1 219.8,91.4 220.5,91.7 221.2,92.0 221.9,92.2 222.6,92.5 223.3,92.8 224.0,93.1 224.7,93.4 225.4,93.7 226.1,94.0 226.8,94.2 227.5,94.5 228.2,94.8 228.9,95.1 229.6,95.4 230.3,95.7 231.0,96.0 231.7,96.3 232.4,96.5 233.1,96.8 233.8,97.1 234.5,97.4 235.2,97.7 235.9,98.0 236.6,98.3 237.3,98.6 238.0,98.8 238.7,99.1 239.4,99.4 240.1,99.7 240.8,100.0 241.5,100.3 242.2,100.6 242.9,100.8 243.6,101.1 244.3,101.4 245.0,101.7 245.7,102.0 246.4,102.3 247.1,102.6 247.8,102.9 248.5,103.1 249.2,103.4 249.9,103.7 250.6,104.0 251.3,104.3 252.0,104.6 252.7,104.9 253.4,105.1 254.1,105.4 254.8,105.7 255.5,106.0 256.2,106.3 256.9,106.6 257.6,106.9 258.3,107.2 259.0,107.4 259.7,107.7 260.4,108.0 261.1,108.3 261.8,108.6 262.5,108.9 263.2,109.2 263.9,109.5 264.6,109.7 265.3,110.0 266.0,110.3 266.7,110.6 267.4,110.9 268.1,111.2 268.8,111.5 269.5,111.7 270.2,112.0 270.9,112.3 271.6,112.6 272.3,112.9 273.0,113.2 273.7,113.5 274.4,113.8 275.1,114.0 275.8,114.3 276.5,114.6 277.2,114.9 277.9,115.2 278.6,115.5 279.3,115.8 280.0,116.0 280.7,116.3 281.4,116.6 282.1,116.9 282.8,117.2 283.5,117.5 284.2,117.8 284.9,118.1 285.6,118.3 286.3,118.6 287.0,118.9 287.7,119.2 288.4,119.5 289.1,119.8 289.8,120.1 290.5,120.4 291.2,120.6 291.9,120.9 292.6,121.2 293.3,121.5 294.0,121.8 294.7,122.1 295.4,122.4 296.1,122.6 296.8,122.9 297.5,123.2 298.2,123.5 298.9,123.8 299.6,124.1 300.3,124.4 301.0,124.7 301.7,124.9 302.4,125.2 303.1,125.5 303.8,125.8 304.5,126.1 305.2,126.4 305.9,126.7 306.6,126.9 307.3,127.2 308.0,127.5 308.7,127.8 309.4,128.1 310.1,128.4 310.8,128.7 311.5,129.0 312.2,129.2 312.9,129.5 313.6,129.8 314.3,130.1 315.0,130.4 315.7,130.7 316.4,131.0 317.1,131.3 317.8,131.5 318.5,131.8 319.2,132.1 319.9,132.4 320.6,132.7 321.3,133.0 322.0,133.3 322.7,133.5 323.4,133.8 324.1,134.1 324.8,134.4 325.5,134.7 326.2,135.0 326.9,135.3 327.6,135.6 328.3,135.8 329.0,136.1 329.7,136.4 330.4,136.7 331.1,137.0 331.8,137.3 332.5,137.6 333.2,137.8 333.9,138.1 334.6,138.4 335.3,138.7 336.0,139.0 336.7,139.3 337.4,139.6 338.1,139.9 338.8,140.1 339.5,140.4 340.2,140.7 340.9,141.0 341.6,141.3 342.3,141.6 343.0,141.9 343.7,142.2 344.4,142.4 345.1,142.7 345.8,143.0 346.5,143.3 347.2,143.6 347.9,143.9 348.6,144.2 349.3,144.4 350.0,144.7 350.7,145.0 351.4,145.3 352.1,145.6 352.8,145.9 353.5,146.2 354.2,146.5 354.9,146.7 355.6,147.0 356.3,147.3 357.0,147.6 357.7,147.9 358.4,148.2 359.1,148.5 359.8,148.7 360.5,149.0 361.2,149.3 361.9,149.6 362.6,149.9 363.3,150.2 364.0,150.5 364.7,150.8 365.4,151.0 366.1,151.3 366.8,151.6 367.5,151.9 368.2,152.2 368.9,152.5 369.6,152.8 370.3,153.1 371.0,153.3 371.7,153.6 372.4,153.9 373.1,154.2 373.8,154.5 374.5,154.8 375.2,155.1 375.9,155.3 376.6,155.6 377.3,155.9 378.0,156.2 378.7,156.5 379.4,156.8 380.1,157.1 380.8,157.4 381.5,157.6 382.2,157.9 382.9,158.2 383.6,158.5 384.3,158.8 385.0,159.1 385.7,159.4 386.4,159.6 387.1,159.9 387.8,160.2 388.5,160.5 389.2,160.8 389.9,161.1 390.6,161.4 391.3,161.7 392.0,161.9 392.7,162.2 393.4,162.5 394.1,162.8 394.8,163.1 395.5,163.4 396.2,163.7 396.9,163.9 397.6,164.2 398.3,164.5 399.0,164.8 399.7,165.1 400.4,165.4 401.1,165.7 401.8,166.0 402.5,166.2 403.2,166.5 403.9,166.8 404.6,167.1 405.3,167.4 406.0,167.7 406.7,168.0 407.4,168.3 408.1,168.5 408.8,168.8 409.5,169.1 410.2,169.4 410.9,169.7 411.6,170.0 412.3,170.3 413.0,170.5 413.7,170.8 414.4,171.1 415.1,171.4 415.8,171.7 416.5,172.0 417.2,172.3 417.9,172.6 418.6,172.8 419.3,173.1 420.0,173.4 420.7,173.7 421.4,174.0 422.1,174.3 422.8,174.6 423.5,174.8 424.2,175.1 424.9,175.4 425.6,175.7 426.3,176.0 427.0,176.3 427.7,176.6 428.4,176.9 429.1,177.1 429.8,177.4 430.5,177.7 431.2,178.0 431.9,178.3 432.6,178.6 433.3,178.9 434.0,179.2 434.7,179.4 435.4,179.7 436.1,180.0 436.8,180.3 437.5,180.6 438.2,180.9 438.9,181.2 439.6,181.4 440.3,181.7 441.0,182.0 441.7,182.3 442.4,182.6 443.1,182.9 443.8,183.2 444.5,183.5 445.2,183.7 445.9,184.0 446.6,184.3 447.3,184.6 448.0,184.9 448.7,185.2 449.4,185.5 450.1,185.7 450.8,186.0 451.5,186.3 452.2,186.6 452.9,186.9 453.6,187.2 454.3,187.5 455.0,187.8 455.7,188.0 456.4,188.3 457.1,188.6 457.8,188.9 458.5,189.2 459.2,189.5 459.9,189.8 460.6,190.1 461.3,190.3 462.0,190.6 462.7,190.9 463.4,191.2 464.1,191.5 464.8,191.8 465.5,192.1 466.2,192.3 466.9,192.6 467.6,192.9 468.3,193.2 469.0,193.5 469.7,193.8 470.4,194.1 471.1,194.4 471.8,194.6 472.5,194.9 473.2,195.2 473.9,195.5 474.6,195.8 475.3,196.1 476.0,196.4 476.7,196.6 477.4,196.9 478.1,197.2 478.8,197.5 479.5,197.8 480.2,198.1 480.9,198.4 481.6,198.7 482.3,198.9 483.0,199.2 483.7,199.5 484.4,199.8 485.1,200.1 485.8,200.4 486.5,200.7 487.2,201.0 487.9,201.2 488.6,201.5 489.3,201.8 490.0,202.1 490.7,202.4 491.4,202.7 492.1,203.0 492.8,203.2 493.5,203.5 494.2,203.8 494.9,204.1 495.6,204.4 496.3,204.7 497.0,205.0 497.7,205.3 498.4,205.5 499.1,205.8 499.8,206.1 500.5,206.4 501.2,206.7 501.9,207.0 502.6,207.3 503.3,207.5 504.0,207.8 504.7,208.1 505.4,208.4 506.1,208.7 506.8,209.0 507.5,209.3 508.2,209.6 508.9,209.8 509.6,210.1 510.3,210.4 511.0,210.7 511.7,211.0 512.4,211.3 513.1,211.6 513.8,211.9 514.5,212.1 515.2,212.4 515.9,212.7 516.6,213.0 517.3,213.3 518.0,213.6 518.7,213.9 519.4,214.1 520.1,214.4 520.8,214.7 521.5,215.0 522.2,215.3 522.9,215.6 523.6,215.9 524.3,216.2 525.0,216.4 525.7,216.7 526.4,217.0 527.1,217.3 527.8,217.6 528.5,217.9 529.2,218.2 529.9,218.4 530.6,218.7 531.3,219.0 532.0,219.3 532.7,219.6 533.4,219.9 534.1,220.2 534.8,220.5 535.5,220.7 536.2,221.0 536.9,221.3 537.6,221.6 538.3,221.9 539.0,222.2 539.7,222.5 540.4,222.7 541.1,223.0 541.8,223.3 542.5,223.6 543.2,223.9 543.9,224.2 544.6,224.5 545.3,224.8 546.0,225.0 546.7,225.3 547.4,225.6 548.1,225.9 548.8,226.2 549.5,226.5 550.2,226.8 550.9,227.1 551.6,227.3" fill="none" class="stroke-a" />
<circle cx="116.2" cy="227.1" r="4.5" class="fill-b"/>
<text x="124.2" y="243.1" class="fig-small" text-anchor="start">66 steps</text>
<circle cx="551.6" cy="227.3" r="4.5" class="fill-a"/>
<text x="543.6" y="243.1" class="fig-small" text-anchor="end">688 steps</text>
<line x1="330.0" y1="60.0" x2="360.0" y2="60.0" class="stroke-b" />
<text x="368.0" y="64.0" class="fig-small" text-anchor="start">condition number 10: factor 0.9 per step</text>
<line x1="330.0" y1="82.0" x2="360.0" y2="82.0" class="stroke-a" />
<text x="368.0" y="86.0" class="fig-small" text-anchor="start">condition number 100: factor 0.99 per step</text>
</svg>
Figure 4. The distance along the flat axis, relative to its start, shrinks by the same factor each step, so on a log scale it is a straight line. A shrink factor of 0.9 reaches one thousandth of the start after 66 steps. A factor of 0.99 needs 688 steps, about ten times as many.
```

To cut the miss by a factor $F$, you need about $\kappa\ln F$ steps. For $F=1000$ that is about $6.9\,\kappa$. For $\kappa=10$ it is 69, close to the exact 66. For $\kappa=100$ it is 691, close to 688. The number of steps is proportional to $\kappa$. A bowl with $\kappa=10{,}000$ needs about 69,000 steps.

You might expect a zigzag on a long valley. At exactly $1/L$ there is none, because the steep factor is 0 and that coordinate stops moving. A zigzag needs a step between $1/L$ and $2/L$, where the steep factor is negative. On the $\kappa=10$ bowl with step $0.18$, the steep factor is $-0.8$ and the flat factor is $0.82$. From $(0,0)$ the walk visits $(0.18,\,1.8)$, then $(0.33,\,0.36)$, then $(0.45,\,1.51)$: the steep coordinate overshoots the floor and back while the flat coordinate creeps forward. This longer step is faster. The flat factor $0.82$ reaches one thousandth in about 35 steps, against 66, which matches the earlier claim that a step near $2/L$ roughly halves the count.

> [!KEY]
> One step size has to suit both axes, and the steepest axis caps it at $2/L$. The flat axis then shrinks by only $1-1/\kappa$ per step, so a long valley needs about $\kappa\ln F$ steps to cut the miss by a factor $F$.

## Pick the step by looking down the line

Instead of choosing a number, you can ask the bowl. Stand at $(0,0)$, where the gradient is $g=(-1,-10)$, and walk along the downhill line. After a step $\alpha$ you are at $\alpha\,(1,10)$. The height there is $\tfrac12\cdot 1001\,\alpha^2-101\,\alpha$, a parabola in $\alpha$, so its lowest point is the best step. For any bowl, expanding the height gives $f(w)-\alpha\,g^\top g+\tfrac12\alpha^2\,g^\top A g$, whose lowest point is at

$$
\alpha = \frac{g^\top g}{g^\top A\,g}
$$

Read it aloud as "the slope along the line, divided by the curvature along the line." Here $g^\top g=1+100=101$ and $g^\top A g=1+1000=1001$, so $\alpha=101/1001\approx0.1009$. This step is *exact line search*. It does not need $L$. In numpy, `alpha = (g @ g) / (g @ A @ g)`. Flip the fraction and the step is about 9.9, far above $2/L$, so the walk explodes.

At the best step the slope along the line is 0, so the next gradient is perpendicular to the old one. Check it: the point after the step is $(0.1009,\,1.0090)$, the gradient there is $(-0.8991,\,0.0899)$, and its dot product with $(-1,-10)$ is 0. On a long valley each step turns at a right angle to the last, so the walk zigzags, and exact line search does not cure the slowness.

## Make the valley round

Real bowls have large condition numbers. Take a house-price fit with one column for size in square feet, about 1000, and one for the number of bedrooms, about 3. The curvature along a column grows with the sum of its squared entries, so the size axis curves about $(1000/3)^2\approx 110{,}000$ times as sharply as the bedrooms axis. So $\kappa$ is at least about 110,000.

The cure is to put the columns on the same scale, which people call *standardising*: subtract each column's mean and divide by its standard deviation. If the columns are unrelated, $\kappa$ falls to about 1. Standardising does not remove correlation between columns, so $\kappa$ stays above 1 when features overlap. For least squares, the bowl's matrix is $A=X^\top X$, and its condition number is the square of the condition number of $X$, the same squaring that lost digits in the chapter on the textbook formula.

> [!RECAP]
> - On each axis, one step multiplies the miss by $1-\alpha\lambda$. The sharpest axis, with curvature $L$, blows up first, so plain descent needs $\alpha<2/L$, and $1/L$ is safe.
> - At step $1/L$ the flat axis shrinks by $1-1/\kappa$ per step, with $\kappa=L/\mu$.
> - Cutting the miss by a factor $F$ takes about $\kappa\ln F$ steps.
> - Exact line search uses $\alpha=(g^\top g)/(g^\top A g)$, and the next gradient is perpendicular to the old.
> - Standardising columns shrinks $\kappa$.

## Build it step by step

The starter includes `gradient_descent`, which you wrote in the previous chapter. The checks call the functions you write in this order:

1. **`eigenvalue_range(A)`** takes a symmetric matrix of shape `(d, d)` and returns `(mu, L)` as two floats, the first and last of `np.linalg.eigvalsh(A)`. Use `eigvalsh`, not `eigvals`, because it returns real numbers for a symmetric matrix.
2. **`axis_factors(A, step)`** returns `np.abs(1 - step * eigenvalues)`, one entry per eigenvalue in ascending order. For the running bowl at step 0.3 you get `[0.7, 2.0]`.
3. **`exact_line_search_step(A, grad_vec)`** returns `(grad_vec @ grad_vec) / (grad_vec @ A @ grad_vec)` as one float. Writing `grad_vec @ grad_vec @ A` gives a nonsense shape.
4. **`condition_number(A)`** returns `L / mu` from `eigenvalue_range`. Dividing the other way gives a number below 1.
5. **`contraction_ratios(path, w_star)`** takes a path of shape `(iters + 1, d)` and returns an array of shape `(iters,)` with each error over the one before. Take `np.linalg.norm(path - w_star, axis=1)`, one length per row, then `errors[1:] / errors[:-1]`.
6. **`steps_to_tolerance(path, w_star, tol)`** returns the first row index whose error is below `tol` times the starting error, or `-1` if no row gets there. Use `np.nonzero(errors < tol * errors[0])[0]`. Row 0 never counts.

## Where this breaks

**You do not know $L$ or $\kappa$.** You rarely compute eigenvalues of a large loss. You see the effect: a loss that rises means the step is too big, and a loss curve that falls fast and then crawls means a large $\kappa$.

**The loss is not a bowl.** The curvature changes from place to place, so a step that is safe in one place can blow up in another, and $\kappa$ is a local number.

## The exercise

You build `eigenvalue_range`, `axis_factors`, `exact_line_search_step`, `condition_number`, `contraction_ratios`, and `steps_to_tolerance`. The starter supplies `gradient_descent`. The checks compare the curvature functions with known values on a tilted matrix, and `axis_factors` with the hand values at steps 0.1, 0.18, and 0.3. With your own $L$, a step of $1.9/L$ must shrink the error and $2.1/L$ must grow it. The line-search step must equal the closed form and make the next gradient perpendicular. Last, the ratios must equal $1-\mu/L$ along the flat axis, and the step counts must be 66 and 688 for $\kappa=10$ and $\kappa=100$, and 1 for a circle.
