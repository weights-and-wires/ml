---
title: "How big a step: curvature and the 2/L limit"
slug: 010-how-big-a-step
part: "Part 1 — Foundations"
summary: Each step multiplies your distance from the bottom by a factor per axis, so the sharpest axis sets the longest safe step.
minutes: 40
gpu: false
objectives:
  - Explain why a step multiplies the distance to the bottom by 1 minus the step size times the curvature, on each axis.
  - Find the longest safe step, 2/L, from the largest curvature, and tell creeping, overshooting, and blowing up apart.
  - Compute the best step along a downhill line in closed form.
  - Predict from the curvatures whether a step size converges.
prereqs:
  - 009-walking-downhill
exercise: 010-how-big-a-step
---

# How big a step: curvature and the 2/L limit

You walk the bowl from the last chapter, and each step costs you the same effort whether it is short or long. So why not take a huge step and get there faster? You can try it: replace the step 0.1 with 0.3 and walk again.

The loss does not fall. It goes from 0 to 14.7 to 74.6 to 314.6. Each step lands you higher on the opposite wall than where you started, and the walk flies apart. A step that is too small wastes time, and a step that is too big wrecks the walk.

This chapter answers a practical question: how do you know a safe step size without trying every number? The answer comes from looking at one axis at a time.

## What you already know

Five ideas carry this chapter:

- Gradient descent moves from $w$ to $w-\alpha\nabla f(w)$ (see "Walking downhill: slope, gradient, and one step").
- The running bowl has height $\tfrac12(w_1-1)^2+5(w_2-1)^2-5.5$, with its bottom at $(1,1)$.
- The *miss* of a coordinate is its distance from the bottom value. The $w_2$ axis is steep and the $w_1$ axis is flat.
- With step 0.1 the steep coordinate landed on its bottom value in one step, and the flat miss went $1,\,0.9,\,0.81$.
- `np.linalg.eigvalsh(A)` returns the eigenvalues of a symmetric matrix, in ascending order. You meet what they mean below.

## One axis at a time

Look at a single coordinate of the walk. On the steep axis the height is $5(w_2-1)^2$, whose slope is $10(w_2-1)$, which is 10 times the miss. A step moves you by $\alpha$ times that slope, backwards. So the next miss is the old miss minus $10\alpha$ times the old miss:

$$
\text{miss}_{\text{next}} = (1-10\,\alpha)\;\text{miss}
$$

Read it aloud as "each step multiplies the miss by one minus ten times the step size." On the flat axis the same argument gives $1-1\cdot\alpha$. The matrix $A$ is diagonal, so the axes do not mix: each coordinate follows its own multiplication.

Try $\alpha=0.3$. The steep factor is $1-3=-2$. Count the miss as positive when you are short of the bottom, so $w_2=0$ starts at a miss of 1. The misses go $1,\,-2,\,4,\,-8$. The sign flips and the size doubles, which is exactly the blow-up you saw. The flat factor is $1-0.3=0.7$, so the flat misses shrink: $1,\,0.7,\,0.49,\,0.343$. One axis converges and the other explodes, and the explosion wins.

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

The number that matters on each axis is its *curvature*, how sharply the bowl bends along it. It is how fast the slope itself changes: the slope $10(w_2-1)$ grows by 10 for every unit you move, and the flat slope $1(w_1-1)$ grows by 1. So the curvatures here are 10 and 1. For a general symmetric $A$, the curvatures along the bowl's own axes are its *eigenvalues*, written $\lambda$ (lambda). The largest is $L$ and the smallest is $\mu$ (mu, said "mew"). Here $L=10$ and $\mu=1$. If $A$ is not diagonal, the bowl's own axes are tilted, and the same argument holds along each tilted axis.

The factor for one axis is $1-\alpha\lambda$. Figure 2 plots its size against the step size for both axes. Read the steep axis, in orange, from left to right:

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

The flat axis never gets in trouble first. On this bowl its factor $1-\alpha\mu$ stays between 0.8 and 1 for every step below $0.2$. The steepest axis is the first to blow up, and that is why the largest curvature sets the limit.

In symbols, plain gradient descent on a bowl converges from every start when

$$
0 < \alpha < \frac{2}{L}
$$

Read it aloud as "the step must be positive and below two over the largest curvature." The common safe choice is $1/L$, because it stays well inside the limit and zeroes the steepest axis. In numpy, the factors for all axes at once are:

```python
factors = np.abs(1 - step * eigenvalues)  # one per axis
```

If any entry is above 1, that axis grows. The limit $2/L$ is for plain gradient descent. Later, momentum changes it.

> [!KEY]
> On each axis a step multiplies your distance from the bottom by $1-\alpha\lambda$. The axis with the largest curvature $L$ blows up first, so the step must stay below $2/L$, and $1/L$ is the usual safe choice.

## Find the curvature

The rule is only useful if you can find $L$. For a bowl with matrix $A$, you read it from the eigenvalues: the smallest and largest of `np.linalg.eigvalsh(A)`. The reading works even when $A$ is not diagonal, because a tilted bowl is the same bowl seen from a rotated angle, and the eigenvalues do not change when you rotate. For a loss without a matrix you cannot read $L$ off, so people start with a guess, watch the loss, and halve the step whenever the loss rises. A rising loss is the signal that some axis has a factor above 1, so halving is the $2/L$ rule applied by hand.

## Pick the step by looking down the line

Instead of choosing a number, you can ask the bowl. Stand at $(0,0)$, where the gradient is $g=(-1,-10)$, and walk straight along the downhill line. After a step $\alpha$ you are at $\alpha\,(1,10)$. The height there is $\tfrac12\cdot 1001\,\alpha^2-101\,\alpha$. At $\alpha=0.1$ that is $5.005-10.1=-5.095$, the height you reached in the last chapter. The height along the line is a parabola in $\alpha$, so its lowest point is the best step.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Height along the downhill line as a function of step size">
<defs></defs>
<line x1="70.0" y1="190.0" x2="560.0" y2="190.0" class="stroke-muted dashed" />
<line x1="70.0" y1="270.0" x2="560.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="270.0" class="stroke-grid" />
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted" />
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="233.3" y1="270.0" x2="233.3" y2="275.0" class="stroke-muted" />
<text x="233.3" y="290.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="396.7" y1="270.0" x2="396.7" y2="275.0" class="stroke-muted" />
<text x="396.7" y="290.0" class="fig-small" text-anchor="middle">0.2</text>
<line x1="560.0" y1="270.0" x2="560.0" y2="275.0" class="stroke-muted" />
<text x="560.0" y="290.0" class="fig-small" text-anchor="middle">0.3</text>
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">-5</text>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">0</text>
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">5</text>
<text x="62.0" y="94.0" class="fig-small" text-anchor="end">10</text>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">15</text>
<polyline points="70.0,190.0 74.1,192.5 78.2,194.9 82.2,197.3 86.3,199.6 90.4,201.8 94.5,204.0 98.6,206.1 102.7,208.2 106.8,210.2 110.8,212.1 114.9,214.0 119.0,215.8 123.1,217.5 127.2,219.2 131.2,220.8 135.3,222.4 139.4,223.9 143.5,225.3 147.6,226.7 151.7,228.0 155.8,229.2 159.8,230.4 163.9,231.5 168.0,232.6 172.1,233.6 176.2,234.5 180.2,235.4 184.3,236.2 188.4,236.9 192.5,237.6 196.6,238.2 200.7,238.8 204.8,239.3 208.8,239.7 212.9,240.1 217.0,240.4 221.1,240.6 225.2,240.8 229.2,240.9 233.3,241.0 237.4,240.9 241.5,240.9 245.6,240.7 249.7,240.5 253.8,240.3 257.8,240.0 261.9,239.6 266.0,239.1 270.1,238.6 274.2,238.0 278.2,237.4 282.3,236.7 286.4,236.0 290.5,235.1 294.6,234.2 298.7,233.3 302.8,232.3 306.8,231.2 310.9,230.1 315.0,228.9 319.1,227.6 323.2,226.3 327.2,224.9 331.3,223.5 335.4,222.0 339.5,220.4 343.6,218.8 347.7,217.1 351.8,215.3 355.8,213.5 359.9,211.6 364.0,209.6 368.1,207.6 372.2,205.6 376.2,203.4 380.3,201.2 384.4,199.0 388.5,196.6 392.6,194.2 396.7,191.8 400.8,189.3 404.8,186.7 408.9,184.1 413.0,181.4 417.1,178.6 421.2,175.8 425.2,172.9 429.3,170.0 433.4,166.9 437.5,163.9 441.6,160.7 445.7,157.5 449.8,154.3 453.8,150.9 457.9,147.6 462.0,144.1 466.1,140.6 470.2,137.0 474.3,133.4 478.3,129.7 482.4,125.9 486.5,122.1 490.6,118.2 494.7,114.3 498.8,110.2 502.8,106.2 506.9,102.0 511.0,97.8 515.1,93.6 519.2,89.2 523.2,84.9 527.3,80.4 531.4,75.9 535.5,71.3 539.6,66.7 543.7,62.0 547.8,57.2 551.8,52.4 555.9,47.5 560.0,42.5" fill="none" class="stroke-a" />
<circle cx="234.8" cy="241.0" r="5" class="fill-b"/>
<text x="246.8" y="257.0" class="fig-small" text-anchor="start">best step 0.1009, height -5.095</text>
<circle cx="364.0" cy="209.6" r="4.5" class="fill-d"/>
<text x="374.0" y="213.6" class="fig-small" text-anchor="start">0.18</text>
<circle cx="560.0" cy="42.5" r="4.5" class="fill-bad"/>
<text x="550.0" y="40.5" class="fig-small" text-anchor="end">0.3</text>
<circle cx="70.0" cy="190.0" r="4.5" class="fill-ink"/>
<text x="80.0" y="180.0" class="fig-small" text-anchor="start">start, height 0</text>
<circle cx="399.6" cy="190.0" r="4.5" class="fill-ink"/>
<text x="389.6" y="180.0" class="fig-small" text-anchor="end">back to height 0 at 0.2018</text>
<text x="315.0" y="312.0" class="fig-small" text-anchor="middle">step size along the downhill line</text>
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">height after the step</text>
</svg>
Figure 3. Walk from (0, 0) along the downhill direction and record the height after each possible step size. The result is a parabola. Its lowest point is the best step. The height comes back to 0, the start height, at twice that step, so any step longer than about 0.2 leaves you higher than you began.
```

For any bowl, expanding the height gives $f(w)-\alpha\,g^\top g+\tfrac12\alpha^2\,g^\top A g$, whose lowest point is at

$$
\alpha = \frac{g^\top g}{g^\top A\,g}
$$

Read it aloud as "the slope along the line, divided by the curvature along the line." That is the bottom of any one-weight parabola: slope over curvature. Along the downhill line the slope at the start is $g^\top g$ (the gradient dotted with itself) and the curvature is $g^\top A g$. Here $g^\top g=1+100=101$ and $g^\top A g=1+1000=1001$, so $\alpha=101/1001\approx0.1009$. This step is *exact line search*. It does not need $L$. In numpy:

```python
alpha = (g @ g) / (g @ A @ g)
```

Flip the fraction and the step is about 9.9, far above $2/L$, so the walk explodes.

At the best step the slope along the line is 0, which means the next gradient is perpendicular to the old one. Check it: the point after the step is $(0.1009,\,1.0090)$, the gradient there is $(-0.8991,\,0.0899)$, and its dot product with $(-1,-10)$ is 0. That perpendicular turn is a warning. On a long valley, each step turns at a right angle to the last, so the walk zigzags, and exact line search does not stop it. The next chapter, "Condition number: why long valleys are slow," measures how slow such a walk is.

> [!KEY]
> A step that is too big wrecks the walk because the sharpest axis multiplies your miss by more than 1 each step. You can stay below $2/L$, use $1/L$ when you are unsure, or let the bowl pick the best step down the line. A step that is too small is safe but wastes steps on the flat axis.

> [!RECAP]
> - On each axis, one step multiplies the miss by $1-\alpha\lambda$.
> - The sharpest axis, with curvature $L$, blows up first, so plain descent needs $\alpha<2/L$. The step $1/L$ is safe.
> - Between $1/L$ and $2/L$ you overshoot but still converge.
> - Exact line search uses $\alpha=(g^\top g)/(g^\top A g)$, and the next gradient is perpendicular to the old.
> - Eigenvalues of `A` give the curvatures.

## Build it step by step

The starter already includes `gradient_descent`, which you wrote in the previous chapter. The checks call the functions you write in this order:

1. **`eigenvalue_range(A)`** takes a symmetric matrix of shape `(d, d)` and returns `(mu, L)` as two floats:

   ```python
   eigs = np.linalg.eigvalsh(A)  # ascending
   # return float(eigs[0]), float(eigs[-1])
   ```

   Use `eigvalsh`, not `eigvals`, because it is built for symmetric matrices and returns real numbers.

2. **`axis_factors(A, step)`** returns an array with one entry per eigenvalue, in ascending order of eigenvalue:

   ```python
   # np.abs(1 - step * eigenvalues)
   ```

   For the running bowl at step 0.3 you get `[0.7, 2.0]`.

3. **`exact_line_search_step(A, grad_vec)`** takes `A` of shape `(d, d)` and a gradient of shape `(d,)`, and returns one float:

   ```python
   # (grad_vec @ grad_vec) / (grad_vec @ A @ grad_vec)
   ```

   Putting `A` on the wrong side, as `grad_vec @ grad_vec @ A`, gives a nonsense shape.

## Where this breaks

**You do not know $L$.** Estimate it from the eigenvalues when you have a matrix, and halve the step when the loss rises when you do not.

**One step size for all axes.** A step that suits the steep axis is slow on the flat axis. The next chapter measures how slow.

**The loss is not a bowl.** The curvature changes from place to place, so a step that is safe in one place can blow up in another.

## The exercise

You build `eigenvalue_range`, `axis_factors`, and `exact_line_search_step`. The starter supplies `gradient_descent`.

The checks compare `eigenvalue_range` with known values on a tilted matrix, and `axis_factors` with the hand values at steps 0.1, 0.18, and 0.3. They also start a walk exactly on one tilted axis and verify that the miss shrinks by the factor you reported. With your own $L$, a step of $1.9/L$ must shrink the error and $2.1/L$ must grow it. Last, the line-search step must equal the closed form, minimise the height along its line, and make the next gradient perpendicular to the old.
