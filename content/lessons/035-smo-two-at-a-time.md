---
title: "Two at a time: the SMO solver"
slug: 035-smo-two-at-a-time
part: "Part 2 — Classical models"
summary: Raise the dual by moving two pushes together along a line, jumping to the top of a parabola and stopping at the wall of the box.
minutes: 45
gpu: false
objectives:
  - Explain why one push at a time breaks the balance rule and why a pair can slide along a line instead.
  - Find the ends L and H of the allowed range from the square [0, C] and the balance line.
  - Jump to the top of the one-variable parabola, clip it, recover the partner push, and update the bias.
  - Build the solver loop that picks houses by the KKT test and stops when the largest violation is small.
prereqs:
  - 033-dual-multipliers-kernels
  - 034-support-vectors-kkt
exercise: 035-smo-two-at-a-time
---

# Two at a time: the SMO solver

You know what the best pushes look like, and you can test any set of pushes against the checklist. One piece is still missing: how to find them. The dual objective $D$ depends on one push per house, so with thousands of houses you have thousands of numbers to choose.

Why not change one push at a time? Raise one push, find where $D$ is highest, move to the next push, and repeat. The trouble is the balance rule. In the two-house example, start at $\alpha = [0, 0]$ and raise only the first push to 0.5. Then $\sum_i \alpha_i y_i = 0.5$, not 0, so the plank slides and you have left the set of allowed pushes. Any move of a single push breaks the balance.

Why not hand all $n$ pushes to a general quadratic solver? It works for sixteen houses, but it optimises every push at once and factors an $n \times n$ matrix again and again, which is far too heavy for large $n$. The idea of this chapter sits between the two: change two pushes together, so that the second one can cancel what the first one disturbs.

## What you already know

These five ideas carry the chapter:

- The best pushes maximise $D(\alpha) = \sum_i \alpha_i - \tfrac12 v^{\top}Kv$ with $v = \alpha y$, as in "Pushes from each point: the dual and kernels".
- Every push stays in the box $0 \le \alpha_i \le C$, and the pushes balance, $\sum_i \alpha_i y_i = 0$.
- The score is $f_i = \sum_k \alpha_k y_k K_{ik} + b$.
- The checklist number is $g_i = y_i f_i - 1$, and the largest violation of it is the stopping measure from "Which points matter: support vectors and the KKT checklist".
- A step that sets a slope to zero and divides by a curvature is Newton's step, which lands exactly on the peak of a parabola.

## Why two pushes can move

Pick two houses, $i$ and $j$, and freeze every other push. The pair $(\alpha_j, \alpha_i)$ is a point in the plane, and the box rule keeps it inside the square from 0 to $C$. The balance rule says $y_i\alpha_i + y_j\alpha_j$ must keep its present value. That is the equation of a line, so the pair can only slide along a line, as Figure 1 shows.

```figure
<svg viewBox="0 0 600 342" role="img" aria-label="Two squares from zero to ten in both alpha values with a sloping line through the current pair. When the labels differ the line rises at 45 degrees through the corner and the allowed segment runs corner to corner. When the labels match the line falls and the allowed segment runs from the left edge of the top side to the right side">
<text x="152" y="18" class="fig-bold" text-anchor="middle">labels differ: α_i − α_j stays at 0</text>
<rect x="34" y="34" width="236" height="236" class="fill-paper stroke-grid"/>
<polygon points="61.2,242.8 242.8,242.8 242.8,61.2 61.2,61.2" class="fill-soft-a"/>
<line x1="61.2" y1="242.8" x2="61.2" y2="61.2" class="stroke-muted"/>
<line x1="61.2" y1="242.8" x2="242.8" y2="242.8" class="stroke-muted"/>
<line x1="242.8" y1="242.8" x2="242.8" y2="61.2" class="stroke-muted"/>
<line x1="61.2" y1="61.2" x2="242.8" y2="61.2" class="stroke-muted"/>
<line x1="34" y1="270" x2="270" y2="34" class="stroke-b dashed"/>
<line x1="61.2" y1="242.8" x2="242.8" y2="61.2" class="stroke-b" style="stroke-width:6"/>
<circle cx="61.2" cy="242.8" r="5.5" class="fill-ink"/>
<text x="61.2" y="286" class="fig-small" text-anchor="middle">0</text>
<text x="28" y="246.8" class="fig-small" text-anchor="end">0</text>
<text x="242.8" y="286" class="fig-small" text-anchor="middle">10</text>
<text x="28" y="65.2" class="fig-small" text-anchor="end">10</text>
<text x="152" y="322" class="fig-small" text-anchor="middle">α_j</text>
<text x="10" y="152" class="fig-small" text-anchor="middle">α_i</text>
<circle cx="61.2" cy="270" r="4" class="fill-b"/>
<circle cx="242.8" cy="270" r="4" class="fill-b"/>
<text x="63.2" y="300" class="fig-bold" text-anchor="start">L = 0</text>
<text x="242.8" y="300" class="fig-bold" text-anchor="middle">H = 10</text>
<text x="448" y="18" class="fig-bold" text-anchor="middle">labels match: α_i + α_j stays at 12</text>
<rect x="330" y="34" width="236" height="236" class="fill-paper stroke-grid"/>
<polygon points="357.2,242.8 538.8,242.8 538.8,61.2 357.2,61.2" class="fill-soft-a"/>
<line x1="357.2" y1="242.8" x2="357.2" y2="61.2" class="stroke-muted"/>
<line x1="357.2" y1="242.8" x2="538.8" y2="242.8" class="stroke-muted"/>
<line x1="538.8" y1="242.8" x2="538.8" y2="61.2" class="stroke-muted"/>
<line x1="357.2" y1="61.2" x2="538.8" y2="61.2" class="stroke-muted"/>
<line x1="366.3" y1="34" x2="566" y2="233.7" class="stroke-b dashed"/>
<line x1="393.5" y1="61.2" x2="538.8" y2="206.5" class="stroke-b" style="stroke-width:6"/>
<circle cx="448" cy="115.7" r="5.5" class="fill-ink"/>
<text x="357.2" y="286" class="fig-small" text-anchor="middle">0</text>
<text x="324" y="246.8" class="fig-small" text-anchor="end">0</text>
<text x="538.8" y="286" class="fig-small" text-anchor="middle">10</text>
<text x="324" y="65.2" class="fig-small" text-anchor="end">10</text>
<text x="448" y="322" class="fig-small" text-anchor="middle">α_j</text>
<text x="306" y="152" class="fig-small" text-anchor="middle">α_i</text>
<circle cx="393.5" cy="270" r="4" class="fill-b"/>
<circle cx="538.8" cy="270" r="4" class="fill-b"/>
<text x="393.5" y="300" class="fig-bold" text-anchor="middle">L = 2</text>
<text x="538.8" y="300" class="fig-bold" text-anchor="middle">H = 10</text>
</svg>
Figure 1. The pair (α_j, α_i) must stay on the orange line to keep the balance rule, and inside the shaded square to keep the box rule. The thick orange part is what remains. Its ends, read on the α_j axis, are L and H. The dot is the pair before the step.
```

If the labels differ, the line is $\alpha_i - \alpha_j = \text{constant}$, and it rises at 45 degrees. If the labels match, it is $\alpha_i + \alpha_j = \text{constant}$, and it falls. Only the part of the line inside the square is allowed, and its ends, read on the $\alpha_j$ axis, are $L$ and $H$. Solve for $\alpha_i$ along the line and require $0 \le \alpha_i \le C$, and you get:

$$
\begin{aligned}
y_i \ne y_j:&\quad L = \max(0,\, \alpha_j - \alpha_i), \quad H = \min(C,\, C + \alpha_j - \alpha_i), \\
y_i = y_j:&\quad L = \max(0,\, \alpha_i + \alpha_j - C), \quad H = \min(C,\, \alpha_i + \alpha_j).
\end{aligned}
$$

Read the first line aloud: if you add some amount to $\alpha_j$, $\alpha_i$ rises by the same amount, so $\alpha_j$ stops when either push reaches 0 or $C$. In the left panel, $C = 10$ and $\alpha = [0, 0]$, so $L = 0$ and $H = 10$. In the right panel the labels match and $\alpha_i + \alpha_j = 12$, so $L = 2$ and $H = 10$. The same numbers in numpy are two `max` and `min` calls on scalars.

## Jump to the top of a parabola

Along the allowed line, $D$ depends on one number, $\alpha_j$, because $\alpha_i$ follows it. The quadratic part of $D$ has a negative sign, so $D$ is an upside-down parabola in $\alpha_j$, and its best point is its peak. You can jump to the peak in one move.

Two quantities set the peak. The first is the error of the score, $E_i = f_i - y_i$. It is zero when house $i$ scores exactly its label, which means it sits on its edge. The slope of $D$ along the line is $y_j(E_i - E_j)$: how differently the current model mis-scores the two houses.

Why that slope? Raising one push $\alpha_k$ alone changes $D$ at the rate $1 - y_k(f_k - b) = -y_kE_k + y_kb$. Moving along the line raises $\alpha_j$ by $t$ and moves $\alpha_i$ by $-y_iy_jt$, so the rate along the line is $(-y_jE_j + y_jb) - y_iy_j(-y_iE_i + y_ib) = y_j(E_i - E_j)$. The bias cancels, and $y_i^2 = 1$ does the rest. The curvature has a similar source. Moving $t$ moves the weights by $y_jt\,(x_j - x_i)$ for the dot-product table, and $D$ loses half the squared length of that move, which is $\tfrac12\eta t^2$.

The curvature is $-\eta$, with

$$
\eta = K_{ii} + K_{jj} - 2K_{ij} .
$$

For the dot-product table, $\eta$ is the squared distance between the two houses. A pair far apart has a large $\eta$, a sharply curved parabola, and a small step. A pair that nearly coincides has $\eta$ near 0, a flat parabola, and a huge step, so you skip the pair when $\eta \le 10^{-12}$. The peak is the Newton step, and the clipped result and the partner follow:

$$
\alpha_j^{\text{new}} = \operatorname{clip}\!\left(\alpha_j + \frac{y_j(E_i - E_j)}{\eta},\; L,\; H\right), \qquad
\alpha_i^{\text{new}} = \alpha_i + y_i y_j\,(\alpha_j - \alpha_j^{\text{new}}) .
$$

Read it aloud: move to the top of the parabola, but stop at the wall if the peak is outside the box, then let the partner move the opposite way so the balance holds.

Work the two-house example, with $x = +1$ for house $i$ and $x = -1$ for house $j$, labels $+1$ and $-1$, $C = 10$, and $\alpha = [0, 0]$, $b = 0$. Every score is 0, so $E_i = 0 - 1 = -1$ and $E_j = 0 - (-1) = 1$. Then $\eta = 1 + 1 - 2(-1) = 4$, and the slope is $y_j(E_i - E_j) = (-1)(-2) = 2$. The step is $2/4 = 0.5$, so $\alpha_j^{\text{new}} = 0.5$, inside $[0, 10]$. The partner is $0 + (1)(-1)(0 - 0.5) = 0.5$. So $\alpha = [0.5, 0.5]$ and $D$ rose from 0 to 0.5 in one step, the road of the earlier chapters.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two copies of the same upside-down parabola of D against alpha j. On the left the peak at 0.5 is inside the allowed box and the step lands on it. On the right a wall at 0.3 cuts the box short, and the step stops at the wall">
<text x="163" y="18" class="fig-bold" text-anchor="middle">C = 10: the peak is inside the box</text>
<rect x="48" y="34" width="230" height="190" class="fill-paper stroke-grid"/>
<polyline points="59,209.4 61.1,203.6 63.3,197.9 65.5,192.4 67.7,186.9 69.9,181.6 72.1,176.4 74.3,171.3 76.5,166.4 78.7,161.5 80.9,156.8 83,152.2 85.2,147.6 87.4,143.3 89.6,139 91.8,134.8 94,130.8 96.2,126.9 98.4,123.1 100.6,119.4 102.8,115.8 105,112.4 107.1,109.1 109.3,105.8 111.5,102.8 113.7,99.8 115.9,96.9 118.1,94.2 120.3,91.5 122.5,89 124.7,86.6 126.9,84.3 129,82.2 131.2,80.1 133.4,78.2 135.6,76.4 137.8,74.7 140,73.1 142.2,71.6 144.4,70.3 146.6,69.1 148.8,68 151,67 153.1,66.1 155.3,65.3 157.5,64.7 159.7,64.2 161.9,63.8 164.1,63.5 166.3,63.3 168.5,63.2 170.7,63.3 172.9,63.5 175,63.8 177.2,64.2 179.4,64.7 181.6,65.3 183.8,66.1 186,67 188.2,68 190.4,69.1 192.6,70.3 194.8,71.6 197,73.1 199.1,74.7 201.3,76.4 203.5,78.2 205.7,80.1 207.9,82.2 210.1,84.3 212.3,86.6 214.5,89 216.7,91.5 218.9,94.2 221,96.9 223.2,99.8 225.4,102.8 227.6,105.8 229.8,109.1 232,112.4 234.2,115.8 236.4,119.4 238.6,123.1 240.8,126.9 243,130.8 245.1,134.8 247.3,139 249.5,143.3 251.7,147.6 253.9,152.2 256.1,156.8 258.3,161.5 260.5,166.4 262.7,171.3 264.9,176.4 267,181.6 269.2,186.9 271.4,192.4 273.6,197.9 275.8,203.6 278,209.4" class="stroke-a"/>
<line x1="168.5" y1="224" x2="168.5" y2="63.2" class="stroke-muted dashed" stroke-width="1"/>
<text x="168.5" y="240" class="fig-small" text-anchor="middle">peak, 0.5</text>
<circle cx="59" cy="209.4" r="5" class="fill-ink"/>
<text x="69" y="201.4" class="fig-small">start</text>
<circle cx="168.5" cy="63.2" r="6" class="fill-b"/>
<text x="168.5" y="49.2" class="fig-small" text-anchor="middle">after the step: D = 0.5</text>
<text x="163" y="280" class="fig-small" text-anchor="middle">the wall at 10 is far off to the right</text>
<text x="163" y="262" class="fig-small" text-anchor="middle">α_j</text>
<text x="453" y="18" class="fig-bold" text-anchor="middle">C = 0.3: the wall is in the way</text>
<rect x="338" y="34" width="230" height="190" class="fill-paper stroke-grid"/>
<polygon points="414.7,224 568,224 568,34 414.7,34" class="fill-soft-d"/>
<line x1="414.7" y1="224" x2="414.7" y2="34" class="stroke-bad"/>
<text x="420.7" y="45.7" class="fig-small">wall at 0.3</text>
<polyline points="349,209.4 351.1,203.6 353.3,197.9 355.5,192.4 357.7,186.9 359.9,181.6 362.1,176.4 364.3,171.3 366.5,166.4 368.7,161.5 370.9,156.8 373,152.2 375.2,147.6 377.4,143.3 379.6,139 381.8,134.8 384,130.8 386.2,126.9 388.4,123.1 390.6,119.4 392.8,115.8 395,112.4 397.1,109.1 399.3,105.8 401.5,102.8 403.7,99.8 405.9,96.9 408.1,94.2 410.3,91.5 412.5,89 414.7,86.6 416.9,84.3 419,82.2 421.2,80.1 423.4,78.2 425.6,76.4 427.8,74.7 430,73.1 432.2,71.6 434.4,70.3 436.6,69.1 438.8,68 441,67 443.1,66.1 445.3,65.3 447.5,64.7 449.7,64.2 451.9,63.8 454.1,63.5 456.3,63.3 458.5,63.2 460.7,63.3 462.9,63.5 465,63.8 467.2,64.2 469.4,64.7 471.6,65.3 473.8,66.1 476,67 478.2,68 480.4,69.1 482.6,70.3 484.8,71.6 487,73.1 489.1,74.7 491.3,76.4 493.5,78.2 495.7,80.1 497.9,82.2 500.1,84.3 502.3,86.6 504.5,89 506.7,91.5 508.9,94.2 511,96.9 513.2,99.8 515.4,102.8 517.6,105.8 519.8,109.1 522,112.4 524.2,115.8 526.4,119.4 528.6,123.1 530.8,126.9 533,130.8 535.1,134.8 537.3,139 539.5,143.3 541.7,147.6 543.9,152.2 546.1,156.8 548.3,161.5 550.5,166.4 552.7,171.3 554.9,176.4 557,181.6 559.2,186.9 561.4,192.4 563.6,197.9 565.8,203.6 568,209.4" class="stroke-a"/>
<line x1="458.5" y1="224" x2="458.5" y2="63.2" class="stroke-muted dashed" stroke-width="1"/>
<text x="458.5" y="240" class="fig-small" text-anchor="middle">peak, 0.5</text>
<circle cx="349" cy="209.4" r="5" class="fill-ink"/>
<text x="359" y="201.4" class="fig-small">start</text>
<circle cx="414.7" cy="86.6" r="6" class="fill-b"/>
<text x="404.7" y="76.6" class="fig-small" text-anchor="end">after the step: D = 0.42</text>
<text x="453" y="262" class="fig-small" text-anchor="middle">α_j</text>
</svg>
Figure 2. Along the allowed line, D is a parabola in α_j. The step jumps straight to its peak at 0.5 when the box allows it. When a wall at 0.3 stands in the way, the step stops at the wall, where D is 0.42 instead of 0.5. Clipping means stopping at the wall.
```

Shrink the box to $C = 0.3$. The peak is still at 0.5, which is past the wall, so $\alpha_j^{\text{new}} = \min(0.5, 0.3) = 0.3$, and both pushes end at $[0.3, 0.3]$. Figure 2 shows both cases, and $D$ is 0.42 at the wall.

> [!WARNING]
> Read both errors, $E_i$ and $E_j$, before you write either push. If you change $\alpha_i$ first and then compute $E_j$, the score $f_j$ already includes the updated $\alpha_i$, and the step is wrong.

## Move the bias too

The scores changed, so $b$ must change. Choose it so that a house strictly inside $(0, C)$ lands exactly on its edge, because the checklist demands $y_i f_i = 1$ for such a house. Write $\Delta_i$ and $\Delta_j$ for the changes in the two pushes. Putting house $i$ on its edge gives $b_1$, and putting house $j$ on its edge gives $b_2$:

$$
b_1 = b - E_i - y_i\Delta_i K_{ii} - y_j\Delta_j K_{ij}, \qquad
b_2 = b - E_j - y_i\Delta_i K_{ij} - y_j\Delta_j K_{jj} .
$$

Use $b_1$ if the updated $\alpha_i$ is strictly inside $(0, C)$, else $b_2$ if the updated $\alpha_j$ is, else the average of the two, because when both pushes sit at a wall, any bias between $b_1$ and $b_2$ keeps both houses within their rules. In the example, $\Delta_i = \Delta_j = 0.5$, so $b_1 = 0 - (-1) - 0.5 - 0.5 = 0$, and $b = 0$.

The step returns the inputs unchanged in three cases. If $i = j$ there is no pair. If $\eta \le 10^{-12}$ the parabola is flat and the formula divides by almost nothing. If $L \ge H$ the line touches the square at one point and there is no room to move.

## Choose the pairs

The solver loop needs a rule for picking $i$. Use the checklist. Since $y_i^2 = 1$, we get $y_i E_i = y_i f_i - 1 = g_i$, so $y_i E_i$ is the same number as the last chapter's $g_i$, computed for one house. House $i$ needs attention when $y_i E_i < -\text{tol}$ and $\alpha_i < C$, because it falls short of the edge and can push harder. It also needs attention when $y_i E_i > \text{tol}$ and $\alpha_i > 0$, because it lies beyond the edge and is still pushing. For such an $i$, scan $j$ from 0 and keep the first step that moves $\alpha_j$ by more than $10^{-7}$. Stop after a pass in which the largest violation is below `tol`, after a pass that changed nothing, or after `max_iter` passes.

Two tolerances are at work. `tol` in `fit_smo` decides when the whole solve is good enough, and the default tolerance inside `kkt_violation` only decides which end of the box a push sits at.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Two line charts over 72 update steps. The dual objective climbs from zero and never falls, flattening near 0.238. The largest KKT violation starts at 1, wanders down in a ragged way, and reaches zero at the last step">
<text x="168" y="18" class="fig-bold" text-anchor="middle">Dual objective D</text>
<rect x="56" y="34" width="224" height="190" class="fill-paper stroke-grid"/>
<polyline points="56,224 59.1,115.3 62.2,100.8 65.3,86.3 68.4,83.6 71.6,83.5 74.7,83.3 77.8,79.9 80.9,76.3 84,76.2 87.1,76 90.2,75.8 93.3,73.7 96.4,73.7 99.6,73.4 102.7,73.1 105.8,72.1 108.9,71.9 112,71.6 115.1,70.6 118.2,70.4 121.3,70.1 124.4,69.1 127.6,68.9 130.7,68.6 133.8,67.6 136.9,67.4 140,67.1 143.1,66.1 146.2,65.9 149.3,65.6 152.4,64.6 155.6,64.4 158.7,64.1 161.8,63.1 164.9,62.9 168,62.6 171.1,61.6 174.2,61.4 177.3,61.1 180.4,60.1 183.6,59.9 186.7,59.6 189.8,58.6 192.9,58.4 196,58.1 199.1,57.1 202.2,57.1 205.3,56.8 208.4,56.5 211.6,56.1 214.7,55.8 217.8,55.5 220.9,55.3 224,55 227.1,54.7 230.2,54.4 233.3,54.1 236.4,53.8 239.6,53.5 242.7,53.3 245.8,53 248.9,52.7 252,52.5 255.1,52.2 258.2,51.9 261.3,51.7 264.4,51.4 267.6,51.2 270.7,50.9 273.8,50.7 276.9,50.5 280,50.4" class="stroke-a"/>
<line x1="280" y1="224" x2="280" y2="34" class="stroke-muted dashed" stroke-width="1"/>
<circle cx="280" cy="50.4" r="5" class="fill-ink"/>
<text x="50" y="228" class="fig-small" text-anchor="end">0</text>
<text x="50" y="154.9" class="fig-small" text-anchor="end">0.1</text>
<text x="50" y="81.8" class="fig-small" text-anchor="end">0.2</text>
<text x="56" y="240" class="fig-small" text-anchor="middle">0</text>
<text x="118.2" y="240" class="fig-small" text-anchor="middle">20</text>
<text x="180.4" y="240" class="fig-small" text-anchor="middle">40</text>
<text x="242.7" y="240" class="fig-small" text-anchor="middle">60</text>
<text x="168" y="258" class="fig-small" text-anchor="middle">update steps</text>
<text x="274" y="66.4" class="fig-small" text-anchor="end">0.2376</text>
<text x="458" y="18" class="fig-bold" text-anchor="middle">Largest KKT violation</text>
<rect x="346" y="34" width="224" height="190" class="fill-paper stroke-grid"/>
<polyline points="346,43 349.1,167.2 352.2,154.3 355.3,107.4 358.4,175.5 361.6,173.4 364.7,171.3 367.8,194 370.9,162.8 374,197.9 377.1,196.6 380.2,195.3 383.3,192.9 386.4,196.5 389.6,209.2 392.7,194.2 395.8,196.5 398.9,209.4 402,194.2 405.1,196.5 408.2,209.5 411.3,194.2 414.4,196.5 417.6,209.7 420.7,194.2 423.8,196.5 426.9,209.6 430,194.2 433.1,196.5 436.2,209.4 439.3,194.2 442.4,196.5 445.6,209.3 448.7,194.3 451.8,196.5 454.9,209.2 458,194.3 461.1,196.5 464.2,209 467.3,194.3 470.4,196.5 473.6,208.9 476.7,194.3 479.8,196.5 482.9,208.8 486,194.3 489.1,202.6 492.2,208.7 495.3,208.1 498.4,208.9 501.6,208.3 504.7,209 507.8,208.4 510.9,209.2 514,208.6 517.1,209.3 520.2,208.7 523.3,209.5 526.4,208.9 529.6,209.6 532.7,209.1 535.8,209.8 538.9,209.2 542,209.9 545.1,209.4 548.2,210.1 551.3,209.5 554.4,210.2 557.6,209.7 560.7,210.4 563.8,209.8 566.9,215.5 570,224" class="stroke-b"/>
<line x1="570" y1="224" x2="570" y2="34" class="stroke-muted dashed" stroke-width="1"/>
<circle cx="570" cy="224" r="5" class="fill-ink"/>
<text x="340" y="228" class="fig-small" text-anchor="end">0</text>
<text x="340" y="137.5" class="fig-small" text-anchor="end">0.5</text>
<text x="340" y="47" class="fig-small" text-anchor="end">1</text>
<text x="346" y="240" class="fig-small" text-anchor="middle">0</text>
<text x="408.2" y="240" class="fig-small" text-anchor="middle">20</text>
<text x="470.4" y="240" class="fig-small" text-anchor="middle">40</text>
<text x="532.7" y="240" class="fig-small" text-anchor="middle">60</text>
<text x="458" y="258" class="fig-small" text-anchor="middle">update steps</text>
<text x="562" y="169.7" class="fig-small" text-anchor="end">stop: violation 0</text>
</svg>
Figure 3. One run of the solver on sixteen houses in two clusters, C = 1. It takes 26 passes and 72 update steps. The objective D rises at every step and never falls. The largest violation does not fall at every step, and it jumps up when a step fixes one house and disturbs another, but it ends at 0.
```

Figure 3 shows a run on sixteen houses with $C = 1$. It takes 26 passes and 72 steps, and $D$ rises at every step, as the theory promises, because each step goes to the best point of its own parabola. The largest violation does not fall at every step, because fixing one house can disturb another, but it ends at 0. The slow tail comes from the scan order, which often picks poor pairs. A solver such as libsvm picks the pair that violates the checklist most.

> [!KEY]
> You find the best pushes by changing two at a time. The balance rule leaves one free direction, along a line. On that line, $D$ is a parabola, so you jump to its peak, stop at the wall of the box if you overshoot, and repeat on any house that breaks the checklist.

## Cost and scaling

The table $K$ has $n^2$ entries, which is the main memory cost. One step reads two scores, each a sum over $n$ houses, so a step costs about $n$. A pass tries up to $n$ partners for each of $n$ houses, which is up to $n^3$ operations in the worst case. Usually the first partner works, and a pass costs about $n^2$. A general quadratic solver must factor an $n \times n$ matrix, which costs about $n^3$ at every iteration.

## Where this breaks

**Duplicate houses.** Two identical houses give $\eta = 0$, so the step skips them. The solver then needs another partner to make progress.

**A table that is not a valid kernel.** Then $\eta$ can be negative, the parabola opens upward, and skipping the pair, as the $\eta \le 10^{-12}$ rule does, is the safe choice. The chapter "What makes a similarity a valid kernel" says when this happens.

> [!RECAP]
> - Changing one push breaks the balance rule, and changing a pair slides along the line $y_i\alpha_i + y_j\alpha_j = $ constant, inside the square $[0, C]^2$.
> - $L$ and $H$ are the ends of that line inside the square, read on the $\alpha_j$ axis.
> - $D$ is a parabola along the line, with slope $y_j(E_i - E_j)$ and curvature $-\eta$, so the step is slope over $\eta$, clipped to $[L, H]$.
> - The bias is set so a strictly inside house rests on its edge, and $y_i E_i$ is the checklist number $g_i$.

## Build it step by step

You write three functions, in the order the checks test them. The four helpers `dual_scores`, `dual_objective`, `margin_groups`, and `kkt_violation` are already written, from the earlier chapters.

1. **`clip_bounds(alpha, y, C, i, j)`**. It returns `(L, H)` as two floats. Branch on whether `y[i] != y[j]`, and use the two lines of formulas above. The common mistake is to swap $i$ and $j$, which gives the bounds of $\alpha_i$ in place of $\alpha_j$.

2. **`smo_step(alpha, y, K, b, C, i, j)`**. It returns `(new_alpha, new_b)`. Copy `alpha` first, then return early for the three unchanged cases. The shape of the code is:

```python
alpha = alpha.copy()
f = dual_scores(alpha, y, K, b)          # read both scores first
E_i, E_j = f[i] - y[i], f[j] - y[j]
eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
```

   The common mistake is computing `E_j` after writing `alpha[i]`.

3. **`fit_smo(K, y, C, max_iter, tol)`**. It returns `(alpha, b)`. Start from zeros and write the pass loop. Use `y[i] * (f[i] - y[i])` for the test on house $i$. The common mistake is forgetting the stop on a pass that changes nothing, which makes the loop run to `max_iter`.

## The exercise

You build `clip_bounds`, `smo_step`, and `fit_smo`. The checks confirm $L$ and $H$ for the chapter's two cases and for swapped $i$ and $j$. They run the two-house step and expect $\alpha = [0.5, 0.5]$, $b = 0$, and $D = 0.5$, then the wall case with $C = 0.3$. They test the three unchanged returns. Over many random pairs, $\alpha$ must stay in $[0, C]$, the pushes must stay balanced, and $D$ must never fall. A strictly inside house must rest on its edge after a step. Finally `fit_smo` must drive the violation below its tolerance and find the same road as an independent solver.
