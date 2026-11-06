---
title: "The dual problem and the support vectors"
slug: 031-the-dual-and-support-vectors
part: "Part 2 — Classical models"
summary: Describe the widest road by how hard each house pushes on it, see that only a few houses push, and test any set of pushes with the KKT checklist.
minutes: 60
gpu: false
objectives:
  - Explain a multiplier alpha as the push one house puts on the road, rebuild the weights as w = Σ α·y·x, and score points from the pushes alone.
  - Compute the dual objective D(α) from a Gram matrix, and show that swapping the Gram matrix changes the shape of the boundary.
  - Sort every house into one of three groups by the size of its push, and define a support vector.
  - Compute the violation of the KKT checklist, and say what a violation of zero does and does not tell you.
prereqs:
  - 030-widest-road-hinge-loss
exercise: 031-the-dual-and-support-vectors
---

# The dual problem and the support vectors

The last chapter described the widest road with a list of weights $w$, one number per feature. This chapter describes the same road a different way. Instead of asking "what is $w$?", you ask "how much does each house hold the road in place?"

Why bother? Picture a village centre with a ring of houses around it. No straight road separates them. The pushes description asks only how alike two houses are, so you can change that one measure and the same solver draws a curve. And when a colleague's solver hands you a list of pushes, you can test them without retraining, and learn which houses decide where the road goes.

You build the pushes, sort the houses, write the test, and then swap the measure of alikeness on the ring.

## What you already know

- The widest road has its edges at $z = \pm 1$, where $z = y(w^{\top}x + b)$, and its half-width is $1/\lVert w \rVert$.
- A house with $z > 1$ is outside the road and costs nothing, and a house with $z < 1$ pays slack, $\max(0, 1 - z)$.
- $C$ is the price of one unit of slack.
- The dot product $x_i^{\top}x_j$ is large and positive when two points point the same way, and negative when they point opposite ways.
- At the bottom of a bowl the slope is zero.

## The road as a plank between pegs

Picture the road as a stiff plank clamped between pegs, and each house as a peg. A peg pushes on the plank with some force $\alpha_i \ge 0$, called its *multiplier*, in the direction away from the peg's own side. A peg that touches nothing pushes with force 0.

When the plank is at rest it does not slide, so the pushes from the two sides cancel, and the direction of the road is built from the pushes themselves:

$$
\sum_i \alpha_i y_i = 0, \qquad w = \sum_i \alpha_i y_i x_i .
$$

The first equation is the *balance rule*. Read the second aloud: $w$ is every house's position, weighted by how hard it pushes and signed by its side.

Take the smallest case: a square at $x = +1$ with $y = +1$ and a circle at $x = -1$ with $y = -1$, in that order, which is also the order of the exercise. The road between them has edges at $\pm 1$, centre at 0, and $w = 1$. Equal pushes $\alpha = [0.5, 0.5]$ rebuild it, because each house adds $\alpha\cdot y\cdot x = \alpha$ to $w$, so two equal pushes must each be 0.5 to reach $w = 1$.

```figure
<svg viewBox="0 0 600 258" role="img" aria-label="Two houses at x equals minus one and plus one, each pushing the road with strength 0.5, and the same two pushes added head to tail to make w equal to 1">
<defs><marker id="c033-arrow-d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker><marker id="c033-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="12" y="22" class="fig-bold">Pushes on the road</text>
<rect x="190" y="46" width="220" height="56" class="fill-soft-b"/>
<line x1="190" y1="46" x2="190" y2="102" class="stroke-b dashed"/>
<line x1="410" y1="46" x2="410" y2="102" class="stroke-b dashed"/>
<line x1="300" y1="46" x2="300" y2="102" class="stroke-b"/>
<line x1="80" y1="74" x2="520" y2="74" class="stroke-muted" stroke-width="1"/>
<circle cx="190" cy="74" r="7" class="fill-a"/>
<rect x="403" y="67" width="14" height="14" class="fill-c"/>
<line x1="200" y1="74" x2="245" y2="74" class="stroke-d" marker-end="url(#c033-arrow-d)"/>
<line x1="400" y1="74" x2="355" y2="74" class="stroke-d" marker-end="url(#c033-arrow-d)"/>
<text x="190" y="124" text-anchor="middle" class="fig-small">circle, x = −1</text>
<text x="410" y="124" text-anchor="middle" class="fig-small">square, x = +1</text>
<text x="190" y="140" text-anchor="middle" class="fig-small">pushes east, α = 0.5</text>
<text x="410" y="140" text-anchor="middle" class="fig-small">pushes west, α = 0.5</text>
<text x="300" y="40" text-anchor="middle" class="fig-small">road centre</text>
<text x="12" y="176" class="fig-bold">Adding the pushes up gives w</text>
<line x1="278" y1="224" x2="454" y2="224" class="stroke-muted" stroke-width="1"/>
<line x1="300" y1="219" x2="300" y2="229" class="stroke-muted" stroke-width="1"/>
<text x="300" y="244" text-anchor="middle" class="fig-small">0</text>
<line x1="355" y1="219" x2="355" y2="229" class="stroke-muted" stroke-width="1"/>
<text x="355" y="244" text-anchor="middle" class="fig-small">0.5</text>
<line x1="410" y1="219" x2="410" y2="229" class="stroke-muted" stroke-width="1"/>
<text x="410" y="244" text-anchor="middle" class="fig-small">1</text>
<line x1="300" y1="196" x2="354" y2="196" class="stroke-d" marker-end="url(#c033-arrow-d)"/>
<line x1="355" y1="196" x2="409" y2="196" class="stroke-d" marker-end="url(#c033-arrow-d)"/>
<text x="290" y="200" text-anchor="end" class="fig-small">circle adds +0.5</text>
<text x="422" y="200" class="fig-small">square adds +0.5</text>
<line x1="300" y1="214" x2="409" y2="214" class="stroke-b" marker-end="url(#c033-arrow-b)"/>
<text x="422" y="222" class="fig-bold">w = 1</text>
</svg>
Figure 1. Top: each house pushes the road away from itself, and the two pushes are equal and opposite, so the road does not drift. Bottom: the weight vector is those same pushes added head to tail, α·y·x for each house. Two pushes of 0.5 make w = 1, the road from the last chapter.
```

Where do the pushes come from? Give each demand "keep $z_i \ge 1$" a price $\alpha_i \ge 0$, and let a referee raise the cost $\tfrac12\lVert w \rVert^2 - \sum_i \alpha_i\,(z_i - 1)$ as high as possible. A road that breaks a demand loses, because the referee raises that $\alpha_i$, and a house with room to spare keeps $\alpha_i = 0$. Setting the slopes in $w$ and $b$ to zero gives exactly the two equations above. A multiplier is the price of its house's demand.

## Score the pushes with one number

Which pushes are the best ones? Put $w = \sum_i \alpha_i y_i x_i$ back into the cost. The $w$ disappears, and what remains depends only on $\alpha$:

$$
D(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i}\sum_{j}\alpha_i\alpha_j y_i y_j\, x_i^{\top}x_j .
$$

Read it aloud: the first term rewards total push, and the second charges half the squared length of the $w$ those pushes build. The best pushes maximise $D$ under the balance rule and a *box rule*, $0 \le \alpha_i \le C$: no house pushes harder than the price $C$ of its slack.

On the two-house example, take equal pushes $\alpha = [a, a]$. Then $D = 2a - 2a^2$, which peaks at $a=0.5$. The peak value $0.5$ equals $\tfrac12\lVert w \rVert^2$, the road's own objective. The best pushes reach the same value as the best weights.

Write $K_{ij} = x_i^{\top}x_j$. The table $K$ is the *Gram matrix*: entry $(i, j)$ says how alike houses $i$ and $j$ are. The double sum becomes one line of numpy, with $v = \alpha y$:

```python
v = alpha * y
D = alpha.sum() - 0.5 * v @ K @ v
```

On the two-house example, $K = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$ and $v = [0.5, -0.5]$, so $v^{\top}Kv = 1$ and $D = 1 - 0.5 = 0.5$. If you drop the labels and use $\alpha$ where $v$ belongs, $D$ comes out as 1.

Scoring a house needs $K$ too. Since $w^{\top}x_i = \sum_k \alpha_k y_k\, x_k^{\top}x_i$, the score of house $i$ is

$$
f_i = \sum_k \alpha_k y_k K_{ik} + b .
$$

Read it aloud: each house votes with its push and its side, weighted by how alike it is to house $i$. In numpy it is `(alpha * y) @ K + b`. On the example with $b = 0$, the scores are $[1, -1]$, so $y_i f_i = 1$ for both: each house rests on its edge. A house on its edge fixes $b$ by requiring $y_i f_i = 1$.

## Three kinds of houses

Figure 2 shows nine houses fitted with $C = 0.3$. Only four push at all. Five sit far from the road with a push of exactly zero. Two sit on the dashed edges and push gently. Two are capped at the largest push $C$ allows, and one of those is across the centre line.

```figure
<svg viewBox="0 0 600 346" role="img" aria-label="Nine houses with a road. Five faint houses far from the road push with zero force, two houses on the edges push a little, and two capped houses inside the road or across it push as hard as allowed">
<defs><marker id="c033-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<rect x="20" y="12" width="560" height="300" class="fill-paper stroke-grid"/>
<polygon points="198.4,312 471.9,312 392,12 118.5,12" class="fill-soft-b"/>
<line x1="335.2" y1="312" x2="255.2" y2="12" class="stroke-b"/>
<line x1="471.9" y1="312" x2="392" y2="12" class="stroke-b dashed"/>
<line x1="198.4" y1="312" x2="118.5" y2="12" class="stroke-b dashed"/>
<circle cx="85.9" cy="80.8" r="5" class="fill-muted"/>
<circle cx="118.8" cy="205.8" r="5" class="fill-muted"/>
<circle cx="151.8" cy="137" r="5" class="fill-a"/>
<circle cx="151.8" cy="137" r="9.5" class="stroke-ink"/>
<line x1="164.4" y1="133.8" x2="176.6" y2="130.7" class="stroke-d" marker-end="url(#c033-arrow)"/>
<circle cx="178.1" cy="249.5" r="5" class="fill-muted"/>
<rect x="410.3" y="94.5" width="10" height="10" class="fill-c"/>
<circle cx="415.3" cy="99.5" r="9.5" class="stroke-ink"/>
<line x1="402.7" y1="102.7" x2="390.5" y2="105.8" class="stroke-d" marker-end="url(#c033-arrow)"/>
<rect x="443.2" y="207" width="10" height="10" class="fill-muted"/>
<rect x="397.1" y="150.8" width="10" height="10" class="fill-c"/>
<circle cx="402.1" cy="155.8" r="9.5" class="stroke-bad"/>
<line x1="389.5" y1="158.9" x2="356.6" y2="167.3" class="stroke-d" marker-end="url(#c033-arrow)"/>
<rect x="476.2" y="257" width="10" height="10" class="fill-muted"/>
<circle cx="342.8" cy="174.5" r="5" class="fill-a"/>
<circle cx="342.8" cy="174.5" r="9.5" class="stroke-bad"/>
<line x1="355.4" y1="171.3" x2="388.4" y2="163" class="stroke-d" marker-end="url(#c033-arrow)"/>
<text x="30" y="336" class="fig-small">faint: α = 0, no push</text>
<text x="190" y="336" class="fig-small">plain ring: 0 &lt; α &lt; C, on an edge</text>
<text x="410" y="336" class="fig-small">red ring: α = C, capped</text>
</svg>
Figure 2. The same nine houses as before, now fitted with C = 0.3. Purple arrows show the pushes, and a longer arrow means a bigger α. Only four houses push at all. The two houses on the dashed edges push gently. The two capped houses push at the maximum, and one of them is across the centre line.
```

Why does each house land in its group? Take the plank picture.

- A house outside the road, with $z > 1$, has room to spare. Nothing binds, so its push is 0.
- A house that pushes by some amount below the cap must be touching the plank, so it rests exactly on its edge, $z = 1$.
- A house short of the edge, with $z < 1$, pays $C$ per unit of shortfall, the most it can cost, so its push is the largest allowed, $\alpha_i = C$. A house across the centre line is in this group too.

So three facts hold together: $\alpha = 0$ means $z \ge 1$, $0 < \alpha < C$ means $z = 1$, and $\alpha = C$ means $z \le 1$. A *support vector* is a house with $\alpha_i > 0$, so it is an edge house or a capped house. The road is built only from $w = \sum_i \alpha_i y_i x_i$, and a house with $\alpha_i = 0$ adds nothing to that sum. Delete the five zero-push houses and refit, and you get the same $w$. The support vectors decide the road, and the rest could be erased.

## The KKT checklist

To test a set of pushes, measure how far each house is past its edge:

$$
g_i = y_i f_i - 1 = z_i - 1 .
$$

Read it aloud: $g_i$ is zero on the edge, positive beyond it, and negative inside the road. Each group has a rule on $g$, and a house breaks its rule by the distance to its allowed band, as in Figure 3:

| Group | Meaning | Rule | Violation |
|---|---|---|---|
| $\alpha_i = 0$ | far, no push | $g_i \ge 0$ | $\max(0, -g_i)$ |
| $0 < \alpha_i < C$ | on an edge, free | $g_i = 0$ | $\lvert g_i \rvert$ |
| $\alpha_i = C$ | capped, at or short of the edge | $g_i \le 0$ | $\max(0, g_i)$ |

```figure
<svg viewBox="0 0 600 314" role="img" aria-label="Number lines for the value g. Each group of houses has an allowed band: g at least zero for no push, g exactly zero for a free push, g at most zero for a capped push. At the best pushes every house sits inside its band. With all pushes zero, every house sits at g equal to minus one, one unit outside its band">
<text x="12" y="20" class="fig-bold">The best pushes</text>
<text x="12" y="56" class="fig-small">α = 0</text>
<text x="12" y="70" class="fig-small">needs g ≥ 0</text>
<line x1="81.50000000000003" y1="56" x2="518.5" y2="56" class="stroke-grid"/>
<rect x="300" y="42" width="209" height="28" class="fill-soft-c"/>
<circle cx="426.7" cy="56" r="5" class="fill-ink"/>
<text x="426.7" y="38" text-anchor="middle" class="fig-small">x = −3</text>
<circle cx="363.3" cy="56" r="5" class="fill-ink"/>
<text x="363.3" y="38" text-anchor="middle" class="fig-small">x = 3</text>
<text x="12" y="102" class="fig-small">0 &lt; α &lt; C</text>
<text x="12" y="116" class="fig-small">needs g = 0</text>
<line x1="81.50000000000003" y1="102" x2="518.5" y2="102" class="stroke-grid"/>
<rect x="297" y="88" width="6" height="28" class="fill-b"/>
<circle cx="300" cy="102" r="5" class="fill-ink"/>
<text x="300" y="84" text-anchor="middle" class="fig-small">x = −1 and x = 2</text>
<circle cx="300" cy="102" r="5" class="fill-ink"/>
<text x="12" y="148" class="fig-small">α = C</text>
<text x="12" y="162" class="fig-small">needs g ≤ 0</text>
<line x1="81.50000000000003" y1="148" x2="518.5" y2="148" class="stroke-grid"/>
<rect x="91" y="134" width="209" height="28" class="fill-soft-d"/>
<circle cx="205" cy="148" r="5" class="fill-ink"/>
<text x="213" y="130" text-anchor="end" class="fig-small">x = 0.5</text>
<circle cx="236.7" cy="148" r="5" class="fill-ink"/>
<text x="228.7" y="130" class="fig-small">x = 1</text>
<line x1="81.50000000000003" y1="174" x2="518.5" y2="174" class="stroke-muted"/>
<line x1="110" y1="170" x2="110" y2="178" class="stroke-muted"/>
<text x="110" y="192" text-anchor="middle" class="fig-small">−2</text>
<line x1="205" y1="170" x2="205" y2="178" class="stroke-muted"/>
<text x="205" y="192" text-anchor="middle" class="fig-small">−1</text>
<line x1="300" y1="170" x2="300" y2="178" class="stroke-muted"/>
<text x="300" y="192" text-anchor="middle" class="fig-small">0</text>
<line x1="395" y1="170" x2="395" y2="178" class="stroke-muted"/>
<text x="395" y="192" text-anchor="middle" class="fig-small">1</text>
<line x1="490" y1="170" x2="490" y2="178" class="stroke-muted"/>
<text x="490" y="192" text-anchor="middle" class="fig-small">2</text>
<text x="300" y="208" text-anchor="middle" class="fig-small">g = y f − 1</text>
<text x="12" y="234" class="fig-bold">Someone hands you α = 0 and b = 0</text>
<text x="12" y="270" class="fig-small">α = 0</text>
<text x="12" y="284" class="fig-small">needs g ≥ 0</text>
<line x1="81.50000000000003" y1="270" x2="518.5" y2="270" class="stroke-grid"/>
<rect x="300" y="256" width="209" height="28" class="fill-soft-c"/>
<rect x="205" y="267" width="95" height="6" class="fill-bad"/>
<circle cx="205" cy="270" r="5" class="fill-ink"/>
<text x="205" y="256" text-anchor="middle" class="fig-small">all six houses, g = −1</text>
<text x="252.5" y="300" text-anchor="middle" class="fig-small">violation = 1</text>
</svg>
Figure 3. Each row is one group, with its allowed band shaded. For the best pushes of a six-house example on a line, every dot lands inside its row's band. For α = 0 and b = 0, every house has g = −1, one unit left of the band it needs, and the red bar is the violation. A violation is the distance a house lies outside its band.
```

Take six houses on a line at $x = -3, -1, 0.5, 1, 3, 2$ with labels $-1, -1, -1, +1, +1, +1$, and $C = 1$. Someone hands you $\alpha = [0, \tfrac1{18}, 1, 1, 0, \tfrac1{18}]$ and $b = -\tfrac13$. The balance rule holds, because $-\tfrac1{18} - 1 + 1 + \tfrac1{18} = 0$. The weight is $w = \tfrac23$, so $f = \tfrac23 x - \tfrac13$, which gives $f = [-\tfrac73, -1, 0, \tfrac13, \tfrac53, 1]$ and $g = [\tfrac43, 0, -1, -\tfrac23, \tfrac23, 0]$. The two zero-push houses have $g \ge 0$, the two free houses have $g = 0$, and the two capped houses have $g \le 0$. No house breaks its rule, so the largest violation is 0.

Next, someone hands you $\alpha = 0$ and $b = 0$. Every $g_i = -1$ but every house needs $g \ge 0$, so each violates by 1.

In numpy, with `f = (alpha * y) @ K + b`:

```python
g = y * f - 1.0
groups = margin_groups(alpha, C, tol)   # 0, 1, or 2 for each house
```

Slack comes from the same $g$: it is $\max(0, -g_i)$, which is $[0, 0, 1, \tfrac23, 0, 0]$ at the best pushes, so only the capped houses pay.

Do not sort the groups with `alpha == 0`. A solver can store $6\times 10^{-16}$ for a push that should be zero, and that house then lands in the free group with a false violation. Compare with a tolerance, `alpha <= tol` and `alpha >= C - tol`.

> [!KEY]
> At the best pushes, every house is in exactly one of three groups: far away and ignored, resting on an edge and holding the road up, or short of the edge and pushing as hard as allowed. The largest violation of those three rules is the test you wanted, and most pushes are zero.

The largest violation is a stopping measure, not a distance from the best answer. The full checklist, the *KKT conditions*, also includes the box rule and the balance rule, which `kkt_violation` does not test. For this problem a set of pushes that obeys all of them is the best set, so a violation of zero is a certificate. A wrong $b$ shifts every $g$ and creates a violation even when $\alpha$ is right.

## Change the table, change the boundary

Nothing in $D$, the rules, or the scores needs $K$ to be a table of dot products. Put another similarity in its place and the same solver runs. The *RBF* similarity $K_{ij} = \exp(-\gamma\lVert x_i - x_j \rVert^2)$ is 1 for a house and itself and falls towards 0 as two houses move apart, so it reads as "how close". The polynomial similarity $(1 + x_i^{\top}x_j)^2$ is another choice. 

Figure 4 uses forty houses, twenty in a village centre and twenty in a ring. With the dot-product $K$ and $C = 1$, the boundary is a line, and 11 of the 40 houses end up on the wrong side. With the RBF $K$, $\gamma = 0.5$, $C = 10$, the same solver draws a closed curve and gets all 40 right. Thirteen houses have a non-zero push. Nothing changed between the panels but $K$.

```figure
<svg viewBox="0 0 600 322" role="img" aria-label="Forty points with a cluster in the middle and a ring around it. On the left, a straight boundary puts eleven points on the wrong side. On the right, the same solver with a different table of similarities draws a closed curve around the cluster and gets every point right">
<rect x="12" y="16" width="276" height="276" class="fill-paper stroke-grid"/>
<line x1="288" y1="98.2" x2="12" y2="131.1" class="stroke-b"/>
<circle cx="180.4" cy="165.5" r="5" class="fill-a"/>
<circle cx="106.8" cy="157.2" r="5" class="fill-a"/>
<circle cx="180.4" cy="158.6" r="5" class="fill-a"/>
<circle cx="155.5" cy="151.2" r="5" class="fill-a"/>
<circle cx="132.1" cy="168.3" r="5" class="fill-a"/>
<circle cx="133.9" cy="137.9" r="5" class="fill-a"/>
<circle cx="157.4" cy="175.6" r="5" class="fill-a"/>
<circle cx="170.2" cy="114.9" r="5" class="fill-a"/>
<circle cx="161" cy="165.5" r="5" class="fill-a"/>
<circle cx="139.9" cy="156.8" r="5" class="fill-a"/>
<circle cx="169.8" cy="167.8" r="5" class="fill-a"/>
<circle cx="121" cy="149.9" r="5" class="fill-a"/>
<circle cx="110.9" cy="135.6" r="5" class="fill-a"/>
<circle cx="156.4" cy="179.8" r="5" class="fill-a"/>
<circle cx="162.4" cy="155.4" r="5" class="fill-a"/>
<circle cx="120.6" cy="122.7" r="5" class="fill-a"/>
<circle cx="172.1" cy="158.6" r="5" class="fill-a"/>
<circle cx="178.1" cy="167.3" r="5" class="fill-a"/>
<circle cx="159.7" cy="134.2" r="5" class="fill-a"/>
<circle cx="142.2" cy="160.4" r="5" class="fill-a"/>
<rect x="247.1" y="121.4" width="10" height="10" class="fill-c"/>
<circle cx="252.1" cy="126.4" r="9" class="stroke-bad"/>
<rect x="201.6" y="45" width="10" height="10" class="fill-c"/>
<rect x="187.3" y="40" width="10" height="10" class="fill-c"/>
<rect x="163.4" y="47.8" width="10" height="10" class="fill-c"/>
<rect x="128.9" y="49.6" width="10" height="10" class="fill-c"/>
<rect x="78.8" y="55.6" width="10" height="10" class="fill-c"/>
<rect x="83.8" y="76.3" width="10" height="10" class="fill-c"/>
<rect x="71.9" y="76.8" width="10" height="10" class="fill-c"/>
<rect x="59" y="70.8" width="10" height="10" class="fill-c"/>
<rect x="39.2" y="98.9" width="10" height="10" class="fill-c"/>
<rect x="49.8" y="124.6" width="10" height="10" class="fill-c"/>
<circle cx="54.8" cy="129.6" r="9" class="stroke-bad"/>
<rect x="53.5" y="170.6" width="10" height="10" class="fill-c"/>
<circle cx="58.5" cy="175.6" r="9" class="stroke-bad"/>
<rect x="105" y="255.3" width="10" height="10" class="fill-c"/>
<circle cx="110" cy="260.3" r="9" class="stroke-bad"/>
<rect x="137.6" y="258" width="10" height="10" class="fill-c"/>
<circle cx="142.6" cy="263" r="9" class="stroke-bad"/>
<rect x="157.9" y="246.5" width="10" height="10" class="fill-c"/>
<circle cx="162.9" cy="251.5" r="9" class="stroke-bad"/>
<rect x="190.1" y="244.7" width="10" height="10" class="fill-c"/>
<circle cx="195.1" cy="249.7" r="9" class="stroke-bad"/>
<rect x="215.4" y="211.6" width="10" height="10" class="fill-c"/>
<circle cx="220.4" cy="216.6" r="9" class="stroke-bad"/>
<rect x="240.7" y="205.6" width="10" height="10" class="fill-c"/>
<circle cx="245.7" cy="210.6" r="9" class="stroke-bad"/>
<rect x="248.5" y="173.8" width="10" height="10" class="fill-c"/>
<circle cx="253.5" cy="178.8" r="9" class="stroke-bad"/>
<rect x="254.9" y="168.3" width="10" height="10" class="fill-c"/>
<circle cx="259.9" cy="173.3" r="9" class="stroke-bad"/>
<rect x="308" y="16" width="276" height="276" class="fill-paper stroke-grid"/>
<path d="M416,208.4 L414.9,208 M420,209.6 L416,208.4 M424,210.4 L420,209.6 M428,211 L424,210.4 M432,211.3 L428,211 M436,211.4 L432,211.3 M440,211.2 L436,211.4 M444,210.8 L440,211.2 M448,210.3 L444,210.8 M452,209.5 L448,210.3 M456,208.4 L452,209.5 M457.4,208 L456,208.4 M408,205 L406.4,204 M412,206.9 L408,205 M414.9,208 L412,206.9 M457.4,208 L460,207.2 M464,205.8 L460,207.2 M468,204.1 L464,205.8 M468.2,204 L468,204.1 M404,202.6 L400.5,200 M406.4,204 L404,202.6 M468.2,204 L472,202.2 M476,200.1 L472,202.2 M476.1,200 L476,200.1 M400,199.6 L396.1,196 M400.5,200 L400,199.6 M476.1,200 L480,197.6 M482.4,196 L480,197.6 M396,195.9 L392.5,192 M396.1,196 L396,195.9 M482.4,196 L484,194.9 M487.7,192 L484,194.9 M392,191.4 L389.6,188 M392.5,192 L392,191.4 M487.7,192 L488,191.7 M492,188.1 L488,191.7 M492.1,188 L492,188.1 M388,185.6 L387.1,184 M389.6,188 L388,185.6 M492.1,188 L495.9,184 M387.1,184 L385,180 M495.9,184 L496,183.9 M499.2,180 L496,183.9 M384,177.9 L383.2,176 M385,180 L384,177.9 M499.2,180 L500,178.9 M502,176 L500,178.9 M383.2,176 L381.7,172 M502,176 L504,172.6 M504.3,172 L504,172.6 M381.7,172 L380.5,168 M504.3,172 L506.3,168 M380,166 L379.5,164 M380.5,168 L380,166 M506.3,168 L507.9,164 M379.5,164 L378.8,160 M507.9,164 L508,163.6 M509.2,160 L508,163.6 M378.8,160 L378.4,156 M509.2,160 L510.1,156 M378.4,156 L378.2,152 M510.1,156 L510.8,152 M378.2,152 L378.2,148 M510.8,152 L511.3,148 M378.2,148 L378.5,144 M511.3,148 L511.5,144 M378.5,144 L379.2,140 M511.5,144 L511.4,140 M379.2,140 L380,136.5 M380.1,136 L380,136.5 M511.4,140 L511.2,136 M380.1,136 L381.4,132 M511.2,136 L510.7,132 M381.4,132 L383,128 M510.7,132 L509.9,128 M383,128 L384,126.1 M385.1,124 L384,126.1 M509.9,128 L508.9,124 M385.1,124 L387.6,120 M508,121 L507.7,120 M508.9,124 L508,121 M387.6,120 L388,119.4 M390.6,116 L388,119.4 M507.7,120 L506.1,116 M390.6,116 L392,114.3 M394.1,112 L392,114.3 M506.1,116 L504.1,112 M394.1,112 L396,110.1 M398.3,108 L396,110.1 M504,111.8 L501.7,108 M504.1,112 L504,111.8 M398.3,108 L400,106.5 M403.2,104 L400,106.5 M500,105.6 L498.7,104 M501.7,108 L500,105.6 M403.2,104 L404,103.4 M408,100.6 L404,103.4 M409,100 L408,100.6 M496,101 L495,100 M498.7,104 L496,101 M409,100 L412,98.2 M416,96 L412,98.2 M416,96 L416,96 M492,97.5 L490,96 M495,100 L492,97.5 M416,96 L420,94.1 M424,92.4 L420,94.1 M425.1,92 L424,92.4 M484,92.4 L483.1,92 M488,94.7 L484,92.4 M490,96 L488,94.7 M425.1,92 L428,90.9 M432,89.7 L428,90.9 M436,88.6 L432,89.7 M438.9,88 L436,88.6 M472,88.1 L471.4,88 M476,89.2 L472,88.1 M480,90.6 L476,89.2 M483.1,92 L480,90.6 M438.9,88 L440,87.8 M444,87.1 L440,87.8 M448,86.6 L444,87.1 M452,86.3 L448,86.6 M456,86.3 L452,86.3 M460,86.4 L456,86.3 M464,86.7 L460,86.4 M468,87.3 L464,86.7 M471.4,88 L468,87.3" class="stroke-b"/>
<circle cx="476.4" cy="165.5" r="5" class="fill-a"/>
<circle cx="476.4" cy="165.5" r="9" class="stroke-ink"/>
<circle cx="402.8" cy="157.2" r="5" class="fill-a"/>
<circle cx="402.8" cy="157.2" r="9" class="stroke-ink"/>
<circle cx="476.4" cy="158.6" r="5" class="fill-a"/>
<circle cx="451.5" cy="151.2" r="5" class="fill-a"/>
<circle cx="428.1" cy="168.3" r="5" class="fill-a"/>
<circle cx="429.9" cy="137.9" r="5" class="fill-a"/>
<circle cx="453.4" cy="175.6" r="5" class="fill-a"/>
<circle cx="466.2" cy="114.9" r="5" class="fill-a"/>
<circle cx="466.2" cy="114.9" r="9" class="stroke-ink"/>
<circle cx="457" cy="165.5" r="5" class="fill-a"/>
<circle cx="435.9" cy="156.8" r="5" class="fill-a"/>
<circle cx="465.8" cy="167.8" r="5" class="fill-a"/>
<circle cx="417" cy="149.9" r="5" class="fill-a"/>
<circle cx="406.9" cy="135.6" r="5" class="fill-a"/>
<circle cx="452.4" cy="179.8" r="5" class="fill-a"/>
<circle cx="458.4" cy="155.4" r="5" class="fill-a"/>
<circle cx="416.6" cy="122.7" r="5" class="fill-a"/>
<circle cx="416.6" cy="122.7" r="9" class="stroke-ink"/>
<circle cx="468.1" cy="158.6" r="5" class="fill-a"/>
<circle cx="474.1" cy="167.3" r="5" class="fill-a"/>
<circle cx="455.7" cy="134.2" r="5" class="fill-a"/>
<circle cx="438.2" cy="160.4" r="5" class="fill-a"/>
<rect x="543.1" y="121.4" width="10" height="10" class="fill-c"/>
<circle cx="548.1" cy="126.4" r="9" class="stroke-ink"/>
<rect x="497.6" y="45" width="10" height="10" class="fill-c"/>
<circle cx="502.6" cy="50" r="9" class="stroke-ink"/>
<rect x="483.3" y="40" width="10" height="10" class="fill-c"/>
<rect x="459.4" y="47.8" width="10" height="10" class="fill-c"/>
<circle cx="464.4" cy="52.8" r="9" class="stroke-ink"/>
<rect x="424.9" y="49.6" width="10" height="10" class="fill-c"/>
<rect x="374.8" y="55.6" width="10" height="10" class="fill-c"/>
<rect x="379.8" y="76.3" width="10" height="10" class="fill-c"/>
<circle cx="384.8" cy="81.3" r="9" class="stroke-ink"/>
<rect x="367.9" y="76.8" width="10" height="10" class="fill-c"/>
<rect x="355" y="70.8" width="10" height="10" class="fill-c"/>
<rect x="335.2" y="98.9" width="10" height="10" class="fill-c"/>
<rect x="345.8" y="124.6" width="10" height="10" class="fill-c"/>
<circle cx="350.8" cy="129.6" r="9" class="stroke-ink"/>
<rect x="349.5" y="170.6" width="10" height="10" class="fill-c"/>
<circle cx="354.5" cy="175.6" r="9" class="stroke-ink"/>
<rect x="401" y="255.3" width="10" height="10" class="fill-c"/>
<circle cx="406" cy="260.3" r="9" class="stroke-ink"/>
<rect x="433.6" y="258" width="10" height="10" class="fill-c"/>
<rect x="453.9" y="246.5" width="10" height="10" class="fill-c"/>
<circle cx="458.9" cy="251.5" r="9" class="stroke-ink"/>
<rect x="486.1" y="244.7" width="10" height="10" class="fill-c"/>
<rect x="511.4" y="211.6" width="10" height="10" class="fill-c"/>
<circle cx="516.4" cy="216.6" r="9" class="stroke-ink"/>
<rect x="536.7" y="205.6" width="10" height="10" class="fill-c"/>
<rect x="544.5" y="173.8" width="10" height="10" class="fill-c"/>
<rect x="550.9" y="168.3" width="10" height="10" class="fill-c"/>
<text x="150" y="312" text-anchor="middle" class="fig-small">Linear K: 11 of 40 houses on the wrong side</text>
<text x="446" y="312" text-anchor="middle" class="fig-small">RBF K: 0 wrong, 13 support houses ringed</text>
</svg>
Figure 4. Nothing changed between the panels except the Gram matrix. With K = X Xᵀ the boundary is a straight line, and the red rings mark the houses it gets wrong. With the RBF similarity the same solver draws a curve around the cluster, and the black rings mark the houses with a non-zero push.
```

> [!KEY]
> The weights are the houses' pushes added up, $w = \sum_i \alpha_i y_i x_i$, and the data enter only through the table $K$. So swapping $K$ bends the boundary, and only the support vectors matter.

> [!RECAP]
> - A multiplier $\alpha_i \ge 0$ is how hard house $i$ pushes. The pushes balance, add up to $w$, and are capped at $C$.
> - $D(\alpha) = \sum_i \alpha_i - \tfrac12 v^{\top}Kv$ with $v = \alpha y$, and scores are $f = (\alpha y)K + b$.
> - $\alpha = 0$ means $z \ge 1$, $0 < \alpha < C$ means $z = 1$, and $\alpha = C$ means $z \le 1$. A support vector has $\alpha > 0$.
> - $g_i = y_i f_i - 1$, and a house's violation is its distance outside its group's band.

## Build it step by step

You write six functions, in the order the checks test them.

1. **`gram_matrix(X, kind, gamma)`**. `X` is `(n, d)`. It returns a symmetric `(n, n)` array. The linear kind is `X @ X.T`. For the RBF kind, build `d2 = sq[:, None] + sq[None, :] - 2 * G` from that table, then clip it at 0 before `np.exp`. Skipping the clip lets a diagonal entry of $-10^{-16}$ give a value slightly above 1.
2. **`dual_scores(alpha, y, K, b)`**. It returns the `(n,)` array `(alpha * y) @ K + b`. The common mistake is `alpha @ K`, which forgets the sides.
3. **`dual_objective(alpha, y, K)`**. It returns one float. Build `v = alpha * y` once and reuse it in the quadratic term.
4. **`margin_groups(alpha, C, tol)`**. It returns an integer array of shape `(n,)` with 0, 1, or 2. Start from ones and overwrite the two ends with `alpha <= tol` and `alpha >= C - tol`. Using `alpha == 0`, or 1 in place of `C`, is the common mistake.
5. **`slack_amounts(alpha, y, K, b)`**. It returns the `(n,)` array `np.maximum(0, 1 - y * f)`. Use `np.maximum`, not `max`.
6. **`kkt_violation(alpha, y, K, b, C, tol)`**. It returns one float. Build `g = y * dual_scores(...) - 1.0` and the groups, pick the violation per group with `np.where`, and return the maximum. The capped group needs `max(0, g)`, not `max(0, -g)`.

## Where this breaks

**Table size.** $K$ has $n^2$ entries: 50,000 houses need 2.5 billion numbers. The weights stay small.

**The RBF width.** A large $\gamma$ makes each house similar only to itself, so the boundary wraps every house separately. A small $\gamma$ makes everything similar and the boundary nears a line. Choose $\gamma$ by holding data back.

**A loose tolerance.** A `tol` that is too large makes a small genuine push count as zero. Keep it near $10^{-8}$.

## The exercise

You build `gram_matrix`, `dual_scores`, `dual_objective`, `margin_groups`, `slack_amounts`, and `kkt_violation`. The checks confirm the two-house Gram matrix, the polynomial and RBF entries, $D = 0.5$ at $\alpha = [0.5, 0.5]$, and scores equal to `X @ w + b` for the linear table. They confirm the groups, including a push of $10^{-16}$ and a cap other than 1, and the slack and violations on the six-house example. An independent solver then finds the pushes on the ring of Figure 4 and the nine houses of Figure 2: your RBF table must separate the ring, and the violation must be tiny at the nine-house optimum.
