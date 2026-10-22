---
title: "Pushes from each point: the dual and kernels"
slug: 033-dual-multipliers-kernels
part: "Part 2 — Classical models"
summary: Describe the widest road by how hard each house pushes on it, and see that only a table of pairwise similarities is needed.
minutes: 40
gpu: false
objectives:
  - Explain a multiplier alpha as the push one house puts on the road, and why the pushes balance.
  - Rebuild the weights as w = Σ α·y·x, and score points from the pushes alone.
  - Compute the dual objective D(α) and say what it rewards and what it charges.
  - Build a Gram matrix and show that swapping it changes the shape of the boundary.
prereqs:
  - 032-widest-road-hinge-loss
exercise: 033-dual-multipliers-kernels
---

# Pushes from each point: the dual and kernels

The last chapter described the widest road with a list of weights $w$, one number per feature. This chapter describes the same road a different way. Instead of asking "what is $w$?", you ask "how much does each house hold the road in place?"

Why bother? Picture a village centre with a ring of houses around it. No straight road keeps the ring on one side and the centre on the other. With the weights description you could add curved features by hand, such as the squares of the coordinates, but each curve you want makes the list of weights longer. The pushes description has no list of features. It asks only how alike two houses are, so you can change that one measure and the same solver draws a curve.

You build the pushes, check them against the road you already know, and then swap the measure of alikeness on the ring of houses in Figure 3.

## What you already know

These five ideas carry the chapter:

- The widest road has its edges at $z = \pm 1$, where $z = y(w^{\top}x + b)$, and its half-width is $1/\lVert w \rVert$.
- A house with $z > 1$ is outside the road and costs nothing, and a house with $z < 1$ pays slack.
- $C$ is the price of one unit of slack.
- The dot product $x_i^{\top}x_j$ is large and positive when two points point the same way, and negative when they point opposite ways.
- At the bottom of a bowl the slope is zero, so you find the bottom by setting slopes to zero.

## The road as a plank between pegs

Picture the road as a stiff plank clamped between pegs, and each house as a peg. A peg pushes on the plank with some force $\alpha_i \ge 0$, called its *multiplier*, in the direction away from the peg's own side. A peg that touches nothing pushes with force 0.

Two things hold when the plank is at rest. First, it does not slide, so the pushes from the circle side and from the square side cancel. Second, the direction of the road is built from the pushes themselves:

$$
\sum_i \alpha_i y_i = 0, \qquad w = \sum_i \alpha_i y_i x_i .
$$

The first equation is the *balance rule*. Read the second aloud: $w$ is every house's position, weighted by how hard it pushes and signed by its side.

Take the smallest case from Figure 1: a square at $x = +1$ with $y = +1$ and a circle at $x = -1$ with $y = -1$, in that order, which is also the order of the exercise. The road between them has edges at $\pm 1$, centre at 0, and $w = 1$. Equal pushes $\alpha = [0.5, 0.5]$ rebuild it, because $w = 0.5\cdot 1\cdot 1 + 0.5\cdot(-1)\cdot(-1) = 1$. The balance rule gives $0.5\cdot 1 + 0.5\cdot(-1) = 0$. Why 0.5? Each house adds $\alpha\cdot y\cdot x = \alpha$ to $w$, so two equal pushes must each be 0.5 to reach $w = 1$.

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

Where do the pushes come from? In the road problem, $\tfrac12\lVert w \rVert^2$ wants $w$ small, and each house says "keep $z_i \ge 1$". Give each demand a price $\alpha_i \ge 0$, set by a referee who wants the cost to be as high as possible: $\tfrac12\lVert w \rVert^2 - \sum_i \alpha_i\,(z_i - 1)$. If house $i$ breaks its demand, $z_i - 1 < 0$, so the referee raises the cost by raising $\alpha_i$, and a road that breaks a demand cannot win. If a house has room to spare, $z_i - 1 > 0$, raising $\alpha_i$ would lower the cost, so the referee leaves $\alpha_i$ at 0. You minimise that cost over $w$ and $b$. Set the slope in $w$ to zero, and you get $w - \sum_i \alpha_i y_i x_i = 0$. Set the slope in $b$ to zero, and you get $\sum_i \alpha_i y_i = 0$. These are the two equations above. A multiplier is the price of its house's demand.

## Score the pushes with one number

Which pushes are the best ones? Put $w = \sum_i \alpha_i y_i x_i$ back into the cost above. The $w$ disappears, and what remains depends only on $\alpha$:

$$
D(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i}\sum_{j}\alpha_i\alpha_j y_i y_j\, x_i^{\top}x_j .
$$

Read it aloud: the first term rewards total push. The second term charges half the squared length of the $w$ that those pushes build, because $\tfrac12\lVert \sum_i \alpha_i y_i x_i \rVert^2$ expands to exactly that double sum. So $D$ is the total push, minus the cost of the road's narrowness. The best pushes maximise $D$ while obeying the balance rule and a *box rule*, $0 \le \alpha_i \le C$. The box rule says a house cannot push harder than the price $C$ of its own slack. Breaking a demand never costs the road more than $C$ per unit, so the referee has no reason to charge more.

On the two-house example, take equal pushes $\alpha = [a, a]$. Then $D = 2a - 2a^2$. At $a = 0.25$ that is $0.375$. At $a = 0.5$ it is $0.5$, and at $a = 0.75$ it is $0.375$ again. The peak is at 0.5, and the peak value $0.5$ equals $\tfrac12\lVert w \rVert^2 = \tfrac12(1)^2$, the road's own objective. The best pushes reach the same value as the best weights.

## Read the data through the Gram matrix

Write $K_{ij} = x_i^{\top}x_j$. The table $K$ is the *Gram matrix*: entry $(i, j)$ says how alike houses $i$ and $j$ are. Figure 2 draws one for four points. The double sum above becomes one line of numpy, with $v = \alpha y$:

```python
v = alpha * y
D = alpha.sum() - 0.5 * v @ K @ v
```

On the two-house example, $K = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$, $v = [0.5, -0.5]$, and $v^{\top}Kv = 1$, so $D = 1 - 0.5 = 0.5$. A common mistake is to drop the labels and use $\alpha$ where $v$ belongs. Then $K\alpha = 0$, and $D$ comes out as 1.

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="A four by four table of dot products between four points, next to the four points drawn on a plane. Neighbouring points have large positive entries and opposite points have large negative entries">
<text x="12" y="22" class="fig-bold">Gram matrix K = X Xᵀ</text>
<text x="88.0" y="42" text-anchor="middle" class="fig-small">P1</text>
<text x="52" y="80.0" text-anchor="end" class="fig-small">P1</text>
<text x="140.0" y="42" text-anchor="middle" class="fig-small">P2</text>
<text x="52" y="132.0" text-anchor="end" class="fig-small">P2</text>
<text x="192.0" y="42" text-anchor="middle" class="fig-small">P3</text>
<text x="52" y="184.0" text-anchor="end" class="fig-small">P3</text>
<text x="244.0" y="42" text-anchor="middle" class="fig-small">P4</text>
<text x="52" y="236.0" text-anchor="end" class="fig-small">P4</text>
<rect x="62" y="50" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.55"/>
<text x="88.0" y="81.0" text-anchor="middle" class="fig-bold">5</text>
<rect x="114" y="50" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.44"/>
<text x="140.0" y="81.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="166" y="50" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.55"/>
<text x="192.0" y="81.0" text-anchor="middle" class="fig-bold">-5</text>
<rect x="218" y="50" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.44"/>
<text x="244.0" y="81.0" text-anchor="middle" class="fig-bold">-4</text>
<rect x="62" y="102" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.44"/>
<text x="88.0" y="133.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="114" y="102" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.55"/>
<text x="140.0" y="133.0" text-anchor="middle" class="fig-bold">5</text>
<rect x="166" y="102" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.44"/>
<text x="192.0" y="133.0" text-anchor="middle" class="fig-bold">-4</text>
<rect x="218" y="102" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.55"/>
<text x="244.0" y="133.0" text-anchor="middle" class="fig-bold">-5</text>
<rect x="62" y="154" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.55"/>
<text x="88.0" y="185.0" text-anchor="middle" class="fig-bold">-5</text>
<rect x="114" y="154" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.44"/>
<text x="140.0" y="185.0" text-anchor="middle" class="fig-bold">-4</text>
<rect x="166" y="154" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.55"/>
<text x="192.0" y="185.0" text-anchor="middle" class="fig-bold">5</text>
<rect x="218" y="154" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.44"/>
<text x="244.0" y="185.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="62" y="206" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.44"/>
<text x="88.0" y="237.0" text-anchor="middle" class="fig-bold">-4</text>
<rect x="114" y="206" width="52" height="52" class="fill-a stroke-grid" fill-opacity="0.55"/>
<text x="140.0" y="237.0" text-anchor="middle" class="fig-bold">-5</text>
<rect x="166" y="206" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.44"/>
<text x="192.0" y="237.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="218" y="206" width="52" height="52" class="fill-b stroke-grid" fill-opacity="0.55"/>
<text x="244.0" y="237.0" text-anchor="middle" class="fig-bold">5</text>
<rect x="340" y="50" width="230" height="208" class="fill-paper stroke-grid"/>
<line x1="340" y1="154" x2="570" y2="154" class="stroke-grid"/>
<line x1="455" y1="258" x2="455" y2="50" class="stroke-grid"/>
<rect x="525.7" y="113.3" width="12" height="12" class="fill-c"/>
<text x="539.3" y="110.7" class="fig-small">P1</text>
<rect x="487.3" y="78.7" width="12" height="12" class="fill-c"/>
<text x="501" y="76" class="fig-small">P2</text>
<circle cx="378.3" cy="188.7" r="6" class="fill-a"/>
<text x="386" y="180" class="fig-small">P3</text>
<circle cx="416.7" cy="223.3" r="6" class="fill-a"/>
<text x="424.3" y="214.7" class="fig-small">P4</text>
<text x="340" y="22" class="fig-bold">The same four points</text>
<text x="170" y="282" text-anchor="middle" class="fig-small">orange: similar, blue: opposite</text>
</svg>
Figure 2. Entry (i, j) of the Gram matrix is the dot product of points i and j, read as how alike they are. P1 and P2 point the same way, so their entry is 4. P1 and P3 point opposite ways, so theirs is −5. The solver in this chapter reads only this table.
```

Scoring a house needs $K$ too. Since $w^{\top}x_i = \sum_k \alpha_k y_k\, x_k^{\top}x_i$, the score of house $i$ is

$$
f_i = \sum_k \alpha_k y_k K_{ik} + b .
$$

Read it aloud: each house votes with its push and its side, weighted by how alike it is to house $i$. In numpy it is `(alpha * y) @ K + b`. On the example with $b = 0$, in the order square then circle, $v = \alpha y = [0.5, -0.5]$ and the scores are $[1, -1]$, so $y_i f_i = 1$ for both: each house rests on its edge. The dual does not pick $b$. A house on its edge fixes $b$ by requiring $y_i f_i = 1$, and the next chapter uses that.

Last chapter's loop already trains a linear SVM, so the dual has to earn its place. It earns it twice. The solver only ever reads $K$, which is the kernel gain you see next in Figure 3. And most pushes turn out to be zero, which is the sparsity gain of the next chapter.

## Change the table, change the boundary

Nothing in $D$, the rules, or the scores asks for $K$ to be a table of dot products. Put another similarity in its place and the same solver runs. The *RBF* similarity is $K_{ij} = \exp(-\gamma\lVert x_i - x_j \rVert^2)$. It is 1 for a house and itself, and it falls towards 0 as two houses move apart, so it reads as "how close". The polynomial similarity $(1 + x_i^{\top}x_j)^2$ is a third choice.

Figure 3 uses forty houses, twenty in a village centre and twenty in a ring. With the dot-product $K$ and $C = 1$, the boundary is a line, and 11 of the 40 houses end up on the wrong side. With the RBF $K$, $\gamma = 0.5$, $C = 10$, the same solver draws a closed curve and gets all 40 right. Thirteen houses have a non-zero push. Nothing changed between the panels but $K$. A later chapter, "Kernels: a similarity score that hides a bigger feature space", explains why such a table works.

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
Figure 3. Nothing changed between the panels except the Gram matrix. With K = X Xᵀ the boundary is a straight line, and the red rings mark the houses it gets wrong. With the RBF similarity the same solver draws a curve around the cluster, and the black rings mark the houses with a non-zero push.
```

> [!KEY]
> The road can be described by the pushes $\alpha_i$ of the houses. The weights are those pushes added up, $w = \sum_i \alpha_i y_i x_i$, and the data enter only through the table $K$. So instead of asking what $w$ is, you ask how hard each house pushes, and swapping $K$ bends the boundary.

## Where this breaks

**Table size.** $K$ has $n^2$ entries. Forty houses need 1600 numbers, and 50,000 houses need 2.5 billion. The weights description stays the size of the feature list however many houses you add.

**The RBF width.** A large $\gamma$ makes each house similar only to itself, so $K$ nears the identity and the boundary wraps every house separately. A small $\gamma$ makes everything similar and the boundary nears a line. You choose $\gamma$ by holding data back, as in "Honest scores: training error, test error, and holding data back".

> [!RECAP]
> - A multiplier $\alpha_i \ge 0$ is how hard house $i$ pushes on the road, and the box rule caps it at $C$.
> - The pushes balance, $\sum_i \alpha_i y_i = 0$, and add up to the weights, $w = \sum_i \alpha_i y_i x_i$.
> - $D(\alpha) = \sum_i \alpha_i - \tfrac12 v^{\top}Kv$ with $v = \alpha y$, and its peak equals the road's own objective.
> - Scores are $f = (\alpha y)K + b$, and the data appear only through $K$.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`gram_matrix(X, kind, gamma)`**. `X` is `(n, d)`. It returns a symmetric `(n, n)` array. The linear kind is `X @ X.T`. For the RBF kind, build the squared distances from that table and its diagonal, `d2 = sq[:, None] + sq[None, :] - 2 * G`, then clip them at 0 before `np.exp`. The common mistake is to skip the clip, so a diagonal entry of $-10^{-16}$ gives a value slightly above 1.

2. **`dual_scores(alpha, y, K, b)`**. It returns the `(n,)` array of scores. The shape of the code is `(alpha * y) @ K + b`. The common mistake is `alpha @ K`, which forgets the sides.

3. **`dual_objective(alpha, y, K)`**. It returns one float. Build `v = alpha * y` once and reuse it. The common mistake is the same as in step 2, using `alpha` where `v` belongs in the quadratic term.

## The exercise

You build `gram_matrix`, `dual_scores`, and `dual_objective`. The checks confirm the two-house Gram matrix, the polynomial and RBF entries against direct formulas, and symmetry on random points. They confirm $D = 0.5$ at $\alpha = [0.5, 0.5]$ and the whole curve $2a - 2a^2$. They check that the scores equal `X @ w + b` for the linear table. Finally an independent solver finds the pushes on the ring of Figure 3. Your RBF table must separate all forty houses, and your linear table must leave many of them wrong.
