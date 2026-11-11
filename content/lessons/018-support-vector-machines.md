---
title: "Support vector machines: the widest road, its dual, and SMO"
slug: 018-support-vector-machines
part: "Part 2 — Classical models"
summary: Pick the widest empty road between two classes, describe it by how hard each house pushes on it, and find the pushes two at a time.
minutes: 80
gpu: false
objectives:
  - Compute the margin z = y(w·x + b), and say why the road's half-width is 1/‖w‖ and what C trades off.
  - Write the hinge objective and its subgradient, check the slope with finite differences, and train a linear SVM with it.
  - Derive w = Σ α·y·x, the balance rule, and the dual objective from the Lagrangian, then score points from a Gram matrix and sort houses into three groups.
  - Compute the violation of the KKT checklist, and say what a violation of zero does and does not tell you.
  - Find the best pushes with the two-at-a-time solver, using the bounds L and H, a parabola jump, and a clip.
prereqs:
  - 006-ridge-regression
  - 008-sigmoid-and-log-loss
  - 009-fitting-logistic-regression
  - 017-gda-logistic-and-naive-bayes
exercise: 018-support-vector-machines
---

# Support vector machines: the widest road, its dual, and SMO

Fifteen lines separate these two groups of houses perfectly, so which one should you build the road along? The map in Figure 1 shows twelve houses, circles in the west and squares in the east. Every one of those lines makes zero mistakes on the houses you can see, so mistakes cannot choose between them. A line that hugs the circles is a poor choice, because a surveyor's error of a few metres drops a house on the wrong side. You want the most room to spare on both sides.

This chapter writes that goal down and solves it twice. First you train the road as a list of weights, by following the slope of a loss that cares only about houses near the road. Then you describe the same road by how hard each house pushes on it, which shows that only a few houses matter and leads to a solver that changes two pushes at a time.

## What you already know

- A linear classifier scores a point with $w^{\top}x + b$ and predicts the sign.
- Gradient descent steps against the slope of a loss ([fitting logistic regression](/lessons/009-fitting-logistic-regression)).
- $\lVert w \rVert$ is the length of $w$. The ridge penalty $\tfrac12\lVert w \rVert^2$ keeps weights small ([ridge regression](/lessons/006-ridge-regression)).
- The [log loss](/lessons/008-sigmoid-and-log-loss) never reaches zero, so on separable data logistic weights grow without limit.

## Many lines, one road

Picture a road laid along any separating line and widened until it touches a house on each side. A line that hugs the circles gives a skinny road. The orange line in Figure 1 gives the fattest road, and only three houses, the ringed ones, touch its edges. Move any other house a little and the road stays put. So the machine only has to care about points near the boundary.

```figure
<svg viewBox="0 0 600 352" role="img" aria-label="Twelve houses in two groups, fifteen thin lines that all separate them, and one wide shaded road with dashed edges and three ringed houses touching the edges">
<rect x="30" y="12" width="540" height="324" class="fill-paper stroke-grid"/>
<line x1="379.2" y1="336" x2="185.8" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="86.8" x2="30" y2="267.1" class="stroke-muted" stroke-width="1"/>
<line x1="397.6" y1="336" x2="180" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="209.7" y1="336" x2="301" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="343.2" y1="336" x2="340.2" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="164.4" y1="336" x2="426.6" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="105.6" x2="30" y2="185.1" class="stroke-muted" stroke-width="1"/>
<line x1="394.9" y1="336" x2="186.9" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="161.3" y1="336" x2="310.1" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="179.3" x2="30" y2="151.3" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="37.3" x2="30" y2="333.7" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="31" x2="30" y2="285.5" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="161.7" x2="30" y2="191.3" class="stroke-muted" stroke-width="1"/>
<line x1="36.8" y1="336" x2="484.1" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="65.4" y1="336" x2="405.5" y2="12" class="stroke-muted" stroke-width="1"/>
<polygon points="42.3,336 278.4,336 514,12 277.9,12" class="fill-soft-b"/>
<line x1="160.3" y1="336" x2="396" y2="12" class="stroke-b"/>
<line x1="278.4" y1="336" x2="514" y2="12" class="stroke-b dashed"/>
<line x1="42.3" y1="336" x2="277.9" y2="12" class="stroke-b dashed"/>
<circle cx="192" cy="93" r="5.5" class="fill-a"/>
<circle cx="175.8" cy="152.4" r="5.5" class="fill-a"/><circle cx="175.8" cy="152.4" r="10" class="stroke-ink"/>
<circle cx="165" cy="157.8" r="5.5" class="fill-a"/>
<circle cx="197.4" cy="44.4" r="5.5" class="fill-a"/>
<circle cx="165" cy="141.6" r="5.5" class="fill-a"/>
<circle cx="219" cy="93" r="5.5" class="fill-a"/><circle cx="219" cy="93" r="10" class="stroke-ink"/>
<rect x="461.9" y="244.1" width="11" height="11" class="fill-c"/>
<rect x="456.5" y="168.5" width="11" height="11" class="fill-c"/>
<rect x="386.3" y="222.5" width="11" height="11" class="fill-c"/>
<rect x="353.9" y="265.7" width="11" height="11" class="fill-c"/>
<rect x="359.3" y="211.7" width="11" height="11" class="fill-c"/><circle cx="364.8" cy="217.2" r="10" class="stroke-ink"/>
<rect x="386.3" y="190.1" width="11" height="11" class="fill-c"/>
<text x="111" y="260.4" class="fig-small">z = −1 edge</text>
<text x="478.2" y="87.6" class="fig-small">z = +1 edge</text>
<text x="321.6" y="30.9" class="fig-bold" text-anchor="middle">widest road</text>
<text x="310.8" y="319.8" class="fig-small">15 thin separating lines</text>
<text x="300" y="346" class="fig-small" text-anchor="middle">east (100 m)</text>
</svg>
Figure 1. Every thin grey line splits the circles from the squares, so training error cannot choose between them. The shaded road is the widest empty strip. Its centre line stays farthest from the nearest house on each side, and only the three ringed houses touch its edges.
```

## One number for "right side"

A circle should score negative and a square positive. Multiply the score $w^{\top}x + b$ by the label $y$, which is $-1$ for circles and $+1$ for squares, and the product is positive on the correct side for both classes:

$$
z_i = y_i(w^{\top}x_i + b) .
$$

Read it aloud: flip the score's sign for circles, so that "positive" always means "correct". Call $z_i$ the *margin* of house $i$. Take $w = [1, 0]$, $b = 0$, and three points. The point $[2, 0]$ with label $+1$ has $z = 2$. The point $[0.5, 0]$ with label $+1$ has $z = 0.5$. The point $[-1, 0]$ with label $-1$ scores $-1$, and $(-1)\times(-1) = 1$, so $z = 1$. In numpy this is `z = y * (X @ w + b)`.

Why ask for $z \ge 1$ and not $z \ge 0$? The scale of $w$ is free: doubling $w$ and $b$ doubles every $z$ and leaves the line untouched. Fixing the scale at $z \ge 1$, with the closest houses at exactly $z = 1$, removes that freedom. The distance from a point to the line is $\lvert w^{\top}x + b \rvert / \lVert w \rVert$, so a house with $z = 1$ sits at distance $1/\lVert w \rVert$. Halve $w = [1, 0]$ to $[0.5, 0]$ and the edge moves from 1 unit to 2 units out.

> [!KEY]
> The road's half-width is $1/\lVert w \rVert$. A wide road means a small $w$, so the goal is to make $\lVert w \rVert$ as small as you can while every house keeps $z \ge 1$. You minimise $\tfrac12\lVert w \rVert^2$ instead, which is smallest at the same $w$ and has the plain slope $w$.

## The hinge loss and the price of slack

You need a loss that charges a house for falling short of $z = 1$. Counting mistakes fails, because the count is flat almost everywhere and gives gradient descent nowhere to go (Figure 2). The logistic loss never reaches zero, so every house, however far away, keeps tugging. You want a loss that is exactly zero once a house is safely outside the road. That is the *hinge loss*, the shortfall from the edge, $\max(0,\, 1 - z)$. For the three example points the charges are $0$, $0.5$, and $0$, so only the house inside the road pays.

```figure
<svg viewBox="0 0 600 318" role="img" aria-label="Three loss curves against z: the 0/1 loss, the logistic loss, and the hinge loss, with the hinge flat at zero beyond z equals 1">
<rect x="50" y="16" width="520" height="250" class="fill-paper stroke-grid"/>
<line x1="50" y1="266" x2="50" y2="16" class="stroke-grid"/>
<text x="50" y="291" class="fig-small" text-anchor="middle">−2</text>
<line x1="154" y1="266" x2="154" y2="16" class="stroke-grid"/>
<text x="154" y="291" class="fig-small" text-anchor="middle">−1</text>
<line x1="258" y1="266" x2="258" y2="16" class="stroke-grid"/>
<text x="258" y="291" class="fig-small" text-anchor="middle">0</text>
<line x1="362" y1="266" x2="362" y2="16" class="stroke-grid"/>
<text x="362" y="291" class="fig-small" text-anchor="middle">1</text>
<line x1="466" y1="266" x2="466" y2="16" class="stroke-grid"/>
<text x="466" y="291" class="fig-small" text-anchor="middle">2</text>
<line x1="570" y1="266" x2="570" y2="16" class="stroke-grid"/>
<text x="570" y="291" class="fig-small" text-anchor="middle">3</text>
<line x1="50" y1="266" x2="570" y2="266" class="stroke-grid"/>
<text x="42.7" y="270.2" class="fig-small" text-anchor="end">0</text>
<line x1="50" y1="182.7" x2="570" y2="182.7" class="stroke-grid"/>
<text x="42.7" y="186.8" class="fig-small" text-anchor="end">1</text>
<line x1="50" y1="99.3" x2="570" y2="99.3" class="stroke-grid"/>
<text x="42.7" y="103.5" class="fig-small" text-anchor="end">2</text>
<line x1="50" y1="16" x2="570" y2="16" class="stroke-grid"/>
<text x="42.7" y="20.2" class="fig-small" text-anchor="end">3</text>
<line x1="50" y1="182.7" x2="258" y2="182.7" class="stroke-muted"/>
<line x1="258" y1="266" x2="570" y2="266" class="stroke-muted"/>
<polyline points="50,88.8 55.2,92.4 60.4,96.1 65.6,99.7 70.8,103.3 76,106.8 81.2,110.4 86.4,113.9 91.6,117.3 96.8,120.8 102,124.2 107.2,127.6 112.4,131 117.6,134.3 122.8,137.6 128,140.8 133.2,144.1 138.4,147.2 143.6,150.4 148.8,153.5 154,156.6 159.2,159.6 164.4,162.6 169.6,165.5 174.8,168.4 180,171.3 185.2,174.1 190.4,176.8 195.6,179.5 200.8,182.2 206,184.8 211.2,187.4 216.4,189.9 221.6,192.4 226.8,194.8 232,197.2 237.2,199.5 242.4,201.8 247.6,204 252.8,206.1 258,208.2 263.2,210.3 268.4,212.3 273.6,214.3 278.8,216.2 284,218 289.2,219.8 294.4,221.6 299.6,223.2 304.8,224.9 310,226.5 315.2,228 320.4,229.5 325.6,231 330.8,232.4 336,233.8 341.2,235.1 346.4,236.3 351.6,237.6 356.8,238.8 362,239.9 367.2,241 372.4,242.1 377.6,243.1 382.8,244.1 388,245 393.2,245.9 398.4,246.8 403.6,247.6 408.8,248.4 414,249.2 419.2,250 424.4,250.7 429.6,251.4 434.8,252 440,252.6 445.2,253.3 450.4,253.8 455.6,254.4 460.8,254.9 466,255.4 471.2,255.9 476.4,256.4 481.6,256.8 486.8,257.2 492,257.6 497.2,258 502.4,258.4 507.6,258.8 512.8,259.1 518,259.4 523.2,259.7 528.4,260 533.6,260.3 538.8,260.6 544,260.8 549.2,261.1 554.4,261.3 559.6,261.5 564.8,261.7 570,262" class="stroke-d"/>
<polyline points="50,16 55.2,20.2 60.4,24.3 65.6,28.5 70.8,32.7 76,36.8 81.2,41 86.4,45.2 91.6,49.3 96.8,53.5 102,57.7 107.2,61.8 112.4,66 117.6,70.2 122.8,74.3 128,78.5 133.2,82.7 138.4,86.8 143.6,91 148.8,95.2 154,99.3 159.2,103.5 164.4,107.7 169.6,111.8 174.8,116 180,120.2 185.2,124.3 190.4,128.5 195.6,132.7 200.8,136.8 206,141 211.2,145.2 216.4,149.3 221.6,153.5 226.8,157.7 232,161.8 237.2,166 242.4,170.2 247.6,174.3 252.8,178.5 258,182.7 263.2,186.8 268.4,191 273.6,195.2 278.8,199.3 284,203.5 289.2,207.7 294.4,211.8 299.6,216 304.8,220.2 310,224.3 315.2,228.5 320.4,232.7 325.6,236.8 330.8,241 336,245.2 341.2,249.3 346.4,253.5 351.6,257.7 356.8,261.8 362,266 367.2,266 372.4,266 377.6,266 382.8,266 388,266 393.2,266 398.4,266 403.6,266 408.8,266 414,266 419.2,266 424.4,266 429.6,266 434.8,266 440,266 445.2,266 450.4,266 455.6,266 460.8,266 466,266 471.2,266 476.4,266 481.6,266 486.8,266 492,266 497.2,266 502.4,266 507.6,266 512.8,266 518,266 523.2,266 528.4,266 533.6,266 538.8,266 544,266 549.2,266 554.4,266 559.6,266 564.8,266 570,266" class="stroke-b"/>
<circle cx="466" cy="266" r="5" class="fill-b"/>
<circle cx="310" cy="224.3" r="5" class="fill-b"/>
<circle cx="362" cy="266" r="5" class="fill-b"/>
<circle cx="392" cy="40" r="5" class="fill-b"/><text x="402" y="44" class="fig-small">point 1: z = 2, charge 0</text>
<circle cx="392" cy="60" r="5" class="fill-b"/><text x="402" y="64" class="fig-small">point 2: z = 0.5, charge 0.5</text>
<circle cx="392" cy="80" r="5" class="fill-b"/><text x="402" y="84" class="fig-small">point 3: z = 1, charge 0</text>
<text x="60.4" y="174.3" class="fig-small">0/1 loss</text>
<text x="133.2" y="36.8" class="fig-bold">hinge</text>
<text x="60.4" y="145.2" class="fig-small">logistic</text>
<text x="310" y="309" text-anchor="middle">z = y (w·x + b)</text>
</svg>
Figure 2. The 0/1 loss is flat everywhere except one jump, so it gives no slope to follow. The logistic loss stays above zero forever, so a far-away point keeps pulling. The hinge falls in a straight line and reaches exactly 0 at z = 1, which is why only points on or inside the road matter. Its three orange dots are the worked example, listed in the key.
```

A road with $z \ge 1$ for every house may not exist, so let a house fall short and charge for its shortfall, its *slack*, at a price $C$ per unit:

$$
J(w, b) = \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\max(0,\, 1 - z_i) .
$$

Read it aloud: a fee for a narrow road, plus $C$ for every unit that houses poke into it. For the example with $C = 1$, the first term is $0.5$ and the slack total is $0.5$, so $J = 1$. In numpy, `J = 0.5 * w @ w + C * np.sum(np.maximum(0.0, 1.0 - z))`. The hinge term is a sum over houses, not a mean. With `np.mean` the example gives $0.667$ instead of 1.

> [!KEY]
> $C$ is the exchange rate between a wide road and slack. A small $C$ buys width and tolerates outliers. A large $C$ buys fit and lets one odd house bend the road.

## Follow the slope

The hinge has a corner at $z = 1$. Left of it the slope is $-1$, and right of it the slope is 0. At the corner any value from $-1$ to 0 is a valid slope, each called a *subgradient*. Pick 0, so a house at exactly $z = 1$ is inactive, and call a house with $z < 1$ *active*. An active house has slope $-y_i x_i$ in $w$ and $-y_i$ in $b$, and inactive houses add nothing:

$$
\frac{\partial J}{\partial w} = w - C\sum_{i:\, z_i < 1} y_i x_i, \qquad
\frac{\partial J}{\partial b} = -C\sum_{i:\, z_i < 1} y_i .
$$

Read it aloud: the road pulls $w$ towards zero, and each active house pushes the road away from itself. In the example only the second point is active, so the slope in $w$ is $[1, 0] - [0.5, 0] = [0.5, 0]$ and the slope in $b$ is $-1$. In numpy, `grad_w = w - C * (y[z < 1] @ X[z < 1])`. With `z <= 1` the third point joins and the slope in $w$ becomes $[-0.5, 0]$.

Start at $w = 0$, $b = 0$, where $J = 12$ for the twelve houses with $C = 1$, and step against the slope with a step size that shrinks, $\text{lr}/(1 + \text{lr}\cdot t)$ at step $t$. After 3000 steps $J$ falls to 0.162 and the road's half-width is 1.76.

## The same road, described by pushes

Next, ask a different question. Instead of "what is $w$?", ask "how much does each house hold the road in place?" Picture the road as a stiff plank clamped between pegs, and each house as a peg. A peg pushes on the plank with a force $\alpha_i \ge 0$, called its *multiplier*, away from its own side. A peg that touches nothing pushes with force 0. Take a square at $x = +1$ with $y = +1$ and a circle at $x = -1$ with $y = -1$, in that order. The road between them has $w = 1$, and equal pushes $\alpha = [0.5, 0.5]$ rebuild it, because each house adds $\alpha\cdot y\cdot x = 0.5$.

## From road to pushes: the Lagrangian

The plank solves a constrained problem. House $i$ may fall short of its edge by a slack $\xi_i \ge 0$, as long as $z_i \ge 1 - \xi_i$. The hinge fills in the smallest such slack, $\xi_i = \max(0,\, 1 - z_i)$, so you minimise $\tfrac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i$.

Give the margin a multiplier $\alpha_i \ge 0$ and give $\xi_i \ge 0$ a multiplier $\mu_i \ge 0$. The Lagrangian charges for each broken constraint:

$$
L = \frac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i - \sum_i \alpha_i(z_i - 1 + \xi_i) - \sum_i \mu_i \xi_i .
$$

Because $z_i = y_i(w^{\top}x_i + b)$, the slope in $w$ is $w - \sum_i \alpha_i y_i x_i$ and the slope in $b$ is $-\sum_i \alpha_i y_i$. Set both to zero:

$$
w = \sum_i \alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0 .
$$

Read it aloud: $w$ is every house's position, weighted by its push and signed by its side, and the second equation is the *balance rule*. The slope in $\xi_i$ is $C - \alpha_i - \mu_i$. Set it to zero and the *box rule* is $0 \le \alpha_i \le C$, because $\mu_i \ge 0$.

Substitute the three slopes. The slack and $b$ cancel, and you maximise what remains:

$$
D(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j\, x_i^{\top}x_j .
$$

Read it aloud: reward every push, then charge half the squared length of the $w$ the pushes build. Equal pushes $[a, a]$ on the two-house road give $D = 2a - 2a^2$, which peaks at $a = 0.5$ with value $0.5$. The hinge objective is $0.5$ too, because the slack is zero, so the gap is zero.

*Complementary slackness* says a constraint and its multiplier are not both loose, $\alpha_i(z_i - 1 + \xi_i) = 0$ and $\mu_i \xi_i = 0$. A house with no push can sit outside the edge, a free push sits on the edge, and a push of $C$ can fall short. The KKT checklist turns that into a number.

Write $K_{ij} = x_i^{\top}x_j$, the *Gram matrix*: entry $(i, j)$ says how alike houses $i$ and $j$ are. With $v = \alpha y$, the double sum in $D$ and the score of a house are one line each:

```python
D = alpha.sum() - 0.5 * v @ K @ v         # v = alpha * y
f = (alpha * y) @ K + b                   # score f_i = sum_k alpha_k y_k K_ik + b
```

On the example $K = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$ and $v = [0.5, -0.5]$, so $v^{\top}Kv = 1$ and $D = 0.5$. If you drop the labels and use $\alpha$ where $v$ belongs, $D$ comes out as 1. With $b = 0$ the scores are $[1, -1]$, so each house rests on its edge.

## Three kinds of houses

Figure 3 shows nine houses fitted with $C = 0.3$. Only four push at all, and the plank picture says why each house lands in its group.

```figure
<svg viewBox="0 0 600 346" role="img" aria-label="Nine houses with a road. Five faint houses far from the road push with zero force, two houses on the edges push a little, and two capped houses inside the road or across it push as hard as allowed">
<defs><marker id="c015-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<rect x="20" y="12" width="560" height="300" class="fill-paper stroke-grid"/>
<polygon points="198.4,312 471.9,312 392,12 118.5,12" class="fill-soft-b"/>
<line x1="335.2" y1="312" x2="255.2" y2="12" class="stroke-b"/>
<line x1="471.9" y1="312" x2="392" y2="12" class="stroke-b dashed"/>
<line x1="198.4" y1="312" x2="118.5" y2="12" class="stroke-b dashed"/>
<circle cx="85.9" cy="80.8" r="5" class="fill-muted"/>
<circle cx="118.8" cy="205.8" r="5" class="fill-muted"/>
<circle cx="151.8" cy="137" r="5" class="fill-a"/>
<circle cx="151.8" cy="137" r="9.5" class="stroke-ink"/>
<line x1="164.4" y1="133.8" x2="176.6" y2="130.7" class="stroke-d" marker-end="url(#c015-arrow)"/>
<circle cx="178.1" cy="249.5" r="5" class="fill-muted"/>
<rect x="410.3" y="94.5" width="10" height="10" class="fill-c"/>
<circle cx="415.3" cy="99.5" r="9.5" class="stroke-ink"/>
<line x1="402.7" y1="102.7" x2="390.5" y2="105.8" class="stroke-d" marker-end="url(#c015-arrow)"/>
<rect x="443.2" y="207" width="10" height="10" class="fill-muted"/>
<rect x="397.1" y="150.8" width="10" height="10" class="fill-c"/>
<circle cx="402.1" cy="155.8" r="9.5" class="stroke-bad"/>
<line x1="389.5" y1="158.9" x2="356.6" y2="167.3" class="stroke-d" marker-end="url(#c015-arrow)"/>
<rect x="476.2" y="257" width="10" height="10" class="fill-muted"/>
<circle cx="342.8" cy="174.5" r="5" class="fill-a"/>
<circle cx="342.8" cy="174.5" r="9.5" class="stroke-bad"/>
<line x1="355.4" y1="171.3" x2="388.4" y2="163" class="stroke-d" marker-end="url(#c015-arrow)"/>
<text x="30" y="336" class="fig-small">faint: α = 0, no push</text>
<text x="190" y="336" class="fig-small">plain ring: 0 &lt; α &lt; C, on an edge</text>
<text x="410" y="336" class="fig-small">red ring: α = C, capped</text>
</svg>
Figure 3. Nine houses fitted with C = 0.3. Purple arrows show the pushes, and a longer arrow means a bigger α. Only four houses push at all. The two houses on the dashed edges push gently. The two capped houses push at the maximum, and one of them is across the centre line.
```

A house outside the road ($z > 1$) pushes with 0. A free push sits on the edge ($z = 1$). A house short of the edge ($z < 1$), including one across the centre line, pushes at the cap $\alpha_i = C$.

A *support vector* is a house with $\alpha_i > 0$. The road is $w = \sum_i \alpha_i y_i x_i$, so a house with $\alpha_i = 0$ adds nothing. Delete the five zero-push houses of Figure 3 and refit, and you get the same $w$.

> [!KEY]
> At the best pushes, every house is in exactly one of three groups: far away and ignored, resting on an edge and holding the road up, or short of the edge and pushing as hard as allowed. Most pushes are zero, so the support vectors decide the road.

Nothing in $D$ or the scores needs $K$ to be dot products. On forty houses in a ring, dot products leave 11 on the wrong side, and the RBF table in [the next chapter](/lessons/019-kernels-and-validity) gets all 40 right.

## The KKT checklist

To test a set of pushes, measure how far each house is past its edge, $g_i = y_i f_i - 1 = z_i - 1$. Each group has a rule on $g$, and a house breaks its rule by the distance to its allowed band:

| Group | Meaning | Rule | Violation |
|---|---|---|---|
| $\alpha_i = 0$ | far, no push | $g_i \ge 0$ | $\max(0, -g_i)$ |
| $0 < \alpha_i < C$ | on an edge, free | $g_i = 0$ | $\lvert g_i \rvert$ |
| $\alpha_i = C$ | capped, at or short of the edge | $g_i \le 0$ | $\max(0, g_i)$ |

Take six houses on a line at $x = -3, -1, 0.5, 1, 3, 2$ with labels $-1, -1, -1, +1, +1, +1$, and $C = 1$. Someone hands you $\alpha = [0, \tfrac1{18}, 1, 1, 0, \tfrac1{18}]$ and $b = -\tfrac13$. The pushes balance, and the weight is $w = \tfrac23$, so $f = [-\tfrac73, -1, 0, \tfrac13, \tfrac53, 1]$ and $g = [\tfrac43, 0, -1, -\tfrac23, \tfrac23, 0]$. The two zero-push houses have $g \ge 0$, the two free houses have $g = 0$, and the two capped houses have $g \le 0$, so the largest violation is 0. Hand over $\alpha = 0$ and $b = 0$ instead, and every house violates by 1.

> [!WARNING]
> Do not sort the groups with `alpha == 0`. A solver can store $6\times 10^{-16}$ for a push that should be zero, and that house then lands in the free group with a false violation. Compare with a tolerance, `alpha <= tol` and `alpha >= C - tol`.

The largest violation is a stopping measure, not a distance from the best answer. The full checklist, the *KKT conditions*, also includes the box and balance rules, which the violation does not test. A wrong $b$ shifts every $g$ and creates a violation even when $\alpha$ is right.

## Finding the pushes two at a time

Why not change one push at a time? In the two-house example, raise only the first push from 0 to 0.5. Then $\sum_i \alpha_i y_i = 0.5$, not 0, so the plank slides. Any move of a single push breaks the balance, so change two together, and let the second cancel what the first disturbs.

Pick houses $i$ and $j$ and freeze every other push. The pair $(\alpha_j, \alpha_i)$ stays inside the square from 0 to $C$ and on the line $y_i\alpha_i + y_j\alpha_j = \text{constant}$. If the labels differ that line is $\alpha_i - \alpha_j = \text{constant}$, and if they match it is $\alpha_i + \alpha_j = \text{constant}$. Only the part inside the square is allowed, and its ends on the $\alpha_j$ axis are $L$ and $H$:

$$
\begin{aligned}
y_i \ne y_j:&\quad L = \max(0,\, \alpha_j - \alpha_i), \quad H = \min(C,\, C + \alpha_j - \alpha_i), \\
y_i = y_j:&\quad L = \max(0,\, \alpha_i + \alpha_j - C), \quad H = \min(C,\, \alpha_i + \alpha_j).
\end{aligned}
$$

Read the first line aloud: if $\alpha_j$ gains an amount, $\alpha_i$ gains the same, so $\alpha_j$ stops when either push reaches 0 or $C$. With $C = 10$, $\alpha = [0, 0]$, and different labels, $L = 0$ and $H = 10$. With matching labels and $\alpha_i + \alpha_j = 12$, $L = 2$.

Along that line $D$ depends on $\alpha_j$ alone, and the quadratic part of $D$ is negative, so $D$ is an upside-down parabola. Let $E_i = f_i - y_i$ be the error of the score, zero when house $i$ sits on its edge. The slope along the line is $y_j(E_i - E_j)$, and the curvature is $-\eta$ with $\eta = K_{ii} + K_{jj} - 2K_{ij}$. Skip the pair when $\eta \le 10^{-12}$, because the houses nearly coincide and the step is huge. The peak is the Newton step, then a clip, then the partner's move:

$$
\alpha_j^{\text{new}} = \operatorname{clip}\!\left(\alpha_j + \frac{y_j(E_i - E_j)}{\eta},\; L,\; H\right), \qquad
\alpha_i^{\text{new}} = \alpha_i + y_i y_j\,(\alpha_j - \alpha_j^{\text{new}}) .
$$

Read it aloud: jump to the top of the parabola, stop at the wall if the peak is outside the box, and let the partner move the opposite way.

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
Figure 4. Along the allowed line, D is a parabola in α_j. The step jumps straight to its peak at 0.5 when the box allows it. When a wall at 0.3 stands in the way, the step stops at the wall, where D is 0.42 instead of 0.5. Clipping means stopping at the wall.
```

Work the two-house example with house $i$ the square, $C = 10$, $\alpha = [0, 0]$, and $b = 0$. Every score is 0, so $E_i = -1$ and $E_j = 1$. Then $\eta = 1 + 1 - 2(-1) = 4$, the slope is $y_j(E_i - E_j) = 2$, and the step is $2/4 = 0.5$. So $\alpha_j^{\text{new}} = 0.5$, the partner is also 0.5, and $D$ rose from 0 to 0.5 in one step. With $C = 0.3$ the peak at 0.5 is past the wall and both pushes end at 0.3 (Figure 4). Then $w = 0.3 + 0.3 = 0.6$, both margins are $0.6$, and the hinge objective is $\tfrac12(0.6)^2 + 0.3 \times 2 \times 0.4 = 0.42$, the same as $D = 2\times 0.3 - \tfrac12(0.6)^2$. The slope in $w$ is $0.6 - 0.3\times 2 = 0$.

> [!WARNING]
> Read both errors, $E_i$ and $E_j$, before you write either push. If you change $\alpha_i$ first and then compute $E_j$, the score $f_j$ already includes the updated $\alpha_i$, and the step is wrong.

The scores changed, so $b$ must change too. Choose it so that a house strictly inside $(0, C)$ lands exactly on its edge. With $\Delta_i$ and $\Delta_j$ the changes in the two pushes, putting house $i$ on its edge gives $b_1$, and putting house $j$ on its edge gives $b_2$:

$$
b_1 = b - E_i - y_i\Delta_i K_{ii} - y_j\Delta_j K_{ij}, \qquad
b_2 = b - E_j - y_i\Delta_i K_{ij} - y_j\Delta_j K_{jj} .
$$

Use $b_1$ if the updated $\alpha_i$ is strictly inside $(0, C)$, else $b_2$ if the updated $\alpha_j$ is, else the average of the two. In the example $\Delta_i = \Delta_j = 0.5$, so $b_1 = 0 + 1 - 0.5 - 0.5 = 0$. The step returns its inputs unchanged when $i = j$, when $\eta \le 10^{-12}$, and when $L \ge H$.

To choose $i$, use the checklist. Because $y_i^2 = 1$, $y_i E_i = g_i$. Update house $i$ when $y_i E_i < -\text{tol}$ and $\alpha_i < C$, or when $y_i E_i > \text{tol}$ and $\alpha_i > 0$. Scan $j$ from 0 and keep the first step that moves $\alpha_j$ by more than $10^{-7}$. Stop when a pass has violation under `tol`, changes nothing, or reaches `max_iter`. Each step raises $D$, because it jumps to that pair's peak.

> [!KEY]
> You find the best pushes by changing two at a time. The balance rule leaves one free direction, along a line. On that line $D$ is a parabola, so you jump to its peak, stop at the wall of the box if you overshoot, and repeat on any house that breaks the checklist.

> [!RECAP]
> - The half-width is $1/\lVert w \rVert$, and the hinge is a sum with $C$ pricing slack. Only houses with $z < 1$ enter the subgradient.
> - The Lagrangian's slopes give $w = \sum_i \alpha_i y_i x_i$ and $\sum_i \alpha_i y_i = 0$. Then $D(\alpha) = \sum_i \alpha_i - \tfrac12 v^{\top}Kv$, and the scores are $f = (\alpha y)K + b$.
> - $\alpha = 0$, $0 < \alpha < C$, and $\alpha = C$ mean $z \ge 1$, $z = 1$, and $z \le 1$. A violation is the distance outside that band.
> - A pair of pushes slides between $L$ and $H$, and $D$ is a parabola on that line.

## Build it step by step

You write ten functions, in the order the checks test them: three train the weights, four describe the road by pushes and test them, and three find the pushes.

1. **`functional_margins(w, b, X, y)`**. Takes `w` `(d,)`, `X` `(n, d)`, and `y` `(n,)` in $\{-1, +1\}$, and returns the `(n,)` margins. The common mistake is labels in $\{0, 1\}$.

2. **`hinge_loss_and_subgradient(w, b, X, y, C)`**. Returns `(loss, grad_w, grad_b)`. Reuse the margins, build the mask `active = z < 1.0`, and apply it to rows of `X` and entries of `y`. The common mistakes are `z <= 1` and `np.mean`.

3. **`fit_subgradient(X, y, C, lr, steps)`**. Start from zeros, loop, and subtract both slopes scaled by `lr / (1 + lr * t)`. Returns `(w, b)`.

4. **`dual_scores(alpha, y, K, b)`**. Returns `(alpha * y) @ K + b`, shape `(n,)`. The common mistake is `alpha @ K`, which forgets the sides.

5. **`dual_objective(alpha, y, K)`**. Returns one float. Build `v = alpha * y` once. The common mistake is to drop `0.5 * v @ K @ v`, which the duality-gap check rejects.

6. **`margin_groups(alpha, C, tol)`**. Returns integers 0, 1, or 2, shape `(n,)`. Start from `np.ones` and overwrite the two ends with `alpha <= tol` and `alpha >= C - tol`. The common mistake is `alpha == 0`.

7. **`kkt_violation(alpha, y, K, b, C, tol)`**. Returns one float. Build `g = y * dual_scores(...) - 1.0`, pick the violation per group with `np.where`, and return the maximum.

8. **`clip_bounds(alpha, y, C, i, j)`**. Returns `(L, H)` as two floats. Branch on `y[i] != y[j]`.

9. **`smo_step(alpha, y, K, b, C, i, j)`**. Returns `(new_alpha, new_b)`. Copy `alpha`, read `f = dual_scores(...)` and both errors before you write anything, and return early for the three unchanged cases. Snap a push within $10^{-12}$ of 0 or $C$ onto it, so rounding cannot leave a house inside the box. The common mistake is computing `E_j` after writing `alpha[i]`.

10. **`fit_smo(K, y, C, max_iter, tol)`**. Returns `(alpha, b)`. Start from zeros, test house $i$ with `y[i] * (f[i] - y[i])`, and loop over passes. The common mistake is forgetting the stop on a pass that changes nothing.

## Where this breaks

**A large C.** Each active house contributes $C$ to the slope, so a step that suits $C = 1$ overshoots for $C = 100$. Choose $C$ by holding data back, as in [honest scores](/lessons/001-loss-and-honest-scores).

**Table size.** $K$ has $n^2$ entries, so 50,000 houses need 2.5 billion numbers while the weights stay small. A pair step costs about $n$, and a pass about $n^2$.

## The exercise

You build all ten functions. The checks confirm margins of 2, 0.5, and 1, a loss of exactly 1 and a slope of `[0.5, 0]` and $-1$ (which catches a mean and a `<=`), and the slope against finite differences. Training on the twelve houses of Figure 1 must put every house on its correct side within 5 percent of the best objective. They confirm $D = 0.5$, the groups, and the violations on the six-house example and at an independent optimum. Last, `smo_step` is checked on the two-house example, at the wall, and over random pairs, and `fit_smo` must find the road of an independent solver. At those pushes the hinge objective and `dual_objective` differ by less than $10^{-6}$, and $w = \sum_i \alpha_i y_i x_i$ matches the weights from `fit_subgradient`.

## Further reading

For the Lagrangian, the stationarity conditions, and the SVM dual, see [CS229 lecture 7, Optimal margin classifier](https://see.stanford.edu/Course/CS229/37) (Andrew Ng).
