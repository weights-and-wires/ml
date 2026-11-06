---
title: "Prune the tree: paying for each extra leaf"
slug: 040-pruning
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Grow a tree too big, then keep a split only if the error it removes is worth the fee for its extra leaves, so a bigger fee gives a smaller tree.
minutes: 40
gpu: false
objectives:
  - Price a tree as its training error plus a fee per leaf, and compute the break-even fee at which a split stops paying for itself.
  - Prune a tree bottom-up, and explain why the children must be pruned before the node is judged.
  - Explain why a bigger fee gives a smaller tree that sits inside the bigger one, and why you choose the fee on held-out rows.
prereqs:
  - 039-grow-and-predict
  - 011-ridge-penalty-and-closed-form
exercise: 040-pruning
---

# Prune the tree: paying for each extra leaf

The tree you grew in the previous chapter has one leaf per training row. It fits all 40 noisy points exactly, and it does worse on fresh rows than a tree of depth 3. Some of its splits capture the shape of the curve. Others only chase noise. How do you tell which is which?

A depth limit is the blunt answer. It cuts every branch at the same level, although one branch may need four questions and another only one. And you picked the limit by looking at the results. A better answer asks each split to justify itself: does it remove enough error to be worth the extra leaf it adds? That is the idea behind *pruning*: grow the tree too big on purpose, then cut back the splits that did not pay.

```figure
<svg viewBox="0 0 600 268" role="img" aria-label="Three snapshots of the four-house tree. At fee 1 all four leaves stay. At fee 3 the two weak splits, which collapse at 2, are gone and two leaves remain. At fee 50 even the strong root split has collapsed and one leaf remains.">
<line x1="105" y1="80" x2="57" y2="108" class="stroke-muted" />
<line x1="105" y1="80" x2="153" y2="108" class="stroke-muted" />
<line x1="57" y1="142" x2="33" y2="170" class="stroke-muted" />
<line x1="57" y1="142" x2="81" y2="170" class="stroke-muted" />
<line x1="153" y1="142" x2="129" y2="170" class="stroke-muted" />
<line x1="153" y1="142" x2="177" y2="170" class="stroke-muted" />
<rect x="67" y="46" width="76" height="36" rx="7" class="fill-paper stroke-ink" />
<text x="105" y="62" text-anchor="middle" >x ≤ 2.5</text>
<text x="105" y="77" class="fig-small" text-anchor="middle" >α ≥ 49</text>
<rect x="21" y="108" width="72" height="34" rx="7" class="fill-paper stroke-ink" />
<text x="57" y="123" text-anchor="middle" >x ≤ 1.5</text>
<text x="57" y="138" class="fig-small" text-anchor="middle" >α ≥ 2</text>
<rect x="117" y="108" width="72" height="34" rx="7" class="fill-paper stroke-ink" />
<text x="153" y="123" text-anchor="middle" >x ≤ 3.5</text>
<text x="153" y="138" class="fig-small" text-anchor="middle" >α ≥ 2</text>
<rect x="13" y="170" width="40" height="28" rx="6" class="fill-soft-d stroke-ink" />
<text x="33" y="189" text-anchor="middle" >1</text>
<rect x="61" y="170" width="40" height="28" rx="6" class="fill-soft-d stroke-ink" />
<text x="81" y="189" text-anchor="middle" >3</text>
<rect x="109" y="170" width="40" height="28" rx="6" class="fill-soft-d stroke-ink" />
<text x="129" y="189" text-anchor="middle" >8</text>
<rect x="157" y="170" width="40" height="28" rx="6" class="fill-soft-d stroke-ink" />
<text x="177" y="189" text-anchor="middle" >10</text>
<text x="105" y="228" class="fig-bold" text-anchor="middle" >fee α = 1</text>
<text x="105" y="248" text-anchor="middle" >4 leaves, cost 4</text>
<line x1="300" y1="80" x2="252" y2="108" class="stroke-muted" />
<line x1="300" y1="80" x2="348" y2="108" class="stroke-muted" />
<line x1="252" y1="142" x2="228" y2="170" class="stroke-muted dashed" />
<line x1="252" y1="142" x2="276" y2="170" class="stroke-muted dashed" />
<line x1="348" y1="142" x2="324" y2="170" class="stroke-muted dashed" />
<line x1="348" y1="142" x2="372" y2="170" class="stroke-muted dashed" />
<rect x="262" y="46" width="76" height="36" rx="7" class="fill-paper stroke-ink" />
<text x="300" y="62" text-anchor="middle" >x ≤ 2.5</text>
<text x="300" y="77" class="fig-small" text-anchor="middle" >α ≥ 49</text>
<rect x="216" y="108" width="72" height="34" rx="7" class="fill-soft-d stroke-ink" />
<text x="252" y="123" text-anchor="middle" >answer 2</text>
<text x="252" y="138" class="fig-small" text-anchor="middle" >one leaf</text>
<rect x="312" y="108" width="72" height="34" rx="7" class="fill-soft-d stroke-ink" />
<text x="348" y="123" text-anchor="middle" >answer 9</text>
<text x="348" y="138" class="fig-small" text-anchor="middle" >one leaf</text>
<rect x="208" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="228" y="189" class="fig-small fill-muted" text-anchor="middle" >1</text>
<rect x="256" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="276" y="189" class="fig-small fill-muted" text-anchor="middle" >3</text>
<rect x="304" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="324" y="189" class="fig-small fill-muted" text-anchor="middle" >8</text>
<rect x="352" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="372" y="189" class="fig-small fill-muted" text-anchor="middle" >10</text>
<text x="300" y="228" class="fig-bold" text-anchor="middle" >fee α = 3</text>
<text x="300" y="248" text-anchor="middle" >2 leaves, cost 10</text>
<line x1="495" y1="80" x2="447" y2="108" class="stroke-muted dashed" />
<line x1="495" y1="80" x2="543" y2="108" class="stroke-muted dashed" />
<line x1="447" y1="142" x2="423" y2="170" class="stroke-muted dashed" />
<line x1="447" y1="142" x2="471" y2="170" class="stroke-muted dashed" />
<line x1="543" y1="142" x2="519" y2="170" class="stroke-muted dashed" />
<line x1="543" y1="142" x2="567" y2="170" class="stroke-muted dashed" />
<rect x="457" y="46" width="76" height="36" rx="7" class="fill-soft-d stroke-ink" />
<text x="495" y="62" text-anchor="middle" >answer 5.5</text>
<text x="495" y="77" class="fig-small" text-anchor="middle" >one leaf</text>
<rect x="411" y="108" width="72" height="34" rx="7" class="stroke-muted dashed" />
<text x="447" y="129" class="fig-small fill-muted" text-anchor="middle" >x ≤ 1.5</text>
<rect x="507" y="108" width="72" height="34" rx="7" class="stroke-muted dashed" />
<text x="543" y="129" class="fig-small fill-muted" text-anchor="middle" >x ≤ 3.5</text>
<rect x="403" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="423" y="189" class="fig-small fill-muted" text-anchor="middle" >1</text>
<rect x="451" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="471" y="189" class="fig-small fill-muted" text-anchor="middle" >3</text>
<rect x="499" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="519" y="189" class="fig-small fill-muted" text-anchor="middle" >8</text>
<rect x="547" y="170" width="40" height="28" rx="6" class="stroke-muted dashed" />
<text x="567" y="189" class="fig-small fill-muted" text-anchor="middle" >10</text>
<text x="495" y="228" class="fig-bold" text-anchor="middle" >fee α = 50</text>
<text x="495" y="248" text-anchor="middle" >1 leaf, cost 103</text>
<text x="300" y="22" class="fig-small" text-anchor="middle" >Each split shows the fee at which it stops paying for itself. Dashed boxes have been pruned away.</text>
</svg>
Figure 1. The four-house tree at three fees. The two weak splits, each separating one pair of houses, stop paying at a fee of 2. The root split, which separates cheap from dear, keeps paying until a fee of 49. A bigger fee gives a smaller tree, and each smaller tree is inside the bigger one.
```

## What you already know

- A tree grows by recursive splitting, and every node records `n_samples`, `impurity`, and `value`, as in *Grow a tree: recursion, stopping, and overfitting*.
- Under squared error, a box's impurity is its average squared miss, so `n_samples * impurity` is its total squared miss.
- A fee on size makes a fit hold back, as the fee on large weights did in *Ridge: a fee on large weights*.
- A rule's error on its own rows flatters it, so you judge it on rows it never saw.

## A fee for every leaf

Training error alone cannot choose. A split never raises the training error, so the biggest tree always scores best. To make a split earn its place, charge a fee $\alpha$ for every leaf. The cost of a tree $T$ is

$$
R_\alpha(T) = \sum_{\text{leaves}} n_\ell \, I_\ell \;+\; \alpha \times (\text{number of leaves}) .
$$

Read it aloud: for each leaf, multiply its rows by its impurity and add those up, which is the training error. Then add $\alpha$ for every leaf. With $\alpha = 0$ the fee vanishes and the biggest tree wins. With a very large $\alpha$ the cheapest tree is a single leaf. In between, a split survives only when it removes more than $\alpha$ of error for each leaf it adds. Figure 2 shows the trade on the 40 noisy points: the training error falls as leaves are added, the fee rises in a straight line, and their sum has a lowest point.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Against the number of leaves: the training error falls, the fee alpha times leaves rises as a straight line, and their sum is a U with its lowest point at 6 leaves for a fee of 0.3.">
<line x1="70" y1="230" x2="560" y2="230" class="stroke-grid" />
<text x="62" y="234" class="fig-small" text-anchor="end" >0</text>
<line x1="70" y1="166.7" x2="560" y2="166.7" class="stroke-grid" />
<text x="62" y="170.7" class="fig-small" text-anchor="end" >8</text>
<line x1="70" y1="103.3" x2="560" y2="103.3" class="stroke-grid" />
<text x="62" y="107.3" class="fig-small" text-anchor="end" >16</text>
<line x1="70" y1="40" x2="560" y2="40" class="stroke-grid" />
<text x="62" y="44" class="fig-small" text-anchor="end" >24</text>
<line x1="70" y1="230" x2="560" y2="230" class="stroke-muted" />
<line x1="70" y1="230" x2="70" y2="234" class="stroke-muted" />
<text x="70" y="250" class="fig-small" text-anchor="middle" >0</text>
<line x1="192.5" y1="230" x2="192.5" y2="234" class="stroke-muted" />
<text x="192.5" y="250" class="fig-small" text-anchor="middle" >10</text>
<line x1="315" y1="230" x2="315" y2="234" class="stroke-muted" />
<text x="315" y="250" class="fig-small" text-anchor="middle" >20</text>
<line x1="437.5" y1="230" x2="437.5" y2="234" class="stroke-muted" />
<text x="437.5" y="250" class="fig-small" text-anchor="middle" >30</text>
<line x1="560" y1="230" x2="560" y2="234" class="stroke-muted" />
<text x="560" y="250" class="fig-small" text-anchor="middle" >40</text>
<text x="315" y="274" class="fig-small" text-anchor="middle" >number of leaves</text>
<text x="70" y="26" class="fig-small" >cost</text>
<polyline points="82.2,56.7 94.5,171.3 106.8,183.6 131.2,201.7 143.5,206.1 180.2,211.4 217,216.5 229.2,218 266,221.4 278.2,222.5 290.5,223.6 302.8,224.6 315,225.3 327.2,226.1 339.5,226.6 388.5,228.1 400.8,228.5 413,228.7 437.5,229.2 474.2,229.6 486.5,229.7 498.8,229.9 511,230 523.2,230 560,230" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="70" y1="230" x2="560" y2="135" class="stroke-b" stroke-width="2.5"/>
<polyline points="82.2,54.3 94.5,166.5 106.8,176.5 131.2,189.8 143.5,191.8 180.2,190 217,188 229.2,187.1 266,183.4 278.2,182.1 290.5,180.8 302.8,179.4 315,177.8 327.2,176.2 339.5,174.4 388.5,166.4 400.8,164.3 413,162.2 437.5,157.9 474.2,151.2 486.5,149 498.8,146.7 511,144.5 523.2,142.1 560,135" fill="none" class="stroke-d" stroke-width="2.5"/>
<circle cx="143.5" cy="191.8" r="6" class="fill-d" />
<text x="153.5" y="179.8" class="fig-small" >lowest total: 6 leaves</text>
<text x="241.5" y="212" class="fig-small fill-a" >training error</text>
<text x="437.5" y="178.8" class="fig-small fill-b" >fee: 0.3 × leaves</text>
<text x="474.2" y="143.2" class="fig-small fill-d" >total</text>
</svg>
Figure 2. Training error alone always prefers more leaves. The fee pushes the other way. The sum has a lowest point, here at 6 leaves for a fee of 0.3, where one more leaf removes less error than it costs.
```

Work it out by hand on the four houses from *Which question helps most: gain and the sorted sweep*, with areas 1 to 4 and prices 1, 3, 8, and 10. The grown tree splits at 2.5, then splits each pair, so it has four leaves. Its cost at each size is:

| Tree | Leaves | Training error | Cost |
|---|---|---|---|
| Root only | 1 | $4 \times 13.25 = 53$ | $53 + \alpha$ |
| Split at 2.5 | 2 | $2 \times 1 + 2 \times 1 = 4$ | $4 + 2\alpha$ |
| All three splits | 4 | 0 | $4\alpha$ |

Compare a pair as two leaves ($2\alpha$ for its two leaves, with no error) against the pair as one leaf ($2 + \alpha$). The split keeps its place only when $2\alpha < 2 + \alpha$, which means $\alpha < 2$. Compare the root split against the root alone: $4 + 2\alpha < 53 + \alpha$ gives $\alpha < 49$. So as the fee rises, the two weak splits go at 2, and the strong split goes at 49. That is what Figure 1 shows.

Both comparisons have the same shape. Collapsing a node into one leaf costs $n I + \alpha$. Keeping the subtree costs $R + \alpha L$, where $R$ is its training error and $L$ is its number of leaves. They are equal at

$$
\alpha = \frac{n I - R}{L - 1} .
$$

Read it aloud: the training error the subtree removes, divided by the extra leaves it adds. This is the *break-even fee* of a split. For the root split of the two-leaf tree it is $(53 - 4)/(2 - 1) = 49$. For a pair it is $(2 - 0)/(2 - 1) = 2$.

## Prune from the bottom up

Apply the formula to the full four-leaf tree and you get $(53 - 0)/(4 - 1) = 17.7$, which is wrong: the root split survives until 49. By a fee of 17.7, the two weak splits are long gone, so the root's subtree has only two leaves, not four. A node must be judged against the subtree that remains after its descendants have been judged. So prune the children first, then the node.

The procedure, for a node:

1. If it is a leaf, its cost is $n I + \alpha$.
2. Otherwise prune both children and add their costs.
3. Collapse the node into a leaf when $n I + \alpha$ is no more than that sum. A tie collapses, because the same cost with fewer leaves is a simpler tree.

In code, the comparison is one line, and the recursion returns both the pruned node and its cost:

```python
leaf_cost = node.n_samples * node.impurity + alpha  # one float
```

The usual mistake is to score the children before pruning them, which gives the 17.7 above. Build new nodes as you go, so the original tree stays intact and you can prune it again with a different $\alpha$.

> [!KEY]
> A split survives only if it removes more training error than the fee for the leaves it adds. Judge a node after its children are pruned, because only the surviving subtree counts.

## A bigger fee gives a smaller tree

In the four-house example, the tree shrinks as $\alpha$ grows and never grows back. This is true in general, and the reason is short. Look at one split and ask what its subtree could be. Every option that keeps at least one split below this node has $L \ge 2$ leaves and costs $R + \alpha L$, against $nI + \alpha$ for collapsing the node. The gap, keep minus collapse, is $(R - nI) + \alpha (L - 1)$. Because $L - 1$ is at least 1, every option's gap rises as $\alpha$ rises, and so does the smallest gap among the options. Collapsing wins when the smallest gap is zero or more, so once it wins, it wins at every larger $\alpha$. A split that goes at $\alpha$ is also gone at larger fees, and so is everything below it. The pruned trees are *nested*: each one lives inside the one before.

## Why not stop early?

Why grow a big tree to cut it back? You could instead refuse to split when the gain is small. The reason is that a split can look useless on its own and open the way to a great one below it. Take 17 houses with two features, scored by Gini impurity. Put 5 houses at (low, low) and 4 at (high, high), all labelled 0. Put 4 at (low, high) and 4 at (high, low), all labelled 1. Opposite corners share a label, which is the XOR pattern from the previous chapter, with one extra house to break the tie. Now the best first cut, on either feature, has a gain of only 0.0015. A rule that splits only when the gain is at least 0.01 stops at the root, and the single leaf gets 8 of the 17 houses wrong. Grown fully, the tree uses a second cut inside each half and ends with four pure leaves and a training error of 0. At $\alpha = 1$ the four leaves cost 4, while the root alone costs $n I + \alpha = 17 \times 0.498 + 1 = 9.47$, so pruning keeps all four. The root split breaks even at $8.47/3 = 2.82$. Pruning judges a split by what it makes possible below it. A stopping rule judges it by its own gain alone.

## Choose the fee on held-out rows

The fee sets the tree size, but which fee? Training error cannot say, because it always prefers $\alpha = 0$. You need rows the tree never trained on, as in *Why training error lies, and how k-fold cross-validation fixes it*. Prune the big tree at many fees, score each pruned tree on held-out rows, and keep the fee with the lowest error.

```figure
<svg viewBox="0 0 600 372" role="img" aria-label="Top: the error on 1000 fresh rows against the fee alpha, a U with its lowest point near a fee of 0.3. Bottom: the tree at three fees. A tiny fee leaves a jagged fit, a middle fee follows the curve, and a large fee leaves two steps.">
<line x1="70" y1="170" x2="560" y2="170" class="stroke-grid" />
<text x="62" y="174" class="fig-small" text-anchor="end" >0.1</text>
<line x1="70" y1="126.7" x2="560" y2="126.7" class="stroke-grid" />
<text x="62" y="130.7" class="fig-small" text-anchor="end" >0.2</text>
<line x1="70" y1="83.3" x2="560" y2="83.3" class="stroke-grid" />
<text x="62" y="87.3" class="fig-small" text-anchor="end" >0.3</text>
<line x1="70" y1="40" x2="560" y2="40" class="stroke-grid" />
<text x="62" y="44" class="fig-small" text-anchor="end" >0.4</text>
<line x1="70" y1="170" x2="560" y2="170" class="stroke-muted" />
<line x1="70" y1="170" x2="70" y2="174" class="stroke-muted" />
<text x="70" y="188" class="fig-small" text-anchor="middle" >0.001</text>
<line x1="179.4" y1="170" x2="179.4" y2="174" class="stroke-muted" />
<text x="179.4" y="188" class="fig-small" text-anchor="middle" >0.01</text>
<line x1="288.9" y1="170" x2="288.9" y2="174" class="stroke-muted" />
<text x="288.9" y="188" class="fig-small" text-anchor="middle" >0.1</text>
<line x1="398.3" y1="170" x2="398.3" y2="174" class="stroke-muted" />
<text x="398.3" y="188" class="fig-small" text-anchor="middle" >1</text>
<line x1="507.8" y1="170" x2="507.8" y2="174" class="stroke-muted" />
<text x="507.8" y="188" class="fig-small" text-anchor="middle" >10</text>
<text x="315" y="204" class="fig-small" text-anchor="middle" >fee per leaf, α (log scale)</text>
<text x="70" y="26" class="fig-small" >error on fresh rows</text>
<polyline points="70,126.2 92.9,126.2 93.7,126.1 197.6,126.1 198.4,125.2 203.3,125.2 204.2,125.6 205,125.8 205.8,125.8 206.6,126.2 228.7,126.2 229.5,128.3 237.7,128.3 238.5,129.1 249.1,129.1 250,129.5 253.2,129.5 254.1,134.1 271.2,134.1 272.1,133.7 284.3,133.7 285.1,128.7 287.6,128.7 288.4,128.6 299.9,128.6 300.7,128.6 302.3,128.6 303.1,130.6 304.8,130.6 305.6,133.4 306.4,132.9 318.7,132.9 319.5,133.4 325.2,133.4 326,136.2 326.9,136.2 327.7,146.9 369.4,146.9 370.2,141.1 403.8,141.1 404.6,117.2 419.3,117.2 420.1,117.8 524.8,117.8 525.6,40 560,40" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="341.6" cy="146.9" r="6" class="fill-b" />
<text x="341.6" y="166.9" class="fig-small" text-anchor="middle" >best: α near 0.3, error 0.153</text>
<text x="117" y="226" class="fig-bold" text-anchor="middle" >α = 0.01: too small</text>
<text x="117" y="362" class="fig-small" text-anchor="middle" >36 leaves</text>
<line x1="34" y1="350" x2="200" y2="350" class="stroke-muted" />
<polyline points="34,297 36.8,293.9 39.5,290.8 42.3,287.8 45.1,284.9 47.8,282.1 50.6,279.4 53.4,276.9 56.1,274.6 58.9,272.6 61.7,270.8 64.4,269.2 67.2,267.9 70,267 72.7,266.3 75.5,265.9 78.3,265.8 81,266.1 83.8,266.6 86.6,267.5 89.3,268.7 92.1,270.1 94.9,271.8 97.6,273.8 100.4,275.9 103.2,278.3 105.9,280.9 108.7,283.7 111.5,286.6 114.2,289.5 117,292.6 119.8,295.7 122.5,298.8 125.3,301.9 128.1,305 130.8,307.9 133.6,310.8 136.4,313.5 139.1,316.1 141.9,318.4 144.7,320.6 147.4,322.5 150.2,324.2 153,325.6 155.7,326.7 158.5,327.5 161.3,328 164,328.2 166.8,328.1 169.6,327.6 172.3,326.9 175.1,325.9 177.9,324.5 180.6,322.9 183.4,321.1 186.2,319 188.9,316.7 191.7,314.2 194.5,311.5 197.2,308.7 200,305.7" fill="none" class="stroke-muted dashed" />
<circle cx="139.7" cy="328.4" r="2.2" class="fill-a" />
<circle cx="78.8" cy="251.7" r="2.2" class="fill-a" />
<circle cx="40.8" cy="276.8" r="2.2" class="fill-a" />
<circle cx="36.7" cy="286.6" r="2.2" class="fill-a" />
<circle cx="169" cy="325.3" r="2.2" class="fill-a" />
<circle cx="185.5" cy="322.4" r="2.2" class="fill-a" />
<circle cx="134.7" cy="298.3" r="2.2" class="fill-a" />
<circle cx="155.1" cy="308.1" r="2.2" class="fill-a" />
<circle cx="124.2" cy="283.9" r="2.2" class="fill-a" />
<circle cx="189.2" cy="304.1" r="2.2" class="fill-a" />
<circle cx="169.4" cy="324.3" r="2.2" class="fill-a" />
<circle cx="34.5" cy="307.8" r="2.2" class="fill-a" />
<circle cx="176.3" cy="325.4" r="2.2" class="fill-a" />
<circle cx="39.6" cy="284.6" r="2.2" class="fill-a" />
<circle cx="155.1" cy="338.5" r="2.2" class="fill-a" />
<circle cx="63.2" cy="266.2" r="2.2" class="fill-a" />
<circle cx="177.3" cy="320.8" r="2.2" class="fill-a" />
<circle cx="123.9" cy="293.8" r="2.2" class="fill-a" />
<circle cx="83.8" cy="277.7" r="2.2" class="fill-a" />
<circle cx="104.2" cy="285.4" r="2.2" class="fill-a" />
<circle cx="38.7" cy="295.8" r="2.2" class="fill-a" />
<circle cx="54.6" cy="286.8" r="2.2" class="fill-a" />
<circle cx="145.3" cy="304.8" r="2.2" class="fill-a" />
<circle cx="141.4" cy="322.7" r="2.2" class="fill-a" />
<circle cx="136.2" cy="310.2" r="2.2" class="fill-a" />
<circle cx="97.7" cy="276.2" r="2.2" class="fill-a" />
<circle cx="199.5" cy="291.4" r="2.2" class="fill-a" />
<circle cx="196.8" cy="296.7" r="2.2" class="fill-a" />
<circle cx="147.8" cy="316.8" r="2.2" class="fill-a" />
<circle cx="142" cy="339.1" r="2.2" class="fill-a" />
<circle cx="148.3" cy="322.6" r="2.2" class="fill-a" />
<circle cx="98.6" cy="268.1" r="2.2" class="fill-a" />
<circle cx="56.4" cy="265" r="2.2" class="fill-a" />
<circle cx="153.8" cy="331.7" r="2.2" class="fill-a" />
<circle cx="121.2" cy="280.3" r="2.2" class="fill-a" />
<circle cx="85.5" cy="279.5" r="2.2" class="fill-a" />
<circle cx="114.6" cy="296.2" r="2.2" class="fill-a" />
<circle cx="181.7" cy="313.5" r="2.2" class="fill-a" />
<circle cx="189.1" cy="316.1" r="2.2" class="fill-a" />
<circle cx="93.4" cy="252.1" r="2.2" class="fill-a" />
<polyline points="34,307.8 35.6,307.8 35.6,286.6 37.7,286.6 37.7,295.8 39.1,295.8 39.1,284.6 40.2,284.6 40.2,276.8 47.7,276.8 47.7,286.8 55.5,286.8 55.5,265.6 71,265.6 71,251.7 81.3,251.7 81.3,278.6 89.4,278.6 89.4,252.1 95.5,252.1 95.5,276.2 98.1,276.2 98.1,268.1 101.4,268.1 101.4,285.4 109.4,285.4 109.4,296.2 117.9,296.2 117.9,280.3 122.5,280.3 122.5,293.8 124.1,293.8 124.1,283.9 129.5,283.9 129.5,298.3 135.4,298.3 135.4,310.2 137.9,310.2 137.9,328.4 140.6,328.4 140.6,322.7 141.7,322.7 141.7,339.1 143.6,339.1 143.6,304.8 146.6,304.8 146.6,316.8 148,316.8 148,322.6 151,322.6 151,331.7 154.4,331.7 154.4,308.1 155.1,308.1 155.1,338.5 162.1,338.5 162.1,325 176.8,325 176.8,320.8 179.5,320.8 179.5,313.5 183.6,313.5 183.6,322.4 187.3,322.4 187.3,316.1 189.1,316.1 189.1,304.1 193,304.1 193,296.7 198.2,296.7 198.2,291.4 200,291.4" fill="none" class="stroke-b" stroke-width="2.2"/>
<text x="313" y="226" class="fig-bold" text-anchor="middle" >α = 0.3: about right</text>
<text x="313" y="362" class="fig-small" text-anchor="middle" >6 leaves</text>
<line x1="230" y1="350" x2="396" y2="350" class="stroke-muted" />
<polyline points="230,297 232.8,293.9 235.5,290.8 238.3,287.8 241.1,284.9 243.8,282.1 246.6,279.4 249.4,276.9 252.1,274.6 254.9,272.6 257.7,270.8 260.4,269.2 263.2,267.9 266,267 268.7,266.3 271.5,265.9 274.3,265.8 277,266.1 279.8,266.6 282.6,267.5 285.3,268.7 288.1,270.1 290.9,271.8 293.6,273.8 296.4,275.9 299.2,278.3 301.9,280.9 304.7,283.7 307.5,286.6 310.2,289.5 313,292.6 315.8,295.7 318.5,298.8 321.3,301.9 324.1,305 326.8,307.9 329.6,310.8 332.4,313.5 335.1,316.1 337.9,318.4 340.7,320.6 343.4,322.5 346.2,324.2 349,325.6 351.7,326.7 354.5,327.5 357.3,328 360,328.2 362.8,328.1 365.6,327.6 368.3,326.9 371.1,325.9 373.9,324.5 376.6,322.9 379.4,321.1 382.2,319 384.9,316.7 387.7,314.2 390.5,311.5 393.2,308.7 396,305.7" fill="none" class="stroke-muted dashed" />
<circle cx="335.7" cy="328.4" r="2.2" class="fill-a" />
<circle cx="274.8" cy="251.7" r="2.2" class="fill-a" />
<circle cx="236.8" cy="276.8" r="2.2" class="fill-a" />
<circle cx="232.7" cy="286.6" r="2.2" class="fill-a" />
<circle cx="365" cy="325.3" r="2.2" class="fill-a" />
<circle cx="381.5" cy="322.4" r="2.2" class="fill-a" />
<circle cx="330.7" cy="298.3" r="2.2" class="fill-a" />
<circle cx="351.1" cy="308.1" r="2.2" class="fill-a" />
<circle cx="320.2" cy="283.9" r="2.2" class="fill-a" />
<circle cx="385.2" cy="304.1" r="2.2" class="fill-a" />
<circle cx="365.4" cy="324.3" r="2.2" class="fill-a" />
<circle cx="230.5" cy="307.8" r="2.2" class="fill-a" />
<circle cx="372.3" cy="325.4" r="2.2" class="fill-a" />
<circle cx="235.6" cy="284.6" r="2.2" class="fill-a" />
<circle cx="351.1" cy="338.5" r="2.2" class="fill-a" />
<circle cx="259.2" cy="266.2" r="2.2" class="fill-a" />
<circle cx="373.3" cy="320.8" r="2.2" class="fill-a" />
<circle cx="319.9" cy="293.8" r="2.2" class="fill-a" />
<circle cx="279.8" cy="277.7" r="2.2" class="fill-a" />
<circle cx="300.2" cy="285.4" r="2.2" class="fill-a" />
<circle cx="234.7" cy="295.8" r="2.2" class="fill-a" />
<circle cx="250.6" cy="286.8" r="2.2" class="fill-a" />
<circle cx="341.3" cy="304.8" r="2.2" class="fill-a" />
<circle cx="337.4" cy="322.7" r="2.2" class="fill-a" />
<circle cx="332.2" cy="310.2" r="2.2" class="fill-a" />
<circle cx="293.7" cy="276.2" r="2.2" class="fill-a" />
<circle cx="395.5" cy="291.4" r="2.2" class="fill-a" />
<circle cx="392.8" cy="296.7" r="2.2" class="fill-a" />
<circle cx="343.8" cy="316.8" r="2.2" class="fill-a" />
<circle cx="338" cy="339.1" r="2.2" class="fill-a" />
<circle cx="344.3" cy="322.6" r="2.2" class="fill-a" />
<circle cx="294.6" cy="268.1" r="2.2" class="fill-a" />
<circle cx="252.4" cy="265" r="2.2" class="fill-a" />
<circle cx="349.8" cy="331.7" r="2.2" class="fill-a" />
<circle cx="317.2" cy="280.3" r="2.2" class="fill-a" />
<circle cx="281.5" cy="279.5" r="2.2" class="fill-a" />
<circle cx="310.6" cy="296.2" r="2.2" class="fill-a" />
<circle cx="377.7" cy="313.5" r="2.2" class="fill-a" />
<circle cx="385.1" cy="316.1" r="2.2" class="fill-a" />
<circle cx="289.4" cy="252.1" r="2.2" class="fill-a" />
<polyline points="230,296.7 235.1,296.7 235.1,282.7 251.5,282.7 251.5,267.1 297.4,267.1 297.4,289.6 331.4,289.6 331.4,321.8 385.1,321.8 385.1,297.4 396,297.4" fill="none" class="stroke-b" stroke-width="2.2"/>
<text x="509" y="226" class="fig-bold" text-anchor="middle" >α = 3: too large</text>
<text x="509" y="362" class="fig-small" text-anchor="middle" >2 leaves</text>
<line x1="426" y1="350" x2="592" y2="350" class="stroke-muted" />
<polyline points="426,297 428.8,293.9 431.5,290.8 434.3,287.8 437.1,284.9 439.8,282.1 442.6,279.4 445.4,276.9 448.1,274.6 450.9,272.6 453.7,270.8 456.4,269.2 459.2,267.9 462,267 464.7,266.3 467.5,265.9 470.3,265.8 473,266.1 475.8,266.6 478.6,267.5 481.3,268.7 484.1,270.1 486.9,271.8 489.6,273.8 492.4,275.9 495.2,278.3 497.9,280.9 500.7,283.7 503.5,286.6 506.2,289.5 509,292.6 511.8,295.7 514.5,298.8 517.3,301.9 520.1,305 522.8,307.9 525.6,310.8 528.4,313.5 531.1,316.1 533.9,318.4 536.7,320.6 539.4,322.5 542.2,324.2 545,325.6 547.7,326.7 550.5,327.5 553.3,328 556,328.2 558.8,328.1 561.6,327.6 564.3,326.9 567.1,325.9 569.9,324.5 572.6,322.9 575.4,321.1 578.2,319 580.9,316.7 583.7,314.2 586.5,311.5 589.2,308.7 592,305.7" fill="none" class="stroke-muted dashed" />
<circle cx="531.7" cy="328.4" r="2.2" class="fill-a" />
<circle cx="470.8" cy="251.7" r="2.2" class="fill-a" />
<circle cx="432.8" cy="276.8" r="2.2" class="fill-a" />
<circle cx="428.7" cy="286.6" r="2.2" class="fill-a" />
<circle cx="561" cy="325.3" r="2.2" class="fill-a" />
<circle cx="577.5" cy="322.4" r="2.2" class="fill-a" />
<circle cx="526.7" cy="298.3" r="2.2" class="fill-a" />
<circle cx="547.1" cy="308.1" r="2.2" class="fill-a" />
<circle cx="516.2" cy="283.9" r="2.2" class="fill-a" />
<circle cx="581.2" cy="304.1" r="2.2" class="fill-a" />
<circle cx="561.4" cy="324.3" r="2.2" class="fill-a" />
<circle cx="426.5" cy="307.8" r="2.2" class="fill-a" />
<circle cx="568.3" cy="325.4" r="2.2" class="fill-a" />
<circle cx="431.6" cy="284.6" r="2.2" class="fill-a" />
<circle cx="547.1" cy="338.5" r="2.2" class="fill-a" />
<circle cx="455.2" cy="266.2" r="2.2" class="fill-a" />
<circle cx="569.3" cy="320.8" r="2.2" class="fill-a" />
<circle cx="515.9" cy="293.8" r="2.2" class="fill-a" />
<circle cx="475.8" cy="277.7" r="2.2" class="fill-a" />
<circle cx="496.2" cy="285.4" r="2.2" class="fill-a" />
<circle cx="430.7" cy="295.8" r="2.2" class="fill-a" />
<circle cx="446.6" cy="286.8" r="2.2" class="fill-a" />
<circle cx="537.3" cy="304.8" r="2.2" class="fill-a" />
<circle cx="533.4" cy="322.7" r="2.2" class="fill-a" />
<circle cx="528.2" cy="310.2" r="2.2" class="fill-a" />
<circle cx="489.7" cy="276.2" r="2.2" class="fill-a" />
<circle cx="591.5" cy="291.4" r="2.2" class="fill-a" />
<circle cx="588.8" cy="296.7" r="2.2" class="fill-a" />
<circle cx="539.8" cy="316.8" r="2.2" class="fill-a" />
<circle cx="534" cy="339.1" r="2.2" class="fill-a" />
<circle cx="540.3" cy="322.6" r="2.2" class="fill-a" />
<circle cx="490.6" cy="268.1" r="2.2" class="fill-a" />
<circle cx="448.4" cy="265" r="2.2" class="fill-a" />
<circle cx="545.8" cy="331.7" r="2.2" class="fill-a" />
<circle cx="513.2" cy="280.3" r="2.2" class="fill-a" />
<circle cx="477.5" cy="279.5" r="2.2" class="fill-a" />
<circle cx="506.6" cy="296.2" r="2.2" class="fill-a" />
<circle cx="573.7" cy="313.5" r="2.2" class="fill-a" />
<circle cx="581.1" cy="316.1" r="2.2" class="fill-a" />
<circle cx="485.4" cy="252.1" r="2.2" class="fill-a" />
<polyline points="426,280.6 527.4,280.6 527.4,318.2 592,318.2" fill="none" class="stroke-b" stroke-width="2.2"/>
</svg>
Figure 3. Scoring each pruned tree on rows it never saw gives a U in the fee. A fee near 0.3 gives the lowest error on fresh rows, and both neighbours are worse: too small keeps the noise, and too large flattens the curve. The step at the far right is clipped at 0.4; its true height is 0.608.
```

Figure 3 does this for the 40-point tree, scoring on 1000 fresh rows. A fee of 0 keeps all 40 leaves, with error 0.201, and a fee of 0.001 keeps 37 and scores about the same. A fee between about 0.22 and 0.55 leaves 6 leaves and an error of 0.153, the lowest. A fee above about 1.6 leaves 2 leaves and an error of 0.220, and a fee above about 14.5 leaves a single leaf with error 0.608. The curve is a U, so the middle fee wins. Real data rarely offers 1000 spare rows, so you would use a validation split or cross-validation. And since you picked the fee on those rows, their error is slightly flattering. Keep a final test set aside for an honest number.

## Where this breaks

- **The fee depends on the scale of the labels.** The error term is in squared units, so a fee of 0.3 means something different for prices in rupees than for prices in lakhs. Search over fees relative to the root's error, which is 21.9 here.
- **Pruning cannot add what growing missed.** It only cuts. If the grown tree never made a needed split, as in exact XOR where the gain is 0, no fee brings it back.

> [!RECAP]
> - Each leaf costs a fee $\alpha$, so the cost of a tree is its training error plus $\alpha$ times its leaves.
> - A split's break-even fee is $(nI - R)/(L - 1)$: the error it removes per extra leaf. It survives while $\alpha$ is below that.
> - Prune bottom-up, children first. Collapse a node when its leaf cost is no more than its children's total.
> - A bigger fee only removes splits, so the pruned trees are nested.
> - Grow big, then prune, because a split can unlock a better one below it. Choose $\alpha$ on held-out rows.

## Build it step by step

Build the three functions in the order the checks test them. Everything from the previous chapters, including `grow_tree`, is already in the file.

1. **`tree_cost(tree, alpha)`**. Takes a root `Node` and a float, and returns a float. Walk the leaves, add `n_samples * impurity` for each, then add `alpha` times the number of leaves. The common mistake is to count every node, not only the leaves. A leaf's share looks like this:

```python
leaf_share = node.n_samples * node.impurity + alpha  # one float
```

2. **`break_even_alpha(node)`**. Takes a node and returns the fee at which collapsing it costs the same as keeping its subtree as it stands. Use `tree_cost(node, 0.0)` for the training error and `count_leaves(node)` for the leaves. Return `float("inf")` for a leaf. The common mistake is dividing by the number of leaves, not by the extra leaves, `leaves - 1`. The division looks like this:

```python
alpha = (node.n_samples * node.impurity - kept_error) / (leaves - 1)
```

3. **`cost_complexity_prune(tree, alpha)`**. Returns a new root. A helper `prune(node)` returns a pruned copy of the node and its cost. For a leaf, return a copy and `n_samples * impurity + alpha`. For a split, prune both children first, and collapse when the leaf cost is at most the children's total. The common mistake is editing the nodes in place, which breaks the original tree. A collapsed node is a copy without children:

```python
leaf = Node(node.n_samples, node.impurity, node.value, is_leaf=True)
```

## The exercise

You implement `tree_cost`, `break_even_alpha`, and `cost_complexity_prune`. Everything from the earlier chapters is given.

The checks confirm the numbers in this chapter: costs of $4 + 2\alpha$ and $53 + \alpha$ for the four-house trees, break-even fees of 49, 2, and 17.7 for the root split, a pair, and the whole four-leaf tree, and a tree that keeps its split at $\alpha = 48$ and collapses at 49, where the costs tie. They confirm that the cost at $\alpha = 0$ equals the sum of squared misses, that pruning leaves the input tree alone, that leaf counts never rise as $\alpha$ grows, that the pruned tree costs no more than any other subtree you could reach by collapsing, and that larger fees only remove splits.
