---
title: Grow a tree, then prune it
slug: 023-grow-and-prune-trees
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: A tree asks the best question in a box, repeats inside each half until a stopping rule fires, then cuts back splits that do not pay for their leaves.
minutes: 75
gpu: false
objectives:
  - Grow a tree by applying the best split to a box and recursing into each half, and name the four stopping rules with a reason for each.
  - Predict by walking a row from the root to a leaf, and explain why a row on the threshold goes left.
  - Show that a tree with no depth limit fits its training rows exactly, and say why that is a warning and not a success.
  - Price a tree as its training error plus a fee per leaf, compute the break-even fee of a split, and prune bottom-up.
  - Explain why a bigger fee gives a smaller tree that sits inside the bigger one, and why you choose the fee on held-out rows.
prereqs:
  - 022-impurity-and-best-split
  - 006-ridge-regression
exercise: 023-grow-and-prune-trees
---

# Grow a tree, then prune it

You can find the single best question for a set of rows. Take eight houses, each with an area and a distance to the metro, and a label that says whether the house sold for a high price. The best first question is "is the area at most 4.75?" It sorts the houses well, but one question is rarely enough. The left box still holds four cheap houses and one expensive one. What do you ask next, when do you stop, and how do you cut back a tree that has gone too far?

If you stop after one question, you leave mixed boxes and give a poor answer to every house in them. If you never stop, you keep asking until every house has a box of its own, and the tree has memorised the houses instead of learning anything. A depth limit is the blunt cure, because it cuts every branch at the same level, although one branch may need four questions and another only one. A better cure asks each split to justify itself: does it remove enough error to be worth the leaf it adds? This chapter grows a tree by repeating one move, shows how it predicts, and then prunes it back with that test.

```figure
<svg viewBox="0 0 600 316" role="img" aria-label="Left: eight houses on a map of area and distance to the metro, cut into three rectangles by two cuts. Right: the tree that makes those cuts. The first cut splits all eight houses, the second splits only the five on the left.">
<text x="20" y="22" class="fig-bold" >Map of the houses</text>
<rect x="50" y="190" width="142.5" height="60" class="fill-soft-d" />
<rect x="50" y="40" width="142.5" height="150" class="fill-soft-a" />
<rect x="192.5" y="40" width="97.5" height="210" class="fill-soft-d" />
<line x1="50" y1="250" x2="290" y2="250" class="stroke-muted" />
<line x1="50" y1="250" x2="50" y2="40" class="stroke-muted" />
<line x1="50" y1="250" x2="50" y2="254" class="stroke-muted" />
<text x="50" y="268" class="fig-small" text-anchor="middle" >0</text>
<line x1="110" y1="250" x2="110" y2="254" class="stroke-muted" />
<text x="110" y="268" class="fig-small" text-anchor="middle" >2</text>
<line x1="170" y1="250" x2="170" y2="254" class="stroke-muted" />
<text x="170" y="268" class="fig-small" text-anchor="middle" >4</text>
<line x1="230" y1="250" x2="230" y2="254" class="stroke-muted" />
<text x="230" y="268" class="fig-small" text-anchor="middle" >6</text>
<line x1="290" y1="250" x2="290" y2="254" class="stroke-muted" />
<text x="290" y="268" class="fig-small" text-anchor="middle" >8</text>
<line x1="46" y1="250" x2="50" y2="250" class="stroke-muted" />
<text x="42" y="254" class="fig-small" text-anchor="end" >0</text>
<line x1="46" y1="190" x2="50" y2="190" class="stroke-muted" />
<text x="42" y="194" class="fig-small" text-anchor="end" >2</text>
<line x1="46" y1="130" x2="50" y2="130" class="stroke-muted" />
<text x="42" y="134" class="fig-small" text-anchor="end" >4</text>
<line x1="46" y1="70" x2="50" y2="70" class="stroke-muted" />
<text x="42" y="74" class="fig-small" text-anchor="end" >6</text>
<text x="170" y="288" class="fig-small" text-anchor="middle" >area, tens of m²</text>
<text x="20" y="36" class="fig-small" >km</text>
<line x1="192.5" y1="40" x2="192.5" y2="250" class="stroke-b" stroke-width="2.5"/>
<line x1="50" y1="190" x2="192.5" y2="190" class="stroke-b" stroke-width="2.5"/>
<text x="198.5" y="54" class="fig-small fill-b" >cut 1</text>
<text x="56" y="184" class="fig-small fill-b" >cut 2</text>
<circle cx="80" cy="100" r="6" class="fill-a" />
<circle cx="110" cy="160" r="6" class="fill-a" />
<circle cx="125" cy="220" r="6" class="fill-d" />
<circle cx="140" cy="70" r="6" class="fill-a" />
<circle cx="170" cy="130" r="6" class="fill-a" />
<circle cx="215" cy="190" r="6" class="fill-d" />
<circle cx="230" cy="100" r="6" class="fill-d" />
<circle cx="260" cy="220" r="6" class="fill-d" />
<text x="170" y="308" class="fig-small" text-anchor="middle" >purple dots: expensive houses.  blue dots: cheap houses.</text>
<text x="340" y="22" class="fig-bold" >The tree</text>
<rect x="405" y="44" width="130" height="44" rx="8" class="fill-paper stroke-muted" />
<text x="470" y="63" text-anchor="middle" >area ≤ 4.75 ?</text>
<text x="470" y="80" class="fig-small" text-anchor="middle" >8 houses, Gini 0.50</text>
<rect x="345" y="138" width="130" height="44" rx="8" class="fill-paper stroke-muted" />
<text x="410" y="157" text-anchor="middle" >distance ≤ 2.0 ?</text>
<text x="410" y="174" class="fig-small" text-anchor="middle" >5 houses, Gini 0.32</text>
<rect x="486" y="138" width="104" height="44" rx="8" class="fill-soft-d stroke-muted" />
<text x="538" y="157" text-anchor="middle" >answer 1.0</text>
<text x="538" y="174" class="fig-small" text-anchor="middle" >3 houses, pure</text>
<rect x="305" y="240" width="100" height="44" rx="8" class="fill-soft-d stroke-muted" />
<text x="355" y="259" text-anchor="middle" >answer 1.0</text>
<text x="355" y="276" class="fig-small" text-anchor="middle" >1 house</text>
<rect x="415" y="240" width="100" height="44" rx="8" class="fill-soft-a stroke-muted" />
<text x="465" y="259" text-anchor="middle" >answer 0.0</text>
<text x="465" y="276" class="fig-small" text-anchor="middle" >4 houses</text>
<line x1="470" y1="88" x2="410" y2="138" class="stroke-muted" />
<text x="430" y="115" class="fig-small" text-anchor="end" >yes</text>
<line x1="470" y1="88" x2="538" y2="138" class="stroke-muted" />
<text x="514" y="115" class="fig-small" >no</text>
<line x1="410" y1="182" x2="355" y2="240" class="stroke-muted" />
<text x="372.5" y="213" class="fig-small" text-anchor="end" >yes</text>
<line x1="410" y1="182" x2="465" y2="240" class="stroke-muted" />
<text x="447.5" y="213" class="fig-small" >no</text>
</svg>
Figure 1. Two cuts turn the map into three rectangles, and each rectangle is a leaf. Cut 1 splits all eight houses. Cut 2 works only inside the left box, because the right box is already pure.
```

## What you already know

- `best_split` finds the best single question for a box of rows, and returns nothing when no question is possible, as in [impurity, gain, and the best split](/lessons/022-impurity-and-best-split).
- A leaf answers with the mean of the labels in its box. A box is pure when its impurity is 0.
- A function can call itself on a smaller copy of its problem, as long as it has a case where it stops.
- Under squared error, a box's impurity is its average squared miss, so `n_samples * impurity` is its total squared miss.
- A fee on size makes a fit hold back, as the fee on large weights did in [ridge regression](/lessons/006-ridge-regression). A rule's error on its own rows flatters it, so you judge it on rows it never saw.

## Ask again inside each box

One cut takes a box of eight rows and produces two boxes with fewer rows each. Each is the same kind of problem as the first one: a set of rows, and a need for the best question. So reuse the procedure. Apply the best question to a box, then apply the whole procedure again to the left half and to the right half. That is recursion, and Figure 1 shows the result.

Trace it on the eight houses. The root box has impurity (Gini) 0.5, and the best cut is area at most 4.75, with gain 0.3. The right box holds three houses, all expensive, so it is pure and becomes a leaf with answer 1.0. The left box holds five houses, one expensive and four cheap, with Gini 0.32. Its best cut is distance at most 2.0, and it leaves a box of one expensive house and a box of four cheap ones. Both are pure, so the tree has three leaves.

Every node stores three numbers, whether it is a leaf or not: how many training rows reached it (`n_samples`), its impurity, and its `value`, the mean of its labels. A node that splits also stores the feature and the threshold, and is an *internal node*. A leaf has no children, and its value is the answer it gives.

## Walk a row to a leaf

To predict for an unseen house, start at the root and answer each question in turn. A house with area 3.5 and distance 1.5 passes "area at most 4.75" and goes left. It passes "distance at most 2.0" and lands in the leaf with answer 1.0. A house with area 3.5 and distance 4.0 fails the second question and lands in the leaf with answer 0.0.

A row goes left when its value is at most the threshold, and right otherwise. This must match the way the tree was grown, which sent rows with `X[:, feature] <= threshold` to the left box. Using `<` would send a row that sits exactly on the threshold the wrong way. One step of the walk is:

```python
node = node.left if row[node.feature] <= node.threshold else node.right
```

Repeat the step until the node is a leaf, and return its value. A prediction costs one question per level of the tree.

## When to stop

A recursive procedure needs a base case, a place where it returns without recursing. A tree has four, and each one has a reason.

1. **The box is pure.** Impurity is 0 (in code, at most `1e-12`), so every row has the same label and any cut has gain 0.
2. **The depth limit is reached.** Depth counts the questions on the path from the root to a leaf. Each level can double the number of boxes, so a cap on depth keeps the tree small.
3. **No cut is possible.** `best_split` returns `None` when the box has fewer than $2 \times$ `min_samples_leaf` rows, or when every feature is constant in the box. Rows with identical features cannot be separated, whatever their labels are.
4. **No cut helps.** The best gain is at most `1e-12`. A cut that does not make the boxes purer adds a question and buys nothing. The check uses a tiny tolerance, not exactly 0, because gains are floating-point numbers that can come out as $10^{-17}$ when they are really zero.

> [!KEY]
> Grow a tree by taking the best question, then repeating inside each half. Stop when the box is pure, the depth limit is reached, no cut is possible, or no cut helps.

## Why not grow until nothing is left?

Suppose you remove the depth limit. The tree keeps cutting until every box is pure. If all 40 training rows have different features and different labels, that means one leaf per row, and the training error is 0. That sounds perfect, and it is a warning.

Figure 2 fits 40 noisy points from a sine curve. The noise has standard deviation 0.3, so no rule can beat a squared error of about $0.3^2 = 0.09$ on fresh rows. A tree of depth 1 has two leaves, an error of 0.185 on its own rows and 0.220 on fresh rows, so it is too stiff. A tree of depth 3 has eight leaves, 0.079 on its own rows and 0.166 on fresh rows. A tree with no limit has 40 leaves, 0.000 on its own rows and 0.201 on fresh rows. It chases each noisy point, so it does worse on unseen rows than the depth-3 tree.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Three step-function fits to 40 noisy points from a sine curve: depth 1, depth 3, and no depth limit. The first is too stiff, the second follows the curve, and the third passes through every point, including the noise.">
<text x="118" y="20" class="fig-bold" text-anchor="middle" >Depth 1</text>
<text x="118" y="38" class="fig-small" text-anchor="middle" >2 leaves</text>
<line x1="40" y1="196" x2="196" y2="196" class="stroke-muted" />
<line x1="40" y1="196" x2="40" y2="200" class="stroke-muted" />
<text x="40" y="214" class="fig-small" text-anchor="middle" >0</text>
<line x1="92" y1="196" x2="92" y2="200" class="stroke-muted" />
<text x="92" y="214" class="fig-small" text-anchor="middle" >2</text>
<line x1="144" y1="196" x2="144" y2="200" class="stroke-muted" />
<text x="144" y="214" class="fig-small" text-anchor="middle" >4</text>
<line x1="196" y1="196" x2="196" y2="200" class="stroke-muted" />
<text x="196" y="214" class="fig-small" text-anchor="middle" >6</text>
<line x1="40" y1="127" x2="196" y2="127" class="stroke-grid" />
<polyline points="40,127 42.6,122.9 45.2,118.9 47.8,115 50.4,111.2 53,107.5 55.6,104.1 58.2,100.9 60.8,97.9 63.4,95.2 66,92.8 68.6,90.8 71.2,89.2 73.8,87.9 76.4,87 79,86.5 81.6,86.4 84.2,86.8 86.8,87.5 89.4,88.6 92,90.1 94.6,92 97.2,94.2 99.8,96.7 102.4,99.6 105,102.7 107.6,106.1 110.2,109.7 112.8,113.4 115.4,117.3 118,121.3 120.6,125.3 123.2,129.4 125.8,133.4 128.4,137.4 131,141.2 133.6,145 136.2,148.5 138.8,151.8 141.4,154.9 144,157.7 146.6,160.2 149.2,162.4 151.8,164.2 154.4,165.6 157,166.7 159.6,167.3 162.2,167.6 164.8,167.4 167.4,166.9 170,165.9 172.6,164.6 175.2,162.9 177.8,160.8 180.4,158.4 183,155.6 185.6,152.6 188.2,149.4 190.8,145.9 193.4,142.2 196,138.3" fill="none" class="stroke-muted dashed" />
<circle cx="139.4" cy="167.9" r="2.6" class="fill-a" />
<circle cx="82.1" cy="68" r="2.6" class="fill-a" />
<circle cx="46.4" cy="100.7" r="2.6" class="fill-a" />
<circle cx="42.6" cy="113.5" r="2.6" class="fill-a" />
<circle cx="166.9" cy="163.8" r="2.6" class="fill-a" />
<circle cx="182.4" cy="160.1" r="2.6" class="fill-a" />
<circle cx="134.6" cy="128.6" r="2.6" class="fill-a" />
<circle cx="153.8" cy="141.5" r="2.6" class="fill-a" />
<circle cx="124.8" cy="109.9" r="2.6" class="fill-a" />
<circle cx="185.9" cy="136.3" r="2.6" class="fill-a" />
<circle cx="167.3" cy="162.6" r="2.6" class="fill-a" />
<circle cx="40.4" cy="141" r="2.6" class="fill-a" />
<circle cx="173.8" cy="163.9" r="2.6" class="fill-a" />
<circle cx="45.2" cy="110.9" r="2.6" class="fill-a" />
<circle cx="153.8" cy="181" r="2.6" class="fill-a" />
<circle cx="67.4" cy="86.9" r="2.6" class="fill-a" />
<circle cx="174.7" cy="158" r="2.6" class="fill-a" />
<circle cx="124.5" cy="122.9" r="2.6" class="fill-a" />
<circle cx="86.8" cy="101.9" r="2.6" class="fill-a" />
<circle cx="105.9" cy="112" r="2.6" class="fill-a" />
<circle cx="44.4" cy="125.5" r="2.6" class="fill-a" />
<circle cx="59.4" cy="113.7" r="2.6" class="fill-a" />
<circle cx="144.6" cy="137.2" r="2.6" class="fill-a" />
<circle cx="141" cy="160.5" r="2.6" class="fill-a" />
<circle cx="136" cy="144.2" r="2.6" class="fill-a" />
<circle cx="99.9" cy="99.9" r="2.6" class="fill-a" />
<circle cx="195.6" cy="119.7" r="2.6" class="fill-a" />
<circle cx="193" cy="126.7" r="2.6" class="fill-a" />
<circle cx="146.9" cy="152.8" r="2.6" class="fill-a" />
<circle cx="141.5" cy="181.8" r="2.6" class="fill-a" />
<circle cx="147.4" cy="160.3" r="2.6" class="fill-a" />
<circle cx="100.7" cy="89.3" r="2.6" class="fill-a" />
<circle cx="61.1" cy="85.4" r="2.6" class="fill-a" />
<circle cx="152.6" cy="172.2" r="2.6" class="fill-a" />
<circle cx="122" cy="105.2" r="2.6" class="fill-a" />
<circle cx="88.4" cy="104.2" r="2.6" class="fill-a" />
<circle cx="115.8" cy="125.9" r="2.6" class="fill-a" />
<circle cx="178.8" cy="148.5" r="2.6" class="fill-a" />
<circle cx="185.7" cy="151.9" r="2.6" class="fill-a" />
<circle cx="95.8" cy="68.6" r="2.6" class="fill-a" />
<polyline points="40,105.7 135.3,105.7 135.3,154.5 196,154.5" fill="none" class="stroke-b" stroke-width="2.2"/>
<text x="118" y="236" class="fig-small" text-anchor="middle" >error on these rows 0.185</text>
<text x="118" y="254" class="fig-small" text-anchor="middle" >on fresh rows 0.220</text>
<text x="314" y="20" class="fig-bold" text-anchor="middle" >Depth 3</text>
<text x="314" y="38" class="fig-small" text-anchor="middle" >8 leaves</text>
<line x1="236" y1="196" x2="392" y2="196" class="stroke-muted" />
<line x1="236" y1="196" x2="236" y2="200" class="stroke-muted" />
<text x="236" y="214" class="fig-small" text-anchor="middle" >0</text>
<line x1="288" y1="196" x2="288" y2="200" class="stroke-muted" />
<text x="288" y="214" class="fig-small" text-anchor="middle" >2</text>
<line x1="340" y1="196" x2="340" y2="200" class="stroke-muted" />
<text x="340" y="214" class="fig-small" text-anchor="middle" >4</text>
<line x1="392" y1="196" x2="392" y2="200" class="stroke-muted" />
<text x="392" y="214" class="fig-small" text-anchor="middle" >6</text>
<line x1="236" y1="127" x2="392" y2="127" class="stroke-grid" />
<polyline points="236,127 238.6,122.9 241.2,118.9 243.8,115 246.4,111.2 249,107.5 251.6,104.1 254.2,100.9 256.8,97.9 259.4,95.2 262,92.8 264.6,90.8 267.2,89.2 269.8,87.9 272.4,87 275,86.5 277.6,86.4 280.2,86.8 282.8,87.5 285.4,88.6 288,90.1 290.6,92 293.2,94.2 295.8,96.7 298.4,99.6 301,102.7 303.6,106.1 306.2,109.7 308.8,113.4 311.4,117.3 314,121.3 316.6,125.3 319.2,129.4 321.8,133.4 324.4,137.4 327,141.2 329.6,145 332.2,148.5 334.8,151.8 337.4,154.9 340,157.7 342.6,160.2 345.2,162.4 347.8,164.2 350.4,165.6 353,166.7 355.6,167.3 358.2,167.6 360.8,167.4 363.4,166.9 366,165.9 368.6,164.6 371.2,162.9 373.8,160.8 376.4,158.4 379,155.6 381.6,152.6 384.2,149.4 386.8,145.9 389.4,142.2 392,138.3" fill="none" class="stroke-muted dashed" />
<circle cx="335.4" cy="167.9" r="2.6" class="fill-a" />
<circle cx="278.1" cy="68" r="2.6" class="fill-a" />
<circle cx="242.4" cy="100.7" r="2.6" class="fill-a" />
<circle cx="238.6" cy="113.5" r="2.6" class="fill-a" />
<circle cx="362.9" cy="163.8" r="2.6" class="fill-a" />
<circle cx="378.4" cy="160.1" r="2.6" class="fill-a" />
<circle cx="330.6" cy="128.6" r="2.6" class="fill-a" />
<circle cx="349.8" cy="141.5" r="2.6" class="fill-a" />
<circle cx="320.8" cy="109.9" r="2.6" class="fill-a" />
<circle cx="381.9" cy="136.3" r="2.6" class="fill-a" />
<circle cx="363.3" cy="162.6" r="2.6" class="fill-a" />
<circle cx="236.4" cy="141" r="2.6" class="fill-a" />
<circle cx="369.8" cy="163.9" r="2.6" class="fill-a" />
<circle cx="241.2" cy="110.9" r="2.6" class="fill-a" />
<circle cx="349.8" cy="181" r="2.6" class="fill-a" />
<circle cx="263.4" cy="86.9" r="2.6" class="fill-a" />
<circle cx="370.7" cy="158" r="2.6" class="fill-a" />
<circle cx="320.5" cy="122.9" r="2.6" class="fill-a" />
<circle cx="282.8" cy="101.9" r="2.6" class="fill-a" />
<circle cx="301.9" cy="112" r="2.6" class="fill-a" />
<circle cx="240.4" cy="125.5" r="2.6" class="fill-a" />
<circle cx="255.4" cy="113.7" r="2.6" class="fill-a" />
<circle cx="340.6" cy="137.2" r="2.6" class="fill-a" />
<circle cx="337" cy="160.5" r="2.6" class="fill-a" />
<circle cx="332" cy="144.2" r="2.6" class="fill-a" />
<circle cx="295.9" cy="99.9" r="2.6" class="fill-a" />
<circle cx="391.6" cy="119.7" r="2.6" class="fill-a" />
<circle cx="389" cy="126.7" r="2.6" class="fill-a" />
<circle cx="342.9" cy="152.8" r="2.6" class="fill-a" />
<circle cx="337.5" cy="181.8" r="2.6" class="fill-a" />
<circle cx="343.4" cy="160.3" r="2.6" class="fill-a" />
<circle cx="296.7" cy="89.3" r="2.6" class="fill-a" />
<circle cx="257.1" cy="85.4" r="2.6" class="fill-a" />
<circle cx="348.6" cy="172.2" r="2.6" class="fill-a" />
<circle cx="318" cy="105.2" r="2.6" class="fill-a" />
<circle cx="284.4" cy="104.2" r="2.6" class="fill-a" />
<circle cx="311.8" cy="125.9" r="2.6" class="fill-a" />
<circle cx="374.8" cy="148.5" r="2.6" class="fill-a" />
<circle cx="381.7" cy="151.9" r="2.6" class="fill-a" />
<circle cx="291.8" cy="68.6" r="2.6" class="fill-a" />
<polyline points="236,141 237.5,141 237.5,119.5 240.8,119.5 240.8,93.6 299.3,93.6 299.3,117.4 331.3,117.4 331.3,144.2 333.7,144.2 333.7,160.2 381.8,160.2 381.8,136.3 385.4,136.3 385.4,123.2 392,123.2" fill="none" class="stroke-b" stroke-width="2.2"/>
<text x="314" y="236" class="fig-small" text-anchor="middle" >error on these rows 0.079</text>
<text x="314" y="254" class="fig-small" text-anchor="middle" >on fresh rows 0.166</text>
<text x="510" y="20" class="fig-bold" text-anchor="middle" >No depth limit</text>
<text x="510" y="38" class="fig-small" text-anchor="middle" >40 leaves</text>
<line x1="432" y1="196" x2="588" y2="196" class="stroke-muted" />
<line x1="432" y1="196" x2="432" y2="200" class="stroke-muted" />
<text x="432" y="214" class="fig-small" text-anchor="middle" >0</text>
<line x1="484" y1="196" x2="484" y2="200" class="stroke-muted" />
<text x="484" y="214" class="fig-small" text-anchor="middle" >2</text>
<line x1="536" y1="196" x2="536" y2="200" class="stroke-muted" />
<text x="536" y="214" class="fig-small" text-anchor="middle" >4</text>
<line x1="588" y1="196" x2="588" y2="200" class="stroke-muted" />
<text x="588" y="214" class="fig-small" text-anchor="middle" >6</text>
<line x1="432" y1="127" x2="588" y2="127" class="stroke-grid" />
<polyline points="432,127 434.6,122.9 437.2,118.9 439.8,115 442.4,111.2 445,107.5 447.6,104.1 450.2,100.9 452.8,97.9 455.4,95.2 458,92.8 460.6,90.8 463.2,89.2 465.8,87.9 468.4,87 471,86.5 473.6,86.4 476.2,86.8 478.8,87.5 481.4,88.6 484,90.1 486.6,92 489.2,94.2 491.8,96.7 494.4,99.6 497,102.7 499.6,106.1 502.2,109.7 504.8,113.4 507.4,117.3 510,121.3 512.6,125.3 515.2,129.4 517.8,133.4 520.4,137.4 523,141.2 525.6,145 528.2,148.5 530.8,151.8 533.4,154.9 536,157.7 538.6,160.2 541.2,162.4 543.8,164.2 546.4,165.6 549,166.7 551.6,167.3 554.2,167.6 556.8,167.4 559.4,166.9 562,165.9 564.6,164.6 567.2,162.9 569.8,160.8 572.4,158.4 575,155.6 577.6,152.6 580.2,149.4 582.8,145.9 585.4,142.2 588,138.3" fill="none" class="stroke-muted dashed" />
<circle cx="531.4" cy="167.9" r="2.6" class="fill-a" />
<circle cx="474.1" cy="68" r="2.6" class="fill-a" />
<circle cx="438.4" cy="100.7" r="2.6" class="fill-a" />
<circle cx="434.6" cy="113.5" r="2.6" class="fill-a" />
<circle cx="558.9" cy="163.8" r="2.6" class="fill-a" />
<circle cx="574.4" cy="160.1" r="2.6" class="fill-a" />
<circle cx="526.6" cy="128.6" r="2.6" class="fill-a" />
<circle cx="545.8" cy="141.5" r="2.6" class="fill-a" />
<circle cx="516.8" cy="109.9" r="2.6" class="fill-a" />
<circle cx="577.9" cy="136.3" r="2.6" class="fill-a" />
<circle cx="559.3" cy="162.6" r="2.6" class="fill-a" />
<circle cx="432.4" cy="141" r="2.6" class="fill-a" />
<circle cx="565.8" cy="163.9" r="2.6" class="fill-a" />
<circle cx="437.2" cy="110.9" r="2.6" class="fill-a" />
<circle cx="545.8" cy="181" r="2.6" class="fill-a" />
<circle cx="459.4" cy="86.9" r="2.6" class="fill-a" />
<circle cx="566.7" cy="158" r="2.6" class="fill-a" />
<circle cx="516.5" cy="122.9" r="2.6" class="fill-a" />
<circle cx="478.8" cy="101.9" r="2.6" class="fill-a" />
<circle cx="497.9" cy="112" r="2.6" class="fill-a" />
<circle cx="436.4" cy="125.5" r="2.6" class="fill-a" />
<circle cx="451.4" cy="113.7" r="2.6" class="fill-a" />
<circle cx="536.6" cy="137.2" r="2.6" class="fill-a" />
<circle cx="533" cy="160.5" r="2.6" class="fill-a" />
<circle cx="528" cy="144.2" r="2.6" class="fill-a" />
<circle cx="491.9" cy="99.9" r="2.6" class="fill-a" />
<circle cx="587.6" cy="119.7" r="2.6" class="fill-a" />
<circle cx="585" cy="126.7" r="2.6" class="fill-a" />
<circle cx="538.9" cy="152.8" r="2.6" class="fill-a" />
<circle cx="533.5" cy="181.8" r="2.6" class="fill-a" />
<circle cx="539.4" cy="160.3" r="2.6" class="fill-a" />
<circle cx="492.7" cy="89.3" r="2.6" class="fill-a" />
<circle cx="453.1" cy="85.4" r="2.6" class="fill-a" />
<circle cx="544.6" cy="172.2" r="2.6" class="fill-a" />
<circle cx="514" cy="105.2" r="2.6" class="fill-a" />
<circle cx="480.4" cy="104.2" r="2.6" class="fill-a" />
<circle cx="507.8" cy="125.9" r="2.6" class="fill-a" />
<circle cx="570.8" cy="148.5" r="2.6" class="fill-a" />
<circle cx="577.7" cy="151.9" r="2.6" class="fill-a" />
<circle cx="487.8" cy="68.6" r="2.6" class="fill-a" />
<polyline points="432,141 433.5,141 433.5,113.5 435.5,113.5 435.5,125.5 436.8,125.5 436.8,110.9 437.8,110.9 437.8,100.7 444.9,100.7 444.9,113.7 452.2,113.7 452.2,85.4 456.2,85.4 456.2,86.9 466.7,86.9 466.7,68 476.4,68 476.4,101.9 479.6,101.9 479.6,104.2 484.1,104.2 484.1,68.6 489.8,68.6 489.8,99.9 492.3,99.9 492.3,89.3 495.3,89.3 495.3,112 502.9,112 502.9,125.9 510.9,125.9 510.9,105.2 515.2,105.2 515.2,122.9 516.6,122.9 516.6,109.9 521.7,109.9 521.7,128.6 527.3,128.6 527.3,144.2 529.7,144.2 529.7,167.9 532.2,167.9 532.2,160.5 533.2,160.5 533.2,181.8 535,181.8 535,137.2 537.8,137.2 537.8,152.8 539.2,152.8 539.2,160.3 542,160.3 542,172.2 545.2,172.2 545.2,141.5 545.8,141.5 545.8,181 552.3,181 552.3,163.8 559.1,163.8 559.1,162.6 562.5,162.6 562.5,163.9 566.2,163.9 566.2,158 568.7,158 568.7,148.5 572.6,148.5 572.6,160.1 576.1,160.1 576.1,151.9 577.8,151.9 577.8,136.3 581.4,136.3 581.4,126.7 586.3,126.7 586.3,119.7 588,119.7" fill="none" class="stroke-b" stroke-width="2.2"/>
<text x="510" y="236" class="fig-small" text-anchor="middle" >error on these rows 0.000</text>
<text x="510" y="254" class="fig-small" text-anchor="middle" >on fresh rows 0.201</text>
</svg>
Figure 2. The same 40 points, three depths. Depth 1 is too stiff. Depth 3 follows the dashed sine curve and does best on fresh rows. With no limit, the tree reaches zero error on its own rows by chasing the noise, and does worse on fresh rows than depth 3.
```

A tree that has seen the answers always looks good on them. To be fair about the claim, zero training error needs rows with different labels to have different features. If two houses have identical features and different labels, no cut separates them, and the leaf stays impure.

## A fee for every leaf

Training error alone cannot choose how big a tree to keep, because a split never raises it and the biggest tree always scores best. To make a split earn its place, charge a fee $\alpha$ for every leaf. The cost of a tree $T$ is

$$
R_\alpha(T) = \sum_{\text{leaves}} n_\ell \, I_\ell \;+\; \alpha \times (\text{number of leaves}) .
$$

Read it aloud: for each leaf, multiply its rows by its impurity and add those up, which is the training error. Then add $\alpha$ for every leaf. With $\alpha = 0$ the fee vanishes and the biggest tree wins. With a very large $\alpha$ the cheapest tree is a single leaf. In between, a split survives only when it removes more than $\alpha$ of error for each leaf it adds. On the 40 noisy points, the training error falls as leaves are added, the fee rises in a straight line, and their sum has a lowest point, at 6 leaves for a fee of 0.3.

Work it out by hand on four houses with areas 1 to 4 and prices 1, 3, 8, and 10. The grown tree splits at 2.5, then splits each pair, so it has four leaves. Its cost at each size is:

| Tree | Leaves | Training error | Cost |
|---|---|---|---|
| Root only | 1 | $4 \times 13.25 = 53$ | $53 + \alpha$ |
| Split at 2.5 | 2 | $2 \times 1 + 2 \times 1 = 4$ | $4 + 2\alpha$ |
| All three splits | 4 | 0 | $4\alpha$ |

Compare a pair as two leaves ($2\alpha$ for its two leaves, with no error) against the pair as one leaf ($2 + \alpha$). The split keeps its place only when $2\alpha < 2 + \alpha$, which means $\alpha < 2$. Compare the root split against the root alone: $4 + 2\alpha < 53 + \alpha$ gives $\alpha < 49$. So as the fee rises, the two weak splits go at 2, and the strong split goes at 49. Figure 3 shows the tree at three fees.

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
Figure 3. The four-house tree at three fees. The two weak splits, each separating one pair of houses, stop paying at a fee of 2. The root split, which separates cheap from dear, keeps paying until a fee of 49. A bigger fee gives a smaller tree, and each smaller tree is inside the bigger one.
```

Both comparisons have the same shape. Collapsing a node into one leaf costs $n I + \alpha$. Keeping the subtree costs $R + \alpha L$, where $R$ is its training error and $L$ is its number of leaves. They are equal at

$$
\alpha = \frac{n I - R}{L - 1} .
$$

Read it aloud: the training error the subtree removes, divided by the extra leaves it adds. This is the *break-even fee* of a split. For the root split of the two-leaf tree it is $(53 - 4)/(2 - 1) = 49$, and for a pair it is $(2 - 0)/(2 - 1) = 2$.

## Prune from the bottom up

Apply the formula to the full four-leaf tree and you get $(53 - 0)/(4 - 1) = 17.7$, which is wrong: the root split survives until 49. By a fee of 17.7 the two weak splits are long gone, so the root's subtree has only two leaves, not four. A node must be judged against the subtree that remains after its descendants have been judged. So prune the children first, then the node. For a node:

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

### A bigger fee gives a smaller tree

In the four-house example, the tree shrinks as $\alpha$ grows and never grows back. The reason is short. Every option that keeps at least one split below a node has $L \ge 2$ leaves and costs $R + \alpha L$, against $nI + \alpha$ for collapsing it. The gap, keep minus collapse, is $(R - nI) + \alpha (L - 1)$. Because $L - 1$ is at least 1, every option's gap rises as $\alpha$ rises, and so does the smallest gap among the options. Collapsing wins when the smallest gap is zero or more, so once it wins, it wins at every larger $\alpha$. A split that goes at $\alpha$ is also gone at larger fees, and so is everything below it. The pruned trees are *nested*: each one lives inside the one before.

### Why not stop early?

You could instead refuse to split when the gain is small. The trouble is that a split can look useless on its own and open the way to a great one below it. Take 17 houses with two features, scored by Gini impurity. Put 5 houses at (low, low) and 4 at (high, high), all labelled 0. Put 4 at (low, high) and 4 at (high, low), all labelled 1. Opposite corners share a label, which is the XOR pattern, with one extra house to break the tie. The best first cut, on either feature, has a gain of only 0.0015. A rule that splits only when the gain is at least 0.01 stops at the root, and the single leaf gets 8 of the 17 houses wrong. Grown fully, the tree uses a second cut inside each half and ends with four pure leaves. At $\alpha = 1$ the four leaves cost 4, while the root alone costs $17 \times 0.498 + 1 = 9.47$, so pruning keeps all four. Pruning judges a split by what it makes possible below it, and a stopping rule judges it by its own gain alone.

## Choose the fee on held-out rows

The fee sets the tree size, but which fee? Training error cannot say, because it always prefers $\alpha = 0$. You need rows the tree never trained on, as in [cross-validation](/lessons/012-cross-validation-and-leave-one-out). Prune the big tree at many fees, score each pruned tree on held-out rows, and keep the fee with the lowest error.

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
Figure 4. Scoring each pruned tree on rows it never saw gives a U in the fee. A fee near 0.3 gives the lowest error on fresh rows, and both neighbours are worse: too small keeps the noise, and too large flattens the curve. The step at the far right is clipped at 0.4; its true height is 0.608.
```

Figure 4 does this for the 40-point tree, scoring on 1000 fresh rows. A fee of 0 keeps all 40 leaves, with error 0.201, and a fee of 0.001 keeps 37 and scores about the same. A fee between about 0.22 and 0.55 leaves 6 leaves and an error of 0.153, the lowest. A fee above about 1.6 leaves 2 leaves and an error of 0.220, and a fee above about 14.5 leaves a single leaf with error 0.608. The curve is a U, so the middle fee wins. Real data rarely offers 1000 spare rows, so use a validation split or cross-validation. Since you picked the fee on those rows, their error is slightly flattering, so keep a final test set aside for an honest number.

> [!KEY]
> A tree is one procedure applied to a box and then to each half, and it stops when a box is pure, too small, too deep, or past help. Grow it big, because a split can unlock a better one below it. Then give every leaf a fee, prune from the bottom up, and choose the fee on rows the tree never saw.

## Cost and scaling

Growing a tree costs about the same at every level. At each level every row sits in exactly one box, so the sorting in `best_split` touches each row once per feature: about $d\, n \log n$ for $d$ features. A balanced tree has about $\log_2 n$ levels, so growing it costs about $d\, n (\log n)^2$. The worst case peels off one row per cut, so it has $n$ levels and costs about $d\, n^2 \log n$. A prediction asks one question per level: about $\log_2 n$ steps in a balanced tree, and up to $n$ in a lopsided one.

## Where this breaks

- **Greedy search can miss two cuts that work together.** The search picks the cut with the largest gain at the moment. In exact XOR data every single cut leaves both sides half and half, so no cut has any gain and the tree stays a single leaf, although two cuts would separate the classes perfectly. Real data is rarely exactly XOR, so a small positive gain usually lets the tree keep growing.
- **Cuts follow one axis.** A diagonal boundary needs a staircase of many small boxes. A feature such as the sum of the two coordinates turns the staircase into a single cut.
- **A few rows can change the root.** If two cuts have nearly the same gain, moving a handful of rows decides which one wins, and everything below changes. [Averaging many trees](/lessons/024-averaging-and-bagging) steadies the answer.
- **The fee depends on the scale of the labels.** The error term is in squared units, so a fee of 0.3 means something different for prices in rupees than for prices in lakhs. Search over fees relative to the root's error, which is 21.9 for the 40 noisy points.
- **Pruning cannot add what growing missed.** It only cuts. If the grown tree never made a needed split, as in exact XOR, no fee brings it back.

> [!RECAP]
> - One cut turns a box into two smaller boxes of the same kind, so a tree is one procedure applied recursively. Every node records `n_samples`, `impurity`, and `value`, and a row goes left when its value is at most the threshold.
> - Stop when the box is pure, the depth limit is reached, no cut is possible, or no cut helps. With no depth limit and distinct rows, a tree fits its training rows exactly and chases noise.
> - Each leaf costs a fee $\alpha$, so the cost of a tree is its training error plus $\alpha$ times its leaves.
> - A split's break-even fee is $(nI - R)/(L - 1)$, and it survives while $\alpha$ is below that.
> - Prune bottom-up, children first, and collapse a node when its leaf cost is no more than its children's total. A bigger fee only removes splits, so the pruned trees are nested. Choose $\alpha$ on held-out rows.

## Build it step by step

Build the six functions in the order the checks test them. The impurity functions, `best_split`, `Node`, and `count_leaves` are already in the file.

1. **`predict_tree(tree, X)`**. Takes the root `Node` and `X` of shape `(n, d)`, and returns an array of shape `(n,)`. For each row, walk down while the node is not a leaf, going left when `row[node.feature] <= node.threshold`. The common mistake is `<`, which misroutes a row on the threshold. The output array looks like this:

```python
out = np.empty(X.shape[0])  # one answer per row
```

2. **`tree_depth(node)`**. Takes a node and returns the number of splits on the longest path to a leaf: 0 for a leaf. It is a short recursion, the same shape as `count_leaves` with a maximum in place of a sum. The common mistake is returning 1 for a leaf. The recursive case looks like this:

```python
depth = 1 + max(tree_depth(node.left), tree_depth(node.right))  # for a split
```

3. **`grow_tree(X, y, criterion, max_depth, min_samples_leaf)`**. Returns the root `Node`. Write an inner function `build(X, y, depth)` that first creates a leaf with `n_samples`, `impurity`, and `value`, then returns it when a stopping rule applies. Otherwise split with `X[:, feature] <= threshold` and recurse on both halves with `depth + 1`. The common mistake is to test the depth limit with `>` and grow one level too many. The split step looks like this:

```python
goes_left = X[:, feature] <= threshold  # boolean mask, shape (n,)
```

4. **`tree_cost(tree, alpha)`**. Takes a root `Node` and a float, and returns a float. Walk the leaves, add `n_samples * impurity` for each, then add `alpha` times the number of leaves. The common mistake is to count every node, not only the leaves. A leaf's share looks like this:

```python
leaf_share = node.n_samples * node.impurity + alpha  # one float
```

5. **`break_even_alpha(node)`**. Takes a node and returns the fee at which collapsing it costs the same as keeping its subtree as it stands. Use `tree_cost(node, 0.0)` for the training error and `count_leaves(node)` for the leaves. Return `float("inf")` for a leaf. The common mistake is dividing by the number of leaves, not by the extra leaves, `leaves - 1`. The division looks like this:

```python
alpha = (node.n_samples * node.impurity - kept_error) / (leaves - 1)
```

6. **`cost_complexity_prune(tree, alpha)`**. Returns a new root. A helper `prune(node)` returns a pruned copy of the node and its cost. For a leaf, return a copy and `n_samples * impurity + alpha`. For a split, prune both children first, and collapse when the leaf cost is at most the children's total. The common mistake is editing the nodes in place, which breaks the original tree. A collapsed node is a copy without children:

```python
leaf = Node(node.n_samples, node.impurity, node.value, is_leaf=True)
```

## The exercise

You implement `predict_tree`, `tree_depth`, `grow_tree`, `tree_cost`, `break_even_alpha`, and `cost_complexity_prune`. The impurity functions, `best_split`, `Node`, and `count_leaves` are given.

The checks walk a hand-built tree, including a row that sits exactly on a threshold. They grow the tree from this chapter's eight houses and expect the cut at 4.75, the cut at 2.0 below it, and three leaves. A tree with no depth limit gets zero training error with one leaf per row, and `max_depth` and `min_samples_leaf` are respected. The cost of the four-house trees is $4 + 2\alpha$ and $53 + \alpha$, and the break-even fees are 49, 2, and 17.7 for the root split, a pair, and the whole four-leaf tree. A tree keeps its split at $\alpha = 48$ and collapses at 49, where the costs tie. Pruning leaves the input tree alone, the pruned tree costs no more than any other subtree you could reach by collapsing, and larger fees only remove splits.
