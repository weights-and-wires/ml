---
title: "Grow a tree: recursion, stopping, and overfitting"
slug: 042-grow-and-predict
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: A tree asks the best question in a box, then repeats inside each half until a stopping rule fires, and an unlimited tree memorises its rows.
minutes: 40
gpu: false
objectives:
  - Grow a tree by applying the best split to a box and recursing into each half, and name the four stopping rules with a reason for each.
  - Predict by walking a row from the root to a leaf, and explain why a row on the threshold goes left.
  - Show that a tree with no depth limit fits its training rows exactly, and say why that is a warning and not a success.
prereqs:
  - 040-leaves-and-impurity
  - 041-gain-and-best-split
exercise: 042-grow-and-predict
---

# Grow a tree: recursion, stopping, and overfitting

You can now find the single best question for a set of rows. Take eight houses, each with an area and a distance to the metro, and a label that says whether the house sold for a high price. The best first question is "is the area at most 4.75?" It sorts the houses well, but one question is rarely enough. The left box still holds four cheap houses and one expensive one. What do you do next, and when do you stop asking?

If you stop after one question, you leave mixed boxes and give a poor answer to every house in them. If you never stop, you keep asking until every house has a box of its own, and then the tree has memorised the houses instead of learning anything. This chapter shows how a tree grows by repeating one move, how it makes predictions, and why the stopping rule decides whether the tree is useful.

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

- `best_split` finds the best single question for a box of rows, as in *Which question helps most: gain and the sorted sweep*. It returns nothing when no question is possible.
- A leaf answers with the mean of the labels in its box. A box is pure when its impurity is 0, as in *Boxes, leaves, and impurity: how bad is one answer for a box?*
- A function can call itself on a smaller copy of its problem, as long as it has a case where it stops.
- A rule's error on its own training rows is too optimistic, as in *Honest scores: training error, test error, and holding data back*.

## Ask again inside each box

Look at what one cut does. It takes a box of eight rows and produces two boxes with fewer rows each. Each new box is the same kind of problem as the first one: a set of rows, and a need for the best question. So you reuse the procedure. Apply the best question to a box, then apply the whole procedure again to the left half and to the right half. That is recursion, and Figure 1 shows the result.

Trace it on the eight houses. The root box has impurity (Gini) 0.5, and the best cut is area at most 4.75, with gain 0.3. The right box holds three houses, all expensive, so it is pure and becomes a leaf with answer 1.0. The left box holds five houses, one expensive and four cheap, with Gini 0.32. Its best cut is distance at most 2.0, and it leaves a box of one expensive house and a box of four cheap ones. Both are pure. The tree has three leaves.

Every node stores three numbers, whether it is a leaf or not: how many training rows reached it (`n_samples`), its impurity, and its `value`, the mean of its labels. A node that splits also stores the feature and the threshold. A leaf is a node with no children, and its value is the answer it gives. A node that splits is an *internal node*.

## Walk a row to a leaf

To predict for a new house, start at the root and answer each question in turn. A house with area 3.5 and distance 1.5 passes "area at most 4.75" and goes left. It passes "distance at most 2.0" and lands in the leaf with answer 1.0. A house with area 3.5 and distance 4.0 fails the second question and lands in the leaf with answer 0.0.

A row goes left when its value is at most the threshold, and right otherwise. This must match the way the tree was grown, which sent rows with `X[:, feature] <= threshold` to the left box. Using `<` instead would send a row that sits exactly on the threshold the wrong way. In code, one step of the walk is:

```python
node = node.left if row[node.feature] <= node.threshold else node.right
```

Repeat the step until the node is a leaf, and return its value. Each step asks one question, so a prediction costs one question per level of the tree.

## When to stop

A recursive procedure needs a base case, a place where it returns without recursing. A tree has four, and each one has a reason.

1. **The box is pure.** Impurity is 0 (in code, at most `1e-12`), so every row has the same label and nothing is left to separate. Any cut has gain 0.
2. **The depth limit is reached.** Depth counts the questions on the path from the root to a leaf. Each level can double the number of boxes, so a cap on depth keeps the tree small. The next section shows what happens without one.
3. **No cut is possible.** `best_split` returns `None` when the box has fewer than $2 \times$ `min_samples_leaf` rows, or when every feature is constant in the box. Rows with identical features cannot be separated, whatever their labels are.
4. **No cut helps.** The best gain is at most `1e-12`. A cut that does not make the boxes purer adds a question and buys nothing.

The check for the fourth rule uses a tiny tolerance, not exactly 0, because gains are floating-point numbers that can come out as $10^{-17}$ when they are really zero.

> [!KEY]
> Grow a tree by taking the best question, then repeating inside each half. Stop when the box is pure, the depth limit is reached, no cut is possible, or no cut helps.

## Why not grow until nothing is left?

Suppose you remove the depth limit. The tree keeps cutting until every box is pure. If all 40 training rows have different features and different labels, that means one leaf per row, and the tree predicts every training label exactly. The training error is 0.

That sounds perfect, and it is a warning. Figure 2 fits 40 noisy points from a sine curve. The noise has standard deviation 0.3, so no rule can beat a squared error of about $0.3^2 = 0.09$ on fresh rows. A tree of depth 1 has two leaves, an error of 0.185 on its own rows, and 0.220 on fresh rows. It is too stiff. A tree of depth 3 has eight leaves, 0.079 on its own rows, and 0.166 on fresh rows. A tree with no limit has 40 leaves, 0.000 on its own rows, and 0.201 on fresh rows. It chases each noisy point, so its error on new rows is worse than the depth-3 tree.

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

The training error is the wrong guide, as in *Honest scores: training error, test error, and holding data back*. A tree that has seen the answers will always look good on them. To be fair about the claim: zero training error needs rows with different labels to have different features. If two houses have identical features and different labels, no cut separates them, and the leaf stays impure. The next chapter, *Prune the tree: paying for each extra leaf*, shows how to grow a big tree and then cut it back.

## Cost and scaling

Growing a tree costs about the same at every level. At each level, every row sits in exactly one box, so the sorting in `best_split` touches each row once per feature: about $d\, n \log n$ for $d$ features. A balanced tree has about $\log_2 n$ levels, so growing it costs about $d\, n (\log n)^2$. The worst case peels off one row per cut, so it has $n$ levels and costs about $d\, n^2 \log n$. A prediction asks one question per level: about $\log_2 n$ steps in a balanced tree, and up to $n$ in a lopsided one.

## Where this breaks

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="Left: four points in an XOR pattern, where either cut leaves both sides half and half, so the gain is zero. Right: a diagonal boundary between two classes approximated by rectangles from a tree, which forms a staircase.">
<text x="30" y="22" class="fig-bold" >XOR: no first cut helps</text>
<rect x="50" y="50" width="200" height="160" class="stroke-muted" />
<line x1="150" y1="50" x2="150" y2="210" class="stroke-b" stroke-width="2.5"/>
<text x="156" y="62" class="fig-small fill-b" >cut: gain 0</text>
<circle cx="94.4" cy="174.4" r="9" class="fill-a" />
<circle cx="94.4" cy="85.6" r="9" class="fill-d" />
<circle cx="205.6" cy="174.4" r="9" class="fill-d" />
<circle cx="205.6" cy="85.6" r="9" class="fill-a" />
<text x="150" y="240" class="fig-small" text-anchor="middle" >each side holds one blue and one purple</text>
<text x="150" y="258" class="fig-small" text-anchor="middle" >so the mix does not change</text>
<text x="320" y="22" class="fig-bold" >A diagonal edge needs a staircase</text>
<rect x="340" y="138.3" width="137.6" height="71.7" class="fill-soft-a stroke-grid" />
<rect x="477.6" y="138.3" width="20.5" height="71.7" class="fill-soft-a stroke-grid" />
<rect x="340" y="103.9" width="100.4" height="34.4" class="fill-soft-a stroke-grid" />
<rect x="440.4" y="103.9" width="57.7" height="34.4" class="fill-soft-d stroke-grid" />
<rect x="498.1" y="194.7" width="61.9" height="15.3" class="fill-soft-a stroke-grid" />
<rect x="498.1" y="103.9" width="26.9" height="90.9" class="fill-soft-d stroke-grid" />
<rect x="525" y="103.9" width="35" height="90.9" class="fill-soft-d stroke-grid" />
<rect x="340" y="60.5" width="36.4" height="43.3" class="fill-soft-a stroke-grid" />
<rect x="340" y="50" width="36.4" height="10.5" class="fill-soft-d stroke-grid" />
<rect x="376.4" y="50" width="183.6" height="53.9" class="fill-soft-d stroke-grid" />
<line x1="340" y1="50" x2="560" y2="210" class="stroke-ink dashed" />
<circle cx="517.1" cy="80.7" r="2.8" class="fill-d" />
<circle cx="453.4" cy="164.3" r="2.8" class="fill-a" />
<circle cx="351.9" cy="148.7" r="2.8" class="fill-a" />
<circle cx="429.9" cy="202.8" r="2.8" class="fill-a" />
<circle cx="350.7" cy="50.1" r="2.8" class="fill-d" />
<circle cx="483.5" cy="172.5" r="2.8" class="fill-a" />
<circle cx="435.7" cy="54.1" r="2.8" class="fill-d" />
<circle cx="537.5" cy="74.9" r="2.8" class="fill-d" />
<circle cx="426.3" cy="131.1" r="2.8" class="fill-a" />
<circle cx="488.9" cy="200.3" r="2.8" class="fill-a" />
<circle cx="462.2" cy="166.6" r="2.8" class="fill-a" />
<circle cx="533.5" cy="199.7" r="2.8" class="fill-a" />
<circle cx="489.4" cy="70.8" r="2.8" class="fill-d" />
<circle cx="390" cy="66.7" r="2.8" class="fill-d" />
<circle cx="531.9" cy="207" r="2.8" class="fill-a" />
<circle cx="495.6" cy="209.8" r="2.8" class="fill-a" />
<circle cx="450.7" cy="140.1" r="2.8" class="fill-a" />
<circle cx="384.7" cy="158" r="2.8" class="fill-a" />
<circle cx="517.4" cy="159.4" r="2.8" class="fill-d" />
<circle cx="372.8" cy="98.2" r="2.8" class="fill-a" />
<circle cx="438.7" cy="82.2" r="2.8" class="fill-d" />
<circle cx="391.8" cy="158.8" r="2.8" class="fill-a" />
<circle cx="516" cy="128.9" r="2.8" class="fill-d" />
<circle cx="451.4" cy="172.2" r="2.8" class="fill-a" />
<circle cx="343.2" cy="60.7" r="2.8" class="fill-a" />
<circle cx="358.9" cy="74.8" r="2.8" class="fill-a" />
<circle cx="420.9" cy="57.8" r="2.8" class="fill-d" />
<circle cx="427.9" cy="60.2" r="2.8" class="fill-d" />
<circle cx="462.4" cy="171.6" r="2.8" class="fill-a" />
<circle cx="503.1" cy="102.1" r="2.8" class="fill-d" />
<circle cx="490.5" cy="135.8" r="2.8" class="fill-d" />
<circle cx="388.8" cy="107.4" r="2.8" class="fill-a" />
<circle cx="363.6" cy="99.2" r="2.8" class="fill-a" />
<circle cx="479.8" cy="149.8" r="2.8" class="fill-d" />
<circle cx="515.7" cy="179" r="2.8" class="fill-a" />
<circle cx="425.9" cy="82.3" r="2.8" class="fill-d" />
<circle cx="423.7" cy="95.9" r="2.8" class="fill-d" />
<circle cx="474.8" cy="59.4" r="2.8" class="fill-d" />
<circle cx="558.2" cy="94.2" r="2.8" class="fill-d" />
<circle cx="517.9" cy="185.5" r="2.8" class="fill-a" />
<circle cx="496.8" cy="74.4" r="2.8" class="fill-d" />
<circle cx="428.3" cy="121.5" r="2.8" class="fill-a" />
<circle cx="445.5" cy="56.6" r="2.8" class="fill-d" />
<circle cx="409.8" cy="145.7" r="2.8" class="fill-a" />
<circle cx="340.2" cy="142.8" r="2.8" class="fill-a" />
<circle cx="478.9" cy="60.4" r="2.8" class="fill-d" />
<circle cx="543.2" cy="157.6" r="2.8" class="fill-d" />
<circle cx="557.5" cy="180" r="2.8" class="fill-d" />
<circle cx="521.1" cy="184.8" r="2.8" class="fill-a" />
<circle cx="429.1" cy="198.2" r="2.8" class="fill-a" />
<circle cx="528.8" cy="77.4" r="2.8" class="fill-d" />
<circle cx="370.8" cy="125.7" r="2.8" class="fill-a" />
<circle cx="396.8" cy="131.3" r="2.8" class="fill-a" />
<circle cx="461.7" cy="193" r="2.8" class="fill-a" />
<circle cx="530.2" cy="165.4" r="2.8" class="fill-d" />
<circle cx="438.4" cy="200.8" r="2.8" class="fill-a" />
<circle cx="340.6" cy="178.8" r="2.8" class="fill-a" />
<circle cx="415.2" cy="61.5" r="2.8" class="fill-d" />
<circle cx="535.7" cy="133.1" r="2.8" class="fill-d" />
<circle cx="440" cy="103.3" r="2.8" class="fill-d" />
<circle cx="528.9" cy="156" r="2.8" class="fill-d" />
<circle cx="514.6" cy="146.2" r="2.8" class="fill-d" />
<circle cx="470.7" cy="92" r="2.8" class="fill-d" />
<circle cx="450.3" cy="99.5" r="2.8" class="fill-d" />
<circle cx="493.4" cy="209.1" r="2.8" class="fill-a" />
<circle cx="348.6" cy="186.2" r="2.8" class="fill-a" />
<circle cx="387" cy="177.5" r="2.8" class="fill-a" />
<circle cx="350.8" cy="175.2" r="2.8" class="fill-a" />
<circle cx="472.2" cy="68.2" r="2.8" class="fill-d" />
<circle cx="416.9" cy="151.4" r="2.8" class="fill-a" />
<circle cx="431.7" cy="101.1" r="2.8" class="fill-d" />
<circle cx="512.8" cy="59.4" r="2.8" class="fill-d" />
<circle cx="422.7" cy="96.9" r="2.8" class="fill-d" />
<circle cx="415.2" cy="78.2" r="2.8" class="fill-d" />
<circle cx="390.7" cy="70.6" r="2.8" class="fill-d" />
<circle cx="451.1" cy="93.9" r="2.8" class="fill-d" />
<circle cx="457.7" cy="159.4" r="2.8" class="fill-a" />
<circle cx="448.7" cy="201.9" r="2.8" class="fill-a" />
<circle cx="349.5" cy="125.2" r="2.8" class="fill-a" />
<circle cx="445" cy="76.7" r="2.8" class="fill-d" />
<circle cx="344.4" cy="120.9" r="2.8" class="fill-a" />
<circle cx="447.9" cy="114.3" r="2.8" class="fill-d" />
<circle cx="500.5" cy="126" r="2.8" class="fill-d" />
<circle cx="463.9" cy="131.3" r="2.8" class="fill-d" />
<circle cx="475.3" cy="170.3" r="2.8" class="fill-a" />
<circle cx="461.6" cy="102.4" r="2.8" class="fill-d" />
<circle cx="382" cy="51.5" r="2.8" class="fill-d" />
<circle cx="504.2" cy="57.1" r="2.8" class="fill-d" />
<circle cx="404.7" cy="139" r="2.8" class="fill-a" />
<circle cx="397.5" cy="202.5" r="2.8" class="fill-a" />
<rect x="340" y="50" width="220" height="160" class="stroke-muted" />
<text x="450" y="240" class="fig-small" text-anchor="middle" >depth 4 tree: 10 leaves</text>
<text x="450" y="258" class="fig-small" text-anchor="middle" >dashed line: the true boundary</text>
</svg>
Figure 3. Two ways a tree can fail. Left: in XOR, the best single cut gains nothing, so a greedy search stops, even though two cuts would separate the classes perfectly. Right: every cut follows one axis, so a diagonal edge turns into a staircase.
```

- **Greedy search can miss two cuts that work together.** The search picks the cut with the largest gain now. In XOR data, shown on the left of Figure 3, every single cut leaves both sides half and half, so no cut has any gain, and the tree stays a single leaf. Two cuts would separate the classes perfectly. Searching over all trees is far too expensive, so trees accept this weakness. Real data is rarely exactly XOR, so a small positive gain usually lets the tree keep growing.
- **Cuts follow one axis.** A diagonal boundary needs a staircase of many small boxes, as on the right of Figure 3. A new feature such as the sum of the two coordinates turns the staircase into a single cut.
- **A few rows can change the root.** If two cuts have nearly the same gain, moving a handful of rows decides which one wins, and everything below changes. Later chapters average many trees to steady the answer.

> [!RECAP]
> - One cut turns a box into two smaller boxes of the same kind, so a tree is one procedure applied recursively.
> - Every node records `n_samples`, `impurity`, and `value` (the mean). A row goes left when its value is at most the threshold.
> - Stop when the box is pure, the depth limit is reached, no cut is possible, or no cut helps.
> - With no depth limit and distinct rows, a tree fits its training rows exactly and chases noise.
> - Greedy search misses cuts that help only together, and axis cuts turn diagonals into staircases.

## Build it step by step

Build the three functions in the order the checks test them. The impurity functions, `best_split`, `Node`, and `count_leaves` are already in the file.

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

## The exercise

You implement `predict_tree`, `tree_depth`, and `grow_tree`. The impurity functions, `best_split`, `Node`, and `count_leaves` are given.

The checks walk a hand-built tree, including a row that sits exactly on a threshold. They grow the tree from this chapter's eight houses and expect the cut at 4.75, the cut at 2.0 below it, and three leaves. They confirm that every leaf predicts the mean of its rows, that a tree with no depth limit gets zero training error with one leaf per row, that `max_depth` and `min_samples_leaf` are respected, and that XOR stays a single leaf.
