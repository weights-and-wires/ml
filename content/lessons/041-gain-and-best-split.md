---
title: "Which question helps most: gain and the sorted sweep"
slug: 041-gain-and-best-split
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: A good question leaves the boxes purer, weighted by their size, and one sorted pass with running totals scores every cut at once.
minutes: 40
gpu: false
objectives:
  - Score a split by its gain, the parent impurity minus the size-weighted child impurities, and explain why the weights are needed.
  - List the thresholds worth trying on one feature, and say why they sit halfway between distinct sorted values.
  - Find the best split of a feature in one sorted pass using running sums, and explain why the variance needs only two running totals.
prereqs:
  - 040-leaves-and-impurity
exercise: 041-gain-and-best-split
---

# Which question helps most: gain and the sorted sweep

Four houses have areas 1, 2, 3, and 4 (in tens of square metres) and prices 1, 3, 8, and 10 (in tens of lakhs). You may ask one question of the form "is the area at most $t$?", and each answer sends the houses to a box with its own price guess. There are three places to put the cut: between 1 and 2, between 2 and 3, and between 3 and 4. Which one is best?

You can probably see that the middle cut is best, because it puts the two cheap houses together and the two dear ones together. The trouble is that you need a number, so a computer can compare thousands of cuts on dozens of features. Without one, a tree has no way to choose its questions. And a number has to be cheap to compute: with a thousand houses there are about a thousand cuts per feature, and rescoring each one from scratch makes the search too slow to repeat at every box of a tree.

This chapter builds the score, called gain, and the trick that scores every cut on a feature in one pass over the sorted values.

## What you already know

- A box answers with the mean of its labels, and its impurity says how badly that answer misses, as in *Boxes, leaves, and impurity: how bad is one answer for a box?*
- For numbers the impurity is the variance, the average squared gap to the mean. For 0/1 labels it is Gini, $2p(1-p)$, or entropy.
- Sorting $n$ values costs about $n \log n$ steps.
- A loop inside a loop over $n$ rows costs about $n^2$ steps.

## A good question leaves purer boxes

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Three panels, each with the four houses (area against price) and one cut. Each cut gives a two-step staircase of left and right means. The cut at 2.5 fits best, with gain 12.25.">
<text x="123" y="22" class="fig-bold" text-anchor="middle" >cut at 1.5</text>
<text x="123" y="40" class="fig-small" text-anchor="middle" >gain 6.75</text>
<line x1="50" y1="190" x2="196" y2="190" class="stroke-muted" />
<line x1="50" y1="190" x2="50" y2="52" class="stroke-muted" />
<line x1="68.2" y1="190" x2="68.2" y2="194" class="stroke-muted" />
<text x="68.2" y="208" class="fig-small" text-anchor="middle" >1</text>
<line x1="104.8" y1="190" x2="104.8" y2="194" class="stroke-muted" />
<text x="104.8" y="208" class="fig-small" text-anchor="middle" >2</text>
<line x1="141.2" y1="190" x2="141.2" y2="194" class="stroke-muted" />
<text x="141.2" y="208" class="fig-small" text-anchor="middle" >3</text>
<line x1="177.8" y1="190" x2="177.8" y2="194" class="stroke-muted" />
<text x="177.8" y="208" class="fig-small" text-anchor="middle" >4</text>
<line x1="46" y1="190" x2="50" y2="190" class="stroke-muted" />
<text x="42" y="194" class="fig-small" text-anchor="end" >0</text>
<line x1="46" y1="127.3" x2="50" y2="127.3" class="stroke-muted" />
<text x="42" y="131.3" class="fig-small" text-anchor="end" >5</text>
<line x1="46" y1="64.5" x2="50" y2="64.5" class="stroke-muted" />
<text x="42" y="68.5" class="fig-small" text-anchor="end" >10</text>
<text x="123" y="228" class="fig-small" text-anchor="middle" >area, tens of m²</text>
<line x1="50" y1="177.5" x2="86.5" y2="177.5" class="stroke-b" stroke-width="3"/>
<line x1="86.5" y1="102.2" x2="196" y2="102.2" class="stroke-b" stroke-width="3"/>
<line x1="86.5" y1="52" x2="86.5" y2="190" class="stroke-muted dashed" />
<circle cx="68.2" cy="177.5" r="5" class="fill-a" />
<circle cx="104.8" cy="152.4" r="5" class="fill-a" />
<circle cx="141.2" cy="89.6" r="5" class="fill-a" />
<circle cx="177.8" cy="64.5" r="5" class="fill-a" />
<text x="52" y="171.5" class="fig-small" >1</text>
<text x="194" y="96.2" class="fig-small" text-anchor="end" >7</text>
<text x="319" y="22" class="fig-bold" text-anchor="middle" >cut at 2.5</text>
<text x="319" y="40" class="fig-small" text-anchor="middle" >gain 12.25  (best)</text>
<line x1="246" y1="190" x2="392" y2="190" class="stroke-muted" />
<line x1="246" y1="190" x2="246" y2="52" class="stroke-muted" />
<line x1="264.2" y1="190" x2="264.2" y2="194" class="stroke-muted" />
<text x="264.2" y="208" class="fig-small" text-anchor="middle" >1</text>
<line x1="300.8" y1="190" x2="300.8" y2="194" class="stroke-muted" />
<text x="300.8" y="208" class="fig-small" text-anchor="middle" >2</text>
<line x1="337.2" y1="190" x2="337.2" y2="194" class="stroke-muted" />
<text x="337.2" y="208" class="fig-small" text-anchor="middle" >3</text>
<line x1="373.8" y1="190" x2="373.8" y2="194" class="stroke-muted" />
<text x="373.8" y="208" class="fig-small" text-anchor="middle" >4</text>
<line x1="242" y1="190" x2="246" y2="190" class="stroke-muted" />
<text x="238" y="194" class="fig-small" text-anchor="end" >0</text>
<line x1="242" y1="127.3" x2="246" y2="127.3" class="stroke-muted" />
<text x="238" y="131.3" class="fig-small" text-anchor="end" >5</text>
<line x1="242" y1="64.5" x2="246" y2="64.5" class="stroke-muted" />
<text x="238" y="68.5" class="fig-small" text-anchor="end" >10</text>
<text x="319" y="228" class="fig-small" text-anchor="middle" >area, tens of m²</text>
<line x1="246" y1="164.9" x2="319" y2="164.9" class="stroke-b" stroke-width="3"/>
<line x1="319" y1="77.1" x2="392" y2="77.1" class="stroke-b" stroke-width="3"/>
<line x1="319" y1="52" x2="319" y2="190" class="stroke-muted dashed" />
<circle cx="264.2" cy="177.5" r="5" class="fill-a" />
<circle cx="300.8" cy="152.4" r="5" class="fill-a" />
<circle cx="337.2" cy="89.6" r="5" class="fill-a" />
<circle cx="373.8" cy="64.5" r="5" class="fill-a" />
<text x="248" y="158.9" class="fig-small" >2</text>
<text x="390" y="71.1" class="fig-small" text-anchor="end" >9</text>
<text x="515" y="22" class="fig-bold" text-anchor="middle" >cut at 3.5</text>
<text x="515" y="40" class="fig-small" text-anchor="middle" >gain 6.75</text>
<line x1="442" y1="190" x2="588" y2="190" class="stroke-muted" />
<line x1="442" y1="190" x2="442" y2="52" class="stroke-muted" />
<line x1="460.2" y1="190" x2="460.2" y2="194" class="stroke-muted" />
<text x="460.2" y="208" class="fig-small" text-anchor="middle" >1</text>
<line x1="496.8" y1="190" x2="496.8" y2="194" class="stroke-muted" />
<text x="496.8" y="208" class="fig-small" text-anchor="middle" >2</text>
<line x1="533.2" y1="190" x2="533.2" y2="194" class="stroke-muted" />
<text x="533.2" y="208" class="fig-small" text-anchor="middle" >3</text>
<line x1="569.8" y1="190" x2="569.8" y2="194" class="stroke-muted" />
<text x="569.8" y="208" class="fig-small" text-anchor="middle" >4</text>
<line x1="438" y1="190" x2="442" y2="190" class="stroke-muted" />
<text x="434" y="194" class="fig-small" text-anchor="end" >0</text>
<line x1="438" y1="127.3" x2="442" y2="127.3" class="stroke-muted" />
<text x="434" y="131.3" class="fig-small" text-anchor="end" >5</text>
<line x1="438" y1="64.5" x2="442" y2="64.5" class="stroke-muted" />
<text x="434" y="68.5" class="fig-small" text-anchor="end" >10</text>
<text x="515" y="228" class="fig-small" text-anchor="middle" >area, tens of m²</text>
<line x1="442" y1="139.8" x2="551.5" y2="139.8" class="stroke-b" stroke-width="3"/>
<line x1="551.5" y1="64.5" x2="588" y2="64.5" class="stroke-b" stroke-width="3"/>
<line x1="551.5" y1="52" x2="551.5" y2="190" class="stroke-muted dashed" />
<circle cx="460.2" cy="177.5" r="5" class="fill-a" />
<circle cx="496.8" cy="152.4" r="5" class="fill-a" />
<circle cx="533.2" cy="89.6" r="5" class="fill-a" />
<circle cx="569.8" cy="64.5" r="5" class="fill-a" />
<text x="444" y="133.8" class="fig-small" >4</text>
<text x="586" y="58.5" class="fig-small" text-anchor="end" >10</text>
</svg>
Figure 1. Each panel cuts the four houses once and predicts the mean on each side (orange steps). The cut at 2.5 puts one step near each pair of houses, so the misses are smallest and the gain is largest. The two outer cuts each leave one far-off house in a mixed side.
```

Figure 1 shows the three cuts. A cut at $t$ sends every house with area at most $t$ to the left box and the rest to the right box. Each box answers with its own mean, so the model becomes a two-step staircase, drawn in orange. The cut at 2.5 puts one step near each pair of houses. The other two cuts leave a far-off house in a mixed box, so one step misses badly.

To turn the picture into a number, ask how much the impurity fell. The four prices have mean 5.5, and the squared gaps $20.25, 6.25, 6.25, 20.25$ sum to 53, so the parent variance is $53/4 = 13.25$. At the cut 2.5 the left prices 1 and 3 have variance 1, and so do the right prices 8 and 10. The impurity fell from 13.25 to 1.

The two children have two impurities, and you need one number. The obvious choice is to average them. Here is why that fails. Take ten rows, five of each class, so the Gini impurity is 0.5. Peel off one row of class 1 into its own box. That box is pure, with impurity 0, and the other nine rows hold four ones and five zeros, with impurity $2 \cdot \tfrac49 \cdot \tfrac59 = 0.494$. The plain average is 0.247, which looks like a drop of 0.253. Now compare a real split, five rows against five, with a four-to-one mix in each: each child has impurity 0.32, so the average is 0.32 and the drop is only 0.18. The plain average prefers the one-row cut, although that cut barely changed anything. One row cannot tell you much, but a pure one-row box scores perfectly.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bar chart of the gain of two splits of ten rows, scored two ways. Averaging the two child impurities rates peeling off one row at 0.253, above a real split at 0.180. Weighting by size rates them 0.056 and 0.180.">
<line x1="70" y1="210" x2="560" y2="210" class="stroke-grid" />
<text x="62" y="214" class="fig-small" text-anchor="end" >0.0</text>
<line x1="70" y1="167.1" x2="560" y2="167.1" class="stroke-grid" />
<text x="62" y="171.1" class="fig-small" text-anchor="end" >0.1</text>
<line x1="70" y1="124.3" x2="560" y2="124.3" class="stroke-grid" />
<text x="62" y="128.3" class="fig-small" text-anchor="end" >0.2</text>
<line x1="70" y1="81.4" x2="560" y2="81.4" class="stroke-grid" />
<text x="62" y="85.4" class="fig-small" text-anchor="end" >0.3</text>
<line x1="70" y1="210" x2="560" y2="210" class="stroke-muted" />
<text x="70" y="28" class="fig-small" >gain of the split (Gini)</text>
<rect x="100" y="101.5" width="70" height="108.5" class="fill-d" />
<rect x="190" y="132.9" width="70" height="77.1" class="fill-a" />
<text x="135" y="95.5" text-anchor="middle" >0.253</text>
<text x="225" y="126.9" text-anchor="middle" >0.180</text>
<text x="180" y="232" class="fig-small" text-anchor="middle" >Plain average of the two child impurities</text>
<rect x="330" y="186.2" width="70" height="23.8" class="fill-d" />
<rect x="420" y="132.9" width="70" height="77.1" class="fill-a" />
<text x="365" y="180.2" text-anchor="middle" >0.056</text>
<text x="455" y="126.9" text-anchor="middle" >0.180</text>
<text x="410" y="232" class="fig-small" text-anchor="middle" >Average weighted by child size</text>
<rect x="70" y="246" width="12" height="12" class="fill-d" />
<text x="88" y="257" class="fig-small" >peel off one row (1 row against 9)</text>
<rect x="330" y="246" width="12" height="12" class="fill-a" />
<text x="348" y="257" class="fig-small" >split 5 against 5, mostly separated</text>
<text x="180" y="74" class="fig-small fill-bad" text-anchor="middle" >winner: the one-row cut</text>
<text x="410" y="74" class="fig-small" text-anchor="middle" >winner: the real split</text>
</svg>
Figure 2. Ten rows, five of each class. Peeling off a single pure row looks great if you average the child impurities, because one child scores 0. Weighting each child by its share of the rows shows that the cut barely helps, and the real split wins.
```

The fix is to weight each child by its share of the rows, so the one-row box counts for one tenth of the answer. The peel-off then has a drop of $0.5 - 0.9 \times 0.494 = 0.056$, and the real split keeps 0.18. As a formula, with $n_L$ and $n_R$ rows on the two sides and $n = n_L + n_R$,

$$
\text{gain} = I_{\text{parent}} - \frac{n_L}{n}\, I_L - \frac{n_R}{n}\, I_R .
$$

Read it aloud: the parent's impurity, minus the impurity you still have on average once every row sits in its child box. Here $I$ is whichever impurity you chose. On the four houses, the weighted child impurity at the cut 2.5 is $\tfrac24 \cdot 1 + \tfrac24 \cdot 1 = 1$, so the gain is $13.25 - 1 = 12.25$. The other two cuts work out as follows.

| Cut | Left prices | Right prices | Left variance | Right variance | Gain |
|---|---|---|---|---|---|
| 1.5 | 1 | 3, 8, 10 | 0 | 8.667 | 6.75 |
| 2.5 | 1, 3 | 8, 10 | 1 | 1 | 12.25 |
| 3.5 | 1, 3, 8 | 10 | 8.667 | 0 | 6.75 |

In numpy, apply a boolean mask: `y[left_mask]` and `y[~left_mask]` give the two boxes, and each box's impurity comes from the previous chapter. The usual mistake is to forget the weights.

> [!KEY]
> The best question is the one with the largest gain, and the gain weights each child by its share of the rows. Without the weights, a one-row box looks like a triumph.

## Which thresholds to try

A threshold is a number $t$, and a row goes left when its value is at most $t$. Look at the sorted areas 2 and 3. Any $t$ from 2 up to, but not including, 3 sends the row with area 2 left and the row with area 3 right. Training data cannot tell these thresholds apart, because they all make the same split. So you try one, and the middle, 2.5, is the natural pick: it sits as far as possible from both neighbours, so a future house with area 2.4 joins the 2 and one with area 2.6 joins the 3.

This also answers "why not use a data value, such as 2?" It makes the same split on the training rows, but it sits on top of one house. And two rows with the same value, such as two houses of area 2, can never be separated, because every threshold sends both to the same side. So you take the midpoint between each pair of neighbours that differ, and skip the pairs that are equal.

## Score every cut in one pass

With $n$ rows there are up to $n - 1$ cuts. Scoring one cut from scratch means visiting every row to build the two boxes, about $n$ visits, so all the cuts cost about $n^2$. For the four houses that is 12 visits, which nobody notices. For 1,000 rows, it is about 999,000 visits for one feature in one box. A tree repeats this for every feature and at every box.

The way out is to sort the feature once and slide the cut from left to right. Moving the cut past one row moves that row from the right box to the left box, and nothing else changes. So you keep running totals of the left box, and each cut costs one update. Sorting costs about $n \log n$, which for 1,000 rows is about 10,000 steps, followed by one pass of about 1,000 updates. That is roughly 90 times less work than the 999,000 visits.

For 0/1 labels you only need a running count of ones, because $p$ is the count divided by the rows. For numbers, the variance seems to need the mean first, and the mean changes at every cut. But the variance can be rewritten using two running totals. Expand the squared gap:

$$
V = \frac{1}{n}\sum_i (y_i - \bar y)^2 = \frac{Q}{n} - \left(\frac{S}{n}\right)^2 ,
$$

where $S = \sum_i y_i$ is the sum of the labels and $Q = \sum_i y_i^2$ is the sum of their squares. Read it aloud: the mean of the squares minus the square of the mean. For the two prices 1 and 3, $S = 4$ and $Q = 10$, so $V = 10/2 - 2^2 = 1$, which matches. The right box needs no new work: its sums are the totals minus the left sums.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Six sorted points on a number line with candidate cuts at the midpoints between distinct neighbours, and no cut between the two equal values at 2. Under each cut, the running count, sum, and sum of squares of the labels on its left.">
<text x="30" y="22" class="fig-small" >sorted feature values, with the label y written inside each dot</text>
<line x1="40" y1="90" x2="580" y2="90" class="stroke-muted" />
<line x1="95.7" y1="90" x2="95.7" y2="94" class="stroke-muted" />
<line x1="167.1" y1="90" x2="167.1" y2="94" class="stroke-muted" />
<line x1="238.6" y1="90" x2="238.6" y2="94" class="stroke-muted" />
<line x1="310" y1="90" x2="310" y2="94" class="stroke-muted" />
<line x1="381.4" y1="90" x2="381.4" y2="94" class="stroke-muted" />
<line x1="452.9" y1="90" x2="452.9" y2="94" class="stroke-muted" />
<line x1="524.3" y1="90" x2="524.3" y2="94" class="stroke-muted" />
<circle cx="95.7" cy="68" r="10" class="fill-a" />
<text x="95.7" y="72" class="fig-small fill-paper" text-anchor="middle" >1</text>
<text x="95.7" y="110" class="fig-small" text-anchor="middle" >x = 1</text>
<circle cx="167.1" cy="68" r="10" class="fill-a" />
<text x="167.1" y="72" class="fig-small fill-paper" text-anchor="middle" >3</text>
<text x="167.1" y="110" class="fig-small" text-anchor="middle" >x = 2</text>
<circle cx="167.1" cy="46" r="10" class="fill-a" />
<text x="167.1" y="50" class="fig-small fill-paper" text-anchor="middle" >2</text>
<circle cx="310" cy="68" r="10" class="fill-a" />
<text x="310" y="72" class="fig-small fill-paper" text-anchor="middle" >8</text>
<text x="310" y="110" class="fig-small" text-anchor="middle" >x = 4</text>
<circle cx="381.4" cy="68" r="10" class="fill-a" />
<text x="381.4" y="72" class="fig-small fill-paper" text-anchor="middle" >10</text>
<text x="381.4" y="110" class="fig-small" text-anchor="middle" >x = 5</text>
<circle cx="524.3" cy="68" r="10" class="fill-a" />
<text x="524.3" y="72" class="fig-small fill-paper" text-anchor="middle" >9</text>
<text x="524.3" y="110" class="fig-small" text-anchor="middle" >x = 7</text>
<line x1="131.4" y1="40" x2="131.4" y2="128" class="stroke-b dashed" stroke-width="2"/>
<text x="131.4" y="142" class="fig-small fill-b" text-anchor="middle" >cut 1.5</text>
<text x="131.4" y="172" text-anchor="middle" >n = 1</text>
<text x="131.4" y="192" text-anchor="middle" >S = 1</text>
<text x="131.4" y="212" text-anchor="middle" >Q = 1</text>
<line x1="238.6" y1="40" x2="238.6" y2="128" class="stroke-b dashed" stroke-width="2"/>
<text x="238.6" y="142" class="fig-small fill-b" text-anchor="middle" >cut 3</text>
<text x="238.6" y="172" text-anchor="middle" >n = 3</text>
<text x="238.6" y="192" text-anchor="middle" >S = 6</text>
<text x="238.6" y="212" text-anchor="middle" >Q = 14</text>
<line x1="345.7" y1="40" x2="345.7" y2="128" class="stroke-b dashed" stroke-width="2"/>
<text x="345.7" y="142" class="fig-small fill-b" text-anchor="middle" >cut 4.5</text>
<text x="345.7" y="172" text-anchor="middle" >n = 4</text>
<text x="345.7" y="192" text-anchor="middle" >S = 14</text>
<text x="345.7" y="212" text-anchor="middle" >Q = 78</text>
<line x1="452.9" y1="40" x2="452.9" y2="128" class="stroke-b dashed" stroke-width="2"/>
<text x="452.9" y="142" class="fig-small fill-b" text-anchor="middle" >cut 6</text>
<text x="452.9" y="172" text-anchor="middle" >n = 5</text>
<text x="452.9" y="192" text-anchor="middle" >S = 24</text>
<text x="452.9" y="212" text-anchor="middle" >Q = 178</text>
<text x="167.1" y="128" class="fig-small fill-bad" text-anchor="middle" >tie: no cut</text>
<text x="300" y="250" class="fig-small" text-anchor="middle" >Moving the cut right adds the next row: n grows by 1, S by y, Q by y squared.</text>
<text x="300" y="270" class="fig-small" text-anchor="middle" >The right side is the totals (n = 6, S = 33, Q = 259) minus the left side.</text>
</svg>
Figure 3. Sorting once lines the cuts up in order. The two equal values at x = 2 give no cut, because no threshold can pull them apart. Each cut's left-side totals come from the cut before it plus one row.
```

Figure 3 shows the sweep on six sorted rows. Each cut adds the next rows to $n$, $S$, and $Q$. On the four houses, the running sums of the prices are $[1, 4, 12, 22]$ and the running sums of the squares are $[1, 10, 74, 174]$. At the cut 2.5, the left box has $n = 2$, $S = 4$, and $Q = 10$, so $V_L = 1$. The right box has $S = 22 - 4 = 18$ and $Q = 174 - 10 = 164$, so $V_R = 164/2 - 9^2 = 1$. The gain is 12.25 again, with no pass over the rows.

In numpy, the running totals are `np.cumsum`. Entry `i` of a cumulative sum covers `i + 1` rows, so the left counts are `np.arange(1, n)`, not `np.arange(n - 1)`.

> [!WARNING]
> $Q/n - (S/n)^2$ subtracts two big numbers that nearly cancel when the labels sit far from zero. For the prices $10^8 + 1$ and $10^8 + 3$, double precision returns 0.0 where the true variance is 1. Variance does not change when every label shifts by the same amount, so subtract the mean of $y$ first.

## Where this breaks

- **Many distinct values.** A feature with a different value in every row offers the most cuts, so it has the most chances to win by luck. The next two chapters limit tree size to protect against this.
- **Tiny boxes.** Under squared error, a cut that isolates one extreme row can win, because that row inflates the variance. A minimum number of rows per leaf, `min_samples_leaf`, rules these cuts out.
- **More than two classes.** The sweep keeps a count of ones, so Gini and entropy need 0/1 labels here.

> [!RECAP]
> - A split's gain is the parent impurity minus the child impurities, each weighted by its share of the rows.
> - The weights matter: a plain average makes a one-row cut look better than a real split.
> - Try one threshold, the midpoint, between each pair of distinct sorted neighbours.
> - Sort once and slide the cut, keeping running sums, so each cut costs one update instead of a pass over the rows.
> - Variance is $Q/n - (S/n)^2$, so two running totals suffice. Centre the labels first to protect the digits.

## Build it step by step

Build the three functions in the order the checks test them. The impurity functions from the previous chapter are already in the file.

1. **`split_gain(y, left_mask, criterion)`**. Takes labels of shape `(n,)`, a boolean mask of shape `(n,)`, and a criterion name, and returns a float. Compute the impurity of the whole box, then subtract the weighted impurities of `y[left_mask]` and `y[~left_mask]`. Return `0.0` when a side is empty. The common mistake is leaving out the weights. The weighted part looks like this:

```python
weighted = (n_left * imp_left + n_right * imp_right) / n  # one float
```

2. **`candidate_thresholds(x, min_samples_leaf)`**. Takes one feature of shape `(n,)` and returns the kept midpoints in increasing order. Sort with `np.sort`, take the midpoint of each neighbouring pair, and keep a pair when its two values differ and both sides keep at least `min_samples_leaf` rows. The common mistake is keeping a midpoint between equal values. The test for validity looks like this:

```python
valid = xs[:-1] != xs[1:]  # shape (n - 1,), False between equal values
```

3. **`best_split(X, y, criterion, min_samples_leaf)`**. Takes `X` of shape `(n, d)` and returns `(feature, threshold, gain)` or `None`. For each feature, sort once, get the thresholds from `candidate_thresholds`, and read the left counts off with `np.searchsorted(xs, thresholds, side="right")`. Index the cumulative sums with `n_left - 1`. The common mistake is recomputing impurities in a loop, which works but is slow. The left sums look like this:

```python
left_sum = np.cumsum(ys)[n_left - 1]  # entry k covers n_left[k] rows
```

## The exercise

You implement `split_gain`, `candidate_thresholds`, and `best_split`.

The checks confirm the numbers in this chapter: the gains 6.75, 12.25, and 6.75 for the three cuts of the four houses, and 1/18 for peeling off one row against 0.18 for a real split. They confirm that thresholds are midpoints, never data values, and that equal neighbours give none. They compare `best_split` against a slow search that rescores every cut from scratch, under squared error, Gini, and entropy, with and without `min_samples_leaf`. The last check adds $10^8$ to every label and requires the same answer, so a naive $Q/n - (S/n)^2$ fails it.
