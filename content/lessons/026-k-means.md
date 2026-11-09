---
title: "k-means: Lloyd's steps and a good start"
slug: 026-k-means
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Alternate "assign each point to its nearest centre" with "move each centre to its group's mean", and start from centres picked by squared distance.
minutes: 50
gpu: false
objectives:
  - Compute the k-means objective J, and explain why the mean is the best centre and why a Lloyd pass never raises J.
  - Assign points to their nearest centre with one matrix product, and run Lloyd's loop with the empty-group rule.
  - Explain why Lloyd's loop can stop at a worse grouping, and why uniform and always-the-farthest starts both fail.
  - Seed centres with chance proportional to squared distance, and keep the best of several restarts.
  - Say why J alone cannot choose the number of groups, and how an elbow helps.
prereqs:
  - 025-power-iteration-and-probabilistic-pca
exercise: 026-k-means
---

# k-means: Lloyd's steps and a good start

You run a delivery service. Twenty-four customers have ordered, you have the coordinates of every address, and you must open three depots. Nobody has labelled the addresses "north", "east", or "south". You only have points on a map. A sensible set of depots is one where every address is close to the depot that serves it, so the vans drive short routes.

Trying every way to split 24 addresses among three depots means $3^{24}$, about 280 billion splits, so you need a procedure that walks toward a good split.

This chapter builds that procedure, called k-means. It has a catch: run it twice from different starting depots and the answers can differ by a factor of five. So the chapter also builds the rule most libraries use to start well.

## What you already know

Five ideas are enough to start:

- A score is a single number that lets you compare two attempts ("Scoring a guess: loss and empirical risk").
- `X` has shape `(n, d)`, one row per point, and the squared distance between two points is the sum of their squared coordinate gaps.
- The mean of a group is `X[mask].mean(axis=0)`, the point where the group would balance on a pin.
- `centers[assignments]` returns, for each point, the row of its own centre.
- A probability vector is a list of non-negative numbers that add up to 1, and `rng.choice(n, p=probs)` draws an index from it.

## What clustering returns

Figure 1 shows the job. On the left are the 24 addresses as the computer sees them, a cloud with no labels. On the right is what **clustering** returns: a **group** for every point, and a **centre** for every group.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Twenty-four unlabelled points on the left, and the same points split into three groups with a cross at each group's mean on the right">
<defs></defs>
<rect x="50.0" y="40.0" width="220.0" height="209.0" class="fill-sunk stroke-grid"/>
<text x="160.0" y="26.0" class="fig-bold" text-anchor="middle" >Unlabelled addresses</text>
<rect x="330.0" y="40.0" width="220.0" height="209.0" class="fill-sunk stroke-grid"/>
<text x="440.0" y="26.0" class="fig-bold" text-anchor="middle" >Grouped, with three depots</text>
<circle cx="96.2" cy="207.3" r="5" class="fill-muted" />
<circle cx="105.3" cy="203.2" r="5" class="fill-muted" />
<circle cx="84.6" cy="198.6" r="5" class="fill-muted" />
<circle cx="117.0" cy="188.3" r="5" class="fill-muted" />
<circle cx="81.6" cy="227.3" r="5" class="fill-muted" />
<circle cx="83.0" cy="204.3" r="5" class="fill-muted" />
<circle cx="53.1" cy="208.9" r="5" class="fill-muted" />
<circle cx="72.1" cy="217.9" r="5" class="fill-muted" />
<circle cx="216.4" cy="210.6" r="5" class="fill-muted" />
<circle cx="233.2" cy="186.7" r="5" class="fill-muted" />
<circle cx="223.7" cy="181.0" r="5" class="fill-muted" />
<circle cx="214.3" cy="198.8" r="5" class="fill-muted" />
<circle cx="241.9" cy="203.3" r="5" class="fill-muted" />
<circle cx="212.9" cy="221.2" r="5" class="fill-muted" />
<circle cx="217.9" cy="201.1" r="5" class="fill-muted" />
<circle cx="208.2" cy="208.7" r="5" class="fill-muted" />
<circle cx="157.2" cy="85.5" r="5" class="fill-muted" />
<circle cx="163.8" cy="88.7" r="5" class="fill-muted" />
<circle cx="148.5" cy="97.3" r="5" class="fill-muted" />
<circle cx="173.8" cy="68.7" r="5" class="fill-muted" />
<circle cx="137.8" cy="68.4" r="5" class="fill-muted" />
<circle cx="183.7" cy="81.2" r="5" class="fill-muted" />
<circle cx="164.7" cy="100.5" r="5" class="fill-muted" />
<circle cx="185.7" cy="60.5" r="5" class="fill-muted" />
<circle cx="376.2" cy="207.3" r="5" class="fill-a" />
<circle cx="385.3" cy="203.2" r="5" class="fill-a" />
<circle cx="364.6" cy="198.6" r="5" class="fill-a" />
<circle cx="397.0" cy="188.3" r="5" class="fill-a" />
<circle cx="361.6" cy="227.3" r="5" class="fill-a" />
<circle cx="363.0" cy="204.3" r="5" class="fill-a" />
<circle cx="333.1" cy="208.9" r="5" class="fill-a" />
<circle cx="352.1" cy="217.9" r="5" class="fill-a" />
<circle cx="496.4" cy="210.6" r="5" class="fill-c" />
<circle cx="513.2" cy="186.7" r="5" class="fill-c" />
<circle cx="503.7" cy="181.0" r="5" class="fill-c" />
<circle cx="494.3" cy="198.8" r="5" class="fill-c" />
<circle cx="521.9" cy="203.3" r="5" class="fill-c" />
<circle cx="492.9" cy="221.2" r="5" class="fill-c" />
<circle cx="497.9" cy="201.1" r="5" class="fill-c" />
<circle cx="488.2" cy="208.7" r="5" class="fill-c" />
<circle cx="437.2" cy="85.5" r="5" class="fill-d" />
<circle cx="443.8" cy="88.7" r="5" class="fill-d" />
<circle cx="428.5" cy="97.3" r="5" class="fill-d" />
<circle cx="453.8" cy="68.7" r="5" class="fill-d" />
<circle cx="417.8" cy="68.4" r="5" class="fill-d" />
<circle cx="463.7" cy="81.2" r="5" class="fill-d" />
<circle cx="444.7" cy="100.5" r="5" class="fill-d" />
<circle cx="465.7" cy="60.5" r="5" class="fill-d" />
<line x1="358.6" y1="199.0" x2="374.6" y2="215.0" class="stroke-b" stroke-width="3"/><line x1="358.6" y1="215.0" x2="374.6" y2="199.0" class="stroke-b" stroke-width="3"/>
<line x1="436.4" y1="73.4" x2="452.4" y2="89.4" class="stroke-b" stroke-width="3"/><line x1="436.4" y1="89.4" x2="452.4" y2="73.4" class="stroke-b" stroke-width="3"/>
<line x1="493.1" y1="193.4" x2="509.1" y2="209.4" class="stroke-b" stroke-width="3"/><line x1="493.1" y1="209.4" x2="509.1" y2="193.4" class="stroke-b" stroke-width="3"/>
<text x="440.0" y="271.0" class="fig-small" text-anchor="middle" >orange crosses: the depots (group means)</text>
<text x="160.0" y="271.0" class="fig-small" text-anchor="middle" >no colours, no labels</text>
</svg>
Figure 1. Clustering turns the grey cloud on the left into the three coloured groups on the right. Notice that each cross sits in the middle of its own group.
```

You choose $k$, the number of groups, so three depots mean $k=3$. The method returns `assignments`, an integer from 0 to $k-1$ for each point, and `centers`, a `(k, d)` array. The crosses in Figure 1 are the centres. They are positions, not necessarily addresses.

## Score a grouping with springs

To improve a grouping you need a number that says how good it is. Picture a spring from each address to its depot. A stretched spring stores energy that grows with the square of its length, so the total energy is small only when every spring is short. That total is the **k-means objective**, called $J$.

Take four addresses on a line, at positions $0, 1, 4, 5$, and depots at $0$ and $1$. The nearest depot of each address is $0, 1, 1, 1$. The squared gaps are $0, 0, 9, 16$, and they add up to $25$.

$$
J = \sum_{i=1}^{n} \lVert x_i - c_{a_i} \rVert^2 .
$$

Read it aloud: for each point $x_i$, find its own centre $c_{a_i}$, square the distance between them, and add up. In numpy:

```python
gaps = X - centers[assignments]
J = float(np.sum(gaps * gaps))
```

Why squared gaps? Squaring punishes one long spring more than two short ones, so no address gets stranded, and it gives a clean answer for where each centre belongs. For the group $\{1, 4, 5\}$, a centre at $3$ gives squared gaps of $4+1+4=9$, and at $4$ it gives $9+0+1=10$. At the mean, $10/3$, it gives $\tfrac{49+4+25}{9}\approx 8.67$, smaller than both. That is no luck: the mean minimises the sum of squared gaps.

## Guess, then alternate

If you knew the groups, the centres would be easy: take each group's mean. If you knew the centres, the groups would be easy: send each point to its nearest centre. You know neither, and each needs the other. The way out is to guess one and let the two rules take turns. Start with any $k$ centres, then repeat two moves:

1. **Assign.** Send every point to its nearest centre.
2. **Update.** Move every centre to the mean of the points sent to it.

This loop is **Lloyd's algorithm**. Figure 2 follows it on the four points.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Four points at 0, 1, 4, 5 on a line, in three frames: depots at 0 and 1, then at 0 and 10/3, then at 0.5 and 4.5">
<defs></defs>
<text x="20.0" y="24.0" class="fig-bold" text-anchor="start" >Start: depots at 0 and 1, J = 25</text>
<line x1="33.0" y1="62.0" x2="537.0" y2="62.0" class="stroke-grid" />
<circle cx="60.0" cy="62.0" r="6" class="fill-a" />
<text x="60.0" y="50.0" class="fig-small" text-anchor="middle" >0</text>
<circle cx="150.0" cy="62.0" r="6" class="fill-a" />
<text x="150.0" y="50.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="420.0" y1="62.0" x2="150.0" y2="92.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="420.0" cy="62.0" r="6" class="fill-a" />
<text x="420.0" y="50.0" class="fig-small" text-anchor="middle" >9</text>
<line x1="510.0" y1="62.0" x2="150.0" y2="92.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="510.0" cy="62.0" r="6" class="fill-a" />
<text x="510.0" y="50.0" class="fig-small" text-anchor="middle" >16</text>
<rect x="54.0" y="86.0" width="12.0" height="12.0" class="fill-b" />
<rect x="144.0" y="86.0" width="12.0" height="12.0" class="fill-b" />
<text x="20.0" y="134.0" class="fig-bold" text-anchor="start" >After one pass: depots at 0 and 10/3, J = 4.22</text>
<line x1="33.0" y1="172.0" x2="537.0" y2="172.0" class="stroke-grid" />
<circle cx="60.0" cy="172.0" r="6" class="fill-a" />
<text x="60.0" y="160.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="150.0" y1="172.0" x2="60.0" y2="202.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="150.0" cy="172.0" r="6" class="fill-a" />
<text x="150.0" y="160.0" class="fig-small" text-anchor="middle" >1</text>
<line x1="420.0" y1="172.0" x2="360.0" y2="202.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="420.0" cy="172.0" r="6" class="fill-a" />
<text x="420.0" y="160.0" class="fig-small" text-anchor="middle" >0.44</text>
<line x1="510.0" y1="172.0" x2="360.0" y2="202.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="510.0" cy="172.0" r="6" class="fill-a" />
<text x="510.0" y="160.0" class="fig-small" text-anchor="middle" >2.78</text>
<rect x="54.0" y="196.0" width="12.0" height="12.0" class="fill-b" />
<rect x="354.0" y="196.0" width="12.0" height="12.0" class="fill-b" />
<text x="20.0" y="244.0" class="fig-bold" text-anchor="start" >After two passes: depots at 0.5 and 4.5, J = 1</text>
<line x1="33.0" y1="282.0" x2="537.0" y2="282.0" class="stroke-grid" />
<line x1="60.0" y1="282.0" x2="105.0" y2="312.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="60.0" cy="282.0" r="6" class="fill-a" />
<text x="60.0" y="270.0" class="fig-small" text-anchor="middle" >0.25</text>
<line x1="150.0" y1="282.0" x2="105.0" y2="312.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="150.0" cy="282.0" r="6" class="fill-a" />
<text x="150.0" y="270.0" class="fig-small" text-anchor="middle" >0.25</text>
<line x1="420.0" y1="282.0" x2="465.0" y2="312.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="420.0" cy="282.0" r="6" class="fill-a" />
<text x="420.0" y="270.0" class="fig-small" text-anchor="middle" >0.25</text>
<line x1="510.0" y1="282.0" x2="465.0" y2="312.0" class="stroke-muted" stroke-width="1.5"/>
<circle cx="510.0" cy="282.0" r="6" class="fill-a" />
<text x="510.0" y="270.0" class="fig-small" text-anchor="middle" >0.25</text>
<rect x="99.0" y="306.0" width="12.0" height="12.0" class="fill-b" />
<rect x="459.0" y="306.0" width="12.0" height="12.0" class="fill-b" />
<text x="580.0" y="330.0" class="fig-small" text-anchor="end" >numbers above points: squared gap to the point's depot</text>
</svg>
Figure 2. Orange squares are the depots. The depot at 1 slides to 10/3, point 1 switches to the depot at 0, and then both depots settle at the middle of their pairs. J falls from 25 to 4.22 to 1.
```

Start with depots at $0$ and $1$, so $J=25$. The update moves the depot at $1$ to the mean of $\{1,4,5\}$, which is $10/3$. The depot at $0$ owns only the point at $0$, so it stays. Now reassign: point $1$ is at squared gap $1$ from the depot at $0$ but $(7/3)^2\approx 5.44$ from the depot at $10/3$, so it switches. Then

$$
J = 0 + 1 + \left(\tfrac{2}{3}\right)^2 + \left(\tfrac{5}{3}\right)^2 = \tfrac{38}{9} \approx 4.22 .
$$

The next update gives depots at $0.5$ and $4.5$, so every squared gap is $0.25$ and $J=1$. One more pass changes nothing, so the history is $[25,\ 38/9,\ 1,\ 1]$ and you stop when $J$ repeats. (On large data, libraries stop when $J$ falls by less than a small tolerance.)

Each half of a pass can only lower $J$ or leave it alone. In the assign move, every point picks the centre with the smallest gap. In the update move, each centre goes to its group's mean, which has the smallest total squared gap of any position. $J$ stays above 0 and only a finite number of groupings exist, so the loop must stop.

> [!KEY]
> A Lloyd pass, assign then update, never raises $J$. Each move picks the best choice for what the other one fixed.

## Assign with one matrix product

A matrix product finds every nearest centre at once. Expand the squared distance:

$$
\lVert x - c \rVert^2 = \lVert x \rVert^2 - 2\, x \cdot c + \lVert c \rVert^2 .
$$

The first term is the same for every centre, so drop it and compare $\lVert c \rVert^2 - 2\, x\cdot c$. For the point $4$ and the depots $0$ and $1$ the scores are $0$ and $1-8=-7$, so depot $1$ wins, and the true squared distances, $16$ and $9$, differ by the same $7$. In numpy:

```python
scores = np.sum(centers**2, axis=1) - 2 * X @ centers.T   # shape (n, k)
assignments = np.argmin(scores, axis=1)
```

> [!WARNING]
> The scores can be negative, so use them only inside `argmin` and compute $J$ from the real gaps. If no point is nearest to a centre, the mean of zero points is `nan`, so keep the old centre.

## A bad start gets stuck

Lloyd's loop never raises $J$, but it never promised to find the lowest $J$. Run it on 60 delivery addresses twice, changing only the random starting depots. The first run ends with $J=$ 88.5. The second ends with $J=$ 439.8, about five times worse. Picture $J$ as a landscape with many valleys. The start decides which valley you stand in, and the loop stops at the bottom of that one. That bottom is a **local minimum**: no single pass can improve it, yet another valley may be much deeper. Figure 3 shows both outcomes.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The same 60 points clustered twice: a stuck run with two centres in the top cluster and one between the lower two, and a good run with one centre per cluster">
<defs></defs>
<rect x="50.0" y="40.0" width="220.0" height="209.0" class="fill-sunk stroke-grid" />
<text x="160.0" y="26.0" class="fig-bold" text-anchor="middle" >A stuck run: J = 439.8</text>
<circle cx="129.9" cy="250.0" r="4.5" class="fill-d" />
<circle cx="101.4" cy="215.0" r="4.5" class="fill-d" />
<circle cx="86.0" cy="208.8" r="4.5" class="fill-d" />
<circle cx="58.4" cy="209.1" r="4.5" class="fill-d" />
<circle cx="78.8" cy="146.5" r="4.5" class="fill-c" />
<circle cx="98.0" cy="211.2" r="4.5" class="fill-d" />
<circle cx="89.0" cy="216.8" r="4.5" class="fill-d" />
<circle cx="75.4" cy="211.9" r="4.5" class="fill-d" />
<circle cx="102.5" cy="209.2" r="4.5" class="fill-d" />
<circle cx="110.9" cy="208.5" r="4.5" class="fill-d" />
<circle cx="94.4" cy="177.8" r="4.5" class="fill-d" />
<circle cx="103.6" cy="213.9" r="4.5" class="fill-d" />
<circle cx="90.8" cy="195.5" r="4.5" class="fill-d" />
<circle cx="128.1" cy="209.7" r="4.5" class="fill-d" />
<circle cx="89.7" cy="187.4" r="4.5" class="fill-d" />
<circle cx="78.4" cy="210.1" r="4.5" class="fill-d" />
<circle cx="109.5" cy="194.8" r="4.5" class="fill-d" />
<circle cx="95.6" cy="193.2" r="4.5" class="fill-d" />
<circle cx="44.2" cy="187.0" r="4.5" class="fill-d" />
<circle cx="77.1" cy="234.4" r="4.5" class="fill-d" />
<circle cx="230.9" cy="192.7" r="4.5" class="fill-d" />
<circle cx="218.2" cy="223.9" r="4.5" class="fill-d" />
<circle cx="226.5" cy="205.9" r="4.5" class="fill-d" />
<circle cx="250.7" cy="191.8" r="4.5" class="fill-d" />
<circle cx="229.4" cy="185.4" r="4.5" class="fill-d" />
<circle cx="222.4" cy="221.3" r="4.5" class="fill-d" />
<circle cx="236.3" cy="194.7" r="4.5" class="fill-d" />
<circle cx="222.2" cy="218.8" r="4.5" class="fill-d" />
<circle cx="230.0" cy="248.9" r="4.5" class="fill-d" />
<circle cx="238.1" cy="196.4" r="4.5" class="fill-d" />
<circle cx="197.2" cy="203.9" r="4.5" class="fill-d" />
<circle cx="209.0" cy="191.7" r="4.5" class="fill-d" />
<circle cx="190.2" cy="221.1" r="4.5" class="fill-d" />
<circle cx="238.5" cy="184.6" r="4.5" class="fill-d" />
<circle cx="188.0" cy="213.8" r="4.5" class="fill-d" />
<circle cx="231.8" cy="215.7" r="4.5" class="fill-d" />
<circle cx="254.0" cy="226.0" r="4.5" class="fill-d" />
<circle cx="232.2" cy="223.5" r="4.5" class="fill-d" />
<circle cx="250.7" cy="205.4" r="4.5" class="fill-d" />
<circle cx="219.4" cy="235.2" r="4.5" class="fill-d" />
<circle cx="189.6" cy="81.8" r="4.5" class="fill-a" />
<circle cx="173.3" cy="75.0" r="4.5" class="fill-a" />
<circle cx="166.1" cy="106.3" r="4.5" class="fill-c" />
<circle cx="145.9" cy="109.1" r="4.5" class="fill-c" />
<circle cx="184.1" cy="120.7" r="4.5" class="fill-c" />
<circle cx="149.5" cy="100.7" r="4.5" class="fill-c" />
<circle cx="164.0" cy="84.9" r="4.5" class="fill-a" />
<circle cx="138.0" cy="125.4" r="4.5" class="fill-c" />
<circle cx="159.9" cy="73.6" r="4.5" class="fill-a" />
<circle cx="173.3" cy="91.2" r="4.5" class="fill-a" />
<circle cx="154.4" cy="89.8" r="4.5" class="fill-a" />
<circle cx="155.7" cy="80.6" r="4.5" class="fill-a" />
<circle cx="146.0" cy="92.6" r="4.5" class="fill-a" />
<circle cx="158.1" cy="85.4" r="4.5" class="fill-a" />
<circle cx="164.0" cy="50.1" r="4.5" class="fill-a" />
<circle cx="186.4" cy="68.7" r="4.5" class="fill-a" />
<circle cx="124.1" cy="101.0" r="4.5" class="fill-c" />
<circle cx="149.3" cy="85.6" r="4.5" class="fill-a" />
<circle cx="119.9" cy="74.3" r="4.5" class="fill-a" />
<circle cx="178.8" cy="117.9" r="4.5" class="fill-c" />
<line x1="153.1" y1="71.5" x2="169.1" y2="87.5" class="stroke-b" stroke-width="3"/><line x1="153.1" y1="87.5" x2="169.1" y2="71.5" class="stroke-b" stroke-width="3"/>
<line x1="137.7" y1="107.9" x2="153.7" y2="123.9" class="stroke-b" stroke-width="3"/><line x1="137.7" y1="123.9" x2="153.7" y2="107.9" class="stroke-b" stroke-width="3"/>
<line x1="153.0" y1="200.8" x2="169.0" y2="216.8" class="stroke-b" stroke-width="3"/><line x1="153.0" y1="216.8" x2="169.0" y2="200.8" class="stroke-b" stroke-width="3"/>
<rect x="330.0" y="40.0" width="220.0" height="209.0" class="fill-sunk stroke-grid" />
<text x="440.0" y="26.0" class="fig-bold" text-anchor="middle" >A good run: J = 88.5</text>
<circle cx="409.9" cy="250.0" r="4.5" class="fill-d" />
<circle cx="381.4" cy="215.0" r="4.5" class="fill-d" />
<circle cx="366.0" cy="208.8" r="4.5" class="fill-d" />
<circle cx="338.4" cy="209.1" r="4.5" class="fill-d" />
<circle cx="358.8" cy="146.5" r="4.5" class="fill-d" />
<circle cx="378.0" cy="211.2" r="4.5" class="fill-d" />
<circle cx="369.0" cy="216.8" r="4.5" class="fill-d" />
<circle cx="355.4" cy="211.9" r="4.5" class="fill-d" />
<circle cx="382.5" cy="209.2" r="4.5" class="fill-d" />
<circle cx="390.9" cy="208.5" r="4.5" class="fill-d" />
<circle cx="374.4" cy="177.8" r="4.5" class="fill-d" />
<circle cx="383.6" cy="213.9" r="4.5" class="fill-d" />
<circle cx="370.8" cy="195.5" r="4.5" class="fill-d" />
<circle cx="408.1" cy="209.7" r="4.5" class="fill-d" />
<circle cx="369.7" cy="187.4" r="4.5" class="fill-d" />
<circle cx="358.4" cy="210.1" r="4.5" class="fill-d" />
<circle cx="389.5" cy="194.8" r="4.5" class="fill-d" />
<circle cx="375.6" cy="193.2" r="4.5" class="fill-d" />
<circle cx="324.2" cy="187.0" r="4.5" class="fill-d" />
<circle cx="357.1" cy="234.4" r="4.5" class="fill-d" />
<circle cx="510.9" cy="192.7" r="4.5" class="fill-c" />
<circle cx="498.2" cy="223.9" r="4.5" class="fill-c" />
<circle cx="506.5" cy="205.9" r="4.5" class="fill-c" />
<circle cx="530.7" cy="191.8" r="4.5" class="fill-c" />
<circle cx="509.4" cy="185.4" r="4.5" class="fill-c" />
<circle cx="502.4" cy="221.3" r="4.5" class="fill-c" />
<circle cx="516.3" cy="194.7" r="4.5" class="fill-c" />
<circle cx="502.2" cy="218.8" r="4.5" class="fill-c" />
<circle cx="510.0" cy="248.9" r="4.5" class="fill-c" />
<circle cx="518.1" cy="196.4" r="4.5" class="fill-c" />
<circle cx="477.2" cy="203.9" r="4.5" class="fill-c" />
<circle cx="489.0" cy="191.7" r="4.5" class="fill-c" />
<circle cx="470.2" cy="221.1" r="4.5" class="fill-c" />
<circle cx="518.5" cy="184.6" r="4.5" class="fill-c" />
<circle cx="468.0" cy="213.8" r="4.5" class="fill-c" />
<circle cx="511.8" cy="215.7" r="4.5" class="fill-c" />
<circle cx="534.0" cy="226.0" r="4.5" class="fill-c" />
<circle cx="512.2" cy="223.5" r="4.5" class="fill-c" />
<circle cx="530.7" cy="205.4" r="4.5" class="fill-c" />
<circle cx="499.4" cy="235.2" r="4.5" class="fill-c" />
<circle cx="469.6" cy="81.8" r="4.5" class="fill-a" />
<circle cx="453.3" cy="75.0" r="4.5" class="fill-a" />
<circle cx="446.1" cy="106.3" r="4.5" class="fill-a" />
<circle cx="425.9" cy="109.1" r="4.5" class="fill-a" />
<circle cx="464.1" cy="120.7" r="4.5" class="fill-a" />
<circle cx="429.5" cy="100.7" r="4.5" class="fill-a" />
<circle cx="444.0" cy="84.9" r="4.5" class="fill-a" />
<circle cx="418.0" cy="125.4" r="4.5" class="fill-a" />
<circle cx="439.9" cy="73.6" r="4.5" class="fill-a" />
<circle cx="453.3" cy="91.2" r="4.5" class="fill-a" />
<circle cx="434.4" cy="89.8" r="4.5" class="fill-a" />
<circle cx="435.7" cy="80.6" r="4.5" class="fill-a" />
<circle cx="426.0" cy="92.6" r="4.5" class="fill-a" />
<circle cx="438.1" cy="85.4" r="4.5" class="fill-a" />
<circle cx="444.0" cy="50.1" r="4.5" class="fill-a" />
<circle cx="466.4" cy="68.7" r="4.5" class="fill-a" />
<circle cx="404.1" cy="101.0" r="4.5" class="fill-a" />
<circle cx="429.3" cy="85.6" r="4.5" class="fill-a" />
<circle cx="399.9" cy="74.3" r="4.5" class="fill-a" />
<circle cx="458.8" cy="117.9" r="4.5" class="fill-a" />
<line x1="431.0" y1="82.7" x2="447.0" y2="98.7" class="stroke-b" stroke-width="3"/><line x1="431.0" y1="98.7" x2="447.0" y2="82.7" class="stroke-b" stroke-width="3"/>
<line x1="497.8" y1="202.0" x2="513.8" y2="218.0" class="stroke-b" stroke-width="3"/><line x1="497.8" y1="218.0" x2="513.8" y2="202.0" class="stroke-b" stroke-width="3"/>
<line x1="364.1" y1="196.5" x2="380.1" y2="212.5" class="stroke-b" stroke-width="3"/><line x1="364.1" y1="212.5" x2="380.1" y2="196.5" class="stroke-b" stroke-width="3"/>
<text x="160.0" y="271.0" class="fig-small" text-anchor="middle" >two centres share a cluster; one sits between two</text>
<text x="440.0" y="271.0" class="fig-small" text-anchor="middle" >one centre in each cluster</text>
</svg>
Figure 3. Both runs use the same data and the same Lloyd loop, and each stopped because J stopped falling. The left run reached a valley bottom that is not the lowest, with J = 439.8, about five times the good run.
```

In the stuck run, two depots started in the top cluster, so it got split in two, and the other two clusters were merged under one depot halfway between them. Moving a top depot down would cut $J$ sharply, but Lloyd never tries it, because it only moves a depot to the mean of the points it already owns.

How often does it happen? Over 2000 runs from three random data points, about 8 percent got stuck like Figure 3. From the seeding built below it was about 1 percent, and the average final $J$ fell from 116 to 92.

### Why not pick the starts uniformly, or the farthest points?

Pick three data points at random, and two can land in the same cluster. With 20 points in each of three clusters, the chance that the three picks land in three different clusters is $\frac{6 \cdot 20 \cdot 20 \cdot 20}{60 \cdot 59 \cdot 58} \approx 0.23$. Barely one start in four has a depot in each cluster.

Alternatively, pick the first depot at random, then always take the point farthest from every depot so far. That spreads the depots out, but one stray address far from everybody, say a typo in the coordinates, is the farthest point every time, so it steals a depot. You want a rule between the two: far points much more likely, but never certain.

### Squared distance as a chance

Let $D(x)$ be the distance from a point $x$ to its nearest centre chosen so far. The **k-means++** rule picks the first centre uniformly, and each new centre from the data points with probability proportional to $D(x)^2$:

$$
p(x_i) = \frac{D(x_i)^2}{\sum_{j=1}^{n} D(x_j)^2} .
$$

Read it aloud: a point's chance is its squared distance over the total of all squared distances. A point already chosen has $D=0$, so it can never be chosen twice.

Try it on the points $0, 1, 10$ with the first centre at $0$. The squared distances $0, 1, 100$ add to $101$, so the chances are $0,\ 1/101,\ 100/101$. The far point wins about 99 times in 100, but the point at $1$ keeps a small chance. Plain distances would give the near point 9 percent, so squaring is what makes the far points dominate.

In numpy, keep one array of squared distances to the nearest chosen centre. After each pick, a point's distance can only shrink, so update it with `np.minimum`:

```python
chosen = rng.choice(n, p=sq_dist / sq_dist.sum())
sq_dist = np.minimum(sq_dist, np.sum((X - X[chosen]) ** 2, axis=1))
```

If every point already sits on a centre, `sq_dist.sum()` is 0 and the division gives `nan`. Fall back to equal chances.

Figure 4 shows the rule on three clusters of 10 points plus one stray point. The circle on each point has area proportional to its squared distance to the nearest chosen centre, which is its chance of being picked.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Two panels of three clusters and one stray point, with a circle on each point whose area is its squared distance to the nearest chosen centre">
<defs></defs>
<rect x="20.0" y="40.0" width="270.0" height="230.0" class="fill-sunk stroke-grid" />
<text x="155.0" y="26.0" class="fig-bold" text-anchor="middle" >After the first centre</text>
<circle cx="60.5" cy="211.0" r="4.5" class="fill-soft-a stroke-a" />
<circle cx="77.1" cy="237.1" r="3.1" class="fill-soft-a stroke-a" />
<circle cx="55.8" cy="237.4" r="3.3" class="fill-soft-a stroke-a" />
<circle cx="68.0" cy="230.8" r="3.4" class="fill-soft-a stroke-a" />
<circle cx="70.5" cy="255.9" r="2.0" class="fill-a" />
<circle cx="81.9" cy="231.4" r="3.5" class="fill-soft-a stroke-a" />
<circle cx="69.5" cy="231.9" r="3.3" class="fill-soft-a stroke-a" />
<circle cx="54.7" cy="223.5" r="3.9" class="fill-soft-a stroke-a" />
<circle cx="71.5" cy="232.8" r="3.2" class="fill-soft-a stroke-a" />
<circle cx="57.9" cy="220.4" r="4.0" class="fill-soft-a stroke-a" />
<circle cx="167.8" cy="251.2" r="7.3" class="fill-soft-a stroke-a" />
<circle cx="185.5" cy="239.4" r="8.3" class="fill-soft-a stroke-a" />
<circle cx="153.1" cy="241.4" r="6.5" class="fill-soft-a stroke-a" />
<circle cx="173.5" cy="246.7" r="7.6" class="fill-soft-a stroke-a" />
<circle cx="159.1" cy="229.5" r="7.0" class="fill-soft-a stroke-a" />
<circle cx="192.6" cy="233.3" r="8.7" class="fill-soft-a stroke-a" />
<circle cx="169.6" cy="224.6" r="7.6" class="fill-soft-a stroke-a" />
<circle cx="190.0" cy="234.2" r="8.6" class="fill-soft-a stroke-a" />
<circle cx="187.6" cy="215.4" r="8.7" class="fill-soft-a stroke-a" />
<circle cx="177.1" cy="241.4" r="7.8" class="fill-soft-a stroke-a" />
<circle cx="124.9" cy="126.5" r="9.6" class="fill-soft-a stroke-a" />
<circle cx="135.4" cy="148.0" r="8.8" class="fill-soft-a stroke-a" />
<circle cx="110.7" cy="141.7" r="8.6" class="fill-soft-a stroke-a" />
<circle cx="95.7" cy="128.2" r="9.0" class="fill-soft-a stroke-a" />
<circle cx="127.4" cy="140.3" r="9.0" class="fill-soft-a stroke-a" />
<circle cx="139.4" cy="118.5" r="10.3" class="fill-soft-a stroke-a" />
<circle cx="128.8" cy="124.4" r="9.8" class="fill-soft-a stroke-a" />
<circle cx="133.4" cy="148.6" r="8.7" class="fill-soft-a stroke-a" />
<circle cx="128.6" cy="121.6" r="9.9" class="fill-soft-a stroke-a" />
<circle cx="95.3" cy="125.1" r="9.2" class="fill-soft-a stroke-a" />
<circle cx="250.0" cy="70.0" r="16.0" class="fill-soft-a stroke-a" />
<line x1="61.5" y1="246.9" x2="79.5" y2="264.9" class="stroke-b" stroke-width="3"/><line x1="61.5" y1="264.9" x2="79.5" y2="246.9" class="stroke-b" stroke-width="3"/>
<rect x="310.0" y="40.0" width="270.0" height="230.0" class="fill-sunk stroke-grid" />
<text x="445.0" y="26.0" class="fig-bold" text-anchor="middle" >After the second centre</text>
<circle cx="350.5" cy="211.0" r="5.3" class="fill-soft-a stroke-a" />
<circle cx="367.1" cy="237.1" r="3.4" class="fill-soft-a stroke-a" />
<circle cx="345.8" cy="237.4" r="3.7" class="fill-soft-a stroke-a" />
<circle cx="358.0" cy="230.8" r="3.8" class="fill-soft-a stroke-a" />
<circle cx="360.5" cy="255.9" r="2.0" class="fill-a" />
<circle cx="371.9" cy="231.4" r="3.9" class="fill-soft-a stroke-a" />
<circle cx="359.5" cy="231.9" r="3.7" class="fill-soft-a stroke-a" />
<circle cx="344.7" cy="223.5" r="4.6" class="fill-soft-a stroke-a" />
<circle cx="361.5" cy="232.8" r="3.6" class="fill-soft-a stroke-a" />
<circle cx="347.9" cy="220.4" r="4.7" class="fill-soft-a stroke-a" />
<circle cx="457.8" cy="251.2" r="3.3" class="fill-soft-a stroke-a" />
<circle cx="475.5" cy="239.4" r="4.3" class="fill-soft-a stroke-a" />
<circle cx="443.1" cy="241.4" r="2.0" class="fill-a" />
<circle cx="463.5" cy="246.7" r="3.5" class="fill-soft-a stroke-a" />
<circle cx="449.1" cy="229.5" r="2.9" class="fill-soft-a stroke-a" />
<circle cx="482.6" cy="233.3" r="4.9" class="fill-soft-a stroke-a" />
<circle cx="459.6" cy="224.6" r="3.7" class="fill-soft-a stroke-a" />
<circle cx="480.0" cy="234.2" r="4.7" class="fill-soft-a stroke-a" />
<circle cx="477.6" cy="215.4" r="5.1" class="fill-soft-a stroke-a" />
<circle cx="467.1" cy="241.4" r="3.7" class="fill-soft-a stroke-a" />
<circle cx="414.9" cy="126.5" r="10.4" class="fill-soft-a stroke-a" />
<circle cx="425.4" cy="148.0" r="8.8" class="fill-soft-a stroke-a" />
<circle cx="400.7" cy="141.7" r="9.7" class="fill-soft-a stroke-a" />
<circle cx="385.7" cy="128.2" r="11.0" class="fill-soft-a stroke-a" />
<circle cx="417.4" cy="140.3" r="9.4" class="fill-soft-a stroke-a" />
<circle cx="429.4" cy="118.5" r="10.8" class="fill-soft-a stroke-a" />
<circle cx="418.8" cy="124.4" r="10.5" class="fill-soft-a stroke-a" />
<circle cx="423.4" cy="148.6" r="8.7" class="fill-soft-a stroke-a" />
<circle cx="418.6" cy="121.6" r="10.7" class="fill-soft-a stroke-a" />
<circle cx="385.3" cy="125.1" r="11.2" class="fill-soft-a stroke-a" />
<circle cx="540.0" cy="70.0" r="16.0" class="fill-soft-a stroke-a" />
<line x1="351.5" y1="246.9" x2="369.5" y2="264.9" class="stroke-b" stroke-width="3"/><line x1="351.5" y1="264.9" x2="369.5" y2="246.9" class="stroke-b" stroke-width="3"/>
<line x1="434.1" y1="232.4" x2="452.1" y2="250.4" class="stroke-b" stroke-width="3"/><line x1="434.1" y1="250.4" x2="452.1" y2="232.4" class="stroke-b" stroke-width="3"/>
<text x="155.0" y="292.0" class="fig-small" text-anchor="middle" >circle area is proportional to squared distance</text>
<text x="445.0" y="292.0" class="fig-small" text-anchor="middle" >the second pick shrinks the circles near it</text>
<text x="250.0" y="98.0" class="fig-small" text-anchor="middle" >stray point</text>
<text x="540.0" y="98.0" class="fig-small" text-anchor="middle" >stray point</text>
</svg>
Figure 4. The big circles are the points most likely to be picked. After the first centre, the far clusters and the stray point have the large circles. After the second centre, the circles near it collapse, so a third centre rarely lands in a cluster that already has one.
```

After the first centre in the lower-left cluster, the second cluster holds 31 percent of the total, the third cluster 49 percent, the stray point 18 percent, and the first cluster only 2 percent. The stray point has a real chance, but under one in five. Once the second pick lands in the second cluster, its circles collapse, and a third centre rarely lands in a cluster that already has one.

> [!KEY]
> Pick each new centre with probability proportional to its squared distance from the centres already chosen. Far points are favoured, so depots spread out, and no single point is certain, so one stray point cannot take over.

## Restarts, and choosing k

Seeding makes a bad start unlikely, not impossible, and a run is cheap. So seed and run Lloyd to the end `n_init` times, and keep the lowest $J$.

Restarts also make the choice of $k$ trustworthy. If $k$ is as large as $n$, every address gets its own depot and $J=0$, which tells you nothing, so you fix $k$ first. To choose it, plot the best $J$ for each $k$ and look for the elbow, where adding a depot stops buying much. On the 24 addresses, $J$ falls from 336 at one depot to 165 at two and 18.9 at three, then only to 14.6 at four, so the elbow is at three. Each point is the best of 20 seeded runs, because a stuck run that sneaks in sits too high and hides the elbow.

> [!RECAP]
> - $J$ is the sum of squared gaps from each point to its own centre. The mean of a group has the smallest total squared gap, so it is the best centre.
> - Assign sends each point to its nearest centre with $\lVert c\rVert^2 - 2x\cdot c$ and `argmin`. Update moves each centre to its group's mean, and an empty group keeps its old centre. $J$ never rises.
> - Lloyd's loop stops at the bottom of the valley it starts in. Uniform starts often double up, and the farthest point is always a stray point.
> - k-means++ picks each centre with probability $D^2$ over the sum of $D^2$, with the first centre uniform. Keep the lowest-$J$ run of several restarts.

## Build it step by step

You write six functions, in the order the checks test them.

1. **`kmeans_objective(X, centers, assignments)`** returns $J$ as a Python float. Pair each point with its centre via `centers[assignments]`, square the gaps, and add them all. The usual mistake is taking the mean.
2. **`assign_nearest(X, centers)`** returns an integer array of shape `(n,)`. Build the `(n, k)` score matrix with the matrix product and take `argmin` over axis 1. The usual mistake is `argmin` over axis 0, which returns the nearest point to each centre.
3. **`lloyd(X, k, init, iters)`** copies `init`, assigns, and records $J$. Then it loops up to `iters` times: move every centre whose group has a point to that group's mean, reassign, append $J$, and stop when $J$ repeats. It returns `(centers, assignments, history)`. The usual mistake is writing into `init`.
4. **`seeding_probabilities(sq_dist)`** divides the `(n,)` array by its sum, or returns `np.full(n, 1 / n)` when the sum is 0. The usual mistake is skipping the zero case, which gives `nan` when every point is a duplicate.
5. **`kmeans_plusplus_init(X, k, rng)`** returns a `(k, d)` array of rows of `X`. Draw the first index with `rng.integers(n)`, then draw each further centre with `rng.choice(n, p=seeding_probabilities(sq_dist))` and shrink `sq_dist` with `np.minimum`. The usual mistake is using plain distances.
6. **`best_of_restarts(X, k, n_init, rng)`** seeds, runs `lloyd(X, k, init, 100)`, `n_init` times, and returns `(centers, assignments, J)` for the run whose last history entry is smallest. The usual mistake is reseeding `rng` inside the loop, so every restart repeats the same start.

## Where this breaks

- **Units matter.** If one column is in kilometres and another in grams, the large numbers dominate every distance. Standardise each column first.
- **It prefers round lumps of similar size.** The border between two groups is the line halfway between their centres, so long thin groups get cut in the middle.
- **Seeding can still pick a stray point,** as in Figure 4, so use restarts. Ten restarts take about ten times as long, so stop when the best $J$ stops improving.

## The exercise

You build `kmeans_objective`, `assign_nearest`, `lloyd`, `seeding_probabilities`, `kmeans_plusplus_init`, and `best_of_restarts`. The checks reproduce the chapter's numbers: $J=25$, the history $[25, 38/9, 1, 1]$, and the chances $0, 1/101, 100/101$. They compare `assign_nearest` with a brute-force search, confirm that $J$ never rises and that an empty group keeps its centre, and draw seeds thousands of times to confirm the far point appears about 99 percent of the time. They also confirm that seeding beats uniform picks at covering three clusters, and that the best of several restarts finds the good grouping.
