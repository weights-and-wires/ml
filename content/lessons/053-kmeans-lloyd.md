---
title: "k-means: group points by moving centres"
slug: 053-kmeans-lloyd
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Alternate "assign each point to its nearest centre" with "move each centre to its group's mean", and the total squared distance falls on every pass.
minutes: 40
gpu: false
objectives:
  - Compute the k-means objective J and explain it as the energy of springs from each point to its centre.
  - Explain why the mean is the best centre for a group, and why neither half of a Lloyd pass can raise J.
  - Assign points to their nearest centre with one matrix product, and run Lloyd's loop with the empty-group rule.
  - Say why J alone cannot choose the number of groups, and how an elbow helps.
prereqs:
  - 052-probabilistic-pca-em
exercise: 053-kmeans-lloyd
---

# k-means: group points by moving centres

You run a delivery service. Twenty-four customers have ordered, you have the coordinates of every address, and you must open three depots. Nobody has labelled the addresses "north", "east", or "south". You only have points on a map. A sensible set of depots is one where every address is close to the depot that serves it, so the vans drive short routes.

You could try every way to split 24 addresses among three depots. That is $3^{24}$, about 280 billion splits, and with 100 addresses it is about $5\times 10^{47}$. No computer checks that many, so you need a procedure that walks toward a good split instead of searching for it.

This chapter builds that procedure, called k-means. It is short, it is the first thing people try on unlabelled data, and the idea behind it, guess and then improve the guess in two alternating moves, comes back in the chapters that follow.

## What you already know

Five ideas are enough to start:

- A score is a single number that lets you compare two attempts (see "Scoring a guess: loss and empirical risk").
- `X` has shape `(n, d)`, one row per point.
- The squared distance between two points is the sum of their squared coordinate gaps.
- The mean of a group is `X[mask].mean(axis=0)`, the point where the group would balance on a pin.
- `centers[assignments]` returns, for each point, the row of its own centre, because an integer array picks rows.

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

You choose $k$, the number of groups. Your three depots mean $k=3$. The method returns `assignments`, an integer from 0 to $k-1$ for each point, and `centers`, a `(k, d)` array. The crosses in Figure 1 are the centres. They do not have to be addresses. They are positions.

## Score a grouping with springs

Before you improve a grouping, you need a number that says how good it is. Picture a spring from each address to its depot. A stretched spring stores energy that grows with the square of its length, so the total energy of all springs is small only when every spring is short. That total is the **k-means objective**, called $J$.

Take four addresses on a line, at positions $0, 1, 4, 5$, and depots at $0$ and $1$. The nearest depot of each address is $0, 1, 1, 1$. The squared gaps are $0, 0, 9, 16$, and they add up to $25$.

$$
J = \sum_{i=1}^{n} \lVert x_i - c_{a_i} \rVert^2 .
$$

Read it aloud: for each point $x_i$, find its own centre $c_{a_i}$ (the centre whose number is $a_i$), square the distance between them, and add up. In numpy it is two lines:

```python
gaps = X - centers[assignments]
J = float(np.sum(gaps * gaps))
```

### Why squared, and why to a centre

You might ask why the gaps are squared, and why they go to a centre rather than to the other points. Squaring punishes one long spring more than two short ones, so no address gets stranded. It also gives a clean answer for where each centre belongs.

Take the group $\{1, 4, 5\}$ and try three positions for its centre. At $3$ the squared gaps are $4+1+4=9$. At $4$ they are $9+0+1=10$. At the mean, $10/3$, they are $\tfrac{49+4+25}{9}=\tfrac{78}{9}\approx 8.67$, which is smaller than both. That is not luck. The mean is the position that minimises the sum of squared gaps, which is why people call it the balance point. Measure gaps without squaring and the best centre becomes the median instead, which is a different method with its own name, k-medians.

### Why not choose $k$ to make $J$ small?

If you let $k$ be as large as $n$, every address gets a depot of its own and $J=0$. The score is perfect and tells you nothing. So you fix $k$ first and minimise $J$ for that $k$. How to choose $k$ is at the end of the chapter.

## Guess, then alternate

Here is the difficulty. If you knew the groups, the centres would be easy: take each group's mean. If you knew the centres, the groups would be easy: send each point to its nearest centre. You know neither, and each answer needs the other.

The way out is to guess one and let the two rules take turns. Start with any $k$ centres. Then repeat two moves:

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

Start with depots at $0$ and $1$, so $J=25$. The update moves the depot at $1$ to the mean of $\{1,4,5\}$, which is $10/3$. The depot at $0$ owns only the point at $0$, so it stays. Now reassign. Point $1$ is at gap $1$ from the depot at $0$, but at gap $(7/3)^2\approx 5.44$ from the depot at $10/3$, so it switches. Then

$$
J = 0 + 1 + \left(\tfrac{2}{3}\right)^2 + \left(\tfrac{5}{3}\right)^2 = \tfrac{38}{9} \approx 4.22 .
$$

The next update gives depots at $0.5$ and $4.5$, the means of $\{0,1\}$ and $\{4,5\}$. Every gap is $0.5$, every squared gap is $0.25$, and $J=1$. One more pass changes nothing, so the history is $[25,\ 38/9,\ 1,\ 1]$ and you stop when $J$ repeats.

### Why J can only fall

Each half of a pass can only lower $J$ or leave it alone. In the assign move, every point picks the centre with the smallest gap, so its own gap cannot grow. In the update move, each group's centre goes to its mean, and the mean has the smallest total squared gap of any position, so the group's total cannot grow. $J$ never goes below 0 and only a finite number of groupings exist, so the loop must stop.

> [!KEY]
> A Lloyd pass, assign then update, never raises $J$. Each move picks the best choice for what the other one fixed.

The exercise records $J$ after the first assignment, then each pass is update then reassign, so a history of $n$ passes has $n+1$ entries. Stopping when $J$ repeats exactly is fine for small examples. On large data, libraries stop when $J$ falls by less than a small tolerance, because rounding can keep $J$ from ever repeating.

## Assign with one matrix product

Finding the nearest centre by looping over every point and every centre is slow in Python. A matrix product does the whole job. Expand the squared distance:

$$
\lVert x - c \rVert^2 = \lVert x \rVert^2 - 2\, x \cdot c + \lVert c \rVert^2 .
$$

The first term is the same for every centre, so it cannot change which centre wins. Drop it and compare $\lVert c \rVert^2 - 2\, x\cdot c$. For the point $4$ and the depots $0$ and $1$ the scores are $0-0=0$ and $1-8=-7$, so depot $1$ wins. The true squared distances are $16$ and $9$, which differ by the same $7$. In numpy:

```python
scores = np.sum(centers**2, axis=1) - 2 * X @ centers.T   # shape (n, k)
assignments = np.argmin(scores, axis=1)
```

> [!WARNING]
> The scores can be negative, because $\lVert x \rVert^2$ was dropped. Use them only inside `argmin`. Compute $J$ from the real gaps.

An empty group is a second trap. If no point is nearest to a centre, its mean is the mean of zero points, which is `nan`. Keep the old centre for that group.

## Where this breaks

Figure 3 shows two facts about $J$ on the 24 addresses. On the left, $J$ falls from 308.8 to 74.9 to 18.9 over two passes, and the third pass leaves it unchanged. On the right, $J$ at the end of the loop for each $k$ shows why $J$ cannot choose $k$ for you.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Left, J falling over three passes from 308.8 to 18.9. Right, final J against the number of centres, dropping steeply to three and then flattening">
<defs></defs>
<line x1="55.0" y1="210.0" x2="255.0" y2="210.0" class="stroke-muted" />
<line x1="55.0" y1="40.0" x2="55.0" y2="210.0" class="stroke-muted" />
<line x1="345.0" y1="210.0" x2="565.0" y2="210.0" class="stroke-muted" />
<line x1="345.0" y1="40.0" x2="345.0" y2="210.0" class="stroke-muted" />
<line x1="66.8" y1="210.0" x2="66.8" y2="214.0" class="stroke-muted" />
<text x="66.8" y="227.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="125.6" y1="210.0" x2="125.6" y2="214.0" class="stroke-muted" />
<text x="125.6" y="227.0" class="fig-small" text-anchor="middle" >1</text>
<line x1="184.4" y1="210.0" x2="184.4" y2="214.0" class="stroke-muted" />
<text x="184.4" y="227.0" class="fig-small" text-anchor="middle" >2</text>
<line x1="243.2" y1="210.0" x2="243.2" y2="214.0" class="stroke-muted" />
<text x="243.2" y="227.0" class="fig-small" text-anchor="middle" >3</text>
<line x1="55.0" y1="210.0" x2="255.0" y2="210.0" class="stroke-grid" />
<line x1="51.0" y1="210.0" x2="55.0" y2="210.0" class="stroke-muted" />
<text x="48.0" y="214.0" class="fig-small" text-anchor="end" >0</text>
<line x1="55.0" y1="158.5" x2="255.0" y2="158.5" class="stroke-grid" />
<line x1="51.0" y1="158.5" x2="55.0" y2="158.5" class="stroke-muted" />
<text x="48.0" y="162.5" class="fig-small" text-anchor="end" >100</text>
<line x1="55.0" y1="107.0" x2="255.0" y2="107.0" class="stroke-grid" />
<line x1="51.0" y1="107.0" x2="55.0" y2="107.0" class="stroke-muted" />
<text x="48.0" y="111.0" class="fig-small" text-anchor="end" >200</text>
<line x1="55.0" y1="55.5" x2="255.0" y2="55.5" class="stroke-grid" />
<line x1="51.0" y1="55.5" x2="55.0" y2="55.5" class="stroke-muted" />
<text x="48.0" y="59.5" class="fig-small" text-anchor="end" >300</text>
<line x1="358.8" y1="210.0" x2="358.8" y2="214.0" class="stroke-muted" />
<text x="358.8" y="227.0" class="fig-small" text-anchor="middle" >1</text>
<line x1="386.2" y1="210.0" x2="386.2" y2="214.0" class="stroke-muted" />
<text x="386.2" y="227.0" class="fig-small" text-anchor="middle" >2</text>
<line x1="413.8" y1="210.0" x2="413.8" y2="214.0" class="stroke-muted" />
<text x="413.8" y="227.0" class="fig-small" text-anchor="middle" >3</text>
<line x1="441.2" y1="210.0" x2="441.2" y2="214.0" class="stroke-muted" />
<text x="441.2" y="227.0" class="fig-small" text-anchor="middle" >4</text>
<line x1="468.8" y1="210.0" x2="468.8" y2="214.0" class="stroke-muted" />
<text x="468.8" y="227.0" class="fig-small" text-anchor="middle" >5</text>
<line x1="496.2" y1="210.0" x2="496.2" y2="214.0" class="stroke-muted" />
<text x="496.2" y="227.0" class="fig-small" text-anchor="middle" >6</text>
<line x1="523.8" y1="210.0" x2="523.8" y2="214.0" class="stroke-muted" />
<text x="523.8" y="227.0" class="fig-small" text-anchor="middle" >7</text>
<line x1="551.2" y1="210.0" x2="551.2" y2="214.0" class="stroke-muted" />
<text x="551.2" y="227.0" class="fig-small" text-anchor="middle" >8</text>
<line x1="345.0" y1="210.0" x2="565.0" y2="210.0" class="stroke-grid" />
<line x1="341.0" y1="210.0" x2="345.0" y2="210.0" class="stroke-muted" />
<text x="338.0" y="214.0" class="fig-small" text-anchor="end" >0</text>
<line x1="345.0" y1="161.4" x2="565.0" y2="161.4" class="stroke-grid" />
<line x1="341.0" y1="161.4" x2="345.0" y2="161.4" class="stroke-muted" />
<text x="338.0" y="165.4" class="fig-small" text-anchor="end" >100</text>
<line x1="345.0" y1="112.9" x2="565.0" y2="112.9" class="stroke-grid" />
<line x1="341.0" y1="112.9" x2="345.0" y2="112.9" class="stroke-muted" />
<text x="338.0" y="116.9" class="fig-small" text-anchor="end" >200</text>
<line x1="345.0" y1="64.3" x2="565.0" y2="64.3" class="stroke-grid" />
<line x1="341.0" y1="64.3" x2="345.0" y2="64.3" class="stroke-muted" />
<text x="338.0" y="68.3" class="fig-small" text-anchor="end" >300</text>
<text x="155.0" y="24.0" class="fig-bold" text-anchor="middle" >J on each pass (three depots)</text>
<text x="455.0" y="24.0" class="fig-bold" text-anchor="middle" >Final J for each k (best of 20 starts)</text>
<text x="155.0" y="246.0" class="fig-small" text-anchor="middle" >pass (0 is the first grouping)</text>
<text x="455.0" y="246.0" class="fig-small" text-anchor="middle" >k, the number of depots</text>
<polyline points="66.8,50.9 125.6,171.4 184.4,200.2 243.2,200.2" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="66.8" cy="50.9" r="4.5" class="fill-b" />
<text x="74.8" y="42.9" class="fig-small" text-anchor="start" >308.8</text>
<circle cx="125.6" cy="171.4" r="4.5" class="fill-b" />
<text x="133.6" y="163.4" class="fig-small" text-anchor="start" >74.9</text>
<circle cx="184.4" cy="200.2" r="4.5" class="fill-b" />
<text x="192.4" y="192.2" class="fig-small" text-anchor="start" >18.9</text>
<circle cx="243.2" cy="200.2" r="4.5" class="fill-b" />
<text x="251.2" y="206.2" class="fig-small" text-anchor="start" >18.9</text>
<polyline points="358.8,46.7 386.2,130.0 413.8,200.8 441.2,202.9 468.8,204.8 496.2,206.1 523.8,206.9 551.2,207.6" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="358.8" cy="46.7" r="4.5" class="fill-b" />
<circle cx="386.2" cy="130.0" r="4.5" class="fill-b" />
<circle cx="413.8" cy="200.8" r="4.5" class="fill-b" />
<circle cx="441.2" cy="202.9" r="4.5" class="fill-b" />
<circle cx="468.8" cy="204.8" r="4.5" class="fill-b" />
<circle cx="496.2" cy="206.1" r="4.5" class="fill-b" />
<circle cx="523.8" cy="206.9" r="4.5" class="fill-b" />
<circle cx="551.2" cy="207.6" r="4.5" class="fill-b" />
<text x="425.8" y="178.8" class="fig-small" text-anchor="start" >elbow at k = 3</text>
<line x1="421.8" y1="182.8" x2="415.8" y2="194.8" class="stroke-muted" />
</svg>
Figure 3. Left: J never rises and stops changing after two passes. Right: J keeps falling as you add depots, so it cannot choose k alone. Look for the elbow, where the steep drop ends.
```

- **You must pick $k$.** $J$ always falls as $k$ grows, from 336 at one depot to 165 at two and 18.9 at three. The drop then slows to 14.6 at four. Look for the elbow, where adding a depot stops buying much. Here it is at three. Each point on the curve is the best of 20 runs from different starting depots, because one unlucky start can hide the elbow. The next chapter explains how to start well.
- **Units matter.** If one column is in kilometres and another in grams, the large numbers dominate every distance. Standardise each column first.
- **It prefers round lumps of similar size.** Every point goes to its nearest centre, so the border between two groups is the line halfway between their centres. Long thin groups get cut in the middle.
- **It stops at a bottom, not always the lowest.** A different start can end at a worse $J$. The next chapter, "k-means++: starting well and choosing k", fixes that.

> [!RECAP]
> Before you build, keep these points in hand:
> - $J$ is the sum of squared gaps from each point to its own centre, a sum and not a mean.
> - The mean of a group has the smallest total squared gap, so it is the best centre.
> - Assign sends each point to its nearest centre, using $\lVert c\rVert^2 - 2x\cdot c$ and `argmin`.
> - Update moves each centre to its group's mean, and an empty group keeps its old centre.
> - $J$ never rises, so the loop stops when $J$ repeats.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`kmeans_objective(X, centers, assignments)`** takes `(n, d)`, `(k, d)`, and `(n,)` and returns $J$ as a Python float. Use `centers[assignments]` to pair each point with its centre, square the gaps, and add them all. The usual mistake is taking the mean. Its shape is:

    ```python
    def kmeans_objective(X, centers, assignments):
        # gaps = X - centers[assignments], then a float sum of squares
    ```

2. **`assign_nearest(X, centers)`** returns an integer array of shape `(n,)`. Build the `(n, k)` score matrix with the matrix product and take `argmin` over axis 1. The usual mistake is `argmin` over axis 0, which returns the nearest point to each centre.

    ```python
    def assign_nearest(X, centers):
        # scores = ||c||^2 - 2 x.c, then argmin(axis=1)
    ```

3. **`lloyd(X, k, init, iters)`** copies `init`, assigns, and records $J$. Then it loops up to `iters` times: move every centre whose group has at least one point to that group's mean, reassign, append $J$, and stop when $J$ equals the previous entry. It returns `(centers, assignments, history)`. The usual mistake is writing into `init`, which changes the reader's array.

    ```python
    def lloyd(X, k, init, iters):
        # copy init; for each j, average X[assignments == j] only if it has rows
    ```

## The exercise

You build `kmeans_objective`, `assign_nearest`, and `lloyd`. The checks reproduce the chapter's numbers: $J=25$ for the first grouping and the history $[25, 38/9, 1, 1]$. They compare `assign_nearest` with a brute-force search, confirm that $J$ never rises on random clusters, and confirm that an empty group keeps its centre and never turns into `nan`. Running the file prints the four-point example and the delivery clusters.
