---
title: "k-means++: starting well and choosing k"
slug: 054-kmeans-plusplus-seeding
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Lloyd's loop only rolls downhill from where it starts, so k-means++ spreads the start by picking centres with chance proportional to squared distance.
minutes: 40
gpu: false
objectives:
  - Explain why Lloyd's loop can stop at a worse grouping, and what a local minimum is.
  - Say why uniform starts and always-the-farthest starts both fail, and what squared-distance sampling does instead.
  - Turn squared distances into seeding probabilities and write the full k-means++ seeding.
  - Keep the best of several restarts, and use it to draw a trustworthy elbow curve for choosing k.
prereqs:
  - 052-probabilistic-pca-em
  - 053-kmeans-lloyd
exercise: 054-kmeans-plusplus-seeding
---

# k-means++: starting well and choosing k

You run k-means on the same 60 delivery addresses twice, changing only the random starting depots. The first run ends with $J=$ 88.5. The second ends with $J=$ 439.8, about five times worse, and a map of its depots shows two of them crowded into one neighbourhood while a third floats in empty space between two others.

Same data, same algorithm, same $k$. The only difference is where the depots started. The previous chapter proved that each Lloyd pass lowers $J$, but it never promised that the loop finds the lowest $J$. Which start do you trust, and can you choose a start that is likely to be a good one?

This chapter explains why the loop gets stuck, shows why the two obvious fixes fail, and builds the rule that most libraries use to start k-means.

## What you already know

Five ideas carry this chapter:

- Lloyd's loop assigns every point to its nearest centre, then moves each centre to its group's mean (see "k-means: group points by moving centres").
- $J$ is the sum of squared gaps from each point to its centre, and a pass never raises it.
- `assign_nearest` and `lloyd` are your own functions from that chapter.
- A probability vector is a list of non-negative numbers that add up to 1, and `rng.choice(n, p=probs)` draws an index from it.
- A seeded generator `rng` gives the same draws every time, so a figure or a check can be repeated.

## A bad start gets stuck

Picture $J$ as a landscape with many valleys. Where you put the starting depots decides which valley you stand in. Lloyd's loop rolls downhill from there, and it stops at the bottom of that valley. That bottom is a **local minimum**: no single pass can improve it, yet a different valley may be much deeper.

Figure 1 shows both outcomes on the same 60 points.

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
Figure 1. Both runs use the same data and the same Lloyd loop, and each stopped because J stopped falling. The left run reached a valley bottom that is not the lowest, with J = 439.8, about five times the good run.
```

In the stuck run, two depots started in the top cluster and one in the lower left. Each cluster of points belongs to whichever depot is nearest, so the top cluster got split in two, and the other two clusters were merged under one depot halfway between them. Moving one of the top depots down to the empty cluster would cut $J$ sharply. Lloyd never tries it, because it only moves a depot to the mean of the points it already owns.

How often does it happen? Run Lloyd 2000 times from three random data points, and 2000 times from the seeding you build in this chapter. About 8 percent of the random starts got stuck like Figure 1. For the seeded starts it was about 1 percent, and the average final $J$ fell from 116 to 92.

## Why not pick the starts uniformly?

Pick three data points at random, and nothing stops two from landing in the same cluster. With 20 points in each of three clusters, the chance that the three picks fall in three different clusters is

$$
\frac{6 \cdot 20 \cdot 20 \cdot 20}{60 \cdot 59 \cdot 58} \approx 0.23 .
$$

Read it aloud: six orderings of the three clusters, times twenty choices in each, divided by every way to pick an ordered triple of points. Barely one start in four has a depot in each cluster. Luck does not scale to many clusters.

## Why not always take the farthest point?

A different idea is to pick the first depot at random, then always take the point that lies farthest from every depot so far. That spreads the depots out, and it is wrong in a different way: one stray address far from everybody, say a typo in the coordinates, is the farthest point every time, so it steals a depot. The seeding then has one depot on noise and one too few for the real clusters.

You want a rule between the two: far points should be much more likely, but never certain.

## Squared distance as a chance

Let $D(x)$ be the distance from a point $x$ to its nearest centre chosen so far. The **k-means++** rule picks each new centre from the data points with a probability proportional to $D(x)^2$:

$$
p(x_i) = \frac{D(x_i)^2}{\sum_{j=1}^{n} D(x_j)^2} .
$$

Read it aloud: the chance of picking a point is its squared distance divided by the total of all squared distances, so the chances add up to 1. A point already chosen has $D=0$, so it can never be chosen twice.

Try it on the points $0, 1, 10$ with the first centre at $0$. The squared distances are $0, 1, 100$, which add to $101$, so the chances are $0,\ 1/101,\ 100/101$. The far point wins about 99 times in 100, but the point at $1$ keeps a small chance. Plain distances $0, 1, 10$ would give $0,\ 1/11,\ 10/11$, a 9 percent chance for the near point. Squaring is what makes the far points dominate. Figure 2 shows both.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bar chart of the chance each of three points at 0, 1, and 10 becomes the next centre, using distance and using distance squared">
<defs></defs>
<line x1="60.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-muted" />
<line x1="60.0" y1="50.0" x2="60.0" y2="220.0" class="stroke-muted" />
<line x1="60.0" y1="220.0" x2="540.0" y2="220.0" class="stroke-grid" />
<line x1="56.0" y1="220.0" x2="60.0" y2="220.0" class="stroke-muted" />
<text x="53.0" y="224.0" class="fig-small" text-anchor="end" >0</text>
<line x1="60.0" y1="177.5" x2="540.0" y2="177.5" class="stroke-grid" />
<line x1="56.0" y1="177.5" x2="60.0" y2="177.5" class="stroke-muted" />
<text x="53.0" y="181.5" class="fig-small" text-anchor="end" >0.25</text>
<line x1="60.0" y1="135.0" x2="540.0" y2="135.0" class="stroke-grid" />
<line x1="56.0" y1="135.0" x2="60.0" y2="135.0" class="stroke-muted" />
<text x="53.0" y="139.0" class="fig-small" text-anchor="end" >0.5</text>
<line x1="60.0" y1="92.5" x2="540.0" y2="92.5" class="stroke-grid" />
<line x1="56.0" y1="92.5" x2="60.0" y2="92.5" class="stroke-muted" />
<text x="53.0" y="96.5" class="fig-small" text-anchor="end" >0.75</text>
<line x1="60.0" y1="50.0" x2="540.0" y2="50.0" class="stroke-grid" />
<line x1="56.0" y1="50.0" x2="60.0" y2="50.0" class="stroke-muted" />
<text x="53.0" y="54.0" class="fig-small" text-anchor="end" >1</text>
<text x="300.0" y="30.0" class="fig-bold" text-anchor="middle" >Chance of becoming the next centre (first centre at 0)</text>
<rect x="85.0" y="220.0" width="52.0" height="0.5" class="fill-c" />
<text x="111.0" y="214.0" class="fig-small" text-anchor="middle" >0</text>
<rect x="143.0" y="220.0" width="52.0" height="0.5" class="fill-b" />
<text x="169.0" y="214.0" class="fig-small" text-anchor="middle" >0</text>
<text x="140.0" y="240.0" text-anchor="middle" >point at 0</text>
<rect x="245.0" y="204.5" width="52.0" height="15.5" class="fill-c" />
<text x="271.0" y="198.5" class="fig-small" text-anchor="middle" >0.091</text>
<rect x="303.0" y="218.3" width="52.0" height="1.7" class="fill-b" />
<text x="329.0" y="212.3" class="fig-small" text-anchor="middle" >0.010</text>
<text x="300.0" y="240.0" text-anchor="middle" >point at 1</text>
<rect x="405.0" y="65.5" width="52.0" height="154.5" class="fill-c" />
<text x="431.0" y="59.5" class="fig-small" text-anchor="middle" >0.909</text>
<rect x="463.0" y="51.7" width="52.0" height="168.3" class="fill-b" />
<text x="489.0" y="45.7" class="fig-small" text-anchor="middle" >0.990</text>
<text x="460.0" y="240.0" text-anchor="middle" >point at 10</text>
<rect x="90.0" y="75.0" width="12.0" height="12.0" class="fill-c" />
<text x="108.0" y="86.0" class="fig-small" text-anchor="start" >plain distance: 0, 1/11, 10/11</text>
<rect x="90.0" y="95.0" width="12.0" height="12.0" class="fill-b" />
<text x="108.0" y="106.0" class="fig-small" text-anchor="start" >distance squared: 0, 1/101, 100/101</text>
</svg>
Figure 2. With plain distance the point at 1 keeps a 9 percent chance. With squared distance it keeps under 1 percent, and the far point takes over 99 percent, but the nearer point is never ruled out.
```

In numpy, keep one array of squared distances to the nearest chosen centre. After each pick, a point's distance can only shrink, so update it with `np.minimum`:

```python
chosen = rng.choice(n, p=sq_dist / sq_dist.sum())
sq_dist = np.minimum(sq_dist, np.sum((X - X[chosen]) ** 2, axis=1))
```

If every point already sits on a centre, `sq_dist.sum()` is 0 and the division gives `nan`. Fall back to equal chances.

### What the guarantee says

Arthur and Vassilvitskii proved that, on average over the random draws, the seeds alone already cost at most $8(\ln k + 2)$ times the lowest possible $J$. For $k=3$ that factor is $8(1.10+2)\approx 25$. It is a loose promise about the average, and a single draw can still be poor. The measured results above are far better than the bound.

## Watching it choose

Figure 3 shows the rule on three clusters of 10 points plus one stray point. The circle on each point has area proportional to its squared distance to the nearest chosen centre, which is its chance of being picked.

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
Figure 3. The big circles are the points most likely to be picked. After the first centre, the far clusters and the stray point have the large circles. After the second centre, the circles near it collapse, so a third centre rarely lands in a cluster that already has one.
```

After the first centre in the lower-left cluster, the second cluster holds 31 percent of the total, the third cluster 49 percent, the stray point 18 percent, and the first cluster only 2 percent. The stray point has a real chance, but under one in five, not a certainty. Suppose the second pick lands in the second cluster. The circles near it collapse, and the shares become 71 percent for the third cluster, 21 percent for the stray point, and 4 and 4 percent for the clusters that already hold a depot. A third centre rarely lands in a cluster that already has one.

> [!KEY]
> Pick each new centre with probability proportional to its squared distance from the centres already chosen. Far points are favoured, so depots spread out, and no single point is certain, so one stray point cannot take over.

## Keep the best of several restarts

Seeding makes a bad start unlikely, not impossible, and a run is cheap. So run it several times and keep the lowest $J$. This is the **restart** habit: seed, run Lloyd to the end, remember the result, and repeat `n_init` times. The cost grows by a factor of `n_init`.

Restarts also make the elbow curve from the previous chapter trustworthy. To choose $k$, you plot the best $J$ you can find for each $k$ and look for the elbow. If a stuck run sneaks in, a point on the curve sits too high and hides the elbow. Figure 3 of the previous chapter used the best of 20 seeded restarts at every $k$ for this reason.

> [!RECAP]
> Before you build, keep these points in hand:
> - Lloyd's loop stops at the bottom of the valley it starts in, which can be a poor local minimum.
> - Uniform starts often put two depots in one cluster. The farthest point is always a stray point.
> - k-means++ picks each centre with probability $D^2$ over the sum of $D^2$, with the first centre uniform.
> - Update the distances with `np.minimum`, and fall back to equal chances when their sum is 0.
> - Keep the lowest-$J$ run of several restarts.

## Build it step by step

You write three functions. The three k-means functions from the previous chapter are already in the file.

1. **`seeding_probabilities(sq_dist)`** takes an array of shape `(n,)` of squared distances and returns probabilities of the same shape. Divide by the sum. If the sum is 0, return `np.full(n, 1 / n)`. The usual mistake is skipping the zero case, which gives `nan` when every point is a duplicate.

    ```python
    def seeding_probabilities(sq_dist):
        # sq_dist / sq_dist.sum(), or uniform when the sum is 0
    ```

2. **`kmeans_plusplus_init(X, k, rng)`** returns a `(k, d)` array of rows of `X`. Draw the first index with `rng.integers(n)`, then for each further centre draw with `rng.choice(n, p=seeding_probabilities(sq_dist))` and shrink `sq_dist` with `np.minimum`. The usual mistake is using plain distances, so the point at 1 gets 9 percent instead of 1 percent.

    ```python
    def kmeans_plusplus_init(X, k, rng):
        # first centre uniform; then sample by squared distance, np.minimum
    ```

3. **`best_of_restarts(X, k, n_init, rng)`** runs `kmeans_plusplus_init` and then `lloyd(X, k, init, 100)`, `n_init` times, and returns `(centers, assignments, J)` for the run whose last history entry is smallest. The usual mistake is reseeding `rng` inside the loop, so every restart repeats the same start.

    ```python
    def best_of_restarts(X, k, n_init, rng):
        # keep the run with the smallest history[-1]
    ```

## Where this breaks

- **Seeding can still pick a stray point.** The stray point in Figure 3 had a one in five chance. Restarts mean one such start does not decide the result.
- **Restarts cost time.** Ten restarts take about ten times as long. Stop when the best $J$ stops improving.
- **The bound is loose.** The guarantee is about the average, so trust the restarts, not the number.

## The exercise

You build `seeding_probabilities`, `kmeans_plusplus_init`, and `best_of_restarts`. The checks confirm the chances $0, 1/101, 100/101$ and the uniform fallback. They draw the seeds thousands of times on the points $0, 1, 10$ to confirm the far point appears about 99 percent of the time, and they confirm that seeding puts one seed in each of three clusters far more often than uniform picks. They also confirm that the best of several restarts is never worse than a single run and finds the good grouping. Running the file prints the seeding probabilities and the best $J$ over ten restarts.
