---
title: "Nearest neighbours and matrix factorisation: predict from similar rows, then from a low-rank model"
slug: 032-nearest-neighbours-and-matrix-factorisation
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Predict from the k most similar rows, see why that fails in high dimension and on sparse ratings, then fit a low-rank model by alternating least squares.
minutes: 70
gpu: false
objectives:
  - Predict a label or a number from the k nearest training rows with vectorised squared distances, and choose k by leave-one-out error.
  - Explain why distances concentrate as the dimension grows, and measure the nearest-to-farthest ratio.
  - Explain why user-based k-NN struggles on a sparse ratings table.
  - Fit R ≈ U V^T on the observed entries only, with ridge penalties, by alternating least squares.
  - Compare held-out RMSE against the global-mean and user-bias baselines.
prereqs:
  - 031-gaussian-mixtures-and-em
exercise: 032-nearest-neighbours-and-matrix-factorisation
---

# Nearest neighbours and matrix factorisation: predict from similar rows, then from a low-rank model

A streaming service has 120 users and 80 films. Each user has rated about a dozen films, so the table of ratings is 85 percent blank. You must guess the blank cells: what would user 17 give film 42? A good guess is what to recommend next.

The most natural idea needs no training. Find users who rated films the way user 17 did, and average what they gave film 42. That idea is *nearest neighbours*, and the first half of this chapter builds it. Then it breaks twice. In many dimensions every row is about as far from every other row, so "nearest" stops meaning anything. On a sparse table two users share almost no films, so you cannot even measure how similar they are.

The second half changes the question. Instead of comparing rows, you assume every user and every film has a short list of hidden numbers, and a rating is their dot product. That is *matrix factorisation*, and you fit it by solving small ridge regressions in turn.

## What you already know

- `X` has shape `(n, d)`, one row per point, and `k`-fold or leave-one-out cross-validation scores a choice on rows the model did not see ([Cross-validation and the leave-one-out shortcut](/lessons/012-cross-validation-and-leave-one-out)).
- Ridge regression adds a fee on large weights and has a closed-form solution, one linear system ([Ridge regression](/lessons/006-ridge-regression)).
- The SVD writes a matrix as a sum of rank-one pieces, and keeping the biggest $k$ pieces gives the best rank-$k$ approximation ([Principal components](/lessons/028-principal-components)).
- A squared distance is the sum of squared coordinate gaps, and `||a||^2 - 2 a.b + ||b||^2` computes all of them with one matrix product ([k-means](/lessons/030-k-means)).

## Predict from the k nearest rows

Figure 1 shows the whole method. Five training points sit on a plane, each with a label and a price. A query point arrives at $(1, 1)$. Find the training points closest to it, and let them vote.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Five labelled training points and a query at (1, 1) with a circle through its three nearest neighbours">
<rect x="40" y="20" width="480" height="210" class="fill-sunk stroke-grid"/>
<circle cx="145.0" cy="155.0" r="106.1" class="fill-soft-b stroke-b dashed"/>
<circle cx="70.0" cy="210.0" r="7" class="fill-a"/>
<text x="81.0" y="201.0" class="fig-small">#0</text>
<circle cx="145.0" cy="210.0" r="7" class="fill-a"/>
<text x="156.0" y="201.0" class="fig-small">#1</text>
<circle cx="70.0" cy="100.0" r="7" class="fill-c"/>
<text x="81.0" y="91.0" class="fig-small">#2</text>
<circle cx="295.0" cy="45.0" r="7" class="fill-c"/>
<text x="306.0" y="36.0" class="fig-small">#3</text>
<circle cx="370.0" cy="45.0" r="7" class="fill-c"/>
<text x="381.0" y="36.0" class="fig-small">#4</text>
<rect x="138.0" y="148.0" width="14" height="14" class="fill-b"/>
<text x="156.0" y="177.0" class="fig-bold">query</text>
<text x="340" y="192" class="fig-small">blue: class 0, green: class 1</text>
<text x="340" y="210" class="fig-small">circle: k = 3 reach</text></svg>
Figure 1. The query at (1, 1) and the circle that reaches its three nearest neighbours. Points 0 and 2 tie at squared distance 2, and the lower index wins when k = 2.
```

The bet is that rows that look alike have similar answers. The method has no training step: it stores the table and works when a query arrives. Four choices define it: the distance (here, squared Euclidean), the number of neighbours $k$, how to combine their answers (a vote for labels, the mean for numbers), and how to break ties.

### A worked example

Take the five points in Figure 1, labelled 0, 0, 1, 1, 1, with prices 10, 12, 20, 30, 34. The query is $(1, 1)$. Its squared distances to the five points are

$$
2,\quad 1,\quad 2,\quad 8,\quad 13.
$$

For the first point, $(1-0)^2 + (1-0)^2 = 2$. For the second, $(1-1)^2+(1-0)^2 = 1$. Sorted, the order is point 1, then points 0 and 2, then 3 and 4.

- With $k=1$, the neighbour is point 1. The label is 0 and the price is 12.
- With $k=2$, points 0 and 2 tie for the second place. A stable sort keeps the lower index, so the second neighbour is point 0. The labels are 0 and 0, and the mean price is $(12+10)/2 = 11$.
- With $k=3$, the neighbours are points 1, 0, and 2. The labels are 0, 0, 1, so the vote gives 0. The mean price is $(12+10+20)/3 = 14$.

A sort that breaks ties differently on two runs gives two answers for the same data. In this chapter, a tie in distance goes to the lower training index, and a tie in the vote goes to the smaller label.

### All the distances in one matrix product

A loop over every query and every training row is slow in Python. Expand the squared distance:

$$
\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2\, a \cdot b + \lVert b \rVert^2 .
$$

Read it aloud: the squared distance is the squared length of $a$, minus twice the dot product, plus the squared length of $b$. The dot products for every pair are one matrix product, `A @ B.T`, and the two length terms are row sums that broadcast.

```python
d = (A**2).sum(axis=1)[:, None] - 2 * A @ B.T + (B**2).sum(axis=1)[None, :]
d = np.maximum(d, 0.0)        # rounding can leave -1e-16
idx = np.argsort(d, axis=1, kind="stable")[:, :k]
```

The result has shape `(n, m)`, and `idx` holds, for each query, the indices of its $k$ nearest training rows. A classifier then counts labels in `ytr[idx[i]]` with `np.bincount`. A regressor takes `ytr[idx].mean(axis=1)`.

> [!WARNING]
> When you predict on the training set itself, each point is its own nearest neighbour at distance 0. With $k=1$ the error is exactly 0, and the model looks perfect. To score k-NN on rows it stores, exclude the row itself, which is the leave-one-out estimate.

### Choose k with cross-validation

Small $k$ follows every wiggle: one mislabelled neighbour flips the answer, so the variance is high. Large $k$ averages over a wide area and washes out real structure, so the bias is high. It is the bias-variance trade-off with $k$ as the knob, and held-out data picks the value ([cross-validation](/lessons/012-cross-validation-and-leave-one-out)).

For k-NN the leave-one-out score is cheap, because you compute the distance matrix once, set its diagonal to infinity, and read off the neighbours for every $k$. On 200 seeded points from two overlapping Gaussian clouds, the leave-one-out error is:

| $k$ | 1 | 3 | 5 | 9 | 15 | 25 | 51 | 101 |
|---|---|---|---|---|---|---|---|---|
| error | 0.235 | 0.210 | 0.195 | 0.190 | 0.165 | 0.175 | 0.175 | 0.175 |

The error falls from 0.235 at $k=1$ to 0.165 at $k=15$, then rises a little and flattens, because the clouds overlap and no $k$ can beat the overlap.

> [!KEY]
> k-NN stores the data and answers by looking up the $k$ most similar rows. $k$ is its only dial: small $k$ has high variance, large $k$ has high bias, and leave-one-out error picks it. Always exclude a row from its own neighbours.

## Why neighbours fail in many dimensions

The bet behind k-NN is that nearby rows have similar answers. That bet needs "nearby" to mean something: the nearest row must be much closer than a typical row. In many dimensions it is not.

Draw 200 points uniformly from the unit square, and a few are close to a query while a few are far. Draw them from a unit cube in 1000 dimensions, and every coordinate adds a little squared distance. The sum of 1000 small random amounts is close to its average, so every point ends up at about the same distance from the query.

Figure 2 measures it: for each dimension $d$, the ratio of the nearest distance to the farthest, for 200 uniform points and one query, averaged over 20 seeds.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Final loss against learning rate at widths 32, 128, and 512. Under standard parameterisation the best rate moves left as the width grows. Under muP the best rate stays in one place."><text x="170" y="26" text-anchor="middle" class="fig-bold">Standard parameterisation</text><line x1="55" y1="220" x2="285" y2="220" class="stroke-muted"/><line x1="55" y1="50" x2="55" y2="220" class="stroke-muted"/><line x1="55.0" y1="50" x2="55.0" y2="220" class="stroke-grid"/><text x="55.0" y="236" text-anchor="middle" class="fig-small">2^-12</text><line x1="124.0" y1="50" x2="124.0" y2="220" class="stroke-grid"/><text x="124.0" y="236" text-anchor="middle" class="fig-small">2^-9</text><line x1="193.0" y1="50" x2="193.0" y2="220" class="stroke-grid"/><text x="193.0" y="236" text-anchor="middle" class="fig-small">2^-6</text><line x1="262.0" y1="50" x2="262.0" y2="220" class="stroke-grid"/><text x="262.0" y="236" text-anchor="middle" class="fig-small">2^-3</text><line x1="55" y1="202.4" x2="285" y2="202.4" class="stroke-grid"/><text x="49" y="206.4" text-anchor="end" class="fig-small">0.03</text><line x1="55" y1="150.1" x2="285" y2="150.1" class="stroke-grid"/><text x="49" y="154.1" text-anchor="end" class="fig-small">0.1</text><line x1="55" y1="102.3" x2="285" y2="102.3" class="stroke-grid"/><text x="49" y="106.3" text-anchor="end" class="fig-small">0.3</text><line x1="55" y1="50.0" x2="285" y2="50.0" class="stroke-grid"/><text x="49" y="54.0" text-anchor="end" class="fig-small">1</text><polyline points="55.0,60.2 78.0,105.6 101.0,134.4 124.0,148.6 147.0,159.1 170.0,168.3 193.0,177.4 216.0,169.6 239.0,164.8 262.0,155.6 285.0,100.1" fill="none" class="stroke-a" stroke-width="2"/><circle cx="193.0" cy="177.4" r="5" class="fill-a"/><polyline points="55.0,140.1 78.0,151.3 101.0,159.4 124.0,167.8 147.0,174.7 170.0,190.2 193.0,181.7 216.0,171.9 239.0,166.4 262.0,71.8 285.0,50.0" fill="none" class="stroke-b" stroke-width="2"/><circle cx="170.0" cy="190.2" r="5" class="fill-b"/><polyline points="55.0,165.7 78.0,170.0 101.0,187.4 124.0,189.7 147.0,178.8 170.0,161.9 193.0,159.6 216.0,156.8 239.0,136.9 262.0,58.9 285.0,50.0" fill="none" class="stroke-c" stroke-width="2"/><circle cx="124.0" cy="189.7" r="5" class="fill-c"/><text x="170" y="256" text-anchor="middle" class="fig-small">learning rate</text><text x="455" y="26" text-anchor="middle" class="fig-bold">muP</text><line x1="340" y1="220" x2="570" y2="220" class="stroke-muted"/><line x1="340" y1="50" x2="340" y2="220" class="stroke-muted"/><line x1="340.0" y1="50" x2="340.0" y2="220" class="stroke-grid"/><text x="340.0" y="236" text-anchor="middle" class="fig-small">2^-12</text><line x1="409.0" y1="50" x2="409.0" y2="220" class="stroke-grid"/><text x="409.0" y="236" text-anchor="middle" class="fig-small">2^-9</text><line x1="478.0" y1="50" x2="478.0" y2="220" class="stroke-grid"/><text x="478.0" y="236" text-anchor="middle" class="fig-small">2^-6</text><line x1="547.0" y1="50" x2="547.0" y2="220" class="stroke-grid"/><text x="547.0" y="236" text-anchor="middle" class="fig-small">2^-3</text><line x1="340" y1="202.4" x2="570" y2="202.4" class="stroke-grid"/><line x1="340" y1="150.1" x2="570" y2="150.1" class="stroke-grid"/><line x1="340" y1="102.3" x2="570" y2="102.3" class="stroke-grid"/><line x1="340" y1="50.0" x2="570" y2="50.0" class="stroke-grid"/><polyline points="340.0,51.1 363.0,110.2 386.0,142.3 409.0,151.9 432.0,159.8 455.0,167.4 478.0,169.1 501.0,174.0 524.0,165.2 547.0,164.5 570.0,63.7" fill="none" class="stroke-a" stroke-width="2"/><circle cx="501.0" cy="174.0" r="5" class="fill-a"/><polyline points="340.0,54.6 363.0,139.7 386.0,156.6 409.0,160.2 432.0,166.4 455.0,181.3 478.0,192.2 501.0,192.4 524.0,173.0 547.0,171.0 570.0,112.7" fill="none" class="stroke-b" stroke-width="2"/><circle cx="501.0" cy="192.4" r="5" class="fill-b"/><polyline points="340.0,56.6 363.0,151.2 386.0,160.9 409.0,162.8 432.0,168.9 455.0,185.1 478.0,200.9 501.0,202.5 524.0,183.0 547.0,181.8 570.0,156.2" fill="none" class="stroke-c" stroke-width="2"/><circle cx="501.0" cy="202.5" r="5" class="fill-c"/><text x="455" y="256" text-anchor="middle" class="fig-small">learning rate</text><line x1="200" y1="278" x2="220" y2="278" class="stroke-a" stroke-width="2"/><text x="226" y="282" class="fig-small">width 32</text><line x1="290" y1="278" x2="310" y2="278" class="stroke-b" stroke-width="2"/><text x="316" y="282" class="fig-small">width 128</text><line x1="380" y1="278" x2="400" y2="278" class="stroke-c" stroke-width="2"/><text x="406" y="282" class="fig-small">width 512</text><text x="8" y="40" class="fig-small">final loss</text></svg>
Figure 2. The mean ratio of nearest to farthest distance for 200 uniform points, over 20 seeds. It is 0.04 at d = 2 and 0.91 at d = 1000: the points are almost equally far away.
```

At $d=2$ the ratio is 0.04, so the nearest point is 25 times closer than the farthest. At $d=10$ it is 0.33. At $d=100$ it is 0.75. At $d=1000$ it is 0.91, so the nearest neighbour is barely closer than the farthest point. Beyer and colleagues proved that, under broad conditions, this ratio tends to 1 as the dimension grows.

When all points are about equally far away, the $k$ "nearest" are close to random picks. Real rows often lie near a lower-dimensional surface, because columns are correlated, so k-NN can still work. What counts is the number of directions in which the rows really vary, not the number of columns. When many columns are noise, distances measure the noise. Standardise the columns, and reduce the dimension first, for example with [principal components](/lessons/028-principal-components).

> [!KEY]
> As the dimension grows, the nearest and farthest distances come together. Their ratio tends to 1, so "the nearest rows" carries less and less information.

## Neighbours on a ratings table

Return to the ratings. Call the table $R$, with one row per user and one column per film. Only some entries are observed, and a mask $M$ records which: $M_{ij} = 1$ when user $i$ rated film $j$, and 0 otherwise.

*User-based collaborative filtering* is k-NN on this table. To guess user $i$'s rating of film $j$:

1. Measure how far user $i$ is from every other user, using only films both rated.
2. Keep the $k$ nearest users who rated film $j$.
3. Average their ratings of film $j$.

Step 1 compares two users on their *co-rated* films, and a sparse table has very few.

We planted a table to measure this. It has 120 users and 80 films. Every entry is the dot product of two hidden numbers per user and per film, plus noise with standard deviation 0.2, and 15 percent of the entries are observed. After holding out a fifth of the observed entries for testing, a typical pair of users has 1.2 co-rated films, and 86 percent of the pairs share fewer than 3. The distance between two users is then an average over one or two films, which is mostly noise.

Held-out RMSE bears this out. Predicting the global mean of the training ratings gives 1.12. Predicting the mean plus each user's average offset, the *user-bias baseline*, gives 1.18, slightly worse, because the planted users have no real offsets and a dozen ratings estimate one badly. User-based k-NN with $k=10$ gives 1.08. All three sit near the spread of the table, so they predict almost nothing.

> [!KEY]
> Neighbour methods need a similarity measure, and a sparse table makes the measure unreliable. The fix is to stop comparing rows with each other and instead explain every row with a few shared hidden factors.

## Predict from a low-rank model

Suppose each user has two hidden numbers, such as how much they like action and how much they like romance, and each film has two matching numbers. A rating is the dot product: a user who loves action gives high marks to action films. No one labels these numbers. The fit finds them.

Write the user numbers as the rows of a matrix $U$ with shape $(n, k)$, and the film numbers as the rows of $V$ with shape $(m, k)$. The table is then approximated by a product:

$$
R \approx U V^\top, \qquad \hat r_{ij} = u_i \cdot v_j .
$$

Read it aloud: the predicted rating of user $i$ for film $j$ is the dot product of their two rows of hidden numbers. The number $k$ is the *rank*, and it is small, so $U$ and $V$ together hold far fewer numbers than $R$ has cells. That is why the model can guess the blanks: the 120 by 80 table has 9,600 cells, but rank 2 needs only $2 \times (120 + 80) = 400$ numbers.

If $R$ had no blanks, you would find $U$ and $V$ with the SVD from the [principal components chapter](/lessons/028-principal-components): keep the top $k$ singular values, and no other rank-$k$ table is closer. The SVD cannot take a table with holes. Filling the holes with zeros would tell the SVD that users rated those films 0, which is false. So you minimise the error over observed entries only:

$$
J(U, V) = \sum_{i,j} M_{ij}\,\bigl(r_{ij} - u_i \cdot v_j\bigr)^2 + \lambda\Bigl(\sum_i \lVert u_i \rVert^2 + \sum_j \lVert v_j \rVert^2\Bigr).
$$

The first term is the squared miss, counted only where $M_{ij}=1$. The second is the [ridge](/lessons/006-ridge-regression) fee on every hidden number. Without the fee, a user with two ratings can fit them exactly with enormous numbers, and predict nothing else.

### Alternate between two ridge regressions

$J$ is hard to minimise over $U$ and $V$ together, because the product $u_i \cdot v_j$ multiplies two unknowns. But fix $V$ and look at one user. The loss on their row is

$$
\sum_{j \text{ rated by } i} (r_{ij} - u_i \cdot v_j)^2 + \lambda \lVert u_i \rVert^2 ,
$$

which is a ridge regression. The targets are the user's ratings, the features are the rows $v_j$ of the films they rated, and the weights are $u_i$. Stack those rows into a matrix $V_i$ with one row per rated film, and the ridge solution is

$$
u_i = \bigl(V_i^\top V_i + \lambda I\bigr)^{-1} V_i^\top r_i .
$$

Read it aloud: take the hidden numbers of the films this user rated, and solve the same linear system as in ridge regression to find the user's numbers. It is a $k \times k$ solve, so it is small and fast. Every user gets their own solve, and each one sees only their own ratings. That is how the method skips the blanks without inventing values.

By symmetry, with $U$ fixed, every film gets its own ridge regression against the users who rated it. *Alternating least squares* (ALS) takes turns: solve for every row of $U$, then for every row of $V$, and repeat. In code, the film update is the user update on the transposed table: `als_step(R.T, M.T, V, U, lam)`.

### A worked row

Take one user and rank $k=1$. Three films have hidden numbers $V = (1, 2, 3)$. The user rated film 0 as 2 and film 2 as 5, and left film 1 blank. Let $\lambda = 1$.

The rated films have hidden numbers 1 and 3, so $V_i^\top V_i = 1^2 + 3^2 = 10$, and $V_i^\top r_i = 1\cdot 2 + 3 \cdot 5 = 17$. Then

$$
u_i = \frac{17}{10 + 1} = \frac{17}{11} \approx 1.545 .
$$

The predicted rating for the blank film 1, whose hidden number is 2, is $1.545 \times 2 \approx 3.09$.

Compare a wrong way. If you treat the blank as a rating of 0 and include it, $V^\top V = 14$ and $V^\top r = 17$, so $u_i = 17/15 \approx 1.133$ and the prediction falls to 2.27. The zero pulls the fit down. This is the usual bug in a first implementation: forgetting the mask.

> [!WARNING]
> Never let an unobserved entry reach the loss. A `0` in the table means "unknown", not "rated 0". If you pass the full `R` to the solve instead of the observed rows, ALS learns to predict 0 for everything.

### Why every step helps

Each row solve minimises $J$ exactly over that row, with the other rows fixed, so it cannot raise $J$. The loss never rises across an update of $U$ or of $V$, and it is bounded below by 0, so it converges.

It settles at a point where neither $U$ nor $V$ can improve alone. That is a local minimum, not a guaranteed global one, because $J$ is not convex in $U$ and $V$ together. Small seeded random starts work well in practice. In our planted example the loss falls from 1817 to 329 in 25 sweeps.

### Does it beat the baselines?

Figure 3 trains rank-2 ALS with $\lambda = 1$ on the same planted table and measures RMSE on the held-out entries after each of 30 sweeps. The dashed lines are the three methods from the last section.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Singular value out against singular value in for one quintic Newton-Schulz step, five quintic steps, and five cubic steps. Five quintic steps put every input above 0.01 inside a band near 1."><line x1="60" y1="210" x2="540" y2="210" class="stroke-muted"/><line x1="60" y1="40" x2="60" y2="210" class="stroke-muted"/><rect x="60" y="52.1" width="480" height="72.9" class="fill-soft-c"/><line x1="60.0" y1="40" x2="60.0" y2="210" class="stroke-grid"/><text x="60.0" y="226" text-anchor="middle" class="fig-small">0</text><line x1="180.0" y1="40" x2="180.0" y2="210" class="stroke-grid"/><text x="180.0" y="226" text-anchor="middle" class="fig-small">0.25</text><line x1="300.0" y1="40" x2="300.0" y2="210" class="stroke-grid"/><text x="300.0" y="226" text-anchor="middle" class="fig-small">0.5</text><line x1="420.0" y1="40" x2="420.0" y2="210" class="stroke-grid"/><text x="420.0" y="226" text-anchor="middle" class="fig-small">0.75</text><line x1="540.0" y1="40" x2="540.0" y2="210" class="stroke-grid"/><text x="540.0" y="226" text-anchor="middle" class="fig-small">1</text><line x1="60" y1="210.0" x2="540" y2="210.0" class="stroke-grid"/><text x="54" y="214.0" text-anchor="end" class="fig-small">0</text><line x1="60" y1="149.3" x2="540" y2="149.3" class="stroke-grid"/><text x="54" y="153.3" text-anchor="end" class="fig-small">0.5</text><line x1="60" y1="88.6" x2="540" y2="88.6" class="stroke-grid"/><text x="54" y="92.6" text-anchor="end" class="fig-small">1</text><polyline points="60.0,210.0 64.8,205.8 69.6,201.6 74.4,197.5 79.2,193.3 84.0,189.2 88.8,185.0 93.6,180.9 98.4,176.8 103.2,172.8 108.0,168.8 112.8,164.8 117.6,160.8 122.4,156.9 127.2,153.0 132.0,149.2 136.8,145.4 141.6,141.7 146.4,138.0 151.2,134.4 156.0,130.9 160.8,127.4 165.6,124.0 170.4,120.7 175.2,117.4 180.0,114.3 184.8,111.2 189.6,108.1 194.4,105.2 199.2,102.3 204.0,99.6 208.8,96.9 213.6,94.3 218.4,91.8 223.2,89.5 228.0,87.2 232.8,85.0 237.6,82.9 242.4,80.9 247.2,79.0 252.0,77.3 256.8,75.6 261.6,74.1 266.4,72.6 271.2,71.3 276.0,70.1 280.8,69.0 285.6,68.0 290.4,67.1 295.2,66.3 300.0,65.6 304.8,65.1 309.6,64.7 314.4,64.3 319.2,64.1 324.0,64.0 328.8,64.0 333.6,64.1 338.4,64.3 343.2,64.7 348.0,65.1 352.8,65.6 357.6,66.3 362.4,67.0 367.2,67.8 372.0,68.7 376.8,69.8 381.6,70.8 386.4,72.0 391.2,73.3 396.0,74.6 400.8,76.1 405.6,77.5 410.4,79.1 415.2,80.7 420.0,82.4 424.8,84.1 429.6,85.9 434.4,87.7 439.2,89.5 444.0,91.4 448.8,93.3 453.6,95.3 458.4,97.2 463.2,99.2 468.0,101.1 472.8,103.0 477.6,105.0 482.4,106.9 487.2,108.8 492.0,110.6 496.8,112.4 501.6,114.1 506.4,115.8 511.2,117.4 516.0,118.9 520.8,120.3 525.6,121.6 530.4,122.8 535.2,123.9 540.0,124.9" fill="none" class="stroke-muted dashed" stroke-width="2"/><polyline points="60.0,210.0 64.8,200.8 69.6,191.7 74.4,182.7 79.2,174.0 84.0,165.6 88.8,157.5 93.6,149.9 98.4,142.8 103.2,136.1 108.0,130.0 112.8,124.4 117.6,119.4 122.4,114.8 127.2,110.8 132.0,107.2 136.8,104.1 141.6,101.4 146.4,99.1 151.2,97.1 156.0,95.5 160.8,94.1 165.6,93.0 170.4,92.0 175.2,91.3 180.0,90.7 184.8,90.2 189.6,89.8 194.4,89.5 199.2,89.3 204.0,89.1 208.8,88.9 213.6,88.8 218.4,88.8 223.2,88.7 228.0,88.7 232.8,88.6 237.6,88.6 242.4,88.6 247.2,88.6 252.0,88.6 256.8,88.6 261.6,88.6 266.4,88.6 271.2,88.6 276.0,88.6 280.8,88.6 285.6,88.6 290.4,88.6 295.2,88.6 300.0,88.6 304.8,88.6 309.6,88.6 314.4,88.6 319.2,88.6 324.0,88.6 328.8,88.6 333.6,88.6 338.4,88.6 343.2,88.6 348.0,88.6 352.8,88.6 357.6,88.6 362.4,88.6 367.2,88.6 372.0,88.6 376.8,88.6 381.6,88.6 386.4,88.6 391.2,88.6 396.0,88.6 400.8,88.6 405.6,88.6 410.4,88.6 415.2,88.6 420.0,88.6 424.8,88.6 429.6,88.6 434.4,88.6 439.2,88.6 444.0,88.6 448.8,88.6 453.6,88.6 458.4,88.6 463.2,88.6 468.0,88.6 472.8,88.6 477.6,88.6 482.4,88.6 487.2,88.6 492.0,88.6 496.8,88.6 501.6,88.6 506.4,88.6 511.2,88.6 516.0,88.6 520.8,88.6 525.6,88.6 530.4,88.6 535.2,88.6 540.0,88.6" fill="none" class="stroke-a" stroke-width="2"/><polyline points="60.0,210.0 64.8,125.1 69.6,127.2 74.4,73.7 79.2,94.5 84.0,117.9 88.8,87.9 93.6,72.5 98.4,94.4 103.2,124.9 108.0,123.5 112.8,118.6 117.6,123.2 122.4,126.9 127.2,114.6 132.0,95.8 136.8,85.2 141.6,83.4 146.4,89.0 151.2,102.6 156.0,119.3 160.8,127.1 165.6,124.1 170.4,119.4 175.2,119.1 180.0,123.2 184.8,127.1 189.6,122.9 194.4,108.3 199.2,90.9 204.0,78.9 208.8,73.4 213.6,72.3 218.4,73.7 223.2,76.5 228.0,79.5 232.8,81.8 237.6,82.9 242.4,82.4 247.2,80.7 252.0,78.1 256.8,75.4 261.6,73.3 266.4,72.3 271.2,72.9 276.0,75.7 280.8,81.3 285.6,89.5 290.4,99.4 295.2,109.2 300.0,117.1 304.8,122.3 309.6,125.3 314.4,126.6 319.2,127.1 324.0,127.2 328.8,127.2 333.6,127.0 338.4,126.5 343.2,125.2 348.0,122.2 352.8,117.1 357.6,109.6 362.4,100.4 367.2,90.9 372.0,82.8 376.8,77.0 381.6,73.6 386.4,72.3 391.2,72.6 396.0,74.0 400.8,76.1 405.6,78.6 410.4,80.8 415.2,82.3 420.0,82.9 424.8,82.4 429.6,81.0 434.4,78.8 439.2,76.3 444.0,74.1 448.8,72.6 453.6,72.3 458.4,73.8 463.2,77.7 468.0,84.7 472.8,95.0 477.6,107.0 482.4,117.7 487.2,124.8 492.0,127.2 496.8,126.1 501.6,123.5 506.4,120.8 511.2,119.1 516.0,118.6 520.8,119.1 525.6,120.4 530.4,122.2 535.2,124.0 540.0,125.4" fill="none" class="stroke-b" stroke-width="2"/><text x="357.6" y="46.1" class="fig-small">tuned quintic, 5 steps</text><text x="252.0" y="143.2" class="fig-small">cubic, 5 steps</text><text x="261.6" y="185.7" class="fig-small">quintic, 1 step</text><text x="300" y="252" text-anchor="middle" class="fig-small">singular value going in (after dividing by the Frobenius norm)</text><text x="12" y="30" class="fig-small">singular value coming out</text></svg>
Figure 3. Held-out RMSE of rank-2 ALS after each sweep, against three baselines on a planted rank-2 table. ALS falls well below all three.
```

The RMSE starts near the baselines, falls steadily, and settles at 0.34, against 1.08 to 1.18 for the baselines. The noise has standard deviation 0.2, so 0.2 is the floor nobody can beat. The settings matter, and the held-out set is how you pick them:

| setting | held-out RMSE |
|---|---|
| rank 2, $\lambda = 1$ | 0.338 |
| rank 2, $\lambda = 0.1$ | 0.575 |
| rank 2, $\lambda = 5$ | 0.743 |
| rank 1, $\lambda = 1$ | 0.796 |
| rank 5, $\lambda = 1$ | 0.409 |

Too small a $\lambda$ lets the factors chase noise, and too large a $\lambda$ shrinks them toward zero. Rank 1 cannot express two independent tastes, and rank 5 has spare room that fits noise.

> [!KEY]
> Matrix factorisation replaces "find similar rows" with "describe every row by $k$ hidden numbers". Fixing one factor makes each row of the other a small ridge regression on the observed entries, so ALS alternates exact solves and never raises the masked loss. On the planted table it reaches an RMSE of 0.34 where the baselines stay above 1.0.

> [!RECAP]
> - k-NN predicts from the $k$ nearest stored rows: a vote for labels, a mean for numbers. It has no training step, and all squared distances come from one matrix product.
> - Fix a tie rule, and exclude a row from its own neighbours when you score it. Pick $k$ by leave-one-out error.
> - In high dimension the ratio of nearest to farthest distance tends to 1, so neighbours stop being informative. On a sparse ratings table, two users share too few films to compare.
> - Matrix factorisation writes $R \approx U V^\top$ and fits the observed entries only, with a ridge fee on $U$ and $V$.
> - ALS solves one $k \times k$ ridge system per row, alternating between users and films. The masked loss never rises, and held-out RMSE against the global-mean and user-bias baselines is the test.

## Build it step by step

You write ten functions, in the order the checks test them.

1. **`pairwise_sq_dists(A, B)`** returns an `(n, m)` array of squared distances. Use the expansion with one matrix product and clip at 0. The usual mistake is skipping the clip, which leaves tiny negative values.
2. **`knn_predict(Xtr, ytr, Xte, k)`** returns an integer label for every query. Take the first `k` columns of a stable `argsort`, count the labels with `np.bincount`, and take `argmax`. The usual mistake is an unstable sort, which breaks distance ties differently from run to run.
3. **`knn_regress(Xtr, ytr, Xte, k)`** returns the mean target of the same neighbours. The usual mistake is averaging over axis 0 instead of axis 1.
4. **`loo_knn_error(X, y, k)`** predicts each row from the others and returns the fraction wrong. Set the diagonal of the distance matrix to `np.inf` before sorting. The usual mistake is leaving the row in its own neighbour list, which gives an error of 0 at $k=1$.
5. **`distance_ratio(d, n, rng)`** draws `n` uniform points, then one query, and returns the smallest distance over the largest. The usual mistake is using squared distances, which exaggerates the ratio's trend.
6. **`masked_loss(R, M, U, V, lam)`** returns $J$ as a float. Multiply the residual by `M` before squaring. The usual mistake is summing the residual over the blanks.
7. **`als_step(R, M, U, V, lam)`** returns an updated `U`, one ridge solve per row, using only the films that row's user rated. The usual mistake is solving against all of `V`, which fits the blanks as zeros. A user with no ratings gets a row of zeros.
8. **`user_bias_baseline(R, M)`** and **`masked_rmse(R, M, P)`** build the table to beat and the score to beat it with. The usual mistake is dividing by the size of the table instead of by the number of observed entries.
9. **`fit_als(R, M, k, lam, iters, rng)`** starts `U` and `V` as `0.1 * rng.normal(...)`, then in each sweep updates `U`, then `V` through the transposed table, and records the loss. It returns `(U, V, history)`. The usual mistake is updating `V` with the old `U`.

## Cost and scaling

Predicting $q$ queries against $n$ stored rows of dimension $d$ costs $O(qnd)$ for the distances, and the `(q, n)` distance matrix is $q \cdot n$ numbers of memory. Training is free, but every query pays.

One ALS sweep costs about $O(N k^2 + (n + m)k^3)$, where $N$ is the number of observed entries: forming each $k \times k$ system takes $k^2$ per rating, and solving it takes $k^3$ per row. The cost grows with the number of ratings, not with the size of the full table, and every row's solve is independent.

## Where this breaks

- **Scale and noise columns hurt k-NN.** A column in dollars beats a column in fractions, and each noisy column adds distance with no signal. Standardise, then select or compress columns.
- **A user with no ratings gets zeros.** Every prediction for them is 0. This is the *cold-start* problem. Services ask for a few ratings, or fall back on popularity.
- **Choose the rank $k$ and $\lambda$ on held-out entries,** never on the entries you fit.

## The exercise

You build `pairwise_sq_dists`, `knn_predict`, `knn_regress`, `loo_knn_error`, `distance_ratio`, `masked_loss`, `als_step`, `user_bias_baseline`, `masked_rmse`, and `fit_als`. The checks reproduce the chapter's numbers: distances $2, 1, 2, 8, 13$, predictions $12$, $11$, $14$, and the hand row $17/11$. They compare `knn_predict` with a brute-force loop, test both tie rules, and check that `loo_knn_error` gives about 0.5 on random labels, which a row that votes for itself cannot do. They check that `distance_ratio` rises from below 0.2 at $d=2$ to above 0.8 at $d=1000$. For ALS, they compare `als_step` with an independent least-squares solve, confirm that changing an unobserved entry changes nothing, check that the masked loss never rises, and require held-out RMSE under 60 percent of the better baseline on a planted rank-2 table.

## Further reading

- Stanford CS231n, [Lecture 2, "Image Classification with Linear Classifiers"](https://cs231n.stanford.edu/schedule.html). The schedule lists k-nearest neighbor as a topic of that lecture.
- Stanford [CS229](https://cs229.stanford.edu/), the machine learning course. Its notes cover non-parametric methods and latent factors.
- Yehuda Koren, Robert Bell, and Chris Volinsky, "Matrix Factorization Techniques for Recommender Systems," *Computer* 42(8), 2009. Factorisation for ratings, with biases and alternating least squares.
- Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft, "When Is 'Nearest Neighbor' Meaningful?", ICDT 1999. The result that nearest and farthest distances converge.
- Jure Leskovec, Anand Rajaraman, and Jeffrey Ullman, [*Mining of Massive Datasets*](http://www.mmds.org/), chapter 9, "Recommendation Systems". User-based and item-based collaborative filtering, and the UV decomposition.
