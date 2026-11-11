---
title: Impurity, gain, and the best split
slug: 024-impurity-and-best-split
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: A tree box answers with a mean, impurity scores how badly, and gain with one sorted sweep picks the question that helps most.
minutes: 70
gpu: false
objectives:
  - Explain why the mean is the best single answer for a box under squared error, and why a 0/1 box answers with the fraction of ones.
  - Compute variance, Gini impurity, and entropy for a set of labels, and show that for 0/1 labels Gini is twice the variance.
  - Score a split by its gain, the parent impurity minus the size-weighted child impurities, and explain why the weights are needed.
  - List the thresholds worth trying on one feature, and find the best split of a feature in one sorted pass with running sums.
prereqs:
  - 001-loss-and-honest-scores
exercise: 024-impurity-and-best-split
---

# Impurity, gain, and the best split

Eight houses stand on one street. Their prices, in tens of lakhs, are 1, 2, 3, 4, 8, 9, 10, and 11. A buyer asks what a ninth house on the street would cost, and you may give only one number. Which number do you pick, and how do you say how bad that answer is?

The average is 6, and no house on the street costs anything like 6. The left half of Figure 1 shows why. The street is really two groups, cheap and dear, and a single answer lands in the gap between them. Every house misses by at least 2. If you could first ask one question that sorts the houses into the two groups, each group would get its own answer, and the misses would shrink to about a unit, as on the right.

That sorting is what a decision tree does. It asks a chain of yes-or-no questions, and each group of rows it ends with gives one answer. To choose good questions you need three things: the best single answer for a group, a score for how badly that answer fits (the *impurity*), and a number that says how much a question improves things (the *gain*). The number must also be cheap, because a tree scores thousands of questions at every group. This chapter builds all three, and the sorted sweep that makes the last one fast.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Eight house prices on a number line. Left: one answer at 6, with long misses. Right: two answers at 2.5 and 9.5, with short misses.">
<text x="20" y="22" class="fig-bold" >One box, one answer</text>
<text x="20" y="40" class="fig-small" >answer 6, impurity 13.5</text>
<line x1="20" y1="200" x2="284" y2="200" class="stroke-muted" />
<line x1="20" y1="200" x2="20" y2="204" class="stroke-muted" />
<text x="20" y="218" class="fig-small" text-anchor="middle" >0</text>
<line x1="108" y1="200" x2="108" y2="204" class="stroke-muted" />
<text x="108" y="218" class="fig-small" text-anchor="middle" >4</text>
<line x1="196" y1="200" x2="196" y2="204" class="stroke-muted" />
<text x="196" y="218" class="fig-small" text-anchor="middle" >8</text>
<line x1="284" y1="200" x2="284" y2="204" class="stroke-muted" />
<text x="284" y="218" class="fig-small" text-anchor="middle" >12</text>
<text x="152" y="238" class="fig-small" text-anchor="middle" >price, in tens of lakhs</text>
<line x1="42" y1="62" x2="152" y2="62" class="stroke-muted" />
<circle cx="42" cy="62" r="4.5" class="fill-a" />
<line x1="64" y1="78" x2="152" y2="78" class="stroke-muted" />
<circle cx="64" cy="78" r="4.5" class="fill-a" />
<line x1="86" y1="94" x2="152" y2="94" class="stroke-muted" />
<circle cx="86" cy="94" r="4.5" class="fill-a" />
<line x1="108" y1="110" x2="152" y2="110" class="stroke-muted" />
<circle cx="108" cy="110" r="4.5" class="fill-a" />
<line x1="196" y1="126" x2="152" y2="126" class="stroke-muted" />
<circle cx="196" cy="126" r="4.5" class="fill-a" />
<line x1="218" y1="142" x2="152" y2="142" class="stroke-muted" />
<circle cx="218" cy="142" r="4.5" class="fill-a" />
<line x1="240" y1="158" x2="152" y2="158" class="stroke-muted" />
<circle cx="240" cy="158" r="4.5" class="fill-a" />
<line x1="262" y1="174" x2="152" y2="174" class="stroke-muted" />
<circle cx="262" cy="174" r="4.5" class="fill-a" />
<line x1="152" y1="53" x2="152" y2="183" class="stroke-b" stroke-width="2.5"/>
<text x="316" y="22" class="fig-bold" >Two boxes, two answers</text>
<text x="316" y="40" class="fig-small" >answers 2.5 and 9.5, impurity 1.25 each</text>
<line x1="316" y1="200" x2="580" y2="200" class="stroke-muted" />
<line x1="316" y1="200" x2="316" y2="204" class="stroke-muted" />
<text x="316" y="218" class="fig-small" text-anchor="middle" >0</text>
<line x1="404" y1="200" x2="404" y2="204" class="stroke-muted" />
<text x="404" y="218" class="fig-small" text-anchor="middle" >4</text>
<line x1="492" y1="200" x2="492" y2="204" class="stroke-muted" />
<text x="492" y="218" class="fig-small" text-anchor="middle" >8</text>
<line x1="580" y1="200" x2="580" y2="204" class="stroke-muted" />
<text x="580" y="218" class="fig-small" text-anchor="middle" >12</text>
<text x="448" y="238" class="fig-small" text-anchor="middle" >price, in tens of lakhs</text>
<line x1="338" y1="62" x2="371" y2="62" class="stroke-muted" />
<circle cx="338" cy="62" r="4.5" class="fill-a" />
<line x1="360" y1="78" x2="371" y2="78" class="stroke-muted" />
<circle cx="360" cy="78" r="4.5" class="fill-a" />
<line x1="382" y1="94" x2="371" y2="94" class="stroke-muted" />
<circle cx="382" cy="94" r="4.5" class="fill-a" />
<line x1="404" y1="110" x2="371" y2="110" class="stroke-muted" />
<circle cx="404" cy="110" r="4.5" class="fill-a" />
<line x1="492" y1="126" x2="525" y2="126" class="stroke-muted" />
<circle cx="492" cy="126" r="4.5" class="fill-a" />
<line x1="514" y1="142" x2="525" y2="142" class="stroke-muted" />
<circle cx="514" cy="142" r="4.5" class="fill-a" />
<line x1="536" y1="158" x2="525" y2="158" class="stroke-muted" />
<circle cx="536" cy="158" r="4.5" class="fill-a" />
<line x1="558" y1="174" x2="525" y2="174" class="stroke-muted" />
<circle cx="558" cy="174" r="4.5" class="fill-a" />
<line x1="371" y1="53" x2="371" y2="119" class="stroke-b" stroke-width="2.5"/>
<line x1="525" y1="117" x2="525" y2="183" class="stroke-b" stroke-width="2.5"/>
</svg>
Figure 1. The same eight houses, with each miss drawn as a line to the answer (orange). With one answer, every house misses by 2 or more and the impurity is 13.5. After one question that separates cheap from dear, the lines are short and the impurity is 1.25 in each box.
```

## What you already know

- A loss scores one guess against the truth. The squared miss of a guess $g$ for a true value $y$ is $(y - g)^2$, as in [loss and honest scores](/lessons/001-loss-and-honest-scores).
- The average loss over many rows is the score of a rule, so a smaller average means a better rule.
- A yes-or-no label can be stored as a number: 1 for yes and 0 for no, and the fraction of ones is a fair estimate of the probability of yes.
- Sorting $n$ values costs about $n \log n$ steps, and a loop inside a loop over $n$ rows costs about $n^2$ steps.

## One box, one answer

A tree ends in *leaves*. A leaf owns a *box*, the group of training rows that reached it, and it gives the same answer to every row that lands there. So the first question is which single number to give for a box.

Score the answer by the squared miss, because it punishes a far miss much more than a near one. Take two houses priced 1 and 3, and try three answers $c$. Answer 0 misses by 1 and 3, so the total squared error is $1 + 9 = 10$. Answer 2 misses by 1 and 1, so the total is $1 + 1 = 2$. Answer 4 costs $9 + 1 = 10$. Plotted against $c$, the total is a bowl, and its bottom is the average, 2. Each house pulls the answer toward itself with a strength equal to its distance, and at 2 the two pulls cancel. In symbols, the slope of $\sum_i (y_i - c)^2$ with respect to $c$ is $-2\sum_i (y_i - c)$, which is zero exactly when $c$ is the mean:

$$
\text{value} = \frac{1}{m}\sum_{i=1}^{m} y_i .
$$

Read it aloud: add up the $m$ labels in the box and divide by $m$. In numpy that is `float(np.mean(y))`.

The same rule covers classes. Store "expensive" as 1 and everything else as 0. A box of four houses with three expensive ones has mean $0.75$, the fraction of ones, which you can read as the probability that a house in this box is expensive. Squared error agrees that 0.75 is the best answer: it costs $3 \times 0.25^2 + 0.75^2 = 0.75$, while answering 1 costs 1 and answering 0 costs 3.

> [!KEY]
> A leaf answers with the mean of the labels in its box. For 0/1 labels, the mean is the fraction of ones.

## How bad is one answer?

A box is *pure* when every label in it is the same, so one answer fits perfectly. *Impurity* measures how far a box is from pure. It is 0 for a pure box and grows as the labels spread out.

For numbers, the natural measure is the average squared miss at the leaf's answer. That average has a name, the *variance*:

$$
V = \frac{1}{m}\sum_{i=1}^{m} (y_i - \bar y)^2 ,
$$

where $\bar y$ is the mean of the box. For the eight houses the mean is 6, the gaps are $-5, -4, -3, -2, 2, 3, 4, 5$, and the squared gaps sum to 108, so $V = 108/8 = 13.5$. For the cheap four alone the mean is 2.5, the squared gaps are $2.25, 0.25, 0.25, 2.25$, they sum to 5, and $V = 1.25$. The dear four give 1.25 as well. One question dropped the impurity from 13.5 to 1.25, which is the whole case for asking it.

In numpy, write `np.mean((y - y.mean()) ** 2)`. The usual mistake is to forget the division, which returns 108.

## Impurity for classes

Class labels are names, not distances, so the squared miss makes no sense for three classes. You need a score that is zero for a pure box and larger for a mixed one. Draw two houses from the box at random, putting the first one back before the second draw, and ask whether their labels agree. The chance that they disagree is *Gini impurity*: 0 in a pure box, and larger the more mixed the box is.

Take the six labels $[0, 0, 1, 1, 1, 2]$. The fractions are $2/6$, $3/6$, and $1/6$. Both draws are class 0 with chance $(2/6)^2 = 4/36$, class 1 with chance $9/36$, and class 2 with chance $1/36$. So the draws agree with chance $14/36$ and disagree with chance $1 - 14/36 = 22/36 \approx 0.611$. With $p_k$ the fraction of labels in class $k$,

$$
G = 1 - \sum_k p_k^2 .
$$

Read it aloud: one minus the chance that two draws match. A pure box gives $1 - 1 = 0$, and four equal classes give $1 - 4 \times (1/4)^2 = 0.75$. In numpy, take `p = np.bincount(y.astype(int)) / y.size` and return `1 - np.sum(p ** 2)`. The usual mistake is to divide by the number of classes, not by the number of labels.

> [!WARNING]
> `np.bincount` accepts only non-negative whole numbers. Labels stored as floats, such as `1.0`, need `astype(int)` first.

The third score asks how surprised you are by a label drawn from the box. A label with probability $p$ has surprise $\log(1/p)$: a certain label surprises you by 0, and a rare label surprises you a lot. Averaging over the labels gives *entropy*:

$$
H = -\sum_k p_k \log p_k .
$$

Read it aloud: each class contributes its fraction times the log of one over its fraction. The log is the natural log, so the unit is nats. For $[0, 0, 1, 1, 1, 2]$ the three terms are $0.3662 + 0.3466 + 0.2986$, so $H \approx 1.0114$. A pure box gives 0, and four equal classes give $\log 4 \approx 1.3863$. In numpy, drop the zero counts first (the convention $0 \log 0 = 0$), then compute `-np.sum(p * np.log(p))`.

Gini and entropy usually choose the same questions. They differ in shape, as the curves in Figure 2 show: entropy climbs faster when a nearly pure box picks up a stray label. At $p = 0.01$, Gini is 0.0198 and entropy is 0.0560.

## Three scores, one idea

For 0/1 labels with a fraction $p$ of ones, the Gini impurity is $1 - p^2 - (1-p)^2 = 2p(1-p)$. The variance is just as short. Because $y^2 = y$ for a 0 or a 1, the mean of the squares is $p$, so $V = p - p^2 = p(1-p)$. Put them together: **for 0/1 labels, Gini is exactly twice the variance.** Check it on the box $[0, 0, 0, 1]$: $p = 0.25$, $V = 0.1875$, and $G = 0.375$.

So prices and yes-or-no labels are measured by the same idea, how much one answer misses, and the same leaf rule, the mean, serves both.

### Why not count mistakes?

The obvious impurity is the misclassification rate: how many rows the box's majority answer gets wrong. It equals $\min(p, 1-p)$, and it fails in an instructive way. Take eight houses, six expensive and two not. The box answers "expensive" and gets $2/8$ wrong. Split it into a pure box of four expensive houses and a box of two expensive and two not. The pure box makes 0 mistakes, and the mixed box answers "expensive" on a tie and makes 2. The split still makes 2 mistakes out of 8, so the error rate says it achieved nothing, although it isolated four houses perfectly.

Gini sees the progress. The parent has $G = 0.375$, and the children have $0$ and $0.5$. Each holds half the rows, so the weighted average is $0.25$, a fall of $0.125$. Entropy falls from $0.562$ to $0.347$. The error rate is a straight line on each side of $p = 1/2$, so when both children share the parent's majority, the weighted average of their $p$ values lands exactly on the parent's value. Gini and entropy bend, so any move toward purity shows up.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Three impurity scores for a two-class box, against the fraction p of class 1. Gini and entropy are zero at p = 0 and p = 1 and peak at p = 1/2. Misclassification rate has sharp corners.">
<line x1="70" y1="230" x2="540" y2="230" class="stroke-grid" />
<text x="62" y="234" class="fig-small" text-anchor="end" >0</text>
<line x1="70" y1="166.7" x2="540" y2="166.7" class="stroke-grid" />
<text x="62" y="170.7" class="fig-small" text-anchor="end" >0.25</text>
<line x1="70" y1="103.3" x2="540" y2="103.3" class="stroke-grid" />
<text x="62" y="107.3" class="fig-small" text-anchor="end" >0.5</text>
<line x1="70" y1="40" x2="540" y2="40" class="stroke-grid" />
<text x="62" y="44" class="fig-small" text-anchor="end" >0.75</text>
<line x1="70" y1="230" x2="540" y2="230" class="stroke-muted" />
<line x1="70" y1="230" x2="70" y2="234" class="stroke-muted" />
<text x="70" y="250" class="fig-small" text-anchor="middle" >0</text>
<line x1="187.5" y1="230" x2="187.5" y2="234" class="stroke-muted" />
<text x="187.5" y="250" class="fig-small" text-anchor="middle" >0.25</text>
<line x1="305" y1="230" x2="305" y2="234" class="stroke-muted" />
<text x="305" y="250" class="fig-small" text-anchor="middle" >0.5</text>
<line x1="422.5" y1="230" x2="422.5" y2="234" class="stroke-muted" />
<text x="422.5" y="250" class="fig-small" text-anchor="middle" >0.75</text>
<line x1="540" y1="230" x2="540" y2="234" class="stroke-muted" />
<text x="540" y="250" class="fig-small" text-anchor="middle" >1</text>
<text x="305" y="274" class="fig-small" text-anchor="middle" >fraction of class 1 in the box, p</text>
<polyline points="70,230 72.3,228.7 74.7,227.5 77,226.2 79.4,224.9 81.8,223.7 84.1,222.4 86.5,221.1 88.8,219.9 91.2,218.6 93.5,217.3 95.8,216.1 98.2,214.8 100.5,213.5 102.9,212.3 105.2,211 107.6,209.7 110,208.5 112.3,207.2 114.7,205.9 117,204.7 119.3,203.4 121.7,202.1 124.1,200.9 126.4,199.6 128.8,198.3 131.1,197.1 133.4,195.8 135.8,194.5 138.1,193.3 140.5,192 142.8,190.7 145.2,189.5 147.6,188.2 149.9,186.9 152.2,185.7 154.6,184.4 156.9,183.1 159.3,181.9 161.7,180.6 164,179.3 166.4,178.1 168.7,176.8 171.1,175.5 173.4,174.3 175.8,173 178.1,171.7 180.4,170.5 182.8,169.2 185.1,167.9 187.5,166.7 189.9,165.4 192.2,164.1 194.6,162.9 196.9,161.6 199.2,160.3 201.6,159.1 204,157.8 206.3,156.5 208.7,155.3 211,154 213.3,152.7 215.7,151.5 218.1,150.2 220.4,148.9 222.8,147.7 225.1,146.4 227.5,145.1 229.8,143.9 232.2,142.6 234.5,141.3 236.8,140.1 239.2,138.8 241.5,137.5 243.9,136.3 246.2,135 248.6,133.7 251,132.5 253.3,131.2 255.7,129.9 258,128.7 260.4,127.4 262.7,126.1 265.1,124.9 267.4,123.6 269.8,122.3 272.1,121.1 274.4,119.8 276.8,118.5 279.1,117.3 281.5,116 283.9,114.7 286.2,113.5 288.6,112.2 290.9,110.9 293.2,109.7 295.6,108.4 297.9,107.1 300.3,105.9 302.6,104.6 305,103.3 307.4,104.6 309.7,105.9 312.1,107.1 314.4,108.4 316.8,109.7 319.1,110.9 321.5,112.2 323.8,113.5 326.2,114.7 328.5,116 330.9,117.3 333.2,118.5 335.6,119.8 337.9,121.1 340.3,122.3 342.6,123.6 344.9,124.9 347.3,126.1 349.6,127.4 352,128.7 354.3,129.9 356.7,131.2 359.1,132.5 361.4,133.7 363.8,135 366.1,136.3 368.4,137.5 370.8,138.8 373.2,140.1 375.5,141.3 377.9,142.6 380.2,143.9 382.6,145.1 384.9,146.4 387.2,147.7 389.6,148.9 392,150.2 394.3,151.5 396.7,152.7 399,154 401.3,155.3 403.7,156.5 406.1,157.8 408.4,159.1 410.8,160.3 413.1,161.6 415.4,162.9 417.8,164.1 420.1,165.4 422.5,166.7 424.9,167.9 427.2,169.2 429.6,170.5 431.9,171.7 434.2,173 436.6,174.3 438.9,175.5 441.3,176.8 443.7,178.1 446,179.3 448.4,180.6 450.7,181.9 453.1,183.1 455.4,184.4 457.8,185.7 460.1,186.9 462.4,188.2 464.8,189.5 467.1,190.7 469.5,192 471.8,193.3 474.2,194.5 476.6,195.8 478.9,197.1 481.2,198.3 483.6,199.6 485.9,200.9 488.3,202.1 490.7,203.4 493,204.7 495.4,205.9 497.7,207.2 500.1,208.5 502.4,209.7 504.8,211 507.1,212.3 509.5,213.5 511.8,214.8 514.2,216.1 516.5,217.3 518.9,218.6 521.2,219.9 523.5,221.1 525.9,222.4 528.2,223.7 530.6,224.9 533,226.2 535.3,227.5 537.6,228.7 540,230" fill="none" class="stroke-d dashed" />
<polyline points="70,230 72.3,222 74.7,215.8 77,210.3 79.4,205.2 81.8,200.4 84.1,195.9 86.5,191.6 88.8,187.5 91.2,183.5 93.5,179.7 95.8,176 98.2,172.5 100.5,169.1 102.9,165.7 105.2,162.5 107.6,159.4 110,156.3 112.3,153.4 114.7,150.5 117,147.6 119.3,144.9 121.7,142.2 124.1,139.6 126.4,137 128.8,134.6 131.1,132.1 133.4,129.7 135.8,127.4 138.1,125.1 140.5,122.9 142.8,120.7 145.2,118.6 147.6,116.5 149.9,114.5 152.2,112.5 154.6,110.6 156.9,108.7 159.3,106.8 161.7,105 164,103.2 166.4,101.5 168.7,99.8 171.1,98.1 173.4,96.5 175.8,94.9 178.1,93.4 180.4,91.9 182.8,90.4 185.1,89 187.5,87.5 189.9,86.2 192.2,84.8 194.6,83.5 196.9,82.2 199.2,81 201.6,79.8 204,78.6 206.3,77.5 208.7,76.3 211,75.2 213.3,74.2 215.7,73.2 218.1,72.2 220.4,71.2 222.8,70.3 225.1,69.3 227.5,68.5 229.8,67.6 232.2,66.8 234.5,66 236.8,65.2 239.2,64.5 241.5,63.8 243.9,63.1 246.2,62.4 248.6,61.8 251,61.2 253.3,60.6 255.7,60 258,59.5 260.4,59 262.7,58.5 265.1,58.1 267.4,57.7 269.8,57.3 272.1,56.9 274.4,56.5 276.8,56.2 279.1,55.9 281.5,55.7 283.9,55.4 286.2,55.2 288.6,55 290.9,54.9 293.2,54.7 295.6,54.6 297.9,54.5 300.3,54.5 302.6,54.4 305,54.4 307.4,54.4 309.7,54.5 312.1,54.5 314.4,54.6 316.8,54.7 319.1,54.9 321.5,55 323.8,55.2 326.2,55.4 328.5,55.7 330.9,55.9 333.2,56.2 335.6,56.5 337.9,56.9 340.3,57.3 342.6,57.7 344.9,58.1 347.3,58.5 349.6,59 352,59.5 354.3,60 356.7,60.6 359.1,61.2 361.4,61.8 363.8,62.4 366.1,63.1 368.4,63.8 370.8,64.5 373.2,65.2 375.5,66 377.9,66.8 380.2,67.6 382.6,68.5 384.9,69.3 387.2,70.3 389.6,71.2 392,72.2 394.3,73.2 396.7,74.2 399,75.2 401.3,76.3 403.7,77.5 406.1,78.6 408.4,79.8 410.8,81 413.1,82.2 415.4,83.5 417.8,84.8 420.1,86.2 422.5,87.5 424.9,89 427.2,90.4 429.6,91.9 431.9,93.4 434.2,94.9 436.6,96.5 438.9,98.1 441.3,99.8 443.7,101.5 446,103.2 448.4,105 450.7,106.8 453.1,108.7 455.4,110.6 457.8,112.5 460.1,114.5 462.4,116.5 464.8,118.6 467.1,120.7 469.5,122.9 471.8,125.1 474.2,127.4 476.6,129.7 478.9,132.1 481.2,134.6 483.6,137 485.9,139.6 488.3,142.2 490.7,144.9 493,147.6 495.4,150.5 497.7,153.4 500.1,156.3 502.4,159.4 504.8,162.5 507.1,165.7 509.5,169.1 511.8,172.5 514.2,176 516.5,179.7 518.9,183.5 521.2,187.5 523.5,191.6 525.9,195.9 528.2,200.4 530.6,205.2 533,210.3 535.3,215.8 537.6,222 540,230" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70,230 72.3,227.5 74.7,225 77,222.5 79.4,220.1 81.8,217.7 84.1,215.3 86.5,212.9 88.8,210.5 91.2,208.2 93.5,205.9 95.8,203.7 98.2,201.4 100.5,199.2 102.9,197 105.2,194.8 107.6,192.7 110,190.6 112.3,188.5 114.7,186.4 117,184.4 119.3,182.4 121.7,180.4 124.1,178.4 126.4,176.5 128.8,174.6 131.1,172.7 133.4,170.8 135.8,169 138.1,167.2 140.5,165.4 142.8,163.6 145.2,161.9 147.6,160.2 149.9,158.5 152.2,156.8 154.6,155.2 156.9,153.6 159.3,152 161.7,150.5 164,148.9 166.4,147.4 168.7,145.9 171.1,144.5 173.4,143.1 175.8,141.6 178.1,140.3 180.4,138.9 182.8,137.6 185.1,136.3 187.5,135 189.9,133.7 192.2,132.5 194.6,131.3 196.9,130.1 199.2,129 201.6,127.9 204,126.8 206.3,125.7 208.7,124.6 211,123.6 213.3,122.6 215.7,121.6 218.1,120.7 220.4,119.7 222.8,118.8 225.1,118 227.5,117.1 229.8,116.3 232.2,115.5 234.5,114.7 236.8,114 239.2,113.3 241.5,112.6 243.9,111.9 246.2,111.2 248.6,110.6 251,110 253.3,109.5 255.7,108.9 258,108.4 260.4,107.9 262.7,107.4 265.1,107 267.4,106.6 269.8,106.2 272.1,105.8 274.4,105.5 276.8,105.2 279.1,104.9 281.5,104.6 283.9,104.4 286.2,104.1 288.6,104 290.9,103.8 293.2,103.7 295.6,103.5 297.9,103.4 300.3,103.4 302.6,103.3 305,103.3 307.4,103.3 309.7,103.4 312.1,103.4 314.4,103.5 316.8,103.7 319.1,103.8 321.5,104 323.8,104.1 326.2,104.4 328.5,104.6 330.9,104.9 333.2,105.2 335.6,105.5 337.9,105.8 340.3,106.2 342.6,106.6 344.9,107 347.3,107.4 349.6,107.9 352,108.4 354.3,108.9 356.7,109.5 359.1,110 361.4,110.6 363.8,111.2 366.1,111.9 368.4,112.6 370.8,113.3 373.2,114 375.5,114.7 377.9,115.5 380.2,116.3 382.6,117.1 384.9,118 387.2,118.9 389.6,119.7 392,120.7 394.3,121.6 396.7,122.6 399,123.6 401.3,124.6 403.7,125.7 406.1,126.8 408.4,127.9 410.8,129 413.1,130.1 415.4,131.3 417.8,132.5 420.1,133.7 422.5,135 424.9,136.3 427.2,137.6 429.6,138.9 431.9,140.3 434.2,141.7 436.6,143.1 438.9,144.5 441.3,145.9 443.7,147.4 446,148.9 448.4,150.5 450.7,152 453.1,153.6 455.4,155.2 457.8,156.9 460.1,158.5 462.4,160.2 464.8,161.9 467.1,163.6 469.5,165.4 471.8,167.2 474.2,169 476.6,170.8 478.9,172.7 481.2,174.6 483.6,176.5 485.9,178.4 488.3,180.4 490.7,182.4 493,184.4 495.4,186.4 497.7,188.5 500.1,190.6 502.4,192.7 504.8,194.9 507.1,197 509.5,199.2 511.8,201.4 514.2,203.7 516.5,205.9 518.9,208.2 521.2,210.5 523.5,212.9 525.9,215.3 528.2,217.6 530.6,220.1 533,222.5 535.3,225 537.6,227.5 540,230" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="305" cy="103.3" r="4.5" class="fill-a" />
<circle cx="305" cy="54.4" r="4.5" class="fill-b" />
<text x="295" y="48.4" class="fig-small" text-anchor="end" >entropy peaks at 0.693</text>
<text x="76" y="30" class="fig-small" >impurity</text>
<line x1="372" y1="50" x2="400" y2="50" class="stroke-a" stroke-width="2.5"/>
<text x="408" y="54" class="fig-small" >Gini, 2p(1-p)</text>
<line x1="372" y1="70" x2="400" y2="70" class="stroke-b" stroke-width="2.5"/>
<text x="408" y="74" class="fig-small" >entropy</text>
<line x1="372" y1="90" x2="400" y2="90" class="stroke-d dashed" />
<text x="408" y="94" class="fig-small" >misclassification, min(p, 1-p)</text>
</svg>
Figure 2. All three scores are zero for a pure box and highest at p = 1/2. Gini and entropy are smooth bowls turned upside down. The misclassification rate is two straight lines, so a split that keeps the parent's majority on both sides cannot lower it.
```

> [!KEY]
> Impurity says how much one answer misses for a box: the average squared miss for numbers, the chance of a disagreement (Gini) or the average surprise (entropy) for classes. A split is worth making when it leaves the boxes purer.

## A good question leaves purer boxes

Four houses have areas 1, 2, 3, and 4 (in tens of square metres) and prices 1, 3, 8, and 10. You may ask one question of the form "is the area at most $t$?". A cut at $t$ sends every house with area at most $t$ to the left box and the rest to the right box, and each box answers with its own mean. There are three places to put the cut: between 1 and 2, between 2 and 3, and between 3 and 4. The middle cut looks best, because it puts the two cheap houses together and the two dear ones together, but a computer needs a number.

Ask how much the impurity fell. The four prices have mean 5.5, and the squared gaps $20.25, 6.25, 6.25, 20.25$ sum to 53, so the parent variance is $53/4 = 13.25$. At the cut 2.5 the left prices 1 and 3 have variance 1, and so do the right prices 8 and 10. The impurity fell from 13.25 to 1.

Do not average the two child impurities, because that rates a one-row box as a triumph. Figure 3 shows ten rows, five of each class. Peeling off a single pure row looks great under a plain average, because one child scores 0. Weight each child by its share of the rows instead, so the one-row box counts for one tenth of the answer. The peel-off then has a drop of $0.5 - 0.9 \times 0.494 = 0.056$, while a real split that leaves four-to-one mixes on both sides keeps 0.18.

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
Figure 3. Ten rows, five of each class. Peeling off a single pure row looks great if you average the child impurities, because one child scores 0. Weighting each child by its share of the rows shows that the cut barely helps, and the real split wins.
```

With $n_L$ and $n_R$ rows on the two sides and $n = n_L + n_R$,

$$
\text{gain} = I_{\text{parent}} - \frac{n_L}{n}\, I_L - \frac{n_R}{n}\, I_R .
$$

Read it aloud: the parent's impurity, minus the impurity you still have on average once every row sits in its child box. Here $I$ is whichever impurity you chose. On the four houses, the weighted child impurity at the cut 2.5 is $\tfrac24 \cdot 1 + \tfrac24 \cdot 1 = 1$, so the gain is $13.25 - 1 = 12.25$. The other two cuts work out as follows.

| Cut | Left prices | Right prices | Left variance | Right variance | Gain |
|---|---|---|---|---|---|
| 1.5 | 1 | 3, 8, 10 | 0 | 8.667 | 6.75 |
| 2.5 | 1, 3 | 8, 10 | 1 | 1 | 12.25 |
| 3.5 | 1, 3, 8 | 10 | 8.667 | 0 | 6.75 |

In numpy, apply a boolean mask: `y[left_mask]` and `y[~left_mask]` give the two boxes, and each box's impurity comes from the functions above. The usual mistake is to forget the weights.

> [!KEY]
> The best question is the one with the largest gain, and the gain weights each child by its share of the rows. Without the weights, a one-row box looks like a triumph.

## Which thresholds to try

A threshold is a number $t$, and a row goes left when its value is at most $t$. Look at the sorted areas 2 and 3. Any $t$ from 2 up to, but not including, 3 sends the row with area 2 left and the row with area 3 right. Training data cannot tell these thresholds apart, because they all make the same split. So you try one, and the middle, 2.5, is the natural pick: it sits as far as possible from both neighbours. Two rows with the same value can never be separated, because every threshold sends both to the same side. So take the midpoint between each pair of neighbours that differ, and skip the pairs that are equal.

## Score every cut in one pass

With $n$ rows there are up to $n - 1$ cuts. Scoring one cut from scratch means visiting every row to build the two boxes, so all the cuts cost about $n^2$. For 1,000 rows that is about 999,000 visits for one feature in one box, and a tree repeats it for every feature and at every box.

The way out is to sort the feature once and slide the cut from left to right. Moving the cut past one row moves that row from the right box to the left box, and nothing else changes. So you keep running totals of the left box, and each cut costs one update. Sorting costs about $n \log n$, about 10,000 steps for 1,000 rows, plus one pass of about 1,000 updates. That is roughly 90 times less work.

For 0/1 labels you only need a running count of ones, because $p$ is the count divided by the rows. For numbers, the variance seems to need the mean first, and the mean changes at every cut. But the variance can be rewritten with two running totals. Expand the squared gap:

$$
V = \frac{1}{n}\sum_i (y_i - \bar y)^2 = \frac{Q}{n} - \left(\frac{S}{n}\right)^2 ,
$$

where $S = \sum_i y_i$ is the sum of the labels and $Q = \sum_i y_i^2$ is the sum of their squares. Read it aloud: the mean of the squares minus the square of the mean. For the two prices 1 and 3, $S = 4$ and $Q = 10$, so $V = 10/2 - 2^2 = 1$, which matches. The right box needs no new work, because its sums are the totals minus the left sums.

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
Figure 4. Sorting once lines the cuts up in order. The two equal values at x = 2 give no cut, because no threshold can pull them apart. Each cut's left-side totals come from the cut before it plus one row.
```

Figure 4 shows the sweep on six sorted rows. On the four houses, the running sums of the prices are $[1, 4, 12, 22]$ and the running sums of the squares are $[1, 10, 74, 174]$. At the cut 2.5, the left box has $n = 2$, $S = 4$, and $Q = 10$, so $V_L = 1$. The right box has $S = 22 - 4 = 18$ and $Q = 174 - 10 = 164$, so $V_R = 164/2 - 9^2 = 1$. The gain is 12.25 again, with no pass over the rows.

In numpy, the running totals are `np.cumsum`. Entry `i` of a cumulative sum covers `i + 1` rows, so the left counts are `np.arange(1, n)`, not `np.arange(n - 1)`.

> [!WARNING]
> $Q/n - (S/n)^2$ subtracts two big numbers that nearly cancel when the labels sit far from zero. For the prices $10^8 + 1$ and $10^8 + 3$, double precision returns 0.0 where the true variance is 1. Variance does not change when every label shifts by the same amount, so subtract the mean of $y$ first.

> [!KEY]
> A box answers with its mean, and impurity says how badly. A question's gain is the parent's impurity minus the size-weighted impurities of its two children, and one sorted pass scores every cut on a feature. So ask the question with the largest gain, and answer with a mean on each side.

## Where this breaks

- **More than two classes.** The mean of the labels 0, 1, and 2 is not an answer, because "class 1.4" means nothing. The leaf rule here works for numbers and for 0/1 labels, and the sweep keeps a count of ones, so Gini and entropy need 0/1 labels. For several classes a leaf would store the whole list of fractions.
- **Many distinct values.** A feature with a different value in every row offers the most cuts, so it has the most chances to win by luck. [Growing and pruning trees](/lessons/025-grow-and-prune-trees) limits tree size to protect against this.
- **Tiny boxes.** Under squared error, a cut that isolates one extreme row can win, because that row inflates the variance. A minimum number of rows per leaf, `min_samples_leaf`, rules these cuts out.

> [!RECAP]
> - A leaf answers with the mean of its box's labels, because the mean is the bottom of the squared-error bowl. For 0/1 labels it is the fraction of ones.
> - Variance is the average squared miss, Gini is $1 - \sum p_k^2$, and entropy is $-\sum p_k \log p_k$. For 0/1 labels Gini is twice the variance.
> - A split's gain is the parent impurity minus the child impurities, each weighted by its share of the rows.
> - Try one threshold, the midpoint, between each pair of distinct sorted neighbours.
> - Sort once and slide the cut, keeping running sums. Variance is $Q/n - (S/n)^2$, so centre the labels first to protect the digits.

## Build it step by step

Build the six functions in the order the checks test them. `leaf_value` and `node_impurity` are written for you, along with a small helper that turns a fraction of ones into an impurity.

1. **`gini(y)`**. Takes labels of shape `(n,)` and returns a float. Divide the `np.bincount` counts by `y.size` and return one minus the sum of the squared fractions, with `0.0` for an empty array. The common mistake is dividing by the number of classes. The fractions look like this:

```python
p = np.bincount(y.astype(int)) / y.size  # shape (K,), sums to 1
```

2. **`entropy(y)`**. Same input, one float. Keep only the nonzero counts before the log. The common mistake is the log of a zero fraction, which returns `nan`. The kept fractions look like this:

```python
counts = np.bincount(y.astype(int))
p = counts[counts > 0] / y.size  # skips classes with no labels
```

3. **`variance_impurity(y)`**. Takes numbers of shape `(n,)` and returns the mean squared gap to the mean, with `0.0` for an empty array. The common mistake is the sum without the division. The gaps look like this:

```python
gaps = y - y.mean()  # shape (n,), adds up to 0
```

4. **`split_gain(y, left_mask, criterion)`**. Takes labels, a boolean mask of shape `(n,)`, and a criterion name, and returns a float. Subtract the weighted impurities of `y[left_mask]` and `y[~left_mask]` from `node_impurity(y, criterion)`. Return `0.0` when a side is empty. The common mistake is leaving out the weights. The weighted part looks like this:

```python
weighted = (n_left * imp_left + n_right * imp_right) / n  # one float
```

5. **`candidate_thresholds(x, min_samples_leaf)`**. Takes one feature of shape `(n,)` and returns the kept midpoints in increasing order. Sort with `np.sort`, take the midpoint of each neighbouring pair, and keep a pair when its two values differ and both sides keep at least `min_samples_leaf` rows. The common mistake is keeping a midpoint between equal values. The test for validity looks like this:

```python
valid = xs[:-1] != xs[1:]  # shape (n - 1,), False between equal values
```

6. **`best_split(X, y, criterion, min_samples_leaf)`**. Takes `X` of shape `(n, d)` and returns `(feature, threshold, gain)` or `None`. For each feature, sort once, get the thresholds from `candidate_thresholds`, and read the left counts off with `np.searchsorted(xs, thresholds, side="right")`. Index the cumulative sums with `n_left - 1`. The common mistake is recomputing impurities in a loop, which works but is slow. The left sums look like this:

```python
left_sum = np.cumsum(ys)[n_left - 1]  # entry k covers n_left[k] rows
```

## The exercise

You implement `gini`, `entropy`, `variance_impurity`, `split_gain`, `candidate_thresholds`, and `best_split`. The functions `leaf_value` and `node_impurity` are given.

The checks confirm the numbers in this chapter: Gini of $[0, 0, 1, 1, 1, 2]$ is $22/36$, entropy of a four-way tie is $\log 4$, the variance of the eight houses is 13.5, and Gini of a 0/1 box is twice its variance. The three four-house cuts gain 6.75, 12.25, and 6.75, and a one-row peel-off is a weak split. `best_split` finds the cut at 2.5, agrees with a slow search that rescores every cut from scratch, returns `None` when no cut is valid, and survives labels with a huge offset.
