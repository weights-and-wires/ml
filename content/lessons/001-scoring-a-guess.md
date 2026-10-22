---
title: "Scoring a guess: loss and empirical risk"
slug: 001-scoring-a-guess
part: "Part 1 — Foundations"
summary: Turn "how wrong was that guess?" into one number, so you can compare rules and pick the better one.
minutes: 35
gpu: false
objectives:
  - Turn a miss into a score with squared loss, absolute loss, or zero-one loss, and say what each one cares about.
  - Average per-example scores into the empirical risk, and say why you average instead of add.
  - Pick the best value of a rule's knob by scanning candidates and comparing their empirical risk.
  - Explain why an empirical risk is a fair estimate of future error only for a rule chosen without looking at those examples.
exercise: 001-scoring-a-guess
---

# Scoring a guess: loss and empirical risk

You collect four flats. A 20 square metre flat rents for £200 a week, a 30 square metre flat for £300, and a 40 square metre flat for £400. The rule "weekly rent equals ten times the size" matches all three. Then you meet a 50 square metre flat. The rule says £500, and the landlord asks £650.

Is the rule bad? Would "eleven times the size" be better? You cannot answer by looking at the rule. You can only answer if each rule produces one number that says how wrong it was, so that you can put two numbers side by side and pick the smaller. Without that number, with forty flats and five candidate rules, you would argue about each flat separately and never finish.

This chapter builds that number in three steps: score one guess, average the scores over many flats, and use the average to choose between rules. Every later chapter in the course, from fitting lines to training neural networks, uses a score like this.

## What you already know

You already use these ideas:

- The average of 2, 4, and 6 is 4.
- An error of 2, squared, is 4. Two errors of 1, squared and added, make 2. Squaring makes big errors count for more.
- numpy subtracts two arrays entry by entry, so `a - b` gives one difference per position.
- A function is a value you can pass around, for example `lambda p, t: (p - t) ** 2`.
- A rule such as `rent = 10 * size` is a function with one knob. Here the knob is the number 10.

## Turn a miss into a score

Start where the opening left off. Score the rule "eleven times the size" on your four flats. The rule guesses £220, £330, £440, and £550, and the real rents are £200, £300, £400, and £650. Figure 1 draws each guess in orange and each real rent in blue. The red bracket under each pair is the miss.

```figure
<svg viewBox="0 0 600 400" role="img" aria-label="Four guesses of the slope-11 rent rule against four real rents, with signed misses and squared misses drawn as squares">
<text x="8.0" y="22.0" class="fig-bold">Guess (orange) and real rent (blue), in £ per week</text>
<text x="400.0" y="22.0" class="fig-bold">Squared miss, as an area</text>
<line x1="60.0" y1="75.0" x2="340.0" y2="75.0" class="stroke-grid" />
<line x1="85.5" y1="88.0" x2="95.6" y2="88.0" class="stroke-bad" />
<line x1="85.5" y1="84.0" x2="85.5" y2="92.0" class="stroke-bad" />
<line x1="95.6" y1="84.0" x2="95.6" y2="92.0" class="stroke-bad" />
<circle cx="85.5" cy="75.0" r="5" class="fill-a"/>
<circle cx="95.6" cy="75.0" r="5" class="fill-b"/>
<text x="8.0" y="79.0">Flat 1</text>
<text x="350.0" y="79.0" class="fig-mono">+20</text>
<rect x="430.0" y="69.0" width="12.0" height="12.0" class="fill-soft-b stroke-b" />
<text x="450.0" y="79.0" class="fig-mono">400</text>
<line x1="60.0" y1="147.0" x2="340.0" y2="147.0" class="stroke-grid" />
<line x1="136.4" y1="160.0" x2="151.6" y2="160.0" class="stroke-bad" />
<line x1="136.4" y1="156.0" x2="136.4" y2="164.0" class="stroke-bad" />
<line x1="151.6" y1="156.0" x2="151.6" y2="164.0" class="stroke-bad" />
<circle cx="136.4" cy="147.0" r="5" class="fill-a"/>
<circle cx="151.6" cy="147.0" r="5" class="fill-b"/>
<text x="8.0" y="151.0">Flat 2</text>
<text x="350.0" y="151.0" class="fig-mono">+30</text>
<rect x="430.0" y="138.0" width="18.0" height="18.0" class="fill-soft-b stroke-b" />
<text x="456.0" y="151.0" class="fig-mono">900</text>
<line x1="60.0" y1="219.0" x2="340.0" y2="219.0" class="stroke-grid" />
<line x1="187.3" y1="232.0" x2="207.6" y2="232.0" class="stroke-bad" />
<line x1="187.3" y1="228.0" x2="187.3" y2="236.0" class="stroke-bad" />
<line x1="207.6" y1="228.0" x2="207.6" y2="236.0" class="stroke-bad" />
<circle cx="187.3" cy="219.0" r="5" class="fill-a"/>
<circle cx="207.6" cy="219.0" r="5" class="fill-b"/>
<text x="8.0" y="223.0">Flat 3</text>
<text x="350.0" y="223.0" class="fig-mono">+40</text>
<rect x="430.0" y="207.0" width="24.0" height="24.0" class="fill-soft-b stroke-b" />
<text x="462.0" y="223.0" class="fig-mono">1600</text>
<line x1="60.0" y1="291.0" x2="340.0" y2="291.0" class="stroke-grid" />
<line x1="263.6" y1="304.0" x2="314.5" y2="304.0" class="stroke-bad" />
<line x1="263.6" y1="300.0" x2="263.6" y2="308.0" class="stroke-bad" />
<line x1="314.5" y1="300.0" x2="314.5" y2="308.0" class="stroke-bad" />
<circle cx="314.5" cy="291.0" r="5" class="fill-a"/>
<circle cx="263.6" cy="291.0" r="5" class="fill-b"/>
<text x="8.0" y="295.0">Flat 4</text>
<text x="350.0" y="295.0" class="fig-mono">−100</text>
<rect x="430.0" y="261.0" width="60.0" height="60.0" class="fill-soft-b stroke-b" />
<text x="498.0" y="295.0" class="fig-mono">10000</text>
<line x1="85.5" y1="325.0" x2="85.5" y2="331.0" class="stroke-muted" />
<text x="85.5" y="345.0" text-anchor="middle" class="fig-small">200</text>
<line x1="136.4" y1="325.0" x2="136.4" y2="331.0" class="stroke-muted" />
<text x="136.4" y="345.0" text-anchor="middle" class="fig-small">300</text>
<line x1="187.3" y1="325.0" x2="187.3" y2="331.0" class="stroke-muted" />
<text x="187.3" y="345.0" text-anchor="middle" class="fig-small">400</text>
<line x1="238.2" y1="325.0" x2="238.2" y2="331.0" class="stroke-muted" />
<text x="238.2" y="345.0" text-anchor="middle" class="fig-small">500</text>
<line x1="289.1" y1="325.0" x2="289.1" y2="331.0" class="stroke-muted" />
<text x="289.1" y="345.0" text-anchor="middle" class="fig-small">600</text>
<line x1="340.0" y1="325.0" x2="340.0" y2="331.0" class="stroke-muted" />
<text x="340.0" y="345.0" text-anchor="middle" class="fig-small">700</text>
<line x1="60.0" y1="328.0" x2="340.0" y2="328.0" class="stroke-muted" />
<text x="8.0" y="375.0" class="fig-small">Signed misses add to −10, so they hide the error.</text>
<text x="400.0" y="375.0" class="fig-small">Squares add to 12900. Mean: 3225.</text>
</svg>
Figure 1. The rule "eleven times the size" on the four flats. The signed misses nearly cancel, because they add to −10, but the squares cannot cancel. Notice that the £100 miss on the last flat has 25 times the area of the £20 miss on the first.
```

Define the miss as the guess minus the target, where the target is the real rent. The four misses are +20, +30, +40, and −100.

Why not add the misses to get a score? Because a guess that is too high cancels a guess that is too low. The four misses add to −10, which suggests the rule is nearly perfect, even though it is wrong on every flat and £100 off on the last one. Two flats off by +100 and −100 would add to 0. A score has to be a number that errors cannot cancel, so it must never be negative.

The two obvious fixes are to drop the sign or to square the miss. Dropping the sign gives the *absolute loss*, written $|\hat{y}-y|$. Squaring gives the *squared loss*. Here $\hat{y}$ (said "y-hat") is the guess and $y$ is the target. The squared loss is

$$
\ell(\hat{y},y)=(\hat{y}-y)^2
$$

Read it aloud: take the guess, subtract the target, and square the result. The symbol $\ell$ is the Greek letter ell, short for "loss". Flat 4 has guess 550 and target 650, so its loss is $(550-650)^2=10000$. The four squared losses are 400, 900, 1600, and 10000. In numpy:

```python
squared = (predictions - targets) ** 2   # one score per flat, shape (n,)
```

Why prefer squares to absolute values when both work? Figure 2 shows the difference. One miss of 2 costs 4 under squared loss, but two misses of 1 cost only 2 together. Under absolute loss both cost 2. Squared loss therefore punishes one big miss harder than several small ones, which is why the £100 miss dominates the squares in Figure 1. Its curve is also smooth at 0, while the absolute curve has a corner there. Later chapters walk downhill along the score, and a smooth bowl is easier to walk down. Neither choice is correct in itself. The loss is a statement about what you care about, and this course uses squared loss unless it says otherwise.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three loss functions drawn against the miss: squared, absolute, and zero-one">
<text x="102.5" y="30.0" text-anchor="middle" class="fig-bold">Squared loss</text>
<line x1="20.0" y1="200.0" x2="185.0" y2="200.0" class="stroke-muted" />
<line x1="20.0" y1="200.0" x2="185.0" y2="200.0" class="stroke-grid" />
<line x1="20.0" y1="162.5" x2="185.0" y2="162.5" class="stroke-grid" />
<line x1="20.0" y1="125.0" x2="185.0" y2="125.0" class="stroke-grid" />
<line x1="20.0" y1="87.5" x2="185.0" y2="87.5" class="stroke-grid" />
<line x1="20.0" y1="50.0" x2="185.0" y2="50.0" class="stroke-grid" />
<line x1="102.5" y1="200.0" x2="102.5" y2="50.0" class="stroke-grid" />
<polyline points="20.0,50.0 22.1,57.4 24.1,64.6 26.2,71.7 28.2,78.5 30.3,85.2 32.4,91.6 34.4,97.9 36.5,104.0 38.6,109.9 40.6,115.6 42.7,121.2 44.8,126.5 46.8,131.7 48.9,136.6 50.9,141.4 53.0,146.0 55.1,150.4 57.1,154.6 59.2,158.7 61.2,162.5 63.3,166.2 65.4,169.6 67.4,172.9 69.5,176.0 71.6,178.9 73.6,181.6 75.7,184.2 77.8,186.5 79.8,188.7 81.9,190.6 83.9,192.4 86.0,194.0 88.1,195.4 90.1,196.6 92.2,197.7 94.2,198.5 96.3,199.2 98.4,199.6 100.4,199.9 102.5,200.0 104.6,199.9 106.6,199.6 108.7,199.2 110.8,198.5 112.8,197.7 114.9,196.6 116.9,195.4 119.0,194.0 121.1,192.4 123.1,190.6 125.2,188.7 127.2,186.5 129.3,184.2 131.4,181.6 133.4,178.9 135.5,176.0 137.6,172.9 139.6,169.6 141.7,166.2 143.8,162.5 145.8,158.7 147.9,154.6 149.9,150.4 152.0,146.0 154.1,141.4 156.1,136.6 158.2,131.7 160.3,126.5 162.3,121.2 164.4,115.6 166.4,109.9 168.5,104.0 170.6,97.9 172.6,91.6 174.7,85.2 176.8,78.5 178.8,71.7 180.9,64.6 182.9,57.4 185.0,50.0" fill="none" class="stroke-b" />
<circle cx="20.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="61.2" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="143.8" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="185.0" cy="50.0" r="3.5" class="fill-b"/>
<text x="20.0" y="218.0" text-anchor="middle" class="fig-small">−2</text>
<text x="102.5" y="218.0" text-anchor="middle" class="fig-small">0</text>
<text x="185.0" y="218.0" text-anchor="middle" class="fig-small">2</text>
<text x="102.5" y="238.0" text-anchor="middle" class="fig-small">miss</text>
<text x="297.5" y="30.0" text-anchor="middle" class="fig-bold">Absolute loss</text>
<line x1="215.0" y1="200.0" x2="380.0" y2="200.0" class="stroke-muted" />
<line x1="215.0" y1="200.0" x2="380.0" y2="200.0" class="stroke-grid" />
<line x1="215.0" y1="162.5" x2="380.0" y2="162.5" class="stroke-grid" />
<line x1="215.0" y1="125.0" x2="380.0" y2="125.0" class="stroke-grid" />
<line x1="215.0" y1="87.5" x2="380.0" y2="87.5" class="stroke-grid" />
<line x1="215.0" y1="50.0" x2="380.0" y2="50.0" class="stroke-grid" />
<line x1="297.5" y1="200.0" x2="297.5" y2="50.0" class="stroke-grid" />
<polyline points="215.0,125.0 217.1,126.9 219.1,128.8 221.2,130.6 223.2,132.5 225.3,134.4 227.4,136.2 229.4,138.1 231.5,140.0 233.6,141.9 235.6,143.8 237.7,145.6 239.8,147.5 241.8,149.4 243.9,151.2 245.9,153.1 248.0,155.0 250.1,156.9 252.1,158.8 254.2,160.6 256.2,162.5 258.3,164.4 260.4,166.2 262.4,168.1 264.5,170.0 266.6,171.9 268.6,173.8 270.7,175.6 272.8,177.5 274.8,179.4 276.9,181.2 278.9,183.1 281.0,185.0 283.1,186.9 285.1,188.8 287.2,190.6 289.2,192.5 291.3,194.4 293.4,196.2 295.4,198.1 297.5,200.0 299.6,198.1 301.6,196.2 303.7,194.4 305.8,192.5 307.8,190.6 309.9,188.8 311.9,186.9 314.0,185.0 316.1,183.1 318.1,181.2 320.2,179.4 322.2,177.5 324.3,175.6 326.4,173.8 328.4,171.9 330.5,170.0 332.6,168.1 334.6,166.2 336.7,164.4 338.8,162.5 340.8,160.6 342.9,158.8 344.9,156.9 347.0,155.0 349.1,153.1 351.1,151.2 353.2,149.4 355.2,147.5 357.3,145.6 359.4,143.8 361.4,141.9 363.5,140.0 365.6,138.1 367.6,136.2 369.7,134.4 371.8,132.5 373.8,130.6 375.9,128.8 377.9,126.9 380.0,125.0" fill="none" class="stroke-b" />
<circle cx="215.0" cy="125.0" r="3.5" class="fill-b"/>
<circle cx="256.2" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="338.8" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="380.0" cy="125.0" r="3.5" class="fill-b"/>
<text x="215.0" y="218.0" text-anchor="middle" class="fig-small">−2</text>
<text x="297.5" y="218.0" text-anchor="middle" class="fig-small">0</text>
<text x="380.0" y="218.0" text-anchor="middle" class="fig-small">2</text>
<text x="297.5" y="238.0" text-anchor="middle" class="fig-small">miss</text>
<text x="492.5" y="30.0" text-anchor="middle" class="fig-bold">Zero-one loss</text>
<line x1="410.0" y1="200.0" x2="575.0" y2="200.0" class="stroke-muted" />
<line x1="410.0" y1="200.0" x2="575.0" y2="200.0" class="stroke-grid" />
<line x1="410.0" y1="162.5" x2="575.0" y2="162.5" class="stroke-grid" />
<line x1="410.0" y1="125.0" x2="575.0" y2="125.0" class="stroke-grid" />
<line x1="410.0" y1="87.5" x2="575.0" y2="87.5" class="stroke-grid" />
<line x1="410.0" y1="50.0" x2="575.0" y2="50.0" class="stroke-grid" />
<line x1="492.5" y1="200.0" x2="492.5" y2="50.0" class="stroke-grid" />
<line x1="410.0" y1="162.5" x2="488.5" y2="162.5" class="stroke-b" />
<line x1="496.5" y1="162.5" x2="575.0" y2="162.5" class="stroke-b" />
<circle cx="492.5" cy="200.0" r="4" class="fill-b"/>
<circle cx="492.5" cy="162.5" r="4" class="fill-paper stroke-b"/>
<circle cx="410.0" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="451.2" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="533.8" cy="162.5" r="3.5" class="fill-b"/>
<circle cx="575.0" cy="162.5" r="3.5" class="fill-b"/>
<text x="410.0" y="218.0" text-anchor="middle" class="fig-small">−2</text>
<text x="492.5" y="218.0" text-anchor="middle" class="fig-small">0</text>
<text x="575.0" y="218.0" text-anchor="middle" class="fig-small">2</text>
<text x="492.5" y="238.0" text-anchor="middle" class="fig-small">miss</text>
<text x="2.0" y="54.0" class="fig-small">4</text>
<text x="2.0" y="129.0" class="fig-small">2</text>
<text x="2.0" y="204.0" class="fig-small">0</text>
</svg>
Figure 2. Three ways to score a miss. Squared loss grows fastest beyond a miss of 1, so a miss of 2 costs four times a miss of 1, and it is smaller than absolute loss for misses under 1. Absolute loss grows evenly. Zero-one loss only asks whether the guess was exactly right.
```

Some targets are categories, not amounts. A spam filter predicts 1 for spam and 0 for not spam. Subtracting categories is meaningless, because the distance between "spam" and "not spam" has no size. The only question is whether the guess matches. The *zero-one loss* answers it: 0 for a match, 1 for a mismatch. In numpy, `(predictions != targets)` gives booleans, and `.astype(float)` turns them into 0.0 and 1.0.

## Average the scores

One flat gives one score. To score a rule you need one number for all the flats. Add the four squared losses from Figure 1: $400+900+1600+10000=12900$. Then divide by 4 to get 3225.

Why divide? Without it, a rule scored on 400 flats looks about 100 times worse than the same rule scored on 4, only because it was scored on more flats. Dividing by the count makes the score mean "the typical miss", whatever the number of flats. This average is the *empirical risk*, where "empirical" means "measured on data you have". With $n$ flats and $\ell_i$ the loss on flat $i$:

$$
\hat{R}=\frac{1}{n}\sum_{i=1}^{n}\ell_i
$$

Read it aloud: add up the $n$ scores, then divide by $n$. The symbol $\hat{R}$ is said "R-hat", and the hat marks an estimate computed from data. For the four flats, $\hat{R}=12900/4=3225$. The unit is pounds squared, because the loss squares pounds. In numpy:

```python
risk = float(np.mean(loss(predictions, targets)))   # loss is any per-flat score function
```

For the spam example, predictions 1, 0, 1 against targets 1, 1, 0 give zero-one losses 0, 1, and 1. Two of three are wrong, so $\hat{R}=2/3$.

## Use the score to choose a rule

The opening question has an answer. The rule has a knob $s$, the slope: rent equals $s$ times the size. The four flats have sizes 20, 30, 40, and 50, and rents 200, 300, 400, and 650. Slope 11 scored 3225.

Score slope 10 the same way. The guesses are 200, 300, 400, and 500, so the misses are 0, 0, 0, and −150. The squares are 0, 0, 0, and 22500, and the mean is $22500/4=5625$. Slope 11 wins, 3225 against 5625. A rule that is slightly wrong on every flat beats a rule that is exactly right on three and far off on one, because squares punish the big miss so heavily.

Why stop at two slopes? Try every whole slope from 6 to 14, score each one, and keep the lowest. Figure 3 plots the scores. They form a bowl, and the bottom sits between 11 and 12 (at about 11.4), so slope 11 wins on the whole-number grid.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Mean squared loss of the rent rule against its slope, a bowl with its lowest point near slope 11.4">
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="40.0" x2="70.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-grid" />
<text x="62.0" y="274.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="218.9" x2="570.0" y2="218.9" class="stroke-grid" />
<text x="62.0" y="222.9" text-anchor="end" class="fig-small">10k</text>
<line x1="70.0" y1="167.8" x2="570.0" y2="167.8" class="stroke-grid" />
<text x="62.0" y="171.8" text-anchor="end" class="fig-small">20k</text>
<line x1="70.0" y1="116.7" x2="570.0" y2="116.7" class="stroke-grid" />
<text x="62.0" y="120.7" text-anchor="end" class="fig-small">30k</text>
<line x1="70.0" y1="65.6" x2="570.0" y2="65.6" class="stroke-grid" />
<text x="62.0" y="69.6" text-anchor="end" class="fig-small">40k</text>
<polyline points="70.0,54.2 73.1,57.9 76.2,61.6 79.4,65.2 82.5,68.8 85.6,72.3 88.7,75.9 91.9,79.4 95.0,82.8 98.1,86.3 101.2,89.6 104.4,93.0 107.5,96.3 110.6,99.6 113.8,102.9 116.9,106.1 120.0,109.3 123.1,112.4 126.3,115.5 129.4,118.6 132.5,121.7 135.6,124.7 138.7,127.6 141.9,130.6 145.0,133.5 148.1,136.4 151.2,139.2 154.4,142.0 157.5,144.8 160.6,147.5 163.8,150.2 166.9,152.9 170.0,155.5 173.1,158.1 176.2,160.7 179.4,163.2 182.5,165.7 185.6,168.1 188.8,170.6 191.9,173.0 195.0,175.3 198.1,177.6 201.2,179.9 204.4,182.2 207.5,184.4 210.6,186.6 213.8,188.7 216.9,190.8 220.0,192.9 223.1,195.0 226.2,197.0 229.4,199.0 232.5,200.9 235.6,202.8 238.7,204.7 241.9,206.5 245.0,208.3 248.1,210.1 251.3,211.8 254.4,213.5 257.5,215.2 260.6,216.8 263.8,218.4 266.9,220.0 270.0,221.5 273.1,223.0 276.3,224.5 279.4,225.9 282.5,227.3 285.6,228.6 288.8,229.9 291.9,231.2 295.0,232.5 298.1,233.7 301.2,234.9 304.4,236.0 307.5,237.1 310.6,238.2 313.8,239.3 316.9,240.3 320.0,241.2 323.1,242.2 326.3,243.1 329.4,244.0 332.5,244.8 335.6,245.6 338.8,246.4 341.9,247.1 345.0,247.8 348.1,248.5 351.2,249.1 354.4,249.7 357.5,250.3 360.6,250.8 363.7,251.3 366.9,251.7 370.0,252.2 373.1,252.6 376.2,252.9 379.4,253.2 382.5,253.5 385.6,253.8 388.8,254.0 391.9,254.2 395.0,254.3 398.1,254.4 401.3,254.5 404.4,254.5 407.5,254.6 410.6,254.5 413.8,254.5 416.9,254.4 420.0,254.3 423.1,254.1 426.2,253.9 429.4,253.7 432.5,253.4 435.6,253.1 438.8,252.8 441.9,252.4 445.0,252.0 448.1,251.5 451.3,251.1 454.4,250.6 457.5,250.0 460.6,249.4 463.8,248.8 466.9,248.2 470.0,247.5 473.1,246.8 476.2,246.0 479.4,245.3 482.5,244.4 485.6,243.6 488.7,242.7 491.9,241.8 495.0,240.8 498.1,239.8 501.2,238.8 504.4,237.7 507.5,236.7 510.6,235.5 513.8,234.4 516.9,233.2 520.0,231.9 523.1,230.7 526.2,229.4 529.4,228.0 532.5,226.7 535.6,225.2 538.8,223.8 541.9,222.3 545.0,220.8 548.1,219.3 551.2,217.7 554.4,216.1 557.5,214.4 560.6,212.8 563.8,211.1 566.9,209.3 570.0,207.5" fill="none" class="stroke-b" />
<circle cx="70.0" cy="54.2" r="3.5" class="fill-b"/>
<text x="70.0" y="288.0" text-anchor="middle" class="fig-small">6</text>
<circle cx="132.5" cy="121.7" r="3.5" class="fill-b"/>
<text x="132.5" y="288.0" text-anchor="middle" class="fig-small">7</text>
<circle cx="195.0" cy="175.3" r="3.5" class="fill-b"/>
<text x="195.0" y="288.0" text-anchor="middle" class="fig-small">8</text>
<circle cx="257.5" cy="215.2" r="3.5" class="fill-b"/>
<text x="257.5" y="288.0" text-anchor="middle" class="fig-small">9</text>
<circle cx="320.0" cy="241.2" r="3.5" class="fill-b"/>
<text x="320.0" y="288.0" text-anchor="middle" class="fig-small">10</text>
<circle cx="382.5" cy="253.5" r="3.5" class="fill-b"/>
<text x="382.5" y="288.0" text-anchor="middle" class="fig-small">11</text>
<circle cx="445.0" cy="252.0" r="3.5" class="fill-b"/>
<text x="445.0" y="288.0" text-anchor="middle" class="fig-small">12</text>
<circle cx="507.5" cy="236.7" r="3.5" class="fill-b"/>
<text x="507.5" y="288.0" text-anchor="middle" class="fig-small">13</text>
<circle cx="570.0" cy="207.5" r="3.5" class="fill-b"/>
<text x="570.0" y="288.0" text-anchor="middle" class="fig-small">14</text>
<line x1="406.8" y1="270.0" x2="406.8" y2="40.0" class="stroke-c dashed" />
<text x="414.8" y="56.0" class="fig-small">exact lowest point, 11.4</text>
<circle cx="382.5" cy="253.5" r="7" class="stroke-a"/>
<circle cx="320.0" cy="241.2" r="7" class="stroke-bad"/>
<circle cx="426.2" cy="100.0" r="6" class="stroke-bad"/>
<text x="440.2" y="104.0" class="fig-small">slope 10: mean loss 5625</text>
<circle cx="426.2" cy="124.0" r="6" class="stroke-a"/>
<text x="440.2" y="128.0" class="fig-small">slope 11: mean loss 3225</text>
<text x="320.0" y="312.0" text-anchor="middle" class="fig-small">slope in £ per m²</text>
<text x="10.0" y="28.0" class="fig-small">mean squared loss (£²)</text>
</svg>
Figure 3. Mean squared loss of the rule "rent = slope × size" on the four flats. The scores form a bowl. Notice that the bottom is not at slope 10, the slope that fits three flats exactly.
```

This is the first taste of learning: a rule has knobs, a score says how good each setting is, and learning is a search for the setting with the best score. The loss also decides what "best" means. Under absolute loss the same scan picks slope 10, with mean 37.5 against 47.5 at slope 11, because absolute loss does not let the £650 flat dominate. The code is a loop over candidates:

```python
risks = [empirical_risk(s * sizes, rents, loss) for s in slopes]
best = slopes[int(np.argmin(risks))]   # argmin gives the position of the smallest risk
```

## What the score can and cannot tell you

The four flats are a handful from all the flats the rule will meet. The average miss over all of them is the *true risk*. You cannot compute it, because you cannot list every flat. The empirical risk is a stand-in, like surveying four people to guess a city's mood. The more flats you add, the closer the stand-in gets.

The stand-in has a condition. Imagine a pollster who surveys only the people who already agree with her. Her sample says nothing about the city. In the same way, the empirical risk is honest only if the rule was chosen without looking at those flats. We picked slope 11 because it scored best on exactly these four flats, and the £650 flat pulled the choice upward. So 3225 is the score of a rule tuned to the flats that grade it, and a new flat will probably score worse. The next chapter, "Honest scores: training error, test error, and holding data back", shows how to get a score that does not flatter.

> [!KEY]
> To compare rules you need one number. A per-flat loss turns each miss into a non-negative score, and the empirical risk averages those scores. Learning starts as a search for the knob setting with the lowest risk, and the risk is a fair estimate of future error only for flats the search never touched.

> [!RECAP]
> - A loss maps one guess and one target to one non-negative score.
> - Use squared loss for amounts, absolute loss when outliers must not dominate, and zero-one loss for categories.
> - Empirical risk is the mean of the losses. Return it as a Python `float`.
> - To choose a knob, compute the risk at each candidate and take the `argmin`.
> - A risk measured on the flats you tuned on is too kind.

## Build it step by step

The starter file provides `absolute_loss` and `zero_one_loss` already, because they are one-line cousins of squared loss. You write three functions, in the order the checks test them.

1. **`squared_loss(predictions, targets)`** takes two arrays of shape `(n,)` and returns an array of shape `(n,)`:

   ```python
   miss = predictions - targets   # signed miss per flat
   # Square it.
   ```

   The common mistake is to return a single number. This function returns one score per flat.

2. **`empirical_risk(predictions, targets, loss)`** calls `loss` on the two arrays and returns the mean as a Python `float`:

   ```python
   scores = loss(predictions, targets)   # shape (n,)
   # Average them, then wrap in float().
   ```

   `np.sum` skips the division and grows with the number of flats.

3. **`best_slope(sizes, rents, slopes, loss)`** returns the candidate slope with the lowest empirical risk for the rule `rent = slope * size`:

   ```python
   guesses = slope * sizes   # one guess per flat for this slope
   # Score each candidate, then pick the smallest risk.
   ```

   `np.argmin` returns a position, not a slope. Index into `slopes` with it.

## Where this breaks

**A few huge misses dominate.** At slope 11, the £650 flat contributes 10000 of the total 12900, about 78 percent, as Figure 1 shows. If one flat is a typing error, it steers the whole search. Look at the per-flat scores as well as the mean, and consider absolute loss when outliers are junk.

**The score hides units.** A squared loss of 3225 is in pounds squared, which has no everyday meaning. Reports often take the square root, which returns to pounds.

**The knob scan is coarse.** A grid of whole slopes cannot find 11.4. Later chapters find the exact bottom of the bowl with algebra, and then with walking downhill.

## The exercise

You write `squared_loss`, `empirical_risk`, and `best_slope`. The checks verify that `squared_loss` returns one score per flat and matches the hand values from Figure 1 (400, 900, 1600, and 10000), and that `empirical_risk` returns 3225 for the four-flat example and 2/3 for the spam example, as a Python `float`. A flat repeated twice must not change the mean, which catches `np.sum`.

The last checks run `best_slope` on the four-flat data. Under squared loss it must return 11. Under absolute loss it must return 10. On seeded random data, it must agree with a brute-force scan of its own.
