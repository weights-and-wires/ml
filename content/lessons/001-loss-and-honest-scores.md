---
title: "Scoring a rule honestly: loss, risk, and held-out flats"
slug: 001-loss-and-honest-scores
part: "Part 1 — Foundations"
summary: Turn misses into one score, use it to pick a rule, then hold flats back so the score does not flatter the rule.
minutes: 65
gpu: false
objectives:
  - Turn a miss into a score with squared loss, absolute loss, or zero-one loss, and say what each one cares about.
  - Average per-example scores into the empirical risk, and pick a rule's best knob setting by comparing risks.
  - Explain why a rule that memorises its training flats scores zero error and still fails on flats it has not seen.
  - Split data into training and test sets with a seeded shuffle that keeps each size paired with its rent.
  - Compute the generalisation gap, and name two ways a held-out score can still mislead.
exercise: 001-loss-and-honest-scores
---

# Scoring a rule honestly: loss, risk, and held-out flats

You collect four flats. A 20 square metre flat rents for £200 a week, a 30 square metre flat for £300, and a 40 square metre flat for £400. The rule "weekly rent equals ten times the size" matches all three. Then you meet a 50 square metre flat. The rule says £500, and the landlord asks £650.

Is the rule bad? Would "eleven times the size" be better? You cannot answer by looking at the rule. You can answer only if each rule produces one number that says how wrong it was, so that you can put two numbers side by side and pick the smaller. Without that number, with forty flats and five candidate rules, you would argue about each flat separately and never finish.

Once you have the number, a second problem appears. A friend proposes a rule that looks up the closest flat in your folder and copies its rent. The rule scores a perfect 0 on the flats in your folder, and it cannot possibly be perfect. This chapter builds the number, uses it to choose between rules, and then shows how to get a score that does not lie: hold some flats back, and grade the rule only on those. Every later chapter, from fitting lines to training neural networks, uses these two tools.

## What you already know

You already use these ideas:

- The average of 2, 4, and 6 is 4.
- An error of 2, squared, is 4. Two errors of 1, squared and added, make 2. Squaring makes big errors count for more.
- numpy subtracts two arrays entry by entry, so `a - b` gives one difference per position.
- A function is a value you can pass around, such as `lambda p, t: (p - t) ** 2`.
- A seed makes a random generator repeat the same numbers, so a result can be reproduced.

## Turn a miss into a score

Score the rule "eleven times the size" on your four flats. The rule guesses £220, £330, £440, and £550, and the real rents are £200, £300, £400, and £650. Figure 1 draws each guess in orange and each real rent in blue. The red bracket under each pair is the miss.

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

Why not add the misses to get a score? Because a guess that is too high cancels a guess that is too low. The four misses add to −10, which suggests the rule is nearly perfect, even though it is wrong on every flat and £100 off on the last one. A score has to be a number that errors cannot cancel, so it must never be negative.

The two obvious fixes are to drop the sign or to square the miss. Dropping the sign gives the *absolute loss*, written $|\hat{y}-y|$. Squaring gives the *squared loss*. Here $\hat{y}$ (said "y-hat") is the guess and $y$ is the target. The squared loss is

$$
\ell(\hat{y},y)=(\hat{y}-y)^2
$$

Read it aloud: take the guess, subtract the target, and square the result. The symbol $\ell$ is the Greek letter ell, short for "loss". Flat 4 has guess 550 and target 650, so its loss is $(550-650)^2=10000$. The four squared losses are 400, 900, 1600, and 10000. In numpy:

```python
squared = (predictions - targets) ** 2   # one score per flat, shape (n,)
```

Why prefer squares to absolute values when both work? One miss of 2 costs 4 under squared loss, but two misses of 1 cost only 2 together. Under absolute loss both cost 2. Squared loss therefore punishes one big miss harder than several small ones, which is why the £100 miss dominates the squares in Figure 1. Its curve is also smooth at 0, while the absolute curve has a corner there, and later chapters walk downhill along the score, which is easier on a smooth bowl. Neither choice is correct in itself. The loss is a statement about what you care about, and this course uses squared loss unless it says otherwise.

Some targets are categories, not amounts. A spam filter predicts 1 for spam and 0 for not spam, and subtracting categories is meaningless, because the distance between "spam" and "not spam" has no size. The only question is whether the guess matches. The *zero-one loss* answers it: 0 for a match, 1 for a mismatch. In numpy, `(predictions != targets)` gives booleans, and `.astype(float)` turns them into 0.0 and 1.0.

## Average the scores

One flat gives one score, and to grade a rule you need one number for all the flats. Add the four squared losses from Figure 1: $400+900+1600+10000=12900$. Then divide by 4 to get 3225.

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

Why stop at two slopes? Try every whole slope from 6 to 14, score each one, and keep the lowest. Figure 2 plots the scores. They form a bowl, and the bottom sits at about 11.4, so slope 11 wins on the whole-number grid.

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
Figure 2. Mean squared loss of the rule "rent = slope × size" on the four flats. The scores form a bowl. Notice that the bottom is not at slope 10, the slope that fits three flats exactly.
```

This is the first taste of learning: a rule has knobs, a score says how good each setting is, and learning is a search for the setting with the best score. The loss also decides what "best" means. Under absolute loss the same scan picks slope 10, with mean 37.5 against 47.5 at slope 11, because absolute loss does not let the £650 flat dominate. The code is a loop over candidates:

```python
risks = [empirical_risk(s * sizes, rents, loss) for s in slopes]
best = slopes[int(np.argmin(risks))]   # argmin gives the position of the smallest risk
```

> [!KEY]
> To compare rules you need one number. A per-flat loss turns each miss into a non-negative score, and the empirical risk averages those scores. Learning starts as a search for the knob setting with the lowest risk.

## A score the rule has seen is too kind

The four flats are a handful from all the flats the rule will meet. The average miss over all of them is the *true risk*. You cannot compute it, because you cannot list every flat, so the empirical risk is a stand-in, like surveying four people to guess a city's mood. The stand-in has a condition. A pollster who surveys only the people who already agree with her learns nothing about the city. In the same way, the empirical risk is honest only if the rule was chosen without looking at the flats that grade it. We picked slope 11 because it scored best on exactly these four flats, so 3225 is the score of a rule tuned to its own grader.

To see how bad this can get, move to a market where you know the truth. Measure flat size on a scale from 0 (the smallest flat in the folder) to 1 (the largest). Measure rent as how many hundreds of pounds the flat sits above or below the typical rent. In this made-up market the rent follows a smooth wave in the size, plus a random wobble from the landlord's mood. The wave is the pattern worth learning, and the wobble is chance that a new flat will not repeat. Figure 3 shows 20 such flats.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Twenty flats: a memorising rule passes through every training flat and misses the held-out flats">
<line x1="50.0" y1="290.0" x2="580.0" y2="290.0" class="stroke-muted" />
<line x1="50.0" y1="30.0" x2="50.0" y2="290.0" class="stroke-muted" />
<line x1="50.0" y1="236.5" x2="580.0" y2="236.5" class="stroke-grid" />
<text x="42.0" y="240.5" text-anchor="end" class="fig-small">−1</text>
<line x1="50.0" y1="160.0" x2="580.0" y2="160.0" class="stroke-grid" />
<text x="42.0" y="164.0" text-anchor="end" class="fig-small">0</text>
<line x1="50.0" y1="83.5" x2="580.0" y2="83.5" class="stroke-grid" />
<text x="42.0" y="87.5" text-anchor="end" class="fig-small">1</text>
<text x="50.0" y="308.0" text-anchor="middle" class="fig-small">0</text>
<text x="315.0" y="308.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="580.0" y="308.0" text-anchor="middle" class="fig-small">1</text>
<text x="315.0" y="325.0" text-anchor="middle" class="fig-small">flat size (0 = smallest in the folder, 1 = largest)</text>
<text x="8.0" y="22.0" class="fig-small">rent, above or below typical (£100)</text>
<polyline points="50.0,163.5 55.1,163.5 55.1,144.0 100.9,144.0 100.9,83.6 168.0,83.6 168.0,52.8 200.9,52.8 200.9,69.2 272.9,69.2 272.9,182.7 337.5,182.7 337.5,191.2 362.9,191.2 362.9,220.9 412.1,220.9 412.1,257.0 436.7,257.0 436.7,230.9 459.6,230.9 459.6,253.2 493.4,253.2 493.4,223.4 506.0,223.4 506.0,232.9 520.6,232.9 520.6,197.7 539.7,197.7 539.7,185.3 580.0,185.3" fill="none" class="stroke-b" />
<line x1="481.0" y1="209.8" x2="481.0" y2="253.2" class="stroke-bad" />
<line x1="274.0" y1="90.0" x2="274.0" y2="182.7" class="stroke-bad" />
<line x1="371.5" y1="224.5" x2="371.5" y2="220.9" class="stroke-bad" />
<line x1="71.7" y1="155.8" x2="71.7" y2="144.0" class="stroke-bad" />
<line x1="67.8" y1="131.6" x2="67.8" y2="144.0" class="stroke-bad" />
<circle cx="507.5" cy="232.9" r="5" class="fill-a"/>
<circle cx="58.8" cy="144.0" r="5" class="fill-a"/>
<circle cx="51.5" cy="163.5" r="5" class="fill-a"/>
<circle cx="482.4" cy="253.2" r="5" class="fill-a"/>
<circle cx="338.1" cy="191.2" r="5" class="fill-a"/>
<circle cx="387.6" cy="220.9" r="5" class="fill-a"/>
<circle cx="504.4" cy="223.4" r="5" class="fill-a"/>
<circle cx="436.6" cy="257.0" r="5" class="fill-a"/>
<circle cx="533.8" cy="197.7" r="5" class="fill-a"/>
<circle cx="208.8" cy="69.2" r="5" class="fill-a"/>
<circle cx="337.0" cy="182.7" r="5" class="fill-a"/>
<circle cx="436.7" cy="230.9" r="5" class="fill-a"/>
<circle cx="545.6" cy="185.3" r="5" class="fill-a"/>
<circle cx="193.0" cy="52.8" r="5" class="fill-a"/>
<circle cx="143.1" cy="83.6" r="5" class="fill-a"/>
<circle cx="481.0" cy="209.8" r="5.5" class="fill-paper stroke-a"/>
<circle cx="274.0" cy="90.0" r="5.5" class="fill-paper stroke-a"/>
<circle cx="371.5" cy="224.5" r="5.5" class="fill-paper stroke-a"/>
<circle cx="71.7" cy="155.8" r="5.5" class="fill-paper stroke-a"/>
<circle cx="67.8" cy="131.6" r="5.5" class="fill-paper stroke-a"/>
<circle cx="400.0" cy="50.0" r="5" class="fill-a"/>
<text x="410.0" y="54.0" class="fig-small">15 training flats</text>
<circle cx="400.0" cy="70.0" r="5.5" class="fill-paper stroke-a"/>
<text x="410.0" y="74.0" class="fig-small">5 held-out flats</text>
<line x1="392.0" y1="90.0" x2="408.0" y2="90.0" class="stroke-b" />
<text x="414.0" y="94.0" class="fig-small">memorising rule</text>
<line x1="400.0" y1="106.0" x2="400.0" y2="114.0" class="stroke-bad" />
<text x="414.0" y="114.0" class="fig-small">held-out miss</text>
</svg>
Figure 3. Twenty flats and a memorising rule. The orange staircase hits every blue training dot, so the training error is 0. Notice the red segments: the rule misses the hollow held-out flats, one of them by more than 1 (the segment at size 0.42 runs along a step edge).
```

Your friend's rule is the orange staircase in Figure 3. For any size, it returns the rent of the closest training flat. It passes through every blue dot, so its score on those flats is exactly 0.

Take a tiny case. Two training flats have sizes 0.2 and 0.6 and rents 1.0 and −0.5. A third flat, which the rule has not seen, has size 0.35 and rent 0.7. The closest training size is 0.2, because $|0.35-0.2|=0.15$ and $|0.35-0.6|=0.25$. The rule guesses 1.0, the miss is 0.3, and the squared loss is 0.09. On its own two flats the rule scores 0, and on the third flat it scores 0.09. The gap between those two scores is the lie.

Figure 3 shows the same thing at scale. The five hollow dots are flats the rule never saw. Their squared losses are 0.322, 1.467, 0.002, 0.024, and 0.026, which average to $1.842/5=0.368$. A rule that scores 0 on 15 flats and 0.368 on 5 others has not learned the market.

A *training set* is the flats you fit the rule to, and the score on them is the *training error*. The score on flats the rule has not seen is the *test error*. Why is the training error too low? Every rent is the wave plus a wobble, and the wobble is luck: the landlord of flat 7 happened to be in a good mood. A rule that is free to bend will bend toward that luck, because doing so lowers its score on the folder. The memoriser takes this to the limit and copies each wobble as if it were a fact about the market. The next flat arrives with its own wobble, so the copied luck does not help and sometimes hurts. The training error counts luck as skill, and the test error does not.

This matters because searching for the lowest training score picks the memoriser every time. Picking the rule with the lowest average loss on the training data is called *empirical risk minimisation*, and it is a sensible way to fit. The trouble is only in what you report afterwards. The score you minimised cannot also be the score you trust. You need flats the fit never touched.

## Hold some flats back

Set aside a fraction of the flats before you fit anything. With $n=20$ flats and a fraction 0.25, you get $0.25\times20=5$ test flats and 15 training flats. You fit on the 15 and score on the 5, which play the role of next year's flats.

Why shuffle first, and not take the last 5 rows? If the folder is sorted by size, the last 5 flats are the five biggest, and the rule would be tested on a kind of flat it never trained on. A random shuffle makes the test flats look like the training flats on average, while keeping them different individual flats.

Shuffle positions, not the two arrays separately. The permutation is a list of positions from 0 to 19 in random order. Use the same list on both the sizes and the rents, and each size keeps its own rent. Two independent shuffles would pair 30 square metres with somebody else's rent. A seed fixes the shuffle, so a colleague can reproduce your split. The first `n_test` positions are the test flats, and the rest are the training flats:

```python
order = np.random.default_rng(seed).permutation(n)   # positions 0..n-1 in random order
test_idx, train_idx = order[:n_test], order[n_test:]
x_train, y_train = x[train_idx], y[train_idx]        # the same index list for x and y
```

The count rounds down: $\lfloor0.25\times20\rfloor=5$. Beware of floating-point arithmetic. In Python, `0.29 * 100` is 28.999999999999996, so a plain `int(...)` gives 28 test flats when the answer is 29. Add a tiny nudge before you truncate, as in `int(test_size * n + 1e-9)`.

> [!KEY]
> A held-out score is honest because the rule never saw those flats while it was being fit. The training score measures memory, and the test score measures what the rule does with a flat it has not met.

The memorising rule is a nearest-neighbour lookup. Compare the size of each flat you want to guess to every training size, and take the closest:

```python
d = np.abs(x_new[:, None] - x_train[None, :])   # shape (m, n): distance from each queried flat to each training flat
nearest = np.argmin(d, axis=1)                  # closest training flat for each queried flat
guesses = y_train[nearest]
```

The difference between the two risks has a name. The *generalisation gap* is the test risk minus the training risk. For the split in Figure 3 it is $0.368-0=0.368$. To compute it, chain the tools: split, memorise, and score both sets with `empirical_risk`.

## One split is noisy

The held-out score is a fair estimate, but it comes from only five flats. Split the same 20 flats with 30 different seeds and score the memoriser on each test set. Figure 4 shows the results. The training risk is 0 on every split. The test risk ranges from 0.028 to 0.385, with a mean of 0.131, and the split in Figure 3 happens to be one of the unlucky ones, at 0.368.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Held-out risk of the memorising rule on thirty different random splits, all above the training risk of zero">
<line x1="60.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-muted" />
<line x1="60.0" y1="40.0" x2="60.0" y2="220.0" class="stroke-grid" />
<text x="60.0" y="238.0" text-anchor="middle" class="fig-small">0.0</text>
<line x1="173.3" y1="40.0" x2="173.3" y2="220.0" class="stroke-grid" />
<text x="173.3" y="238.0" text-anchor="middle" class="fig-small">0.1</text>
<line x1="286.7" y1="40.0" x2="286.7" y2="220.0" class="stroke-grid" />
<text x="286.7" y="238.0" text-anchor="middle" class="fig-small">0.2</text>
<line x1="400.0" y1="40.0" x2="400.0" y2="220.0" class="stroke-grid" />
<text x="400.0" y="238.0" text-anchor="middle" class="fig-small">0.3</text>
<line x1="513.3" y1="40.0" x2="513.3" y2="220.0" class="stroke-grid" />
<text x="513.3" y="238.0" text-anchor="middle" class="fig-small">0.4</text>
<text x="315.0" y="262.0" text-anchor="middle" class="fig-small">held-out risk (mean squared loss), one dot per seed</text>
<circle cx="91.4" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="113.4" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="116.4" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="118.0" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="123.8" cy="173.0" r="5.5" class="fill-b"/>
<circle cx="127.8" cy="160.0" r="5.5" class="fill-b"/>
<circle cx="130.1" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="138.4" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="138.8" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="143.1" cy="173.0" r="5.5" class="fill-b"/>
<circle cx="157.0" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="159.4" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="160.5" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="162.3" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="169.6" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="173.0" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="183.5" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="186.3" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="190.9" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="193.1" cy="173.0" r="5.5" class="fill-b"/>
<circle cx="205.8" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="216.4" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="219.0" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="224.0" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="255.2" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="258.1" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="455.7" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="477.5" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="481.3" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="496.3" cy="212.0" r="5.5" class="fill-b"/>
<line x1="60.0" y1="40.0" x2="60.0" y2="220.0" class="stroke-a" />
<text x="68.0" y="56.0" class="fig-small">training risk: 0 on every split</text>
<line x1="208.9" y1="90.0" x2="208.9" y2="220.0" class="stroke-c dashed" />
<text x="214.9" y="104.0" class="fig-small">mean 0.131</text>
</svg>
Figure 4. The same rule scored on thirty random splits of the same flats. Notice that the training risk sits at 0 on every split, to the left of the whole cloud, and that the held-out risk jumps around.
```

Two lessons follow. A single test score has noise because a few flats carry it, so do not read much into small differences. A bigger test set steadies the score, but every flat you hold back is a flat the fit cannot learn from, so there is a trade-off. Also, touch the test set once. If you try ten rules and keep the one with the best test score, you have tuned to the test flats, and the score flatters again.

> [!KEY]
> A rule is scored by its average loss, but only the average over flats it never saw is honest. The chapter gave you both halves: a loss and risk to compare rules, and a held-out split to keep the comparison fair. The gap between the training and test risk measures how much a rule memorised.

> [!RECAP]
> - A loss maps one guess and one target to one non-negative score. Use squared loss for amounts, absolute loss when outliers must not dominate, and zero-one loss for categories.
> - Empirical risk is the mean of the losses. To choose a knob, compute the risk at each candidate and take the `argmin`.
> - The training error is too low on average, because the rule was tuned to those flats.
> - Shuffle positions once with a seed, index sizes and rents with the same positions, and score only on the held-out flats.
> - The generalisation gap is the test risk minus the training risk.

## Build it step by step

The starter file provides `absolute_loss` and `zero_one_loss` already, because they are one-line cousins of squared loss. You write six functions, in the order the checks test them.

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

4. **`train_test_split(x, y, test_size, seed)`** returns `(x_train, y_train, x_test, y_test)`. It takes `int(test_size * n + 1e-9)` test flats:

   ```python
   order = np.random.default_rng(seed).permutation(len(x))
   # The first n_test positions are the test flats. The rest are training flats.
   ```

   Do not shuffle `x` and `y` separately. That breaks every pair.

5. **`memoriser_predict(x_train, y_train, x_new)`** returns an array of shape `(m,)`, one rent per size you ask about:

   ```python
   d = np.abs(x_new[:, None] - x_train[None, :])   # (m, n)
   # argmin along axis 1 gives the nearest training flat for each queried flat.
   ```

   `axis=0` finds the nearest queried flat for each training flat, which is the wrong question.

6. **`generalisation_gap(x, y, test_size, seed)`** splits, memorises, and returns `(train_risk, test_risk, gap)` as Python floats, using squared loss:

   ```python
   x_train, y_train, x_test, y_test = train_test_split(x, y, test_size, seed)
   # Score the memoriser on the training flats, then on the test flats.
   ```

   Scoring only on the test flats would leave out the training risk, which must be 0.

## Where this breaks

**A few huge misses dominate.** At slope 11, the £650 flat contributes 10000 of the total 12900, about 78 percent, as Figure 1 shows. If one flat is a typing error, it steers the whole search. Look at the per-flat scores as well as the mean, and consider absolute loss when outliers are junk.

**The score hides units.** A squared loss of 3225 is in pounds squared, which has no everyday meaning. Reports often take the square root, which returns to pounds.

**The held-out flats are not like the flats you will meet.** A test score estimates future error only when the test flats resemble future flats. Statisticians call this condition *i.i.d.*, short for independent and identically distributed. *Identically distributed* means every flat comes from the same market: a folder of London flats says little about Leeds. *Independent* means one flat does not give away another. If the folder holds ten adverts for the same building, a shuffle can put one advert in the training set and its twin in the test set, and the memoriser scores brilliantly without learning anything about other buildings. Split by building, not by row.

**You pick the best test score.** Choosing among many rules by their test score tunes the rule to the test flats. Keep a separate set of flats for choosing between rules, and use the test set once. The chapter on [cross-validation](/lessons/011-cross-validation-and-leave-one-out) shows how to use every flat for testing without cheating.

## The exercise

You write `squared_loss`, `empirical_risk`, `best_slope`, `train_test_split`, `memoriser_predict`, and `generalisation_gap`. The first checks verify that `squared_loss` returns one score per flat and matches the hand values from Figure 1 (400, 900, 1600, and 10000), and that `empirical_risk` returns 3225 for the four-flat example and 2/3 for the spam example, as a Python `float`. A flat repeated twice must not change the mean, which catches `np.sum`.

`best_slope` must return 11 under squared loss and 10 under absolute loss on the four flats, and agree with a brute-force scan on seeded random data. The split checks confirm that every flat lands in exactly one set with its size paired to its rent, that one seed repeats the split and different seeds differ, and that the test count rounds the way the chapter describes. Finally, the memoriser must copy the rent of the nearest training size and score exactly 0 on its own training flats, and the gap check compares your numbers with an independent computation on seeded flats.
