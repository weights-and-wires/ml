---
title: "Price houses honestly"
slug: 010p-project-price-houses
part: "Part 1 — Foundations"
kind: project
summary: Price listings and predict fast sales with a split, QR, ridge, lasso, and Newton logistic pipeline that you judge on untouched test rows.
minutes: 180
gpu: false
objectives:
  - Build a seeded train, validation, and test pipeline in which only training rows shape the scaling and the fits.
  - Choose a ridge fee on validation rows from one SVD, and read what it does to two near-duplicate columns.
  - Use the lasso to find which columns carry the price, and check that it zeroes the noise.
  - Fit a Newton logistic model for sells-within-30-days and show it beats the base rate on log loss.
prereqs:
  - 001-loss-and-honest-scores
  - 004-stable-least-squares
  - 006-ridge-regression
  - 007-lasso-and-fista
  - 008-sigmoid-and-log-loss
  - 009-fitting-logistic-regression
exercise: 010p-project-price-houses
---

# Price houses honestly

A small estate agency has 500 past listings. For each one it recorded the floor area, the number of rooms, the age, the distance to the town centre, and two things that happened later: the sale price, and whether the house sold within 30 days. The agency wants two answers for every listing it has yet to advertise. What price should it advertise, and how likely is a quick sale?

The obvious approach is to fit least squares to every column and quote the training error. That fails here for three reasons. Two columns are the same measurement in different units, so the fit has no stable answer for either one. Five columns are pure noise, and a model that uses them reports a training error that is better than the truth. And a training error never says how the model will do on a house it has not seen.

In this project you build the whole pipeline of Part 1 and judge it the only honest way, on rows that took no part in fitting or choosing anything. You make four design decisions, and each one has a chapter behind it.

## The brief

You deliver, as functions in `starter.py`:

- a seeded split of the listings into training, validation, and test rows;
- a price model that is stable even though two columns are nearly copies, with its fee chosen on validation rows;
- a list of the columns that matter, found by the lasso;
- a probability of selling within 30 days, fitted with Newton's method.

The checks judge the answer in numbers. The price noise in the data has a standard deviation of 25 (thousands), so no model can score a test RMSE below about 25. A good answer reaches a test RMSE under 33. The lasso must set all five noise columns to exactly zero and keep area, age, the age gap, and distance. The classifier must beat the predictor that always says the training sell rate by at least 0.08 in mean log loss on the test rows.

## What you will use

- [Holding data back and scoring on it](/lessons/001-loss-and-honest-scores) tells you why the test rows stay sealed until the end.
- [Lost digits, and how QR and the SVD keep them](/lessons/004-stable-least-squares) explains why you must not solve the normal equations when two columns almost match, and gives you `fit_qr` and the SVD you reuse for ridge.
- [Ridge: a fee on large weights](/lessons/006-ridge-regression) defines the penalty that keeps weights small.
- [Ridge for every lambda](/lessons/006-ridge-regression) gives you every fee from one SVD, which makes a grid search cheap.
- [Lasso and soft thresholding](/lessons/007-lasso-and-fista) and [ISTA and FISTA](/lessons/007-lasso-and-fista) give you weights that are exactly zero.
- [How wrong is a probability? The log loss](/lessons/008-sigmoid-and-log-loss) is the score for the classifier, and [Curvature and Newton's method](/lessons/009-fitting-logistic-regression) is how you fit it.

## Meet the data

`make_data(seed)` returns a matrix `X` with 500 rows and 11 columns, the price in thousands of pounds, and a 0-or-1 label `sold`. Every column is made by a seeded generator, so seed 0 always gives the same houses. The columns, in order, are:

| Column | Meaning |
|---|---|
| `area_m2` | Floor area in square metres, between 45 and 220. |
| `area_ft2` | The same area in square feet, with a little noise added. |
| `rooms` | Number of rooms, which follows the area. |
| `age` | Age in years, between 0 and 80. |
| `age_gap_sq` | The squared distance of the age from 40 years. |
| `dist_km` | Distance to the centre in kilometres. |
| `noise_1` to `noise_5` | Standard normal noise. |

The data hides a planted truth, and your result is judged against it. The price depends on exactly four columns: $2.2$ per square metre of area, $-6$ per kilometre of distance, and a U-shaped effect of age built from $-1.0 \times \text{age} + 0.04 \times \text{age\_gap\_sq}$. Old and very young houses cost more than houses near 40 years old. Random noise with a standard deviation of 25 is added on top. The columns `area_ft2`, `rooms`, and the five noise columns have a true weight of zero.

The quadratic effect is why `age_gap_sq` exists. A linear model cannot bend, so you hand it the squared gap as a column and it stays linear in its inputs. The sell label is separate. Each house sells within 30 days with a probability that falls with distance, age, and area through a logistic curve, and about 54 percent of houses sell.

The duplicate is the trap. `area_ft2` equals 10.7639 times `area_m2` plus noise with a standard deviation of 2 square feet, which makes their correlation 0.999994. After standardising, the smallest singular value of the training matrix is 0.04 and the largest is 29.5. A weight vector can move a long way along the weak direction and barely change a prediction.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Left: price rises almost in a straight line with floor area, with scatter around it. Right: after removing the area and distance effects, price against age falls and then rises, a U shape with its low point near 40 years.">
<line x1="60" y1="204" x2="280" y2="204" class="stroke-muted"/>
<line x1="60" y1="44" x2="60" y2="204" class="stroke-muted"/>
<line x1="71.6" y1="204" x2="71.6" y2="208" class="stroke-muted"/><text x="71.6" y="222" class="fig-small" text-anchor="middle">50</text>
<line x1="129.5" y1="204" x2="129.5" y2="208" class="stroke-muted"/><text x="129.5" y="222" class="fig-small" text-anchor="middle">100</text>
<line x1="187.4" y1="204" x2="187.4" y2="208" class="stroke-muted"/><text x="187.4" y="222" class="fig-small" text-anchor="middle">150</text>
<line x1="245.3" y1="204" x2="245.3" y2="208" class="stroke-muted"/><text x="245.3" y="222" class="fig-small" text-anchor="middle">200</text>
<line x1="56" y1="204.0" x2="60" y2="204.0" class="stroke-muted"/><text x="52" y="208.0" class="fig-small" text-anchor="end">0</text>
<line x1="56" y1="150.7" x2="60" y2="150.7" class="stroke-muted"/><text x="52" y="154.7" class="fig-small" text-anchor="end">200</text>
<line x1="56" y1="97.3" x2="60" y2="97.3" class="stroke-muted"/><text x="52" y="101.3" class="fig-small" text-anchor="end">400</text>
<line x1="56" y1="44.0" x2="60" y2="44.0" class="stroke-muted"/><text x="52" y="48.0" class="fig-small" text-anchor="end">600</text>
<circle cx="145.8" cy="139.1" r="2.8" class="fill-a"/>
<circle cx="217.4" cy="103.3" r="2.8" class="fill-a"/>
<circle cx="225.8" cy="80.2" r="2.8" class="fill-a"/>
<circle cx="260.8" cy="101.3" r="2.8" class="fill-a"/>
<circle cx="119.7" cy="147.0" r="2.8" class="fill-a"/>
<circle cx="151.7" cy="156.6" r="2.8" class="fill-a"/>
<circle cx="138.0" cy="152.8" r="2.8" class="fill-a"/>
<circle cx="196.7" cy="134.5" r="2.8" class="fill-a"/>
<circle cx="90.7" cy="147.3" r="2.8" class="fill-a"/>
<circle cx="250.8" cy="57.5" r="2.8" class="fill-a"/>
<circle cx="184.6" cy="129.9" r="2.8" class="fill-a"/>
<circle cx="244.1" cy="71.4" r="2.8" class="fill-a"/>
<circle cx="145.2" cy="160.1" r="2.8" class="fill-a"/>
<circle cx="217.5" cy="100.6" r="2.8" class="fill-a"/>
<circle cx="182.7" cy="116.9" r="2.8" class="fill-a"/>
<circle cx="98.9" cy="170.5" r="2.8" class="fill-a"/>
<circle cx="131.2" cy="156.8" r="2.8" class="fill-a"/>
<circle cx="111.7" cy="179.8" r="2.8" class="fill-a"/>
<circle cx="175.9" cy="142.0" r="2.8" class="fill-a"/>
<circle cx="111.6" cy="157.5" r="2.8" class="fill-a"/>
<circle cx="179.4" cy="117.4" r="2.8" class="fill-a"/>
<circle cx="260.1" cy="98.6" r="2.8" class="fill-a"/>
<circle cx="163.3" cy="149.1" r="2.8" class="fill-a"/>
<circle cx="129.1" cy="151.5" r="2.8" class="fill-a"/>
<circle cx="97.4" cy="156.9" r="2.8" class="fill-a"/>
<circle cx="253.4" cy="113.8" r="2.8" class="fill-a"/>
<circle cx="206.8" cy="116.7" r="2.8" class="fill-a"/>
<circle cx="206.4" cy="120.5" r="2.8" class="fill-a"/>
<circle cx="110.9" cy="173.9" r="2.8" class="fill-a"/>
<circle cx="222.4" cy="139.2" r="2.8" class="fill-a"/>
<circle cx="186.2" cy="122.6" r="2.8" class="fill-a"/>
<circle cx="164.2" cy="135.5" r="2.8" class="fill-a"/>
<circle cx="99.2" cy="163.2" r="2.8" class="fill-a"/>
<circle cx="80.7" cy="180.3" r="2.8" class="fill-a"/>
<circle cx="267.4" cy="91.7" r="2.8" class="fill-a"/>
<circle cx="152.3" cy="160.3" r="2.8" class="fill-a"/>
<circle cx="183.1" cy="144.0" r="2.8" class="fill-a"/>
<circle cx="101.7" cy="156.7" r="2.8" class="fill-a"/>
<circle cx="102.8" cy="172.3" r="2.8" class="fill-a"/>
<circle cx="149.7" cy="154.9" r="2.8" class="fill-a"/>
<circle cx="236.3" cy="78.8" r="2.8" class="fill-a"/>
<circle cx="151.4" cy="126.4" r="2.8" class="fill-a"/>
<circle cx="135.6" cy="164.5" r="2.8" class="fill-a"/>
<circle cx="241.8" cy="102.5" r="2.8" class="fill-a"/>
<circle cx="182.7" cy="102.3" r="2.8" class="fill-a"/>
<circle cx="130.6" cy="125.2" r="2.8" class="fill-a"/>
<circle cx="69.4" cy="189.6" r="2.8" class="fill-a"/>
<circle cx="162.4" cy="122.2" r="2.8" class="fill-a"/>
<circle cx="243.6" cy="92.5" r="2.8" class="fill-a"/>
<circle cx="262.7" cy="83.2" r="2.8" class="fill-a"/>
<circle cx="204.7" cy="77.0" r="2.8" class="fill-a"/>
<circle cx="132.2" cy="119.0" r="2.8" class="fill-a"/>
<circle cx="194.3" cy="126.1" r="2.8" class="fill-a"/>
<circle cx="138.9" cy="161.4" r="2.8" class="fill-a"/>
<circle cx="217.4" cy="115.6" r="2.8" class="fill-a"/>
<circle cx="77.7" cy="187.6" r="2.8" class="fill-a"/>
<circle cx="264.8" cy="89.4" r="2.8" class="fill-a"/>
<circle cx="104.7" cy="163.3" r="2.8" class="fill-a"/>
<circle cx="232.2" cy="106.7" r="2.8" class="fill-a"/>
<circle cx="261.9" cy="84.4" r="2.8" class="fill-a"/>
<circle cx="133.9" cy="150.2" r="2.8" class="fill-a"/>
<circle cx="66.3" cy="183.8" r="2.8" class="fill-a"/>
<circle cx="143.4" cy="148.9" r="2.8" class="fill-a"/>
<circle cx="192.2" cy="110.0" r="2.8" class="fill-a"/>
<circle cx="124.2" cy="148.6" r="2.8" class="fill-a"/>
<circle cx="91.6" cy="147.0" r="2.8" class="fill-a"/>
<circle cx="152.0" cy="138.1" r="2.8" class="fill-a"/>
<circle cx="145.7" cy="137.3" r="2.8" class="fill-a"/>
<circle cx="205.3" cy="117.4" r="2.8" class="fill-a"/>
<circle cx="201.8" cy="125.9" r="2.8" class="fill-a"/>
<circle cx="253.0" cy="82.3" r="2.8" class="fill-a"/>
<circle cx="95.3" cy="186.0" r="2.8" class="fill-a"/>
<circle cx="139.8" cy="136.6" r="2.8" class="fill-a"/>
<circle cx="178.8" cy="140.9" r="2.8" class="fill-a"/>
<circle cx="214.1" cy="100.2" r="2.8" class="fill-a"/>
<circle cx="134.7" cy="164.0" r="2.8" class="fill-a"/>
<circle cx="264.0" cy="97.5" r="2.8" class="fill-a"/>
<circle cx="265.5" cy="80.0" r="2.8" class="fill-a"/>
<circle cx="131.8" cy="166.1" r="2.8" class="fill-a"/>
<circle cx="76.3" cy="157.8" r="2.8" class="fill-a"/>
<circle cx="161.5" cy="134.2" r="2.8" class="fill-a"/>
<circle cx="146.2" cy="130.8" r="2.8" class="fill-a"/>
<circle cx="169.7" cy="134.9" r="2.8" class="fill-a"/>
<circle cx="213.6" cy="82.0" r="2.8" class="fill-a"/>
<circle cx="103.8" cy="163.3" r="2.8" class="fill-a"/>
<circle cx="155.5" cy="144.9" r="2.8" class="fill-a"/>
<circle cx="166.4" cy="136.6" r="2.8" class="fill-a"/>
<circle cx="180.0" cy="139.8" r="2.8" class="fill-a"/>
<circle cx="91.6" cy="148.9" r="2.8" class="fill-a"/>
<circle cx="149.5" cy="131.9" r="2.8" class="fill-a"/>
<text x="170.0" y="242" class="fig-small" text-anchor="middle">floor area (m²)</text>
<text x="60" y="32" class="fig-small" text-anchor="start">price (thousands)</text>
<line x1="360" y1="204" x2="570" y2="204" class="stroke-muted"/>
<line x1="360" y1="44" x2="360" y2="204" class="stroke-muted"/>
<line x1="360.0" y1="204" x2="360.0" y2="208" class="stroke-muted"/><text x="360.0" y="222" class="fig-small" text-anchor="middle">0</text>
<line x1="412.5" y1="204" x2="412.5" y2="208" class="stroke-muted"/><text x="412.5" y="222" class="fig-small" text-anchor="middle">20</text>
<line x1="465.0" y1="204" x2="465.0" y2="208" class="stroke-muted"/><text x="465.0" y="222" class="fig-small" text-anchor="middle">40</text>
<line x1="517.5" y1="204" x2="517.5" y2="208" class="stroke-muted"/><text x="517.5" y="222" class="fig-small" text-anchor="middle">60</text>
<line x1="570.0" y1="204" x2="570.0" y2="208" class="stroke-muted"/><text x="570.0" y="222" class="fig-small" text-anchor="middle">80</text>
<line x1="356" y1="204.0" x2="360" y2="204.0" class="stroke-muted"/><text x="352" y="208.0" class="fig-small" text-anchor="end">-100</text>
<line x1="356" y1="140.0" x2="360" y2="140.0" class="stroke-muted"/><text x="352" y="144.0" class="fig-small" text-anchor="end">0</text>
<line x1="356" y1="76.0" x2="360" y2="76.0" class="stroke-muted"/><text x="352" y="80.0" class="fig-small" text-anchor="end">100</text>
<circle cx="532.1" cy="137.2" r="2.8" class="fill-a"/>
<circle cx="453.2" cy="110.6" r="2.8" class="fill-a"/>
<circle cx="439.6" cy="89.8" r="2.8" class="fill-a"/>
<circle cx="450.4" cy="135.2" r="2.8" class="fill-a"/>
<circle cx="367.0" cy="81.3" r="2.8" class="fill-a"/>
<circle cx="415.2" cy="145.8" r="2.8" class="fill-a"/>
<circle cx="505.0" cy="116.7" r="2.8" class="fill-a"/>
<circle cx="423.6" cy="123.3" r="2.8" class="fill-a"/>
<circle cx="364.7" cy="68.4" r="2.8" class="fill-a"/>
<circle cx="384.2" cy="68.8" r="2.8" class="fill-a"/>
<circle cx="524.0" cy="94.8" r="2.8" class="fill-a"/>
<circle cx="392.5" cy="70.0" r="2.8" class="fill-a"/>
<circle cx="539.8" cy="129.2" r="2.8" class="fill-a"/>
<circle cx="423.5" cy="108.1" r="2.8" class="fill-a"/>
<circle cx="377.1" cy="95.0" r="2.8" class="fill-a"/>
<circle cx="395.1" cy="94.1" r="2.8" class="fill-a"/>
<circle cx="568.1" cy="110.6" r="2.8" class="fill-a"/>
<circle cx="498.3" cy="177.7" r="2.8" class="fill-a"/>
<circle cx="562.0" cy="130.7" r="2.8" class="fill-a"/>
<circle cx="384.7" cy="77.0" r="2.8" class="fill-a"/>
<circle cx="428.7" cy="82.2" r="2.8" class="fill-a"/>
<circle cx="511.4" cy="123.4" r="2.8" class="fill-a"/>
<circle cx="447.4" cy="114.6" r="2.8" class="fill-a"/>
<circle cx="535.1" cy="130.8" r="2.8" class="fill-a"/>
<circle cx="361.8" cy="67.3" r="2.8" class="fill-a"/>
<circle cx="477.6" cy="147.6" r="2.8" class="fill-a"/>
<circle cx="512.2" cy="132.0" r="2.8" class="fill-a"/>
<circle cx="480.8" cy="111.6" r="2.8" class="fill-a"/>
<circle cx="471.1" cy="121.1" r="2.8" class="fill-a"/>
<circle cx="463.0" cy="161.9" r="2.8" class="fill-a"/>
<circle cx="513.6" cy="127.2" r="2.8" class="fill-a"/>
<circle cx="412.2" cy="87.7" r="2.8" class="fill-a"/>
<circle cx="473.8" cy="120.5" r="2.8" class="fill-a"/>
<circle cx="421.6" cy="97.3" r="2.8" class="fill-a"/>
<circle cx="500.9" cy="164.1" r="2.8" class="fill-a"/>
<circle cx="401.8" cy="134.0" r="2.8" class="fill-a"/>
<circle cx="439.6" cy="151.9" r="2.8" class="fill-a"/>
<circle cx="563.5" cy="108.4" r="2.8" class="fill-a"/>
<circle cx="552.2" cy="126.6" r="2.8" class="fill-a"/>
<circle cx="503.2" cy="135.1" r="2.8" class="fill-a"/>
<circle cx="383.2" cy="74.9" r="2.8" class="fill-a"/>
<circle cx="423.8" cy="98.8" r="2.8" class="fill-a"/>
<circle cx="487.3" cy="142.5" r="2.8" class="fill-a"/>
<circle cx="531.4" cy="127.3" r="2.8" class="fill-a"/>
<circle cx="385.6" cy="86.1" r="2.8" class="fill-a"/>
<circle cx="389.7" cy="70.8" r="2.8" class="fill-a"/>
<circle cx="450.3" cy="116.9" r="2.8" class="fill-a"/>
<circle cx="475.4" cy="94.8" r="2.8" class="fill-a"/>
<circle cx="390.8" cy="86.9" r="2.8" class="fill-a"/>
<circle cx="393.0" cy="89.3" r="2.8" class="fill-a"/>
<circle cx="394.4" cy="62.5" r="2.8" class="fill-a"/>
<circle cx="379.9" cy="72.3" r="2.8" class="fill-a"/>
<circle cx="426.6" cy="107.9" r="2.8" class="fill-a"/>
<circle cx="478.0" cy="132.3" r="2.8" class="fill-a"/>
<circle cx="521.2" cy="150.4" r="2.8" class="fill-a"/>
<circle cx="554.0" cy="120.7" r="2.8" class="fill-a"/>
<circle cx="546.0" cy="115.6" r="2.8" class="fill-a"/>
<circle cx="560.1" cy="129.1" r="2.8" class="fill-a"/>
<circle cx="503.0" cy="118.7" r="2.8" class="fill-a"/>
<circle cx="520.2" cy="135.8" r="2.8" class="fill-a"/>
<circle cx="502.9" cy="117.8" r="2.8" class="fill-a"/>
<circle cx="502.6" cy="135.3" r="2.8" class="fill-a"/>
<circle cx="466.1" cy="107.4" r="2.8" class="fill-a"/>
<circle cx="491.2" cy="128.5" r="2.8" class="fill-a"/>
<circle cx="550.8" cy="136.9" r="2.8" class="fill-a"/>
<circle cx="369.5" cy="91.1" r="2.8" class="fill-a"/>
<circle cx="428.9" cy="144.2" r="2.8" class="fill-a"/>
<circle cx="439.6" cy="105.9" r="2.8" class="fill-a"/>
<circle cx="561.2" cy="127.4" r="2.8" class="fill-a"/>
<circle cx="492.9" cy="121.9" r="2.8" class="fill-a"/>
<circle cx="557.6" cy="130.2" r="2.8" class="fill-a"/>
<circle cx="546.0" cy="144.0" r="2.8" class="fill-a"/>
<circle cx="559.9" cy="78.2" r="2.8" class="fill-a"/>
<circle cx="560.9" cy="119.6" r="2.8" class="fill-a"/>
<circle cx="453.2" cy="119.0" r="2.8" class="fill-a"/>
<circle cx="436.9" cy="128.1" r="2.8" class="fill-a"/>
<circle cx="559.4" cy="128.1" r="2.8" class="fill-a"/>
<circle cx="394.9" cy="87.5" r="2.8" class="fill-a"/>
<circle cx="395.4" cy="128.8" r="2.8" class="fill-a"/>
<circle cx="418.6" cy="96.3" r="2.8" class="fill-a"/>
<circle cx="429.4" cy="118.4" r="2.8" class="fill-a"/>
<circle cx="405.7" cy="93.4" r="2.8" class="fill-a"/>
<circle cx="540.6" cy="128.7" r="2.8" class="fill-a"/>
<circle cx="404.1" cy="84.9" r="2.8" class="fill-a"/>
<circle cx="520.1" cy="135.6" r="2.8" class="fill-a"/>
<circle cx="412.2" cy="94.1" r="2.8" class="fill-a"/>
<circle cx="432.5" cy="106.3" r="2.8" class="fill-a"/>
<circle cx="435.6" cy="121.7" r="2.8" class="fill-a"/>
<circle cx="434.1" cy="99.8" r="2.8" class="fill-a"/>
<circle cx="402.7" cy="122.3" r="2.8" class="fill-a"/>
<text x="465.0" y="242" class="fig-small" text-anchor="middle">age (years)</text>
<text x="360" y="32" class="fig-small" text-anchor="start">price minus area and distance effects</text>
<polyline points="360.0,60.6 365.2,65.9 370.5,71.0 375.8,75.8 381.0,80.5 386.2,85.0 391.5,89.2 396.8,93.3 402.0,97.1 407.2,100.7 412.5,104.2 417.8,107.4 423.0,110.4 428.2,113.2 433.5,115.8 438.8,118.2 444.0,120.4 449.2,122.4 454.5,124.2 459.8,125.8 465.0,127.2 470.2,128.4 475.5,129.4 480.8,130.1 486.0,130.7 491.2,131.0 496.5,131.2 501.8,131.1 507.0,130.9 512.2,130.4 517.5,129.8 522.8,128.9 528.0,127.8 533.2,126.5 538.5,125.0 543.8,123.4 549.0,121.5 554.2,119.4 559.5,117.1 564.8,114.6 570.0,111.8" class="stroke-b" fill="none" stroke-width="2"/>
</svg>
Figure 1. Price climbs steadily with area, while the age effect only shows once the area and distance effects are removed: the orange line is the planted U shape, and the dots follow it.
```

## Plan the build

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Pipeline: 500 listings are split into train, validation and test, standardised with training statistics, then feed four fits: QR least squares, ridge over a lambda grid with lambda chosen on validation rows, a lasso that picks columns, and a Newton logistic model. The test rows are touched once, at the end.">
<defs><marker id="p023-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="10" y="120" width="86" height="56" rx="6" class="fill-soft-a stroke-a"/>
<text x="53" y="144" class="fig-bold" text-anchor="middle">make_data</text><text x="53" y="162" class="fig-small" text-anchor="middle">500 listings</text>
<line x1="96" y1="148" x2="116" y2="148" class="stroke-ink" marker-end="url(#p023-arrow)"/>
<rect x="118" y="120" width="96" height="56" rx="6" class="fill-soft-a stroke-a"/>
<text x="166" y="144" class="fig-bold" text-anchor="middle">split</text><text x="166" y="162" class="fig-small" text-anchor="middle">300 / 100 / 100</text>
<line x1="214" y1="148" x2="234" y2="148" class="stroke-ink" marker-end="url(#p023-arrow)"/>
<rect x="236" y="120" width="104" height="56" rx="6" class="fill-soft-c stroke-c"/>
<text x="288" y="144" class="fig-bold" text-anchor="middle">standardise</text><text x="288" y="162" class="fig-small" text-anchor="middle">train mean and std</text>
<line x1="340" y1="148" x2="372" y2="148" class="stroke-ink"/>
<line x1="372" y1="40" x2="372" y2="256" class="stroke-ink"/>
<line x1="372" y1="40" x2="392" y2="40" class="stroke-ink" marker-end="url(#p023-arrow)"/>
<line x1="372" y1="112" x2="392" y2="112" class="stroke-ink" marker-end="url(#p023-arrow)"/>
<line x1="372" y1="184" x2="392" y2="184" class="stroke-ink" marker-end="url(#p023-arrow)"/>
<line x1="372" y1="256" x2="392" y2="256" class="stroke-ink" marker-end="url(#p023-arrow)"/>
<rect x="394" y="14" width="196" height="52" rx="6" class="fill-soft-b stroke-b"/>
<text x="492" y="36" class="fig-bold" text-anchor="middle">fit_qr</text><text x="492" y="54" class="fig-small" text-anchor="middle">stable least squares</text>
<rect x="394" y="86" width="196" height="52" rx="6" class="fill-soft-b stroke-b"/>
<text x="492" y="108" class="fig-bold" text-anchor="middle">ridge_path, choose_lambda</text><text x="492" y="126" class="fig-small" text-anchor="middle">one SVD, pick on validation</text>
<rect x="394" y="158" width="196" height="52" rx="6" class="fill-soft-b stroke-b"/>
<text x="492" y="180" class="fig-bold" text-anchor="middle">fista_lasso</text><text x="492" y="198" class="fig-small" text-anchor="middle">which columns matter</text>
<rect x="394" y="230" width="196" height="52" rx="6" class="fill-soft-d stroke-d"/>
<text x="492" y="252" class="fig-bold" text-anchor="middle">fit_logistic_newton</text><text x="492" y="270" class="fig-small" text-anchor="middle">sells in 30 days, log loss</text>
<rect x="10" y="236" width="330" height="46" rx="6" class="fill-sunk stroke-muted dashed"/>
<text x="175" y="256" class="fig-small" text-anchor="middle">test rows: untouched until the last line,</text><text x="175" y="272" class="fig-small" text-anchor="middle">then scored once</text>
</svg>
Figure 2. Data flows left to right through the split and the scaler, then into four fits. The test rows sit outside every fit and are scored once at the end.
```

The order of the stages is not arbitrary. You split first, because every later choice must be made without the test rows, and a split made after fitting cannot undo what the fit already saw. You standardise second, and with training statistics only, because the validation and test rows stand in for houses the agency has not met, and a mean computed from them is knowledge the agency would not have.

You standardise before every fit, not only for convenience. Ridge and the lasso charge a fee on weight size, and a weight's size depends on its column's units. A column in square feet gets a weight about ten times smaller than the same column in square metres, so it would be charged ten times less. Standardising puts every column on one scale so the fee treats them alike.

Then the four fits run side by side on the same standardised training rows. QR is the baseline. Ridge competes over a grid of fees, and the validation rows pick the winner. The lasso picks columns. The logistic model answers the second question. Only after all of that do you score the test rows, once.

## Build it step by step

> [!RECAP]
> - Hold data back first, and fit every statistic and weight on the training rows only ([Honest scores](/lessons/001-loss-and-honest-scores)).
> - Solve least squares through QR, because $X^\top X$ squares the condition number ([Lost digits, and how QR and the SVD keep them](/lessons/004-stable-least-squares)).
> - One SVD gives ridge weights for every fee, and the validation rows choose the fee ([Ridge for every lambda](/lessons/006-ridge-regression)).
> - The lasso fee at which everything is zero is $\max|X^\top y|$, and FISTA finds exact zeros ([ISTA and FISTA](/lessons/007-lasso-and-fista)).
> - Newton's method fits logistic regression in a few steps, and log loss scores it ([Curvature and Newton's method](/lessons/009-fitting-logistic-regression)).

`make_data`, `sigmoid`, `softplus`, `logistic_loss`, `soft_threshold`, `rmse`, the ridge grid `LAMBDAS`, and the lasso strength `LASSO_FRACTION` are given. You write the rest, in this order.

1. `split_indices(n, seed)` takes a row count and a seed and returns three arrays of row numbers: training, validation, and test, with 60, 20, and 20 percent of the rows. Shuffle once with `np.random.default_rng(seed).permutation(n)` and cut the result. The common mistake is to slice the first 300 rows as they come, which is not a shuffle, so any ordering in the data lands in one group.

2. `fit_scaler(X_train)` returns a mean and a standard deviation for each column, and `apply_scaler(X, mean, std)` uses them. The scaler is fitted on `X[train]` and applied to all three groups. The common mistake is to call `fit_scaler` on the full matrix, which leaks test statistics into training. Replace a zero standard deviation with 1 so a constant column does not divide by zero.

3. `fit_qr(X, y)` returns the least-squares weights from `np.linalg.qr(X, mode="reduced")` and a triangular solve with `scipy.linalg.solve_triangular`. The check uses columns that agree to seven digits, where the normal equations lose the answer. The common mistake is to write `np.linalg.solve(X.T @ X, X.T @ y)` because it is shorter.

4. `ridge_path(X, y, lambdas)` returns one row of weights per fee, shape `(m, d)`, from a single `np.linalg.svd`. Compute `c = U.T @ y`, then each row is `(s / (s**2 + lam) * c) @ Vt`. Broadcast over the fees instead of looping with a fresh solve. The common mistake is to take the SVD again for every fee.

5. `choose_lambda(X_tr, y_tr, X_val, y_val, lambdas)` fits the whole path on the training rows, scores each row of weights on the validation rows with `rmse`, and returns the best fee, all validation errors, and the best weights. The common mistake is to score on the training rows, where the smallest fee always wins.

6. `lambda_max(X, y)` returns `np.abs(X.T @ y).max()`. At that fee, every lasso weight is exactly zero, and any smaller fee lets the first column in. `fista_lasso(X, y, lam, iters)` runs the FISTA loop from [ISTA and FISTA](/lessons/007-lasso-and-fista) with step $1/L$, where $L$ is the largest eigenvalue of $X^\top X$. Return `w`, not the extrapolated point `z`. The common mistake is to return `z`, which is not the output of the soft threshold and so holds no exact zeros.

7. `add_intercept(X)` puts a column of ones in front, and `fit_logistic_newton(X, y, ridge)` repeats $w \leftarrow w - H^{-1} g$ from zero, where $g = X^\top(p - y) + \text{ridge} \cdot w$ and $H = X^\top \operatorname{diag}(p(1-p)) X + \text{ridge} \cdot I$. Keep the ridge at 1: the area columns nearly match, so a ridge of zero makes $H$ close to singular. The common mistake is to build the $n \times n$ diagonal matrix. Scale the rows with `X * (p * (1 - p))[:, None]` instead.

8. `mean_log_loss(p, y)` returns the average of $-[y\ln p + (1-y)\ln(1-p)]$ after clipping `p` into $[\epsilon, 1-\epsilon]$. The common mistake is to skip the clip, so a confident wrong answer gives infinity.

9. `run_pipeline(seed)` wires it together and returns the dictionary the docstring lists. Centre the price with the training mean, fit the regression models on the standardised training rows, and score on the test rows. The common mistake is to centre with the mean of all prices, or to let the test rows choose the fee.

## Read your results

Run `starter.py` once the functions work. For seed 0 you should see numbers close to these.

| Quantity | Seed 0 | What it tells the agency |
|---|---|---|
| Test RMSE, predicting the mean price | 122.3 | The price varies a lot, so a model has plenty to explain. |
| Training RMSE, QR least squares | 24.3 | Below the noise of 25, which is the optimism of a training score. |
| Test RMSE, QR least squares | 27.8 | The honest error: about 28 thousand off on an unseen house. |
| Test RMSE, ridge (fee 2.5) | 27.8 | The same accuracy, with stable weights. |
| Test RMSE, lasso (5 percent of the maximum fee) | 31.0 | Shrunk too hard to predict well, which is expected. |
| Test log loss, Newton logistic | 0.507 | The quick-sale probabilities carry real information. |
| Test log loss, base rate | 0.687 | The score of saying "54 percent" for every house. |

Read the regression rows in order. The honest test error is above the training error by 3.5, which is the cost of fitting 11 weights to 300 rows. A test error well below 25 means the test rows leaked into fitting, because no model can beat the noise.

Ridge and QR have the same test error here, and that is a true result, not a mistake. With 300 rows and 11 columns there is enough data to pin down the predictions. What differs is the weights. QR gives the two area columns $-27$ and $+142$ in standardised units, because the weak direction has a tiny singular value and the fit amplifies noise along it. Ridge gives 56 and 56. Across seeds 0 to 4, the largest QR weight ranges from 142 to 889, while the largest ridge weight ranges from 50 to 172. A price model for the agency should not swing because one listing changed by a few square feet.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Paired bars of ridge and lasso weights for the eleven standardised columns. Both give the area columns about 45 to 60 and distance about minus 25 to minus 31. The lasso bars for rooms and the five noise columns are exactly zero, while ridge leaves small non-zero bars there.">
<line x1="50" y1="144.2" x2="570" y2="144.2" class="stroke-muted"/>
<line x1="46" y1="206.0" x2="50" y2="206.0" class="stroke-muted"/><text x="42" y="210.0" class="fig-small" text-anchor="end">-40</text>
<line x1="46" y1="144.2" x2="50" y2="144.2" class="stroke-muted"/><text x="42" y="148.2" class="fig-small" text-anchor="end">0</text>
<line x1="46" y1="82.4" x2="50" y2="82.4" class="stroke-muted"/><text x="42" y="86.4" class="fig-small" text-anchor="end">40</text>
<rect x="56.6" y="57.6" width="16" height="86.5" class="fill-d"/>
<rect x="74.6" y="73.3" width="16" height="70.9" class="fill-b"/>
<text x="73.6" y="230" class="fig-small" text-anchor="middle">m²</text>
<rect x="103.9" y="57.5" width="16" height="86.7" class="fill-d"/>
<rect x="121.9" y="52.8" width="16" height="91.4" class="fill-b"/>
<text x="120.9" y="230" class="fig-small" text-anchor="middle">ft²</text>
<rect x="151.2" y="144.2" width="16" height="4.1" class="fill-d"/>
<rect x="169.2" y="144.2" width="16" height="1.5" class="fill-b"/>
<text x="168.2" y="230" class="fig-small" text-anchor="middle">rooms</text>
<rect x="198.5" y="144.2" width="16" height="36.4" class="fill-d"/>
<rect x="216.5" y="144.2" width="16" height="27.7" class="fill-b"/>
<text x="215.5" y="230" class="fig-small" text-anchor="middle">age</text>
<rect x="245.7" y="118.0" width="16" height="26.2" class="fill-d"/>
<rect x="263.7" y="127.1" width="16" height="17.1" class="fill-b"/>
<text x="262.7" y="230" class="fig-small" text-anchor="middle">age gap²</text>
<rect x="293.0" y="144.2" width="16" height="48.2" class="fill-d"/>
<rect x="311.0" y="144.2" width="16" height="39.5" class="fill-b"/>
<text x="310.0" y="230" class="fig-small" text-anchor="middle">dist</text>
<rect x="340.3" y="143.0" width="16" height="1.5" class="fill-d"/>
<rect x="358.3" y="144.2" width="16" height="1.5" class="fill-b"/>
<text x="357.3" y="230" class="fig-small" text-anchor="middle">n1</text>
<rect x="387.5" y="143.0" width="16" height="1.5" class="fill-d"/>
<rect x="405.5" y="144.2" width="16" height="1.5" class="fill-b"/>
<text x="404.5" y="230" class="fig-small" text-anchor="middle">n2</text>
<rect x="434.8" y="143.4" width="16" height="1.5" class="fill-d"/>
<rect x="452.8" y="144.2" width="16" height="1.5" class="fill-b"/>
<text x="451.8" y="230" class="fig-small" text-anchor="middle">n3</text>
<rect x="482.1" y="144.0" width="16" height="1.5" class="fill-d"/>
<rect x="500.1" y="144.2" width="16" height="1.5" class="fill-b"/>
<text x="499.1" y="230" class="fig-small" text-anchor="middle">n4</text>
<rect x="529.4" y="144.2" width="16" height="1.5" class="fill-d"/>
<rect x="547.4" y="144.2" width="16" height="1.5" class="fill-b"/>
<text x="546.4" y="230" class="fig-small" text-anchor="middle">n5</text>
<rect x="380" y="8" width="12" height="12" class="fill-d"/><text x="398" y="19" class="fig-small">ridge</text><rect x="460" y="8" width="12" height="12" class="fill-b"/><text x="478" y="19" class="fig-small">lasso</text>
</svg>
Figure 3. Ridge gives every column some weight, including the noise. The lasso sets rooms and all five noise columns to exactly zero, and splits the area weight between the two copies.
```

The figure shows the lasso's view. The lasso keeps five columns: `area_m2`, `area_ft2`, `age`, `age_gap_sq`, and `dist_km`. It sets `rooms` and all five noise columns to exactly zero. It gives the area pair 46 and 59, which add up to about 105, close to ridge's 112, because the two columns are one measurement and the lasso splits the credit between them. On other seeds the lasso gives the whole weight to one of the pair. Treat the pair as a single feature.

The lasso's own test error is worse (31.0) because the fee shrinks the kept weights. Use it to decide which columns to measure, and refit without the penalty on the chosen columns if you want its predictions.

For the classifier, a test log loss of 0.507 against 0.687 means a gain of 0.18. Across seeds 0 to 4 the gains are 0.18, 0.22, 0.17, 0.24, and 0.23. If your gain is near zero, you probably fitted the model without the intercept, or scored it on the training rows.

What a wrong number usually means:

- A ridge test RMSE near 64 usually means you skipped the standardisation. The fee then charges the columns unequally and the grid has no good fee, and the lasso on seed 0 then keeps only `area_ft2` and `age_gap_sq` and drops age and distance.
- A noise column with a non-zero lasso weight points to a fee that is too small, or to `fista_lasso` returning `z` instead of `w`.
- A test RMSE far below 25 points to test rows used while fitting or choosing, because no model can beat the noise.

## Take it further

- Replace the single validation split with 5-fold cross-validation to choose lambda. You will see how much the chosen fee moves from one split to another.
- Refit least squares on only the columns the lasso kept, then compare its test RMSE with ridge. You will see how much of the lasso's gap is shrinkage rather than lost information.
- Add a column `age_gap_sq` times `area_m2` and ask the lasso whether it matters. You will learn what the lasso says about a feature that has no true weight.
- Move the classifier threshold from 0.5 to 0.7 and count how many fast sales you call correctly. You will see the trade-off that the log loss hides.
- Make the duplicate noisier, with a standard deviation of 20 square feet, and rerun the QR weights. You will see the instability fade as the two columns become distinct.

## The exercise

You write `split_indices`, `fit_scaler`, `apply_scaler`, `fit_qr`, `ridge_path`, `choose_lambda`, `lambda_max`, `fista_lasso`, `add_intercept`, `fit_logistic_newton`, `mean_log_loss`, and `run_pipeline`.

The first checks test each stage alone. They check that the split is seeded, disjoint, and complete, and that the scaler uses training statistics. They check that `fit_qr` recovers weights from columns that agree to seven digits, that each row of `ridge_path` matches a direct ridge solve, and that `choose_lambda` returns the validation winner. They check that `fista_lasso` meets the optimality conditions of the lasso, and that Newton's method reaches a zero finite-difference gradient. The last three checks run `run_pipeline` on seeds 0 to 4. They require a test RMSE under 33 and above the noise floor, exact zeros on the noise columns with the four planted columns kept, and a log-loss gain of at least 0.08.
