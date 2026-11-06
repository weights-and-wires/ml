---
title: "Power laws and the straight line on log paper"
slug: 102-power-laws-log-log
part: "Part 7 — Large language models"
summary: A power law bends on ordinary axes but is a straight line on log axes, so a least-squares slope gives its exponent and predicts a bigger model.
minutes: 35
gpu: false
objectives:
  - Explain why equal multiplications of size give equal multiplications of loss, and compute the factor for one doubling.
  - Turn a power law into a straight line by taking logs of both axes, and read the exponent as the slope.
  - Fit the exponent and coefficient by hand on three points, then with numpy.
  - Explain why a small error in the fitted slope grows with the distance you extrapolate.
prereqs:
  - 005-least-squares-line
  - 003-fitting-curves
  - 101-fertility-and-fairness
exercise: 102-power-laws-log-log
---

# Power laws and the straight line on log paper

A lab trains five language models, from 10 million parameters up to 1 billion, and writes down how far each one's loss sits above its floor, the lowest loss any model could reach, which no amount of size removes. The numbers are 1.67, 1.15, 0.76, 0.52, and 0.35. Each model costs more than the last, so the curve through them bends and flattens. The question is what a model with 10 billion parameters would score, because a run that size costs about ten times more than the 1-billion model, and the lab wants to know whether it is worth paying for before it starts.

You could draw a smooth curve by eye and continue it, but two people would continue it two ways. You could fit a polynomial, and the chapter on fitting curves showed what happens to a polynomial a little beyond the data it was fitted on: it swings away. You could fit a straight line to the raw numbers, and it would cross zero somewhere and predict a negative loss, which is impossible. What the lab needs is a curve with a small number of knobs that matches the shape of the measurements and keeps behaving past their edge.

This chapter gives that curve, called a power law, and a trick that turns it into a straight line so that the least-squares fit you already know finds its two knobs. The next chapter uses the fit to say what the loss of a language model depends on.

## What you already know

- A least-squares line has a slope and an intercept, and both come from one short formula over centred data.
- The logarithm turns a product into a sum, so $\ln(ab) = \ln a + \ln b$, and a power into a product, so $\ln(x^k) = k \ln x$.
- A high-degree polynomial fitted on a range can swing wildly a little beyond it.
- A model's loss is a single number that falls as the model improves, and a floor under it cannot be crossed.
- A tokeniser decides how many tokens a text costs, and the training data of a language model is counted in tokens.

## The picture: only the axes change

```figure
<svg viewBox="0 0 600 256" role="img" aria-label="The same five points on straight axes, where they bend, and on log axes, where they fall on a straight line">
<text x="157.0" y="24.0" text-anchor="middle" class="fig-bold">Straight axes</text>
<text x="447.0" y="24.0" text-anchor="middle" class="fig-bold">Both axes on a log scale</text>
<line x1="50.0" y1="210.0" x2="265.0" y2="210.0" class="stroke-muted"/><line x1="50.0" y1="50.0" x2="50.0" y2="210.0" class="stroke-muted"/><line x1="50.0" y1="50.0" x2="50.0" y2="210.0" class="stroke-grid"/><text x="50.0" y="226.0" text-anchor="middle" class="fig-small">0</text><line x1="103.8" y1="50.0" x2="103.8" y2="210.0" class="stroke-grid"/><text x="103.8" y="226.0" text-anchor="middle" class="fig-small">250M</text><line x1="157.5" y1="50.0" x2="157.5" y2="210.0" class="stroke-grid"/><text x="157.5" y="226.0" text-anchor="middle" class="fig-small">500M</text><line x1="211.2" y1="50.0" x2="211.2" y2="210.0" class="stroke-grid"/><text x="211.2" y="226.0" text-anchor="middle" class="fig-small">750M</text><line x1="265.0" y1="50.0" x2="265.0" y2="210.0" class="stroke-grid"/><text x="265.0" y="226.0" text-anchor="middle" class="fig-small">1B</text><line x1="50.0" y1="210.0" x2="265.0" y2="210.0" class="stroke-grid"/><text x="44.0" y="214.0" text-anchor="end" class="fig-small">0</text><line x1="50.0" y1="165.6" x2="265.0" y2="165.6" class="stroke-grid"/><text x="44.0" y="169.6" text-anchor="end" class="fig-small">0.5</text><line x1="50.0" y1="121.1" x2="265.0" y2="121.1" class="stroke-grid"/><text x="44.0" y="125.1" text-anchor="end" class="fig-small">1</text><line x1="50.0" y1="76.7" x2="265.0" y2="76.7" class="stroke-grid"/><text x="44.0" y="80.7" text-anchor="end" class="fig-small">1.5</text>
<polyline points="52.1,61.8 53.2,80.8 54.3,92.8 55.4,101.3 56.4,107.9 57.5,113.1 58.6,117.4 59.6,121.0 60.7,124.1 61.8,126.9 62.8,129.3 63.9,131.5 65.0,133.4 66.1,135.2 67.1,136.8 68.2,138.3 69.3,139.7 70.3,141.0 71.4,142.1 72.5,143.3 73.5,144.3 74.6,145.3 75.7,146.2 76.8,147.1 77.8,147.9 78.9,148.7 80.0,149.5 81.0,150.2 82.1,150.9 83.2,151.5 84.2,152.2 85.3,152.8 86.4,153.3 87.4,153.9 88.5,154.4 89.6,154.9 90.7,155.4 91.7,155.9 92.8,156.4 93.9,156.8 94.9,157.3 96.0,157.7 97.1,158.1 98.1,158.5 99.2,158.9 100.3,159.2 101.4,159.6 102.4,160.0 103.5,160.3 104.6,160.6 105.6,161.0 106.7,161.3 107.8,161.6 108.8,161.9 109.9,162.2 111.0,162.5 112.0,162.7 113.1,163.0 114.2,163.3 115.3,163.6 116.3,163.8 117.4,164.1 118.5,164.3 119.5,164.5 120.6,164.8 121.7,165.0 122.7,165.2 123.8,165.5 124.9,165.7 126.0,165.9 127.0,166.1 128.1,166.3 129.2,166.5 130.2,166.7 131.3,166.9 132.4,167.1 133.4,167.3 134.5,167.5 135.6,167.6 136.6,167.8 137.7,168.0 138.8,168.2 139.9,168.3 140.9,168.5 142.0,168.7 143.1,168.8 144.1,169.0 145.2,169.1 146.3,169.3 147.3,169.5 148.4,169.6 149.5,169.8 150.6,169.9 151.6,170.0 152.7,170.2 153.8,170.3 154.8,170.5 155.9,170.6 157.0,170.7 158.0,170.9 159.1,171.0 160.2,171.1 161.2,171.3 162.3,171.4 163.4,171.5 164.5,171.6 165.5,171.8 166.6,171.9 167.7,172.0 168.7,172.1 169.8,172.2 170.9,172.3 171.9,172.4 173.0,172.6 174.1,172.7 175.2,172.8 176.2,172.9 177.3,173.0 178.4,173.1 179.4,173.2 180.5,173.3 181.6,173.4 182.6,173.5 183.7,173.6 184.8,173.7 185.8,173.8 186.9,173.9 188.0,174.0 189.1,174.1 190.1,174.2 191.2,174.3 192.3,174.4 193.3,174.5 194.4,174.5 195.5,174.6 196.5,174.7 197.6,174.8 198.7,174.9 199.8,175.0 200.8,175.1 201.9,175.1 203.0,175.2 204.0,175.3 205.1,175.4 206.2,175.5 207.2,175.6 208.3,175.6 209.4,175.7 210.5,175.8 211.5,175.9 212.6,175.9 213.7,176.0 214.7,176.1 215.8,176.2 216.9,176.2 217.9,176.3 219.0,176.4 220.1,176.5 221.1,176.5 222.2,176.6 223.3,176.7 224.4,176.7 225.4,176.8 226.5,176.9 227.6,177.0 228.6,177.0 229.7,177.1 230.8,177.2 231.8,177.2 232.9,177.3 234.0,177.3 235.1,177.4 236.1,177.5 237.2,177.5 238.3,177.6 239.3,177.7 240.4,177.7 241.5,177.8 242.5,177.8 243.6,177.9 244.7,178.0 245.7,178.0 246.8,178.1 247.9,178.1 249.0,178.2 250.0,178.3 251.1,178.3 252.2,178.4 253.2,178.4 254.3,178.5 255.4,178.5 256.4,178.6 257.5,178.7 258.6,178.7 259.7,178.8 260.7,178.8 261.8,178.9 262.9,178.9 263.9,179.0 265.0,179.0" fill="none" class="stroke-muted dashed" />
<circle cx="52.1" cy="61.8" r="5" class="fill-a"/><circle cx="56.5" cy="108.0" r="5" class="fill-a"/><circle cx="71.5" cy="142.3" r="5" class="fill-a"/><circle cx="114.5" cy="163.4" r="5" class="fill-a"/><circle cx="265.0" cy="179.0" r="5" class="fill-a"/>
<text x="80.1" y="103.3" text-anchor="start" class="fig-small">three small models</text>
<text x="80.1" y="118.3" text-anchor="start" class="fig-small">bunch up here</text>
<line x1="335.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-muted"/><line x1="335.0" y1="50.0" x2="335.0" y2="210.0" class="stroke-muted"/><line x1="335.0" y1="50.0" x2="335.0" y2="210.0" class="stroke-grid"/><text x="335.0" y="226.0" text-anchor="middle" class="fig-small">10M</text><line x1="447.5" y1="50.0" x2="447.5" y2="210.0" class="stroke-grid"/><text x="447.5" y="226.0" text-anchor="middle" class="fig-small">100M</text><line x1="560.0" y1="50.0" x2="560.0" y2="210.0" class="stroke-grid"/><text x="560.0" y="226.0" text-anchor="middle" class="fig-small">1B</text><line x1="335.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-grid"/><text x="329.0" y="214.0" text-anchor="end" class="fig-small">0.3</text><line x1="335.0" y1="166.9" x2="560.0" y2="166.9" class="stroke-grid"/><text x="329.0" y="170.9" text-anchor="end" class="fig-small">0.5</text><line x1="335.0" y1="108.5" x2="560.0" y2="108.5" class="stroke-grid"/><text x="329.0" y="112.5" text-anchor="end" class="fig-small">1</text><line x1="335.0" y1="50.0" x2="560.0" y2="50.0" class="stroke-grid"/><text x="329.0" y="54.0" text-anchor="end" class="fig-small">2</text>
<polyline points="335.0,65.3 560.0,197.4" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="335.0" cy="65.3" r="5" class="fill-a"/><circle cx="388.7" cy="96.8" r="5" class="fill-a"/><circle cx="447.5" cy="131.4" r="5" class="fill-a"/><circle cx="501.2" cy="162.9" r="5" class="fill-a"/><circle cx="560.0" cy="197.4" r="5" class="fill-a"/>
<line x1="447.5" y1="131.4" x2="560.0" y2="131.4" class="stroke-d" />
<line x1="560.0" y1="131.4" x2="560.0" y2="197.4" class="stroke-d" />
<text x="503.8" y="125.4" text-anchor="middle" class="fig-small">×10 in size</text>
<text x="554.0" y="168.4" text-anchor="end" class="fig-small">×0.46</text>
<text x="157.5" y="244.0" text-anchor="middle" class="fig-small">model size (parameters)</text>
<text x="447.5" y="244.0" text-anchor="middle" class="fig-small">model size (parameters)</text>
<text x="12.0" y="36.0" text-anchor="start" class="fig-small">loss above the floor</text>
</svg>
Figure 1. The same five points twice. On straight axes (left) the curve bends and the three smallest models crowd the left edge. On log axes (right) only the axes changed, and the points lie on a straight line: every tenfold step in size multiplies the loss by the same 0.46.
```

Figure 1 plots the lab's five points twice. The left panel is the ordinary plot. The curve bends, and the three smaller models crowd together at the left edge because the sizes span a factor of a hundred. The right panel plots the same points with both axes on a log scale, which means equal distances along an axis are equal multiplications: from 10 million to 100 million is the same distance as from 100 million to 1 billion. On that scale the five points fall on a straight line.

Nothing was fitted to produce that line. Only the way distance is measured changed. So the curve in the left panel has a special property: equal multiplications of size produce equal multiplications of loss. That property, in plain words, is what a power law is.

## Why a power law: every doubling costs the same share

Say it with numbers. Start at 10 million parameters, where the loss above the floor is 1.67. Every time you multiply the size by 10, the loss is multiplied by about 0.457.

| Parameters | Loss above the floor | Factor from the row above |
|---|---|---|
| 10 million | 1.667 | |
| 100 million | 0.762 | 0.457 |
| 1 billion | 0.348 | 0.457 |
| 10 billion | 0.159 | 0.457 |

Doubling works the same way, with a smaller factor, 0.79. In Figure 2, each doubling takes off 21 percent of what is left. The size of the drop in loss shrinks from 0.35 to 0.14 because 21 percent of a smaller number is a smaller number. That is why returns diminish without ever stopping, and it is not a sign that training is broken.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Loss above the floor at five doublings of model size, each step multiplying the loss by 0.79">
<defs><marker id="c113-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted"/><line x1="60.0" y1="50.0" x2="60.0" y2="200.0" class="stroke-muted"/><line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-grid"/><text x="54.0" y="204.0" text-anchor="end" class="fig-small">0</text><line x1="60.0" y1="160.5" x2="560.0" y2="160.5" class="stroke-grid"/><text x="54.0" y="164.5" text-anchor="end" class="fig-small">0.5</text><line x1="60.0" y1="121.1" x2="560.0" y2="121.1" class="stroke-grid"/><text x="54.0" y="125.1" text-anchor="end" class="fig-small">1</text><line x1="60.0" y1="81.6" x2="560.0" y2="81.6" class="stroke-grid"/><text x="54.0" y="85.6" text-anchor="end" class="fig-small">1.5</text>
<text x="60.0" y="216.0" text-anchor="middle" class="fig-small">10M</text>
<text x="160.0" y="216.0" text-anchor="middle" class="fig-small">20M</text>
<text x="260.0" y="216.0" text-anchor="middle" class="fig-small">40M</text>
<text x="360.0" y="216.0" text-anchor="middle" class="fig-small">80M</text>
<text x="460.0" y="216.0" text-anchor="middle" class="fig-small">160M</text>
<text x="560.0" y="216.0" text-anchor="middle" class="fig-small">320M</text>
<polyline points="60.0,68.4 65.1,70.0 70.1,71.5 75.2,73.0 80.2,74.5 85.3,76.0 90.3,77.5 95.4,78.9 100.4,80.3 105.5,81.8 110.5,83.2 115.6,84.5 120.6,85.9 125.7,87.3 130.7,88.6 135.8,89.9 140.8,91.2 145.9,92.5 150.9,93.8 156.0,95.0 161.0,96.3 166.1,97.5 171.1,98.7 176.2,99.9 181.2,101.1 186.3,102.3 191.3,103.4 196.4,104.6 201.4,105.7 206.5,106.8 211.5,107.9 216.6,109.0 221.6,110.1 226.7,111.1 231.7,112.2 236.8,113.2 241.8,114.3 246.9,115.3 251.9,116.3 257.0,117.3 262.0,118.2 267.1,119.2 272.1,120.2 277.2,121.1 282.2,122.0 287.3,123.0 292.3,123.9 297.4,124.8 302.4,125.7 307.5,126.6 312.5,127.4 317.6,128.3 322.6,129.1 327.7,130.0 332.7,130.8 337.8,131.6 342.8,132.4 347.9,133.2 352.9,134.0 358.0,134.8 363.0,135.6 368.1,136.3 373.1,137.1 378.2,137.8 383.2,138.6 388.3,139.3 393.3,140.0 398.4,140.7 403.4,141.4 408.5,142.1 413.5,142.8 418.6,143.5 423.6,144.1 428.7,144.8 433.7,145.5 438.8,146.1 443.8,146.7 448.9,147.4 453.9,148.0 459.0,148.6 464.0,149.2 469.1,149.8 474.1,150.4 479.2,151.0 484.2,151.6 489.3,152.1 494.3,152.7 499.4,153.3 504.4,153.8 509.5,154.4 514.5,154.9 519.6,155.4 524.6,156.0 529.7,156.5 534.7,157.0 539.8,157.5 544.8,158.0 549.9,158.5 554.9,159.0 560.0,159.5" fill="none" class="stroke-muted dashed" />
<circle cx="60.0" cy="68.4" r="5.5" class="fill-a"/><circle cx="160.0" cy="96.0" r="5.5" class="fill-a"/><circle cx="260.0" cy="117.9" r="5.5" class="fill-a"/><circle cx="360.0" cy="135.1" r="5.5" class="fill-a"/><circle cx="460.0" cy="148.7" r="5.5" class="fill-a"/><circle cx="560.0" cy="159.5" r="5.5" class="fill-a"/>
<text x="110.0" y="64.2" text-anchor="middle" class="fig-bold">×0.79</text>
<text x="110.0" y="238.0" text-anchor="middle" class="fig-small">drops by 0.35</text>
<text x="210.0" y="88.9" text-anchor="middle" class="fig-bold">×0.79</text>
<text x="210.0" y="238.0" text-anchor="middle" class="fig-small">drops by 0.28</text>
<text x="310.0" y="108.5" text-anchor="middle" class="fig-bold">×0.79</text>
<text x="310.0" y="238.0" text-anchor="middle" class="fig-small">drops by 0.22</text>
<text x="410.0" y="123.9" text-anchor="middle" class="fig-bold">×0.79</text>
<text x="410.0" y="238.0" text-anchor="middle" class="fig-small">drops by 0.17</text>
<text x="510.0" y="136.1" text-anchor="middle" class="fig-bold">×0.79</text>
<text x="510.0" y="238.0" text-anchor="middle" class="fig-small">drops by 0.14</text>
<text x="12.0" y="28.0" text-anchor="start" class="fig-small">loss above the floor</text>
<text x="310.0" y="256.0" text-anchor="middle" class="fig-small">model size, doubling at every step</text>
</svg>
Figure 2. Five doublings from 10 million parameters. The multiplier (bold) is the same at every step, 0.79, so the drop in loss (bottom row) shrinks from 0.35 to 0.14. Keeping the same percentage gain costs a doubling each time.
```

Why this shape and not another? Two simpler curves fail on these measurements. A straight line in size takes the same amount off at every step, so it reaches zero and goes negative. An exponential decay in size, one that loses a fixed share per fixed number of added parameters, would flatten completely within a few steps, and the measurements keep falling steadily across a hundredfold range. A power law takes a fixed share off per fixed multiplication, which is the pattern the lab sees. The rule is written $y = c \, x^{k}$, with $x$ the size, $y$ the loss above the floor, $k$ an exponent that is negative when $y$ falls, and $c$ a coefficient. Doubling $x$ multiplies $y$ by $2^{k}$. For $k = -0.34$ that is $2^{-0.34} \approx 0.79$.

> [!KEY]
> A power law means every multiplication of the input multiplies the output by a fixed factor. Doubling gives $2^{k}$, and a tenfold step gives $10^{k}$.

## Taking logs makes the line

Take logs of both sides of $y = c\,x^{k}$. The log of a product is a sum and the log of a power is a product, so

$$
\ln y = \ln c + k \ln x
$$

Read it aloud: the log of $y$ equals a constant plus $k$ times the log of $x$. That is the equation of a straight line, with $u = \ln x$ on the horizontal axis and $v = \ln y$ on the vertical axis. The slope of the line is the exponent $k$, and the intercept is $\ln c$. Every point on Figure 1's right panel obeys this equation, which is why they line up.

Next, a tiny example with every step visible. Take three models of 10, 20, and 40 million parameters. Measure the size in units of 10 million, so $x = 1, 2, 4$, and let their losses above the floor be $y = 1.667, 1.317, 1.040$. Each step doubles the size, so each step also multiplies the loss by the same 0.79.

| $x$ | $y$ | $u = \ln x$ | $v = \ln y$ | $u - \bar u$ | $v - \bar v$ |
|---|---|---|---|---|---|
| 1 | 1.667 | 0.000 | 0.511 | −0.693 | 0.236 |
| 2 | 1.317 | 0.693 | 0.275 | 0.000 | 0.000 |
| 4 | 1.040 | 1.386 | 0.039 | 0.693 | −0.236 |

The means are $\bar u = 0.693$ and $\bar v = 0.275$. The least-squares slope is the sum of products of the centred columns divided by the sum of squares of the centred $u$:

$$
k = \frac{\sum_i (u_i - \bar u)(v_i - \bar v)}{\sum_i (u_i - \bar u)^2},
\qquad c = \exp(\bar v - k \bar u)
$$

Here the numerator is $(-0.693)(0.236) + 0 + (0.693)(-0.236) = -0.327$, and the denominator is $0.480 + 0 + 0.480 = 0.961$. So $k = -0.327 / 0.961 = -0.340$, and $c = \exp(0.275 + 0.340 \times 0.693) = \exp(0.511) = 1.667$. The slope matches the exponent that generated the data, and $c$ equals the loss at $x = 1$.

This is the least-squares line from the earlier chapter, run on logs. Why not fit the curve $c\,x^{k}$ to the raw numbers directly? You can, but that fit has no short formula and needs a search. It also lets the large losses at small sizes dominate the error, while the log fit treats an error of 10 percent the same at every size, which is how a scaling measurement is wrong. The same lines in numpy:

```python
u, v = np.log(x), np.log(y)
uc, vc = u - u.mean(), v - v.mean()
k = np.sum(uc * vc) / np.sum(uc**2)
c = np.exp(v.mean() - k * u.mean())
```

The coefficient $c$ depends on the unit you measure $x$ in, because it is the value at $x = 1$. Counting parameters one by one instead of in units of 10 million turns $c$ into 400, and $k$ stays $-0.34$. The exponent is the part that does not depend on units, and it is the number that says how fast the curve falls.

> [!WARNING]
> The intercept of the log-log line is $\ln c$, not $c$. Return the exponential of the intercept, or every prediction will be wrong by a huge factor.

## Extrapolating, and why it gets less certain

Real measurements are noisy, so the fitted slope is never exactly the true one. Figure 3 uses eight noisy measurements of the same curve. Fitting them gives $k = -0.348$, and extending the fitted line to 10 billion parameters predicts a loss of 0.151 above the floor. The law that generated the data gives 0.159.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Noisy measurements from 10 million to 1 billion parameters, a fitted line extended to 10 billion, and two dashed lines with slopes 5 percent apart that spread out">
<defs><marker id="c113-arrowb" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="60.0" y1="210.0" x2="480.0" y2="210.0" class="stroke-muted"/><line x1="60.0" y1="40.0" x2="60.0" y2="210.0" class="stroke-muted"/><line x1="60.0" y1="40.0" x2="60.0" y2="210.0" class="stroke-grid"/><text x="60.0" y="226.0" text-anchor="middle" class="fig-small">10M</text><line x1="200.0" y1="40.0" x2="200.0" y2="210.0" class="stroke-grid"/><text x="200.0" y="226.0" text-anchor="middle" class="fig-small">100M</text><line x1="340.0" y1="40.0" x2="340.0" y2="210.0" class="stroke-grid"/><text x="340.0" y="226.0" text-anchor="middle" class="fig-small">1B</text><line x1="480.0" y1="40.0" x2="480.0" y2="210.0" class="stroke-grid"/><text x="480.0" y="226.0" text-anchor="middle" class="fig-small">10B</text><line x1="60.0" y1="210.0" x2="480.0" y2="210.0" class="stroke-grid"/><text x="54.0" y="214.0" text-anchor="end" class="fig-small">0.1</text><line x1="60.0" y1="170.7" x2="480.0" y2="170.7" class="stroke-grid"/><text x="54.0" y="174.7" text-anchor="end" class="fig-small">0.2</text><line x1="60.0" y1="118.7" x2="480.0" y2="118.7" class="stroke-grid"/><text x="54.0" y="122.7" text-anchor="end" class="fig-small">0.5</text><line x1="60.0" y1="79.3" x2="480.0" y2="79.3" class="stroke-grid"/><text x="54.0" y="83.3" text-anchor="end" class="fig-small">1</text><line x1="60.0" y1="40.0" x2="480.0" y2="40.0" class="stroke-grid"/><text x="54.0" y="44.0" text-anchor="end" class="fig-small">2</text>
<rect x="340.0" y="40.0" width="140.0" height="170.0" class="fill-soft-a" opacity="0.35"/>
<text x="410.0" y="54.0" text-anchor="middle" class="fig-small">not measured</text>
<polyline points="60.0,50.0 340.0,141.0" fill="none" class="stroke-b" stroke-width="2"/>
<polyline points="340.0,141.0 480.0,186.5" fill="none" class="stroke-b dashed" stroke-width="2"/>
<polyline points="60.0,52.3 480.0,182.0" fill="none" class="stroke-d dashed" />
<polyline points="60.0,47.8 480.0,191.1" fill="none" class="stroke-c dashed" />
<circle cx="60.0" cy="46.9" r="5" class="fill-a"/><circle cx="100.0" cy="67.4" r="5" class="fill-a"/><circle cx="140.0" cy="75.0" r="5" class="fill-a"/><circle cx="180.0" cy="89.4" r="5" class="fill-a"/><circle cx="220.0" cy="101.9" r="5" class="fill-a"/><circle cx="260.0" cy="114.2" r="5" class="fill-a"/><circle cx="300.0" cy="129.9" r="5" class="fill-a"/><circle cx="340.0" cy="139.6" r="5" class="fill-a"/>
<text x="488.0" y="174.5" text-anchor="start" class="fig-small fig-bold fill-d">0.164  shallower</text>
<text x="488.0" y="189.5" text-anchor="start" class="fig-small fig-bold fill-b">0.151  fitted</text>
<text x="488.0" y="204.5" text-anchor="start" class="fig-small fig-bold fill-c">0.140  steeper</text>
<text x="12.0" y="26.0" text-anchor="start" class="fig-small">loss above the floor</text>
<text x="270.0" y="248.0" text-anchor="middle" class="fig-small">model size (parameters), log scale</text>
</svg>
Figure 3. Eight noisy measurements (blue), the fitted line (orange), and its extension to 10 billion parameters. The two dashed lines pivot at the middle of the data with slopes 5% shallower and 5% steeper. They sit close together inside the data and spread apart in the unmeasured region.
```

The two dashed lines in Figure 3 pivot on the middle of the data, which is where a fitted line is most certain, with slopes five percent shallower and five percent steeper. Inside the measured range they are almost on top of each other. At 10 billion parameters they predict 0.164 and 0.140, which differ by about 16 percent of the fitted value. A slope error becomes a vertical error multiplied by the distance you travel on the log axis, so every extra decade of extrapolation widens the gap. A fitted law is a good guide one decade beyond the data and a weak one three decades beyond.

Two habits make the fit more trustworthy. Measure across as wide a range as you can afford, because a wider range pins the slope down. Fit the loss above the floor, not the raw loss: the raw loss has a floor under it, so it is not a power law, and the log-log plot of it curves instead of staying straight.

> [!KEY]
> On log-log axes a power law is a straight line whose slope is the exponent. A least-squares fit finds the slope, and the lab's question about 10 billion parameters becomes "continue the line", with an error that grows the further you go.

> [!RECAP]
> - A power law $y = c\,x^{k}$ multiplies $y$ by the same factor, $2^{k}$ for a doubling, whenever $x$ is multiplied by the same amount.
> - Taking logs gives $\ln y = \ln c + k \ln x$, a straight line with slope $k$ and intercept $\ln c$.
> - Fit with centred logs: $k$ is the slope, and $c$ is the exponential of the intercept.
> - The coefficient $c$ is the value at $x = 1$, so it changes with units, and the exponent does not.
> - A slope error grows in proportion to how far you extrapolate.

## Build it step by step

You write three functions, and the checks call them in this order.

1. `power_law_eval(x, k, c)` takes sizes `x` (a float, a list, or an array) and returns `c * x**k` as a float array. Turn `x` into an array first with `np.asarray`, because a Python list raised to a power fails. The most common mistake is to put the exponent on `c`.
2. `fit_power_law(x, y)` takes positive arrays of shape `(n,)` and returns `(k, c)` as plain floats. Take `np.log` of both, subtract each mean, divide the sum of products by the sum of squares, and exponentiate the intercept. The mistake people make most often is to return the intercept itself.
3. `extrapolate(x_train, y_train, x_new)` fits the law to the measured points and evaluates it at the unmeasured sizes. It needs only the two functions above, so it is a few lines.

The shape of the second function:

```python
def fit_power_law(x, y):
    # logs, centre both, slope, then exp of the intercept
    return k, c
```

> [!TIP]
> Test a fit on data you built yourself, such as `y = 3 * x**1.7`. If the exponent and coefficient do not come back as 1.7 and 3, the bug is in your code, not in the data.

## Where this breaks

A power law is a description of a stretch of measurements, so a bend in the log-log line is information. If the line curves toward flat at the large end, you may have reached the floor, or the model may have stopped improving for a reason the law does not capture. Fit only the straight part, and be wary of continuing it past a bend.

## The exercise

You implement `power_law_eval`, `fit_power_law`, and `extrapolate` in `starter.py`. The checks confirm that `power_law_eval` returns $c\,x^{k}$ and that doubling $x$ multiplies it by $2^{k}$, that `fit_power_law` recovers a planted exponent and coefficient, reproduces the three-point example above, and stays close under noise, and that `extrapolate` matches the true law on clean data and lands within 10 percent on noisy data, with a larger error far from the data than inside it.
