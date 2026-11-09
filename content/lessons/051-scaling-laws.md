---
title: "Scaling laws and the compute budget"
slug: 051-scaling-laws
part: "Part 7 — Large language models"
summary: Fit a power law on log axes, split a model's loss into a floor and two penalties, then spend a fixed compute budget on size and data.
minutes: 75
gpu: false
objectives:
  - Turn a power law into a straight line with logs, fit its exponent and coefficient by least squares, and explain why extrapolation error grows with distance.
  - Split a loss into a floor, a size penalty, and a data penalty, and compute what growing the model or the data buys.
  - Count the weights that multiply, and derive the six in C = 6ND from one weight, one input, and one gradient.
  - Find the best split of a fixed budget between model size and tokens, and say why heavy serving favours a smaller model.
prereqs:
  - 003-least-squares-and-projection
  - 002-curves-and-bias-variance
  - 050-tokenisation
  - 048-next-token-loss-and-safe-steps
  - 027-multilayer-perceptrons
  - 047-the-transformer-block
  - 049-training-loop-and-cost
exercise: 051-scaling-laws
---

# Scaling laws and the compute budget

A lab trains five language models, from 10 million parameters up to 1 billion, and records how far each one's loss sits above its floor, the lowest loss any model could reach. The numbers are 1.67, 1.15, 0.76, 0.52, and 0.35. Each model costs more than the last, so the curve through them bends and flattens. The lab wants to know what a model with 10 billion parameters would score, because that run costs about ten times more than the 1-billion model, and the answer decides whether to pay.

A hand-drawn continuation is a guess, a high-degree polynomial swings away just beyond its data, as in [fitting curves](/lessons/002-curves-and-bias-variance), and a straight line through the raw numbers crosses zero and predicts a negative loss. The lab needs a curve with few knobs that keeps behaving past the data. A second question follows: with a fixed pile of arithmetic, should you build a bigger model or train a smaller one on more text?

This chapter answers both in one arc. You fit a power law with the least-squares line you already know, extend it into a loss that depends on model size and training text, price a training run in flops, and find the split of a budget where the loss is lowest.

## What you already know

- A least-squares line has a slope and an intercept, and both come from one short formula over centred data, as in [least squares and projection](/lessons/003-least-squares-and-projection).
- The logarithm turns a product into a sum and a power into a product, so $\ln(x^k) = k \ln x$.
- A language model's loss is the average of $-\ln p$ over the true next tokens, as in [next-token loss](/lessons/048-next-token-loss-and-safe-steps), and the training text is counted in tokens.
- A flop is one multiply or one add, and a gradient says how much a result moves when you nudge one input.
- A transformer block is attention plus a feed-forward network, as in [the transformer block](/lessons/047-the-transformer-block).

## A power law is a straight line on log paper

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

Figure 1 plots the lab's five points twice. On straight axes the curve bends, and three models crowd the left edge because the sizes span a factor of a hundred. On log axes, where equal distances are equal multiplications, the same points fall on a straight line. Nothing was fitted. The line says that every multiplication of size multiplies the loss by a fixed factor: here, every tenfold step gives 0.457, and every doubling gives $2^{-0.34} \approx 0.79$. That property is what a power law is:

$$
y = c\,x^{k}
$$

Here $x$ is the size, $y$ is the loss above the floor, $k$ is an exponent that is negative when $y$ falls, and $c$ is a coefficient. Each doubling takes off 21 percent of what is left, so the drop in loss shrinks from 0.35 to 0.28 to 0.22 while the percentage stays fixed. Diminishing returns are built in, and they are not a sign that training is broken.

Take logs of both sides:

$$
\ln y = \ln c + k \ln x
$$

Read it aloud: the log of $y$ is a constant plus $k$ times the log of $x$. That is a straight line with $u = \ln x$ across, $v = \ln y$ up, slope $k$, and intercept $\ln c$. Now fit three models of 10, 20, and 40 million parameters. Measure size in units of 10 million, so $x = 1, 2, 4$, with losses $y = 1.667, 1.317, 1.040$.

| $x$ | $y$ | $u = \ln x$ | $v = \ln y$ | $u - \bar u$ | $v - \bar v$ |
|---|---|---|---|---|---|
| 1 | 1.667 | 0.000 | 0.511 | −0.693 | 0.236 |
| 2 | 1.317 | 0.693 | 0.275 | 0.000 | 0.000 |
| 4 | 1.040 | 1.386 | 0.039 | 0.693 | −0.236 |

The least-squares slope is the sum of products of the centred columns over the sum of squares of centred $u$:

$$
k = \frac{\sum_i (u_i - \bar u)(v_i - \bar v)}{\sum_i (u_i - \bar u)^2},
\qquad c = \exp(\bar v - k \bar u)
$$

The numerator is $(-0.693)(0.236) + 0 + (0.693)(-0.236) = -0.327$ and the denominator is $0.961$, so $k = -0.340$. Then $c = \exp(0.275 + 0.340 \times 0.693) = 1.667$, the loss at $x = 1$. In numpy:

```python
u, v = np.log(x), np.log(y)
uc = u - u.mean()
k = np.sum(uc * (v - v.mean())) / np.sum(uc**2)
c = np.exp(v.mean() - k * u.mean())
```

Why not fit $c\,x^{k}$ to the raw numbers? That fit has no short formula and lets the large losses at small sizes dominate, while the log fit treats a 10 percent error the same at every size. The coefficient $c$ depends on the unit of $x$, because it is the value at $x = 1$. The exponent does not, and it says how fast the curve falls.

> [!WARNING]
> The intercept of the log-log line is $\ln c$, not $c$. Return the exponential of the intercept, or every prediction is wrong by a huge factor.

Real measurements are noisy, so the fitted slope is never exact. Fit eight noisy measurements of the same curve and you get $k = -0.348$, which predicts 0.151 at 10 billion parameters where the true law gives 0.159. Now tilt the slope five percent shallower and five percent steeper, pivoting at the middle of the data. The two lines nearly overlap inside the data, and at 10 billion they predict 0.164 and 0.140, about 16 percent apart. A slope error becomes a vertical error multiplied by the distance you travel on the log axis, so a fitted law guides one decade past the data and misleads three decades past it.

> [!KEY]
> On log-log axes a power law is a straight line whose slope is the exponent. A least-squares fit finds it, the question about 10 billion parameters becomes "continue the line", and the error grows the further you continue.

## Loss has a floor and two penalties

Now a different situation. A model with 1 billion parameters, trained on 2 billion tokens, has loss 2.795. Doubling the model moves it to 2.722. Doubling the training text instead gives 2.636, more than twice the improvement, for about the same extra compute. The loss is not one number that falls as you spend more. It has several causes, and spending on the wrong one barely helps.

Think of a student with a notebook and a pile of pages. A tiny notebook cannot hold what the student learns, and a huge notebook after two pages is mostly blank. A bigger notebook fixes only the first problem, and more pages fix only the second. A model has the same two limits, plus the part of language nobody can predict, which is the floor. The loss adds the three:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Read it aloud: the loss is the floor $E$, plus a size penalty that is a constant $A$ over the parameters $N$ raised to a small positive power $\alpha$, plus a data penalty that is a constant $B$ over the tokens $D$ raised to $\beta$. Each penalty is a power law that falls toward zero as its own input grows. They add because they are separate causes, each depending on one variable. The additive form is a fitted description of experiments, not a derivation.

Use the stylised constants $E = 1.69$, $A = 400$, $B = 1100$, and $\alpha = \beta = 0.34$. They are round numbers for checking arithmetic, and the published data exponent is smaller, about 0.28. At $N = 10^9$ and $D = 2 \times 10^{9}$, the size penalty is $400 / (10^9)^{0.34} = 0.3484$ and the data penalty is $1100 / (2 \times 10^{9})^{0.34} = 0.7569$. The loss is $1.69 + 0.3484 + 0.7569 = 2.7953$, and the data piece is more than twice the size piece. In numpy:

```python
E, A, B, alpha, beta = coeffs
size_penalty = A / N**alpha
data_penalty = B / D**beta
```

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Three stacked bars of loss split into floor, size penalty, and data penalty: the starting model, the model doubled, and the data doubled">
<line x1="70.0" y1="230.0" x2="470.0" y2="230.0" class="stroke-muted"/><line x1="70.0" y1="30.0" x2="70.0" y2="230.0" class="stroke-muted"/><line x1="70.0" y1="230.0" x2="470.0" y2="230.0" class="stroke-grid"/><text x="64.0" y="234.0" text-anchor="end" class="fig-small">0</text><line x1="70.0" y1="167.5" x2="470.0" y2="167.5" class="stroke-grid"/><text x="64.0" y="171.5" text-anchor="end" class="fig-small">1</text><line x1="70.0" y1="105.0" x2="470.0" y2="105.0" class="stroke-grid"/><text x="64.0" y="109.0" text-anchor="end" class="fig-small">2</text><line x1="70.0" y1="42.5" x2="470.0" y2="42.5" class="stroke-grid"/><text x="64.0" y="46.5" text-anchor="end" class="fig-small">3</text>
<rect x="90.0" y="124.4" width="80" height="105.6" class="fill-sunk stroke-muted" opacity="1"/>
<text x="130.0" y="181.2" text-anchor="middle" class="fig-small fig-bold">1.690</text>
<rect x="90.0" y="102.6" width="80" height="21.8" class="fill-a" opacity="1"/>
<text x="130.0" y="117.5" text-anchor="middle" class="fig-small fig-bold">0.348</text>
<rect x="90.0" y="55.3" width="80" height="47.3" class="fill-b" opacity="1"/>
<text x="130.0" y="82.9" text-anchor="middle" class="fig-small fig-bold">0.757</text>
<text x="130.0" y="47.3" text-anchor="middle" class="fig-small fig-bold">total 2.795</text>
<text x="130.0" y="246.0" text-anchor="middle" class="fig-small fig-bold">start</text>
<text x="130.0" y="261.0" text-anchor="middle" class="fig-small">1B params,</text>
<text x="130.0" y="275.0" text-anchor="middle" class="fig-small">2B tokens</text>
<rect x="230.0" y="124.4" width="80" height="105.6" class="fill-sunk stroke-muted" opacity="1"/>
<text x="270.0" y="181.2" text-anchor="middle" class="fig-small fig-bold">1.690</text>
<rect x="230.0" y="107.2" width="80" height="17.2" class="fill-a" opacity="1"/>
<text x="270.0" y="119.8" text-anchor="middle" class="fig-small fig-bold">0.275</text>
<rect x="230.0" y="59.9" width="80" height="47.3" class="fill-b" opacity="1"/>
<text x="270.0" y="87.5" text-anchor="middle" class="fig-small fig-bold">0.757</text>
<text x="270.0" y="51.9" text-anchor="middle" class="fig-small fig-bold">total 2.722</text>
<text x="270.0" y="246.0" text-anchor="middle" class="fig-small fig-bold">2× the model</text>
<text x="270.0" y="261.0" text-anchor="middle" class="fig-small">2B params,</text>
<text x="270.0" y="275.0" text-anchor="middle" class="fig-small">2B tokens</text>
<rect x="370.0" y="124.4" width="80" height="105.6" class="fill-sunk stroke-muted" opacity="1"/>
<text x="410.0" y="181.2" text-anchor="middle" class="fig-small fig-bold">1.690</text>
<rect x="370.0" y="102.6" width="80" height="21.8" class="fill-a" opacity="1"/>
<text x="410.0" y="117.5" text-anchor="middle" class="fig-small fig-bold">0.348</text>
<rect x="370.0" y="65.2" width="80" height="37.4" class="fill-b" opacity="1"/>
<text x="410.0" y="87.9" text-anchor="middle" class="fig-small fig-bold">0.598</text>
<text x="410.0" y="57.2" text-anchor="middle" class="fig-small fig-bold">total 2.636</text>
<text x="410.0" y="246.0" text-anchor="middle" class="fig-small fig-bold">2× the data</text>
<text x="410.0" y="261.0" text-anchor="middle" class="fig-small">1B params,</text>
<text x="410.0" y="275.0" text-anchor="middle" class="fig-small">4B tokens</text>
<rect x="500" y="50" width="14" height="14" class="fill-b"/><text x="520.0" y="62.0" text-anchor="start" class="fig-small">data penalty</text>
<rect x="500" y="74" width="14" height="14" class="fill-a"/><text x="520.0" y="86.0" text-anchor="start" class="fig-small">size penalty</text>
<rect x="500" y="98" width="14" height="14" class="fill-sunk stroke-muted"/><text x="520.0" y="110.0" text-anchor="start" class="fig-small">floor E</text>
</svg>
Figure 2. The same split of loss in three situations. Doubling the model shrinks only the blue size piece, by 21%, from 0.348 to 0.275. Doubling the data shrinks only the orange data piece by the same 21%, but that piece starts larger, so it removes more loss. The floor never moves.
```

Figure 2 shows both doublings. Doubling the model multiplies the size piece by $2^{-0.34} = 0.79$, from 0.3484 to 0.2752, and the loss falls by 0.073. Doubling the data multiplies the data piece by the same 0.79, from 0.7569 to 0.5979, and the loss falls by 0.159. The percentages are equal and the amounts are not, because the data piece starts out larger. The larger penalty has more to give back, so grow that one.

One result is easy to miss: the gain from doubling the model, $A/N^{\alpha}\,(1 - 2^{-\alpha})$, contains no $D$. A bigger model gains 0.073 whether you have 2 billion tokens or 2 trillion. The data changes only how small that gain looks beside the loss that remains.

Nobody derives $A$ or $\alpha$. You measure them with the tool from the last section: train many sizes on so much text that the data penalty is nearly constant, subtract the floor and that constant, and a log-log fit of what is left has slope $-\alpha$ and coefficient $A$.

> [!KEY]
> Loss is a floor plus a size penalty plus a data penalty. Each penalty shrinks only as its own input grows, so growing one input alone eventually stops helping, and the larger penalty is the one worth shrinking.

## Pricing a run: the weights that multiply

To spend a budget you first need the price of a model. A transformer holds attention matrices, feed-forward matrices, and lookup tables, and counting them as equal overstates a small model's cost by half.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A toy transformer with two blocks: four attention matrices and two feed-forward matrices per block count toward N, while the token and position tables are lookups that do not">
<text x="20.0" y="22.0" text-anchor="start" class="fig-bold">One block (counted in N)</text>
<rect x="14.0" y="30.0" width="360.0" height="208.0" rx="6" class="fill-sunk stroke-muted dashed"/>
<text x="24.0" y="50.0" text-anchor="start" class="fig-small fig-bold">attention: four 8×8 matrices</text>
<rect x="24.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="38.0" y="76.0" text-anchor="middle" class="fig-bold">Q</text>
<rect x="66.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="80.0" y="76.0" text-anchor="middle" class="fig-bold">K</text>
<rect x="108.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="122.0" y="76.0" text-anchor="middle" class="fig-bold">V</text>
<rect x="150.0" y="58.0" width="28.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="164.0" y="76.0" text-anchor="middle" class="fig-bold">O</text>
<text x="182.0" y="76.0" text-anchor="start" class="fig-small fig-bold">4 × 64 = 256</text>
<text x="24.0" y="126.0" text-anchor="start" class="fig-small fig-bold">feed-forward: an 8×32 matrix, then a 32×8 matrix</text>
<rect x="24.0" y="134.0" width="112.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="80.0" y="152.0" text-anchor="middle" class="fig-small">up: 8 × 32 = 256</text>
<rect x="24.0" y="172.0" width="112.0" height="28.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="80.0" y="190.0" text-anchor="middle" class="fig-small">down: 32 × 8 = 256</text>
<text x="148.0" y="166.0" text-anchor="start" class="fig-small fig-bold">2 × 256 = 512</text>
<text x="24.0" y="226.0" text-anchor="start" class="fig-small fig-bold">per block 256 + 512 = 768, and 2 blocks make 1,536</text>
<text x="410.0" y="22.0" text-anchor="start" class="fig-bold">Lookup tables (not in N)</text>
<rect x="404.0" y="30.0" width="180.0" height="208.0" rx="6" class="fill-sunk stroke-muted dashed"/>
<rect x="414.0" y="48.0" width="160.0" height="52.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="494.0" y="70.0" text-anchor="middle" class="fig-bold">token table</text>
<text x="494.0" y="88.0" text-anchor="middle" class="fig-small">100 × 8 = 800</text>
<rect x="414.0" y="112.0" width="160.0" height="52.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="494.0" y="134.0" text-anchor="middle" class="fig-bold">position table</text>
<text x="494.0" y="152.0" text-anchor="middle" class="fig-small">16 × 8 = 128</text>
<text x="494.0" y="186.0" text-anchor="middle" class="fig-small">a token id picks one row:</text>
<text x="494.0" y="202.0" text-anchor="middle" class="fig-small">no multiply-adds</text>
<text x="494.0" y="226.0" text-anchor="middle" class="fig-small fig-bold">800 + 128 = 928</text>
</svg>
Figure 3. The toy model: 2 blocks, width 8, feed-forward width 32, vocabulary 100, and 16 positions. The orange boxes are weights that multiply, and they make N = 1,536. The green tables only return a row, so their 928 numbers stay out of N. Box areas match the sizes of the matrices.
```

In Figure 3, each block has four $d \times d$ attention matrices (query, key, value, output) and two feed-forward matrices, one widening from $d$ to $d_{\text{ff}}$ and one narrowing back. Every token passes through every one of them and gets multiplied, so each weight does arithmetic for each token. A lookup table only returns a row, so nothing is multiplied and its numbers stay out of the compute count $N$. For the toy model, attention holds $4 \times 8^2 = 256$ weights per block and the feed-forward network holds $2 \times 8 \times 32 = 512$, so a block holds 768 and two blocks hold 1,536. The tables hold $100 \times 8 + 16 \times 8 = 928$. The output layer reuses the token table, a trick called weight tying, so it adds nothing. In symbols, with $L$ blocks:

$$
N = L\,\bigl(4 d^{2} + 2\, d\, d_{\text{ff}}\bigr),
\qquad \text{embeddings} = (\text{vocab} + \text{positions})\, d
$$

When $d_{\text{ff}} = 4d$ a block holds $12 d^2$ weights, so $N = 12 L d^2$. The count skips biases and layer-norm scales, which are smaller by a factor of $d$. The tables matter at small sizes: they hold 38 percent of the toy model's parameters and 32 percent of a 12-block, width-768 model, but 0.37 percent of a 96-block, width-12288 model.

Next, the cost of one weight. Take $w = 0.5$ and an input $x = 3$. Forward, it multiplies, $0.5 \times 3 = 1.5$, and the product is added into a running sum, which is 2 flops. Backward, suppose the gradient $g = 2$ arrives at the sum. The weight needs the gradient for itself, $g \times x = 6$, and must hand back the gradient for its input, $g \times w = 1$. Each is a multiply and an add, so the backward pass costs 4 flops, and training costs $2 + 4 = 6$ per weight per token:

$$
C = 6\,N\,D
$$

Read it aloud: compute is six times the weights that multiply, times the tokens. A model with 1,474,560,000 such weights (48 blocks, width 1600) trained on 300 billion tokens costs $6 \times 1.4746 \times 10^{9} \times 3 \times 10^{11} = 2.65 \times 10^{21}$ flops. A chip that sustains $2 \times 10^{14}$ flops per second takes $1.33 \times 10^{7}$ seconds, which is 154 days. A $10^{20}$ run takes 5.8 days.

> [!WARNING]
> A numpy 64-bit integer overflows above about $9.2 \times 10^{18}$, so `6 * params * tokens` can wrap to a wrong number. Write `6.0 * float(params) * float(tokens)`. Using the total parameter count with the tables, instead of the weights that multiply, overstates a 12-block, width-768 model's compute by almost half.

## Spending the budget

Because $D = C/(6N)$, a bigger model at fixed compute means fewer tokens. Fix the budget at $1.2 \times 10^{20}$ flops, which is 1 billion parameters on 20 billion tokens, and slide along it.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Size penalty falling, data penalty rising, and their sum making a U shape along a line of fixed compute, with the lowest point at 1 billion parameters">
<line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-muted"/><line x1="70.0" y1="36.0" x2="70.0" y2="236.0" class="stroke-muted"/><line x1="120.9" y1="36.0" x2="120.9" y2="236.0" class="stroke-grid"/><text x="120.9" y="252.0" text-anchor="middle" class="fig-small">100M</text><line x1="290.0" y1="36.0" x2="290.0" y2="236.0" class="stroke-grid"/><text x="290.0" y="252.0" text-anchor="middle" class="fig-small">1B</text><line x1="459.1" y1="36.0" x2="459.1" y2="236.0" class="stroke-grid"/><text x="459.1" y="252.0" text-anchor="middle" class="fig-small">10B</text><line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-grid"/><text x="64.0" y="240.0" text-anchor="end" class="fig-small">0</text><line x1="70.0" y1="156.0" x2="510.0" y2="156.0" class="stroke-grid"/><text x="64.0" y="160.0" text-anchor="end" class="fig-small">0.5</text><line x1="70.0" y1="76.0" x2="510.0" y2="76.0" class="stroke-grid"/><text x="64.0" y="80.0" text-anchor="end" class="fig-small">1</text>
<polyline points="70.0,81.6 71.5,82.7 72.9,83.7 74.4,84.8 75.9,85.8 77.4,86.8 78.8,87.8 80.3,88.8 81.8,89.8 83.2,90.8 84.7,91.8 86.2,92.8 87.7,93.8 89.1,94.7 90.6,95.7 92.1,96.6 93.5,97.6 95.0,98.5 96.5,99.5 98.0,100.4 99.4,101.3 100.9,102.2 102.4,103.1 103.8,104.0 105.3,104.9 106.8,105.8 108.3,106.7 109.7,107.6 111.2,108.4 112.7,109.3 114.1,110.2 115.6,111.0 117.1,111.9 118.6,112.7 120.0,113.6 121.5,114.4 123.0,115.2 124.4,116.0 125.9,116.9 127.4,117.7 128.9,118.5 130.3,119.3 131.8,120.1 133.3,120.8 134.7,121.6 136.2,122.4 137.7,123.2 139.2,123.9 140.6,124.7 142.1,125.5 143.6,126.2 145.1,126.9 146.5,127.7 148.0,128.4 149.5,129.2 150.9,129.9 152.4,130.6 153.9,131.3 155.4,132.0 156.8,132.7 158.3,133.4 159.8,134.1 161.2,134.8 162.7,135.5 164.2,136.2 165.7,136.9 167.1,137.5 168.6,138.2 170.1,138.9 171.5,139.5 173.0,140.2 174.5,140.8 176.0,141.5 177.4,142.1 178.9,142.8 180.4,143.4 181.8,144.0 183.3,144.7 184.8,145.3 186.3,145.9 187.7,146.5 189.2,147.1 190.7,147.7 192.1,148.3 193.6,148.9 195.1,149.5 196.6,150.1 198.0,150.7 199.5,151.2 201.0,151.8 202.4,152.4 203.9,153.0 205.4,153.5 206.9,154.1 208.3,154.6 209.8,155.2 211.3,155.7 212.7,156.3 214.2,156.8 215.7,157.4 217.2,157.9 218.6,158.4 220.1,159.0 221.6,159.5 223.0,160.0 224.5,160.5 226.0,161.0 227.5,161.5 228.9,162.0 230.4,162.5 231.9,163.0 233.3,163.5 234.8,164.0 236.3,164.5 237.8,165.0 239.2,165.5 240.7,166.0 242.2,166.4 243.6,166.9 245.1,167.4 246.6,167.8 248.1,168.3 249.5,168.8 251.0,169.2 252.5,169.7 253.9,170.1 255.4,170.6 256.9,171.0 258.4,171.5 259.8,171.9 261.3,172.3 262.8,172.8 264.2,173.2 265.7,173.6 267.2,174.0 268.7,174.5 270.1,174.9 271.6,175.3 273.1,175.7 274.5,176.1 276.0,176.5 277.5,176.9 279.0,177.3 280.4,177.7 281.9,178.1 283.4,178.5 284.8,178.9 286.3,179.3 287.8,179.7 289.3,180.1 290.7,180.4 292.2,180.8 293.7,181.2 295.2,181.6 296.6,181.9 298.1,182.3 299.6,182.7 301.0,183.0 302.5,183.4 304.0,183.8 305.5,184.1 306.9,184.5 308.4,184.8 309.9,185.2 311.3,185.5 312.8,185.8 314.3,186.2 315.8,186.5 317.2,186.9 318.7,187.2 320.2,187.5 321.6,187.9 323.1,188.2 324.6,188.5 326.1,188.8 327.5,189.1 329.0,189.5 330.5,189.8 331.9,190.1 333.4,190.4 334.9,190.7 336.4,191.0 337.8,191.3 339.3,191.6 340.8,191.9 342.2,192.2 343.7,192.5 345.2,192.8 346.7,193.1 348.1,193.4 349.6,193.7 351.1,194.0 352.5,194.3 354.0,194.6 355.5,194.8 357.0,195.1 358.4,195.4 359.9,195.7 361.4,195.9 362.8,196.2 364.3,196.5 365.8,196.8 367.3,197.0 368.7,197.3 370.2,197.5 371.7,197.8 373.1,198.1 374.6,198.3 376.1,198.6 377.6,198.8 379.0,199.1 380.5,199.3 382.0,199.6 383.4,199.8 384.9,200.1 386.4,200.3 387.9,200.6 389.3,200.8 390.8,201.0 392.3,201.3 393.7,201.5 395.2,201.8 396.7,202.0 398.2,202.2 399.6,202.4 401.1,202.7 402.6,202.9 404.0,203.1 405.5,203.3 407.0,203.6 408.5,203.8 409.9,204.0 411.4,204.2 412.9,204.4 414.3,204.7 415.8,204.9 417.3,205.1 418.8,205.3 420.2,205.5 421.7,205.7 423.2,205.9 424.6,206.1 426.1,206.3 427.6,206.5 429.1,206.7 430.5,206.9 432.0,207.1 433.5,207.3 434.9,207.5 436.4,207.7 437.9,207.9 439.4,208.1 440.8,208.3 442.3,208.5 443.8,208.6 445.3,208.8 446.7,209.0 448.2,209.2 449.7,209.4 451.1,209.6 452.6,209.7 454.1,209.9 455.6,210.1 457.0,210.3 458.5,210.5 460.0,210.6 461.4,210.8 462.9,211.0 464.4,211.1 465.9,211.3 467.3,211.5 468.8,211.6 470.3,211.8 471.7,212.0 473.2,212.1 474.7,212.3 476.2,212.5 477.6,212.6 479.1,212.8 480.6,212.9 482.0,213.1 483.5,213.2 485.0,213.4 486.5,213.6 487.9,213.7 489.4,213.9 490.9,214.0 492.3,214.2 493.8,214.3 495.3,214.5 496.8,214.6 498.2,214.7 499.7,214.9 501.2,215.0 502.6,215.2 504.1,215.3 505.6,215.5 507.1,215.6 508.5,215.7 510.0,215.9" fill="none" class="stroke-a" stroke-width="2"/>
<polyline points="70.0,216.0 71.5,215.9 72.9,215.7 74.4,215.6 75.9,215.5 77.4,215.3 78.8,215.2 80.3,215.0 81.8,214.9 83.2,214.7 84.7,214.6 86.2,214.5 87.7,214.3 89.1,214.2 90.6,214.0 92.1,213.9 93.5,213.7 95.0,213.6 96.5,213.4 98.0,213.2 99.4,213.1 100.9,212.9 102.4,212.8 103.8,212.6 105.3,212.5 106.8,212.3 108.3,212.1 109.7,212.0 111.2,211.8 112.7,211.6 114.1,211.5 115.6,211.3 117.1,211.1 118.6,211.0 120.0,210.8 121.5,210.6 123.0,210.5 124.4,210.3 125.9,210.1 127.4,209.9 128.9,209.7 130.3,209.6 131.8,209.4 133.3,209.2 134.7,209.0 136.2,208.8 137.7,208.7 139.2,208.5 140.6,208.3 142.1,208.1 143.6,207.9 145.1,207.7 146.5,207.5 148.0,207.3 149.5,207.1 150.9,206.9 152.4,206.7 153.9,206.5 155.4,206.3 156.8,206.1 158.3,205.9 159.8,205.7 161.2,205.5 162.7,205.3 164.2,205.1 165.7,204.9 167.1,204.7 168.6,204.4 170.1,204.2 171.5,204.0 173.0,203.8 174.5,203.6 176.0,203.4 177.4,203.1 178.9,202.9 180.4,202.7 181.8,202.5 183.3,202.2 184.8,202.0 186.3,201.8 187.7,201.5 189.2,201.3 190.7,201.1 192.1,200.8 193.6,200.6 195.1,200.3 196.6,200.1 198.0,199.8 199.5,199.6 201.0,199.3 202.4,199.1 203.9,198.8 205.4,198.6 206.9,198.3 208.3,198.1 209.8,197.8 211.3,197.6 212.7,197.3 214.2,197.0 215.7,196.8 217.2,196.5 218.6,196.2 220.1,195.9 221.6,195.7 223.0,195.4 224.5,195.1 226.0,194.8 227.5,194.6 228.9,194.3 230.4,194.0 231.9,193.7 233.3,193.4 234.8,193.1 236.3,192.8 237.8,192.5 239.2,192.2 240.7,191.9 242.2,191.6 243.6,191.3 245.1,191.0 246.6,190.7 248.1,190.4 249.5,190.1 251.0,189.8 252.5,189.5 253.9,189.2 255.4,188.8 256.9,188.5 258.4,188.2 259.8,187.9 261.3,187.5 262.8,187.2 264.2,186.9 265.7,186.5 267.2,186.2 268.7,185.9 270.1,185.5 271.6,185.2 273.1,184.8 274.5,184.5 276.0,184.1 277.5,183.8 279.0,183.4 280.4,183.0 281.9,182.7 283.4,182.3 284.8,181.9 286.3,181.6 287.8,181.2 289.3,180.8 290.7,180.5 292.2,180.1 293.7,179.7 295.2,179.3 296.6,178.9 298.1,178.5 299.6,178.1 301.0,177.7 302.5,177.3 304.0,176.9 305.5,176.5 306.9,176.1 308.4,175.7 309.9,175.3 311.3,174.9 312.8,174.5 314.3,174.1 315.8,173.6 317.2,173.2 318.7,172.8 320.2,172.3 321.6,171.9 323.1,171.5 324.6,171.0 326.1,170.6 327.5,170.1 329.0,169.7 330.5,169.2 331.9,168.8 333.4,168.3 334.9,167.9 336.4,167.4 337.8,166.9 339.3,166.5 340.8,166.0 342.2,165.5 343.7,165.0 345.2,164.5 346.7,164.0 348.1,163.6 349.6,163.1 351.1,162.6 352.5,162.1 354.0,161.5 355.5,161.0 357.0,160.5 358.4,160.0 359.9,159.5 361.4,159.0 362.8,158.4 364.3,157.9 365.8,157.4 367.3,156.8 368.7,156.3 370.2,155.8 371.7,155.2 373.1,154.7 374.6,154.1 376.1,153.5 377.6,153.0 379.0,152.4 380.5,151.8 382.0,151.3 383.4,150.7 384.9,150.1 386.4,149.5 387.9,148.9 389.3,148.3 390.8,147.7 392.3,147.1 393.7,146.5 395.2,145.9 396.7,145.3 398.2,144.7 399.6,144.0 401.1,143.4 402.6,142.8 404.0,142.1 405.5,141.5 407.0,140.9 408.5,140.2 409.9,139.5 411.4,138.9 412.9,138.2 414.3,137.6 415.8,136.9 417.3,136.2 418.8,135.5 420.2,134.8 421.7,134.1 423.2,133.4 424.6,132.7 426.1,132.0 427.6,131.3 429.1,130.6 430.5,129.9 432.0,129.2 433.5,128.4 434.9,127.7 436.4,127.0 437.9,126.2 439.4,125.5 440.8,124.7 442.3,124.0 443.8,123.2 445.3,122.4 446.7,121.6 448.2,120.9 449.7,120.1 451.1,119.3 452.6,118.5 454.1,117.7 455.6,116.9 457.0,116.1 458.5,115.2 460.0,114.4 461.4,113.6 462.9,112.7 464.4,111.9 465.9,111.0 467.3,110.2 468.8,109.3 470.3,108.5 471.7,107.6 473.2,106.7 474.7,105.8 476.2,104.9 477.6,104.0 479.1,103.1 480.6,102.2 482.0,101.3 483.5,100.4 485.0,99.5 486.5,98.5 487.9,97.6 489.4,96.7 490.9,95.7 492.3,94.7 493.8,93.8 495.3,92.8 496.8,91.8 498.2,90.8 499.7,89.8 501.2,88.9 502.6,87.8 504.1,86.8 505.6,85.8 507.1,84.8 508.5,83.8 510.0,82.7" fill="none" class="stroke-d" stroke-width="2"/>
<polyline points="70.0,61.7 71.5,62.6 72.9,63.5 74.4,64.4 75.9,65.2 77.4,66.1 78.8,67.0 80.3,67.9 81.8,68.7 83.2,69.6 84.7,70.4 86.2,71.2 87.7,72.1 89.1,72.9 90.6,73.7 92.1,74.5 93.5,75.3 95.0,76.1 96.5,76.9 98.0,77.6 99.4,78.4 100.9,79.2 102.4,79.9 103.8,80.6 105.3,81.4 106.8,82.1 108.3,82.8 109.7,83.6 111.2,84.3 112.7,85.0 114.1,85.7 115.6,86.3 117.1,87.0 118.6,87.7 120.0,88.4 121.5,89.0 123.0,89.7 124.4,90.3 125.9,91.0 127.4,91.6 128.9,92.2 130.3,92.8 131.8,93.4 133.3,94.0 134.7,94.6 136.2,95.2 137.7,95.8 139.2,96.4 140.6,97.0 142.1,97.5 143.6,98.1 145.1,98.7 146.5,99.2 148.0,99.7 149.5,100.3 150.9,100.8 152.4,101.3 153.9,101.8 155.4,102.4 156.8,102.9 158.3,103.4 159.8,103.8 161.2,104.3 162.7,104.8 164.2,105.3 165.7,105.7 167.1,106.2 168.6,106.7 170.1,107.1 171.5,107.5 173.0,108.0 174.5,108.4 176.0,108.8 177.4,109.3 178.9,109.7 180.4,110.1 181.8,110.5 183.3,110.9 184.8,111.3 186.3,111.6 187.7,112.0 189.2,112.4 190.7,112.8 192.1,113.1 193.6,113.5 195.1,113.8 196.6,114.2 198.0,114.5 199.5,114.8 201.0,115.2 202.4,115.5 203.9,115.8 205.4,116.1 206.9,116.4 208.3,116.7 209.8,117.0 211.3,117.3 212.7,117.6 214.2,117.9 215.7,118.1 217.2,118.4 218.6,118.7 220.1,118.9 221.6,119.2 223.0,119.4 224.5,119.6 226.0,119.9 227.5,120.1 228.9,120.3 230.4,120.5 231.9,120.8 233.3,121.0 234.8,121.2 236.3,121.4 237.8,121.5 239.2,121.7 240.7,121.9 242.2,122.1 243.6,122.3 245.1,122.4 246.6,122.6 248.1,122.7 249.5,122.9 251.0,123.0 252.5,123.2 253.9,123.3 255.4,123.4 256.9,123.5 258.4,123.7 259.8,123.8 261.3,123.9 262.8,124.0 264.2,124.1 265.7,124.2 267.2,124.2 268.7,124.3 270.1,124.4 271.6,124.5 273.1,124.5 274.5,124.6 276.0,124.6 277.5,124.7 279.0,124.7 280.4,124.8 281.9,124.8 283.4,124.8 284.8,124.9 286.3,124.9 287.8,124.9 289.3,124.9 290.7,124.9 292.2,124.9 293.7,124.9 295.2,124.9 296.6,124.9 298.1,124.8 299.6,124.8 301.0,124.8 302.5,124.7 304.0,124.7 305.5,124.6 306.9,124.6 308.4,124.5 309.9,124.5 311.3,124.4 312.8,124.3 314.3,124.2 315.8,124.2 317.2,124.1 318.7,124.0 320.2,123.9 321.6,123.8 323.1,123.7 324.6,123.5 326.1,123.4 327.5,123.3 329.0,123.2 330.5,123.0 331.9,122.9 333.4,122.7 334.9,122.6 336.4,122.4 337.8,122.3 339.3,122.1 340.8,121.9 342.2,121.7 343.7,121.5 345.2,121.4 346.7,121.2 348.1,121.0 349.6,120.8 351.1,120.5 352.5,120.3 354.0,120.1 355.5,119.9 357.0,119.6 358.4,119.4 359.9,119.2 361.4,118.9 362.8,118.7 364.3,118.4 365.8,118.1 367.3,117.9 368.7,117.6 370.2,117.3 371.7,117.0 373.1,116.7 374.6,116.4 376.1,116.1 377.6,115.8 379.0,115.5 380.5,115.2 382.0,114.8 383.4,114.5 384.9,114.2 386.4,113.8 387.9,113.5 389.3,113.1 390.8,112.8 392.3,112.4 393.7,112.0 395.2,111.7 396.7,111.3 398.2,110.9 399.6,110.5 401.1,110.1 402.6,109.7 404.0,109.3 405.5,108.8 407.0,108.4 408.5,108.0 409.9,107.6 411.4,107.1 412.9,106.7 414.3,106.2 415.8,105.8 417.3,105.3 418.8,104.8 420.2,104.3 421.7,103.9 423.2,103.4 424.6,102.9 426.1,102.4 427.6,101.9 429.1,101.3 430.5,100.8 432.0,100.3 433.5,99.8 434.9,99.2 436.4,98.7 437.9,98.1 439.4,97.6 440.8,97.0 442.3,96.4 443.8,95.8 445.3,95.2 446.7,94.7 448.2,94.1 449.7,93.5 451.1,92.8 452.6,92.2 454.1,91.6 455.6,91.0 457.0,90.3 458.5,89.7 460.0,89.0 461.4,88.4 462.9,87.7 464.4,87.0 465.9,86.4 467.3,85.7 468.8,85.0 470.3,84.3 471.7,83.6 473.2,82.8 474.7,82.1 476.2,81.4 477.6,80.7 479.1,79.9 480.6,79.2 482.0,78.4 483.5,77.6 485.0,76.9 486.5,76.1 487.9,75.3 489.4,74.5 490.9,73.7 492.3,72.9 493.8,72.1 495.3,71.3 496.8,70.4 498.2,69.6 499.7,68.7 501.2,67.9 502.6,67.0 504.1,66.1 505.6,65.3 507.1,64.4 508.5,63.5 510.0,62.6" fill="none" class="stroke-b" stroke-width="3"/>
<circle cx="290.8" cy="124.9" r="6" class="fill-b"/>
<line x1="290.8" y1="36.0" x2="290.8" y2="236.0" class="stroke-muted dashed" />
<text x="290.8" y="30.0" text-anchor="middle" class="fig-small fig-bold">best split: N* = 1.01B, D* = 19.8B</text>
<text x="78.3" y="48.8" text-anchor="start" class="fig-bold fig-small">model too small</text>
<text x="506.0" y="48.8" text-anchor="end" class="fig-bold fig-small">too few tokens</text>
<text x="83.4" y="152.8" text-anchor="start" class="fig-small fig-bold fill-a">size penalty</text>
<text x="508.1" y="152.8" text-anchor="end" class="fig-small fig-bold fill-d">data penalty</text>
<text x="300.8" y="103.6" text-anchor="start" class="fig-small fig-bold fill-b">sum 0.694</text>
<text x="290.0" y="272.0" text-anchor="middle" class="fig-small">model size N, with D = compute / 6N so that 6ND = 1.2e20</text>
<text x="12.0" y="20.0" text-anchor="start" class="fig-small">loss above the floor</text>
</svg>
Figure 4. Fix the budget at 1.2e20 flops and slide along it. A larger model leaves fewer tokens, so the size penalty (blue) falls while the data penalty (purple) rises. The sum (orange) is a U. The bottom of the U is where the two penalties are equal, because the exponents are equal here.
```

Figure 4 shows a U. On the left the model is small, so the size penalty is large and the data penalty is small. Moving right, the size penalty falls and the data penalty rises. Their sum falls, bottoms out, and climbs again, because a model that is too small and a model that sees too few tokens both lose. The bottom sits at about $N^{*} = 1.01$ billion and $D^{*} = 19.8$ billion tokens, where the two penalties cross at 0.347 each.

Find it exactly. Growing $N$ by 1 percent shrinks $D$ by about 1 percent, so one side pays what the other receives. The size penalty falls by about $\alpha$ percent of itself and the data penalty rises by about $\beta$ percent of itself. You stop sliding where the two are equal:

$$
\frac{\alpha A}{N^{\alpha}} = \frac{\beta B}{D^{\beta}}
$$

Read it aloud: 1 percent more model buys the same loss as 1 percent more data, so no bargain is left. Put $D = C/(6N)$ on the right, so $\beta B / D^{\beta} = \beta B\,(6N/C)^{\beta}$, and collect powers of $N$:

$$
N^{\alpha + \beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}
$$

With $a = \beta/(\alpha+\beta)$ and $G = (\alpha A/\beta B)^{1/(\alpha+\beta)}$, the root is

$$
N^{*} = G \cdot 6^{-a} \cdot C^{a}, \qquad D^{*} = \frac{C}{6N^{*}}
$$

For $10^{20}$ flops with the constants above, $a = 0.5$, $G = (400/1100)^{1/0.68} = 0.2259$, and $6^{-0.5} = 0.4082$. So $N^{*} = 0.2259 \times 0.4082 \times 10^{10} = 9.22 \times 10^{8}$ and $D^{*} = 10^{20}/(6 \times 9.22 \times 10^{8}) = 1.81 \times 10^{10}$. In code:

```python
a = beta / (alpha + beta)
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
n_star = G * 6.0 ** (-a) * compute ** a
d_star = compute / (6.0 * n_star)
```

A grid search checks the algebra: try many sizes, set $D = C/(6N)$ for each, and keep the lowest loss. It lands within 0.2 percent of the closed form.

For these constants $D^{*}/N^{*} = 19.6$, about 20 tokens per parameter at every budget, but only because $\alpha = \beta$. Then $a = 0.5$ and both $N^{*}$ and $D^{*}$ grow as $\sqrt{C}$. When the exponents differ, $N^{*}$ grows as $C^{a}$ and $D^{*}$ as $C^{1-a}$, so the ratio drifts. The balanced split pays: at $1.2 \times 10^{20}$ flops it gives loss 2.3844, while a model 4 times larger on a quarter of the tokens gives $1.69 + 0.2175 + 0.5542 = 2.4617$.

One more cost sits outside this optimum. It minimises training compute for a given loss, and a deployed model also pays about $2N$ flops per token it reads or writes. A model half the size reaches the same loss of 2.3844 if you train it on $4.9 \times 10^{10}$ tokens, 24 percent more training compute, but each served token costs half as much:

$$
\text{total} = 6\,N\,D_{\text{train}} + 2\,N\,D_{\text{served}}
$$

The two totals cross at $2.86 \times 10^{10}$ served tokens. After a trillion served tokens, the balanced model has cost $2.1 \times 10^{21}$ flops and the half-size one $1.2 \times 10^{21}$, so a heavily used model is often trained far past 20 tokens per parameter.

> [!KEY]
> A power-law fit turns a loss measured at small sizes into a prediction, the loss splits into a floor and two penalties, and training costs $6ND$. At a fixed budget the loss along size-versus-data is a U, and its bottom is where 1 percent more model and 1 percent more data are worth the same.

> [!RECAP]
> - A power law $y = c\,x^{k}$ is a line on log axes: fit centred logs, take $k$ as the slope, and return $c = \exp(\text{intercept})$.
> - The loss is $E + A/N^{\alpha} + B/D^{\beta}$, with constants in the order `(E, A, B, alpha, beta)`, and doubling $N$ gains $A/N^{\alpha}(1 - 2^{-\alpha})$ whatever $D$ is.
> - $N = L(4d^2 + 2 d\, d_{\text{ff}})$ counts the weights that multiply, and training costs $C = 6ND$ in floats.
> - At fixed $C$, $D = C/(6N)$, and the loss is lowest where $\alpha A/N^{\alpha} = \beta B/D^{\beta}$.
> - Serving adds $2N$ per token, which pushes toward smaller models trained longer.

## Build it step by step

You write eleven functions, and the checks call them in this order. `predict_loss` is also used by later functions, so write it before `grid_optimal`.

1. `fit_power_law(x, y)` takes positive arrays of shape `(n,)` and returns `(k, c)` as plain floats. Take `np.log` of both, subtract each mean, divide the sum of products by the sum of squares, and exponentiate the intercept. The usual mistake is to return the intercept itself.
2. `extrapolate(x_train, y_train, x_new)` fits the law, then evaluates $c\,x^{k}$ at `x_new` as a float array, using `np.asarray` so a list works. The usual mistake is to put the exponent on `c`.
3. `loss_terms(N, D, coeffs)` returns `(floor, size_penalty, data_penalty)`. The usual mistake is to unpack `coeffs` in another order. At $N = 10^9$ and $D = 2 \times 10^{10}$ expect roughly 1.69, 0.3484, and 0.3460.
4. `predict_loss(N, D, coeffs)` returns the sum of the three terms as one float. For that example the total is 2.3844.
5. `marginal_gain(N, D, coeffs, factor=2.0)` returns the loss before growing the model by `factor` minus the loss after, positive when the loss falls. The usual slip is the sign.
6. `transformer_params(layers, d_model, d_ff, vocab, seq_len)` returns a dict of integers with keys `"attention"`, `"feedforward"`, `"embedding"`, `"non_embedding"`, and `"total"`. The toy sizes give 512, 1024, 928, 1536, and 2464, and the embedding count must not change with the number of blocks.
7. `training_flops(params, tokens)` returns `6.0 * params * tokens`, and `training_days(flops, flops_per_second=2e14)` divides by the speed, then by 86,400. Call `float()` on each input first. At the default speed $10^{20}$ flops gives 5.79 days.
8. `chinchilla_optimal(compute)` returns `(n_star, d_star)` from the closed form, using the stylised constants. The usual mistake is to drop the $6^{-a}$, which pushes the ratio far from 19.6.
9. `grid_optimal(compute, n_points=2000)` builds a log-spaced grid of sizes with `np.geomspace`, sets `D = compute / (6 * N)`, evaluates the loss on the arrays, and returns the `(N, D)` with the lowest loss through `np.argmin`. If it disagrees with the closed form by more than a percent, widen the range before you blame the formula.
10. `lifetime_flops(n_params, train_tokens, served_tokens)` returns $6ND_{\text{train}} + 2ND_{\text{served}}$ as a float. Convert each input first.

> [!TIP]
> Test a fit on data you built, such as `y = 3 * x**1.7`. If 1.7 and 3 do not come back, the bug is in your code, not the data.

## Where this breaks

The constants here are stylised, so the numbers are illustrations, not forecasts. Another data mix, tokeniser, or architecture moves all five, and published fits disagree on the exponents, so a scaling law is a fit to experiments, not a law of physics. A bend in a log-log line is information: if it flattens at the large end, you may have reached the floor, so fit the straight part and be wary of continuing past a bend. Fit the loss above the floor, not the raw loss, which is not a power law. The data also has to exist: at $10^{24}$ flops, $D^{*}$ asks for 1.8 trillion tokens of distinct text, and repeated text helps less than fresh text.

## The exercise

You implement `fit_power_law`, `extrapolate`, `loss_terms`, `predict_loss`, `marginal_gain`, `transformer_params`, `training_flops`, `training_days`, `chinchilla_optimal`, `grid_optimal`, and `lifetime_flops` in `starter.py`. The checks verify the fit against a planted law, the hand example, and noise, that extrapolation lands within 10 percent and errs more far away than near, the three loss terms and the 2.3844 total, the closed form for the gain and its independence of $D$, the toy model's hand count and the 154 days, that `chinchilla_optimal` spends the budget, sits at the bottom of the U, and gives a ratio of 19.6, that the grid agrees within 1 percent, and that `lifetime_flops` reproduces the break-even.
