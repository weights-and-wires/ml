---
title: One dial, three distributions
slug: 021-one-dial-three-distributions
part: "Part 1 — Foundations"
summary: Temperatures, coins, and counts share one shape with one dial, and a single function of the dial gives the mean and the spread.
minutes: 40
gpu: false
objectives:
  - Explain how tilting a small distribution and dividing by the total turns the log of that total into a function whose slope is the mean.
  - Write the Gaussian, Bernoulli, and Poisson distributions in the shared form and read off the mean and variance of each.
  - Compute the log-partition, the mean, and the variance for all three families, and check them with finite differences.
  - Explain why the mean of a count model must be a positive function of the dial, and why a Poisson count's spread grows with its mean.
prereqs:
  - 017-sigmoid-and-probability
  - 018-log-loss
  - 019-gradient-descent-for-logistic
  - 020-newton-method
exercise: 021-one-dial-three-distributions
---

# One dial, three distributions

Three jobs land on your desk in one week. Tomorrow's temperature is a number on a line. Whether a visitor clicks a button is yes or no. The number of people who walk into a shop between 9 and 10 is a count: 0, 1, 2, or more.

You can already fit the first two. Squared error fits the temperature, and the log loss fits the click. Each came with its own prediction rule, its own loss, and its own gradient, and you derived each one from scratch. The count job looks like it needs a third set.

The tempting shortcut is a straight line through the counts, and it fails. A line fitted to counts 3, 1, and 0 at hours 0, 1, and 2 predicts −1.67 arrivals at hour 3, and nobody walks into a shop −1.67 times. A line also ignores that a busy hour naturally varies by more than a quiet one. This chapter asks whether one recipe can give the right mean and the right spread for all three jobs. It can, and the recipe is a single shape with one dial.

## What you already know

Four ideas from earlier chapters carry this one:

- The sigmoid from "From a score to a probability" squashes any score into a probability between 0 and 1, and the log loss for a coin, from "How wrong is a probability? The log loss", is $\operatorname{softplus}(z) - yz$, where softplus is $\log(1 + e^{z})$.
- A slope says how fast a curve rises, and a bend (the second derivative) says how fast the slope itself changes.
- A mean is the probability-weighted average of the outcomes, and a variance is the probability-weighted average squared distance from the mean.
- The probabilities of all outcomes add up to exactly 1.

## One dial slides the distribution

Figure 1 shows the picture before any formula. Each panel is one family, drawn at three settings of a single number $\eta$, pronounced EE-tah. Call $\eta$ the *dial*.

```figure
<svg viewBox="0 0 600 228" role="img" aria-label="Three small plots of one family as its dial moves: Gaussian, Bernoulli, and Poisson"><text x="100.0" y="28.0" text-anchor="middle" class="fig-bold fig-small">Gaussian: slides, same width</text>
<line x1="15.0" y1="175.0" x2="190.0" y2="175.0" class="stroke-muted" />
<rect x="17.0" y="207.0" width="10.0" height="10.0" class="fill-a" />
<text x="31.0" y="216.0" text-anchor="start" class="fig-small">η = −2</text>
<rect x="77.0" y="207.0" width="10.0" height="10.0" class="fill-b" />
<text x="91.0" y="216.0" text-anchor="start" class="fig-small">η = 0</text>
<rect x="137.0" y="207.0" width="10.0" height="10.0" class="fill-c" />
<text x="151.0" y="216.0" text-anchor="start" class="fig-small">η = 2</text>
<text x="295.0" y="28.0" text-anchor="middle" class="fig-bold fig-small">Bernoulli: mass moves to 1</text>
<line x1="210.0" y1="175.0" x2="385.0" y2="175.0" class="stroke-muted" />
<rect x="212.0" y="207.0" width="10.0" height="10.0" class="fill-a" />
<text x="226.0" y="216.0" text-anchor="start" class="fig-small">η = −2</text>
<rect x="272.0" y="207.0" width="10.0" height="10.0" class="fill-b" />
<text x="286.0" y="216.0" text-anchor="start" class="fig-small">η = 0</text>
<rect x="332.0" y="207.0" width="10.0" height="10.0" class="fill-c" />
<text x="346.0" y="216.0" text-anchor="start" class="fig-small">η = 2</text>
<text x="490.0" y="28.0" text-anchor="middle" class="fig-bold fig-small">Poisson: slides and widens</text>
<line x1="405.0" y1="175.0" x2="580.0" y2="175.0" class="stroke-muted" />
<rect x="407.0" y="207.0" width="10.0" height="10.0" class="fill-a" />
<text x="421.0" y="216.0" text-anchor="start" class="fig-small">μ = 0.5</text>
<rect x="467.0" y="207.0" width="10.0" height="10.0" class="fill-b" />
<text x="481.0" y="216.0" text-anchor="start" class="fig-small">μ = 1</text>
<rect x="527.0" y="207.0" width="10.0" height="10.0" class="fill-c" />
<text x="541.0" y="216.0" text-anchor="start" class="fig-small">μ = 3</text>
<polyline points="15.0,173.7 17.2,173.1 19.4,172.3 21.6,171.2 23.9,169.7 26.1,167.8 28.3,165.4 30.5,162.3 32.7,158.5 34.9,154.0 37.2,148.6 39.4,142.4 41.6,135.3 43.8,127.6 46.0,119.1 48.2,110.3 50.4,101.2 52.7,92.1 54.9,83.5 57.1,75.5 59.3,68.6 61.5,63.0 63.7,59.0 65.9,56.7 68.2,56.4 70.4,57.9 72.6,61.2 74.8,66.2 77.0,72.6 79.2,80.2 81.5,88.6 83.7,97.5 85.9,106.6 88.1,115.6 90.3,124.2 92.5,132.3 94.7,139.7 97.0,146.2 99.2,151.9 101.4,156.8 103.6,160.9 105.8,164.2 108.0,166.9 110.3,169.0 112.5,170.6 114.7,171.9 116.9,172.8 119.1,173.5 121.3,174.0 123.5,174.3 125.8,174.5 128.0,174.7 130.2,174.8 132.4,174.9 134.6,174.9 136.8,175.0 139.1,175.0 141.3,175.0 143.5,175.0 145.7,175.0 147.9,175.0 150.1,175.0 152.3,175.0 154.6,175.0 156.8,175.0 159.0,175.0 161.2,175.0 163.4,175.0 165.6,175.0 167.8,175.0 170.1,175.0 172.3,175.0 174.5,175.0 176.7,175.0 178.9,175.0 181.1,175.0 183.4,175.0 185.6,175.0 187.8,175.0 190.0,175.0" fill="none" class="stroke-a" stroke-width="2.5" />
<polyline points="15.0,175.0 17.2,175.0 19.4,175.0 21.6,175.0 23.9,175.0 26.1,175.0 28.3,175.0 30.5,175.0 32.7,175.0 34.9,174.9 37.2,174.9 39.4,174.8 41.6,174.7 43.8,174.6 46.0,174.4 48.2,174.0 50.4,173.6 52.7,172.9 54.9,172.1 57.1,170.9 59.3,169.4 61.5,167.3 63.7,164.8 65.9,161.6 68.2,157.7 70.4,153.0 72.6,147.4 74.8,141.0 77.0,133.8 79.2,125.9 81.5,117.4 83.7,108.4 85.9,99.3 88.1,90.4 90.3,81.8 92.5,74.0 94.7,67.4 97.0,62.1 99.2,58.4 101.4,56.5 103.6,56.5 105.8,58.4 108.0,62.1 110.3,67.4 112.5,74.0 114.7,81.8 116.9,90.4 119.1,99.3 121.3,108.4 123.5,117.4 125.8,125.9 128.0,133.8 130.2,141.0 132.4,147.4 134.6,153.0 136.8,157.7 139.1,161.6 141.3,164.8 143.5,167.3 145.7,169.4 147.9,170.9 150.1,172.1 152.3,172.9 154.6,173.6 156.8,174.0 159.0,174.4 161.2,174.6 163.4,174.7 165.6,174.8 167.8,174.9 170.1,174.9 172.3,175.0 174.5,175.0 176.7,175.0 178.9,175.0 181.1,175.0 183.4,175.0 185.6,175.0 187.8,175.0 190.0,175.0" fill="none" class="stroke-b" stroke-width="2.5" />
<polyline points="15.0,175.0 17.2,175.0 19.4,175.0 21.6,175.0 23.9,175.0 26.1,175.0 28.3,175.0 30.5,175.0 32.7,175.0 34.9,175.0 37.2,175.0 39.4,175.0 41.6,175.0 43.8,175.0 46.0,175.0 48.2,175.0 50.4,175.0 52.7,175.0 54.9,175.0 57.1,175.0 59.3,175.0 61.5,175.0 63.7,175.0 65.9,175.0 68.2,175.0 70.4,174.9 72.6,174.9 74.8,174.8 77.0,174.7 79.2,174.5 81.5,174.3 83.7,174.0 85.9,173.5 88.1,172.8 90.3,171.9 92.5,170.6 94.7,169.0 97.0,166.9 99.2,164.2 101.4,160.9 103.6,156.8 105.8,151.9 108.0,146.2 110.3,139.7 112.5,132.3 114.7,124.2 116.9,115.6 119.1,106.6 121.3,97.5 123.5,88.6 125.8,80.2 128.0,72.6 130.2,66.2 132.4,61.2 134.6,57.9 136.8,56.4 139.1,56.7 141.3,59.0 143.5,63.0 145.7,68.6 147.9,75.5 150.1,83.5 152.3,92.1 154.6,101.2 156.8,110.3 159.0,119.1 161.2,127.6 163.4,135.3 165.6,142.4 167.8,148.6 170.1,154.0 172.3,158.5 174.5,162.3 176.7,165.4 178.9,167.8 181.1,169.7 183.4,171.2 185.6,172.3 187.8,173.1 190.0,173.7" fill="none" class="stroke-c" stroke-width="2.5" />
<rect x="226.0" y="69.3" width="14.0" height="105.7" class="fill-a" />
<rect x="242.0" y="115.0" width="14.0" height="60.0" class="fill-b" />
<rect x="258.0" y="160.7" width="14.0" height="14.3" class="fill-c" />
<text x="250.0" y="191.0" text-anchor="middle" class="fig-small">y = 0</text>
<rect x="316.0" y="160.7" width="14.0" height="14.3" class="fill-a" />
<rect x="332.0" y="115.0" width="14.0" height="60.0" class="fill-b" />
<rect x="348.0" y="69.3" width="14.0" height="105.7" class="fill-c" />
<text x="340.0" y="191.0" text-anchor="middle" class="fig-small">y = 1</text>
<rect x="409.0" y="63.0" width="5.5" height="112.0" class="fill-a" />
<rect x="415.0" y="107.1" width="5.5" height="67.9" class="fill-b" />
<rect x="421.0" y="165.8" width="5.5" height="9.2" class="fill-c" />
<text x="418.0" y="191.0" text-anchor="middle" class="fig-small">0</text>
<rect x="430.5" y="119.0" width="5.5" height="56.0" class="fill-a" />
<rect x="436.5" y="107.1" width="5.5" height="67.9" class="fill-b" />
<rect x="442.5" y="147.4" width="5.5" height="27.6" class="fill-c" />
<text x="439.5" y="191.0" text-anchor="middle" class="fig-small">1</text>
<rect x="452.0" y="161.0" width="5.5" height="14.0" class="fill-a" />
<rect x="458.0" y="141.0" width="5.5" height="34.0" class="fill-b" />
<rect x="464.0" y="133.6" width="5.5" height="41.4" class="fill-c" />
<text x="461.0" y="191.0" text-anchor="middle" class="fig-small">2</text>
<rect x="473.5" y="172.7" width="5.5" height="2.3" class="fill-a" />
<rect x="479.5" y="163.7" width="5.5" height="11.3" class="fill-b" />
<rect x="485.5" y="133.6" width="5.5" height="41.4" class="fill-c" />
<text x="482.5" y="191.0" text-anchor="middle" class="fig-small">3</text>
<rect x="495.0" y="174.7" width="5.5" height="0.3" class="fill-a" />
<rect x="501.0" y="172.2" width="5.5" height="2.8" class="fill-b" />
<rect x="507.0" y="144.0" width="5.5" height="31.0" class="fill-c" />
<text x="504.0" y="191.0" text-anchor="middle" class="fig-small">4</text>
<rect x="516.5" y="175.0" width="5.5" height="0.0" class="fill-a" />
<rect x="522.5" y="174.4" width="5.5" height="0.6" class="fill-b" />
<rect x="528.5" y="156.4" width="5.5" height="18.6" class="fill-c" />
<text x="525.5" y="191.0" text-anchor="middle" class="fig-small">5</text>
<rect x="538.0" y="175.0" width="5.5" height="0.0" class="fill-a" />
<rect x="544.0" y="174.9" width="5.5" height="0.1" class="fill-b" />
<rect x="550.0" y="165.7" width="5.5" height="9.3" class="fill-c" />
<text x="547.0" y="191.0" text-anchor="middle" class="fig-small">6</text>
<rect x="559.5" y="175.0" width="5.5" height="0.0" class="fill-a" />
<rect x="565.5" y="175.0" width="5.5" height="0.0" class="fill-b" />
<rect x="571.5" y="171.0" width="5.5" height="4.0" class="fill-c" />
<text x="568.5" y="191.0" text-anchor="middle" class="fig-small">7</text></svg>
Figure 1. Turning the dial moves each distribution. The Gaussian only slides, the Bernoulli moves its mass from 0 to 1, and the Poisson both slides and widens, with dial settings η = −0.69, 0, and 1.10 for the means 0.5, 1, and 3.
```

Look at the widths. The Gaussian keeps the same width wherever the dial puts it. The Poisson bars are narrow and tall at mean 0.5 and low and wide at mean 3. A Poisson's spread is not a free choice, because the family ties it to the mean. That tie is what the chapter explains, and it is the reason a busy hour varies by more than a quiet one.

## Tilt a tiny distribution to see where the mean comes from

Take three outcomes, 0, 1, and 2, and give each a starting weight of 1. To turn the dial to $\eta$, multiply the weight of outcome $y$ by $e^{\eta y}$. At $\eta = 0$ every weight stays 1. At a positive $\eta$ the big outcomes gain weight, and at a negative $\eta$ the small ones do.

These weights are not probabilities, because they do not add up to 1. So you divide each by their total. The table shows three dial settings.

| $\eta$ | Weights for $y = 0, 1, 2$ | Total | Probabilities | Mean |
|---|---|---|---|---|
| $-1$ | 1, 0.368, 0.135 | 1.503 | 0.665, 0.245, 0.090 | 0.425 |
| $0$ | 1, 1, 1 | 3 | 0.333, 0.333, 0.333 | 1.000 |
| $1$ | 1, 2.718, 7.389 | 11.107 | 0.090, 0.245, 0.665 | 1.575 |

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Weights e to the eta y for outcomes 0, 1, 2 at three dial settings, with their total and mean"><text x="100.0" y="24.0" text-anchor="middle" class="fig-bold">η = −1</text>
<line x1="20.0" y1="160.0" x2="180.0" y2="160.0" class="stroke-muted" />
<rect x="34.0" y="146.5" width="32.0" height="13.5" class="fill-soft-a stroke-a"/>
<text x="50.0" y="140.5" text-anchor="middle" class="fig-small">1.00</text>
<text x="50.0" y="175.0" text-anchor="middle" class="fig-small">y = 0</text>
<rect x="84.0" y="155.0" width="32.0" height="5.0" class="fill-soft-a stroke-a"/>
<text x="100.0" y="149.0" text-anchor="middle" class="fig-small">0.37</text>
<text x="100.0" y="175.0" text-anchor="middle" class="fig-small">y = 1</text>
<rect x="134.0" y="158.2" width="32.0" height="1.8" class="fill-soft-a stroke-a"/>
<text x="150.0" y="152.2" text-anchor="middle" class="fig-small">0.14</text>
<text x="150.0" y="175.0" text-anchor="middle" class="fig-small">y = 2</text>
<polygon points="71.2,182.0 65.2,192.0 77.2,192.0" class="fill-b"/>
<text x="100.0" y="212.0" text-anchor="middle" class="fig-small">total 1.50, A = log total = 0.41</text>
<text x="100.0" y="230.0" text-anchor="middle" class="fig-small">mean (orange marker) = 0.42</text>
<text x="290.0" y="24.0" text-anchor="middle" class="fig-bold">η = 0</text>
<line x1="210.0" y1="160.0" x2="370.0" y2="160.0" class="stroke-muted" />
<rect x="224.0" y="146.5" width="32.0" height="13.5" class="fill-soft-a stroke-a"/>
<text x="240.0" y="140.5" text-anchor="middle" class="fig-small">1.00</text>
<text x="240.0" y="175.0" text-anchor="middle" class="fig-small">y = 0</text>
<rect x="274.0" y="146.5" width="32.0" height="13.5" class="fill-soft-a stroke-a"/>
<text x="290.0" y="140.5" text-anchor="middle" class="fig-small">1.00</text>
<text x="290.0" y="175.0" text-anchor="middle" class="fig-small">y = 1</text>
<rect x="324.0" y="146.5" width="32.0" height="13.5" class="fill-soft-a stroke-a"/>
<text x="340.0" y="140.5" text-anchor="middle" class="fig-small">1.00</text>
<text x="340.0" y="175.0" text-anchor="middle" class="fig-small">y = 2</text>
<polygon points="290.0,182.0 284.0,192.0 296.0,192.0" class="fill-b"/>
<text x="290.0" y="212.0" text-anchor="middle" class="fig-small">total 3.00, A = log total = 1.10</text>
<text x="290.0" y="230.0" text-anchor="middle" class="fig-small">mean (orange marker) = 1.00</text>
<text x="480.0" y="24.0" text-anchor="middle" class="fig-bold">η = 1</text>
<line x1="400.0" y1="160.0" x2="560.0" y2="160.0" class="stroke-muted" />
<rect x="414.0" y="146.5" width="32.0" height="13.5" class="fill-soft-a stroke-a"/>
<text x="430.0" y="140.5" text-anchor="middle" class="fig-small">1.00</text>
<text x="430.0" y="175.0" text-anchor="middle" class="fig-small">y = 0</text>
<rect x="464.0" y="123.2" width="32.0" height="36.8" class="fill-soft-a stroke-a"/>
<text x="480.0" y="117.2" text-anchor="middle" class="fig-small">2.72</text>
<text x="480.0" y="175.0" text-anchor="middle" class="fig-small">y = 1</text>
<rect x="514.0" y="60.0" width="32.0" height="100.0" class="fill-soft-a stroke-a"/>
<text x="530.0" y="54.0" text-anchor="middle" class="fig-small">7.39</text>
<text x="530.0" y="175.0" text-anchor="middle" class="fig-small">y = 2</text>
<polygon points="508.8,182.0 502.8,192.0 514.8,192.0" class="fill-b"/>
<text x="480.0" y="212.0" text-anchor="middle" class="fig-small">total 11.11, A = log total = 2.41</text>
<text x="480.0" y="230.0" text-anchor="middle" class="fig-small">mean (orange marker) = 1.58</text></svg>
Figure 2. Raising the dial piles weight onto the large outcomes, so the mean (orange marker) moves right. The log of the total, A, climbs with it.
```

Call the logarithm of the total $A(\eta)$. At $\eta = 0$ it is $\log 3 = 1.0986$. Here is the surprise. The slope of $A$ is the mean. Check it with a small step of 0.1 on each side:

$$
\frac{A(0.1) - A(-0.1)}{0.2} = \frac{1.2019 - 1.0019}{0.2} = 1.000
$$

That is exactly the mean at $\eta = 0$. The reason is short. A tiny nudge of the dial scales the weight of outcome $y$ by about $1 + y \cdot \text{nudge}$. So the total grows by a factor of $1 + (\text{average } y) \cdot \text{nudge}$, where the average uses the current probabilities. The log of that factor is the mean times the nudge, so the slope of the log total is the mean.

The bend of $A$ is the variance. The bend is how fast the slope changes, and the slope is the mean, so ask how fast the mean moves when you nudge the dial. After a nudge of size $\delta$, the weight of outcome $y$ grows by $1 + y\delta$ and the total grows by $1 + \mu\delta$, where $\mu$ is the current mean. Dividing one by the other, each probability gets multiplied by about $1 + (y - \mu)\delta$. Outcomes above the mean gain probability and outcomes below it lose, and the new mean is

$$
\mu + \delta \sum_y p(y)\,(y - \mu)\,y = \mu + \delta \sum_y p(y)\,(y - \mu)^2 = \mu + \delta \cdot \text{variance}
$$

The two sums agree because $\sum_y p(y)(y - \mu)\mu = 0$: the gaps around the mean cancel. So the mean moves by variance times the nudge. If every outcome were the same, the variance would be 0 and tilting would change nothing. The wider the spread, the faster the mean moves. At $\eta = 0$ the check gives $(1.2019 + 1.0019 - 2 \times 1.0986) / 0.01 = 0.666$, and the variance of outcomes 0, 1, 2 with equal probability is $2/3$. A variance is never negative, so the bend is never negative, and $A$ curves upward.

## One formula for all three

The tilt picture has a compact form. Starting weights are called $h(y)$, and the probability of outcome $y$ is

$$
p(y \mid \eta) = h(y)\, \exp\bigl(\eta\, y - A(\eta)\bigr), \qquad A(\eta) = \log \sum_{y} h(y)\, e^{\eta y}
$$

Read it aloud. The probability is a starting weight $h(y)$, tilted by $e^{\eta y}$, and divided by the total $e^{A(\eta)}$. The division is the part that makes the probabilities add up to 1, and it is the only job of $A$. Because $A$ is the log of that total, people call it the *log-partition*: the total is what you divide by when you share probability out among the outcomes. People call $\eta$ the *natural parameter* because it sits in the exponent in the plainest possible way, multiplied by $y$.

Because $A$ is the log total, the tilt argument gives both results at once:

$$
\text{mean} = \mu = A'(\eta), \qquad \text{variance} = A''(\eta)
$$

Here $\mu$, pronounced mu, is the mean, and $A'$ and $A''$ are the first and second derivatives. Three families fit this shape:

| Family | Outcomes | $h(y)$ | $A(\eta)$ | Mean $A'(\eta)$ | Variance $A''(\eta)$ |
|---|---|---|---|---|---|
| Gaussian, variance 1 | any number | $e^{-y^2/2}/\sqrt{2\pi}$ | $\eta^2/2$ | $\eta$ | $1$ |
| Bernoulli | 0 or 1 | $1$ | $\log(1 + e^{\eta})$ | $\sigma(\eta)$ | $\sigma(\eta)(1 - \sigma(\eta))$ |
| Poisson | 0, 1, 2, … | $1/y!$ | $e^{\eta}$ | $e^{\eta}$ | $e^{\eta}$ |

You can verify the Bernoulli row in one line. The weights are 1 for $y = 0$ and $e^{\eta}$ for $y = 1$, so the total is $1 + e^{\eta}$, and $A$ is its log. The Poisson total is $\sum_y e^{\eta y} / y! = \exp(e^{\eta})$, so $A = e^{\eta}$. For a Gaussian, whose outcomes fill a whole line, the sum becomes an integral, and the result works out to the familiar bell curve centred at $\eta$, because $e^{\eta y - \eta^2/2 - y^2/2} = e^{-(y - \eta)^2/2}$.

A small example confirms the Poisson form. At $\eta = 0$ and $y = 2$, $h(2) = 1/2$ and the exponent is $0 \cdot 2 - 1 = -1$. The probability is $0.5 \times e^{-1} = 0.1839$, the chance of two arrivals when the mean is 1. For a Bernoulli with $\eta = \log 4$, $A = \log 5$, and $P(y = 1) = 4/5 = 0.8$.

> [!KEY]
> Every family here is a tilt of starting weights, divided by a total. The log of that total, $A(\eta)$, has the mean as its slope and the variance as its bend. Only $A$ changes from family to family.

## The count answer: the spread belongs to the mean

Look at the Poisson row again. Its mean and its variance are both $e^{\eta}$. A count averaging 100 has a variance of 100, so a typical wobble of 10. A count averaging 2 has a typical wobble of about 1.4. Figure 3 shows what the bend looks like for two families.

```figure
<svg viewBox="0 0 600 266" role="img" aria-label="The Poisson and Bernoulli log-partition curves with their tangents at eta zero"><line x1="60.0" y1="220.0" x2="60.0" y2="224.0" class="stroke-muted" />
<text x="60.0" y="240.0" text-anchor="middle" class="fig-small">−2</text>
<line x1="185.0" y1="220.0" x2="185.0" y2="224.0" class="stroke-muted" />
<text x="185.0" y="240.0" text-anchor="middle" class="fig-small">−1</text>
<line x1="310.0" y1="220.0" x2="310.0" y2="224.0" class="stroke-muted" />
<text x="310.0" y="240.0" text-anchor="middle" class="fig-small">0</text>
<line x1="435.0" y1="220.0" x2="435.0" y2="224.0" class="stroke-muted" />
<text x="435.0" y="240.0" text-anchor="middle" class="fig-small">1</text>
<line x1="560.0" y1="220.0" x2="560.0" y2="224.0" class="stroke-muted" />
<text x="560.0" y="240.0" text-anchor="middle" class="fig-small">2</text>
<line x1="60.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-muted" />
<line x1="60.0" y1="20.0" x2="60.0" y2="220.0" class="stroke-muted" />
<text x="310.0" y="258.0" text-anchor="middle" class="fig-small">η (the dial)</text>
<line x1="56.0" y1="220.0" x2="60.0" y2="220.0" class="stroke-muted" />
<text x="50.0" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="167.4" x2="60.0" y2="167.4" class="stroke-muted" />
<text x="50.0" y="171.4" text-anchor="end" class="fig-small">2</text>
<line x1="56.0" y1="114.7" x2="60.0" y2="114.7" class="stroke-muted" />
<text x="50.0" y="118.7" text-anchor="end" class="fig-small">4</text>
<line x1="56.0" y1="62.1" x2="60.0" y2="62.1" class="stroke-muted" />
<text x="50.0" y="66.1" text-anchor="end" class="fig-small">6</text>
<polygon points="60.0,216.4 65.1,216.3 70.1,216.1 75.2,216.0 80.2,215.8 85.3,215.6 90.3,215.5 95.4,215.3 100.4,215.1 105.5,214.9 110.5,214.7 115.6,214.4 120.6,214.2 125.7,214.0 130.7,213.7 135.8,213.5 140.8,213.2 145.9,212.9 150.9,212.6 156.0,212.3 161.0,212.0 166.1,211.7 171.1,211.3 176.2,211.0 181.2,210.6 186.3,210.2 191.3,209.8 196.4,209.4 201.4,209.0 206.5,208.5 211.5,208.0 216.6,207.5 221.6,207.0 226.7,206.5 231.7,205.9 236.8,205.4 241.8,204.7 246.9,204.1 251.9,203.5 257.0,202.8 262.0,202.1 267.1,201.3 272.1,200.6 277.2,199.8 282.2,198.9 287.3,198.1 292.3,197.2 297.4,196.2 302.4,195.2 307.5,194.2 312.5,193.1 317.6,192.0 322.6,190.9 327.7,189.7 332.7,188.4 337.8,187.1 342.8,185.8 347.9,184.4 352.9,182.9 358.0,181.4 363.0,179.8 368.1,178.1 373.1,176.4 378.2,174.6 383.2,172.7 388.3,170.8 393.3,168.7 398.4,166.6 403.4,164.4 408.5,162.1 413.5,159.8 418.6,157.3 423.6,154.7 428.7,152.0 433.7,149.2 438.8,146.3 443.8,143.2 448.9,140.1 453.9,136.8 459.0,133.3 464.0,129.8 469.1,126.0 474.1,122.2 479.2,118.1 484.2,113.9 489.3,109.6 494.3,105.0 499.4,100.3 504.4,95.3 509.5,90.2 514.5,84.8 519.6,79.3 524.6,73.5 529.7,67.4 534.7,61.1 539.8,54.6 544.8,47.7 549.9,40.6 554.9,33.3 560.0,25.6 560.0,141.1 554.9,142.1 549.9,143.2 544.8,144.2 539.8,145.3 534.7,146.4 529.7,147.4 524.6,148.5 519.6,149.6 514.5,150.6 509.5,151.7 504.4,152.7 499.4,153.8 494.3,154.9 489.3,155.9 484.2,157.0 479.2,158.1 474.1,159.1 469.1,160.2 464.0,161.3 459.0,162.3 453.9,163.4 448.9,164.4 443.8,165.5 438.8,166.6 433.7,167.6 428.7,168.7 423.6,169.8 418.6,170.8 413.5,171.9 408.5,173.0 403.4,174.0 398.4,175.1 393.3,176.1 388.3,177.2 383.2,178.3 378.2,179.3 373.1,180.4 368.1,181.5 363.0,182.5 358.0,183.6 352.9,184.6 347.9,185.7 342.8,186.8 337.8,187.8 332.7,188.9 327.7,190.0 322.6,191.0 317.6,192.1 312.5,193.2 307.5,194.2 302.4,195.3 297.4,196.3 292.3,197.4 287.3,198.5 282.2,199.5 277.2,200.6 272.1,201.7 267.1,202.7 262.0,203.8 257.0,204.8 251.9,205.9 246.9,207.0 241.8,208.0 236.8,209.1 231.7,210.2 226.7,211.2 221.6,212.3 216.6,213.4 211.5,214.4 206.5,215.5 201.4,216.5 196.4,217.6 191.3,218.7 186.3,219.7 181.2,220.8 176.2,221.9 171.1,222.9 166.1,224.0 161.0,225.1 156.0,226.1 150.9,227.2 145.9,228.2 140.8,229.3 135.8,230.4 130.7,231.4 125.7,232.5 120.6,233.6 115.6,234.6 110.5,235.7 105.5,236.7 100.4,237.8 95.4,238.9 90.3,239.9 85.3,241.0 80.2,242.1 75.2,243.1 70.1,244.2 65.1,245.3 60.0,246.3" class="fill-soft-b"/>
<polyline points="60.0,216.4 65.1,216.3 70.1,216.1 75.2,216.0 80.2,215.8 85.3,215.6 90.3,215.5 95.4,215.3 100.4,215.1 105.5,214.9 110.5,214.7 115.6,214.4 120.6,214.2 125.7,214.0 130.7,213.7 135.8,213.5 140.8,213.2 145.9,212.9 150.9,212.6 156.0,212.3 161.0,212.0 166.1,211.7 171.1,211.3 176.2,211.0 181.2,210.6 186.3,210.2 191.3,209.8 196.4,209.4 201.4,209.0 206.5,208.5 211.5,208.0 216.6,207.5 221.6,207.0 226.7,206.5 231.7,205.9 236.8,205.4 241.8,204.7 246.9,204.1 251.9,203.5 257.0,202.8 262.0,202.1 267.1,201.3 272.1,200.6 277.2,199.8 282.2,198.9 287.3,198.1 292.3,197.2 297.4,196.2 302.4,195.2 307.5,194.2 312.5,193.1 317.6,192.0 322.6,190.9 327.7,189.7 332.7,188.4 337.8,187.1 342.8,185.8 347.9,184.4 352.9,182.9 358.0,181.4 363.0,179.8 368.1,178.1 373.1,176.4 378.2,174.6 383.2,172.7 388.3,170.8 393.3,168.7 398.4,166.6 403.4,164.4 408.5,162.1 413.5,159.8 418.6,157.3 423.6,154.7 428.7,152.0 433.7,149.2 438.8,146.3 443.8,143.2 448.9,140.1 453.9,136.8 459.0,133.3 464.0,129.8 469.1,126.0 474.1,122.2 479.2,118.1 484.2,113.9 489.3,109.6 494.3,105.0 499.4,100.3 504.4,95.3 509.5,90.2 514.5,84.8 519.6,79.3 524.6,73.5 529.7,67.4 534.7,61.1 539.8,54.6 544.8,47.7 549.9,40.6 554.9,33.3 560.0,25.6" fill="none" class="stroke-b" stroke-width="2.5" />
<polyline points="60.0,246.3 65.1,245.3 70.1,244.2 75.2,243.1 80.2,242.1 85.3,241.0 90.3,239.9 95.4,238.9 100.4,237.8 105.5,236.7 110.5,235.7 115.6,234.6 120.6,233.6 125.7,232.5 130.7,231.4 135.8,230.4 140.8,229.3 145.9,228.2 150.9,227.2 156.0,226.1 161.0,225.1 166.1,224.0 171.1,222.9 176.2,221.9 181.2,220.8 186.3,219.7 191.3,218.7 196.4,217.6 201.4,216.5 206.5,215.5 211.5,214.4 216.6,213.4 221.6,212.3 226.7,211.2 231.7,210.2 236.8,209.1 241.8,208.0 246.9,207.0 251.9,205.9 257.0,204.8 262.0,203.8 267.1,202.7 272.1,201.7 277.2,200.6 282.2,199.5 287.3,198.5 292.3,197.4 297.4,196.3 302.4,195.3 307.5,194.2 312.5,193.2 317.6,192.1 322.6,191.0 327.7,190.0 332.7,188.9 337.8,187.8 342.8,186.8 347.9,185.7 352.9,184.6 358.0,183.6 363.0,182.5 368.1,181.5 373.1,180.4 378.2,179.3 383.2,178.3 388.3,177.2 393.3,176.1 398.4,175.1 403.4,174.0 408.5,173.0 413.5,171.9 418.6,170.8 423.6,169.8 428.7,168.7 433.7,167.6 438.8,166.6 443.8,165.5 448.9,164.4 453.9,163.4 459.0,162.3 464.0,161.3 469.1,160.2 474.1,159.1 479.2,158.1 484.2,157.0 489.3,155.9 494.3,154.9 499.4,153.8 504.4,152.7 509.5,151.7 514.5,150.6 519.6,149.6 524.6,148.5 529.7,147.4 534.7,146.4 539.8,145.3 544.8,144.2 549.9,143.2 554.9,142.1 560.0,141.1" fill="none" class="stroke-b dashed" stroke-width="2.5" />
<polyline points="60.0,216.7 65.1,216.5 70.1,216.4 75.2,216.3 80.2,216.1 85.3,216.0 90.3,215.8 95.4,215.7 100.4,215.5 105.5,215.3 110.5,215.1 115.6,215.0 120.6,214.8 125.7,214.6 130.7,214.4 135.8,214.2 140.8,214.0 145.9,213.7 150.9,213.5 156.0,213.3 161.0,213.0 166.1,212.8 171.1,212.5 176.2,212.2 181.2,212.0 186.3,211.7 191.3,211.4 196.4,211.1 201.4,210.8 206.5,210.5 211.5,210.1 216.6,209.8 221.6,209.5 226.7,209.1 231.7,208.7 236.8,208.4 241.8,208.0 246.9,207.6 251.9,207.2 257.0,206.8 262.0,206.3 267.1,205.9 272.1,205.4 277.2,205.0 282.2,204.5 287.3,204.0 292.3,203.6 297.4,203.1 302.4,202.5 307.5,202.0 312.5,201.5 317.6,200.9 322.6,200.4 327.7,199.8 332.7,199.3 337.8,198.7 342.8,198.1 347.9,197.5 352.9,196.9 358.0,196.2 363.0,195.6 368.1,194.9 373.1,194.3 378.2,193.6 383.2,192.9 388.3,192.2 393.3,191.6 398.4,190.8 403.4,190.1 408.5,189.4 413.5,188.7 418.6,187.9 423.6,187.2 428.7,186.4 433.7,185.6 438.8,184.9 443.8,184.1 448.9,183.3 453.9,182.5 459.0,181.7 464.0,180.8 469.1,180.0 474.1,179.2 479.2,178.3 484.2,177.5 489.3,176.6 494.3,175.8 499.4,174.9 504.4,174.0 509.5,173.1 514.5,172.3 519.6,171.4 524.6,170.5 529.7,169.6 534.7,168.7 539.8,167.7 544.8,166.8 549.9,165.9 554.9,165.0 560.0,164.0" fill="none" class="stroke-a" stroke-width="2.5" />
<polyline points="60.0,228.1 65.1,227.5 70.1,227.0 75.2,226.5 80.2,225.9 85.3,225.4 90.3,224.9 95.4,224.4 100.4,223.8 105.5,223.3 110.5,222.8 115.6,222.2 120.6,221.7 125.7,221.2 130.7,220.6 135.8,220.1 140.8,219.6 145.9,219.0 150.9,218.5 156.0,218.0 161.0,217.4 166.1,216.9 171.1,216.4 176.2,215.8 181.2,215.3 186.3,214.8 191.3,214.3 196.4,213.7 201.4,213.2 206.5,212.7 211.5,212.1 216.6,211.6 221.6,211.1 226.7,210.5 231.7,210.0 236.8,209.5 241.8,208.9 246.9,208.4 251.9,207.9 257.0,207.3 262.0,206.8 267.1,206.3 272.1,205.7 277.2,205.2 282.2,204.7 287.3,204.2 292.3,203.6 297.4,203.1 302.4,202.6 307.5,202.0 312.5,201.5 317.6,201.0 322.6,200.4 327.7,199.9 332.7,199.4 337.8,198.8 342.8,198.3 347.9,197.8 352.9,197.2 358.0,196.7 363.0,196.2 368.1,195.6 373.1,195.1 378.2,194.6 383.2,194.1 388.3,193.5 393.3,193.0 398.4,192.5 403.4,191.9 408.5,191.4 413.5,190.9 418.6,190.3 423.6,189.8 428.7,189.3 433.7,188.7 438.8,188.2 443.8,187.7 448.9,187.1 453.9,186.6 459.0,186.1 464.0,185.5 469.1,185.0 474.1,184.5 479.2,183.9 484.2,183.4 489.3,182.9 494.3,182.4 499.4,181.8 504.4,181.3 509.5,180.8 514.5,180.2 519.6,179.7 524.6,179.2 529.7,178.6 534.7,178.1 539.8,177.6 544.8,177.0 549.9,176.5 554.9,176.0 560.0,175.4" fill="none" class="stroke-a dashed" stroke-width="2.5" />
<circle cx="310.0" cy="193.7" r="5" class="fill-b"/>
<circle cx="310.0" cy="201.8" r="5" class="fill-a"/>
<text x="491.2" y="85.8" text-anchor="end" class="fig-bold">Poisson: A = e^η</text>
<text x="491.2" y="101.6" text-anchor="end" class="fig-small">slope 1, bend 1</text>
<text x="556.0" y="203.7" text-anchor="end" class="fig-bold">Bernoulli: softplus</text>
<text x="556.0" y="214.7" text-anchor="end" class="fig-small">slope 0.5, bend 0.25</text>
<text x="72.5" y="151.6" text-anchor="start" class="fig-small">shaded gap = how far A</text>
<text x="72.5" y="164.7" text-anchor="start" class="fig-small">leaves its tangent</text></svg>
Figure 3. Each curve is a log-partition A(η) with its tangent at η = 0. The curve leaves the tangent faster where the bend is larger, so the Poisson (bend 1) bows away much faster than the Bernoulli softplus (bend 0.25).
```

So the shared recipe answers the opening question about spread. It also settles why a straight line on the counts fails. Whatever dial setting you choose, $A'(\eta) = e^{\eta}$ is positive. The prediction cannot go below 0, which a straight line cannot promise.

## The link: from a mean back to a dial setting

Why not skip the dial and set the mean to a linear score directly? Because a mean lives in a restricted range: a probability between 0 and 1, or a count above 0. A straight line leaves any range, which is the −1.67 arrivals from the start of the chapter. The dial has no such limit. Any real number is a valid $\eta$, so you can let a line choose it, and then let $A'$ map the dial into the range the mean must stay in. For the Poisson, $A'(\eta) = e^{\eta}$ is positive for every dial setting. For the Bernoulli, $A'(\eta) = \sigma(\eta)$ is the sigmoid you already know, squeezing any score into $(0, 1)$.

The reverse question also comes up: given a mean you want, which dial setting produces it? That reverse map is the *link*, written $g$, and it undoes $A'$. Because a variance is positive, $A'$ always rises as the dial rises, so it can be undone. When a model uses exactly this reverse map, it uses the *canonical* link. The three canonical links are:

- Gaussian: the identity, $\eta = \mu$.
- Bernoulli: the logit, $\eta = \log\frac{\mu}{1 - \mu}$. A mean of 0.8 gives $\log 4 = 1.386$.
- Poisson: the log, $\eta = \log \mu$. A mean of 1 gives $\eta = 0$.

You never need to code the link in this chapter, because you only go from dial to mean. It matters later: the chapter "Curvature and Fisher scoring" explains what changes when a model uses some other link.

## Check it with numbers

Two lines of numpy test both derivatives of any $A$ against its definition:

```python
mean_estimate = (A(eta + h) - A(eta - h)) / (2 * h)
var_estimate = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2
```

> [!TIP]
> Use a step near `1e-6` for the first difference and near `1e-3` for the second. A much smaller second-difference step lets rounding error swamp the bend. The checks in this chapter use these step sizes.

> [!WARNING]
> Writing the Bernoulli $A$ as `np.log(1 + np.exp(eta))` overflows once `eta` passes about 700. The form `np.maximum(eta, 0) + np.log1p(np.exp(-np.abs(eta)))` stays finite at any size.

> [!RECAP]
> Hold these ideas before you write code:
>
> - The dial $\eta$ tilts starting weights, and $A(\eta)$ is the log of their total.
> - The mean is $A'(\eta)$ and the variance is $A''(\eta)$, because tilting moves the log total by the average $y$.
> - $A$ is $\eta^2/2$, softplus, or $e^{\eta}$ for the Gaussian, Bernoulli, and Poisson.
> - A mean must stay in its range, so the prediction is always $A'$ of the dial, never the dial itself.
> - A Poisson's variance equals its mean, so its spread grows as it moves right.

## Build it step by step

You write three functions. Each takes a family name, one of `"gaussian"`, `"bernoulli"`, or `"poisson"`, and an array `eta`. Each returns an array of the same shape as `eta` and raises `ValueError` for any other name.

1. `log_partition(family, eta)` returns $A(\eta)$. Branch on the family and return `0.5 * eta ** 2`, the stable softplus, or `np.exp(eta)`. The common mistake is the naive softplus, which overflows at large `eta`.
2. `mean_from_natural(family, eta)` returns $A'(\eta)$. For the Bernoulli, `np.exp(-np.logaddexp(0, -eta))` is a sigmoid that stays finite on either side. The common mistake is `1 / (1 + np.exp(-eta))`, which overflows for very negative `eta`.
3. `variance_from_natural(family, eta)` returns $A''(\eta)$. Use `np.ones_like(eta)`, `p * (1 - p)` from the mean you just wrote, or `np.exp(eta)`. The common mistake is returning the bare number `1` for the Gaussian, which has the wrong shape.

## Where this breaks

- **Counts that spread wider than their mean.** Real arrival counts often vary more than a Poisson allows, because the family forces variance to equal the mean. This is *overdispersion*, and a model that assumes the Poisson tie reports more confidence than the data deserve.
- **A Gaussian with another variance.** Here the Gaussian has variance fixed at 1. Real temperatures have their own noise scale, and handling that takes an extra parameter that this chapter leaves out.
- **Distributions outside the family.** Not every distribution has this shape. A Cauchy bump, whose tails are too heavy to have a mean at all, has no $A$.

## The exercise

You implement `log_partition`, `mean_from_natural`, and `variance_from_natural` in `starter.py`. The checks compare `log_partition` with $\eta^2/2$, the stable softplus, and $e^{\eta}$, including at $\eta = \pm 800$. They check that $e^{\eta y - A(\eta)}$, multiplied by the right $h(y)$, adds up to 1 for each family, which confirms that $A$ really is the log of the total. They compare `mean_from_natural` with a finite-difference slope of `log_partition`, and with the average $y$ under the Poisson probabilities. They compare `variance_from_natural` with a finite-difference bend, and they check that the Poisson mean equals its variance. The first check passes as soon as `log_partition` is right.
