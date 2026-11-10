---
title: "Learning theory: why generalisation bounds hold"
slug: 014-learning-theory
part: "Part 2 — Classical models"
summary: "Turn Hoeffding, the union bound, VC dimension, and the perceptron mistake bound into answers about data needs, and see where double descent breaks the picture."
minutes: 70
gpu: false
objectives:
  - "Write Hoeffding's inequality, check it with a coin-flip simulation, and explain why it is loose but has the right rate."
  - "Solve the union bound for the number of examples n, or for the gap epsilon, when you choose among k models."
  - "Shatter small point sets by brute force to find the VC dimension of thresholds, intervals, and half-planes."
  - "Run a perceptron and compare its mistake count with the (R / margin)^2 bound."
  - "Sweep min-norm least squares across p = n and explain why test error spikes there, then falls."
prereqs:
  - 013-learning-curves-and-winners-curse
exercise: 014-learning-theory
---

# Learning theory: why generalisation bounds hold

You are about to label emails as spam or not spam by hand, and you want to train a filter on the result. Labelling costs money, so you need to know how many emails to label. Fifty is cheap and may not be enough. Fifty thousand is safe and may be a waste. So far the course has given you an empirical answer: draw a learning curve and see where it flattens. That answer arrives after you have paid for the data.

Learning theory answers before you spend anything. It bounds the distance between a model's error on the training emails and on all future emails, using the number of emails $n$ and the richness of the family of candidate filters. This chapter builds the bound in steps, tests each step with a simulation, and ends with double descent, where "richer means worse" fails.

You will meet four ideas. *Hoeffding's inequality* says an average of bounded values rarely strays from its mean. The *union bound* stretches that to $k$ models. *VC dimension* replaces $k$ when the family is infinite. The *perceptron mistake bound* guarantees something without any probability.

## What you already know

- Training error is a model's average loss on the examples it was fitted on, and true error is its average loss on all future examples. A bound controls the gap. See [loss and honest scores](/lessons/001-loss-and-honest-scores).
- Too few weights underfit and too many chase noise. See [curves and bias-variance](/lessons/002-curves-and-bias-variance).
- The best of many noisy scores is too good, and the discount grows with the logarithm of the number of models. See [learning curves and the winner's curse](/lessons/013-learning-curves-and-winners-curse).
- If several events each fail with some chance, the chance that at least one fails is at most the sum of the chances.
- The pseudo-inverse gives the shortest weights that fit. See [stable least squares](/lessons/004-stable-least-squares).

## An average is rarely far from its mean

Start with a fair coin, flipped $n$ times. With $n = 4$ you can see 0.75 heads, because three in four is ordinary luck. With $n = 400$ you almost never will. More flips squeeze the fraction of heads toward 0.5, and the question is how fast.

Figure 1 answers by brute force. For each $n$ from 10 to 150, it flips $n$ coins 400,000 times and records how often the average misses 0.5 by more than 0.1. Next to those frequencies it draws a formula.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="Hoeffding bound and simulated deviation chance against n on a log axis; both fall in parallel lines">
<line x1="60" y1="230.0" x2="600" y2="230.0" class="stroke-grid"/>
<text x="54" y="234.0" text-anchor="end" class="fig-small">1e-5</text>
<line x1="60" y1="192.0" x2="600" y2="192.0" class="stroke-grid"/>
<text x="54" y="196.0" text-anchor="end" class="fig-small">1e-4</text>
<line x1="60" y1="154.0" x2="600" y2="154.0" class="stroke-grid"/>
<text x="54" y="158.0" text-anchor="end" class="fig-small">1e-3</text>
<line x1="60" y1="116.0" x2="600" y2="116.0" class="stroke-grid"/>
<text x="54" y="120.0" text-anchor="end" class="fig-small">1e-2</text>
<line x1="60" y1="78.0" x2="600" y2="78.0" class="stroke-grid"/>
<text x="54" y="82.0" text-anchor="end" class="fig-small">1e-1</text>
<line x1="60" y1="40.0" x2="600" y2="40.0" class="stroke-grid"/>
<text x="54" y="44.0" text-anchor="end" class="fig-small">1</text>
<text x="60.0" y="246" text-anchor="middle" class="fig-small">10</text>
<text x="214.3" y="246" text-anchor="middle" class="fig-small">50</text>
<text x="407.1" y="246" text-anchor="middle" class="fig-small">100</text>
<text x="600.0" y="246" text-anchor="middle" class="fig-small">150</text>
<line x1="60" y1="230" x2="600" y2="230" class="stroke-muted"/><line x1="60" y1="40" x2="60" y2="230" class="stroke-muted"/>
<polyline points="60.0,40.0 98.6,40.0 137.1,40.0 175.7,41.8 214.3,45.1 252.9,48.4 291.4,51.7 330.0,55.0 368.6,58.3 407.1,61.6 445.7,64.9 484.3,68.2 522.9,71.5 561.4,74.8 600.0,78.1" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="60.0" cy="40.0" r="4" class="fill-b"/>
<circle cx="98.6" cy="40.0" r="4" class="fill-b"/>
<circle cx="137.1" cy="40.0" r="4" class="fill-b"/>
<circle cx="175.7" cy="41.8" r="4" class="fill-b"/>
<circle cx="214.3" cy="45.1" r="4" class="fill-b"/>
<circle cx="252.9" cy="48.4" r="4" class="fill-b"/>
<circle cx="291.4" cy="51.7" r="4" class="fill-b"/>
<circle cx="330.0" cy="55.0" r="4" class="fill-b"/>
<circle cx="368.6" cy="58.3" r="4" class="fill-b"/>
<circle cx="407.1" cy="61.6" r="4" class="fill-b"/>
<circle cx="445.7" cy="64.9" r="4" class="fill-b"/>
<circle cx="484.3" cy="68.2" r="4" class="fill-b"/>
<circle cx="522.9" cy="71.5" r="4" class="fill-b"/>
<circle cx="561.4" cy="74.8" r="4" class="fill-b"/>
<circle cx="600.0" cy="78.1" r="4" class="fill-b"/>
<polyline points="60.0,57.6 98.6,62.0 137.1,66.5 175.7,70.9 214.3,75.2 252.9,79.4 291.4,83.4 330.0,87.4 368.6,91.4 407.1,95.1 445.7,99.0 484.3,102.9 522.9,106.5 561.4,110.2 600.0,114.2" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="60.0" cy="57.6" r="4" class="fill-a"/>
<circle cx="98.6" cy="62.0" r="4" class="fill-a"/>
<circle cx="137.1" cy="66.5" r="4" class="fill-a"/>
<circle cx="175.7" cy="70.9" r="4" class="fill-a"/>
<circle cx="214.3" cy="75.2" r="4" class="fill-a"/>
<circle cx="252.9" cy="79.4" r="4" class="fill-a"/>
<circle cx="291.4" cy="83.4" r="4" class="fill-a"/>
<circle cx="330.0" cy="87.4" r="4" class="fill-a"/>
<circle cx="368.6" cy="91.4" r="4" class="fill-a"/>
<circle cx="407.1" cy="95.1" r="4" class="fill-a"/>
<circle cx="445.7" cy="99.0" r="4" class="fill-a"/>
<circle cx="484.3" cy="102.9" r="4" class="fill-a"/>
<circle cx="522.9" cy="106.5" r="4" class="fill-a"/>
<circle cx="561.4" cy="110.2" r="4" class="fill-a"/>
<circle cx="600.0" cy="114.2" r="4" class="fill-a"/>
<text x="417.1" y="51.6" class="fig-bold">bound 2 exp(-2 n eps²)</text>
<text x="417.1" y="121.1" class="fig-bold">simulated chance</text>
<text x="330" y="266" text-anchor="middle">n, coin flips per average</text>
<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">chance of a miss above 0.1</text>
</svg>
Figure 1. Hoeffding's bound and the simulated chance that the average of n fair-coin flips misses 0.5 by more than 0.1. The bound stays above the simulation, and both fall at nearly the same exponential rate. The bound is capped at 1 for small n.
```

The formula is *Hoeffding's inequality*. Take $n$ independent values, each between 0 and 1, from one source, with mean $\mu$ and average $\bar{x}$. For any gap $\varepsilon > 0$:

$$
P\big(\,|\bar{x} - \mu| > \varepsilon\,\big) \;\le\; 2\,e^{-2 n \varepsilon^2}
$$

Read it aloud: the chance that the average misses the mean by more than $\varepsilon$ is at most 2 times $e$ to the power minus 2 times $n$ times $\varepsilon$ squared. Because $n$ sits in the exponent, each extra example multiplies the ceiling by a constant. Because $\varepsilon^2$ does, a tenfold tighter gap costs a hundredfold more data. The leading 2 counts both tails, an average too high and one too low. In numpy: `min(1.0, 2 * np.exp(-2 * n * eps**2))`, capped at 1 because a probability cannot exceed 1.

Work one case. For $n = 100$ and $\varepsilon = 0.1$ the exponent is $-2 \cdot 100 \cdot 0.01 = -2$, so the bound is $2e^{-2} = 0.2707$. The simulation says 0.035. The bound holds, and it overshoots by a factor of 7.6.

### Why the bound is loose, and why that is fine

Figure 1 shows two things at once. The bound sits above the simulation at every $n$, so it never fails. And the two curves fall at nearly the same slope on the log axis. The gap between them grows only from a factor of 3 at $n = 10$ to 9 at $n = 150$, while the chance itself falls by a factor of 30. The exponent is right. What the bound misses is a slowly growing prefactor that the exact tail carries.

The bound is loose because it must hold for every distribution on $[0, 1]$ and uses only the range. Treat it as a ceiling to plan with, not a forecast.

> [!KEY]
> Hoeffding's bound is a ceiling that holds for every distribution on $[0, 1]$, so it overshoots the typical case. Its rate is right: the chance of a miss falls exponentially in $n\varepsilon^2$, so data cost grows as $1/\varepsilon^2$.

> [!WARNING]
> The bound needs values in $[0, 1]$, independent draws, and one source. Duplicated emails break independence, and squared error on house prices has no upper limit. The bound does not fail gracefully when you ignore these.

## From one model to a class of k models

Hoeffding bounds the gap for one model whose loss is between 0 and 1, such as 1 when a filter is wrong and 0 when it is right. The training error of a model fixed before you saw the data is an average of $n$ such numbers.

The trouble starts when you choose among models by training error. The winner may have fit the training emails best by luck. To trust it, you need every model's training error to be close to its own true error at the same time. The [previous chapter](/lessons/013-learning-curves-and-winners-curse) quoted the result. Here you derive it, so you can solve it for the quantity you care about.

Suppose the family holds $k$ models. Each one breaks Hoeffding's guarantee with chance at most $2e^{-2n\varepsilon^2}$, so the chance that at least one of the $k$ does is at most $k$ times that. This use of the sum rule is the *union bound*. Set the total failure chance to a number $\delta$ that you choose, such as 0.05, and solve for $n$ or for $\varepsilon$:

$$
2k\,e^{-2n\varepsilon^2} = \delta
\quad\Longrightarrow\quad
n = \frac{\ln(2k/\delta)}{2\varepsilon^2}
\quad\text{or}\quad
\varepsilon = \sqrt{\frac{\ln(2k/\delta)}{2n}}
$$

Read the first formula aloud: the number of examples is the log of twice the family size over the failure chance, divided by twice the squared gap. With chance at least $1 - \delta$, every one of the $k$ models has a training error within $\varepsilon$ of its true error. The family size sits inside a logarithm because taking logs of both sides turns the factor $k$ into an addition. In numpy: `int(np.ceil(np.log(2 * k / delta) / (2 * eps**2)))`.

Work an example with $\varepsilon = 0.05$ and $\delta = 0.05$. For one model, $\ln(2 \cdot 1 / 0.05) = \ln 40 = 3.689$, and dividing by $2 \cdot 0.0025 = 0.005$ gives 737.8, so you need 738 examples. For $k = 1000$, $\ln 40000 = 10.597$, and the answer is 2120. The table shows how little the family size costs.

| Models in the family, $k$ | Examples needed for $\varepsilon = \delta = 0.05$ |
|---|---|
| 1 | 738 |
| 10 | 1199 |
| 1,000 | 2120 |
| 1,000,000 | 3501 |

A family a million times larger needs under five times the data. Each factor of 10 in $k$ adds the same $\ln 10 / (2\varepsilon^2) = 461$ examples.

If every training error is within $\varepsilon$ of its true error, the model with the lowest training error has a true error within $2\varepsilon$ of the best model in the family. You pay $\varepsilon$ once because the winner looks better than it is, and once because the best model looks worse than it is.

> [!WARNING]
> The family must be fixed before you look at the data. If you inspect the data, pick 1000 promising filters, and then train, the real family is every filter you could have picked.

## When the family is infinite: shattering and VC dimension

A threshold is a continuous knob, so there are infinitely many models and $\ln k$ is meaningless. The fix is to count the *behaviours* the models can show on $n$ points. Two thresholds that differ by 0.001 give the same labels on 20 emails, so they act as one.

A *dichotomy* is one pattern of 0 and 1 labels on a set of points, such as $(1, 0, 1)$. A family of rules *shatters* $m$ points when it produces all $2^m$ patterns on them. The *VC dimension*, named after Vapnik and Chervonenkis, is the largest $m$ for which some set of $m$ points is shattered. One shattered set shows the dimension is at least $m$. To show it is exactly $m$, you must show that no set of $m + 1$ points is shattered.

Work three families by hand.

**Thresholds on a line**, $h_t(x) = 1$ when $x \ge t$. One point is shattered: put $t$ below it for label 1 and above it for label 0. Two points at $x = 2$ and $x = 3$ are not, because the pattern $(1, 0)$ needs 2 labelled 1 and 3 labelled 0, and every threshold that labels 2 as 1 labels 3 as 1. The VC dimension is 1.

**Intervals on a line**, $h_{a,b}(x) = 1$ when $a \le x \le b$. Two points are shattered: an interval around both, around one, or around neither. Three points at $x = 1, 2, 3$ are not, because $(1, 0, 1)$ needs an interval with a hole. The VC dimension is 2.

**Half-planes in the plane**, a line with one side labelled 1. Three corners of a triangle are shattered, which you can check by listing all 8 patterns. Four points are not. For the corners of a square, label the ends of one diagonal 1 and the ends of the other 0, and no line separates them. The VC dimension is 3, and in $d$ dimensions it is $d + 1$.

### From shattering to a bound

Past $d$ points the family runs out of patterns: on $n$ points it makes at most about $(en/d)^d$ dichotomies instead of $2^n$. It behaves like a finite family of $k \approx n^d$ models, so $\ln k$ becomes $d \ln n$. Up to constant factors, with chance at least $1 - \delta$, every rule's training error is within

$$
\sqrt{\frac{d \ln (n/d) + \ln (1/\delta)}{n}}
$$

of its true error. Read it aloud: the gap shrinks like one over the square root of $n$ and grows like the square root of $d$. Halving the gap takes four times the data, and a linear filter with 50 weights needs roughly ten times the examples of one with 5.

## The perceptron mistake bound

The *perceptron* keeps a weight vector $w$, starting at zero, and goes through the training points in order. For a point $x$ with label $y$ in $\{-1, +1\}$ it predicts $\operatorname{sign}(w \cdot x)$. When $y\,(w \cdot x) \le 0$ the point is a mistake, and the perceptron updates $w \leftarrow w + y\,x$. It repeats passes until one makes no mistake.

If the data is *separable*, meaning some direction $w^*$ puts every point on the correct side, it stops. Let the *margin* $\gamma$ be the smallest distance of any point from the separating line, $\min_i y_i (w^* \cdot x_i) / \lVert w^* \rVert$, and let $R$ be the length of the longest point. The number of mistakes is at most

$$
\left(\frac{R}{\gamma}\right)^2
$$

The argument has two facts. Each mistake adds at least $\gamma$ to the component of $w$ along the unit separator, so after $k$ mistakes that component is at least $k\gamma$. Each mistake adds at most $R^2$ to $\lVert w \rVert^2$, because on a mistake $2y\,(w \cdot x) \le 0$, so $\lVert w \rVert \le \sqrt{k}\,R$. A component cannot exceed the full length, so $k\gamma \le \sqrt{k}\,R$, which gives $k \le (R/\gamma)^2$. Neither $n$ nor the dimension appears.

Work four points: $x_1 = (1, 1)$ with label $+1$, $x_2 = (1, -2)$ with $+1$, $x_3 = (-1, -2)$ with $-1$, and $x_4 = (1, 2)$ with $+1$. The direction $w^* = (1, 0)$ separates them with $\gamma = 1$, and $R^2 = 5$, so the bound is 5. Trace the perceptron from $w = (0, 0)$:

| Step | Point | $w \cdot x$ | $y\,(w \cdot x)$ | Result | $w$ after |
|---|---|---|---|---|---|
| 1 | $x_1$ | 0 | 0 | mistake | $(1, 1)$ |
| 2 | $x_2$ | $-1$ | $-1$ | mistake | $(2, -1)$ |
| 3 | $x_3$ | 0 | 0 | mistake | $(3, 1)$ |
| 4 | $x_4$ | 5 | 5 | correct | $(3, 1)$ |

A second pass makes no mistake, so the perceptron stops after 3 mistakes, under the bound of 5. The zero at step 1 matters: a rule that needs a strictly negative product never moves from $w = 0$.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="Perceptron mistakes against the bound, both on log axes: every point sits below the line where mistakes equal the bound">
<line x1="70.0" y1="30" x2="70.0" y2="230" class="stroke-grid"/>
<line x1="70" y1="230.0" x2="590" y2="230.0" class="stroke-grid"/>
<text x="70.0" y="246" text-anchor="middle" class="fig-small">1</text>
<text x="64" y="234.0" text-anchor="end" class="fig-small">1</text>
<line x1="174.0" y1="30" x2="174.0" y2="230" class="stroke-grid"/>
<line x1="70" y1="190.0" x2="590" y2="190.0" class="stroke-grid"/>
<text x="174.0" y="246" text-anchor="middle" class="fig-small">10</text>
<text x="64" y="194.0" text-anchor="end" class="fig-small">10</text>
<line x1="278.0" y1="30" x2="278.0" y2="230" class="stroke-grid"/>
<line x1="70" y1="150.0" x2="590" y2="150.0" class="stroke-grid"/>
<text x="278.0" y="246" text-anchor="middle" class="fig-small">100</text>
<text x="64" y="154.0" text-anchor="end" class="fig-small">100</text>
<line x1="382.0" y1="30" x2="382.0" y2="230" class="stroke-grid"/>
<line x1="70" y1="110.0" x2="590" y2="110.0" class="stroke-grid"/>
<text x="382.0" y="246" text-anchor="middle" class="fig-small">1000</text>
<text x="64" y="114.0" text-anchor="end" class="fig-small">1000</text>
<line x1="486.0" y1="30" x2="486.0" y2="230" class="stroke-grid"/>
<line x1="70" y1="70.0" x2="590" y2="70.0" class="stroke-grid"/>
<text x="486.0" y="246" text-anchor="middle" class="fig-small">10000</text>
<text x="64" y="74.0" text-anchor="end" class="fig-small">10000</text>
<line x1="590.0" y1="30" x2="590.0" y2="230" class="stroke-grid"/>
<line x1="70" y1="30.0" x2="590" y2="30.0" class="stroke-grid"/>
<text x="590.0" y="246" text-anchor="middle" class="fig-small">100000</text>
<text x="64" y="34.0" text-anchor="end" class="fig-small">100000</text>
<line x1="70" y1="230" x2="590" y2="230" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="230" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="590.0" y2="30.0" class="stroke-b dashed" stroke-width="2"/>
<circle cx="197.9" cy="218.0" r="4" class="fill-a"/>
<circle cx="197.6" cy="230.0" r="4" class="fill-a"/>
<circle cx="195.3" cy="210.9" r="4" class="fill-a"/>
<circle cx="196.1" cy="230.0" r="4" class="fill-a"/>
<circle cx="188.4" cy="218.0" r="4" class="fill-a"/>
<circle cx="250.2" cy="218.0" r="4" class="fill-a"/>
<circle cx="256.0" cy="218.0" r="4" class="fill-a"/>
<circle cx="248.0" cy="218.0" r="4" class="fill-a"/>
<circle cx="252.3" cy="210.9" r="4" class="fill-a"/>
<circle cx="256.5" cy="210.9" r="4" class="fill-a"/>
<circle cx="305.5" cy="198.9" r="4" class="fill-a"/>
<circle cx="309.2" cy="196.2" r="4" class="fill-a"/>
<circle cx="316.3" cy="202.0" r="4" class="fill-a"/>
<circle cx="322.9" cy="218.0" r="4" class="fill-a"/>
<circle cx="320.6" cy="218.0" r="4" class="fill-a"/>
<circle cx="372.7" cy="218.0" r="4" class="fill-a"/>
<circle cx="369.5" cy="196.2" r="4" class="fill-a"/>
<circle cx="345.1" cy="205.9" r="4" class="fill-a"/>
<circle cx="375.9" cy="198.9" r="4" class="fill-a"/>
<circle cx="375.4" cy="210.9" r="4" class="fill-a"/>
<circle cx="450.1" cy="157.8" r="4" class="fill-a"/>
<circle cx="443.6" cy="183.0" r="4" class="fill-a"/>
<circle cx="371.8" cy="196.2" r="4" class="fill-a"/>
<circle cx="426.3" cy="205.9" r="4" class="fill-a"/>
<circle cx="445.3" cy="196.2" r="4" class="fill-a"/>
<circle cx="435.5" cy="198.9" r="4" class="fill-a"/>
<circle cx="481.4" cy="198.9" r="4" class="fill-a"/>
<circle cx="500.4" cy="205.9" r="4" class="fill-a"/>
<circle cx="420.8" cy="218.0" r="4" class="fill-a"/>
<circle cx="408.4" cy="218.0" r="4" class="fill-a"/>
<circle cx="492.5" cy="185.4" r="4" class="fill-a"/>
<circle cx="528.3" cy="190.0" r="4" class="fill-a"/>
<circle cx="431.8" cy="198.9" r="4" class="fill-a"/>
<circle cx="466.7" cy="190.0" r="4" class="fill-a"/>
<circle cx="552.9" cy="196.2" r="4" class="fill-a"/>
<circle cx="517.1" cy="218.0" r="4" class="fill-a"/>
<circle cx="424.5" cy="205.9" r="4" class="fill-a"/>
<circle cx="445.7" cy="218.0" r="4" class="fill-a"/>
<circle cx="443.2" cy="210.9" r="4" class="fill-a"/>
<circle cx="553.7" cy="188.3" r="4" class="fill-a"/>
<text x="101.3" y="103.0" class="fig-bold">mistakes = bound</text>
<text x="330" y="266" text-anchor="middle">bound (R / margin)²</text>
<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">mistakes made</text>
</svg>
Figure 2. Perceptron mistakes against the (R / margin)² bound on 40 seeded separable problems. Every dot sits below the dashed line where mistakes equal the bound, and most sit far below.
```

Figure 2 repeats this on 40 seeded problems with margins from 1 down to 0.008. Every dot sits below the dashed line where mistakes equal the bound, and a typical problem used about 2 percent of it. The bound is a worst case. It carries the right message: a smaller margin allows far more mistakes.

> [!KEY]
> A bound has two jobs: to hold, and to say what drives the answer. Hoeffding says the data you need grows as $1/\varepsilon^2$ and as $\ln k$. The VC bound says $\ln k$ becomes $d \ln n$. The perceptron bound says the mistakes depend on $R/\gamma$ and on nothing else.

## Double descent: more weights, then better

The classical story says a model with more freedom than the data supports overfits, so test error rises as the model gets richer. You saw this in [curves and bias-variance](/lessons/002-curves-and-bias-variance), and the VC bound agrees, because $d$ grows with the weights. Belkin and colleagues showed in 2019 that the story has a second act.

The setup: $n = 30$ training points in 3 dimensions, labels equal to a smooth function plus noise of size 0.3. You draw $p$ random ReLU features and fit a linear model on them. When $p < 30$ you use least squares. When $p > 30$ many weight vectors fit the 30 points exactly, so you take the shortest, $w = F^{+}y$, where $F$ is the 30 by $p$ feature matrix and $F^{+}$ its pseudo-inverse. Figure 3 plots the median test error as $p$ sweeps from 5 to 600.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="Test error against the number of features p with 30 training points: low, then a spike at p = 30, then falling again">
<line x1="60" y1="230.0" x2="600" y2="230.0" class="stroke-grid"/>
<text x="54" y="234.0" text-anchor="end" class="fig-small">0.1</text>
<line x1="60" y1="166.7" x2="600" y2="166.7" class="stroke-grid"/>
<text x="54" y="170.7" text-anchor="end" class="fig-small">1</text>
<line x1="60" y1="103.3" x2="600" y2="103.3" class="stroke-grid"/>
<text x="54" y="107.3" text-anchor="end" class="fig-small">10</text>
<line x1="60" y1="40.0" x2="600" y2="40.0" class="stroke-grid"/>
<text x="54" y="44.0" text-anchor="end" class="fig-small">100</text>
<text x="60.0" y="246" text-anchor="middle" class="fig-small">5</text>
<text x="138.2" y="246" text-anchor="middle" class="fig-small">10</text>
<text x="262.1" y="246" text-anchor="middle" class="fig-small">30</text>
<text x="397.9" y="246" text-anchor="middle" class="fig-small">100</text>
<text x="521.8" y="246" text-anchor="middle" class="fig-small">300</text>
<text x="600.0" y="246" text-anchor="middle" class="fig-small">600</text>
<line x1="60" y1="230" x2="600" y2="230" class="stroke-muted"/><line x1="60" y1="40" x2="60" y2="230" class="stroke-muted"/>
<line x1="262.1" y1="40" x2="262.1" y2="230" class="stroke-muted dashed"/>
<polyline points="60.0,182.8 138.2,178.6 183.9,178.5 216.4,166.3 241.5,131.4 254.3,109.5 262.1,82.7 269.4,100.6 279.5,124.8 294.5,145.8 319.7,158.8 340.3,171.0 386.0,177.9 443.6,184.5 521.8,190.8 600.0,185.8" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="60.0" cy="182.8" r="4" class="fill-b"/>
<circle cx="138.2" cy="178.6" r="4" class="fill-b"/>
<circle cx="183.9" cy="178.5" r="4" class="fill-b"/>
<circle cx="216.4" cy="166.3" r="4" class="fill-b"/>
<circle cx="241.5" cy="131.4" r="4" class="fill-b"/>
<circle cx="254.3" cy="109.5" r="4" class="fill-b"/>
<circle cx="262.1" cy="82.7" r="4" class="fill-b"/>
<circle cx="269.4" cy="100.6" r="4" class="fill-b"/>
<circle cx="279.5" cy="124.8" r="4" class="fill-b"/>
<circle cx="294.5" cy="145.8" r="4" class="fill-b"/>
<circle cx="319.7" cy="158.8" r="4" class="fill-b"/>
<circle cx="340.3" cy="171.0" r="4" class="fill-b"/>
<circle cx="386.0" cy="177.9" r="4" class="fill-b"/>
<circle cx="443.6" cy="184.5" r="4" class="fill-b"/>
<circle cx="521.8" cy="190.8" r="4" class="fill-b"/>
<circle cx="600.0" cy="185.8" r="4" class="fill-b"/>
<text x="270.1" y="52" class="fig-bold">p = n = 30</text>
<text x="330" y="266" text-anchor="middle">p, number of random features (log axis)</text>
<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">median test error</text>
</svg>
Figure 3. Median test error of min-norm least squares on p random ReLU features with n = 30 training points. The error spikes at p = n, then falls again as p grows.
```

The curve is not a U. Test error is 0.65 at $p = 15$, rises to 21.2 at $p = 30$, and falls to 0.42 at $p = 300$, below its level at $p = 15$ although 300 weights can fit any 30 points. The VC dimension of the $p$-feature linear family is $p$, so the bound says nothing useful for $p > 30$. It is an upper bound, not a prediction.

### Why the spike sits at p = n

At $p = n$ the feature matrix $F$ is square, and a random square matrix tends to nearly flatten one direction. Take $F = \begin{bmatrix} 1 & 1 \\ 1 & 1.01 \end{bmatrix}$ and labels $y = (1, 1.2)$. Solving $Fw = y$ gives $w = (-19, 20)$. Change the second label by 0.01 and the solution becomes $(-20, 21)$. A change of 0.01 in a label moved the weights by 1, because the solve divides by 0.01. The labels contain noise, the fit must match it exactly, and the noise gets amplified by the reciprocal of the smallest singular value of $F$. That amplification is the spike.

Above $p = n$ the matrix is wide, and its smallest singular value grows as features pile in. The minimum-norm solution

$$
w = F^{\top}\,(F F^{\top})^{-1}\,y
$$

solves the 30 equations with the shortest weight vector. Read it aloud: solve a 30 by 30 system against the labels, then map back to $p$ weights. Extra features let the shortest solution spread small weights across many features instead of huge opposing weights on a few. Below $p = n$ the matching formula is $(F^{\top}F)^{-1}F^{\top}y$. Both are `np.linalg.pinv(F) @ y`.

The peak is a property of one count: the number of weights equals the number of points. A small ridge penalty, from [ridge regression](/lessons/006-ridge-regression), flattens it by stopping the solve from dividing by a tiny number. When the solution is chosen by length, counting weights is the wrong measure of capacity, and the length of $w$ is the better one. [Weight decay](/lessons/035-weight-decay-and-early-stopping) works on that idea.

> [!KEY]
> Test error of min-norm least squares on random features spikes where $p = n$, because the fit must match noisy labels with a nearly singular square matrix. Past that point, more features give the shortest solution more room, and test error falls again.

## Build it step by step

The exercise has one function per idea, in the order the checks test them. The starter gives you `threshold_class`, `interval_class`, `halfplane_class`, and `draw_problem` in full.

> [!RECAP]
> - Hoeffding: $P(|\bar{x} - \mu| > \varepsilon) \le 2e^{-2n\varepsilon^2}$ for $n$ independent values in $[0, 1]$. It holds for every distribution, so it is loose, but its exponent is right.
> - The union bound over $k$ models gives $n = \ln(2k/\delta)/(2\varepsilon^2)$ and $\varepsilon = \sqrt{\ln(2k/\delta)/(2n)}$. The family size enters through a logarithm.
> - A family shatters $m$ points when it makes all $2^m$ label patterns on them. The VC dimension is the largest shattered $m$: 1 for thresholds, 2 for intervals, 3 for half-planes in the plane.
> - The perceptron makes at most $(R/\gamma)^2$ mistakes on separable data.
> - Min-norm least squares has test error that spikes at $p = n$ and falls for $p > n$.

1. **`hoeffding_bound(n, eps)`** returns $\min(1, 2e^{-2n\varepsilon^2})$. The mistake people most often make is `exp(-n * eps**2)`, which drops both 2s and gives 0.368 instead of 0.2707 at $n = 100$, $\varepsilon = 0.1$.
2. **`deviation_frequency(n, eps, trials, seed)`** draws `rng.binomial(n, p, size=trials) / n` and returns the fraction more than `eps` from `p`. The check confirms the bound holds and is at least 4 times too large.
3. **`union_bound_sample_size(k, eps, delta)`** and **`union_bound_eps(k, n, delta)`** return the two formulas above. The mistake people most often make is `log(2 / delta)`, which drops $k$ and returns 738 for every family.
4. **`num_dichotomies(points, hypotheses)`** and **`shatters(points, hypotheses)`** put `tuple(h(points))` for every rule in a set and compare its size with $2^m$.
5. **`largest_shattered(candidates, hypotheses)`** tries `itertools.combinations` of $m = 1, 2, \ldots$ rows and returns the last $m$ that had a shattered subset.
6. **`perceptron(X, y, max_epochs)`** returns the weights and the number of updates. Test `y_i * (w @ x_i) <= 0`, not `< 0`, because the first step starts at exactly 0.
7. **`perceptron_mistake_bound(X, y, w_star)`** returns $(R/\gamma)^2$. Divide the margin by the norm of `w_star`, or it depends on the scale of `w_star`.
8. **`relu_features(X, W, b)`** returns `np.maximum(X @ W + b, 0)`.
9. **`min_norm_fit(F, y)`** returns `np.linalg.pinv(F) @ y`. The normal equations fail for $p > n$, because `F.T @ F` is singular.
10. **`min_norm_test_error(n, p, trials, seed)`** loops over `draw_problem`, fits, scores against `f_test`, adds `noise**2`, and returns the median.

## Where this breaks

- **The bound is not a forecast.** Constants in the proofs make absolute numbers pessimistic. Use a bound for scaling, such as how data needs change when $d$ doubles.
- **VC dimension is not the number of weights.** The one-parameter family $\operatorname{sign}(\sin(ax))$ shatters any number of points, so its VC dimension is infinite. Weight count is a proxy that works for linear models.
- **Deep networks.** A network can have far more weights than examples and still generalise, so bounds that depend on weight count are empty there.

## The exercise

You implement the twelve functions named above. The starter's closing program prints a bound, a VC dimension, a perceptron run, and test errors at three values of $p$.

The checks confirm the bound and its cap at 1, a simulated frequency near 0.035, and the sample sizes 738, 1199, and 2120. They confirm VC dimensions 1, 2, and 3, 3 perceptron mistakes on the chapter's four points, and no violation of the bound on 10 separable problems. They confirm that `min_norm_fit` interpolates for $p > n$ with the shortest weights, and that test error spikes at $p = n$.

## Further reading

- Belkin, Hsu, Ma, and Mandal, "Reconciling modern machine learning practice and the bias-variance trade-off," arXiv:1812.11118. Published in the Proceedings of the National Academy of Sciences, 2019.
- Shalev-Shwartz and Ben-David, *Understanding Machine Learning: From Theory to Algorithms*, chapters 2 to 6. Covers the finite-class bound, uniform convergence, and VC dimension with full proofs.
- Stanford CS229 lecture notes on learning theory (bias and variance, uniform convergence, and VC dimension), at cs229.stanford.edu.
- MIT 6.S898 (Deep Learning, Fall 2023), the session "Bayesian analysis of learning and generalisation," at phillipi.github.io/6.s898.
