---
title: "Uncertainty: ensembles, MC dropout, and conformal prediction"
slug: 044-uncertainty-and-conformal-prediction
part: "Part 4 — Neural networks"
summary: Say how sure a prediction is. Read the spread of an ensemble, keep dropout on at test time, and wrap any model in an interval with a coverage guarantee.
minutes: 75
gpu: false
objectives:
  - Separate noise in the data from not knowing the model, and say which one more data can shrink.
  - Build a deep ensemble and read its variance, which grows away from the training range.
  - Run MC dropout as a cheap ensemble, and name what it cannot tell you.
  - Compute the split conformal quantile with the (n + 1) correction, and state what marginal coverage promises and what it does not.
  - Build conformal prediction sets for a classifier from the score 1 - p of the true class.
prereqs:
  - 011-likelihood-map-and-bayesian-regression
  - 016-scoring-classifiers
  - 026-averaging-and-bagging
  - 042-dropout-and-label-smoothing
  - 043-debugging-and-tuning
exercise: 044-uncertainty-and-conformal-prediction
---

# Uncertainty: ensembles, MC dropout, and conformal prediction

A network predicts that a house will sell for 310 thousand. A buyer asks the obvious next question: how sure are you? A single number cannot answer it. Maybe the model has seen hundreds of similar houses and 310 is a safe guess. Maybe the house is a converted lighthouse, nothing like the training data, and 310 is a shrug that the model printed with the same confidence as everything else.

Without an answer, you cannot decide when to trust the model or when to ask a person. A network trained with the loss from earlier chapters returns a point, and the point looks equally firm everywhere.

This chapter gives three ways to attach a statement of doubt to a prediction. An **ensemble** trains several models and reads how much they disagree. **MC dropout** gets a cheap ensemble from one network by keeping dropout on at test time. **Conformal prediction** wraps any model, however it was trained, in an interval that contains the true answer a stated share of the time. Each one promises something different, and knowing which promise you get is the point of the chapter.

## What you already know

- A model's error has a part you can shrink with a better model and a part you cannot ([bias and variance](/lessons/002-curves-and-bias-variance)).
- A Bayesian predictive variance has two terms, the noise and a term that grows away from the data ([the Bayesian view](/lessons/011-likelihood-map-and-bayesian-regression)).
- Averaging models that disagree cuts variance ([averaging and bagging](/lessons/026-averaging-and-bagging)).
- Inverted dropout draws a keep-mask and divides the survivors by $1 - p$ ([dropout](/lessons/042-dropout-and-label-smoothing)).
- A model is calibrated when the cases it scores at 0.8 are right about 80 percent of the time ([scoring classifiers](/lessons/016-scoring-classifiers)).

## Two kinds of not knowing

Two different things can make a prediction at one size uncertain.

The first is **noise in the data**. Two identical houses sell for different prices, because one buyer was in a hurry. No model can predict that difference, because it is not in the inputs. Collecting a million more sales does not shrink it. People call this **aleatoric** uncertainty, from the Latin for dice.

The second is **not knowing the model**. Where you have few or no sales, many different curves fit the data you have equally well, and they disagree about what happens in between and beyond. More data in that region shrinks the disagreement. People call this **epistemic** uncertainty, from the Greek for knowledge.

You met both terms already in the Bayesian predictive variance, which was the noise variance plus a term that grows away from the houses you sold. A neural network has no closed-form posterior over weights, so the methods here use tools you have: train several models, switch units off at random, or hold some data back.

> [!KEY]
> Noise is the part of the doubt that more data cannot remove. Not knowing the model is the part that more data removes, and it is largest where the training data is thin. An ensemble and MC dropout estimate the second part. Conformal prediction covers both at once, without separating them.

## Disagreement as a signal

Take a real model and see what the second kind looks like. Figure 1 fits the same 30 noisy points on $x$ between $-2$ and $2$ with eight models that differ only in their random seed. Each model is a ridge regression on 60 random tanh features, so each one starts from different features and ends at a different curve.

```figure
<svg viewBox="0 0 560 236" role="img" aria-label="Eight fitted curves agree inside the training range from -2 to 2 and fan out sharply outside it. An orange band for the ensemble mean plus or minus two standard deviations is thin inside the range and wide outside it.">
<rect x="160" y="6" width="240" height="208" class="fill-soft-a"/>
<line x1="40" y1="110" x2="520" y2="110" class="stroke-grid"/>
<polygon points="40,10 48,12 55,15 62,18 70,21 78,25 85,29 92,33 100,38 108,43 115,49 122,55 130,61 138,68 145,74 152,82 160,89 168,96 175,102 182,109 190,115 198,120 205,124 212,127 220,129 228,129 235,129 242,127 250,125 258,121 265,115 272,107 280,101 288,99 295,97 302,95 310,93 318,91 325,89 332,89 340,90 348,92 355,96 362,100 370,104 378,109 385,114 392,119 400,124 408,128 415,132 422,135 430,138 438,141 445,143 452,145 460,146 468,148 475,149 482,150 490,150 498,151 505,152 512,152 520,153 520,195 512,192 505,189 498,186 490,182 482,179 475,175 468,171 460,166 452,162 445,157 438,152 430,147 422,142 415,136 408,131 400,125 392,120 385,115 378,110 370,105 362,100 355,96 348,93 340,91 332,90 325,90 318,91 310,94 302,96 295,98 288,100 280,105 272,111 265,117 258,122 250,126 242,128 235,129 228,130 220,129 212,128 205,125 198,121 190,116 182,110 175,103 168,96 160,90 152,84 145,79 138,74 130,70 122,66 115,63 108,60 100,58 92,56 85,54 78,52 70,51 62,50 55,48 48,47 40,47" class="fill-soft-b"/>
<polyline points="40,29 48,30 55,32 62,34 70,36 78,39 85,42 92,45 100,48 108,52 115,56 122,61 130,65 138,71 145,77 152,83 160,89 168,96 175,103 182,110 190,116 198,121 205,125 212,128 220,129 228,129 235,129 242,127 250,125 258,122 265,117 272,111 280,105 288,100 295,97 302,95 310,93 318,91 325,90 332,89 340,90 348,93 355,96 362,100 370,105 378,109 385,115 392,120 400,125 408,129 415,134 422,138 430,142 438,145 445,148 452,151 460,153 468,155 475,157 482,159 490,160 498,162 505,163 512,164 520,165" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,8 48,11 55,14 62,17 70,21 78,25 85,29 92,34 100,39 108,44 115,50 122,56 130,62 138,69 145,75 152,82 160,89 168,96 175,103 182,109 190,115 198,120 205,124 212,127 220,129 228,130 235,129 242,128 250,125 258,122 265,116 272,110 280,104 288,100 295,97 302,96 310,93 318,91 325,89 332,89 340,90 348,93 355,96 362,100 370,104 378,109 385,114 392,120 400,125 408,130 415,135 422,140 430,145 438,149 445,154 452,158 460,162 468,166 475,170 482,174 490,177 498,181 505,184 512,187 520,189" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,29 48,31 55,33 62,35 70,37 78,40 85,42 92,45 100,49 108,52 115,57 122,61 130,66 138,71 145,77 152,83 160,89 168,96 175,103 182,109 190,116 198,121 205,125 212,128 220,129 228,129 235,129 242,128 250,126 258,122 265,115 272,109 280,104 288,100 295,97 302,95 310,93 318,91 325,89 332,89 340,90 348,93 355,96 362,100 370,105 378,109 385,114 392,120 400,125 408,129 415,134 422,138 430,142 438,146 445,149 452,152 460,155 468,158 475,161 482,163 490,165 498,167 505,169 512,171 520,173" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,20 48,21 55,24 62,26 70,28 78,31 85,35 92,38 100,42 108,46 115,51 122,57 130,62 138,69 145,75 152,82 160,89 168,96 175,103 182,110 190,116 198,121 205,125 212,127 220,129 228,129 235,129 242,128 250,126 258,122 265,115 272,108 280,102 288,99 295,98 302,96 310,93 318,91 325,90 332,90 340,91 348,93 355,96 362,100 370,104 378,109 385,114 392,120 400,125 408,131 415,136 422,142 430,147 438,152 445,156 452,161 460,165 468,169 475,172 482,175 490,178 498,181 505,183 512,186 520,188" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,35 48,37 55,39 62,41 70,43 78,46 85,49 92,51 100,54 108,58 115,61 122,65 130,69 138,74 145,79 152,84 160,90 168,96 175,102 182,109 190,115 198,120 205,125 212,128 220,129 228,130 235,129 242,128 250,126 258,122 265,115 272,108 280,103 288,100 295,98 302,96 310,93 318,91 325,89 332,89 340,90 348,93 355,96 362,100 370,105 378,110 385,114 392,119 400,124 408,129 415,134 422,138 430,143 438,147 445,151 452,155 460,159 468,162 475,166 482,169 490,172 498,175 505,177 512,180 520,182" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,36 48,37 55,39 62,41 70,43 78,45 85,47 92,50 100,53 108,56 115,59 122,63 130,68 138,72 145,78 152,83 160,90 168,96 175,103 182,109 190,116 198,121 205,125 212,128 220,129 228,129 235,129 242,128 250,126 258,122 265,116 272,109 280,104 288,100 295,97 302,95 310,93 318,91 325,90 332,89 340,90 348,93 355,96 362,100 370,105 378,110 385,115 392,120 400,124 408,129 415,133 422,137 430,140 438,143 445,146 452,148 460,150 468,151 475,153 482,154 490,156 498,157 505,158 512,159 520,159" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,32 48,33 55,35 62,36 70,38 78,40 85,43 92,45 100,49 108,52 115,56 122,60 130,65 138,71 145,76 152,83 160,89 168,96 175,103 182,110 190,116 198,121 205,125 212,128 220,129 228,129 235,129 242,128 250,126 258,122 265,116 272,109 280,103 288,99 295,98 302,96 310,94 318,91 325,89 332,89 340,90 348,93 355,96 362,100 370,105 378,110 385,115 392,120 400,124 408,129 415,133 422,137 430,140 438,143 445,146 452,149 460,151 468,153 475,156 482,157 490,159 498,161 505,163 512,164 520,165" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,37 48,38 55,39 62,41 70,42 78,44 85,46 92,49 100,51 108,54 115,58 122,62 130,66 138,71 145,77 152,83 160,89 168,96 175,103 182,110 190,116 198,121 205,125 212,128 220,129 228,129 235,129 242,127 250,125 258,122 265,116 272,110 280,103 288,99 295,97 302,96 310,94 318,91 325,89 332,89 340,90 348,92 355,96 362,100 370,105 378,110 385,115 392,119 400,124 408,129 415,133 422,137 430,141 438,145 445,148 452,151 460,154 468,157 475,159 482,161 490,163 498,165 505,166 512,168 520,169" fill="none" class="stroke-muted" stroke-width="0.8"/>
<polyline points="40,28 48,30 55,32 62,34 70,36 78,39 85,42 92,45 100,48 108,52 115,56 122,61 130,66 138,71 145,77 152,83 160,89 168,96 175,103 182,109 190,116 198,121 205,125 212,128 220,129 228,129 235,129 242,128 250,126 258,122 265,116 272,109 280,103 288,100 295,97 302,96 310,93 318,91 325,90 332,89 340,90 348,93 355,96 362,100 370,105 378,109 385,114 392,120 400,125 408,129 415,134 422,138 430,143 438,146 445,150 452,153 460,156 468,159 475,162 482,164 490,166 498,168 505,170 512,172 520,174" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="313" cy="94" r="3.2" class="fill-a"/>
<circle cx="225" cy="132" r="3.2" class="fill-a"/>
<circle cx="170" cy="99" r="3.2" class="fill-a"/>
<circle cx="164" cy="94" r="3.2" class="fill-a"/>
<circle cx="355" cy="96" r="3.2" class="fill-a"/>
<circle cx="379" cy="112" r="3.2" class="fill-a"/>
<circle cx="306" cy="96" r="3.2" class="fill-a"/>
<circle cx="335" cy="89" r="3.2" class="fill-a"/>
<circle cx="290" cy="100" r="3.2" class="fill-a"/>
<circle cx="384" cy="112" r="3.2" class="fill-a"/>
<circle cx="356" cy="101" r="3.2" class="fill-a"/>
<circle cx="161" cy="89" r="3.2" class="fill-a"/>
<circle cx="366" cy="99" r="3.2" class="fill-a"/>
<circle cx="168" cy="95" r="3.2" class="fill-a"/>
<circle cx="335" cy="88" r="3.2" class="fill-a"/>
<circle cx="202" cy="122" r="3.2" class="fill-a"/>
<circle cx="367" cy="100" r="3.2" class="fill-a"/>
<circle cx="290" cy="96" r="3.2" class="fill-a"/>
<circle cx="232" cy="126" r="3.2" class="fill-a"/>
<circle cx="261" cy="118" r="3.2" class="fill-a"/>
<circle cx="167" cy="96" r="3.2" class="fill-a"/>
<circle cx="190" cy="117" r="3.2" class="fill-a"/>
<circle cx="321" cy="88" r="3.2" class="fill-a"/>
<circle cx="315" cy="88" r="3.2" class="fill-a"/>
<circle cx="308" cy="97" r="3.2" class="fill-a"/>
<circle cx="252" cy="126" r="3.2" class="fill-a"/>
<circle cx="399" cy="125" r="3.2" class="fill-a"/>
<circle cx="395" cy="122" r="3.2" class="fill-a"/>
<circle cx="325" cy="92" r="3.2" class="fill-a"/>
<circle cx="316" cy="92" r="3.2" class="fill-a"/>
<text x="280" y="22" text-anchor="middle" class="fig-small">training range</text>
<text x="40" y="230" text-anchor="middle" class="fig-small">-4</text>
<text x="160" y="230" text-anchor="middle" class="fig-small">-2</text>
<text x="280" y="230" text-anchor="middle" class="fig-small">0</text>
<text x="400" y="230" text-anchor="middle" class="fig-small">2</text>
<text x="520" y="230" text-anchor="middle" class="fig-small">4</text>
<text x="530" y="230" class="fig-small">x</text>
</svg>
Figure 1. Eight models fitted to the same 30 blue points (grey curves), with their mean (orange line) and mean plus or minus two standard deviations (orange band). Inside the shaded training range the standard deviation across models is at most about 0.04. At $x = \pm 4$ it is about 0.4 to 0.5, ten times larger or more. The curves know where the data is.
```

Inside the shaded range, the data pins all eight curves down. Outside it, nothing does, each model extrapolates in its own way, and the curves fan out. The disagreement is a free signal about where the data is thin.

## Deep ensembles

A **deep ensemble** is the plain version of this idea. Train $M$ networks with the same recipe and different random seeds, which changes the initial weights and the order of the batches. At a point $x$, call model $m$'s prediction $f_m(x)$. The ensemble reports the mean and the variance of those predictions:

$$
\bar f(x) = \frac{1}{M}\sum_{m=1}^{M} f_m(x), \qquad
v(x) = \frac{1}{M}\sum_{m=1}^{M}\bigl(f_m(x) - \bar f(x)\bigr)^2 .
$$

Read the first as "the average prediction" and the second as "the average squared distance from that average". In numpy, with the predictions stacked into an array `preds` of shape `(M, n)`, these are `preds.mean(axis=0)` and `preds.var(axis=0)`.

A tiny example with numbers. Three models predict a house at size 1, which is inside the data, and at size 4, which is outside it:

| Size | Model 1 | Model 2 | Model 3 | Mean | Variance |
|---|---|---|---|---|---|
| 1 | 0.50 | 0.52 | 0.48 | 0.50 | $(0 + 0.0004 + 0.0004)/3 \approx 0.0003$ |
| 4 | 1.00 | 1.40 | 0.60 | 1.00 | $(0 + 0.16 + 0.16)/3 \approx 0.107$ |

The means are 0.50 and 1.00, and the variances differ by a factor of 400. The model has a firm answer at size 1 and a loose one at size 4, and the ensemble says so without being told where the data is.

This is the averaging you know from [bagging](/lessons/026-averaging-and-bagging), with one difference. Bagging shows each member a different bootstrap sample. A deep ensemble usually shows every member the same data and relies on random initialisation to make them differ. That is enough for neural networks, because their loss surfaces have many good minima, and different seeds land in different ones. The mean is also more accurate than a typical member, and the spread is the bonus.

> [!WARNING]
> The variance above measures only disagreement, the "not knowing the model" part. It does not include the noise in the data. If you want a predictive variance that covers both, add an estimate of the noise, for example the mean squared residual of the ensemble mean on held-out data. An ensemble whose members all overfit in the same way agrees confidently and wrongly.

## MC dropout: a cheap ensemble

Training $M$ networks costs $M$ times as much. **MC dropout** gets many different predictions out of one network. Recall from [dropout](/lessons/042-dropout-and-label-smoothing) that training with random masks trains a crowd of thinned networks that share weights, and that at test time you turn the masks off and use the full layer as an approximation to their average. MC dropout leaves the masks on. Each forward pass then uses a different thinned network, and the passes behave like members of an ensemble. "MC" stands for Monte Carlo, which here means estimating an average by drawing random samples.

The recipe is short. Draw $n$ forward passes with dropout on, each with a fresh mask. Report the mean and the variance of the $n$ outputs, the same two formulas as before with the passes in place of the models.

A tiny example. Four hidden units contribute $c = [1, 2, 3, 4]$ to an output, so with no dropout the output is $10$. With $p = 0.5$, each unit is kept with probability one half and, when kept, doubled by the $1/(1-p)$ scaling. Three passes might draw the masks `1010`, `0110`, and `1111`:

| Pass | Mask | Output |
|---|---|---|
| 1 | `1010` | $(1 + 3) \cdot 2 = 8$ |
| 2 | `0110` | $(2 + 3) \cdot 2 = 10$ |
| 3 | `1111` | $(1 + 2 + 3 + 4) \cdot 2 = 20$ |

The three outputs average to 12.7, and three passes are too few to be steady. In practice you draw 30 to 100. In the long run, the mean is exactly $10$, because the scaling makes each unit's expected contribution equal to its full contribution. Each unit's contribution has variance $c_i^2 \, p/(1-p)$, and the units are independent, so the output has variance $(1^2 + 2^2 + 3^2 + 4^2) \cdot 1 = 30$. The test in the exercise uses these numbers.

What are the limits? Gal and Ghahramani showed that this procedure approximates a Bayesian posterior under particular assumptions. Treat that as a claim about a modelling choice, not a guarantee about your network:

- **The spread comes from $p$, not from the data.** The variance of 30 above depends on $p$ and the contributions, not on how much data you had. Change $p$ from 0.5 to 0.1 and the variance falls by a factor of 9, although nothing about the world changed.
- **The members are correlated.** All passes share one set of weights, so they cannot disagree about things that the one network is confidently wrong about. A deep ensemble of separately trained networks explores different minima. One network with dropout explores variations around one minimum, and tends to give narrower spreads.
- **It does not have to grow away from the data.** Figure 1 shows a spread that rises outside the training range because the members were fitted separately. A single network with dropout can extrapolate with a small spread there.

MC dropout is cheap and useful as a flag. Treat its variance as a relative score, not as a calibrated measure.

## From a spread to a guarantee

An ensemble gives you a number such as "standard deviation 0.4". What does the reader do with it? One habit is to report $\bar f \pm 2\sqrt{v}$ and call it a 95 percent interval. That is a guess dressed as a statement, because the spread can come out wrong: too narrow when members share mistakes, and unrelated to the real error when it comes from $p$.

**Conformal prediction** asks for less and delivers a promise. You do not need the model's spread to be right. You need a way to measure how wrong the model is on data it did not train on, and then you use that measurement to size the interval. The recipe has three steps.

1. Split the labelled data. Train the model on one part, and keep a **calibration set** of $n$ points that the model never saw.
2. On each calibration point, compute a **score**, a number that says how badly the model missed. For regression, use the absolute residual $|y_i - \hat f(x_i)|$.
3. Take a high quantile $q$ of those scores. For any unseen $x$, report the interval $\hat f(x) \pm q$.

The reasoning is a fairness argument. Suppose the $n$ calibration points and the unseen point are drawn the same way, so that none of the $n + 1$ is special. This property is called **exchangeability**, and independent draws from one distribution have it. Then the unseen point's score is equally likely to land in any rank among the $n + 1$ scores, from smallest to largest. If you take $q$ as the $k$-th smallest calibration score, the unseen score is at or below $q$ whenever its rank is $k$ or lower among the $n + 1$, which happens with probability $k/(n + 1)$. To make that at least $1 - \alpha$, choose

$$
k = \lceil (n + 1)(1 - \alpha) \rceil, \qquad q = \text{the } k\text{-th smallest of the } n \text{ scores}.
$$

Here $\alpha$ is the miss rate you accept, so $\alpha = 0.1$ asks for 90 percent coverage, and the brackets $\lceil\cdot\rceil$ mean "round up". If $k$ exceeds $n$, no calibration score is large enough, and the honest answer is an infinite interval.

A worked example with real numbers. Nine calibration residuals, sorted, are $0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.1, 1.5$. You want $\alpha = 0.2$.

- $n = 9$, so $(n + 1)(1 - \alpha) = 10 \times 0.8 = 8$, and $k = 8$.
- The 8th smallest score is $q = 1.1$.
- The model predicts $3.0$ for an unseen house, so the interval is $[1.9, 4.1]$.

Check the reasoning. Ten scores, nine calibration and one unseen, fall in a random order. The unseen score is above the 8th smallest calibration score only when it ranks 9th or 10th, which has probability $2/10 = 0.2$. The coverage is $8/10 = 0.8$, the target.

### Why the (n + 1) matters

The tempting shortcut takes the quantile at level $1 - \alpha$ directly, which uses $k = \lceil n(1 - \alpha) \rceil$ and leaves out the extra point. With $n = 10$ and $\alpha = 0.1$, the shortcut picks $k = 9$ and covers $9/11 \approx 0.82$, while the correct $k = 10$ covers $10/11 \approx 0.91$. The unseen point is a draw too, and it can land above every calibration score. The $(n + 1)$ accounts for that point. For $n = 1000$ the gap is a tenth of a percentage point, and for $n = 10$ it is nine.

Figure 2 repeats the experiment 200 times, with fresh calibration sets of 10 points and 500 test points, a fixed predictor, and heavy-tailed noise. Both methods vary a lot from split to split, because 10 points is a small calibration set.

```figure
<svg viewBox="0 0 560 236" role="img" aria-label="Histograms of coverage over 200 splits with 10 calibration points. With the correction, coverage piles up at or above 0.90. Without it, coverage is spread lower, with a mean near 0.83.">
<rect x="50" y="190" width="17" height="0" class="fill-a"/>
<rect x="68" y="190" width="17" height="0" class="fill-bad"/>
<text x="50" y="208" class="fig-small" text-anchor="middle">0.40</text>
<rect x="90" y="190" width="17" height="0" class="fill-a"/>
<rect x="108" y="187" width="17" height="3" class="fill-bad"/>
<text x="90" y="208" class="fig-small" text-anchor="middle">0.45</text>
<rect x="130" y="190" width="17" height="0" class="fill-a"/>
<rect x="148" y="188" width="17" height="2" class="fill-bad"/>
<text x="130" y="208" class="fig-small" text-anchor="middle">0.50</text>
<rect x="170" y="188" width="17" height="2" class="fill-a"/>
<rect x="188" y="184" width="17" height="6" class="fill-bad"/>
<text x="170" y="208" class="fig-small" text-anchor="middle">0.55</text>
<rect x="210" y="190" width="17" height="0" class="fill-a"/>
<rect x="228" y="182" width="17" height="8" class="fill-bad"/>
<text x="210" y="208" class="fig-small" text-anchor="middle">0.60</text>
<rect x="250" y="184" width="17" height="6" class="fill-a"/>
<rect x="268" y="160" width="17" height="30" class="fill-bad"/>
<text x="250" y="208" class="fig-small" text-anchor="middle">0.65</text>
<rect x="290" y="179" width="17" height="11" class="fill-a"/>
<rect x="308" y="161" width="17" height="29" class="fill-bad"/>
<text x="290" y="208" class="fig-small" text-anchor="middle">0.70</text>
<rect x="330" y="177" width="17" height="13" class="fill-a"/>
<rect x="348" y="153" width="17" height="37" class="fill-bad"/>
<text x="330" y="208" class="fig-small" text-anchor="middle">0.75</text>
<rect x="370" y="152" width="17" height="38" class="fill-a"/>
<rect x="388" y="153" width="17" height="37" class="fill-bad"/>
<text x="370" y="208" class="fig-small" text-anchor="middle">0.80</text>
<rect x="410" y="153" width="17" height="37" class="fill-a"/>
<rect x="428" y="113" width="17" height="77" class="fill-bad"/>
<text x="410" y="208" class="fig-small" text-anchor="middle">0.85</text>
<rect x="450" y="108" width="17" height="82" class="fill-a"/>
<rect x="468" y="126" width="17" height="64" class="fill-bad"/>
<text x="450" y="208" class="fig-small" text-anchor="middle">0.90</text>
<rect x="490" y="59" width="17" height="131" class="fill-a"/>
<rect x="508" y="163" width="17" height="27" class="fill-bad"/>
<text x="490" y="208" class="fig-small" text-anchor="middle">0.95</text>
<text x="530" y="208" class="fig-small" text-anchor="middle">1.00</text>
<line x1="450" y1="30" x2="450" y2="190" class="stroke-ink dashed"/>
<text x="455" y="42" class="fig-small">goal 0.90</text>
<line x1="50" y1="190" x2="530" y2="190" class="stroke-muted"/>
<rect x="60" y="14" width="12" height="12" class="fill-a"/><text x="78" y="25" class="fig-small">with the (n + 1) correction</text>
<rect x="60" y="32" width="12" height="12" class="fill-bad"/><text x="78" y="43" class="fig-small">at level 1 - alpha, no correction</text>
<text x="290" y="226" text-anchor="middle" class="fig-small">coverage of one split on 500 test points</text>
</svg>
Figure 2. Coverage of 200 seeded splits at $n = 10$ and $\alpha = 0.1$. With the $(n + 1)$ correction (blue), the average coverage is 0.911, close to the exact $10/11$. Without it (red), the average is 0.826, close to $9/11$, well under the goal. Any single split can fall short, because the guarantee is about the average over splits.
```

> [!KEY]
> Split conformal prediction guarantees **marginal coverage**: averaged over the random draw of the calibration set and the unseen point, the true value lands inside the interval with probability at least $1 - \alpha$. The guarantee holds for any model and any data distribution, and needs only exchangeability. It is also at most $1 - \alpha + 1/(n + 1)$, so the interval is not wastefully wide.

### What the guarantee does not say

The guarantee is narrower than it sounds, and these are the limits worth remembering.

- **It is marginal, not conditional.** The 90 percent is an average over all inputs. The interval has the same half-width $q$ at every $x$. If the noise is larger for some inputs, such as the expensive houses, the interval covers them less than 90 percent of the time and the cheap houses more. Covering each kind of input at 90 percent is a harder problem, and split conformal prediction does not solve it.
- **It needs exchangeability.** If the unseen inputs come from a different distribution than the calibration data, say houses from another city or last decade's prices, the guarantee is gone. The intervals can undercover badly, and nothing in the output warns you.
- **The model must be fixed before the calibration data is used.** Tuning on the calibration set spoils the scores.

## Prediction sets for a classifier

Conformal prediction is not only for regression. For a classifier, the **score** of a calibration example is one minus the probability the model gave to the true class:

$$
s_i = 1 - p_{i, y_i}.
$$

Read it as "how surprised was the model by the truth". A score near 0 means the model gave the true class almost all its probability. A score near 1 means it gave the true class almost none. Compute $q$ from these scores with the same rule, $k = \lceil (n + 1)(1 - \alpha) \rceil$. The **prediction set** for an unseen input holds every class whose probability is high enough that its score would have been at or below $q$:

$$
C(x) = \{\, c : 1 - p_c(x) \le q \,\}.
$$

A worked example with $n = 9$ calibration examples whose true classes had probabilities $0.9, 0.8, 0.95, 0.6, 0.7, 0.5, 0.85, 0.4, 0.99$. The scores are $0.1, 0.2, 0.05, 0.4, 0.3, 0.5, 0.15, 0.6, 0.01$, and sorted they are $0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6$. For $\alpha = 0.1$, $k = \lceil 10 \times 0.9 \rceil = 9$, so $q = 0.6$ and a class enters the set when its probability is at least $0.4$.

- A confident input with probabilities $[0.7, 0.2, 0.1]$ gives the set $\{0\}$.
- An unsure input with $[0.45, 0.40, 0.15]$ gives $\{0, 1\}$, two classes, because both pass the bar.

The set grows when the model is unsure and shrinks when it is confident, and it contains the true class at least $1 - \alpha$ of the time on average. Set size is the useful signal. A reader who sees a set of one class acts on it, and a reader who sees a set of five sends the case to a person.

The same marginal caveat applies. The guarantee averages over inputs, so the sets may cover easy classes more than 90 percent of the time and hard ones less. Conformal coverage does not need a [calibrated](/lessons/016-scoring-classifiers) model, but a better-calibrated one makes smaller sets for the same coverage.

## Cost and scaling

An ensemble of $M$ networks costs $M$ times the training, memory, and prediction work. MC dropout costs no extra training and $n$ forward passes per prediction. Conformal prediction costs one pass over the calibration set and a sort. Its real price is data: with 1000 labelled points, holding back 200 leaves 800 to fit.

## Where this breaks

- **Members that agree for the wrong reason.** Networks trained on the same data with the same recipe share blind spots, which give a small spread and a wrong answer.
- **A calibration set that does not match the deployment data.** The guarantee holds for exchangeable data only. When the data drifts, recompute the quantile on recent labelled data, and watch the empirical coverage on a stream of labelled cases.
- **A tiny calibration set.** With $n < 1/\alpha - 1$, the index $k$ exceeds $n$ and the interval is infinite. For $\alpha = 0.1$ you need at least 9 calibration points to get a finite interval.
- **Intervals that are the same width everywhere.** If the noise varies with the input, divide each score by a spread estimate such as the ensemble standard deviation, so the intervals widen where the noise is larger.

> [!RECAP]
> - There are two kinds of doubt: noise in the data, which more data does not shrink, and not knowing the model, which more data does shrink.
> - An ensemble of independently trained models reports the mean and the variance of their predictions, and the variance grows where the training data is thin.
> - MC dropout keeps dropout on at test time and averages many passes. It is cheap, but its spread is set by $p$ and by one network's weights, so treat it as a relative score.
> - Split conformal prediction takes the $k$-th smallest calibration score with $k = \lceil (n + 1)(1 - \alpha) \rceil$ and promises marginal coverage of at least $1 - \alpha$ under exchangeability, not coverage for each input and not under distribution shift.
> - For a classifier, the score is $1 - p$ of the true class, and the prediction set holds every class with $1 - p_c \le q$.

## Build it step by step

1. **`ensemble_predict(models, X)`** takes a list of callables, each mapping an array `X` of shape `(n,)` to predictions of shape `(n,)`, and returns the mean and the variance across models, both of shape `(n,)`. Stack the predictions into an array of shape `(M, n)` and reduce over axis 0. The variance divides by $M$, as in the formula. The mistake people most often make is to reduce over axis 1, which averages over inputs instead of over models.
2. **`mc_dropout_predict(forward, X, p, n, rng)`** calls `forward(X, p, rng)` exactly `n` times, where each call is one stochastic pass with a fresh mask, and returns the mean and variance over the passes, each of shape `(len(X),)`. Pass the same `rng` object to every call. The mistake people most often make is to create a fresh generator with the same seed inside the loop, which repeats one mask `n` times and reports a variance of zero.
3. **`conformal_quantile(scores, alpha)`** takes a one-dimensional array of calibration scores and returns the $k$-th smallest with $k = \lceil (n + 1)(1 - \alpha) \rceil$. Sort the scores and index with $k - 1$. Return `np.inf` when $k > n$. The mistake people most often make is `np.quantile(scores, 1 - alpha)`, which leaves out the $(n + 1)$ and undercovers on small calibration sets.
4. **`conformal_interval(pred_cal, y_cal, pred_test, alpha)`** takes the model's predictions and the labels on the calibration set, plus the predictions on the test inputs. It scores each calibration point by the absolute residual, calls `conformal_quantile`, and returns the lower and upper ends, `pred_test - q` and `pred_test + q`.
5. **`conformal_sets(probs_cal, y_cal, probs_test, alpha)`** takes class probabilities of shape `(n, K)` and integer labels of shape `(n,)`, plus test probabilities of shape `(m, K)`. Score each calibration example by `1 - probs_cal[i, y_cal[i]]`, which needs fancy indexing with `np.arange(n)`, and return a boolean array of shape `(m, K)` that is true where `1 - probs_test <= q`. The mistake people most often make is to score with the largest probability instead of the true class's probability.

## The exercise

You implement `ensemble_predict`, `mc_dropout_predict`, `conformal_quantile`, `conformal_interval`, and `conformal_sets`. The checks start with the ensemble mean and variance on the three-model table, then confirm that seeded random-feature models disagree more far outside the training range than inside it. They compare MC dropout with the exact mean of 10 and variance of 30 from the four-unit example, and `conformal_quantile` with the nine-score example. Then they measure coverage over 200 seeded splits with $n = 10$, where the corrected interval lands near 0.91 and the version without the $(n + 1)$ lands near 0.82. The last check builds classifier sets and measures their average coverage.

## Further reading

- Anastasios N. Angelopoulos and Stephen Bates, [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511), 2021. The source for split conformal prediction, the $(n + 1)$ correction, and classification sets.
- Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, 2017. The deep ensemble recipe and its comparison with Bayesian methods.
- Yarin Gal and Zoubin Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, 2016. The argument for MC dropout, and its assumptions.
