---
title: "Scoring classifiers: precision, recall, ROC, and calibration"
slug: 014-scoring-classifiers
part: "Part 2 — Classical models"
summary: "Count misses and false alarms, trace an ROC curve, and rescale an overconfident probability."
minutes: 75
gpu: false
objectives:
  - Build a confusion matrix and compute precision, recall, and F1, including the case with no predicted defaults.
  - Trace an ROC curve by sweeping a threshold, and read the area as the chance a default outranks a repayment.
  - Choose a probability cut from the cost of a false alarm and the cost of a miss.
  - Measure calibration with a reliability table and expected calibration error, and fit a temperature that lowers the log loss.
prereqs:
  - 008-sigmoid-and-log-loss
  - 009-fitting-logistic-regression
  - 013-learning-curves-and-winners-curse
exercise: 014-scoring-classifiers
---

# Scoring classifiers: precision, recall, ROC, and calibration

A lender looks at 100 loan applications. One borrower defaults, and the other 99 repay. A model that calls every loan repaid is right 99 times. Its accuracy is 0.99. You would not use it. The one loan that matters is the one it missed, and accuracy counts that miss as one error among a hundred.

A second model flags 10 loans and catches that default, together with 9 false alarms. It is right on the other 90, so its accuracy is 0.91. Accuracy says the first model is the better one. The desk, which loses the loan when a default slips through, says the second model is the one that did the job.

This chapter replaces that single number. You count the four ways a call can land, you sweep the cut that turns a probability into a yes or no, and you check that a probability of 0.8 is a default about 8 times in 10.

## What you already know

- A logistic model turns a score $z$ into a probability $\sigma(z)$, and the log loss of one row is $\operatorname{softplus}(z) - yz$. See [From a score to a probability](/lessons/008-sigmoid-and-log-loss).
- You turn the probability into a label by comparing it with a threshold. At 0.5 the label follows the sign of the score. The same chapter says the threshold is a choice about costs.
- Accuracy is the fraction of rows whose label matches the truth. See [Fitting logistic regression](/lessons/009-fitting-logistic-regression).
- A score on the rows a model was fitted on is kinder than a score on rows it has not seen. See [Scoring a rule honestly](/lessons/001-loss-and-honest-scores).
- The lowest of many noisy scores sits below the truth. See [Learning curves and the winner's curse](/lessons/013-learning-curves-and-winners-curse). The number you feed that comparison still has to be the right number.

## Count the four outcomes

Picture the 100 loans as a grid. Rows are what happened. Columns are what the model said. The model that calls every loan repaid fills one cell: 99 repaid loans it called repaid, and 1 default it called repaid. The other two cells are empty. Accuracy reads the diagonal, 99 out of 100, and the cell that costs the lender sits off that diagonal with a 1 in it.

```figure
<svg viewBox="0 0 640 240" role="img" aria-label="Confusion matrix of 100 loans. The one default was called repaid, and 99 repaid loans were called repaid.">
<text x="250" y="32" text-anchor="middle" class="fig-bold">called a default</text>
<text x="470" y="32" text-anchor="middle" class="fig-bold">called repaid</text>
<text x="20" y="104" class="fig-bold">defaulted</text>
<text x="20" y="184" class="fig-bold">repaid</text>
<rect x="160" y="64" width="180" height="72" class="fill-sunk stroke-muted"/>
<rect x="380" y="64" width="180" height="72" class="fill-soft-b stroke-muted"/>
<rect x="160" y="148" width="180" height="72" class="fill-sunk stroke-muted"/>
<rect x="380" y="148" width="180" height="72" class="fill-soft-a stroke-muted"/>
<text x="250" y="108" text-anchor="middle" class="fig-bold">0</text>
<text x="470" y="108" text-anchor="middle" class="fig-bold">1</text>
<text x="250" y="192" text-anchor="middle" class="fig-bold">0</text>
<text x="470" y="192" text-anchor="middle" class="fig-bold">99</text>
</svg>
Figure 1. Of 100 loans, the model that calls every loan repaid is right on 99 and misses the only default. Accuracy reads the diagonal and treats that miss as 1 percent of the grid.
```

Call a default the *positive* class, the event you are trying to find. The four cells then have names. A *true positive* is a default you flagged. A *false positive* is a repaid loan you flagged, a false alarm. A *false negative* is a default you called repaid, a miss. A *true negative* is a repaid loan you called repaid. On this grid the counts are 0, 0, 1, and 99.

Accuracy is $(\mathrm{tp} + \mathrm{tn}) / n$. With one default in a hundred, the true negatives do almost all of the work, which is why 0.99 and 0.91 disagree with the desk.

Eight loans are enough to fill every cell. Take these probabilities of default, and call a loan a default when the probability is at least 0.5:

| Loan | Probability | Defaulted |
|---|---|---|
| A | 0.95 | yes |
| B | 0.80 | yes |
| C | 0.70 | no |
| D | 0.60 | yes |
| E | 0.45 | no |
| F | 0.30 | no |
| G | 0.25 | yes |
| H | 0.10 | no |

The cut flags A, B, C, and D. A, B, and D did default, so there are 3 true positives. C repaid, so there is 1 false positive. G defaulted and was not flagged, so there is 1 false negative. E, F, and H repaid and were left alone, so there are 3 true negatives.

*Precision* asks, of the loans you flagged, how many defaulted. That is $3 / (3 + 1) = 0.75$. *Recall* asks, of the loans that defaulted, how many you flagged. That is $3 / (3 + 1) = 0.75$. In formulas, with the four counts:

$$
\mathrm{precision} = \frac{\mathrm{tp}}{\mathrm{tp} + \mathrm{fp}}, \qquad \mathrm{recall} = \frac{\mathrm{tp}}{\mathrm{tp} + \mathrm{fn}}
$$

Precision divides caught defaults by flags. Recall divides caught defaults by defaults. The model that flags nobody has $\mathrm{tp} + \mathrm{fp} = 0$, and $0/0$ is not a number, so this chapter sets precision to 0, recall to 0 when there are no defaults, and F1 to 0 when both rates are 0. A blank would hide the model that refuses to flag anyone.

Precision and recall pull in opposite directions. Flag only loan A and precision is 1, while recall is $1/4 = 0.25$, because three defaults were missed. The *F1 score* is the harmonic mean of the two rates, which is small when either rate is small:

$$
F_1 = \frac{2 \cdot \mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
$$

On that one-loan cut, an ordinary average of 1 and 0.25 is 0.625, which treats a recall of one in four as nearly fine. The harmonic mean is $2 \times 1 \times 0.25 / 1.25 = 0.4$. On the eight loans at the cut 0.5, both rates are 0.75, so $F_1 = 0.75$ as well. When the denominator is 0, report 0 rather than a blank.

```python
tp, fp, fn, tn = confusion_counts(y_true, y_pred)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

> [!KEY]
> Accuracy counts repaid loans you correctly left alone, so a rare default barely moves it. Precision and recall count the flagged loans and the defaults, and a model that flags nobody has precision 0, not a blank.

## Pick a threshold from a cost

The cut at 0.5 is a habit, not a fact about the lender. Suppose a false alarm costs 1, because someone reviews a loan that would have repaid, and a miss costs 9, because the default is the loss the desk exists to avoid. For a loan whose probability of default is $p$, the expected cost of flagging it is $(1 - p) \times 1$, and the expected cost of letting it through is $p \times 9$.

Flag it when flagging is the cheaper call:

$$
(1 - p) \, c_{\mathrm{fp}} < p \, c_{\mathrm{fn}}
$$

Move the terms. The probability has to clear the false-alarm share of the two costs:

$$
p > \frac{c_{\mathrm{fp}}}{c_{\mathrm{fp}} + c_{\mathrm{fn}}}
$$

With costs 1 and 9 the cut is $1/10 = 0.1$. At exactly 0.1 the two calls have the same expected cost, so either is fine. On the eight loans, every probability above 0.1 is flagged, which is A through G, and H at 0.10 stays out. You catch G, the default the 0.5 cut missed, and you add E and F as false alarms. The 0.5 cut pays $9 + 1 = 10$. The 0.1 cut pays $1 + 1 + 1 = 3$.

```python
cut = cost_fp / (cost_fp + cost_fn)
```

> [!TIP]
> The cut is a probability. A raw score that is not a probability, such as a margin, does not plug into this ratio. Calibrate the score first, then apply the cost.

## Sweep the threshold

Sort the eight loans from high probability to low and lower the cut one loan at a time. A default raises the true-positive rate, which is recall. A repayment raises the false-positive rate, false alarms divided by the repaid loans. Start above every score, at $(0, 0)$, and finish below every score, at $(1, 1)$. Loans that share a score are one step, so a tie moves the curve on a diagonal instead of in whatever order the sort listed them. These eight scores do not tie:

| After this score | False-positive rate | True-positive rate |
|---|---|---|
| none | 0 | 0 |
| 0.95 | 0 | 0.25 |
| 0.80 | 0 | 0.50 |
| 0.70 | 0.25 | 0.50 |
| 0.60 | 0.25 | 0.75 |
| 0.45 | 0.50 | 0.75 |
| 0.30 | 0.75 | 0.75 |
| 0.25 | 0.75 | 1 |
| 0.10 | 1 | 1 |

The cut at 0.5 sits on the row "after 0.60", because A through D are included and E is not. That point is a false-positive rate of 0.25 and a true-positive rate of 0.75, the same 1 false alarm and 3 caught defaults as before. Figure 2 draws the staircase. A model that does not rank defaults above repayments follows the diagonal, and the area under that diagonal is 1/2.

```figure
<svg viewBox="0 0 640 340" role="img" aria-label="ROC staircase for eight loans. The curve climbs when a default is added and steps right when a repayment is added. The area is 0.75.">
<line x1="64" y1="288" x2="304" y2="288" class="stroke-muted"/>
<line x1="64" y1="168" x2="304" y2="168" class="stroke-grid"/>
<line x1="64" y1="48" x2="304" y2="48" class="stroke-grid"/>
<line x1="64" y1="288" x2="64" y2="48" class="stroke-muted"/>
<line x1="184" y1="288" x2="184" y2="48" class="stroke-grid"/>
<line x1="304" y1="288" x2="304" y2="48" class="stroke-grid"/>
<line x1="64" y1="288" x2="304" y2="48" class="stroke-muted" stroke-dasharray="5 4"/>
<polyline points="64.0,288.0 64.0,228.0 64.0,168.0 124.0,168.0 124.0,108.0 184.0,108.0 244.0,108.0 244.0,48.0 304.0,48.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="64.0" cy="288.0" r="3.5" class="fill-b"/>
<circle cx="64.0" cy="228.0" r="3.5" class="fill-b"/>
<circle cx="64.0" cy="168.0" r="3.5" class="fill-b"/>
<circle cx="124.0" cy="168.0" r="3.5" class="fill-b"/>
<circle cx="124.0" cy="108.0" r="5" class="fill-a"/>
<circle cx="184.0" cy="108.0" r="3.5" class="fill-b"/>
<circle cx="244.0" cy="108.0" r="3.5" class="fill-b"/>
<circle cx="244.0" cy="48.0" r="3.5" class="fill-b"/>
<circle cx="304.0" cy="48.0" r="3.5" class="fill-b"/>
<text x="136" y="100" class="fig-small">cut at 0.5</text>
<text x="64" y="308" text-anchor="middle" class="fig-small">0</text>
<text x="184" y="308" text-anchor="middle" class="fig-small">0.5</text>
<text x="304" y="308" text-anchor="middle" class="fig-small">1</text>
<text x="52" y="292" text-anchor="end" class="fig-small">0</text>
<text x="52" y="172" text-anchor="end" class="fig-small">0.5</text>
<text x="52" y="52" text-anchor="end" class="fig-small">1</text>
<text x="184" y="332" text-anchor="middle">false-positive rate</text>
<text x="18" y="168" text-anchor="middle" transform="rotate(-90 18 168)">true-positive rate</text>
<text x="360" y="78" class="fig-bold">area 0.75</text>
<text x="360" y="104" class="fig-small">The dashed line is a ranking</text>
<text x="360" y="122" class="fig-small">that does not separate the</text>
<text x="360" y="140" class="fig-small">two classes. Its area is 0.5.</text>
<text x="360" y="172" class="fig-small">Horizontal steps are repaid</text>
<text x="360" y="190" class="fig-small">loans. Vertical steps are</text>
<text x="360" y="208" class="fig-small">defaults. The marked point</text>
<text x="360" y="226" class="fig-small">is the cut at probability 0.5.</text>
</svg>
Figure 2. The ROC of the eight loans. The marked point is the cut at 0.5, three defaults caught and one false alarm. The staircase encloses area 0.75, above the diagonal of area 0.5.
```

The area under this *receiver operating characteristic* curve, the ROC, is also a chance. Pick one default and one repaid loan. The area is the chance the default scores higher, and a tie counts as half because it is not a ranking either way:

$$
\mathrm{AUC} = \frac{1}{n_{+} n_{-}} \sum_{y_i = 1} \sum_{y_j = 0} \left( \mathbf{1}[s_i > s_j] + \tfrac{1}{2}\, \mathbf{1}[s_i = s_j] \right)
$$

$n_{+}$ and $n_{-}$ are the two class counts, so the denominator is the number of pairs. Add 1, $1/2$, or 0, then divide. The eight loans make $4 \times 4 = 16$ pairs. Scores 0.95 and 0.80 each beat all 4 repayments (8). Score 0.60 loses only to 0.70 (3). Score 0.25 beats only 0.10 (1). The total is 12, and $12/16 = 0.75$.

The trapezoid agrees. Vertical steps have width 0. The horizontal steps have width 0.25 and heights 0.50, 0.75, 0.75, and 1, so the area is $0.25 \times (0.50 + 0.75 + 0.75 + 1) = 0.75$. Counting a tie as a win, or drawing the tie by listing defaults first, inflates both forms together.

```python
diff = pos[:, None] - neg[None, :]
area = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5)).mean()
```

Equal scores make every pair a half, so the area is $1/2$. A ranking that puts every repayment strictly above every default has area 0. That is a separation in the wrong direction, not a missing one.

The same walk is a precision-recall curve, and precision is allowed to fall and then rise. After loans A and B, precision is 1 and recall is 0.50. Loan C, a repayment, drops precision to $2/3$ at the same recall. Loan D, a default, lifts precision to $3/4$ and recall to 0.75. On the 100 loans, catching the one default together with 9 false alarms is a true-positive rate of 1 and a false-positive rate of $9/99 \approx 0.091$, near the top-left of an ROC. Precision at that point is $1/10 = 0.1$, because precision divides by the 10 loans on the desk, not by all 99 repayments.

> [!KEY]
> AUC is the chance a random default outranks a random repayment. A tie counts as $1/2$. Scores that do not order the two classes have area $1/2$, and the trapezoid of the ROC agrees when each tied score is one step.

## Read a reliability table

The cost cut treats 0.25 as a probability: in a large pile of loans scored 0.25, about one in four defaults. A model can rank loans well and still be wrong about that. *Calibration* is the match between the probability and the rate. A *reliability table* bins the probabilities and, in each bin, compares the mean probability with the fraction of defaults.

Split the eight probabilities into four equal bins of width 0.25. A probability $p$ goes to bin $\min(\lfloor 4p \rfloor, 3)$, so 0.25 starts the second bin, because $\lfloor 1 \rfloor = 1$, and a probability of 1 would be caught by the last bin rather than falling off the end.

| Bin | Loans | Mean probability | Fraction of defaults | Gap |
|---|---|---|---|---|
| 0 to 0.25 | H | 0.10 | 0 | 0.10 |
| 0.25 to 0.50 | G, F, E | 0.333 | 0.333 | 0 |
| 0.50 to 0.75 | D, C | 0.65 | 0.50 | 0.15 |
| 0.75 to 1 | B, A | 0.875 | 1 | 0.125 |

The middle-left bin is a clean check. The probabilities $0.25$, $0.30$, and $0.45$ average to $1/3$, and one of the three loans defaulted, so the gap is 0. The other bins are small, and a gap on one or two loans is mostly noise, which is why the next example uses hundreds of rows.

*Expected calibration error*, ECE, is the average gap, weighted by how many loans sit in the bin. Empty bins add nothing:

$$
\mathrm{ECE} = \sum_m \frac{n_m}{n} \left| \bar{y}_m - \bar{p}_m \right|
$$

On the eight loans that is $(1/8)(0.10) + (2/8)(0.15) + (2/8)(0.125) = 0.0125 + 0.0375 + 0.03125 = 0.08125$. Guo, Pleiss, Sun, and Weinberger define the same weighted average of absolute gaps for the confidence of a predicted class. For a probability of default, the gap you want is the one in the table above, because the cost cut is a cut on that probability.

## Soften an overconfident probability

Take 500 loans whose true log-odds $z$ are standard normal, and let each loan default with probability $\sigma(z)$. The seed is 0. A model that reports $3z$ instead of $z$ is too sure: probabilities near $1/2$ are pushed toward 0 and 1. Dividing the reported score by a *temperature* $T > 0$ pulls them back,

$$
p_i = \sigma(z_i / T)
$$

$T = 1$ leaves the score alone. $T > 1$ softens it. $T < 1$ makes it sharper. The order of the loans does not change, so the ROC does not change either. What changes is the probability you would feed to the cost cut.

Choose $T$ by minimising the log loss, the same sum as in the logistic chapter, on rows the model was not fitted on:

$$
L(T) = \sum_i \left( \operatorname{softplus}(z_i / T) - y_i z_i / T \right)
$$

The slope in $T$ follows the chain rule. The slope of one row's loss in its score $u = z/T$ is $p - y$, and $u$ falls by $z/T^2$ when $T$ rises by 1, so

$$
\frac{dL}{dT} = \sum_i (p_i - y_i) \left( -\frac{z_i}{T^2} \right)
$$

At $T = 1$ on these 500 loans the slope is $-159.0$, so raising the temperature lowers the loss. The bend is how fast that slope changes. A central difference of the slope measures it, and Newton steps by slope divided by bend, starting at $T = 1$:

```python
slope = temperature_nll_grad(logits, y, temperature)
bend = (
    temperature_nll_grad(logits, y, temperature + 1e-4)
    - temperature_nll_grad(logits, y, temperature - 1e-4)
) / 2e-4
temperature = temperature - slope / bend
```

If a step would make the temperature negative, halve the step and try again. On these rows the raw steps stay positive. The scores were tripled, so a temperature of 3 would undo that multiplication on the whole population. These 500 loans are a sample, and the loss on them is lowest at $T = 2.370$: 336.48 at temperature 1, 280.07 at 2.370, and 282.27 at 3.

With 10 bins, ECE falls from 0.146 to 0.047. Figure 3 shows the two large piles: 134 loans with mean probability 0.044, of which 0.224 default, and 130 loans with mean 0.959, of which 0.815 default. After the division those piles move inward.

```figure
<svg viewBox="0 0 640 360" role="img" aria-label="Reliability diagram of 500 loans. Before temperature scaling the extreme bins sit off the diagonal. After scaling they sit closer to it.">
<line x1="72" y1="284" x2="332" y2="284" class="stroke-muted"/>
<line x1="72" y1="154" x2="332" y2="154" class="stroke-grid"/>
<line x1="72" y1="24" x2="332" y2="24" class="stroke-grid"/>
<line x1="72" y1="284" x2="72" y2="24" class="stroke-muted"/>
<line x1="202" y1="284" x2="202" y2="24" class="stroke-grid"/>
<line x1="332" y1="284" x2="332" y2="24" class="stroke-grid"/>
<line x1="72" y1="284" x2="332" y2="24" class="stroke-muted" stroke-dasharray="5 4"/>
<circle cx="83.5" cy="225.8" r="4.5" class="fill-b"/>
<circle cx="119.4" cy="202.1" r="4.5" class="fill-b"/>
<circle cx="151.5" cy="160.5" r="4.5" class="fill-b"/>
<circle cx="185.4" cy="207.5" r="4.5" class="fill-b"/>
<circle cx="220.7" cy="122.6" r="4.5" class="fill-b"/>
<circle cx="254.1" cy="157.3" r="4.5" class="fill-b"/>
<circle cx="285.7" cy="115.0" r="4.5" class="fill-b"/>
<circle cx="321.2" cy="72.0" r="4.5" class="fill-b"/>
<circle cx="87.2" cy="284.0" r="4.5" class="fill-c"/>
<circle cx="119.5" cy="240.1" r="4.5" class="fill-c"/>
<circle cx="152.6" cy="178.1" r="4.5" class="fill-c"/>
<circle cx="184.8" cy="191.1" r="4.5" class="fill-c"/>
<circle cx="219.8" cy="139.9" r="4.5" class="fill-c"/>
<circle cx="252.4" cy="118.5" r="4.5" class="fill-c"/>
<circle cx="283.4" cy="64.3" r="4.5" class="fill-c"/>
<circle cx="310.4" cy="33.6" r="4.5" class="fill-c"/>
<text x="72" y="304" text-anchor="middle" class="fig-small">0</text>
<text x="202" y="304" text-anchor="middle" class="fig-small">0.5</text>
<text x="332" y="304" text-anchor="middle" class="fig-small">1</text>
<text x="60" y="288" text-anchor="end" class="fig-small">0</text>
<text x="60" y="158" text-anchor="end" class="fig-small">0.5</text>
<text x="60" y="28" text-anchor="end" class="fig-small">1</text>
<text x="202" y="332" text-anchor="middle">mean predicted probability</text>
<text x="16" y="154" text-anchor="middle" transform="rotate(-90 16 154)">fraction of defaults</text>
<circle cx="390" cy="78" r="4.5" class="fill-b"/>
<text x="404" y="82">before, temperature 1</text>
<circle cx="390" cy="108" r="4.5" class="fill-c"/>
<text x="404" y="112">after, temperature 2.37</text>
<text x="390" y="156" class="fig-small">The dashed line is a</text>
<text x="390" y="174" class="fig-small">probability that matches</text>
<text x="390" y="192" class="fig-small">the fraction of defaults.</text>
<text x="390" y="222" class="fig-small">ECE with 10 bins falls</text>
<text x="390" y="240" class="fig-small">from 0.146 to 0.047.</text>
</svg>
Figure 3. Reliability of 500 loans whose scores were tripled, in eighths. Notice the two large bins before scaling: 134 loans near probability 0, of which 0.224 default, and 130 loans near 0.96, of which 0.815 default. Dividing by 2.37 moves the bins toward the diagonal.
```

> [!WARNING]
> Fit the temperature on rows the model did not train on. On rows it already fits, a sharper probability lowers the loss, so the fitted temperature shrinks and the model becomes more sure, which is the opposite of the repair.

> [!KEY]
> Temperature scaling divides every score by one positive number chosen to minimise the log loss. It does not change which loan outranks which. It changes the probability you compare with the cost cut, and on this sample it cuts ECE from 0.146 to 0.047.

## Weight the rare row, or copy it

A rare default can also vanish from the fit, not only from the score. The log loss of one row is $\operatorname{softplus}(z) - yz$. Take a default with score $z = 1$. The loss is $\operatorname{softplus}(1) - 1 = 0.3133$. Multiply that row's loss by 4, because misses are costly or because defaults are scarce, and the row contributes 1.253. Paste the same row into the table four times and the four copies contribute $4 \times 0.3133 = 1.253$. The slope in the score, $p - y$, is multiplied by 4 either way. A repaid loan at the same score has loss $\operatorname{softplus}(1) = 1.313$, and you leave it alone, because it is the common class.

Copying lengthens the table. Dropping repaid loans until the classes match throws rows away. Weighting keeps every row and only changes the multiplier, so the full-batch loss matches the copied table without a second training loop.

## Where this breaks

**The cost cut needs a probability.** Temperature scaling changes the size of a score, not which loan outranks which, and only for rows drawn from the same population.

**A bin of one loan is not a rate.** Its fraction of defaults is 0 or 1. The eight loans use 4 bins and the 500 loans use 10. The lowest bin after scaling holds 20 loans, and a bin that small can leave the diagonal by chance.

**Quote the cut you can afford.** A false-positive rate of 0.091 on the 100 loans still means 9 false alarms per caught default. Name the positive class, because swapping it changes precision, recall, and the ROC, while accuracy on two classes stays put.

> [!RECAP]
> - Accuracy on a 1 percent default rate can read 0.99 for a model that catches nothing.
> - Precision is caught defaults over flags, recall is caught defaults over defaults, and both are 0 when the denominator is 0.
> - The cost cut is $c_{\mathrm{fp}} / (c_{\mathrm{fp}} + c_{\mathrm{fn}})$. Costs 1 and 9 put it at 0.1.
> - AUC is the chance a default outranks a repayment, with a tie worth $1/2$, and it matches the trapezoid of the ROC.
> - ECE is the weighted average gap in a reliability table. Dividing the scores by a fitted temperature lowers it without reordering the loans.

## Build it step by step

You write eight functions, in the order the checks test them. Labels are 0 or 1, with 1 for a default. A higher score means a default is more likely.

1. **`confusion_counts(y_true, y_pred)`** returns `[tp, fp, fn, tn]`, shape `(4,)`. The eight loans at 0.5 give `[3, 1, 1, 3]`, and the 100 loans with no flags give `[0, 0, 1, 99]`. The common mistake is to swap false positives with false negatives.

2. **`precision_recall_f1(y_true, y_pred)`** returns precision, recall, and F1, shape `(3,)`. Return 0 when a denominator is 0. The eight loans give `[0.75, 0.75, 0.75]`. The common mistake is an unguarded `tp / (tp + fp)`, which is `NaN` when you flagged nobody.

3. **`threshold_from_cost(cost_fp, cost_fn)`** returns `cost_fp / (cost_fp + cost_fn)`. Costs 1 and 9 return 0.1. The common mistake is `cost_fn / cost_fp`, which is 9.

4. **`roc_curve(scores, y)`** returns the false-positive rates and the true-positive rates. Sort with `np.argsort(-scores, kind="mergesort")`, write `(0, 0)`, then one point after each distinct score, and finish at `(1, 1)`. The common mistake is a point per row, which breaks a tie by sort order.

5. **`auc(scores, y)`** returns the pair average. The eight loans give 0.75, and equal scores give 0.5. The common mistake is to count a tie as 1.

6. **`ece(probs, y, bins)`** returns the weighted mean absolute gap. Bin with $\min(\lfloor p \cdot \mathrm{bins} \rfloor, \mathrm{bins} - 1)$. Four bins on the eight loans return 0.08125. The common mistake is a bin index of `bins` when the probability is 1, or a gap from an empty bin.

7. **`temperature_nll_grad(logits, y, temperature)`** returns the slope of the sum $L(T)$. A central difference of that sum has to match it. The common mistake flips the sign: on the 500 loans at temperature 1 the slope is about $-159$, not $+159$.

8. **`temperature_scale(logits, y)`** starts at temperature 1 and steps by Newton, with the bend a central difference of the slope. Stop below an absolute slope of $10^{-8}$, and halve a step that would make the temperature non-positive. On the 500 loans the result is 2.370, the loss is lower than at the neighbouring temperatures, and ECE falls. The common mistake is to return 1, or to multiply the scores by the temperature.

## The exercise

You implement `confusion_counts`, `precision_recall_f1`, `threshold_from_cost`, `roc_curve`, `auc`, `ece`, `temperature_nll_grad`, and `temperature_scale`. The checks cover the eight loans, the 100 loans with no flags, a seeded sample with tied scores, and the 500 loans whose scores were tripled. They compare the trapezoid with the pair average to $10^{-12}$, they require AUC $1/2$ when the scores do not order the classes, and they check the temperature slope by a finite difference.

## Further reading

The notes and the paper below are the ones this chapter follows:

- For the confusion matrix, precision, recall, F1, and the ROC and precision-recall curves, see the [Fall 2019 evaluation metrics notes](https://cs229.stanford.edu/section/evaluation_metrics_fall2019.pdf) (Anand Avati). The [CS229 Fall 2019 syllabus](https://cs229.stanford.edu/syllabus-fall2019.html) lists that session as Section 4, "Friday Lecture: Evaluation Metrics".
- For the area under the ROC as the probability that a random positive ranks above a random negative, see the [Fall 2021 evaluation metrics notes](https://cs229.stanford.edu/notes2021fall/section6notes-evaluation-metrics.pdf) (Nandita Bhaskhar, 29 October 2021). A tie, which those notes do not spell out, contributes one half in the sum above.
- For expected calibration error as a weighted average of reliability-diagram gaps, and for temperature scaling as the one-parameter rescaling that minimises the log loss, see [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) (Guo, Pleiss, Sun, and Weinberger, 2017).
