---
title: "Debugging and tuning a network"
slug: 042-debugging-and-tuning
part: "Part 4 — Neural networks"
summary: "Predict the untrained loss, sweep the learning rate, and read the size of one step before you change the model."
minutes: 75
gpu: false
objectives:
  - Predict the loss of an untrained classifier and use a large gap as a bug detector.
  - Memorise one small batch so a broken step is not mistaken for a hard dataset.
  - Sweep learning rates from a fresh start and keep the rate at the bottom of the loss curve.
  - Draw hyperparameters log-uniformly and read the update-to-weight ratio of one step.
  - Decide from the two losses whether to train longer, regularise, or collect more data.
prereqs:
  - 014-learning-curves-and-winners-curse
  - 036-softmax-and-whole-network-gradients
  - 037-sgd-adam-and-schedules
  - 038-signal-size-and-initialisation
  - 040-weight-decay-and-early-stopping
  - 041-dropout-and-label-smoothing
exercise: 042-debugging-and-tuning
---

# Debugging and tuning a network

The digit reader that follows this chapter reports a loss of 4.1 after its first pass over the images. Two stories fit that number. The network is fine and the digits are hard. Or the loss is the wrong function, the labels are shifted by one, or the weights started so large that the network is already sure and already wrong. Another hour of training cannot separate those stories, because a broken update and a difficult dataset can draw the same flat line.

You already have the pieces: a starting scale, normalisation, weight decay, early stopping, and dropout. This chapter is the order you use them in. You predict the loss of a network that has learned nothing, you ask it to memorise a handful of rows, you sweep the learning rate, and you compare one step with the size of the weights. Each check answers a different question, so you run the checks before you change the model.

## What you already know

- The cross-entropy of a one-hot label is the negative natural log of the probability the model gave the true class. See [softmax and cross-entropy](/lessons/036-softmax-and-whole-network-gradients).
- A gradient step subtracts the learning rate times the gradient from the weights. See [SGD and Adam](/lessons/037-sgd-adam-and-schedules).
- The scale of the initial weights decides whether a deep stack shouts or goes silent. See [signal size and initialisation](/lessons/038-signal-size-and-initialisation).
- Weight decay, early stopping, and dropout are the guards against memorising the training rows.
- A learning curve that stays high on both sets is a different problem from a curve that is low on the training set and high on the held-out set. See [learning curves](/lessons/014-learning-curves-and-winners-curse).

## The loss of a network that has learned nothing

Suppose the network has four classes and has not seen a single row. The honest guess gives each class probability one quarter, and cross-entropy charges $\log 4$, about 1.39, on every row. Figure 1 sets that guess beside one that started too sure.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="Two groups of four bars. The left group gives each class probability one quarter. The right group puts probability 0.95 on the wrong class and 0.017 on the true class.">
  <text x="70" y="28" class="fig-bold">Untrained, loss 1.386</text>
  <text x="380" y="28" class="fig-bold">Large start, loss 4.053</text>
  <line x1="60" y1="200" x2="280" y2="200" class="stroke-ink"/>
  <line x1="370" y1="200" x2="590" y2="200" class="stroke-ink"/>
  <rect x="70" y="162.50" width="32" height="37.50" class="fill-a"/>
  <rect x="114" y="162.50" width="32" height="37.50" class="fill-a"/>
  <rect x="158" y="162.50" width="32" height="37.50" class="fill-c"/>
  <rect x="202" y="162.50" width="32" height="37.50" class="fill-a"/>
  <rect x="380" y="57.81" width="32" height="142.19" class="fill-bad"/>
  <rect x="424" y="197.40" width="32" height="2.60" class="fill-a"/>
  <rect x="468" y="197.40" width="32" height="2.60" class="fill-c"/>
  <rect x="512" y="197.40" width="32" height="2.60" class="fill-a"/>
  <text x="86" y="156" text-anchor="middle" class="fig-small">0.25</text>
  <text x="130" y="156" text-anchor="middle" class="fig-small">0.25</text>
  <text x="174" y="156" text-anchor="middle" class="fig-small">0.25</text>
  <text x="218" y="156" text-anchor="middle" class="fig-small">0.25</text>
  <text x="396" y="50" text-anchor="middle" class="fig-small">0.948</text>
  <text x="496" y="188" text-anchor="middle" class="fig-small">0.017 each</text>
  <text x="86" y="220" text-anchor="middle">0</text>
  <text x="130" y="220" text-anchor="middle">1</text>
  <text x="174" y="220" text-anchor="middle">2</text>
  <text x="218" y="220" text-anchor="middle">3</text>
  <text x="396" y="220" text-anchor="middle">0</text>
  <text x="440" y="220" text-anchor="middle">1</text>
  <text x="484" y="220" text-anchor="middle">2</text>
  <text x="528" y="220" text-anchor="middle">3</text>
  <text x="174" y="242" text-anchor="middle" class="fig-small">true class</text>
  <text x="484" y="242" text-anchor="middle" class="fig-small">true class</text>
</svg>
Figure 1. An untrained four-way classifier spreads probability evenly, and the loss is log 4, which is 1.386. Scores of (4, 0, 0, 0) put probability 0.948 on class 0. The true class is 2, whose probability is 0.017, and the loss is 4.053.
```

Take one row, four classes, and scores of $(0, 0, 0, 0)$. Each exponential is 1, the four sum to 4, and each probability is $1/4$, so the loss is

$$
-\log\frac{1}{4} = \log 4 = 1.386294.
$$

The same arithmetic with $C$ classes and scores of zero is probability $1/C$ and loss $\log C$. For ten digits that is $\log 10 = 2.303$:

```python
loss = np.log(C)
```

A large random start does not look like that. Scores of $(4, 0, 0, 0)$ have exponentials $(e^{4}, 1, 1, 1)$. Since $e^{4} = 54.598$, the sum is $57.598$, class 0 receives $0.948$, and each other class receives $0.01736$. If the true class is 2, the loss is $-\log(0.01736) = 4.053$. Nothing has been trained. The weights are the bug.

> [!KEY]
> An untrained classifier whose scores are near zero has loss $\log C$. A loss far from that number is a bug in the scores, the labels, or the loss.

The check allows an absolute gap of 0.5. On the seeded two-class network in Figure 2, scale 0.1 gives a loss of 0.696 against $\log 2 = 0.693$. The same draw at scale 2 gives 4.330, and the check flags it. A loss of $10 \log C$ is 23.03 for ten classes, far outside 0.5. Compare the data loss on its own: a weight penalty moves the number even when the probabilities are uniform, and the check then flags a healthy network.

## Memorise a handful of rows

The initial-loss check says the network is awake. It does not say the gradient points the right way, so the next check asks the network to memorise a batch too small to blame on the data.

Six rows, two features each, labels $(0, 1, 0, 1, 1, 0)$, a hidden width of 16, and a weight scale of 0.1. The features come from `np.random.default_rng(0)` and the weights from seed 2. Full-batch SGD at learning rate 0.8 changes the mean loss, measured before the update on that step, like this:

| Step about to run | Loss |
|---|---|
| 1 | 0.683 |
| 30 | 0.087 |
| 60 | 0.027 |
| 200 | 0.004 |

The first loss is $\log 2$ again, because the scores start near zero. By the start of step 200 the six rows are essentially memorised. With a learning rate of 0 the same network is still at 0.683, because the weights never move.

Turn weight decay and dropout off for this run. They are built to stop a perfect fit, so leaving them on makes a healthy network look broken. If the six rows reach a loss near zero and a held-out set stays bad, the model can learn, and the remaining question is data or regularisation.

> [!TIP]
> When a tiny batch will not memorise, change the step, the loss, or the gradient. Collecting more rows does not repair a step that cannot fit six of them.

## Sweep the learning rate

A learning rate that is too small leaves the loss sitting on $\log C$ for the whole budget. A learning rate that is too large steps past the fit and the loss climbs again. Between those two the loss after a fixed number of steps forms a bowl. Figure 2 is that bowl for one seeded problem.

```figure
<svg viewBox="0 0 640 310" role="img" aria-label="A curve of loss against learning rate on a log axis. The loss stays near log 2 on the left, bottoms at rate 3.16, and rises again at rate 10.">
  <line x1="78" y1="48" x2="610" y2="48" class="stroke-grid"/>
  <line x1="78" y1="98" x2="610" y2="98" class="stroke-grid"/>
  <line x1="78" y1="148" x2="610" y2="148" class="stroke-grid"/>
  <line x1="78" y1="198" x2="610" y2="198" class="stroke-grid"/>
  <line x1="78" y1="109.4" x2="610" y2="109.4" class="stroke-c dashed"/>
  <line x1="78" y1="248" x2="610" y2="248" class="stroke-ink"/>
  <line x1="78" y1="48" x2="78" y2="248" class="stroke-ink"/>
  <polyline points="78.0,108.8 144.5,109.1 211.0,109.7 277.5,111.4 344.0,124.5 410.5,217.1 477.0,235.2 543.5,237.5 610.0,101.4" fill="none" class="stroke-b" stroke-width="2"/>
  <circle cx="543.5" cy="237.5" r="5" class="fill-b"/>
  <text x="68" y="52" text-anchor="end" class="fig-small">1.0</text>
  <text x="68" y="102" text-anchor="end" class="fig-small">0.75</text>
  <text x="68" y="152" text-anchor="end" class="fig-small">0.5</text>
  <text x="68" y="202" text-anchor="end" class="fig-small">0.25</text>
  <text x="68" y="252" text-anchor="end" class="fig-small">0</text>
  <text x="78" y="272" text-anchor="middle" class="fig-small">0.001</text>
  <text x="211" y="272" text-anchor="middle" class="fig-small">0.01</text>
  <text x="344" y="272" text-anchor="middle" class="fig-small">0.1</text>
  <text x="477" y="272" text-anchor="middle" class="fig-small">1</text>
  <text x="610" y="272" text-anchor="middle" class="fig-small">10</text>
  <text x="500" y="100" class="fig-small">log 2</text>
  <text x="520" y="228" text-anchor="end" class="fig-small">3.16</text>
  <text x="344" y="300" text-anchor="middle">learning rate</text>
  <text x="40" y="36" class="fig-small">loss</text>
</svg>
Figure 2. Loss after 40 full-batch steps, against the learning rate on a log scale. Rates from 0.001 to 0.03 stay on the untrained loss, log 2. The bottom is at 3.16, where the loss is 0.053. At rate 10 the loss has climbed back to 0.733.
```

The data are 40 rows and two features, drawn from `np.random.default_rng(0)`. The label is 1 when the first feature is positive. The network has eight hidden units, the weights are drawn at scale 0.1 from seed 1, and each rate gets 40 full-batch steps from that same start. The nine rates are evenly spaced in $\log_{10}$ from 0.001 to 10. The losses are:

| Learning rate | Loss after 40 steps |
|---|---|
| 0.001 | 0.696 |
| 0.01 | 0.691 |
| 0.1 | 0.617 |
| 0.316 | 0.155 |
| 1 | 0.064 |
| 3.16 | 0.053 |
| 10 | 0.733 |

The bottom is 3.16. The smallest rate, 0.001, is still at the untrained loss, about 3,000 times smaller than the bottom. The pick is the rate with the smallest finite loss:

$$
\alpha^{\star} = \operatorname*{arg\,min}_{\alpha} L(\alpha),
$$

where $L(\alpha)$ is the mean loss after the fixed budget at learning rate $\alpha$, and a non-finite loss is left out of the minimum. When two rates tie, take the earlier one.

Each rate starts from a fresh copy of the same weights. One run that raises the rate as it goes mixes two questions: this rate is bad, and an earlier rate has already wrecked the weights. Leslie Smith's range test climbs along a single trajectory and reads accuracy. The bowl here restarts, so each rate is judged from the same weights.

> [!KEY]
> After a fixed budget, the loss against the learning rate is a bowl. Keep the rate at the bottom, not the smallest rate on the left.

## Search on a log scale

A learning rate of 0.01 and one of 0.02 are neighbours. A rate of 0.01 and a rate of 0.1 are different regimes. The rate multiplies the gradient, so the right ruler is a uniform draw on the log, not a uniform draw on the rate itself.

A setting $\theta$ drawn log-uniformly between $a$ and $b$ is

$$
\theta = 10^{u}, \qquad u \sim \mathrm{Uniform}(\log_{10} a, \log_{10} b).
$$

In numpy, with the names visited in sorted order so a seed means one sequence:

```python
value = 10 ** rng.uniform(np.log10(low), np.log10(high))
```

Using `np.exp` of a uniform draw on the natural log gives the same number. A uniform draw between `low` and `high` on the raw scale does not.

Four trials, seed 0, decay from $10^{-4}$ to $10^{-1}$ and learning rate from $10^{-3}$ to $1$, produce these configs. The score is $(\log_{10} \alpha + 1)^{2} + (\log_{10} \lambda + 2)^{2}$, which is zero at learning rate 0.1 and decay 0.01, so lower is closer to that pair:

| Trial | Decay | Learning rate | Score |
|---|---|---|---|
| 0 | 0.008145 | 0.006447 | 1.426 |
| 1 | 0.000133 | 0.001121 | 7.328 |
| 2 | 0.02753 | 0.5474 | 0.738 |
| 3 | 0.006606 | 0.1543 | 0.068 |

Trial 3 is the closest of the four, and the four learning rates are all different.

A grid repeats the axes that do not matter. Three learning rates times three decays is nine runs and only three distinct learning rates, while nine log-uniform draws test the rate nine times. The CS231n notes on hyperparameter optimisation put the trials on the axis that moves the loss and sample that axis on a log scale. If the winning sample sits on the edge of the range, widen the range and draw again. A random search still misses a spike narrower than the gaps between samples. It does not miss the broad bowl in Figure 2.

## Read the size of one step

The bowl says which rate wins in 40 steps. The next three readings say how hard one step hits the weights.

The worked network has three rows,

$$
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad y = (0, 1, 0),
$$

and these weights, with both biases zero:

$$
W_1 = \begin{bmatrix} 0.2 & -0.1 \\ 0 & 0.3 \end{bmatrix}, \qquad
W_2 = \begin{bmatrix} 0.4 & -0.2 \\ 0.1 & 0.2 \end{bmatrix}.
$$

The first row is $x = (1, 0)$. Its pre-activations are $(0.2, -0.1)$, and the ReLU zeroes the negative one, so the hidden row is $(0.2, 0)$. The scores are $0.2$ times the first row of $W_2$, which is $(0.08, -0.04)$. Across all three rows the hidden activations have mean 0.150 and standard deviation 0.112. One unit is off on the first row and on for the other two, so the column is not dead, and it is not shouting into the next layer.

The mean loss on the three rows is 0.653, close to $\log 2$ because the scores are still small. The gradient of that mean has norm 0.212 on $W_1$ and 0.0925 on $W_2$. The weights themselves have norms 0.374 and 0.500. The **update ratio** of a matrix is the norm of the step divided by the norm of the matrix. For plain SGD the step is the learning rate times the gradient, so

$$
r = \frac{\lVert \alpha g \rVert}{\lVert w \rVert}.
$$

Here $\alpha$ is the learning rate, $g$ is the gradient of the mean loss, and $w$ is one weight matrix. One line computes it:

```python
ratio = np.linalg.norm(lr * grad) / np.linalg.norm(weight)
```

At learning rate 0.01 the ratios are 0.00567 on $W_1$ and 0.00185 on $W_2$, near $10^{-3}$: each step nudges the weights by about a thousandth of their length, which is the size the CS231n notes use for a long SGD run. At learning rate 1 the same ratios are 0.567 and 0.185, so one step rewrites a large fraction of the matrix.

The bowl in Figure 2 picked 3.16, and the first step at that rate on the seeded network is not a nudge. There $\lVert g \rVert / \lVert w \rVert$ is 0.331 on $W_1$ and 0.133 on $W_2$, so the ratios are 1.05 and 0.42. A 40-step budget rewards that large opening step, because the gradient shrinks once the labels start to fit. A long run does not want to open by rewriting the weights. Read both instruments: the bowl picks the rate for the budget you measured, and the ratio says whether one step is a nudge.

> [!WARNING]
> The ratio multiplies the raw gradient by the learning rate. Adam does not step by that product, so $10^{-3}$ is not Adam's step size. The step in this chapter is plain SGD.

On the seeded network the ratios at learning rate 0.01 are 0.00331 and 0.00133. The check accepts anything from $10^{-4}$ to $10^{-2}$, a factor of ten around $10^{-3}$, so a correct ratio cannot miss the band. At learning rate 1 the ratios are 0.331 and 0.133, outside it. The same check compares your ratio with a central finite difference of the mean loss.

## Decide what to change

When the held-out loss is disappointing, three levers are in reach: train longer, turn on a regulariser, or collect more rows. The two losses say which lever. These runs draw each row from `np.random.default_rng(0)` and label it by a coin flip whose probability is the sigmoid of $1.4 x_1 - 0.6 x_2$. The weights start from seed 3 at scale 0.2, and the validation set is the next 80 rows from that generator. A perfect model still has a floor, because the label is a coin flip.

| What you trained | Train loss | Validation loss | What you do |
|---|---|---|---|
| 40 rows, 4 hidden units, 3 steps at rate 0.05 | 0.660 | 0.648 | Train longer. Both losses are still near $\log 2$. More rows will not move a fit that has not started. |
| 8 rows, 16 hidden units, 200 steps at rate 1 | 0.027 | 2.00 | Get more data, or turn on weight decay, dropout, or early stopping. The training rows are memorised and the held-out loss is worse than a coin flip. |
| 80 rows, 16 hidden units, 80 steps at rate 0.5 | 0.473 | 0.503 | Stop. The two losses meet, above the floor the coin-flip labels force. |

Regularisation fixes the second row and does nothing for the first, which is the flat left side of Figure 2. Once the two losses meet, as in the third row, stop.

> [!KEY]
> Both losses high means the fit has not started. Training loss low and validation loss high means the model memorised. Both losses meeting means you stop.

> [!RECAP]
> - An untrained classifier with small scores has loss $\log C$. A gap much larger than 0.5 means the scores, the labels, or the loss is wrong.
> - A handful of rows can be memorised with the regularisers off. If they cannot, the step is the bug.
> - The loss after a fixed budget, plotted against a log-spaced learning rate, is a bowl. Keep the bottom.
> - One SGD step of about the right size moves each weight matrix by about a thousandth of its length. A much larger ratio is a shove.
> - Both losses high, a low training loss with a high validation loss, and two losses that meet are three different decisions.

## Build it step by step

1. **`expected_initial_loss(C)`** returns one float, the natural log of `C`. For 10 classes that is 2.302585. The common mistake is $\log_{10}$, which returns 1 for ten classes.

2. **`check_initial_loss(model, X, y)`** returns a bool. Read the mean loss from `model.loss_and_grads(X, y)` and the class count from `model.n_classes`, and return true when the absolute gap from `expected_initial_loss` is at most 0.5. A loss of $10 \log C$ returns false. The common mistake is a band so wide that ten times the target still passes.

3. **`lr_range_test(model_fn, X, y, lrs, steps=40)`** returns one loss per rate, in order. Call `model_fn()` once per rate, take `steps` full-batch `sgd_step` updates, and record the loss after those updates. A non-finite loss becomes `inf` for that rate. The common mistake is to keep training the same network while the rate changes, so a late rate inherits a wrecked start.

4. **`pick_lr(lrs, losses)`** returns one float, the rate at the smallest finite loss, and the earliest rate when several tie. On the seeded bowl that is 3.16, not 0.001. The common mistake is to return the smallest rate.

5. **`random_search(objective, space, n, seed)`** returns a dict with `samples`, `scores`, and `best`. Visit `sorted(space)`, and draw each value as `10 ** rng.uniform(np.log10(low), np.log10(high))` from `np.random.default_rng(seed)`. Score every sample with `objective`, and let `best` be the earliest sample with the lowest score. The common mistake is a uniform draw on the raw interval, which piles samples near the top of a wide range.

6. **`update_ratio_stats(model, X, y, lr)`** returns a dict with keys `W1` and `W2`. Each value is the norm of `lr` times that matrix's gradient, divided by the norm of the matrix. Read the gradient from `loss_and_grads` and do not change the weights. The common mistake is to divide the gradient by the weights and forget to multiply by the learning rate, which reports a ratio about 0.3 where the nudge is about 0.003.

## Cost and scaling

A range test trains one short run per learning rate. With $R$ rates, $S$ steps, and a hidden width $H$, that is about $R \cdot S$ passes, and each pass is a pair of products with a side of length $H$. A random search multiplies that bill by the number of trials. Run both on a slice of the data: you are reading the shape of the bowl, and a slice keeps the cost near one real epoch instead of $R$ of them.

## Where this breaks

The $\log C$ check wants the last layer's scores near zero. A He-sized last layer fails it on purpose, which is the check working. A weight penalty inside the reported loss moves $\log C$ even when the probabilities are uniform, so read the data loss alone, then turn the penalty back on for the real run.

The $10^{-3}$ ratio is a heuristic for SGD on the weight matrices, not a constant a derivation produces. A bias that is still zero has an undefined ratio, because the denominator is zero. A ratio near $10^{-2}$ can be the right opening step for the short bowl in Figure 2 and a poor one for a run of many thousands of steps. A tiny budget also prefers a larger rate than a long run, so quote the budget next to the rate you keep. If the bottom is the largest rate you tried, the next sweep has to go higher.

Memorising six rows proves the model can fit the numbers it was given. It does not prove those numbers are the features you meant to load. If a shuffled copy of the features memorises equally well, the bug is in the data pipeline.

## The exercise

You implement `expected_initial_loss`, `check_initial_loss`, `lr_range_test`, `pick_lr`, `random_search`, and `update_ratio_stats`. The network, its cross-entropy, and its SGD step are already in the file. The checks cover $\log C$, a small initialisation that must pass, a loss of $10 \log 10$ that must be flagged, and a network drawn at scale 2 that must be flagged. They rebuild the seeded bowl, require the picked rate to sit within a factor of ten of the bottom, recompute a seeded log-uniform draw, and compare both update ratios with a central finite difference.

## Further reading

The notes and lectures below are the ones this chapter follows:

- For the untrained loss $\log C$, the update-to-weight ratio near $10^{-3}$, and random search on a log scale, see the CS231n notes on [learning and evaluation](https://cs231n.github.io/neural-networks-3/), in the sections "Before learning: sanity checks", "Ratio of weights:updates", and "Hyperparameter optimization". The [CS231n Spring 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html) titles the matching lecture "Lecture 3: Regularization and Optimization".
- For a range test that raises the learning rate and reads the curve, see [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) (Leslie N. Smith), section 3.3, "How can one estimate reasonable minimum and maximum boundary values?". That test climbs along one trajectory. This chapter restarts from the same weights at each rate.
- For activation and gradient statistics as a health check on a multilayer perceptron, see [Lecture 4: Building makemore Part 3: Activations & Gradients, BatchNorm](https://github.com/karpathy/nn-zero-to-hero) (Andrej Karpathy).
- For memorising a tiny slice before you scale the run up, see [Lecture 6: Final Projects & Practical Tips](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture06-final-project.pdf) (CS224N). The [course schedule](https://web.stanford.edu/class/cs224n/) lists that session as "Final Projects: Custom and Default; Practical Tips".
- For debugging a learning algorithm from the gap between training and held-out error, see [CS229 lecture 11, Bayesian statistics and regularization](https://see.stanford.edu/Course/CS229/36). The page titles the lecture that way and lists "Advice for Applying Machine Learning Algorithms", "Debugging/fixing Learning Algorithms", and "Diagnostics for Bias & Variance" among its topics.
- For a practical lecture on getting a deep network to train, see [Lecture 9: Hacker's guide to deep learning](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_lec09_mp4/) (Phillip Isola, MIT 6.7960).
