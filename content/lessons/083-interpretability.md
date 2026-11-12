---
title: "Interpretability: probes, attribution, and sparse autoencoders"
slug: 083-interpretability
part: "Part 7 — Large language models"
summary: "Read a feature out of frozen activations, share an output among its inputs with Shapley values, and untangle superposed features with a sparse autoencoder."
minutes: 75
gpu: false
objectives:
  - Train a logistic probe on frozen hidden states, and explain why a probe that works does not show the model uses the feature.
  - Use a shuffled-label control and a random network to judge what a probe's accuracy means.
  - Compute gradient times input and integrated gradients, and check that integrated gradients sums to f(x) - f(baseline).
  - Compute exact Shapley values over every subset of a few features, and check that they sum to f(x) - f(baseline).
  - Explain superposition, and train a sparse autoencoder that recovers planted features, scored by cosine against the true directions.
prereqs:
  - 009-fitting-logistic-regression
  - 007-lasso-and-fista
  - 082-mixtures-of-experts
exercise: 083-interpretability
---

# Interpretability: probes, attribution, and sparse autoencoders

A bank's model turns down a loan, and the applicant asks why. The model is a stack of matrix products, so no sentence inside it explains the decision. You can still ask it questions. Does a middle layer know the applicant's job type? Which of the twelve inputs pushed the score down the most? What are the thousand units in that layer storing?

This chapter gives you one tool for each question. A *probe* reads a feature out of frozen hidden states. *Attribution* shares one output among the inputs that produced it. A *sparse autoencoder* takes apart a layer that stores more features than it has units. Each tool gives an answer that sounds more certain than it is, so the chapter spends as much time on what each answer does not prove as on how to compute it. You build all three on small models where you planted the truth, so you can check each tool against a known answer.

## What you already know

- Logistic regression turns a score into a probability, and you fit it by gradient descent on the log loss ([fitting logistic regression](/lessons/009-fitting-logistic-regression)).
- A gradient is the slope of the output with respect to each input, and the chain rule computes it through layers.
- A hidden state is the vector of numbers a layer passes on, and a ReLU sets its negative entries to zero.
- An L1 penalty drives weights to exact zeros ([the lasso](/lessons/007-lasso-and-fista)).
- A router sends each token to a few experts, so most of a layer's units stay quiet for any one token ([mixtures of experts](/lessons/082-mixtures-of-experts)).

## Probes: read a feature from frozen activations

Suppose a network reads sentences, and you suspect that a middle layer represents whether the subject is singular or plural. You freeze the network, collect the layer's hidden state for many sentences, and label each state with the answer. Then you fit a logistic regression from the hidden state to the label. That classifier is a *linear probe*. If it predicts well on sentences it never saw, the activations hold the feature in a form a straight line can read.

A tiny example shows what the fitting does. Take four hidden states with two units each and labels 1, 1, 0, 0:

| row | hidden state | label |
|---|---|---|
| 1 | (1, 0) | 1 |
| 2 | (2, 1) | 1 |
| 3 | (-1, 0) | 0 |
| 4 | (-2, 1) | 0 |

Start the probe at $w = (0, 0)$ and $b = 0$. Every score is 0, so every predicted probability is $\sigma(0) = 0.5$, and the loss is $\ln 2 = 0.693$. The gradient of the log loss on the weights is the average of (probability minus label) times the hidden state:

$$
\nabla_w L = \frac{1}{n} H^\top (p - y) + \lambda w
$$

Read it aloud: each row pushes the weights toward its own hidden state by the amount the prediction missed, and $\lambda w$ pulls them back toward zero. Here $p - y = (-0.5, -0.5, 0.5, 0.5)$, and the sum of (miss times state) is $(-3, 0)$. Dividing by $n = 4$ gives the gradient $(-0.75, 0)$, and a step of size 0.5 moves the weights to $(0.375, 0)$. The probe has found that unit 0 carries the label. In numpy, the loss and gradient are:

```python
z = H @ w + b
loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * l2 * (w @ w)
r = sigmoid(z) - y                       # miss on each row
grad_w, grad_b = H.T @ r / n + l2 * w, r.mean()
```

The expression `np.logaddexp(0, z) - y * z` is the log loss written so that it never overflows.

### Why a working probe is not proof

It is tempting to read "the probe reaches 98 percent" as "the model knows this". Three things stand in the way.

1. **The probe can do the work itself.** A probe with many weights can learn the task from the raw input and ignore the layer.
2. **The information can be present but unused.** The probe reads the activations, not what the later layers do with them.
3. **A random network passes inputs through.** An untrained network's activations are a scrambled copy of its input, so a probe can often read the feature from them.

The fix is a control: train the same probe on activations that cannot hold the feature, and compare. Hewitt and Liang call the gap between the real task and a control task the probe's *selectivity*. Two controls are cheap. Shuffle the labels, so no feature can be real, or probe a random network on the same inputs.

The exercise builds both. Inputs have six numbers, and the label is whether the product of the first two is positive, which no straight line on the raw inputs can read. A "planted" network has eight hidden units, and one computes $\tanh(2 x_0)\tanh(2 x_1)$, the feature. A random network has eight random ReLU units. Figure 1 shows the held-out accuracy of the same probe on three inputs.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Held-out probe accuracy: planted activations, random network, shuffled labels">
<line x1="60" y1="200" x2="580" y2="200" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="200" class="stroke-muted"/>
<line x1="56" y1="200" x2="580" y2="200" class="stroke-grid"/>
<text x="50" y="204" text-anchor="end" class="fig-small">0</text>
<line x1="56" y1="115" x2="580" y2="115" class="stroke-grid"/>
<text x="50" y="119" text-anchor="end" class="fig-small">0.5</text>
<line x1="56" y1="30" x2="580" y2="30" class="stroke-grid"/>
<text x="50" y="34" text-anchor="end" class="fig-small">1</text>
<rect x="110" y="33.4" width="90" height="166.6" class="fill-a"/>
<text x="155" y="27.4" text-anchor="middle" class="fig-bold">0.98</text>
<text x="155" y="222" text-anchor="middle" class="fig-small">planted</text>
<rect x="270" y="105.9" width="90" height="94.1" class="fill-muted"/>
<text x="315" y="99.9" text-anchor="middle" class="fig-bold">0.55</text>
<text x="315" y="222" text-anchor="middle" class="fig-small">random network</text>
<rect x="430" y="119.5" width="90" height="80.5" class="fill-muted"/>
<text x="475" y="113.5" text-anchor="middle" class="fig-bold">0.47</text>
<text x="475" y="222" text-anchor="middle" class="fig-small">shuffled labels</text>
<text x="20" y="22" class="fig-small">held-out accuracy of the same logistic probe</text>
</svg>
Figure 1. The same probe gets 0.98 on the planted activations, which hold the feature, but 0.55 on a random network and 0.47 when the labels are shuffled. The two low bars are the controls: without them, you could not tell 0.98 from a probe that fits anything.
```

On an easier label, a random network would score much higher than 0.55, which is why you run the control every time.

> [!WARNING]
> Never report a probe's accuracy on its training rows. Hold out rows, and report a control beside the real number.

> [!KEY]
> A linear probe shows that a straight line can read a feature from the activations. It does not show that the model uses the feature. To judge a probe, put a control beside it. To show use, change the feature inside the model and watch the output move.

## Attribution: share one output among the inputs

A probe asks what a layer holds. Attribution asks, for one input and one output, how much of the output came from each input feature. Use a model small enough to do by hand:

$$
f(x) = x_0 x_1 + 2 x_2
$$

Read it aloud: the output is the product of the first two features plus twice the third. Take the input $x = (2, 3, 1)$ and a *baseline* of $(0, 0, 0)$, the reference input that means "feature absent". Then $f(x) = 6 + 2 = 8$ and $f(\text{baseline}) = 0$, so there are 8 units of output to share. A fair sharing would sum to 8.

### Gradient times input

The obvious attribution is the gradient times the input. The gradient is the slope of the output with respect to each feature, so slope times size looks like "how much this feature adds". Here the gradient is $(x_1, x_0, 2) = (3, 2, 2)$, and times the input $(2, 3, 1)$ it gives $(6, 6, 2)$. These sum to 14, not 8.

The reason is that a slope describes the model at one point. The slope of $x_0 x_1$ with respect to $x_0$ is 3 at the end, but it was smaller on the way there. Multiplying the end-point slope by the whole distance counts the product term twice.

### Integrated gradients

The fix is to take the slope along the whole walk from the baseline to $x$, not at one point, and to average it. Walk the straight path $\text{baseline} + \alpha (x - \text{baseline})$ as $\alpha$ goes from 0 to 1, and average the gradient along it:

$$
\text{IG}_i = (x_i - b_i) \int_0^1 \frac{\partial f}{\partial x_i}\bigl(b + \alpha (x - b)\bigr)\, d\alpha
$$

Read it aloud: feature $i$'s attribution is how far it moved, times the average slope it met on the way. On the worked model the gradient at the point $\alpha x$ is $(3\alpha, 2\alpha, 2)$, whose averages over $\alpha$ from 0 to 1 are $(1.5, 1, 2)$. Multiplying by $x = (2, 3, 1)$ gives $\text{IG} = (3, 3, 2)$, which sums to 8.

That sum is no accident. The *completeness* property says integrated gradients always sum to $f(x) - f(\text{baseline})$, by the fundamental theorem of calculus: a gradient integrated along a path adds up to the total change in $f$. It is also a check you can run on your own code.

In code, replace the integral with a sum over `steps` pieces, evaluating the gradient at the middle of each piece (the midpoint rule):

```python
alphas = (np.arange(steps) + 0.5) / steps
avg_grad = np.mean([grad_f(baseline + a * (x - baseline)) for a in alphas], axis=0)
ig = (x - baseline) * avg_grad
```

The completeness error shrinks as `steps` grows. For a model with a `tanh` unit, it is $1.1 \times 10^{-2}$ with 2 steps, $6.4 \times 10^{-4}$ with 8, $4.0 \times 10^{-5}$ with 32, and $2.5 \times 10^{-6}$ with 128. Four times the steps cut the error about sixteen times, the $1/\text{steps}^2$ of the midpoint rule.

> [!WARNING]
> The most common bug is to return the average gradient and forget to multiply by $(x - b)$. The result looks plausible but does not sum to $f(x) - f(b)$.

### Exact Shapley values

Integrated gradients needs a gradient. A second method only runs the model: treat each feature as a player and the output as the prize. A feature is "present" when it takes its value from $x$ and "absent" when it takes the baseline value. Let $v(S)$ be the output when exactly the features in $S$ are present. The following table lists $v$ for the worked model.

| present set $S$ | none | {0} | {1} | {2} | {0,1} | {0,2} | {1,2} | {0,1,2} |
|---|---|---|---|---|---|---|---|---|
| $v(S)$ | 0 | 0 | 0 | 2 | 6 | 2 | 2 | 8 |

Feature 0 adds nothing alone and adds 6 once feature 1 is present, so a fair share averages over the orders in which features could arrive. The *Shapley value* of feature $i$ averages its marginal contribution $v(S \cup \{i\}) - v(S)$ over every set $S$ that leaves $i$ out:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n - |S| - 1)!}{n!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr]
$$

Read it aloud: for every set of other features, take what feature $i$ adds to it, and weight the set by the fraction of arrival orders in which exactly that set arrives before $i$. With $n = 3$, sets of size 0 and 2 have weight $1/3$ and sets of size 1 have weight $1/6$. For feature 0, the marginals are 0, 6, 0, and 6 for the sets {}, {1}, {2}, and {1,2}, so

$$
\phi_0 = \tfrac13 (0) + \tfrac16 (6) + \tfrac16 (0) + \tfrac13 (6) = 3
$$

Feature 1 is symmetric and also gets 3. Feature 2 adds 2 to every set, so it gets 2. The values $(3, 3, 2)$ sum to 8. Shapley values always sum to $f(x) - f(\text{baseline})$; this is the *efficiency* property. For a linear model $w \cdot x + c$, feature $i$ always adds $w_i (x_i - b_i)$, so its Shapley value is exactly that.

Figure 2 sets the three methods side by side. Integrated gradients and exact Shapley values agree on this model and differ on many others. What they share is the promise to add up.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Bars for three attribution methods on the worked model; gradient times input sums to 14, the others to 8">
<line x1="170.0" y1="14" x2="170.0" y2="206" class="stroke-muted"/>
<text x="160.0" y="52" text-anchor="end" class="fig-bold">gradient × input</text>
<rect x="170.0" y="22" width="156.0" height="13" class="fill-b"/>
<text x="332.0" y="33" class="fig-small">x0: 6</text>
<rect x="170.0" y="38" width="156.0" height="13" class="fill-b"/>
<text x="332.0" y="49" class="fig-small">x1: 6</text>
<rect x="170.0" y="54" width="52.0" height="13" class="fill-b"/>
<text x="228.0" y="65" class="fig-small">x2: 2</text>
<text x="418.0" y="52" class="fig-small">sum 14</text>
<text x="160.0" y="114" text-anchor="end" class="fig-bold">integrated gradients</text>
<rect x="170.0" y="84" width="78.0" height="13" class="fill-a"/>
<text x="254.0" y="95" class="fig-small">x0: 3</text>
<rect x="170.0" y="100" width="78.0" height="13" class="fill-a"/>
<text x="254.0" y="111" class="fig-small">x1: 3</text>
<rect x="170.0" y="116" width="52.0" height="13" class="fill-a"/>
<text x="228.0" y="127" class="fig-small">x2: 2</text>
<text x="418.0" y="114" class="fig-small">sum 8</text>
<text x="160.0" y="176" text-anchor="end" class="fig-bold">exact Shapley</text>
<rect x="170.0" y="146" width="78.0" height="13" class="fill-c"/>
<text x="254.0" y="157" class="fig-small">x0: 3</text>
<rect x="170.0" y="162" width="78.0" height="13" class="fill-c"/>
<text x="254.0" y="173" class="fig-small">x1: 3</text>
<rect x="170.0" y="178" width="52.0" height="13" class="fill-c"/>
<text x="228.0" y="189" class="fig-small">x2: 2</text>
<text x="418.0" y="176" class="fig-small">sum 8</text>
<line x1="378.0" y1="14" x2="378.0" y2="206" class="stroke-b dashed"/>
<text x="378.0" y="226" text-anchor="middle" class="fig-small">f(x) - f(baseline) = 8</text>
</svg>
Figure 2. Gradient times input sums to 14 on a model whose output change is 8, because it counts the product term twice. Integrated gradients and exact Shapley values both sum to 8, the dashed line.
```

A frequent mistake is to weight every set by $1/n$. On the worked model that gives $\phi_0 = (0 + 6 + 0 + 6)/3 = 4$ instead of 3, and the values no longer add up. The price is $2^n$ model calls, so exact values stop at about six features and practical tools, such as Lundberg and Lee's SHAP, approximate them.

> [!KEY]
> Gradient times input uses the slope at one point, so it need not add up. Integrated gradients averages the slope along a path and sums to $f(x) - f(b)$. Exact Shapley values average over every subset of features and also sum to $f(x) - f(b)$. Both depend on the baseline you choose.

## Superposition and sparse autoencoders

Attribution works on inputs. The third question is about the inside of a layer. You might hope each of its units tracks one feature, such as "inside a quotation". But a model can track more features than it has units. Figure 3 shows how it fits.

Draw a layer with two units as a plane. Three features can each own a direction, three arrows at 120 degrees, and an active feature puts the activation on its arrow. If the features were active together often, the sum of two arrows would be confused with the third. But if each is active 12 percent of the time, then 70 percent of inputs have none active, 27 percent have one, and only 3 percent have two. Almost every point lies on one arrow, so you can read which feature is active. Sparsity pays for the crowding. Elhage and colleagues call this *superposition*. It makes single units hard to read, because a unit's axis cuts across several arrows and responds to several unrelated features. Such a unit is *polysemantic*.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Three feature directions in a plane, the points they make, and the directions a sparse autoencoder learns">
<defs><marker id="c066-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="150.0" y1="150.0" x2="450.0" y2="150.0" class="stroke-grid"/>
<line x1="300.0" y1="10.0" x2="300.0" y2="290.0" class="stroke-grid"/>
<circle cx="390.9" cy="202.5" r="3.5" class="fill-soft-a stroke-a"/>
<circle cx="300.0" cy="45.0" r="3.5" class="fill-soft-a stroke-a"/>
<circle cx="300.0" cy="150.0" r="3.5" class="fill-soft-a stroke-a"/>
<circle cx="209.1" cy="202.5" r="3.5" class="fill-soft-a stroke-a"/>
<circle cx="390.9" cy="97.5" r="3.5" class="fill-soft-a stroke-a"/>
<circle cx="209.1" cy="97.5" r="3.5" class="fill-soft-a stroke-a"/>
<circle cx="300.0" cy="255.0" r="3.5" class="fill-soft-a stroke-a"/>
<line x1="300.0" y1="150.0" x2="300.0" y2="32.4" class="stroke-ink" marker-end="url(#c066-arrow)"/>
<line x1="300.0" y1="150.0" x2="198.2" y2="208.8" class="stroke-ink" marker-end="url(#c066-arrow)"/>
<line x1="300.0" y1="150.0" x2="401.8" y2="208.8" class="stroke-ink" marker-end="url(#c066-arrow)"/>
<line x1="300.0" y1="150.0" x2="300.9" y2="55.5" class="stroke-b dashed" stroke-width="2.5"/>
<line x1="300.0" y1="150.0" x2="217.1" y2="195.4" class="stroke-b dashed" stroke-width="2.5"/>
<line x1="300.0" y1="150.0" x2="383.1" y2="195.0" class="stroke-b dashed" stroke-width="2.5"/>
<text x="14" y="26" class="fig-small">black arrows: planted directions</text>
<text x="14" y="44" class="fig-small">orange dashes: learned directions</text>
<text x="14" y="62" class="fig-small">blue dots: activations</text>
<text x="300.0" y="290.0" text-anchor="middle" class="fig-small">three features in two dimensions</text>
</svg>
Figure 3. Three planted feature directions in two dimensions (black), the activations they make (blue), and the three directions a sparse autoencoder learns from those points alone (orange dashes). The learned directions land on the planted ones, with cosine 1.000.
```

### The sparse autoencoder

A *sparse autoencoder* (SAE) undoes superposition. It maps an activation $x$ to a wider vector of feature strengths $f$, and back to a reconstruction $\hat{x}$:

$$
f = \operatorname{ReLU}(x\,W_e + b_e), \qquad \hat{x} = f\,W_d
$$

Read it aloud: the encoder scores every learned feature and keeps the positive scores, and the decoder rebuilds the activation as a sum of directions, one per row of $W_d$, weighted by $f$. The loss rewards a good rebuild and few active features:

$$
L = \frac{1}{n}\sum_{r} \lVert x_r - \hat{x}_r \rVert^2 \;+\; \lambda \,\frac{1}{n}\sum_{r}\sum_j f_{rj}
$$

The first term is the reconstruction error. The second is the L1 penalty, $\lambda$ times the sum of the strengths. Each piece has a job:

- **ReLU** gives exact zeros, so a point uses only a few directions.
- **L1** makes every active feature cost something, so the autoencoder prefers few.
- **Unit-length rows of $W_d$** stop a trick. Without them, the autoencoder could shrink $f$ and grow $W_d$ by the same factor, pay less L1, and learn nothing.

A tiny example shows why L1 picks the aligned direction. Let $x = (0.8, 0.6)$, and let the dictionary hold $d_0 = (1, 0)$, $d_1 = (0, 1)$, and $d_2 = (0.8, 0.6)$, all of unit length. Two codes rebuild $x$ exactly:

| code $f$ | rebuild | reconstruction error | L1 sum |
|---|---|---|---|
| (0, 0, 1) | $d_2 = (0.8, 0.6)$ | 0 | 1.0 |
| (0.8, 0.6, 0) | $0.8 d_0 + 0.6 d_1$ | 0 | 1.4 |

The rebuild is the same, but the second code costs 1.4 against 1.0. With $\lambda = 0.3$ the loss is 0.30 against 0.42, so training favours the direction already pointing at $x$. That pull turns a planted direction into a row of $W_d$.

The gradient comes from the chain rule. With $r = \hat{x} - x$, the gradient on the decoder is $f^\top (2r/n)$. The gradient that reaches $f$ is $(2r/n)\,W_d^\top + \lambda/n$. The ReLU gate $[\,\text{pre} > 0\,]$ multiplies it, and the encoder gradients follow:

```python
pre = X @ We + be
f = np.maximum(pre, 0)
resid = f @ Wd - X
df = (2 * resid / n) @ Wd.T + (l1 / n) * (pre > 0)
dpre = df * (pre > 0)       # the gate masks the slope of relu
```

> [!WARNING]
> The gate $[\,\text{pre} > 0\,]$ appears twice. It masks the gradient from the reconstruction, and it sets the slope of the L1 term, which is 1 for an active unit and 0 otherwise.

### Scoring the match

Because you planted the features, you can score the result. For each true direction, find the learned row with the largest absolute cosine, then average over true directions. The score is 1 when every planted direction appears in the dictionary, up to sign and scale.

On a larger problem, 16 features in 8 dimensions, each active 4 percent of the time, the trained dictionary scores 0.94 to 0.97 over five seeds. A random dictionary scores 0.67 to 0.71, not zero, because 16 random rows in 8 dimensions always sit fairly close to some planted direction. Compare against that baseline.

Bricken and colleagues train this kind of autoencoder on a small language model's activations. A real model has no list of true directions, so no such score exists, and you judge a learned feature by what makes it fire and what it changes in the output. The planted test tells you whether your code does what you think.

### Adversarial examples, in one paragraph

An attack uses the same gradient as attribution. Move each input coordinate by a small step $\varepsilon$ in the direction of the sign of the loss gradient $g$, and to first order the loss rises by $\varepsilon \sum_i |g_i|$. A tiny change in every coordinate adds up across many coordinates, so a change too small to notice can flip a prediction. Robustness is a field of its own, and this chapter leaves it here.

## Cost and scaling

Exact Shapley values cost $2^n$ model calls, so six features cost 64 and thirty cost about a billion. Integrated gradients costs `steps` gradient calls, and 20 to 50 is a usual choice. A probe costs one logistic regression on stored activations. A sparse autoencoder costs the most: its dictionary is several times wider than the layer, and it trains on millions of activations.

## Where this breaks

- **The baseline changes the answer.** A black image, a zero vector, and an average input mean different things by "absent". Report which one you used.
- **Probes mislead without controls.** Use a line, hold out data, and add a control.
- **Learned features need a check.** A small dictionary merges features, and a large one splits a feature in several. Look at what fires each feature before you name it.

> [!RECAP]
> - A linear probe is a logistic regression on frozen activations. Its held-out accuracy shows that a line can read the feature, and a shuffled-label or random-network control tells you how much to trust it.
> - Gradient times input uses the slope at one point and can miss the total. Integrated gradients averages the slope along a path from a baseline and sums to $f(x) - f(b)$.
> - Exact Shapley values average a feature's marginal contribution over every subset, with weight $|S|!\,(n - |S| - 1)!/n!$, and sum to $f(x) - f(b)$ at a cost of $2^n$ calls.
> - Superposition packs more features than dimensions into a layer, kept apart by sparsity. A sparse autoencoder (ReLU encoder, unit-length decoder rows, L1 penalty) can recover planted directions, scored by the best cosine per true direction.

## Build it step by step

The steps follow the order of the checks. A model is a Python function from a vector of shape `(d,)` to a number.

1. **`grad_times_input(grad_f, x)`** returns `grad_f(x) * x`. The mistake is to pass `f` instead of its gradient.
2. **`integrated_gradients(f, grad_f, x, baseline, steps)`** averages `grad_f` at the midpoints `(k + 0.5) / steps` of the path, then multiplies by `x - baseline`. The mistake is to skip that factor.
3. **`exact_shapley(f, x, baseline)`** loops over all bitmasks, stores `f` of each mixed input from `np.where(present, x, baseline)`, then sums weighted marginals per feature with `math.factorial`. The mistake is the weight `1 / n`.
4. **`probe_loss_grad(w, b, H, y, l2)`** returns the loss, `grad_w`, and `grad_b`. Leave the bias out of the penalty.
5. **`train_probe(H, y)`** runs full-batch gradient descent from zeros, and **`probe_accuracy(w, b, H, y)`** compares `H @ w + b > 0` with the labels. Evaluate on held-out rows.
6. **`match_score(Wd, D)`** normalises both sets of rows, takes absolute cosines with a matrix product, and averages the maximum over learned rows for each true direction. The mistake is to take the maximum over the wrong axis.
7. **`sae_loss_grads(params, X, l1)`** returns the loss and a dict of gradients for `We`, `be`, and `Wd`. Mask with `pre > 0` in both places.
8. **`sae_step(params, X, l1, lr)`** steps on a copy, then divides every row of `Wd` by its norm, and returns the updated dict and the loss before the step. The mistake is to skip the rescale.
9. **`train_sae(X, k)`** draws `k` random unit rows with `np.random.default_rng(seed)`, sets `We = Wd.T` and `be = 0`, and calls `sae_step` in a loop.

## The exercise

You implement the functions above in `starter.py`. The checks verify each tool against planted truth. `grad_times_input`, `integrated_gradients`, and `exact_shapley` must give the hand values on the worked model, `integrated_gradients` must sum to $f(x) - f(b)$ with an error that shrinks as `steps` grows, and `exact_shapley` must be efficient on six features and equal $w_i (x_i - b_i)$ on a linear model. `probe_loss_grad` and `sae_loss_grads` are compared with finite differences. `train_probe` must reach 0.9 held-out accuracy on planted activations while a random network and shuffled labels stay near chance. `train_sae` must recover 16 planted directions with a mean best cosine of at least 0.9, at least 0.15 above a random dictionary.

## Further reading

These are the papers and lectures this chapter follows. Lecture numbers appear only where the page prints them.

- For integrated gradients and the completeness property, see [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Sundararajan, Taly, and Yan, 2017).
- For Shapley values as feature attributions, see [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (Lundberg and Lee, 2017).
- For linear probes on intermediate layers, see [Understanding Intermediate Layers Using Linear Classifier Probes](https://arxiv.org/abs/1610.01644) (Alain and Bengio, 2016).
- For control tasks and selectivity, see [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) (Hewitt and Liang, 2019).
- For superposition in a toy model, see [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage and others, 2022).
- For sparse autoencoders on a language model, see [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken and others, 2023).
- For visualising and understanding what a vision network learned, see [CS231n lecture 9, Object Detection, Image Segmentation, Visualizing and Understanding](https://cs231n.stanford.edu/schedule.html). The Spring 2026 schedule prints "Lecture 9".
- For a talk on interpretability, see the guest lecture "Interpretability" by Been Kim on the [CS224N schedule](https://web.stanford.edu/class/cs224n/). The Winter 2026 schedule prints no lecture number for it.
