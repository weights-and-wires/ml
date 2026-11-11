---
title: "Normalisation layers: batch norm, layer norm, and their gradients"
slug: 036-normalisation-layers
part: "Part 4 — Neural networks"
summary: Re-set each group of numbers to mean 0 and variance 1 on every pass, compare batch norm and layer norm, and send the gradient back through both.
minutes: 80
gpu: false
objectives:
  - Normalise a group of numbers by hand, and say why the variance gets a small constant added and why a learned scale and shift follow.
  - Explain what batch norm and layer norm each average over, and predict what a shifted example and a batch of one do to each.
  - Build the forward passes of batch norm and layer norm with the right axis and the right shapes.
  - Explain why the gradient of one input has three routes, and write the one formula that collects them for both layers.
  - Check a backward pass against a finite difference, and choose a sensible nudge size.
prereqs:
  - 033-softmax-and-whole-network-gradients
  - 032-backpropagation
  - 035-signal-size-and-initialisation
exercise: 036-normalisation-layers
---

# Normalisation layers: batch norm, layer norm, and their gradients

A careful initialisation puts every layer's gain at 1 at step 0. Then training starts, and every update moves every weight. Nothing in the update keeps $n_{\text{in}} s^2$ at 1, so the size of the signal in layer 30 can drift from where you set it. A layer also sees more than a size. It sees features that sit at different places: in a grid of six examples, one feature might average $-0.7$ and another $6.2$, and the next layer has to cope with both.

Can a layer re-set its own signal to a fixed spread every time it runs? If it can, the drift stops mattering, because the correction repeats on every pass. The catch is that a layer has many numbers, and you must choose which of them to compare with which. This chapter builds the correction, shows that the choice of group decides almost everything about how it behaves, and then asks the question training needs answered: how does the gradient reach an input that touches every output of its group?

## What you already know

Four ideas carry this chapter:

- The mean of a group is its average. The variance is the average squared distance from the mean, and it measures spread.
- A layer's output for a batch is an array of shape `(N, D)`: $N$ examples in rows, $D$ features in columns. In numpy, `axis=0` averages each column and `axis=1` averages each row.
- Initialisation sets the scale once, before training (see [signal size and initialisation](/lessons/035-signal-size-and-initialisation)).
- A value used in several places collects the sum of the gradients that come back from each place, and each array operation has a small backward rule (see [backpropagation](/lessons/032-backpropagation)).

## Two ways to cut a grid into groups

Why not rescale the data once at the start? Only the first layer sees the data. The input of layer 5 is the output of four layers of weights that are still changing, so a fix applied once cannot hold. Why not rescale each layer by hand? The right factor changes as the weights change, so it has to be computed from the numbers on each pass.

To compute it, group the numbers and measure each group. Figure 1 shows the output of one layer for six examples as a grid. A column as a group compares one feature across examples, and a row as a group compares one example's features with each other. The first choice is **batch normalisation** and the second is **layer normalisation**. The arithmetic on a group is the same for both, so start there.

```figure
<svg viewBox="0 0 640 360" role="img" aria-label="A grid of 6 examples by 4 features. Batch norm groups a column. Layer norm groups a row.">
<text x="78.0" y="76.0" class="fig-small" text-anchor="middle">f1</text>
<text x="134.0" y="76.0" class="fig-small" text-anchor="middle">f2</text>
<text x="190.0" y="76.0" class="fig-small" text-anchor="middle">f3</text>
<text x="246.0" y="76.0" class="fig-small" text-anchor="middle">f4</text>
<text x="42.0" y="102.0" class="fig-small" text-anchor="end">1</text>
<text x="42.0" y="130.0" class="fig-small" text-anchor="end">2</text>
<text x="42.0" y="158.0" class="fig-small" text-anchor="end">3</text>
<text x="42.0" y="186.0" class="fig-small" text-anchor="end">4</text>
<text x="42.0" y="214.0" class="fig-small" text-anchor="end">5</text>
<text x="42.0" y="242.0" class="fig-small" text-anchor="end">6</text>
<rect x="50.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="78.0" y="102.0" class="fig-small" text-anchor="middle">0.0</text>
<rect x="106.0" y="84.0" width="56.0" height="28.0" class="fill-soft-a stroke-a"/>
<text x="134.0" y="102.0" class="fig-small" text-anchor="middle">3.4</text>
<rect x="162.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="190.0" y="102.0" class="fig-small" text-anchor="middle">-2.3</text>
<rect x="218.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="246.0" y="102.0" class="fig-small" text-anchor="middle">4.2</text>
<rect x="50.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="78.0" y="130.0" class="fig-small" text-anchor="middle">-0.5</text>
<rect x="106.0" y="112.0" width="56.0" height="28.0" class="fill-soft-a stroke-a"/>
<text x="134.0" y="130.0" class="fig-small" text-anchor="middle">1.5</text>
<rect x="162.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="190.0" y="130.0" class="fig-small" text-anchor="middle">-1.9</text>
<rect x="218.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="246.0" y="130.0" class="fig-small" text-anchor="middle">8.7</text>
<rect x="50.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="78.0" y="158.0" class="fig-small" text-anchor="middle">-0.5</text>
<rect x="106.0" y="140.0" width="56.0" height="28.0" class="fill-soft-a stroke-a"/>
<text x="134.0" y="158.0" class="fig-small" text-anchor="middle">2.1</text>
<rect x="162.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="190.0" y="158.0" class="fig-small" text-anchor="middle">-1.5</text>
<rect x="218.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="246.0" y="158.0" class="fig-small" text-anchor="middle">6.7</text>
<rect x="50.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="78.0" y="186.0" class="fig-small" text-anchor="middle">0.1</text>
<rect x="106.0" y="168.0" width="56.0" height="28.0" class="fill-soft-a stroke-a"/>
<text x="134.0" y="186.0" class="fig-small" text-anchor="middle">1.6</text>
<rect x="162.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="190.0" y="186.0" class="fig-small" text-anchor="middle">-2.0</text>
<rect x="218.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="246.0" y="186.0" class="fig-small" text-anchor="middle">7.4</text>
<rect x="50.0" y="196.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="78.0" y="214.0" class="fig-small" text-anchor="middle">-1.3</text>
<rect x="106.0" y="196.0" width="56.0" height="28.0" class="fill-soft-a stroke-a"/>
<text x="134.0" y="214.0" class="fig-small" text-anchor="middle">2.3</text>
<rect x="162.0" y="196.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="190.0" y="214.0" class="fig-small" text-anchor="middle">-3.9</text>
<rect x="218.0" y="196.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="246.0" y="214.0" class="fig-small" text-anchor="middle">3.4</text>
<rect x="50.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="78.0" y="242.0" class="fig-small" text-anchor="middle">-1.8</text>
<rect x="106.0" y="224.0" width="56.0" height="28.0" class="fill-soft-a stroke-a"/>
<text x="134.0" y="242.0" class="fig-small" text-anchor="middle">2.6</text>
<rect x="162.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="190.0" y="242.0" class="fig-small" text-anchor="middle">-3.3</text>
<rect x="218.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="246.0" y="242.0" class="fig-small" text-anchor="middle">6.5</text>
<text x="394.0" y="76.0" class="fig-small" text-anchor="middle">f1</text>
<text x="450.0" y="76.0" class="fig-small" text-anchor="middle">f2</text>
<text x="506.0" y="76.0" class="fig-small" text-anchor="middle">f3</text>
<text x="562.0" y="76.0" class="fig-small" text-anchor="middle">f4</text>
<text x="358.0" y="102.0" class="fig-small" text-anchor="end">1</text>
<text x="358.0" y="130.0" class="fig-small" text-anchor="end">2</text>
<text x="358.0" y="158.0" class="fig-small" text-anchor="end">3</text>
<text x="358.0" y="186.0" class="fig-small" text-anchor="end">4</text>
<text x="358.0" y="214.0" class="fig-small" text-anchor="end">5</text>
<text x="358.0" y="242.0" class="fig-small" text-anchor="end">6</text>
<rect x="366.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="394.0" y="102.0" class="fig-small" text-anchor="middle">0.0</text>
<rect x="422.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="450.0" y="102.0" class="fig-small" text-anchor="middle">3.4</text>
<rect x="478.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="506.0" y="102.0" class="fig-small" text-anchor="middle">-2.3</text>
<rect x="534.0" y="84.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="562.0" y="102.0" class="fig-small" text-anchor="middle">4.2</text>
<rect x="366.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="394.0" y="130.0" class="fig-small" text-anchor="middle">-0.5</text>
<rect x="422.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="450.0" y="130.0" class="fig-small" text-anchor="middle">1.5</text>
<rect x="478.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="506.0" y="130.0" class="fig-small" text-anchor="middle">-1.9</text>
<rect x="534.0" y="112.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="562.0" y="130.0" class="fig-small" text-anchor="middle">8.7</text>
<rect x="366.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="394.0" y="158.0" class="fig-small" text-anchor="middle">-0.5</text>
<rect x="422.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="450.0" y="158.0" class="fig-small" text-anchor="middle">2.1</text>
<rect x="478.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="506.0" y="158.0" class="fig-small" text-anchor="middle">-1.5</text>
<rect x="534.0" y="140.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="562.0" y="158.0" class="fig-small" text-anchor="middle">6.7</text>
<rect x="366.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="394.0" y="186.0" class="fig-small" text-anchor="middle">0.1</text>
<rect x="422.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="450.0" y="186.0" class="fig-small" text-anchor="middle">1.6</text>
<rect x="478.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="506.0" y="186.0" class="fig-small" text-anchor="middle">-2.0</text>
<rect x="534.0" y="168.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="562.0" y="186.0" class="fig-small" text-anchor="middle">7.4</text>
<rect x="366.0" y="196.0" width="56.0" height="28.0" class="fill-soft-b stroke-b"/>
<text x="394.0" y="214.0" class="fig-small" text-anchor="middle">-1.3</text>
<rect x="422.0" y="196.0" width="56.0" height="28.0" class="fill-soft-b stroke-b"/>
<text x="450.0" y="214.0" class="fig-small" text-anchor="middle">2.3</text>
<rect x="478.0" y="196.0" width="56.0" height="28.0" class="fill-soft-b stroke-b"/>
<text x="506.0" y="214.0" class="fig-small" text-anchor="middle">-3.9</text>
<rect x="534.0" y="196.0" width="56.0" height="28.0" class="fill-soft-b stroke-b"/>
<text x="562.0" y="214.0" class="fig-small" text-anchor="middle">3.4</text>
<rect x="366.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="394.0" y="242.0" class="fig-small" text-anchor="middle">-1.8</text>
<rect x="422.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="450.0" y="242.0" class="fig-small" text-anchor="middle">2.6</text>
<rect x="478.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="506.0" y="242.0" class="fig-small" text-anchor="middle">-3.3</text>
<rect x="534.0" y="224.0" width="56.0" height="28.0" class="fill-paper stroke-grid"/>
<text x="562.0" y="242.0" class="fig-small" text-anchor="middle">6.5</text>
<text x="162.0" y="26.0" class="fig-bold" text-anchor="middle">Batch norm: each group is a column</text>
<text x="478.0" y="26.0" class="fig-bold" text-anchor="middle">Layer norm: each group is a row</text>
<text x="162.0" y="46.0" class="fig-small" text-anchor="middle">one feature across the 6 examples</text>
<text x="478.0" y="46.0" class="fig-small" text-anchor="middle">one example across its 4 features</text>
<text x="40.0" y="276.0" class="fig-small" text-anchor="end">mean</text>
<text x="600.0" y="76.0" class="fig-small" text-anchor="start">mean</text>
<text x="78.0" y="276.0" class="fig-small fig-bold" text-anchor="middle">-0.67</text>
<text x="134.0" y="276.0" class="fig-small fig-bold" text-anchor="middle">2.25</text>
<text x="190.0" y="276.0" class="fig-small fig-bold" text-anchor="middle">-2.48</text>
<text x="246.0" y="276.0" class="fig-small fig-bold" text-anchor="middle">6.15</text>
<text x="600.0" y="102.0" class="fig-small" text-anchor="start">1.33</text>
<text x="600.0" y="130.0" class="fig-small" text-anchor="start">1.95</text>
<text x="600.0" y="158.0" class="fig-small" text-anchor="start">1.70</text>
<text x="600.0" y="186.0" class="fig-small" text-anchor="start">1.78</text>
<text x="600.0" y="214.0" class="fig-small" text-anchor="start">0.12</text>
<text x="600.0" y="242.0" class="fig-small" text-anchor="start">1.00</text>
<text x="162.0" y="310.0" class="fig-small" text-anchor="middle">feature 2 has mean 2.25 and spread 0.64</text>
<text x="478.0" y="310.0" class="fig-small" text-anchor="middle">example 5 has mean 0.12 and spread 2.90</text>
</svg>
Figure 1. The output of one layer for 6 examples is a grid with one row per example and one column per feature. Batch norm (left) takes a column as its group, so it compares an example with the other examples on the same feature. Layer norm (right) takes a row as its group, so it compares a feature with the other features of the same example. The numbers are real, and the shaded groups are the ones the text normalises.
```

## Normalise one group

Take the group $[1, 2, 3]$. Its mean is $2$. Subtract the mean and the gaps are $-1, 0, 1$. Their squares are $1, 0, 1$, so the variance is $2/3 = 0.6667$, and the spread, its square root, is $0.8165$. Divide each gap by the spread and you get $-1.2247$, $0$, $1.2247$, a group with mean 0 and variance 1.

The result does not depend on where the group started or how big it was, because subtracting the mean removes the position and dividing by the spread removes the size. The groups $[10, 20, 30]$ and $[38.5, 39.5, 40.5]$ give the same three numbers. The division can fail, though: $[5, 5, 5]$ has variance 0, so the quotient is $0/0$. Adding a tiny constant $\epsilon$, such as $10^{-5}$, to the variance before the square root keeps the division finite, and the output is $0, 0, 0$. For a group of $M$ numbers $x_1, \dots, x_M$:

$$
\mu = \frac{1}{M}\sum_{i=1}^{M} x_i, \qquad
\sigma^2 = \frac{1}{M}\sum_{i=1}^{M} (x_i - \mu)^2, \qquad
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Read it aloud: subtract the group's mean, then divide by its spread, with a little extra under the square root. The hat on $\hat{x}$ marks a normalised value, and the variance divides by $M$. In numpy:

```python
xhat = (x - x.mean()) / np.sqrt(x.var() + eps)
```

Forcing every group to mean 0 and variance 1 may not be what the next layer wants. If a ReLU follows, a group centred on 0 has about half of its values cut to zero. So after normalising, give each feature two numbers it can learn, a scale $\gamma$ (gamma) and a shift $\beta$ (beta):

$$
y_i = \gamma\, \hat{x}_i + \beta
$$

Read it aloud: stretch by gamma, then slide by beta. On the group above with $\gamma = 2$ and $\beta = 1$ the outputs are $-1.449, 1, 3.449$. Why add them back right after removing the scale? Because they keep the layer at least as expressive as before. Set $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$, and $y_i = x_i$ exactly, so the layer can undo the normalisation when that is best. What changes is that the position and size of the output come from two numbers $\gamma$ and $\beta$, and not from the layers before it.

## Batch norm and layer norm in numpy

The two layers differ only in which numbers form a group. With `x` of shape `(N, D)`, the batch norm group is a column, so $M = N$ and you reduce over `axis=0`. The layer norm group is a row, so $M = D$ and you reduce over `axis=1`. In both, `gamma` and `beta` have shape `(D,)`, one pair per feature, and broadcast across the rows.

```python
mean = x.mean(axis=axis, keepdims=True)
var = x.var(axis=axis, keepdims=True)
istd = 1.0 / np.sqrt(var + eps)        # the inverse standard deviation
xhat = (x - mean) * istd
out = gamma * xhat + beta
```

Figure 1 has the numbers. Feature 2 of the batch-norm grid is $[3.4, 1.5, 2.1, 1.6, 2.3, 2.6]$, with mean 2.25 and spread 0.64, so it normalises to $[1.80, -1.17, -0.23, -1.02, 0.08, 0.55]$. Example 5 of the layer-norm grid is $[-1.3, 2.3, -3.9, 3.4]$, with mean 0.125 and spread 2.90, so it normalises to $[-0.49, 0.75, -1.39, 1.13]$. The `keepdims=True` argument leaves a length-1 axis, so `mean` has shape `(1, D)` for batch norm and `(N, 1)` for layer norm and lines up with `x`.

> [!TIP]
> Print `mean.shape` while you debug. A wrong axis gives `(N, 1)` where you expected `(1, D)`, and numpy may still broadcast without an error.

The choice of group has three consequences.

- **A shifted example.** Add the same constant to every feature of one example. Layer norm compares that example only with itself, so the constant cancels and the output does not change. Batch norm compares each feature with the other examples, so the shifted example becomes an outlier in every column, and its output moves. On a 10 by 6 matrix with a shift of 37.5, all six batch-norm outputs of that row jump to about 3.
- **A batch of one.** Each column then holds one number, which is its own mean, so every gap and every variance is 0, and batch norm returns $\hat{x} = 0$, so $y = \beta$ whatever the values. Layer norm on $[1, 2, 3, 4]$ still gives $[-1.34, -0.45, 0.45, 1.34]$.
- **Test time.** Batch norm's output for one example depends on which others share its batch, which is wrong for a prediction. So batch norm keeps a running average of each column's mean and variance during training and uses those stored numbers at test time. Layer norm uses only the example's own features, so it has nothing to store. That is one reason recurrent networks and transformers usually use layer norm.

> [!KEY]
> Initialisation sets the scale once and training moves it away, so a layer can instead re-set its signal on every pass: subtract the group mean, divide by the group spread, then apply a learned scale and shift that let the layer undo it. Batch norm takes a column as its group and layer norm takes a row, and that choice decides what happens with a shifted example, a batch of one, and test time.

## Why the gradient has three routes

Run layer norm on the row $[1, 2, 3]$, whose outputs are $[-1.2247, 0, 1.2247]$. Suppose the loss reads only the first output, and you nudge $x_1$ up by a tiny amount. The first output rises, which is the obvious part. But $x_1$ also belongs to the average, so the mean rises a little, and every output is measured from the mean. It belongs to the variance too, so the spread changes, and every output is divided by the spread. One nudge reaches all three outputs.

Why not treat the mean and spread as constants? Then $\hat{x}_i = (x_i - \mu)\,\text{istd}$ is a plain rescaling, where $\text{istd} = 1/\sqrt{\sigma^2 + \epsilon}$ is $1.2247$ for this row, and with upstream gradient $[1, 0, 0]$ the gradient would be $[1.2247, 0, 0]$. The true gradient, derived below and confirmed by a finite difference, is $[0.2041, -0.4082, 0.2041]$. The shortcut is six times too large in the first entry and misses the other two, because raising $x_1$ raises the mean, which pulls all three outputs down, and changes the spread. Figure 2 draws the three routes from one input to the outputs.

```figure
<svg viewBox="0 0 640 360" role="img" aria-label="Three routes from one input x_k to the outputs of its group: direct, through the mean, and through the variance">
<defs><marker id="c031-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c031-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c031-arrow-d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<line x1="100.0" y1="170.0" x2="506.0" y2="170.0" class="stroke-a" marker-end="url(#c031-arrow-a)"/>
<line x1="92.0" y1="150.0" x2="264.0" y2="66.0" class="stroke-b" marker-end="url(#c031-arrow-b)"/>
<line x1="316.0" y1="64.0" x2="510.0" y2="78.0" class="stroke-b" marker-end="url(#c031-arrow-b)"/>
<line x1="316.0" y1="64.0" x2="510.0" y2="158.0" class="stroke-b" marker-end="url(#c031-arrow-b)"/>
<line x1="316.0" y1="64.0" x2="510.0" y2="238.0" class="stroke-b" marker-end="url(#c031-arrow-b)"/>
<line x1="92.0" y1="190.0" x2="264.0" y2="274.0" class="stroke-d" marker-end="url(#c031-arrow-d)"/>
<line x1="316.0" y1="276.0" x2="510.0" y2="102.0" class="stroke-d" marker-end="url(#c031-arrow-d)"/>
<line x1="316.0" y1="276.0" x2="510.0" y2="182.0" class="stroke-d" marker-end="url(#c031-arrow-d)"/>
<line x1="316.0" y1="276.0" x2="510.0" y2="262.0" class="stroke-d" marker-end="url(#c031-arrow-d)"/>
<rect x="45.0" y="154.0" width="50.0" height="32.0" class="fill-paper stroke-a" rx="6"/>
<text x="70.0" y="175.0" class="fig-bold" text-anchor="middle">x_k</text>
<rect x="265.0" y="44.0" width="50.0" height="32.0" class="fill-paper stroke-b" rx="6"/>
<text x="290.0" y="65.0" class="fig-bold" text-anchor="middle">μ</text>
<rect x="265.0" y="264.0" width="50.0" height="32.0" class="fill-paper stroke-d" rx="6"/>
<text x="290.0" y="285.0" class="fig-bold" text-anchor="middle">σ²</text>
<rect x="512.0" y="74.0" width="56.0" height="32.0" class="fill-paper stroke-muted" rx="6"/>
<text x="540.0" y="95.0" class="fig-bold" text-anchor="middle">x̂_1</text>
<rect x="512.0" y="154.0" width="56.0" height="32.0" class="fill-paper stroke-muted" rx="6"/>
<text x="540.0" y="175.0" class="fig-bold" text-anchor="middle">x̂_k</text>
<rect x="512.0" y="234.0" width="56.0" height="32.0" class="fill-paper stroke-muted" rx="6"/>
<text x="540.0" y="255.0" class="fig-bold" text-anchor="middle">x̂_M</text>
<text x="540.0" y="50.0" class="fig-small" text-anchor="middle">every output in the group</text>
<text x="165.0" y="196.0" class="fig-small" text-anchor="start">direct</text>
<text x="165.0" y="213.0" class="fig-small" text-anchor="start">istd · dx̂_k</text>
<text x="150.0" y="60.0" class="fig-small" text-anchor="middle">through the mean</text>
<text x="150.0" y="77.0" class="fig-small" text-anchor="middle">−istd/M · Σ dx̂</text>
<text x="150.0" y="322.0" class="fig-small" text-anchor="middle">through the variance</text>
<text x="150.0" y="339.0" class="fig-small" text-anchor="middle">−istd/M · x̂_k · Σ dx̂·x̂</text>
</svg>
Figure 2. One input x_k reaches the outputs by three routes. The blue route is direct: x_k changes its own x̂_k. The orange route goes through the group mean μ, which every output uses. The purple route goes through the group variance σ², which every output also uses. The gradient of x_k is the sum of the three routes, and each label is that route's term in the formula.
```

Work from the output backward. The layer computes $y = \gamma \hat{x} + \beta$, and the loss sends back `dout`, the same shape as the output:

- The shift $\beta$ is added to every output, so its gradient is the sum of `dout`.
- The scale $\gamma$ multiplies $\hat{x}$, so its gradient is the sum of `dout * xhat`.
- The gradient at $\hat{x}$ is `dxhat = dout * gamma`.

The sums for $\gamma$ and $\beta$ run over examples, `axis=0`, for batch norm and for layer norm. The reason is that $\gamma_j$ is one number shared by all $N$ examples in column $j$, and a value used $N$ times collects the sum of $N$ gradients. This holds for layer norm too, even though it normalises along rows. Take two examples, $[1, 2, 3]$ and $[10, 20, 30]$, which normalise to the same $\hat{x}$, and put a gradient of 1 on the first feature of each. Then $d\beta_1 = 1 + 1 = 2$ and $d\gamma_1 = 2 \cdot (-1.2247) = -2.449$.

```python
dbeta = dout.sum(axis=0)
dgamma = (dout * xhat).sum(axis=0)
dxhat = dout * gamma
```

Next comes the hard part: from `dxhat` back to `x`. Take one input $x_k$ in a group of $M$ numbers and ask how it changes each $\hat{x}_i$. There are three routes.

1. **Directly.** Only $\hat{x}_k$ contains $x_k$ by itself, and it changes by $\text{istd}$ per unit of $x_k$.
2. **Through the mean.** Raising $x_k$ by 1 raises $\mu$ by $1/M$. Every $\hat{x}_i$ is measured from $\mu$, so each one falls by $\text{istd}/M$.
3. **Through the variance.** Raising $x_k$ by 1 changes $\sigma^2$ by $2(x_k - \mu)/M$. (The part that passes through $\mu$ adds to zero, because the gaps from the mean sum to zero.) Each $\hat{x}_i$ changes by $-\tfrac{1}{2}\hat{x}_i\,\text{istd}^2$ per unit of $\sigma^2$, so the route changes it by $-\hat{x}_i \hat{x}_k\,\text{istd}/M$, using $(x_k - \mu)\,\text{istd} = \hat{x}_k$.

Weight each route by `dxhat` and add. The direct route contributes $\text{istd}\cdot d\hat{x}_k$, the mean route $-\frac{\text{istd}}{M}\sum_i d\hat{x}_i$, and the variance route $-\frac{\text{istd}}{M}\,\hat{x}_k \sum_i d\hat{x}_i \hat{x}_i$. Their sum, over a common factor, is

$$
dx_k = \frac{\text{istd}}{M}\Big(M\, d\hat{x}_k - \sum_i d\hat{x}_i - \hat{x}_k \sum_i d\hat{x}_i\, \hat{x}_i\Big)
$$

Read it aloud: take $M$ times the gradient at $\hat{x}_k$, subtract the group's total gradient, which undoes the mean, subtract $\hat{x}_k$ times the group's total of gradient times $\hat{x}$, which undoes the variance, and scale by istd over $M$. On the row $[1, 2, 3]$ with $d\hat{x} = [1, 0, 0]$ we have $M = 3$, $\text{istd} = 1.2247$, $\sum d\hat{x} = 1$, and $\sum d\hat{x}\,\hat{x} = -1.2247$. The routes for each input are in this table, and Figure 3 draws them.

| Input | Direct | Through mean | Through variance | Sum |
|---|---|---|---|---|
| $x_1$ | $1.2247$ | $-0.4082$ | $-0.6124$ | $0.2041$ |
| $x_2$ | $0$ | $-0.4082$ | $0$ | $-0.4082$ |
| $x_3$ | $0$ | $-0.4082$ | $0.6124$ | $0.2041$ |

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Four bar charts for the row [1, 2, 3] with upstream gradient [1, 0, 0]: the direct term, the mean term, the variance term, and their sum">
<line x1="26.0" y1="150.0" x2="146.0" y2="150.0" class="stroke-muted"/>
<rect x="36.0" y="84.1" width="24.0" height="65.9" class="fill-a"/>
<text x="48.0" y="78.1" class="fig-small" text-anchor="middle">1.22</text>
<text x="48.0" y="240.0" class="fig-small" text-anchor="middle">x1</text>
<rect x="72.0" y="150.0" width="24.0" height="0.8" class="fill-a"/>
<text x="84.0" y="144.0" class="fig-small" text-anchor="middle">0.00</text>
<text x="84.0" y="240.0" class="fig-small" text-anchor="middle">x2</text>
<rect x="108.0" y="150.0" width="24.0" height="0.8" class="fill-a"/>
<text x="120.0" y="144.0" class="fig-small" text-anchor="middle">0.00</text>
<text x="120.0" y="240.0" class="fig-small" text-anchor="middle">x3</text>
<text x="86.0" y="60.0" class="fig-bold" text-anchor="middle">direct</text>
<text x="163.0" y="154.0" class="fig-bold" text-anchor="middle">+</text>
<line x1="184.0" y1="150.0" x2="304.0" y2="150.0" class="stroke-muted"/>
<rect x="194.0" y="150.0" width="24.0" height="22.0" class="fill-b"/>
<text x="206.0" y="188.0" class="fig-small" text-anchor="middle">−0.41</text>
<text x="206.0" y="240.0" class="fig-small" text-anchor="middle">x1</text>
<rect x="230.0" y="150.0" width="24.0" height="22.0" class="fill-b"/>
<text x="242.0" y="188.0" class="fig-small" text-anchor="middle">−0.41</text>
<text x="242.0" y="240.0" class="fig-small" text-anchor="middle">x2</text>
<rect x="266.0" y="150.0" width="24.0" height="22.0" class="fill-b"/>
<text x="278.0" y="188.0" class="fig-small" text-anchor="middle">−0.41</text>
<text x="278.0" y="240.0" class="fig-small" text-anchor="middle">x3</text>
<text x="244.0" y="60.0" class="fig-bold" text-anchor="middle">through mean</text>
<text x="321.0" y="154.0" class="fig-bold" text-anchor="middle">+</text>
<line x1="342.0" y1="150.0" x2="462.0" y2="150.0" class="stroke-muted"/>
<rect x="352.0" y="150.0" width="24.0" height="33.0" class="fill-d"/>
<text x="364.0" y="199.0" class="fig-small" text-anchor="middle">−0.61</text>
<text x="364.0" y="240.0" class="fig-small" text-anchor="middle">x1</text>
<rect x="388.0" y="150.0" width="24.0" height="0.8" class="fill-d"/>
<text x="400.0" y="144.0" class="fig-small" text-anchor="middle">0.00</text>
<text x="400.0" y="240.0" class="fig-small" text-anchor="middle">x2</text>
<rect x="424.0" y="117.0" width="24.0" height="33.0" class="fill-d"/>
<text x="436.0" y="111.0" class="fig-small" text-anchor="middle">0.61</text>
<text x="436.0" y="240.0" class="fig-small" text-anchor="middle">x3</text>
<text x="402.0" y="60.0" class="fig-bold" text-anchor="middle">through variance</text>
<text x="479.0" y="154.0" class="fig-bold" text-anchor="middle">=</text>
<line x1="500.0" y1="150.0" x2="620.0" y2="150.0" class="stroke-muted"/>
<rect x="510.0" y="139.0" width="24.0" height="11.0" class="fill-c"/>
<text x="522.0" y="133.0" class="fig-small" text-anchor="middle">0.20</text>
<text x="522.0" y="240.0" class="fig-small" text-anchor="middle">x1</text>
<rect x="546.0" y="150.0" width="24.0" height="22.0" class="fill-c"/>
<text x="558.0" y="188.0" class="fig-small" text-anchor="middle">−0.41</text>
<text x="558.0" y="240.0" class="fig-small" text-anchor="middle">x2</text>
<rect x="582.0" y="139.0" width="24.0" height="11.0" class="fill-c"/>
<text x="594.0" y="133.0" class="fig-small" text-anchor="middle">0.20</text>
<text x="594.0" y="240.0" class="fig-small" text-anchor="middle">x3</text>
<text x="560.0" y="60.0" class="fig-bold" text-anchor="middle">sum = dx</text>
<text x="320.0" y="284.0" class="fig-small" text-anchor="middle">the three bars of the sum add to 0, because shifting the whole group changes nothing</text>
</svg>
Figure 3. The gradient of the three inputs of the row [1, 2, 3], when the loss only looks at the first output (upstream gradient [1, 0, 0]). The direct route (blue) says only x1 matters. The mean route (orange) pulls all three down equally. The variance route (purple) pulls x1 down and x3 up, because x1 is below the mean and x3 above it. The three bars of the sum, 0.20, −0.41, and 0.20, add to zero (to four places, 0.2041, −0.4082, and 0.2041).
```

In numpy the same code serves both layers, and only `axis` and $M$ change: `axis=0` and $M = N$ for batch norm, `axis=1` and $M = D$ for layer norm.

```python
m = dout.shape[axis]
dx = (istd / m) * (
    m * dxhat
    - dxhat.sum(axis=axis, keepdims=True)
    - xhat * (dxhat * xhat).sum(axis=axis, keepdims=True)
)
```

> [!WARNING]
> For layer norm, `dx` reduces over `axis=1`, but `dgamma` and `dbeta` still sum over `axis=0`. Summing them over the normalised axis gives shape `(N,)` and breaks the update of a `(D,)` parameter.

The formula has two properties you can use as sanity checks. First, $dx$ sums to zero over each group. Add the same constant to every input of a group and the mean moves with it and nothing else changes, so the loss cannot change and the gradients must cancel. In the table, $0.2041 - 0.4082 + 0.2041 = 0$. Second, $dx$ is orthogonal to $\hat{x}$, meaning $\sum_k dx_k\,\hat{x}_k$ is zero. Scale all inputs of a group by a constant and the spread scales with them, so the outputs do not change. The sum is exactly zero when $\epsilon = 0$, and tiny with the usual $\epsilon = 10^{-5}$.

## Check it with a finite difference

A formula with three terms is easy to get slightly wrong, and a slightly wrong gradient still trains. The loss goes down worse than it should. So check it against a number you can compute without the formula. The **finite difference** nudges one input and measures the effect directly. For the row $[1, 2, 3]$ with `dout` $= [1, 0, 0]$, differentiate the scalar $f(x_1) = \sum_j \text{dout}_j \cdot y_j$, a loss whose gradient at the output is exactly `dout`. Nudge $x_1$ by $h = 10^{-6}$ each way:

$$
\frac{f(x_1 + h) - f(x_1 - h)}{2h} = 0.20413
$$

That agrees with the formula's $0.2041$. For arrays, nudge each entry in turn and compare with the formula's entry. How big should $h$ be? Figure 4 shows the largest error over a 5 by 7 layer as $h$ shrinks. The error is smallest near $h = 10^{-5}$ and rises on both sides: a big $h$ is too coarse to follow the curve, and a tiny $h$ lets round-off in the difference take over. So $10^{-6}$ is a safe choice, $10^{-14}$ is useless, and a tolerance such as $10^{-4}$ is how the checks decide.

```figure
<svg viewBox="0 0 620 350" role="img" aria-label="Error of the finite-difference gradient against nudge size h on log axes, with a tolerance line at 1e-4">
<line x1="70.0" y1="290.0" x2="500.0" y2="290.0" class="stroke-muted"/>
<line x1="70.0" y1="290.0" x2="70.0" y2="30.0" class="stroke-grid"/>
<line x1="70.0" y1="268.3" x2="500.0" y2="268.3" class="stroke-grid"/>
<text x="62.0" y="272.3" class="fig-small" text-anchor="end">10⁻¹⁰</text>
<line x1="70.0" y1="225.0" x2="500.0" y2="225.0" class="stroke-grid"/>
<text x="62.0" y="229.0" class="fig-small" text-anchor="end">10⁻⁸</text>
<line x1="70.0" y1="181.7" x2="500.0" y2="181.7" class="stroke-grid"/>
<text x="62.0" y="185.7" class="fig-small" text-anchor="end">10⁻⁶</text>
<line x1="70.0" y1="138.3" x2="500.0" y2="138.3" class="stroke-grid"/>
<text x="62.0" y="142.3" class="fig-small" text-anchor="end">10⁻⁴</text>
<line x1="70.0" y1="95.0" x2="500.0" y2="95.0" class="stroke-grid"/>
<text x="62.0" y="99.0" class="fig-small" text-anchor="end">10⁻²</text>
<line x1="70.0" y1="51.7" x2="500.0" y2="51.7" class="stroke-grid"/>
<text x="62.0" y="55.7" class="fig-small" text-anchor="end">10⁰</text>
<line x1="500.0" y1="290.0" x2="500.0" y2="295.0" class="stroke-muted"/>
<text x="500.0" y="310.0" class="fig-small" text-anchor="middle">10⁻¹</text>
<line x1="433.8" y1="290.0" x2="433.8" y2="295.0" class="stroke-muted"/>
<text x="433.8" y="310.0" class="fig-small" text-anchor="middle">10⁻³</text>
<line x1="367.7" y1="290.0" x2="367.7" y2="295.0" class="stroke-muted"/>
<text x="367.7" y="310.0" class="fig-small" text-anchor="middle">10⁻⁵</text>
<line x1="301.5" y1="290.0" x2="301.5" y2="295.0" class="stroke-muted"/>
<text x="301.5" y="310.0" class="fig-small" text-anchor="middle">10⁻⁷</text>
<line x1="235.4" y1="290.0" x2="235.4" y2="295.0" class="stroke-muted"/>
<text x="235.4" y="310.0" class="fig-small" text-anchor="middle">10⁻⁹</text>
<line x1="169.2" y1="290.0" x2="169.2" y2="295.0" class="stroke-muted"/>
<text x="169.2" y="310.0" class="fig-small" text-anchor="middle">10⁻¹¹</text>
<line x1="103.1" y1="290.0" x2="103.1" y2="295.0" class="stroke-muted"/>
<text x="103.1" y="310.0" class="fig-small" text-anchor="middle">10⁻¹³</text>
<text x="285.0" y="332.0" class="fig-small" text-anchor="middle">nudge size h</text>
<text x="62.0" y="20.0" class="fig-small" text-anchor="start">largest error over all entries</text>
<line x1="70.0" y1="138.3" x2="500.0" y2="138.3" class="stroke-c dashed"/>
<text x="285.0" y="130.3" class="fig-small" text-anchor="middle">tolerance 10⁻⁴</text>
<polyline points="500.0,93.8 466.9,137.1 433.8,180.4 400.8,223.7 367.7,261.1 334.6,240.8 301.5,217.0 268.5,196.8 235.4,174.3 202.3,152.3 169.2,131.8 136.2,110.8 103.1,90.5 70.0,63.9" fill="none" class="stroke-b"/>
<circle cx="500.0" cy="93.8" r="3.5" class="fill-b"/>
<circle cx="466.9" cy="137.1" r="3.5" class="fill-b"/>
<circle cx="433.8" cy="180.4" r="3.5" class="fill-b"/>
<circle cx="400.8" cy="223.7" r="3.5" class="fill-b"/>
<circle cx="367.7" cy="261.1" r="3.5" class="fill-b"/>
<circle cx="334.6" cy="240.8" r="3.5" class="fill-b"/>
<circle cx="301.5" cy="217.0" r="3.5" class="fill-b"/>
<circle cx="268.5" cy="196.8" r="3.5" class="fill-b"/>
<circle cx="235.4" cy="174.3" r="3.5" class="fill-b"/>
<circle cx="202.3" cy="152.3" r="3.5" class="fill-b"/>
<circle cx="169.2" cy="131.8" r="3.5" class="fill-b"/>
<circle cx="136.2" cy="110.8" r="3.5" class="fill-b"/>
<circle cx="103.1" cy="90.5" r="3.5" class="fill-b"/>
<circle cx="70.0" cy="63.9" r="3.5" class="fill-b"/>
<text x="367.7" y="283.1" class="fig-small" text-anchor="middle">best: h = 10⁻5</text>
</svg>
Figure 4. The largest gap between the formula's dx and a central finite difference, for a 5 by 7 layer-norm layer, as the nudge h shrinks. For big h the nudge is too coarse to follow the curve, so the gap falls as h shrinks. For tiny h the two function values agree in almost every digit, and round-off in their difference takes over. The gap is lowest near h = 10⁻⁵, and the exercise's checks use 10⁻⁶, well inside the tolerance.
```

> [!KEY]
> Every input of a group affects every output through the shared mean and variance, so its gradient is the sum of three routes: direct, through the mean, and through the variance. They collapse into one formula that is the same for batch norm and layer norm apart from the axis. You trust it because a finite difference agrees with it, its group sums to zero, and it is orthogonal to $\hat{x}$.

> [!RECAP]
> - Normalising a group subtracts its mean and divides by $\sqrt{\sigma^2 + \epsilon}$, then applies a learned `gamma` and `beta` of shape `(D,)`. The constant $\epsilon$ keeps a constant group from giving $0/0$.
> - Batch norm reduces over `axis=0` and layer norm over `axis=1`, with `keepdims=True`. The cache is `(xhat, istd, gamma, axis)`.
> - `dbeta = dout.sum(axis=0)` and `dgamma = (dout * xhat).sum(axis=0)` for both layers, because $\gamma$ and $\beta$ are shared by every example.
> - With `dxhat = dout * gamma`, `dx = (istd / M) * (M * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))`, summing over the normalised axis.
> - `dx` sums to zero over each group and is orthogonal to `xhat`. Check it against a central finite difference with $h$ near $10^{-6}$.

## Build it step by step

You write six functions, in the order the checks test them. Two helpers, one forward and one backward, can serve both layers.

1. **`normalise_group(x, eps)`** takes an array of any shape, treats every entry as one group, and returns the same shape: `(x - x.mean()) / np.sqrt(x.var() + eps)`. The usual mistake is to forget `eps`, which turns a constant group into `nan`.
2. **`batchnorm_forward(x, gamma, beta, eps)`** takes `x` of shape `(N, D)` and returns `(out, cache)`. Reduce over `axis=0` with `keepdims=True`, and store the tuple `(xhat, istd, gamma, axis)` as the cache. The usual mistake is to drop `keepdims=True` and subtract a length-`D` mean from the wrong axis.
3. **`layernorm_forward(x, gamma, beta, eps)`** applies the same arithmetic over `axis=1`, so `istd` has shape `(N, 1)`. The usual mistake is to reduce over columns, which makes it batch norm.
4. **`affine_param_grads(dout, xhat)`** takes two arrays of shape `(N, D)` and returns `(dgamma, dbeta)`, each of shape `(D,)`, both summed over `axis=0`. The usual mistake is to sum over the normalised axis for layer norm.
5. **`batchnorm_backward(dout, cache)`** reads `(xhat, istd, gamma, axis)` from the cache and returns `(dx, dgamma, dbeta)`. Compute `dxhat = dout * gamma`, then apply the formula with `M = dout.shape[axis]`. The usual mistake is a sign error on the variance term, which a finite difference catches and which stops `dx` summing to zero.
6. **`layernorm_backward(dout, cache)`** is the same formula, and the cache holds `axis=1`. The usual mistake is to hard-code `axis=0` in the shared helper, which makes layer norm silently wrong.

## Cost and scaling

The backward pass reads the cached `xhat` and `istd` and makes a few passes over an `(N, D)` array, so its time grows in a straight line with the number of values, as the forward pass does. It stores nothing beyond the forward cache, which is about the size of `x`, and next to the matrix multiplies of the surrounding layers the cost is small.

## Where this breaks

- A small batch makes batch norm's column statistics noisy, and a batch of one makes them useless. Use layer norm, or group norm, which splits each row into a few groups. A trained network that uses batch norm must also switch to stored statistics at test time.
- Layer norm treats all features of an example as comparable. If one feature is on a very different scale, it dominates the row's mean and spread.
- A very small $\epsilon$ lets a near-constant group blow up its noise and its gradient, and a large one weakens the normalisation.
- The backward formula assumes the forward pass used exactly the `axis`, `eps`, and variance definition you cached. If the forward divides the variance by $M - 1$, the backward must too.
- A finite difference makes two forward passes for every entry, so use it on a small array as a test and not in training.

## The exercise

You write `normalise_group`, `batchnorm_forward`, `layernorm_forward`, `affine_param_grads`, `batchnorm_backward`, and `layernorm_backward`.

The forward checks verify that `[1, 2, 3]` becomes about $[-1.2247, 0, 1.2247]$, that the result ignores shift and scale, and that a constant group stays finite. They check mean 0 and variance 1 on the right axis for each layer and the output $\gamma\hat{x} + \beta$. They check that a batch of one gives $\beta$ under batch norm, and that shifting one example leaves layer norm unchanged and moves batch norm. The backward checks verify the scale and shift sums, the hand example with $dx = [0.2041, -0.4082, 0.2041]$, and both backward passes against finite differences for $x$, $\gamma$, and $\beta$. They also check that `dx` sums to zero and is orthogonal to `xhat`, and that layer norm's `dx` equals batch norm's `dx` on the transposed input.
