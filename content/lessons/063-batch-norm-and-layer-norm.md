---
title: "Normalising activations: batch norm versus layer norm"
slug: 063-batch-norm-and-layer-norm
part: "Part 4 — Neural networks"
summary: Rescale each group of numbers to mean 0 and variance 1 on every pass, then let the layer undo it. Batch norm groups columns, layer norm rows.
minutes: 35
gpu: false
objectives:
  - Normalise a group of numbers by hand, and say why the variance gets a small constant added and why a learned scale and shift follow.
  - Explain what batch norm and layer norm each average over, using the grid of examples by features.
  - Build the forward passes of batch norm and layer norm with the right axis and the right shapes.
  - Predict what a shifted example and a batch of one do to each layer.
prereqs:
  - 057-backprop-mlp-and-gradient-check
  - 061-signal-size-through-layers
  - 062-xavier-and-he
exercise: 063-batch-norm-and-layer-norm
---

# Normalising activations: batch norm versus layer norm

A careful initialisation puts every layer's gain at 1 at step 0. Then training starts, and every update moves every weight. Nothing in the update keeps $n_{\text{in}} s^2$ at 1, so the size of the signal in layer 30 can drift from where you set it. A layer also does not just see a size. It sees features that sit at different places: in a grid of six examples, one feature might average $-0.7$ and another $6.2$, and the next layer has to cope with both.

Can a layer re-set its own signal to a fixed spread every time it runs? If it can, the drift stops mattering, because the correction is repeated on every pass. The catch is that a layer has many numbers, and you must choose which of them to compare with which. This chapter builds the correction, and then shows that the choice of group decides almost everything about how it behaves.

## What you already know

Four ideas carry this chapter:

- The mean of a group is its average. The variance is the average squared distance from the mean, and it measures spread.
- A layer's output for a batch is an array of shape `(N, D)`: $N$ examples in rows, $D$ features in columns.
- Initialisation sets the scale once, before training (see "Xavier and He initialisation").
- In numpy, `axis=0` runs down the rows, so it averages each column, and `axis=1` runs along the columns, so it averages each row.

## Two ways to cut a grid into groups

Why not rescale the data once at the start, and be done? Only the first layer sees the data. The input of layer 5 is the output of four layers of weights that are still changing, so a fix applied once at the start cannot hold. Why not rescale each layer by hand? The right factor changes as the weights change, so it has to be computed from the numbers on each pass.

To compute it, group the numbers and measure each group. Figure 1 shows the output of one layer for six examples as a grid. You can take a column as a group, which compares one feature across examples, or a row, which compares one example's features with each other. The first choice is **batch normalisation** and the second is **layer normalisation**. The arithmetic on a group is the same for both, so start there.

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

Take the group $[1, 2, 3]$. Its mean is $(1 + 2 + 3)/3 = 2$. Subtract the mean and the gaps are $-1, 0, 1$. Their squares are $1, 0, 1$, so the variance is $2/3 = 0.6667$, and the spread, its square root, is $0.8165$. Divide each gap by the spread and you get $-1.2247$, $0$, $1.2247$. This group now has mean 0 and variance 1, and Figure 2 shows how the points moved.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three number lines showing 1, 2, 3 before normalising, after normalising, and after a scale of 2 and shift of 1">
<line x1="80.0" y1="235.0" x2="560.0" y2="235.0" class="stroke-muted"/>
<line x1="80.0" y1="235.0" x2="80.0" y2="240.0" class="stroke-muted"/>
<text x="80.0" y="255.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="160.0" y1="235.0" x2="160.0" y2="240.0" class="stroke-muted"/>
<text x="160.0" y="255.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="240.0" y1="235.0" x2="240.0" y2="240.0" class="stroke-muted"/>
<text x="240.0" y="255.0" class="fig-small" text-anchor="middle">0</text>
<line x1="320.0" y1="235.0" x2="320.0" y2="240.0" class="stroke-muted"/>
<text x="320.0" y="255.0" class="fig-small" text-anchor="middle">1</text>
<line x1="400.0" y1="235.0" x2="400.0" y2="240.0" class="stroke-muted"/>
<text x="400.0" y="255.0" class="fig-small" text-anchor="middle">2</text>
<line x1="480.0" y1="235.0" x2="480.0" y2="240.0" class="stroke-muted"/>
<text x="480.0" y="255.0" class="fig-small" text-anchor="middle">3</text>
<line x1="560.0" y1="235.0" x2="560.0" y2="240.0" class="stroke-muted"/>
<text x="560.0" y="255.0" class="fig-small" text-anchor="middle">4</text>
<line x1="240.0" y1="30.0" x2="240.0" y2="235.0" class="stroke-grid"/>
<text x="80.0" y="44.0" class="fig-small" text-anchor="start">before: 1, 2, 3 (mean 2)</text>
<line x1="80.0" y1="70.0" x2="560.0" y2="70.0" class="stroke-grid"/>
<circle cx="320.0" cy="70.0" r="7" class="fill-a"/>
<text x="320.0" y="92.0" class="fig-small" text-anchor="middle">1.00</text>
<circle cx="400.0" cy="70.0" r="7" class="fill-a"/>
<text x="400.0" y="92.0" class="fig-small" text-anchor="middle">2.00</text>
<circle cx="480.0" cy="70.0" r="7" class="fill-a"/>
<text x="480.0" y="92.0" class="fig-small" text-anchor="middle">3.00</text>
<text x="80.0" y="114.0" class="fig-small" text-anchor="start">normalised: mean 0, spread 1</text>
<line x1="80.0" y1="140.0" x2="560.0" y2="140.0" class="stroke-grid"/>
<circle cx="142.0" cy="140.0" r="7" class="fill-b"/>
<text x="142.0" y="162.0" class="fig-small" text-anchor="middle">-1.22</text>
<circle cx="240.0" cy="140.0" r="7" class="fill-b"/>
<text x="240.0" y="162.0" class="fig-small" text-anchor="middle">0</text>
<circle cx="338.0" cy="140.0" r="7" class="fill-b"/>
<text x="338.0" y="162.0" class="fig-small" text-anchor="middle">1.22</text>
<text x="80.0" y="179.0" class="fig-small" text-anchor="start">gamma = 2, beta = 1</text>
<line x1="80.0" y1="205.0" x2="560.0" y2="205.0" class="stroke-grid"/>
<circle cx="124.0" cy="205.0" r="7" class="fill-d"/>
<text x="124.0" y="227.0" class="fig-small" text-anchor="middle">-1.45</text>
<circle cx="320.0" cy="205.0" r="7" class="fill-d"/>
<text x="320.0" y="227.0" class="fig-small" text-anchor="middle">1.00</text>
<circle cx="516.0" cy="205.0" r="7" class="fill-d"/>
<text x="516.0" y="227.0" class="fig-small" text-anchor="middle">3.45</text>
<text x="562.0" y="112.0" class="fig-small" text-anchor="end">↓ subtract 2, divide by 0.8165</text>
<text x="562.0" y="178.0" class="fig-small" text-anchor="end">↓ multiply by 2, add 1</text>
</svg>
Figure 2. The row [1, 2, 3] in three steps. Subtracting the mean 2 moves the middle point onto 0, and dividing by the spread 0.8165 stretches the outer points to −1.22 and 1.22 (top to middle). The learned scale 2 and shift 1 then move the same three points to −1.45, 1, and 3.45 (middle to bottom). The shape of the row, evenly spaced with the middle one in the centre, never changes.
```

Two things are worth noticing. The result does not depend on where the group started or how big it was, because subtracting the mean removes the position and dividing by the spread removes the size. The group $[10, 20, 30]$ gives the same three numbers, and so does $[38.5, 39.5, 40.5]$. And the division can fail: the group $[5, 5, 5]$ has variance 0, so the sum is $0/0$. Adding a tiny constant $\epsilon$ (epsilon), such as $10^{-5}$, to the variance before the square root keeps the division finite, and the output is $0, 0, 0$.

For a group of $M$ numbers $x_1, \dots, x_M$, the whole step is

$$
\mu = \frac{1}{M}\sum_{i=1}^{M} x_i, \qquad
\sigma^2 = \frac{1}{M}\sum_{i=1}^{M} (x_i - \mu)^2, \qquad
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Read it aloud as "subtract the group's mean, then divide by its spread, with a little extra under the square root." Mu, $\mu$, is the group mean. Sigma squared, $\sigma^2$, is the group variance, and it divides by $M$. The hat on $\hat{x}$ marks a normalised value. In numpy:

```python
xhat = (x - x.mean()) / np.sqrt(x.var() + eps)
```

## Then let the layer undo it

Normalising forces every group to mean 0 and variance 1. That might not be what the next layer wants. If a ReLU follows, a group centred on 0 has about half of its values cut to zero. A layer may do better with its values shifted so more of them survive: for bell-shaped values with spread 1, a shift of 1 keeps about 84% positive. So after normalising, give each feature two numbers it can learn, a scale $\gamma$ (gamma) and a shift $\beta$ (beta):

$$
y_i = \gamma\, \hat{x}_i + \beta
$$

Read it aloud as "stretch by gamma, then slide by beta." On the group above with $\gamma = 2$ and $\beta = 1$, the outputs are $-1.449, 1, 3.449$. These are what the layer passes on, and training adjusts them like any other weights.

Why add them back right after removing the scale? Because they make the layer at least as expressive as before. Set $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$, and $y_i = x_i$ exactly, so the layer can undo the normalisation completely when that is best. What changes is where the layer starts and what it controls. The position and size of the output now come from two numbers $\gamma$ and $\beta$, and not from the layers before it. Whether stabilising the input is the whole reason this helps is still argued. What you can rely on is the effect: the scale of each group is held steady on every pass.

## Batch norm and layer norm in numpy

The two layers differ only in which numbers form a group. With `x` of shape `(N, D)`, the batch norm group is a column, so $M = N$, and you reduce over `axis=0`. The layer norm group is a row, so $M = D$, and you reduce over `axis=1`. In both, `gamma` and `beta` have shape `(D,)`, one pair per feature, and they broadcast across the rows.

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

## What the choice of group changes

The choice has three consequences.

**A shifted example.** Add the same constant to every feature of one example. Layer norm compares that example only with itself, so the constant cancels and the output does not change. Batch norm compares each feature with the other examples, so the shifted example becomes an outlier in every column, and its output moves. Figure 3 shows this on a real 10 by 6 matrix with a shift of 37.5.

```figure
<svg viewBox="0 0 620 320" role="img" aria-label="Two bar charts of row 0 before and after adding 37.5 to every feature of that example, for layer norm and for batch norm">
<line x1="40.0" y1="175.4" x2="310.0" y2="175.4" class="stroke-muted"/>
<text x="34.0" y="247.2" class="fig-small" text-anchor="end">-2</text>
<text x="34.0" y="179.4" class="fig-small" text-anchor="end">0</text>
<text x="34.0" y="111.5" class="fig-small" text-anchor="end">2</text>
<rect x="50.0" y="175.4" width="22.0" height="20.7" class="fill-soft-a stroke-a"/>
<rect x="59.0" y="175.4" width="22.0" height="20.7" class="stroke-b"/>
<text x="69.0" y="272.0" class="fig-small" text-anchor="middle">f1</text>
<rect x="94.0" y="175.4" width="22.0" height="4.9" class="fill-soft-a stroke-a"/>
<rect x="103.0" y="175.4" width="22.0" height="4.9" class="stroke-b"/>
<text x="113.0" y="272.0" class="fig-small" text-anchor="middle">f2</text>
<rect x="138.0" y="119.7" width="22.0" height="55.7" class="fill-soft-a stroke-a"/>
<rect x="147.0" y="119.7" width="22.0" height="55.7" class="stroke-b"/>
<text x="157.0" y="272.0" class="fig-small" text-anchor="middle">f3</text>
<rect x="182.0" y="152.8" width="22.0" height="22.6" class="fill-soft-a stroke-a"/>
<rect x="191.0" y="152.8" width="22.0" height="22.6" class="stroke-b"/>
<text x="201.0" y="272.0" class="fig-small" text-anchor="middle">f4</text>
<rect x="226.0" y="175.4" width="22.0" height="53.3" class="fill-soft-a stroke-a"/>
<rect x="235.0" y="175.4" width="22.0" height="53.3" class="stroke-b"/>
<text x="245.0" y="272.0" class="fig-small" text-anchor="middle">f5</text>
<rect x="270.0" y="174.7" width="22.0" height="0.7" class="fill-soft-a stroke-a"/>
<rect x="279.0" y="174.7" width="22.0" height="0.7" class="stroke-b"/>
<text x="289.0" y="272.0" class="fig-small" text-anchor="middle">f6</text>
<text x="175.0" y="34.0" class="fig-bold" text-anchor="middle">Layer norm, row 0</text>
<text x="175.0" y="52.0" class="fig-small" text-anchor="middle">the two bars coincide</text>
<line x1="340.0" y1="175.4" x2="610.0" y2="175.4" class="stroke-muted"/>
<text x="334.0" y="247.2" class="fig-small" text-anchor="end">-2</text>
<text x="334.0" y="179.4" class="fig-small" text-anchor="end">0</text>
<text x="334.0" y="111.5" class="fig-small" text-anchor="end">2</text>
<rect x="350.0" y="175.4" width="22.0" height="31.9" class="fill-soft-a stroke-a"/>
<rect x="359.0" y="73.9" width="22.0" height="101.5" class="stroke-b"/>
<text x="369.0" y="272.0" class="fig-small" text-anchor="middle">f1</text>
<rect x="394.0" y="168.1" width="22.0" height="7.2" class="fill-soft-a stroke-a"/>
<rect x="403.0" y="73.8" width="22.0" height="101.6" class="stroke-b"/>
<text x="413.0" y="272.0" class="fig-small" text-anchor="middle">f2</text>
<rect x="438.0" y="90.2" width="22.0" height="85.2" class="fill-soft-a stroke-a"/>
<rect x="447.0" y="73.7" width="22.0" height="101.7" class="stroke-b"/>
<text x="457.0" y="272.0" class="fig-small" text-anchor="middle">f3</text>
<rect x="482.0" y="164.7" width="22.0" height="10.7" class="fill-soft-a stroke-a"/>
<rect x="491.0" y="73.9" width="22.0" height="101.4" class="stroke-b"/>
<text x="501.0" y="272.0" class="fig-small" text-anchor="middle">f4</text>
<rect x="526.0" y="175.4" width="22.0" height="51.5" class="fill-soft-a stroke-a"/>
<rect x="535.0" y="73.8" width="22.0" height="101.5" class="stroke-b"/>
<text x="545.0" y="272.0" class="fig-small" text-anchor="middle">f5</text>
<rect x="570.0" y="175.4" width="22.0" height="7.0" class="fill-soft-a stroke-a"/>
<rect x="579.0" y="74.0" width="22.0" height="101.4" class="stroke-b"/>
<text x="589.0" y="272.0" class="fig-small" text-anchor="middle">f6</text>
<text x="475.0" y="34.0" class="fig-bold" text-anchor="middle">Batch norm, row 0</text>
<text x="475.0" y="52.0" class="fig-small" text-anchor="middle">outlined bars jump up</text>
<rect x="150.0" y="290.0" width="18.0" height="12.0" class="fill-soft-a stroke-a"/>
<text x="174.0" y="300.0" class="fig-small" text-anchor="start">before the shift</text>
<rect x="300.0" y="290.0" width="18.0" height="12.0" class="stroke-b"/>
<text x="324.0" y="300.0" class="fig-small" text-anchor="start">after adding 37.5 to every feature of row 0</text>
</svg>
Figure 3. Row 0 of a 10 by 6 matrix, before (shaded) and after (outlined) adding 37.5 to all six of its features. Layer norm compares the row with itself, so the shift cancels and the bars coincide. Batch norm compares each feature with the other nine examples, so the shifted row becomes an outlier and all six outputs jump to about 3.
```

**A batch of one.** Give batch norm a single example. Each column then holds one number, which is its own mean, so every gap is 0, every variance is 0, and the output is $\hat{x} = 0$ everywhere, so $y = \beta$. The example $[1, 2, 3, 4]$ comes out as $\beta$ whatever its values. Layer norm on the same example uses its four features and gives $[-1.34, -0.45, 0.45, 1.34]$.

**Test time.** Batch norm's output for one example depends on which others share its batch, which is wrong for a prediction. So during training batch norm keeps a running average of each column's mean and variance, and at test time it uses those stored numbers in place of the batch's. Layer norm uses only the example's own features, in training and at test time, so it has nothing to store. That is one reason recurrent networks and transformers usually use layer norm.

> [!KEY]
> Initialisation sets the scale once and training moves it away, so a layer can instead re-set its signal on every pass: subtract the group mean, divide by the group spread, then apply a learned scale and shift that let the layer undo it. Batch norm takes a column as its group and layer norm takes a row, and that one choice decides what happens with a shifted example, a batch of one, and test time.

> [!RECAP]
> - Normalising a group subtracts its mean and divides by $\sqrt{\sigma^2 + \epsilon}$, so the result ignores the group's position and size.
> - The constant $\epsilon$ keeps a constant group from giving $0/0$.
> - A learned $\gamma$ and $\beta$ follow, so the layer can pick any scale and position, including the original.
> - Batch norm reduces over `axis=0` and layer norm over `axis=1`. Use `keepdims=True`, and give `gamma` and `beta` shape `(D,)`.
> - Layer norm ignores a shifted example and needs no stored statistics. Batch norm of a single example returns $\beta$.

## Build it step by step

The checks call the functions in this order:

1. **`normalise_group(x, eps)`** takes an array and returns the same shape, with every entry treated as one group:

   ```python
   (x - x.mean()) / np.sqrt(x.var() + eps)
   ```

   The usual mistake is to forget `eps`, which turns a constant group into `nan`.

2. **`batchnorm_forward(x, gamma, beta, eps)`** takes `x` of shape `(N, D)` and returns `(out, cache)`. Reduce over `axis=0` with `keepdims=True`, and store the tuple `(xhat, istd, gamma, axis)` as the cache. The next chapter reads that tuple. The usual mistake is to drop `keepdims=True` and subtract a length-`D` mean from the wrong axis.

3. **`layernorm_forward(x, gamma, beta, eps)`** applies the same arithmetic over `axis=1`, so `istd` has shape `(N, 1)`. The usual mistake is to reduce over columns, which makes it batch norm.

## Where this breaks

- A small batch makes batch norm's column statistics noisy, and a batch of one makes them useless. Use layer norm, or group norm, which splits each row into a few groups.
- A trained network that uses batch norm must switch to stored statistics at test time. If it keeps using the batch, the prediction depends on its neighbours.
- Layer norm treats all features of an example as comparable. If one feature is on a very different scale, it dominates the row's mean and spread.
- The constant $\epsilon$ is not a free setting. A very small value lets a near-constant group blow up its noise, and a large one weakens the normalisation.

## The exercise

You write `normalise_group`, `batchnorm_forward`, and `layernorm_forward`. The checks verify that `[1, 2, 3]` becomes about $[-1.2247, 0, 1.2247]$, that the result ignores shift and scale, and that a constant group stays finite. They check mean 0 and variance 1 on the right axis for each layer, and that the output equals $\gamma\hat{x} + \beta$. They check that a batch of one gives $\beta$ under batch norm and not under layer norm, and that shifting one example leaves layer norm unchanged and moves batch norm.
