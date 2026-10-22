---
title: Backpropagating through a normalisation layer
slug: 072-normalisation-backward
part: "Part 4 — Neural networks"
summary: One input reaches every output of its group through the shared mean and variance, so its gradient has three routes that collapse into one formula.
minutes: 40
gpu: false
objectives:
  - Explain why the gradient of one input of a normalised group has three routes, and name each one.
  - Write the compact formula for the input gradient and the sums for the scale and shift gradients, for both batch norm and layer norm.
  - Predict two properties of the input gradient, that it sums to zero over a group and is orthogonal to the normalised values.
  - Check a backward pass against a finite difference, and choose a sensible nudge size.
prereqs:
  - 061-backprop-the-graph-and-reverse-order
  - 062-backprop-array-operations
  - 071-batch-norm-and-layer-norm
exercise: 072-normalisation-backward
---

# Backpropagating through a normalisation layer

Run layer norm on the row $[1, 2, 3]$. The outputs are $[-1.2247, 0, 1.2247]$. Suppose the loss only reads the first output, and you ask: if I nudge the first input $x_1$ up by a tiny amount, how much does the loss change? The first output rises, which is the obvious part. But $x_1$ also belongs to the average, so the mean rises a little, and every output is measured from the mean. It belongs to the variance too, so the spread changes, and every output is divided by the spread. A single nudge reaches all three outputs.

A backward pass has to collect all of these effects for every input, and a formula that misses one gives a wrong gradient. Training with a wrong gradient does not fail loudly. The loss just goes down worse than it should. So the question is how the gradient of one input collects all of those contributions, and how you know that your formula is right.

## What you already know

Four ideas carry this chapter:

- A value that is used in several places collects the sum of the gradients that come back from each place (see "Walking slopes backwards through a calculation").
- Each array operation has a small backward rule, and the gradient keeps the same shape as the value (see "The backward rule for each array operation").
- The forward pass subtracts the group mean, divides by $\sqrt{\sigma^2 + \epsilon}$, then applies $\gamma$ and $\beta$ (see "Normalising activations: batch norm versus layer norm").
- A finite difference nudges one input by a small $h$ and divides the change in the output by $h$.

## Why not treat the mean and spread as constants

The simplest backward pass would pretend that $\mu$ and $\sigma$ are fixed numbers. Then $\hat{x}_i = (x_i - \mu) \cdot \text{istd}$ is just a rescaling, and the gradient is $\text{istd}$ times the upstream gradient. Here $\text{istd}$ is $1/\sqrt{\sigma^2 + \epsilon}$, which is $1.2247$ for our row. With upstream gradient $[1, 0, 0]$ that gives $[1.2247, 0, 0]$.

The true gradient, which you will derive below and which a finite difference confirms, is $[0.2041, -0.4082, 0.2041]$. It is six times smaller in the first entry, and it is nonzero for the other two. The shortcut ignores two things: raising $x_1$ raises the mean, which pulls all three outputs down, and it changes the spread. Figure 1 draws the three routes from one input to the outputs.

```figure
<svg viewBox="0 0 640 360" role="img" aria-label="Three routes from one input x_k to the outputs of its group: direct, through the mean, and through the variance">
<defs><marker id="c072-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c072-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c072-arrow-d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<line x1="100.0" y1="170.0" x2="506.0" y2="170.0" class="stroke-a" marker-end="url(#c072-arrow-a)"/>
<line x1="92.0" y1="150.0" x2="264.0" y2="66.0" class="stroke-b" marker-end="url(#c072-arrow-b)"/>
<line x1="316.0" y1="64.0" x2="510.0" y2="78.0" class="stroke-b" marker-end="url(#c072-arrow-b)"/>
<line x1="316.0" y1="64.0" x2="510.0" y2="158.0" class="stroke-b" marker-end="url(#c072-arrow-b)"/>
<line x1="316.0" y1="64.0" x2="510.0" y2="238.0" class="stroke-b" marker-end="url(#c072-arrow-b)"/>
<line x1="92.0" y1="190.0" x2="264.0" y2="274.0" class="stroke-d" marker-end="url(#c072-arrow-d)"/>
<line x1="316.0" y1="276.0" x2="510.0" y2="102.0" class="stroke-d" marker-end="url(#c072-arrow-d)"/>
<line x1="316.0" y1="276.0" x2="510.0" y2="182.0" class="stroke-d" marker-end="url(#c072-arrow-d)"/>
<line x1="316.0" y1="276.0" x2="510.0" y2="262.0" class="stroke-d" marker-end="url(#c072-arrow-d)"/>
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
Figure 1. One input x_k reaches the outputs by three routes. The blue route is direct: x_k changes its own x̂_k. The orange route goes through the group mean μ, which every output uses. The purple route goes through the group variance σ², which every output also uses. The gradient of x_k is the sum of the three routes, and each label is that route's term in the formula.
```

## The easy end: gamma, beta, and the gradient of x̂

Work from the output backward. The layer computes $y = \gamma \hat{x} + \beta$, and the loss sends back `dout`, a gradient the same shape as the output. Each output depends on $\hat{x}$ through a simple product and sum, so:

- The shift $\beta$ is added to every output, so its gradient is the sum of `dout`.
- The scale $\gamma$ multiplies $\hat{x}$, so its gradient is the sum of `dout * xhat`.
- The gradient at $\hat{x}$ is `dxhat = dout * gamma`.

The sums for $\gamma$ and $\beta$ run over examples, `axis=0`, for batch norm and for layer norm. The reason is that $\gamma_j$ is one number shared by all $N$ examples in column $j$, and a value used $N$ times collects the sum of $N$ gradients. This holds for layer norm too, even though layer norm normalises along rows. Take two examples, $[1, 2, 3]$ and $[10, 20, 30]$, which normalise to the same $\hat{x}$, and put a gradient of 1 on the first feature of each. Then $d\beta_1 = 1 + 1 = 2$ and $d\gamma_1 = 2 \cdot (-1.2247) = -2.449$.

```python
dbeta = dout.sum(axis=0)
dgamma = (dout * xhat).sum(axis=0)
dxhat = dout * gamma
```

## Three routes from one input

Now the hard part: from `dxhat` back to `x`. Use the group of $M$ numbers from the last chapter, with mean $\mu$, variance $\sigma^2$, and $\hat{x}_i = (x_i - \mu)\,\text{istd}$. Take one input $x_k$ and ask how it changes each $\hat{x}_i$. There are three routes.

1. **Directly.** Only $\hat{x}_k$ contains $x_k$ by itself, and $\hat{x}_k$ changes by $\text{istd}$ per unit of $x_k$.
2. **Through the mean.** The mean is $\mu = \frac{1}{M}\sum x_i$, so raising $x_k$ by 1 raises $\mu$ by $1/M$. Every $\hat{x}_i$ is measured from $\mu$, so each one falls by $\text{istd}/M$.
3. **Through the variance.** Raising $x_k$ by 1 changes $\sigma^2$ by $2(x_k - \mu)/M$. (The part of that which passes through $\mu$ adds to zero, because the gaps from the mean sum to zero.) Each $\hat{x}_i$ changes by $-\tfrac{1}{2}\hat{x}_i\,\text{istd}^2$ per unit of $\sigma^2$, so the route changes it by $-\hat{x}_i \hat{x}_k\,\text{istd}/M$, using $(x_k - \mu)\,\text{istd} = \hat{x}_k$.

The gradient of $x_k$ adds up what each $\hat{x}_i$ sends back, weighted by `dxhat`. The direct route contributes $\text{istd}\cdot d\hat{x}_k$. The mean route contributes $-\frac{\text{istd}}{M}\sum_i d\hat{x}_i$. The variance route contributes $-\frac{\text{istd}}{M}\,\hat{x}_k \sum_i d\hat{x}_i \hat{x}_i$. Their sum, pulled over a common factor, is

$$
dx_k = \frac{\text{istd}}{M}\Big(M\, d\hat{x}_k - \sum_i d\hat{x}_i - \hat{x}_k \sum_i d\hat{x}_i\, \hat{x}_i\Big)
$$

Read it aloud as "take $M$ times the gradient at $\hat{x}_k$, subtract the group's total gradient, which undoes the mean, subtract $\hat{x}_k$ times the group's total of gradient times $\hat{x}$, which undoes the variance, and scale by istd over $M$." Each of the three terms is one route.

On the row $[1, 2, 3]$ with $d\hat{x} = [1, 0, 0]$, we have $M = 3$, $\text{istd} = 1.2247$, $\sum d\hat{x} = 1$, and $\sum d\hat{x}\,\hat{x} = -1.2247$. The three routes for each input are in this table, and Figure 2 draws them.

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
Figure 2. The gradient of the three inputs of the row [1, 2, 3], when the loss only looks at the first output (upstream gradient [1, 0, 0]). The direct route (blue) says only x1 matters. The mean route (orange) pulls all three down equally. The variance route (purple) pulls x1 down and x3 up, because x1 is below the mean and x3 above it. The three bars of the sum, 0.20, −0.41, and 0.20, add to zero (to four places, 0.2041, −0.4082, and 0.2041).
```

In numpy, the same code serves both layers. Only `axis` and $M$ change. For batch norm they are `axis=0` and $M = N$, and for layer norm they are `axis=1` and $M = D$:

```python
m = dout.shape[axis]
dx = (istd / m) * (
    m * dxhat
    - dxhat.sum(axis=axis, keepdims=True)
    - xhat * (dxhat * xhat).sum(axis=axis, keepdims=True)
)
```

The formula has two properties you can use as sanity checks. First, $dx$ sums to zero over each group. If you add the same constant to every input of a group, the mean moves with it and nothing else changes, so the loss cannot change, and the gradients must cancel. In the table the last column sums to $0.2041 - 0.4082 + 0.2041 = 0$. Second, $dx$ is orthogonal to $\hat{x}$: $\sum_k dx_k\,\hat{x}_k$ is close to zero. If you scale all inputs of a group by a constant, the spread scales with them and the outputs do not change. The sum is exactly zero when $\epsilon = 0$, and tiny with the usual $\epsilon = 10^{-5}$.

> [!WARNING]
> For layer norm, `dx` reduces over `axis=1`, but `dgamma` and `dbeta` still sum over `axis=0`. Summing them over the normalised axis gives shape `(N,)` and breaks the update of a `(D,)` parameter.

## Check it with a finite difference

A formula with three terms is easy to get slightly wrong, and a slightly wrong gradient still trains. So check it against a number you can compute without the formula. The **finite difference** nudges one input and measures the effect directly. For the row $[1, 2, 3]$ with `dout` $= [1, 0, 0]$, the scalar to differentiate is $f(x_1) = \sum_j \text{dout}_j \cdot y_j$, which here is the first output. At $x_1 = 1$ it is $-1.2247$. Nudge $x_1$ by $h = 10^{-6}$ in each direction and divide:

$$
\frac{f(x_1 + h) - f(x_1 - h)}{2h} = 0.20413
$$

That agrees with the formula's $0.2041$. Why the scalar $\sum \text{dout} \cdot y$? Because the backward pass computes the gradient of a loss whose gradient at the output is `dout`, and this sum is a loss with exactly that gradient. The same recipe works for arrays: nudge each entry in turn, keep the function value, and compare to the formula's entry.

How big should $h$ be? Figure 3 shows the largest error over a whole 5 by 7 layer as $h$ shrinks.

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
Figure 3. The largest gap between the formula's dx and a central finite difference, for a 5 by 7 layer-norm layer, as the nudge h shrinks. For big h the nudge is too coarse to follow the curve, so the gap falls as h shrinks. For tiny h the two function values agree in almost every digit, and round-off in their difference takes over. The gap is lowest near h = 10⁻⁵, and the exercise's checks use 10⁻⁶, well inside the tolerance.
```

The error is smallest near $h = 10^{-5}$ and rises on both sides, so $10^{-6}$ is a safe choice, and an $h$ of $10^{-14}$ is useless. Compare the result with a tolerance such as $10^{-4}$, which is how the checks in the exercise decide.

> [!KEY]
> Every input of a group affects every output through the shared mean and variance, so its gradient is the sum of three routes: direct, through the mean, and through the variance. They collapse into one formula that is the same for batch norm and layer norm apart from the axis. You trust it because a finite difference agrees with it, its group sums to zero, and it is orthogonal to $\hat{x}$.

> [!RECAP]
> - `dbeta = dout.sum(axis=0)` and `dgamma = (dout * xhat).sum(axis=0)` for both layers, because $\gamma$ and $\beta$ are shared by every example.
> - Set `dxhat = dout * gamma`, then `dx = (istd / M) * (M * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))`, summing over the normalised axis with `keepdims=True`.
> - The three terms are the direct route, the route through the mean, and the route through the variance.
> - `dx` sums to zero over each group and is orthogonal to `xhat` over it.
> - Check against a central finite difference with $h$ near $10^{-6}$.

## Build it step by step

The starter includes `batchnorm_forward` and `layernorm_forward`, which you wrote in the previous chapter. The checks call the functions you write in this order:

1. **`affine_param_grads(dout, xhat)`** takes two arrays of shape `(N, D)` and returns `(dgamma, dbeta)`, each of shape `(D,)`:

   ```python
   dbeta = dout.sum(axis=0)
   dgamma = (dout * xhat).sum(axis=0)
   ```

   The usual mistake is to sum over the normalised axis for layer norm.

2. **`batchnorm_backward(dout, cache)`** reads `(xhat, istd, gamma, axis)` from the cache and returns `(dx, dgamma, dbeta)`. Compute `dxhat = dout * gamma`, then apply the formula with `axis=0` and `M = N`. The usual mistake is a sign error on the variance term. A finite difference catches it, and `dx` stops summing to zero.

3. **`layernorm_backward(dout, cache)`** is the same formula, and the cache holds `axis=1`. Write one helper that both functions call. The usual mistake is to hard-code `axis=0` in the helper and make layer norm silently wrong.

## Cost and scaling

The backward pass reads the cached `xhat` and `istd` and makes a few passes over an `(N, D)` array, so its time grows in a straight line with the number of values, as the forward pass does. It stores nothing beyond the forward cache, which is about the size of `x`. Compared with the matrix multiplies of the surrounding layers, the cost is small.

## Where this breaks

- The formula assumes the forward pass used exactly the `axis`, `eps`, and variance definition you cache. If the forward divides the variance by $M - 1$, the backward must too.
- A tiny $\epsilon$ and a nearly constant group make `istd` huge, and the gradient with it. The constant $\epsilon$ is also what keeps the gradient finite.
- Batch norm's backward pass couples all examples in a batch, so a bug in one example's gradient moves the others. Test with a small `(N, D)` and check every entry.
- A finite difference is slow, because it makes two forward passes for every entry. Use it on a small array, as a test, and not in training.

## The exercise

You write `affine_param_grads`, `batchnorm_backward`, and `layernorm_backward`. The starter supplies `batchnorm_forward` and `layernorm_forward`.

The checks verify the scale and shift sums on a two-by-two example, and the hand example for $[1, 2, 3]$ with $dx = [0.2041, -0.4082, 0.2041]$. They compare both backward passes with finite differences for $x$, $\gamma$, and $\beta$. They check that `dx` sums to zero along the normalised axis, that it is orthogonal to `xhat`, and that layer norm's `dx` equals batch norm's `dx` on the transposed input.
