---
title: Multilayer perceptrons
slug: 029-multilayer-perceptrons
part: "Part 4 — Neural networks"
summary: Why stacked layers need a bend, how a forward pass runs, how ReLU hinges approximate any curve, and how width, depth, and cost trade off.
minutes: 80
gpu: false
objectives:
  - Explain why no straight line separates the XOR points, and why two affine layers with nothing between them collapse into one.
  - Compute ReLU, GELU, and their slopes, and solve XOR with two ReLU units followed by one straight cut.
  - Trace an input through a network by hand and write the forward pass for a list of layers.
  - Count the straight pieces of a one-input ReLU network exactly, and show that each extra layer of width-2 units doubles them.
  - Build the piecewise-linear interpolant of any knots with one ReLU hinge per segment, and state the universal approximation theorem without overclaiming.
  - Count a network's parameters, flops, and stored activations from its layer sizes.
  - Explain why doubling the width roughly quadruples the cost while depth reaches the same pieces with far fewer parameters.
prereqs:
  - 008-sigmoid-and-log-loss
  - 028-gaussian-mixtures-and-em
exercise: 029-multilayer-perceptrons
---

# Multilayer perceptrons

Two light switches control one lamp, wired so the lamp is on when exactly one switch is up. Write each setting as a point: switch one on the horizontal axis and switch two on the vertical axis, with 0 for down and 1 for up. The lamp is off at $(0,0)$ and $(1,1)$ and on at $(0,1)$ and $(1,0)$. You want a model that predicts the lamp from the two switches.

Logistic regression ([sigmoid and log loss](/lessons/008-sigmoid-and-log-loss)) squashes one weighted sum of the inputs, so its decision boundary is a single straight line. On these four points no amount of training fixes that. The fix is to let the model move the points first and cut afterwards, and the move has to include a bend.

This chapter builds that model, the multilayer perceptron, traces numbers through it, shows what it can approximate, and counts what it costs.

## What you already know

- An affine map is a weighted sum plus a constant: `W @ x + b`.
- Logistic regression is one affine map followed by a sigmoid, so it cuts the plane along a straight line.
- A slope tells you how much the output moves per unit nudge of the input, and a finite difference estimates it: `(f(x + h) - f(x - h)) / (2 * h)`.

## Why no straight line works

Look at Figure 1, left. The blue points are the two lamp-off settings and the purple points are the two lamp-on settings. The segment between the two blue points and the segment between the two purple points cross at $(0.5, 0.5)$, the midpoint of each.

Suppose a line did separate the classes. A side of a line is a half-plane, so if both blue points lie on one side, their midpoint does too. The same holds for the purple pair, on the other side. The midpoint is the same point, so no such line exists.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Left: the four switch settings as points, blue for lamp off and purple for lamp on, with the two diagonals crossing at (0.5, 0.5) and one orange line that misclassifies a point. Right: the same four points after two ReLU units, where the blue points both land at the origin and one straight orange line separates the classes.">
<defs><marker id="c026-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="50.0" y1="198.3" x2="250.0" y2="198.3" class="stroke-muted"/>
<line x1="83.3" y1="230.0" x2="83.3" y2="40.0" class="stroke-muted"/>
<text x="250.0" y="214.3" text-anchor="end" class="fig-small">x1</text>
<text x="75.3" y="44.0" text-anchor="end" class="fig-small">x2</text>
<text x="83.3" y="222.3" text-anchor="middle" class="fig-small"></text>
<text x="66.3" y="202.3" text-anchor="end" class="fig-small">0</text>
<text x="216.7" y="222.3" text-anchor="middle" class="fig-small">1</text>
<text x="66.3" y="75.7" text-anchor="end" class="fig-small">1</text>
<line x1="350.0" y1="198.3" x2="550.0" y2="198.3" class="stroke-muted"/>
<line x1="383.3" y1="230.0" x2="383.3" y2="40.0" class="stroke-muted"/>
<text x="550.0" y="214.3" text-anchor="end" class="fig-small">h1</text>
<text x="375.3" y="44.0" text-anchor="end" class="fig-small">h2</text>
<text x="383.3" y="222.3" text-anchor="middle" class="fig-small"></text>
<text x="366.3" y="202.3" text-anchor="end" class="fig-small">0</text>
<text x="516.7" y="222.3" text-anchor="middle" class="fig-small">1</text>
<text x="366.3" y="75.7" text-anchor="end" class="fig-small">1</text>
<text x="150.0" y="22.0" text-anchor="middle" class="fig-bold">Before: the raw switches</text>
<text x="450.0" y="22.0" text-anchor="middle" class="fig-bold">After: the two hidden units</text>
<line x1="83.3" y1="198.3" x2="216.7" y2="71.7" class="stroke-a dashed" stroke-width="1.5"/>
<line x1="83.3" y1="71.7" x2="216.7" y2="198.3" class="stroke-d dashed" stroke-width="1.5"/>
<line x1="56.7" y1="109.7" x2="176.7" y2="223.7" class="stroke-b" stroke-width="2.2"/>
<circle cx="150.0" cy="135.0" r="4" class="fill-ink"/>
<text x="226.0" y="139.0" text-anchor="start" class="fig-small">(0.5, 0.5)</text>
<line x1="156" y1="135" x2="222" y2="135" class="stroke-muted" stroke-width="1"/>
<circle cx="83.3" cy="198.3" r="7" class="fill-a"/>
<circle cx="216.7" cy="71.7" r="7" class="fill-a"/>
<circle cx="83.3" cy="71.7" r="7" class="fill-d"/>
<circle cx="216.7" cy="198.3" r="7" class="fill-d"/>
<text x="150.0" y="240.0" text-anchor="start" class="fig-small">one line</text>
<line x1="356.7" y1="109.7" x2="476.7" y2="223.7" class="stroke-b" stroke-width="2.2"/>
<circle cx="383.3" cy="198.3" r="10" class="fill-a"/>
<circle cx="383.3" cy="198.3" r="5" class="fill-paper"/>
<circle cx="516.7" cy="198.3" r="7" class="fill-d"/>
<circle cx="383.3" cy="71.7" r="7" class="fill-d"/>
<text x="397.3" y="220.3" text-anchor="start" class="fig-small">both blue</text>
<text x="420.0" y="100.0" text-anchor="start" class="fig-small">h1 + h2 = 0.5</text>
<line x1="292.0" y1="150.0" x2="340.0" y2="150.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c026-arrow)"/>
<text x="316.0" y="172.0" text-anchor="middle" class="fig-small">ReLU</text>
<text x="316.0" y="188.0" text-anchor="middle" class="fig-small">units</text>
</svg>
Figure 1. Left: the blue pair and the purple pair have the same midpoint, (0.5, 0.5), so no line can put the blue pair on one side and the purple pair on the other. Right: after two ReLU units the blue points land together at the origin and the purple points move away, so one line cuts them apart.
```

The right panel shows the way out. If a first stage moves the points so that the blue pair lands together and the purple pair lands apart, a line can finish the job. A neural network does this: hidden units re-draw the points, and the last layer is logistic regression on the result.

## Why stacking plain layers does not help

A layer holds several **units**, each a small detector: a weighted sum of all the inputs plus its own constant. Put the weights of the units in the rows of a matrix `W` and the constants in a vector `b`. With 2 inputs and 3 units, `W` has shape `(out, in)` = `(3, 2)`, and `W @ x + b` gives the three readings at once. For a row of data you write `x @ W.T + b`.

The obvious fix is more layers: feed the first layer's output into a second one. Take $x = [2, -3]$. Let the first layer have rows $(1, 0)$ and $(0, 2)$ with constants $[1, 0]$, and let the second have the single row $[3, 1]$ with constant $4$. Run it:

- First layer: $[2, -6] + [1, 0] = [3, -6]$.
- Second layer: $3\cdot 3 + 1\cdot(-6) + 4 = 7$.

Now merge them. The product of the weight matrices is $[3, 1]\begin{bmatrix}1&0\\0&2\end{bmatrix} = [3, 2]$, and the first constants carried through the second layer give $3\cdot 1 + 1\cdot 0 + 4 = 7$. The merged layer on $x$ gives $3\cdot 2 + 2\cdot(-3) + 7 = 7$, the same answer. In symbols, for any two affine layers:

$$
W_2 (W_1 x + b_1) + b_2 = (W_2 W_1)\, x + (W_2 b_1 + b_2) .
$$

Read it aloud: "two layers equal one layer whose weights are the product of the two matrices." In numpy:

```python
W = W2 @ W1
b = W2 @ b1 + b2
```

A hundred layers with nothing between them are still one affine map. A narrow middle layer makes it worse, because $W_2 W_1$ cannot have more independent output directions (its **rank**) than the middle layer has units.

> [!KEY]
> Without something between them, any stack of affine layers is one affine layer. Depth buys nothing until a layer does something that is not affine.

## Bend each number with ReLU

The cheapest thing that is not affine is a comparison. **ReLU**, short for rectified linear unit, keeps positive numbers and replaces negative ones with 0. Apply it after the first layer of the example, to the readings $[3, -6]$, and you get $[3, 0]$. The second layer now gives $3\cdot 3 + 0 + 4 = 13$ instead of 7. The answer changed, so the two layers no longer merge. One corner was enough to break the collapse.

As a formula, with the slope next to it:

$$
\operatorname{relu}(x) = \max(x, 0), \qquad
\operatorname{relu}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases} .
$$

Read it aloud: "ReLU returns the larger of $x$ and 0, and its slope is 1 where the input passes and 0 where it is cut off." The slope is undefined at the corner $x = 0$, and the checks skip it. For the readings $[2, -3, -2]$ the outputs are $[2, 0, 0]$ and the slopes are $[1, 0, 0]$:

```python
np.maximum(x, 0.0)
(x > 0).astype(float)
```

Now solve XOR. Use two units with ReLU: $h_1 = \operatorname{relu}(x_1 - x_2)$ and $h_2 = \operatorname{relu}(x_2 - x_1)$. The first fires when switch one leads, the second when switch two leads.

| $(x_1, x_2)$ | readings $(x_1 - x_2,\ x_2 - x_1)$ | after ReLU $(h_1, h_2)$ | $h_1 + h_2$ |
|---|---|---|---|
| $(0, 0)$ | $(0, 0)$ | $(0, 0)$ | 0 |
| $(1, 1)$ | $(0, 0)$ | $(0, 0)$ | 0 |
| $(0, 1)$ | $(-1, 1)$ | $(0, 1)$ | 1 |
| $(1, 0)$ | $(1, -1)$ | $(1, 0)$ | 1 |

Both lamp-off settings land on $(0,0)$ and both lamp-on settings land beyond the line $h_1 + h_2 = 0.5$, as in Figure 1, right. The sign of $h_1 + h_2 - 0.5$ is the answer. The hidden layer computed $|x_1 - x_2|$, which an affine map alone can never produce.

> [!KEY]
> No line separates XOR because both classes share the midpoint $(0.5, 0.5)$. Stacked affine layers merge into one, so they cannot help. A bend between layers lets the hidden layer move the points so that one final line separates them.

## Trace one input by hand

Figure 2 shows a network with two inputs, three hidden units, and one output. The weights are chosen so each unit asks a question you can read off.

```figure
<svg viewBox="0 0 600 318" role="img" aria-label="A network with two inputs 2 and minus 3, three hidden units whose values before ReLU are 2, minus 3 and minus 2 and after ReLU are 2, 0 and 0, and one output 2.5.">

<line x1="114.0" y1="105.0" x2="276.0" y2="68.0" class="stroke-a" stroke-width="1.8"/>
<line x1="114.0" y1="105.0" x2="276.0" y2="156.0" class="stroke-muted dashed" stroke-width="1"/>
<line x1="114.0" y1="105.0" x2="276.0" y2="244.0" class="stroke-a" stroke-width="1.8"/>
<line x1="114.0" y1="195.0" x2="276.0" y2="68.0" class="stroke-muted dashed" stroke-width="1"/>
<line x1="114.0" y1="195.0" x2="276.0" y2="156.0" class="stroke-a" stroke-width="1.8"/>
<line x1="114.0" y1="195.0" x2="276.0" y2="244.0" class="stroke-a" stroke-width="1.8"/>
<line x1="324.0" y1="68.0" x2="486.0" y2="150.0" class="stroke-a" stroke-width="1.8"/>
<line x1="324.0" y1="156.0" x2="486.0" y2="150.0" class="stroke-a" stroke-width="1.8"/>
<line x1="324.0" y1="244.0" x2="486.0" y2="150.0" class="stroke-a" stroke-width="1.8"/>
<circle cx="90.0" cy="105.0" r="24" class="fill-soft-a stroke-a"/>
<text x="90.0" y="110.0" text-anchor="middle" class="fig-bold">2</text>
<text x="90.0" y="145.0" text-anchor="middle" class="fig-small">x1</text>
<circle cx="90.0" cy="195.0" r="24" class="fill-soft-a stroke-a"/>
<text x="90.0" y="200.0" text-anchor="middle" class="fig-bold">−3</text>
<text x="90.0" y="235.0" text-anchor="middle" class="fig-small">x2</text>
<circle cx="300.0" cy="68.0" r="24" class="fill-soft-b stroke-b"/>
<text x="300.0" y="73.0" text-anchor="middle" class="fig-bold">2</text>
<text x="330.0" y="46.0" text-anchor="start" class="fig-small">z = 2</text>
<circle cx="300.0" cy="156.0" r="24" class="fill-sunk stroke-muted"/>
<text x="300.0" y="161.0" text-anchor="middle" class="fig-bold">0</text>
<text x="330.0" y="134.0" text-anchor="start" class="fig-small">z = −3</text>
<line x1="283.2" y1="139.2" x2="316.8" y2="172.8" class="stroke-bad" stroke-width="2"/>
<circle cx="300.0" cy="244.0" r="24" class="fill-sunk stroke-muted"/>
<text x="300.0" y="249.0" text-anchor="middle" class="fig-bold">0</text>
<text x="330.0" y="278.0" text-anchor="start" class="fig-small">z = −2</text>
<line x1="283.2" y1="227.2" x2="316.8" y2="260.8" class="stroke-bad" stroke-width="2"/>
<text x="294.0" y="286.0" text-anchor="end" class="fig-small">unit 3 has constant −1</text>
<circle cx="510.0" cy="150.0" r="24" class="fill-soft-d stroke-d"/>
<text x="510.0" y="155.0" text-anchor="middle" class="fig-bold">2.5</text>
<text x="510.0" y="192.0" text-anchor="middle" class="fig-small">output</text>
<text x="510.0" y="114.0" text-anchor="middle" class="fig-small">+ bias 0.5</text>
<text x="90.0" y="22.0" text-anchor="middle" class="fig-bold">inputs</text>
<text x="300.0" y="22.0" text-anchor="middle" class="fig-bold">hidden layer (3 units)</text>
<text x="510.0" y="22.0" text-anchor="middle" class="fig-bold">output layer</text>
<text x="300.0" y="306.0" text-anchor="middle" class="fig-small">z = value before the bend; number inside the circle = value after ReLU</text>
</svg>
Figure 2. The running example, with every number retraced. Solid blue lines are weights equal to 1 and dashed grey lines are weights equal to 0. The value z is each unit's reading before the bend. ReLU keeps 2 and cuts the other two to 0 (crossed out), so the output unit adds 2 + 0 + 0 and the bias 0.5 to get 2.5.
```

Unit 1 has weights $(1, 0)$ and reads $2$. Unit 2 has weights $(0, 1)$ and reads $-3$. Unit 3 has weights $(1, 1)$ and constant $-1$, and it reads $2 + (-3) - 1 = -2$. Each raw reading, before any bend, is a **pre-activation**: $[2, -3, -2]$. The **activations** after ReLU are $[2, 0, 0]$. The output unit has weights $(1, 1, 1)$, constant $0.5$, and no bend, so it gives $2 + 0 + 0 + 0.5 = 2.5$.

Running the numbers from the input to the output, layer by layer, is the **forward pass**. Every layer multiplies by the weights, adds the constants, and bends every entry, so a network of thousands of units is still a short loop. For row data one step is

```python
z = h @ W.T + b
h = activation(z)
```

The common mistake is `h @ W` without the transpose. A square `W` runs and silently returns wrong numbers, because it reads columns as units.

## Soften the corner with GELU

ReLU has a corner at 0, and a slightly negative unit is cut to exactly 0. Many networks use a smooth cousin, the **Gaussian error linear unit**, GELU for short: multiply $x$ by the probability that a standard bell-curve random number falls below $x$. That is near 0 for very negative $x$ and near 1 for large $x$. Code uses a stand-in built from **tanh**, a smooth S-shaped function that rises from $-1$ to $1$:

$$
\operatorname{gelu}(x) = \tfrac{1}{2}\, x \left(1 + \tanh(u)\right), \qquad
u = \sqrt{2/\pi}\,\left(x + 0.044715\, x^3\right) .
$$

Read it aloud: "squash a scaled cubic in $x$ with tanh, turn the result into a switch between 0 and 1, and multiply by $x$." At $x = 1$, $u = 0.797885 \times 1.044715 = 0.833562$, $\tanh(u) = 0.682384$, the switch is $0.841192$, and $\operatorname{gelu}(1) = 0.841192$. At $x = -1$ the same steps give $\operatorname{gelu}(-1) = -0.158808$, a small dip below 0 that ReLU never makes. Figure 3 plots both curves.

```python
u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
return 0.5 * x * (1 + np.tanh(u))
```

Training needs the slope of each bend. For GELU the **product rule** says the slope of $a(x)\,s(x)$ is $a'(x)\,s(x) + a(x)\,s'(x)$. Write $\operatorname{gelu}(x) = x \cdot s(x)$ with $s = \tfrac12(1 + t)$ and $t = \tanh(u)$. The slope of $\tanh$ is $1 - t^2$, and $u$ has slope $\sqrt{2/\pi}\,(1 + 3 \cdot 0.044715\, x^2)$. Then

$$
\operatorname{gelu}'(x) = \tfrac{1}{2}(1 + t) + \tfrac{1}{2}\, x\, (1 - t^2)\, \sqrt{2/\pi}\,\left(1 + 3 \cdot 0.044715\, x^2\right) .
$$

At $x = 1$ the switch is $0.841192$, and the second term is $0.5 \times 0.534352 \times 0.904917 = 0.241772$, so the slope is $1.082964$.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Left: ReLU in grey and GELU in orange on the range minus 4 to 4; GELU dips just below zero for negative inputs and matches ReLU far from zero. Right: ReLU's slope is a step from 0 to 1, while GELU's slope rises smoothly, passing 0.5 at zero and peaking slightly above 1 near x equal to 1.4.">

<line x1="50.0" y1="181.0" x2="290.0" y2="181.0" class="stroke-muted"/>
<text x="50.0" y="232.0" text-anchor="middle" class="fig-small">−4</text>
<text x="110.0" y="232.0" text-anchor="middle" class="fig-small">−2</text>
<text x="170.0" y="232.0" text-anchor="middle" class="fig-small">0</text>
<text x="230.0" y="232.0" text-anchor="middle" class="fig-small">2</text>
<text x="290.0" y="232.0" text-anchor="middle" class="fig-small">4</text>
<line x1="340.0" y1="190.7" x2="570.0" y2="190.7" class="stroke-muted"/>
<text x="340.0" y="206.7" text-anchor="middle" class="fig-small">−4</text>
<text x="397.5" y="206.7" text-anchor="middle" class="fig-small">−2</text>
<text x="455.0" y="206.7" text-anchor="middle" class="fig-small">0</text>
<text x="512.5" y="206.7" text-anchor="middle" class="fig-small">2</text>
<text x="570.0" y="206.7" text-anchor="middle" class="fig-small">4</text>
<text x="46.0" y="185.0" text-anchor="end" class="fig-small">0</text>
<text x="46.0" y="151.0" text-anchor="end" class="fig-small">1</text>
<text x="46.0" y="117.0" text-anchor="end" class="fig-small">2</text>
<text x="46.0" y="83.0" text-anchor="end" class="fig-small">3</text>
<text x="46.0" y="219.0" text-anchor="end" class="fig-small">−1</text>
<text x="336.0" y="194.7" text-anchor="end" class="fig-small">0</text>
<text x="336.0" y="73.3" text-anchor="end" class="fig-small">1</text>
<text x="170.0" y="22.0" text-anchor="middle" class="fig-bold">ReLU and GELU</text>
<text x="455.0" y="22.0" text-anchor="middle" class="fig-bold">their slopes</text>
<path d="M50.0,181.0 L50.6,181.0 L51.2,181.0 L51.8,181.0 L52.4,181.0 L53.0,181.0 L53.6,181.0 L54.2,181.0 L54.8,181.0 L55.4,181.0 L56.0,181.0 L56.6,181.0 L57.2,181.0 L57.8,181.0 L58.4,181.0 L59.0,181.0 L59.6,181.0 L60.2,181.0 L60.8,181.0 L61.4,181.0 L62.0,181.0 L62.6,181.0 L63.2,181.0 L63.8,181.0 L64.4,181.0 L65.0,181.0 L65.6,181.0 L66.2,181.0 L66.8,181.0 L67.4,181.0 L68.0,181.0 L68.6,181.0 L69.2,181.0 L69.8,181.0 L70.4,181.0 L71.0,181.0 L71.6,181.0 L72.2,181.0 L72.8,181.0 L73.4,181.0 L74.0,181.0 L74.6,181.0 L75.2,181.0 L75.8,181.0 L76.4,181.0 L77.0,181.0 L77.6,181.0 L78.2,181.0 L78.8,181.0 L79.4,181.0 L80.0,181.0 L80.6,181.0 L81.2,181.0 L81.8,181.0 L82.4,181.0 L83.0,181.0 L83.6,181.0 L84.2,181.0 L84.8,181.0 L85.4,181.0 L86.0,181.0 L86.6,181.0 L87.2,181.0 L87.8,181.0 L88.4,181.0 L89.0,181.0 L89.6,181.0 L90.2,181.0 L90.8,181.0 L91.4,181.0 L92.0,181.0 L92.6,181.0 L93.2,181.0 L93.8,181.0 L94.4,181.0 L95.0,181.0 L95.6,181.0 L96.2,181.0 L96.8,181.0 L97.4,181.0 L98.0,181.0 L98.6,181.0 L99.2,181.0 L99.8,181.0 L100.4,181.0 L101.0,181.0 L101.6,181.0 L102.2,181.0 L102.8,181.0 L103.4,181.0 L104.0,181.0 L104.6,181.0 L105.2,181.0 L105.8,181.0 L106.4,181.0 L107.0,181.0 L107.6,181.0 L108.2,181.0 L108.8,181.0 L109.4,181.0 L110.0,181.0 L110.6,181.0 L111.2,181.0 L111.8,181.0 L112.4,181.0 L113.0,181.0 L113.6,181.0 L114.2,181.0 L114.8,181.0 L115.4,181.0 L116.0,181.0 L116.6,181.0 L117.2,181.0 L117.8,181.0 L118.4,181.0 L119.0,181.0 L119.6,181.0 L120.2,181.0 L120.8,181.0 L121.4,181.0 L122.0,181.0 L122.6,181.0 L123.2,181.0 L123.8,181.0 L124.4,181.0 L125.0,181.0 L125.6,181.0 L126.2,181.0 L126.8,181.0 L127.4,181.0 L128.0,181.0 L128.6,181.0 L129.2,181.0 L129.8,181.0 L130.4,181.0 L131.0,181.0 L131.6,181.0 L132.2,181.0 L132.8,181.0 L133.4,181.0 L134.0,181.0 L134.6,181.0 L135.2,181.0 L135.8,181.0 L136.4,181.0 L137.0,181.0 L137.6,181.0 L138.2,181.0 L138.8,181.0 L139.4,181.0 L140.0,181.0 L140.6,181.0 L141.2,181.0 L141.8,181.0 L142.4,181.0 L143.0,181.0 L143.6,181.0 L144.2,181.0 L144.8,181.0 L145.4,181.0 L146.0,181.0 L146.6,181.0 L147.2,181.0 L147.8,181.0 L148.4,181.0 L149.0,181.0 L149.6,181.0 L150.2,181.0 L150.8,181.0 L151.4,181.0 L152.0,181.0 L152.6,181.0 L153.2,181.0 L153.8,181.0 L154.4,181.0 L155.0,181.0 L155.6,181.0 L156.2,181.0 L156.8,181.0 L157.4,181.0 L158.0,181.0 L158.6,181.0 L159.2,181.0 L159.8,181.0 L160.4,181.0 L161.0,181.0 L161.6,181.0 L162.2,181.0 L162.8,181.0 L163.4,181.0 L164.0,181.0 L164.6,181.0 L165.2,181.0 L165.8,181.0 L166.4,181.0 L167.0,181.0 L167.6,181.0 L168.2,181.0 L168.8,181.0 L169.4,181.0 L170.0,181.0 L170.6,180.3 L171.2,179.6 L171.8,179.0 L172.4,178.3 L173.0,177.6 L173.6,176.9 L174.2,176.2 L174.8,175.6 L175.4,174.9 L176.0,174.2 L176.6,173.5 L177.2,172.8 L177.8,172.2 L178.4,171.5 L179.0,170.8 L179.6,170.1 L180.2,169.4 L180.8,168.8 L181.4,168.1 L182.0,167.4 L182.6,166.7 L183.2,166.0 L183.8,165.4 L184.4,164.7 L185.0,164.0 L185.6,163.3 L186.2,162.6 L186.8,162.0 L187.4,161.3 L188.0,160.6 L188.6,159.9 L189.2,159.2 L189.8,158.6 L190.4,157.9 L191.0,157.2 L191.6,156.5 L192.2,155.8 L192.8,155.2 L193.4,154.5 L194.0,153.8 L194.6,153.1 L195.2,152.4 L195.8,151.8 L196.4,151.1 L197.0,150.4 L197.6,149.7 L198.2,149.0 L198.8,148.4 L199.4,147.7 L200.0,147.0 L200.6,146.3 L201.2,145.6 L201.8,145.0 L202.4,144.3 L203.0,143.6 L203.6,142.9 L204.2,142.2 L204.8,141.6 L205.4,140.9 L206.0,140.2 L206.6,139.5 L207.2,138.8 L207.8,138.2 L208.4,137.5 L209.0,136.8 L209.6,136.1 L210.2,135.4 L210.8,134.8 L211.4,134.1 L212.0,133.4 L212.6,132.7 L213.2,132.0 L213.8,131.4 L214.4,130.7 L215.0,130.0 L215.6,129.3 L216.2,128.6 L216.8,128.0 L217.4,127.3 L218.0,126.6 L218.6,125.9 L219.2,125.2 L219.8,124.6 L220.4,123.9 L221.0,123.2 L221.6,122.5 L222.2,121.8 L222.8,121.2 L223.4,120.5 L224.0,119.8 L224.6,119.1 L225.2,118.4 L225.8,117.8 L226.4,117.1 L227.0,116.4 L227.6,115.7 L228.2,115.0 L228.8,114.4 L229.4,113.7 L230.0,113.0 L230.6,112.3 L231.2,111.6 L231.8,111.0 L232.4,110.3 L233.0,109.6 L233.6,108.9 L234.2,108.2 L234.8,107.6 L235.4,106.9 L236.0,106.2 L236.6,105.5 L237.2,104.8 L237.8,104.2 L238.4,103.5 L239.0,102.8 L239.6,102.1 L240.2,101.4 L240.8,100.8 L241.4,100.1 L242.0,99.4 L242.6,98.7 L243.2,98.0 L243.8,97.4 L244.4,96.7 L245.0,96.0 L245.6,95.3 L246.2,94.6 L246.8,94.0 L247.4,93.3 L248.0,92.6 L248.6,91.9 L249.2,91.2 L249.8,90.6 L250.4,89.9 L251.0,89.2 L251.6,88.5 L252.2,87.8 L252.8,87.2 L253.4,86.5 L254.0,85.8 L254.6,85.1 L255.2,84.4 L255.8,83.8 L256.4,83.1 L257.0,82.4 L257.6,81.7 L258.2,81.0 L258.8,80.4 L259.4,79.7 L260.0,79.0 L260.6,78.3 L261.2,77.6 L261.8,77.0 L262.4,76.3 L263.0,75.6 L263.6,74.9 L264.2,74.2 L264.8,73.6 L265.4,72.9 L266.0,72.2 L266.6,71.5 L267.2,70.8 L267.8,70.2 L268.4,69.5 L269.0,68.8 L269.6,68.1 L270.2,67.4 L270.8,66.8 L271.4,66.1 L272.0,65.4 L272.6,64.7 L273.2,64.0 L273.8,63.4 L274.4,62.7 L275.0,62.0 L275.6,61.3 L276.2,60.6 L276.8,60.0 L277.4,59.3 L278.0,58.6 L278.6,57.9 L279.2,57.2 L279.8,56.6 L280.4,55.9 L281.0,55.2 L281.6,54.5 L282.2,53.8 L282.8,53.2 L283.4,52.5 L284.0,51.8 L284.6,51.1 L285.2,50.4 L285.8,49.8 L286.4,49.1 L287.0,48.4 L287.6,47.7 L288.2,47.0 L288.8,46.4 L289.4,45.7 L290.0,45.0" fill="none" class="stroke-muted" stroke-width="2"/>
<path d="M50.0,181.0 L50.6,181.0 L51.2,181.0 L51.8,181.0 L52.4,181.0 L53.0,181.0 L53.6,181.0 L54.2,181.0 L54.8,181.0 L55.4,181.0 L56.0,181.0 L56.6,181.0 L57.2,181.0 L57.8,181.0 L58.4,181.0 L59.0,181.0 L59.6,181.0 L60.2,181.0 L60.8,181.0 L61.4,181.0 L62.0,181.0 L62.6,181.0 L63.2,181.0 L63.8,181.0 L64.4,181.0 L65.0,181.0 L65.6,181.0 L66.2,181.0 L66.8,181.0 L67.4,181.0 L68.0,181.0 L68.6,181.0 L69.2,181.0 L69.8,181.0 L70.4,181.0 L71.0,181.0 L71.6,181.0 L72.2,181.1 L72.8,181.1 L73.4,181.1 L74.0,181.1 L74.6,181.1 L75.2,181.1 L75.8,181.1 L76.4,181.1 L77.0,181.1 L77.6,181.1 L78.2,181.1 L78.8,181.1 L79.4,181.1 L80.0,181.1 L80.6,181.1 L81.2,181.1 L81.8,181.1 L82.4,181.2 L83.0,181.2 L83.6,181.2 L84.2,181.2 L84.8,181.2 L85.4,181.2 L86.0,181.2 L86.6,181.2 L87.2,181.3 L87.8,181.3 L88.4,181.3 L89.0,181.3 L89.6,181.3 L90.2,181.3 L90.8,181.4 L91.4,181.4 L92.0,181.4 L92.6,181.4 L93.2,181.4 L93.8,181.5 L94.4,181.5 L95.0,181.5 L95.6,181.5 L96.2,181.6 L96.8,181.6 L97.4,181.6 L98.0,181.7 L98.6,181.7 L99.2,181.7 L99.8,181.8 L100.4,181.8 L101.0,181.8 L101.6,181.9 L102.2,181.9 L102.8,181.9 L103.4,182.0 L104.0,182.0 L104.6,182.1 L105.2,182.1 L105.8,182.2 L106.4,182.2 L107.0,182.3 L107.6,182.3 L108.2,182.4 L108.8,182.4 L109.4,182.5 L110.0,182.5 L110.6,182.6 L111.2,182.7 L111.8,182.7 L112.4,182.8 L113.0,182.9 L113.6,182.9 L114.2,183.0 L114.8,183.1 L115.4,183.1 L116.0,183.2 L116.6,183.3 L117.2,183.3 L117.8,183.4 L118.4,183.5 L119.0,183.6 L119.6,183.7 L120.2,183.7 L120.8,183.8 L121.4,183.9 L122.0,184.0 L122.6,184.1 L123.2,184.2 L123.8,184.2 L124.4,184.3 L125.0,184.4 L125.6,184.5 L126.2,184.6 L126.8,184.7 L127.4,184.8 L128.0,184.9 L128.6,184.9 L129.2,185.0 L129.8,185.1 L130.4,185.2 L131.0,185.3 L131.6,185.4 L132.2,185.5 L132.8,185.5 L133.4,185.6 L134.0,185.7 L134.6,185.8 L135.2,185.9 L135.8,185.9 L136.4,186.0 L137.0,186.1 L137.6,186.1 L138.2,186.2 L138.8,186.3 L139.4,186.3 L140.0,186.4 L140.6,186.5 L141.2,186.5 L141.8,186.6 L142.4,186.6 L143.0,186.6 L143.6,186.7 L144.2,186.7 L144.8,186.7 L145.4,186.7 L146.0,186.8 L146.6,186.8 L147.2,186.8 L147.8,186.8 L148.4,186.8 L149.0,186.8 L149.6,186.7 L150.2,186.7 L150.8,186.7 L151.4,186.6 L152.0,186.6 L152.6,186.5 L153.2,186.5 L153.8,186.4 L154.4,186.3 L155.0,186.2 L155.6,186.2 L156.2,186.0 L156.8,185.9 L157.4,185.8 L158.0,185.7 L158.6,185.5 L159.2,185.4 L159.8,185.2 L160.4,185.1 L161.0,184.9 L161.6,184.7 L162.2,184.5 L162.8,184.3 L163.4,184.1 L164.0,183.9 L164.6,183.6 L165.2,183.4 L165.8,183.1 L166.4,182.8 L167.0,182.6 L167.6,182.3 L168.2,182.0 L168.8,181.7 L169.4,181.3 L170.0,181.0 L170.6,180.7 L171.2,180.3 L171.8,179.9 L172.4,179.6 L173.0,179.2 L173.6,178.8 L174.2,178.4 L174.8,177.9 L175.4,177.5 L176.0,177.1 L176.6,176.6 L177.2,176.1 L177.8,175.7 L178.4,175.2 L179.0,174.7 L179.6,174.2 L180.2,173.7 L180.8,173.2 L181.4,172.6 L182.0,172.1 L182.6,171.5 L183.2,171.0 L183.8,170.4 L184.4,169.8 L185.0,169.2 L185.6,168.7 L186.2,168.0 L186.8,167.4 L187.4,166.8 L188.0,166.2 L188.6,165.6 L189.2,164.9 L189.8,164.3 L190.4,163.6 L191.0,163.0 L191.6,162.3 L192.2,161.6 L192.8,160.9 L193.4,160.3 L194.0,159.6 L194.6,158.9 L195.2,158.2 L195.8,157.5 L196.4,156.8 L197.0,156.0 L197.6,155.3 L198.2,154.6 L198.8,153.9 L199.4,153.1 L200.0,152.4 L200.6,151.7 L201.2,150.9 L201.8,150.2 L202.4,149.4 L203.0,148.7 L203.6,147.9 L204.2,147.2 L204.8,146.4 L205.4,145.7 L206.0,144.9 L206.6,144.1 L207.2,143.4 L207.8,142.6 L208.4,141.9 L209.0,141.1 L209.6,140.3 L210.2,139.6 L210.8,138.8 L211.4,138.0 L212.0,137.3 L212.6,136.5 L213.2,135.7 L213.8,134.9 L214.4,134.2 L215.0,133.4 L215.6,132.6 L216.2,131.9 L216.8,131.1 L217.4,130.4 L218.0,129.6 L218.6,128.8 L219.2,128.1 L219.8,127.3 L220.4,126.5 L221.0,125.8 L221.6,125.0 L222.2,124.3 L222.8,123.5 L223.4,122.8 L224.0,122.0 L224.6,121.2 L225.2,120.5 L225.8,119.7 L226.4,119.0 L227.0,118.3 L227.6,117.5 L228.2,116.8 L228.8,116.0 L229.4,115.3 L230.0,114.5 L230.6,113.8 L231.2,113.1 L231.8,112.3 L232.4,111.6 L233.0,110.9 L233.6,110.1 L234.2,109.4 L234.8,108.7 L235.4,108.0 L236.0,107.2 L236.6,106.5 L237.2,105.8 L237.8,105.1 L238.4,104.3 L239.0,103.6 L239.6,102.9 L240.2,102.2 L240.8,101.5 L241.4,100.8 L242.0,100.1 L242.6,99.3 L243.2,98.6 L243.8,97.9 L244.4,97.2 L245.0,96.5 L245.6,95.8 L246.2,95.1 L246.8,94.4 L247.4,93.7 L248.0,93.0 L248.6,92.3 L249.2,91.6 L249.8,90.9 L250.4,90.2 L251.0,89.5 L251.6,88.8 L252.2,88.1 L252.8,87.4 L253.4,86.7 L254.0,86.0 L254.6,85.3 L255.2,84.6 L255.8,84.0 L256.4,83.3 L257.0,82.6 L257.6,81.9 L258.2,81.2 L258.8,80.5 L259.4,79.8 L260.0,79.1 L260.6,78.4 L261.2,77.7 L261.8,77.1 L262.4,76.4 L263.0,75.7 L263.6,75.0 L264.2,74.3 L264.8,73.6 L265.4,72.9 L266.0,72.3 L266.6,71.6 L267.2,70.9 L267.8,70.2 L268.4,69.5 L269.0,68.8 L269.6,68.2 L270.2,67.5 L270.8,66.8 L271.4,66.1 L272.0,65.4 L272.6,64.7 L273.2,64.1 L273.8,63.4 L274.4,62.7 L275.0,62.0 L275.6,61.3 L276.2,60.7 L276.8,60.0 L277.4,59.3 L278.0,58.6 L278.6,57.9 L279.2,57.3 L279.8,56.6 L280.4,55.9 L281.0,55.2 L281.6,54.5 L282.2,53.8 L282.8,53.2 L283.4,52.5 L284.0,51.8 L284.6,51.1 L285.2,50.4 L285.8,49.8 L286.4,49.1 L287.0,48.4 L287.6,47.7 L288.2,47.0 L288.8,46.4 L289.4,45.7 L290.0,45.0" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="200.0" cy="152.4" r="4.5" class="fill-ink"/>
<text x="192.0" y="144.4" text-anchor="end" class="fig-small">x = 1: 0.841</text>
<circle cx="140.0" cy="186.4" r="4.5" class="fill-ink"/>
<text x="134.0" y="208.4" text-anchor="middle" class="fig-small">x = −1: −0.159</text>
<text x="86.0" y="62.6" text-anchor="start" class="fig-small">ReLU (grey)</text>
<text x="86.0" y="83.0" text-anchor="start" class="fig-small">GELU (orange)</text>
<path d="M340.0,190.7 L455.0,190.7" fill="none" class="stroke-muted" stroke-width="2"/>
<path d="M455.0,69.3 L570.0,69.3" fill="none" class="stroke-muted" stroke-width="2"/>
<path d="M340.0,190.8 L340.6,190.8 L341.1,190.8 L341.7,190.8 L342.3,190.8 L342.9,190.8 L343.4,190.8 L344.0,190.8 L344.6,190.8 L345.2,190.8 L345.8,190.8 L346.3,190.8 L346.9,190.8 L347.5,190.8 L348.1,190.8 L348.6,190.9 L349.2,190.9 L349.8,190.9 L350.4,190.9 L350.9,190.9 L351.5,190.9 L352.1,190.9 L352.6,191.0 L353.2,191.0 L353.8,191.0 L354.4,191.0 L354.9,191.0 L355.5,191.1 L356.1,191.1 L356.7,191.1 L357.2,191.1 L357.8,191.2 L358.4,191.2 L359.0,191.2 L359.6,191.3 L360.1,191.3 L360.7,191.3 L361.3,191.4 L361.9,191.4 L362.4,191.5 L363.0,191.5 L363.6,191.6 L364.1,191.6 L364.7,191.7 L365.3,191.7 L365.9,191.8 L366.4,191.8 L367.0,191.9 L367.6,192.0 L368.2,192.0 L368.8,192.1 L369.3,192.2 L369.9,192.3 L370.5,192.4 L371.1,192.5 L371.6,192.6 L372.2,192.6 L372.8,192.7 L373.4,192.9 L373.9,193.0 L374.5,193.1 L375.1,193.2 L375.6,193.3 L376.2,193.4 L376.8,193.6 L377.4,193.7 L377.9,193.8 L378.5,194.0 L379.1,194.1 L379.7,194.3 L380.2,194.5 L380.8,194.6 L381.4,194.8 L382.0,195.0 L382.6,195.1 L383.1,195.3 L383.7,195.5 L384.3,195.7 L384.9,195.9 L385.4,196.1 L386.0,196.3 L386.6,196.5 L387.1,196.7 L387.7,197.0 L388.3,197.2 L388.9,197.4 L389.4,197.6 L390.0,197.9 L390.6,198.1 L391.2,198.4 L391.8,198.6 L392.3,198.8 L392.9,199.1 L393.5,199.4 L394.1,199.6 L394.6,199.9 L395.2,200.1 L395.8,200.4 L396.4,200.6 L396.9,200.9 L397.5,201.2 L398.1,201.4 L398.6,201.7 L399.2,202.0 L399.8,202.2 L400.4,202.5 L400.9,202.7 L401.5,203.0 L402.1,203.2 L402.7,203.5 L403.2,203.7 L403.8,203.9 L404.4,204.2 L405.0,204.4 L405.6,204.6 L406.1,204.8 L406.7,205.0 L407.3,205.2 L407.9,205.3 L408.4,205.5 L409.0,205.7 L409.6,205.8 L410.1,205.9 L410.7,206.0 L411.3,206.1 L411.9,206.2 L412.4,206.3 L413.0,206.3 L413.6,206.4 L414.2,206.4 L414.8,206.4 L415.3,206.3 L415.9,206.3 L416.5,206.2 L417.1,206.1 L417.6,206.0 L418.2,205.9 L418.8,205.7 L419.4,205.5 L419.9,205.3 L420.5,205.0 L421.1,204.7 L421.6,204.4 L422.2,204.1 L422.8,203.7 L423.4,203.3 L423.9,202.9 L424.5,202.4 L425.1,201.9 L425.7,201.4 L426.2,200.8 L426.8,200.2 L427.4,199.5 L428.0,198.9 L428.6,198.2 L429.1,197.4 L429.7,196.6 L430.3,195.8 L430.9,194.9 L431.4,194.0 L432.0,193.1 L432.6,192.1 L433.1,191.1 L433.7,190.1 L434.3,189.0 L434.9,187.8 L435.4,186.7 L436.0,185.5 L436.6,184.2 L437.2,183.0 L437.8,181.7 L438.3,180.3 L438.9,178.9 L439.5,177.5 L440.1,176.1 L440.6,174.6 L441.2,173.1 L441.8,171.6 L442.4,170.0 L442.9,168.4 L443.5,166.8 L444.1,165.1 L444.6,163.4 L445.2,161.7 L445.8,160.0 L446.4,158.2 L446.9,156.4 L447.5,154.6 L448.1,152.8 L448.7,151.0 L449.2,149.1 L449.8,147.3 L450.4,145.4 L451.0,143.5 L451.6,141.6 L452.1,139.7 L452.7,137.7 L453.3,135.8 L453.9,133.9 L454.4,131.9 L455.0,130.0 L455.6,128.1 L456.1,126.1 L456.7,124.2 L457.3,122.3 L457.9,120.3 L458.4,118.4 L459.0,116.5 L459.6,114.6 L460.2,112.7 L460.8,110.9 L461.3,109.0 L461.9,107.2 L462.5,105.4 L463.1,103.6 L463.6,101.8 L464.2,100.0 L464.8,98.3 L465.4,96.6 L465.9,94.9 L466.5,93.2 L467.1,91.6 L467.6,90.0 L468.2,88.4 L468.8,86.9 L469.4,85.4 L470.0,83.9 L470.5,82.5 L471.1,81.1 L471.7,79.7 L472.2,78.3 L472.8,77.0 L473.4,75.8 L474.0,74.5 L474.5,73.3 L475.1,72.2 L475.7,71.0 L476.3,69.9 L476.9,68.9 L477.4,67.9 L478.0,66.9 L478.6,66.0 L479.1,65.1 L479.7,64.2 L480.3,63.4 L480.9,62.6 L481.4,61.8 L482.0,61.1 L482.6,60.5 L483.2,59.8 L483.8,59.2 L484.3,58.6 L484.9,58.1 L485.5,57.6 L486.1,57.1 L486.6,56.7 L487.2,56.3 L487.8,55.9 L488.4,55.6 L488.9,55.3 L489.5,55.0 L490.1,54.7 L490.6,54.5 L491.2,54.3 L491.8,54.1 L492.4,54.0 L493.0,53.9 L493.5,53.8 L494.1,53.7 L494.7,53.7 L495.2,53.6 L495.8,53.6 L496.4,53.6 L497.0,53.7 L497.6,53.7 L498.1,53.8 L498.7,53.9 L499.3,54.0 L499.9,54.1 L500.4,54.2 L501.0,54.3 L501.6,54.5 L502.1,54.7 L502.7,54.8 L503.3,55.0 L503.9,55.2 L504.4,55.4 L505.0,55.6 L505.6,55.8 L506.2,56.1 L506.8,56.3 L507.3,56.5 L507.9,56.8 L508.5,57.0 L509.0,57.3 L509.6,57.5 L510.2,57.8 L510.8,58.0 L511.4,58.3 L511.9,58.6 L512.5,58.8 L513.1,59.1 L513.6,59.4 L514.2,59.6 L514.8,59.9 L515.4,60.1 L516.0,60.4 L516.5,60.6 L517.1,60.9 L517.7,61.2 L518.2,61.4 L518.8,61.6 L519.4,61.9 L520.0,62.1 L520.5,62.4 L521.1,62.6 L521.7,62.8 L522.3,63.0 L522.9,63.3 L523.4,63.5 L524.0,63.7 L524.6,63.9 L525.1,64.1 L525.7,64.3 L526.3,64.5 L526.9,64.7 L527.5,64.9 L528.0,65.0 L528.6,65.2 L529.2,65.4 L529.8,65.5 L530.3,65.7 L530.9,65.9 L531.5,66.0 L532.0,66.2 L532.6,66.3 L533.2,66.4 L533.8,66.6 L534.4,66.7 L534.9,66.8 L535.5,66.9 L536.1,67.0 L536.6,67.1 L537.2,67.3 L537.8,67.4 L538.4,67.4 L539.0,67.5 L539.5,67.6 L540.1,67.7 L540.7,67.8 L541.2,67.9 L541.8,68.0 L542.4,68.0 L543.0,68.1 L543.5,68.2 L544.1,68.2 L544.7,68.3 L545.3,68.3 L545.9,68.4 L546.4,68.4 L547.0,68.5 L547.6,68.5 L548.1,68.6 L548.7,68.6 L549.3,68.7 L549.9,68.7 L550.5,68.7 L551.0,68.8 L551.6,68.8 L552.2,68.8 L552.8,68.9 L553.3,68.9 L553.9,68.9 L554.5,68.9 L555.0,69.0 L555.6,69.0 L556.2,69.0 L556.8,69.0 L557.4,69.0 L557.9,69.1 L558.5,69.1 L559.1,69.1 L559.6,69.1 L560.2,69.1 L560.8,69.1 L561.4,69.1 L562.0,69.2 L562.5,69.2 L563.1,69.2 L563.7,69.2 L564.2,69.2 L564.8,69.2 L565.4,69.2 L566.0,69.2 L566.5,69.2 L567.1,69.2 L567.7,69.2 L568.3,69.2 L568.9,69.2 L569.4,69.2 L570.0,69.2" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="483.8" cy="59.2" r="4.5" class="fill-ink"/>
<text x="491.8" y="45.2" text-anchor="middle" class="fig-small">x = 1: 1.083</text>
<text x="443.5" y="134.0" text-anchor="end" class="fig-small">0.5 at 0</text>
<text x="489.5" y="148.2" text-anchor="start" class="fig-small">grey: ReLU, a step</text>
<text x="489.5" y="166.4" text-anchor="start" class="fig-small">orange: GELU, a dial</text>
</svg>
Figure 3. Left: GELU follows ReLU far from 0 and rounds the corner, with a small dip below 0 for negative inputs. Right: ReLU's slope is a switch, 0 then 1, while GELU's slope is a dial that reaches 0.5 at x = 0 and overshoots 1 (peak about 1.13 near x = 1.4) before settling back toward 1.
```

> [!KEY]
> A network is one move repeated: form each unit's pre-activation as a weighted sum, then bend it into an activation, and feed the activations to the next layer. The bend can be sharp (ReLU) or smooth (GELU), and in both cases it has a slope you can compute and check.

## Width adds bends, depth multiplies them

To fit a signal with many bends, start with one ReLU unit with one input $x$ and constant $-t$ outputs $\operatorname{relu}(x - t)$, a hinge that is flat until $x$ reaches $t$ and then rises at slope 1. Add hinges that bend at different places and the slopes add up, so the sum is a chain of straight pieces. Each unit's reading is a straight line in $x$, which crosses 0 at most once, so $m$ units make at most $m+1$ pieces.

If every bend costs one unit, 256 bends cost 256 units. Stacked layers are cheaper, because each layer can fold what the last one made.

Start with the **tent map**, which takes $x$ in $[0, 1]$ up to 1 at $x = \tfrac12$ and back down:

$$
T(x) = 2\min(x,\, 1-x) = 2\operatorname{relu}(x) - 4\operatorname{relu}\!\left(x - \tfrac12\right) \quad \text{on } [0,1] .
$$

Read it aloud: "a rise of slope 2 from the first hinge, then a drop of 4 from the second that turns the slope into $-2$." Check $x = 0.75$: $2(0.75) - 4(0.25) = 0.5 = 2\cdot\min(0.75, 0.25)$.

Now feed $T$'s output into $T$ again. Because $T$ takes $[0,1]$ onto $[0,1]$, the second tent has a full-size input and folds each half again, $T(T(x))$ on the grid $x = 0, 0.125, \dots, 1$ is $0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0$: two peaks and four pieces. Figure 4 shows one, two, and three folds. After $k$ folds there are $2^k$ pieces.

```figure
<svg viewBox="0 0 600 302" role="img" aria-label="Three stacked panels showing the tent map and its repeats on the range 0 to 1, with 2, 4 and 8 straight pieces coloured alternately, every peak reaching 1.">

<text x="70.0" y="32.0" text-anchor="start" class="fig-bold">T(x): 2 pieces</text>
<line x1="70.0" y1="94.0" x2="540.0" y2="94.0" class="stroke-muted"/>
<path d="M70.0,94.0 L305.0,40.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M305.0,40.0 L540.0,94.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<circle cx="70.0" cy="94.0" r="3" class="fill-ink"/>
<circle cx="305.0" cy="40.0" r="3" class="fill-ink"/>
<circle cx="540.0" cy="94.0" r="3" class="fill-ink"/>
<text x="70.0" y="108.0" text-anchor="middle" class="fig-small">0</text>
<text x="540.0" y="108.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="126.0" text-anchor="start" class="fig-bold">T(T(x)): 4 pieces</text>
<line x1="70.0" y1="188.0" x2="540.0" y2="188.0" class="stroke-muted"/>
<path d="M70.0,188.0 L187.5,134.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M187.5,134.0 L305.0,188.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M305.0,188.0 L422.5,134.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M422.5,134.0 L540.0,188.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<circle cx="70.0" cy="188.0" r="3" class="fill-ink"/>
<circle cx="187.5" cy="134.0" r="3" class="fill-ink"/>
<circle cx="305.0" cy="188.0" r="3" class="fill-ink"/>
<circle cx="422.5" cy="134.0" r="3" class="fill-ink"/>
<circle cx="540.0" cy="188.0" r="3" class="fill-ink"/>
<text x="70.0" y="202.0" text-anchor="middle" class="fig-small">0</text>
<text x="540.0" y="202.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="138.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="220.0" text-anchor="start" class="fig-bold">T(T(T(x))): 8 pieces</text>
<line x1="70.0" y1="282.0" x2="540.0" y2="282.0" class="stroke-muted"/>
<path d="M70.0,282.0 L128.8,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M128.8,228.0 L187.5,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M187.5,282.0 L246.2,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M246.2,228.0 L305.0,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M305.0,282.0 L363.8,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M363.8,228.0 L422.5,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M422.5,282.0 L481.2,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M481.2,228.0 L540.0,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<circle cx="70.0" cy="282.0" r="3" class="fill-ink"/>
<circle cx="128.8" cy="228.0" r="3" class="fill-ink"/>
<circle cx="187.5" cy="282.0" r="3" class="fill-ink"/>
<circle cx="246.2" cy="228.0" r="3" class="fill-ink"/>
<circle cx="305.0" cy="282.0" r="3" class="fill-ink"/>
<circle cx="363.8" cy="228.0" r="3" class="fill-ink"/>
<circle cx="422.5" cy="282.0" r="3" class="fill-ink"/>
<circle cx="481.2" cy="228.0" r="3" class="fill-ink"/>
<circle cx="540.0" cy="282.0" r="3" class="fill-ink"/>
<text x="70.0" y="296.0" text-anchor="middle" class="fig-small">0</text>
<text x="540.0" y="296.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="232.0" text-anchor="end" class="fig-small">1</text>
</svg>
Figure 4. Each fold doubles the number of pieces: 2, then 4, then 8. The colours alternate piece by piece. Notice that every peak reaches 1 and every valley reaches 0, so the next fold has a full-height copy of the zigzag to work on in each half.
```

Each fold needs two hinges. Layer 1 holds $h_1 = \operatorname{relu}(x)$ and $h_2 = \operatorname{relu}(x - \tfrac12)$, so the tent's output is $y = 2h_1 - 4h_2$. The next layer needs $\operatorname{relu}(y)$ and $\operatorname{relu}(y - \tfrac12)$, but a layer only takes weighted sums of the previous layer's values. So you fuse the combine step into its weights: both rows are $[2, -4]$, and the constants are $[0, -\tfrac12]$. The last layer is $[2, -4]$ with no bend, so depth $k$ uses $2k$ hidden units. One hidden layer needs at least $2^k - 1$ units for $2^k$ pieces. At $k = 6$ the folded network has 64 pieces from 12 units.

## Count the pieces exactly

To test the claim you need to count pieces without plotting. Inside any stretch of the input where no unit changes sign, the network is one straight line. So the counter keeps, for every stretch, a slope and an intercept for each value in the current layer, starting with slope 1 and intercept 0 on $[x_{\min}, x_{\max}]$.

For each layer, on each stretch, do four things:

1. Compute every unit's reading as a line: slope `W @ slopes`, intercept `W @ intercepts + b`.
2. A reading crosses 0 at $x = -\text{intercept}/\text{slope}$. For a ReLU unit, split the stretch there if the crossing lands strictly inside it. An identity unit never splits anything.
3. On each part, test the sign at the midpoint. A ReLU unit with a negative midpoint reading is cut to 0: its slope and intercept become 0.
4. After the last layer, merge neighbours whose final lines are equal.

Take the hinge $\operatorname{relu}(x - 0.5)$ on $[0, 1]$, with output weight 1. The reading has slope 1 and intercept $-0.5$, so it crosses at $x = 0.5$ and the stretch splits in two. On $[0, 0.5]$ the midpoint reading is $-0.25$, so the unit is cut to slope 0, intercept 0. On $[0.5, 1]$ it stays slope 1, intercept $-0.5$. The lines differ, so the count is 2. With output weight 0 both parts are slope 0, intercept 0, and the merge leaves 1 piece.

Depth multiplies here. In the sawtooth's second layer, the reading $y - \tfrac12$ crosses 0 once inside each of the two stretches the first layer made, so it splits both and you get four pieces. One hidden layer cannot do this, because its readings are lines over the whole input.

> [!KEY]
> Width adds bends one at a time. A layer can fold the curve the previous layer made, so $k$ layers of width 2 make $2^k$ pieces from $2k$ units, while one hidden layer needs about $2^k$.

## Hinges reach any curve

Pick points on a curve and join them with straight segments. A one-hidden-layer ReLU network builds that zigzag with one hinge per segment: start at the first height, then at each knot add a hinge that changes the slope.

Take the knots $(0,1)$, $(1,3)$, $(2,2)$, $(4,2)$. The segment slopes are $2$, $-1$, $0$. The first hinge, at $x=0$, takes the slope from 0 to 2, so its output weight is $2$. The hinges at $x=1$ and $x=2$ change it by $-3$ and $+1$. At $x=3$ the network gives $1 + 2(3) - 3(2) + 1(1) = 2$, on the flat segment from $(2,2)$ to $(4,2)$. Using the slopes $2, -1, 0$ as output weights instead gives 5.

With knots $x_0 < \dots < x_n$, heights $y_j$, and slopes $s_j = (y_{j+1}-y_j)/(x_{j+1}-x_j)$:

$$
f(x) = y_0 + \sum_{j=0}^{n-1} c_j \operatorname{relu}(x - x_j), \qquad c_0 = s_0,\quad c_j = s_j - s_{j-1} .
$$

Read it aloud: "the first height, plus a hinge at each segment's left knot, weighted by the change in slope." On $[x_j, x_{j+1}]$ only hinges $0$ to $j$ are on, and their weights sum to $s_j$, so $f$ follows each segment and hits every knot.

For smooth $g$ and knots $h$ apart, the error is at most $\tfrac{h^2}{8}\max|g''|$. For $\sin$ on $[0, \pi]$ the measured maximum error is $0.0188$ with 8 segments and $0.00479$ with 16, a ratio of $3.93$: doubling the knots cuts the error about 4 times.

The **universal approximation theorem** makes this general. For a compact set $K \subset \mathbb{R}^d$, a continuous $g$ on $K$, and any $\varepsilon > 0$, some one-hidden-layer network is within $\varepsilon$ of $g$ at every point of $K$. Cybenko (1989) proved it for sigmoid units, Hornik (1991) for bounded, non-constant activations, and Leshno et al. (1993) for locally bounded, piecewise continuous ones exactly when the activation is not a polynomial.

> [!WARNING]
> The theorem says a network exists. It says nothing about how many units you need, whether training finds the weights, or the fit outside $K$.

## Cost and scaling

A **parameter** is one number the network learns: every entry of every `W` and `b`. A layer has `out * in` weights and `out` biases. Layers of size 4, 8, 8, and 3 have shapes $(8, 4)$, $(8, 8)$, and $(3, 8)$, so

$$
(32 + 8) + (64 + 8) + (24 + 3) = 40 + 72 + 27 = 139 .
$$

A **flop** is one floating-point operation, such as one addition or one multiplication, and flops set how long a prediction takes. Each of the `in * out` weights takes one multiplication and one addition, so the product costs $2 \cdot \text{in} \cdot \text{out}$ flops. Add `out` for the bias, then the bend per output: 1 for ReLU, 9 for GELU (counting tanh as one), 0 for identity. These are counting conventions, so compare totals and do not time with them. For the 4-8-8-3 network with ReLU, ReLU, then identity, the layers cost $64 + 8 + 8 = 80$, $128 + 8 + 8 = 144$, and $48 + 3 + 0 = 51$, a total of 275. As a formula for one layer:

$$
\text{flops} = 2\,\text{in}\cdot\text{out} + \text{out} + c_{\text{act}}\cdot\text{out} ,\qquad c_{\text{relu}} = 1,\; c_{\text{gelu}} = 9,\; c_{\text{identity}} = 0 .
$$

Training adds a third cost: walking the network in reverse needs the values each layer produced going forward, so it keeps the input and every layer's output. For one input, the 4-8-8-3 network holds $4 + 8 + 8 + 3 = 23$ numbers, and for a batch of 100 it holds 2300.

A layer with $m$ inputs and $m$ outputs has $m^2$ weights, so doubling the width roughly quadruples parameters and flops, while doubling the depth roughly doubles them.

The sawtooth at depth $k$ has $4 + 6(k-1) + 3 = 6k + 1$ parameters. A one-hidden-layer network with the $2^k$ units it needs has $3\cdot 2^k + 1$. They match at $k = 1$ (7) and $k = 2$ (13), and then part ways: at $k = 10$ it is 61 against 3073.

> [!KEY]
> A network's cost follows from its layer shapes alone: parameters are `out * in + out` per layer, flops are $2\,\text{in}\cdot\text{out} + \text{out} + c\cdot\text{out}$, and training keeps the input and every layer's output. Doubling the width roughly quadruples the cost, while folding with depth makes $2^k$ pieces for $6k + 1$ parameters.

> [!RECAP]
> - A layer is `h @ W.T + b` with `W` shaped `(out, in)`, and two affine layers merge into `W2 @ W1` and `W2 @ b1 + b2`.
> - A bend between layers, ReLU or GELU, is what makes depth useful.
> - A hinge at each knot, weighted by the change in slope, reproduces the piecewise-linear interpolant.
> - Counting pieces tracks a slope and an intercept per stretch, splits at interior roots, and merges equal neighbours.
> - Parameters per layer are `out * in + out`, flops add the product, bias, and bend, and stored activations are `in_dim` plus the sum of the `out` sizes.

## Build it step by step

Build twelve functions, in this order.

1. **`relu(x)`** returns the elementwise larger of the entry and 0, in the same shape.

2. **`relu_grad(x)`** returns 1.0 where `x > 0` and 0.0 where `x < 0`, as floats. Cast a boolean mask with `.astype(float)`.

3. **`compose_affine(W1, b1, W2, b2)`** returns the pair `(W, b)` of the single layer that does the same job as the two. Mind the order: the first layer's matrix sits on the right of the product.

4. **`gelu(x)`** returns the tanh version. Dropping the cubic term gives close numbers that fail the check at 1.

5. **`gelu_grad(x)`** returns the slope. A common mistake is forgetting the `3 *` in the slope of `u`.

6. **`mlp_forward(params, x)`** runs a list of layers on a vector `x` and returns the output vector. Pick the bend by name: `"identity"` leaves `z` unchanged, and an unknown name raises `ValueError`.

7. **`count_linear_regions_1d(params, x_min, x_max)`** returns the number of pieces of a one-input network as an integer. Split only at roots strictly inside a stretch, and skip units whose slope is near 0. Skipping the merge over-counts a crossing that changes nothing.

8. **`sawtooth_network(depth)`** returns `depth + 1` layers: the first with `W` of shape `(2, 1)` and constants `[0, -0.5]`, then `depth - 1` ReLU layers with the fused weights above, then `W = [[2, -4]]` with `"identity"`.

9. **`parameter_count(params)`** returns the size of `W` plus the size of `b`, summed over layers. Do not count only the weights.

10. **`flop_count(params)`** sums the product, bias, and bend cost per layer. Do not charge identity a flop.

11. **`activation_memory(params, in_dim)`** returns the input size plus every layer's output size. Do not leave out the input.

12. **`hinge_interpolant(xs, ys)`** returns two layers in the `mlp_forward` format: `n - 1` ReLU hinges with `W` of ones and `b = -xs[:-1]`, then an identity layer whose weights are the changes in slope and whose bias is `ys[0]`. Weights equal to the slopes themselves are the mistake to avoid. Raise `ValueError` unless `xs` strictly increases and matches `ys`.

> [!TIP]
> Check any slope you write with `(f(x + h) - f(x - h)) / (2 * h)` at `h = 1e-5`. If the two disagree away from a corner, the slope is wrong.

> [!WARNING]
> `W` has shape `(out, in)`. Swapped, the product term of the flop count still comes out right, but the bias and bend terms use `out` and come out wrong.

## Where this breaks

A ReLU unit whose reading is negative for every input outputs 0 and has slope 0, so nothing flowing back can change it. People call this a dead unit. The sawtooth network exists, but training rarely finds it, so depth makes a function possible, not likely.

## The exercise

You implement the twelve functions above. The checks compare the bends and slopes with hand values and finite differences, solve XOR with your `relu`, run `mlp_forward` on the hand example (it returns $[2.5]$), count hinges, confirm the sawtooth has $2^k$ pieces, and check the 4-8-8-3 network (139 parameters, 275 flops, 23 stored activations). Three more checks test `hinge_interpolant`: the hand example, exact heights and straight segments, and the 4 times error drop on $\sin$.

## Further reading

- The [6.S898 Deep Learning, Fall 2023 schedule](https://phillipi.github.io/6.s898/) lists "Approximation theory" on Thursday 9/14 (rows are dates, not numbered lectures), covering universal approximation and Barron's theorem. The [MIT OCW 6.7960 Fall 2024 notes](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/lists/lecture-notes/) title the topic "Lecture 3: Approximation Theory".
- Cybenko (1989), Hornik (1991), and Leshno, Lin, Pinkus, and Schocken (1993, *Neural Networks* 6(6), 861-867) state the theorem.
