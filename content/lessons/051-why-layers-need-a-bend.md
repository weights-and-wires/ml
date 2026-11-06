---
title: Why a stack of layers needs a bend
slug: 051-why-layers-need-a-bend
part: "Part 4 — Neural networks"
summary: Two stacked affine layers merge into one, so a network needs a bend such as ReLU between layers to cut a boundary that is not a straight line.
minutes: 40
gpu: false
objectives:
  - Explain why no straight line separates the XOR points, using the shared midpoint of the two diagonals.
  - Show that two affine layers with nothing between them collapse into one, and compute the merged weights and bias.
  - Compute ReLU and its slope, and say why one corner between layers is enough to stop the collapse.
  - Solve XOR with two ReLU units followed by one straight cut.
prereqs:
  - 015-sigmoid-and-probability
  - 050-em-refit-and-converge
exercise: 051-why-layers-need-a-bend
---

# Why a stack of layers needs a bend

Two light switches control one lamp, wired so the lamp is on when exactly one switch is up. Write each setting as a point: switch one on the horizontal axis and switch two on the vertical axis, with 0 for down and 1 for up. The lamp is off at $(0,0)$ and $(1,1)$ and on at $(0,1)$ and $(1,0)$. You want a model that predicts the lamp from the two switches.

Logistic regression ("Sigmoid and probability") forms one weighted sum of the inputs and squashes it, so its decision boundary is a single straight line. On these four points that goes wrong, and no amount of training fixes it: no straight line puts the two off points on one side and the two on points on the other. The model is too stiff for the data.

This chapter answers two questions. Why can no line separate these points? And what is the smallest change that lets a line do it? The change is to let the model move the points first and cut afterwards, and the move has to include a bend. You will see why a stack of plain layers cannot supply one, and build the cheapest bend there is.

## What you already know

- An affine map is a weighted sum plus a constant: `W @ x + b`. This chapter says "affine" because the constant `b` is part of it, and "linear" only when `b` is absent.
- Logistic regression is one affine map followed by a sigmoid, so it cuts the plane along the line where the weighted sum is 0.
- A slope tells you how much the output moves per unit nudge of the input. For $x^2$ at $x=3$ the slope is 6.
- A finite difference estimates a slope: nudge the input up and down by a small $h$, and divide the change in output by $2h$.
- The chain rule multiplies slopes along a chain of functions.

## Why no straight line works

Look at Figure 1, left. The blue points are the two lamp-off settings and the purple points are the two lamp-on settings. Draw the segment between the two blue points and the segment between the two purple points. They cross at $(0.5, 0.5)$, the midpoint of each.

Now suppose a line did separate the classes. Both blue points lie on one side of it, and a side of a line is a half-plane, so every point between two points on it is also on it, including their midpoint. By the same argument the midpoint of the purple points lies on the other side. But it is the same point, and one point cannot be on both sides. So no line exists, and the orange line in the figure, like any other, misclassifies at least one point.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Left: the four switch settings as points, blue for lamp off and purple for lamp on, with the two diagonals crossing at (0.5, 0.5) and one orange line that misclassifies a point. Right: the same four points after two ReLU units, where the blue points both land at the origin and one straight orange line separates the classes.">
<defs><marker id="c057-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="292.0" y1="150.0" x2="340.0" y2="150.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow)"/>
<text x="316.0" y="172.0" text-anchor="middle" class="fig-small">ReLU</text>
<text x="316.0" y="188.0" text-anchor="middle" class="fig-small">units</text>
</svg>
Figure 1. Left: the blue pair and the purple pair have the same midpoint, (0.5, 0.5), so no line can put the blue pair on one side and the purple pair on the other. Right: after two ReLU units the blue points land together at the origin and the purple points move away, so one line cuts them apart.
```

The right panel shows the way out. If a first stage moves the points so that the blue pair lands together and the purple pair lands apart, a line can finish the job. A neural network does exactly this: hidden units re-draw the points, and the last layer is logistic regression on those redrawn points. The rest of the chapter asks what kind of move is allowed.

## A layer is a set of detectors

Before stacking layers, fix what one layer is. A layer holds several **units**. Each unit is a small detector: it takes a weighted sum of all the inputs and adds its own constant. For two switches, one unit might compute `1*x1 - 1*x2 + 0`, which is large when switch one is up and switch two is down.

Put the weights of the units in the rows of a matrix `W`, one row per unit, and the constants in a vector `b`. With 2 inputs and 3 units, `W` has 3 rows and 2 columns, so its shape is `(out, in)` = `(3, 2)`. Then `W @ x + b` gives the three detector readings at once. If your input is a row of data, as in most numpy code, you write `x @ W.T + b` to get the same numbers. Either way, entry $i$ of the answer comes from row $i$ of `W`.

## Why stacking plain layers does not help

The obvious fix is more layers: feed the first layer's output into a second one. Take $x = [2, -3]$. Let the first layer have rows $(1, 0)$ and $(0, 2)$ with constants $[1, 0]$, and let the second have the single row $[3, 1]$ with constant $4$. Run it:

- First layer: $[2, -6] + [1, 0] = [3, -6]$.
- Second layer: $3\cdot 3 + 1\cdot(-6) + 4 = 7$.

Now merge them. Multiply the weight matrices to get $[3, 1]\begin{bmatrix}1&0\\0&2\end{bmatrix} = [3, 2]$, and carry the first constants through the second layer to get $3\cdot 1 + 1\cdot 0 + 4 = 7$. The merged layer on $x$ gives $3\cdot 2 + 2\cdot(-3) + 7 = 7$, the same answer. In symbols, for any two affine layers:

$$
W_2 (W_1 x + b_1) + b_2 = (W_2 W_1)\, x + (W_2 b_1 + b_2) .
$$

Read it aloud: "the second layer applied to the first layer's output equals one layer whose weights are the product of the two weight matrices, and whose constant is the first constant pushed through the second layer, plus the second constant." In numpy that is two lines:

```python
W = W2 @ W1
b = W2 @ b1 + b2
```

The reason is plain. A line fed through a line is still a line: scaling a number and then scaling it again is one scaling. Figure 2 shows the two boxes merging into one. So a hundred layers with nothing between them are still one affine map, and a logistic regression on top still draws one straight cut.

```figure
<svg viewBox="0 0 600 282" role="img" aria-label="Top: x equals [2, minus 3] passes through two affine layers and gives 7. Middle: one merged layer with W equal to [3, 2] and b equal to 7 also gives 7. Bottom: with a ReLU between the layers the answer is 13, so the layers no longer merge.">
<defs><marker id="c057-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="8.0" y="20.0" text-anchor="start" class="fig-bold">Two affine layers</text>
<text x="30.0" y="68.0" text-anchor="middle" class="fig-small">[2, −3]</text>
<line x1="62.0" y1="64.0" x2="100.0" y2="64.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<rect x="102" y="44" width="110" height="40" rx="6" class="fill-soft-a stroke-a"/><text x="157.0" y="68.0" text-anchor="middle" class="fig-small">W1, b1</text>
<line x1="214.0" y1="64.0" x2="252.0" y2="64.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<text x="285.0" y="68.0" text-anchor="middle" class="fig-small">[3, −6]</text>
<line x1="318.0" y1="64.0" x2="356.0" y2="64.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<rect x="358" y="44" width="110" height="40" rx="6" class="fill-soft-a stroke-a"/><text x="413.0" y="68.0" text-anchor="middle" class="fig-small">W2, b2</text>
<line x1="470.0" y1="64.0" x2="508.0" y2="64.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<text x="530.0" y="68.0" text-anchor="middle" class="fig-small">7</text>
<text x="8.0" y="118.0" text-anchor="start" class="fig-bold">One merged affine layer</text>
<text x="30.0" y="154.0" text-anchor="middle" class="fig-small">[2, −3]</text>
<line x1="62.0" y1="150.0" x2="100.0" y2="150.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<rect x="102" y="130" width="366" height="40" rx="6" class="fill-soft-b stroke-b"/><text x="285.0" y="154.0" text-anchor="middle" class="fig-small">W = [3, 2], b = 7</text>
<line x1="470.0" y1="150.0" x2="508.0" y2="150.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<text x="530.0" y="154.0" text-anchor="middle" class="fig-small">7</text>
<text x="8.0" y="208.0" text-anchor="start" class="fig-bold">Insert a ReLU between them</text>
<text x="30.0" y="244.0" text-anchor="middle" class="fig-small">[2, −3]</text>
<line x1="62.0" y1="240.0" x2="100.0" y2="240.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<rect x="102" y="220" width="80" height="40" rx="6" class="fill-soft-a stroke-a"/><text x="142.0" y="244.0" text-anchor="middle" class="fig-small">W1, b1</text>
<line x1="184.0" y1="240.0" x2="204.0" y2="240.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<text x="228.0" y="244.0" text-anchor="middle" class="fig-small">[3, −6]</text>
<line x1="252.0" y1="240.0" x2="272.0" y2="240.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<rect x="274" y="220" width="60" height="40" rx="6" class="fill-soft-d stroke-d"/><text x="304.0" y="244.0" text-anchor="middle" class="fig-small">ReLU</text>
<line x1="336.0" y1="240.0" x2="356.0" y2="240.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<text x="380.0" y="244.0" text-anchor="middle" class="fig-small">[3, 0]</text>
<line x1="404.0" y1="240.0" x2="424.0" y2="240.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<rect x="426" y="220" width="80" height="40" rx="6" class="fill-soft-a stroke-a"/><text x="466.0" y="244.0" text-anchor="middle" class="fig-small">W2, b2</text>
<line x1="508.0" y1="240.0" x2="526.0" y2="240.0" class="stroke-muted" stroke-width="1.8" marker-end="url(#c057-arrow2)"/>
<text x="552.0" y="244.0" text-anchor="middle" class="fig-bold">13</text>
</svg>
Figure 2. Two affine layers give 7, and so does the single merged layer, so the middle disappears. Put a ReLU between the layers and the answer changes to 13, so no single affine layer can stand in for the pair.
```

A narrow middle layer makes it worse. The merged matrix $W_2 W_1$ cannot have more independent directions in its output (its **rank**) than the middle layer has units, so a width-1 middle layer squeezes every input onto a single line before the second layer sees it.

> [!KEY]
> Without something between them, any stack of affine layers is one affine layer. Depth buys nothing until a layer does something that is not affine.

## Bend each number with ReLU

The cheapest thing that is not affine is a comparison. **ReLU**, short for rectified linear unit, keeps positive numbers and replaces negative ones with 0. Apply it after the first layer of the example, to the readings $[3, -6]$, and you get $[3, 0]$. The second layer now gives $3\cdot 3 + 0 + 4 = 13$ instead of 7, as the bottom of Figure 2 shows. The answer changed, so the two layers no longer merge. One corner was enough to break the collapse.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Top: the ReLU curve, flat at zero for negative inputs and rising with slope one for positive inputs, with the inputs 2, minus 3 and minus 2 landing on 2, 0 and 0. Bottom: its slope, 0 on the left of the corner and 1 on the right.">

<text x="60.0" y="18.0" text-anchor="start" class="fig-bold">relu(x)</text>
<line x1="60.0" y1="139.7" x2="540.0" y2="139.7" class="stroke-muted"/>
<text x="75.0" y="153.7" text-anchor="middle" class="fig-small">-3</text>
<text x="150.0" y="153.7" text-anchor="middle" class="fig-small">-2</text>
<text x="225.0" y="153.7" text-anchor="middle" class="fig-small">-1</text>
<text x="300.0" y="153.7" text-anchor="middle" class="fig-small">0</text>
<text x="375.0" y="153.7" text-anchor="middle" class="fig-small">1</text>
<text x="450.0" y="153.7" text-anchor="middle" class="fig-small">2</text>
<text x="525.0" y="153.7" text-anchor="middle" class="fig-small">3</text>
<path d="M60.0,139.7 L300.0,139.7 L540.0,30.0" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="300.0" cy="139.7" r="5" class="fill-ink"/>
<text x="308.0" y="99.7" text-anchor="start" class="fig-small">the corner</text>
<line x1="304.0" y1="103.7" x2="301.0" y2="131.7" class="stroke-muted" stroke-width="1.2"/>
<line x1="450.0" y1="139.7" x2="450.0" y2="71.1" class="stroke-muted dashed" stroke-width="1.2"/>
<circle cx="450.0" cy="139.7" r="4" class="fill-a"/>
<circle cx="450.0" cy="71.1" r="5" class="fill-b"/>
<line x1="75.0" y1="139.7" x2="75.0" y2="139.7" class="stroke-muted dashed" stroke-width="1.2"/>
<circle cx="75.0" cy="139.7" r="4" class="fill-a"/>

<line x1="150.0" y1="139.7" x2="150.0" y2="139.7" class="stroke-muted dashed" stroke-width="1.2"/>
<circle cx="150.0" cy="139.7" r="4" class="fill-a"/>

<text x="462.0" y="89.1" text-anchor="start" class="fig-small">2 → 2</text>
<text x="75.0" y="127.7" text-anchor="middle" class="fig-small">−3 → 0</text>
<text x="150.0" y="127.7" text-anchor="middle" class="fig-small">−2 → 0</text>
<text x="60.0" y="196.0" text-anchor="start" class="fig-bold">slope of relu</text>
<line x1="60.0" y1="255.0" x2="540.0" y2="255.0" class="stroke-muted"/>
<path d="M60.0,255.0 L300.0,255.0" fill="none" class="stroke-b" stroke-width="2.2"/>
<path d="M300.0,200.0 L540.0,200.0" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="300.0" cy="255.0" r="4" class="fill-b"/>
<circle cx="300.0" cy="200.0" r="4" class="fill-paper"/>
<text x="180.0" y="247.0" text-anchor="middle" class="fig-small">slope 0: a nudge does nothing</text>
<text x="420.0" y="216.0" text-anchor="middle" class="fig-small">slope 1: a nudge passes through</text>
<text x="60.0" y="271.0" text-anchor="start" class="fig-small"></text>
</svg>
Figure 3. ReLU has one corner, at 0. Blue dots are the inputs from the running example and orange dots are where they land. Notice that two of the three inputs are cut to 0, and that the slope below the curve is only ever 0 or 1.
```

As a formula, with the slope next to it:

$$
\operatorname{relu}(x) = \max(x, 0), \qquad
\operatorname{relu}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases} .
$$

Read it aloud: "ReLU returns the larger of $x$ and 0, and its slope is 1 where the input passes through and 0 where it is cut off." At the corner $x = 0$ the slope is undefined, and the checks skip that point. For the readings $[2, -3, -2]$ the outputs are $[2, 0, 0]$ and the slopes are $[1, 0, 0]$:

```python
np.maximum(x, 0.0)
(x > 0).astype(float)
```

The slope will matter later: a later chapter uses it to decide how to change the weights, and a slope of 0 means a nudge to that number does nothing.

You could ask why not use a smooth bend such as the sigmoid. You can, and a later chapter meets a smooth one. ReLU wins as a first choice because it is one comparison, and because the sigmoid flattens at both ends so its slope there is nearly 0.

## Solve XOR with two bends

Use two units with ReLU: $h_1 = \operatorname{relu}(x_1 - x_2)$ and $h_2 = \operatorname{relu}(x_2 - x_1)$. The first fires when switch one leads, the second when switch two leads.

| $(x_1, x_2)$ | readings $(x_1 - x_2,\ x_2 - x_1)$ | after ReLU $(h_1, h_2)$ | $h_1 + h_2$ |
|---|---|---|---|
| $(0, 0)$ | $(0, 0)$ | $(0, 0)$ | 0 |
| $(1, 1)$ | $(0, 0)$ | $(0, 0)$ | 0 |
| $(0, 1)$ | $(-1, 1)$ | $(0, 1)$ | 1 |
| $(1, 0)$ | $(1, -1)$ | $(1, 0)$ | 1 |

Both lamp-off settings land on $(0,0)$ and both lamp-on settings land on the far side of the line $h_1 + h_2 = 0.5$, as in Figure 1, right. One affine cut, the last layer, finishes the job: the sign of $h_1 + h_2 - 0.5$ is the answer. The hidden layer computed $|x_1 - x_2|$, how far apart the switches are, which an affine map alone can never produce.

> [!KEY]
> No line separates XOR because both classes share the midpoint $(0.5, 0.5)$. Stacked affine layers merge into one, so they cannot help. A bend between layers, with ReLU as the cheapest choice, lets the hidden layer move the points so that one final line separates them.

> [!RECAP]
> - A unit is one row of `W`; `W` has shape `(out, in)`, and a layer is `x @ W.T + b` for row data.
> - Two affine layers merge into `W2 @ W1` and `W2 @ b1 + b2`, so stacking without a bend adds nothing.
> - ReLU is `np.maximum(x, 0.0)`, with slope `(x > 0).astype(float)`.
> - A ReLU between the layers changes the answer, so the layers no longer merge.
> - Two ReLU units turn XOR into a problem for one straight line.

## Build it step by step

Build three functions, in this order.

1. **`relu(x)`** takes an array of any shape and returns the elementwise larger of the entry and 0, in the same shape. Reach for `np.maximum`, because Python's `max` does not work on arrays. Write it like this:

    ```python
    def relu(x):
        # Compare every entry with 0.0.
    ```

2. **`relu_grad(x)`** returns 1.0 where `x > 0` and 0.0 where `x < 0`, as floats. The comparison gives booleans, so cast them with `.astype(float)`, and a mask of booleans fails the finite-difference check. The checks skip `x == 0`. Write it like this:

    ```python
    def relu_grad(x):
        # Compare with 0, then cast to float.
    ```

3. **`compose_affine(W1, b1, W2, b2)`** takes two affine layers and returns the pair `(W, b)` of the single layer that does the same job. Mind the order: the first layer's matrix sits on the right of the product. The result must match applying the layers one after the other. Write it like this:

    ```python
    def compose_affine(W1, b1, W2, b2):
        # Multiply the matrices, then push b1 through the second layer.
    ```

> [!TIP]
> Check any slope you write with `(f(x + h) - f(x - h)) / (2 * h)` at `h = 1e-5`. If the two disagree away from a corner, the slope is wrong.

## Where this breaks

A ReLU unit whose reading is negative for every input outputs 0 for all of them, and its slope is 0 too, so nothing flowing back can change it. People call this a dead unit, and later chapters on training show how to avoid it. ReLU also leaves the final layer alone in most networks: a classifier or a predictor usually ends with a plain affine layer.

## The exercise

You implement `relu`, `relu_grad`, and `compose_affine`. The checks confirm `relu` on a row, a matrix, and the readings $[2, -3, -2]$, and compare `relu_grad` with finite differences away from 0. They verify `compose_affine` on the hand example (weights $[3, 2]$, constant 7), against applying two random layers one after the other, and against the rank limit of a narrow middle layer. The last two checks use your `relu` to solve XOR with the two hidden units above, and show that a ReLU between two layers changes the answer.
