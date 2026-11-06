---
title: "The residual block, forward and backward"
slug: 073-the-residual-block
part: "Part 5 — Vision and sequences"
summary: A residual block adds a correction to its input, so passing the input through is the default, and its plus sign sends the gradient down both paths.
minutes: 55
gpu: false
objectives:
  - Explain why a 56-layer plain network can train worse than a 20-layer one, and how adding the input back onto the output makes "do nothing" the default.
  - Compute a residual block's forward pass by hand, and show that a zero branch returns the input exactly.
  - Explain why the gradient at the block's input is the incoming gradient plus a term from the branch, and why the ReLU mask acts as a switch.
  - Build the forward and backward passes in numpy, and verify the backward pass against finite differences.
prereqs:
  - 052-forward-pass-and-gelu
  - 054-backprop-the-graph-and-reverse-order
  - 055-backprop-array-operations
  - 057-backprop-mlp-and-gradient-check
  - 061-signal-size-through-layers
  - 072-convolution-backward
exercise: 073-the-residual-block
---

# The residual block, forward and backward

You train a network with 20 layers and record its error on the training set. Then you stack 36 more layers on top, keep the data and the optimiser the same, and train again. The 56-layer network ends with a higher training error than the 20-layer one. This result, measured on the CIFAR-10 image set, led to residual networks.

It is not overfitting, because the error on the training set itself is worse. It is also strange, because the deep network could copy the 20 trained layers and let the 36 extra layers pass their input along unchanged, which would match the shallow network exactly. So a solution at least as good exists, and gradient descent fails to find it.

This chapter explains why "pass the input along unchanged" is a hard setting for an ordinary layer, and how one change makes it the easy default: add the block's input back onto its output. You build the forward pass of that block. Then you send gradients back through it, which is harder than for a plain layer because the input reaches the output by two routes.

## What you already know

- A linear layer turns each row of `x` into `x @ W.T + b` ("Run a network forward, with a sharp or smooth bend").
- A plain layer with small random weights shrinks its signal toward zero ("Why a deep network's signal shrinks or explodes").
- The chain rule multiplies slopes along a path, and a number that reaches the loss by several paths collects the slope of each ("Walking slopes backwards through a calculation").
- A gradient check compares a computed gradient with a finite-difference estimate ("Putting it together: gradients for a whole network").

## Why an ordinary layer struggles to do nothing

Picture a chain of 36 people, each handed a page and asked to hand it on. In a plain network each person redraws the page from memory. Any small slip is passed on and added to by the next person, so "do nothing" needs every person to be exact. A plain block's output comes only from its weights, so handing the input along needs one precise setting of every weight.

Why not start the extra layers at that setting? A block with a ReLU in the middle has no single identity matrix, because the ReLU wipes out negative entries. And the usual starting point, small random weights, does not copy anything: a plain block with small weights outputs numbers close to zero, the opposite of the input.

Change the picture. Hand the page along untouched, and let each person attach a sticky note with a correction. If nobody writes a note, the page arrives as it started, and writing no note is the easiest thing to do, because zero is where small weights already sit. Figure 1 draws both versions. The page is $x$, the sticky note is $F(x)$, and the result is $y$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A plain block with three steps, above a residual block whose input also travels around those steps to a plus sign">
<defs><marker id="c081-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<defs><marker id="c081-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<text x="20" y="22" class="fig-bold">Plain block</text>
<text x="22" y="65" class="fig-mono">x</text>
<line x1="34.0" y1="60.0" x2="68.0" y2="60.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<rect x="70.0" y="40.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="120.0" y="64.5" text-anchor="middle">linear</text>
<line x1="170.0" y1="60.0" x2="188.0" y2="60.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<rect x="190.0" y="40.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="240.0" y="64.5" text-anchor="middle">ReLU</text>
<line x1="290.0" y1="60.0" x2="308.0" y2="60.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<rect x="310.0" y="40.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="360.0" y="64.5" text-anchor="middle">linear</text>
<line x1="410.0" y1="60.0" x2="500.0" y2="60.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<text x="508" y="65" class="fig-mono">y = F(x)</text>
<text x="20" y="108" class="fig-small">Zero weights give y = 0: the input is gone.</text>
<text x="20" y="150" class="fig-bold">Residual block</text>
<text x="22" y="213" class="fig-mono">x</text>
<line x1="34" y1="208" x2="68" y2="208" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<rect x="70.0" y="188.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="120.0" y="212.5" text-anchor="middle">linear</text>
<line x1="170.0" y1="208.0" x2="188.0" y2="208.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<rect x="190.0" y="188.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="240.0" y="212.5" text-anchor="middle">ReLU</text>
<line x1="290.0" y1="208.0" x2="308.0" y2="208.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<rect x="310.0" y="188.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="360.0" y="212.5" text-anchor="middle">linear</text>
<line x1="410.0" y1="208.0" x2="474.0" y2="208.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<circle cx="490" cy="208" r="15" class="fill-paper stroke-ink"/>
<text x="490" y="214" text-anchor="middle" class="fig-bold">+</text>
<line x1="505.0" y1="208.0" x2="556.0" y2="208.0" class="stroke-ink" marker-end="url(#c081-arrow)"/>
<text x="560" y="213" class="fig-mono">y</text>
<path d="M48,208 L48,162 L490,162 L490,192" fill="none" class="stroke-a" marker-end="url(#c081-arrow-a)"/>
<text x="269" y="156" text-anchor="middle" class="fig-small">shortcut: x travels unchanged</text>
<text x="240" y="250" text-anchor="middle" class="fig-small">branch F(x): the correction</text>
<text x="20" y="286" class="fig-small">Zero weights in the last layer give F(x) = 0, so y = x. No ReLU after the plus sign.</text>
</svg>
Figure 1. The plain block (top) has one path, so a zero last layer erases the input. The residual block (bottom) adds a second path, the shortcut in blue, that carries x around the three steps to the plus sign. Notice that nothing sits between the plus sign and y.
```

The block in the lower half is a **residual block**. Its output is the input plus a correction:

$$
y = x + F(x)
$$

Read it aloud: the block's output $y$ is its input $x$ plus whatever $F$ computes from $x$. $F$ is the **residual branch**, the layers that write the sticky note. The arc that carries $x$ around them to the plus sign is the **shortcut**.

## Work one block forward

Take two examples with two features each, one example per row:

$$
x = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}
$$

This pair is the running example for the whole chapter. The branch has two linear layers with a ReLU between them. The first has $W_1$ equal to the 2 by 2 identity and $b_1 = [0, -1]$. The second has $W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ and $b_2 = [0.1, 0]$. The steps are:

1. The first layer gives $h_1 = x W_1^{\top} + b_1 = \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$, because the identity leaves $x$ alone and $b_1$ subtracts 1 from the second column.
2. The ReLU turns the $-1$ into 0, so $a_1 = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$.
3. The second layer gives $h_2 = a_1 W_2^{\top} + b_2 = \begin{bmatrix} 2.1 & 0.5 \\ 0.1 & 1.0 \end{bmatrix}$. This is $F(x)$, the correction.
4. The addition gives $y = x + h_2 = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.

The top row of Figure 2 shows step 4 as grids. Notice that example 2 keeps its negative entry, $-0.9$, because the block applies no ReLU after the plus sign. A ReLU there would clip negative entries, so even a zero correction would no longer return $x$.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Two two-by-two grids of numbers added together: the input plus the branch output gives the block output, and below, the same input plus an all-zero branch gives the input back">
<rect x="60" y="40.0" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="83.0" y="63.4" text-anchor="middle" class="fig-mono">1</text>
<rect x="106" y="40.0" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="129.0" y="63.4" text-anchor="middle" class="fig-mono">2</text>
<rect x="60" y="76.8" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="83.0" y="100.2" text-anchor="middle" class="fig-mono">-1</text>
<rect x="106" y="76.8" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="129.0" y="100.2" text-anchor="middle" class="fig-mono">3</text>
<text x="170" y="74" text-anchor="middle" class="fig-bold">+</text>
<rect x="190" y="40.0" width="46" height="36.8" class="fill-soft-b stroke-b"/>
<text x="213.0" y="63.4" text-anchor="middle" class="fig-mono">2.1</text>
<rect x="236" y="40.0" width="46" height="36.8" class="fill-soft-b stroke-b"/>
<text x="259.0" y="63.4" text-anchor="middle" class="fig-mono">0.5</text>
<rect x="190" y="76.8" width="46" height="36.8" class="fill-soft-b stroke-b"/>
<text x="213.0" y="100.2" text-anchor="middle" class="fig-mono">0.1</text>
<rect x="236" y="76.8" width="46" height="36.8" class="fill-soft-b stroke-b"/>
<text x="259.0" y="100.2" text-anchor="middle" class="fig-mono">1</text>
<text x="300" y="74" text-anchor="middle" class="fig-bold">=</text>
<rect x="320" y="40.0" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="343.0" y="63.4" text-anchor="middle" class="fig-mono">3.1</text>
<rect x="366" y="40.0" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="389.0" y="63.4" text-anchor="middle" class="fig-mono">2.5</text>
<rect x="320" y="76.8" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="343.0" y="100.2" text-anchor="middle" class="fig-mono">-0.9</text>
<rect x="366" y="76.8" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="389.0" y="100.2" text-anchor="middle" class="fig-mono">4</text>
<text x="106" y="32" text-anchor="middle" class="fig-small">input x</text>
<text x="236" y="32" text-anchor="middle" class="fig-small">branch F(x)</text>
<text x="366" y="32" text-anchor="middle" class="fig-small">output y</text>
<rect x="60" y="170.0" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="83.0" y="193.4" text-anchor="middle" class="fig-mono">1</text>
<rect x="106" y="170.0" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="129.0" y="193.4" text-anchor="middle" class="fig-mono">2</text>
<rect x="60" y="206.8" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="83.0" y="230.2" text-anchor="middle" class="fig-mono">-1</text>
<rect x="106" y="206.8" width="46" height="36.8" class="fill-soft-a stroke-a"/>
<text x="129.0" y="230.2" text-anchor="middle" class="fig-mono">3</text>
<text x="170" y="204" text-anchor="middle" class="fig-bold">+</text>
<rect x="190" y="170.0" width="46" height="36.8" class="fill-soft-d stroke-d"/>
<text x="213.0" y="193.4" text-anchor="middle" class="fig-mono">0</text>
<rect x="236" y="170.0" width="46" height="36.8" class="fill-soft-d stroke-d"/>
<text x="259.0" y="193.4" text-anchor="middle" class="fig-mono">0</text>
<rect x="190" y="206.8" width="46" height="36.8" class="fill-soft-d stroke-d"/>
<text x="213.0" y="230.2" text-anchor="middle" class="fig-mono">0</text>
<rect x="236" y="206.8" width="46" height="36.8" class="fill-soft-d stroke-d"/>
<text x="259.0" y="230.2" text-anchor="middle" class="fig-mono">0</text>
<text x="300" y="204" text-anchor="middle" class="fig-bold">=</text>
<rect x="320" y="170.0" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="343.0" y="193.4" text-anchor="middle" class="fig-mono">1</text>
<rect x="366" y="170.0" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="389.0" y="193.4" text-anchor="middle" class="fig-mono">2</text>
<rect x="320" y="206.8" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="343.0" y="230.2" text-anchor="middle" class="fig-mono">-1</text>
<rect x="366" y="206.8" width="46" height="36.8" class="fill-soft-c stroke-c"/>
<text x="389.0" y="230.2" text-anchor="middle" class="fig-mono">3</text>
<text x="106" y="162" text-anchor="middle" class="fig-small">input x</text>
<text x="236" y="162" text-anchor="middle" class="fig-small">branch with W2 = 0, b2 = 0</text>
<text x="366" y="162" text-anchor="middle" class="fig-small">output y = x</text>
<text x="430" y="70" class="fig-small">row 1 of x is example 1,</text><text x="430" y="88" class="fig-small">row 2 is example 2</text>
<text x="430" y="200" class="fig-small">nothing added, so</text><text x="430" y="218" class="fig-small">nothing changes</text>
</svg>
Figure 2. The worked example as grids. Top row: input, plus the branch output, equals the block output, with the numbers from the text. Bottom row: with W2 and b2 at zero the branch adds nothing, so every cell of y equals the matching cell of x.
```

Zero the last layer, with $W_2 = 0$ and $b_2 = 0$. Then $h_2$ is all zeros whatever $a_1$ holds, and $y = x$ exactly, as the bottom row of Figure 2 shows. A plain block with the same zeros returns $F(x) = 0$ and loses the input. With small weights instead of zero the difference is as sharp. Take five random inputs of width 8 and weights drawn with standard deviation 0.05. The residual block moves no entry by more than 0.06, while the plain block's output is near zero and differs from the input by 2.46, the size of the largest input entry. In numpy:

```python
h1 = x @ W1.T + b1
a1 = np.maximum(h1, 0.0)
h2 = a1 @ W2.T + b2
y = x + h2
```

Return to the 56-layer network. Replace each layer with a residual block whose last layer starts at zero. Every one of the 36 extra blocks starts as the identity, so the deep network begins as a copy of the shallow computation, and training only has to find corrections that help. An identity start is a good starting point, not a guarantee, because training moves the correction away from zero.

> [!KEY]
> The deep network failed because "do nothing" is a precise setting for a plain block, and small random weights miss it. A residual block turns "do nothing" into a zero correction, which small weights already land near.

## A plus sign splits the gradient in two

Now the backward pass. The loss reports, for every number in the output, how much it would like that number to change: the gradient `dY`. Training needs how each weight and bias should change, and what message goes to the layer below. If you follow only the branch, you lose the shortcut's contribution, and that message is wrong.

Start with one number. Let $x = 2$ and let the branch be $f(x) = 0.5 x^2$. The block computes $y = x + f(x) = 2 + 2 = 4$. Nudge $x$ to 2.001 and $y$ becomes about 4.003, three times the nudge. The slope of $y$ is $1 + f'(x) = 1 + x = 3$ here. One part of that 3 comes through the shortcut, and the other two come through the branch. As a formula:

$$
\frac{dy}{dx} = 1 + f'(x)
$$

Read it aloud: the output's slope with respect to the input is 1, from the shortcut, plus the branch's own slope. When the message $dY$ arrives, each path multiplies it by that path's slope. The shortcut multiplies by 1, so it returns $dY$ unchanged, and the branch multiplies by its own slopes. Figure 3 shows this on the running example, with $dY$ all ones, meaning "raise every output number a little".

```figure
<svg viewBox="0 0 620 360" role="img" aria-label="Gradient flowing backward through a residual block: it arrives at the plus sign, splits into a shortcut path that carries it unchanged and a branch path through W2, the ReLU mask, and W1, then adds back together at the input">
<defs><marker id="c081-back-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<circle cx="530" cy="150" r="15" class="fill-paper stroke-ink"/><text x="530" y="156" text-anchor="middle" class="fig-bold">+</text>
<circle cx="90" cy="150" r="15" class="fill-paper stroke-ink"/><text x="90" y="156" text-anchor="middle" class="fig-bold">+</text>
<text x="595" y="140" text-anchor="middle" class="fig-bold">dY</text>
<line x1="610" y1="150" x2="546" y2="150" class="stroke-d" marker-end="url(#c081-back-arrow)"/>
<path d="M530,134 L530,70 L90,70 L90,133" fill="none" class="stroke-d" stroke-width="2.5" marker-end="url(#c081-back-arrow)"/>
<text x="310" y="60" text-anchor="middle" class="fig-bold">shortcut: dY passes unchanged</text>
<text x="310" y="86" text-anchor="middle" class="fig-small fig-mono">[[1, 1], [1, 1]]</text>
<line x1="530" y1="166" x2="530" y2="230" class="stroke-d"/>
<line x1="530" y1="230" x2="516" y2="230" class="stroke-d" marker-end="url(#c081-back-arrow)"/>
<rect x="430.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="472.0" y="234.5" text-anchor="middle">× W2</text>
<line x1="430" y1="230" x2="396" y2="230" class="stroke-d" marker-end="url(#c081-back-arrow)"/>
<rect x="310.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="352.0" y="234.5" text-anchor="middle">× mask</text>
<line x1="310" y1="230" x2="276" y2="230" class="stroke-d" marker-end="url(#c081-back-arrow)"/>
<rect x="190.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="232.0" y="234.5" text-anchor="middle">× W1</text>
<path d="M190,230 L90,230 L90,167" fill="none" class="stroke-d" marker-end="url(#c081-back-arrow)"/>
<text x="522" y="264" text-anchor="middle" class="fig-small">dh2 = dY</text>
<text x="412" y="264" text-anchor="middle" class="fig-small">da1</text>
<text x="412.0" y="280" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="412.0" y="295" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="292" y="264" text-anchor="middle" class="fig-small">dh1</text>
<text x="292.0" y="280" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="292.0" y="295" text-anchor="middle" class="fig-mono fig-small">0  0.5</text>
<text x="150" y="264" text-anchor="middle" class="fig-small">branch term</text>
<text x="150.0" y="280" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="150.0" y="295" text-anchor="middle" class="fig-mono fig-small">0  0.5</text>
<text x="120" y="146" class="fig-bold">dx = dY + branch term</text>
<text x="120" y="166" text-anchor="start" class="fig-mono fig-small">3  1.5</text>
<text x="120" y="181" text-anchor="start" class="fig-mono fig-small">1  1.5</text>
</svg>
Figure 3. The backward pass of the running example, with dY all ones. The gradient splits at the plus sign. The shortcut carries it to the input unchanged, and the branch carries it through W2, the mask, and W1 as a smaller term. The two meet at the input and add to dx.
```

## Walk the branch backward

The plus sign gives the branch the whole message, so $dh_2 = dY$. The entry $W_2[i, j]$ multiplies $a_1[n, j]$ on its way into $h_2[n, i]$ for every example $n$, so its gradient adds up $dh_2[n, i] \cdot a_1[n, j]$ over the examples. That sum is one entry of a matrix product, so $dW_2 = dh_2^{\top} a_1 = \begin{bmatrix} 1 & 3 \\ 1 & 3 \end{bmatrix}$. The entry 3 is $1 \cdot 1 + 1 \cdot 2$, the second column of $a_1$ summed over both examples. The bias is added to every row, so its gradient adds the message over the rows: $db_2 = [2, 2]$.

To pass the message down, notice that $a_1[n, j]$ feeds $h_2[n, i]$ through $W_2[i, j]$, so $da_1 = dh_2 W_2 = \begin{bmatrix} 2 & 0.5 \\ 2 & 0.5 \end{bmatrix}$.

The ReLU is a switch. Its slope is 1 where the unit is on and 0 where it is off, because nudging an off unit's input changes nothing at its output. The mask $[h_1 > 0]$ holds a 1 for each unit that is on:

$$
dh_1 = da_1 \odot [h_1 > 0]
$$

The symbol $\odot$ multiplies entry by entry. The one negative $h_1$ is in example 2, feature 1, so the mask is $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and $dh_1 = \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix}$, as Figure 4 shows.

```figure
<svg viewBox="0 0 620 230" role="img" aria-label="The ReLU mask as a switch: the sign of h1 gives the mask, and the mask multiplies the incoming gradient so the cell where h1 is negative is blocked">
<text x="66" y="52" text-anchor="middle" class="fig-small">h1</text>
<rect x="20" y="64" width="46" height="36" class="fill-soft-a stroke-a"/>
<text x="43.0" y="87.0" text-anchor="middle" class="fig-mono">1</text>
<rect x="66" y="64" width="46" height="36" class="fill-soft-a stroke-a"/>
<text x="89.0" y="87.0" text-anchor="middle" class="fig-mono">1</text>
<rect x="20" y="100" width="46" height="36" class="fill-sunk stroke-muted"/>
<text x="43.0" y="123.0" text-anchor="middle" class="fig-mono">-1</text>
<rect x="66" y="100" width="46" height="36" class="fill-soft-a stroke-a"/>
<text x="89.0" y="123.0" text-anchor="middle" class="fig-mono">2</text>
<text x="216" y="52" text-anchor="middle" class="fig-small">mask: h1 > 0</text>
<rect x="170" y="64" width="46" height="36" class="fill-soft-a stroke-a"/>
<text x="193.0" y="87.0" text-anchor="middle" class="fig-mono">1</text>
<rect x="216" y="64" width="46" height="36" class="fill-soft-a stroke-a"/>
<text x="239.0" y="87.0" text-anchor="middle" class="fig-mono">1</text>
<rect x="170" y="100" width="46" height="36" class="fill-sunk stroke-muted"/>
<text x="193.0" y="123.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="216" y="100" width="46" height="36" class="fill-soft-a stroke-a"/>
<text x="239.0" y="123.0" text-anchor="middle" class="fig-mono">1</text>
<text x="366" y="52" text-anchor="middle" class="fig-small">da1 (arriving)</text>
<rect x="320" y="64" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="343.0" y="87.0" text-anchor="middle" class="fig-mono">2</text>
<rect x="366" y="64" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="389.0" y="87.0" text-anchor="middle" class="fig-mono">0.5</text>
<rect x="320" y="100" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="343.0" y="123.0" text-anchor="middle" class="fig-mono">2</text>
<rect x="366" y="100" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="389.0" y="123.0" text-anchor="middle" class="fig-mono">0.5</text>
<text x="526" y="52" text-anchor="middle" class="fig-small">dh1 = da1 × mask</text>
<rect x="480" y="64" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="503.0" y="87.0" text-anchor="middle" class="fig-mono">2</text>
<rect x="526" y="64" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="549.0" y="87.0" text-anchor="middle" class="fig-mono">0.5</text>
<rect x="480" y="100" width="46" height="36" class="fill-sunk stroke-muted"/>
<text x="503.0" y="123.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="526" y="100" width="46" height="36" class="fill-soft-d stroke-d"/>
<text x="549.0" y="123.0" text-anchor="middle" class="fig-mono">0.5</text>
<text x="143" y="106" text-anchor="middle" class="fig-bold">→</text>
<text x="293" y="106" text-anchor="middle" class="fig-bold">×</text>
<text x="443" y="106" text-anchor="middle" class="fig-bold">=</text>
<text x="20" y="170" class="fig-small">Grey cells are switched off. The entry h1 = -1 (example 2, feature 1) is negative, so its mask is 0 and the</text>
<text x="20" y="188" class="fig-small">incoming 2 becomes 0. Blue cells are on and pass their gradient unchanged.</text>
</svg>
Figure 4. The ReLU mask as a switch, with the numbers from the running example. The one entry with a negative h1 is switched off, so the gradient that arrives there (2) is replaced by 0 in dh1. Every other entry passes through unchanged.
```

The first layer repeats the same two moves: $dW_1 = dh_1^{\top} x = \begin{bmatrix} 2 & 4 \\ 0 & 2.5 \end{bmatrix}$ and $db_1 = [2, 1]$. Its message to the input is $dh_1 W_1$, which equals $dh_1$ because $W_1$ is the identity. Last, the two paths meet at the input:

$$
dx = dY + dh_1 W_1
$$

On the example, $dx = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} + \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix} = \begin{bmatrix} 3 & 1.5 \\ 1 & 1.5 \end{bmatrix}$. The ones from the shortcut sit inside every entry of $dx$, and no switch can turn them off. As numpy:

```python
dW2 = dY.T @ a1
da1 = dY @ W2
dh1 = da1 * (h1 > 0)
dx = dY + dh1 @ W1
```

> [!KEY]
> Because a plus sign sends the incoming gradient down both paths, the input's gradient is `dY` from the shortcut plus the branch's term. The shortcut's share never goes through a ReLU switch.

The forward pass computes `x @ W.T`, but the backward pass computes `dY @ W`, with no transpose. A weight of shape `(d_out, d_in)` carries `d_in` numbers to `d_out` numbers, so forward needs `W.T` to line up with rows of length `d_in`. The message coming back has `d_out` numbers per row and must leave with `d_in`, which `W` does.

> [!WARNING]
> In a block with square weights both `dY @ W` and `dY @ W.T` run, and only one is right. Choose by reasoning about the forward pass, because the error will not announce itself.

## Check it with finite differences

A formula for a gradient can be wrong while the code still runs. A **finite difference** gives an independent answer: nudge one input number up by a tiny amount, nudge it down by the same amount, and divide the change in the loss by twice the nudge. Pick the loss $L = \sum (dY \odot y)$, whose gradient with respect to $y$ is exactly `dY`, so its gradient with respect to $x$ must equal your backward pass. Repeat for each weight and bias. Keep the test points away from entries where $h_1$ is within the nudge of zero, because the ReLU has a corner there.

> [!RECAP]
> Hold these points before you write code:
>
> - A residual block computes $y = x + F(x)$ with no ReLU after the sum, so a zero correction returns the input exactly. On the running example $y = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.
> - The plus sign sends `dY` to the branch and to the shortcut, so `dx = dY + dh1 @ W1`, which is `[[3, 1.5], [1, 1.5]]` on the example.
> - A weight gradient is the upstream gradient, transposed, times the layer's input. A bias gradient sums over the rows.
> - The ReLU mask `h1 > 0` passes the gradient where the unit was on and blocks it where off.
> - Forward multiplies by `W.T`, and backward multiplies by `W`.

## Build it step by step

The checks call the functions in this order:

1. **`relu(z)`** returns a new array of the same shape with every negative entry replaced by 0, using `np.maximum(z, 0.0)`. The usual mistake is to change `z` in place.
2. **`residual_block_forward(x, params)`** takes `x` of shape `(n, d)` and `params` holding `W1`, `b1`, `W2`, `b2`. It returns `(y, cache)`, with `cache` the tuple `(x, W1, W2, h1, a1)` in that order. The usual mistake is `W1 @ x`, or a ReLU after the sum.
3. **`identity_gap(x, params, residual)`** returns the largest absolute difference between the block's output and `x`, as a `float`. When `residual` is false the output is a plain block's `F(x)`, which is `y - x` for your forward pass. The usual mistake is to return a plain gap of zero because you reused `y`.
4. **`relu_mask(h)`** returns a float array with 1.0 where `h > 0` and 0.0 elsewhere. The usual mistake is `h >= 0`, which lets a unit at exactly 0 pass gradient.
5. **`branch_gradient(dY, cache)`** returns only the branch's share, `dh1 @ W1`, shaped `(n, d)`. The usual mistake is to include `dY` here.
6. **`residual_block_backward(dY, cache)`** returns `(dx, grads)` with keys `W1`, `b1`, `W2`, `b2`, each shaped like its parameter. Use `.T @`, `.sum(axis=0)` for the biases, and your two earlier functions. The usual mistake is to return only the branch term and forget to add `dY`.

## Where this breaks

- You can add $x$ and $F(x)$ only when they have the same shape. A stage that changes the width or the image size must put a learned map on the shortcut, and the chapter "Doing the wide work at a narrow width: bottleneck cost" shows one.
- Real blocks put normalisation inside the branch, which this chapter leaves out. The chapter "Stacking it up: residuals, layer norm, and the pre-norm block" builds that version.

## The exercise

You build `relu`, `residual_block_forward`, `identity_gap`, `relu_mask`, `branch_gradient`, and `residual_block_backward`. The checks match the forward pass and every gradient to the worked numbers above, check that a zero branch returns the input, and compare all gradients with finite differences. Running the file pushes one input through twenty blocks, then trains a stack of twenty residual blocks and prints the loss falling.
