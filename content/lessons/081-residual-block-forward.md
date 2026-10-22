---
title: Why deeper networks got worse, and the residual block
slug: 081-residual-block-forward
part: "Part 5 — Vision and sequences"
summary: A residual block adds a small correction to its input, so passing the input through unchanged is the default and not a setting to find.
minutes: 40
gpu: false
objectives:
  - Explain why a 56-layer plain network can train worse than a 20-layer one even though it could copy the shallow network.
  - Describe a residual block as the input plus a correction, and say what the shortcut and the branch are.
  - Compute a residual block's forward pass by hand on two examples, and show that a zero branch returns the input exactly.
  - Build the forward pass in numpy and measure how far a block moves its input.
prereqs:
  - 058-forward-pass-and-gelu
  - 069-signal-size-through-layers
  - 080-convolution-backward
exercise: 081-residual-block-forward
---

# Why deeper networks got worse, and the residual block

You train a network with 20 layers and record its error on the training set. Then you stack 36 more layers on top, keep the data and the optimiser the same, and train again. The 56-layer network ends with a higher training error than the 20-layer one. This result, measured on the CIFAR-10 image set, led to residual networks.

It is not overfitting. Overfitting means a low error on the training set and a high error on unseen data, and here the error on the training set itself is worse. It is also strange, because the deep network can do everything the shallow one can. It could copy the 20 trained layers and let the 36 extra layers pass their input along unchanged, and that would match the shallow network exactly. So a solution at least as good as the shallow one exists, and gradient descent fails to find it.

This chapter explains why "pass the input along unchanged" is a hard setting for an ordinary layer, and how one change makes it the easy default: add the block's input back onto its output. You build the forward pass of that block, and the next chapter sends gradients back through it.

## What you already know

Five ideas carry this chapter:

- A linear layer turns each row of `x` into `x @ W.T + b` (see "Run a network forward, with a sharp or smooth bend").
- ReLU replaces every negative number with zero, and leaves the others alone.
- A plain layer with small random weights shrinks its signal toward zero (see "Why a deep network's signal shrinks or explodes").
- Training starts from small random weights and then moves them with gradient descent.
- A square weight matrix, $d \times d$, gives an output as wide as its input.

## Why an ordinary layer struggles to do nothing

Picture a chain of 36 people, each handed a page and asked to hand it on. In a plain network each person redraws the page from memory. If they redraw it perfectly, the page arrives unchanged. Any small slip is passed on and added to by the next person, so "do nothing" needs every person to be exact.

A plain block is that person. Its output comes only from its weights, so to hand the input along it needs one precise setting of every weight. That raises a question: why not start the extra layers at that setting, an identity? There are three reasons it does not rescue the plain block.

First, a block with a ReLU in the middle has no single identity matrix. A ReLU wipes out negative entries, so copying negative inputs needs biases tuned to keep them positive inside the block. Second, even at the right setting, training moves every weight in whatever direction fits the data, and nothing pulls the block back. Third, the usual starting point is small random weights, and a plain block with small weights does not copy anything. It outputs numbers close to zero, which is the opposite of the input.

Change the picture. Hand the page along untouched, and let each person attach a sticky note with a correction. The next person receives the page plus the notes so far. If nobody writes a note, the page arrives as it started. Writing no note is the easiest thing to do, because zero is where small weights already sit.

Figure 1 draws both versions. The page is $x$, the sticky note is $F(x)$, and the result is $y$.

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

Read it aloud: the block's output $y$ is its input $x$ plus whatever $F$ computes from $x$. $F$ is the **residual branch**, the layers that write the sticky note. The arc that carries $x$ around them to the plus sign is the **shortcut**. A residual, in general, is what is left over, and here $F$ computes what is left to add to $x$.

## Work one block by hand

Take two examples with two features each, one example per row:

$$
x = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}
$$

This pair is the running example for this chapter and the next. The branch has two linear layers with a ReLU between them. The first layer has $W_1$ equal to the 2 by 2 identity and $b_1 = [0, -1]$. The second has $W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ and $b_2 = [0.1, 0]$. The steps are as follows:

1. The first layer gives $h_1 = x W_1^{\top} + b_1 = \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$, because the identity leaves $x$ alone and $b_1$ subtracts 1 from the second column.
2. The ReLU turns the $-1$ into 0, so $a_1 = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$.
3. The second layer gives $h_2 = a_1 W_2^{\top} + b_2 = \begin{bmatrix} 2.1 & 0.5 \\ 0.1 & 1.0 \end{bmatrix}$. This is $F(x)$, the correction.
4. The addition gives $y = x + h_2 = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.

The top row of Figure 2 shows step 4 as grids. Notice that example 2 keeps its negative entry in the first column, $-0.9$, because the block adds to $x$ and applies no ReLU after the plus sign. A ReLU there would clip negative entries of $x + F(x)$, so even a zero correction would no longer return $x$.

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

Zero the last layer, with $W_2 = 0$ and $b_2 = 0$. Then $h_2$ is all zeros whatever $a_1$ holds, and $y = x$ exactly, as the bottom row of Figure 2 shows. A plain block with the same zeros returns $F(x) = 0$ and loses the input. With the weights small instead of zero, the difference is as sharp. Take five random inputs of width 8 and weights drawn with standard deviation 0.05 (the last check in the exercise uses exactly this setup, with seed 3). The residual block moves no entry by more than 0.06. The plain block's output is near zero, so it differs from the input by 2.46, which is the size of the largest input entry.

In numpy, each example is a row, so the weights are transposed:

```python
h1 = x @ W1.T + b1
a1 = np.maximum(h1, 0.0)
h2 = a1 @ W2.T + b2
y = x + h2
```

## Why the deep network has a way out

Return to the 56-layer network. Replace each layer with a residual block whose last layer starts at zero. Every one of the 36 extra blocks starts as the identity, so the deep network begins as a copy of the shallow computation, and training only has to find corrections that help. Figure 3 shows the failure that residual blocks are meant to remove. The curves are schematic, and drawn to show the shape of the reported result, not as a measurement.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Schematic training error against training time: a 56-layer plain network stays above a 20-layer plain network">
<line x1="70" y1="260" x2="520" y2="260" class="stroke-muted"/><line x1="70" y1="260" x2="70" y2="40" class="stroke-muted"/>
<line x1="70" y1="260.0" x2="520" y2="260.0" class="stroke-grid"/><text x="62" y="264.0" text-anchor="end" class="fig-small">0.0</text>
<line x1="70" y1="216.0" x2="520" y2="216.0" class="stroke-grid"/><text x="62" y="220.0" text-anchor="end" class="fig-small">0.1</text>
<line x1="70" y1="172.0" x2="520" y2="172.0" class="stroke-grid"/><text x="62" y="176.0" text-anchor="end" class="fig-small">0.2</text>
<line x1="70" y1="128.0" x2="520" y2="128.0" class="stroke-grid"/><text x="62" y="132.0" text-anchor="end" class="fig-small">0.3</text>
<line x1="70" y1="84.0" x2="520" y2="84.0" class="stroke-grid"/><text x="62" y="88.0" text-anchor="end" class="fig-small">0.4</text>
<line x1="70" y1="40.0" x2="520" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" text-anchor="end" class="fig-small">0.5</text>
<text x="70" y="26" class="fig-small">training error (schematic)</text>
<text x="295.0" y="294" text-anchor="middle" class="fig-small">training time (schematic)</text>
<polyline points="70.0,84.0 79.0,97.0 88.0,108.9 97.0,119.7 106.0,129.6 115.0,138.6 124.0,146.9 133.0,154.4 142.0,161.3 151.0,167.6 160.0,173.3 169.0,178.6 178.0,183.3 187.0,187.7 196.0,191.7 205.0,195.3 214.0,198.7 223.0,201.7 232.0,204.5 241.0,207.0 250.0,209.3 259.0,211.4 268.0,213.4 277.0,215.1 286.0,216.7 295.0,218.2 304.0,219.5 313.0,220.7 322.0,221.9 331.0,222.9 340.0,223.8 349.0,224.7 358.0,225.4 367.0,226.2 376.0,226.8 385.0,227.4 394.0,227.9 403.0,228.4 412.0,228.9 421.0,229.3 430.0,229.7 439.0,230.0 448.0,230.3 457.0,230.6 466.0,230.9 475.0,231.1 484.0,231.3 493.0,231.5 502.0,231.7 511.0,231.9 520.0,232.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="70.0,62.0 79.0,69.3 88.0,76.2 97.0,82.8 106.0,89.1 115.0,95.1 124.0,100.8 133.0,106.2 142.0,111.3 151.0,116.2 160.0,120.9 169.0,125.3 178.0,129.5 187.0,133.5 196.0,137.3 205.0,140.9 214.0,144.4 223.0,147.7 232.0,150.8 241.0,153.7 250.0,156.6 259.0,159.2 268.0,161.8 277.0,164.2 286.0,166.5 295.0,168.7 304.0,170.8 313.0,172.8 322.0,174.7 331.0,176.5 340.0,178.2 349.0,179.8 358.0,181.4 367.0,182.9 376.0,184.3 385.0,185.6 394.0,186.9 403.0,188.1 412.0,189.2 421.0,190.3 430.0,191.4 439.0,192.3 448.0,193.3 457.0,194.2 466.0,195.0 475.0,195.8 484.0,196.6 493.0,197.3 502.0,198.0 511.0,198.7 520.0,199.3" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="340" y="246" class="fig-bold">20 layers</text>
<text x="340" y="168" class="fig-bold">56 layers</text>
<line x1="502.0" y1="198.7" x2="502.0" y2="231.9" class="stroke-bad dashed"/>
<text x="520" y="102" text-anchor="end" class="fig-small">the deeper net could copy the shallow one,</text>
<text x="520" y="120" text-anchor="end" class="fig-small">but training does not find that copy</text>
</svg>
Figure 3. Schematic training curves, drawn to show the shape of the reported result and not as measured values. The 56-layer plain network ends above the 20-layer one, although it could have copied it. The gap is the failure to find a setting that exists.
```

Be careful about what this shows. A block that starts as the identity is a good starting point, not a guarantee, because training moves the correction away from zero and a large correction can still hurt. The next chapters show a second benefit, a path for the gradient.

> [!KEY]
> The deep network failed because "do nothing" is a precise setting for a plain block, and small random weights miss it. A residual block turns "do nothing" into a zero correction, which small weights already land near, so extra blocks start out harmless and training only has to find what to add.

> [!RECAP]
> Hold these points before you write code:
>
> - A plain block can pass its input along only at a precise weight setting, and small random weights give the opposite, a near-zero output.
> - A residual block computes $y = x + F(x)$, so a zero correction returns the input exactly.
> - The forward lines are `h1 = x @ W1.T + b1`, a ReLU, `h2 = a1 @ W2.T + b2`, and `y = x + h2`, with no ReLU after the sum.
> - On the running example, $y = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.
> - The plain block's output is $F(x)$, which is `y - x`.

## Build it step by step

The checks call the functions in this order:

1. **`relu(z)`.** `z` is an array of any shape. Return an array of the same shape with every negative entry replaced by 0. Use `np.maximum(z, 0.0)`. The usual mistake is to change `z` in place, which silently changes the caller's data.
2. **`residual_block_forward(x, params)`.** `x` is `(n, d)`. `params` holds `W1`, `b1`, `W2`, `b2`, shaped `(d, d)`, `(d,)`, `(d, d)`, `(d,)`. Return `(y, cache)` with `y` shaped `(n, d)`. Call your `relu` for the middle step. Put `x`, `W1`, `W2`, `h1`, and `a1` into `cache` as a tuple in that order, because the next chapter's backward pass reads them. The usual mistake is `W1 @ x`, or a ReLU after the sum. With `W2` and `b2` at zero, `y` must equal `x`.
3. **`identity_gap(x, params, residual)`.** Return the largest absolute difference between the block's output and `x`, as a `float`. When `residual` is true, the output is the residual block's `y`. When it is false, the output is a plain block's `F(x)`, which you get from the same call as `y - x`. The usual mistake is to return the plain block's gap as zero because you reused `y`. With `W2` and `b2` at zero, the residual gap is exactly 0 and the plain gap is the largest $|x|$.

## Where this breaks

A residual block has three limits you can already see:

- You can add $x$ and $F(x)$ only when they have the same shape. In the block here the weights are square, so the shapes match. A stage that changes the width or the image size must put a learned map on the shortcut, and the chapter "Doing the wide work at a narrow width: bottleneck cost" shows one.
- Zero is the starting point, not the destination. After a few updates the correction is not zero.
- Real blocks put normalisation inside the branch, which this chapter leaves out. The chapter "Stacking it up: residuals, layer norm, and the pre-norm block" builds that version.

## The exercise

You build `relu`, `residual_block_forward`, and `identity_gap`. The checks match `relu` on hand-picked values, and match the forward pass to the worked numbers above. A zero branch must return the input, the cache must keep the inputs unchanged, and `identity_gap` must separate the two block types: exactly 0 versus the largest $|x|$ with zeros, and under 0.1 versus over 1 with small random weights. Running the file trains nothing yet. It pushes one input through twenty blocks with small random weights, so you can watch the residual stack keep the input's size while the plain stack drains it.
