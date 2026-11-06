---
title: The backward rule for each array operation
slug: 055-backprop-array-operations
part: "Part 4 — Neural networks"
summary: Each array operation has a small backward rule that follows from asking which inputs fed each output entry, and the shapes check your work.
minutes: 40
gpu: false
objectives:
  - Explain why a bias copied onto many rows collects the sum of the slopes of its copies.
  - Derive the matrix-product backward rule from one output entry, and say why the transposes appear.
  - Explain how ReLU gates and sums pass slope through.
  - Use the rule that a gradient has the shape of its value to catch a wrong rule.
prereqs:
  - 053-width-depth-and-cost
  - 054-backprop-the-graph-and-reverse-order
exercise: 055-backprop-array-operations
---

# The backward rule for each array operation

A real layer does not multiply two numbers. It multiplies a batch of rows by a weight matrix, adds one bias to every row, and bends the result with a ReLU. Take a batch of two examples with three features each, and a layer with two outputs. The weight matrix is 3 by 2, and the loss sits somewhere downstream.

The chapter *Walking slopes backwards through a calculation* gave you rules for single numbers: a sum passes the slope through, a product hands each input the other input, and a value used twice collects both slopes. It said nothing about what a slope looks like for an array. Without an answer, you end up guessing: put a transpose here, try the other factor there, and keep whichever version does not crash. That works until two dimensions happen to match, and then a wrong rule runs silently and trains badly.

This chapter shows that every array rule falls out of the same question you already asked about single numbers: which inputs fed this output, and with what weight? The question to answer: what does a slope look like for an array, and how do you get the shapes right without guessing?

## What you already know

- A slope says how much the loss moves for a small nudge of one input (see *Walking slopes backwards through a calculation*).
- A value used in two places collects the sum of the slopes from both routes.
- The backward walk adds each contribution onto a stored slope with `+=`.
- A matrix product `A @ B` multiplies shapes `(n, k)` and `(k, m)` to give `(n, m)` (see *Run a network forward, with a sharp or smooth bend*).
- Numpy broadcasting stretches a bias of shape `(m,)` across every row of an `(n, m)` matrix.

## A slope for an array has the shape of the array

The loss is one number, but the value you are asking about is an array. The slope of the loss with respect to an array has one entry per entry of the array: entry $(i, j)$ says how much the loss moves per unit nudge of that one entry. So the slope of a $3 \times 2$ weight matrix is itself a $3 \times 2$ array, and this rule holds for every operation in the chapter.

> [!KEY]
> A gradient has exactly the shape of its value. If a backward rule hands back the wrong shape, the rule is wrong.

Write $\bar{Z}$ for the slope of the loss with respect to $Z$, as in the previous chapter. For the running example, suppose the slope arriving at the output of the bias add is $\bar{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, the same shape as the output.

## A bias is a value used on every row

Adding a bias $b = [10, -30]$, of shape $(2,)$, to a $2 \times 2$ matrix looks like one operation, but numpy copies the bias onto each row first. The bias is one value used twice, exactly like $a$ in the previous chapter, and a value used twice collects the sum of its slopes. In Figure 1, each copy of the bias receives the slope of its own row, $[1, 2]$ (row 0) and $[3, 4]$ (row 1), and the bias gets their sum, $[1 + 3,\ 2 + 4] = [4, 6]$. Shape $(2,)$, the shape of the bias.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A bias of two numbers copied onto two rows going forward, and the two row slopes adding back into one bias slope going backward.">
<defs><marker id="c062-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker><marker id="c062-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="10.0" y="22.0" text-anchor="start" class="fig-bold">Forward: one bias, copied onto every row</text>
<rect x="40.0" y="86.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="60.0" y="104.0" text-anchor="middle" class="fig-small">10</text>
<rect x="80.0" y="86.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="100.0" y="104.0" text-anchor="middle" class="fig-small">-30</text>
<text x="80.0" y="134.0" text-anchor="middle" class="fig-small">bias b</text>
<rect x="330.0" y="50.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="350.0" y="68.0" text-anchor="middle" class="fig-small">10</text>
<rect x="370.0" y="50.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="390.0" y="68.0" text-anchor="middle" class="fig-small">-30</text>
<text x="430.0" y="68.0" text-anchor="start" class="fig-small">copy for row 0</text>
<path d="M124,100 C210,100 240,64 324,64" class="stroke-muted dashed" marker-end="url(#c062-arrow-muted)"/>
<rect x="330.0" y="122.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="350.0" y="140.0" text-anchor="middle" class="fig-small">10</text>
<rect x="370.0" y="122.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="390.0" y="140.0" text-anchor="middle" class="fig-small">-30</text>
<text x="430.0" y="140.0" text-anchor="start" class="fig-small">copy for row 1</text>
<path d="M124,100 C210,100 240,136 324,136" class="stroke-muted dashed" marker-end="url(#c062-arrow-muted)"/>
<text x="10.0" y="200.0" text-anchor="start" class="fig-bold">Backward: each copy sends its slope back, and the slopes add</text>
<rect x="330.0" y="222.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="350.0" y="240.0" text-anchor="middle" class="fig-small">1</text>
<rect x="370.0" y="222.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="390.0" y="240.0" text-anchor="middle" class="fig-small">2</text>
<text x="430.0" y="240.0" text-anchor="start" class="fig-small fill-b">slope at row 0</text>
<path d="M324,236 C240,236 210,246 126,246" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<rect x="330.0" y="262.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="350.0" y="280.0" text-anchor="middle" class="fig-small">3</text>
<rect x="370.0" y="262.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="390.0" y="280.0" text-anchor="middle" class="fig-small">4</text>
<text x="430.0" y="280.0" text-anchor="start" class="fig-small fill-b">slope at row 1</text>
<path d="M324,276 C240,276 210,246 126,246" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<rect x="40.0" y="232.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="60.0" y="250.0" text-anchor="middle" class="fig-small">4</text>
<rect x="80.0" y="232.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="100.0" y="250.0" text-anchor="middle" class="fig-small">6</text>
<text x="80.0" y="280.0" text-anchor="middle" class="fig-small fill-b">slope at b = [1+3, 2+4]</text>
</svg>
Figure 1. A bias used on two rows is one value used twice. Going forward it is copied; going backward each copy sends back its own slope, and the bias receives their sum, one number per column.
```

The helper that undoes the copying is `unbroadcast(grad, shape)`. It sums over the axes numpy added or stretched, until the slope has the shape of the original value:

```python
while grad.ndim > len(shape):
    grad = grad.sum(axis=0)
for axis, size in enumerate(shape):
    if size == 1 and grad.shape[axis] != 1:
        grad = grad.sum(axis=axis, keepdims=True)
```

Every `+=` in an add or a multiply that might broadcast goes through `unbroadcast` first. Without it, the bias would receive a $2 \times 2$ slope on a value of shape $(2,)$, and the shape rule would catch it.

## A matrix product: ask who fed each output

Next, the transposes. A product $Z = A B$ has $Z_{ij} = \sum_k A_{ik} B_{kj}$, which reads: output entry $(i, j)$ is row $i$ of $A$ dotted with column $j$ of $B$.

Take one input entry, $A_{ik}$. Which outputs did it help build? Every entry in row $i$ of $Z$, one for each column $j$, and in $Z_{ij}$ it is multiplied by $B_{kj}$. Nudge $A_{ik}$ by $h$ and $Z_{ij}$ moves by $B_{kj} h$. Each $Z_{ij}$ then moves the loss by $\bar{Z}_{ij}$ per unit, so by the sum rule from the previous chapter,

$$
\bar{A}_{ik} = \sum_j \bar{Z}_{ij}\, B_{kj} .
$$

Read it aloud: the slope at $A_{ik}$ is row $i$ of $\bar{Z}$ dotted with row $k$ of $B$. A row of $\bar{Z}$ dotted with a *row* of $B$ is the same as a row of $\bar{Z}$ times a *column* of $B^\top$. That is why the transpose appears: it turns "row of $B$" into the column a product expects. The same argument for $B_{kj}$, which fed every row $i$ of $Z$ with weight $A_{ik}$, gives $\bar{B}_{kj} = \sum_i A_{ik}\, \bar{Z}_{ij}$. Together:

$$
\bar{A} \mathrel{+}= \bar{Z}\, B^\top, \qquad \bar{B} \mathrel{+}= A^\top \bar{Z} .
$$

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Matrix product grids: A times B gives Z, then Z-slope times B transposed gives A-slope, with one entry highlighted in each row.">

<text x="10.0" y="22.0" text-anchor="start" class="fig-bold">Forward: Z = A B</text>
<rect x="20.0" y="50.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="37.0" y="67.0" text-anchor="middle" class="fig-small">1</text>
<rect x="54.0" y="50.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="71.0" y="67.0" text-anchor="middle" class="fig-small">2</text>
<rect x="88.0" y="50.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="105.0" y="67.0" text-anchor="middle" class="fig-small">3</text>
<rect x="20.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="37.0" y="93.0" text-anchor="middle" class="fig-small">4</text>
<rect x="54.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="71.0" y="93.0" text-anchor="middle" class="fig-small">5</text>
<rect x="88.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="105.0" y="93.0" text-anchor="middle" class="fig-small">6</text>
<text x="71.0" y="42.0" text-anchor="middle" class="fig-small">A (2×3)</text>
<text x="142.0" y="76.0" text-anchor="middle">×</text>
<rect x="162.0" y="50.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="179.0" y="67.0" text-anchor="middle" class="fig-small">1</text>
<rect x="196.0" y="50.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="213.0" y="67.0" text-anchor="middle" class="fig-small">0</text>
<rect x="162.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="179.0" y="93.0" text-anchor="middle" class="fig-small">2</text>
<rect x="196.0" y="76.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="213.0" y="93.0" text-anchor="middle" class="fig-small">1</text>
<rect x="162.0" y="102.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="179.0" y="119.0" text-anchor="middle" class="fig-small">0</text>
<rect x="196.0" y="102.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="213.0" y="119.0" text-anchor="middle" class="fig-small">3</text>
<text x="196.0" y="42.0" text-anchor="middle" class="fig-small">B (3×2)</text>
<text x="250.0" y="76.0" text-anchor="middle">=</text>
<rect x="270.0" y="50.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="67.0" text-anchor="middle" class="fig-small">5</text>
<rect x="304.0" y="50.0" width="34" height="26" class="fill-soft-b stroke-muted"/>
<text x="321.0" y="67.0" text-anchor="middle" class="fig-small">11</text>
<rect x="270.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="93.0" text-anchor="middle" class="fig-small">14</text>
<rect x="304.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="321.0" y="93.0" text-anchor="middle" class="fig-small">23</text>
<text x="304.0" y="42.0" text-anchor="middle" class="fig-small">Z (2×2)</text>
<text x="358.0" y="58.0" text-anchor="start" class="fig-small">Z[0,1] = row 0 of A</text>
<text x="358.0" y="74.0" text-anchor="start" class="fig-small">· column 1 of B</text>
<text x="358.0" y="90.0" text-anchor="start" class="fig-small fill-a">= 0 + 2 + 9 = 11</text>
<text x="10.0" y="170.0" text-anchor="start" class="fig-bold">Backward: slope of A = (slope of Z) × Bᵀ</text>
<rect x="20.0" y="200.0" width="34" height="26" class="fill-soft-b stroke-muted"/>
<text x="37.0" y="217.0" text-anchor="middle" class="fig-small">1</text>
<rect x="54.0" y="200.0" width="34" height="26" class="fill-soft-b stroke-muted"/>
<text x="71.0" y="217.0" text-anchor="middle" class="fig-small">2</text>
<rect x="20.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="37.0" y="243.0" text-anchor="middle" class="fig-small">3</text>
<rect x="54.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="71.0" y="243.0" text-anchor="middle" class="fig-small">4</text>
<text x="54.0" y="192.0" text-anchor="middle" class="fig-small">Z slope (2×2)</text>
<text x="108.0" y="219.0" text-anchor="middle">×</text>
<rect x="128.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="145.0" y="217.0" text-anchor="middle" class="fig-small">1</text>
<rect x="162.0" y="200.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="179.0" y="217.0" text-anchor="middle" class="fig-small">2</text>
<rect x="196.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="213.0" y="217.0" text-anchor="middle" class="fig-small">0</text>
<rect x="128.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="145.0" y="243.0" text-anchor="middle" class="fig-small">0</text>
<rect x="162.0" y="226.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="179.0" y="243.0" text-anchor="middle" class="fig-small">1</text>
<rect x="196.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="213.0" y="243.0" text-anchor="middle" class="fig-small">3</text>
<text x="179.0" y="192.0" text-anchor="middle" class="fig-small">Bᵀ (2×3)</text>
<text x="250.0" y="219.0" text-anchor="middle">=</text>
<rect x="270.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="217.0" text-anchor="middle" class="fig-small">1</text>
<rect x="304.0" y="200.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="321.0" y="217.0" text-anchor="middle" class="fig-small">4</text>
<rect x="338.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="355.0" y="217.0" text-anchor="middle" class="fig-small">6</text>
<rect x="270.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="243.0" text-anchor="middle" class="fig-small">3</text>
<rect x="304.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="321.0" y="243.0" text-anchor="middle" class="fig-small">10</text>
<rect x="338.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="355.0" y="243.0" text-anchor="middle" class="fig-small">12</text>
<text x="321.0" y="192.0" text-anchor="middle" class="fig-small">A slope (2×3)</text>
<text x="386.0" y="208.0" text-anchor="start" class="fig-small">A slope[0,1] = row 0 of Z slope</text>
<text x="386.0" y="224.0" text-anchor="start" class="fig-small fill-b">· row 1 of B = 1·2 + 2·1 = 4</text>
<text x="300.0" y="300.0" text-anchor="middle" class="fig-small fill-muted">Each entry of the A slope is a row of the Z slope dotted with a row of B.</text>
</svg>
Figure 2. A product of a 2×3 matrix and a 3×2 matrix. Forward, the highlighted output entry uses one row of A and one column of B. Backward, the highlighted slope for A uses the same row of B, dotted with a row of the output slope, which is why Bᵀ appears.
```

Check it on numbers, with $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$, $B = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 3 \end{bmatrix}$, and $\bar{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$. Then $Z = \begin{bmatrix} 5 & 11 \\ 14 & 23 \end{bmatrix}$. Entry $\bar{A}_{01}$ is row 0 of $\bar{Z}$, $[1, 2]$, dotted with row 1 of $B$, $[2, 1]$, giving $2 + 2 = 4$. The full answer is $\bar{A} = \begin{bmatrix} 1 & 4 & 6 \\ 3 & 10 & 12 \end{bmatrix}$ and $\bar{B} = \begin{bmatrix} 13 & 18 \\ 17 & 24 \\ 21 & 30 \end{bmatrix}$. Their shapes, $2 \times 3$ and $3 \times 2$, are the shapes of $A$ and $B$. In numpy:

```python
self.grad += out.grad @ other.value.T
other.grad += self.value.T @ out.grad
```

The shape rule pays off here. Suppose you wrote `out.grad @ other.value` without the transpose: a $2 \times 2$ times a $3 \times 2$ does not multiply, and numpy raises an error. The shapes force the transposes, but only when the matrices are not square, so test with a non-square product. The engineers' name for this rule is a **vector-Jacobian product**. The full table of local slopes, one entry for every pair of an output entry and an input entry (a **Jacobian**), would be enormous. You never build it. You only multiply the slope arriving at the operation by it, and the two formulas above are that product written out.

## ReLU and sum: gates and spreading

ReLU computes $\max(x, 0)$ one entry at a time. A positive input passes through unchanged, so its local slope is 1. A negative input is replaced by 0 whatever you do to it nearby, so its local slope is 0. In Figure 3, take $x = [-1, 2, 0.5, -3]$ with incoming slope $[5, 7, 1, 2]$. The slope on entry 0 meets a closed gate and stops, so the outgoing slope is $[0, 7, 1, 0]$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A ReLU with four gates that pass or block slopes, and a sum that hands one slope to every entry.">
<defs><marker id="c062-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="10.0" y="22.0" text-anchor="start" class="fig-bold">ReLU: open gate passes the slope, closed gate stops it</text>
<text x="60.0" y="48.0" text-anchor="middle" class="fig-small fill-b">slope out</text>
<text x="190.0" y="48.0" text-anchor="middle" class="fig-small fill-a">input x</text>
<text x="330.0" y="48.0" text-anchor="middle" class="fig-small fill-b">slope in</text>
<rect x="145.0" y="62.0" width="90.0" height="36.0" rx="6" class="fill-sunk stroke-bad"/>
<text x="190.0" y="78.0" text-anchor="middle" class="fig-bold fill-a">x = -1</text>
<text x="190.0" y="92.0" text-anchor="middle" class="fig-small fill-bad">closed</text>
<text x="330.0" y="86.0" text-anchor="middle" class="fig-bold fill-b">5</text>
<line x1="312.0" y1="80.0" x2="240.0" y2="80.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<line x1="142.0" y1="80.0" x2="120.0" y2="80.0" class="stroke-bad"/>
<line x1="120.0" y1="70.0" x2="120.0" y2="90.0" class="stroke-bad"/>
<text x="60.0" y="86.0" text-anchor="middle" class="fig-bold fill-b">0</text>
<rect x="145.0" y="114.0" width="90.0" height="36.0" rx="6" class="fill-soft-c stroke-c"/>
<text x="190.0" y="130.0" text-anchor="middle" class="fig-bold fill-a">x = 2</text>
<text x="190.0" y="144.0" text-anchor="middle" class="fig-small fill-c">open</text>
<text x="330.0" y="138.0" text-anchor="middle" class="fig-bold fill-b">7</text>
<line x1="312.0" y1="132.0" x2="240.0" y2="132.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<line x1="142.0" y1="132.0" x2="90.0" y2="132.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<text x="60.0" y="138.0" text-anchor="middle" class="fig-bold fill-b">7</text>
<rect x="145.0" y="166.0" width="90.0" height="36.0" rx="6" class="fill-soft-c stroke-c"/>
<text x="190.0" y="182.0" text-anchor="middle" class="fig-bold fill-a">x = 0.5</text>
<text x="190.0" y="196.0" text-anchor="middle" class="fig-small fill-c">open</text>
<text x="330.0" y="190.0" text-anchor="middle" class="fig-bold fill-b">1</text>
<line x1="312.0" y1="184.0" x2="240.0" y2="184.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<line x1="142.0" y1="184.0" x2="90.0" y2="184.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<text x="60.0" y="190.0" text-anchor="middle" class="fig-bold fill-b">1</text>
<rect x="145.0" y="218.0" width="90.0" height="36.0" rx="6" class="fill-sunk stroke-bad"/>
<text x="190.0" y="234.0" text-anchor="middle" class="fig-bold fill-a">x = -3</text>
<text x="190.0" y="248.0" text-anchor="middle" class="fig-small fill-bad">closed</text>
<text x="330.0" y="242.0" text-anchor="middle" class="fig-bold fill-b">2</text>
<line x1="312.0" y1="236.0" x2="240.0" y2="236.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<line x1="142.0" y1="236.0" x2="120.0" y2="236.0" class="stroke-bad"/>
<line x1="120.0" y1="226.0" x2="120.0" y2="246.0" class="stroke-bad"/>
<text x="60.0" y="242.0" text-anchor="middle" class="fig-bold fill-b">0</text>
<text x="440.0" y="22.0" text-anchor="start" class="fig-bold">Sum: same slope each</text>
<rect x="440.0" y="100.0" width="70.0" height="40.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="475.0" y="118.0" text-anchor="middle" class="fig-bold fill-a">sum = 6</text>
<text x="475.0" y="134.0" text-anchor="middle" class="fig-small fill-b">slope in: 4</text>
<rect x="540.0" y="62.0" width="50.0" height="36.0" rx="6" class="fill-soft-a stroke-muted"/>
<text x="565.0" y="76.0" text-anchor="middle" class="fig-small fill-a">x = 1</text>
<text x="565.0" y="91.0" text-anchor="middle" class="fig-small fill-b">gets 4</text>
<line x1="512.0" y1="120.0" x2="538.0" y2="80.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<rect x="540.0" y="122.0" width="50.0" height="36.0" rx="6" class="fill-soft-a stroke-muted"/>
<text x="565.0" y="136.0" text-anchor="middle" class="fig-small fill-a">x = 2</text>
<text x="565.0" y="151.0" text-anchor="middle" class="fig-small fill-b">gets 4</text>
<line x1="512.0" y1="120.0" x2="538.0" y2="140.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
<rect x="540.0" y="182.0" width="50.0" height="36.0" rx="6" class="fill-soft-a stroke-muted"/>
<text x="565.0" y="196.0" text-anchor="middle" class="fig-small fill-a">x = 3</text>
<text x="565.0" y="211.0" text-anchor="middle" class="fig-small fill-b">gets 4</text>
<line x1="512.0" y1="120.0" x2="538.0" y2="200.0" class="stroke-b dashed" marker-end="url(#c062-arrow-b)"/>
</svg>
Figure 3. ReLU is a row of gates. A positive input lets the slope through unchanged, and a zero or negative input stops it, so the slope out is 0. A sum has no gates: every entry it added up gets the whole incoming slope.
```

A sum, $z = \sum_i x_i$, has no gates. Nudge any single entry by $h$ and the sum moves by $h$, so every local slope is 1, and every entry receives the whole incoming slope. For $x = [1, 2, 3]$ with the sum 6 and incoming slope 4, each entry receives 4.

$$
z = \operatorname{relu}(x):\quad \bar{x} \mathrel{+}= \bar{z} \odot [x > 0]
\qquad\qquad
z = \textstyle\sum_i x_i:\quad \bar{x} \mathrel{+}= \bar{z} \cdot \mathbf{1} .
$$

The symbol $\odot$ is entry-by-entry multiplication, and $[x > 0]$ is 1 where the test holds and 0 elsewhere. In numpy:

```python
self.grad += out.grad * (self.value > 0)
self.grad += out.grad * np.ones_like(self.value)
```

## What each rule must remember

Look at what each backward rule used. The matrix product needed both inputs, because each slope is multiplied by the other factor. The ReLU needed to know which inputs were positive. The sum and the add needed nothing from the forward pass. So the forward pass must keep its values until the backward walk uses them, which is the reason memory grows with depth. For ReLU you may keep either the input or the output, because $\operatorname{relu}(x) > 0$ exactly when $x > 0$; the mask is identical, including 0 at $x = 0$. At exactly 0 the ReLU has a corner, so it has no single slope there. The engine picks 0, and real activations almost never land on exactly 0.

## Run a whole layer backwards

Put the pieces together on the running example: $A$ and $B$ as above, a bias $b = [10, -30]$, and the layer $\operatorname{relu}(AB + b)$. Forward, $AB + b = \begin{bmatrix} 15 & -19 \\ 24 & -7 \end{bmatrix}$, and the ReLU gives $\begin{bmatrix} 15 & 0 \\ 24 & 0 \end{bmatrix}$. Say the slope arriving at the ReLU output is $\bar{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ again. The second column sits behind closed gates, so the slope passing back is $\begin{bmatrix} 1 & 0 \\ 3 & 0 \end{bmatrix}$. The bias collects its column sums, $[4, 0]$. The product hands back $\bar{A} = \begin{bmatrix} 1 & 2 & 0 \\ 3 & 6 & 0 \end{bmatrix}$ and $\bar{B} = \begin{bmatrix} 13 & 0 \\ 17 & 0 \\ 21 & 0 \end{bmatrix}$. The second column of $B$ gets nothing, because every example blocked it. A weight whose unit is always off learns nothing from that batch.

> [!KEY]
> Every array rule comes from asking which inputs fed each output entry. A copied value sums the slopes of its copies, a product sums over the entries it touched, a gate passes or blocks, and the result always has the shape of the value.

> [!RECAP]
> - A gradient has the shape of its value, so a wrong shape means a wrong rule.
> - A bias or any broadcast value is one value used many times, so `unbroadcast` sums its copies' slopes.
> - For `Z = A @ B`, $\bar{A} = \bar{Z} B^\top$ and $\bar{B} = A^\top \bar{Z}$, from summing over the entries each input fed.
> - ReLU multiplies the slope by 1 where the input is positive and 0 elsewhere. A sum gives every entry the whole slope.
> - Backward needs the stored forward values: both inputs of a product, the sign of a ReLU input.

## Build it step by step

The engine from the previous chapter is provided, with `__add__` and `__mul__` already calling your `unbroadcast`, and with `Tensor.sum` and `Tensor.backward` provided. You write three functions.

1. **`unbroadcast(grad, shape)`** takes a slope array and the shape of the original value, and returns an array of exactly that shape. Sum leading axes away first, then sum with `keepdims=True` over each axis where `shape` has size 1. The common mistake is forgetting `keepdims=True`, which drops an axis and breaks a `(4, 1)` value.

    ```python
    def unbroadcast(grad, shape):
        # sum leading axes, then size-1 axes with keepdims
    ```

2. **`Tensor.__matmul__(other)`** builds `out` from `self.value @ other.value` with children `(self, other)`. Its backward adds `out.grad @ other.value.T` to `self.grad` and `self.value.T @ out.grad` to `other.grad`. The common mistake is putting the transpose on the wrong factor, which only fails loudly on non-square shapes.

    ```python
    def __matmul__(self, other):
        # two += lines; no unbroadcast, because the shapes already match
    ```

3. **`Tensor.relu()`** returns `np.maximum(self.value, 0)` with one child. Its backward adds `out.grad * (self.value > 0)`. The common mistake is `=` instead of `+=`, which breaks a tensor that feeds two places.

    ```python
    def relu(self):
        # one child; the mask is self.value > 0
    ```

## The exercise

You implement `unbroadcast`, `Tensor.__matmul__`, and `Tensor.relu`. The rest of the engine is provided.

The checks start with `unbroadcast` on the $[[1, 2], [3, 4]]$ example, which gives $[4, 6]$, and on larger shapes. They check a broadcast bias and a broadcast multiply against finite differences, then the $2 \times 3$ times $3 \times 2$ worked example with its exact slopes, then a non-square product against finite differences. They finish with the ReLU gates on $[-1, 2, 0.5, -3]$ and a whole layer, $\operatorname{relu}(XW + b)$, whose $W$ and $b$ slopes must match finite differences.
