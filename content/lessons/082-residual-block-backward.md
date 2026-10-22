---
title: "Gradients through a plus sign: the residual backward pass"
slug: 082-residual-block-backward
part: "Part 5 — Vision and sequences"
summary: A plus sign sends the incoming gradient down both paths, so the block's input gradient is that gradient plus the branch's term.
minutes: 40
gpu: false
objectives:
  - Explain why the gradient at a block's input is the incoming gradient plus a term from the branch, using a one-number example.
  - Compute every gradient of a residual block by hand on two examples, and read each formula aloud.
  - Explain why the backward pass multiplies by W and not by W.T, and why the ReLU mask acts as a switch.
  - Build the backward pass in numpy and verify it against finite differences.
prereqs:
  - 061-backprop-the-graph-and-reverse-order
  - 062-backprop-array-operations
  - 064-backprop-mlp-and-gradient-check
  - 081-residual-block-forward
exercise: 082-residual-block-backward
---

# Gradients through a plus sign: the residual backward pass

A residual block has produced its output, and the loss has looked at it. The loss reports, for every number in the output, how much it would like that number to change: the gradient `dY`, one entry per output number. Training needs three answers, taken from that single message. How should each weight change? How should each bias change? And what message goes to the layer below, the gradient at the block's input?

The block makes this harder than a plain layer, because its input reaches the output by two routes. It goes through the branch, and it also goes straight to the plus sign. If you only follow the branch, you lose the shortcut's contribution, and the gradient that reaches the layer below is wrong. This chapter shows where the shortcut's share comes from, why it is exactly `dY`, and how to compute all of it in numpy. You will check your answer against finite differences, which are the check that tells you the math is right.

## What you already know

Four ideas carry this chapter:

- The forward pass of a residual block is `y = x + F(x)`, with a two-layer branch (see "Why deeper networks got worse, and the residual block").
- The chain rule multiplies slopes along a path, and a number that reaches the loss by several paths collects the slope of each (see "Walking slopes backwards through a calculation").
- A weight gradient and a bias gradient come from the same upstream message (see "The backward rule for each array operation").
- A gradient check compares a computed gradient with a finite-difference estimate (see "Putting it together: gradients for a whole network").

## A plus sign splits the gradient in two

Start with one number. Let $x = 2$ and let the branch be $f(x) = 0.5 x^2$. The block computes $y = x + f(x) = 2 + 2 = 4$. Nudge $x$ from 2 to 2.001. Then $y$ becomes about 4.003, three times the nudge. The slope of $y$ is $1 + f'(x)$, which is $1 + x = 3$ here. One part of that 3 comes through the shortcut, and the other two come through the branch.

That is the whole idea, written as a formula:

$$
\frac{dy}{dx} = 1 + f'(x)
$$

Read it aloud: the output's slope with respect to the input is 1, from the shortcut, plus the branch's own slope. The 1 is there because the shortcut is $y = x$, whose slope is 1. A block with two paths from $x$ to $y$ has two terms, one per path. When the loss's message $dY$ arrives, each path multiplies it by that path's slope. The shortcut multiplies by 1, so it returns $dY$ unchanged. The branch multiplies by its own slopes, which take longer to compute. Figure 1 shows this on the running example.

```figure
<svg viewBox="0 0 620 360" role="img" aria-label="Gradient flowing backward through a residual block: it arrives at the plus sign, splits into a shortcut path that carries it unchanged and a branch path through W2, the ReLU mask, and W1, then adds back together at the input">
<defs><marker id="c082-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<circle cx="530" cy="150" r="15" class="fill-paper stroke-ink"/><text x="530" y="156" text-anchor="middle" class="fig-bold">+</text>
<circle cx="90" cy="150" r="15" class="fill-paper stroke-ink"/><text x="90" y="156" text-anchor="middle" class="fig-bold">+</text>
<text x="595" y="140" text-anchor="middle" class="fig-bold">dY</text>
<line x1="610" y1="150" x2="546" y2="150" class="stroke-d" marker-end="url(#c082-arrow)"/>
<path d="M530,134 L530,70 L90,70 L90,133" fill="none" class="stroke-d" stroke-width="2.5" marker-end="url(#c082-arrow)"/>
<text x="310" y="60" text-anchor="middle" class="fig-bold">shortcut: dY passes unchanged</text>
<text x="310" y="86" text-anchor="middle" class="fig-small fig-mono">[[1, 1], [1, 1]]</text>
<line x1="530" y1="166" x2="530" y2="230" class="stroke-d"/>
<line x1="530" y1="230" x2="516" y2="230" class="stroke-d" marker-end="url(#c082-arrow)"/>
<rect x="430.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="472.0" y="234.5" text-anchor="middle">× W2</text>
<line x1="430" y1="230" x2="396" y2="230" class="stroke-d" marker-end="url(#c082-arrow)"/>
<rect x="310.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="352.0" y="234.5" text-anchor="middle">× mask</text>
<line x1="310" y1="230" x2="276" y2="230" class="stroke-d" marker-end="url(#c082-arrow)"/>
<rect x="190.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="232.0" y="234.5" text-anchor="middle">× W1</text>
<path d="M190,230 L90,230 L90,167" fill="none" class="stroke-d" marker-end="url(#c082-arrow)"/>
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
Figure 1. The backward pass of the running example, with dY all ones. The gradient splits at the plus sign. The shortcut carries it to the input unchanged, and the branch carries it through W2, the mask, and W1 as a smaller term. The two meet at the input and add to dx.
```

Here is the same example as in the previous chapter, with $x = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}$, $W_1$ the identity, $b_1 = [0, -1]$, $W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$, and $b_2 = [0.1, 0]$. Its forward values were $h_1 = \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$ and $a_1 = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$. Suppose the loss sends back `dY` equal to all ones, which means "raise every output number a little". The backward pass then runs in the reverse of the forward order.

## Walk the branch backward

The plus sign gives the branch the whole message, so $dh_2 = dY$. The second layer's weight gradient follows from what that weight does. The entry $W_2[i, j]$ multiplies $a_1[n, j]$ on its way into $h_2[n, i]$ for every example $n$, so its gradient adds up $dh_2[n, i] \cdot a_1[n, j]$ over the examples. That sum is one entry of a matrix product, so $dW_2 = dh_2^{\top} a_1$. On the example it gives $\begin{bmatrix} 1 & 3 \\ 1 & 3 \end{bmatrix}$. The entry 3 is $1 \cdot 1 + 1 \cdot 2$, the second column of $a_1$ summed over both examples.

The bias is added to every row of the output, so its gradient adds the message over the rows: $db_2 = [2, 2]$.

To pass the message one layer down, notice that $a_1[n, j]$ feeds $h_2[n, i]$ through $W_2[i, j]$. So $da_1 = dh_2 W_2$, which is $\begin{bmatrix} 2 & 0.5 \\ 2 & 0.5 \end{bmatrix}$ here.

The ReLU is a switch. Its output is $h_1$ when $h_1 > 0$ and 0 otherwise, so its slope is 1 where the unit is on and 0 where it is off. An off unit passes no gradient, because nudging its input changes nothing at its output. The mask $[h_1 > 0]$ holds a 1 for each unit that is on:

$$
dh_1 = da_1 \odot [h_1 > 0]
$$

The symbol $\odot$ multiplies entry by entry. On the example the one negative $h_1$ is in example 2, feature 1, so the mask is $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and $dh_1 = \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix}$. Figure 2 shows the switch at work.

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
Figure 2. The ReLU mask as a switch, with the numbers from the running example. The one entry with a negative h1 is switched off, so the gradient that arrives there (2) is replaced by 0 in dh1. Every other entry passes through unchanged.
```

The first layer repeats the same two moves. Its weight gradient is $dW_1 = dh_1^{\top} x = \begin{bmatrix} 2 & 4 \\ 0 & 2.5 \end{bmatrix}$ and its bias gradient is $db_1 = [2, 1]$. Its message to the input is the branch term $dh_1 W_1$. Because $W_1$ is the identity, that term equals $dh_1$.

Last, the two paths meet at the input. The input's gradient is the shortcut's share plus the branch's share:

$$
dx = dY + dh_1 W_1
$$

On the example, $dx = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} + \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix} = \begin{bmatrix} 3 & 1.5 \\ 1 & 1.5 \end{bmatrix}$. The ones from the shortcut are sitting inside every entry of $dx$, and no switch can turn them off.

Here are those steps as numpy:

```python
dW2 = dY.T @ a1
da1 = dY @ W2
dh1 = da1 * (h1 > 0)
dx = dY + dh1 @ W1
```

> [!KEY]
> Because a plus sign sends the incoming gradient down both paths, the input's gradient is `dY` from the shortcut plus the branch's term. The shortcut's share never goes through a ReLU switch.

## Why backward multiplies by W, not W.T

The forward pass computes `x @ W.T`. The backward pass computes `dY @ W`, with no transpose, and the reason is shapes. A weight `W` of shape `(d_out, d_in)` carries `d_in` numbers to `d_out` numbers, so forward needs `W.T` to line up with rows of length `d_in`. The message coming back has `d_out` numbers per row and must leave with `d_in`, which `W` does. Figure 3 uses a 3 by 2 weight, where the wrong choice fails loudly.

```figure
<svg viewBox="0 0 620 400" role="img" aria-label="Shape bookkeeping for a layer with a three-by-two weight: forward multiplies by W transposed, backward multiplies by W, and multiplying by W transposed on the way back fails because the inner sizes 3 and 2 do not match">
<text x="20" y="22" class="fig-bold">Forward: out = x @ W.T</text>
<rect x="40" y="34" width="48" height="72" class="fill-soft-a stroke-a"/><text x="64.0" y="74.0" text-anchor="middle" class="fig-small">x  (n, 2)</text>
<text x="102" y="75.0" text-anchor="middle" class="fig-bold">@</text>
<rect x="116" y="34" width="72" height="48" class="fill-soft-b stroke-b"/><text x="152.0" y="62.0" text-anchor="middle" class="fig-small">W.T (2, 3)</text>
<text x="202" y="75.0" text-anchor="middle" class="fig-bold">=</text>
<rect x="216" y="34" width="72" height="72" class="fill-soft-c stroke-c"/><text x="252.0" y="74.0" text-anchor="middle" class="fig-small">out (n, 3)</text>
<text x="300" y="70.0" class="fig-small">each of n examples goes from 2 numbers</text>
<text x="300" y="88.0" class="fig-small">to 3 numbers</text>
<text x="20" y="142" class="fig-bold">Backward: dx = dY @ W</text>
<rect x="40" y="154" width="72" height="72" class="fill-soft-a stroke-a"/><text x="76.0" y="194.0" text-anchor="middle" class="fig-small">dY (n, 3)</text>
<text x="126" y="195.0" text-anchor="middle" class="fig-bold">@</text>
<rect x="140" y="154" width="48" height="72" class="fill-soft-b stroke-b"/><text x="164.0" y="194.0" text-anchor="middle" class="fig-small">W (3, 2)</text>
<text x="202" y="195.0" text-anchor="middle" class="fig-bold">=</text>
<rect x="216" y="154" width="48" height="72" class="fill-soft-d stroke-d"/><text x="240.0" y="194.0" text-anchor="middle" class="fig-small">dx (n, 2)</text>
<text x="300" y="190.0" class="fig-small">inner sizes match, 3 and 3: the result</text>
<text x="300" y="208.0" class="fig-small">has the input's width, 2</text>
<text x="20" y="262" class="fig-bold">Wrong: dY @ W.T</text>
<rect x="40" y="274" width="72" height="72" class="fill-soft-a stroke-a"/><text x="76.0" y="314.0" text-anchor="middle" class="fig-small">dY (n, 3)</text>
<text x="126" y="315.0" text-anchor="middle" class="fig-bold">@</text>
<rect x="140" y="274" width="72" height="48" class="fill-soft-b stroke-bad"/><text x="176.0" y="302.0" text-anchor="middle" class="fig-small">W.T (2, 3)</text>
<text x="226" y="315.0" text-anchor="middle" class="fig-bold">=</text>
<text x="300" y="310.0" class="fig-small">inner sizes 3 and 2 do not match,</text>
<text x="300" y="328.0" class="fig-small">so numpy raises an error</text>
</svg>
Figure 3. Shapes decide which weight matrix the backward pass uses. Forward uses W.T. Backward uses W itself, because its inner size then matches. On a non-square weight the wrong choice fails loudly. On a square weight both choices run, and only one is right.
```

> [!WARNING]
> In a block with square weights both `dY @ W` and `dY @ W.T` run, and only one is right. Choose by reasoning about the forward pass, because the error will not announce itself.

## Check it with finite differences

A formula for a gradient can be wrong while the code still runs. A **finite difference** gives an independent answer. Nudge one input number up by a tiny amount, nudge it down by the same amount, and divide the change in the loss by twice the nudge. To use it here, pick the loss $L = \sum (dY \odot y)$. Its gradient with respect to $y$ is exactly `dY`, so its gradient with respect to $x$ must equal what your backward pass returns. Repeat for each weight and bias. The check is slow, because it runs the forward pass twice per number, so you use it on tiny inputs. Keep the test points away from entries where $h_1$ is within the nudge of zero, because the ReLU has a corner there.

> [!TIP]
> After a backward pass, compare `grads[k].shape` with `params[k].shape` for every key. A wrong transpose often shows up as a shape error before any number is wrong.

> [!KEY]
> The loss's message reaches a block's input by two paths, so the input's gradient is the shortcut's `dY` plus the branch's term, and each weight's gradient is a transposed product of a message with the layer's input. Finite differences confirm each one.

> [!RECAP]
> Hold these points before you write code:
>
> - The plus sign sends `dY` to the branch and to the shortcut, so `dx = dY + dh1 @ W1`.
> - A weight gradient is the upstream gradient, transposed, times the layer's input. A bias gradient sums over the rows.
> - The ReLU mask `h1 > 0` is a switch: it passes the gradient where the unit was on and blocks it where off.
> - Backward multiplies by `W`. Forward multiplies by `W.T`.
> - On the running example, `dx` is `[[3, 1.5], [1, 1.5]]` for `dY` of all ones.

## Build it step by step

The starter already contains `relu` and `residual_block_forward`, finished in the previous chapter. The checks call the functions you write in this chapter in this order:

1. **`relu_mask(h)`.** `h` is an array of any shape. Return a float array of the same shape with 1.0 where `h > 0` and 0.0 elsewhere. Use `(h > 0).astype(float)`. The usual mistake is to use `h >= 0`, which lets a unit at exactly 0 pass gradient.
2. **`branch_gradient(dY, cache)`.** `dY` is `(n, d)` and `cache` is the tuple `(x, W1, W2, h1, a1)`. Return the branch's share of the input gradient, `dh1 @ W1`, shaped `(n, d)`. Work out `da1` and `dh1` with your mask. The usual mistake is to include `dY` here. This function returns only the branch term.
3. **`residual_block_backward(dY, cache)`.** Return `(dx, grads)` with keys `W1`, `b1`, `W2`, `b2`, each shaped like its parameter. Use `.T @`, `.sum(axis=0)` for the biases, and your two earlier functions. The usual mistake is to return only the branch term and forget to add `dY`.

## The exercise

You build `relu_mask`, `branch_gradient`, and `residual_block_backward`. The checks match the worked example, with every gradient above. They match finite differences for `x` and for all four parameters. When `W1` is zero, the branch term is zero and `dx` equals `dY` exactly. Running the file trains a stack of twenty residual blocks on a small regression problem and prints the loss falling, which only happens when every gradient is right.
