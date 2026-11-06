---
title: "Putting it together: gradients for a whole network"
slug: 057-backprop-mlp-and-gradient-check
part: "Part 4 — Neural networks"
summary: Wire the engine and the softmax loss into a two-layer network, read its slopes off one backward pass, and prove them with a finite-difference check.
minutes: 40
gpu: false
objectives:
  - Trace slopes by hand through a two-layer network and name where each parameter's slope comes from.
  - Compute the loss and every parameter gradient of a network with one backward pass.
  - Check a whole network's gradients against finite differences, and read the relative error.
  - Compare the cost of backprop with nudging every weight, and say what memory it spends.
prereqs:
  - 054-backprop-the-graph-and-reverse-order
  - 055-backprop-array-operations
  - 056-softmax-cross-entropy-gradient
exercise: 057-backprop-mlp-and-gradient-check
---

# Putting it together: gradients for a whole network

You have three pieces. An engine that records a calculation and walks slopes back through it, a rule for each array operation, and a loss with a readable slope. Each was tested alone. A network is where they meet: a matrix product, a bias, a ReLU, another product, another bias, and a softmax loss, all chained, with the weights buried in the middle.

Two things can go wrong when you wire them. A slope that is right for the last layer can be wrong for the first, because it has to survive every step between them, and one wrong rule far from the loss produces a gradient that looks reasonable and trains badly. The second worry is cost: is one backward pass really cheaper than nudging each weight and rerunning?

This chapter wires up a small network, traces its slopes by hand so you can see them cross layers, and builds the check that tells you whether any gradient can be trusted. The question to answer: do the slopes come out right for the weights deep inside, and is this really cheaper than nudging each weight?

## What you already know

- The engine from *Walking slopes backwards through a calculation* gives every input its slope in one backward walk.
- *The backward rule for each array operation* gave the rules for `@`, bias adds, ReLU, and sum.
- *Softmax and cross-entropy: a loss with a gradient you can read* gave a loss whose slope on the scores is $(p - \text{onehot}(y))/n$.
- A two-layer network computes `relu(X @ W0 + b0) @ W1 + b1` and feeds the result to the loss (see *Run a network forward, with a sharp or smooth bend*).
- Two-sided nudging, `central_difference`, estimates any slope.

## The picture: slopes crossing layers

Take one example with two inputs, $X = [1, 2]$, a hidden layer of two units, and two classes. Let the weights be

$$
W_0 = \begin{bmatrix} 1 & -1 \\ 0.5 & 0.5 \end{bmatrix},\quad
b_0 = [0.5, -0.5],\quad
W_1 = \begin{bmatrix} 0.4 & -0.2 \\ 1 & 0.5 \end{bmatrix},\quad
b_1 = [0, 0],
$$

and let the correct class be 1. Figure 1 shows every number.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A left-to-right chain from the inputs through a layer, a ReLU, a second layer and the loss, with forward values above and backward slopes below.">
<defs><marker id="c064-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker><marker id="c064-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="6.0" y="80.0" width="78.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="45.0" y="100.0" text-anchor="middle" class="fig-bold">X</text>
<text x="45.0" y="122.0" text-anchor="middle" class="fig-small fill-a">[1, 2]</text>
<text x="45.0" y="154.0" text-anchor="middle" class="fig-small fill-b">[0.491, 0.245]</text>
<rect x="132.0" y="80.0" width="78.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="171.0" y="100.0" text-anchor="middle" class="fig-bold">z0</text>
<text x="171.0" y="122.0" text-anchor="middle" class="fig-small fill-a">[2.5, -0.5]</text>
<text x="171.0" y="154.0" text-anchor="middle" class="fig-small fill-b">[0.491, 0]</text>
<rect x="258.0" y="80.0" width="78.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="297.0" y="100.0" text-anchor="middle" class="fig-bold">h</text>
<text x="297.0" y="122.0" text-anchor="middle" class="fig-small fill-a">[2.5, 0]</text>
<text x="297.0" y="154.0" text-anchor="middle" class="fig-small fill-b">[0.491, 0.409]</text>
<rect x="384.0" y="80.0" width="78.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="423.0" y="100.0" text-anchor="middle" class="fig-bold">z1</text>
<text x="423.0" y="122.0" text-anchor="middle" class="fig-small fill-a">[1, -0.5]</text>
<text x="423.0" y="154.0" text-anchor="middle" class="fig-small fill-b">[0.818, -0.818]</text>
<rect x="510.0" y="80.0" width="78.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="549.0" y="100.0" text-anchor="middle" class="fig-bold">loss</text>
<text x="549.0" y="122.0" text-anchor="middle" class="fig-small fill-a">1.7014</text>
<text x="549.0" y="154.0" text-anchor="middle" class="fig-small fill-b">1</text>
<line x1="86.0" y1="100.0" x2="129.0" y2="100.0" class="stroke-muted" marker-end="url(#c064-arrow-muted)"/>
<line x1="129.0" y1="116.0" x2="86.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c064-arrow-b)"/>
<text x="107.5" y="88.0" text-anchor="middle" class="fig-small">layer 1</text>
<line x1="212.0" y1="100.0" x2="255.0" y2="100.0" class="stroke-muted" marker-end="url(#c064-arrow-muted)"/>
<line x1="255.0" y1="116.0" x2="212.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c064-arrow-b)"/>
<text x="233.5" y="88.0" text-anchor="middle" class="fig-small">ReLU</text>
<line x1="338.0" y1="100.0" x2="381.0" y2="100.0" class="stroke-muted" marker-end="url(#c064-arrow-muted)"/>
<line x1="381.0" y1="116.0" x2="338.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c064-arrow-b)"/>
<text x="359.5" y="88.0" text-anchor="middle" class="fig-small">layer 2</text>
<line x1="464.0" y1="100.0" x2="507.0" y2="100.0" class="stroke-muted" marker-end="url(#c064-arrow-muted)"/>
<line x1="507.0" y1="116.0" x2="464.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c064-arrow-b)"/>
<text x="485.5" y="88.0" text-anchor="middle" class="fig-small">softmax</text>
<text x="300.0" y="24.0" text-anchor="middle" class="fig-small fill-muted">blue: values going forward   orange: slopes coming back</text>
<text x="6.0" y="182.0" text-anchor="middle"></text>
<text x="10.0" y="214.0" text-anchor="start" class="fig-bold">Slopes of the parameters, built from the slope at each layer's output</text>
<text x="20.0" y="240.0" text-anchor="start" class="fig-small">W1 slope = hᵀ × (slope at z1)</text>
<text x="240.0" y="240.0" text-anchor="start" class="fig-small fill-b">= [[2.044, -2.044], [0, 0]]</text>
<text x="20.0" y="262.0" text-anchor="start" class="fig-small">b1 slope = slope at z1</text>
<text x="240.0" y="262.0" text-anchor="start" class="fig-small fill-b">= [0.818, -0.818]</text>
<text x="20.0" y="284.0" text-anchor="start" class="fig-small">W0 slope = Xᵀ × (slope at z0)</text>
<text x="240.0" y="284.0" text-anchor="start" class="fig-small fill-b">= [[0.491, 0], [0.981, 0]]</text>
<text x="20.0" y="306.0" text-anchor="start" class="fig-small">b0 slope = slope at z0</text>
<text x="240.0" y="306.0" text-anchor="start" class="fig-small fill-b">= [0.491, 0]</text>
</svg>
Figure 1. One example flowing through a two-layer network, with every number computed. The slope at the loss is 1, and each step hands back a slope built from the one it received. The second hidden unit (unit 1) is closed, because its pre-activation is -0.5, so its slope is blocked and its weights get nothing.
```

Forward first. The hidden pre-activation is $z_0 = XW_0 + b_0 = [1 + 1 + 0.5,\ -1 + 1 - 0.5] = [2.5, -0.5]$. The ReLU makes it $h = [2.5, 0]$. The scores are $z_1 = hW_1 + b_1 = [1, -0.5]$. Softmax gives $p = [0.8176, 0.1824]$, and the loss is $-\log 0.1824 = 1.7014$.

Backward next, starting at the loss with slope 1. The previous chapter gives the slope on the scores: $p - \text{onehot}(1) = [0.8176, -0.8176]$. Call it $\bar{z}_1$. From there, each step uses a rule from the earlier chapters:

1. The second bias collects $\bar{z}_1$, so its slope is $[0.8176, -0.8176]$.
2. The second weights get $h^\top \bar{z}_1 = \begin{bmatrix} 2.044 & -2.044 \\ 0 & 0 \end{bmatrix}$. The second row is zero because hidden unit 1 output 0, so those weights did not affect the loss.
3. The hidden values get $\bar{h} = \bar{z}_1 W_1^\top = [0.4906, 0.4088]$.
4. The ReLU gate for hidden unit 1 is closed, because its pre-activation, the second entry of $z_0$, is $-0.5 < 0$. The slope passes unit 0 and stops at unit 1, so $\bar{z}_0 = [0.4906, 0]$.
5. The first bias collects $\bar{z}_0$, and the first weights get $X^\top \bar{z}_0 = \begin{bmatrix} 0.4906 & 0 \\ 0.9811 & 0 \end{bmatrix}$.

Notice what happened. The slope at $W_0$ came from the slope at the loss, passed back through every layer in turn, and each pass multiplied by that layer's own local slopes. Nothing about the first layer needed to know about the loss directly.

## Wire it up

The code mirrors the hand trace. Wrap every weight and bias in a fresh `Tensor`. Run the forward pass with the engine's operations, apply `.relu()` after every layer except the last, pass the final scores to `softmax_cross_entropy`, call `backward()` once on the loss, and copy each tensor's `grad`:

```python
h = Tensor(X)
for i, layer in enumerate(tensors):
    z = h @ layer["W"] + layer["b"]
    h = z.relu() if i < len(tensors) - 1 else z
loss = softmax_cross_entropy(h, y)
loss.backward()
```

Build fresh tensors on every call. A tensor's `grad` accumulates with `+=`, so a tensor reused across calls would add the second gradient onto the first. Likewise calling `backward()` twice on the same graph doubles every slope.

## Trust it: the gradient check

A wrong rule that runs without a crash is the dangerous kind. The cure is the two-sided nudge from *Walking slopes backwards through a calculation*, applied to every entry of every parameter. For one parameter array, nudge each entry up and down by $h$, rerun the loss, and divide the change by $2h$. That gives a **numeric gradient** of the same shape as the array.

Then compare it with the analytic gradient from backprop. Do not compare entry by entry against a fixed tolerance, because gradients vary in size. Use a relative error for each array:

$$
\text{error} = \frac{\lVert g_{\text{backprop}} - g_{\text{numeric}} \rVert}{\lVert g_{\text{backprop}} \rVert + \lVert g_{\text{numeric}} \rVert} .
$$

Here $\lVert g \rVert$ is the length of $g$ when you flatten it into one long vector. Read the formula aloud: the size of the disagreement, measured against the size of the two gradients. If the two agree, the error is near 0. If one is twice the other, the error is $1/3$. In float64 with $h = 10^{-5}$, a correct small network gives an error around $10^{-10}$, and anything above about $10^{-4}$ is a bug, not rounding.

Figure 2 shows what the check sees on a network with 12 first-layer weights. Its second layer is square, so a matrix-product rule with the transpose left off still runs instead of crashing, which is the dangerous case from *The backward rule for each array operation*. With the right rule, every point sits on the line where backprop equals the nudge. With the transpose missing, the points scatter, and the relative error jumps from about $3 \times 10^{-11}$ to about $0.6$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two scatter plots of backprop gradients against finite-difference gradients for the twelve weights of the first layer: one on the line, one scattered.">

<text x="150.0" y="30.0" text-anchor="middle" class="fig-bold">Correct rule</text>
<rect x="40.0" y="42.0" width="220.0" height="220.0" rx="0" class="stroke-muted"/>
<line x1="129.6" y1="262.0" x2="129.6" y2="42.0" class="stroke-grid"/>
<line x1="40.0" y1="172.4" x2="260.0" y2="172.4" class="stroke-grid"/>
<line x1="40.0" y1="262.0" x2="260.0" y2="42.0" class="stroke-muted dashed"/>
<circle cx="118.1" cy="183.9" r="4" class="fill-c"/>
<circle cx="169.6" cy="132.4" r="4" class="fill-c"/>
<circle cx="91.1" cy="210.9" r="4" class="fill-c"/>
<circle cx="100.1" cy="201.9" r="4" class="fill-c"/>
<circle cx="90.0" cy="212.0" r="4" class="fill-c"/>
<circle cx="244.8" cy="57.2" r="4" class="fill-c"/>
<circle cx="87.4" cy="214.6" r="4" class="fill-c"/>
<circle cx="162.1" cy="139.9" r="4" class="fill-c"/>
<circle cx="201.3" cy="100.7" r="4" class="fill-c"/>
<circle cx="112.6" cy="189.4" r="4" class="fill-c"/>
<circle cx="123.6" cy="178.4" r="4" class="fill-c"/>
<circle cx="90.1" cy="211.9" r="4" class="fill-c"/>
<text x="150.0" y="282.0" text-anchor="middle" class="fig-small">finite-difference gradient</text>
<text x="46.0" y="58.0" text-anchor="start" class="fig-small fill-muted">backprop gradient</text>
<text x="440.0" y="30.0" text-anchor="middle" class="fig-bold">Missing transpose</text>
<rect x="330.0" y="42.0" width="220.0" height="220.0" rx="0" class="stroke-muted"/>
<line x1="419.6" y1="262.0" x2="419.6" y2="42.0" class="stroke-grid"/>
<line x1="330.0" y1="172.4" x2="550.0" y2="172.4" class="stroke-grid"/>
<line x1="330.0" y1="262.0" x2="550.0" y2="42.0" class="stroke-muted dashed"/>
<circle cx="408.1" cy="180.9" r="4" class="fill-bad"/>
<circle cx="459.6" cy="96.5" r="4" class="fill-bad"/>
<circle cx="381.1" cy="165.3" r="4" class="fill-bad"/>
<circle cx="390.1" cy="169.6" r="4" class="fill-bad"/>
<circle cx="380.0" cy="246.8" r="4" class="fill-bad"/>
<circle cx="534.8" cy="191.4" r="4" class="fill-bad"/>
<circle cx="377.4" cy="213.0" r="4" class="fill-bad"/>
<circle cx="452.1" cy="94.8" r="4" class="fill-bad"/>
<circle cx="491.3" cy="177.8" r="4" class="fill-bad"/>
<circle cx="402.6" cy="173.6" r="4" class="fill-bad"/>
<circle cx="413.6" cy="173.5" r="4" class="fill-bad"/>
<circle cx="380.1" cy="169.6" r="4" class="fill-bad"/>
<text x="440.0" y="282.0" text-anchor="middle" class="fig-small">finite-difference gradient</text>
<text x="336.0" y="58.0" text-anchor="start" class="fig-small fill-muted">backprop gradient</text>
<text x="40.0" y="288.0" text-anchor="middle"></text>
<text x="8.0" y="140.0" text-anchor="middle"></text>
</svg>
Figure 2. For each of the twelve weights in the first layer, backprop's gradient (vertical) against the finite-difference gradient (horizontal). With the right rule every point sits on the dashed line y = x. With the transpose missing from one rule, the points scatter, and a check on the whole network would flag it at once.
```

The check has one blind spot, the ReLU kink. At exactly 0 the ReLU has no slope; the engine picks 0. A nudge of $\pm h$ around a pre-activation of $10^{-6}$ straddles the kink: with $h = 10^{-5}$ the two outputs are $1.1 \times 10^{-5}$ and 0, so the numeric slope is $0.55$, while the engine says 1. A disagreement from this is a false alarm. Check by trying another seed.

The check itself is expensive: it reruns the loss twice for every parameter entry, the $2P$ cost you are about to compare with backprop. So use it as a test on a small network, never as a way to train.

## Why this beats nudging every weight

A forward pass has some cost, call it one unit. Backprop adds a backward pass, and a backward pass costs about two units, because the matrix product does two products backward for the one it did forward. So all gradients cost about 3 units, whether you have a thousand parameters or a million. Nudging each weight up and down costs $2P$ units for $P$ parameters, and the forward-mode walk described in *Walking slopes backwards through a calculation* costs about $P$. Figure 3 puts these side by side.

```figure
<svg viewBox="0 0 600 336" role="img" aria-label="Horizontal bars on a log scale for the number of forward-pass equivalents needed by backprop, forward mode and finite differences, for a thousand and for a million parameters.">

<line x1="190.0" y1="42.0" x2="190.0" y2="306.0" class="stroke-grid"/>
<text x="190.0" y="34.0" text-anchor="middle" class="fig-small">1</text>
<line x1="244.3" y1="42.0" x2="244.3" y2="306.0" class="stroke-grid"/>
<text x="244.3" y="34.0" text-anchor="middle" class="fig-small">10</text>
<line x1="298.6" y1="42.0" x2="298.6" y2="306.0" class="stroke-grid"/>
<text x="298.6" y="34.0" text-anchor="middle" class="fig-small">100</text>
<line x1="352.9" y1="42.0" x2="352.9" y2="306.0" class="stroke-grid"/>
<text x="352.9" y="34.0" text-anchor="middle" class="fig-small">1k</text>
<line x1="407.1" y1="42.0" x2="407.1" y2="306.0" class="stroke-grid"/>
<text x="407.1" y="34.0" text-anchor="middle" class="fig-small">10k</text>
<line x1="461.4" y1="42.0" x2="461.4" y2="306.0" class="stroke-grid"/>
<text x="461.4" y="34.0" text-anchor="middle" class="fig-small">100k</text>
<line x1="515.7" y1="42.0" x2="515.7" y2="306.0" class="stroke-grid"/>
<text x="515.7" y="34.0" text-anchor="middle" class="fig-small">1M</text>
<line x1="570.0" y1="42.0" x2="570.0" y2="306.0" class="stroke-grid"/>
<text x="570.0" y="34.0" text-anchor="middle" class="fig-small">10M</text>
<text x="380.0" y="326.0" text-anchor="middle" class="fig-small fill-muted">forward-pass equivalents (log scale, rough)</text>
<text x="10.0" y="60.0" text-anchor="start" class="fig-bold">1,000 parameters</text>
<rect x="190.0" y="72.0" width="25.9" height="24.0" rx="3" class="fill-c"/>
<text x="182.0" y="88.0" text-anchor="end" class="fig-small">backprop</text>
<text x="221.9" y="88.0" text-anchor="start" class="fig-small">3</text>
<rect x="190.0" y="102.0" width="162.9" height="24.0" rx="3" class="fill-a"/>
<text x="182.0" y="118.0" text-anchor="end" class="fig-small">forward mode</text>
<text x="358.9" y="118.0" text-anchor="start" class="fig-small">1,000</text>
<rect x="190.0" y="132.0" width="179.2" height="24.0" rx="3" class="fill-b"/>
<text x="182.0" y="148.0" text-anchor="end" class="fig-small">nudging each weight</text>
<text x="375.2" y="148.0" text-anchor="start" class="fig-small">2,000</text>
<text x="10.0" y="182.0" text-anchor="start" class="fig-bold">1,000,000 parameters</text>
<rect x="190.0" y="194.0" width="25.9" height="24.0" rx="3" class="fill-c"/>
<text x="182.0" y="210.0" text-anchor="end" class="fig-small">backprop</text>
<text x="221.9" y="210.0" text-anchor="start" class="fig-small">3</text>
<rect x="190.0" y="224.0" width="325.7" height="24.0" rx="3" class="fill-a"/>
<text x="182.0" y="240.0" text-anchor="end" class="fig-small">forward mode</text>
<text x="521.7" y="240.0" text-anchor="start" class="fig-small">1,000,000</text>
<rect x="190.0" y="254.0" width="342.1" height="24.0" rx="3" class="fill-b"/>
<text x="182.0" y="270.0" text-anchor="end" class="fig-small">nudging each weight</text>
<text x="538.1" y="270.0" text-anchor="start" class="fig-small">2,000,000</text>
</svg>
Figure 3. How many forward-pass equivalents it costs to get every gradient. Backprop stays near 3 however many parameters there are, while the other two grow with the parameter count. The scale is logarithmic, so each gridline is ten times the last.
```

The price is memory. The backward walk needs the forward values: both inputs of every product, and the sign of every ReLU input. That means keeping, for every layer, arrays the size of batch times width, so memory grows with depth times batch size. **Checkpointing** keeps only a few of those layers and recomputes the rest during the backward walk, trading about one extra forward pass for less memory.

## Where this goes next

A gradient says which way is uphill. Training walks the other way: the updated weights are the old weights minus a **learning rate** times the gradient, `W - lr * grad`. One such step is `sgd_step`, which the exercise provides. The next chapter, *Noisy gradients: stochastic gradient descent and mini-batches*, builds better steps from the same gradients.

> [!KEY]
> One backward pass gives the gradient of every weight and bias in a network at about the cost of three forward passes, however many parameters there are, and a finite-difference check on the whole network is how you know the pass is right.

> [!RECAP]
> - Slopes cross layers by passing back through each layer's local rule: the loss slope becomes the score slope, then the hidden slope, then the ReLU-gated slope.
> - Parameter slopes come from the slope at a layer's output: `W` gets inputᵀ times it, and `b` gets its column sum.
> - A gradient check compares backprop with a two-sided nudge, using a relative error, and a correct network gives about $10^{-10}$.
> - A ReLU kink can fool the check, so rerun with other data before you doubt a rule.
> - Backprop costs about 3 forward passes for all gradients, and spends memory on the stored forward values.

## Build it step by step

The engine, `softmax_cross_entropy`, `central_difference`, and `sgd_step` are provided. You write three functions.

1. **`mlp_loss_and_grad(params, X, y)`** takes `params`, a list of `{"W": (in, out), "b": (out,)}` dicts, inputs of shape `(n, d)`, and integer labels. It returns `(loss, grads)` with `loss` a Python `float` and `grads` a list of dicts of the same shapes as `params`. The common mistake is applying ReLU to the last layer.

    ```python
    def mlp_loss_and_grad(params, X, y):
        # fresh Tensors; h @ W + b; relu except last; one backward; copy grads
    ```

2. **`numeric_grad(loss_fn, array, h=1e-5)`** takes a function from an array to a float and returns the two-sided gradient, entry by entry. It must restore each entry after the nudge and must not change the caller's array. The common mistake is leaving an entry nudged.

    ```python
    def numeric_grad(loss_fn, array, h=1e-5):
        # copy; for each index: nudge up, nudge down, restore
    ```

3. **`gradient_check(params, X, y)`** compares `mlp_loss_and_grad`'s gradients with `numeric_grad` for every `W` and `b`, scores each array with the relative error above, and returns the largest score. The common mistake is a loss function that nudges one array but leaves the others at stale values.

    ```python
    def gradient_check(params, X, y):
        # for each array: loss_fn replaces only that array
    ```

## The exercise

You implement `mlp_loss_and_grad`, `numeric_grad`, and `gradient_check`. The rest is provided.

The checks compare the loss with a plain numpy forward pass and the gradients with finite differences, and trace the hand-worked network above for loss $1.701413$ and its slopes. They check `numeric_grad` on a known function and `gradient_check` on a correct network, where the error must fall below $10^{-6}$. They also swap in a deliberately broken gradient, twice too large, and check that the error comes back near $1/3$. A last check runs `sgd_step` and confirms the loss falls.
