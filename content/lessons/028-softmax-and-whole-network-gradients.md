---
title: Softmax and whole-network gradients
slug: 028-softmax-and-whole-network-gradients
part: "Part 4 — Neural networks"
summary: Build softmax cross-entropy with its slope of prediction minus truth, then wire it into a network and prove the gradients with a nudge check.
minutes: 70
gpu: false
objectives:
  - Turn raw scores into probabilities with a softmax that does not overflow.
  - Explain why the loss is minus the log of the probability given to the correct class.
  - Derive that the slope of the loss on the scores is predicted probabilities minus the one-hot truth, divided by the batch size.
  - Trace slopes by hand through a two-layer network, and compute every parameter gradient with one backward pass.
  - Check a whole network's gradients against finite differences, and read the relative error.
prereqs:
  - 009-fitting-logistic-regression
  - 027-backpropagation
exercise: 028-softmax-and-whole-network-gradients
---

# Softmax and whole-network gradients

A network looks at a photo of a cat and ends with three raw scores, one for each of cat, dog, and fox: $2$, $1$, and $0$. The network leans toward cat, which is right, but how right? To train it you need one number that is small when the correct class has a high score and large when it does not, and you need the slope of that number with respect to each score, so the backward walk has something to start from.

The raw scores cannot be the loss. They can be negative, and a score of 2 means nothing until you compare it with the others. A tempting fix is the squared gap to the truth $[1, 0, 0]$. That charges the right class a penalty for scoring 2 instead of 1, although a higher score is exactly what you want.

This chapter builds the loss in two steps, turn the scores into probabilities and then score the probability given to the right answer, and finds that its gradient is simple enough to read off. Then it wires the loss into a small network, because a slope that is right for the last layer can be wrong for the first. A wrong rule far from the loss produces a gradient that looks reasonable and trains badly, so you also build the check that tells you whether any gradient can be trusted.

## What you already know

- A probability is between 0 and 1, and the probabilities of all the classes add to 1.
- The slope of logistic regression's loss is prediction minus truth ([fitting logistic regression](/lessons/009-fitting-logistic-regression)), and you hope to see the same shape again.
- The engine from [backpropagation](/lessons/027-backpropagation) gives every input its slope in one backward walk, with rules for `@`, bias adds, ReLU, and sum.
- Floating-point numbers overflow to `inf` when they get too big.

## The picture first

Figure 1 shows the first half of the chapter for the cat photo. The left panel is the raw scores, the middle panel is the same scores turned into probabilities, and the right panel is the slope the backward walk starts from, one number per score.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three bar panels for scores 2, 1, 0: the scores, the softmax probabilities, and the slope on each score.">

<text x="90.0" y="22.0" text-anchor="middle" class="fig-bold">Scores z</text>
<line x1="15.0" y1="200.0" x2="165.0" y2="200.0" class="stroke-muted"/>
<rect x="28.0" y="100.0" width="32.0" height="100.0" rx="2" class="fill-c"/>
<text x="44.0" y="94.0" text-anchor="middle" class="fig-small">2</text>
<text x="44.0" y="226.0" text-anchor="middle" class="fig-small">class 0</text>
<rect x="74.0" y="150.0" width="32.0" height="50.0" rx="2" class="fill-a"/>
<text x="90.0" y="144.0" text-anchor="middle" class="fig-small">1</text>
<text x="90.0" y="226.0" text-anchor="middle" class="fig-small">class 1</text>
<rect x="120.0" y="200.0" width="32.0" height="0.0" rx="2" class="fill-a"/>
<text x="136.0" y="194.0" text-anchor="middle" class="fig-small">0</text>
<text x="136.0" y="226.0" text-anchor="middle" class="fig-small">class 2</text>
<text x="285.0" y="22.0" text-anchor="middle" class="fig-bold">Probabilities p</text>
<line x1="210.0" y1="200.0" x2="360.0" y2="200.0" class="stroke-muted"/>
<rect x="223.0" y="86.9" width="32.0" height="113.1" rx="2" class="fill-c"/>
<text x="239.0" y="80.9" text-anchor="middle" class="fig-small">0.665</text>
<text x="239.0" y="226.0" text-anchor="middle" class="fig-small">class 0</text>
<rect x="269.0" y="158.4" width="32.0" height="41.6" rx="2" class="fill-b"/>
<text x="285.0" y="152.4" text-anchor="middle" class="fig-small">0.245</text>
<text x="285.0" y="226.0" text-anchor="middle" class="fig-small">class 1</text>
<rect x="315.0" y="184.7" width="32.0" height="15.3" rx="2" class="fill-b"/>
<text x="331.0" y="178.7" text-anchor="middle" class="fig-small">0.090</text>
<text x="331.0" y="226.0" text-anchor="middle" class="fig-small">class 2</text>
<text x="480.0" y="22.0" text-anchor="middle" class="fig-bold">Slope p − onehot</text>
<line x1="405.0" y1="150.0" x2="595.0" y2="150.0" class="stroke-muted"/>
<rect x="420.0" y="150.0" width="38.0" height="40.2" rx="2" class="fill-c"/>
<text x="439.0" y="204.2" text-anchor="middle" class="fig-small">-0.335</text>
<text x="439.0" y="226.0" text-anchor="middle" class="fig-small">class 0</text>
<rect x="476.0" y="120.6" width="38.0" height="29.4" rx="2" class="fill-b"/>
<text x="495.0" y="114.6" text-anchor="middle" class="fig-small">+0.245</text>
<text x="495.0" y="226.0" text-anchor="middle" class="fig-small">class 1</text>
<rect x="532.0" y="139.2" width="38.0" height="10.8" rx="2" class="fill-b"/>
<text x="551.0" y="133.2" text-anchor="middle" class="fig-small">+0.090</text>
<text x="551.0" y="226.0" text-anchor="middle" class="fig-small">class 2</text>
<text x="300.0" y="246.0" text-anchor="middle" class="fig-small fill-muted">green bar: the correct class (class 0)</text>
</svg>
Figure 1. For scores 2, 1, 0 and correct class 0. Softmax turns scores into probabilities 0.665, 0.245, 0.090. The slope on the scores is probability minus truth: negative for the correct class (raise it), positive for the others (lower them), larger where the model put more probability.
```

The correct class has a negative slope, which means the loss falls if that score rises. The two wrong classes have positive slopes, so the loss falls if they drop, and the more probability a wrong class has taken, the harder it is pushed down. The next sections earn this picture.

## Softmax: scores to probabilities

You need positive numbers that add to 1 and keep the order of the scores. Exponentiate each score, so every value becomes positive, then divide by the total. That is the **softmax**:

$$
p_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}} .
$$

Read it aloud: the probability of class $c$ is its exponentiated score divided by the sum of every exponentiated score. Dividing each score by the total fails because scores can be negative or sum to 0, and squaring would rank a score of $-3$ above a score of 1. The exponential turns adding to a score into multiplying its weight: raise a score by 1 and its share is multiplied by $e \approx 2.718$, whatever the other scores are. By hand, with $z = [2, 1, 0]$: the exponentials are $7.389$, $2.718$, and $1$, which sum to $11.107$, so the probabilities are $0.665$, $0.245$, and $0.090$.

## Cross-entropy: score the right answer

If the correct class is class 0, the model gave the truth probability $p_0 = 0.665$. You want a loss that is 0 when $p_0 = 1$ and grows as $p_0$ shrinks. Use minus the logarithm:

$$
\text{loss} = -\log p_{y} ,
$$

where $y$ is the index of the correct class. Why the logarithm? First, $-\log 1 = 0$, and $-\log p$ grows without limit as $p$ falls to 0, so a confident wrong answer is punished hard. Second, probabilities of independent examples multiply, and a logarithm turns multiplying into adding, so the loss over a batch is a plain average. The squared gap $(1 - p)^2$ never exceeds 1 and fades at the wrong end: with probability 0.01 on the right class, its slope on that score is about $-0.02$, while cross-entropy's is $-0.99$. Squared error whispers exactly when the model most needs a shout.

Compute everything for $z = [2, 1, 0]$ and correct class 0. Subtract the largest score, 2, to get $[0, -1, -2]$ (the reason comes shortly). Exponentiate to $[1,\ 0.367879,\ 0.135335]$, add to $1.503214$, and divide: $p = [0.665241,\ 0.244728,\ 0.090031]$. The loss is $-\log 0.665241 = 0.407606$. If the correct class were class 1, the loss would be $1.407606$, and for class 2 it is $2.407606$. Each class further down costs exactly 1 more, because the scores are 1 apart.

## The slope on each score

For one example, write the loss in another form. Because $p_y = e^{z_y} / S$ with $S = \sum_c e^{z_c}$,

$$
\text{loss} = -\log \frac{e^{z_y}}{S} = -z_y + \log S .
$$

The first term has slope $-1$ on $z_y$ and 0 on every other score. The second term depends on every score. Nudge $z_k$ by $h$ and $S$ grows by about $e^{z_k} h$, so $\log S$ grows by $(e^{z_k} / S)\, h = p_k h$. The slope of $\log S$ on $z_k$ is the probability itself. Adding the two pieces:

$$
\frac{\partial\, \text{loss}}{\partial z_k} = p_k - [k = y] .
$$

Read it aloud: the slope on each score is the predicted probability minus 1 if it is the right class and 0 if it is not. That is prediction minus truth, the same shape as logistic regression. For the example, $p - \text{onehot}(y) = [-0.334759,\ 0.244728,\ 0.090031]$. These slopes add to 0, which makes sense: adding the same number to all scores changes no probability, so the total slope on such a move must be 0.

A batch has $n$ examples and the loss is the mean of their losses, so each example's slope is divided by $n$:

$$
\frac{\partial\, \text{loss}}{\partial Z} = \frac{1}{n}\left(P - \text{onehot}(y)\right) .
$$

The numpy is three lines:

```python
g = probs.copy()
g[np.arange(n), labels] -= 1.0
g /= n
```

## Do not let exp overflow

Floating-point numbers top out near $1.8 \times 10^{308}$, so $e^z$ overflows to `inf` once $z$ passes about 709.8 (Figure 2). A score of 1000 would give `inf / inf`, which is `nan`. Subtract the largest score in each row first. The probabilities do not change, because the shift multiplies top and bottom of the fraction by the same factor $e^{-m}$, and every exponent is then at most 0, so every $e^z$ is at most 1.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="A line showing the size of e to the z in digits as z grows, crossing the largest representable number near z equal to 709.8, beside the shifted scores.">

<line x1="70.0" y1="220.0" x2="360.0" y2="220.0" class="stroke-muted"/>
<line x1="70.0" y1="220.0" x2="70.0" y2="30.0" class="stroke-muted"/>
<line x1="70.0" y1="220.0" x2="70.0" y2="224.0" class="stroke-muted"/>
<text x="70.0" y="238.0" text-anchor="middle" class="fig-small">0</text>
<line x1="142.5" y1="220.0" x2="142.5" y2="224.0" class="stroke-muted"/>
<text x="142.5" y="238.0" text-anchor="middle" class="fig-small">200</text>
<line x1="215.0" y1="220.0" x2="215.0" y2="224.0" class="stroke-muted"/>
<text x="215.0" y="238.0" text-anchor="middle" class="fig-small">400</text>
<line x1="287.5" y1="220.0" x2="287.5" y2="224.0" class="stroke-muted"/>
<text x="287.5" y="238.0" text-anchor="middle" class="fig-small">600</text>
<line x1="360.0" y1="220.0" x2="360.0" y2="224.0" class="stroke-muted"/>
<text x="360.0" y="238.0" text-anchor="middle" class="fig-small">800</text>
<line x1="66.0" y1="220.0" x2="70.0" y2="220.0" class="stroke-muted"/>
<text x="62.0" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="66.0" y1="162.4" x2="70.0" y2="162.4" class="stroke-muted"/>
<text x="62.0" y="166.4" text-anchor="end" class="fig-small">100</text>
<line x1="70.0" y1="162.4" x2="360.0" y2="162.4" class="stroke-grid"/>
<line x1="66.0" y1="104.8" x2="70.0" y2="104.8" class="stroke-muted"/>
<text x="62.0" y="108.8" text-anchor="end" class="fig-small">200</text>
<line x1="70.0" y1="104.8" x2="360.0" y2="104.8" class="stroke-grid"/>
<line x1="66.0" y1="47.3" x2="70.0" y2="47.3" class="stroke-muted"/>
<text x="62.0" y="51.3" text-anchor="end" class="fig-small">300</text>
<line x1="70.0" y1="47.3" x2="360.0" y2="47.3" class="stroke-grid"/>
<path d="M70.0,220.0 L327.3,42.5" class="stroke-b"/>
<path d="M327.3,42.5 L360.0,42.5" class="stroke-bad dashed"/>
<circle cx="327.3" cy="42.5" r="5" class="fill-bad"/>
<text x="315.3" y="16.5" text-anchor="end" class="fig-small fill-bad">z = 709.8: largest float</text>
<text x="360.0" y="34.5" text-anchor="end" class="fig-small fill-bad">beyond: inf</text>
<text x="215.0" y="262.0" text-anchor="middle" class="fig-small">score z (a logit)</text>
<text x="70.0" y="20.0" text-anchor="middle" class="fig-small">digits in e^z</text>
<text x="480.0" y="30.0" text-anchor="middle" class="fig-bold">Shift by the row maximum</text>
<text x="480.0" y="70.0" text-anchor="middle" class="fig-small fill-muted">scores</text>
<rect x="400.0" y="78.0" width="56.0" height="30.0" rx="6" class="fill-soft-a"/>
<text x="428.0" y="98.0" text-anchor="middle" class="fig-small">1000</text>
<rect x="462.0" y="78.0" width="56.0" height="30.0" rx="6" class="fill-soft-a"/>
<text x="490.0" y="98.0" text-anchor="middle" class="fig-small">998</text>
<rect x="524.0" y="78.0" width="56.0" height="30.0" rx="6" class="fill-soft-a"/>
<text x="552.0" y="98.0" text-anchor="middle" class="fig-small">990</text>
<text x="480.0" y="150.0" text-anchor="middle" class="fig-small fill-muted">after z − max</text>
<rect x="400.0" y="158.0" width="56.0" height="30.0" rx="6" class="fill-soft-b"/>
<text x="428.0" y="178.0" text-anchor="middle" class="fig-small">0</text>
<rect x="462.0" y="158.0" width="56.0" height="30.0" rx="6" class="fill-soft-b"/>
<text x="490.0" y="178.0" text-anchor="middle" class="fig-small">-2</text>
<rect x="524.0" y="158.0" width="56.0" height="30.0" rx="6" class="fill-soft-b"/>
<text x="552.0" y="178.0" text-anchor="middle" class="fig-small">-10</text>
<text x="480.0" y="222.0" text-anchor="middle" class="fig-small">e^1000 overflows to inf. After</text>
<text x="480.0" y="238.0" text-anchor="middle" class="fig-small">the shift: e^0 = 1, e^-2 = 0.135,</text>
<text x="480.0" y="254.0" text-anchor="middle" class="fig-small">e^-10 = 0.000045. No overflow.</text>
</svg>
Figure 2. Left: the number of digits in e^z grows in a straight line, and a float runs out near z = 709.8, so a score of 1000 gives inf. Right: subtracting the row maximum makes the largest score 0, so every exponent is at most 0 and e^z is at most 1. The probabilities do not change.
```

The loss needs the same care. Computing `np.log(softmax(z))` breaks when a probability underflows to 0, because $\log 0$ is $-\infty$ while the true loss is large but finite. Use the shifted form instead:

$$
\log p_c = (z_c - m) - \log \sum_{c'} e^{z_{c'} - m} .
$$

In numpy, `log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))`, and the loss is `-log_probs[np.arange(n), labels].mean()`. The usual mistake is `log_probs[:, labels]`, which returns an `(n, n)` block, not one entry per row.

> [!KEY]
> Softmax turns scores into probabilities, the loss is minus the log of the probability given to the right class, and the slope on each score is predicted probability minus truth, divided by the batch size. Subtract the row maximum so nothing overflows.

## Trace slopes through a whole network

Now wire the loss to a network: a matrix product, a bias, a ReLU, another product, another bias, and the softmax loss, with the weights buried in the middle. Take one example with two inputs, $X = [1, 2]$, a hidden layer of two units, and two classes. Let the weights be

$$
W_0 = \begin{bmatrix} 1 & -1 \\ 0.5 & 0.5 \end{bmatrix},\quad
b_0 = [0.5, -0.5],\quad
W_1 = \begin{bmatrix} 0.4 & -0.2 \\ 1 & 0.5 \end{bmatrix},\quad
b_1 = [0, 0],
$$

and let the correct class be 1. Figure 3 shows every number.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A left-to-right chain from the inputs through a layer, a ReLU, a second layer and the loss, with forward values above and backward slopes below.">
<defs><marker id="c028-arrow3-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker><marker id="c028-arrow3-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="86.0" y1="100.0" x2="129.0" y2="100.0" class="stroke-muted" marker-end="url(#c028-arrow3-muted)"/>
<line x1="129.0" y1="116.0" x2="86.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c028-arrow3-b)"/>
<text x="107.5" y="88.0" text-anchor="middle" class="fig-small">layer 1</text>
<line x1="212.0" y1="100.0" x2="255.0" y2="100.0" class="stroke-muted" marker-end="url(#c028-arrow3-muted)"/>
<line x1="255.0" y1="116.0" x2="212.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c028-arrow3-b)"/>
<text x="233.5" y="88.0" text-anchor="middle" class="fig-small">ReLU</text>
<line x1="338.0" y1="100.0" x2="381.0" y2="100.0" class="stroke-muted" marker-end="url(#c028-arrow3-muted)"/>
<line x1="381.0" y1="116.0" x2="338.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c028-arrow3-b)"/>
<text x="359.5" y="88.0" text-anchor="middle" class="fig-small">layer 2</text>
<line x1="464.0" y1="100.0" x2="507.0" y2="100.0" class="stroke-muted" marker-end="url(#c028-arrow3-muted)"/>
<line x1="507.0" y1="116.0" x2="464.0" y2="116.0" class="stroke-b dashed" marker-end="url(#c028-arrow3-b)"/>
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
Figure 3. One example flowing through a two-layer network, with every number computed. The slope at the loss is 1, and each step hands back a slope built from the one it received. The second hidden unit (unit 1) is closed, because its pre-activation is -0.5, so its slope is blocked and its weights get nothing.
```

Forward first. The hidden pre-activation is $z_0 = XW_0 + b_0 = [2.5, -0.5]$, and the ReLU makes it $h = [2.5, 0]$. The scores are $z_1 = hW_1 + b_1 = [1, -0.5]$. Softmax gives $p = [0.8176, 0.1824]$, and the loss is $-\log 0.1824 = 1.7014$.

Backward next, starting at the loss with slope 1. The slope on the scores is $\bar{z}_1 = p - \text{onehot}(1) = [0.8176, -0.8176]$. From there, each step uses a rule from the previous chapter:

1. The second bias collects $\bar{z}_1$, so its slope is $[0.8176, -0.8176]$.
2. The second weights get $h^\top \bar{z}_1 = \begin{bmatrix} 2.044 & -2.044 \\ 0 & 0 \end{bmatrix}$. The second row is zero because hidden unit 1 output 0, so those weights did not affect the loss.
3. The hidden values get $\bar{h} = \bar{z}_1 W_1^\top = [0.4906, 0.4088]$.
4. The ReLU gate for hidden unit 1 is closed, because its pre-activation $-0.5$ is negative. The slope passes unit 0 and stops at unit 1, so $\bar{z}_0 = [0.4906, 0]$.
5. The first bias collects $\bar{z}_0$, and the first weights get $X^\top \bar{z}_0 = \begin{bmatrix} 0.4906 & 0 \\ 0.9811 & 0 \end{bmatrix}$.

The slope at $W_0$ came from the slope at the loss, passed back through every layer in turn, and each pass multiplied by that layer's own local slopes. The code mirrors the trace. Wrap every weight and bias in a fresh `Tensor`, run the forward pass, apply `.relu()` after every layer except the last, pass the final scores to `softmax_cross_entropy`, call `backward()` once, and copy each tensor's `grad`:

```python
h = Tensor(X)
for i, layer in enumerate(tensors):
    z = h @ layer["W"] + layer["b"]
    h = z.relu() if i < len(tensors) - 1 else z
loss = softmax_cross_entropy(h, y)
loss.backward()
```

> [!WARNING]
> Build fresh tensors on every call. A tensor's `grad` accumulates with `+=`, so a tensor reused across calls adds the second gradient onto the first, and calling `backward()` twice on the same graph doubles every slope.

## Trust it: the gradient check

A wrong rule that runs without a crash is the dangerous kind. The cure is the two-sided nudge applied to every entry of every parameter: nudge each entry up and down by $h$, rerun the loss, and divide the change by $2h$. That gives a **numeric gradient** of the same shape as the array. Compare it with the analytic gradient from backprop using a relative error for each array, because gradients vary in size and a fixed tolerance would be unfair:

$$
\text{error} = \frac{\lVert g_{\text{backprop}} - g_{\text{numeric}} \rVert}{\lVert g_{\text{backprop}} \rVert + \lVert g_{\text{numeric}} \rVert} .
$$

Here $\lVert g \rVert$ is the length of $g$ flattened into one long vector. Read it aloud: the size of the disagreement, measured against the size of the two gradients. If the two agree, the error is near 0. If one is twice the other, the error is $1/3$. In float64 with $h = 10^{-5}$, a correct small network gives an error around $10^{-10}$, and anything above about $10^{-4}$ is a bug, not rounding.

Figure 4 shows what the check sees on a network with 12 first-layer weights. Its second layer is square, so a matrix-product rule with the transpose left off still runs instead of crashing. With the right rule, every point sits on the line where backprop equals the nudge. With the transpose missing, the points scatter, and the relative error jumps from about $3 \times 10^{-11}$ to about $0.6$.

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
Figure 4. For each of the twelve weights in the first layer, backprop's gradient (vertical) against the finite-difference gradient (horizontal). With the right rule every point sits on the dashed line y = x. With the transpose missing from one rule, the points scatter, and a check on the whole network would flag it at once.
```

The check has one blind spot, the ReLU kink. At exactly 0 the ReLU has no slope, and the engine picks 0. With a pre-activation of $10^{-6}$ and $h = 10^{-5}$, the nudged outputs are $1.1 \times 10^{-5}$ and 0, so the numeric slope is $0.55$ while the engine says 1. A disagreement from this is a false alarm, so try another seed.

> [!TIP]
> Run the check on a small network and never use it to train: it reruns the loss twice for every parameter entry. Backprop gets all gradients for about three forward passes, whether you have a thousand parameters or a million, at the cost of storing the forward values.

> [!KEY]
> One backward pass gives the gradient of every weight and bias in a network, starting from a loss whose slope on the scores is prediction minus truth. A finite-difference check on the whole network, scored by relative error, is how you know the pass is right.

> [!RECAP]
> - Softmax exponentiates each score and divides by the sum. Cross-entropy is $-\log p_y$, and the slope on score $k$ is $p_k - [k = y]$, divided by $n$ for a mean over $n$ rows.
> - Subtract the row maximum in both softmax and log-softmax, and pick the correct entry per row with `[np.arange(n), labels]`.
> - Slopes cross layers by passing back through each layer's local rule. `W` gets inputᵀ times the slope at its output, and `b` gets its column sum.
> - A gradient check compares backprop with a two-sided nudge using a relative error, and a correct network gives about $10^{-10}$.
> - A ReLU kink can fool the check, so rerun with other data before you doubt a rule.

## Build it step by step

The `Tensor` engine, `central_difference`, and `sgd_step` are provided. You write six functions, and each builds on the one before.

1. **`softmax(z)`** takes an `(n, c)` array and returns an `(n, c)` array whose rows are positive and add to 1. Subtract `z.max(axis=1, keepdims=True)`, exponentiate, and divide by the row sums. The common mistake is forgetting `keepdims=True`, which breaks the broadcast.

2. **`log_softmax(z)`** returns the same shape, the logarithm of the softmax computed in the shifted form. It must stay finite at scores of plus or minus 1000. The common mistake is `np.log(softmax(z))`.

3. **`softmax_cross_entropy(logits, labels)`** takes a `Tensor` of shape `(n, c)` and an integer array of shape `(n,)`, and returns a scalar `Tensor` whose only child is `logits`. The backward rule adds `out.grad * (probs - onehot) / n` to `logits.grad`. The common mistake is a missing `/ n`.

4. **`mlp_loss_and_grad(params, X, y)`** takes `params`, a list of `{"W": (in, out), "b": (out,)}` dicts, inputs of shape `(n, d)`, and integer labels. It returns `(loss, grads)` with `loss` a Python `float` and `grads` a list of dicts of the same shapes as `params`. The common mistake is applying ReLU to the last layer.

5. **`numeric_grad(loss_fn, array, h=1e-5)`** takes a function from an array to a float and returns the two-sided gradient, entry by entry. It must restore each entry after the nudge and must not change the caller's array. The common mistake is leaving an entry nudged.

6. **`gradient_check(params, X, y)`** compares `mlp_loss_and_grad`'s gradients with `numeric_grad` for every `W` and `b`, scores each array with the relative error above, and returns the largest score. The common mistake is a loss function that nudges one array but leaves the others at stale values.

## The exercise

You implement `softmax`, `log_softmax`, `softmax_cross_entropy`, `mlp_loss_and_grad`, `numeric_grad`, and `gradient_check`. The checks run `softmax` on $[2, 1, 0]$, confirm that a shift changes nothing, and that scores of 1000 stay finite. They check `log_softmax` at plus or minus 1000, the loss $0.407606$ for the worked example, and the slope $[-0.334759, 0.244728, 0.090031]$ against finite differences and $(p - \text{onehot}) / n$. For the network, they compare `mlp_loss_and_grad` with a plain numpy forward pass, finite differences, and the hand-traced loss $1.701413$. They check `numeric_grad` on a known function, and `gradient_check` on a correct network, where the error must fall below $10^{-6}$, and on a deliberately broken gradient, twice too large, where it must come back near $1/3$. A last check runs `sgd_step` and confirms the loss falls.
