---
title: "Softmax and cross-entropy: a loss with a gradient you can read"
slug: 063-softmax-cross-entropy-gradient
part: "Part 4 — Neural networks"
summary: Softmax turns scores into probabilities, cross-entropy scores the one given to the right class, and the slope on each score is prediction minus truth.
minutes: 40
gpu: false
objectives:
  - Turn raw scores into probabilities with a softmax that does not overflow.
  - Explain why the loss is minus the log of the probability given to the correct class.
  - Derive that the slope of the loss on the scores is predicted probabilities minus the one-hot truth, divided by the batch size.
  - Compute a stable log-softmax and the loss for a batch.
prereqs:
  - 019-gradient-descent-for-logistic
  - 061-backprop-the-graph-and-reverse-order
  - 062-backprop-array-operations
exercise: 063-softmax-cross-entropy-gradient
---

# Softmax and cross-entropy: a loss with a gradient you can read

A network looks at a photo of a cat and ends with three raw scores, one for each of cat, dog, and fox: $2$, $1$, and $0$. The network is leaning toward cat, which is right, but how right? To train it you need one number that is small when the correct class has a high score and large when it does not, and you need the slope of that number with respect to each of the three scores, so the backward walk has something to start from.

The raw scores cannot be the loss. They can be negative, they do not add up to anything, and a score of 2 means nothing until you compare it with the others. A tempting fix is to compare to the truth directly: the truth is $[1, 0, 0]$, the scores are $[2, 1, 0]$, so take the squared gap. That fix charges the right class a penalty for scoring 2 instead of 1, although a higher score is exactly what you want.

This chapter builds the loss in two steps: turn the scores into probabilities, then score the probability given to the right answer. The result has a gradient so simple you can read it off. The question to answer: how do you turn scores into a loss, and what is its slope on each score?

## What you already know

- A probability is between 0 and 1, and the probabilities of all the classes add to 1.
- A loss is one number, small for good predictions, and training walks against its slope (see *Learning the weights: gradient and descent*).
- The slope of logistic regression's loss is prediction minus truth, and you hope to see the same shape again.
- The backward rules for sums, products, and broadcasting from the previous two chapters, and `+=` accumulation.
- Floating-point numbers overflow to `inf` when they get too big.

## The picture first

Figure 1 shows the whole chapter for the cat photo. The left panel is the raw scores. The middle panel is the same scores turned into probabilities. The right panel is the slope the backward walk will start from, one number per score.

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

Look at the right panel. The correct class has a negative slope, which means the loss falls if that score rises. The two wrong classes have positive slopes, so the loss falls if they drop. The bigger the probability a wrong class has taken, the harder it is pushed down. The rest of the chapter earns this picture.

## Softmax: scores to probabilities

You need positive numbers that add to 1, and that keep the order of the scores. Exponentiate each score, so every value becomes positive, then divide by the total so they add to 1. That is the **softmax**:

$$
p_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}} .
$$

Read it aloud: the probability of class $c$ is its exponentiated score divided by the sum of every exponentiated score. Why not divide each score by the total, or square them first? Dividing fails because scores can be negative or sum to 0, and squaring would rank a score of $-3$ above a score of 1. Why the exponential? An exponential turns adding to a score into multiplying its weight: raise a score by 1 and its share is multiplied by $e \approx 2.718$, whatever the other scores are. It is also never negative, so the division always gives a valid share.

By hand, with $z = [2, 1, 0]$: the exponentials are $e^2 = 7.389$, $e^1 = 2.718$, and $e^0 = 1$. Their sum is $11.107$, so the probabilities are $0.665$, $0.245$, and $0.090$. You will compute them without overflow below.

## Cross-entropy: score the right answer

If the correct class is class 0, the probability the model gave to the truth is $p_0 = 0.665$. We want a loss that is 0 when $p_0 = 1$ and grows as $p_0$ shrinks. Use minus the logarithm:

$$
\text{loss} = -\log p_{y} ,
$$

where $y$ is the index of the correct class. Why the logarithm? Two reasons. First, $-\log 1 = 0$, and $-\log p$ grows without limit as $p$ falls to 0, so a confident wrong answer is punished hard (Figure 2). Second, probabilities of independent examples multiply, and a logarithm turns multiplying into adding, so the loss over a batch is a plain average.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The curve minus log p against p, rising without limit near zero, compared with the capped curve one minus p squared.">

<line x1="60.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="60.0" y1="250.0" x2="60.0" y2="20.0" class="stroke-muted"/>
<line x1="60.0" y1="250.0" x2="60.0" y2="254.0" class="stroke-muted"/>
<text x="60.0" y="268.0" text-anchor="middle" class="fig-small">0</text>
<line x1="185.0" y1="250.0" x2="185.0" y2="254.0" class="stroke-muted"/>
<text x="185.0" y="268.0" text-anchor="middle" class="fig-small">0.25</text>
<line x1="310.0" y1="250.0" x2="310.0" y2="254.0" class="stroke-muted"/>
<text x="310.0" y="268.0" text-anchor="middle" class="fig-small">0.5</text>
<line x1="435.0" y1="250.0" x2="435.0" y2="254.0" class="stroke-muted"/>
<text x="435.0" y="268.0" text-anchor="middle" class="fig-small">0.75</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="254.0" class="stroke-muted"/>
<text x="560.0" y="268.0" text-anchor="middle" class="fig-small">1</text>
<line x1="56.0" y1="250.0" x2="60.0" y2="250.0" class="stroke-muted"/>
<text x="52.0" y="254.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="202.1" x2="60.0" y2="202.1" class="stroke-muted"/>
<text x="52.0" y="206.1" text-anchor="end" class="fig-small">1</text>
<line x1="60.0" y1="202.1" x2="560.0" y2="202.1" class="stroke-grid"/>
<line x1="56.0" y1="154.2" x2="60.0" y2="154.2" class="stroke-muted"/>
<text x="52.0" y="158.2" text-anchor="end" class="fig-small">2</text>
<line x1="60.0" y1="154.2" x2="560.0" y2="154.2" class="stroke-grid"/>
<line x1="56.0" y1="106.2" x2="60.0" y2="106.2" class="stroke-muted"/>
<text x="52.0" y="110.2" text-anchor="end" class="fig-small">3</text>
<line x1="60.0" y1="106.2" x2="560.0" y2="106.2" class="stroke-grid"/>
<line x1="56.0" y1="58.3" x2="60.0" y2="58.3" class="stroke-muted"/>
<text x="52.0" y="62.3" text-anchor="end" class="fig-small">4</text>
<line x1="60.0" y1="58.3" x2="560.0" y2="58.3" class="stroke-grid"/>
<path d="M65.0,29.3 L68.3,53.6 L71.6,69.7 L74.9,81.7 L78.2,91.2 L81.5,99.2 L84.8,106.1 L88.1,112.1 L91.4,117.4 L94.7,122.2 L98.0,126.5 L101.3,130.5 L104.6,134.2 L107.9,137.6 L111.2,140.8 L114.5,143.8 L117.8,146.6 L121.1,149.3 L124.4,151.8 L127.7,154.2 L131.0,156.5 L134.3,158.6 L137.6,160.7 L140.9,162.7 L144.2,164.6 L147.5,166.5 L150.8,168.3 L154.1,170.0 L157.4,171.6 L160.7,173.2 L164.0,174.8 L167.3,176.3 L170.6,177.7 L173.9,179.1 L177.2,180.5 L180.5,181.8 L183.8,183.1 L187.1,184.4 L190.4,185.6 L193.7,186.8 L197.0,188.0 L200.3,189.1 L203.6,190.2 L206.9,191.3 L210.2,192.4 L213.5,193.4 L216.8,194.4 L220.1,195.4 L223.4,196.4 L226.7,197.4 L230.0,198.3 L233.3,199.2 L236.6,200.1 L239.9,201.0 L243.2,201.9 L246.5,202.7 L249.8,203.6 L253.1,204.4 L256.4,205.2 L259.7,206.0 L263.0,206.8 L266.3,207.6 L269.6,208.3 L272.9,209.1 L276.2,209.8 L279.5,210.6 L282.8,211.3 L286.1,212.0 L289.4,212.7 L292.7,213.4 L296.0,214.0 L299.3,214.7 L302.6,215.3 L305.9,216.0 L309.2,216.6 L312.5,217.3 L315.8,217.9 L319.1,218.5 L322.4,219.1 L325.7,219.7 L329.0,220.3 L332.3,220.9 L335.6,221.5 L338.9,222.0 L342.2,222.6 L345.5,223.1 L348.8,223.7 L352.1,224.2 L355.4,224.8 L358.7,225.3 L362.0,225.8 L365.3,226.4 L368.6,226.9 L371.9,227.4 L375.2,227.9 L378.5,228.4 L381.8,228.9 L385.1,229.4 L388.4,229.9 L391.7,230.3 L395.0,230.8 L398.3,231.3 L401.6,231.7 L404.9,232.2 L408.2,232.7 L411.5,233.1 L414.8,233.6 L418.1,234.0 L421.4,234.4 L424.7,234.9 L428.0,235.3 L431.3,235.7 L434.6,236.2 L437.9,236.6 L441.2,237.0 L444.5,237.4 L447.8,237.8 L451.1,238.2 L454.4,238.6 L457.7,239.0 L461.0,239.4 L464.3,239.8 L467.6,240.2 L470.9,240.6 L474.2,241.0 L477.5,241.4 L480.8,241.7 L484.1,242.1 L487.4,242.5 L490.7,242.9 L494.0,243.2 L497.3,243.6 L500.6,243.9 L503.9,244.3 L507.2,244.7 L510.5,245.0 L513.8,245.4 L517.1,245.7 L520.4,246.0 L523.7,246.4 L527.0,246.7 L530.3,247.1 L533.6,247.4 L536.9,247.7 L540.2,248.1 L543.5,248.4 L546.8,248.7 L550.1,249.0 L553.4,249.4 L556.7,249.7 L560.0,250.0" class="stroke-b"/>
<path d="M65.0,203.0 L68.3,203.7 L71.6,204.3 L74.9,204.9 L78.2,205.5 L81.5,206.1 L84.8,206.7 L88.1,207.3 L91.4,207.9 L94.7,208.5 L98.0,209.1 L101.3,209.7 L104.6,210.3 L107.9,210.8 L111.2,211.4 L114.5,212.0 L117.8,212.5 L121.1,213.1 L124.4,213.6 L127.7,214.2 L131.0,214.7 L134.3,215.3 L137.6,215.8 L140.9,216.3 L144.2,216.9 L147.5,217.4 L150.8,217.9 L154.1,218.4 L157.4,218.9 L160.7,219.4 L164.0,219.9 L167.3,220.4 L170.6,220.9 L173.9,221.4 L177.2,221.9 L180.5,222.4 L183.8,222.9 L187.1,223.3 L190.4,223.8 L193.7,224.3 L197.0,224.7 L200.3,225.2 L203.6,225.7 L206.9,226.1 L210.2,226.5 L213.5,227.0 L216.8,227.4 L220.1,227.9 L223.4,228.3 L226.7,228.7 L230.0,229.1 L233.3,229.5 L236.6,230.0 L239.9,230.4 L243.2,230.8 L246.5,231.2 L249.8,231.6 L253.1,231.9 L256.4,232.3 L259.7,232.7 L263.0,233.1 L266.3,233.5 L269.6,233.8 L272.9,234.2 L276.2,234.6 L279.5,234.9 L282.8,235.3 L286.1,235.6 L289.4,236.0 L292.7,236.3 L296.0,236.6 L299.3,237.0 L302.6,237.3 L305.9,237.6 L309.2,237.9 L312.5,238.3 L315.8,238.6 L319.1,238.9 L322.4,239.2 L325.7,239.5 L329.0,239.8 L332.3,240.1 L335.6,240.3 L338.9,240.6 L342.2,240.9 L345.5,241.2 L348.8,241.5 L352.1,241.7 L355.4,242.0 L358.7,242.2 L362.0,242.5 L365.3,242.7 L368.6,243.0 L371.9,243.2 L375.2,243.5 L378.5,243.7 L381.8,243.9 L385.1,244.1 L388.4,244.4 L391.7,244.6 L395.0,244.8 L398.3,245.0 L401.6,245.2 L404.9,245.4 L408.2,245.6 L411.5,245.8 L414.8,246.0 L418.1,246.1 L421.4,246.3 L424.7,246.5 L428.0,246.7 L431.3,246.8 L434.6,247.0 L437.9,247.1 L441.2,247.3 L444.5,247.4 L447.8,247.6 L451.1,247.7 L454.4,247.9 L457.7,248.0 L461.0,248.1 L464.3,248.2 L467.6,248.4 L470.9,248.5 L474.2,248.6 L477.5,248.7 L480.8,248.8 L484.1,248.9 L487.4,249.0 L490.7,249.1 L494.0,249.2 L497.3,249.2 L500.6,249.3 L503.9,249.4 L507.2,249.5 L510.5,249.5 L513.8,249.6 L517.1,249.6 L520.4,249.7 L523.7,249.7 L527.0,249.8 L530.3,249.8 L533.6,249.9 L536.9,249.9 L540.2,249.9 L543.5,249.9 L546.8,250.0 L550.1,250.0 L553.4,250.0 L556.7,250.0 L560.0,250.0" class="stroke-muted dashed"/>
<text x="88.0" y="39.2" text-anchor="start" class="fig-bold fill-b">−log p</text>
<text x="260.0" y="247.6" text-anchor="start" class="fig-small">(1 − p)²</text>
<circle cx="392.6" cy="230.5" r="5" class="fill-c"/>
<text x="400.6" y="222.5" text-anchor="start" class="fig-small">class 0: p = 0.665, loss 0.408</text>
<circle cx="182.4" cy="182.6" r="5" class="fill-ink"/>
<text x="190.4" y="174.6" text-anchor="start" class="fig-small">class 1: p = 0.245, loss 1.408</text>
<circle cx="105.0" cy="134.6" r="5" class="fill-ink"/>
<text x="113.0" y="126.6" text-anchor="start" class="fig-small">class 2: p = 0.090, loss 2.408</text>
<text x="310.0" y="290.0" text-anchor="middle" class="fig-small">probability p given to the correct class</text>
</svg>
Figure 2. The loss as a function of the probability given to the correct answer. Cross-entropy (orange) grows without limit as p falls toward 0, while squared error (dashed) never exceeds 1. The three dots are the example's three possible correct answers.
```

Compare this with the squared gap $(1 - p)^2$, which never exceeds 1. It also fades at the wrong end: when the model puts probability 0.01 on the right class, the squared-error slope on that class's score is about $-0.02$, while cross-entropy's is $-0.99$. Squared error whispers exactly when the model most needs a shout.

## The worked example

Compute everything for $z = [2, 1, 0]$ and correct class 0.

1. Subtract the largest score, 2. The shifted scores are $[0, -1, -2]$. The reason comes at the end of the chapter.
2. Exponentiate: $[1,\ 0.367879,\ 0.135335]$.
3. Add: $1.503214$.
4. Divide: $p = [0.665241,\ 0.244728,\ 0.090031]$. These add to 1.
5. The loss is $-\log 0.665241 = 0.407606$.

If the correct class were class 1 instead, the loss would be $-\log 0.244728 = 1.407606$, and for class 2 it is $2.407606$. Each class further down costs exactly 1 more, because the scores are 1 apart.

## The slope on each score

Next comes the part the backward walk needs. For one example, write the loss in another form. Because $p_y = e^{z_y} / S$ with $S = \sum_c e^{z_c}$,

$$
\text{loss} = -\log \frac{e^{z_y}}{S} = -z_y + \log S .
$$

The first term is plain: its slope on $z_y$ is $-1$ and on every other score is 0. The second term, $\log S$, depends on every score. Nudge $z_k$ by $h$ and $S$ grows by about $e^{z_k} h$, so $\log S$ grows by $(e^{z_k} / S)\, h = p_k h$. The slope of $\log S$ on $z_k$ is $p_k$, the probability itself. Adding the two pieces:

$$
\frac{\partial\, \text{loss}}{\partial z_k} = p_k - [k = y] .
$$

Read it aloud: the slope on each score is the predicted probability minus 1 if it is the right class and 0 if it is not. That is prediction minus truth, the same shape as logistic regression. For the example, subtract 1 at class 0:

$$
p - \text{onehot}(y) = [0.665241 - 1,\ 0.244728,\ 0.090031] = [-0.334759,\ 0.244728,\ 0.090031] .
$$

These slopes add to 0. That makes sense: adding the same number to all scores changes no probability, so no loss, and the total slope on such a move must be 0.

A batch has $n$ examples, and the loss is the mean of their individual losses, so each example's slope is divided by $n$:

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

Floating-point numbers top out near $1.8 \times 10^{308}$, so $e^z$ overflows to `inf` once $z$ passes about 709.8 (Figure 3). A network that ends with a score of 1000 would then give `inf / inf`, which is `nan`. Subtract the largest score in each row first. The probabilities do not change, because the shift multiplies top and bottom of the fraction by the same factor $e^{-m}$, and every exponent is then at most 0, so every $e^z$ is at most 1.

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
Figure 3. Left: the number of digits in e^z grows in a straight line, and a float runs out near z = 709.8, so a score of 1000 gives inf. Right: subtracting the row maximum makes the largest score 0, so every exponent is at most 0 and e^z is at most 1. The probabilities do not change.
```

The loss needs the same care. Computing `np.log(softmax(z))` breaks when a probability underflows to 0, because $\log 0$ is $-\infty$ while the true loss is large but finite. Use the shifted form instead:

$$
\log p_c = (z_c - m) - \log \sum_{c'} e^{z_{c'} - m} .
$$

In numpy, `log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))`, and the loss is `-log_probs[np.arange(n), labels].mean()`. The usual mistake is `log_probs[:, labels]`, which returns an `(n, n)` block, not one entry per row.

> [!KEY]
> Softmax turns scores into probabilities, the loss is minus the log of the probability given to the right class, and the slope on each score is predicted probability minus truth, divided by the batch size. Subtract the row maximum so nothing overflows.

> [!RECAP]
> - Softmax exponentiates each score and divides by the sum, so the results are positive and add to 1.
> - Cross-entropy is $-\log p_y$: 0 for a certain and correct model, unbounded for a confident wrong one.
> - Writing the loss as $-z_y + \log \sum e^{z}$ shows the slope on score $k$ is $p_k - [k = y]$, divided by $n$ for a mean over $n$ rows.
> - Subtract the row maximum in both softmax and log-softmax. The result is unchanged and nothing overflows.
> - Pick the correct entry per row with `[np.arange(n), labels]`.

## Build it step by step

The `Tensor` engine from the previous chapters is provided complete. You write three functions, and each one builds on the one before.

1. **`softmax(z)`** takes an `(n, c)` array and returns an `(n, c)` array whose rows are positive and add to 1. Subtract `z.max(axis=1, keepdims=True)`, exponentiate, and divide by the row sums. The common mistake is forgetting `keepdims=True`, which breaks the broadcast.

    ```python
    def softmax(z):
        # shift by the row max, exp, divide by the row sum
    ```

2. **`log_softmax(z)`** returns the same shape, the logarithm of the softmax computed in the shifted form. It must stay finite at scores of plus or minus 1000. The common mistake is `np.log(softmax(z))`.

    ```python
    def log_softmax(z):
        # shifted - log(sum(exp(shifted)))
    ```

3. **`softmax_cross_entropy(logits, labels)`** takes a `Tensor` of shape `(n, c)` and an integer array of shape `(n,)`, and returns a scalar `Tensor` whose only child is `logits`. The backward rule adds `out.grad * (probs - onehot) / n` to `logits.grad`. The common mistake is a missing `/ n`.

    ```python
    def softmax_cross_entropy(logits, labels):
        # loss from log_softmax; backward: subtract 1 at the label, divide by n
    ```

## The exercise

You implement `softmax`, `log_softmax`, and `softmax_cross_entropy`.

The checks run `softmax` on $[2, 1, 0]$ for $[0.665241, 0.244728, 0.090031]$, and confirm that rows add to 1, that a shift changes nothing, and that scores of 1000 stay finite. They check `log_softmax` against the log of `softmax` and at plus or minus 1000, the loss $0.407606$ for the worked example, and the slope $[-0.334759, 0.244728, 0.090031]$ against both finite differences and $(p - \text{onehot}) / n$. They also check that a batch of 6 identical rows has slopes one sixth the size of a single row.
