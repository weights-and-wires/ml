---
title: "Scaling a learning rate with width, and the Muon update"
slug: 074-optimiser-scaling-rules
part: "Part 7 — Large language models"
summary: Why a learning rate tuned on a narrow network fails on a wide one, how muP fixes the rate per layer, and how Muon bounds an update by orthogonalising it.
minutes: 70
gpu: false
objectives:
  - Explain why one Adam step moves a wide layer's activations in proportion to its width, using per-coordinate and spectral views of the update.
  - Apply the muP rules for initialisation and Adam learning rates, and check that an activation's change after one step is flat across widths.
  - Read a learning-rate sweep to see whether the best rate transfers from a narrow proxy to a wide model.
  - Orthogonalise an update with Newton-Schulz iterations, and explain why that bounds its spectral norm.
prereqs:
  - 004-stable-least-squares
  - 038-sgd-adam-and-schedules
  - 039-signal-size-and-initialisation
  - 073-scaling-laws
exercise: 074-optimiser-scaling-rules
---

# Scaling a learning rate with width, and the Muon update

A team tunes the Adam learning rate of a small network. The network has two hidden layers of 32 units, and the best rate is $2^{-6}$. The team then widens the hidden layers to 512 units and keeps the rate. The loss at the end is 0.076, almost twice the 0.040 that the best rate for the wide network reaches, and that best rate is $2^{-9}$, eight times smaller. Nothing is broken. The same rate means something different at a different width.

This matters because the sweep that found the rate is cheap only when the network is small. For a model with billions of parameters, you cannot afford to repeat it at full size. [Scaling laws](/lessons/073-scaling-laws) tell you what size to build, but they say nothing about which rate to train it with. You want a rule that carries a rate from a narrow proxy to a wide model.

This chapter gives two such ideas. The first, muP (the *maximal update parameterisation*), rescales initialisation and per-layer learning rates with width, so that one step changes every layer by the same amount at every width. The second, Muon, changes the update itself: it keeps the direction that momentum chooses and equalises how hard every direction is pushed. Both start from one question: how large is an update, measured the way the next layer feels it?

## What you already know

- Adam divides each coordinate's gradient by a running estimate of its size, so every weight moves by about the learning rate per step, as in [SGD, Adam, and schedules](/lessons/038-sgd-adam-and-schedules).
- Batch size and learning rate co-scale through the gradient noise scale, also covered in that chapter. That is a separate knob from width, and this chapter does not repeat it.
- A weight matrix has singular values, its stretch factors along orthogonal axes, as in [stable least squares](/lessons/004-stable-least-squares).
- A standard initialisation gives weights a standard deviation of $1/\sqrt{\text{fan\_in}}$, so that activations keep a steady size through a layer, as in [signal size and initialisation](/lessons/039-signal-size-and-initialisation).
- A tuning run costs compute, and a larger model costs more, as in [scaling laws](/lessons/073-scaling-laws).

## One step, many small pushes

Take one hidden layer of $n$ units, fed by $n$ activations $\phi_1, \dots, \phi_n$ from the layer before. Unit $i$ computes $h_i = \sum_j W_{ij}\,\phi_j$. Adam's first step sets every weight to move by exactly the learning rate $\eta$, with the sign of its gradient. For one training example, the gradient of $W_{ij}$ is $\delta_i\,\phi_j$, where $\delta_i$ is how much the loss wants unit $i$ to change. So the step is

$$
\Delta W_{ij} = -\eta\,\operatorname{sign}(\delta_i)\,\operatorname{sign}(\phi_j)
$$

and the change in unit $i$ on that same example is

$$
\Delta h_i = \sum_j \Delta W_{ij}\,\phi_j = -\eta\,\operatorname{sign}(\delta_i)\sum_j |\phi_j|.
$$

Read it aloud: every weight in a row moves a little, each move has the sign that agrees with $\phi_j$, so every term in the sum pushes the unit the same way and none cancels. The unit's change is $\eta$ times the total size of its inputs, and that total grows with $n$.

Work it by hand with $\eta = 0.001$ and four activations $(1, 0, 2, 1)$, which sum to 4. A wider layer repeats the pattern, so 16 units give $(1, 0, 2, 1, 1, 0, 2, 1, \dots)$ and a sum of 16. To hold the change fixed, shrink the rate by the ratio of widths, here $4/n$.

| Hidden width $n$ | $\sum_j \lvert\phi_j\rvert$ | Change per unit, same $\eta$ | Change per unit, $\eta \times 4/n$ |
|---|---|---|---|
| 4 | 4 | 0.0040 | 0.0040 |
| 16 | 16 | 0.0160 | 0.0040 |

With one rate, four times the width gives four times the change. The first column is the *per coordinate* view: each weight moved the same amount, and that looked safe. The second is the *per layer* view: how far the layer's output moved, which is what the next layer reads. A learning rate chosen by watching one weight cannot see the sum.

Figure 1 measures this on a real network. It has input size 16, two hidden layers of the width on the axis, and one output. The weights start at the usual $1/\sqrt{\text{fan\_in}}$, and Adam takes one step with $\eta = 0.001$ on a batch of 8 random examples. The plot shows the root mean square (RMS) of the change in the second hidden layer's pre-activations, averaged over 8 seeds.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Five labelled training points and a query at (1, 1) with a circle through its three nearest neighbours">
<rect x="40" y="20" width="480" height="210" class="fill-sunk stroke-grid"/>
<circle cx="145.0" cy="155.0" r="106.1" class="fill-soft-b stroke-b dashed"/>
<circle cx="70.0" cy="210.0" r="7" class="fill-a"/>
<text x="81.0" y="201.0" class="fig-small">#0</text>
<circle cx="145.0" cy="210.0" r="7" class="fill-a"/>
<text x="156.0" y="201.0" class="fig-small">#1</text>
<circle cx="70.0" cy="100.0" r="7" class="fill-c"/>
<text x="81.0" y="91.0" class="fig-small">#2</text>
<circle cx="295.0" cy="45.0" r="7" class="fill-c"/>
<text x="306.0" y="36.0" class="fig-small">#3</text>
<circle cx="370.0" cy="45.0" r="7" class="fill-c"/>
<text x="381.0" y="36.0" class="fig-small">#4</text>
<rect x="138.0" y="148.0" width="14" height="14" class="fill-b"/>
<text x="156.0" y="177.0" class="fig-bold">query</text>
<text x="340" y="192" class="fig-small">blue: class 0, green: class 1</text>
<text x="340" y="210" class="fig-small">circle: k = 3 reach</text></svg>
Figure 1. One Adam step at learning rate 0.001, averaged over 8 seeds. Under standard parameterisation the change grows about tenfold from width 32 to 512. Under muP it does not move.
```

Under the standard parameterisation (SP), the change grows from 0.0110 at width 32 to 0.1402 at width 512, a factor of 12.7. That is close to the factor 16 that "proportional to width" predicts. It falls short because a batch of 8 examples gives gradients whose signs do not all agree, so the pushes cancel a little. The orange curve is the same network with the muP rules of the next sections. It sits at 0.0115 and 0.0112, a ratio of 0.97.

> [!KEY]
> A learning rate fixes how far each weight moves, and the next layer feels the sum over every weight in a row. When the sum has more terms and they agree in sign, the same rate moves a wide layer further. A rule that scales a rate with width has to act on the layer's output, not on a single weight.

## The spectral view of an update

There is a cleaner way to say this, and it is the one Muon uses. A weight update $\Delta W$ is a matrix, and the most it can stretch any input is its *spectral norm* $\lVert\Delta W\rVert_2$, which is its largest singular value. For any activation vector $\phi$ with $n$ entries,

$$
\lVert \Delta W\,\phi \rVert_2 \le \lVert \Delta W \rVert_2\,\lVert \phi \rVert_2 .
$$

Divide both sides by $\sqrt{n}$. The left side becomes the RMS of the change in the layer's output, and $\lVert\phi\rVert_2/\sqrt{n}$ is the RMS of the input, so

$$
\text{RMS}(\Delta h) \le \lVert \Delta W \rVert_2 \cdot \text{RMS}(\phi).
$$

This says that the spectral norm of the update, times the size of a typical input coordinate, caps how far a typical output coordinate moves. To keep that move independent of width, keep $\lVert\Delta W\rVert_2$ independent of width.

Measure it for Adam. With one example, the sign matrix $\operatorname{sign}(\delta)\operatorname{sign}(\phi)^\top$ has rank 1, and a rank-1 matrix of $\pm 1$ entries with $n$ rows and $n$ columns has spectral norm $n$. So an Adam step with rate $\eta$ has spectral norm $\eta\,n$. At $\eta = 0.001$ that is 0.032 at width 32, 0.128 at width 128, and 0.512 at width 512, which matches the growth in Figure 1. If you scale the rate by $32/n$, the norm is 0.032 at every width.

Yang, Simon, and Bernstein (2023) turn this into a rule. For features to keep learning at every width, the spectral norm of a layer's update must be of order $\sqrt{\text{fan\_out}/\text{fan\_in}}$. For a square hidden layer that is 1, so the update's spectral norm must not grow or shrink with $n$, and an Adam rate that scales like $1/n$ does exactly that for the sign matrix above.

## muP: three layers, three rules

muP applies this reasoning layer by layer. Call the layer that reads the input the *input* layer, the layer that makes the output the *output* layer, and everything between them *hidden*. For Adam, the rules, following Yang and colleagues (2022), are:

| Layer | Initial standard deviation | Adam learning rate at width $n$ |
|---|---|---|
| Input (fan-in is the fixed input size) | $1/\sqrt{\text{fan\_in}}$ | $\eta$ |
| Hidden | $1/\sqrt{\text{fan\_in}}$ | $\eta \cdot n_0/n$ |
| Output | $1/\text{fan\_in}$ | $\eta \cdot n_0/n$ |

Here $\eta$ is the rate tuned at the narrow *base width* $n_0$. The input layer's fan-in is the data's dimension, which does not grow with $n$, so its rate stays put. The hidden rate falls as $1/n$ for the reason above. The output layer is the odd one: its init is $1/\text{fan\_in}$, not $1/\sqrt{\text{fan\_in}}$. The output is a sum of $n$ terms, so with the usual init it starts at a size of order 1, and one Adam step moves it by order $\eta\,n$. With init $1/n$ it starts near zero, and with rate $\eta\,n_0/n$ one step moves it by order $\eta$, the same at every width.

Why not the tempting middle rule, $1/\sqrt{n}$, since the init uses $\sqrt{n}$? Because the step's size does not follow the init's size. The step is a sign matrix scaled by the rate, so only $1/n$ cancels the $n$ in its spectral norm. A hidden rate of $\eta\sqrt{n_0/n}$ leaves a factor of $\sqrt{n/n_0}$ in the hidden layer's own push, which is 4 at width 512 against base 32. The earlier layers' pushes do not carry that factor, so in the exercise's network the measured change at width 512 comes out 2.3 times its width-32 value instead of 1, and the activation check fails.

In code, the rates and the init are short:

```python
def adam_lr_for_width(base_lr, base_width, width, layer):
    # "input" keeps base_lr; "hidden" and "output" scale by base_width / width
    ...

def mup_init(widths, rng):
    # std 1/sqrt(fan_in) everywhere except the last matrix, which gets 1/fan_in
    ...
```

The check that you got it right is the one from Figure 1. After one Adam step, the RMS change of a hidden activation must stay within a small ratio as the width grows. It is a necessary check, not a full proof: muP is derived for every step of training, and the paper's *coordinate check* repeats this measurement over several steps.

> [!WARNING]
> The muP rates above are for Adam. A different optimiser makes a different update, so its rule differs. Copying the Adam table into SGD, or the other way, breaks the cancellation.

## Transfer: tune small, train wide

Put the rule to work. The task is a regression: a small, fixed random teacher network produces the targets from inputs drawn from a normal distribution, and a student MLP learns to match it. Figure 2 trains students of width 32, 128, and 512 for 200 Adam steps over a grid of rates from $2^{-12}$ to $2^{-2}$, and plots the final test loss, averaged over 4 seeds. The left panel uses SP with one rate for every layer. The right panel uses muP with base width 32.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Mean ratio of nearest to farthest distance against dimension, rising from near 0 toward 1">
<line x1="70" y1="200" x2="520" y2="200" class="stroke-muted"/>
<line x1="70" y1="200" x2="70" y2="30" class="stroke-muted"/>
<line x1="70" y1="200.0" x2="520" y2="200.0" class="stroke-grid"/>
<text x="62" y="204.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="115.0" x2="520" y2="115.0" class="stroke-grid"/>
<text x="62" y="119.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="30.0" x2="520" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="218" text-anchor="middle" class="fig-small">1</text>
<text x="220.0" y="218" text-anchor="middle" class="fig-small">10</text>
<text x="370.0" y="218" text-anchor="middle" class="fig-small">100</text>
<text x="520.0" y="218" text-anchor="middle" class="fig-small">1000</text>
<polyline points="70.0,199.3 115.2,193.0 174.8,169.9 220.0,143.8 265.2,116.8 324.8,89.6 370.0,72.9 415.2,63.3 474.8,51.5 520.0,45.5" fill="none" class="stroke-b"/>
<circle cx="70.0" cy="199.3" r="4" class="fill-b"/>
<circle cx="115.2" cy="193.0" r="4" class="fill-b"/>
<circle cx="174.8" cy="169.9" r="4" class="fill-b"/>
<circle cx="220.0" cy="143.8" r="4" class="fill-b"/>
<circle cx="265.2" cy="116.8" r="4" class="fill-b"/>
<circle cx="324.8" cy="89.6" r="4" class="fill-b"/>
<circle cx="370.0" cy="72.9" r="4" class="fill-b"/>
<circle cx="415.2" cy="63.3" r="4" class="fill-b"/>
<circle cx="474.8" cy="51.5" r="4" class="fill-b"/>
<circle cx="520.0" cy="45.5" r="4" class="fill-b"/>
<text x="295.0" y="240" text-anchor="middle" class="fig-small">dimension d (log scale)</text>
<text x="78" y="34" class="fig-small">nearest / farthest distance</text></svg>
Figure 2. Final loss against learning rate. The dots mark the best rate. Under standard parameterisation it slides left by a factor of 8 as the width grows, and under muP it stays at 2^-5.
```

Under SP the best rate falls from $2^{-6}$ at width 32 to $2^{-7}$ at width 128 and $2^{-9}$ at width 512, a drop of a factor of 8. Under muP the best rate is $2^{-5}$ at all three widths, so a rate found at width 32 is the right one at width 512. There is also a bonus. The best muP loss falls with width, 0.058, 0.038, and 0.030, so wider is better and you do not pay for the tuning to find that out.

This is *hyperparameter transfer*, the idea of Yang and colleagues' $\mu$Transfer: tune on a narrow proxy, then train the wide model with the same hyperparameters. The paper reports that it holds for Transformers and ResNets, and reports tuning a 13-million-parameter proxy to train a model that beat published BERT-large results. This chapter checks the claim only on tiny MLPs, which is enough to see the mechanism and not enough to promise it for your model.

Two limits matter. The curves are noisy, so the best rate lands on a grid point, and the one-octave steps are coarse. And muP transfers across *width*. Depth, batch size, and the schedule are separate axes, with their own rules. [Scaling laws](/lessons/073-scaling-laws) and the linear batch-size rule cover other axes; muP does not replace them.

> [!KEY]
> Under standard parameterisation the best rate depends on width, so a narrow sweep gives a wrong rate for a wide model. Under muP the best rate is a property of the recipe, so you tune once on the proxy and reuse the result.

## Muon: make every direction move equally

The spectral view suggests a different fix. Instead of scaling a rate so that the update's biggest stretch stays constant, make the update's stretches all equal by construction.

Write the update as its singular value decomposition, $U\,\Sigma\,V^\top$. The gradient of a layer is often close to low rank: a few large singular values carry the step, and the other directions barely move. Muon (Jordan, 2024) keeps the singular vectors and replaces every singular value with 1, giving $U V^\top$, the *orthogonalised* update. Every direction the momentum picked now moves by the same amount, and the spectral norm is 1.

An SVD per step is expensive and awkward on GPUs, so Muon uses the *Newton-Schulz iteration*. It uses only matrix products, and it acts on each singular value separately. If $X = U\Sigma V^\top$, then $XX^\top = U\Sigma^2U^\top$, and the update

$$
X \leftarrow a\,X + b\,(XX^\top)X + c\,(XX^\top)^2X
$$

returns $U\,p(\Sigma)\,V^\top$ with $p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$. The singular vectors stay, and each singular value goes through the polynomial $p$. Repeat it a few times and the values settle near 1.

Work it by hand on $G = \operatorname{diag}(3, 1)$, whose singular values are 3 and 1. First divide by the Frobenius norm, $\sqrt{3^2 + 1^2} = 3.162$, giving $0.949$ and $0.316$. Try the *cubic* iteration ($a = 1.5$, $b = -0.5$, $c = 0$), which has the fixed point at 1.

| Step | Large value | Small value |
|---|---|---|
| 0 | 0.9487 | 0.3162 |
| 1 | 0.9961 | 0.4585 |
| 2 | 1.0000 | 0.6396 |
| 3 | 1.0000 | 0.8286 |

The large value arrives in two steps. The small one is slow, because for small $\sigma$ the cubic gains only a factor of 1.5 per step. Jordan tuned the quintic coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ to gain 3.44 per step at small $\sigma$, at the price of not converging exactly to 1. Five steps on the same two values give

| Step | Large value | Small value |
|---|---|---|
| 1 | 0.7518 | 0.9447 |
| 2 | 1.0484 | 0.7568 |
| 3 | 0.6819 | 1.0414 |
| 4 | 1.1343 | 0.6825 |
| 5 | 0.7530 | 1.1337 |

The values bounce inside a band instead of settling. Jordan's write-up sets the target at $[0.7, 1.3]$ and finds that harmless for training. Figure 3 plots what five steps do to every input value in $[0, 1]$: for inputs of 0.01 and above, the output lies in $[0.68, 1.13]$.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Singular value out against singular value in for one quintic Newton-Schulz step, five quintic steps, and five cubic steps. Five quintic steps put every input above 0.01 inside a band near 1."><line x1="60" y1="210" x2="540" y2="210" class="stroke-muted"/><line x1="60" y1="40" x2="60" y2="210" class="stroke-muted"/><rect x="60" y="52.1" width="480" height="72.9" class="fill-soft-c"/><line x1="60.0" y1="40" x2="60.0" y2="210" class="stroke-grid"/><text x="60.0" y="226" text-anchor="middle" class="fig-small">0</text><line x1="180.0" y1="40" x2="180.0" y2="210" class="stroke-grid"/><text x="180.0" y="226" text-anchor="middle" class="fig-small">0.25</text><line x1="300.0" y1="40" x2="300.0" y2="210" class="stroke-grid"/><text x="300.0" y="226" text-anchor="middle" class="fig-small">0.5</text><line x1="420.0" y1="40" x2="420.0" y2="210" class="stroke-grid"/><text x="420.0" y="226" text-anchor="middle" class="fig-small">0.75</text><line x1="540.0" y1="40" x2="540.0" y2="210" class="stroke-grid"/><text x="540.0" y="226" text-anchor="middle" class="fig-small">1</text><line x1="60" y1="210.0" x2="540" y2="210.0" class="stroke-grid"/><text x="54" y="214.0" text-anchor="end" class="fig-small">0</text><line x1="60" y1="149.3" x2="540" y2="149.3" class="stroke-grid"/><text x="54" y="153.3" text-anchor="end" class="fig-small">0.5</text><line x1="60" y1="88.6" x2="540" y2="88.6" class="stroke-grid"/><text x="54" y="92.6" text-anchor="end" class="fig-small">1</text><polyline points="60.0,210.0 64.8,205.8 69.6,201.6 74.4,197.5 79.2,193.3 84.0,189.2 88.8,185.0 93.6,180.9 98.4,176.8 103.2,172.8 108.0,168.8 112.8,164.8 117.6,160.8 122.4,156.9 127.2,153.0 132.0,149.2 136.8,145.4 141.6,141.7 146.4,138.0 151.2,134.4 156.0,130.9 160.8,127.4 165.6,124.0 170.4,120.7 175.2,117.4 180.0,114.3 184.8,111.2 189.6,108.1 194.4,105.2 199.2,102.3 204.0,99.6 208.8,96.9 213.6,94.3 218.4,91.8 223.2,89.5 228.0,87.2 232.8,85.0 237.6,82.9 242.4,80.9 247.2,79.0 252.0,77.3 256.8,75.6 261.6,74.1 266.4,72.6 271.2,71.3 276.0,70.1 280.8,69.0 285.6,68.0 290.4,67.1 295.2,66.3 300.0,65.6 304.8,65.1 309.6,64.7 314.4,64.3 319.2,64.1 324.0,64.0 328.8,64.0 333.6,64.1 338.4,64.3 343.2,64.7 348.0,65.1 352.8,65.6 357.6,66.3 362.4,67.0 367.2,67.8 372.0,68.7 376.8,69.8 381.6,70.8 386.4,72.0 391.2,73.3 396.0,74.6 400.8,76.1 405.6,77.5 410.4,79.1 415.2,80.7 420.0,82.4 424.8,84.1 429.6,85.9 434.4,87.7 439.2,89.5 444.0,91.4 448.8,93.3 453.6,95.3 458.4,97.2 463.2,99.2 468.0,101.1 472.8,103.0 477.6,105.0 482.4,106.9 487.2,108.8 492.0,110.6 496.8,112.4 501.6,114.1 506.4,115.8 511.2,117.4 516.0,118.9 520.8,120.3 525.6,121.6 530.4,122.8 535.2,123.9 540.0,124.9" fill="none" class="stroke-muted dashed" stroke-width="2"/><polyline points="60.0,210.0 64.8,200.8 69.6,191.7 74.4,182.7 79.2,174.0 84.0,165.6 88.8,157.5 93.6,149.9 98.4,142.8 103.2,136.1 108.0,130.0 112.8,124.4 117.6,119.4 122.4,114.8 127.2,110.8 132.0,107.2 136.8,104.1 141.6,101.4 146.4,99.1 151.2,97.1 156.0,95.5 160.8,94.1 165.6,93.0 170.4,92.0 175.2,91.3 180.0,90.7 184.8,90.2 189.6,89.8 194.4,89.5 199.2,89.3 204.0,89.1 208.8,88.9 213.6,88.8 218.4,88.8 223.2,88.7 228.0,88.7 232.8,88.6 237.6,88.6 242.4,88.6 247.2,88.6 252.0,88.6 256.8,88.6 261.6,88.6 266.4,88.6 271.2,88.6 276.0,88.6 280.8,88.6 285.6,88.6 290.4,88.6 295.2,88.6 300.0,88.6 304.8,88.6 309.6,88.6 314.4,88.6 319.2,88.6 324.0,88.6 328.8,88.6 333.6,88.6 338.4,88.6 343.2,88.6 348.0,88.6 352.8,88.6 357.6,88.6 362.4,88.6 367.2,88.6 372.0,88.6 376.8,88.6 381.6,88.6 386.4,88.6 391.2,88.6 396.0,88.6 400.8,88.6 405.6,88.6 410.4,88.6 415.2,88.6 420.0,88.6 424.8,88.6 429.6,88.6 434.4,88.6 439.2,88.6 444.0,88.6 448.8,88.6 453.6,88.6 458.4,88.6 463.2,88.6 468.0,88.6 472.8,88.6 477.6,88.6 482.4,88.6 487.2,88.6 492.0,88.6 496.8,88.6 501.6,88.6 506.4,88.6 511.2,88.6 516.0,88.6 520.8,88.6 525.6,88.6 530.4,88.6 535.2,88.6 540.0,88.6" fill="none" class="stroke-a" stroke-width="2"/><polyline points="60.0,210.0 64.8,125.1 69.6,127.2 74.4,73.7 79.2,94.5 84.0,117.9 88.8,87.9 93.6,72.5 98.4,94.4 103.2,124.9 108.0,123.5 112.8,118.6 117.6,123.2 122.4,126.9 127.2,114.6 132.0,95.8 136.8,85.2 141.6,83.4 146.4,89.0 151.2,102.6 156.0,119.3 160.8,127.1 165.6,124.1 170.4,119.4 175.2,119.1 180.0,123.2 184.8,127.1 189.6,122.9 194.4,108.3 199.2,90.9 204.0,78.9 208.8,73.4 213.6,72.3 218.4,73.7 223.2,76.5 228.0,79.5 232.8,81.8 237.6,82.9 242.4,82.4 247.2,80.7 252.0,78.1 256.8,75.4 261.6,73.3 266.4,72.3 271.2,72.9 276.0,75.7 280.8,81.3 285.6,89.5 290.4,99.4 295.2,109.2 300.0,117.1 304.8,122.3 309.6,125.3 314.4,126.6 319.2,127.1 324.0,127.2 328.8,127.2 333.6,127.0 338.4,126.5 343.2,125.2 348.0,122.2 352.8,117.1 357.6,109.6 362.4,100.4 367.2,90.9 372.0,82.8 376.8,77.0 381.6,73.6 386.4,72.3 391.2,72.6 396.0,74.0 400.8,76.1 405.6,78.6 410.4,80.8 415.2,82.3 420.0,82.9 424.8,82.4 429.6,81.0 434.4,78.8 439.2,76.3 444.0,74.1 448.8,72.6 453.6,72.3 458.4,73.8 463.2,77.7 468.0,84.7 472.8,95.0 477.6,107.0 482.4,117.7 487.2,124.8 492.0,127.2 496.8,126.1 501.6,123.5 506.4,120.8 511.2,119.1 516.0,118.6 520.8,119.1 525.6,120.4 530.4,122.2 535.2,124.0 540.0,125.4" fill="none" class="stroke-b" stroke-width="2"/><line x1="324.0" y1="159.0" x2="352.0" y2="159.0" class="stroke-b" stroke-width="2"/><text x="360.0" y="163.0" class="fig-small">tuned quintic, 5 steps</text><line x1="324.0" y1="179.0" x2="352.0" y2="179.0" class="stroke-a" stroke-width="2"/><text x="360.0" y="183.0" class="fig-small">cubic, 5 steps</text><line x1="324.0" y1="199.0" x2="352.0" y2="199.0" class="stroke-muted dashed" stroke-width="2"/><text x="360.0" y="203.0" class="fig-small">tuned quintic, 1 step</text><text x="300" y="252" text-anchor="middle" class="fig-small">singular value going in (after dividing by the Frobenius norm)</text><text x="12" y="30" class="fig-small">singular value coming out</text></svg>
Figure 3. Output singular value against input singular value. One quintic step is far from 1, and five steps leave every input from 0.01 up inside a band between 0.68 and 1.13. Five cubic steps are still short of 1 for small inputs.
```

The division by the Frobenius norm is not optional. The Frobenius norm is at least the largest singular value, so after dividing, every singular value sits in $[0, 1]$, which is where $p$ is tuned to work. Without it, take the cubic and a singular value of 3: the step gives $-9$, then $351$, then $-21{,}621{,}249$. The iteration diverges. In numpy:

```python
X = G / (np.linalg.norm(G) + 1e-7)
if X.shape[0] > X.shape[1]:
    X = X.T          # work on the short side; transpose back at the end
for _ in range(steps):
    S = X @ X.T
    X = a * X + b * (S @ X) + c * (S @ S @ X)
```

Small singular values are the weak spot. At 0.01 five quintic steps lift the value to 0.70, but at 0.001 only to 0.47, so an update with directions below about $10^{-3}$ of its Frobenius norm keeps them under-weighted.

Muon turns this into an optimiser. It applies momentum first, then orthogonalises:

$$
M \leftarrow \beta M + G,\qquad W \leftarrow W - \eta\,\text{NS}(M)
$$

where $G$ is the layer's gradient, $\beta$ is the momentum coefficient, and NS is Newton-Schulz. Jordan's public code defaults to a Nesterov variant of the momentum; the exercise uses the plain one above.

Why does this bound the update? The step is $\eta\,\text{NS}(M)$, whose singular values are all at most about 1.13, so $\lVert\Delta W\rVert_2 \le 1.13\,\eta$ whatever the size of $G$. Plain SGD gives $\eta\lVert G\rVert_2$, which grows with the gradient. Adam's sign step on a rank-1 gradient gives $\eta\,n$. A bounded spectral norm is the quantity the previous section said must stay under control, so a constant $\eta$ works across widths for a square layer. For a layer with unequal sizes, the spectral condition asks for a factor of $\sqrt{\text{fan\_out}/\text{fan\_in}}$ on the step.

The price is compute: for an $n \times m$ matrix with $n \le m$, one iteration costs about $6n^2m + 2n^3$ flops, five times per step. Compared with the forward and backward passes, that is small for a wide layer, and large for a tiny one.

> [!KEY]
> Adam bounds each coordinate's step. Muon bounds the update's spectral norm and puts every direction near the same size. Both are ways to keep the output change of a layer under control, and muP's learning-rate rule does it with a width-dependent factor while Muon does it by construction.

> [!RECAP]
> - One Adam step moves a layer's output in proportion to the sum over its row of weights, so the same rate moves a wider layer further.
> - The spectral norm of the update caps the output change, and an Adam step on a rank-1 gradient has spectral norm $\eta\,n$.
> - muP, for Adam, keeps the input rate, scales hidden and output rates by $n_0/n$, and initialises the output with $1/\text{fan\_in}$.
> - A muP network has the same best rate at every width, so you tune a narrow proxy once.
> - Muon orthogonalises momentum with Newton-Schulz: divide by the Frobenius norm, iterate a polynomial, and every singular value lands near 1.

## Build it step by step

You write six functions, and the checks call them in this order. The helpers `sp_init`, `loss_and_grads`, `adam_step`, and `train_loss` are written for you. `train_loss` calls your `mup_init` and `adam_lr_for_width`, so those two come first.

1. `adam_lr_for_width(base_lr, base_width, width, layer)` returns a float. For `"input"` return `base_lr`, and for `"hidden"` or `"output"` return `base_lr * base_width / width`. Raise `ValueError` for any other name. The usual mistake is a square root, which looks natural next to the init.
2. `mup_init(widths, rng)` returns a list of matrices, the $i$-th of shape `(widths[i+1], widths[i])`. Draw them in order, with standard deviation `1 / np.sqrt(fan_in)` for all but the last, and `1 / fan_in` for the last. The usual mistake is to use the square root on the output layer too.
3. `activation_update_size(width, param, seed=0, lr=1e-3)` builds the network, takes one Adam step through `adam_step` with zero moments, and returns the RMS change of `pre[1]`, the second hidden layer's pre-activation. Draw `x` and then `y` from the same generator that built the weights. At width 32 expect about 0.011 for either `param`.
4. `best_lr(width, param, lr_grid, seeds)` averages `train_loss` over the seeds for each rate and returns the rate with the lowest average. Replace a non-finite loss with `1e9` first, or one diverged run decides the answer.
5. `newton_schulz(G, steps=5)` divides by `np.linalg.norm(G) + 1e-7`, transposes a tall matrix, runs the quintic loop with the coefficients in `NS_COEFFS`, and transposes back. For `diag(3, 1)` expect 0.7530 and 1.1337 after 5 steps. The usual mistake is to forget the division, which turns a large `G` into `nan`.
6. `muon_step(W, G, M, lr, beta=0.95)` returns `(W_new, M_new)` with `M_new = beta * M + G` and `W_new = W - lr * newton_schulz(M_new)`. The usual mistake is to orthogonalise `G` and apply momentum afterwards.

> [!TIP]
> Test `newton_schulz` on a matrix you can read, such as `np.diag([3.0, 1.0])`. The off-diagonal entries must stay at zero, and the diagonal must match the table above.

## Where this breaks

muP holds under assumptions. It is derived for a particular family of layers and for width going to infinity, and a real architecture adds parts, such as attention logits, normalisation, and embeddings, that each need their own scaling. If transfer fails, a likely cause is one such part left on the standard rule. Run the coordinate check at a few widths before you trust a transfer.

The rates are also only one of several hyperparameters that must stay put. A weight decay, a warmup length, or a batch size that changes with the width breaks the comparison between the proxy and the wide model.

Muon has its own limits. Orthogonalising treats every direction equally, including directions that are mostly noise, and a matrix with tiny singular values is only partly corrected within five steps. Muon is defined for matrices, and Jordan's write-up is about hidden layers, so biases and norm gains need a different optimiser.

## The exercise

You implement `adam_lr_for_width`, `mup_init`, `activation_update_size`, `best_lr`, `newton_schulz`, and `muon_step`. Eight checks verify them. The first compares the muP rates with exact values, including the case where a 1/sqrt rule would differ. The second checks the shapes and the standard deviations of `mup_init`. Two checks run `activation_update_size` at widths 32, 128, and 512: it must stay within a factor of 2 under muP, and grow by a factor between 8 and 20 under SP. A fifth check runs a learning-rate sweep and requires that the best muP rate moves by at most a factor of 2 from width 32 to 512, while the best SP rate falls by at least a factor of 4. Two checks cover `newton_schulz`: the singular values of a tall, a wide, and a hand-made matrix land in a band near 1, and scaling `G` by $10^{-3}$ or $10^{4}$ changes nothing. The last check confirms that `muon_step` accumulates momentum and bounds the update's spectral norm by about the learning rate, even when the gradient is $10^4$ times larger.

## Further reading

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) (Yang, Hu, Babuschkin, and others, 2022) introduces muP and $\mu$Transfer, and gives the per-layer rules.
- [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813) (Yang, Simon, and Bernstein, 2023) derives the $\sqrt{\text{fan\_out}/\text{fan\_in}}$ rule on update norms and muP from it.
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (Keller Jordan, 2024) gives the quintic coefficients, the five-step recipe, and the momentum handling.
- [Automatic Gradient Descent: Deep Learning without Hyperparameters](https://arxiv.org/abs/2304.05187) (Bernstein, Mingard, Huang, Azizan, and Yue, 2023) builds an optimiser from norms of the layers.
- [Stanford CS336, Spring 2025](http://cs336.stanford.edu/spring2025/) lists lecture 3, "Architectures, hyperparameters", and lectures 9 and 11, both titled "Scaling laws". The schedule page does not say whether they cover muP, so read the lecture materials to find out.
