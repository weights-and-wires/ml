---
title: Vanishing gradients, exploding gradients, and clipping
slug: 052-vanishing-gradients-and-clipping
part: "Part 5 — Vision and sequences"
summary: A gradient across T steps shrinks or grows like a power set by the spectral radius. Truncate the backward pass and clip by global norm to tame it.
minutes: 70
gpu: false
objectives:
  - Explain why a gradient that crosses many steps behaves like compound interest.
  - Compute the spectral radius, including for complex eigenvalues, and predict a gradient's size after T steps with (kappa times rho) to the power T.
  - Write truncated backpropagation through time, and explain what it saves and what it costs.
  - Clip a gradient list by its global norm, and explain why one shared factor keeps the direction.
prereqs:
  - 029-power-iteration-and-probabilistic-pca
  - 047-residual-networks
  - 051-recurrent-cells-and-bptt
exercise: 052-vanishing-gradients-and-clipping
---

# Vanishing gradients, exploding gradients, and clipping

You train the sensor model on runs of 50 readings. The label depends on the first reading: a spike at step 1 means the machine will overheat at step 50. The model learns to use the last few readings well, and it ignores the spike at step 1, even though the spike decides the answer. Training for longer does not help. Where did the learning signal go?

The weights learn from the gradient that reaches them. The gradient for step 1 has to travel back through 49 cells, and [backpropagation through time](/lessons/051-recurrent-cells-and-bptt) multiplies it at every cell. A product of 49 numbers can be extremely small, so the weights hear almost nothing about step 1 even though the forward pass carries the information fine. It can also be extremely large, which breaks training in a louder way: one huge gradient sends the weights to `nan`. Longer sequences add a third problem, because the backward pass needs every state the forward pass produced, so memory grows with the length.

This chapter explains the product with one picture and one rule, measures it with the backward pass you wrote, and then gives the two standard remedies. Truncation cuts the backward pass into windows to save memory. Clipping caps the length of a gradient so that an explosion cannot wreck the weights.

## What you already know

- In a backward pass through time, the carry is multiplied by a tanh slope and by $W_{hh}$ at every step, and the backward pass needs every state from the forward pass.
- The tanh slope $1 - h^2$ is between 0 and 1.
- An eigenvector of a square matrix $W$ is a direction $v$ that $W$ only stretches: $Wv = \lambda v$ for a number $\lambda$, its eigenvalue. Repeating a multiplication by a matrix, as in power iteration, lines a vector up with one direction.
- In a deep stack of plain layers, many factors below one shrink the gradient toward zero.
- A gradient over many weights can be treated as one long vector, and its norm is its length: the square root of the sum of the squares of its entries.

## One factor per step

Start with the simplest case, where the state is a single number and the weight is a single number $w$. Ignore the tanh slope for a moment. Each step back multiplies the carry by $w$, so after $T$ steps the gradient is $w^T$ times its starting size. That is compound interest. With $w = 0.9$, thirty steps leave $0.9^{30} = 0.042$ of the signal. With $w = 1.1$, thirty steps give $1.1^{30} = 17.4$. A weight 10% away from one, in either direction, is enough to wreck a long chain.

Now put the tanh slope back in. In the running example of the last chapter, each step back multiplied the gradient by $0.8$ for $W_{hh}$ and by a slope near $0.6$. That is about $0.48$ per step, so a gradient 30 steps back is $0.48^{30} \approx 3 \times 10^{-10}$ of the gradient at the end. The step-1 spike in the opening is out of reach.

For a real cell the state is a vector and $W_{hh}$ is a matrix, so the question becomes what repeated multiplication by a matrix does. Figure 1 gives the answer first. It plots the size of the gradient that reaches a state $t$ steps back, for three recurrent matrices that differ only in their spectral radius $\rho$, a number defined in the next section. The lines are a rule, and the dots are gradients measured with `bptt`.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Gradient size against steps back on a log axis: lines for rho 0.9, 1.0, and 1.05 with measured dots on top">
<defs></defs>
<line x1="80" y1="260" x2="520" y2="260" class="stroke-grid"/>
<text x="72" y="264" text-anchor="end" class="fig-small">1e-5</text>
<line x1="80" y1="233.8" x2="520" y2="233.8" class="stroke-grid"/>
<line x1="80" y1="207.5" x2="520" y2="207.5" class="stroke-grid"/>
<text x="72" y="211.5" text-anchor="end" class="fig-small">1e-3</text>
<line x1="80" y1="181.2" x2="520" y2="181.2" class="stroke-grid"/>
<line x1="80" y1="155" x2="520" y2="155" class="stroke-grid"/>
<text x="72" y="159" text-anchor="end" class="fig-small">1e-1</text>
<line x1="80" y1="128.8" x2="520" y2="128.8" class="stroke-grid"/>
<line x1="80" y1="102.5" x2="520" y2="102.5" class="stroke-grid"/>
<text x="72" y="106.5" text-anchor="end" class="fig-small">1e1</text>
<line x1="80" y1="76.2" x2="520" y2="76.2" class="stroke-grid"/>
<line x1="80" y1="50" x2="520" y2="50" class="stroke-grid"/>
<text x="72" y="54" text-anchor="end" class="fig-small">1e3</text>
<text x="72" y="132.8" text-anchor="end" class="fig-small">1</text>
<text x="80" y="282" text-anchor="middle" class="fig-small">0</text>
<text x="168" y="282" text-anchor="middle" class="fig-small">20</text>
<text x="256" y="282" text-anchor="middle" class="fig-small">40</text>
<text x="344" y="282" text-anchor="middle" class="fig-small">60</text>
<text x="432" y="282" text-anchor="middle" class="fig-small">80</text>
<text x="520" y="282" text-anchor="middle" class="fig-small">100</text>
<text x="300" y="302" text-anchor="middle">steps back</text>
<text x="20" y="28" class="fig-bold">gradient size relative to the last step</text>
<polyline points="80,128.8 84.4,130 88.8,131.2 93.2,132.4 97.6,133.6 102,134.8 106.4,136 110.8,137.2 115.2,138.4 119.6,139.6 124,140.8 128.4,142 132.8,143.2 137.2,144.4 141.6,145.6 146,146.8 150.4,148 154.8,149.2 159.2,150.4 163.6,151.6 168,152.8 172.4,154 176.8,155.2 181.2,156.4 185.6,157.6 190,158.8 194.4,160 198.8,161.2 203.2,162.4 207.6,163.6 212,164.8 216.4,166 220.8,167.2 225.2,168.4 229.6,169.6 234,170.8 238.4,172 242.8,173.2 247.2,174.4 251.6,175.6 256,176.8 260.4,178 264.8,179.2 269.2,180.4 273.6,181.6 278,182.8 282.4,184 286.8,185.2 291.2,186.4 295.6,187.6 300,188.8 304.4,190 308.8,191.2 313.2,192.4 317.6,193.6 322,194.8 326.4,196 330.8,197.2 335.2,198.4 339.6,199.6 344,200.8 348.4,202 352.8,203.2 357.2,204.4 361.6,205.6 366,206.8 370.4,208 374.8,209.2 379.2,210.4 383.6,211.6 388,212.8 392.4,214 396.8,215.2 401.2,216.4 405.6,217.6 410,218.8 414.4,220 418.8,221.2 423.2,222.4 427.6,223.6 432,224.8 436.4,226 440.8,227.2 445.2,228.4 449.6,229.6 454,230.8 458.4,232 462.8,233.2 467.2,234.4 471.6,235.7 476,236.9 480.4,238.1 484.8,239.3 489.2,240.5 493.6,241.7 498,242.9 502.4,244.1 506.8,245.3 511.2,246.5 515.6,247.7 520,248.9" class="stroke-a"/>
<circle cx="124" cy="140.8" r="3.5" class="fill-a"/>
<circle cx="168" cy="152.8" r="3.5" class="fill-a"/>
<circle cx="212" cy="164.8" r="3.5" class="fill-a"/>
<circle cx="256" cy="176.8" r="3.5" class="fill-a"/>
<circle cx="300" cy="188.9" r="3.5" class="fill-a"/>
<circle cx="344" cy="200.9" r="3.5" class="fill-a"/>
<circle cx="388" cy="212.9" r="3.5" class="fill-a"/>
<circle cx="432" cy="224.9" r="3.5" class="fill-a"/>
<circle cx="476" cy="236.9" r="3.5" class="fill-a"/>
<circle cx="520" cy="248.9" r="3.5" class="fill-a"/>
<text x="528" y="252.9" class="fig-small">ρ = 0.9</text>
<polyline points="80,128.8 84.4,128.8 88.8,128.8 93.2,128.8 97.6,128.8 102,128.8 106.4,128.8 110.8,128.8 115.2,128.8 119.6,128.8 124,128.8 128.4,128.8 132.8,128.8 137.2,128.8 141.6,128.8 146,128.8 150.4,128.8 154.8,128.8 159.2,128.8 163.6,128.8 168,128.8 172.4,128.8 176.8,128.8 181.2,128.8 185.6,128.8 190,128.8 194.4,128.8 198.8,128.8 203.2,128.8 207.6,128.8 212,128.8 216.4,128.8 220.8,128.8 225.2,128.8 229.6,128.8 234,128.8 238.4,128.8 242.8,128.8 247.2,128.8 251.6,128.8 256,128.8 260.4,128.8 264.8,128.8 269.2,128.8 273.6,128.8 278,128.8 282.4,128.8 286.8,128.8 291.2,128.8 295.6,128.8 300,128.8 304.4,128.8 308.8,128.8 313.2,128.8 317.6,128.8 322,128.8 326.4,128.8 330.8,128.8 335.2,128.8 339.6,128.8 344,128.8 348.4,128.8 352.8,128.8 357.2,128.8 361.6,128.8 366,128.8 370.4,128.8 374.8,128.8 379.2,128.8 383.6,128.8 388,128.8 392.4,128.8 396.8,128.8 401.2,128.8 405.6,128.8 410,128.8 414.4,128.8 418.8,128.8 423.2,128.8 427.6,128.8 432,128.8 436.4,128.8 440.8,128.8 445.2,128.8 449.6,128.8 454,128.8 458.4,128.8 462.8,128.8 467.2,128.8 471.6,128.8 476,128.8 480.4,128.8 484.8,128.8 489.2,128.8 493.6,128.8 498,128.8 502.4,128.8 506.8,128.8 511.2,128.8 515.6,128.8 520,128.8" class="stroke-muted"/>
<circle cx="124" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="168" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="212" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="256" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="300" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="344" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="388" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="432" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="476" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="520" cy="128.8" r="3.5" class="fill-ink"/>
<text x="528" y="132.8" class="fig-small">ρ = 1</text>
<polyline points="80,128.8 84.4,128.2 88.8,127.6 93.2,127.1 97.6,126.5 102,126 106.4,125.4 110.8,124.9 115.2,124.3 119.6,123.7 124,123.2 128.4,122.6 132.8,122.1 137.2,121.5 141.6,121 146,120.4 150.4,119.9 154.8,119.3 159.2,118.7 163.6,118.2 168,117.6 172.4,117.1 176.8,116.5 181.2,116 185.6,115.4 190,114.8 194.4,114.3 198.8,113.7 203.2,113.2 207.6,112.6 212,112.1 216.4,111.5 220.8,111 225.2,110.4 229.6,109.8 234,109.3 238.4,108.7 242.8,108.2 247.2,107.6 251.6,107.1 256,106.5 260.4,105.9 264.8,105.4 269.2,104.8 273.6,104.3 278,103.7 282.4,103.2 286.8,102.6 291.2,102.1 295.6,101.5 300,100.9 304.4,100.4 308.8,99.8 313.2,99.3 317.6,98.7 322,98.2 326.4,97.6 330.8,97 335.2,96.5 339.6,95.9 344,95.4 348.4,94.8 352.8,94.3 357.2,93.7 361.6,93.2 366,92.6 370.4,92 374.8,91.5 379.2,90.9 383.6,90.4 388,89.8 392.4,89.3 396.8,88.7 401.2,88.1 405.6,87.6 410,87 414.4,86.5 418.8,85.9 423.2,85.4 427.6,84.8 432,84.3 436.4,83.7 440.8,83.1 445.2,82.6 449.6,82 454,81.5 458.4,80.9 462.8,80.4 467.2,79.8 471.6,79.2 476,78.7 480.4,78.1 484.8,77.6 489.2,77 493.6,76.5 498,75.9 502.4,75.4 506.8,74.8 511.2,74.2 515.6,73.7 520,73.1" class="stroke-b"/>
<circle cx="124" cy="123.2" r="3.5" class="fill-b"/>
<circle cx="168" cy="117.7" r="3.5" class="fill-b"/>
<circle cx="212" cy="112.1" r="3.5" class="fill-b"/>
<circle cx="256" cy="106.6" r="3.5" class="fill-b"/>
<circle cx="300" cy="101" r="3.5" class="fill-b"/>
<circle cx="344" cy="95.5" r="3.5" class="fill-b"/>
<circle cx="388" cy="89.9" r="3.5" class="fill-b"/>
<circle cx="432" cy="84.5" r="3.5" class="fill-b"/>
<circle cx="476" cy="79.3" r="3.5" class="fill-b"/>
<circle cx="520" cy="74.5" r="3.5" class="fill-b"/>
<text x="528" y="77.1" class="fig-small">ρ = 1.05</text>
</svg>
Figure 1. Lines are the prediction ρ^t and dots are gradients measured with the `bptt` you wrote earlier, on matrices with that spectral radius. Notice that the dots sit on or just under the lines (tanh flattens a little once the state has grown): after 100 steps the ρ = 0.9 signal is down to 0.00003 and the ρ = 1.05 signal is up to about 117, against a prediction of 131.5.
```

The $\rho = 0.9$ line dives, the $\rho = 1$ line stays flat, and the $\rho = 1.05$ line climbs, and the measured dots lie on the lines.

## Why an eigenvalue appears

Take the matrix $W = \mathrm{diag}(0.5, 0.9)$ and the vector $(1, 1)$. Multiplying by $W$ halves the first entry and takes 90% of the second, so after $t$ steps the vector is $(0.5^t, 0.9^t)$. After five steps that is $(0.031, 0.590)$. Figure 2 draws this.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Repeatedly applying diag(0.5, 0.9) to the vector (1, 1) squeezes it toward the 0.9 direction">
<defs></defs>
<text x="20" y="28" class="fig-bold">Repeated multiplication by W = diag(0.5, 0.9)</text>
<line x1="70" y1="250" x2="270" y2="250" class="stroke-muted"/>
<line x1="70" y1="250" x2="70" y2="50" class="stroke-muted"/>
<text x="272" y="266" text-anchor="end" class="fig-small">0.5 direction</text>
<text x="78" y="48" class="fig-small">0.9 direction</text>
<polyline points="260,60 165,79 117.5,96.1 93.8,111.5 81.9,125.3 75.9,137.8" class="stroke-b"/>
<circle cx="260" cy="60" r="4" class="fill-b"/>
<circle cx="165" cy="79" r="4" class="fill-b"/>
<circle cx="117.5" cy="96.1" r="4" class="fill-b"/>
<circle cx="93.8" cy="111.5" r="4" class="fill-b"/>
<circle cx="81.9" cy="125.3" r="4" class="fill-b"/>
<circle cx="75.9" cy="137.8" r="4" class="fill-b"/>
<text x="264" y="48" text-anchor="end" class="fig-small">t=0: (1, 1)</text>
<text x="159" y="67" text-anchor="middle" class="fig-small">t=1: (0.5, 0.9)</text>
<text x="125.5" y="112.1" class="fig-small">t=2</text>
<text x="101.8" y="127.5" class="fig-small">t=3</text>
<text x="89.9" y="143.3" class="fig-small">t=4</text>
<text x="83.9" y="159.8" class="fig-small">t=5</text>
<text x="460" y="28" text-anchor="middle" class="fig-bold">Size of each component</text>
<line x1="355" y1="250" x2="590" y2="250" class="stroke-muted"/>
<rect x="360" y="80" width="14" height="170" class="fill-a"/>
<rect x="376" y="80" width="14" height="170" class="fill-b"/>
<text x="375" y="266" text-anchor="middle" class="fig-small">0</text>
<rect x="398" y="165" width="14" height="85" class="fill-a"/>
<rect x="414" y="97" width="14" height="153" class="fill-b"/>
<text x="413" y="266" text-anchor="middle" class="fig-small">1</text>
<rect x="436" y="207.5" width="14" height="42.5" class="fill-a"/>
<rect x="452" y="112.3" width="14" height="137.7" class="fill-b"/>
<text x="451" y="266" text-anchor="middle" class="fig-small">2</text>
<rect x="474" y="228.8" width="14" height="21.2" class="fill-a"/>
<rect x="490" y="126.1" width="14" height="123.9" class="fill-b"/>
<text x="489" y="266" text-anchor="middle" class="fig-small">3</text>
<rect x="512" y="239.4" width="14" height="10.6" class="fill-a"/>
<rect x="528" y="138.5" width="14" height="111.5" class="fill-b"/>
<text x="527" y="266" text-anchor="middle" class="fig-small">4</text>
<rect x="550" y="244.7" width="14" height="5.3" class="fill-a"/>
<rect x="566" y="149.6" width="14" height="100.4" class="fill-b"/>
<text x="565" y="266" text-anchor="middle" class="fig-small">5</text>
<text x="475" y="286" text-anchor="middle" class="fig-small">steps t</text>
<rect x="470" y="44" width="12" height="12" class="fill-a"/>
<text x="486" y="55" class="fig-small">0.5 direction</text>
<rect x="470" y="62" width="12" height="12" class="fill-b"/>
<text x="486" y="73" class="fig-small">0.9 direction</text>
</svg>
Figure 2. Each multiplication by W keeps 0.9 of the vertical part and only 0.5 of the horizontal part. After five steps the horizontal part is 0.03 and the vertical part is 0.59, so the direction that survives longest, the one with the largest eigenvalue, decides what is left.
```

The reason is that along an eigenvector, multiplying by $W$ is the same as multiplying by a number. If $Wv = \lambda v$, then $W^2 v = \lambda W v = \lambda^2 v$, and $W^T v = \lambda^T v$. A diagonal matrix has the axes as its eigenvectors, so each axis shrinks by its own power. A general vector is a mix of eigenvectors, so after many steps the part with the largest $|\lambda|$ outgrows the rest. One number, the largest absolute eigenvalue, therefore tells you the growth rate. It is called the **spectral radius**:

$$
\rho(W) = \max_i |\lambda_i|
$$

Read it aloud: take every eigenvalue, measure its distance from zero, and keep the biggest. Here $\rho = 0.9$. The distance matters because the eigenvalues of a real matrix can be complex. A rotation by 90 degrees has no real eigenvalue. Its eigenvalues are $i$ and $-i$, and each has absolute value 1, which is right, because a rotation neither shrinks nor stretches a vector.

```python
rho = np.max(np.abs(np.linalg.eigvals(W)))
```

The usual mistake is leaving out `np.abs`, so a large negative or complex eigenvalue is missed.

## The rule, and when to trust it

Each step back multiplies the gradient by the tanh slope, which we summarise as one number $\kappa$ between 0 and 1, and by the matrix, which acts like $\rho$. After $T$ steps the gradient has size about

$$
\left(\kappa\,\rho(W_{hh})\right)^T
$$

Read it aloud: multiply the typical slope by the spectral radius to get a rate per step, then raise the rate to the number of steps. If the rate is below one, the gradient vanishes. If it is above one, the gradient can explode.

The rule is exact when $W_{hh}$ is a rotation scaled by $\rho$, as in Figure 1, and when the slope is the same at every step. The dots land on the lines because the state in that run stays tiny, so $\kappa$ is 1. For a general matrix the rule gives the long-run rate, and over a short stretch the gradient can grow first and shrink later.

The two directions are not equally reliable. With $\kappa \le 1$, a spectral radius below one guarantees that the gradient vanishes. A spectral radius above one is only a warning. Take the same $\rho = 1.05$ matrix, but start the state with entries of size about 1 instead of 0.001. Over four seeded runs the gradient after 100 steps came out between 0.1 and 0.7, not 131.5, because large states sit on the flat parts of tanh, which pushes $\kappa$ far below one. Setting $\rho = 1$ exactly does not rescue a plain cell either. With a state of size about 1, the $\rho = 1$ matrix gave between 0.03 and 0.16 after 100 steps, and training moves the weights away from $\rho = 1$ as soon as it starts. [The LSTM](/lessons/053-the-lstm) builds a path whose factor is held near one on purpose.

You can turn the rate into a distance. Call a signal too faint to learn from when it falls below a fraction, say 1%, of the signal at the end. Because $\log(\kappa\rho)$ is negative when the rate is below one, taking logarithms flips the inequality, and $(\kappa\rho)^T < 0.01$ becomes $T > \log(0.01)/\log(\kappa\rho)$. With $\kappa\rho = 0.9$ that gives $T = 44$, since $0.9^{44} = 0.0097$. With the running example's rate of 0.48, it is 7 steps.

Fading and blowing up feel different in practice. An exploding gradient is loud: the weights jump and the loss turns into `nan`. A vanishing gradient is silent: the loss flattens at a mediocre value, and the model looks like it has finished learning. The rest of the chapter treats the loud failure and the memory bill. Clipping cures only the loud failure.

> [!KEY]
> The gradient to a state $T$ steps back is a product of one factor per step, so it behaves like $(\kappa\rho)^T$. With a rate below one, the step-1 spike in the opening has almost no voice in training, and that is where the signal went.

## Cut the gradient, keep the state

Truncated backpropagation through time keeps the forward pass whole and cuts only the backward pass. Pick a window size $k$ and split the $T$ steps into chunks of $k$. Run the backward pass on each chunk alone, treating the state that entered the chunk as a plain constant. Figure 3 shows six steps with $k = 2$.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Two rows of six cells. In full backpropagation the gradient arrows cross every boundary. In truncated backpropagation with chunks of two, they stop at the chunk edges while forward arrows continue">
<defs><marker id="c038-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c038-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="24" class="fig-bold">Full backpropagation: the gradient crosses every boundary</text>
<rect x="40" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="72" y="65" text-anchor="middle" class="fig-small">step 1</text>
<line x1="104" y1="60" x2="132" y2="60" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="132" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="164" y="65" text-anchor="middle" class="fig-small">step 2</text>
<line x1="196" y1="60" x2="224" y2="60" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="224" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="256" y="65" text-anchor="middle" class="fig-small">step 3</text>
<line x1="288" y1="60" x2="316" y2="60" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="316" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="348" y="65" text-anchor="middle" class="fig-small">step 4</text>
<line x1="380" y1="60" x2="408" y2="60" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="408" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="440" y="65" text-anchor="middle" class="fig-small">step 5</text>
<line x1="472" y1="60" x2="500" y2="60" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="500" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="532" y="65" text-anchor="middle" class="fig-small">step 6</text>
<line x1="516" y1="112" x2="456" y2="112" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="424" y1="112" x2="364" y2="112" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="332" y1="112" x2="272" y2="112" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="240" y1="112" x2="180" y2="112" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="148" y1="112" x2="88" y2="112" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<text x="20" y="170" class="fig-bold">Truncated, k = 2: the gradient stops at each chunk edge</text>
<rect x="32" y="182" width="172" height="112" class="fill-soft-a"/>
<rect x="216" y="182" width="172" height="112" class="fill-soft-c"/>
<rect x="400" y="182" width="172" height="112" class="fill-soft-d"/>
<rect x="40" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="72" y="217" text-anchor="middle" class="fig-small">step 1</text>
<line x1="104" y1="212" x2="132" y2="212" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="132" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="164" y="217" text-anchor="middle" class="fig-small">step 2</text>
<line x1="196" y1="212" x2="224" y2="212" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="224" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="256" y="217" text-anchor="middle" class="fig-small">step 3</text>
<line x1="288" y1="212" x2="316" y2="212" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="316" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="348" y="217" text-anchor="middle" class="fig-small">step 4</text>
<line x1="380" y1="212" x2="408" y2="212" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="408" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="440" y="217" text-anchor="middle" class="fig-small">step 5</text>
<line x1="472" y1="212" x2="500" y2="212" class="stroke-a" marker-end="url(#c038-arrow-a)"/>
<rect x="500" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="532" y="217" text-anchor="middle" class="fig-small">step 6</text>
<line x1="148" y1="264" x2="88" y2="264" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="332" y1="264" x2="272" y2="264" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="516" y1="264" x2="456" y2="264" class="stroke-b" marker-end="url(#c038-arrow-b)"/>
<line x1="204" y1="258" x2="216" y2="270" class="stroke-bad"/>
<line x1="204" y1="270" x2="216" y2="258" class="stroke-bad"/>
<text x="210" y="288" text-anchor="middle" class="fig-small fill-bad">cut</text>
<line x1="388" y1="258" x2="400" y2="270" class="stroke-bad"/>
<line x1="388" y1="270" x2="400" y2="258" class="stroke-bad"/>
<text x="394" y="288" text-anchor="middle" class="fig-small fill-bad">cut</text>
<text x="114" y="175" text-anchor="middle" class="fig-small"></text>
<text x="298" y="175" text-anchor="middle" class="fig-small"></text>
<text x="482" y="175" text-anchor="middle" class="fig-small"></text>
</svg>
Figure 3. Blue arrows are the forward pass, orange arrows are gradients, and the three tinted blocks are chunks. Forward values still run through all six steps, so the state keeps its memory, but a gradient starting at step 6 stops at the edge between steps 4 and 5, and never reaches steps 1 to 4.
```

The forward arrows still run through all six steps, so the model keeps its memory: the state entering each chunk is the true state from the step before it, not a fresh $h_0$. But no gradient crosses a boundary, so the model gets no signal about how an earlier chunk shaped a later one.

Use the running example once more, with $h_0 = 0$, readings $1, 0.5, 0, 0, 0, 0$, and a loss that pulls on the last state with strength 1. Full backpropagation gives $dx_1 = 0.0601$, and the weights get $dW_{xh} = 0.2158$, $dW_{hh} = 0.9626$, and $db = 2.8532$. With $k = 2$, only the last chunk, steps 5 and 6, sees the loss. Its gradient reaches back two steps and stops, so $dx_1$ through $dx_4$ are exactly 0, $dW_{hh} = 0.4710$, $db = 1.6815$, and $dW_{xh} = 0$, because the last two readings are zero and the only readings that mattered are out of reach.

That last number is the price. A dependency longer than $k$ steps cannot be learned, because no gradient ever connects its two ends. The step-1 spike in the opening would be invisible with $k = 2$. Pick $k$ longer than the dependencies you care about and shorter than your memory budget allows.

The saving is in memory. If you run the forward pass chunk by chunk and drop each chunk's activations after its backward pass, you keep $n k d_h$ numbers instead of $n T d_h$. With $n = 32$, $d_h = 256$, and 8 bytes per number, 10,000 steps needs about 655 MB for the states alone, and a window of $k = 100$ needs about 6.6 MB. Time does not grow, since each step is still visited once forward and once backward. The exercise runs the forward pass once over the whole sequence to keep the code short, and the gradients it produces are the same.

```python
h_in = h_seq[start - 1] if start > 0 else h0   # true state entering the chunk
dx_c, dW_xh_c, dW_hh_c, db_c, _ = bptt(dH[start:end], x[start:end], h_in, W_xh, W_hh, b)
```

The usual mistake is to restart every chunk from `h0`, which changes the forward values as well as the gradient. The last output of `bptt`, the gradient on the chunk's starting state, is the one that would cross the boundary, so the code drops it.

## A step that is too long

Truncation does not stop the loud failure. Sometimes the gradient is correct and still enormous. Take a loss with a bowl in the middle, $0.05\,w^2$, and steep walls beyond $|w| = 1$, where the loss climbs as $5(|w|-1)^2$ on top. Recurrent networks can have walls like this, because the same weight is multiplied into the output many times.

Start at $w = 1.5$, where the gradient is $0.1 \times 1.5 + 10 \times 0.5 = 5.15$. It points the right way, toward the bowl, but with a learning rate of 1 the step has length 5.15. That jumps over the bowl to $w = -3.65$, high on the far wall, where the loss is 35.8 and the gradient is $-26.9$. The next step lands near $w = 23$. Each step is bigger than the last, which is how weights end up at `nan`.

You might say: lower the learning rate. That tames this wall, but it also shrinks every ordinary step on the flat parts of training, and you cannot know in advance how steep the next wall is. A better rule leaves small gradients alone and shortens only the ones that are too long. Cap the step at length 1, and the same start lands at $w = 0.5$, inside the bowl, with loss 0.0125. The direction was never the problem, only the length.

A model has many weight arrays, so the gradient is a list of arrays. Treat the whole list as one long vector and call its length $G$, the global norm. If $G$ is above a limit, multiply every array by the same factor so that the length lands exactly on the limit:

$$
g \leftarrow \min\left(1,\ \frac{\text{max\_norm}}{G}\right) g
$$

Read it aloud: if the gradient is shorter than the limit, the factor is 1 and nothing changes. If it is longer, the factor is the limit divided by the length, which shrinks it to exactly the limit. Two gradient arrays $[3]$ and $[4]$ have $G = \sqrt{9 + 16} = 5$. With a limit of 1, the factor is $0.2$, giving $[0.6]$ and $[0.8]$. In numpy:

```python
G = np.sqrt(sum(np.sum(g ** 2) for g in grads))
scale = min(1.0, max_norm / G) if G > 0 else 1.0
```

The guard on `G > 0` keeps an all-zero gradient from dividing by zero. Figure 4 explains why the factor is shared.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A gradient arrow of length 5 in the direction (3, 4), clipped on the left by one shared factor so its direction is kept, and clipped on the right array by array, which turns it">
<defs><marker id="c038-fig4-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c038-fig4-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c038-fig4-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-bad"/></marker></defs>
<text x="30" y="28" class="fig-bold">Global clip, limit 1</text>
<line x1="50" y1="265" x2="213.2" y2="265" class="stroke-muted"/>
<line x1="50" y1="265" x2="50" y2="53.8" class="stroke-muted"/>
<path d="M98,265 A48,48 0 0 0 50,217" class="stroke-muted dashed"/>
<line x1="50" y1="265" x2="194" y2="73" class="stroke-a" marker-end="url(#c038-fig4-arrow-a)"/>
<text x="202" y="87" class="fig-small">(3, 4)</text>
<text x="202" y="101" class="fig-small">length 5</text>
<line x1="50" y1="265" x2="78.8" y2="226.6" class="stroke-b" marker-end="url(#c038-fig4-arrow-b)"/>
<text x="116" y="243" class="fig-small fill-b">(0.6, 0.8)</text>
<text x="116" y="257" class="fig-small">length 1</text>
<text x="320" y="28" class="fig-bold">Clip each array alone, limit 1</text>
<line x1="340" y1="265" x2="503.2" y2="265" class="stroke-muted"/>
<line x1="340" y1="265" x2="340" y2="53.8" class="stroke-muted"/>
<path d="M388,265 A48,48 0 0 0 340,217" class="stroke-muted dashed"/>
<line x1="340" y1="265" x2="484" y2="73" class="stroke-a dashed" marker-end="url(#c038-fig4-arrow-a)"/>
<text x="492" y="87" class="fig-small">(3, 4)</text>
<line x1="340" y1="265" x2="388" y2="217" class="stroke-bad" marker-end="url(#c038-fig4-arrow-bad)"/>
<text x="410" y="243" class="fig-small fill-bad">(1, 1)</text>
<text x="410" y="257" class="fig-small">points at 45°, not 53°</text>
</svg>
Figure 4. The dashed arc is part of a circle of radius 1. On the left, one shared factor 1/5 shortens (3, 4) to (0.6, 0.8) along the same line. On the right, clipping [3] and [4] separately gives (1, 1), which has length 1.41, not 1, and points in a different direction.
```

With one shared factor, the clipped gradient is the original made shorter, so it still points the same way and still goes downhill. Clip each array by its own norm, and the arrays are scaled by different amounts, so the combined direction turns. In the figure, $[3]$ and $[4]$ clipped separately both become 1, giving $(1, 1)$, which points at 45 degrees instead of 53 degrees and has the wrong length. That is a different update from the one the gradient asked for.

Clipping has a price too. When it triggers, the step is shorter than the true gradient step, so the update is biased toward caution. And it only ever shrinks: a gradient of $10^{-10}$ passes through with a factor of 1, so clipping does nothing for vanishing gradients.

> [!KEY]
> Truncation cuts the gradient at chunk edges but carries the state across them, so memory stays small and long-range memory still exists, at the cost of never learning links longer than $k$. Clipping caps a gradient's length with one shared factor, so it keeps the direction and stops the jump to `nan`. Neither cures fading, and that failure needs a different structure.

> [!RECAP]
> - The carry is multiplied once per step, so a gradient crossing $T$ steps is a $T$-fold product, about $(\kappa\rho)^T$, with $\rho$ the largest absolute eigenvalue.
> - $\rho < 1$ with $\kappa \le 1$ guarantees vanishing. $\rho > 1$ warns of explosion but does not guarantee it.
> - Truncated backpropagation runs the backward pass on chunks of $k$ steps, each starting from the true state that entered it, so no gradient crosses a boundary.
> - Clipping multiplies every array by $\min(1, \text{max\_norm}/G)$, with $G$ the global norm, so the direction is kept.

## Build it step by step

The checks call the functions in this order. The forward pass and `bptt` from earlier chapters are already written for you, so you can compare your prediction with a measured gradient.

1. **`spectral_radius(W)`.** `W` is a square matrix. Return the largest absolute eigenvalue as a float. Use `np.linalg.eigvals`, `np.abs`, and `np.max`. The usual mistake is forgetting `np.abs`, which fails on the rotation.
2. **`gradient_scale_over_time(W, kappa, T)`.** Return `(kappa * rho) ** T` as a float, with `rho` from your first function. The usual mistake is applying the power to $\rho$ alone and then multiplying by $\kappa$.
3. **`steps_until_below(rho, kappa, threshold)`.** Return the smallest whole number $T$ with $(\kappa\rho)^T <$ threshold, or `None` when $\kappa\rho$ is 1 or more. Use logarithms and round up to the next whole number. The usual mistake is an off-by-one when the answer is exactly a whole number, as with $0.5^2 = 0.25$ and a threshold of 0.25, where two steps is not below.
4. **`chunk_bounds(T, k)`.** Return a list of `(start, end)` pairs, with `end` exclusive, covering `0` to `T` in order. Use `range(0, T, k)` and `min(start + k, T)`. The usual mistake is a last chunk that runs past `T`.
5. **`clip_by_global_norm(grads, max_norm)`.** Return a separate list with every array times one shared scale, computed from the norm of all the arrays together. Guard a zero norm. The usual mistake is clipping each array by its own norm.
6. **`truncated_bptt(dH, x, h0, W_xh, W_hh, b, k)`.** Run `rnn_forward` once. For each pair from `chunk_bounds`, call `bptt` on that slice with the true state that entered it, copy `dx` into the slice, and add the weight gradients. Return `(dx, dW_xh, dW_hh, db)`. The usual mistake is passing `h0` to every chunk.

## Cost and scaling

The prediction costs one eigenvalue computation, which grows with the cube of the hidden size $d_h$ and is paid once per weight matrix, not once per step. The measurement costs one backward pass. Full backpropagation needs memory that grows with $n T d_h$ and can learn dependencies up to $T$ steps long. Truncation needs memory that grows with $n k d_h$ and can learn dependencies up to $k$ steps long, at the same time cost. Clipping costs one pass over the gradients to compute the norm, which is small beside the backward pass.

## Where this breaks

- The rule treats the slope $\kappa$ as one number, while the real slope changes at every step. Treat the result as an order of magnitude, and expect the early part of a curve to sit above it for a matrix that is not a scaled rotation.
- A window shorter than the dependency you need hides it completely, and nothing in the loss tells you so. Test $k$ on a task where you know how far back the answer lies.
- A clip limit that is too low shortens every step, and training slows down. Watch how often clipping triggers. If it triggers on most steps, the limit is cutting the signal, not only the spikes.
- Fixing the explosion does not fix the fade, and truncation makes fading worse, since the gradient has fewer steps in which to live.

## The exercise

You build `spectral_radius`, `gradient_scale_over_time`, `steps_until_below`, `chunk_bounds`, `clip_by_global_norm`, and `truncated_bptt`. The checks use a diagonal matrix, a rotation whose eigenvalues are complex, and a random symmetric matrix compared against an independent computation. They test `steps_until_below` against counting one step at a time, including the case that returns `None`, and they run your prediction beside the backward pass at $\rho = 0.9$ and $\rho = 1.05$, requiring agreement within a factor of two. They then check the chunk boundaries on 6 and 7 steps, clip $[3]$ and $[4]$ to $[0.6]$ and $[0.8]$, and confirm that a long gradient lands on the limit with the same direction. For `truncated_bptt`, they compare to full `bptt` when $k = T$, and to finite differences of a loss in which each chunk's starting state is held constant. Last, they confirm that a loss on the final step sends no gradient into earlier chunks.
