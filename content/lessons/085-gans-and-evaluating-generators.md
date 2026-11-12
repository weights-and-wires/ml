---
title: "GANs, and how to judge a generator"
slug: 085-gans-and-evaluating-generators
part: "Part 8 — Generative models"
summary: "A forger and an inspector train against each other, and FID, precision, and recall tell you whether the forger is any good."
minutes: 80
gpu: false
objectives:
  - Derive the optimal discriminator p / (p + q) and link the game to the Jensen-Shannon divergence.
  - Explain why the saturating generator loss stalls and the non-saturating loss does not.
  - Train a one-dimensional GAN, and recognise mode collapse on a two-bell mixture.
  - Compute FID as the Frechet distance between two Gaussians, with a matrix square root from an eigendecomposition.
  - Read precision and recall of distributions, and say why likelihood and sample quality can disagree.
prereqs:
  - 008-sigmoid-and-log-loss
  - 084-autoencoders-and-vaes
exercise: 085-gans-and-evaluating-generators
---

# GANs, and how to judge a generator

A forger paints fake banknotes, and an inspector at the bank tries to spot them. At first the forger is clumsy, and the inspector catches every note by the paper alone. Each catch teaches the forger what gave the note away, so the next batch fixes that flaw. If the game goes well, the forger ends up making notes that the inspector can only call at random.

That game is a generative adversarial network, or GAN. The forger is a generator that turns random noise into a sample. The inspector is a discriminator that outputs the probability that a sample is real. You never write down a likelihood for the data, so you need no bound like the one in the [autoencoder chapter](/lessons/084-autoencoders-and-vaes). The discriminator is the only measuring stick, and it learns while the generator learns.

The missing likelihood creates a second problem. When a generator hands you a pile of samples, how do you know whether they are good? A generator that only draws one kind of picture can look perfect in ten samples. This chapter builds the game, shows what it optimises and where it breaks, and then builds the scores people use to compare generators: FID, and precision and recall of distributions.

## What you already know

- A sigmoid turns a score into a probability, and the log loss of a probability is a softplus of the score ([from a score to a probability](/lessons/008-sigmoid-and-log-loss)).
- The KL divergence measures how far one distribution sits from another, and it is 0 only when they match.
- A Gaussian bell is fixed by its mean and its covariance, and a covariance matrix has an eigendecomposition.
- A variational autoencoder draws a sample by decoding noise, and it trains against a lower bound on the log probability.

## The inspector's best answer

Call the real data's density $p(x)$, and call the density of the generator's samples $q(x)$. The discriminator $D(x)$ is a number between 0 and 1: its belief that $x$ came from the real data. Suppose half the notes it sees are real and half are fake. At a spot $x$ where real notes are common and fakes are rare, the best answer is close to 1. Where both are equally common, the best answer is $1/2$. Bayes' rule turns that into a formula, because the chance that $x$ is real is the real share of the density at $x$:

$$
D^*(x) = \frac{p(x)}{p(x) + q(x)}.
$$

Read it aloud: the density of real notes at $x$, divided by the total density of real and fake notes at $x$. The star marks the best possible discriminator for this particular $p$ and $q$.

Take the real data as a standard bell, $p = \mathrm{N}(0, 1)$, and a generator that has the right shape but sits in the wrong place, $q = \mathrm{N}(2, 1)$. At $x = 0$ the real density is $0.39894$ and the fake density is $0.05399$, so

$$
D^*(0) = \frac{0.39894}{0.39894 + 0.05399} = 0.8808.
$$

At $x = 1$ the two bells have the same height, $0.24197$, so $D^*(1) = 0.5$. Because the log of the ratio of two unit-variance bells is a straight line, $D^*(x) = \operatorname{sigmoid}(2 - 2x)$ for this pair. At $x = 0$ that gives $\operatorname{sigmoid}(2) = 0.8808$, which matches. Figure 1 draws both bells and $D^*$.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="Two bells, real at 0 and generated at 2, and the optimal discriminator falling from 1 to 0 and crossing one half where they meet">
  <line x1="60" y1="280" x2="560" y2="280" class="stroke-muted"/>
  <line x1="60" y1="160" x2="560" y2="160" class="stroke-grid dashed"/>
  <line x1="60" y1="40" x2="560" y2="40" class="stroke-grid dashed"/>
  <polyline points="60.0,280.0 70.0,279.9 80.0,279.8 90.0,279.6 100.0,279.3 110.0,278.7 120.0,277.7 130.0,276.0 140.0,273.4 150.0,269.6 160.0,264.2 170.0,256.8 180.0,247.5 190.0,236.1 200.0,223.0 210.0,209.0 220.0,195.0 230.0,182.3 240.0,172.0 250.0,165.3 260.0,163.0 270.0,165.3 280.0,172.0 290.0,182.3 300.0,195.0 310.0,209.0 320.0,223.0 330.0,236.1 340.0,247.5 350.0,256.8 360.0,264.2 370.0,269.6 380.0,273.4 390.0,276.0 400.0,277.7 410.0,278.7 420.0,279.3 430.0,279.6 440.0,279.8 450.0,279.9 460.0,280.0 470.0,280.0 480.0,280.0 490.0,280.0 500.0,280.0 510.0,280.0 520.0,280.0 530.0,280.0 540.0,280.0 550.0,280.0 560.0,280.0" fill="none" class="stroke-a" stroke-width="2.5"/>
  <polyline points="60.0,280.0 70.0,280.0 80.0,280.0 90.0,280.0 100.0,280.0 110.0,280.0 120.0,280.0 130.0,280.0 140.0,280.0 150.0,280.0 160.0,280.0 170.0,279.9 180.0,279.8 190.0,279.6 200.0,279.3 210.0,278.7 220.0,277.7 230.0,276.0 240.0,273.4 250.0,269.6 260.0,264.2 270.0,256.8 280.0,247.5 290.0,236.1 300.0,223.0 310.0,209.0 320.0,195.0 330.0,182.3 340.0,172.0 350.0,165.3 360.0,163.0 370.0,165.3 380.0,172.0 390.0,182.3 400.0,195.0 410.0,209.0 420.0,223.0 430.0,236.1 440.0,247.5 450.0,256.8 460.0,264.2 470.0,269.6 480.0,273.4 490.0,276.0 500.0,277.7 510.0,278.7 520.0,279.3 530.0,279.6 540.0,279.8 550.0,279.9 560.0,280.0" fill="none" class="stroke-b" stroke-width="2.5"/>
  <polyline points="60.0,40.0 70.0,40.0 80.0,40.0 90.0,40.0 100.0,40.1 110.0,40.1 120.0,40.1 130.0,40.2 140.0,40.3 150.0,40.4 160.0,40.6 170.0,40.9 180.0,41.3 190.0,42.0 200.0,42.9 210.0,44.3 220.0,46.4 230.0,49.4 240.0,53.8 250.0,60.0 260.0,68.6 270.0,80.3 280.0,95.6 290.0,114.4 300.0,136.3 310.0,160.0 320.0,183.7 330.0,205.6 340.0,224.4 350.0,239.7 360.0,251.4 370.0,260.0 380.0,266.2 390.0,270.6 400.0,273.6 410.0,275.7 420.0,277.1 430.0,278.0 440.0,278.7 450.0,279.1 460.0,279.4 470.0,279.6 480.0,279.7 490.0,279.8 500.0,279.9 510.0,279.9 520.0,279.9 530.0,280.0 540.0,280.0 550.0,280.0 560.0,280.0" fill="none" class="stroke-c" stroke-width="2.5"/>
  <circle cx="260" cy="68.6" r="5" class="fill-c"/>
  <circle cx="310" cy="160" r="5" class="fill-c"/>
  <text x="270" y="64" class="fig-small">D*(0) = 0.881</text>
  <text x="320" y="154" class="fig-small">D*(1) = 0.5</text>
  <text x="14" y="44" class="fig-small">D* = 1</text>
  <text x="14" y="164" class="fig-small">0.5</text>
  <text x="14" y="284" class="fig-small">0</text>
  <text x="128" y="206" class="fig-small">real p = N(0, 1)</text>
  <text x="396" y="206" class="fig-small">generated q = N(2, 1)</text>
  <text x="418" y="64" class="fig-small">discriminator D*</text>
  <text x="256" y="300" class="fig-small">x = 0</text>
  <text x="306" y="300" class="fig-small">1</text>
  <text x="356" y="300" class="fig-small">2</text>
</svg>
Figure 1. The optimal discriminator is near 1 where the real bell dominates, near 0 where the generated bell dominates, and exactly one half where the two bells have equal height, at x = 1. The curve is p / (p + q) computed on a grid.
```

> [!KEY]
> For a fixed generator, the best discriminator is $p / (p + q)$. It answers 1/2 everywhere only when $q = p$, which is the moment the forger has nothing left to learn.

## The game, and what it optimises

You train the discriminator to be right and the generator to make it wrong. Write the generator as $G$, so that a sample is $G(z)$ for a noise vector $z$. Score a pair with the average log probability the discriminator gives to the truth:

$$
V(D, G) = \mathbb{E}_{x \sim p}[\log D(x)] + \mathbb{E}_{x \sim q}[\log(1 - D(x))].
$$

The first term rewards $D$ for calling real samples real. The second rewards $D$ for calling generated samples fake. The discriminator climbs $V$, and the generator descends it, so the whole game is $\min_G \max_D V(D, G)$.

For a fixed $G$, you can maximise over $D$ one point at a time. At a single $x$ the integrand is $p \log d + q \log(1 - d)$, with slope $p/d - q/(1-d)$. That slope is 0 at $d = p/(p+q)$, which is $D^*$. For $p = 0.4$ and $q = 0.1$, $d = 0.4/0.5 = 0.8$.

Put $D^*$ back into $V$ and see what the generator is left to minimise. Write $m = (p + q)/2$ for the average of the two densities. Then $p + q = 2m$, so $\log D^* = \log(p/m) - \log 2$, and the same for $\log(1 - D^*)$ with $q$. The two $\log 2$ terms add to $-\log 4$, and the rest are two KL divergences:

$$
V(D^*, G) = 2\,\mathrm{JS}(p, q) - \log 4, \qquad \mathrm{JS}(p, q) = \tfrac{1}{2}\mathrm{KL}(p \,\|\, m) + \tfrac{1}{2}\mathrm{KL}(q \,\|\, m).
$$

$\mathrm{JS}$ is the Jensen-Shannon divergence. It is symmetric, it is 0 only when $p = q$, and it never exceeds $\log 2 = 0.6931$. The bound is reached when the two densities have no overlap. For $p = (0.5, 0.5, 0, 0)$ and $q = (0, 0, 0.5, 0.5)$ the average $m$ is $(0.25, 0.25, 0.25, 0.25)$, each KL equals $\log 2$, and $\mathrm{JS} = \log 2$.

On the two bells, $\mathrm{JS}(\mathrm{N}(0,1), \mathrm{N}(2,1)) = 0.3368$, so $V(D^*, G) = 0.6736 - 1.3863 = -0.7127$. Setting $q = p$ drives JS to 0 and $V$ to $-\log 4 = -1.3863$. A generator that plays against a perfect inspector is therefore descending the Jensen-Shannon divergence, and the game ends at the point where $q = p$ and $D^* = 1/2$.

> [!KEY]
> Against the best possible discriminator, the generator's objective is $2\,\mathrm{JS}(p, q) - \log 4$. Its minimum, $-\log 4$, is at $q = p$.

## Why the generator needs a different loss

The formula says the generator should minimise $\log(1 - D(G(z)))$. That loss has a problem at the start of training, when the inspector is sure. Let $b$ be the discriminator's logit on a fake sample, so $D = \operatorname{sigmoid}(b)$. The derivative of $\log(1 - \operatorname{sigmoid}(b))$ with respect to $b$ is $-\operatorname{sigmoid}(b) = -D$. A fake that the discriminator catches has $D$ near 0, so the gradient is near 0, and the generator receives almost no signal exactly when it needs the most. At $b = -6$, $D = 0.002473$, so the gradient has size $0.0025$ and the loss is nearly flat.

The fix is to change the generator's loss to $-\log D(G(z))$. It pushes in the same direction, because it also wants $D$ larger. Its derivative with respect to $b$ is $\operatorname{sigmoid}(b) - 1 = -(1 - D)$. At $b = -6$ that is $-0.9975$, and it shrinks only as the generator wins. At $b = 0$ both gradients have size $0.5$, and at $b = 4$ they swap, $0.982$ against $0.018$. This is the *non-saturating* loss. Figure 2 compares the two. The non-saturating loss is a heuristic, because it no longer minimises JS exactly, but it moves in the same direction and it works.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="Gradient size against the fake logit: the saturating loss is flat for a confident discriminator, the non-saturating loss is steep">
  <line x1="60" y1="280" x2="560" y2="280" class="stroke-muted"/>
  <line x1="60" y1="40" x2="60" y2="280" class="stroke-muted"/>
  <polyline points="60.0,279.9 67.8,279.9 75.6,279.9 83.4,279.8 91.2,279.8 99.1,279.7 106.9,279.6 114.7,279.5 122.5,279.4 130.3,279.2 138.1,279.0 145.9,278.7 153.8,278.4 161.6,277.9 169.4,277.4 177.2,276.6 185.0,275.7 192.8,274.5 200.6,273.0 208.4,271.0 216.2,268.6 224.1,265.6 231.9,261.8 239.7,257.1 247.5,251.4 255.3,244.5 263.1,236.2 270.9,226.6 278.8,215.5 286.6,203.0 294.4,189.4 302.2,174.9 310.0,160.0 317.8,145.1 325.6,130.6 333.4,117.0 341.2,104.5 349.1,93.4 356.9,83.8 364.7,75.5 372.5,68.6 380.3,62.9 388.1,58.2 395.9,54.4 403.8,51.4 411.6,49.0 419.4,47.0 427.2,45.5 435.0,44.3 442.8,43.4 450.6,42.6 458.4,42.1 466.2,41.6 474.1,41.3 481.9,41.0 489.7,40.8 497.5,40.6 505.3,40.5 513.1,40.4 520.9,40.3 528.8,40.2 536.6,40.2 544.4,40.1 552.2,40.1 560.0,40.1" fill="none" class="stroke-bad" stroke-width="2.5"/>
  <polyline points="60.0,40.1 67.8,40.1 75.6,40.1 83.4,40.2 91.2,40.2 99.1,40.3 106.9,40.4 114.7,40.5 122.5,40.6 130.3,40.8 138.1,41.0 145.9,41.3 153.8,41.6 161.6,42.1 169.4,42.6 177.2,43.4 185.0,44.3 192.8,45.5 200.6,47.0 208.4,49.0 216.2,51.4 224.1,54.4 231.9,58.2 239.7,62.9 247.5,68.6 255.3,75.5 263.1,83.8 270.9,93.4 278.8,104.5 286.6,117.0 294.4,130.6 302.2,145.1 310.0,160.0 317.8,174.9 325.6,189.4 333.4,203.0 341.2,215.5 349.1,226.6 356.9,236.2 364.7,244.5 372.5,251.4 380.3,257.1 388.1,261.8 395.9,265.6 403.8,268.6 411.6,271.0 419.4,273.0 427.2,274.5 435.0,275.7 442.8,276.6 450.6,277.4 458.4,277.9 466.2,278.4 474.1,278.7 481.9,279.0 489.7,279.2 497.5,279.4 505.3,279.5 513.1,279.6 520.9,279.7 528.8,279.8 536.6,279.8 544.4,279.9 552.2,279.9 560.0,279.9" fill="none" class="stroke-c" stroke-width="2.5"/>
  <line x1="122.5" y1="40" x2="122.5" y2="280" class="stroke-muted dashed"/>
  <circle cx="122.5" cy="279.4" r="5" class="fill-bad"/>
  <circle cx="122.5" cy="40.6" r="5" class="fill-c"/>
  <text x="134" y="270" class="fig-small">logit -6: saturating 0.0025</text>
  <text x="134" y="58" class="fig-small">non-saturating 0.9975</text>
  <text x="404" y="270" class="fig-small">saturating grows only</text>
  <text x="404" y="286" class="fig-small">when D is already near 1</text>
  <text x="14" y="44" class="fig-small">1</text>
  <text x="14" y="284" class="fig-small">0</text>
  <text x="226" y="316" class="fig-small">discriminator logit on a fake sample, -8 to 8</text>
</svg>
Figure 2. At a logit of -6 the discriminator is confident the sample is fake. The saturating loss log(1 - D) has gradient size 0.0025 there, and the non-saturating loss -log D has 0.9975.
```

Write both losses through softplus, the stable $\log(1 + e^{z})$ from the [log-loss chapter](/lessons/008-sigmoid-and-log-loss). Since $\log(1 - \operatorname{sigmoid}(b)) = -\operatorname{softplus}(b)$ and $-\log\operatorname{sigmoid}(b) = \operatorname{softplus}(-b)$:

```python
d_loss = softplus(-a).mean() + softplus(b).mean()   # a: real logits, b: fake logits
g_sat = (-softplus(b)).mean()
g_nonsat = softplus(-b).mean()
grad_sat, grad_nonsat = -sigmoid(b) / b.size, (sigmoid(b) - 1) / b.size
```

> [!WARNING]
> Writing `np.log(1 - sigmoid(b))` overflows. At $b = 800$ the sigmoid rounds to exactly 1, and the log of 0 is minus infinity. Go through softplus instead.

## A GAN you can train in a few hundred steps

Make everything one-dimensional. The real data is $\mathrm{N}(3, 1.5^2)$. The generator is a straight line of noise, $G(z) = a z + b$ with $z \sim \mathrm{N}(0, 1)$, so it can match the real data only when $b = 3$ and $|a| = 1.5$.

The discriminator needs a design choice. A logistic discriminator on $x$ alone, $\operatorname{sigmoid}(wx + c)$, is monotone in $x$, so it can separate two bells by their means and has nothing to say about their spread. The optimal discriminator for two Gaussians with different variances has a log-ratio that is quadratic in $x$. Give the logit the features $x$ and $x^2$: $D(x) = \operatorname{sigmoid}(w_1 x + w_2 x^2 + c)$.

Each iteration does one discriminator step and one generator step, both on fresh batches of 256. The discriminator loss is `softplus(-logit_real)` plus `softplus(logit_fake)`, averaged. Its gradient with respect to the weights $(w_1, w_2, c)$ is the mean of $-(1 - D(x))\,\phi(x)$ over real samples plus the mean of $D(x)\,\phi(x)$ over fakes, where $\phi(x) = (x, x^2, 1)$. The generator uses the non-saturating loss. By the chain rule, the gradient with respect to a fake sample $x$ is $(D(x) - 1)(w_1 + 2 w_2 x)$ divided by the batch size, and $x$ depends on $a$ with slope $z$ and on $b$ with slope 1:

```python
dx = (sigmoid(logit) - 1.0) / batch * (w1 + 2 * w2 * x)   # one entry per fake sample
grad_a, grad_b = np.sum(dx * z), np.sum(dx)
```

Use Adam with $\beta_1 = 0.5$ and $\beta_2 = 0.9$, learning rate $0.1$ for the discriminator and $0.02$ for the generator, and start from $a = 1$, $b = 0$ with a discriminator of zeros. After 300 iterations with seed 0, averaging the generator over the last 50 iterations, the mean is $2.91$ and the standard deviation is $1.49$, against targets $3$ and $1.5$. The generator learned the mean and the spread without ever seeing a density, only the discriminator's answers.

## Mode collapse on a mixture

Make the real data two bells: $p = \tfrac{1}{2}\mathrm{N}(-3, 0.5^2) + \tfrac{1}{2}\mathrm{N}(3, 0.5^2)$. Suppose the generator learns to produce one bell perfectly, $q = \mathrm{N}(3, 0.5^2)$, and nothing at $-3$.

The generator only receives gradient where its own samples land. At $x = 3$ the real density is half the generated density, so $D^*(3) = 1/3$, and the generator is nudged slightly. At $x = -3$ the discriminator would say 1, and it is right, but no fake sample lands there, so no gradient reaches the generator from that region. Nothing tells the generator to cover the missing bell. This is *mode collapse*: a generator that covers part of the data and does it well.

Collapse is also cheap in the game's own score. The collapsed generator has $\mathrm{JS} = 0.2158$, against a maximum of $0.6931$. A single wide Gaussian with the mixture's mean 0 and variance $9.25$ does worse by JS, $0.2804$, even though it never drops a mode. Figure 3 draws the three densities.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="A two-bell real density, a generator collapsed on the right bell, and a wide Gaussian with the same mean and variance as the mixture">
  <line x1="60" y1="280" x2="560" y2="280" class="stroke-muted"/>
  <polyline points="60.0,274.8 70.0,274.0 80.0,273.0 90.0,272.0 100.0,270.8 110.0,269.6 120.0,268.2 130.0,266.8 140.0,265.3 150.0,263.7 160.0,262.0 170.0,260.4 180.0,258.6 190.0,256.9 200.0,255.2 210.0,253.5 220.0,251.9 230.0,250.4 240.0,248.9 250.0,247.7 260.0,246.5 270.0,245.6 280.0,244.8 290.0,244.3 300.0,243.9 310.0,243.8 320.0,243.9 330.0,244.3 340.0,244.8 350.0,245.6 360.0,246.5 370.0,247.7 380.0,248.9 390.0,250.4 400.0,251.9 410.0,253.5 420.0,255.2 430.0,256.9 440.0,258.6 450.0,260.4 460.0,262.0 470.0,263.7 480.0,265.3 490.0,266.8 500.0,268.2 510.0,269.6 520.0,270.8 530.0,272.0 540.0,273.0 550.0,274.0 560.0,274.8" fill="none" class="stroke-c" stroke-width="2.5"/>
  <polyline points="60.0,280.0 70.0,280.0 80.0,280.0 90.0,280.0 100.0,280.0 110.0,279.8 120.0,279.2 130.0,276.6 140.0,269.3 150.0,253.2 160.0,226.5 170.0,195.1 180.0,173.1 190.0,173.1 200.0,195.1 210.0,226.5 220.0,253.2 230.0,269.3 240.0,276.6 250.0,279.2 260.0,279.8 270.0,280.0 280.0,280.0 290.0,280.0 300.0,280.0 310.0,280.0 320.0,280.0 330.0,280.0 340.0,280.0 350.0,280.0 360.0,279.8 370.0,279.2 380.0,276.6 390.0,269.3 400.0,253.2 410.0,226.5 420.0,195.1 430.0,173.1 440.0,173.1 450.0,195.1 460.0,226.5 470.0,253.2 480.0,269.3 490.0,276.6 500.0,279.2 510.0,279.8 520.0,280.0 530.0,280.0 540.0,280.0 550.0,280.0 560.0,280.0" fill="none" class="stroke-a" stroke-width="2.5"/>
  <polyline points="60.0,280.0 70.0,280.0 80.0,280.0 90.0,280.0 100.0,280.0 110.0,280.0 120.0,280.0 130.0,280.0 140.0,280.0 150.0,280.0 160.0,280.0 170.0,280.0 180.0,280.0 190.0,280.0 200.0,280.0 210.0,280.0 220.0,280.0 230.0,280.0 240.0,280.0 250.0,280.0 260.0,280.0 270.0,280.0 280.0,280.0 290.0,280.0 300.0,280.0 310.0,280.0 320.0,280.0 330.0,280.0 340.0,280.0 350.0,279.9 360.0,279.7 370.0,278.3 380.0,273.3 390.0,258.7 400.0,226.4 410.0,172.9 420.0,110.2 430.0,66.2 440.0,66.2 450.0,110.2 460.0,172.9 470.0,226.4 480.0,258.7 490.0,273.3 500.0,278.3 510.0,279.7 520.0,279.9 530.0,280.0 540.0,280.0 550.0,280.0 560.0,280.0" fill="none" class="stroke-b" stroke-width="2.5" stroke-dasharray="6 4"/>
  <text x="96" y="140" class="fig-small">real p: two bells</text>
  <text x="400" y="62" class="fig-small">collapsed q</text>
  <text x="228" y="204" class="fig-small">wide Gaussian, same mean and variance</text>
  <text x="186" y="300" class="fig-small">-3</text>
  <text x="306" y="300" class="fig-small">0</text>
  <text x="426" y="300" class="fig-small">3</text>
</svg>
Figure 3. The collapsed generator (dashed) sits on one real bell and misses the other. The wide Gaussian (green) matches the mixture's mean and variance and puts 38% of its mass in the gap, between -1.5 and 1.5, where the real density is nearly 0.
```

> [!WARNING]
> A generator can score well on a measure of how real its samples look and still miss half the data. Judge a generator on both what it draws and what it never draws.

## Scoring a generator: FID

You want one number for how close a pile of generated samples is to a pile of real ones. Raw pixels are useless, because two good faces differ in most pixels. Instead, push every sample through a fixed pretrained network and compare the features it returns. Fit a Gaussian to each cloud of features, with mean $\mu$ and covariance $\Sigma$, and measure how far apart the two Gaussians are. This is the Frechet Inception Distance, or FID, from Heusel et al. (2017). It uses the 2048-dimensional features of an Inception network.

The distance between two Gaussians has a closed form:

$$
\mathrm{FID} = \lVert \mu_1 - \mu_2 \rVert^2 + \operatorname{tr}\!\left(\Sigma_1 + \Sigma_2 - 2\left(\Sigma_1^{1/2}\,\Sigma_2\,\Sigma_1^{1/2}\right)^{1/2}\right).
$$

Read it aloud: the squared distance between the two means, plus a term that measures how different the two spreads are. A small FID means the two feature clouds are close. In one dimension the matrices become numbers and the formula is $(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$, with $\sigma$ the standard deviation.

Use the mixture. The real mixture has mean 0 and standard deviation $\sqrt{9.25} = 3.0414$. The collapsed generator has mean 3 and standard deviation $0.5$, so $\mathrm{FID} = 3^2 + (3.0414 - 0.5)^2 = 9 + 6.4586 = 15.46$. The wide Gaussian has the same mean and variance as the mixture, so its FID is exactly 0, even though it puts 38% of its samples in the empty gap between the bells.

When two Gaussians share the same covariance $\Sigma$, the second term vanishes: the root is $\Sigma$, and $\operatorname{tr}(2\Sigma - 2\Sigma) = 0$. When both are scaled identity matrices, $\Sigma_1 = s_1^2 I$ and $\Sigma_2 = s_2^2 I$ in $d$ dimensions, the formula gives $d\,(s_1 - s_2)^2$. With $s_1 = 2$, $s_2 = 3$, and $d = 3$ that is $3$.

Why the square root of a matrix, and not something simpler? Matrices do not commute, so $(\Sigma_1 \Sigma_2)^{1/2}$ cannot be split into a product of two roots, which is why the formula sandwiches $\Sigma_2$ between two copies of $\Sigma_1^{1/2}$. Two shortcuts both break the identical-Gaussian test. Dropping the root and using $\operatorname{tr}(\Sigma_1 + \Sigma_2 - 2\Sigma_1\Sigma_2)$ gives $\operatorname{tr}(2\Sigma - 2\Sigma^2)$, which is not 0 unless $\Sigma = I$. For $\Sigma = \begin{bmatrix} 2 & 0.5 \\ 0.5 & 1 \end{bmatrix}$ it is $-5$. Taking the square root of each entry of $\Sigma_1\Sigma_2$ gives $-0.36$ on the same matrix, a negative distance.

You build the matrix square root from an eigendecomposition. A symmetric positive semi-definite matrix factors as $A = V \Lambda V^{\top}$ with eigenvalues $\Lambda$ and orthonormal eigenvectors $V$. Its square root is $V \Lambda^{1/2} V^{\top}$, because squaring it cancels the $V^{\top} V$ in the middle. Clip eigenvalues that rounding pushed slightly below 0:

```python
vals, vecs = np.linalg.eigh(a)
root = (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
```

> [!KEY]
> FID compares two Gaussians fitted to feature clouds: the squared gap between the means, plus a trace term that needs a true matrix square root. It sees only the mean and the covariance of the features, so a generator can reach 0 while still being wrong.

## Precision and recall of distributions

FID folds two failures into one number. Samples that look bad hurt it, and so do missing modes, and you cannot tell which from the score. Kynkaanniemi et al. (2019) split them apart with nearest neighbours. Around each sample, draw a ball whose radius is the distance to its $k$-th nearest neighbour in its own set. The union of the balls around real samples is an estimate of where the real data lives. The union around generated samples estimates where the generator lives.

*Precision* is the fraction of generated samples that land inside the real balls, which asks whether the samples look real. *Recall* is the fraction of real samples inside the generated balls, which asks whether the generator covers the data. This is one of two definitions in the literature; Sajjadi et al. (2018) define a curve instead.

On 600 samples from each source, with $k = 3$ and seed 0 in one dimension:

| Generator | Precision | Recall | FID (one dimension) |
|---|---|---|---|
| Both bells, right shape | 0.980 | 0.992 | near 0 |
| Collapsed on one bell | 0.960 | 0.528 | 15.46 |
| Wide Gaussian | 0.678 | 0.980 | 0 |

The collapsed generator has precision near 1 and recall near $1/2$, the signature of a missing mode. The wide Gaussian has the reverse, high recall and low precision, because a third of its samples land in the gap. FID alone rates the wide Gaussian as perfect.

## When likelihood and sample quality disagree

A GAN does not give you $q(x)$, so you cannot compute the likelihood of held-out data under it. For a model that does, such as a VAE or the [diffusion models in the next chapter](/lessons/086-diffusion-models), likelihood and sample quality are still separate questions.

Take a model that is 1% a good generator and 99% garbage, $q' = 0.01\,q_{\text{good}} + 0.99\,q_{\text{garbage}}$. For every real $x$, $q'(x) \ge 0.01\,q_{\text{good}}(x)$, so the log likelihood is at most $\log 100 = 4.6$ nats worse than the good model. On a $32 \times 32$ colour image that is $4.6 / 3072 = 0.0015$ nats per dimension, a number nobody would notice, and yet 99 of every 100 samples are garbage. The log likelihood is blind to it. The reverse also happens. A collapsed generator draws sharp samples, but a real point at the missing bell has density 0, so its log likelihood is minus infinity. Theis et al. (2015) discuss these gaps in detail.

> [!KEY]
> Likelihood, FID, and precision and recall measure different things. Report samples and at least two scores, and read recall as the check on the one-bell failure.

## Build it step by step

1. `optimal_discriminator(p, q)` takes two arrays of density values and returns `p / (p + q)`, with `0.5` where both are 0. At $p = (0.4, 0, 0.2, 0)$ and $q = (0.1, 0.3, 0.2, 0)$ it returns $(0.8, 0, 0.5, 0.5)$. The common mistake is dividing by zero and returning `nan`.
2. `js_divergence(p, q, dx)` returns $\tfrac{1}{2}\mathrm{KL}(p \,\|\, m) + \tfrac{1}{2}\mathrm{KL}(q \,\|\, m)$ as a sum times `dx`, where a term with density 0 contributes 0. It is 0 for equal densities, $\log 2$ for disjoint ones, and $0.3368$ for the two bells. Take care with `0 * log(0)`.
3. `minimax_value(p, q, d, dx)` returns `sum(p * log(d) + q * log(1 - d)) * dx`, with `d` clipped into $[10^{-12}, 1 - 10^{-12}]$ first. At $D^*$ it equals $2\,\mathrm{JS} - \log 4$, and at $D = 1/2$ it equals $-\log 4$. Any other discriminator scores lower.
4. `gan_losses(d_real_logit, d_fake_logit)` returns `(d_loss, g_saturating, g_nonsaturating)` as means, through softplus. It must stay finite at logits of $\pm 800$. The common mistake is `np.log(1 - sigmoid(b))`, which overflows.
5. `generator_loss_grads(d_fake_logit)` returns the gradients of the two generator losses with respect to the fake logits, each of shape `(m,)` and including the $1/m$ from the mean. A central finite difference of `gan_losses` matches both. At a logit of $-6$ the saturating one is about $-0.0025$ and the non-saturating one about $-0.9975$, before dividing by $m$.
6. `sqrtm_psd(a)` takes a symmetric positive semi-definite matrix and returns its symmetric square root through `np.linalg.eigh`, clipping eigenvalues at 0. The square root of $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ has $\sqrt{2}/2$ in every entry. The mistake is `np.sqrt(a)`, which takes the square root of each entry.
7. `frechet_distance(mu1, s1, mu2, s2)` returns the squared mean gap plus $\operatorname{tr}(\Sigma_1) + \operatorname{tr}(\Sigma_2) - 2\operatorname{tr}\big((\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\big)$. It is 0 for identical Gaussians and $d(s_1 - s_2)^2$ for two scaled identities. The mistake is using `s1 @ s2` where the square root belongs.
8. `train_gan_1d(target_mean, target_std, steps, seed)` runs the loop from the training section and returns the generator's mean $b$ and standard deviation $|a|$, each averaged over the last 50 iterations. Draw a real batch, then a noise batch for the discriminator step, then fresh noise for the generator step. The common mistake is to forget that $a$ can go negative, which is why the standard deviation is $|a|$.

> [!RECAP]
> - For a fixed generator, the best discriminator is $p / (p + q)$. For $\mathrm{N}(0,1)$ against $\mathrm{N}(2,1)$ it is $\operatorname{sigmoid}(2 - 2x)$, equal to $0.8808$ at $x = 0$.
> - At $D^*$ the objective equals $2\,\mathrm{JS}(p, q) - \log 4$, which is $-0.7127$ for those two bells and $-\log 4$ when $q = p$.
> - The saturating generator loss has gradient size $D$, which is $0.0025$ at a logit of $-6$. The non-saturating loss has $1 - D = 0.9975$ there.
> - FID is $\lVert \mu_1 - \mu_2 \rVert^2 + \operatorname{tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})$, and it needs a true matrix square root.
> - Recall catches mode collapse (0.528 for one bell of two), precision catches bad samples, and likelihood can disagree with both.

## Cost and scaling

FID costs a pass of every sample through the feature network, then an eigendecomposition of a $d \times d$ matrix with $d = 2048$, which grows as $d^3$. A covariance of $d$ features needs more than $d$ samples to be full rank, and FID shifts with the sample count, so compare scores computed with the same count.

## Where this breaks

- **The two players are not matched.** On this chapter's toy loop, $\beta_1 = 0.9$ with learning rates of $0.05$ for both players left the final standard deviation up to $0.96$ away from the target over six seeds. Lower $\beta_1$ and different step sizes fixed it.
- **The generator collapses onto some modes.** Check recall, not only precision and FID.
- **FID sees two moments.** Two clouds with the same mean and covariance score 0 even when their shapes differ, and the score depends on which feature network you use.
- **There is no likelihood.** A GAN cannot tell you the probability of a held-out sample.

## The exercise

You implement `optimal_discriminator`, `js_divergence`, `minimax_value`, `gan_losses`, `generator_loss_grads`, `sqrtm_psd`, `frechet_distance`, and `train_gan_1d`. The file provides `stable_sigmoid`, `softplus`, and an `Adam` class. The checks run in that order: $D^*$ on a hand array and on the two bells, the Jensen-Shannon values, the minimax identity at $D^*$, the three losses and their stability at $\pm 800$, the gradients against finite differences and the vanishing saturating gradient, the matrix square root, the Frechet distance against its closed forms and against SciPy's matrix square root, and a 300-step GAN that matches two different targets.

## Further reading

- For the original game, the optimal discriminator, and the Jensen-Shannon result, see [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014).
- For FID, see [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (Heusel et al., 2017).
- For precision and recall of distributions with nearest-neighbour balls, see [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991) (Kynkaanniemi et al., 2019). For the curve-based definition, see [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035) (Sajjadi et al., 2018).
- For why likelihood and sample quality can disagree, see [A note on the evaluation of generative models](https://arxiv.org/abs/1511.01844) (Theis, van den Oord, and Bethge, 2015).
- For the course context, see the [CS236 syllabus](https://deepgenerativemodels.github.io/syllabus.html) (Stanford, Fall 2023). It lists "GANs" over two sessions, in weeks 5 and 6, and "Evaluation of Generative Models" in week 8. The page does not print lecture numbers.
- For GANs next to variational autoencoders in a vision course, see [CS231n lecture 13, Generative Models 1](https://cs231n.stanford.edu/schedule.html). The schedule prints "Lecture 13: Generative Models 1" and lists "Generative Adversarial Network".
