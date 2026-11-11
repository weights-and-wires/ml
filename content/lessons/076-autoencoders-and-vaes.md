---
title: "Autoencoders and variational autoencoders"
slug: 076-autoencoders-and-vaes
part: "Part 8 — Generative models"
summary: "A linear autoencoder keeps the first principal component, and a VAE draws rows by climbing a bound on the log probability."
minutes: 75
gpu: false
objectives:
  - Recover the first principal component with a linear autoencoder.
  - Write the closed-form KL of a diagonal Gaussian and its gradient.
  - Reparameterise a sample so the gradient in the log-variance still exists.
  - Match a known log probability with the ELBO, and miss it with a wider posterior.
  - Train a tiny VAE and read posterior collapse off a large beta.
prereqs:
  - 028-principal-components
  - 032-gaussian-mixtures-and-em
exercise: 076-autoencoders-and-vaes
---

# Autoencoders and variational autoencoders

You have a stack of house photos, and each photo is a long row of pixel values. You want a short code for each photo, so a file of codes is much smaller than the stack, and a code still rebuilds a photo you would recognise. You also want to invent a photo of a house the stack does not contain, a row that looks as if it could have been one of the photos.

An autoencoder does the first job. It squeezes a row through a narrow middle, the bottleneck, and stretches that middle back out into a rebuilt row. You train it so the rebuilt row stays close to the original. When the squeeze and the stretch are both straight lines, the line you keep is the one principal components already picked. Draw a code at random and the stretch paints a smear, because nothing asked the codes to form a shape you know how to draw from.

A variational autoencoder asks for that shape. It treats the code as a draw from a bell curve, and it trains the squeeze so those bells sit near a standard normal, mean 0 and variance 1 on every coordinate. You can then draw a code from that bell and stretch it into a row. The probability of a row hides a sum over every code that could have produced it, and that sum is out of reach once the stretch is more than a straight line. You climb a lower bound instead. When the bell you use is the true one, the bound touches. When the bell is wider, the bound is strictly less.

## What you already know

- Centring a table subtracts the column means, so the middle of the cloud sits at the origin.
- The first principal component is the direction whose scores have the largest variance, and the SVD of the centred table lists those directions in order.
- The log probability of a Gaussian bell is a squared distance from the mean, plus a term that depends only on the variance.
- For a Gaussian mixture, a lower bound on the log probability touches it when the guessed shares are the true posterior shares, and the gap is a KL divergence.
- A gradient is the direction to move a weight so a scalar loss goes down.

## Squeeze a row down and rebuild it

Reduce two photos to two numbers each, and the stack is a matrix with two rows. A bottleneck of width 1 rebuilds each row from one number, so the rebuilt row has to lie on a line. You want the line that loses the least.

The first line to try is the average row, the arrow from the origin through the middle of the cloud. That arrow records where the cloud sits, not how it spreads. After you centre, the long axis is a different direction. On the worked matrix the two directions are orthogonal, so the average-row line keeps none of the centred points.

Take the matrix with rows $(3, 1)$ and $(1, 3)$. The column means are $(3+1)/2 = 2$ and $(1+3)/2 = 2$, so the middle is $(2, 2)$ and the average row points along $(1, 1)$. Subtract the middle and the rows become $(1, -1)$ and $(-1, 1)$. Both are multiples of $(1, -1)$. The unit vector along that line is

$$
u = \frac{(1, -1)}{\sqrt{2}}.
$$

The centred rows already lie on this one line. The scores are $\sqrt{2}$ and $-\sqrt{2}$, the squared scores sum to 4, and the singular value is 2. The other singular value is 0. Figure 1 draws both lines: the points sit on $u$, and the average row sits on the other diagonal.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="Two points on the principal-component diagonal, and the row-mean diagonal crossing at the mean">
  <line x1="40" y1="271.8" x2="330" y2="271.8" class="stroke-muted"/>
  <line x1="64.4" y1="20" x2="64.4" y2="312" class="stroke-muted"/>
  <line x1="47.4" y1="288.7" x2="308.6" y2="29.3" class="stroke-bad dashed"/>
  <line x1="47.4" y1="29.3" x2="308.6" y2="288.7" class="stroke-b" stroke-width="2.5"/>
  <circle cx="121.2" cy="102.6" r="5.5" class="fill-a"/>
  <circle cx="234.8" cy="215.4" r="5.5" class="fill-a"/>
  <circle cx="178.0" cy="159.0" r="4" class="fill-ink"/>
  <text x="70" y="286" class="fig-small">0</text>
  <text x="40" y="96" class="fig-small">(1, 3)</text>
  <text x="248" y="230" class="fig-small">(3, 1)</text>
  <text x="188" y="148" class="fig-small">mean (2, 2)</text>
  <text x="250" y="78" class="fig-small">row mean</text>
  <text x="200" y="268" class="fig-small">principal component</text>
</svg>
Figure 1. The rows (3, 1) and (1, 3) sit on the line through the mean along (1, −1). The mean of the rows points along (1, 1), the other diagonal, and the two lines are orthogonal.
```

A linear autoencoder keeps the first principal component of the centred rows, the line the chapter on [principal components](/lessons/028-principal-components) reads from the SVD. The decoder's columns are those directions. One component is a single column, $u$ or $-u$, and either sign rebuilds the same shadows, because the score flips with the vector and the product does not.

Centre the rows, factor $X_c = U S V^{\top}$, and keep the first $k$ rows of $V^{\top}$ as the decoder's columns:

$$
\text{decoder} = V_{:,\, 1:k}.
$$

The decoder's columns are the leading right singular vectors of the centred data:

```python
centred = X - X.mean(axis=0)
_, _, vt = np.linalg.svd(centred, full_matrices=False)
decoder = vt[:rank].T
```

On the worked matrix the column is parallel to $(1, -1)/\sqrt{2}$. The normalised mean of the rows is orthogonal to it, so a decoder built from the average row fails.

> [!KEY]
> A linear autoencoder, after centring, keeps the same subspace as the top principal components. The sign of each column does not matter.

## The bound under the log probability

A code you cannot draw from is not yet a generator. Call the code $z$ and the row $x$. The prior $p(z)$ is the distribution of the code before you see a row, and the decoder $p(x \mid z)$ is the row given the code. $\log p(x)$ sums $p(z)\,p(x \mid z)$ over every code. That sum is closed form for the straight-line model in this section, and it is not once the decoder bends, so you climb a bound instead of $\log p(x)$.

Pick any guess $q(z)$ at the posterior $p(z \mid x)$. The gap between $\log p(x)$ and the bound is at least 0, and it is 0 when $q$ is the true posterior. The gap is a KL divergence, the same gap the chapter on [Gaussian mixtures and EM](/lessons/032-gaussian-mixtures-and-em) drives to zero when the guessed shares are the true posterior shares. Write the bound as a reconstruction term minus a penalty that keeps $q$ near the prior:

$$
\operatorname{bound}(q) = \mathbb{E}_{q}[\log p(x \mid z)] - \operatorname{KL}(q(z) \,\|\, p(z)).
$$

That is the average log probability of the row given a code from $q$, minus how far $q$ sits from the prior. The bound is at most $\log p(x)$:

$$
\log p(x) = \operatorname{bound}(q) + \operatorname{KL}(q(z) \,\|\, p(z \mid x)).
$$

The second term is that gap. The model is one-dimensional. The prior is $z \sim \mathrm{N}(0, 1)$, and the decoder is $x \mid z \sim \mathrm{N}(z, 1)$. The marginal of $x$ is then normal with mean 0 and variance 2:

$$
\log p(x) = -\frac{1}{2}\log(4\pi) - \frac{x^2}{4}.
$$

That is the log density of a bell with variance 2. At $x = 2$ it equals $-2.265512$.

The prior pulls $z$ toward 0 with precision 1, and the decoder pulls $z$ toward $x$ with precision 1. Precisions add, so the true posterior variance is $1/2$ and the mean is $x/2$. At $x = 2$ the true posterior is $\mathrm{N}(1, 1/2)$, with log-variance $\log(1/2)$.

For any bell $q = \mathrm{N}(\mu, e^{\ell})$, the average of $(x-z)^2$ under $q$ is the squared gap of the means plus the variance of $q$:

$$
\mathbb{E}_{q}[\log p(x \mid z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\left((x-\mu)^2 + e^{\ell}\right).
$$

At the true posterior, $\mu = 1$ and $e^{\ell} = 1/2$, so $(x-\mu)^2 + e^{\ell} = 3/2$. Half of $\log(2\pi)$ is $0.918939$, and half of $3/2$ is $0.75$, so the reconstruction term is $-0.918939 - 0.75 = -1.668939$. The KL against the standard normal is $0.596574$, so the bound is $-1.668939 - 0.596574 = -2.265513$. That matches $\log p(x) = -2.265512$ up to the last rounded digit. At full precision both equal $-2.265512123$. The bound touches.

Widen the bell and keep the mean: $\mu = 1$, variance $2$, so $\ell = \log 2$. The reconstruction term is $-2.418939$, the KL is $0.653426$, and the bound is $-3.072365$, strictly less than $-2.265512$. Returning the reconstruction term alone reports $-2.418939$ and misses the bound. Figure 2 holds the mean at 1. The curve touches $\log p(x)$ at variance $1/2$ and is lower at variance $2$.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="The ELBO peaks on the log probability at variance one half and is lower at variance two">
  <line x1="78" y1="36" x2="78" y2="300" class="stroke-muted"/>
  <line x1="78" y1="300" x2="570" y2="300" class="stroke-muted"/>
  <line x1="78.0" y1="67.6" x2="560.0" y2="67.6" class="stroke-ink dashed"/>
  <line x1="411.7" y1="67.6" x2="411.7" y2="185.9" class="stroke-muted dashed"/>
  <polyline points="81.7,86.7 87.2,81.8 92.6,78.0 98.1,74.9 103.6,72.6 109.0,70.8 114.5,69.4 120.0,68.5 125.4,67.9 130.9,67.6 136.4,67.6 141.9,67.9 147.3,68.3 152.8,69.0 158.3,69.8 163.7,70.8 169.2,71.9 174.7,73.2 180.1,74.6 185.6,76.1 191.1,77.7 196.5,79.4 202.0,81.2 207.5,83.1 212.9,85.0 218.4,87.1 223.9,89.2 229.3,91.3 234.8,93.5 240.3,95.8 245.7,98.2 251.2,100.6 256.7,103.0 262.1,105.5 267.6,108.0 273.1,110.6 278.5,113.2 284.0,115.9 289.5,118.6 294.9,121.3 300.4,124.1 305.9,126.9 311.3,129.7 316.8,132.6 322.3,135.4 327.8,138.4 333.2,141.3 338.7,144.3 344.2,147.3 349.6,150.3 355.1,153.3 360.6,156.4 366.0,159.5 371.5,162.6 377.0,165.7 382.4,168.8 387.9,172.0 393.4,175.2 398.8,178.4 404.3,181.6 409.8,184.8 415.2,188.1 420.7,191.3 426.2,194.6 431.6,197.9 437.1,201.2 442.6,204.5 448.0,207.8 453.5,211.2 459.0,214.6 464.4,217.9 469.9,221.3 475.4,224.7 480.8,228.1 486.3,231.5 491.8,235.0 497.3,238.4 502.7,241.9 508.2,245.3 513.7,248.8" fill="none" class="stroke-b" stroke-width="2.5"/>
  <circle cx="133.6" cy="67.6" r="5" class="fill-b"/>
  <circle cx="411.7" cy="185.9" r="5" class="fill-bad"/>
  <text x="250" y="24" class="fig-bold">bound against posterior variance</text>
  <text x="468" y="60" class="fig-small">log p(x) = -2.266</text>
  <text x="96" y="324" class="fig-small">variance 1/2, bound touches</text>
  <text x="360" y="324" class="fig-small">variance 2, bound is lower</text>
  <text x="422" y="140" class="fig-small">gap</text>
</svg>
Figure 2. With the posterior mean held at 1 and the row at 2, the bound touches log p(x) at variance 1/2 and is lower at variance 2. The dashed vertical segment is that gap.
```

| Posterior | Reconstruction term | KL | Bound |
|---|---|---|---|
| True, variance $1/2$ | $-1.668939$ | $0.596574$ | $-2.265512$ |
| Wider, variance $2$ | $-2.418939$ | $0.653426$ | $-3.072365$ |

In code you subtract the KL from the reconstruction term computed for this row:

```python
bound = reconstruction - gaussian_kl(mu_q, logvar_q)
```

> [!KEY]
> The ELBO equals $\log p(x)$ when $q$ is the true posterior, and it is strictly less for every other $q$. Dropping the KL reports a number that is too high.

## The penalty for leaving the standard normal

The penalty in the bound is the KL from your bell to the standard normal. Take $q = \mathrm{N}(\mu, \operatorname{diag}(e^{\ell}))$, a bell whose coordinates are independent, with mean $\mu_i$ and log-variance $\ell_i$ on coordinate $i$. The prior is $\mathrm{N}(0, I)$, the standard normal on every coordinate. The KL has a closed form, a sum and no integral:

$$
\operatorname{KL}(q \,\|\, p) = \frac{1}{2} \sum_i \left( \mu_i^2 + e^{\ell_i} - 1 - \ell_i \right).
$$

That is half the sum, over coordinates, of squared mean plus variance minus 1 minus log-variance. $\mu_i^2$ charges a mean away from 0. $e^{\ell_i} - 1 - \ell_i$ is 0 at $\ell_i = 0$ and positive otherwise. The sum does not average.

At $\mu = (0, 0)$ and $\ell = (0, 0)$ every term is 0, so the KL is 0. At $\mu = (1, 0)$ and $\ell = (0, 0)$ the first coordinate contributes 1 and the second contributes 0, so the KL is $1/2$. An average would report $1/4$.

At $\mu = (0, 0)$ and $\ell = (1, -1)$ the means contribute nothing, and the log-variance terms are the whole story:

$$
\frac{1}{2}\left(e^{1} + e^{-1} - 2\right) = \frac{1}{2}(2.718282 + 0.367879 - 2) = 0.543081.
$$

A version that drops the log-variance terms and keeps only $\tfrac{1}{2}\sum_i \mu_i^2$ returns 0 here. The two formulas agree whenever $\ell = 0$, which is why the origin does not catch the bug.

The derivative in $\mu_i$ is $\mu_i$. The derivative in $\ell_i$ is $\tfrac{1}{2}(e^{\ell_i} - 1)$, which is 0 at $\ell_i = 0$. In numpy:

```python
kl = 0.5 * np.sum(mu**2 + np.exp(logvar) - 1.0 - logvar)
grad_mu = mu
grad_logvar = 0.5 * (np.exp(logvar) - 1.0)
```

A central finite difference of `gaussian_kl` matches `grad_mu` and `grad_logvar`.

> [!WARNING]
> Dropping the log-variance terms leaves the KL unchanged at $\ell = 0$, and it returns 0 instead of $0.543$ at $\ell = (1, -1)$.

> [!TIP]
> Store the log-variance, not the variance. Then $e^{\ell/2}$ is positive for every finite $\ell$, and a gradient step cannot produce a negative width.

## A sample the gradient can follow

When the decoder bends, the average over codes has no closed form, so you draw one code and use it as a stand-in. `z = rng.normal(mu, exp(0.5 * logvar))` returns a number with no path back to $\mu$ and $\ell$, so a gradient step moves the decoder and leaves the encoder's bell fixed. Write the same draw as fixed noise plus a scale. Take $\varepsilon$ from a standard normal:

$$
z = \mu + e^{\ell/2}\, \varepsilon.
$$

Start at the mean and add the noise stretched by the standard deviation. Hold $\varepsilon$ fixed and the derivative in $\mu$ is 1. The derivative in $\ell$ is $\tfrac{1}{2} e^{\ell/2} \varepsilon$.

At $\varepsilon = 1$, $\mu = 0$, and $\ell = 0$, the sample is $0 + 1 \cdot 1 = 1$ and the log-variance derivative is $0.5$. At $\mu = 2$, $\ell = 0$, and $\varepsilon = 1$, the sample is 3. The noise term alone is 1, so returning only $e^{\ell/2}\varepsilon$ fails that point even though it agrees at $\mu = 0$.

```python
z = mu + np.exp(0.5 * logvar) * eps
d_mu = np.ones_like(mu)
d_logvar = 0.5 * np.exp(0.5 * logvar) * eps
```

A central finite difference of the sum of $z$, with $\varepsilon$ held fixed, matches these derivatives.

> [!KEY]
> A sample $z = \mu + e^{\ell/2}\varepsilon$ has a gradient in $\mu$ and in $\ell$. A sample drawn as a finished number does not.

## Train the squeeze, and the weight that silences the code

The encoder reads a row and returns $(\mu, \ell)$. The decoder reads a code and returns a mean. On this linear decoder the expectation is closed form, so a training step does not draw a code.

The exercise uses 40 rows in two dimensions, drawn from a generator seeded at 0, with means $(1.5, -0.8)$ and scales $(1.0, 0.55)$. This cloud has mean about $(1.566, -0.704)$ and variance about $(0.948, 0.273)$. The latent code has two coordinates. The encoder is an affine map to $\mu$ and another to $\ell$. The decoder is an affine map from the code, plus Gaussian noise of variance $0.05$. With decoded mean $m$ and spread $s_j = \sum_k W_{kj}^2 e^{\ell_k}$, the reconstruction cost of one row is

$$
\frac{1}{2}\sum_j \frac{(x_j - m_j)^2 + s_j}{0.05} + \log(2\pi \cdot 0.05).
$$

The fraction scales squared error and spread by the decoder noise, so a wide code blurs the rebuild even when $m$ sits on the row. Average the cost over the 40 rows and add $\beta$ times the average KL. At $\beta = 1$ this is the negative of the bound, with decoder noise $0.05$ rather than 1. A larger $\beta$ charges the KL more.

Adam, learning rate $0.08$, runs 40 steps. After that at $\beta = 1$, 400 samples from the prior through the decoder have mean $(0.845, -0.415)$ and variance $(0.790, 0.305)$. The mean is still short of the cloud, because the code still carries location and a prior draw has mean 0. The checks accept a sample mean inside $[0.3, 2.0]$ and $[-1.2, -0.1]$, and a sample variance inside $[0.35, 1.5]$ and $[0.12, 0.70]$.

The same 40 steps at $\beta = 20$ raise the reconstruction cost from $0.910$ to $10.742$ and drop the KL from $2.722$ to $0.035$. Sample variance falls to about $(0.083, 0.054)$, under the floor the check uses. The encoder sits on the prior and the decoder stops using the code. That direction is posterior collapse.

> [!WARNING]
> A large weight on the KL lowers the KL and raises the reconstruction cost. The samples then shrink toward the decoder noise, and the code stops describing the row.

> [!KEY]
> You compress a row by squeezing it onto the long axis of the cloud, and you draw a row by sampling a standard normal code and decoding it. The bound you climb touches the log probability only when the encoder's bell is the true posterior.

## Cost and scaling

The linear autoencoder is one SVD of the centred matrix, and the KL is one pass over the latent coordinates. A bent decoder on images costs a forward and a backward pass per step. The reparameterisation adds only the scale-and-shift.

## Where this breaks

- **The weight on the KL is large.** Then $q$ collapses to the prior, the KL looks small, and the reconstructions get worse. Lower the weight until the code still moves the rebuild.
- **The decoder is a straight line and the cloud is not an ellipse.** Samples fill an ellipse. A ring needs a bend in the decoder, and then the reparameterised sample carries the gradient to the encoder.
- **The prior is not a standard normal, or the bell is not diagonal.** The closed-form sum is for this pair. A full covariance needs a different KL.
- **You forget to centre.** The leading singular vector of the raw matrix points along the average row, which on the worked matrix is orthogonal to the principal component.

> [!RECAP]
> - Centre, then take the leading right singular vectors. On rows $(3, 1)$ and $(1, 3)$ that direction is $(1, -1)/\sqrt{2}$, up to sign. The average row points along $(1, 1)$.
> - The diagonal-Gaussian KL against a standard normal is $\tfrac{1}{2}\sum_i(\mu_i^2 + e^{\ell_i} - 1 - \ell_i)$. It is 0 at the origin and $1/2$ at $\mu = (1, 0)$, $\ell = 0$.
> - The reparameterised sample is $\mu + e^{\ell/2}\varepsilon$. At $\varepsilon = 1$, $\mu = 0$, $\ell = 0$, the sample is 1 and the log-variance derivative is $0.5$.
> - On the linear-Gaussian model at $x = 2$, the bound equals $\log p(x) = -2.265512$ at the true posterior and equals $-3.072365$ at variance 2.
> - Forty Adam steps at $\beta = 1$ put decoder samples near the cloud. At $\beta = 20$ the reconstruction cost rises and the KL falls.

## Build it step by step

The KL checks come first, then the sample, then the bound, then the decoder, then the training run.

1. `gaussian_kl(mu, logvar)` returns $\tfrac{1}{2}\sum(\mu^2 + e^{\ell} - 1 - \ell)$, summed over every element. The starter already does this. The value is 0 at the origin and $0.5$ at $\mu = (1, 0)$, $\ell = (0, 0)$. Averaging over coordinates reports $0.25$ there. Dropping the log-variance terms, `kl_without_logvar`, reports 0 at $\ell = (1, -1)$ where the value is $0.543081$.

2. `gaussian_kl_grad(mu, logvar)` returns `mu` and `0.5 * (exp(logvar) - 1)`, each the same shape as `mu`. A central finite difference of `gaussian_kl` matches both. Differentiating $e^{\ell}$ and forgetting the factor $1/2$ does not.

3. `reparameterise(mu, logvar, eps)` returns `mu + exp(0.5 * logvar) * eps`. At `eps = 1`, `mu = 0`, `logvar = 0` the sample is 1. At `mu = 2` it is 3. Returning only the scaled noise, `sample_without_mu`, is 1 at both points.

4. `reparameterise_grad(mu, logvar, eps)` returns ones, and `0.5 * exp(0.5 * logvar) * eps`. At the worked point the second array is $0.5$. A central finite difference of the sum, with `eps` fixed, matches both. Dropping the $1/2$ in the exponent uses the variance where the standard deviation belongs.

5. `elbo(x, mu_q, logvar_q, reconstruction)` returns the reconstruction term minus `gaussian_kl(mu_q, logvar_q)`. On $x = 2$ the bound equals $-2.265512$ at the true posterior and $-3.072365$ at variance 2. Returning the reconstruction term alone, `reconstruction_only`, reports $-2.418939$ on the wider bell.

6. `linear_autoencoder_fit(X, rank)` returns a decoder of shape `(d, rank)`: centre the rows and take the first `rank` right singular vectors as columns. On rows $(3, 1)$ and $(1, 3)$ the column is parallel to $(1, -1)/\sqrt{2}$, either sign. Using the mean of the rows, `mean_direction`, points along $(1, 1)$ and is orthogonal to that column.

7. `train_vae(X, steps, seed, beta=1.0)` returns 400 decoder samples of shape `(400, 2)`, the mean reconstruction cost, and the mean KL. Draw the initial weights from `default_rng(seed)` in this order: $\mu$ weights at scale $0.1$, $\mu$ bias 0, log-variance weights at scale $0.05$, log-variance bias $-0.5$, decoder weights at scale $0.1$, decoder bias 0. Use the latent size, noise variance, and Adam settings from the training section. Draw the 400 codes from `default_rng(seed + 1)`, decode them, and add noise of standard deviation $\sqrt{0.05}$. Treating a sampled code as a constant leaves the encoder fixed. On the seeded cloud, 40 steps at `beta = 1` land inside the window in the training section. At `beta = 20` the reconstruction cost is higher by more than 1 and the KL is lower by more than $0.5$.

## The exercise

You implement `gaussian_kl_grad`, `reparameterise`, `reparameterise_grad`, `elbo`, `linear_autoencoder_fit`, and `train_vae`. The file already contains `gaussian_kl`. The checks walk the worked values in that order: the KL at 0 and at $0.5$, the finite-difference gradient, the samples 1 and 3, the bound against $\log p(x)$, the principal-component column, and a 40-step training run at `beta = 1` and at `beta = 20`.

## Further reading

These are the lectures and the paper this chapter follows. Lecture numbers appear only where the page prints them.

- For the reparameterised lower bound and the encoder that amortises the posterior, see [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma and Welling).
- For the evidence lower bound on a latent-variable model, and why it touches when $q$ is the posterior, see [CS236 lecture 5, Latent variable models](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture5.pdf) (Stefano Ermon). The slide prints "Lecture 5".
- For the reparameterisation trick and amortised inference, see [CS236 lecture 6, Latent variable models](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture6.pdf) (Stefano Ermon). The slide prints "Lecture 6".
- For variational autoencoders in a vision course, see [CS231n lecture 13, Generative models 1](https://cs231n.stanford.edu/2026/schedule.html). The schedule prints "Lecture 13: Generative Models 1" and lists variational autoencoders.
- For a picture of the encoder, the decoder, and a reparameterised sample, see [Deep generative models](https://introtodeeplearning.com/2019/materials/2019_6S191_L4.pdf) (Alexander Amini, MIT 6.S191, 2019). The [2019 schedule](https://introtodeeplearning.com/2019/) lists this talk as session 2, part 2.
