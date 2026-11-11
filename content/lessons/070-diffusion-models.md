---
title: "Diffusion models: add noise, then walk it back"
slug: 070-diffusion-models
part: "Part 8 — Generative models"
summary: "Noise a two-blob cloud on a schedule, predict that noise, and walk a standard normal back onto the blobs."
minutes: 75
gpu: false
objectives:
  - Write the closed-form noisy point from the clean point, one standard normal, and the cumulative product of the schedule.
  - Connect the noise prediction to the score, and show that the minimum-MSE predictor on a standard normal has expected squared error equal to alpha-bar.
  - Take one ancestral step and one deterministic step from the same noise prediction.
  - Draw samples with the exact denoiser of a two-bell mixture and land inside the stated window of its mean and covariance.
  - Extrapolate a conditional noise prediction past the unconditional one.
prereqs:
  - 031-gaussian-mixtures-and-em
  - 032-multilayer-perceptrons
  - 069-autoencoders-and-vaes
exercise: 070-diffusion-models
---

# Diffusion models: add noise, then walk it back

You marked two kinds of flower on a map. One kind grows on the west side of a field and the other on the east side, and each patch is a blob of dots. You want another dot that could have been one of those flowers: on a blob, not in the empty middle, and not out in the grass where nothing grows.

The chapter on [variational autoencoders](/lessons/069-autoencoders-and-vaes) draws a code from a standard bell and decodes it. This chapter skips the decoder. Add Gaussian noise until the two blobs melt into one bell, then walk a draw from that bell backwards until it sits on a blob. One jump from pure noise fails: the clean dot and the noise share almost no information, so the guess averages the blobs into the empty middle. A short step stays local, because a slightly noisy dot still sits near the dot it came from.

## What you already know

- A standard normal draw has mean 0 and variance 1 on each coordinate.
- Independent Gaussian noise adds its variance to the variance of the point it lands on.
- A variational autoencoder draws a code from a standard normal and decodes it. This chapter starts from that same bell and walks back.
- A mixture of two bells gives each point a responsibility for each bell, as in the chapter on [Gaussian mixtures](/lessons/031-gaussian-mixtures-and-em).
- A small [multilayer perceptron](/lessons/032-multilayer-perceptrons) is an affine map, a bend such as a ReLU, and another affine map. A training step follows the gradient of a scalar loss.

## Melt the two blobs into one bell

The west blob is a bell centered at $(-1.5, 0)$ and the east blob is a bell centered at $(1.5, 0)$. Both have variance 1 on each coordinate, and a dot comes from either bell with equal probability. The mean of the whole cloud is $(0, 0)$. The east-west variance is the variance inside a blob plus the spread of the centers: $1 + 1.5^2 = 1 + 2.25 = 3.25$. The north-south variance is only the variance inside a blob, which is 1, because the two centers share a $y$ coordinate. The covariance is diagonal, with entries 3.25 and 1.

```figure
<svg viewBox="0 0 640 330" role="img" aria-label="Two blobs of points, and the same points after three closed-form noise steps">
  <line x1="48" y1="166" x2="292" y2="166" class="stroke-muted"/>
  <line x1="170" y1="46" x2="170" y2="286" class="stroke-muted"/>
  <line x1="356" y1="166" x2="600" y2="166" class="stroke-muted"/>
  <line x1="478" y1="46" x2="478" y2="286" class="stroke-muted"/>
<circle cx="136.5" cy="169.8" r="3.4" class="fill-a"/>
  <circle cx="149.0" cy="163.0" r="3.4" class="fill-a"/>
  <circle cx="120.3" cy="155.7" r="3.4" class="fill-a"/>
  <circle cx="165.2" cy="138.9" r="3.4" class="fill-a"/>
  <circle cx="116.2" cy="202.2" r="3.4" class="fill-a"/>
  <circle cx="118.2" cy="164.8" r="3.4" class="fill-a"/>
  <circle cx="76.7" cy="172.3" r="3.4" class="fill-a"/>
  <circle cx="103.0" cy="186.9" r="3.4" class="fill-a"/>
  <circle cx="120.1" cy="175.0" r="3.4" class="fill-a"/>
  <circle cx="143.4" cy="136.2" r="3.4" class="fill-a"/>
  <circle cx="130.3" cy="127.0" r="3.4" class="fill-a"/>
  <circle cx="117.2" cy="156.0" r="3.4" class="fill-a"/>
  <circle cx="155.4" cy="163.3" r="3.4" class="fill-a"/>
  <circle cx="115.3" cy="192.3" r="3.4" class="fill-a"/>
  <circle cx="122.2" cy="159.7" r="3.4" class="fill-a"/>
  <circle cx="108.8" cy="172.0" r="3.4" class="fill-a"/>
  <circle cx="129.5" cy="150.5" r="3.4" class="fill-a"/>
  <circle cx="138.6" cy="155.8" r="3.4" class="fill-a"/>
  <circle cx="117.4" cy="169.7" r="3.4" class="fill-a"/>
  <circle cx="152.5" cy="123.3" r="3.4" class="fill-a"/>
  <circle cx="102.7" cy="122.7" r="3.4" class="fill-a"/>
  <circle cx="166.2" cy="143.7" r="3.4" class="fill-a"/>
  <circle cx="139.9" cy="175.0" r="3.4" class="fill-a"/>
  <circle cx="169.0" cy="110.0" r="3.4" class="fill-a"/>
  <circle cx="177.4" cy="128.4" r="3.4" class="fill-a"/>
  <circle cx="142.1" cy="200.5" r="3.4" class="fill-a"/>
  <circle cx="133.3" cy="147.2" r="3.4" class="fill-a"/>
  <circle cx="102.0" cy="154.7" r="3.4" class="fill-a"/>
  <circle cx="143.9" cy="146.1" r="3.4" class="fill-a"/>
  <circle cx="104.5" cy="184.9" r="3.4" class="fill-a"/>
  <circle cx="122.8" cy="199.4" r="3.4" class="fill-a"/>
  <circle cx="175.8" cy="180.2" r="3.4" class="fill-a"/>
  <circle cx="141.4" cy="173.4" r="3.4" class="fill-a"/>
  <circle cx="172.0" cy="128.3" r="3.4" class="fill-a"/>
  <circle cx="148.9" cy="229.0" r="3.4" class="fill-a"/>
  <circle cx="134.7" cy="146.5" r="3.4" class="fill-a"/>
  <circle cx="231.1" cy="183.7" r="3.4" class="fill-d"/>
  <circle cx="251.1" cy="203.7" r="3.4" class="fill-d"/>
  <circle cx="190.5" cy="139.3" r="3.4" class="fill-d"/>
  <circle cx="207.8" cy="108.8" r="3.4" class="fill-d"/>
  <circle cx="211.2" cy="184.1" r="3.4" class="fill-d"/>
  <circle cx="197.4" cy="197.2" r="3.4" class="fill-d"/>
  <circle cx="175.4" cy="148.0" r="3.4" class="fill-d"/>
  <circle cx="220.8" cy="129.0" r="3.4" class="fill-d"/>
  <circle cx="188.2" cy="117.7" r="3.4" class="fill-d"/>
  <circle cx="199.6" cy="121.0" r="3.4" class="fill-d"/>
  <circle cx="196.0" cy="187.0" r="3.4" class="fill-d"/>
  <circle cx="212.7" cy="136.5" r="3.4" class="fill-d"/>
  <circle cx="210.5" cy="182.7" r="3.4" class="fill-d"/>
  <circle cx="173.9" cy="206.0" r="3.4" class="fill-d"/>
  <circle cx="218.9" cy="137.7" r="3.4" class="fill-d"/>
  <circle cx="202.6" cy="196.7" r="3.4" class="fill-d"/>
  <circle cx="227.9" cy="202.6" r="3.4" class="fill-d"/>
  <circle cx="189.2" cy="148.3" r="3.4" class="fill-d"/>
  <circle cx="151.7" cy="155.0" r="3.4" class="fill-d"/>
  <circle cx="192.4" cy="162.9" r="3.4" class="fill-d"/>
  <circle cx="204.8" cy="160.2" r="3.4" class="fill-d"/>
  <circle cx="223.5" cy="187.7" r="3.4" class="fill-d"/>
  <circle cx="241.3" cy="145.3" r="3.4" class="fill-d"/>
  <circle cx="227.2" cy="132.7" r="3.4" class="fill-d"/>
  <circle cx="225.8" cy="141.9" r="3.4" class="fill-d"/>
  <circle cx="208.4" cy="206.8" r="3.4" class="fill-d"/>
  <circle cx="203.3" cy="188.0" r="3.4" class="fill-d"/>
  <circle cx="171.9" cy="158.6" r="3.4" class="fill-d"/>
  <circle cx="192.7" cy="195.4" r="3.4" class="fill-d"/>
  <circle cx="181.2" cy="158.3" r="3.4" class="fill-d"/>
  <circle cx="215.4" cy="128.2" r="3.4" class="fill-d"/>
  <circle cx="206.3" cy="136.2" r="3.4" class="fill-d"/>
  <circle cx="240.8" cy="133.1" r="3.4" class="fill-d"/>
  <circle cx="148.9" cy="130.9" r="3.4" class="fill-d"/>
  <circle cx="214.9" cy="153.9" r="3.4" class="fill-d"/>
  <circle cx="215.7" cy="155.1" r="3.4" class="fill-d"/>
  <circle cx="461.8" cy="176.2" r="3.4" class="fill-a"/>
  <circle cx="429.2" cy="166.4" r="3.4" class="fill-a"/>
  <circle cx="430.6" cy="136.0" r="3.4" class="fill-a"/>
  <circle cx="469.5" cy="146.4" r="3.4" class="fill-a"/>
  <circle cx="427.0" cy="200.8" r="3.4" class="fill-a"/>
  <circle cx="443.7" cy="197.2" r="3.4" class="fill-a"/>
  <circle cx="422.2" cy="172.4" r="3.4" class="fill-a"/>
  <circle cx="412.2" cy="231.4" r="3.4" class="fill-a"/>
  <circle cx="454.7" cy="178.3" r="3.4" class="fill-a"/>
  <circle cx="450.8" cy="151.5" r="3.4" class="fill-a"/>
  <circle cx="485.3" cy="141.4" r="3.4" class="fill-a"/>
  <circle cx="444.9" cy="191.4" r="3.4" class="fill-a"/>
  <circle cx="498.7" cy="144.5" r="3.4" class="fill-a"/>
  <circle cx="461.6" cy="182.3" r="3.4" class="fill-a"/>
  <circle cx="463.5" cy="153.9" r="3.4" class="fill-a"/>
  <circle cx="449.0" cy="173.2" r="3.4" class="fill-a"/>
  <circle cx="424.3" cy="133.7" r="3.4" class="fill-a"/>
  <circle cx="421.8" cy="164.5" r="3.4" class="fill-a"/>
  <circle cx="439.7" cy="190.9" r="3.4" class="fill-a"/>
  <circle cx="477.8" cy="142.3" r="3.4" class="fill-a"/>
  <circle cx="425.7" cy="126.4" r="3.4" class="fill-a"/>
  <circle cx="466.8" cy="121.4" r="3.4" class="fill-a"/>
  <circle cx="464.6" cy="182.1" r="3.4" class="fill-a"/>
  <circle cx="441.6" cy="157.3" r="3.4" class="fill-a"/>
  <circle cx="502.8" cy="142.4" r="3.4" class="fill-a"/>
  <circle cx="454.5" cy="153.3" r="3.4" class="fill-a"/>
  <circle cx="430.3" cy="166.3" r="3.4" class="fill-a"/>
  <circle cx="424.6" cy="146.0" r="3.4" class="fill-a"/>
  <circle cx="448.6" cy="166.1" r="3.4" class="fill-a"/>
  <circle cx="405.4" cy="162.7" r="3.4" class="fill-a"/>
  <circle cx="461.8" cy="198.2" r="3.4" class="fill-a"/>
  <circle cx="484.8" cy="203.1" r="3.4" class="fill-a"/>
  <circle cx="450.5" cy="141.2" r="3.4" class="fill-a"/>
  <circle cx="481.8" cy="91.5" r="3.4" class="fill-a"/>
  <circle cx="449.6" cy="194.9" r="3.4" class="fill-a"/>
  <circle cx="451.2" cy="141.0" r="3.4" class="fill-a"/>
  <circle cx="518.0" cy="189.7" r="3.4" class="fill-d"/>
  <circle cx="515.3" cy="124.8" r="3.4" class="fill-d"/>
  <circle cx="490.0" cy="191.9" r="3.4" class="fill-d"/>
  <circle cx="490.9" cy="113.8" r="3.4" class="fill-d"/>
  <circle cx="495.9" cy="148.6" r="3.4" class="fill-d"/>
  <circle cx="514.4" cy="189.8" r="3.4" class="fill-d"/>
  <circle cx="472.9" cy="175.8" r="3.4" class="fill-d"/>
  <circle cx="498.5" cy="173.4" r="3.4" class="fill-d"/>
  <circle cx="512.1" cy="100.0" r="3.4" class="fill-d"/>
  <circle cx="474.3" cy="161.9" r="3.4" class="fill-d"/>
  <circle cx="462.6" cy="200.6" r="3.4" class="fill-d"/>
  <circle cx="448.9" cy="171.2" r="3.4" class="fill-d"/>
  <circle cx="528.5" cy="184.4" r="3.4" class="fill-d"/>
  <circle cx="496.3" cy="202.8" r="3.4" class="fill-d"/>
  <circle cx="542.5" cy="143.1" r="3.4" class="fill-d"/>
  <circle cx="492.4" cy="131.2" r="3.4" class="fill-d"/>
  <circle cx="510.1" cy="216.3" r="3.4" class="fill-d"/>
  <circle cx="494.3" cy="155.2" r="3.4" class="fill-d"/>
  <circle cx="485.6" cy="178.6" r="3.4" class="fill-d"/>
  <circle cx="507.5" cy="145.6" r="3.4" class="fill-d"/>
  <circle cx="488.6" cy="158.7" r="3.4" class="fill-d"/>
  <circle cx="497.9" cy="129.7" r="3.4" class="fill-d"/>
  <circle cx="511.9" cy="162.1" r="3.4" class="fill-d"/>
  <circle cx="496.0" cy="151.6" r="3.4" class="fill-d"/>
  <circle cx="514.6" cy="133.6" r="3.4" class="fill-d"/>
  <circle cx="492.0" cy="196.8" r="3.4" class="fill-d"/>
  <circle cx="473.8" cy="198.3" r="3.4" class="fill-d"/>
  <circle cx="529.9" cy="138.7" r="3.4" class="fill-d"/>
  <circle cx="478.6" cy="214.1" r="3.4" class="fill-d"/>
  <circle cx="467.4" cy="161.4" r="3.4" class="fill-d"/>
  <circle cx="508.4" cy="157.2" r="3.4" class="fill-d"/>
  <circle cx="478.2" cy="115.8" r="3.4" class="fill-d"/>
  <circle cx="532.9" cy="152.5" r="3.4" class="fill-d"/>
  <circle cx="460.1" cy="154.3" r="3.4" class="fill-d"/>
  <circle cx="453.5" cy="155.6" r="3.4" class="fill-d"/>
  <circle cx="488.3" cy="180.4" r="3.4" class="fill-d"/>
  <text x="48" y="22" class="fig-bold">clean cloud</text>
  <text x="356" y="22" class="fig-bold">after three steps</text>
  <text x="48" y="312" class="fig-small">west</text>
  <text x="240" y="312" class="fig-small">east</text>
  <text x="356" y="312" class="fig-small">alpha-bar = 0.432</text>
</svg>
Figure 1. The west blob is centered at −1.5 and the east blob at 1.5. After three steps of the worked schedule, alpha-bar is 0.432, the centers have moved in to ±0.986, and the two colors overlap more.
```

Call the clean dot $x_0$ and the dot after $t$ steps of noise $x_t$. One step shrinks the dot and adds Gaussian noise:

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, z.
$$

$z$ is a fresh standard normal, and $\beta_t$ is the noise added at step $t$. The schedule is the list $\beta_1, \ldots, \beta_T$.

You draw $x_t$ from $x_0$ in one shot. Write $\alpha_t = 1 - \beta_t$ and let $\bar\alpha_t$ be the product of those factors up to $t$:

$$
\bar\alpha_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s).
$$

$\bar\alpha_t$ is the fraction of the clean dot that is still visible. It starts near 1 and falls toward 0. The closed form is

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon.
$$

$\varepsilon$ is one standard normal. The coefficients square-sum to 1, so conditional on $x_0$ the mean is $\sqrt{\bar\alpha_t}\, x_0$ and the variance is $1 - \bar\alpha_t$.

> [!KEY]
> The noisy dot is the clean dot times $\sqrt{\bar\alpha_t}$, plus one standard normal times $\sqrt{1 - \bar\alpha_t}$. You draw it without walking the intermediate steps.

## One dot, worked by hand

Take the schedule $\beta = (0.1, 0.2, 0.4)$. Then $\alpha = (0.9, 0.8, 0.6)$, and the cumulative product is

$$
\bar\alpha_1 = 0.9, \qquad \bar\alpha_2 = 0.9 \times 0.8 = 0.72, \qquad \bar\alpha_3 = 0.72 \times 0.6 = 0.432.
$$

Returning $1 - \beta$ and stopping there matches 0.9 at the first step and returns 0.8 where the product is 0.72. The product runs over every step, not the factor from that step alone.

Use the east center as the clean dot, $x_0 = (1.5, 0)$, and pick $\varepsilon = (1, -1)$ by hand. At $t = 2$, $\sqrt{0.72} = 0.848528$ and $\sqrt{0.28} = 0.529150$. The signal and the noise are

$$
0.848528 \times (1.5, 0) = (1.272792, 0),
$$

$$
0.529150 \times (1, -1) = (0.529150, -0.529150).
$$

Add them:

$$
x_2 = (1.801942, -0.529150).
$$

At $t = 3$ the centers move from $\pm 1.5$ to $\pm \sqrt{0.432} \times 1.5 = \pm 0.985901$, and the added noise has standard deviation $\sqrt{0.568} = 0.753658$.

A longer schedule drives $\bar\alpha_T$ near 0, which is what makes the final bell a standard normal no matter where $x_0$ sat. Eighty values spaced from $0.0001$ to $0.1$ give $\bar\alpha_{80} = 0.015835$. One thousand values spaced from $0.0001$ to $0.02$ give $\bar\alpha_{1000} = 4.036 \times 10^{-5}$. The check asks for that last product to sit in $(0, 10^{-4})$.

Four thousand standard normals at seed 0, with this $x_0$ and $t = 2$, have mean $(1.271960, 0.001454)$ and variances $(0.285444, 0.277660)$. The closed form says the mean is $(1.272792, 0)$ and the variance is 0.28.

> [!WARNING]
> Dropping $\sqrt{\bar\alpha_t}$ on the clean dot leaves the Monte Carlo mean at $x_0$. The variance is still $1 - \bar\alpha_t$, so a variance check does not catch it.

> [!TIP]
> Count $t$ from 1, and read the cumulative product at index $t - 1$. An off-by-one matches the product at neither of two different times.

## Predict the noise, and read it as a score

You observe $x_t$ and $t$, and you want the noise $\varepsilon$ from the closed form. A network $\varepsilon_\theta(x_t, t)$ is trained to match it. The loss on one pair is the mean of the squared coordinates:

$$
L = \operatorname{mean}\left( (\varepsilon - \varepsilon_\theta(x_t, t))^2 \right).
$$

Averaging $L$ over the clean dot, the time, and $\varepsilon$ is denoising score matching, up to a weight that depends only on $t$.

The score of a density is the gradient of its log. Conditional on $x_0$, the noisy dot is normal with mean $\sqrt{\bar\alpha_t}\, x_0$ and variance $1 - \bar\alpha_t$, so

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\, x_0}{1 - \bar\alpha_t}.
$$

The numerator is $\sqrt{1 - \bar\alpha_t}\, \varepsilon$, and the score simplifies to $-\varepsilon / \sqrt{1 - \bar\alpha_t}$. At the worked point, $\sqrt{0.28} = 0.529150$ and $\varepsilon = (1, -1)$, so the conditional score is $(-1.889822, 1.889822)$.

> [!KEY]
> The score implied by a noise prediction is that prediction divided by $-\sqrt{1 - \bar\alpha_t}$. The squared score gap is the squared noise gap divided by $1 - \bar\alpha_t$, so at a fixed time the two losses share a minimizer.

## The minimum-MSE predictor on one bell

The network sees $x_t$, not $x_0$, so it cannot return the particular $\varepsilon$ of this sample unless that noise is a function of $x_t$. The predictor with the smallest expected squared error is the mean of $\varepsilon$ over every clean dot that could have produced $x_t$.

Take the cloud to be a standard normal, so $x_0 \sim \mathrm{N}(0, I)$. Then $x_t$ is standard normal too, because its variance is $\bar\alpha_t \cdot 1 + (1 - \bar\alpha_t) = 1$. The minimum-MSE noise prediction is

$$
\varepsilon^\star(x_t) = \sqrt{1 - \bar\alpha_t}\, x_t.
$$

The score it implies is $-x_t$, the score of the standard normal. Write the residual and expand $x_t$:

$$
\varepsilon - \sqrt{1 - \bar\alpha_t}\, x_t = \bar\alpha_t\, \varepsilon - \sqrt{\bar\alpha_t(1 - \bar\alpha_t)}\, x_0.
$$

Square this and take the expectation. The cross term vanishes because $\varepsilon$ and $x_0$ are independent and both have mean 0. What remains is $\bar\alpha_t^2 + \bar\alpha_t(1 - \bar\alpha_t) = \bar\alpha_t$. The expected squared error on each coordinate equals $\bar\alpha_t$. At $t = 2$ that value is 0.72. Four thousand fresh pairs from `default_rng(0)` average to 0.718339.

On the worked pair, one chosen dot rather than a draw from that bell, the same formula returns

$$
\varepsilon^\star = 0.529150 \times (1.801942, -0.529150) = (0.953498, -0.280000).
$$

The second coordinate is $-0.28$ exactly, because $x_0$ has $y = 0$ and $\sqrt{0.28}^2 = 0.28$. The two squared coordinates are $0.046502^2 = 0.002162$ and $(-0.72)^2 = 0.518400$, and their mean is 0.260281. Swapping the scales, and returning $\sqrt{\bar\alpha_t}\, x_t$ instead, scores 0.291721 on this pair. The gap is small, so the check uses the exact value.

The gradient check uses a linear denoiser, $\hat\varepsilon = w \odot x_t$, one weight per coordinate. On one row the 2 from the square cancels the 2 from the mean:

$$
\frac{\partial L}{\partial w_j} = -\, r_j\, x_{t,j}, \qquad r = \varepsilon - w \odot x_t.
$$

At $w = (0.4, 0.1)$ the prediction is $(0.720777, -0.052915)$, the residual is $(0.279223, -0.947085)$, the loss is 0.487468, and the gradient is $(-0.503144, -0.501150)$. A central difference of $L$ matches that gradient. With $n$ rows and $d$ coordinates the same derivative is $-2/(nd)$ times the sum of residual times $x_t$ on that coordinate.

## Walk back one step

Sampling starts at $x_T \sim \mathrm{N}(0, I)$, where the product $\bar\alpha_T$ is near 0, and steps $t = T, \ldots, 1$. Given a prediction $\varepsilon_\theta$, the mean of the reverse step is

$$
\mu = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \varepsilon_\theta \right).
$$

For $t > 1$, add a fresh standard normal scaled by $\sqrt{\tilde\beta_t}$, with

$$
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t.
$$

Set $\bar\alpha_0 = 1$. Then $\tilde\beta_1 = 0$, and the last step is the mean alone.

On the worked step, $t = 2$, hand the formula the true $\varepsilon = (1, -1)$ and the fresh noise $(0.5, -0.25)$. Here $\alpha_2 = 0.8$, $\beta_2 = 0.2$, $\bar\alpha_2 = 0.72$, and $\bar\alpha_1 = 0.9$, so

$$
\tilde\beta_2 = \frac{0.1}{0.28} \times 0.2 = \frac{1}{14} = 0.071429, \qquad \sqrt{\tilde\beta_2} = 0.267261.
$$

The mean is $(1.592056, -0.169031)$. Adding the scaled noise gives the step $(1.725686, -0.235846)$. Using $\sqrt{\beta_t}$ in place of $\sqrt{\tilde\beta_t}$ changes the scale from 0.267261 to $\sqrt{0.2} = 0.447214$, and the landing point moves.

> [!KEY]
> Ancestral sampling starts at a standard normal. Each step removes noise with the reverse mean, and every step except the last adds noise of variance $\tilde\beta_t$.

## Take the same step without fresh noise

A deterministic step predicts the clean dot and then rebuilds the previous time with the same noise prediction, and with no extra normal. Song, Meng, and Ermon write it, in the notation of this chapter, as

$$
\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \varepsilon_\theta}{\sqrt{\bar\alpha_t}},
$$

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{t-1}}\, \varepsilon_\theta.
$$

With $\bar\alpha_0 = 1$ the last step returns $\hat x_0$.

Hand this step the worked $x_2$ and the true $\varepsilon$. Then $\hat x_0$ is $(1.5, 0)$, the clean dot, and

$$
x_1 = \sqrt{0.9}\,(1.5, 0) + \sqrt{0.1}\,(1, -1) = (1.739253, -0.316228).
$$

Given the true noise, that vector is also the closed form at $t = 1$. Returning only $\hat x_0$ gives $(1.5, 0)$ and misses it.

## A small network on the two blobs

The exact denoiser for the two blobs is Gaussian conditioning. At time $t$ blob $k$ is centered at $\sqrt{\bar\alpha_t}\, \mu_k$ and still has variance 1, because $\bar\alpha_t + (1 - \bar\alpha_t) = 1$. Responsibilities are a softmax of $-\tfrac{1}{2}\lVert x_t - \sqrt{\bar\alpha_t}\, \mu_k \rVert^2$. The minimum-MSE clean dot is $(1 - \bar\alpha_t)$ times those weighted centers, plus $\sqrt{\bar\alpha_t}\, x_t$. The noise prediction divides $x_t - \sqrt{\bar\alpha_t}$ times that dot by $\sqrt{1 - \bar\alpha_t}$.

The figure script trains a stand-in with one hidden layer of width 48 and a ReLU. The inputs are $x_t$, $\sqrt{\bar\alpha_t}$, and $\sqrt{1 - \bar\alpha_t}$. Adam uses $\beta_1 = 0.9$, $\beta_2 = 0.999$, and learning rate 0.03, for 1500 steps of batch 256 from seed 0, on 40 betas spaced from 0.0001 to 0.2. The first loss is 1.660 and the mean of the last 20 losses is 0.403. Two thousand ancestral samples from that network, seed 1, have mean $(-0.0089, 0.0034)$ and covariance entries 2.976, 0.014, and 0.857. The target is $\operatorname{diag}(3.25, 1)$. The east-west spread is close, and the north-south spread is still short. The exercise grades the exact denoiser, where that target is known.

```figure
<svg viewBox="0 0 640 340" role="img" aria-label="Ancestral samples from the exact denoiser sitting on the two blobs">
  <line x1="78" y1="167" x2="560" y2="167" class="stroke-muted"/>
  <line x1="319" y1="42" x2="319" y2="292" class="stroke-muted"/>
  <circle cx="345.1" cy="255.7" r="3.4" class="fill-soft-a"/>
  <circle cx="266.9" cy="186.7" r="3.4" class="fill-soft-a"/>
  <circle cx="224.9" cy="174.5" r="3.4" class="fill-soft-a"/>
  <circle cx="149.3" cy="175.1" r="3.4" class="fill-soft-a"/>
  <circle cx="205.0" cy="51.6" r="3.4" class="fill-soft-a"/>
  <circle cx="257.6" cy="179.2" r="3.4" class="fill-soft-a"/>
  <circle cx="233.1" cy="190.2" r="3.4" class="fill-soft-a"/>
  <circle cx="195.8" cy="180.6" r="3.4" class="fill-soft-a"/>
  <circle cx="269.9" cy="175.3" r="3.4" class="fill-soft-a"/>
  <circle cx="292.9" cy="173.9" r="3.4" class="fill-soft-a"/>
  <circle cx="247.9" cy="113.3" r="3.4" class="fill-soft-a"/>
  <circle cx="273.0" cy="184.5" r="3.4" class="fill-soft-a"/>
  <circle cx="237.9" cy="148.2" r="3.4" class="fill-soft-a"/>
  <circle cx="340.0" cy="176.4" r="3.4" class="fill-soft-a"/>
  <circle cx="235.0" cy="132.2" r="3.4" class="fill-soft-a"/>
  <circle cx="204.0" cy="177.1" r="3.4" class="fill-soft-a"/>
  <circle cx="289.2" cy="146.8" r="3.4" class="fill-soft-a"/>
  <circle cx="251.1" cy="143.7" r="3.4" class="fill-soft-a"/>
  <circle cx="110.4" cy="131.5" r="3.4" class="fill-soft-a"/>
  <circle cx="200.4" cy="224.9" r="3.4" class="fill-soft-a"/>
  <circle cx="260.0" cy="142.7" r="3.4" class="fill-soft-a"/>
  <circle cx="225.3" cy="204.4" r="3.4" class="fill-soft-a"/>
  <circle cx="248.0" cy="168.8" r="3.4" class="fill-soft-a"/>
  <circle cx="314.4" cy="141.0" r="3.4" class="fill-soft-a"/>
  <circle cx="256.0" cy="128.4" r="3.4" class="fill-soft-a"/>
  <circle cx="236.8" cy="199.1" r="3.4" class="fill-soft-a"/>
  <circle cx="274.9" cy="146.8" r="3.4" class="fill-soft-a"/>
  <circle cx="236.3" cy="194.2" r="3.4" class="fill-soft-a"/>
  <circle cx="257.7" cy="253.6" r="3.4" class="fill-soft-a"/>
  <circle cx="280.0" cy="149.9" r="3.4" class="fill-soft-a"/>
  <circle cx="312.3" cy="164.9" r="3.4" class="fill-soft-d"/>
  <circle cx="344.8" cy="140.7" r="3.4" class="fill-soft-d"/>
  <circle cx="293.3" cy="198.8" r="3.4" class="fill-soft-d"/>
  <circle cx="425.5" cy="126.8" r="3.4" class="fill-soft-d"/>
  <circle cx="287.3" cy="184.3" r="3.4" class="fill-soft-d"/>
  <circle cx="407.1" cy="188.2" r="3.4" class="fill-soft-d"/>
  <circle cx="468.0" cy="208.4" r="3.4" class="fill-soft-d"/>
  <circle cx="408.4" cy="203.4" r="3.4" class="fill-soft-d"/>
  <circle cx="459.1" cy="167.8" r="3.4" class="fill-soft-d"/>
  <circle cx="373.4" cy="226.7" r="3.4" class="fill-soft-d"/>
  <circle cx="472.4" cy="140.9" r="3.4" class="fill-soft-d"/>
  <circle cx="427.6" cy="127.5" r="3.4" class="fill-soft-d"/>
  <circle cx="408.1" cy="189.2" r="3.4" class="fill-soft-d"/>
  <circle cx="352.7" cy="194.8" r="3.4" class="fill-soft-d"/>
  <circle cx="457.3" cy="217.7" r="3.4" class="fill-soft-d"/>
  <circle cx="362.6" cy="178.2" r="3.4" class="fill-soft-d"/>
  <circle cx="402.1" cy="147.0" r="3.4" class="fill-soft-d"/>
  <circle cx="331.1" cy="227.1" r="3.4" class="fill-soft-d"/>
  <circle cx="391.1" cy="124.9" r="3.4" class="fill-soft-d"/>
  <circle cx="427.8" cy="159.5" r="3.4" class="fill-soft-d"/>
  <circle cx="376.0" cy="156.8" r="3.4" class="fill-soft-d"/>
  <circle cx="379.6" cy="138.6" r="3.4" class="fill-soft-d"/>
  <circle cx="353.0" cy="162.3" r="3.4" class="fill-soft-d"/>
  <circle cx="386.0" cy="148.1" r="3.4" class="fill-soft-d"/>
  <circle cx="402.1" cy="78.5" r="3.4" class="fill-soft-d"/>
  <circle cx="463.5" cy="115.0" r="3.4" class="fill-soft-d"/>
  <circle cx="293.0" cy="178.8" r="3.4" class="fill-soft-d"/>
  <circle cx="362.0" cy="148.5" r="3.4" class="fill-soft-d"/>
  <circle cx="281.5" cy="126.2" r="3.4" class="fill-soft-d"/>
  <circle cx="442.7" cy="212.2" r="3.4" class="fill-soft-d"/>
  <circle cx="221.9" cy="133.6" r="3.4" class="fill-b"/>
  <circle cx="336.1" cy="179.0" r="3.4" class="fill-b"/>
  <circle cx="361.0" cy="164.8" r="3.4" class="fill-b"/>
  <circle cx="387.9" cy="128.7" r="3.4" class="fill-b"/>
  <circle cx="438.0" cy="167.1" r="3.4" class="fill-b"/>
  <circle cx="305.1" cy="175.7" r="3.4" class="fill-b"/>
  <circle cx="173.2" cy="138.0" r="3.4" class="fill-b"/>
  <circle cx="276.3" cy="162.6" r="3.4" class="fill-b"/>
  <circle cx="196.2" cy="143.0" r="3.4" class="fill-b"/>
  <circle cx="395.9" cy="173.3" r="3.4" class="fill-b"/>
  <circle cx="281.3" cy="161.7" r="3.4" class="fill-b"/>
  <circle cx="244.6" cy="166.9" r="3.4" class="fill-b"/>
  <circle cx="355.0" cy="225.0" r="3.4" class="fill-b"/>
  <circle cx="293.1" cy="221.1" r="3.4" class="fill-b"/>
  <circle cx="260.8" cy="145.7" r="3.4" class="fill-b"/>
  <circle cx="499.3" cy="127.0" r="3.4" class="fill-b"/>
  <circle cx="372.3" cy="120.1" r="3.4" class="fill-b"/>
  <circle cx="217.7" cy="99.1" r="3.4" class="fill-b"/>
  <circle cx="306.8" cy="141.6" r="3.4" class="fill-b"/>
  <circle cx="245.5" cy="83.4" r="3.4" class="fill-b"/>
  <circle cx="457.6" cy="161.7" r="3.4" class="fill-b"/>
  <circle cx="318.9" cy="124.2" r="3.4" class="fill-b"/>
  <circle cx="220.4" cy="52.6" r="3.4" class="fill-b"/>
  <circle cx="223.8" cy="157.3" r="3.4" class="fill-b"/>
  <circle cx="248.9" cy="135.6" r="3.4" class="fill-b"/>
  <circle cx="336.2" cy="175.6" r="3.4" class="fill-b"/>
  <circle cx="410.4" cy="153.3" r="3.4" class="fill-b"/>
  <circle cx="243.5" cy="157.3" r="3.4" class="fill-b"/>
  <circle cx="357.2" cy="144.7" r="3.4" class="fill-b"/>
  <circle cx="218.0" cy="155.6" r="3.4" class="fill-b"/>
  <circle cx="419.1" cy="172.7" r="3.4" class="fill-b"/>
  <circle cx="326.7" cy="199.8" r="3.4" class="fill-b"/>
  <circle cx="475.6" cy="156.7" r="3.4" class="fill-b"/>
  <circle cx="247.4" cy="186.3" r="3.4" class="fill-b"/>
  <circle cx="418.8" cy="181.0" r="3.4" class="fill-b"/>
  <circle cx="328.6" cy="140.9" r="3.4" class="fill-b"/>
  <circle cx="400.8" cy="126.5" r="3.4" class="fill-b"/>
  <circle cx="368.0" cy="189.1" r="3.4" class="fill-b"/>
  <circle cx="261.8" cy="265.6" r="3.4" class="fill-b"/>
  <circle cx="326.4" cy="153.2" r="3.4" class="fill-b"/>
  <circle cx="392.8" cy="177.5" r="3.4" class="fill-b"/>
  <circle cx="287.4" cy="143.0" r="3.4" class="fill-b"/>
  <circle cx="374.6" cy="153.7" r="3.4" class="fill-b"/>
  <circle cx="200.2" cy="169.0" r="3.4" class="fill-b"/>
  <circle cx="490.1" cy="195.9" r="3.4" class="fill-b"/>
  <circle cx="429.4" cy="107.9" r="3.4" class="fill-b"/>
  <circle cx="264.8" cy="231.6" r="3.4" class="fill-b"/>
  <circle cx="466.8" cy="206.5" r="3.4" class="fill-b"/>
  <circle cx="210.0" cy="161.9" r="3.4" class="fill-b"/>
  <circle cx="312.3" cy="206.8" r="3.4" class="fill-b"/>
  <circle cx="259.4" cy="180.3" r="3.4" class="fill-b"/>
  <circle cx="408.5" cy="141.5" r="3.4" class="fill-b"/>
  <circle cx="392.7" cy="167.5" r="3.4" class="fill-b"/>
  <circle cx="259.0" cy="137.6" r="3.4" class="fill-b"/>
  <circle cx="342.5" cy="141.2" r="3.4" class="fill-b"/>
  <circle cx="296.3" cy="152.2" r="3.4" class="fill-b"/>
  <circle cx="438.9" cy="241.8" r="3.4" class="fill-b"/>
  <circle cx="196.1" cy="174.0" r="3.4" class="fill-b"/>
  <circle cx="357.8" cy="193.6" r="3.4" class="fill-b"/>
  <circle cx="291.2" cy="189.2" r="3.4" class="fill-b"/>
  <circle cx="439.2" cy="140.3" r="3.4" class="fill-b"/>
  <circle cx="432.2" cy="186.1" r="3.4" class="fill-b"/>
  <circle cx="453.6" cy="237.2" r="3.4" class="fill-b"/>
  <circle cx="231.2" cy="125.4" r="3.4" class="fill-b"/>
  <circle cx="182.7" cy="210.0" r="3.4" class="fill-b"/>
  <circle cx="330.2" cy="191.0" r="3.4" class="fill-b"/>
  <circle cx="365.8" cy="165.0" r="3.4" class="fill-b"/>
  <circle cx="446.4" cy="142.6" r="3.4" class="fill-b"/>
  <circle cx="264.9" cy="147.7" r="3.4" class="fill-b"/>
  <circle cx="368.8" cy="157.1" r="3.4" class="fill-b"/>
  <circle cx="316.8" cy="125.0" r="3.4" class="fill-b"/>
  <circle cx="210.4" cy="117.3" r="3.4" class="fill-b"/>
  <circle cx="426.3" cy="222.9" r="3.4" class="fill-b"/>
  <circle cx="286.2" cy="147.6" r="3.4" class="fill-b"/>
  <circle cx="409.0" cy="174.0" r="3.4" class="fill-b"/>
  <circle cx="247.0" cy="136.0" r="3.4" class="fill-b"/>
  <circle cx="268.7" cy="193.0" r="3.4" class="fill-b"/>
  <circle cx="183.6" cy="129.5" r="3.4" class="fill-b"/>
  <circle cx="294.2" cy="117.6" r="3.4" class="fill-b"/>
  <circle cx="434.9" cy="197.9" r="3.4" class="fill-b"/>
  <circle cx="246.7" cy="167.0" r="4.2" class="fill-ink"/>
  <circle cx="391.3" cy="167.0" r="4.2" class="fill-ink"/>
  <text x="78" y="24" class="fig-bold">exact denoiser, ancestral samples</text>
  <text x="78" y="322" class="fig-small">pale: the two blobs</text>
  <text x="280" y="322" class="fig-small">solid: samples</text>
  <text x="450" y="322" class="fig-small">black: centers</text>
</svg>
Figure 2. Pale points are draws from the two blobs. Solid points are 80 ancestral samples from the exact denoiser, seed 2. They sit on the blobs. The black dots mark the centers at ±1.5.
```

The graded sampler uses 80 betas spaced from 0.0001 to 0.1, so $\bar\alpha_{80} = 0.015835$, and draws 2500 dots. Seed 0 lands at mean $(0.018944, -0.002615)$ with covariance entries 3.107841, $-0.080120$, and 0.942913. The check accepts a mean inside $[-0.15, 0.15]$ on each coordinate, an $x$ variance inside $[2.80, 3.55]$, a $y$ variance inside $[0.80, 1.15]$, and an off-diagonal inside $[-0.20, 0.20]$. Across seeds 0 through 7 the $x$ variance stays inside $[3.019, 3.173]$ and the $y$ variance inside $[0.893, 0.953]$. Dropping the factor $1/\sqrt{\alpha_t}$ in the reverse mean shrinks a cloud of this size to a variance near $(1.30, 0.47)$, which misses the window.

## Push a sample toward one blob

You want a dot from the east blob, not from either blob. Train one denoiser to predict the noise with the blob's identity, and, on some steps, with that identity dropped. Write $\varepsilon_c$ for the conditional prediction and $\varepsilon_u$ for the unconditional one. Classifier-free guidance is the extrapolation

$$
\varepsilon_{\text{guided}} = \varepsilon_u + s\,(\varepsilon_c - \varepsilon_u).
$$

At $s = 0$ this is the unconditional prediction. At $s = 1$ it is the conditional prediction. At $s > 1$ it steps past the conditional prediction, away from the unconditional one. Ho and Salimans write the same line as $(1 + w)\varepsilon_c - w\,\varepsilon_u$, which is this formula with $s = 1 + w$.

On $\varepsilon_c = (1, -0.5)$, $\varepsilon_u = (0.2, 0)$, and $s = 3$, the difference is $(0.8, -0.5)$ and the guided prediction is $(0.2, 0) + 3 \times (0.8, -0.5) = (2.6, -1.5)$.

> [!WARNING]
> At scale 3, forgetting to subtract the unconditional prediction before scaling returns 3.2 in the first coordinate, where the extrapolation is 2.6.

## A straight path from noise to a dot

Flow matching replaces the bent schedule with a straight path. Pick a clean dot $x_0$ and a standard normal $\varepsilon$, and for $t$ between 0 and 1 set

$$
x_t = (1 - t)\, x_0 + t\, \varepsilon.
$$

The velocity is $\varepsilon - x_0$, and it does not change with $t$. One Euler step adds $\Delta t$ times that velocity.

On the east center and $\varepsilon = (0, 1)$, at $t = 0.25$ the point is $0.75 \times (1.5, 0) + 0.25 \times (0, 1) = (1.125, 0.25)$ and the velocity is $(-1.5, 1)$. A step of $\Delta t = -0.25$ is $(1.125, 0.25) + (-0.25) \times (-1.5, 1) = (1.5, 0)$.

This chapter keeps the variance-preserving schedule. The straight path is a bridge, not a second model you train here.

## Cost and scaling

The graded sampler calls the denoiser 80 times for one cloud. A deterministic step can skip times, because it reads $\bar\alpha$ only where it visits. On this toy the bill is the number of calls.

## Where this breaks

- **$\bar\alpha_T$ stays away from 0.** $x_T$ is not a standard normal, so lengthen the schedule or raise the last $\beta$.
- **The denoiser averages the two blobs.** Samples pile up in the middle. Smaller steps, or the exact responsibilities, split them.
- **The guidance scale is large.** Samples leave the blob. Lower the scale until they sit on the blob you asked for.
- **The last ancestral step still adds noise.** $\tilde\beta_1$ is 0, and extra noise there blurs the finished dot.
- **A straight-path velocity is paired with diffusion coefficients.** The schedules differ, and a mixed step misses $x_0$.

> [!KEY]
> You draw a dot the field does not contain by noising the flowers down to a standard bell and walking back one step at a time. The walk uses a noise prediction, and that prediction is a scaled score.

> [!RECAP]
> - On $\beta = (0.1, 0.2, 0.4)$ the cumulative product is $(0.9, 0.72, 0.432)$. At $t = 2$, $x_0 = (1.5, 0)$, and $\varepsilon = (1, -1)$, the noisy dot is $(1.801942, -0.529150)$.
> - For a standard normal cloud the minimum-MSE noise prediction is $\sqrt{1 - \bar\alpha_t}\, x_t$, and its expected squared error equals $\bar\alpha_t$. On the worked pair that prediction scores 0.260281, and the linear gradient at $w = (0.4, 0.1)$ is $(-0.503144, -0.501150)$.
> - The ancestral step at $t = 2$ with noise $(0.5, -0.25)$ lands at $(1.725686, -0.235846)$. The deterministic step with the true noise lands at $(1.739253, -0.316228)$, which is the closed form at $t = 1$.
> - Seed 0 of the exact two-blob sampler, 2500 dots, has mean $(0.018944, -0.002615)$ and covariance entries 3.107841, $-0.080120$, 0.942913.
> - Guidance at scale 3 sends $(1, -0.5)$ and $(0.2, 0)$ to $(2.6, -1.5)$.

## Build it step by step

$t$ counts from 1.

1. `alpha_bar(betas)` returns the cumulative product of $1 - \beta$, one entry per time. On $(0.1, 0.2, 0.4)$ that is $(0.9, 0.72, 0.432)$. Returning $1 - \beta$ matches the first entry and misses 0.72. On 1000 betas from 0.0001 to 0.02, the last entry is $4.036 \times 10^{-5}$.

2. `q_sample(x0, t, eps, betas)` returns $\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$. `x0` and `eps` share a shape. Dropping the factor on $x_0$ keeps the variance and moves the Monte Carlo mean to $x_0$.

3. `ddpm_loss(model, x0, t, eps, betas)` builds $x_t$, calls `model(x_t, t)`, and returns the mean squared error against `eps`. For a standard normal cloud the model is $\sqrt{1 - \bar\alpha_t}\, x_t$. Swapping the scales scores 0.291721 on the worked pair, against 0.260281. Seed 0 on 4000 standard-normal pairs averages to 0.718339, against 0.72.

4. `linear_eps_grad(weight, x0, t, eps, betas)` is the gradient of that loss for $\hat\varepsilon = w \odot x_t$. `weight` broadcasts over the rows. A central difference of `ddpm_loss` matches it, including the worked value $(-0.503144, -0.501150)$.

5. `ddpm_step(x, eps_hat, t, betas, noise)` takes one ancestral step. Divide by $\sqrt{\alpha_t}$ after subtracting $(\beta_t / \sqrt{1 - \bar\alpha_t})\, \varepsilon_\theta$. For $t > 1$, add noise of scale $\sqrt{\tilde\beta_t}$. At $t = 1$, ignore `noise`. Using $\beta_t$ in place of $\tilde\beta_t$ changes the scale from 0.267261 to 0.447214.

6. `ddpm_sample(denoiser, n, betas, rng)` draws `n` rows in 2 dimensions. Start from `rng.normal`, step from time $T$ down to time 1, and draw a fresh normal only when $t > 1$. `denoiser(x, t)` returns a noise prediction shaped like `x`. The exact two-blob denoiser, 2500 dots, and the 80-step schedule put seeds 0 through 7 inside the window stated with Figure 2. Omitting the division by $\sqrt{\alpha_t}$ misses it.

7. `ddim_step(x, eps_hat, t, betas)` takes the deterministic step. Use $\bar\alpha_0 = 1$ at $t = 1$. Returning only the predicted clean dot gives $(1.5, 0)$ and misses the recap's step.

8. `classifier_free_guidance(cond, uncond, scale)` returns `uncond + scale * (cond - uncond)`. Scale 0 returns `uncond` and scale 1 returns `cond`. Adding `scale * cond` without the subtraction returns $(3.2, -1.5)$ where the recap has $(2.6, -1.5)$.

## The exercise

You implement `alpha_bar`, `q_sample`, `ddpm_loss`, `linear_eps_grad`, `ddpm_step`, `ddpm_sample`, `ddim_step`, and `classifier_free_guidance`. A `q_sample` that omits $\sqrt{\bar\alpha_t}$ on the clean dot passes a variance check and fails the mean.

## Further reading

A lecture number appears only where the page or the slide prints it.

- The forward process, the noise-prediction loss, and the ancestral sampler are in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho, Jain, and Abbeel).
- The deterministic step is in [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song, Meng, and Ermon).
- The conditional and unconditional combination is in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho and Salimans). Their weight $w$ is this chapter's scale minus 1.
- Straight conditional paths are in [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman, Chen, Ben-Hamu, Nickel, and Le).
- [CS236 lecture 13, Score-Based Models](https://deepgenerativemodels.github.io/assets/slides/lecture%2013.pptx) (Stefano Ermon). The title slide prints "lecture 13". The [Fall 2023 syllabus](https://deepgenerativemodels.github.io/syllabus.html) lists Score Based Models on November 8.
- [CS236 lecture 16, Score-Based Diffusion Models](https://deepgenerativemodels.github.io/assets/slides/lecture16-2023-comp.pptx). The title slide prints "lecture 16". The syllabus lists Score Based Diffusion Models on November 27.
- [CS231n lecture 14, Generative Models 2](https://cs231n.stanford.edu/2026/schedule.html). The schedule prints "Lecture 14: Generative Models 2" and the topic "Diffusion models".
- [Lecture 4, Deep Generative Modeling](https://introtodeeplearning.com/2025/) (MIT 6.S191, 2025). The schedule prints that title on March 24. The slides are [6S191_MIT_DeepLearning_L4.pdf](https://introtodeeplearning.com/2025/slides/6S191_MIT_DeepLearning_L4.pdf).
- The [6.S898 Fall 2023 schedule](https://phillipi.github.io/6.s898/) lists "Generative models — basics" on October 31, and the details name diffusion models. It lists "Generative models — conditional models" on November 7. Neither line prints a lecture number.
- [Lecture 20, Generative Models](https://dlsyscourse.org/lectures/) (CMU 10-414/714). The lectures page prints "20 - Generative Models".
