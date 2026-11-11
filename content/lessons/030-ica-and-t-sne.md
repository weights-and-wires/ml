---
title: "ICA and t-SNE: unmixing signals and drawing neighbourhoods"
slug: 030-ica-and-t-sne
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Unmix independent sources that PCA leaves rotated, with FastICA, then squash many columns into a 2-D map of neighbourhoods with t-SNE.
minutes: 80
gpu: false
objectives:
  - Explain why decorrelating whitened Gaussian data leaves a rotation that no method can undo, and why non-Gaussian sources break the tie.
  - Run one FastICA update with a kurtosis or log cosh contrast, find several sources by deflation, and say what is recovered only up to sign, scale, and order.
  - Find each point's Gaussian bandwidth by binary search on perplexity, and symmetrise the neighbour probabilities.
  - Write the t-SNE objective and its gradient, and say why the map uses a Student-t kernel.
  - Say what a t-SNE plot does and does not tell you about cluster sizes and distances between clusters.
prereqs:
  - 028-principal-components
  - 029-power-iteration-and-probabilistic-pca
exercise: 030-ica-and-t-sne
---

# ICA and t-SNE: unmixing signals and drawing neighbourhoods

Two people talk at once in a room, and two microphones sit at different spots. Each microphone records a blend of both voices, with a different volume for each. You want each voice on its own. This is the **cocktail-party problem**. The recordings are columns of a table, and the voices are hidden columns you never see.

PCA seems like the right tool, because it turns correlated columns into uncorrelated ones. It does not finish the job: uncorrelated is weaker than separate. The first half of this chapter shows what PCA leaves undone and the extra clue that finishes it, which is that a voice is not a bell curve. **Independent component analysis**, or ICA, uses that clue.

The second half asks what a table with many columns looks like. Two columns throw away the rest, and PCA's top two directions can lay separate clumps on top of each other. **t-SNE** builds a 2-D map that keeps each point beside its neighbours. It makes clear pictures, and they are easy to misread.

## What you already know

- Whitening rotates the centred data onto its principal directions and divides each score by its standard deviation, so the covariance becomes the identity ([principal components](/lessons/028-principal-components)).
- A model with hidden coordinates pushed through a matrix, plus noise, is fitted by guessing the hidden part and refitting ([probabilistic PCA](/lessons/029-power-iteration-and-probabilistic-pca)).
- Two columns are **independent** when knowing one tells you nothing about the other. Independent columns are uncorrelated, and the reverse does not hold.
- Gradient descent steps against the gradient, and momentum adds a fraction of the last step ([gradient descent and momentum](/lessons/005-gradient-descent-and-momentum)).

## Two voices, two microphones

Write the hidden voices for one moment in time as a vector $s$, and the microphone readings as $x$. The model says each microphone hears a fixed blend:

$$
x = A s .
$$

Read it aloud as "the readings are the voices pushed through a mixing matrix $A$". You observe many moments, so $X$ has a row per moment. You know neither $A$ nor $s$, and you want both. Two assumptions make it possible: the voices are independent of each other, and at most one of them is Gaussian.

Start by whitening, which you already know how to do. Scale each voice to variance 1. The whitened data is then $z = R s$ for some matrix $R$ with $RR^{\top} = I$, because $z$ and $s$ both have identity covariance. Such an $R$ is a rotation, a turn with possibly a flip.

So the hard work of unmixing shrinks to one unknown: the angle. In two dimensions, that is one number. Figure 1 shows 150 whitened points made from two independent uniform voices. The cloud fills a tilted square, so a pair of axes along its sides recovers the voices. Turn the axes by 34 degrees and the covariance is still the identity, so PCA has no reason to prefer one pair.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Whitened points that fill a tilted square, with the two ICA axes along its sides and another pair of axes rotated away">
<circle cx="180.6" cy="47.3" r="2.6" class="fill-a"/>
<circle cx="229.7" cy="199.6" r="2.6" class="fill-a"/>
<circle cx="153.5" cy="79.0" r="2.6" class="fill-a"/>
<circle cx="246.0" cy="89.5" r="2.6" class="fill-a"/>
<circle cx="287.9" cy="117.6" r="2.6" class="fill-a"/>
<circle cx="182.5" cy="132.9" r="2.6" class="fill-a"/>
<circle cx="177.8" cy="149.3" r="2.6" class="fill-a"/>
<circle cx="167.0" cy="249.0" r="2.6" class="fill-a"/>
<circle cx="148.7" cy="138.7" r="2.6" class="fill-a"/>
<circle cx="256.9" cy="140.1" r="2.6" class="fill-a"/>
<circle cx="62.9" cy="150.2" r="2.6" class="fill-a"/>
<circle cx="198.9" cy="88.6" r="2.6" class="fill-a"/>
<circle cx="241.7" cy="212.5" r="2.6" class="fill-a"/>
<circle cx="156.6" cy="183.9" r="2.6" class="fill-a"/>
<circle cx="105.2" cy="112.8" r="2.6" class="fill-a"/>
<circle cx="241.5" cy="64.4" r="2.6" class="fill-a"/>
<circle cx="159.9" cy="234.5" r="2.6" class="fill-a"/>
<circle cx="140.7" cy="125.2" r="2.6" class="fill-a"/>
<circle cx="137.2" cy="155.1" r="2.6" class="fill-a"/>
<circle cx="271.1" cy="132.2" r="2.6" class="fill-a"/>
<circle cx="222.7" cy="215.3" r="2.6" class="fill-a"/>
<circle cx="179.0" cy="220.3" r="2.6" class="fill-a"/>
<circle cx="152.8" cy="175.7" r="2.6" class="fill-a"/>
<circle cx="309.5" cy="135.6" r="2.6" class="fill-a"/>
<circle cx="263.5" cy="177.0" r="2.6" class="fill-a"/>
<circle cx="248.0" cy="87.5" r="2.6" class="fill-a"/>
<circle cx="257.3" cy="140.3" r="2.6" class="fill-a"/>
<circle cx="283.5" cy="161.5" r="2.6" class="fill-a"/>
<circle cx="222.9" cy="138.2" r="2.6" class="fill-a"/>
<circle cx="259.2" cy="113.7" r="2.6" class="fill-a"/>
<circle cx="120.2" cy="139.6" r="2.6" class="fill-a"/>
<circle cx="225.1" cy="190.5" r="2.6" class="fill-a"/>
<circle cx="300.4" cy="157.6" r="2.6" class="fill-a"/>
<circle cx="296.0" cy="142.2" r="2.6" class="fill-a"/>
<circle cx="246.2" cy="228.5" r="2.6" class="fill-a"/>
<circle cx="146.9" cy="233.2" r="2.6" class="fill-a"/>
<circle cx="197.8" cy="229.9" r="2.6" class="fill-a"/>
<circle cx="157.2" cy="63.5" r="2.6" class="fill-a"/>
<circle cx="186.7" cy="40.5" r="2.6" class="fill-a"/>
<circle cx="108.7" cy="161.7" r="2.6" class="fill-a"/>
<circle cx="177.6" cy="62.3" r="2.6" class="fill-a"/>
<circle cx="148.6" cy="198.7" r="2.6" class="fill-a"/>
<circle cx="234.2" cy="241.8" r="2.6" class="fill-a"/>
<circle cx="153.9" cy="112.9" r="2.6" class="fill-a"/>
<circle cx="292.9" cy="104.0" r="2.6" class="fill-a"/>
<circle cx="243.7" cy="149.6" r="2.6" class="fill-a"/>
<circle cx="187.9" cy="184.0" r="2.6" class="fill-a"/>
<circle cx="219.9" cy="172.2" r="2.6" class="fill-a"/>
<circle cx="250.9" cy="105.9" r="2.6" class="fill-a"/>
<circle cx="253.2" cy="86.4" r="2.6" class="fill-a"/>
<circle cx="192.0" cy="148.3" r="2.6" class="fill-a"/>
<circle cx="253.1" cy="166.5" r="2.6" class="fill-a"/>
<circle cx="129.6" cy="83.4" r="2.6" class="fill-a"/>
<circle cx="124.4" cy="142.2" r="2.6" class="fill-a"/>
<circle cx="201.7" cy="111.4" r="2.6" class="fill-a"/>
<circle cx="271.9" cy="185.1" r="2.6" class="fill-a"/>
<circle cx="200.9" cy="100.9" r="2.6" class="fill-a"/>
<circle cx="255.8" cy="79.4" r="2.6" class="fill-a"/>
<circle cx="145.7" cy="217.3" r="2.6" class="fill-a"/>
<circle cx="173.5" cy="191.9" r="2.6" class="fill-a"/>
<circle cx="212.4" cy="260.1" r="2.6" class="fill-a"/>
<circle cx="188.5" cy="227.6" r="2.6" class="fill-a"/>
<circle cx="279.6" cy="151.1" r="2.6" class="fill-a"/>
<circle cx="220.1" cy="118.4" r="2.6" class="fill-a"/>
<circle cx="156.7" cy="123.6" r="2.6" class="fill-a"/>
<circle cx="157.7" cy="138.7" r="2.6" class="fill-a"/>
<circle cx="268.3" cy="95.2" r="2.6" class="fill-a"/>
<circle cx="207.8" cy="120.3" r="2.6" class="fill-a"/>
<circle cx="287.8" cy="109.7" r="2.6" class="fill-a"/>
<circle cx="152.4" cy="101.7" r="2.6" class="fill-a"/>
<circle cx="162.9" cy="157.9" r="2.6" class="fill-a"/>
<circle cx="220.4" cy="75.2" r="2.6" class="fill-a"/>
<circle cx="296.4" cy="165.0" r="2.6" class="fill-a"/>
<circle cx="182.9" cy="62.5" r="2.6" class="fill-a"/>
<circle cx="217.4" cy="80.5" r="2.6" class="fill-a"/>
<circle cx="142.0" cy="218.3" r="2.6" class="fill-a"/>
<circle cx="160.5" cy="80.6" r="2.6" class="fill-a"/>
<circle cx="167.7" cy="228.3" r="2.6" class="fill-a"/>
<circle cx="320.3" cy="158.8" r="2.6" class="fill-a"/>
<circle cx="150.1" cy="209.7" r="2.6" class="fill-a"/>
<circle cx="125.5" cy="97.5" r="2.6" class="fill-a"/>
<circle cx="116.3" cy="142.0" r="2.6" class="fill-a"/>
<circle cx="254.8" cy="211.6" r="2.6" class="fill-a"/>
<circle cx="194.9" cy="9.9" r="2.6" class="fill-a"/>
<circle cx="138.2" cy="187.7" r="2.6" class="fill-a"/>
<circle cx="139.2" cy="135.3" r="2.6" class="fill-a"/>
<circle cx="220.7" cy="95.8" r="2.6" class="fill-a"/>
<circle cx="255.7" cy="143.6" r="2.6" class="fill-a"/>
<circle cx="144.0" cy="168.9" r="2.6" class="fill-a"/>
<circle cx="158.3" cy="188.8" r="2.6" class="fill-a"/>
<circle cx="251.1" cy="94.7" r="2.6" class="fill-a"/>
<circle cx="138.4" cy="195.5" r="2.6" class="fill-a"/>
<circle cx="141.9" cy="226.5" r="2.6" class="fill-a"/>
<circle cx="154.0" cy="51.8" r="2.6" class="fill-a"/>
<circle cx="210.2" cy="211.3" r="2.6" class="fill-a"/>
<circle cx="184.7" cy="179.2" r="2.6" class="fill-a"/>
<circle cx="225.4" cy="225.7" r="2.6" class="fill-a"/>
<circle cx="197.4" cy="161.5" r="2.6" class="fill-a"/>
<circle cx="248.2" cy="140.9" r="2.6" class="fill-a"/>
<circle cx="206.7" cy="186.5" r="2.6" class="fill-a"/>
<circle cx="242.2" cy="181.6" r="2.6" class="fill-a"/>
<circle cx="129.3" cy="85.9" r="2.6" class="fill-a"/>
<circle cx="229.1" cy="216.4" r="2.6" class="fill-a"/>
<circle cx="235.4" cy="148.1" r="2.6" class="fill-a"/>
<circle cx="148.6" cy="159.6" r="2.6" class="fill-a"/>
<circle cx="88.8" cy="144.1" r="2.6" class="fill-a"/>
<circle cx="185.3" cy="269.1" r="2.6" class="fill-a"/>
<circle cx="205.0" cy="35.3" r="2.6" class="fill-a"/>
<circle cx="268.6" cy="184.2" r="2.6" class="fill-a"/>
<circle cx="185.2" cy="95.4" r="2.6" class="fill-a"/>
<circle cx="270.7" cy="114.7" r="2.6" class="fill-a"/>
<circle cx="175.1" cy="168.4" r="2.6" class="fill-a"/>
<circle cx="300.6" cy="99.9" r="2.6" class="fill-a"/>
<circle cx="252.3" cy="177.9" r="2.6" class="fill-a"/>
<circle cx="182.1" cy="146.2" r="2.6" class="fill-a"/>
<circle cx="113.8" cy="175.2" r="2.6" class="fill-a"/>
<circle cx="154.0" cy="112.3" r="2.6" class="fill-a"/>
<circle cx="137.3" cy="117.6" r="2.6" class="fill-a"/>
<circle cx="200.3" cy="260.2" r="2.6" class="fill-a"/>
<circle cx="131.5" cy="114.4" r="2.6" class="fill-a"/>
<circle cx="297.3" cy="152.8" r="2.6" class="fill-a"/>
<circle cx="157.4" cy="166.7" r="2.6" class="fill-a"/>
<circle cx="239.8" cy="170.5" r="2.6" class="fill-a"/>
<circle cx="194.7" cy="190.7" r="2.6" class="fill-a"/>
<circle cx="157.7" cy="148.7" r="2.6" class="fill-a"/>
<circle cx="182.2" cy="186.7" r="2.6" class="fill-a"/>
<circle cx="225.1" cy="121.5" r="2.6" class="fill-a"/>
<circle cx="130.0" cy="202.0" r="2.6" class="fill-a"/>
<circle cx="207.4" cy="47.6" r="2.6" class="fill-a"/>
<circle cx="81.8" cy="132.9" r="2.6" class="fill-a"/>
<circle cx="305.3" cy="171.7" r="2.6" class="fill-a"/>
<circle cx="211.0" cy="66.2" r="2.6" class="fill-a"/>
<circle cx="215.9" cy="156.8" r="2.6" class="fill-a"/>
<circle cx="286.0" cy="144.3" r="2.6" class="fill-a"/>
<circle cx="257.3" cy="191.8" r="2.6" class="fill-a"/>
<circle cx="198.9" cy="234.7" r="2.6" class="fill-a"/>
<circle cx="256.8" cy="201.7" r="2.6" class="fill-a"/>
<circle cx="206.2" cy="169.9" r="2.6" class="fill-a"/>
<circle cx="156.9" cy="102.9" r="2.6" class="fill-a"/>
<circle cx="199.9" cy="267.7" r="2.6" class="fill-a"/>
<circle cx="110.3" cy="104.8" r="2.6" class="fill-a"/>
<circle cx="228.4" cy="64.8" r="2.6" class="fill-a"/>
<circle cx="107.6" cy="141.5" r="2.6" class="fill-a"/>
<circle cx="260.4" cy="211.7" r="2.6" class="fill-a"/>
<circle cx="182.4" cy="130.8" r="2.6" class="fill-a"/>
<circle cx="209.8" cy="229.1" r="2.6" class="fill-a"/>
<circle cx="181.1" cy="40.6" r="2.6" class="fill-a"/>
<circle cx="251.7" cy="194.9" r="2.6" class="fill-a"/>
<circle cx="181.7" cy="210.2" r="2.6" class="fill-a"/>
<circle cx="185.1" cy="200.5" r="2.6" class="fill-a"/>
<line x1="79.7" y1="254.8" x2="320.3" y2="45.2" class="stroke-b" stroke-width="2.5"/><text x="324.3" y="45.2" class="fig-small">ICA 1</text>
<line x1="304.8" y1="270.3" x2="95.2" y2="29.7" class="stroke-b" stroke-width="2.5"/><text x="99.2" y="29.7" class="fig-small">ICA 2</text>
<line x1="159.9" y1="304.4" x2="240.1" y2="-4.4" class="stroke-muted dashed" stroke-width="2.5"/><text x="244.1" y="-4.4" class="fig-small">other axis</text>
<line x1="354.4" y1="190.1" x2="45.6" y2="109.9" class="stroke-muted dashed" stroke-width="2.5"/><text x="49.6" y="109.9" class="fig-small"></text>
<text x="400" y="110" class="fig-small">covariance is the identity,</text>
<text x="400" y="128" class="fig-small">so every rotation is a valid whitening.</text>
<text x="400" y="160" class="fig-small">only the ICA axes run along</text>
<text x="400" y="178" class="fig-small">the sides of the square.</text></svg>
Figure 1. Two independent uniform voices, mixed and whitened, fill a tilted square. Whitening fixes the covariance but not the tilt. FastICA's axes (solid) lie along the sides of the square, 41.1 degrees from the horizontal, and the output along the second one matches a true voice with correlation 0.998. A rotated pair (dashed) is equally white and mixes the voices.
```

## Why PCA cannot finish, and what breaks the tie

Suppose the voices were Gaussian instead. A whitened Gaussian cloud is a round ball, and a ball looks the same after any rotation. So $z$ and $Rz$ have the same distribution, and the data carry no trace of which $R$ was used. No method could find the voices.

A uniform voice is different. Its cloud has corners, and corners show which way it is turned. That is the clue: **the voices are not Gaussian**. A mixture of independent voices is closer to Gaussian than the voices themselves, because a sum of independent pieces looks more bell-shaped than each piece. So search for the direction in which the data is *least* Gaussian, and expect to land on one voice.

You need a number that scores non-Gaussianity. The simplest is the **excess kurtosis** of a unit-variance column $u$, which is $E[u^4] - 3$. The 3 is $E[u^4]$ for a Gaussian, so a Gaussian scores 0. A uniform column on $[-\sqrt{3}, \sqrt{3}]$ has variance 1 and $E[u^4] = 9/5$, so it scores $-1.2$. Project the whitened mixture onto the direction at angle $\theta$. At the angle of a voice you see that voice alone, with score $-1.2$. At 45 degrees you see $(s_1 + s_2)/\sqrt{2}$. The score of a sum of independent pieces adds up with each weight raised to the fourth power, so it is $2 \cdot \tfrac14 \cdot (-1.2) = -0.6$, halfway to Gaussian. Figure 2 plots the score for every angle, computed from 200,000 samples.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Excess kurtosis of a whitened projection against its angle: the uniform pair swings between -1.2 and -0.6, the Gaussian pair stays at 0">
<line x1="60.0" y1="53.6" x2="440.0" y2="53.6" class="stroke-grid"/><text x="52.0" y="57.6" class="fig-small" text-anchor="end">0</text>
<line x1="60.0" y1="135.0" x2="440.0" y2="135.0" class="stroke-grid"/><text x="52.0" y="139.0" class="fig-small" text-anchor="end">-0.6</text>
<line x1="60.0" y1="216.4" x2="440.0" y2="216.4" class="stroke-grid"/><text x="52.0" y="220.4" class="fig-small" text-anchor="end">-1.2</text>
<text x="60.0" y="256" class="fig-small" text-anchor="middle">0°</text>
<text x="250.0" y="256" class="fig-small" text-anchor="middle">45°</text>
<text x="440.0" y="256" class="fig-small" text-anchor="middle">90°</text>
<polyline points="60.0,216.4 72.7,215.4 85.3,212.8 98.0,208.5 110.7,202.8 123.3,195.9 136.0,188.2 148.7,179.9 161.3,171.4 174.0,163.1 186.7,155.4 199.3,148.5 212.0,142.9 224.7,138.7 237.3,136.1 250.0,135.3 262.7,136.2 275.3,138.9 288.0,143.1 300.7,148.9 313.3,155.7 326.0,163.5 338.7,171.8 351.3,180.3 364.0,188.6 376.7,196.4 389.3,203.2 402.0,208.9 414.7,213.1 427.3,215.7 440.0,216.6" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="60.0,52.7 72.7,52.8 85.3,52.8 98.0,52.9 110.7,53.0 123.3,53.0 136.0,53.1 148.7,53.1 161.3,53.2 174.0,53.2 186.7,53.2 199.3,53.1 212.0,53.1 224.7,53.1 237.3,53.0 250.0,53.0 262.7,52.9 275.3,52.9 288.0,52.9 300.7,52.8 313.3,52.8 326.0,52.8 338.7,52.8 351.3,52.9 364.0,52.9 376.7,53.0 389.3,53.0 402.0,53.1 414.7,53.2 427.3,53.3 440.0,53.3" fill="none" class="stroke-a" stroke-width="2.5"/>
<text x="448" y="133.3" class="fig-small">uniform pair</text>
<text x="448" y="57.6" class="fig-small">Gaussian pair</text>
<text x="250" y="278" text-anchor="middle">angle of the projection direction</text>
<text transform="rotate(-90 14 135)" x="14" y="135" text-anchor="middle">excess kurtosis</text></svg>
Figure 2. Excess kurtosis of a whitened projection against the angle of the projection. For two uniform voices the score swings between -1.2 at the voice directions (0 and 90 degrees) and -0.6 at 45 degrees, so the extremes mark the voices. For a Gaussian pair the line is flat at 0 (within 0.007 of it), so no angle stands out.
```

> [!KEY]
> Whitening leaves an unknown rotation. For Gaussian voices every rotation is equally good, so the problem has no answer. For non-Gaussian voices, the directions where the projection is least Gaussian are the voices.

## The FastICA update

You need to climb the non-Gaussianity score, not scan every angle. Pick a **contrast** $G$, a function whose average $E[G(w^{\top} z)]$ measures non-Gaussianity of the projection onto the unit vector $w$. Two common choices, with $g = G'$ and $g'$ their slopes:

| Contrast | $G(u)$ | $g(u)$ | $g'(u)$ |
|---|---|---|---|
| kurtosis | $u^4/4$ | $u^3$ | $3u^2$ |
| log cosh | $\log\cosh u$ | $\tanh u$ | $1 - \tanh^2 u$ |

Kurtosis is fragile, because the fourth power lets one outlier dominate. Log cosh grows like $|u|$, so outliers count for less.

At a maximum of $E[G(w^{\top} z)]$ over unit vectors, the gradient must point along $w$ itself: $E[z\, g(w^{\top} z)] = \beta w$ for some number $\beta$. FastICA solves this with a Newton step. Near a solution, the Jacobian of the left side is close to $E[g'(w^{\top}z)]\, I$, because whitening makes $E[zz^{\top}] = I$. The step then collapses, after rescaling, into the **fixed-point update**

$$
w \leftarrow E\big[z\, g(w^{\top} z)\big] - E\big[g'(w^{\top} z)\big]\, w, \qquad w \leftarrow \frac{w}{\lVert w \rVert} .
$$

Read it aloud as "average the data weighted by the contrast's response to the current projection, subtract the average slope times the current direction, then rescale to length 1". In numpy, with `Xw` holding whitened rows and a function that returns $g$ and $g'$:

```python
u = Xw @ w                              # projections, shape (n,)
gu, gpu = contrast_terms(u, "logcosh")
w = Xw.T @ gu / len(Xw) - gpu.mean() * w
w = w / np.linalg.norm(w)
```

Try the kurtosis contrast on four whitened points $(\pm\sqrt{2}, 0)$ and $(0, \pm\sqrt{2})$, which have mean 0 and covariance $I$. Start from $w = (0.6, 0.8)$. The projections are $u = (0.849, -0.849, 1.131, -1.131)$, and $g(u) = u^3 = (0.611, -0.611, 1.448, -1.448)$. The first term is $E[z\,g] = (0.432, 1.024)$. The second is $E[g'] = 3\,E[u^2] = 3$, so subtract $3w = (1.8, 2.4)$. That leaves $(-1.368, -1.376)$, which has length 1.940, so the updated $w$ is $(-0.705, -0.709)$. Notice the sign: the update can flip $w$ to the other end of the same line, which is harmless, because $w$ and $-w$ give the same voice with its sign reversed.

Repeat until $w$ stops changing, up to sign.

### More than one voice, and what you do not recover

With $k$ voices you need $k$ directions at right angles, because the rows of a rotation are orthonormal. **Deflation** finds them one at a time. After every update, subtract from $w$ its parts along the directions already found, then rescale:

```python
w = w - W_found.T @ (W_found @ w)
w = w / np.linalg.norm(w)
```

Without this step, every start climbs to the same most non-Gaussian voice, and you get that voice $k$ times. Once all rows are found, the voices are `Xw @ W.T`.

What do you get back? Three things are lost, and no method can recover them:

- **Scale.** Doubling a voice and halving its column of $A$ gives the same recordings. By convention each recovered voice has variance 1.
- **Sign.** Flipping a voice and its column of $A$ changes nothing either.
- **Order.** Which voice comes first depends on the random start.

To check a result, match each recovered column to its best true voice by absolute correlation. A value near 1 means the voice came back.

> [!WARNING]
> Do not read meaning into the order or sign of ICA outputs. Two seeds return the same voices in another order, some negated.

> [!KEY]
> FastICA maximises non-Gaussianity of a projection with the fixed-point update $w \leftarrow E[z\,g(w^{\top}z)] - E[g'(w^{\top}z)]\,w$. Deflation makes the directions orthogonal. The voices come back up to sign, scale, and order.

## A map that keeps neighbours

The second problem is a picture. You have 90 points in 10 columns, in three clumps, and you must choose 2 numbers per point. t-SNE aims to keep each point near the points that were its neighbours. It turns distances into **neighbour probabilities** in the original space, defines the same kind of probabilities for points on a 2-D map, and moves the map points until the two agree.

### Neighbour probabilities and perplexity

For point $i$, give every other point $j$ a weight that falls with squared distance, and rescale the weights to sum to 1:

$$
p_{j \mid i} = \frac{\exp(-\lVert x_i - x_j \rVert^2 \,\beta_i)}{\sum_{k \ne i} \exp(-\lVert x_i - x_k \rVert^2 \,\beta_i)} .
$$

Read it aloud as "the chance that $i$ picks $j$ as a neighbour, which is a Gaussian bump around $x_i$ evaluated at $x_j$, divided by the total over all other points". The number $\beta_i = 1/(2\sigma_i^2)$ sets the bump's width, and every point gets its own. A dense clump needs a narrow bump, or the point sees hundreds of neighbours. A sparse region needs a wide one, or the point sees none.

How should you pick $\beta_i$? Fix the effective number of neighbours. A row's **entropy** is $H_i = -\sum_j p_{j\mid i} \log_2 p_{j \mid i}$, measured in bits, and the **perplexity** is $2^{H_i}$. A row that is uniform over $m$ points has entropy $\log_2 m$, so its perplexity is exactly $m$. Perplexity therefore reads as "the number of neighbours this point effectively has". You choose one perplexity, usually between 5 and 50, and every row must match it.

Take a point with two others at squared distances 1 and 4, and set $\beta = 1$. The weights are $e^{-1} = 0.368$ and $e^{-4} = 0.018$, so $p = (0.9526, 0.0474)$. The entropy is $0.2754$ bits and the perplexity $2^{0.2754} = 1.21$, about one effective neighbour. Shrinking $\beta$ flattens the row and raises the perplexity, up to 2 at $\beta = 0$. Perplexity falls steadily as $\beta$ grows, so you find $\beta_i$ by **binary search**. If the entropy is above $\log_2$(target), the bump is too wide, so raise $\beta$; if below, lower it. While no upper bound is known, double $\beta$.

> [!WARNING]
> Entropy and perplexity must use the same base. Computing the entropy with `np.log` (nats) and comparing it with `np.log2(perplexity)` puts every row on the wrong perplexity, and nothing crashes. This chapter uses bits throughout.

The conditionals are not symmetric, since $p_{j\mid i} \neq p_{i \mid j}$ when the bandwidths differ. Average them into one table over all pairs:

$$
p_{ij} = \frac{p_{j \mid i} + p_{i \mid j}}{2n} .
$$

Each conditional row sums to 1, so the sum over every ordered pair is $n$ before dividing, and $p_{ij}$ sums to 1. Dividing by $2n$ also gives every point total weight at least $1/(2n)$, so outliers still pull on the map.

### The 2-D side: a heavier tail

On the map, point $i$ sits at $y_i$ in 2-D. Use the same idea, with a different bump:

$$
q_{ij} = \frac{(1 + \lVert y_i - y_j \rVert^2)^{-1}}{\sum_{k \ne l} (1 + \lVert y_k - y_l \rVert^2)^{-1}} .
$$

Read it aloud as "the similarity of two map points is one over one plus their squared distance, rescaled so that all pairs sum to 1". This is a **Student-t** kernel with one degree of freedom. Figure 3 compares it with a Gaussian. At distance 3 the Gaussian has fallen to 0.011 and the Student-t kernel is still 0.100. At 4 they are 0.0003 and 0.0588.

Why a heavy tail? A point in 10 dimensions can have many neighbours at moderate distance, but a plane has little room for them, so a Gaussian map crowds them together. The heavy tail lets map points sit far apart at little cost, so clumps separate.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Left: the Gaussian kernel falls faster than the Student-t kernel with one degree of freedom. Right: a t-SNE map of three planted clusters">
<line x1="50.0" y1="220.0" x2="270.0" y2="220.0" class="stroke-grid"/><line x1="50.0" y1="30.0" x2="50.0" y2="220.0" class="stroke-grid"/>
<polyline points="50.0,30.0 55.5,30.9 61.0,33.8 66.5,38.4 72.0,44.6 77.5,52.3 83.0,61.3 88.5,71.3 94.0,82.0 99.5,93.3 105.0,104.8 110.5,116.2 116.0,127.5 121.5,138.4 127.0,148.7 132.5,158.3 138.0,167.2 143.5,175.2 149.0,182.4 154.5,188.7 160.0,194.3 165.5,199.1 171.0,203.1 176.5,206.5 182.0,209.3 187.5,211.7 193.0,213.5 198.5,215.0 204.0,216.2 209.5,217.2 215.0,217.9 220.5,218.4 226.0,218.9 231.5,219.2 237.0,219.4 242.5,219.6 248.0,219.7 253.5,219.8 259.0,219.9 264.5,219.9 270.0,219.9" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="50.0,30.0 55.5,31.9 61.0,37.3 66.5,45.7 72.0,56.2 77.5,68.0 83.0,80.3 88.5,92.5 94.0,104.1 99.5,115.0 105.0,125.0 110.5,134.0 116.0,142.1 121.5,149.4 127.0,155.8 132.5,161.5 138.0,166.6 143.5,171.2 149.0,175.2 154.5,178.8 160.0,182.0 165.5,184.9 171.0,187.5 176.5,189.8 182.0,191.9 187.5,193.8 193.0,195.5 198.5,197.1 204.0,198.5 209.5,199.8 215.0,201.0 220.5,202.1 226.0,203.1 231.5,204.0 237.0,204.9 242.5,205.7 248.0,206.4 253.5,207.1 259.0,207.7 264.5,208.3 270.0,208.8" fill="none" class="stroke-b" stroke-width="2.5"/>
<line x1="130" y1="50" x2="160" y2="50" class="stroke-a" stroke-width="2.5"/><text x="166" y="54" class="fig-small">Gaussian: exp(-d²/2)</text>
<line x1="130" y1="72" x2="160" y2="72" class="stroke-b" stroke-width="2.5"/><text x="166" y="76" class="fig-small">Student-t: 1/(1+d²)</text>
<text x="160.0" y="256" text-anchor="middle">distance d</text>
<circle cx="392.8" cy="71.4" r="3.4" class="fill-a"/>
<circle cx="382.5" cy="82.6" r="3.4" class="fill-a"/>
<circle cx="399.4" cy="58.5" r="3.4" class="fill-a"/>
<circle cx="423.1" cy="65.9" r="3.4" class="fill-a"/>
<circle cx="381.9" cy="42.2" r="3.4" class="fill-a"/>
<circle cx="381.4" cy="58.9" r="3.4" class="fill-a"/>
<circle cx="409.6" cy="49.1" r="3.4" class="fill-a"/>
<circle cx="391.9" cy="35.3" r="3.4" class="fill-a"/>
<circle cx="389.6" cy="78.9" r="3.4" class="fill-a"/>
<circle cx="401.9" cy="44.0" r="3.4" class="fill-a"/>
<circle cx="380.9" cy="67.8" r="3.4" class="fill-a"/>
<circle cx="391.9" cy="42.9" r="3.4" class="fill-a"/>
<circle cx="417.4" cy="53.4" r="3.4" class="fill-a"/>
<circle cx="406.4" cy="81.4" r="3.4" class="fill-a"/>
<circle cx="399.8" cy="37.3" r="3.4" class="fill-a"/>
<circle cx="400.8" cy="64.0" r="3.4" class="fill-a"/>
<circle cx="418.2" cy="41.8" r="3.4" class="fill-a"/>
<circle cx="400.2" cy="76.0" r="3.4" class="fill-a"/>
<circle cx="366.2" cy="48.9" r="3.4" class="fill-a"/>
<circle cx="410.0" cy="63.4" r="3.4" class="fill-a"/>
<circle cx="411.6" cy="30.0" r="3.4" class="fill-a"/>
<circle cx="429.7" cy="57.4" r="3.4" class="fill-a"/>
<circle cx="382.5" cy="49.5" r="3.4" class="fill-a"/>
<circle cx="407.6" cy="55.0" r="3.4" class="fill-a"/>
<circle cx="368.1" cy="39.3" r="3.4" class="fill-a"/>
<circle cx="370.5" cy="59.1" r="3.4" class="fill-a"/>
<circle cx="390.8" cy="61.5" r="3.4" class="fill-a"/>
<circle cx="376.5" cy="31.8" r="3.4" class="fill-a"/>
<circle cx="390.1" cy="49.3" r="3.4" class="fill-a"/>
<circle cx="407.0" cy="72.3" r="3.4" class="fill-a"/>
<circle cx="554.0" cy="192.4" r="3.4" class="fill-b"/>
<circle cx="537.8" cy="182.0" r="3.4" class="fill-b"/>
<circle cx="539.6" cy="170.0" r="3.4" class="fill-b"/>
<circle cx="529.1" cy="174.7" r="3.4" class="fill-b"/>
<circle cx="538.3" cy="196.2" r="3.4" class="fill-b"/>
<circle cx="546.6" cy="215.5" r="3.4" class="fill-b"/>
<circle cx="563.6" cy="192.7" r="3.4" class="fill-b"/>
<circle cx="513.8" cy="201.7" r="3.4" class="fill-b"/>
<circle cx="557.6" cy="175.9" r="3.4" class="fill-b"/>
<circle cx="544.3" cy="174.8" r="3.4" class="fill-b"/>
<circle cx="525.9" cy="216.0" r="3.4" class="fill-b"/>
<circle cx="521.4" cy="209.0" r="3.4" class="fill-b"/>
<circle cx="533.1" cy="190.3" r="3.4" class="fill-b"/>
<circle cx="543.8" cy="209.4" r="3.4" class="fill-b"/>
<circle cx="551.6" cy="230.0" r="3.4" class="fill-b"/>
<circle cx="529.2" cy="184.5" r="3.4" class="fill-b"/>
<circle cx="523.7" cy="180.4" r="3.4" class="fill-b"/>
<circle cx="563.8" cy="184.0" r="3.4" class="fill-b"/>
<circle cx="580.0" cy="195.5" r="3.4" class="fill-b"/>
<circle cx="570.5" cy="193.6" r="3.4" class="fill-b"/>
<circle cx="514.3" cy="211.6" r="3.4" class="fill-b"/>
<circle cx="541.1" cy="186.2" r="3.4" class="fill-b"/>
<circle cx="556.0" cy="200.7" r="3.4" class="fill-b"/>
<circle cx="533.4" cy="200.8" r="3.4" class="fill-b"/>
<circle cx="549.3" cy="198.1" r="3.4" class="fill-b"/>
<circle cx="534.9" cy="211.7" r="3.4" class="fill-b"/>
<circle cx="553.3" cy="219.0" r="3.4" class="fill-b"/>
<circle cx="524.7" cy="201.7" r="3.4" class="fill-b"/>
<circle cx="539.3" cy="221.9" r="3.4" class="fill-b"/>
<circle cx="561.4" cy="206.7" r="3.4" class="fill-b"/>
<circle cx="373.6" cy="181.5" r="3.4" class="fill-c"/>
<circle cx="378.2" cy="202.9" r="3.4" class="fill-c"/>
<circle cx="354.3" cy="185.0" r="3.4" class="fill-c"/>
<circle cx="349.5" cy="208.0" r="3.4" class="fill-c"/>
<circle cx="358.3" cy="201.0" r="3.4" class="fill-c"/>
<circle cx="361.4" cy="187.2" r="3.4" class="fill-c"/>
<circle cx="404.9" cy="217.5" r="3.4" class="fill-c"/>
<circle cx="341.1" cy="199.1" r="3.4" class="fill-c"/>
<circle cx="398.4" cy="187.8" r="3.4" class="fill-c"/>
<circle cx="360.9" cy="168.4" r="3.4" class="fill-c"/>
<circle cx="384.3" cy="217.4" r="3.4" class="fill-c"/>
<circle cx="389.1" cy="181.6" r="3.4" class="fill-c"/>
<circle cx="347.5" cy="216.5" r="3.4" class="fill-c"/>
<circle cx="352.8" cy="194.5" r="3.4" class="fill-c"/>
<circle cx="330.0" cy="196.7" r="3.4" class="fill-c"/>
<circle cx="398.7" cy="213.2" r="3.4" class="fill-c"/>
<circle cx="370.0" cy="208.8" r="3.4" class="fill-c"/>
<circle cx="369.1" cy="201.2" r="3.4" class="fill-c"/>
<circle cx="355.6" cy="176.2" r="3.4" class="fill-c"/>
<circle cx="391.7" cy="197.2" r="3.4" class="fill-c"/>
<circle cx="366.9" cy="196.4" r="3.4" class="fill-c"/>
<circle cx="377.0" cy="175.4" r="3.4" class="fill-c"/>
<circle cx="363.6" cy="179.4" r="3.4" class="fill-c"/>
<circle cx="391.7" cy="207.7" r="3.4" class="fill-c"/>
<circle cx="386.0" cy="189.8" r="3.4" class="fill-c"/>
<circle cx="369.9" cy="188.8" r="3.4" class="fill-c"/>
<circle cx="381.8" cy="195.0" r="3.4" class="fill-c"/>
<circle cx="358.4" cy="211.5" r="3.4" class="fill-c"/>
<circle cx="396.2" cy="202.6" r="3.4" class="fill-c"/>
<circle cx="392.3" cy="217.8" r="3.4" class="fill-c"/>
<text x="455" y="256" text-anchor="middle">t-SNE map, three planted clusters</text></svg>
Figure 3. Left: the Gaussian bump (blue) falls to nearly zero by distance 4, while the Student-t kernel (orange) keeps a heavy tail. Right: a t-SNE map of the chapter's three planted clusters, 30 points each in 10 columns. The clusters separate, and the 5 nearest map neighbours of every point share its cluster in all 90 cases.
```

### The objective and its gradient

You want $q$ to match $p$. Measure the mismatch with the **KL divergence**:

$$
\mathrm{KL}(P \,\Vert\, Q) = \sum_{i \ne j} p_{ij} \ln \frac{p_{ij}}{q_{ij}} .
$$

Read it aloud as "for each pair, the true weight times the log of true over map weight". It is 0 when $q = p$ and positive otherwise. The log is natural, which the gradient below assumes. A pair with large $p_{ij}$ and tiny $q_{ij}$ costs a lot, so true neighbours cannot be far apart on the map. A pair with tiny $p_{ij}$ and large $q_{ij}$ costs almost nothing, so non-neighbours may sit close. That asymmetry is why t-SNE keeps local structure and loses global structure.

The gradient of the KL with respect to the map point $y_i$ is

$$
\frac{\partial\, \mathrm{KL}}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})\,(y_i - y_j)\,\big(1 + \lVert y_i - y_j \rVert^2\big)^{-1} .
$$

Read it aloud. For every other point $j$, look at the line from $y_j$ to $y_i$. If $p_{ij} > q_{ij}$, the pair should be closer than it is, and the term pulls them together under gradient descent. If $p_{ij} < q_{ij}$, it pushes them apart. The last factor damps the force at long range, which is what the Student-t kernel buys.

Take three points on a line, $y = 0, 1, 2$ (second coordinate 0), and $p_{01} = 0.25$, $p_{02} = 1/6$, $p_{12} = 1/12$ for each ordered pair after symmetrising. The kernel values are $0.5$ for distance 1 and $0.2$ for distance 2. Their sum over ordered pairs is $2(0.5 + 0.5 + 0.2) = 2.4$, so $q_{01} = q_{12} = 0.2083$ and $q_{02} = 0.0833$. The KL is $0.1695$. For point 0, $p_{01} - q_{01} = 0.0417$ and $p_{02} - q_{02} = 0.0833$. The gradient is $4\,[\,0.0417 \cdot 0.5 \cdot (0 - 1) + 0.0833 \cdot 0.2 \cdot (0 - 2)\,] = -0.2167$. Gradient descent moves point 0 by $+0.2167$, toward the others, which is right, since both $p$ values beat their $q$ values. For point 1, the pull toward point 0 and the push away from point 2 (where $p_{12} < q_{12}$) agree, and the gradient is $+0.3333$. The three gradients, $-0.2167$, $0.3333$, and $-0.1167$, sum to 0.

In numpy, with `inv` holding the kernel with a zero diagonal and `Q = inv / inv.sum()`:

```python
M = (P - Q) * inv
grad = 4 * (M.sum(axis=1)[:, None] * Y - M @ Y)
```

The first term is $\sum_j M_{ij}\, y_i$ and the second is $\sum_j M_{ij}\, y_j$, so together they give $\sum_j M_{ij}(y_i - y_j)$ for every row at once.

### Early exaggeration

From random points, plain descent can trap stray points between clumps. **Early exaggeration** multiplies $P$ by a factor such as 12 for the first 100 steps. Pulls between neighbours then dominate, so each clump contracts into a tight ball and the balls slide past each other. After step 100, the true $P$ lets them expand. The update is `v = momentum * v - lr * grad`, then `Y = Y + v`, and you centre `Y` at every step.

## What a t-SNE plot cannot tell you

The map is faithful to neighbourhoods, and to little else.

- **Cluster size means nothing.** The bandwidth adapts to local density, so a tight clump and a loose one can be drawn at the same size. Wattenberg, Viégas, and Johnson show clusters of different spread that look alike in size.
- **Distances between clusters mean little.** The cost only demands that true neighbours land close, so the gaps between clumps are left to the optimiser, and they change with perplexity and the random start. Read the grouping, not the gaps.
- **Perplexity changes the picture.** Try several perplexities and seeds.

UMAP, from McInnes, Healy, and Melville, follows the same plan with different ingredients. It builds a weighted graph of each point's nearest neighbours, then lays points out in 2-D so that the map's graph matches, with its own cost and a heavy-tailed kernel. It runs faster on large tables, but the same cautions about sizes and gaps apply.

> [!KEY]
> t-SNE turns distances into neighbour probabilities with one Gaussian per point, tuned by binary search so every row has the same perplexity. It places points in 2-D with a heavy-tailed Student-t kernel, and it moves them down the gradient of the KL divergence. The result shows who is near whom. It does not show how big a clump is or how far apart two clumps are.

> [!RECAP]
> - Whitening leaves the data rotated by an unknown $R$. Gaussian voices make every $R$ equivalent, so non-Gaussian voices are essential.
> - A projection's excess kurtosis, $E[u^4] - 3$, scores non-Gaussianity. FastICA climbs a contrast with $w \leftarrow E[z\,g(w^{\top}z)] - E[g'(w^{\top}z)]\,w$, then rescales. Deflation keeps the rows orthonormal.
> - Voices come back up to sign, scale, and order, so compare with absolute correlation.
> - Each row of $p_{j\mid i}$ has its own $\beta_i$, found by binary search so that the entropy in bits equals $\log_2$(perplexity). Symmetrise with $(p_{j\mid i} + p_{i\mid j})/(2n)$.
> - The map uses $q_{ij}$ from $1/(1 + d^2)$. The gradient of the KL (natural log) is $4\sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + d_{ij}^2)^{-1}$. Early exaggeration multiplies $P$ by about 12 for the first 100 steps.
> - Read t-SNE for neighbourhoods only. Cluster size and the gaps between clusters are not information.

## Build it step by step

`whiten` is written for you. Build ten functions in this order.

1. **`contrast_terms(u, g)`** takes an array of projections and a name, `"logcosh"` or `"kurtosis"`, and returns `(g(u), g'(u))`, two arrays shaped like `u`. The usual mistake is to return $\log\cosh$ instead of its slope $\tanh$.
2. **`fastica_one_unit(Xw, w0, g, steps)`** takes whitened data `(n, d)`, a start `(d,)`, a contrast name, and a step count. It returns a unit vector `(d,)`. Scale `w0` to length 1, then repeat the update. The usual mistake is to forget `gpu.mean() * w`.
3. **`fastica(Xw, k, rng, g, steps)`** returns `W` of shape `(k, d)` with orthonormal rows. Start each row from `rng.normal`, and deflate inside the loop. The usual mistake is to deflate only after the loop, so every row finds the same voice.
4. **`conditional_p(D, betas)`** takes squared distances `(n, n)` and one `beta` per row, and returns `(n, n)` with a zero diagonal and rows that sum to 1. Set the diagonal logits to `-np.inf` and subtract each row's maximum before `np.exp`. The usual mistake is to skip the maximum, so a distant row underflows to 0 divided by 0.
5. **`perplexity_bandwidths(D, perplexity)`** returns `betas` of shape `(n,)`. Bisect on each row's `beta` until the entropy in bits is within `tol` of `np.log2(perplexity)`. The usual mistake is a base mismatch.
6. **`joint_p(D, perplexity)`** returns `(n, n)`, symmetric, summing to 1. The usual mistake is to divide by `n` instead of `2 * n`, so the table sums to 2.
7. **`student_q(Y)`** takes map points `(n, 2)` and returns `(Q, inv)`, both `(n, n)`. Zero the diagonal of `1 / (1 + d2)` before you divide by the sum. The usual mistake is to leave the diagonal at 1, which adds $n$ to the denominator.
8. **`kl_divergence(P, Q)`** returns the float $\sum p \ln(p/q)$ over entries where `P > 0`. The usual mistake is `0 * log 0`, which gives `nan`, so mask the zeros first.
9. **`tsne_grad(P, Y)`** returns `(n, 2)`. The usual mistake is to drop the `inv` factor, which gives the gradient of the Gaussian version (SNE).
10. **`tsne_embed(P, Y0, steps, lr, momentum, exaggeration, exag_steps)`** returns the final map `(n, 2)`. The usual mistake is to leave the exaggeration on for the whole run, so the clusters never expand.

## Cost and scaling

FastICA costs about $n d$ operations per update per direction. t-SNE is expensive, because $P$ and $Q$ are $n \times n$: building the distances costs about $n^2 d$, and every gradient step about $n^2$ time and memory. That is fine for a few thousand points and slow for a million.

## Where this breaks

- ICA needs at most one Gaussian voice. Two or more Gaussian voices stay mixed among themselves.
- A heavy outlier can drag a kurtosis-based update to itself. Prefer log cosh on messy data.
- A t-SNE step size that is too small leaves a ball, and one that is too large throws points far away. Watch the KL fall.

## The exercise

You write `contrast_terms`, `fastica_one_unit`, `fastica`, `conditional_p`, `perplexity_bandwidths`, `joint_p`, `student_q`, `kl_divergence`, `tsne_grad`, and `tsne_embed`. The checks confirm the hand numbers: $g$ and $g'$ for both contrasts, the update that turns $(0.6, 0.8)$ into $(-0.705, -0.709)$, the row $(0.9526, 0.0474)$, and the kernel values $0.5$ and $0.2$ with $q = 0.2083$ and $0.0833$.

Then they plant voices and recover them, with a correlation of 0.99 for one voice and a distinct match for each of three, which catches a missing deflation step. The perplexity check measures every row's perplexity in bits for targets 5, 15, and 40 and the boundary $n - 1$, which catches a base mismatch. `tsne_grad` is compared with finite differences of the KL, which catches the Gaussian gradient. The last check runs `tsne_embed` on three planted clusters and requires the KL to halve and 95 percent of each point's 5 nearest map neighbours to share its cluster.

## Further reading

- For the ICA model, the ambiguities of sign, scale, and order, and the Gaussian case, see the class notes "Independent Components Analysis" listed on the Autumn 2018 [CS229 syllabus](https://cs229.stanford.edu/syllabus-autumn2018.html), and Lecture 15, "Principal Component Analysis. Independent Component Analysis."
- For the contrasts, the fixed-point update, and deflation, see Hyvärinen and Oja, "Independent component analysis: algorithms and applications", Neural Networks 13(4-5), 2000.
- For the original t-SNE method, see van der Maaten and Hinton, "Visualizing data using t-SNE", Journal of Machine Learning Research 9, 2008.
- For what the plots can and cannot say, see Wattenberg, Viégas, and Johnson, [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/), Distill, 2016. Its sections "Cluster sizes in a t-SNE plot mean nothing" and "Distances between clusters might not mean anything" cover the two traps above.
- For UMAP, see McInnes, Healy, and Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction", arXiv:1802.03426, 2018.
