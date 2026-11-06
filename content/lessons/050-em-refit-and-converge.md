---
title: "EM: refit the blobs and watch the likelihood rise"
slug: 050-em-refit-and-converge
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Refit each blob as a share-weighted mean and covariance, and alternate with the E-step. The log-likelihood never falls, and tiny blobs give k-means.
minutes: 45
gpu: false
objectives:
  - Refit a Gaussian mixture's weights, means, and covariances as share-weighted versions of the ordinary mean and covariance.
  - Explain why one E-step followed by one M-step can never lower the log-likelihood, using the bound and the gap.
  - Compute the bound, and check that it touches log L when the shares are the posterior and sits below it otherwise.
  - Run the full EM loop from k-means++ seeds, and explain why tiny equal round blobs turn EM into k-means.
prereqs:
  - 047-probabilistic-pca-em
  - 048-k-means
  - 049-gmm-soft-assignment
exercise: 050-em-refit-and-converge
---

# EM: refit the blobs and watch the likelihood rise

For the two points at $0$ and $2$ the E-step said that point $0$ belongs 88 percent to blob 0 and 12 percent to blob 1, and the point at $2$ the other way round. If you trust those shares for a moment, how should blob 0 move? Its old mean was $0$. The point at $2$ owns only 12 percent of it, so blob 0 should shift a little toward $2$, not all the way and not at all.

Without that idea you are stuck on both sides. If you freeze the blobs, nothing improves. If you try to find the best blobs in one go, the log of a sum from the last chapter blocks you. And if you snap each share to the larger one and take plain means, you are back to k-means and its brittle verdicts.

This chapter shows how to refit the blobs from the shares, which is the M-step, and why alternating the E-step and the M-step can never make the fit worse. It also shows that k-means is what EM becomes when the blobs shrink.

## What you already know

Four ideas carry this chapter:

- The E-step gives each point a share in each blob, and the log-likelihood $\log L$ scores how well the blobs explain the data (see "Gaussian mixtures and soft assignment").
- k-means moves each centre to the mean of its group, and k-means++ seeds good starting centres (see "k-means: Lloyd's steps and a good start").
- EM is a guess-and-refit loop whose log-likelihood rises every round (see "A noisy model for PCA, fitted by guess and refit").
- A mean is a balance point, and a covariance is the average of the products of gaps around it.

## Refit with weighted statistics

Here is the whole idea. Compute the ordinary count, mean, and covariance of the points that belong to a blob, but let each point count in proportion to its share. A point with share $0.12$ counts as $0.12$ of a point.

The **soft count** of blob $j$ is $N_j=\sum_i r_{ij}$, the effective number of points the blob owns. For blob 0 in the two-point example, $N_0=0.8808+0.1192=1$. The three refits are then:

$$
\pi_j = \frac{N_j}{n}, \qquad
\mu_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, x_i, \qquad
\Sigma_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, (x_i - \mu_j)(x_i - \mu_j)^\top + \epsilon I .
$$

Read them aloud. The weight is the blob's share of all $n$ points. The mean is the weighted average of the points. The covariance is the weighted average of the products of the gaps from that new mean, plus a tiny $\epsilon I$ on the diagonal that you meet later. If every share were exactly 0 or 1, $N_j$ would be the size of a group, $\mu_j$ would be that group's plain mean, and $\Sigma_j$ its plain covariance. For the means, that is exactly the k-means update.

Check it on the two points. The new mean of blob 0 is $(0.8808\cdot 0+0.1192\cdot 2)/1=0.2384$. The squared gaps from it are $0.0568$ and $3.1032$. Weighted, they are $0.8808\cdot0.0568=0.0501$ and $0.1192\cdot3.1032=0.3699$, which add to $0.4200$, so the new variance is $0.42$ and the weight is $1/2$. By symmetry blob 1 moves to $1.7616$ with the same variance. Each blob moved a little toward the other point, which it owns only a small share of. It also narrowed from $1$ to $0.42$, because blob 0 now describes mostly the one point at $0$ plus a small slice of the point at $2$, which is a tighter bunch than a blob of variance $1$ assumed.

In numpy, for the means:

```python
counts = responsibilities.sum(axis=0)              # N_j, shape (k,)
means = (responsibilities.T @ X) / counts[:, None]
```

> [!WARNING]
> The covariance must be measured from the new mean $\mu_j$, the one you just computed in this step. Gaps measured from any other centre, such as the old mean the E-step used or the plain average of all points, give a covariance that is too wide, because the weighted mean is the centre that makes the weighted squared gaps smallest.

## Alternate, and watch

The **EM loop** runs the E-step, then the M-step, then the E-step again. The E stands for expectation, the average over the hidden choices that the shares describe, and the M for maximisation, because the refit picks the best blobs for those shares. Start with three equal round blobs, with means at the k-means++ seeds, every covariance equal to the average variance of the data, and weights $1/3$. Figure 1 shows 60 points from three clusters under this loop. Each frame prints $\log L$.

```figure
<svg viewBox="0 0 600 530" role="img" aria-label="Four frames of EM on 60 points from three clusters: round equal ellipses at the start, then ellipses that move and reshape to cover each cluster, with log L rising from -292.7 to -215.8">
<defs><clipPath id="c056-p0"><rect x="20" y="40" width="270" height="222"/></clipPath><clipPath id="c056-p1"><rect x="310" y="40" width="270" height="222"/></clipPath><clipPath id="c056-p3"><rect x="20" y="295" width="270" height="222"/></clipPath><clipPath id="c056-p10"><rect x="310" y="295" width="270" height="222"/></clipPath></defs>
<rect x="20.0" y="40.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="20.0" y="32.0" class="fig-bold" text-anchor="start" >Start</text>
<text x="290.0" y="32.0" class="fig-small" text-anchor="end" >log L = -292.7</text>
<g clip-path="url(#c056-p0)">
<circle cx="121.4" cy="253.3" r="4" class="fill-c" opacity="0.53"/>
<circle cx="121.4" cy="253.3" r="4" class="fill-d" opacity="0.46"/>
<circle cx="89.6" cy="214.2" r="4" class="fill-a" opacity="0.02"/>
<circle cx="89.6" cy="214.2" r="4" class="fill-c" opacity="0.26"/>
<circle cx="89.6" cy="214.2" r="4" class="fill-d" opacity="0.73"/>
<circle cx="72.5" cy="207.3" r="4" class="fill-a" opacity="0.02"/>
<circle cx="72.5" cy="207.3" r="4" class="fill-c" opacity="0.17"/>
<circle cx="72.5" cy="207.3" r="4" class="fill-d" opacity="0.81"/>
<circle cx="41.7" cy="207.6" r="4" class="fill-c" opacity="0.08"/>
<circle cx="41.7" cy="207.6" r="4" class="fill-d" opacity="0.91"/>
<circle cx="64.4" cy="137.8" r="4" class="fill-a" opacity="0.09"/>
<circle cx="64.4" cy="137.8" r="4" class="fill-c" opacity="0.07"/>
<circle cx="64.4" cy="137.8" r="4" class="fill-d" opacity="0.83"/>
<circle cx="85.8" cy="210.0" r="4" class="fill-a" opacity="0.02"/>
<circle cx="85.8" cy="210.0" r="4" class="fill-c" opacity="0.23"/>
<circle cx="85.8" cy="210.0" r="4" class="fill-d" opacity="0.75"/>
<circle cx="75.8" cy="216.2" r="4" class="fill-a" opacity="0.01"/>
<circle cx="75.8" cy="216.2" r="4" class="fill-c" opacity="0.20"/>
<circle cx="75.8" cy="216.2" r="4" class="fill-d" opacity="0.79"/>
<circle cx="60.6" cy="210.8" r="4" class="fill-a" opacity="0.01"/>
<circle cx="60.6" cy="210.8" r="4" class="fill-c" opacity="0.13"/>
<circle cx="60.6" cy="210.8" r="4" class="fill-d" opacity="0.85"/>
<circle cx="90.8" cy="207.8" r="4" class="fill-a" opacity="0.02"/>
<circle cx="90.8" cy="207.8" r="4" class="fill-c" opacity="0.25"/>
<circle cx="90.8" cy="207.8" r="4" class="fill-d" opacity="0.73"/>
<circle cx="100.2" cy="207.0" r="4" class="fill-a" opacity="0.03"/>
<circle cx="100.2" cy="207.0" r="4" class="fill-c" opacity="0.30"/>
<circle cx="100.2" cy="207.0" r="4" class="fill-d" opacity="0.68"/>
<circle cx="81.8" cy="172.7" r="4" class="fill-a" opacity="0.05"/>
<circle cx="81.8" cy="172.7" r="4" class="fill-c" opacity="0.16"/>
<circle cx="81.8" cy="172.7" r="4" class="fill-d" opacity="0.79"/>
<circle cx="92.1" cy="213.0" r="4" class="fill-a" opacity="0.02"/>
<circle cx="92.1" cy="213.0" r="4" class="fill-c" opacity="0.27"/>
<circle cx="92.1" cy="213.0" r="4" class="fill-d" opacity="0.71"/>
<circle cx="77.8" cy="192.5" r="4" class="fill-a" opacity="0.03"/>
<circle cx="77.8" cy="192.5" r="4" class="fill-c" opacity="0.17"/>
<circle cx="77.8" cy="192.5" r="4" class="fill-d" opacity="0.80"/>
<circle cx="119.4" cy="208.4" r="4" class="fill-a" opacity="0.04"/>
<circle cx="119.4" cy="208.4" r="4" class="fill-c" opacity="0.41"/>
<circle cx="119.4" cy="208.4" r="4" class="fill-d" opacity="0.55"/>
<circle cx="76.6" cy="183.4" r="4" class="fill-a" opacity="0.03"/>
<circle cx="76.6" cy="183.4" r="4" class="fill-c" opacity="0.15"/>
<circle cx="76.6" cy="183.4" r="4" class="fill-d" opacity="0.81"/>
<circle cx="64.0" cy="208.8" r="4" class="fill-a" opacity="0.01"/>
<circle cx="64.0" cy="208.8" r="4" class="fill-c" opacity="0.14"/>
<circle cx="64.0" cy="208.8" r="4" class="fill-d" opacity="0.85"/>
<circle cx="98.7" cy="191.7" r="4" class="fill-a" opacity="0.04"/>
<circle cx="98.7" cy="191.7" r="4" class="fill-c" opacity="0.26"/>
<circle cx="98.7" cy="191.7" r="4" class="fill-d" opacity="0.70"/>
<circle cx="83.2" cy="189.9" r="4" class="fill-a" opacity="0.03"/>
<circle cx="83.2" cy="189.9" r="4" class="fill-c" opacity="0.19"/>
<circle cx="83.2" cy="189.9" r="4" class="fill-d" opacity="0.78"/>
<circle cx="25.8" cy="183.0" r="4" class="fill-a" opacity="0.01"/>
<circle cx="25.8" cy="183.0" r="4" class="fill-c" opacity="0.05"/>
<circle cx="25.8" cy="183.0" r="4" class="fill-d" opacity="0.94"/>
<circle cx="62.5" cy="235.9" r="4" class="fill-c" opacity="0.17"/>
<circle cx="62.5" cy="235.9" r="4" class="fill-d" opacity="0.82"/>
<circle cx="234.1" cy="189.3" r="4" class="fill-a" opacity="0.14"/>
<circle cx="234.1" cy="189.3" r="4" class="fill-c" opacity="0.80"/>
<circle cx="234.1" cy="189.3" r="4" class="fill-d" opacity="0.06"/>
<circle cx="219.9" cy="224.2" r="4" class="fill-a" opacity="0.04"/>
<circle cx="219.9" cy="224.2" r="4" class="fill-c" opacity="0.89"/>
<circle cx="219.9" cy="224.2" r="4" class="fill-d" opacity="0.07"/>
<circle cx="229.1" cy="204.1" r="4" class="fill-a" opacity="0.09"/>
<circle cx="229.1" cy="204.1" r="4" class="fill-c" opacity="0.85"/>
<circle cx="229.1" cy="204.1" r="4" class="fill-d" opacity="0.06"/>
<circle cx="256.2" cy="188.4" r="4" class="fill-a" opacity="0.15"/>
<circle cx="256.2" cy="188.4" r="4" class="fill-c" opacity="0.82"/>
<circle cx="256.2" cy="188.4" r="4" class="fill-d" opacity="0.03"/>
<circle cx="232.4" cy="181.3" r="4" class="fill-a" opacity="0.18"/>
<circle cx="232.4" cy="181.3" r="4" class="fill-c" opacity="0.75"/>
<circle cx="232.4" cy="181.3" r="4" class="fill-d" opacity="0.06"/>
<circle cx="224.6" cy="221.3" r="4" class="fill-a" opacity="0.05"/>
<circle cx="224.6" cy="221.3" r="4" class="fill-c" opacity="0.89"/>
<circle cx="224.6" cy="221.3" r="4" class="fill-d" opacity="0.06"/>
<circle cx="240.1" cy="191.7" r="4" class="fill-a" opacity="0.13"/>
<circle cx="240.1" cy="191.7" r="4" class="fill-c" opacity="0.82"/>
<circle cx="240.1" cy="191.7" r="4" class="fill-d" opacity="0.05"/>
<circle cx="224.4" cy="218.5" r="4" class="fill-a" opacity="0.05"/>
<circle cx="224.4" cy="218.5" r="4" class="fill-c" opacity="0.88"/>
<circle cx="224.4" cy="218.5" r="4" class="fill-d" opacity="0.06"/>
<circle cx="233.1" cy="252.1" r="4" class="fill-a" opacity="0.02"/>
<circle cx="233.1" cy="252.1" r="4" class="fill-c" opacity="0.94"/>
<circle cx="233.1" cy="252.1" r="4" class="fill-d" opacity="0.04"/>
<circle cx="242.2" cy="193.4" r="4" class="fill-a" opacity="0.13"/>
<circle cx="242.2" cy="193.4" r="4" class="fill-c" opacity="0.83"/>
<circle cx="242.2" cy="193.4" r="4" class="fill-d" opacity="0.05"/>
<circle cx="196.5" cy="201.9" r="4" class="fill-a" opacity="0.09"/>
<circle cx="196.5" cy="201.9" r="4" class="fill-c" opacity="0.77"/>
<circle cx="196.5" cy="201.9" r="4" class="fill-d" opacity="0.14"/>
<circle cx="209.7" cy="188.2" r="4" class="fill-a" opacity="0.14"/>
<circle cx="209.7" cy="188.2" r="4" class="fill-c" opacity="0.75"/>
<circle cx="209.7" cy="188.2" r="4" class="fill-d" opacity="0.11"/>
<circle cx="188.7" cy="221.0" r="4" class="fill-a" opacity="0.04"/>
<circle cx="188.7" cy="221.0" r="4" class="fill-c" opacity="0.80"/>
<circle cx="188.7" cy="221.0" r="4" class="fill-d" opacity="0.15"/>
<circle cx="242.6" cy="180.4" r="4" class="fill-a" opacity="0.19"/>
<circle cx="242.6" cy="180.4" r="4" class="fill-c" opacity="0.76"/>
<circle cx="242.6" cy="180.4" r="4" class="fill-d" opacity="0.05"/>
<circle cx="186.3" cy="212.9" r="4" class="fill-a" opacity="0.06"/>
<circle cx="186.3" cy="212.9" r="4" class="fill-c" opacity="0.77"/>
<circle cx="186.3" cy="212.9" r="4" class="fill-d" opacity="0.17"/>
<circle cx="235.1" cy="215.1" r="4" class="fill-a" opacity="0.06"/>
<circle cx="235.1" cy="215.1" r="4" class="fill-c" opacity="0.89"/>
<circle cx="235.1" cy="215.1" r="4" class="fill-d" opacity="0.05"/>
<circle cx="259.9" cy="226.5" r="4" class="fill-a" opacity="0.04"/>
<circle cx="259.9" cy="226.5" r="4" class="fill-c" opacity="0.93"/>
<circle cx="259.9" cy="226.5" r="4" class="fill-d" opacity="0.02"/>
<circle cx="235.6" cy="223.7" r="4" class="fill-a" opacity="0.05"/>
<circle cx="235.6" cy="223.7" r="4" class="fill-c" opacity="0.91"/>
<circle cx="235.6" cy="223.7" r="4" class="fill-d" opacity="0.05"/>
<circle cx="256.2" cy="203.5" r="4" class="fill-a" opacity="0.09"/>
<circle cx="256.2" cy="203.5" r="4" class="fill-c" opacity="0.88"/>
<circle cx="256.2" cy="203.5" r="4" class="fill-d" opacity="0.03"/>
<circle cx="221.3" cy="236.8" r="4" class="fill-a" opacity="0.03"/>
<circle cx="221.3" cy="236.8" r="4" class="fill-c" opacity="0.91"/>
<circle cx="221.3" cy="236.8" r="4" class="fill-d" opacity="0.06"/>
<circle cx="188.0" cy="65.6" r="4" class="fill-a" opacity="0.91"/>
<circle cx="188.0" cy="65.6" r="4" class="fill-c" opacity="0.05"/>
<circle cx="188.0" cy="65.6" r="4" class="fill-d" opacity="0.04"/>
<circle cx="169.8" cy="58.0" r="4" class="fill-a" opacity="0.91"/>
<circle cx="169.8" cy="58.0" r="4" class="fill-c" opacity="0.04"/>
<circle cx="169.8" cy="58.0" r="4" class="fill-d" opacity="0.05"/>
<circle cx="161.9" cy="92.9" r="4" class="fill-a" opacity="0.75"/>
<circle cx="161.9" cy="92.9" r="4" class="fill-c" opacity="0.11"/>
<circle cx="161.9" cy="92.9" r="4" class="fill-d" opacity="0.14"/>
<circle cx="139.3" cy="96.1" r="4" class="fill-a" opacity="0.65"/>
<circle cx="139.3" cy="96.1" r="4" class="fill-c" opacity="0.11"/>
<circle cx="139.3" cy="96.1" r="4" class="fill-d" opacity="0.24"/>
<circle cx="181.9" cy="109.0" r="4" class="fill-a" opacity="0.69"/>
<circle cx="181.9" cy="109.0" r="4" class="fill-c" opacity="0.19"/>
<circle cx="181.9" cy="109.0" r="4" class="fill-d" opacity="0.12"/>
<circle cx="143.3" cy="86.7" r="4" class="fill-a" opacity="0.73"/>
<circle cx="143.3" cy="86.7" r="4" class="fill-c" opacity="0.09"/>
<circle cx="143.3" cy="86.7" r="4" class="fill-d" opacity="0.19"/>
<circle cx="159.4" cy="69.1" r="4" class="fill-a" opacity="0.86"/>
<circle cx="159.4" cy="69.1" r="4" class="fill-c" opacity="0.05"/>
<circle cx="159.4" cy="69.1" r="4" class="fill-d" opacity="0.09"/>
<circle cx="130.5" cy="114.3" r="4" class="fill-a" opacity="0.47"/>
<circle cx="130.5" cy="114.3" r="4" class="fill-c" opacity="0.16"/>
<circle cx="130.5" cy="114.3" r="4" class="fill-d" opacity="0.37"/>
<circle cx="154.9" cy="56.5" r="4" class="fill-a" opacity="0.89"/>
<circle cx="154.9" cy="56.5" r="4" class="fill-c" opacity="0.04"/>
<circle cx="154.9" cy="56.5" r="4" class="fill-d" opacity="0.07"/>
<circle cx="169.9" cy="76.1" r="4" class="fill-a" opacity="0.85"/>
<circle cx="169.9" cy="76.1" r="4" class="fill-c" opacity="0.07"/>
<circle cx="169.9" cy="76.1" r="4" class="fill-d" opacity="0.08"/>
<circle cx="148.8" cy="74.6" r="4" class="fill-a" opacity="0.81"/>
<circle cx="148.8" cy="74.6" r="4" class="fill-c" opacity="0.06"/>
<circle cx="148.8" cy="74.6" r="4" class="fill-d" opacity="0.13"/>
<circle cx="150.2" cy="64.3" r="4" class="fill-a" opacity="0.86"/>
<circle cx="150.2" cy="64.3" r="4" class="fill-c" opacity="0.05"/>
<circle cx="150.2" cy="64.3" r="4" class="fill-d" opacity="0.10"/>
<circle cx="139.4" cy="77.7" r="4" class="fill-a" opacity="0.76"/>
<circle cx="139.4" cy="77.7" r="4" class="fill-c" opacity="0.07"/>
<circle cx="139.4" cy="77.7" r="4" class="fill-d" opacity="0.17"/>
<circle cx="152.8" cy="69.7" r="4" class="fill-a" opacity="0.84"/>
<circle cx="152.8" cy="69.7" r="4" class="fill-c" opacity="0.05"/>
<circle cx="152.8" cy="69.7" r="4" class="fill-d" opacity="0.10"/>
<circle cx="159.4" cy="30.3" r="4" class="fill-a" opacity="0.95"/>
<circle cx="159.4" cy="30.3" r="4" class="fill-c" opacity="0.01"/>
<circle cx="159.4" cy="30.3" r="4" class="fill-d" opacity="0.03"/>
<circle cx="184.4" cy="51.0" r="4" class="fill-a" opacity="0.94"/>
<circle cx="184.4" cy="51.0" r="4" class="fill-c" opacity="0.03"/>
<circle cx="184.4" cy="51.0" r="4" class="fill-d" opacity="0.03"/>
<circle cx="115.0" cy="87.0" r="4" class="fill-a" opacity="0.60"/>
<circle cx="115.0" cy="87.0" r="4" class="fill-c" opacity="0.07"/>
<circle cx="115.0" cy="87.0" r="4" class="fill-d" opacity="0.33"/>
<circle cx="143.0" cy="69.9" r="4" class="fill-a" opacity="0.82"/>
<circle cx="143.0" cy="69.9" r="4" class="fill-c" opacity="0.05"/>
<circle cx="143.0" cy="69.9" r="4" class="fill-d" opacity="0.13"/>
<circle cx="110.2" cy="57.3" r="4" class="fill-a" opacity="0.76"/>
<circle cx="110.2" cy="57.3" r="4" class="fill-c" opacity="0.03"/>
<circle cx="110.2" cy="57.3" r="4" class="fill-d" opacity="0.21"/>
<circle cx="176.0" cy="105.9" r="4" class="fill-a" opacity="0.70"/>
<circle cx="176.0" cy="105.9" r="4" class="fill-c" opacity="0.17"/>
<circle cx="176.0" cy="105.9" r="4" class="fill-d" opacity="0.13"/>
<polyline points="252.7,65.6 252.3,58.8 251.3,52.1 249.5,45.6 247.1,39.3 244.0,33.3 240.3,27.6 236.1,22.3 231.3,17.5 226.0,13.3 220.4,9.6 214.3,6.5 208.0,4.1 201.5,2.3 194.8,1.3 188.0,0.9 181.3,1.3 174.6,2.3 168.0,4.1 161.7,6.5 155.7,9.6 150.0,13.3 144.8,17.5 140.0,22.3 135.7,27.6 132.0,33.3 129.0,39.3 126.5,45.6 124.8,52.1 123.7,58.8 123.4,65.6 123.7,72.3 124.8,79.0 126.5,85.6 129.0,91.9 132.0,97.9 135.7,103.6 140.0,108.8 144.8,113.6 150.0,117.9 155.7,121.6 161.7,124.6 168.0,127.1 174.6,128.8 181.3,129.9 188.0,130.2 194.8,129.9 201.5,128.8 208.0,127.1 214.3,124.6 220.4,121.6 226.0,117.9 231.3,113.6 236.1,108.8 240.3,103.6 244.0,97.9 247.1,91.9 249.5,85.6 251.3,79.0 252.3,72.3 252.7,65.6" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="183.0" y1="60.6" x2="193.0" y2="70.6" class="stroke-ink" stroke-width="2"/><line x1="183.0" y1="70.6" x2="193.0" y2="60.6" class="stroke-ink" stroke-width="2"/>
<polyline points="253.3,221.0 253.0,214.3 251.9,207.6 250.2,201.1 247.8,194.8 244.7,188.7 241.0,183.0 236.7,177.8 232.0,173.0 226.7,168.7 221.0,165.1 215.0,162.0 208.7,159.6 202.1,157.8 195.5,156.7 188.7,156.4 181.9,156.7 175.3,157.8 168.7,159.6 162.4,162.0 156.4,165.1 150.7,168.7 145.4,173.0 140.6,177.8 136.4,183.0 132.7,188.7 129.6,194.8 127.2,201.1 125.5,207.6 124.4,214.3 124.0,221.0 124.4,227.8 125.5,234.5 127.2,241.0 129.6,247.3 132.7,253.4 136.4,259.1 140.6,264.3 145.4,269.1 150.7,273.4 156.4,277.0 162.4,280.1 168.7,282.5 175.3,284.3 181.9,285.3 188.7,285.7 195.5,285.3 202.1,284.3 208.7,282.5 215.0,280.1 221.0,277.0 226.7,273.4 232.0,269.1 236.7,264.3 241.0,259.1 244.7,253.4 247.8,247.3 250.2,241.0 251.9,234.5 253.0,227.8 253.3,221.0" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="183.7" y1="216.0" x2="193.7" y2="226.0" class="stroke-ink" stroke-width="2"/><line x1="183.7" y1="226.0" x2="193.7" y2="216.0" class="stroke-ink" stroke-width="2"/>
<polyline points="141.2,183.4 140.9,176.7 139.8,170.0 138.1,163.4 135.6,157.1 132.6,151.1 128.9,145.4 124.6,140.1 119.8,135.4 114.6,131.1 108.9,127.4 102.9,124.3 96.6,121.9 90.0,120.2 83.3,119.1 76.6,118.8 69.8,119.1 63.1,120.2 56.6,121.9 50.3,124.3 44.3,127.4 38.6,131.1 33.3,135.4 28.5,140.1 24.3,145.4 20.6,151.1 17.5,157.1 15.1,163.4 13.3,170.0 12.3,176.7 11.9,183.4 12.3,190.2 13.3,196.9 15.1,203.4 17.5,209.7 20.6,215.7 24.3,221.4 28.5,226.7 33.3,231.5 38.6,235.7 44.3,239.4 50.3,242.5 56.6,244.9 63.1,246.7 69.8,247.7 76.6,248.1 83.3,247.7 90.0,246.7 96.6,244.9 102.9,242.5 108.9,239.4 114.6,235.7 119.8,231.5 124.6,226.7 128.9,221.4 132.6,215.7 135.6,209.7 138.1,203.4 139.8,196.9 140.9,190.2 141.2,183.4" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="71.6" y1="178.4" x2="81.6" y2="188.4" class="stroke-ink" stroke-width="2"/><line x1="71.6" y1="188.4" x2="81.6" y2="178.4" class="stroke-ink" stroke-width="2"/>
</g>
<rect x="310.0" y="40.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="310.0" y="32.0" class="fig-bold" text-anchor="start" >After 1 round</text>
<text x="580.0" y="32.0" class="fig-small" text-anchor="end" >log L = -259.9</text>
<g clip-path="url(#c056-p1)">
<circle cx="411.4" cy="253.3" r="4" class="fill-c" opacity="0.38"/>
<circle cx="411.4" cy="253.3" r="4" class="fill-d" opacity="0.62"/>
<circle cx="379.6" cy="214.2" r="4" class="fill-c" opacity="0.16"/>
<circle cx="379.6" cy="214.2" r="4" class="fill-d" opacity="0.84"/>
<circle cx="362.5" cy="207.3" r="4" class="fill-c" opacity="0.11"/>
<circle cx="362.5" cy="207.3" r="4" class="fill-d" opacity="0.89"/>
<circle cx="331.7" cy="207.6" r="4" class="fill-c" opacity="0.05"/>
<circle cx="331.7" cy="207.6" r="4" class="fill-d" opacity="0.95"/>
<circle cx="354.4" cy="137.8" r="4" class="fill-c" opacity="0.08"/>
<circle cx="354.4" cy="137.8" r="4" class="fill-d" opacity="0.92"/>
<circle cx="375.8" cy="210.0" r="4" class="fill-c" opacity="0.15"/>
<circle cx="375.8" cy="210.0" r="4" class="fill-d" opacity="0.85"/>
<circle cx="365.8" cy="216.2" r="4" class="fill-c" opacity="0.11"/>
<circle cx="365.8" cy="216.2" r="4" class="fill-d" opacity="0.89"/>
<circle cx="350.6" cy="210.8" r="4" class="fill-c" opacity="0.08"/>
<circle cx="350.6" cy="210.8" r="4" class="fill-d" opacity="0.92"/>
<circle cx="380.8" cy="207.8" r="4" class="fill-c" opacity="0.17"/>
<circle cx="380.8" cy="207.8" r="4" class="fill-d" opacity="0.83"/>
<circle cx="390.2" cy="207.0" r="4" class="fill-c" opacity="0.23"/>
<circle cx="390.2" cy="207.0" r="4" class="fill-d" opacity="0.77"/>
<circle cx="371.8" cy="172.7" r="4" class="fill-c" opacity="0.13"/>
<circle cx="371.8" cy="172.7" r="4" class="fill-d" opacity="0.86"/>
<circle cx="382.1" cy="213.0" r="4" class="fill-c" opacity="0.18"/>
<circle cx="382.1" cy="213.0" r="4" class="fill-d" opacity="0.82"/>
<circle cx="367.8" cy="192.5" r="4" class="fill-c" opacity="0.13"/>
<circle cx="367.8" cy="192.5" r="4" class="fill-d" opacity="0.87"/>
<circle cx="409.4" cy="208.4" r="4" class="fill-c" opacity="0.37"/>
<circle cx="409.4" cy="208.4" r="4" class="fill-d" opacity="0.62"/>
<circle cx="366.6" cy="183.4" r="4" class="fill-c" opacity="0.12"/>
<circle cx="366.6" cy="183.4" r="4" class="fill-d" opacity="0.88"/>
<circle cx="354.0" cy="208.8" r="4" class="fill-c" opacity="0.08"/>
<circle cx="354.0" cy="208.8" r="4" class="fill-d" opacity="0.92"/>
<circle cx="388.7" cy="191.7" r="4" class="fill-c" opacity="0.21"/>
<circle cx="388.7" cy="191.7" r="4" class="fill-d" opacity="0.79"/>
<circle cx="373.2" cy="189.9" r="4" class="fill-c" opacity="0.14"/>
<circle cx="373.2" cy="189.9" r="4" class="fill-d" opacity="0.86"/>
<circle cx="315.8" cy="183.0" r="4" class="fill-c" opacity="0.05"/>
<circle cx="315.8" cy="183.0" r="4" class="fill-d" opacity="0.95"/>
<circle cx="352.5" cy="235.9" r="4" class="fill-c" opacity="0.06"/>
<circle cx="352.5" cy="235.9" r="4" class="fill-d" opacity="0.94"/>
<circle cx="524.1" cy="189.3" r="4" class="fill-a" opacity="0.04"/>
<circle cx="524.1" cy="189.3" r="4" class="fill-c" opacity="0.95"/>
<circle cx="509.9" cy="224.2" r="4" class="fill-a" opacity="0.01"/>
<circle cx="509.9" cy="224.2" r="4" class="fill-c" opacity="0.98"/>
<circle cx="519.1" cy="204.1" r="4" class="fill-a" opacity="0.03"/>
<circle cx="519.1" cy="204.1" r="4" class="fill-c" opacity="0.97"/>
<circle cx="546.2" cy="188.4" r="4" class="fill-a" opacity="0.02"/>
<circle cx="546.2" cy="188.4" r="4" class="fill-c" opacity="0.98"/>
<circle cx="522.4" cy="181.3" r="4" class="fill-a" opacity="0.07"/>
<circle cx="522.4" cy="181.3" r="4" class="fill-c" opacity="0.92"/>
<circle cx="522.4" cy="181.3" r="4" class="fill-d" opacity="0.01"/>
<circle cx="514.6" cy="221.3" r="4" class="fill-a" opacity="0.01"/>
<circle cx="514.6" cy="221.3" r="4" class="fill-c" opacity="0.98"/>
<circle cx="530.1" cy="191.7" r="4" class="fill-a" opacity="0.03"/>
<circle cx="530.1" cy="191.7" r="4" class="fill-c" opacity="0.96"/>
<circle cx="514.4" cy="218.5" r="4" class="fill-a" opacity="0.02"/>
<circle cx="514.4" cy="218.5" r="4" class="fill-c" opacity="0.98"/>
<circle cx="523.1" cy="252.1" r="4" class="fill-c" opacity="0.99"/>
<circle cx="532.2" cy="193.4" r="4" class="fill-a" opacity="0.03"/>
<circle cx="532.2" cy="193.4" r="4" class="fill-c" opacity="0.97"/>
<circle cx="486.5" cy="201.9" r="4" class="fill-a" opacity="0.05"/>
<circle cx="486.5" cy="201.9" r="4" class="fill-c" opacity="0.91"/>
<circle cx="486.5" cy="201.9" r="4" class="fill-d" opacity="0.04"/>
<circle cx="499.7" cy="188.2" r="4" class="fill-a" opacity="0.09"/>
<circle cx="499.7" cy="188.2" r="4" class="fill-c" opacity="0.88"/>
<circle cx="499.7" cy="188.2" r="4" class="fill-d" opacity="0.03"/>
<circle cx="478.7" cy="221.0" r="4" class="fill-a" opacity="0.02"/>
<circle cx="478.7" cy="221.0" r="4" class="fill-c" opacity="0.93"/>
<circle cx="478.7" cy="221.0" r="4" class="fill-d" opacity="0.05"/>
<circle cx="532.6" cy="180.4" r="4" class="fill-a" opacity="0.05"/>
<circle cx="532.6" cy="180.4" r="4" class="fill-c" opacity="0.94"/>
<circle cx="476.3" cy="212.9" r="4" class="fill-a" opacity="0.03"/>
<circle cx="476.3" cy="212.9" r="4" class="fill-c" opacity="0.91"/>
<circle cx="476.3" cy="212.9" r="4" class="fill-d" opacity="0.06"/>
<circle cx="525.1" cy="215.1" r="4" class="fill-a" opacity="0.01"/>
<circle cx="525.1" cy="215.1" r="4" class="fill-c" opacity="0.98"/>
<circle cx="549.9" cy="226.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="525.6" cy="223.7" r="4" class="fill-c" opacity="0.99"/>
<circle cx="546.2" cy="203.5" r="4" class="fill-c" opacity="0.99"/>
<circle cx="511.3" cy="236.8" r="4" class="fill-c" opacity="0.99"/>
<circle cx="478.0" cy="65.6" r="4" class="fill-a" opacity="0.98"/>
<circle cx="478.0" cy="65.6" r="4" class="fill-d" opacity="0.02"/>
<circle cx="459.8" cy="58.0" r="4" class="fill-a" opacity="0.98"/>
<circle cx="459.8" cy="58.0" r="4" class="fill-d" opacity="0.02"/>
<circle cx="451.9" cy="92.9" r="4" class="fill-a" opacity="0.93"/>
<circle cx="451.9" cy="92.9" r="4" class="fill-d" opacity="0.06"/>
<circle cx="429.3" cy="96.1" r="4" class="fill-a" opacity="0.86"/>
<circle cx="429.3" cy="96.1" r="4" class="fill-c" opacity="0.01"/>
<circle cx="429.3" cy="96.1" r="4" class="fill-d" opacity="0.13"/>
<circle cx="471.9" cy="109.0" r="4" class="fill-a" opacity="0.89"/>
<circle cx="471.9" cy="109.0" r="4" class="fill-c" opacity="0.04"/>
<circle cx="471.9" cy="109.0" r="4" class="fill-d" opacity="0.07"/>
<circle cx="433.3" cy="86.7" r="4" class="fill-a" opacity="0.92"/>
<circle cx="433.3" cy="86.7" r="4" class="fill-d" opacity="0.07"/>
<circle cx="449.4" cy="69.1" r="4" class="fill-a" opacity="0.97"/>
<circle cx="449.4" cy="69.1" r="4" class="fill-d" opacity="0.03"/>
<circle cx="420.5" cy="114.3" r="4" class="fill-a" opacity="0.61"/>
<circle cx="420.5" cy="114.3" r="4" class="fill-c" opacity="0.05"/>
<circle cx="420.5" cy="114.3" r="4" class="fill-d" opacity="0.33"/>
<circle cx="444.9" cy="56.5" r="4" class="fill-a" opacity="0.98"/>
<circle cx="444.9" cy="56.5" r="4" class="fill-d" opacity="0.02"/>
<circle cx="459.9" cy="76.1" r="4" class="fill-a" opacity="0.97"/>
<circle cx="459.9" cy="76.1" r="4" class="fill-d" opacity="0.03"/>
<circle cx="438.8" cy="74.6" r="4" class="fill-a" opacity="0.96"/>
<circle cx="438.8" cy="74.6" r="4" class="fill-d" opacity="0.04"/>
<circle cx="440.2" cy="64.3" r="4" class="fill-a" opacity="0.97"/>
<circle cx="440.2" cy="64.3" r="4" class="fill-d" opacity="0.03"/>
<circle cx="429.4" cy="77.7" r="4" class="fill-a" opacity="0.94"/>
<circle cx="429.4" cy="77.7" r="4" class="fill-d" opacity="0.06"/>
<circle cx="442.8" cy="69.7" r="4" class="fill-a" opacity="0.97"/>
<circle cx="442.8" cy="69.7" r="4" class="fill-d" opacity="0.03"/>
<circle cx="449.4" cy="30.3" r="4" class="fill-a" opacity="0.99"/>
<circle cx="474.4" cy="51.0" r="4" class="fill-a" opacity="0.98"/>
<circle cx="474.4" cy="51.0" r="4" class="fill-d" opacity="0.01"/>
<circle cx="405.0" cy="87.0" r="4" class="fill-a" opacity="0.81"/>
<circle cx="405.0" cy="87.0" r="4" class="fill-d" opacity="0.19"/>
<circle cx="433.0" cy="69.9" r="4" class="fill-a" opacity="0.96"/>
<circle cx="433.0" cy="69.9" r="4" class="fill-d" opacity="0.04"/>
<circle cx="400.2" cy="57.3" r="4" class="fill-a" opacity="0.96"/>
<circle cx="400.2" cy="57.3" r="4" class="fill-d" opacity="0.04"/>
<circle cx="466.0" cy="105.9" r="4" class="fill-a" opacity="0.90"/>
<circle cx="466.0" cy="105.9" r="4" class="fill-c" opacity="0.03"/>
<circle cx="466.0" cy="105.9" r="4" class="fill-d" opacity="0.08"/>
<polyline points="424.8,100.7 422.8,96.0 421.2,91.3 419.8,86.5 418.8,81.8 418.1,77.1 417.8,72.5 417.9,68.2 418.3,64.0 419.1,60.1 420.2,56.5 421.7,53.3 423.5,50.5 425.6,48.1 427.9,46.1 430.5,44.6 433.4,43.6 436.4,43.0 439.6,43.0 442.9,43.4 446.3,44.4 449.8,45.8 453.3,47.7 456.7,50.1 460.1,52.9 463.4,56.0 466.5,59.6 469.5,63.4 472.3,67.5 474.8,71.9 477.1,76.4 479.1,81.0 480.8,85.8 482.1,90.6 483.2,95.3 483.8,100.0 484.1,104.5 484.1,108.9 483.6,113.1 482.9,117.0 481.7,120.5 480.3,123.7 478.5,126.6 476.4,129.0 474.0,131.0 471.4,132.5 468.6,133.5 465.5,134.1 462.3,134.1 459.0,133.6 455.6,132.7 452.1,131.2 448.7,129.3 445.2,127.0 441.8,124.2 438.6,121.0 435.4,117.5 432.4,113.7 429.7,109.6 427.1,105.2 424.8,100.7" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="446.0" y1="83.5" x2="456.0" y2="93.5" class="stroke-ink" stroke-width="2"/><line x1="446.0" y1="93.5" x2="456.0" y2="83.5" class="stroke-ink" stroke-width="2"/>
<polyline points="482.6,236.9 476.5,235.7 470.6,234.1 464.9,232.2 459.5,230.0 454.3,227.5 449.5,224.6 445.2,221.5 441.3,218.2 438.0,214.8 435.2,211.1 432.9,207.4 431.3,203.6 430.3,199.8 429.9,196.0 430.2,192.3 431.1,188.7 432.7,185.2 434.8,181.9 437.5,178.8 440.8,176.0 444.6,173.4 448.9,171.2 453.7,169.3 458.8,167.7 464.2,166.5 469.9,165.7 475.7,165.3 481.8,165.3 487.9,165.7 494.0,166.5 500.0,167.7 505.9,169.2 511.6,171.1 517.1,173.4 522.2,175.9 527.0,178.7 531.4,181.8 535.2,185.1 538.6,188.6 541.4,192.2 543.6,195.9 545.3,199.7 546.2,203.5 546.6,207.3 546.3,211.1 545.4,214.7 543.9,218.2 541.7,221.5 539.0,224.6 535.7,227.4 531.9,230.0 527.6,232.2 522.9,234.1 517.8,235.7 512.4,236.8 506.7,237.6 500.8,238.0 494.8,238.0 488.7,237.7 482.6,236.9" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="483.3" y1="196.7" x2="493.3" y2="206.7" class="stroke-ink" stroke-width="2"/><line x1="483.3" y1="206.7" x2="493.3" y2="196.7" class="stroke-ink" stroke-width="2"/>
<polyline points="413.3,212.9 409.2,216.6 404.9,220.0 400.4,223.0 395.7,225.6 390.9,227.7 386.1,229.4 381.2,230.6 376.4,231.2 371.7,231.4 367.1,231.1 362.8,230.3 358.7,228.9 354.9,227.1 351.4,224.9 348.4,222.1 345.7,219.0 343.5,215.6 341.7,211.7 340.4,207.6 339.6,203.3 339.3,198.7 339.5,194.0 340.3,189.2 341.5,184.4 343.2,179.6 345.4,174.8 348.0,170.1 351.1,165.6 354.5,161.3 358.2,157.3 362.3,153.6 366.6,150.2 371.1,147.2 375.8,144.6 380.6,142.5 385.5,140.8 390.3,139.7 395.1,139.0 399.8,138.8 404.4,139.1 408.7,140.0 412.8,141.3 416.6,143.1 420.1,145.4 423.2,148.1 425.8,151.2 428.1,154.7 429.8,158.5 431.1,162.6 431.9,166.9 432.2,171.5 432.0,176.2 431.2,181.0 430.0,185.8 428.3,190.7 426.1,195.4 423.5,200.1 420.5,204.6 417.1,208.9 413.3,212.9" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="380.8" y1="180.1" x2="390.8" y2="190.1" class="stroke-ink" stroke-width="2"/><line x1="380.8" y1="190.1" x2="390.8" y2="180.1" class="stroke-ink" stroke-width="2"/>
</g>
<rect x="20.0" y="295.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="20.0" y="287.0" class="fig-bold" text-anchor="start" >After 3 rounds</text>
<text x="290.0" y="287.0" class="fig-small" text-anchor="end" >log L = -222.7</text>
<g clip-path="url(#c056-p3)">
<circle cx="121.4" cy="508.3" r="4" class="fill-c" opacity="0.16"/>
<circle cx="121.4" cy="508.3" r="4" class="fill-d" opacity="0.84"/>
<circle cx="89.6" cy="469.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="72.5" cy="462.3" r="4" class="fill-d" opacity="1.00"/>
<circle cx="41.7" cy="462.6" r="4" class="fill-d" opacity="1.00"/>
<circle cx="64.4" cy="392.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="85.8" cy="465.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="75.8" cy="471.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="60.6" cy="465.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="90.8" cy="462.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="100.2" cy="462.0" r="4" class="fill-d" opacity="0.99"/>
<circle cx="81.8" cy="427.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="92.1" cy="468.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="77.8" cy="447.5" r="4" class="fill-d" opacity="1.00"/>
<circle cx="119.4" cy="463.4" r="4" class="fill-c" opacity="0.12"/>
<circle cx="119.4" cy="463.4" r="4" class="fill-d" opacity="0.88"/>
<circle cx="76.6" cy="438.4" r="4" class="fill-d" opacity="1.00"/>
<circle cx="64.0" cy="463.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="98.7" cy="446.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="83.2" cy="444.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="25.8" cy="438.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="62.5" cy="490.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="234.1" cy="444.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="219.9" cy="479.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="229.1" cy="459.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="256.2" cy="443.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="232.4" cy="436.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="224.6" cy="476.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="240.1" cy="446.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="224.4" cy="473.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="233.1" cy="507.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="242.2" cy="448.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="196.5" cy="456.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="209.7" cy="443.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="188.7" cy="476.0" r="4" class="fill-c" opacity="1.00"/>
<circle cx="242.6" cy="435.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="186.3" cy="467.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="235.1" cy="470.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="259.9" cy="481.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="235.6" cy="478.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="256.2" cy="458.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="221.3" cy="491.8" r="4" class="fill-c" opacity="1.00"/>
<circle cx="188.0" cy="320.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="169.8" cy="313.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="161.9" cy="347.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="139.3" cy="351.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="181.9" cy="364.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="143.3" cy="341.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="159.4" cy="324.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="130.5" cy="369.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="154.9" cy="311.5" r="4" class="fill-a" opacity="1.00"/>
<circle cx="169.9" cy="331.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="148.8" cy="329.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="150.2" cy="319.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="139.4" cy="332.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="152.8" cy="324.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="159.4" cy="285.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="184.4" cy="306.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="115.0" cy="342.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="143.0" cy="324.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="110.2" cy="312.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="176.0" cy="360.9" r="4" class="fill-a" opacity="1.00"/>
<polyline points="166.7,345.6 164.9,346.9 163.0,348.0 160.9,349.0 158.9,349.7 156.7,350.2 154.6,350.5 152.4,350.6 150.2,350.4 148.1,350.1 146.1,349.5 144.1,348.7 142.3,347.7 140.6,346.5 139.0,345.1 137.7,343.5 136.4,341.9 135.4,340.0 134.6,338.1 134.0,336.1 133.7,334.0 133.5,331.9 133.6,329.8 133.9,327.7 134.4,325.5 135.2,323.5 136.2,321.5 137.3,319.6 138.6,317.9 140.2,316.2 141.8,314.7 143.6,313.4 145.6,312.3 147.6,311.4 149.7,310.6 151.8,310.1 154.0,309.8 156.1,309.8 158.3,309.9 160.4,310.3 162.4,310.9 164.4,311.7 166.2,312.7 167.9,313.9 169.5,315.3 170.9,316.8 172.1,318.5 173.1,320.3 173.9,322.2 174.5,324.2 174.8,326.3 175.0,328.4 174.9,330.6 174.6,332.7 174.1,334.8 173.3,336.8 172.4,338.8 171.2,340.7 169.9,342.5 168.3,344.1 166.7,345.6" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="149.3" y1="325.2" x2="159.3" y2="335.2" class="stroke-ink" stroke-width="2"/><line x1="149.3" y1="335.2" x2="159.3" y2="325.2" class="stroke-ink" stroke-width="2"/>
<polyline points="221.4,483.2 217.4,483.5 213.4,483.6 209.5,483.4 205.8,483.1 202.1,482.5 198.7,481.8 195.5,480.8 192.5,479.7 189.9,478.4 187.5,476.9 185.6,475.3 184.0,473.6 182.7,471.8 181.9,469.9 181.5,467.9 181.5,465.9 182.0,463.9 182.8,461.9 184.1,459.9 185.7,458.0 187.7,456.1 190.0,454.3 192.7,452.6 195.7,451.1 198.9,449.6 202.4,448.4 206.0,447.3 209.8,446.4 213.7,445.7 217.7,445.2 221.7,445.0 225.6,444.9 229.5,445.0 233.3,445.4 236.9,445.9 240.4,446.7 243.6,447.6 246.5,448.8 249.2,450.1 251.5,451.5 253.5,453.1 255.1,454.8 256.3,456.6 257.1,458.5 257.5,460.5 257.5,462.5 257.1,464.5 256.2,466.6 255.0,468.5 253.4,470.5 251.4,472.4 249.0,474.2 246.3,475.8 243.4,477.4 240.1,478.8 236.7,480.0 233.0,481.1 229.2,482.0 225.3,482.7 221.4,483.2" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="214.5" y1="459.2" x2="224.5" y2="469.2" class="stroke-ink" stroke-width="2"/><line x1="214.5" y1="469.2" x2="224.5" y2="459.2" class="stroke-ink" stroke-width="2"/>
<polyline points="57.7,462.9 56.9,460.4 56.2,457.7 55.8,455.0 55.7,452.4 55.8,449.7 56.1,447.1 56.7,444.7 57.5,442.3 58.6,440.0 59.8,437.9 61.3,436.1 62.9,434.4 64.7,432.9 66.7,431.7 68.8,430.7 71.0,430.0 73.2,429.6 75.6,429.5 77.9,429.6 80.3,430.0 82.6,430.7 84.9,431.7 87.1,432.9 89.2,434.3 91.2,436.0 93.0,437.9 94.7,440.0 96.2,442.2 97.5,444.6 98.6,447.1 99.5,449.7 100.1,452.3 100.5,455.0 100.7,457.7 100.6,460.3 100.3,462.9 99.7,465.4 98.9,467.8 97.8,470.0 96.6,472.1 95.1,474.0 93.4,475.7 91.6,477.1 89.7,478.3 87.6,479.3 85.4,480.0 83.1,480.4 80.8,480.6 78.5,480.4 76.1,480.0 73.8,479.3 71.5,478.4 69.3,477.2 67.2,475.7 65.2,474.0 63.3,472.1 61.7,470.1 60.1,467.8 58.8,465.4 57.7,462.9" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="73.2" y1="450.0" x2="83.2" y2="460.0" class="stroke-ink" stroke-width="2"/><line x1="73.2" y1="460.0" x2="83.2" y2="450.0" class="stroke-ink" stroke-width="2"/>
</g>
<rect x="310.0" y="295.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="310.0" y="287.0" class="fig-bold" text-anchor="start" >After 10 rounds</text>
<text x="580.0" y="287.0" class="fig-small" text-anchor="end" >log L = -215.8</text>
<g clip-path="url(#c056-p10)">
<circle cx="411.4" cy="508.3" r="4" class="fill-d" opacity="1.00"/>
<circle cx="379.6" cy="469.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="362.5" cy="462.3" r="4" class="fill-d" opacity="1.00"/>
<circle cx="331.7" cy="462.6" r="4" class="fill-d" opacity="1.00"/>
<circle cx="354.4" cy="392.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="375.8" cy="465.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="365.8" cy="471.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="350.6" cy="465.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="380.8" cy="462.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="390.2" cy="462.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="371.8" cy="427.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="382.1" cy="468.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="367.8" cy="447.5" r="4" class="fill-d" opacity="1.00"/>
<circle cx="409.4" cy="463.4" r="4" class="fill-d" opacity="1.00"/>
<circle cx="366.6" cy="438.4" r="4" class="fill-d" opacity="1.00"/>
<circle cx="354.0" cy="463.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="388.7" cy="446.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="373.2" cy="444.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="315.8" cy="438.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="352.5" cy="490.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="524.1" cy="444.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="509.9" cy="479.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="519.1" cy="459.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="546.2" cy="443.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="522.4" cy="436.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="514.6" cy="476.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="530.1" cy="446.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="514.4" cy="473.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="523.1" cy="507.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="532.2" cy="448.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="486.5" cy="456.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="499.7" cy="443.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="478.7" cy="476.0" r="4" class="fill-c" opacity="1.00"/>
<circle cx="532.6" cy="435.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="476.3" cy="467.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="525.1" cy="470.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="549.9" cy="481.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="525.6" cy="478.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="546.2" cy="458.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="511.3" cy="491.8" r="4" class="fill-c" opacity="1.00"/>
<circle cx="478.0" cy="320.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="459.8" cy="313.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="451.9" cy="347.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="429.3" cy="351.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="471.9" cy="364.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="433.3" cy="341.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="449.4" cy="324.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="420.5" cy="369.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="444.9" cy="311.5" r="4" class="fill-a" opacity="1.00"/>
<circle cx="459.9" cy="331.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="438.8" cy="329.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="440.2" cy="319.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="429.4" cy="332.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="442.8" cy="324.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="449.4" cy="285.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="474.4" cy="306.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="405.0" cy="342.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="433.0" cy="324.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="400.2" cy="312.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="466.0" cy="360.9" r="4" class="fill-a" opacity="1.00"/>
<polyline points="457.2,345.2 455.5,346.6 453.6,347.9 451.6,349.0 449.6,349.9 447.4,350.5 445.3,351.0 443.1,351.2 440.9,351.2 438.8,351.0 436.7,350.6 434.7,349.9 432.8,349.0 431.0,348.0 429.4,346.7 427.9,345.3 426.6,343.7 425.5,341.9 424.6,340.0 423.9,338.1 423.4,336.0 423.1,333.9 423.1,331.8 423.3,329.6 423.7,327.4 424.4,325.3 425.2,323.3 426.3,321.3 427.6,319.4 429.0,317.7 430.6,316.1 432.3,314.6 434.2,313.3 436.2,312.2 438.3,311.4 440.4,310.7 442.5,310.2 444.7,310.0 446.9,310.0 449.0,310.2 451.1,310.7 453.1,311.3 455.0,312.2 456.8,313.3 458.4,314.5 459.9,316.0 461.2,317.6 462.3,319.3 463.2,321.2 463.9,323.1 464.4,325.2 464.7,327.3 464.7,329.5 464.5,331.6 464.1,333.8 463.4,335.9 462.6,337.9 461.5,339.9 460.3,341.8 458.8,343.5 457.2,345.2" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="438.9" y1="325.6" x2="448.9" y2="335.6" class="stroke-ink" stroke-width="2"/><line x1="438.9" y1="335.6" x2="448.9" y2="325.6" class="stroke-ink" stroke-width="2"/>
<polyline points="528.9,478.0 527.1,479.3 525.2,480.3 523.1,481.2 521.1,481.9 519.0,482.4 516.9,482.7 514.8,482.8 512.7,482.7 510.7,482.3 508.8,481.8 507.0,481.1 505.3,480.2 503.8,479.0 502.4,477.8 501.2,476.3 500.2,474.8 499.4,473.1 498.8,471.3 498.4,469.4 498.3,467.5 498.3,465.5 498.6,463.5 499.1,461.5 499.8,459.5 500.7,457.6 501.8,455.8 503.1,454.0 504.5,452.3 506.1,450.8 507.9,449.4 509.7,448.1 511.7,447.1 513.7,446.2 515.7,445.5 517.8,445.0 520.0,444.7 522.0,444.6 524.1,444.7 526.1,445.1 528.0,445.6 529.8,446.3 531.5,447.3 533.0,448.4 534.4,449.7 535.6,451.1 536.6,452.6 537.4,454.3 538.0,456.1 538.4,458.0 538.5,459.9 538.5,461.9 538.2,463.9 537.7,465.9 537.0,467.9 536.1,469.8 535.0,471.7 533.7,473.4 532.3,475.1 530.7,476.6 528.9,478.0" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="513.4" y1="458.7" x2="523.4" y2="468.7" class="stroke-ink" stroke-width="2"/><line x1="513.4" y1="468.7" x2="523.4" y2="458.7" class="stroke-ink" stroke-width="2"/>
<polyline points="355.7,470.2 353.9,468.1 352.2,465.9 350.8,463.6 349.6,461.2 348.5,458.8 347.7,456.4 347.2,454.0 346.9,451.7 346.8,449.4 346.9,447.2 347.4,445.1 348.0,443.1 348.9,441.3 350.0,439.7 351.3,438.3 352.8,437.1 354.5,436.1 356.4,435.3 358.4,434.8 360.5,434.6 362.7,434.6 365.0,434.8 367.3,435.3 369.7,436.1 372.0,437.0 374.3,438.2 376.6,439.6 378.8,441.3 380.9,443.0 382.8,445.0 384.6,447.1 386.2,449.3 387.7,451.6 388.9,453.9 389.9,456.3 390.7,458.7 391.3,461.1 391.6,463.5 391.7,465.8 391.5,468.0 391.1,470.1 390.4,472.0 389.6,473.8 388.5,475.4 387.1,476.9 385.6,478.1 383.9,479.1 382.1,479.8 380.1,480.3 377.9,480.6 375.7,480.6 373.5,480.3 371.1,479.8 368.8,479.1 366.4,478.1 364.1,476.9 361.9,475.5 359.7,473.9 357.6,472.1 355.7,470.2" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="364.2" y1="452.6" x2="374.2" y2="462.6" class="stroke-ink" stroke-width="2"/><line x1="364.2" y1="462.6" x2="374.2" y2="452.6" class="stroke-ink" stroke-width="2"/>
</g>
</svg>
Figure 1. EM from a deliberately plain start: three equal round blobs. Each round moves the blobs onto the clusters and shrinks them to fit, and log L climbs from -292.7 to -215.8. Ellipses are one-sigma contours, crosses are the blob means, and points are coloured by their responsibilities.
```

At the start the blobs are fat circles in the wrong places and $\log L=$ -292.7. After one round it is -259.9, after three -222.7, and after ten -215.8, when the ellipses sit on the clusters and stop moving. Each round the shares sharpen, which lets the blobs fit tighter, which sharpens the shares again.

## Why log L never falls

Look at Figure 1 and you may suspect the numbers always climb. They do, and here is why. The M-step improves a different score than $\log L$: a **bound** that sits underneath it.

Picture $\log L$ as a hill over the parameters. The E-step builds a second curve, the bound, that lies under the hill everywhere and touches it exactly at the current parameters. The M-step climbs to the top of the bound, which is at least as high as where it started. The hill is at least as high as the bound at that new point. So the hill is higher than before. Figure 2 shows this with the two-point numbers.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A curve for log L and a lower curve for the bound that touches it at the old parameter; the M-step moves to the top of the bound, where log L is higher still">
<defs></defs>
<line x1="190.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted" />
<line x1="190.0" y1="40.0" x2="190.0" y2="250.0" class="stroke-muted" />
<line x1="190.0" y1="236.9" x2="570.0" y2="236.9" class="stroke-grid" />
<line x1="186.0" y1="236.9" x2="190.0" y2="236.9" class="stroke-muted" />
<text x="183.0" y="240.9" class="fig-small" text-anchor="end" >-3.5</text>
<line x1="190.0" y1="171.2" x2="570.0" y2="171.2" class="stroke-grid" />
<line x1="186.0" y1="171.2" x2="190.0" y2="171.2" class="stroke-muted" />
<text x="183.0" y="175.2" class="fig-small" text-anchor="end" >-3.0</text>
<line x1="190.0" y1="105.6" x2="570.0" y2="105.6" class="stroke-grid" />
<line x1="186.0" y1="105.6" x2="190.0" y2="105.6" class="stroke-muted" />
<text x="183.0" y="109.6" class="fig-small" text-anchor="end" >-2.5</text>
<line x1="190.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid" />
<line x1="186.0" y1="40.0" x2="190.0" y2="40.0" class="stroke-muted" />
<text x="183.0" y="44.0" class="fig-small" text-anchor="end" >-2.0</text>
<text x="380.0" y="24.0" class="fig-bold" text-anchor="middle" >log L and the bound under it, over one parameter</text>
<text x="380.0" y="274.0" class="fig-small" text-anchor="middle" >the parameter (schematic shapes, true heights)</text>
<polyline points="190.0,229.0 192.1,226.7 194.2,224.4 196.3,222.1 198.4,219.9 200.6,217.7 202.7,215.4 204.8,213.3 206.9,211.1 209.0,208.9 211.1,206.8 213.2,204.6 215.3,202.5 217.4,200.4 219.6,198.4 221.7,196.3 223.8,194.3 225.9,192.2 228.0,190.2 230.1,188.2 232.2,186.2 234.3,184.3 236.4,182.3 238.6,180.4 240.7,178.5 242.8,176.6 244.9,174.7 247.0,172.8 249.1,171.0 251.2,169.2 253.3,167.4 255.4,165.6 257.6,163.8 259.7,162.0 261.8,160.3 263.9,158.5 266.0,156.8 268.1,155.1 270.2,153.4 272.3,151.8 274.4,150.1 276.6,148.5 278.7,146.9 280.8,145.3 282.9,143.7 285.0,142.1 287.1,140.6 289.2,139.0 291.3,137.5 293.4,136.0 295.6,134.5 297.7,133.1 299.8,131.6 301.9,130.2 304.0,128.8 306.1,127.4 308.2,126.0 310.3,124.6 312.4,123.2 314.6,121.9 316.7,120.6 318.8,119.3 320.9,118.0 323.0,116.7 325.1,115.5 327.2,114.2 329.3,113.0 331.4,111.8 333.6,110.6 335.7,109.5 337.8,108.3 339.9,107.2 342.0,106.0 344.1,104.9 346.2,103.8 348.3,102.8 350.4,101.7 352.6,100.7 354.7,99.7 356.8,98.7 358.9,97.7 361.0,96.7 363.1,95.7 365.2,94.8 367.3,93.9 369.4,93.0 371.6,92.1 373.7,91.2 375.8,90.3 377.9,89.5 380.0,88.7 382.1,87.9 384.2,87.1 386.3,86.3 388.4,85.5 390.6,84.8 392.7,84.1 394.8,83.4 396.9,82.7 399.0,82.0 401.1,81.3 403.2,80.7 405.3,80.1 407.4,79.4 409.6,78.8 411.7,78.3 413.8,77.7 415.9,77.2 418.0,76.6 420.1,76.1 422.2,75.6 424.3,75.1 426.4,74.7 428.6,74.2 430.7,73.8 432.8,73.4 434.9,73.0 437.0,72.6 439.1,72.3 441.2,71.9 443.3,71.6 445.4,71.3 447.6,71.0 449.7,70.7 451.8,70.4 453.9,70.2 456.0,69.9 458.1,69.7 460.2,69.5 462.3,69.3 464.4,69.2 466.6,69.0 468.7,68.9 470.8,68.8 472.9,68.7 475.0,68.6 477.1,68.5 479.2,68.5 481.3,68.4 483.4,68.4 485.6,68.4 487.7,68.4 489.8,68.5 491.9,68.5 494.0,68.6 496.1,68.7 498.2,68.8 500.3,68.9 502.4,69.0 504.6,69.2 506.7,69.3 508.8,69.5 510.9,69.7 513.0,69.9 515.1,70.1 517.2,70.4 519.3,70.7 521.4,70.9 523.6,71.2 525.7,71.5 527.8,71.9 529.9,72.2 532.0,72.6 534.1,73.0 536.2,73.4 538.3,73.8 540.4,74.2 542.6,74.6 544.7,75.1 546.8,75.6 548.9,76.1 551.0,76.6 553.1,77.1 555.2,77.6 557.3,78.2 559.4,78.8 561.6,79.4 563.7,80.0 565.8,80.6 567.9,81.3 570.0,81.9" fill="none" class="stroke-ink" stroke-width="2.5"/>
<polyline points="190.0,237.8 192.1,234.9 194.2,232.1 196.3,229.3 198.4,226.5 200.6,223.8 202.7,221.1 204.8,218.4 206.9,215.8 209.0,213.2 211.1,210.7 213.2,208.2 215.3,205.7 217.4,203.3 219.6,200.9 221.7,198.5 223.8,196.2 225.9,193.9 228.0,191.6 230.1,189.4 232.2,187.2 234.3,185.1 236.4,183.0 238.6,180.9 240.7,178.8 242.8,176.8 244.9,174.9 247.0,172.9 249.1,171.0 251.2,169.2 253.3,167.4 255.4,165.6 257.6,163.8 259.7,162.1 261.8,160.4 263.9,158.8 266.0,157.2 268.1,155.6 270.2,154.1 272.3,152.6 274.4,151.1 276.6,149.7 278.7,148.3 280.8,146.9 282.9,145.6 285.0,144.3 287.1,143.1 289.2,141.9 291.3,140.7 293.4,139.6 295.6,138.5 297.7,137.4 299.8,136.4 301.9,135.4 304.0,134.4 306.1,133.5 308.2,132.6 310.3,131.8 312.4,130.9 314.6,130.2 316.7,129.4 318.8,128.7 320.9,128.1 323.0,127.4 325.1,126.8 327.2,126.3 329.3,125.7 331.4,125.3 333.6,124.8 335.7,124.4 337.8,124.0 339.9,123.7 342.0,123.4 344.1,123.1 346.2,122.9 348.3,122.7 350.4,122.5 352.6,122.4 354.7,122.3 356.8,122.2 358.9,122.2 361.0,122.2 363.1,122.3 365.2,122.4 367.3,122.5 369.4,122.7 371.6,122.9 373.7,123.1 375.8,123.4 377.9,123.7 380.0,124.0 382.1,124.4 384.2,124.8 386.3,125.3 388.4,125.7 390.6,126.3 392.7,126.8 394.8,127.4 396.9,128.1 399.0,128.7 401.1,129.4 403.2,130.2 405.3,130.9 407.4,131.8 409.6,132.6 411.7,133.5 413.8,134.4 415.9,135.4 418.0,136.4 420.1,137.4 422.2,138.5 424.3,139.6 426.4,140.7 428.6,141.9 430.7,143.1 432.8,144.3 434.9,145.6 437.0,146.9 439.1,148.3 441.2,149.7 443.3,151.1 445.4,152.6 447.6,154.1 449.7,155.6 451.8,157.2 453.9,158.8 456.0,160.4 458.1,162.1 460.2,163.8 462.3,165.6 464.4,167.4 466.6,169.2 468.7,171.0 470.8,172.9 472.9,174.9 475.0,176.8 477.1,178.8 479.2,180.9 481.3,183.0 483.4,185.1 485.6,187.2 487.7,189.4 489.8,191.6 491.9,193.9 494.0,196.2 496.1,198.5 498.2,200.9 500.3,203.3 502.4,205.7 504.6,208.2 506.7,210.7 508.8,213.2 510.9,215.8 513.0,218.4 515.1,221.1 517.2,223.8 519.3,226.5 521.4,229.3 523.6,232.1 525.7,234.9 527.8,237.8 529.9,240.7 532.0,243.6 534.1,246.6" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="522.5" y="59.1" class="fig-bold" text-anchor="start" >log L</text>
<text x="525.9" y="221.1" class="fig-bold" text-anchor="start" >the bound</text>
<line x1="358.9" y1="122.2" x2="358.9" y2="97.7" class="stroke-muted" stroke-width="2"/>
<circle cx="253.3" cy="167.4" r="6" class="fill-ink" />
<circle cx="358.9" cy="122.2" r="6" class="fill-b" />
<circle cx="358.9" cy="97.7" r="6" class="fill-ink" />
<text x="348.9" y="89.7" class="fig-small" text-anchor="end" >new log L = -2.439</text>
<line x1="253.3" y1="175.0" x2="253.3" y2="222.0" class="stroke-muted" />
<text x="262.0" y="196.0" class="fig-small" text-anchor="start" >old parameter:</text>
<text x="262.0" y="212.0" class="fig-small" text-anchor="start" >log L = -2.970,</text>
<text x="262.0" y="228.0" class="fig-small" text-anchor="start" >the bound touches it</text>
<text x="372.0" y="190.0" class="fig-small" text-anchor="start" >bound after the</text>
<text x="372.0" y="206.0" class="fig-small" text-anchor="start" >M-step: -2.626</text>
<line x1="364.9" y1="130.2" x2="364.9" y2="210.0" class="stroke-muted" />
</svg>
Figure 2. The E-step builds the orange bound so that it touches log L at the old parameter. The M-step climbs to the top of the bound, from -2.970 to -2.626. Log L at that new parameter, -2.439, is at least as high as the bound, so log L cannot have fallen.
```

Now the formula, as a summary of the picture. For shares $r$,

$$
\text{bound}(r) = \sum_{i=1}^{n} \sum_{j=1}^{k} r_{ij} \left( \log\pi_j + \log \mathrm{N}(x_i; \mu_j, \Sigma_j) - \log r_{ij} \right) .
$$

Read it aloud. For each point and blob, take the log score from the E-step, subtract the log of the share, and weight by the share. The difference $\log L-\text{bound}$ is the **gap**. It is never negative, because the shares you plug in are a guess about the hidden choices, and no guess can beat the true posterior shares. It is exactly 0 when the shares are the posterior shares the E-step computes. The standard names are ELBO for the bound and KL divergence for the gap. That is why the E-step makes the bound touch.

The chain is then:

$$
\log L(\text{new}) \ \ge\ \text{bound}(\text{new}) \ \ge\ \text{bound}(\text{old}) \ =\ \log L(\text{old}).
$$

The first step holds because the gap is not negative. The second holds because the weighted mean and covariance are the values that make the bound as large as possible for fixed shares (with $\epsilon=0$, and a tiny $\epsilon$ only changes the last digits). The last holds because the E-step made the gap 0.

Check the numbers. At the E-step each point gives $-1.308-0.177=-1.485$, so the bound is $-2.970$, equal to $\log L$. After the M-step, with the same shares, the bound is $-2.626$. The true $\log L$ at the new blobs is $-2.439$, above the bound by a gap of $0.187$. The next E-step resets the gap to 0 at $-2.439$.

In numpy, build the table of log scores and weight it:

```python
log_r = np.log(r, out=np.zeros_like(r), where=r > 0)
bound = np.sum(r * (scores - log_r))
```

> [!WARNING]
> A share can be exactly 0 when a blob is far away, and $0\cdot\log 0$ is `nan` in numpy. Take the log of $r$ only where $r>0$, as in the code above, since such a term contributes 0.

> [!KEY]
> One E-step then one M-step never lowers $\log L$. A drop in the history means a bug, most often a covariance measured from the wrong centre.

## Tiny blobs give k-means

Give every blob equal weight and the covariance $\sigma^2 I$. The share of a blob then depends on the gap in squared distances divided by $2\sigma^2$. Take a point at $x=1$ with blobs at $0$ and $4$. The squared distances are $1$ and $9$, so the score gap is $(9-1)/(2\sigma^2)=4/\sigma^2$, and the far blob's share is $1/(1+e^{4/\sigma^2})$.

| $\sigma^2$ | Score gap | Far blob's share |
|---|---|---|
| 1 | 4 | 0.018 |
| 0.1 | 40 | 4.2e-18 |
| 0.01 | 400 | 1.9e-174 |
| $10^{-6}$ | 4,000,000 | 0 (underflows) |

As $\sigma^2$ shrinks, any gap in distance becomes an enormous gap in score, so the nearest blob takes all the weight. The right panel of Figure 3 shows the shares turning into a hard assignment. With hard shares the M-step's mean is a plain group mean, which is the k-means update. On the 60 points, take the Lloyd centres, give them covariances $10^{-6}I$, and run the E-step: the nearest-centre assignment is reproduced for 60 of 60 points, and the M-step returns the Lloyd centres.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Left, log L rising every round. Middle, the k-means objective J falling every pass from the same starting centres. Right, the nearer blob's share at x = 1 rising to 1 as the variance shrinks from 1 to 0.000001">
<defs></defs>
<line x1="45.0" y1="200.0" x2="195.0" y2="200.0" class="stroke-muted" />
<line x1="45.0" y1="40.0" x2="45.0" y2="200.0" class="stroke-muted" />
<line x1="250.0" y1="200.0" x2="380.0" y2="200.0" class="stroke-muted" />
<line x1="250.0" y1="40.0" x2="250.0" y2="200.0" class="stroke-muted" />
<line x1="49.2" y1="200.0" x2="49.2" y2="204.0" class="stroke-muted" />
<text x="49.2" y="217.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="120.0" y1="200.0" x2="120.0" y2="204.0" class="stroke-muted" />
<text x="120.0" y="217.0" class="fig-small" text-anchor="middle" >5</text>
<line x1="190.8" y1="200.0" x2="190.8" y2="204.0" class="stroke-muted" />
<text x="190.8" y="217.0" class="fig-small" text-anchor="middle" >10</text>
<line x1="45.0" y1="200.0" x2="195.0" y2="200.0" class="stroke-grid" />
<line x1="41.0" y1="200.0" x2="45.0" y2="200.0" class="stroke-muted" />
<text x="38.0" y="204.0" class="fig-small" text-anchor="end" >-300</text>
<line x1="45.0" y1="146.7" x2="195.0" y2="146.7" class="stroke-grid" />
<line x1="41.0" y1="146.7" x2="45.0" y2="146.7" class="stroke-muted" />
<text x="38.0" y="150.7" class="fig-small" text-anchor="end" >-270</text>
<line x1="45.0" y1="93.3" x2="195.0" y2="93.3" class="stroke-grid" />
<line x1="41.0" y1="93.3" x2="45.0" y2="93.3" class="stroke-muted" />
<text x="38.0" y="97.3" class="fig-small" text-anchor="end" >-240</text>
<line x1="45.0" y1="40.0" x2="195.0" y2="40.0" class="stroke-grid" />
<line x1="41.0" y1="40.0" x2="45.0" y2="40.0" class="stroke-muted" />
<text x="38.0" y="44.0" class="fig-small" text-anchor="end" >-210</text>
<line x1="260.8" y1="200.0" x2="260.8" y2="204.0" class="stroke-muted" />
<text x="260.8" y="217.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="296.9" y1="200.0" x2="296.9" y2="204.0" class="stroke-muted" />
<text x="296.9" y="217.0" class="fig-small" text-anchor="middle" >1</text>
<line x1="333.1" y1="200.0" x2="333.1" y2="204.0" class="stroke-muted" />
<text x="333.1" y="217.0" class="fig-small" text-anchor="middle" >2</text>
<line x1="369.2" y1="200.0" x2="369.2" y2="204.0" class="stroke-muted" />
<text x="369.2" y="217.0" class="fig-small" text-anchor="middle" >3</text>
<line x1="250.0" y1="200.0" x2="380.0" y2="200.0" class="stroke-grid" />
<line x1="246.0" y1="200.0" x2="250.0" y2="200.0" class="stroke-muted" />
<text x="243.0" y="204.0" class="fig-small" text-anchor="end" >0</text>
<line x1="250.0" y1="50.4" x2="380.0" y2="50.4" class="stroke-grid" />
<line x1="246.0" y1="50.4" x2="250.0" y2="50.4" class="stroke-muted" />
<text x="243.0" y="54.4" class="fig-small" text-anchor="end" >200</text>
<text x="120.0" y="24.0" class="fig-bold" text-anchor="middle" >EM: log L by round</text>
<text x="315.0" y="24.0" class="fig-bold" text-anchor="middle" >Lloyd: J by pass</text>
<text x="120.0" y="234.0" class="fig-small" text-anchor="middle" >round</text>
<text x="315.0" y="234.0" class="fig-small" text-anchor="middle" >pass</text>
<polyline points="49.2,187.0 63.4,128.8 77.5,87.4 91.7,62.5 105.8,51.5 120.0,50.3 134.2,50.3 148.3,50.3 162.5,50.3 176.6,50.3 190.8,50.3" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="49.2" cy="187.0" r="3.5" class="fill-b" />
<circle cx="63.4" cy="128.8" r="3.5" class="fill-b" />
<circle cx="77.5" cy="87.4" r="3.5" class="fill-b" />
<circle cx="91.7" cy="62.5" r="3.5" class="fill-b" />
<circle cx="105.8" cy="51.5" r="3.5" class="fill-b" />
<circle cx="120.0" cy="50.3" r="3.5" class="fill-b" />
<circle cx="134.2" cy="50.3" r="3.5" class="fill-b" />
<circle cx="148.3" cy="50.3" r="3.5" class="fill-b" />
<circle cx="162.5" cy="50.3" r="3.5" class="fill-b" />
<circle cx="176.6" cy="50.3" r="3.5" class="fill-b" />
<circle cx="190.8" cy="50.3" r="3.5" class="fill-b" />
<polyline points="260.8,51.9 296.9,132.8 333.1,133.8 369.2,133.8" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="260.8" cy="51.9" r="3.5" class="fill-b" />
<circle cx="296.9" cy="132.8" r="3.5" class="fill-b" />
<circle cx="333.1" cy="133.8" r="3.5" class="fill-b" />
<circle cx="369.2" cy="133.8" r="3.5" class="fill-b" />
<text x="489.0" y="24.0" class="fig-bold" text-anchor="middle" >Nearer blob's share at x = 1</text>
<rect x="430.0" y="66.0" width="147.3" height="20.0" class="fill-b" />
<rect x="577.3" y="66.0" width="2.7" height="20.0" class="fill-c" />
<text x="424.0" y="81.0" class="fig-small" text-anchor="end" >1</text>
<text x="436.0" y="81.0" class="fig-small fill-paper" text-anchor="start" >0.982 / 0.018</text>
<rect x="430.0" y="102.0" width="150.0" height="20.0" class="fill-b" />
<rect x="580.0" y="102.0" width="0.0" height="20.0" class="fill-c" />
<text x="424.0" y="117.0" class="fig-small" text-anchor="end" >0.1</text>
<text x="436.0" y="117.0" class="fig-small fill-paper" text-anchor="start" >1 / 4e-18</text>
<rect x="430.0" y="138.0" width="150.0" height="20.0" class="fill-b" />
<rect x="580.0" y="138.0" width="0.0" height="20.0" class="fill-c" />
<text x="424.0" y="153.0" class="fig-small" text-anchor="end" >0.01</text>
<text x="436.0" y="153.0" class="fig-small fill-paper" text-anchor="start" >1 / 2e-174</text>
<rect x="430.0" y="174.0" width="150.0" height="20.0" class="fill-b" />
<rect x="580.0" y="174.0" width="0.0" height="20.0" class="fill-c" />
<text x="424.0" y="189.0" class="fig-small" text-anchor="end" >1e-06</text>
<text x="436.0" y="189.0" class="fig-small fill-paper" text-anchor="start" >1 / 0</text>
<text x="424.0" y="50.0" class="fig-small" text-anchor="end" >variance</text>
<text x="505.0" y="220.0" class="fig-small" text-anchor="middle" >orange: blob at 0, green: blob at 4</text>
</svg>
Figure 3. Both loops climb or fall every round and never reverse. On the right, shrinking the blobs turns the soft split into a hard one: at variance 1 the far blob keeps a sliver, and from variance 0.1 down the bar is all orange, which is a k-means assignment.
```

Figure 3 puts the two loops side by side. Both start from the same three centres. EM's $\log L$ climbs from -292.7 to -215.8, and Lloyd's $J$ falls from 198 to 88.5. They are the same idea at two settings: guess the hidden assignment, then refit.

> [!RECAP]
> Before you build, keep these points in hand:
> - The M-step is the ordinary weight, mean, and covariance with each point counted by its share, using the new mean in the covariance.
> - The bound sums share times (log score minus log share). It touches $\log L$ when the shares are the posterior.
> - $\log L \ge$ bound, and the M-step raises the bound, so $\log L$ never falls.
> - Start EM from k-means++ means, equal round covariances, and equal weights, and run the E-step first.
> - Equal blobs with $\sigma^2\to 0$ give hard shares, which is k-means.

## Build it step by step

You write three functions. The seeding, the k-means functions, and the E-step from the earlier chapters are already in the file.

1. **`gmm_m_step(X, responsibilities, reg_covar)`** returns `means` `(k, d)`, `covs` `(k, d, d)`, and `weights` `(k,)`. Compute `counts = responsibilities.sum(axis=0)`, then the weights, the means, and for each blob the weighted outer-product sum around its new mean, divided by its count, plus `reg_covar * np.eye(d)`. The usual mistake is dividing by $n$ instead of $N_j$.

    ```python
    def gmm_m_step(X, responsibilities, reg_covar):
        # counts N_j; weights N_j / n; weighted means; covariance around the NEW mean
    ```

2. **`gmm_elbo(X, responsibilities, means, covs, weights)`** returns the bound as a Python float. Build the `(n, k)` table `log(weights[j]) + log_gaussian(...)`, subtract $\log r$ taken only where $r>0$, multiply by $r$, and sum. The usual mistake is `np.log(r)` on a zero share.

    ```python
    def gmm_elbo(X, responsibilities, means, covs, weights):
        # sum of r * (log weight + log density - log r)
    ```

3. **`fit_gmm(X, k, iters, rng, reg_covar)`** returns `(means, covs, weights, responsibilities, history)`. Seed the means with `kmeans_plusplus_init`, give every covariance `np.var(X, axis=0).mean() * np.eye(d)`, and use weights of `1 / k`. Run the E-step and record $\log L$, then repeat `iters` times: M-step, E-step, record. The history has `iters + 1` entries. The usual mistake is running the M-step before any E-step, when the shares do not exist yet.

    ```python
    def fit_gmm(X, k, iters, rng, reg_covar):
        # E-step first; then M-step, E-step, record, repeated
    ```

## Cost and scaling

Each round costs about the same as one E-step plus one M-step: roughly $n k d^2$ operations for the shares and the covariances, and $k d^3$ for the inverses. The memory holds the `(n, k)` shares and $k$ covariances of $d^2$ numbers each. Diagonal covariances cut the covariance memory to $d$ numbers per blob.

## Where this breaks

- **A blob can collapse onto one point.** In one dimension, a blob centred on a data point with variance $\sigma^2$ has height $1/\sqrt{2\pi\sigma^2}$ at that point, which is about 399 at $\sigma^2=10^{-6}$ and about 398,942 at $10^{-12}$. The log-likelihood goes to infinity with a useless blob. The $\epsilon I$ in the M-step, `reg_covar`, stops it.
- **EM can start slowly or stop at a bump.** Like Lloyd, it only climbs, so a poor start can take many rounds or end at a worse fit. Seed with k-means++ and keep the run with the best $\log L$.
- **You choose the number of blobs.** More blobs always raise the training $\log L$. Score a held-out set instead, and keep the $k$ where it stops improving.
- **Scales matter.** A column in grams next to one in kilometres dominates every distance. Standardise the columns first.

## The exercise

You build `gmm_m_step`, `gmm_elbo`, and `fit_gmm`. The checks reproduce the two-point refit, with means $0.2384$ and $1.7616$ and variance $0.42$. They compare the weighted mean and covariance with numpy's weighted statistics, confirm the bound equals $\log L$ at the posterior shares, sits below it for other shares, and stays finite when a share is 0. They also confirm that `fit_gmm` returns a history of length `iters + 1` that never falls, and recovers the three planted blobs.
