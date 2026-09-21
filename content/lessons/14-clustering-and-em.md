---
title: Clustering and expectation maximisation
slug: 14-clustering-and-em
part: "Part 2 — Classical machine learning"
summary: k-means as alternating minimisation, k-means++'s guarantee, and EM derived from the likelihood bound it tightens and loosens.
minutes: 50
gpu: false
objectives:
  - State why Lloyd's algorithm never increases the k-means objective and always terminates.
  - State the k-means++ approximation guarantee and explain why the seeding distribution earns it.
  - State the ELBO decomposition and use it to explain why EM never decreases the incomplete-data likelihood.
  - Recover k-means as the zero-variance limit of EM for an isotropic Gaussian mixture.
prereqs:
  - 13-pca-and-svd
exercise: 14-clustering-and-em
---

# Clustering and expectation maximisation

Every method so far has assumed labels. Clustering asks what the data
organises itself into when there are none: a partition into $k$ groups,
chosen to make each group look like a single object. That objective is
NP-hard to optimise exactly, so every practical algorithm for it is a local
search with a story about why the local optimum it finds is good enough.

k-means is the sharpest version of that story: its local search is provably
monotone, and its seeding rule comes with a provable guarantee. It is also
a special case of something more general: fitting a Gaussian mixture by
maximum likelihood. There, group membership is a latent variable, and
expectation maximisation maximises a likelihood you cannot see all of.
This lesson derives both, and shows they are the same algorithm at two
different values of a variance you can send to zero.

## What you already know

- The multivariate Gaussian density, and that its log is a negative
  quadratic form plus a normalising constant depending on the
  log-determinant of the covariance.
- Maximum likelihood estimation, and that the sample mean and the sample
  covariance are the maximum likelihood estimates of a single Gaussian's
  parameters.
- Jensen's inequality: for a concave function $\phi$ and a random variable
  $Z$, $\phi(\mathbb{E}[Z]) \geq \mathbb{E}[\phi(Z)]$.
- The Kullback-Leibler divergence $\mathrm{KL}(q \Vert p) = \mathbb{E}_q[\log
  (q/p)]$ is nonnegative, and zero exactly when $q = p$ almost everywhere.
- A nonconvex optimisation problem can have many local minima with no
  ordering between them.

## The k-means objective

Given points $x_1, \dots, x_n \in \mathbb{R}^d$ and a number of clusters
$k$, k-means chooses centers $\mu_1, \dots, \mu_k \in \mathbb{R}^d$ and an
assignment $z_i \in \{1, \dots, k\}$ for each point to minimise

$$
J(\{\mu_j\}, \{z_i\}) = \sum_{i=1}^{n} \lVert x_i - \mu_{z_i} \rVert_2^2 .
$$

This is a joint optimisation over a continuous variable, the centers, and a
combinatorial one, the assignment. The combinatorial part is what makes it
hard: finding the partition that minimises $J$ is NP-hard, even for $k=2$
once the dimension is part of the input. No algorithm is known that solves
it exactly in polynomial time.

Every practical algorithm settles for a local minimum: an assignment and a
set of centers that no move it is willing to make can improve. Lloyd's
algorithm is the standard one.

## Lloyd's algorithm as alternating minimisation

Lloyd's algorithm fixes one of the two variables in $J$ and minimises over
the other, alternately.

1. **Assignment step.** With centers $\{\mu_j\}$ fixed, assign each point to
   its nearest center: $z_i \leftarrow \operatorname{arg\,min}_j \lVert x_i -
   \mu_j \rVert^2$.
2. **Update step.** With the assignment fixed, move each center to the mean
   of its assigned points: $\mu_j \leftarrow \frac{1}{|\{i : z_i = j\}|}
   \sum_{i : z_i = j} x_i$.

Both steps are exact minimisations of $J$ over the variable they touch, so
neither can increase $J$. The assignment step sets each point's label to
minimise that point's own term in the sum. The update step sets each
center to the unique minimiser of a strictly convex quadratic, the mean.
The objective sequence is therefore nonincreasing and bounded below by
zero. It is also a sequence of values of $J$ at one of only finitely many
possible partitions of $n$ points into $k$ groups, so it must eventually
repeat a value. At that point, the algorithm has reached a fixed point and
terminates. Breaking ties in the assignment step by a fixed
rule rules out oscillating between two equally good assignments. Nothing
about this argument says the point reached is the global minimum, and in
practice it usually is not.

## k-means++ seeding

Local search is only as good as where it starts. Uniform random seeding, or
choosing $k$ data points as the initial centers, can lock Lloyd's algorithm
into a bad partition before it runs a single iteration. Two centers can
land in the same true cluster, leaving another with none, and alternating
minimisation only ever refines that mistake.

k-means++ fixes the seeding distribution instead of the algorithm. It
chooses the first center uniformly at random from the data, then chooses
each subsequent center from the remaining points with probability
proportional to its squared distance to the nearest center already chosen:

$$
\Pr[\text{choose } x_i \text{ next}] = \frac{D(x_i)^2}{\sum_{i'} D(x_{i'})^2},
\qquad D(x_i) = \min_{j \text{ chosen}} \lVert x_i - \mu_j \rVert .
$$

Points far from every existing center are likelier to be chosen, which
spreads the initial centers across the clusters. This change comes with a
guarantee uniform seeding has none of: if $J^\star$ is the global minimum of
the k-means objective, the expected objective after k-means++ seeding alone,
before any Lloyd iteration, satisfies

$$
\mathbb{E}[J] \leq 8(\ln k + 2)\, J^\star .
$$

The bound grows only logarithmically in $k$; uniform seeding has no such
bound, and an adversarial cluster configuration can make it arbitrarily bad
relative to $J^\star$. Running Lloyd's algorithm to convergence afterward
only decreases $J$ further, so the bound carries over to the local minimum
the full algorithm reaches. The result is due to Arthur and Vassilvitskii;
its proof is a counting argument this lesson does not reproduce.

## Gaussian mixtures and the likelihood you cannot see

k-means commits to a hard assignment: every point belongs to exactly one
cluster. A Gaussian mixture model softens that commitment into a
probability distribution and asks for the maximum likelihood parameters
instead of the minimum-distortion partition.

Introduce a latent variable $z_i \in \{1, \dots, k\}$ for each point, with
$\Pr[z_i = j] = \pi_j$, $\sum_j \pi_j = 1$, and $x_i \mid z_i = j \sim
\mathcal{N}(\mu_j, \Sigma_j)$. Marginalising $z_i$ out gives the mixture
density $p(x_i \mid \theta) = \sum_{j=1}^{k} \pi_j\, \mathcal{N}(x_i;
\mu_j, \Sigma_j)$, with $\theta = \{\pi_j, \mu_j, \Sigma_j\}_{j=1}^{k}$. The
**incomplete-data log-likelihood**, incomplete because $z$ is never
observed, is

$$
\ell(\theta) = \sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_j\, \mathcal{N}(x_i;
\mu_j, \Sigma_j) .
$$

The sum sits inside the log, which is exactly the structure that makes
maximum likelihood easy for a single Gaussian and hard for a mixture.
There is no closed form for the $\theta$ that maximises $\ell$, and $\ell$
is not concave in $\theta$. Had $z$ been observed, the **complete-data
log-likelihood** $\log p(x, z \mid \theta) = \sum_i (\log \pi_{z_i} + \log
\mathcal{N}(x_i; \mu_{z_i}, \Sigma_{z_i}))$ would separate over components
and reduce to $k$ independent weighted Gaussian fits, each with a closed
form. EM gets the benefit of that closed form without observing $z$.

## The EM derivation

Fix a distribution $q_i(z_i)$ over each point's latent label, write $q$ for
the collection, and define the **evidence lower bound**, or ELBO,

$$
\mathcal{L}(q, \theta) = \sum_{i=1}^{n} \sum_{z_i} q_i(z_i) \log
\frac{p(x_i, z_i \mid \theta)}{q_i(z_i)} .
$$

An algebraic identity connects it to the log-likelihood exactly:
$\log p(x \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}\big(q(z)
\,\Vert\, p(z \mid x, \theta)\big)$, for **every** $q$, not only a
well-chosen one. Since $\mathrm{KL} \geq 0$ always, $\mathcal{L}(q, \theta)
\leq \log p(x \mid \theta)$: the ELBO lower-bounds the log-likelihood, with
a gap equal to a KL divergence.

**The E-step closes the gap.** Fixing $\theta$, maximising
$\mathcal{L}(q,\theta)$ over $q$ minimises that KL divergence, because
$\log p(x\mid\theta)$ does not depend on $q$, and the divergence is zero
exactly when $q_i(z_i)$ equals the posterior $p(z_i \mid x_i, \theta)$. The
optimal $q$ is the **responsibility**

$$
q_i^\star(j) = \gamma_{ij} = p(z_i = j \mid x_i, \theta) = \frac{\pi_j\,
\mathcal{N}(x_i; \mu_j, \Sigma_j)}{\sum_{j'} \pi_{j'}\, \mathcal{N}(x_i;
\mu_{j'}, \Sigma_{j'})} ,
$$

and at $q^\star$ the bound is tight: $\mathcal{L}(q^\star, \theta) = \log
p(x \mid \theta)$ exactly.

**The M-step cannot decrease the likelihood.** With $q = q^\star$ fixed at
$\theta^{\text{old}}$, the M-step maximises $\mathcal{L}(q^\star, \theta)$
to get $\theta^{\text{new}}$, giving the chain

$$
\log p(x \mid \theta^{\text{old}}) \;=\; \mathcal{L}(q^\star,
\theta^{\text{old}}) \;\leq\; \mathcal{L}(q^\star, \theta^{\text{new}})
\;\leq\; \log p(x \mid \theta^{\text{new}}) .
$$

The first equality is the E-step closing the gap, the middle inequality is
the M-step's maximisation, and the last is the general ELBO bound applied
where the gap is no longer zero. The conclusion holds regardless of how
tight the middle inequality is: EM never decreases the incomplete-data
likelihood, for any model where both steps are exact. Nothing here depends
on the mixture being Gaussian; only the M-step's closed form does.

## The M-step in closed form

Maximising $\mathcal{L}(q^\star, \theta) = \sum_i \sum_j \gamma_{ij} \big(
\log \pi_j + \log \mathcal{N}(x_i; \mu_j, \Sigma_j) \big)$ over $\theta$,
subject to $\sum_j \pi_j = 1$, separates into three independent problems,
each solved by setting a gradient (or a Lagrangian's gradient, for the
weights) to zero:

$$
\pi_j = \frac{N_j}{n}, \qquad
\mu_j = \frac{1}{N_j} \sum_{i=1}^{n} \gamma_{ij}\, x_i, \qquad
\Sigma_j = \frac{1}{N_j} \sum_{i=1}^{n} \gamma_{ij} (x_i - \mu_j)(x_i -
\mu_j)^\top ,
$$

where $N_j = \sum_i \gamma_{ij}$. Every update is a responsibility-weighted
version of the ordinary Gaussian maximum likelihood estimate: the same mean
and covariance formulas, weighted by how much of each point's probability
mass the E-step assigned to that component. This is the closed form the
complete-data likelihood would give if $z$ had been observed, with the
indicator $\mathbb{1}[z_i = j]$ replaced by its expectation $\gamma_{ij}$.

## k-means as the zero-variance limit of EM

Restrict the mixture to a shared isotropic covariance, $\Sigma_j = \sigma^2
I$ for every $j$, and inspect the responsibility as $\sigma \to 0$.
Comparing components $j$ and $j'$ for a fixed point $x_i$, the log-ratio of
their responsibilities is a constant (from the mixing weights) minus a term
proportional to $\big(\lVert x_i - \mu_j \rVert^2 - \lVert x_i - \mu_{j'}
\rVert^2\big) / \sigma^2$. If component $j$ is strictly closer to $x_i$,
that term diverges to $+\infty$ as $\sigma \to 0$. As a result, $\gamma_{ij}
\to 1$ for the nearest component and $\gamma_{ij'} \to 0$ for every other.
The E-step hardens into exactly the k-means assignment rule,
$z_i = \operatorname{arg\,min}_j \lVert x_i - \mu_j \rVert^2$.

With hardened responsibilities, $N_j \to |\{i : z_i = j\}|$ and the M-step
mean update converges to $\frac{1}{|\{i:z_i=j\}|} \sum_{i:z_i=j} x_i$ —
exactly Lloyd's update step. The covariance update degenerates to zero
within each hardened cluster, consistent with having sent $\sigma \to 0$
rather than something the data determines. k-means is not an approximation
to this limit. It is the limit, with the variance held at zero throughout
rather than annealed there.

## Cost and scaling

Let $n$ be the number of points, $d$ the dimension, and $k$ the number of
clusters or components.

| Step | Flops per iteration | Notes |
|---|---|---|
| k-means assignment | $O(nkd)$ | A distance to every center for every point. |
| k-means update | $O(nd)$ | One pass to accumulate sums per cluster. |
| k-means++ seeding | $O(nkd)$ | One distance-update pass per center chosen. |
| GMM E-step | $O(nkd^2)$ | The Mahalanobis form costs $O(d^2)$ per point per component with $\Sigma_j^{-1}$ precomputed, plus $O(kd^3)$ once per iteration to invert every $\Sigma_j$. |
| GMM M-step | $O(nkd^2)$ | The covariance update is an outer product accumulated for every point and component. |

k-means is cheaper than EM for a full-covariance mixture by a factor of
$d$, because it never forms a $d \times d$ matrix. That gap is why k-means
remains the default first pass even when the eventual model is a mixture:
its output, or the zero-variance limit's assignment directly, seeds
$\{\mu_j\}$ before turning the covariance back on. Both algorithms are
linear in $n$ per iteration, so the number of iterations to convergence,
not the per-iteration cost, dominates at scale.

## Where this breaks

**The likelihood surface is unbounded above.** Let one component's mean
coincide with a single data point, and let its covariance shrink toward
zero along any path that keeps it positive definite. The density diverges,
because the normalising constant carries a factor of
$\det(\Sigma_j)^{-1/2}$. As a result, the incomplete-data likelihood has no
finite maximiser. EM rarely walks into this from a reasonable start, but it
happens whenever a component's effective sample size drifts toward one
during iteration. The standard defenses are a floor on the eigenvalues of
$\Sigma_j$ and a check that discards any run where a component's weight
falls below a minimum count. Another option is a Bayesian prior over
$\Sigma_j$ that removes the incentive to collapse onto a point.

**Non-convex clusters.** Both algorithms fit each component with a single
mean, assuming clusters are convex and, for k-means, roughly spherical.
Two clusters shaped like nested rings are not separable by
nearest-centroid assignment no matter how it is initialised. Spectral
clustering, which runs k-means on the eigenvectors of a graph Laplacian
rather than on the raw coordinates, is the standard response.

**Choosing $k$.** Neither algorithm says how many clusters there are. $J$
and $\ell$ both improve, weakly, as $k$ grows, per the unbounded-likelihood
point above. Practitioners use a penalised criterion such as the Bayesian
information criterion, the elbow of $J$ against $k$, or a nonparametric
alternative such as a Dirichlet process mixture that treats $k$ as a random
variable.

**Sensitivity to scale.** k-means minimises Euclidean distance, so a
feature measured in different units dominates the objective for reasons
unrelated to cluster structure. Standardise features first, as in linear
regression, or use a Mahalanobis distance, which for a shared covariance is
what a GMM with tied $\Sigma_j = \Sigma$ effectively does.

**Local minima in high dimension.** The number of distinct local minima of
$J$ grows quickly with $d$ and $k$. Lloyd's algorithm gives no guarantee
about which one it lands in beyond k-means++'s bound on the starting
point. Running the pipeline from several independent k-means++
seeds and keeping the lowest-$J$ run is standard practice, not a
workaround.

## The exercise

You implement `kmeans_objective`, `lloyd(X, k, init, iters)`, and
`kmeans_plusplus_init(X, k, rng)`, then `gmm_e_step`, `gmm_m_step`, and
`fit_gmm` for a Gaussian mixture with full covariance per component.
`lloyd` returns the history of the objective at every iteration, and
`fit_gmm` returns the history of the log-likelihood. Both monotone
properties derived above are therefore things the checks can see directly.

The checks verify the objective on a hand-computed example, and that
Lloyd's objective never increases across iterations on a synthetic
dataset. They verify that k-means++ seeding beats uniform seeding on
average over several seeds on a planted five-cluster problem, and that the
E-step's responsibilities sum to one per point and its log-likelihood
matches a direct log-sum-exp computation. They also verify that the M-step
recovers the ordinary sample mean and covariance when the responsibilities
are hardened onto a single component, and that EM's log-likelihood never
decreases and improves substantially over its iterations. Finally, they
verify that hardening the E-step with a shared, near-zero isotropic
covariance reproduces Lloyd's own assignment exactly.
