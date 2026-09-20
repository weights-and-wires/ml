---
title: Clustering and expectation maximisation
slug: 14-clustering-and-em
part: "Part 2 — Classical machine learning"
summary: k-means as alternating minimisation with a proof of its descent, k-means++'s guarantee, and EM derived from the ELBO gap it closes and reopens.
minutes: 60
gpu: false
objectives:
  - Prove that Lloyd's algorithm never increases the k-means objective and terminates at a local minimum.
  - State the k-means++ approximation guarantee and explain why the seeding distribution earns it.
  - Derive the ELBO decomposition and use it to show that EM never decreases the incomplete-data likelihood.
  - Recover k-means as the zero-variance limit of EM for an isotropic Gaussian mixture.
prereqs:
  - 13-pca-and-svd
exercise: 14-clustering-and-em
---

# Clustering and expectation maximisation

Every method so far has assumed labels. Clustering asks what the data
organises itself into when there are none: a partition into $k$ groups, chosen
to make each group look like a single object. That objective turns out to be
NP-hard to optimise exactly, which is why every practical algorithm for it is
a local search with a story about why the local optimum it finds is good
enough.

k-means is the sharpest version of that story, because its local search is
provably monotone and its seeding rule comes with a provable guarantee. It
is also a special case of something more general: fitting a Gaussian mixture
by maximum likelihood, where the group membership is a latent variable and
expectation maximisation is the general recipe for maximising a likelihood
you cannot see all of. This lesson derives both, and shows that they are the
same algorithm at two different values of a variance you can send to zero.

## What you already know

- The multivariate Gaussian density, and that its log is a negative quadratic
  form plus a normalising constant depending on the log-determinant of the
  covariance.
- Maximum likelihood estimation, and that the sample mean and the sample
  covariance are the maximum likelihood estimates of a single Gaussian's
  parameters.
- Jensen's inequality: for a concave function $\phi$ and a random variable
  $Z$, $\phi(\mathbb{E}[Z]) \geq \mathbb{E}[\phi(Z)]$.
- The Kullback-Leibler divergence $\mathrm{KL}(q \Vert p) = \mathbb{E}_q[\log
  (q/p)]$ is nonnegative, and zero exactly when $q = p$ almost everywhere.
- A convex optimisation problem with a compact feasible set attains its
  minimum, but a nonconvex one can have many local minima with no ordering
  between them.

## The k-means objective

Given points $x_1, \dots, x_n \in \mathbb{R}^d$ and a number of clusters $k$,
k-means chooses centers $\mu_1, \dots, \mu_k \in \mathbb{R}^d$ and an
assignment $z_i \in \{1, \dots, k\}$ for each point to minimise

$$
J(\{\mu_j\}, \{z_i\}) = \sum_{i=1}^{n} \lVert x_i - \mu_{z_i} \rVert_2^2 .
$$

This is a joint optimisation over a continuous variable (the centers) and a
combinatorial one (the assignment), and the combinatorial part is what makes
it hard: the number of distinct partitions to search over is the Stirling
number of the second kind $S(n,k)$, which grows asymptotically like
$k^n / k!$, and finding
the one that minimises $J$ is NP-hard, exactly for $k = 2$ when the dimension
$d$ is part of the input, and separately for any fixed $d \geq 2$ when $k$ is
part of the input. No algorithm is known that solves it exactly in time
polynomial in $n$, $d$, and $k$, and none is expected to exist.

Every practical algorithm therefore settles for a local minimum: an
assignment and a set of centers that cannot be improved by the moves the
algorithm is willing to make. Lloyd's algorithm is the standard one, and it
is worth stating exactly what kind of local minimum it finds.

## Lloyd's algorithm as alternating minimisation

Lloyd's algorithm fixes one of the two variables in $J$ and minimises over
the other, alternately:

1. **Assignment step.** With centers $\{\mu_j\}$ fixed, assign each point to
   its nearest center: $z_i \leftarrow \operatorname{arg\,min}_j \lVert x_i -
   \mu_j \rVert^2$.
2. **Update step.** With the assignment fixed, move each center to the mean
   of its assigned points: $\mu_j \leftarrow \frac{1}{|\{i : z_i = j\}|}
   \sum_{i : z_i = j} x_i$.

Both steps are exact minimisations of $J$ over the variable they touch, and
that is the entire content of the convergence proof.

**The assignment step cannot increase $J$.** For fixed centers, $J$
decomposes over points: $J = \sum_i \lVert x_i - \mu_{z_i}\rVert^2$, and each
term depends only on that point's own assignment. Setting $z_i$ to the index
of the nearest center minimises that term exactly, so no other assignment
gives a smaller or equal value for every point simultaneously while this one
does.

**The update step cannot increase $J$.** For fixed assignment, $J$
decomposes over clusters: $J = \sum_j \sum_{i : z_i = j} \lVert x_i - \mu_j
\rVert^2$. Each inner sum is a strictly convex quadratic in $\mu_j$ alone,
with gradient $2\sum_{i : z_i = j}(\mu_j - x_i)$, which vanishes exactly at
the mean of the assigned points. That stationary point is the unique global
minimiser of the cluster's contribution, by strict convexity.

**Termination.** The sequence of objective values produced by alternating
these two steps is nonincreasing, by the two claims above, and bounded below
by zero. It is also a sequence of values of $J$ evaluated at a sequence of
*partitions* of $\{1, \dots, n\}$ into $k$ groups, and there are only finitely
many such partitions. A nonincreasing real sequence indexed by a quantity
drawn from a finite set must eventually repeat a value, and once the
assignment step reproduces the previous partition exactly, the update step
reproduces the previous centers and the algorithm has reached a fixed point.
Ties in the assignment step are broken by a fixed rule (for instance, the
smallest index among nearest centers), which rules out the possibility of
oscillating between two assignments that give the same objective value. The
algorithm therefore terminates in finitely many iterations, at a point where
neither half-step can improve $J$ — a coordinatewise local minimum, jointly
over centers and assignment. Nothing about this argument says the point
reached is the global minimum, and in practice it usually is not.

## k-means++ seeding

Local search is only as good as where it starts. Uniform random seeding —
choosing $k$ data points as the initial centers — can lock Lloyd's algorithm
into a bad partition before it ever runs a single iteration: two centers can
land in the same true cluster, leaving another cluster with none, and the
alternating minimisation only ever refines that mistake.

k-means++ fixes the seeding distribution instead of the algorithm. It chooses
the first center uniformly at random from the data, and then chooses each
subsequent center from the remaining points with probability proportional to
its squared distance to the *nearest center already chosen*:

$$
\Pr[\text{choose } x_i \text{ next}] = \frac{D(x_i)^2}{\sum_{i'} D(x_{i'})^2},
\qquad D(x_i) = \min_{j \text{ chosen}} \lVert x_i - \mu_j \rVert .
$$

Points far from every existing center are likelier to be chosen, which
spreads the initial centers across the clusters instead of letting them
cluster together. This single change comes with a guarantee that uniform
seeding has none of: if $J^\star$ is the global minimum of the k-means
objective, then the expected objective after k-means++ seeding alone (before
any Lloyd iteration) satisfies

$$
\mathbb{E}[J] \leq 8(\ln k + 2)\, J^\star .
$$

The bound is $O(\log k)$ — it grows only logarithmically in the number of
clusters, in contrast to uniform seeding, which has no such bound at all: an
adversarial configuration of clusters can make uniform seeding arbitrarily
bad relative to $J^\star$. Running Lloyd's algorithm to convergence after
k-means++ seeding can only decrease $J$ further, so the bound carries over to
the local minimum the full algorithm reaches. The result is due to Arthur and
Vassilvitskii; its proof analyses, cluster by cluster, the probability that
k-means++ picks its next center from a cluster that has not yet received one,
and is a counting argument this lesson does not reproduce.

## Gaussian mixtures and the likelihood you cannot see

k-means commits to a hard assignment: every point belongs to exactly one
cluster. A Gaussian mixture model softens that commitment into a probability
distribution and asks for the maximum likelihood parameters instead of the
minimum-distortion partition.

Introduce a latent variable $z_i \in \{1, \dots, k\}$ for each point, with
$\Pr[z_i = j] = \pi_j$, $\sum_j \pi_j = 1$, and $x_i \mid z_i = j \sim
\mathcal{N}(\mu_j, \Sigma_j)$. Marginalising $z_i$ out gives the mixture
density

$$
p(x_i \mid \theta) = \sum_{j=1}^{k} \pi_j\, \mathcal{N}(x_i; \mu_j, \Sigma_j),
\qquad \theta = \{\pi_j, \mu_j, \Sigma_j\}_{j=1}^{k} .
$$

The **incomplete-data log-likelihood** — incomplete because $z$ is never
observed — is

$$
\ell(\theta) = \sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_j\, \mathcal{N}(x_i;
\mu_j, \Sigma_j) .
$$

This is a log of a sum, and the sum is inside the log for every term, which
is exactly the structure that makes maximum likelihood easy for a single
Gaussian and hard for a mixture: there is no closed form for the $\theta$
that maximises $\ell$, and $\ell$ is not concave in $\theta$. Had $z$ been
observed, the **complete-data log-likelihood**

$$
\log p(x, z \mid \theta) = \sum_{i=1}^{n} \Big( \log \pi_{z_i} + \log
\mathcal{N}(x_i; \mu_{z_i}, \Sigma_{z_i}) \Big)
$$

would separate over components and reduce to $k$ independent weighted
Gaussian fits, each with a closed form. EM is the procedure that gets the
benefit of that closed form without observing $z$.

## The EM derivation

Fix a distribution $q_i(z_i)$ over each point's latent label — write $q$ for
the collection — and decompose the log-likelihood of a single point using
$\log p(x_i \mid \theta) = \log \sum_{z_i} q_i(z_i) \frac{p(x_i, z_i \mid
\theta)}{q_i(z_i)}$. Summing over $i$ and defining

$$
\mathcal{L}(q, \theta) = \sum_{i=1}^{n} \sum_{z_i} q_i(z_i) \log
\frac{p(x_i, z_i \mid \theta)}{q_i(z_i)}
$$

gives the **evidence lower bound**, or ELBO. The exact identity underneath it
is

$$
\log p(x \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}\big(q(z) \,\Vert\,
p(z \mid x, \theta)\big) ,
$$

where $\log p(x\mid\theta) = \sum_i \log p(x_i\mid\theta)$. To see this,
expand the right side for a single point:

$$
\mathcal{L}_i + \mathrm{KL}_i = \sum_{z_i} q_i(z_i) \log \frac{p(x_i, z_i \mid
\theta)}{q_i(z_i)} + \sum_{z_i} q_i(z_i) \log \frac{q_i(z_i)}{p(z_i \mid x_i,
\theta)} .
$$

The $q_i(z_i)$ terms cancel between the two logs, leaving $\sum_{z_i}
q_i(z_i) \log \frac{p(x_i, z_i \mid \theta)}{p(z_i \mid x_i, \theta)}$, and
$p(x_i, z_i \mid \theta) / p(z_i \mid x_i, \theta) = p(x_i \mid \theta)$ by the
definition of conditional probability, a constant in $z_i$ that $\sum_{z_i}
q_i(z_i) = 1$ pulls out whole. This holds for **every** distribution $q$, not
only a good one, and since $\mathrm{KL} \geq 0$ always, $\mathcal{L}(q,
\theta) \leq \log p(x \mid \theta)$ for every $q$ — the ELBO is a lower bound
on the log-likelihood, with a gap equal to a KL divergence.

**The E-step closes the gap.** Fixing $\theta$ and maximising $\mathcal{L}(q,
\theta)$ over $q$ is the same as minimising the gap $\mathrm{KL}(q \Vert
p(z\mid x,\theta))$, because $\log p(x\mid\theta)$ does not depend on $q$. The
KL divergence is minimised, at zero, exactly when $q_i(z_i) = p(z_i \mid x_i,
\theta)$ — the posterior over the latent label given the current parameters.
So the optimal $q$ in the E-step is the **responsibility**

$$
q_i^\star(j) = \gamma_{ij} = p(z_i = j \mid x_i, \theta) = \frac{\pi_j\,
\mathcal{N}(x_i; \mu_j, \Sigma_j)}{\sum_{j'} \pi_{j'}\, \mathcal{N}(x_i;
\mu_{j'}, \Sigma_{j'})} ,
$$

and at this $q^\star$ the bound is tight: $\mathcal{L}(q^\star, \theta) =
\log p(x \mid \theta)$ exactly, with zero gap.

**The M-step cannot decrease the likelihood.** Having set $q = q^\star$ at
the current parameters $\theta^{\text{old}}$, the M-step maximises
$\mathcal{L}(q^\star, \theta)$ over $\theta$ to get $\theta^{\text{new}}$.
Chain three facts together:

$$
\log p(x \mid \theta^{\text{old}}) \;=\; \mathcal{L}(q^\star,
\theta^{\text{old}}) \;\leq\; \mathcal{L}(q^\star, \theta^{\text{new}})
\;\leq\; \log p(x \mid \theta^{\text{new}}) .
$$

The first equality is the E-step closing the gap at $\theta^{\text{old}}$.
The middle inequality is because the M-step maximises $\mathcal{L}(q^\star,
\cdot)$, so it cannot do worse at $\theta^{\text{new}}$ than it did at
$\theta^{\text{old}}$ — in particular it does no worse even without
maximising fully, as long as it does not decrease $\mathcal{L}$. The last
inequality is the general bound $\mathcal{L}(q, \theta) \leq \log p(x \mid
\theta)$ applied with $q = q^\star$ at the new $\theta$, where the gap is
generally no longer zero. The conclusion, $\log p(x \mid \theta^{\text{new}})
\geq \log p(x \mid \theta^{\text{old}})$, holds regardless of how tight the
middle inequality is: EM never decreases the incomplete-data likelihood, on
every iteration, for any model where the E-step and M-step are computed
exactly.

This argument is agnostic to what the model is. Everything about "Gaussian"
so far has been unused; it is only the M-step's closed form that depends on
the mixture being Gaussian.

## The M-step in closed form

Maximising $\mathcal{L}(q^\star, \theta) = \sum_i \sum_j \gamma_{ij} \big(
\log \pi_j + \log \mathcal{N}(x_i; \mu_j, \Sigma_j) \big)$ over $\theta$,
subject to $\sum_j \pi_j = 1$, separates into three independent problems.

For the weights, form the Lagrangian $\sum_{i,j} \gamma_{ij} \log \pi_j -
\lambda(\sum_j \pi_j - 1)$; differentiating in $\pi_j$ gives $\sum_i
\gamma_{ij} / \pi_j = \lambda$, so $\pi_j \propto \sum_i \gamma_{ij} =: N_j$,
and the constraint fixes the proportionality constant:

$$
\pi_j = \frac{N_j}{n}, \qquad N_j = \sum_{i=1}^{n} \gamma_{ij} .
$$

For $\mu_j$, only the quadratic form inside $\log \mathcal{N}$ depends on it;
differentiating $-\tfrac{1}{2}\sum_i \gamma_{ij} (x_i - \mu_j)^\top
\Sigma_j^{-1} (x_i - \mu_j)$ and setting it to zero gives $\sum_i \gamma_{ij}
\Sigma_j^{-1}(x_i - \mu_j) = 0$, and since $\Sigma_j^{-1}$ is invertible,

$$
\mu_j = \frac{1}{N_j} \sum_{i=1}^{n} \gamma_{ij}\, x_i .
$$

For $\Sigma_j$, differentiating the same term plus the $-\tfrac{1}{2}\log\det
\Sigma_j$ normaliser with respect to $\Sigma_j^{-1}$ and setting the result
to zero gives the responsibility-weighted scatter:

$$
\Sigma_j = \frac{1}{N_j} \sum_{i=1}^{n} \gamma_{ij} (x_i - \mu_j)(x_i -
\mu_j)^\top .
$$

Every update is a responsibility-weighted version of the ordinary Gaussian
maximum likelihood estimate: the same mean and covariance formulas, weighted
by how much of each point's probability mass the E-step assigned to that
component. This is exactly the closed form the complete-data likelihood
would have given if $z$ had been observed, with the indicator $\mathbb{1}[z_i
= j]$ replaced by its expectation $\gamma_{ij}$.

## k-means as the zero-variance limit of EM

Restrict the mixture to a shared isotropic covariance, $\Sigma_j = \sigma^2 I$
for every $j$, and inspect the responsibility as $\sigma \to 0$. Comparing
components $j$ and $j'$ for a fixed point $x_i$,

$$
\log \frac{\gamma_{ij}}{\gamma_{ij'}} = \log \frac{\pi_j}{\pi_{j'}} -
\frac{\lVert x_i - \mu_j \rVert^2 - \lVert x_i - \mu_{j'} \rVert^2}{2\sigma^2}
.
$$

The first term is a constant in $\sigma$. If $\lVert x_i - \mu_j\rVert^2 <
\lVert x_i - \mu_{j'}\rVert^2$ — component $j$ is strictly closer — the
second term diverges to $+\infty$ as $\sigma \to 0$, regardless of the
mixing weights, as long as every $\pi_j > 0$. So $\gamma_{ij} \to 1$ for the
nearest component and $\gamma_{ij'} \to 0$ for every other: the E-step
hardens into exactly the k-means assignment rule, $z_i =
\operatorname{arg\,min}_j \lVert x_i - \mu_j \rVert^2$.

With hardened responsibilities, $N_j \to |\{i : z_i = j\}|$ and the M-step
mean update $\mu_j = \frac{1}{N_j}\sum_i \gamma_{ij} x_i \to
\frac{1}{|\{i:z_i=j\}|} \sum_{i:z_i=j} x_i$ — exactly Lloyd's update step.
The covariance update degenerates too, to zero within each hardened cluster,
which is consistent with having sent $\sigma \to 0$ in the first place rather
than something the data determines. k-means is not an approximation to this
limit; it is the limit, with the variance held fixed at zero throughout
rather than annealed there.

## Degenerate solutions

The likelihood surface of a Gaussian mixture is unbounded above. Let one
component's mean coincide exactly with a single data point, $\mu_j = x_i$,
and let its covariance shrink, $\Sigma_j \to 0$ along any path that keeps it
positive definite. The density $\mathcal{N}(x_i; \mu_j, \Sigma_j)$ diverges,
because the normalising constant carries a factor of $\det(\Sigma_j)^{-1/2}$,
so $\log p(x_i \mid \theta) \to \infty$ and the incomplete-data likelihood
has no finite maximiser. EM does not usually walk into this singularity from
a reasonable start, because it requires a component to collapse onto a
single point while every other point stays away from it, but it happens on
real data whenever a component's effective sample size $N_j$ drifts toward
one during iteration — a small, tight cluster of size one is a stationary
point of the wrong kind.

The standard defenses are a floor on the eigenvalues of $\Sigma_j$ (adding
$\epsilon I$ after every M-step, or a technique in the exercise), a check
that discards any run where some $N_j$ falls below a minimum count, or
switching to a proper Bayesian treatment with a prior over $\Sigma_j$ — an
inverse-Wishart is conjugate — that gives the posterior a reason not to prefer
infinite density over a single point.

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
$d$: it never forms a $d \times d$ matrix. That gap is why k-means remains
the default first pass even when the eventual model is a mixture — using its
output, or the zero-variance limit's assignment directly, to seed
$\{\mu_j\}$ before turning the covariance back on. Both algorithms are
linear in $n$ per iteration, so the number of iterations to convergence, not
the per-iteration cost, dominates at scale; in practice both tend to make
most of their progress in the first handful of passes and then crawl.

## Where this breaks

**Non-convex clusters.** Both algorithms fit each component with a single
mean, implicitly assuming clusters are convex and, for k-means, roughly
spherical and similarly sized. Two clusters shaped like nested rings, or one
large and one small cluster of similar density, are not separable by nearest-
centroid assignment no matter how it is initialised. Spectral clustering,
which is essentially k-means run on the eigenvectors of a graph Laplacian
rather than on the raw coordinates, is the standard response.

**Choosing $k$.** Neither algorithm says how many clusters there are. $J$ and
$\ell$ both improve, weakly, as $k$ grows — a mixture with $k = n$ components
can place one component on every point and drive the likelihood to
infinity, per the degenerate-solutions discussion above. Practitioners use a
penalised criterion (the Bayesian information criterion for GMMs), the
elbow of $J$ against $k$, or a nonparametric alternative such as a Dirichlet
process mixture that treats $k$ itself as a random variable.

**Sensitivity to scale.** k-means minimises Euclidean distance, so a feature
measured in different units dominates the objective for no reason connected
to the clustering structure. Standardise features first, exactly as in
linear regression, or replace the Euclidean metric with a Mahalanobis
distance — which, for a shared covariance, is what a GMM with tied
$\Sigma_j = \Sigma$ effectively does.

**Local minima in high dimension.** The number of distinct local minima of
$J$ grows quickly with $d$ and $k$, and Lloyd's algorithm gives no guarantee
about which one it lands in beyond k-means++'s bound on the starting point.
Running the whole pipeline from several independent k-means++ seeds and
keeping the run with the lowest $J$ is standard practice, not a hack.

## The exercise

You implement `kmeans_objective`, `lloyd(X, k, init, iters)`, and
`kmeans_plusplus_init(X, k, rng)`, then `gmm_e_step`, `gmm_m_step`, and
`fit_gmm` for a Gaussian mixture with full covariance per component.
`lloyd` returns the history of the objective at every iteration, and
`fit_gmm` returns the history of the log-likelihood, so both monotone
properties derived above are things the checks can see directly rather than
take on faith.

The checks verify the objective on a hand-computed example, that Lloyd's
objective never increases across iterations on a synthetic dataset, that
k-means++ seeding beats uniform seeding on average over several seeds on a
planted five-cluster problem, that the E-step's responsibilities sum to one
per point and its log-likelihood matches a direct log-sum-exp computation,
that the M-step recovers the ordinary sample mean and covariance when the
responsibilities are hardened onto a single component, that EM's
log-likelihood never decreases and improves substantially over its
iterations, and that hardening the E-step with a shared, near-zero isotropic
covariance reproduces Lloyd's own assignment exactly — the zero-variance
limit, checked rather than only asserted.
