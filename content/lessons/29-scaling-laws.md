---
title: Scaling laws
slug: 29-scaling-laws
part: "Part 6 — Large language models"
summary: The power law that predicts a transformer's loss from its size and data, and the compute split it implies.
minutes: 40
gpu: false
objectives:
  - Count a transformer's parameters by component, and separate the embedding parameters from the rest.
  - State the compute estimate C ≈ 6ND, and say why the forward and backward pass account for the factor of six.
  - Compute the compute-optimal allocation N*(C) and D*(C) from a closed-form solution, and explain why the two exponents sum to one.
  - Explain what changed between the Kaplan and Chinchilla fits, and why inference cost moves the optimum further still.
prereqs:
  - 27-training-a-transformer
exercise: 29-scaling-laws
---

# Scaling laws

You know how to train a transformer. You don't yet know how big to make it.
Doubling the width, doubling the depth, and doubling the token count spend the
same compute budget in three different ways, and none of them trades off one
for one. A model twice as wide isn't twice as good. A model trained on twice
the data isn't either. Before scaling laws, the field picked a size by
intuition and by what fit on the hardware at hand.

This lesson makes a narrower claim precise: a transformer's pretraining loss,
as a function of its parameter count $N$ and its token count $D$, follows a
smooth, predictable curve. That curve lets you run a handful of small, cheap
experiments and predict the loss of a model a thousand times larger before you
pay to train it. It also turns "how big should this model be" into an
optimization problem with a closed-form answer, which the rest of this lesson
derives.

## What you already know

- A forward pass through an $(m, k)$ by $(k, n)$ matrix multiply costs $2mkn$
  flops, and the backward pass costs about twice that.
- A dense layer of shape $(d_{\text{in}}, d_{\text{out}})$ holds
  $d_{\text{in}} d_{\text{out}}$ parameters.
- A pre-norm transformer block is multi-head attention followed by a
  feed-forward network, each wrapped in a residual connection.
- A constrained optimum can be found by setting the gradient of the
  Lagrangian to zero.
- Ordinary least squares fits a line by minimizing a sum of squared
  residuals.

## Loss follows a power law

Train transformers of many sizes on many amounts of data, and the final
pretraining loss of each run sits close to

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} ,
$$

with $\alpha, \beta \in (0, 1)$ fit from the runs. $E$ is the loss an
infinitely large model trained on infinite data would still suffer — the
entropy of natural text under the tokenizer used. $A / N^{\alpha}$ is the
error from finite capacity, and it shrinks as $N$ grows. $B / D^{\beta}$ is
the error from finite data, and it shrinks as $D$ grows. The two error terms
are additive and independent: capacity and data act as separate bottlenecks,
not one shared resource. Holding $D$ fixed and sending $N \to \infty$ leaves
$E + B/D^{\beta}$ — an unlimited model still can't beat a data-limited floor.

## From Kaplan to Chinchilla

The same functional form was fit twice, three years apart, with different
conclusions. Kaplan and collaborators trained models across a wide range of
$N$, but gave each one a training budget set by a fixed schedule instead of
sweeping $N$ and $D$ independently at each compute level. Their fit said
compute should go overwhelmingly to parameters.

Hoffmann and collaborators, whose fit supplies this lesson's exercise its
constants, trained a grid that varied $N$ and $D$ independently at several
compute budgets. The fitted exponents came out close to equal, and the
conclusion reversed: most large models of the Kaplan era, GPT-3 included,
were undertrained for their size. A model a quarter the size, trained on four
times the data, would have reached the same loss for the same compute. The
lesson outlasts this one comparison: an extrapolated power law is only as
trustworthy as the experiment design behind it.

## Counting a transformer's parameters

Take a transformer with `layers` blocks, model width `d_model`, feed-forward
width `d_ff`, vocabulary size `vocab`, and a learned position-embedding table
of length `seq_len`. Each block's attention sublayer holds four square
projections, $W_Q$, $W_K$, $W_V$, and $W_O$, each $d_{\text{model}} \times
d_{\text{model}}$, for $4 d_{\text{model}}^2$ parameters regardless of how
many heads split that width. Each block's feed-forward sublayer holds
$W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $W_2 \in
\mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, for $2 d_{\text{model}}
d_{\text{ff}}$ parameters. Summed over `layers` blocks, and using the common
choice $d_{\text{ff}} = 4 d_{\text{model}}$, a block costs $12
d_{\text{model}}^2$ parameters — the figure quoted everywhere in the
scaling-law literature.

On top of that sit the token embedding, $\text{vocab} \times d_{\text{model}}$
parameters, and the position embedding, $\text{seq\_len} \times
d_{\text{model}}$ parameters, with the output projection tied to the token
embedding and so free. This split matters beyond bookkeeping: an embedding
lookup is a table index, not a matrix multiply, so it costs no flops beyond
the update itself. The $N$ in the power law above is the **non-embedding**
parameter count. Embedding parameters buy capacity without buying compute.

## Estimating training compute

Every non-embedding parameter takes part in one multiply-accumulate per token
during the forward pass, for $2N$ flops per token summed over all parameters.
Backpropagation needs two more matmuls of that same size — one gradient with
respect to the weights, one with respect to the incoming activations — for
another $4N$. Forward and backward together cost about $6N$ flops per token,
so training on $D$ tokens costs

$$
C \approx 6ND .
$$

The factor of six ignores the quadratic-in-sequence-length cost of attention
scores, which stays small next to $6N$ whenever $d_{\text{model}}$ is large
relative to the sequence length. It's accurate to within a few percent at
every scale this lesson's numbers are drawn from.

## Splitting compute optimally

Fix a compute budget $C$ and minimize $L(N, D)$ subject to $C = 6ND$. Setting
the Lagrangian's partial derivatives to zero and eliminating the multiplier
gives the stationarity condition

$$
\alpha A N^{-\alpha} = \beta B D^{-\beta} ,
$$

which says the two error terms' rates of improvement must balance: spending
one more unit of compute on parameters has to buy the same loss reduction as
spending it on tokens. Substituting $D = C/(6N)$ and solving for $N$ gives a
closed form. Writing $a = \beta/(\alpha+\beta)$ and $b = \alpha/(\alpha+\beta)$
— so $a + b = 1$ by construction —

$$
N^{*}(C) = G \, 6^{-a} \, C^{a}, \qquad
D^{*}(C) = \frac{C}{6 N^{*}(C)},
\qquad
G = \left( \frac{\alpha A}{\beta B} \right)^{1/(\alpha + \beta)} .
$$

Both optimal quantities are power laws in the compute budget, and $a + b = 1$
means the exponents split the budget between them in fixed proportion. When
$\alpha = \beta$, that split is exactly even: doubling compute should double
both $N$ and $D$, never one at the other's expense. That evenness, not any
particular exponent value, is the Chinchilla conclusion.

The exercise fixes $\alpha = \beta = 0.34$ and $(E, A, B) = (1.69, 400.0,
1100.0)$, close in spirit to the published fit. With equal exponents, the
ratio $D^{*}/N^{*}$ is constant across every compute budget, and for these
constants it works out to about 19.6 tokens per parameter — the single number
most often quoted from this line of work.

## Inference changes the optimum

The frontier above minimizes training compute alone. A model queried billions
of times after training pays a second bill, roughly $2N$ flops per generated
token, that scales with $N$ and not with $D$. Write the lifetime cost of a
model that will generate $D_{\text{inf}}$ tokens as $C_{\text{total}} = 6ND +
2N D_{\text{inf}}$, and minimize it subject to reaching a target loss instead
of a fixed training budget. Because the inference term grows only with $N$,
a larger $D_{\text{inf}}$ makes a smaller $N$ more attractive: the target loss
can still be met by shifting the burden onto $D$, which carries no inference
penalty. A model destined for heavy deployment should sit smaller and better
trained than $N^{*}(C)$ and $D^{*}(C)$ alone recommend.

## Fitting the law in log space

Given runs $(x_i, y_i)$ believed to follow $y = c x^{k}$, take logarithms:
$\log y_i = \log c + k \log x_i$. That's an ordinary linear regression in
$\log x$ and $\log y$, whose slope recovers $k$ and whose intercept
exponentiates to $c$. Fitting in the untransformed space would let one
large-scale run's residual dwarf every small run's; log-space regression is
what lets many cheap runs anchor the fit alongside a few expensive ones.

## Cost and scaling

The compute-optimal frontier turns a compute budget directly into a training
time. Assuming a sustained 200 TFLOP/s per accelerator and this lesson's
constants for $N^{*}(C)$ and $D^{*}(C)$:

| Compute $C$ (FLOPs) | $N^{*}$ | $D^{*}$ (tokens) | Single-accelerator time |
|---|---|---|---|
| $10^{18}$ | 92 M | 1.8 B | 1.4 hours |
| $10^{20}$ | 922 M | 18 B | 5.8 days |
| $10^{22}$ | 9.2 B | 181 B | 1.6 years |
| $10^{24}$ | 92 B | 1.8 T | 159 years |

The last row is why frontier training runs are distributed across thousands
of accelerators instead of sped up on one: the compute is fixed by the
frontier, and only parallelism turns years into weeks. Training memory scales
more gently, at roughly 16 bytes per parameter for mixed-precision Adam, so a
92-billion-parameter model needs on the order of 1.5 TB of optimizer state
alone.

## Where this breaks

**Data exhaustion.** $D^{*}(C)$ grows without bound as compute grows, but the
supply of distinct, high-quality tokens doesn't. Once a run needs more unique
tokens than exist, the choice is between repeating data, which degrades the
returns $B/D^{\beta}$ assumes, or accepting a worse loss than the frontier
promises.

**Small-scale extrapolation.** Fits built from small models are noisy:
optimizer variance and finite-size effects the smooth power law ignores are
largest exactly where the cheap experiments live. The Kaplan-to-Chinchilla
correction came from a flawed experiment design, not a flawed power-law form.

**Constants that don't transfer.** A change in data mixture, tokenizer, or
architecture invalidates $E$, $A$, $B$, $\alpha$, and $\beta$ without warning.
Only a fresh set of runs reveals that the constants have moved.

**Training compute isn't the only cost.** The frontier optimizes training
compute, not the total cost of building and serving a model, and the previous
section already showed how much inference volume can move the right answer.

**Loss isn't a benchmark score.** The fitted curve predicts pretraining
cross-entropy and nothing else. It doesn't predict accuracy on any downstream
task, which can move in visible steps even while the loss underneath moves
smoothly, and it says nothing about behavior introduced after pretraining.

## The exercise

You implement the machinery this lesson derived: a parameter counter, a
compute estimator, the compute-optimal frontier, and the log-space regression
that fits a power law's exponent from data.

Write `transformer_params(layers, d_model, d_ff, vocab, seq_len)`, returning a
breakdown by component — attention, feed-forward, embedding, non-embedding,
and total parameter counts — following the derivation above, with a tied
output projection and a learned position embedding. Write
`training_flops(params, tokens)`, returning $6 \times \text{params} \times
\text{tokens}$, and `chinchilla_optimal(compute)`, returning $(N^{*}, D^{*})$
from the closed form above using this lesson's documented constants. Write
`fit_power_law(x, y)`, returning the exponent and coefficient of the
best-fitting power law by least squares in log space, and
`predict_loss(N, D, coeffs)`, evaluating $L(N, D)$ for a five-element
`(E, A, B, alpha, beta)` tuple of coefficients.

The checks verify the parameter breakdown against a hand-computed value for a
stated architecture, that doubling `d_model` (with `d_ff` scaled to match)
exactly quadruples the non-embedding count, that `chinchilla_optimal`
satisfies $C = 6 N^{*} D^{*}$ to high precision across several orders of
magnitude of compute, that its token-to-parameter ratio lands near the twenty
this lesson derived, that `fit_power_law` recovers a planted exponent from
clean data to high precision and stays close under noise, and that
`predict_loss` decreases monotonically in both $N$ and $D$.
