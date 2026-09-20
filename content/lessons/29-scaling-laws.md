---
title: Scaling laws
slug: 29-scaling-laws
part: "Part 6 — Large language models"
summary: The power law that predicts a transformer's loss from its size and data, and the compute split it implies.
minutes: 50
gpu: false
objectives:
  - Derive a transformer's parameter count as 12 n_layer d_model^2 plus embeddings.
  - Justify the compute estimate C ≈ 6ND from the flop cost of a forward and backward pass.
  - Derive the compute-optimal allocation N*(C), D*(C) from a Lagrange multiplier and state why the exponents sum to one.
  - Explain what changed between the Kaplan and Chinchilla fits, and why inference cost moves the optimum further still.
prereqs:
  - 27-training-a-transformer
exercise: 29-scaling-laws
---

# Scaling laws

You now know how to train a transformer. What you do not know is how big to make
it. Doubling the width, doubling the depth, and doubling the token count are three
different ways to spend the same compute budget, and they do not trade off one for
one: a model twice as wide is not twice as good, and a model trained on twice the
data is not twice as good either. Before scaling laws, the field's answer was
intuition and whatever fit on the hardware at hand. After them, the answer is a
regression.

The claim this lesson makes precise is that a transformer's pretraining loss, as a
function of its parameter count $N$ and the number of tokens it trains on $D$,
follows a smooth and predictable curve. That curve lets you run a handful of
small, cheap experiments and extrapolate the loss of a model a thousand times
larger before you pay to train it, and it turns "how big should this model be"
into a constrained optimization problem with a closed-form answer — the
compute-optimal frontier this lesson derives.

## What you already know

- The chain rule cost of a matrix multiply: a forward pass through a $(m,k)$ by
  $(k,n)$ matmul costs $2mkn$ flops, and the backward pass costs about twice that.
- A dense layer of shape $(d_{\text{in}}, d_{\text{out}})$ holds
  $d_{\text{in}} d_{\text{out}}$ parameters.
- The structure of a pre-norm transformer block: multi-head attention followed by
  a feed-forward network, each wrapped in a residual connection.
- A Lagrange multiplier turns a constrained optimum into a stationarity condition
  on an unconstrained Lagrangian.
- Ordinary least squares minimises a sum of squared residuals and has a closed
  form in terms of means and covariances.

## The empirical power law

Train transformers of many sizes on many amounts of data, record the final
pretraining loss of each, and the results sit almost exactly on

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} ,
$$

with $\alpha, \beta \in (0, 1)$ fitted from the runs. Each term has a distinct
meaning:

- $E$ is the loss an infinitely large model trained on infinite data would still
  suffer, the entropy of natural text under whatever tokenizer you used. It is
  irreducible: no amount of $N$ or $D$ removes it.
- $A / N^{\alpha}$ is the error from finite capacity. A model with too few
  parameters cannot represent the true conditional distribution over tokens, and
  the gap shrinks as a power of $N$.
- $B / D^{\beta}$ is the error from finite data. A model sees only a sample of the
  distribution it is meant to learn, and the gap from that sampling shrinks as a
  power of $D$.

Holding $D$ fixed and letting $N \to \infty$ leaves $E + B/D^{\beta}$: a model of
unlimited size still cannot beat the data-limited floor. That the two error terms
are additive and independent is itself the empirical finding: capacity and data
act as separate bottlenecks, not a single effective resource.

## From Kaplan to Chinchilla

The form $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$ was fitted twice, three years
apart, with materially different conclusions.

Kaplan and collaborators trained models across a wide range of $N$, but gave each
one a training budget set by a fixed schedule rather than independently sweeping
$N$ and $D$ at each compute level, and did not retune the learning rate schedule
or warm-up per run. The resulting fit said compute should go overwhelmingly to
parameters: model size should grow close to linearly with compute while token
count grows far more slowly.

Hoffmann and collaborators, whose fit supplies this lesson's exercise its
constants, instead trained a grid of models that independently varied $N$ and $D$
at each of several fixed compute budgets, with a cosine decay tuned to the length
of each run. Under that more controlled design, the fitted exponents came out
close to equal, and the compute-optimal conclusion reversed: most large models of
the Kaplan era, GPT-3 included, were significantly undertrained for their size —
a model a quarter the size, trained on four times the data, would have reached
the same loss for the same compute. The general lesson outlasts this one
comparison: an extrapolated power law is only as trustworthy as the experimental
design it was fitted from, and a schedule confounded with model size was enough
to send the Kaplan exponents the wrong way.

## Counting a transformer's parameters

The compute estimate below needs a parameter count, so derive it first. Take a
transformer with `layers` blocks, model width `d_model`, feed-forward width
`d_ff`, vocabulary size `vocab`, and a learned position embedding table of length
`seq_len`, biases omitted as lower order.

Each block's attention sublayer holds four square projections, $W_Q$, $W_K$,
$W_V$, and $W_O$, each $d_{\text{model}} \times d_{\text{model}}$ regardless of
how many heads split that width, for $4 d_{\text{model}}^2$ parameters. Each
block's feed-forward sublayer holds $W_1 \in \mathbb{R}^{d_{\text{model}} \times
d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$,
for $2 d_{\text{model}} d_{\text{ff}}$ parameters. With the common choice
$d_{\text{ff}} = 4 d_{\text{model}}$, a block costs

$$
4 d_{\text{model}}^2 + 8 d_{\text{model}}^2 = 12 d_{\text{model}}^2 ,
$$

and $n_{\text{layer}}$ of them cost $12\, n_{\text{layer}}\, d_{\text{model}}^2$,
the figure quoted everywhere in the scaling-law literature. On top of it sit the
token embedding, $\text{vocab} \times d_{\text{model}}$ parameters, and the
position embedding, $\text{seq\_len} \times d_{\text{model}}$ parameters, with
the output projection tied to the token embedding and so free.

This split matters beyond bookkeeping: an embedding lookup is an index into a
table, not a matrix multiply, so it costs no floating point operations beyond the
update itself. The $N$ in the compute estimate below and in the power law above
is the **non-embedding** parameter count, $12\, n_{\text{layer}}\,
d_{\text{model}}^2$ under the $4\times$ feed-forward convention. Embedding
parameters buy capacity without buying compute, so flop-counting excludes them.

## The compute cost of training

Every non-embedding parameter takes part in exactly one multiply-accumulate per
token during the forward pass — one multiplication and one addition — so the
forward pass costs $2N$ flops per token, summed over all parameters.
Backpropagation needs two more matmuls of that same total size: one for the
gradient with respect to the weights, and one for the gradient with respect to
the incoming activations, so it can propagate to the layer before. Each costs
another $2N$ flops per token, for $4N$ in the backward pass. Forward and backward
together cost

$$
C_{\text{token}} \approx 2N + 4N = 6N \quad \text{flops per token},
$$

and training on $D$ tokens costs

$$
C \approx 6ND .
$$

The factor of six is an approximation: it ignores the quadratic-in-sequence-length
cost of the attention scores themselves, which is small next to $6N$ whenever
$d_{\text{model}}$ is large relative to the sequence length, and it treats every
token as independent, which activation checkpointing and gradient accumulation
preserve exactly. It is accurate to within a few percent for every transformer
scale this lesson's numbers are drawn from.

## The compute-optimal frontier

Fix a compute budget $C$ and ask how to split it between $N$ and $D$ to minimise
$L(N,D)$ subject to $C = 6ND$. Form the Lagrangian

$$
\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
+ \lambda (6ND - C) ,
$$

and set both partial derivatives to zero:

$$
\frac{\partial \mathcal{L}}{\partial N} = -\alpha A N^{-\alpha - 1} + 6\lambda D = 0,
\qquad
\frac{\partial \mathcal{L}}{\partial D} = -\beta B D^{-\beta - 1} + 6\lambda N = 0 .
$$

Solving each for $\lambda$ and equating gives $\alpha A N^{-\alpha - 1} D =
\beta B D^{-\beta - 1} N$, which simplifies to

$$
\alpha A N^{-\alpha} = \beta B D^{-\beta} .
$$

Substitute $D = C / (6N)$ and collect powers of $N$:

$$
\alpha A N^{-\alpha} = \beta B \left(\frac{6N}{C}\right)^{\beta}
\;\Longrightarrow\;
N^{\alpha + \beta} = \frac{\alpha A}{\beta B \, 6^{\beta}} \, C^{\beta} .
$$

Writing $a = \beta / (\alpha + \beta)$ and $b = \alpha / (\alpha + \beta)$, so that
$a + b = 1$ by construction,

$$
N^{*}(C) = G \, 6^{-a} \, C^{a}, \qquad
D^{*}(C) = \frac{C}{6 N^{*}(C)} = \frac{C^{b}}{G \, 6^{b}},
\qquad
G = \left( \frac{\alpha A}{\beta B} \right)^{1/(\alpha + \beta)} .
$$

Both optimal quantities are power laws in the compute budget, and the exponents
split the budget between them in fixed proportion. When $\alpha = \beta$, the
split is exactly even, $a = b = 1/2$: doubling the compute budget should double
both the parameter count and the token count, never one at the expense of the
other. That evenness, not any particular pair of exponents, is the Chinchilla
conclusion. It follows from the same equation as the Kaplan result; only the
data that went into fitting $\alpha$ and $\beta$ changed.

The exercise fixes $\alpha = \beta = 0.34$ and $(E, A, B) = (1.69, 400.0,
1100.0)$, close in spirit to the published fit. With equal exponents the ratio
$D^{*}/N^{*} = 1/G^2$ is constant across every compute budget, and for these
constants it works out to

$$
\frac{D^{*}(C)}{N^{*}(C)} \approx 19.6 \; \text{tokens per parameter} ,
$$

independent of $C$. That a 100-million-parameter model and a 100-billion-
parameter model should both be trained at close to twenty tokens per parameter is
the single number most often quoted from this line of work.

## Inference changes the optimum

The frontier above minimises training compute alone. A model trained once and
queried billions of times pays a second bill, roughly $2N$ flops per generated
token, that scales with $N$ and not with $D$. Write the lifetime cost of a model
that will generate $D_{\text{inf}}$ tokens after training as

$$
C_{\text{total}} = 6ND + 2N D_{\text{inf}} ,
$$

and minimise it subject to reaching a target loss $L(N, D) = L_{\text{target}}$
instead of a fixed training budget. Because the inference term grows linearly in
$N$ alone, increasing $D_{\text{inf}}$ makes small $N$ more attractive: the
constraint can still be met by pushing more of the burden onto $D$, which carries
no inference penalty. A model destined for heavy deployment should sit to the
left of the training-compute-optimal point, smaller and trained on more tokens
than $N^{*}(C)$ and $D^{*}(C)$ alone recommend — every token of extra training is
repaid each time the smaller model answers a query more cheaply than a larger
one would.

## Fitting the law in log space

Given a handful of runs $(x_i, y_i)$ that are believed to follow $y = c
x^{k}$, take logarithms: $\log y_i = \log c + k \log x_i$, an ordinary linear
regression in $\log x$ and $\log y$ that recovers the exponent $k$ as its slope.

Fitting in the untransformed space instead, minimising $\sum_i (y_i - c
x_i^{k})^2$ directly, weights every point by the square of its own scale: a run
at $x = 10^{9}$ contributes a residual that dwarfs anything a run at $x = 10^{6}$
could produce, however badly the small run is fit. Fitting in log space
minimises $\sum_i (\log y_i - \log \hat{y}_i)^2$, which is, to first order, a sum
of squared *relative* errors, $\sum_i \big((y_i - \hat{y}_i)/y_i\big)^2$: a ten
percent error counts the same whether it comes from the cheapest run in the study
or the most expensive one. A scaling-law study can usually afford many small runs
and only a few large ones, so log-space regression is what lets the many cheap
points anchor the fit rather than being swamped by the one or two large enough to
dominate an unweighted sum of squares.

## What scaling laws do not tell you

The fitted curve predicts pretraining cross-entropy, and only that. It does not
predict accuracy on any downstream benchmark, which can move in steps that look
discontinuous even while the loss underneath moves smoothly, and it does not
predict when a specific capability appears, only that the loss will keep
falling. The constants $E$, $A$, $B$, $\alpha$, and $\beta$ belong to one data
distribution, one tokenizer, and one architecture family: change any of them and
the constants need refitting, not just rescaling. It says nothing about
behaviour introduced after pretraining, and it is only as reliable outside the
compute range it was fitted on as the experimental design that produced it —
precisely the failure mode that separated the Kaplan exponents from the
Chinchilla ones.

## Cost and scaling

The compute-optimal frontier turns a compute budget directly into a training
time. Assuming a sustained 200 TFLOP/s per accelerator, a realistic bf16 figure
for a single modern accelerator at reasonable utilisation, and using this
lesson's constants for $N^{*}(C)$ and $D^{*}(C)$:

| Compute $C$ (FLOPs) | $N^{*}$ | $D^{*}$ (tokens) | Single-accelerator time |
|---|---|---|---|
| $10^{18}$ | 92 M | 1.8 B | 1.4 hours |
| $10^{20}$ | 922 M | 18 B | 5.8 days |
| $10^{22}$ | 9.2 B | 181 B | 1.6 years |
| $10^{24}$ | 92 B | 1.8 T | 159 years |

The last row is why frontier training runs are distributed across thousands of
accelerators rather than sped up on one: the compute is fixed by the frontier,
and only parallelism turns years into weeks. Training memory scales more gently,
at roughly 16 bytes per parameter for mixed-precision Adam, so a 92 billion
parameter model needs on the order of 1.5 TB of optimizer state alone, split
across the same accelerators that share the compute.

## Where this breaks

**Data exhaustion.** $D^{*}(C)$ grows without bound as compute grows, but the
supply of distinct, high-quality tokens does not. Once a training run would need
more unique tokens than exist, the choice is between repeating data, which
degrades the returns that $B/D^{\beta}$ assumes, or accepting a worse loss than
the frontier promises.

**Small-scale extrapolation.** Fits built from small models are noisy: optimizer
variance and finite-size effects that the smooth power law ignores are largest
exactly where the cheap experiments live. The Kaplan-to-Chinchilla correction
happened because a fit's own experimental design, not the power-law form, was
wrong.

**Nonstationary constants.** A change in data mixture, tokenizer, or
architecture invalidates $E$, $A$, $B$, $\alpha$, and $\beta$ without warning.
Nothing about the functional form tells you the constants have moved; only a
fresh set of runs does.

**A single-objective optimum.** The frontier optimises training compute, not
the total cost of building and serving a model, and the previous section already
showed how much that omission can move the right answer once inference volume is
large.

## The exercise

You implement the machinery this lesson derived: a parameter counter, a compute
estimator, the compute-optimal frontier, and the log-space regression that fits a
power law's exponent from data.

Write `transformer_params(layers, d_model, d_ff, vocab, seq_len)`, returning a
breakdown by component — attention, feed-forward, embedding, non-embedding, and
total parameter counts — following the derivation above with a tied output
projection and a learned position embedding. Write `training_flops(params,
tokens)` returning $6 \times \text{params} \times \text{tokens}$, and
`chinchilla_optimal(compute)`, returning $(N^{*}, D^{*})$ from the closed form
derived above using this lesson's documented constants. Write `fit_power_law(x,
y)`, returning the exponent and coefficient of the best-fitting power law by
least squares in log space, and `predict_loss(N, D, coeffs)`, evaluating
$L(N,D)$ for a five-element `(E, A, B, alpha, beta)` tuple of coefficients.

The checks verify the parameter breakdown against a hand-computed value for a
stated architecture, that doubling `d_model` (with `d_ff` scaled to match)
exactly quadruples the non-embedding count, that `chinchilla_optimal` satisfies
$C = 6 N^{*} D^{*}$ to high precision across several orders of magnitude of
compute, that its token-to-parameter ratio lands near the twenty this lesson
derived, that `fit_power_law` recovers a planted exponent from clean data to
high precision and stays close under noise, and that `predict_loss` decreases
monotonically in both $N$ and $D$, which is the property that makes the whole
frontier argument well posed in the first place.
