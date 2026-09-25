---
title: Scaling laws
slug: 29-scaling-laws
part: "Part 6 — Large language models"
summary: Predict a transformer's loss from its size and data, and split a compute budget between the two.
minutes: 50
gpu: false
objectives:
  - Count a transformer's parameters by component, and separate the embedding parameters from the rest.
  - Compute training cost with C = 6ND, and explain where the six comes from.
  - Compute the best model size and token count for a compute budget from a closed-form rule.
  - Fit a power law with a straight line in log space.
  - Explain why the Chinchilla result changed how big models are trained, and how inference cost shifts the answer.
prereqs:
  - 27-training-a-transformer
exercise: 29-scaling-laws
---

# Scaling laws

You know how to train a transformer. You don't yet know how big to make it.
You could make it wider, make it deeper, or feed it more text. Each choice
spends compute in a different way, and none of them pays back one for one. A
model twice as big isn't twice as good.

Scaling laws answer the sizing question with a formula. Researchers trained
hundreds of models of different sizes on different amounts of text. The final
loss of each run landed close to one smooth curve. With that curve, you can run
a few small, cheap experiments and predict how a model a thousand times larger
will do before you pay to train it.

The curve also turns "how big should the model be?" into a small optimisation
problem with an exact answer. In this lesson you count parameters, estimate
compute, find the best split of a budget, and fit a curve to data.

## What you already know

- A dense layer with `d_in` inputs and `d_out` outputs holds `d_in * d_out`
  weights.
- A transformer block is attention followed by a feed-forward network, each
  with a residual connection.
- Backpropagation costs about twice as much as the forward pass.
- A straight-line fit by least squares finds the slope and intercept that
  minimise the squared errors.
- Taking a logarithm turns multiplication into addition and powers into
  multiplication.

## Loss follows a power law

Think of two bottlenecks on a factory line. One is how skilled the workers
are: the model's size. The other is how much raw material arrives: the
training data. Improving either one helps, but with shrinking returns. And even
a perfect factory can't beat the limit set by the material itself.

The loss $L$ of a trained model with $N$ parameters and $D$ training tokens is
close to

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Read it piece by piece:

- $E$ is the floor. It's the loss even an infinite model on infinite data
  would still have, because text itself is partly unpredictable.
- $A / N^{\alpha}$ is the error from having too few parameters. It shrinks as
  $N$ grows.
- $B / D^{\beta}$ is the error from having too little data. It shrinks as $D$
  grows.
- $\alpha$ and $\beta$ are exponents between 0 and 1. They set how fast each
  error term shrinks. $A$ and $B$ set how big each term starts.

A *power law* is any formula of the shape $c \cdot x^{k}$. Both error terms
are power laws with a negative exponent. The five numbers $E, A, B, \alpha,
\beta$ are fit from experiments. This lesson uses $E = 1.69$, $A = 400$,
$B = 1100$, and $\alpha = \beta = 0.34$, which are close to the published
Chinchilla fit.

In numpy, the formula is one line:

```python
E, A, B, alpha, beta = coeffs
loss = E + A / N**alpha + B / D**beta
```

### Worked example

Take $N = 10^9$ parameters and $D = 2 \times 10^{10}$ tokens:

- $A / N^{\alpha} = 400 / (10^9)^{0.34} = 400 / 1148.2 \approx 0.3484$
- $B / D^{\beta} = 1100 / (2 \times 10^{10})^{0.34} = 1100 / 3179.4 \approx 0.3460$
- $L = 1.69 + 0.3484 + 0.3460 \approx 2.3844$

The two error terms are almost the same size here. That's a hint of the
balance you find later in this lesson.

> [!TIP]
> Check each scaling function against a worked value before you trust it at
> $10^{24}$. `predict_loss(1e9, 2e10, CHINCHILLA_COEFFS)` should give about
> 2.3844.

Holding $D$ fixed and letting $N$ grow forever leaves $E + B / D^{\beta}$. An
unlimited model still can't beat the limit set by its data.

> [!KEY]
> Loss falls as a power law in both model size and data, down to a floor $E$.
> Growing only one of them leaves the other error term as a limit.

## Count a transformer's parameters

The $N$ in the law counts the weights that do matrix multiplication. Here's
where they live in a transformer with `layers` blocks, width `d_model`, and
feed-forward width `d_ff`:

- **Attention.** Each block has four square matrices: query, key, value, and
  output. Each is `d_model` by `d_model`. That's $4 d_{\text{model}}^2$ per
  block, no matter how many heads split the width.
- **Feed-forward.** Each block has one `d_model` by `d_ff` matrix and one
  `d_ff` by `d_model` matrix. That's $2 d_{\text{model}} d_{\text{ff}}$ per
  block.
- **Embeddings.** One token table of `vocab` rows and one position table of
  `seq_len` rows, each row `d_model` wide. That's
  $(\text{vocab} + \text{seq\_len}) \cdot d_{\text{model}}$ in total. They
  appear once, not once per layer. The output layer reuses the token table,
  so it adds nothing.

The non-embedding count is attention plus feed-forward, summed over all
blocks:

$$
N = \text{layers} \cdot \left(4 d_{\text{model}}^2 + 2 d_{\text{model}} d_{\text{ff}}\right)
$$

In words: per block, four square attention matrices plus two feed-forward
matrices, times the number of blocks. With the common choice $d_{\text{ff}} =
4 d_{\text{model}}$, a block holds $12 d_{\text{model}}^2$ weights.

Why leave embeddings out of $N$? An embedding lookup is a table index, not a
matrix multiply. It costs almost no compute. The scaling law tracks the
parameters that cost compute.

> [!KEY]
> The $N$ in the scaling law counts only the non-embedding weights: four
> attention matrices and two feed-forward matrices per block. Embeddings count
> once, and they sit outside $N$.

### Worked example

Take `layers=2`, `d_model=8`, `d_ff=32`, `vocab=100`, `seq_len=16`:

| Part | Formula | Value |
|---|---|---|
| Attention | $2 \cdot 4 \cdot 8^2$ | 512 |
| Feed-forward | $2 \cdot 2 \cdot 8 \cdot 32$ | 1024 |
| Embedding | $(100 + 16) \cdot 8$ | 928 |
| Non-embedding | $512 + 1024$ | 1536 |
| Total | $1536 + 928$ | 2464 |

## Estimate training compute

Compute is counted in *flops*, short for floating-point operations: one
multiply or one add. During the forward pass, every weight does one multiply
and one add per token. That's $2N$ flops per token. The backward pass does two
more matrix multiplies of the same size: one for the gradient of the weights
and one for the gradient of the inputs. That's another $4N$. So training on $D$
tokens costs

$$
C = 6ND
$$

Read it as: six flops per parameter per token. The six is two for the forward
pass plus four for the backward pass. This skips the attention-score cost,
which stays small when the model is wide compared with its sequence length.

In numpy terms, it's `6.0 * params * tokens`. Use floats: $6 \times 7 \times
10^9 \times 3 \times 10^{11}$ is about $1.3 \times 10^{22}$, which overflows a
64-bit integer.

For example, $N = 10^9$ and $D = 2 \times 10^{10}$ give $C = 6 \times 10^9
\times 2 \times 10^{10} = 1.2 \times 10^{20}$ flops.

## Split a compute budget

Now fix the budget $C$. You can spend it on a big model with few tokens or a
small model with many tokens, as long as $6ND = C$. Which split gives the
lowest loss?

Picture two buckets you fill with the same water. You stop pouring into one
when an extra cup there helps less than an extra cup in the other. At the best
split, the last unit of compute buys the same loss drop whether it goes to
parameters or to data. Working that condition through with calculus gives an
exact answer. You don't need the derivation, only the result.

First, define two helper numbers:

$$
a = \frac{\beta}{\alpha + \beta}, \qquad
G = \left( \frac{\alpha A}{\beta B} \right)^{1/(\alpha + \beta)}
$$

Then the best model size and token count are

$$
N^{*} = G \cdot 6^{-a} \cdot C^{a}, \qquad D^{*} = \frac{C}{6 N^{*}}
$$

Read it aloud:

- $a$ is the share of each new order of magnitude of compute that goes to
  model size. With $\alpha = \beta$, $a = 0.5$: half to size, half to data.
- $G$ is a constant that depends only on the fit. It sets the starting ratio
  between size and data.
- $N^{*}$ grows as $C^{a}$. With $a = 0.5$, a budget 100 times bigger makes
  the best model 10 times bigger.
- $D^{*}$ is whatever token count spends the rest of the budget, from $C =
  6ND$.

The star in $N^{*}$ marks "the best value". In numpy:

```python
a = beta / (alpha + beta)
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
n_star = G * 6.0 ** (-a) * compute ** a
```

### Worked example

Use $C = 10^{20}$ flops and this lesson's constants:

- $a = 0.34 / 0.68 = 0.5$
- $G = (0.34 \cdot 400 / (0.34 \cdot 1100))^{1/0.68} = (0.3636)^{1.4706}
  \approx 0.2259$
- $6^{-0.5} \approx 0.4082$, and $(10^{20})^{0.5} = 10^{10}$
- $N^{*} = 0.2259 \times 0.4082 \times 10^{10} \approx 9.22 \times 10^{8}$
- $D^{*} = 10^{20} / (6 \times 9.22 \times 10^{8}) \approx 1.81 \times 10^{10}$

That's about 922 million parameters trained on 18 billion tokens. The ratio
$D^{*} / N^{*}$ is about 19.6. Because $\alpha = \beta$, this ratio is the same
at every budget. This is the famous rule of thumb: about 20 tokens per
parameter.

> [!KEY]
> For a fixed budget, the best model size grows as $C^{a}$. With
> $\alpha = \beta$, that means about 20 training tokens per parameter at every
> budget.

## From Kaplan to Chinchilla

The same formula was fit twice, a few years apart, with opposite advice. The
first study, by Kaplan and colleagues, trained each model size with a fixed
training schedule instead of varying size and data separately. Their fit said
to spend most extra compute on a bigger model.

The second study, by Hoffmann and colleagues, known as Chinchilla, varied
model size and token count independently at several budgets. Their exponents
came out nearly equal, so the best split is even. It meant that many large
models of that time, GPT-3 among them, were undertrained. A model a quarter
the size, trained on four times the data, would reach the same loss for the
same compute. The wider lesson: a fitted curve is only as good as the
experiments behind it.

## Inference changes the answer

The split above minimises training compute only. A deployed model also pays
for every token it generates, about $2N$ flops each. That bill grows with $N$
and doesn't depend on $D$. If a model will serve billions of requests, a
smaller model trained on more tokens can reach the same loss and cost less
overall. That's why widely deployed models are often trained well past 20
tokens per parameter.

## Fit a power law in log space

To get the constants in the first place, you fit a power law $y = c \cdot
x^{k}$ to experimental points. Curved data is hard to fit. Take logarithms
and it becomes a straight line:

$$
\log y = \log c + k \log x
$$

In words: on a log-log plot, the slope is the exponent $k$ and the intercept
is $\log c$. So you fit an ordinary straight line to $(\log x, \log y)$. The
least-squares slope is the covariance of the two over the variance of
$\log x$:

$$
k = \frac{\sum_i (u_i - \bar{u})(v_i - \bar{v})}{\sum_i (u_i - \bar{u})^2},
\qquad c = \exp(\bar{v} - k \bar{u})
$$

Here $u_i = \log x_i$, $v_i = \log y_i$, and the bar means the average. Read
it as: center both log arrays, multiply them together and sum, and divide by
the sum of squared centered $u$. The intercept makes the line pass through the
average point. In numpy:

```python
u, v = np.log(x), np.log(y)
uc = u - u.mean()
k = np.sum(uc * (v - v.mean())) / np.sum(uc**2)
```

> [!WARNING]
> The line's intercept is $\log c$, not $c$. Take `np.exp` of the intercept
> before you return it.

Log space also stops one huge run from dominating the fit. Every run counts
by its relative error.

### Worked example

Take $x = [1, 2, 4]$ and $y = [2, 8, 32]$:

- $u = [0, 0.693, 1.386]$, average $\bar{u} = 0.693$
- $v = [0.693, 2.079, 3.466]$, average $\bar{v} = 2.079$
- Centered $u$: $[-0.693, 0, 0.693]$. Centered $v$: $[-1.386, 0, 1.386]$
- Numerator: $0.961 + 0 + 0.961 = 1.922$. Denominator: $0.480 + 0 + 0.480 =
  0.961$
- $k = 1.922 / 0.961 = 2$, and $c = \exp(2.079 - 2 \times 0.693) = \exp(0.693)
  = 2$

The fit recovers $y = 2x^{2}$ exactly.

> [!TIP]
> Test your fit on data you made yourself, such as `y = 3 * x**1.7`. It should
> return $k = 1.7$ and $c = 3$ to many decimal places.

> [!RECAP]
> - Loss is $E + A / N^{\alpha} + B / D^{\beta}$, unpacked from
>   `(E, A, B, alpha, beta)` in that order.
> - $N$ is `layers` times the attention and feed-forward weights per block.
>   Embeddings count once.
> - Training compute is $C = 6ND$, computed in floats.
> - A power law fit is a least-squares line on $(\log x, \log y)$, and
>   $c = e^{\text{intercept}}$.
> - The best split is $N^{*} = G \cdot 6^{-a} \cdot C^{a}$ and
>   $D^{*} = C / (6 N^{*})$.

## Build it step by step

The checks test the functions in this order.

1. **`predict_loss(N, D, coeffs)`.** Takes two numbers and a 5-tuple
   `(E, A, B, alpha, beta)`. Returns one float. Unpack the tuple and write the
   formula from the first section. Common mistake: unpacking the tuple in the
   wrong order. Check against the worked value of 2.3844.
2. **`training_flops(params, tokens)`.** Returns `6 * params * tokens` as a
   float. Common mistake: integer arithmetic that overflows at large scales.
   Convert with `float()` first.
3. **`transformer_params(layers, d_model, d_ff, vocab, seq_len)`.** Returns a
   dict with the keys `"attention"`, `"feedforward"`, `"embedding"`,
   `"non_embedding"`, and `"total"`, each an integer. Common mistake:
   multiplying the embedding count by `layers`. Embeddings appear once. Check
   against the table above.
4. **`fit_power_law(x, y)`.** Takes two positive arrays of shape `(n,)`.
   Returns `(k, c)`. Use `np.log`, `.mean()`, `np.sum`, and `np.exp`. Common
   mistake: returning the intercept $\log c$ instead of $c$ itself.
5. **`chinchilla_optimal(compute)`.** Takes a budget in flops. Returns
   `(n_star, d_star)`. Unpack `CHINCHILLA_COEFFS`, compute `a` and `G`, then
   $N^{*}$, then $D^{*}$ from $C = 6ND$. Don't run an optimiser. Common
   mistake: dropping the $6^{-a}$ factor. The budget still adds up, because
   $D^{*}$ comes from $N^{*}$, but the ratio $D^{*} / N^{*}$ drifts away from
   about 20. Check against $N^{*} \approx 9.22 \times 10^{8}$ at $C = 10^{20}$.

## Cost and scaling

The best split turns a budget into a training time. The table assumes one
accelerator that sustains 200 trillion flops per second, and this lesson's
constants.

| Compute $C$ (flops) | $N^{*}$ | $D^{*}$ (tokens) | Time on one accelerator |
|---|---|---|---|
| $10^{18}$ | 92 M | 1.8 B | 1.4 hours |
| $10^{20}$ | 922 M | 18 B | 5.8 days |
| $10^{22}$ | 9.2 B | 181 B | 1.6 years |
| $10^{24}$ | 92 B | 1.8 T | 159 years |

The last row is why big training runs spread across thousands of
accelerators. The compute is fixed; only running in parallel turns years into
weeks. Memory grows more gently: training with the Adam optimiser in mixed
precision needs about 16 bytes per parameter. A 92-billion-parameter model
needs about 1.5 TB for weights and optimiser state alone.

Your exercise functions are all tiny. Each one does a handful of arithmetic
operations, or one pass over the `n` data points for the fit.

## Where this breaks

**Running out of data.** $D^{*}$ keeps growing with compute, but the supply of
good, distinct text doesn't. Past that point you either repeat data, which
gives less than the law promises, or accept a worse loss.

**Noisy small runs.** Fits come from small, cheap models, where random
variation between runs is largest. A small error in an exponent grows large
when you extrapolate a thousandfold. People run several seeds and fit across a
wide range of sizes.

**Constants that don't transfer.** Change the data mix, the tokeniser, or the
architecture, and all five constants can move. Only fresh experiments reveal
it.

**Training cost isn't total cost.** The best split ignores serving cost. For a
heavily used model, train smaller and longer.

**Loss isn't a test score.** The law predicts pretraining loss and nothing
else. Accuracy on a specific task can jump suddenly even while the loss
improves smoothly.

## The exercise

You implement the tools from this lesson. Write `predict_loss(N, D, coeffs)`
for the loss formula, `training_flops(params, tokens)` for $6ND$, and
`transformer_params(layers, d_model, d_ff, vocab, seq_len)` for the
parameter breakdown by component. Then write `fit_power_law(x, y)`, which
returns the exponent and coefficient of a power law fit in log space, and
`chinchilla_optimal(compute)`, which returns $(N^{*}, D^{*})$ from the closed
form using `CHINCHILLA_COEFFS`.

The checks verify the loss formula and that it falls as $N$ or $D$ grows, the
exact compute formula, and the parameter breakdown against a hand count. They
check that doubling `d_model` and `d_ff` quadruples the non-embedding count,
that `fit_power_law` recovers a planted exponent with and without noise, and
that `chinchilla_optimal` spends the whole budget with about 20 tokens per
parameter.
