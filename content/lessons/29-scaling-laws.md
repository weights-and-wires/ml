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

You can train a transformer, and you still have to pick its width, its depth, and how much text to feed it. Doubling the width leaves a large share of the loss in place.

Researchers trained many models of different sizes on different amounts of text. The final loss sat close to one smooth curve, so a few cheap runs predict a much larger model.

That curve turns sizing into a small optimisation. You count parameters, estimate training cost, split a budget between size and data, and fit the curve.

## What you already know

This lesson uses five ideas from earlier work:

- A dense layer with `d_in` inputs and `d_out` outputs holds `d_in * d_out` weights.
- A transformer block is attention, then a feed-forward network, each with a residual connection.
- The backward pass costs about twice the forward pass.
- Least squares finds the slope and intercept that minimise squared errors.
- A logarithm turns a product into a sum and a power into a product.

## Predict a loss from size and data

Take $N = 10^9$ parameters and $D = 2 \times 10^{10}$ tokens. Use a floor of 1.69, a size coefficient of 400, a data coefficient of 1100, and both exponents 0.34.

$(10^9)^{0.34} = 10^{3.06} \approx 1148.2$, so $400 / 1148.2 \approx 0.3484$. $(2 \times 10^{10})^{0.34} \approx 3179.4$, so $1100 / 3179.4 \approx 0.3460$. Add the floor: $1.69 + 0.3484 + 0.3460 \approx 2.3844$. The two error pieces are almost equal.

This is a scaling law. Each shrinking piece is a power law, a formula of the shape $c \cdot x^{k}$ with $k$ negative. The Greek letters are $\alpha$ (alpha, pronounced AL-fuh) and $\beta$ (beta, pronounced BEE-tuh):

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

$E$ is the floor left when text is partly unpredictable. $A / N^{\alpha}$ is too few parameters. $B / D^{\beta}$ is too few tokens. The same $N$ and $D$ give 2.3844 again. The expression has this shape:

```python
E, A, B, alpha, beta = coeffs  # that order
loss = E + A / N**alpha + B / D**beta
```

Hold $D$ fixed and let $N$ grow. The loss falls to $E + B / D^{\beta}$ and stops. The usual mistake unpacks `(E, A, B, alpha, beta)` in another order.

> [!TIP]
> Check `predict_loss(1e9, 2e10, CHINCHILLA_COEFFS)` before a budget of $10^{24}$. The value is about 2.3844.

> [!KEY]
> Loss falls as a power law in model size and in data, down to a floor $E$. Growing only one of them leaves the other term in place.

## Count the parameters that multiply

Take 2 blocks, width 8, feed-forward width 32, vocabulary 100, and 16 positions. Four square attention matrices per block give $2 \times 4 \times 8^2 = 512$. Two feed-forward matrices per block give $2 \times 2 \times 8 \times 32 = 1024$. The token table and the position table appear once: $(100 + 16) \times 8 = 928$. The output layer reuses the token table. Non-embedding weights are $512 + 1024 = 1536$. The total is $1536 + 928 = 2464$.

Those 1536 weights are the $N$ in the scaling law. An embedding lookup is an index, so it costs almost no flops:

$$
N = \text{layers} \cdot \left(4 d_{\text{model}}^2 + 2 d_{\text{model}} d_{\text{ff}}\right)
$$

Per block that is four attention matrices plus two feed-forward matrices, times the blocks. With $d_{\text{ff}} = 4 d_{\text{model}}$, one block holds $12 d_{\text{model}}^2$ weights. The usual mistake multiplies the embedding count by `layers`.

> [!KEY]
> The $N$ in the scaling law counts non-embedding weights only. Embeddings are counted once, outside $N$.

## Turn parameters and tokens into flops

A flop is one floating-point multiply or add. A gradient is the change in a result when you nudge one input. The forward pass does one multiply and one add per weight per token, which is $2N$. The backward pass does two more matrix multiplies of that size, one for the weights and one for the inputs, which is $4N$. Training on $D$ tokens costs

$$
C = 6ND
$$

The six is two forward flops plus four backward flops. The formula skips attention scores, which stay small when the model is wide next to its sequence length. For $N = 10^9$ and $D = 2 \times 10^{10}$, $C = 1.2 \times 10^{20}$.

Use floats. The product $6 \times 7 \times 10^9 \times 3 \times 10^{11}$ is about $1.3 \times 10^{22}$, and a 64-bit integer overflows. The usual mistake multiplies Python ints at that scale.

## Split one compute budget

Fix $C$ with $6ND = C$. At the best split, the last unit of compute buys the same loss drop on either side. Two helpers set that split:

$$
a = \frac{\beta}{\alpha + \beta}, \qquad
G = \left( \frac{\alpha A}{\beta B} \right)^{1/(\alpha + \beta)}
$$

$a$ is the share of each extra order of magnitude of compute that goes to size. $G$ depends only on the fit. The best size and token count are

$$
N^{*} = G \cdot 6^{-a} \cdot C^{a}, \qquad D^{*} = \frac{C}{6 N^{*}}
$$

$N^{*}$ grows as $C^{a}$, and $D^{*}$ spends what is left. Equal exponents give $a = 0.5$.

For $C = 10^{20}$, $a = 0.5$ and $G = (400 / 1100)^{1/0.68} \approx 0.2259$. Then $6^{-0.5} \approx 0.4082$, so $N^{*} \approx 0.2259 \times 0.4082 \times 10^{10} = 9.22 \times 10^{8}$ and $D^{*} \approx 1.81 \times 10^{10}$. The ratio is about 19.6 at every budget, about 20 tokens per parameter. The arithmetic is:

```python
a = beta / (alpha + beta)
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
n_star = G * 6.0 ** (-a) * compute ** a
```

The usual mistake drops $6^{-a}$. The budget still adds up, because $D^{*}$ is computed from $N^{*}$, and the ratio drifts away from about 20.

> [!KEY]
> For a fixed budget, the best model size grows as $C^{a}$. Equal exponents mean about 20 training tokens per parameter at every budget.

## Read two different fits

Kaplan and colleagues trained each size on a fixed schedule. Their fit spent most extra compute on a bigger model. Hoffmann and colleagues, in the Chinchilla study, varied size and token count separately. The exponents came out nearly equal, so the best split is even.

Many large models of that period, GPT-3 among them, were undertrained. A model a quarter the size, on four times the data, reaches the same loss for the same compute. A fitted curve is only as good as the experiments behind it.

That split minimises training compute. One generated token costs about $2N$ flops, a bill that grows with $N$ and ignores $D$. A heavily used model is often trained past 20 tokens per parameter, with a smaller $N$ and a larger $D$.

## Fit the power law on a log-log line

Take $x = [1, 2, 4]$ and $y = [2, 8, 32]$, which is $y = 2x^{2}$. The logs are $u = [0, 0.693, 1.386]$ and $v = [0.693, 2.079, 3.466]$, with averages 0.693 and 2.079. Centred, $u$ is $[-0.693, 0, 0.693]$ and $v$ is $[-1.386, 0, 1.386]$. The numerator is $0.961 + 0.961 = 1.922$. The denominator is $0.480 + 0.480 = 0.961$. The slope is $2$, and $c = \exp(2.079 - 2 \times 0.693) = 2$.

Logs turn the curve into a line:

$$
\log y = \log c + k \log x
$$

The slope is the exponent $k$ and the intercept is $\log c$. The least-squares slope is

$$
k = \frac{\sum_i (u_i - \bar{u})(v_i - \bar{v})}{\sum_i (u_i - \bar{u})^2},
\qquad c = \exp(\bar{v} - k \bar{u})
$$

Here $u_i = \log x_i$ and $v_i = \log y_i$. Centre both logs, divide their summed product by the summed squares of centred $u$, and set $c$ to $e$ raised to the intercept. The three points give $k = 2$ and $c = 2$ again. The slope has this shape:

```python
u, v = np.log(x), np.log(y)
uc = u - u.mean()
k = np.sum(uc * (v - v.mean())) / np.sum(uc**2)
```

> [!WARNING]
> The intercept of the line is $\log c$. Return $c$ from `np.exp` of that intercept.

The usual mistake returns the intercept itself.

> [!TIP]
> Test the fit on data you built, such as `y = 3 * x**1.7`. It returns $k = 1.7$ and $c = 3$.

> [!RECAP]
> Keep these five facts in front of you while you code:
>
> - Loss is $E + A / N^{\alpha} + B / D^{\beta}$, from `(E, A, B, alpha, beta)` in that order.
> - $N$ is `layers` times the attention and feed-forward weights. Embeddings count once.
> - Training compute is $C = 6ND$, computed in floats.
> - A power-law fit is a least-squares line on $(\log x, \log y)$, and $c$ is $e$ to the intercept.
> - The best split is $N^{*} = G \cdot 6^{-a} \cdot C^{a}$ and $D^{*} = C / (6 N^{*})$.

## Build it step by step

The checks call your functions in this order:

1. `predict_loss(N, D, coeffs)` returns a float from `(E, A, B, alpha, beta)`. The usual mistake is the tuple order. Check 2.3844. The expression has this shape:

```python
def predict_loss(N, D, coeffs):
    # E + A / N**alpha + B / D**beta
    return loss
```

2. `training_flops(params, tokens)` returns `6 * params * tokens` as a float. Call `float()` first. The usual mistake overflows an integer. The product has this shape:

```python
def training_flops(params, tokens):
    # 6.0 * float(params) * float(tokens)
    return flops
```

3. `transformer_params(layers, d_model, d_ff, vocab, seq_len)` returns integers under `"attention"`, `"feedforward"`, `"embedding"`, `"non_embedding"`, and `"total"`. The usual mistake scales embeddings by `layers`. On the toy sizes, non-embedding weights are 1536. The dict has this shape:

```python
def transformer_params(layers, d_model, d_ff, vocab, seq_len):
    # attention, feedforward, embedding, non_embedding, total
    return breakdown
```

4. `fit_power_law(x, y)` takes positive arrays of shape `(n,)` and returns `(k, c)`. Use `np.log`, `.mean()`, `np.sum`, and `np.exp`. The usual mistake returns $\log c$. The slope has this shape:

```python
def fit_power_law(x, y):
    # centre the logs, then slope and exp(intercept)
    return k, c
```

5. `chinchilla_optimal(compute)` returns `(n_star, d_star)` from `CHINCHILLA_COEFFS`. Compute `a`, `G`, $N^{*}$, then $D^{*}$ from $C = 6ND$. Use the closed form. The usual mistake drops $6^{-a}$. At $C = 10^{20}$, $N^{*}$ is about $9.22 \times 10^{8}$. The rule has this shape:

```python
def chinchilla_optimal(compute):
    # a, G, N*, then C / (6 N*)
    return n_star, d_star
```

## Cost and scaling

One accelerator here sustains 200 trillion flops per second. The best split and the time line up as follows:

| Compute $C$ (flops) | $N^{*}$ | $D^{*}$ (tokens) | Time on one accelerator |
|---|---|---|---|
| $10^{18}$ | 92 million | 1.8 billion | 1.4 hours |
| $10^{20}$ | 922 million | 18 billion | 5.8 days |
| $10^{22}$ | 9.2 billion | 181 billion | 1.6 years |
| $10^{24}$ | 92 billion | 1.8 trillion | 159 years |

The last row is why a large run uses thousands of accelerators. Adam in mixed precision needs about 16 bytes per parameter, so 92 billion parameters need about 1.5 TB.

## Where this breaks

$D^{*}$ grows with compute faster than the supply of distinct text. Repeated data then gives less than the law promises.

Small models are noisy, and a small error in an exponent grows across a long extrapolation. Fit several seeds over a wide range of sizes.

Another data mix, tokeniser, or architecture can move all five constants. The training split also ignores serving cost, so a heavily used model is trained smaller and longer. The law predicts pretraining loss. A task score can jump while that loss falls smoothly.

## The exercise

`predict_loss(N, D, coeffs)` evaluates the loss. `training_flops(params, tokens)` returns $6ND$. `transformer_params(layers, d_model, d_ff, vocab, seq_len)` splits the parameter count. `fit_power_law(x, y)` returns the exponent and coefficient of a log-space line. `chinchilla_optimal(compute)` returns $(N^{*}, D^{*})$ from `CHINCHILLA_COEFFS`.

The checks cover the loss, including that it falls as $N$ or $D$ grows, the compute formula, and a hand count of parameters. Doubling `d_model` and `d_ff` quadruples the non-embedding count. `fit_power_law` recovers a planted exponent, with noise and without it. `chinchilla_optimal` spends the whole budget at about 20 tokens per parameter.
