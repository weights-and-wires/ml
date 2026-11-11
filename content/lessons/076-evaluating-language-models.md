---
title: "Evaluating language models: bits per byte, pass@k, and honest error bars"
slug: 076-evaluating-language-models
part: "Part 7 — Large language models"
summary: Compare language models fairly across tokenisers, score tests and code, put error bars on the result, and catch test items the model has already seen.
minutes: 75
gpu: false
objectives:
  - Explain why per-token perplexity cannot compare two models with different tokenisers, and compute bits per byte instead.
  - Score a multiple-choice question by log-likelihood, with and without length normalisation, and say when they disagree.
  - Compute the unbiased pass@k estimate from n samples and c correct ones, without overflow.
  - Put a bootstrap confidence interval on a benchmark score, and compare two models on the same items with a paired bootstrap.
  - Flag test items that appear in the training data by n-gram overlap.
prereqs:
  - 068-next-token-loss-and-safe-steps
  - 071-tokenisation
  - 075-decoding-and-sampling
exercise: 076-evaluating-language-models
---

# Evaluating language models: bits per byte, pass@k, and honest error bars

Two labs each train a language model and report one number. Lab A says its model has a perplexity of 16 on a page of text. Lab B says its model has a perplexity of 4 on the same page. Lab B's model looks four times better. It is not. Lab B's tokeniser cuts the page into twice as many pieces, and a model that predicts small pieces finds each one easier. The two numbers measure different things.

That is the first of three traps in this chapter. The second is a benchmark score reported with no error bars: model X gets 60.0 percent, model Y gets 55.5 percent, and the headline says X wins. Is a 4.5-point gap real, or would a different draw of 200 questions have flipped it? The third is a test that the model read during training. Its score then measures memory, not skill.

This chapter gives you the tools to avoid all three. You compare models on a scale that does not depend on the tokeniser, score tests and code in a way that stays honest, put an error bar on the result, and check the test set for leaks.

## What you already know

- A language model gives a probability to the true next token, and the loss is minus the log of that probability, averaged over positions. Perplexity is $e^{L}$, as in [next-token loss](/lessons/068-next-token-loss-and-safe-steps).
- A tokeniser turns text into bytes and then into tokens, and different tokenisers give different token counts for the same text, as in [tokenisation](/lessons/071-tokenisation).
- A model can write many different answers to one prompt, depending on the temperature and the seed, as in [decoding and sampling](/lessons/075-decoding-and-sampling).
- An average over a finite sample is a noisy estimate of the average over all possible items.

## Why perplexity cannot compare tokenisers

Start from a picture. Take one text of 40 bytes. Tokeniser A cuts it into 10 tokens of about 4 bytes each. Tokeniser B cuts the same text into 20 tokens of about 2 bytes each. Suppose two models read the text and assign it the same total probability. They are exactly as good at predicting it. The model that assigned the probability piece by piece with fewer, larger pieces had to pay for 4 bytes of surprise in each step. The other paid for 2 bytes in each step.

So the per-token loss is the total loss divided by the number of tokens, and the number of tokens belongs to the tokeniser, not to the model. Work it through with real numbers. Say the total loss over the text is 27.73 nats, which is $40 \ln 2$. A nat is the unit of loss when you use the natural logarithm.

| Tokeniser | Tokens | Loss per token | Perplexity |
|---|---|---|---|
| A | 10 | $27.73 / 10 = 2.773$ | $e^{2.773} = 16$ |
| B | 20 | $27.73 / 20 = 1.386$ | $e^{1.386} = 4$ |

Same text, same total probability, and a fourfold gap in perplexity. Doubling the token count halves the loss per token, and perplexity, the exponential of that loss, falls to its square root: 16 becomes 4.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Two bar charts for one 40-byte text. Perplexity per token is 16 for tokeniser A and 4 for tokeniser B. Bits per byte is 1.0 for both.">
  <text x="150" y="30" text-anchor="middle" class="fig-bold">Perplexity per token</text>
  <text x="450" y="30" text-anchor="middle" class="fig-bold">Bits per byte</text>
  <line x1="50" y1="190" x2="250" y2="190" class="stroke-muted"/>
  <line x1="350" y1="190" x2="550" y2="190" class="stroke-muted"/>
  <rect x="80" y="62" width="60" height="128" class="fill-a"/>
  <rect x="170" y="158" width="60" height="32" class="fill-b"/>
  <rect x="380" y="62" width="60" height="128" class="fill-a"/>
  <rect x="460" y="62" width="60" height="128" class="fill-b"/>
  <text x="110" y="54" text-anchor="middle" class="fig-mono">16</text>
  <text x="200" y="150" text-anchor="middle" class="fig-mono">4</text>
  <text x="410" y="54" text-anchor="middle" class="fig-mono">1.0</text>
  <text x="490" y="54" text-anchor="middle" class="fig-mono">1.0</text>
  <text x="110" y="210" text-anchor="middle" class="fig-small">A, 10 tokens</text>
  <text x="200" y="210" text-anchor="middle" class="fig-small">B, 20 tokens</text>
  <text x="410" y="210" text-anchor="middle" class="fig-small">A, 10 tokens</text>
  <text x="490" y="210" text-anchor="middle" class="fig-small">B, 20 tokens</text>
</svg>
Figure 1. One text, one total probability, two tokenisers. Perplexity per token splits by a factor of four because the token counts differ, while bits per byte is the same for both.
```

The fix is to divide by something that does not depend on the tokeniser. Every tokeniser in the course starts from bytes, and the byte count of a text is a fact about the text. Divide the total loss by the number of bytes:

$$
\text{bpb} = \frac{S}{N_{\text{bytes}} \ln 2}
$$

Read it aloud: $S$ is the total loss in nats, summed over every token of the text, and $N_{\text{bytes}}$ is the number of bytes of the text. Dividing by $\ln 2$ turns nats into **bits**, the unit you get from a base-2 logarithm. The result is **bits per byte**: how many bits of surprise the model pays for each byte it reads. A model that spreads probability evenly over all 256 byte values pays $\log_2 256 = 8$ bits per byte, so 8 is the score of knowing nothing. In the example, both tokenisers give $27.73 / (40 \times 0.6931) = 1.0$.

```python
def bits_per_byte(loss_nats_sum, n_bytes):
    return loss_nats_sum / (n_bytes * np.log(2))
```

> [!KEY]
> Per-token loss divides by a count the tokeniser chose. Bits per byte divides the total loss by the bytes of the text, so a model that gives the text the same probability scores the same under any tokeniser.

The Pile paper reports bits per byte for this reason (see Further reading). Two honest caveats apply. First, the total loss must be the loss of the whole text under the model's own tokenisation, with no tokens dropped, such as the first token of each window. Second, a model scores best under the tokenisation it was trained on. A byte count compares models fairly. It does not turn a poorly matched tokenisation into a good one.

> [!WARNING]
> Never divide the total loss by the token count and call the result bits per byte. That mistake keeps the tokeniser in the denominator, so the number still moves when you change the tokeniser and nothing else.

## Scoring a multiple-choice test by log-likelihood

Many benchmarks are multiple choice: a question and four answer strings. A base model does not write "B". Instead, you ask the model how likely each answer string is after the question, and you pick the likeliest. For one option, add the log probability of each of its tokens given the question and the tokens before it. This is the same sum you used for the next-token loss, with the sign flipped.

The sum has a flaw. Every extra token multiplies in another probability below 1, so the log-likelihood falls with length. A long, correct answer can lose to a short, wrong one. Take three options with total log probabilities of $-6.0$, $-9.0$, and $-12.0$ nats, and byte lengths of 3, 6, and 12. The raw score picks the first option, the shortest. Divide each total by its length and you get $-2.0$, $-1.5$, and $-1.0$ nats per byte, and the longest option wins. The two rules disagree on every item like this, and which one is right depends on the benchmark.

The two rules in numpy:

```python
scores = logprobs / lengths if normalise else logprobs
choice = int(np.argmax(scores))
```

Normalising by bytes and not tokens follows the same logic as bits per byte: token counts change with the tokeniser, and byte counts do not. Normalising has its own failure. It lets a long answer of filler win because its average cost per byte is low. Report both numbers, say which the benchmark's authors defined, and don't compare a normalised score with a raw one.

## Scoring code with pass@k

A coding benchmark gives a function description and runs the model's code against unit tests. One sample either passes or it does not. Because the model can write many different programs, a fairer question is: if you let the model try $k$ times, how often does at least one try pass? That rate is **pass@k**.

The obvious way to measure it is to draw $k$ samples and see whether any pass. That is noisy, because each problem gets one draw. A better way draws $n$ samples per problem, with $n$ larger than $k$, counts the $c$ that pass, and reuses them. Two tempting formulas exist, and only one is right.

Suppose $n = 5$ samples and $c = 2$ passed. You want pass@2. The first guess treats each draw as independent with success rate $c/n = 0.4$. Then the chance that at least one of 2 draws passes is $1 - (1 - 0.4)^2 = 0.64$. That formula samples with replacement: it assumes that you can draw the same sample twice. You can't. The $k$ attempts you are modelling are $k$ different programs out of the $n$ you wrote.

The right question is: pick $k$ of the $n$ samples at random, without replacement. What is the chance that all of them fail? There are $\binom{n}{k}$ ways to pick $k$ samples, and $\binom{n-c}{k}$ of them use only failures. Here that gives $\binom{3}{2} / \binom{5}{2} = 3/10$, so the chance of at least one pass is $1 - 0.3 = 0.7$.

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

Read it aloud: one minus the fraction of size-$k$ subsets that contain only failures. The with-replacement guess gave 0.64 against the true 0.7, so it underestimates. A simulation agrees with the combinatorial formula: draw 2 of 5 without replacement 100,000 times, and the share with at least one pass is 0.701.

```figure
<svg viewBox="0 0 600 150" role="img" aria-label="Five samples in a row, two marked correct. Ten ways to pick two of them, of which three pick only wrong samples.">
  <circle cx="140" cy="50" r="18" class="fill-b"/>
  <circle cx="220" cy="50" r="18" class="fill-b"/>
  <circle cx="300" cy="50" r="18" class="fill-a"/>
  <circle cx="380" cy="50" r="18" class="fill-a"/>
  <circle cx="460" cy="50" r="18" class="fill-a"/>
  <text x="140" y="55" text-anchor="middle" class="fig-bold fill-paper">ok</text>
  <text x="220" y="55" text-anchor="middle" class="fig-bold fill-paper">ok</text>
  <text x="300" y="55" text-anchor="middle" class="fig-bold fill-paper">x</text>
  <text x="380" y="55" text-anchor="middle" class="fig-bold fill-paper">x</text>
  <text x="460" y="55" text-anchor="middle" class="fig-bold fill-paper">x</text>
  <text x="300" y="105" text-anchor="middle">pick 2 of 5: 10 ways, 3 are all-wrong (x x)</text>
  <text x="300" y="130" text-anchor="middle">pass@2 = 1 - 3/10 = 0.70</text>
</svg>
Figure 2. Five samples, two correct. Of the ten ways to pick two, only the three that pick two wrong samples fail, so pass@2 is 0.70, not the 0.64 you get by drawing with replacement.
```

The binomial coefficients cause trouble in code. With $n = 5000$ samples, $\binom{n}{k}$ has thousands of digits and overflows a float. Cancel the common factors instead. The ratio equals a product of $c$ factors:

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}} = \prod_{i=n-c+1}^{n} \left(1 - \frac{k}{i}\right)
$$

Read it aloud: multiply $c$ factors, one for each integer from $n-c+1$ up to $n$, each slightly below 1. Check it on the example: $(1 - 2/4)(1 - 2/5) = 0.5 \times 0.6 = 0.3$. Each factor is between 0 and 1, so nothing overflows. When $n - c < k$, the range includes $i = k$, the factor is exactly 0, and pass@k is 1, which is right because every group of $k$ must hold a pass.

```python
i = np.arange(n - c + 1, n + 1)
pass_k = 1.0 - np.prod(1.0 - k / i)
```

> [!KEY]
> pass@k asks what happens when you draw $k$ different samples from the $n$ you have. It is one minus the share of $k$-subsets with no pass, and the product form computes it with no large numbers. For $k = 1$ it equals $c/n$.

## How much to trust a score: the bootstrap

A benchmark of 200 questions gives a score of 0.60. Another draw of 200 questions from the same source would give a different score, because which questions you got is partly luck. You want a range of plausible scores. The textbook route is a formula for the standard error, but formulas exist for simple scores like accuracy and not for scores like F1 or a mean of ratios. The **bootstrap** works for any score with the same recipe.

Treat your 200 items as a stand-in for the population they came from. Draw 200 items from them with replacement, so some items appear twice and some not at all. Compute the score on that resample. Do it $B$ times, say 1000. The spread of those $B$ scores shows how much the score moves when the items change. The 2.5th and 97.5th percentiles of the resampled scores give a 95 percent **confidence interval**: a range built so that, over many repeats of the whole experiment, it contains the true value about 95 percent of the time.

```python
idx = rng.integers(0, n, size=(B, n))        # B resamples of item indices
means = scores[idx].mean(axis=1)             # one score per resample
lo, hi = np.percentile(means, [2.5, 97.5])
```

Resample items, not tokens or characters, because the items are the units that you drew independently. Seed the generator and pass it in, so a rerun gives the same interval.

Two conditions keep this honest. The items must be independent draws, so 200 paraphrases of 20 questions are not 200 items. And $n$ must be large enough that the resamples look like the population. With 20 items the percentile interval is too narrow, and its coverage falls well below 95 percent.

### Comparing two models on the same items

Models X and Y are scored on the same 200 items. X gets 120 right (0.600), and Y gets 111 right (0.555). Bootstrap each score alone and you get intervals of 0.530 to 0.665 and 0.485 to 0.625. They overlap heavily, so you might conclude that nothing separates the models.

That conclusion throws away information. The two models were tested on the same items, so the easy items are easy for both and the hard items are hard for both. In the example, Y matches X on 185 items. X is right and Y is wrong on 12 items, and Y is right and X is wrong on 3. The gap of $-0.045$ comes from only 15 items that disagree. A **paired bootstrap** keeps that structure: draw item indices once, and use the same indices to score both models, then take the difference of the two scores on each resample.

```python
idx = rng.integers(0, n, size=(B, n))
diffs = scores_b[idx].mean(axis=1) - scores_a[idx].mean(axis=1)
```

The 95 percent interval for the difference is $-0.085$ to $-0.010$, which excludes zero. Resample the two models separately and ignore the pairing, and the interval is $-0.140$ to $0.050$, which includes zero. The same data gives opposite conclusions, and the paired one uses the data's structure.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Confidence intervals on one axis from -0.16 to 0.70. Model X accuracy 0.53 to 0.665, model Y 0.485 to 0.625, the paired difference -0.085 to -0.01, which excludes zero, and the unpaired difference -0.14 to 0.05, which includes zero.">
  <line x1="226.3" y1="40" x2="226.3" y2="225" class="stroke-muted dashed"/>
  <line x1="150" y1="225" x2="560" y2="225" class="stroke-muted"/>
  <text x="226.3" y="245" text-anchor="middle" class="fig-small">0</text>
  <text x="321.6" y="245" text-anchor="middle" class="fig-small">0.2</text>
  <text x="417" y="245" text-anchor="middle" class="fig-small">0.4</text>
  <text x="512.3" y="245" text-anchor="middle" class="fig-small">0.6</text>
  <text x="8" y="74" class="fig-small">X accuracy</text>
  <text x="8" y="114" class="fig-small">Y accuracy</text>
  <text x="8" y="164" class="fig-small">Y minus X, paired</text>
  <text x="8" y="204" class="fig-small">Y minus X, unpaired</text>
  <line x1="478.9" y1="70" x2="543.3" y2="70" class="stroke-a"/>
  <circle cx="512.3" cy="70" r="5" class="fill-a"/>
  <line x1="457.5" y1="110" x2="524.2" y2="110" class="stroke-a"/>
  <circle cx="490.8" cy="110" r="5" class="fill-a"/>
  <line x1="185.8" y1="160" x2="221.5" y2="160" class="stroke-b"/>
  <circle cx="204.8" cy="160" r="5" class="fill-b"/>
  <line x1="159.5" y1="200" x2="250.1" y2="200" class="stroke-bad"/>
  <circle cx="204.8" cy="200" r="5" class="fill-bad"/>
</svg>
Figure 3. The same 200 items, two ways. The accuracy intervals of X and Y overlap, and the unpaired interval for the gap straddles zero. The paired interval for the gap lies entirely below zero, because pairing removes the item difficulty both models share.
```

> [!KEY]
> A bootstrap interval measures how much a score moves when the items change. When two models share the same items, resample the items once and score both models on them. Pairing is a free narrowing of the interval, and it is the fair comparison.

## Catching test items the model has seen

A score also lies when the model read the answers during training. Web-scale training text includes copies of many benchmarks, with questions, answers, and discussion. A model that memorised the test gets a high score for the wrong reason.

The simple detector counts shared n-grams. An **n-gram** is a run of $n$ consecutive tokens. For a test item, list its n-grams, and check which of them appear anywhere in the training documents. The fraction that do is the overlap. A test item whose 13-grams mostly appear in training was almost certainly in the corpus.

Work a small case with $n = 3$. The test text is `a b c d e`, and the training text is `x b c d y`. The test trigrams are `abc`, `bcd`, and `cde`. The training trigrams are `xbc`, `bcd`, and `cdy`. Only `bcd` is shared, so the overlap is $1/3$.

```python
def grams(doc, n):
    return {tuple(doc[i:i + n]) for i in range(len(doc) - n + 1)}

seen = set().union(*[grams(d, n) for d in train_docs])
overlap = sum(g in seen for g in test_grams) / len(test_grams)
```

Pick $n$ large enough that an accidental match is rare. Common word pairs show up everywhere, so a bigram overlap flags harmless text, while a run of ten or more words rarely matches by chance. Flag the items above a threshold, then report the score on the clean items too. If the clean score is much lower, contamination was inflating the headline.

The detector has a blind spot. It finds copies and near-copies, but not a paraphrase of the test, a translation, or an answer key posted without the question. A clean overlap result means no verbatim leak, not no leak.

## One paragraph on judges

Some tasks have no answer key, such as open-ended chat, so a stronger model is asked to grade the answers. This is called an LLM judge, and it has biases you have to measure. A judge shown two answers prefers the first more often than chance says it should, which is position bias, and swapping the order exposes it. A judge also tends to prefer the longer answer, which is length bias. The reward-model and judge side of evaluation belongs to the reinforcement learning course, which covers [evaluating a reward model and a judge](https://rl.ifkash.dev/lessons/055-judge-evaluation), including swap consistency and length-controlled win rates.

## Where this breaks

Bits per byte compares models on text. It does not say which model is more useful, and a lower value on a web-text sample can come with worse reasoning. A benchmark with a few hundred items cannot separate models that differ by a point or two, and the bootstrap tells you so only when you run it. A pass@k estimate from few samples per problem is noisy, so use more samples than the largest $k$ you report. Overlap checks miss paraphrases, and a benchmark that has been public for years is probably in the corpus. People answer these in practice by holding out private test sets, rotating the questions, and reporting more than one metric.

> [!KEY]
> A score is only meaningful when you know the unit, the error bar, and whether the test was seen. Bits per byte fixes the unit across tokenisers, pass@k and normalised log-likelihood say what counts as a hit, the paired bootstrap gives the error bar, and n-gram overlap checks the third.

> [!RECAP]
> - Total loss divided by tokens depends on the tokeniser. Total loss divided by $N_{\text{bytes}} \ln 2$ is bits per byte, which stays the same for any tokenisation that gives the text the same probability.
> - For multiple choice, add each option's token log probabilities. Raw sums favour short options, so also divide by length, and say which you report.
> - pass@k is $1 - \binom{n-c}{k}/\binom{n}{k}$, computed as $1 - \prod_{i=n-c+1}^{n}(1 - k/i)$. Drawing with replacement gives a different, wrong number.
> - A bootstrap resamples items with replacement, and the 2.5th and 97.5th percentiles give a 95 percent interval. For two models on the same items, reuse the same indices for both.
> - Overlap is the fraction of a test item's n-grams found in the training text. A high value flags a leak, and a low value rules out only copies.

## Build it step by step

You write six functions, and the checks call them in this order.

1. **`bits_per_byte(loss_nats_sum, n_bytes)`** takes the total loss in nats and the byte count, and returns a float. Divide by `n_bytes * np.log(2)`. The usual mistake is to divide by the number of tokens, so the tokeniser leaks back into the number.
2. **`choose_answer(logprobs, lengths, normalise)`** takes one total log probability and one length per option, and returns the index of the best option as an `int`. When `normalise` is true, divide each total by its length first. Use `np.argmax`, which breaks ties toward the first option.
3. **`pass_at_k(n, c, k)`** returns a float. Build `i = np.arange(n - c + 1, n + 1)` and return `1.0 - np.prod(1.0 - k / i)`. The usual mistake is `1 - (1 - c / n) ** k`, which draws with replacement and gives 0.64 where the answer is 0.7. Don't call `math.comb` with floats, and don't build the ratio of two huge numbers.
4. **`bootstrap_ci(scores, B, rng, alpha=0.05)`** draws `idx = rng.integers(0, n, size=(B, n))`, takes the mean of `scores[idx]` along axis 1, and returns the `alpha / 2` and `1 - alpha / 2` percentiles as a pair of floats. Take the percentiles of the resampled means, not of the raw scores.
5. **`paired_bootstrap_diff(scores_a, scores_b, B, rng, alpha=0.05)`** returns the observed mean of `scores_b - scores_a` and the interval for it. Draw one index array and apply it to both score arrays. The usual mistake is to draw two index arrays, which discards the pairing and widens the interval.
6. **`ngram_overlap(train_docs, test_doc, n)`** takes documents as lists of tokens and returns the fraction of the test document's n-grams that appear in any training document. Build a set of tuples, because lists can't go in a set. Return 0.0 when the test document has fewer than `n` tokens.

## The exercise

You write `bits_per_byte`, `choose_answer`, `pass_at_k`, `bootstrap_ci`, `paired_bootstrap_diff`, and `ngram_overlap`. The checks confirm that a model that spreads probability evenly over bytes scores 8 bits per byte, and that two tokenisations of one text with the same total loss give the same value. They check that the raw and normalised rules pick different options on the worked example, that `pass_at_k(5, 2, 2)` is 0.7 and not 0.64, that it stays finite for $n = 5000$, and that it matches a simulation. They check that a bootstrap interval covers the true mean about 95 percent of the time over seeded repeats, that the paired interval for planted data excludes zero, and that n-gram overlap flags planted copies and passes clean items.

## Further reading

- For the pass@k estimator and why drawing $n$ samples per problem beats drawing $k$, see the section on evaluation, "Functional correctness", of [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Chen et al., 2021).
- For bits per byte as a tokeniser-independent measure, see the evaluation sections of [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (Gao et al., 2020).
- For the bootstrap, see *An Introduction to the Bootstrap* (Efron and Tibshirani, 1993), whose chapters on bootstrap confidence intervals cover the percentile interval you built.
- For how courses teach this material, see the lecture titled "Benchmarking and Evaluation" on the schedule of [Stanford CS224N](https://web.stanford.edu/class/cs224n/), and Lecture 12, "Evaluation", on the schedule of [Stanford CS336](http://cs336.stanford.edu/spring2025/).
