---
title: Tokenisation
slug: 28-tokenisation
part: "Part 6 — Large language models"
summary: Byte-pair encoding, traced by hand, against the vocabulary-length trade it exists to solve.
minutes: 45
gpu: false
objectives:
  - Explain the vocabulary-size against sequence-length trade, and why byte-level tokenisation removes out-of-vocabulary tokens entirely.
  - Trace one full run of byte-pair encoding by hand, and explain why applying the same merges out of order changes the result.
  - Contrast WordPiece's likelihood-based merge criterion and Unigram's pruning direction with byte-pair encoding's frequency-greedy one.
  - Compute a tokeniser's fertility and explain why it rises for scripts and data underrepresented in the training corpus.
prereqs:
  - 27-training-a-transformer
exercise: 28-tokenisation
---

# Tokenisation

Every lesson so far has taken the input sequence as given: a row of integer
token IDs, ready for an embedding lookup. Deciding what those integers stand
for is a separate problem, solved once ahead of any training run. It sets
two numbers every later lesson's flop and parameter counts depend on: $V$,
the vocabulary size, and $T$, how many tokens a stretch of text turns into.
Choosing the units badly compounds through the whole of training.

The two obvious choices fail in opposite directions. Split on characters and
$V$ stays small, but $T$ grows to the character count of the text. Lesson
27's $6ND$ estimate, together with attention's quadratic term in $T$, then
make every character an expensive way to move one bit of information through
the network. Split on words instead and $T$ shrinks, but $V$ becomes the size of a
language's vocabulary, open-ended by construction. A word absent from a fixed
vocabulary at training time has no representation at inference time, and the
standard fix, a reserved "unknown" token, throws the word away rather than
represent it. Byte-pair encoding, and the family of subword schemes around
it, is the compromise that keeps $V$ and $T$ both moderate. It is chosen
directly from a training corpus's statistics, rather than from a linguistic
unit that turns out to be the wrong size for either number.

## What you already know

- Attention's score computation costs $O(T)$ flops per token per head, so the
  full sequence costs $O(T^2)$ (lesson 25, lesson 26).
- The embedding table and the (possibly tied) output projection each cost
  $V \times d_{\text{model}}$ parameters, and the output projection costs
  $2 V d_{\text{model}}$ forward flops per token (lesson 26).
- The causal language-modelling loss is a sum of per-token cross-entropy
  terms over a fixed vocabulary of size $V$ (lesson 27).
- UTF-8 encodes every Unicode code point as one to four bytes, and every byte
  string is a valid sequence of values in $0, \dots, 255$.
- A greedy algorithm commits to the locally best choice at each step and
  never revisits it.

## The vocabulary-length trade

Fix a piece of text and compare three ways to tokenise it. Character-level
tokenisation sets $T$ to the character count and $V$ to a few hundred, so the
model pays the quadratic attention term and the linear $6ND$ term at their
largest for text this short in information content. Word-level tokenisation
sets $T$ to the word count, far smaller. But $V$ must cover a language's
practical vocabulary — every inflection, proper noun, and compound — which
pushes past $10^5$. Every token the model has never seen at training time
then becomes unrepresentable at inference time. Between the two, the output
projection alone costs $2 V d_{\text{model}}$ flops per token, so shrinking
$T$ by choosing bigger units buys back attention's quadratic cost only by
growing a term that scales linearly in every token regardless of sequence
length. There is no unit size that minimises both at once; there is a unit
size, chosen from data, that keeps both within reach — typically $V$ in the
tens of thousands and $T$ a small multiple of the word count.

**Byte-level tokenisation** resolves the out-of-vocabulary problem
independently of that trade, by fixing the base vocabulary to the 256
possible byte values before any merging happens. Every input, in every
script, is some sequence of bytes, so it decomposes into the base vocabulary
with no remainder and no reserved unknown token. Whatever a byte-level scheme
adds on top of those 256 symbols shortens $T$; it never makes more of the
input representable, because the input was always representable.

## Byte-pair encoding

BPE builds a vocabulary bottom-up from the 256 byte values by repeatedly
merging the most frequent adjacent pair of symbols into a new symbol.

**Training.** Split the training corpus into words with a fixed
pre-tokeniser (below), and represent each word as the list of byte values of
its UTF-8 encoding. At each step, count every adjacent pair of symbols across
every word, and take the pair with the highest count. Assign it a new
symbol ID, then rewrite every word by replacing every non-overlapping
occurrence of that pair with the new ID, scanning left to right. Repeat until the vocabulary
reaches the target size. Counting every pair from scratch costs $O(n)$ for a
corpus of $n$ symbols, done once per merge, so training a vocabulary of $V$
symbols this way costs $O(|V| \cdot n)$. A priority queue that tracks each
pair's count and updates only the pairs adjacent to each merge turns that
cost into work proportional to the number of merge occurrences, the standard
approach once a corpus is large.

**The model is the merge list.** Nothing about the training corpus matters
after training finishes. The tokeniser's entire state is the ordered list of
merges, plus the resulting vocabulary mapping each ID to its byte string.
Encoding never recomputes frequencies; it only ever asks, for a fresh piece
of text, which of these merges apply, in the order they were learned.

**Encoding is order-dependent, not longest-match.** Encode a word by starting
from its raw bytes and applying the merge list in the order it was trained,
each merge scanning left to right and replacing every non-overlapping
occurrence of its pair. A later merge can combine two symbols only if they
already exist in the sequence, which, for a merge that produces a symbol a
later merge uses, means the earlier merge has to fire first. Apply the same
merges in a different order and a pair that depended on an earlier merge may
never appear, so some merges silently fail to fire. The result is still a
valid tokenisation using symbols from the same vocabulary, but a worse one,
with more tokens than the correctly ordered encoding produces. This is also why
encoding is not a matter of greedily matching the longest available
vocabulary substring at each position. Some substrings are meant to appear
only after specific shorter ones combine first, and matching by length alone
ignores that dependency.

## Alternatives to raw frequency

**WordPiece** replaces raw pair frequency with a likelihood-based score,
$\operatorname{count}(a, b) \,/\, \big(\operatorname{count}(a) \cdot
\operatorname{count}(b)\big)$, the ratio of how often the pair appears
together to how often its two halves would appear together if independent. A
pair that occurs constantly only because both halves are themselves extremely
common scores low under this criterion, even at a high raw count. This
steers merges away from combinations frequent merely because their parts are
frequent, and toward ones that co-occur more than chance predicts.

**Unigram** builds in the opposite direction. It starts from a large
candidate vocabulary and a probabilistic unigram model over it, then
alternates re-estimating token probabilities by expectation-maximisation with
removing whichever tokens cost the corpus's total likelihood the least,
pruning down to the target size. Where BPE constructs a small vocabulary
upward one merge at a time, Unigram prunes a large one downward one removal
at a time. Because the removal criterion is a token's contribution to a
likelihood model rather than its raw frequency, the surviving vocabulary need
not correspond to any sequence of pairwise merges.

## Pre-tokenisation and the leading-space convention

Running BPE on a raw, unsplit byte stream lets merges cross word boundaries
freely, wasting vocabulary slots on combinations like a word's final letter
and the following space. Production tokenisers pre-split the corpus,
typically on whitespace and punctuation boundaries, before training, so
merges only ever combine symbols within one token span. They apply the
identical split at encoding time, so the same word decomposes the same way
regardless of what precedes it.

A naive split discards the whitespace itself: a word after one space and the
same word after two spaces or a newline would tokenise identically, making
exact reconstruction of the original text impossible. The fix, standard since
GPT-2, attaches a single leading space to the word that follows it as part of
that word's span, and treats any run of two or more whitespace characters as
its own span. Every character of the input belongs to exactly one span this
way, so concatenating the spans reproduces the input exactly, and a word's
tokenisation no longer depends on whether it opens a sentence or follows a
single space mid-line.

## Cost and scaling

For a fixed piece of text, the unit choice trades $V$ against $T$ directly.

| Scheme | Vocabulary $V$ | Sequence length $T$ | Per-token output-projection flops |
|---|---|---|---|
| Character-level | Hundreds | Largest (one per character) | $2 V d_{\text{model}}$, smallest $V$ |
| Byte-pair encoding | Tens of thousands | A small multiple of the word count | $2 V d_{\text{model}}$, moderate $V$ |
| Word-level | Unbounded, in practice capped | Smallest (one per word) | $2 V d_{\text{model}}$, largest $V$, plus unrepresentable words |

**Fertility**, the average number of tokens a tokeniser produces per word,
makes the length side of the trade measurable independently of any model.
Training BPE to a larger vocabulary strictly increases the number of merges
available. A word that split into several byte-level fragments at a small
vocabulary size increasingly collapses into one or two tokens as the
vocabulary grows past the frequency threshold those substrings need to merge.
Fertility on a fixed evaluation text falls monotonically, in practice, as
vocabulary size grows, with diminishing returns once most common words
already merge to a single token.

## Where this breaks

**Digits.** Whether "1234" merges as one token, as "12" and "34", or as four
separate digits depends on how those digit groupings happened to occur in the
training corpus, not on their numeric value. The same number can therefore
tokenise differently depending on its surroundings. A model trained on inconsistently
chunked numbers has no uniform representation of place value, a large part of
why arithmetic is hard for a transformer to learn from ordinary text; some
tokenisers force single-digit splitting specifically to remove this
inconsistency.

**Code indentation.** Whether a run of four or eight leading spaces merges
into one token depends on how often exactly that many consecutive spaces
occurred in training. Source code with inconsistent indentation, or
indentation depths past what the training corpus favoured, pays for every
additional space as a separate token, inflating the token count of deeply
nested code past what its information content would suggest.

**Non-Latin scripts.** A vocabulary trained on a corpus dominated by English
text spends most of its merge budget on English morphemes. Text in a script
whose characters need two to four bytes each under UTF-8, common outside the
Latin alphabet, starts every word at several bytes per character before any
merge. If that script is underrepresented in training, few of its byte
sequences occur often enough to merge. It ends up facing the character-level
tokeniser's cost, in the same model that treats English near the word level.

## The exercise

You build byte-level BPE from its two primitives up through training and
encoding. Implement `get_pair_counts(token_lists)`, counting every adjacent
pair across a list of token-ID lists, and `merge_pair(token_lists, pair,
new_id)`, rewriting every occurrence of that pair. Build `train_bpe(corpus,
vocab_size)` from them, returning the ordered list of merges and the
resulting vocabulary, where the symbol a merge produces is always $256$ plus
that merge's position in the list. Then write `encode(text, merges)`,
applying the merges in order to a fresh string, `decode(ids, vocab)`,
reversing it, and `fertility(text, merges)`, the average tokens per word.

The checks verify `get_pair_counts` and `merge_pair` on small hand-checkable
inputs, and that `encode` and `decode` round-trip text holding non-ASCII
bytes both before and after training. They verify that `train_bpe` on a
short, classic example reproduces the exact three merges you can trace by
hand, and that the returned vocabulary is exactly the requested size. They
also verify that encoding the same merges out of order gives a different
tokenisation with strictly more tokens than the correctly ordered one, and
that fertility on a fixed text falls as the trained vocabulary grows.
