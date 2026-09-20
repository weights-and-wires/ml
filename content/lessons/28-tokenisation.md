---
title: Tokenisation
slug: 28-tokenisation
part: "Part 6 — Large language models"
summary: Byte-pair encoding derived and hand-traced, against the vocabulary-length trade it exists to solve.
minutes: 50
gpu: false
objectives:
  - Derive the vocabulary-size against sequence-length trade and its direct flop consequence, and explain why byte-level tokenisation removes out-of-vocabulary tokens entirely.
  - Trace one full run of byte-pair encoding by hand, and explain why permuting the same set of merges changes the result at encode time.
  - Contrast WordPiece's likelihood-based merge criterion and Unigram's pruning direction with byte-pair encoding's frequency-greedy one.
  - Compute a tokeniser's fertility and explain why it rises for scripts and data underrepresented in the training corpus.
prereqs:
  - 27-training-a-transformer
exercise: 28-tokenisation
---

# Tokenisation

Every lesson so far has taken the input sequence as given: a row of integer
token IDs, ready for an embedding lookup. Deciding what those integers stand
for is a separate problem, solved once, ahead of any training run, and it
sets two numbers that every later lesson's flop and parameter counts depend
on: $V$, the vocabulary size, and $T$, how many tokens a given stretch of text
turns into. Choose the units badly and the choice compounds through the whole
of training on both terms at once.

The two obvious choices fail in opposite directions. Split on characters and
$V$ stays small — a few hundred symbols cover every script — but $T$ grows to
the character count of the text, and lesson 27's $6ND$ estimate together with
attention's quadratic term in $T$ (lesson 26) make every character an
expensive way to move one bit of information through the network. Split on
words instead and $T$ shrinks, but $V$ becomes the size of a language's
vocabulary, open-ended by construction: any word absent from a fixed
vocabulary at training time has no representation at inference time, and
the standard fix — a reserved "unknown" token — throws the word away rather
than represent it. Byte-pair encoding, and the family of subword schemes
around it, is the compromise that lets $V$ and $T$ both stay moderate, chosen
directly from the statistics of a training corpus rather than from a
linguistic unit that turns out to be the wrong size for either number.

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
tokenisation sets $T$ to the character count and $V$ to a few hundred; the
model pays the quadratic attention term and the linear $6ND$ term at their
largest, for text this short in information content. Word-level tokenisation
sets $T$ to the word count, far smaller, but $V$ must cover a language's
practical vocabulary — every inflection, every proper noun, every compound —
which pushes past $10^5$ before it stops discovering new words, and every
token the model has never seen at training time becomes unrepresentable at
inference time. Between the two, the output projection alone costs
$2 V d_{\text{model}}$ flops per token: shrinking $T$ by choosing bigger units
buys back attention's quadratic cost only by growing $V$, which grows a term
that scales linearly in every token regardless of sequence length. There is no
unit size that minimises both at once; there is a unit size, chosen from data,
that keeps both within reach — typically $V$ in the tens of thousands and $T$
a small multiple of the word count.

**Byte-level tokenisation** resolves the out-of-vocabulary problem
independently of that trade, by fixing the base vocabulary to the 256 possible
byte values before any merging happens. Every input, in every script, is some
sequence of bytes, so every input decomposes into the base vocabulary with no
remainder and no reserved unknown token. Whatever a byte-level scheme adds on
top of those 256 symbols is there purely to shorten $T$, never to make more of
the input representable — the input was always representable.

## Byte-pair encoding

BPE builds a vocabulary bottom-up from the 256 byte values by repeatedly
merging the most frequent adjacent pair of symbols into a new symbol.

**Training.** Split the training corpus into words with a fixed
pre-tokeniser (below), and represent each word as the list of byte values of
its UTF-8 encoding. At each step, count every adjacent pair of symbols across
every word, take the pair with the highest count, assign it a new symbol ID,
and rewrite every word by replacing every non-overlapping occurrence of that
pair with the new ID, scanning left to right. Repeat until the vocabulary
reaches the target size. Counting every pair from scratch at every step costs
$O(n)$ for a corpus of $n$ symbols, done once per merge, so training a
vocabulary of $V$ symbols this way costs $O(|V| \cdot n)$; a priority queue
that tracks each pair's count and updates only the pairs adjacent to each
merge, rather than rescanning the whole corpus, turns the per-step cost from
a full pass into work proportional to the number of merge occurrences, which
is the standard production approach once a corpus is large.

**The model is the merge list.** Nothing about the training corpus matters
after training finishes. The tokeniser's entire state is the ordered list of
merges — pair one merged first, pair two second, and so on — plus the
resulting vocabulary mapping each ID to its byte string. Encoding never
recomputes frequencies; it only ever asks, for a fresh piece of text, which
of these merges apply, in the order they were learned.

**Encoding is order-dependent, not longest-match.** Encode a word by
starting from its raw bytes and applying the merge list in the order it was
trained, each merge scanning left to right through the current sequence and
replacing every non-overlapping occurrence of its pair. A later merge in the
list can combine two symbols only if the symbols it names already exist in
the sequence — which, for a merge that produces a symbol used by a later
merge, means the earlier merge has to have fired first. Apply the same set of
merges in a different order and a pair that depended on an earlier merge
having already run may never appear, so some merges silently fail to fire;
the result is a valid tokenisation of the same text, using symbols from the
same vocabulary, and it is a worse one, with more tokens than the
correctly ordered encoding would produce. This is also why encoding is not a
matter of greedily matching the longest available vocabulary substring at
each position: the vocabulary contains substrings that were only ever meant
to appear after specific shorter ones combined first, and matching by length
alone ignores that dependency entirely.

## Alternatives to raw frequency

**WordPiece** replaces raw pair frequency with a likelihood-based score,
$\operatorname{count}(a, b) \,/\, \big(\operatorname{count}(a) \cdot
\operatorname{count}(b)\big)$, the ratio of how often the pair appears
together to how often its two halves would appear together if they were
independent. A pair that occurs constantly only because both of its halves
are themselves extremely common scores low under this criterion even if its
raw count is high, which steers merges away from combinations that are
frequent merely by coincidence of their parts being frequent, and towards
combinations that co-occur more than chance would predict.

**Unigram** builds in the opposite direction. It starts from a large
candidate vocabulary and a probabilistic unigram model that assigns each
candidate token a probability, then alternates re-estimating those
probabilities by expectation-maximisation with removing whichever tokens cost
the corpus's total likelihood the least, pruning down to the target
vocabulary size. Where BPE constructs a small vocabulary upward one merge at
a time, Unigram prunes a large one downward one removal at a time, and
because the criterion for removal is the token's contribution to a
likelihood model rather than its raw frequency, the vocabulary that survives
need not correspond to any particular sequence of pairwise merges at all.

## Pre-tokenisation and the leading-space convention

Running BPE on a raw, unsplit byte stream lets merges cross word boundaries
freely, which wastes vocabulary slots on combinations like a word's final
letter and the following space. Production tokenisers pre-split the corpus —
typically on whitespace and punctuation boundaries — before training, so
merges only ever combine symbols within what will become one token span, and
apply the identical split at encoding time so the same word always
decomposes the same way regardless of what precedes it.

The naive split discards the whitespace itself, which loses information: a
word after a single space and the same word after two spaces or a newline
would tokenise identically, making exact reconstruction of the original text
impossible. The fix, standard since GPT-2, is to attach a single leading
space to the word that follows it as part of that word's span, and to treat
any run of two or more whitespace characters as its own span. Every character
of the input belongs to exactly one span this way, so concatenating the spans
back together reproduces the input exactly, and a word's tokenisation no
longer depends on whether it opens a sentence or follows a single space
mid-line.

## Cost and scaling

For a fixed piece of text, the unit choice trades $V$ against $T$ directly.

| Scheme | Vocabulary $V$ | Sequence length $T$ | Per-token output-projection flops |
|---|---|---|---|
| Character-level | Hundreds | Largest (one per character) | $2 V d_{\text{model}}$, smallest $V$ |
| Byte-pair encoding | Tens of thousands | A small multiple of the word count | $2 V d_{\text{model}}$, moderate $V$ |
| Word-level | Unbounded, in practice capped | Smallest (one per word) | $2 V d_{\text{model}}$, largest $V$, plus unrepresentable words |

**Fertility** — the average number of tokens a tokeniser produces per word —
makes the length side of the trade measurable independently of any model.
Training BPE to a larger vocabulary strictly increases the number of merges
available, so a word that split into several byte-level fragments at a small
vocabulary size increasingly collapses into one or two tokens as the
vocabulary grows past the frequency threshold that word's substrings need to
merge; fertility on a fixed evaluation text falls monotonically, in
practice, as vocabulary size grows, with diminishing returns once most common
words already merge to a single token.

## Where this breaks

**Digits.** Whether the sequence "1234" merges as one token, as "12" and
"34", or as four separate digits depends on how those particular digit
groupings happened to occur in the training corpus, not on their numeric
value, so the same number can tokenise differently depending on its
surroundings. A model trained on inconsistently chunked numbers has no
uniform representation of place value, which is a large part of why
arithmetic is difficult for a transformer to learn reliably from ordinary
text; some tokenisers force single-digit splitting specifically to remove
this inconsistency.

**Code indentation.** Whether a run of four or eight leading spaces on a
line merges into one token depends on how often exactly that many
consecutive spaces occurred in the training corpus. Source code with
inconsistent indentation styles, or indentation depths past whatever the
tokeniser's training corpus favoured, pays for every additional space as a
separate token, inflating the token count of deeply nested code far past
what its information content would suggest.

**Non-Latin scripts.** A vocabulary trained on a corpus dominated by
English text spends most of its merge budget on English morphemes. Text in a
script whose characters need two, three, or four bytes each under UTF-8 —
common outside the Latin alphabet — starts every word at several bytes per
character before any merge, and if that script is underrepresented in
training, few of its byte sequences ever occur often enough to merge, so it
keeps paying that multi-byte cost per character throughout, in effect facing
the character-level tokeniser's cost from the section above, in the same
model that treats English near the word level.

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
inputs, that `encode` and `decode` round-trip text holding non-ASCII bytes
both before and after training, that `train_bpe` on a short, classic example
reproduces the exact three merges you can trace by hand, that the returned
vocabulary is exactly the requested size, that encoding the same merges out
of order gives a different tokenisation with strictly more tokens than the
correctly ordered one, and that fertility on a fixed text falls as the
trained vocabulary grows.
