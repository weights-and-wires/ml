---
title: Tokenisation
slug: 28-tokenisation
part: "Part 6 — Large language models"
summary: Build a byte-pair encoding tokeniser from scratch, trace its merges by hand, and measure how well it compresses text.
minutes: 60
gpu: false
objectives:
  - Explain the trade between vocabulary size and sequence length, and why starting from bytes means no word is ever unknown.
  - Build byte-pair encoding training from two pieces, a pair counter and a pair merger.
  - Encode and decode text with a learned merge list, and explain why the merges must run in order.
  - Compute a tokeniser's fertility and explain why it is higher for languages the tokeniser saw little of.
prereqs:
  - 27-training-a-transformer
exercise: 28-tokenisation
---

# Tokenisation

A transformer reads integers, not text. Something has to turn "the cat sat"
into a list like `[464, 3797, 3332]`. That something is the tokeniser, and it
decides what each integer stands for: a letter, a word, or something in
between. You build the tokeniser once, before training, and every later
number in the model depends on it.

Two numbers matter most. The vocabulary size $V$ is how many different tokens
exist. The sequence length $T$ is how many tokens a piece of text turns into.
You can't make both small. Tiny units, like single letters, give a small $V$
but a long $T$. Big units, like whole words, give a short $T$ but a huge $V$,
and any word you never saw has no token at all.

This lesson builds byte-pair encoding (BPE), the method most language models
use to land in the middle. It starts from single bytes and glues together the
pairs that show up most often.

## What you already know

- Attention costs grow with the square of the sequence length $T$, from
  lesson 25.
- The embedding table and output layer each hold $V \times d$ weights, from
  lesson 26.
- Training compute is about $6ND$, where $D$ is the number of tokens, from
  lesson 27.
- UTF-8 stores every character as one to four bytes, and each byte is a
  number from 0 to 255.
- A Python `dict` maps keys to values, and a tuple like `(97, 98)` can be a
  key.

## The trade between V and T

Picture the sentence "the cat sat" three ways.

| Unit | Tokens | $T$ | Typical $V$ |
|---|---|---|---|
| Characters | t, h, e, space, c, a, t, ... | 11 | a few hundred |
| Words | the, cat, sat | 3 | over 100,000, and still missing words |
| Subwords (BPE) | the, " cat", " sat" | 3 | tens of thousands |

A long $T$ is expensive, because attention grows with $T^2$ and training
compute grows with the token count. A huge $V$ is expensive too: the output
layer scores every token in the vocabulary, for every position. BPE keeps
both moderate, and it picks its units from real text instead of guessing.

### Start from bytes

A **byte-level** tokeniser starts with exactly 256 base tokens, one for each
possible byte. Any text in any language, even emoji, is some list of bytes. So
nothing is ever unknown. For example, "hé" is the bytes $[104, 195, 169]$:
"h" is one byte and "é" takes two. Merges added on top only make $T$ shorter.

```python
ids = list("hé".encode("utf-8"))   # [104, 195, 169]
```

> [!KEY]
> A byte-level tokeniser starts from 256 base tokens, one per byte, so it can
> encode any text. Merges only make the sequence shorter.

## Byte-pair encoding

Here's the idea in one sentence: find the two neighbouring tokens that appear
together most often, glue them into a new token, and repeat. It's like
noticing you keep writing "t" then "h" and inventing a single symbol for "th".

Each new token gets the next free ID. The first merge makes ID 256, the second
makes 257, and so on. In general:

$$
\text{id}(\text{merge } i) = 256 + i
$$

Read aloud: the merge at position $i$ in the list (counting from 0) produces
token ID 256 plus $i$.

### The two building blocks

**Count pairs.** Walk along each word and count every neighbouring pair.

```python
for a, b in zip(tokens, tokens[1:]):
    counts[(a, b)] = counts.get((a, b), 0) + 1
```

For the words `[1, 2, 3, 1, 2]` and `[1, 2]`, the pairs are $(1,2)$ three
times, $(2,3)$ once, and $(3,1)$ once. Pairs never span two different words.

**Merge a pair.** Walk along each word from the left. If the current token and
the next one match the pair, write the new ID and jump ahead two. Otherwise,
copy the current token and move ahead one.

```python
if i < n - 1 and (tokens[i], tokens[i + 1]) == pair:
    merged.append(new_id); i += 2
else:
    merged.append(tokens[i]); i += 1
```

Merging $(1, 2)$ into 9 turns `[1, 2, 3, 1, 2]` into `[9, 3, 9]`. Merging
$(5, 5)$ into 20 turns `[5, 5, 5]` into `[20, 5]`: the first two merge, and
the third has no partner left.

### Training: a worked example

Train on the text "aaabdaaabac". In bytes, "a" is 97, "b" is 98, "c" is 99,
and "d" is 100.

1. Count pairs. "aa" appears 4 times (twice in each "aaab"). "ab" appears 2
   times. "bd", "da", "ba", and "ac" appear once each. The winner is
   $(97, 97)$. It becomes ID 256, standing for the bytes "aa".
2. Merge. Scanning left to right, "aaab" becomes 256, a, b. The text is now
   256 a b d 256 a b a c.
3. Count again. Both $(256, 97)$ and $(97, 98)$ appear twice. It's a tie.
   Break ties by picking the smaller pair as a tuple, so $(97, 98)$ wins. It
   becomes ID 257, "ab". The text is now 256 257 d 256 257 a c.
4. Count again. $(256, 257)$ appears twice and wins. It becomes ID 258,
   "aaab". The text is now 258 d 258 a c.

> [!TIP]
> `min(counts, key=lambda p: (-counts[p], p))` does both jobs in one call. It
> picks the highest count, and among equal counts it picks the smaller pair.

The merge list is $[(97, 97), (97, 98), (256, 257)]$, and the vocabulary has
$256 + 3 = 259$ tokens. The original 11 bytes became 5 tokens.

> [!KEY]
> Each training round counts pairs, glues the most common one into a new
> token, and gives it the next free ID, $256 + i$.

The vocabulary maps each ID to its bytes. A merged token's bytes are its two
parts joined together:

```python
vocab[new_id] = vocab[a] + vocab[b]   # b"a" + b"a" == b"aa"
```

### The merge list is the tokeniser

After training, you can throw away the training text. All you keep is the
ordered merge list and the vocabulary. You never count pairs again.

### Encoding: run the merges in order

To encode new text, start from its bytes. Apply merge 0 everywhere, then
merge 1 everywhere, and so on, to the end of the list. Then join every word's
tokens into one flat list.

Order matters. Merge 2 in the example glues 256 and 257, and those tokens
only exist after merges 0 and 1 have run. Run the list backwards and merge
$(256, 257)$ fires first, when there is nothing to glue. Encoding
"aaabdaaabac" with the list reversed gives 7 tokens instead of 5. It's still
a valid tokenisation, but a worse one.

> [!KEY]
> To encode, run the merges in the order you learned them. Later merges glue
> tokens that only exist once the earlier merges have run.

For the same reason, encoding isn't "find the longest vocabulary entry that
matches". Some tokens only appear after specific shorter ones join first.

### Decoding

Decoding looks up each ID's bytes, joins them, and turns the bytes back into
text:

```python
text = b"".join(vocab[i] for i in ids).decode("utf-8")
```

> [!WARNING]
> Join all the bytes first, then decode once. One token can hold half of a
> two-byte character, so decoding tokens one at a time can fail.

Because every token stands for exact bytes, decoding always reproduces the
original text, spaces and all.

## Splitting text before BPE

If BPE ran on the raw text, it would waste tokens on things like "the last
letter of one word plus the space after it". So tokenisers first split text
into words, and merges never cross from one word into the next. The exercise
gives you this splitter, `_pretokenize`.

One detail keeps the text reversible. A single space stays attached to the
start of the word after it, so " cat" is one piece. A run of two or more
spaces, a tab, or a newline becomes its own piece. For example, "the cat  sat"
followed by a newline splits into "the", " cat", a two-space piece, "sat",
and the newline. Join the pieces and you get the input back exactly.

## Other ways to pick merges

These are worth knowing by name. You don't build them here.

- **WordPiece** scores a pair by how often it appears together divided by how
  often each half appears on its own. A pair of two very common pieces
  doesn't win only because both pieces are common.
- **Unigram** works the other way round. It starts with a huge vocabulary
  and removes the tokens that matter least, until it reaches the target size.

## Fertility

**Fertility** is the average number of tokens per word. Lower is better: the
text fits in fewer tokens.

$$
\text{fertility} = \frac{\text{number of tokens}}{\text{number of words}}
$$

Read aloud: encode the text, count the tokens, and divide by the number of
words, where words are what you get by splitting on whitespace.

```python
f = len(encode(text, merges)) / len(text.split())
```

With no merges, "the quick fox" is 13 bytes and 3 words, so its fertility is
$13 / 3 = 4.33$. As you train more merges, common words shrink to one or two
tokens, and fertility falls.

> [!RECAP]
> - Start each word from `list(piece.encode("utf-8"))`, so nothing is ever
>   unknown.
> - Count pairs inside each word only, never across word boundaries.
> - Merge from the left and jump two after each merge, so `[5, 5, 5]` becomes
>   `[20, 5]`.
> - Merge $i$ makes ID $256 + i$, and ties go to the smaller pair.
> - Encode by running every merge in order. Decode by joining all the bytes
>   before one `.decode("utf-8")`.

## Build it step by step

The checks test your functions in this order.

1. **`get_pair_counts(token_lists)`** takes a list of lists of ints and
   returns a `dict` from pair tuples to counts. Use `zip(tokens, tokens[1:])`
   inside a loop over the words. Common mistake: joining all words into one
   list first, which counts pairs across word boundaries.
2. **`merge_pair(token_lists, pair, new_id)`** returns a new list of lists.
   Use a `while` loop with an index, not a `for` loop, so you can jump ahead
   by two. Common mistake: turning `[5, 5, 5]` into `[20, 20]` by letting
   the middle token join both pairs.
3. **`encode(text, merges)`** returns a flat list of ints. Call
   `_pretokenize`, turn each piece into `list(piece.encode("utf-8"))`, loop
   over `enumerate(merges)` calling `merge_pair` with `256 + i`, then flatten.
   With an empty merge list, it returns the raw bytes.
4. **`decode(ids, vocab)`** returns a string. Join the bytes, then call
   `.decode("utf-8")`. Common mistake: decoding each token to text on its
   own. A token can hold half of a two-byte character, which fails alone.
5. **`train_bpe(corpus, vocab_size)`** returns `(merges, vocab)`. Start
   `vocab` as `{i: bytes([i]) for i in range(256)}`. While the next ID is
   below `vocab_size`, count pairs, stop if there are none, pick the best
   pair with `min(counts, key=lambda p: (-counts[p], p))`, record it, add its
   bytes to `vocab`, and merge it. Common mistake: breaking ties with `max`,
   which picks the larger pair and fails the worked example.
6. **`fertility(text, merges)`** returns a float, the token count from
   `encode` divided by `len(text.split())`.

## Cost and scaling

Training does one full pass over the corpus for every merge. With $n$ bytes
of training text and $M$ merges, that's about $n \times M$ steps. The table
shows how the pieces grow.

| Piece | Work | Grows with |
|---|---|---|
| One round of counting pairs | one pass over all tokens | corpus size |
| Training | one pass per merge | corpus size times merges |
| Encoding a text | one pass per merge | text length times merges |
| Output layer in the model | $2Vd$ flops per token | vocabulary size |

Real tokenisers avoid recounting everything. They keep a priority queue of
pair counts and only update the pairs next to each merge.

## Where this breaks

- **Numbers.** "1234" might be one token, "12" plus "34", or four digits,
  depending on what the training text happened to contain. The model sees no
  consistent place value, which makes arithmetic harder. Some tokenisers
  always split digits one by one.
- **Code indentation.** A run of spaces only becomes one token if that exact
  run was common in training. Deeply nested code can cost one token per
  space.
- **Languages the tokeniser rarely saw.** If most training text is English,
  most merges are English pieces. A script that needs two to four bytes per
  character, with few merges of its own, can cost several tokens per
  character. The same model then charges those users far more tokens for the
  same meaning.

## The exercise

You build byte-level BPE from the bottom up: `get_pair_counts`, `merge_pair`,
`encode`, `decode`, `train_bpe`, and `fertility`. The splitter `_pretokenize`
is provided.

The checks confirm the pair counts and merges on the small lists from this
lesson, including the `[5, 5, 5]` case. They check that `encode` and `decode`
round-trip text with accents, Chinese characters, and emoji, both with no
merges and after training. They check that `train_bpe` on "aaabdaaabac"
produces exactly the three merges you traced by hand, that the vocabulary
reaches the requested size, and that running the merges out of order gives
more tokens than running them in order. Finally, they check that fertility
falls as the vocabulary grows.
