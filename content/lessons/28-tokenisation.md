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

A transformer reads integers. Text has to become a list of those integers before training starts. The program that does this is the tokeniser, and every later weight depends on the ids it chose.

Two sizes pull against each other. Letters need a small vocabulary and a long sequence. Whole words need a short sequence and a huge vocabulary, and a word that never appeared has no id. Attention cost grows with the square of the sequence length, so both extremes get expensive.

You build byte-pair encoding. You start from raw bytes, glue the pair you see most often, and keep an ordered merge list. That list encodes text. The byte table decodes it. You also measure fertility, the average number of tokens per word.

## What you already know

This lesson uses five ideas from earlier work:

- Attention cost grows with the square of the sequence length.
- The embedding table and the output layer each store one row per vocabulary id.
- Training compute grows in proportion to the number of tokens.
- UTF-8 stores each character as one to four bytes, and each byte is an integer from 0 to 255.
- A Python dict can use a tuple such as `(97, 98)` as a key.

## Merge one pair by hand

Take the one-word corpus `"aaab"`. The letter `"a"` is byte 97 and `"b"` is byte 98, so the word is `[97, 97, 97, 98]`.

Walk the neighbours. The pairs are `(97, 97)`, `(97, 97)`, and `(97, 98)`. The count of `(97, 97)` is 2. The count of `(97, 98)` is 1. The winning pair is `(97, 97)`.

The first merge gets the next free id, 256. That id stands for the two parts joined, here the bytes of `"aa"`. Scan from the left and skip two positions after a hit. The first two 97s become 256, and the word is `[256, 97, 98]`.

This one step is a merge in byte-pair encoding, or BPE. Merge number $i$, counting from 0, gets a fixed id:

$$
\text{id}(\text{merge } i) = 256 + i
$$

The id is 256 plus the merge's position in the list. Here $i = 0$, so the id is 256, the same number as in the hand merge. The count lives in a dict:

```python
counts = {}
# zip [97, 97, 97, 98] with that list shifted by one
# add 1 to counts[(a, b)] for each neighbour pair
```

The usual mistake is to let a run of three equal ids merge twice. `[97, 97, 97]` becomes `[256, 97]`.

> [!KEY]
> A byte-level tokeniser starts from 256 ids, one per byte, so every text has an encoding. A merge only shortens the sequence.

## Train by repeating that merge

Training repeats the same step. Split the corpus into words, turn each word into bytes, count pairs inside each word, and merge the winner. Pairs never cross from one word into the next. Stop when the vocabulary reaches the size you asked for, or when no pair remains.

On a tie, pick the pair that is smaller as a tuple of ids. The vocabulary maps each id to bytes. A merged id's bytes are its two parts concatenated.

Train on `"aaabdaaabac"`. The bytes are 97 for `"a"`, 98 for `"b"`, 99 for `"c"`, and 100 for `"d"`. The string is 11 bytes.

In round 0, `(97, 97)` appears 4 times. Every other pair appears once or twice. The winner is `(97, 97)`, and it becomes id 256, the bytes `b"aa"`. After the left-to-right merge the ids are `256, 97, 98, 100, 256, 97, 98, 97, 99`.

In round 1, `(256, 97)` and `(97, 98)` both appear twice. The smaller tuple is `(97, 98)`, so it wins and becomes id 257, the bytes `b"ab"`. The ids become `256, 257, 100, 256, 257, 97, 99`.

In round 2, `(256, 257)` appears twice and wins. It becomes id 258, the bytes `b"aaab"`. The ids are `258, 100, 258, 97, 99`. Eleven bytes have become five tokens.

The merge list is `[(97, 97), (97, 98), (256, 257)]`, and the vocabulary holds $256 + 3 = 259$ ids. You can drop the training text. The ordered list and the byte table are the tokeniser.

> [!TIP]
> `min(counts, key=lambda p: (-counts[p], p))` picks the largest count, and on a tie it picks the smaller pair.

> [!KEY]
> Each round counts pairs, merges the winner, and gives it id $256 + i$. On a tie, the smaller pair wins.

Store the bytes with one concatenation:

```python
vocab[new_id] = vocab[a] + vocab[b]  # bytes, not a string
```

The usual mistake is to break the tie with `max`. That chooses the larger pair and fails this corpus.

## Encode in learned order

Encoding starts from bytes again. Apply merge 0 everywhere, then merge 1, then the rest of the list. Flatten the words into one list of ids. You recount nothing on the text you encode.

Order matters, because a later pair is built from ids that exist only after earlier merges. Reverse those three merges and encode `"aaabdaaabac"`. You get `[258, 257, 100, 258, 257, 97, 99]`, seven tokens rather than five. The result is a valid encoding, and it is a longer one.

Some long tokens appear only after particular shorter merges. Walking the learned list is the encoding rule.

> [!KEY]
> Encode by running the merges in the order they were learned. A later merge glues ids that exist only after the earlier merges.

Decoding looks up each id's bytes, joins them all, and turns the bytes into text once:

```python
text = b"".join(vocab[i] for i in ids).decode("utf-8")
```

The usual mistake is to decode each id to text on its own. One id can hold half of a character that uses two bytes, and that half is not valid text.

> [!WARNING]
> Join every byte first, then call `.decode("utf-8")` once. Decoding one token at a time can fail on a split character.

Each id stands for exact bytes, so decoding returns the original text, spaces included.

## Keep merges inside words

Running BPE on the raw byte stream spends merges on the end of one word plus the space after it. A splitter runs first, and merges stay inside each piece. The exercise gives you that splitter as `_pretokenize`.

A single space stays on the word that follows, so `" cat"` is one piece. Two or more spaces, a tab, or a newline is its own piece. The text `"the cat  sat"` plus a newline splits into `"the"`, `" cat"`, a two-space piece, `"sat"`, and the newline. Join those pieces and the input comes back.

## Score pairs another way

Two other training scores are worth naming. You do not build them here:

- WordPiece divides the pair count by the product of the two piece counts. Two common pieces do not win only because each half is common.
- Unigram starts from a large vocabulary and deletes the tokens that help least, until the size hits the target.

The usual mistake is to change the encode loop when the training score changes. Encoding still applies a learned list in order.

## Measure fertility

Fertility is the average number of tokens per word. A lower value means the same text costs fewer tokens. Count words with a plain whitespace split. That split is separate from `_pretokenize`.

$$
\text{fertility} = \frac{\text{number of tokens}}{\text{number of words}}
$$

Fertility is the encoded length divided by the word count. With no merges, `"the quick fox"` is 13 bytes and 3 words, so fertility is $13 / 3 = 4.33$. The same ratio is one division:

```python
ratio = len(encode(text, merges)) / len(text.split())
```

Further merges shrink common words toward one token, and fertility falls. The usual mistake is to count words with `_pretokenize` instead of `text.split()`.

> [!RECAP]
> Hold these five checks while you write the functions:
>
> - Start each word from its UTF-8 bytes, so no text is left unknown.
> - Count pairs inside each word, and never across a word boundary.
> - Merge from the left and jump two after a hit, so `[5, 5, 5]` becomes `[20, 5]`.
> - Merge $i$ creates id $256 + i$, and a tie goes to the smaller pair.
> - Encode in list order. Decode by joining all the bytes, then one `.decode("utf-8")`.

## Build it step by step

The checks call your functions in this order:

1. `get_pair_counts(token_lists)` takes a list of lists of ints and returns a dict from pair tuples to counts. `zip` neighbours inside each word. Leave missing pairs out of the dict. The usual mistake is to join the words first. The loop has this shape:

```python
def get_pair_counts(token_lists):
    counts = {}
    # zip neighbours inside one word at a time
    return counts
```

2. `merge_pair(token_lists, pair, new_id)` returns a separate list of lists. Use a `while` index so a hit jumps by two. The usual mistake turns `[5, 5, 5]` into `[20, 20]`. The index moves like this:

```python
def merge_pair(token_lists, pair, new_id):
    # on a hit, write new_id and add 2 to the index
    # otherwise write the current id and add 1
    return merged
```

3. `encode(text, merges)` returns one flat list of ints. Call `_pretokenize`, then `list(piece.encode("utf-8"))`, then `merge_pair` at position $i$ with id $256 + i$. Flatten at the end. An empty list returns raw bytes. The calls sit in this order:

```python
def encode(text, merges):
    # pretokenize, bytes, then merge i with id 256 + i
    return ids
```

4. `decode(ids, vocab)` returns a string. Join the bytes, then call `.decode("utf-8")` once. The usual mistake decodes each id alone. Two operations do the job:

```python
def decode(ids, vocab):
    # join vocab[i], then decode utf-8 once
    return text
```

5. `train_bpe(corpus, vocab_size)` returns `(merges, vocab)`. Start from `{i: bytes([i]) for i in range(256)}`. Stop at `vocab_size`, or when no pair remains. Pick the winner, store the joined bytes, and merge. The usual mistake breaks ties with `max`. The loop has this shape:

```python
def train_bpe(corpus, vocab_size):
    # while the next id is below vocab_size and a pair exists
    # pick min(counts, key=lambda p: (-counts[p], p))
    return merges, vocab
```

6. `fertility(text, merges)` returns a float from `encode` divided by `len(text.split())`. The ratio has this shape:

```python
def fertility(text, merges):
    # len(encode(text, merges)) / len(text.split())
    return ratio
```

## Cost and scaling

Training walks the corpus once per merge. With $n$ bytes of text and $M$ merges, that is about $n \times M$ steps. Encoding one string does one pass per merge. A practical tokeniser keeps a priority queue of pair counts and updates only the pairs beside each merge, so it does not recount everything.

The work grows as follows:

| Piece | Work | Grows with |
|---|---|---|
| One counting pass | one walk over the tokens | corpus size |
| Training | one pass per merge | corpus size times merges |
| Encoding one text | one pass per merge | text length times merges |
| Output layer | $2Vd$ flops per token | vocabulary size |

## Where this breaks

Digits are unstable. `"1234"` can be one token, `"12"` plus `"34"`, or four separate digits, depending on the training text. There is then no fixed place value, and arithmetic gets harder. Some tokenisers split every digit on its own.

Spaces have the same problem in code. A run of spaces becomes one token only when that exact run was common in training. Deep indentation can cost one token per space.

A corpus that is mostly one language spends its merges there. A script that needs two to four bytes per character, with few merges of its own, can cost several tokens per character. Readers of that script then spend far more tokens for the same meaning.

## The exercise

You build byte-level BPE from the bottom: `get_pair_counts`, `merge_pair`, `encode`, `decode`, `train_bpe`, and `fertility`. The splitter `_pretokenize` is already written.

The checks confirm the pair counts and the left-to-right merge, including `[5, 5, 5]`. They check that `encode` and `decode` round-trip accents, Chinese characters, and emoji, both with no merges and after training. They check that `train_bpe` on `"aaabdaaabac"` returns the three merges from this lesson, that the vocabulary reaches the requested size, and that the reversed list produces more tokens. They also check that fertility falls as the vocabulary grows.
