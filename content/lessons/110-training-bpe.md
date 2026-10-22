---
title: "Training byte-pair encoding: merge the most common pair"
slug: 110-training-bpe
part: "Part 7 — Large language models"
summary: Build byte-pair encoding training from a pair counter and a pair merger, and see why the ordered merge list is the whole tokeniser.
minutes: 40
gpu: false
objectives:
  - Explain why the most frequent adjacent pair is the best use of a scarce vocabulary slot, and why its count only approximates the tokens it saves.
  - Trace three rounds of byte-pair encoding by hand on a short string, including a tie.
  - Write the pair counter, the left-to-right merger, and the training loop, and explain why merge i gets id 256 plus i.
  - Explain why training splits text into words first, so merges never cross a word boundary.
prereqs:
  - 109-bytes-and-the-vocabulary-trade
exercise: 110-training-bpe
---

# Training byte-pair encoding: merge the most common pair

Suppose you had to copy the string `aaabdaaabac` a thousand times by hand, and you were allowed to invent a few shorthand symbols to save effort. You would not abbreviate something that appears once. You would look for the neighbours that appear together most often, give them one symbol, and then look again, because each shorthand has neighbours of its own. A tokeniser faces the same job. Each shorthand costs one row in the vocabulary, so you want the shorthand that saves the most writing.

Without that idea, a byte tokeniser stays at one position per byte, and the 5,700 positions of a paragraph, as in the chapter "Why bytes, and the trade between vocabulary and length", stay 5,700. With it, you merge the pair you see most, repeat, and keep the ordered list of merges. That ordered list is the tokeniser. This chapter builds the training loop. The next chapter uses the list to encode and decode.

## What you already know

Four ideas carry this chapter:

- The base alphabet is the 256 byte values, so no text is unknown (see "Why bytes, and the trade between vocabulary and length").
- Every vocabulary row costs memory in the embedding table and the output layer, so rows are scarce.
- A sequence that gets shorter makes attention cheaper, because attention cost grows with the square of the length.
- A Python dict can use a tuple such as `(97, 98)` as a key.

## Shorthand for repeated neighbours

Picture the string as a strip of cells. Figure 1 shows `aaabdaaabac` as 11 cells, then the strip after each merge. In each strip the orange cells are the pair about to be glued, and the green cells are tokens that already stand for more than one byte.

```figure
<svg viewBox="0 0 640 330" role="img" aria-label="Four strips showing aaabdaaabac shrinking from 11 tokens to 9, 7, and 5 as three merges are applied">
<text x="20" y="28" text-anchor="start" class="fig-bold">start: 11 tokens</text>
<rect x="20" y="36" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="35.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="54" y="36" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="69.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="88" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="103.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="122" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="137.5" y="56" text-anchor="middle" class="fig-mono">b</text>
<rect x="156" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="171.5" y="56" text-anchor="middle" class="fig-mono">d</text>
<rect x="190" y="36" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="205.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="224" y="36" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="239.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="258" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="273.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="292" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="307.5" y="56" text-anchor="middle" class="fig-mono">b</text>
<rect x="326" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="341.5" y="56" text-anchor="middle" class="fig-mono">a</text>
<rect x="360" y="36" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="375.5" y="56" text-anchor="middle" class="fig-mono">c</text>
<text x="412" y="48" text-anchor="start" class="fig-small">(a,a) seen 4×</text>
<text x="412" y="64" text-anchor="start" class="fig-small">merges 2× into 256</text>
<text x="20" y="104" text-anchor="start" class="fig-bold">after merge 0: 9 tokens</text>
<rect x="20" y="112" width="31" height="30" rx="4" class="fill-soft-c stroke-c"/>
<text x="35.5" y="132" text-anchor="middle" class="fig-mono">256</text>
<rect x="54" y="112" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="69.5" y="132" text-anchor="middle" class="fig-mono">a</text>
<rect x="88" y="112" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="103.5" y="132" text-anchor="middle" class="fig-mono">b</text>
<rect x="122" y="112" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="137.5" y="132" text-anchor="middle" class="fig-mono">d</text>
<rect x="156" y="112" width="31" height="30" rx="4" class="fill-soft-c stroke-c"/>
<text x="171.5" y="132" text-anchor="middle" class="fig-mono">256</text>
<rect x="190" y="112" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="205.5" y="132" text-anchor="middle" class="fig-mono">a</text>
<rect x="224" y="112" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="239.5" y="132" text-anchor="middle" class="fig-mono">b</text>
<rect x="258" y="112" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="273.5" y="132" text-anchor="middle" class="fig-mono">a</text>
<rect x="292" y="112" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="307.5" y="132" text-anchor="middle" class="fig-mono">c</text>
<text x="412" y="124" text-anchor="start" class="fig-small">(a,b) seen 2×</text>
<text x="412" y="140" text-anchor="start" class="fig-small">merges 2× into 257</text>
<text x="20" y="180" text-anchor="start" class="fig-bold">after merge 1: 7 tokens</text>
<rect x="20" y="188" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="35.5" y="208" text-anchor="middle" class="fig-mono">256</text>
<rect x="54" y="188" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="69.5" y="208" text-anchor="middle" class="fig-mono">257</text>
<rect x="88" y="188" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="103.5" y="208" text-anchor="middle" class="fig-mono">d</text>
<rect x="122" y="188" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="137.5" y="208" text-anchor="middle" class="fig-mono">256</text>
<rect x="156" y="188" width="31" height="30" rx="4" class="fill-soft-b stroke-b"/>
<text x="171.5" y="208" text-anchor="middle" class="fig-mono">257</text>
<rect x="190" y="188" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="205.5" y="208" text-anchor="middle" class="fig-mono">a</text>
<rect x="224" y="188" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="239.5" y="208" text-anchor="middle" class="fig-mono">c</text>
<text x="412" y="200" text-anchor="start" class="fig-small">(256,257) seen 2×</text>
<text x="412" y="216" text-anchor="start" class="fig-small">merges 2× into 258</text>
<text x="20" y="256" text-anchor="start" class="fig-bold">after merge 2: 5 tokens</text>
<rect x="20" y="264" width="31" height="30" rx="4" class="fill-soft-c stroke-c"/>
<text x="35.5" y="284" text-anchor="middle" class="fig-mono">258</text>
<rect x="54" y="264" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="69.5" y="284" text-anchor="middle" class="fig-mono">d</text>
<rect x="88" y="264" width="31" height="30" rx="4" class="fill-soft-c stroke-c"/>
<text x="103.5" y="284" text-anchor="middle" class="fig-mono">258</text>
<rect x="122" y="264" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="137.5" y="284" text-anchor="middle" class="fig-mono">a</text>
<rect x="156" y="264" width="31" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="171.5" y="284" text-anchor="middle" class="fig-mono">c</text>
</svg>
Figure 1. Three merges shrink 11 tokens to 5. In each strip the orange cells are the pair about to be merged. Notice that the first strip has three a's in a row, and only the first two are glued, so the third one stays alone.
```

Why the most frequent pair? A merge adds one row to the vocabulary and shortens every sequence that contains the pair. If a pair appears $k$ times, gluing it removes about $k$ tokens, so the pair with the biggest count is the best use of the row. Why pairs and not a longer chunk straight away? Because pairs are cheap to count, and larger chunks arise on their own: once `aa` and `ab` are tokens, the pair of those two tokens is `aaab`.

The count is a stand-in for the saving, and it can be off. Here is the first round on the 11 bytes, where `a` is 97, `b` is 98, `c` is 99, and `d` is 100:

| Pair | Count | Tokens saved if merged |
|---|---|---|
| (a, a) | 4 | 2 |
| (a, b) | 2 | 2 |
| (b, d), (d, a), (b, a), (a, c) | 1 each | 1 each |

The pair `(a, a)` appears four times, but each `aaa` run overlaps itself: in `aaa` the first two a's and the last two a's both form the pair, yet you can glue only one of them. So the count is 4 and the saving is 2. Training still picks by count, because counting is quick and the count is right when pairs do not overlap.

## One merge, by hand

Take the word `aaab`, which is `[97, 97, 97, 98]`. The pair `(97, 97)` appears twice. Scan from the left, and when two neighbours match, write one id for the pair and skip both. The first two 97s become the new id, and the third 97 has no partner left. The word is `[id, 97, 98]`, where `id` is the number for the glued pair.

The merged id must not clash with a byte, and the bytes occupy 0 to 255. So the first merge gets 256, the second gets 257, and the merge at position $i$ in the list gets:

$$
\text{id}(\text{merge } i) = 256 + i
$$

Read it aloud: the id is the first free number, 256, plus how many merges came before. With $i = 0$ the first merge has id 256, so the word becomes `[256, 97, 98]`. You never need to store ids anywhere else, because the position in the list already says them. Each id also stands for bytes. Id 256 stands for the bytes of `aa`, which is the bytes of 97 and 97 joined.

## Repeat until the vocabulary is big enough

Training repeats the step: count pairs, take the winner, merge it everywhere, record it. On `aaabdaaabac` the three rounds go like this.

Round 0: `(97, 97)` wins with 4. It becomes 256, and the strip is 9 tokens. Round 1: `(256, 97)` and `(97, 98)` each appear twice, a tie. The rule is to pick the smaller pair as a tuple, so `(97, 98)` wins and becomes 257, bytes `ab`. The strip is 7 tokens: `256, 257, 100, 256, 257, 97, 99`. Round 2: `(256, 257)` appears twice and wins. It becomes 258, bytes `aaab`, and the strip is `258, 100, 258, 97, 99`, 5 tokens.

Why does a tie rule exist? Ties are common in small texts, and two training runs must agree. Any fixed rule works, and "smaller tuple" is the one this exercise uses. Other tokenisers break ties differently, for example by earliest occurrence, so treat it as a convention, not a law.

The result is the merge list `[(97, 97), (97, 98), (256, 257)]`. Figure 2 shows how it builds up.

```figure
<svg viewBox="0 0 620 304" role="img" aria-label="A tree: token 258 splits into 256 and 257, 256 splits into two 97s, and 257 splits into 97 and 98, leaves spell a a a b">
<rect x="215" y="30" width="150" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="290" y="48" text-anchor="middle" class="fig-bold">258</text>
<text x="290" y="63" text-anchor="middle" class="fig-small">aaab</text>
<rect x="115" y="120" width="110" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="170" y="138" text-anchor="middle" class="fig-bold">256</text>
<text x="170" y="153" text-anchor="middle" class="fig-small">aa</text>
<rect x="355" y="120" width="110" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="410" y="138" text-anchor="middle" class="fig-bold">257</text>
<text x="410" y="153" text-anchor="middle" class="fig-small">ab</text>
<rect x="65" y="210" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="110" y="228" text-anchor="middle" class="fig-bold">97</text>
<text x="110" y="243" text-anchor="middle" class="fig-small">a</text>
<rect x="185" y="210" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="230" y="228" text-anchor="middle" class="fig-bold">97</text>
<text x="230" y="243" text-anchor="middle" class="fig-small">a</text>
<rect x="305" y="210" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="350" y="228" text-anchor="middle" class="fig-bold">97</text>
<text x="350" y="243" text-anchor="middle" class="fig-small">a</text>
<rect x="425" y="210" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="470" y="228" text-anchor="middle" class="fig-bold">98</text>
<text x="470" y="243" text-anchor="middle" class="fig-small">b</text>
<line x1="260" y1="70" x2="190" y2="120" class="stroke-muted"/>
<line x1="320" y1="70" x2="390" y2="120" class="stroke-muted"/>
<line x1="150" y1="160" x2="120" y2="210" class="stroke-muted"/>
<line x1="190" y1="160" x2="220" y2="210" class="stroke-muted"/>
<line x1="395" y1="160" x2="355" y2="210" class="stroke-muted"/>
<line x1="425" y1="160" x2="465" y2="210" class="stroke-muted"/>
<text x="375" y="55" text-anchor="start" class="fig-small">merge 2: (256, 257)</text>
<text x="235" y="145" text-anchor="start" class="fig-small">merge 0: (97, 97)</text>
<text x="475" y="145" text-anchor="start" class="fig-small">merge 1: (97, 98)</text>
<text x="60" y="290" text-anchor="start" class="fig-small">256 joins 97 and 97, so it reads aa. 258 joins 256 and 257, so it reads aaab.</text>
</svg>
Figure 2. Token 258 is built from 256 and 257, which are built from earlier tokens, down to the bytes. The merge list is a hierarchy. A later merge uses ids that earlier merges created.
```

Each id's bytes come from joining its parts, and one line of Python records that: `vocab[256 + i] = vocab[a] + vocab[b]`. The tokeniser is then the ordered merge list plus the 256 byte entries. You can throw the training text away.

> [!KEY]
> Each round counts adjacent pairs, merges the most common one into the next free id, and records it. The ordered merge list is the tokeniser. Ties go to the smaller pair, so every run agrees.

On a real text the same loop discovers words. Train it on a 532-byte paragraph of ordinary English, about 100 words, which is the same English text the exercise in the chapter "Measuring a tokeniser: fertility, fairness, and where it breaks" uses. The first twelve merges are `' t'`, `'in'`, `'he'`, `' the'`, `'ing'`, `' a'`, `' w'`, `'re'`, `' s'`, `'nd'`, `'at'`, and `' and'`. After 80 merges the 532 bytes of that paragraph are 199 tokens. Nobody told the loop about words or suffixes. The pieces that repeat most, such as `' the'`, `'ing'`, and `' and'`, won on their own.

## Keep merges inside words

Run the loop on the raw stream of bytes, and spaces become part of the pairs. On that English corpus the most common pair is `'e'` followed by a space, 28 times. Gluing the end of one word to the space that follows is wasteful, as Figure 3 shows: the token `'e '` appears wherever a word ends in e, but it says nothing about which word.

```figure
<svg viewBox="0 0 620 270" role="img" aria-label="The bytes of the tree: a merge across the word boundary versus separate pieces">
<text x="20" y="30" text-anchor="start" class="fig-bold">Without pre-splitting: bytes of the text, one stream</text>
<rect x="60" y="46" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="80" y="69" text-anchor="middle" class="fig-mono">t</text>
<rect x="104" y="46" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="124" y="69" text-anchor="middle" class="fig-mono">h</text>
<rect x="148" y="46" width="40" height="34" rx="4" class="fill-sunk stroke-bad"/>
<text x="168" y="69" text-anchor="middle" class="fig-mono">e</text>
<rect x="192" y="46" width="40" height="34" rx="4" class="fill-sunk stroke-bad"/>
<text x="212" y="69" text-anchor="middle" class="fig-mono">␣</text>
<rect x="236" y="46" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="256" y="69" text-anchor="middle" class="fig-mono">t</text>
<rect x="280" y="46" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="300" y="69" text-anchor="middle" class="fig-mono">r</text>
<rect x="324" y="46" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="344" y="69" text-anchor="middle" class="fig-mono">e</text>
<rect x="368" y="46" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="388" y="69" text-anchor="middle" class="fig-mono">e</text>
<text x="60" y="106" text-anchor="start" class="fig-small">'e␣' would glue the end of 'the' to the space before 'tree'</text>
<text x="20" y="170" text-anchor="start" class="fig-bold">With pre-splitting: the pieces 'the' and ' tree', merged separately</text>
<rect x="60" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="80" y="209" text-anchor="middle" class="fig-mono">t</text>
<rect x="104" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="124" y="209" text-anchor="middle" class="fig-mono">h</text>
<rect x="148" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="168" y="209" text-anchor="middle" class="fig-mono">e</text>
<rect x="216" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="236" y="209" text-anchor="middle" class="fig-mono">␣</text>
<rect x="260" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="280" y="209" text-anchor="middle" class="fig-mono">t</text>
<rect x="304" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="324" y="209" text-anchor="middle" class="fig-mono">r</text>
<rect x="348" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="368" y="209" text-anchor="middle" class="fig-mono">e</text>
<rect x="392" y="186" width="40" height="34" rx="4" class="fill-soft-a stroke-a"/>
<text x="412" y="209" text-anchor="middle" class="fig-mono">e</text>
<line x1="200" y1="178" x2="200" y2="232" class="stroke-ink dashed"/>
<text x="126" y="250" text-anchor="middle" class="fig-small">the</text>
<text x="326" y="250" text-anchor="middle" class="fig-small"> tree (space stays with the word that follows)</text>
</svg>
Figure 3. Top: on the raw stream, the bytes e and space can merge across the boundary between two words. Bottom: split first into the pieces `the` and ` tree`, and merges happen only inside a piece. The space attaches to the word that follows it.
```

So training splits the text into pieces first, and counts and merges only inside each piece. The exercise gives you the splitter `_pretokenize`. A piece is a word with at most one leading space, so `" cat"` is one piece. Punctuation stays attached, so `" dog."` is one piece. Why attach the space to the word after it? Then each piece carries its own start marker, `"cat"` after a space and `"cat"` at the start of a text are different pieces, and nothing needs to look outside a piece to decide a merge. A run of whitespace becomes its own piece, and so does a lone tab or newline. A space that follows other whitespace joins that run instead of the word, so two spaces before a word are one two-space piece and the word has no leading space. On `"the cat  sat\n"` the splitter returns `"the"`, `" cat"`, a two-space piece, `"sat"`, and `"\n"`. Joined together, the pieces give the original text back.

> [!KEY]
> Merges are counted and applied inside pieces, never across them. The split is what keeps the vocabulary for words and word parts, and not for the join between two words.

> [!TIP]
> To pick the winner and apply the tie rule in one line, use `min(counts, key=lambda p: (-counts[p], p))`. It sorts by largest count first, then by the smaller pair.

> [!RECAP]
> Hold these points while you write the functions:
>
> - A merge saves about as many tokens as the pair's count, less for overlapping runs like `aaa`.
> - Scan left to right and skip two after a hit, so `[5, 5, 5]` becomes `[20, 5]`.
> - Merge $i$ gets id $256 + i$, and its bytes are the joined bytes of its two parts.
> - A tie goes to the smaller pair as a tuple.
> - Split into pieces first, and count and merge inside each piece only.

## Build it step by step

The checks call three functions, in this order:

1. `get_pair_counts(token_lists)` takes a list of lists of ints and returns a dict from pair tuples to counts. Walk each inner list on its own with `zip(tokens, tokens[1:])`. Leave out pairs that never occur. The usual mistake is joining the lists first, which counts a pair across the end of one word and the start of the next.

```python
def get_pair_counts(token_lists):
    counts = {}
    # for each list, zip it with itself shifted by one, add 1 to counts[pair]
    return counts
```

2. `merge_pair(token_lists, pair, new_id)` returns fresh lists with the pair replaced. Use a `while` loop with an index: on a match write `new_id` and add 2 to the index, otherwise write the current id and add 1. The usual mistake is turning `[5, 5, 5]` into `[20, 20]`, which would decode to four 5s.

```python
def merge_pair(token_lists, pair, new_id):
    # on a match: out.append(new_id); i += 2     otherwise: out.append(tokens[i]); i += 1
    return merged
```

3. `train_bpe(corpus, vocab_size)` returns `(merges, vocab)`. Split with `_pretokenize`, turn each piece into a list of byte ints, and start the vocabulary at `{i: bytes([i]) for i in range(256)}`. Loop until the vocabulary reaches `vocab_size` or no pair is left: count, pick the winner, record its joined bytes, merge. The usual mistake is breaking ties with `max`, which picks the larger pair.

```python
def train_bpe(corpus, vocab_size):
    # while 256 + len(merges) < vocab_size and there are pairs:
    #   best = min(counts, key=lambda p: (-counts[p], p))
    return merges, vocab
```

## Where this breaks

The loop recounts every pair after every merge, so training on a large corpus with thousands of merges is slow when written this way. Practical tokenisers update only the counts next to each merge. Training is also greedy: an early merge is never undone, even if a later one would have made a better vocabulary.

## The exercise

You write `get_pair_counts`, `merge_pair`, and `train_bpe`. The splitter `_pretokenize` is given. The checks verify pair counts across several words, the left-to-right merge including `[5, 5, 5]`, and that merging never changes what the tokens expand to. They also verify that `train_bpe` on `aaabdaaabac` returns the three merges from this chapter, breaks ties to the smaller pair, never merges across words, and stops when no pair is left.
