---
title: "Byte-pair encoding: train it, then use it"
slug: 100-byte-pair-encoding
part: "Part 7 — Large language models"
summary: Learn an ordered merge list by gluing the most common pair again and again, then replay it to encode text and join bytes to decode it exactly.
minutes: 55
gpu: false
objectives:
  - Explain why the most frequent adjacent pair is the best use of a scarce vocabulary slot, and why its count only approximates the tokens it saves.
  - Trace three rounds of byte-pair encoding training by hand, including a tie, and explain why merge i gets id 256 plus i and why merges stay inside words.
  - Explain why encoding replays the frozen merges in the order they were learned and never recounts pairs.
  - Explain why decoding joins all the bytes before it decodes UTF-8, and why the round trip is exact.
  - Write the pair counter, the left-to-right merger, the training loop, encode, and decode.
prereqs:
  - 099-bytes-and-the-vocabulary-trade
exercise: 100-byte-pair-encoding
---

# Byte-pair encoding: train it, then use it

Suppose you had to copy the string `aaabdaaabac` a thousand times by hand, and you were allowed to invent a few shorthand symbols to save effort. You would not abbreviate something that appears once. You would look for the neighbours that appear together most often, give them one symbol, and then look again, because each shorthand has neighbours of its own. A tokeniser faces the same job. Each shorthand costs one row in the vocabulary, so you want the shorthand that saves the most writing.

Without that idea, a byte tokeniser stays at one position per byte. With it, you merge the pair you see most, repeat, and keep the ordered list of merges. That ordered list is the whole tokeniser. This chapter builds the training loop, and then answers the question that follows: given a sentence the tokeniser has never seen, how do you turn it into ids, and how do you get exactly the same sentence back?

## What you already know

- The base alphabet is the 256 byte values, so no text is unknown (see "Why bytes, and the trade between vocabulary and length").
- Every vocabulary row costs memory, so rows are scarce, and a shorter sequence makes attention cheaper.
- Text is a string of UTF-8 bytes, and one character can take up to four bytes.
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

Why the most frequent pair? A merge adds one row to the vocabulary and shortens every sequence that contains the pair. If a pair appears $k$ times, gluing it removes about $k$ tokens, so the pair with the biggest count is the best use of the row. Pairs are cheap to count, and larger chunks arise on their own: once `aa` and `ab` are tokens, the pair of those two tokens is `aaab`.

The count is a stand-in for the saving, and it can be off. Here is the first round on the 11 bytes, where `a` is 97, `b` is 98, `c` is 99, and `d` is 100:

| Pair | Count | Tokens saved if merged |
|---|---|---|
| (a, a) | 4 | 2 |
| (a, b) | 2 | 2 |
| (b, d), (d, a), (b, a), (a, c) | 1 each | 1 each |

The pair `(a, a)` appears four times, but each `aaa` run overlaps itself: in `aaa` the first two a's and the last two a's both form the pair, yet you can glue only one of them. So the count is 4 and the saving is 2. Training still picks by count, because counting is quick and the count is right when pairs do not overlap.

## One merge, by hand

Take the word `aaab`, which is `[97, 97, 97, 98]`. Scan from the left, and when two neighbours match, write one id for the pair and skip both. The first two 97s become the new id, and the third 97 has no partner left, so the word is `[id, 97, 98]`.

The merged id must not clash with a byte, and the bytes occupy 0 to 255. So the first merge gets 256, the second gets 257, and the merge at position $i$ in the list gets:

$$
\text{id}(\text{merge } i) = 256 + i
$$

Read it aloud: the id is the first free number, 256, plus how many merges came before. You never need to store ids anywhere else, because the position in the list already says them.

## Repeat until the vocabulary is big enough

Training repeats the step: count pairs, take the winner, merge it everywhere, record it. On `aaabdaaabac` the three rounds go like this.

Round 0: `(97, 97)` wins with 4 and becomes 256, leaving 9 tokens. Round 1: `(256, 97)` and `(97, 98)` each appear twice, a tie. The rule is to pick the smaller pair as a tuple, so `(97, 98)` wins and becomes 257, bytes `ab`. The strip is 7 tokens: `256, 257, 100, 256, 257, 97, 99`. Round 2: `(256, 257)` appears twice and wins. It becomes 258, bytes `aaab`, and the strip is `258, 100, 258, 97, 99`, 5 tokens.

Ties are common in small texts, and two training runs must agree, so a fixed rule is needed. "Smaller tuple" is this exercise's convention, not a law.

The merge list is `[(97, 97), (97, 98), (256, 257)]`. Each id's bytes come from joining its parts, and one line of Python records that: `vocab[256 + i] = vocab[a] + vocab[b]`. The tokeniser is the ordered merge list plus the 256 byte entries. You can throw the training text away.

On a real text the same loop discovers words. Trained on a 532-byte paragraph of ordinary English, the first twelve merges are `' t'`, `'in'`, `'he'`, `' the'`, `'ing'`, `' a'`, `' w'`, `'re'`, `' s'`, `'nd'`, `'at'`, and `' and'`. After 80 merges the 532 bytes are 199 tokens. Nobody told the loop about words or suffixes.

> [!KEY]
> Each round counts adjacent pairs, merges the most common one into the next free id, and records it. The ordered merge list is the tokeniser. Ties go to the smaller pair, so every run agrees.

## Keep merges inside words

Run the loop on the raw stream of bytes, and spaces become part of the pairs. On that corpus the most common pair is `'e'` followed by a space, 28 times. Gluing the end of one word to the space that follows is wasteful, as Figure 2 shows: the token `'e '` appears wherever a word ends in e, but it says nothing about which word.

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
Figure 2. Top: on the raw stream, the bytes e and space can merge across the boundary between two words. Bottom: split first into the pieces `the` and ` tree`, and merges happen only inside a piece. The space attaches to the word that follows it.
```

So training splits the text into pieces first, and counts and merges only inside each piece. The exercise gives you the splitter `_pretokenize`. A piece is a word with at most one leading space, so `" cat"` is one piece, and punctuation stays attached, so `" dog."` is one piece. The space attaches to the word after it, so nothing needs to look outside a piece to decide a merge. A run of whitespace becomes its own piece, and joined together, the pieces give the original text back.

> [!KEY]
> Merges are counted and applied inside pieces, never across them. The split keeps the vocabulary for words and word parts, not for the join between two words.

## Using the list: replay the merges in order

You now have a learned list and a sentence the tokeniser has never seen. One tempting answer is to run the training loop again on the sentence. That fails for a plain reason. The ids are a fixed code that the model's embedding table was trained on, so the word `cat` must get the same ids in every document. If the cut depended on the other words in the sentence, the same word would get different ids in different places, and the table rows would mean nothing. So the list is frozen, and encoding only replays it: for each piece, take its bytes, apply merge 0, then merge 1, and so on, then join the pieces' ids into one flat list.

Take the string you traced above, `aaabdaaabac`, as one piece of 11 bytes, with the learned list. Merge 0 glues `(97, 97)` into 256 and leaves 9 tokens. Merge 1 glues `(97, 98)` into 257 and leaves 7. Merge 2 glues `(256, 257)` into 258 and leaves 5: `[258, 100, 258, 97, 99]`. The left half of Figure 3 shows each step.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="Replaying the three merges of aaabdaaabac in learned order reaches five tokens, while replaying them in reverse order reaches seven">
<text x="10" y="22" text-anchor="start" class="fig-bold">Learned order</text>
<text x="10" y="40" text-anchor="start" class="fig-small">bytes: 11</text>
<rect x="10" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="22" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="37" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="49" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="64" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="76" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="91" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="103" y="65" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="118" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="130" y="65" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="145" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="157" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="172" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="184" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="199" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="211" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="226" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="238" y="65" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="253" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="265" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="280" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="292" y="65" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="10" y="102" text-anchor="start" class="fig-small">merge 0 (97, 97): 9</text>
<rect x="10" y="108" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="22" y="127" text-anchor="middle" class="fig-mono fig-small">256</text>
<rect x="37" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="49" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="64" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="76" y="127" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="91" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="103" y="127" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="118" y="108" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="130" y="127" text-anchor="middle" class="fig-mono fig-small">256</text>
<rect x="145" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="157" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="172" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="184" y="127" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="199" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="211" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="226" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="238" y="127" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="10" y="164" text-anchor="start" class="fig-small">merge 1 (97, 98): 7</text>
<rect x="10" y="170" width="24" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="22" y="189" text-anchor="middle" class="fig-mono fig-small">256</text>
<rect x="37" y="170" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="49" y="189" text-anchor="middle" class="fig-mono fig-small">257</text>
<rect x="64" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="76" y="189" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="91" y="170" width="24" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="103" y="189" text-anchor="middle" class="fig-mono fig-small">256</text>
<rect x="118" y="170" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="130" y="189" text-anchor="middle" class="fig-mono fig-small">257</text>
<rect x="145" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="157" y="189" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="172" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="184" y="189" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="10" y="226" text-anchor="start" class="fig-small">merge 2 (256, 257): 5</text>
<rect x="10" y="232" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="22" y="251" text-anchor="middle" class="fig-mono fig-small">258</text>
<rect x="37" y="232" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="49" y="251" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="64" y="232" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="76" y="251" text-anchor="middle" class="fig-mono fig-small">258</text>
<rect x="91" y="232" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="103" y="251" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="118" y="232" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="130" y="251" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="335" y="22" text-anchor="start" class="fig-bold">Reversed order</text>
<text x="335" y="40" text-anchor="start" class="fig-small">bytes: 11</text>
<rect x="335" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="347" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="362" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="374" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="389" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="401" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="416" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="428" y="65" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="443" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="455" y="65" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="470" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="482" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="497" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="509" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="524" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="536" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="551" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="563" y="65" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="578" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="590" y="65" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="605" y="46" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="617" y="65" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="335" y="102" text-anchor="start" class="fig-small">merge 0 (256, 257) no hit: 11</text>
<rect x="335" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="347" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="362" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="374" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="389" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="401" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="416" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="428" y="127" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="443" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="455" y="127" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="470" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="482" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="497" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="509" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="524" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="536" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="551" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="563" y="127" text-anchor="middle" class="fig-mono fig-small">b</text>
<rect x="578" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="590" y="127" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="605" y="108" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="617" y="127" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="335" y="164" text-anchor="start" class="fig-small">merge 1 (97, 98): 9</text>
<rect x="335" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="347" y="189" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="362" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="374" y="189" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="389" y="170" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="401" y="189" text-anchor="middle" class="fig-mono fig-small">257</text>
<rect x="416" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="428" y="189" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="443" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="455" y="189" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="470" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="482" y="189" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="497" y="170" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="509" y="189" text-anchor="middle" class="fig-mono fig-small">257</text>
<rect x="524" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="536" y="189" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="551" y="170" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="563" y="189" text-anchor="middle" class="fig-mono fig-small">c</text>
<text x="335" y="226" text-anchor="start" class="fig-small">merge 2 (97, 97): 7</text>
<rect x="335" y="232" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="347" y="251" text-anchor="middle" class="fig-mono fig-small">258</text>
<rect x="362" y="232" width="24" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="374" y="251" text-anchor="middle" class="fig-mono fig-small">257</text>
<rect x="389" y="232" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="401" y="251" text-anchor="middle" class="fig-mono fig-small">d</text>
<rect x="416" y="232" width="24" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="428" y="251" text-anchor="middle" class="fig-mono fig-small">258</text>
<rect x="443" y="232" width="24" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="455" y="251" text-anchor="middle" class="fig-mono fig-small">257</text>
<rect x="470" y="232" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="482" y="251" text-anchor="middle" class="fig-mono fig-small">a</text>
<rect x="497" y="232" width="24" height="28" rx="4" class="fill-soft-a stroke-a"/>
<text x="509" y="251" text-anchor="middle" class="fig-mono fig-small">c</text>
</svg>
Figure 3. Orange cells are the tokens a merge has made. Left: the merges replayed in the order they were learned, giving 5 tokens. Right: the same three merges in reverse order give 7 tokens. In the reversed list the first merge has no hit, because the ids 256 and 257 do not exist yet.
```

Why must the order match? Merge 2 glues the ids 256 and 257, and those ids exist only after merges 0 and 1 have run. More generally, merge $i$ was chosen on text that merges 0 to $i-1$ had already rewritten. Replaying in the same order rebuilds the same rewritten text, so each merge finds the pairs it was chosen for. 

Now reverse the list to `[(256, 257), (97, 98), (97, 97)]` and encode the same string. The first merge looks for ids 256 and 257, finds none, and does nothing. The second glues `(97, 98)`, which in this list has id 257. The third glues `(97, 97)` into id 258. The result is `[258, 257, 100, 258, 257, 97, 99]`, 7 tokens instead of 5, as the right half of Figure 3 shows. The length is the smaller problem: in the reversed list 257 means `ab` and 258 means `aa`. Decoded with the original vocabulary, where 258 is `aaab`, those ids give `aaababdaaababac`, not the text you started with. Order is part of the meaning of the ids.

> [!KEY]
> A trained tokeniser is a frozen recipe. Encode runs the merges in the order they were learned, because merge $i$ was chosen on text that merges 0 to $i-1$ had already rewritten. It never recounts, so the same word gets the same ids in every document.

Because the pieces are split before any merge, `encode("the cat")` equals `encode("the")` followed by `encode(" cat")`. One word's ids never depend on its neighbours. 

## Decode: join, then decode once

Every id stands for bytes. The vocabulary maps an id to its bytes, so decoding is a lookup, a join, and one decode:

```python
text = b"".join(vocab[i] for i in ids).decode("utf-8")
```

Join first, and decode once. A token is a slice of bytes, and a slice can cut a character in half, because a token boundary does not know about characters. With no merges at all, every token is a single byte, so the emoji 🎉 becomes four tokens holding 240, 159, 142, and 137. Figure 4 cuts those four into two tokens and decodes each way.

```figure
<svg viewBox="0 0 600 226" role="img" aria-label="Four bytes of an emoji split across two tokens: decoding the first token alone fails, joining all four bytes then decoding gives the emoji">
<text x="20" y="26" text-anchor="start" class="fig-bold">The emoji as bytes, cut into two tokens</text>
<rect x="70" y="40" width="64" height="36" rx="4" class="fill-soft-c stroke-c"/>
<text x="102" y="63" text-anchor="middle" class="fig-mono">240</text>
<rect x="140" y="40" width="64" height="36" rx="4" class="fill-soft-c stroke-c"/>
<text x="172" y="63" text-anchor="middle" class="fig-mono">159</text>
<rect x="210" y="40" width="64" height="36" rx="4" class="fill-soft-c stroke-c"/>
<text x="242" y="63" text-anchor="middle" class="fig-mono">142</text>
<rect x="280" y="40" width="64" height="36" rx="4" class="fill-soft-c stroke-c"/>
<text x="312" y="63" text-anchor="middle" class="fig-mono">137</text>
<rect x="66" y="84" width="138" height="6" class="fill-b"/>
<rect x="206" y="84" width="138" height="6" class="fill-d"/>
<text x="137" y="108" text-anchor="middle" class="fig-small">token 1: bytes 240 159</text>
<text x="277" y="108" text-anchor="middle" class="fig-small">token 2: bytes 142 137</text>
<text x="20" y="150" text-anchor="start" class="fig-bold">Decode one token at a time</text>
<rect x="20" y="160" width="270" height="44" rx="6" class="fill-sunk stroke-bad"/>
<text x="155" y="179" text-anchor="middle" class="fig-mono fig-small">bytes([240, 159]).decode()</text>
<text x="155" y="195" text-anchor="middle" class="fig-small">UnicodeDecodeError: unexpected end of data</text>
<text x="310" y="150" text-anchor="start" class="fig-bold">Join the bytes, then decode once</text>
<rect x="310" y="160" width="270" height="44" rx="6" class="fill-soft-c stroke-c"/>
<text x="445" y="179" text-anchor="middle" class="fig-mono fig-small">bytes([240, 159, 142, 137]).decode()</text>
<text x="445" y="195" text-anchor="middle" class="fig-small">returns the emoji, party popper</text>
</svg>
Figure 4. The emoji is four bytes. If a token boundary falls between its second and third byte, decoding the first token alone fails, because bytes 240 and 159 are only the start of a character. Joining all four bytes and decoding once returns the emoji.
```

> [!WARNING]
> Never call `.decode("utf-8")` on one token at a time. A token can hold half of a character, and half of a character is not valid text. Join all the bytes first.

The round trip follows. Each merge replaces a pair by an id whose bytes are those two parts joined, so the bytes of the whole sequence never change while you encode. Decoding joins them back into the same byte string, and a byte string that came from valid text decodes to that text. So `decode(encode(text)) == text` for every text, with merges or without, including accents, Chinese characters, and emoji.

> [!TIP]
> To pick the winner and apply the tie rule in one line, use `min(counts, key=lambda p: (-counts[p], p))`. It sorts by largest count first, then by the smaller pair.

> [!RECAP]
> Hold these points while you write the functions:
>
> - A merge saves about as many tokens as the pair's count, less for overlapping runs like `aaa`.
> - Scan left to right and skip two after a hit, so `[5, 5, 5]` becomes `[20, 5]`.
> - Merge $i$ gets id $256 + i$, a tie goes to the smaller pair, and counting and merging stay inside each piece.
> - Encode replays the merges in list order on each piece and never recounts.
> - Decode joins all the bytes, then calls `.decode("utf-8")` once.

## Build it step by step

The checks call six functions, in this order:

1. `get_pair_counts(token_lists)` takes a list of lists of ints and returns a dict from pair tuples to counts. Walk each inner list on its own with `zip(tokens, tokens[1:])`, and leave out pairs that never occur. The usual mistake is joining the lists first, which counts a pair across the end of one word and the start of the next.
2. `merge_pair(token_lists, pair, new_id)` returns fresh lists with the pair replaced. Use a `while` loop with an index: on a match write `new_id` and add 2 to the index, otherwise write the current id and add 1. The usual mistake is turning `[5, 5, 5]` into `[20, 20]`, which would decode to four 5s.
3. `train_bpe(corpus, vocab_size)` returns `(merges, vocab)`. Split with `_pretokenize`, turn each piece into a list of byte ints, and start the vocabulary at `{i: bytes([i]) for i in range(256)}`. Loop until the vocabulary reaches `vocab_size` or no pair is left: count, pick the winner, record its joined bytes, merge. The usual mistake is breaking ties with `max`, which picks the larger pair.
4. `apply_merges_to_word(word_bytes, merges)` takes the byte ints of one piece and returns its ids after the replay. For each `i, pair` in the list, call `merge_pair([ids], pair, 256 + i)` and keep the first list back. The usual mistake is using `i` instead of `256 + i` as the id.
5. `encode(text, merges)` returns one flat list of ids. Split with `_pretokenize`, turn each piece into `list(piece.encode("utf-8"))`, replay with the helper, and extend a flat list. With an empty merge list the result is the raw bytes.
6. `decode(ids, vocab)` returns a string. Join `vocab[i]` for every id, then decode once. The usual mistake is decoding each id alone, which raises an error on a half character.

## Where this breaks

The training loop recounts every pair after every merge, so a large corpus with thousands of merges is slow when written this way. Practical tokenisers update only the counts next to each merge. Training is also greedy: an early merge is never undone. Encoding costs one pass per merge, so practical tokenisers keep each merge's rank in a table and apply the lowest-ranked pair present, which gives the same answer faster.

## The exercise

You write `get_pair_counts`, `merge_pair`, `train_bpe`, `apply_merges_to_word`, `encode`, and `decode`. The splitter `_pretokenize` is given. The checks verify pair counts, the left-to-right merge including `[5, 5, 5]`, and that `train_bpe` on `aaabdaaabac` returns the three merges from this chapter, breaks ties to the smaller pair, never merges across words, and stops when no pair is left. They also verify the one-word replay, that reversing the merge list gives 7 tokens instead of 5, that `decode` handles a character split across two tokens, and that text round-trips after training, including accents, Chinese characters, and emoji.
