---
title: "Tokenisation: bytes, byte-pair encoding, and what a tokeniser charges"
slug: 049-tokenisation
part: "Part 7 — Large language models"
summary: Cut text into bytes, merge frequent pairs into a vocabulary, replay the merges to encode, and measure what the tokeniser charges.
minutes: 80
gpu: false
objectives:
  - Explain why letters make sequences long and words make the vocabulary huge, and why bytes cover every text.
  - Train byte-pair encoding by hand and in code, and explain why merge i gets id 256 plus i and why merges stay inside words.
  - Explain why encoding replays the frozen merges in the order they were learned, and why decoding joins all the bytes before it decodes UTF-8.
  - Compute fertility and tokens per character, and show why a tokeniser trained on one language over-charges another.
  - Show how the cut points of a digit string depend on the training text, and why that hurts arithmetic.
prereqs:
  - 044-attention-backward-and-cost
  - 048-training-loop-and-cost
exercise: 049-tokenisation
---

# Tokenisation: bytes, byte-pair encoding, and what a tokeniser charges

You have a paragraph of 1000 words and a transformer to feed it to. The transformer cannot read letters. It reads integers, one per position, and each integer picks one row from a table of learned vectors. So something must cut the paragraph into pieces and number them before any learning starts. The pieces are tokens, the list of every allowed piece is the vocabulary, and the program that cuts is the tokeniser.

Both obvious cuts are expensive. Single letters give a tiny vocabulary but a very long sequence, and attention compares every position with every other one. Whole words give a short sequence but a table of hundreds of thousands of rows, and a missing word has no number at all. The way out is to start from bytes, so nothing is ever unknown, and glue together the neighbours that appear together often.

This chapter builds that tokeniser and then audits it. You count what each cut costs, train byte-pair encoding on a string small enough to follow by hand, and turn text into ids and back without losing a character. Then you ask who benefits: a tokeniser trained on English cuts `the cat sat on the mat.` into 9 tokens and its Hindi translation into 61, so the same meaning costs about seven times as much. Why, and how do you measure it?

## What you already know

- Attention cost grows with the square of the sequence length (see [the cost of attention](/lessons/044-attention-backward-and-cost)).
- Training cost grows with the number of tokens the model reads, about $6ND$ flops (see [the cost of a run](/lessons/048-training-loop-and-cost)).
- An embedding table has one row for each id, and a byte is an integer from 0 to 255.

## Two ways to cut, and what each costs

Figure 1 cuts the same 1000-word paragraph three ways and measures three costs for each cut.

```figure
<svg viewBox="0 0 640 340" role="img" aria-label="Bars comparing bytes, whole words, and BPE tokens for a 1000-word paragraph on sequence length, vocabulary rows, and attention pairs">
<text x="20" y="24" text-anchor="start" class="fig-bold">Positions the model reads (sequence length)</text>
<text x="82" y="48" text-anchor="end">bytes</text>
<rect x="90" y="34" width="300" height="18" class="fill-a"/>
<text x="398" y="48" text-anchor="start" class="fig-small fig-mono">5,700</text>
<text x="82" y="72" text-anchor="end">words</text>
<rect x="90" y="58" width="52.6316" height="18" class="fill-b"/>
<text x="150.632" y="72" text-anchor="start" class="fig-small fig-mono">1,000</text>
<text x="82" y="96" text-anchor="end">BPE</text>
<rect x="90" y="82" width="68.4211" height="18" class="fill-c"/>
<text x="166.421" y="96" text-anchor="start" class="fig-small fig-mono">1,300</text>
<text x="20" y="132" text-anchor="start" class="fig-bold">Rows in the embedding table (log scale)</text>
<text x="82" y="156" text-anchor="end">bytes</text>
<rect x="90" y="142" width="120.412" height="18" class="fill-a"/>
<text x="218.412" y="156" text-anchor="start" class="fig-small fig-mono">256</text>
<text x="82" y="180" text-anchor="end">words</text>
<rect x="90" y="166" width="284.949" height="18" class="fill-b"/>
<text x="382.949" y="180" text-anchor="start" class="fig-small fig-mono">500,000</text>
<text x="82" y="204" text-anchor="end">BPE</text>
<rect x="90" y="190" width="225.257" height="18" class="fill-c"/>
<text x="323.257" y="204" text-anchor="start" class="fig-small fig-mono">32,000</text>
<text x="20" y="240" text-anchor="start" class="fig-bold">Query-key pairs in attention (length squared)</text>
<text x="82" y="264" text-anchor="end">bytes</text>
<rect x="90" y="250" width="300" height="18" class="fill-a"/>
<text x="398" y="264" text-anchor="start" class="fig-small fig-mono">32.5 million</text>
<text x="82" y="288" text-anchor="end">words</text>
<rect x="90" y="274" width="9.23361" height="18" class="fill-b"/>
<text x="107.234" y="288" text-anchor="start" class="fig-small fig-mono">1.0 million</text>
<text x="82" y="312" text-anchor="end">BPE</text>
<rect x="90" y="298" width="15.6048" height="18" class="fill-c"/>
<text x="113.605" y="312" text-anchor="start" class="fig-small fig-mono">1.7 million</text>
</svg>
Figure 1. The same 1000-word paragraph cut three ways. Bytes need the smallest table and the longest sequence. Words need the shortest sequence and a table about 2000 times larger. Byte-pair encoding, built later in this chapter, lands between the two. The word and token counts are typical for English, not measured on one text.
```

An English word averages about 4.7 letters and a space, so 1000 words are about 5,700 bytes. Bytes give the model 5,700 positions and words give it 1000. Because attention scores every pair of positions, the number of pairs is the square of the length:

$$
\text{pairs} = n^2
$$

Read it aloud: a sequence of $n$ positions has $n$ queries and $n$ keys, so $n \cdot n$ query–key pairs. Bytes give $5700^2$, about 32.5 million pairs. Words give $1000^2$, which is 1 million. The byte sequence is 5.7 times longer and costs 32.5 times as much attention.

Letters also make the model spend its early layers learning to spell. A word list that covers most of a language has hundreds of thousands of rows, and the output layer, which scores every row for every next token, grows with it. No list is complete either: names, typos, and words like "héllo" are missing, and a word seen once or twice gets a barely trained row.

Take a tiny word list, `{"the", "cat", "sat"}`, and the text `"the cat sat on the mat"`. The text has 6 words. Two of them, `"on"` and `"mat"`, are not in the list, so a word-level tokeniser has no id for them. The unknown rate is the share of words the list misses:

$$
\text{unknown rate} = \frac{\text{words missing from the list}}{\text{words in the text}}
$$

Read it aloud: count the words the list lacks and divide by the total. Here it is $2 / 6 = 0.33$, so a third of the sentence is invisible. In Python:

```python
words = text.split()
rate = sum(w not in word_vocab for w in words) / len(words)
```

The match is exact, so `"The"` and `"the"` are different words. A bigger list lowers the rate, but each extra row costs memory in two layers, and the rate never reaches zero.

## Bytes: an alphabet nothing falls outside

A row for each character fails too: Unicode, the standard that lists written characters, holds more than 140,000 of them, and the list keeps growing. Bytes avoid both problems. UTF-8, the usual way to store text, writes every character as one to four bytes, and every byte is a number from 0 to 255. Plain English letters take one byte, accented letters take two, many scripts take three, and emoji take four. The character `h` is the byte 104, `é` is the two bytes 195 and 169, `世` is 228, 184, and 150, and `🎉` is 240, 159, 142, and 137. All ten bytes lie between 0 and 255, so the model needs only 256 symbols however exotic the text is.

The vocabulary has exactly 256 ids, so the unknown rate is 0 for every text in every language, and the mapping reverses:

```python
ids = list("hé世🎉".encode("utf-8"))   # [104, 195, 169, 228, 184, 150, 240, 159, 142, 137]
text = bytes(ids).decode("utf-8")       # "hé世🎉"
```

The price is length: a three-byte character takes three positions.

> [!KEY]
> Bytes give 256 table rows and cover every text, because UTF-8 turns every character into numbers from 0 to 255. The price is about 5,700 positions and 32 million attention pairs for a 1000-word paragraph, against 1000 positions for words, whose table needs hundreds of thousands of rows and still misses some words. To get short sequences as well, merge frequent byte neighbours.

## Shorthand for repeated neighbours

Suppose you had to copy the string `aaabdaaabac` a thousand times by hand, and you could invent a few shorthand symbols. You would give one symbol to the neighbours that appear together most often, then look again, because each shorthand has neighbours of its own. A tokeniser faces the same job, and each shorthand costs one row in the vocabulary, so you want the one that saves the most writing. Figure 2 shows `aaabdaaabac` as 11 cells, then the strip after each merge. The orange cells are the pair about to be glued.

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
Figure 2. Three merges shrink 11 tokens to 5. In each strip the orange cells are the pair about to be merged. Notice that the first strip has three a's in a row, and only the first two are glued, so the third one stays alone.
```

Why the most frequent pair? A merge adds one row to the vocabulary, and a pair that appears $k$ times loses about $k$ tokens when glued, so the biggest count is the best use of the row. Larger chunks arise on their own: once `aa` and `ab` are tokens, the pair of those two tokens is `aaab`.

The count is a stand-in for the saving, and it can be off. On the 11 bytes, where `a` is 97, `b` is 98, `c` is 99, and `d` is 100, the pair `(a, a)` appears 4 times and `(a, b)` twice. But each `aaa` run overlaps itself, and you can glue only one of its two pairs, so merging `(a, a)` saves 2 tokens, the same as `(a, b)`. Training still picks by count, because counting is quick and the count is right when pairs do not overlap.

To merge, scan from the left, and when two neighbours match, write one id for the pair and skip both. The word `aaab`, which is `[97, 97, 97, 98]`, becomes `[id, 97, 98]`: the third 97 has no partner left. The merged id must not clash with a byte (0 to 255), so the merge at position $i$ in the list gets:

$$
\text{id}(\text{merge } i) = 256 + i
$$

Read it aloud: the id is the first free number, 256, plus how many merges came before. The position in the list already says the id, so you store nothing else.

Training repeats the step: count pairs, take the winner, merge it everywhere, record it. On `aaabdaaabac`:

- Round 0: `(97, 97)` wins with 4 and becomes 256, leaving 9 tokens.
- Round 1: `(256, 97)` and `(97, 98)` each appear twice, a tie. The rule is to pick the smaller pair as a tuple, so `(97, 98)` wins and becomes 257, the bytes `ab`. The strip is 7 tokens: `256, 257, 100, 256, 257, 97, 99`.
- Round 2: `(256, 257)` appears twice and wins. It becomes 258, the bytes `aaab`, and the strip is `258, 100, 258, 97, 99`, 5 tokens.

Two runs must agree on ties, so a fixed rule is needed; "smaller tuple" is this exercise's convention. The merge list is `[(97, 97), (97, 98), (256, 257)]`, and each id's bytes come from joining its parts: `vocab[256 + i] = vocab[a] + vocab[b]`. The tokeniser is the ordered merge list plus the 256 byte entries, and you can throw the training text away.

> [!KEY]
> Each round counts adjacent pairs, merges the most common one into the next free id, and records it. The ordered merge list is the tokeniser. Ties go to the smaller pair, so every run agrees.

On a raw stream of bytes, spaces become part of the pairs, and a common pair in English is a letter followed by a space. Gluing the end of one word to the space before the next wastes a row, because the token says nothing about which word it came from. So training splits the text into pieces first, and counts and merges only inside each piece. The exercise gives you the splitter `_pretokenize`. A piece is a word with at most one leading space, so `" cat"` is one piece, and punctuation stays attached, so `" dog."` is one piece. Joined together, the pieces give the original text back.

## Replay the merges to encode

To encode a sentence the tokeniser has never seen, you could rerun training on it, but the ids are a fixed code that the embedding table was trained on, so the word `cat` must get the same ids in every document. The list is frozen, and encoding only replays it: for each piece, take its bytes, apply merge 0, then merge 1, and so on, then join the pieces' ids into one flat list.

Take `aaabdaaabac` as one piece of 11 bytes. Merge 0 glues `(97, 97)` into 256 and leaves 9 tokens. Merge 1 glues `(97, 98)` into 257 and leaves 7. Merge 2 glues `(256, 257)` into 258 and leaves 5: `[258, 100, 258, 97, 99]`. The left half of Figure 3 shows each step.

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

Why must the order match? Merge 2 glues the ids 256 and 257, which exist only after merges 0 and 1 have run. More generally, merge $i$ was chosen on text that merges 0 to $i-1$ had already rewritten, so replaying in order rebuilds that text, and each merge finds the pairs it was chosen for.

Reverse the list to `[(256, 257), (97, 98), (97, 97)]` and encode the same string. The first merge looks for ids 256 and 257, finds none, and does nothing. The second glues `(97, 98)` into id 257, and the third glues `(97, 97)` into id 258. The result is `[258, 257, 100, 258, 257, 97, 99]`, 7 tokens instead of 5, as the right half of Figure 3 shows. Length is the smaller problem: in the reversed list 258 means `aa`, and decoded with the original vocabulary, where 258 is `aaab`, those ids give `aaababdaaababac`, not the text you started with. Order is part of the meaning of the ids. Because pieces are split before any merge, `encode("the cat")` equals `encode("the")` followed by `encode(" cat")`.

> [!KEY]
> A trained tokeniser is a frozen recipe. Encode runs the merges in the order they were learned, because merge $i$ was chosen on text that merges 0 to $i-1$ had already rewritten. It never recounts, so the same word gets the same ids in every document.

## Decode: join, then decode once

Every id stands for bytes. The vocabulary maps an id to its bytes, so decoding is a lookup, a join, and one decode:

```python
text = b"".join(vocab[i] for i in ids).decode("utf-8")
```

Join first, and decode once. A token is a slice of bytes, and a slice can cut a character in half, because a token boundary does not know about characters. If one token held the first two bytes of 🎉, 240 and 159, then `bytes([240, 159]).decode()` would raise `UnicodeDecodeError`. Joining all four bytes and decoding once returns the emoji.

> [!WARNING]
> Never call `.decode("utf-8")` on one token at a time. A token can hold half of a character, and half of a character is not valid text. Join all the bytes first.

The round trip is exact. Each merge replaces a pair by an id whose bytes are those two parts joined, so the bytes of the whole sequence never change while you encode. So `decode(encode(text)) == text` for every text, with merges or without, including accents, Chinese characters, and emoji.

## Measure what a tokeniser charges

The price of a call to a language model is per token, the context window is a number of tokens, and the compute for each token is the same. A tokeniser that uses fewer tokens for the same text is cheaper, faster, and fits more in the window. Fertility is the average number of tokens a tokeniser spends on one word, and lower is better. Count words with a plain whitespace split, not with the pieces `_pretokenize` makes:

$$
\text{fertility} = \frac{\text{number of tokens}}{\text{number of words}}
$$

Read it aloud: divide the length of the encoded list by the number of whitespace-separated words. With no merges, the text `the quick fox` is 13 bytes and 3 words, so fertility is $13 / 3 = 4.33$. In code it is one division:

```python
ratio = len(encode(text, merges)) / len(text.split())
```

Training lowers it with diminishing returns, because the first merges are the most common pairs. For `the dog and the cat were singing in the garden`, fertility falls from 4.6 with no merges to 2.4 after 20 merges, and then flattens near 1.1. Words the corpus never held do not shrink: after 80 merges `the quick fox` still costs 10 tokens, a fertility of 3.33.

Fertility misleads for a script such as Chinese that puts no spaces between words. A steadier measure is tokens per character:

$$
\text{tokens per character} = \frac{\text{number of tokens}}{\text{number of characters}}
$$

Read it aloud: how many tokens each character costs on average. The two sentences from the opening each have 23 characters. After 80 merges learned from English, the English sentence takes 9 tokens, or 0.39 per character. The Hindi sentence takes 61, or 2.65 per character, about 6.8 times as much. Figure 4 shows both, and what changes when the Hindi shares the merges.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="Three rows of token boxes: nine for the English sentence, sixty-one for the Hindi sentence with an English-trained tokeniser, and twenty-six when trained on both">
<text x="20" y="24" text-anchor="start" class="fig-bold">English sentence, English-trained tokeniser: 9 tokens, 1.5 per word</text>
<rect x="20" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="30" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="40" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="50" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="60" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="70" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="80" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="90" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<rect x="100" y="34" width="9" height="26" class="fill-soft-a stroke-a"/>
<text x="20" y="96" text-anchor="start" class="fig-bold">Hindi sentence, English-trained tokeniser: 61 tokens, 12.2 per word</text>
<rect x="20" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="30" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="40" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="50" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="60" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="70" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="80" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="90" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="100" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="110" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="120" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="130" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="140" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="150" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="160" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="170" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="180" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="190" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="200" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="210" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="220" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="230" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="240" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="250" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="260" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="270" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="280" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="290" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="300" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="310" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="320" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="330" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="340" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="350" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="360" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="370" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="380" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="390" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="400" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="410" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="420" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="430" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="440" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="450" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="460" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="470" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="480" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="490" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="500" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="510" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="520" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="530" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="540" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="550" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="560" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="570" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="580" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="590" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="600" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="610" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="620" y="106" width="9" height="26" class="fill-soft-b stroke-b"/>
<text x="20" y="168" text-anchor="start" class="fig-bold">Hindi sentence, tokeniser trained on both: 26 tokens, 5.2 per word</text>
<rect x="20" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="30" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="40" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="50" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="60" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="70" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="80" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="90" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="100" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="110" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="120" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="130" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="140" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="150" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="160" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="170" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="180" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="190" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="200" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="210" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="220" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="230" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="240" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="250" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="260" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<rect x="270" y="178" width="9" height="26" class="fill-soft-b stroke-b"/>
<text x="20" y="240" text-anchor="start" class="fig-small">Each box is one token. Same meaning in every row.</text>
</svg>
Figure 4. Each box is one token. The English-trained tokeniser spends 9 tokens on the English sentence and 61 on the Hindi one, which has the same meaning. Training the same 80 merges on both languages cuts the Hindi cost to 26 tokens.
```

The reason is in the bytes. A Devanagari character takes three bytes in UTF-8, and the English corpus never repeated them, so no merge glued them, and the Hindi sentence stayed as 61 raw bytes. Real tokenisers trained on some Hindi do far better, but they show the same direction: a language with less training text costs more tokens. A window of 4096 tokens holds about 2,730 English words at 1.5 tokens per word, and about 336 Hindi words at 12.2. A call is priced per token, so the Hindi sentence costs about 6.8 times as much.

Can you fix it by training on both languages? Partly. With the same 80 merges learned from English and Hindi together, the Hindi sentence drops from 61 tokens to 26, and the English sentence rises from 9 to 11. The vocabulary has a fixed number of slots, so every merge spent on Hindi is one not spent on English. Both sentences also appear in their training text, which flatters the tokeniser, so read these numbers as a toy-sized version of the effect.

> [!KEY]
> Text is paid for in tokens, so fertility and tokens per character are the measurements that matter. A tokeniser spends its merges on the text it was trained on, so another language is cut into many small pieces and pays more for the same meaning, in price and in context window. A fixed number of merges means the mix of training languages is a decision.

## Digits and spaces: where the cuts move

The cut points of a number depend on what the tokeniser saw. The string `1234` is one token for a tokeniser that saw `1234` many times, two tokens for one that saw `12` and `34` apart, and four tokens for one trained on English alone. In the first, `11234` is the two tokens `1` and `1234`, and `12345` is `1234` and `5`, so the boundaries do not line up with thousands or tens, and the model must learn arithmetic across cuts that move. Some tokenisers split every digit for that reason. Spaces behave the same way in code: eight spaces followed by `x` is nine tokens unless the training text taught a merge for that run.

> [!RECAP]
> - Bytes are a vocabulary of 256 ids, so the unknown rate is 0.
> - Scan left to right and skip two after a hit, so `[5, 5, 5]` becomes `[20, 5]`. Merge $i$ gets id $256 + i$, a tie goes to the smaller pair, and counting and merging stay inside each piece.
> - Encode replays the merges in list order on each piece and never recounts. Decode joins all the bytes, then calls `.decode("utf-8")` once.
> - Fertility is tokens divided by `len(text.split())`, tokens per character divides by `len(text)`, and merges go to the training text's language.

## Build it step by step

The checks call eleven functions, in this order. The file already holds `_pretokenize` and `build_vocab`.

1. **`text_to_bytes(text)`** returns the UTF-8 bytes as a list of ints. The usual mistake is calling `ord` on each character, which gives code points up to 1,114,111 instead of bytes.
2. **`unknown_rate(text, word_vocab)`** splits on whitespace, counts the words missing from the set, and divides by the number of words. Return `0.0` for a text with no words.
3. **`get_pair_counts(token_lists)`** returns a dict from pair tuples to counts. Walk each inner list on its own with `zip(tokens, tokens[1:])`. The usual mistake is joining the lists first, which counts a pair across the end of one word and the start of the next.
4. **`merge_pair(token_lists, pair, new_id)`** returns fresh lists with the pair replaced. Use a `while` loop: on a match write `new_id` and advance by 2, otherwise write the current id and advance by 1. The usual mistake is turning `[5, 5, 5]` into `[20, 20]`.
5. **`train_bpe(corpus, vocab_size)`** returns `(merges, vocab)`. Split with `_pretokenize`, turn each piece into a list of byte ints, and start the vocabulary at `{i: bytes([i]) for i in range(256)}`. Loop until the vocabulary reaches `vocab_size` or no pair is left: count, pick the winner, record its joined bytes, merge. The usual mistake is breaking ties with `max`, which picks the larger pair.
6. **`apply_merges_to_word(word_bytes, merges)`** returns the ids of one piece after the replay. For each `i, pair`, call `merge_pair([ids], pair, 256 + i)` and keep the first list back. The usual mistake is using `i` instead of `256 + i` as the id.
7. **`encode(text, merges)`** returns one flat list of ids. Split with `_pretokenize`, replay the merges on each piece's bytes, and extend a flat list. With an empty merge list the result is the raw bytes.
8. **`decode(ids, vocab)`** joins `vocab[i]` for every id, then decodes once. The usual mistake is decoding each id alone, which raises an error on a half character.
9. **`fertility(text, merges)`** divides the encoded length by the number of whitespace-split words. The usual mistake is dividing by `len(_pretokenize(text))`, which counts pieces and not words.
10. **`tokens_per_char(text, merges)`** divides the encoded length by `len(text)`. The usual mistake is dividing by the number of bytes, which hides the extra cost of a three-byte script.
11. **`digit_token_lengths(text, merges)`** encodes the text, looks up each token's bytes with the provided `build_vocab(merges)`, counts the bytes that are ASCII digits (48 to 57), and keeps the count when it is above 0. For `1234` under the first tokeniser the answer is `[4]`, and under the third it is `[1, 1, 1, 1]`.

## Cost and scaling

A bigger vocabulary $V$ gives fewer tokens for the same text, which makes attention cheaper. But the output layer multiplies a vector of $d$ numbers, the model width, by a $d \times V$ matrix, so it costs $2Vd$ flops per token. With $d = 4096$, a vocabulary of 32,000 costs about 262 million flops per token there, and 128,000 costs about 1.05 billion, still under 7 percent of the 16 billion flops per token of an 8-billion-parameter model. The best size balances a shorter sequence against a bigger output layer.

## Where this breaks

- The counts in Figure 1 for words and byte-pair tokens are typical for English, so treat them as orders of magnitude.
- Training recounts every pair after every merge, which is slow for thousands of merges. Practical tokenisers update only the counts next to each merge, and encode with a table of merge ranks instead of one pass per merge. Training is greedy: an early merge is never undone.
- BPE picks pairs by raw count. WordPiece divides the pair's count by the product of its two pieces' counts, and Unigram deletes the tokens that help least from a large vocabulary.

## The exercise

You write the eleven functions above, and `_pretokenize` and `build_vocab` are given. The checks verify the bytes of `"hé世🎉"`, the unknown rate, the pair counts, and the merge of `[5, 5, 5]`. They check that `train_bpe` on `aaabdaaabac` returns this chapter's three merges, breaks ties to the smaller pair, never merges across words, and stops when no pair is left. They check that reversing the merge list gives 7 tokens instead of 5, that `decode` handles a split character, and that text round-trips. The last checks cover fertility, tokens per character, the Hindi gap, and the three cuts of `1234`.
