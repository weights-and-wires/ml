---
title: "Why bytes, and the trade between vocabulary and length"
slug: 099-bytes-and-the-vocabulary-trade
part: "Part 7 — Large language models"
summary: A transformer reads integers, so text must be cut into pieces, and bytes are the one alphabet that no text can fall outside.
minutes: 35
gpu: false
objectives:
  - Explain why cutting text into letters makes sequences long and cutting it into words makes the vocabulary huge, using numbers for a 1000-word paragraph.
  - Compute the share of words a fixed word list cannot represent, and explain why bytes never have this problem.
  - Show how UTF-8 turns any character into one to four numbers between 0 and 255, and round-trip text through bytes.
  - Explain why decoding must happen on the whole byte list and not piece by piece.
prereqs:
  - 090-attention-backward-and-cost
  - 098-cost-of-training
exercise: 099-bytes-and-the-vocabulary-trade
---

# Why bytes, and the trade between vocabulary and length

You have a paragraph of 1000 words and a transformer to feed it to. The transformer cannot read letters. It reads integers, one per position, and each integer picks one row from a table of learned vectors. So something has to cut the paragraph into pieces and give each piece a number before any learning starts. The pieces are called tokens, the list of every allowed piece is the vocabulary, and the program that does the cutting is the tokeniser.

The cutting rule is a real choice, and both obvious choices are expensive. Cut into single letters and the vocabulary is tiny, but the sequence is very long, and attention compares every position with every other one. Cut into whole words and the sequence is short, but the table needs hundreds of thousands of rows, and a word that is missing from it has no number at all. Without a better idea you pay one price or the other.

This chapter puts numbers on both prices. It then shows why bytes are the safe alphabet to start from, because no text can fall outside it. The next chapters build the middle path on top of that base.

## What you already know

Four ideas carry this chapter:

- Attention compares every position with every other, so its cost grows with the square of the sequence length (see "The gradient and the cost of attention").
- Training cost grows with the number of tokens the model reads (see "The cost of a run: flops, memory, and number formats").
- An embedding table has one row of numbers for each id, and the model looks a token up by its id.
- A byte is an integer from 0 to 255, and Python's `str.encode` turns text into bytes.

## Two ways to cut, and what each costs

Start with a picture. Figure 1 cuts the same 1000-word paragraph three ways and measures three costs for each cut.

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
Figure 1. The same 1000-word paragraph cut three ways. Bytes need the smallest table and the longest sequence. Words need the shortest sequence and a table about 2000 times larger. Byte-pair encoding, the idea of the next chapters, lands between the two. The word and token counts are typical for English, not measured on one text.
```

An English word averages about 4.7 letters, and a space follows it, so 1000 words are about 5,700 bytes. Cut into bytes and the model reads 5,700 positions. Cut into words and it reads 1000. Because attention scores every pair of positions, the number of pairs is the square of the length:

$$
\text{pairs} = n^2
$$

Read it aloud: a sequence of $n$ positions has $n$ queries and $n$ keys, so $n \cdot n$ query–key pairs. Bytes give $5700^2$, about 32.5 million pairs. Words give $1000^2$, which is 1 million. The byte sequence is 5.7 times longer and costs 32.5 times as much attention. Figure 2 shows the same effect at a size you can count.

```figure
<svg viewBox="0 0 640 336" role="img" aria-label="Two squares drawn at the same scale: 8 by 8 holds 64 cells and 32 by 32 holds 1024 cells, sixteen times more for a sequence four times longer">
<rect x="60" y="120" width="64" height="64" class="fill-soft-a"/>
<line x1="60" y1="120" x2="124" y2="120" class="stroke-grid"/>
<line x1="60" y1="120" x2="60" y2="184" class="stroke-grid"/>
<line x1="60" y1="128" x2="124" y2="128" class="stroke-grid"/>
<line x1="68" y1="120" x2="68" y2="184" class="stroke-grid"/>
<line x1="60" y1="136" x2="124" y2="136" class="stroke-grid"/>
<line x1="76" y1="120" x2="76" y2="184" class="stroke-grid"/>
<line x1="60" y1="144" x2="124" y2="144" class="stroke-grid"/>
<line x1="84" y1="120" x2="84" y2="184" class="stroke-grid"/>
<line x1="60" y1="152" x2="124" y2="152" class="stroke-grid"/>
<line x1="92" y1="120" x2="92" y2="184" class="stroke-grid"/>
<line x1="60" y1="160" x2="124" y2="160" class="stroke-grid"/>
<line x1="100" y1="120" x2="100" y2="184" class="stroke-grid"/>
<line x1="60" y1="168" x2="124" y2="168" class="stroke-grid"/>
<line x1="108" y1="120" x2="108" y2="184" class="stroke-grid"/>
<line x1="60" y1="176" x2="124" y2="176" class="stroke-grid"/>
<line x1="116" y1="120" x2="116" y2="184" class="stroke-grid"/>
<line x1="60" y1="184" x2="124" y2="184" class="stroke-grid"/>
<line x1="124" y1="120" x2="124" y2="184" class="stroke-grid"/>
<rect x="60" y="120" width="64" height="64" class="stroke-a"/>
<rect x="250" y="36" width="256" height="256" class="fill-soft-a"/>
<line x1="250" y1="36" x2="506" y2="36" class="stroke-grid"/>
<line x1="250" y1="36" x2="250" y2="292" class="stroke-grid"/>
<line x1="250" y1="44" x2="506" y2="44" class="stroke-grid"/>
<line x1="258" y1="36" x2="258" y2="292" class="stroke-grid"/>
<line x1="250" y1="52" x2="506" y2="52" class="stroke-grid"/>
<line x1="266" y1="36" x2="266" y2="292" class="stroke-grid"/>
<line x1="250" y1="60" x2="506" y2="60" class="stroke-grid"/>
<line x1="274" y1="36" x2="274" y2="292" class="stroke-grid"/>
<line x1="250" y1="68" x2="506" y2="68" class="stroke-grid"/>
<line x1="282" y1="36" x2="282" y2="292" class="stroke-grid"/>
<line x1="250" y1="76" x2="506" y2="76" class="stroke-grid"/>
<line x1="290" y1="36" x2="290" y2="292" class="stroke-grid"/>
<line x1="250" y1="84" x2="506" y2="84" class="stroke-grid"/>
<line x1="298" y1="36" x2="298" y2="292" class="stroke-grid"/>
<line x1="250" y1="92" x2="506" y2="92" class="stroke-grid"/>
<line x1="306" y1="36" x2="306" y2="292" class="stroke-grid"/>
<line x1="250" y1="100" x2="506" y2="100" class="stroke-grid"/>
<line x1="314" y1="36" x2="314" y2="292" class="stroke-grid"/>
<line x1="250" y1="108" x2="506" y2="108" class="stroke-grid"/>
<line x1="322" y1="36" x2="322" y2="292" class="stroke-grid"/>
<line x1="250" y1="116" x2="506" y2="116" class="stroke-grid"/>
<line x1="330" y1="36" x2="330" y2="292" class="stroke-grid"/>
<line x1="250" y1="124" x2="506" y2="124" class="stroke-grid"/>
<line x1="338" y1="36" x2="338" y2="292" class="stroke-grid"/>
<line x1="250" y1="132" x2="506" y2="132" class="stroke-grid"/>
<line x1="346" y1="36" x2="346" y2="292" class="stroke-grid"/>
<line x1="250" y1="140" x2="506" y2="140" class="stroke-grid"/>
<line x1="354" y1="36" x2="354" y2="292" class="stroke-grid"/>
<line x1="250" y1="148" x2="506" y2="148" class="stroke-grid"/>
<line x1="362" y1="36" x2="362" y2="292" class="stroke-grid"/>
<line x1="250" y1="156" x2="506" y2="156" class="stroke-grid"/>
<line x1="370" y1="36" x2="370" y2="292" class="stroke-grid"/>
<line x1="250" y1="164" x2="506" y2="164" class="stroke-grid"/>
<line x1="378" y1="36" x2="378" y2="292" class="stroke-grid"/>
<line x1="250" y1="172" x2="506" y2="172" class="stroke-grid"/>
<line x1="386" y1="36" x2="386" y2="292" class="stroke-grid"/>
<line x1="250" y1="180" x2="506" y2="180" class="stroke-grid"/>
<line x1="394" y1="36" x2="394" y2="292" class="stroke-grid"/>
<line x1="250" y1="188" x2="506" y2="188" class="stroke-grid"/>
<line x1="402" y1="36" x2="402" y2="292" class="stroke-grid"/>
<line x1="250" y1="196" x2="506" y2="196" class="stroke-grid"/>
<line x1="410" y1="36" x2="410" y2="292" class="stroke-grid"/>
<line x1="250" y1="204" x2="506" y2="204" class="stroke-grid"/>
<line x1="418" y1="36" x2="418" y2="292" class="stroke-grid"/>
<line x1="250" y1="212" x2="506" y2="212" class="stroke-grid"/>
<line x1="426" y1="36" x2="426" y2="292" class="stroke-grid"/>
<line x1="250" y1="220" x2="506" y2="220" class="stroke-grid"/>
<line x1="434" y1="36" x2="434" y2="292" class="stroke-grid"/>
<line x1="250" y1="228" x2="506" y2="228" class="stroke-grid"/>
<line x1="442" y1="36" x2="442" y2="292" class="stroke-grid"/>
<line x1="250" y1="236" x2="506" y2="236" class="stroke-grid"/>
<line x1="450" y1="36" x2="450" y2="292" class="stroke-grid"/>
<line x1="250" y1="244" x2="506" y2="244" class="stroke-grid"/>
<line x1="458" y1="36" x2="458" y2="292" class="stroke-grid"/>
<line x1="250" y1="252" x2="506" y2="252" class="stroke-grid"/>
<line x1="466" y1="36" x2="466" y2="292" class="stroke-grid"/>
<line x1="250" y1="260" x2="506" y2="260" class="stroke-grid"/>
<line x1="474" y1="36" x2="474" y2="292" class="stroke-grid"/>
<line x1="250" y1="268" x2="506" y2="268" class="stroke-grid"/>
<line x1="482" y1="36" x2="482" y2="292" class="stroke-grid"/>
<line x1="250" y1="276" x2="506" y2="276" class="stroke-grid"/>
<line x1="490" y1="36" x2="490" y2="292" class="stroke-grid"/>
<line x1="250" y1="284" x2="506" y2="284" class="stroke-grid"/>
<line x1="498" y1="36" x2="498" y2="292" class="stroke-grid"/>
<line x1="250" y1="292" x2="506" y2="292" class="stroke-grid"/>
<line x1="506" y1="36" x2="506" y2="292" class="stroke-grid"/>
<rect x="250" y="36" width="256" height="256" class="stroke-a"/>
<text x="92" y="108" text-anchor="middle" class="fig-bold">n = 8</text>
<text x="92" y="208" text-anchor="middle">64 cells</text>
<text x="378" y="24" text-anchor="middle" class="fig-bold">n = 32</text>
<text x="378" y="320" text-anchor="middle">1,024 cells</text>
<text x="560" y="150" text-anchor="middle" class="fig-bold">4× longer</text>
<text x="560" y="172" text-anchor="middle" class="fig-bold">16× more cells</text>
</svg>
Figure 2. Every cell is one query–key pair. A sequence four times longer has sixteen times as many cells. A causal model uses about half the cells, but half of a quadratic is still quadratic.
```

So why not use letters? Two reasons. The pairs grow quickly, and a model that reads letters must also spend its early layers learning to spell, gluing letters into words that a word-level reader gets for free.

So why not use words? Look at the table side. A word list that covers most of a language has hundreds of thousands of rows, and the output layer, which scores every row for every next token, grows with it. Worse, no list is complete. Names, typos, invented words, and words like "héllo" are missing. A word that appears once or twice in training also gets a row that is barely trained.

## Count the words you cannot represent

Take a tiny word list, `{"the", "cat", "sat"}`, and the text `"the cat sat on the mat"`. The text has 6 words. Two of them, `"on"` and `"mat"`, are not in the list, so a word-level tokeniser has no id for them. The unknown rate is the share of words the list misses:

$$
\text{unknown rate} = \frac{\text{words missing from the list}}{\text{words in the text}}
$$

Read it aloud: count the words the list lacks and divide by the total. Here it is $2 / 6 = 0.33$, so a third of the sentence is invisible. In numpy and Python, two lines do it:

```python
words = text.split()
rate = sum(w not in word_vocab for w in words) / len(words)
```

The split is on whitespace and the match is exact, so `"The"` and `"the"` are different words. You can lower the miss rate by making the list bigger, but each extra row costs memory in two layers, and the rate never reaches zero.

## Bytes: an alphabet nothing falls outside

The same worry about letters returns in a sharper form. Unicode, the standard that lists written characters, holds more than 140,000 of them, and the list keeps growing. A table with a row for each character is large, and a character added tomorrow has no row.

Bytes avoid both problems. UTF-8, the usual way to store text, writes every character as one to four bytes, and every byte is a number from 0 to 255. Plain English letters take one byte. Accented letters take two, many scripts take three, and emoji take four. Figure 3 splits four characters into their real bytes.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="The characters h, é, 世 and 🎉 each split into one, two, three, and four byte boxes with their numeric values">
<rect x="40" y="40" width="48" height="44" rx="6" class="fill-soft-a stroke-a"/>
<text x="64" y="69" text-anchor="middle" class="fig-bold">h</text>
<text x="64" y="32" text-anchor="middle" class="fig-small fig-mono">U+0068</text>
<line x1="64" y1="84" x2="64" y2="112" class="stroke-muted dashed"/>
<rect x="40" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="64" y="145" text-anchor="middle" class="fig-mono">104</text>
<text x="64" y="180" text-anchor="middle" class="fig-small">1 byte</text>
<rect x="92" y="40" width="100" height="44" rx="6" class="fill-soft-a stroke-a"/>
<text x="142" y="69" text-anchor="middle" class="fig-bold">é</text>
<text x="142" y="32" text-anchor="middle" class="fig-small fig-mono">U+00E9</text>
<line x1="142" y1="84" x2="142" y2="112" class="stroke-muted dashed"/>
<rect x="92" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="116" y="145" text-anchor="middle" class="fig-mono">195</text>
<rect x="144" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="168" y="145" text-anchor="middle" class="fig-mono">169</text>
<text x="142" y="180" text-anchor="middle" class="fig-small">2 bytes</text>
<rect x="196" y="40" width="152" height="44" rx="6" class="fill-soft-a stroke-a"/>
<text x="272" y="69" text-anchor="middle" class="fig-bold">世</text>
<text x="272" y="32" text-anchor="middle" class="fig-small fig-mono">U+4E16</text>
<line x1="272" y1="84" x2="272" y2="112" class="stroke-muted dashed"/>
<rect x="196" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="220" y="145" text-anchor="middle" class="fig-mono">228</text>
<rect x="248" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="272" y="145" text-anchor="middle" class="fig-mono">184</text>
<rect x="300" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="324" y="145" text-anchor="middle" class="fig-mono">150</text>
<text x="272" y="180" text-anchor="middle" class="fig-small">3 bytes</text>
<rect x="352" y="40" width="204" height="44" rx="6" class="fill-soft-a stroke-a"/>
<text x="454" y="69" text-anchor="middle" class="fig-bold">🎉</text>
<text x="454" y="32" text-anchor="middle" class="fig-small fig-mono">U+1F389</text>
<line x1="454" y1="84" x2="454" y2="112" class="stroke-muted dashed"/>
<rect x="352" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="376" y="145" text-anchor="middle" class="fig-mono">240</text>
<rect x="404" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="428" y="145" text-anchor="middle" class="fig-mono">159</text>
<rect x="456" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="480" y="145" text-anchor="middle" class="fig-mono">142</text>
<rect x="508" y="120" width="48" height="40" rx="4" class="fill-soft-c stroke-c"/>
<text x="532" y="145" text-anchor="middle" class="fig-mono">137</text>
<text x="454" y="180" text-anchor="middle" class="fig-small">4 bytes</text>
<text x="40" y="220" text-anchor="start" class="fig-small">Every box on the bottom row holds a number from 0 to 255, so the model needs only 256 symbols.</text>
</svg>
Figure 3. The characters h, é, 世, and 🎉 become 1, 2, 3, and 4 bytes. All ten bytes are numbers between 0 and 255, so the model needs only 256 symbols however exotic the text is.
```

That is the whole argument for bytes. The vocabulary has exactly 256 ids, so the unknown rate is 0 for every text in every language. And the mapping reverses: join the bytes and decode them, and the original text returns.

```python
ids = list("hé世🎉".encode("utf-8"))   # [104, 195, 169, 228, 184, 150, 240, 159, 142, 137]
text = bytes(ids).decode("utf-8")       # "hé世🎉"
```

The price is length. A character that takes three bytes takes three positions. The chapter "Measuring a tokeniser: fertility, fairness, and where it breaks" returns to what that costs.

> [!WARNING]
> Decode the whole list of bytes in one call. Cut the list in the middle of a character and `decode` fails, because half of a character is not valid text. The chapter "Byte-pair encoding: train it, then use it" shows how this trap shows up once tokens are bigger than a byte.

## The middle path

Bytes guarantee coverage, and words give short sequences. You want both. The idea is to start from the 256 bytes, so nothing is ever unknown, and then glue together the neighbours that appear together often, so the common words shrink to one or two pieces. A sequence of 5,700 bytes becomes about 1,300 tokens, with a table of tens of thousands of rows instead of hundreds of thousands. That is byte-pair encoding, and you build it in the next two chapters.

> [!KEY]
> To feed a 1000-word paragraph to a transformer you must cut it into pieces and number them. Letters or bytes give 256 table rows but about 5,700 positions and 32 million attention pairs. Words give about 1000 positions but a table of hundreds of thousands of rows, and some words are missing. Bytes cover every text, because UTF-8 turns every character into numbers from 0 to 255. The way to get short sequences as well is to merge frequent byte neighbours.

> [!RECAP]
> Hold these four points while you write the functions:
>
> - Attention pairs grow as $n^2$, so a sequence five times longer costs 25 times as much.
> - A word list has hundreds of thousands of rows and still misses words, so its unknown rate is above 0.
> - UTF-8 gives every character one to four bytes, each between 0 and 255, so a byte vocabulary has no unknown.
> - Encode with `.encode("utf-8")` and decode the whole list with one `.decode("utf-8")`.

## Build it step by step

The checks call three functions in this order:

1. `text_to_bytes(text)` takes a `str` and returns a list of ints from 0 to 255. Encode to UTF-8 and convert the result to a list. The usual mistake is calling `ord` on each character, which gives code points up to 1,114,111 instead of bytes. The shape of the code is:

```python
def text_to_bytes(text):
    # encode to utf-8, then list(...) so each element is an int
    return ids
```

2. `bytes_to_text(ids)` takes a list of ints and returns the string. Build a `bytes` object from the whole list, then decode it once. The usual mistake is `chr(i)` for each id, which breaks for every character that needs more than one byte.

```python
def bytes_to_text(ids):
    # bytes(ids) then one .decode("utf-8")
    return text
```

3. `unknown_rate(text, word_vocab)` takes a string and a set of words and returns a float. Split on whitespace, count the words missing from the set, and divide by the number of words. Return `0.0` for a text with no words, so the empty string does not divide by zero.

```python
def unknown_rate(text, word_vocab):
    # words = text.split(); count misses; divide, guarding the empty case
    return rate
```

## Where this breaks

Bytes remove the unknown word, and they cost length. English text takes one position per letter and a script with three-byte characters takes three. The attention bill in Figure 2 grows with the square of that length, which is why a tokeniser built on bytes must also merge them. The counts in Figure 1 for words and for byte-pair tokens are typical for English, and they change with the text and the vocabulary size. Treat them as orders of magnitude.

## The exercise

You write `text_to_bytes`, `bytes_to_text`, and `unknown_rate`. The checks verify the exact bytes of `"hé世🎉"`, that every id for random Unicode text lies between 0 and 255, and that text round-trips through bytes. They also verify unknown rates on the worked example, the empty text, and the extremes of a full list and an empty one.
