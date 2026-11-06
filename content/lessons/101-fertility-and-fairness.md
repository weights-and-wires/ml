---
title: "Measuring a tokeniser: fertility, fairness, and where it breaks"
slug: 101-fertility-and-fairness
part: "Part 7 — Large language models"
summary: Measure a tokeniser in tokens per word and per character, see why one trained on English over-charges other scripts, and find where it breaks.
minutes: 40
gpu: false
objectives:
  - Explain why text is paid for in tokens, and compute fertility and tokens per character for a text.
  - Show with numbers that a tokeniser trained on one language over-charges another, and what that costs in context window and price.
  - Explain why a fixed budget of merges has to be shared between languages.
  - Show how the cut points of a digit string depend on the training text, and why that hurts arithmetic.
prereqs:
  - 099-bytes-and-the-vocabulary-trade
  - 100-byte-pair-encoding
exercise: 101-fertility-and-fairness
---

# Measuring a tokeniser: fertility, fairness, and where it breaks

Take two sentences with the same meaning: `the cat sat on the mat.` and `बिल्ली चटाई पर बैठी थी।`. A tokeniser trained on a small English corpus cuts the first into 9 tokens and the second into 61. The same meaning costs about seven times as much. Why does that happen, and why should anyone care?

Without a measurement you cannot see the gap. A language model is charged by the token, its context window is counted in tokens, and every token costs the same amount of computation. So a reader of the second sentence pays more for the same idea, and fits less into the window. This chapter defines the measurements, shows where the gap comes from with the tokeniser you built, and then finds the places where tokenisers break: digits and runs of spaces.

## What you already know

Four ideas carry this chapter:

- Training learns an ordered merge list, and encoding replays it (see "Byte-pair encoding: train it, then use it").
- Text becomes UTF-8 bytes first, and a character can take up to four bytes (see "Why bytes, and the trade between vocabulary and length").
- Attention cost grows with the square of the number of tokens, and the output layer scores every vocabulary row for every token.
- A vocabulary slot is scarce, so merges go where the training text has the most repetition.

## Tokens are the unit of cost

Three costs are counted in tokens, not in words or letters. The price of a call to a language model is per token. The context window, the amount of text the model can read at once, is a number of tokens. And the compute for each token is the same, so more tokens means more compute. A tokeniser that uses fewer tokens for the same text is cheaper, faster, and fits more in the window.

## Fertility: tokens per word

Fertility is the average number of tokens a tokeniser spends on one word. A lower value is better. Count words with a plain whitespace split, which is separate from the pieces `_pretokenize` makes:

$$
\text{fertility} = \frac{\text{number of tokens}}{\text{number of words}}
$$

Read it aloud: divide the length of the encoded list by the number of whitespace-separated words. With no merges, the text `the quick fox` is 13 bytes and 3 words, so fertility is $13 / 3 = 4.33$. In code it is one division:

```python
ratio = len(encode(text, merges)) / len(text.split())
```

Figure 1 shows what training does to it. The curve is the fertility of the sentence `the dog and the cat were singing in the garden` as merges are learned from a short English corpus.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Fertility of an English sentence against the number of merges, falling steeply then flattening near 1.1">
<line x1="70" y1="250" x2="570" y2="250" class="stroke-grid"/>
<text x="62" y="254" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="208" x2="570" y2="208" class="stroke-grid"/>
<text x="62" y="212" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="166" x2="570" y2="166" class="stroke-grid"/>
<text x="62" y="170" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="124" x2="570" y2="124" class="stroke-grid"/>
<text x="62" y="128" text-anchor="end" class="fig-small">3</text>
<line x1="70" y1="82" x2="570" y2="82" class="stroke-grid"/>
<text x="62" y="86" text-anchor="end" class="fig-small">4</text>
<line x1="70" y1="40" x2="570" y2="40" class="stroke-grid"/>
<text x="62" y="44" text-anchor="end" class="fig-small">5</text>
<line x1="70" y1="250" x2="70" y2="255" class="stroke-muted"/>
<text x="70" y="270" text-anchor="middle" class="fig-small">0</text>
<line x1="215.349" y1="250" x2="215.349" y2="255" class="stroke-muted"/>
<text x="215.349" y="270" text-anchor="middle" class="fig-small">50</text>
<line x1="360.698" y1="250" x2="360.698" y2="255" class="stroke-muted"/>
<text x="360.698" y="270" text-anchor="middle" class="fig-small">100</text>
<line x1="506.047" y1="250" x2="506.047" y2="255" class="stroke-muted"/>
<text x="506.047" y="270" text-anchor="middle" class="fig-small">150</text>
<line x1="70" y1="250" x2="570" y2="250" class="stroke-muted"/>
<line x1="70" y1="250" x2="70" y2="40" class="stroke-muted"/>
<polyline points="70.0,56.8 81.6,98.8 93.3,119.8 104.9,136.6 116.5,145.0 128.1,149.2 139.8,153.4 151.4,157.6 163.0,157.6 174.7,166.0 186.3,166.0 197.9,166.0 209.5,170.2 221.2,170.2 232.8,178.6 244.4,178.6 256.0,182.8 267.7,182.8 279.3,187.0 290.9,187.0 302.6,187.0 314.2,187.0 325.8,187.0 337.4,187.0 349.1,191.2 360.7,191.2 372.3,191.2 384.0,191.2 395.6,195.4 407.2,195.4 418.8,195.4 430.5,195.4 442.1,195.4 453.7,199.6 465.3,199.6 477.0,199.6 488.6,199.6 500.2,203.8 511.9,203.8 523.5,203.8 535.1,203.8 546.7,203.8 558.4,203.8 570.0,203.8" class="stroke-b"/>
<text x="320" y="292" text-anchor="middle">number of merges learned</text>
<text x="20" y="30" text-anchor="start" class="fig-small">tokens per word</text>
<circle cx="70.0" cy="56.8" r="4" class="fill-b"/>
<text x="80" y="48.8" text-anchor="start" class="fig-small fig-mono">4.6</text>
<circle cx="128.1" cy="149.2" r="4" class="fill-b"/>
<text x="138.14" y="141.2" text-anchor="start" class="fig-small fig-mono">2.4</text>
<circle cx="360.7" cy="191.2" r="4" class="fill-b"/>
<text x="370.698" y="183.2" text-anchor="start" class="fig-small fig-mono">1.4</text>
</svg>
Figure 1. Fertility of an English sentence as the tokeniser learns more merges. It falls steeply, from 4.6 with no merges to 2.4 after 20, and then flattens near 1.1. Each extra merge buys less than the one before.
```

The first merges are the most common pairs, so they remove many tokens each. Later merges are rarer, and each removes few. That is diminishing returns on vocabulary. The curve also depends on the text: after 80 merges the sentence `the quick fox` still costs 10 tokens, a fertility of 3.33, because `quick` and `fox` never appeared in the corpus and have nothing to shrink them.

## Fairness: the same meaning at a different price

Fertility counts whitespace-separated words, so it misleads for a script such as Chinese that puts no spaces between words, and a word in one language does not carry the same amount of meaning as a word in another. A steadier measure is tokens per character:

$$
\text{tokens per character} = \frac{\text{number of tokens}}{\text{number of characters}}
$$

Read it aloud: how many tokens each character costs on average. The two sentences each have 23 characters. After 80 merges learned from English, the English sentence takes 9 tokens, or 0.39 per character. The Hindi sentence takes 61, or 2.65 per character, about 6.8 times as much. Figure 2 shows both.

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
Figure 2. Each box is one token. The English-trained tokeniser spends 9 tokens on the English sentence and 61 on the Hindi one, which has the same meaning. Training the same 80 merges on both languages cuts the Hindi cost to 26 tokens.
```

The reason is in the bytes. A Devanagari character takes three bytes in UTF-8, and the English corpus never repeated those bytes, so no merge ever glued them. The Hindi sentence stayed as 61 raw bytes. Real tokenisers trained on some Hindi text do far better than this toy, but measurements on production tokenisers show the same direction: a language with less training text costs more tokens.

What the gap costs: a window of 4096 tokens holds about 2,730 English words at 1.5 tokens per word, and about 336 Hindi words at 12.2. A call is priced per token, so the Hindi sentence costs about 6.8 times as much. And a model that sees text as many tiny pieces tends to have a harder job learning which pieces belong together.

Can you fix it by training on both languages? Partly. With the same 80 merges learned from English and Hindi together, the Hindi sentence drops from 61 tokens to 26, and the English sentence rises from 9 to 11. The vocabulary has a fixed number of slots, so every merge spent on Hindi is one not spent on English. Both sentences also appear in their language's training text, which flatters the tokeniser, so read these numbers as a toy-sized version of the effect. Fairness is a budget you divide, and the way to improve it is to choose the training mix on purpose.

> [!KEY]
> Text is paid for in tokens, so fertility and tokens per character are the measurements that matter. A tokeniser spends its merges on the text it was trained on, so another language gets cut into many small pieces and pays more for the same meaning, in price and in context window. A fixed number of merges means the mix of training languages is a decision.

## Where it breaks: digits and spaces

The cut points of a number depend on what the tokeniser saw. Figure 3 cuts the string `1234` under three training texts.

```figure
<svg viewBox="0 0 620 290" role="img" aria-label="The number 1234 cut into one token, two tokens, or four tokens depending on the training text">
<text x="20" y="28" text-anchor="start" class="fig-bold">Trained on 1234 many times</text>
<rect x="40" y="40" width="218" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="149" y="63" text-anchor="middle" class="fig-mono">1234</text>
<text x="300" y="62" text-anchor="start" class="fig-small">1 token</text>
<text x="20" y="108" text-anchor="start" class="fig-bold">Trained on 12 and 34 apart</text>
<rect x="40" y="120" width="106" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="93" y="143" text-anchor="middle" class="fig-mono">12</text>
<rect x="152" y="120" width="106" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="205" y="143" text-anchor="middle" class="fig-mono">34</text>
<text x="300" y="142" text-anchor="start" class="fig-small">2 tokens</text>
<text x="20" y="188" text-anchor="start" class="fig-bold">Trained on English only</text>
<rect x="40" y="200" width="50" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="65" y="223" text-anchor="middle" class="fig-mono">1</text>
<rect x="96" y="200" width="50" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="121" y="223" text-anchor="middle" class="fig-mono">2</text>
<rect x="152" y="200" width="50" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="177" y="223" text-anchor="middle" class="fig-mono">3</text>
<rect x="208" y="200" width="50" height="34" rx="4" class="fill-soft-b stroke-b"/>
<text x="233" y="223" text-anchor="middle" class="fig-mono">4</text>
<text x="300" y="222" text-anchor="start" class="fig-small">4 tokens</text>
<text x="20" y="266" text-anchor="start" class="fig-small">Same four digits, three different cuts. The cuts come from the training text.</text>
</svg>
Figure 3. The same four digits, cut three ways. A tokeniser that saw `1234` many times keeps it whole. One that saw `12` and `34` apart cuts it in two. One trained on English cuts every digit.
```

Place value is the trouble. In the first tokeniser, `1234` is one token, `11234` is the two tokens `1` and `1234`, and `12345` is `1234` and `5`. So the token boundaries do not line up with thousands or tens, and the model has to learn arithmetic across cuts that move. Some tokenisers split every digit on its own for that reason.

Spaces behave the same way in code. A run of eight spaces followed by `x` is nine tokens, one per space, unless the training text had that exact run often enough to learn a merge for it. Deeply indented code can cost one token for each space.

## Other ways to choose merges

BPE picks pairs by raw count, and there are other choices. WordPiece divides the pair's count by the product of the counts of its two pieces, so two common pieces do not win only because each half is common. Unigram starts from a large vocabulary and deletes the tokens that help least until the size is right. Encoding still applies a fixed learned list in order.

> [!RECAP]
> Hold these points while you write the functions:
>
> - Fertility is tokens divided by `len(text.split())`, and tokens per character divides by `len(text)`.
> - Count digits by token: a token that holds ASCII digits reports how many.
> - With no merges, a character costs as many tokens as it has UTF-8 bytes.
> - Merges go to the training text's language, so a script it did not see costs more.

## Build it step by step

The starter already holds `train_bpe`, `encode`, and `decode` from the earlier chapters, solved. The checks call three functions:

1. `fertility(text, merges)` returns a float. Encode, take the length, and divide by the number of whitespace-split words. The usual mistake is dividing by `len(_pretokenize(text))`, which counts pieces and not words.

```python
def fertility(text, merges):
    # len(encode(text, merges)) / len(text.split())
    return ratio
```

2. `tokens_per_char(text, merges)` returns tokens divided by characters. The usual mistake is dividing by the number of bytes, which hides the extra cost of a script that takes three bytes per character.

```python
def tokens_per_char(text, merges):
    # len(encode(...)) / len(text)
    return ratio
```

3. `digit_token_lengths(text, merges)` returns a list of ints. Encode the text, look up each token's bytes with the provided `build_vocab(merges)`, count how many of those bytes are ASCII digits (48 to 57), and keep the count when it is above 0. For `1234` under the first tokeniser of Figure 3 the answer is `[4]`, and under the third it is `[1, 1, 1, 1]`.

```python
def digit_token_lengths(text, merges):
    vocab = build_vocab(merges)
    # for each id in encode(...): count bytes of vocab[id] that are b"0" .. b"9"
    return lengths
```

## Cost and scaling

The vocabulary size $V$ pulls two ways. A bigger vocabulary gives fewer tokens for the same text, which shortens the sequence and makes attention cheaper, as in the chapter "Why bytes, and the trade between vocabulary and length". But the output layer has to score every row for every token. It multiplies a vector of $d$ numbers, the model width, by a $d \times V$ matrix. Each of the $V$ outputs is a dot product, which takes $d$ multiplies and $d$ adds, so the layer costs $2Vd$ flops per token. With $d = 4096$, a vocabulary of 32,000 costs about 262 million flops per token there, and a vocabulary of 128,000 costs about 1.05 billion. For scale, a whole 8-billion-parameter model costs about 16 billion flops per token, two for each weight, so the larger output layer is under 7 percent of it. The best size balances a shorter sequence against a bigger output layer.

## The exercise

You write `fertility`, `tokens_per_char`, and `digit_token_lengths`. The checks verify fertility and tokens per character with no merges, that fertility falls as merges grow, that an English-trained tokeniser charges a Hindi sentence several times more per character, and the three cuts of `1234` from Figure 3. They also verify that the digit lengths always sum to the number of digits in the text.
