---
title: "Encoding and decoding: replay the merges in order"
slug: 111-encode-and-decode
part: "Part 7 — Large language models"
summary: Turn text into ids by replaying the learned merges in order, turn ids back into text by joining bytes, and see why the round trip is exact.
minutes: 35
gpu: false
objectives:
  - Explain why a trained tokeniser is a frozen recipe, and why encoding must not recount pairs on the text it encodes.
  - Explain why the merges must be replayed in the order they were learned, and what goes wrong in a different order.
  - Write encode and decode, and show that decode(encode(text)) returns the text exactly.
  - Explain why decoding must join all the bytes before it decodes UTF-8.
prereqs:
  - 109-bytes-and-the-vocabulary-trade
  - 110-training-bpe
exercise: 111-encode-and-decode
---

# Encoding and decoding: replay the merges in order

You have a learned list of merges, and a sentence the tokeniser has never seen. How do you turn the sentence into ids, and how do you get exactly the same sentence back from the ids?

One tempting answer is to run the training loop again on the sentence: count pairs, merge the winner, repeat. That fails for a plain reason. The ids are a fixed code that the model's embedding table was trained on, so the word `cat` must get the same ids in every document. If the cut depended on the other words in the sentence, then the same word would get different ids in different places, and the table rows would mean nothing. So the list of merges is frozen after training, and encoding only replays it. This chapter shows how the replay works, why its order is forced, and why the way back is exact.

## What you already know

Four ideas carry this chapter:

- The merge at position $i$ creates id $256 + i$, and its bytes are the joined bytes of its two parts (see "Training byte-pair encoding: merge the most common pair").
- Training splits text into pieces first, and merges never cross a piece boundary (same chapter).
- `merge_pair` glues a pair from the left and skips two after a hit.
- Text is a string of UTF-8 bytes, and one character can take up to four bytes (see "Why bytes, and the trade between vocabulary and length").

## A frozen recipe

Figure 1 shows the whole path. Encoding goes left to right: split the text into pieces, turn each piece into bytes, run the merges in list order over each piece, and flatten the result into one list of ids. Decoding goes the other way and is shorter, because every id already knows its bytes.

```figure
<svg viewBox="0 0 620 236" role="img" aria-label="Encode flow from text to pieces to bytes to ordered merges to ids, and decode flow in the other direction through vocabulary lookup, joining bytes, and one decode"><defs><marker id="c111-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="10" y="30" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="55" y="55" text-anchor="middle">text</text>
<rect x="124" y="30" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="169" y="55" text-anchor="middle">pieces</text>
<rect x="238" y="30" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="283" y="55" text-anchor="middle">bytes</text>
<rect x="352" y="30" width="130" height="40" rx="6" class="fill-soft-b stroke-b"/>
<text x="417" y="55" text-anchor="middle">merges in order</text>
<rect x="506" y="30" width="90" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="551" y="55" text-anchor="middle">ids</text>
<line x1="102" y1="50" x2="122" y2="50" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<line x1="216" y1="50" x2="236" y2="50" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<line x1="330" y1="50" x2="350" y2="50" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<line x1="484" y1="50" x2="504" y2="50" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<text x="10" y="20" text-anchor="start" class="fig-bold">encode</text>
<text x="10" y="126" text-anchor="start" class="fig-bold">decode</text>
<rect x="506" y="136" width="90" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="551" y="161" text-anchor="middle">ids</text>
<rect x="352" y="136" width="130" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="417" y="161" text-anchor="middle">vocab lookup</text>
<rect x="238" y="136" width="90" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="283" y="161" text-anchor="middle">join bytes</text>
<rect x="124" y="136" width="90" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="169" y="161" text-anchor="middle">decode once</text>
<rect x="10" y="136" width="90" height="40" rx="6" class="fill-soft-c stroke-c"/>
<text x="55" y="161" text-anchor="middle">text</text>
<line x1="504" y1="156" x2="484" y2="156" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<line x1="350" y1="156" x2="330" y2="156" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<line x1="236" y1="156" x2="216" y2="156" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<line x1="122" y1="156" x2="102" y2="156" class="stroke-ink" marker-end="url(#c111-arrow)"/>
<text x="10" y="220" text-anchor="start" class="fig-small">Encode runs the merges in the order they were learned. Decode only looks up bytes and joins them.</text>
</svg>
Figure 1. Encode applies the merges in the order they were learned. Decode only looks up bytes, joins them, and decodes once. The list of merges and the vocabulary are never changed by either.
```

The training chapter gave you the pieces. Here you use them without changing them. You never count anything on the text you encode.

## Replay the merges in order

Take the word you traced in the previous chapter, `aaabdaaabac`, and the merge list learned from it: `[(97, 97), (97, 98), (256, 257)]`. Treat the whole string as one piece, as 11 bytes. Merge 0 glues `(97, 97)` into 256 and leaves 9 tokens. Merge 1 glues `(97, 98)` into 257 and leaves 7. Merge 2 glues `(256, 257)` into 258 and leaves 5: `[258, 100, 258, 97, 99]`. The left half of Figure 2 shows each step.

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
Figure 2. Orange cells are the tokens a merge has made. Left: the merges replayed in the order they were learned, giving 5 tokens. Right: the same three merges in reverse order give 7 tokens. In the reversed list the first merge has no hit, because the ids 256 and 257 do not exist yet.
```

Why must the order match? Look at what merge 2 expects. It glues the ids 256 and 257, and those ids exist only after merges 0 and 1 have run. More generally, merge $i$ was chosen on text that merges 0 to $i-1$ had already rewritten. Replaying in the same order rebuilds the same rewritten text, so each merge finds the pairs it was chosen for. A merge list is a recipe with steps, not a set of facts.

Reverse the list to `[(256, 257), (97, 98), (97, 97)]` and encode the same string. The first merge looks for ids 256 and 257, finds none, and does nothing. The second glues `(97, 98)`, and in this list that pair has id 257. The third glues `(97, 97)` into id 258. The result is `[258, 257, 100, 258, 257, 97, 99]`, 7 tokens instead of 5. The right half of Figure 2 shows it.

The length is the smaller problem. The ids mean different things: in the reversed list 257 is `ab` and 258 is `aa`. Decode those ids with the original vocabulary, where 258 is `aaab`, and you get `aaababdaaababac`, not the text you started with. Order is part of the meaning of the ids.

Now try a string the tokeniser never saw, `abaab`, with the learned list. As 5 bytes it is `[97, 98, 97, 97, 98]`. Merge 0 glues the one `(97, 97)` pair and leaves `[97, 98, 256, 98]`. Merge 1 glues `(97, 98)` at the start and leaves `[257, 256, 98]`. Merge 2 looks for `256` followed by `257`, finds `257` then `256`, and does nothing. The result is 3 tokens, which decode to `ab`, `aa`, `b`. Notice that the last `aab` was cut as `aa` and `b`, not as `a` and `ab`. Merge 0 ran first and used up the second `a`, so merge 1 had no `ab` left to find there. Earlier merges have priority over later ones, wherever they sit in the text, and that is also why the cut is the same every time.

> [!KEY]
> A trained tokeniser is a frozen recipe. Encode runs the merges in the order they were learned, because merge $i$ was chosen on text that merges 0 to $i-1$ had already rewritten. It never recounts, so the same word gets the same ids in every document.

## Encode a whole text

Encoding a text applies the same replay to every piece and then flattens. You can read it as a formula:

$$
\text{encode}(\text{text}) = \text{replay}(\text{bytes}(p_1)) \;\Vert\; \text{replay}(\text{bytes}(p_2)) \;\Vert\; \cdots
$$

Read it aloud: split the text into pieces $p_1, p_2, \ldots$, replay all the merges on the bytes of each piece, and join the results end to end, where $\Vert$ means join. Because the pieces are split before any merge, `encode("the cat")` equals `encode("the")` followed by `encode(" cat")`. One word's ids never depend on its neighbours. In code the shape is a loop over `_pretokenize(text)` that calls a one-piece helper, `apply_merges_to_word`, and extends a flat list.

## Decode: join, then decode once

Every id stands for bytes. The vocabulary maps an id to its bytes, so decoding is a lookup, a join, and one decode:

```python
text = b"".join(vocab[i] for i in ids).decode("utf-8")
```

Join first, and decode once. The reason is that a token is a slice of bytes, and a slice can cut a character in half. A token boundary does not know about characters. With no merges at all, every token is a single byte, so the emoji 🎉 becomes four tokens holding 240, 159, 142, and 137. Figure 3 cuts those four into two tokens and decodes each way.

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
Figure 3. The emoji is four bytes. If a token boundary falls between its second and third byte, decoding the first token alone fails, because bytes 240 and 159 are only the start of a character. Joining all four bytes and decoding once returns the emoji.
```

> [!WARNING]
> Never call `.decode("utf-8")` on one token at a time. A token can hold half of a character, and half of a character is not valid text. Join all the bytes first.

The round trip follows. Each merge replaces a pair by an id whose bytes are those two parts joined, so the bytes of the whole sequence never change while you encode. Decoding joins them back into the same byte string, and a byte string that came from valid text decodes to that text. So `decode(encode(text)) == text` for every text, with merges or without, including accents, Chinese characters, and emoji.

> [!RECAP]
> Hold these points while you write the functions:
>
> - Replay merge $i$ with id $256 + i$, in list order, on each piece of the text.
> - Never recount pairs while encoding, so the same piece always gets the same ids.
> - A different order breaks the ids, because later merges expect ids that earlier merges create.
> - Decode by looking up each id's bytes, joining them all, and calling `.decode("utf-8")` once.
> - The round trip is exact because merging never changes the bytes.

## Build it step by step

The checks call three functions, in this order. The starter already contains `get_pair_counts`, `merge_pair`, and `train_bpe` from the previous chapter, solved.

1. `apply_merges_to_word(word_bytes, merges)` takes a list of byte ints for one piece and the merge list, and returns the list of ids after the replay. For each position `i` in the list, call `merge_pair([ids], pair, 256 + i)` and take the first (and only) list back. The usual mistake is using `i` instead of `256 + i` as the id.

```python
def apply_merges_to_word(word_bytes, merges):
    ids = list(word_bytes)
    # for i, pair in enumerate(merges): ids = merge_pair([ids], pair, 256 + i)[0]
    return ids
```

2. `encode(text, merges)` returns one flat list of ids. Split with `_pretokenize`, turn each piece into `list(piece.encode("utf-8"))`, replay with the helper, and extend a flat list. With an empty merge list the result is the raw bytes.

```python
def encode(text, merges):
    # for piece in _pretokenize(text): extend ids with apply_merges_to_word(...)
    return ids
```

3. `decode(ids, vocab)` returns a string. Join `vocab[i]` for every id, then decode once. The usual mistake is decoding each id alone, which raises an error on a half character.

```python
def decode(ids, vocab):
    # b"".join(...).decode("utf-8")
    return text
```

## Where this breaks

Replaying every merge on every piece costs one pass per merge, which is slow for a vocabulary of 100,000 ids. Practical tokenisers keep each merge's rank in a table and always apply the lowest-ranked pair that is present, which gives the same answer faster. Text that is not valid UTF-8 cannot come out of `decode`, but it cannot go in through `encode` either, since encode starts from text.

## The exercise

You write `apply_merges_to_word`, `encode`, and `decode`. The training code is given. The checks verify the one-word replay on `aaabdaaabac`, the effect of reversing the list, that encoding with no merges gives raw bytes, that one piece's ids do not depend on its neighbours, that `decode` handles a character split across two tokens, and that text round-trips after training, including accents, Chinese characters, and emoji.
