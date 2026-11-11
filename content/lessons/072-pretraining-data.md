---
title: "Pretraining data: dedup, filtering, and mixing"
slug: 072-pretraining-data
part: "Part 7 — Large language models"
summary: Drop copies with hashing and MinHash, filter noise with a quality score, and set how much of each source the model reads.
minutes: 75
gpu: false
objectives:
  - Drop exact copies by hashing normalised text, and say why raw web text needs cleaning.
  - Estimate Jaccard similarity from MinHash signatures, with an error that shrinks as 1/sqrt(k).
  - Find near-duplicate pairs with LSH bands and the probability 1 - (1 - s^r)^b.
  - Score text quality with a character bigram model, and set domain shares with a temperature.
prereqs:
  - 067-counts-to-a-language-model
  - 071-tokenisation
exercise: 072-pretraining-data
---

# Pretraining data: dedup, filtering, and mixing

A language model reads trillions of tokens, and nobody reads them first. The text comes from a crawl of the web, and a crawl is messy. The same terms-of-service footer sits on 60,000 pages. A product page appears once per colour. A forum post is quoted in a hundred replies. Lee and colleagues found a single 61-word sentence that occurs more than 60,000 times in C4, a popular cleaned crawl. A model that reads a sentence that often memorises it, and the compute it spends on the 60,000th copy teaches it nothing.

Copies are one problem. Junk is another: pages of keywords, menus, random strings, and text scraped from the wrong encoding teach the model to produce junk. A third problem is the recipe. Web pages, code, and books differ in size by orders of magnitude, so reading each source in proportion to its size means the model sees almost no books.

This chapter builds the three tools that handle these problems. You drop exact copies with a hash. You find near-copies with shingles, MinHash, and locality-sensitive hashing, which compares short signatures instead of whole documents. You score quality with a small language model, and you set the share of each source with one knob, a temperature. Each tool is a few lines of numpy, and each has a number you can check against theory.

## What you already know

- A tokeniser turns text into ids, and training cost grows with the number of tokens read (see [tokenisation](/lessons/071-tokenisation)).
- A bigram model counts how often one symbol follows another, and add-one smoothing keeps unseen pairs from getting probability zero (see [from counts to a language model](/lessons/067-counts-to-a-language-model)).
- A model's loss falls as a power law in data, and repeated data is worth less than fresh data (see [scaling laws](/lessons/073-scaling-laws)).
- The mean of $k$ independent coin flips with chance $p$ has spread $\sqrt{p(1-p)/k}$.
- A probability of "at least one" is one minus the probability of "none".

## Where the text comes from

The largest open pretraining sets start from Common Crawl, a public archive of web snapshots. FineWeb is built from 96 snapshots and holds 15 trillion tokens. A pipeline extracts the main text from the HTML, drops pages in the wrong language, and then runs filters: functions that take a list of documents and return a shorter list. Cheap filters run first. Hashing comes first, MinHash second, and quality scoring last, because it runs a model on every document it sees.

## Exact copies: hash the text

Two documents that are the same text are the easiest copies to drop. Comparing every pair is far too slow, so you compare fingerprints. A hash function turns a document of any length into a fixed-size number, and equal documents give equal numbers. Keep a document only the first time its hash appears.

Before hashing, normalise the text: lowercase it and collapse runs of whitespace. Otherwise `Hello  world` and `hello world` get different hashes. A 32-byte SHA-256 hash puts a billion documents in 32 gigabytes of fingerprints.

```python
digest = hashlib.sha256(normalise(doc).encode("utf-8")).hexdigest()
if digest not in seen:
    seen.add(digest)
    kept.append(i)
```

Exact hashing misses most real duplicates. Change one word or swap a banner, and the hash changes completely.

## Near copies: shingles and Jaccard similarity

Think of each document as a bag of overlapping phrases. A **shingle** is a run of $k$ consecutive words. For $k = 2$, the sentence `the cat sat on the mat` has the shingles `the cat`, `cat sat`, `sat on`, `on the`, and `the mat`. Write the set of shingles of document $A$ as $S_A$. Repeated shingles count once, because a set holds each member once.

Near-copies have shingle sets that overlap heavily. The usual measure is the **Jaccard similarity**:

$$
J(A, B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}
$$

Read it aloud: count the shingles both documents have, and divide by the count of shingles that either has. It is 1 for identical sets and 0 for sets with nothing in common.

Take a second sentence, `the cat sat on the rug`. Its shingles are `the cat`, `cat sat`, `sat on`, `on the`, and `the rug`. The two sentences share four shingles, and together they have six, so $J = 4/6 = 0.667$. Changing one word of six cost a third of the similarity, which is why you need a threshold and not an equality test. In Python:

```python
sa, sb = shingles(a, 2), shingles(b, 2)
j = len(sa & sb) / len(sa | sb)
```

Longer shingles are stricter: with $k = 1$ any two English documents share many words, and with $k = 5$ they share a shingle only when they share a whole phrase.

The trouble is cost. A real document has thousands of shingles, and a billion documents have $5 \times 10^{17}$ pairs. You cannot compute a Jaccard similarity for every pair, and you cannot even hold every shingle set in memory. The next two sections fix both.

## MinHash: a short signature that remembers overlap

Here is the picture. Take all possible shingles and shuffle them into a random order. For a document, find which of its shingles comes first in that order. Do the same for a second document. Figure 1 does this with six shingles.

When do the two first shingles agree? Look at the first shingle of the union $S_A \cup S_B$ under the shuffle. Every member of the union is equally likely to be first. The two documents share their first shingle exactly when the union's first shingle lies in the intersection. The chance of that is

$$
P(\text{same minimum}) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|} = J(A, B)
$$

So one shuffle is one coin flip with chance $J$ of heads. Use $k$ independent shuffles and the fraction of heads estimates $J$. A **MinHash signature** is the list of $k$ first-shingles, one per shuffle. A hash function stands in for a shuffle: give each shingle a number and take the smallest. Hash $i$ is $x \mapsto (a_i x + b_i) \bmod p$ with random $a_i$ and $b_i$, applied to a fixed integer for each shingle.

Work an example by hand. Number the six shingles of the two sentences: `the cat` is 0, `cat sat` is 1, `sat on` is 2, `on the` is 3, `the mat` is 4, and `the rug` is 5. Then $S_A = \{0,1,2,3,4\}$ and $S_B = \{0,1,2,3,5\}$. Use three hash functions on the numbers, each reduced mod 7:

| Hash | Values for shingles 0 to 5 | Minimum for A | Minimum for B | Match |
|---|---|---|---|---|
| $(x+1) \bmod 7$ | 1, 2, 3, 4, 5, 6 | 1 | 1 | yes |
| $(3x+1) \bmod 7$ | 1, 4, 0, 3, 6, 2 | 0 | 0 | yes |
| $(x+3) \bmod 7$ | 3, 4, 5, 6, 0, 1 | 0 | 1 | no |

Two of the three signature positions match, so the estimate is $2/3 = 0.667$. The true value is also $0.667$. With only three hashes that agreement is luck. The third hash is a miss: its smallest value falls on shingle 4 for $A$, which $B$ lacks, and on shingle 5 for $B$.

```figure
<svg viewBox="0 0 640 270" role="img" aria-label="Three hash functions over six shingles, with the smallest value in each document outlined and matches in green">
<text x="164" y="22" text-anchor="middle" class="fig-small">the cat</text>
<text x="164" y="40" text-anchor="middle" class="fig-small fig-bold">in A and B</text>
<text x="240" y="22" text-anchor="middle" class="fig-small">cat sat</text>
<text x="240" y="40" text-anchor="middle" class="fig-small fig-bold">in A and B</text>
<text x="316" y="22" text-anchor="middle" class="fig-small">sat on</text>
<text x="316" y="40" text-anchor="middle" class="fig-small fig-bold">in A and B</text>
<text x="392" y="22" text-anchor="middle" class="fig-small">on the</text>
<text x="392" y="40" text-anchor="middle" class="fig-small fig-bold">in A and B</text>
<text x="468" y="22" text-anchor="middle" class="fig-small">the mat</text>
<text x="468" y="40" text-anchor="middle" class="fig-small fig-bold">in A</text>
<text x="544" y="22" text-anchor="middle" class="fig-small">the rug</text>
<text x="544" y="40" text-anchor="middle" class="fig-small fig-bold">in B</text>
<text x="120" y="82" text-anchor="end" class="fig-small fig-mono">(x+1) mod 7</text>
<rect x="130" y="56" width="68" height="40" rx="4" class="fill-soft-c stroke-c" stroke-width="2"/>
<text x="164" y="82" text-anchor="middle" class="fig-mono">1</text>
<rect x="206" y="56" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="240" y="82" text-anchor="middle" class="fig-mono">2</text>
<rect x="282" y="56" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="316" y="82" text-anchor="middle" class="fig-mono">3</text>
<rect x="358" y="56" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="392" y="82" text-anchor="middle" class="fig-mono">4</text>
<rect x="434" y="56" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="468" y="82" text-anchor="middle" class="fig-mono">5</text>
<rect x="510" y="56" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="544" y="82" text-anchor="middle" class="fig-mono">6</text>
<text x="590" y="82" class="fig-small fig-bold">match</text>
<text x="120" y="138" text-anchor="end" class="fig-small fig-mono">(3x+1) mod 7</text>
<rect x="130" y="112" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="164" y="138" text-anchor="middle" class="fig-mono">1</text>
<rect x="206" y="112" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="240" y="138" text-anchor="middle" class="fig-mono">4</text>
<rect x="282" y="112" width="68" height="40" rx="4" class="fill-soft-c stroke-c" stroke-width="2"/>
<text x="316" y="138" text-anchor="middle" class="fig-mono">0</text>
<rect x="358" y="112" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="392" y="138" text-anchor="middle" class="fig-mono">3</text>
<rect x="434" y="112" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="468" y="138" text-anchor="middle" class="fig-mono">6</text>
<rect x="510" y="112" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="544" y="138" text-anchor="middle" class="fig-mono">2</text>
<text x="590" y="138" class="fig-small fig-bold">match</text>
<text x="120" y="194" text-anchor="end" class="fig-small fig-mono">(x+3) mod 7</text>
<rect x="130" y="168" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="164" y="194" text-anchor="middle" class="fig-mono">3</text>
<rect x="206" y="168" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="240" y="194" text-anchor="middle" class="fig-mono">4</text>
<rect x="282" y="168" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="316" y="194" text-anchor="middle" class="fig-mono">5</text>
<rect x="358" y="168" width="68" height="40" rx="4" class="fill-sunk stroke-grid" stroke-width="2"/>
<text x="392" y="194" text-anchor="middle" class="fig-mono">6</text>
<rect x="434" y="168" width="68" height="40" rx="4" class="fill-soft-a stroke-a" stroke-width="2"/>
<text x="468" y="194" text-anchor="middle" class="fig-mono">0</text>
<rect x="510" y="168" width="68" height="40" rx="4" class="fill-soft-b stroke-b" stroke-width="2"/>
<text x="544" y="194" text-anchor="middle" class="fig-mono">1</text>
<text x="590" y="194" class="fig-small fig-bold">miss</text>
<text x="20" y="236" class="fig-small">Blue: smallest over A. Orange: smallest over B. Green: the same shingle for both.</text>
<text x="20" y="256" class="fig-small">Matches: 2 of 3, so the estimate is 0.667; the true J is 4/6 = 0.667.</text>
</svg>
Figure 1. Where a signature position matches and where it misses. Two first-shingles agree only when the smallest hash over the union lies in the intersection.
```

How good is the estimate? Each position is a coin flip with chance $J$, and the estimate is the mean of $k$ flips. Its spread is

$$
\sigma = \sqrt{\frac{J(1 - J)}{k}}
$$

Read it aloud: the spread is the coin's own spread, $\sqrt{J(1-J)}$, divided by the square root of the number of flips. For $J = 0.5$ and $k = 128$ the spread is $\sqrt{0.25/128} = 0.044$. To halve the error you need four times as many hashes. Figure 2 measures it: 400 seeded pairs with true $J = 0.5$ at four signature lengths. The measured error sits on the theory.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Bars of the measured and theoretical error of the MinHash estimate for 16, 64, 256, and 1024 hashes">
<text x="70" y="40" text-anchor="end">k = 16</text>
<rect x="80" y="20" width="380.0" height="20" class="fill-a"/>
<text x="468.0" y="35" class="fig-small fig-mono">0.126</text>
<rect x="80" y="44" width="376.6" height="12" class="fill-muted"/>
<text x="464.6" y="55" class="fig-small fig-mono">0.125</text>
<text x="70" y="96" text-anchor="end">k = 64</text>
<rect x="80" y="76" width="189.3" height="20" class="fill-a"/>
<text x="277.3" y="91" class="fig-small fig-mono">0.063</text>
<rect x="80" y="100" width="188.3" height="12" class="fill-muted"/>
<text x="276.3" y="111" class="fig-small fig-mono">0.062</text>
<text x="70" y="152" text-anchor="end">k = 256</text>
<rect x="80" y="132" width="94.7" height="20" class="fill-a"/>
<text x="182.7" y="147" class="fig-small fig-mono">0.031</text>
<rect x="80" y="156" width="94.2" height="12" class="fill-muted"/>
<text x="182.2" y="167" class="fig-small fig-mono">0.031</text>
<text x="70" y="208" text-anchor="end">k = 1024</text>
<rect x="80" y="188" width="48.9" height="20" class="fill-a"/>
<text x="136.9" y="203" class="fig-small fig-mono">0.016</text>
<rect x="80" y="212" width="47.1" height="12" class="fill-muted"/>
<text x="135.1" y="223" class="fig-small fig-mono">0.016</text>
<rect x="440" y="196" width="14" height="14" class="fill-a"/><text x="460" y="208" class="fig-small">measured</text>
<rect x="440" y="218" width="14" height="10" class="fill-muted"/><text x="460" y="228" class="fig-small">sqrt(0.25 / k)</text>
</svg>
Figure 2. Root-mean-square error of the MinHash estimate at J = 0.5. Each step multiplies k by 4 and halves the error, as sqrt(0.25 / k) predicts.
```

In numpy, one signature is a single matrix operation over all hashes and all shingles:

```python
ids = np.array([stable_id(s) for s in shingle_set])
hashed = (a[:, None] * ids[None, :] + b[:, None]) % PRIME   # (n_hashes, n_shingles)
signature = hashed.min(axis=1)
estimate = np.mean(sig_a == sig_b)
```

Use a seeded hash such as `blake2b` for the shingle id. Python's built-in `hash` for strings changes between runs, so signatures from two runs would never match.

> [!KEY]
> MinHash turns a set of thousands of shingles into $k$ numbers. Two signatures agree at a position with probability equal to the Jaccard similarity, so the fraction of agreeing positions estimates it, with spread $\sqrt{J(1-J)/k}$.

> [!WARNING]
> A bottom-$k$ sketch is not a MinHash signature. It uses one hash and keeps the $k$ smallest values, so its entries are not independent coin flips, and comparing two such lists by position measures nothing. The test is the union rule: the signature of $A \cup B$ is the elementwise minimum of the two signatures.

## LSH: find the pairs without comparing all pairs

A billion documents still make $5 \times 10^{17}$ pairs of signatures. **Locality-sensitive hashing**, or LSH, avoids most of them: put documents into buckets so that similar ones land together, and compare only documents that share a bucket.

Cut the signature into $b$ **bands** of $r$ rows each, so $k = b \cdot r$. For each band, use the $r$ numbers as a bucket key. Two documents become a **candidate pair** if they get the same key in at least one band.

What is the chance of that? A pair with similarity $s$ matches in one row with probability $s$. A band needs all $r$ of its rows to match, which has probability $s^r$. The pair misses a given band with probability $1 - s^r$, and misses all $b$ bands with probability $(1 - s^r)^b$. So

$$
P(\text{candidate}) = 1 - (1 - s^r)^b
$$

Read it aloud: one minus the chance that every band fails. This is an S-shaped curve of $s$. Work one point by hand with $r = 4$ and $b = 5$ at $s = 0.5$. One band matches with probability $0.5^4 = 0.0625$. All five bands miss with probability $0.9375^5 = 0.724$. So the pair is a candidate with probability $1 - 0.724 = 0.276$.

```python
def lsh_candidate_prob(s, r, b):
    return 1 - (1 - s**r) ** b
```

The curve rises steeply near a **threshold** of about $(1/b)^{1/r}$. Figure 3 spends the same 128 hashes two ways. With $r = 4$ and $b = 32$ the threshold is about 0.42, and a pair at $s = 0.5$ is a candidate with probability 0.873. With $r = 8$ and $b = 16$ the threshold is about 0.71, and the same pair is a candidate with probability only 0.061. The dots are a Monte Carlo of the first curve: draw each row as a coin with chance $s$, and count how often some band has all heads. They land on the curve.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="S-curves of the chance that two documents share a bucket, for 4 rows by 32 bands and 8 rows by 16 bands">
<line x1="60" y1="250.0" x2="500" y2="250.0" class="stroke-grid"/>
<text x="52" y="254.0" text-anchor="end" class="fig-small">0</text>
<line x1="60" y1="135.0" x2="500" y2="135.0" class="stroke-grid"/>
<text x="52" y="139.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="60" y1="20.0" x2="500" y2="20.0" class="stroke-grid"/>
<text x="52" y="24.0" text-anchor="end" class="fig-small">1</text>
<text x="60.0" y="268" text-anchor="middle" class="fig-small">0</text>
<text x="170.0" y="268" text-anchor="middle" class="fig-small">0.25</text>
<text x="280.0" y="268" text-anchor="middle" class="fig-small">0.5</text>
<text x="390.0" y="268" text-anchor="middle" class="fig-small">0.75</text>
<text x="500.0" y="268" text-anchor="middle" class="fig-small">1</text>
<text x="280.0" y="290" text-anchor="middle">true Jaccard similarity s</text>
<text x="14" y="135.0" text-anchor="middle" transform="rotate(-90 14 135.0)">P(candidate pair)</text>
<polyline points="60.0,250.0 64.4,250.0 68.8,250.0 73.2,250.0 77.6,250.0 82.0,250.0 86.4,249.9 90.8,249.8 95.2,249.7 99.6,249.5 104.0,249.3 108.4,248.9 112.8,248.5 117.2,247.9 121.6,247.2 126.0,246.3 130.4,245.2 134.8,243.9 139.2,242.4 143.6,240.6 148.0,238.5 152.4,236.1 156.8,233.4 161.2,230.3 165.6,226.8 170.0,222.9 174.4,218.6 178.8,213.9 183.2,208.8 187.6,203.3 192.0,197.3 196.4,190.9 200.8,184.1 205.2,177.0 209.6,169.5 214.0,161.8 218.4,153.8 222.8,145.5 227.2,137.2 231.6,128.8 236.0,120.3 240.4,111.9 244.8,103.6 249.2,95.6 253.6,87.7 258.0,80.2 262.4,73.1 266.8,66.4 271.2,60.2 275.6,54.4 280.0,49.2 284.4,44.4 288.8,40.3 293.2,36.6 297.6,33.4 302.0,30.7 306.4,28.4 310.8,26.5 315.2,24.9 319.6,23.7 324.0,22.7 328.4,22.0 332.8,21.4 337.2,21.0 341.6,20.6 346.0,20.4 350.4,20.3 354.8,20.2 359.2,20.1 363.6,20.1 368.0,20.0 372.4,20.0 376.8,20.0 381.2,20.0 385.6,20.0 390.0,20.0 394.4,20.0 398.8,20.0 403.2,20.0 407.6,20.0 412.0,20.0 416.4,20.0 420.8,20.0 425.2,20.0 429.6,20.0 434.0,20.0 438.4,20.0 442.8,20.0 447.2,20.0 451.6,20.0 456.0,20.0 460.4,20.0 464.8,20.0 469.2,20.0 473.6,20.0 478.0,20.0 482.4,20.0 486.8,20.0 491.2,20.0 495.6,20.0 500.0,20.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="60.0,250.0 64.4,250.0 68.8,250.0 73.2,250.0 77.6,250.0 82.0,250.0 86.4,250.0 90.8,250.0 95.2,250.0 99.6,250.0 104.0,250.0 108.4,250.0 112.8,250.0 117.2,250.0 121.6,250.0 126.0,250.0 130.4,250.0 134.8,250.0 139.2,250.0 143.6,250.0 148.0,250.0 152.4,250.0 156.8,250.0 161.2,250.0 165.6,250.0 170.0,249.9 174.4,249.9 178.8,249.9 183.2,249.9 187.6,249.8 192.0,249.8 196.4,249.7 200.8,249.6 205.2,249.5 209.6,249.3 214.0,249.2 218.4,249.0 222.8,248.7 227.2,248.4 231.6,248.0 236.0,247.6 240.4,247.1 244.8,246.5 249.2,245.7 253.6,244.9 258.0,243.9 262.4,242.7 266.8,241.4 271.2,239.8 275.6,238.1 280.0,236.0 284.4,233.7 288.8,231.1 293.2,228.1 297.6,224.8 302.0,221.0 306.4,216.9 310.8,212.2 315.2,207.1 319.6,201.5 324.0,195.4 328.4,188.7 332.8,181.6 337.2,173.9 341.6,165.7 346.0,157.0 350.4,147.9 354.8,138.5 359.2,128.8 363.6,118.9 368.0,108.9 372.4,99.0 376.8,89.3 381.2,79.9 385.6,70.9 390.0,62.5 394.4,54.8 398.8,47.9 403.2,41.8 407.6,36.5 412.0,32.2 416.4,28.7 420.8,25.9 425.2,23.9 429.6,22.4 434.0,21.4 438.4,20.8 442.8,20.4 447.2,20.2 451.6,20.1 456.0,20.0 460.4,20.0 464.8,20.0 469.2,20.0 473.6,20.0 478.0,20.0 482.4,20.0 486.8,20.0 491.2,20.0 495.6,20.0 500.0,20.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="104.0" cy="249.2" r="4" class="fill-ink"/>
<circle cx="148.0" cy="238.6" r="4" class="fill-ink"/>
<circle cx="192.0" cy="196.9" r="4" class="fill-ink"/>
<circle cx="236.0" cy="120.9" r="4" class="fill-ink"/>
<circle cx="280.0" cy="49.1" r="4" class="fill-ink"/>
<circle cx="324.0" cy="22.7" r="4" class="fill-ink"/>
<circle cx="368.0" cy="20.0" r="4" class="fill-ink"/>
<circle cx="412.0" cy="20.0" r="4" class="fill-ink"/>
<circle cx="456.0" cy="20.0" r="4" class="fill-ink"/>
<text x="68.8" y="107.4" class="fig-small fig-bold fill-a">r = 4, b = 32</text>
<text x="385.6" y="158.0" class="fig-small fig-bold fill-b">r = 8, b = 16</text>
</svg>
Figure 3. The chance that two documents become a candidate pair, as a function of their true similarity, for two ways to spend 128 hashes. More rows per band push the S-curve to the right. The dots are a Monte Carlo of the first curve.
```

> [!KEY]
> Bands trade recall for cost. More bands with fewer rows catch more true near-copies and also let more unrelated pairs through. Choose $r$ and $b$ so that the curve's knee sits slightly below the similarity you want to treat as a copy.

The full dedup then has four steps. Shingle and sign every document. Find candidate pairs from the bands. For each candidate pair, compute the estimated similarity from the signatures and keep only the pairs at or above a threshold, such as 0.7. Link each kept pair, so that documents form clusters, and keep one document from each cluster. A cluster can hold a chain: $A$ is close to $B$ and $B$ is close to $C$, so all three share a cluster even if $A$ and $C$ are not close. A union-find structure tracks the clusters, and keeping the earliest document makes the result easy to check.

Take the exercise corpus: four clusters of three documents, where each copy of a 200-word text changes four words, plus eight unrelated documents. Each changed word breaks up to three of the 198 three-word shingles, so a copy has $J \approx 0.89$ with its original, and the chance that the pair becomes a candidate is above 0.9999. Twenty documents go in and twelve come out.

## Quality: keep text worth learning from

Dedup leaves unique text, and unique text can still be junk. Two kinds of filter handle it.

**Heuristic rules** are cheap checks on simple statistics. Drop a document when it has too few words, when its mean word length is far from that of prose, when too few of its characters are letters, or when many of its lines repeat. Each rule is a line or two:

```python
words = text.split()
mean_len = sum(len(w) for w in words) / len(words)
alpha = sum(c.isalpha() for c in text) / len(text)
keep = len(words) >= 50 and 3 <= mean_len <= 10 and alpha >= 0.6
```

The thresholds come from reading samples. Each rule removes some good documents too, such as a table of numbers, and misses text that is fluent and worthless.

A **model-based filter** scores each document with a model trained on text you trust. The simplest version is the language model you already know. Train a character bigram model on a reference set of clean text. Score a document by its mean log probability per adjacent character pair:

$$
\text{score}(x) = \frac{1}{n - 1} \sum_{t=1}^{n-1} \log P(x_{t+1} \mid x_t)
$$

Read it aloud: for each adjacent pair of characters, ask the model how likely the second character is after the first, take the log, and average. A score near zero means the model finds the text expected. A very negative score means surprise. The average per pair, not the sum, keeps a long clean document from scoring below a short noisy one.

```python
counts = np.zeros((128, 128))
np.add.at(counts, (codes[:-1], codes[1:]), 1)      # for each clean training text
logp = np.log((counts + 1) / (counts + 1).sum(axis=1, keepdims=True))
score = logp[codes[:-1], codes[1:]].mean()          # for one document
```

The exercise plants both kinds of text. Clean documents are short English sentences from a template. Noisy documents are random printable characters, keyword and symbol spam, and random lowercase letters in word-sized pieces. On a seeded run the clean scores lie between $-2.17$ and $-1.94$, and the noisy scores lie between $-5.61$ and $-4.71$. A single cut at $-3.5$ separates every planted document.

Real pipelines use the same idea at a larger scale, with a word-level model or a classifier that tells a trusted set from random crawl. FineWeb-Edu is the part of FineWeb filtered for educational text. A classifier learns to prefer text that looks like its reference set, including that set's style, topics, and biases.

> [!WARNING]
> A quality filter cannot tell you what a document means. It tells you how much the document resembles the reference text. A filter trained on encyclopedia prose removes slang, dialects, and technical writing along with the spam. Check what it drops, not only what it keeps.

## Mixing: how much of each source

After cleaning, you have several pools: web, code, books. Reading each in proportion to its size gives the model a diet that is almost all web. Reading each equally repeats a small pool many times. A temperature picks a point between the two.

Say pool $i$ holds $n_i$ tokens. The sampling share is

$$
p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}
$$

Read it aloud: raise each size to the power $1/T$, then divide by the total so the shares add to 1. At $T = 1$ the power is 1, so $p_i$ is proportional to size. As $T$ grows, the power goes to 0, every $n_i^{1/T}$ goes to 1, and the shares become equal. A larger $T$ lifts the small pools and lowers the large ones.

Work an example with three pools of 900, 90, and 10 million tokens. At $T = 1$ the shares are 0.9, 0.09, and 0.01. At $T = 2$ take square roots: 30, 9.49, and 3.16, which sum to 42.65. The shares are 0.703, 0.222, and 0.074. Figure 4 shows $T = 1, 2, 4$, and a very large $T$.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Stacked bars of the sampling share of web, code, and books at T = 1, 2, 4, and a very large T">
<text x="70" y="40" text-anchor="end">T = 1</text>
<rect x="80.0" y="20" width="396.0" height="28" class="fill-a"/>
<text x="278.0" y="39" text-anchor="middle" class="fig-small fill-paper">0.90</text>
<rect x="476.0" y="20" width="39.6" height="28" class="fill-b"/>
<text x="495.8" y="39" text-anchor="middle" class="fig-small fill-paper">0.09</text>
<rect x="515.6" y="20" width="4.4" height="28" class="fill-c"/>
<text x="70" y="86" text-anchor="end">T = 2</text>
<rect x="80.0" y="66" width="309.5" height="28" class="fill-a"/>
<text x="234.8" y="85" text-anchor="middle" class="fig-small fill-paper">0.70</text>
<rect x="389.5" y="66" width="97.9" height="28" class="fill-b"/>
<text x="438.4" y="85" text-anchor="middle" class="fig-small fill-paper">0.22</text>
<rect x="487.4" y="66" width="32.6" height="28" class="fill-c"/>
<text x="70" y="132" text-anchor="end">T = 4</text>
<rect x="80.0" y="112" width="233.2" height="28" class="fill-a"/>
<text x="196.6" y="131" text-anchor="middle" class="fig-small fill-paper">0.53</text>
<rect x="313.2" y="112" width="131.1" height="28" class="fill-b"/>
<text x="378.7" y="131" text-anchor="middle" class="fig-small fill-paper">0.30</text>
<rect x="444.3" y="112" width="75.7" height="28" class="fill-c"/>
<text x="482.1" y="131" text-anchor="middle" class="fig-small fill-paper">0.17</text>
<text x="70" y="178" text-anchor="end">T large</text>
<rect x="80.0" y="158" width="147.0" height="28" class="fill-a"/>
<text x="153.5" y="177" text-anchor="middle" class="fig-small fill-paper">0.33</text>
<rect x="227.0" y="158" width="146.7" height="28" class="fill-b"/>
<text x="300.3" y="177" text-anchor="middle" class="fig-small fill-paper">0.33</text>
<rect x="373.7" y="158" width="146.3" height="28" class="fill-c"/>
<text x="446.8" y="177" text-anchor="middle" class="fig-small fill-paper">0.33</text>
<rect x="90" y="208" width="14" height="14" class="fill-a"/><text x="110" y="220" class="fig-small">web 900</text>
<rect x="240" y="208" width="14" height="14" class="fill-b"/><text x="260" y="220" class="fig-small">code 90</text>
<rect x="390" y="208" width="14" height="14" class="fill-c"/><text x="410" y="220" class="fig-small">books 10</text>
</svg>
Figure 4. Sampling shares of three pools of 900, 90, and 10 million tokens. The small pools grow from 1 percent to 7 percent at T = 2 and to 17 percent at T = 4. At a very large T the three shares are equal.
```

```python
w = np.asarray(sizes, dtype=float) ** (1.0 / T)
shares = w / w.sum()
```

> [!KEY]
> One knob sets the diet. $T = 1$ reads each source in proportion to its size, and a large $T$ reads each source equally. A middle value gives small sources a larger share without giving them all the compute.

## Repeats and the scaling laws

A larger share for a small pool has a price: the pool is read more than once. The passes over pool $i$ are

$$
\text{epochs}_i = \frac{\text{budget} \cdot p_i}{n_i}
$$

Read it aloud: the tokens a run takes from pool $i$, divided by the tokens the pool holds. For a 300-million-token run at $T = 2$ the three pools get 0.23, 0.74, and 2.22 passes, and at $T = 1$ each gets 0.3.

The [scaling laws](/lessons/073-scaling-laws) chapter treats every token as fresh, but repeated tokens are worth less. Muennighoff and colleagues ran 400 training runs and report that up to 4 epochs of repeated data gave nearly the same loss as unique data, and that more repetition drove the value of extra compute toward zero. So a mixing weight is also a repeat budget.

## Build it step by step

> [!RECAP]
> - Exact copies share a hash of the normalised text, so you keep the first of each hash.
> - Shingles turn a document into a set, and Jaccard similarity is intersection over union.
> - A MinHash signature has one minimum per hash, and the fraction of agreeing positions estimates $J$ with spread $\sqrt{J(1-J)/k}$.
> - LSH bands propose pairs with probability $1 - (1 - s^r)^b$, and a threshold check on signatures confirms them.
> - A character bigram score separates clean text from noise, and $p_i \propto n_i^{1/T}$ sets the mixing shares.

The checks call twelve functions, in this order. The file already holds `normalise`, `stable_id`, and `hash_params`.

1. **`shingles(text, k)`** returns a set of strings: normalise, split into words, and join each window of $k$ words. The usual mistake is an off-by-one that drops the last window. A text shorter than $k$ words gives one shingle.
2. **`jaccard(a, b)`** returns $|a \cap b| / |a \cup b|$, or `0.0` for an empty union. The usual mistake is dividing by `len(a)`.
3. **`exact_dedup(docs)`** returns the kept indices. Hash the normalised text with `hashlib.sha256`. The usual mistake is hashing the raw text.
4. **`minhash_signature(shingle_set, n_hashes, seed)`** returns an int64 array of shape `(n_hashes,)`. Broadcast the hashes over the ids and take the minimum over the shingle axis. The usual mistake is one hash and the $k$ smallest values.
5. **`estimate_jaccard(sig_a, sig_b)`** returns the fraction of equal positions. The usual mistake is comparing sorted signatures.
6. **`lsh_candidate_prob(s, r, b)`** returns $1 - (1 - s^r)^b$ and works on arrays. The usual mistake is $(1 - (1 - s)^r)^b$.
7. **`lsh_candidates(sigs, b, r)`** returns a set of pairs `(i, j)` with `i < j`. Group rows by `tuple(sigs[i, band * r:(band + 1) * r])` in each band. A set counts a pair found in two bands once.
8. **`dedup(docs, threshold, k, n_hashes, b, seed)`** returns the sorted kept indices. Keep candidate pairs whose estimate reaches `threshold`, and union them with the smaller index as the root. The usual mistake is forgetting that links chain.
9. **`mix_weights(sizes, T)`** and **`epochs_per_domain(sizes, weights, budget)`** return shares and passes. Raise the sizes to `1 / T` as floats. The usual mistake is using `T` instead of `1 / T`.
10. **`train_char_bigram(texts, alpha)`** and **`quality_score(text, logp)`** return a `(128, 128)` log table and a mean log probability. Count pairs with `np.add.at`, add `alpha`, and normalise each row. The usual mistake is normalising the columns.

## Where this breaks

- A threshold is a choice. At 0.7 a long boilerplate with a short different body counts as a copy. At 0.9 a lightly edited copy survives. Look at samples at each threshold.
- Long repeated substrings inside different documents, such as a licence header, barely move the document-level similarity. Lee and colleagues also removed repeated substrings with a suffix array, which this chapter does not cover.
- A quality score inherits the reference set's taste, and a spam writer can copy that style.
- Mixing weights are tuned on small runs and trusted for large ones, and the best weights shift with scale.

## The exercise

You implement the twelve functions above. The ten checks run from the smallest piece to the whole. They test shingles and Jaccard on worked examples, that exact dedup ignores case and spacing, that a MinHash signature obeys the union rule, and that the estimate's error at $k = 32$ and $k = 512$ sits on $\sqrt{0.25/k}$. They compare your banding probability with its closed form and a Monte Carlo, and require `dedup` to keep the earliest document of each of four planted clusters among eight unique documents. The last two checks verify the temperature shares and passes, and require every planted clean document to outscore every planted noisy one.

## Further reading

- Stanford CS336 (Spring 2025), lectures 13 and 14, both titled "Data", on the [course schedule](http://cs336.stanford.edu/spring2025/).
- Stanford CS224N, the lecture "Pretraining (Scaling, Systems, Data)" on the [course page](https://web.stanford.edu/class/cs224n/).
- Andrei Broder, "On the resemblance and containment of documents" (1997), the paper that introduced shingles and MinHash.
- Jure Leskovec, Anand Rajaraman, and Jeff Ullman, *Mining of Massive Datasets*, chapter 3, free online at [mmds.org](http://www.mmds.org/).
- Katherine Lee and others, ["Deduplicating Training Data Makes Language Models Better"](https://arxiv.org/abs/2107.06499) (2021).
- Guilherme Penedo and others, ["The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"](https://arxiv.org/abs/2406.17557) (2024).
- Niklas Muennighoff and others, ["Scaling Data-Constrained Language Models"](https://arxiv.org/abs/2305.16264) (2023).
