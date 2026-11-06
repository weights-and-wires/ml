---
title: Naive Bayes for many features and words
slug: 029-naive-bayes
part: "Part 2 — Classical models"
summary: Treat features as independent given the class, so a classifier needs only per-feature averages and word counts, never a covariance matrix.
minutes: 40
gpu: false
objectives:
  - Explain what independence given the class assumes, and what it throws away.
  - Fit Gaussian naive Bayes with per-class variances and a floor, and a word-count model with Laplace smoothing.
  - Score a document by summing count times log word probability, and say why an unseen word must not veto a class.
prereqs:
  - 026-bayes-rule-classifier
  - 027-gaussian-discriminant-analysis
  - 028-gda-is-logistic-regression
exercise: 029-naive-bayes
---

# Naive Bayes for many features and words

An email arrives, and you must decide whether it is spam. A vocabulary of ten thousand words describes it: for each word, how many times the email uses it. The method from the last two chapters would fit a bell cloud to the spam emails, which needs a shared covariance matrix with ten thousand rows and ten thousand columns. That is $10^8$ numbers, about 800 megabytes, to be estimated from a few thousand emails. There are far fewer emails than numbers, so the estimate cannot be trusted, and the matrix cannot even be factored.

The way out is to stop asking how every pair of words moves together. Instead, you ask about one word at a time and combine the answers. That shortcut is wrong in a specific way, and this chapter shows how, and why it still classifies well. It costs one number per word and class, where the full model needed $10^8$.

## What you already know

- The posterior of a class is its prior times the density of the point, divided by the total, from *Classify by asking who made this*.
- A log turns a product into a sum, so a product of ten thousand small factors is the sum of ten thousand logs.
- A Gaussian is fitted by averaging, as in *Fit a Gaussian cloud to each class*, and a covariance matrix records how features move together.
- A feature that is zero for every row in a class has variance zero.
- `log_sum_exp` turns log scores into probabilities.

## Bag of words, and the independence trick

To feed an email to a classifier, throw away the order and keep the counts. This is a *bag of words*: the email becomes a row of $V$ counts, where $V$ is the vocabulary size. Figure 1 shows a toy version with six words and four documents, two spam and two ham.

```figure
<svg viewBox="0 0 620 440" role="img" aria-label="A grid of word counts for four toy documents, then per-class word probabilities before and after adding one to every count">
<text x="10" y="22" class="fig-bold">Word counts per document</text>
<text x="160.0" y="46" text-anchor="middle" class="fig-small">free</text>
<text x="220.0" y="46" text-anchor="middle" class="fig-small">money</text>
<text x="280.0" y="46" text-anchor="middle" class="fig-small">win</text>
<text x="340.0" y="46" text-anchor="middle" class="fig-small">meeting</text>
<text x="400.0" y="46" text-anchor="middle" class="fig-small">report</text>
<text x="460.0" y="46" text-anchor="middle" class="fig-small">lunch</text>
<text x="120" y="71.0" text-anchor="end" class="fig-small">spam 1</text>
<rect x="132" y="56" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="160.0" y="71.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="192" y="56" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="220.0" y="71.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="252" y="56" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="280.0" y="71.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="312" y="56" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="340.0" y="71.0" text-anchor="middle" class="fig-small">0</text>
<rect x="372" y="56" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="400.0" y="71.0" text-anchor="middle" class="fig-small">0</text>
<rect x="432" y="56" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="71.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="97.0" text-anchor="end" class="fig-small">spam 2</text>
<rect x="132" y="82" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="160.0" y="97.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="192" y="82" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="220.0" y="97.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="252" y="82" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="97.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="82" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="340.0" y="97.0" text-anchor="middle" class="fig-small">0</text>
<rect x="372" y="82" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="400.0" y="97.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="432" y="82" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="97.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="123.0" text-anchor="end" class="fig-small">ham 1</text>
<rect x="132" y="108" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="160.0" y="123.0" text-anchor="middle" class="fig-small">0</text>
<rect x="192" y="108" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="220.0" y="123.0" text-anchor="middle" class="fig-small">0</text>
<rect x="252" y="108" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="123.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="108" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="340.0" y="123.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="372" y="108" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="400.0" y="123.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="432" y="108" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="460.0" y="123.0" text-anchor="middle" class="fig-bold">1</text>
<text x="120" y="149.0" text-anchor="end" class="fig-small">ham 2</text>
<rect x="132" y="134" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="160.0" y="149.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="192" y="134" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="220.0" y="149.0" text-anchor="middle" class="fig-small">0</text>
<rect x="252" y="134" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="149.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="134" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="340.0" y="149.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="372" y="134" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="400.0" y="149.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="432" y="134" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="149.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="183.0" text-anchor="end" class="fig-small">spam total</text>
<rect x="132" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="160.0" y="183.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="192" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="220.0" y="183.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="252" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="280.0" y="183.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="312" y="168" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="340.0" y="183.0" text-anchor="middle" class="fig-small">0</text>
<rect x="372" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="400.0" y="183.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="432" y="168" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="183.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="209.0" text-anchor="end" class="fig-small">ham total</text>
<rect x="132" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="160.0" y="209.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="192" y="194" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="220.0" y="209.0" text-anchor="middle" class="fig-small">0</text>
<rect x="252" y="194" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="209.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="340.0" y="209.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="372" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="400.0" y="209.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="432" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="460.0" y="209.0" text-anchor="middle" class="fig-bold">1</text>
<text x="496" y="183.0" class="fig-small">= 8</text><text x="496" y="209.0" class="fig-small">= 8</text>
<text x="30" y="252" class="fig-bold">spam: word probability</text>
<line x1="30" y1="380" x2="290" y2="380" class="stroke-muted"/>
<rect x="34.0" y="286.2" width="17" height="93.8" class="fill-soft-b stroke-b"/>
<rect x="53.0" y="308.6" width="17" height="71.4" class="fill-b"/>
<text x="52.0" y="394" text-anchor="middle" class="fig-small">free</text>
<rect x="77.3" y="286.2" width="17" height="93.8" class="fill-soft-b stroke-b"/>
<rect x="96.3" y="308.6" width="17" height="71.4" class="fill-b"/>
<text x="95.3" y="394" text-anchor="middle" class="fig-small">money</text>
<rect x="120.7" y="348.8" width="17" height="31.2" class="fill-soft-b stroke-b"/>
<rect x="139.7" y="344.3" width="17" height="35.7" class="fill-b"/>
<text x="138.7" y="394" text-anchor="middle" class="fig-small">win</text>
<rect x="164.0" y="380.0" width="17" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="183.0" y="362.1" width="17" height="17.9" class="fill-b"/>
<text x="182.0" y="394" text-anchor="middle" class="fig-small">meeting</text>
<text x="172.0" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<rect x="207.3" y="348.8" width="17" height="31.2" class="fill-soft-b stroke-b"/>
<rect x="226.3" y="344.3" width="17" height="35.7" class="fill-b"/>
<text x="225.3" y="394" text-anchor="middle" class="fig-small">report</text>
<rect x="250.7" y="380.0" width="17" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="269.7" y="362.1" width="17" height="17.9" class="fill-b"/>
<text x="268.7" y="394" text-anchor="middle" class="fig-small">lunch</text>
<text x="258.7" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<text x="330" y="252" class="fig-bold">ham: word probability</text>
<line x1="330" y1="380" x2="590" y2="380" class="stroke-muted"/>
<rect x="334.0" y="348.8" width="17" height="31.2" class="fill-soft-a stroke-a"/>
<rect x="353.0" y="344.3" width="17" height="35.7" class="fill-a"/>
<text x="352.0" y="394" text-anchor="middle" class="fig-small">free</text>
<rect x="377.3" y="380.0" width="17" height="0.0" class="fill-soft-a stroke-a"/>
<rect x="396.3" y="362.1" width="17" height="17.9" class="fill-a"/>
<text x="395.3" y="394" text-anchor="middle" class="fig-small">money</text>
<text x="385.3" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<rect x="420.7" y="380.0" width="17" height="0.0" class="fill-soft-a stroke-a"/>
<rect x="439.7" y="362.1" width="17" height="17.9" class="fill-a"/>
<text x="438.7" y="394" text-anchor="middle" class="fig-small">win</text>
<text x="428.7" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<rect x="464.0" y="286.2" width="17" height="93.8" class="fill-soft-a stroke-a"/>
<rect x="483.0" y="308.6" width="17" height="71.4" class="fill-a"/>
<text x="482.0" y="394" text-anchor="middle" class="fig-small">meeting</text>
<rect x="507.3" y="286.2" width="17" height="93.8" class="fill-soft-a stroke-a"/>
<rect x="526.3" y="308.6" width="17" height="71.4" class="fill-a"/>
<text x="525.3" y="394" text-anchor="middle" class="fig-small">report</text>
<rect x="550.7" y="348.8" width="17" height="31.2" class="fill-soft-a stroke-a"/>
<rect x="569.7" y="344.3" width="17" height="35.7" class="fill-a"/>
<text x="568.7" y="394" text-anchor="middle" class="fig-small">lunch</text>
<text x="30" y="416" class="fig-small">Outlined bars: counts divided by the class total (alpha = 0). Solid bars: add 1 to every count first (alpha = 1).</text>
</svg>
Figure 1. Four toy documents and their word counts, with the counts summed per class. The outlined bars show raw frequencies, where spam never uses meeting or lunch and ham never uses money or win, so those bars are exactly zero. The solid bars add one to every count, so no word has probability zero.
```

To use Bayes' rule, you need the density of a document given its class: how likely it is that a spam email contains these exact counts. Estimating that for every combination of words is hopeless. *Naive Bayes* makes it easy with one assumption: once you know the class, the words are independent. If you are told an email is spam, whether it contains "free" says nothing about whether it contains "money". Under that assumption, the density of the whole document is the product of one small probability per word, and its log is a sum.

The assumption is naive because it is false. Spam that says "free" often says "money". Suppose the two words always appear together. The word "free" appears in half of spam and a tenth of ham, so it is evidence of 5 to 1 for spam. "Money" appears exactly when "free" does, so it adds no further information, but naive Bayes multiplies in another 5 to 1, and claims 25 to 1. With equal priors it reports spam at 0.96, where the truth is 0.83.

The error is in the confidence, not the choice. Both words point toward spam, so the double count makes the winner more certain, and it rarely changes which class wins. That is why naive Bayes classifies well even when its probabilities are too sharp. Figure 2 shows the other side of the assumption.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="A tilted ellipse of correlated points with an axis-aligned ellipse inside it, and a corner point the independent model finds ordinary">
<polygon points="370.0,74.0 369.8,69.6 369.0,65.7 367.8,62.1 366.2,59.0 364.0,56.3 361.4,54.1 358.4,52.4 354.9,51.1 351.0,50.3 346.6,50.0 341.9,50.2 336.8,50.8 331.4,52.0 325.7,53.6 319.6,55.7 313.3,58.2 306.7,61.2 299.8,64.6 292.8,68.5 285.6,72.7 278.3,77.4 270.8,82.4 263.3,87.8 255.7,93.5 248.1,99.5 240.5,105.8 232.9,112.4 225.4,119.2 218.0,126.2 210.8,133.4 203.6,140.7 196.7,148.1 190.0,155.6 183.5,163.2 177.3,170.8 171.4,178.5 165.8,186.0 160.6,193.5 155.7,201.0 151.2,208.2 147.0,215.4 143.3,222.3 140.1,229.1 137.2,235.6 134.9,241.8 132.9,247.8 131.5,253.4 130.5,258.7 130.1,263.7 130.1,268.2 130.5,272.4 131.5,276.2 132.9,279.5 134.9,282.4 137.2,284.8 140.1,286.8 143.3,288.3 147.0,289.4 151.2,289.9 155.7,290.0 160.6,289.6 165.8,288.7 171.4,287.3 177.3,285.4 183.5,283.1 190.0,280.4 196.7,277.1 203.6,273.5 210.8,269.4 218.0,265.0 225.4,260.1 232.9,254.9 240.5,249.4 248.1,243.5 255.7,237.4 263.3,230.9 270.8,224.2 278.3,217.3 285.6,210.2 292.8,203.0 299.8,195.6 306.7,188.1 313.3,180.6 319.6,173.0 325.7,165.3 331.4,157.8 336.8,150.2 341.9,142.7 346.6,135.4 351.0,128.2 354.9,121.1 358.4,114.3 361.4,107.6 364.0,101.3 366.2,95.2 367.8,89.4 369.0,83.9 369.8,78.8 370.0,74.0" fill="none" class="stroke-a" stroke-width="2"/>
<polygon points="370.0,170.0 369.8,162.4 369.0,154.8 367.8,147.3 366.2,139.9 364.0,132.6 361.4,125.4 358.4,118.4 354.9,111.7 351.0,105.1 346.6,98.9 341.9,92.9 336.8,87.2 331.4,81.8 325.7,76.9 319.6,72.3 313.3,68.0 306.7,64.2 299.8,60.8 292.8,57.9 285.6,55.4 278.3,53.4 270.8,51.8 263.3,50.7 255.7,50.1 248.1,50.0 240.5,50.4 232.9,51.2 225.4,52.5 218.0,54.3 210.8,56.6 203.6,59.3 196.7,62.5 190.0,66.1 183.5,70.1 177.3,74.5 171.4,79.3 165.8,84.5 160.6,90.0 155.7,95.8 151.2,102.0 147.0,108.4 143.3,115.0 140.1,121.9 137.2,129.0 134.9,136.2 132.9,143.6 131.5,151.0 130.5,158.6 130.1,166.2 130.1,173.8 130.5,181.4 131.5,189.0 132.9,196.4 134.9,203.8 137.2,211.0 140.1,218.1 143.3,225.0 147.0,231.6 151.2,238.0 155.7,244.2 160.6,250.0 165.8,255.5 171.4,260.7 177.3,265.5 183.5,269.9 190.0,273.9 196.7,277.5 203.6,280.7 210.8,283.4 218.0,285.7 225.4,287.5 232.9,288.8 240.5,289.6 248.1,290.0 255.7,289.9 263.3,289.3 270.8,288.2 278.3,286.6 285.6,284.6 292.8,282.1 299.8,279.2 306.7,275.8 313.3,272.0 319.6,267.7 325.7,263.1 331.4,258.2 336.8,252.8 341.9,247.1 346.6,241.1 351.0,234.9 354.9,228.3 358.4,221.6 361.4,214.6 364.0,207.4 366.2,200.1 367.8,192.7 369.0,185.2 369.8,177.6 370.0,170.0" fill="none" class="stroke-b dashed" stroke-width="2"/>
<circle cx="249.2" cy="190.7" r="3" class="fill-a"/>
<circle cx="319.8" cy="192.8" r="3" class="fill-a"/>
<circle cx="125.8" cy="250.7" r="3" class="fill-a"/>
<circle cx="253.8" cy="136.8" r="3" class="fill-a"/>
<circle cx="244.5" cy="196.5" r="3" class="fill-a"/>
<circle cx="200.2" cy="231.5" r="3" class="fill-a"/>
<circle cx="283.7" cy="166.3" r="3" class="fill-a"/>
<circle cx="226.0" cy="197.8" r="3" class="fill-a"/>
<circle cx="230.5" cy="212.6" r="3" class="fill-a"/>
<circle cx="259.7" cy="194.1" r="3" class="fill-a"/>
<circle cx="198.5" cy="214.3" r="3" class="fill-a"/>
<circle cx="223.4" cy="181.0" r="3" class="fill-a"/>
<circle cx="292.7" cy="97.6" r="3" class="fill-a"/>
<circle cx="164.0" cy="318.2" r="3" class="fill-a"/>
<circle cx="377.0" cy="100.1" r="3" class="fill-a"/>
<circle cx="199.7" cy="215.5" r="3" class="fill-a"/>
<circle cx="174.9" cy="217.7" r="3" class="fill-a"/>
<circle cx="232.6" cy="176.6" r="3" class="fill-a"/>
<circle cx="243.2" cy="143.9" r="3" class="fill-a"/>
<circle cx="322.3" cy="113.7" r="3" class="fill-a"/>
<circle cx="202.0" cy="149.6" r="3" class="fill-a"/>
<circle cx="314.0" cy="147.0" r="3" class="fill-a"/>
<circle cx="263.7" cy="119.5" r="3" class="fill-a"/>
<circle cx="238.2" cy="131.5" r="3" class="fill-a"/>
<circle cx="335.2" cy="42.0" r="3" class="fill-a"/>
<circle cx="218.0" cy="255.8" r="3" class="fill-a"/>
<circle cx="218.2" cy="155.7" r="3" class="fill-a"/>
<circle cx="226.0" cy="156.0" r="3" class="fill-a"/>
<circle cx="109.6" cy="300.0" r="3" class="fill-a"/>
<circle cx="280.6" cy="168.7" r="3" class="fill-a"/>
<circle cx="216.8" cy="210.6" r="3" class="fill-a"/>
<circle cx="262.2" cy="161.8" r="3" class="fill-a"/>
<circle cx="233.2" cy="227.2" r="3" class="fill-a"/>
<circle cx="383.3" cy="129.1" r="3" class="fill-a"/>
<circle cx="180.4" cy="236.8" r="3" class="fill-a"/>
<circle cx="164.2" cy="256.1" r="3" class="fill-a"/>
<circle cx="283.2" cy="118.7" r="3" class="fill-a"/>
<circle cx="278.2" cy="189.4" r="3" class="fill-a"/>
<circle cx="266.9" cy="86.1" r="3" class="fill-a"/>
<circle cx="221.9" cy="182.3" r="3" class="fill-a"/>
<circle cx="278.8" cy="167.3" r="3" class="fill-a"/>
<circle cx="201.2" cy="222.7" r="3" class="fill-a"/>
<circle cx="295.8" cy="129.3" r="3" class="fill-a"/>
<circle cx="297.9" cy="114.8" r="3" class="fill-a"/>
<circle cx="201.6" cy="250.1" r="3" class="fill-a"/>
<circle cx="181.4" cy="264.0" r="3" class="fill-a"/>
<circle cx="257.1" cy="194.6" r="3" class="fill-a"/>
<circle cx="261.8" cy="156.5" r="3" class="fill-a"/>
<circle cx="288.1" cy="158.6" r="3" class="fill-a"/>
<circle cx="304.2" cy="147.0" r="3" class="fill-a"/>
<circle cx="344.4" cy="85.6" r="3" class="fill-a"/>
<circle cx="209.9" cy="175.6" r="3" class="fill-a"/>
<circle cx="166.5" cy="160.2" r="3" class="fill-a"/>
<circle cx="240.5" cy="196.8" r="3" class="fill-a"/>
<circle cx="163.3" cy="296.4" r="3" class="fill-a"/>
<circle cx="213.0" cy="243.2" r="3" class="fill-a"/>
<circle cx="267.2" cy="227.5" r="3" class="fill-a"/>
<circle cx="201.8" cy="224.2" r="3" class="fill-a"/>
<circle cx="273.2" cy="83.1" r="3" class="fill-a"/>
<circle cx="205.6" cy="212.7" r="3" class="fill-a"/>
<path d="M328.0,240.0 L335.0,254.0 L321.0,254.0 z" class="fill-ink"/>
<text x="342.0" y="253.0" class="fig-bold">corner point</text>
<text x="470" y="40" class="fig-bold">One class, two features</text>
<line x1="470" y1="58" x2="500" y2="58" class="stroke-a" stroke-width="2"/><text x="508" y="62" class="fig-small">true cloud</text>
<line x1="470" y1="78" x2="500" y2="78" class="stroke-b dashed" stroke-width="2"/><text x="508" y="82" class="fig-small">independent model</text>
<text x="470" y="120" class="fig-small">Corner point,</text><text x="470" y="136" class="fig-small">squared distance:</text>
<text x="470" y="156" class="fig-small">independent model: 3.4</text>
<text x="470" y="172" class="fig-small">true cloud: 16.9</text>
</svg>
Figure 2. Two features that rise and fall together (correlation 0.8) make a tilted cloud. Independence replaces it with an upright ellipse (here a circle) that has the same widths. The corner point is rare in the true cloud, at squared distance 16.9, but sits inside the independent model's circle at 3.4, because that model has forgotten the tilt. Both outlines mark squared distance 4 from the centre in their own model.
```

Figure 2 is the same assumption seen as a shape. Two features that rise and fall together make a tilted cloud. When you ignore how they move together, the tilt is lost and you keep only each feature's own width, which is an upright ellipse. The corner point sits where one feature is high and the other low, so it is rare in the real cloud, which almost never goes there. The independent model has no way to know that, and finds the point ordinary. This is the same double counting as "free" and "money", seen from the point's side: each feature alone looks unremarkable, and only the pair is unusual.

## Gaussian naive Bayes: one bell per feature

For measured features, give each class and each feature its own one-dimensional bell. The class mean is the average of that feature over the class. The variance is the average squared gap from that mean, taken over that class's own rows:

$$
\mu_{kj} = \frac{1}{n_k}\sum_{i:\, y_i = k} x_{ij}, \qquad
\sigma^2_{kj} = \frac{1}{n_k}\sum_{i:\, y_i = k} (x_{ij} - \mu_{kj})^2.
$$

Read it aloud: for class $k$ and feature $j$, the centre is the class average, and the width is the average squared gap from that centre. Take the class with values 1 and 3 on one feature. The mean is 2, the gaps are $-1$ and $+1$, and the squared gaps add to 2. Divide by the class count 2 to get variance 1. Dividing by all four rows would give 0.5, which is wrong here. Naive Bayes keeps a separate variance for every class, so each class divides by its own count, unlike the pooled covariance of GDA.

A feature that never changes within a class has variance zero, and the density would divide by zero. So you set a *floor* under every variance. Choose it relative to the other variances, because a floor as tiny as $10^{-9}$ makes any change in that feature look impossible, which is another veto.

To classify, add up one log density per feature, and add the log prior:

$$
\log p(x \mid y = k) = \sum_{j=1}^{d} \log \mathcal N(x_j;\, \mu_{kj},\, \sigma^2_{kj}).
$$

This is the log density from the previous chapters, one feature at a time, where `normal_log_density_1d` does the work. Then the chapter on Bayes' rule finishes it: add the log prior and subtract the log-sum-exp.

## Word counts and smoothing

For counts, give each class a probability for each word, written $\theta_{kj}$ (the Greek letter theta): the chance that a word drawn at random from a class-$k$ document is word $j$. The natural estimate is the share of the class's words that are word $j$. In the toy set, spam uses 8 words in total and "free" 3 times, so $\theta = 3/8$.

Scoring a document takes the sum, over words, of the count times the log probability, plus the log prior. A word seen three times votes three times, because each appearance is a separate draw. (The count of orderings that give the same bag is the same for every class, so it cancels when you compare classes.)

Here is the trap. Spam never uses "meeting" in the toy set, so its estimated probability is 0. Ham never uses "money", so ham's is 0. Take the document "free free money meeting". Spam would score $\log 0 = -\infty$ because of "meeting", and ham $-\infty$ because of "money". One unseen word vetoes a class, whatever the other words say, and here both classes are vetoed, so there is nothing to compare.

Why not add a tiny $\epsilon$ to only the zeros? Because $\log\epsilon$ is a huge negative number, so a single unseen word still decides the whole document alone. The better fix says something true: eight words of evidence are not enough to call a word impossible. *Laplace smoothing* pretends you saw every word an extra $\alpha$ times:

$$
\theta_{kj} = \frac{c_{kj} + \alpha}{\sum_{j'} c_{kj'} + \alpha V}.
$$

Here $c_{kj}$ is the count of word $j$ in class $k$, and the denominator adds $\alpha$ once per word, so the probabilities still add to 1. With three words and $\alpha = 1$, counts $[2, 0, 1]$ become $[3, 1, 2]$, which add to 6, so $\theta = [1/2, 1/6, 1/3]$. The unseen word gets $1/6$. The solid bars in Figure 1 show the effect: spam's total of 8 plus 6 makes 14, so "meeting" gets $1/14$ and "free" drops from $3/8$ to $4/14$. A bigger $\alpha$ trusts the counts less, and $\alpha = 1$ is the usual start, which you can tune with cross-validation.

## Score a document

Back to the document "free free money meeting", with counts $[2, 1, 0, 1, 0, 0]$.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="A flow diagram of scoring one document: word counts, a score for spam and a score for ham, a normalising step, and the final posterior">
<marker id="c031-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
<rect x="10" y="120" width="130" height="90" rx="6" class="fill-soft-a stroke-muted"/>
<text x="75.0" y="138" text-anchor="middle" class="fig-bold">Document</text>
<text x="75.0" y="154" text-anchor="middle" class="fig-small">free 2</text>
<text x="75.0" y="170" text-anchor="middle" class="fig-small">money 1</text>
<text x="75.0" y="186" text-anchor="middle" class="fig-small">meeting 1</text>
<rect x="190" y="40" width="230" height="90" rx="6" class="fill-soft-b stroke-muted"/>
<text x="305.0" y="58" text-anchor="middle" class="fig-bold">Spam score</text>
<text x="305.0" y="74" text-anchor="middle" class="fig-small">2(-1.253) + 1(-1.253) + 1(-2.639)</text>
<text x="305.0" y="90" text-anchor="middle" class="fig-small">+ log 0.5 (-0.693)</text>
<text x="305.0" y="106" text-anchor="middle" class="fig-small">= -7.090</text>
<rect x="190" y="200" width="230" height="90" rx="6" class="fill-soft-a stroke-muted"/>
<text x="305.0" y="218" text-anchor="middle" class="fig-bold">Ham score</text>
<text x="305.0" y="234" text-anchor="middle" class="fig-small">2(-1.946) + 1(-2.639) + 1(-1.253)</text>
<text x="305.0" y="250" text-anchor="middle" class="fig-small">+ log 0.5 (-0.693)</text>
<text x="305.0" y="266" text-anchor="middle" class="fig-small">= -8.477</text>
<line x1="140" y1="150" x2="188" y2="100" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<line x1="140" y1="185" x2="188" y2="225" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<rect x="470" y="120" width="140" height="90" rx="6" class="fill-sunk stroke-muted"/>
<text x="540.0" y="138" text-anchor="middle" class="fig-bold">Normalise</text>
<text x="540.0" y="154" text-anchor="middle" class="fig-small">log-sum-exp</text>
<text x="540.0" y="170" text-anchor="middle" class="fig-small">= -6.867</text>
<line x1="420" y1="90" x2="468" y2="140" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<line x1="420" y1="245" x2="468" y2="190" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<text x="540" y="240" text-anchor="middle" class="fig-bold">spam 0.80</text>
<text x="540" y="258" text-anchor="middle" class="fig-bold">ham 0.20</text>
<line x1="540" y1="212" x2="540" y2="226" class="stroke-ink" marker-end="url(#c031-arrow)"/>
<text x="310" y="316" text-anchor="middle" class="fig-small">Each score adds count times log word probability, once per word, then the class prior.</text>
</svg>
Figure 3. Scoring the document "free free money meeting" with the smoothed word probabilities from the toy set. The word free is counted twice, so it adds its log probability twice. The scores differ by log 4 = 1.386, so spam wins by four to one, which is a posterior of 0.80.
```

Figure 3 follows it through. Spam's score is $2\log\tfrac{4}{14} + \log\tfrac{4}{14} + \log\tfrac{1}{14} + \log 0.5 = -7.090$, and ham's is $-8.477$. They differ by $\log 4$, so spam is four times more likely than ham. The log-sum-exp is $-6.867$, so the posteriors are $e^{-0.223} = 0.80$ and $e^{-1.609} = 0.20$. In numpy, with `X` of counts shaped `(n, V)`:

```python
joint = X @ log_theta.T + log_priors          # shape (n, K)
log_post = joint - log_sum_exp(joint, axis=1)[:, None]
```

> [!KEY]
> Back to ten thousand words. Pretending features are independent given the class turns the class density into a product of per-feature terms, so you store one number per feature and class instead of a covariance matrix. The pretence makes the probabilities too confident, but the ranking is often still right. Laplace smoothing keeps an unseen word from vetoing a class.

> [!RECAP]
> - A bag of words keeps counts and drops order, and independence given the class makes the log density a sum over features.
> - Naive Bayes is overconfident when features are correlated, because correlated evidence is counted twice.
> - Gaussian naive Bayes fits a mean and a variance per class and feature, dividing by the class count, with a floor under every variance.
> - Word models give each class a probability per word, add $\alpha$ to every count, and score by count times log probability.
> - Without smoothing, one unseen word sends a class to $-\infty$.

## Build it step by step

The helpers from the first chapter and `gaussian_nb_log_posterior` are already in `starter.py`. You write three functions, in this order.

1. **`fit_naive_bayes_gaussian(X, y, var_floor)`**. Takes `X` of shape `(n, d)` and integer labels. Returns `means`, `variances` (both `(K, d)`), and `priors` `(K,)`. Use `X[y == k].var(axis=0)` and `np.maximum` with the floor. The usual mistake is `ddof=1`, which divides by the class count minus one.
2. **`fit_naive_bayes_multinomial(X, y, alpha)`**. Takes counts `X` of shape `(n, V)`. Returns `log_theta` `(K, V)` and `log_priors` `(K,)`. Sum the class rows, add `alpha`, divide by the smoothed total, and take the log. The usual mistake is smoothing each row before summing, which adds $\alpha$ once per document.
3. **`multinomial_nb_log_posterior(X, log_theta, log_priors)`**. Returns `(n, K)`. Compute `X @ log_theta.T + log_priors`, then subtract the row log-sum-exp as a column. The usual mistake is passing the priors where the log priors go.

## Cost and scaling

Times use $n$ rows, $d$ features, $K$ classes, and a vocabulary of $V$ words:

| Model | Fit time | Memory |
|---|---|---|
| GDA with a shared covariance | grows like $n d^2$ | $Kd + d^2$ |
| Gaussian naive Bayes | grows like $n d$ | $2Kd$ |
| Word-count naive Bayes | grows like the nonzero counts | $KV$ |

At $V = 10{,}000$, a covariance holds $10^8$ numbers, and a two-class word model holds $2 \times 10^4$.

## Where this breaks

**Features that are not bell-shaped.** A skewed measurement breaks the Gaussian model. Take a log, or use a count model.

**Copied features.** A duplicated column is counted twice, as with "free" and "money", and the probabilities get sharper. Drop near-duplicates first.

**Outliers.** One far point drags a class mean and inflates its variance.

## The exercise

You implement `fit_naive_bayes_gaussian`, `fit_naive_bayes_multinomial`, and `multinomial_nb_log_posterior`. The checks confirm the hand example with variance 1, per-class estimates with unequal class sizes, and a floor under a constant feature. They compare the smoothed word probabilities with the toy set and confirm that "free free money meeting" scores 0.80 for spam. They check that a repeated word votes twice, and that fitted Gaussian naive Bayes agrees with a sum of one-dimensional Gaussians from scipy.
