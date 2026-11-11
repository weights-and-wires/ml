---
title: "Build and choose a spam filter"
slug: 021p-project-spam-filter
part: "Part 2 — Classical models"
kind: project
summary: Build two spam classifiers, tune each with your own cross-validation, and report an honest test score on mail that selection never saw.
minutes: 180
gpu: false
objectives:
  - Fit multinomial naive Bayes in log space and read the planted spam words back from its log ratios.
  - Choose a smoothing strength and a kernel bandwidth with k-fold cross-validation that you wrote.
  - Read a learning curve to decide whether more mail would help.
  - Report a test error from rows that no selection step touched, and compare it with the best cross-validation score.
prereqs:
  - 012-cross-validation-and-leave-one-out
  - 014-learning-curves-and-winners-curse
  - 018-gda-logistic-and-naive-bayes
  - 020-kernels-and-validity
  - 021-kernel-ridge-and-random-features
exercise: 021p-project-spam-filter
---

# Build and choose a spam filter

A small mail provider has no spam filter. Its customers forward the junk by hand, and the support team spends its mornings answering "why did this one get through?" The founder wants two things by Friday: a filter that works, and a number she can put in a sales deck that is true. If she promises "93% of spam caught" and the filter delivers 85%, the provider loses customers and credibility.

You have 2,000 labelled emails. Each one comes as a vector of word counts plus two dense numbers, the count of links and the share of capital letters. The obvious approach is to try some model, tune it until its score looks good, and report that score. Part 2 taught you why that fails twice over. The tuning chooses the settings that flatter the data you tuned on, and the best of many scores is partly luck.

So this project is not only "fit a classifier". You build two candidate filters, choose each one's settings with cross-validation you write yourself, check whether more data would help, pick a winner, and score it once on mail that nothing in the pipeline has seen. The number you report is the test score, and you also report how far it sits from the best cross-validation score.

## The brief

You deliver five things, all in `starter.py`:

- A multinomial naive Bayes filter on the word counts, with Laplace smoothing, computed in log space.
- A kernel ridge classifier on the two numeric features, with an RBF kernel.
- Your own k-fold cross-validation, used to choose the smoothing strength for naive Bayes and the bandwidth and penalty for the kernel model.
- A learning curve for each model, and a rule that says whether more mail would help.
- One function, `select_and_test`, that runs everything and returns the winner, its best cross-validation error, its test error, and the gap between the two.

A good answer has a test error under 0.11 on every seed from 0 to 4, picks real smoothing rather than a value near zero, and recovers at least seven of the ten planted spam words in the top ten of its naive Bayes log ratios. The reference solution reaches about 0.07 test error. Changing the test labels must leave every selection result unchanged, because no selection step may read them.

## What you will use

- [Cross-validation](/lessons/012-cross-validation-and-leave-one-out) gives each setting an honest score from rows the fit did not use.
- [Learning curves](/lessons/014-learning-curves-and-winners-curse) show whether the error is limited by too little data or by the model.
- [The winner's curse](/lessons/014-learning-curves-and-winners-curse) explains why the best cross-validation score is too good, and why you need a test set that nothing touched.
- [Naive Bayes](/lessons/018-gda-logistic-and-naive-bayes) gives the word model, its smoothing, and the log-space scoring.
- [Kernels as similarity](/lessons/020-kernels-and-validity) and [kernel ridge regression](/lessons/021-kernel-ridge-and-random-features) give the model for curved boundaries; the exercise supplies `rbf_kernel` and `kernel_ridge_fit`, which you wrote in those chapters.
- [Random features](/lessons/021-kernel-ridge-and-random-features) tell you what to do when the kernel matrix gets too big, which is a "Take it further" idea here.

## Meet the data

`make_data(seed)` builds a synthetic mailbox of 2,000 emails and returns a dictionary. The first 600 rows are the development set, used for every choice. The last 1,400 rows are the test set, which you open once.

- `counts`, shape (2000, 300): how many times each of 300 vocabulary words appears in each email. An email has about 30 words.
- `numeric`, shape (2000, 2): `links` and `caps`. The links column is a count-like number between 0 and 6, and `caps` is a ratio between 0 and 0.3. Their units differ by a factor of about 20, which matters for any model that measures distance.
- `y`, shape (2000,): 0 for ham, 1 for spam. About 40% of emails are spam.
- `vocab`, `spam_idx`, `ham_idx`: the word for each column, and the columns of the planted words.

Three patterns are planted, and your results are judged against them:

1. **Ten spammy words** (`free`, `winner`, `prize`, `cash`, `offer`, `click`, `urgent`, `credit`, `bonus`, `deal`) are four times likelier in spam than their neutral weight. **Ten hammy words** (`meeting`, `report`, `lunch`, `project`, `schedule`, `thanks`, `team`, `draft`, `review`, `agenda`) are four times likelier in ham. The other 280 words are neutral and follow the same skewed frequencies in both classes. Their columns are shuffled, so the planted words sit at random positions.
2. **Ham makes a blob and spam makes a ring.** In the numeric features, ham emails cluster around the middle, and spam emails sit at a similar distance from the middle in every direction. The class means are almost equal, so no straight line separates them: a least-squares line on these features misses as often as always answering "ham", at 0.426 test error.
3. **The numeric features are weaker evidence than the words.** The ring and the blob overlap, so even a perfect kernel model errs on a noticeable share of emails.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Left, links and capitals ratio of 200 emails: spam forms a ring around a blob of ham. Right, how often nine words appear per 1000 words in spam and in ham">
<text x="150" y="22" text-anchor="middle" class="fig-bold">numeric features, 200 emails</text>
<line x1="50" y1="255" x2="270" y2="255" class="stroke-muted"/><line x1="50" y1="255" x2="50" y2="40" class="stroke-muted"/>
<text x="50" y="272" text-anchor="middle" class="fig-small">0</text>
<text x="123" y="272" text-anchor="middle" class="fig-small">2</text>
<text x="197" y="272" text-anchor="middle" class="fig-small">4</text>
<text x="270" y="272" text-anchor="middle" class="fig-small">6</text>
<text x="44" y="259" text-anchor="end" class="fig-small">0.0</text>
<text x="44" y="187" text-anchor="end" class="fig-small">0.1</text>
<text x="44" y="116" text-anchor="end" class="fig-small">0.2</text>
<text x="44" y="44" text-anchor="end" class="fig-small">0.3</text>
<text x="160" y="292" text-anchor="middle" class="fig-small">links</text><text x="14" y="150" text-anchor="middle" class="fig-small" transform="rotate(-90 14 150)">capitals ratio</text>
<circle cx="50.0" cy="187.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="163.2" cy="90.4" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="143.5" cy="172.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="124.7" cy="165.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="145.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="87.8" cy="210.4" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="76.8" cy="179.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="185.8" cy="187.6" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="125.4" cy="167.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="134.7" cy="176.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="91.5" cy="147.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="143.3" cy="166.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="94.8" cy="104.9" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="118.6" cy="162.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="110.6" cy="94.9" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="128.3" cy="157.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="105.8" cy="159.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="91.7" cy="141.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="136.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="65.4" cy="156.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="119.0" cy="165.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="77.4" cy="147.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="118.7" cy="166.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="110.9" cy="184.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="141.4" cy="161.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="160.7" cy="178.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="165.8" cy="219.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="103.2" cy="132.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="140.5" cy="190.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="121.3" cy="255.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="73.3" cy="177.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="130.8" cy="161.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="161.4" cy="151.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="126.6" cy="194.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="80.7" cy="157.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="96.5" cy="106.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="128.4" cy="177.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="107.7" cy="150.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="51.9" cy="216.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="188.4" cy="202.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="152.3" cy="205.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="237.6" cy="185.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="142.0" cy="173.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="101.8" cy="179.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="139.5" cy="171.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="204.0" cy="178.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="132.3" cy="164.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="165.2" cy="227.3" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="117.9" cy="165.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="106.2" cy="150.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="184.8" cy="130.4" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="136.7" cy="170.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="175.4" cy="153.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="165.9" cy="208.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="145.1" cy="205.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="125.6" cy="168.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="126.1" cy="155.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="162.8" cy="145.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="112.5" cy="190.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="141.0" cy="223.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="125.4" cy="168.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="138.8" cy="135.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="131.7" cy="156.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="142.6" cy="172.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="185.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="118.8" cy="166.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="129.8" cy="206.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="166.8" cy="77.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="161.1" cy="128.1" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="172.0" cy="221.6" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="93.7" cy="116.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="110.6" cy="161.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="121.3" cy="166.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="106.9" cy="159.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="160.6" cy="149.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="159.1" cy="138.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="126.4" cy="141.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="130.1" cy="160.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="122.7" cy="168.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="84.0" cy="108.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="59.1" cy="217.4" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="138.6" cy="150.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="180.5" cy="158.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="85.2" cy="117.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="97.9" cy="230.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="124.6" cy="169.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="121.7" cy="172.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="160.7" cy="226.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="176.0" cy="200.7" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="177.5" cy="191.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="155.4" cy="223.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="109.8" cy="145.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="172.6" cy="185.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="177.1" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="146.8" cy="117.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="167.8" cy="221.4" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="244.7" cy="138.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="119.5" cy="174.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="142.8" cy="186.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="141.6" cy="168.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="91.2" cy="189.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="228.3" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="102.4" cy="193.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="70.2" cy="246.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="174.0" cy="221.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="124.1" cy="169.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="190.1" cy="114.3" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="154.9" cy="96.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="136.8" cy="151.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="126.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="125.8" cy="170.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="163.7" cy="195.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="68.8" cy="149.3" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="89.7" cy="170.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="122.1" cy="167.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="118.6" cy="156.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="69.1" cy="101.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="174.3" cy="157.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="141.0" cy="112.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="129.9" cy="156.3" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="113.3" cy="158.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="166.2" cy="183.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="116.4" cy="168.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="167.0" cy="255.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="123.6" cy="149.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="151.2" cy="159.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="93.8" cy="255.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="132.3" cy="158.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="146.0" cy="164.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="107.0" cy="118.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="135.3" cy="156.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="84.9" cy="172.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="102.6" cy="154.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="213.2" cy="140.7" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="119.9" cy="176.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="144.1" cy="195.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="138.4" cy="87.1" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="50.0" cy="101.6" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="126.6" cy="160.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="152.8" cy="180.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="161.4" cy="189.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="187.4" cy="128.6" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="145.5" cy="100.3" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="56.3" cy="186.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="179.8" cy="152.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="167.4" cy="166.9" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="120.0" cy="172.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="97.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="139.9" cy="211.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="202.9" cy="128.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="121.2" cy="84.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="119.6" cy="167.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="119.7" cy="166.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="115.1" cy="203.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="60.1" cy="75.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="175.2" cy="223.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="154.5" cy="168.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="128.5" cy="165.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="115.1" cy="249.5" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="95.4" cy="101.4" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="120.5" cy="156.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="165.6" cy="121.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="100.4" cy="151.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="106.5" cy="161.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="72.3" cy="85.7" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="171.7" cy="171.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="84.8" cy="189.9" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="64.5" cy="153.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="120.9" cy="171.6" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="50.0" cy="170.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="116.7" cy="187.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="82.0" cy="172.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="112.1" cy="186.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="102.7" cy="245.1" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="142.2" cy="160.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="179.0" cy="174.1" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="74.0" cy="209.0" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="72.1" cy="215.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="124.8" cy="167.7" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="127.1" cy="170.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="208.7" cy="191.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="126.4" cy="239.7" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="259.5" cy="179.2" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="68.4" cy="144.5" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="64.9" cy="136.8" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="154.9" cy="183.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="129.1" cy="126.6" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="69.1" cy="175.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="134.5" cy="155.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="139.1" cy="157.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="127.3" cy="147.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="91.3" cy="168.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="123.7" cy="162.8" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="110.1" cy="170.2" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="86.9" cy="159.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="154.0" cy="178.7" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="104.4" cy="170.1" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="135.1" cy="125.0" r="3" class="fill-b" fill-opacity="0.8"/>
<circle cx="134.5" cy="88.4" r="3" class="fill-a" fill-opacity="0.8"/>
<circle cx="151.5" cy="69.0" r="3" class="fill-b" fill-opacity="0.8"/>
<text x="196" y="52" class="fig-small fig-bold">spam</text><circle cx="188" cy="48" r="3.5" class="fill-b"/><text x="196" y="68" class="fig-small fig-bold">ham</text><circle cx="188" cy="64" r="3.5" class="fill-a"/>
<text x="460" y="22" text-anchor="middle" class="fig-bold">uses per 1000 words</text>
<line x1="340" y1="255" x2="630" y2="255" class="stroke-muted"/>
<rect x="345" y="211.9" width="12" height="43.1" class="fill-b"/>
<rect x="358" y="244.5" width="12" height="10.5" class="fill-a"/>
<text x="357" y="271" text-anchor="middle" class="fig-small">free</text>
<rect x="376" y="210.0" width="12" height="45.0" class="fill-b"/>
<rect x="389" y="243.0" width="12" height="12.0" class="fill-a"/>
<text x="388" y="271" text-anchor="middle" class="fig-small">prize</text>
<rect x="407" y="210.4" width="12" height="44.6" class="fill-b"/>
<rect x="420" y="243.8" width="12" height="11.2" class="fill-a"/>
<text x="419" y="271" text-anchor="middle" class="fig-small">cash</text>
<rect x="438" y="241.8" width="12" height="13.2" class="fill-b"/>
<rect x="451" y="207.2" width="12" height="47.8" class="fill-a"/>
<text x="450" y="271" text-anchor="middle" class="fig-small">meeting</text>
<rect x="469" y="242.3" width="12" height="12.7" class="fill-b"/>
<rect x="482" y="211.6" width="12" height="43.4" class="fill-a"/>
<text x="481" y="271" text-anchor="middle" class="fig-small">report</text>
<rect x="500" y="243.6" width="12" height="11.4" class="fill-b"/>
<rect x="513" y="210.6" width="12" height="44.4" class="fill-a"/>
<text x="512" y="271" text-anchor="middle" class="fig-small">agenda</text>
<rect x="531" y="65.1" width="12" height="189.9" class="fill-b"/>
<rect x="544" y="71.4" width="12" height="183.6" class="fill-a"/>
<text x="543" y="271" text-anchor="middle" class="fig-small">w0</text>
<rect x="562" y="146.8" width="12" height="108.2" class="fill-b"/>
<rect x="575" y="145.6" width="12" height="109.4" class="fill-a"/>
<text x="574" y="271" text-anchor="middle" class="fig-small">w1</text>
<rect x="593" y="177.7" width="12" height="77.3" class="fill-b"/>
<rect x="606" y="180.0" width="12" height="75.0" class="fill-a"/>
<text x="605" y="271" text-anchor="middle" class="fig-small">w2</text>
<text x="336" y="259" text-anchor="end" class="fig-small">0</text>
<text x="336" y="206" text-anchor="end" class="fig-small">20</text>
<text x="336" y="154" text-anchor="end" class="fig-small">40</text>
<text x="336" y="102" text-anchor="end" class="fig-small">60</text>
<text x="336" y="49" text-anchor="end" class="fig-small">80</text>
</svg>
Figure 1. Left: a straight line cannot split the classes, because spam surrounds the ham blob. Right: three spammy words and three hammy words flip their rates between classes, while the three most common neutral words (w0, w1, w2) stay level.
```

## Plan the build

```figure
<svg viewBox="0 0 640 330" role="img" aria-label="Pipeline: 2000 emails split into 600 development rows and 1400 locked test rows. Development rows feed two models, each tuned by five-fold cross-validation. The lower error wins, is refit on all development rows, and is scored once on the test rows. A learning curve runs beside the selection.">
<defs><marker id="p039p-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="10" y="130" width="110" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="65" y="153" text-anchor="middle" class="fig-bold">2000 emails</text><text x="65" y="171" text-anchor="middle" class="fig-small">make_data</text>
<line x1="120" y1="148" x2="160" y2="95" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<line x1="120" y1="166" x2="160" y2="265" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<rect x="160" y="70" width="110" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="215" y="93" text-anchor="middle" class="fig-bold">dev: 600</text><text x="215" y="111" text-anchor="middle" class="fig-small">all choices</text>
<rect x="160" y="240" width="110" height="52" rx="6" class="fill-sunk stroke-muted"/>
<text x="215" y="263" text-anchor="middle" class="fig-bold">test: 1400</text><text x="215" y="281" text-anchor="middle" class="fig-small">locked until the end</text>
<line x1="270" y1="85" x2="320" y2="55" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<line x1="270" y1="108" x2="320" y2="150" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<rect x="320" y="20" width="150" height="62" rx="6" class="fill-soft-b stroke-b"/>
<text x="395" y="42" text-anchor="middle" class="fig-bold">naive Bayes, words</text><text x="395" y="60" text-anchor="middle" class="fig-small">cv over 7 values of alpha</text><text x="395" y="74" text-anchor="middle" class="fig-small">fit_nb, predict_nb</text>
<rect x="320" y="120" width="150" height="62" rx="6" class="fill-soft-b stroke-b"/>
<text x="395" y="142" text-anchor="middle" class="fig-bold">kernel ridge, numbers</text><text x="395" y="160" text-anchor="middle" class="fig-small">cv over 12 (gamma, lambda)</text><text x="395" y="174" text-anchor="middle" class="fig-small">standardise inside each fold</text>
<line x1="470" y1="52" x2="505" y2="85" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<line x1="470" y1="150" x2="505" y2="115" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<rect x="505" y="70" width="125" height="62" rx="6" class="fill-soft-c stroke-c"/>
<text x="567" y="92" text-anchor="middle" class="fig-bold">pick lower cv</text><text x="567" y="110" text-anchor="middle" class="fig-small">refit on all dev rows</text>
<line x1="567" y1="132" x2="567" y2="205" class="stroke-muted" marker-end="url(#p039p-arrow)"/>
<line x1="270" y1="266" x2="505" y2="236" class="stroke-muted dashed" marker-end="url(#p039p-arrow)"/>
<rect x="505" y="205" width="125" height="62" rx="6" class="fill-soft-d stroke-d"/>
<text x="567" y="227" text-anchor="middle" class="fig-bold">score once</text><text x="567" y="245" text-anchor="middle" class="fig-small">test error, gap to cv</text>
<rect x="320" y="220" width="150" height="52" rx="6" class="fill-sunk stroke-muted dashed"/>
<text x="395" y="242" text-anchor="middle" class="fig-bold">learning curve</text><text x="395" y="259" text-anchor="middle" class="fig-small">more mail, or a better model?</text>
</svg>
Figure 2. Everything that chooses something reads only the 600 development rows. The 1,400 test rows connect to the pipeline at one place, the final score.
```

The stages come in this order because each one protects the next. The word model and the kernel model each need their own settings, and you choose those with cross-validation on the development rows, so the choice never reads the test set. A learning curve costs only a few fits and tells you whether to spend money on more labelled mail before you invest in a fancier model. Only after the winner is fixed do you open the test rows, once, because every look at them that influences a decision makes the final number optimistic.

## Build it step by step

> [!RECAP]
> - Naive Bayes adds `alpha` to every word count, so a word that never appeared in spam does not make the spam score minus infinity.
> - Scores are sums of count times log probability, never products of probabilities, because a 30-word product underflows quickly and a 3,000-word product does not exist as a float.
> - Cross-validation fits on k - 1 folds and scores the held-out fold. Every setting is scored on the same folds, so the comparison is fair.
> - Kernel ridge on a dense feature needs the features on one scale, with the scale learned from training rows only.
> - The best cross-validation score is the minimum of noisy numbers. Report the test score.

Work through the functions in this order. Each check in `tests.py` assumes the earlier ones work.

1. **`fit_nb(X, y, alpha)`** takes counts of shape (n, 300) and labels, and returns `(log_theta, log_priors)` with shapes (2, 300) and (2,). For each class, sum the class's rows, add `alpha` to every word, divide by the smoothed total, then take the log, exactly as in [naive Bayes](/lessons/018-gda-logistic-and-naive-bayes). The most common mistake is to divide by the unsmoothed total, so that rows of `np.exp(log_theta)` no longer sum to 1.

2. **`predict_nb(model, X)`** returns 0 or 1 for each row. Compute `X @ log_theta.T + log_priors` and take the larger column. You can skip the normalising constant because it is the same for both classes. The mistake to avoid is to exponentiate first: a long email then underflows to 0 for both classes, and `argmax` returns class 0 for everything.

3. **`top_spam_words(model, vocab, n_words)`** subtracts the ham row of `log_theta` from the spam row and returns the `n_words` words with the largest difference, strongest first. This difference, the log ratio, is how much one occurrence of the word shifts the score toward spam. Sort with `np.argsort(-ratio)`. A frequent mistake is to rank by spam probability alone, which puts common neutral words on top.

4. **`k_fold_indices(n, k, rng)`** shuffles the row numbers and cuts them into k nearly equal folds, as in [cross-validation](/lessons/012-cross-validation-and-leave-one-out). `np.array_split` handles sizes that do not divide evenly.

5. **`cv_error(fit, predict, X, y, k, seed=0)`** returns the mean held-out error rate. Build the folds once. For each fold, the training rows are all other rows. Fit on those, predict the held-out rows, and average the k fold errors. The mistake is to fit on all rows before the loop, which lets the held-out fold into training and makes every model look excellent. The `error_rate` helper is given.

6. **`grid_search_cv(make_fit, predict, X, y, settings, k)`** calls `cv_error` for each setting and returns the best setting and the array of errors. `make_fit(setting)` returns a `fit` function, so one search routine serves both models. Keep the seed fixed across settings. If each setting draws its own folds, a difference in score might be a difference in folds, not in settings.

7. **`standardise(X_train, X_other)`** subtracts the training mean and divides by the training standard deviation, column by column. Pass the training rows as the first argument even when you standardise something else. If you standardise held-out rows with their own statistics, you move them to where the training rows are, and the model never meets the shift it will see in production.

8. **`fit_krr_classifier(X, y, gamma, lam)`** standardises the features, builds `K = rbf_kernel(Z, Z, gamma)`, and solves for the weights on targets of -1 for ham and +1 for spam with `kernel_ridge_fit`, as in [kernel ridge regression](/lessons/021-kernel-ridge-and-random-features). **`predict_krr(model, X)`** standardises the queries with the training statistics, forms the (m, n) cross-kernel, and returns 1 where the prediction is above 0. The model has to keep the raw training features so `predict_krr` can do this. Skipping standardisation lets `links` dominate the distance, because it varies by about 20 times more than `caps`, and the RBF kernel then ignores the second feature.

9. **`learning_curve(fit, predict, X, y, sizes, n_val)`** holds out the last `n_val` rows, fits on the first `size` rows of the rest for each size, and returns training and validation error rates, following [learning curves](/lessons/014-learning-curves-and-winners-curse). **`more_data_would_help(train_errors, val_errors, gap_tol)`** returns `True` only when the final gap exceeds `gap_tol` and the validation error ended below where it started. A large gap alone is not enough, because a gap beside a validation curve that has stopped falling means more rows will not close it.

10. **`select_and_test(data, k)`** ties everything together. Slice `[:n_dev]` for every selection step. Search seven smoothing values for naive Bayes and twelve `(gamma, lam)` pairs for the kernel model, take the model with the lower best cv error, refit it on all development rows, and score it on `[n_dev:]` exactly once. Return the dictionary described in the docstring. The mistake that ruins the project is to let any test row into a choice, including through a standardisation mean or a threshold picked on test accuracy.

## Read your results

Run `starter.py` once the functions work. These are the numbers the reference solution gives on seed 0.

**Naive Bayes smoothing.** The five-fold error for alpha = 0.01, 0.1, 0.3, 1, 3, 10, and 30 is 0.078, 0.077, 0.077, 0.072, 0.063, 0.067, and 0.130. Cross-validation picks alpha = 3. The curve is a shallow U (Figure 3, left): too little smoothing trusts counts of 0 and 1 for rare words, and too much flattens the real differences between the classes. If your best alpha is 0.01 or 30, check that `fit_nb` normalises after adding `alpha`.

**The kernel model.** Its best five-fold error is 0.140, at gamma = 0.3 and lambda = 0.01. It beats the straight line (0.426) by a wide margin, so the ring is real and the kernel finds it. It is still twice as bad as naive Bayes, because the ring and blob overlap. Without standardisation, accuracy on the test rows with the same settings falls from 0.840 to 0.767. If your kernel model sits near 0.426, it probably ignores one feature or predicts one class.

**The winner and its score.** Naive Bayes wins, on a best cv error of 0.063. Its test error on the 1,400 locked rows is 0.070, so the gap is +0.007. Across seeds 0 to 4 the test errors are 0.070, 0.066, 0.072, 0.070, and 0.076, and the gaps are +0.007, -0.004, -0.003, -0.015, and +0.011, with a mean near zero. The result looks like the winner's curse failing to show up, but two effects cancel. Taking the minimum of seven noisy scores makes the best score optimistic. Cross-validation also trains each fold on 480 emails, not 600, which makes each score pessimistic. In this problem the two offsets are about the same size. The spread of the gaps, about one percentage point either way, is the real message: report 7%, with an uncertainty of a point or so, and not 6.3%.

**The learning curve.** Averaged over seeds 0 to 4, naive Bayes' validation error falls from 0.222 at 25 emails to 0.063 at 400, while its training error stays near 0.03. Validation error is still falling and a gap of 0.026 remains, so more mail should help a little (Figure 3, right). The kernel model's two curves meet at about 0.14 and the final gap is -0.003. More mail will not help it, and a larger model or better features would.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Left, five-fold error of naive Bayes for seven smoothing strengths, lowest at 3. Right, training and validation error as the training set grows from 25 to 400 emails">
<text x="160" y="22" text-anchor="middle" class="fig-bold">cv error against smoothing</text>
<line x1="55" y1="250" x2="285" y2="250" class="stroke-muted"/><line x1="55" y1="250" x2="55" y2="45" class="stroke-muted"/>
<line x1="55" y1="250.0" x2="285" y2="250.0" class="stroke-grid"/><text x="49" y="254" text-anchor="end" class="fig-small">0.00</text>
<line x1="55" y1="176.8" x2="285" y2="176.8" class="stroke-grid"/><text x="49" y="181" text-anchor="end" class="fig-small">0.05</text>
<line x1="55" y1="103.6" x2="285" y2="103.6" class="stroke-grid"/><text x="49" y="108" text-anchor="end" class="fig-small">0.10</text>
<polyline points="75.0,135.3 106.7,137.7 138.3,137.7 170.0,145.1 201.7,157.3 233.3,152.4 265.0,59.6" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="75.0" cy="135.3" r="4.5" class="fill-b"/><text x="75.0" y="268" text-anchor="middle" class="fig-small">0.01</text>
<circle cx="106.7" cy="137.7" r="4.5" class="fill-b"/><text x="106.7" y="268" text-anchor="middle" class="fig-small">0.1</text>
<circle cx="138.3" cy="137.7" r="4.5" class="fill-b"/><text x="138.3" y="268" text-anchor="middle" class="fig-small">0.3</text>
<circle cx="170.0" cy="145.1" r="4.5" class="fill-b"/><text x="170.0" y="268" text-anchor="middle" class="fig-small">1</text>
<circle cx="201.7" cy="157.3" r="4.5" class="fill-bad"/><text x="201.7" y="268" text-anchor="middle" class="fig-small">3</text>
<circle cx="233.3" cy="152.4" r="4.5" class="fill-b"/><text x="233.3" y="268" text-anchor="middle" class="fig-small">10</text>
<circle cx="265.0" cy="59.6" r="4.5" class="fill-b"/><text x="265.0" y="268" text-anchor="middle" class="fig-small">30</text>
<text x="170" y="290" text-anchor="middle" class="fig-small">alpha</text>
<text x="202" y="179" text-anchor="middle" class="fig-small fig-bold">chosen</text>
<text x="480" y="22" text-anchor="middle" class="fig-bold">error against training size</text>
<line x1="360" y1="250" x2="620" y2="250" class="stroke-muted"/><line x1="360" y1="250" x2="360" y2="45" class="stroke-muted"/>
<line x1="360" y1="250.0" x2="620" y2="250.0" class="stroke-grid"/><text x="354" y="254" text-anchor="end" class="fig-small">0.0</text>
<line x1="360" y1="168.0" x2="620" y2="168.0" class="stroke-grid"/><text x="354" y="172" text-anchor="end" class="fig-small">0.1</text>
<line x1="360" y1="86.0" x2="620" y2="86.0" class="stroke-grid"/><text x="354" y="90" text-anchor="end" class="fig-small">0.2</text>
<polyline points="375.0,223.8 421.0,233.6 467.0,232.0 513.0,228.7 559.0,224.3 605.0,219.7" fill="none" class="stroke-a" stroke-width="2"/>
<circle cx="375.0" cy="223.8" r="3.5" class="fill-a"/>
<circle cx="421.0" cy="233.6" r="3.5" class="fill-a"/>
<circle cx="467.0" cy="232.0" r="3.5" class="fill-a"/>
<circle cx="513.0" cy="228.7" r="3.5" class="fill-a"/>
<circle cx="559.0" cy="224.3" r="3.5" class="fill-a"/>
<circle cx="605.0" cy="219.7" r="3.5" class="fill-a"/>
<polyline points="375.0,68.0 421.0,87.6 467.0,131.1 513.0,185.2 559.0,195.9 605.0,198.3" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="375.0" cy="68.0" r="3.5" class="fill-b"/>
<circle cx="421.0" cy="87.6" r="3.5" class="fill-b"/>
<circle cx="467.0" cy="131.1" r="3.5" class="fill-b"/>
<circle cx="513.0" cy="185.2" r="3.5" class="fill-b"/>
<circle cx="559.0" cy="195.9" r="3.5" class="fill-b"/>
<circle cx="605.0" cy="198.3" r="3.5" class="fill-b"/>
<text x="375.0" y="268" text-anchor="middle" class="fig-small">25</text>
<text x="421.0" y="268" text-anchor="middle" class="fig-small">50</text>
<text x="467.0" y="268" text-anchor="middle" class="fig-small">100</text>
<text x="513.0" y="268" text-anchor="middle" class="fig-small">200</text>
<text x="559.0" y="268" text-anchor="middle" class="fig-small">300</text>
<text x="605.0" y="268" text-anchor="middle" class="fig-small">400</text>
<text x="490" y="290" text-anchor="middle" class="fig-small">training emails</text>
<text x="475" y="121" class="fig-small fig-bold">validation</text><text x="475" y="224" class="fig-small fig-bold">training</text>
</svg>
Figure 3. Left: smoothing that is too weak or too strong costs accuracy, so the choice matters. Right: the average over seeds 0 to 4; validation error is still falling at 400 emails while training error is flat.
```

**The planted words.** With alpha = 3, the top ten by log ratio on seed 0 are `bonus`, `urgent`, `click`, `winner`, `cash`, `credit`, `deal`, `free`, `offer`, and `prize`, which is all ten planted spam words, each with a log ratio between 1.05 and 1.53. Across seeds 0 to 4, the filter recovers 10, 10, 9, 10, and 8 of the ten. A word that is rare in the development rows can still land in the top ten by luck, which is why smoothing matters for interpretation too.

## Take it further

- Add the kernel model's score and the naive Bayes score as two features of a stacked classifier, and see whether the ring in the numeric features adds anything the words missed. This teaches you when a weaker model still helps an ensemble.
- Replace the exact kernel with `random_fourier_features` and ridge on those features, and time both at n = 600 and n = 6,000. This shows where the n-squared kernel matrix starts to hurt.
- Wrap `select_and_test` in nested cross-validation from [the winner's curse](/lessons/014-learning-curves-and-winners-curse), and compare its error with the single test error across seeds. This shows how much one 1,400-email test set can mislead.
- Shrink the development set to 100 emails and repeat the five-seed gap table. This shows how the winner's curse grows as the data shrinks.
- Plant the signal in word pairs, such as `free` followed by `money`, which no bag-of-words model can see. This teaches you what the independence assumption in naive Bayes costs.

## The exercise

The checks in `tests.py` verify the pipeline from the smallest piece to the whole. `fit_nb` and `predict_nb` must match a hand calculation, stay finite under smoothing, and not underflow on 3,000-word emails. `top_spam_words` must rank by log ratio. `k_fold_indices` must partition the rows. `cv_error` must keep held-out rows out of training, which a memorising model exposes, and `grid_search_cv` must find a known threshold. `standardise` must use the training statistics, and `fit_krr_classifier` with `predict_krr` must separate a ring from a blob with unequal feature units. `learning_curve` and `more_data_would_help` must score on rows training never sees. Two end-to-end checks run `select_and_test` on seeds 0 to 2. One requires test error below 0.11 and that the word model wins with real smoothing, and the other requires that the planted spam words appear in the top ten. A leak check changes only the test labels and requires every selection result to stay the same.
