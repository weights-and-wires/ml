---
title: "Find customer segments and predict who leaves"
slug: 050p-project-segment-customers
part: "Part 3 — Trees, ensembles, and unsupervised learning"
kind: project
summary: Compress a customer table with PCA, find segments with k-means and a Gaussian mixture, then predict churn with bagged trees.
minutes: 180
gpu: false
objectives:
  - Standardise a table with mixed units, and choose how many principal components to keep from the explained-variance curve.
  - Choose k from the elbow of the k-means inertia curve, and judge the clusters against known segments with a purity score.
  - Fit a Gaussian mixture by EM from the k-means centres, and confirm that the log-likelihood never falls.
  - Predict churn with a bagged ensemble of shallow regression trees, and compare its out-of-bag and held-out error with one deep tree.
prereqs:
  - 039-grow-and-predict
  - 042-bagging-and-out-of-bag-error
  - 043-gradient-boosting
  - 045-compress-and-know-what-you-lost
  - 048-k-means
  - 050-em-refit-and-converge
exercise: 050p-project-segment-customers
---

# Find customer segments and predict who leaves

A subscription business has a table of 1,500 customers with ten behavioural columns: how often they log in, how long they stay, what they spend, how many times they contact support. The product team asks two questions. What kinds of customers do we have? And which of them are about to cancel?

The obvious approach is to point a clustering routine at the raw table and feed the raw table to a model. It fails for three reasons. The columns have different units, so the column measured in the hundreds drowns out the one measured in fractions. Some columns carry no structure at all, and they add noise to every distance. And a cancellation rule such as "many support tickets and short sessions" involves two columns at once, which a straight line cannot express.

This project asks you to fix each failure with a tool from this part. You standardise and compress the table, find the segments two ways (hard labels, then soft shares), and predict churn with an ensemble of shallow trees that you score without touching the test rows.

## The brief

Deliver these functions, which together answer both questions:

- `standardise`, `explained_variance_ratio`, `choose_components`, and `pca_scores`, to compress the table.
- `kmeans_plusplus_init`, `kmeans`, `choose_k`, and `purity`, to find hard segments and judge them.
- `gmm_m_step` and `fit_gmm`, to turn the hard segments into soft shares.
- `grow_tree`, `predict_tree`, `fit_bagged_trees`, `bagged_predict`, and `oob_error`, to predict churn.
- `find_segments`, which chains the unsupervised half.

A good answer, in numbers: the pipeline keeps 3 components and picks $k = 4$, the clusters match the planted segments with a purity of at least 0.88 (k-means) and 0.90 (mixture), the log-likelihood never falls during EM, and on held-out customers the ensemble's Brier score beats both the base rate and one deep tree.

## What you will use

- [Principal directions from the SVD](/lessons/045-compress-and-know-what-you-lost) gives you the explained-variance curve and the scores, and teaches why you standardise first.
- [k-means with Lloyd's loop and k-means++ seeding](/lessons/048-k-means) moves each centre to the mean of its points, gives you good starting centres, and gives you the best-of-restarts curve for choosing $k$. You reuse it unchanged.
- [Gaussian mixtures](/lessons/049-gmm-soft-assignment) gives you the E-step that turns distances into shares.
- [EM refits and converges](/lessons/050-em-refit-and-converge) gives you the M-step and the guarantee that the log-likelihood rises.
- [Growing a tree](/lessons/039-grow-and-predict) and [the best split](/lessons/038-gain-and-best-split) give you the recursion and the sorted sweep.
- [Bagging and out-of-bag error](/lessons/042-bagging-and-out-of-bag-error) give you the ensemble and its built-in test set.
- [Gradient boosting](/lessons/043-gradient-boosting) is the alternative ensemble, which you try in "Take it further".

## Meet the data

`make_data(seed)` returns three arrays: `X` with 1,500 rows and 10 columns, `segment` with each customer's planted segment, and `churn` with 0 or 1. Your code never uses `segment` to fit anything. It only uses it afterwards, to grade the clusters.

| Column | Name | Role |
|---|---|---|
| 0 to 7 | `sessions_per_week`, `minutes_per_session`, `monthly_spend`, `support_tickets`, `days_since_login`, `feature_count`, `referrals`, `support_chats` | Noisy mixes of a hidden three-number "taste" vector |
| 8, 9 | `ad_clicks`, `app_version_age` | Pure noise |

Here is what is planted, because your result is judged against it:

- **Four segments** of 35, 30, 20, and 15 percent of customers. Each customer's taste vector sits near its segment's centre, so the segments live in a 3-dimensional subspace of the first eight columns.
- **Wildly different units.** Each column has its own offset and scale, from 0.5 to 150, so the raw table is dominated by whichever column has the largest numbers.
- **Churn rule.** A customer churns with probability 0.05, plus a bump that depends on the segment, plus 0.45 when `support_tickets` is above 0.2 and `minutes_per_session` is below $-0.2$ (both standardised). The overall churn rate is about 19 percent.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="Left: 300 customers on the first two principal directions, coloured by planted segment, forming four overlapping clouds. Right: bars for the ten explained-variance ratios, three tall, two at about one tenth, five small, with a dashed cut line at 0.15">
<rect x="30" y="40" width="270" height="230" class="fill-sunk stroke-grid"/>
<text x="30" y="30" class="fig-bold">Customers on PC1 and PC2</text>
<circle cx="154.2" cy="112.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="259.9" cy="124.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="121.6" cy="143.1" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="166.2" cy="187.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="165.8" cy="193.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="197.3" cy="119.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="122.2" cy="167.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="174.7" cy="187.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="155.0" cy="129.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="243.4" cy="175.4" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="106.1" cy="184.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="148.3" cy="186.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="220.1" cy="73.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="88.4" cy="134.7" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="147.9" cy="110.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="147.2" cy="88.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="88.1" cy="152.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="116.1" cy="140.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="219.3" cy="166.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="214.2" cy="161.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="199.8" cy="126.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="132.7" cy="124.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="145.3" cy="124.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="159.6" cy="122.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="166.3" cy="182.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="176.1" cy="224.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="159.5" cy="94.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="152.8" cy="105.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="148.8" cy="116.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="176.0" cy="126.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="255.5" cy="196.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="77.8" cy="140.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="138.7" cy="115.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="173.1" cy="175.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="198.4" cy="199.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="137.9" cy="126.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="213.5" cy="145.2" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="142.4" cy="234.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="181.1" cy="203.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="158.3" cy="96.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="115.2" cy="195.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="172.2" cy="125.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="212.9" cy="164.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="177.9" cy="85.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="227.6" cy="156.4" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="179.2" cy="104.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="167.8" cy="164.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="164.4" cy="116.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="176.1" cy="187.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="205.8" cy="163.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="167.9" cy="136.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="141.9" cy="119.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="131.7" cy="147.7" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="162.5" cy="90.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="90.9" cy="124.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="227.6" cy="157.5" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="171.3" cy="136.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="117.3" cy="145.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="146.9" cy="84.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="150.3" cy="99.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="253.7" cy="123.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="142.7" cy="170.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="146.8" cy="165.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="173.6" cy="174.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="161.9" cy="103.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="182.9" cy="197.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="215.5" cy="141.4" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="225.7" cy="189.2" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="134.1" cy="173.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="181.4" cy="179.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="248.2" cy="141.3" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="152.8" cy="199.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="63.5" cy="156.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="152.5" cy="203.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="162.6" cy="137.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="116.5" cy="160.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="141.6" cy="195.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="80.1" cy="168.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="77.8" cy="144.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="177.4" cy="190.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="160.6" cy="190.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="224.0" cy="159.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="219.8" cy="181.4" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="141.5" cy="147.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="192.8" cy="66.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="139.0" cy="111.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="146.5" cy="104.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="152.1" cy="135.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="119.4" cy="165.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="119.5" cy="144.7" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="136.7" cy="150.1" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="170.5" cy="125.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="202.6" cy="141.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="158.4" cy="130.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="161.8" cy="165.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="103.8" cy="133.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="159.9" cy="115.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="101.9" cy="170.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="163.5" cy="188.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="176.2" cy="124.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="186.8" cy="179.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="147.5" cy="128.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="168.3" cy="90.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="197.2" cy="124.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="104.8" cy="171.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="96.6" cy="142.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="198.4" cy="129.3" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="115.2" cy="215.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="277.0" cy="172.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="156.2" cy="166.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="197.7" cy="130.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="195.0" cy="203.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="103.6" cy="139.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="186.8" cy="195.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="175.6" cy="114.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="204.4" cy="209.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="99.1" cy="178.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="76.5" cy="155.6" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="227.8" cy="158.2" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="150.8" cy="132.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="118.2" cy="202.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="159.3" cy="113.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="138.6" cy="131.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="173.8" cy="110.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="166.3" cy="190.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="187.3" cy="170.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="153.0" cy="166.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="165.8" cy="166.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="168.6" cy="184.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="162.5" cy="97.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="197.5" cy="108.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="226.5" cy="166.5" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="202.6" cy="158.2" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="203.1" cy="82.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="238.9" cy="162.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="182.1" cy="156.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="163.0" cy="208.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="194.4" cy="247.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="135.1" cy="169.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="148.8" cy="196.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="100.8" cy="176.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="124.1" cy="167.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="227.4" cy="127.4" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="122.1" cy="164.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="138.9" cy="214.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="116.9" cy="162.1" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="147.3" cy="149.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="151.4" cy="119.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="203.7" cy="158.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="243.3" cy="142.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="191.2" cy="225.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="120.5" cy="136.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="190.9" cy="173.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="129.1" cy="172.1" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="181.7" cy="214.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="106.1" cy="207.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="173.3" cy="143.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="208.3" cy="134.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="196.3" cy="162.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="167.3" cy="113.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="185.6" cy="154.5" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="151.6" cy="199.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="96.2" cy="173.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="159.6" cy="162.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="152.8" cy="140.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="140.5" cy="189.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="82.7" cy="128.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="139.7" cy="144.5" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="146.4" cy="124.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="132.6" cy="173.3" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="220.8" cy="194.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="119.4" cy="218.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="208.3" cy="172.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="210.3" cy="159.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="74.5" cy="154.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="184.8" cy="151.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="136.8" cy="147.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="194.4" cy="132.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="146.1" cy="112.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="276.9" cy="164.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="247.2" cy="197.9" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="129.9" cy="168.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="201.1" cy="141.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="131.8" cy="96.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="175.5" cy="229.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="163.7" cy="175.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="169.8" cy="180.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="140.9" cy="187.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="99.0" cy="187.7" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="232.8" cy="96.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="106.6" cy="143.7" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="130.6" cy="151.7" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="87.9" cy="173.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="214.1" cy="89.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="146.1" cy="145.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="221.5" cy="192.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="290.9" cy="162.5" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="96.7" cy="164.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="174.3" cy="116.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="218.0" cy="158.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="144.5" cy="208.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="234.5" cy="127.7" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="206.6" cy="227.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="193.5" cy="132.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="90.5" cy="159.1" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="212.6" cy="125.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="199.3" cy="103.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="196.9" cy="206.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="243.7" cy="217.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="203.4" cy="101.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="158.6" cy="131.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="126.9" cy="119.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="184.3" cy="190.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="279.5" cy="130.3" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="229.1" cy="191.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="155.5" cy="109.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="186.3" cy="121.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="150.4" cy="180.5" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="144.1" cy="177.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="196.3" cy="183.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="138.4" cy="185.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="237.8" cy="190.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="91.7" cy="173.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="152.1" cy="250.3" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="165.1" cy="116.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="50.5" cy="117.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="167.8" cy="193.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="160.7" cy="200.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="184.7" cy="188.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="163.1" cy="182.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="145.9" cy="100.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="223.0" cy="194.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="182.1" cy="115.1" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="61.4" cy="167.5" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="132.4" cy="144.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="165.5" cy="184.2" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="144.6" cy="136.5" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="119.5" cy="181.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="184.2" cy="198.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="172.2" cy="170.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="149.1" cy="190.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="80.3" cy="160.5" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="222.1" cy="139.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="176.1" cy="104.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="89.5" cy="203.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="70.6" cy="183.6" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="140.1" cy="103.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="104.5" cy="137.1" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="157.7" cy="108.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="157.9" cy="78.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="145.0" cy="166.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="196.4" cy="187.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="55.9" cy="156.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="180.7" cy="130.6" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="115.5" cy="160.6" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="167.3" cy="133.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="84.2" cy="143.0" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="167.8" cy="106.0" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="179.1" cy="121.7" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="160.9" cy="215.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="116.7" cy="189.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="234.2" cy="204.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="148.6" cy="121.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="140.5" cy="203.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="153.1" cy="203.6" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="158.0" cy="114.3" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="186.1" cy="170.4" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="107.1" cy="150.5" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="170.3" cy="208.7" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="182.3" cy="226.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="158.4" cy="114.6" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="173.5" cy="151.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="172.0" cy="101.9" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="236.6" cy="175.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="124.2" cy="157.2" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="105.3" cy="181.9" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="197.8" cy="170.1" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="230.9" cy="143.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="226.1" cy="148.8" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="258.2" cy="157.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="235.3" cy="131.0" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="166.4" cy="185.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="204.8" cy="125.5" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="114.7" cy="147.6" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="132.0" cy="102.4" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="212.9" cy="155.8" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="170.8" cy="206.9" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="43.2" cy="128.4" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="238.4" cy="132.1" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="187.7" cy="208.8" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="103.8" cy="180.5" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="149.4" cy="122.2" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="176.8" cy="196.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="180.8" cy="117.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="188.8" cy="191.0" r="3.2" class="fill-a" opacity="0.75"/>
<circle cx="191.6" cy="168.4" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="73.3" cy="115.8" r="3.2" class="fill-b" opacity="0.75"/>
<circle cx="183.1" cy="171.6" r="3.2" class="fill-d" opacity="0.75"/>
<circle cx="115.9" cy="143.8" r="3.2" class="fill-c" opacity="0.75"/>
<circle cx="60.0" cy="164.0" r="3.2" class="fill-c" opacity="0.75"/>
<text x="165.0" y="290" class="fig-small" text-anchor="middle">PC1</text>
<text x="22" y="52" class="fig-small" text-anchor="end">PC2</text>
<circle cx="38" cy="308" r="4" class="fill-a"/><text x="46" y="312" class="fig-small">seg 0</text>
<circle cx="98" cy="308" r="4" class="fill-b"/><text x="106" y="312" class="fig-small">seg 1</text>
<circle cx="158" cy="308" r="4" class="fill-c"/><text x="166" y="312" class="fig-small">seg 2</text>
<circle cx="218" cy="308" r="4" class="fill-d"/><text x="226" y="312" class="fig-small">seg 3</text>
<text x="350" y="30" class="fig-bold">Explained variance by direction</text>
<line x1="350" y1="270" x2="600" y2="270" class="stroke-muted"/>
<rect x="354.0" y="68.4" width="17.0" height="201.6" class="fill-a"/>
<text x="362.5" y="286" class="fig-small" text-anchor="middle">1</text>
<text x="362.5" y="64.4" class="fig-small" text-anchor="middle">0.31</text>
<rect x="379.0" y="100.4" width="17.0" height="169.6" class="fill-a"/>
<text x="387.5" y="286" class="fig-small" text-anchor="middle">2</text>
<text x="387.5" y="96.4" class="fig-small" text-anchor="middle">0.26</text>
<rect x="404.0" y="149.4" width="17.0" height="120.6" class="fill-a"/>
<text x="412.5" y="286" class="fig-small" text-anchor="middle">3</text>
<text x="412.5" y="145.4" class="fig-small" text-anchor="middle">0.18</text>
<rect x="429.0" y="203.4" width="17.0" height="66.6" class="fill-muted"/>
<text x="437.5" y="286" class="fig-small" text-anchor="middle">4</text>
<text x="437.5" y="199.4" class="fig-small" text-anchor="middle">0.10</text>
<rect x="454.0" y="205.3" width="17.0" height="64.7" class="fill-muted"/>
<text x="462.5" y="286" class="fig-small" text-anchor="middle">5</text>
<text x="462.5" y="201.3" class="fig-small" text-anchor="middle">0.10</text>
<rect x="479.0" y="254.1" width="17.0" height="15.9" class="fill-muted"/>
<text x="487.5" y="286" class="fig-small" text-anchor="middle">6</text>
<text x="487.5" y="250.1" class="fig-small" text-anchor="middle">0.02</text>
<rect x="504.0" y="262.5" width="17.0" height="7.5" class="fill-muted"/>
<text x="512.5" y="286" class="fig-small" text-anchor="middle">7</text>
<text x="512.5" y="258.5" class="fig-small" text-anchor="middle">0.01</text>
<rect x="529.0" y="265.8" width="17.0" height="4.2" class="fill-muted"/>
<text x="537.5" y="286" class="fig-small" text-anchor="middle">8</text>
<text x="537.5" y="261.8" class="fig-small" text-anchor="middle">0.01</text>
<rect x="554.0" y="266.1" width="17.0" height="3.9" class="fill-muted"/>
<text x="562.5" y="286" class="fig-small" text-anchor="middle">9</text>
<text x="562.5" y="262.1" class="fig-small" text-anchor="middle">0.01</text>
<rect x="579.0" y="267.3" width="17.0" height="2.7" class="fill-muted"/>
<text x="587.5" y="286" class="fig-small" text-anchor="middle">10</text>
<text x="587.5" y="263.3" class="fig-small" text-anchor="middle">0.00</text>
<line x1="350" y1="171.4" x2="600" y2="171.4" class="stroke-bad dashed"/>
<text x="600" y="166.4" class="fig-small" text-anchor="end">cut at 1.5 / 10 = 0.15</text>
<text x="475.0" y="306" class="fig-small" text-anchor="middle">direction number</text>
</svg>
Figure 1. Left: the planted segments are four overlapping clouds, and PC1 and PC2 already separate most of them. Right: three directions carry real structure, two sit at about one tenth (the two pure-noise columns), and the rest are small.
```

Figure 1 shows the data as you will see it once you compress it. The segments overlap a little, so no method can reach a purity of 1.0, and the bars show why you need a rule for how many directions to keep.

## Plan the build

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="Pipeline: raw table, standardise, PCA, k-means and elbow, Gaussian mixture, then features of standardised columns plus shares feed a bagged ensemble of trees, which gives churn probabilities">
<defs><marker id="p056p-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="10" y="20" width="110" height="70" rx="6" class="fill-soft-a stroke-muted"/>
<text x="65.0" y="42" class="fig-bold" text-anchor="middle">Raw table</text>
<text x="65.0" y="60" class="fig-small" text-anchor="middle">1500 x 10</text>
<text x="65.0" y="75" class="fig-small" text-anchor="middle">mixed units</text>
<rect x="155" y="20" width="110" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="210.0" y="42" class="fig-bold" text-anchor="middle">standardise</text>
<text x="210.0" y="60" class="fig-small" text-anchor="middle">mean 0, sd 1</text>
<text x="210.0" y="75" class="fig-small" text-anchor="middle">per column</text>
<rect x="300" y="20" width="130" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="365.0" y="42" class="fig-bold" text-anchor="middle">PCA by SVD</text>
<text x="365.0" y="60" class="fig-small" text-anchor="middle">keep 3 directions</text>
<text x="365.0" y="75" class="fig-small" text-anchor="middle">(variance curve)</text>
<rect x="465" y="20" width="145" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="537.5" y="42" class="fig-bold" text-anchor="middle">k-means++</text>
<text x="537.5" y="60" class="fig-small" text-anchor="middle">k from the elbow</text>
<text x="537.5" y="75" class="fig-small" text-anchor="middle">hard labels</text>
<line x1="120" y1="55" x2="155" y2="55" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
<line x1="265" y1="55" x2="300" y2="55" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
<line x1="430" y1="55" x2="465" y2="55" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
<rect x="465" y="130" width="145" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="537.5" y="152" class="fig-bold" text-anchor="middle">Gaussian mixture</text>
<text x="537.5" y="170" class="fig-small" text-anchor="middle">EM from k-means</text>
<text x="537.5" y="185" class="fig-small" text-anchor="middle">soft shares, log L up</text>
<line x1="537" y1="90" x2="537" y2="130" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
<rect x="235" y="130" width="175" height="70" rx="6" class="fill-soft-c stroke-muted"/>
<text x="322.5" y="152" class="fig-bold" text-anchor="middle">Features</text>
<text x="322.5" y="170" class="fig-small" text-anchor="middle">10 standardised columns</text>
<text x="322.5" y="185" class="fig-small" text-anchor="middle">+ 4 shares</text>
<line x1="465" y1="165" x2="410" y2="165" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
<path d="M210,90 L210,165 L235,165" class="stroke-muted" fill="none" marker-end="url(#p056p-arrow)"/>
<rect x="235" y="240" width="175" height="70" rx="6" class="fill-soft-b stroke-muted"/>
<text x="322.5" y="262" class="fig-bold" text-anchor="middle">Bagged shallow trees</text>
<text x="322.5" y="280" class="fig-small" text-anchor="middle">100 trees, depth 6</text>
<text x="322.5" y="295" class="fig-small" text-anchor="middle">out-of-bag error</text>
<line x1="322" y1="200" x2="322" y2="240" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
<rect x="465" y="240" width="145" height="70" rx="6" class="fill-soft-b stroke-muted"/>
<text x="537.5" y="262" class="fig-bold" text-anchor="middle">Churn probability</text>
<text x="537.5" y="280" class="fig-small" text-anchor="middle">held-out Brier score</text>
<text x="537.5" y="295" class="fig-small" text-anchor="middle">vs one tree</text>
<line x1="410" y1="275" x2="465" y2="275" class="stroke-muted" marker-end="url(#p056p-arrow)"/>
</svg>
Figure 2. Standardising feeds PCA, PCA feeds both clusterings, and the shares from the mixture join the standardised columns as features for the trees. Churn labels appear only in the last stage, so everything before it is unsupervised.
```

The stages come in this order for one reason each. Standardising comes first because PCA and k-means both measure distances, and distances only mean something when the columns share a unit. PCA comes before clustering because eight of the ten columns say the same three things, and the two noise columns would otherwise add spread that has nothing to do with segments. k-means comes before the mixture because it is cheap and robust, and its centres are good starting means for EM. The trees come last, and they are the only stage that sees churn. Everything before them is unsupervised, so you can fit it on all 1,500 rows without leaking an answer.

## Build it step by step

> [!RECAP]
> - Standardise, because distances mean nothing across units. PCA then splits the variance into directions, and the variance curve tells you how many are real.
> - k-means++ seeds and best-of-restarts make Lloyd's loop trustworthy, and a falling inertia curve with an elbow chooses $k$.
> - EM refits each blob as a share-weighted mean and covariance, and the log-likelihood rises on every round.
> - A tree splits on the best threshold and recurses. Averaging trees grown on bootstrap samples cuts their variance, and the rows each tree skipped give a free test set.

Write the functions in file order. The helpers `assign_nearest`, `lloyd`, `log_gaussian`, `gmm_e_step`, `best_split`, and `brier` are your own code from earlier chapters and are written for you.

1. `standardise(X)` takes a matrix of shape `(n, d)` and returns the same shape with every column at mean 0 and standard deviation 1. Use `X.mean(axis=0)` and `X.std(axis=0)`. Divide by 1 where the deviation is 0, or a constant column becomes `nan`. The common mistake is to standardise with `axis=1`, which mixes the columns you are trying to make comparable.

2. `explained_variance_ratio(Z)` and `choose_components(ratios, factor=1.5)`. The ratio of direction $j$ is $s_j^2 / \sum_i s_i^2$, where $s_j$ are the singular values of the centred data, as in [chapter 045](/lessons/045-compress-and-know-what-you-lost). On standardised data every column carries $1/d$ of the variance, so a direction made of noise sits near $1/d$. Keep a direction when its ratio is above $\text{factor}/d$. For ten columns the cut is $1.5/10 = 0.15$, so ratios of $0.31, 0.26, 0.18, 0.10, 0.10$ keep the first three. The common mistake is a cut of exactly $1/d$, which keeps pure-noise directions that happen to land at 0.101.

3. `pca_scores(Z, k)` returns the coordinates of each row on the top $k$ directions, shape `(n, k)`. Centre `Z`, call `np.linalg.svd(..., full_matrices=False)`, and multiply by `Vt[:k].T`. Do not multiply by `Vt[:k]` without the transpose.

4. `kmeans_plusplus_init(X, k, rng)` returns `k` rows of `X`. Pick the first at random. Draw each next one with probability proportional to its squared distance from the nearest centre chosen so far, and update those distances with `np.minimum`. The common mistake is to use distance instead of squared distance, which seeds too close together.

5. `kmeans(X, k, rng, n_init=5)` runs `lloyd` from `n_init` seeded starts and returns the run with the lowest inertia, as `(centers, labels, inertia)`. Compare the third value `lloyd` returns, not the number of iterations.

6. `choose_k(X, ks, rng, drop=0.15)` computes the best inertia for each $k$ and returns the first $k$ whose next inertia is not below $(1 - \text{drop})$ times its own. In words: stop adding clusters when one more cluster cuts the inertia by less than 15 percent. It returns `(k, inertias)`. Run it on the PCA scores, not the raw table, so the noise columns cannot flatten the curve.

7. `purity(labels, truth)` is the share of rows that belong to the majority planted segment of their cluster. For six rows with segments `[0,0,1,1,1,2]` and clusters `[A,A,A,B,B,B]`, cluster A is led by segment 0 (two rows) and B by segment 1 (two rows), so purity is $4/6 = 0.667$. Renaming the clusters cannot change the answer, because each cluster is matched to whichever segment fills it most. Use `np.bincount(...).max()` per cluster.

8. `gmm_m_step(X, resp, reg_covar)` refits the weights, means, and covariances from the shares, as in [chapter 050](/lessons/050-em-refit-and-converge). The common mistake is to measure each covariance around the old mean, which gives a covariance that is too wide and lets the log-likelihood fall.

9. `fit_gmm(X, init_means, iters=50)` starts from the k-means centres, with equal weights and every covariance equal to `np.var(X, axis=0).mean() * np.eye(d)`. Run the E-step, then alternate the M-step and the E-step, and record the log-likelihood after every E-step, so `history` holds `iters + 1` values. Starting from k-means centres, not random points, is what makes this robust.

10. `grow_tree(X, y, max_depth, min_leaf=5)` and `predict_tree(tree, X)`. A leaf is a dict holding the mean of `y`. An internal node splits on the feature and threshold that `best_split` returns, then recurses on each side with `max_depth - 1`. Stop when depth runs out, when `best_split` returns `None`, or when the gain is 1e-12 or less. For prediction, recurse with the row indices that reach each node instead of walking one row at a time. The common mistake is to pass the full `X` to the children instead of the rows that went left or right.

11. `fit_bagged_trees(X, y, n_trees, max_depth, rng)` grows each tree on `n` rows drawn with replacement and records which rows were drawn in a boolean array `in_bag` of shape `(n_trees, n)`. `bagged_predict(trees, X)` averages the trees. About 63 percent of rows land in any one sample, so about 37 percent are out of the bag.

12. `oob_error(X, y, trees, in_bag)` predicts each row with only the trees that did not draw it, averages those predictions per row, and returns the mean squared error over rows that at least one tree skipped. The common mistake is to predict with every tree, which gives an error that looks too good because most trees saw the row.

13. `find_segments(X, rng)` chains steps 1 to 9 and returns a dict with the standardised table `Z`, `n_components`, the PCA `scores`, `k`, the k-means `labels`, the mixture `resp`, and the log-likelihood `history`.

The trees see the standardised columns plus the mixture shares as features. Hold out the last 500 rows, grow one tree of depth 6 on the first 1,000, and grow 100 bagged trees of depth 6 on the same rows.

## Read your results

Run the starter's `__main__` block with `make_data(0)` and generator seed 0. Your numbers should land close to these.

**The variance curve.** The ratios are $0.31, 0.26, 0.18$, then $0.10, 0.10$, then five values below $0.03$. The first three hold 75 percent of the variance and are the planted taste vector. The two at 0.10 are the two noise columns, each worth one standardised column. If you skip standardising, one direction holds 96 percent of the variance, because the column with the largest units wins. If that happens, your `standardise` is missing or wrong.

**The elbow.** Inertia falls as $11226, 8103, 5707, 3586, 3178, 2912, 2664, 2491$ for $k = 1$ to $8$. Each extra cluster cut it by 28, 30, and 37 percent up to $k = 4$, then by only 11, 8, 9, and 7 percent. The first $k$ whose next step is under 15 percent is 4.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Left: k-means inertia against k for 1 to 8, dropping steeply to 4 then flattening. Right: Gaussian mixture log-likelihood by EM round, rising steeply then flattening">
<rect x="30" y="40" width="260" height="200" class="fill-sunk stroke-grid"/>
<text x="30" y="30" class="fig-bold">k-means inertia against k</text>
<polyline points="40.0,50.0 74.3,114.4 108.6,163.7 142.9,207.5 177.1,215.9 211.4,221.3 245.7,226.5 280.0,230.0" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="40.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="74.3" cy="114.4" r="3.5" class="fill-b"/>
<circle cx="108.6" cy="163.7" r="3.5" class="fill-b"/>
<circle cx="142.9" cy="207.5" r="3.5" class="fill-b"/>
<circle cx="177.1" cy="215.9" r="3.5" class="fill-b"/>
<circle cx="211.4" cy="221.3" r="3.5" class="fill-b"/>
<circle cx="245.7" cy="226.5" r="3.5" class="fill-b"/>
<circle cx="280.0" cy="230.0" r="3.5" class="fill-b"/>
<circle cx="142.9" cy="207.5" r="8" fill="none" class="stroke-bad" stroke-width="2"/>
<text x="160.0" y="258" class="fig-small" text-anchor="middle">k (circled: chosen)</text>
<text x="284" y="56" class="fig-small" text-anchor="end">11226</text>
<text x="284" y="226" class="fig-small" text-anchor="end">2491</text>
<rect x="340" y="40" width="260" height="200" class="fill-sunk stroke-grid"/>
<text x="340" y="30" class="fig-bold">EM log-likelihood by round</text>
<polyline points="350.0,230.0 362.0,104.2 374.0,77.0 386.0,59.1 398.0,52.7 410.0,51.1 422.0,50.6 434.0,50.4 446.0,50.2 458.0,50.1 470.0,50.1 482.0,50.1 494.0,50.0 506.0,50.0 518.0,50.0 530.0,50.0 542.0,50.0 554.0,50.0 566.0,50.0 578.0,50.0 590.0,50.0" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="350.0" cy="230.0" r="3.5" class="fill-b"/>
<circle cx="362.0" cy="104.2" r="3.5" class="fill-b"/>
<circle cx="374.0" cy="77.0" r="3.5" class="fill-b"/>
<circle cx="386.0" cy="59.1" r="3.5" class="fill-b"/>
<circle cx="398.0" cy="52.7" r="3.5" class="fill-b"/>
<circle cx="410.0" cy="51.1" r="3.5" class="fill-b"/>
<circle cx="422.0" cy="50.6" r="3.5" class="fill-b"/>
<circle cx="434.0" cy="50.4" r="3.5" class="fill-b"/>
<circle cx="446.0" cy="50.2" r="3.5" class="fill-b"/>
<circle cx="458.0" cy="50.1" r="3.5" class="fill-b"/>
<circle cx="470.0" cy="50.1" r="3.5" class="fill-b"/>
<circle cx="482.0" cy="50.1" r="3.5" class="fill-b"/>
<circle cx="494.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="506.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="518.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="530.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="542.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="554.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="566.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="578.0" cy="50.0" r="3.5" class="fill-b"/>
<circle cx="590.0" cy="50.0" r="3.5" class="fill-b"/>
<text x="470.0" y="258" class="fig-small" text-anchor="middle">EM round</text>
<text x="594" y="56" class="fig-small" text-anchor="end">-7703</text>
<text x="594" y="226" class="fig-small" text-anchor="end">-8526</text>
</svg>
Figure 3. Left: inertia falls steeply until k = 4, then each extra cluster buys little, so the elbow sits at 4. Right: the mixture's log-likelihood rises on every round and flattens after about ten.
```

**The segments.** k-means on the three PCA scores reaches a purity of 0.937, and the mixture's hard labels reach 0.957. The mixture does a little better, probably because the segments have different spreads and a covariance per blob can follow them. On the raw table with $k = 4$, k-means reaches only 0.64, which is the cost of skipping the first two stages. A purity far below 0.9 on the compressed scores usually means a missing standardise, or a $k$ that is not 4.

**The shares.** The average top share is 0.95, and 8.5 percent of customers have a top share under 0.8. Those are the customers on a boundary, and a campaign aimed at one segment would reach them only partly. This is information a hard label throws away.

**EM.** The log-likelihood climbs from $-8527$ to $-7703$ over 50 rounds. It goes $-8527, -7951, -7826, -7744, -7715, -7708$ in the first five rounds and then crawls. If any round falls, check the covariance centre in `gmm_m_step`.

**Churn.** On the 500 held-out customers, the Brier score $\frac{1}{n}\sum_i (p_i - y_i)^2$ of the base rate (predict 0.19 for everyone) is 0.160, and a least-squares line on the same features scores 0.160, no better. One depth-6 tree scores 0.183, worse than the base rate, because it fits its training rows almost perfectly (0.085) and memorises noise. The 100-tree ensemble scores 0.146. The out-of-bag error on the training rows is 0.142, close to the held-out score, so you could have judged the model without spending the test rows. It also falls as trees are added: 0.204 with 1 tree, 0.180 with 5, 0.151 with 20, 0.145 with 50, and 0.142 with 100.

> [!KEY]
> Each stage fixes one failure of the obvious approach. Standardising stops big units from winning, PCA drops the noise directions, the elbow chooses $k$ from the data, EM gives soft shares, and averaging trees cuts the variance that makes one deep tree worse than a constant.

A forest with an out-of-bag error far below its held-out error can point to leakage. A forest no better than one tree usually means the trees are too shallow or too few.

## Take it further

- Swap the bagged trees for gradient boosting with depth-2 trees, as in [chapter 043](/lessons/043-gradient-boosting). It teaches you how a sequential ensemble compares with an averaging one on the same data.
- Drop the mixture shares from the features and refit the forest. It teaches you how much a feature the trees could already rebuild from the raw columns adds (on this data the held-out Brier score is 0.146 either way).
- Choose $k$ with the mixture's log-likelihood instead of the inertia, and compare. It teaches you why an elbow needs a penalty for model size once the blobs are not round.
- Plot churn rate against the mixture shares for each segment. It teaches you how to turn a clustering into a statement the product team can act on.
- Replace `purity` with the best one-to-one match between clusters and segments. It teaches you what purity forgives when the number of clusters is too large.

## The exercise

You implement every function named in "The brief" in `starter.py`; `make_data` and the helpers from earlier chapters are written for you. The checks run from the smallest piece to the whole pipeline. `standardise` is tested for zero mean, unit deviation, and a constant column. `explained_variance_ratio` is compared with covariance eigenvalues, `choose_components` with a hand-made curve, and `pca_scores` for uncorrelated coordinates with the top variances. `kmeans_plusplus_init` must spread seeds across four distant blobs, `kmeans` and `choose_k` must recover four planted blobs and the elbow at 4, and `purity` must ignore cluster names.

`gmm_m_step` must reproduce the two-point refit from chapter 050, and `fit_gmm` must never lower the log-likelihood. `grow_tree` and `predict_tree` must learn a step function and respect `max_depth`. `fit_bagged_trees`, `bagged_predict`, and `oob_error` are checked for bootstrap masks, averaging, and an honest out-of-bag error. The last two checks run the pipeline on `make_data` and judge it: three components, $k = 4$, purity of at least 0.88 and 0.90, a log-likelihood that never falls, and an ensemble that beats one tree and the base rate on held-out churn.
