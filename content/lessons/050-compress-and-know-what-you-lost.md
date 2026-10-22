---
title: "Compress, and know what you lost"
slug: 050-compress-and-know-what-you-lost
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: The top k directions are the best squeeze to k directions, and the exact error is the sum of the squares of the dropped singular values.
minutes: 40
gpu: false
objectives:
  - Rebuild data from k components, and say what the rebuild error measures.
  - Show that the error equals the sum of the squared singular values you dropped, and explain why no other k directions do better.
  - Choose k from an error budget using the singular values alone.
  - Standardise columns when their units differ, and say when not to.
prereqs:
  - 049-svd-hands-you-every-direction
  - 048-spread-shadows-first-direction
exercise: 050-compress-and-know-what-you-lost
---

# Compress, and know what you lost

You have a table with six readings per row, and you keep only the two best directions. Each row becomes two numbers instead of six, so the table is three times smaller. You can rebuild six numbers from the two, but the rebuilt rows are not the originals. How wrong are they? And could a different choice of two directions have done better?

Without an answer, compression is a leap of faith. You would compress, hope nothing important vanished, and find out later. This chapter gives the exact size of the error before you rebuild anything, and a reason to believe that the top directions from the previous chapter are the best choice.

## What you already know

Five ideas carry this chapter:

- Centring subtracts the column mean, and `pca_svd` returns the components, the scores, and the explained variances, as in "The SVD hands you every direction at once".
- Each point's squared length from the middle splits into a kept part (shadow squared) and a lost part (gap squared), and the total never changes ("Spread, shadows, and the first direction").
- The squared singular value $s_i^2$ is the total squared stretch of the centred data along direction $v_i$.
- A point's score on a direction, times that direction, is its shadow. Adding the mean puts the shadow back in the original table.
- The share of variance held by direction $i$ is $s_i^2 / \sum_j s_j^2$.

## Rebuild from k directions

To rebuild a point from $k$ scores, put each score back on its direction and add the mean:

$$
\hat{X} = \bar{x} + Z V_k^{\top}
$$

Read it aloud as "the rebuilt table is the mean, plus each point's scores times the $k$ directions". $Z$ is the $n \times k$ table of scores and $V_k$ is the $d \times k$ table whose columns are the kept directions. The result has rank $k$, which means every row is a mix of the same $k$ directions plus the mean.

On the three-point example with $k = 1$, the scores $-1.41$, $-1.41$, and $2.83$ rebuild $(1,1)$, $(1,1)$, and $(4,4)$. Figure 1 draws the originals, the rebuilds, and the gaps between them.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="The three example points, their rebuilt positions on the best line, and short dashed segments for the error">
<line x1="167.1" y1="315.0" x2="167.1" y2="15.0" class="stroke-grid" />
<line x1="120.0" y1="272.1" x2="450.0" y2="272.1" class="stroke-grid" />
<line x1="214.3" y1="315.0" x2="214.3" y2="15.0" class="stroke-grid" />
<line x1="120.0" y1="229.3" x2="450.0" y2="229.3" class="stroke-grid" />
<line x1="261.4" y1="315.0" x2="261.4" y2="15.0" class="stroke-grid" />
<line x1="120.0" y1="186.4" x2="450.0" y2="186.4" class="stroke-grid" />
<line x1="308.6" y1="315.0" x2="308.6" y2="15.0" class="stroke-grid" />
<line x1="120.0" y1="143.6" x2="450.0" y2="143.6" class="stroke-grid" />
<line x1="355.7" y1="315.0" x2="355.7" y2="15.0" class="stroke-grid" />
<line x1="120.0" y1="100.7" x2="450.0" y2="100.7" class="stroke-grid" />
<line x1="402.9" y1="315.0" x2="402.9" y2="15.0" class="stroke-grid" />
<line x1="120.0" y1="57.9" x2="450.0" y2="57.9" class="stroke-grid" />
<clipPath id="c050-clip"><rect x="120" y="15" width="330" height="300"/></clipPath><g clip-path="url(#c050-clip)">
<line x1="128.1" y1="307.6" x2="394.8" y2="65.2" class="stroke-b" stroke-width="2.5"/>
</g>
<line x1="261.4" y1="272.1" x2="214.3" y2="229.3" class="stroke-ink dashed" stroke-width="2"/>
<line x1="167.1" y1="186.4" x2="214.3" y2="229.3" class="stroke-ink dashed" stroke-width="2"/>
<circle cx="214.3" cy="229.3" r="6" class="fill-b"/>
<circle cx="355.7" cy="100.7" r="6" class="fill-b"/>
<circle cx="261.4" cy="272.1" r="5.5" class="fill-a"/>
<circle cx="167.1" cy="186.4" r="5.5" class="fill-a"/>
<circle cx="355.7" cy="100.7" r="5.5" class="fill-a"/>
<circle cx="261.4" cy="186.4" r="4" class="fill-ink"/>
<text x="271.4" y="276.1" class="fig-small" text-anchor="start">(2, 0)</text>
<text x="157.1" y="178.4" class="fig-small" text-anchor="end">(0, 2)</text>
<text x="367.7" y="122.7" class="fig-small" text-anchor="start">(4, 4)</text>
<text x="273.4" y="206.4" class="fig-small" text-anchor="start">mean</text>
<text x="174.7" y="211.9" class="fig-small" text-anchor="end">error² 2 each</text>
<text x="462.0" y="110.0" class="fig-small" text-anchor="start">rebuilt points:</text>
<text x="462.0" y="128.0" class="fig-small" text-anchor="start">(1,1), (1,1), (4,4)</text>
<text x="462.0" y="170.0" class="fig-small" text-anchor="start">error² total:</text>
<text x="462.0" y="190.0" class="fig-bold" text-anchor="start">2 + 2 + 0 = 4</text>
</svg>
Figure 1. The three example points (blue) and their rebuilds from one direction (orange) sit on the best line. The dashed segments are the lost part, and their squared lengths add up to 4. The point (4, 4) lies on the line, so nothing is lost there.
```

The squared errors are $(2-1)^2 + (0-1)^2 = 2$, then 2, then 0, so the total error is 4. In numpy:

```python
X_hat = mean + scores @ components.T    # shape (n, d)
error = np.sum((X - X_hat) ** 2)
```

The usual mistake is to leave the mean out. Without it the rebuilt points sit around the origin instead of around the cloud.

## The error is the dropped singular values

You did not need to rebuild to know that the error was 4. The singular values $s = (3.46, 2)$ already said it: $s_2^2 = 4$. Here is why.

Each point's squared length splits into kept and lost parts. Add the parts over every point. Because the directions $v_1, \dots, v_d$ sit at right angles, Pythagoras works in $d$ dimensions too: a point's squared length is the sum of its squared shadows on all $d$ directions. Adding over points, the total of the squared lengths is the sum of all the squared stretches, $s_1^2 + s_2^2 + \cdots$, because each squared stretch is the total squared length of the shadows on one direction. The kept part is the shadows on the $k$ kept directions, which add up to $s_1^2 + \cdots + s_k^2$. Whatever is left is lost:

$$
\sum_{\text{all entries}} (X - \hat{X})^2 = \sum_{i > k} s_i^2 .
$$

Read it aloud as "the total squared error of the rebuild is the sum of the squared singular values you dropped". The singular values here belong to the centred data $X_c$, because the rebuild is about the cloud around its middle. On the example, $12 + 4 = 16$ is the total, 12 is kept, and $4$ is lost.

On the six-dimensional data used in the previous chapter ($n = 300$, two strong directions plus noise), the rebuild error falls as you keep more directions. Figure 2 shows it, and each orange dot is the sum of the dropped $s_i^2$, computed without rebuilding anything. Each sits on top of the bar from the actual rebuild.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Bars of the reconstruction error for k from 0 to 6, with orange dots at the sum of the dropped squared singular values sitting on top of each bar">
<line x1="80.0" y1="240.0" x2="550.0" y2="240.0" class="stroke-grid" />
<text x="72.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="180.3" x2="550.0" y2="180.3" class="stroke-grid" />
<text x="72.0" y="184.3" class="fig-small" text-anchor="end">1000</text>
<line x1="80.0" y1="120.6" x2="550.0" y2="120.6" class="stroke-grid" />
<text x="72.0" y="124.6" class="fig-small" text-anchor="end">2000</text>
<line x1="80.0" y1="60.8" x2="550.0" y2="60.8" class="stroke-grid" />
<text x="72.0" y="64.8" class="fig-small" text-anchor="end">3000</text>
<rect x="93.4" y="35.5" width="40.3" height="204.5" class="fill-a"/>
<circle cx="113.6" cy="35.5" r="5.5" class="fill-b"/>
<text x="113.6" y="25.5" class="fig-small" text-anchor="middle">3424</text>
<text x="113.6" y="258.0" class="fig-small" text-anchor="middle">0</text>
<rect x="160.6" y="195.5" width="40.3" height="44.5" class="fill-a"/>
<circle cx="180.7" cy="195.5" r="5.5" class="fill-b"/>
<text x="180.7" y="185.5" class="fig-small" text-anchor="middle">746</text>
<text x="180.7" y="258.0" class="fig-small" text-anchor="middle">1</text>
<rect x="227.7" y="237.2" width="40.3" height="2.8" class="fill-a"/>
<circle cx="247.9" cy="237.2" r="5.5" class="fill-b"/>
<text x="247.9" y="227.2" class="fig-small" text-anchor="middle">47</text>
<text x="247.9" y="258.0" class="fig-small" text-anchor="middle">2</text>
<rect x="294.9" y="238.0" width="40.3" height="2.0" class="fill-a"/>
<circle cx="315.0" cy="238.0" r="5.5" class="fill-b"/>
<text x="315.0" y="228.0" class="fig-small" text-anchor="middle">34</text>
<text x="315.0" y="258.0" class="fig-small" text-anchor="middle">3</text>
<rect x="362.0" y="238.7" width="40.3" height="1.3" class="fill-a"/>
<circle cx="382.1" cy="238.7" r="5.5" class="fill-b"/>
<text x="382.1" y="228.7" class="fig-small" text-anchor="middle">21</text>
<text x="382.1" y="258.0" class="fig-small" text-anchor="middle">4</text>
<rect x="429.1" y="239.4" width="40.3" height="0.6" class="fill-a"/>
<circle cx="449.3" cy="239.4" r="5.5" class="fill-b"/>
<text x="449.3" y="229.4" class="fig-small" text-anchor="middle">10</text>
<text x="449.3" y="258.0" class="fig-small" text-anchor="middle">5</text>
<rect x="496.3" y="240.0" width="40.3" height="0.0" class="fill-a"/>
<circle cx="516.4" cy="240.0" r="5.5" class="fill-b"/>
<text x="516.4" y="230.0" class="fig-small" text-anchor="middle">0</text>
<text x="516.4" y="258.0" class="fig-small" text-anchor="middle">6</text>
<text x="315.0" y="276.0" text-anchor="middle">k, the number of directions kept</text>
<text transform="rotate(-90 14 130)" x="14.0" y="130.0" text-anchor="middle">total squared error</text>
<rect x="330.0" y="34.0" width="14.0" height="14.0" class="fill-a"/>
<text x="350.0" y="46.0" class="fig-small" text-anchor="start">error of the rebuild</text>
<circle cx="337.0" cy="68.0" r="6" class="fill-b"/>
<text x="350.0" y="72.0" class="fig-small" text-anchor="start">sum of dropped s² values</text>
</svg>
Figure 2. The error of the rebuild (blue bars) for each number of kept directions k on generated six-dimensional data. The orange dots are the sum of the dropped squared singular values, and each one sits on top of its bar. Two directions already cut the error from 3424 to 47.
```

Keeping two directions leaves an error of 47.3 out of a total of 3424.1. The share of the variance you kept is $1 - 47.3/3424.1 = 0.986$, the cumulative share you saw on the scree plot. In numpy, the error from the singular values alone is one line, so you can choose $k$ for an error budget without rebuilding:

```python
s = np.linalg.svd(Xc, compute_uv=False)
error_at_k = np.sum(s[k:] ** 2)
```

## No other k directions do better

Could a different pair of directions lose less? Picture any flat sheet through the middle of the cloud. Each point casts a shadow onto it, and the sheet's kept part is the sum of squared shadows. The best sheet is the one that keeps the most.

The **Eckart-Young theorem** says the sheet spanned by the top $k$ directions keeps the most, and so loses the least. Said for matrices, no matrix of rank $k$ is closer to the centred data than the top-$k$ rebuild, measured by total squared error. The reason is the one from "Spread, shadows, and the first direction": most kept means least lost. To see why the top $k$ directions keep the most, ask how much of each $v_i$ a sheet contains. Call that amount $w_i$, the squared length of $v_i$'s shadow on the sheet. Each $w_i$ lies between 0 and 1, and the $w_i$ add up to $k$, because the sheet has $k$ dimensions. The sheet keeps $w_1 s_1^2 + w_2 s_2^2 + \cdots$, a weighted sum of the squared stretches. You can hand out a total weight of $k$, at most 1 per direction, and the sum is largest when the weight goes to the $k$ biggest stretches. That is exactly the sheet spanned by $v_1, \dots, v_k$, the same weighted-average argument that made the first direction win in the previous chapter.

You can test the theorem rather than trust it. Draw random sheets of two directions in the six-dimensional space, rebuild from each, and compare. Every one of them loses more than 47.3. The exercise runs this test.

> [!KEY]
> Keeping the top $k$ directions loses exactly $\sum_{i>k} s_i^2$, and no other $k$ directions lose less. You know the error before you rebuild, and you know that no other choice could have been better.

## Standardise when units differ

PCA measures spread, and spread depends on units. Take the earlier three-point example and write the second reading in hundredths: $(2, 0)$, $(0, 200)$, and $(4, 400)$. Nothing about the cloud has changed except the ruler. The centred rows are $(0, -200)$, $(-2, 0)$, and $(2, 200)$, the variances are 4 and 40000, and the first direction is $(0.005, 1.000)$, which is the second column and nothing else.

The fix is to **standardise**: centre each column, then divide it by its standard deviation, which is the square root of its variance. The columns are then $(0, -1, 1)$ and $(-1, 0, 1)$, with variance 1 each, and the first direction returns to $(0.71, 0.71)$ with variances 1.5 and 0.5.

```python
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
```

Standardise when the columns are in different units, or when their sizes are arbitrary. Do not standardise when the columns share a unit and the size is the signal, such as pixel brightnesses. Dividing by the standard deviation there would blow up the quiet, noisy pixels to the same weight as the informative ones.

> [!RECAP]
> - Rebuild with `mean + scores @ components.T`, and never leave out the mean.
> - The total squared error of a rank-$k$ rebuild is `np.sum(s[k:] ** 2)`, with $s$ from the centred data.
> - No other $k$ directions lose less, so the top ones are optimal. That is the Eckart-Young theorem.
> - The share of variance you kept is $1 - \text{error}/\text{total}$.
> - Standardise columns when units differ, because spread depends on the ruler.

## Build it step by step

`center`, `pca_svd`, and `explained_variance_ratio` are written for you in the starter, from earlier chapters. Build three functions, in this order.

1. **`reconstruct(mean, components, scores)`** takes a mean `(d,)`, components `(d, k)`, and scores `(n, k)`, and returns the rebuilt table `(n, d)`. The usual mistake is leaving out the mean. The shape of the code:

    ```python
    def reconstruct(mean, components, scores):
        # The mean plus scores times components transposed.
    ```

2. **`reconstruction_error(X, k)`** returns the total squared error of the rank-$k$ rebuild as a float. Do not use the singular-value formula inside it. Run `pca_svd`, rebuild with your `reconstruct`, and add up the squared differences from `X`. The checks then confirm the formula for you. The usual mistake is to shortcut with the singular-value formula, which makes the check against that formula circular. The shape of the code:

    ```python
    def reconstruction_error(X, k):
        # pca_svd, then reconstruct, then the sum of squared differences.
    ```

3. **`standardise(X)`** returns the centred table with each column divided by its standard deviation (`ddof=1`), shape `(n, d)`. A column that is constant has standard deviation 0, so leave its entries at 0 rather than dividing. The usual mistake is dividing by the variance instead of the standard deviation. The shape of the code:

    ```python
    def standardise(X):
        # Centre, then divide by the column standard deviation (ddof=1).
    ```

## Cost and scaling

Three ways to get the directions cost different amounts for $n$ rows and $d$ columns:

| Route | How the work grows | Notes |
|---|---|---|
| SVD of $X_c$ | About $n\,d\,\min(n, d)$ | The default, and the most accurate. |
| Eigenvectors of $X_c^{\top}X_c$ | About $n\,d^2 + d^3$ | Forming the product squares the data and loses digits. |
| Reading the error from `s` | Nothing extra | The singular values are already computed. |

The table $X$ uses about $n\,d$ numbers. The full $U$ adds about $n\,\min(n, d)$ more.

## Where this breaks

Figure 3 shows three failures.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three failures of PCA: a far-off point tilting the first line, a curved cloud cut by a straight line, and a column in large units taking over the first direction">
<defs></defs>
<rect x="15.0" y="20.0" width="175.0" height="175.0" class="stroke-grid" fill="none"/>
<clipPath id="c050-ca"><rect x="15" y="20" width="175" height="175"/></clipPath><g clip-path="url(#c050-ca)">
<line x1="-135.9" y1="173.3" x2="341.0" y2="154.9" class="stroke-a" stroke-width="2.5"/>
<line x1="-92.4" y1="297.0" x2="299.8" y2="25.0" class="stroke-b" stroke-width="2.5"/>
</g>
<circle cx="110.7" cy="171.0" r="2.8" class="fill-a"/>
<circle cx="122.1" cy="167.5" r="2.8" class="fill-a"/>
<circle cx="110.4" cy="163.6" r="2.8" class="fill-a"/>
<circle cx="71.4" cy="169.2" r="2.8" class="fill-a"/>
<circle cx="124.1" cy="163.8" r="2.8" class="fill-a"/>
<circle cx="113.2" cy="162.6" r="2.8" class="fill-a"/>
<circle cx="89.7" cy="163.0" r="2.8" class="fill-a"/>
<circle cx="116.4" cy="166.4" r="2.8" class="fill-a"/>
<circle cx="111.2" cy="159.4" r="2.8" class="fill-a"/>
<circle cx="109.5" cy="157.5" r="2.8" class="fill-a"/>
<circle cx="103.2" cy="161.1" r="2.8" class="fill-a"/>
<circle cx="115.5" cy="168.4" r="2.8" class="fill-a"/>
<circle cx="84.9" cy="158.5" r="2.8" class="fill-a"/>
<circle cx="98.6" cy="166.4" r="2.8" class="fill-a"/>
<circle cx="91.0" cy="157.6" r="2.8" class="fill-a"/>
<circle cx="116.8" cy="170.0" r="2.8" class="fill-a"/>
<circle cx="103.4" cy="157.4" r="2.8" class="fill-a"/>
<circle cx="95.5" cy="163.3" r="2.8" class="fill-a"/>
<circle cx="83.8" cy="171.1" r="2.8" class="fill-a"/>
<circle cx="96.4" cy="165.2" r="2.8" class="fill-a"/>
<circle cx="102.7" cy="162.8" r="2.8" class="fill-a"/>
<circle cx="95.9" cy="161.4" r="2.8" class="fill-a"/>
<circle cx="133.4" cy="169.4" r="2.8" class="fill-a"/>
<circle cx="126.5" cy="170.2" r="2.8" class="fill-a"/>
<circle cx="37.8" cy="161.9" r="2.8" class="fill-a"/>
<circle cx="57.4" cy="166.2" r="2.8" class="fill-a"/>
<circle cx="98.3" cy="161.7" r="2.8" class="fill-a"/>
<circle cx="92.4" cy="158.3" r="2.8" class="fill-a"/>
<circle cx="107.6" cy="173.7" r="2.8" class="fill-a"/>
<circle cx="107.7" cy="161.6" r="2.8" class="fill-a"/>
<circle cx="153.0" cy="155.4" r="2.8" class="fill-a"/>
<circle cx="76.0" cy="165.1" r="2.8" class="fill-a"/>
<circle cx="93.5" cy="168.3" r="2.8" class="fill-a"/>
<circle cx="151.2" cy="158.4" r="2.8" class="fill-a"/>
<circle cx="117.9" cy="161.6" r="2.8" class="fill-a"/>
<circle cx="118.3" cy="157.5" r="2.8" class="fill-a"/>
<circle cx="90.2" cy="165.4" r="2.8" class="fill-a"/>
<circle cx="63.2" cy="172.6" r="2.8" class="fill-a"/>
<circle cx="106.5" cy="163.9" r="2.8" class="fill-a"/>
<circle cx="105.1" cy="166.0" r="2.8" class="fill-a"/>
<circle cx="150.2" cy="35.9" r="5" class="fill-bad"/>
<text x="142.2" y="39.9" class="fig-small" text-anchor="end">outlier</text>
<text x="102.0" y="214.0" class="fig-bold" text-anchor="middle">one far point</text>
<text x="102.0" y="232.0" class="fig-small" text-anchor="middle">blue 2°, orange 35°</text>
<rect x="215.0" y="20.0" width="175.0" height="175.0" class="stroke-grid" fill="none"/>
<clipPath id="c050-cb"><rect x="215" y="20" width="175" height="175"/></clipPath><g clip-path="url(#c050-cb)">
<line x1="138.7" y1="114.2" x2="466.8" y2="123.6" class="stroke-b" stroke-width="2.5"/>
</g>
<circle cx="357.2" cy="164.4" r="3" class="fill-a"/>
<circle cx="355.4" cy="161.0" r="3" class="fill-a"/>
<circle cx="357.7" cy="144.7" r="3" class="fill-a"/>
<circle cx="352.2" cy="138.0" r="3" class="fill-a"/>
<circle cx="350.8" cy="133.9" r="3" class="fill-a"/>
<circle cx="349.0" cy="126.5" r="3" class="fill-a"/>
<circle cx="343.5" cy="121.3" r="3" class="fill-a"/>
<circle cx="341.0" cy="113.8" r="3" class="fill-a"/>
<circle cx="336.9" cy="109.7" r="3" class="fill-a"/>
<circle cx="331.7" cy="105.8" r="3" class="fill-a"/>
<circle cx="328.0" cy="99.3" r="3" class="fill-a"/>
<circle cx="322.1" cy="95.6" r="3" class="fill-a"/>
<circle cx="314.6" cy="92.3" r="3" class="fill-a"/>
<circle cx="314.2" cy="96.4" r="3" class="fill-a"/>
<circle cx="302.4" cy="95.3" r="3" class="fill-a"/>
<circle cx="301.1" cy="91.4" r="3" class="fill-a"/>
<circle cx="296.0" cy="90.8" r="3" class="fill-a"/>
<circle cx="289.1" cy="93.5" r="3" class="fill-a"/>
<circle cx="283.2" cy="94.5" r="3" class="fill-a"/>
<circle cx="276.5" cy="100.7" r="3" class="fill-a"/>
<circle cx="273.9" cy="99.4" r="3" class="fill-a"/>
<circle cx="267.7" cy="110.8" r="3" class="fill-a"/>
<circle cx="263.9" cy="111.2" r="3" class="fill-a"/>
<circle cx="260.6" cy="116.2" r="3" class="fill-a"/>
<circle cx="254.5" cy="123.1" r="3" class="fill-a"/>
<circle cx="256.4" cy="136.0" r="3" class="fill-a"/>
<circle cx="252.5" cy="136.9" r="3" class="fill-a"/>
<circle cx="250.4" cy="145.0" r="3" class="fill-a"/>
<circle cx="252.8" cy="154.6" r="3" class="fill-a"/>
<circle cx="247.6" cy="165.0" r="3" class="fill-a"/>
<text x="302.0" y="214.0" class="fig-bold" text-anchor="middle">a curved cloud</text>
<text x="302.0" y="232.0" class="fig-small" text-anchor="middle">a line cuts across it</text>
<rect x="415.0" y="20.0" width="175.0" height="175.0" class="stroke-grid" fill="none"/>
<rect x="435.0" y="174.5" width="22.0" height="0.5" class="fill-a"/>
<text x="446.0" y="169.5" class="fig-small" text-anchor="middle">0.01</text>
<rect x="463.0" y="85.0" width="22.0" height="90.0" class="fill-b"/>
<text x="474.0" y="80.0" class="fig-small" text-anchor="middle">1.00</text>
<text x="460.0" y="189.0" class="fig-small" text-anchor="middle">raw</text>
<rect x="515.0" y="111.4" width="22.0" height="63.6" class="fill-a"/>
<text x="526.0" y="106.4" class="fig-small" text-anchor="middle">0.71</text>
<rect x="543.0" y="111.4" width="22.0" height="63.6" class="fill-b"/>
<text x="554.0" y="106.4" class="fig-small" text-anchor="middle">0.71</text>
<text x="540.0" y="189.0" class="fig-small" text-anchor="middle">standardised</text>
<text x="435.0" y="36.0" class="fig-small" text-anchor="start">weight of each column</text>
<text x="435.0" y="52.0" class="fig-small" text-anchor="start">in the first direction</text>
<text x="502.0" y="214.0" class="fig-bold" text-anchor="middle">mixed units</text>
<text x="502.0" y="232.0" class="fig-small" text-anchor="middle">blue: column 1, orange: column 2</text>
</svg>
Figure 3. Left: one far-off point tilts the first line from 2° (blue) to 35° (orange). Middle: a straight line cannot follow an arc. Right: when one column is in much larger units, it takes over the first direction, and standardising restores an even split.
```

- A far-off point drags a direction, because variance squares distance. Remove it, or use a robust variant.
- A curved cloud, such as an arc or a spiral, has no flat sheet that fits. Kernel PCA handles curves.
- When two singular values are nearly equal, which of the two directions you keep is unstable, even though the plane they span is fine.

## The exercise

You write `reconstruct`, `reconstruction_error`, and `standardise`. The checks confirm that the rebuild on the example gives $(1,1)$, $(1,1)$, and $(4,4)$, and that keeping every direction gives back $X$ exactly. They check that the error equals $\sum_{i>k} s_i^2$ for several $k$, that random sheets of $k$ directions never beat it, and that one minus the error over the total equals the cumulative share. Finally they check that standardised columns have mean 0 and standard deviation 1, and that standardising removes the effect of a change of units.
