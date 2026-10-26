---
title: Explore before you model
slug: 004p-project-explore-before-you-model
part: "Part 1 — Foundations"
kind: project
summary: Load, clean, summarise, and plot a messy export of rental listings, then show that cleaning makes a straight-line rent model honest.
minutes: 180
gpu: false
objectives:
  - Load a CSV into a pandas DataFrame, and profile its shape, dtypes, missing values, and duplicates.
  - Clean money, units, categories, dates, and yes/no columns, and remove impossible values and typo outliers with a stated rule.
  - Summarise with groupby and pivot tables, and plot with matplotlib figures that you can test.
  - Find a hidden furnished premium, a seasonal peak, and a nonlinear distance effect, and show that cleaning lowers the honest test error.
prereqs:
  - 001-scoring-a-guess
  - 002-honest-scores
  - 003-fitting-curves
  - 004-bias-variance-noise
exercise: 004p-project-explore-before-you-model
---

# Explore before you model

A letting agency has exported about two thousand rental listings from its booking system and sent you the file. It wants to know what drives the weekly rent of a flat, so that it can price future listings and spot bad ones. Nobody has built a model yet, and that is the point: before you fit anything, you have to look at the data.

The obvious approach is to fit a straight line of rent on size and read off the error. That fails here, because the export is a mess. Some rents are per week and some are per month. Some sizes are in square metres and some in square feet. A flat's neighbourhood is spelled ninety-three different ways. A few rents have an extra zero, and 70 rows appear twice. A line fitted to that table describes the mess, not the market.

In this project you do the work that comes before every model in the rest of the course. You load the file, check what is in it, clean it, summarise it, plot it, and then close the loop with the first four chapters: you fit the same straight line on the raw rows and on the cleaned rows, and score both honestly.

## The brief

You deliver, as functions in `starter.py`:

- a loader and a profile of the raw export;
- one cleaning function per kind of mess, and a `clean_listings` that wires them together;
- features, groupby and pivot summaries, and four plotting functions that return matplotlib figures;
- a dictionary of key insights;
- a comparison of two straight-line fits, one trained on the raw rows and one on the cleaned rows.

The checks judge the answer in numbers. A correct cleaning of seed 0 keeps 1,919 of the 2,070 rows, and every rent is in weekly pounds to within 1 percent. The insights must find the planted effects described below. The straight line trained on cleaned rows must reach a test error under half that of the line trained on raw rows, on seeds 0 to 4.

## What you will use

- [Scoring a guess](/lessons/001-scoring-a-guess) gives you the squared loss behind the root-mean-square error (RMSE) that scores both lines.
- [Honest scores](/lessons/002-honest-scores) tells you to score on flats held back from fitting, which is how you compare the two lines fairly.
- [Fitting a curve](/lessons/003-fitting-curves) gives you `fit_polynomial` with degree 1, which is the straight line.
- [Bias, variance, and noise](/lessons/004-bias-variance-noise) explains why noise sets a floor on the test error, so a straight line on size alone cannot reach zero, and why a clean score still has to be judged against that floor.

## Meet the data

`make_data(seed)` returns the export as one string of CSV text, 2,070 lines under a header of ten columns. You read it with `pd.read_csv(io.StringIO(text))`. Every number is made by a seeded generator, so seed 0 always gives the same listings. The columns are:

| Column | Meaning |
|---|---|
| `listing_id` | A number that identifies the flat. |
| `listed_on` | The date the flat was listed, in 2023. |
| `neighbourhood` | One of eight neighbourhoods. |
| `size` | The floor area as text, sometimes with a unit. |
| `size_unit` | The unit of `size`, or blank. |
| `bedrooms` | The number of bedrooms. |
| `dist_station_km` | The distance to the nearest station in kilometres. |
| `furnished` | Whether the flat is furnished. |
| `rent` | The rent as text, such as `£1,250`. |
| `rent_period` | `week`, `month`, or blank. |

### What is planted

The generator plants mess and patterns, and the checks judge your result against them. Counts below are of the 2,000 distinct listings.

- **Rents** are text such as `£1,250`. About a quarter are per month, flagged by `rent_period` or, when that is blank, by a ` pcm` suffix. Blank-period weekly rents carry ` pw`.
- **Sizes** are text such as `54.1`, `54.1 m2`, or `582 sqft`. About 22 percent are in square feet, named by `size_unit` or by the suffix. Six sizes are 0, four are between 1 and 5, and three are in the thousands.
- **Typos.** Eight rents have an extra zero.
- **Missing values.** Twenty rents, 40 sizes, 40 dates, 60 furnished flags, 90 distances, and 100 bedroom counts are blank.
- **Spelling.** Neighbourhood names vary in case and stray spaces, and some are misspelt. Furnished is `Y`, `N`, `yes`, `no`, `True`, or `False` in mixed case. Dates use three formats, day first.
- **Duplicates.** Seventy listings appear twice as exact copies.

Four patterns are planted underneath the mess, and the project is to find them:

- Rent per square metre differs by neighbourhood, from Docklands at the top to Northway at the bottom.
- Furnished flats cost 12 percent more than unfurnished flats of the same neighbourhood and size. They are mostly in cheap neighbourhoods and are mostly small, so they look cheaper overall.
- Rent peaks in September.
- Rent falls steeply in the first kilometre or two from a station and is nearly flat after that.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Two histograms of rent. Before cleaning, most listings sit near 100 to 300 with a second hump near 900 for monthly rents and a tail out to 2,900 for typos. After cleaning, one bell-shaped hump with a right tail between 53 and 537 pounds a week.">
<text x="170.0" y="38" class="fig-bold" text-anchor="middle">Before: rent as written</text>
<line x1="60" y1="215" x2="60" y2="60" class="stroke-muted"/>
<line x1="56" y1="215.0" x2="60" y2="215.0" class="stroke-muted"/><text x="53" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="56" y1="170.7" x2="60" y2="170.7" class="stroke-muted"/><text x="53" y="174.7" class="fig-small" text-anchor="end">200</text>
<line x1="56" y1="126.4" x2="60" y2="126.4" class="stroke-muted"/><text x="53" y="130.4" class="fig-small" text-anchor="end">400</text>
<line x1="56" y1="82.1" x2="60" y2="82.1" class="stroke-muted"/><text x="53" y="86.1" class="fig-small" text-anchor="end">600</text>
<line x1="60" y1="215" x2="280" y2="215" class="stroke-muted"/>
<rect x="60.0" y="172.5" width="4.9" height="42.5" class="fill-bad"/>
<rect x="65.5" y="71.1" width="4.9" height="143.9" class="fill-bad"/>
<rect x="71.0" y="123.1" width="4.9" height="91.9" class="fill-bad"/>
<rect x="76.5" y="177.8" width="4.9" height="37.2" class="fill-bad"/>
<rect x="82.0" y="201.9" width="4.9" height="13.1" class="fill-bad"/>
<rect x="87.5" y="205.9" width="4.9" height="9.1" class="fill-bad"/>
<rect x="93.0" y="209.0" width="4.9" height="6.0" class="fill-bad"/>
<rect x="98.5" y="205.0" width="4.9" height="10.0" class="fill-bad"/>
<rect x="104.0" y="203.7" width="4.9" height="11.3" class="fill-bad"/>
<rect x="109.5" y="201.7" width="4.9" height="13.3" class="fill-bad"/>
<rect x="115.0" y="203.3" width="4.9" height="11.7" class="fill-bad"/>
<rect x="120.5" y="205.0" width="4.9" height="10.0" class="fill-bad"/>
<rect x="126.0" y="204.4" width="4.9" height="10.6" class="fill-bad"/>
<rect x="131.5" y="209.7" width="4.9" height="5.3" class="fill-bad"/>
<rect x="137.0" y="207.0" width="4.9" height="8.0" class="fill-bad"/>
<rect x="142.5" y="209.9" width="4.9" height="5.1" class="fill-bad"/>
<rect x="148.0" y="212.8" width="4.9" height="2.2" class="fill-bad"/>
<rect x="153.5" y="211.5" width="4.9" height="3.5" class="fill-bad"/>
<rect x="159.0" y="211.7" width="4.9" height="3.3" class="fill-bad"/>
<rect x="164.5" y="212.3" width="4.9" height="2.7" class="fill-bad"/>
<rect x="170.0" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="175.5" y="213.9" width="4.9" height="1.1" class="fill-bad"/>
<rect x="181.0" y="214.6" width="4.9" height="0.4" class="fill-bad"/>
<rect x="186.5" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<rect x="192.0" y="213.9" width="4.9" height="1.1" class="fill-bad"/>
<rect x="197.5" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="203.0" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<rect x="208.5" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="214.0" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<rect x="219.5" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<rect x="225.0" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<rect x="230.5" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="236.0" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="241.5" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="247.0" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="252.5" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<rect x="258.0" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="263.5" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="269.0" y="215.0" width="4.9" height="0.0" class="fill-bad"/>
<rect x="274.5" y="214.8" width="4.9" height="0.2" class="fill-bad"/>
<line x1="55.9" y1="215" x2="55.9" y2="219" class="stroke-muted"/><text x="55.9" y="233" class="fig-small" text-anchor="middle">0</text>
<line x1="133.2" y1="215" x2="133.2" y2="219" class="stroke-muted"/><text x="133.2" y="233" class="fig-small" text-anchor="middle">1000</text>
<line x1="210.5" y1="215" x2="210.5" y2="219" class="stroke-muted"/><text x="210.5" y="233" class="fig-small" text-anchor="middle">2000</text>
<line x1="287.7" y1="215" x2="287.7" y2="219" class="stroke-muted"/><text x="287.7" y="233" class="fig-small" text-anchor="middle">3000</text>
<text x="470.0" y="38" class="fig-bold" text-anchor="middle">After: weekly £</text>
<line x1="360" y1="215" x2="360" y2="60" class="stroke-muted"/>
<line x1="356" y1="215.0" x2="360" y2="215.0" class="stroke-muted"/><text x="353" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="356" y1="171.9" x2="360" y2="171.9" class="stroke-muted"/><text x="353" y="175.9" class="fig-small" text-anchor="end">50</text>
<line x1="356" y1="128.9" x2="360" y2="128.9" class="stroke-muted"/><text x="353" y="132.9" class="fig-small" text-anchor="end">100</text>
<line x1="356" y1="85.8" x2="360" y2="85.8" class="stroke-muted"/><text x="353" y="89.8" class="fig-small" text-anchor="end">150</text>
<line x1="360" y1="215" x2="580" y2="215" class="stroke-muted"/>
<rect x="360.0" y="214.1" width="4.9" height="0.9" class="fill-a"/>
<rect x="365.5" y="205.5" width="4.9" height="9.5" class="fill-a"/>
<rect x="371.0" y="195.2" width="4.9" height="19.8" class="fill-a"/>
<rect x="376.5" y="172.8" width="4.9" height="42.2" class="fill-a"/>
<rect x="382.0" y="157.3" width="4.9" height="57.7" class="fill-a"/>
<rect x="387.5" y="132.3" width="4.9" height="82.7" class="fill-a"/>
<rect x="393.0" y="110.8" width="4.9" height="104.2" class="fill-a"/>
<rect x="398.5" y="103.1" width="4.9" height="111.9" class="fill-a"/>
<rect x="404.0" y="85.8" width="4.9" height="129.2" class="fill-a"/>
<rect x="409.5" y="72.1" width="4.9" height="142.9" class="fill-a"/>
<rect x="415.0" y="76.4" width="4.9" height="138.6" class="fill-a"/>
<rect x="420.5" y="104.8" width="4.9" height="110.2" class="fill-a"/>
<rect x="426.0" y="115.1" width="4.9" height="99.9" class="fill-a"/>
<rect x="431.5" y="122.0" width="4.9" height="93.0" class="fill-a"/>
<rect x="437.0" y="128.9" width="4.9" height="86.1" class="fill-a"/>
<rect x="442.5" y="153.9" width="4.9" height="61.1" class="fill-a"/>
<rect x="448.0" y="145.2" width="4.9" height="69.8" class="fill-a"/>
<rect x="453.5" y="156.4" width="4.9" height="58.6" class="fill-a"/>
<rect x="459.0" y="169.4" width="4.9" height="45.6" class="fill-a"/>
<rect x="464.5" y="183.1" width="4.9" height="31.9" class="fill-a"/>
<rect x="470.0" y="185.7" width="4.9" height="29.3" class="fill-a"/>
<rect x="475.5" y="184.9" width="4.9" height="30.1" class="fill-a"/>
<rect x="481.0" y="192.6" width="4.9" height="22.4" class="fill-a"/>
<rect x="486.5" y="198.6" width="4.9" height="16.4" class="fill-a"/>
<rect x="492.0" y="203.8" width="4.9" height="11.2" class="fill-a"/>
<rect x="497.5" y="203.8" width="4.9" height="11.2" class="fill-a"/>
<rect x="503.0" y="208.1" width="4.9" height="6.9" class="fill-a"/>
<rect x="508.5" y="209.0" width="4.9" height="6.0" class="fill-a"/>
<rect x="514.0" y="211.6" width="4.9" height="3.4" class="fill-a"/>
<rect x="519.5" y="208.1" width="4.9" height="6.9" class="fill-a"/>
<rect x="525.0" y="215.0" width="4.9" height="0.0" class="fill-a"/>
<rect x="530.5" y="210.7" width="4.9" height="4.3" class="fill-a"/>
<rect x="536.0" y="213.3" width="4.9" height="1.7" class="fill-a"/>
<rect x="541.5" y="214.1" width="4.9" height="0.9" class="fill-a"/>
<rect x="547.0" y="213.3" width="4.9" height="1.7" class="fill-a"/>
<rect x="552.5" y="214.1" width="4.9" height="0.9" class="fill-a"/>
<rect x="558.0" y="213.3" width="4.9" height="1.7" class="fill-a"/>
<rect x="563.5" y="214.1" width="4.9" height="0.9" class="fill-a"/>
<rect x="569.0" y="215.0" width="4.9" height="0.0" class="fill-a"/>
<rect x="574.5" y="214.1" width="4.9" height="0.9" class="fill-a"/>
<line x1="381.4" y1="215" x2="381.4" y2="219" class="stroke-muted"/><text x="381.4" y="233" class="fig-small" text-anchor="middle">100</text>
<line x1="426.8" y1="215" x2="426.8" y2="219" class="stroke-muted"/><text x="426.8" y="233" class="fig-small" text-anchor="middle">200</text>
<line x1="472.3" y1="215" x2="472.3" y2="219" class="stroke-muted"/><text x="472.3" y="233" class="fig-small" text-anchor="middle">300</text>
<line x1="517.7" y1="215" x2="517.7" y2="219" class="stroke-muted"/><text x="517.7" y="233" class="fig-small" text-anchor="middle">400</text>
<line x1="563.2" y1="215" x2="563.2" y2="219" class="stroke-muted"/><text x="563.2" y="233" class="fig-small" text-anchor="middle">500</text>
<text x="170" y="256" class="fig-small" text-anchor="middle">rent (£, mixed periods, typos included)</text>
<text x="470" y="256" class="fig-small" text-anchor="middle">weekly rent (£)</text>
<text x="8" y="22" class="fig-small">number of listings</text>
</svg>
Figure 1. The same rents before and after cleaning. The raw histogram has a hump of weekly rents near £200, a second hump near £900 that is really monthly rents, and a few typo rents out to £2,900. Cleaning leaves one hump.
```

### Three pandas ideas that the first stage needs

You probably know numpy arrays. A pandas **DataFrame** is a table: each column has a name and holds one kind of thing, and each row is one record. A single column is a **Series**, which is a numpy-like array that carries an index (a label for each row) beside the values. You have used an array to hold flat sizes. A DataFrame holds sizes, rents, and neighbourhoods side by side, so one row stays one flat.

Every column has a **dtype**, the kind of value it holds. `int64` and `float64` are numbers. `object` means text (or a mix), and a column that should hold rents but shows `object` is the first sign that something is wrong. A missing value in a float column is `NaN`, which means "not a number". Because `NaN` is a float, a whole column of whole numbers that has one missing value becomes `float64`. You will see that for `bedrooms`.

A **boolean mask** is a Series of `True` and `False` that you use as a filter. Here is a tiny example you can check by hand:

```python
rent = pd.Series([100, 250, 4000, 180])
mask = rent < 1000          # [True, True, False, True]
rent[mask]                  # keeps 100, 250, 180
rent.where(mask, 0)         # keeps where True, uses 0 elsewhere: [100, 250, 0, 180]
```

Masks combine with `&` (and), `|` (or), and `~` (not). Missing values make masks tricky, and you will meet that trap in step 2.

## Plan the build

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="Pipeline: CSV text goes through load, profile, clean, features, summarise, plot, insights, and finally two straight-line fits, one on raw rows and one on clean rows, scored on the same clean test flats.">
<defs><marker id="p004-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="10" y="20" width="130" height="52" rx="6" class="fill-soft-d stroke-d"/>
<text x="75.0" y="42" class="fig-bold" text-anchor="middle">CSV text</text>
<text x="75.0" y="60" class="fig-small" text-anchor="middle">make_data(seed)</text>
<rect x="170" y="20" width="130" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="235.0" y="42" class="fig-bold" text-anchor="middle">1. load, profile</text>
<text x="235.0" y="60" class="fig-small" text-anchor="middle">2,070 rows, 10 cols</text>
<rect x="330" y="20" width="130" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="395.0" y="42" class="fig-bold" text-anchor="middle">2. clean</text>
<text x="395.0" y="60" class="fig-small" text-anchor="middle">1,919 rows left</text>
<rect x="490" y="20" width="120" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="550.0" y="42" class="fig-bold" text-anchor="middle">3. features</text>
<text x="550.0" y="60" class="fig-small" text-anchor="middle">£ per m², month, band</text>
<line x1="140" y1="46" x2="168" y2="46" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<line x1="300" y1="46" x2="328" y2="46" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<line x1="460" y1="46" x2="488" y2="46" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<rect x="10" y="130" width="180" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="100.0" y="152" class="fig-bold" text-anchor="middle">4. summarise</text>
<text x="100.0" y="170" class="fig-small" text-anchor="middle">groupby and pivot tables</text>
<rect x="220" y="130" width="180" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="310.0" y="152" class="fig-bold" text-anchor="middle">5. plot</text>
<text x="310.0" y="170" class="fig-small" text-anchor="middle">4 Figures, none shown yet</text>
<rect x="430" y="130" width="180" height="52" rx="6" class="fill-soft-a stroke-a"/>
<text x="520.0" y="152" class="fig-bold" text-anchor="middle">6. insights</text>
<text x="520.0" y="170" class="fig-small" text-anchor="middle">dict of headline numbers</text>
<path d="M550,72 L550,100 L100,100 L100,128" fill="none" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<line x1="190" y1="156" x2="218" y2="156" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<line x1="400" y1="156" x2="428" y2="156" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<rect x="10" y="245" width="270" height="62" rx="6" class="fill-soft-c stroke-c"/>
<text x="145.0" y="267" class="fig-bold" text-anchor="middle">7a. line on raw rows</text>
<text x="145.0" y="285" class="fig-small" text-anchor="middle">face-value numbers, duplicates, typos</text>
<rect x="340" y="245" width="270" height="62" rx="6" class="fill-soft-b stroke-b"/>
<text x="475.0" y="267" class="fig-bold" text-anchor="middle">7b. line on clean rows</text>
<text x="475.0" y="285" class="fig-small" text-anchor="middle">weekly £, m², no duplicates</text>
<path d="M520,182 L520,214 L475,214 L475,243" fill="none" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<path d="M520,214 L145,214 L145,243" fill="none" class="stroke-muted" marker-end="url(#p004-arrow)"/>
<text x="310" y="324" class="fig-small" text-anchor="middle">same clean test flats score both lines</text>
</svg>
Figure 4. The pipeline. Each stage is one function (or a few), and the final comparison scores both straight lines on the same clean test flats.
```

Figure 4 shows the pipeline. The stages come in this order because each one needs the previous one to be trustworthy. You cannot clean a column before you know what is wrong with it, so you load and profile first. You cannot summarise before you clean, because a mean over mixed pounds-per-week and pounds-per-month is meaningless. You cannot compute rent per square metre before the units are right. And you compare the two straight lines last, because the comparison is the proof that the earlier stages mattered.

Cleaning has a second ordering rule. Drop the duplicates first, so that no flat counts twice when you compute quartiles. Drop the impossible values before the outlier fence, so that a size of 0 does not distort the fence. Then fence the rest.

## Build it step by step

> [!RECAP]
> - A DataFrame is a table, a column is a Series, and a dtype says what a column holds.
> - A boolean mask filters rows, and a comparison with a missing value gives `<NA>`, so fill it before you combine masks.
> - Clean units first: weekly pounds and square metres, or no mean means anything.
> - Remove outliers with a stated rule, such as a domain bound and a fence on the quartiles ([Bias, variance, and noise](/lessons/004-bias-variance-noise) says why noise sets a floor).
> - Judge any fit on rows held back from fitting ([Honest scores](/lessons/002-honest-scores)).

`make_data`, `fit_polynomial`, `predict_polynomial`, `rmse`, and the constants (`NEIGHBOURHOODS`, `SQFT_PER_M2`, the area bounds, and the band edges and labels) are given. You write the rest, in this order.

1. `load_listings(csv_text)` wraps the text in `io.StringIO` and calls `pd.read_csv`. It returns the raw frame untouched. `profile(df)` returns a dict with the row count, column count, a dict of dtypes, a dict of missing counts, and the number of duplicate rows. Use `df.shape`, `df.dtypes`, `df.isna().sum()`, and `df.duplicated().sum()`. The common mistake is to count both copies of a repeated row as duplicates. `duplicated()` marks only the later copies, which is what `drop_duplicates()` removes.

2. `parse_number(text)` pulls the first number out of text such as `£1,250 pcm`. A string method on a Series lives under `.str`, so `text.astype("string").str.extract(regex)` returns the first match of a pattern. The common mistake is to delete every non-digit character. That turns `58 m2` into 582, because the 2 in the unit looks like a digit. Take only the leading number.
   `to_weekly_rent(rent, period)` builds a mask `monthly` from the period column (`month`, any case) and, where the period is blank, from a `pcm` suffix, then multiplies those rents by 12 and divides by 52. A year has 12 months and 52 weeks, so £1,300 a month is $1300 \times 12 / 52 = \text{£}300$ a week. The common mistake is the missing-value trap. In a pandas string Series, `period.str.lower() == "month"` is `<NA>` where the period is blank, and a mask that holds `<NA>` can flip rows you did not mean to flip. Fill missing words with an empty string first.

3. `to_square_metres(size, unit)` reads the unit from `size_unit` and, where that is blank, from letters in `size`, then divides the square-foot rows by 10.7639. A square metre is 10.7639 square feet, so 582 sqft is about 54.1 m². Use `unit.fillna(size)` to fall back from one column to the other, and `Series.where` to touch only the square-foot rows. The common mistake is to convert every row because the first few you looked at were in feet.

4. `normalise_neighbourhood(names)` strips the ends, squeezes repeated spaces, lowers the case, and matches each name to the closest entry of `NEIGHBOURHOODS` with `difflib.get_close_matches(word, options, n=1, cutoff=0.8)`. A cutoff of 0.8 accepts `Rivrside` and rejects `Atlantis`. Match each distinct spelling once and use `Series.map`. `parse_yes_no(text)` returns a Series of dtype `"boolean"`, which can hold `<NA>` beside `True` and `False`. `parse_dates(text)` tries the three formats in turn with `pd.to_datetime(text, format=..., errors="coerce")` and fills the gaps from the next format with `fillna`. The common mistake is to let pandas guess the format: `04/03/2023` is the 4th of March here, and a guess can read it as April.

5. `clean_listings(df)` drops exact duplicates, builds a fresh frame from your parse functions, and drops rows by four rules in the order its docstring gives. It also resets the index. Drop a row when rent, area, or neighbourhood is missing, because rent per square metre cannot be computed without them. Drop an area outside 10 to 300 m², which is a **domain bound**: no flat in this market is under 10 m² or over 300 m². Then fence the typo rents. Compute rent per square metre, then its first and third quartiles, $Q_1$ and $Q_3$, and the interquartile range $\text{IQR} = Q_3 - Q_1$. Drop rows outside $Q_1 - 3\,\text{IQR}$ to $Q_3 + 3\,\text{IQR}$. For seed 0 the quartiles are 3.01 and 4.26, so the fence runs from -0.73 to 8.00. A typo rent of £2,610 on 70 m² is £37 per m², far outside, while the dearest honest flat is 7.15. The common mistake is to use 1.5 IQR, the textbook fence for a bell curve. These rents are skewed to the right, so 1.5 IQR throws away about ten honest flats on seed 0.

6. `add_features(df)` copies the frame and adds `rent_per_m2` (rent divided by area), `month` (from `.dt.month`, as `Int64` so a missing date stays `<NA>`), and `size_band` from `pd.cut(area, bins=SIZE_EDGES, labels=SIZE_LABELS, right=False)`. `pd.cut` sorts numbers into labelled bins, and `right=False` puts a flat of exactly 40 m² in the band that starts at 40. The common mistake is to change the caller's frame. Work on `df.copy()`.

7. The summaries all use **groupby**: split the rows into groups, apply a function to each, and combine the answers. `df.groupby("neighbourhood")["rent_per_m2"].mean()` makes one mean per neighbourhood. `rent_per_m2_by_neighbourhood(df)` returns that, sorted from highest to lowest. `rent_table(df)` uses `pivot_table(index="neighbourhood", columns="size_band", values="rent_week", aggfunc="median", observed=False)` to lay the medians out as a grid with one row per neighbourhood and one column per band. `furnished_table(df, min_each=5)` groups by neighbourhood, band, and furnished, unstacks the furnished level into columns, and keeps only cells with at least five flats of each kind. `monthly_profile(df)` gives the median rent per m² by month, and `distance_profile(df)` cuts `dist_station_km` at `DIST_EDGES` and takes the mean per band. The common mistake with `furnished_table` is to compare the two groups without the neighbourhood and band, which gives the misleading overall answer.

8. The four plotting functions each return a `Figure` and never call `plt.show()`, because a function that shows its own figure cannot be tested or reused. In matplotlib, a **Figure** is the whole canvas and an **Axes** is one set of axes drawn on it. `fig, ax = plt.subplots()` makes one of each, and `plt.subplots(1, 2)` makes two side by side. `plot_rent_hist(before, after, bins)` draws a histogram of counts on each of two axes. `plot_rent_vs_size(df, by)` draws one `ax.scatter` call per value of `by`, so each group gets its own colour and legend entry. `plot_neighbourhood_bars(df)` draws a bar per neighbourhood with `ax.bar`. `plot_furnished_paradox(df)` draws the overall means on the left axis and grouped bars from `furnished_table` on the right. Label every axis: a title, an x label, and a y label. The common mistake is to pass missing values to `ax.hist` and have the counts disagree with the row count. Drop them first.

9. `key_insights(df)` reuses the summaries to return the top and bottom neighbourhood, the ratio between them, the overall and within-cell furnished gaps, the peak month, and two distance drops. The within-cell gap is the average of furnished minus unfurnished over the cells of `furnished_table`, weighted by cell size with `np.average(..., weights=...)`. The common mistake is to report only the overall gap, which has the wrong sign.

10. `raw_rows(df)` keeps the raw frame's `listing_id` and two columns read at face value: `parse_number` on `size`, and `parse_number` on `rent`, with no conversion and no duplicate removal. `compare_raw_vs_clean(csv_text, seed)` loads and cleans the file and holds back a quarter of the cleaned flats, chosen by a seeded shuffle, as the test set. It fits `fit_polynomial(x, y, 1)` on the other cleaned flats, and again on every raw row whose `listing_id` is not a test flat. It scores both lines on the same test flats with `rmse`. The common mistake is to let a test flat's duplicate or raw copy into the raw training set. Remove test ids from the raw rows by `listing_id`.

## Read your results

Run `starter.py` once the functions work. The main block prints tables and shows four figures in the console. For seed 0 you should see numbers close to these.

| Quantity | Seed 0 | What it tells the agency |
|---|---|---|
| Raw rows, duplicates | 2,070 and 70 | The export repeats 70 rows. |
| Distinct neighbourhood spellings | 93 | Eight places, written 93 ways. |
| Rows after each rule | 2,000, 1,940, 1,927, 1,919 | Duplicates, then missing rent or size (60), the area bound (13), the fence (8). |
| Weekly rent after cleaning | 53 to 537, mean 197.5 | One scale, with no £2,900 typo left. |
| Mean rent per m², Docklands and Northway | £4.91 and £2.53 | A flat in Docklands costs 1.94 times as much per m². |
| Furnished gap, overall | -£44.8 | Furnished flats look £45 cheaper a week. |
| Furnished gap, same neighbourhood and band | +£17.5 | Like for like, they cost £17.50 more. |
| Peak month, median rent per m² | September, £4.43 | Other months sit between £3.30 and £3.82. |
| Distance drops | 14.4 percent near, 2.2 percent far | The first kilometre or two matters. The tail does not. |
| Test RMSE, line on raw rows | £196.6 a week | The mixed units and typos wreck the fit. |
| Test RMSE, line on clean rows | £47.4 a week | About a quarter of the raw error. |

Read the first rows in order, as the agency would. The profile shows what you must fix, and the row counts after each rule show that no step is removing more than it should. Duplicates come out first, so the missing-value count (60) is of distinct flats.

The paradox is the part to remember. Furnished flats make up 10 percent of Docklands and 77 percent of Northway, and 59 percent of flats under 40 m² against 22 percent of flats of 70 m² or more. So the furnished group is full of cheap, small flats, which drags its average down. Inside one neighbourhood and size band, that composition effect is gone and the furnished premium shows. This pattern, where a comparison in the totals reverses when you split by a third variable, is called **Simpson's paradox**. Here the third variable (a confounder) is the mix of neighbourhood and size.

```figure
<svg viewBox="0 0 620 290" role="img" aria-label="Left panel: furnished flats average 173 pounds a week against 218 for unfurnished flats. Right panel: inside the same neighbourhood and size band, furnished flats cost more in all eight neighbourhoods, by 8 to 35 pounds.">
<text x="75" y="38" class="fig-bold" text-anchor="middle">Overall</text>
<line x1="40" y1="232" x2="40" y2="62" class="stroke-muted"/>
<line x1="36" y1="232.0" x2="40" y2="232.0" class="stroke-muted"/><text x="33" y="236.0" class="fig-small" text-anchor="end">0</text>
<line x1="36" y1="175.3" x2="40" y2="175.3" class="stroke-muted"/><text x="33" y="179.3" class="fig-small" text-anchor="end">100</text>
<line x1="36" y1="118.7" x2="40" y2="118.7" class="stroke-muted"/><text x="33" y="122.7" class="fig-small" text-anchor="end">200</text>
<line x1="36" y1="62.0" x2="40" y2="62.0" class="stroke-muted"/><text x="33" y="66.0" class="fig-small" text-anchor="end">300</text>
<line x1="40" y1="232" x2="120" y2="232" class="stroke-muted"/>
<rect x="46" y="108.4" width="32" height="123.6" class="fill-a"/>
<rect x="80" y="133.8" width="32" height="98.2" class="fill-b"/>
<text x="62" y="103" class="fig-small" text-anchor="middle">218</text>
<text x="96" y="129" class="fig-small" text-anchor="middle">173</text>
<text x="62" y="250" class="fig-small" text-anchor="middle">unfurn.</text><text x="98" y="250" class="fig-small" text-anchor="middle">furn.</text>
<text x="395" y="38" class="fig-bold" text-anchor="middle">Same neighbourhood and size band</text>
<line x1="180" y1="232" x2="180" y2="62" class="stroke-muted"/>
<line x1="176" y1="232.0" x2="180" y2="232.0" class="stroke-muted"/><text x="173" y="236.0" class="fig-small" text-anchor="end">0</text>
<line x1="176" y1="175.3" x2="180" y2="175.3" class="stroke-muted"/><text x="173" y="179.3" class="fig-small" text-anchor="end">100</text>
<line x1="176" y1="118.7" x2="180" y2="118.7" class="stroke-muted"/><text x="173" y="122.7" class="fig-small" text-anchor="end">200</text>
<line x1="176" y1="62.0" x2="180" y2="62.0" class="stroke-muted"/><text x="173" y="66.0" class="fig-small" text-anchor="end">300</text>
<line x1="180" y1="232" x2="600" y2="232" class="stroke-muted"/>
<rect x="188" y="98.7" width="20" height="133.3" class="fill-a"/>
<rect x="209" y="85.6" width="20" height="146.4" class="fill-b"/>
<text x="208" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 208 247)">Docklands</text>
<rect x="240" y="148.5" width="20" height="83.5" class="fill-a"/>
<rect x="261" y="142.9" width="20" height="89.1" class="fill-b"/>
<text x="260" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 260 247)">Eastgate</text>
<rect x="292" y="135.2" width="20" height="96.8" class="fill-a"/>
<rect x="313" y="126.1" width="20" height="105.9" class="fill-b"/>
<text x="312" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 312 247)">Hillcrest</text>
<rect x="344" y="127.9" width="20" height="104.1" class="fill-a"/>
<rect x="365" y="118.1" width="20" height="113.9" class="fill-b"/>
<text x="364" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 364 247)">Kingsfield</text>
<rect x="396" y="114.4" width="20" height="117.6" class="fill-a"/>
<rect x="417" y="101.9" width="20" height="130.1" class="fill-b"/>
<text x="416" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 416 247)">Marlow Park</text>
<rect x="448" y="155.3" width="20" height="76.7" class="fill-a"/>
<rect x="469" y="151.0" width="20" height="81.0" class="fill-b"/>
<text x="468" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 468 247)">Northway</text>
<rect x="500" y="123.0" width="20" height="109.0" class="fill-a"/>
<rect x="521" y="115.6" width="20" height="116.4" class="fill-b"/>
<text x="520" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 520 247)">Old Town</text>
<rect x="552" y="96.7" width="20" height="135.3" class="fill-a"/>
<rect x="573" y="77.0" width="20" height="155.0" class="fill-b"/>
<text x="572" y="247" class="fig-small" text-anchor="middle" transform="rotate(35 572 247)">Riverside</text>
<rect x="190" y="52" width="10" height="10" class="fill-a"/><text x="204" y="61" class="fig-small">unfurnished</text><rect x="290" y="52" width="10" height="10" class="fill-b"/><text x="304" y="61" class="fig-small">furnished</text>
<text x="8" y="22" class="fig-small">mean weekly rent (£)</text>
</svg>
Figure 2. The furnished paradox from seed 0. Overall (left), furnished flats average £173 against £218 for unfurnished flats. Inside one neighbourhood and size band (right), furnished flats cost more in every neighbourhood.
```

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Left: median rent per square metre by month listed is flat near 3.4 to 3.8 pounds, with a spike to 4.4 in September. Right: mean rent per square metre by distance to the station falls from 4.2 within half a kilometre to 3.6 at one to two kilometres, then flattens near 3.5.">
<text x="165" y="38" class="fig-bold" text-anchor="middle">Median rent per m² by month</text>
<line x1="60" y1="205" x2="270" y2="205" class="stroke-muted"/><line x1="60" y1="205" x2="60" y2="60" class="stroke-muted"/>
<line x1="60" y1="205.0" x2="270" y2="205.0" class="stroke-grid"/><text x="54" y="209.0" class="fig-small" text-anchor="end">3.0</text>
<line x1="60" y1="159.7" x2="270" y2="159.7" class="stroke-grid"/><text x="54" y="163.7" class="fig-small" text-anchor="end">3.5</text>
<line x1="60" y1="114.4" x2="270" y2="114.4" class="stroke-grid"/><text x="54" y="118.4" class="fig-small" text-anchor="end">4.0</text>
<line x1="60" y1="69.1" x2="270" y2="69.1" class="stroke-grid"/><text x="54" y="73.1" class="fig-small" text-anchor="end">4.5</text>
<polyline points="72.0,166.1 88.9,177.6 105.8,150.1 122.7,156.2 139.6,166.3 156.5,139.1 173.5,130.4 190.4,147.8 207.3,75.4 224.2,155.7 241.1,160.2 258.0,167.9" fill="none" class="stroke-a"/>
<circle cx="72.0" cy="166.1" r="3" class="fill-a"/>
<text x="72.0" y="222" class="fig-small" text-anchor="middle">1</text>
<circle cx="88.9" cy="177.6" r="3" class="fill-a"/>
<circle cx="105.8" cy="150.1" r="3" class="fill-a"/>
<text x="105.8" y="222" class="fig-small" text-anchor="middle">3</text>
<circle cx="122.7" cy="156.2" r="3" class="fill-a"/>
<circle cx="139.6" cy="166.3" r="3" class="fill-a"/>
<circle cx="156.5" cy="139.1" r="3" class="fill-a"/>
<text x="156.5" y="222" class="fig-small" text-anchor="middle">6</text>
<circle cx="173.5" cy="130.4" r="3" class="fill-a"/>
<circle cx="190.4" cy="147.8" r="3" class="fill-a"/>
<circle cx="207.3" cy="75.4" r="4.2" class="fill-b"/>
<text x="207.3" y="222" class="fig-small" text-anchor="middle">9</text>
<circle cx="224.2" cy="155.7" r="3" class="fill-a"/>
<circle cx="241.1" cy="160.2" r="3" class="fill-a"/>
<circle cx="258.0" cy="167.9" r="3" class="fill-a"/>
<text x="258.0" y="222" class="fig-small" text-anchor="middle">12</text>
<text x="201.3" y="67.4" class="fig-small" text-anchor="end">Sept 4.43</text>
<text x="165" y="238" class="fig-small" text-anchor="middle">month listed</text>
<text x="465" y="38" class="fig-bold" text-anchor="middle">Mean rent per m² by distance</text>
<line x1="360" y1="205" x2="570" y2="205" class="stroke-muted"/><line x1="360" y1="205" x2="360" y2="60" class="stroke-muted"/>
<line x1="360" y1="205.0" x2="570" y2="205.0" class="stroke-grid"/><text x="354" y="209.0" class="fig-small" text-anchor="end">3.0</text>
<line x1="360" y1="159.7" x2="570" y2="159.7" class="stroke-grid"/><text x="354" y="163.7" class="fig-small" text-anchor="end">3.5</text>
<line x1="360" y1="114.4" x2="570" y2="114.4" class="stroke-grid"/><text x="354" y="118.4" class="fig-small" text-anchor="end">4.0</text>
<line x1="360" y1="69.1" x2="570" y2="69.1" class="stroke-grid"/><text x="354" y="73.1" class="fig-small" text-anchor="end">4.5</text>
<polyline points="382.0,94.1 423.5,124.3 465.0,149.1 506.5,155.6 548.0,162.6" fill="none" class="stroke-b"/>
<circle cx="382.0" cy="94.1" r="3.5" class="fill-b"/><text x="382.0" y="222" class="fig-small" text-anchor="middle">0-0.5</text><text x="382.0" y="85.1" class="fig-small" text-anchor="middle">4.22</text>
<circle cx="423.5" cy="124.3" r="3.5" class="fill-b"/><text x="423.5" y="222" class="fig-small" text-anchor="middle">0.5-1</text><text x="423.5" y="115.3" class="fig-small" text-anchor="middle">3.89</text>
<circle cx="465.0" cy="149.1" r="3.5" class="fill-b"/><text x="465.0" y="222" class="fig-small" text-anchor="middle">1-2</text><text x="465.0" y="140.1" class="fig-small" text-anchor="middle">3.62</text>
<circle cx="506.5" cy="155.6" r="3.5" class="fill-b"/><text x="506.5" y="222" class="fig-small" text-anchor="middle">2-3</text><text x="506.5" y="146.6" class="fig-small" text-anchor="middle">3.55</text>
<circle cx="548.0" cy="162.6" r="3.5" class="fill-b"/><text x="548.0" y="222" class="fig-small" text-anchor="middle">3+</text><text x="548.0" y="153.6" class="fig-small" text-anchor="middle">3.47</text>
<text x="465" y="238" class="fig-small" text-anchor="middle">distance to station (km)</text>
<text x="8" y="22" class="fig-small">£ per m² per week</text>
</svg>
Figure 3. Two more planted patterns. Median rent per m² is flat for most months and jumps in September (left). Mean rent per m² falls steeply within the first kilometre or two of the station and then levels off (right).
```

The two figures above are the SVG versions of what your matplotlib plots show. In `plot_rent_hist`, the left panel has a tall hump near £200, a second low hump near £900 (monthly rents), and a tail out to £2,900 (typos). The right panel has one bell with a right tail. In `plot_rent_vs_size`, rent rises with size, and the furnished dots sit at the lower left. In `plot_neighbourhood_bars`, the bars step down from Docklands to Northway. In `plot_furnished_paradox`, the left bars say furnished is cheaper and every pair on the right says the opposite.

The two lines compare like this across seeds 0 to 4.

| Seed | RMSE, raw rows | RMSE, clean rows |
|---|---|---|
| 0 | 196.6 | 47.4 |
| 1 | 187.4 | 50.5 |
| 2 | 202.3 | 46.5 |
| 3 | 213.5 | 45.4 |
| 4 | 194.4 | 49.5 |

Chapter 4 says that error has a floor, and a floor sits under this score too. The straight line sees only size, so it cannot explain the neighbourhood, the distance, the furnishing, or the season. That is why the clean error is £47 and not near zero. The line is honest about what it knows, and every column it ignores is a column a better model can use.

What a wrong number usually means:

- More than 1,919 rows means a typo outlier or a duplicate survived. Fewer than about 1,500 means square-foot sizes were not converted, so large areas fell outside the 10 to 300 m² bound.
- A weekly rent mean far above 200, or a bump near £900 in the histogram, means monthly rents were not converted. A bump at £2,000 or more means the fence is missing.
- A negative within-cell furnished gap means you compared furnished and unfurnished flats without grouping by neighbourhood and band.
- A peak month other than 9 means the dates were parsed month first, or the month was taken from the wrong column.
- A clean test RMSE far above £60 means a unit error or an outlier survived cleaning, because the clean line should sit near £47.

## Take it further

- Add `bedrooms` to the straight line by fitting least squares on two columns with numpy. You will see how much of the error that remains is explained by another honest column.
- Replace the area bound and the fence with a fence computed within each neighbourhood. You will see that an outlier rule is a modelling choice, and when a local rule is stricter or looser than a global one.
- Fit a degree-3 polynomial of rent per m² on distance to the station with `fit_polynomial`, then compare its test error with a straight line. You will see the nonlinear distance effect as a better score, not only as a plot.
- Add `neighbourhood` as dummy columns (one 0-or-1 column per name) and refit. You will see how much of the £47 floor was the missing neighbourhood.
- Compute the seasonal index as rent per m² divided by the neighbourhood median, then plot it by month. You will learn how to remove one effect so a second one shows more clearly.

## The exercise

You write `load_listings`, `profile`, `parse_number`, `to_weekly_rent`, `to_square_metres`, `normalise_neighbourhood`, `parse_dates`, `parse_yes_no`, `clean_listings`, `add_features`, `rent_per_m2_by_neighbourhood`, `rent_table`, `furnished_table`, `monthly_profile`, `distance_profile`, `plot_rent_hist`, `plot_rent_vs_size`, `plot_neighbourhood_bars`, `plot_furnished_paradox`, `key_insights`, `raw_rows`, and `compare_raw_vs_clean`.

The first checks test each stage on tiny hand-made frames. They check that `profile` counts shape, missing values, and duplicates, that `parse_number` and `to_weekly_rent` handle commas, suffixes, and blank periods, that `to_square_metres` reads units from either column, and that the neighbourhood, yes/no, and date parsers return the planted set. Two checks run `clean_listings` on seed 0. They require the planted number of rows with no duplicates and no impossible values, and weekly rents, areas, flags, and months that match the planted truth. The next checks test `add_features` and the summaries against values worked out by hand, including a small frame that contains the furnished paradox. Two checks call each plotting function on a small frame. They require a `Figure` with the right number of axes, labelled axes, histogram counts that sum to the row count, bar heights equal to the computed means, and every scatter point present. A check on seeds 0 to 2 requires the insights to recover the top and bottom neighbourhoods, the sign of the furnished gap overall and within cells, September, and the steep-then-flat distance pattern. The last check runs `compare_raw_vs_clean` on seeds 0 to 4 and requires the clean test RMSE to be under half the raw one.
