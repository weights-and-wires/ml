"""Project: explore before you model.

A letting agency hands you a messy export of about two thousand rental
listings. Before anyone builds a model, you load it, profile it, clean it,
summarise it, plot it, and check that cleaning makes a straight-line model
honest. You write the stages in this order:

1. `load_listings` and `profile`: read the CSV and describe it.
2. `parse_number`, `to_weekly_rent`, `to_square_metres`,
   `normalise_neighbourhood`, `parse_dates`, `parse_yes_no`: clean one kind of
   column at a time.
3. `clean_listings`: wire the steps together and drop rows you cannot trust.
4. `add_features`: rent per m2, month, and size band.
5. `rent_per_m2_by_neighbourhood`, `rent_table`, `furnished_table`,
   `monthly_profile`, `distance_profile`: groupby and pivot summaries.
6. `plot_rent_hist`, `plot_rent_vs_size`, `plot_neighbourhood_bars`,
   `plot_furnished_paradox`: each returns a matplotlib Figure.
7. `key_insights`: the headline numbers as a dict.
8. `raw_rows` and `compare_raw_vs_clean`: fit a line on raw rows and on clean
   rows, and score both on the same clean test flats.

`make_data`, `fit_polynomial`, `predict_polynomial`, `rmse`, and the constants
are provided. Rents are in pounds per week and areas in square metres.
"""

from __future__ import annotations

import csv
import difflib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

NEIGHBOURHOODS = [
    "Docklands", "Riverside", "Old Town", "Marlow Park",
    "Kingsfield", "Hillcrest", "Eastgate", "Northway",
]
SQFT_PER_M2 = 10.7639
MIN_AREA_M2, MAX_AREA_M2 = 10.0, 300.0
SIZE_EDGES = [0.0, 40.0, 55.0, 70.0, np.inf]
SIZE_LABELS = ["<40", "40-55", "55-70", "70+"]
DIST_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, np.inf]
DIST_LABELS = ["0-0.5", "0.5-1", "1-2", "2-3", "3+"]


def make_data(seed: int = 0, with_truth: bool = False):
    """Build the messy export of a letting agency (provided, seeded, instant).

    Args:
        seed: Seed for every random choice.
        with_truth: When True, return (csv_text, truth) so you can check your
            cleaning against the planted facts. The default returns only the CSV text.

    Returns:
        The raw CSV as one string: 2,070 lines under a 10-column header
        (listing_id, listed_on, neighbourhood, size, size_unit, bedrooms,
        dist_station_km, furnished, rent, rent_period).

    Planted mess (counts are of the 2,000 distinct listings):
        - Rent is text such as "£1,250". About a quarter are per month, flagged
          by the rent_period column ("week" or "month", any case) or, when that
          is blank, by a " pcm" or " pw" suffix.
        - Size is text such as "54.1", "54.1 m2", or "582 sqft"; about 22% are
          square feet, named by size_unit or by the suffix.
        - 6 sizes are 0, 4 are 1 to 5, and 3 are in the thousands (impossible).
        - 8 rents have an extra zero (typo outliers). 20 rents and 40 sizes are
          missing. 90 distances, 60 furnished flags, 40 dates, and 100 bedroom
          counts are missing too.
        - Neighbourhood names vary in case and stray spaces, and a few are
          misspelt. Dates use "2023-03-04", "04/03/2023", or "04 Mar 2023"
          (day first). Furnished is Y, N, yes, no, True, or False in mixed case.
        - 70 listings appear twice, as exact duplicate lines.

    Planted insights, true before the mess is added:
        - Rent per m2 differs by neighbourhood, from Docklands (highest) to
          Northway (lowest).
        - Furnished flats cost 12% more than unfurnished flats of the same
          neighbourhood and size, but furnished flats are common in the cheap
          neighbourhoods and among small flats, so they look cheaper overall.
        - Rent peaks in September.
        - Rent falls steeply with the first kilometre from the station and
          is nearly flat beyond 2 km.

    truth has keys n_raw_rows, n_unique, n_duplicates, n_clean (rows a correct
    cleaning keeps), dropped_ids (ids a correct cleaning removes), rent_week,
    area_m2, furnished, and month (each maps id -> true value for the kept ids),
    neighbourhoods, and the counts of each planted fault.
    """
    rng = np.random.default_rng(seed)
    n = 2000

    # --- the true world -------------------------------------------------
    rate = np.array([5.2, 4.6, 4.3, 3.9, 3.5, 3.2, 2.8, 2.5])  # pounds per m2 per week
    share = np.array([0.10, 0.15, 0.12, 0.14, 0.15, 0.12, 0.12, 0.10])
    far = np.full(8, 1.0)  # distance scale (km), the same everywhere
    furn_logit = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.2, 1.5])
    season = np.array([0.97, 0.96, 0.97, 0.98, 0.99, 1.01, 1.04, 1.07, 1.18, 1.03, 0.98, 0.96])

    hood = rng.choice(8, size=n, p=share)
    area = np.clip(rng.lognormal(np.log(52.0), 0.30, size=n), 20.0, 150.0)
    dist = np.clip(rng.gamma(2.0, far[hood]), 0.05, 9.0)
    p_furn = 1 / (1 + np.exp(-(furn_logit[hood] - 0.045 * (area - 55.0) + 0.15 * (dist - 2.0))))
    furnished = rng.uniform(size=n) < p_furn
    day = rng.integers(0, 365, size=n)
    dates = np.datetime64("2023-01-01") + day.astype("timedelta64[D]")
    month = dates.astype("datetime64[M]").astype(int) % 12 + 1
    station = 0.86 + 0.34 * np.exp(-dist / 0.7)
    rent_week = (
        rate[hood] * 50.0 * (area / 50.0) ** 0.9 * station
        * np.where(furnished, 1.12, 1.0) * season[month - 1]
        * np.exp(rng.normal(0.0, 0.09, size=n))
    )
    bedrooms = np.clip(np.round(area / 24.0 + rng.normal(0, 0.4, size=n)), 1, 5)

    # --- disjoint index sets for the planted mess -----------------------
    order = rng.permutation(n)
    zero_size, tiny_size, huge_size = order[0:6], order[6:10], order[10:13]
    typo = order[13:21]
    miss_rent, miss_size = order[21:41], order[41:81]
    bad_size = np.concatenate([zero_size, tiny_size, huge_size])
    dropped = np.concatenate([bad_size, typo, miss_rent, miss_size])

    # --- area text ------------------------------------------------------
    area_val = np.round(area, 1)
    area_val[zero_size] = 0.0
    area_val[tiny_size] = rng.integers(1, 6, size=4)
    area_val[huge_size] = rng.choice([5400, 6200, 8800], size=3)
    sqft = rng.uniform(size=n) < 0.22
    size_txt, unit_txt = [], []
    miss_size_set = set(miss_size.tolist())
    for i in range(n):
        v = area_val[i]
        if sqft[i]:
            num = f"{int(round(v * 10.7639))}"
            unit = rng.choice(["sqft", "sq ft", "SQFT"])
        else:
            num = f"{v:g}"
            unit = rng.choice(["m2", "sqm", "M2"])
        mode = rng.uniform()
        if mode < 0.72:
            size_txt.append(num); unit_txt.append(unit)
        else:
            size_txt.append(f"{num} {unit}"); unit_txt.append("")
        if i in miss_size_set:
            size_txt[-1] = ""; unit_txt[-1] = ""
    # --- rent text ------------------------------------------------------
    rent_txt, period_txt = [], []
    typo_set, miss_rent_set = set(typo.tolist()), set(miss_rent.tolist())
    for i in range(n):
        monthly = rng.uniform() < 0.26
        amount = rent_week[i] * 52 / 12 if monthly else rent_week[i]
        amount = int(round(amount))
        if i in typo_set:
            amount *= 10
        text = f"£{amount:,}"
        r = rng.uniform()
        if r < 0.80:
            period = rng.choice(["week", "Week"]) if not monthly else rng.choice(["month", "MONTH"])
        else:
            period = ""
            text += " pcm" if monthly else " pw"
        if i in miss_rent_set:
            text, period = "", ""
        rent_txt.append(text); period_txt.append(period)

    # --- the other columns ----------------------------------------------
    hood_txt = []
    typos_map = {
        "Docklands": ["Dockland", "Docklnds"], "Riverside": ["Rivrside", "Riverside "],
        "Old Town": ["Oldtown"], "Marlow Park": ["Marlow Pk", "Marlo Park"],
        "Kingsfield": ["Kingsfeild"], "Hillcrest": ["Hilcrest"],
        "Eastgate": ["Eastgat"], "Northway": ["Northwy"],
    }
    for h in hood:
        name = NEIGHBOURHOODS[h]
        r = rng.uniform()
        if r < 0.03:
            name = rng.choice(typos_map[name])
        elif r < 0.25:
            name = name.lower()
        elif r < 0.40:
            name = name.upper()
        r = rng.uniform()
        if r < 0.10:
            name = "  " + name
        elif r < 0.20:
            name = name + "   "
        hood_txt.append(name)
    yes = ["Y", "yes", "True", "y", "YES"]
    no = ["N", "no", "False", "n", "No"]
    furn_txt = [str(rng.choice(yes if f else no)) for f in furnished]
    for i in rng.choice(n, size=60, replace=False):
        furn_txt[i] = ""
    fmt = rng.uniform(size=n)
    date_txt = []
    for d, f in zip(dates.astype("datetime64[D]").astype(object), fmt):
        date_txt.append(d.strftime("%Y-%m-%d") if f < 0.40 else d.strftime("%d/%m/%Y") if f < 0.75 else d.strftime("%d %b %Y"))
    for i in rng.choice(n, size=40, replace=False):
        date_txt[i] = ""
    dist_txt = [f"{d:.2f}" for d in dist]
    for i in rng.choice(n, size=90, replace=False):
        dist_txt[i] = ""
    bed_txt = [f"{int(b)}" for b in bedrooms]
    for i in rng.choice(n, size=100, replace=False):
        bed_txt[i] = ""

    rows = [
        [1000 + i, date_txt[i], hood_txt[i], size_txt[i], unit_txt[i], bed_txt[i],
         dist_txt[i], furn_txt[i], rent_txt[i], period_txt[i]]
        for i in range(n)
    ]
    dup = rng.choice(n, size=70, replace=False)
    rows += [list(rows[i]) for i in dup]
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["listing_id", "listed_on", "neighbourhood", "size", "size_unit",
                "bedrooms", "dist_station_km", "furnished", "rent", "rent_period"])
    w.writerows(rows)
    text = buf.getvalue()
    if not with_truth:
        return text
    keep = np.setdiff1d(np.arange(n), dropped)
    truth = {
        "n_raw_rows": len(rows), "n_unique": n, "n_duplicates": 70,
        "n_clean": int(len(keep)),
        "dropped_ids": sorted(int(1000 + i) for i in dropped),
        "rent_week": {int(1000 + i): float(rent_week[i]) for i in keep},
        "area_m2": {int(1000 + i): float(area[i]) for i in keep},
        "neighbourhoods": list(NEIGHBOURHOODS),
        "furnished": {int(1000 + i): bool(furnished[i]) for i in keep},
        "month": {int(1000 + i): int(month[i]) for i in keep},
        "n_sqft": int(sqft.sum()),
        "n_zero_size": 6, "n_tiny_size": 4, "n_huge_size": 3,
        "n_rent_typos": 8, "n_missing_rent": 20, "n_missing_size": 40,
        "n_missing_dist": 90, "n_missing_furnished": 60, "n_missing_date": 40, "n_missing_bedrooms": 100,
        "furnished_premium": 0.12, "peak_month": 9,
    }
    return text, truth


# Written in the chapter "Fitting a curve: polynomials, least squares, and the degree dial".
def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(np.asarray(x, dtype=float)[:, None] ** np.arange(degree + 1)[None, :], y, rcond=None)[0]


def predict_polynomial(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[:, None] ** np.arange(len(w))[None, :] @ w


# Written in the chapter "Scoring a guess: loss and empirical risk".
def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2)))


# ---------------------------------------------------------------------------
# Stage 1: load and profile
# ---------------------------------------------------------------------------
def load_listings(csv_text: str) -> pd.DataFrame:
    """Read the raw export into a DataFrame, changing nothing.

    Args:
        csv_text: The CSV text returned by `make_data`.

    Returns:
        A DataFrame with one row per line of the CSV and the 10 columns of the
        header, in file order, with the dtypes that `pd.read_csv` infers.

    Steps:
    1. Wrap the text in io.StringIO so read_csv can read it like a file.
    2. Call pd.read_csv on that and return the frame without changing it.
    """
    return pd.read_csv(io.StringIO(csv_text))


def profile(df: pd.DataFrame) -> dict:
    """Describe a DataFrame before touching it.

    Returns:
        A dict with these keys:
        "rows": number of rows (int).
        "columns": number of columns (int).
        "dtypes": dict mapping each column name to str(dtype).
        "missing": dict mapping each column name to its count of missing values.
        "duplicates": rows that exactly repeat an earlier row (int).

    Steps:
    1. df.shape gives (rows, columns); df.dtypes maps column to dtype.
    2. df.isna().sum() counts missing values per column.
    3. df.duplicated() marks every row that repeats an earlier row; sum it.
    """
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(k) for c, k in df.isna().sum().items()},
        "duplicates": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Stage 2: clean, one column kind at a time
# ---------------------------------------------------------------------------
def parse_number(text: pd.Series) -> pd.Series:
    """Pull the first number out of strings such as "£1,250", "£245 pw", "58 m2" or "".

    Args:
        text: A Series of strings or missing values, length n.

    Returns:
        A float Series of length n with the same index. The value is the first
        run of digits in the string, with thousands commas removed and an
        optional decimal part, so "£1,250 pcm" gives 1250.0 and "58 m2" gives
        58.0 (not 582). Missing text, or text with no digits, gives NaN.

    Steps:
    1. Cast to the pandas "string" dtype so missing values stay missing.
    2. Series.str.extract with a regex pulls out the first run of digits and commas.
    3. Remove the commas, then pd.to_numeric(..., errors="coerce").
    """
    first = text.astype("string").str.extract(r"(\d[\d,]*\.?\d*)", expand=False)
    return pd.to_numeric(first.str.replace(",", ""), errors="coerce").astype(float)


def to_weekly_rent(rent: pd.Series, period: pd.Series) -> pd.Series:
    """Convert every rent to pounds per week.

    Args:
        rent: Raw rent strings, length n.
        period: Raw period strings, length n. "week" or "month" in any case, or
            missing. When the period is missing, a "pcm" suffix in `rent` means
            monthly, and anything else means weekly.

    Returns:
        A float Series of weekly rents. A monthly rent becomes
        amount * 12 / 52. Missing rent gives NaN.

    Steps:
    1. Get the amount with parse_number.
    2. Build a boolean mask `monthly` from the period words and the "pcm" suffix. A comparison with a missing value gives <NA>, so fill missing with False before you combine masks.
    3. Series.where(cond, other) keeps values where cond is True and uses `other` elsewhere.
    """
    amount = parse_number(rent)
    word = period.astype("string").str.strip().str.lower().fillna("")
    says_month = word.eq("month")
    says_week = word.eq("week")
    suffix_month = rent.astype("string").str.contains("pcm", case=False, na=False)
    monthly = says_month | (~says_week & suffix_month)
    return amount.where(~monthly, amount * 12 / 52)


def to_square_metres(size: pd.Series, unit: pd.Series) -> pd.Series:
    """Convert every floor area to square metres.

    Args:
        size: Raw size strings such as "54.0", "58 sqm" or "624 sqft", length n.
        unit: Raw unit strings such as "m2", "SQFT", "sq ft", or missing.
            When the unit is missing, the letters in `size` name it. When
            neither names a unit, assume square metres.

    Returns:
        A float Series of areas in m2. Square feet are divided by 10.7639.
        Missing size gives NaN. Zero stays 0.

    Steps:
    1. Get the number with parse_number (watch "58 m2": the 2 is not part of the size).
    2. Take the unit from `unit`, and from `size` where `unit` is missing, then lower-case it.
    3. Build a mask of square-foot rows and divide only those by SQFT_PER_M2.
    """
    amount = parse_number(size)
    label = unit.astype("string").fillna(size.astype("string")).fillna("").str.lower()
    is_sqft = label.str.contains("sq") & label.str.contains("ft")
    return amount.where(~is_sqft, amount / SQFT_PER_M2)


def normalise_neighbourhood(names: pd.Series) -> pd.Series:
    """Map every spelling of a neighbourhood to one of NEIGHBOURHOODS.

    Args:
        names: Raw names with stray spaces, odd case, and misspellings.

    Returns:
        A Series of the same length. Strip the ends, squeeze repeated spaces,
        ignore case, and match each name to the closest entry of NEIGHBOURHOODS
        with difflib (cutoff 0.8). A name with no close match, or a missing
        name, becomes NaN.

    Steps:
    1. Clean each distinct name once: split and rejoin to squeeze spaces, then lower-case.
    2. difflib.get_close_matches(word, possibilities, n=1, cutoff=0.8) returns the closest entries.
    3. Series.map applies a function to every element.
    """
    canon = {name.lower(): name for name in NEIGHBOURHOODS}
    cache: dict[str, object] = {}

    def fix(raw: object) -> object:
        if pd.isna(raw):
            return np.nan
        key = " ".join(str(raw).split()).lower()
        if key not in cache:
            hit = difflib.get_close_matches(key, list(canon), n=1, cutoff=0.8)
            cache[key] = canon[hit[0]] if hit else np.nan
        return cache[key]

    return names.map(fix)


def parse_dates(text: pd.Series) -> pd.Series:
    """Parse dates written in three formats, day first.

    Args:
        text: Strings such as "2023-03-04", "04/03/2023", "04 Mar 2023", or missing.

    Returns:
        A datetime64 Series. "04/03/2023" is the 4th of March. Missing or
        unreadable text gives NaT.

    Steps:
    1. pd.to_datetime(text, format=..., errors="coerce") gives NaT for text that does not match.
    2. Try each format in turn and fill the gaps from the next one with Series.fillna.
    """
    clean = text.astype("string").str.strip()
    out = pd.to_datetime(clean, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%d %b %Y"):
        out = out.fillna(pd.to_datetime(clean, format=fmt, errors="coerce"))
    return out


def parse_yes_no(text: pd.Series) -> pd.Series:
    """Turn Y, N, yes, no, True, False (any case) into booleans.

    Args:
        text: Raw strings or missing values, length n.

    Returns:
        A Series of dtype "boolean" (which can hold missing values). Y, yes,
        and True in any case give True. N, no, and False in any case give
        False. Anything else, including missing, gives <NA>.

    Steps:
    1. Strip and lower-case the text.
    2. Start from a Series of dtype "boolean" filled with pd.NA, then set the True and False rows with Series.isin masks.
    """
    word = text.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=text.index, dtype="boolean")
    out[word.isin(["y", "yes", "true"]).fillna(False)] = True
    out[word.isin(["n", "no", "false"]).fillna(False)] = False
    return out


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step and drop rows that cannot be trusted.

    Args:
        df: The raw DataFrame from `load_listings`.

    Returns:
        A new DataFrame with a fresh 0..n-1 index and these columns, in order:
        listing_id (int), listed_on (datetime64), neighbourhood (str),
        area_m2 (float), bedrooms (float), dist_station_km (float),
        furnished (boolean), rent_week (float). Rows are removed for these
        reasons, in this order:
        1. Exact duplicate rows of the raw frame (keep the first).
        2. Missing rent, missing area, or an unmatched neighbourhood.
        3. Area outside the domain bound MIN_AREA_M2 to MAX_AREA_M2.
        4. Weekly rent per m2 outside Tukey's fence with k = 3, computed on the
           rows that remain: below Q1 - 3 * IQR or above Q3 + 3 * IQR.

    Steps:
    1. Drop duplicates first, so one flat does not count twice in the quartiles.
    2. Build the new frame from your parse functions, column by column.
    3. dropna(subset=[...]) removes rows missing a required column; Series.between(low, high) makes a bound mask.
    4. Series.quantile([0.25, 0.75]) gives Q1 and Q3; reset the index at the end.
    """
    raw = df.drop_duplicates()
    out = pd.DataFrame({
        "listing_id": raw["listing_id"],
        "listed_on": parse_dates(raw["listed_on"]),
        "neighbourhood": normalise_neighbourhood(raw["neighbourhood"]),
        "area_m2": to_square_metres(raw["size"], raw["size_unit"]),
        "bedrooms": raw["bedrooms"],
        "dist_station_km": raw["dist_station_km"],
        "furnished": parse_yes_no(raw["furnished"]),
        "rent_week": to_weekly_rent(raw["rent"], raw["rent_period"]),
    })
    out = out.dropna(subset=["rent_week", "area_m2", "neighbourhood"])
    out = out[out["area_m2"].between(MIN_AREA_M2, MAX_AREA_M2)]
    per_m2 = out["rent_week"] / out["area_m2"]
    q1, q3 = per_m2.quantile([0.25, 0.75])
    fence = 3.0 * (q3 - q1)
    out = out[per_m2.between(q1 - fence, q3 + fence)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 3: features and summaries
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three derived columns to a cleaned frame, without changing the input.

    Returns:
        A copy of `df` with extra columns:
        rent_per_m2: rent_week / area_m2 (float).
        month: the month number 1..12 of listed_on, dtype "Int64" (<NA> where the date
            is missing).
        size_band: area_m2 cut into SIZE_LABELS by SIZE_EDGES, with each band
            including its lower edge (pd.cut with right=False).

    Steps:
    1. Work on df.copy() so the caller's frame is unchanged.
    2. The .dt.month accessor reads the month from a datetime column; .astype("Int64") keeps missing dates as <NA> without turning months into floats.
    3. pd.cut(values, bins=SIZE_EDGES, labels=SIZE_LABELS, right=False) sorts areas into bands.
    """
    out = df.copy()
    out["rent_per_m2"] = out["rent_week"] / out["area_m2"]
    out["month"] = out["listed_on"].dt.month.astype("Int64")
    out["size_band"] = pd.cut(out["area_m2"], bins=SIZE_EDGES, labels=SIZE_LABELS, right=False)
    return out


def rent_per_m2_by_neighbourhood(df: pd.DataFrame) -> pd.Series:
    """Mean weekly rent per m2 for each neighbourhood.

    Args:
        df: A frame with the columns added by `add_features`.

    Returns:
        A float Series indexed by neighbourhood name, sorted from the highest
        mean to the lowest.

    Steps:
    1. df.groupby("neighbourhood")["rent_per_m2"].mean() gives one mean per group.
    2. sort_values(ascending=False) puts the highest first.
    """
    return df.groupby("neighbourhood")["rent_per_m2"].mean().sort_values(ascending=False)


def rent_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median weekly rent for every neighbourhood and size band.

    Returns:
        A pivot table with one row per neighbourhood (sorted by name), one
        column per size band in the order of SIZE_LABELS, and the median of
        rent_week in each cell. An empty cell holds NaN.

    Steps:
    1. DataFrame.pivot_table(index=..., columns=..., values=..., aggfunc="median") builds the grid.
    2. Pass observed=False so every band appears, then order the columns with SIZE_LABELS.
    """
    table = df.pivot_table(
        index="neighbourhood", columns="size_band", values="rent_week",
        aggfunc="median", observed=False,
    )
    return table[SIZE_LABELS]


def furnished_table(df: pd.DataFrame, min_each: int = 5) -> pd.DataFrame:
    """Compare furnished and unfurnished flats inside the same neighbourhood and size band.

    Args:
        df: A frame with the columns added by `add_features`. Rows with a
            missing `furnished` or `size_band` are ignored.
        min_each: Keep a (neighbourhood, size_band) cell only when it holds at
            least this many furnished and this many unfurnished flats.

    Returns:
        A DataFrame indexed by (neighbourhood, size_band) with the columns
        "unfurnished" and "furnished" (mean rent_week of each group), and
        "n" (the number of flats in the cell, both groups together).

    Steps:
    1. Group by neighbourhood, size_band, and furnished, then take the mean and the size of each group.
    2. Series.unstack("furnished") turns the last index level into columns False and True.
    3. Keep the cells where both counts reach min_each with a boolean mask.
    """
    rows = df.dropna(subset=["furnished", "size_band"])
    grouped = rows.groupby(["neighbourhood", "size_band", "furnished"], observed=True)["rent_week"]
    mean = grouped.mean().unstack("furnished")
    count = grouped.size().unstack("furnished")
    keep = (count[True] >= min_each) & (count[False] >= min_each)
    table = pd.DataFrame({
        "unfurnished": mean[False],
        "furnished": mean[True],
        "n": count[True] + count[False],
    })
    return table[keep]


def monthly_profile(df: pd.DataFrame) -> pd.Series:
    """Median weekly rent per m2 for each month a flat was listed.

    Returns:
        A float Series indexed by month number (1..12, only months that occur),
        sorted by month. Rows with a missing month are ignored.

    Steps:
    1. Group by the month column and take the median of rent_per_m2.
    2. sort_index puts the months in order.
    """
    return df.groupby("month")["rent_per_m2"].median().sort_index()


def distance_profile(df: pd.DataFrame) -> pd.Series:
    """Mean weekly rent per m2 for each distance-to-station band.

    Returns:
        A float Series indexed by the strings in DIST_LABELS, in that order.
        Distances are cut at DIST_EDGES, each band including its lower edge.
        Rows with a missing distance are ignored.

    Steps:
    1. pd.cut turns distances into bands, as in add_features.
    2. Group by that band Series and take the mean of rent_per_m2; pass observed=False to keep empty bands.
    """
    band = pd.cut(df["dist_station_km"], bins=DIST_EDGES, labels=DIST_LABELS, right=False)
    return df.groupby(band, observed=False)["rent_per_m2"].mean()


# ---------------------------------------------------------------------------
# Stage 4: plots (each returns a Figure and never calls plt.show)
# ---------------------------------------------------------------------------
def plot_rent_hist(before: pd.Series, after: pd.Series, bins: int = 40) -> Figure:
    """Two histograms of weekly rent side by side: before and after cleaning.

    Args:
        before: Rents at face value, in any units, with outliers. Missing
            values are ignored.
        after: Cleaned weekly rents. Missing values are ignored.
        bins: Number of bins in each histogram.

    Returns:
        A Figure with 2 axes (left: before, right: after). Each axis holds one
        histogram of counts, so its bar heights sum to the number of non-missing
        values it was given. Each axis has a title, an x label, and a y label.

    Steps:
    1. plt.subplots(1, 2) returns a figure and two axes.
    2. ax.hist(values, bins=bins) draws counts.
    3. Set the title, x label, and y label of each axis, and return the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, values, title in ((axes[0], before, "Before cleaning"), (axes[1], after, "After cleaning")):
        ax.hist(values.dropna(), bins=bins, color="#4c72b0")
        ax.set_title(title)
        ax.set_xlabel("rent as written (£)" if values is before else "weekly rent (£)")
        ax.set_ylabel("number of listings")
    fig.tight_layout()
    return fig


def plot_rent_vs_size(df: pd.DataFrame, by: str = "furnished") -> Figure:
    """Scatter plot of weekly rent against floor area, coloured by a column.

    Args:
        df: A cleaned frame with `area_m2`, `rent_week`, and the column `by`.
        by: Name of the column that sets the colour. Rows where it is missing
            are left out.

    Returns:
        A Figure with 1 axis holding one scatter call per distinct value of
        `by`, so that every plotted row appears exactly once across
        ax.collections. The axis has a title, an x label, a y label, and a legend.

    Steps:
    1. Drop rows where `by` is missing, then loop over rows.groupby(by).
    2. Call ax.scatter once per group with label=str(value), then ax.legend().
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = df.dropna(subset=[by])
    for value, group in rows.groupby(by):
        ax.scatter(group["area_m2"], group["rent_week"], s=10, alpha=0.5, label=str(value))
    ax.set_title(f"Rent against size, coloured by {by}")
    ax.set_xlabel("floor area (m²)")
    ax.set_ylabel("weekly rent (£)")
    ax.legend(title=by)
    fig.tight_layout()
    return fig


def plot_neighbourhood_bars(df: pd.DataFrame) -> Figure:
    """Bar chart of mean weekly rent per m2 by neighbourhood.

    Args:
        df: A frame with the columns added by `add_features`.

    Returns:
        A Figure with 1 axis and one bar per neighbourhood, in the order of
        `rent_per_m2_by_neighbourhood`, each bar as tall as that mean. The tick
        labels are the neighbourhood names. The axis has a title and a y label.

    Steps:
    1. Reuse rent_per_m2_by_neighbourhood for the heights.
    2. ax.bar(names, heights) draws one bar per name; rotate long tick labels.
    """
    means = rent_per_m2_by_neighbourhood(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(means.index, means.to_numpy(), color="#4c72b0")
    ax.set_title("Rent per square metre by neighbourhood")
    ax.set_xlabel("neighbourhood")
    ax.set_ylabel("mean weekly rent per m² (£)")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig


def plot_furnished_paradox(df: pd.DataFrame) -> Figure:
    """Show that furnished flats look cheaper overall but cost more like for like.

    Args:
        df: A frame with the columns added by `add_features`.

    Returns:
        A Figure with 2 axes.
        Left: 2 bars, the mean rent_week of unfurnished and then furnished
        flats over the whole frame (rows with missing `furnished` ignored).
        Right: grouped bars, two per neighbourhood in sorted name order
        (unfurnished then furnished), each the mean over that neighbourhood's
        cells in `furnished_table(df)` of the cell's "unfurnished" or
        "furnished" column. Both axes have a title and a y label, and
        the right axis has an x label and a legend.

    Steps:
    1. Left axis: groupby("furnished")["rent_week"].mean() gives two numbers for ax.bar.
    2. Right axis: furnished_table(df).groupby(level="neighbourhood").mean() gives one row per neighbourhood.
    3. Draw grouped bars by shifting the x positions of the two series by -0.2 and +0.2 with width 0.4.
    """
    overall = df.dropna(subset=["furnished"]).groupby("furnished")["rent_week"].mean()
    cells = furnished_table(df).groupby(level="neighbourhood").mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    left.bar(["unfurnished", "furnished"], [overall[False], overall[True]], color=["#4c72b0", "#dd8452"])
    left.set_title("Overall")
    left.set_ylabel("mean weekly rent (£)")
    x = np.arange(len(cells))
    right.bar(x - 0.2, cells["unfurnished"], width=0.4, color="#4c72b0", label="unfurnished")
    right.bar(x + 0.2, cells["furnished"], width=0.4, color="#dd8452", label="furnished")
    right.set_xticks(x, cells.index)
    right.tick_params(axis="x", labelrotation=45)
    right.set_title("Same neighbourhood and size band")
    right.set_xlabel("neighbourhood")
    right.set_ylabel("mean weekly rent (£)")
    right.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stage 5: insights and the loop back to chapters 001-004
# ---------------------------------------------------------------------------
def key_insights(df: pd.DataFrame) -> dict:
    """Collect the headline numbers from a cleaned frame with features.

    Returns:
        A dict with these keys:
        "top_neighbourhood": name with the highest mean rent per m2 (str).
        "bottom_neighbourhood": name with the lowest mean rent per m2 (str).
        "per_m2_ratio": top mean divided by bottom mean (float).
        "furnished_gap_overall": mean rent_week of furnished minus unfurnished
            flats over the whole frame, in pounds (float).
        "furnished_gap_within": the same difference inside each cell of
            `furnished_table(df)`, averaged over cells with weights n (float).
        "peak_month": the month (int) with the highest `monthly_profile`.
        "near_drop": 1 - (mean rent per m2 in band "1-2") / (mean in band "0-0.5"),
            from `distance_profile` (float).
        "far_drop": 1 - (mean in band "3+") / (mean in band "2-3") (float).

    Steps:
    1. Reuse the summary functions above instead of regrouping.
    2. np.average(values, weights=...) gives the weighted mean over cells.
    3. Series.idxmax() returns the label of the largest value.
    """
    by_hood = rent_per_m2_by_neighbourhood(df)
    furn = df.dropna(subset=["furnished"]).groupby("furnished")["rent_week"].mean()
    cells = furnished_table(df)
    within = np.average(cells["furnished"] - cells["unfurnished"], weights=cells["n"])
    month = monthly_profile(df)
    dist = distance_profile(df)
    return {
        "top_neighbourhood": str(by_hood.index[0]),
        "bottom_neighbourhood": str(by_hood.index[-1]),
        "per_m2_ratio": float(by_hood.iloc[0] / by_hood.iloc[-1]),
        "furnished_gap_overall": float(furn[True] - furn[False]),
        "furnished_gap_within": float(within),
        "peak_month": int(month.idxmax()),
        "near_drop": float(1 - dist["1-2"] / dist["0-0.5"]),
        "far_drop": float(1 - dist["3+"] / dist["2-3"]),
    }


def raw_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows of the raw frame whose rent and size can be read at face value.

    Returns:
        A DataFrame with columns listing_id, area (float), rent (float), where
        area is parse_number(df["size"]) and rent is parse_number(df["rent"]),
        with no unit conversion, no duplicate removal, and no outlier removal.
        Rows where either number is missing are dropped. The index is reset.

    Steps:
    1. parse_number on the "size" and "rent" columns gives area and rent at face value.
    2. DataFrame.dropna() removes rows with either number missing.
    """
    out = pd.DataFrame({
        "listing_id": df["listing_id"],
        "area": parse_number(df["size"]),
        "rent": parse_number(df["rent"]),
    })
    return out.dropna().reset_index(drop=True)


def compare_raw_vs_clean(csv_text: str, seed: int = 0) -> dict:
    """Fit a straight line of rent on size twice and score both on clean test flats.

    Steps: load the CSV, build the raw rows with `raw_rows`, and clean the
    frame. Choose the test flats from the cleaned frame with a seeded shuffle:
    the last floor(0.25 * n) listing_ids of np.random.default_rng(seed).permutation(n).
    Train the "clean" line on the other cleaned flats (area_m2, rent_week) and
    the "raw" line on every raw row whose listing_id is not a test id. Fit each
    line with `fit_polynomial(x, y, 1)`, and score both on the test flats with
    `rmse`.

    Returns:
        A dict with "rmse_raw" and "rmse_clean" (floats, in pounds per week),
        "n_train_raw" and "n_train_clean" (ints), and "n_test" (int).

    Steps:
    1. Shuffle the cleaned rows once with np.random.default_rng(seed).permutation(n) and cut off the test part.
    2. Series.isin removes the test ids from the raw rows, so no test flat helps train either line.
    3. Score both fitted lines on the same test flats.
    """
    df = load_listings(csv_text)
    raw = raw_rows(df)
    clean = clean_listings(df)
    n = len(clean)
    order = np.random.default_rng(seed).permutation(n)
    n_test = int(0.25 * n + 1e-9)
    test = clean.iloc[order[n - n_test:]]
    train = clean.iloc[order[: n - n_test]]
    raw_train = raw[~raw["listing_id"].isin(test["listing_id"])]
    w_clean = fit_polynomial(train["area_m2"].to_numpy(), train["rent_week"].to_numpy(), 1)
    w_raw = fit_polynomial(raw_train["area"].to_numpy(), raw_train["rent"].to_numpy(), 1)
    x_test, y_test = test["area_m2"].to_numpy(), test["rent_week"].to_numpy()
    return {
        "rmse_raw": rmse(predict_polynomial(w_raw, x_test), y_test),
        "rmse_clean": rmse(predict_polynomial(w_clean, x_test), y_test),
        "n_train_raw": int(len(raw_train)),
        "n_train_clean": int(len(train)),
        "n_test": int(n_test),
    }


if __name__ == "__main__":
    text, truth = make_data(0, with_truth=True)
    raw = load_listings(text)
    print("The first rows of the raw export:")
    print(raw.head(6).to_string())

    info = profile(raw)
    print(f"\nShape: {info['rows']} rows, {info['columns']} columns, {info['duplicates']} exact duplicates")
    print("Dtypes and missing values per column:")
    print(pd.DataFrame({"dtype": info["dtypes"], "missing": info["missing"]}).to_string())

    clean = add_features(clean_listings(raw))
    print(f"\nAfter cleaning: {len(clean)} rows (planted truth: {truth['n_clean']})")
    print(clean.head(4).to_string())

    print("\nMean weekly rent per m2 by neighbourhood:")
    print(rent_per_m2_by_neighbourhood(clean).round(2).to_string())
    print("\nMedian weekly rent (£) by neighbourhood and size band:")
    print(rent_table(clean).round(0).to_string())
    print("\nFurnished against unfurnished inside the same neighbourhood and size band (first rows):")
    print(furnished_table(clean).head(6).round(1).to_string())
    print("\nMedian rent per m2 by month listed:")
    print(monthly_profile(clean).round(2).to_string())
    print("\nMean rent per m2 by distance to the station (km):")
    print(distance_profile(clean).round(2).to_string())

    print("\nKey insights:")
    for name, value in key_insights(clean).items():
        print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")

    result = compare_raw_vs_clean(text, seed=0)
    print("\nStraight line of rent on size, scored on the same clean test flats:")
    print(f"  trained on raw rows   : test RMSE £{result['rmse_raw']:.1f} per week ({result['n_train_raw']} rows)")
    print(f"  trained on clean rows : test RMSE £{result['rmse_clean']:.1f} per week ({result['n_train_clean']} rows)")

    plot_rent_hist(raw_rows(raw)["rent"], clean["rent_week"])
    plt.show()
    plot_rent_vs_size(clean, by="furnished")
    plt.show()
    plot_neighbourhood_bars(clean)
    plt.show()
    plot_furnished_paradox(clean)
    plt.show()
