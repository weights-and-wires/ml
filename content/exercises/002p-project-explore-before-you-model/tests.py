"""Checks for the project: explore before you model."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mlcheck import almost, check, close, equal, fail, has

SIZE_EDGES = [0.0, 40.0, 55.0, 70.0, np.inf]
SIZE_LABELS = ["<40", "40-55", "55-70", "70+"]
CLEAN_COLUMNS = [
    "listing_id", "listed_on", "neighbourhood", "area_m2", "bedrooms",
    "dist_station_km", "furnished", "rent_week",
]

_CACHE: dict = {}


def close_nan(got, want, label, rtol=1e-6):
    """Like close, but a NaN in want must be a NaN in got."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape or (np.isnan(got) != np.isnan(want)).any():
        fail(f"{label}: expected {want.tolist()}, got {got.tolist()}")
    ok = ~np.isnan(want)
    close(got[ok], want[ok], rtol=rtol, label=label)


def pipeline(sub, seed):
    """Run load and clean on one seed, once per run of the checks."""
    key = (id(sub), seed)
    if key not in _CACHE:
        text, truth = sub.make_data(seed, with_truth=True)
        raw = sub.load_listings(text)
        before = raw.copy()
        clean = sub.clean_listings(raw)
        _CACHE[key] = (text, truth, raw, before, clean)
    return _CACHE[key]


def tiny_listings():
    """Eight hand-made cleaned flats with the feature columns already added."""
    df = pd.DataFrame({
        "neighbourhood": ["A", "A", "A", "B", "B", "B", "C", "C"],
        "area_m2": [39.9, 40.0, 54.9, 55.0, 69.9, 70.0, 100.0, 30.0],
        "rent_week": [100.0, 200.0, 300.0, 330.0, 140.0, 280.0, 500.0, 90.0],
        "listed_on": pd.to_datetime([
            "2023-01-10", "2023-01-20", "2023-02-01", "2023-02-15",
            "2023-02-20", "2023-03-05", "2023-03-06", "2023-03-07",
        ]),
        "dist_station_km": [0.0, 0.5, 0.99, 1.0, 1.99, 2.0, 3.0, 5.0],
    })
    return df


@check("load_listings reads the CSV text and profile reports shape, dtypes, missing values, and duplicates")
def test_load_and_profile(sub):
    text = "a,b\n1,x\n1,x\n2,\n3,y\n"
    df = has(sub, "load_listings")(text)
    if not isinstance(df, pd.DataFrame):
        fail(f"load_listings should return a DataFrame, got {type(df).__name__}")
    equal(df.shape, (4, 2), "shape of the loaded frame")
    got = has(sub, "profile")(df)
    equal(got["rows"], 4, "profile rows")
    equal(got["columns"], 2, "profile columns")
    equal(got["missing"], {"a": 0, "b": 1}, "profile missing counts")
    equal(got["duplicates"], 1, "profile duplicates (a repeat of an earlier row, not both rows)")
    equal(sorted(got["dtypes"]), ["a", "b"], "profile dtypes keys")
    equal(got["dtypes"]["a"], "int64", "dtype of column a")
    big = sub.load_listings(sub.make_data(0))
    equal(big.shape, (2070, 10), "shape of the raw export")


@check("parse_number and to_weekly_rent read money text and convert monthly rents to weekly")
def test_money(sub):
    s = pd.Series(["£1,250", "£245 pw", "58 m2", "", "1,234.5", None, "624 sqft"])
    got = has(sub, "parse_number")(s)
    close_nan(got.to_numpy(dtype=float), np.array([1250, 245, 58, np.nan, 1234.5, np.nan, 624.0]),
          label="parse_number (58 m2 must give 58, not 582)")
    rent = pd.Series(["£520", "£1,300 pcm", "£260 pw", "£2,600", None, "£100"], index=[10, 11, 12, 13, 14, 15])
    period = pd.Series(["week", "month", None, "Month", None, None], index=[10, 11, 12, 13, 14, 15])
    out = has(sub, "to_weekly_rent")(rent, period)
    equal(list(out.index), [10, 11, 12, 13, 14, 15], "index of to_weekly_rent")
    close_nan(out.to_numpy(dtype=float), np.array([520.0, 300.0, 260.0, 600.0, np.nan, 100.0]),
          label="weekly rents (a missing period with no pcm suffix means weekly; monthly x 12 / 52)")


@check("to_square_metres reads the unit from the unit column or the size text and converts square feet")
def test_area(sub):
    size = pd.Series(["54", "58 sqm", "1076.39", "582 sqft", "", "0", "538.195"], index=list("abcdefg"))
    unit = pd.Series(["m2", None, "sqft", None, None, "M2", "sq ft"], index=list("abcdefg"))
    out = has(sub, "to_square_metres")(size, unit)
    equal(list(out.index), list("abcdefg"), "index of to_square_metres")
    close_nan(out.to_numpy(dtype=float), np.array([54.0, 58.0, 100.0, 582 / 10.7639, np.nan, 0.0, 50.0]),
              label="areas in m2", rtol=1e-4)


@check("Neighbourhood names, yes/no flags, and dates are normalised")
def test_categories_and_dates(sub):
    names = pd.Series(["riverside", " OLD  TOWN ", "Rivrside", "Dockland", "Marlow Pk",
                       "Northwy", "Kingsfeild", None, "Atlantis", "Hilcrest", "Eastgat"])
    got = has(sub, "normalise_neighbourhood")(names).tolist()
    want = ["Riverside", "Old Town", "Riverside", "Docklands", "Marlow Park",
            "Northway", "Kingsfield", None, None, "Hillcrest", "Eastgate"]
    for g, w, raw in zip(got, want, names):
        if (w is None) != pd.isna(g) or (w is not None and g != w):
            fail(f"normalise_neighbourhood({raw!r}) gave {g!r}, expected {w!r}")
    flags = pd.Series(["Y", "n", "YES", "no", "True", "False", " y ", None, "maybe", ""])
    out = has(sub, "parse_yes_no")(flags)
    if str(out.dtype) != "boolean":
        fail(f"parse_yes_no should return dtype 'boolean' (it holds missing values), got {out.dtype}")
    want_flags = [True, False, True, False, True, False, True, None, None, None]
    for g, w, raw in zip(out.tolist(), want_flags, flags):
        if (w is None) != pd.isna(g) or (w is not None and bool(g) != w):
            fail(f"parse_yes_no({raw!r}) gave {g!r}, expected {w!r}")
    dates = pd.Series(["2023-03-04", "04/03/2023", "04 Mar 2023", "31/12/2023", None, "garbage"])
    d = has(sub, "parse_dates")(dates)
    if not str(d.dtype).startswith("datetime64"):
        fail(f"parse_dates should return a datetime64 Series, got {d.dtype}")
    want_d = pd.to_datetime(["2023-03-04", "2023-03-04", "2023-03-04", "2023-12-31", None, None])
    if not (d.isna().to_numpy() == want_d.isna()).all() or not (d.dropna().to_numpy() == want_d.dropna().to_numpy()).all():
        fail(f"parse_dates gave {d.tolist()}, expected {want_d.tolist()} (day comes first in 04/03/2023)")


@check("clean_listings keeps exactly the planted number of rows, with no duplicates or impossible values")
def test_clean_rows(sub):
    text, truth, raw, before, clean = pipeline(sub, 0)
    if not before.equals(raw):
        fail("clean_listings changed the DataFrame it was given; work on a copy")
    equal(list(clean.columns), CLEAN_COLUMNS, "columns of the cleaned frame")
    n, want = len(clean), truth["n_clean"]
    if not (want - 3 <= n <= want):
        fail(f"cleaned frame has {n} rows, expected {want} (up to 3 fewer is accepted). "
             f"Raw rows: {len(raw)}, planted duplicates: {truth['n_duplicates']}.")
    if clean["listing_id"].duplicated().any():
        fail("the cleaned frame still has repeated listing_id values")
    left = sorted(set(clean["listing_id"]) & set(truth["dropped_ids"]))
    if left:
        fail(f"{len(left)} listings that should be dropped are still there, for example ids {left[:5]} "
             "(a zero or impossible area, a rent with an extra zero, or a missing rent or size)")
    for col in ("rent_week", "area_m2", "neighbourhood"):
        if clean[col].isna().any():
            fail(f"column {col} still has missing values")
    if clean["area_m2"].min() < 10 or clean["area_m2"].max() > 300:
        fail(f"areas run from {clean['area_m2'].min():.1f} to {clean['area_m2'].max():.1f} m2; "
             "an impossible area is still in the frame")
    top = (clean["rent_week"] / clean["area_m2"]).max()
    if top > 12:
        fail(f"the largest rent per m2 is {top:.1f} pounds a week, so a typo outlier survived")
    if clean.index.tolist() != list(range(len(clean))):
        fail("reset the index of the cleaned frame to 0..n-1")


@check("clean_listings gives weekly pounds, square metres, one set of neighbourhoods, and the right flags and months")
def test_clean_values(sub):
    text, truth, raw, before, clean = pipeline(sub, 0)
    ids = clean["listing_id"].to_numpy()
    stray = [int(i) for i in ids if i not in truth["rent_week"]]
    if stray:
        fail(f"{len(stray)} rows should have been dropped (a bad area, a typo rent, or a missing value), for example ids {stray[:5]}")
    rent_true = np.array([truth["rent_week"][i] for i in ids])
    area_true = np.array([truth["area_m2"][i] for i in ids])
    bad_rent = np.abs(clean["rent_week"].to_numpy() - rent_true) / rent_true > 0.01
    if bad_rent.any():
        i = int(np.argmax(bad_rent))
        fail(f"{bad_rent.sum()} rents are not weekly pounds. Listing {ids[i]}: got {clean['rent_week'].iloc[i]:.1f}, "
             f"expected about {rent_true[i]:.1f} (monthly rents need x 12 / 52)")
    bad_area = np.abs(clean["area_m2"].to_numpy() - area_true) > 0.1 + 0.01 * area_true
    if bad_area.any():
        i = int(np.argmax(bad_area))
        fail(f"{bad_area.sum()} areas are not in m2. Listing {ids[i]}: got {clean['area_m2'].iloc[i]:.1f}, "
             f"expected about {area_true[i]:.1f} (square feet need / 10.7639)")
    equal(sorted(clean["neighbourhood"].unique().tolist()), sorted(truth["neighbourhoods"]), "set of neighbourhoods")
    if str(clean["furnished"].dtype) != "boolean":
        fail(f"furnished should have dtype 'boolean', got {clean['furnished'].dtype}")
    known = clean["furnished"].notna().to_numpy()
    furn_true = np.array([truth["furnished"][i] for i in ids])
    if not (clean["furnished"][known].astype(bool).to_numpy() == furn_true[known]).all():
        fail("some furnished flags disagree with the planted truth (check Y, N, yes, no, True, False in any case)")
    if known.sum() < 0.9 * len(clean):
        fail("too many furnished flags are missing after cleaning; only the planted blanks should be missing")
    if not str(clean["listed_on"].dtype).startswith("datetime64"):
        fail(f"listed_on should be datetime64, got {clean['listed_on'].dtype}")
    dated = clean["listed_on"].notna().to_numpy()
    month_true = np.array([truth["month"][i] for i in ids])
    if dated.sum() < 0.9 * len(clean):
        fail("too many dates failed to parse; all three formats should work")
    if not (clean["listed_on"].dt.month[dated].to_numpy() == month_true[dated]).all():
        fail("some months disagree with the planted truth (day comes first in 04/03/2023)")


@check("add_features adds rent per m2, month, and size band without touching the input")
def test_features(sub):
    df = tiny_listings()
    keep = df.copy()
    out = has(sub, "add_features")(df)
    if not df.equals(keep):
        fail("add_features changed its input; work on a copy")
    close(out["rent_per_m2"].to_numpy(dtype=float), (df["rent_week"] / df["area_m2"]).to_numpy(), label="rent_per_m2")
    equal(out["month"].astype(int).tolist(), [1, 1, 2, 2, 2, 3, 3, 3], "month")
    got = out["size_band"].astype(str).tolist()
    want = ["<40", "40-55", "40-55", "55-70", "55-70", "70+", "70+", "<40"]
    equal(got, want, "size_band (each band includes its lower edge: 40.0 is in 40-55, 70.0 is in 70+)")
    gap = df.copy()
    gap.loc[0, "listed_on"] = pd.NaT
    month = sub.add_features(gap)["month"]
    if not pd.isna(month.iloc[0]):
        fail("a missing date should give a missing month")


@check("The groupby and pivot summaries match hand-computed values")
def test_summaries(sub):
    df = sub.add_features(tiny_listings())
    per = df["rent_week"] / df["area_m2"]
    got = has(sub, "rent_per_m2_by_neighbourhood")(df)
    want = per.groupby(df["neighbourhood"]).mean().sort_values(ascending=False)
    equal(got.index.tolist(), want.index.tolist(), "neighbourhood order (highest mean rent per m2 first)")
    close(got.to_numpy(dtype=float), want.to_numpy(), label="mean rent per m2")
    table = has(sub, "rent_table")(df)
    equal(table.index.tolist(), ["A", "B", "C"], "rent_table rows")
    equal([str(c) for c in table.columns], SIZE_LABELS, "rent_table columns")
    close(float(table.loc["A", "40-55"]), 250.0, label="median rent of A in band 40-55")
    close(float(table.loc["B", "55-70"]), 235.0, label="median rent of B in band 55-70")
    if not pd.isna(table.loc["A", "70+"]):
        fail("an empty neighbourhood and band should hold NaN in rent_table")
    # a furnished-vs-unfurnished frame with a planted paradox
    rows = []
    for k in range(10):
        rows.append(("A", 60.0, 300.0 + k, False))
    for k in range(5):
        rows.append(("A", 60.0, 330.0 + k, True))
    for k in range(5):
        rows.append(("B", 30.0, 100.0 + k, False))
    for k in range(10):
        rows.append(("B", 30.0, 110.0 + k, True))
    rows.append(("A", 60.0, 999.0, None))
    f = pd.DataFrame(rows, columns=["neighbourhood", "area_m2", "rent_week", "furnished"])
    f["furnished"] = f["furnished"].astype("boolean")
    f["listed_on"] = pd.to_datetime("2023-05-01")
    f = sub.add_features(f)
    t = has(sub, "furnished_table")(f)
    equal(len(t), 2, "cells kept (a cell needs at least 5 furnished and 5 unfurnished flats)")
    equal(sorted(t.columns.tolist()), ["furnished", "n", "unfurnished"], "furnished_table columns")
    close(t.loc[("A", "55-70"), "unfurnished"], 304.5, label="A unfurnished mean")
    close(t.loc[("A", "55-70"), "furnished"], 332.0, label="A furnished mean")
    close(t.loc[("B", "<40"), "unfurnished"], 102.0, label="B unfurnished mean")
    close(t.loc[("B", "<40"), "furnished"], 114.5, label="B furnished mean")
    close(float(t.loc[("A", "55-70"), "n"]), 15.0, label="cell size n (the row with a missing flag is ignored)")
    # months and distances
    mp = has(sub, "monthly_profile")(sub.add_features(tiny_listings()))
    equal([int(m) for m in mp.index], [1, 2, 3], "months in the profile")
    jan = np.median([100 / 39.9, 200 / 40.0])
    close(float(mp.iloc[0]), jan, label="median rent per m2 in January")
    dp = has(sub, "distance_profile")(sub.add_features(tiny_listings()))
    equal([str(i) for i in dp.index], ["0-0.5", "0.5-1", "1-2", "2-3", "3+"], "distance bands")
    close(float(dp["0-0.5"]), 100 / 39.9, label="band 0-0.5 holds only the flat at 0.0 km")
    close(float(dp["0.5-1"]), np.mean([200 / 40.0, 300 / 54.9]), label="band 0.5-1 includes 0.5 km and 0.99 km")
    close(float(dp["3+"]), np.mean([500 / 100.0, 90 / 30.0]), label="band 3+ includes 3.0 km")


@check("plot_rent_hist and plot_neighbourhood_bars return Figures whose data match the numbers")
def test_plots_a(sub):
    before = pd.Series([100.0, 120.0, 130.0, np.nan, 2500.0, 140.0, 110.0, 135.0])
    after = pd.Series([100.0, 120.0, 130.0, 140.0, 110.0, 135.0, np.nan])
    fig = has(sub, "plot_rent_hist")(before, after, bins=10)
    try:
        if not isinstance(fig, Figure):
            fail(f"plot_rent_hist should return a matplotlib Figure, got {type(fig).__name__}")
        equal(len(fig.axes), 2, "number of axes in plot_rent_hist")
        for ax, series, name in zip(fig.axes, (before, after), ("before", "after")):
            total = sum(p.get_height() for p in ax.patches)
            almost(total, series.notna().sum(), 1e-9, label=f"histogram counts in the {name} panel (the bar heights must add up to the number of values)")
            if not (ax.get_xlabel() and ax.get_ylabel() and ax.get_title()):
                fail(f"the {name} panel needs a title, an x label, and a y label")
    finally:
        plt.close(fig)

    df = sub.add_features(tiny_listings())
    fig = has(sub, "plot_neighbourhood_bars")(df)
    try:
        if not isinstance(fig, Figure):
            fail(f"plot_neighbourhood_bars should return a Figure, got {type(fig).__name__}")
        equal(len(fig.axes), 1, "number of axes in plot_neighbourhood_bars")
        ax = fig.axes[0]
        per = (df["rent_week"] / df["area_m2"]).groupby(df["neighbourhood"]).mean().sort_values(ascending=False)
        close(np.array([p.get_height() for p in ax.patches]), per.to_numpy(), label="bar heights (mean rent per m2, highest first)")
        equal([t.get_text() for t in ax.get_xticklabels()], per.index.tolist(), "bar labels")
        if not (ax.get_ylabel() and ax.get_title()):
            fail("the bar chart needs a title and a y label")
    finally:
        plt.close(fig)


@check("plot_rent_vs_size and plot_furnished_paradox plot every row and reveal the paradox")
def test_plots_b(sub):
    df = tiny_listings()
    df["furnished"] = pd.array([True, False, True, None, False, True, False, True], dtype="boolean")
    fig = has(sub, "plot_rent_vs_size")(df, by="furnished")
    try:
        if not isinstance(fig, Figure):
            fail(f"plot_rent_vs_size should return a Figure, got {type(fig).__name__}")
        equal(len(fig.axes), 1, "number of axes in plot_rent_vs_size")
        ax = fig.axes[0]
        points = np.vstack([c.get_offsets() for c in ax.collections])
        known = df.dropna(subset=["furnished"])
        equal(len(points), len(known), "number of plotted points (rows with a missing 'by' value are left out)")
        got = sorted(map(tuple, np.round(points, 6).tolist()))
        want = sorted(zip(known["area_m2"].round(6), known["rent_week"].round(6)))
        close(np.array(got), np.array(want), label="plotted (area, rent) points")
        if len(ax.collections) < 2:
            fail("draw one scatter per value of the colouring column so each gets its own colour and legend entry")
        if ax.get_legend() is None:
            fail("add a legend")
        if not (ax.get_xlabel() and ax.get_ylabel() and ax.get_title()):
            fail("the scatter needs a title, an x label, and a y label")
    finally:
        plt.close(fig)

    rows = []
    for k in range(10):
        rows.append(("A", 60.0, 300.0, False))
    for k in range(5):
        rows.append(("A", 60.0, 330.0, True))
    for k in range(5):
        rows.append(("B", 30.0, 100.0, False))
    for k in range(10):
        rows.append(("B", 30.0, 110.0, True))
    f = pd.DataFrame(rows, columns=["neighbourhood", "area_m2", "rent_week", "furnished"])
    f["furnished"] = f["furnished"].astype("boolean")
    f["rent_per_m2"] = f["rent_week"] / f["area_m2"]
    f["size_band"] = pd.cut(f["area_m2"], bins=SIZE_EDGES, labels=SIZE_LABELS, right=False)
    fig = has(sub, "plot_furnished_paradox")(f)
    try:
        if not isinstance(fig, Figure):
            fail(f"plot_furnished_paradox should return a Figure, got {type(fig).__name__}")
        equal(len(fig.axes), 2, "number of axes in plot_furnished_paradox")
        left, right = fig.axes
        lh = sorted(p.get_height() for p in left.patches)
        close(np.array(lh), np.array([550 / 3, 700 / 3]), label="left bars: overall means (furnished 183.3 looks cheaper than unfurnished 233.3)")
        rh = sorted(p.get_height() for p in right.patches)
        close(np.array(rh), np.array([100.0, 110.0, 300.0, 330.0]), label="right bars: like-for-like means per neighbourhood")
        if right.get_legend() is None:
            fail("add a legend to the right panel")
        for ax in (left, right):
            if not (ax.get_ylabel() and ax.get_title()):
                fail("every panel needs a title and a y label")
        if not right.get_xlabel():
            fail("the right panel needs an x label")
    finally:
        plt.close(fig)


@check("key_insights recovers the planted effects on seeds 0 to 2")
def test_insights(sub):
    for seed in range(3):
        text, truth, raw, before, clean = pipeline(sub, seed)
        feats = sub.add_features(clean)
        k = has(sub, "key_insights")(feats)
        for key in ("top_neighbourhood", "bottom_neighbourhood", "per_m2_ratio", "furnished_gap_overall",
                    "furnished_gap_within", "peak_month", "near_drop", "far_drop"):
            if key not in k:
                fail(f"key_insights is missing the key {key!r}")
        label = f"seed {seed}"
        if k["top_neighbourhood"] != "Docklands" or k["bottom_neighbourhood"] != "Northway":
            fail(f"{label}: expected Docklands highest and Northway lowest, got {k['top_neighbourhood']} and {k['bottom_neighbourhood']}")
        if not (1.6 < k["per_m2_ratio"] < 2.4):
            fail(f"{label}: top-to-bottom rent per m2 ratio is {k['per_m2_ratio']:.2f}, expected between 1.6 and 2.4")
        if not k["furnished_gap_overall"] < -20:
            fail(f"{label}: overall, furnished flats should look cheaper by more than 20 pounds; gap is {k['furnished_gap_overall']:.1f}")
        if not k["furnished_gap_within"] > 5:
            fail(f"{label}: within a neighbourhood and size band, furnished flats should cost more; gap is {k['furnished_gap_within']:.1f}")
        if k["peak_month"] != 9:
            fail(f"{label}: expected rent per m2 to peak in month 9, got {k['peak_month']}")
        if not (k["near_drop"] > 0.08 and k["far_drop"] < 0.06 and k["near_drop"] > k["far_drop"] + 0.07):
            fail(f"{label}: expected a steep drop near the station and a flat tail, got near_drop {k['near_drop']:.3f}, far_drop {k['far_drop']:.3f}")


@check("A straight line trained on cleaned rows beats one trained on raw rows, on the same clean test flats, for seeds 0 to 4")
def test_end_to_end(sub):
    for seed in range(5):
        text = sub.make_data(seed)
        r = has(sub, "compare_raw_vs_clean")(text, seed)
        label = f"seed {seed}"
        n_clean = len(pipeline(sub, seed)[4])
        equal(r["n_train_clean"] + r["n_test"], n_clean, f"{label}: clean train plus test rows")
        if not (0.2 * n_clean < r["n_test"] < 0.3 * n_clean):
            fail(f"{label}: expected about 25% of {n_clean} clean flats as test rows, got {r['n_test']}")
        if r["n_train_raw"] <= r["n_train_clean"]:
            fail(f"{label}: the raw training set should keep duplicates and bad rows, so it is larger than the clean one; "
                 f"got {r['n_train_raw']} raw against {r['n_train_clean']} clean")
        if not (20 < r["rmse_clean"] < 80):
            fail(f"{label}: clean test RMSE is {r['rmse_clean']:.1f}; expected between 20 and 80 pounds a week")
        if not r["rmse_clean"] < 0.5 * r["rmse_raw"]:
            fail(f"{label}: cleaning should cut the test RMSE to under half: raw {r['rmse_raw']:.1f}, clean {r['rmse_clean']:.1f}")
