"""Tests for edp_tool.

These use the non-interactive Agg backend so they run headless.
"""

import matplotlib

matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

from edp_tool import edp, _wilson_interval, _make_continuous_groups


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    p = 1 / (1 + np.exp(-(x - 5)))
    y = (rng.random(200) < p).astype(int)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def regression_df():
    x = np.linspace(0, 10, 200)
    y = 2 * x + 1.0
    return pd.DataFrame({"x": x, "y": y})


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #
def test_wilson_matches_known_value():
    # 8 successes out of 10 -> classic textbook Wilson 95% CI ~ (0.49, 0.94).
    center, low, high = _wilson_interval(8, 10)
    assert low == pytest.approx(0.490, abs=1e-3)
    assert high == pytest.approx(0.943, abs=1e-3)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_stays_in_unit_interval_at_extremes():
    _, low, high = _wilson_interval(0, 20)
    assert low >= 0.0
    _, low2, high2 = _wilson_interval(20, 20)
    assert high2 <= 1.0


def test_wilson_empty_is_nan():
    center, low, high = _wilson_interval(0, 0)
    assert np.isnan(center) and np.isnan(low) and np.isnan(high)


# --------------------------------------------------------------------------- #
# Bin construction bug fixes
# --------------------------------------------------------------------------- #
def test_max_value_included_in_last_bin():
    # Historically the maximum value fell outside the final half-open bin.
    x = np.arange(0, 101, dtype=float)  # includes 100.0, the max
    groups = _make_continuous_groups(
        x, n=4, binning="quantile", digits=2, even_spaced_ticks=False
    )
    total = sum(int(g["mask"].sum()) for g in groups)
    assert total == x.size  # every point, including the max, is counted


def test_n_does_not_leak_across_features():
    # A near-constant feature must not shrink the bin count used for later,
    # well-behaved features. n is copied locally, so this holds by construction.
    df = pd.DataFrame(
        {
            "flat": [1.0, 1.0, 1.0, 1.0, 2.0],  # collapses to few bins
            "good": np.linspace(0, 10, 5),
            "y": [0, 1, 0, 1, 1],
        }
    )
    res = edp(df, ["flat", "good"], "y", n=4, show=False, close=True)
    good = next(r for r in res if r["feature"] == "good")
    # 'good' still gets its full binning (line has n data points).
    line = good["ax"].lines[0]
    assert len(line.get_xdata()) >= 3


# --------------------------------------------------------------------------- #
# End-to-end behaviour
# --------------------------------------------------------------------------- #
def test_binary_returns_one_series_and_rates_increase(binary_df):
    res = edp(binary_df, ["x"], "y", n=5, show=False, close=True)
    assert len(res) == 1
    ax = res[0]["ax"]
    # single positive-class line (+ bin count on the twin axis).
    y = ax.lines[0].get_ydata()
    assert y.min() >= 0.0 and y.max() <= 1.0
    # monotone logistic signal -> last bin rate clearly above first.
    assert y[-1] > y[0]


def test_regression_mean_tracks_linear_signal(regression_df):
    res = edp(regression_df, ["x"], "y", n=5, show=False, close=True)
    y = res[0]["ax"].lines[0].get_ydata()
    assert np.all(np.diff(y) > 0)  # increasing feature -> increasing mean


def test_multiclass_one_line_per_class():
    rng = np.random.default_rng(1)
    x = rng.random(300)
    y = pd.cut(x, [0, 0.33, 0.66, 1.0], labels=["a", "b", "c"]).astype(str)
    df = pd.DataFrame({"x": x, "y": y})
    res = edp(df, ["x"], "y", n=4, show=False, close=True, show_baseline=False)
    # 3 class lines + 1 bin-count line across the two axes.
    total_lines = sum(len(a.lines) for a in res[0]["fig"].axes)
    assert total_lines == 4


def test_categorical_feature_uses_one_group_per_value():
    df = pd.DataFrame(
        {"grp": ["a", "b", "c"] * 20, "y": ([0, 1, 1] * 20)}
    )
    res = edp(df, ["grp"], "y", show=False, close=True)
    ax = res[0]["ax"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_writefolder_saves_png(tmp_path, binary_df):
    res = edp(binary_df, ["x"], "y", writefolder=str(tmp_path), show=False)
    assert (tmp_path / res[0]["path"].split("\\")[-1].split("/")[-1]).exists()
    assert res[0]["path"].endswith(".png")


def test_missing_feature_is_skipped(binary_df):
    res = edp(binary_df, ["nope", "x"], "y", show=False, close=True)
    assert [r["feature"] for r in res] == ["x"]


def test_feature_equal_to_target_is_skipped(binary_df):
    res = edp(binary_df, ["y", "x"], "y", show=False, close=True)
    assert [r["feature"] for r in res] == ["x"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_non_str_yname_raises(binary_df):
    with pytest.raises(TypeError):
        edp(binary_df, ["x"], ["y"], show=False)


def test_missing_yname_raises(binary_df):
    with pytest.raises(ValueError):
        edp(binary_df, ["x"], "missing", show=False)


def test_str_features_raises(binary_df):
    with pytest.raises(TypeError):
        edp(binary_df, "x", "y", show=False)


# --------------------------------------------------------------------------- #
# Backwards-compat shim
# --------------------------------------------------------------------------- #
def test_pdp_shim_warns(binary_df):
    from pdp_tool import pdp

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pdp(binary_df, ["x"], "y", show=False, close=True)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
