"""edp_tool -- Empirical Dependence Plot (EDP).

An exploratory data analysis tool that shows how the empirical mean (for a
continuous target) or the observed class rate (for a categorical/binary target)
of a variable changes across binned or grouped values of a feature.

Unlike a Partial Dependence Plot (PDP), the EDP uses **no model** and performs
**no marginalization**: it plots the *observed* conditional mean of the target
given a feature, computed directly from the data. It is closely related to the
"M-plot" (marginal plot) from the ALE literature (Apley & Zhu, 2020).

Basic usage
-----------
>>> from edp_tool import edp
>>> edp(df, ["age", "income"], "converted")          # binary target
>>> edp(df, ["age"], "price")                         # continuous target
>>> edp(df, ["age"], "species")                       # multiclass target

The function is import-safe: it only imports matplotlib's pyplot lazily-friendly
default backend. For headless use set ``matplotlib.use("Agg")`` before calling.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["edp"]

# z-score for a two-sided 95% confidence interval.
_Z_95 = 1.959963984540054

_FILENAME_REPLACE = ("/", "\\", " ", "(", ")", ":", "*", "?", '"', "<", ">", "|")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sanitize_filename(text: Any) -> str:
    """Turn an arbitrary label into a filesystem-safe token."""
    out = str(text)
    for ch in _FILENAME_REPLACE:
        out = out.replace(ch, "_")
    return out


def _wilson_interval(
    k: np.ndarray, n_obs: np.ndarray, z: float = _Z_95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : successes (array-like)
    n_obs : trials (array-like)
    z : z-score for the desired confidence level.

    Returns
    -------
    (center, low, high) as float arrays. Entries where ``n_obs == 0`` are NaN.

    The Wilson interval is far better behaved than the normal approximation
    near p = 0 and p = 1 (where binary rates commonly live) and never leaves
    the [0, 1] range.
    """
    k = np.asarray(k, dtype=float)
    n_obs = np.asarray(n_obs, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n_obs
        denom = 1.0 + z ** 2 / n_obs
        center = (p + z ** 2 / (2 * n_obs)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n_obs + z ** 2 / (4 * n_obs ** 2))
        low = center - half
        high = center + half
    empty = n_obs == 0
    center = np.where(empty, np.nan, center)
    low = np.where(empty, np.nan, low)
    high = np.where(empty, np.nan, high)
    return center, low, high


def _is_categorical_feature(s: pd.Series, max_categories: int) -> bool:
    """Decide whether a feature should be treated as categorical.

    A feature is categorical when its dtype is non-numeric (object, category,
    bool) or when it is numeric but has few distinct values.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if s.dtype.kind in ("O", "U", "S", "b"):
        return True
    return s.nunique(dropna=True) <= max_categories


def _target_kind(s: pd.Series, max_categories: int) -> str:
    """Classify the target as 'binary', 'multiclass' or 'continuous'."""
    n_unique = s.nunique(dropna=True)
    if n_unique <= 2:
        return "binary"
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype.kind in ("O", "U", "S", "b"):
        return "multiclass"
    if n_unique <= max_categories:
        return "multiclass"
    return "continuous"


def _make_continuous_groups(
    x: np.ndarray,
    n: int,
    binning: str,
    digits: int,
    even_spaced_ticks: bool,
) -> List[Dict[str, Any]]:
    """Build contiguous bins for a continuous feature.

    ``n`` is treated as a *local* upper bound on the number of bins and is
    shrunk only if repeated values collapse the bin edges -- it is never
    mutated for the caller (this fixes the historical bug where ``n`` leaked
    across features).
    """
    n_bins = int(n)
    if binning == "quantile":
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
        while edges.size <= 2 and n_bins > 1:
            n_bins -= 1
            edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    elif binning == "uniform":
        edges = np.unique(np.linspace(x.min(), x.max(), n_bins + 1))
    else:
        raise ValueError(f"binning must be 'quantile' or 'uniform', got {binning!r}.")

    if edges.size <= 2:
        return []

    groups: List[Dict[str, Any]] = []
    last = edges.size - 2
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == last:
            # Include the maximum value in the final bin (right-closed) so the
            # extreme points are never silently dropped.
            mask = (x >= lo) & (x <= hi)
            label = "[{:.{d}f}-{:.{d}f}]".format(lo, hi, d=digits)
        else:
            mask = (x >= lo) & (x < hi)
            label = "[{:.{d}f}-{:.{d}f}[".format(lo, hi, d=digits)
        xpos = (lo + hi) / 2 if even_spaced_ticks else i
        groups.append({"label": label, "x": xpos, "mask": mask})
    return groups


def _make_categorical_groups(x: np.ndarray) -> List[Dict[str, Any]]:
    """Build one group per distinct value of a categorical feature."""
    values = pd.unique(pd.Series(x).dropna())
    try:
        values = np.sort(values)
    except TypeError:
        values = np.array(sorted(values, key=str), dtype=object)
    groups: List[Dict[str, Any]] = []
    for i, val in enumerate(values):
        groups.append({"label": str(val), "x": i, "mask": x == val})
    return groups


def _regression_series(
    y: np.ndarray, groups: Sequence[Dict[str, Any]], show_ci: bool, z: float
) -> Dict[str, Any]:
    point, low, high = [], [], []
    for g in groups:
        yv = y[g["mask"]]
        m = float(np.mean(yv))
        point.append(m)
        if show_ci and yv.size > 1:
            sem = float(np.std(yv, ddof=1)) / np.sqrt(yv.size)
            low.append(m - z * sem)
            high.append(m + z * sem)
        else:
            low.append(np.nan)
            high.append(np.nan)
    return {"point": np.array(point), "low": np.array(low), "high": np.array(high)}


def _class_series(
    y: np.ndarray,
    cls: Any,
    groups: Sequence[Dict[str, Any]],
    show_ci: bool,
    ci_method: str,
    z: float,
) -> Dict[str, Any]:
    point, low, high = [], [], []
    for g in groups:
        mask = g["mask"]
        n_g = int(mask.sum())
        k = int((y[mask] == cls).sum())
        rate = k / n_g
        point.append(rate)
        if not show_ci:
            low.append(np.nan)
            high.append(np.nan)
        elif ci_method == "sem":
            se = np.sqrt(rate * (1 - rate) / n_g)
            low.append(rate - z * se)
            high.append(rate + z * se)
        else:  # wilson
            _, lo, hi = _wilson_interval(k, n_g, z)
            low.append(float(lo))
            high.append(float(hi))
    return {"point": np.array(point), "low": np.array(low), "high": np.array(high)}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def edp(
    df: pd.DataFrame,
    features: Sequence[str],
    yname: str,
    n: int = 4,
    *,
    kind: str = "auto",
    binning: str = "quantile",
    ci: Optional[str] = "auto",
    max_categories: int = 10,
    digits: int = 2,
    figsize: Tuple[float, float] = (8, 6),
    show_bincount: bool = True,
    ylim_origin: bool = True,
    even_spaced_ticks: bool = False,
    show_baseline: bool = True,
    writefolder: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Empirical Dependence Plot.

    For every feature, plot how the target ``yname`` behaves across grouped
    values of that feature:

    * **continuous target** -> the mean of the target per bin (with a
      standard-error band);
    * **binary target** -> the positive-class rate per bin (with a Wilson
      confidence band);
    * **multiclass target** -> one line per class, each showing the class rate
      per bin (with a Wilson band).

    Parameters
    ----------
    df : DataFrame
        Source data.
    features : list of str
        Feature columns to plot (use ``[feature]`` for a single one).
    yname : str
        Target column.
    n : int, default 4
        Number of bins for continuous features (upper bound; shrinks only if
        repeated values collapse the quantile edges).
    kind : {"auto", "continuous", "categorical"}, default "auto"
        How to treat each *feature*. "auto" decides per feature from its dtype
        and cardinality (see ``max_categories``).
    binning : {"quantile", "uniform"}, default "quantile"
        Binning strategy for continuous features.
    ci : {"auto", "wilson", "sem", None}, default "auto"
        Confidence band. "auto" uses Wilson for classification targets and the
        standard error of the mean for continuous ones. ``None`` disables it.
    max_categories : int, default 10
        A numeric feature/target with at most this many distinct values is
        treated as categorical.
    digits : int, default 2
        Decimal places in continuous bin labels.
    figsize : tuple, default (8, 6)
    show_bincount : bool, default True
        Draw the per-bin sample count on a secondary axis.
    ylim_origin : bool, default True
        Start the primary y-axis at 0.
    even_spaced_ticks : bool, default False
        Place continuous bins at their real midpoints instead of evenly.
    show_baseline : bool, default True
        Draw the global mean/rate of the target as a horizontal reference.
    writefolder : str, optional
        If given, save each figure there as PNG instead of showing it.
    show : bool, default True
        Call ``plt.show()`` when not writing to disk.
    close : bool, optional
        Close each figure after handling it. Defaults to True when
        ``writefolder`` is set (avoids leaking figures across many features),
        False otherwise.

    Returns
    -------
    list of dict
        One entry per plotted feature with keys ``feature``, ``fig``, ``ax``
        (primary axis) and ``path`` (when saved to disk).
    """
    if not isinstance(yname, str):
        raise TypeError("yname must be a string (a single column name).")
    if yname not in df.columns:
        raise ValueError(f"yname column {yname!r} is not in the dataframe.")
    if isinstance(features, str):
        raise TypeError("features must be a list; use [feature] for a single feature.")
    if kind not in ("auto", "continuous", "categorical"):
        raise ValueError("kind must be 'auto', 'continuous' or 'categorical'.")
    if ci not in ("auto", "wilson", "sem", None):
        raise ValueError("ci must be 'auto', 'wilson', 'sem' or None.")

    z = _Z_95
    if close is None:
        close = writefolder is not None

    target_kind = _target_kind(df[yname], max_categories)

    # Determine the series (one per target class, or a single regression line).
    if target_kind == "binary":
        classes = list(pd.unique(df[yname].dropna()))
        try:
            positive = max(classes)
        except TypeError:
            positive = sorted(classes, key=str)[-1]
        series_classes = [positive]
    elif target_kind == "multiclass":
        classes = list(pd.unique(df[yname].dropna()))
        try:
            series_classes = sorted(classes)
        except TypeError:
            series_classes = sorted(classes, key=str)
    else:
        series_classes = None  # regression

    results: List[Dict[str, Any]] = []

    for feature in features:
        if feature == yname:
            continue
        if feature not in df.columns:
            print(f"[SKIPPED] feature {feature!r} not in dataframe.")
            continue

        df_temp = df[[feature, yname]].dropna()
        if df_temp.empty:
            print(f"[SKIPPED] feature {feature!r} has no non-null rows.")
            continue

        x = df_temp[feature].to_numpy()
        y = df_temp[yname].to_numpy()

        if np.all(x == x[0]):
            print(f"[SKIPPED] feature {feature!r} has all identical values.")
            continue

        # ---- Decide feature treatment and build groups -------------------- #
        if kind == "continuous":
            feat_categorical = False
        elif kind == "categorical":
            feat_categorical = True
        else:
            feat_categorical = _is_categorical_feature(df_temp[feature], max_categories)

        if feat_categorical:
            groups = _make_categorical_groups(x)
        else:
            groups = _make_continuous_groups(
                x, n, binning, digits, even_spaced_ticks
            )

        # Drop empty groups so every series stays aligned with the bin counts.
        groups = [g for g in groups if int(g["mask"].sum()) > 0]
        if len(groups) < 2:
            print(f"[SKIPPED] feature {feature!r}: not enough populated bins.")
            continue

        counts = np.array([int(g["mask"].sum()) for g in groups])
        xpos = [g["x"] for g in groups]
        xlabels = [g["label"] for g in groups]
        show_ci = ci is not None

        # ---- Compute series ---------------------------------------------- #
        plot_series: List[Dict[str, Any]] = []
        if series_classes is None:  # regression
            s = _regression_series(y, groups, show_ci, z)
            s["name"] = f"mean {yname}"
            s["baseline"] = float(np.mean(y))
            plot_series.append(s)
            ylabel = f"mean {yname}"
        else:
            ci_method = "sem" if ci == "sem" else "wilson"
            for cls in series_classes:
                s = _class_series(y, cls, groups, show_ci, ci_method, z)
                if target_kind == "binary":
                    s["name"] = f"P({yname}={cls})"
                else:
                    s["name"] = f"{yname}={cls}"
                s["baseline"] = float(np.mean(y == cls))
                plot_series.append(s)
            ylabel = f"P({yname})" if target_kind != "binary" else plot_series[0]["name"]

        # ---- Plot --------------------------------------------------------- #
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel(str(feature))
        ax1.set_ylabel(ylabel)

        for s in plot_series:
            line, = ax1.plot(xpos, s["point"], "o-", label=s["name"])
            color = line.get_color()
            if show_ci:
                ax1.fill_between(
                    xpos, s["low"], s["high"], alpha=0.15, color=color
                )
            if show_baseline:
                ax1.axhline(
                    s["baseline"], ls=":", lw=1, alpha=0.5, color=color
                )

        # y limits (ignoring NaN band ends).
        all_pts = np.concatenate([s["point"] for s in plot_series])
        all_hi = np.concatenate([s["high"] for s in plot_series])
        all_lo = np.concatenate([s["low"] for s in plot_series])
        top = np.nanmax(np.concatenate([all_pts, all_hi]))
        bot = np.nanmin(np.concatenate([all_pts, all_lo]))
        if series_classes is not None:  # classification -> rates in [0, 1]
            ymin = 0 if ylim_origin else bot * 0.95
            ax1.set_ylim([ymin, min(1.02, top * 1.05)])
        else:
            ymin = 0 if ylim_origin and bot >= 0 else bot * 0.9
            ax1.set_ylim([ymin, top * 1.05])

        ax1.set_xticks(xpos)
        ax1.set_xticklabels(xlabels, rotation=35, ha="right")

        ax2 = None
        if show_bincount:
            color = "tab:red"
            ax2 = ax1.twinx()
            ax2.plot(xpos, counts, "o--", color=color, label="bin count")
            ax2.set_ylim([0, counts.max() * 1.2])
            ax2.set_ylabel("bin count", color=color)
            ax2.tick_params(axis="y", labelcolor=color)

        # Combined legend across both axes.
        handles, labels = ax1.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            ax1.legend(handles, labels, loc="best", fontsize=8)

        fig.tight_layout()

        entry: Dict[str, Any] = {"feature": feature, "fig": fig, "ax": ax1}
        if writefolder:
            os.makedirs(writefolder, exist_ok=True)
            fname = "edp_feature_{}_y_{}.png".format(
                _sanitize_filename(feature), _sanitize_filename(yname)
            )
            path = os.path.join(writefolder, fname)
            fig.savefig(path)
            entry["path"] = path
        elif show:
            plt.show()

        if close:
            plt.close(fig)

        results.append(entry)

    return results
