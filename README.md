# edp-tool — Empirical Dependence Plot (EDP)

[![PyPI](https://img.shields.io/pypi/v/edp-tool.svg)](https://pypi.org/project/edp-tool/)
[![Python versions](https://img.shields.io/pypi/pyversions/edp-tool.svg)](https://pypi.org/project/edp-tool/)
[![License: MIT](https://img.shields.io/pypi/l/edp-tool.svg)](LICENSE)

A small exploratory-data-analysis tool that shows how a target variable behaves
across grouped values of a feature. It is designed with **categorical / binary
targets** in mind (it plots the observed class rate), but it also works for
continuous targets (it plots the mean).

![example](https://user-images.githubusercontent.com/9744889/173620131-b056f517-e285-4309-909b-b02b56df6a7e.png)

## EDP vs. PDP — why the name changed

This tool was originally called a *Partial Dependence Plot (PDP)*, but that name
is technically inaccurate. A true PDP requires a **fitted model** and
**marginalizes** its predictions over the other features.

The EDP uses **no model** and does **no marginalization** — it plots the
*observed* conditional mean of the target given a single feature, straight from
the data. That makes it closely related to the **M-plot** (marginal plot) from
the ALE literature (Apley & Zhu, 2020). Calling it *Empirical* Dependence Plot
reflects exactly what it computes.

The old `pdp` function still works as a deprecated alias (see below).

## Install

```bash
pip install edp-tool
```

In a notebook / Colab cell:

```python
!pip install -q edp-tool
from edp_tool import edp
```

The distribution is published on PyPI as
[`edp-tool`](https://pypi.org/project/edp-tool/); the import name is `edp_tool`
(`pip install edp_tool` also works — pip normalizes the two spellings).

Alternatively, just copy `edp_tool.py` next to your notebook:

```python
import os
url = 'https://raw.githubusercontent.com/attilalr/edp-tool/main/edp_tool.py'
if not os.path.isfile('edp_tool.py'):
    !wget -q {url}
from edp_tool import edp
```

Requirements: `numpy`, `pandas`, `matplotlib` (installed automatically by pip;
see also `requirements.txt`).

## Usage

```python
from edp_tool import edp

# Binary target -> positive-class rate per bin, with a Wilson CI band
edp(df, ['age', 'income'], 'converted')

# Multiclass target -> one line per class
edp(df, ['petal length (cm)'], 'species')

# Continuous target -> mean per bin, with a standard-error band
edp(df, ['age'], 'price')

# Save figures instead of showing them
edp(df, features, 'target', writefolder='figs')
```

`edp()` returns a list of dicts (`feature`, `fig`, `ax`, and `path` when saved),
so you can post-process or embed the figures.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n` | `4` | Number of bins for continuous features (upper bound) |
| `kind` | `"auto"` | `"auto"` / `"continuous"` / `"categorical"` treatment per feature |
| `binning` | `"quantile"` | `"quantile"` or `"uniform"` bin edges |
| `ci` | `"auto"` | `"wilson"`, `"sem"`, `"auto"`, or `None` (Wilson for classes, SEM for regression) |
| `max_categories` | `10` | Numeric columns with ≤ this many distinct values are treated as categorical |
| `show_bincount` | `True` | Draw per-bin sample count on a secondary axis |
| `show_baseline` | `True` | Draw the global target mean/rate as a reference line |
| `ylim_origin` | `True` | Start the y-axis at 0 |
| `even_spaced_ticks` | `False` | Place continuous bins at real midpoints |
| `writefolder` | `None` | Save PNGs to this folder instead of showing |

Multiclass targets are handled natively — no manual one-hot encoding needed.

## What's new in this version

- Renamed to **EDP** (`edp_tool.edp`); `pdp_tool.pdp` kept as a deprecated alias.
- Fixed the dead categorical branch (feature type is now detected correctly).
- Fixed `n` leaking across features and the maximum value being dropped from
  the last bin.
- Native multiclass support, **Wilson** confidence intervals for rates, optional
  baseline line and uniform binning.
- Figures are returned and properly closed; validation raises real exceptions.

## Development

```bash
git clone https://github.com/attilalr/edp-tool
cd edp-tool
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
