"""Deprecated module. Use :mod:`edp_tool` instead.

The tool was renamed from PDP (Partial Dependence Plot) to
EDP (Empirical Dependence Plot), because it computes an *empirical* conditional
mean directly from the data -- it does not fit a model or marginalize over
other features, which is what a true partial dependence plot does.

``pdp`` remains as a thin alias for :func:`edp_tool.edp` and emits a
``DeprecationWarning``. It will be removed in a future release.
"""

from __future__ import annotations

import warnings
from functools import wraps

from edp_tool import edp

__all__ = ["pdp", "edp"]


@wraps(edp)
def pdp(*args, **kwargs):
    warnings.warn(
        "pdp() is deprecated and will be removed; import and use "
        "edp() from edp_tool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return edp(*args, **kwargs)
