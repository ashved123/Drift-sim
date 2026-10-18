"""Shared look for Drift Studio's charts: validated palette, matplotlib style, small helpers."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

# Validated categorical palette (light mode, fixed order) + chart chrome.
S1, S2, S3, S4, S5, S6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SERIES = [S1, S2, S3, S4, S5, S6]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, ROAD, SURFACE = "#e1e0d9", "#c3c2b7", "#d6d5ce", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

matplotlib.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.grid": True, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2, "legend.frameon": False,
    "legend.fontsize": 8, "figure.facecolor": SURFACE,
})


def rolling(y, w: int) -> np.ndarray:
    """Trailing mean over up to `w` points (shorter at the start)."""
    y = np.asarray(y, float)
    if len(y) == 0:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    idx = np.arange(1, len(y) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def fmt_duration(s) -> str:
    if s is None:
        return "–"
    h, m = int(s // 3600), int(s % 3600 // 60)
    return f"{h}h {m}m" if h else f"{m}m {int(s % 60)}s"


def clear_map_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_road(ax, track: dict, pad: float = 25) -> None:
    """Track outline as a band + start marker (a map, not to scale across)."""
    x, y = np.array(track["x"]), np.array(track["y"])
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), color=ROAD, lw=6, solid_capstyle="round", zorder=1)
    ax.plot([x[0]], [y[0]], "s", color=INK, ms=5, zorder=3)
    ax.annotate("Start", (x[0], y[0]), xytext=(6, 6), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
