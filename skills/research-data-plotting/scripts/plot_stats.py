"""Pure-matplotlib statistical plotting helpers (theme_bw + npg style).

Uses NPG colors and markers from style.py. Never changes the global theme;
callers should call style.use(...) themselves. Every function that draws error
bars/bands annotates the error type and n, and returns a dict of computed stats.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import style
from plot_layout import subplots as layout_subplots

# Local aliases — do not mutate style module state
NPG = list(style.NPG)
MARKERS = list(style.MARKERS)


import matplotlib.patheffects as _pe
from matplotlib.colors import to_rgb as _to_rgb

FILL_ALPHA = 0.72  # box and violin share one fill opacity


def _edge(i: int, k: float = 0.45) -> tuple:
    """Group colour darkened toward ink (#333333) so hollow markers stay visible on deep fills."""
    c = np.array(_to_rgb(_color(i)))
    ink = np.array(_to_rgb("#333333"))
    return tuple((1 - k) * c + k * ink)

def _as_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def _group_order(series: pd.Series, order: Optional[Sequence] = None) -> List:
    if order is not None:
        return list(order)
    # Preserve appearance order for categorical-like data
    return list(pd.unique(series.dropna()))


def _color(i: int) -> str:
    return NPG[i % len(NPG)]


def _marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


def _annotate_error(ax: Axes, text: str, loc: str = "below") -> Any:
    """Caption stating error type / n — placed below the axes (never over data).

    Default *loc* ``'below'`` puts the note under the xlabel at axes-fraction
    ``y < 0``. Legacy in-axes corners (``lower right`` etc.) remain available
    but are discouraged because they collide with strip/scatter points.
    """
    positions = {
        "below": ("center", "top", 0.5, -0.22),
        "below left": ("left", "top", 0.0, -0.22),
        "below right": ("right", "top", 1.0, -0.22),
        "lower right": ("right", "bottom", 0.98, 0.02),
        "lower left": ("left", "bottom", 0.02, 0.02),
        "upper right": ("right", "top", 0.98, 0.98),
        "upper left": ("left", "top", 0.02, 0.98),
    }
    ha, va, x, y = positions.get(loc, positions["below"])
    fs = mpl.rcParams["font.size"] * 0.85
    if y < 0:
        # Anchor under the x-axis label's actual bbox (x in axes fraction, y from the label),
        # so the caption never collides with multi-line tick labels or large talk fonts.
        t = ax.annotate(
            text, xy=(x, 0), xycoords=(ax.transAxes, ax.xaxis.label),
            xytext=(0, -0.4 * fs), textcoords="offset points",
            ha=ha, va="top", fontsize=fs, color="#4D4D4D", annotation_clip=False, zorder=5,
        )
        t.set_in_layout(True)
    else:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
                    color="#4D4D4D", clip_on=False, zorder=5)
    return t


def _jitter(n: int, width: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-width, width, size=n)


def box_strip(
    ax: Axes,
    data,
    x: str,
    y: str,
    order: Optional[Sequence] = None,
    jitter: float = 0.15,
    seed: int = 0,
    fill_alpha: float = FILL_ALPHA,
    **kw,
) -> Dict[str, Any]:
    """Box plot (fill alpha FILL_ALPHA=0.72, same as violin) + jittered strip points; n in tick labels."""
    df = _as_dataframe(data)
    cats = _group_order(df[x], order)
    rng = np.random.default_rng(seed)
    stats: Dict[str, Any] = {"n": {}, "median": {}, "q1": {}, "q3": {}}
    positions = np.arange(len(cats))
    box_data = []
    tick_labels = []
    for i, cat in enumerate(cats):
        vals = df.loc[df[x] == cat, y].dropna().to_numpy(dtype=float)
        box_data.append(vals)
        n = int(vals.size)
        stats["n"][cat] = n
        if n:
            stats["median"][cat] = float(np.median(vals))
            stats["q1"][cat] = float(np.percentile(vals, 25))
            stats["q3"][cat] = float(np.percentile(vals, 75))
        else:
            stats["median"][cat] = stats["q1"][cat] = stats["q3"][cat] = np.nan
        tick_labels.append(f"{cat}\n(n={n})")

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.6,
                         path_effects=[_pe.Stroke(linewidth=2.8, foreground="#333333"), _pe.Normal()]),
        whiskerprops=dict(color="#333333"),
        capprops=dict(color="#333333"),
        boxprops=dict(linewidth=0.8, edgecolor="#333333"),
        **{k: v for k, v in kw.items() if k in ("whis", "showmeans", "meanline")},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_color(i))
        patch.set_alpha(fill_alpha)

    for i, (cat, vals) in enumerate(zip(cats, box_data)):
        if vals.size == 0:
            continue
        jx = i + _jitter(vals.size, jitter, rng)
        ax.scatter(
            jx,
            vals,
            s=18,
            marker=_marker(i),
            facecolors="none",
            edgecolors=_edge(i),
            linewidths=0.9,
            zorder=3,
            label=str(cat) if i == 0 else None,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    total_n = int(sum(stats["n"].values()))
    ann = f"box: median/IQR; whiskers: 1.5·IQR; n={total_n}"
    _annotate_error(ax, ann)
    stats["annotation"] = ann
    stats["categories"] = cats
    return stats


def _violin_kde_support(
    vals: np.ndarray,
    cut: float,
    points: int = 256,
    bw_method="scott",
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:
    """KDE density along y with support extended by *cut* bandwidths.

    Returns (y_grid, density, bw, (y_lo, y_hi)).
    """
    from scipy.stats import gaussian_kde

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    data_min, data_max = float(vals.min()), float(vals.max())
    if vals.size < 2 or np.isclose(data_min, data_max):
        # Degenerate: thin stick at the single value
        y0 = data_min
        return np.array([y0, y0]), np.array([1.0, 1.0]), 0.0, (y0, y0)

    if bw_method == "nrd0":
        # R's bw.nrd0 (Silverman's rule of thumb), same default as ggplot2 geom_violin
        sd = float(np.std(vals, ddof=1))
        iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
        lo = min(sd, iqr / 1.34) if iqr > 0 else sd
        h = 0.9 * (lo if lo > 0 else (sd or abs(vals[0]) or 1.0)) * vals.size ** -0.2
        kde = gaussian_kde(vals, bw_method=h / sd if sd > 0 else None)
    else:
        kde = gaussian_kde(vals, bw_method=bw_method)
    bw = float(np.sqrt(kde.covariance[0, 0]))
    if cut is None:
        cut = 0.5
    cut = float(cut)
    if cut <= 0:
        y_lo, y_hi = data_min, data_max
    else:
        y_lo, y_hi = data_min - cut * bw, data_max + cut * bw
    y_grid = np.linspace(y_lo, y_hi, points)
    dens = np.asarray(kde(y_grid), dtype=float)
    dens = np.clip(dens, 0.0, None)
    return y_grid, dens, bw, (y_lo, y_hi)


def violin(
    ax: Axes,
    data,
    x: str,
    y: str,
    order: Optional[Sequence] = None,
    inner: str = "box",
    seed: int = 0,
    cut: Optional[float] = None,
    width: Optional[float] = None,
    look: str = "gg",
    scale: Optional[str] = None,
    points: bool = False,
    point_shape: str = "circle",
    point_edge: str = "group",
    point_alpha: float = 0.8,
    point_lw: Optional[float] = None,
    jitter: Optional[float] = None,
    **kw,
) -> Dict[str, Any]:
    """Violin plot with n annotation; inner is 'box'|'quartile'|'points'.

    look='gg' (default) follows the ggplot2 publication look: R bw.nrd0
    bandwidth, tails extended 3 bandwidths (geom_violin trim=FALSE), equal-area
    scaling, width 0.9, thin grey20 outline, slim white inner boxplot (median
    line, 1.5·IQR whiskers, no outliers). look='classic' keeps the earlier style
    (Scott bandwidth, cut 0.5, width-normalised, dark IQR bar + white median dot).
    *cut* = bandwidths of tail extension past data min/max; ``cut=0`` clips.
    points=True overlays jittered hollow points. Point defaults match ggplot2
    geom_jitter(width=0.08, shape=21, size=1.5, stroke=0.5, alpha=0.8): circles,
    edge = group colour, uniform jitter +-0.08. point_shape='group' uses per-group
    markers; point_edge='dark' darkens the edge (float 0-1 = darkening amount).
    """
    if inner not in ("box", "quartile", "points"):
        raise ValueError("inner must be 'box'|'quartile'|'points'")
    if look not in ("gg", "classic"):
        raise ValueError("look must be 'gg'|'classic'")
    gg = look == "gg"
    if cut is None:
        cut = 3.0 if gg else 0.5
    if width is None:
        width = 0.9 if gg else 0.7
    if scale is None:
        scale = "area" if gg else "width"
    bw_method = "nrd0" if gg else "scott"
    edge_c, edge_lw = ("#333333", 0.5) if gg else ("#333333", 0.8)

    df = _as_dataframe(data)
    cats = _group_order(df[x], order)
    rng = np.random.default_rng(seed)
    stats: Dict[str, Any] = {
        "n": {}, "median": {}, "q1": {}, "q3": {}, "cut": float(cut),
        "support": {}, "data_range": {}, "bw": {}, "look": look, "scale": scale,
    }
    positions = list(range(len(cats)))
    datasets = []
    for cat in cats:
        vals = df.loc[df[x] == cat, y].dropna().to_numpy(dtype=float)
        datasets.append(vals)
        n = int(vals.size)
        stats["n"][cat] = n
        if n:
            stats["median"][cat] = float(np.median(vals))
            stats["q1"][cat] = float(np.percentile(vals, 25))
            stats["q3"][cat] = float(np.percentile(vals, 75))
            stats["data_range"][cat] = (float(vals.min()), float(vals.max()))
        else:
            stats["median"][cat] = stats["q1"][cat] = stats["q3"][cat] = np.nan
            stats["data_range"][cat] = (np.nan, np.nan)

    kdes = {}
    for i, (cat, vals) in enumerate(zip(cats, datasets)):
        if vals.size:
            kdes[i] = _violin_kde_support(vals, cut=cut, bw_method=bw_method)
    # area scaling: densities share one scale (largest peak -> full width)
    gmax = max((float(np.max(k[1])) for k in kdes.values()), default=0.0)
    half_w = 0.5 * width
    for i, (cat, vals) in enumerate(zip(cats, datasets)):
        if i not in kdes:
            continue
        y_grid, dens, bw, (y_lo, y_hi) = kdes[i]
        stats["bw"][cat] = bw
        stats["support"][cat] = (float(y_lo), float(y_hi))
        peak = gmax if scale == "area" else float(np.max(dens))
        if peak <= 0:
            continue
        xs = dens / peak * half_w
        xp = np.concatenate([positions[i] - xs, (positions[i] + xs)[::-1]])
        yp = np.concatenate([y_grid, y_grid[::-1]])
        ax.fill(xp, yp, facecolor=_color(i), edgecolor=edge_c, linewidth=edge_lw,
                alpha=FILL_ALPHA, zorder=2, **{k: v for k, v in kw.items() if k in ("label",)})
        # outline drawn opaque on top so alpha does not wash it out
        ax.plot(xp, yp, color=edge_c, lw=edge_lw, zorder=2.1)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        if inner == "box" and gg:
            iqr = q3 - q1
            lo_w = float(vals[vals >= q1 - 1.5 * iqr].min())
            hi_w = float(vals[vals <= q3 + 1.5 * iqr].max())
            bw_box = 0.1
            ax.vlines(i, lo_w, q1, color=edge_c, lw=edge_lw + 0.1, zorder=3)
            ax.vlines(i, q3, hi_w, color=edge_c, lw=edge_lw + 0.1, zorder=3)
            ax.add_patch(mpl.patches.Rectangle((i - bw_box / 2, q1), bw_box, q3 - q1, facecolor="white",
                                               edgecolor=edge_c, lw=edge_lw + 0.1, zorder=4))
            ax.hlines(med, i - bw_box / 2, i + bw_box / 2, color=edge_c, lw=1.2, zorder=5)
        elif inner == "box":
            ax.vlines(i, q1, q3, color="#333333", lw=3, zorder=3,
                      path_effects=[_pe.Stroke(linewidth=4.6, foreground="white"), _pe.Normal()])
            ax.scatter([i], [med], color="white", edgecolors="#333333", s=12, zorder=4, linewidths=0.6)
        elif inner == "quartile":
            ax.hlines([q1, med, q3], i - 0.15, i + 0.15, colors="white", lw=1.2, zorder=3,
                      path_effects=[_pe.Stroke(linewidth=2.4, foreground="#333333"), _pe.Normal()])
        if inner == "points" or points:
            jw = jitter if jitter is not None else (0.08 if gg else 0.12)
            jx = i + _jitter(vals.size, jw, rng)
            # gg: ggplot2 geom_jitter default size 1.5, stroke 0.5 (shape 21): grid draws the circle at
            # 0.75*(size*.pt + stroke*.stroke/2) = 3.9 pt diameter -> s ~= 15.3 pt^2; stroke = 0.71 pt.
            if point_edge == "group":
                ec = _color(i)
            elif point_edge == "dark":
                ec = _edge(i, 0.25)
            else:
                ec = _edge(i, float(point_edge))
            mk = "o" if point_shape == "circle" else _marker(i)
            lw = point_lw if point_lw is not None else (0.71 if gg else 0.6)
            ax.scatter(jx, vals, s=15.3 if gg else 12, marker=mk, facecolors="none",
                       edgecolors=ec, linewidths=lw, alpha=point_alpha, zorder=6 if gg else 3.5)

    ax.set_xlim(-0.6, len(cats) - 0.4)
    total_n = int(sum(stats["n"].values()))
    tick_labels = [f"{c}\n(n={stats['n'][c]})" for c in cats]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if gg and inner == "box":
        ann = f"violin: KDE (bw.nrd0, tails {cut:g}·bw); box: median/IQR, whiskers 1.5·IQR; n={total_n}"
    else:
        ann = f"violin (inner={inner}); n={total_n}"
    _annotate_error(ax, ann)
    stats["annotation"] = ann
    stats["categories"] = cats
    return stats


def _poly_design(x: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack([x ** k for k in range(order + 1)])


def _ols_fit(x: np.ndarray, y: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (coef, yhat_on_sorted_grid_unused, sigma2). coef lowest degree first."""
    A = _poly_design(x, order)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    dof = max(len(y) - len(coef), 1)
    sigma2 = float(np.sum((y - yhat) ** 2) / dof)
    return coef, yhat, sigma2


def _predict_band(
    x_line: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    ci: float,
    n_boot: Union[int, str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return y_fit, lo, hi, coef."""
    coef, _, sigma2 = _ols_fit(x, y, order)
    A_line = _poly_design(x_line, order)
    y_fit = A_line @ coef
    alpha = 1.0 - ci / 100.0

    if n_boot == "analytic" or n_boot == 0:
        # Analytic CI from OLS variance (approx, t-based)
        from scipy import stats as sps

        A = _poly_design(x, order)
        # cov(coef) ≈ sigma2 * (A'A)^{-1}
        ata = A.T @ A
        try:
            cov = sigma2 * np.linalg.inv(ata)
        except np.linalg.LinAlgError:
            cov = sigma2 * np.linalg.pinv(ata)
        se = np.sqrt(np.sum(A_line @ cov * A_line, axis=1))
        dof = max(len(y) - (order + 1), 1)
        tcrit = float(sps.t.ppf(1 - alpha / 2, dof))
        lo = y_fit - tcrit * se
        hi = y_fit + tcrit * se
        return y_fit, lo, hi, coef

    n_boot = int(n_boot)
    rng = np.random.default_rng(seed)
    preds = np.empty((n_boot, x_line.size), dtype=float)
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        try:
            cb, _, _ = _ols_fit(xb, yb, order)
            preds[b] = _poly_design(x_line, order) @ cb
        except Exception:
            preds[b] = np.nan
    lo = np.nanpercentile(preds, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(preds, 100 * (1 - alpha / 2), axis=0)
    return y_fit, lo, hi, coef


def regplot(
    ax: Axes,
    data,
    x: str,
    y: str,
    ci: float = 95,
    n_boot: Union[int, str] = 1000,
    order: int = 1,
    seed: int = 0,
    **kw,
) -> Dict[str, Any]:
    """Scatter (open markers) + polynomial fit + CI band; annotated."""
    df = _as_dataframe(data)
    sub = df[[x, y]].dropna()
    xv = sub[x].to_numpy(dtype=float)
    yv = sub[y].to_numpy(dtype=float)
    n = int(len(xv))
    color = _color(0)

    ax.scatter(
        xv,
        yv,
        s=28,
        marker=MARKERS[0],
        facecolors="none",
        edgecolors=color,
        linewidths=0.9,
        zorder=3,
        label="data",
        **{k: v for k, v in kw.items() if k in ("alpha",)},
    )

    if n >= order + 1:
        x_line = np.linspace(xv.min(), xv.max(), 200)
        y_fit, lo, hi, coef = _predict_band(x_line, xv, yv, order, ci, n_boot, seed)
        ax.plot(x_line, y_fit, color=color, lw=1.4, zorder=2, label="OLS fit")
        ax.fill_between(x_line, lo, hi, color=color, alpha=0.22, zorder=1, linewidth=0, label=f"{ci:g}% CI")
    else:
        coef = np.full(order + 1, np.nan)
        x_line = y_fit = lo = hi = np.array([])

    method = "analytic" if n_boot == "analytic" or n_boot == 0 else f"bootstrap ({n_boot})"
    ann = f"line: OLS fit; band: {ci:g}% CI; n={n}"
    _annotate_error(ax, ann)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return {
        "n": n,
        "order": order,
        "ci": ci,
        "n_boot": n_boot,
        "method": method,
        "coef": coef,
        "annotation": ann,
        "x_line": x_line,
        "y_fit": y_fit,
        "lo": lo,
        "hi": hi,
    }


def _err_band(vals: np.ndarray, err: str, ci: float) -> Tuple[float, float, float]:
    """Return mean, lower_delta, upper_delta for a 1d sample."""
    vals = vals.astype(float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(vals))
    if n == 1:
        return m, 0.0, 0.0
    sd = float(np.std(vals, ddof=1))
    se = sd / np.sqrt(n)
    if err == "sd":
        return m, sd, sd
    if err == "se":
        return m, se, se
    if err == "ci":
        from scipy import stats as sps

        tcrit = float(sps.t.ppf(1 - (1 - ci / 100.0) / 2, n - 1))
        half = tcrit * se
        return m, half, half
    raise ValueError("err must be 'sd'|'se'|'ci'")


def lineplot(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str] = None,
    err: str = "sd",
    ci: float = 95,
    **kw,
) -> Dict[str, Any]:
    """Mean line + error band per group; annotates error type and n."""
    df = _as_dataframe(data)
    if hue is None:
        groups = [(None, df)]
    else:
        cats = _group_order(df[hue])
        groups = [(c, df[df[hue] == c]) for c in cats]

    stats: Dict[str, Any] = {"groups": {}, "err": err, "ci": ci}
    n_total = 0
    for i, (g, gdf) in enumerate(groups):
        grouped = gdf.groupby(x, sort=True)[y]
        xs, means, los, his, ns = [], [], [], [], []
        for xv, series in grouped:
            arr = series.to_numpy(dtype=float)
            m, dlo, dhi = _err_band(arr, err, ci)
            xs.append(xv)
            means.append(m)
            los.append(m - dlo)
            his.append(m + dhi)
            ns.append(int(np.sum(~np.isnan(arr))))
        xs = np.asarray(xs)
        means = np.asarray(means)
        los = np.asarray(los)
        his = np.asarray(his)
        color = _color(i)
        label = str(g) if g is not None else y
        ax.plot(xs, means, color=color, marker=_marker(i), mfc="none", mec=color, mew=0.9, lw=1.3, label=label)
        ax.fill_between(xs, los, his, color=color, alpha=0.22, linewidth=0)
        n_g = int(sum(ns))
        n_total += n_g
        key = g if g is not None else "_all"
        stats["groups"][key] = {"n": n_g, "x": xs, "mean": means, "lo": los, "hi": his}

    err_label = {"sd": "SD", "se": "SE", "ci": f"{ci:g}% CI"}[err]
    if hue is None:
        ann = f"band: {err_label}; n={n_total}"
    else:
        parts = [f"{k}: n={v['n']}" for k, v in stats["groups"].items()]
        ann = f"band: {err_label}; " + ", ".join(parts)
    _annotate_error(ax, ann)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    stats["n"] = n_total
    stats["annotation"] = ann
    return stats


def scatter(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    **kw,
) -> Dict[str, Any]:
    """Scatter with optional hue (color) and style (marker) groupings."""
    df = _as_dataframe(data)
    stats: Dict[str, Any] = {"n": int(len(df)), "groups": {}}

    if hue is None and style is None:
        ax.scatter(
            df[x],
            df[y],
            s=28,
            marker=MARKERS[0],
            facecolors="none",
            edgecolors=_color(0),
            linewidths=0.9,
            **kw,
        )
        stats["groups"]["_all"] = {"n": int(len(df))}
    else:
        # Build unique combo keys
        if hue is not None and style is not None:
            # color by hue, marker by style
            hues = _group_order(df[hue])
            styles = _group_order(df[style])
            hue_idx = {h: i for i, h in enumerate(hues)}
            style_idx = {s: i for i, s in enumerate(styles)}
            for (h, s), gdf in df.groupby([hue, style], sort=False):
                i, j = hue_idx[h], style_idx[s]
                ax.scatter(
                    gdf[x],
                    gdf[y],
                    s=28,
                    marker=_marker(j),
                    facecolors="none",
                    edgecolors=_color(i),
                    linewidths=0.9,
                    label=f"{h}, {s}",
                    **kw,
                )
                stats["groups"][(h, s)] = {"n": int(len(gdf))}
        elif hue is not None:
            for i, (h, gdf) in enumerate(df.groupby(hue, sort=False)):
                # groupby may yield name as scalar
                ax.scatter(
                    gdf[x],
                    gdf[y],
                    s=28,
                    marker=_marker(i),
                    facecolors="none",
                    edgecolors=_color(i),
                    linewidths=0.9,
                    label=str(h),
                    **kw,
                )
                stats["groups"][h] = {"n": int(len(gdf))}
        else:
            for i, (s, gdf) in enumerate(df.groupby(style, sort=False)):
                ax.scatter(
                    gdf[x],
                    gdf[y],
                    s=28,
                    marker=_marker(i),
                    facecolors="none",
                    edgecolors=_color(i),
                    linewidths=0.9,
                    label=str(s),
                    **kw,
                )
                stats["groups"][s] = {"n": int(len(gdf))}

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return stats


def facet(
    data,
    col: Optional[str] = None,
    row: Optional[str] = None,
    func: Optional[Callable] = None,
    sharex: Union[bool, str] = True,
    sharey: Union[bool, str] = True,
    refwidth_mm: float = 55.0,
    panel_labels: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kw,
) -> Tuple[Figure, Any]:
    """Faceted figure using plot_layout.subplots with shared axes.

    *func(ax, data_panel, **kw)* is called for each facet panel.

    Inner tick labels and per-axes x/y labels are hidden when axes are
    shared; use *xlabel* / *ylabel* (or ``plot_layout.set_shared_labels``)
    for a single figure-level supxlabel / supylabel. Panel labels fold into
    facet titles on the same baseline.
    """
    from plot_layout import label_outer_shared, set_shared_labels, sync_panel_labels

    if func is None:
        raise ValueError("func is required")
    df = _as_dataframe(data)
    col_levels = _group_order(df[col]) if col is not None else [None]
    row_levels = _group_order(df[row]) if row is not None else [None]
    nrows, ncols = len(row_levels), len(col_levels)

    fig, axes = layout_subplots(
        nrows,
        ncols,
        refwidth_mm=refwidth_mm,
        sharex=sharex,
        sharey=sharey,
        panel_labels=panel_labels,
    )
    axes_arr = np.atleast_2d(axes)
    if axes_arr.shape != (nrows, ncols):
        axes_arr = np.array(_flatten_axes(axes)).reshape(nrows, ncols)

    for i, rv in enumerate(row_levels):
        for j, cv in enumerate(col_levels):
            ax = axes_arr[i, j]
            panel = df
            if row is not None:
                panel = panel[panel[row] == rv]
            if col is not None:
                panel = panel[panel[col] == cv]
            func(ax, panel, **kw)
            title_parts = []
            if row is not None:
                title_parts.append(f"{row}={rv}")
            if col is not None:
                title_parts.append(f"{col}={cv}")
            if title_parts:
                ax.set_title(", ".join(title_parts), fontsize=mpl.rcParams["font.size"])

    # Re-hide inner ticks/labels after panel funcs (e.g. scatter) set them
    if sharex or sharey:
        label_outer_shared(axes_arr)
    # Drop all per-axes labels when using figure-level shared labels
    for ax in axes_arr.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
    if xlabel is not None or ylabel is not None:
        set_shared_labels(fig, xlabel=xlabel, ylabel=ylabel)
    sync_panel_labels(fig)

    return fig, axes_arr


def _flatten_axes(axes) -> List[Axes]:
    if isinstance(axes, Axes):
        return [axes]
    return list(np.atleast_1d(axes).ravel())
