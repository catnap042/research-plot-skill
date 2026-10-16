"""Tests for plot_stats — Agg backend, numbers vs numpy/scipy, annotations, rcParams."""
import copy

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import plot_stats as cs
import style


def _synth(n=40, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(["A", "B", "C"], n)
    y = np.concatenate(
        [
            rng.normal(0, 1, n),
            rng.normal(1.2, 1.1, n),
            rng.normal(-0.5, 0.8, n),
        ]
    )
    x = rng.normal(size=3 * n)
    hue = np.tile(["u", "v"], 3 * n // 2 + 1)[: 3 * n]
    return pd.DataFrame({"group": g, "y": y, "x": x, "hue": hue, "t": np.tile(np.arange(n), 3)})


@pytest.fixture(autouse=True)
def _isolate_rc():
    before = copy.deepcopy(mpl.rcParams)
    yield
    mpl.rcParams.update(before)
    plt.close("all")


def test_box_strip_n_and_medians():
    df = _synth()
    fig, ax = plt.subplots()
    stats = cs.box_strip(ax, df, x="group", y="y", seed=1)
    for g in ["A", "B", "C"]:
        vals = df.loc[df["group"] == g, "y"].to_numpy()
        assert stats["n"][g] == len(vals)
        assert stats["median"][g] == pytest.approx(np.median(vals))
        assert stats["q1"][g] == pytest.approx(np.percentile(vals, 25))
        assert stats["q3"][g] == pytest.approx(np.percentile(vals, 75))
    # tick labels contain n=
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any("n=40" in lab for lab in labels)
    # annotation via _annotate_error (below axes, not over data)
    assert "median/IQR" in stats["annotation"]
    assert "1.5" in stats["annotation"] and "IQR" in stats["annotation"]
    assert f"n={len(df)}" in stats["annotation"]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "median/IQR" in texts
    assert f"n={len(df)}" in texts
    # Caption sits below the axes (axes-fraction y < 0)
    ann = [t for t in ax.texts if "median/IQR" in t.get_text()]
    assert ann and ann[0].get_position()[1] < 0
    plt.close(fig)


def test_stats_annotations_below_axes_not_over_data():
    """All stats error captions live below the axes patch."""
    df = _synth(n=30)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), layout="constrained")
    cs.box_strip(axes[0], df, x="group", y="y", seed=0)
    cs.violin(axes[1], df, x="group", y="y", inner="box")
    cs.regplot(axes[2], df, x="x", y="y", n_boot="analytic", seed=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        patch = ax.patch.get_window_extent(renderer)
        for t in ax.texts:
            if not t.get_text().strip():
                continue
            # Heuristic: captions mention n= or CI/SD/SE/box/violin
            s = t.get_text().lower()
            if not any(k in s for k in ("n=", "ci", "sd", "se", "box", "violin", "ols")):
                continue
            assert t.get_position()[1] < 0, f"annotation should be below axes: {t.get_text()!r}"
            bb = t.get_window_extent(renderer)
            # Allow touching the bottom edge but not the data interior
            interior = __import__("matplotlib.transforms", fromlist=["Bbox"]).Bbox.from_extents(
                patch.x0 + 2, patch.y0 + 2, patch.x1 - 2, patch.y1 - 2
            )
            assert not bb.overlaps(interior), f"annotation overlaps data: {t.get_text()!r}"
    plt.close(fig)


def test_violin_annotation_and_n():
    df = _synth()
    fig, ax = plt.subplots()
    stats = cs.violin(ax, df, x="group", y="y", inner="quartile")
    assert sum(stats["n"].values()) == len(df)
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "n=" in texts
    assert "violin" in texts.lower() or "n=" in texts
    plt.close(fig)


def test_violin_cut_extends_or_clips_support():
    df = _synth(n=50)
    fig, ax = plt.subplots()
    st_ext = cs.violin(ax, df, x="group", y="y", cut=0.5)
    for g in st_ext["categories"]:
        dmin, dmax = st_ext["data_range"][g]
        slo, shi = st_ext["support"][g]
        assert slo < dmin - 1e-9, "cut>0 should extend below data min"
        assert shi > dmax + 1e-9, "cut>0 should extend above data max"
    ax.cla()
    st_clip = cs.violin(ax, df, x="group", y="y", cut=0)
    for g in st_clip["categories"]:
        dmin, dmax = st_clip["data_range"][g]
        slo, shi = st_clip["support"][g]
        assert slo == pytest.approx(dmin)
        assert shi == pytest.approx(dmax)
    plt.close(fig)


def test_regplot_ols_and_annotation():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 80)
    y = 2.5 * x + 0.3 + rng.normal(0, 0.4, 80)
    df = pd.DataFrame({"x": x, "y": y})
    fig, ax = plt.subplots()
    stats = cs.regplot(ax, df, x="x", y="y", ci=95, n_boot="analytic", order=1, seed=0)
    # Reference OLS via numpy
    A = np.column_stack([np.ones_like(x), x])
    coef_ref, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    assert stats["coef"][0] == pytest.approx(coef_ref[0], rel=1e-6)
    assert stats["coef"][1] == pytest.approx(coef_ref[1], rel=1e-6)
    assert stats["n"] == 80
    assert "95% CI" in stats["annotation"]
    assert "n=80" in stats["annotation"]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "95% CI" in texts and "n=80" in texts
    # CI band exists (lo < hi)
    assert np.all(stats["lo"] <= stats["hi"])
    plt.close(fig)


def test_regplot_bootstrap_ci_covers_line():
    rng = np.random.default_rng(3)
    x = np.linspace(-1, 1, 50)
    y = 1.0 * x + rng.normal(0, 0.2, 50)
    df = pd.DataFrame({"x": x, "y": y})
    fig, ax = plt.subplots()
    stats = cs.regplot(ax, df, x="x", y="y", ci=95, n_boot=200, order=1, seed=3)
    assert stats["y_fit"].shape == stats["lo"].shape
    # Most of the fit line should lie inside the band
    inside = np.mean((stats["y_fit"] >= stats["lo"]) & (stats["y_fit"] <= stats["hi"]))
    assert inside > 0.95
    plt.close(fig)


def test_lineplot_sd_matches_numpy():
    df = _synth(n=30)
    # single group mean±sd over t for group A
    sub = df[df["group"] == "A"]
    fig, ax = plt.subplots()
    stats = cs.lineplot(ax, sub, x="t", y="y", err="sd")
    # At each t there is one point in this synthetic data (tiled), so SD=0 — use aggregated
    # Build multi-replicate data
    rng = np.random.default_rng(4)
    rows = []
    for t in range(5):
        for _ in range(20):
            rows.append({"t": t, "y": rng.normal(t, 1.5)})
    df2 = pd.DataFrame(rows)
    ax.cla()
    stats = cs.lineplot(ax, df2, x="t", y="y", err="sd")
    for t in range(5):
        vals = df2.loc[df2["t"] == t, "y"].to_numpy()
        m = vals.mean()
        sd = vals.std(ddof=1)
        g = stats["groups"]["_all"]
        idx = list(g["x"]).index(t)
        assert g["mean"][idx] == pytest.approx(m)
        assert g["hi"][idx] - g["mean"][idx] == pytest.approx(sd)
    assert "SD" in stats["annotation"]
    assert "n=" in stats["annotation"]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "SD" in texts and "n=" in texts
    plt.close(fig)


def test_lineplot_se_and_ci():
    rng = np.random.default_rng(5)
    rows = []
    for t in range(4):
        for _ in range(25):
            rows.append({"t": t, "y": rng.normal(0, 2.0), "hue": "a"})
            rows.append({"t": t, "y": rng.normal(1, 2.0), "hue": "b"})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    st_se = cs.lineplot(ax, df[df["hue"] == "a"], x="t", y="y", err="se")
    assert "SE" in st_se["annotation"]
    # check SE value at t=0
    vals = df.loc[(df["hue"] == "a") & (df["t"] == 0), "y"].to_numpy()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    g = st_se["groups"]["_all"]
    idx = list(g["x"]).index(0)
    assert g["hi"][idx] - g["mean"][idx] == pytest.approx(se)
    ax.cla()
    st_ci = cs.lineplot(ax, df[df["hue"] == "a"], x="t", y="y", err="ci", ci=95)
    assert "95% CI" in st_ci["annotation"]
    tcrit = sps.t.ppf(0.975, len(vals) - 1)
    assert st_ci["groups"]["_all"]["hi"][idx] - st_ci["groups"]["_all"]["mean"][idx] == pytest.approx(tcrit * se)
    plt.close(fig)


def test_scatter_and_facet():
    df = _synth(n=20)
    fig, ax = plt.subplots()
    st = cs.scatter(ax, df, x="x", y="y", hue="group")
    assert st["n"] == len(df)
    assert set(st["groups"]) == {"A", "B", "C"}
    plt.close(fig)

    def _plot(ax, data, **kw):
        cs.scatter(ax, data, x="x", y="y")

    fig, axes = cs.facet(df, col="group", func=_plot, refwidth_mm=40)
    assert np.asarray(axes).shape == (1, 3)
    plt.close(fig)


def test_rcparams_unchanged_by_stats():
    # Do not call style.use — stats must not alter theme
    before = copy.deepcopy(dict(mpl.rcParams))
    df = _synth(n=15)
    fig, ax = plt.subplots()
    cs.box_strip(ax, df, "group", "y")
    cs.violin(ax, df, "group", "y", inner="box")
    cs.regplot(ax, df, "x", "y", n_boot=50)
    cs.lineplot(ax, df, "t", "y", hue="group", err="se")
    cs.scatter(ax, df, "x", "y", hue="group", style="hue")
    after = dict(mpl.rcParams)
    for k in before:
        if k.startswith("figure.") or k in ("backend", "backend_fallback"):
            continue
        assert after[k] == before[k], f"rcParam {k} changed"
    plt.close(fig)


def test_style_use_still_ok():
    """style.use may change rcParams; that's allowed and outside stats module."""
    style.use("paper", variant="C", palette="npg")
    df = _synth(n=10)
    fig, ax = plt.subplots()
    st = cs.box_strip(ax, df, "group", "y")
    assert sum(st["n"].values()) == len(df)
    plt.close(fig)


def test_box_violin_share_deep_fill_alpha():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plot_stats as cs
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(list("ABC"), 20), "v": rng.normal(size=60)})
    fig, (a1, a2) = plt.subplots(1, 2)
    cs.box_strip(a1, df, "g", "v")
    cs.violin(a2, df, "g", "v")
    box_alphas = {p.get_alpha() for p in a1.patches}
    vio_alphas = {p.get_alpha() for p in a2.patches if p.get_facecolor()[:3] != (1.0, 1.0, 1.0)}  # skip white inner boxes
    assert box_alphas == {cs.FILL_ALPHA} and vio_alphas == {cs.FILL_ALPHA}
    assert cs.FILL_ALPHA == 0.72
    plt.close(fig)


def test_violin_gg_default_and_classic_option():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plot_stats as cs
    rng = np.random.default_rng(1)
    v = rng.normal(size=40)
    df = pd.DataFrame({"g": ["A"] * 40, "v": v})
    fig, (a1, a2) = plt.subplots(1, 2)
    st = cs.violin(a1, df, "g", "v")
    sd = v.std(ddof=1); iqr = np.subtract(*np.percentile(v, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34) * 40 ** -0.2
    assert st["look"] == "gg" and abs(st["bw"]["A"] - h) < 1e-9
    lo, hi = st["support"]["A"]
    assert abs((v.min() - lo) - 3 * h) < 1e-9 and abs((hi - v.max()) - 3 * h) < 1e-9
    assert "whiskers 1.5·IQR" in st["annotation"]
    st2 = cs.violin(a2, df, "g", "v", look="classic")
    assert st2["cut"] == 0.5 and st2["scale"] == "width"
    plt.close(fig)


def test_violin_points_default_matches_ggplot2():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import to_rgba
    import plot_stats as cs
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": np.repeat(list("AB"), 30), "v": rng.normal(size=60)})
    fig, ax = plt.subplots()
    cs.violin(ax, df, "g", "v", points=True)
    pts = [c for c in ax.collections if len(c.get_offsets()) == 30]
    assert len(pts) == 2
    for i, c in enumerate(pts):
        assert c.get_facecolor().size == 0 or c.get_facecolor()[0][3] == 0  # hollow
        assert np.allclose(c.get_edgecolor()[0][:3], to_rgba(cs._color(i))[:3])
        assert abs(c.get_alpha() - 0.8) < 1e-9
        off = c.get_offsets()[:, 0] - i
        assert np.all(np.abs(off) <= 0.08 + 1e-12)
        p = c.get_paths()[0]; assert len(p.vertices) > 8  # circle path, not polygon marker
    plt.close(fig)
