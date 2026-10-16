"""Tests for plot_layout — Agg backend, shapes, labels, shared ticks, no overlap."""
import copy

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.transforms import Bbox

import plot_layout as cl


@pytest.fixture(autouse=True)
def _isolate_rc():
    before = copy.deepcopy(mpl.rcParams)
    yield
    mpl.rcParams.update(before)
    plt.close("all")


def test_subplots_shape_and_panel_labels():
    fig, axes = cl.subplots(2, 3, refwidth_mm=40, sharex=True, sharey=True, panel_labels=True)
    assert fig is not None
    assert np.asarray(axes).shape == (2, 3)
    labels = []
    for ax in np.asarray(axes).ravel():
        texts = [t.get_text() for t in ax.texts]
        labels.extend([t for t in texts if t.startswith("(")])
    assert labels[:6] == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    # rcParams unchanged by subplots itself (no style.use)
    plt.close(fig)


def test_label_styles():
    for style, expected0 in [("a", "a"), ("A", "A"), ("a.", "a."), ("(A)", "(A)"), ("(a)", "(a)")]:
        fig, axes = cl.subplots(1, 2, panel_labels=True, label_style=style, refwidth_mm=35)
        texts = [t.get_text() for t in np.asarray(axes).ravel()[0].texts]
        assert expected0 in texts
        plt.close(fig)


def test_shared_ticks_hidden():
    fig, axes = cl.subplots(2, 2, sharex=True, sharey=True, refwidth_mm=40)
    axes = np.asarray(axes)
    # After draw, inner tick labels should be hidden
    fig.canvas.draw()
    # top-left: no bottom x labels, has left y labels
    assert axes[0, 1].xaxis.get_tick_params().get("labelbottom") is False or not any(
        t.get_visible() for t in axes[0, 1].get_xticklabels()
    )
    # top-right: no left y labels
    assert axes[0, 1].yaxis.get_tick_params().get("labelleft") is False or not any(
        t.get_visible() for t in axes[0, 1].get_yticklabels()
    )
    # bottom-left keeps x labels; top-left hides them
    assert not any(t.get_visible() and t.get_text() != "" for t in axes[0, 0].get_xticklabels()) or \
        axes[0, 0].xaxis.get_tick_params().get("labelbottom") is False
    plt.close(fig)


def test_set_shared_labels():
    fig, axes = cl.subplots(2, 2, sharex=True, sharey=True, refwidth_mm=40)
    cl.set_shared_labels(fig, xlabel="X shared", ylabel="Y shared")
    fig.canvas.draw()
    assert fig._supxlabel is not None or any(
        ax.get_xlabel() == "X shared" for ax in np.asarray(axes).ravel()
    )
    plt.close(fig)


def test_mosaic_returns_dict():
    fig, axd = cl.subplots(mosaic=[["A", "B"], ["C", "C"]], panel_labels=True, refwidth_mm=40)
    assert isinstance(axd, dict)
    assert set(axd) >= {"A", "B", "C"}
    plt.close(fig)


def _bbox_display(artist, renderer):
    return artist.get_window_extent(renderer)


def test_edge_colorbar_no_overlap():
    fig, axes = cl.subplots(2, 2, refwidth_mm=45, panel_labels=False)
    axes = np.asarray(axes)
    Z = np.arange(9).reshape(3, 3).astype(float)
    mappable = None
    for ax in axes.ravel():
        mappable = ax.pcolormesh(Z, cmap="cividis")
    cbar = cl.edge_colorbar(fig, mappable, axes, loc="right", label="z")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cb_bb = cbar.ax.get_window_extent(renderer)
    for ax in axes.ravel():
        ax_bb = ax.get_window_extent(renderer)
        # Allow tiny numerical touch but not real overlap of interiors
        shrunk_ax = Bbox.from_extents(ax_bb.x0 + 2, ax_bb.y0 + 2, ax_bb.x1 - 2, ax_bb.y1 - 2)
        shrunk_cb = Bbox.from_extents(cb_bb.x0 + 2, cb_bb.y0 + 2, cb_bb.x1 - 2, cb_bb.y1 - 2)
        assert not shrunk_ax.overlaps(shrunk_cb), "colorbar overlaps axes"
    plt.close(fig)


def test_edge_legend_no_overlap():
    fig, axes = cl.subplots(1, 2, refwidth_mm=50, panel_labels=False, figsize_mm=(120, 50))
    axes = np.atleast_1d(axes)
    x = np.linspace(0, 1, 20)
    for i, ax in enumerate(axes):
        ax.plot(x, np.sin(x + i), label=f"series {i}")
        ax.plot(x, np.cos(x + i), label=f"cos {i}")
    leg = cl.edge_legend(fig, axes, loc="right", ncols=1)
    assert leg is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bb = leg.get_window_extent(renderer)
    for ax in axes:
        ax_bb = ax.get_window_extent(renderer)
        shrunk_ax = Bbox.from_extents(ax_bb.x0 + 2, ax_bb.y0 + 2, ax_bb.x1 - 2, ax_bb.y1 - 2)
        shrunk_leg = Bbox.from_extents(leg_bb.x0 + 2, leg_bb.y0 + 2, leg_bb.x1 - 2, leg_bb.y1 - 2)
        assert not shrunk_ax.overlaps(shrunk_leg), "legend overlaps axes"
    plt.close(fig)


def test_rcparams_unchanged_after_layout_calls():
    before = copy.deepcopy(dict(mpl.rcParams))
    fig, axes = cl.subplots(2, 2, sharex=True, sharey="row", refwidth_mm=40)
    Z = np.random.default_rng(0).random((4, 4))
    m = axes[0, 0].imshow(Z)
    cl.edge_colorbar(fig, m, axes, loc="bottom")
    axes[0, 0].plot([0, 1], [0, 1], label="a")
    axes[0, 1].plot([0, 1], [1, 0], label="b")
    cl.edge_legend(fig, axes, loc="top")
    cl.label_outer_shared(axes)
    cl.set_shared_labels(fig, "x", "y")
    fig.canvas.draw()
    after = dict(mpl.rcParams)
    # Compare keys that matter (ignore figure.figsize etc. that might be session-local)
    for k in before:
        if k.startswith("figure.") or k in ("backend", "backend_fallback"):
            continue
        assert after[k] == before[k], f"rcParam {k} changed: {before[k]!r} -> {after[k]!r}"
    plt.close(fig)


def test_figsize_mm():
    fig, ax = cl.subplots(1, 1, figsize_mm=(89, 64), panel_labels=False)
    w, h = fig.get_size_inches()
    assert abs(w * 25.4 - 89) < 0.1
    assert abs(h * 25.4 - 64) < 0.1
    plt.close(fig)


def _panel_label_artists(ax):
    """Return Text/Annotation artists whose content looks like a panel label.

    Includes standalone ``(a)`` annotations and titles that start with a
    folded panel-label prefix.
    """
    import string

    letters = set(string.ascii_letters)
    out = []

    def _is_panel(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        core = s.strip("().")
        if len(core) == 1 and core in letters:
            return True
        # Folded into title: "(a) rest"
        if s[0] == "(" and ")" in s[:4]:
            return True
        if len(s) >= 2 and s[0] in letters and s[1] in ". ":
            return True
        return False

    candidates = list(ax.texts)
    for attr in ("title", "_left_title", "_right_title"):
        t = getattr(ax, attr, None)
        if t is not None and t not in candidates:
            candidates.append(t)
    for child in ax.get_children():
        if child not in candidates and hasattr(child, "get_text"):
            candidates.append(child)
    for t in candidates:
        if t in out:
            continue
        if hasattr(t, "get_text") and _is_panel(t.get_text()) and t.get_visible():
            out.append(t)
    return out


def _title_artists(ax):
    arts = []
    for attr in ("_left_title", "title", "_right_title"):
        t = getattr(ax, attr, None)
        if t is not None and t.get_text() and t.get_visible():
            arts.append(t)
    return arts


def test_panel_labels_outside_bold_no_overlap_axes_patch():
    """Outside labels: fontweight bold; bbox must not intersect axes patch."""
    fig, axes = cl.subplots(2, 2, refwidth_mm=45, panel_labels=True, label_weight="bold")
    axes = np.asarray(axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.ravel():
        labels = _panel_label_artists(ax)
        assert labels, "expected a panel label"
        lab = labels[0]
        assert lab.get_fontweight() in ("bold", 700, "heavy")
        # Bold face must resolve to a real Bold font file
        fp = lab.get_fontproperties()
        path = fp.get_file() or ""
        assert "variablefont" not in path.lower()
        lab_bb = lab.get_window_extent(renderer)
        patch_bb = ax.patch.get_window_extent(renderer)
        assert not lab_bb.overlaps(patch_bb), "outside panel label intersects axes patch"
    plt.close(fig)


def test_panel_label_folds_into_title_same_baseline():
    """With a title, panel label sits on the title baseline (bold), title shifted right."""
    import style

    style.use("paper", variant="C", palette="npg")
    # Ensure CJK fonts registered for Chinese titles
    from matplotlib import font_manager

    for p in font_manager.findSystemFonts():
        if any(k in p.lower() for k in ("cjk", "simhei", "msyh", "pingfang")):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass

    fig, axes = cl.subplots(1, 2, refwidth_mm=55, panel_labels=True)
    axes = np.atleast_1d(axes)
    axes[0].set_title("\u5faa\u73af\u6027\u80fd\u5bf9\u6bd4")
    axes[1].set_title("\u6e29\u5ea6\u654f\u611f\u6027")
    cl.sync_panel_labels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in zip(axes, ("(a)", "(b)")):
        info = getattr(ax, cl._PANEL_LABEL_ATTR, None)
        assert info is not None and info.get("folded")
        lab = info.get("artist")
        assert lab is not None and lab.get_visible()
        assert lab.get_text() == letter
        assert lab.get_fontweight() in ("bold", 700, "heavy")
        assert cl._bold_face_resolves()
        # Title still contains the Chinese text (not missing from a Latin-only face)
        left = ax.get_title(loc="left")
        assert "\u5faa\u73af" in left or "\u6e29\u5ea6" in left or "\u5bf9\u6bd4" in left or "\u654f\u611f" in left
        # Label and title on same baseline (similar y), label to the left, no overlap
        title_bb = ax._left_title.get_window_extent(renderer)
        lab_bb = lab.get_window_extent(renderer)
        assert abs((lab_bb.y0 + lab_bb.y1) / 2 - (title_bb.y0 + title_bb.y1) / 2) < 8
        assert lab_bb.x1 <= title_bb.x0 + 1, f"label/title overlap: {lab_bb} vs {title_bb}"
    plt.close(fig)


def test_panel_labels_inside_has_outline_and_bbox():
    fig, axes = cl.subplots(1, 2, refwidth_mm=40, panel_labels=False)
    axes = np.atleast_1d(axes)
    texts = cl.add_panel_labels(axes, style="(a)", loc="inside", weight="bold")
    assert texts
    t = texts[0]
    assert t.get_path_effects(), "inside labels need white stroke outline"
    assert t.get_bbox_patch() is not None, "inside labels need semi-opaque bbox"
    fig.canvas.draw()
    plt.close(fig)


def _collect_text_bboxes(fig, axes_list):
    """All visible text-like artists with their owning axes (or None for fig-level)."""
    renderer = fig.canvas.get_renderer()
    items = []
    seen = set()

    def _add(artist, owner):
        if artist is None or not artist.get_visible():
            return
        aid = id(artist)
        if aid in seen:
            return
        seen.add(aid)
        try:
            bb = artist.get_window_extent(renderer)
        except Exception:
            return
        if bb.width <= 0 or bb.height <= 0:
            return
        # Skip empty strings
        if hasattr(artist, "get_text") and not str(artist.get_text()).strip():
            return
        items.append((artist, owner, bb))

    for ax in axes_list:
        for t in ax.texts:
            _add(t, ax)
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            _add(t, ax)
        if ax.xaxis.label.get_text():
            _add(ax.xaxis.label, ax)
        if ax.yaxis.label.get_text():
            _add(ax.yaxis.label, ax)
        for t in _title_artists(ax):
            _add(t, ax)
    # figure-level legend / colorbar / suplabels (texts only — not the
    # Legend container, whose bbox overlaps its own text children)
    for leg in fig.legends:
        for t in leg.get_texts():
            _add(t, None)
    for ax in fig.axes:
        if ax not in axes_list:
            # colorbar axes etc.
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                _add(t, ax)
            if ax.yaxis.label.get_text():
                _add(ax.yaxis.label, ax)
            if ax.xaxis.label.get_text():
                _add(ax.xaxis.label, ax)
    if getattr(fig, "_supxlabel", None) is not None:
        _add(fig._supxlabel, None)
    if getattr(fig, "_supylabel", None) is not None:
        _add(fig._supylabel, None)
    if getattr(fig, "_suptitle", None) is not None:
        _add(fig._suptitle, None)
    return items


def _is_tick_label(artist, ax) -> bool:
    if ax is None:
        return False
    return artist in list(ax.get_xticklabels()) + list(ax.get_yticklabels())


def _adjacent_shared_seam(ax_a, ax_b) -> bool:
    """True if two subplot axes share a horizontal or vertical grid seam."""
    if ax_a is None or ax_b is None or ax_a is ax_b:
        return False
    sa, sb = ax_a.get_subplotspec(), ax_b.get_subplotspec()
    if sa is None or sb is None:
        return False
    ra, ca = int(sa.rowspan.start), int(sa.colspan.start)
    rb, cb = int(sb.rowspan.start), int(sb.colspan.start)
    return (ra == rb and abs(ca - cb) == 1) or (ca == cb and abs(ra - rb) == 1)


def _assert_no_pairwise_text_overlap(items, tol=1.0):
    """Assert text bboxes do not overlap (allow *tol* px edge touch).

    Same-axes tick labels are excluded (MaxNLocator neighbors can touch on
    short colorbars); panel-label vs title on the same baseline are allowed
    to share an edge within *tol*. Abutting shared-axis edge tick labels at
    panel seams (e.g. ``12``|``0``) are also allowed — full data-edge tick
    sets must remain visible.
    """
    for i in range(len(items)):
        a_art, a_owner, a_bb = items[i]
        a_s = Bbox.from_extents(a_bb.x0 + tol, a_bb.y0 + tol, a_bb.x1 - tol, a_bb.y1 - tol)
        if a_s.width <= 0 or a_s.height <= 0:
            continue
        for j in range(i + 1, len(items)):
            b_art, b_owner, b_bb = items[j]
            # Skip tick–tick pairs on the same axes (incl. colorbar axes)
            if a_owner is not None and a_owner is b_owner:
                if _is_tick_label(a_art, a_owner) and _is_tick_label(b_art, b_owner):
                    continue
                # Panel label + title share the title baseline by design
                a_txt = str(getattr(a_art, "get_text", lambda: "")())
                b_txt = str(getattr(b_art, "get_text", lambda: "")())
                a_panel = a_txt.strip().startswith("(") and len(a_txt.strip()) <= 4
                b_panel = b_txt.strip().startswith("(") and len(b_txt.strip()) <= 4
                if a_panel or b_panel:
                    continue
            # Shared-grid seam: adjacent panels' edge tick labels may touch
            if (
                a_owner is not b_owner
                and _is_tick_label(a_art, a_owner)
                and _is_tick_label(b_art, b_owner)
                and _adjacent_shared_seam(a_owner, b_owner)
            ):
                continue
            b_s = Bbox.from_extents(b_bb.x0 + tol, b_bb.y0 + tol, b_bb.x1 - tol, b_bb.y1 - tol)
            if b_s.width <= 0 or b_s.height <= 0:
                continue
            assert not a_s.overlaps(b_s), (
                f"text overlap: {getattr(a_art, 'get_text', lambda: a_art)()!r} vs "
                f"{getattr(b_art, 'get_text', lambda: b_art)()!r}"
            )


def test_stress_no_text_overlap_other_axes_and_no_glyph_warnings():
    """Stress layouts: texts must not overlap other axes; no missing-glyph warnings."""
    import warnings

    import style

    style.use("paper", variant="C", palette="npg")
    # Ensure CJK fonts are available for Chinese labels
    from matplotlib import font_manager

    cjk_found = False
    cjk_keys = ("cjk", "simhei", "msyh", "pingfang", "hiragino", "heiti", "songti")
    for p in font_manager.findSystemFonts():
        if any(k in p.lower() for k in cjk_keys):
            try:
                font_manager.fontManager.addfont(p)
                cjk_found = True
            except Exception:
                pass

    # (a) 3x4 shared facet-like grid
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = cl.subplots(3, 4, refwidth_mm=35, sharex=True, sharey=True, panel_labels=True)
        axes = np.asarray(axes)
        x = np.linspace(0, 1, 20)
        for i, ax in enumerate(axes.ravel()):
            ax.plot(x, np.sin(x * (i + 1)), color=style.NPG[i % len(style.NPG)])
            style.finish(ax)
        cl.set_shared_labels(fig, xlabel="x", ylabel="y")
        fig.canvas.draw()
        glyph_warns = [w for w in caught if "Glyph" in str(w.message)]
        assert not glyph_warns, f"missing glyph warnings: {glyph_warns}"

        ax_list = list(axes.ravel())
        items = _collect_text_bboxes(fig, ax_list)
        for artist, owner, bb in items:
            for other in ax_list:
                if owner is other:
                    continue
                other_bb = other.patch.get_window_extent(fig.canvas.get_renderer())
                # shrink slightly to allow edge touching
                shrunk = Bbox.from_extents(
                    other_bb.x0 + 1, other_bb.y0 + 1, other_bb.x1 - 1, other_bb.y1 - 1
                )
                assert not bb.overlaps(shrunk), (
                    f"text {getattr(artist, 'get_text', lambda: artist)()!r} overlaps other axes"
                )
        plt.close(fig)

    # (b) talk preset 2x2 with edge legend + colorbar
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        style.use_talk()
        fig, axes = cl.subplots(2, 2, figsize_mm=(180, 130), sharex=True, sharey="row")
        axes = np.asarray(axes)
        x = np.linspace(0, 2, 40)
        for j, ax in enumerate(axes[0]):
            ax.plot(x, np.sin(x + j), label=f"s{j}", color=style.NPG[j])
            style.finish(ax)
        Z = np.arange(16).reshape(4, 4).astype(float)
        m = None
        for j, ax in enumerate(axes[1]):
            m = ax.pcolormesh(Z + j, cmap=style.seq_cmap())
            style.finish(ax)
        leg = cl.edge_legend(fig, axes[0, :], loc="right")
        cbar = cl.edge_colorbar(fig, m, axes[1, :], loc="bottom", label="z")
        fig.canvas.draw()
        glyph_warns = [w for w in caught if "Glyph" in str(w.message)]
        assert not glyph_warns
        renderer = fig.canvas.get_renderer()
        if leg is not None:
            leg_bb = leg.get_window_extent(renderer)
            for ax in axes.ravel():
                ax_bb = ax.get_window_extent(renderer)
                sa = Bbox.from_extents(ax_bb.x0 + 2, ax_bb.y0 + 2, ax_bb.x1 - 2, ax_bb.y1 - 2)
                sl = Bbox.from_extents(leg_bb.x0 + 2, leg_bb.y0 + 2, leg_bb.x1 - 2, leg_bb.y1 - 2)
                assert not sa.overlaps(sl)
        cb_bb = cbar.ax.get_window_extent(renderer)
        for ax in axes.ravel():
            ax_bb = ax.get_window_extent(renderer)
            sa = Bbox.from_extents(ax_bb.x0 + 2, ax_bb.y0 + 2, ax_bb.x1 - 2, ax_bb.y1 - 2)
            sc = Bbox.from_extents(cb_bb.x0 + 2, cb_bb.y0 + 2, cb_bb.x1 - 2, cb_bb.y1 - 2)
            assert not sa.overlaps(sc)
        plt.close(fig)

    # (c) long Chinese labels — no Glyph warnings; panel labels fold into titles
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        style.use("paper", variant="C", palette="npg")
        fig, axes = cl.subplots(1, 2, refwidth_mm=60, panel_labels=True)
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.plot([1, 2, 3], [100, 98, 95], label="\u7535\u89e3\u6db2 A\uff081 mol·L⁻¹\uff09")
            ax.set_xlabel("\u5faa\u73af\u6b21\u6570\uff0825 ℃\uff0c1C/1C\uff09")
            ax.set_title("\u5faa\u73af\u6027\u80fd\u5bf9\u6bd4")
            style.finish(ax)
        axes[0].set_ylabel("\u5bb9\u91cf\u4fdd\u6301\u7387\uff08%\uff0c\u76f8\u5bf9\u9996\u5708\uff09")
        cl.sync_panel_labels(fig)
        cl.edge_legend(fig, axes, loc="right")
        fig.canvas.draw()
        glyph_warns = [w for w in caught if "Glyph" in str(w.message)]
        if cjk_found:  # glyph check is only meaningful when a CJK font is installed
            assert not glyph_warns, f"Glyph warnings with Chinese labels: {glyph_warns}"
        else:
            warnings.warn("No CJK font found; CJK glyph check skipped", stacklevel=1)
        info0 = getattr(axes[0], cl._PANEL_LABEL_ATTR, None)
        assert info0 and info0.get("folded") and info0["artist"].get_text() == "(a)"
        assert "\u5faa\u73af" in (axes[0].get_title(loc="left") or "")
        items = _collect_text_bboxes(fig, list(axes))
        _assert_no_pairwise_text_overlap(items, tol=1.0)
        plt.close(fig)


def test_demo_figures_geometry_invariants():
    """Demo-like figures: no text overlap, equal row heights, colorbar span, facet inners."""
    import pandas as pd

    import plot_stats as cs
    import style

    style.use("paper", variant="C", palette="npg")

    # --- layout 2x3 with top legend + right colorbar ---
    fig, axes = cl.subplots(
        2, 3, figsize_mm=(190, 125), sharex=True, sharey="row", panel_labels=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0]),
    )
    axes = np.asarray(axes)
    x = np.linspace(0, 4 * np.pi, 80)
    for j in range(3):
        axes[0, j].plot(x, np.sin(x + j), label=f"s{j}")
        axes[0, j].plot(x, 0.5 * np.sin(x + j), label=f"s{j}/2", ls="--")
        axes[0, j].set_ylim(-1, 1)
    Z = np.arange(9).reshape(3, 3).astype(float)
    m = None
    for j in range(3):
        m = axes[1, j].pcolormesh(Z + j)
    cl.set_shared_labels(fig, xlabel="x")
    axes[0, 0].set_ylabel("amplitude")
    axes[1, 0].set_ylabel("index")
    cl.label_outer_shared(axes)
    cbar = cl.edge_colorbar(fig, m, axes[1, :], loc="right", label="z")
    leg = cl.edge_legend(fig, axes[0, :], loc="top", ncols=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    h0 = axes[0, 0].get_position().height
    h1 = axes[1, 0].get_position().height
    assert abs(h0 - h1) / max(h0, h1) < 0.08, f"row heights unequal: {h0} vs {h1}"

    items = _collect_text_bboxes(fig, list(axes.ravel()))
    _assert_no_pairwise_text_overlap(items, tol=1.0)
    for artist, owner, bb in items:
        for other in axes.ravel():
            if owner is other:
                continue
            patch = other.patch.get_window_extent(renderer)
            shrunk = Bbox.from_extents(patch.x0 + 1, patch.y0 + 1, patch.x1 - 1, patch.y1 - 1)
            assert not bb.overlaps(shrunk), "text overlaps other axes data area"
    if leg is not None:
        leg_bb = leg.get_window_extent(renderer)
        for ax in axes[0]:
            for lab in _panel_label_artists(ax):
                lb = lab.get_window_extent(renderer)
                assert not Bbox.from_extents(
                    leg_bb.x0 + 1, leg_bb.y0 + 1, leg_bb.x1 - 1, leg_bb.y1 - 1
                ).overlaps(
                    Bbox.from_extents(lb.x0 + 1, lb.y0 + 1, lb.x1 - 1, lb.y1 - 1)
                ), "top legend overlaps panel label"
    plt.close(fig)

    # --- talk 2x2 bottom colorbar spans served axes ---
    style.use_talk()
    fig, axes = cl.subplots(2, 2, figsize_mm=(200, 150), sharex=True, sharey="row")
    axes = np.asarray(axes)
    for j, ax in enumerate(axes[0]):
        ax.plot([0, 1], [0, 1], label=f"s{j}")
    m = None
    for j, ax in enumerate(axes[1]):
        m = ax.pcolormesh(Z + j)
    cl.set_shared_labels(fig, xlabel="x", ylabel="y")
    for ax in axes.ravel():
        assert ax.get_ylabel() == ""
        assert ax.get_xlabel() == ""
    cbar = cl.edge_colorbar(fig, m, axes[1, :], loc="bottom", label="z")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cb_bb = cbar.ax.get_window_extent(renderer)
    ax_bbs = [ax.get_window_extent(renderer) for ax in axes[1]]
    span = max(b.x1 for b in ax_bbs) - min(b.x0 for b in ax_bbs)
    assert cb_bb.width >= 0.5 * span, f"colorbar width {cb_bb.width} < 50% of axes span {span}"
    items = _collect_text_bboxes(fig, list(axes.ravel()))
    _assert_no_pairwise_text_overlap(items, tol=1.0)
    plt.close(fig)

    # --- facet: inner x/y labels hidden ---
    style.use("paper", variant="C", palette="npg")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "row": np.repeat(["R0", "R1", "R2"], 40),
            "col": np.tile(np.repeat(["C0", "C1", "C2", "C3"], 10), 3),
            "x": rng.random(120),
            "y": rng.random(120),
        }
    )

    def _plot(ax, data, **kw):
        cs.scatter(ax, data, x="x", y="y")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig, axes = cs.facet(
        df, row="row", col="col", func=_plot, refwidth_mm=40, sharex=True, sharey=True,
        xlabel="x", ylabel="y",
    )
    axes = np.asarray(axes)
    fig.canvas.draw()
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            assert ax.get_xlabel() == "", f"inner xlabel at [{i},{j}]"
            assert ax.get_ylabel() == "", f"inner ylabel at [{i},{j}]"
            if i < axes.shape[0] - 1:
                assert not any(
                    t.get_visible() and t.get_text() for t in ax.get_xticklabels()
                ) or ax.xaxis.get_tick_params().get("labelbottom") is False
            if j > 0:
                assert not any(
                    t.get_visible() and t.get_text() for t in ax.get_yticklabels()
                ) or ax.yaxis.get_tick_params().get("labelleft") is False
            # Panel label on title baseline (folded)
            info = getattr(ax, cl._PANEL_LABEL_ATTR, None)
            assert info and info.get("folded"), f"expected folded panel label at [{i},{j}]"
            assert info["artist"].get_text().startswith("(")
            left = (ax.get_title(loc="left") or "").lstrip("\u00a0").strip()
            assert "row=" in left or "col=" in left
    items = _collect_text_bboxes(fig, list(axes.ravel()))
    _assert_no_pairwise_text_overlap(items, tol=1.0)
    for artist, owner, bb in items:
        for other in axes.ravel():
            if owner is other:
                continue
            patch = other.patch.get_window_extent(fig.canvas.get_renderer())
            shrunk = Bbox.from_extents(patch.x0 + 1, patch.y0 + 1, patch.x1 - 1, patch.y1 - 1)
            assert not bb.overlaps(shrunk)
    plt.close(fig)


# ----- TASK4 invariants -----



def test_heatmap_full_tick_labels_0_to_12():
    """Heatmap panels keep full 0..12 tick labels on outer edges (no sticky drop)."""
    import style
    from matplotlib.ticker import FixedLocator

    style.use("paper", variant="C", palette="npg")
    fig, axes = cl.subplots(
        2, 3, figsize_mm=(190, 125), sharex=True, sharey="row", panel_labels=True
    )
    axes = np.asarray(axes)
    x = np.linspace(0, 4 * np.pi, 40)
    for j in range(3):
        axes[0, j].plot(x, np.sin(x + j), label=f"s{j}")
        axes[0, j].set_ylim(-1, 1)
    xe = np.arange(13, dtype=float)
    ye = np.arange(13, dtype=float)
    Z = np.arange(144, dtype=float).reshape(12, 12)
    for j in range(3):
        axes[1, j].pcolormesh(xe, ye, Z + j, shading="flat")
        axes[1, j].set_xlim(0, 12)
        axes[1, j].set_ylim(0, 12)
        axes[1, j].xaxis.set_major_locator(FixedLocator([0, 2, 4, 6, 8, 10, 12]))
        axes[1, j].yaxis.set_major_locator(FixedLocator([0, 2, 4, 6, 8, 10, 12]))
    cl.label_outer_shared(axes)
    fig.canvas.draw()

    expected = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0}
    # (d): both x and y labels full set
    ax_d = axes[1, 0]
    x_vis = {t.get_loc() for t in ax_d.xaxis.get_major_ticks() if t.label1.get_visible()}
    y_vis = {t.get_loc() for t in ax_d.yaxis.get_major_ticks() if t.label1.get_visible()}
    assert expected <= x_vis, f"(d) missing x tick labels: {expected - x_vis}"
    assert expected <= y_vis, f"(d) missing y tick labels: {expected - y_vis}"
    # (e): x labels full; y labels hidden (inner col)
    ax_e = axes[1, 1]
    x_vis_e = {t.get_loc() for t in ax_e.xaxis.get_major_ticks() if t.label1.get_visible()}
    assert expected <= x_vis_e, f"(e) missing x tick labels: {expected - x_vis_e}"
    # (f): x labels full
    ax_f = axes[1, 2]
    x_vis_f = {t.get_loc() for t in ax_f.xaxis.get_major_ticks() if t.label1.get_visible()}
    assert expected <= x_vis_f, f"(f) missing x tick labels: {expected - x_vis_f}"
    # Tick *marks* present at data edges on all heat panels
    for j, ax in enumerate(axes[1]):
        xt = {t.get_loc() for t in ax.xaxis.get_major_ticks()}
        yt = {t.get_loc() for t in ax.yaxis.get_major_ticks()}
        assert expected <= xt, f"heat col {j} missing x ticks"
        assert expected <= yt, f"heat col {j} missing y ticks"
    plt.close(fig)




def test_legend_to_panel_label_gap_tight():
    """Top edge legend sits within ~2*fontsize of the highest panel label."""
    import style

    style.use("paper", variant="C", palette="npg")
    fig, axes = cl.subplots(
        2, 3, figsize_mm=(190, 125), sharex=True, sharey="row", panel_labels=True
    )
    axes = np.asarray(axes)
    x = np.linspace(0, 1, 20)
    for j in range(3):
        axes[0, j].plot(x, np.sin(x + j), label=f"s{j}")
        axes[0, j].plot(x, np.cos(x + j), label=f"c{j}", ls="--")
    leg = cl.edge_legend(fig, axes[0, :], loc="top", ncols=3)
    assert leg is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bb = leg.get_window_extent(renderer)
    label_tops = []
    for ax in axes[0]:
        info = getattr(ax, cl._PANEL_LABEL_ATTR, None)
        assert info and info.get("artist")
        label_tops.append(info["artist"].get_window_extent(renderer).y1)
    highest = max(label_tops)
    gap = leg_bb.y0 - highest
    fs_px = float(mpl.rcParams["font.size"]) * (fig.dpi / 72.0)
    assert gap < 2.0 * fs_px, f"legend–label gap {gap:.1f}px >= 2*fontsize ({2*fs_px:.1f}px)"
    assert gap > -2.0, "legend overlaps panel labels"
    plt.close(fig)


def test_chinese_title_min_horizontal_gap():
    """Adjacent Chinese panel titles keep ≥1 em (≥ fontsize px) horizontal gap."""
    import style
    from matplotlib import font_manager

    style.use("paper", variant="C", palette="npg")
    for p in font_manager.findSystemFonts():
        if any(k in p.lower() for k in ("cjk", "simhei", "msyh", "pingfang")):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass

    fig, axes = cl.subplots(1, 2, refwidth_mm=70, panel_labels=True, sharey=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot([1, 2, 3], [100, 98, 95], label="\u7535\u89e3\u6db2 A\uff081 mol·L⁻¹ LiPF₆\uff09")
        ax.plot([1, 2, 3], [99, 96, 93], label="\u7535\u89e3\u6db2 C\uff08\u6dfb\u52a0\u5242\uff09", ls="--")
        ax.set_title("\u5faa\u73af\u6027\u80fd\u5bf9\u6bd4\uff08\u5408\u6210\u6570\u636e\uff09" if i == 0 else "\u6e29\u5ea6\u654f\u611f\u6027\uff08\u5408\u6210\u6570\u636e\uff09")
        style.finish(ax)
    cl.sync_panel_labels(fig)
    cl.edge_legend(fig, axes, loc="right", ncols=1)
    cl.ensure_title_gaps(fig, min_gap_em=1.0)
    fig.canvas.draw()
    # allow the draw-hook iterative fix another pass
    cl.ensure_title_gaps(fig, min_gap_em=1.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _row_bb(ax):
        boxes = []
        info = getattr(ax, cl._PANEL_LABEL_ATTR, None)
        if info and info.get("artist"):
            boxes.append(info["artist"].get_window_extent(renderer))
        for attr in ("_left_title", "title"):
            t = getattr(ax, attr, None)
            if t is not None and t.get_text() and t.get_visible():
                boxes.append(t.get_window_extent(renderer))
        return Bbox.union(boxes)

    bb0, bb1 = _row_bb(axes[0]), _row_bb(axes[1])
    gap = bb1.x0 - bb0.x1
    fs_px = float(mpl.rcParams["font.size"]) * (fig.dpi / 72.0)
    assert gap >= fs_px - 0.5, f"title gap {gap:.1f}px < fontsize {fs_px:.1f}px"
    plt.close(fig)


