"""Pure-matplotlib layout helpers (theme_bw + npg style), built on constrained layout / subplot_mosaic.

Never mutates global rcParams except when the caller passes overrides through **kw
that matplotlib itself applies (e.g. subplot_kw). Spacing pads are derived from the
current font size at call time without writing back to rcParams.
"""
from __future__ import annotations

import string
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

MM_PER_INCH = 25.4

_ShareSpec = Union[bool, str]
_PANEL_LABEL_ATTR = "_rp_panel_label"
_PANEL_SYNC_CID = "_rp_panel_label_cid"
_TICK_PRUNE_CID = "_rp_tick_prune_cid"
_TITLE_GAP_CID = "_rp_title_gap_cid"
_TITLE_GAP_ATTR = "_rp_title_gap_wrapped"


def _renderer(fig: Figure):
    """Renderer for text extents; works for Agg and vector (PDF/SVG) canvases."""
    get = getattr(fig.canvas, "get_renderer", None)
    if get is not None:
        try:
            return get()
        except Exception:
            pass
    return fig._get_renderer()


def _mm_to_in(mm: float) -> float:
    return float(mm) / MM_PER_INCH


def _font_pt() -> float:
    return float(mpl.rcParams["font.size"])


def _resolve_fontsize(value) -> float:
    """Resolve rcParams font size entries that may be relative strings."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    # Relative size names → scale of font.size (matplotlib convention)
    scales = {
        "xx-small": 0.579,
        "x-small": 0.694,
        "small": 0.833,
        "medium": 1.0,
        "large": 1.2,
        "x-large": 1.440,
        "xx-large": 1.728,
        "larger": 1.2,
        "smaller": 0.833,
    }
    if isinstance(value, str) and value.lower() in scales:
        return _font_pt() * scales[value.lower()]
    return _font_pt()


def _layout_pads() -> Dict[str, float]:
    """Constrained-layout pads scaled to current font size (in inches).

    Kept tight (~1 line) so a top edge legend sits close to panel labels
    and row gaps stay near one line-height.
    """
    pt = _font_pt()
    # ~1 line-height vertical pad; slightly tighter horizontal
    h = 0.55 * pt / 72.0
    w = 0.45 * pt / 72.0
    return dict(w_pad=w, h_pad=h, wspace=w * 0.35, hspace=h * 0.45)


def _bold_fontproperties(fontsize: float, weight: str = "bold") -> FontProperties:
    """FontProperties whose bold weight resolves to a real Bold face.

    Arimo's variable font often fails to bold under matplotlib; put
    Liberation Sans / DejaVu Sans first (discrete Bold TTFs) while keeping
    the rc family list afterward so CJK glyphs still fall back correctly.
    """
    rc_fam = mpl.rcParams.get("font.family", ["DejaVu Sans"])
    if isinstance(rc_fam, str):
        rc_fam = [rc_fam]
    # Prefer families with discrete Bold TTFs; keep CJK fallbacks from rc.
    # Skip Arimo — its variable font fails matplotlib's bold lookup.
    ordered: List[str] = []
    for fam in ["Liberation Sans", "DejaVu Sans", *list(rc_fam), "Noto Sans CJK SC"]:
        if fam and fam not in ordered and fam != "Arimo":
            ordered.append(fam)
    # Family *list* preserves per-glyph fallback for Chinese titles
    return FontProperties(family=ordered, weight=weight, size=fontsize)


def _bold_face_resolves(fontsize: float = 10.0) -> bool:
    """True if a bold Liberation/DejaVu face file is findable."""
    for fam in ("Liberation Sans", "DejaVu Sans"):
        fp = FontProperties(family=fam, weight="bold", size=fontsize)
        try:
            path = findfont(fp, fallback_to_default=False)
        except Exception:
            continue
        if path and "bold" in path.lower() and "variablefont" not in path.lower():
            return True
    return False


def _axes_title_text(ax: Axes) -> Tuple[str, str]:
    """Return (title_text, loc) preferring left/center/right with content."""
    for loc in ("left", "center", "right"):
        try:
            t = ax.get_title(loc=loc)
        except TypeError:
            t = ax.get_title() if loc == "center" else ""
        if t and str(t).strip():
            return str(t).strip(), loc
    return "", str(mpl.rcParams.get("axes.titlelocation", "center") or "center")


def _strip_panel_prefix(title: str, label: str) -> str:
    """Remove a previously folded panel-label prefix / leading pads from *title*."""
    t = title.strip().lstrip("\u00a0").strip()
    for sep in (f"{label} ", f"{label}\u00a0", label):
        if t.startswith(sep):
            return t[len(sep) :].lstrip("\u00a0").strip()
    return t


def _title_x_offset_for_label(label: str, fontsize: float) -> float:
    """Axes-fraction x offset so a left title clears a bold panel label."""
    # ~0.55em per character + pad; convert pt → rough axes fraction assuming
    # a typical panel is ≥1.5" wide (clamped so short panels still clear).
    em = fontsize * 0.55 * max(len(label), 1) + fontsize * 0.7
    return float(np.clip(em / 120.0, 0.06, 0.22))


def sync_panel_labels(fig: Figure) -> None:
    """Fold panel labels into titles when present; else keep outside labels.

    When an axes has a title, the bold panel label sits on the same baseline
    as the title (left-aligned) and the title is shifted right. Constrained
    layout then treats them as one title row, avoiding collisions with a top
    edge legend.
    """
    if getattr(fig, "_rp_syncing_panel_labels", False):
        return
    setattr(fig, "_rp_syncing_panel_labels", True)
    try:
        _sync_panel_labels_impl(fig)
    finally:
        setattr(fig, "_rp_syncing_panel_labels", False)


def _ensure_outside_label_artist(ax: Axes, info: dict) -> Any:
    """Return a visible bold outside panel-label annotate, creating if needed."""
    label = info["label"]
    fontsize = info["fontsize"]
    weight = info["weight"]
    pad_pt = info["pad_pt"]
    artist = info.get("artist")
    bold_fp = _bold_fontproperties(fontsize, weight)
    if artist is None or getattr(artist, "axes", None) is None:
        artist = ax.annotate(
            label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-2.0, pad_pt),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight=weight,
            fontproperties=bold_fp,
            annotation_clip=False,
            clip_on=False,
            zorder=11,
        )
        info["artist"] = artist
    else:
        artist.set_text(label)
        artist.set_fontsize(fontsize)
        artist.set_fontweight(weight)
        try:
            artist.set_fontproperties(bold_fp)
        except Exception:
            pass
    artist.set_visible(True)
    try:
        artist.set_in_layout(True)
    except Exception:
        pass
    return artist


def _sync_panel_labels_impl(fig: Figure) -> None:
    for ax in list(fig.axes):
        info = getattr(ax, _PANEL_LABEL_ATTR, None)
        if not info or info.get("mode") != "outside":
            continue
        label = info["label"]
        fontsize = info["fontsize"]
        title, _loc = _axes_title_text(ax)
        raw = _strip_panel_prefix(title, label) if title else ""

        if raw:
            # Bold label on title baseline; title shifted right (CJK-safe)
            _ensure_outside_label_artist(ax, info)
            cur = (ax.get_title(loc="left") or "").strip().lstrip("\u00a0").strip()
            cur_raw = _strip_panel_prefix(cur, label)
            if cur_raw != raw:
                for L in ("center", "right"):
                    try:
                        if ax.get_title(loc=L):
                            ax.set_title("", loc=L)
                    except TypeError:
                        pass
                ax.set_title(raw, loc="left", fontweight="normal", fontsize=fontsize)
            # Always re-apply x shift (set_title resets position)
            x_off = _title_x_offset_for_label(label, fontsize)
            try:
                ax._left_title.set_x(x_off)
                ax._left_title.set_horizontalalignment("left")
            except Exception:
                pass
            info["folded"] = True
            info["title_x"] = x_off
        else:
            # No title: plain outside label above axes
            info["folded"] = False
            _ensure_outside_label_artist(ax, info)
            try:
                # Restore default left-title x if we had shifted it
                if info.get("title_x") is not None:
                    ax._left_title.set_x(0.0)
                    info["title_x"] = None
            except Exception:
                pass


def _on_panel_label_draw(event) -> None:
    fig = event.canvas.figure
    if fig is None:
        return
    sync_panel_labels(fig)


def _ensure_panel_label_sync(fig: Figure) -> None:
    if getattr(fig, _PANEL_SYNC_CID, None) is not None:
        return
    cid = fig.canvas.mpl_connect("draw_event", _on_panel_label_draw)
    setattr(fig, _PANEL_SYNC_CID, cid)
    _ensure_title_gap_sync(fig)


def _wrap_title_text(text: str) -> str:
    """Insert a line break near the middle of a long single-line title."""
    t = text.strip()
    if "\n" in t or len(t) < 8:
        return t
    # Prefer breaks at punctuation / whitespace near midpoint
    mid = len(t) // 2
    candidates = []
    for i, ch in enumerate(t):
        if ch in "\uff08\uff09()\uff0c\u3001\u2014-\t ":  # incl. full-width CJK punctuation
            candidates.append(i)
    if candidates:
        br = min(candidates, key=lambda i: abs(i - mid))
        # break after opening punct / space; before closing punct
        if t[br] in "\uff09)":
            left, right = t[: br + 1], t[br + 1 :]
        else:
            left, right = t[:br], t[br + (1 if t[br] in " \t" else 0) :]
            if t[br] not in " \t" and t[br] not in "\uff08(":
                left, right = t[: br + 1], t[br + 1 :]
        left, right = left.strip(), right.strip()
        if left and right:
            return f"{left}\n{right}"
    return f"{t[:mid].rstrip()}\n{t[mid:].lstrip()}"


def _title_row_bbox(ax: Axes, renderer) -> Optional[Bbox]:
    """Union bbox of outside panel label + visible title texts for *ax*."""
    boxes = []
    info = getattr(ax, _PANEL_LABEL_ATTR, None)
    if info and info.get("mode") == "outside":
        artist = info.get("artist")
        if artist is not None and artist.get_visible():
            try:
                bb = artist.get_window_extent(renderer)
                if bb.width > 0 and bb.height > 0:
                    boxes.append(bb)
            except Exception:
                pass
    for attr in ("_left_title", "title", "_right_title"):
        t = getattr(ax, attr, None)
        if t is None or not t.get_visible():
            continue
        if not str(t.get_text()).strip():
            continue
        try:
            bb = t.get_window_extent(renderer)
            if bb.width > 0 and bb.height > 0:
                boxes.append(bb)
        except Exception:
            pass
    if not boxes:
        return None
    return Bbox.union(boxes)


def _subplot_row_col(ax: Axes) -> Optional[Tuple[int, int]]:
    ss = ax.get_subplotspec()
    if ss is None:
        return None
    return int(ss.rowspan.start), int(ss.colspan.start)


def ensure_title_gaps(fig: Figure, min_gap_em: float = 1.0) -> bool:
    """Ensure ≥ *min_gap_em* horizontal gap between adjacent panel title rows.

    Wraps long titles and/or bumps constrained-layout ``wspace`` when the
    title/label texts of neighboring columns would collide. Returns True if
    a layout change was applied (caller may need another draw).
    """
    if getattr(fig, "_rp_ensuring_title_gaps", False):
        return False
    ax_list = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    if len(ax_list) < 2:
        return False

    setattr(fig, "_rp_ensuring_title_gaps", True)
    changed = False
    try:
        # Need a renderer; draw once if necessary
        renderer = _renderer(fig)

        by_row: Dict[int, List[Tuple[int, Axes]]] = {}
        for ax in ax_list:
            rc = _subplot_row_col(ax)
            if rc is None:
                continue
            r, c = rc
            by_row.setdefault(r, []).append((c, ax))

        fontsize = _font_pt()
        min_gap_px = min_gap_em * fontsize * (fig.dpi / 72.0)
        need_wspace = False

        for _row, items in by_row.items():
            items = sorted(items, key=lambda t: t[0])
            for i in range(len(items) - 1):
                _c0, ax0 = items[i]
                _c1, ax1 = items[i + 1]
                bb0 = _title_row_bbox(ax0, renderer)
                bb1 = _title_row_bbox(ax1, renderer)
                if bb0 is None or bb1 is None:
                    continue
                gap = bb1.x0 - bb0.x1
                if gap >= min_gap_px:
                    continue
                # First try wrapping long titles that haven't been wrapped yet
                wrapped_any = False
                for ax in (ax0, ax1):
                    if getattr(ax, _TITLE_GAP_ATTR, False):
                        continue
                    title, _loc = _axes_title_text(ax)
                    raw = title
                    info = getattr(ax, _PANEL_LABEL_ATTR, None)
                    if info:
                        raw = _strip_panel_prefix(title, info["label"])
                    if raw and "\n" not in raw and len(raw) >= 8:
                        wrapped = _wrap_title_text(raw)
                        if wrapped != raw:
                            ax.set_title(wrapped, loc="left", fontweight="normal")
                            setattr(ax, _TITLE_GAP_ATTR, True)
                            wrapped_any = True
                            changed = True
                if wrapped_any:
                    sync_panel_labels(fig)
                    continue
                need_wspace = True

        if need_wspace:
            pads = _layout_pads()
            # Bump wspace toward ~1.2 em in figure inches
            bump = max(pads["wspace"], 1.2 * fontsize / 72.0)
            eng = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
            cur = pads["wspace"]
            if eng is not None and hasattr(eng, "set"):
                try:
                    # grow wspace; leave other pads alone
                    eng.set(wspace=max(cur, bump) * (1.0 + 0.35 * getattr(fig, "_rp_wspace_bumps", 0) + 0.35))
                    setattr(fig, "_rp_wspace_bumps", getattr(fig, "_rp_wspace_bumps", 0) + 1)
                    changed = True
                except Exception:
                    pass
            if not changed:
                try:
                    fig.set_constrained_layout_pads(wspace=bump)
                    changed = True
                except Exception:
                    pass
    finally:
        setattr(fig, "_rp_ensuring_title_gaps", False)
    return changed


def _on_title_gap_draw(event) -> None:
    fig = event.canvas.figure
    if fig is None:
        return
    # Limit iterative bumps per figure to avoid infinite draw loops
    if getattr(fig, "_rp_wspace_bumps", 0) >= 4:
        return
    if getattr(fig, "_rp_title_gap_fixes", 0) >= 6:
        return
    if ensure_title_gaps(fig, min_gap_em=1.0):
        setattr(fig, "_rp_title_gap_fixes", getattr(fig, "_rp_title_gap_fixes", 0) + 1)
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass


def _ensure_title_gap_sync(fig: Figure) -> None:
    if getattr(fig, _TITLE_GAP_CID, None) is not None:
        return
    cid = fig.canvas.mpl_connect("draw_event", _on_title_gap_draw)
    setattr(fig, _TITLE_GAP_CID, cid)


def _apply_constrained_pads(fig: Figure) -> None:
    pads = _layout_pads()
    eng = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if eng is not None and hasattr(eng, "set"):
        try:
            eng.set(**pads)
            return
        except Exception:
            pass
    try:
        fig.set_constrained_layout_pads(**pads)
    except Exception:
        pass


def _normalize_axes(axes) -> List[Axes]:
    if isinstance(axes, dict):
        return [ax for ax in axes.values() if isinstance(ax, Axes)]
    if isinstance(axes, Axes):
        return [axes]
    arr = np.atleast_1d(axes)
    return [ax for ax in arr.ravel() if isinstance(ax, Axes)]


def _format_panel_label(index: int, style: str) -> str:
    """index is 0-based."""
    letter = string.ascii_lowercase[index % 26]
    Letter = letter.upper()
    styles = {
        "(a)": f"({letter})",
        "a": letter,
        "A": Letter,
        "a.": f"{letter}.",
        "(A)": f"({Letter})",
    }
    if style not in styles:
        raise ValueError(f"label_style must be one of {list(styles)}; got {style!r}")
    return styles[style]


def add_panel_labels(
    axes,
    style: str = "(a)",
    loc: str = "outside",
    fontsize: Optional[float] = None,
    weight: str = "bold",
    pad_pt: float = 3.0,
) -> List[Any]:
    """Place panel labels outside (default) or inside the axes.

    Default *loc* ``'outside'`` (also accepts ``'upper left'``): bold labels
    above the axes at the top-left corner via ``annotate`` with
    ``xycoords='axes fraction'`` at (0, 1) and a points offset, so they sit
    outside the axes patch (aligned to the left edge, above the frame) and
    participate in constrained layout. They never cover data.

    If an axes later gets a title, a draw-time sync prefixes the panel label
    into the title on the same baseline (bold), avoiding title collisions and
    keeping a top edge legend above the title/label row.

    *loc* ``'inside'``: drawn in the upper-left corner with a white stroke
    outline and a white semi-opaque bbox so labels stay readable over data.
    """
    from matplotlib import patheffects

    ax_list = _normalize_axes(axes)
    if fontsize is None:
        fontsize = _resolve_fontsize(mpl.rcParams.get("axes.titlesize", mpl.rcParams["font.size"]))

    # Normalize aliases
    loc_key = (loc or "outside").lower().replace("_", " ").strip()
    outside = loc_key in ("outside", "upper left", "ul", "top left")
    inside = loc_key in ("inside", "inner")
    if not outside and not inside:
        outside = True  # safe default: never cover data

    bold_fp = _bold_fontproperties(fontsize, weight)
    texts = []
    for i, ax in enumerate(ax_list):
        label = _format_panel_label(i, style)
        if outside:
            # Above axes, left-aligned to axes left edge (tick-label side).
            # Negative x offset nudges into the y-tick-label strip.
            t = ax.annotate(
                label,
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(-2.0, pad_pt),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=fontsize,
                fontweight=weight,
                fontproperties=bold_fp,
                annotation_clip=False,
                clip_on=False,
                zorder=10,
            )
            try:
                t.set_in_layout(True)
            except Exception:
                pass
            setattr(
                ax,
                _PANEL_LABEL_ATTR,
                dict(
                    label=label,
                    mode="outside",
                    fontsize=fontsize,
                    weight=weight,
                    pad_pt=pad_pt,
                    artist=t,
                    folded=False,
                ),
            )
        else:
            t = ax.text(
                0.02,
                0.98,
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight=weight,
                fontproperties=bold_fp,
                ha="left",
                va="top",
                clip_on=False,
                zorder=10,
                color="#333333",
                path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
            try:
                t.set_in_layout(False)  # inside: do not expand layout
            except Exception:
                pass
            setattr(
                ax,
                _PANEL_LABEL_ATTR,
                dict(
                    label=label,
                    mode="inside",
                    fontsize=fontsize,
                    weight=weight,
                    pad_pt=pad_pt,
                    artist=t,
                    folded=False,
                ),
            )
        texts.append(t)

    if ax_list and outside:
        _ensure_panel_label_sync(ax_list[0].figure)
        # Sync once now in case titles already exist
        sync_panel_labels(ax_list[0].figure)
    return texts


def label_outer_shared(axes) -> None:
    """Hide redundant inner tick *labels* (and axis labels) for shared axes.

    Only toggles label visibility on inner panels — never removes ticks at
    the data edge. A post-draw hook restores per-tick label visibility so
    sticky ``set_visible(False)`` state cannot drop edge ticks like 10/12.
    """
    ax_list = _normalize_axes(axes)
    if not ax_list:
        return

    # Prefer matplotlib's built-in when available on a SubplotBase grid
    try:
        for ax in ax_list:
            if hasattr(ax, "label_outer"):
                ax.label_outer()
    except Exception:
        # Manual fallback: hide x labels on non-bottom, y labels on non-left
        for ax in ax_list:
            ss = ax.get_subplotspec()
            if ss is None:
                continue
            if not ss.is_last_row():
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
            if not ss.is_first_col():
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
        if ax_list:
            _ensure_tick_prune(ax_list[0].figure)
        return

    # Ensure axis *labels* (not only ticks) are cleared on inner panels —
    # callers often set xlabel/ylabel after the initial label_outer pass.
    for ax in ax_list:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        if not ss.is_last_row():
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if not ss.is_first_col():
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    # Restore per-tick label visibility after draw (shared-axes sticky state)
    if ax_list:
        _ensure_tick_prune(ax_list[0].figure)
        _restore_shared_tick_labels(ax_list[0].figure)


def _axes_shows_xticklabels(ax: Axes) -> bool:
    ss = ax.get_subplotspec()
    if ss is not None and not ss.is_last_row():
        return False
    params = ax.xaxis.get_tick_params()
    if params.get("labelbottom") is False and params.get("labeltop") is False:
        return False
    return True


def _axes_shows_yticklabels(ax: Axes) -> bool:
    ss = ax.get_subplotspec()
    if ss is not None and not ss.is_first_col():
        return False
    params = ax.yaxis.get_tick_params()
    if params.get("labelleft") is False and params.get("labelright") is False:
        return False
    return True


def _restore_shared_tick_labels(fig: Figure) -> None:
    """Re-show all major tick labels on outer panels; keep inner panels hidden.

    Matplotlib + prior prune hooks can leave individual tick labels stuck
    invisible even when the axes should display the full locator set.
    """
    for ax in list(fig.axes):
        if ax.get_subplotspec() is None:
            continue
        show_x = _axes_shows_xticklabels(ax)
        show_y = _axes_shows_yticklabels(ax)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_visible(show_x)
            if hasattr(tick, "label2"):
                # keep secondary labels off unless the axes explicitly enables them
                if ax.xaxis.get_tick_params().get("labeltop"):
                    tick.label2.set_visible(show_x)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_visible(show_y)
            if hasattr(tick, "label2"):
                if ax.yaxis.get_tick_params().get("labelright"):
                    tick.label2.set_visible(show_y)


def _on_tick_prune_draw(event) -> None:
    fig = event.canvas.figure
    if fig is None:
        return
    _restore_shared_tick_labels(fig)


def _ensure_tick_prune(fig: Figure) -> None:
    if getattr(fig, _TICK_PRUNE_CID, None) is not None:
        return
    cid = fig.canvas.mpl_connect("draw_event", _on_tick_prune_draw)
    setattr(fig, _TICK_PRUNE_CID, cid)


def set_shared_labels(
    fig_or_axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Set one figure-level supxlabel / supylabel; clear per-axes duplicates."""
    if isinstance(fig_or_axes, Figure):
        fig = fig_or_axes
        ax_list = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    else:
        ax_list = _normalize_axes(fig_or_axes)
        if not ax_list:
            return
        fig = ax_list[0].figure

    if xlabel is not None:
        for ax in ax_list:
            ax.set_xlabel("")
        try:
            fig.supxlabel(xlabel)
        except Exception:
            for ax in ax_list:
                if ax.get_subplotspec() and ax.get_subplotspec().is_last_row():
                    ax.set_xlabel(xlabel)
                    break
    if ylabel is not None:
        for ax in ax_list:
            ax.set_ylabel("")
        try:
            fig.supylabel(ylabel)
        except Exception:
            for ax in ax_list:
                if ax.get_subplotspec() and ax.get_subplotspec().is_first_col():
                    ax.set_ylabel(ylabel)
                    break
    # Re-prune outer ticks now that limits/labels are finalized
    if ax_list:
        label_outer_shared(ax_list)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    refwidth_mm: Optional[float] = None,
    figsize_mm: Optional[Tuple[float, float]] = None,
    sharex: _ShareSpec = False,
    sharey: _ShareSpec = False,
    mosaic: Optional[Any] = None,
    panel_labels: bool = True,
    label_style: str = "(a)",
    label_weight: str = "bold",
    **kw,
) -> Tuple[Figure, Any]:
    """Create a constrained-layout figure with optional mosaic and panel labels.

    Parameters
    ----------
    nrows, ncols : grid size when *mosaic* is None.
    refwidth_mm : width of one panel in mm; figure width = ncols * refwidth_mm
        (height from nrows with a mild aspect), used when figsize_mm is None.
    figsize_mm : explicit (width, height) in mm.
    sharex / sharey : False | True | 'row' | 'col' (matplotlib semantics).
    mosaic : passed to Figure.subplot_mosaic when not None (list of lists or string).
    panel_labels : add (a)(b)... style labels.
    label_style : '(a)' | 'a' | 'A' | 'a.' | '(A)'
    **kw : forwarded to plt.subplots / subplot_mosaic (layout forced to constrained).
    """
    kw = dict(kw)
    kw.setdefault("layout", "constrained")

    if figsize_mm is not None:
        figsize = (_mm_to_in(figsize_mm[0]), _mm_to_in(figsize_mm[1]))
    elif refwidth_mm is not None:
        # Near-square panels (matches ultraplot refwidth sizing more closely)
        ref_h = refwidth_mm * 0.95
        if mosaic is not None:
            # Estimate mosaic shape
            if isinstance(mosaic, str):
                rows = [r for r in mosaic.strip().splitlines() if r.strip()]
                nr, nc = len(rows), max(len(r) for r in rows)
            else:
                nr, nc = len(mosaic), max(len(r) for r in mosaic)
        else:
            nr, nc = nrows, ncols
        figsize = (_mm_to_in(refwidth_mm * nc), _mm_to_in(ref_h * nr))
    else:
        # Sensible default: ~55 mm panel width, near-square panels
        rw = 55.0
        figsize = (_mm_to_in(rw * ncols), _mm_to_in(rw * 0.95 * nrows))

    if mosaic is not None:
        mosaic_kw = {k: kw.pop(k) for k in list(kw) if k in ("empty_sentinel", "subplot_kw", "gridspec_kw")}
        # sharex/sharey go into subplot_kw for mosaic
        sk = dict(mosaic_kw.pop("subplot_kw", {}) or {})
        if sharex:
            sk["sharex"] = sharex if sharex in ("row", "col") else "all" if sharex is True else False
        if sharey:
            sk["sharey"] = sharey if sharey in ("row", "col") else "all" if sharey is True else False
        fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize, layout=kw.get("layout", "constrained"), subplot_kw=sk or None, **mosaic_kw)
        _apply_constrained_pads(fig)
        axes_out = axd
        ax_list = _normalize_axes(axd)
        if sharex or sharey:
            label_outer_shared(ax_list)
        if panel_labels:
            add_panel_labels(ax_list, style=label_style, weight=label_weight)
        return fig, axes_out

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        **kw,
    )
    _apply_constrained_pads(fig)
    if sharex or sharey:
        label_outer_shared(axes)
    if panel_labels:
        add_panel_labels(axes, style=label_style, weight=label_weight)
    return fig, axes


def _axes_union_bbox(fig: Figure, axes: Sequence[Axes]) -> Bbox:
    fig.canvas.draw()
    renderer = _renderer(fig)
    bboxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
    bboxes = [b for b in bboxes if b is not None]
    if not bboxes:
        return Bbox.from_bounds(0, 0, 0, 0)
    return Bbox.union(bboxes)


def edge_colorbar(
    fig: Figure,
    mappable,
    axes,
    loc: str = "right",
    label: Optional[str] = None,
    fraction: Optional[float] = None,
    pad: float = 0.02,
    aspect: Optional[float] = None,
    **kw,
):
    """Place a colorbar at the figure edge relative to *axes* (no overlap).

    Uses constrained-layout-friendly colorbar via fig.colorbar with location.
    Horizontal (bottom/top) bars span the served axes; tick count is capped
    with ``MaxNLocator(5)`` so labels stay readable.
    """
    ax_list = _normalize_axes(axes)
    if not ax_list:
        raise ValueError("axes must contain at least one Axes")
    loc = loc.lower()
    if loc not in ("right", "bottom", "left", "top"):
        raise ValueError("loc must be 'right'|'bottom'|'left'|'top'")

    horizontal = loc in ("top", "bottom")
    if fraction is None:
        # Horizontal bars need a larger fraction so aspect×thickness can
        # span the served axes (talk fonts / constrained layout shrink room).
        fraction = 0.14 if horizontal else 0.045
    if aspect is None:
        aspect = 40 if horizontal else 20

    cb_kw = dict(kw)
    cb_kw.setdefault("aspect", aspect)
    cb_kw.setdefault("shrink", 1.0)
    try:
        cbar = fig.colorbar(
            mappable,
            ax=ax_list,
            location=loc,
            fraction=fraction,
            pad=pad,
            **cb_kw,
        )
    except TypeError:
        orientation = "horizontal" if horizontal else "vertical"
        cbar = fig.colorbar(
            mappable,
            ax=ax_list,
            orientation=orientation,
            fraction=fraction,
            pad=pad,
            **cb_kw,
        )
    if label is not None:
        cbar.set_label(label)
    # Readable tick count (avoids '0.02.5' style collisions on short bars)
    try:
        axis = cbar.ax.xaxis if horizontal else cbar.ax.yaxis
        axis.set_major_locator(MaxNLocator(nbins=5))
    except Exception:
        pass
    return cbar


def edge_legend(
    fig: Figure,
    axes,
    loc: str = "right",
    ncols: int = 1,
    **kw,
):
    """Collect legend handles from *axes* and place a figure-edge legend.

    Top legends use matplotlib's ``outside upper center`` so constrained
    layout reserves room *above* panel labels / titles.
    """
    ax_list = _normalize_axes(axes)
    if not ax_list:
        raise ValueError("axes must contain at least one Axes")
    loc = loc.lower()

    handles, labels = [], []
    seen = set()
    for ax in ax_list:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li and li not in seen:
                seen.add(li)
                handles.append(hi)
                labels.append(li)
        # Remove any axes-level legends so only the edge legend remains
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if not handles:
        return None

    # Sync panel labels into titles before placing a top legend so layout
    # reserves a single title row beneath the legend.
    sync_panel_labels(fig)

    legend_kw = dict(kw)
    legend_kw.setdefault("frameon", False)

    # Prefer constrained-layout-aware "outside …" locations (mpl 3.6+)
    outside_map = {
        "right": "outside right center",
        "left": "outside left center",
        "top": "outside upper center",
        "bottom": "outside lower center",
    }
    if loc not in outside_map:
        raise ValueError("loc must be 'right'|'bottom'|'top'|'left'")

    if loc in ("top", "bottom"):
        ncols = max(ncols, len(labels))
    legend_kw.setdefault("ncols", ncols)

    try:
        leg = fig.legend(handles, labels, loc=outside_map[loc], **legend_kw)
    except (ValueError, TypeError):
        # Fallback for older mpl: manual figure-fraction anchor above axes
        fig.canvas.draw()
        bbs = [ax.get_position() for ax in ax_list]
        x0 = min(b.x0 for b in bbs)
        y0 = min(b.y0 for b in bbs)
        x1 = max(b.x1 for b in bbs)
        y1 = max(b.y1 for b in bbs)
        fpad = 0.02 + 0.003 * (_font_pt() / 8.0)
        # Small gap above panel labels / title row for top legends (~1 line)
        label_gap = (0.6 * _font_pt() + 2.0) / (fig.get_figheight() * 72.0)
        if loc == "right":
            legend_kw.update(loc="center left", bbox_to_anchor=(x1 + fpad, (y0 + y1) / 2))
        elif loc == "left":
            legend_kw.update(loc="center right", bbox_to_anchor=(x0 - fpad, (y0 + y1) / 2))
        elif loc == "top":
            legend_kw.update(
                loc="lower center",
                bbox_to_anchor=((x0 + x1) / 2, y1 + fpad + label_gap),
            )
        else:
            legend_kw.update(loc="upper center", bbox_to_anchor=((x0 + x1) / 2, y0 - fpad))
        leg = fig.legend(handles, labels, bbox_transform=fig.transFigure, **legend_kw)

    try:
        leg.set_in_layout(True)
    except Exception:
        pass
    return leg


def bboxes_overlap(a: Bbox, b: Bbox, tol: float = 0.0) -> bool:
    """Utility for tests: True if two bboxes overlap (with optional tolerance shrink)."""
    if a.width <= 0 or a.height <= 0 or b.width <= 0 or b.height <= 0:
        return False
    if tol:
        a = a.shrunk_to_aspect(a.height / a.width).expanded(1 - tol, 1 - tol) if False else Bbox.from_extents(
            a.x0 + tol, a.y0 + tol, a.x1 - tol, a.y1 - tol
        )
        b = Bbox.from_extents(b.x0 + tol, b.y0 + tol, b.x1 - tol, b.y1 - tol)
    return a.overlaps(b)
