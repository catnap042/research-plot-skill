"""Research figure style: ggplot2 theme_bw layout + ggsci palettes for matplotlib.

Usage:
    import style
    style.use("paper", variant="C", palette="npg")   # or style.use_talk() for slides
    fig, ax = style.fig("single")                     # 89 mm; "double" 183 mm
    ax.plot(x, y, **style.measured(0))                # hollow markers, sample colour 0
    style.finish(ax, "Cycle number", "Capacity retention (%)")
    style.save(fig, "out/name")                       # PNG + PDF + SVG (600 dpi paper, 300 dpi talk)
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Neutral UI colours (ink, muted text, light rule) and the two status colours.
NEUTRAL = dict(ink="#333333", secondary="#666666", line="#DDDDDD", orange="#E69F00", green="#009E73")
PALETTES = {}
# --- ggplot2 default hue wheel: scales::hue_pal()(n) = hcl(h = seq(15, 375, length = n + 1)[1:n], c = 100, l = 65)
def _hcl2hex(h, c=100.0, l=65.0):
    import math
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    un = 4 * Xn / (Xn + 15 * Yn + 3 * Zn); vn = 9 * Yn / (Xn + 15 * Yn + 3 * Zn)
    u, v = c * math.cos(math.radians(h)), c * math.sin(math.radians(h))
    Y = Yn * (((l + 16) / 116) ** 3 if l > 8 else l / 903.3)
    up, vp = u / (13 * l) + un, v / (13 * l) + vn
    X = 9 * Y * up / (4 * vp); Z = -X / 3 - 5 * Y + 3 * Y / vp
    X, Y, Z = X / 100, Y / 100, Z / 100
    rgb = (3.240479 * X - 1.537150 * Y - 0.498535 * Z, -0.969256 * X + 1.875992 * Y + 0.041556 * Z,
           0.055648 * X - 0.204043 * Y + 1.057311 * Z)
    g = lambda t: (lambda t: 12.92 * t if t <= 0.0031308 else 1.055 * t ** (1 / 2.4) - 0.055)(min(max(t, 0), 1))
    return "#%02X%02X%02X" % tuple(int(round(255 * g(t))) for t in rgb)

def hue_pal(n):
    """Exact ggplot2 default discrete colours; n=3 -> #F8766D #00BA38 #619CFF."""
    return [_hcl2hex(15 + 360 * i / n) for i in range(n)]

# ggsci palettes: only the colour hex values are copied from the ggsci R package, R/palettes.R (no code)
# (github.com/nanxstats/ggsci, file last changed in commit 9553e2de63, 2026-04-14; fetched 2026-10-15).
# "Inspired by" journal figures; NOT official journal palettes.
NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]  # ggsci_db$npg$nrc
LANCET = ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"]        # ggsci_db$lancet$lanonc
JCO = ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990"]  # ggsci_db$jco$default
PALETTES["lancet"] = LANCET
PALETTES["jco"] = JCO
PALETTES["npg"] = NPG
PALETTES["hue"] = hue_pal(3)   # replaced in use() by hue_pal(n_colors)
GG_GRADIENT = ["#132B43", "#56B1F7"]   # ggplot2 scale_colour_gradient() default low/high

STATUS = {  # status = colour + explicit text; never reuse as a sample colour meaning
    "hue":      dict(pending=NEUTRAL["orange"], passed=NEUTRAL["green"]),
    "npg":      dict(pending="#DC0000", passed="#00A087"),
    "lancet":   dict(pending="#ED0000", passed="#42B540"),
    "jco":      dict(pending="#CD534C", passed="#0073C2"),
}
MARKERS = ["o", "s", "^", "D", "v"]          # circle -> square -> triangle -> diamond -> down-triangle
CMAP_SEQ, CMAP_DIV = "cividis", "RdBu_r"
FIGSIZE_MM = {"single": (89, 64), "double": (183, 80), "slide": (240, 135)}   # single / double journal column

_state = {"mode": "paper", "palette": "npg", "variant": "C"}

# Named style variants (same layout/data; differ only in frame, ticks, grid, weights, palette).
VARIANTS = {
    "A": dict(name="v0.2 L-axes",        palette="npg", spines="L",   ticks="out", grid=None,   scale=1.0),
    "B": dict(name="Boxed journal",      palette="npg", spines="box", ticks="in",  grid=None,   scale=1.0),
    "C": dict(name="ggplot2 theme_bw",   palette="npg", spines="box", ticks="out", grid="both", scale=1.0),
    "D": dict(name="Nature minimal",     palette="npg", spines="L",   ticks="out", grid=None,   scale=0.8),
}
_FONT_PREF = ["Arial", "Helvetica", "Arimo", "Liberation Sans", "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"]
# A family counts as CJK-capable only if its font file has glyphs for ALL of these
# common Simplified Chinese characters (plus full-width punctuation). The registered
# name depends on platform and matplotlib version (matplotlib < 3.11 registers only
# the first face of a .ttc, e.g. "Noto Sans CJK JP" not "... SC"; macOS ships
# Hiragino / Heiti / Songti rather than a findable "PingFang SC"), and display fonts
# with partial CJK coverage exist, so neither names nor one test glyph are enough.
_CJK_PROBE = (
    "\u4eec\u8fd9\u8bf4\u4e3a\u53d1\u5bf9\u5b66\u56fe\u6570\u636e"          # common words
    "\u7535\u89e3\u6db2\u5faa\u73af\u6b21\u6027\u80fd\u6bd4\u5bb9\u91cf\u4fdd"  # test labels
    "\u6301\u7387\u76f8\u9996\u5708\u5408\u6210\u6e29\u5ea6\u654f\u611f"
    "\uff08\uff09\uff0c\u3001"
)
_cjk_cache = {}


def _is_cjk_family(name):
    if name not in _cjk_cache:
        fm = mpl.font_manager
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            cmap = mpl.ft2font.FT2Font(path).get_charmap()
            _cjk_cache[name] = all(ord(c) in cmap for c in _CJK_PROBE)
        except Exception:
            _cjk_cache[name] = False
    return _cjk_cache[name]


_SC_TAGS = ("SC", "GB", "CN", "Simplified", "Hans")
_SANS_TAGS = ("Sans CJK", "Sans", "Hei", "Gothic", "YaHei")
_SKIP_TAGS = ("Mono", "Display", "One", "Rounded", "Serif", "Mincho", "Song", "Ming", "Kai")


def _cjk_candidate_order(names):
    """Fallback probe order: Simplified-Chinese sans names first, then other SC names,
    then other sans. Families whose name marks them as neither SC nor sans (e.g.
    calligraphy), and monospace / display / serif families, are dropped; glyph
    coverage still decides among the rest."""
    def tier(n):
        words = n.replace("-", " ").split()
        sc = any(t in words or t in n for t in _SC_TAGS[2:]) or any(t in words for t in _SC_TAGS[:2])
        sans = any(t in n for t in _SANS_TAGS)
        return (0 if sc and sans else 1 if sc else 2 if sans else 3, n)
    keep = [n for n in names if not any(t in n for t in _SKIP_TAGS) and tier(n)[0] < 3]
    return sorted(keep, key=tier)


def _fonts():
    have = {f.name for f in mpl.font_manager.fontManager.ttflist}
    fams = [f for f in _FONT_PREF if f in have]
    if not any(_is_cjk_family(f) for f in fams):
        hit = next((n for n in _cjk_candidate_order(have) if _is_cjk_family(n)), None)
        if hit:
            fams.insert(len(fams) - ("DejaVu Sans" in fams), hit)
    return fams or ["DejaVu Sans"]


def cjk_family():
    """Name of the CJK-capable family in the active font.family list, or None."""
    fam = mpl.rcParams.get("font.family", [])
    fam = [fam] if isinstance(fam, str) else list(fam)
    return next((f for f in fam if _is_cjk_family(f)), None)

# ---- Talk preset (theme_bw + npg, enlarged for 16:9 slides, figure ~ half slide width) ----
# 16:9 slide = 13.33 in (338.7 mm) wide -> half width ≈ 165 mm. Sizes in pt.
TALK = dict(tick=18, label=20, title=20, legend=16, frame_lw=1.2, line_lw=2.5, marker=8,
               grid_lw=0.6, grid_color="#E1E1E1", tick_w=1.2, tick_len=5, mew=1.8, ref_lw=1.6)
FIGSIZE_MM_TALK = {"single": (165, 120), "double": (330, 140), "slide": (330, 185)}

def use_talk(frame_lw=None, **overrides):
    """Talk preset: variant C + npg with slide-sized type and weights. frame_lw overrides the 1.2 pt border."""
    use("paper", palette="npg", variant="C")
    t = dict(TALK, **overrides)
    if frame_lw is not None: t["frame_lw"] = t["tick_w"] = frame_lw   # ticks follow the frame weight
    mpl.rcParams.update({
        "font.size": t["legend"], "xtick.labelsize": t["tick"], "ytick.labelsize": t["tick"],
        "axes.labelsize": t["label"], "axes.titlesize": t["title"], "legend.fontsize": t["legend"],
        "axes.linewidth": t["frame_lw"], "lines.linewidth": t["line_lw"], "lines.markersize": t["marker"],
        "xtick.major.width": t["tick_w"], "ytick.major.width": t["tick_w"],
        "xtick.major.size": t["tick_len"], "ytick.major.size": t["tick_len"],
        "xtick.minor.width": t["tick_w"] * 0.7, "ytick.minor.width": t["tick_w"] * 0.7,
        "grid.linewidth": t["grid_lw"], "grid.color": t["grid_color"],
        "axes.edgecolor": "#333333", "polaraxes.grid": True,
    })
    _state.update(talk=t)

def use(mode="paper", palette=None, variant="C", n_colors=3):
    """variant A (L-axes) | B (boxed, ticks in) | C (theme_bw, default) | D (minimal);
    palette: npg (default) | hue | jco | lancet.
    n_colors: number of discrete groups, used by the hue palette (ggplot2 picks colours from n)."""
    import os
    if os.environ.get("PLOT_PRESET") == "talk" and not _state.get("_in_talk"):
        _state["_in_talk"] = True
        try: use_talk(frame_lw=float(os.environ["PLOT_FRAME_LW"]) if os.environ.get("PLOT_FRAME_LW") else None)
        finally: _state["_in_talk"] = False
        return
    _state.pop("talk", None)
    v = VARIANTS[variant]
    palette = palette or os.environ.get("PLOT_PALETTE") or v["palette"]
    if palette == "hue":
        PALETTES["hue"] = hue_pal(n_colors)
    if mode not in ("paper", "talk") or palette not in PALETTES:
        raise ValueError("mode: paper|talk; palette: npg|hue|jco|lancet")
    plt.style.use(HERE / f"{mode}.mplstyle")
    mpl.rcParams.update({
        "font.family": _fonts(),                       # explicit list -> per-glyph CJK fallback
        "axes.prop_cycle": mpl.cycler(color=PALETTES[palette]),
        "axes.unicode_minus": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "normal",
        "legend.loc": "best",
        "savefig.bbox": "tight",
        "xtick.major.width": mpl.rcParams["axes.linewidth"],
        "ytick.major.width": mpl.rcParams["axes.linewidth"],
        "xtick.major.size": 3 if mode == "paper" else 5,
        "ytick.major.size": 3 if mode == "paper" else 5,
    })
    k = v["scale"]
    if k != 1.0:   # D: smaller type, thinner lines (text 5-7 pt)
        for key in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
                    "legend.fontsize", "lines.markersize", "axes.linewidth", "xtick.major.size", "ytick.major.size"):
            mpl.rcParams[key] = mpl.rcParams[key] * k
        mpl.rcParams["lines.linewidth"] = 0.9
        for key in ("xtick.major.width", "ytick.major.width"):
            mpl.rcParams[key] = mpl.rcParams["axes.linewidth"]
    # Frame follows the variant at rcParams level too, so axes that never pass through finish()
    # (helpers, insets, facets) still get the full theme_bw box for variants B/C.
    box = v["spines"] == "box"
    mpl.rcParams.update({"axes.spines.top": box, "axes.spines.right": box})
    if v["grid"] == "both":
        mpl.rcParams.update({"axes.grid": True, "grid.color": "#E1E1E1", "grid.linewidth": 0.5, "axes.axisbelow": True})
    if v["ticks"] == "in":
        mpl.rcParams.update({"xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    _state.update(mode=mode, palette=palette, variant=variant)

def color(i):   p = PALETTES[_state["palette"]]; return p[i % len(p)]
def status(k):  return STATUS[_state["palette"]][k]

def measured(i, **kw):
    """Discrete measured data: hollow markers with a transparent centre."""
    mew = (0.9 if _state["mode"] == "paper" else 1.4) * VARIANTS[_state["variant"]]["scale"]
    if _state.get("talk"): mew = _state["talk"]["mew"]
    return dict(ls="none", marker=MARKERS[i % 5], mfc="none", mec=color(i), mew=mew, color=color(i), **kw)

def seq_cmap():
    """Sequential colormap for ordered variables (cycle number, temperature, SOC)."""
    if _state["palette"] == "hue":
        return mpl.colors.LinearSegmentedColormap.from_list("gg_gradient", GG_GRADIENT)
    if _state["palette"] == "npg":
        return mpl.colors.LinearSegmentedColormap.from_list("npg_seq", ["#3C5488", "#4DBBD5", "#91D1C2"])
    return mpl.colormaps[CMAP_SEQ]

def model(i, **kw):      return dict(ls="-", color=color(i), **kw)
def forecast(i, **kw):   return dict(ls="--", color=color(i), **kw)
def reference(**kw):
    return dict(ls=":", color=NEUTRAL["secondary"], lw=_state["talk"]["ref_lw"] if _state.get("talk") else 0.8, **kw)

def fig(size="single", nrows=1, ncols=1, **kw):
    w, h = (FIGSIZE_MM_TALK if _state.get("talk") else FIGSIZE_MM)[size]
    return plt.subplots(nrows, ncols, figsize=(w / 25.4, h / 25.4), layout="constrained", **kw)

def finish(ax, xlabel=None, ylabel=None):
    """Apply the variant's frame, ticks and grid to one axes; drop the legend frame."""
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    v = VARIANTS[_state["variant"]]
    box = v["spines"] == "box"
    ax.spines[["top", "right"]].set_visible(box)
    if v["ticks"] == "in":
        ax.tick_params(direction="in", top=True, right=True, which="both")
        ax.minorticks_on()
        ax.tick_params(which="minor", length=1.5, width=mpl.rcParams["axes.linewidth"] * 0.8)
    else:
        ax.tick_params(direction="out")
        if _state.get("talk"):
            t = _state["talk"]; ax.tick_params(width=t["tick_w"], length=t["tick_len"])
    ax.grid(False)
    if v["grid"] == "both":
        t = _state.get("talk")
        ax.grid(True, color="#E1E1E1", linewidth=t["grid_lw"] if t else 0.5)
        ax.tick_params(colors="#4D4D4D")
    elif _state["mode"] == "talk":
        ax.grid(axis="y", color=NEUTRAL["line"], linewidth=0.6)
    ax.set_axisbelow(True)
    leg = ax.get_legend()
    if leg: leg.set_frame_on(False)
    return ax

def synthetic_tag(fig, text="Example: synthetic data"):
    """Below the figure, bottom-right (bbox=tight keeps it); never collides with title, labels or data."""
    fig.text(0.995, -0.01, text, ha="right", va="top", color=NEUTRAL["secondary"],
             fontsize=mpl.rcParams["legend.fontsize"])

def save(fig, stem, formats=("png", "pdf", "svg"), dpi=None):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(stem.with_suffix("." + ext), dpi=dpi or (300 if (_state["mode"] == "talk" or _state.get("talk")) else 600), facecolor="white")
    plt.close(fig)

def title(ax, text, letter=None):
    """Panel title. Variant D (Nature minimal) keeps only the bold panel letter; others show letter + text."""
    if _state["variant"] == "D":
        if letter: ax.set_title(letter, fontweight="bold", loc="left")
        return
    ax.set_title(f"({letter}) {text}" if letter else text)
