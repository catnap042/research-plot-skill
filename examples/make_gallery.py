"""9-chart gallery in five style sets. Same synthetic data, same 9 charts; matplotlib only.

Usage:
    python examples/make_gallery.py                 # all sets -> docs/images/gallery/
    python examples/make_gallery.py bw talk --out OUT
All numbers below are synthetic and illustrative only.
"""
import sys
import pathlib
import argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "research-data-plotting" / "scripts"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont
import style

SETS = {
    "bw": dict(name="theme_bw + ggsci npg (default)", frame="bw", pal=style.NPG, direct=False),
    "minimal": dict(name="Minimal flat, direct labels", frame="flat", pal=style.NPG, direct=True),
    "consulting": dict(name="Consulting: one accent (#3C5488)", frame="consult", pal=None, direct=True, accent="#3C5488"),
    "jco": dict(name="ggsci jco, soft", frame="soft", pal=style.JCO, direct=False),
    "talk": dict(name="theme_bw + npg, talk preset (half 16:9 slide)", frame="bw", pal=style.NPG, direct=False, talk=True),
}
OUT = ROOT / "docs" / "images" / "gallery"
GREYS = ["#8C8C8C", "#A6A6A6", "#BFBFBF", "#D0D0D0", "#7A7A7A", "#B3B3B3"]
INK, MUTED = "#333333", "#666666"
SIZE = (100, 80)
CFG = {}

# ---------------- set-aware helpers ----------------
def col(i, n=None, key=None):
    """Colour for category i. Consulting: accent only for the key category, greys for the rest."""
    if CFG["frame"] == "consult":
        return CFG["accent"] if i == key else GREYS[i % len(GREYS)]
    return CFG["pal"][i % len(CFG["pal"])]

def FS(x):   # inline annotation sizes: x1 paper, ~x2.2 talk (6.5 pt -> ~14 pt)
    return x * (2.2 if CFG.get("talk") else 1)
def LW(x):
    return x * (2.0 if CFG.get("talk") else 1)

def alpha_fill():
    return 0.55 if CFG["frame"] == "soft" else 0.85

def new(polar=False):
    fr = CFG["frame"]
    if CFG.get("talk"): style.use_talk()
    else: style.use("paper", palette="npg", variant="C" if fr == "bw" else "A")
    if fr == "soft":
        mpl.rcParams.update({"axes.edgecolor": "#9AA3AD", "xtick.color": "#5B6570", "ytick.color": "#5B6570"})
    sz = (165, 125) if CFG.get("talk") else SIZE
    fig = plt.figure(figsize=(sz[0] / 25.4, sz[1] / 25.4), layout="constrained")
    return fig, fig.add_subplot(projection="polar" if polar else None)

def frame(ax, axes=True):
    fr = CFG["frame"]
    if not axes: return
    if fr == "bw":
        style.finish(ax)
    elif fr == "flat":
        for s in ax.spines.values(): s.set_visible(False)
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color("#BFBFBF")
        ax.tick_params(left=False, colors=MUTED); ax.grid(False)
    elif fr == "consult":
        ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#BFBFBF")
        ax.tick_params(colors=MUTED); ax.grid(False)
    elif fr == "soft":
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6E9ED", lw=LW(0.6), ls="--")
        ax.set_axisbelow(True)

def legend(ax, **kw):
    if not CFG["direct"]:
        ax.legend(frameon=False, **kw)

def done(fig, ax, stem, en, axes=True, subtitle=None, pad=None):
    frame(ax, axes)
    ax.set_title(en if not subtitle else f"{en}\n{subtitle}", loc="left",
                 color=INK, fontsize=mpl.rcParams["axes.titlesize"], pad=pad if pad else 6)
    style.synthetic_tag(fig)
    out = OUT / CFG["id"] / stem
    style.save(fig, out, formats=("png",)); CHARTS.append((stem + ".png", en.split(":")[0]))

def lighten(c, f):
    r, g, b = mpl.colors.to_rgb(c); return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

# ---------------- the 9 charts (data identical across sets) ----------------
def charts():
    rng = np.random.default_rng(42)
    consult = CFG["frame"] == "consult"

    # 1 Donut
    labels = ["Li$_2$CO$_3$", "FePO$_4$ precursor", "Energy", "Carbon & additives", "Labour", "Depreciation"]
    vals = np.array([48, 20, 9, 5, 7, 11.0]); key = 0
    fig, ax = new()
    w, _ = ax.pie(vals, colors=[col(i, 6, key) for i in range(6)], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1, alpha=alpha_fill()))
    for i, (wedge, lab, v) in enumerate(zip(w, labels, vals)):
        a = np.deg2rad((wedge.theta1 + wedge.theta2) / 2); x, y = np.cos(a), np.sin(a)
        ax.annotate(f"{lab}\n{v:.0f}%", (0.81 * x, 0.81 * y),
                    xytext=((1.22 if CFG.get("talk") else 1.24) * x, (1.30 if CFG.get("talk") else 1.18) * y),
                    ha=("center" if not CFG.get("talk") or abs(x) < 0.25 else ("left" if x > 0 else "right")), va="center",
                    color=CFG.get("accent") if consult and i == key else INK,
                    fontweight="bold" if consult and i == key else "normal",
                    arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=LW(0.5)))
    ax.text(0, 0, "LFP cathode\ncost = 100%", ha="center", va="center", color=INK); ax.set_aspect("equal")
    if CFG.get("talk"): ax.set_ylim(-1.45, 1.50); ax.set_xlim(-1.75, 1.75)
    done(fig, ax, "01_donut_cost", "Donut: LFP cathode cost breakdown", axes=False,
         subtitle="Li$_2$CO$_3$ is ~half of cost" if consult else None, pad=34 if CFG.get("talk") else None)

    # 2 Sunburst
    inner = {"Mechanical": {"Particle cracking": 14, "Electrode delamination": 10, "Separator damage": 6},
             "Chemical": {"Fe dissolution": 18, "SEI growth (anode)": 26, "Electrolyte oxidation": 8},
             "Process": {"Moisture": 12, "Metal contamination": 16, "Coating defect": 10}}
    fig, ax = new(); key = 1; cols = [col(i, 3, key) for i in range(3)]
    iv = [sum(d.values()) for d in inner.values()]
    ax.pie(iv, radius=0.62, colors=cols, startangle=90, counterclock=False,
           labels=[f"{k}\n{v}" for k, v in zip(inner, iv)], labeldistance=0.45,
           wedgeprops=dict(width=0.62, edgecolor="white", linewidth=1, alpha=alpha_fill()),
           textprops=dict(color="white", fontsize=FS(6.5)))
    ov, oc, ol, ok = [], [], [], []
    for gi, (c, d) in enumerate(zip(cols, inner.values())):
        for j, (k, v) in enumerate(d.items()):
            ov.append(v); oc.append(lighten(c, 0.25 + 0.2 * j)); ol.append(f"{k} {v}"); ok.append(gi == key and j == 1)
    w, _ = ax.pie(ov, radius=1.0, colors=oc, startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1, alpha=alpha_fill()))
    for wedge, lab, isk in zip(w, ol, ok):
        a = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        ax.text(1.13 * np.cos(a), 1.13 * np.sin(a), lab, ha="left" if np.cos(a) > 0 else "right", va="center",
                fontsize=FS(6), color=CFG["accent"] if consult and isk else INK, fontweight="bold" if consult and isk else "normal")
    ax.set_aspect("equal")
    done(fig, ax, "02_sunburst_failure", "Sunburst: failure modes, n = 120 cells", axes=False,
         subtitle="Chemical modes dominate (52/120)" if consult else None)

    # 3 Sankey
    fig, ax = new()
    nodes = {"Spent LFP cells": (0, 0, 100), "Black mass": (1, 38, 62), "Cu/Al foil": (1, 18, 16),
             "Casing & plastics": (1, 4, 12), "Electrolyte/loss": (1, -8, 10),
             "Li$_2$CO$_3$ (recovered)": (2, 82, 18), "FePO$_4$ (recovered)": (2, 56, 24), "Graphite": (2, 38, 14), "Residue": (2, 24, 6)}
    flows = [("Spent LFP cells", "Electrolyte/loss", 10), ("Spent LFP cells", "Casing & plastics", 12),
             ("Spent LFP cells", "Cu/Al foil", 16), ("Spent LFP cells", "Black mass", 62),
             ("Black mass", "Residue", 6), ("Black mass", "Graphite", 14),
             ("Black mass", "FePO$_4$ (recovered)", 24), ("Black mass", "Li$_2$CO$_3$ (recovered)", 18)]
    mids = ["Black mass", "Cu/Al foil", "Casing & plastics", "Electrolyte/loss"]
    fc = {m: col(i, 4, 0) for i, m in enumerate(mids)}
    if consult: fc.update({m: GREYS[2] for m in mids}); fc["Black mass"] = GREYS[0]
    keyflow = "Li$_2$CO$_3$ (recovered)"
    oo = {k: 0 for k in nodes}; io = {k: 0 for k in nodes}; W = 0.06
    for a, b, v in flows:
        xa, ya, _ = nodes[a]; xb, yb, _ = nodes[b]
        y0, y1 = ya + oo[a], yb + io[b]; oo[a] += v; io[b] += v
        x0, x1 = xa + W, xb - W; xm = (x0 + x1) / 2
        verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + v), (xm, y1 + v), (xm, y0 + v), (x0, y0 + v), (x0, y0)]
        codes = [MPath.MOVETO] + [MPath.CURVE4] * 3 + [MPath.LINETO] + [MPath.CURVE4] * 3 + [MPath.CLOSEPOLY]
        c = fc.get(b, GREYS[1]) if a == "Spent LFP cells" else fc[a]
        if consult and b == keyflow: c = CFG["accent"]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=c, alpha=0.40 if CFG["frame"] == "soft" else 0.5, lw=LW(0)))
    for k, (x, y, h) in nodes.items():
        c = fc.get(k, "#7F7F7F")
        if consult and k == keyflow: c = CFG["accent"]
        ax.add_patch(Rectangle((x - W, y), 2 * W, h, fc=c, lw=LW(0)))
        ha, dx = ("right", -W - 0.03) if x == 0 else ("left", W + 0.03)
        isk = consult and k == keyflow
        ax.text(x + dx, y + h / 2, f"{k}\n{h:.0f}%", ha=ha, va="center", fontsize=FS(6.5),
                color=CFG["accent"] if isk else INK, fontweight="bold" if isk else "normal")
    ax.set(xlim=(-0.75, 2.85), ylim=(-12, 104)); ax.axis("off")
    done(fig, ax, "03_sankey_recycling", "Sankey: LFP recycling mass flow (%)", axes=False,
         subtitle="18% recovered as Li$_2$CO$_3$" if consult else None)

    # 4 Stacked bar
    steps = ["Mixing", "Coating", "Calendering", "Slitting", "Assembly", "Formation"]
    loss = np.array([[0.4, 1.8, 0.7, 0.5, 1.2, 0.9], [0.3, 1.1, 0.9, 0.4, 0.8, 0.7], [0.6, 2.4, 0.6, 0.7, 1.5, 1.3]])
    fig, ax = new(); lines = ["Line 1", "Line 2", "Line 3"]; bottom = np.zeros(3); key = 1
    for j, s in enumerate(steps):
        ax.bar(range(3), loss[:, j], bottom=bottom, color=col(j, 6, key), label=s, width=0.55,
               edgecolor="white", lw=LW(0.6), alpha=alpha_fill())
        if CFG["direct"]:
            ax.text(2.33, bottom[2] + loss[2, j] / 2, s, va="center", fontsize=FS(6),
                    color=CFG["accent"] if consult and j == key else MUTED, fontweight="bold" if consult and j == key else "normal")
        bottom += loss[:, j]
    for i, t in enumerate(bottom): ax.text(i, t + 0.1, f"{t:.1f}%", ha="center", va="bottom", color=INK)
    ax.set_xticks(range(3), lines); ax.set_xlim(-0.5, 3.2 if CFG["direct"] else 2.5); ax.set_ylim(0, 9)
    legend(ax, ncol=2, loc="upper left", fontsize=FS(6.5))
    ax.set_ylabel("Cumulative yield loss (%)")
    done(fig, ax, "04_stacked_bar_yield", "Stacked bar: yield loss by process step",
         subtitle="Coating is the largest loss on every line" if consult else None)

    # 5 Heatmap
    T = [25, 35, 45, 55, 60]; SOC = [0, 30, 50, 70, 100]
    Z = np.array([[99.5 - 0.004 * (t - 25) ** 2 * (1 + s / 60) for s in SOC] for t in T])
    fig, ax = new()
    if consult:
        cmap = mpl.colors.LinearSegmentedColormap.from_list("g", ["#F2F2F2", "#9A9A9A"])
    elif CFG["frame"] == "soft":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("j", ["#003C67", "#7AA6DC", "#EAF1FA"])
    else:
        cmap = mpl.colors.LinearSegmentedColormap.from_list("n", ["#3C5488", "#4DBBD5", "#E8F6F3"])
    im = ax.imshow(Z, cmap=cmap, aspect="auto", origin="lower")
    for i in range(5):
        for j in range(5):
            isk = consult and i == 4 and j == 4
            if isk: ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=CFG["accent"], lw=LW(1.6)))
            dark = (Z[i, j] < 93) and not consult
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=FS(6.5),
                    color=CFG["accent"] if isk else ("white" if dark else INK), fontweight="bold" if isk else "normal")
    ax.set_xticks(range(5), [f"{s}%" for s in SOC]); ax.set_yticks(range(5), [f"{t} °C" for t in T])
    ax.set(xlabel="Storage SOC", ylabel="Storage temperature")
    if not CFG["direct"]: fig.colorbar(im, ax=ax, label="Retention after 90 d (%)")
    frame(ax); ax.grid(False)
    if CFG["frame"] == "flat": ax.spines["bottom"].set_visible(False); ax.tick_params(bottom=False)
    done(fig, ax, "05_heatmap_storage", "Heatmap: retention after 90-day storage (%)", axes=False,
         subtitle="Worst: 60 °C / 100% SOC" if consult else None)

    # 6 Radar (fixed: labels pushed out, aligned by angle, no clash with outer ring)
    props = ["Capacity", "Rate 5C", "Low-T (−20 °C)", "Cycle life", "Tap density", "Cost (inv.)"]
    S = {"Ref": [0.70, 0.55, 0.40, 0.75, 0.60, 0.80], "Nano-C coat": [0.78, 0.82, 0.62, 0.80, 0.50, 0.55],
         "Mn-doped": [0.74, 0.65, 0.55, 0.68, 0.72, 0.70]}
    fig, ax = new(polar=True); key = 1
    ang = np.linspace(0, 2 * np.pi, len(props), endpoint=False); angc = np.r_[ang, ang[0]]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for i, (k, v) in enumerate(S.items()):
        c = col(i, 3, key); v = np.r_[v, v[0]]
        ax.plot(angc, v, color=c, label=k, marker=style.MARKERS[i], ms=LW(3), mfc="none",
                lw=LW(1.6) if consult and i == key else 1.0)
        ax.fill(angc, v, color=c, alpha=0.18 if not consult or i == key else 0.05)
        if CFG["direct"]:   # coloured text key instead of a boxed legend
            ax.text(-0.30, 0.02 + 0.07 * (2 - i), k, transform=ax.transAxes, fontsize=FS(6.5), color=c,
                    fontweight="bold" if consult and i == key else "normal")
    ax.set_xticks(ang); ax.set_xticklabels([])
    for a, lab in zip(ang, props):                     # manual labels outside the ring
        x = np.sin(a); ha = "center" if abs(x) < 0.1 else ("left" if x > 0 else "right")
        ax.text(a, 1.16, lab, ha=ha, va="center", fontsize=FS(6.5), color=INK)
    ax.set_ylim(0, 1); ax.set_yticks([0.5, 1.0], ["0.5", "1.0"], fontsize=FS(5.5), color=MUTED)
    ax.set_rlabel_position(90 / 2)
    ax.grid(color="#E1E1E1", lw=LW(0.5)); ax.spines["polar"].set_color("#9A9A9A"); ax.spines["polar"].set_linewidth(LW(0.6))
    legend(ax, loc="lower left", bbox_to_anchor=(-0.36, -0.06) if not CFG.get("talk") else (-0.50, -0.20), fontsize=FS(6.5))
    done(fig, ax, "06_radar_properties", "Radar: normalised property profile", axes=False,
         subtitle="Nano-C coat leads on rate & low-T" if consult else None, pad=44 if CFG.get("talk") else 26)

    # 7 Violin
    D = {"Supplier A": rng.normal(1.2, 0.15, 40), "Supplier B": np.r_[rng.normal(1.0, 0.1, 30), rng.normal(1.5, 0.08, 10)],
         "Supplier C": rng.lognormal(np.log(1.3), 0.2, 40)}
    fig, ax = new(); key = 1
    parts = ax.violinplot(list(D.values()), showmedians=False, showextrema=False, widths=0.75)
    for i, b in enumerate(parts["bodies"]):
        c = col(i, 3, key); b.set_facecolor(c); b.set_alpha(0.30 if CFG["frame"] == "soft" else 0.38); b.set_edgecolor(c)
    for i, v in enumerate(D.values(), 1):
        c = col(i - 1, 3, key)
        ax.plot(i + rng.uniform(-0.08, 0.08, v.size), v, ls="none", marker=style.MARKERS[i - 1], ms=LW(2.5),
                mfc="none", mec=c, mew=0.8)
        ax.plot([i - 0.2, i + 0.2], [np.median(v)] * 2, color=INK, lw=LW(1.2))
        if CFG["direct"]: ax.text(i + 0.24, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=FS(6), color=MUTED)
    if consult: ax.annotate("bimodal", (2.15, 1.5), xytext=(2.45, 1.68), color=CFG["accent"], fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=CFG["accent"], lw=LW(0.7)))
    ax.set_xticks([1, 2, 3], [f"{k}\nn={len(v)}" for k, v in D.items()]); ax.set_ylabel("D50 (µm)")
    done(fig, ax, "07_violin_d50", "Violin: D50 by supplier",
         subtitle="Supplier B is bimodal" if consult else None)

    # 8 Waffle
    cats = {"Coating defect": 7, "Fe dissolution": 5, "Pass": 82, "Li plating": 4, "Other": 2}
    fig, ax = new(); key = 0
    if CFG["pal"] is not None:   # semantic: Pass = neutral/positive colour from the palette
        cols = [col(0), col(1), col(2), col(3), col(4)]
    else:
        cols = [CFG["accent"], GREYS[0], "#E3E3E3", GREYS[1], GREYS[2]]
    seq = []
    for i, (k, v) in enumerate(cats.items()): seq += [i] * v
    for idx, ci in enumerate(seq):
        r, c = divmod(idx, 10)
        ax.add_patch(Rectangle((c, 9 - r), 0.86, 0.86, fc=cols[ci], lw=LW(0), alpha=alpha_fill() + 0.1))
    ax.set(xlim=(-0.2, 10), ylim=(-0.2, 10)); ax.set_aspect("equal"); ax.axis("off")
    if CFG["direct"]:
        # label right of grid, aligned to where each category sits
        for ci, ((k, v), y) in enumerate(zip(cats.items(), [9.43, 8.43, 5.0, 1.15, 0.2])):
            ax.text(10.3, y, f"{k} {v}", va="center", fontsize=FS(6.5), color=cols[ci] if cols[ci] != "#E3E3E3" else MUTED,
                    fontweight="bold" if consult and ci == key else "normal")
        ax.set_xlim(-0.2, 14.5)
    else:
        ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=c) for c in cols], labels=[f"{k} {v}" for k, v in cats.items()],
                  loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    done(fig, ax, "08_waffle_outcomes", "Waffle: outcomes of 100 aged cells", axes=False,
         subtitle="Coating defect is the top failure" if consult else None)

    # 9 Waterfall
    steps9 = [("2025 cost", 62, "total"), ("Li$_2$CO$_3$ price", -9, "d"), ("Yield +2 pp", -3, "d"), ("Energy", 1.5, "d"),
              ("Carbon coat", 1.0, "d"), ("Scale", -2.5, "d"), ("2026 cost", None, "total")]
    fig, ax = new(); run = 0
    up, down = (GREYS[2], GREYS[0]) if consult else (col(0), col(1))
    for i, (lab, v, kind) in enumerate(steps9):
        if kind == "total":
            v = run if v is None else v; ax.bar(i, v, color="#7F7F7F" if not consult else "#5A5A5A", width=0.6); run = v; y = v
        else:
            c = up if v > 0 else down
            if consult and i == 1: c = CFG["accent"]
            ax.bar(i, v, bottom=run, color=c, width=0.6, alpha=alpha_fill() + 0.1); run += v; y = max(run, run - v)
        ax.text(i, y + 0.6, f"{v:+.1f}" if kind != "total" else f"{v:.1f}", ha="center", va="bottom", fontsize=FS(6.5),
                color=CFG["accent"] if consult and i == 1 else INK)
        if i < len(steps9) - 1: ax.plot([i + 0.3, i + 0.7], [run, run], color="#9A9A9A", lw=LW(0.5), ls=":")
    ax.set_xticks(range(len(steps9)), [s[0] for s in steps9], rotation=30, ha="right")
    ax.set(ylim=(40, 68), ylabel="Cell cost ($/kWh)")
    ax.text(0.99, 0.97, "axis starts at 40 $/kWh", transform=ax.transAxes, ha="right", va="top", fontsize=FS(6), color=MUTED)
    if not consult:
        ax.text(0.99, 0.91, "▲ cost up", transform=ax.transAxes, ha="right", va="top", fontsize=FS(6), color=up)
        ax.text(0.99, 0.86, "▼ cost down", transform=ax.transAxes, ha="right", va="top", fontsize=FS(6), color=down)
    done(fig, ax, "09_waterfall_cost", "Waterfall: cell cost bridge 2025→2026",
         subtitle="Li$_2$CO$_3$ price drives most of the drop" if consult else None)

def grid(gid):
    """Contact sheet of the 9 charts for one set (PNG, no metadata)."""
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    f, fT = ImageFont.truetype(path, 40), ImageFont.truetype(path, 64)
    cw, ch, head = 1250, 1000, 170
    G = Image.new("RGB", (3 * cw, head + 3 * (ch + 80)), "white"); d = ImageDraw.Draw(G)
    d.text((40, 50), SETS[gid]["name"], font=fT, fill=INK)
    for k, (fn, en) in enumerate(CHARTS):
        im = Image.open(OUT / gid / fn).convert("RGB"); r, c = divmod(k, 3); im.thumbnail((cw - 40, ch - 20))
        x0, y0 = c * cw, head + r * (ch + 80)
        d.text((x0 + 30, y0 + 15), en, font=f, fill=INK)
        G.paste(im, (x0 + (cw - im.width) // 2, y0 + 80))
    G.thumbnail((2400, 2400)); G.save(OUT / gid / f"_gallery_{gid}.png", optimize=True); return G


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="*", help="style sets: " + ", ".join(SETS) + " (default: all)")
    ap.add_argument("--out", type=pathlib.Path, default=OUT, help="output folder")
    a = ap.parse_args(); OUT = a.out; ids = a.sets or list(SETS)
    for gid in ids:
        CFG.clear(); CFG.update(SETS[gid]); CFG["id"] = gid; (OUT / gid).mkdir(parents=True, exist_ok=True)
        CHARTS = []; charts(); grid(gid)
    print("wrote", ", ".join(ids), "to", OUT.name)
