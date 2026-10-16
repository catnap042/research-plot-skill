# Extra recipes and pitfalls (companion to SKILL.md)

## Helper API (plot_layout.py, plot_stats.py; pure matplotlib, no seaborn/ultraplot)
Import from `scripts/` after `style.use(...)`. Tests live in the repository (`python -m pytest tests`).

Layout (`plot_layout`):
- `subplots(nrows, ncols, refwidth_mm=None, figsize_mm=None, sharex=False|True|'row'|'col', sharey=..., mosaic=None, panel_labels=True, label_style='(a)'|'a'|'A'|'a.'|'(A)')` -> (fig, axes). Constrained layout with font-size-aware pads. Panel labels are bold, sit outside the top-left, and share the title baseline when the axes has a title. Long titles wrap so neighbouring panels keep a gap of at least 1 em.
- `add_panel_labels(axes, style, loc='outside'|'inside')`: 'inside' adds a white outline and box.
- `label_outer_shared(axes)`, `set_shared_labels(fig, xlabel, ylabel)`: one supxlabel/supylabel; inner panels hide only the tick labels, not the ticks.
- `edge_colorbar(fig, mappable, axes, loc='right'|'bottom', label=...)`: spans the axes it serves, at most 5 ticks. `edge_legend(fig, axes, loc='top'|'right'|'bottom', ncols=...)`: space is reserved above the panel labels.

Stats (`plot_stats`); every function returns a dict of the computed numbers:
- `box_strip(ax, data, x, y, jitter=0.15)`: box fill α 0.72, white median with a dark edge, hollow jittered points, n in the tick labels, caption "box: median/IQR; whiskers: 1.5·IQR; n=…".
- `violin(ax, data, x, y, look='gg'|'classic', inner='box'|'quartile'|'points', points=False, cut=None, width=None, scale=None, point_shape='circle'|'group', point_edge='group'|'dark'|0-1, point_alpha=0.8, point_lw=None, jitter=None)`. In the gg look, the defaults follow ggplot2 `geom_violin(trim=FALSE)` + `geom_boxplot(width=0.1)` + `geom_jitter(width=0.08)`. Point size s=15.3 pt² (diameter 3.9 pt, the ggplot2 default size 1.5) with a 0.71 pt stroke. Points are drawn above the inner box. `cut=0` clips to the data range.
- `regplot(ax, data, x, y, ci=95, order=1)`: hollow points, OLS line, CI band, caption "line: OLS fit; band: 95% CI; n=…".
- `lineplot(ax, data, x, y, hue=None, err='sd'|'se'|'ci')`: mean ± band, caption states the error type and n per group. `scatter(ax, data, x, y, hue=None)`. `facet(data, col=, row=, func=lambda ax, d, **kw: ..., xlabel=, ylabel=)`.
- Captions sit under the x-axis label, anchored to its real size, so they never cover data or multi-line tick labels.
- Points are always hollow with a transparent centre (`facecolors='none'`).

Known limits: the talk preset enlarges fonts and frame, but helper point sizes and violin outlines stay at paper size. Pass `point_lw` or edit the size if the talk figure looks thin.

All recipes assume `import style; style.use("paper", variant="C", palette="npg")` (or `style.use_talk()`) was called first. Column names below are placeholders; use the real DataFrame columns.

## Predicted vs measured
- Square axes: `ax.set_aspect("equal")`, same limits on x and y.
- `y = x` reference line (`**style.reference()`) plus a ±5 % band: `ax.fill_between(lim, lim*0.95, lim*1.05, color="grey", alpha=0.10, label="±5%")`.
- Background: training set light grey `#BDBDBD`, internal test set dark grey `#636363`; foreground groups in palette colors with `**style.measured(i)`.
- Optional thin vertical error segments from each point to the y = x line; write MAE/RMSE and n in the legend or a corner note.

## OOD diagnostic (sigma_pred vs Mahalanobis distance D_M)
- Scatter per group (color + marker). Threshold lines use the reference style, each with a text label (e.g. "D_M green = 2.1", "2×baseline σ"); never rely on green/red alone.
- Put the legend outside on the right if it covers points.

## Training-set boxplot with IQR bounds + group points
- Small multiples, `ncols=3`, one panel per feature; position 0 = training-set box in light grey, positions 1..k = jittered points per group.
- Dashed lines at Q1 − 1.5·IQR and Q3 + 1.5·IQR of the training set mark the outlier bounds.
- One shared figure-level legend for the groups; hide unused panels.
- Residual boxplot variant: box per group (fill alpha ≈ 0.55) + jitter points + `y = 0` reference.

## ultraplot pitfalls (only if the user asks for ultraplot)
- `fig.legend(...)` is a panel legend: `loc` must be `'l' | 'r' | 'b' | 't'`; `'center right'` or `bbox_to_anchor` raise `KeyError: Invalid panel location`.
- `ax.boxplot` uses native args `fill=True, fillcolor=..., fillalpha=..., edgecolor=..., mediancolor=..., meancolor=...`; matplotlib `boxprops/medianprops/patch_artist` cause `PathPatch object has no property ...`.
- Axes-level legend outside: `ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))` with constrained layout or `fig.subplots_adjust(right=0.78)`.

- `uplt.rc.update()` rebuilds the font list and drops `.ttc` fonts (Noto CJK, PingFang, Microsoft YaHei), so Chinese renders as boxes; re-register CJK fonts with `addfont` afterwards (see header below).
- `ax.boxplot` reads a list of arrays row-wise and draws dozens of tiny boxes; pass a 2D array with one column per group (`np.column_stack(groups)`).
- Grid draws over boxes by default; set `axes.axisbelow=True`.
- Subplots share axes by default; use `share=False` for panels with different units.

### Tested ultraplot header
Tested with ultraplot 2.7.1 + matplotlib 3.11.2. Main path stays pure matplotlib (`style.py`); use this only when ultraplot is requested.
```python
import ultraplot as uplt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
       '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
MARKERS = ['o', 's', '^', 'D', 'v']
FONTS = ['Arial', 'Helvetica', 'Arimo', 'Liberation Sans', 'SimHei',
         'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans']

def use_bw_npg_uplt(mode='paper'):
    talk = mode == 'talk'
    frame = 1.2 if talk else 0.7
    have = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in FONTS if f in have] or ['DejaVu Sans']
    uplt.rc.update({
        'font.size': 16 if talk else 8,
        'title.loc': 'left',
        'axes.linewidth': frame, 'axes.edgecolor': '#333333',
        'grid': True, 'grid.color': '#E1E1E1', 'grid.linewidth': 0.6 if talk else 0.5,
        'grid.alpha': 1, 'gridminor': False,
        'tick.dir': 'out', 'tick.width': frame, 'tick.len': 5 if talk else 3,
        'tick.minor': False,
        'cycle': NPG, 'lines.linewidth': 2.5 if talk else 1.2,
        'legend.frameon': False,
        'savefig.dpi': 300 if talk else 600,
    })
    # ultraplot's rc.update() rebuilds the font list and drops .ttc fonts (Noto CJK, PingFang, msyh),
    # so re-register CJK fonts afterwards, then use an explicit family list.
    for p in font_manager.findSystemFonts():
        if any(k in p.lower() for k in ('cjk', 'simhei', 'msyh', 'pingfang')):
            font_manager.fontManager.addfont(p)
    have = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in FONTS if f in have] or ['DejaVu Sans']
    mpl.rcParams['font.family'] = fonts   # explicit list -> per-glyph CJK fallback
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.axisbelow': True})
```

## Chinese font troubleshooting
- Font list must be explicit and include a CJK font: Windows `SimHei` / `Microsoft YaHei`, macOS `PingFang SC`, Linux `Noto Sans CJK SC` (`apt install fonts-noto-cjk`). Set `axes.unicode_minus=False`.
- Verify: `matplotlib.font_manager.findfont("SimHei")` should not return a DejaVu path.
- Still missing after install: delete `~/.cache/matplotlib/fontlist-*.json` (or `~/.matplotlib/`) and restart the kernel, or register directly with `matplotlib.font_manager.fontManager.addfont("/path/to/font.ttf")`.

## Honest-encoding rules (adapted from K-Dense scientific-visualization, MIT)
- Error bars/bands must state what they are (SD, SE, 95 % CI, percentile) and n plus the replicate unit, in the caption or axis label.
- No dual Y axes; put incompatible units in separate panels.
- Disclose smoothing, fitting, binning and normalization: method, parameters, window, baseline cycle/formula.
- Bars and areas start at zero; lines/points may use a non-zero range if ticks show it. Do not silently connect across missing data.
- EIS: Nyquist plots use equal aspect (`ax.set_aspect("equal")`), label units of Z′ and −Z″, mark frequency direction; fits as lines distinct from measured points, with residuals.
- dQ/dV / ICA: record differentiation step, filter/smoothing algorithm, window and edge handling.
- Cycling curves: state C-rate, temperature, cut-off voltages and capacity basis.

## Optional audited delivery (for papers or reviewed reports)
```
figure_name/
  data_processed.csv
  plot_figure.py
  figure.png / figure.pdf / figure.svg
  manifest.json   # source data, transforms, missing-data policy, stats definitions, software versions, size, export settings
  audit.txt       # checks done at final size: fonts, clipping, overlaps, legend, DPI
```
Never describe a style preset or an automatic check as journal-compliance certification.
