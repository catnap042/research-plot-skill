---
name: research-data-plotting
description: >-
  Use this when drawing research data figures (capacity fade, dV/dQ, XRD,
  particle size, box/violin with significance, heatmap, Sankey, waterfall, etc.)
  for papers, reports or slides in Python/matplotlib, to get one unified
  theme_bw + ggsci-npg style with paper and talk presets.
---
# Research data plotting (theme_bw + npg)

Default look: ggplot2 theme_bw layout (full panel border, light grey grid) with the ggsci `npg` palette. Helpers live in `scripts/` next to this file: `style.py` (presets, palettes, export), `plot_layout.py` (multi-panel layout) and `plot_stats.py` (box/violin/regression/line/scatter/facet with honest captions). They are pure matplotlib; add `scripts/` to `sys.path` (`sys.path.insert(0, "<this folder>/scripts")`) and import them. Requirements: matplotlib, numpy, pandas, scipy. `recipes.md` documents the helper API and holds extra recipes (predicted vs measured, OOD diagnostic, IQR-bounded boxplots), ultraplot pitfalls, font troubleshooting and honest-encoding rules.

## Steps
1. Pick the target:
   - **paper**: base font 8 pt, 600 dpi, 89 mm (single column) or 183 mm (double column) wide.
   - **talk** (slides, the talk preset): tick labels 18 pt, axis titles 20 pt, legend 16 pt, direct labels about 14 pt.
2. Call the preset before drawing: `import style; style.use_talk()` for slides, or `style.use("paper", variant="C", palette="npg")` for papers. Variants B/C set the full frame in rcParams, so every axes (including helper-made ones) gets the theme_bw box.
   - Multi-panel figures: `plot_layout.subplots(...)` (bold (a)/(b) labels outside the top-left, shared axes hide inner tick labels), `edge_colorbar`, `edge_legend`, `set_shared_labels`.
   - Stats plots: `plot_stats.box_strip`, `violin`, `regplot`, `lineplot`, `scatter`, `facet`. Each adds a caption under the x label stating the error type and n, and returns the computed stats. They never change the global theme. Use plain matplotlib only.
3. Pick the chart type by what the data says:
   - trends over cycles or storage time: lines; split into small multiples when there are many conditions
   - comparing groups or recipes: raw points plus median/IQR, with n written in the axis label
   - model fits (OCV, ECM): measured points, model line, residuals, independent validation
   - composition: donut or waffle, 6 categories or fewer
   - flows: Sankey; check that every node's inflow equals its outflow
   - matrices: heatmap with a sequential colormap
   - cost or loss bridges: waterfall
   - model evaluation: predicted vs measured (equal axes, y=x, ±5% band), OOD diagnostic (sigma vs distance with labelled thresholds), training-set boxplot with Q1−1.5·IQR / Q3+1.5·IQR bounds plus group points; see `recipes.md`
4. Encode meaning consistently:
   - one sample keeps one color across every figure, and also gets its own marker shape
   - measured data as hollow markers with a transparent centre (`mfc="none"`), models as solid lines, predictions as dashed lines, reference lines as grey dotted lines
   - continuous variables (cycle number, temperature, SOC) use a sequential colormap such as cividis, never the categorical palette
   - box and violin fills use one opacity, 0.72, for both
   - violins default to the ggplot2 look (`look="gg"`): bw.nrd0 bandwidth, tails 3·bw, equal area, width 0.9, thin grey outline, slim white inner boxplot; `points=True` adds hollow circles in the group colour (ggplot2 geom_jitter default size, ±0.08 jitter, α 0.8). `look="classic"` keeps the older narrow style
   - the risk / to-be-verified color is orange, always paired with a text label; never rely on red versus green
5. Lines and frame:
   - **talk**: panel border, axis lines and ticks 1.2 pt (ticks 5 pt long), data lines 2.5 pt, markers about 8, grid 0.6 pt light grey. For a big room or dim projector, go to frame 1.4 pt and data lines 3 pt together. Avoid 0.8 pt (washes out) and 1.6 pt (the black box dominates the slide).
   - **paper**: thinner, about 0.7 pt axes and 1.2 pt data lines.
   - Tick width always matches the frame weight.
6. Size for slides: a single figure is 165x120 mm (half a 16:9 slide); a figure with two or more panels uses the full width (about 330x140 mm).
7. Label synthetic or example data in the lower-right corner ("Example: synthetic data"; `style.synthetic_tag(fig)` does this).
8. Be honest in the encoding:
   - error bars state SD, SE or CI, plus n and the replicate unit
   - no dual Y axes; incompatible units go in separate panels
   - disclose smoothing, fitting, binning and normalization parameters
   - EIS Nyquist plots use equal-aspect axes; dQ/dV states the differentiation step and filter
   - details in `recipes.md`
9. Export PNG plus PDF and SVG with editable text; paper exports at 600 dpi (`style.save` does this), talk at 300 dpi. Set font.family to an explicit list that includes a CJK font (Arial, Arimo, SimHei, Microsoft YaHei, PingFang SC, Noto Sans CJK SC), and set `axes.unicode_minus=False`, or Chinese and minus signs render as boxes. CJK labels are supported this way. If a font is still not found, see the troubleshooting in `recipes.md` (findfont check, clear fontlist cache, addfont).
10. If the user wants ultraplot: figure-level `fig.legend` only takes loc 'l'/'r'/'b'/'t', and boxplot takes fill/fillcolor/mediancolor, not boxprops. Legend covering data: move it outside right (`loc="center left", bbox_to_anchor=(1.02, 0.5)`) or label lines directly.
11. For papers or reviewed reports, optionally deliver the audited bundle (processed data, script, figures, manifest, audit note) described in `recipes.md`.
12. Before delivering, look at every image at full size. Check for labels touching the ring or the title (donut and radar charts are the usual offenders), legends covering data, and consistent style across all figures in the set.

## Alternatives
- `palette="hue"`: ggplot2 default colors (#F8766D, #00BA38, #619CFF for n=3).
- `palette="jco"`: softer look.
- `palette="lancet"`: ggsci lancet colours.
- Minimal flat style (no frame or grid, direct value labels) for clean slides.
- Consulting style: one accent color (#3C5488) with everything else grey, to highlight a single finding.

The ggsci hex values come from the ggsci R package (`R/palettes.R`). npg imitates the Nature look; it is not an official journal requirement.
