# Third-party notices

No third-party code, fonts or images are bundled in this repository.
Neutral greys (#333333, #666666, #DDDDDD) and the Okabe-Ito orange/green used for status colours are generic values.

| What | Where | Source and license | How it is used |
|---|---|---|---|
| ggsci colour hex values (npg, jco, lancet) | `scripts/style.py` | ggsci R package, `R/palettes.R`, https://github.com/nanxstats/ggsci (GPL-3.0 or later) | Only the colour hex values are copied; no ggsci code is copied. The palettes are "inspired by" journal figures and are not official journal palettes. |
| ggplot2 defaults (hue wheel formula, `theme_bw` look, violin/jitter defaults) | `scripts/style.py`, `scripts/plot_stats.py` | ggplot2 / scales R packages (MIT) | Re-implemented from the documented behaviour; no R code is copied. |
| Honest-encoding rules | `recipes.md` | K-Dense scientific-visualization skill, https://github.com/K-Dense-AI/scientific-agent-skills (MIT, Copyright (c) 2025 K-Dense Inc.) | Rules paraphrased and adapted, credited in the section title. No text or code is copied verbatim. |
| Fonts | not bundled | Arial / Helvetica / Liberation Sans / Noto Sans CJK / DejaVu Sans, whatever is installed | Looked up by name at run time. |

`comparisons/violin_ggplot2.R` calls ggplot2 and ggsci at run time; neither is redistributed.
