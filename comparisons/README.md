# ggplot2 comparison

`violin_compare.py` generates 120 synthetic points (three normal groups, seed 0), writes
them to `violin_data.csv`, renders them with real ggplot2 via `violin_ggplot2.R`
(`geom_violin(trim = FALSE)` + `geom_boxplot(width = 0.1)` + `geom_jitter(width = 0.08)`,
`theme_bw`, `scale_fill_npg`), renders the same data with `plot_stats.violin(look="gg")`,
and places both side by side in `docs/images/violin_compare.png`.

Committed image made with R 4.5.0, ggplot2 3.5.1, ggsci 3.2.0, matplotlib 3.11.2.

```bash
Rscript -e 'install.packages(c("ggplot2", "ggsci"))'   # once
python comparisons/violin_compare.py
```
