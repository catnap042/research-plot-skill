"""Side-by-side check: real ggplot2 render vs plot_stats.violin(look="gg") on identical synthetic data.

    python comparisons/violin_compare.py      # needs R with ggplot2 + ggsci (Rscript on PATH)
Writes comparisons/violin_data.csv, comparisons/violin_ggplot2.png, comparisons/violin_ours_gg.png
and docs/images/violin_compare.png.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "skills" / "research-data-plotting" / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_stats
import style

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "group": np.repeat(["A", "B", "C"], 40),
    "y": np.r_[rng.normal(0, 1, 40), rng.normal(1.2, 1.1, 40), rng.normal(-0.5, 0.8, 40)],
})
df.to_csv(HERE / "violin_data.csv", index=False)
subprocess.run(["Rscript", "violin_ggplot2.R"], cwd=HERE, check=True)

style.use("paper", variant="C", palette="npg")
fig, ax = plt.subplots(figsize=(89 / 25.4, 80 / 25.4), layout="constrained")
plot_stats.violin(ax, df, "group", "y", look="gg", points=True)
style.finish(ax)
fig.savefig(HERE / "violin_ours_gg.png", dpi=300, metadata={"Software": None})
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, (f, t) in zip(axs, [("violin_ggplot2.png", "ggplot2 (R): theme_bw + scale_fill_npg"),
                            ("violin_ours_gg.png", "this skill: plot_stats.violin(look='gg')")]):
    ax.imshow(plt.imread(HERE / f)); ax.set_title(t, fontsize=11); ax.axis("off")
fig.text(0.995, 0.01, "Example: synthetic data", ha="right", fontsize=8, color="#666666")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "images" / "violin_compare.png", dpi=150, metadata={"Software": None})
print("ok")
