"""30-second quickstart: a 2x2 research figure from synthetic data.

    python examples/quickstart.py            # writes out/quickstart.png/.pdf/.svg
All data are generated here with a fixed seed; nothing is read from disk.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "research-data-plotting" / "scripts"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import plot_layout
import plot_stats
import style


def synthetic_data(seed=0):
    rng = np.random.default_rng(seed)
    cycles = np.arange(0, 501, 25)
    fade = []
    for k, (name, rate) in enumerate({"Baseline": 0.040, "Coated": 0.025, "Doped": 0.032}.items()):
        for cell in range(3):  # three cells per recipe
            q = 100 - rate * cycles - 2e-5 * cycles**2 + rng.normal(0, 0.4, cycles.size)
            fade.append(pd.DataFrame({"recipe": name, "cell": cell, "cycle": cycles, "retention": q}))
    groups = pd.DataFrame({
        "recipe": np.repeat(["Baseline", "Coated", "Doped"], 30),
        "d50": np.r_[rng.normal(1.2, 0.15, 30),
                     np.r_[rng.normal(1.0, 0.1, 22), rng.normal(1.5, 0.08, 8)],
                     rng.lognormal(np.log(1.3), 0.2, 30)],
    })
    x = rng.uniform(150, 175, 40)
    reg = pd.DataFrame({"predicted": x, "measured": x + rng.normal(0, 2.5, x.size)})
    return pd.concat(fade), groups, reg


def main(out=ROOT / "out" / "quickstart"):
    style.use("paper", variant="C", palette="npg")
    fade, groups, reg = synthetic_data()
    fig, axes = plot_layout.subplots(2, 2, figsize_mm=(183, 140))
    a, b, c, d = np.asarray(axes).ravel()
    plot_stats.lineplot(a, fade, "cycle", "retention", hue="recipe", err="sd")
    style.finish(a, "Cycle number", "Capacity retention (%)"); a.set_title("Capacity fade, 3 cells per recipe"); a.legend(loc="lower left")
    plot_stats.box_strip(b, groups, "recipe", "d50")
    style.finish(b, "Recipe", "D50 (µm)"); b.set_title("Particle size")
    plot_stats.violin(c, groups, "recipe", "d50", points=True)
    style.finish(c, "Recipe", "D50 (µm)"); c.set_title("Same data, violin (ggplot2 look)")
    plot_stats.regplot(d, reg, "predicted", "measured")
    style.finish(d, "Predicted capacity (mAh/g)", "Measured capacity (mAh/g)"); d.set_title("Predicted vs measured")
    style.synthetic_tag(fig)
    style.save(fig, out)
    print(f"wrote {Path(out).name}.png/.pdf/.svg to {Path(out).parent.name}/")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "quickstart")
