# Scope

## In scope
- A matplotlib style (ggplot2 `theme_bw` layout + ggsci palettes) with paper and talk presets.
- Layout and statistics helpers that make figures honest by default (error type and n in captions).
- Instructions an AI coding agent can follow to choose and draw research charts.

## Out of scope
- Real, proprietary or confidential data. All examples use synthetic data generated in code.
- Journal-compliance certification. A preset is a starting point, not an approval.
- Interactive plotting (Plotly, Bokeh), seaborn or ultraplot as dependencies.
- Data analysis beyond what a figure needs (no fitting libraries, no statistics tests API).

## Privacy and network
| | |
|---|---|
| Network access | None. No downloads, no update checks. |
| Telemetry | None. |
| Files read | Only what your own code passes in. |
| Files written | Only the paths you pass to `style.save` or the example scripts (default `out/`). |
