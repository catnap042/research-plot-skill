# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-16

### Added
- `research-data-plotting` Agent Skill: `SKILL.md`, `recipes.md` and helper scripts
  (`style.py`, `plot_layout.py`, `plot_stats.py`, paper/talk `.mplstyle` files).
- theme_bw + ggsci npg default, paper and talk presets, variants A-D, npg/jco/lancet/ggplot2-hue palettes.
- Multi-panel layout helpers and statistics helpers that caption error type and n.
- Synthetic-data quickstart, 9-chart gallery in five style sets, ggplot2 violin comparison.
- Tests, CI on Linux/Windows/macOS, privacy and scope statements.

### Fixed
- Title-gap and caption helpers no longer fail when saving to PDF/SVG.
- Gallery markers are hollow (`mfc="none"`), matching the skill's encoding rules.
