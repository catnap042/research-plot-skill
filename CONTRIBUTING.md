# Contributing

Issues and pull requests are welcome.

1. `python -m pip install -r requirements-dev.txt`
2. `ruff check .` and `python -m pytest -q tests` must pass.
3. Regenerate images you touched (`python examples/make_gallery.py`, `python examples/quickstart.py`)
   and look at them at full size before committing.
4. Use synthetic or clearly licensed public data only. Never commit real or confidential measurements.
5. Keep code, comments and docs in English. CJK text in tests must use `\u` escapes.
