# Contributing

Thank you for helping improve Retrieval Lab.

1. Open or select one focused issue.
2. Create a feature branch; never work directly on `main`.
3. Keep public behavior typed and covered by tests.
4. Run `pytest`, `ruff check .`, `ruff format --check .`, `mypy src`, and
   `python -m build`.
5. Open a pull request describing purpose, changes, verification, and known limits.

Metric changes must include a hand-calculated fixture and document any definition or
schema-version change. Do not include private corpora, API keys, generated model
weights, or datasets without clear redistribution rights.

