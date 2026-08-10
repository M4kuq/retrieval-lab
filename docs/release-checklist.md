# Release candidate checklist

This checklist is a review record for the release candidate. CI validates the
package, documentation, and artifacts locally; publication actions remain
deliberately manual.

## Required verification

- [ ] `uv lock --check`
- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest`
- [ ] `mkdocs build --strict`
- [ ] Build and inspect both sdist and wheel.
- [ ] Install core, Dense, docs, and all extras in clean environments.
- [ ] Confirm no model, LLM, secret, or external API is required for the
      verification commands.
- [ ] Review the release notes and package metadata for the intended version.
- [ ] Recheck that `retrieval-lab-sdk` is the available distribution name on
      PyPI immediately before any public upload; the occupied `retrieval-lab`
      name must not be used.

## Publication approval boundary

Each action below may run only after the user gives explicit approval for
that specific publication action. CI and this repository do not perform them:

- [ ] Upload to TestPyPI.
- [ ] Upload to PyPI.
- [ ] Create or push a Git tag.
- [ ] Create a GitHub Release.
- [ ] Deploy GitHub Pages.

No credential, token, secret, or publishing endpoint belongs in the workflow or
the release artifacts. A failed check blocks publication until the artifact is
repaired and the full checklist is rerun.
