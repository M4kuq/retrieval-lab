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

Publication runs through the tag-driven `release.yml` workflow, which
publishes to PyPI with Trusted Publishing (OIDC); no PyPI token or secret is
stored anywhere. The workflow can only run after both prerequisites below are
complete, and pushing the version tag remains the explicit approval act:

- [ ] One-time PyPI-side setup (project owner only): register a Trusted
      Publisher for the `retrieval-lab-sdk` project on PyPI with owner
      `M4kuq`, repository `retrieval-lab`, workflow `release.yml`, and
      environment `pypi`. For a first release, add it as a pending publisher
      on PyPI before the tag is pushed.
- [ ] Confirm the `pypi` environment exists in the GitHub repository settings
      (it carries no secrets; it exists so the OIDC claim matches).

Each action below may run only after the user gives explicit approval for
that specific publication action:

- [ ] Create and push the version tag (for example `v0.1.0rc1`); the tag name
      must match the `pyproject.toml` version exactly, which the workflow
      verifies before building.
- [ ] Upload to TestPyPI (manual dry run, outside the release workflow).
- [ ] Create a GitHub Release.
- [ ] Deploy GitHub Pages.

No credential, token, secret, or publishing endpoint belongs in the workflow or
the release artifacts. A failed check blocks publication until the artifact is
repaired and the full checklist is rerun.
