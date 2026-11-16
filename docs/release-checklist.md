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

## Cutting a new version (tag-time procedure)

The version decision and the bump belong to the publication act itself
(RET-12); preparatory changes must not bump the version. When a new
version is cut, every location below pins the version literally and
must move together, in one change:

- `pyproject.toml`: `[project] version` — the source of truth; the
  release workflow requires the tag to match it exactly.
- Fallback literals used when the distribution metadata is
  unavailable: `src/retrieval_lab/cli/app.py`,
  `src/retrieval_lab/runner.py`,
  `src/retrieval_lab/retrievers/callable.py`, and
  `benchmarks/harness.py`.
- `docs/index.md`: the pinned install command
  (`pip install retrieval-lab-sdk==<version>`).
- Release notes: add `docs/release-notes-<version>.md` and update the
  file-name pins in `mkdocs.yml` (nav), `MANIFEST.in` (include line),
  and `tests/unit/test_packaging.py` (manifest, nav, and sdist lists).
- `.github/workflows/release-candidate.yml`: the artifact name
  (`retrieval-lab-sdk-<version>-dist`, two places).
- `CHANGELOG.md`: rename the `[Unreleased]` section to the new
  version with its release date and start a fresh `[Unreleased]`.
- Tests that assert the version literal:
  `tests/unit/test_packaging.py`, `tests/unit/test_callable.py`,
  `tests/integration/test_runner.py`, and
  `tests/integration/test_cli.py`.

After the bump, rerun the full Required verification list above (the
packaging tests assert that the pyproject, runtime, and wheel
versions agree), dispatch the Release candidate validation workflow
on the bumped tree and confirm its install matrix passes, and grep
the tree for the old version string: outside the `CHANGELOG.md`
history sections and the previous release-notes file, no occurrence
may remain. Only then push the tag, as the approval boundary below
describes.

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

- [ ] Create and push the version tag (`v<version>`); the tag name
      must match the `pyproject.toml` version exactly, which the workflow
      verifies before building. If the tag carries a new version, cut it
      first by following the tag-time procedure above.
- [ ] Upload to TestPyPI (manual dry run, outside the release workflow).
- [ ] Create a GitHub Release.
- [ ] Deploy GitHub Pages.

No credential, token, secret, or publishing endpoint belongs in the workflow or
the release artifacts. A failed check blocks publication until the artifact is
repaired and the full checklist is rerun.
