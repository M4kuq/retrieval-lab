# Dataset authoring

Retrieval Lab v0.2 adds dataset authoring assistance: tools for
building the evaluation datasets themselves, under the same
local-first, Python-API-first contract as evaluation. Authoring works
on drafts, not on `EvaluationDataset`. A draft carries explicit
provenance — where the data came from and whether a human has
reviewed it — and it cannot be evaluated until it is finalized, so
synthetic or unreviewed data is never silently promoted to trusted
ground truth.

## Drafts

A `DatasetDraft` is immutable authoring state: a tuple of
`DraftQuery` values (each with graded positive relevance that may
still be incomplete), a relevance level (`document` or `chunk`), and a
`DatasetProvenance`. Build one directly, generate one (see below), or
derive one from an existing dataset with
`DatasetDraft.from_dataset(dataset, provenance=...)`.

- `add_query()` / `replace_query()` return updated drafts; relevance
  is edited per query with `DraftQuery.with_relevance()` and
  `without_relevance()`.
- `pending_query_ids` lists queries that still need judgments, and
  `complete` reports whether every query has at least one.
- `finalize()` builds the validated `EvaluationDataset`. It raises
  `DatasetValidationError` while the draft is empty or any judgment
  is pending.

Drafts serialize deterministically. `to_jsonl()` emits the canonical
evaluation JSONL form, and `save_bundle(directory)` writes that JSONL
plus a `dataset-manifest.json` sidecar (schema version 1) recording
the relevance level, the provenance, the query count, and the pending
query IDs. `load_dataset_draft(bundle)` reads a bundle back and
rejects it if `query_count` or `pending_query_ids` disagree with the
JSONL, or if the manifest's `human_reviewed` / `reliability` values
contradict its own origin and review status.

## Provenance and review state

`DatasetProvenance` records two independent facts:

- `origin`: `unknown`, `human`, `synthetic`, or `mixed`.
- `review_status`: `unreviewed`, `in_review`, or `reviewed`.

Its derived `reliability` label is what reports and statuses show
(first matching row, top to bottom):

| Origin | Review status | Reliability |
| --- | --- | --- |
| `synthetic` or `mixed` | not `reviewed` | `experimental` |
| `unknown`, or any origin not yet reviewed | — | `unverified` |
| `human`, `synthetic`, or `mixed` | `reviewed` | `reviewed` |

Review transitions are explicit: `mark_draft_in_review()` and
`mark_draft_reviewed()` return a draft with only the review status
(and notes) changed. The review transitions — and the persisted
review service below — never rewrite the origin: a reviewed synthetic
dataset stays `origin=synthetic`, and only its reliability label
changes. The lower-level `with_provenance()` is different: it
replaces the whole provenance record with whatever the caller
supplies, so relabeling a draft through it is the caller's explicit
responsibility, not something the review flow does.

The provenance lives in the draft and bundle layer. `finalize()`
produces a plain `EvaluationDataset`, which has no provenance field,
so keep the bundle manifest alongside any finalized dataset whose
origin matters.

## Persisted review workflow

The application services in `retrieval_lab.dataset_workflow` operate
on a saved bundle:

- `dataset_draft_status(bundle)` returns a `DatasetDraftStatus`
  (counts, pending IDs, origin, review status, reliability) with a
  strict JSON form for automation.
- `review_dataset_query(bundle, query_id=..., relevance=...,
  corpus=...)` replaces one query's judgments, validates the IDs
  against the corpus documents, and saves the bundle. It currently
  supports document-level drafts. With `complete_review=True` it
  marks the draft reviewed, but only when no judgments are pending
  anywhere in the draft.
- `finalize_dataset_bundle(bundle)` validates that the persisted
  draft can be finalized and returns its status.

The CLI is a thin adapter over these services:

```console
retrieval-lab dataset status BUNDLE [--json]
retrieval-lab dataset review BUNDLE --corpus CORPUS --query-id ID \
    [--relevant ID[:GRADE]]... [--complete-review] [--notes TEXT] [--json]
retrieval-lab dataset finalize BUNDLE [--json]
```

Repeat `--relevant` once per positive document. Without
`--relevant`, `dataset review` prompts interactively: it prints the
query and the corpus documents, then reads comma-separated document
IDs (with optional grades) from standard input.

## Inspection

`inspect_dataset(dataset, documents=...)` (or `chunks=...` for
chunk-level data) returns a `DatasetInspectionReport` of
deterministic findings. Inspection never modifies the dataset; every
finding is a candidate for human review.

| Code | Meaning |
| --- | --- |
| `duplicate_query_text` | Two or more queries normalize (NFKC, case-folding, whitespace) to identical text. |
| `relevance_concentration` | One relevance ID is judged positive for more than `relevance_concentration_threshold` (default 0.5) of the queries; only checked once the dataset has at least `min_queries_for_concentration` (default 5) queries. |
| `verbatim_query_in_relevant_text` | The normalized query (at least `min_verbatim_query_characters`, default 12) appears verbatim in a positively judged document or chunk — a low-difficulty or generation-leakage candidate. |

The duplicate and concentration checks need only the dataset; the
verbatim check also needs the corpus texts.

## Synthetic generation

Synthetic generation sits behind an explicit provider boundary. A
generator implements the `SyntheticQueryGenerator` protocol — a
`name` and `generate(documents, *, count, seed)` returning
`DraftQuery` values — and `CallableSyntheticGenerator` adapts a plain
callable to it. No provider client is bundled and nothing performs
network access.

`generate_synthetic_draft(documents, generator=..., count=...,
seed=...)` validates the output (exactly `count` queries, unique
IDs, every query carrying positive judgments that reference the given
documents) and stamps the draft's provenance itself:
`origin=synthetic`, `review_status=unreviewed`, the generator's name,
and the seed in the notes. The resulting draft is experimental by
construction. The review transitions and the review service can move
it to `reviewed`, and they never change its origin; as noted above,
only the caller-responsibility `with_provenance()` replaces the
provenance record wholesale.

## End to end

`examples/dataset_authoring.py` runs the whole flow — deterministic
synthetic generation, hand-written additions, bundle persistence, the
persisted review workflow, finalization, inspection, and evaluation —
and is executed as an integration test so it cannot drift from the
public API:

```console
uv run python examples/dataset_authoring.py
```
