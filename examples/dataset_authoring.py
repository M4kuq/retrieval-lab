"""Author an evaluation dataset end to end with the v0.2 workflow.

A deterministic stand-in generator produces a synthetic draft, which is
born experimental and unreviewed. The draft gains two hand-written
queries, is saved as a bundle, moves through the persisted review
workflow until a human marks it reviewed (its synthetic origin is never
rewritten), is finalized, inspected for quality issues, and only then
evaluated.
"""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from retrieval_lab import (
    CallableSyntheticGenerator,
    Document,
    DraftQuery,
    RetrievedQueryResult,
    dataset_draft_status,
    evaluate_results,
    finalize_dataset_bundle,
    generate_synthetic_draft,
    inspect_dataset,
    load_dataset_draft,
    mark_draft_in_review,
    review_dataset_query,
)

DOCUMENTS = (
    Document(
        "d-refund",
        "Refunds are issued to the original payment method within five business days.",
    ),
    Document(
        "d-shipping",
        "Standard shipping takes three to five business days and is free "
        "for orders over fifty dollars.",
    ),
    Document(
        "d-cache",
        "Clearing the browser cache removes temporary files but keeps "
        "saved passwords and sign-in data.",
    ),
    Document(
        "d-password",
        "You can reset a forgotten password from the sign-in page by "
        "using the account recovery link.",
    ),
)


def _phrase_queries(
    documents: Sequence[Document], count: int, seed: int
) -> list[DraftQuery]:
    queries = []
    for index in range(count):
        document = documents[(seed + index) % len(documents)]
        phrase = " ".join(document.text.split()[1:6])
        queries.append(DraftQuery(f"sq-{index + 1}", phrase, {document.id: 1}))
    return queries


generator = CallableSyntheticGenerator("phrase-fixture", _phrase_queries)
draft = generate_synthetic_draft(DOCUMENTS, generator=generator, count=3, seed=0)
assert draft.provenance.origin == "synthetic"
assert draft.provenance.review_status == "unreviewed"
assert draft.provenance.reliability == "experimental"

# Hand-written additions: sq-4 duplicates sq-1's text, and sq-5 makes
# d-refund the relevant document for three of the five queries.
draft = draft.add_query(
    DraftQuery("sq-4", "are issued to the original", {"d-refund": 1})
)
draft = draft.add_query(
    DraftQuery("sq-5", "payment method for a refund", {"d-refund": 1})
)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    corpus_path = root / "corpus.jsonl"
    corpus_path.write_text(
        "".join(
            json.dumps({"id": document.id, "text": document.text}) + "\n"
            for document in DOCUMENTS
        ),
        encoding="utf-8",
    )
    bundle = root / "bundle"
    draft.save_bundle(bundle)

    initial_status = dataset_draft_status(bundle)
    assert initial_status.reliability == "experimental"
    assert initial_status.pending_query_ids == ()

    mark_draft_in_review(load_dataset_draft(bundle)).save_bundle(bundle)

    reviewed_status = review_dataset_query(
        bundle,
        query_id="sq-5",
        relevance={"d-refund": 2},
        corpus=corpus_path,
        complete_review=True,
        notes="judgments checked against the corpus",
    )
    # Review changes the review state only; the origin stays synthetic.
    assert reviewed_status.review_status == "reviewed"
    assert reviewed_status.origin == "synthetic"
    assert reviewed_status.reliability == "reviewed"

    final_status = finalize_dataset_bundle(bundle)
    assert final_status.query_count == 5
    dataset = load_dataset_draft(bundle).finalize()

report = inspect_dataset(dataset, documents=DOCUMENTS)
issue_codes = {issue.code for issue in report.issues}
assert issue_codes == {
    "duplicate_query_text",
    "relevance_concentration",
    "verbatim_query_in_relevant_text",
}

rankings = []
for query in dataset.queries:
    relevant = next(iter(dataset.relevance_grades_by_query[query.id]))
    others = tuple(document.id for document in DOCUMENTS if document.id != relevant)
    rankings.append(RetrievedQueryResult(query.id, (relevant, *others)))
result = evaluate_results(
    dataset=dataset, retrieved_results=rankings, top_k=(1, 3), name="authored"
)
assert result.metrics["authored"].recall_at(1) == 1.0
print("inspection issues:", sorted(issue_codes))
print("recall@1:", result.metrics["authored"].recall_at(1))
