import pytest

from retrieval_lab import (
    DatasetInspectionReport,
    Document,
    EvaluationDataset,
    EvaluationQuery,
    inspect_dataset,
)
from retrieval_lab.domain import Chunk
from retrieval_lab.exceptions import DatasetValidationError


def _dataset(*queries: EvaluationQuery) -> EvaluationDataset:
    return EvaluationDataset(list(queries))


def test_inspect_dataset_detects_normalized_duplicate_queries() -> None:
    dataset = _dataset(
        EvaluationQuery(
            "q-1",
            "ＡＷＳ  Secrets",
            relevant_document_ids={"doc-1"},
        ),
        EvaluationQuery(
            "q-2",
            "aws secrets",
            relevant_document_ids={"doc-2"},
        ),
    )

    report = inspect_dataset(dataset)

    assert isinstance(report, DatasetInspectionReport)
    assert report.has_issues
    duplicate = report.issues[0]
    assert duplicate.code == "duplicate_query_text"
    assert duplicate.query_ids == ("q-1", "q-2")


def test_inspect_dataset_detects_relevance_concentration() -> None:
    dataset = _dataset(
        *[
            EvaluationQuery(
                f"q-{index}",
                f"query {index}",
                relevant_document_ids={"shared" if index < 4 else "other"},
            )
            for index in range(5)
        ]
    )

    report = inspect_dataset(dataset, relevance_concentration_threshold=0.5)

    concentration = next(
        issue for issue in report.issues if issue.code == "relevance_concentration"
    )
    assert concentration.relevance_ids == ("shared",)
    assert report.max_relevance_share == pytest.approx(0.8)


def test_inspect_dataset_detects_verbatim_query_in_positive_document() -> None:
    query = EvaluationQuery(
        "q-1",
        "AWS Secrets Manager の設定方法",
        relevant_document_ids={"doc-1"},
    )
    dataset = _dataset(query)
    documents = [
        Document(
            "doc-1",
            "ここでは AWS Secrets Manager の設定方法 を説明します。",
        )
    ]

    report = inspect_dataset(dataset, documents=documents)

    issue = next(
        item
        for item in report.issues
        if item.code == "verbatim_query_in_relevant_text"
    )
    assert issue.query_ids == ("q-1",)
    assert issue.relevance_ids == ("doc-1",)


def test_inspect_dataset_skips_verbatim_check_without_corpus() -> None:
    dataset = _dataset(
        EvaluationQuery(
            "q-1",
            "this is a sufficiently long query",
            relevant_document_ids={"doc-1"},
        )
    )

    report = inspect_dataset(dataset)

    assert all(
        issue.code != "verbatim_query_in_relevant_text" for issue in report.issues
    )


def test_inspect_dataset_supports_chunk_level_corpus() -> None:
    query = EvaluationQuery(
        "q-1",
        "sufficiently long chunk query",
        relevant_chunk_ids={"chunk-1"},
    )
    dataset = EvaluationDataset([query], relevance_level="chunk")
    chunks = [
        Chunk(
            "chunk-1",
            "doc-1",
            "sufficiently long chunk query",
            0,
            29,
        )
    ]

    report = inspect_dataset(dataset, chunks=chunks)

    assert any(
        issue.code == "verbatim_query_in_relevant_text" for issue in report.issues
    )


def test_inspection_report_to_dict_is_json_compatible() -> None:
    dataset = _dataset(
        EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})
    )

    payload = inspect_dataset(dataset).to_dict()

    assert payload["query_count"] == 1
    assert payload["unique_relevance_id_count"] == 1
    assert isinstance(payload["issues"], list)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("inf"), True, "0.5"])
def test_inspect_dataset_rejects_invalid_concentration_threshold(
    value: object,
) -> None:
    dataset = _dataset(
        EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})
    )

    with pytest.raises(DatasetValidationError, match="threshold"):
        inspect_dataset(
            dataset,
            relevance_concentration_threshold=value,  # type: ignore[arg-type]
        )


def test_inspect_dataset_validates_supplied_corpus_ids() -> None:
    dataset = _dataset(
        EvaluationQuery(
            "q-1",
            "this query is long enough",
            relevant_document_ids={"missing"},
        )
    )

    with pytest.raises(DatasetValidationError, match="missing"):
        inspect_dataset(dataset, documents=[Document("doc-1", "text")])
