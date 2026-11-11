import json
from pathlib import Path

import pytest

from retrieval_lab import (
    BenchmarkDataset,
    DatasetValidationError,
    load_beir_dataset,
    load_msmarco_dataset,
    load_natural_questions_dataset,
)


def _write_beir(
    root: Path,
    *,
    corpus: list[dict[str, object]],
    queries: list[dict[str, object]],
    qrels: str,
    split: str = "test",
) -> Path:
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in corpus), encoding="utf-8"
    )
    (root / "queries.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in queries), encoding="utf-8"
    )
    (root / "qrels" / f"{split}.tsv").write_text(qrels, encoding="utf-8")
    return root


@pytest.fixture
def beir_dir(tmp_path: Path) -> Path:
    return _write_beir(
        tmp_path / "mini-beir",
        corpus=[
            {"_id": "d1", "title": "Alpha", "text": "first body"},
            {"_id": "d2", "text": "second body"},
            {"_id": "d3", "title": "", "text": "third body"},
        ],
        queries=[
            {"_id": "q1", "text": "alpha question"},
            {"_id": "q2", "text": "unjudged question"},
            {"_id": "q3", "text": "zero-only question"},
        ],
        qrels=(
            "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq1\td3\t0\nq3\td1\t0\n"
        ),
    )


def test_load_beir_dataset_converts_records(beir_dir: Path) -> None:
    benchmark = load_beir_dataset(beir_dir)

    assert isinstance(benchmark, BenchmarkDataset)
    documents = {document.id: document for document in benchmark.documents}
    assert documents["d1"].text == "Alpha\nfirst body"
    assert documents["d2"].text == "second body"
    assert documents["d3"].text == "third body"

    [query] = benchmark.dataset.queries
    assert query.id == "q1"
    assert query.query == "alpha question"
    assert query.relevant_document_ids == frozenset({"d1", "d2"})
    assert benchmark.dataset.relevance_grades_by_query["q1"] == {"d1": 2, "d2": 1}
    assert benchmark.excluded_query_ids == ("q2", "q3")


def test_load_beir_dataset_accepts_named_split(tmp_path: Path) -> None:
    root = _write_beir(
        tmp_path / "dev-split",
        corpus=[{"_id": "d1", "text": "body"}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="q1\td1\t1\n",
        split="dev",
    )
    benchmark = load_beir_dataset(root, split="dev")
    assert benchmark.dataset.queries[0].id == "q1"
    with pytest.raises(DatasetValidationError):
        load_beir_dataset(root)  # no qrels/test.tsv


def test_load_beir_dataset_rejects_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError):
        load_beir_dataset(tmp_path / "does-not-exist")
    with pytest.raises(DatasetValidationError):
        load_beir_dataset(tmp_path / "x", split="../escape")


def test_load_beir_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    root = _write_beir(
        tmp_path / "dup",
        corpus=[
            {"_id": "d1", "text": "one"},
            {"_id": "d1", "text": "two"},
        ],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="q1\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="duplicate corpus _id"):
        load_beir_dataset(root)


def test_load_beir_dataset_rejects_unknown_qrels_references(tmp_path: Path) -> None:
    root = _write_beir(
        tmp_path / "unknown-doc",
        corpus=[{"_id": "d1", "text": "body"}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="q1\tghost\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="unknown corpus ID"):
        load_beir_dataset(root)

    root = _write_beir(
        tmp_path / "unknown-query",
        corpus=[{"_id": "d1", "text": "body"}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="ghost\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="unknown query"):
        load_beir_dataset(root)


def test_load_beir_dataset_rejects_malformed_files(tmp_path: Path) -> None:
    root = _write_beir(
        tmp_path / "bad-score",
        corpus=[{"_id": "d1", "text": "body"}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="q1\td1\thigh\n",
    )
    with pytest.raises(DatasetValidationError, match="must be an integer"):
        load_beir_dataset(root)

    root = _write_beir(
        tmp_path / "empty-doc",
        corpus=[{"_id": "d1", "title": "", "text": "  "}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="q1\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="no title or text"):
        load_beir_dataset(root)

    broken = tmp_path / "broken"
    (broken / "qrels").mkdir(parents=True)
    (broken / "corpus.jsonl").write_text('{"_id": "d1", "text": ', encoding="utf-8")
    (broken / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "q"}\n', encoding="utf-8"
    )
    (broken / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="invalid JSON"):
        load_beir_dataset(broken)


def test_load_beir_dataset_duplicate_qrels_last_row_wins(tmp_path: Path) -> None:
    corpus = [
        {"_id": "d1", "text": "one"},
        {"_id": "d2", "text": "two"},
    ]
    queries = [
        {"_id": "q1", "text": "first"},
        {"_id": "q2", "text": "second"},
        {"_id": "q3", "text": "third"},
    ]
    qrels = (
        "q1\td1\t2\n"
        "q1\td1\t1\n"  # positive -> positive: grade overwritten
        "q1\td2\t2\n"
        "q1\td2\t0\n"  # positive -> zero: judgment removed
        "q2\td1\t0\n"
        "q2\td1\t2\n"  # zero -> positive: judgment added
        "q3\td1\t1\n"
        "q3\td1\t0\n"  # last positive removed: query excluded
    )
    root = _write_beir(tmp_path / "dups", corpus=corpus, queries=queries, qrels=qrels)
    benchmark = load_beir_dataset(root)
    assert benchmark.dataset.relevance_grades_by_query["q1"] == {"d1": 1}
    assert benchmark.dataset.relevance_grades_by_query["q2"] == {"d1": 2}
    assert benchmark.excluded_query_ids == ("q3",)


def test_load_beir_dataset_headerless_query_named_query_id(tmp_path: Path) -> None:
    root = _write_beir(
        tmp_path / "headerless",
        corpus=[{"_id": "d1", "text": "body"}],
        queries=[{"_id": "query-id", "text": "question"}],
        qrels="query-id\td1\t1\n",
    )
    benchmark = load_beir_dataset(root)
    assert benchmark.dataset.queries[0].id == "query-id"
    assert benchmark.excluded_query_ids == ()


def test_load_beir_dataset_preserves_unicode_line_separators(tmp_path: Path) -> None:
    root = tmp_path / "unicode"
    (root / "qrels").mkdir(parents=True)
    text = "first\u2028second\u0085third"
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "question"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
    benchmark = load_beir_dataset(root)
    assert benchmark.documents[0].text == text


def test_load_beir_dataset_rejects_invalid_utf8(tmp_path: Path) -> None:
    root = tmp_path / "bad-utf8"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_bytes(
        b'{"_id": "d0", "text": "fine"}\n{"_id": "d1", "text": "\xff"}\n'
    )
    (root / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "q"}\n', encoding="utf-8"
    )
    (root / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=r"corpus\.jsonl:2:.*UTF-8"):
        load_beir_dataset(root)

    (root / "corpus.jsonl").write_text(
        '{"_id": "d1", "text": "ok"}\n', encoding="utf-8"
    )
    (root / "qrels" / "test.tsv").write_bytes(
        b"query-id\tcorpus-id\tscore\nq1\td1\t\xff\n"
    )
    with pytest.raises(DatasetValidationError, match=r"test\.tsv:2:.*UTF-8"):
        load_beir_dataset(root)


def test_load_beir_dataset_rejects_all_unjudged(tmp_path: Path) -> None:
    root = _write_beir(
        tmp_path / "no-positive",
        corpus=[{"_id": "d1", "text": "body"}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels="q1\td1\t0\n",
    )
    with pytest.raises(DatasetValidationError, match="no queries"):
        load_beir_dataset(root)


def _write_msmarco(
    root: Path,
    *,
    collection: str,
    queries: str,
    qrels: str,
    split: str = "train",
) -> Path:
    root.mkdir(parents=True)
    (root / "collection.tsv").write_text(collection, encoding="utf-8")
    (root / f"queries.{split}.tsv").write_text(queries, encoding="utf-8")
    (root / f"qrels.{split}.tsv").write_text(qrels, encoding="utf-8")
    return root


@pytest.fixture
def msmarco_dir(tmp_path: Path) -> Path:
    return _write_msmarco(
        tmp_path / "mini-msmarco",
        collection="d1\tfirst passage\nd2\tsecond passage\nd3\tthird passage\n",
        queries="q1\tfirst question\nq2\tunjudged question\nq3\tzero-only question\n",
        qrels="q1\t0\td1\t2\nq1\t0\td2\t1\nq1\t0\td3\t0\nq3\t0\td1\t0\n",
    )


def test_load_msmarco_dataset_converts_records(msmarco_dir: Path) -> None:
    benchmark = load_msmarco_dataset(msmarco_dir)

    assert isinstance(benchmark, BenchmarkDataset)
    documents = {document.id: document for document in benchmark.documents}
    assert documents["d1"].text == "first passage"
    assert documents["d3"].text == "third passage"

    [query] = benchmark.dataset.queries
    assert query.id == "q1"
    assert query.query == "first question"
    assert query.relevant_document_ids == frozenset({"d1", "d2"})
    assert benchmark.dataset.relevance_grades_by_query["q1"] == {"d1": 2, "d2": 1}
    assert benchmark.excluded_query_ids == ("q2", "q3")


def test_load_msmarco_dataset_accepts_named_split(tmp_path: Path) -> None:
    root = _write_msmarco(
        tmp_path / "dev-split",
        collection="d1\tbody\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t1\n",
        split="dev",
    )
    benchmark = load_msmarco_dataset(root, split="dev")
    assert benchmark.dataset.queries[0].id == "q1"
    with pytest.raises(DatasetValidationError):
        load_msmarco_dataset(root)


def test_load_msmarco_dataset_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError):
        load_msmarco_dataset(tmp_path / "does-not-exist")
    with pytest.raises(DatasetValidationError):
        load_msmarco_dataset(tmp_path / "x", split="../escape")

    root = _write_msmarco(
        tmp_path / "dup",
        collection="d1\tone\nd1\ttwo\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="duplicate collection ID"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "unknown",
        collection="d1\tbody\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\tghost\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="unknown corpus ID"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "bad-qrels",
        collection="d1\tbody\n",
        queries="q1\tquestion\n",
        qrels="q1\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="exactly four"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "bad-score",
        collection="d1\tbody\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\thigh\n",
    )
    with pytest.raises(DatasetValidationError, match="must be an integer"):
        load_msmarco_dataset(root)


def test_load_msmarco_dataset_duplicate_qrels_last_row_wins(tmp_path: Path) -> None:
    root = _write_msmarco(
        tmp_path / "dups",
        collection="d1\tone\nd2\ttwo\n",
        queries="q1\tfirst\nq2\tsecond\n",
        qrels=(
            "q1\t0\td1\t2\n"
            "q1\t0\td1\t1\n"
            "q1\t0\td2\t2\n"
            "q1\t0\td2\t0\n"
            "q2\t0\td1\t0\n"
            "q2\t0\td1\t2\n"
        ),
    )
    benchmark = load_msmarco_dataset(root)
    assert benchmark.dataset.relevance_grades_by_query["q1"] == {"d1": 1}
    assert benchmark.dataset.relevance_grades_by_query["q2"] == {"d1": 2}


def _nq_record(
    example_id: object,
    *,
    question: str = "a question",
    document: str = "alpha beta gamma delta",
    candidate_index: int = 0,
    start_token: int = 0,
    end_token: int = 2,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "question_text": question,
        "document_text": document,
        "annotations": [
            {
                "long_answer": {
                    "candidate_index": candidate_index,
                    "start_token": start_token,
                    "end_token": end_token,
                },
                "short_answers": [],
                "yes_no_answer": "NONE",
            }
        ],
    }


def _write_nq(root: Path, records: list[dict[str, object]]) -> Path:
    root.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


def test_load_natural_questions_dataset_converts_records(tmp_path: Path) -> None:
    path = _write_nq(
        tmp_path / "nq.jsonl",
        [
            _nq_record(101, question="first question"),
            _nq_record(-202, question="second question", candidate_index=1),
            _nq_record("303", question="unanswered question", candidate_index=-1),
        ],
    )

    benchmark = load_natural_questions_dataset(path)

    assert [document.id for document in benchmark.documents] == ["101", "-202", "303"]
    assert benchmark.documents[0].text == "alpha beta gamma delta"
    assert [query.id for query in benchmark.dataset.queries] == ["101", "-202"]
    assert benchmark.dataset.queries[0].query == "first question"
    assert benchmark.dataset.queries[0].relevant_document_ids == frozenset({"101"})
    assert benchmark.dataset.relevance_grades_by_query["-202"] == {"-202": 1}
    assert benchmark.excluded_query_ids == ("303",)


def test_load_natural_questions_dataset_accepts_any_positive_annotation(
    tmp_path: Path,
) -> None:
    record = _nq_record(7, candidate_index=-1)
    annotations = record["annotations"]
    assert isinstance(annotations, list)
    annotations.append(
        {
            "long_answer": {
                "candidate_index": 2,
                "start_token": 1,
                "end_token": 3,
            },
            "short_answers": [],
            "yes_no_answer": "NONE",
        }
    )
    benchmark = load_natural_questions_dataset(
        _write_nq(tmp_path / "nq.jsonl", [record])
    )
    assert benchmark.excluded_query_ids == ()
    assert benchmark.dataset.queries[0].id == "7"


def test_load_natural_questions_dataset_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(DatasetValidationError):
        load_natural_questions_dataset(tmp_path / "does-not-exist.jsonl")

    path = _write_nq(tmp_path / "dup.jsonl", [_nq_record(1), _nq_record(1)])
    with pytest.raises(DatasetValidationError, match="duplicate example_id"):
        load_natural_questions_dataset(path)

    path = _write_nq(
        tmp_path / "bad-span.jsonl", [_nq_record(1, start_token=0, end_token=99)]
    )
    with pytest.raises(DatasetValidationError, match="token span"):
        load_natural_questions_dataset(path)

    record = _nq_record(1)
    record["annotations"] = []
    path = _write_nq(tmp_path / "no-annotations.jsonl", [record])
    with pytest.raises(DatasetValidationError, match="annotations"):
        load_natural_questions_dataset(path)

    path = _write_nq(
        tmp_path / "none-answered.jsonl", [_nq_record(1, candidate_index=-1)]
    )
    with pytest.raises(DatasetValidationError, match="no examples"):
        load_natural_questions_dataset(path)


def test_load_msmarco_dataset_skips_blank_lines_and_rejects_bad_rows(
    tmp_path: Path,
) -> None:
    root = _write_msmarco(
        tmp_path / "blanks",
        collection="\nd1\tbody\n\n",
        queries="q1\tquestion\n",
        qrels="\nq1\t0\td1\t1\n",
    )
    assert load_msmarco_dataset(root).documents[0].id == "d1"

    with pytest.raises(DatasetValidationError):
        load_msmarco_dataset(root, split="")

    root = _write_msmarco(
        tmp_path / "no-tab",
        collection="d1 body without a tab\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="exactly two"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "empty-id",
        collection="\tbody\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="ID must be a non-empty"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "empty-text",
        collection="d1\t \n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="text must be a non-empty"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "empty-collection",
        collection="",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "unknown-query",
        collection="d1\tbody\n",
        queries="q1\tquestion\n",
        qrels="ghost\t0\td1\t1\n",
    )
    with pytest.raises(DatasetValidationError, match="unknown query"):
        load_msmarco_dataset(root)

    root = _write_msmarco(
        tmp_path / "negative",
        collection="d1\tbody\n",
        queries="q1\tquestion\n",
        qrels="q1\t0\td1\t-1\n",
    )
    with pytest.raises(DatasetValidationError, match="must not be negative"):
        load_msmarco_dataset(root)


def test_load_natural_questions_dataset_rejects_malformed_records(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        load_natural_questions_dataset(empty)

    record = _nq_record(1, question=" ")
    path = _write_nq(tmp_path / "no-question.jsonl", [record])
    with pytest.raises(DatasetValidationError, match="question_text"):
        load_natural_questions_dataset(path)

    record = _nq_record(1, document="")
    path = _write_nq(tmp_path / "no-document.jsonl", [record])
    with pytest.raises(DatasetValidationError, match="document_text"):
        load_natural_questions_dataset(path)

    for bad_id in (True, 1.5, ""):
        path = _write_nq(tmp_path / "bad-id.jsonl", [_nq_record(bad_id)])
        with pytest.raises(DatasetValidationError, match="example_id"):
            load_natural_questions_dataset(path)

    record = _nq_record(1)
    record["annotations"] = ["not-an-object"]
    path = _write_nq(tmp_path / "bad-annotation.jsonl", [record])
    with pytest.raises(DatasetValidationError, match="contain objects"):
        load_natural_questions_dataset(path)

    record = _nq_record(1)
    record["annotations"] = [{"short_answers": []}]
    path = _write_nq(tmp_path / "no-long-answer.jsonl", [record])
    with pytest.raises(DatasetValidationError, match="long_answer object"):
        load_natural_questions_dataset(path)

    record = _nq_record(1)
    annotations = record["annotations"]
    assert isinstance(annotations, list)
    annotations[0]["long_answer"]["candidate_index"] = "0"
    path = _write_nq(tmp_path / "bad-candidate.jsonl", [record])
    with pytest.raises(DatasetValidationError, match="candidate_index"):
        load_natural_questions_dataset(path)
