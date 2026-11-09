import json
from pathlib import Path

import pytest

from retrieval_lab import BenchmarkDataset, DatasetValidationError, load_beir_dataset


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
