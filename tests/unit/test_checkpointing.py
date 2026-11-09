import json
import os
from pathlib import Path

import pytest

from retrieval_lab import (
    EvaluationCheckpoint,
    EvaluationError,
    EvaluationQuery,
    RetrievedQueryResult,
    run_with_checkpoint,
)

FINGERPRINT = "dataset-v1:bm25:topk10"


def _query(query_id: str) -> EvaluationQuery:
    return EvaluationQuery(query_id, f"text of {query_id}", frozenset({"d1"}))


def _result(query_id: str, *document_ids: str) -> RetrievedQueryResult:
    return RetrievedQueryResult(query_id, document_ids or ("d1",))


def test_checkpoint_records_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "run.checkpoint.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    checkpoint.record(_result("q1", "d1", "d2"))
    checkpoint.record(_result("q2", "d3"))

    reopened = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert len(reopened) == 2
    assert reopened.completed_query_ids == frozenset({"q1", "q2"})
    assert [result.query_id for result in reopened.results()] == ["q1", "q2"]
    assert reopened.results()[0].retrieved_document_ids == ("d1", "d2")
    assert reopened.path == path
    assert reopened.fingerprint == FINGERPRINT


def test_checkpoint_rejects_duplicate_records(tmp_path: Path) -> None:
    checkpoint = EvaluationCheckpoint(tmp_path / "c.jsonl", fingerprint=FINGERPRINT)
    checkpoint.record(_result("q1"))
    with pytest.raises(EvaluationError):
        checkpoint.record(_result("q1"))
    with pytest.raises(EvaluationError):
        checkpoint.record(object())  # type: ignore[arg-type]


def test_checkpoint_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    EvaluationCheckpoint(path, fingerprint=FINGERPRINT).record(_result("q1"))
    with pytest.raises(EvaluationError, match="fingerprint"):
        EvaluationCheckpoint(path, fingerprint="other-config")
    with pytest.raises(EvaluationError):
        EvaluationCheckpoint(tmp_path / "x.jsonl", fingerprint=" ")


def test_checkpoint_rejects_foreign_or_corrupt_files(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.jsonl"
    foreign.write_text('{"hello": "world"}\n', encoding="utf-8")
    with pytest.raises(EvaluationError, match="not a retrieval-lab"):
        EvaluationCheckpoint(foreign, fingerprint=FINGERPRINT)

    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="invalid JSON"):
        EvaluationCheckpoint(garbage, fingerprint=FINGERPRINT)

    mid_corrupt = tmp_path / "mid.jsonl"
    checkpoint = EvaluationCheckpoint(mid_corrupt, fingerprint=FINGERPRINT)
    checkpoint.record(_result("q1"))
    with mid_corrupt.open("a", encoding="utf-8") as handle:
        handle.write("garbage\n")
        handle.write(
            json.dumps({"query_id": "q2", "retrieved_document_ids": ["d1"]}) + "\n"
        )
    with pytest.raises(EvaluationError, match="invalid JSON"):
        EvaluationCheckpoint(mid_corrupt, fingerprint=FINGERPRINT)


def test_checkpoint_drops_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "torn.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    checkpoint.record(_result("q1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"query_id": "q2", "retrieved_docu')  # crash mid-append

    reopened = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert reopened.completed_query_ids == frozenset({"q1"})
    reopened.record(_result("q2"))
    reread = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert reread.completed_query_ids == frozenset({"q1", "q2"})


def _write_checkpoint_file(
    path: Path, lines: list[str], *, trailing_newline: bool
) -> None:
    content = "\n".join(lines) + ("\n" if trailing_newline else "")
    path.write_text(content, encoding="utf-8")


def _header_line(**overrides: object) -> str:
    header: dict[str, object] = {
        "format": "retrieval-lab-evaluation-checkpoint",
        "version": 1,
        "fingerprint": FINGERPRINT,
    }
    header.update(overrides)
    return json.dumps(header)


def test_checkpoint_appends_after_unterminated_final_line(tmp_path: Path) -> None:
    record = json.dumps({"query_id": "q1", "retrieved_document_ids": ["d1"]})
    for name, lines in (
        ("header-only.jsonl", [_header_line()]),
        ("with-record.jsonl", [_header_line(), record]),
    ):
        path = tmp_path / name
        _write_checkpoint_file(path, lines, trailing_newline=False)
        checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
        checkpoint.record(_result("q2"))
        reread = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
        expected = {"q2"} if name.startswith("header") else {"q1", "q2"}
        assert reread.completed_query_ids == frozenset(expected)


@pytest.mark.parametrize("version", [True, 1.0, 2, "1"])
def test_checkpoint_rejects_non_integer_or_unknown_versions(
    tmp_path: Path, version: object
) -> None:
    path = tmp_path / "versioned.jsonl"
    _write_checkpoint_file(path, [_header_line(version=version)], trailing_newline=True)
    with pytest.raises(EvaluationError, match="version"):
        EvaluationCheckpoint(path, fingerprint=FINGERPRINT)


def test_checkpoint_io_errors_become_evaluation_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    with pytest.raises(EvaluationError):
        EvaluationCheckpoint(blocker / "c.jsonl", fingerprint=FINGERPRINT)

    directory = tmp_path / "a-directory"
    directory.mkdir()
    with pytest.raises(EvaluationError):
        EvaluationCheckpoint(directory, fingerprint=FINGERPRINT)

    path = tmp_path / "swapped.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    path.unlink()
    path.mkdir()
    with pytest.raises(EvaluationError):
        checkpoint.record(_result("q1"))


def test_checkpoint_recovers_a_torn_header(tmp_path: Path) -> None:
    path = tmp_path / "torn-header.jsonl"
    header = json.dumps(
        {
            "format": "retrieval-lab-evaluation-checkpoint",
            "version": 1,
            "fingerprint": FINGERPRINT,
        },
        sort_keys=True,
    )
    path.write_bytes(header.encode("utf-8")[:20])
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert len(checkpoint) == 0
    checkpoint.record(_result("q1"))
    reread = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert reread.completed_query_ids == frozenset({"q1"})

    foreign = tmp_path / "foreign-prefix.jsonl"
    foreign.write_bytes(b'{"format": "other-format", "vers')
    with pytest.raises(EvaluationError):
        EvaluationCheckpoint(foreign, fingerprint=FINGERPRINT)


def test_failed_record_rolls_back_and_retry_on_same_instance_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollback.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    checkpoint.record(_result("q1"))
    size_before = path.stat().st_size

    real_fsync = os.fsync
    state = {"fail": True}

    def flaky_fsync(descriptor: int) -> None:
        if state["fail"]:
            state["fail"] = False
            raise OSError("simulated fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", flaky_fsync)
    with pytest.raises(EvaluationError):
        checkpoint.record(_result("q2"))
    assert path.stat().st_size == size_before

    checkpoint.record(_result("q2"))
    reread = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert reread.completed_query_ids == frozenset({"q1", "q2"})


def test_stat_failure_before_append_never_deletes_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stat-failure.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    checkpoint.record(_result("q1"))
    content_before = path.read_bytes()

    real_stat = Path.stat
    state = {"fail": True}

    def flaky_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if state["fail"] and self == path:
            state["fail"] = False
            raise PermissionError("simulated stat failure")
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", flaky_stat)
    with pytest.raises(EvaluationError):
        checkpoint.record(_result("q2"))
    assert path.read_bytes() == content_before
    reread = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert reread.completed_query_ids == frozenset({"q1"})


def test_creation_write_failure_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "subdir" / "created.jsonl"

    def broken_fsync(descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", broken_fsync)
    with pytest.raises(EvaluationError) as excinfo:
        EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert not path.exists()

    monkeypatch.undo()
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    assert len(checkpoint) == 0


def test_unrecoverable_write_failure_poisons_the_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "poison.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)

    def broken_fsync(descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    def broken_truncate(self: EvaluationCheckpoint, size: int) -> None:
        raise EvaluationError("simulated rollback failure")

    monkeypatch.setattr(os, "fsync", broken_fsync)
    monkeypatch.setattr(EvaluationCheckpoint, "_truncate", broken_truncate)
    with pytest.raises(EvaluationError) as excinfo:
        checkpoint.record(_result("q1"))
    assert isinstance(excinfo.value.__cause__, OSError)
    with pytest.raises(EvaluationError, match="unusable after a failed write"):
        checkpoint.record(_result("q1"))


def test_run_with_checkpoint_runs_only_missing_queries(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    queries = [_query("q1"), _query("q2"), _query("q3")]
    calls: list[str] = []

    def run_query(query: EvaluationQuery) -> RetrievedQueryResult:
        calls.append(query.id)
        return _result(query.id)

    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    results = run_with_checkpoint(checkpoint, queries, run_query)
    assert calls == ["q1", "q2", "q3"]
    assert [result.query_id for result in results] == ["q1", "q2", "q3"]

    calls.clear()
    resumed = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    results = run_with_checkpoint(resumed, queries, run_query)
    assert calls == []
    assert [result.query_id for result in results] == ["q1", "q2", "q3"]


def test_run_with_checkpoint_resumes_after_failure(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    queries = [_query("q1"), _query("q2"), _query("q3")]
    attempts: list[str] = []

    def flaky(query: EvaluationQuery) -> RetrievedQueryResult:
        attempts.append(query.id)
        if query.id == "q2" and attempts.count("q2") == 1:
            raise RuntimeError("retriever exploded")
        return _result(query.id)

    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    with pytest.raises(RuntimeError, match="exploded"):
        run_with_checkpoint(checkpoint, queries, flaky)
    assert checkpoint.completed_query_ids == frozenset({"q1"})

    results = run_with_checkpoint(checkpoint, queries, flaky)
    assert [result.query_id for result in results] == ["q1", "q2", "q3"]
    assert attempts == ["q1", "q2", "q2", "q3"]


def test_run_with_checkpoint_returns_query_order(tmp_path: Path) -> None:
    path = tmp_path / "order.jsonl"
    checkpoint = EvaluationCheckpoint(path, fingerprint=FINGERPRINT)
    checkpoint.record(_result("q2"))
    queries = [_query("q1"), _query("q2"), _query("q3")]
    results = run_with_checkpoint(checkpoint, queries, lambda q: _result(q.id))
    assert [result.query_id for result in results] == ["q1", "q2", "q3"]


def test_run_with_checkpoint_validates_runner_output(tmp_path: Path) -> None:
    checkpoint = EvaluationCheckpoint(tmp_path / "c.jsonl", fingerprint=FINGERPRINT)
    with pytest.raises(EvaluationError, match="returned query_id"):
        run_with_checkpoint(checkpoint, [_query("q1")], lambda q: _result("other"))
    assert len(checkpoint) == 0
    with pytest.raises(EvaluationError):
        run_with_checkpoint(checkpoint, [_query("q1"), _query("q1")], _result)  # type: ignore[arg-type]
