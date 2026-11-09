"""Crash-safe checkpoints for long-running evaluations.

A large evaluation spends its time producing one ranking per query.
Recording each ranking to a checkpoint file as soon as it exists
means an interrupted run — a crash, a killed process, a spot-instance
reclaim — resumes where it stopped instead of starting over: reopen
the checkpoint, and only the missing queries run again.

The checkpoint is a JSONL file. Its first line is a header carrying
the format name, a format version, and a caller-supplied
``fingerprint`` identifying the run (for example a hash of the
dataset, retriever configuration, and top-k values). Reopening a
checkpoint with a different fingerprint fails, so results from a
different configuration can never leak into a resumed run. Every
record is flushed and fsynced before the next query starts, and on
POSIX systems the containing directory is synced when the checkpoint
file is created, so the file's existence also survives an OS crash;
on other platforms the durability guarantee covers process crashes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from retrieval_lab.domain import EvaluationQuery
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult
from retrieval_lab.exceptions import EvaluationError

__all__ = [
    "EvaluationCheckpoint",
    "run_with_checkpoint",
]

_FORMAT_NAME = "retrieval-lab-evaluation-checkpoint"
_FORMAT_VERSION = 1


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class EvaluationCheckpoint:
    """An append-only JSONL store of per-query rankings for one run."""

    def __init__(self, path: str | Path, *, fingerprint: str) -> None:
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise EvaluationError("fingerprint must be a non-empty string")
        self._path = Path(path)
        self._fingerprint = fingerprint
        self._results: list[RetrievedQueryResult] = []
        self._completed: set[str] = set()
        self._broken = False
        try:
            exists = self._path.is_file() and self._path.stat().st_size > 0
        except OSError as error:
            raise EvaluationError(
                f"{self._path}: cannot inspect checkpoint file: {error}"
            ) from error
        if exists:
            self._load()
        else:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise EvaluationError(
                    f"{self._path}: cannot create checkpoint directory: {error}"
                ) from error
            self._write_line(self._header())
            self._sync_ancestors()

    @property
    def path(self) -> Path:
        """Return the checkpoint file path."""

        return self._path

    @property
    def fingerprint(self) -> str:
        """Return the run fingerprint this checkpoint is pinned to."""

        return self._fingerprint

    @property
    def completed_query_ids(self) -> frozenset[str]:
        """Return the query IDs whose rankings are already recorded."""

        return frozenset(self._completed)

    def __len__(self) -> int:
        return len(self._results)

    def results(self) -> tuple[RetrievedQueryResult, ...]:
        """Return every recorded ranking in the order it was recorded."""

        return tuple(self._results)

    def record(self, result: RetrievedQueryResult) -> None:
        """Append one query's ranking and flush it to disk."""

        if not isinstance(result, RetrievedQueryResult):
            raise EvaluationError("result must be a RetrievedQueryResult")
        if result.query_id in self._completed:
            raise EvaluationError(
                f"query {result.query_id!r} is already recorded in this checkpoint"
            )
        self._write_line(
            {
                "query_id": result.query_id,
                "retrieved_document_ids": list(result.retrieved_document_ids),
            }
        )
        self._results.append(result)
        self._completed.add(result.query_id)

    def _header(self) -> dict[str, object]:
        return {
            "format": _FORMAT_NAME,
            "version": _FORMAT_VERSION,
            "fingerprint": self._fingerprint,
        }

    def _sync_ancestors(self) -> None:
        for directory in (self._path.parent, *self._path.parent.parents):
            _sync_directory(directory)

    def _write_line(self, payload: dict[str, object]) -> None:
        if self._broken:
            raise EvaluationError(
                f"{self._path}: checkpoint is unusable after a failed write; "
                "reopen the checkpoint file to recover"
            )
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        try:
            size_before: int | None = self._path.stat().st_size
        except FileNotFoundError:
            size_before = None
        except OSError as error:
            # Nothing has been written yet, and without a trustworthy
            # pre-write size a later rollback could not tell this file
            # apart from one this call created — refuse to write.
            raise EvaluationError(
                f"{self._path}: cannot inspect checkpoint file before "
                f"appending: {error}"
            ) from error
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                if self._missing_trailing_newline():
                    # A crash between writing a record's JSON body and
                    # its newline leaves a valid but unterminated final
                    # line; terminate it before appending or the two
                    # records would concatenate into one corrupt line.
                    handle.write("\n")
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            # A failed append can leave a partial record behind. Roll
            # the file back to its pre-write size so a retry on this
            # instance — or a later reopen — never meets the fragment
            # as a corrupt middle line; if even the rollback fails,
            # refuse further writes on this instance.
            try:
                if size_before is None:
                    self._path.unlink()
                else:
                    self._truncate(size_before)
            except Exception:
                self._broken = True
            raise EvaluationError(
                f"{self._path}: cannot append to checkpoint file: {error}"
            ) from error

    def _missing_trailing_newline(self) -> bool:
        # A missing file needs no separator; any other read failure
        # propagates so the caller's rollback path handles it instead
        # of appending onto a line whose end was never verified.
        try:
            with self._path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    return False
                handle.seek(-1, os.SEEK_END)
                return handle.read(1) != b"\n"
        except FileNotFoundError:
            return False

    def _load(self) -> None:
        try:
            raw = self._path.read_bytes()
        except OSError as error:
            raise EvaluationError(
                f"{self._path}: cannot read checkpoint file: {error}"
            ) from error
        expected_header = json.dumps(
            self._header(), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        if (
            raw
            and b"\n" not in raw
            and len(raw) < len(expected_header)
            and expected_header.startswith(raw)
        ):
            # A crash while the header itself was being written left
            # a strict prefix of it and no records; start the file over.
            self._truncate(0)
            self._write_line(self._header())
            self._sync_ancestors()
            return
        pieces = raw.split(b"\n")
        offset = 0
        for line_number, piece in enumerate(pieces, start=1):
            is_last = line_number == len(pieces)
            if is_last and not piece:
                break
            try:
                text = piece.decode("utf-8")
            except UnicodeDecodeError:
                if is_last and line_number > 1:
                    self._truncate(offset)
                    break
                raise EvaluationError(
                    f"{self._path}:{line_number}: checkpoint is not valid UTF-8"
                ) from None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                if is_last and line_number > 1:
                    # A crash during append can leave one torn final
                    # line; drop it and keep every complete record.
                    self._truncate(offset)
                    break
                raise EvaluationError(
                    f"{self._path}:{line_number}: invalid JSON in checkpoint"
                ) from None
            offset += len(piece) + 1
            if line_number == 1:
                self._validate_header(parsed)
            else:
                self._load_record(parsed, line_number)

    def _truncate(self, size: int) -> None:
        try:
            with self._path.open("r+b") as handle:
                handle.truncate(size)
        except OSError as error:
            raise EvaluationError(
                f"{self._path}: cannot repair checkpoint file: {error}"
            ) from error

    def _validate_header(self, parsed: object) -> None:
        if not isinstance(parsed, dict) or parsed.get("format") != _FORMAT_NAME:
            raise EvaluationError(
                f"{self._path}:1: not a retrieval-lab evaluation checkpoint"
            )
        version = parsed.get("version")
        if type(version) is not int or version != _FORMAT_VERSION:
            raise EvaluationError(
                f"{self._path}:1: unsupported checkpoint version {version!r}"
            )
        if parsed.get("fingerprint") != self._fingerprint:
            raise EvaluationError(
                f"{self._path}:1: checkpoint fingerprint "
                f"{parsed.get('fingerprint')!r} does not match this run's "
                f"fingerprint {self._fingerprint!r}"
            )

    def _load_record(self, parsed: object, line_number: int) -> None:
        if not isinstance(parsed, dict):
            raise EvaluationError(
                f"{self._path}:{line_number}: checkpoint records must be objects"
            )
        query_id = parsed.get("query_id")
        document_ids = parsed.get("retrieved_document_ids")
        if not isinstance(query_id, str) or not isinstance(document_ids, list):
            raise EvaluationError(
                f"{self._path}:{line_number}: checkpoint record needs a string "
                "query_id and a list retrieved_document_ids"
            )
        try:
            result = RetrievedQueryResult(query_id, document_ids)
        except Exception as error:
            raise EvaluationError(
                f"{self._path}:{line_number}: invalid checkpoint record: {error}"
            ) from error
        if result.query_id in self._completed:
            raise EvaluationError(
                f"{self._path}:{line_number}: duplicate query "
                f"{result.query_id!r} in checkpoint"
            )
        self._results.append(result)
        self._completed.add(result.query_id)


def run_with_checkpoint(
    checkpoint: EvaluationCheckpoint,
    queries: Sequence[EvaluationQuery],
    run_query: Callable[[EvaluationQuery], RetrievedQueryResult],
) -> tuple[RetrievedQueryResult, ...]:
    """Run the queries missing from ``checkpoint`` and return all rankings.

    Queries already recorded are never re-run. Each new ranking is
    recorded before the next query starts, so if ``run_query`` raises,
    the exception propagates and every ranking produced so far stays
    in the checkpoint for the next attempt. The returned rankings are
    in the input query order regardless of recording order.
    """

    if not isinstance(checkpoint, EvaluationCheckpoint):
        raise EvaluationError("checkpoint must be an EvaluationCheckpoint")
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise EvaluationError("queries must be a sequence of EvaluationQuery values")
    ordered = tuple(queries)
    seen: set[str] = set()
    for position, query in enumerate(ordered):
        if not isinstance(query, EvaluationQuery):
            raise EvaluationError(f"queries[{position}] must be an EvaluationQuery")
        if query.id in seen:
            raise EvaluationError(f"duplicate query ID {query.id!r} in queries")
        seen.add(query.id)
    if not callable(run_query):
        raise EvaluationError("run_query must be callable")
    by_id = {result.query_id: result for result in checkpoint.results()}
    for query in ordered:
        if query.id in by_id:
            continue
        result = run_query(query)
        if not isinstance(result, RetrievedQueryResult):
            raise EvaluationError(
                f"run_query must return a RetrievedQueryResult for query {query.id!r}"
            )
        if result.query_id != query.id:
            raise EvaluationError(
                f"run_query returned query_id {result.query_id!r} for query "
                f"{query.id!r}"
            )
        checkpoint.record(result)
        by_id[query.id] = result
    return tuple(by_id[query.id] for query in ordered)
