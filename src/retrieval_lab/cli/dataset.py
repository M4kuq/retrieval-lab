"""Thin CLI presentation for persisted dataset-authoring workflows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from retrieval_lab import (
    DatasetDraftStatus,
    DatasetValidationError,
    dataset_draft_status,
    finalize_dataset_bundle,
    load_dataset_draft,
    load_documents,
    review_dataset_query,
)


def configure_dataset_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Attach dataset authoring commands to the root CLI parser."""

    dataset = commands.add_parser("dataset", help="author and review evaluation data")
    actions = dataset.add_subparsers(dest="dataset_command", required=True)

    status = actions.add_parser("status", help="show persisted draft review status")
    status.add_argument("bundle", type=Path)
    status.add_argument("--json", dest="json_output", action="store_true")
    status.add_argument("--debug", action="store_true")

    review = actions.add_parser("review", help="set relevance for one draft query")
    review.add_argument("bundle", type=Path)
    review.add_argument("--corpus", required=True, type=Path)
    review.add_argument("--query-id", required=True)
    review.add_argument(
        "--relevant",
        action="append",
        help="document ID or ID:GRADE; repeat for multiple positives",
    )
    review.add_argument("--complete-review", action="store_true")
    review.add_argument("--notes")
    review.add_argument("--json", dest="json_output", action="store_true")
    review.add_argument("--debug", action="store_true")

    finalize = actions.add_parser(
        "finalize", help="verify a draft can be loaded as an evaluation dataset"
    )
    finalize.add_argument("bundle", type=Path)
    finalize.add_argument("--json", dest="json_output", action="store_true")
    finalize.add_argument("--debug", action="store_true")


def run_dataset_command(args: argparse.Namespace) -> int:
    """Execute one dataset command using package-root Application APIs."""

    if args.dataset_command == "status":
        _emit_status(dataset_draft_status(args.bundle), json_output=args.json_output)
        return 0
    if args.dataset_command == "review":
        specs = args.relevant
        if specs is None:
            specs = _prompt_relevance(args.bundle, args.corpus, args.query_id)
        status = review_dataset_query(
            args.bundle,
            query_id=args.query_id,
            relevance=_parse_relevance_specs(specs),
            corpus=args.corpus,
            complete_review=args.complete_review,
            notes=args.notes,
        )
        _emit_status(status, json_output=args.json_output)
        return 0
    if args.dataset_command == "finalize":
        _emit_status(
            finalize_dataset_bundle(args.bundle),
            json_output=args.json_output,
        )
        return 0
    raise DatasetValidationError("unknown dataset command")


def _prompt_relevance(bundle: Path, corpus: Path, query_id: str) -> list[str]:
    draft = load_dataset_draft(bundle)
    query = next((item for item in draft.queries if item.id == query_id), None)
    if query is None:
        raise DatasetValidationError(f"unknown draft query ID {query_id!r}")
    documents = load_documents(corpus)
    print(f"query[{query.id}]: {query.query}")
    print("available documents:")
    for document in documents:
        preview = " ".join(document.text.split())[:80]
        print(f"  {document.id}: {preview}")
    raw = input("relevant document IDs (ID or ID:GRADE, comma-separated): ")
    specs = [item.strip() for item in raw.split(",") if item.strip()]
    if not specs:
        raise DatasetValidationError("at least one relevant document is required")
    return specs


def _parse_relevance_specs(values: list[str]) -> Mapping[str, int]:
    relevance: dict[str, int] = {}
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise DatasetValidationError("--relevant values must be non-empty")
        identifier, separator, grade_text = raw.rpartition(":")
        if not separator:
            identifier = raw
            grade = 1
        else:
            if not identifier.strip() or not grade_text.strip():
                raise DatasetValidationError(
                    "--relevant must use ID or ID:GRADE syntax"
                )
            try:
                grade = int(grade_text)
            except ValueError as exc:
                raise DatasetValidationError(
                    "--relevant grade must be an integer"
                ) from exc
        if grade < 1:
            raise DatasetValidationError("--relevant grade must be >= 1")
        if identifier in relevance:
            raise DatasetValidationError(
                f"duplicate --relevant document ID {identifier!r}"
            )
        relevance[identifier] = grade
    if not relevance:
        raise DatasetValidationError("at least one relevant document is required")
    return relevance


def _emit_status(status: DatasetDraftStatus, *, json_output: bool) -> None:
    if json_output:
        print(status.to_json(), end="")
        return
    print(f"queries: {status.complete_query_count}/{status.query_count} complete")
    print(f"relevance_level: {status.relevance_level}")
    print(f"origin: {status.origin}")
    print(f"review_status: {status.review_status}")
    print(f"reliability: {status.reliability}")
    if status.pending_query_ids:
        print(f"pending: {', '.join(status.pending_query_ids)}")


__all__ = ["configure_dataset_parser", "run_dataset_command"]
