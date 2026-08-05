"""Typed evaluation results and canonical JSON serialization."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from retrieval_lab.exceptions import EvaluationError

from ._validation import (
    normalize_json_mapping,
    require_finite_float,
    require_non_empty_string,
    require_positive_int,
)
from .json_types import JSONValue


def _normalize_metrics(
    values: Mapping[int, Mapping[str, float]],
    *,
    owner: str,
) -> Mapping[int, Mapping[str, float]]:
    if not isinstance(values, Mapping) or not values:
        raise EvaluationError(f"{owner}.metrics_by_cutoff must not be empty")

    normalized: dict[int, Mapping[str, float]] = {}
    for raw_cutoff, raw_metrics in values.items():
        cutoff = require_positive_int(
            raw_cutoff,
            field_name=f"{owner}.metrics_by_cutoff cutoff",
            error_type=EvaluationError,
        )
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise EvaluationError(
                f"{owner}.metrics_by_cutoff[{cutoff}] must not be empty"
            )
        metrics: dict[str, float] = {}
        for raw_name, raw_value in raw_metrics.items():
            name = require_non_empty_string(
                raw_name,
                field_name=f"{owner}.metrics_by_cutoff[{cutoff}] metric name",
                error_type=EvaluationError,
            )
            metrics[name] = require_finite_float(
                raw_value,
                field_name=f"{owner}.metrics_by_cutoff[{cutoff}][{name!r}]",
                error_type=EvaluationError,
            )
        normalized[cutoff] = MappingProxyType(metrics)
    return MappingProxyType(dict(sorted(normalized.items())))


def _flatten_metrics(
    metrics_by_cutoff: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    return {
        f"{name}@{cutoff}": value
        for cutoff, metrics in metrics_by_cutoff.items()
        for name, value in sorted(metrics.items())
    }


@dataclass(frozen=True)
class QueryEvaluation:
    """Per-query ranking evidence and metric values at each cutoff."""

    query_id: str
    retrieved_ids: tuple[str, ...]
    metrics_by_cutoff: Mapping[int, Mapping[str, float]]

    def __post_init__(self) -> None:
        query_id = require_non_empty_string(
            self.query_id,
            field_name="QueryEvaluation.query_id",
            error_type=EvaluationError,
        )
        if isinstance(self.retrieved_ids, (str, bytes)) or not isinstance(
            self.retrieved_ids, Sequence
        ):
            raise EvaluationError(
                f"QueryEvaluation[{query_id!r}].retrieved_ids must be a sequence"
            )
        retrieved_ids = tuple(
            require_non_empty_string(
                item,
                field_name=f"QueryEvaluation[{query_id!r}].retrieved_ids item",
                error_type=EvaluationError,
            )
            for item in self.retrieved_ids
        )
        if len(set(retrieved_ids)) != len(retrieved_ids):
            raise EvaluationError(
                f"QueryEvaluation[{query_id!r}].retrieved_ids must be unique"
            )
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "retrieved_ids", retrieved_ids)
        object.__setattr__(
            self,
            "metrics_by_cutoff",
            _normalize_metrics(
                self.metrics_by_cutoff,
                owner=f"QueryEvaluation[{query_id!r}]",
            ),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return this query result in the canonical report shape."""

        metric_values: dict[str, JSONValue] = {
            key: value
            for key, value in _flatten_metrics(self.metrics_by_cutoff).items()
        }
        return {
            "metrics": metric_values,
            "query_id": self.query_id,
            "retrieved_ids": list(self.retrieved_ids),
        }


@dataclass(frozen=True)
class RetrieverMetrics:
    """Macro-averaged metric values for one retriever."""

    metrics_by_cutoff: Mapping[int, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics_by_cutoff",
            _normalize_metrics(
                self.metrics_by_cutoff,
                owner="RetrieverMetrics",
            ),
        )

    def recall_at(self, k: int) -> float:
        """Return macro Recall@k, raising ``KeyError`` if it was not evaluated."""

        if k not in self.metrics_by_cutoff:
            raise KeyError(f"Recall@{k} was not evaluated")
        try:
            return self.metrics_by_cutoff[k]["recall"]
        except KeyError as exc:
            raise KeyError(f"Recall@{k} was not evaluated") from exc

    def to_dict(self) -> dict[str, float]:
        """Return aggregate metrics with canonical ``metric@k`` keys."""

        return _flatten_metrics(self.metrics_by_cutoff)


@dataclass(frozen=True, init=False)
class EvaluationResult:
    """A deterministic evaluation result with aggregate and query evidence."""

    run_id: str
    metrics: Mapping[str, RetrieverMetrics]
    query_results: Mapping[str, tuple[QueryEvaluation, ...]]
    manifest: Mapping[str, JSONValue]
    schema_version: int

    def __init__(
        self,
        run_id: str,
        metrics: Mapping[str, RetrieverMetrics],
        query_results: Mapping[str, Sequence[QueryEvaluation]],
        manifest: Mapping[str, JSONValue] | None = None,
        schema_version: int = 1,
    ) -> None:
        """Create a result after validating retriever and schema consistency."""

        normalized_run_id = require_non_empty_string(
            run_id,
            field_name="EvaluationResult.run_id",
            error_type=EvaluationError,
        )
        if schema_version != 1:
            raise EvaluationError("EvaluationResult.schema_version must be 1")
        normalized_metrics = _normalize_retriever_metrics(metrics)
        normalized_query_results = _normalize_query_results(query_results)
        if normalized_metrics.keys() != normalized_query_results.keys():
            raise EvaluationError(
                "EvaluationResult metrics and query_results must contain the "
                "same retriever names"
            )
        normalized_manifest = normalize_json_mapping(
            {} if manifest is None else manifest,
            field_name="EvaluationResult.manifest",
            error_type=EvaluationError,
        )

        object.__setattr__(self, "run_id", normalized_run_id)
        object.__setattr__(self, "metrics", normalized_metrics)
        object.__setattr__(self, "query_results", normalized_query_results)
        object.__setattr__(self, "manifest", normalized_manifest)
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a fresh dictionary using the canonical result schema."""

        retrievers: dict[str, JSONValue] = {}
        for name in sorted(self.metrics):
            metric_values: dict[str, JSONValue] = {
                key: value for key, value in self.metrics[name].to_dict().items()
            }
            retrievers[name] = {
                "metrics": metric_values,
                "per_query": [
                    query_result.to_dict() for query_result in self.query_results[name]
                ],
            }
        return {
            "quality_gates": [],
            "retrievers": retrievers,
            "run": {
                "id": self.run_id,
                "manifest": _json_mapping_to_dict(self.manifest),
            },
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize this result as deterministic, UTF-8-friendly JSON."""

        try:
            payload = json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationError("Evaluation result could not be serialized") from exc
        return f"{payload}\n"

    def save_json(self, path: str | os.PathLike[str]) -> None:
        """Save canonical JSON as UTF-8, creating parent directories."""

        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self.to_json(), encoding="utf-8", newline="")
        except (OSError, TypeError, ValueError) as exc:
            raise EvaluationError(
                f"Evaluation result could not be saved to {path!s}"
            ) from exc


def _normalize_retriever_metrics(
    values: Mapping[str, RetrieverMetrics],
) -> Mapping[str, RetrieverMetrics]:
    if not isinstance(values, Mapping) or not values:
        raise EvaluationError("EvaluationResult.metrics must not be empty")
    normalized: dict[str, RetrieverMetrics] = {}
    for raw_name, value in values.items():
        name = require_non_empty_string(
            raw_name,
            field_name="EvaluationResult.metrics retriever name",
            error_type=EvaluationError,
        )
        if not isinstance(value, RetrieverMetrics):
            raise EvaluationError(
                f"EvaluationResult.metrics[{name!r}] must be RetrieverMetrics"
            )
        normalized[name] = value
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_query_results(
    values: Mapping[str, Sequence[QueryEvaluation]],
) -> Mapping[str, tuple[QueryEvaluation, ...]]:
    if not isinstance(values, Mapping) or not values:
        raise EvaluationError("EvaluationResult.query_results must not be empty")
    normalized: dict[str, tuple[QueryEvaluation, ...]] = {}
    for raw_name, raw_results in values.items():
        name = require_non_empty_string(
            raw_name,
            field_name="EvaluationResult.query_results retriever name",
            error_type=EvaluationError,
        )
        if isinstance(raw_results, (str, bytes)) or not isinstance(
            raw_results, Sequence
        ):
            raise EvaluationError(
                f"EvaluationResult.query_results[{name!r}] must be a sequence"
            )
        results = tuple(raw_results)
        if not results:
            raise EvaluationError(
                f"EvaluationResult.query_results[{name!r}] must not be empty"
            )
        if not all(isinstance(item, QueryEvaluation) for item in results):
            raise EvaluationError(
                f"EvaluationResult.query_results[{name!r}] contains an "
                "invalid query result"
            )
        query_ids = [item.query_id for item in results]
        if len(set(query_ids)) != len(query_ids):
            raise EvaluationError(
                f"EvaluationResult.query_results[{name!r}] query IDs must be unique"
            )
        normalized[name] = results
    return MappingProxyType(dict(sorted(normalized.items())))


def _json_mapping_to_dict(values: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result[key] = _json_mapping_to_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _json_mapping_to_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


__all__ = ["EvaluationResult", "QueryEvaluation", "RetrieverMetrics"]
