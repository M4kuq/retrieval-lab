"""Command-line entry point for the local Retrieval Lab benchmark."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

if not __package__:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root / "src"))
    sys.path.insert(0, str(_repository_root))

from retrieval_lab.exceptions import EvaluationError

_harness = importlib.import_module("benchmarks.harness")
BenchmarkSpec = _harness.BenchmarkSpec
run_benchmark = _harness.run_benchmark
save_benchmark = _harness.save_benchmark


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _top_k_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from exc
    try:
        return cast(tuple[int, ...], BenchmarkSpec(top_k=values).top_k)
    except EvaluationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark parser without performing work."""

    parser = argparse.ArgumentParser(
        description="Run a deterministic local Retrieval Lab synthetic benchmark."
    )
    parser.add_argument("--size", choices=("small", "medium"), default="small")
    parser.add_argument("--seed", type=_non_negative_int, default=42)
    parser.add_argument(
        "--top-k",
        type=_top_k_values,
        default=(1, 3, 5),
        metavar="K[,K...]",
        help="sorted positive cutoffs (default: 1,3,5)",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
        help="fixed at 1 by the v0.1 runner",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path (default: benchmark-{size}.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and return a shell-friendly status code."""

    args = build_parser().parse_args(argv)
    try:
        spec = BenchmarkSpec(
            size=args.size,
            seed=args.seed,
            top_k=args.top_k,
            repetitions=args.repetitions,
        )
        payload = run_benchmark(spec)
        output = args.output or Path(f"benchmark-{args.size}.json")
        destination = save_benchmark(payload, output)
    except (EvaluationError, OSError, ValueError, TypeError):
        print("retrieval-lab benchmark: configuration or input error", file=sys.stderr)
        return 2
    except Exception:
        print("retrieval-lab benchmark: unexpected runtime error", file=sys.stderr)
        return 3
    print(f"benchmark report written: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
