"""The thin ``retrieval-lab`` command-line application."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from retrieval_lab import (
    ComparisonOutput,
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    EvaluationError,
    GateOutput,
    InspectionOutput,
    OptionalDependencyError,
    RetrievalLabError,
    compare_result_files,
    evaluate_configured_quality_gates,
    initialize_project,
    inspect_result,
    run_configured_experiment,
    validate_config_inputs,
)
from retrieval_lab.cli.dataset import configure_dataset_parser, run_dataset_command


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading files or executing an experiment."""

    parser = argparse.ArgumentParser(
        prog="retrieval-lab",
        description="Run offline Retrieval Lab retrieval evaluations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a runnable project template")
    init.add_argument("target", type=Path, help="project directory")
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite the template files owned by Retrieval Lab",
    )
    init.add_argument("--debug", action="store_true")

    validate = commands.add_parser("validate", help="validate config and inputs")
    validate.add_argument("-c", "--config", required=True, type=Path)
    validate.add_argument("--debug", action="store_true")

    run = commands.add_parser("run", help="run an evaluation and save reports")
    run.add_argument("-c", "--config", required=True, type=Path)
    run.add_argument("-o", "--output-dir", type=Path)
    run.add_argument(
        "-f",
        "--format",
        dest="formats",
        action="append",
        choices=("json", "csv", "html"),
        help="report format; may be supplied more than once",
    )
    run.add_argument("--debug", action="store_true")

    inspect_command = commands.add_parser(
        "inspect", help="inspect a saved evaluation result"
    )
    inspect_command.add_argument("result", type=Path)
    inspect_command.add_argument("--query-id", type=str)
    inspect_command.add_argument("--json", dest="json_output", action="store_true")
    inspect_command.add_argument("--debug", action="store_true")

    compare = commands.add_parser("compare", help="compare two saved results")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--json", dest="json_output", action="store_true")
    compare.add_argument("--debug", action="store_true")

    gate = commands.add_parser("gate", help="evaluate configured quality gates")
    gate.add_argument("-c", "--config", type=Path)
    gate.add_argument("candidate", type=Path)
    gate.add_argument("--baseline", type=Path)
    gate.add_argument("--json", dest="json_output", action="store_true")
    gate.add_argument("--debug", action="store_true")

    configure_dataset_parser(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process status code."""

    parser = build_parser()
    args: argparse.Namespace | None = None
    try:
        args = parser.parse_args(argv)
        if args.command == "init":
            project = initialize_project(args.target, force=args.force)
            names = ", ".join(
                path.relative_to(project.target).as_posix() for path in project.files
            )
            print(f"Initialized project ({names})")
            return 0
        if args.command == "validate":
            validated = validate_config_inputs(args.config)
            retrievers = ", ".join(validated.retriever_names)
            print(
                f"Configuration valid ({validated.document_count} documents, "
                f"{validated.query_count} queries; retrievers: {retrievers})"
            )
            return 0
        if args.command == "run":
            output = run_configured_experiment(
                args.config,
                output_dir=args.output_dir,
                formats=args.formats,
            )
            names = ", ".join(path.name for path in output.paths)
            print(f"Evaluation complete ({names})")
            return 0
        if args.command == "inspect":
            inspection = inspect_result(args.result, query_id=args.query_id)
            _emit_inspection(inspection, json_output=args.json_output)
            return 0
        if args.command == "compare":
            comparison = compare_result_files(args.baseline, args.candidate)
            _emit_comparison(comparison, json_output=args.json_output)
            return 0
        if args.command == "gate":
            gate_output = evaluate_configured_quality_gates(
                args.config,
                args.candidate,
                baseline_path=args.baseline,
            )
            _emit_gate(gate_output, json_output=args.json_output)
            return 0 if gate_output.report.passed else 1
        if args.command == "dataset":
            return run_dataset_command(args)
        parser.error("a command is required")
    except (ConfigurationError, CorpusValidationError, DatasetValidationError) as exc:
        _write_error(_input_error_message(exc), debug=_is_debug(args))
        return 2
    except (OptionalDependencyError, EvaluationError, RetrievalLabError) as exc:
        _write_error(_evaluation_error_message(exc), debug=_is_debug(args))
        return 3 if args is not None and args.command == "run" else 2
    except (OSError, ValueError, TypeError) as exc:
        _write_error(_input_error_message(exc), debug=_is_debug(args))
        return 2
    except Exception:
        _write_error("unexpected runtime error", debug=_is_debug(args))
        return 3


def _is_debug(args: argparse.Namespace | None) -> bool:
    return args is not None and bool(getattr(args, "debug", False))


def _write_error(
    message: str,
    *,
    debug: bool = False,
) -> None:
    if debug:
        traceback.print_exc()
    print(f"retrieval-lab: {message}", file=sys.stderr)


def _input_error_message(_exc: BaseException) -> str:
    return "configuration or input error"


def _evaluation_error_message(_exc: BaseException) -> str:
    return "evaluation error"


def _emit_inspection(output: InspectionOutput, *, json_output: bool) -> None:
    if json_output:
        print(output.to_json(), end="")
        return
    result = output.result
    print(f"run_id: {result.run_id}")
    print(f"schema_version: {result.schema_version}")
    print(f"retrievers: {', '.join(sorted(result.metrics))}")
    print("quality_gates:")
    if not output.gate_status:
        print("  none")
    for index, retriever, metric, passed in output.gate_status:
        print(f"  [{index}] {retriever} {metric}: {'PASS' if passed else 'FAIL'}")
    print("summary:")
    print(result.summary(), end="")
    if output.query_id is not None:
        print(f"query_evidence: {output.query_id}")
        for evidence in output.evidence:
            print(f"  retriever: {evidence.retriever}")
            print(f"    retrieved_ids: {', '.join(evidence.retrieved_ids)}")
            for cutoff, ids in evidence.retrieved_ids_by_cutoff:
                print(f"    retrieved_ids@{cutoff}: {', '.join(ids)}")
            for metric, value in evidence.metrics:
                print(f"    {metric}: {value:.12g}")
            if evidence.search_latency_ms is not None:
                print(f"    search_latency_ms: {evidence.search_latency_ms:.12g}")
            for warning in evidence.warnings:
                print(f"    warning: {warning}")


def _emit_comparison(output: ComparisonOutput, *, json_output: bool) -> None:
    if json_output:
        print(output.to_json(), end="")
        return
    comparison = output.comparison
    print(f"baseline_run_id: {comparison.baseline_run_id}")
    print(f"candidate_run_id: {comparison.candidate_run_id}")
    print(f"common_retrievers: {', '.join(comparison.comparability.common_retrievers)}")
    print("metrics:")
    for row in output.rows:
        identity = row.metric if row.cutoff is None else f"{row.metric}@{row.cutoff}"
        relative = (
            "null" if row.relative_delta is None else f"{row.relative_delta:.12g}"
        )
        print(
            f"  {row.retriever} {identity}: baseline={row.baseline:.12g}, "
            f"candidate={row.candidate:.12g}, "
            f"absolute_delta={row.absolute_delta:.12g}, "
            f"relative_delta={relative}, direction={row.direction}, "
            f"classification={row.classification}"
        )
    for issue in comparison.comparability.diagnostics:
        print(f"diagnostic: {issue.field}: {issue.reason}")
    for issue in comparison.comparability.variable_differences:
        print(f"experimental_difference: {issue.field}: {issue.reason}")


def _emit_gate(output: GateOutput, *, json_output: bool) -> None:
    if json_output:
        print(output.to_json(), end="")
        return
    report = output.report
    print(f"candidate_run_id: {report.candidate_run_id}")
    if report.baseline_run_id is not None:
        print(f"baseline_run_id: {report.baseline_run_id}")
    print(f"passed: {'PASS' if report.passed else 'FAIL'}")
    for result in report.results:
        print(
            f"gate[{result.gate_index}] {result.retriever} {result.metric}: "
            f"{'PASS' if result.passed else 'FAIL'}"
        )
        for check in result.checks:
            actual = "null" if check.actual is None else f"{check.actual:.12g}"
            print(
                f"  {check.constraint}: {'PASS' if check.passed else 'FAIL'} "
                f"actual={actual}, threshold={check.threshold:.12g}"
            )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
