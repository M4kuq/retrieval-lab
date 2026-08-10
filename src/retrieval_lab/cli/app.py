"""The thin ``retrieval-lab`` command-line application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from retrieval_lab.application import (
    initialize_project,
    run_configured_experiment,
    validate_config_inputs,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    EvaluationError,
    OptionalDependencyError,
    RetrievalLabError,
)


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

    validate = commands.add_parser("validate", help="validate config and inputs")
    validate.add_argument("-c", "--config", required=True, type=Path)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process status code."""

    parser = build_parser()
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
        parser.error("a command is required")
    except (ConfigurationError, CorpusValidationError, DatasetValidationError) as exc:
        _write_error(_input_error_message(exc))
        return 2
    except (OptionalDependencyError, EvaluationError, RetrievalLabError) as exc:
        _write_error(_evaluation_error_message(exc))
        return 2
    except (OSError, ValueError, TypeError) as exc:
        _write_error(_input_error_message(exc))
        return 2
    except Exception:
        _write_error("unexpected runtime error")
        return 3


def _write_error(message: str) -> None:
    print(f"retrieval-lab: {message}", file=sys.stderr)


def _input_error_message(_exc: BaseException) -> str:
    return "configuration or input error"


def _evaluation_error_message(_exc: BaseException) -> str:
    return "evaluation error"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
