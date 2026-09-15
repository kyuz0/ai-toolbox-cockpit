"""Public command-line entry point for AI Toolbox Cockpit."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO

from .contracts import CliParseError, CommandError, EXIT_FAILURE, EXIT_INVALID_ARGUMENTS
from .dispatch import dispatch
from .parser import output_requested, parse_invocation
from .registry import command_for_path
from .renderers import render_error, render_result


def run_cli(args: Sequence[str], stdout: TextIO, stderr: TextIO) -> int | None:
    if not args:
        return None
    output = output_requested(args)
    try:
        parsed = parse_invocation(args)
    except CliParseError as error:
        render_error(
            CommandError(code="invalid_arguments", message=str(error), exit_code=EXIT_INVALID_ARGUMENTS),
            (),
            output,
            stderr,
        )
        return EXIT_INVALID_ARGUMENTS

    try:
        outcome = dispatch(parsed.invocation)
    except Exception:
        render_error(
            CommandError(
                code="internal_error",
                message="An unexpected error occurred.",
                exit_code=EXIT_FAILURE,
            ),
            parsed.invocation.command,
            parsed.output,
            stderr,
        )
        return EXIT_FAILURE
    command = command_for_path(parsed.invocation.command)
    if isinstance(outcome, CommandError):
        render_error(outcome, parsed.invocation.command, parsed.output, stderr)
        return outcome.exit_code
    render_result(outcome, command, parsed.output, stdout)
    return outcome.exit_code


def run_from_argv(argv: Sequence[str] | None = None) -> int | None:
    return run_cli(sys.argv[1:] if argv is None else argv, sys.stdout, sys.stderr)