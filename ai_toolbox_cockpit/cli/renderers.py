"""Output encoders for completed CLI commands and future event streams."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TextIO

from .contracts import CLI_SCHEMA_VERSION, CommandError, CommandEvent, CommandResult, OutputFormat
from .registry import CommandSpec


def render_result(result: CommandResult, command: CommandSpec, output: OutputFormat, stdout: TextIO) -> None:
    if output is OutputFormat.JSON:
        _write_json(
            stdout,
            {
                "schema_version": CLI_SCHEMA_VERSION,
                "command": list(command.path),
                "ok": True,
                "data": result.data,
                "warnings": list(result.warnings),
            },
        )
        return
    if command.text_presenter is not None:
        stdout.write(command.text_presenter(result.data))
    else:
        stdout.write("Command completed.\n")


def render_error(error: CommandError, command: tuple[str, ...], output: OutputFormat, stderr: TextIO) -> None:
    if output is OutputFormat.JSON:
        _write_json(
            stderr,
            {
                "schema_version": CLI_SCHEMA_VERSION,
                "command": list(command),
                "ok": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                },
            },
        )
        return
    stderr.write(f"Error: {error.message}\n")


def render_events(events: Iterable[CommandEvent], command: tuple[str, ...], output: OutputFormat, stdout: TextIO, stderr: TextIO) -> None:
    """Render the future foreground server event protocol without owning lifecycle."""
    for event in events:
        if output is OutputFormat.JSON:
            _write_json_line(
                stdout,
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "command": list(command),
                    "event": event.kind,
                    "sequence": event.sequence,
                    "run_id": event.run_id,
                    "session_id": event.session_id,
                    "stream": event.stream,
                    "data": event.data,
                },
            )
            continue
        destination = stderr if event.stream == "stderr" else stdout
        message = event.data.get("message", "")
        destination.write(f"{message}\n" if message else "")


def _write_json(stream: TextIO, payload: object) -> None:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def _write_json_line(stream: TextIO, payload: object) -> None:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")
