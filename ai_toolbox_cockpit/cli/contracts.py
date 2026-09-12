"""Types shared by CLI parsing, command handlers, and renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


CLI_SCHEMA_VERSION = 2
EXIT_FAILURE = 1
EXIT_INVALID_ARGUMENTS = 2
EXIT_UNAVAILABLE = 3


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True)
class Invocation:
    command: tuple[str, ...]
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ParsedInvocation:
    invocation: Invocation
    output: OutputFormat


@dataclass(frozen=True)
class CommandResult:
    data: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    exit_code: int = 0


@dataclass(frozen=True)
class CommandError:
    code: str
    message: str
    exit_code: int
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandEvent:
    kind: str
    sequence: int
    data: Mapping[str, Any]
    run_id: str = ""
    session_id: str = ""
    stream: str = "stdout"


class CliParseError(ValueError):
    """An argument error that can be rendered using the selected output format."""
