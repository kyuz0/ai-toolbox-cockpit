"""Build and parse the CLI grammar declared in the command registry."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from .contracts import CliParseError, Invocation, OutputFormat, ParsedInvocation
from .registry import COMMANDS, OUTPUT_ARGUMENT, CommandSpec, command_for_path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliParseError(message)


def normalize_legacy_args(args: Sequence[str]) -> list[str]:
    """Map the explicitly supported legacy command spellings to canonical syntax."""
    normalized: list[str] = []
    command_seen = False
    output_value_expected = False
    for argument in args:
        if output_value_expected:
            normalized.append(argument)
            output_value_expected = False
            continue
        if argument == "-help":
            normalized.append("--help")
        elif argument == "--output":
            normalized.append(argument)
            output_value_expected = True
        elif argument.startswith("--output="):
            normalized.append(argument)
        elif not command_seen and argument == "-info":
            normalized.append("info")
            command_seen = True
        elif not command_seen and argument == "-fullinfo":
            normalized.extend(("info", "--full"))
            command_seen = True
        else:
            normalized.append(argument)
            if not argument.startswith("-"):
                command_seen = True
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="ai-toolbox-cockpit",
        description="Inspect AI Toolbox Cockpit and manage supported server workflows.",
    )
    _add_output_argument(parser, suppress_default=False)
    nodes: dict[tuple[str, ...], argparse.ArgumentParser] = {(): parser}
    subparsers: dict[tuple[str, ...], Any] = {}

    for command in sorted(COMMANDS, key=lambda item: item.path):
        for index, part in enumerate(command.path):
            parent_path = command.path[:index]
            path = command.path[: index + 1]
            if path in nodes:
                continue
            parent = nodes[parent_path]
            group = subparsers.get(parent_path)
            if group is None:
                group = parent.add_subparsers(dest=f"_command_{index}", required=True)
                subparsers[parent_path] = group
            child = group.add_parser(part, help=command.help if path == command.path else None)
            _add_output_argument(child, suppress_default=True)
            nodes[path] = child

        command_parser = nodes[command.path]
        _add_command_arguments(command_parser, command)
        command_parser.set_defaults(_command_path=command.path)
    return parser


def parse_invocation(args: Sequence[str]) -> ParsedInvocation:
    namespace = build_parser().parse_args(normalize_legacy_args(args))
    values = vars(namespace)
    path = values.get("_command_path")
    if path is None:
        raise CliParseError("a command is required")
    command = command_for_path(path)
    arguments = {
        key: value
        for key, value in values.items()
        if not key.startswith("_command_") and key not in {"_command_path", "output"}
    }
    return ParsedInvocation(
        invocation=Invocation(command=command.path, arguments=arguments),
        output=OutputFormat(values["output"]),
    )


def output_requested(args: Sequence[str]) -> OutputFormat:
    """Infer a requested output format before a parser error can be rendered."""
    normalized = normalize_legacy_args(args)
    requested = OutputFormat.TEXT
    for index, argument in enumerate(normalized):
        if argument.startswith("--output="):
            value = argument.split("=", 1)[1]
        elif argument == "--output" and index + 1 < len(normalized):
            value = normalized[index + 1]
        else:
            continue
        try:
            requested = OutputFormat(value)
        except ValueError:
            return OutputFormat.TEXT
    return requested


def _add_output_argument(parser: argparse.ArgumentParser, *, suppress_default: bool) -> None:
    OUTPUT_ARGUMENT.add_to(parser, suppress_default=suppress_default)


def _add_command_arguments(parser: argparse.ArgumentParser, command: CommandSpec) -> None:
    groups: dict[str, Any] = {}
    for argument in command.arguments:
        if argument.group is None:
            argument.add_to(parser)
            continue
        group = groups.get(argument.group)
        if group is None:
            group = parser.add_mutually_exclusive_group(required=argument.group_required)
            groups[argument.group] = group
        argument.add_to(group)
