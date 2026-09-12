"""The explicit, single source of truth for CLI commands and arguments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .commands.info import handle_info, render_info_text
from .commands.server import handle_reserved_server_command
from .contracts import CommandError, CommandResult, Invocation


CommandHandler = Callable[[Invocation], CommandResult | CommandError]
TextPresenter = Callable[[Mapping[str, Any]], str]
_DEFAULT_UNSET = object()


@dataclass(frozen=True)
class ArgumentSpec:
    flags: tuple[str, ...]
    help: str
    action: str | type[argparse.Action] | None = None
    choices: tuple[str, ...] | None = None
    value_type: type | None = None
    required: bool = False
    default: Any = _DEFAULT_UNSET
    group: str | None = None
    group_required: bool = False

    def add_to(self, target: Any, *, suppress_default: bool = False) -> None:
        kwargs: dict[str, Any] = {"help": self.help}
        if self.action is not None:
            kwargs["action"] = self.action
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.value_type is not None:
            kwargs["type"] = self.value_type
        if self.required:
            kwargs["required"] = True
        if suppress_default:
            kwargs["default"] = argparse.SUPPRESS
        elif self.default is not _DEFAULT_UNSET:
            kwargs["default"] = self.default
        target.add_argument(*self.flags, **kwargs)


@dataclass(frozen=True)
class CommandSpec:
    path: tuple[str, ...]
    help: str
    state: str
    handler: CommandHandler
    arguments: tuple[ArgumentSpec, ...] = ()
    text_presenter: TextPresenter | None = None


OUTPUT_ARGUMENT = ArgumentSpec(
    flags=("--output",),
    choices=("text", "json"),
    default="text",
    help="Write text (default) or JSON output.",
)

LLAMA_CPP_LAUNCH_ARGUMENTS = (
    ArgumentSpec(("--model",), required=True, help="Local GGUF model path, or auto."),
    ArgumentSpec(("--platform",), help="Platform catalogue ID."),
    ArgumentSpec(("--toolbox",), help="llama.cpp toolbox catalogue ID."),
    ArgumentSpec(("--engine",), choices=("podman", "docker"), help="Container engine."),
    ArgumentSpec(("--host",), help="Server bind address."),
    ArgumentSpec(("--port",), value_type=int, help="Server port."),
    ArgumentSpec(("--context-size",), value_type=int, help="Context size."),
    ArgumentSpec(("--gpu-layers",), value_type=int, help="GPU layers."),
    ArgumentSpec(("--load-mode",), choices=("none", "mmap", "dio"), help="Model load mode."),
    ArgumentSpec(
        ("--flash-attention",),
        action=argparse.BooleanOptionalAction,
        help="Enable or disable Flash Attention.",
    ),
    ArgumentSpec(("--kv-cache-type",), help="KV-cache quantization type."),
    ArgumentSpec(("--batch-size",), value_type=int, help="Logical batch size."),
    ArgumentSpec(("--ubatch-size",), value_type=int, help="Physical micro-batch size."),
    ArgumentSpec(("--parallel-sequences",), value_type=int, help="Parallel sequence count."),
)

SERVER_BACKENDS = ("llama-cpp", "ds4", "vllm", "comfyui")

COMMANDS = (
    CommandSpec(
        path=("info",),
        help="Print application and static catalogue information.",
        state="supported",
        handler=handle_info,
        arguments=(ArgumentSpec(("--full",), action="store_true", help="Include all static catalogue fields."),),
        text_presenter=render_info_text,
    ),
    CommandSpec(
        path=("server", "plan", "llama-cpp"),
        help="Preview a llama.cpp launch configuration (reserved).",
        state="reserved",
        handler=handle_reserved_server_command,
        arguments=LLAMA_CPP_LAUNCH_ARGUMENTS,
    ),
    CommandSpec(
        path=("server", "start", "llama-cpp"),
        help="Start llama.cpp in the foreground (reserved).",
        state="reserved",
        handler=handle_reserved_server_command,
        arguments=LLAMA_CPP_LAUNCH_ARGUMENTS,
    ),
    CommandSpec(
        path=("server", "status"),
        help="Show Cockpit-managed servers (reserved).",
        state="reserved",
        handler=handle_reserved_server_command,
        arguments=(
            ArgumentSpec(("--backend",), choices=SERVER_BACKENDS, help="Filter by backend."),
            ArgumentSpec(("--session",), help="Filter by session ID."),
        ),
    ),
    CommandSpec(
        path=("server", "logs"),
        help="Show managed server logs (reserved).",
        state="reserved",
        handler=handle_reserved_server_command,
        arguments=(
            ArgumentSpec(("--run",), required=True, help="Managed server run ID."),
            ArgumentSpec(("--follow",), action="store_true", help="Follow log output."),
        ),
    ),
    CommandSpec(
        path=("server", "stop"),
        help="Stop managed servers (reserved).",
        state="reserved",
        handler=handle_reserved_server_command,
        arguments=(
            ArgumentSpec(("--run",), group="target", group_required=True, help="Managed server run ID."),
            ArgumentSpec(("--session",), group="target", group_required=True, help="Managed server session ID."),
            ArgumentSpec(("--all",), action="store_true", group="target", group_required=True, help="Target all Cockpit-managed servers."),
            ArgumentSpec(("--yes",), action="store_true", help="Confirm a non-interactive bulk stop."),
        ),
    ),
)


def command_for_path(path: tuple[str, ...]) -> CommandSpec:
    for command in COMMANDS:
        if command.path == path:
            return command
    raise KeyError(path)


def command_capabilities() -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    for command in COMMANDS:
        current = capabilities
        for part in command.path[:-1]:
            current = current.setdefault(part, {})
        current[command.path[-1]] = command.state
    return capabilities
