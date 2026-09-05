"""Dispatch parsed CLI invocations to explicitly registered handlers."""

from __future__ import annotations

from .contracts import CommandError, CommandResult, Invocation
from .registry import command_for_path


def dispatch(invocation: Invocation) -> CommandResult | CommandError:
    return command_for_path(invocation.command).handler(invocation)
