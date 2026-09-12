"""Reserved server command handlers.

Runtime lifecycle, container labels, and event streaming are intentionally
deferred until they can be validated on remote GPU systems.
"""

from __future__ import annotations

from ..contracts import CommandError, EXIT_UNAVAILABLE, Invocation


def handle_reserved_server_command(invocation: Invocation) -> CommandError:
    return CommandError(
        code="not_implemented",
        message=f"{' '.join(invocation.command)} is not implemented yet. No runtime action was taken.",
        exit_code=EXIT_UNAVAILABLE,
    )
