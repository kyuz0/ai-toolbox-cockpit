"""Helpers for commands run while Textual has yielded control to the terminal."""

from __future__ import annotations

import signal
import sys


def command_failed(return_code: int) -> bool:
    """Treat normal completion and an intentional Ctrl+C as non-failures."""
    return return_code not in {0, 130, -signal.SIGINT}


def pause_after_failure(message: str) -> None:
    """Keep a failed command visible until the user acknowledges its output."""
    print(f"\n{message}", file=sys.stderr)
    try:
        input("\nPress Enter to return to AI Toolbox Cockpit...")
    except (EOFError, KeyboardInterrupt, OSError):
        # Non-interactive callers have no terminal to acknowledge the failure.
        pass
