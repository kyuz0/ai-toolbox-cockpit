"""Copy text to the clipboard through the terminal with OSC 52."""

from __future__ import annotations

import base64
import os
import sys


def osc52_sequence(text: str) -> str:
    """Return the OSC 52 escape sequence that sets the clipboard to text."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    return f"\033]52;c;{encoded}\a"


def tmux_passthrough(sequence: str) -> str:
    """Wrap an escape sequence so tmux relays it to the outer terminal.

    tmux consumes OSC 52 itself and only re-emits it when set-clipboard is
    configured to forward clipboard sequences, so the wrapped form lets the
    sequence through even when tmux is not. tmux strips the DCS wrapper and
    passes the inner sequence to the outer terminal unchanged.
    """
    doubled = sequence.replace("\033", "\033\033")
    return f"\033Ptmux;{doubled}\033\\"


def _terminal_stream():
    """Return the stream that actually reaches the terminal.

    While a Textual app runs, sys.stdout and sys.stderr are redirected to
    Textual's capture buffers, so writes to them never reach the terminal.
    The dunder streams still point at the original terminal — the same
    stream Textual's own driver writes the UI and its clipboard sequence to.
    """
    candidates = (
        getattr(sys, "__stderr__", None),
        getattr(sys, "__stdout__", None),
        sys.stdout,
    )
    for stream in candidates:
        if stream is not None:
            return stream
    return sys.stdout


def copy_to_clipboard_osc52(text: str, stream=None) -> None:
    """Copy text to the local clipboard using the OSC 52 escape sequence.

    The terminal emulator on the user's desktop performs the copy, so this
    works when Cockpit runs on a remote host over SSH or inside tmux, where
    clipboard helpers on the remote machine would target the remote,
    headless clipboard instead of the user's desktop. Inside tmux the
    sequence is additionally wrapped in a DCS passthrough so tmux relays it
    to the outer terminal instead of applying it to the remote session.
    """
    try:
        sequence = osc52_sequence(text)
        if os.environ.get("TMUX"):
            sequence += tmux_passthrough(sequence)
        out = stream if stream is not None else _terminal_stream()
        out.write(sequence)
        out.flush()
    except Exception:
        # Clipboard support is best-effort and must never break the TUI.
        pass
