import base64
import io
import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.clipboard import (
    _terminal_stream,
    copy_to_clipboard_osc52,
    osc52_sequence,
    tmux_passthrough,
)
from ai_toolbox_cockpit.widgets import ConfirmModal

OSC52_HELLO = "\033]52;c;aGVsbG8=\a"
TMUX_PASSTHROUGH_HELLO = "\033Ptmux;\033\033]52;c;aGVsbG8=\a\033\\"
TMUX_ENV = {"TMUX": "/tmp/tmux-1000/default,1234,0"}


class Osc52SequenceTests(TestCase):
    def test_sequence_targets_the_clipboard_selection(self) -> None:
        self.assertEqual(osc52_sequence("hello"), OSC52_HELLO)

    def test_sequence_carries_base64_utf8_text(self) -> None:
        text = "podman run image — ü"
        encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        self.assertEqual(osc52_sequence(text), f"\033]52;c;{encoded}\a")


class TmuxPassthroughTests(TestCase):
    def test_wraps_sequence_with_doubled_escape_and_dcs_frame(self) -> None:
        self.assertEqual(tmux_passthrough(OSC52_HELLO), TMUX_PASSTHROUGH_HELLO)


class TerminalStreamTests(TestCase):
    def test_prefers_the_original_stderr_stream(self) -> None:
        # While a Textual app runs, sys.stdout is redirected to a capture
        # buffer; the dunder streams still reach the real terminal.
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch("sys.__stderr__", stderr),
            patch("sys.__stdout__", None),
            patch("sys.stdout", stdout),
        ):
            self.assertIs(_terminal_stream(), stderr)

    def test_falls_back_to_stdout_when_no_original_stream(self) -> None:
        stdout = io.StringIO()
        with (
            patch("sys.__stderr__", None),
            patch("sys.__stdout__", None),
            patch("sys.stdout", stdout),
        ):
            self.assertIs(_terminal_stream(), stdout)


class CopyToClipboardOsc52Tests(TestCase):
    def test_writes_plain_sequence_outside_tmux(self) -> None:
        out = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            copy_to_clipboard_osc52("hello", stream=out)
        self.assertEqual(out.getvalue(), OSC52_HELLO)

    def test_adds_tmux_passthrough_inside_tmux(self) -> None:
        out = io.StringIO()
        with patch.dict(os.environ, TMUX_ENV, clear=True):
            copy_to_clipboard_osc52("hello", stream=out)
        self.assertEqual(out.getvalue(), OSC52_HELLO + TMUX_PASSTHROUGH_HELLO)

    def test_writes_to_original_terminal_stream_by_default(self) -> None:
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("sys.__stderr__", stderr),
            patch("sys.stdout", io.StringIO()),
        ):
            copy_to_clipboard_osc52("hello")
        self.assertEqual(stderr.getvalue(), OSC52_HELLO)

    def test_swallows_stream_errors(self) -> None:
        class BrokenStream:
            def write(self, data: str) -> None:
                raise OSError("closed")

        copy_to_clipboard_osc52("hello", stream=BrokenStream())


class AppCopyToClipboardTests(TestCase):
    def test_app_copies_with_osc52_and_tracks_clipboard(self) -> None:
        app = AiToolboxCockpitApp()
        stderr = io.StringIO()
        with patch("sys.__stderr__", stderr):
            app.copy_to_clipboard("hello")
        self.assertEqual(app.clipboard, "hello")
        self.assertEqual(stderr.getvalue(), OSC52_HELLO)


class CopyCommandButtonTests(IsolatedAsyncioTestCase):
    async def test_copy_command_button_emits_osc52(self) -> None:
        command = "podman run example/image:latest"
        encoded = base64.b64encode(command.encode("utf-8")).decode("utf-8")
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(100, 30)) as pilot:
                app.push_screen(ConfirmModal("Run this command?", copy_text=command))
                await pilot.pause()

                stderr = io.StringIO()
                with patch("sys.__stderr__", stderr):
                    await pilot.click("#btn_copy")
                    await pilot.pause()

                self.assertEqual(stderr.getvalue(), f"\033]52;c;{encoded}\a")
                self.assertEqual(app.clipboard, command)
                self.assertIsInstance(app.screen, ConfirmModal)
