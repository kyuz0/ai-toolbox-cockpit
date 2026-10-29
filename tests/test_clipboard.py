import base64
import io
import os
from contextlib import redirect_stdout
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.clipboard import (
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


class CopyToClipboardOsc52Tests(TestCase):
    def test_writes_plain_sequence_outside_tmux(self) -> None:
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(stdout):
                copy_to_clipboard_osc52("hello")
        self.assertEqual(stdout.getvalue(), OSC52_HELLO)

    def test_adds_tmux_passthrough_inside_tmux(self) -> None:
        stdout = io.StringIO()
        with patch.dict(os.environ, TMUX_ENV, clear=True):
            with redirect_stdout(stdout):
                copy_to_clipboard_osc52("hello")
        self.assertEqual(stdout.getvalue(), OSC52_HELLO + TMUX_PASSTHROUGH_HELLO)

    def test_swallows_stdout_errors(self) -> None:
        with patch("sys.stdout") as stdout:
            stdout.write.side_effect = OSError("closed")
            copy_to_clipboard_osc52("hello")


class AppCopyToClipboardTests(TestCase):
    def test_app_copies_with_osc52_and_tracks_clipboard(self) -> None:
        app = AiToolboxCockpitApp()
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(stdout):
                app.copy_to_clipboard("hello")
        self.assertEqual(app.clipboard, "hello")
        self.assertEqual(stdout.getvalue(), OSC52_HELLO)


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

                stdout = io.StringIO()
                with patch.dict(os.environ, {}, clear=True):
                    with redirect_stdout(stdout):
                        await pilot.click("#btn_copy")
                        await pilot.pause()

                self.assertEqual(stdout.getvalue(), f"\033]52;c;{encoded}\a")
                self.assertEqual(app.clipboard, command)
                self.assertIsInstance(app.screen, ConfirmModal)
