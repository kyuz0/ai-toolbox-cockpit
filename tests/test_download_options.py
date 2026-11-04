"""Network options persist and reach the download subprocess without network work."""

import os
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from textual.widgets import Checkbox, Input, Static, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.huggingface import DownloadOptions, get_download_options, huggingface_environment, save_download_options
from ai_toolbox_cockpit.settings import get_setting
from ai_toolbox_cockpit.views.download_options import DownloadOptionsModal


class DownloadOptionsUITests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=False))
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
        for target, value in [
            ("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", None),
            ("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", None),
            ("ai_toolbox_cockpit.app.available_update", None),
            ("ai_toolbox_cockpit.app.load_active_platform", "strix-halo"),
            ("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", []),
        ]:
            self.stack.enter_context(patch(target, return_value=value))

    async def test_compact_entry_save_reopen_and_gufo_subprocess_environment(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one(TabbedContent).active = "tab-models"
            await pilot.pause()
            button = app.query_one("#model-download-options")
            self.assertTrue(button.visible)
            self.assertLessEqual(button.region.right, app.size.width)
            self.assertEqual(len(app.query("#download-disable-xet")), 0)
            await pilot.click("#model-download-options")
            self.assertIsInstance(app.screen, DownloadOptionsModal)
            self.assertTrue(app.screen.query_one("#download-timeout-seconds", Input).disabled)
            await pilot.click("#download-disable-xet")
            await pilot.click("#download-custom-timeout")
            timeout = app.screen.query_one("#download-timeout-seconds", Input)
            self.assertFalse(timeout.disabled)
            self.assertEqual(timeout.value, "300")
            timeout.value = "600"
            await pilot.click("#download-options-save")
            await pilot.pause()
            self.assertEqual(get_download_options(), DownloadOptions(True, 600))

            await pilot.click("#model-download-options")
            self.assertTrue(app.screen.query_one("#download-disable-xet", Checkbox).value)
            self.assertEqual(app.screen.query_one("#download-timeout-seconds", Input).value, "600")
            await pilot.click("#download-options-cancel")
            await pilot.pause()

            panel = app.query_one("#model-panel-gufo")
            panel._pending_model = {}
            panel._pending_directory = Path("/tmp/cockpit-test-models")
            with (
                patch.object(app, "suspend", return_value=nullcontext()),
                patch("ai_toolbox_cockpit.backends.gufo.models.save_models_dir", return_value=True),
                patch("ai_toolbox_cockpit.backends.gufo.models.get_download_commands", return_value=[["hf", "download", "example/model"]]),
                patch("ai_toolbox_cockpit.backends.gufo.models.subprocess.run") as run,
                patch("ai_toolbox_cockpit.backends.gufo.models.missing_files", return_value=[]),
                patch.object(panel, "refresh_all_model_controls"),
            ):
                panel._download_confirmed(True)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_DISABLE_XET"], "1")
            self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_DOWNLOAD_TIMEOUT"], "600")

    async def test_invalid_timeout_save_failure_cancel_and_reset(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one(TabbedContent).active = "tab-models"
            await pilot.pause()
            await pilot.click("#model-download-options")
            await pilot.click("#download-custom-timeout")
            for value in ("", "0", "-1", "1.5", "abc"):
                app.screen.query_one("#download-timeout-seconds", Input).value = value
                await pilot.click("#download-options-save")
                # Textual ignores clicks while a button's pressed animation is active.
                await pilot.pause(0.35)
                self.assertIsInstance(app.screen, DownloadOptionsModal)
                self.assertIn("positive whole number", str(app.screen.query_one("#download-options-error", Static).render()))
                self.assertIsNone(get_setting("hf_download_options"))

            app.screen.query_one("#download-timeout-seconds", Input).value = "300"
            with patch("ai_toolbox_cockpit.views.download_options.save_download_options", return_value=False):
                await pilot.click("#download-options-save")
                self.assertIsInstance(app.screen, DownloadOptionsModal)
                self.assertIn("Could not save", str(app.screen.query_one("#download-options-error", Static).render()))
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsNone(get_setting("hf_download_options"))

            save_download_options(DownloadOptions(True, 300))
            await pilot.click("#model-download-options")
            await pilot.click("#download-disable-xet")
            await pilot.click("#download-custom-timeout")
            self.assertTrue(app.screen.query_one("#download-timeout-seconds", Input).disabled)
            await pilot.click("#download-options-save")
            await pilot.pause()
            self.assertEqual(get_download_options(), DownloadOptions())
            self.assertEqual(huggingface_environment()["HF_HUB_DISABLE_XET"], "0")
            self.assertNotIn("HF_HUB_DOWNLOAD_TIMEOUT", huggingface_environment())
