from unittest.mock import patch
from unittest import IsolatedAsyncioTestCase

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect
from textual.widgets import Button, Static, Tab


class AppMountTests(IsolatedAsyncioTestCase):
    async def test_app_mounts_without_running_container_commands(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(80, 24)) as pilot:
                self.assertIsNotNone(app.query_one("#toolboxes-view"))
                self.assertIsNotNone(app.query_one("#server-panel-llama_cpp"))
                self.assertIsNotNone(app.query_one("#model-panel-comfyui"))
                support = str(app.query_one("#server-platform-support", Static).content)
                self.assertIn("llama.cpp: supported", support)
                self.assertIn("vLLM: supported", support)
                self.assertEqual((app.size.width, app.size.height), (80, 24))
                self.assertGreaterEqual(app.query_one("#title-banner", Static).region.height, 5)
                self.assertIn("Server Mode", {tab.label_text for tab in app.query(Tab)})
                self.assertEqual(app.query_one("#platform-select", SearchableSelect).region.height, 1)
                self.assertEqual(app.query_one("#toolbox-backend-filter", SearchableSelect).region.height, 1)
                self.assertEqual(app.query_one("#toolbox-channel-filter", SearchableSelect).region.height, 1)
                self.assertEqual(app.query_one("#toolbox-refresh", Button).region.height, 1)
                await pilot.press("tab")
                self.assertIsNotNone(app.focused)
