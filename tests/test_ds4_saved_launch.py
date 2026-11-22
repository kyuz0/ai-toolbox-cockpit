"""DwarfStar (ds4) Server Mode remembers launch settings per model and role."""

import tempfile
from contextlib import ExitStack, asynccontextmanager, nullcontext
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.ds4.launch_settings import (
    clear_launch_settings,
    describe_changes,
    load_launch_settings,
    save_launch_settings,
)
from ai_toolbox_cockpit.backends.ds4.server import Ds4ServerPanel
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings
from ai_toolbox_cockpit.widgets import SearchableSelect
from textual.widgets import Checkbox, Input, TabbedContent, TextArea

Q2_1017 = "DeepSeek-V4.1-Flash-1017-Q2.gguf"
VISION_1017 = "DeepSeek-V4.1-Flash-1017-vision-BF16.gguf"
Q2 = "DeepSeek-V4.1-Flash-Q2.gguf"
VISION = "DeepSeek-V4.1-Flash-Vision.gguf"
MODELS = [
    {"name": name, "path": "/models/" + name}
    for name in (Q2_1017, VISION_1017, Q2, VISION)
]


def path_of(name: str) -> str:
    return "/models/" + name


class LaunchSettingsStoreTests(TestCase):
    def test_round_trip_is_per_model_and_per_role(self) -> None:
        self.assertEqual(load_launch_settings(Q2, "Standalone"), {})
        self.assertTrue(save_launch_settings(Q2, "Standalone", {"context": "1000"}))
        self.assertTrue(save_launch_settings(Q2, "Worker", {"context": "2000"}))
        self.assertTrue(save_launch_settings(Q2_1017, "Standalone", {"context": "3000"}))
        self.assertEqual(load_launch_settings(Q2, "Standalone"), {"context": "1000"})
        self.assertEqual(load_launch_settings(Q2, "Worker"), {"context": "2000"})
        self.assertEqual(load_launch_settings(Q2_1017, "Standalone"), {"context": "3000"})
        self.assertEqual(load_launch_settings(Q2_1017, "Worker"), {})

    def test_saving_keeps_the_other_ds4_settings(self) -> None:
        save_backend_settings("ds4", {"models_dir": "/data/ds4", "default_toolboxes": {"strix-halo": "x"}})
        save_backend_settings("halogen", {"context": "262144"})
        self.assertTrue(save_launch_settings(Q2, "Standalone", {"context": "1000"}))
        ds4 = get_backend_settings("ds4")
        self.assertEqual(ds4["models_dir"], "/data/ds4")
        self.assertEqual(ds4["default_toolboxes"], {"strix-halo": "x"})
        self.assertEqual(get_backend_settings("halogen"), {"context": "262144"})

    def test_only_strings_and_booleans_are_stored(self) -> None:
        values = {"context": "1000", "ssd_enabled": True, "bad": 5, "worse": ["a"], "none": None}
        self.assertTrue(save_launch_settings(Q2, "Standalone", values))
        self.assertEqual(
            load_launch_settings(Q2, "Standalone"),
            {"context": "1000", "ssd_enabled": True},
        )

    def test_unknown_role_or_empty_model_is_refused(self) -> None:
        self.assertFalse(save_launch_settings(Q2, "Boss", {"context": "1"}))
        self.assertFalse(save_launch_settings("", "Standalone", {"context": "1"}))
        self.assertEqual(load_launch_settings("", "Standalone"), {})

    def test_damaged_launch_entry_is_ignored(self) -> None:
        save_backend_settings("ds4", {"launch": "not a dict"})
        self.assertEqual(load_launch_settings(Q2, "Standalone"), {})
        self.assertTrue(save_launch_settings(Q2, "Standalone", {"context": "1000"}))
        self.assertEqual(load_launch_settings(Q2, "Standalone"), {"context": "1000"})

    def test_clear_removes_only_that_model_and_role(self) -> None:
        save_launch_settings(Q2, "Standalone", {"context": "1"})
        save_launch_settings(Q2, "Worker", {"context": "2"})
        save_launch_settings(Q2_1017, "Standalone", {"context": "3"})
        self.assertTrue(clear_launch_settings(Q2, "Standalone"))
        self.assertEqual(load_launch_settings(Q2, "Standalone"), {})
        self.assertEqual(load_launch_settings(Q2, "Worker"), {"context": "2"})
        self.assertEqual(load_launch_settings(Q2_1017, "Standalone"), {"context": "3"})
        self.assertTrue(clear_launch_settings(Q2, "Worker"))
        self.assertNotIn(Q2, get_backend_settings("ds4")["launch"])
        self.assertTrue(clear_launch_settings("never-saved.gguf", "Standalone"))

    def test_describe_changes_lists_only_differences_in_form_order(self) -> None:
        current = {"context": "524288", "ssd_enabled": True, "host": "localhost", "extra_args": "", "kv_dir": "x" * 80}
        target = {"context": "262144", "ssd_enabled": False, "host": "localhost", "extra_args": "--a", "kv_dir": ""}
        self.assertEqual(
            describe_changes(current, target),
            [
                "Context length: 524288 -> 262144",
                "KV cache directory: " + "x" * 57 + "... -> (empty)",
                "SSD streaming: on -> off",
                "Extra args: (empty) -> --a",
            ],
        )
        self.assertEqual(describe_changes(current, current), [])


class SavedLaunchUiTests(IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def open_ds4(self, models=MODELS):
        with ExitStack() as stack:
            for target, value in (
                ("ai_toolbox_cockpit.app.load_active_platform", "strix-halo"),
                ("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", None),
                ("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", None),
                ("ai_toolbox_cockpit.app.available_update", None),
                ("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", []),
                ("ai_toolbox_cockpit.backends.ds4.server.scan_local_models", models),
                ("ai_toolbox_cockpit.backends.ds4.server.detect_container_engines", [ContainerEngine.PODMAN]),
            ):
                stack.enter_context(patch(target, return_value=value))
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(200, 70)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()
                yield app, pilot, app.query_one(Ds4ServerPanel)

    async def launch(self, app, panel) -> None:
        with (
            patch("ai_toolbox_cockpit.backends.ds4.server.build_server_cmd", return_value=["podman", "run"]),
            patch.object(app, "push_screen"),
        ):
            panel.start_pressed()
        with (
            patch.object(app, "suspend", return_value=nullcontext()),
            patch("ai_toolbox_cockpit.backends.ds4.server.run_foreground_server") as run,
        ):
            panel._start_confirmed(True)
        run.assert_called_once()

    async def select_model(self, app, pilot, name: str) -> None:
        app.query_one("#ds4-model", SearchableSelect).value = path_of(name)
        await pilot.pause()

    async def test_launch_settings_survive_a_restart(self) -> None:
        kv_dir = tempfile.mkdtemp()
        async with self.open_ds4() as (app, pilot, panel):
            default_model = app.query_one("#ds4-model", SearchableSelect).value
            launched = Q2 if default_model != path_of(Q2) else Q2_1017
            await self.select_model(app, pilot, launched)
            default_context = app.query_one("#ds4-context", Input).value
            app.query_one("#ds4-context", Input).value = "524288"
            app.query_one("#ds4-ssd-experts", Input).value = "80GB"
            app.query_one("#ds4-kv-enabled", Checkbox).value = True
            await pilot.pause()
            app.query_one("#ds4-kv-dir", Input).value = kv_dir
            app.query_one("#ds4-kv-mb", Input).value = "4096"
            app.query_one("#ds4-extra-args", TextArea).text = "--no-warmup"
            vision_name = VISION if launched == Q2 else VISION_1017
            app.query_one("#ds4-vision", SearchableSelect).value = path_of(vision_name)
            await pilot.pause()
            await self.launch(app, panel)
        self.assertNotEqual(default_context, "524288")

        async with self.open_ds4() as (app, pilot, panel):
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-model", SearchableSelect).value, path_of(launched))
            self.assertEqual(app.query_one("#ds4-context", Input).value, "524288")
            self.assertEqual(app.query_one("#ds4-ssd-experts", Input).value, "80GB")
            self.assertTrue(app.query_one("#ds4-kv-enabled", Checkbox).value)
            self.assertFalse(app.query_one("#ds4-kv-dir", Input).disabled)
            self.assertEqual(app.query_one("#ds4-kv-dir", Input).value, kv_dir)
            self.assertEqual(app.query_one("#ds4-kv-mb", Input).value, "4096")
            self.assertEqual(app.query_one("#ds4-extra-args", TextArea).text, "--no-warmup")
            self.assertEqual(app.query_one("#ds4-vision", SearchableSelect).value, path_of(vision_name))
            self.assertEqual(app.query_one("#ds4-engine", SearchableSelect).value, "podman")

    async def test_saved_values_do_not_leak_to_other_models_or_roles(self) -> None:
        async with self.open_ds4() as (app, pilot, panel):
            default_model = app.query_one("#ds4-model", SearchableSelect).value
            launched = Q2 if default_model != path_of(Q2) else Q2_1017
            other = Q2_1017 if launched == Q2 else Q2
            await self.select_model(app, pilot, other)
            other_context = app.query_one("#ds4-context", Input).value
            await self.select_model(app, pilot, launched)
            app.query_one("#ds4-context", Input).value = "524288"
            await self.launch(app, panel)

            await self.select_model(app, pilot, other)
            self.assertEqual(app.query_one("#ds4-context", Input).value, other_context)

            await self.select_model(app, pilot, launched)
            self.assertEqual(app.query_one("#ds4-context", Input).value, "524288")

            role = app.query_one("#ds4-role", SearchableSelect)
            role.value = "Coordinator"
            await pilot.pause()
            self.assertNotEqual(app.query_one("#ds4-context", Input).value, "524288")
            role.value = "Standalone"
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-context", Input).value, "524288")

    async def test_saved_encoder_that_is_no_longer_installed_is_ignored(self) -> None:
        async with self.open_ds4() as (app, pilot, panel):
            default_model = app.query_one("#ds4-model", SearchableSelect).value
            launched = Q2 if default_model != path_of(Q2) else Q2_1017
            await self.select_model(app, pilot, launched)
            vision_name = VISION if launched == Q2 else VISION_1017
            app.query_one("#ds4-vision", SearchableSelect).value = path_of(vision_name)
            await pilot.pause()
            await self.launch(app, panel)

        without_encoders = [model for model in MODELS if "ision" not in model["name"]]
        async with self.open_ds4(without_encoders) as (app, pilot, panel):
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-model", SearchableSelect).value, path_of(launched))
            self.assertEqual(app.query_one("#ds4-vision", SearchableSelect).value, "")

    async def test_a_failed_save_warns_but_still_starts(self) -> None:
        async with self.open_ds4() as (app, pilot, panel):
            with (
                patch("ai_toolbox_cockpit.backends.ds4.server.save_launch_settings", return_value=False),
                patch.object(panel, "notify") as notify,
            ):
                await self.launch(app, panel)
            notify.assert_called_once()
            self.assertEqual(notify.call_args.kwargs["severity"], "warning")

    async def test_cancelling_the_confirmation_saves_nothing(self) -> None:
        async with self.open_ds4() as (app, pilot, panel):
            app.query_one("#ds4-context", Input).value = "524288"
            with (
                patch("ai_toolbox_cockpit.backends.ds4.server.build_server_cmd", return_value=["podman", "run"]),
                patch.object(app, "push_screen"),
            ):
                panel.start_pressed()
            panel._start_confirmed(False)
        self.assertEqual(get_backend_settings("ds4").get("launch", {}), {})

    async def test_reset_shows_the_difference_and_changes_nothing_until_confirmed(self) -> None:
        async with self.open_ds4() as (app, pilot, panel):
            default_context = app.query_one("#ds4-context", Input).value
            self.assertTrue(default_context.isdigit())
            with patch.object(app, "push_screen") as push, patch.object(panel, "notify") as notify:
                panel.reset_pressed()
            push.assert_not_called()
            self.assertIn("already matches", notify.call_args.args[0])

            app.query_one("#ds4-context", Input).value = "524288"
            app.query_one("#ds4-kv-enabled", Checkbox).value = True
            await pilot.pause()
            app.query_one("#ds4-extra-args", TextArea).text = "--no-warmup"
            with patch.object(app, "push_screen") as push:
                panel.reset_pressed()
            message = push.call_args.args[0].message
            self.assertIn(f"Context length: 524288 -> {default_context}", message)
            self.assertIn("Disk KV cache: on -> off", message)
            self.assertIn("Extra args: --no-warmup -> (empty)", message)
            self.assertNotIn("Host:", message)
            self.assertNotIn("saved settings", message)
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-context", Input).value, "524288")
            self.assertTrue(app.query_one("#ds4-kv-enabled", Checkbox).value)
            self.assertFalse(app.query_one("#ds4-kv-dir", Input).disabled)
            self.assertEqual(app.query_one("#ds4-extra-args", TextArea).text, "--no-warmup")

            panel._reset_confirmed(False)
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-context", Input).value, "524288")

    async def test_confirmed_reset_restores_defaults_and_forgets_the_saved_values(self) -> None:
        async with self.open_ds4() as (app, pilot, panel):
            model = app.query_one("#ds4-model", SearchableSelect).value
            default_context = app.query_one("#ds4-context", Input).value
            default_experts = app.query_one("#ds4-ssd-experts", Input).value
            app.query_one("#ds4-context", Input).value = "524288"
            app.query_one("#ds4-ssd-experts", Input).value = "80GB"
            app.query_one("#ds4-extra-args", TextArea).text = "--no-warmup"
            await pilot.pause()
            await self.launch(app, panel)
            self.assertEqual(load_launch_settings(model.rsplit("/", 1)[1], "Standalone")["context"], "524288")

            with patch.object(app, "push_screen") as push:
                panel.reset_pressed()
            message = push.call_args.args[0].message
            self.assertIn("The saved settings for this model and role are removed.", message)
            panel._reset_confirmed(True)
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-context", Input).value, default_context)
            self.assertEqual(app.query_one("#ds4-ssd-experts", Input).value, default_experts)
            self.assertEqual(app.query_one("#ds4-extra-args", TextArea).text, "")
            self.assertEqual(load_launch_settings(model.rsplit("/", 1)[1], "Standalone"), {})

        async with self.open_ds4() as (app, pilot, panel):
            await pilot.pause()
            self.assertEqual(app.query_one("#ds4-context", Input).value, default_context)
