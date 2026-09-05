import tempfile
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch
from unittest import IsolatedAsyncioTestCase

from rich.text import Text

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.runtime.interactive import InteractiveBackend, InteractiveRuntime
from ai_toolbox_cockpit.runtime.toolboxes import InstalledToolbox
from ai_toolbox_cockpit.storage import DiskSpace
from ai_toolbox_cockpit.updates import RELAUNCH_AFTER_UPDATE
from ai_toolbox_cockpit.views.toolboxes import ToolboxesView
from ai_toolbox_cockpit.widgets import CockpitCheckbox, SearchableSelect
from textual.containers import Vertical
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Static, Tab, TabbedContent, TextArea


class AppMountTests(IsolatedAsyncioTestCase):
    async def test_application_update_strip_runs_pipx_upgrade(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.ds4.server.scan_local_models", return_value=[]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app._show_application_update("2026.8.20.1500")
                await pilot.pause()

                update_row = app.query_one("#application-update-row")
                update_message = app.query_one("#application-update-message", Label)
                update_button = app.query_one("#application-update-run", Button)
                self.assertEqual(update_row.styles.display, "block")
                self.assertIn("pipx upgrade ai-toolbox-cockpit", str(update_message.render()))
                self.assertFalse(update_button.disabled)

                await pilot.click("#application-update-run")
                await pilot.pause()
                confirm_message = app.screen.query_one("#confirm_message", Label)
                self.assertIn(
                    "pipx upgrade ai-toolbox-cockpit",
                    str(confirm_message.render()),
                )
                await pilot.click("#btn_no")
                await pilot.pause()

                with (
                    patch.object(app, "suspend", return_value=nullcontext()),
                    patch("ai_toolbox_cockpit.app.subprocess.run") as run,
                    patch.object(app, "exit") as exit_app,
                ):
                    app._application_update_confirmed(True)

                run.assert_called_once_with(
                    ["pipx", "upgrade", "ai-toolbox-cockpit"],
                    check=True,
                )
                self.assertTrue(update_button.disabled)
                self.assertEqual(str(update_button.label), "Updated")
                self.assertIn("Relaunching", str(update_message.render()))
                exit_app.assert_called_once_with(result=RELAUNCH_AFTER_UPDATE)

    async def test_app_mounts_without_running_container_commands(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(80, 24)) as pilot:
                self.assertIsNotNone(app.query_one("#toolboxes-view"))
                self.assertIsNotNone(app.query_one("#server-panel-llama_cpp"))
                self.assertIsNotNone(app.query_one("#model-panel-comfyui"))
                self.assertEqual(len(app.query("#server-platform-support")), 0)
                self.assertEqual((app.size.width, app.size.height), (80, 24))
                self.assertGreaterEqual(app.query_one("#title-banner", Static).region.height, 5)
                tab_labels = {tab.label_text for tab in app.query(Tab)}
                self.assertEqual(tab_labels, {"Toolboxes", "Server Mode", "Models"})
                self.assertEqual(app.query_one("#platform-select", SearchableSelect).region.height, 1)
                self.assertEqual(app.query_one("#toolbox-backend-filter", SearchableSelect).region.height, 1)
                self.assertEqual(app.query_one("#toolbox-channel-filter", SearchableSelect).region.height, 1)
                self.assertEqual(app.query_one("#toolbox-refresh", Button).region.height, 1)
                self.assertEqual(
                    app.query_one("#toolbox-channel-filter", SearchableSelect).value,
                    "stable",
                )
                field_labels = {
                    label.render().plain
                    for label in app.query(".toolbox-filters .field-label")
                }
                self.assertEqual(field_labels, {"Backend", "Channel"})
                self.assertLess(
                    app.query_one("#toolbox-create-update", Button).region.y,
                    app.query_one("#toolbox-catalog-table", DataTable).region.y,
                )
                self.assertEqual(len(app.query("#toolbox-check-updates")), 0)
                await pilot.press("tab")
                self.assertIsNotNone(app.focused)

    async def test_extra_args_wrap_and_expand_to_show_the_full_value(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(100, 30)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                for control_id in (
                    "#llama-extra-args",
                    "#ds4-extra-args",
                    "#vllm-extra-args",
                    "#comfy-extra-args",
                ):
                    self.assertTrue(app.query_one(control_id, TextArea).soft_wrap)

                extra_args = app.query_one("#llama-extra-args", TextArea)
                long_value = " ".join(f"--argument-{index} value-{index}" for index in range(20))
                extra_args.text = long_value
                await pilot.pause()

                self.assertEqual(extra_args.text, long_value)
                self.assertGreater(extra_args.region.height, 1)
                self.assertEqual(extra_args.parent.region.height, extra_args.region.height)

    async def test_backend_selectors_only_offer_backends_with_platform_toolboxes(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.app.load_active_platform", return_value="strix-halo"),
            patch("ai_toolbox_cockpit.app.save_active_platform"),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                toolbox_backend = app.query_one(
                    "#toolbox-backend-filter", SearchableSelect
                )
                server_backend = app.query_one(
                    "#server-backend-select", SearchableSelect
                )
                model_backend = app.query_one(
                    "#model-backend-select", SearchableSelect
                )

                def option_values(select: SearchableSelect) -> set[str]:
                    return {value for _, value in select._options}

                strix_backends = {"llama_cpp", "ds4", "vllm", "comfyui", "halogen"}
                self.assertEqual(
                    option_values(toolbox_backend), {"all", *strix_backends}
                )
                self.assertEqual(option_values(server_backend), strix_backends)
                self.assertEqual(option_values(model_backend), strix_backends)

                toolbox_backend.value = "vllm"
                server_backend.value = "vllm"
                model_backend.value = "vllm"
                await pilot.pause()

                platform = app.query_one("#platform-select", SearchableSelect)
                platform.value = "r9700"
                await pilot.pause()

                self.assertEqual(
                    option_values(toolbox_backend), {"all", "llama_cpp"}
                )
                self.assertEqual(option_values(server_backend), {"llama_cpp"})
                self.assertEqual(option_values(model_backend), {"llama_cpp"})

                self.assertEqual(toolbox_backend.value, "all")
                self.assertEqual(server_backend.value, "llama_cpp")
                self.assertEqual(model_backend.value, "llama_cpp")
                self.assertEqual(
                    app.query_one("#server-content-switcher").current,
                    "server-panel-llama_cpp",
                )
                self.assertEqual(
                    app.query_one("#model-content-switcher").current,
                    "model-panel-llama_cpp",
                )

    async def test_toolbox_checkbox_remains_visible_when_selected(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                table = app.query_one("#toolbox-catalog-table", DataTable)
                row = table.get_row_index("strix-halo-llama-rocm-10-0")

                unchecked = table.get_cell_at((row, 0))
                self.assertIsInstance(unchecked, Text)
                self.assertEqual(unchecked.plain, "[ ]")

                table.focus()
                table.move_cursor(row=row, column=0, animate=False)
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()

                checked = table.get_cell_at((row, 0))
                self.assertIsInstance(checked, Text)
                self.assertEqual(checked.plain, "[x]")
                toolboxes_view = app.query_one("#toolboxes-view")
                self.assertIn("strix-halo-llama-rocm-10-0", toolboxes_view.selected_toolboxes)

    async def test_toolbox_checkbox_toggles_with_one_mouse_click(self) -> None:
        toolbox_id = "strix-halo-llama-vulkan-radv"
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                table = app.query_one("#toolbox-catalog-table", DataTable)
                row = table.get_row_index(toolbox_id)
                click_offset = (2, row + int(table.show_header))

                await pilot.click(table, offset=click_offset)
                await pilot.pause()
                self.assertIn(
                    toolbox_id,
                    app.query_one("#toolboxes-view").selected_toolboxes,
                )

                await pilot.click(table, offset=click_offset)
                await pilot.pause()
                self.assertNotIn(
                    toolbox_id,
                    app.query_one("#toolboxes-view").selected_toolboxes,
                )

    async def test_installed_r9700_update_uses_persisted_toolbx_runtime(self) -> None:
        toolbox_id = "r9700-llama-vulkan-radv"
        container_name = "r9700-llama-vulkan-radv"
        persisted_runtime = InteractiveRuntime(
            InteractiveBackend.TOOLBOX,
            ContainerEngine.PODMAN,
        )
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                view = app.query_one("#toolboxes-view", ToolboxesView)
                view.selected_toolboxes = {toolbox_id}
                view.installed = {
                    container_name: InstalledToolbox(
                        name=container_name,
                        image="docker.io/kyuz0/amd-r9700-toolboxes:vulkan-radv",
                        status="Created",
                        created="2026-05-20",
                        engine=ContainerEngine.PODMAN,
                        runtime=persisted_runtime,
                    )
                }
                view.remote_dates = {toolbox_id: "2026-08-11"}

                with (
                    patch(
                        "ai_toolbox_cockpit.views.toolboxes.detect_interactive_backend",
                        return_value=None,
                    ),
                    patch.object(app, "push_screen") as push_screen,
                ):
                    view._prepare_create_update(view.selected())

                push_screen.assert_called_once()
                confirmation = push_screen.call_args.args[0]
                self.assertIn(
                    "toolbox rm -f r9700-llama-vulkan-radv",
                    confirmation.message,
                )
                self.assertIn(
                    "podman pull docker.io/kyuz0/amd-r9700-toolboxes:vulkan-radv",
                    confirmation.message,
                )

    async def test_create_update_checks_installed_toolbox_before_offering_update(self) -> None:
        toolbox_id = "r9700-llama-vulkan-radv"
        container_name = "r9700-llama-vulkan-radv"
        persisted_runtime = InteractiveRuntime(
            InteractiveBackend.TOOLBOX,
            ContainerEngine.PODMAN,
        )
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                view = app.query_one("#toolboxes-view", ToolboxesView)
                view.selected_toolboxes = {toolbox_id}
                view.installed = {
                    container_name: InstalledToolbox(
                        name=container_name,
                        image="docker.io/kyuz0/amd-r9700-toolboxes:vulkan-radv",
                        status="Created",
                        created="2026-05-20",
                        engine=ContainerEngine.PODMAN,
                        runtime=persisted_runtime,
                    )
                }

                with patch.object(view, "check_updates_for_create") as check_updates:
                    view.create_update_pressed()

                check_updates.assert_called_once()
                checked = check_updates.call_args.args[0]
                self.assertEqual(tuple(toolbox.id for toolbox in checked), (toolbox_id,))

    async def test_failed_toolbox_create_restores_cockpit_application_mode(self) -> None:
        runtime = InteractiveRuntime(
            InteractiveBackend.TOOLBOX,
            ContainerEngine.PODMAN,
        )
        resumed = False

        @contextmanager
        def recording_suspend():
            nonlocal resumed
            yield
            resumed = True

        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                view = app.query_one("#toolboxes-view", ToolboxesView)
                toolbox = app.toolbox_catalog.toolboxes["strix-halo-llama-rocm-10-0"]
                view._pending_create = (toolbox,)
                view.selected_toolboxes = {toolbox.id}

                with (
                    patch(
                        "ai_toolbox_cockpit.views.toolboxes.detect_interactive_backend",
                        return_value=runtime,
                    ),
                    patch(
                        "ai_toolbox_cockpit.views.toolboxes.create_toolbox",
                        side_effect=subprocess.CalledProcessError(1, ["toolbox", "create"]),
                    ),
                    patch.object(app, "suspend", side_effect=recording_suspend),
                    patch.object(view, "refresh_installed", return_value=None),
                    patch.object(view, "notify") as notify,
                ):
                    view._create_update_confirmed(True)

                self.assertTrue(resumed)
                self.assertEqual(view.selected_toolboxes, set())
                self.assertIn("Toolbox operation failed", notify.call_args.args[0])

    async def test_llama_download_refreshes_server_model_dropdown(self) -> None:
        inventory: list[dict[str, str]] = []

        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch(
                "ai_toolbox_cockpit.backends.llama_cpp.models.scan_local_models",
                side_effect=lambda: list(inventory),
            ),
            patch(
                "ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models",
                side_effect=lambda: list(inventory),
            ),
            patch("ai_toolbox_cockpit.backends.ds4.models.scan_local_models", return_value=[]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                panel = app.query_one("#model-panel-llama_cpp")
                panel._download_repo = "example/model"
                inventory.append({"name": "new-model.gguf", "path": "/models/new-model.gguf"})

                with (
                    patch.object(app, "suspend", return_value=nullcontext()),
                    patch(
                        "ai_toolbox_cockpit.backends.llama_cpp.models.get_download_cmd",
                        return_value=["hf", "download"],
                    ),
                    patch("ai_toolbox_cockpit.backends.llama_cpp.models.subprocess.run"),
                ):
                    panel._download_quant("Q4_K_M")

                self.assertEqual(
                    app.query_one("#llama-model", SearchableSelect).value,
                    "/models/new-model.gguf",
                )

    async def test_llama_download_prompts_for_and_remembers_hf_token(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.models.get_hf_token", return_value=""),
            patch("ai_toolbox_cockpit.backends.ds4.models.scan_local_models", return_value=[]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.models.scan_local_models", return_value=[]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                panel = app.query_one("#model-panel-llama_cpp")
                repo = panel.catalog.entries[0]["repo"]
                app.query_one("#llama-download-repo", SearchableSelect).value = repo

                with (
                    patch.object(panel, "load_quants") as load_quants,
                    patch(
                        "ai_toolbox_cockpit.backends.llama_cpp.models.save_hf_token",
                        return_value=True,
                    ) as save_token,
                ):
                    panel.download_pressed()
                    await pilot.pause()
                    app.screen.query_one("#hf-token-input", Input).value = "hf_example"
                    app.screen.query_one("#hf-token-remember", Checkbox).value = True
                    await pilot.click("#hf-token-continue")
                    await pilot.pause()

                save_token.assert_called_once_with("hf_example")
                load_quants.assert_called_once_with(repo, "hf_example")

    async def test_ds4_download_refreshes_server_model_dropdown(self) -> None:
        inventory: list[dict[str, str]] = []

        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[]),
            patch(
                "ai_toolbox_cockpit.backends.ds4.models.scan_local_models",
                side_effect=lambda: list(inventory),
            ),
            patch(
                "ai_toolbox_cockpit.backends.ds4.server.scan_local_models",
                side_effect=lambda: list(inventory),
            ),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                panel = app.query_one("#model-panel-ds4")
                self.assertEqual(
                    app.query_one("#ds4-download-model", SearchableSelect).value,
                    "antirez/deepseek-v4-gguf::DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf",
                )
                panel._pending_repo = "example/model"
                panel._pending_filename = "new-ds4-model.gguf"
                inventory.append(
                    {
                        "name": "new-ds4-model.gguf",
                        "path": "/models/new-ds4-model.gguf",
                    }
                )

                with (
                    patch.object(app, "suspend", return_value=nullcontext()),
                    patch(
                        "ai_toolbox_cockpit.backends.ds4.models.get_download_cmd",
                        return_value=["hf", "download"],
                    ),
                    patch("ai_toolbox_cockpit.backends.ds4.models.subprocess.run"),
                ):
                    panel._download_model()

                self.assertEqual(
                    app.query_one("#ds4-model", SearchableSelect).value,
                    "/models/new-ds4-model.gguf",
                )

    async def test_ds4_download_warns_when_artifact_will_not_fit(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[]),
            patch("ai_toolbox_cockpit.backends.ds4.models.scan_local_models", return_value=[]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                panel = app.query_one("#model-panel-ds4")
                selected = app.query_one(
                    "#ds4-download-model", SearchableSelect
                ).value
                panel._pending_repo, panel._pending_filename = selected.split("::", 1)

                with (
                    patch(
                        "ai_toolbox_cockpit.backends.ds4.models.disk_space_for_path",
                        return_value=DiskSpace(
                            total=100_000_000_000,
                            used=99_000_000_000,
                            free=1_000_000_000,
                        ),
                    ),
                    patch(
                        "ai_toolbox_cockpit.backends.ds4.models.is_model_downloaded",
                        return_value=False,
                    ),
                ):
                    panel._confirm_download()
                    await pilot.pause()

                message = str(
                    app.screen.query_one("#confirm_message", Label).render()
                )
                self.assertIn("Estimated download size:", message)
                self.assertIn("WARNING", message)
                self.assertIn("unlikely to fit", message)
                await pilot.click("#btn_no")

    async def test_model_tables_are_backend_owned_and_local_inventory_rescans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            with (
                patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
                patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
                patch("ai_toolbox_cockpit.app.available_update", return_value=None),
                patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=models_dir),
                patch("ai_toolbox_cockpit.backends.llama_cpp.models.get_models_dir", return_value=models_dir),
                patch("ai_toolbox_cockpit.backends.ds4.models.scan_local_models", return_value=[]),
            ):
                app = AiToolboxCockpitApp()
                async with app.run_test(size=(180, 45)) as pilot:
                    llama_table = app.query_one("#llama-local-models", DataTable)
                    ds4_table = app.query_one("#ds4-local-models", DataTable)
                    vllm_table = app.query_one("#vllm-curated-models", DataTable)
                    comfy_table = app.query_one("#comfy-bundles", DataTable)

                    self.assertEqual(len(llama_table.columns), 2)
                    self.assertEqual(len(ds4_table.columns), 2)
                    self.assertEqual(len(vllm_table.columns), 6)
                    self.assertEqual(len(comfy_table.columns), 4)
                    self.assertIn(
                        "Available space:",
                        str(app.query_one("#llama-disk-space", Static).render()),
                    )
                    self.assertIn(
                        "Available space:",
                        str(app.query_one("#ds4-disk-space", Static).render()),
                    )
                    self.assertEqual(llama_table.row_count, 0)
                    self.assertEqual(
                        vllm_table.row_count,
                        len(app.model_catalog.backends["vllm"].entries),
                    )
                    self.assertEqual(
                        comfy_table.row_count,
                        len(app.model_catalog.backends["comfyui"].entries),
                    )

                    first_model = models_dir / "first.gguf"
                    first_model.touch()
                    app.query_one(TabbedContent).active = "tab-models"
                    await pilot.pause()
                    self.assertEqual(llama_table.row_count, 1)
                    self.assertEqual(llama_table.get_cell_at((0, 1)), str(first_model))

                    second_model = models_dir / "second.gguf"
                    second_model.touch()
                    backend = app.query_one("#model-backend-select", SearchableSelect)
                    backend.value = "vllm"
                    await pilot.pause()
                    backend.value = "llama_cpp"
                    await pilot.pause()

                    self.assertEqual(llama_table.row_count, 2)
                    for row in range(llama_table.row_count):
                        self.assertTrue(Path(str(llama_table.get_cell_at((row, 1)))).is_file())

    async def test_models_backend_selector_and_intro_are_compact(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-models"
                await pilot.pause()

                label = app.query_one("#model-backend-select-label", Label)
                backend = app.query_one("#model-backend-select", SearchableSelect)
                intro = app.query_one(".model-view-copy", Static)
                panel = app.query_one("#model-panel-llama_cpp")

                self.assertEqual(str(label.render()), "Backend")
                self.assertLessEqual(backend.region.width, 40)
                self.assertLess(backend.region.width, app.size.width // 2)
                self.assertEqual(intro.region.height, 1)
                self.assertEqual(len(panel.query(".panel-title")), 0)

    async def test_server_backend_selector_and_profile_spacing_are_compact(self) -> None:
        local_model = {"name": "profile-test.gguf", "path": "/tmp/profile-test.gguf"}
        profiles = {
            "Thinking": {
                "args": "--reasoning on",
                "description": "Profile note",
            }
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.get_inference_profiles", return_value=profiles),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                label = app.query_one("#server-backend-select-label", Label)
                backend = app.query_one("#server-backend-select", SearchableSelect)
                model_row = app.query_one("#llama-model").parent
                profile_zone = app.query_one("#llama-profile-zone")

                self.assertEqual(str(label.render()), "Inference engine")
                self.assertLessEqual(backend.region.width, 40)
                self.assertLess(backend.region.width, app.size.width // 2)
                self.assertGreaterEqual(
                    profile_zone.region.y - model_row.region.bottom,
                    1,
                )

    async def test_deepseek_server_defaults_to_high_reasoning_effort(self) -> None:
        local_model = {
            "name": "DeepSeek-V4-Flash-0731-UD-IQ2_XXS.gguf",
            "path": (
                "/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_XXS/"
                "DeepSeek-V4-Flash-0731-UD-IQ2_XXS.gguf"
            ),
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                profile = app.query_one("#llama-profile", SearchableSelect)
                extra_args = app.query_one("#llama-extra-args", TextArea).text
                self.assertEqual(profile.value, "Thinking (Effort: High)")
                self.assertIn('"reasoning_effort":"high"', extra_args)
                self.assertNotIn('"reasoning_effort":"max"', extra_args)

    async def test_qwen_3_8_server_profiles_set_reasoning_effort(self) -> None:
        local_model = {
            "name": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
            "path": "/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                profile = app.query_one("#llama-profile", SearchableSelect)
                extra_args = app.query_one("#llama-extra-args", TextArea)
                self.assertEqual(profile.value, "Thinking (Effort: XHigh)")
                self.assertIn('"reasoning_effort":"xhigh"', extra_args.text)
                self.assertTrue(app.query_one("#llama-mtp-enabled", Checkbox).value)
                self.assertIn("--spec-type draft-mtp --spec-draft-n-max 2 -np 1", extra_args.text)
                self.assertEqual(app.query_one("#llama-batch", Input).value, "")
                self.assertEqual(app.query_one("#llama-ubatch", Input).value, "")

                app.query_one("#llama-mtp-enabled", Checkbox).value = False
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-batch", Input).value, "2048")
                self.assertEqual(app.query_one("#llama-ubatch", Input).value, "256")

                image = app.query_one("#llama-image", SearchableSelect)
                image.value = "strix-halo-llama-vulkan-radv"
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-ubatch", Input).value, "256")

                app.query_one("#llama-mtp-enabled", Checkbox).value = True
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-batch", Input).value, "")
                self.assertEqual(app.query_one("#llama-ubatch", Input).value, "")

                profile.value = "Thinking (Effort: Low)"
                await pilot.pause()
                self.assertIn('"reasoning_effort":"low"', extra_args.text)

                profile.value = "Instruct (No Reasoning)"
                await pilot.pause()
                self.assertIn('"enable_thinking":false', extra_args.text)
                self.assertIn("--reasoning off", extra_args.text)

    async def test_qwen_3_8_flash_next_unsets_gpu_layers_only_on_r9700(self) -> None:
        local_model = {
            "name": "Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf",
            "path": (
                "/models/Qwen3.8-Flash-Next-GGUF/"
                "Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf"
            ),
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.app.load_active_platform", return_value="strix-halo"),
            patch("ai_toolbox_cockpit.app.save_active_platform"),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                gpu_layers = app.query_one("#llama-ngl", Input)
                self.assertEqual(gpu_layers.value, "999")

                platform = app.query_one("#platform-select", SearchableSelect)
                platform.value = "r9700"
                await pilot.pause()
                self.assertEqual(gpu_layers.value, "")

                image = app.query_one("#llama-image", SearchableSelect)
                image.value = "r9700-llama-vulkan-radv"
                await pilot.pause()
                self.assertEqual(gpu_layers.value, "")

                platform.value = "strix-halo"
                await pilot.pause()
                self.assertEqual(gpu_layers.value, "999")

    async def test_rocmfp4_qwen_3_8_uses_external_mtp_flags_only(self) -> None:
        local_model = {
            "name": "Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf",
            "path": (
                "/models/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF/"
                "Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf"
            ),
        }
        mtp_model = Path("/models/qwen38/mtp-Qwen3.8-27B-Q8_0.gguf")
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.get_local_mtp_models", return_value=[mtp_model]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                self.assertEqual(app.query_one("#llama-mtp-draft", Input).value, "4")
                self.assertEqual(app.query_one("#llama-mtp-np", Input).value, "1")
                self.assertEqual(
                    app.query_one("#llama-mtp-model", SearchableSelect).value,
                    str(mtp_model),
                )
                mtp_label = app.query_one("#llama-mtp-model-label", Label)
                self.assertEqual(str(mtp_label.render()), "MTP model")
                extra_args = app.query_one("#llama-extra-args", TextArea).text
                for expected in (
                    "--spec-type draft-mtp",
                    "--spec-draft-ngl 99",
                    "--spec-draft-device ROCm0",
                    "--spec-draft-n-max 4",
                    "--spec-draft-n-min 0",
                    "--spec-draft-p-min 0.0",
                    "-fit off",
                    "--parallel 1",
                    "-dev ROCm0",
                ):
                    self.assertIn(expected, extra_args)
                self.assertNotIn(" -np ", extra_args)

    async def test_deepseek_server_enables_official_dspark_drafter(self) -> None:
        local_model = {
            "name": "DeepSeek-V4-Flash-0731-UD-IQ3_XXS.gguf",
            "path": (
                "/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS/"
                "DeepSeek-V4-Flash-0731-UD-IQ3_XXS.gguf"
            ),
        }
        drafter = Path(
            "/models/DeepSeek-V4-Flash-0731-GGUF/"
            "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
        )
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.get_local_dspark_models", return_value=[drafter]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                self.assertTrue(app.query_one("#llama-dspark-enabled", Checkbox).value)
                self.assertEqual(
                    app.query_one("#llama-dspark-model", SearchableSelect).value,
                    str(drafter),
                )
                extra_args = app.query_one("#llama-extra-args", TextArea).text
                self.assertIn("--spec-type draft-dspark", extra_args)
                self.assertIn("--spec-draft-n-max 3", extra_args)
                self.assertIn("--fit off -ngld 99", extra_args)

                image = app.query_one("#llama-image", SearchableSelect)
                image.value = "strix-halo-llama-vulkan-radv-performance"
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-batch", Input).value, "2048")
                self.assertEqual(app.query_one("#llama-ubatch", Input).value, "2048")
                self.assertEqual(app.query_one("#llama-parallel", Input).value, "1")
                self.assertTrue(app.query_one("#llama-kv-enabled", Checkbox).value)
                self.assertEqual(app.query_one("#llama-kv-type", SearchableSelect).value, "q8_0")

                image.value = "strix-halo-llama-rocm-10-0"
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-batch", Input).value, "2048")
                self.assertEqual(app.query_one("#llama-ubatch", Input).value, "2048")
                self.assertEqual(app.query_one("#llama-parallel", Input).value, "")
                self.assertFalse(app.query_one("#llama-kv-enabled", Checkbox).value)

                expected_labels = {
                    "llama-mtp-draft": ("llama-mtp-draft-label", "Draft tokens"),
                    "llama-mtp-np": ("llama-mtp-np-label", "Parallel sequences"),
                    "llama-dspark-model": ("llama-dspark-model-label", "Drafter model"),
                    "llama-dspark-draft": ("llama-dspark-draft-label", "Draft tokens"),
                    "llama-dspark-ngl": ("llama-dspark-ngl-label", "Draft GPU layers"),
                    "llama-projector": ("llama-projector-label", "Projector"),
                    "llama-profile": ("llama-profile-label", "Profile"),
                    "llama-batch": ("llama-batch-label", "Batch size"),
                    "llama-ubatch": ("llama-ubatch-label", "Ubatch size"),
                    "llama-parallel": ("llama-parallel-label", "Parallel sequences"),
                }
                for control_id, (label_id, expected_text) in expected_labels.items():
                    control = app.query_one(f"#{control_id}")
                    label = app.query_one(f"#{label_id}", Label)
                    self.assertEqual(str(label.render()), expected_text)
                    self.assertIs(label.parent, control.parent)

    async def test_every_input_and_searchable_select_has_its_own_persistent_label(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                unlabeled = []
                for control in (*app.query(SearchableSelect), *app.query(Input)):
                    if not control.id:
                        continue
                    labels = list(app.query(f"#{control.id}-label"))
                    if (
                        len(labels) != 1
                        or not isinstance(labels[0], Label)
                        or labels[0].parent is not control.parent
                        or not labels[0].render().plain.strip()
                    ):
                        unlabeled.append(control.id)

                self.assertEqual(unlabeled, [])

    async def test_every_boolean_control_uses_cockpit_checkbox(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)):
                checkboxes = list(app.query(Checkbox))

                self.assertGreater(len(checkboxes), 0)
                self.assertTrue(
                    all(isinstance(control, CockpitCheckbox) for control in checkboxes)
                )
                self.assertEqual(len(app.query("Switch")), 0)

    async def test_ds4_fields_have_visible_labels_and_horizontal_gutters(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(200, 60)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()

                for control_id in (
                    "ds4-context", "ds4-prefill", "ds4-host", "ds4-port",
                    "ds4-kv-dir", "ds4-kv-mb", "ds4-ssd-experts", "ds4-ssd-layers",
                    "ds4-role", "ds4-layers", "ds4-peer",
                    "ds4-dist-prefill", "ds4-dist-window",
                ):
                    control = app.query_one(f"#{control_id}")
                    label = app.query_one(f"#{control_id}-label", Label)
                    self.assertIs(label.parent, control.parent)
                    self.assertEqual(label.region.x, control.region.x)
                    self.assertLess(label.region.y, control.region.y)

                row = app.query_one("#ds4-context").parent.parent
                fields = list(row.query(".compact-field"))
                for left, right in zip(fields, fields[1:]):
                    self.assertGreaterEqual(right.region.x - left.region.right, 1)

                tile4 = app.query_one("#ds4-mxfp4-tile4-enabled", Checkbox)
                rgroup = app.query_one("#ds4-mxfp4-down-rgroup-enabled", Checkbox)
                self.assertEqual(
                    app.query_one("#ds4-mxfp4-zone", Vertical).styles.display,
                    "none",
                )
                self.assertFalse(tile4.value)
                self.assertFalse(rgroup.value)
                self.assertIn("DS4_ROCM_ENABLE_MXFP4_TILE4=1", str(tile4.label))
                self.assertIn("DS4_ROCM_MXFP4_DOWN_RGROUP=4", str(rgroup.label))

    async def test_ds4_mxfp4_controls_follow_selected_model_filename(self) -> None:
        plain = {
            "name": "DeepSeek-V4-Flash-Q4_K.gguf",
            "path": "/models/DeepSeek-V4-Flash-Q4_K.gguf",
        }
        mxfp4 = {
            "name": "DeepSeek-V4-Flash-MXFP4-Experts.gguf",
            "path": "/models/DeepSeek-V4-Flash-MXFP4-Experts.gguf",
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[]),
            patch(
                "ai_toolbox_cockpit.backends.ds4.server.scan_local_models",
                return_value=[plain, mxfp4],
            ),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 55)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()

                model = app.query_one("#ds4-model", SearchableSelect)
                zone = app.query_one("#ds4-mxfp4-zone", Vertical)
                tile4 = app.query_one(
                    "#ds4-mxfp4-tile4-enabled", CockpitCheckbox
                )
                rgroup = app.query_one(
                    "#ds4-mxfp4-down-rgroup-enabled", CockpitCheckbox
                )

                self.assertEqual(model.value, plain["path"])
                self.assertEqual(zone.styles.display, "none")
                self.assertFalse(tile4.value)
                self.assertFalse(rgroup.value)

                model.value = mxfp4["path"]
                await pilot.pause()
                self.assertEqual(zone.styles.display, "block")
                self.assertTrue(tile4.value)
                self.assertTrue(rgroup.value)

                model.value = plain["path"]
                await pilot.pause()
                self.assertEqual(zone.styles.display, "none")
                self.assertFalse(tile4.value)
                self.assertFalse(rgroup.value)

    async def test_ds4_ssd_checkboxes_show_state_and_toggle_with_one_click(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(160, 50)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()

                ssd = app.query_one("#ds4-ssd-enabled", CockpitCheckbox)
                cold = app.query_one("#ds4-ssd-cold", CockpitCheckbox)
                ssd.scroll_visible(immediate=True, force=True)
                await pilot.pause()

                self.assertTrue(ssd.render().plain.startswith("[ ]"))
                self.assertTrue(cold.disabled)

                self.assertTrue(await pilot.click(ssd))
                await pilot.pause()
                self.assertTrue(ssd.value)
                self.assertTrue(ssd.render().plain.startswith("[x]"))
                self.assertFalse(cold.disabled)
                self.assertTrue(ssd.has_focus)
                self.assertNotEqual(
                    ssd.get_visual_style("toggle--label").foreground.hex6,
                    "#F2B544",
                )

                self.assertTrue(await pilot.click(cold))
                await pilot.pause()
                self.assertTrue(cold.value)
                self.assertTrue(cold.render().plain.startswith("[x]"))

                self.assertTrue(await pilot.click(ssd))
                await pilot.pause()
                self.assertFalse(ssd.value)
                self.assertTrue(ssd.render().plain.startswith("[ ]"))
                self.assertTrue(cold.disabled)
                self.assertFalse(cold.value)
                self.assertTrue(cold.render().plain.startswith("[ ]"))

    async def test_ds4_disk_kv_checkbox_toggles_fields_with_one_click(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(160, 50)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()

                disk_kv = app.query_one("#ds4-kv-enabled", CockpitCheckbox)
                kv_dir = app.query_one("#ds4-kv-dir", Input)
                kv_mb = app.query_one("#ds4-kv-mb", Input)
                disk_kv.scroll_visible(immediate=True, force=True)
                await pilot.pause()

                self.assertTrue(disk_kv.render().plain.startswith("[ ]"))
                self.assertTrue(kv_dir.disabled)
                self.assertTrue(kv_mb.disabled)

                self.assertTrue(await pilot.click(disk_kv))
                await pilot.pause()
                self.assertTrue(disk_kv.value)
                self.assertTrue(disk_kv.render().plain.startswith("[x]"))
                self.assertFalse(kv_dir.disabled)
                self.assertFalse(kv_mb.disabled)

                self.assertTrue(await pilot.click(disk_kv))
                await pilot.pause()
                self.assertFalse(disk_kv.value)
                self.assertTrue(disk_kv.render().plain.startswith("[ ]"))
                self.assertTrue(kv_dir.disabled)
                self.assertTrue(kv_mb.disabled)

    async def test_ds4_deepseek_enables_optimized_dspark_support(self) -> None:
        target = {
            "name": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
            "path": "/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
        }
        support = {
            "name": "DeepSeek-V4-Flash-DSpark-support-0731.gguf",
            "path": "/models/DeepSeek-V4-Flash-DSpark-support-0731.gguf",
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[]),
            patch("ai_toolbox_cockpit.backends.ds4.server.scan_local_models", return_value=[target, support]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(200, 60)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()

                self.assertEqual(app.query_one("#ds4-model", SearchableSelect).value, target["path"])
                self.assertTrue(app.query_one("#ds4-dspark-enabled", Checkbox).value)
                self.assertEqual(app.query_one("#ds4-dspark-model", SearchableSelect).value, support["path"])
                self.assertEqual(app.query_one("#ds4-dspark-confidence", Input).value, "0")
                self.assertFalse(app.query_one("#ds4-ssd-enabled", Checkbox).value)
                self.assertTrue(app.query_one("#ds4-ssd-enabled", Checkbox).disabled)
                model_control = app.query_one("#ds4-dspark-model", SearchableSelect)
                model_label = app.query_one("#ds4-dspark-model-label", Label)
                self.assertIs(model_label.parent, model_control.parent)
                confidence_control = app.query_one("#ds4-dspark-confidence", Input)
                confidence_label = app.query_one("#ds4-dspark-confidence-label", Label)
                self.assertIs(confidence_label.parent, confidence_control.parent)
                self.assertEqual(confidence_label.region.x, confidence_control.region.x)
                self.assertLess(confidence_label.region.y, confidence_control.region.y)

    async def test_ds4_deepseek_vision_encoder_is_selectable(self) -> None:
        legacy_target = {
            "name": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
            "path": "/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
        }
        target = {
            "name": "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf",
            "path": "/models/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf",
        }
        encoder = {
            "name": "DeepSeek-V4-Flash-Vision-Encoder.gguf",
            "path": "/models/DeepSeek-V4-Flash-Vision-Encoder.gguf",
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[]),
            patch(
                "ai_toolbox_cockpit.backends.ds4.server.scan_local_models",
                return_value=[legacy_target, target, encoder],
            ),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(200, 60)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "ds4"
                await pilot.pause()

                self.assertEqual(
                    app.query_one("#ds4-model", SearchableSelect).value,
                    target["path"],
                )
                vision = app.query_one("#ds4-vision", SearchableSelect)
                self.assertFalse(vision.disabled)
                vision.value = encoder["path"]
                await pilot.pause()
                self.assertEqual(vision.value, encoder["path"])

    async def test_vllm_server_controls_have_persistent_labels(self) -> None:
        expected_labels = {
            "vllm-tp": ("vllm-tp-label", "Tensor parallel"),
            "vllm-seqs": ("vllm-seqs-label", "Max sequences"),
            "vllm-context": ("vllm-context-label", "Context length"),
            "vllm-util": ("vllm-util-label", "GPU memory"),
            "vllm-host": ("vllm-host-label", "Host"),
            "vllm-port": ("vllm-port-label", "Port"),
            "vllm-dtype": ("vllm-dtype-label", "Data type"),
            "vllm-attention": ("vllm-attention-label", "Attention backend"),
            "vllm-hf-cache": ("vllm-hf-cache-label", "Hugging Face cache"),
            "vllm-compile-cache": ("vllm-compile-cache-label", "vLLM cache"),
            "vllm-triton-cache": ("vllm-triton-cache-label", "Triton cache"),
            "vllm-aiter-cache": ("vllm-aiter-cache-label", "AITER cache"),
        }
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                backend = app.query_one("#server-backend-select", SearchableSelect)
                backend.value = "vllm"
                await pilot.pause()

                for control_id, (label_id, expected_text) in expected_labels.items():
                    control = app.query_one(f"#{control_id}")
                    label = app.query_one(f"#{label_id}", Label)
                    self.assertEqual(str(label.render()), expected_text)
                    self.assertIs(label.parent, control.parent)
                    self.assertEqual(label.region.x, control.region.x)
                    self.assertLess(label.region.y, control.region.y)

    async def test_vllm_force_eager_uses_compact_checkbox(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                backend = app.query_one("#server-backend-select", SearchableSelect)
                backend.value = "vllm"
                await pilot.pause()

                eager = app.query_one("#vllm-eager", Checkbox)
                self.assertEqual(str(eager.label), "Force eager mode")
                self.assertFalse(eager.value)
                self.assertEqual(eager.region.height, 1)
                self.assertLessEqual(eager.region.width, 21)

                eager.focus()
                await pilot.press("space")
                await pilot.pause()
                self.assertTrue(eager.value)

    async def test_vllm_attention_control_is_the_single_source_of_truth(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "vllm"
                await pilot.pause()

                attention = app.query_one("#vllm-attention", SearchableSelect)
                label = app.query_one("#vllm-attention-label", Label)
                self.assertEqual(len(app.query("#vllm-policy-note")), 0)
                panel = app.query_one("#server-panel-vllm")
                visible_copy = " ".join(str(widget.content) for widget in panel.query(Static))
                self.assertNotIn("policy", visible_copy.lower())
                self.assertNotIn("policy", app.query_one("#vllm-model", SearchableSelect).prompt.lower())
                self.assertNotIn("policy", app.query_one("#vllm-custom-model", Input).placeholder.lower())

                attention.value = "ROCM_ATTN"
                await pilot.pause()
                self.assertEqual(attention.value, "ROCM_ATTN")
                self.assertEqual(attention.query_one(Input).value, "ROCM_ATTN")
                self.assertEqual(str(label.render()), "Attention backend")

                model = app.query_one("#vllm-model", SearchableSelect)
                model.value = "vllm-deepseek-ai-deepseek-v4-flash-0731"
                await pilot.pause()
                self.assertTrue(attention.disabled)
                self.assertEqual(str(label.render()), "Required attention backend")
                self.assertEqual(
                    attention.query_one(Input).value,
                    "ROCM_FLASHMLA_SPARSE_DSV4",
                )

                model.value = "vllm-meta-llama-meta-llama-3-1-8b-instruct"
                await pilot.pause()
                self.assertFalse(attention.disabled)
                self.assertEqual(attention.value, "TRITON_ATTN")
                self.assertEqual(str(label.render()), "Attention backend")
