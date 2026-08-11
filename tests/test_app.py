import tempfile
from pathlib import Path
from unittest.mock import patch
from unittest import IsolatedAsyncioTestCase

from rich.text import Text

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Static, Tab, TabbedContent


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

    async def test_toolbox_checkbox_remains_visible_when_selected(self) -> None:
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                table = app.query_one("#toolbox-catalog-table", DataTable)
                row = table.get_row_index("strix-vllm-dev")

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
                self.assertIn("strix-vllm-dev", toolboxes_view.selected_toolboxes)

    async def test_model_tables_are_backend_owned_and_local_inventory_rescans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            with (
                patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None),
                patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
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
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.get_inference_profiles", return_value=profiles),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                label = app.query_one("#server-backend-label", Label)
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
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=[local_model]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 45)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()

                profile = app.query_one("#llama-profile", SearchableSelect)
                extra_args = app.query_one("#llama-extra-args", Input).value
                self.assertEqual(profile.value, "Thinking (Effort: High)")
                self.assertIn('"reasoning_effort":"high"', extra_args)
                self.assertNotIn('"reasoning_effort":"max"', extra_args)

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
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
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
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
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
            patch("ai_toolbox_cockpit.views.benchmarks.inspect_installed_toolboxes", return_value=()),
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
