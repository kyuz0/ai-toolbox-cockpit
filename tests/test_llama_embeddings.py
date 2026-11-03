import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input, TabbedContent, TextArea
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.llama_cpp.config import get_model_config, get_toolbox_defaults
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import get_download_cmd, scan_local_models
from ai_toolbox_cockpit.backends.llama_cpp.models import get_download_sources
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd
from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.widgets import SearchableSelect

REPO = "ggml-org/embeddinggemma-2-GGUF"
MODEL = "embeddinggemma-2-Q8_0.gguf"
PROJECTOR = "mmproj-embeddinggemma-2-Q8_0.gguf"
REV = "6ea3e3a309d12a4779cfda98b8c19800d2d5da71"
ROOT = "/home/kyuz0/models"
PATH = f"{ROOT}/embeddinggemma-2-GGUF/{MODEL}"

class EmbeddingTests(unittest.TestCase):
    def test_download_revision_inventory_and_model_defaults(self):
        source = next(s for s in get_download_sources(load_model_catalog().backends["llama_cpp"].entries) if s["repo"] == REPO)
        self.assertEqual(source["revision"], REV)
        self.assertEqual(source["recommended_pattern"], MODEL)
        self.assertIn(PROJECTOR, source["description"])
        for filename in (MODEL, PROJECTOR):
            cmd = get_download_cmd(REPO, filename, source["revision"])
            self.assertIn(filename, cmd)
            self.assertEqual(cmd[cmd.index("--revision") + 1], REV)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in (MODEL, PROJECTOR):
                (root / name).touch()
            with patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=root):
                self.assertEqual([m["path"] for m in scan_local_models()], [str(root / MODEL)])
        config = get_model_config(PATH)
        self.assertEqual(config["server_mode"], "embedding")
        self.assertEqual(get_toolbox_defaults(config, "strix-halo-llama-rocm-10-0")["ubatch_size"], 8192)

    def test_builder_keeps_embedding_mode_and_rejects_invalid_allocations(self):
        kwargs = dict(engine="podman", image="example:rocm", model_path=PATH, context_size=8192, use_fa=True, use_no_mmap=True, custom_args="", batch_size=8192, ubatch_size=8192, parallel_sequences=1, engine_args=[])
        with patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=Path(ROOT)):
            cmd = build_server_cmd(**kwargs)
            self.assertIn("--embedding", cmd)
            self.assertEqual(cmd[cmd.index("--pooling")+1], "mean")
            self.assertNotIn("--jinja", cmd)
            for change in ({"context_size":8193}, {"ubatch_size":512}, {"batch_size":4096}, {"custom_args":"--jinja"}, {"custom_args":"--pooling=last"}, {"draft_model_path":PATH}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    build_server_cmd(**{**kwargs, **change})
            chat = build_server_cmd(**{**kwargs, "model_path":f"{ROOT}/custom-chat.gguf", "custom_args":"--jinja"})
            self.assertNotIn("--embedding", chat)
            self.assertIn("--jinja", chat)

class EmbeddingUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_to_server_profile_and_chat_transition(self):
        local = [{"name":MODEL, "path":PATH}, {"name":"custom-chat.gguf", "path":f"{ROOT}/custom-chat.gguf"}]
        with (
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed"),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update"),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.app.load_active_platform", return_value="strix-halo"),
            patch("ai_toolbox_cockpit.app.save_active_platform"),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.detect_container_engines", return_value=[]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.models.scan_local_models", return_value=local),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models", return_value=local),
            patch("ai_toolbox_cockpit.backends.llama_cpp.server.get_local_vision_projectors", return_value=[Path(PATH).with_name(PROJECTOR)]),
            patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=Path(ROOT)),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(140, 65)) as pilot:
                app.query_one(TabbedContent).active = "tab-models"
                await pilot.pause()
                app.query_one("#llama-download-repo", SearchableSelect).value = REPO
                await pilot.pause()
                downloads = app.query_one("#model-panel-llama_cpp")
                self.assertEqual(downloads._download_sources[REPO]["revision"], REV)
                downloads._show_quants(REPO, [PROJECTOR, MODEL], {MODEL:309855456, PROJECTOR:554821056})
                await pilot.pause()
                self.assertIn("Recommended", app.screen.options[0])
                self.assertNotIn("R9700", app.screen.options[0])
                self.assertIn(PROJECTOR, app.screen.options[1])
                if output := os.environ.get("EMBEDDING_TEST_OUTPUT"):
                    Path(output).mkdir(parents=True, exist_ok=True)
                    app.save_screenshot("downloads.svg", path=output)
                app.pop_screen(); await pilot.pause()
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()
                engine = app.query_one("#llama-engine", SearchableSelect)
                engine.set_options([("podman", "podman")]); engine.value = "podman"
                app.query_one("#llama-image", SearchableSelect).value = "strix-halo-llama-rocm-10-0"
                app.query_one("#llama-model", SearchableSelect).value = PATH
                app.query_one("#llama-port", Input).value = "18081"
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-profile", SearchableSelect).value, "Embeddings")
                for field in ("context", "batch", "ubatch"):
                    self.assertEqual(app.query_one(f"#llama-{field}", Input).value, "8192")
                self.assertEqual(app.query_one("#llama-extra-args", TextArea).text, "")
                from ai_toolbox_cockpit.backends.llama_cpp.server import LlamaCppServerPanel
                panel = app.query_one(LlamaCppServerPanel)
                panel.start_pressed()
                await pilot.pause()
                text_cmd = panel._pending_command
                self.assertIn("--embedding", text_cmd)
                self.assertNotIn("--mmproj", text_cmd)
                app.pop_screen(); await pilot.pause()
                app.query_one("#llama-projector", SearchableSelect).value = str(Path(PATH).with_name(PROJECTOR))
                await pilot.pause()
                with patch("ai_toolbox_cockpit.backends.llama_cpp.server.os.path.isfile", return_value=True):
                    panel.start_pressed()
                await pilot.pause()
                command = panel._pending_command
                self.assertEqual(command[command.index("--mmproj")+1], f"/models/embeddinggemma-2-GGUF/{PROJECTOR}")
                if output := os.environ.get("EMBEDDING_TEST_OUTPUT"):
                    Path(output).mkdir(parents=True, exist_ok=True)
                    Path(output, "command.json").write_text(json.dumps(command, indent=2)+"\n")
                    app.save_screenshot("command.svg", path=output)
                app.pop_screen(); await pilot.pause()
                if output:
                    app.save_screenshot("server.svg", path=output)
                app.query_one("#llama-model", SearchableSelect).value = local[1]["path"]
                await pilot.pause()
                self.assertEqual(app.query_one("#llama-context", Input).value, "126976")
                self.assertEqual(app.query_one("#llama-extra-args", TextArea).text, "--jinja")
                self.assertEqual(app.query_one("#llama-projector", SearchableSelect).value, "")
