import json
from pathlib import Path
import struct
import tempfile
import unittest
from ai_toolbox_cockpit.backends.vllm.model_manager import checkpoint_ready, build_prepare_cmd, get_download_cmd, incomplete_files
from ai_toolbox_cockpit.backends.vllm.runner import build_device_probe_cmd
from ai_toolbox_cockpit.catalog.loader import load_model_catalog, load_toolbox_catalog


def weight(path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0,4]}}).encode()
    path.write_bytes(struct.pack("<Q",len(header)) + header + b"data")


class VllmArtifactTests(unittest.TestCase):
    def test_config_alone_is_not_a_ready_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (directory / name).write_text("{}")
            self.assertFalse(checkpoint_ready(directory))
            weight(directory / "model.safetensors")
            self.assertTrue(checkpoint_ready(directory))
            (directory / "model.safetensors").write_bytes((directory / "model.safetensors").read_bytes()[:-1])
            self.assertFalse(checkpoint_ready(directory))

    def test_all_indexed_shards_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "config.json").write_text("{}")
            (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"first":"a.safetensors","last":"b.safetensors"}}))
            weight(directory / "a.safetensors")
            self.assertFalse(checkpoint_ready(directory, draft=True))
            weight(directory / "b.safetensors")
            self.assertTrue(checkpoint_ready(directory, draft=True))

    def test_download_manifest_covers_weights_and_sidecars_at_exact_revision(self):
        entries = [entry for entry in load_model_catalog().backends["vllm"].entries if "download" in entry]
        self.assertEqual(len(entries),3)
        for entry in entries:
            command = get_download_cmd(entry, Path("/models/original"))
            self.assertEqual(command[command.index("--revision")+1],entry["revision"])
            self.assertTrue(all(item["path"] in command for item in entry["download"]["files"]))
        draft = next(entry for entry in entries if entry.get("artifact_role")=="draft")
        self.assertIn("model-kvscales.safetensors",[item["path"] for item in draft["download"]["files"]])

    def test_preparation_has_no_gpu_devices_and_preserves_original(self):
        command = build_prepare_cmd("podman", "example:rolling", Path("/models/original"),Path("/models/prepared"))
        self.assertNotIn("--device",command)
        self.assertIn("/models/original:/models/source:ro",command)
        self.assertIn("--network=none",command)
        with self.assertRaises(ValueError):
            build_prepare_cmd("podman","example:rolling",Path("/models/original"),Path("/models/original/prepared"))

    def test_episode_installation_uses_rolling_public_images(self):
        for toolbox in load_toolbox_catalog().platform_toolboxes("r9700"):
            self.assertNotIn("@sha256:",toolbox.image)
            self.assertFalse(toolbox.backend_config.get("local_image",False))

    def test_measured_allocations_match_server_capacity_not_client_count(self):
        catalog = load_toolbox_catalog()
        expected = {"r9700-llama-vulkan-qwen27-dual-q8":(1089536,16,256), "r9700-llama-rocm-10-qwen27-dual-f16":(544768,8,256)}
        for identifier, values in expected.items():
            defaults = catalog.toolboxes[identifier].backend_config["performance_profiles"]["episode"]["server_defaults"]
            self.assertEqual((defaults["context_size"],defaults["parallel_sequences"],defaults["ubatch_size"]),values)

    def test_device_discovery_uses_selected_runtime_without_model_mounts(self):
        command = build_device_probe_cmd("podman","example:rolling",["--device","/dev/kfd","--group-add","video"])
        self.assertIn("/dev/kfd",command)
        self.assertIn("--network=none",command)
        self.assertNotIn("-v",command)
        self.assertIn("example:rolling",command)


class VllmAcquisitionUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_measured_allocation_speculation_and_download_cancel(self):
        from unittest.mock import patch
        from ai_toolbox_cockpit.app import AiToolboxCockpitApp
        from ai_toolbox_cockpit.widgets import SearchableSelect
        from textual.widgets import Input, Label, TabbedContent
        with (
            tempfile.TemporaryDirectory() as root,
            patch("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed"),
            patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update"),
            patch("ai_toolbox_cockpit.app.available_update", return_value=None),
            patch("ai_toolbox_cockpit.backends.ds4.server.scan_local_models", return_value=[]),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(160, 50)) as pilot:
                app.query_one("#platform-select", SearchableSelect).value = "r9700"
                app.query_one(TabbedContent).active = "tab-servers"
                app.query_one("#server-backend-select", SearchableSelect).value = "vllm"
                await pilot.pause()
                app.query_one("#vllm-image", SearchableSelect).value = "r9700-ggz14-mxfp4-tp2"
                await pilot.pause()
                app.query_one("#vllm-allocation", SearchableSelect).value = "16"
                await pilot.pause()
                self.assertEqual(app.query_one("#vllm-seqs", Input).value, "16")
                self.assertEqual(app.query_one("#vllm-speculation", SearchableSelect).value, "baseline")
                app.query_one("#vllm-speculation", SearchableSelect).value = "dflash2"
                await pilot.pause()
                self.assertEqual(app.query_one("#vllm-seqs", Input).value, "1")
                self.assertEqual(app.query_one("#vllm-allocation", SearchableSelect).value, "everyday")
                app.query_one(TabbedContent).active = "tab-models"
                app.query_one("#model-backend-select", SearchableSelect).value = "vllm"
                await pilot.pause()
                panel = app.query_one("#model-panel-vllm")
                target = Path(root) / "original"
                app.query_one("#vllm-download-directory", Input).value = str(target)
                entry = panel.selected_artifact()
                with patch("ai_toolbox_cockpit.backends.vllm.models.subprocess.run") as run:
                    panel.download_pressed()
                    await pilot.pause()
                    message = str(app.screen.query_one("#confirm_message", Label).render())
                    self.assertIn(entry["revision"], message)
                    self.assertIn(str(target), message)
                    await pilot.click("#btn_no")
                    await pilot.pause()
                    run.assert_not_called()
                self.assertFalse(target.exists())
