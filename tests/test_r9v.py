import copy
import hashlib
import json
import runpy
import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from textual.widgets import Button, Input, Label, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.r9v.model_manager import (
    PLE_FILENAME, get_download_cmd, get_paths, incomplete_files, load_packages,
    ple_ready, save_paths, verify_file, verify_package,
)
from ai_toolbox_cockpit.backends.r9v.runner import CONTAINER_NAME, DEFAULTS, build_prepare_cmd, build_server_cmd
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import CatalogError, ModelCatalog
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.widgets import SearchableSelect

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX_ID = "r9700-r9v-rocm-10-0"


def fixture(root):
    package = copy.deepcopy(load_packages()[0])
    for item in package["files"]:
        item.update(size_bytes=4, sha256=hashlib.sha256(b"test").hexdigest())
        path = root / "models" / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    package["ple"].update(size_bytes=4, sha256=hashlib.sha256(b"test").hexdigest())
    (root / "ple").mkdir()
    (root / "ple" / PLE_FILENAME).write_bytes(b"test")
    return package


def options(root, package):
    catalog = load_toolbox_catalog()
    toolbox = catalog.toolboxes[TOOLBOX_ID]
    return dict(engine="podman", image=toolbox.image, platform_id="r9700",
                engine_args=list(catalog.runtime_profiles[toolbox.runtime_profile].engine_args),
                models_dir=root / "models", ple_dir=root / "ple", cache_dir=root / "cache",
                package_id=package["id"])


class R9vTests(TestCase):
    def test_catalog_is_r9700_only_and_keeps_llama_default(self):
        catalog = load_toolbox_catalog()
        for platform in catalog.platforms:
            self.assertEqual("r9v" in catalog.platform_backend_ids(platform.id), platform.id == "r9700")
        self.assertEqual(catalog.platform("r9700").defaults["r9v"], TOOLBOX_ID)
        self.assertEqual(catalog.platform("r9700").defaults["llama_cpp"], "r9700-llama-rocm-10-0")
        self.assertTrue(catalog.toolboxes[TOOLBOX_ID].toolbox_compatible)
        self.assertEqual(load_model_catalog().backends["r9v"].kind, "r9v_package")
        package = load_packages()[0]
        self.assertEqual(package["repo"], "Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS")
        self.assertEqual(package["revision"], "bf836f0c20b6c92fcad4226ad3115eb8a19f7582")
        self.assertEqual(len(package["files"]), 19)
        self.assertEqual(sum(f["role"] == "target" for f in package["files"]), 3)
        self.assertEqual(package["ple"]["size_bytes"], 28800138240)

    def test_download_is_pinned_and_includes_every_required_artifact(self):
        package = load_packages()[0]
        cmd = get_download_cmd(package, Path("/tmp/models with spaces"))
        self.assertEqual(cmd[1:3], ["download", package["repo"]])
        self.assertEqual(cmd[3:-4], [item["path"] for item in package["files"]])
        self.assertEqual(cmd[-4:], ["--revision", package["revision"], "--local-dir", "/tmp/models with spaces"])

    def test_schema_rejects_unpinned_or_incomplete_packages(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        for mutation in (lambda p: p.update(revision="main"),
                         lambda p: p["files"].pop(2),
                         lambda p: p["files"][0].update(path="../escape"),
                         lambda p: p["files"][0].update(sha256="bad"),
                         lambda p: p["ple"].update(size_bytes=-1)):
            data = copy.deepcopy(original)
            mutation(data["backends"]["r9v"]["models"][0])
            with self.assertRaises(CatalogError):
                ModelCatalog.from_dict(data)

    def test_inventory_and_hash_checks_detect_partial_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = fixture(root)
            self.assertFalse(incomplete_files(package, root / "models"))
            verify_package(package, root / "models")
            file = root / "models" / package["files"][0]["path"]
            file.write_bytes(b"evil")
            self.assertFalse(incomplete_files(package, root / "models"))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_package(package, root / "models")
            file.write_bytes(b"par")
            self.assertTrue(incomplete_files(package, root / "models"))
            self.assertTrue(ple_ready(package, root / "ple"))
            (root / "ple" / PLE_FILENAME).write_bytes(b"partial")
            self.assertFalse(ple_ready(package, root / "ple"))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_file(root / "ple" / PLE_FILENAME, package["ple"]["sha256"])

    def test_default_launch_matches_tested_container_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = fixture(root)
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                cmd = build_server_cmd(**options(root, package))
            self.assertEqual(cmd[-1], "r9v-serve")
            self.assertIn("127.0.0.1:8004:8000", cmd)
            self.assertIn("--ipc=host", cmd)
            self.assertIn("keep-groups", cmd)
            self.assertIn("crun", cmd)
            self.assertIn(f"{root / 'models'}:/models:ro", cmd)
            self.assertIn(f"{root / 'ple' / PLE_FILENAME}:/ple/{PLE_FILENAME}:ro", cmd)
            for value in ("R9V_VISIBLE_DEVICES=0,1", "R9V_MAX_MODEL_LEN=131072", "R9V_MAX_NUM_SEQS=1",
                          "R9V_MAX_NUM_BATCHED_TOKENS=1024", "R9V_KV_CACHE_MEMORY_BYTES=2285670400",
                          "R9V_CPU_OFFLOAD_GB=112.5", "R9V_CPU_OFFLOAD_GB_BY_DEVICE=112.5,112.5",
                          "R9V_MTP_SPEC_TOKENS=2", "R9V_SERIALIZE_EXPERT_LOAD=1", "R9V_PLE_RESIDENCY_MODE=ssd"):
                self.assertIn(value, cmd)
            self.assertFalse((root / "cache").exists(), "Command preview must not create paths")

    def test_docker_and_edited_values_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = fixture(root); args = options(root, package)
            args.update(engine="docker", values={"devices": "1,2", "context": "32768", "port": "9000",
                        "host": "::1", "batch": "512", "served_model": "my-qwen"},
                        api_key="private-key", extra_args="--disable-log-requests")
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                cmd = build_server_cmd(**args)
            self.assertIn("[::1]:9000:8000", cmd)
            self.assertIn("R9V_MAX_MODEL_LEN=32768", cmd)
            self.assertIn("R9V_VISIBLE_DEVICES=1,2", cmd)
            self.assertNotIn("keep-groups", cmd)
            self.assertNotIn("crun", cmd)
            self.assertEqual(cmd[-4:], ["r9v-serve", "--api-key", "private-key", "--disable-log-requests"])

    def test_invalid_launch_values_and_missing_payload_fail_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = fixture(root); args = options(root, package)
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                for edits in ({"devices": "0"}, {"devices": "0,0"}, {"devices": "0,1,2"},
                              {"context": "262145"}, {"port": "0"}, {"kv_bytes": "0"},
                              {"expert_cache_slots": "-1"}, {"expert_cache_slots": "17"},
                              {"offload": "nan"}, {"offload_devices": "112.5"}):
                    with self.subTest(edits=edits), self.assertRaises(ValueError):
                        build_server_cmd(**args, values=edits)
                for extra in ("--tensor-parallel-size=3", "--async-scheduling", "--max-model-len 1", "--api-key secret"):
                    with self.assertRaisesRegex(ValueError, "cannot override"):
                        build_server_cmd(**args, extra_args=extra)
                with self.assertRaisesRegex(ValueError, "R9700"):
                    build_server_cmd(**{**args, "platform_id": "strix-halo"})
                (root / "ple" / PLE_FILENAME).unlink()
                with self.assertRaisesRegex(ValueError, "Prepare PLE"):
                    build_server_cmd(**args)
                (root / "models" / package["files"][2]["path"]).unlink()
                with self.assertRaisesRegex(ValueError, "Download / Repair"):
                    build_server_cmd(**args)

    def test_256k_launch_forwards_both_memory_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = fixture(root)
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                cmd = build_server_cmd(**options(root, package), values={
                    "context": "262144", "kv_bytes": "4160749568", "expert_cache_slots": "0"})
            for value in ("R9V_MAX_MODEL_LEN=262144", "R9V_KV_CACHE_MEMORY_BYTES=4160749568",
                          "R9V_TIERED_EXPERT_CACHE_SLOTS=0", "R9V_MAX_NUM_SEQS=1"):
                self.assertIn(value, cmd)

    def test_preparation_is_cpu_only_and_keeps_model_mount_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = fixture(root)
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                cmd = build_prepare_cmd(engine="podman", image="image:tag", models_dir=root / "models",
                                        ple_dir=root / "ple", package_id=package["id"])
            self.assertEqual(cmd[-3:], ["image:tag", "r9v-model", "prepare"])
            self.assertIn(f"{root / 'models'}:/models:ro", cmd)
            self.assertNotIn("--device", cmd)
            self.assertIn("--network=none", cmd)

    def test_source_import_preserves_r9v_model_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {"llama-models": "[]", "ds4-models": '{"repo":"example/models","models":[]}',
                       "vllm-models": "MODEL_TABLE = {}", "comfy-manager": "MODEL_FAMILIES = []"}
            argv = ["import_source_catalogs.py"]
            for flag, content in sources.items():
                path = root / flag; path.write_text(content); argv += [f"--{flag}", str(path)]
            output = root / "models.json"
            argv += ["--comfy-workflows", str(root), "--output", str(output)]
            with patch("sys.argv", argv):
                runpy.run_path(str(ROOT / "scripts/import_source_catalogs.py"), run_name="__main__")
            self.assertEqual(ModelCatalog.from_dict(json.loads(output.read_text())).backends["r9v"],
                             load_model_catalog().backends["r9v"])

    def test_toolbox_source_import_preserves_r9v_platform_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama = root / "llama.json"; llama.write_text('{"platforms":[]}')
            ds4 = root / "ds4.json"; ds4.write_text('{"groups":[]}')
            output = root / "toolboxes.json"
            argv = ["import_source_toolboxes.py", "--llama", str(llama), "--ds4", str(ds4),
                    "--existing", str(ROOT / "ai_toolbox_cockpit/assets/toolboxes.json"), "--output", str(output)]
            with patch("sys.argv", argv):
                runpy.run_path(str(ROOT / "scripts/import_source_toolboxes.py"), run_name="__main__")
            from ai_toolbox_cockpit.catalog.schema import ToolboxCatalog
            catalog = ToolboxCatalog.from_dict(json.loads(output.read_text()))
            self.assertEqual(catalog.platform("r9700").defaults["r9v"], TOOLBOX_ID)
            self.assertIn(TOOLBOX_ID, catalog.platform("r9700").toolbox_ids)
            self.assertNotIn(TOOLBOX_ID, catalog.platform("strix-halo").toolbox_ids)


class R9vAppTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict("os.environ", {"XDG_CONFIG_HOME": str(self.tmp)}))
        for target in ("ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed",
                       "ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update",
                       "ai_toolbox_cockpit.app.available_update"):
            self.stack.enter_context(patch(target, return_value=None))
        self.stack.enter_context(patch("ai_toolbox_cockpit.app.load_active_platform", return_value="r9700"))
        self.stack.enter_context(patch("ai_toolbox_cockpit.app.save_active_platform"))
        for target in ("server", "models"):
            self.stack.enter_context(patch(f"ai_toolbox_cockpit.backends.r9v.{target}.detect_container_engines",
                                          return_value=(ContainerEngine.PODMAN,)))

    async def test_context_buttons_apply_memory_settings_and_restore_defaults(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(120, 45)) as pilot:
            app.query_one(TabbedContent).active = "tab-servers"
            app.query_one("#server-backend-select", SearchableSelect).value = "r9v"
            await pilot.pause()
            app.query_one("#r9v-devices", Input).value = "1,2"
            app.query_one("#r9v-context-256k", Button).press()
            await pilot.pause()
            for field, expected in {"context": "262144", "kv_bytes": "4160749568",
                                    "expert_cache_slots": "0", "sequences": "1", "batch": "1024"}.items():
                self.assertEqual(app.query_one(f"#r9v-{field}", Input).value, expected)
            self.assertTrue(app.query_one("#r9v-expert_cache_slots-label", Label))
            self.assertEqual(app.query_one("#r9v-devices", Input).value, "1,2")
            app.query_one("#r9v-context-128k", Button).press()
            await pilot.pause()
            for field in ("context", "kv_bytes", "expert_cache_slots", "sequences", "batch"):
                self.assertEqual(app.query_one(f"#r9v-{field}", Input).value, DEFAULTS[field])

    async def test_platform_selection_reveals_tested_package_and_labels(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(100, 35)) as pilot:
            for tab, selector, controls in (
                ("tab-servers", "#server-backend-select", ("engine", "image", "model")),
                ("tab-models", "#model-backend-select", ("download-model", "model-engine", "model-image"))):
                app.query_one(TabbedContent).active = tab
                app.query_one(selector, SearchableSelect).value = "r9v"
                await pilot.pause()
                for control in controls:
                    self.assertTrue(app.query_one(f"#r9v-{control}-label", Label).visible)
                self.assertEqual(app.query_one("#r9v-download-model", SearchableSelect).value, load_packages()[0]["id"])
            self.assertIn("Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS", str(app.query_one("#r9v-package-details").render()))
            app.query_one("#platform-select", SearchableSelect).value = "strix-halo"
            await pilot.pause()
            self.assertNotIn("r9v", {v for _, v in app.query_one("#model-backend-select", SearchableSelect)._options})
            self.assertTrue(app.query_one("#r9v-start", Button).disabled)

    async def test_download_cancel_has_no_writes_then_uses_edited_path_and_hashes(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(120, 45)) as pilot:
            panel = app.query_one("#model-panel-r9v")
            app.query_one(TabbedContent).active = "tab-models"
            app.query_one("#model-backend-select", SearchableSelect).value = "r9v"
            await pilot.pause()
            model = self.tmp / "new models"; ple = self.tmp / "new ple"
            app.query_one("#r9v-model-models_dir", Input).value = str(model)
            app.query_one("#r9v-model-ple_dir", Input).value = str(ple)
            panel._hf_token_prompted = True
            panel.download_pressed(); await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn("Qwen Community License", message)
            self.assertIn(str(model), message)
            self.assertIn("--revision", message)
            await pilot.click("#btn_no"); await pilot.pause()
            self.assertFalse(model.exists())
            with patch.object(app, "suspend", return_value=nullcontext()), \
                 patch("ai_toolbox_cockpit.backends.r9v.models.subprocess.run") as run, \
                 patch("ai_toolbox_cockpit.backends.r9v.models.verify_package") as verify:
                panel._operation_confirmed(True)
                self.assertEqual(run.call_args.args[0][-1], str(model))
                verify.assert_called_once()
            self.assertEqual(get_paths()["models_dir"], model)
            self.assertEqual(app.query_one("#r9v-models_dir", Input).value, str(model))

    async def test_launch_preview_redacts_key_and_cancel_does_not_start(self):
        package = fixture(self.tmp)
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(140, 45)) as pilot:
            panel = app.query_one("#server-panel-r9v")
            app.query_one(TabbedContent).active = "tab-servers"
            app.query_one("#server-backend-select", SearchableSelect).value = "r9v"
            await pilot.pause()
            for key, path in (("models_dir", "models"), ("ple_dir", "ple"), ("cache_dir", "cache")):
                app.query_one(f"#r9v-{key}", Input).value = str(self.tmp / path)
            app.query_one("#r9v-api-key", Input).value = "do-not-print-this"
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                panel.start_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertNotIn("do-not-print-this", message)
            self.assertIn("redacted", message)
            self.assertIn("r9v-serve", message)
            await pilot.click("#btn_no"); await pilot.pause()
            self.assertFalse((self.tmp / "cache").exists())
            with patch.object(app, "suspend", return_value=nullcontext()), \
                 patch("ai_toolbox_cockpit.backends.r9v.server.run_foreground_server") as run:
                panel._start_confirmed(True)
                self.assertEqual(run.call_args.args[2], CONTAINER_NAME)
                self.assertNotIn("do-not-print-this", str(run.call_args.kwargs["display_command"]))
            settings = (self.tmp / "ai-toolbox-cockpit/config.json").read_text()
            self.assertNotIn("do-not-print-this", settings)

    async def test_prepare_previews_cpu_command_and_cancel_does_not_extract(self):
        package = fixture(self.tmp)
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(120, 45)) as pilot:
            panel = app.query_one("#model-panel-r9v")
            app.query_one(TabbedContent).active = "tab-models"
            app.query_one("#model-backend-select", SearchableSelect).value = "r9v"
            await pilot.pause()
            app.query_one("#r9v-model-models_dir", Input).value = str(self.tmp / "models")
            app.query_one("#r9v-model-ple_dir", Input).value = str(self.tmp / "new ple")
            with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
                panel.prepare_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn("r9v-model prepare", message)
            self.assertNotIn("--device", message)
            await pilot.click("#btn_no"); await pilot.pause()
            self.assertFalse((self.tmp / "new ple").exists())
            with patch.object(app, "suspend", return_value=nullcontext()), \
                 patch("ai_toolbox_cockpit.backends.r9v.models.subprocess.run") as run:
                panel._operation_confirmed(True)
                self.assertEqual(run.call_args.args[0][-2:], ["r9v-model", "prepare"])
