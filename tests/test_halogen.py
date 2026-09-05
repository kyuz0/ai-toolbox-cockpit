import copy
import json
import runpy
import subprocess
import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from textual.widgets import Button, Input, Label, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.halogen.model_manager import (
    get_download_cmd, get_models_dir, incomplete_files, load_bundles, save_models_dir,
)
from ai_toolbox_cockpit.backends.halogen.runner import CONTAINER_NAME, build_server_cmd
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import CatalogError, ModelCatalog, ToolboxCatalog
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.runtime.images import LocalImage, inspect_local_images
from ai_toolbox_cockpit.runtime.interactive import InteractiveBackend, InteractiveRuntime
from ai_toolbox_cockpit.views.toolboxes import ToolboxesView
from ai_toolbox_cockpit.widgets import SearchableSelect


ROOT = Path(__file__).resolve().parents[1]
TOOLBOX_ID = "strix-halo-halogen-flash"


def small_bundle(directory: Path) -> dict:
    """Small fixture files exercise real inventory checks without any weights."""
    bundle = copy.deepcopy(load_bundles()[0])
    for item in bundle["files"]:
        item["size_bytes"] = 4
        path = directory / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    return bundle


class HalogenTests(TestCase):
    def test_source_import_preserves_curated_halogen_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "llama-models": "[]", "ds4-models": '{"repo": "example/models", "models": []}',
                "vllm-models": "MODEL_TABLE = {}", "comfy-manager": "MODEL_FAMILIES = []",
            }
            argv = ["import_source_catalogs.py"]
            for flag, content in sources.items():
                path = root / flag
                path.write_text(content)
                argv.extend([f"--{flag}", str(path)])
            output = root / "models.json"
            argv.extend(["--comfy-workflows", str(root), "--output", str(output)])
            with patch("sys.argv", argv):
                runpy.run_path(str(ROOT / "scripts/import_source_catalogs.py"), run_name="__main__")
            imported = ModelCatalog.from_dict(json.loads(output.read_text()))
            self.assertEqual(imported.backends["halogen"], load_model_catalog().backends["halogen"])

    def test_catalog_is_strix_only_and_server_only(self):
        catalog = load_toolbox_catalog()
        item = catalog.toolboxes[TOOLBOX_ID]
        self.assertFalse(item.toolbox_compatible)
        self.assertEqual(item.image, "ghcr.io/peonist-ai/halogen-flash-server:0.4.4")
        self.assertEqual(item.feature_state("interactive"), "unavailable")
        for platform in catalog.platforms:
            self.assertEqual("halogen" in catalog.platform_backend_ids(platform.id), platform.id == "strix-halo")
        self.assertTrue(catalog.toolboxes["strix-halo-llama-rocm-10-0"].toolbox_compatible)
        self.assertEqual(load_model_catalog().backends["halogen"].kind, "hgn_bundle")

    def test_schema_rejects_inconsistent_server_only_flag(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/toolboxes.json").read_text())
        for flag in ("false", 0):
            data = copy.deepcopy(original)
            data["toolboxes"][-1]["toolbox_compatible"] = flag
            with self.assertRaisesRegex(CatalogError, "must be boolean"):
                ToolboxCatalog.from_dict(data)
        original["toolboxes"][-1]["features"]["interactive"] = "supported"
        with self.assertRaisesRegex(CatalogError, "server-only"):
            ToolboxCatalog.from_dict(original)

    def test_bundle_schema_requires_sidecar_tokenizer_and_safe_paths(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        for filename in ("qwen38-flash-next-w4b.overlay.hgn", "tokenizer/tokenizer.json"):
            data = copy.deepcopy(original)
            entry = data["backends"]["halogen"]["models"][0]
            entry["files"] = [item for item in entry["files"] if item["path"] != filename]
            with self.assertRaisesRegex(CatalogError, "must include"):
                ModelCatalog.from_dict(data)
        for filename in ("../escape.hgn", "/escape.hgn", "--bad.hgn"):
            data = copy.deepcopy(original)
            data["backends"]["halogen"]["models"][0]["files"][0]["path"] = filename
            with self.assertRaisesRegex(CatalogError, "invalid or duplicate path"):
                ModelCatalog.from_dict(data)

    def test_download_selects_exact_precision_and_pins_revision(self):
        quality, speed = load_bundles()
        for bundle in (quality, speed):
            command = get_download_cmd(bundle, Path("/tmp/halogen models"))
            self.assertIn(bundle["checkpoint"], command)
            self.assertIn(bundle["overlay"], command)
            self.assertIn("tokenizer/chat_template.jinja", command)
            self.assertIn("tokenizer/tokenizer_config.json", command)
            self.assertEqual(command[command.index("--revision") + 1], bundle["revision"])
            self.assertEqual(command[-1], "/tmp/halogen models")
        self.assertNotIn(speed["overlay"], get_download_cmd(quality, Path("/tmp/models")))
        self.assertNotIn(quality["overlay"], get_download_cmd(speed, Path("/tmp/models")))

    def test_directory_creation_persistence_and_invalid_paths(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"XDG_CONFIG_HOME": temporary}):
            self.assertEqual(get_models_dir(), Path("~/halogen-models").expanduser().resolve())
            path = Path(temporary) / "new" / "models with spaces"
            self.assertTrue(save_models_dir(str(path)))
            self.assertTrue(path.is_dir())
            self.assertEqual(get_models_dir(), path)
            self.assertFalse(save_models_dir(""))
            bad = Path(temporary) / "file"
            bad.touch()
            self.assertFalse(save_models_dir(str(bad)))
            self.assertEqual(get_models_dir(), path)

    def test_inventory_detects_truncated_sidecar_and_external_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = small_bundle(root / "models")
            self.assertEqual(incomplete_files(bundle, root / "models"), [])
            overlay = root / "models" / bundle["overlay"]
            overlay.write_bytes(b"bad")
            self.assertEqual([item["path"] for item in incomplete_files(bundle, root / "models")], [bundle["overlay"]])
            overlay.unlink()
            outside = root / "outside.hgn"
            outside.write_bytes(b"test")
            overlay.symlink_to(outside)
            self.assertEqual(len(incomplete_files(bundle, root / "models")), 1)

    def test_native_commands_preserve_entrypoint_mount_bundle_and_publish_only_api(self):
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        profile = load_toolbox_catalog().runtime_profiles[toolbox.runtime_profile]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models with spaces"
            bundle = small_bundle(root)
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                for engine in ("podman", "docker"):
                    command = build_server_cmd(engine=engine, image=toolbox.image,
                                               engine_args=list(profile.engine_args), platform_id="strix-halo",
                                               models_dir=root, bundle_id=bundle["id"], port=9000)
                    self.assertEqual(command[-1], toolbox.image)
                    self.assertNotIn("--entrypoint", command)
                    self.assertEqual(command[command.index("-p") + 1], "127.0.0.1:9000:9000")
                    self.assertEqual(command.count("-p"), 1)
                    self.assertIn("HALOGEN_API_PORT=9000", command)
                    self.assertIn(f"HALOGEN_CK_OVERLAY=/models/{bundle['overlay']}", command)
                    self.assertIn("HALOGEN_TOKENIZER=/models/tokenizer", command)
                    self.assertIn(f"{root}:/models:ro", command)
                    self.assertIn("--ipc=host", command)
                    self.assertIn("memlock=-1:-1", command)
                    self.assertEqual("keep-groups" in command, engine == "podman")
                    self.assertEqual("render" in command, engine == "docker")
                    self.assertNotIn("HALOGEN_DOWNLOAD", " ".join(command))
                    self.assertIn(CONTAINER_NAME, command)

    def test_builder_refuses_invalid_settings_and_incomplete_bundles(self):
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        with tempfile.TemporaryDirectory() as temporary:
            bundle = small_bundle(Path(temporary))
            options = dict(engine="podman", image=toolbox.image, engine_args=[], platform_id="strix-halo",
                           models_dir=Path(temporary), bundle_id=bundle["id"])
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                for invalid in ({"platform_id": "r9700"}, {"engine": "toolbox"}, {"port": 65536},
                                {"context_size": 1048576}, {"kv_pool_positions": 1}, {"kv_slots": 0},
                                {"prompt_cache": "3"}, {"host": ""}):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        build_server_cmd(**{**options, **invalid})
                (Path(temporary) / bundle["overlay"]).unlink()
                with self.assertRaisesRegex(ValueError, "Missing/incomplete.*overlay"):
                    build_server_cmd(**options)

    def test_image_inspection_falls_back_to_docker_without_starting_anything(self):
        image = load_toolbox_catalog().toolboxes[TOOLBOX_ID].image
        def runner(command, **kwargs):
            self.assertEqual(command[1:3], ["image", "inspect"])
            if command[0] == "podman":
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, '[{"Created": "2026-09-05"}]')
        result = inspect_local_images((image,), (ContainerEngine.PODMAN, ContainerEngine.DOCKER), runner)
        self.assertEqual(result[image].engine, ContainerEngine.DOCKER)
        self.assertEqual(result[image].created, "2026-09-05")


class HalogenAppTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.dict("os.environ", {"XDG_CONFIG_HOME": self.temporary}))
        for target in (
            "ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed",
            "ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update",
            "ai_toolbox_cockpit.app.available_update",
        ):
            self.stack.enter_context(patch(target, return_value=None))
        self.stack.enter_context(patch("ai_toolbox_cockpit.backends.halogen.server.detect_container_engines",
                                      return_value=(ContainerEngine.PODMAN,)))

    async def test_server_image_pull_and_enter_need_no_toolbox_wrapper(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 45)) as pilot:
            view = app.query_one(ToolboxesView)
            toolbox = app.toolbox_catalog.toolboxes[TOOLBOX_ID]
            view.selected_toolboxes = {TOOLBOX_ID}
            view.refresh_rows()
            self.assertFalse(app.query_one("#toolbox-enter", Button).disabled)
            with patch.object(view, "notify") as notify, patch("ai_toolbox_cockpit.views.toolboxes.enter_toolbox") as enter:
                view.enter_pressed()
                self.assertIn("Server Mode", notify.call_args.args[0])
                enter.assert_not_called()
            with patch("ai_toolbox_cockpit.views.toolboxes.detect_interactive_backend", return_value=None), \
                 patch("ai_toolbox_cockpit.views.toolboxes.detect_container_engines", return_value=(ContainerEngine.DOCKER,)):
                view.create_update_pressed()
                await pilot.pause()
                message = str(app.screen.query_one("#confirm_message", Label).render())
                self.assertIn(f"docker pull {toolbox.image}", message)
                self.assertNotIn("toolbox create", message)
                await pilot.click("#btn_no")
                await pilot.pause()
                with patch.object(app, "suspend", return_value=nullcontext()), \
                     patch("ai_toolbox_cockpit.views.toolboxes.subprocess.run") as run, \
                     patch("ai_toolbox_cockpit.views.toolboxes.create_toolbox") as create:
                    view._create_update_confirmed(True)
                    run.assert_called_once_with(["docker", "pull", toolbox.image], check=True)
                    create.assert_not_called()

    async def test_server_image_update_and_delete_use_image_store(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 45)) as pilot:
            view = app.query_one(ToolboxesView)
            toolbox = app.toolbox_catalog.toolboxes[TOOLBOX_ID]
            view.selected_toolboxes = {TOOLBOX_ID}
            view.server_images = {toolbox.image: LocalImage(toolbox.image, ContainerEngine.DOCKER)}
            view.refresh_rows()
            self.assertFalse(app.query_one("#toolbox-delete", Button).disabled)
            with patch("ai_toolbox_cockpit.views.toolboxes.get_remote_image_date") as remote:
                view.create_update_pressed()
                await pilot.pause()
                self.assertIn("docker pull", str(app.screen.query_one("#confirm_message", Label).render()))
                remote.assert_not_called()
                await pilot.click("#btn_no")
                await pilot.pause()
            view.delete_pressed()
            await pilot.pause()
            self.assertIn("docker image rm", str(app.screen.query_one("#confirm_message", Label).render()))
            await pilot.click("#btn_no")
            await pilot.pause()
            with patch.object(app, "suspend", return_value=nullcontext()), \
                 patch("ai_toolbox_cockpit.views.toolboxes.subprocess.run") as run, \
                 patch("ai_toolbox_cockpit.views.toolboxes.delete_toolbox") as delete:
                view._delete_confirmed(True)
                run.assert_called_once_with(["docker", "image", "rm", toolbox.image], check=True)
                delete.assert_not_called()

    async def test_mixed_batch_pulls_server_and_creates_only_real_toolbox(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 45)) as pilot:
            view = app.query_one(ToolboxesView)
            toolbox = app.toolbox_catalog.toolboxes["strix-halo-llama-rocm-10-0"]
            server = app.toolbox_catalog.toolboxes[TOOLBOX_ID]
            view.selected_toolboxes = {toolbox.id, server.id}
            runtime = InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)
            with patch("ai_toolbox_cockpit.views.toolboxes.detect_interactive_backend", return_value=runtime), \
                 patch("ai_toolbox_cockpit.views.toolboxes.detect_container_engines", return_value=(ContainerEngine.PODMAN,)):
                view.create_update_pressed()
                await pilot.pause()
                await pilot.click("#btn_no")
                await pilot.pause()
                with patch.object(app, "suspend", return_value=nullcontext()), \
                     patch("ai_toolbox_cockpit.views.toolboxes.subprocess.run") as run, \
                     patch("ai_toolbox_cockpit.views.toolboxes.create_toolbox") as create:
                    view._create_update_confirmed(True)
                    run.assert_called_once_with(["podman", "pull", server.image], check=True)
                    self.assertEqual(create.call_count, 1)
                    self.assertEqual(create.call_args.args[1], toolbox.container_name)

    async def test_download_uses_edited_path_and_refreshes_server(self):
        app = AiToolboxCockpitApp()
        directory = Path(self.temporary) / "new models"
        async with app.run_test(size=(180, 45)) as pilot:
            panel = app.query_one("#model-panel-halogen")
            app.query_one(TabbedContent).active = "tab-models"
            app.query_one("#model-backend-select", SearchableSelect).value = "halogen"
            await pilot.pause()
            app.query_one("#halogen-models-dir", Input).value = str(directory)
            panel._hf_token_prompted = True
            panel.download_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn(str(directory), message)
            self.assertIn("overlay.hgn", message)
            self.assertIn("tokenizer/tokenizer.json", message)
            self.assertFalse(directory.exists())
            await pilot.click("#btn_no")
            await pilot.pause()
            with patch.object(app, "suspend", return_value=nullcontext()), \
                 patch("ai_toolbox_cockpit.backends.halogen.models.subprocess.run") as run:
                panel._download_confirmed(True)
                self.assertEqual(run.call_args.args[0][-1], str(directory))
            self.assertTrue(directory.is_dir())
            self.assertEqual(app.query_one("#halogen-server-dir", Input).value, str(directory))
            self.assertEqual(get_models_dir(), directory)

    async def test_server_requires_bundle_then_previews_and_suspends(self):
        app = AiToolboxCockpitApp()
        directory = Path(self.temporary) / "models"
        bundle = small_bundle(directory)
        async with app.run_test(size=(180, 45)) as pilot:
            panel = app.query_one("#server-panel-halogen")
            app.query_one(TabbedContent).active = "tab-servers"
            app.query_one("#server-backend-select", SearchableSelect).value = "halogen"
            await pilot.pause()
            app.query_one("#halogen-server-dir", Input).value = str(directory)
            with patch.object(panel, "notify") as notify:
                panel.start_pressed()
                self.assertIn("Download or repair", notify.call_args.args[0])
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                panel.start_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn("HALOGEN_CK_OVERLAY", message)
            self.assertIn("127.0.0.1:8731:8731", message)
            await pilot.click("#btn_no")
            await pilot.pause()
            suspended = []
            from contextlib import contextmanager
            @contextmanager
            def suspension():
                suspended.append(True)
                yield
                suspended.pop()
            def run(*args):
                self.assertEqual(suspended, [True])
                self.assertEqual(args[2], CONTAINER_NAME)
            with patch.object(app, "suspend", side_effect=suspension), \
                 patch("ai_toolbox_cockpit.backends.halogen.server.run_foreground_server", side_effect=run) as runner:
                panel._start_confirmed(True)
                runner.assert_called_once()

    async def test_halogen_selects_have_visible_labels(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 45)) as pilot:
            for tab, backend, ids in (
                ("tab-servers", "#server-backend-select", ("engine", "image", "model", "prompt-cache")),
                ("tab-models", "#model-backend-select", ("download-model",)),
            ):
                app.query_one(TabbedContent).active = tab
                app.query_one(backend, SearchableSelect).value = "halogen"
                await pilot.pause()
                for control in ids:
                    label = app.query_one(f"#halogen-{control}-label", Label)
                    self.assertTrue(label.visible)
                    self.assertGreater(label.region.width, 0)
