import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.gufo.model_manager import resolved_files
from ai_toolbox_cockpit.backends.gufo.server_runner import build_server_cmd
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.widgets import SearchableSelect


ROCM_ARGS = ["--device", "/dev/dri", "--device", "/dev/kfd", "--group-add", "render"]


class GufoCatalogTests(unittest.TestCase):
    def test_experimental_toolbox_and_tested_models_are_catalogued(self) -> None:
        toolbox_catalog = load_toolbox_catalog()
        toolbox = toolbox_catalog.toolboxes["strix-halo-gufo-rocm-10-0"]
        self.assertEqual(toolbox.backend, "gufo")
        self.assertEqual(toolbox.channel, "experimental")
        self.assertEqual(toolbox.maturity, "experimental")
        self.assertEqual(
            toolbox.image,
            "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0-gufo",
        )
        self.assertEqual(
            toolbox.backend_config["source_revision"],
            "b42fa8c89cbbeb0941f8deb7947f598702ed5125",
        )
        self.assertEqual(
            toolbox.backend_config["published_digest"],
            "sha256:cf41f792fe1594121178974fa3354fad9f88be798bdecc3583422d6e84481295",
        )
        self.assertIn(toolbox.id, toolbox_catalog.platform("strix-halo").toolbox_ids)
        self.assertEqual(
            toolbox_catalog.platform("strix-halo").defaults["gufo"], toolbox.id
        )

        models = load_model_catalog().backends["gufo"]
        self.assertEqual(models.kind, "gguf_bundle")
        self.assertEqual(
            {entry["id"] for entry in models.entries},
            {
                "gufo-qwen38-flash-next-ud-q4-k-xl",
                "gufo-qwen38-27b-ud-q4-k-xl",
                "gufo-deepseek-v4-flash-0731-iq2xxs",
            },
        )


class GufoModelDiscoveryTests(unittest.TestCase):
    def test_resolves_pinned_bundle_paths_without_recursive_home_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "bundle" / "quant" / "target.gguf"
            sidecar = root / "bundle" / "MTP" / "sidecar.gguf"
            target.parent.mkdir(parents=True)
            sidecar.parent.mkdir(parents=True)
            target.write_bytes(b"target")
            sidecar.write_bytes(b"sidecar")
            model = {
                "directory": "bundle",
                "model_path": "quant/target.gguf",
                "files": [{"path": "quant/target.gguf", "size_bytes": 6}],
                "speculation": {"path": "MTP/sidecar.gguf", "size_bytes": 7},
            }
            with patch(
                "ai_toolbox_cockpit.backends.gufo.model_manager.search_roots",
                return_value=(root,),
            ):
                resolved = resolved_files(model)
        self.assertEqual(resolved["model"], target)
        self.assertEqual(resolved["sidecar"], sidecar)


class GufoServerPanelTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_model_profiles_are_populated_after_platform_mount(self) -> None:
        with (
            patch(
                "ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed",
                return_value=None,
            ),
            patch(
                "ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update",
                return_value=None,
            ),
            patch(
                "ai_toolbox_cockpit.app.load_active_platform",
                return_value="strix-halo",
            ),
            patch("ai_toolbox_cockpit.app.save_active_platform"),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 60)) as pilot:
                backend = app.query_one("#server-backend-select", SearchableSelect)
                backend.value = "gufo"
                await pilot.pause()

                model = app.query_one("#gufo-model", SearchableSelect)
                profile = app.query_one("#gufo-speculation", SearchableSelect)
                self.assertEqual(
                    model.value,
                    "gufo-qwen38-flash-next-ud-q4-k-xl",
                )
                self.assertEqual(len(model._options), 3)
                self.assertEqual(profile.value, "baseline")
                self.assertEqual(
                    {value for _, value in profile._options},
                    {"baseline", "mtp"},
                )


class GufoCommandTests(unittest.TestCase):
    def build(self, directory: str, *, mode: str = "baseline", extra_args: str = "") -> list[str]:
        root = Path(directory)
        target = root / "target.gguf"
        target.touch()
        sidecar = root / "sidecar.gguf"
        sidecar.touch()
        model = {
            "id": "test-model",
            "model_path": target.name,
            "served_model_name": "test-served-model",
            "context_size": 133760,
            "files": [{"path": target.name, "size_bytes": 0}],
            "speculation": {
                "mode": "mtp",
                "path": sidecar.name,
                "size_bytes": 0,
                "draft_tokens": 7,
            },
        }
        resolved = {
            "targets": {target.name: target},
            "model": target,
            "sidecar": sidecar,
        }
        with (
            patch("ai_toolbox_cockpit.backends.gufo.server_runner.get_model", return_value=model),
            patch("ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files", return_value=resolved),
        ):
            return build_server_cmd(
                engine="podman",
                image="docker.io/example/gufo:test",
                engine_args=ROCM_ARGS,
                platform_id="strix-halo",
                model_id=model["id"],
                speculation_mode=mode,
                host="127.0.0.1",
                port=18080,
                context_size=133760,
                sessions=2,
                max_tokens=16384,
                draft_tokens=7,
                extra_args=extra_args,
            )

    def test_baseline_uses_gufo_serve_and_read_only_model_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory)
        self.assertEqual(command[command.index("gufo"):command.index("gufo") + 2], ["gufo", "serve"])
        self.assertIn(f"{directory}:/models/target:ro", command)
        self.assertEqual(command[command.index("--model") + 1], "/models/target/target.gguf")
        self.assertEqual(command[command.index("--context") + 1], "133760")
        self.assertEqual(command[command.index("--max-tokens") + 1], "16384")
        self.assertEqual(command[command.index("--served-model-name") + 1], "test-served-model")
        self.assertNotIn("--speculative", command)
        self.assertNotIn("--mtp-model", command)
        self.assertIn("keep-groups", command)

    def test_mtp_uses_tested_mode_sidecar_and_draft_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, mode="mtp")
        self.assertEqual(command[command.index("--speculative") + 1], "mtp")
        self.assertEqual(command[command.index("--mtp-model") + 1], "/models/target/sidecar.gguf")
        self.assertEqual(command[command.index("--draft-tokens") + 1], "7")

    def test_dspark_uses_native_gufo_sidecar_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.gguf"
            target.touch()
            sidecar_root = root / "draft"
            sidecar_root.mkdir()
            sidecar = sidecar_root / "dspark.gguf"
            sidecar.touch()
            model = {
                "model_path": target.name,
                "served_model_name": "deepseek",
                "context_size": 133760,
                "files": [{"path": target.name, "size_bytes": 0}],
                "speculation": {"mode": "dspark", "path": sidecar.name, "size_bytes": 0},
            }
            resolved = {"targets": {target.name: target}, "model": target, "sidecar": sidecar}
            with (
                patch("ai_toolbox_cockpit.backends.gufo.server_runner.get_model", return_value=model),
                patch("ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files", return_value=resolved),
            ):
                command = build_server_cmd(
                    engine="podman", image="docker.io/example/gufo:test",
                    engine_args=ROCM_ARGS, platform_id="strix-halo", model_id="deepseek",
                    speculation_mode="dspark",
                )
        self.assertEqual(command[command.index("--dspark-model") + 1], "/models/speculation/dspark.gguf")
        self.assertNotIn("--speculative", command)

    def test_rejects_owned_options_in_extra_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "dedicated Gufo control"):
                self.build(directory, extra_args="--context=4096")

    def test_rejects_missing_target_and_sidecar(self) -> None:
        model = {
            "model_path": "target.gguf",
            "served_model_name": "test",
            "context_size": 4096,
            "files": [{"path": "target.gguf", "size_bytes": 1}],
            "speculation": {"mode": "mtp", "path": "mtp.gguf", "size_bytes": 1},
        }
        with (
            patch("ai_toolbox_cockpit.backends.gufo.server_runner.get_model", return_value=model),
            patch(
                "ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files",
                return_value={"targets": {"target.gguf": None}, "model": None, "sidecar": None},
            ),
            self.assertRaisesRegex(ValueError, "Missing/incomplete"),
        ):
            build_server_cmd(
                engine="podman", image="image", engine_args=[], platform_id="strix-halo",
                model_id="test", speculation_mode="mtp",
            )


if __name__ == "__main__":
    unittest.main()
