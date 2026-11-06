import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.gufo.model_manager import get_download_commands, missing_files, model_size, resolved_files
from ai_toolbox_cockpit.backends.gufo.server_runner import build_server_cmd
from ai_toolbox_cockpit.catalog import CatalogError, ModelCatalog, load_model_catalog, load_toolbox_catalog
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

    def test_downloads_sidecar_repository_to_its_own_directory(self) -> None:
        model = {
            "repo": "example/target", "revision": "target-revision",
            "directory": "target-dir", "files": [{"path": "target.gguf"}],
            "speculation": {
                "repo": "example/draft", "revision": "draft-revision",
                "directory": "draft-dir", "path": "draft.gguf",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            commands = get_download_commands(model, Path(directory))
            destinations = {command[command.index("--local-dir") + 1] for command in commands}
        self.assertEqual(destinations, {f"{directory}/target-dir", f"{directory}/draft-dir"})


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
            patch(
                "ai_toolbox_cockpit.backends.gufo.server.resolved_files",
                side_effect=lambda model: {
                    "targets": {},
                    "model": Path("target.gguf"),
                    "sidecar": (
                        None
                        if model["id"] == "gufo-deepseek-v4-flash-0731-iq2xxs"
                        else Path("mtp.gguf")
                    ),
                },
            ),
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
                self.assertEqual(profile.value, "mtp")
                vision = app.query_one("#gufo-vision", SearchableSelect)
                self.assertEqual(vision.value, "off")
                self.assertFalse(vision.disabled)
                self.assertIn("projector missing", vision._options[1][0])
                vision.value = "on"
                self.assertEqual(app.query_one("#gufo-think", SearchableSelect).value, "high")
                self.assertEqual(
                    {value for _, value in profile._options},
                    {"baseline", "mtp"},
                )
                profile.value = "baseline"
                self.assertEqual(profile.value, "baseline")

                model.value = "gufo-deepseek-v4-flash-0731-iq2xxs"
                await pilot.pause()
                self.assertEqual(profile.value, "baseline")
                self.assertTrue(vision.disabled)
                self.assertEqual(vision.value, "off")
                self.assertIn("sidecar missing", profile._options[1][0])
                self.assertEqual(app.query_one("#gufo-context", Input).value, "262144")
                self.assertEqual(app.query_one("#gufo-max-tokens", Input).value, "32768")
                self.assertEqual(
                    app.query_one("#gufo-max-pending-per-client", Input).value,
                    "4",
                )

                model.value = "gufo-qwen38-27b-ud-q4-k-xl"
                await pilot.pause()
                self.assertEqual(profile.value, "dflash2")
                self.assertIn("DFlash2", profile._options[1][0])
                self.assertFalse(vision.disabled)
                self.assertEqual(vision.value, "off")
                vision.value = "on"
                with patch("ai_toolbox_cockpit.backends.gufo.server.build_server_cmd", return_value=["podman", "run"]) as build:
                    app.query_one("#gufo-port", Input).value = "18080"
                    app.query_one("#gufo-model", SearchableSelect).value = "gufo-qwen38-27b-ud-q4-k-xl"
                    app.query_one("#gufo-start").press()
                    await pilot.pause()
                    self.assertTrue(build.call_args.kwargs["vision_enabled"])


class GufoModelPanelTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_vision_selection_reaches_download_confirmation(self):
        from textual.widgets import Label
        with tempfile.TemporaryDirectory() as directory, patch(
            "ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed", return_value=None
        ), patch("ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update", return_value=None), patch(
            "ai_toolbox_cockpit.app.load_active_platform", return_value="strix-halo"
        ), patch("ai_toolbox_cockpit.app.save_active_platform"):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 60)) as pilot:
                app.query_one("#model-backend-select", SearchableSelect).value = "gufo"
                await pilot.pause()
                model = app.query_one("#gufo-download-model", SearchableSelect)
                vision = app.query_one("#gufo-download-vision", SearchableSelect)
                self.assertEqual(vision.value, "off")
                self.assertFalse(vision.disabled)
                self.assertEqual(str(app.query_one("#gufo-download-vision-label", Label).render()), "Image support")
                vision.value = "on"
                panel = app.query_one("#model-panel-gufo")
                panel._hf_token = "test-token"
                app.query_one("#gufo-models-dir", Input).value = directory
                with patch("ai_toolbox_cockpit.backends.gufo.models.get_download_commands", return_value=[
                    ["hf", "download", "mmproj-BF16.gguf"]
                ]) as commands, patch("ai_toolbox_cockpit.backends.gufo.models.missing_files", return_value=["mmproj-BF16.gguf"]):
                    panel.download_pressed()
                    await pilot.pause()
                    self.assertTrue(panel._pending_vision)
                    self.assertTrue(commands.call_args.kwargs["include_vision"])
                    self.assertTrue(commands.call_args.kwargs["missing_only"])
                    app.pop_screen()
                model.value = "gufo-deepseek-v4-flash-0731-iq2xxs"
                await pilot.pause()
                self.assertTrue(vision.disabled)
                self.assertEqual(vision.value, "off")


class GufoCommandTests(unittest.TestCase):
    def build(
        self, directory: str, *, mode: str = "baseline",
        thinking_effort: str = "high", max_pending_per_client: int = 4,
        extra_args: str = "",
    ) -> list[str]:
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
                thinking_effort=thinking_effort,
                max_pending_per_client=max_pending_per_client,
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
        self.assertEqual(command[command.index("--think") + 1], "on")
        self.assertEqual(command[command.index("--reasoning-effort") + 1], "high")
        self.assertEqual(command[command.index("--max-pending-per-client") + 1], "4")
        self.assertNotIn("--speculative", command)
        self.assertNotIn("--mtp-model", command)
        self.assertIn("keep-groups", command)

    def test_mtp_uses_tested_mode_sidecar_and_draft_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, mode="mtp")
        self.assertEqual(command[command.index("--speculative") + 1], "mtp")
        self.assertEqual(command[command.index("--mtp-model") + 1], "/models/target/sidecar.gguf")
        self.assertEqual(command[command.index("--draft-tokens") + 1], "7")

    def test_thinking_and_client_queue_are_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(
                directory,
                thinking_effort="xhigh",
                max_pending_per_client=8,
            )
        self.assertEqual(command[command.index("--think") + 1], "on")
        self.assertEqual(command[command.index("--reasoning-effort") + 1], "xhigh")
        self.assertEqual(command[command.index("--max-pending-per-client") + 1], "8")

    def test_thinking_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, thinking_effort="off")
        self.assertEqual(command[command.index("--think") + 1], "off")
        self.assertNotIn("--reasoning-effort", command)

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

    def test_dflash2_uses_native_gufo_sidecar_flag_and_draft_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.gguf"
            target.touch()
            sidecar_root = root / "draft"
            sidecar_root.mkdir()
            sidecar = sidecar_root / "dflash.gguf"
            sidecar.touch()
            model = {
                "model_path": target.name,
                "served_model_name": "qwen38-27b",
                "context_size": 262144,
                "files": [{"path": target.name, "size_bytes": 0}],
                "speculation": {
                    "mode": "dflash2", "path": sidecar.name,
                    "size_bytes": 0, "draft_tokens": 7,
                },
            }
            resolved = {"targets": {target.name: target}, "model": target, "sidecar": sidecar}
            with (
                patch("ai_toolbox_cockpit.backends.gufo.server_runner.get_model", return_value=model),
                patch("ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files", return_value=resolved),
            ):
                command = build_server_cmd(
                    engine="podman", image="docker.io/example/gufo:test",
                    engine_args=ROCM_ARGS, platform_id="strix-halo", model_id="qwen",
                    speculation_mode="dflash2", draft_tokens=7,
                )
        self.assertEqual(command[command.index("--speculative") + 1], "dflash2")
        self.assertEqual(command[command.index("--dflash-model") + 1], "/models/speculation/dflash.gguf")
        self.assertEqual(command[command.index("--draft-tokens") + 1], "7")

    def test_extra_args_supply_the_served_model_name(self) -> None:
        for form in (
            "--served-model-name=my-model",
            "--served-model-name my-model",
            "'--served-model-name my-model'",
        ):
            with self.subTest(form=form):
                with tempfile.TemporaryDirectory() as directory:
                    command = self.build(directory, extra_args=form)
                self.assertEqual(command.count("--served-model-name"), 1)
                self.assertEqual(
                    command[command.index("--served-model-name") + 1], "my-model"
                )

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


class GufoVisionTests(unittest.TestCase):
    def models(self):
        return load_model_catalog().backends["gufo"].entries

    def test_projectors_match_the_exact_target_revision_and_are_opt_in(self):
        for model in self.models()[:2]:
            with self.subTest(model=model["id"]):
                vision = model["vision"]
                self.assertEqual(vision["repo"], model["repo"])
                self.assertEqual(vision["revision"], model["revision"])
                text_commands = get_download_commands(model, Path("/tmp/gufo models"))
                self.assertFalse(any("mmproj-BF16.gguf" in command for command in text_commands))
                image_commands = get_download_commands(model, Path("/tmp/gufo models"), include_vision=True)
                command = next(command for command in image_commands if "mmproj-BF16.gguf" in command)
                self.assertEqual(command[command.index("--revision") + 1], vision["revision"])
                self.assertEqual(command[command.index("--local-dir") + 1], "/tmp/gufo models/" + vision["directory"])
                self.assertEqual(model_size(model, include_vision=True) - model_size(model), vision["size_bytes"])
        with self.assertRaisesRegex(ValueError, "no vision profile"):
            get_download_commands(self.models()[2], Path("/tmp"), include_vision=True)

    def test_vision_repair_reuses_complete_targets_and_predictor(self):
        model = self.models()[0]
        with patch("ai_toolbox_cockpit.backends.gufo.model_manager.resolved_files", return_value={
            "targets": {item["path"]: Path("/existing") / Path(item["path"]).name for item in model["files"]},
            "sidecar": Path("/existing/mtp.gguf"), "vision": None,
        }):
            commands = get_download_commands(model, Path("/tmp/gufo"), include_vision=True, missing_only=True)
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][3:4], ["mmproj-BF16.gguf"])
            self.assertFalse(any(item["path"] in commands[0] for item in model["files"]))
        with patch("ai_toolbox_cockpit.backends.gufo.model_manager.resolved_files", return_value={
            "targets": {item["path"]: Path("/existing") for item in model["files"]},
            "sidecar": Path("/existing/mtp.gguf"), "vision": Path("/existing/mmproj-BF16.gguf"),
        }):
            self.assertEqual(get_download_commands(model, Path("/tmp"), include_vision=True, missing_only=True), [])

    def test_schema_rejects_mismatched_projector_provenance_and_unsafe_paths(self):
        original = json.loads((Path(__file__).parents[1] / "ai_toolbox_cockpit/assets/models.json").read_text())
        for key, value in [("repo", "example/other"), ("revision", "a" * 40),
                           ("directory", "../other"), ("path", "mmproj-Q8.gguf"),
                           ("size_bytes", True), ("sha256", "invalid")]:
            data = copy.deepcopy(original)
            data["backends"]["gufo"]["models"][0]["vision"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(CatalogError, "vision"):
                ModelCatalog.from_dict(data)

    def test_same_projector_basename_does_not_mix_model_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, size in enumerate((5, 7)):
                path = root / f"bundle{index}/vision/mmproj-BF16.gguf"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"v" * size)
            # An incomplete file in the selected bundle must not be accepted.
            selected = root / "bundle1/vision/mmproj-BF16.gguf"
            selected.write_bytes(b"v" * 4)
            model = {"directory": "bundle1", "model_path": "target.gguf",
                     "files": [{"path": "target.gguf", "size_bytes": 1}],
                     "vision": {"directory": "bundle1/vision", "path": "mmproj-BF16.gguf", "size_bytes": 7}}
            with patch("ai_toolbox_cockpit.backends.gufo.model_manager.search_roots", return_value=(root,)):
                self.assertIsNone(resolved_files(model)["vision"])
                self.assertNotIn("mmproj-BF16.gguf", missing_files(model))
                self.assertIn("mmproj-BF16.gguf", missing_files(model, include_vision=True))
                selected.write_bytes(b"v" * 7)
                self.assertEqual(resolved_files(model)["vision"], selected)

    def command(self, root, *, vision_enabled=False, mode="baseline", projector=True, supported=True):
        target = root / "target.gguf"
        target.write_bytes(b"target")
        adjacent = root / "mmproj-BF16.gguf"
        adjacent.write_bytes(b"projector")
        model = {"directory": "bundle", "model_path": target.name,
                 "files": [{"path": target.name, "size_bytes": 6}],
                 "served_model_name": "vision-test", "context_size": 262144,
                 "speculation": {"mode": mode, "draft_tokens": 7}}
        if supported:
            model["vision"] = {"path": adjacent.name}
        resolved = {"targets": {target.name: target}, "model": target,
                    "sidecar": root / "draft.gguf", "vision": adjacent if projector else None}
        with patch("ai_toolbox_cockpit.backends.gufo.server_runner.get_model", return_value=model), patch(
            "ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files", return_value=resolved
        ):
            return build_server_cmd(engine="podman", image="example/gufo", engine_args=ROCM_ARGS,
                                    platform_id="strix-halo", model_id="test", vision_enabled=vision_enabled,
                                    speculation_mode=mode)

    def test_text_only_isolates_targets_from_an_adjacent_projector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.command(root)
            self.assertNotIn("--mmproj", command)
            self.assertIn(f"{root}/target.gguf:/models/target/target.gguf:ro", command)
            self.assertNotIn(f"{root}:/models/target:ro", command)
            self.assertFalse(any("/models/vision/" in arg for arg in command))

    def test_vision_mounts_matching_projector_with_baseline_mtp_and_dflash2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("baseline", "mtp", "dflash2"):
                with self.subTest(mode=mode):
                    command = self.command(root, vision_enabled=True, mode=mode)
                    self.assertEqual(command[command.index("--mmproj") + 1], "/models/vision/mmproj-BF16.gguf")
                    self.assertIn(f"{root}/mmproj-BF16.gguf:/models/vision/mmproj-BF16.gguf:ro", command)
                    if mode != "baseline":
                        flag = "--mtp-model" if mode == "mtp" else "--dflash-model"
                        self.assertEqual(command[command.index(flag) + 1], "/models/speculation/draft.gguf")

    def test_vision_rejects_missing_projector_and_unsupported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "repair.*vision projector"):
                self.command(root, vision_enabled=True, projector=False)
            with self.assertRaisesRegex(ValueError, "no vision profile"):
                self.command(root, vision_enabled=True, supported=False)

    def test_projector_cannot_bypass_the_dedicated_control(self):
        from ai_toolbox_cockpit.backends.gufo.server_runner import _extra_arguments
        for arguments in ("--mmproj other.gguf", "--mmproj=other.gguf"):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "dedicated Gufo control"):
                _extra_arguments(arguments)


if __name__ == "__main__":
    unittest.main()
