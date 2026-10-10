import copy
import json
import runpy
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from textual.widgets import Button, Checkbox, Input, Label, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.halogen import npu_host
from ai_toolbox_cockpit.backends.halogen.model_manager import (
    get_download_cmd, get_models_dir, get_npu_download_cmds, get_npu_model, incomplete_files,
    incomplete_npu_files, load_bundles, load_npu_models, save_models_dir,
)
from ai_toolbox_cockpit.backends.halogen.runner import CONTAINER_NAME, build_server_cmd
from ai_toolbox_cockpit.backends.halogen.server import _npu_checkbox_id
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import CatalogError, ModelCatalog, ToolboxCatalog
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.runtime.images import LocalImage, inspect_local_images
from ai_toolbox_cockpit.runtime.interactive import InteractiveBackend, InteractiveRuntime
from ai_toolbox_cockpit.settings import save_backend_settings
from ai_toolbox_cockpit.views.toolboxes import ToolboxesView
from ai_toolbox_cockpit.widgets import SearchableSelect


ROOT = Path(__file__).resolve().parents[1]
TOOLBOX_ID = "strix-halo-halogen-flash"


def small_bundle(directory: Path, entry: dict | None = None) -> dict:
    """Small fixture files exercise real inventory checks without any weights."""
    bundle = copy.deepcopy(entry if entry is not None else load_bundles()[0])
    for item in bundle["files"]:
        item["size_bytes"] = 4
        path = directory / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    return bundle


def small_npu(directory: Path, model_id: str) -> tuple[dict, dict[str, dict]]:
    """Write tiny NPU files and return the entry and the resolver the patches use."""
    entries = {entry["id"]: copy.deepcopy(entry) for entry in load_npu_models()}
    for item in entries[model_id]["files"]:
        item["size_bytes"] = 4
        path = directory / "npu" / model_id / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    donor_id = entries[model_id].get("devices_from")
    if donor_id:
        for item in entries[donor_id]["files"]:
            if item["path"].startswith("devices/"):
                item["size_bytes"] = 4
                path = directory / "npu" / donor_id / item["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
    return entries[model_id], entries


@contextmanager
def patched_npu(entries: dict[str, dict]):
    """Resolve NPU entries through the fixture in every module that looks them up."""
    def resolve(model_id):
        entry = entries.get(model_id)
        if entry is None:
            raise ValueError("Select a catalogued Halogen NPU model.")
        return entry

    with patch("ai_toolbox_cockpit.backends.halogen.runner.get_npu_model", side_effect=resolve), \
         patch("ai_toolbox_cockpit.backends.halogen.model_manager.get_npu_model", side_effect=resolve):
        yield


@contextmanager
def npu_host_ready():
    """Present a host whose NPU device, XRT and fabric clock all pass the launch checks."""
    with patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.npu_device_available", return_value=True), \
         patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.xrt_mount_arguments",
               return_value=["-v", "/opt/xilinx/xrt:/opt/xilinx/xrt:ro"]), \
         patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.fabric_clock_held", return_value=True):
        yield


class NpuHostTests(TestCase):
    """Pure host probes: no containers, no devices, no network."""

    def test_device_availability_follows_the_accel_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(npu_host.npu_device_available(root))
            node = root / "dev" / "accel" / "accel0"
            node.parent.mkdir(parents=True)
            node.touch()
            self.assertTrue(npu_host.npu_device_available(root))

    def test_directory_xrt_mounts_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in npu_host.XRT_LIBRARIES:
                path = root / "opt" / "xilinx" / "xrt" / "lib" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(npu_host.xrt_mount_arguments(root),
                             ["-v", f"{root / 'opt' / 'xilinx' / 'xrt'}:/opt/xilinx/xrt:ro"])

    def test_system_xrt_mounts_each_library_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "usr" / "lib"
            for name in npu_host.XRT_LIBRARIES:
                base.mkdir(parents=True, exist_ok=True)
                (base / name).touch()
            arguments = npu_host.xrt_mount_arguments(root)
            self.assertEqual(len(arguments), 12)
            for name in npu_host.XRT_LIBRARIES:
                self.assertIn(f"{base / name}:/opt/xilinx/xrt/lib/{name}:ro", arguments)
                self.assertIn(f"{base / name}:{base / name}:ro", arguments)

    def test_links_into_the_system_libraries_fall_back_to_per_file_mounts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "usr" / "lib"
            xrt_lib = root / "opt" / "xilinx" / "xrt" / "lib"
            for name in npu_host.XRT_LIBRARIES:
                base.mkdir(parents=True, exist_ok=True)
                (base / name).touch()
                xrt_lib.mkdir(parents=True, exist_ok=True)
                (xrt_lib / name).symlink_to(base / name)
            arguments = npu_host.xrt_mount_arguments(root)
            self.assertIn(f"{base / npu_host.XRT_LIBRARIES[0]}:/opt/xilinx/xrt/lib/{npu_host.XRT_LIBRARIES[0]}:ro",
                          arguments)
            self.assertNotIn(f"{root / 'opt' / 'xilinx' / 'xrt'}:/opt/xilinx/xrt:ro", arguments)

    def test_missing_xrt_names_the_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "xrt-plugin-amdxdna"):
                npu_host.xrt_mount_arguments(Path(temporary))

    def test_fabric_clock_is_held_only_for_high_or_the_starred_top_level(self):
        def write_device(root: Path, mode: str, levels: list[str]) -> None:
            device = root / "sys" / "class" / "drm" / "card0" / "device"
            device.mkdir(parents=True)
            (device / "power_dpm_force_performance_level").write_text(mode + "\n")
            (device / "pp_dpm_fclk").write_text("\n".join(levels) + "\n")

        cases = (
            ("high", ["0: 400Mhz", "1: 1000Mhz", "2: 2000Mhz"], True),
            ("manual", ["0: 400Mhz", "1: 1000Mhz", "2: 2000Mhz *"], True),
            ("manual", ["0: 400Mhz", "1: 1000Mhz *", "2: 2000Mhz"], False),
            ("auto", ["0: 400Mhz", "1: 1000Mhz", "2: 2000Mhz *"], False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(npu_host.fabric_clock_held(root))
            for index, (mode, levels, expected) in enumerate(cases):
                with self.subTest(mode=mode, levels=levels):
                    case_root = root / str(index)
                    write_device(case_root, mode, levels)
                    self.assertEqual(npu_host.fabric_clock_held(case_root), expected)


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
        self.assertEqual(item.image, "ghcr.io/peonist-ai/halogen-flash-server:latest")
        self.assertEqual(item.feature_state("interactive"), "unavailable")
        for platform in catalog.platforms:
            self.assertEqual("halogen" in catalog.platform_backend_ids(platform.id), platform.id == "strix-halo")
        self.assertTrue(catalog.toolboxes["strix-halo-llama-rocm-10-0"].toolbox_compatible)
        self.assertEqual(load_model_catalog().backends["halogen"].kind, "hgn_bundle")

    def test_schema_rejects_inconsistent_server_only_flag(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/toolboxes.json").read_text())
        for flag in ("false", 0):
            data = copy.deepcopy(original)
            next(entry for entry in data["toolboxes"] if entry["backend"] == "halogen")["toolbox_compatible"] = flag
            with self.assertRaisesRegex(CatalogError, "must be boolean"):
                ToolboxCatalog.from_dict(data)
        next(entry for entry in original["toolboxes"] if entry["backend"] == "halogen")["features"]["interactive"] = "supported"
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
        quality, speed = load_bundles()[:2]
        for bundle in load_bundles():
            command = get_download_cmd(bundle, Path("/tmp/halogen models"))
            self.assertIn(bundle["checkpoint"], command)
            self.assertIn(bundle.get("overlay", bundle.get("ngram_table")), command)
            self.assertIn("tokenizer/chat_template.jinja", command)
            self.assertIn("tokenizer/tokenizer_config.json", command)
            self.assertEqual(command[command.index("--revision") + 1], bundle["revision"])
            self.assertEqual(command[-1], "/tmp/halogen models")
            self.assertEqual("qwen38-flash-next-vision.hgn" in command, "vision_tower" in bundle)
        self.assertNotIn(speed["overlay"], get_download_cmd(quality, Path("/tmp/models")))
        self.assertNotIn(quality["overlay"], get_download_cmd(speed, Path("/tmp/models")))

    def test_v2_schema_requires_exactly_one_complete_companion(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        index = next(i for i, entry in enumerate(original["backends"]["halogen"]["models"])
                     if entry["id"] == "qwen38-flash-next-v2")
        for companion in (None, "missing.hgn", "tokenizer/tokenizer.json", "", True):
            data = copy.deepcopy(original)
            entry = data["backends"]["halogen"]["models"][index]
            if companion is None:
                del entry["ngram_table"]
            else:
                entry["ngram_table"] = companion
            with self.subTest(companion=companion), self.assertRaises(CatalogError):
                ModelCatalog.from_dict(data)
        entry = original["backends"]["halogen"]["models"][index]
        entry["overlay"] = "qwen38-flash-next-w4b.overlay.hgn"
        with self.assertRaisesRegex(CatalogError, "exactly one"):
            ModelCatalog.from_dict(original)

    def test_v2_launch_mounts_table_without_w4b_files_and_blocks_incomplete_table(self):
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        self.assertEqual([entry["id"] for entry in load_bundles() if entry.get("recommended")],
                         ["qwen38-flash-next-v2"])
        for entry in load_bundles():
            if "ngram_table" not in entry:
                continue
            with self.subTest(bundle=entry["id"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = small_bundle(root, entry)
                with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                    for engine in ("podman", "docker"):
                        options = dict(engine=engine, image=toolbox.image, engine_args=[],
                                       platform_id="strix-halo", models_dir=root, bundle_id=bundle["id"])
                        command = build_server_cmd(**options)
                        self.assertIn("HALOGEN_CHECKPOINT=/models/qwen38-flash-next-v2.hgn", command)
                        self.assertIn("HALOGEN_NGRAM_TABLE=/models/qwen38-flash-next-ngram.hgn", command)
                        self.assertIn("HALOGEN_CK_OVERLAY=none", command)
                        self.assertNotIn("w4b", " ".join(command))
                        self.assertIn("--network=none", command)
                        mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]
                        self.assertEqual(mounts, [f"{root / item['path']}:/models/{item['path']}:ro"
                                                  for item in bundle["files"]])
                        table = root / bundle["ngram_table"]
                        for content in (b"bad", None):
                            if content is None:
                                table.unlink()
                            else:
                                table.write_bytes(content)
                            with self.assertRaisesRegex(ValueError, "Missing/incomplete.*ngram"):
                                build_server_cmd(**options)
                        table.write_bytes(b"test")

    def test_vision_schema_requires_a_listed_hgn_sidecar(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        for tower in ("missing.hgn", "../escape.hgn", "tokenizer/tokenizer.json", "", True):
            data = copy.deepcopy(original)
            data["backends"]["halogen"]["models"][0]["vision_tower"] = tower
            with self.subTest(tower=tower), self.assertRaisesRegex(CatalogError, "vision_tower"):
                ModelCatalog.from_dict(data)

    def test_vision_launch_requires_complete_sidecar_and_text_leaves_it_off(self):
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        for entry in load_bundles():
            with self.subTest(bundle=entry["id"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = small_bundle(root, entry)
                options = dict(engine="podman", image=toolbox.image, engine_args=[],
                               platform_id="strix-halo", models_dir=root, bundle_id=bundle["id"])
                with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                    command = build_server_cmd(**options)
                    tower = bundle.get("vision_tower")
                    if tower:
                        self.assertIn(f"HALOGEN_VISION_TOWER=/models/{tower}", command)
                        for content in (b"", b"par"):
                            (root / tower).write_bytes(content)
                            self.assertEqual(incomplete_files(bundle, root), [bundle["files"][-1]])
                            with self.assertRaisesRegex(ValueError, "Missing/incomplete.*vision"):
                                build_server_cmd(**options)
                        (root / tower).unlink()
                        with self.assertRaisesRegex(ValueError, "Missing/incomplete.*vision"):
                            build_server_cmd(**options)
                    else:
                        # A sidecar left by a previous vision download must not enable images.
                        (root / "qwen38-flash-next-vision.hgn").write_bytes(b"test")
                        self.assertFalse(any(arg.startswith("HALOGEN_VISION_TOWER=")
                                             for arg in build_server_cmd(**options)))

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

    def test_native_commands_preserve_entrypoint_and_isolate_network_and_mounts(self):
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
                    self.assertIn("--pull=always", command[:command.index(toolbox.image)])
                    self.assertNotIn("--entrypoint", command)
                    self.assertNotIn("-p", command)
                    self.assertIn("--network=none", command)
                    self.assertIn("--cap-drop=NET_ADMIN", command)
                    self.assertIn("--cap-drop=NET_RAW", command)
                    self.assertIn("no-new-privileges", command)
                    self.assertIn("HALOGEN_API_PORT=9000", command)
                    self.assertIn(f"HALOGEN_CK_OVERLAY=/models/{bundle['overlay']}", command)
                    self.assertIn("HALOGEN_TOKENIZER=/models/tokenizer", command)
                    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]
                    self.assertEqual(mounts, [f"{root / item['path']}:/models/{item['path']}:ro"
                                              for item in bundle["files"]])
                    self.assertIn("--ipc=host", command)
                    self.assertIn("memlock=-1:-1", command)
                    self.assertEqual("keep-groups" in command, engine == "podman")
                    self.assertEqual("render" in command, engine == "docker")
                    self.assertNotIn("HALOGEN_DOWNLOAD", " ".join(command))
                    self.assertIn(CONTAINER_NAME, command)

    def test_profile_cannot_override_network_or_add_host_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = small_bundle(root)
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                for args in (["--network=host"], ["--net", "bridge"], ["--privileged"],
                             ["-v", "/:/host"], ["--mount=type=bind,src=/,dst=/host"],
                             ["--device", "/dev/infiniband"], ["--cap-add=NET_ADMIN"],
                             ["--pid=host"], ["--env", "HF_TOKEN=secret"],
                             ["--security-opt", "label=disable"], ["--device"]):
                    for engine in ("podman", "docker"):
                        with self.subTest(args=args, engine=engine), self.assertRaisesRegex(ValueError, "isolation"):
                            build_server_cmd(engine=engine, image="example/image:latest", engine_args=args,
                                             platform_id="strix-halo", models_dir=root, bundle_id=bundle["id"])

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

    def test_npu_catalogue_covers_the_curated_models(self):
        entries = load_npu_models()
        self.assertEqual([entry["id"] for entry in entries], [
            "decider-0.8b", "decider-4b", "qwen3-embedding-0.6b", "qwen3-embedding-4b",
            "qwen3-reranker-0.6b", "qwen3-reranker-4b", "qwen3guard-gen-0.6b",
            "qwen3.5-2b", "flux2-klein-4b",
        ])
        self.assertEqual([entry["id"] for entry in entries if entry.get("default_enabled")], [
            "decider-0.8b", "qwen3-embedding-0.6b", "qwen3-reranker-0.6b",
            "qwen3guard-gen-0.6b", "qwen3.5-2b",
        ])
        for entry in entries:
            paths = {item["path"] for item in entry["files"]}
            self.assertIn(f"{entry['id']}.hnpw", paths)
            self.assertIn("tokenizer/tokenizer.json", paths)
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            if entry.get("devices_from"):
                self.assertIn(entry["devices_from"], {item["id"] for item in entries})
            else:
                self.assertTrue(any(path.startswith("devices/") and path.endswith(".elf")
                                    for path in paths))
        shared = {entry["id"]: entry.get("devices_from") for entry in entries}
        self.assertEqual(shared["qwen3-reranker-0.6b"], "qwen3-embedding-0.6b")
        self.assertEqual(shared["qwen3guard-gen-0.6b"], "qwen3-embedding-0.6b")
        self.assertEqual(shared["qwen3-reranker-4b"], "qwen3-embedding-4b")

    def test_npu_schema_rejects_incomplete_or_unsafe_entries(self):
        original = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        cases = (
            (lambda models: models[0]["files"].remove(
                next(item for item in models[0]["files"] if item["path"].endswith(".hnpw"))), "hnpw"),
            (lambda models: models[0]["files"].remove(
                next(item for item in models[0]["files"] if item["path"] == "tokenizer/tokenizer.json")),
             "tokenizer/tokenizer.json"),
            (lambda models: models[0]["files"][0].update({"path": "../escape.hnpw"}),
             "invalid or duplicate path"),
            (lambda models: models[0]["files"][0].update({"path": "-bad.hnpw"}),
             "invalid or duplicate path"),
            (lambda models: models[0].update({"revision": "main"}), "revision"),
            (lambda models: models[0].update({"task": "translate"}), "task"),
            (lambda models: models[0].update({"devices_from": "missing-model"}), "devices_from"),
            (lambda models: models[0].update({"devices_from": models[0]["id"]}), "itself"),
            (lambda models: models[2].update({"devices_from": "qwen3guard-gen-0.6b"}),
             "owns its device program"),
            (lambda models: models[1].update({"id": models[0]["id"]}), "invalid or duplicate"),
            (lambda models: models[0].update({"default_enabled": "yes"}), "default_enabled"),
        )
        for apply, pattern in cases:
            data = copy.deepcopy(original)
            apply(data["backends"]["halogen"]["npu_models"])
            with self.subTest(pattern=pattern), self.assertRaisesRegex(CatalogError, pattern):
                ModelCatalog.from_dict(data)

    def test_npu_catalogue_is_halogen_only_and_required(self):
        data = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        data["backends"]["llama_cpp"]["npu_models"] = data["backends"]["halogen"]["npu_models"]
        with self.assertRaisesRegex(CatalogError, "only supported for the halogen"):
            ModelCatalog.from_dict(data)
        data = json.loads((ROOT / "ai_toolbox_cockpit/assets/models.json").read_text())
        del data["backends"]["halogen"]["npu_models"]
        with self.assertRaisesRegex(CatalogError, "npu_models must be a non-empty array"):
            ModelCatalog.from_dict(data)

    def test_npu_download_pins_revisions_and_borrows_the_device_program(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for model_id in ("decider-0.8b", "qwen3-reranker-0.6b"):
                with self.subTest(model=model_id):
                    entry = get_npu_model(model_id)
                    commands = get_npu_download_cmds(entry, root)
                    self.assertEqual(len(commands), 2 if entry.get("devices_from") else 1)
                    self.assertEqual(commands[0][commands[0].index("--revision") + 1], entry["revision"])
                    self.assertIn(f"{entry['id']}.hnpw", commands[0])
                    self.assertEqual(commands[0][commands[0].index("--local-dir") + 1],
                                     str(root.resolve() / "npu" / entry["id"]))
                    if entry.get("devices_from"):
                        donor = get_npu_model(entry["devices_from"])
                        self.assertEqual(commands[1][commands[1].index("--revision") + 1], donor["revision"])
                        self.assertEqual(commands[1][commands[1].index("--local-dir") + 1],
                                         str(root.resolve() / "npu" / donor["id"]))
                        for item in donor["files"]:
                            if item["path"].startswith("devices/"):
                                self.assertIn(item["path"], commands[1])

    def test_npu_inventory_checks_the_shared_device_program(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry, entries = small_npu(root, "qwen3guard-gen-0.6b")
            with patched_npu(entries):
                self.assertEqual(incomplete_npu_files(entry, root), [])
                donor_id = entry["devices_from"]
                victim = root / "npu" / donor_id / "devices" / "u0.elf"
                victim.unlink()
                self.assertEqual([(owner, item["path"]) for owner, item in incomplete_npu_files(entry, root)],
                                 [(donor_id, "devices/u0.elf")])
                victim.symlink_to(root / "outside.elf")
                self.assertEqual(len(incomplete_npu_files(entry, root)), 1)

    def test_npu_launch_mounts_models_device_and_xrt(self):
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            bundle = small_bundle(root)
            entry, entries = small_npu(root, "qwen3guard-gen-0.6b")
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle), \
                 patched_npu(entries), npu_host_ready():
                command = build_server_cmd(
                    engine="podman", image=toolbox.image, engine_args=[], platform_id="strix-halo",
                    models_dir=root, bundle_id=bundle["id"], npu_models=(entry["id"],),
                )
            self.assertIn(f"HALOGEN_NPU_MODELS={entry['id']}", command)
            self.assertIn("/dev/accel/accel0", command)
            self.assertIn("--network=none", command)
            mounts = [command[index + 1] for index, argument in enumerate(command) if argument == "-v"]
            self.assertIn(f"{root.resolve() / 'npu' / entry['id']}:/models/npu/{entry['id']}:ro", mounts)
            donor_id = entry["devices_from"]
            self.assertIn(f"{root.resolve() / 'npu' / donor_id}:/models/npu/{donor_id}:ro", mounts)
            self.assertIn("/opt/xilinx/xrt:/opt/xilinx/xrt:ro", mounts)
            self.assertEqual(len(mounts), len(set(mounts)))

    def test_npu_launch_blocks_missing_files_device_xrt_and_clock(self):
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            bundle = small_bundle(root)
            options = dict(engine="podman", image=toolbox.image, engine_args=[],
                           platform_id="strix-halo", models_dir=root, bundle_id=bundle["id"])
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "NPU models in Models"):
                    build_server_cmd(**options, npu_models=("decider-0.8b",))
                with self.assertRaisesRegex(ValueError, "Select a catalogued"):
                    build_server_cmd(**options, npu_models=("missing-model",))
            entry, entries = small_npu(root, "decider-0.8b")
            options["npu_models"] = (entry["id"],)
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle), \
                 patched_npu(entries):
                with patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.npu_device_available",
                           return_value=False), self.assertRaisesRegex(ValueError, "accel0"):
                    build_server_cmd(**options)
                with patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.npu_device_available",
                           return_value=True), \
                     patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.xrt_mount_arguments",
                           side_effect=ValueError("No host XRT with its NPU plugin was found.")), \
                     self.assertRaisesRegex(ValueError, "No host XRT"):
                    build_server_cmd(**options)
                with patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.npu_device_available",
                           return_value=True), \
                     patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.xrt_mount_arguments",
                           return_value=[]), \
                     patch("ai_toolbox_cockpit.backends.halogen.runner.npu_host.fabric_clock_held",
                           return_value=False), \
                     self.assertRaisesRegex(ValueError, "fabric clock"):
                    build_server_cmd(**options)


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
            self.assertEqual(app.query_one("#halogen-download-model", SearchableSelect).value,
                             "qwen38-flash-next-v2")
            self.assertIn("qwen38-flash-next-v2.hgn", message)
            self.assertIn("qwen38-flash-next-ngram.hgn", message)
            self.assertNotIn("overlay.hgn", message)
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
            self.assertEqual(app.query_one("#halogen-model", SearchableSelect).value,
                             "qwen38-flash-next-v2")
            app.query_one("#halogen-server-dir", Input).value = str(directory)
            with patch.object(panel, "notify") as notify:
                panel.start_pressed()
                self.assertIn("Download or repair", notify.call_args.args[0])
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
                panel.start_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn("HALOGEN_CK_OVERLAY", message)
            self.assertIn("Host API relay: 127.0.0.1:8731", message)
            self.assertIn("--network=none", message)
            await pilot.click("#btn_no")
            await pilot.pause()
            suspended = []
            from contextlib import contextmanager
            @contextmanager
            def suspension():
                suspended.append(True)
                yield
                suspended.pop()
            def run(*args, **kwargs):
                self.assertEqual(suspended, [True])
                self.assertEqual(args[2], CONTAINER_NAME)
                self.assertEqual(kwargs["isolated_api"], ("127.0.0.1", 8731))
            with patch.object(app, "suspend", side_effect=suspension), \
                 patch("ai_toolbox_cockpit.backends.halogen.server.run_foreground_server", side_effect=run) as runner:
                panel._start_confirmed(True)
                runner.assert_called_once()

    async def test_halogen_selects_have_visible_labels(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 45)) as pilot:
            for tab, backend, ids in (
                ("tab-servers", "#server-backend-select", ("engine", "image", "model", "prompt-cache")),
                ("tab-models", "#model-backend-select", ("download-model", "npu-download-model")),
            ):
                app.query_one(TabbedContent).active = tab
                app.query_one(backend, SearchableSelect).value = "halogen"
                await pilot.pause()
                for control in ids:
                    label = app.query_one(f"#halogen-{control}-label", Label)
                    self.assertTrue(label.visible)
                    self.assertGreater(label.region.width, 0)

    async def test_npu_checkboxes_default_to_ready_models_and_flux_stays_off(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 45)) as pilot:
            app.query_one(TabbedContent).active = "tab-servers"
            app.query_one("#server-backend-select", SearchableSelect).value = "halogen"
            await pilot.pause()
            panel = app.query_one("#server-panel-halogen")
            with patch("ai_toolbox_cockpit.backends.halogen.server.incomplete_npu_files",
                       return_value=[("decider-0.8b", {"path": "decider-0.8b.hnpw", "size_bytes": 1})]):
                panel.refresh_model_inventory()
            for entry in load_npu_models():
                self.assertFalse(app.query_one(f"#{_npu_checkbox_id(entry['id'])}", Checkbox).value)
            with patch("ai_toolbox_cockpit.backends.halogen.server.incomplete_npu_files",
                       return_value=[]):
                panel.refresh_model_inventory()
            for entry in load_npu_models():
                self.assertEqual(app.query_one(f"#{_npu_checkbox_id(entry['id'])}", Checkbox).value,
                                 bool(entry.get("default_enabled")))
            save_backend_settings("halogen", {"npu_models": ["flux2-klein-4b"]})
            panel.refresh_model_inventory()
            self.assertTrue(app.query_one(f"#{_npu_checkbox_id('flux2-klein-4b')}", Checkbox).value)
            self.assertFalse(app.query_one(f"#{_npu_checkbox_id('decider-0.8b')}", Checkbox).value)

    async def test_npu_download_previews_shared_device_fetch(self):
        app = AiToolboxCockpitApp()
        directory = Path(self.temporary) / "npu models"
        async with app.run_test(size=(180, 45)) as pilot:
            panel = app.query_one("#model-panel-halogen")
            app.query_one(TabbedContent).active = "tab-models"
            app.query_one("#model-backend-select", SearchableSelect).value = "halogen"
            await pilot.pause()
            app.query_one("#halogen-models-dir", Input).value = str(directory)
            app.query_one("#halogen-npu-download-model", SearchableSelect).value = "qwen3-reranker-0.6b"
            panel.npu_download_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn("qwen3-reranker-0.6b", message)
            self.assertIn("npu/qwen3-embedding-0.6b", message)
            self.assertIn("--revision", message)
            self.assertFalse(directory.exists())
            await pilot.click("#btn_no")
            await pilot.pause()
            with patch.object(app, "suspend", return_value=nullcontext()), \
                 patch("ai_toolbox_cockpit.backends.halogen.models.subprocess.run") as run:
                panel._npu_download_confirmed(True)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[0].args[0][-1],
                                 str(directory / "npu" / "qwen3-reranker-0.6b"))
                self.assertEqual(run.call_args_list[1].args[0][-1],
                                 str(directory / "npu" / "qwen3-embedding-0.6b"))
        self.assertTrue(directory.is_dir())

    async def test_npu_selection_previews_device_and_suspends(self):
        app = AiToolboxCockpitApp()
        directory = Path(self.temporary) / "models"
        bundle = small_bundle(directory)
        entry, entries = small_npu(directory, "decider-0.8b")
        async with app.run_test(size=(180, 45)) as pilot:
            panel = app.query_one("#server-panel-halogen")
            app.query_one(TabbedContent).active = "tab-servers"
            app.query_one("#server-backend-select", SearchableSelect).value = "halogen"
            await pilot.pause()
            app.query_one("#halogen-server-dir", Input).value = str(directory)
            app.query_one(f"#{_npu_checkbox_id(entry['id'])}", Checkbox).value = True
            with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle), \
                 patched_npu(entries), npu_host_ready():
                panel.start_pressed()
            await pilot.pause()
            message = str(app.screen.query_one("#confirm_message", Label).render())
            self.assertIn("HALOGEN_NPU_MODELS=decider-0.8b", message)
            self.assertIn("/dev/accel/accel0", message)
            self.assertIn("NPU models on the Ryzen AI NPU: decider-0.8b", message)
            await pilot.click("#btn_no")
            await pilot.pause()
