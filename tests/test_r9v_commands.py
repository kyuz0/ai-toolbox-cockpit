import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.r9v.model_manager import (
    PLE_FILENAME,
    build_prepare_ple_cmd,
    get_download_cmd,
    verify_package,
    verify_ple,
)
from ai_toolbox_cockpit.backends.r9v.preflight import inspect_host
from ai_toolbox_cockpit.backends.r9v.runner import (
    CONTAINER_PLE,
    R9V_ENV,
    build_runtime_probe_cmd,
    build_server_cmd,
)


class R9vCommandTests(unittest.TestCase):
    def test_server_command_preserves_fixed_dual_r9700_profile(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("ai_toolbox_cockpit.backends.r9v.runner.grp.getgrnam") as getgrnam,
        ):
            getgrnam.side_effect = lambda name: type("Group", (), {"gr_gid": {"video": 39, "render": 105}[name]})()
            root = Path(temporary)
            command = build_server_cmd(
                engine="docker",
                image="example.test/r9v:rocm-10.0",
                engine_args=[
                    "--device", "/dev/dri",
                    "--device", "/dev/kfd",
                    "--group-add", "video",
                    "--group-add", "render",
                    "--ipc=host",
                    "--security-opt", "seccomp=unconfined",
                ],
                model_dir=root / "model",
                ple_path=root / PLE_FILENAME,
                cache_dir=root / "cache",
            )

        self.assertEqual(command[:7], [
            "docker", "run", "--rm", "-it", "--name",
            "ai-toolbox-cockpit-r9v-server", "--device",
        ])
        self.assertIn("105", command)
        self.assertIn("39", command)
        self.assertIn("127.0.0.1:8004:8000", command)
        self.assertIn(f"GGUF_PLE_MMAP_PATH={CONTAINER_PLE}", command)
        self.assertIn("HIP_VISIBLE_DEVICES=0,1", command)
        self.assertIn("RADIANCE_USE_R4D=0", command)
        self.assertIn("VLLM_PLE_RESIDENCY_MODE=ssd", command)
        self.assertIn("QWEN38_TIERED_EXPERT_CACHE_RANKS=1", command)
        self.assertIn("--tensor-parallel-size", command)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--max-model-len") + 1], "131072")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertEqual(command[command.index("--cpu-offload-gb") + 1], "112.5")
        self.assertEqual(R9V_ENV["VLLM_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES"], "28800138240")
        image_index = command.index("example.test/r9v:rocm-10.0")
        self.assertEqual(command[image_index + 1:image_index + 3], ["vllm", "serve"])

    def test_server_command_rejects_non_dual_or_repeated_devices(self) -> None:
        common = {
            "engine": "podman",
            "image": "example.test/r9v:rocm-10.0",
            "engine_args": [],
            "model_dir": Path("/models"),
            "ple_path": Path("/ple.bin"),
            "cache_dir": Path("/cache"),
        }
        for devices in ("0", "0,1,2", "0,0", "gpu0,gpu1"):
            with self.subTest(devices=devices):
                with self.assertRaises(ValueError):
                    build_server_cmd(**common, visible_devices=devices)

    def test_runtime_probe_uses_same_two_devices_and_rocm_runtime(self) -> None:
        command = build_runtime_probe_cmd(
            engine="podman",
            image="example.test/r9v:rocm-10.0",
            engine_args=["--device", "/dev/kfd", "--group-add", "render"],
            visible_devices="1,0",
        )

        self.assertIn("--group-add", command)
        self.assertIn("keep-groups", command)
        self.assertNotIn("render", command)
        self.assertIn("HIP_VISIBLE_DEVICES=1,0", command)
        self.assertIn("ROCR_VISIBLE_DEVICES=1,0", command)
        probe = command[-1]
        self.assertIn("amdrocm-base10.0", probe)
        self.assertIn("torch.version.hip", probe)
        self.assertIn("torch.cuda.device_count() == 2", probe)
        self.assertIn("gfx1201", probe)

    def test_host_preflight_accepts_dual_gfx1201_and_warns_below_reference_ram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sys_root = root / "sys"
            proc_root = root / "proc"
            dev_root = root / "dev"
            for node, location in ((1, 256), (2, 512)):
                properties = sys_root / f"class/kfd/kfd/topology/nodes/{node}/properties"
                properties.parent.mkdir(parents=True)
                properties.write_text(
                    f"location_id {location}\ngfx_target_version 120001\n",
                    encoding="utf-8",
                )
            proc_root.mkdir()
            proc_root.joinpath("meminfo").write_text(
                "MemTotal:       65536000 kB\nMemAvailable:   50000000 kB\n",
                encoding="utf-8",
            )
            dev_root.mkdir()
            (dev_root / "kfd").touch()
            (dev_root / "dri").mkdir()
            ple = root / "ple.bin"
            ple.touch()
            completed = [
                type("Result", (), {"returncode": 0, "stdout": "/dev/nvme0n1p1\n"})(),
                type("Result", (), {"returncode": 0, "stdout": "disk 0 nvme\n"})(),
            ]
            with patch(
                "ai_toolbox_cockpit.backends.r9v.preflight.subprocess.run",
                side_effect=completed,
            ):
                report = inspect_host(
                    "0,1",
                    ple,
                    sys_root=sys_root,
                    proc_root=proc_root,
                    dev_root=dev_root,
                )

        self.assertTrue(report.ok)
        self.assertTrue(any("two gfx1201" in item for item in report.details))
        self.assertTrue(any("below R9V's qualified" in item for item in report.warnings))

    def test_download_is_revision_pinned_and_ple_command_uses_upstream_tool(self) -> None:
        model = {
            "repo": "owner/model",
            "revision": "a" * 40,
            "directory": "package",
            "ple": {"source_shards": ["target/one.gguf", "target/two.gguf"]},
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ai_toolbox_cockpit.backends.r9v.model_manager.get_models_dir",
                return_value=Path(temporary),
            ),
        ):
            download = get_download_cmd(model)
            prepare = build_prepare_ple_cmd("podman", "example.test/r9v:rocm-10.0", model)
            self.assertFalse((Path(temporary) / "package" / "derived").exists())

        self.assertEqual(download[1:5], [
            "download", "owner/model", "--revision", "a" * 40,
        ])
        self.assertEqual(download[-2], "--local-dir")
        self.assertTrue(download[-1].endswith("/package"))
        self.assertIn("/usr/local/libexec/r9v/prepare_ple.py", prepare)
        self.assertIn("/models/target/one.gguf", prepare)
        self.assertIn("--network", prepare)
        self.assertIn("none", prepare)

    def test_package_and_ple_hash_verification(self) -> None:
        package_content = b"package artifact"
        ple_content = b"derived ple"
        model = {
            "directory": "package",
            "artifacts": [{
                "path": "target/model.gguf",
                "bytes": len(package_content),
                "sha256": hashlib.sha256(package_content).hexdigest(),
            }],
            "ple": {
                "bytes": len(ple_content),
                "sha256": hashlib.sha256(ple_content).hexdigest(),
            },
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ai_toolbox_cockpit.backends.r9v.model_manager.get_models_dir",
                return_value=Path(temporary),
            ),
        ):
            root = Path(temporary) / "package"
            (root / "target").mkdir(parents=True)
            (root / "target/model.gguf").write_bytes(package_content)
            (root / "derived").mkdir()
            (root / "derived" / PLE_FILENAME).write_bytes(ple_content)

            self.assertEqual(verify_package(model), [])
            self.assertIsNone(verify_ple(model))
            (root / "derived" / PLE_FILENAME).write_bytes(b"bad payload")
            self.assertEqual(verify_ple(model), f"hash mismatch: derived/{PLE_FILENAME}")


if __name__ == "__main__":
    unittest.main()
