import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.ds4.config import get_model_server_defaults
from ai_toolbox_cockpit.backends.ds4.server_runner import build_server_cmd


class Ds4CommandTests(unittest.TestCase):
    def build(self, directory: str, **overrides) -> list[str]:
        model = Path(directory) / "model.gguf"
        model.touch()
        values = {
            "engine": "podman",
            "image": "docker.io/example/ds4:latest",
            "model_path": str(model),
            "ctx": 126000,
            "host": "localhost",
            "port": "8000",
            "kv_disk_enabled": False,
            "kv_disk_dir": "",
            "kv_disk_mb": 0,
            "prefill_chunk": None,
            "mtp_path": "",
            "custom_args": "",
            "role": "Standalone",
            "layers": "",
            "peer_addr": "",
            "toolbox_config": {"args": ["--device", "/dev/kfd"], "server_binary": "ds4-server"},
        }
        values.update(overrides)
        with patch("ai_toolbox_cockpit.backends.ds4.server_runner.get_models_dir", return_value=Path(directory)):
            return build_server_cmd(**values)

    def test_standalone_uses_ipc_ptrace_port_and_read_only_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory)
        self.assertIn("--ipc=host", command)
        self.assertIn("--cap-add=SYS_PTRACE", command)
        self.assertIn("127.0.0.1:8000:8000", command)
        self.assertIn(f"{directory}:/models:ro", command)
        self.assertNotIn("--network=host", command)

    def test_disk_kv_and_prefill_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(
                directory,
                kv_disk_enabled=True,
                kv_disk_dir="/tmp/ds4-kv-test",
                kv_disk_mb=8192,
                prefill_chunk=2048,
            )
        self.assertIn("/tmp/ds4-kv-test:/var/cache/ds4-kv", command)
        self.assertEqual(command[command.index("--kv-disk-space-mb") + 1], "8192")
        self.assertEqual(command[command.index("--prefill-chunk") + 1], "2048")

    def test_coordinator_uses_host_network_and_distributed_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(
                directory,
                role="Coordinator",
                layers="0:21",
                peer_addr="0.0.0.0:8081",
                dist_prefill_chunk=512,
                dist_prefill_window=2,
            )
        self.assertIn("--network=host", command)
        self.assertNotIn("-p", command)
        self.assertEqual(command[command.index("--role") + 1], "coordinator")
        self.assertEqual(command[command.index("--listen") + 1:command.index("--listen") + 3], ["0.0.0.0", "8081"])
        self.assertEqual(command[command.index("--dist-prefill-chunk") + 1], "512")

    def test_curated_hybrid_model_keeps_prefill_default(self) -> None:
        filename = "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
        self.assertEqual(get_model_server_defaults(filename)["prefill_chunk"], 2048)


if __name__ == "__main__":
    unittest.main()
