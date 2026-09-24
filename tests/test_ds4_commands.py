import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.ds4.config import (
    get_artifact_role,
    get_model_server_defaults,
    is_tensor_parallel_cli_worker,
    normalize_distributed_transport,
    resolve_server_binary,
)
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

    def test_external_mtp_model_uses_current_ds4_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mtp = Path(directory) / "mtp.gguf"
            mtp.touch()
            command = self.build(directory, mtp_path=str(mtp))

        self.assertEqual(
            command[command.index("--mtp-model") + 1],
            "/models/mtp.gguf",
        )
        self.assertNotIn("--mtp", command)

    def test_embedded_mtp_and_glm53_vision_can_be_enabled_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "GLM-5.3-Flash-Q4K-Base-Q2-Experts-L03-28.gguf"
            vision = Path(directory) / "GLM-5.3-Flash-Vision-Encoder.gguf"
            model.touch()
            vision.touch()
            command = self.build(
                directory, model_path=str(model), mtp_enabled=True,
                vision_path=str(vision), custom_args="--rocm",
            )

        self.assertEqual(command[command.index("-m") + 1], f"/models/{model.name}")
        self.assertEqual(command.count("--mtp"), 1)
        self.assertNotIn("--mtp-model", command)
        self.assertEqual(command[command.index("--vision") + 1], f"/models/{vision.name}")
        self.assertIn("--rocm", command)

    def test_embedded_mtp_and_vision_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory)
        self.assertNotIn("--mtp", command)
        self.assertNotIn("--mtp-model", command)
        self.assertNotIn("--vision", command)

    def test_mxfp4_rocm_environment_is_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory)
        self.assertIn("DS4_ROCM_ENABLE_MXFP4_TILE4=1", command)
        self.assertIn("DS4_ROCM_MXFP4_DOWN_RGROUP=4", command)

    def test_mxfp4_rocm_environment_can_be_disabled_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tile4_disabled = self.build(directory, mxfp4_tile4_enabled=False)
            rgroup_disabled = self.build(directory, mxfp4_down_rgroup_enabled=False)
        self.assertNotIn("DS4_ROCM_ENABLE_MXFP4_TILE4=1", tile4_disabled)
        self.assertIn("DS4_ROCM_MXFP4_DOWN_RGROUP=4", tile4_disabled)
        self.assertIn("DS4_ROCM_ENABLE_MXFP4_TILE4=1", rgroup_disabled)
        self.assertNotIn("DS4_ROCM_MXFP4_DOWN_RGROUP=4", rgroup_disabled)

    def test_v41_decoder_swa_bounded_replay_environment_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            disabled = self.build(directory)
            enabled = self.build(
                directory, v41_decoder_swa_bounded_replay_enabled=True
            )

        environment = "DS4_ENABLE_V41_DECODER_SWA_BOUNDED_REPLAY=1"
        self.assertNotIn(environment, disabled)
        self.assertIn(environment, enabled)

    def test_glm53_vision_encoder_is_passed_as_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = Path(directory) / "GLM-5.3-Flash-Vision-Encoder.gguf"
            vision.touch()
            command = self.build(directory, vision_path=str(vision))

        self.assertEqual(
            command[command.index("--vision") + 1],
            "/models/GLM-5.3-Flash-Vision-Encoder.gguf",
        )
        self.assertEqual(get_artifact_role(str(vision)), "vision_encoder")

    def test_deepseek_vision_encoder_is_passed_as_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vision = Path(directory) / "DeepSeek-V4-Flash-Vision-Encoder.gguf"
            vision.touch()
            command = self.build(directory, vision_path=str(vision))

        self.assertEqual(
            command[command.index("--vision") + 1],
            "/models/DeepSeek-V4-Flash-Vision-Encoder.gguf",
        )
        self.assertEqual(get_artifact_role(str(vision)), "vision_encoder")

    def test_glm53_catalog_defaults_match_strix_halo_starting_points(self) -> None:
        q2 = get_model_server_defaults("GLM-5.3-Flash-Q2.gguf")
        mixed = get_model_server_defaults("GLM-5.3-Flash-Q4K-Base-Q2-Experts-L03-28.gguf")
        q4 = get_model_server_defaults("GLM-5.3-Flash-Q4_K.gguf")

        self.assertEqual(q2["standalone_ctx"], 262144)
        self.assertFalse(q2.get("ssd_streaming", False))
        self.assertNotIn("ssd_experts", q2)
        self.assertEqual(mixed["standalone_ctx"], 262144)
        self.assertFalse(mixed.get("ssd_streaming", False))
        self.assertEqual(q4["standalone_ctx"], 4096)
        self.assertFalse(q4.get("ssd_streaming", False))

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

    def test_deepseek_v41_flash_q2_defaults_match_strix_halo_configuration(self) -> None:
        defaults = get_model_server_defaults("DeepSeek-V4.1-Flash-Q2.gguf")

        self.assertEqual(defaults["standalone_ctx"], 262144)
        self.assertEqual(defaults["distributed_ctx"], 262144)
        self.assertTrue(defaults["ssd_streaming"])
        self.assertEqual(defaults["ssd_experts"], "92GB")
        self.assertTrue(defaults["tensor_parallel"])
        self.assertEqual(defaults["distributed_transport"], "tcp")
        self.assertEqual(defaults["distributed_port"], 9911)
        self.assertEqual(defaults["rdma_device"], "rocep194s0")
        self.assertEqual(defaults["rdma_port"], 1)
        self.assertEqual(defaults["rdma_gid_index"], 1)

    def test_deepseek_v41_flash_q2_standalone_ssd_streaming_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, ctx=262144, ssd_enabled=True, ssd_experts="92GB")

        self.assertEqual(command[command.index("--ctx") + 1], "262144")
        self.assertIn("--ssd-streaming", command)
        self.assertEqual(command[command.index("--ssd-streaming-cache-experts") + 1], "92GB")
        self.assertNotIn("--tensor-parallel", command)
        self.assertNotIn("--transport", command)

    def test_deepseek_v41_flash_q2_standalone_resident_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, ctx=262144)

        self.assertNotIn("--ssd-streaming", command)
        self.assertNotIn("--ssd-streaming-cache-experts", command)

    def test_deepseek_v41_flash_q2_tensor_parallel_tcp_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.build(
                directory,
                role="Coordinator",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
                peer_default_port="9911",
            )
            worker = self.build(
                directory,
                role="Worker",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
                peer_default_port="9911",
            )

        self.assertIn("--tensor-parallel", coordinator)
        self.assertEqual(coordinator[coordinator.index("--role") + 1], "coordinator")
        self.assertEqual(
            coordinator[coordinator.index("--listen") + 1:coordinator.index("--listen") + 3],
            ["192.168.100.1", "9911"],
        )
        self.assertEqual(coordinator[coordinator.index("--transport") + 1], "tcp")
        self.assertNotIn("--layers", coordinator)
        self.assertNotIn("--ssd-streaming", coordinator)
        self.assertEqual(worker[worker.index("--role") + 1], "worker")
        self.assertEqual(
            worker[worker.index("--coordinator") + 1:worker.index("--coordinator") + 3],
            ["192.168.100.1", "9911"],
        )
        self.assertEqual(worker[worker.index("--transport") + 1], "tcp")

    def test_deepseek_v41_flash_q2_tensor_parallel_roce_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.build(
                directory,
                role="Coordinator",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="rdma",
                rdma_device="rocep194s0",
                rdma_port="1",
                rdma_gid_index="1",
                peer_default_port="9911",
            )

        self.assertEqual(coordinator[coordinator.index("--transport") + 1], "rdma")
        self.assertEqual(coordinator[coordinator.index("--rdma-device") + 1], "rocep194s0")
        self.assertEqual(coordinator[coordinator.index("--rdma-port") + 1], "1")
        self.assertEqual(coordinator[coordinator.index("--rdma-gid-index") + 1], "1")

    def test_infiniband_transport_uses_native_gid_and_generic_rdma_cli_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "DeepSeek-V4.1-Flash-Q2.gguf"
            model.touch()
            common = {
                "model_path": str(model),
                "tensor_parallel": True,
                "transport": "infiniband",
                "rdma_device": "testhca0",
                "rdma_port": "2",
                "rdma_gid_index": "0",
                "peer_default_port": "9911",
            }
            coordinator = self.build(
                directory,
                role="Coordinator",
                peer_addr="192.0.2.10",
                **common,
            )
            worker = self.build(
                directory,
                role="Worker",
                peer_addr="192.0.2.10",
                **common,
            )

        for command in (coordinator, worker):
            self.assertIn("--network=host", command)
            self.assertEqual(command[command.index("--transport") + 1], "rdma")
            self.assertEqual(command[command.index("--rdma-device") + 1], "testhca0")
            self.assertEqual(command[command.index("--rdma-port") + 1], "2")
            self.assertEqual(command[command.index("--rdma-gid-index") + 1], "0")
        self.assertEqual(self.binary(worker), "ds4")
        self.assertNotIn("--host", worker)
        self.assertNotIn("--port", worker)

    def test_distributed_fabric_choices_map_to_ds4_wire_transports(self) -> None:
        self.assertEqual(normalize_distributed_transport("tcp"), "tcp")
        self.assertEqual(normalize_distributed_transport("roce"), "rdma")
        self.assertEqual(normalize_distributed_transport("infiniband"), "rdma")
        self.assertEqual(normalize_distributed_transport("rdma"), "rdma")
        with self.assertRaises(ValueError):
            normalize_distributed_transport("arbitrary")

    def test_tensor_parallel_omits_layer_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, role="Coordinator", layers="0:21", tensor_parallel=True)

        self.assertIn("--tensor-parallel", command)
        self.assertNotIn("--layers", command)

    def test_rdma_transport_requires_a_verbs_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for transport in ("rdma", "roce", "infiniband"):
                with self.subTest(transport=transport), self.assertRaises(ValueError):
                    self.build(
                        directory,
                        role="Coordinator",
                        tensor_parallel=True,
                        transport=transport,
                    )

    def binary(self, command: list[str]) -> str:
        return command[command.index("docker.io/example/ds4:latest") + 1]

    def test_deepseek_v41_flash_tensor_parallel_worker_uses_the_ds4_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "DeepSeek-V4.1-Flash-Q2.gguf"
            model.touch()
            worker = self.build(
                directory,
                model_path=str(model),
                role="Worker",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
            )
            coordinator = self.build(
                directory,
                model_path=str(model),
                role="Coordinator",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
            )
            standalone = self.build(directory, model_path=str(model))

        self.assertEqual(self.binary(worker), "ds4")
        self.assertEqual(self.binary(coordinator), "ds4-server")
        self.assertEqual(self.binary(standalone), "ds4-server")

    def test_ds4_cli_worker_is_not_given_http_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "DeepSeek-V4.1-Flash-Q2.gguf"
            model.touch()
            worker = self.build(
                directory,
                model_path=str(model),
                role="Worker",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
            )
            coordinator = self.build(
                directory,
                model_path=str(model),
                role="Coordinator",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
            )

        self.assertNotIn("--host", worker)
        self.assertNotIn("--port", worker)
        self.assertEqual(coordinator[coordinator.index("--host") + 1], "0.0.0.0")
        self.assertEqual(coordinator[coordinator.index("--port") + 1], "8000")

    def test_ds4_cli_worker_matches_the_documented_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "DeepSeek-V4.1-Flash-Q2.gguf"
            vision = Path(directory) / "DeepSeek-V4.1-Flash-Vision.gguf"
            model.touch()
            vision.touch()
            worker = self.build(
                directory,
                model_path=str(model),
                ctx=262144,
                role="Worker",
                peer_addr="192.168.100.1",
                tensor_parallel=True,
                transport="tcp",
                vision_path=str(vision),
                peer_default_port="9911",
            )

        self.assertEqual(
            worker[worker.index("ds4"):],
            [
                "ds4",
                "-m", "/models/DeepSeek-V4.1-Flash-Q2.gguf",
                "--ctx", "262144",
                "--vision", "/models/DeepSeek-V4.1-Flash-Vision.gguf",
                "--tensor-parallel",
                "--role", "worker",
                "--coordinator", "192.168.100.1", "9911",
                "--transport", "tcp",
            ],
        )

    def devices(self, command: list[str]) -> list[str]:
        return [
            command[index + 1]
            for index, token in enumerate(command)
            if token == "--device"
        ]

    def test_infiniband_devices_reach_the_podman_ds4_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as infiniband:
            for node in ("issm0", "rdma_cm", "uverbs0"):
                (Path(infiniband) / node).touch()
            model = Path(directory) / "DeepSeek-V4.1-Flash-Q2.gguf"
            model.touch()
            worker = self.build(
                directory,
                model_path=str(model),
                role="Worker",
                tensor_parallel=True,
                transport="rdma",
                rdma_device="rocep194s0",
                rdma_path=infiniband,
            )

        self.assertIn(infiniband, self.devices(worker))
        self.assertIn("memlock=-1", worker)
        self.assertEqual(worker.count("--group-add"), 1)
        self.assertEqual(worker[worker.index("--group-add") + 1], "keep-groups")

    def test_infiniband_device_nodes_reach_the_docker_ds4_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as infiniband:
            for node in ("rdma_cm", "uverbs0"):
                (Path(infiniband) / node).touch()
            model = Path(directory) / "DeepSeek-V4.1-Flash-Q2.gguf"
            model.touch()
            worker = self.build(
                directory,
                engine="docker",
                model_path=str(model),
                role="Worker",
                tensor_parallel=True,
                transport="rdma",
                rdma_device="rocep194s0",
                rdma_path=infiniband,
            )

        self.assertIn(f"{infiniband}/rdma_cm", self.devices(worker))
        self.assertIn(f"{infiniband}/uverbs0", self.devices(worker))
        self.assertIn("memlock=-1", worker)

    def test_hosts_without_infiniband_get_no_rdma_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, rdma_path=str(Path(directory) / "absent"))

        self.assertNotIn("memlock=-1", command)
        self.assertNotIn("rdma", self.devices(command))

    def test_is_tensor_parallel_cli_worker_is_family_scoped(self) -> None:
        self.assertTrue(
            is_tensor_parallel_cli_worker("DeepSeek-V4.1-Flash-Q2.gguf", "Worker", True)
        )
        self.assertFalse(
            is_tensor_parallel_cli_worker("DeepSeek-V4.1-Flash-Q2.gguf", "Coordinator", True)
        )
        self.assertFalse(is_tensor_parallel_cli_worker("DeepSeek-V4.1-Flash-Q2.gguf", "Worker", False))
        self.assertFalse(is_tensor_parallel_cli_worker("GLM-5.3-Flash-Q2.gguf", "Worker", True))

    def test_tensor_parallel_worker_binary_override_is_family_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "DeepSeek-V4-Flash-Q4KExperts-F16HC.gguf"
            other.touch()
            unknown = Path(directory) / "untracked-model.gguf"
            unknown.touch()
            curated_worker = self.build(
                directory, model_path=str(other), role="Worker", tensor_parallel=True
            )
            unknown_worker = self.build(
                directory, model_path=str(unknown), role="Worker", tensor_parallel=True
            )

        self.assertEqual(self.binary(curated_worker), "ds4-server")
        self.assertEqual(self.binary(unknown_worker), "ds4-server")

    def test_resolve_server_binary_only_overrides_tensor_parallel_workers(self) -> None:
        model = "DeepSeek-V4.1-Flash-Q2.gguf"

        self.assertEqual(resolve_server_binary(model, "Worker", True), "ds4")
        self.assertEqual(resolve_server_binary(model, "worker", True, "ds4-server"), "ds4")
        self.assertEqual(resolve_server_binary(model, "Coordinator", True), "ds4-server")
        self.assertEqual(resolve_server_binary(model, "Worker", False), "ds4-server")
        self.assertEqual(resolve_server_binary(model, "Standalone", True), "ds4-server")
        self.assertEqual(resolve_server_binary(model, "Worker", True, "ds4-custom"), "ds4")
        self.assertEqual(resolve_server_binary("GLM-5.3-Flash-Q2.gguf", "Worker", True), "ds4-server")

    def test_builder_never_emits_rocm_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, ctx=262144, ssd_enabled=True, ssd_experts="92GB")

        self.assertNotIn("--rocm", command)


if __name__ == "__main__":
    unittest.main()
