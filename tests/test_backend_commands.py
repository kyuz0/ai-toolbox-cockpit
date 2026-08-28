import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.comfyui.runner import ComfyPaths, build_server_cmd as build_comfy
from ai_toolbox_cockpit.backends.ds4.server_runner import build_server_cmd as build_ds4
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd as build_llama
from ai_toolbox_cockpit.backends.vllm.runner import VllmCachePaths, build_server_cmd as build_vllm


ROCM_ARGS = ["--device", "/dev/dri", "--device", "/dev/kfd"]


class BackendCommandTests(unittest.TestCase):
    def test_llama_preserves_load_mode_api_key_mtp_and_vision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "Qwen3.6-27B-MTP-GGUF" / "model.gguf"
            projector = model.parent / "mmproj-model.gguf"
            model.parent.mkdir()
            model.touch()
            projector.touch()
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=root,
            ):
                command = build_llama(
                    "podman", "docker.io/example/llama:latest", str(model), 65536,
                    True, True,
                    "--spec-type draft-mtp --spec-draft-n-max 2 -np 1",
                    platform_id="strix-halo", engine_args=ROCM_ARGS,
                    supports_load_mode=True, api_key="secret",
                    vision_projector_path=str(projector),
                )
        self.assertIn("--load-mode", command)
        self.assertIn("--api-key", command)
        self.assertIn("XDG_CACHE_HOME=/tmp", command)
        self.assertEqual(command[command.index("--mmproj") + 1], "/models/Qwen3.6-27B-MTP-GGUF/mmproj-model.gguf")
        self.assertIn("draft-mtp", command)

    def test_ds4_preserves_disk_kv_ssd_and_distributed_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.gguf"
            model.touch()
            with patch(
                "ai_toolbox_cockpit.backends.ds4.server_runner.get_models_dir",
                return_value=root,
            ):
                command = build_ds4(
                    "podman", "docker.io/example/ds4:latest", str(model), 126000,
                    "localhost", "8080", True, str(root / "kv"), 65536, 2048,
                    "", "", "Coordinator", "0:21", "10.0.0.1:8081",
                    {"args": ROCM_ARGS, "server_binary": "ds4-server"},
                    True, "8", "0:4", True, 512, 2,
                )
        for option in ("--kv-disk-dir", "--ssd-streaming", "--role", "--listen", "--dist-prefill-chunk"):
            self.assertIn(option, command)

    def test_ds4_builds_optimized_dspark_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "DeepSeek-V4-Flash-target.gguf"
            support = root / "DeepSeek-V4-Flash-DSpark-support-0731.gguf"
            model.touch()
            support.touch()
            with patch(
                "ai_toolbox_cockpit.backends.ds4.server_runner.get_models_dir",
                return_value=root,
            ):
                command = build_ds4(
                    "podman", "docker.io/example/ds4:latest", str(model), 126000,
                    "localhost", "8080", False, "", 0, None,
                    "", "", "Standalone", "", "",
                    {"args": ROCM_ARGS, "server_binary": "ds4-server"},
                    dspark_enabled=True,
                    dspark_path=str(support),
                    dspark_confidence=0.0,
                )

        self.assertEqual(command[command.index("--mtp") + 1], "/models/DeepSeek-V4-Flash-DSpark-support-0731.gguf")
        self.assertIn("--dspark", command)
        self.assertEqual(command[command.index("--dspark-confidence") + 1], "0")
        self.assertNotIn("--mtp-draft", command)

    def test_ds4_rejects_dspark_with_ssd_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "target.gguf"
            support = root / "support.gguf"
            model.touch()
            support.touch()
            with (
                patch(
                    "ai_toolbox_cockpit.backends.ds4.server_runner.get_models_dir",
                    return_value=root,
                ),
                self.assertRaisesRegex(ValueError, "SSD streaming"),
            ):
                build_ds4(
                    "podman", "docker.io/example/ds4:latest", str(model), 126000,
                    "localhost", "8080", False, "", 0, None,
                    "", "", "Standalone", "", "",
                    {"args": ROCM_ARGS, "server_binary": "ds4-server"},
                    ssd_enabled=True,
                    dspark_enabled=True,
                    dspark_path=str(support),
                )

    def test_vllm_applies_deepseek_locked_attention_policy(self) -> None:
        policy = {
            "trust_remote": True,
            "valid_tp": [1],
            "enforce_eager": True,
            "attention_backend": None,
            "env": {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_LINEAR": "0"},
            "extra_flags": ["--kv-cache-dtype", "fp8", "--logprobs-mode", "processed_logprobs"],
        }
        command = build_vllm(
            engine="podman", image="docker.io/example/vllm:latest", engine_args=ROCM_ARGS,
            model_id="deepseek-ai/DeepSeek-V4-Flash-0731", policy=policy,
            cache_paths=VllmCachePaths(
                Path("~/hf"), Path("~/vllm"), Path("~/triton"), Path("~/aiter")
            ),
            host="localhost", port=8000, tensor_parallel=1, max_num_seqs=1,
            max_model_len="262144", gpu_memory_utilization=0.9, dtype="auto",
            attention_backend=None, enforce_eager=True, api_key="", extra_args="",
        )
        self.assertNotIn("--attention-backend", command)
        self.assertIn("VLLM_ROCM_USE_AITER=1", command)
        self.assertIn("--enforce-eager", command)
        self.assertIn("processed_logprobs", command)

    def test_vllm_rejects_unvalidated_tensor_parallel_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Tensor parallel"):
            build_vllm(
                engine="docker", image="docker.io/example/vllm:latest", engine_args=[],
                model_id="example/model", policy={"valid_tp": [1]},
                cache_paths=VllmCachePaths(
                    Path("~/hf"), Path("~/vllm"), Path("~/triton"), Path("~/aiter")
                ),
                host="localhost", port=8000, tensor_parallel=2, max_num_seqs=1,
                max_model_len="8192", gpu_memory_utilization=0.9, dtype="auto",
                attention_backend="TRITON_ATTN", enforce_eager=False, api_key="", extra_args="",
            )

    def test_comfy_matches_toolbox_defaults_and_persistent_mounts(self) -> None:
        command = build_comfy(
            engine="podman", image="docker.io/example/comfy:latest", engine_args=ROCM_ARGS,
            paths=ComfyPaths(
                Path("~/comfy-models"), Path("~/comfy-inputs"),
                Path("~/comfy-outputs"), Path("~/comfy-user")
            ),
            host="localhost", port=8000, bf16_vae=True, gpu_only=True,
            disable_mmap=True, disable_smart_memory=True, cache_none=True, extra_args="",
        )
        for option in ("--bf16-vae", "--gpu-only", "--disable-mmap", "--disable-smart-memory", "--cache-none"):
            self.assertIn(option, command)
        mounts = " ".join(command)
        self.assertIn(":/opt/ComfyUI/models", mounts)
        self.assertIn(":/workspace/comfy-outputs", mounts)


if __name__ == "__main__":
    unittest.main()
