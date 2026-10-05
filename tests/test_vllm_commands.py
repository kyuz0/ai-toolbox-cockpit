import json
import tempfile
import unittest
from pathlib import Path

from ai_toolbox_cockpit.backends.vllm.runner import (
    VllmCachePaths,
    apply_toolbox_policy_overrides,
    build_server_cmd,
    default_cache_paths,
)
from ai_toolbox_cockpit.backends.vllm.server import validate_compiled_cache_roots
from ai_toolbox_cockpit.catalog import load_model_catalog


class VllmCommandTests(unittest.TestCase):
    def test_gb10_toolbox_policy_clears_rocm_settings(self) -> None:
        policy = {
            "valid_tp": [1, 2],
            "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
            "env": {"VLLM_ROCM_USE_AITER": "0"},
        }
        effective = apply_toolbox_policy_overrides(
            policy,
            {
                "policy_overrides": {
                    "valid_tp": [1],
                    "attention_backend": None,
                    "env": {},
                }
            },
        )
        self.assertEqual(effective["valid_tp"], [1])
        self.assertIsNone(effective["attention_backend"])
        self.assertEqual(effective["env"], {})

    def test_gb10_docker_command_uses_native_gpu_flag(self) -> None:
        command = self.build(
            "LiquidAI/LFM2.5-1.2B-Instruct",
            engine="docker",
            engine_args=[
                "--runtime",
                "/usr/bin/nvidia-container-runtime",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all",
            ],
            policy=apply_toolbox_policy_overrides(
                self.policies["LiquidAI/LFM2.5-1.2B-Instruct"],
                {
                    "policy_overrides": {
                        "valid_tp": [1],
                        "attention_backend": None,
                        "env": {},
                    }
                },
            ),
        )
        self.assertEqual(command[0], "docker")
        self.assertNotIn("/usr/bin/nvidia-container-runtime", command)
        self.assertIn("--gpus", command)
        self.assertNotIn("--attention-backend", command)
        self.assertFalse(
            any(value.startswith("VLLM_ROCM_") for value in command)
        )

    @classmethod
    def setUpClass(cls) -> None:
        entries = load_model_catalog().backends["vllm"].entries
        cls.policies = {entry["repo"]: dict(entry) for entry in entries}

    def build(self, repo: str, **overrides) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "engine": "podman",
                "image": "docker.io/kyuz0/vllm-therock-gfx1151:latest",
                "engine_args": ["--device", "/dev/kfd", "--group-add", "keep-groups"],
                "model_id": repo,
                "policy": self.policies[repo],
                "tensor_parallel": min(self.policies[repo].get("valid_tp", [1])),
                "cache_paths": VllmCachePaths(root / "hf", root / "vllm", root / "triton", root / "aiter"),
            }
            values.update(overrides)
            return build_server_cmd(**values)

    def test_pinned_repository_and_local_snapshot_are_distinct_launches(self) -> None:
        original = self.policies["openai/gpt-oss-20b"]
        policy = dict(original, revision="a" * 40)
        remote = self.build("openai/gpt-oss-20b", policy=policy)
        self.assertEqual(remote[remote.index("--revision") + 1], "a" * 40)
        local = self.build("openai/gpt-oss-20b", policy=policy, model_directory=Path("/tmp/prepared-model"))
        self.assertIn("/tmp/prepared-model:/models/target:ro", local)
        self.assertIn("/models/target", local)
        self.assertNotIn("--revision", local)
        with self.assertRaisesRegex(ValueError, "requires a local"):
            self.build("openai/gpt-oss-20b", policy=dict(original, requires_local_model=True))
        with self.assertRaisesRegex(ValueError, "colons"):
            self.build("openai/gpt-oss-20b", model_directory=Path("/tmp/bad:mount"))

    def test_qualified_dflash_profile_mounts_draft_and_applies_dispatch(self) -> None:
        from ai_toolbox_cockpit.catalog import load_toolbox_catalog
        toolbox = load_toolbox_catalog().toolboxes["r9700-ggz14-mxfp4-tp1"]
        base = self.policies["amd/Qwen3.8-27B-Quark-AWQ-MXFP4"]
        policy = apply_toolbox_policy_overrides(base, toolbox.backend_config)
        command = self.build(base["repo"], policy=policy, model_directory=Path("/tmp/target"),
                             speculation="dflash2", draft_directory=Path("/tmp/draft"))
        self.assertIn("/tmp/target:/models/target:ro", command)
        self.assertIn("/tmp/draft:/models/draft:ro", command)
        self.assertIn("RADIANCE_FAST_DRAFT=1", command)
        self.assertIn("RADIANCE_VERIFY_HEAD=1", command)
        config = json.loads(command[command.index("--speculative-config") + 1])
        self.assertEqual(config["model"], "/models/draft")
        self.assertEqual(config["method"], "dflash")
        self.assertEqual(config["num_speculative_tokens"], 7)
        self.assertEqual(command[command.index("--compilation-config.cudagraph_capture_sizes") + 1], "[1,2,4,8]")
        with self.assertRaisesRegex(ValueError, "one sequence"):
            self.build(base["repo"], policy=policy, model_directory=Path("/tmp/target"),
                       speculation="dflash2", draft_directory=Path("/tmp/draft"), max_num_seqs=4)
        with self.assertRaisesRegex(ValueError, "requires a local draft"):
            self.build(base["repo"], policy=policy, model_directory=Path("/tmp/target"), speculation="dflash2")
        with self.assertRaisesRegex(ValueError, "colons"):
            self.build(base["repo"], policy=policy, model_directory=Path("/tmp/target"),
                       speculation="dflash2", draft_directory=Path("/tmp/bad:mount"))
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.build("openai/gpt-oss-20b", speculation="dflash2", draft_directory=Path("/tmp/draft"))

    def test_dual_ggz14_dflash_uses_dual_dispatch_and_memory_budget(self) -> None:
        from ai_toolbox_cockpit.catalog import load_toolbox_catalog
        toolbox = load_toolbox_catalog().toolboxes["r9700-ggz14-mxfp4-tp2"]
        base = self.policies["amd/Qwen3.8-27B-Quark-AWQ-MXFP4"]
        policy = apply_toolbox_policy_overrides(base, toolbox.backend_config)
        command = self.build(base["repo"], policy=policy, tensor_parallel=2,
                             gpu_memory_utilization=policy["speculation"]["dflash2"]["gpu_memory_utilization"],
                             model_directory=Path("/tmp/target"), speculation="dflash2",
                             draft_directory=Path("/tmp/draft"))
        self.assertIn("HIP_VISIBLE_DEVICES=0,1", command)
        self.assertIn("RADIANCE_FP8_STREAM=1", command)
        self.assertIn("RADIANCE_FAST_DRAFT=1", command)
        self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.92")
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "8192")
        self.assertEqual(json.loads(command[command.index("--speculative-config") + 1])["num_speculative_tokens"], 7)

    def test_radiance_profiles_do_not_inherit_ggz14_speculation(self) -> None:
        from ai_toolbox_cockpit.catalog import load_toolbox_catalog
        catalog = load_toolbox_catalog()
        for toolbox_id, repo, mxfp4 in (
            ("r9700-radiance-fp8-tp2", "Qwen/Qwen3.8-27B-FP8", False),
            ("r9700-radiance-mxfp4-tp2", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4", True),
        ):
            with self.subTest(toolbox=toolbox_id):
                toolbox = catalog.toolboxes[toolbox_id]
                policy = apply_toolbox_policy_overrides(self.policies[repo], toolbox.backend_config)
                command = self.build(repo, image=toolbox.image, policy=policy,
                                     tensor_parallel=2, model_directory=Path("/tmp/target"))
                self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "fp8")
                self.assertEqual(command[command.index("--attention-backend") + 1], "R4D")
                self.assertIn("--no-async-scheduling", command)
                self.assertIn("RADIANCE_FP8_STREAM=0", command)
                self.assertIn("RADIANCE_FAST_DRAFT=0", command)
                self.assertIn("RADIANCE_MXFP4_W4A8=" + ("1" if mxfp4 else "0"), command)
                self.assertNotIn("--speculative-config", command)
                with self.assertRaisesRegex(ValueError, "does not support"):
                    self.build(repo, policy=policy, tensor_parallel=2,
                               model_directory=Path("/tmp/target"), speculation="dflash2",
                               draft_directory=Path("/tmp/draft"))

    def test_default_llama_policy_adds_tools_and_triton_attention(self) -> None:
        command = self.build("meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.assertEqual(command[command.index("--attention-backend") + 1], "TRITON_ATTN")
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "llama3_json")

    def test_selected_attention_backend_overrides_the_model_default(self) -> None:
        command = self.build(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attention_backend="ROCM_ATTN",
        )
        self.assertEqual(command[command.index("--attention-backend") + 1], "ROCM_ATTN")

    def test_fp8_policy_forces_eager_and_model_environment(self) -> None:
        command = self.build("RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic")
        self.assertIn("--enforce-eager", command)
        self.assertIn("VLLM_STRIX_FP8_TRITON=1", command)
        self.assertIn("VLLM_ROCM_USE_AITER=0", command)

    def test_deepseek_uses_model_specific_attention_and_validated_flags(self) -> None:
        command = self.build("deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertNotIn("--attention-backend", command)
        self.assertIn("VLLM_ROCM_USE_AITER=1", command)
        self.assertIn("VLLM_ROCM_USE_AITER_LINEAR=0", command)
        self.assertEqual(command[command.index("--max-model-len") + 1], "262144")
        self.assertEqual(command[command.index("--logprobs-mode") + 1], "processed_logprobs")

    def test_qwen_uses_unified_attention_without_broad_aiter(self) -> None:
        command = self.build("Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(command[command.index("--attention-backend") + 1], "ROCM_AITER_UNIFIED_ATTN")
        self.assertIn("VLLM_ROCM_USE_AITER=0", command)
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")

    def test_lfm_uses_native_repo_and_unified_attention(self) -> None:
        command = self.build("LiquidAI/LFM2.5-1.2B-Instruct")
        self.assertIn("LiquidAI/LFM2.5-1.2B-Instruct", command)
        self.assertFalse(any("LFM2.5-1.2B-Instruct-GGUF" in argument for argument in command))
        self.assertEqual(command[command.index("--attention-backend") + 1], "ROCM_AITER_UNIFIED_ATTN")
        self.assertNotIn("--tokenizer", command)
        self.assertNotIn("--hf-config-path", command)
        self.assertIn("VLLM_ROCM_USE_AITER=0", command)
        self.assertIn("VLLM_ROCM_USE_AITER_LINEAR=0", command)

    def test_muse_glimmer_uses_transformers_and_unified_attention(self) -> None:
        command = self.build("meta-models/Muse-Glimmer-30B")
        self.assertEqual(command[command.index("--attention-backend") + 1], "ROCM_AITER_UNIFIED_ATTN")
        self.assertEqual(command[command.index("--model-impl") + 1], "transformers")
        self.assertEqual(command[command.index("--max-model-len") + 1], "131072")
        self.assertIn("VLLM_ROCM_USE_AITER=0", command)
        self.assertIn("VLLM_ROCM_USE_AITER_LINEAR=0", command)

    def test_tp_policy_rejects_invalid_single_gpu_minimax(self) -> None:
        with self.assertRaises(ValueError):
            self.build("cyankiwi/MiniMax-M2.7-AWQ-4bit", tensor_parallel=1)

    def test_awq_policy_forces_eager_qwen_parsers(self) -> None:
        command = self.build("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        self.assertIn("--enforce-eager", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_coder")

    def test_gemma_policy_preserves_tool_and_reasoning_parsers(self) -> None:
        command = self.build("google/gemma-4-26B-A4B-it")
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "gemma4")
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "gemma4")

    def test_gpt_oss_policy_preserves_openai_parsers(self) -> None:
        command = self.build("openai/gpt-oss-20b")
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "openai")
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "openai_gptoss")

    def test_all_four_caches_are_persistent_mounts(self) -> None:
        command = self.build("openai/gpt-oss-20b")
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "-v"]
        self.assertEqual(len(mounts), 4)
        self.assertTrue(any(value.endswith(":/workspace/.cache/huggingface") for value in mounts))
        self.assertTrue(any(value.endswith(":/workspace/.cache/triton") for value in mounts))

    def test_triton_cache_is_host_persistent_for_podman_and_docker(self) -> None:
        expected_container_path = "/workspace/.cache/triton"
        expected_environment = f"TRITON_CACHE_DIR={expected_container_path}"
        expected_tilelang_environment = (
            f"TILELANG_CACHE_DIR={expected_container_path}/tilelang"
        )
        for engine in ("podman", "docker"):
            with self.subTest(engine=engine):
                command = self.build("openai/gpt-oss-20b", engine=engine)
                mounts = [
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "-v"
                ]
                self.assertIn(expected_environment, command)
                self.assertIn(expected_tilelang_environment, command)
                self.assertTrue(
                    any(value.endswith(f":{expected_container_path}") for value in mounts)
                )

    def test_default_triton_cache_matches_host_shared_toolbox_path(self) -> None:
        self.assertEqual(default_cache_paths().triton, Path.home() / ".cache" / "triton")

    def test_vllm_config_uses_writable_cache_and_usage_stats_are_disabled(self) -> None:
        command = self.build("openai/gpt-oss-20b")
        self.assertIn("VLLM_CONFIG_ROOT=/workspace/.cache/vllm/config", command)
        self.assertIn("TRITON_CACHE_DIR=/workspace/.cache/triton", command)
        self.assertIn(
            "TILELANG_CACHE_DIR=/workspace/.cache/triton/tilelang", command
        )
        self.assertIn("VLLM_NO_USAGE_STATS=1", command)
        self.assertIn("HOME=/workspace", command)

    def test_dtype_api_key_and_host_hf_token_are_forwarded(self) -> None:
        command = self.build(
            "openai/gpt-oss-20b",
            dtype="bfloat16",
            api_key="not-for-logs",
        )
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(command[command.index("--api-key") + 1], "not-for-logs")
        self.assertEqual(command[command.index("HF_TOKEN") - 1], "-e")

        authenticated = self.build("openai/gpt-oss-20b", hf_token="hf_example")
        self.assertIn("HF_TOKEN=hf_example", authenticated)

    def test_cache_reset_rejects_broad_or_mismatched_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = VllmCachePaths(
                root / "huggingface",
                root / "vllm",
                root / "triton",
                root / "aiter",
            )
            self.assertEqual(validate_compiled_cache_roots(safe)[0], (root / "vllm").resolve())
            unsafe = VllmCachePaths(root / "huggingface", Path.home(), root / "triton", root / "aiter")
            with self.assertRaisesRegex(ValueError, "unsafe vLLM cache root"):
                validate_compiled_cache_roots(unsafe)


if __name__ == "__main__":
    unittest.main()
