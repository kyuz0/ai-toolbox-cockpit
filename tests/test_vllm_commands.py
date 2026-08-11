import tempfile
import unittest
from pathlib import Path

from ai_toolbox_cockpit.backends.vllm.runner import VllmCachePaths, build_server_cmd
from ai_toolbox_cockpit.backends.vllm.server import validate_compiled_cache_roots
from ai_toolbox_cockpit.catalog import load_model_catalog


class VllmCommandTests(unittest.TestCase):
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

    def test_lfm_gguf_bf16_uses_external_config_and_unified_attention(self) -> None:
        command = self.build("LiquidAI/LFM2.5-1.2B-Instruct-GGUF:BF16")
        self.assertIn("LiquidAI/LFM2.5-1.2B-Instruct-GGUF:BF16", command)
        self.assertEqual(command[command.index("--attention-backend") + 1], "ROCM_AITER_UNIFIED_ATTN")
        self.assertEqual(command[command.index("--tokenizer") + 1], "LiquidAI/LFM2.5-1.2B-Instruct")
        self.assertEqual(command[command.index("--hf-config-path") + 1], "LiquidAI/LFM2.5-1.2B-Instruct")
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
        self.assertTrue(any(value.endswith(":/opt/triton_cache") for value in mounts))

    def test_dtype_api_key_and_host_hf_token_are_forwarded(self) -> None:
        command = self.build(
            "openai/gpt-oss-20b",
            dtype="bfloat16",
            api_key="not-for-logs",
        )
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(command[command.index("--api-key") + 1], "not-for-logs")
        self.assertEqual(command[command.index("HF_TOKEN") - 1], "-e")

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
