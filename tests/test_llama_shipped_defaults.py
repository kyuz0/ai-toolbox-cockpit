import unittest
from unittest.mock import patch

from ai_toolbox_cockpit.backends.llama_cpp.config import (
    get_default_inference_profile,
    get_platform,
    get_preferred_ubatch,
    load_models,
)


class ShippedDefaultsTest(unittest.TestCase):
    def assert_shipped_ubatches(
        self, model_path: str, *, rocm: int, vulkan: int
    ) -> None:
        with patch("ai_toolbox_cockpit.backends.llama_cpp.config.get_calibrated_ubatch", return_value=None):
            self.assertEqual(
                get_preferred_ubatch(model_path, "strix-halo", "rocm"), rocm
            )
            self.assertEqual(
                get_preferred_ubatch(model_path, "strix-halo", "vulkan"), vulkan
            )

    def test_deepseek_v4_flash_defaults(self):
        self.assert_shipped_ubatches(
            "/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_XXS/"
            "DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf",
            rocm=2048,
            vulkan=1024,
        )
        deepseek = next(
            model
            for model in load_models()
            if model["repo"] == "unsloth/DeepSeek-V4-Flash-0731-GGUF"
        )
        self.assertEqual(
            get_default_inference_profile(deepseek),
            "Thinking (Effort: High)",
        )
        self.assertIn(
            '"reasoning_effort":"high"',
            deepseek["inference_profiles"]["Thinking (Effort: High)"]["args"],
        )

    def test_qwen_36_27b_mtp_defaults(self):
        self.assert_shipped_ubatches(
            "/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf",
            rocm=1024,
            vulkan=256,
        )

    def test_qwen_36_35b_a3b_mtp_defaults(self):
        self.assert_shipped_ubatches(
            "/models/Qwen3.6-35B-A3B-MTP-GGUF/"
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            rocm=2048,
            vulkan=2048,
        )

    def test_local_calibration_overrides_shipped_default(self):
        with patch("ai_toolbox_cockpit.backends.llama_cpp.config.get_calibrated_ubatch", return_value=512):
            self.assertEqual(
                get_preferred_ubatch(
                    "/models/Qwen3.6-27B-MTP-GGUF/model.gguf",
                    "strix-halo",
                    "vulkan",
                ),
                512,
            )

    def test_experimental_vulkan_performance_image_is_fetchable(self):
        platform = get_platform("strix-halo")
        self.assertIsNotNone(platform)
        experimental = next(
            group
            for group in platform["groups"]
            if group["name"] == "Dev/Experimental Toolboxes"
        )
        toolbox = next(
            item
            for item in experimental["toolboxes"]
            if item["name"] == "llama-vulkan-radv-performance"
        )
        self.assertEqual(toolbox["tag"], "vulkan-radv-performance")
        self.assertEqual(
            f'{platform["registry"]}:{toolbox["tag"]}',
            "docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv-performance",
        )


if __name__ == "__main__":
    unittest.main()
