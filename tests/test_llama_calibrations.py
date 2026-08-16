import unittest

from ai_toolbox_cockpit.backends.llama_cpp.config import (
    get_calibrated_ubatch_defaults,
    load_models,
)
from ai_toolbox_cockpit.catalog import load_model_catalog


class LlamaCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {model["id"]: model for model in load_models()}

    def test_catalog_contains_only_explicit_llama_cpp_calibrations(self) -> None:
        records = load_model_catalog().backends["llama_cpp"].config[
            "calibrated_ubatches"
        ]
        self.assertEqual(len(records), 34)
        self.assertTrue(all(record["source_job"] for record in records))

    def test_qwen38_q4_baseline_uses_backend_specific_calibration(self) -> None:
        model = self.models["llama-unsloth-qwen3-8-27b-gguf"]
        path = "/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"

        for toolbox_id in (
            "strix-halo-llama-rocm-7-14",
            "strix-halo-llama-vulkan-radv",
        ):
            self.assertEqual(
                get_calibrated_ubatch_defaults(
                    model, path, toolbox_id, "baseline", "default"
                ),
                {"batch_size": 2048, "ubatch_size": 256},
            )

    def test_calibration_requires_quant_serving_and_kv_identity(self) -> None:
        model = self.models["llama-unsloth-qwen3-8-27b-gguf"]
        path = "/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"

        self.assertEqual(
            get_calibrated_ubatch_defaults(
                model, path, "strix-halo-llama-rocm-7-14", "mtp-2", "default"
            ),
            {},
        )
        self.assertEqual(
            get_calibrated_ubatch_defaults(
                model,
                path.replace("UD-Q4_K_XL", "UD-Q8_K_XL"),
                "strix-halo-llama-rocm-7-14",
                "baseline",
                "default",
            ),
            {},
        )
        self.assertEqual(
            get_calibrated_ubatch_defaults(
                model, path, "strix-halo-llama-rocm-7-14", "baseline", "q8_0"
            ),
            {},
        )

    def test_qwen36_q8_preserves_rocm_and_radv_difference(self) -> None:
        model = self.models["llama-unsloth-qwen3-6-27b-mtp-gguf"]
        path = "/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q8_K_XL.gguf"

        rocm = get_calibrated_ubatch_defaults(
            model, path, "strix-halo-llama-rocm-7-14", "baseline", "default"
        )
        radv = get_calibrated_ubatch_defaults(
            model, path, "strix-halo-llama-vulkan-radv", "baseline", "default"
        )
        self.assertEqual(rocm["ubatch_size"], 2048)
        self.assertEqual(radv["ubatch_size"], 256)


if __name__ == "__main__":
    unittest.main()
