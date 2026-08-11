import tempfile
import unittest
from pathlib import Path

from ai_toolbox_cockpit.backends.llama_cpp.ubatch_profiles import (
    backend_from_name,
    get_calibrated_ubatch,
    model_key,
    save_profile,
)


class UbatchProfilesTest(unittest.TestCase):
    def test_model_key_normalizes_multipart_shard(self):
        self.assertEqual(
            model_key("/models/DeepSeek-UD-IQ2_XXS-00001-of-00003.gguf"),
            "DeepSeek-UD-IQ2_XXS",
        )

    def test_round_trip_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiles.json"
            model = "/models/Qwen3.6-27B-UD-Q8_K_XL.gguf"
            save_profile(
                "strix-halo",
                "rocm",
                model,
                {"selected_ubatch": 256, "candidates": []},
                path,
            )
            self.assertEqual(
                get_calibrated_ubatch(model, "strix-halo", "rocm", path),
                256,
            )
            self.assertIsNone(
                get_calibrated_ubatch(model, "strix-halo", "vulkan", path)
            )

            save_profile(
                "strix-halo",
                "rocm",
                model,
                {"selected_ubatch": None, "candidates": []},
                path,
            )
            self.assertEqual(
                get_calibrated_ubatch(model, "strix-halo", "rocm", path),
                256,
            )

    def test_backend_family(self):
        self.assertEqual(backend_from_name("llama-vulkan-radv"), "vulkan")
        self.assertEqual(backend_from_name("llama-rocm-7.14"), "rocm")
        self.assertEqual(backend_from_name("llama-therock-nightly"), "rocm")


if __name__ == "__main__":
    unittest.main()
