import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.llama_cpp.config import get_vision_projector_config, load_models
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import get_local_vision_projectors, scan_local_models
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd


class VisionProjectorsTest(unittest.TestCase):
    def test_catalog_marks_muse_and_all_curated_qwen_entries_as_vision_models(self):
        configs = {model["repo"]: model for model in load_models()}
        expected_repos = {
            "unsloth/Muse-Glimmer-30B-GGUF",
            "unsloth/Qwen3.5-122B-A10B-GGUF",
            "unsloth/Qwen3.5-122B-A10B-MTP-GGUF",
            "unsloth/Qwen3.6-27B-GGUF",
            "unsloth/Qwen3.8-27B-GGUF",
            "kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF",
            "cafonez/Qwen3.8-27B-ROCmI4-MTP-GGUF",
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        }

        for repo in expected_repos:
            self.assertEqual(
                get_vision_projector_config(configs[repo])["patterns"],
                ["mmproj-*.gguf"],
            )

    def test_projectors_are_discovered_beside_the_model_but_not_listed_as_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            model_dir = models_dir / "Muse-Glimmer-30B-GGUF"
            model_dir.mkdir()
            model = model_dir / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
            projector = model_dir / "mmproj-Muse-Glimmer-30B-BF16.gguf"
            model.touch()
            projector.touch()

            self.assertEqual(
                get_local_vision_projectors(str(model), ["mmproj-*.gguf"]),
                [projector],
            )
            with patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=models_dir):
                discovered = scan_local_models()

            self.assertEqual([entry["path"] for entry in discovered], [str(model)])

    def test_server_adds_only_the_explicitly_selected_vision_projector(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "Muse-Glimmer-30B-GGUF"
            model_dir.mkdir()
            model = model_dir / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
            projector = model_dir / "mmproj-Muse-Glimmer-30B-BF16.gguf"
            model.touch()
            projector.touch()
            kwargs = dict(
                engine="podman",
                image="docker.io/example/llama-rocm:latest",
                model_path=str(model),
                context_size=65536,
                use_fa=True,
                use_no_mmap=True,
                platform_id="strix-halo",
                engine_args=[],
                custom_args="",
            )
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=Path(temporary),
            ):
                text_only = build_server_cmd(**kwargs)
                vision = build_server_cmd(
                    vision_projector_path=str(projector), **kwargs
                )

            self.assertNotIn("--mmproj", text_only)
            projector_index = vision.index("--mmproj")
            self.assertEqual(
                vision[projector_index + 1],
                "/models/Muse-Glimmer-30B-GGUF/mmproj-Muse-Glimmer-30B-BF16.gguf",
            )


if __name__ == "__main__":
    unittest.main()
