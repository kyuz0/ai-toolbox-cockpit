import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.llama_cpp.config import get_dspark_config, load_models
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
    get_local_dspark_models,
    scan_local_models,
)
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd


class LlamaDsparkTests(unittest.TestCase):
    def test_deepseek_0731_uses_official_unsloth_dspark_defaults(self):
        configs = {model["repo"]: model for model in load_models()}
        dspark = get_dspark_config(configs["unsloth/DeepSeek-V4-Flash-0731-GGUF"])

        self.assertEqual(
            dspark["default_pattern"],
            "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
        )
        self.assertEqual(dspark["default_draft_n"], 3)
        self.assertEqual(dspark["default_ngl"], 99)
        self.assertEqual(dspark["fit"], "off")

    def test_dspark_drafters_are_discovered_but_not_listed_as_main_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            repo_dir = models_dir / "DeepSeek-V4-Flash-0731-GGUF"
            quant_dir = repo_dir / "UD-IQ3_XXS"
            dspark_dir = repo_dir / "dspark"
            quant_dir.mkdir(parents=True)
            dspark_dir.mkdir()
            model = quant_dir / "DeepSeek-V4-Flash-0731-UD-IQ3_XXS.gguf"
            q8_drafter = repo_dir / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
            bf16_drafter = dspark_dir / "dspark-DeepSeek-V4-Flash-0731-BF16.gguf"
            for path in (model, q8_drafter, bf16_drafter):
                path.touch()

            patterns = [
                "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
                "dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf",
            ]
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                drafters = get_local_dspark_models(patterns, patterns[0])
                discovered = scan_local_models()

            self.assertEqual(drafters, [q8_drafter, bf16_drafter])
            self.assertEqual([entry["path"] for entry in discovered], [str(model)])

    def test_server_mounts_selected_dspark_drafter(self):
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            repo_dir = models_dir / "DeepSeek-V4-Flash-0731-GGUF"
            repo_dir.mkdir()
            model = repo_dir / "DeepSeek-V4-Flash-0731-UD-IQ3_XXS.gguf"
            drafter = repo_dir / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
            model.touch()
            drafter.touch()

            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                command = build_server_cmd(
                    engine="podman",
                    image="docker.io/example/llama:latest",
                    model_path=str(model),
                    draft_model_path=str(drafter),
                    context_size=8192,
                    use_fa=True,
                    use_no_mmap=True,
                    custom_args=(
                        "--spec-type draft-dspark --spec-draft-n-max 3 "
                        "--fit off -ngld 99"
                    ),
                    engine_args=[],
                )

            self.assertEqual(
                command[command.index("-md") + 1],
                "/models/DeepSeek-V4-Flash-0731-GGUF/"
                "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
            )
            self.assertIn("draft-dspark", command)
            self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "3")
            self.assertEqual(command[command.index("-ngld") + 1], "99")

    def test_server_rejects_drafter_outside_models_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary) / "models"
            models_dir.mkdir()
            model = models_dir / "model.gguf"
            drafter = Path(temporary) / "drafter.gguf"
            model.touch()
            drafter.touch()

            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                with self.assertRaisesRegex(ValueError, "Draft model"):
                    build_server_cmd(
                        engine="podman",
                        image="docker.io/example/llama:latest",
                        model_path=str(model),
                        draft_model_path=str(drafter),
                        context_size=8192,
                        use_fa=True,
                        use_no_mmap=True,
                        custom_args="",
                        engine_args=[],
                    )


if __name__ == "__main__":
    unittest.main()
