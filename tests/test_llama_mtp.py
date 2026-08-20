import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.llama_cpp.config import get_mtp_config, load_models
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
    get_external_mtp_model,
    scan_local_models,
)
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd


class LlamaMtpTests(unittest.TestCase):
    def test_only_rocmfp4_qwen_3_8_uses_an_external_mtp_model(self) -> None:
        configs = {model["repo"]: model for model in load_models()}

        standard = get_mtp_config(configs["unsloth/Qwen3.8-27B-GGUF"])
        rocmfp4 = get_mtp_config(
            configs["kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF"]
        )

        self.assertNotIn("draft_model", standard)
        self.assertEqual(standard["default_draft_n"], 2)
        self.assertEqual(rocmfp4["draft_model"], "mtp-Qwen3.8-27B-Q4_0.gguf")
        self.assertEqual(rocmfp4["default_draft_n"], 4)

    def test_external_mtp_file_is_resolved_beside_main_model_and_not_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            repo_dir = models_dir / "Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF"
            repo_dir.mkdir()
            model = repo_dir / "Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf"
            mtp = repo_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
            model.touch()
            mtp.touch()

            self.assertEqual(
                get_external_mtp_model(str(model), mtp.name),
                mtp,
            )
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                discovered = scan_local_models()

            self.assertEqual([entry["path"] for entry in discovered], [str(model)])

    def test_server_mounts_external_mtp_with_model_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            repo_dir = models_dir / "Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF"
            repo_dir.mkdir()
            model = repo_dir / "Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf"
            mtp = repo_dir / "mtp-Qwen3.8-27B-Q4_0.gguf"
            model.touch()
            mtp.touch()

            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                command = build_server_cmd(
                    engine="podman",
                    image="docker.io/example/llama-rocmfpx:latest",
                    model_path=str(model),
                    mtp_draft_model_path=str(mtp),
                    context_size=65536,
                    use_fa=True,
                    use_no_mmap=True,
                    custom_args="--spec-type draft-mtp --spec-draft-n-max 4",
                    engine_args=[],
                )

            self.assertEqual(
                command[command.index("--model-draft") + 1],
                "/models/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF/"
                "mtp-Qwen3.8-27B-Q4_0.gguf",
            )
            self.assertIn("draft-mtp", command)


if __name__ == "__main__":
    unittest.main()
