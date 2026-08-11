import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.backends.llama_cpp.benchmark_runner import BenchmarkSettings, build_benchmark_jobs
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd


class UbatchCommandsTest(unittest.TestCase):
    def test_benchmark_uses_calibrated_ubatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.touch()
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.benchmark_runner.get_preferred_benchmark_ubatch",
                return_value=1024,
            ):
                jobs = build_benchmark_jobs(
                    "toolbox",
                    ["llama-rocm-7.14"],
                    [str(model)],
                    Path(temporary),
                    BenchmarkSettings(platform_id="strix-halo"),
                )
            self.assertEqual(len(jobs), 2)
            for job in jobs:
                index = job.command.index("-ub")
                self.assertEqual(job.command[index + 1], "1024")

    def test_server_uses_calibration_but_respects_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.touch()
            kwargs = dict(
                engine="podman",
                image="docker.io/example/llama-rocm:latest",
                model_path=str(model),
                context_size=65536,
                use_fa=True,
                use_no_mmap=True,
                platform_id="strix-halo",
                engine_args=[],
            )
            with (
                patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=Path(temporary)),
                patch("ai_toolbox_cockpit.backends.llama_cpp.server_runner.get_preferred_ubatch", return_value=1024),
            ):
                automatic = build_server_cmd(custom_args="", **kwargs)
                explicit = build_server_cmd(custom_args="-ub 256", **kwargs)

            auto_index = automatic.index("-ub")
            self.assertEqual(automatic[auto_index + 1], "1024")
            self.assertEqual(explicit.count("-ub"), 1)
            explicit_index = explicit.index("-ub")
            self.assertEqual(explicit[explicit_index + 1], "256")

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
            with patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=Path(temporary)):
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
