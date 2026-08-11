import tempfile
import unittest
from pathlib import Path

from ai_toolbox_cockpit.backends.comfyui.runner import ComfyPaths, build_server_cmd
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.runtime.interactive import InteractiveBackend, InteractiveRuntime
from ai_toolbox_cockpit.runtime.toolboxes import run_in_toolbox_command


class ComfyUiCommandTests(unittest.TestCase):
    def build(self, directory: str, **overrides) -> list[str]:
        root = Path(directory)
        values = {
            "engine": "podman",
            "image": "docker.io/kyuz0/amd-strix-halo-comfyui:latest",
            "engine_args": ["--device", "/dev/kfd", "--group-add", "keep-groups"],
            "paths": ComfyPaths(root / "models", root / "inputs", root / "outputs", root / "user"),
        }
        values.update(overrides)
        return build_server_cmd(**values)

    def test_direct_server_preserves_all_persistent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory)
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "-v"]
        self.assertEqual(len(mounts), 4)
        self.assertTrue(any(value.endswith(":/opt/ComfyUI/models") for value in mounts))
        self.assertTrue(any(value.endswith(":/opt/ComfyUI/input") for value in mounts))
        self.assertTrue(any(value.endswith(":/workspace/comfy-outputs") for value in mounts))
        self.assertTrue(any(value.endswith(":/opt/ComfyUI/user") for value in mounts))

    def test_direct_server_uses_toolbox_startup_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory)
        for flag in ("--disable-mmap", "--gpu-only", "--disable-smart-memory", "--cache-none", "--bf16-vae"):
            self.assertIn(flag, command)
        self.assertIn("/opt/ComfyUI/main.py", command)
        self.assertEqual(command[command.index("--port") + 1], "8000")

    def test_individual_startup_flags_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self.build(directory, cache_none=False, bf16_vae=False)
        self.assertNotIn("--cache-none", command)
        self.assertNotIn("--bf16-vae", command)

    def test_toolbox_model_manager_command_uses_wrapper(self) -> None:
        toolbox_runtime = InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)
        distrobox_runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)
        self.assertEqual(
            run_in_toolbox_command(toolbox_runtime, "comfyui-strix-halo", ["model_manager"]),
            ["toolbox", "run", "--container", "comfyui-strix-halo", "model_manager"],
        )
        self.assertEqual(
            run_in_toolbox_command(distrobox_runtime, "comfyui-strix-halo", ["model_manager"]),
            ["distrobox", "enter", "comfyui-strix-halo", "--", "model_manager"],
        )


if __name__ == "__main__":
    unittest.main()
