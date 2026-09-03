import subprocess
import unittest
from unittest.mock import patch

from ai_toolbox_cockpit.runtime.engines import ContainerEngine, adapt_nvidia_runtime_args
from ai_toolbox_cockpit.runtime.interactive import (
    InteractiveBackend,
    InteractiveRuntime,
    build_create_command,
    build_delete_command,
    build_enter_command,
    build_pull_command,
    interactive_runtime_for_engine,
)
from ai_toolbox_cockpit.runtime.toolboxes import (
    InstalledToolbox,
    inspect_installed_toolboxes,
    runtime_for_installed_toolbox,
)


class RuntimeCommandTests(unittest.TestCase):
    def test_gb10_podman_runtime_args_are_preserved(self) -> None:
        args = [
            "--runtime",
            "/usr/bin/nvidia-container-runtime",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all",
        ]
        self.assertEqual(adapt_nvidia_runtime_args(ContainerEngine.PODMAN, args), args)

    def test_gb10_docker_runtime_args_use_gpus_flag(self) -> None:
        args = [
            "--runtime",
            "/usr/bin/nvidia-container-runtime",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all",
        ]
        self.assertEqual(
            adapt_nvidia_runtime_args(ContainerEngine.DOCKER, args),
            [
                "--env",
                "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all",
                "--gpus",
                "all",
            ],
        )

    def test_toolbox_create_uses_podman_host_integration(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)
        command = build_create_command(runtime, "sample", "docker.io/example/image:latest", ("--device", "/dev/kfd"))
        self.assertEqual(command, ["toolbox", "create", "--image", "docker.io/example/image:latest", "sample"])

    def test_distrobox_create_passes_backend_flags(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)
        command = build_create_command(runtime, "sample", "docker.io/example/image:latest", ("--device", "/dev/kfd"))
        self.assertEqual(command[-2:], ["--additional-flags", "--device /dev/kfd"])

    def test_distrobox_create_adapts_gb10_flags_for_docker(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)
        command = build_create_command(
            runtime,
            "sample",
            "docker.io/example/image:latest",
            (
                "--runtime",
                "/usr/bin/nvidia-container-runtime",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all",
            ),
        )
        self.assertNotIn("nvidia-container-runtime", command[-1])
        self.assertIn("--gpus all", command[-1])

    def test_enter_command_is_wrapper_specific(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.PODMAN)
        self.assertEqual(build_enter_command(runtime, "sample"), ["distrobox", "enter", "sample"])

    def test_mutation_commands_are_explicit_and_target_one_item(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.PODMAN)
        self.assertEqual(build_pull_command(runtime, "docker.io/example/image:latest"), ["podman", "pull", "docker.io/example/image:latest"])
        self.assertEqual(build_delete_command(runtime, "sample"), ["distrobox", "rm", "-f", "sample"])

    def test_installed_docker_container_uses_distrobox_docker(self) -> None:
        with (
            patch("ai_toolbox_cockpit.runtime.interactive.detect_interactive_backend", return_value=None),
            patch("ai_toolbox_cockpit.runtime.interactive.shutil.which", side_effect=lambda value: "/bin/distrobox" if value == "distrobox" else None),
        ):
            runtime = interactive_runtime_for_engine(ContainerEngine.DOCKER)
        self.assertEqual(runtime, InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER))

    def test_inspection_recovers_toolbx_ownership_from_container_labels(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)

        def runner(command, **kwargs):
            self.assertIn("{{.Labels}}", command[-1])
            return subprocess.CompletedProcess(
                command,
                0,
                "sample|example:latest|Up 2 hours|2026-08-11|com.github.containers.toolbox=true|\n",
                "",
            )

        installed = inspect_installed_toolboxes([ContainerEngine.PODMAN], runner)
        self.assertEqual(installed[0].runtime, runtime)

    def test_inspection_recovers_distrobox_engine_and_ownership(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)

        def runner(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    "sample|example:latest|Exited|2026-08-11|"
                    "manager=distrobox,com.github.containers.toolbox=true|/run/host\n"
                ),
                "",
            )

        installed = inspect_installed_toolboxes([ContainerEngine.DOCKER], runner)
        self.assertEqual(installed[0].runtime, runtime)

    def test_persisted_ownership_does_not_require_host_default_detection(self) -> None:
        expected = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.PODMAN)
        installed = InstalledToolbox(
            name="sample",
            image="example:latest",
            status="Up",
            created="2026-08-11",
            engine=ContainerEngine.PODMAN,
            runtime=expected,
        )
        with patch(
            "ai_toolbox_cockpit.runtime.toolboxes.interactive_runtime_for_engine",
            return_value=None,
        ):
            runtime = runtime_for_installed_toolbox(installed)
        self.assertEqual(runtime, expected)


if __name__ == "__main__":
    unittest.main()
