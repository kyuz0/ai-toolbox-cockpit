import unittest
from unittest.mock import patch

from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.runtime.interactive import (
    InteractiveBackend,
    InteractiveRuntime,
    build_create_command,
    build_delete_command,
    build_enter_command,
    build_pull_command,
    interactive_runtime_for_engine,
)


class RuntimeCommandTests(unittest.TestCase):
    def test_toolbox_create_uses_podman_host_integration(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)
        command = build_create_command(runtime, "sample", "docker.io/example/image:latest", ("--device", "/dev/kfd"))
        self.assertEqual(command, ["toolbox", "create", "--image", "docker.io/example/image:latest", "sample"])

    def test_distrobox_create_passes_backend_flags(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)
        command = build_create_command(runtime, "sample", "docker.io/example/image:latest", ("--device", "/dev/kfd"))
        self.assertEqual(command[-2:], ["--additional-flags", "--device /dev/kfd"])

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


if __name__ == "__main__":
    unittest.main()
