import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from types import SimpleNamespace

from ai_toolbox_cockpit.runtime.engines import ContainerEngine, adapt_nvidia_runtime_args
from ai_toolbox_cockpit.runtime.groups import docker_host_group_ids
from ai_toolbox_cockpit.runtime.rdma import container_rdma_args, host_rdma_device_nodes
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
    upgrade_groups_for_podman,
)


class RdmaPassthroughTests(unittest.TestCase):
    def infiniband(self, *nodes: str) -> tuple[str, list[str]]:
        root = tempfile.TemporaryDirectory()
        for node in nodes:
            (Path(root.name) / node).touch()
        return root, list(nodes)

    def test_podman_passes_the_infiniband_directory_and_rdma_group(self) -> None:
        root, _ = self.infiniband("rdma_cm", "uverbs0")
        with root:
            self.assertEqual(
                container_rdma_args(ContainerEngine.PODMAN, root.name),
                ["--device", root.name, "--group-add", "rdma", "--ulimit", "memlock=-1"],
            )

    def test_docker_passes_each_infiniband_device_node(self) -> None:
        root, nodes = self.infiniband("issm0", "rdma_cm", "uverbs0")
        with root:
            args = container_rdma_args("docker", root.name)
        expected: list[str] = []
        for node in sorted(nodes):
            expected.extend(["--device", f"{root.name}/{node}"])
        expected.extend(["--ulimit", "memlock=-1"])
        self.assertEqual(args, expected)

    def test_docker_without_device_nodes_passes_nothing(self) -> None:
        root, _ = self.infiniband()
        with root:
            self.assertEqual(container_rdma_args("docker", root.name), [])

    def test_absent_infiniband_passes_no_flags(self) -> None:
        for engine in ("podman", "docker"):
            with self.subTest(engine=engine):
                self.assertEqual(container_rdma_args(engine, "/nonexistent-infiniband"), [])
                self.assertEqual(host_rdma_device_nodes("/nonexistent-infiniband"), [])


class DockerHostGroupTests(unittest.TestCase):
    def host_group(self, name: str):
        groups = {"video": 44, "render": 992}
        if name not in groups:
            raise KeyError(name)
        return SimpleNamespace(gr_gid=groups[name])

    def test_named_device_groups_become_host_gids(self) -> None:
        args = ["--device", "/dev/dri", "--group-add", "video", "--group-add", "render"]
        with patch("ai_toolbox_cockpit.runtime.groups.grp.getgrnam", side_effect=self.host_group):
            result = docker_host_group_ids(args)
        self.assertEqual(
            result,
            ["--device", "/dev/dri", "--group-add", "44", "--group-add", "992"],
        )
        self.assertEqual(args[3], "video")

    def test_missing_group_name_is_preserved(self) -> None:
        with patch("ai_toolbox_cockpit.runtime.groups.grp.getgrnam", side_effect=KeyError):
            self.assertEqual(docker_host_group_ids(["--group-add", "render"]), ["--group-add", "render"])

    def test_numeric_gids_and_equals_form_pass_through(self) -> None:
        with patch("ai_toolbox_cockpit.runtime.groups.grp.getgrnam", side_effect=KeyError):
            self.assertEqual(
                docker_host_group_ids(["--group-add", "987", "--group-add=44"]),
                ["--group-add", "987", "--group-add=44"],
            )

    def test_keep_groups_expands_to_existing_device_groups(self) -> None:
        with patch("ai_toolbox_cockpit.runtime.groups.grp.getgrnam", side_effect=self.host_group):
            self.assertEqual(
                docker_host_group_ids(["--group-add", "keep-groups", "--device", "/dev/kfd"]),
                ["--group-add", "44", "--group-add", "992", "--device", "/dev/kfd"],
            )

    def test_server_adapter_translates_docker_and_preserves_podman(self) -> None:
        args = ["--group-add", "video", "--group-add", "render"]
        self.assertEqual(upgrade_groups_for_podman("podman", args), ["--group-add", "keep-groups"])
        with patch("ai_toolbox_cockpit.runtime.groups.grp.getgrnam", side_effect=self.host_group):
            self.assertEqual(
                upgrade_groups_for_podman("docker", args),
                ["--group-add", "44", "--group-add", "992"],
            )
        self.assertEqual(args, ["--group-add", "video", "--group-add", "render"])

    def test_distrobox_docker_create_uses_host_group_ids(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)
        with patch("ai_toolbox_cockpit.runtime.groups.grp.getgrnam", side_effect=self.host_group):
            command = build_create_command(
                runtime,
                "sample",
                "docker.io/example/image:latest",
                ("--group-add", "render", "--group-add", "video"),
            )
        self.assertIn("--group-add 992", command[-1])
        self.assertIn("--group-add 44", command[-1])
        self.assertNotIn("render", command[-1])
        self.assertNotIn("video", command[-1])


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
