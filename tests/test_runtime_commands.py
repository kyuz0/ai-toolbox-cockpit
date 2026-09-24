import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_toolbox_cockpit.runtime.engines import ContainerEngine, adapt_nvidia_runtime_args
from ai_toolbox_cockpit.runtime.rdma import (
    RDMAEndpoint,
    container_rdma_args,
    discover_rdma_endpoints,
    host_rdma_device_nodes,
)
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
        (Path(root.name) / "by-ibdev").mkdir()
        (Path(root.name) / "by-path").mkdir()
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


class RdmaDiscoveryTests(unittest.TestCase):
    def write_endpoint(
        self,
        root: Path,
        device: str,
        port: int,
        gid_index: int,
        gid: str,
        gid_type: str,
        link_layer: str,
        state: str,
    ) -> Path:
        port_path = root / "class" / "infiniband" / device / "ports" / str(port)
        (port_path / "gids").mkdir(parents=True)
        (port_path / "gid_attrs" / "types").mkdir(parents=True)
        (port_path / "link_layer").write_text(f"{link_layer}\n")
        (port_path / "state").write_text(f"{state}\n")
        (port_path / "gids" / str(gid_index)).write_text(f"{gid}\n")
        (port_path / "gid_attrs" / "types" / str(gid_index)).write_text(
            f"{gid_type}\n"
        )
        return root / "class" / "infiniband" / device

    def test_native_infiniband_endpoint_and_ipoib_netdev_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = self.write_endpoint(
                root,
                device="testhca0",
                port=1,
                gid_index=0,
                gid="fe80::20:fe:80:00:1",
                gid_type="IB/RoCE v1",
                link_layer="InfiniBand",
                state="4: ACTIVE",
            )
            # Linux exposes an empty fe80:: entry at GID index 1 on some drivers.
            unused = device / "ports" / "1" / "gids" / "1"
            unused.write_text("fe80::\n")
            (device / "ports" / "1" / "gid_attrs" / "types" / "1").write_text(
                "IB/RoCE v1\n"
            )
            netdev = root / "class" / "net" / "ib-test0"
            (netdev / "device").mkdir(parents=True)
            (netdev / "mode").write_text("connected\n")
            (netdev / "device" / "infiniband").symlink_to(
                device, target_is_directory=True
            )

            endpoints = discover_rdma_endpoints(
                root / "class" / "infiniband",
                root / "class" / "net",
            )

        self.assertEqual(
            endpoints,
            [
                RDMAEndpoint(
                    device="testhca0",
                    port=1,
                    gid_index=0,
                    link_layer="InfiniBand",
                    gid_type="IB/RoCE v1",
                    state="4: ACTIVE",
                    netdev="ib-test0",
                    netdev_mode="connected",
                )
            ],
        )
        self.assertTrue(endpoints[0].active)
        self.assertEqual(endpoints[0].link_type, "infiniband")

    def test_roce_and_inactive_port_link_layers_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_endpoint(
                root,
                device="rocehca0",
                port=2,
                gid_index=3,
                gid="fe80::1",
                gid_type="RoCE v2",
                link_layer="Ethernet",
                state="4: ACTIVE",
            )
            self.write_endpoint(
                root,
                device="ibhca0",
                port=1,
                gid_index=0,
                gid="fe80::2",
                gid_type="IB/RoCE v1",
                link_layer="InfiniBand",
                state="1: DOWN",
            )
            self.write_endpoint(
                root,
                device="wrongtypehca0",
                port=1,
                gid_index=0,
                gid="fe80::3",
                gid_type="IB/RoCE v1",
                link_layer="Ethernet",
                state="4: ACTIVE",
            )

            endpoints = discover_rdma_endpoints(
                root / "class" / "infiniband",
                root / "class" / "net",
            )

        by_device = {endpoint.device: endpoint for endpoint in endpoints}
        self.assertEqual(by_device["rocehca0"].link_type, "roce")
        self.assertTrue(by_device["rocehca0"].active)
        self.assertEqual(by_device["ibhca0"].link_type, "infiniband")
        self.assertFalse(by_device["ibhca0"].active)
        self.assertEqual(by_device["wrongtypehca0"].link_type, "unknown")

    def test_missing_sysfs_returns_no_endpoints(self) -> None:
        self.assertEqual(
            discover_rdma_endpoints("/nonexistent-rdma", "/nonexistent-net"),
            [],
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

    def test_distrobox_create_excludes_rdma_convenience_directories(self) -> None:
        runtime = InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine.DOCKER)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for node in ("rdma_cm", "uverbs0"):
                (root / node).touch()
            (root / "by-ibdev").mkdir()
            (root / "by-path").mkdir()
            command = build_create_command(
                runtime,
                "sample",
                "docker.io/example/image:latest",
                (),
                rdma_path=root,
            )

        additional_flags = command[-1]
        self.assertIn(f"--device {root}/rdma_cm", additional_flags)
        self.assertIn(f"--device {root}/uverbs0", additional_flags)
        self.assertNotIn("by-ibdev", additional_flags)
        self.assertNotIn("by-path", additional_flags)
        self.assertIn("--ulimit memlock=-1", additional_flags)

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
