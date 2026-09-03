"""Interactive Toolbx/Distrobox command construction.

This module is intentionally side-effect free. Detection inspects executable
availability; command builders never execute a container operation.
"""

import grp
import os
import shlex
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .engines import (
    ContainerEngine,
    adapt_nvidia_runtime_args,
    detect_container_engines,
)


class InteractiveBackend(StrEnum):
    TOOLBOX = "toolbox"
    DISTROBOX = "distrobox"


@dataclass(frozen=True)
class InteractiveRuntime:
    wrapper: InteractiveBackend
    engine: ContainerEngine


def _host_os_ids(path: Path = Path("/etc/os-release")) -> set[str]:
    identifiers: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.lower() in {"id", "id_like"}:
                identifiers.update(value.strip().strip("\"'").lower().split())
    except OSError:
        pass
    return identifiers


def detect_interactive_backend() -> InteractiveRuntime | None:
    """Choose a wrapper/engine pair compatible with the current host."""
    engines = detect_container_engines()
    configured = os.path.basename(os.environ.get("DBX_CONTAINER_MANAGER", ""))
    distrobox_engine = (
        ContainerEngine(configured)
        if configured in {engine.value for engine in engines}
        else ContainerEngine.PODMAN
        if ContainerEngine.PODMAN in engines
        else ContainerEngine.DOCKER
        if ContainerEngine.DOCKER in engines
        else None
    )
    has_toolbox = bool(shutil.which("toolbox") and ContainerEngine.PODMAN in engines)
    has_distrobox = bool(shutil.which("distrobox") and distrobox_engine)

    if configured and has_distrobox and configured in {engine.value for engine in engines}:
        return InteractiveRuntime(InteractiveBackend.DISTROBOX, ContainerEngine(configured))
    if _host_os_ids().intersection({"ubuntu", "debian", "arch"}):
        if has_distrobox:
            return InteractiveRuntime(InteractiveBackend.DISTROBOX, distrobox_engine)
        if has_toolbox:
            return InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)
    else:
        if has_toolbox:
            return InteractiveRuntime(InteractiveBackend.TOOLBOX, ContainerEngine.PODMAN)
        if has_distrobox:
            return InteractiveRuntime(InteractiveBackend.DISTROBOX, distrobox_engine)
    return None


def interactive_runtime_for_engine(engine: ContainerEngine) -> InteractiveRuntime | None:
    """Resolve an installed container's engine to a compatible wrapper."""
    preferred = detect_interactive_backend()
    if preferred and preferred.engine is engine:
        return preferred
    if engine is ContainerEngine.PODMAN and shutil.which("toolbox"):
        return InteractiveRuntime(InteractiveBackend.TOOLBOX, engine)
    if shutil.which("distrobox"):
        return InteractiveRuntime(InteractiveBackend.DISTROBOX, engine)
    return None


def _remove_group_add_values(args: list[str], values: set[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--group-add" and index + 1 < len(args):
            if args[index + 1] in values:
                index += 2
                continue
        elif args[index].startswith("--group-add="):
            if args[index].split("=", 1)[1] in values:
                index += 1
                continue
        result.append(args[index])
        index += 1
    return result


def _podman_keep_groups(args: list[str]) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--group-add" and index + 1 < len(args):
            values.append(args[index + 1])
            index += 2
            continue
        if args[index].startswith("--group-add="):
            values.append(args[index].split("=", 1)[1])
        index += 1
    if not set(values).intersection({"video", "render", "rdma", "keep-groups"}):
        return args
    result: list[str] = []
    added = False
    index = 0
    while index < len(args):
        if args[index] == "--group-add" and index + 1 < len(args):
            if not added:
                result.extend(["--group-add", "keep-groups"])
                added = True
            index += 2
            continue
        if args[index].startswith("--group-add="):
            if not added:
                result.extend(["--group-add", "keep-groups"])
                added = True
            index += 1
            continue
        result.append(args[index])
        index += 1
    return result


def _docker_host_group_ids(args: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--group-add" and index + 1 < len(args):
            value = args[index + 1]
            try:
                value = str(grp.getgrnam(value).gr_gid)
            except KeyError:
                pass
            result.extend(["--group-add", value])
            index += 2
            continue
        if args[index].startswith("--group-add="):
            value = args[index].split("=", 1)[1]
            try:
                value = str(grp.getgrnam(value).gr_gid)
            except KeyError:
                pass
            result.append(f"--group-add={value}")
            index += 1
            continue
        result.append(args[index])
        index += 1
    return result


def _rdma_args(runtime: InteractiveRuntime, rdma_path: Path) -> list[str]:
    if runtime.wrapper is not InteractiveBackend.DISTROBOX or not rdma_path.is_dir():
        return []
    if runtime.engine is ContainerEngine.PODMAN:
        return ["--device", str(rdma_path), "--group-add", "rdma", "--ulimit", "memlock=-1"]
    result: list[str] = []
    gids: set[int] = set()
    try:
        devices = sorted(rdma_path.iterdir())
    except OSError:
        devices = []
    for device in devices:
        result.extend(["--device", str(device)])
        try:
            gids.add(device.stat().st_gid)
        except OSError:
            pass
    for gid in sorted(gids):
        result.extend(["--group-add", str(gid)])
    if result:
        result.extend(["--ulimit", "memlock=-1"])
    return result


def _extend_missing_pairs(args: list[str], extras: list[str]) -> list[str]:
    result = list(args)
    for index in range(0, len(extras), 2):
        pair = extras[index:index + 2]
        if len(pair) == 2 and not any(
            result[pos:pos + 2] == pair for pos in range(max(0, len(result) - 1))
        ):
            result.extend(pair)
    return result


def build_create_command(
    runtime: InteractiveRuntime,
    name: str,
    image: str,
    engine_args: tuple[str, ...],
    rdma_path: Path = Path("/dev/infiniband"),
) -> list[str]:
    if runtime.wrapper is InteractiveBackend.TOOLBOX:
        if runtime.engine is not ContainerEngine.PODMAN:
            raise ValueError("Toolbx requires Podman")
        return ["toolbox", "create", "--image", image, name]

    args = adapt_nvidia_runtime_args(runtime.engine, list(engine_args))
    args = _remove_group_add_values(args, {"sudo"})
    args = _extend_missing_pairs(args, _rdma_args(runtime, rdma_path))
    if runtime.engine is ContainerEngine.PODMAN:
        args = _podman_keep_groups(args)
    elif runtime.engine is ContainerEngine.DOCKER:
        args = _docker_host_group_ids(args)
    command = ["distrobox", "create", "--name", name, "--image", image]
    if args:
        command.extend(["--additional-flags", shlex.join(args)])
    return command


def build_pull_command(runtime: InteractiveRuntime, image: str) -> list[str]:
    return [runtime.engine.value, "pull", image]


def build_enter_command(runtime: InteractiveRuntime, name: str) -> list[str]:
    return [runtime.wrapper.value, "enter", name]


def build_delete_command(runtime: InteractiveRuntime, name: str) -> list[str]:
    return [runtime.wrapper.value, "rm", "-f", name]


def runtime_environment(runtime: InteractiveRuntime) -> dict[str, str]:
    environment = os.environ.copy()
    if runtime.wrapper is InteractiveBackend.DISTROBOX:
        environment["DBX_CONTAINER_MANAGER"] = runtime.engine.value
    return environment
