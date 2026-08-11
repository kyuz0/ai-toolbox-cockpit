"""Container inspection and explicit toolbox lifecycle operations."""

import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

from .engines import ContainerEngine, detect_container_engines
from .interactive import (
    InteractiveRuntime,
    build_create_command,
    build_delete_command,
    build_enter_command,
    build_pull_command,
    detect_interactive_backend,
    runtime_environment,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class InstalledToolbox:
    name: str
    image: str
    status: str
    created: str
    engine: ContainerEngine


def inspect_installed_toolboxes(
    engines: Sequence[ContainerEngine] | None = None,
    runner: Runner = subprocess.run,
) -> tuple[InstalledToolbox, ...]:
    """Inspect all containers. Failures from one engine do not hide the other."""
    found: dict[tuple[ContainerEngine, str], InstalledToolbox] = {}
    for engine in engines if engines is not None else detect_container_engines():
        try:
            result = runner(
                [
                    engine.value,
                    "ps",
                    "-a",
                    "--format",
                    "{{.Names}}|{{.Image}}|{{.Status}}|{{.CreatedAt}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split("|", 3)]
            if len(parts) < 3 or not parts[0]:
                continue
            item = InstalledToolbox(
                name=parts[0],
                image=parts[1],
                status=parts[2].replace("292 years ago", "Unknown Date"),
                created=parts[3] if len(parts) == 4 else "",
                engine=engine,
            )
            found[(engine, item.name)] = item
    return tuple(found.values())


def create_toolbox(
    runtime: InteractiveRuntime,
    name: str,
    image: str,
    engine_args: tuple[str, ...],
    runner: Runner = subprocess.run,
) -> None:
    environment = runtime_environment(runtime)
    runner(build_pull_command(runtime, image), check=True, env=environment)
    runner(build_create_command(runtime, name, image, engine_args), check=True, env=environment)


def delete_toolbox(
    runtime: InteractiveRuntime,
    name: str,
    runner: Runner = subprocess.run,
) -> None:
    runner(build_delete_command(runtime, name), check=True, env=runtime_environment(runtime))


def enter_toolbox(
    runtime: InteractiveRuntime,
    name: str,
    caller: Callable[..., int] = subprocess.call,
) -> int:
    return caller(build_enter_command(runtime, name), env=runtime_environment(runtime))


def run_in_toolbox_command(
    runtime: InteractiveRuntime,
    name: str,
    command: Sequence[str],
) -> list[str]:
    """Build a wrapper command for a trusted, code-defined in-toolbox command."""
    if runtime.wrapper.value == "toolbox":
        return ["toolbox", "run", "--container", name, *command]
    return ["distrobox", "enter", name, "--", *command]


def extend_missing_option_pairs(args: list[str], extras: list[str]) -> list[str]:
    """Append flag/value pairs that are not already present."""
    result = list(args)
    for index in range(0, len(extras), 2):
        pair = extras[index:index + 2]
        if len(pair) == 2 and not any(
            result[position:position + 2] == pair
            for position in range(max(0, len(result) - 1))
        ):
            result.extend(pair)
    return result


def upgrade_groups_for_podman(engine: str, args: list[str]) -> list[str]:
    """Replace named device groups with Podman's host supplementary groups."""
    if engine != "podman":
        return list(args)
    group_values: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--group-add" and index + 1 < len(args):
            group_values.append(args[index + 1])
            index += 2
            continue
        if args[index].startswith("--group-add="):
            group_values.append(args[index].split("=", 1)[1])
        index += 1
    if not set(group_values).intersection({"video", "render", "rdma", "keep-groups"}):
        return list(args)
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


def get_os_toolbox_cmd() -> str:
    runtime = detect_interactive_backend()
    return runtime.wrapper.value if runtime else ""
