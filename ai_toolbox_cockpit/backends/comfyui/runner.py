"""Pure direct-container ComfyUI command builder."""

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman


@dataclass(frozen=True)
class ComfyPaths:
    models: Path
    inputs: Path
    outputs: Path
    user: Path


def default_paths() -> ComfyPaths:
    return ComfyPaths(
        Path("~/comfy-models").expanduser(),
        Path("~/comfy-inputs").expanduser(),
        Path("~/comfy-outputs").expanduser(),
        Path("~/.local/share/ai-toolbox-cockpit/comfyui/user").expanduser(),
    )


def build_server_cmd(
    *,
    engine: str,
    image: str,
    engine_args: list[str],
    paths: ComfyPaths,
    host: str = "localhost",
    port: int = 8000,
    disable_mmap: bool = True,
    gpu_only: bool = True,
    disable_smart_memory: bool = True,
    cache_none: bool = True,
    bf16_vae: bool = True,
    extra_args: str = "",
) -> list[str]:
    if port <= 0:
        raise ValueError("port must be positive")
    cleaned: list[str] = []
    skip = False
    for index, arg in enumerate(engine_args):
        if skip:
            skip = False
            continue
        if arg == "--group-add" and index + 1 < len(engine_args) and engine_args[index + 1] == "sudo":
            skip = True
            continue
        if arg != "--group-add=sudo":
            cleaned.append(arg)
    cleaned = upgrade_groups_for_podman(engine, cleaned)

    command = [engine, "run", "--rm", "-it", "--name", "ai-toolbox-cockpit-comfyui-server"]
    command.extend(cleaned)
    command.extend(["--ipc=host", "--cap-add=SYS_PTRACE"])
    if engine == "podman":
        command.extend(["--security-opt", "label=disable", "--userns=keep-id"])
    elif engine == "docker":
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    bind_host = "127.0.0.1" if host == "localhost" else host
    mapping = f"{port}:{port}" if bind_host == "0.0.0.0" else f"{bind_host}:{port}:{port}"
    command.extend(["-p", mapping, "-e", "HOME=/workspace"])
    for host_path, container_path in (
        (paths.models, "/opt/ComfyUI/models"),
        (paths.inputs, "/opt/ComfyUI/input"),
        (paths.outputs, "/workspace/comfy-outputs"),
        (paths.user, "/opt/ComfyUI/user"),
    ):
        command.extend(["-v", f"{host_path}:{container_path}"])
    command.extend([
        image,
        "/opt/venv/bin/python", "/opt/ComfyUI/main.py",
        "--listen", "0.0.0.0",
        "--port", str(port),
        "--output-directory", "/workspace/comfy-outputs",
        "--input-directory", "/opt/ComfyUI/input",
        "--user-directory", "/opt/ComfyUI/user",
    ])
    for enabled, flag in (
        (disable_mmap, "--disable-mmap"),
        (gpu_only, "--gpu-only"),
        (disable_smart_memory, "--disable-smart-memory"),
        (cache_none, "--cache-none"),
        (bf16_vae, "--bf16-vae"),
    ):
        if enabled:
            command.append(flag)
    command.extend(shlex.split(extra_args) if extra_args else [])
    return command
