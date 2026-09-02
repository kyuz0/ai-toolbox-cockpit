"""Pure vLLM direct-container command builder."""

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman


@dataclass(frozen=True)
class VllmCachePaths:
    huggingface: Path
    vllm: Path
    triton: Path
    aiter: Path


def default_cache_paths() -> VllmCachePaths:
    return VllmCachePaths(
        Path("~/.cache/huggingface").expanduser(),
        Path("~/.cache/vllm").expanduser(),
        Path("~/.cache/triton").expanduser(),
        Path("~/.aiter").expanduser(),
    )


def build_server_cmd(
    *,
    engine: str,
    image: str,
    engine_args: list[str],
    model_id: str,
    policy: dict,
    host: str = "localhost",
    port: int = 8000,
    tensor_parallel: int = 1,
    max_num_seqs: int = 1,
    max_model_len: str = "auto",
    gpu_memory_utilization: float = 0.90,
    attention_backend: str | None = None,
    enforce_eager: bool | None = None,
    dtype: str = "auto",
    api_key: str = "",
    hf_token: str = "",
    extra_args: str = "",
    cache_paths: VllmCachePaths | None = None,
) -> list[str]:
    if not model_id.strip():
        raise ValueError("model_id is required")
    if tensor_parallel not in policy.get("valid_tp", [1]):
        raise ValueError(f"Tensor parallel size {tensor_parallel} is not permitted for {model_id}")
    if port <= 0 or max_num_seqs <= 0 or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("port, max sequences, and GPU utilization are invalid")

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

    caches = cache_paths or default_cache_paths()
    command = [engine, "run", "--rm", "-it", "--name", "ai-toolbox-cockpit-vllm-server"]
    command.extend(cleaned)
    command.extend(["--ipc=host", "--cap-add=SYS_PTRACE"])
    if engine == "podman":
        command.extend(["--security-opt", "label=disable", "--userns=keep-id"])
    elif engine == "docker":
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    bind_host = "127.0.0.1" if host == "localhost" else host
    mapping = f"{port}:{port}" if bind_host == "0.0.0.0" else f"{bind_host}:{port}:{port}"
    command.extend([
        "-p", mapping,
        "-e", "HOME=/workspace",
        "-e", "VLLM_CONFIG_ROOT=/workspace/.cache/vllm/config",
        "-e", "TRITON_CACHE_DIR=/workspace/.cache/triton",
        "-e", "TILELANG_CACHE_DIR=/workspace/.cache/triton/tilelang",
        "-e", "VLLM_NO_USAGE_STATS=1",
        "-e", f"HF_TOKEN={hf_token}" if hf_token else "HF_TOKEN",
    ])
    mounts = (
        (caches.huggingface, "/workspace/.cache/huggingface"),
        (caches.vllm, "/workspace/.cache/vllm"),
        (caches.triton, "/workspace/.cache/triton"),
        (caches.aiter, "/workspace/.aiter"),
    )
    for host_path, container_path in mounts:
        command.extend(["-v", f"{host_path}:{container_path}"])
    for key, value in policy.get("env", {}).items():
        command.extend(["-e", f"{key}={value}"])

    command.extend([image, "vllm", "serve", model_id])
    resolved_model_len = policy.get("ctx", "auto") if max_model_len == "auto" else max_model_len
    command.extend([
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(resolved_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ])
    if policy.get("trust_remote"):
        command.append("--trust-remote-code")
    eager = policy.get("enforce_eager", False) if enforce_eager is None else enforce_eager
    if eager:
        command.append("--enforce-eager")
    if api_key:
        command.extend(["--api-key", api_key])
    configured_backend = policy.get("attention_backend", "TRITON_ATTN")
    if configured_backend is not None:
        command.extend(["--attention-backend", attention_backend or configured_backend])
    command.extend(str(item) for item in policy.get("extra_flags", []))
    command.extend(shlex.split(extra_args) if extra_args else [])
    return command
