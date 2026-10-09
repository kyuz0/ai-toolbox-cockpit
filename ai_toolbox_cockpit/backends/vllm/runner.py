"""Pure vLLM direct-container command builder."""

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from ai_toolbox_cockpit.runtime.engines import adapt_nvidia_runtime_args
from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman


@dataclass(frozen=True)
class VllmCachePaths:
    huggingface: Path
    vllm: Path
    triton: Path
    aiter: Path
    offload: Path = Path("~/.cache/tcclaviger-r9700")


def apply_toolbox_policy_overrides(policy: dict, backend_config: dict | None) -> dict:
    """Apply image-specific model policy overrides without mutating the catalogue."""
    result = dict(policy)
    if backend_config:
        overrides = backend_config.get("policy_overrides", {})
        if isinstance(overrides, dict):
            result.update(overrides)
    return result


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
    model_directory: Path | None = None,
    speculation: str = "baseline",
    draft_directory: Path | None = None,
) -> list[str]:
    if not model_id.strip():
        raise ValueError("model_id is required")
    if tensor_parallel not in policy.get("valid_tp", [1]):
        raise ValueError(f"Tensor parallel size {tensor_parallel} is not permitted for {model_id}")
    if port <= 0 or max_num_seqs <= 0 or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("port, max sequences, and GPU utilization are invalid")

    tcclaviger = policy.get("runtime_variant") == "tcclaviger"
    if tcclaviger and (model_directory is None or speculation != "mtp"):
        raise ValueError("tcclaviger requires the downloaded Flash Next checkpoint and MTP-3")

    speculative_config = None
    draft_path = None
    if speculation != "baseline":
        recipe = policy.get("speculation", {}).get(speculation)
        if speculation not in {"dflash2", "mtp"} or not recipe:
            raise ValueError("The selected toolbox/model does not support this speculative mode")
        if speculation == "dflash2":
            if max_num_seqs != 1:
                raise ValueError("This qualified DFlash2 profile supports one sequence")
            if draft_directory is None:
                raise ValueError("DFlash2 requires a local draft model directory")
            draft_path = draft_directory.expanduser().resolve()
            if any(char in str(draft_path) for char in (":", "\n", "\r", "\0")):
                raise ValueError("Draft model directory cannot contain colons or control characters")
            speculative_config = dict(recipe["config"], model="/models/draft")
        else:
            if draft_directory is not None:
                raise ValueError("Embedded MTP does not use a separate draft directory")
            speculative_config = dict(recipe["config"])
        policy = dict(policy, env={**policy.get("env", {}), **recipe.get("env", {})},
                      extra_flags=recipe.get("extra_flags", policy.get("extra_flags", [])))
    elif draft_directory is not None:
        raise ValueError("Select DFlash2 before supplying a draft model directory")

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
    cleaned = adapt_nvidia_runtime_args(engine, cleaned)
    cleaned = upgrade_groups_for_podman(engine, cleaned)

    caches = cache_paths or default_cache_paths()
    command = [engine, "run", "--rm", "-it", "--name", "ai-toolbox-cockpit-vllm-server"]
    command.extend(cleaned)
    if tcclaviger:
        command.extend(["--memory=56g", "--memory-swap=56g", "--shm-size=32g",
                        "--ulimit", "memlock=-1:-1", "--entrypoint", "/usr/bin/env"])
        if engine == "podman":
            command.extend(["--runtime", "crun"])
    else:
        command.extend(["--ipc=host", "--cap-add=SYS_PTRACE"])
    if engine == "podman":
        command.extend(["--security-opt", "label=disable"])
        if not tcclaviger:
            command.append("--userns=keep-id")
    elif engine == "docker" and not tcclaviger:
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
    if tcclaviger:
        offload = caches.offload.expanduser().resolve()
        if any(char in str(offload) for char in (":", "\n", "\r", "\0")):
            raise ValueError("NVMe cache directory cannot contain colons or control characters")
        for source, target in ((offload, "/cache"), (offload / "ple", "/app/pleoffload"),
                               (offload / "tunableop", "/tunableop"), (offload / "lru_store", "/lru_store")):
            command.extend(["-v", f"{source}:{target}:rw"])
    for key, value in policy.get("env", {}).items():
        command.extend(["-e", f"{key}={value}"])

    model_ref = model_id
    if model_directory is not None:
        directory = model_directory.expanduser().resolve()
        if any(char in str(directory) for char in (":", "\n", "\r", "\0")):
            raise ValueError("Local model directory cannot contain colons or control characters")
        command.extend(["-v", f"{directory}:/models/target:ro"])
        model_ref = "/models/target"
    elif policy.get("requires_local_model"):
        raise ValueError("This prepared checkpoint requires a local model directory")
    if draft_path is not None:
        command.extend(["-v", f"{draft_path}:/models/draft:ro"])
    if tcclaviger:
        command.extend(["--workdir", "/app", image, "/app/tools/image_entrypoint.sh", model_ref])
    else:
        command.extend(["--workdir", "/workspace", image, "vllm", "serve", model_ref])
    if model_directory is None and policy.get("revision"):
        command.extend(["--revision", str(policy["revision"])])
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
    if speculative_config is not None:
        command.extend(["--speculative-config", json.dumps(speculative_config, separators=(",", ":"))])
    command.extend(shlex.split(extra_args) if extra_args else [])
    return command


def build_device_probe_cmd(engine: str, image: str, engine_args: list[str]) -> list[str]:
    if engine not in {"podman", "docker"}:
        raise ValueError("Choose Podman or Docker")
    arguments = upgrade_groups_for_podman(engine, adapt_nvidia_runtime_args(engine, list(engine_args)))
    probe = "import json,torch; print(json.dumps([{'index':i,'name':torch.cuda.get_device_name(i),'architecture':getattr(torch.cuda.get_device_properties(i),'gcnArchName','')} for i in range(torch.cuda.device_count())]))"
    return [engine, "run", "--rm", "--network=none", *arguments, "--entrypoint", "/opt/vllm/bin/python", image, "-c", probe]
