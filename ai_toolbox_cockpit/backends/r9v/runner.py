"""R9V commands delegate model/kernel flags to the image's tested r9v-serve."""

import ipaddress
import math
import re
import shlex
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman
from .model_manager import PLE_FILENAME, get_package, incomplete_files, mount_path, ple_ready

CONTAINER_NAME = "ai-toolbox-cockpit-r9v-server"
DEFAULTS = {"host": "127.0.0.1", "port": "8004", "devices": "0,1",
            "context": "131072", "batch": "1024", "sequences": "1",
            "kv_bytes": "2285670400", "expert_cache_slots": "16", "offload": "112.5",
            "offload_devices": "112.5,112.5", "served_model": "qwen3.8-flash-next"}
# These are owned by the form or fixed by the tested TP2/MTP2/SSD profile.
RESERVED_ARGS = {"--model", "--tokenizer", "--speculative-config", "--load-format",
                 "--quantization", "--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp",
                 "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
                 "--kv-cache-memory-bytes", "--cpu-offload-gb", "--cpu-offload-params",
                 "--host", "--port", "--served-model-name", "--async-scheduling",
                 "--no-async-scheduling", "--compilation-config", "--api-key"}


def _engine(engine: str) -> None:
    if engine not in {"podman", "docker"}:
        raise ValueError("Select Podman or Docker.")


def build_prepare_cmd(*, engine: str, image: str, models_dir: Path, ple_dir: Path,
                      package_id: str) -> list[str]:
    _engine(engine)
    package = get_package(package_id)
    model, ple = mount_path(models_dir), mount_path(ple_dir)
    if incomplete_files(package, model):
        raise ValueError("Download / Repair the R9V package before preparing PLE.")
    return [engine, "run", "--rm", "-it", "--network=none", "--user", "0:0",
            "--security-opt", "label=disable", "-v", f"{model}:/models:ro",
            "-v", f"{ple}:/ple", image, "r9v-model", "prepare"]


def build_server_cmd(*, engine: str, image: str, engine_args: list[str], platform_id: str,
                     models_dir: Path, ple_dir: Path, cache_dir: Path, package_id: str,
                     values: dict[str, str] | None = None, api_key: str = "",
                     extra_args: str = "") -> list[str]:
    if platform_id != "r9700":
        raise ValueError("R9V is tested only for Qwen3.8 Flash Next on 2× R9700 + 64 GB RAM.")
    _engine(engine)
    options = {**DEFAULTS, **(values or {})}
    devices = options["devices"].strip()
    if not re.fullmatch(r"\d+,\d+", devices) or len({int(x) for x in devices.split(",")}) != 2:
        raise ValueError("GPU devices must be two distinct indices, for example 0,1.")
    numbers = {}
    for key, maximum in (("port", 65535), ("context", 262144), ("batch", 131072),
                         ("sequences", 16), ("kv_bytes", 32 * 1024**3),
                         ("expert_cache_slots", 16)):
        try:
            value = int(options[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be an integer.") from error
        minimum = 0 if key == "expert_cache_slots" else 1
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}.")
        numbers[key] = value
    if numbers["batch"] < numbers["sequences"]:
        raise ValueError("Batched prefill tokens must be at least the sequence count.")
    offload = options["offload"].strip()
    offload_devices = options["offload_devices"].strip()
    per_device = offload_devices.split(",")
    try:
        if len(per_device) != 2 or any(not math.isfinite(float(v)) or float(v) < 0
                                       for v in [offload, *per_device]):
            raise ValueError
    except ValueError as error:
        raise ValueError("Offload requires non-negative logical GB and two per-device values.") from error
    host = options["host"].strip()
    try:
        address = ipaddress.ip_address("127.0.0.1" if host == "localhost" else host)
    except ValueError as error:
        raise ValueError("Host must be an IP address or localhost.") from error
    binding = f"[{address}]" if address.version == 6 else str(address)
    served_model = options["served_model"].strip()
    if not served_model or any(c.isspace() for c in served_model):
        raise ValueError("API model name must be non-empty and contain no whitespace.")
    extra = shlex.split(extra_args)
    if any(arg.split("=", 1)[0] in RESERVED_ARGS for arg in extra):
        raise ValueError("Extra arguments cannot override the form or the fixed TP2/MTP2/SSD profile.")
    package = get_package(package_id)
    model, ple, cache = (mount_path(path) for path in (models_dir, ple_dir, cache_dir))
    if model == cache or model in cache.parents or ple == cache:
        raise ValueError("Use a separate compilation-cache directory outside the model package.")
    missing = incomplete_files(package, model)
    if missing:
        raise ValueError("Download / Repair the R9V package in Models first. Missing/incomplete: "
                         + ", ".join(item["path"] for item in missing))
    if not ple_ready(package, ple):
        raise ValueError("Prepare PLE in Models first (26.82 GiB extracted file required).")
    env = {"R9V_VISIBLE_DEVICES": devices, "R9V_MAX_MODEL_LEN": str(numbers["context"]),
           "R9V_MAX_NUM_BATCHED_TOKENS": str(numbers["batch"]),
           "R9V_MAX_NUM_SEQS": str(numbers["sequences"]),
           "R9V_KV_CACHE_MEMORY_BYTES": str(numbers["kv_bytes"]),
           "R9V_CPU_OFFLOAD_GB": offload, "R9V_CPU_OFFLOAD_GB_BY_DEVICE": offload_devices,
           "R9V_SERVED_MODEL_NAME": served_model, "R9V_TENSOR_PARALLEL_SIZE": "2",
           "R9V_MTP_SPEC_TOKENS": "2", "R9V_PLE_RESIDENCY_MODE": "ssd",
           "R9V_PLE_WORKER_TIMING": "1", "R9V_SERIALIZE_EXPERT_LOAD": "1",
           "R9V_TIERED_EXPERT_CACHE_SLOTS": str(numbers["expert_cache_slots"])}
    command = [engine, "run", "--rm", "-it", "--name", CONTAINER_NAME]
    if engine == "podman":
        command += ["--runtime", "crun"]
    command += [*upgrade_groups_for_podman(engine, engine_args), "--user", "0:0",
                "--ipc=host", "--security-opt", "label=disable",
                "-p", f"{binding}:{numbers['port']}:8000", "-v", f"{model}:/models:ro",
                "-v", f"{ple / PLE_FILENAME}:/ple/{PLE_FILENAME}:ro", "-v", f"{cache}:/cache"]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += [image, "r9v-serve"]
    if api_key:
        command += ["--api-key", api_key]
    return [*command, *extra]
