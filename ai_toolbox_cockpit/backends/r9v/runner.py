"""Pure command builder for the pinned dual-R9700 R9V Qwen profile."""

from __future__ import annotations

import grp
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman


CONTAINER_NAME = "ai-toolbox-cockpit-r9v-server"
CONTAINER_PLE = "/ple/per_layer_token_embd.iq4_nl.bin"

R9V_ENV = {
    "RADIANCE_CPU_OFFLOAD_GB_BY_DEVICE": "112.5,112.5",
    "R9V_CPU_OFFLOAD_GB_BY_DEVICE": "112.5,112.5",
    "RADIANCE_TIERED_EXPERT_MANIFEST": "/models/manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
    "RADIANCE_UVA_HOST_COHERENCE": "default",
    "RADIANCE_UVA_HOST_NONCOHERENT": "0",
    "RADIANCE_USE_R4D": "0",
    "RADIANCE_USE_R4D_AR": "0",
    "RADIANCE_USE_R4D_GDN": "0",
    "RADIANCE_USE_R4D_AR_QUANT": "0",
    "QWEN38_USE_TIERED_IQ_MOE_HIP": "1",
    "QWEN38_TIERED_IQ_MOE_VARIANT": "reuse3v2",
    "QWEN38_TIERED_PREFILL_GROUP_SIZE": "16",
    "QWEN38_TIERED_EXPERT_CACHE_SLOTS": "16",
    "QWEN38_TIERED_EXPERT_CACHE_RANKS": "1",
    "QWEN38_TIERED_EXPERT_CACHE_POLICY": "lru",
    "QWEN38_TIERED_EXPERT_CACHE_ASYNC": "0",
    "QWEN38_USE_DENSE_MMVQ_HIP": "1",
    "QWEN38_USE_DENSE_MMVQ_REUSE2": "1",
    "QWEN38_USE_DENSE_MMVQ_Q8_REUSE2": "1",
    "QWEN38_USE_DENSE_MMVQ_REUSE3": "1",
    "QWEN38_USE_DENSE_MMVQ_REUSE4": "0",
    "QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3": "1",
    "QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT": "exact4-w8",
    "QWEN38_USE_DENSE_HC_DOWN_BF16_M3": "1",
    "QWEN38_FUSED_HC_UP_MIX": "1",
    "VLLM_GGUF_FUSED_MOE_SHARED_EPILOGUE": "1",
    "QWEN38_USE_HIP_FUSED_GDN_MTP": "1",
    "VLLM_QWEN4_EXP_RDNA4_QSA_STRIDED": "1",
    "VLLM_GGUF_NATIVE_SAFE_MOE_IDS": "1",
    "VLLM_GGUF_QWEN4_EXP_MULTIMODAL": "1",
    "VLLM_QWEN4_EXP_MTP_FP8_EXPERT_ONLY": "0",
    "VLLM_QWEN4_EXP_MTP_FUSED_FC_GATHER": "0",
    "VLLM_KV_CACHE_LAYOUT": "BLHNC",
    "VLLM_ROCM_MOE_PADDING": "0",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
    "VLLM_ROCM_USE_AITER_MHA": "0",
    "VLLM_ROCM_USE_AITER_MLA": "0",
    "VLLM_ROCM_USE_AITER_MOE": "0",
    "VLLM_ROCM_USE_AITER_RMSNORM": "0",
    "VLLM_ROCM_USE_AITER_FP8BMM": "0",
    "VLLM_ROCM_USE_AITER_FP4BMM": "0",
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": "1",
    "VLLM_PLE_CPU_OFFLOAD": "1",
    "VLLM_PLE_RESIDENCY_MODE": "ssd",
    "VLLM_PLE_MMAP_HOST_REGISTER": "0",
    "VLLM_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES": "28800138240",
    "VLLM_PLE_PINNED_RESERVE_BYTES": "17179869184",
    "VLLM_PLE_BOUNDED_BYTES": "4294967296",
    "VLLM_PLE_BOUNDED_CHUNK_BYTES": "4096",
    "VLLM_PLE_MMAP_READAHEAD": "1",
    "VLLM_PLE_RSS_LOG_ROWS": "131072",
    "VLLM_PLE_WORKER_TIMING": "0",
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "0",
    "QWEN38_PROFILE_DENSE_SHAPES": "0",
    "GGUF_PLE_MMAP_PATH": CONTAINER_PLE,
    "GGUF_PLE_MMAP_TRIM_ROWS": "131072",
}

RUNTIME_PROBE = """import json, os, subprocess, torch
rocm_package = subprocess.check_output(
    ['dpkg-query', '-W', '-f=${Version}', 'amdrocm-base10.0'], text=True
).strip()
assert rocm_package.startswith('10.0.0-'), rocm_package
assert torch.version.hip and torch.version.hip.startswith('7.15'), torch.version.hip
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
devices = []
for index in range(2):
    properties = torch.cuda.get_device_properties(index)
    architecture = getattr(properties, 'gcnArchName', '')
    assert 'gfx1201' in architecture, architecture
    value = torch.ones(32, device=f'cuda:{index}')
    assert float(value.sum()) == 32.0
    devices.append({'index': index, 'name': properties.name, 'arch': architecture})
for path in (
    '/opt/r9v/kernels/qwen38_dense_mmvq_hip.so',
    '/opt/r9v/kernels/qwen38_tiered_iq_moe_hip.so',
    '/opt/r9v/kernels/qwen38_fused_gdn_mtp_hip.so',
):
    assert os.path.isfile(path) and os.path.getsize(path) > 0, path
print(json.dumps({'rocm_package': rocm_package, 'hip': torch.version.hip, 'devices': devices}))
"""


def _resolve_docker_groups(args: list[str]) -> list[str]:
    """Use host GIDs for Docker device groups, as the R9V launcher does."""
    result: list[str] = []
    index = 0
    while index < len(args):
        value = args[index]
        if value == "--group-add" and index + 1 < len(args):
            group = args[index + 1]
            if group in {"render", "video"}:
                try:
                    group = str(grp.getgrnam(group).gr_gid)
                except KeyError:
                    pass
            result.extend([value, group])
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def _runtime_args(engine: str, engine_args: list[str]) -> list[str]:
    args = upgrade_groups_for_podman(engine, engine_args)
    return _resolve_docker_groups(args) if engine == "docker" else args


def _validate_devices(visible_devices: str) -> list[str]:
    devices = [value.strip() for value in visible_devices.split(",")]
    if len(devices) != 2 or not all(value.isdigit() for value in devices):
        raise ValueError("R9V requires exactly two numeric HIP device indices")
    if devices[0] == devices[1]:
        raise ValueError("R9V tensor-parallel ranks must use different GPUs")
    return devices


def build_runtime_probe_cmd(
    *,
    engine: str,
    image: str,
    engine_args: list[str],
    visible_devices: str,
) -> list[str]:
    devices = _validate_devices(visible_devices)
    command = [engine, "run", "--rm", *_runtime_args(engine, engine_args)]
    command.extend([
        "--security-opt", "label=disable",
        "--env", f"HIP_VISIBLE_DEVICES={','.join(devices)}",
        "--env", f"ROCR_VISIBLE_DEVICES={','.join(devices)}",
        image,
        "python", "-c", RUNTIME_PROBE,
    ])
    return command


def build_server_cmd(
    *,
    engine: str,
    image: str,
    engine_args: list[str],
    model_dir: Path,
    ple_path: Path,
    cache_dir: Path,
    visible_devices: str = "0,1",
    host: str = "localhost",
    port: int = 8004,
) -> list[str]:
    devices = _validate_devices(visible_devices)
    if port <= 0 or port > 65535:
        raise ValueError("port must be between 1 and 65535")

    args = _runtime_args(engine, engine_args)
    command = [engine, "run", "--rm", "-it", "--name", CONTAINER_NAME]
    command.extend(args)
    if "--ipc=host" not in args:
        command.append("--ipc=host")
    command.extend(["--security-opt", "label=disable"])

    bind_host = "127.0.0.1" if host == "localhost" else host
    mapping = f"{port}:8000" if bind_host == "0.0.0.0" else f"{bind_host}:{port}:8000"
    command.extend([
        "--publish", mapping,
        "--volume", f"{model_dir.resolve()}:/models:ro",
        "--volume", f"{ple_path.resolve()}:{CONTAINER_PLE}:ro",
        "--volume", f"{cache_dir.resolve()}:/cache",
        "--env", f"HIP_VISIBLE_DEVICES={','.join(devices)}",
        "--env", f"ROCR_VISIBLE_DEVICES={','.join(devices)}",
        "--env", "VLLM_CACHE_ROOT=/cache/vllm",
    ])
    for key, value in R9V_ENV.items():
        command.extend(["--env", f"{key}={value}"])

    command.extend([
        image,
        "vllm", "serve", "/models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
        "--tokenizer", "/models/metadata",
        "--hf-config-path", "/models/metadata",
        "--served-model-name", "qwen3.8-flash-next",
        "--load-format", "gguf",
        "--quantization", "gguf",
        "--tensor-parallel-size", "2",
        "--pipeline-parallel-size", "1",
        "--cpu-offload-gb", "112.5",
        "--cpu-offload-params", "experts",
        "--kv-cache-memory-bytes", "2285670400",
        "--speculative-config", '{"method":"mtp","model":"/models/mtp","num_speculative_tokens":2,"draft_tensor_parallel_size":2,"quantization":"fp8","use_local_argmax_reduction":true,"draft_load_config":{"load_format":"auto"}}',
        "--max-model-len", "131072",
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", "1024",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,3],"max_cudagraph_capture_size":3}',
        "--model-loader-extra-config", '{"mm_proj":"/models/vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"}',
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--mm-processor-kwargs", '{"min_pixels":65536,"max_pixels":262144}',
        "--mm-processor-cache-gb", "0",
        "--mm-encoder-tp-mode", "weights",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", "8000",
    ])
    return command
