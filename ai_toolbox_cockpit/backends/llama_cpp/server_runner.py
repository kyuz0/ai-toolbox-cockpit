import os
from pathlib import Path
from .model_manager import resolve_model_path
from ai_toolbox_cockpit.runtime.engines import adapt_nvidia_runtime_args
from ai_toolbox_cockpit.runtime.toolboxes import (
    extend_missing_option_pairs,
    upgrade_groups_for_podman,
)

import shlex

def get_server_rdma_args(
    engine: str,
    platform_id: str,
    rdma_path: str = "/dev/infiniband",
) -> list[str]:
    """Return native container-engine RDMA flags for supported Strix Halo images."""
    if platform_id != "strix-halo" or not os.path.isdir(rdma_path):
        return []

    if engine == "podman":
        return [
            "--device", rdma_path,
            "--group-add", "rdma",
            "--ulimit", "memlock=-1",
        ]

    if engine == "docker":
        args = []
        for device in sorted(Path(rdma_path).iterdir()):
            args.extend(["--device", str(device)])
        if args:
            args.extend(["--ulimit", "memlock=-1"])
        return args

    return []

def build_server_cmd(engine: str, image: str, model_path: str, context_size: int, use_fa: bool, use_no_mmap: bool, custom_args: str, host: str = "localhost", port: str = "8080", ngl: int | None = None, hip_devices: str = "", platform_id: str = "", engine_args: list[str] = None, kv_cache_type: str = "", supports_load_mode: bool = False, api_key: str = "", vision_projector_path: str = "", draft_model_path: str = "", mtp_draft_model_path: str = "", batch_size: int | None = None, ubatch_size: int | None = None, parallel_sequences: int | None = None, load_mode: str = "") -> list[str]:
    from .model_manager import get_models_dir
    models_dir = str(get_models_dir())
    
    if engine_args is None:
        # fallback based on image/platform_id
        if "intel" in image.lower() or "intel" in platform_id.lower():
            engine_args = [
                "--device", "/dev/dri",
                "--group-add", "video",
                "--group-add", "render",
                "--security-opt", "seccomp=unconfined"
            ]
        else:
            engine_args = [
                "--device", "/dev/dri",
                "--device", "/dev/kfd",
                "--group-add", "video",
                "--group-add", "render",
                "--security-opt", "seccomp=unconfined"
            ]
    else:
        # Copy to avoid modifying the original list in-place
        clean_args = []
        skip_next = False
        for i in range(len(engine_args)):
            if skip_next:
                skip_next = False
                continue
            if engine_args[i] == "--group-add" and i + 1 < len(engine_args) and engine_args[i+1] == "sudo":
                skip_next = True
                continue
            if engine_args[i] == "--group-add=sudo":
                continue
            clean_args.append(engine_args[i])
        engine_args = clean_args

    engine_args = adapt_nvidia_runtime_args(engine, engine_args)
    rdma_args = get_server_rdma_args(engine, platform_id)
    engine_args = extend_missing_option_pairs(engine_args, rdma_args)
    engine_args = upgrade_groups_for_podman(engine, engine_args)

    cmd = [
        engine, "run", "--rm", "-it", "--name", "llama-cockpit-server"
    ]
    cmd.extend(engine_args)
    
    if hip_devices:
        if "intel" in image.lower() or "intel" in platform_id.lower():
            cmd.extend(["-e", f"ZE_AFFINITY_MASK={hip_devices}"])
        else:
            cmd.extend(["-e", f"HIP_VISIBLE_DEVICES={hip_devices}"])
            
    if "rocmfp4" in image.lower():
        cmd.extend([
            "-e", "HSA_OVERRIDE_GFX_VERSION=11.5.1",
            "-e", "GGML_HIP_ENABLE_UNIFIED_MEMORY=1"
        ])
        
    # Podman requires these flags to read host volumes without permission issues (SELinux / UID mapping)
    if engine == "podman":
        cmd.extend([
            "--security-opt", "label=disable",
            "--userns=keep-id"
        ])
        
    port_mapping = f"{port}:{port}"
    if host and host != "0.0.0.0":
        bind_ip = "127.0.0.1" if host == "localhost" else host
        port_mapping = f"{bind_ip}:{port}:{port}"

    cmd.extend([
        "-e", "XDG_CACHE_HOME=/tmp",
        "-v", f"{models_dir}:/models:ro",
        "-p", port_mapping,
        image
    ])
    
    # Resolve first file if sharded
    actual_file = resolve_model_path(model_path)
    
    # Calculate the inner path based on relative position
    rel_path = os.path.relpath(actual_file, models_dir)
    inner_model_path = f"/models/{rel_path}"

    cmd.extend([
        "llama-server",
        "-m", inner_model_path,
    ])

    if draft_model_path:
        draft_file = Path(resolve_model_path(draft_model_path)).resolve()
        models_root = Path(models_dir).resolve()
        try:
            draft_rel_path = draft_file.relative_to(models_root)
        except ValueError as error:
            raise ValueError("Draft model must be inside the models directory") from error
        cmd.extend(["-md", f"/models/{draft_rel_path}"])

    if mtp_draft_model_path:
        mtp_draft_file = Path(resolve_model_path(mtp_draft_model_path)).resolve()
        models_root = Path(models_dir).resolve()
        try:
            mtp_draft_rel_path = mtp_draft_file.relative_to(models_root)
        except ValueError as error:
            raise ValueError("MTP draft model must be inside the models directory") from error
        cmd.extend(["--model-draft", f"/models/{mtp_draft_rel_path}"])

    cmd.extend(["-c", str(context_size)])
    if ngl is not None:
        cmd.extend(["-ngl", str(ngl)])
    cmd.extend([
        "--host", "0.0.0.0",
        "--port", str(port),
    ])

    if vision_projector_path:
        projector_file = Path(resolve_model_path(vision_projector_path)).resolve()
        models_root = Path(models_dir).resolve()
        try:
            projector_rel_path = projector_file.relative_to(models_root)
        except ValueError as error:
            raise ValueError("Vision projector must be inside the models directory") from error
        cmd.extend(["--mmproj", f"/models/{projector_rel_path}"])
    
    if supports_load_mode:
        selected_load_mode = load_mode or ("none" if use_no_mmap else "mmap")
        if selected_load_mode not in {"none", "mmap", "dio"}:
            raise ValueError(f"Unsupported llama.cpp load mode: {selected_load_mode}")
        cmd.extend(["--load-mode", selected_load_mode])
    elif use_no_mmap:
        cmd.append("--no-mmap")
        
    if use_fa:
        cmd.extend(["-fa", "1"])
    
    if kv_cache_type:
        cmd.extend(["--cache-type-k", kv_cache_type, "--cache-type-v", kv_cache_type])

    if batch_size is not None:
        cmd.extend(["-b", str(batch_size)])
    if ubatch_size is not None:
        cmd.extend(["-ub", str(ubatch_size)])
    if parallel_sequences is not None:
        cmd.extend(["-np", str(parallel_sequences)])

    if api_key:
        cmd.extend(["--api-key", api_key])

    cmd.extend(shlex.split(custom_args) if custom_args else [])
    
    return cmd
