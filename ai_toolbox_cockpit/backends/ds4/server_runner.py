import os
import shlex
from .model_manager import get_models_dir
from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman

KV_DISK_CONTAINER_DIR = "/var/cache/ds4-kv"
MXFP4_TILE4_ENV = "DS4_ROCM_ENABLE_MXFP4_TILE4=1"
MXFP4_DOWN_RGROUP_ENV = "DS4_ROCM_MXFP4_DOWN_RGROUP=4"

def _parse_peer_addr(peer_addr: str) -> tuple[str, str]:
    """Parse peer address input into (ip, port). Supports 'IP PORT', 'IP:PORT', or bare 'IP'."""
    if ":" in peer_addr and len(peer_addr.split()) == 1:
        parts = peer_addr.split(":")
    else:
        parts = peer_addr.split()
    ip = parts[0]
    port = parts[1] if len(parts) > 1 else "8081"
    return ip, port

def _clean_engine_args(engine_args: list[str]) -> list[str]:
    """Remove --group-add sudo from engine args (not needed for server mode)."""
    clean = []
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
        clean.append(engine_args[i])
    return clean

def build_server_cmd(engine: str, image: str, model_path: str, ctx: int,
                     host: str, port: str,
                     kv_disk_enabled: bool, kv_disk_dir: str, kv_disk_mb: int,
                     prefill_chunk: int | None, mtp_path: str, custom_args: str,
                     role: str, layers: str, peer_addr: str,
                     toolbox_config: dict,
                     ssd_enabled: bool = False, ssd_experts: str = "",
                     ssd_full_layers: str = "", ssd_cold: bool = False,
                     dist_prefill_chunk: int | None = None,
                     dist_prefill_window: int | None = None,
                     mxfp4_tile4_enabled: bool = True,
                     mxfp4_down_rgroup_enabled: bool = True,
                     dspark_enabled: bool = False,
                     dspark_path: str = "",
                     dspark_confidence: float = 0.0,
                     vision_path: str = "") -> list[str]:
    
    models_dir = str(get_models_dir())
    engine_args = _clean_engine_args(toolbox_config.get("args", []))
    engine_args = upgrade_groups_for_podman(engine, engine_args)
    server_binary = toolbox_config.get("server_binary", "ds4-server")
    
    is_multinode = role and role != "Standalone"

    if dspark_enabled and not dspark_path:
        raise ValueError("Choose a DSpark support model")
    if dspark_enabled and ssd_enabled:
        raise ValueError("DSpark cannot be combined with SSD streaming")
    if dspark_enabled and is_multinode:
        raise ValueError("DSpark is available only in standalone mode")
    if not 0.0 <= dspark_confidence <= 1.0:
        raise ValueError("DSpark confidence must be between 0 and 1")

    docker_args = [engine, "run", "--rm", "-it", "--name", "ds4-cockpit-server"]
    docker_args.extend(engine_args)
    
    # ROCm requires host IPC sharing and ptrace capabilities to avoid HSA memory mapping errors
    docker_args.extend([
        "--ipc=host",
        "--cap-add=SYS_PTRACE"
    ])

    if mxfp4_tile4_enabled:
        docker_args.extend(["--env", MXFP4_TILE4_ENV])
    if mxfp4_down_rgroup_enabled:
        docker_args.extend(["--env", MXFP4_DOWN_RGROUP_ENV])
        
    if engine == "podman":
        docker_args.extend([
            "--security-opt", "label=disable",
            "--userns=keep-id"
        ])

    if is_multinode:
        # Multi-node mode: use host networking so ds4-server can bind directly
        # to host IPs for --listen/--coordinator. Podman's rootless pasta
        # network backend cannot forward arbitrary host IPs via -p.
        docker_args.append("--network=host")
    else:
        # Standalone mode: use standard port mapping
        port_mapping = f"{port}:{port}"
        if host and host != "0.0.0.0":
            bind_ip = "127.0.0.1" if host == "localhost" else host
            port_mapping = f"{bind_ip}:{port}:{port}"
        docker_args.extend(["-p", port_mapping])

    docker_args.extend(["-v", f"{models_dir}:/models:ro"])
    if kv_disk_enabled:
        docker_args.extend(["-v", f"{kv_disk_dir}:{KV_DISK_CONTAINER_DIR}"])
    
    # Calculate relative paths for /models
    rel_path = os.path.relpath(model_path, models_dir)
    inner_model_path = f"/models/{rel_path}"

    server_args = [
        server_binary,
        "-m", inner_model_path,
        "--ctx", str(ctx),
        "--host", "0.0.0.0",
        "--port", str(port)
    ]

    if vision_path:
        vision_rel = os.path.relpath(vision_path, models_dir)
        server_args.extend(["--vision", f"/models/{vision_rel}"])
    
    if kv_disk_enabled:
        server_args.extend([
            "--kv-disk-dir", KV_DISK_CONTAINER_DIR,
            "--kv-disk-space-mb", str(kv_disk_mb),
        ])

    if ssd_enabled:
        server_args.append("--ssd-streaming")
        if ssd_experts.strip():
            server_args.extend(["--ssd-streaming-cache-experts", ssd_experts.strip()])
        if ssd_full_layers.strip():
            server_args.extend(["--ssd-streaming-full-layers", ssd_full_layers.strip()])
        if ssd_cold:
            server_args.append("--ssd-streaming-cold")

    if prefill_chunk is not None:
        server_args.extend(["--prefill-chunk", str(prefill_chunk)])
        
    if dspark_enabled:
        dspark_rel = os.path.relpath(dspark_path, models_dir)
        server_args.extend([
            "--mtp-model", f"/models/{dspark_rel}",
            "--dspark",
            "--dspark-confidence", f"{dspark_confidence:g}",
        ])
    elif mtp_path:
        mtp_rel = os.path.relpath(mtp_path, models_dir)
        server_args.extend(["--mtp-model", f"/models/{mtp_rel}"])
        
    if is_multinode:
        server_args.extend(["--role", role.lower()])
        if layers:
            server_args.extend(["--layers", layers])
        if peer_addr:
            coord_ip, coord_port = _parse_peer_addr(peer_addr)
            if role.lower() == "coordinator":
                server_args.extend(["--listen", coord_ip, coord_port])
                if dist_prefill_chunk is not None:
                    server_args.extend(["--dist-prefill-chunk", str(dist_prefill_chunk)])
                if dist_prefill_window is not None:
                    server_args.extend(["--dist-prefill-window", str(dist_prefill_window)])
            elif role.lower() == "worker":
                server_args.extend(["--coordinator", coord_ip, coord_port])

    if custom_args:
        server_args.extend(shlex.split(custom_args))
    
    return docker_args + [image] + server_args
