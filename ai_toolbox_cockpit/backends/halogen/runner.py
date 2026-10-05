"""Pure single-container Halogen launch builder; retain the shipped entrypoint."""

import ipaddress
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman

from .model_manager import get_bundle, incomplete_files


CONTAINER_NAME = "ai-toolbox-cockpit-halogen-server"


def _validated_engine_args(arguments: list[str]) -> list[str]:
    """Only retain the reviewed GPU profile; reject isolation overrides."""
    switches = {"--pull=always", "--ipc=host"}
    values = {"--device": {"/dev/kfd", "/dev/dri"},
              "--group-add": {"video", "render"},
              "--security-opt": {"seccomp=unconfined"},
              "--ulimit": {"memlock=-1:-1"}}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in switches:
            index += 1
        elif (argument in values and index + 1 < len(arguments)
              and arguments[index + 1] in values[argument]):
            index += 2
        else:
            raise ValueError(f"Halogen isolation does not allow engine argument: {argument}")
    return arguments


def build_server_cmd(
    *, engine: str, image: str, engine_args: list[str], platform_id: str,
    models_dir: Path, bundle_id: str, host: str = "127.0.0.1", port: int = 8731,
    context_size: int = 262144, kv_pool_positions: int = 524288,
    kv_slots: int = 4, prompt_cache: str = "2",
) -> list[str]:
    if platform_id != "strix-halo":
        raise ValueError("Halogen Flash supports Strix Halo (gfx1151) only.")
    if engine not in {"podman", "docker"}:
        raise ValueError("Select Podman or Docker.")
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    if not 1 <= context_size <= 262144:
        raise ValueError("Context must be between 1 and the native 262144 positions.")
    if kv_pool_positions < context_size:
        raise ValueError("KV pool positions must be at least the request context size.")
    if kv_slots < 1 or prompt_cache not in {"0", "1", "2"}:
        raise ValueError("KV slots must be positive and prompt cache must be 0, 1 or 2.")
    host = host.strip()
    if host == "localhost":
        host = "127.0.0.1"
    try:
        ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("Host must be an IP address or localhost.") from error
    directory = models_dir.expanduser().resolve()
    if ":" in str(directory):
        raise ValueError("Models directory cannot contain ':' in a container volume mount.")
    bundle = get_bundle(bundle_id)
    missing = incomplete_files(bundle, directory)
    if missing:
        raise ValueError("Download or repair the selected bundle in Models first. Missing/incomplete: "
                         + ", ".join(item["path"] for item in missing))
    environment = {
        "HALOGEN_CHECKPOINT": f"/models/{bundle['checkpoint']}",
        "HALOGEN_TOKENIZER": f"/models/{bundle['tokenizer_dir']}",
        "HALOGEN_API_PORT": str(port),
        "HALOGEN_CTX": str(context_size),
        "HALOGEN_KV_POOL_POSITIONS": str(kv_pool_positions),
        "HALOGEN_KV_SLOTS": str(kv_slots),
        "HALOGEN_PROMPT_CACHE": prompt_cache,
    }
    if bundle.get("overlay"):
        environment["HALOGEN_CK_OVERLAY"] = f"/models/{bundle['overlay']}"
    else:
        environment["HALOGEN_CK_OVERLAY"] = "none"
        environment["HALOGEN_NGRAM_TABLE"] = f"/models/{bundle['ngram_table']}"
    if bundle.get("vision_tower"):
        environment["HALOGEN_VISION_TOWER"] = f"/models/{bundle['vision_tower']}"
    command = [engine, "run", "--rm", "-it", "--name", CONTAINER_NAME,
               *upgrade_groups_for_podman(engine, _validated_engine_args(engine_args)),
               "--network=none", "--cap-drop=NET_ADMIN", "--cap-drop=NET_RAW",
               "--security-opt", "no-new-privileges"]
    for item in bundle["files"]:
        source = (directory / item["path"]).resolve(strict=True)
        if not source.is_relative_to(directory) or ":" in str(source):
            raise ValueError("Bundle files must stay inside the models directory and have mount-safe paths.")
        command.extend(["-v", f"{source}:/models/{item['path']}:ro"])
    for key, value in environment.items():
        command.extend(["-e", f"{key}={value}"])
    return [*command, image]
