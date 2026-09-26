"""Pure Gufo container command builder."""

import ipaddress
import shlex
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman

from .model_manager import get_model, resolved_files


CONTAINER_NAME = "gufo-cockpit-server"
_OWNED_OPTIONS = {
    "--host", "--port", "--sessions", "--model", "--context", "--max-tokens",
    "--served-model-name", "--speculative", "--mtp-model", "--dspark-model",
    "--draft-tokens",
}


def _extra_arguments(value: str) -> list[str]:
    arguments = shlex.split(value)
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in _OWNED_OPTIONS:
            raise ValueError(f"Use the dedicated Gufo control for {option}.")
    return arguments


def build_server_cmd(
    *, engine: str, image: str, engine_args: list[str], platform_id: str,
    model_id: str, speculation_mode: str = "baseline",
    host: str = "127.0.0.1", port: int = 18080, context_size: int | None = None,
    sessions: int = 1, max_tokens: int = 8192, draft_tokens: int | None = None,
    extra_args: str = "",
) -> list[str]:
    if platform_id != "strix-halo":
        raise ValueError("Gufo supports Strix Halo (gfx1151) only.")
    if engine not in {"podman", "docker"}:
        raise ValueError("Select Podman or Docker.")
    host = host.strip()
    if host == "localhost":
        host = "127.0.0.1"
    try:
        ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("Host must be an IP address or localhost.") from error
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    if not 1 <= sessions <= 16:
        raise ValueError("Sessions must be between 1 and 16.")
    if not 1 <= max_tokens <= 262144:
        raise ValueError("Maximum output tokens must be between 1 and 262144.")

    model = get_model(model_id)
    context = context_size if context_size is not None else model["context_size"]
    if not 1 <= context <= 262144:
        raise ValueError("Context must be between 1 and Gufo's native 262144 positions.")
    resolved = resolved_files(model)
    missing_targets = [name for name, path in resolved["targets"].items() if path is None]
    if missing_targets:
        raise ValueError(
            "Download or repair the Gufo target first. Missing/incomplete: "
            + ", ".join(missing_targets)
        )
    target_paths = [Path(path) for path in resolved["targets"].values()]
    target_parents = {path.parent for path in target_paths}
    if len(target_parents) != 1:
        raise ValueError("All target shards must be stored in one directory.")
    model_path = Path(resolved["model"])
    target_parent = model_path.parent

    speculation = model.get("speculation")
    sidecar_path = Path(resolved["sidecar"]) if resolved["sidecar"] else None
    if speculation_mode != "baseline":
        if not speculation or speculation_mode != speculation["mode"]:
            raise ValueError("The selected model does not support that Gufo speculative mode.")
        if sidecar_path is None:
            raise ValueError(f"Download or repair the {speculation_mode.upper()} sidecar first.")
    if draft_tokens is None and speculation:
        draft_tokens = speculation.get("draft_tokens")
    if draft_tokens is not None and not 1 <= draft_tokens <= 7:
        raise ValueError("Gufo draft tokens must be between 1 and 7.")

    command = [engine, "run", "--rm", "-it", "--name", CONTAINER_NAME]
    command.extend(upgrade_groups_for_podman(engine, list(engine_args)))
    command.append("--ipc=host")
    if engine == "podman":
        command.extend(["--security-opt", "label=disable", "--userns=keep-id"])
    port_mapping = f"{port}:{port}" if host == "0.0.0.0" else f"{host}:{port}:{port}"
    command.extend([
        "-p", port_mapping,
        "-v", f"{target_parent}:/models/target:ro",
    ])
    container_sidecar = ""
    if speculation_mode != "baseline" and sidecar_path is not None:
        if sidecar_path.parent == target_parent:
            container_sidecar = f"/models/target/{sidecar_path.name}"
        else:
            command.extend(["-v", f"{sidecar_path.parent}:/models/speculation:ro"])
            container_sidecar = f"/models/speculation/{sidecar_path.name}"
    command.extend([
        image,
        "gufo", "serve",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--sessions", str(sessions),
        "llm",
        "--model", f"/models/target/{model_path.name}",
        "--context", str(context),
        "--max-tokens", str(max_tokens),
        "--served-model-name", model["served_model_name"],
        "--think", "off",
        "--max-pending-per-client", "8",
    ])
    if speculation_mode == "mtp":
        command.extend([
            "--speculative", "mtp",
            "--mtp-model", container_sidecar,
            "--draft-tokens", str(draft_tokens),
        ])
    elif speculation_mode == "dspark":
        command.extend(["--dspark-model", container_sidecar])
    command.extend(_extra_arguments(extra_args))
    return command
