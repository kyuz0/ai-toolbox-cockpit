"""Pure Gufo container command builder."""

import ipaddress
import shlex
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman

from .model_manager import get_model, resolved_files


CONTAINER_NAME = "gufo-cockpit-server"
_OWNED_OPTIONS = {
    "--host", "--port", "--sessions", "--model", "--context", "--max-tokens",
    "--served-model-name", "--speculative", "--mtp-model", "--dflash-model",
    "--dspark-model", "--mmproj",
    "--draft-tokens", "--think", "--reasoning-effort", "--max-pending-per-client",
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
    sessions: int = 1, max_tokens: int | None = None, draft_tokens: int | None = None,
    thinking_effort: str = "high", max_pending_per_client: int = 4,
    vision_enabled: bool = False, extra_args: str = "",
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
        raise ValueError("Concurrent requests must be between 1 and 16.")
    if max_tokens is not None and (type(max_tokens) is not int or (max_tokens != -1 and not 1 <= max_tokens <= 262144)):
        raise ValueError("Maximum output tokens must be -1 or between 1 and 262144.")
    if thinking_effort not in {
        "off", "auto", "minimal", "low", "medium", "high", "xhigh", "max",
    }:
        raise ValueError(
            "Thinking effort must be off, model default, minimal, low, medium, "
            "high, xhigh, or max."
        )
    if not 1 <= max_pending_per_client <= 16:
        raise ValueError("Queued requests per client must be between 1 and 16.")

    model = get_model(model_id)
    if max_tokens is None:
        max_tokens = model.get("server_defaults", {}).get("max_tokens", 32768)
    context = context_size if context_size is not None else model["context_size"]
    if type(context) is not int or context < 1:
        raise ValueError('Context must be a positive integer.')
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
    default_draft_tokens = speculation.get("draft_tokens") if speculation else None
    if default_draft_tokens is None or draft_tokens is None:
        draft_tokens = default_draft_tokens
    if draft_tokens is not None and not 1 <= draft_tokens <= 7:
        raise ValueError("Gufo draft tokens must be between 1 and 7.")

    projector_path = None
    if vision_enabled:
        if not model.get("vision"):
            raise ValueError("The selected Gufo model has no vision profile.")
        projector = resolved.get("vision")
        if projector is None:
            raise ValueError("Download or repair the matching BF16 vision projector first.")
        projector_path = Path(projector)

    command = [engine, "run", "--rm", "-it", "--name", CONTAINER_NAME]
    command.extend(upgrade_groups_for_podman(engine, list(engine_args)))
    command.append("--ipc=host")
    if engine == "podman":
        command.extend(["--security-opt", "label=disable", "--userns=keep-id"])
    port_mapping = f"{port}:{port}" if host == "0.0.0.0" else f"{host}:{port}:{port}"
    command.extend(["-p", port_mapping])
    mount_target_files = model.get("vision") or not speculation or speculation["mode"] == "mtp"
    if mount_target_files:
        # Gufo discovers projectors beside the target. Expose only target shards
        # so text-only remains explicit even when a projector exists locally.
        for path in target_paths:
            command.extend(["-v", f"{path}:/models/target/{path.name}:ro"])
    else:
        command.extend(["-v", f"{target_parent}:/models/target:ro"])
    container_sidecar = ""
    if speculation_mode != "baseline" and sidecar_path is not None:
        if sidecar_path.parent == target_parent and not mount_target_files:
            container_sidecar = f"/models/target/{sidecar_path.name}"
        else:
            command.extend(["-v", f"{sidecar_path.parent}:/models/speculation:ro"])
            container_sidecar = f"/models/speculation/{sidecar_path.name}"
    if projector_path is not None:
        command.extend(["-v", f"{projector_path}:/models/vision/{projector_path.name}:ro"])
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
    ])
    if projector_path is not None:
        command.extend(["--mmproj", f"/models/vision/{projector_path.name}"])
    if thinking_effort == "off":
        command.extend(["--think", "off"])
    elif thinking_effort == "auto":
        command.extend(["--think", "auto"])
    else:
        command.extend(["--think", "on", "--reasoning-effort", thinking_effort])
    command.extend(["--max-pending-per-client", str(max_pending_per_client)])
    if speculation_mode == "baseline":
        command.extend(["--speculative", "off"])
    elif speculation_mode == "mtp":
        command.extend([
            "--speculative", "mtp",
            "--mtp-model", container_sidecar,
        ])
        if draft_tokens is not None:
            command.extend(["--draft-tokens", str(draft_tokens)])
    elif speculation_mode == "dflash2":
        command.extend([
            "--speculative", "dflash2",
            "--dflash-model", container_sidecar,
            "--draft-tokens", str(draft_tokens),
        ])
    elif speculation_mode == "dspark":
        command.extend(["--dspark-model", container_sidecar])
    command.extend(_extra_arguments(extra_args))
    return command
