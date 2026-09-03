"""R9V package storage, verification, download, and PLE command helpers."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


PLE_FILENAME = "per_layer_token_embd.iq4_nl.bin"


def get_models_dir() -> Path:
    value = get_backend_settings("r9v").get("models_dir", "~/models/r9v")
    return Path(str(value)).expanduser()


def save_models_dir(path_str: str) -> bool:
    try:
        Path(path_str).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return save_backend_settings("r9v", {"models_dir": path_str})


def package_dir(model: dict) -> Path:
    return get_models_dir() / str(model["directory"])


def ple_path(model: dict) -> Path:
    return package_dir(model) / "derived" / PLE_FILENAME


def invalid_artifacts(model: dict) -> list[str]:
    root = package_dir(model)
    invalid: list[str] = []
    for artifact in model["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file():
            invalid.append(f"missing: {artifact['path']}")
        elif path.stat().st_size != artifact["bytes"]:
            invalid.append(f"wrong size: {artifact['path']}")
    return invalid


def package_is_complete(model: dict) -> bool:
    return not invalid_artifacts(model)


def ple_is_complete(model: dict) -> bool:
    path = ple_path(model)
    return path.is_file() and path.stat().st_size == model["ple"]["bytes"]


def verify_package(model: dict) -> list[str]:
    """Verify every R9V package artifact, returning human-readable failures."""
    failures = invalid_artifacts(model)
    if failures:
        return failures
    root = package_dir(model)
    for artifact in model["artifacts"]:
        path = root / artifact["path"]
        digest = hashlib.sha256()
        print(f"Verifying {artifact['path']}…", flush=True)
        with path.open("rb") as source:
            while chunk := source.read(16 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != artifact["sha256"]:
            failures.append(f"hash mismatch: {artifact['path']}")
    return failures


def get_download_cmd(model: dict) -> list[str]:
    executable = os.path.join(os.path.dirname(sys.executable), "hf")
    if not os.path.exists(executable):
        executable = "hf"
    return [
        executable,
        "download",
        str(model["repo"]),
        "--revision",
        str(model["revision"]),
        "--local-dir",
        str(package_dir(model)),
    ]


def build_prepare_ple_cmd(engine: str, image: str, model: dict) -> list[str]:
    root = package_dir(model).resolve()
    output = ple_path(model).resolve()
    command = [engine, "run", "--rm", "--network", "none"]
    if engine == "podman":
        command.extend(["--security-opt", "label=disable", "--userns=keep-id"])
    elif engine == "docker":
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    command.extend([
        "--volume", f"{root}:/models:ro",
        "--volume", f"{output.parent}:/r9v-data",
        image,
        "python3", "/usr/local/libexec/r9v/prepare_ple.py",
    ])
    command.extend(f"/models/{path}" for path in model["ple"]["source_shards"])
    command.extend(["--output", f"/r9v-data/{PLE_FILENAME}"])
    return command


def verify_ple(model: dict) -> str | None:
    """Verify the derived PLE payload, returning a failure or ``None``."""
    path = ple_path(model)
    if not path.is_file():
        return f"missing: derived/{PLE_FILENAME}"
    if path.stat().st_size != model["ple"]["bytes"]:
        return f"wrong size: derived/{PLE_FILENAME}"
    digest = hashlib.sha256()
    print(f"Verifying derived/{PLE_FILENAME}…", flush=True)
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != model["ple"]["sha256"]:
        return f"hash mismatch: derived/{PLE_FILENAME}"
    return None
