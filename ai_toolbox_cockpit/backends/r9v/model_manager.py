"""Pinned R9V package inventory, exact downloads and explicit hash verification."""

import hashlib
import shutil
import sys
from pathlib import Path

from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings

PLE_FILENAME = "per_layer_token_embd.iq4_nl.bin"


def load_packages() -> tuple[dict, ...]:
    return load_model_catalog().backends["r9v"].entries


def get_package(package_id: str) -> dict:
    for entry in load_packages():
        if entry["id"] == package_id:
            return entry
    raise ValueError("Select the tested R9V Qwen3.8 Flash Next package.")


def get_paths() -> dict[str, Path]:
    settings = get_backend_settings("r9v")
    defaults = {"models_dir": load_model_catalog().backends["r9v"].storage["default"],
                "ple_dir": "~/r9v-data", "cache_dir": "~/r9v-data/cache-toolbox-rocm10"}
    return {key: Path(str(settings.get(key, value))).expanduser().resolve()
            for key, value in defaults.items()}


def mount_path(value: str | Path) -> Path:
    if not str(value).strip():
        raise ValueError("Enter a directory path.")
    path = Path(value).expanduser().resolve()
    if any(char in str(path) for char in (":", "\n", "\r", "\0")):
        raise ValueError("Container mount paths cannot contain colons or control characters.")
    return path


def save_paths(values: dict[str, str]) -> bool:
    try:
        paths = {key: mount_path(value) for key, value in values.items()
                 if key in {"models_dir", "ple_dir", "cache_dir"}}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return save_backend_settings("r9v", {key: str(path) for key, path in paths.items()})
    except (ValueError, OSError):
        return False


def incomplete_files(package: dict, directory: Path) -> list[dict]:
    root = directory.expanduser().resolve()
    missing = []
    for item in package["files"]:
        path = root / item["path"]
        try:
            complete = (path.resolve().is_relative_to(root) and path.is_file()
                        and path.stat().st_size == item["size_bytes"])
        except OSError:
            complete = False
        if not complete:
            missing.append(item)
    return missing


def ple_ready(package: dict, directory: Path) -> bool:
    path = directory / PLE_FILENAME
    try:
        return path.is_file() and path.stat().st_size == package["ple"]["size_bytes"]
    except OSError:
        return False


def verify_file(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}. Repair the package before starting.")


def verify_package(package: dict, directory: Path) -> None:
    if incomplete_files(package, directory):
        raise ValueError("Package files are missing or incomplete. Download / Repair first.")
    for item in package["files"]:
        print(f"Verifying {item['path']}", flush=True)
        verify_file(directory / item["path"], item["sha256"])


def get_download_cmd(package: dict, directory: Path) -> list[str]:
    executable = Path(sys.executable).with_name("hf")
    hf = str(executable) if executable.is_file() else (shutil.which("hf") or "hf")
    return [hf, "download", package["repo"], *(item["path"] for item in package["files"]),
            "--revision", package["revision"], "--local-dir", str(mount_path(directory))]
