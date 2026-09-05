"""Curated HGN bundles: a checkpoint, a precision overlay, and a flat tokenizer."""

import shutil
import sys
from pathlib import Path

from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def get_models_dir() -> Path:
    default = load_model_catalog().backends["halogen"].storage["default"]
    return Path(str(get_backend_settings("halogen").get("models_dir", default))).expanduser().resolve()


def save_models_dir(value: str) -> bool:
    if not value.strip():
        return False
    try:
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return False
    return save_backend_settings("halogen", {"models_dir": str(path)})


def load_bundles() -> tuple[dict, ...]:
    return load_model_catalog().backends["halogen"].entries


def get_bundle(bundle_id: str) -> dict:
    for entry in load_bundles():
        if entry["id"] == bundle_id:
            return entry
    raise ValueError("Select a curated Halogen model / precision bundle.")


def incomplete_files(bundle: dict, directory: Path) -> list[dict]:
    """Size checks catch missing/partial downloads without reading 118 GiB."""
    incomplete = []
    root = directory.expanduser().resolve()
    for item in bundle["files"]:
        path = root / item["path"]
        try:
            complete = (path.resolve().is_relative_to(root) and path.is_file()
                        and path.stat().st_size == item["size_bytes"])
        except OSError:
            complete = False
        if not complete:
            incomplete.append(item)
    return incomplete


def bundle_size(bundle: dict) -> int:
    return sum(item["size_bytes"] for item in bundle["files"])


def get_download_cmd(bundle: dict, directory: Path) -> list[str]:
    executable = Path(sys.executable).with_name("hf")
    hf = str(executable) if executable.is_file() else (shutil.which("hf") or "hf")
    return [hf, "download", bundle["repo"],
            *(item["path"] for item in bundle["files"]),
            "--revision", bundle["revision"], "--local-dir", str(directory.expanduser().resolve())]
