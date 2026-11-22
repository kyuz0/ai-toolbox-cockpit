"""Curated HGN bundles and their companion files."""

import shutil
import sys
from pathlib import Path

from ai_toolbox_cockpit.huggingface import huggingface_download_command
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
    """Check local artifact sizes and any required file signature."""
    incomplete = []
    root = directory.expanduser().resolve()
    signature = bundle.get("file_signature")
    for item in bundle["files"]:
        path = root / item["path"]
        try:
            complete = (path.resolve().is_relative_to(root) and path.is_file()
                        and path.stat().st_size == item["size_bytes"])
            if complete and signature:
                with path.open("rb") as stream:
                    complete = stream.read(len(signature)) == signature
        except (OSError, RuntimeError):
            complete = False
        if not complete:
            incomplete.append(item)
    return incomplete


def fast_draft_download(bundle: dict) -> dict:
    head = bundle["fast_draft_head"]
    return {"name": "Faster MTP head", "files": [head], "file_signature": b"HGN1"}


def bundle_size(bundle: dict) -> int:
    return sum(item["size_bytes"] for item in bundle["files"])


def get_server_defaults(bundle: dict) -> dict[str, int]:
    return {"context": 262144, "pool": 524288, "slots": 4} | bundle.get("server_defaults", {})


def _download_cmd(repo: str, revision: str, files: list[str], directory: Path) -> list[str]:
    executable = Path(sys.executable).with_name("hf")
    hf = str(executable) if executable.is_file() else (shutil.which("hf") or "hf")
    return huggingface_download_command(
        [hf, "download", repo, *files, "--revision", revision,
         "--local-dir", str(directory.expanduser().resolve())]
    )


def get_download_commands(bundle: dict, directory: Path) -> list[list[str]]:
    sources: dict[tuple[str, str], list[str]] = {}
    for item in incomplete_files(bundle, directory):
        source = item.get("source", bundle)
        sources.setdefault((source["repo"], source["revision"]), []).append(item["path"])
    return [
        _download_cmd(repo, revision, files, directory)
        for (repo, revision), files in sources.items()
    ]
