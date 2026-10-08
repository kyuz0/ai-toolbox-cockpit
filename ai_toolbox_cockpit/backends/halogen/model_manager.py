"""Curated HGN bundles: a checkpoint, an overlay or n-gram table, and a flat tokenizer."""

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


def load_npu_models() -> tuple[dict, ...]:
    """Curated Ryzen AI NPU models, pinned to Hub revisions in models.json."""
    return load_model_catalog().backends["halogen"].npu_models


def get_npu_model(model_id: str) -> dict:
    for entry in load_npu_models():
        if entry["id"] == model_id:
            return entry
    raise ValueError("Select a catalogued Halogen NPU model.")


def npu_model_dir(model_id: str, directory: Path) -> Path:
    return directory.expanduser().resolve() / "npu" / model_id


def npu_device_owner(entry: dict) -> str:
    """The model that owns the NPU device program the entry runs."""
    return entry.get("devices_from") or entry["id"]


def npu_shared_device_files(entry: dict) -> list[dict]:
    donor_id = entry.get("devices_from")
    if not donor_id:
        return []
    donor = get_npu_model(donor_id)
    return [item for item in donor["files"] if item["path"].startswith("devices/")]


def incomplete_npu_files(entry: dict, directory: Path) -> list[tuple[str, dict]]:
    """Size checks return (owner model, file) pairs missing or truncated under <dir>/npu/<owner>."""
    incomplete: list[tuple[str, dict]] = []
    root = directory.expanduser().resolve()
    for owner, files in ((entry["id"], entry["files"]),
                         (npu_device_owner(entry), npu_shared_device_files(entry))):
        if not files:
            continue
        model_root = root / "npu" / owner
        for item in files:
            path = model_root / item["path"]
            try:
                complete = (path.resolve().is_relative_to(model_root) and path.is_file()
                            and path.stat().st_size == item["size_bytes"])
            except OSError:
                complete = False
            if not complete:
                incomplete.append((owner, item))
    return incomplete


def npu_model_size(entry: dict) -> int:
    return (sum(item["size_bytes"] for item in entry["files"])
            + sum(item["size_bytes"] for item in npu_shared_device_files(entry)))


def get_bundle(bundle_id: str) -> dict:
    for entry in load_bundles():
        if entry["id"] == bundle_id:
            return entry
    raise ValueError("Select a curated Halogen model / precision bundle.")


def incomplete_files(bundle: dict, directory: Path) -> list[dict]:
    """Size checks catch missing/partial downloads without reading the model weights."""
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


def _hf_download(repo: str, revision: str, directory: Path, paths: list[str]) -> list[str]:
    executable = Path(sys.executable).with_name("hf")
    hf = str(executable) if executable.is_file() else (shutil.which("hf") or "hf")
    return [hf, "download", repo, *paths, "--revision", revision, "--local-dir", str(directory)]


def get_download_cmd(bundle: dict, directory: Path) -> list[str]:
    return _hf_download(bundle["repo"], bundle["revision"], directory.expanduser().resolve(),
                        [item["path"] for item in bundle["files"]])


def get_npu_download_cmds(entry: dict, directory: Path) -> list[list[str]]:
    """One command per repository: the model, then any borrowed device program."""
    root = directory.expanduser().resolve()
    commands = [_hf_download(entry["repo"], entry["revision"], root / "npu" / entry["id"],
                             [item["path"] for item in entry["files"]])]
    donor_id = entry.get("devices_from")
    if donor_id:
        donor = get_npu_model(donor_id)
        commands.append(_hf_download(donor["repo"], donor["revision"], root / "npu" / donor["id"],
                                     [item["path"] for item in npu_shared_device_files(entry)]))
    return commands
