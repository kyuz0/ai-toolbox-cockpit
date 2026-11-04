"""Curated Gufo GGUF bundles and exact Hugging Face download commands."""

import shutil
import sys
from collections import defaultdict
from pathlib import Path

from ai_toolbox_cockpit.huggingface import huggingface_download_command
from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def get_models_dir() -> Path:
    backend = load_model_catalog().backends["gufo"]
    default = backend.storage["default"]
    value = get_backend_settings("gufo").get("models_dir", default)
    return Path(str(value)).expanduser().resolve()


def save_models_dir(value: str) -> bool:
    if not value.strip():
        return False
    try:
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return False
    return save_backend_settings("gufo", {"models_dir": str(path)})


def load_models() -> tuple[dict, ...]:
    return load_model_catalog().backends["gufo"].entries


def get_model(model_id: str) -> dict:
    for entry in load_models():
        if entry["id"] == model_id:
            return entry
    raise ValueError("Select a curated Gufo model profile.")


def search_roots() -> tuple[Path, ...]:
    backend = load_model_catalog().backends["gufo"]
    configured = [get_models_dir(), *(
        Path(str(value)).expanduser().resolve()
        for value in backend.config.get("search_dirs", [])
    )]
    return tuple(dict.fromkeys(configured))


def _find_file(relative: str, expected_size: int, directory: str) -> Path | None:
    relative_path = Path(relative)
    for root in search_roots():
        candidates = (
            root / directory / relative_path,
            root / relative_path,
            root / relative_path.name,
        )
        for candidate in dict.fromkeys(candidates):
            try:
                if candidate.is_file() and candidate.stat().st_size == expected_size:
                    return candidate.resolve()
            except OSError:
                continue
    return None


def resolved_files(model: dict) -> dict:
    targets = {
        item["path"]: _find_file(
            item["path"], item["size_bytes"], model["directory"]
        )
        for item in model["files"]
    }
    speculation = model.get("speculation")
    sidecar = (
        _find_file(
            speculation["path"], speculation["size_bytes"],
            speculation.get("directory", model["directory"]),
        )
        if speculation else None
    )
    vision = model.get("vision")
    projector = (
        _find_file(vision["path"], vision["size_bytes"], vision["directory"])
        if vision else None
    )
    return {
        "vision": projector,
        "targets": targets,
        "model": targets.get(model["model_path"]),
        "sidecar": sidecar,
    }


def missing_files(
    model: dict, *, include_speculation: bool = True, include_vision: bool = False,
) -> list[str]:
    resolved = resolved_files(model)
    missing = [relative for relative, path in resolved["targets"].items() if path is None]
    speculation = model.get("speculation")
    if include_speculation and speculation and resolved["sidecar"] is None:
        missing.append(speculation["path"])
    if include_vision and model.get("vision") and resolved["vision"] is None:
        missing.append(model["vision"]["path"])
    return missing


def model_size(model: dict, *, include_vision: bool = False) -> int:
    total = sum(item["size_bytes"] for item in model["files"])
    if model.get("speculation"):
        total += model["speculation"]["size_bytes"]
    if include_vision and model.get("vision"):
        total += model["vision"]["size_bytes"]
    return total


def model_status(model: dict) -> str:
    resolved = resolved_files(model)
    target_missing = sum(path is None for path in resolved["targets"].values())
    if target_missing:
        return f"{target_missing} target file(s) missing / incomplete"
    speculation = model.get("speculation")
    if speculation and resolved["sidecar"] is None:
        status = f"Baseline ready; {speculation['mode'].upper()} sidecar missing"
    elif speculation:
        status = f"Ready: baseline + {speculation['mode'].upper()}"
    else:
        status = "Ready: baseline"
    if model.get("vision"):
        status += "; vision " + ("ready" if resolved["vision"] is not None else "projector missing")
    return status


def get_download_commands(
    model: dict, directory: Path, *, include_vision: bool = False, missing_only: bool = False,
) -> list[list[str]]:
    executable = Path(sys.executable).with_name("hf")
    hf = str(executable) if executable.is_file() else (shutil.which("hf") or "hf")
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    resolved = resolved_files(model) if missing_only else None
    groups[(model["repo"], model["revision"], model["directory"])].extend(
        item["path"] for item in model["files"]
        if resolved is None or resolved["targets"][item["path"]] is None
    )
    speculation = model.get("speculation")
    if speculation and (resolved is None or resolved["sidecar"] is None):
        groups[(
            speculation["repo"], speculation["revision"],
            speculation.get("directory", model["directory"]),
        )].append(
            speculation["path"]
        )
    if include_vision:
        vision = model.get("vision")
        if not vision:
            raise ValueError("The selected Gufo model has no vision profile.")
        if resolved is None or resolved["vision"] is None:
            groups[(vision["repo"], vision["revision"], vision["directory"])].append(vision["path"])
    return [
        huggingface_download_command([
            hf, "download", repo, *paths,
            "--revision", revision,
            "--local-dir", str(directory.expanduser().resolve() / target_directory),
        ])
        for (repo, revision, target_directory), paths in groups.items() if paths
    ]
