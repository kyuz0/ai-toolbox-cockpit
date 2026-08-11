"""DS4 exact-file model storage and Hugging Face command helpers."""

import os
import sys
from pathlib import Path

from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def get_models_dir() -> Path:
    value = get_backend_settings("ds4").get("models_dir", "~/ds4")
    return Path(str(value)).expanduser()


def save_models_dir(path_str: str) -> bool:
    try:
        Path(path_str).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return save_backend_settings("ds4", {"models_dir": path_str})


def scan_local_models() -> list[dict]:
    directory = get_models_dir()
    if not directory.exists():
        return []
    return [
        {"name": path.name, "path": str(path)}
        for path in sorted(directory.glob("*.gguf"), key=lambda item: item.name.lower())
        if path.is_file()
    ]


def is_model_downloaded(filename: str) -> bool:
    return (get_models_dir() / filename).is_file()


def get_download_cmd(repo: str, filename: str) -> list[str]:
    executable = os.path.join(os.path.dirname(sys.executable), "hf")
    if not os.path.exists(executable):
        executable = "hf"
    return [executable, "download", repo, filename, "--local-dir", str(get_models_dir())]
