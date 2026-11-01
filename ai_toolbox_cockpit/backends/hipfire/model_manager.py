"""Pinned hipfire MQ2R model downloads; no implicit format conversion."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


@dataclass(frozen=True)
class HipfireModel:
    id: str
    name: str
    repo: str
    revision: str
    directory: str
    filename: str
    size_bytes: int
    context_size: int = 67840

    @classmethod
    def from_entry(cls, entry: dict) -> 'HipfireModel':
        model = cls(**{key: entry[key] for key in cls.__dataclass_fields__ if key in entry})
        if not isinstance(model.revision, str) or not re.fullmatch(r'[0-9a-f]{40}', model.revision):
            raise ValueError('hipfire models require an immutable repository revision.')
        if not isinstance(model.repo, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', model.repo):
            raise ValueError('Invalid Hugging Face repository.')
        for path in (model.directory, model.filename):
            if (not isinstance(path, str) or not path or path.startswith('-')
                    or PurePosixPath(path).is_absolute() or '..' in path.split('/')
                    or any(c in path for c in '\\:\n\r\0')):
                raise ValueError('hipfire model paths must be safe relative paths.')
        if not model.filename.endswith('.mq2r') or len(PurePosixPath(model.filename).parts) != 1:
            raise ValueError('Select a native MQ2R model file.')
        if type(model.size_bytes) is not int or model.size_bytes <= 0:
            raise ValueError('hipfire models require an exact positive file size.')
        if type(model.context_size) is not int or not 2048 <= model.context_size <= 67840:
            raise ValueError('Context exceeds the qualified hipfire profile.')
        return model


def get_models_dir() -> Path:
    default = load_model_catalog().backends['hipfire'].storage['default']
    return Path(get_backend_settings('hipfire').get('models_dir', default)).expanduser().resolve()


def save_models_dir(value: str) -> bool:
    if not value.strip():
        return False
    try:
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return False
    return save_backend_settings('hipfire', {'models_dir': str(path)})


def load_models() -> tuple[HipfireModel, ...]:
    return tuple(HipfireModel.from_entry(entry) for entry in load_model_catalog().backends['hipfire'].entries)


def get_model(model_id: str) -> HipfireModel:
    for model in load_models():
        if model.id == model_id:
            return model
    raise ValueError('Select a curated hipfire MQ2R model.')


def resolved_path(model: HipfireModel, directory: Path | None = None) -> Path | None:
    root = directory if directory is not None else get_models_dir()
    path = root / model.directory / model.filename
    try:
        return path.resolve() if path.is_file() and path.stat().st_size == model.size_bytes else None
    except OSError:
        return None


def incomplete_files(model: HipfireModel | dict, directory: Path) -> list[dict]:
    if isinstance(model, dict):
        model = HipfireModel.from_entry(model)
    return [] if resolved_path(model, directory) else [{'size_bytes': model.size_bytes}]


def model_status(model: HipfireModel, directory: Path) -> str:
    return 'Ready' if resolved_path(model, directory) else 'Missing / incomplete'


def download_command(model: HipfireModel, directory: Path) -> list[str]:
    sibling = Path(sys.executable).with_name('hf')
    hf = str(sibling) if sibling.is_file() else (shutil.which('hf') or 'hf')
    return [hf, 'download', model.repo, model.filename, '--revision', model.revision,
            '--local-dir', str(directory.expanduser().resolve() / model.directory)]
