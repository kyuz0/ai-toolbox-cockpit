"""Pinned Affinity parts, assembled target, DSpark sidecar and tokenizer."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def safe_relative(value: str) -> str:
    if (not isinstance(value, str) or not value or value.startswith('-')
            or PurePosixPath(value).is_absolute() or '..' in value.split('/')
            or any(c in value for c in '\\:\n\r\0')):
        raise ValueError('Affinity artifact paths must be safe relative paths.')
    return value


@dataclass(frozen=True)
class AffinityModel:
    id: str
    name: str
    repo: str
    revision: str
    directory: str
    filename: str
    size_bytes: int
    parts: tuple[tuple[str, int], ...]
    draft_filename: str
    draft_size_bytes: int
    tokenizer_repo: str
    tokenizer_revision: str
    tokenizer_directory: str
    tokenizer_filename: str
    tokenizer_size_bytes: int
    context_size: int = 67840

    @classmethod
    def from_entry(cls, entry: dict) -> 'AffinityModel':
        values = {key: entry[key] for key in cls.__dataclass_fields__ if key in entry}
        values['parts'] = tuple((part['path'], part['size_bytes']) for part in entry['parts'])
        model = cls(**values)
        for revision in (model.revision, model.tokenizer_revision):
            if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
                raise ValueError('Affinity artifacts require immutable repository revisions.')
        for repo in (model.repo, model.tokenizer_repo):
            if not isinstance(repo, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', repo):
                raise ValueError('Invalid Hugging Face repository.')
        for path in (model.directory, model.filename, model.draft_filename,
                     model.tokenizer_directory, model.tokenizer_filename, *(p for p, _ in model.parts)):
            safe_relative(path)
        if (not model.filename.endswith('.aff') or not model.draft_filename.endswith('.dspark.aff')
                or model.tokenizer_filename != 'tokenizer.json'):
            raise ValueError('Select the native Affinity target, DSpark and tokenizer formats.')
        paths = [path for path, _ in model.parts]
        if not paths or len(paths) != len(set(paths)) or model.filename in paths or model.draft_filename in paths:
            raise ValueError('Affinity parts must be distinct from the assembled model and drafter.')
        sizes = [model.size_bytes, model.draft_size_bytes, model.tokenizer_size_bytes,
                 *(size for _, size in model.parts)]
        if any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError('Affinity artifacts require exact positive file sizes.')
        if sum(size for _, size in model.parts) != model.size_bytes:
            raise ValueError('Assembled Affinity size must match its ordered parts.')
        if type(model.context_size) is not int or not 2048 <= model.context_size <= 67840:
            raise ValueError('Context exceeds the qualified Affinity profile.')
        return model


def get_models_dir() -> Path:
    default = load_model_catalog().backends['affinity'].storage['default']
    return Path(get_backend_settings('affinity').get('models_dir', default)).expanduser().resolve()


def save_models_dir(value: str) -> bool:
    if not value.strip():
        return False
    try:
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return False
    return save_backend_settings('affinity', {'models_dir': str(path)})


def load_models() -> tuple[AffinityModel, ...]:
    return tuple(AffinityModel.from_entry(entry) for entry in load_model_catalog().backends['affinity'].entries)


def get_model(model_id: str) -> AffinityModel:
    for model in load_models():
        if model.id == model_id:
            return model
    raise ValueError('Select a curated Affinity model.')


def complete(path: Path, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size
    except OSError:
        return False


def artifact_paths(model: AffinityModel, directory: Path) -> tuple[Path, Path, Path]:
    root = directory.expanduser().resolve()
    return (root / model.directory / model.filename,
            root / model.directory / model.draft_filename,
            root / model.tokenizer_directory / model.tokenizer_filename)


def incomplete_files(model: AffinityModel | dict, directory: Path) -> list[dict]:
    if isinstance(model, dict):
        model = AffinityModel.from_entry(model)
    target, draft, tokenizer = artifact_paths(model, directory)
    files = [(target, model.size_bytes), (draft, model.draft_size_bytes),
             (tokenizer, model.tokenizer_size_bytes)]
    return [{'path': str(path), 'size_bytes': size} for path, size in files if not complete(path, size)]


def model_status(model: AffinityModel, directory: Path) -> str:
    return 'Ready' if not incomplete_files(model, directory) else 'Missing / incomplete'


def download_command(model: AffinityModel, directory: Path) -> list[str]:
    return [sys.executable, '-m', 'ai_toolbox_cockpit.backends.affinity.download',
            '--model', model.id, '--directory', str(directory.expanduser().resolve())]


def hf_command(repo: str, revision: str, files: list[str], directory: Path) -> list[str]:
    sibling = Path(sys.executable).with_name('hf')
    hf = str(sibling) if sibling.is_file() else (shutil.which('hf') or 'hf')
    return [hf, 'download', repo, *files, '--revision', revision, '--local-dir', str(directory)]


def download_space_bytes(model: AffinityModel, directory: Path) -> int:
    """Missing downloads plus assembly space; parts remain reusable after preparation."""
    target, draft, tokenizer = artifact_paths(model, directory)
    required = 0
    if not complete(target, model.size_bytes):
        required += model.size_bytes
        required += sum(size for name, size in model.parts if not complete(target.parent / name, size))
    if not complete(draft, model.draft_size_bytes):
        required += model.draft_size_bytes
    if not complete(tokenizer, model.tokenizer_size_bytes):
        required += model.tokenizer_size_bytes
    return required
