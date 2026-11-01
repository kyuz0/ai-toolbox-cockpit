"""Pinned standalone Radiance containers, with bundled vision and drafting tensors."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from ai_toolbox_cockpit.catalog import load_model_catalog
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


@dataclass(frozen=True)
class RadianceServingProfile:
    id: str
    name: str
    tensor_parallel: int
    max_sessions: int = 1
    checkpoint_slots: int | None = None
    description: str = ''

    @classmethod
    def from_entry(cls, entry: dict) -> 'RadianceServingProfile':
        if not isinstance(entry, dict):
            raise ValueError('Invalid Radiance serving profile.')
        try:
            profile = cls(**entry)
        except TypeError as error:
            raise ValueError('Invalid Radiance serving profile fields.') from error
        if not isinstance(profile.id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', profile.id):
            raise ValueError('Invalid Radiance serving profile ID.')
        if not isinstance(profile.name, str) or not profile.name.strip():
            raise ValueError('Radiance serving profiles require a visible name.')
        if type(profile.tensor_parallel) is not int or profile.tensor_parallel not in {1, 2, 4}:
            raise ValueError('Invalid qualified Radiance GPU layout.')
        if type(profile.max_sessions) is not int or not 1 <= profile.max_sessions <= 16:
            raise ValueError('Invalid qualified Radiance concurrency limit.')
        if profile.checkpoint_slots is not None and (type(profile.checkpoint_slots) is not int or profile.checkpoint_slots < 1):
            raise ValueError('State checkpoint slots must be a positive integer.')
        if not isinstance(profile.description, str):
            raise ValueError('Invalid Radiance serving profile description.')
        return profile


@dataclass(frozen=True)
class RadianceModel:
    id: str
    name: str
    repo: str
    revision: str
    directory: str
    filename: str
    size_bytes: int
    family: str
    speculation: str
    draft_tokens: int
    context_size: int
    tensor_parallel: int
    vision: bool
    ngram_placement: str = 'auto'
    max_sessions: int = 1
    serving_profiles: tuple[RadianceServingProfile, ...] = ()

    def available_profiles(self) -> tuple[RadianceServingProfile, ...]:
        default = RadianceServingProfile('default', f'{self.tensor_parallel} R9700 GPU(s) · up to {self.max_sessions} sequence(s)',
                                         self.tensor_parallel, self.max_sessions)
        return (default, *self.serving_profiles)

    def serving_profile(self, profile_id: str = 'default') -> RadianceServingProfile:
        for profile in self.available_profiles():
            if profile.id == profile_id:
                return profile
        raise ValueError('Select a serving profile qualified for this model.')

    @classmethod
    def from_entry(cls, entry: dict) -> 'RadianceModel':
        values = {key: entry[key] for key in cls.__dataclass_fields__ if key in entry}
        raw_profiles = values.get('serving_profiles', ())
        if not isinstance(raw_profiles, (list, tuple)):
            raise ValueError('Invalid Radiance serving profile list.')
        profiles = tuple(RadianceServingProfile.from_entry(vars(item) if isinstance(item, RadianceServingProfile) else item)
                         for item in raw_profiles)
        if len({item.id for item in profiles}) != len(profiles) or any(item.id == 'default' for item in profiles):
            raise ValueError('Radiance serving profile IDs must be unique and cannot replace default.')
        values['serving_profiles'] = profiles
        model = cls(**values)
        if not re.fullmatch(r'[0-9a-f]{40}', model.revision):
            raise ValueError('Radiance models require an immutable repository revision.')
        for path in (model.directory, model.filename):
            if not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path or path.startswith('-'):
                raise ValueError('Radiance model paths must be safe relative paths.')
        if not model.filename.endswith('.rad') or len(PurePosixPath(model.filename).parts) != 1:
            raise ValueError('Select a compatible single-file .rad model container.')
        if type(model.size_bytes) is not int or model.size_bytes <= 0:
            raise ValueError('Radiance models require an exact positive file size.')
        if model.speculation not in {'mtp', 'dflash2'} or type(model.draft_tokens) is not int or not 1 <= model.draft_tokens <= 16:
            raise ValueError('Invalid bundled Radiance draft profile.')
        if isinstance(model.tensor_parallel, bool) or model.tensor_parallel not in {1, 2, 4}:
            raise ValueError('Invalid qualified Radiance GPU layout.')
        if model.ngram_placement not in {'auto', 'ram'}:
            raise ValueError('Invalid qualified n-gram placement.')
        if type(model.max_sessions) is not int or not 1 <= model.max_sessions <= 16:
            raise ValueError('Invalid qualified Radiance concurrency limit.')
        if type(model.context_size) is not int or not 1 <= model.context_size <= 262144:
            raise ValueError('Invalid Radiance context bound.')
        return model


def get_models_dir() -> Path:
    default = load_model_catalog().backends['radiance'].storage['default']
    return Path(get_backend_settings('radiance').get('models_dir', default)).expanduser().resolve()


def save_models_dir(value: str) -> bool:
    if not value.strip():
        return False
    try:
        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return False
    return save_backend_settings('radiance', {'models_dir': str(path)})


def load_models() -> tuple[RadianceModel, ...]:
    return tuple(RadianceModel.from_entry(entry) for entry in load_model_catalog().backends['radiance'].entries)


def get_model(model_id: str) -> RadianceModel:
    for model in load_models():
        if model.id == model_id:
            return model
    raise ValueError('Select a curated standalone Radiance model.')


def resolved_path(model: RadianceModel, directory: Path | None = None) -> Path | None:
    root = directory if directory is not None else get_models_dir()
    for candidate in (root / model.directory / model.filename, root / model.filename):
        try:
            if candidate.is_file() and candidate.stat().st_size == model.size_bytes:
                return candidate.resolve()
        except OSError:
            continue
    return None


def model_status(model: RadianceModel, directory: Path | None = None) -> str:
    return 'Ready' if resolved_path(model, directory) else 'Missing / incomplete'


def download_command(model: RadianceModel, directory: Path) -> list[str]:
    sibling = Path(sys.executable).with_name('hf')
    hf = str(sibling) if sibling.is_file() else (shutil.which('hf') or 'hf')
    return [hf, 'download', model.repo, model.filename, '--revision', model.revision,
            '--local-dir', str(directory.expanduser().resolve() / model.directory)]
