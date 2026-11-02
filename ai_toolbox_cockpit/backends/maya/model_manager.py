"""Pinned Maya GGUF bundles and native model preparation."""
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def relative_path(value):
    if not isinstance(value, str) or not value or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts or any(c in value for c in '\\:\n\r\0') or value.startswith('-'):
        raise ValueError('Maya artifacts require safe relative paths')
    return value


@dataclass(frozen=True)
class MayaModel:
    id: str
    name: str
    family: str
    repo: str
    revision: str
    directory: str
    files: tuple[tuple[str, int], ...]
    model_path: str
    pack_directory: str
    drafter: str | None = None

    @classmethod
    def from_entry(cls, entry):
        for key in ('id', 'name', 'repo', 'revision', 'family'):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f'Maya model requires {key}')
        if entry['family'] not in ('glm', 'deepseek') or not re.fullmatch(r'[0-9a-f]{40}', entry['revision']):
            raise ValueError('Maya requires a known family and immutable model revision')
        for key in ('directory', 'model_path', 'pack_directory'):
            relative_path(entry[key])
        files = tuple((relative_path(f['path']), f['size_bytes']) for f in entry['files'])
        if not files or len({p for p, _ in files}) != len(files) or any(type(n) is not int or n <= 0 for _, n in files):
            raise ValueError('Maya requires complete, unique files with positive sizes')
        paths = {p for p, _ in files}
        if entry['family'] == 'glm' and entry['model_path'] not in paths:
            raise ValueError('GLM preparation requires its first GGUF shard')
        if entry['family'] == 'deepseek' and not any(p.startswith(entry['model_path'] + '/') and p.endswith('.gguf') for p in paths):
            raise ValueError('DeepSeek preparation requires its GGUF directory')
        drafter = entry.get('drafter')
        if entry['family'] == 'deepseek':
            relative_path(drafter)
            if drafter not in paths or str(PurePosixPath(drafter).with_suffix('.bin')) not in paths:
                raise ValueError('DeepSeek requires the matched DSpark JSON and binary')
        elif drafter:
            raise ValueError('GLM uses its included MTP layer')
        return cls(*(entry[k] for k in ('id', 'name', 'family', 'repo', 'revision', 'directory')), files, entry['model_path'], entry['pack_directory'], drafter)

    @property
    def size_bytes(self):
        return sum(n for _, n in self.files)

    @property
    def modes(self):
        return [('Target only', 'baseline'), ('MTP · up to 3 draft tokens', 'mtp')] if self.family == 'glm' else [('Target only', 'baseline'), ('DSpark · up to 5 draft tokens', 'dspark')]


def load_models():
    from ai_toolbox_cockpit.catalog import load_model_catalog
    return [MayaModel.from_entry(e) for e in load_model_catalog().backends['maya'].entries]


def get_model(model_id):
    for model in load_models():
        if model.id == model_id:
            return model
    raise ValueError('Select a Maya model')


def get_models_dir():
    return Path(get_backend_settings('maya').get('models_dir', '~/models')).expanduser()


def save_models_dir(value):
    if not value.strip():
        return False
    try:
        root = Path(value).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        save_backend_settings('maya', {'models_dir': str(root)})
        return True
    except OSError:
        return False


def model_root(model, directory):
    root = (Path(directory).expanduser() / model.directory).resolve()
    if any(c in str(root) for c in ':\n\r\0'):
        raise ValueError('Models directory cannot contain colons or control characters')
    return root


def incomplete_files(model, directory):
    root = model_root(model, directory)
    return [p for p, n in model.files if not (root / p).is_file() or (root / p).stat().st_size != n]


def pack_files(model):
    return ['tokenizer/vocab.json', 'tokenizer/chat_template.jinja'] + (['index.txt', 'native_experts.txt', 'dense.bin'] if model.family == 'glm' else ['ds4.txt', 'tensors.txt', 'engram.bin'])


def record_prepared(model, directory):
    root = model_root(model, directory)
    pack = root / model.pack_directory
    if incomplete_files(model, directory):
        raise ValueError('Maya source weights are incomplete')
    sizes = {p: (pack / p).stat().st_size for p in pack_files(model)}
    if any(n <= 0 for n in sizes.values()):
        raise ValueError('Maya preparation produced empty artifacts')
    receipt = {'model': model.id, 'revision': model.revision, 'source_root': str(root), 'files': sizes}
    (pack / 'cockpit-preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')


def prepared(model, directory):
    try:
        root = model_root(model, directory)
        pack = root / model.pack_directory
        receipt = json.loads((pack / 'cockpit-preparation.json').read_text())
        return (receipt.get('model') == model.id and receipt.get('revision') == model.revision
                and receipt.get('source_root') == str(root) and not incomplete_files(model, directory)
                and all(type(receipt['files'].get(p)) is int and receipt['files'][p] > 0
                        and (pack / p).stat().st_size == receipt['files'][p] for p in pack_files(model)))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download_command(model, directory):
    sibling = Path(sys.executable).with_name('hf')
    hf = str(sibling) if sibling.is_file() else shutil.which('hf') or 'hf'
    return [hf, 'download', model.repo, *[p for p, _ in model.files], '--revision', model.revision,
            '--local-dir', str(model_root(model, directory))]


def preparation_command(model, directory, engine, image):
    if engine not in ('podman', 'docker') or not image or image.startswith('-'):
        raise ValueError('Select a container engine and Maya image')
    if incomplete_files(model, directory):
        raise ValueError('Download all selected model files before preparation')
    root = model_root(model, directory)
    command = [engine, 'run', '--rm', '-it', '--name', 'maya-cockpit-prepare']
    if engine == 'podman':
        command += ['--userns=keep-id', '--security-opt', 'label=disable']
    # Maya packs refer to source paths. Keep their absolute paths identical when serving.
    command += ['-e', 'HOME=/tmp', '-v', f'{root}:{root}', image, 'maya-prepare',
                '--family', model.family, '--model', str(root / model.model_path),
                '--output', str(root / model.pack_directory)]
    if model.drafter:
        command += ['--drafter', str(root / model.drafter)]
    return command
