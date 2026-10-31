"""DeepSeek V4.1 GSQ-RCO artifact, metadata pack and four-R9700 profile."""
from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman


def relative(path):
    if (not isinstance(path, str) or not path or path.startswith('-')
            or PurePosixPath(path).is_absolute() or '..' in path.split('/')
            or any(c in path for c in '\\:\n\r\0')):
        raise ValueError('Strata artifact paths must be safe relative paths.')
    return path


def artifact_files(items):
    files = tuple((relative(item['path']), item['size_bytes']) for item in items)
    if len({path for path, _ in files}) != len(files) or any(type(size) is not int or size <= 0 for _, size in files):
        raise ValueError('Strata artifacts require unique paths and positive exact byte sizes.')
    return files


@dataclass(frozen=True)
class StrataDsv41Model:
    id: str
    name: str
    repo: str
    revision: str
    directory: str
    files: tuple[tuple[str, int], ...]
    auxiliary_repo: str
    auxiliary_revision: str
    auxiliary_directory: str
    auxiliary_files: tuple[tuple[str, int], ...]
    pack_directory: str
    family: str = 'deepseek41'
    platform_id: str = 'r9700'

    @classmethod
    def from_entry(cls, entry):
        values = {key: entry[key] for key in cls.__dataclass_fields__ if key in entry}
        values['files'] = artifact_files(entry['files'])
        values['auxiliary_files'] = artifact_files(entry['auxiliary_files'])
        model = cls(**values)
        for revision in (model.revision, model.auxiliary_revision):
            if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
                raise ValueError('Strata artifacts require immutable repository revisions.')
        for repo in (model.repo, model.auxiliary_repo):
            if not isinstance(repo, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', repo):
                raise ValueError('Invalid Hugging Face repository.')
        for path in (model.directory, model.auxiliary_directory, model.pack_directory):
            relative(path)
        shards = {f'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00004.gguf' for i in range(1, 5)}
        if {path for path, _ in model.files} != shards | {'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'}:
            raise ValueError('Select the four GSQ-RCO IQ2_XXS shards and matched Engram artifact.')
        auxiliary = {'dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf',
                     'UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'}
        if {path for path, _ in model.auxiliary_files} != auxiliary:
            raise ValueError('Select the matched DSpark drafter and Engram metadata sidecar.')
        if model.family != 'deepseek41' or model.platform_id != 'r9700':
            raise ValueError('This DeepSeek V4.1 profile requires four R9700 GPUs.')
        return model

    @property
    def size_bytes(self):
        return sum(size for _, size in (*self.files, *self.auxiliary_files))


def root_path(directory):
    root = Path(directory).expanduser().resolve()
    if any(c in str(root) for c in ':\n\r\0'):
        raise ValueError('Model directory cannot contain colons or control characters.')
    return root


def incomplete_files(model, directory):
    root = root_path(directory)
    missing = []
    for folder, files in [(model.directory, model.files), (model.auxiliary_directory, model.auxiliary_files)]:
        for name, size in files:
            path = root / folder / name
            try:
                ready = path.is_file() and path.stat().st_size == size
            except OSError:
                ready = False
            if not ready:
                missing.append({'path': folder + '/' + name, 'size_bytes': size})
    return missing


def pack_root(model, directory):
    return root_path(directory) / '.prepared' / model.pack_directory


def prepared(model, directory):
    base = pack_root(model, directory)
    try:
        receipt = json.loads((base / 'preparation.json').read_text())
        if receipt.get('main_revision') != model.revision or receipt.get('aux_revision') != model.auxiliary_revision:
            return False
        files = receipt['files']
        required = {'ds4.txt', 'tensors.txt', 'engram.bin', 'expert-profile.bin',
                    'tokenizer/vocab.json', 'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja'}
        if not isinstance(files, dict) or not required.issubset(files):
            return False
        for name, size in files.items():
            relative(name)
            path = base / name
            if type(size) is not int or size <= 0 or not path.is_file() or path.stat().st_size != size:
                return False
        return not incomplete_files(model, directory)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def hf_command(repo, revision, files, destination):
    sibling = Path(sys.executable).with_name('hf')
    hf = str(sibling) if sibling.is_file() else shutil.which('hf') or 'hf'
    return [hf, 'download', repo, *(name for name, _ in files), '--revision', revision,
            '--local-dir', str(destination)]


def download_command(model, directory):
    return [sys.executable, '-m', 'ai_toolbox_cockpit.backends.strata.dsv41',
            '--model', model.id, '--directory', str(root_path(directory))]


def preparation_command(model, directory, engine, image):
    if engine not in {'podman', 'docker'} or not image or image.startswith('-'):
        raise ValueError('Select the Strata container engine and image.')
    if incomplete_files(model, directory):
        raise ValueError('Download the complete DeepSeek V4.1 target and auxiliary artifacts first.')
    if pack_root(model, directory).exists():
        raise ValueError('Native pack already exists. Scan its state before preparing another pack.')
    root = root_path(directory)
    command = [engine, 'run', '--rm', '-it', '--name', 'strata-cockpit-prepare']
    if engine == 'podman':
        command += ['--userns=keep-id', '--security-opt', 'label=disable']
    return command + ['-e', 'HOME=/tmp', '-v', f'{root}:/models:ro',
        '-v', f'{root / ".prepared"}:/prepared:rw', image, 'strata-prepare',
        '--model', '/models/' + model.directory, '--auxiliary', '/models/' + model.auxiliary_directory,
        '--output', '/prepared/' + model.pack_directory, '--main-revision', model.revision,
        '--aux-revision', model.auxiliary_revision]


def build_server_cmd(*, engine, image, engine_args, platform_id, model, directory,
                     host='127.0.0.1', port=18920, context_size=8192, speculation='baseline'):
    if platform_id != 'r9700':
        raise ValueError('This DeepSeek V4.1 profile requires four R9700 GPUs.')
    if engine not in {'podman', 'docker'} or not image or image.startswith('-'):
        raise ValueError('Select the Strata container engine and image.')
    if type(context_size) is not int or not 2048 <= context_size <= 68608:
        raise ValueError('Context must be between 2048 and 68608 tokens.')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid API port.')
    address = ipaddress.ip_address('127.0.0.1' if host == 'localhost' else host)
    if speculation not in {'baseline', 'dspark'}:
        raise ValueError('Select target-only or DSpark decoding.')
    if not prepared(model, directory):
        raise ValueError('Download and prepare DeepSeek V4.1 in Models first.')
    bind = f'[{address}]' if address.version == 6 else str(address)
    command = [engine, 'run', '--rm', '-it', '--name', 'strata-cockpit-server']
    command += upgrade_groups_for_podman(engine, list(engine_args))
    if engine == 'podman':
        command += ['--security-opt', 'label=disable', '--userns=keep-id']
    command += ['-e', 'HOME=/tmp', '-e', 'STRATA_DS4_SLOT_KEEP=0', '--shm-size', '4g',
        '-p', f'{bind}:{port}:{port}', '-v', f'{root_path(directory)}:/models:ro', image,
        'strata-serve', '--pack', '/models/.prepared/' + model.pack_directory,
        '--context', str(context_size), '--decoding', speculation, '--host', '0.0.0.0', '--port', str(port)]
    return command


def main():
    import argparse
    from .model_manager import get_model
    parser = argparse.ArgumentParser(description='Download the pinned Strata DeepSeek V4.1 artifact set.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    model = get_model(args.model)
    if not isinstance(model, StrataDsv41Model):
        parser.error('Select a DeepSeek V4.1 model.')
    root = root_path(args.directory)
    for repo, revision, folder, files in [(model.repo, model.revision, model.directory, model.files),
            (model.auxiliary_repo, model.auxiliary_revision, model.auxiliary_directory, model.auxiliary_files)]:
        (root / folder).mkdir(parents=True, exist_ok=True)
        subprocess.run(hf_command(repo, revision, files, root / folder), check=True)
    if incomplete_files(model, root):
        raise ValueError('One or more required Strata artifacts are incomplete.')
    print('DeepSeek V4.1 model, Engram, metadata and DSpark files are ready.', flush=True)


if __name__ == '__main__':
    main()
