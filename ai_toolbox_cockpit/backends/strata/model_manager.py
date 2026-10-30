"""Pinned AtomicChat shards and Strata-specific native preparation."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import re
import shutil
import sys
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


@dataclass(frozen=True)
class StrataModel:
    id: str
    name: str
    repo: str
    revision: str
    directory: str
    files: tuple[tuple[str, int], ...]

    @classmethod
    def from_entry(cls, entry):
        for key in ('id', 'name', 'repo', 'revision', 'directory'):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f'Strata model requires {key}')
        if not re.fullmatch(r'[0-9a-f]{40}', entry['revision']):
            raise ValueError('Strata models require an immutable revision')
        paths = [entry['directory']] + [item['path'] for item in entry['files']]
        for path in paths:
            if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or any(c in path for c in '\\:\n\r\0') or path.startswith('-'):
                raise ValueError('Strata model paths must be safe relative paths')
        files = tuple((item['path'], item['size_bytes']) for item in entry['files'])
        if len(files) != 33 or len({p for p, _ in files}) != 33 or any(type(n) is not int or n <= 0 for _, n in files):
            raise ValueError('Select the complete qualified 33-shard AtomicChat artifact')
        return cls(*(entry[k] for k in ('id','name','repo','revision','directory')), files)

    @property
    def size_bytes(self):
        return sum(size for _,size in self.files)


def load_models():
    from ai_toolbox_cockpit.catalog import load_model_catalog
    return [StrataModel.from_entry(e) for e in load_model_catalog().backends['strata'].entries]


def get_model(model_id):
    for model in load_models():
        if model.id == model_id:return model
    raise ValueError('Select the qualified Strata Flash Next artifact')


def get_models_dir():
    return Path(get_backend_settings('strata').get('models_dir','~/models/strata')).expanduser()


def save_models_dir(value):
    if not value.strip():return False
    try:
        path=Path(value).expanduser().resolve();path.mkdir(parents=True,exist_ok=True)
        save_backend_settings('strata',{'models_dir':str(path)})
        return True
    except OSError:return False


def incomplete_files(model, root):
    missing=[]
    for path,size in model.files:
        file=Path(root)/model.directory/path
        if not file.is_file() or file.stat().st_size != size:missing.append({'path':path,'size_bytes':size})
    return missing


def model_root(model,root):
    path=(Path(root).expanduser()/model.directory).resolve()
    if any(c in str(path) for c in ':\n\r\0'):raise ValueError('Model directory cannot contain colons or control characters')
    return path


def prepared(model,root,mtp=True):
    base=model_root(model,root)
    try:
        receipt=json.loads((base/'preparation.json').read_text())
        if receipt.get('main_revision') != model.revision or receipt.get('mtp_revision') != 'de4b8e4d43b917e7706784d8bb445c9af86a3540':return False
        files=receipt['files']
        required=['pack/index.txt','pack/tokenizer/vocab.json','pack/tokenizer/chat_template.jinja']
        if mtp:required+=['mtp/rt/dense.txt','mtp/rt/dense.bin','mtp/rt/experts.bin']
        if any(p not in files for p in required):return False
        for path,size in files.items():
            part=PurePosixPath(path)
            if part.is_absolute() or '..' in part.parts or type(size) is not int or size<=0:return False
            if not (base/path).is_file() or (base/path).stat().st_size!=size:return False
        return not incomplete_files(model,root)
    except (OSError,ValueError,KeyError,TypeError):return False


def download_command(model,root):
    sibling=Path(sys.executable).with_name('hf')
    hf=str(sibling) if sibling.is_file() else shutil.which('hf') or 'hf'
    return [hf,'download',model.repo,*[p for p,_ in model.files],'--revision',model.revision,'--local-dir',str(model_root(model,root))]


def preparation_command(model,root,engine,image):
    if engine not in {'podman','docker'}:raise ValueError('Select Podman or Docker')
    if not image or image.startswith('-'):raise ValueError('Select the Strata image')
    if incomplete_files(model,root):raise ValueError('Download all model shards before preparation')
    command=[engine,'run','--rm','-it','--name','strata-cockpit-prepare']
    if engine=='podman':command+=['--userns=keep-id','--security-opt','label=disable']
    command+=['-e','HOME=/tmp','-e','HF_TOKEN','-v',f'{model_root(model,root)}:/models',image,'strata-prepare','--model',f'/models/{model.files[0][0]}','--output','/models']
    return command
