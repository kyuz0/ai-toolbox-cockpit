"""Resumable HF downloads and atomic native target assembly; never scans full-file hashes."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .model_manager import (AffinityModel, artifact_paths, complete, get_model,
                            hf_command, incomplete_files)


def assemble_target(model: AffinityModel, directory: Path) -> Path:
    target, _, _ = artifact_paths(model, directory)
    if complete(target, model.size_bytes):
        print(f'Reusing complete native target: {target}', flush=True)
        return target
    parts = [target.parent / name for name, _ in model.parts]
    for part, (_, size) in zip(parts, model.parts):
        if not complete(part, size):
            raise ValueError(f'Missing or incomplete Affinity part: {part}')
    if shutil.disk_usage(target.parent).free < model.size_bytes:
        raise ValueError('Insufficient space to assemble the native target while preserving downloaded parts.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f'.{target.name}.assembling-', delete=False) as output:
            temporary = Path(output.name)
            for index, part in enumerate(parts, 1):
                print(f'Assembling part {index}/{len(parts)}: {part.name}', flush=True)
                with part.open('rb') as source:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != model.size_bytes:
            raise ValueError('Assembled target size differs from the pinned artifact.')
        # A concurrent preparation may already have completed it. Preserve that artifact.
        if not complete(target, model.size_bytes):
            os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(model: AffinityModel, directory: Path) -> None:
    target, draft, tokenizer = artifact_paths(model, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    files = [] if complete(target, model.size_bytes) else [name for name, _ in model.parts]
    if not complete(draft, model.draft_size_bytes):
        files.append(model.draft_filename)
    if files:
        subprocess.run(hf_command(model.repo, model.revision, files, target.parent), check=True)
    if not complete(tokenizer, model.tokenizer_size_bytes):
        subprocess.run(hf_command(model.tokenizer_repo, model.tokenizer_revision,
                                  [model.tokenizer_filename], tokenizer.parent), check=True)
    assemble_target(model, directory)
    if incomplete_files(model, directory):
        raise ValueError('Required Affinity target, drafter or tokenizer is incomplete.')
    print('Affinity target, DSpark sidecar and tokenizer are ready.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    download(get_model(args.model), args.directory.expanduser().resolve())


if __name__ == '__main__':
    main()
