"""Pure four-GPU, single-request Affinity HTTP-v2 command builder."""
import ipaddress
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman
from .model_manager import AffinityModel, artifact_paths, complete

CONTAINER_NAME = 'affinity-cockpit-server'


def build_server_cmd(*, engine: str, image: str, engine_args: list[str], platform_id: str,
                     model: AffinityModel, models_dir: Path, cache_dir: Path,
                     host: str = '127.0.0.1', port: int = 18915,
                     context_size: int = 8192, speculation: str = 'baseline') -> list[str]:
    if platform_id != 'r9700':
        raise ValueError('Affinity is qualified for four R9700 GPUs only.')
    if engine not in {'podman', 'docker'} or not image or image.startswith('-'):
        raise ValueError('Select the Affinity container engine and image.')
    address = ipaddress.ip_address('127.0.0.1' if host == 'localhost' else host)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535.')
    if type(context_size) is not int or not 2048 <= context_size <= model.context_size:
        raise ValueError('Context exceeds the qualified Affinity allocation.')
    if speculation not in {'baseline', 'dspark'}:
        raise ValueError('Select target-only or DSpark decoding.')
    root = models_dir.expanduser().resolve()
    cache = cache_dir.expanduser().resolve()
    for path in (root, cache):
        if any(c in str(path) for c in ':\n\r\0'):
            raise ValueError('Container mount paths cannot contain colons or control characters.')
    target, draft, tokenizer = artifact_paths(model, root)
    if not complete(target, model.size_bytes) or not complete(tokenizer, model.tokenizer_size_bytes):
        raise ValueError('Download / Repair the complete Affinity target and tokenizer first.')
    if speculation == 'dspark' and not complete(draft, model.draft_size_bytes):
        raise ValueError('Download / Repair the matching DSpark sidecar first.')
    command = [engine, 'run', '--rm', '-it', '--name', CONTAINER_NAME]
    command.extend(upgrade_groups_for_podman(engine, list(engine_args)))
    if engine == 'podman':
        command += ['--security-opt', 'label=disable', '--userns=keep-id']
    bind = f'[{address}]' if address.version == 6 else str(address)
    command += ['-p', f'{bind}:{port}:{port}', '-v', f'{root}:/models:ro', '-v', f'{cache}:/cache:rw',
                '-e', 'AFF_ALLOW_NRANK=1', '-e', 'HIP_VISIBLE_DEVICES=0,1,2,3', image,
                'affinity', '-m', '/models/' + str(target.relative_to(root)),
                '--tokenizer', '/models/' + str(tokenizer.relative_to(root)),
                '--gpus', '4', '--placement-strategy', 'static', '--host-pool-mib', '0',
                '--kv-size', str(context_size + 64), '--kv-commit', str(context_size + 64),
                '--dspark', 'on' if speculation == 'dspark' else 'off',
                '--max-tokens-floor', '0', '--raw-completions',
                '--prefix-cache-dir', '/cache', '--prefix-cache-gib', '64',
                '--host', '0.0.0.0', '--port', str(port), '--ui', 'plain']
    if speculation == 'dspark':
        command += ['--dspark-model', '/models/' + str(draft.relative_to(root))]
    return command
