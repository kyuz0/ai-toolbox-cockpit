"""Pure native hipfire four-GPU baseline command builder."""
import ipaddress
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman
from .model_manager import HipfireModel, resolved_path

CONTAINER_NAME = 'hipfire-cockpit-server'


def build_server_cmd(*, engine: str, image: str, engine_args: list[str], platform_id: str,
                     model: HipfireModel, models_dir: Path, state_dir: Path,
                     host: str = '127.0.0.1', port: int = 18910,
                     context_size: int = 8192) -> list[str]:
    if platform_id != 'r9700':
        raise ValueError('hipfire is qualified for four R9700 GPUs only.')
    if engine not in {'podman', 'docker'} or not image or image.startswith('-'):
        raise ValueError('Select the hipfire container engine and image.')
    address = ipaddress.ip_address('127.0.0.1' if host == 'localhost' else host)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535.')
    if type(context_size) is not int or not 2048 <= context_size <= model.context_size:
        raise ValueError('Context exceeds the qualified hipfire allocation.')
    root = models_dir.expanduser().resolve()
    state = state_dir.expanduser().resolve()
    for path in (root, state):
        if any(c in str(path) for c in ':\n\r\0'):
            raise ValueError('Container mount paths cannot contain colons or control characters.')
    path = resolved_path(model, root)
    if path is None:
        raise ValueError('Download / Repair the complete MQ2R model first.')
    command = [engine, 'run', '--rm', '-it', '--name', CONTAINER_NAME]
    command.extend(upgrade_groups_for_podman(engine, list(engine_args)))
    if engine == 'podman':
        command += ['--security-opt', 'label=disable', '--userns=keep-id']
    bind = f'[{address}]' if address.version == 6 else str(address)
    command += ['-p', f'{bind}:{port}:{port}', '-v', f'{root}:/models:ro', '-v', f'{state}:/state:rw',
                '-e', 'HIP_VISIBLE_DEVICES=0,1,2,3', '-e', 'HIPFIRE_HOME=/state',
                '-e', 'HIPFIRE_KERNEL_CACHE=/state/kernel-cache', '-e', 'HIPFIRE_NO_REGISTRY_FETCH=1',
                image, 'hipfire', 'serve', '/models/' + str(path.relative_to(root)), '0.0.0.0', str(port),
                '--tp', '4', '--spec', 'off', '--max-seq', str(context_size), '--kv-mode', 'f32',
                '--kv-backend', 'legacy', '--continuous-batch-size', '1']
    return command
