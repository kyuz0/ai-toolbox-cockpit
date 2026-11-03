"""Pure Maya command builder for one Strix Halo APU."""
import ipaddress
from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman
from .model_manager import model_root, prepared

CONTAINER_NAME = 'maya-cockpit-server'


def build_server_cmd(*, engine, image, engine_args, platform_id, model, directory,
                     host='127.0.0.1', port=8000, context_size=68096, speculation='baseline'):
    if platform_id != 'strix-halo':
        raise ValueError('This Maya profile requires Strix Halo')
    if engine not in ('podman', 'docker') or not image or image.startswith('-'):
        raise ValueError('Select a container engine and Maya image')
    if type(context_size) is not int or not 2048 <= context_size <= 68096:
        raise ValueError('Context must be between 2048 and 68096 tokens')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid API port')
    address = ipaddress.ip_address('127.0.0.1' if host == 'localhost' else host)
    if speculation not in {value for _, value in model.modes}:
        raise ValueError('Select a decoding mode supported by this model')
    if not prepared(model, directory):
        raise ValueError('Download and prepare this Maya model in Models first')
    root = model_root(model, directory)
    bind = f'[{address}]' if address.version == 6 else str(address)
    cmd = [engine, 'run', '--rm', '-it', '--name', CONTAINER_NAME]
    cmd += upgrade_groups_for_podman(engine, list(engine_args)) + ['--ipc=host']
    if engine == 'podman':
        cmd += ['--security-opt', 'label=disable', '--userns=keep-id']
    # Native pack directories also hold runtime tuning and conversation cache files.
    cmd += ['-e', 'HOME=/tmp', '-p', f'{bind}:{port}:{port}', '-v', f'{root}:{root}',
            image, 'maya-serve', '--family', model.family, '--pack', str(root / model.pack_directory),
            '--model-name', model.id, '--context', str(context_size), '--mode', speculation,
            '--host', '0.0.0.0', '--port', str(port)]
    return cmd
