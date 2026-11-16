"""Pure commands for Radiance, using native flags and .rad containers."""
import ipaddress
from pathlib import Path

from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman
from .model_manager import RadianceModel, validate_rope_scaling

CONTAINER_NAME = 'radiance-cockpit-server'


def build_server_cmd(*, engine: str, image: str, engine_args: list[str],
                     platform_id: str, model: RadianceModel, model_path: Path,
                     host: str = '127.0.0.1', port: int = 18080,
                     tensor_parallel: int | None = None, serving_profile: str = 'default',
                     context_size: int | None = None,
                     sessions: int | None = None, speculation: str = 'baseline',
                     draft_tokens: int | None = None, vision_enabled: bool | None = None,
                     thinking_effort: str = 'auto', placement: str = 'all_vram',
                     host_pool_mib: int | None = None,
                     max_batched_tokens: int = 2048,
                     rope_scaling: str | None = None) -> list[str]:
    if platform_id != 'r9700':
        raise ValueError('This Radiance toolbox is qualified for R9700 only.')
    if engine not in {'podman', 'docker'}:
        raise ValueError('Select Podman or Docker.')
    if not image or image.startswith('-'):
        raise ValueError('Select the Radiance toolbox image.')
    host = '127.0.0.1' if host == 'localhost' else host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError('Host must be an IP address or localhost.') from error
    if not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535.')
    profile = model.serving_profile(serving_profile)
    tensor_parallel = profile.tensor_parallel if tensor_parallel is None else tensor_parallel
    if type(tensor_parallel) is not int or tensor_parallel != profile.tensor_parallel:
        raise ValueError(f'This qualified profile uses {profile.tensor_parallel} GPU(s).')
    context_size = (profile.context_size or model.context_size) if context_size is None else context_size
    sessions = profile.default_sessions if sessions is None else sessions
    rope_scaling = profile.rope_scaling if rope_scaling is None else rope_scaling
    validate_rope_scaling(rope_scaling)
    if type(context_size) is not int or context_size < 1:
        raise ValueError('Context must be a positive integer.')
    if type(sessions) is not int or not 1 <= sessions <= profile.max_sessions or not 1 <= max_batched_tokens <= context_size:
        raise ValueError('Invalid concurrent requests or prefill token limit.')
    if thinking_effort not in {'auto', 'none', 'low', 'medium', 'xhigh'}:
        raise ValueError('Select a supported thinking effort.')
    if placement != 'all_vram':
        raise ValueError('Select an available weight-placement profile.')
    if host_pool_mib is not None and not 0 <= host_pool_mib <= 49152:
        raise ValueError('Invalid per-GPU pinned host-memory budget.')
    if speculation not in {'baseline', model.speculation}:
        raise ValueError('The selected model does not include that drafter.')
    depth = 0 if speculation == 'baseline' else (model.draft_tokens if draft_tokens is None else draft_tokens)
    if type(depth) is not int or (speculation != 'baseline' and not 1 <= depth <= 16):
        raise ValueError('Draft tokens must be an integer between 1 and 16; select Disabled to turn off speculative decoding.')
    vision_enabled = model.vision if vision_enabled is None else vision_enabled
    if vision_enabled and not model.vision:
        raise ValueError('Image input is unavailable for this model profile.')
    path = model_path.expanduser().resolve()
    if path.suffix != '.rad' or not path.is_file() or path.stat().st_size != model.size_bytes:
        raise ValueError('Download or repair the complete .rad container first.')
    command = [engine, 'run', '--rm', '-it', '--name', CONTAINER_NAME]
    command.extend(upgrade_groups_for_podman(engine, list(engine_args)))
    if engine == 'podman':
        command += ['--security-opt', 'label=disable', '--userns=keep-id']
    bind = f'[{address}]' if address.version == 6 else str(address)
    command += ['-p', f'{bind}:{port}:{port}', '-v', f'{path}:/models/model.rad:ro', image]
    if model.config_format == 'yaml':
        command += ['bash', '-c',
                    'set -e; rad-yaml \"$@\" > /tmp/radiance-cockpit.yaml; '
                    'radiance --config /tmp/radiance-cockpit.yaml --check; '
                    'exec radiance --config /tmp/radiance-cockpit.yaml',
                    'radiance-cockpit-launch']
    else:
        command += ['radiance']
    command += ['--model', '/models/model.rad', '--served-model-name', model.name, '--host', '0.0.0.0',
                '--port', str(port), '--tp', str(tensor_parallel),
                '--max-model-len', str(context_size), '--max-num-seqs', str(sessions),
                '--max-num-batched-tokens', str(max_batched_tokens), '--kv-cache-dtype', model.kv_cache_dtype,
                '--tp-wire', 'exact', '--placement', placement,
                '--ngram-placement', profile.ngram_placement or model.ngram_placement,
                '--num-speculative-tokens', str(depth)]
    if profile.checkpoint_slots is not None:
        command += ['--checkpoint-slots', str(profile.checkpoint_slots)]
    for field, flag in (
        ('checkpoint_policy', '--checkpoint-policy'),
        ('prefill_interleave', '--prefill-interleave'),
        ('embedding_placement', '--embedding-placement'),
        ('host_headroom_mib', '--host-headroom-mib'),
        ('prefix_cache_mib', '--prefix-cache-mib'),
    ):
        value = getattr(profile, field)
        if value is not None:
            command += [flag, str(value)]
    if rope_scaling is not None and rope_scaling != 'none':
        command += ['--rope-scaling', rope_scaling]
    if host_pool_mib is not None:
        command += ['--host-pool-mib', str(host_pool_mib)]
    if not vision_enabled:
        command += ['--mm-max-patches', '0']
    if thinking_effort != 'auto':
        command += ['--reasoning-effort', thinking_effort]
    return command
