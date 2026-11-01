"""Standalone Radiance artifacts and launch contracts."""
from pathlib import Path

import pytest

from ai_toolbox_cockpit.backends.radiance.model_manager import RadianceModel, download_command, resolved_path
from ai_toolbox_cockpit.backends.radiance.server_runner import build_server_cmd


@pytest.fixture
def model():
    return RadianceModel.from_entry(dict(
        id='qwen27-test', name='Qwen27 test fixture', repo='owner/model',
        revision='a' * 40, directory='qwen27-a', filename='model.rad',
        size_bytes=16, family='qwen27', speculation='dflash2', draft_tokens=7,
        context_size=65536, tensor_parallel=2, vision=True,
    ))


@pytest.fixture
def launch(tmp_path, model):
    target = tmp_path / 'model.rad'
    target.write_bytes(b'x' * model.size_bytes)
    return dict(engine='podman', image='docker.io/owner/radiance:channel',
                engine_args=['--device', '/dev/kfd', '--device', '/dev/dri',
                             '--security-opt', 'seccomp=unconfined'],
                platform_id='r9700', model=model, model_path=target)


def test_partial_file_never_becomes_launchable(launch):
    launch['model_path'].write_bytes(b'partial')
    with pytest.raises(ValueError, match='complete .rad'):
        build_server_cmd(**launch)


def test_native_model_bundles_do_not_accept_a_different_drafter(launch):
    with pytest.raises(ValueError, match='does not include'):
        build_server_cmd(**launch, speculation='mtp')


def test_baseline_disables_bundled_draft_and_does_not_omit_vision(launch):
    cmd = build_server_cmd(**launch)
    assert cmd[cmd.index('--num-speculative-tokens') + 1] == '0'
    assert '--mm-max-patches' not in cmd
    assert '127.0.0.1:18080:18080' in cmd
    assert any(x.endswith(':/models/model.rad:ro') for x in cmd)
    assert '--speculative-config' not in cmd  # vLLM flags must never enter this engine.


def test_native_speculation_and_thinking_are_explicit(launch):
    cmd = build_server_cmd(**launch, speculation='dflash2', thinking_effort='xhigh', vision_enabled=False)
    assert cmd[cmd.index('--num-speculative-tokens') + 1] == '7'
    assert cmd[cmd.index('--reasoning-effort') + 1] == 'xhigh'
    assert cmd[cmd.index('--mm-max-patches') + 1] == '0'
    assert cmd[cmd.index('--tp-wire') + 1] == 'exact'


def test_ipv6_binding_stays_one_container_argument(launch):
    cmd = build_server_cmd(**launch, host='::1')
    assert '[::1]:18080:18080' in cmd


@pytest.mark.parametrize('override', [dict(tensor_parallel=8), dict(tensor_parallel=4), dict(sessions=2), dict(platform_id='strix-halo'),
                                      dict(context_size=65537), dict(host='127.0.0.1;echo bad'),
                                      dict(port=0), dict(placement='pruned')])
def test_unsupported_profile_is_rejected(launch, override):
    launch.update(override)
    with pytest.raises(ValueError):
        build_server_cmd(**launch)


def test_download_is_exact_and_revision_pinned(model, tmp_path):
    cmd = download_command(model, tmp_path)
    assert cmd[2:4] == [model.repo, 'model.rad']
    assert cmd[cmd.index('--revision') + 1] == 'a' * 40
    assert cmd[cmd.index('--local-dir') + 1] == str(tmp_path / model.directory)
    assert '--include' not in cmd


def test_size_checked_inventory_and_safe_model_paths(model, tmp_path):
    directory = tmp_path / model.directory
    directory.mkdir()
    target = directory / model.filename
    target.write_bytes(b'partial')
    assert resolved_path(model, tmp_path) is None
    target.write_bytes(b'x' * model.size_bytes)
    assert resolved_path(model, tmp_path) == target.resolve()
    data = dict(vars(model), directory='../another-engine')
    with pytest.raises(ValueError, match='safe relative'):
        RadianceModel.from_entry(data)


def test_flash_profile_pins_ngram_ram_instead_of_automatic_disk_fallback(launch):
    from dataclasses import replace
    launch['model'] = replace(launch['model'], family='qwen38-flash', tensor_parallel=4, ngram_placement='ram')
    cmd = build_server_cmd(**launch)
    assert cmd[cmd.index('--ngram-placement') + 1] == 'ram'
    assert cmd[cmd.index('--tp') + 1] == '4'


def test_concurrency_uses_the_selected_models_qualified_limit(launch):
    from dataclasses import replace
    launch['model'] = replace(launch['model'], max_sessions=16)
    cmd = build_server_cmd(**launch, sessions=16)
    assert cmd[cmd.index('--max-num-seqs') + 1] == '16'
    with pytest.raises(ValueError, match='session'):
        build_server_cmd(**launch, sessions=17)


@pytest.mark.parametrize('field,value', [('size_bytes',16.5), ('context_size',True), ('draft_tokens',True), ('max_sessions',17)])
def test_native_artifact_profile_rejects_invalid_numeric_types_and_bounds(model, field, value):
    with pytest.raises(ValueError):
        RadianceModel.from_entry(dict(vars(model), **{field:value}))


@pytest.fixture
def quad_model(model):
    return RadianceModel.from_entry(dict(vars(model), serving_profiles=[dict(
        id='quad-tp4-144', name='Four R9700s', tensor_parallel=4,
        max_sessions=16, checkpoint_slots=144,
    )]))


@pytest.mark.parametrize('mode,depth', [('baseline', '0'), ('dflash2', '7')])
def test_quad_profile_sets_gpu_sequence_checkpoint_and_decode_allocations(launch, quad_model, mode, depth):
    launch['model'] = quad_model
    cmd = build_server_cmd(**launch, serving_profile='quad-tp4-144', sessions=16,
                           context_size=65536, speculation=mode)
    for flag, value in [('--tp', '4'), ('--max-num-seqs', '16'),
                        ('--checkpoint-slots', '144'), ('--num-speculative-tokens', depth)]:
        assert cmd[cmd.index(flag) + 1] == value
    default = build_server_cmd(**launch)
    assert default[default.index('--tp') + 1] == '2'
    assert '--checkpoint-slots' not in default


def test_profile_selection_cannot_override_qualified_layout_or_capacity(launch, quad_model):
    launch['model'] = quad_model
    for extra in ({'tensor_parallel': 2}, {'sessions': 17}):
        with pytest.raises(ValueError):
            build_server_cmd(**launch, serving_profile='quad-tp4-144', **extra)
    with pytest.raises(ValueError, match='serving profile'):
        build_server_cmd(**launch, serving_profile='unmeasured-layout')


@pytest.mark.parametrize('field,value', [('tensor_parallel', True), ('tensor_parallel', 3),
                                        ('max_sessions', 17), ('checkpoint_slots', True),
                                        ('checkpoint_slots', 0), ('checkpoint_slots', 1.5)])
def test_quad_profile_validates_types_before_constructing_a_command(quad_model, field, value):
    entry = dict(vars(quad_model.serving_profiles[0]), **{field:value})
    with pytest.raises(ValueError):
        RadianceModel.from_entry(dict(vars(quad_model), serving_profiles=[entry]))


def test_profile_ids_cannot_replace_default_or_duplicate(quad_model):
    entry = vars(quad_model.serving_profiles[0])
    for profiles in ([dict(entry, id='default')], [entry, entry]):
        with pytest.raises(ValueError):
            RadianceModel.from_entry(dict(vars(quad_model), serving_profiles=profiles))


def test_quad_choices_reuse_existing_native_downloads():
    from ai_toolbox_cockpit.backends.radiance.model_manager import load_models
    models = load_models()
    assert len(models) == 3
    for model in models:
        profiles = model.available_profiles()
        if model.family == 'qwen38_27b':
            quad = model.serving_profile('quad-tp4-144')
            assert (quad.tensor_parallel, quad.max_sessions, quad.checkpoint_slots) == (4, 16, 144)
            assert len(profiles) == 2
        else:
            assert len(profiles) == 1
