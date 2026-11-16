"""Radiance artifacts and launch contracts."""
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
    assert cmd[cmd.index('--max-model-len') + 1] == str(launch['model'].context_size)
    assert '--mm-max-patches' not in cmd
    assert '127.0.0.1:18080:18080' in cmd
    assert any(x.endswith(':/models/model.rad:ro') for x in cmd)
    assert cmd[cmd.index('--served-model-name') + 1] == launch['model'].name
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
                                      dict(context_size=0), dict(host='127.0.0.1;echo bad'),
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
    with pytest.raises(ValueError, match='concurrent requests'):
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


def test_catalogue_replaces_legacy_models_with_exact_qualified_iqrv4():
    from ai_toolbox_cockpit.backends.radiance.model_manager import load_models
    models = load_models()
    assert [model.id for model in models] == ['radiance-qwen38-flash-iqrv4-679c9432']
    model = models[0]
    assert model.revision == '679c943216e5a33da5f458b5da378586219e874b'
    assert model.size_bytes == 121678372864
    assert (model.config_format, model.kv_cache_dtype, model.vision) == ('yaml', 'i8', True)
    assert model.toolbox_ids == ('r9700-radiance-rocm-10-1',)
    native = model.serving_profile()
    extended = model.serving_profile('disk-1m')
    assert (native.id, native.default_sessions, native.context_size) == ('disk-native', 1, 262144)
    assert (extended.default_sessions, extended.context_size, extended.rope_scaling) == (1, 1000000, 'yarn:4:262144')
    assert all(profile.tensor_parallel == 4 for profile in model.available_profiles())


@pytest.mark.parametrize('profile,context,sessions,slots', [('disk-native', 262144, None, 1), ('ram-native', 262144, None, 1), ('disk-1m', 1000000, None, 1), ('disk-native', 262144, 16, 16), ('ram-native', 262144, 16, 16)])
def test_published_profile_defaults_and_explicit_concurrency(tmp_path, profile, context, sessions, slots):
    from dataclasses import replace
    from ai_toolbox_cockpit.backends.radiance.model_manager import load_models
    model = replace(load_models()[0], size_bytes=4)
    target = tmp_path / 'model.rad'
    target.write_bytes(b'test')
    cmd = build_server_cmd(engine='podman', image='example/radiance:test', engine_args=[],
                           platform_id='r9700', model=model, model_path=target,
                           serving_profile=profile, speculation='mtp', sessions=sessions)
    for flag, value in [('--tp', '4'), ('--tp-wire', 'exact'), ('--max-model-len', str(context)),
                        ('--max-num-seqs', str(slots)), ('--kv-cache-dtype', 'i8'),
                        ('--checkpoint-slots', '48' if profile == 'ram-native' else '144'), ('--prefill-interleave', '0'),
                        ('--ngram-placement', 'ram' if profile == 'ram-native' else 'disk'), ('--num-speculative-tokens', '3')]:
        assert cmd[cmd.index(flag) + 1] == value
    assert '--mm-max-patches' not in cmd
    assert 'rad-yaml' in cmd[cmd.index('bash') + 2]
    assert ('--rope-scaling' in cmd) == (profile == 'disk-1m')
    if profile == 'ram-native':
        for flag, value in [('--checkpoint-policy', 'tip-only'), ('--embedding-placement', 'vram'),
                            ('--host-headroom-mib', '2048')]:
            assert cmd[cmd.index(flag) + 1] == value


@pytest.mark.parametrize('depth', [1, 3, 8, 16])
def test_explicit_draft_tokens_reach_native_command(launch, depth):
    cmd = build_server_cmd(**launch, speculation='dflash2', draft_tokens=depth)
    assert cmd[cmd.index('--num-speculative-tokens') + 1] == str(depth)


@pytest.mark.parametrize('depth', [0, -1, 17, True, 2.5])
def test_invalid_draft_tokens_are_rejected(launch, depth):
    with pytest.raises(ValueError, match='Draft tokens'):
        build_server_cmd(**launch, speculation='dflash2', draft_tokens=depth)


@pytest.fixture
def radiance21_model(model):
    return RadianceModel.from_entry(dict(vars(model), config_format='yaml', kv_cache_dtype='i8',
        speculation='mtp', draft_tokens=3, tensor_parallel=4, context_size=262144,
        ngram_placement='disk', serving_profiles=[dict(
            id='single-1m', name='Single user · 1M', tensor_parallel=4,
            context_size=1000000, max_sessions=1, checkpoint_slots=144,
            rope_scaling='yarn:4:262144', prefill_interleave='0',
        ), dict(
            id='ram-48', name='RAM n-grams', tensor_parallel=4,
            context_size=262144, max_sessions=16, default_sessions=16,
            checkpoint_slots=48, checkpoint_policy='tip-only', ngram_placement='ram',
            prefill_interleave='0', embedding_placement='vram', host_headroom_mib=2048,
            prefix_cache_mib=0,
        )]))


def test_21_single_user_extension_matches_qualified_launch(launch, radiance21_model):
    launch['model'] = radiance21_model
    cmd = build_server_cmd(**launch, serving_profile='single-1m', speculation='mtp')
    image_index = cmd.index(launch['image'])
    assert cmd[image_index + 1:image_index + 3] == ['bash', '-c']
    script = cmd[image_index + 3]
    assert 'rad-yaml "$@"' in script
    assert 'radiance --config /tmp/radiance-cockpit.yaml --check;' in script
    assert script.endswith('exec radiance --config /tmp/radiance-cockpit.yaml')
    expected = {'--tp': '4', '--tp-wire': 'exact', '--max-num-seqs': '1',
                '--max-model-len': '1000000', '--max-num-batched-tokens': '2048',
                '--kv-cache-dtype': 'i8', '--ngram-placement': 'disk',
                '--num-speculative-tokens': '3', '--checkpoint-slots': '144',
                '--prefill-interleave': '0', '--rope-scaling': 'yarn:4:262144'}
    for key, value in expected.items():
        assert cmd.count(key) == 1
        assert cmd[cmd.index(key) + 1] == value


def test_21_explicit_context_above_extension_is_never_clamped(launch, radiance21_model):
    launch['model'] = radiance21_model
    cmd = build_server_cmd(**launch, serving_profile='single-1m', context_size=2000001,
                           rope_scaling='yarn:8:262144')
    assert cmd[cmd.index('--max-model-len') + 1] == '2000001'
    assert cmd[cmd.index('--rope-scaling') + 1] == 'yarn:8:262144'
    native = build_server_cmd(**launch, serving_profile='single-1m', rope_scaling='none')
    assert '--rope-scaling' not in native


def test_21_ram_profile_preserves_memory_policy_and_explicit_concurrency(launch, radiance21_model):
    launch['model'] = radiance21_model
    cmd = build_server_cmd(**launch, serving_profile='ram-48', speculation='mtp')
    expected = {'--max-num-seqs': '16', '--max-model-len': '262144',
                '--ngram-placement': 'ram', '--checkpoint-slots': '48',
                '--checkpoint-policy': 'tip-only', '--embedding-placement': 'vram',
                '--host-headroom-mib': '2048', '--prefix-cache-mib': '0'}
    for key, value in expected.items():
        assert cmd[cmd.index(key) + 1] == value
    explicit = build_server_cmd(**launch, serving_profile='ram-48', sessions=4)
    assert explicit[explicit.index('--max-num-seqs') + 1] == '4'
    assert '--rope-scaling' not in cmd


@pytest.mark.parametrize('field,value', [('context_size', True), ('context_size', 0),
    ('default_sessions', 17), ('default_sessions', True), ('prefix_cache_mib', -1),
    ('host_headroom_mib', True), ('checkpoint_policy', 'invented'),
    ('ngram_placement', 'pruned'), ('embedding_placement', 'disk'),
    ('prefill_interleave', '1;echo bad'), ('rope_scaling', 'yarn:0:262144'),
    ('rope_scaling', 'yarn:4:0'), ('rope_scaling', 'yarn:4:262144;echo bad')])
def test_21_profile_fields_validate_before_launch(radiance21_model, field, value):
    profile = dict(vars(radiance21_model.serving_profiles[1]), **{field: value})
    with pytest.raises(ValueError):
        RadianceModel.from_entry(dict(vars(radiance21_model), serving_profiles=[profile]))


@pytest.mark.parametrize('ids', ['radiance-21', ['bad/id'], ['x', 'x'], [True]])
def test_model_runtime_mapping_rejects_invalid_ids(model, ids):
    with pytest.raises(ValueError, match='toolbox IDs'):
        RadianceModel.from_entry(dict(vars(model), toolbox_ids=ids))



def test_explicit_default_profile_preserves_measured_slot_allocation(launch, radiance21_model):
    model = RadianceModel.from_entry(dict(vars(radiance21_model), default_profile='ram-48'))
    launch['model'] = model
    assert [p.id for p in model.available_profiles()] == ['single-1m', 'ram-48']
    assert model.serving_profile().id == 'ram-48'
    cmd = build_server_cmd(**launch)
    assert cmd[cmd.index('--max-num-seqs') + 1] == '16'
    assert cmd[cmd.index('--checkpoint-policy') + 1] == 'tip-only'
    with pytest.raises(ValueError, match='default Radiance'):
        RadianceModel.from_entry(dict(vars(model), default_profile='missing'))
