"""Four-R9700 DeepSeek V4.1 stays distinct from the B70 AtomicChat profile."""
from dataclasses import replace
import json

import pytest
from ai_toolbox_cockpit.backends.strata import dsv41
from ai_toolbox_cockpit.backends.strata.model_manager import get_model, StrataModel, prepared, preparation_command
from ai_toolbox_cockpit.backends.strata.server_runner import build_server_cmd


def fixture(tmp_path, pack=True):
    original = get_model('strata-deepseek-v41-flash-1017-gsq-iq2xxs')
    model = replace(original, files=tuple((name, 1) for name, _ in original.files),
                    auxiliary_files=tuple((name, 1) for name, _ in original.auxiliary_files))
    for directory, files in [(model.directory, model.files), (model.auxiliary_directory, model.auxiliary_files)]:
        for name, _ in files:
            path = tmp_path / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'x')
    if pack:
        base = dsv41.pack_root(model, tmp_path)
        sizes = {name: 1 for name in ['ds4.txt', 'tensors.txt', 'engram.bin', 'expert-profile.bin',
            'tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/token_type.json',
            'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja']}
        for name in sizes:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'x')
        (base / 'preparation.json').write_text(json.dumps({'main_revision': model.revision,
            'aux_revision': model.auxiliary_revision, 'source_revision': 'c62d5e3a15b40ea51c96aae9e8d36928965f53ab', 'files': sizes}))
    return model


def test_pack_requires_both_pinned_artifact_sets_and_every_prepared_file(tmp_path):
    model = fixture(tmp_path)
    assert isinstance(model, dsv41.StrataDsv41Model)
    assert prepared(model, tmp_path)
    (dsv41.pack_root(model, tmp_path) / 'tokenizer/chat_template.jinja').write_bytes(b'')
    assert not prepared(model, tmp_path)


def test_auxiliary_missing_is_not_ready(tmp_path):
    model = fixture(tmp_path)
    (tmp_path / model.auxiliary_directory / model.auxiliary_files[0][0]).unlink()
    assert not prepared(model, tmp_path)
    assert len(dsv41.incomplete_files(model, tmp_path)) == 1


def test_prepare_mounts_weights_readonly_and_separate_output_writable(tmp_path):
    model = fixture(tmp_path, pack=False)
    command = preparation_command(model, tmp_path, 'podman', 'registry.example/strata:tested')
    assert f'{tmp_path}:/models:ro' in command
    assert f'{tmp_path / ".prepared"}:/prepared:rw' in command
    assert model.revision in command and model.auxiliary_revision in command
    assert '--device' not in command


def test_four_gpu_server_uses_dsv41_entrypoint_and_rejects_b70_mtp(tmp_path):
    model = fixture(tmp_path)
    options = dict(engine='podman', image='registry.example/strata:tested', engine_args=['--device', '/dev/kfd'],
        platform_id='r9700', model=model, directory=tmp_path, context_size=8192)
    for mode in ['baseline', 'dspark']:
        command = build_server_cmd(**options, speculation=mode)
        assert command[command.index('--decoding') + 1] == mode
        assert command[command.index('--pack') + 1] == '/models/.prepared/' + model.pack_directory
        assert '--mtp' not in command
        assert 'STRATA_DS4_SLOT_KEEP=0' in command
    with pytest.raises(ValueError):
        build_server_cmd(**options, speculation='mtp')
    with pytest.raises(ValueError):
        build_server_cmd(**(options | {'platform_id': 'intel-b70'}), speculation='baseline')


def test_deepseek_manifest_does_not_weaken_atomicchat_shard_validation():
    original = get_model('strata-deepseek-v41-flash-1017-gsq-iq2xxs')
    entry = vars(original) | {'files': [{'path': n, 'size_bytes': s} for n, s in original.files],
        'auxiliary_files': [{'path': n, 'size_bytes': s} for n, s in original.auxiliary_files]}
    assert isinstance(StrataModel.from_entry(entry), dsv41.StrataDsv41Model)
    with pytest.raises(ValueError, match='33-shard'):
        StrataModel.from_entry(entry | {'family': 'qwen4exp'})
    for field, value in [('auxiliary_revision', 'main'), ('pack_directory', '../bad'), ('platform_id', 'intel-b70')]:
        with pytest.raises(ValueError):
            StrataModel.from_entry(entry | {field: value})


def test_download_entrypoint_accepts_the_selected_canonical_model(tmp_path, monkeypatch):
    import runpy
    import sys
    from ai_toolbox_cockpit.backends.strata import model_manager
    model = fixture(tmp_path, pack=False)
    command = dsv41.download_command(model, tmp_path)
    assert command[2] == 'ai_toolbox_cockpit.backends.strata.download'
    calls = []
    monkeypatch.setattr(model_manager, 'get_model', lambda identifier: model)
    monkeypatch.setattr(dsv41.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(sys, 'argv', command[2:])
    runpy.run_module(command[2], run_name='__main__')
    assert len(calls) == 2
    for (command, options), revision in zip(calls, [model.revision, model.auxiliary_revision]):
        assert command[command.index('--revision') + 1] == revision
        assert options['check'] is True
