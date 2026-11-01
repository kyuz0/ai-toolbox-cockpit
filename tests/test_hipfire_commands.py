from dataclasses import replace
from pathlib import Path

import pytest

from ai_toolbox_cockpit.backends.hipfire.model_manager import HipfireModel, download_command, get_model, resolved_path
from ai_toolbox_cockpit.backends.hipfire.server_runner import build_server_cmd


def test_native_artifact_validation():
    model = get_model('hipfire-deepseek-v4-flash-0731-mq2r')
    for field, value in [('filename', 'model.gguf'), ('directory', '../outside'), ('revision', 'main'), ('size_bytes', True)]:
        with pytest.raises(ValueError):
            HipfireModel.from_entry(vars(model) | {field: value})


def test_pinned_download_and_native_server_do_not_enable_speculation(tmp_path):
    model = replace(get_model('hipfire-deepseek-v4-flash-0731-mq2r'), size_bytes=4)
    path = tmp_path / model.directory / model.filename
    path.parent.mkdir(parents=True)
    path.write_bytes(b'test')
    assert resolved_path(model, tmp_path) == path
    download = download_command(model, tmp_path)
    assert download[download.index('--revision') + 1] == model.revision
    assert model.filename in download
    command = build_server_cmd(engine='podman', image='example.org/hipfire:test', engine_args=[],
        platform_id='r9700', model=model, models_dir=tmp_path, state_dir=tmp_path / 'state')
    assert command[command.index('hipfire') + 1:] == ['serve', f'/models/{model.directory}/{model.filename}',
        '0.0.0.0', '18910', '--tp', '4', '--spec', 'off', '--max-seq', '8192', '--kv-mode', 'f32',
        '--kv-backend', 'legacy', '--continuous-batch-size', '1']
    assert 'HIPFIRE_NO_REGISTRY_FETCH=1' in command
    assert '--entrypoint' not in command
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='complete MQ2R'):
        build_server_cmd(engine='podman', image='example.org/hipfire:test', engine_args=[],
            platform_id='r9700', model=model, models_dir=tmp_path, state_dir=tmp_path / 'state')
