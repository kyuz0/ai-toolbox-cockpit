"""Native artifact safety, resumable assembly and qualified server boundaries."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_toolbox_cockpit.backends.affinity.download import assemble_target, download
from ai_toolbox_cockpit.backends.affinity.model_manager import (AffinityModel, artifact_paths,
    download_space_bytes, get_model, incomplete_files)
from ai_toolbox_cockpit.backends.affinity.server_runner import build_server_cmd


def tiny_model():
    return replace(get_model('affinity-deepseek-v4-flash-0731-2875bpw'), size_bytes=5,
                   parts=(('first', 2), ('second', 3)), draft_size_bytes=2, tokenizer_size_bytes=2)


def prepared(tmp_path):
    model = tiny_model()
    for path, value in zip(artifact_paths(model, tmp_path), (b'abcde', b'dr', b'{}')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return model


def command(tmp_path, **kwargs):
    model = prepared(tmp_path)
    return build_server_cmd(engine='podman', image='docker.io/example/affinity:tested',
        engine_args=['--device', '/dev/kfd'], platform_id='r9700', model=model,
        models_dir=tmp_path, cache_dir=tmp_path / 'cache', **kwargs)


def test_catalog_pins_formats_and_part_sizes():
    model = get_model('affinity-deepseek-v4-flash-0731-2875bpw')
    assert sum(size for _, size in model.parts) == model.size_bytes
    values = vars(model) | {'parts': [{'path': name, 'size_bytes': size} for name, size in model.parts]}
    for field, bad in [('directory', '../outside'), ('filename', '/tmp/model.aff'),
                       ('tokenizer_directory', 'foo:rw'), ('revision', 'main'),
                       ('size_bytes', 1), ('context_size', 100000)]:
        with pytest.raises(ValueError):
            AffinityModel.from_entry(values | {field: bad})


def test_assembly_is_ordered_and_complete_target_is_reused(tmp_path):
    model = tiny_model()
    target, _, _ = artifact_paths(model, tmp_path)
    target.parent.mkdir(parents=True)
    (target.parent / 'first').write_bytes(b'ab')
    (target.parent / 'second').write_bytes(b'cde')
    assert download_space_bytes(model, tmp_path) == 9
    assert assemble_target(model, tmp_path).read_bytes() == b'abcde'
    (target.parent / 'first').unlink()
    assert assemble_target(model, tmp_path).read_bytes() == b'abcde'
    assert not list(target.parent.glob('*.assembling-*'))


def test_failed_assembly_preserves_existing_target(tmp_path):
    model = tiny_model()
    target, _, _ = artifact_paths(model, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old')
    (target.parent / 'first').write_bytes(b'ab')
    (target.parent / 'second').write_bytes(b'cde')
    with patch('shutil.copyfileobj', side_effect=OSError('interrupted')):
        with pytest.raises(OSError):
            assemble_target(model, tmp_path)
    assert target.read_bytes() == b'old'
    assert not list(target.parent.glob('*.assembling-*'))


def test_existing_prepared_model_skips_network_and_parts(tmp_path):
    model = prepared(tmp_path)
    with patch('subprocess.run') as run:
        download(model, tmp_path)
    run.assert_not_called()
    assert not incomplete_files(model, tmp_path)


def test_drafter_repair_does_not_download_parts(tmp_path):
    model = prepared(tmp_path)
    _, draft, _ = artifact_paths(model, tmp_path)
    draft.unlink()
    def repair(args, **kwargs):
        assert model.draft_filename in args
        assert 'first' not in args and 'second' not in args
        draft.write_bytes(b'dr')
    with patch('subprocess.run', side_effect=repair) as run:
        download(model, tmp_path)
    assert run.call_count == 1


def test_quad_baseline_and_dspark_use_native_format_and_isolated_cache(tmp_path):
    base = command(tmp_path)
    assert base[base.index('--gpus') + 1] == '4'
    assert base[base.index('--host-pool-mib') + 1] == '0'
    assert base[base.index('--kv-size') + 1] == '8256'
    assert base[base.index('--dspark') + 1] == 'off'
    assert '--slots' not in base
    assert '--raw-completions' in base
    draft = command(tmp_path, speculation='dspark')
    assert draft[draft.index('--dspark') + 1] == 'on'
    assert draft[draft.index('--dspark-model') + 1].endswith('/model-0731.dspark.aff')
    assert '127.0.0.1:18915:18915' in draft


@pytest.mark.parametrize('kwargs', [{'context_size': 67841}, {'context_size': True},
                                    {'speculation': 'mtp'}, {'port': 0}])
def test_reject_unsupported_launch(tmp_path, kwargs):
    with pytest.raises(ValueError):
        command(tmp_path, **kwargs)


def test_source_import_preserves_new_native_artifact_sets(tmp_path):
    import json
    import runpy
    from ai_toolbox_cockpit.catalog import load_model_catalog
    from ai_toolbox_cockpit.catalog.schema import ModelCatalog
    sources = {'llama-models': '[]', 'ds4-models': '{"repo":"example/models","models":[]}',
               'vllm-models': 'MODEL_TABLE = {}', 'comfy-manager': 'MODEL_FAMILIES = []'}
    argv = ['import_source_catalogs.py']
    for flag, content in sources.items():
        path = tmp_path / flag
        path.write_text(content)
        argv += ['--' + flag, str(path)]
    output = tmp_path / 'models.json'
    argv += ['--comfy-workflows', str(tmp_path), '--output', str(output)]
    with patch('sys.argv', argv):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/import_source_catalogs.py'), run_name='__main__')
    imported = ModelCatalog.from_dict(json.loads(output.read_text()))
    for backend in ['affinity', 'hipfire', 'strata']:
        assert imported.backends[backend] == load_model_catalog().backends[backend]
