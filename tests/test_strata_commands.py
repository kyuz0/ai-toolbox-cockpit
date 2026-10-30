import json
from pathlib import Path
import pytest
from ai_toolbox_cockpit.backends.strata.model_manager import StrataModel, prepared, download_command, preparation_command
from ai_toolbox_cockpit.backends.strata.server_runner import build_server_cmd


def model_entry():
    return dict(id='strata-test',name='Flash Next',repo='AtomicChat/Qwen3.8-Flash-Next-GGUF',
        revision='142262902a46f7daed19c79d0771534c8106ad59',directory='atomicchat',
        files=[{'path':f'model-{i:05}-of-00033.gguf','size_bytes':1} for i in range(1,34)])


def fixture(tmp_path):
    model=StrataModel.from_entry(model_entry());root=tmp_path/model.directory;root.mkdir()
    for path,_ in model.files:(root/path).write_bytes(b'x')
    sizes={p:1 for p in ['pack/index.txt','pack/tokenizer/vocab.json','pack/tokenizer/chat_template.jinja','mtp/rt/dense.txt','mtp/rt/dense.bin','mtp/rt/experts.bin']}
    for p in sizes:
        file=root/p;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'x')
    (root/'preparation.json').write_text(json.dumps({'main_revision':model.revision,'mtp_revision':'de4b8e4d43b917e7706784d8bb445c9af86a3540','files':sizes}))
    return model,root


def test_strata_download_pins_all_shards_and_preparation_is_separate(tmp_path):
    model,_=fixture(tmp_path)
    cmd=download_command(model,tmp_path)
    assert cmd.count('--revision')==1 and model.revision in cmd
    assert all(path in cmd for path,_ in model.files)
    prep=preparation_command(model,tmp_path,'podman','registry/image:strata')
    assert 'strata-prepare' in prep and '--userns=keep-id' in prep
    assert not any('verify'==p for p in prep)


def test_strata_readiness_rejects_truncated_prepared_files(tmp_path):
    model,root=fixture(tmp_path)
    assert prepared(model,tmp_path)
    (root/'mtp/rt/experts.bin').write_bytes(b'')
    assert not prepared(model,tmp_path)


def test_strata_launch_is_two_card_profile_with_readonly_weights(tmp_path):
    model,_=fixture(tmp_path)
    kwargs=dict(engine='podman',image='registry/image:strata',engine_args=['--device','/dev/dri'],platform_id='intel-b70',model=model,directory=tmp_path)
    cmd=build_server_cmd(**kwargs)
    assert '--mtp' in cmd and '/models/mtp/rt' in cmd
    assert cmd[cmd.index('--context')+1]=='68096'
    assert any(p.endswith(':/models:ro') for p in cmd)
    assert '--mtp' not in build_server_cmd(**kwargs,speculation='baseline')
    with pytest.raises(ValueError):build_server_cmd(**dict(kwargs,platform_id='r9700'))
    with pytest.raises(ValueError):build_server_cmd(**kwargs,context_size=100000)


def test_strata_manifest_rejects_traversal_and_missing_shards():
    entry=model_entry();entry['files'][0]['path']='../escape'
    with pytest.raises(ValueError):StrataModel.from_entry(entry)
    entry=model_entry();entry['files'].pop()
    with pytest.raises(ValueError):StrataModel.from_entry(entry)
