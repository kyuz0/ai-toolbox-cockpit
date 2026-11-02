import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import pytest
from ai_toolbox_cockpit.backends.maya.model_manager import (
    load_models, model_root, pack_files, prepared, record_prepared, preparation_command, download_command)
from ai_toolbox_cockpit.backends.maya.server_runner import build_server_cmd


def fixture_model(model, tmp_path):
    model = replace(model, files=tuple((p, 8) for p, _ in model.files))
    root = model_root(model, tmp_path)
    for p, n in model.files:
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b'x' * n)
    for p in pack_files(model):
        f = root / model.pack_directory / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b'fixture')
    record_prepared(model, tmp_path)
    return model


@pytest.mark.parametrize('index,mode,wrong', [(0, 'mtp', 'dspark'), (1, 'dspark', 'mtp')])
def test_preparation_and_serving_use_identical_paths_and_complete_artifacts(tmp_path, index, mode, wrong):
    model = fixture_model(load_models()[index], tmp_path)
    root = model_root(model, tmp_path)
    command = preparation_command(model, tmp_path, 'podman', 'example/maya:tested')
    assert f'{root}:{root}' in command
    assert str(root / model.pack_directory) in command
    download = download_command(model, tmp_path)
    assert download[download.index('--revision') + 1] == model.revision
    params = dict(engine='podman', image='example/maya:tested', engine_args=['--device', '/dev/kfd'],
                  platform_id='strix-halo', model=model, directory=tmp_path, speculation=mode)
    server = build_server_cmd(**params)
    assert f'{root}:{root}' in server
    assert str(root / model.pack_directory) in server
    assert server[server.index('--mode') + 1] == mode
    with pytest.raises(ValueError, match='decoding mode'):
        build_server_cmd(**{**params, 'speculation': wrong})
    (root / model.files[-1][0]).unlink()
    assert not prepared(model, tmp_path)
    with pytest.raises(ValueError, match='Download and prepare'):
        build_server_cmd(**params)


def test_only_selected_pinned_bundles_are_exposed():
    glm, ds = load_models()
    assert glm.size_bytes == 94701676224
    assert ds.size_bytes == 347445667721
    assert len(glm.files) == 3 and len(ds.files) == 8
    assert glm.family == 'glm' and ds.family == 'deepseek'


def test_maya_ui_model_switch_changes_speculation_and_download(tmp_path):
    from textual.widgets import Input, TabbedContent, Label
    from ai_toolbox_cockpit.app import AiToolboxCockpitApp
    from ai_toolbox_cockpit.widgets import SearchableSelect, ConfirmModal
    async def check():
        with patch.dict('os.environ', {'XDG_CONFIG_HOME': str(tmp_path)}), \
             patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'), \
             patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'), \
             patch('ai_toolbox_cockpit.app.available_update', return_value=None), \
             patch('ai_toolbox_cockpit.backends.ds4.server.scan_local_models', return_value=[]):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(160, 50)) as pilot:
                app.query_one('#platform-select', SearchableSelect).value = 'strix-halo'
                app.query_one(TabbedContent).active = 'tab-servers'
                app.query_one('#server-backend-select', SearchableSelect).value = 'maya'
                await pilot.pause()
                assert app.query_one('#maya-speculation', SearchableSelect).value == 'mtp'
                app.save_screenshot('/tmp/maya-server-glm.svg')
                app.query_one('#maya-model', SearchableSelect).value = 'maya-deepseek41-1017-s'
                await pilot.pause()
                assert app.query_one('#maya-speculation', SearchableSelect).value == 'dspark'
                for key in ('engine', 'image', 'model', 'speculation'):
                    assert app.query_one(f'#maya-{key}-label', Label).display
                app.save_screenshot('/tmp/maya-server-deepseek.svg')
                app.query_one(TabbedContent).active = 'tab-models'
                app.query_one('#model-backend-select', SearchableSelect).value = 'maya'
                await pilot.pause()
                app.query_one('#maya-download-model', SearchableSelect).value = 'maya-deepseek41-1017-s'
                app.query_one('#maya-models-dir', Input).value = str(tmp_path / 'models')
                await pilot.click('#maya-download')
                await pilot.pause()
                assert isinstance(app.screen, ConfirmModal)
                app.save_screenshot('/tmp/maya-download.svg')
                await pilot.press('escape')
    asyncio.run(check())
