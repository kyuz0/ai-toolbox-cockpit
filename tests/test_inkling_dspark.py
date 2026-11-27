"""Inkling's matched drafter, independent download choice and editable DSpark controls."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.widgets import Input, TextArea
from textual.app import App

from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.backends.llama_cpp.models import LlamaCppModelPanel
from ai_toolbox_cockpit.backends.llama_cpp.server import LlamaCppServerPanel
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd
from ai_toolbox_cockpit.settings import save_backend_settings
from ai_toolbox_cockpit.widgets import SearchableSelect

REPO='unsloth/Inkling-2-Mini-GGUF'
REV='8078d149286eec7193795413b9e0643da6d11fa1'
TARGET='Inkling-2-Mini-UD-Q4_K_XL.gguf'
DRAFT='dspark-Inkling-2-Mini-Q8_0.gguf'


def test_same_repo_target_and_drafter_download_choices_are_distinct(tmp_path):
    class TestApp(App):
        active_platform_id='strix-halo'
        toolbox_catalog=load_toolbox_catalog()
        def compose(self):
            yield LlamaCppModelPanel(load_model_catalog().backends['llama_cpp'],id='models')
    async def check():
        app=TestApp()
        async with app.run_test(size=(140,60)) as pilot:
            panel=app.query_one('#models',LlamaCppModelPanel)
            panel.set_platform('strix-halo')
            await pilot.pause()
            sources={k:v for k,v in panel._download_sources.items() if v['repo']==REPO}
            assert len(sources)==2
            assert sources[REPO]['role']=='model'
            key=next(k for k,v in sources.items() if v['role']=='dspark')
            assert sources[key]['allowed_patterns']==[DRAFT]
            assert sources[key]['revision']==REV
            panel._show_quants(key,[TARGET,DRAFT],{DRAFT:503889952})
            await pilot.pause()
            assert panel._download_quants==[DRAFT]
            app.pop_screen();await pilot.pause()
            with patch('ai_toolbox_cockpit.backends.llama_cpp.models.get_download_cmd',return_value=['hf','download']) as build:
                panel._quant_selected(0)
                await pilot.pause()
                build.assert_called_once_with(REPO,DRAFT,REV)
    asyncio.run(check())


@pytest.mark.parametrize('saved',[None,'baseline','dspark'])
def test_dspark_mode_draft_count_and_context_survive_refresh(tmp_path,saved):
    target=tmp_path/TARGET;target.touch()
    draft=tmp_path/DRAFT
    with draft.open('wb') as f:f.truncate(503889952)  # Sparse metadata fixture, no weight payload.
    if saved:
        save_backend_settings('llama_cpp',{'model_path':str(target),'context':'16777216','dspark_mode':saved,'draft_tokens':'2'})
    class TestApp(App):
        active_platform_id='strix-halo'
        toolbox_catalog=load_toolbox_catalog()
        def compose(self):yield LlamaCppServerPanel(id='server')
    async def check():
        with patch('ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models',return_value=[{'name':TARGET,'path':str(target)}]),patch('ai_toolbox_cockpit.backends.llama_cpp.server.get_local_dspark_models',return_value=[draft]),patch('ai_toolbox_cockpit.backends.llama_cpp.server.detect_container_engines',return_value=[SimpleNamespace(value='podman')]):
            app=TestApp()
            async with app.run_test(size=(140,65)) as pilot:
                await pilot.pause()
                assert app.query_one('#llama-dspark-mode',SearchableSelect).value==(saved or 'baseline')
                if saved: assert app.query_one('#llama-context',Input).value=='16777216'
                mode=app.query_one('#llama-dspark-mode',SearchableSelect)
                mode.value='baseline';await pilot.pause()
                assert '--spec-type' not in app.query_one('#llama-extra-args',TextArea).text
                mode.value='dspark'
                app.query_one('#llama-context',Input).value='16777216'
                app.query_one('#llama-dspark-draft',Input).value='2'
                await pilot.pause()
                app.query_one('#server',LlamaCppServerPanel).refresh_models()
                await pilot.pause()
                assert mode.value=='dspark'
                assert app.query_one('#llama-context',Input).value=='16777216'
                assert app.query_one('#llama-dspark-draft',Input).value=='2'
                assert '--spec-draft-n-max 2' in app.query_one('#llama-extra-args',TextArea).text
                app.query_one('#llama-image',SearchableSelect).value='strix-halo-llama-vulkan-radv-performance'
                await pilot.pause()
                assert not app.query_one('#llama-dspark-zone').display
                assert '--spec-type draft-dspark' not in app.query_one('#llama-extra-args',TextArea).text
    asyncio.run(check())


def test_rejects_older_same_named_drafter(tmp_path):
    target=tmp_path/TARGET;target.touch()
    draft=tmp_path/DRAFT;draft.write_bytes(b'old-layout')
    with patch('ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir',return_value=tmp_path),pytest.raises(ValueError,match='matching DSpark drafter'):
        build_server_cmd(engine='podman',image='example:test',model_path=str(target),draft_model_path=str(draft),context_size=262144,use_fa=True,use_no_mmap=True,custom_args='--spec-type draft-dspark',engine_args=[])
