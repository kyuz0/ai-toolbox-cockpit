import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch
from textual.widgets import Input, TabbedContent, TextArea
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect
from ai_toolbox_cockpit.backends.llama_cpp.models import LlamaCppModelPanel
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import save_models_dir


def test_b70_flash_download_and_server_defaults():
    async def check():
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {'XDG_CONFIG_HOME':root}), \
             patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'), \
             patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'), \
             patch('ai_toolbox_cockpit.app.available_update',return_value=None), \
             patch('ai_toolbox_cockpit.backends.ds4.server.scan_local_models',return_value=[]):
            models=Path(root)/'models';models.mkdir()
            main=models/'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf';main.write_bytes(b'UI fixture only')
            draft=models/'Qwen3.8-Flash-Next-MTP-GGUF'/'Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf';draft.parent.mkdir();draft.write_bytes(b'UI fixture only')
            assert save_models_dir(str(models))
            app=AiToolboxCockpitApp()
            async with app.run_test(size=(170,55)) as pilot:
                app.query_one('#platform-select',SearchableSelect).value='intel-b70'
                await pilot.pause()
                app.query_one(TabbedContent).active='tab-models'
                app.query_one('#model-backend-select',SearchableSelect).value='llama_cpp'
                await pilot.pause()
                panel=app.query_one(LlamaCppModelPanel)
                assert set(panel._download_sources)=={'unsloth/Qwen3.8-27B-GGUF','AtomicChat/Qwen3.8-Flash-Next-GGUF','dzannotti/Qwen3.8-Flash-Next-MTP-GGUF'}
                repo='AtomicChat/Qwen3.8-Flash-Next-GGUF'
                app.query_one('#llama-download-repo',SearchableSelect).value=repo
                await pilot.pause()
                panel._show_quants(repo,['Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64','Unqualified quant'],{})
                assert panel._download_quants==['Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64']
                await pilot.pause()
                app.save_screenshot('/tmp/b70-flash-download.svg')
                await pilot.click('#btn_cancel')
                await pilot.pause()
                assert app.screen is app.screen_stack[0]
                app.query_one(TabbedContent).active='tab-servers'
                app.query_one('#server-backend-select',SearchableSelect).value='llama_cpp'
                app.query_one('#llama-image',SearchableSelect).value='intel-b70-llama-sycl-flash-next'
                await pilot.pause()
                assert app.query_one('#llama-context',Input).value=='68096'
                assert app.query_one('#llama-mtp-model',SearchableSelect).value==str(draft)
                args=app.query_one('#llama-extra-args',TextArea).text
                assert '--spec-draft-device SYCL1' in args
                assert '--cache-type-v q4_1' in args
                assert '--split-mode layer' in args
                app.save_screenshot('/tmp/b70-flash-server.svg')
    asyncio.run(check())
