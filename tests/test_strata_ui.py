import asyncio
import tempfile
from unittest.mock import patch
from pathlib import Path
from textual.widgets import Input,TabbedContent
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect,ConfirmModal


def test_strata_b70_model_and_server_flow():
    async def check():
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ',{'XDG_CONFIG_HOME':root}), \
             patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'), \
             patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'), \
             patch('ai_toolbox_cockpit.app.available_update',return_value=None), \
             patch('ai_toolbox_cockpit.backends.ds4.server.scan_local_models',return_value=[]):
            app=AiToolboxCockpitApp()
            async with app.run_test(size=(160,50)) as pilot:
                app.query_one('#platform-select',SearchableSelect).value='intel-b70'
                await pilot.pause()
                app.query_one(TabbedContent).active='tab-servers'
                app.query_one('#server-backend-select',SearchableSelect).value='strata'
                await pilot.pause()
                assert app.query_one('#strata-image',SearchableSelect).value=='intel-b70-strata'
                assert app.query_one('#strata-model',SearchableSelect).value=='strata-qwen3-8-flash-next-atomicchat-ad427'
                assert app.query_one('#strata-context',Input).value=='68096'
                app.save_screenshot('/tmp/b70-strata-server.svg')
                assert app.query_one('#strata-speculation',SearchableSelect).value=='mtp'
                app.query_one(TabbedContent).active='tab-models'
                app.query_one('#model-backend-select',SearchableSelect).value='strata'
                await pilot.pause()
                assert app.query_one('#strata-download-model',SearchableSelect).value=='strata-qwen3-8-flash-next-atomicchat-ad427'
                app.query_one('#strata-models-dir',Input).value=root+'/models'
                await pilot.click('#strata-download')
                await pilot.pause()
                assert isinstance(app.screen,ConfirmModal)
                app.save_screenshot('/tmp/b70-strata-download.svg')
                await pilot.press('escape')
    asyncio.run(check())
