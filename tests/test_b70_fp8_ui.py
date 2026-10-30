import tempfile
import unittest
from unittest.mock import patch
from textual.widgets import Input, TabbedContent, Static
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect


class Fp8OnecardUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_engine_applies_single_card_dtype_mtp_and_download(self):
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {'XDG_CONFIG_HOME':root}), patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'), patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'), patch('ai_toolbox_cockpit.app.available_update',return_value=None), patch('ai_toolbox_cockpit.backends.ds4.server.scan_local_models',return_value=[]):
            app=AiToolboxCockpitApp()
            async with app.run_test(size=(160,50)) as pilot:
                app.query_one('#platform-select',SearchableSelect).value='intel-b70'
                await pilot.pause()
                app.query_one(TabbedContent).active='tab-servers'
                app.query_one('#server-backend-select',SearchableSelect).value='b70-fp8-onecard'
                await pilot.pause()
                self.assertEqual(app.query_one('#vllm-image',SearchableSelect).value,'intel-b70-fp8-onecard')
                self.assertEqual(app.query_one('#vllm-dtype',Input).value,'float16')
                self.assertEqual(app.query_one('#vllm-context',Input).value,'32768')
                self.assertEqual(app.query_one('#vllm-tp',SearchableSelect).value,'1')
                self.assertEqual(app.query_one('#vllm-speculation',SearchableSelect).value,'mtp')
                self.assertIn(('MTP · 5 draft tokens (embedded)', 'mtp'), app.query_one('#vllm-speculation',SearchableSelect)._options)
                self.assertIn('prefix caching off',str(app.query_one('#vllm-allocation-note',Static).render()))
                app.query_one(TabbedContent).active='tab-models'
                app.query_one('#model-backend-select',SearchableSelect).value='b70-fp8-onecard'
                await pilot.pause()
                entries=app.query_one('#model-panel-vllm').engine_entries()
                self.assertEqual([e['repo'] for e in entries],['Qwen/Qwen3.8-27B-FP8'])
                self.assertEqual(entries[0]['revision'],'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')
