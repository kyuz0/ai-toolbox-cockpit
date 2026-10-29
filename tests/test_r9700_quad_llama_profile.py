"""The quad launch choice preserves allocation and single-card defaults."""
from contextlib import ExitStack
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from textual.widgets import Checkbox, Input, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect


class QuadLlamaProfileTests(IsolatedAsyncioTestCase):
    async def test_quad_allocations_preview_and_single_card_restore(self):
        model = {'name': 'Qwen3.8-27B-UD-Q4_K_XL.gguf',
                 'path': '/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'}
        with ExitStack() as stack:
            for target, value in [
                ('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed', None),
                ('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update', None),
                ('ai_toolbox_cockpit.app.available_update', None),
                ('ai_toolbox_cockpit.app.load_active_platform', 'r9700'),
                ('ai_toolbox_cockpit.app.save_active_platform', None),
                ('ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models', [model]),
                ('ai_toolbox_cockpit.backends.llama_cpp.server.load_default_toolbox', 'r9700-llama-rocm-10-0'),
            ]:
                stack.enter_context(patch(target, return_value=value))
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 65)) as pilot:
                app.query_one(TabbedContent).active = 'tab-servers'
                await pilot.pause()
                choice = app.query_one('#llama-launch-profile', SearchableSelect)
                self.assertEqual(choice.value, 'r9700-llama-rocm-10-qwen27-single')
                choice.value = 'r9700-llama-rocm-10-qwen27-q4xl-four'
                await pilot.pause()
                self.assertEqual(app.query_one('#llama-context', Input).value, '8192')
                self.assertEqual(app.query_one('#llama-devices', Input).value, '0,1,2,3')
                self.assertEqual(app.query_one('#llama-ubatch', Input).value, '256')
                self.assertEqual(app.query_one('#llama-kv-type', SearchableSelect).value, 'f16')
                self.assertFalse(app.query_one('#llama-mtp-enabled', Checkbox).value)
                app.query_one('#llama-engine', SearchableSelect).value = 'podman'
                server = app.query_one('#server-panel-llama_cpp')
                for count in (1, 4, 8, 16):
                    with self.subTest(count=count):
                        app.query_one('#llama-allocation', SearchableSelect).value = f'depth64k-c{count}'
                        await pilot.pause()
                        self.assertEqual(app.query_one('#llama-context', Input).value, str(68096 * count))
                        self.assertEqual(app.query_one('#llama-parallel', Input).value, str(count))
                        self.assertEqual(app.query_one('#llama-ubatch', Input).value, '256')
                        with patch.object(app, 'push_screen') as confirm:
                            server.start_pressed()
                        self.assertTrue(confirm.called)
                        command = server._pending_command
                        for flag, value in [('-c', str(68096 * count)), ('-np', str(count)),
                                            ('-ub', '256'), ('-b', '2048'),
                                            ('--split-mode', 'layer'), ('--tensor-split', '1,1,1,1')]:
                            self.assertEqual(command[command.index(flag) + 1], value)
                        self.assertIn('HIP_VISIBLE_DEVICES=0,1,2,3', command)
                        self.assertNotIn('--model-draft', command)
                choice.value = 'r9700-llama-rocm-10-qwen27-single'
                await pilot.pause()
                self.assertEqual(app.query_one('#llama-devices', Input).value, '0')
                self.assertEqual(app.query_one('#llama-ubatch', Input).value, '512')
                self.assertEqual(app.query_one('#llama-parallel', Input).value, '1')
