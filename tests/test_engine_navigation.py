"""R9700 viewers can find each engine, its curated weights and measured controls."""
from contextlib import ExitStack
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from textual.widgets import Button, DataTable, TabbedContent
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.widgets import SearchableSelect


class EngineNavigationTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target, value in [
            ('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed', None),
            ('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update', None),
            ('ai_toolbox_cockpit.app.available_update', None),
            ('ai_toolbox_cockpit.app.load_active_platform', 'r9700'),
            ('ai_toolbox_cockpit.app.save_active_platform', None),
            ('ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models', []),
        ]:
            self.stack.enter_context(patch(target, return_value=value))

    async def test_named_engine_navigation_filters_weights_and_selects_server_recipe(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 70)) as pilot:
            await pilot.pause()
            expected = {
                'ggz14': ('r9700-ggz14-mxfp4-tp1', {'vllm-amd-qwen3-8-27b-mxfp4-mtpfp8', 'vllm-qwen3-8-27b-dflash2-fp8'}),
                'radiance': ('r9700-radiance-fp8-tp2', {'vllm-amd-qwen3-8-27b-mxfp4-mtpfp8', 'vllm-qwen-qwen3-8-27b-fp8'}),
                'vllm': ('r9700-vllm-714-fp8', {'vllm-qwen-qwen3-8-27b-fp8'}),
                'tcclaviger': ('r9700-tcclaviger-flash-next-tp2', {'vllm-tcclaviger-qwen3-8-flash-next-mxfp4'}),
            }
            for selector in ['#server-backend-select', '#model-backend-select']:
                options = dict((value, label) for label, value in app.query_one(selector, SearchableSelect)._options)
                self.assertEqual(set(options), {'llama_cpp', 'r9v', *expected})
                self.assertEqual(options['ggz14'], 'GGZ14')
                self.assertEqual(options['radiance'], 'Radiance')
            for engine, (toolbox, artifacts) in expected.items():
                with self.subTest(engine=engine):
                    app.query_one(TabbedContent).active = 'tab-models'
                    app.query_one('#model-backend-select', SearchableSelect).value = engine
                    await pilot.pause()
                    select = app.query_one('#vllm-download-artifact', SearchableSelect)
                    self.assertGreater(select.region.y, 0)
                    self.assertLess(app.query_one('#vllm-download', Button).region.bottom, app.size.height)
                    self.assertEqual({v for _, v in select._options}, artifacts)
                    rows = app.query_one('#vllm-curated-models', DataTable).rows
                    self.assertEqual({key.value for key in rows}, artifacts)
                    if engine in {'ggz14', 'radiance'}:
                        self.assertEqual(app.query_one('#vllm-download-image', SearchableSelect).value, toolbox)
                    app.query_one(TabbedContent).active = 'tab-servers'
                    app.query_one('#server-backend-select', SearchableSelect).value = engine
                    await pilot.pause()
                    self.assertEqual(app.query_one('#vllm-image', SearchableSelect).value, toolbox)
                    self.assertFalse(app.query_one('#vllm-image').parent.display)
                    self.assertEqual({v for _, v in app.query_one('#vllm-model', SearchableSelect)._options}, artifacts - {'vllm-qwen3-8-27b-dflash2-fp8'})
                    if engine == 'ggz14':
                        self.assertEqual(len(app.query_one('#vllm-gpu-profile', SearchableSelect)._options), 2)
                    if engine == 'tcclaviger':
                        self.assertEqual(app.query_one('#vllm-speculation', SearchableSelect).value, 'mtp')
            # Model Manager from the engine list routes to that engine's weights.
            view = app.query_one('#toolboxes-view')
            view.selected_toolboxes = {'r9700-radiance-fp8-tp2'}
            view.model_manager_pressed()
            await pilot.pause()
            self.assertEqual(app.query_one(TabbedContent).active, 'tab-models')
            self.assertEqual(app.query_one('#model-backend-select', SearchableSelect).value, 'radiance')
            # A different platform restores the generic vLLM model catalogue.
            app.query_one('#platform-select', SearchableSelect).value = 'strix-halo'
            await pilot.pause()
            app.query_one('#model-backend-select', SearchableSelect).value = 'vllm'
            await pilot.pause()
            self.assertGreater(app.query_one('#vllm-curated-models', DataTable).row_count, 4)

    async def test_llama_tested_quantizations_and_all_repositories_are_discoverable(self):
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 70)) as pilot:
            app.query_one(TabbedContent).active = 'tab-models'
            await pilot.pause()
            repo = app.query_one('#llama-download-repo', SearchableSelect)
            self.assertEqual(len(repo._options), 2)
            self.assertIn('UD-Q4_K_XL', repo._options[0][0])
            self.assertIn('UD-Q2_K_XL', repo._options[1][0])
            panel = app.query_one('#model-panel-llama_cpp')
            target = 'Qwen3.8-Flash-Next-UD-Q2_K_XL-*-of-*.gguf'
            with patch.object(app, 'push_screen'):
                panel._show_quants('unsloth/Qwen3.8-Flash-Next-GGUF', ['Qwen3.8-Flash-Next-Q8_0.gguf', target], {})
            self.assertEqual(panel._download_quants[0], target)
            app.query_one('#llama-download-scope', SearchableSelect).value = 'all'
            await pilot.pause()
            self.assertGreater(len(repo._options), 2)
            # Profiles remain visible before downloading any local model.
            app.query_one(TabbedContent).active = 'tab-servers'
            await pilot.pause()
            self.assertEqual(len(app.query_one('#llama-launch-profile', SearchableSelect)._options), 6)
