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
                'vllm-radiance': ('r9700-radiance-fp8-tp2', {'vllm-amd-qwen3-8-27b-mxfp4-mtpfp8', 'vllm-qwen-qwen3-8-27b-fp8'}),
                'vllm': ('r9700-vllm-714-fp8', {'vllm-qwen-qwen3-8-27b-fp8'}),
                'tcclaviger': ('r9700-tcclaviger-flash-next-tp2', {'vllm-tcclaviger-qwen3-8-flash-next-mxfp4'}),
            }
            for selector in ['#server-backend-select', '#model-backend-select']:
                options = dict((value, label) for label, value in app.query_one(selector, SearchableSelect)._options)
                self.assertEqual(set(options), {'llama_cpp', 'r9v', 'radiance', 'affinity', 'hipfire', 'strata', *expected})
                self.assertEqual(options['ggz14'], 'GGZ14')
                self.assertEqual(options['vllm-radiance'], 'vLLM Radiance (legacy)'
                )
                self.assertEqual(options['radiance'], 'Standalone Radiance')
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
                    if engine in {'ggz14', 'vllm-radiance'}:
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
            self.assertEqual(app.query_one('#model-backend-select', SearchableSelect).value, 'vllm-radiance')
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
            labels = {value: label for label, value in repo._options}
            self.assertEqual(set(labels), {'unsloth/Qwen3.8-27B-GGUF', 'unsloth/Qwen3.8-Flash-Next-GGUF', 'unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF'})
            self.assertIn('UD-Q4_K_XL', labels['unsloth/Qwen3.8-27B-GGUF'])
            self.assertIn('UD-Q4_K_XL', labels['unsloth/Qwen3.8-Flash-Next-GGUF'])
            self.assertIn('UD-Q2_K_XL', labels['unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF'])
            panel = app.query_one('#model-panel-llama_cpp')
            target = 'Qwen3.8-Flash-Next-UD-Q4_K_XL-*-of-*.gguf'
            with patch.object(app, 'push_screen'):
                panel._show_quants('unsloth/Qwen3.8-Flash-Next-GGUF', ['Qwen3.8-Flash-Next-Q8_0.gguf', target], {})
            self.assertEqual(panel._download_quants[0], target)
            app.query_one('#llama-download-scope', SearchableSelect).value = 'all'
            await pilot.pause()
            self.assertGreater(len(repo._options), 2)
            # Profiles remain visible before downloading any local model.
            app.query_one(TabbedContent).active = 'tab-servers'
            await pilot.pause()
            self.assertEqual(len(app.query_one('#llama-launch-profile', SearchableSelect)._options), 9)

    async def test_standalone_radiance_routes_native_models_and_qualified_gpu_profile(self):
        import tempfile
        from textual.widgets import Input
        from ai_toolbox_cockpit.backends.radiance.model_manager import load_models
        from ai_toolbox_cockpit.runtime.engines import ContainerEngine
        from ai_toolbox_cockpit.settings import save_backend_settings
        with tempfile.TemporaryDirectory() as directory, patch(
            'ai_toolbox_cockpit.backends.radiance.server.detect_container_engines',
            return_value=(ContainerEngine.PODMAN,),
        ), patch('ai_toolbox_cockpit.backends.radiance.models.get_hf_token', return_value='test-only-token'):
            save_backend_settings('radiance', {'models_dir':directory})
            model = load_models()[0]
            target = Path(directory)/model.directory/model.filename
            target.parent.mkdir()
            # Sparse metadata fixture only; no model download or inference.
            with target.open('wb') as stream:
                stream.truncate(model.size_bytes)
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 70)) as pilot:
                app.query_one(TabbedContent).active = 'tab-models'
                app.query_one('#model-backend-select', SearchableSelect).value = 'radiance'
                await pilot.pause()
                self.assertEqual(app.query_one('#model-content-switcher').current, 'model-panel-radiance')
                self.assertEqual(app.query_one('#radiance-local-models', DataTable).row_count, 3)
                models = app.query_one('#model-panel-radiance')
                with patch.object(app, 'push_screen') as confirm:
                    models.download_pressed()
                    self.assertIn(model.revision, confirm.call_args.args[0].copy_text)
                    self.assertIn(model.filename, confirm.call_args.args[0].copy_text)
                app.query_one(TabbedContent).active = 'tab-servers'
                app.query_one('#server-backend-select', SearchableSelect).value = 'radiance'
                await pilot.pause()
                self.assertEqual(app.query_one('#server-content-switcher').current, 'server-panel-radiance')
                self.assertEqual(app.query_one('#radiance-tp', Input).value, '2')
                self.assertTrue(app.query_one('#radiance-tp', Input).disabled)
                server = app.query_one('#server-panel-radiance')
                with patch.object(app, 'push_screen') as confirm:
                    server.start()
                    command = confirm.call_args.args[0].copy_text
                    self.assertIn('radiance --model', command)
                    self.assertIn('--tp 2', command)
                    self.assertIn('--num-speculative-tokens 0', command)
                    self.assertNotIn('vllm serve', command)
                profile = app.query_one('#radiance-profile', SearchableSelect)
                self.assertEqual({value for _, value in profile._options}, {'default', 'quad-tp4-144'})
                profile.value = 'quad-tp4-144'
                await pilot.pause()
                self.assertEqual(app.query_one('#radiance-tp', Input).value, '4')
                self.assertFalse(app.query_one('#radiance-sessions', Input).disabled)
                app.query_one('#radiance-sessions', Input).value = '16'
                with patch.object(app, 'push_screen') as confirm:
                    server.start()
                    command = confirm.call_args.args[0].copy_text
                    self.assertIn('--tp 4', command)
                    self.assertIn('--max-num-seqs 16', command)
                    self.assertIn('--checkpoint-slots 144', command)
                self.assertEqual(app.query_one('#radiance-local-models', DataTable).row_count, 3)
                profile.value = 'default'
                await pilot.pause()
                self.assertEqual(app.query_one('#radiance-tp', Input).value, '2')
                self.assertEqual(app.query_one('#radiance-sessions', Input).value, '1')
                flash = next(item for item in load_models() if item.tensor_parallel == 4)
                app.query_one('#radiance-model', SearchableSelect).value = flash.id
                await pilot.pause()
                self.assertEqual(app.query_one('#radiance-tp', Input).value, '4')
                self.assertFalse(app.query_one('#radiance-sessions', Input).disabled)
                flash_target = Path(directory)/flash.directory/flash.filename
                flash_target.parent.mkdir()
                with flash_target.open('wb') as stream:
                    stream.truncate(flash.size_bytes)
                app.query_one('#radiance-sessions', Input).value = '16'
                with patch.object(app, 'push_screen') as confirm:
                    server.start()
                    command = confirm.call_args.args[0].copy_text
                    self.assertIn('--max-num-seqs 16', command)
                    self.assertIn('--ngram-placement ram', command)
                app.query_one('#server-backend-select', SearchableSelect).value = 'vllm-radiance'
                await pilot.pause()
                self.assertEqual(app.query_one('#server-content-switcher').current, 'server-panel-vllm')
