"""Engine installation stays separate from measured model launch choices."""
import unittest
from unittest.mock import patch

from textual.widgets import Input, TabbedContent, TextArea
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.catalog import load_toolbox_catalog
from ai_toolbox_cockpit.widgets import SearchableSelect


class R9700CatalogueTests(unittest.TestCase):
    def test_each_image_has_one_installable_entry(self):
        catalog = load_toolbox_catalog()
        rows = catalog.platform_toolboxes('r9700')
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({t.image for t in rows}), len(rows))
        vulkan = catalog.toolboxes['r9700-llama-vulkan-radv']
        self.assertIn('RADV_PERFTEST=nogttspill', catalog.runtime_profiles[vulkan.runtime_profile].engine_args)
        for toolbox in rows:
            for old_id in toolbox.backend_config.get('legacy_toolbox_ids', []):
                self.assertNotIn(old_id, catalog.toolboxes)
                self.assertEqual(catalog.resolve_toolbox_id(old_id), toolbox.id)


class R9700ProfileUITests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_presets_migrate_and_server_profiles_keep_measured_settings(self):
        qwen = {'name': 'Qwen3.8-27B-UD-Q4_K_XL.gguf', 'path': '/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'}
        flash = {'name': 'Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf', 'path': '/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf'}
        with (
            patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'),
            patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'),
            patch('ai_toolbox_cockpit.app.available_update', return_value=None),
            patch('ai_toolbox_cockpit.app.load_active_platform', return_value='r9700'),
            patch('ai_toolbox_cockpit.app.save_active_platform'),
            patch('ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models', return_value=[qwen, flash]),
            patch('ai_toolbox_cockpit.backends.llama_cpp.server.load_default_toolbox', return_value='r9700-llama-vulkan-qwen27-dual-q8'),
            patch('ai_toolbox_cockpit.backends.vllm.server.load_default_toolbox', return_value='r9700-radiance-mxfp4-tp2'),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 65)) as pilot:
                await pilot.pause()
                self.assertEqual(app.query_one('#llama-image', SearchableSelect).value, 'r9700-llama-vulkan-radv')
                self.assertEqual(app.query_one('#llama-launch-profile', SearchableSelect).value, 'r9700-llama-vulkan-qwen27-dual-q8')
                self.assertEqual(app.query_one('#vllm-image', SearchableSelect).value, 'r9700-radiance-fp8-tp2')
                self.assertEqual(app.query_one('#vllm-model', SearchableSelect).value, 'vllm-amd-qwen3-8-27b-mxfp4-mtpfp8')
                app.query_one(TabbedContent).active = 'tab-servers'
                app.query_one('#llama-allocation', SearchableSelect).value = 'episode'
                await pilot.pause()
                self.assertEqual(app.query_one('#llama-context', Input).value, '1089536')
                self.assertEqual(app.query_one('#llama-parallel', Input).value, '16')
                self.assertEqual(app.query_one('#llama-ubatch', Input).value, '256')
                app.query_one('#llama-launch-profile', SearchableSelect).value = 'r9700-llama-vulkan-flash-next-disk-ple'
                await pilot.pause()
                self.assertEqual(app.query_one('#llama-model', SearchableSelect).value, flash['path'])
                self.assertEqual(app.query_one('#llama-context', Input).value, '68096')
                self.assertEqual(app.query_one('#llama-parallel', Input).value, '1')
                self.assertEqual(app.query_one('#llama-load-mode', SearchableSelect).value, 'mmap')
                self.assertIn('--lazy-mode on', app.query_one('#llama-extra-args', TextArea).text)
                app.query_one('#server-backend-select', SearchableSelect).value = 'vllm'
                await pilot.pause()
                panel = app.query_one('#server-panel-vllm')
                for model, enabled in [('vllm-qwen-qwen3-8-27b-fp8','0'), ('vllm-amd-qwen3-8-27b-mxfp4-mtpfp8','1')]:
                    app.query_one('#vllm-model', SearchableSelect).value = model
                    await pilot.pause()
                    policy = panel._effective_policy(panel._policy_by_id[model])
                    self.assertEqual(policy['env']['RADIANCE_MXFP4'], enabled)
                    self.assertEqual(policy['attention_backend'], 'R4D')
                    self.assertEqual(policy['speculation'], {})

    async def test_ggz14_single_entry_selects_matching_build_policy_and_download(self):
        from textual.widgets import Button
        with (
            patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'),
            patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'),
            patch('ai_toolbox_cockpit.app.available_update', return_value=None),
            patch('ai_toolbox_cockpit.app.load_active_platform', return_value='r9700'),
            patch('ai_toolbox_cockpit.app.save_active_platform'),
            patch('ai_toolbox_cockpit.backends.vllm.server.load_default_toolbox', return_value='r9700-ggz14-mxfp4-tp2'),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 65)) as pilot:
                await pilot.pause()
                self.assertEqual(app.query_one('#vllm-image', SearchableSelect).value, 'r9700-ggz14-mxfp4-tp1')
                gpu = app.query_one('#vllm-gpu-profile', SearchableSelect)
                self.assertEqual(gpu.value, 'r9700-ggz14-mxfp4-tp2')
                panel = app.query_one('#server-panel-vllm')
                for count in [1, 2, 1]:
                    app.query_one('#vllm-devices', Input).value = '7'
                    gpu.value = f'r9700-ggz14-mxfp4-tp{count}'
                    await pilot.pause()
                    toolbox = panel._selected_toolbox()
                    self.assertTrue(toolbox.image.endswith(f':ggz14-tp{count}'))
                    self.assertEqual(app.query_one('#vllm-tp', SearchableSelect).value, str(count))
                    self.assertEqual(app.query_one('#vllm-devices', Input).value, '')
                    policy = panel._effective_policy(panel._policy_by_id[app.query_one('#vllm-model', SearchableSelect).value])
                    self.assertEqual(policy['env']['HIP_VISIBLE_DEVICES'], '0' if count == 1 else '0,1')
                    self.assertEqual(policy['speculation']['dflash2']['gpu_memory_utilization'], 0.95 if count == 1 else 0.92)
                    app.query_one('#vllm-engine', SearchableSelect).value = 'podman'
                    with patch.object(app, 'push_screen') as push:
                        panel.pull_build_pressed()
                    callback = push.call_args.args[1]
                    with patch.object(panel, '_pull_build_confirmed') as pull:
                        callback(False)
                    self.assertEqual(pull.call_args.args, (False, ['podman', 'pull', toolbox.image]))
