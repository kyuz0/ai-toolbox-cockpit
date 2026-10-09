import json
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from textual.widgets import Input, TabbedContent, Button
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.backends.vllm.model_manager import requires_preparation, get_download_cmd
from ai_toolbox_cockpit.backends.vllm.runner import build_server_cmd, apply_toolbox_policy_overrides, VllmCachePaths
from ai_toolbox_cockpit.widgets import SearchableSelect

TOOLBOX = 'r9700-tcclaviger-flash-next-tp2'
MODEL = 'vllm-tcclaviger-qwen3-8-flash-next-mxfp4'


def configuration():
    catalog = load_toolbox_catalog()
    toolbox = catalog.toolboxes[TOOLBOX]
    model = next(x for x in load_model_catalog().backends['vllm'].entries if x['id'] == MODEL)
    return catalog, toolbox, model


class TcclavigerCommands(TestCase):
    def test_acquisition_needs_no_conversion_or_new_image_build(self):
        catalog, toolbox, model = configuration()
        self.assertFalse(toolbox.toolbox_compatible)
        self.assertEqual(toolbox.image, 'docker.io/tcclaviger/vllm:latest')
        self.assertIn(TOOLBOX, catalog.platform('r9700').toolbox_ids)
        self.assertFalse(requires_preparation(model))
        self.assertEqual(len(model['download']['files']), 43)
        self.assertEqual(sum(x['size_bytes'] for x in model['download']['files']), 120325128835)
        self.assertIn(model['revision'], get_download_cmd(model, Path('/models/flash')))

    def test_measured_recipe_and_persistent_ple_are_preserved(self):
        catalog, toolbox, model = configuration()
        policy = apply_toolbox_policy_overrides(model, toolbox.backend_config)
        args = dict(engine='podman', image=toolbox.image,
                    engine_args=list(catalog.runtime_profiles[toolbox.runtime_profile].engine_args),
                    model_id=model['repo'], policy=policy, model_directory=Path('/models/flash'),
                    tensor_parallel=2, max_num_seqs=16, max_model_len='131072',
                    gpu_memory_utilization=.95, speculation='mtp',
                    cache_paths=VllmCachePaths(Path('/cache/hf'), Path('/cache/vllm'),
                                               Path('/cache/triton'), Path('/cache/aiter'), Path('/nvme/flash')))
        cmd = build_server_cmd(**args)
        self.assertIn('--memory=56g', cmd)
        self.assertIn('--memory-swap=56g', cmd)
        self.assertNotIn('--ipc=host', cmd)
        self.assertNotIn('--userns=keep-id', cmd)
        self.assertEqual(cmd[cmd.index('--entrypoint')+1], '/usr/bin/env')
        self.assertEqual(cmd[cmd.index(toolbox.image)+1], '/app/tools/image_entrypoint.sh')
        self.assertIn('/nvme/flash/ple:/app/pleoffload:rw', cmd)
        self.assertIn('/models/flash:/models/target:ro', cmd)
        for flag,value in [('--tensor-parallel-size','2'),('--max-num-seqs','16'),
                           ('--expert-offload-mem','36'),('--ple-cache-gb','4'),('--kv-cache-dtype','fp8')]:
            self.assertEqual(cmd[cmd.index(flag)+1], value)
        self.assertEqual(json.loads(cmd[cmd.index('--speculative-config')+1]),
                         {'method':'mtp','num_speculative_tokens':3})
        self.assertNotIn('/models/draft', ' '.join(cmd))
        for overrides in [dict(model_directory=None),dict(speculation='baseline'),
                          dict(draft_directory=Path('/draft')),dict(tensor_parallel=1)]:
            with self.assertRaises(ValueError):
                build_server_cmd(**(args | overrides))


class TcclavigerUI(IsolatedAsyncioTestCase):
    async def test_model_download_and_server_defaults_reach_command_preview(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed'),
            patch('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update'),
            patch('ai_toolbox_cockpit.app.available_update', return_value=None),
            patch('ai_toolbox_cockpit.app.load_active_platform', return_value='r9700'),
            patch('ai_toolbox_cockpit.app.save_active_platform'),
            patch('ai_toolbox_cockpit.backends.vllm.server.get_backend_settings', return_value={}),
            patch('ai_toolbox_cockpit.backends.vllm.server.save_backend_settings'),
            patch('ai_toolbox_cockpit.backends.vllm.server.checkpoint_ready', return_value=True),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180,55)) as pilot:
                app.query_one(TabbedContent).active='tab-servers'
                app.query_one('#server-backend-select',SearchableSelect).value='vllm'
                await pilot.pause()
                app.query_one('#vllm-image',SearchableSelect).value=TOOLBOX
                await pilot.pause()
                self.assertEqual(app.query_one('#vllm-model',SearchableSelect).value,MODEL)
                self.assertEqual(app.query_one('#vllm-tp',SearchableSelect).value,'2')
                self.assertEqual(app.query_one('#vllm-speculation',SearchableSelect).value,'mtp')
                self.assertEqual(app.query_one('#vllm-seqs',Input).value,'16')
                self.assertEqual(app.query_one('#vllm-context',Input).value,'131072')
                self.assertEqual(app.query_one('#vllm-util',Input).value,'0.95')
                self.assertTrue(app.query_one('#vllm-draft',Input).disabled)
                for field in ['hf','compile','triton','aiter','offload']:
                    app.query_one('#vllm-'+field+'-cache',Input).value=str(Path(directory)/field)
                app.query_one('#vllm-devices',Input).value='1,2'
                panel=app.query_one('#server-panel-vllm')
                panel.start_pressed()
                self.assertIn('ROCR_VISIBLE_DEVICES=1,2',panel._pending_command)
                self.assertIn('--memory-swap=56g',panel._pending_command)
                self.assertIn('/app/tools/image_entrypoint.sh',panel._pending_command)
                self.assertTrue((Path(directory)/'offload'/'ple').is_dir())
                await pilot.press('escape')
                app.query_one(TabbedContent).active = 'tab-models'
                app.query_one('#model-backend-select', SearchableSelect).value = 'vllm'
                app.query_one('#vllm-download-artifact', SearchableSelect).value = MODEL
                await pilot.pause()
                self.assertTrue(app.query_one('#vllm-prepare', Button).disabled)
                self.assertEqual(app.query_one('#vllm-download-prepared', Input).value, '')
                self.assertIn('Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ',
                              app.query_one('#vllm-download-directory', Input).value)
