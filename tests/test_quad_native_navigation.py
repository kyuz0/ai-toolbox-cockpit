"""Model selection precedes concrete native launch previews; no GPU or downloads."""
from contextlib import ExitStack
from dataclasses import replace
import json
from importlib.resources import files
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from textual.widgets import Label, TabbedContent
from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.catalog import load_toolbox_catalog
from ai_toolbox_cockpit.catalog.schema import ToolboxCatalog
from ai_toolbox_cockpit.runtime.engines import ContainerEngine
from ai_toolbox_cockpit.settings import save_backend_settings
from ai_toolbox_cockpit.widgets import SearchableSelect


def candidate_catalog():
    data = json.loads(files('ai_toolbox_cockpit.assets').joinpath('toolboxes.json').read_text())
    platform = next(p for p in data['platforms'] if p['id'] == 'r9700')
    base = next(t for t in data['toolboxes'] if t['id'] == 'r9700-radiance-rocm-10-1')
    for backend in ['affinity', 'hipfire', 'strata']:
        identifier = f'r9700-{backend}-ui-fixture'
        data['toolboxes'].append(base | {'id': identifier, 'backend': backend, 'name': backend,
            'container_name': identifier, 'image': f'docker.io/example/{backend}:ui-fixture'})
        platform['toolbox_ids'].append(identifier)
    return ToolboxCatalog.from_dict(data)


class QuadNativeNavigationTests(IsolatedAsyncioTestCase):
    def setUp(self):
        import tempfile
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for target, value in [('ai_toolbox_cockpit.app.load_toolbox_catalog', candidate_catalog()),
            ('ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed', None),
            ('ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update', None),
            ('ai_toolbox_cockpit.app.available_update', None),
            ('ai_toolbox_cockpit.app.load_active_platform', 'r9700'),
            ('ai_toolbox_cockpit.app.save_active_platform', None),
            ('ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models', [])]:
            self.stack.enter_context(patch(target, return_value=value))
        for backend in ['affinity', 'hipfire', 'strata']:
            for panel in ['models', 'server']:
                target = f'ai_toolbox_cockpit.backends.{backend}.{panel}.detect_container_engines'
                if panel == 'models' and backend != 'strata':
                    continue
                self.stack.enter_context(patch(target, return_value=(ContainerEngine.PODMAN,)))
            save_backend_settings(backend, {'models_dir': str(self.directory / backend)})
        for backend in ['affinity', 'hipfire']:
            self.stack.enter_context(patch(f'ai_toolbox_cockpit.backends.{backend}.models.get_hf_token', return_value=''))

    async def test_curated_native_models_modes_labels_and_launch_previews(self):
        from ai_toolbox_cockpit.backends.affinity.model_manager import load_models as affinity_models, artifact_paths
        from ai_toolbox_cockpit.backends.hipfire.model_manager import load_models as hipfire_models
        affinity = affinity_models()[0]
        for path, size in zip(artifact_paths(affinity, self.directory / 'affinity'),
                              [affinity.size_bytes, affinity.draft_size_bytes, affinity.tokenizer_size_bytes]):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as stream:
                stream.truncate(size)
        hipfire = hipfire_models()[0]
        path = self.directory / 'hipfire' / hipfire.directory / hipfire.filename
        path.parent.mkdir(parents=True)
        with path.open('wb') as stream:
            stream.truncate(hipfire.size_bytes)
        app = AiToolboxCockpitApp()
        async with app.run_test(size=(180, 70)) as pilot:
            for backend in ['affinity', 'hipfire', 'strata']:
                app.query_one(TabbedContent).active = 'tab-models'
                app.query_one('#model-backend-select', SearchableSelect).value = backend
                await pilot.pause()
                self.assertEqual(app.query_one('#model-content-switcher').current, f'model-panel-{backend}')
                selector = f'#{backend}-download-model'
                options = app.query_one(selector, SearchableSelect)._options
                self.assertEqual(len(options), 1)
                self.assertIn('0731' if backend != 'strata' else 'v41', options[0][1])
                app.query_one(TabbedContent).active = 'tab-servers'
                app.query_one('#server-backend-select', SearchableSelect).value = backend
                await pilot.pause()
                self.assertEqual(app.query_one('#server-content-switcher').current, f'server-panel-{backend}')
                for key in ['engine', 'image', 'model']:
                    self.assertIsInstance(app.query_one(f'#{backend}-{key}-label'), Label)
                if backend != 'hipfire':
                    self.assertEqual({v for _, v in app.query_one(f'#{backend}-speculation', SearchableSelect)._options}, {'baseline', 'dspark'})
                if backend != 'strata':
                    with patch.object(app, 'push_screen') as confirm:
                        app.query_one(f'#server-panel-{backend}').start()
                        command = confirm.call_args.args[0].copy_text
                        self.assertIn('--gpus 4' if backend == 'affinity' else '--tp 4', command)
                        self.assertNotIn('vllm', command)
            # A platform switch must never offer AMD V4.1 files in the B70 panel.
            app.query_one('#platform-select', SearchableSelect).value = 'intel-b70'
            await pilot.pause()
            models = app.query_one('#strata-model', SearchableSelect)._options
            self.assertEqual(len(models), 1)
            self.assertNotIn('v41', models[0][1])
            self.assertEqual({v for _, v in app.query_one('#strata-speculation', SearchableSelect)._options}, {'baseline', 'mtp'})
