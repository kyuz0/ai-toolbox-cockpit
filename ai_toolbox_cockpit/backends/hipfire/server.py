"""Hipfire model-aware four-R9700 server controls."""
import shlex
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect
from .model_manager import get_model, get_models_dir, load_models, model_status, save_models_dir
from .server_runner import CONTAINER_NAME, build_server_cmd


class HipfireServerPanel(BackendServerPanel):
    backend_label = 'Hipfire'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.platform_id = ''
        self._pending_command = []
        self._pending_settings = {}
        self._pending_cache = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes='panel-title')
            yield Static('Four R9700 GPUs · native MQ2R · target-only decoding · F32 KV cache. '
                         'One request at a time. Ctrl+C stops the server and returns here.', classes='panel-copy')
            for key, label in [('engine', 'Container engine'), ('image', 'Toolbox image'),
                               ('model', 'Native model / quant')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label, classes='inline-label', id=f'hipfire-{key}-label')
                    yield SearchableSelect(f'Select {label.lower()}', id=f'hipfire-{key}')
            for key, label, value in [('directory', 'Models directory', str(get_models_dir())),
                                      ('host', 'Bind address', '127.0.0.1'), ('port', 'API port', '18910'),
                                      ('context', 'Context tokens', '8192')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label, classes='inline-label', id=f'hipfire-{key}-label')
                    yield Input(value=value, id=f'hipfire-{key}')
            with Horizontal(classes='inline-row'):
                yield Button('Save Path', id='hipfire-server-save-path')
                yield Button('Scan Local', id='hipfire-server-scan')
            yield Button('Start Hipfire', id='hipfire-start', variant='primary')

    def on_mount(self):
        saved = get_backend_settings('hipfire')
        engines = [(item.value, item.value) for item in detect_container_engines()]
        control = self.query_one('#hipfire-engine', SearchableSelect)
        control.set_options(engines)
        control.value = saved.get('engine') if saved.get('engine') in dict(engines) else (engines[0][1] if engines else '')
        for key in ('host', 'port', 'context'):
            if key in saved:
                self.query_one(f'#hipfire-{key}', Input).value = str(saved[key])
        self.set_platform(self.app.active_platform_id)
        self.refresh_model_inventory()

    def set_platform(self, platform_id):
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        items = [item for item in self.app.toolbox_catalog.platform_toolboxes(platform_id) if item.backend == 'hipfire']
        select = self.query_one('#hipfire-image', SearchableSelect)
        select.set_options([(item.name, item.id) for item in items])
        select.value = items[0].id if items else ''
        self.query_one('#hipfire-start', Button).disabled = not items

    def refresh_model_inventory(self):
        root = get_models_dir()
        self.query_one('#hipfire-directory', Input).value = str(root)
        select = self.query_one('#hipfire-model', SearchableSelect)
        previous = select.value or get_backend_settings('hipfire').get('model', '')
        models = load_models()
        select.set_options([(f'{model.name} — {model_status(model, root)}', model.id) for model in models])
        select.value = previous if previous in {model.id for model in models} else (models[0].id if models else '')

    @on(Button.Pressed, '#hipfire-server-save-path')
    def save_path(self):
        if save_models_dir(self.query_one('#hipfire-directory', Input).value):
            self.refresh_model_inventory()
            self.app.query_one('#model-panel-hipfire').refresh_inventory()
        else:
            self.notify('Could not create or save that directory.', severity='error')

    @on(Button.Pressed, '#hipfire-server-scan')
    def scan(self):
        self.refresh_model_inventory()

    @on(Button.Pressed, '#hipfire-start')
    def start(self):
        try:
            image_id = self.query_one('#hipfire-image', SearchableSelect).value
            image = self.app.toolbox_catalog.toolboxes.get(image_id)
            if not image or image.backend != 'hipfire' or image_id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:
                raise ValueError('Select an Hipfire image for this platform.')
            values = {key: self.query_one(f'#hipfire-{key}', Input).value.strip()
                      for key in ('directory', 'host', 'port', 'context')}
            if not values['directory']:
                raise ValueError('Enter a model directory.')
            selections = {key: self.query_one(f'#hipfire-{key}', SearchableSelect).value
                          for key in ('engine', 'model')}
            root = Path(values['directory']).expanduser().resolve()
            self._pending_cache = root / '.hipfire-state'
            self._pending_command = build_server_cmd(
                engine=selections['engine'], image=image.image, platform_id=self.platform_id,
                engine_args=list(self.app.toolbox_catalog.runtime_profiles[image.runtime_profile].engine_args),
                model=get_model(selections['model']), models_dir=root, state_dir=self._pending_cache,
                host=values['host'], port=int(values['port']), context_size=int(values['context']))
            self._pending_settings = {**values, **selections, 'models_dir': str(root)}
        except (ValueError, OSError) as error:
            self.notify(str(error), severity='error')
            return
        self.app.push_screen(ConfirmModal('Start Hipfire?\n\n' + shlex.join(self._pending_command),
                                         yes_text='Start', copy_text=shlex.join(self._pending_command)), self._confirmed)

    def _confirmed(self, confirmed):
        if not confirmed:
            return
        try:
            self._pending_cache.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.notify(f'Cannot create hipfire state directory: {error}', severity='error')
            return
        save_backend_settings('hipfire', self._pending_settings)
        with self.app.suspend():
            run_foreground_server(self._pending_command, self._pending_command[0], CONTAINER_NAME)
