"""Standalone Radiance server controls and reviewed launch preview."""
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
from .model_manager import get_model, get_models_dir, load_models, model_status, resolved_path, save_models_dir
from .server_runner import CONTAINER_NAME, build_server_cmd


class RadianceServerPanel(BackendServerPanel):
    backend_label = 'Standalone Radiance'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.platform_id = ''
        self._pending_command = []
        self._pending_settings = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes='panel-title')
            yield Static('Native R9700 engine. Download a .rad container in Models. '
                         'Exact inter-GPU communication and FP8 KV cache are used. '
                         'Ctrl+C stops the server and returns here.', classes='panel-copy')
            for key, label in [('engine', 'Container engine'), ('image', 'Toolbox image'),
                               ('model', 'Native model'), ('profile', 'Serving profile'),
                               ('speculation', 'Decoding mode'),
                               ('thinking', 'Thinking effort'), ('vision', 'Image input')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label, classes='inline-label', id=f'radiance-{key}-label')
                    yield SearchableSelect(f'Select {label.lower()}', id=f'radiance-{key}')
                if key == 'profile':
                    yield Static('', id='radiance-profile-summary', classes='panel-copy')
            for key, label, value in [('directory', 'Models directory', str(get_models_dir())),
                                      ('host', 'Bind address', '127.0.0.1'), ('port', 'API port', '18890'),
                                      ('tp', 'GPU count', '2'), ('context', 'Context tokens', '8192'),
                                      ('sessions', 'Server sequence capacity', '1')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label, classes='inline-label', id=f'radiance-{key}-label')
                    yield Input(value=value, id=f'radiance-{key}', disabled=key in {'tp', 'sessions'})
            with Horizontal(classes='inline-row'):
                yield Button('Save Path', id='radiance-server-save-path')
                yield Button('Scan Local', id='radiance-server-scan')
            yield Static('Profiles use the qualified GPU layout and concurrency limit. Context must fit available VRAM. '
                         'Bundled draft mode uses the model’s native DFlash2 or MTP head.', classes='panel-copy')
            yield Button('Start Radiance', id='radiance-start', variant='primary')

    def on_mount(self):
        saved = get_backend_settings('radiance')
        engines = [(item.value, item.value) for item in detect_container_engines()]
        control = self.query_one('#radiance-engine', SearchableSelect)
        control.set_options(engines)
        control.value = saved.get('engine') if saved.get('engine') in dict(engines) else (engines[0][1] if engines else '')
        for key, options, default in [
            ('speculation', [('Target only', 'baseline'), ('Bundled draft', 'bundled')], 'baseline'),
            ('thinking', [('Model default', 'auto'), ('Disabled', 'none'), ('Low', 'low'), ('Medium', 'medium'), ('Extra high', 'xhigh')], 'auto'),
            ('vision', [('Enabled', 'enabled'), ('Disabled', 'disabled')], 'enabled')]:
            control = self.query_one(f'#radiance-{key}', SearchableSelect)
            control.set_options(options)
            control.value = saved.get(key, default)
        for key in ('host', 'port', 'tp', 'context', 'sessions'):
            if key in saved:
                self.query_one(f'#radiance-{key}', Input).value = str(saved[key])
        self.set_platform(self.app.active_platform_id)
        self.refresh_model_inventory()

    def set_platform(self, platform_id):
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        items = [item for item in self.app.toolbox_catalog.platform_toolboxes(platform_id) if item.backend == 'radiance']
        select = self.query_one('#radiance-image', SearchableSelect)
        select.set_options([(item.name, item.id) for item in items])
        select.value = items[0].id if items else ''
        self.query_one('#radiance-start', Button).disabled = not items

    def refresh_model_inventory(self):
        root = get_models_dir()
        self.query_one('#radiance-directory', Input).value = str(root)
        select = self.query_one('#radiance-model', SearchableSelect)
        previous = select.value or get_backend_settings('radiance').get('model', '')
        models = load_models()
        select.set_options([(f'{model.name} — {model_status(model, root)}', model.id) for model in models])
        select.value = previous if previous in {model.id for model in models} else (models[0].id if models else '')

    @on(SearchableSelect.Changed, '#radiance-model')
    def model_changed(self):
        if not self.is_mounted:
            return
        value = self.query_one('#radiance-model', SearchableSelect).value
        if not value:
            return
        model = get_model(value)
        select = self.query_one('#radiance-profile', SearchableSelect)
        previous = select.value or get_backend_settings('radiance').get('profile', 'default')
        profiles = model.available_profiles()
        select.set_options([(profile.name, profile.id) for profile in profiles])
        select.value = previous if previous in {profile.id for profile in profiles} else 'default'
        self.profile_changed()

    @on(SearchableSelect.Changed, '#radiance-profile')
    def profile_changed(self):
        if not self.is_mounted:
            return
        model_id = self.query_one('#radiance-model', SearchableSelect).value
        profile_id = self.query_one('#radiance-profile', SearchableSelect).value
        if not model_id or not profile_id:
            return
        profile = get_model(model_id).serving_profile(profile_id)
        self.query_one('#radiance-tp', Input).value = str(profile.tensor_parallel)
        sessions = self.query_one('#radiance-sessions', Input)
        sessions.disabled = profile.max_sessions == 1
        try:
            count = int(sessions.value)
        except ValueError:
            count = 1
        sessions.value = str(min(max(count, 1), profile.max_sessions))
        self.query_one('#radiance-profile-summary', Static).update(profile.description)

    @on(Button.Pressed, '#radiance-server-save-path')
    def save_path(self):
        if save_models_dir(self.query_one('#radiance-directory', Input).value):
            self.refresh_model_inventory()
            self.app.query_one('#model-panel-radiance').refresh_inventory()
        else:
            self.notify('Could not create or save that directory.', severity='error')

    @on(Button.Pressed, '#radiance-server-scan')
    def scan(self):
        self.refresh_model_inventory()

    @on(Button.Pressed, '#radiance-start')
    def start(self):
        try:
            image_id = self.query_one('#radiance-image', SearchableSelect).value
            image = self.app.toolbox_catalog.toolboxes.get(image_id)
            if not image or image.backend != 'radiance' or image_id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:
                raise ValueError('Select a standalone Radiance image for this platform.')
            values = {key: self.query_one(f'#radiance-{key}', Input).value.strip()
                      for key in ('directory', 'host', 'port', 'tp', 'context', 'sessions')}
            if not values['directory']:
                raise ValueError('Enter a model directory.')
            selections = {key: self.query_one(f'#radiance-{key}', SearchableSelect).value
                          for key in ('engine', 'model', 'profile', 'speculation', 'thinking', 'vision')}
            model = get_model(selections['model'])
            path = resolved_path(model, Path(values['directory']).expanduser().resolve())
            if path is None:
                raise ValueError('Download or repair the selected native model in Models.')
            self._pending_command = build_server_cmd(
                engine=selections['engine'], image=image.image, platform_id=self.platform_id,
                engine_args=list(self.app.toolbox_catalog.runtime_profiles[image.runtime_profile].engine_args),
                model=model, model_path=path, host=values['host'], port=int(values['port']),
                tensor_parallel=int(values['tp']), serving_profile=selections['profile'],
                context_size=int(values['context']), sessions=int(values['sessions']),
                speculation=model.speculation if selections['speculation'] == 'bundled' else 'baseline',
                vision_enabled=selections['vision'] == 'enabled', thinking_effort=selections['thinking'], placement='all_vram')
            self._pending_settings = {**values, **selections, 'models_dir': str(Path(values['directory']).expanduser().resolve())}
        except (ValueError, OSError) as error:
            self.notify(str(error), severity='error')
            return
        self.app.push_screen(ConfirmModal('Start Radiance?\n\n' + shlex.join(self._pending_command),
                                         yes_text='Start', copy_text=shlex.join(self._pending_command)), self._confirmed)

    def _confirmed(self, confirmed):
        if not confirmed:
            return
        save_backend_settings('radiance', self._pending_settings)
        with self.app.suspend():
            run_foreground_server(self._pending_command, self._pending_command[0], CONTAINER_NAME)
