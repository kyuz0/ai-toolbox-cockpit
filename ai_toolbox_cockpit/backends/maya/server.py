"""Maya model and native speculative decoding controls."""
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
from .model_manager import load_models, get_model, get_models_dir, prepared
from .server_runner import build_server_cmd, CONTAINER_NAME


class MayaServerPanel(BackendServerPanel):
    backend_label = 'Maya'
    platform_id = ''

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label('Maya', classes='panel-title')
            yield Static('GLM-5.3-Flash Maya-S24 or DeepSeek-V4.1-Flash-1017 Maya-S on 128 GB Strix Halo. One sequence; sampled generation with high reasoning. Prepare your model in Models first. Ctrl+C stops the server.', classes='panel-copy')
            for key, label in [('engine', 'Container engine'), ('image', 'Toolbox image'), ('model', 'Model / quantization'), ('speculation', 'Decoding mode')]:
                with Horizontal(classes='inline-row'):
                    label_widget = Label(label, id=f'maya-{key}-label', classes='inline-label')
                    label_widget.styles.width = 23
                    yield label_widget
                    yield SearchableSelect(f'Select {label.lower()}', id=f'maya-{key}')
            for key, label, value in [('directory', 'Models directory', str(get_models_dir())), ('host', 'Bind address', '127.0.0.1'), ('port', 'API port', '8000'), ('context', 'Context tokens', '68096')]:
                with Horizontal(classes='inline-row'):
                    label_widget = Label(label, id=f'maya-{key}-label', classes='inline-label')
                    label_widget.styles.width = 23
                    yield label_widget
                    yield Input(value=value, id=f'maya-{key}')
            yield Static('', id='maya-ready', classes='panel-copy')
            yield Button('Start Maya', id='maya-start', variant='primary')

    def on_mount(self):
        saved = get_backend_settings('maya')
        engines = [(e.value, e.value) for e in detect_container_engines()]
        control = self.query_one('#maya-engine', SearchableSelect)
        control.set_options(engines)
        control.value = saved.get('engine') if saved.get('engine') in dict(engines) else (engines[0][1] if engines else '')
        for key in ('host', 'port', 'context'):
            if key in saved:
                self.query_one(f'#maya-{key}', Input).value = str(saved[key])
        self.set_platform(self.app.active_platform_id)

    def set_platform(self, platform_id):
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        items = [t for t in self.app.toolbox_catalog.platform_toolboxes(platform_id) if t.backend == 'maya']
        control = self.query_one('#maya-image', SearchableSelect)
        control.set_options([(t.name, t.id) for t in items])
        control.value = items[0].id if items else ''
        self.query_one('#maya-start', Button).disabled = not items
        self.refresh_model_inventory()

    def refresh_model_inventory(self):
        if not self.is_mounted:
            return
        models = load_models() if self.platform_id == 'strix-halo' else []
        control = self.query_one('#maya-model', SearchableSelect)
        previous = control.value or get_backend_settings('maya').get('model')
        control.set_options([(m.name, m.id) for m in models])
        control.value = previous if previous in {m.id for m in models} else (models[0].id if models else '')
        self.update_selection()

    @on(SearchableSelect.Changed, '#maya-model')
    def update_selection(self):
        if not self.is_mounted:
            return
        try:
            model = get_model(self.query_one('#maya-model', SearchableSelect).value)
        except ValueError:
            self.query_one('#maya-ready', Static).update('No Maya profile for this platform.')
            return
        control = self.query_one('#maya-speculation', SearchableSelect)
        previous = control.value or get_backend_settings('maya').get('speculation')
        control.set_options(model.modes)
        control.value = previous if previous in {v for _, v in model.modes} else model.modes[-1][1]
        root = Path(self.query_one('#maya-directory', Input).value).expanduser()
        self.query_one('#maya-ready', Static).update('Ready' if prepared(model, root) else 'Download and prepare the selected model in Models.')

    @on(Button.Pressed, '#maya-start')
    def start(self):
        try:
            toolbox = self.app.toolbox_catalog.toolboxes.get(self.query_one('#maya-image', SearchableSelect).value)
            if not toolbox or toolbox.backend != 'maya' or toolbox.id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:
                raise ValueError('Select the Maya toolbox for this platform')
            values = {k: self.query_one(f'#maya-{k}', Input).value.strip() for k in ('directory', 'host', 'port', 'context')}
            if not values['directory']:
                raise ValueError('Enter a models directory')
            choices = {k: self.query_one(f'#maya-{k}', SearchableSelect).value for k in ('engine', 'model', 'speculation')}
            self._command = build_server_cmd(engine=choices['engine'], image=toolbox.image,
                engine_args=self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args,
                platform_id=self.platform_id, model=get_model(choices['model']), directory=Path(values['directory']).expanduser(),
                host=values['host'], port=int(values['port']), context_size=int(values['context']), speculation=choices['speculation'])
            self._settings = {**choices, **values, 'models_dir': str(Path(values['directory']).expanduser().resolve())}
        except (ValueError, OSError) as error:
            self.notify(str(error), severity='error')
            return
        self.app.push_screen(ConfirmModal('Start Maya?\n\n' + shlex.join(self._command), yes_text='Start', copy_text=shlex.join(self._command)), self._confirmed)

    def _confirmed(self, confirmed):
        if confirmed:
            save_backend_settings('maya', self._settings)
            with self.app.suspend():
                run_foreground_server(self._command, self._command[0], CONTAINER_NAME)
