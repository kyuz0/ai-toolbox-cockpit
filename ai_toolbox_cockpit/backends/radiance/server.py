"""Radiance server controls and reviewed launch preview."""
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
    backend_label = 'Radiance'
    DEFAULT_CSS = '''
    RadianceServerPanel .inline-row .inline-label { width: 23; }
    #radiance-draft-tokens { width: 8; }
    #radiance-draft-tokens:disabled { color: #8f969e; background: #171a1d; }
    #radiance-draft-tokens-label.-disabled { color: #8f969e; }
    #radiance-draft-help { height: auto; margin: 0 1 1 25; }
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.platform_id = ''
        self._pending_command = []
        self._pending_settings = {}
        self._context_profile = None
        self._context_values = {}
        self._scaling_values = {}
        self._session_values = {}
        self._draft_model = None
        self._draft_values = {}
        self._speculation_choice = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(self.backend_label, classes='panel-title')
            yield Static('Native R9700 engine. Download a .rad container in Models. '
                         'Ctrl+C stops the server and returns here.', classes='panel-copy')
            for key, label in [('engine', 'Container engine'), ('image', 'Toolbox image'),
                               ('model', 'Native model'), ('profile', 'Serving profile'),
                               ('speculation', 'Speculative decoding'),
                               ('thinking', 'Thinking effort'), ('vision', 'Image input')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label, classes='inline-label', id=f'radiance-{key}-label')
                    yield SearchableSelect(f'Select {label.lower()}', id=f'radiance-{key}')
                if key == 'speculation':
                    with Horizontal(classes='inline-row'):
                        yield Label('Draft tokens', classes='inline-label', id='radiance-draft-tokens-label')
                        yield Input('3', id='radiance-draft-tokens')
                        yield Button('Model default', id='radiance-draft-default')
                    yield Static('', id='radiance-draft-help')
                if key == 'profile':
                    yield Static('', id='radiance-profile-summary', classes='panel-copy')
            for key, label, value in [('directory', 'Models directory', str(get_models_dir())),
                                      ('host', 'Bind address', '127.0.0.1'), ('port', 'API port', '18890'),
                                      ('tp', 'GPU count', '2'), ('context', 'Context tokens', ''),
                                      ('scaling', 'Context scaling', 'none'),
                                      ('sessions', 'Concurrent requests', '1')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label, classes='inline-label', id=f'radiance-{key}-label')
                    yield Input(value=value, id=f'radiance-{key}', disabled=key in {'tp', 'sessions'})
                    if key in {'context', 'scaling'}:
                        yield Button('Profile default', id=f'radiance-{key}-default')
            with Horizontal(classes='inline-row'):
                yield Button('Save Path', id='radiance-server-save-path')
                yield Button('Scan Local', id='radiance-server-scan')
            yield Button('Start Radiance', id='radiance-start', variant='primary')

    def on_mount(self):
        # App CSS overrides DEFAULT_CSS; these dimensions define the panel layout.
        for label in self.query('.inline-label'):
            label.styles.width = 23
        self.query_one('#radiance-draft-tokens', Input).styles.width = 8
        self.query_one('#radiance-draft-help').styles.margin = (0, 1, 1, 25)
        saved = get_backend_settings('radiance')
        engines = [(item.value, item.value) for item in detect_container_engines()]
        control = self.query_one('#radiance-engine', SearchableSelect)
        control.set_options(engines)
        control.value = saved.get('engine') if saved.get('engine') in dict(engines) else (engines[0][1] if engines else '')
        for key, options, default in [
            ('thinking', [('Model default', 'auto'), ('Disabled', 'none'), ('Low', 'low'), ('Medium', 'medium'), ('Extra high', 'xhigh')], 'auto'),
            ('vision', [('Enabled', 'enabled'), ('Disabled', 'disabled')], 'enabled')]:
            control = self.query_one(f'#radiance-{key}', SearchableSelect)
            control.set_options(options)
            control.value = saved.get(key, default)
        for key in ('host', 'port', 'tp', 'context', 'scaling', 'sessions'):
            if key in saved:
                self.query_one(f'#radiance-{key}', Input).value = str(saved[key])
        self.set_platform(self.app.active_platform_id)
        self.refresh_model_inventory()

    def set_platform(self, platform_id):
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        model_id = self.query_one('#radiance-model', SearchableSelect).value
        self._refresh_images(get_model(model_id) if model_id else None)

    def _refresh_images(self, model=None):
        items = [item for item in self.app.toolbox_catalog.platform_toolboxes(self.platform_id)
                 if item.backend == 'radiance'
                 and (model is None or not model.toolbox_ids or item.id in model.toolbox_ids)]
        select = self.query_one('#radiance-image', SearchableSelect)
        previous = select.value or get_backend_settings('radiance').get('image', '')
        select.set_options([(item.name, item.id) for item in items])
        select.value = previous if previous in {item.id for item in items} else (items[0].id if items else '')
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
        self._refresh_images(model)
        saved = get_backend_settings('radiance')
        field = self.query_one('#radiance-draft-tokens', Input)
        if self._draft_model != value:
            if self._draft_model is not None:
                self._draft_values[self._draft_model] = field.value
            depth = self._draft_values.get(value, model.draft_tokens)
            if self._draft_model is None and saved.get('model') == value:
                depth = saved.get('draft_tokens', depth)
            field.value = str(depth)
            self._draft_model = value
        decoding = self.query_one('#radiance-speculation', SearchableSelect)
        choice = self._speculation_choice or saved.get('speculation', 'bundled')
        decoding.set_options([('Disabled', 'baseline'),
                              ({'mtp': 'MTP', 'dflash2': 'DFlash2'}[model.speculation], 'bundled')])
        decoding.value = choice if choice in {'baseline', 'bundled'} else 'bundled'
        self.speculation_changed()
        vision = self.query_one('#radiance-vision', SearchableSelect)
        previous_vision = vision.value or saved.get('vision', 'enabled')
        vision.set_options(([('Enabled', 'enabled')] if model.vision else []) + [('Disabled', 'disabled')])
        vision.value = previous_vision if model.vision else 'disabled'
        vision.disabled = not model.vision
        select = self.query_one('#radiance-profile', SearchableSelect)
        previous = select.value or get_backend_settings('radiance').get('profile', 'default')
        profiles = model.available_profiles()
        select.set_options([(profile.name, profile.id) for profile in profiles])
        select.value = previous if previous in {profile.id for profile in profiles} else model.default_profile
        self.profile_changed()

    @on(SearchableSelect.Changed, '#radiance-speculation')
    def speculation_changed(self):
        if not self.is_mounted:
            return
        choice = self.query_one('#radiance-speculation', SearchableSelect).value
        if choice not in {'baseline', 'bundled'}:
            return
        self._speculation_choice = choice
        self.query_one('#radiance-draft-tokens', Input).disabled = choice == 'baseline'
        self.query_one('#radiance-draft-default', Button).disabled = choice == 'baseline'
        self.query_one('#radiance-draft-tokens-label').set_class(choice == 'baseline', '-disabled')
        model = get_model(self.query_one('#radiance-model', SearchableSelect).value)
        self.query_one('#radiance-draft-help', Static).update('Speculative decoding is disabled.' if choice == 'baseline'
            else f'Tokens proposed per step (1–16). Model default: {model.draft_tokens}.')

    @on(Button.Pressed, '#radiance-draft-default')
    def reset_draft_tokens(self):
        model_id = self.query_one('#radiance-model', SearchableSelect).value
        if model_id:
            self.query_one('#radiance-draft-tokens', Input).value = str(get_model(model_id).draft_tokens)

    @on(SearchableSelect.Changed, '#radiance-profile')
    def profile_changed(self):
        if not self.is_mounted:
            return
        model_id = self.query_one('#radiance-model', SearchableSelect).value
        profile_id = self.query_one('#radiance-profile', SearchableSelect).value
        if not model_id or not profile_id:
            return
        model = get_model(model_id)
        profile = model.serving_profile(profile_id)
        identity = (model_id, profile_id)
        if identity != self._context_profile:
            saved = get_backend_settings('radiance')
            saved_profile = saved.get('profile', 'default')
            if saved_profile == 'default':
                saved_profile = model.default_profile
            defaults = {
                'context': profile.context_size or model.context_size,
                'scaling': profile.rope_scaling or 'none',
                'sessions': profile.default_sessions,
            }
            stores = {'context': self._context_values, 'scaling': self._scaling_values,
                      'sessions': self._session_values}
            for key, default in defaults.items():
                field = self.query_one(f'#radiance-{key}', Input)
                if self._context_profile is not None:
                    stores[key][self._context_profile] = field.value
                value = stores[key].get(identity, default)
                if (self._context_profile is None and saved.get('model') == model_id
                        and saved_profile == profile_id):
                    value = saved.get(key, value)
                field.value = str(value)
            self._context_profile = identity
        self.query_one('#radiance-tp', Input).value = str(profile.tensor_parallel)
        self.query_one('#radiance-sessions', Input).disabled = profile.max_sessions == 1
        for selector in ('#radiance-scaling', '#radiance-scaling-label', '#radiance-scaling-default'):
            self.query_one(selector).display = model.config_format == 'yaml'
        summary = self.query_one('#radiance-profile-summary', Static)
        summary.update(profile.description)
        summary.display = bool(profile.description)

    @on(Button.Pressed, '#radiance-context-default')
    def reset_context(self):
        model_id = self.query_one('#radiance-model', SearchableSelect).value
        if model_id:
            model = get_model(model_id)
            profile = model.serving_profile(self.query_one('#radiance-profile', SearchableSelect).value)
            self.query_one('#radiance-context', Input).value = str(profile.context_size or model.context_size)

    @on(Button.Pressed, '#radiance-scaling-default')
    def reset_scaling(self):
        model_id = self.query_one('#radiance-model', SearchableSelect).value
        if model_id:
            model = get_model(model_id)
            profile = model.serving_profile(self.query_one('#radiance-profile', SearchableSelect).value)
            self.query_one('#radiance-scaling', Input).value = profile.rope_scaling or 'none'

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
                raise ValueError('Select a Radiance image for this platform.')
            values = {key: self.query_one(f'#radiance-{key}', Input).value.strip()
                      for key in ('directory', 'host', 'port', 'tp', 'context', 'scaling', 'sessions')}
            if not values['directory']:
                raise ValueError('Enter a model directory.')
            selections = {key: self.query_one(f'#radiance-{key}', SearchableSelect).value
                          for key in ('engine', 'image', 'model', 'profile', 'speculation', 'thinking', 'vision')}
            draft_tokens = None
            if selections['speculation'] == 'bundled':
                try:
                    draft_tokens = int(self.query_one('#radiance-draft-tokens', Input).value)
                except ValueError:
                    raise ValueError('Draft tokens must be an integer between 1 and 16.') from None
            model = get_model(selections['model'])
            if model.toolbox_ids and image_id not in model.toolbox_ids:
                raise ValueError('Select a matching Radiance toolbox for this model.')
            path = resolved_path(model, Path(values['directory']).expanduser().resolve())
            if path is None:
                raise ValueError('Download or repair the selected native model in Models.')
            self._pending_command = build_server_cmd(
                engine=selections['engine'], image=image.image, platform_id=self.platform_id,
                engine_args=list(self.app.toolbox_catalog.runtime_profiles[image.runtime_profile].engine_args),
                model=model, model_path=path, host=values['host'], port=int(values['port']),
                tensor_parallel=int(values['tp']), serving_profile=selections['profile'],
                context_size=int(values['context']), sessions=int(values['sessions']),
                rope_scaling=values['scaling'] if model.config_format == 'yaml' else None,
                speculation=model.speculation if selections['speculation'] == 'bundled' else 'baseline',
                draft_tokens=draft_tokens,
                vision_enabled=selections['vision'] == 'enabled', thinking_effort=selections['thinking'], placement='all_vram')
            self._pending_settings = {**values, **selections, 'draft_tokens': self.query_one('#radiance-draft-tokens', Input).value, 'models_dir': str(Path(values['directory']).expanduser().resolve())}
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
