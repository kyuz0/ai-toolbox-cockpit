"""Strata's qualified Intel B70 serving controls."""
import shlex
from pathlib import Path
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal,VerticalScroll
from textual.widgets import Button,Input,Label,Static
from ai_toolbox_cockpit.backends.base import BackendServerPanel
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.server_process import run_foreground_server
from ai_toolbox_cockpit.settings import get_backend_settings,save_backend_settings
from ai_toolbox_cockpit.widgets import ConfirmModal,SearchableSelect
from .model_manager import load_models,get_model,get_models_dir,prepared
from .server_runner import build_server_cmd,CONTAINER_NAME


class StrataServerPanel(BackendServerPanel):
    DEFAULT_CSS = 'StrataServerPanel .inline-label { width: 23; margin-right: 1; }'
    backend_label='Strata'
    platform_id=''

    def compose(self)->ComposeResult:
        with VerticalScroll():
            yield Label('Strata · two Intel B70 GPUs',classes='panel-title')
            yield Static('Flash Next AtomicChat AD4.27 with INT8 KV and host-memory KV streaming. Prepare the model in Models first. This profile uses both B70 cards and one sequence. Ctrl+C stops the server.',classes='panel-copy')
            for key,label in [('engine','Container engine'),('image','Toolbox image'),('model','Model / quantization'),('speculation','Decoding mode')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label,id=f'strata-{key}-label',classes='inline-label')
                    yield SearchableSelect(f'Select {label.lower()}',id=f'strata-{key}')
            for key,label,value in [('directory','Models directory',str(get_models_dir())),('host','Bind address','127.0.0.1'),('port','API port','8000'),('context','Context tokens','68096')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label,id=f'strata-{key}-label',classes='inline-label')
                    yield Input(value=value,id=f'strata-{key}')
            yield Static('',id='strata-ready',classes='panel-copy')
            yield Button('Start Strata',id='strata-start',variant='primary')

    def on_mount(self):
        saved=get_backend_settings('strata')
        engines=[(e.value,e.value) for e in detect_container_engines()]
        control=self.query_one('#strata-engine',SearchableSelect);control.set_options(engines);control.value=saved.get('engine') if saved.get('engine') in dict(engines) else (engines[0][1] if engines else '')
        control=self.query_one('#strata-speculation',SearchableSelect);control.set_options([('MTP · three draft tokens','mtp'),('Target only','baseline')]);control.value=saved.get('speculation','mtp')
        for key in ('host','port','context'):
            if key in saved:self.query_one(f'#strata-{key}',Input).value=str(saved[key])
        self.set_platform(self.app.active_platform_id);self.refresh_model_inventory()

    def set_platform(self,platform_id):
        self.platform_id=platform_id
        if not self.is_mounted:return
        items=[t for t in self.app.toolbox_catalog.platform_toolboxes(platform_id) if t.backend=='strata']
        control=self.query_one('#strata-image',SearchableSelect);control.set_options([(t.name,t.id) for t in items]);control.value=items[0].id if items else ''
        self.query_one('#strata-start',Button).disabled=not items
        self.refresh_model_inventory()

    def refresh_model_inventory(self):
        if not self.is_mounted:return
        root=get_models_dir();self.query_one('#strata-directory',Input).value=str(root)
        models=load_models();control=self.query_one('#strata-model',SearchableSelect)
        previous=control.value;control.set_options([(m.name,m.id) for m in models]);control.value=previous if previous in {m.id for m in models} else models[0].id
        self.query_one('#strata-ready',Static).update('Ready' if prepared(get_model(control.value),root) else 'Download and prepare the selected model in Models.')

    @on(Button.Pressed,'#strata-start')
    def start(self):
        try:
            image_id=self.query_one('#strata-image',SearchableSelect).value
            toolbox=self.app.toolbox_catalog.toolboxes.get(image_id)
            if not toolbox or toolbox.backend!='strata' or image_id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:raise ValueError('Select the Strata B70 toolbox')
            values={key:self.query_one(f'#strata-{key}',Input).value.strip() for key in ('directory','host','port','context')}
            if not values['directory']:raise ValueError('Enter a models directory')
            choices={key:self.query_one(f'#strata-{key}',SearchableSelect).value for key in ('engine','model','speculation')}
            self._command=build_server_cmd(engine=choices['engine'],image=toolbox.image,
                engine_args=self.app.toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args,
                platform_id=self.platform_id,model=get_model(choices['model']),directory=Path(values['directory']).expanduser(),
                host=values['host'],port=int(values['port']),context_size=int(values['context']),speculation=choices['speculation'])
            self._settings={**choices,**values,'models_dir':str(Path(values['directory']).expanduser().resolve())}
        except (ValueError,OSError) as e:self.notify(str(e),severity='error');return
        self.app.push_screen(ConfirmModal('Start Strata on both B70s?\n\n'+shlex.join(self._command),yes_text='Start',copy_text=shlex.join(self._command)),self._confirmed)

    def _confirmed(self,confirmed):
        if not confirmed:return
        save_backend_settings('strata',self._settings)
        with self.app.suspend():run_foreground_server(self._command,self._command[0],CONTAINER_NAME)
