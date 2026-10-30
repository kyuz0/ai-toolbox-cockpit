"""Strata shard download and native-pack preparation controls."""
import shlex
import subprocess
from pathlib import Path
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, Static
from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import get_hf_token, huggingface_environment
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.storage import disk_space_text
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect
from .model_manager import get_model, get_models_dir, save_models_dir, incomplete_files, prepared, download_command, preparation_command


class StrataModelPanel(BackendModelPanel):
    DEFAULT_CSS = 'StrataModelPanel .inline-label { width: 23; margin-right: 1; }'
    backend_label='Strata Models'

    def compose(self)->ComposeResult:
        with VerticalScroll():
            yield Static('Qwen3.8 Flash Next on two 32 GB B70s. Download the AtomicChat AD4.27 shards, then prepare the native pack and official MTP head. Preparation also downloads about 5.2 GB of MTP tensors.',classes='panel-copy')
            for key,label in [('model','Model / quantization'),('engine','Container engine'),('image','Preparation image')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label,id=f'strata-download-{key}-label',classes='inline-label')
                    yield SearchableSelect(f'Select {label.lower()}',id=f'strata-download-{key}')
            with Horizontal(classes='inline-row'):
                yield Label('Models directory',id='strata-models-dir-label',classes='inline-label')
                yield Input(value=str(get_models_dir()),id='strata-models-dir')
                yield Button('Save Path',id='strata-save-path')
            yield Static('',id='strata-model-state',classes='panel-copy')
            yield Static('',id='strata-disk-space',classes='storage-copy')
            with Horizontal(classes='inline-row'):
                yield Button('Download / Repair',id='strata-download',variant='success')
                yield Button('Prepare for Strata',id='strata-prepare',variant='primary')
                yield Button('Scan Local',id='strata-scan')

    def on_mount(self):
        control=self.query_one('#strata-download-model',SearchableSelect)
        control.set_options([(e['name'],e['id']) for e in self.catalog.entries])
        control.value=self.catalog.entries[0]['id'] if self.catalog.entries else ''
        engines=[(e.value,e.value) for e in detect_container_engines()]
        control=self.query_one('#strata-download-engine',SearchableSelect);control.set_options(engines);control.value=engines[0][1] if engines else ''
        self.set_platform(self.app.active_platform_id)
        self.refresh_inventory()

    def set_platform(self,platform_id):
        if not self.is_mounted:return
        items=[t for t in self.app.toolbox_catalog.platform_toolboxes(platform_id) if t.backend=='strata']
        control=self.query_one('#strata-download-image',SearchableSelect);control.set_options([(t.name,t.id) for t in items]);control.value=items[0].id if items else ''

    def refresh_inventory(self):
        if not self.is_mounted:return
        try:
            model=get_model(self.query_one('#strata-download-model',SearchableSelect).value)
            root=Path(self.query_one('#strata-models-dir',Input).value).expanduser()
            missing=incomplete_files(model,root)
            state=f'{len(missing)} missing/incomplete shards' if missing else ('Ready for Strata' if prepared(model,root) else 'Shards downloaded; preparation required')
            self.query_one('#strata-model-state',Static).update(f'{state}. Main download: {model.size_bytes/1024**3:.1f} GiB. Allow additional disk space for the native pack and MTP files.')
            self.query_one('#strata-disk-space',Static).update(disk_space_text(root))
        except (ValueError,OSError):pass

    @on(Button.Pressed,'#strata-save-path')
    def save_path(self):
        if not save_models_dir(self.query_one('#strata-models-dir',Input).value):
            self.notify('Could not save models directory.',severity='error');return
        self.refresh_inventory();self.app.refresh_server_model_inventory('strata')

    @on(Button.Pressed,'#strata-scan')
    def scan(self):
        self.refresh_inventory();self.app.refresh_server_model_inventory('strata')

    @on(Button.Pressed,'#strata-download')
    def download(self):self._confirm(False)

    @on(Button.Pressed,'#strata-prepare')
    def prepare(self):self._confirm(True)

    def _confirm(self,preparation):
        try:
            value=self.query_one('#strata-models-dir',Input).value.strip()
            if not value:raise ValueError('Enter a models directory')
            root=Path(value).expanduser().resolve()
            model=get_model(self.query_one('#strata-download-model',SearchableSelect).value)
            if preparation:
                toolbox=self.app.toolbox_catalog.toolboxes.get(self.query_one('#strata-download-image',SearchableSelect).value)
                if not toolbox or toolbox.backend!='strata':raise ValueError('Select the Strata preparation image')
                command=preparation_command(model,root,self.query_one('#strata-download-engine',SearchableSelect).value,toolbox.image)
            else:command=download_command(model,root)
            self._pending=(command,root)
        except (ValueError,OSError) as e:self.notify(str(e),severity='error');return
        label='Prepare' if preparation else 'Download'
        summary = shlex.join(command) if preparation else f'Repository: {model.repo}\nRevision: {model.revision}\nFiles: {len(model.files)} GGUF shards ({model.size_bytes/1024**3:.1f} GiB)\nDestination: {root / model.directory}\n\nCopy command shows the exact shard list.'
        self.app.push_screen(ConfirmModal(f'{label} {model.name}?\n\n'+summary,yes_text=label,copy_text=shlex.join(command)),self._confirmed)

    def _confirmed(self,confirmed):
        if not confirmed:return
        command,root=self._pending
        if not save_models_dir(str(root)):self.notify('Could not create models directory',severity='error');return
        with self.app.suspend():
            try:subprocess.run(command,env=huggingface_environment(get_hf_token()),check=True)
            except (OSError,subprocess.SubprocessError) as e:pause_after_failure(f'Strata operation failed: {e}')
            except KeyboardInterrupt:pass
        self.scan()
