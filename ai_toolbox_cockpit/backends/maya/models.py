"""Maya pinned model downloads and native-pack preparation controls."""
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
from .model_manager import get_model, get_models_dir, save_models_dir, incomplete_files, prepared, download_command, preparation_command, record_prepared


class MayaModelPanel(BackendModelPanel):
    DEFAULT_CSS = 'MayaModelPanel .inline-label { width: 23 !important; margin-right: 1; }'
    backend_label='Maya Models'

    def compose(self)->ComposeResult:
        with VerticalScroll():
            yield Static('Select the native Maya model for this platform, download its pinned artifacts, then prepare its engine pack.', id='maya-download-description', classes='panel-copy')
            for key,label in [('model','Model / quantization'),('engine','Container engine'),('image','Preparation image')]:
                with Horizontal(classes='inline-row'):
                    yield Label(label,id=f'maya-download-{key}-label',classes='inline-label')
                    yield SearchableSelect(f'Select {label.lower()}',id=f'maya-download-{key}')
            with Horizontal(classes='inline-row'):
                yield Label('Models directory',id='maya-models-dir-label',classes='inline-label')
                yield Input(value=str(get_models_dir()),id='maya-models-dir')
                yield Button('Save Path',id='maya-save-path')
            yield Static('',id='maya-model-state',classes='panel-copy')
            yield Static('',id='maya-disk-space',classes='storage-copy')
            with Horizontal(classes='inline-row'):
                yield Button('Download / Repair',id='maya-download',variant='success')
                yield Button('Prepare for Maya',id='maya-prepare',variant='primary')
                yield Button('Scan Local',id='maya-scan')

    def on_mount(self):
        control=self.query_one('#maya-download-model',SearchableSelect)
        control.set_options([(e['name'],e['id']) for e in self.catalog.entries])
        control.value=self.catalog.entries[0]['id'] if self.catalog.entries else ''
        engines=[(e.value,e.value) for e in detect_container_engines()]
        control=self.query_one('#maya-download-engine',SearchableSelect);control.set_options(engines);control.value=engines[0][1] if engines else ''
        self.set_platform(self.app.active_platform_id)
        self.refresh_inventory()

    def set_platform(self,platform_id):
        self.platform_id = platform_id
        if not self.is_mounted:return
        models=[e for e in self.catalog.entries if e.get('platform_id', 'strix-halo') == platform_id]
        control=self.query_one('#maya-download-model',SearchableSelect)
        previous=control.value;control.set_options([(e['name'],e['id']) for e in models]);control.value=previous if previous in {e['id'] for e in models} else (models[0]['id'] if models else '')
        items=[t for t in self.app.toolbox_catalog.platform_toolboxes(platform_id) if t.backend=='maya']
        control=self.query_one('#maya-download-image',SearchableSelect);control.set_options([(t.name,t.id) for t in items]);control.value=items[0].id if items else ''

    @on(SearchableSelect.Changed, '#maya-download-model')
    def model_changed(self):
        self.refresh_inventory()

    def refresh_inventory(self):
        if not self.is_mounted:return
        try:
            model=get_model(self.query_one('#maya-download-model',SearchableSelect).value)
            root=Path(self.query_one('#maya-models-dir',Input).value).expanduser()
            missing=incomplete_files(model,root)
            state=f'{len(missing)} missing/incomplete files' if missing else ('Ready for Maya' if prepared(model,root) else 'Files downloaded; preparation required')
            description = 'GLM-S24 includes MTP. DeepSeek-S includes the separate DSpark files and uses SSD-backed Engram tables. Download the pinned bundle, then prepare it for Maya.'
            self.query_one('#maya-download-description',Static).update(description)
            self.query_one('#maya-prepare',Button).disabled=prepared(model,root)
            self.query_one('#maya-model-state',Static).update(f'{state}. Pinned artifacts: {model.size_bytes/1024**3:.1f} GiB. Allow additional space for preparation.')
            self.query_one('#maya-disk-space',Static).update(disk_space_text(root))
        except (ValueError,OSError):pass

    @on(Button.Pressed,'#maya-save-path')
    def save_path(self):
        if not save_models_dir(self.query_one('#maya-models-dir',Input).value):
            self.notify('Could not save models directory.',severity='error');return
        self.refresh_inventory();self.app.refresh_server_model_inventory('maya')

    @on(Button.Pressed,'#maya-scan')
    def scan(self):
        self.refresh_inventory();self.app.refresh_server_model_inventory('maya')

    @on(Button.Pressed,'#maya-download')
    def download(self):self._confirm(False)

    @on(Button.Pressed,'#maya-prepare')
    def prepare(self):self._confirm(True)

    def _confirm(self,preparation):
        try:
            value=self.query_one('#maya-models-dir',Input).value.strip()
            if not value:raise ValueError('Enter a models directory')
            root=Path(value).expanduser().resolve()
            model=get_model(self.query_one('#maya-download-model',SearchableSelect).value)
            if preparation:
                toolbox=self.app.toolbox_catalog.toolboxes.get(self.query_one('#maya-download-image',SearchableSelect).value)
                if not toolbox or toolbox.backend!='maya' or toolbox.id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:raise ValueError('Select the Maya preparation image')
                command=preparation_command(model,root,self.query_one('#maya-download-engine',SearchableSelect).value,toolbox.image)
            else:command=download_command(model,root)
            self._pending=(command,root,model,preparation)
        except (ValueError,OSError) as e:self.notify(str(e),severity='error');return
        label='Prepare' if preparation else 'Download'
        summary = shlex.join(command) if preparation else f'Repository: {model.repo}\nRevision: {model.revision}\nFiles: {len(model.files)} model files ({model.size_bytes/1024**3:.1f} GiB)\nDestination: {root / model.directory}\n\nCopy command shows the exact shard list.'
        self.app.push_screen(ConfirmModal(f'{label} {model.name}?\n\n'+summary,yes_text=label,copy_text=shlex.join(command)),self._confirmed)

    def _confirmed(self,confirmed):
        if not confirmed:return
        command,root,model,preparation=self._pending
        if not save_models_dir(str(root)):self.notify('Could not create models directory',severity='error');return
        with self.app.suspend():
            try:
                subprocess.run(command,env=huggingface_environment(get_hf_token()),check=True)
                if preparation:record_prepared(model,root)
            except (OSError,ValueError,subprocess.SubprocessError) as e:pause_after_failure(f'Maya operation failed: {e}')
            except KeyboardInterrupt:pass
        self.scan()
