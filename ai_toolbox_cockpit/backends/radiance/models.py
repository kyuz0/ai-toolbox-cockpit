"""Radiance model/precision bundle download panel."""

import shlex
import subprocess
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import get_hf_token, huggingface_environment, save_hf_token
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.storage import disk_space_for_path, disk_space_text, download_space_note
from ai_toolbox_cockpit.widgets import ConfirmModal, HfTokenModal, SearchableSelect

from .model_manager import (get_model, get_models_dir, download_command, resolved_path,
                            load_models, save_models_dir, RadianceModel)


def bundle_size(entry):
    return entry['size_bytes']


def get_bundle(model_id):
    return get_model(model_id)


def incomplete_files(model, directory):
    if isinstance(model, dict):
        model = RadianceModel.from_entry(model)
    return [] if resolved_path(model, directory) else [{'size_bytes': model.size_bytes}]


def get_download_cmd(model, directory):
    return download_command(model, directory)



class RadianceModelPanel(BackendModelPanel):
    backend_label = "Radiance Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._hf_token = get_hf_token()
        self._hf_token_prompted = False
        self._pending_bundle: RadianceModel | None = None
        self._pending_directory = Path()

    def compose(self) -> ComposeResult:
        yield Static("Radiance uses pinned .rad model containers with bundled vision and drafting tensors. "
                     "Select the matching native model; GGUF and vLLM checkpoints use other backends.",
                     classes="panel-copy")
        with Vertical(classes="model-zone"):
            yield Label("Curated Radiance bundles", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Model / precision", id="radiance-download-model-label", classes="inline-label")
                yield SearchableSelect("Select Radiance bundle", id="radiance-download-model")
                yield Button("Download / Repair", id="radiance-download", variant="success")
        with Vertical(classes="model-zone"):
            yield Label("Local Radiance files", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Directory", id="radiance-models-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="radiance-models-dir")
                yield Button("Save Path", id="radiance-save-models-dir")
                yield Button("Scan Local", id="radiance-models-scan", variant="primary")
            yield Static("", id="radiance-disk-space", classes="storage-copy")
            yield DataTable(id="radiance-local-models", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        select = self.query_one("#radiance-download-model", SearchableSelect)
        select.set_options([
            (f"{entry['name']} — {bundle_size(entry) / 1024**3:.1f} GiB", entry["id"])
            for entry in self.catalog.entries
        ])
        select.value = next(entry["id"] for entry in self.catalog.entries if entry.get("recommended"))
        self.query_one("#radiance-local-models", DataTable).add_columns("Bundle", "Local state", "Directory")
        self.refresh_inventory()

    def refresh_inventory(self) -> None:
        directory = get_models_dir()
        self.query_one("#radiance-models-dir", Input).value = str(directory)
        table = self.query_one("#radiance-local-models", DataTable)
        table.clear()
        for entry in self.catalog.entries:
            missing = incomplete_files(entry, directory)
            status = f"{len(missing)} missing / incomplete files" if missing else "Ready (file sizes checked)"
            table.add_row(entry["name"], status, str(directory), key=entry["id"])
        self.query_one("#radiance-disk-space", Static).update(disk_space_text(directory))

    def refresh_all_model_controls(self) -> None:
        self.refresh_inventory()
        self.app.refresh_server_model_inventory("radiance")

    @on(Button.Pressed, "#radiance-save-models-dir")
    def save_path_pressed(self) -> None:
        if save_models_dir(self.query_one("#radiance-models-dir", Input).value.strip()):
            self.refresh_all_model_controls()
            self.notify("Radiance model directory saved.")
        else:
            self.notify("Could not create or save that directory.", severity="error")

    @on(Button.Pressed, "#radiance-models-scan")
    def scan_pressed(self) -> None:
        self.refresh_all_model_controls()

    @on(Button.Pressed, "#radiance-download")
    def download_pressed(self) -> None:
        try:
            self._pending_bundle = get_bundle(self.query_one("#radiance-download-model", SearchableSelect).value)
            value = self.query_one("#radiance-models-dir", Input).value.strip()
            if not value:
                raise ValueError("Enter a model directory.")
            self._pending_directory = Path(value).expanduser().resolve()
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        self._hf_token = self._hf_token or get_hf_token()
        if not self._hf_token and not self._hf_token_prompted:
            self.app.push_screen(HfTokenModal(), self._hf_token_received)
            return
        self._confirm_download()

    def _hf_token_received(self, choice: tuple[str, bool] | None) -> None:
        if choice is None:
            return
        self._hf_token, remember = choice
        self._hf_token_prompted = True
        if self._hf_token and remember and not save_hf_token(self._hf_token):
            self.notify("Could not save the token; using it for this session.", severity="warning")
        self._confirm_download()

    def _confirm_download(self) -> None:
        bundle, directory = self._pending_bundle, self._pending_directory
        missing = incomplete_files(bundle, directory)
        space = disk_space_for_path(directory)
        note = download_space_note(sum(item["size_bytes"] for item in missing), space.free if space else None)
        command = get_download_cmd(bundle, directory)
        companion = "bundled vision and drafting tensors"
        self.app.push_screen(
            ConfirmModal(
                f"Download / repair {bundle.name} into {directory}?\n"
                f"Includes {companion}. Existing files are reused.\n\n"
                f"{note}\n\n{shlex.join(command)}", yes_text="Download",
                copy_text=shlex.join(command),
            ), self._download_confirmed,
        )

    def _download_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        bundle, directory = self._pending_bundle, self._pending_directory
        if not save_models_dir(str(directory)):
            self.notify("Could not create or save that directory.", severity="error")
            return
        failed = False
        with self.app.suspend():
            try:
                subprocess.run(get_download_cmd(bundle, directory),
                               env=huggingface_environment(self._hf_token), check=True)
            except KeyboardInterrupt:
                failed = True
            except (OSError, subprocess.SubprocessError) as error:
                failed = True
                pause_after_failure(f"Radiance download failed: {error}")
        self.refresh_all_model_controls()
        if failed:
            self.notify("Download interrupted or failed; Download / Repair resumes it.", severity="warning")
        elif incomplete_files(bundle, directory):
            self.notify("Download finished but required files are missing or incomplete.", severity="error")
        else:
            self.notify("Radiance bundle download complete.")
