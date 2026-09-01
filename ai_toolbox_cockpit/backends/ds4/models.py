"""DS4 exact-file model manager."""

import os
import shlex
import subprocess

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect

from .model_manager import (
    get_download_cmd,
    get_models_dir,
    is_model_downloaded,
    save_models_dir,
    scan_local_models,
)


class Ds4ModelPanel(BackendModelPanel):
    backend_label = "DwarfStar (ds4) Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._pending_repo = ""
        self._pending_filename = ""

    def compose(self) -> ComposeResult:
        yield Static(
            "DwarfStar (ds4) consumes exact curated GGUF artifacts. Downloads use the repository and filename declared in the catalog.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Curated DwarfStar (ds4) artifacts", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Artifact", id="ds4-download-model-label", classes="inline-label")
                yield SearchableSelect("Search DwarfStar (ds4) model files", id="ds4-download-model")
                yield Button("Download", id="ds4-download", variant="success")
        with Vertical(classes="model-zone"):
            yield Label("Local DwarfStar (ds4) directory", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Directory", id="ds4-models-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="ds4-models-dir")
                yield Button("Save Path", id="ds4-save-models-dir")
                yield Button("Scan Local", id="ds4-models-scan", variant="primary")
            yield DataTable(id="ds4-local-models", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        select = self.query_one("#ds4-download-model", SearchableSelect)
        entries = [entry for entry in self.catalog.entries if entry.get("filename")]
        select.set_options([
            (
                f"{entry.get('name', entry['filename'])} — {entry.get('size_gb', '?')} GB",
                f"{entry.get('repo', self.catalog.config.get('default_repo', ''))}::{entry['filename']}",
            )
            for entry in entries
        ])
        recommended = next((entry for entry in entries if entry.get("recommended")), None)
        if recommended:
            select.value = (
                f"{recommended.get('repo', self.catalog.config.get('default_repo', ''))}"
                f"::{recommended['filename']}"
            )
        table = self.query_one("#ds4-local-models", DataTable)
        table.add_columns("Filename", "Path")
        self.refresh_local_models()

    def refresh_local_models(self) -> None:
        table = self.query_one("#ds4-local-models", DataTable)
        table.clear()
        for model in scan_local_models():
            table.add_row(model["name"], model["path"], key=model["path"])

    def refresh_all_model_controls(self) -> None:
        self.refresh_local_models()
        self.app.refresh_server_model_inventory("ds4")

    def refresh_inventory(self) -> None:
        self.refresh_local_models()

    @on(Button.Pressed, "#ds4-save-models-dir")
    def save_path_pressed(self) -> None:
        value = self.query_one("#ds4-models-dir", Input).value.strip()
        if value and save_models_dir(value):
            self.refresh_all_model_controls()
            self.notify("DwarfStar (ds4) model directory saved.")
        else:
            self.notify("Could not create or save that directory.", severity="error")

    @on(Button.Pressed, "#ds4-models-scan")
    def scan_pressed(self) -> None:
        self.refresh_all_model_controls()

    @on(Button.Pressed, "#ds4-download")
    def download_pressed(self) -> None:
        value = self.query_one("#ds4-download-model", SearchableSelect).value
        if "::" not in value:
            self.notify("Select a curated DwarfStar (ds4) artifact.", severity="warning")
            return
        repo, filename = value.split("::", 1)
        self._pending_repo, self._pending_filename = repo, filename
        installed = is_model_downloaded(filename)
        prompt = (
            f"{filename} appears to be installed. Download it again?"
            if installed
            else f"Download {repo} / {filename}?"
        )
        self.app.push_screen(
            ConfirmModal(
                f"{prompt}\n\n{shlex.join(get_download_cmd(repo, filename))}",
                yes_text="Download Again" if installed else "Download",
            ),
            self._download_confirmed,
        )

    def _download_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self._download_model()

    def _download_model(self) -> None:
        command = get_download_cmd(self._pending_repo, self._pending_filename)
        environment = os.environ.copy()
        environment["HF_XET_HIGH_PERFORMANCE"] = "1"
        try:
            with self.app.suspend():
                print(f"Downloading {self._pending_repo} / {self._pending_filename}…")
                subprocess.run(command, env=environment, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            self.notify(f"DwarfStar (ds4) model download failed: {error}", severity="error", timeout=8)
        else:
            self.refresh_all_model_controls()
            self.notify("DwarfStar (ds4) model download complete.", timeout=5)
