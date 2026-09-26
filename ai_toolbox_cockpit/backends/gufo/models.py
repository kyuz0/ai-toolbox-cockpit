"""Gufo model bundle download and inventory panel."""

import shlex
import subprocess
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import (
    get_hf_token,
    huggingface_environment,
    save_hf_token,
)
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.storage import disk_space_for_path, disk_space_text, download_space_note
from ai_toolbox_cockpit.widgets import ConfirmModal, HfTokenModal, SearchableSelect

from .model_manager import (
    get_download_commands,
    get_model,
    get_models_dir,
    missing_files,
    model_size,
    model_status,
    save_models_dir,
    search_roots,
)


class GufoModelPanel(BackendModelPanel):
    backend_label = "Gufo Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._hf_token = get_hf_token()
        self._hf_token_prompted = False
        self._pending_model: dict = {}
        self._pending_directory = Path()

    def compose(self) -> ComposeResult:
        yield Static(
            "Curated Gufo profiles use the exact GGUF targets and optional MTP or DSpark "
            "sidecars validated on Strix Halo. File-size checks reuse compatible files under "
            "the configured model directory and ~/ds4.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Curated Gufo profiles", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Model / quant", id="gufo-download-model-label", classes="inline-label")
                yield SearchableSelect("Select Gufo model profile", id="gufo-download-model")
                yield Button("Download / Repair", id="gufo-download", variant="success")
        with Vertical(classes="model-zone"):
            yield Label("Local Gufo files", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Download directory", id="gufo-models-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="gufo-models-dir")
                yield Button("Save Path", id="gufo-save-models-dir")
                yield Button("Scan Local", id="gufo-models-scan", variant="primary")
            yield Static("", id="gufo-search-roots", classes="storage-copy")
            yield Static("", id="gufo-disk-space", classes="storage-copy")
            yield DataTable(id="gufo-local-models", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        select = self.query_one("#gufo-download-model", SearchableSelect)
        select.set_options([
            (f"{entry['name']} — {model_size(entry) / 1024**3:.1f} GiB", entry["id"])
            for entry in self.catalog.entries
        ])
        select.value = next(entry["id"] for entry in self.catalog.entries if entry.get("recommended"))
        self.query_one("#gufo-local-models", DataTable).add_columns(
            "Model / quant", "Local state"
        )
        self.refresh_inventory()

    def refresh_inventory(self) -> None:
        directory = get_models_dir()
        self.query_one("#gufo-models-dir", Input).value = str(directory)
        roots = ", ".join(str(path) for path in search_roots())
        self.query_one("#gufo-search-roots", Static).update(f"Search paths: {roots}")
        self.query_one("#gufo-disk-space", Static).update(disk_space_text(directory))
        table = self.query_one("#gufo-local-models", DataTable)
        table.clear()
        for entry in self.catalog.entries:
            table.add_row(entry["name"], model_status(entry), key=entry["id"])

    def refresh_all_model_controls(self) -> None:
        self.refresh_inventory()
        self.app.refresh_server_model_inventory("gufo")

    @on(Button.Pressed, "#gufo-save-models-dir")
    def save_path_pressed(self) -> None:
        if save_models_dir(self.query_one("#gufo-models-dir", Input).value.strip()):
            self.refresh_all_model_controls()
            self.notify("Gufo model directory saved.")
        else:
            self.notify("Could not create or save that directory.", severity="error")

    @on(Button.Pressed, "#gufo-models-scan")
    def scan_pressed(self) -> None:
        self.refresh_all_model_controls()

    @on(Button.Pressed, "#gufo-download")
    def download_pressed(self) -> None:
        try:
            self._pending_model = get_model(
                self.query_one("#gufo-download-model", SearchableSelect).value
            )
            value = self.query_one("#gufo-models-dir", Input).value.strip()
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
        model, directory = self._pending_model, self._pending_directory
        missing = missing_files(model)
        required = sum(
            item["size_bytes"] for item in model["files"] if item["path"] in missing
        )
        speculation = model.get("speculation")
        if speculation and speculation["path"] in missing:
            required += speculation["size_bytes"]
        space = disk_space_for_path(directory)
        note = download_space_note(required, space.free if space else None)
        commands = get_download_commands(model, directory)
        rendered = "\n".join(shlex.join(command) for command in commands)
        self.app.push_screen(
            ConfirmModal(
                f"Download / repair {model['name']} into {directory / model['directory']}?\n"
                "Pinned target files and the tested speculative sidecar are downloaded together. "
                "Existing complete files are reused.\n\n"
                f"{note}\n\n{rendered}",
                yes_text="Download",
            ),
            self._download_confirmed,
        )

    def _download_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        model, directory = self._pending_model, self._pending_directory
        if not save_models_dir(str(directory)):
            self.notify("Could not create or save that directory.", severity="error")
            return
        failed = False
        with self.app.suspend():
            try:
                for command in get_download_commands(model, directory):
                    subprocess.run(
                        command,
                        env=huggingface_environment(self._hf_token),
                        check=True,
                    )
            except KeyboardInterrupt:
                failed = True
            except (OSError, subprocess.SubprocessError) as error:
                failed = True
                pause_after_failure(f"Gufo model download failed: {error}")
        self.refresh_all_model_controls()
        if failed:
            self.notify(
                "Download interrupted or failed; Download / Repair resumes it.",
                severity="warning",
            )
        elif missing_files(model):
            self.notify("Download finished but required files are missing or incomplete.", severity="error")
        else:
            self.notify("Gufo model bundle download complete.")
