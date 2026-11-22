"""Halogen model/precision bundle download panel."""

import shlex
import subprocess
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import HubTokenChoice, huggingface_environment
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.storage import disk_space_for_path, disk_space_text, download_space_note
from ai_toolbox_cockpit.widgets import ConfirmModal, HfTokenModal, SearchableSelect

from .model_manager import (
    bundle_size, fast_draft_download, get_bundle, get_download_commands, get_models_dir, incomplete_files, save_models_dir,
)


class HalogenModelPanel(BackendModelPanel):
    backend_label = "Halogen Flash Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._hf_token: HubTokenChoice = ""
        self._pending_bundle: dict = {}
        self._pending_directory = Path()

    def compose(self) -> ComposeResult:
        yield Static(
            "Qwen3.8-Flash-Next v2 uses an HGN checkpoint, a separate n-gram table, and tokenizer "
            "files (about 109.8 GiB; server 0.15 or newer). v2 is the default; "
            "W4B bundles use a precision overlay (about 118 GiB total). Vision bundles add a "
            "0.84 GiB sidecar and enable image input when selected in Server Mode. Shared files are reused.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Curated Halogen bundles", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Model / precision", id="halogen-download-model-label", classes="inline-label")
                yield SearchableSelect("Select Halogen bundle", id="halogen-download-model")
                yield Button("Download / Repair", id="halogen-download", variant="success")
            yield Button("Download faster MTP head (1.39 GiB)", id="halogen-download-fast-head")
        with Vertical(classes="model-zone"):
            yield Label("Local Halogen files", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Directory", id="halogen-models-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="halogen-models-dir")
                yield Button("Save Path", id="halogen-save-models-dir")
                yield Button("Scan Local", id="halogen-models-scan", variant="primary")
            yield Static("", id="halogen-disk-space", classes="storage-copy")
            yield DataTable(id="halogen-local-models", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        select = self.query_one("#halogen-download-model", SearchableSelect)
        select.set_options([
            (f"{entry['name']} — {bundle_size(entry) / 1024**3:.1f} GiB", entry["id"])
            for entry in self.catalog.entries
        ])
        select.value = next(entry["id"] for entry in self.catalog.entries if entry.get("recommended"))
        self.query_one("#halogen-local-models", DataTable).add_columns("Bundle", "Local state", "Directory")
        self.refresh_inventory()

    def refresh_inventory(self) -> None:
        directory = get_models_dir()
        self.query_one("#halogen-models-dir", Input).value = str(directory)
        table = self.query_one("#halogen-local-models", DataTable)
        table.clear()
        for entry in self.catalog.entries:
            missing = incomplete_files(entry, directory)
            status = f"{len(missing)} missing / incomplete files" if missing else "Ready (file sizes checked)"
            table.add_row(entry["name"], status, str(directory), key=entry["id"])
        self.query_one("#halogen-disk-space", Static).update(disk_space_text(directory))

    def refresh_all_model_controls(self) -> None:
        self.refresh_inventory()
        self.app.refresh_server_model_inventory("halogen")

    @on(Button.Pressed, "#halogen-save-models-dir")
    def save_path_pressed(self) -> None:
        if save_models_dir(self.query_one("#halogen-models-dir", Input).value.strip()):
            self.refresh_all_model_controls()
            self.notify("Halogen model directory saved.")
        else:
            self.notify("Could not create or save that directory.", severity="error")

    @on(Button.Pressed, "#halogen-models-scan")
    def scan_pressed(self) -> None:
        self.refresh_all_model_controls()

    @on(SearchableSelect.Changed, "#halogen-download-model")
    def update_fast_head_visibility(self, event: SearchableSelect.Changed) -> None:
        self.query_one("#halogen-download-fast-head").display = bool(
            event.value and get_bundle(event.value).get("fast_draft_head")
        )

    @on(Button.Pressed, "#halogen-download-fast-head")
    def download_fast_head_pressed(self) -> None:
        self._prepare_download(fast_head=True)

    @on(Button.Pressed, "#halogen-download")
    def download_pressed(self) -> None:
        self._prepare_download(fast_head=False)

    def _prepare_download(self, *, fast_head: bool) -> None:
        try:
            bundle = get_bundle(self.query_one("#halogen-download-model", SearchableSelect).value)
            if fast_head and not bundle.get("fast_draft_head"):
                raise ValueError("Select a v2 bundle to download the faster MTP head.")
            self._pending_bundle = fast_draft_download(bundle) if fast_head else bundle
            value = self.query_one("#halogen-models-dir", Input).value.strip()
            if not value:
                raise ValueError("Enter a model directory.")
            self._pending_directory = Path(value).expanduser().resolve()
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        self.app.push_screen(HfTokenModal(), self._hf_token_received)

    def _hf_token_received(self, choice: HubTokenChoice | None) -> None:
        if choice is None:
            return
        self._hf_token = choice
        self._confirm_download()

    def _confirm_download(self) -> None:
        bundle, directory = self._pending_bundle, self._pending_directory
        missing = incomplete_files(bundle, directory)
        space = disk_space_for_path(directory)
        note = download_space_note(sum(item["size_bytes"] for item in missing), space.free if space else None)
        commands = get_download_commands(bundle, directory)
        if not commands:
            self.notify("The selected Halogen download is already complete.")
            return
        preview = "\n".join(shlex.join(command) for command in commands)
        self.app.push_screen(
            ConfirmModal(
                f"Download / repair {bundle['name']} into {directory}?\n"
                "Existing complete files are reused.\n\n"
                f"{note}\n\n{preview}", yes_text="Download",
                copy_text=preview,
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
                env = huggingface_environment(self._hf_token)
                for command in get_download_commands(bundle, directory):
                    subprocess.run(command, env=env, check=True)
            except KeyboardInterrupt:
                failed = True
            except (OSError, subprocess.SubprocessError) as error:
                failed = True
                pause_after_failure(f"Halogen download failed: {error}")
        self.refresh_all_model_controls()
        if failed:
            self.notify("Download interrupted or failed; Download / Repair resumes it.", severity="warning")
        elif incomplete_files(bundle, directory):
            self.notify("Download finished but some files are missing or incomplete.", severity="error")
        else:
            self.notify(f"{bundle['name']} download complete.")
