"""llama.cpp GGUF catalogue, local inventory, and download workflow."""

import shlex
import subprocess

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect, SelectModal

from .model_manager import (
    get_download_cmd,
    get_hf_quants,
    get_models_dir,
    is_quant_downloaded,
    save_models_dir,
    scan_local_models,
)


def get_download_sources(entries: tuple[dict, ...] | list[dict]) -> list[dict]:
    """Flatten primary GGUF repositories and their auxiliary download sources."""
    sources: list[dict] = []
    for entry in entries:
        sources.append({
            "name": entry.get("name", entry["id"]),
            "repo": entry["repo"],
            "description": "Primary model GGUFs.",
            "role": "model",
        })
        for download in entry.get("auxiliary_downloads", []):
            sources.append({
                "name": f"{entry.get('name', entry['id'])} — {download['name']}",
                **download,
            })
    return sources


class LlamaCppModelPanel(BackendModelPanel):
    backend_label = "llama.cpp Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._download_repo = ""
        self._download_quants: list[str] = []
        self._download_sources: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "Search curated Hugging Face GGUF repositories, download a quantization, and inventory local GGUF files and shards.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Curated Hugging Face downloader", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Repository", id="llama-download-repo-label", classes="inline-label")
                yield SearchableSelect("Search curated GGUF repositories", id="llama-download-repo")
                yield Button("Choose GGUF", id="llama-download", variant="success")
            yield Static("", id="llama-download-note")
        with Vertical(classes="model-zone"):
            yield Label("Local GGUF directory", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Directory", id="llama-models-dir-label", classes="inline-label")
                yield Input(value=str(get_models_dir()), id="llama-models-dir")
                yield Button("Save Path", id="llama-save-models-dir")
                yield Button("Scan Local", id="llama-models-scan", variant="primary")
            yield DataTable(id="llama-local-models", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        curated = self.query_one("#llama-download-repo", SearchableSelect)
        sources = get_download_sources(self.catalog.entries)
        self._download_sources = {source["repo"]: source for source in sources}
        curated.set_options([
            (f"{source['name']} — {source['repo']}", source["repo"])
            for source in sources
        ])
        table = self.query_one("#llama-local-models", DataTable)
        table.add_columns("Model / shard pattern", "Path")
        self.refresh_local_models()

    @on(SearchableSelect.Changed, "#llama-download-repo")
    def download_repo_changed(self, event: SearchableSelect.Changed) -> None:
        source = self._download_sources.get(str(event.value or ""), {})
        note = source.get("description", "")
        recommended = source.get("recommended_filename", "")
        if recommended:
            note = f"{note} Recommended file: {recommended}"
        self.query_one("#llama-download-note", Static).update(note)

    def refresh_local_models(self) -> None:
        table = self.query_one("#llama-local-models", DataTable)
        table.clear()
        for model in scan_local_models():
            table.add_row(model["name"], model["path"], key=model["path"])

    def refresh_all_model_controls(self) -> None:
        self.refresh_local_models()
        self.app.refresh_server_model_inventory("llama_cpp")

    def refresh_inventory(self) -> None:
        self.refresh_local_models()

    @on(Button.Pressed, "#llama-save-models-dir")
    def save_path_pressed(self) -> None:
        path = self.query_one("#llama-models-dir", Input).value.strip()
        if not path:
            self.notify("Enter a model directory.", severity="error")
            return
        if save_models_dir(path):
            self.refresh_all_model_controls()
            self.notify("llama.cpp model directory saved.")
        else:
            self.notify("Could not save or create that directory.", severity="error")

    @on(Button.Pressed, "#llama-models-scan")
    def scan_pressed(self) -> None:
        self.refresh_all_model_controls()
        self.notify("Local GGUF directory scanned.")

    @on(Button.Pressed, "#llama-download")
    def download_pressed(self) -> None:
        repo = self.query_one("#llama-download-repo", SearchableSelect).value
        if not repo:
            self.notify("Select a curated repository first.", severity="warning")
            return
        self.notify("Reading repository files from Hugging Face…")
        self.load_quants(repo)

    @work(thread=True, exclusive=True, group="llama-hf-quants")
    def load_quants(self, repo: str) -> None:
        quants = get_hf_quants(repo)
        self.app.call_from_thread(self._show_quants, repo, quants)

    def _show_quants(self, repo: str, quants: list[str]) -> None:
        if not quants:
            self.notify("No GGUF files were found or Hugging Face could not be reached.", severity="error")
            return
        self._download_repo = repo
        self._download_quants = quants
        options = [
            f"{'✓ Installed  ' if is_quant_downloaded(repo, quant) else ''}{quant}"
            for quant in quants
        ]
        source = self._download_sources.get(repo, {})
        title = (
            "Select auxiliary GGUF"
            if source.get("role") != "model"
            else "Select GGUF quantization"
        )
        self.app.push_screen(SelectModal(title, options), self._quant_selected)

    def _quant_selected(self, index: int | None) -> None:
        if index is None or not 0 <= index < len(self._download_quants):
            return
        quant = self._download_quants[index]
        installed = is_quant_downloaded(self._download_repo, quant)
        command = get_download_cmd(self._download_repo, quant)
        prompt = (
            f"{quant} appears to be installed. Download it again?"
            if installed
            else f"Download {self._download_repo} / {quant}?"
        )
        self.app.push_screen(
            ConfirmModal(f"{prompt}\n\n{shlex.join(command)}", yes_text="Download Again" if installed else "Download"),
            lambda confirmed: self._download_quant(quant) if confirmed else None,
        )

    def _download_quant(self, quant: str) -> None:
        command = get_download_cmd(self._download_repo, quant)
        try:
            with self.app.suspend():
                print(f"Downloading {self._download_repo} / {quant} with Hugging Face…")
                subprocess.run(command, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            self.notify(f"Model download failed: {error}", severity="error", timeout=8)
        else:
            self.refresh_all_model_controls()
            self.notify("Model download complete.", timeout=5)
