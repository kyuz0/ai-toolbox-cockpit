"""vLLM Hugging Face catalogue, cache inventory, and artifact downloader and explorer."""

from pathlib import Path
import json
import shlex
import subprocess

from huggingface_hub import HfApi
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import get_hf_token, huggingface_environment
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.storage import disk_space_for_path, download_space_note
from ai_toolbox_cockpit.widgets import ConfirmModal, SearchableSelect
from .model_manager import build_prepare_cmd, checkpoint_ready, get_download_cmd, incomplete_files
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def cache_directory_for_repo(cache_root: Path, repo_id: str) -> Path:
    return cache_root / f"models--{repo_id.replace('/', '--')}"


class VllmModelPanel(BackendModelPanel):
    backend_label = "vLLM Models"

    def compose(self) -> ComposeResult:
        yield Static(
            "Download exact curated artifacts, prepare a separate MXFP4 checkpoint for GGZ14/Radiance, or explore Hugging Face repositories. Scans check every required file and checkpoint payload boundary.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Hugging Face cache", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Cache directory", id="vllm-model-cache-label", classes="inline-label")
                yield Input(id="vllm-model-cache")
                yield Button("Save Path", id="vllm-save-model-cache")
                yield Button("Refresh", id="vllm-refresh-cache", variant="primary")
            yield DataTable(id="vllm-curated-models", cursor_type="row", zebra_stripes=True)
        with Vertical(classes="model-zone"):
            yield Label("Curated artifact acquisition", classes="zone-title")
            for key, label in (("artifact", "Model artifact"), ("engine", "Container engine"), ("image", "Preparation toolbox")):
                with Horizontal(classes="inline-row"):
                    yield Label(label, id=f"vllm-download-{key}-label", classes="inline-label")
                    yield SearchableSelect(f"Select {label.lower()}", id=f"vllm-download-{key}")
            for key, label in (("directory", "Original snapshot directory"), ("prepared", "Prepared checkpoint directory")):
                with Horizontal(classes="inline-row"):
                    yield Label(label, id=f"vllm-download-{key}-label", classes="inline-label")
                    yield Input(id=f"vllm-download-{key}")
            yield Static("", id="vllm-download-details", classes="panel-copy")
            with Horizontal(classes="inline-row"):
                yield Button("Download / Repair", id="vllm-download", variant="success")
                yield Button("Prepare MXFP4", id="vllm-prepare", variant="primary")
        with Vertical(classes="model-zone"):
            yield Label("Hugging Face Hub explorer", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Label("Model search", id="vllm-hub-query-label", classes="inline-label")
                yield Input(placeholder="Search model IDs, e.g. Qwen3.6", id="vllm-hub-query")
                yield Button("Search Hub", id="vllm-hub-search")
            yield DataTable(id="vllm-hub-results", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        settings = get_backend_settings("vllm")
        self.query_one("#vllm-model-cache", Input).value = str(settings.get("hf_cache", "~/.cache/huggingface"))
        curated = self.query_one("#vllm-curated-models", DataTable)
        curated.add_columns("Model repository", "Cached", "TP", "Context", "Attention", "Eager")
        results = self.query_one("#vllm-hub-results", DataTable)
        results.add_columns("Repository", "Pipeline", "Downloads", "Private/Gated")
        entries = [entry for entry in self.catalog.entries if "download" in entry]
        artifact = self.query_one("#vllm-download-artifact", SearchableSelect)
        artifact.set_options([(entry["name"], entry["id"]) for entry in entries])
        artifact.value = entries[0]["id"] if entries else ""
        engines = [item.value for item in detect_container_engines()]
        engine = self.query_one("#vllm-download-engine", SearchableSelect)
        engine.set_options([(item, item) for item in engines])
        engine.value = engines[0] if engines else ""
        self.set_platform(self.app.active_platform_id)
        self.refresh_curated()

    def cache_root(self) -> Path:
        return Path(self.query_one("#vllm-model-cache", Input).value).expanduser()

    def refresh_curated(self) -> None:
        root = self.cache_root()
        table = self.query_one("#vllm-curated-models", DataTable)
        table.clear()
        for entry in self.catalog.entries:
            repo = str(entry.get("repo", ""))
            local_directory = entry.get("local_directory")
            if "download" in entry:
                paths = get_backend_settings("vllm").get("artifact_paths", {}).get(entry["id"], {})
                original = Path(paths.get("source", entry["download"]["directory"])).expanduser()
                local_directory = paths.get("prepared", local_directory) if entry.get("requires_local_model") else paths.get("source", local_directory)
                complete = not incomplete_files(entry, original)
                cached = complete and (not entry.get("requires_local_model") or bool(local_directory and checkpoint_ready(Path(local_directory).expanduser())))
            else:
                cached = bool(local_directory and checkpoint_ready(Path(local_directory).expanduser()))
            attention = entry.get("attention_backend")
            if attention is None:
                attention = entry.get("attention_backend_label", "model-specific")
            table.add_row(
                repo,
                "Yes" if cached else "No",
                ", ".join(str(value) for value in entry.get("valid_tp", [1])),
                str(entry.get("ctx", "auto")),
                str(attention or "TRITON_ATTN"),
                "Yes" if entry.get("enforce_eager") else "No",
                key=entry["id"],
            )

    def refresh_inventory(self) -> None:
        self.refresh_curated()

    @on(Button.Pressed, "#vllm-save-model-cache")
    def save_path_pressed(self) -> None:
        try:
            root = self.cache_root().resolve()
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.notify(f"Could not create Hugging Face cache: {error}", severity="error")
            return
        if save_backend_settings("vllm", {"hf_cache": str(root)}):
            self.refresh_curated()
            self.notify("Hugging Face cache path saved.")
        else:
            self.notify("Could not save Hugging Face cache path.", severity="error")

    @on(Button.Pressed, "#vllm-refresh-cache")
    def refresh_pressed(self) -> None:
        self.refresh_curated()

    @on(Button.Pressed, "#vllm-hub-search")
    def search_pressed(self) -> None:
        query = self.query_one("#vllm-hub-query", Input).value.strip()
        if not query:
            self.notify("Enter a Hugging Face search term.", severity="warning")
            return
        self.notify("Searching Hugging Face Hub…")
        self.search_hub(query)

    @work(thread=True, exclusive=True, group="vllm-hub-search")
    def search_hub(self, query: str) -> None:
        try:
            models = list(
                HfApi(token=get_hf_token() or None).list_models(
                    search=query,
                    sort="downloads",
                    direction=-1,
                    limit=50,
                )
            )
        except Exception as error:
            self.app.call_from_thread(self.notify, f"Hugging Face search failed: {error}", severity="error", timeout=8)
            return
        rows = [
            (
                str(model.id),
                str(model.pipeline_tag or ""),
                str(model.downloads or 0),
                "Private" if model.private else "Gated" if getattr(model, "gated", False) else "Open",
            )
            for model in models
        ]
        self.app.call_from_thread(self._apply_search_results, rows)

    def _apply_search_results(self, rows: list[tuple[str, str, str, str]]) -> None:
        table = self.query_one("#vllm-hub-results", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row, key=row[0])
        self.notify(f"Found {len(rows)} repositories. Copy a repository ID into the vLLM server's Custom HF repo field to use it.")

    def set_platform(self, platform_id: str) -> None:
        if not self.is_mounted:
            return
        entries = [item for item in self.app.toolbox_catalog.platform_toolboxes(platform_id)
                   if item.backend == "vllm" and item.backend_config.get("checkpoint_preparation") == "ggz14-mtp-fp8"]
        image = self.query_one("#vllm-download-image", SearchableSelect)
        image.set_options([(item.name, item.id) for item in entries])
        image.value = entries[0].id if entries else ""

    def selected_artifact(self) -> dict:
        selected = self.query_one("#vllm-download-artifact", SearchableSelect).value
        return next(entry for entry in self.catalog.entries if entry["id"] == selected)

    @on(SearchableSelect.Changed, "#vllm-download-artifact")
    def artifact_changed(self) -> None:
        if not self.is_mounted:
            return
        entry = self.selected_artifact()
        paths = get_backend_settings("vllm").get("artifact_paths", {}).get(entry["id"], {})
        self.query_one("#vllm-download-directory", Input).value = paths.get("source", entry["download"]["directory"])
        self.query_one("#vllm-download-prepared", Input).value = paths.get("prepared", entry.get("local_directory", "")) if entry.get("requires_local_model") else ""
        self.query_one("#vllm-prepare", Button).disabled = not entry.get("requires_local_model")
        size = sum(item["size_bytes"] for item in entry["download"]["files"])
        self.query_one("#vllm-download-details", Static).update(
            f"{entry['repo']} @ {entry['revision']} — {size / 1024**3:.2f} GiB. "
            + ("MXFP4 preparation requires about 20 GB additional disk space." if entry.get("requires_local_model") else ""))

    @on(Button.Pressed, "#vllm-download")
    def download_pressed(self) -> None:
        entry = self.selected_artifact()
        directory = Path(self.query_one("#vllm-download-directory", Input).value).expanduser().resolve()
        missing = incomplete_files(entry, directory)
        size = sum(item["size_bytes"] for item in entry["download"]["files"] if item["path"] in missing)
        space = disk_space_for_path(directory)
        command = get_download_cmd(entry, directory)
        self.app.push_screen(ConfirmModal(
            f"Download / repair {entry['repo']} @ {entry['revision']} into {directory}?\n"
            + download_space_note(size, space.free if space else None)
            + "\n" + shlex.join(command)),
            lambda confirmed: self._download(entry, directory, command) if confirmed else None)

    def _download(self, entry: dict, directory: Path, command: list[str]) -> None:
        paths = dict(get_backend_settings("vllm").get("artifact_paths", {}))
        paths[entry["id"]] = {**paths.get(entry["id"], {}), "source": str(directory)}
        save_backend_settings("vllm", {"artifact_paths": paths})
        try:
            with self.app.suspend():
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(command, env=huggingface_environment(), check=True)
                    missing = incomplete_files(entry, directory)
                    if missing:
                        raise ValueError(f"Download incomplete: {', '.join(missing)}")
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    pause_after_failure(str(error))
                    raise
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.notify(f"Download failed: {error}", severity="error")
            return
        self.refresh_curated()
        self.notify("All required artifact file sizes verified.")

    @on(Button.Pressed, "#vllm-prepare")
    def prepare_pressed(self) -> None:
        try:
            entry = self.selected_artifact()
            if not entry.get("requires_local_model"):
                raise ValueError("This artifact does not require preparation")
            source = Path(self.query_one("#vllm-download-directory", Input).value).expanduser().resolve()
            destination_value = self.query_one("#vllm-download-prepared", Input).value.strip()
            if not destination_value:
                raise ValueError("Choose a separate prepared checkpoint directory")
            destination = Path(destination_value).expanduser().resolve()
            if incomplete_files(entry, source):
                raise ValueError("Download / repair the complete original snapshot first")
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("Choose an empty destination; existing checkpoints are preserved")
            item = self.app.toolbox_catalog.toolboxes.get(self.query_one("#vllm-download-image", SearchableSelect).value)
            if not item or item.backend_config.get("checkpoint_preparation") != "ggz14-mtp-fp8":
                raise ValueError("Choose an installed GGZ14 or Radiance preparation toolbox")
            engine = str(self.query_one("#vllm-download-engine", SearchableSelect).value)
            command = build_prepare_cmd(engine, item.image, source, destination)
            space = disk_space_for_path(destination)
            required = 20_000_000_000
            if space and space.free < required:
                raise ValueError("Preparation needs about 20 GB free disk space")
            self.app.push_screen(ConfirmModal(f"Prepare a separate checkpoint in {destination}?\n"
                + download_space_note(required, space.free if space else None) + "\n" + shlex.join(command)),
                lambda confirmed: self._prepare(entry, engine, item.image, source, destination, command) if confirmed else None)
        except (OSError, ValueError) as error:
            self.notify(str(error), severity="error")

    def _prepare(self, entry: dict, engine: str, image: str, source: Path, destination: Path, command: list[str]) -> None:
        try:
            with self.app.suspend():
                try:
                    identity = json.loads(subprocess.check_output([engine, "image", "inspect", image], text=True))[0]
                    converter = subprocess.check_output([engine, "run", "--rm", "--network=none", "--entrypoint", "/bin/cat", image, "/opt/ggz14/converter.sha256"], text=True).split()[0]
                    destination.mkdir(parents=True, exist_ok=True)
                    subprocess.run(command, check=True)
                    if not checkpoint_ready(destination):
                        raise ValueError("Prepared checkpoint is incomplete")
                    receipt = {"repository": entry["repo"], "revision": entry["revision"], "image": image,
                               "image_id": identity.get("Id"), "converter_sha256": converter,
                               "files": [{"path": path.name, "size_bytes": path.stat().st_size} for path in destination.iterdir() if path.is_file()]}
                    (destination / "cockpit-preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
                except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
                    pause_after_failure(str(error))
                    raise
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            self.notify(f"Preparation failed: {error}", severity="error")
            return
        paths = dict(get_backend_settings("vllm").get("artifact_paths", {}))
        paths[entry["id"]] = {"source": str(source), "prepared": str(destination)}
        save_backend_settings("vllm", {"artifact_paths": paths})
        self.refresh_curated()
        self.notify(f"Prepared checkpoint verified. Select {destination} in Server Mode.", timeout=10)
