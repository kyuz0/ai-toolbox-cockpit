"""vLLM Hugging Face catalogue, cache inventory, and non-mutating explorer."""

from pathlib import Path

from huggingface_hub import HfApi
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.settings import get_backend_settings, save_backend_settings


def cache_directory_for_repo(cache_root: Path, repo_id: str) -> Path:
    return cache_root / f"models--{repo_id.replace('/', '--')}"


class VllmModelPanel(BackendModelPanel):
    backend_label = "vLLM Models"

    def compose(self) -> ComposeResult:
        yield Label(self.backend_label, classes="panel-title")
        yield Static(
            "vLLM downloads repositories through Hugging Face when a server starts. This view is deliberately non-downloading: it shows maintained launch defaults, local cache state, and Hub search results.",
            classes="panel-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Hugging Face cache", classes="zone-title")
            with Horizontal(classes="inline-row"):
                yield Input(id="vllm-model-cache")
                yield Button("Save Path", id="vllm-save-model-cache")
                yield Button("Refresh", id="vllm-refresh-cache", variant="primary")
            yield DataTable(id="vllm-curated-models", cursor_type="row", zebra_stripes=True)
        with Vertical(classes="model-zone"):
            yield Label("Hugging Face Hub explorer", classes="zone-title")
            with Horizontal(classes="inline-row"):
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
        self.refresh_curated()

    def cache_root(self) -> Path:
        return Path(self.query_one("#vllm-model-cache", Input).value).expanduser()

    def refresh_curated(self) -> None:
        root = self.cache_root()
        table = self.query_one("#vllm-curated-models", DataTable)
        table.clear()
        for entry in self.catalog.entries:
            repo = str(entry.get("repo", ""))
            cached = cache_directory_for_repo(root, repo).is_dir()
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
            models = list(HfApi().list_models(search=query, sort="downloads", direction=-1, limit=50))
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
