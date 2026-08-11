from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select, Static

from ai_toolbox_cockpit.backends.llama_cpp.benchmark_runner import (
    BenchmarkSettings,
    build_benchmark_jobs,
    run_benchmark_job,
    write_curve_summary,
)
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
    get_benchmark_results_dir,
    save_benchmark_results_dir,
    scan_local_models,
)
from ai_toolbox_cockpit.runtime.interactive import detect_interactive_backend
from ai_toolbox_cockpit.runtime.toolboxes import inspect_installed_toolboxes


class BenchmarksView(Vertical):
    """The existing llama.cpp depth-curve methodology, kept backend-specific."""

    def __init__(self, platform_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.platform_id = platform_id
        self.selected_toolboxes: set[str] = set()
        self.selected_models: set[str] = set()

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(
                "llama.cpp benchmark: fixed prefill and generation work at identical starting KV depths. DS4, vLLM, and ComfyUI require different methodologies and are not mixed into these results.",
                classes="view-note",
            )
            with Horizontal(classes="split-row"):
                with Vertical(classes="model-zone"):
                    yield Label("Installed llama.cpp toolboxes", classes="zone-title")
                    yield DataTable(id="benchmark-toolboxes", cursor_type="row", zebra_stripes=True)
                with Vertical(classes="model-zone"):
                    yield Label("Local GGUF models", classes="zone-title")
                    yield DataTable(id="benchmark-models", cursor_type="row", zebra_stripes=True)
            with Vertical(classes="model-zone"):
                with Horizontal(classes="inline-row"):
                    yield Input(value="65536", placeholder="Max starting depth", id="bench-max-context", type="integer")
                    yield Input(value="8192", placeholder="Depth step", id="bench-step", type="integer")
                    yield Input(value="2048", placeholder="Prefill", id="bench-prefill", type="integer")
                    yield Input(value="128", placeholder="Generation", id="bench-generation", type="integer")
                    yield Input(value="3", placeholder="Repetitions", id="bench-repetitions", type="integer")
                    yield Input(value="10", placeholder="Cooldown", id="bench-cooldown", type="integer")
                with Horizontal(classes="inline-row"):
                    yield Input(placeholder="ROCm u-batch (auto)", id="bench-rocm-ubatch", type="integer")
                    yield Input(placeholder="Vulkan u-batch (auto)", id="bench-vulkan-ubatch", type="integer")
                    yield Checkbox("Flash attention", value=True, id="bench-fa")
                    yield Checkbox("No memory mapping", value=True, id="bench-no-mmap")
                    yield Checkbox("KV cache quantization", id="bench-kv-enabled")
                    yield Select([(value, value) for value in ("q8_0", "q4_0", "iq4_nl")], value="q8_0", allow_blank=False, id="bench-kv-type")
                yield Input(placeholder="Extra llama-bench arguments", id="bench-extra")
                with Horizontal(classes="inline-row"):
                    yield Input(value=str(get_benchmark_results_dir()), id="bench-results-dir")
                    yield Button("Save Results Path", id="bench-save-path")
                    yield Button("Refresh", id="bench-refresh")
                    yield Button("Calibrate u-batch", id="bench-calibrate")
                    yield Button("Run Benchmark", id="bench-run", variant="primary")
            yield Static("", id="benchmark-status", classes="view-note")

    def on_mount(self) -> None:
        toolboxes = self.query_one("#benchmark-toolboxes", DataTable)
        toolboxes.add_columns("", "Toolbox", "Backend")
        models = self.query_one("#benchmark-models", DataTable)
        models.add_columns("", "Model", "Path")
        self.refresh_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        self.selected_toolboxes.clear()
        if self.is_mounted:
            self.refresh_inventory()

    def refresh_inventory(self) -> None:
        installed = {item.name for item in inspect_installed_toolboxes()}
        table = self.query_one("#benchmark-toolboxes", DataTable)
        table.clear()
        for toolbox in self.app.toolbox_catalog.platform_toolboxes(self.platform_id):
            if toolbox.backend != "llama_cpp" or toolbox.container_name not in installed:
                continue
            backend = "Vulkan" if "vulkan" in toolbox.container_name.lower() else "ROCm"
            table.add_row(
                "[x]" if toolbox.container_name in self.selected_toolboxes else "[ ]",
                toolbox.container_name, backend, key=toolbox.container_name,
            )
        model_table = self.query_one("#benchmark-models", DataTable)
        model_table.clear()
        for model in scan_local_models():
            model_table.add_row(
                "[x]" if model["path"] in self.selected_models else "[ ]",
                model["name"], model["path"], key=model["path"],
            )

    @on(DataTable.RowSelected, "#benchmark-toolboxes")
    def toolbox_selected(self, event: DataTable.RowSelected) -> None:
        value = str(event.row_key.value)
        self.selected_toolboxes.symmetric_difference_update({value})
        self.refresh_inventory()

    @on(DataTable.RowSelected, "#benchmark-models")
    def model_selected(self, event: DataTable.RowSelected) -> None:
        value = str(event.row_key.value)
        self.selected_models.symmetric_difference_update({value})
        self.refresh_inventory()

    @on(Button.Pressed, "#bench-refresh")
    def refresh_pressed(self) -> None:
        self.refresh_inventory()

    @on(Button.Pressed, "#bench-save-path")
    def save_path(self) -> None:
        if save_benchmark_results_dir(self.query_one("#bench-results-dir", Input).value):
            self.notify("Benchmark results path saved.")

    def _optional_int(self, selector: str) -> int | None:
        value = self.query_one(selector, Input).value.strip()
        return int(value) if value else None

    def _settings(self) -> BenchmarkSettings:
        load_mode = frozenset(
            toolbox.container_name
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(self.platform_id)
            if toolbox.supports_load_mode
        )
        return BenchmarkSettings(
            max_context=int(self.query_one("#bench-max-context", Input).value),
            context_step=int(self.query_one("#bench-step", Input).value),
            prefill=int(self.query_one("#bench-prefill", Input).value),
            generation=int(self.query_one("#bench-generation", Input).value),
            repetitions=int(self.query_one("#bench-repetitions", Input).value),
            delay=int(self.query_one("#bench-cooldown", Input).value),
            flash_attention=self.query_one("#bench-fa", Checkbox).value,
            use_mmap=not self.query_one("#bench-no-mmap", Checkbox).value,
            kv_cache_type=(str(self.query_one("#bench-kv-type", Select).value) if self.query_one("#bench-kv-enabled", Checkbox).value else ""),
            platform_id=self.platform_id,
            rocm_ubatch=self._optional_int("#bench-rocm-ubatch"),
            vulkan_ubatch=self._optional_int("#bench-vulkan-ubatch"),
            extra_args=self.query_one("#bench-extra", Input).value,
            load_mode_toolboxes=load_mode,
        )

    @on(Button.Pressed, "#bench-run")
    def run_pressed(self) -> None:
        runtime = detect_interactive_backend()
        if not runtime or not self.selected_toolboxes or not self.selected_models:
            self.notify("Select at least one installed toolbox and local model.", severity="warning")
            return
        try:
            settings = self._settings()
            results = Path(self.query_one("#bench-results-dir", Input).value).expanduser()
            jobs = build_benchmark_jobs(runtime.wrapper.value, sorted(self.selected_toolboxes), sorted(self.selected_models), results, settings)
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        completed = failed = skipped = 0
        with self.app.suspend():
            for index, job in enumerate(jobs, start=1):
                print(f"[{index}/{len(jobs)}] {job.toolbox_name} · {Path(job.model_path).name} · {job.series}")
                status, _ = run_benchmark_job(job)
                completed += status == "completed"
                failed += status == "failed"
                skipped += status == "skipped"
                if index < len(jobs) and status != "skipped" and settings.delay:
                    time.sleep(settings.delay)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            summary = results / f"llama-depth-curves-{timestamp}.csv"
            rows = write_curve_summary(jobs, summary)
            print(f"Summary: {summary} ({rows} rows)")
        self.query_one("#benchmark-status", Static).update(
            f"Completed {completed}, skipped {skipped}, failed {failed}."
        )

    @on(Button.Pressed, "#bench-calibrate")
    def calibrate_pressed(self) -> None:
        if len(self.selected_toolboxes) != 1 or len(self.selected_models) != 1:
            self.notify("Select exactly one toolbox and one model to calibrate.", severity="warning")
            return
        command = [
            sys.executable, "-m", "ai_toolbox_cockpit.backends.llama_cpp.ubatch_calibration",
            "--toolbox", next(iter(self.selected_toolboxes)),
            "--model", next(iter(self.selected_models)),
            "--platform", self.platform_id,
        ]
        with self.app.suspend():
            subprocess.call(command)
