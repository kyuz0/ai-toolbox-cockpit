"""Download the exact R9V package, verify it, and prepare its separate PLE file."""

import shlex
import subprocess
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Input, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.huggingface import get_hf_token, huggingface_environment, save_hf_token
from ai_toolbox_cockpit.runtime.engines import detect_container_engines
from ai_toolbox_cockpit.runtime.terminal import pause_after_failure
from ai_toolbox_cockpit.settings import get_backend_settings
from ai_toolbox_cockpit.storage import disk_space_for_path, disk_space_text, download_space_note
from ai_toolbox_cockpit.widgets import ConfirmModal, HfTokenModal, SearchableSelect
from .model_manager import (PLE_FILENAME, get_download_cmd, get_package, get_paths, incomplete_files,
                            mount_path, ple_ready, save_paths, verify_file, verify_package)
from .runner import build_prepare_cmd


class R9vModelPanel(BackendModelPanel):
    backend_label = "R9V Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self.platform_id = ""
        self._hf_token = get_hf_token()
        self._hf_token_prompted = False
        self._pending: dict = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Tested package: Qwen3.8 Flash Next IQ4_XS + FP8 MTP + Q8 vision, "
                         "for 2× R9700 32 GB + 64 GB host RAM. Download includes all three target "
                         "shards, MTP, vision, tokenizer and expert placement. A generic GGUF is insufficient.",
                         classes="panel-copy")
            with Vertical(classes="model-zone"):
                with Horizontal(classes="inline-row"):
                    yield Label("Model package", id="r9v-download-model-label", classes="inline-label")
                    yield SearchableSelect("Select tested R9V package", id="r9v-download-model")
                yield Static("", id="r9v-package-details", classes="panel-copy")
                for key, label in (("models_dir", "Package directory"), ("ple_dir", "PLE directory")):
                    with Horizontal(classes="inline-row"):
                        yield Label(label, id=f"r9v-model-{key}-label", classes="inline-label")
                        yield Input(value=str(get_paths()[key]), id=f"r9v-model-{key}")
                with Horizontal(classes="inline-row"):
                    yield Button("Save Paths", id="r9v-model-save-paths")
                    yield Button("Scan Local", id="r9v-model-scan")
                yield Static("", id="r9v-disk-space", classes="storage-copy")
                yield DataTable(id="r9v-local-models", cursor_type="row", zebra_stripes=True)
                with Horizontal(classes="inline-row"):
                    yield Button("1. Download / Repair", id="r9v-download", variant="success")
                    yield Button("Verify SHA256", id="r9v-verify")
            with Vertical(classes="model-zone"):
                yield Label("2. Prepare PLE on NVMe", classes="zone-title")
                yield Static("Extracts a separate 26.82 GiB file; allow about 118 GiB total for package + PLE. "
                             "Pull/create the R9V toolbox first. Preparation uses its CPU tools and no GPU devices.",
                             classes="panel-copy")
                for control, label in (("engine", "Container engine"), ("image", "Toolbox image")):
                    with Horizontal(classes="inline-row"):
                        yield Label(label, id=f"r9v-model-{control}-label", classes="inline-label")
                        yield SearchableSelect(f"Select {label.lower()}", id=f"r9v-model-{control}")
                yield Button("Prepare PLE", id="r9v-prepare", variant="primary")
                yield Static("Then select R9V in Server Mode. Scans check file sizes; Verify SHA256 reads "
                             "the full package and any prepared PLE. Downloads are hash-verified automatically.", classes="panel-copy")

    def on_mount(self) -> None:
        select = self.query_one("#r9v-download-model", SearchableSelect)
        select.set_options([(entry["name"], entry["id"]) for entry in self.catalog.entries])
        select.value = next(entry["id"] for entry in self.catalog.entries if entry.get("recommended"))
        engine = self.query_one("#r9v-model-engine", SearchableSelect)
        engines = [(x.value, x.value) for x in detect_container_engines()]
        engine.set_options(engines)
        saved = get_backend_settings("r9v").get("engine", "")
        engine.value = saved if saved in dict(engines) else (engines[0][1] if engines else "")
        self.query_one("#r9v-local-models", DataTable).add_columns("Package", "Package files", "PLE")
        self.set_platform(self.app.active_platform_id)
        self.refresh_inventory()

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        items = [x for x in self.app.toolbox_catalog.platform_toolboxes(platform_id) if x.backend == "r9v"]
        select = self.query_one("#r9v-model-image", SearchableSelect)
        select.set_options([(x.name, x.id) for x in items])
        select.value = items[0].id if items else ""
        for control in ("download", "verify", "prepare"):
            self.query_one(f"#r9v-{control}", Button).disabled = not items

    @on(SearchableSelect.Changed, "#r9v-download-model")
    def package_changed(self) -> None:
        if self.is_mounted:
            package = get_package(self.query_one("#r9v-download-model", SearchableSelect).value)
            size = sum(x["size_bytes"] for x in package["files"]) / 1024**3
            self.query_one("#r9v-package-details", Static).update(
                f"{package['repo']}\nRevision: {package['revision']}\n"
                f"{size:.2f} GiB package + 26.82 GiB PLE. License: {package['license']}.")

    def refresh_inventory(self) -> None:
        paths = get_paths()
        for key in ("models_dir", "ple_dir"):
            self.query_one(f"#r9v-model-{key}", Input).value = str(paths[key])
        table = self.query_one("#r9v-local-models", DataTable)
        table.clear()
        for entry in self.catalog.entries:
            missing = incomplete_files(entry, paths["models_dir"])
            table.add_row(entry["name"], f"{len(missing)} missing/incomplete" if missing else "Sizes OK",
                          "Size OK" if ple_ready(entry, paths["ple_dir"]) else "Preparation required", key=entry["id"])
        self.query_one("#r9v-disk-space", Static).update(
            "Package: " + disk_space_text(paths["models_dir"]) + "\nPLE: " + disk_space_text(paths["ple_dir"]))

    def refresh_all(self) -> None:
        self.refresh_inventory()
        self.app.refresh_server_model_inventory("r9v")

    def edited_paths(self) -> dict[str, str]:
        return {key: str(mount_path(self.query_one(f"#r9v-model-{key}", Input).value.strip()))
                for key in ("models_dir", "ple_dir")}

    @on(Button.Pressed, "#r9v-model-save-paths")
    def save_paths_pressed(self) -> None:
        try:
            if not save_paths(self.edited_paths()):
                raise ValueError("Could not create or save those directories.")
            self.refresh_all()
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")

    @on(Button.Pressed, "#r9v-model-scan")
    def scan_pressed(self) -> None:
        self.refresh_all()

    def select_operation(self, action: str) -> bool:
        try:
            if self.platform_id != "r9700":
                raise ValueError("Select the R9700 platform for R9V.")
            paths = self.edited_paths()
            package = get_package(self.query_one("#r9v-download-model", SearchableSelect).value)
            self._pending = {"action": action, "paths": paths, "package": package, "command": []}
            if action == "download":
                self._pending["command"] = get_download_cmd(package, Path(paths["models_dir"]))
            elif action == "prepare":
                item = self.app.toolbox_catalog.toolboxes.get(self.query_one("#r9v-model-image", SearchableSelect).value)
                if not item or item.backend != "r9v" or item.id not in self.app.toolbox_catalog.platform(self.platform_id).toolbox_ids:
                    raise ValueError("Select an R9V toolbox image.")
                self._pending["command"] = build_prepare_cmd(
                    engine=self.query_one("#r9v-model-engine", SearchableSelect).value, image=item.image,
                    models_dir=Path(paths["models_dir"]), ple_dir=Path(paths["ple_dir"]), package_id=package["id"])
            return True
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error", timeout=10)
            return False

    @on(Button.Pressed, "#r9v-download")
    def download_pressed(self) -> None:
        if not self.select_operation("download"):
            return
        self._hf_token = self._hf_token or get_hf_token()
        if not self._hf_token and not self._hf_token_prompted:
            self.app.push_screen(HfTokenModal(), self._token_received)
        else:
            self.confirm_operation()

    def _token_received(self, choice: tuple[str, bool] | None) -> None:
        if choice is None:
            return
        self._hf_token, remember = choice
        self._hf_token_prompted = True
        if self._hf_token and remember and not save_hf_token(self._hf_token):
            self.notify("Could not save the token; using it for this session.", severity="warning")
        self.confirm_operation()

    @on(Button.Pressed, "#r9v-verify")
    def verify_pressed(self) -> None:
        if self.select_operation("verify"):
            self.confirm_operation()

    @on(Button.Pressed, "#r9v-prepare")
    def prepare_pressed(self) -> None:
        if self.select_operation("prepare"):
            self.confirm_operation()

    def confirm_operation(self) -> None:
        action, package, paths = (self._pending[k] for k in ("action", "package", "paths"))
        directory = Path(paths["ple_dir" if action == "prepare" else "models_dir"])
        space = disk_space_for_path(directory)
        required = (package["ple"]["size_bytes"] if action == "prepare" else sum(
            x["size_bytes"] for x in incomplete_files(package, Path(paths["models_dir"]))))
        note = download_space_note(required, space.free if space else None) if action != "verify" else "Reads full file hashes; this can take minutes."
        license_note = (f"By choosing Download you accept {package['license']}. Read: "
                        f"https://huggingface.co/{package['repo']}/blob/{package['revision']}/LICENSE\n") if action == "download" else ""
        label = {"download": "Download", "prepare": "Prepare PLE", "verify": "Verify SHA256"}[action]
        self.app.push_screen(ConfirmModal(
            f"{label}: {package['name']}\nPackage: {paths['models_dir']}\nPLE: {paths['ple_dir']}\n"
            f"{license_note}\n{note}\n\n{shlex.join(self._pending['command'])}", yes_text=label), self._operation_confirmed)

    def _operation_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        action, package, paths = (self._pending[k] for k in ("action", "package", "paths"))
        if not save_paths(paths):
            self.notify("Could not create or save the directories.", severity="error")
            return
        failed = False
        with self.app.suspend():
            try:
                if action in {"download", "prepare"}:
                    subprocess.run(self._pending["command"], check=True,
                                   env=huggingface_environment(self._hf_token) if action == "download" else None)
                if action in {"download", "verify"}:
                    verify_package(package, Path(paths["models_dir"]))
                if action == "verify" and (Path(paths["ple_dir"]) / PLE_FILENAME).exists():
                    verify_file(Path(paths["ple_dir"]) / PLE_FILENAME, package["ple"]["sha256"])
            except KeyboardInterrupt:
                failed = True
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                failed = True
                pause_after_failure(f"R9V {action} failed: {error}")
        self.refresh_all()
        if failed:
            self.notify("R9V operation failed or was interrupted. Review the terminal output.", severity="error")
        else:
            self.notify("R9V operation completed. Inventory refreshed; prepare PLE if still required.")
