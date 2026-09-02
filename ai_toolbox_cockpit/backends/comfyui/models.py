"""ComfyUI workflow bundle browser and toolbox model-manager bridge."""

import subprocess

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Static

from ai_toolbox_cockpit.backends.base import BackendModelPanel
from ai_toolbox_cockpit.runtime.interactive import runtime_environment
from ai_toolbox_cockpit.runtime.terminal import command_failed, pause_after_failure
from ai_toolbox_cockpit.runtime.toolboxes import (
    inspect_installed_toolboxes,
    run_in_toolbox_command,
    runtime_for_installed_toolbox,
)
from ai_toolbox_cockpit.widgets import SearchableSelect


class ComfyUiModelPanel(BackendModelPanel):
    backend_label = "ComfyUI Workflows & Models"

    def __init__(self, catalog, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self.platform_id = ""

    def compose(self) -> ComposeResult:
        yield Static(
            "Browse every workflow/model family provided by the image. Downloads remain owned by the toolbox's model_manager, which understands each bundle's dependency chain and variants.",
            classes="panel-copy",
        )
        with Horizontal(classes="inline-row"):
            yield Label("Toolbox", id="comfy-manager-toolbox-label", classes="inline-label")
            yield SearchableSelect("Select an installed ComfyUI toolbox", id="comfy-manager-toolbox")
            yield Button("Open Toolbox Model Manager", id="comfy-open-model-manager", variant="primary")
        yield Static(
            "The toolbox manager stores models under ~/comfy-models. Direct server mode mounts that path by default.",
            classes="storage-copy",
        )
        with Vertical(classes="model-zone"):
            yield Label("Workflow model bundles", classes="zone-title")
            yield DataTable(id="comfy-bundles", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#comfy-bundles", DataTable)
        table.add_columns("Bundle", "Downloader", "Variants", "Matching workflows")
        workflows = [str(value) for value in self.catalog.config.get("workflows", [])]
        for entry in self.catalog.entries:
            keywords = [str(value).lower() for value in entry.get("keywords", [])]
            excluded = [str(value).lower() for value in entry.get("exclude_keywords", [])]
            matches = [
                workflow
                for workflow in workflows
                if all(keyword in workflow.lower() for keyword in keywords)
                and not any(keyword in workflow.lower() for keyword in excluded)
            ]
            variants = "; ".join(
                str(variant.get("name", "")) for variant in entry.get("variants", [])
            )
            table.add_row(
                str(entry.get("name", entry["id"])),
                str(entry.get("script", entry.get("recipe_id", "model_manager"))),
                variants,
                ", ".join(matches) or "Image-managed",
                key=entry["id"],
            )
        self.set_platform(self.app.active_platform_id)

    def set_platform(self, platform_id: str) -> None:
        self.platform_id = platform_id
        if not self.is_mounted:
            return
        toolboxes = [
            toolbox
            for toolbox in self.app.toolbox_catalog.platform_toolboxes(platform_id)
            if toolbox.backend == "comfyui"
        ]
        select = self.query_one("#comfy-manager-toolbox", SearchableSelect)
        select.set_options([(f"{toolbox.name} — {toolbox.container_name}", toolbox.container_name) for toolbox in toolboxes])
        select.value = toolboxes[0].container_name if toolboxes else ""

    @on(Button.Pressed, "#comfy-open-model-manager")
    def open_manager_pressed(self) -> None:
        name = self.query_one("#comfy-manager-toolbox", SearchableSelect).value
        if not name:
            self.notify("This platform has no ComfyUI toolbox.", severity="warning")
            return
        installed = next(
            (item for item in inspect_installed_toolboxes() if item.name == name),
            None,
        )
        if not installed:
            self.notify("Create the selected ComfyUI toolbox first.", severity="warning")
            return
        runtime = runtime_for_installed_toolbox(installed)
        if not runtime:
            self.notify("No compatible Toolbx/Distrobox backend is installed.", severity="error")
            return
        command = run_in_toolbox_command(runtime, name, ["model_manager"])
        manager_error = ""
        with self.app.suspend():
            try:
                return_code = subprocess.call(
                    command, env=runtime_environment(runtime)
                )
            except OSError as error:
                return_code = 127
                manager_error = str(error)
            if command_failed(return_code):
                pause_after_failure(
                    f"Model manager failed: {manager_error}"
                    if manager_error
                    else f"Model manager exited with status {return_code}."
                )
        if command_failed(return_code):
            self.notify("Model manager failed.", severity="error", timeout=8)
