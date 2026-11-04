"""Small, shared network-options dialog for Hugging Face model downloads."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static

from ai_toolbox_cockpit.huggingface import DownloadOptions, get_download_options, save_download_options
from ai_toolbox_cockpit.widgets import CockpitCheckbox


class DownloadOptionsModal(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "Cancel")]

    DEFAULT_CSS = """
    DownloadOptionsModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #download-options-dialog {
        width: 90%;
        max-width: 76;
        height: auto;
        max-height: 95%;
        overflow-y: auto;
        border: solid #f2b544;
        background: #25292e;
        padding: 1 2;
    }
    #download-options-title {
        width: 100%;
        height: auto;
        text-style: bold;
        color: #f2b544;
        margin-bottom: 1;
    }
    DownloadOptionsModal Static, DownloadOptionsModal Label {
        height: auto;
        width: 100%;
    }
    #download-options-hint, #download-disable-xet, #download-timeout-row {
        margin-bottom: 1;
    }
    #download-timeout-row {
        height: 1;
        margin-top: 1;
    }
    #download-timeout-label { width: 28; }
    #download-timeout-seconds { width: 1fr; height: 1; border: none; }
    #download-options-error { color: $error; }
    #download-options-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    DownloadOptionsModal Button { height: 1; border: none; }
    """

    def compose(self) -> ComposeResult:
        options = get_download_options()
        with Vertical(id="download-options-dialog"):
            yield Label("Download options", id="download-options-title")
            yield Static(
                "On a slow network and downloads keep timing out? Try these options. "
                "Applies to Hugging Face downloads for all engines.",
                id="download-options-hint",
            )
            yield CockpitCheckbox("Disable Xet (use HTTP downloads)", value=options.disable_xet, id="download-disable-xet")
            yield CockpitCheckbox("Download one file at a time", value=options.single_file, id="download-single-file")
            yield CockpitCheckbox("Use a longer HTTP read timeout", value=options.timeout_seconds is not None, id="download-custom-timeout")
            with Horizontal(id="download-timeout-row"):
                yield Label("HTTP read timeout (seconds)", id="download-timeout-label")
                yield Input(
                    value=str(options.timeout_seconds or 300),
                    type="integer",
                    disabled=options.timeout_seconds is None,
                    id="download-timeout-seconds",
                )
            yield Static("The timeout applies to HTTP downloads, including when Xet is disabled. It limits waiting for data, not total download time.")
            yield Static("Saved choices apply to future downloads and override the corresponding shell settings. Uncheck the timeout to use Hugging Face's default.")
            yield Static("", id="download-options-error")
            with Horizontal(id="download-options-buttons"):
                yield Button("Save", variant="success", id="download-options-save")
                yield Button("Cancel", variant="primary", id="download-options-cancel")

    @on(Checkbox.Changed, "#download-custom-timeout")
    def timeout_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#download-timeout-seconds", Input).disabled = not event.value

    @on(Button.Pressed, "#download-options-cancel")
    def cancel(self) -> None:
        self.dismiss()

    @on(Button.Pressed, "#download-options-save")
    def save(self) -> None:
        timeout = None
        error = self.query_one("#download-options-error", Static)
        if self.query_one("#download-custom-timeout", Checkbox).value:
            field = self.query_one("#download-timeout-seconds", Input)
            try:
                timeout = int(field.value)
                if timeout < 1:
                    raise ValueError
            except ValueError:
                error.update("Enter a positive whole number of seconds.")
                field.focus()
                return
        options = DownloadOptions(
            disable_xet=self.query_one("#download-disable-xet", Checkbox).value,
            timeout_seconds=timeout,
            single_file=self.query_one("#download-single-file", Checkbox).value,
        )
        if not save_download_options(options):
            error.update("Could not save download options. Try again or cancel.")
            return
        self.notify("Download options saved.")
        self.dismiss()
