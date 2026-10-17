from unittest import IsolatedAsyncioTestCase

from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, Label, Static

from ai_toolbox_cockpit.widgets import (
    ConfirmModal,
    HfTokenModal,
    SearchableSelect,
    selection_marker,
)


class _SelectorApp(App):
    def compose(self) -> ComposeResult:
        yield SearchableSelect("Search choices", id="selector")

    def on_mount(self) -> None:
        self.query_one("#selector", SearchableSelect).set_options([
            ("First choice", "first"),
            ("Second choice", "second"),
        ])


class _HfTokenApp(App):
    result: tuple[str, bool] | None = None

    def on_mount(self) -> None:
        self.push_screen(HfTokenModal(), self._token_entered)

    def _token_entered(self, result: tuple[str, bool] | None) -> None:
        self.result = result


class _ConfirmApp(App):
    result: bool | None = None

    def __init__(self, copy_text: str | None, message: str = "Run this command?") -> None:
        super().__init__()
        self.copy_text = copy_text
        self.message = message

    def on_mount(self) -> None:
        self.push_screen(
            ConfirmModal(self.message, copy_text=self.copy_text),
            self._confirmed,
        )

    def _confirmed(self, result: bool) -> None:
        self.result = result


class SearchableSelectTests(IsolatedAsyncioTestCase):
    async def test_keyboard_opens_and_selects_at_narrow_terminal_size(self) -> None:
        app = _SelectorApp()
        async with app.run_test(size=(80, 24)) as pilot:
            selector = app.query_one("#selector", SearchableSelect)
            selector.focus_input()
            await pilot.press("down", "enter")
            self.assertEqual(selector.value, "first")


class SelectionMarkerTests(IsolatedAsyncioTestCase):
    async def test_markers_are_literal_rich_text(self) -> None:
        self.assertEqual(selection_marker(False).plain, "[ ]")
        self.assertEqual(selection_marker(True).plain, "[x]")


class ConfirmModalTests(IsolatedAsyncioTestCase):
    async def test_copy_command_button_copies_without_dismissing_dialog(self) -> None:
        command = "podman run example/image:latest"
        app = _ConfirmApp(command)
        async with app.run_test(size=(100, 30)) as pilot:
            copy_button = app.screen.query_one("#btn_copy", Button)
            self.assertEqual(str(copy_button.label), "Copy command")

            await pilot.click("#btn_copy")
            await pilot.pause()

            self.assertEqual(app.clipboard, command)
            self.assertIsInstance(app.screen, ConfirmModal)
            self.assertIsNone(app.result)

            await pilot.click("#btn_no")
            await pilot.pause()

        self.assertFalse(app.result)

    async def test_copy_button_is_omitted_without_command_text(self) -> None:
        app = _ConfirmApp(None)
        async with app.run_test(size=(100, 30)):
            self.assertEqual(len(app.screen.query("#btn_copy")), 0)

    async def test_command_preview_with_brackets_renders_literally(self) -> None:
        command = "podman run --device /dev/dri ['--entrypoint', '/opt/vllm/bin/python']"
        app = _ConfirmApp(command, message=f"Start vLLM server?\n\n{command}")
        async with app.run_test(size=(100, 30)):
            message = app.screen.query_one("#confirm_message", Label)
            self.assertEqual(str(message.render()), f"Start vLLM server?\n\n{command}")


class HfTokenModalTests(IsolatedAsyncioTestCase):
    async def test_token_is_masked_and_remember_choice_is_returned(self) -> None:
        app = _HfTokenApp()
        async with app.run_test(size=(100, 30)) as pilot:
            message = app.screen.query_one("#hf-token-message", Static)
            token_input = app.screen.query_one("#hf-token-input", Input)
            remember = app.screen.query_one("#hf-token-remember", Checkbox)

            self.assertIn("make downloads faster", str(message.render()))
            self.assertTrue(token_input.password)
            token_input.value = "hf_example"
            remember.value = True
            await pilot.click("#hf-token-continue")
            await pilot.pause()

        self.assertEqual(app.result, ("hf_example", True))
