from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, Static

from ai_toolbox_cockpit.huggingface import HubTokenChoice, get_hf_token, has_saved_hf_token, save_hf_token
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
    result: HubTokenChoice | None = None

    def on_mount(self) -> None:
        self.push_screen(HfTokenModal(), self._token_entered)

    def _token_entered(self, result: HubTokenChoice | None) -> None:
        self.result = result


class _ConfirmApp(App):
    result: bool | None = None

    def __init__(self, copy_text: str | None) -> None:
        super().__init__()
        self.copy_text = copy_text

    def on_mount(self) -> None:
        self.push_screen(
            ConfirmModal("Run this command?", copy_text=self.copy_text),
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


class HfTokenModalTests(IsolatedAsyncioTestCase):
    async def test_token_is_masked_and_can_replace_the_saved_token(self) -> None:
        self.assertTrue(save_hf_token("hf_previous"))
        with patch.dict("os.environ", {"HF_TOKEN": ""}):
            app = _HfTokenApp()
            async with app.run_test(size=(80, 24)) as pilot:
                message = app.screen.query_one("#hf-token-message", Static)
                token_input = app.screen.query_one("#hf-token-input", Input)
                remember = app.screen.query_one("#hf-token-remember", Checkbox)
                self.assertIn("A token is configured", str(message.render()))
                self.assertTrue(token_input.password)
                self.assertEqual(token_input.value, "")
                token_input.value = "hf_example"
                remember.value = True
                await pilot.click("#hf-token-continue")
                await pilot.pause()
            self.assertEqual(app.result, "hf_example")
            self.assertEqual(get_hf_token(), "hf_example")

    async def test_use_token_with_blank_input_selects_current_credential(self) -> None:
        with patch.dict("os.environ", {"HF_TOKEN": "hf_shell"}):
            app = _HfTokenApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.click("#hf-token-continue")
                await pilot.pause()
            self.assertEqual(app.result, "hf_shell")

    async def test_clear_removes_only_the_saved_token_and_anonymous_is_explicit(self) -> None:
        self.assertTrue(save_hf_token("hf_previous"))
        with patch.dict("os.environ", {"HF_TOKEN": "hf_shell"}):
            app = _HfTokenApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.click("#hf-token-clear")
                self.assertFalse(has_saved_hf_token())
                self.assertTrue(app.screen.query_one("#hf-token-clear", Button).disabled)
                self.assertIn("The shell supplies HF_TOKEN", str(app.screen.query_one("#hf-token-message", Static).render()))
                await pilot.click("#hf-token-anonymous")
                await pilot.pause()
            self.assertIs(app.result, False)
