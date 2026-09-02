from unittest import IsolatedAsyncioTestCase

from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Static

from ai_toolbox_cockpit.widgets import HfTokenModal, SearchableSelect, selection_marker


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
