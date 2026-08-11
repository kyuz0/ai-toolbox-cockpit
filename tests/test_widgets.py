from unittest import IsolatedAsyncioTestCase

from textual.app import App, ComposeResult

from ai_toolbox_cockpit.widgets import SearchableSelect, selection_marker


class _SelectorApp(App):
    def compose(self) -> ComposeResult:
        yield SearchableSelect("Search choices", id="selector")

    def on_mount(self) -> None:
        self.query_one("#selector", SearchableSelect).set_options([
            ("First choice", "first"),
            ("Second choice", "second"),
        ])


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
