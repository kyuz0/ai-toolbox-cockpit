from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, OptionList, Button, Label
from textual import on, events
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget


# ── Confirm / Select Modals ─────────────────────────────────────────────────

class ConfirmModal(ModalScreen[bool]):
    """A modal dialog that asks a Yes/No question."""
    def __init__(self, message: str, yes_text: str = "Yes", no_text: str = "No", id: str = None):
        super().__init__(id=id)
        self.message = message
        self.yes_text = yes_text
        self.no_text = no_text

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm_dialog"):
            yield Label(self.message, id="confirm_message")
            with Horizontal(id="confirm_buttons"):
                yield Button(self.yes_text, variant="error", id="btn_yes")
                yield Button(self.no_text, variant="primary", id="btn_no")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


class SelectModal(ModalScreen[int]):
    """A modal dialog that asks the user to select an option."""
    def __init__(self, title: str, options: list[str], id: str = None):
        super().__init__(id=id)
        self.title = title
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="select_dialog"):
            yield Label(self.title, id="select_title")
            yield OptionList(*self.options, id="select_list")
            with Horizontal(id="select_buttons"):
                yield Button("Cancel", variant="error", id="btn_cancel")

    @on(OptionList.OptionSelected, "#select_list")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_index)

    @on(Button.Pressed, "#btn_cancel")
    def on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)


# ── Dropdown Overlay (screen-level floating list) ───────────────────────────

class _DropdownOverlay(Widget):
    """
    A free-floating dropdown mounted directly on the Screen's overlay layer.
    Positioned absolutely using screen-space coordinates.
    Navigation is driven externally via cursor_down/cursor_up/select_current.
    Mouse clicks on options fire the on_select callback immediately.
    """

    DEFAULT_CSS = """
    _DropdownOverlay {
        layer: overlay;
        background: #1e1e1e;
        border: solid #d32f2f;
        height: auto;
        max-height: 12;
        width: 40;
    }
    _DropdownOverlay > OptionList {
        background: transparent;
        border: none;
        height: auto;
        max-height: 10;
        width: 1fr;
    }
    """

    def __init__(
        self,
        options: list[tuple[str, str]],
        on_select,
        x: int,
        y: int,
        width: int,
        highlighted_value: str | None = None,
    ):
        super().__init__()
        self._options = list(options)
        self._on_select = on_select
        self._pos_x = x
        self._pos_y = y
        self._pos_width = width
        self._highlighted_value = highlighted_value
        # Overlay must not steal focus
        self.can_focus = False

    def compose(self) -> ComposeResult:
        opt_list = OptionList(*[label for label, _ in self._options])
        opt_list.can_focus = False
        yield opt_list

    def on_mount(self) -> None:
        self.styles.offset = (self._pos_x, self._pos_y)
        self.styles.width = self._pos_width
        self._set_highlight(self._highlighted_value)

    def _set_highlight(self, preferred_value: str | None = None) -> None:
        """Highlight the selected value, or the first available option."""
        opt_list = self.query_one(OptionList)
        highlighted = 0 if self._options else None
        if preferred_value is not None:
            for index, (_, value) in enumerate(self._options):
                if value == preferred_value:
                    highlighted = index
                    break
        opt_list.highlighted = highlighted

    def update_options(
        self,
        options: list[tuple[str, str]],
        highlighted_value: str | None = None,
    ) -> None:
        self._options = list(options)
        opt_list = self.query_one(OptionList)
        opt_list.clear_options()
        for label, _ in self._options:
            opt_list.add_option(label)
        self._set_highlight(highlighted_value)

    def cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def select_current(self) -> None:
        """Select whichever item the OptionList cursor is on."""
        opt_list = self.query_one(OptionList)
        idx = opt_list.highlighted
        if idx is not None and 0 <= idx < len(self._options):
            label, val = self._options[idx]
            self._on_select(val, label)

    @on(OptionList.OptionSelected)
    def _on_mouse_click_option(self, event: OptionList.OptionSelected) -> None:
        """Handle mouse clicks on options (fires even when can_focus=False)."""
        event.stop()
        index = event.option_index
        if 0 <= index < len(self._options):
            label, value = self._options[index]
            self._on_select(value, label)


# ── SearchableSelect ─────────────────────────────────────────────────────────

class SearchableSelect(Widget):
    """
    A filterable combobox widget.

    Behaviour:
    - Closed → displays the committed selection.
    - Click, Enter, Space, or ↑/↓ → starts an empty search and shows all options.
    - Typing while closed → starts a new search with that character.
    - Type while open → filters the list in real time.
    - ↑/↓ → navigate list while keeping focus on Input.
    - Enter → confirm highlighted option.
    - Escape or Tab/blur → cancel the search and restore the selection.

    Emits SearchableSelect.Changed when the selected value changes.
    Compatible with Select.Changed drop-in replacement pattern via `.value`.
    """

    DEFAULT_CSS = """
    SearchableSelect {
        height: 1;
        width: 1fr;
        background: #262626;
    }
    SearchableSelect > Input {
        border: none;
        height: 1;
        width: 1fr;
        background: transparent;
        padding: 0 3 0 1;
    }
    SearchableSelect > .dropdown-indicator {
        dock: right;
        width: 3;
        height: 1;
        background: transparent;
        color: #e57373;
        content-align: center middle;
    }
    SearchableSelect:focus-within {
        background: #303030;
    }
    SearchableSelect > Input:focus {
        color: #e57373;
        text-style: bold;
    }
    """

    class Changed(Message):
        """Posted when the selected value changes."""
        def __init__(self, value: str, select: "SearchableSelect"):
            self.value = value
            self.select = select
            super().__init__()

        @property
        def control(self) -> "SearchableSelect":
            return self.select

    def __init__(self, prompt: str = "Search...", id: str = None):
        super().__init__(id=id)
        self.border_title = None
        self.prompt = prompt
        self._options: list[tuple[str, str]] = []
        self._current_value: str = ""
        self._overlay: _DropdownOverlay | None = None
        self._search_active: bool = False

    def compose(self) -> ComposeResult:
        # Selecting on focus lets keyboard users replace the closed value with
        # a search term in one keystroke, just like clicking to open it.
        yield Input(placeholder=self.prompt, select_on_focus=True)
        yield Label("▼", classes="dropdown-indicator")

    # ── Public API ──────────────────────────────────────────────────────────

    def set_options(self, options: list) -> None:
        """Set options as (label, value) tuples or plain strings."""
        self._options = []
        for opt in options:
            if isinstance(opt, tuple):
                self._options.append((str(opt[0]), str(opt[1])))
            else:
                self._options.append((str(opt), str(opt)))
        if self._overlay and self._overlay.is_attached:
            query = self.query_one(Input).value
            filtered = self._get_filtered(query)
            preferred = self._current_value if not query else None
            self._overlay.update_options(filtered, preferred)

    @property
    def value(self) -> str:
        return self._current_value

    @value.setter
    def value(self, new_value: str) -> None:
        self._current_value = new_value
        self._search_active = False
        self._close_overlay()
        self._restore_selected_label()
        self.post_message(self.Changed(new_value, self))

    def focus_input(self) -> None:
        self.query_one(Input).focus()

    # ── Overlay lifecycle ───────────────────────────────────────────────────

    def _get_filtered(self, term: str = "") -> list[tuple[str, str]]:
        if not term:
            return self._options
        t = term.lower()
        return [(l, v) for l, v in self._options if t in l.lower() or t in v.lower()]

    def _selected_label(self) -> str:
        """Return the display label for the committed value."""
        for label, value in self._options:
            if value == self._current_value:
                return label
        return self._current_value

    def _restore_selected_label(self) -> None:
        inp = self.query_one(Input)
        with inp.prevent(Input.Changed):
            inp.value = self._selected_label()
        inp.placeholder = self.prompt
        self.query_one(".dropdown-indicator", Label).update("▼")

    def _begin_search(self, initial_query: str = "") -> None:
        """Enter search mode without changing the committed value."""
        if self._search_active or not self._options:
            return
        self._search_active = True
        inp = self.query_one(Input)
        with inp.prevent(Input.Changed):
            inp.value = initial_query
        inp.placeholder = "Type to filter..."
        self.query_one(".dropdown-indicator", Label).update("▲")
        self._open_overlay(initial_query)
        inp.focus()

    def _cancel_search(self) -> None:
        """Discard the temporary query and return to the committed selection."""
        if not self._search_active:
            return
        self._search_active = False
        self._close_overlay()
        self._restore_selected_label()

    def _open_overlay(self, filter_term: str = "") -> None:
        if self._overlay and self._overlay.is_attached:
            return
        if not self._options:
            return
        filtered = self._get_filtered(filter_term)
        inp = self.query_one(Input)
        try:
            region = inp.screen_region
        except Exception:
            region = inp.region
        overlay = _DropdownOverlay(
            options=filtered,
            on_select=self._on_option_selected,
            x=region.x,
            y=region.y + region.height,
            width=region.width,
            highlighted_value=self._current_value if not filter_term else None,
        )
        self._overlay = overlay
        self.app.screen.mount(overlay)

    def _close_overlay(self) -> None:
        if self._overlay and self._overlay.is_attached:
            self._overlay.remove()
        self._overlay = None

    def _on_option_selected(self, value: str, label: str) -> None:
        """Called by the overlay when user selects an option (keyboard or mouse)."""
        self._current_value = value
        self._search_active = False
        inp = self.query_one(Input)
        with inp.prevent(Input.Changed):
            inp.value = label
        inp.placeholder = self.prompt
        self.query_one(".dropdown-indicator", Label).update("▼")
        self._close_overlay()
        self.post_message(self.Changed(value, self))

    # ── Event handlers ──────────────────────────────────────────────────────

    @on(Input.Changed)
    def _on_input_changed(self, event: Input.Changed) -> None:
        if not self._search_active:
            # Space is an activation key for a closed combobox, not a useful
            # first filter term.
            initial_query = "" if event.value.isspace() else event.value
            self._begin_search(initial_query)
            return
        term = event.value
        if self._overlay and self._overlay.is_attached:
            filtered = self._get_filtered(term)
            self._overlay.update_options(filtered)
        else:
            # Open overlay as soon as user starts typing
            self._open_overlay(term)

    @on(events.Click)
    def _on_widget_clicked(self, event: events.Click) -> None:
        """Start a fresh search when clicking the widget."""
        self._begin_search()

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard navigation while overlay is open."""
        if not self._search_active:
            if event.key in ("enter", "space", "down", "up"):
                self._begin_search()
                event.prevent_default()
                event.stop()
            return

        if not (self._overlay and self._overlay.is_attached):
            return

        if event.key == "down":
            event.prevent_default()
            event.stop()
            self._overlay.cursor_down()
        elif event.key == "up":
            event.prevent_default()
            event.stop()
            self._overlay.cursor_up()
        elif event.key == "enter":
            event.prevent_default()
            event.stop()
            self._overlay.select_current()
        elif event.key == "escape":
            event.prevent_default()
            event.stop()
            self._cancel_search()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        """Cancel search when the Input loses focus (tab / click elsewhere)."""
        # Small delay so a mouse-click on the overlay fires its OptionSelected first
        self.set_timer(0.15, self._cancel_search)
