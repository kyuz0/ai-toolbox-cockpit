"""Saved settings and model changes must preserve explicit speculation choices."""
import asyncio
from unittest.mock import patch
import pytest
from textual.app import App
from textual.widgets import Input
from ai_toolbox_cockpit.catalog import load_toolbox_catalog
from ai_toolbox_cockpit.backends.radiance.server import RadianceServerPanel
from ai_toolbox_cockpit.backends.radiance.model_manager import load_models
from ai_toolbox_cockpit.settings import save_backend_settings
from ai_toolbox_cockpit.widgets import SearchableSelect


@pytest.mark.parametrize("saved_choice,choice", [("baseline","baseline"), ("bundled","bundled"), ("invalid","bundled")])
def test_saved_decoding_and_depth_survive_reopen_refresh_and_model_switch(saved_choice, choice):
    models = load_models()
    flash = next(model for model in models if model.speculation == "mtp")
    other = next(model for model in models if model.speculation == "dflash2")
    save_backend_settings("radiance", {"model": flash.id, "speculation": saved_choice, "draft_tokens": "5"})
    class TestApp(App):
        active_platform_id = "r9700"
        toolbox_catalog = load_toolbox_catalog()
        def compose(self):
            yield RadianceServerPanel(id="panel")
    async def check():
        with patch("ai_toolbox_cockpit.backends.radiance.server.detect_container_engines", return_value=[]):
            for _ in range(2):
                app = TestApp()
                async with app.run_test(size=(120, 40)) as pilot:
                    await pilot.pause()
                    select = app.query_one("#radiance-model", SearchableSelect)
                    mode = app.query_one("#radiance-speculation", SearchableSelect)
                    depth = app.query_one("#radiance-draft-tokens", Input)
                    assert select.value == flash.id
                    assert mode.value == choice
                    assert depth.value == "5"
                    assert depth.disabled == (choice == "baseline")
                    app.query_one("#panel").refresh_model_inventory()
                    await pilot.pause()
                    assert depth.value == "5"
                    select.value = other.id
                    await pilot.pause()
                    assert depth.value == "7"
                    assert mode.value == choice
                    select.value = flash.id
                    await pilot.pause()
                    assert depth.value == "5"
                    assert mode.value == choice
    asyncio.run(check())
