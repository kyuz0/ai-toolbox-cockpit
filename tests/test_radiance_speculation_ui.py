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
    from dataclasses import replace
    other = replace(flash, id="dflash-test-fixture", name="DFlash test fixture", speculation="dflash2", draft_tokens=7)
    fixtures = (flash, other)
    save_backend_settings("radiance", {"model": flash.id, "speculation": saved_choice, "draft_tokens": "5"})
    class TestApp(App):
        active_platform_id = "r9700"
        toolbox_catalog = load_toolbox_catalog()
        def compose(self):
            yield RadianceServerPanel(id="panel")
    async def check():
        with patch("ai_toolbox_cockpit.backends.radiance.server.detect_container_engines", return_value=[]), patch("ai_toolbox_cockpit.backends.radiance.server.load_models", return_value=fixtures), patch("ai_toolbox_cockpit.backends.radiance.server.get_model", side_effect=lambda key: next(model for model in fixtures if model.id == key)):
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


@pytest.mark.parametrize('saved', [False, True])
def test_21_profile_defaults_and_explicit_overrides_survive_switch_and_refresh(saved):
    from ai_toolbox_cockpit.backends.radiance.model_manager import RadianceModel
    original = next(item for item in load_models() if item.speculation == 'mtp')
    model = RadianceModel.from_entry(dict(vars(original), id='iqrv4-ui-fixture',
        config_format='yaml', kv_cache_dtype='i8', context_size=262144, default_profile='single-1m',
        serving_profiles=[dict(id='single-1m', name='Single user · 1M', tensor_parallel=4,
            context_size=1000000, rope_scaling='yarn:4:262144'),
            dict(id='ram', name='RAM n-grams', tensor_parallel=4, max_sessions=16,
                 default_sessions=1, context_size=262144)]))
    if saved:
        save_backend_settings('radiance', {'model': model.id, 'profile': 'single-1m',
            'context': '2000001', 'scaling': 'yarn:8:262144', 'sessions': '1'})
    class TestApp(App):
        active_platform_id = 'r9700'
        toolbox_catalog = load_toolbox_catalog()
        def compose(self):
            yield RadianceServerPanel(id='panel')
    async def check():
        module = 'ai_toolbox_cockpit.backends.radiance.server'
        with patch(module + '.detect_container_engines', return_value=[]), \
             patch(module + '.load_models', return_value=(model,)), \
             patch(module + '.get_model', return_value=model):
            for _ in range(2):
                app = TestApp()
                async with app.run_test(size=(140, 48)) as pilot:
                    await pilot.pause()
                    panel = app.query_one('#panel')
                    profile = app.query_one('#radiance-profile', SearchableSelect)
                    assert profile.value == 'single-1m'
                    await pilot.pause()
                    context = app.query_one('#radiance-context', Input)
                    scaling = app.query_one('#radiance-scaling', Input)
                    sessions = app.query_one('#radiance-sessions', Input)
                    assert context.value == ('2000001' if saved else '1000000')
                    assert scaling.value == ('yarn:8:262144' if saved else 'yarn:4:262144')
                    assert scaling.display
                    assert sessions.value == '1'
                    context.value = '3000001'
                    scaling.value = 'yarn:12:262144'
                    profile.value = 'ram'
                    await pilot.pause()
                    assert (context.value, scaling.value, sessions.value) == ('262144', 'none', '1')
                    sessions.value = '8'
                    panel.refresh_model_inventory()
                    await pilot.pause()
                    assert sessions.value == '8'
                    profile.value = 'single-1m'
                    await pilot.pause()
                    assert (context.value, scaling.value) == ('3000001', 'yarn:12:262144')
                    panel.reset_context()
                    assert context.value == '1000000'
                    assert scaling.value == 'yarn:12:262144'
                    panel.reset_scaling()
                    assert scaling.value == 'yarn:4:262144'
                    profile.value = 'ram'
                    await pilot.pause()
                    assert sessions.value == '8'
                    app.save_screenshot('/tmp/radiance21-profile-controls.svg')
    asyncio.run(check())



def test_version_selection_follows_model_and_missing_matching_image_disables_start():
    from dataclasses import replace
    from types import SimpleNamespace
    from textual.widgets import Button
    old = replace(load_models()[0], toolbox_ids=('radiance-legacy',))
    new = replace(old, id='iqrv4-version-fixture', config_format='yaml',
                  kv_cache_dtype='i8', toolbox_ids=('radiance-21',))
    models = {old.id: old, new.id: new}
    images = [SimpleNamespace(id='radiance-legacy', name='Radiance 1.12', backend='radiance'),
              SimpleNamespace(id='radiance-21', name='Radiance 2.1', backend='radiance')]
    save_backend_settings('radiance', {'model': old.id, 'image': 'radiance-legacy'})
    class TestApp(App):
        active_platform_id = 'r9700'
        toolbox_catalog = SimpleNamespace(platform_toolboxes=lambda _: images)
        def compose(self):
            yield RadianceServerPanel(id='panel')
    async def check():
        module = 'ai_toolbox_cockpit.backends.radiance.server'
        with patch(module + '.detect_container_engines', return_value=[]), \
             patch(module + '.load_models', return_value=tuple(models.values())), \
             patch(module + '.get_model', side_effect=models.__getitem__):
            app = TestApp()
            async with app.run_test(size=(140, 48)) as pilot:
                await pilot.pause()
                model = app.query_one('#radiance-model', SearchableSelect)
                image = app.query_one('#radiance-image', SearchableSelect)
                assert image.value == 'radiance-legacy'
                model.value = new.id
                await pilot.pause()
                assert image.value == 'radiance-21'
                assert not app.query_one('#radiance-start', Button).disabled
                model.value = old.id
                await pilot.pause()
                assert image.value == 'radiance-legacy'
                images.pop()
                model.value = new.id
                await pilot.pause()
                assert image.value == ''
                assert app.query_one('#radiance-start', Button).disabled
    asyncio.run(check())
