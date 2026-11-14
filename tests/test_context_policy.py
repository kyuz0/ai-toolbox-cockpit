"""Hard gate: context metadata supplies defaults, never an override ceiling.

Fixtures create tiny files only. No engine, container or download is executed.
"""
import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("context", [262144, 1000000, 16777216])
@pytest.mark.parametrize("backend", ["radiance", "affinity", "hipfire", "strata", "strata_dsv41", "maya", "gufo", "halogen", "r9v", "llama_embedding", "llama_chat", "ds4", "vllm"])
def test_explicit_context_reaches_engine_unchanged(tmp_path, backend, context):
    common = dict(engine="podman", image="example/engine:test", engine_args=[])
    flag = "--context"
    if backend == "radiance":
        from ai_toolbox_cockpit.backends.radiance.model_manager import get_model
        from ai_toolbox_cockpit.backends.radiance.server_runner import build_server_cmd
        model = replace(get_model("radiance-qwen38-flash-iq4r-8267a673"), size_bytes=4)
        target = tmp_path / "model.rad"
        target.write_bytes(b"test")
        command = build_server_cmd(**common, platform_id="r9700", model=model, model_path=target, context_size=context)
        flag = "--max-model-len"
    elif backend == "affinity":
        from test_affinity_commands import command as build
        command = build(tmp_path, context_size=context)
        assert command[command.index("--kv-size") + 1] == str(context + 64)
        assert command[command.index("--kv-commit") + 1] == str(context + 64)
        return
    elif backend == "hipfire":
        from ai_toolbox_cockpit.backends.hipfire.model_manager import get_model
        from ai_toolbox_cockpit.backends.hipfire.server_runner import build_server_cmd
        model = replace(get_model("hipfire-deepseek-v4-flash-0731-mq2r"), size_bytes=4)
        target = tmp_path / model.directory / model.filename
        target.parent.mkdir(parents=True)
        target.write_bytes(b"test")
        command = build_server_cmd(**common, platform_id="r9700", model=model, models_dir=tmp_path, state_dir=tmp_path / "state", context_size=context)
        flag = "--max-seq"
    elif backend in {"strata", "strata_dsv41"}:
        from ai_toolbox_cockpit.backends.strata.server_runner import build_server_cmd
        if backend == "strata":
            from test_strata_commands import fixture
            model, _ = fixture(tmp_path)
            platform = "intel-b70"
        else:
            from test_strata_dsv41 import fixture
            model = fixture(tmp_path)
            platform = "r9700"
        command = build_server_cmd(**common, platform_id=platform, model=model, directory=tmp_path, context_size=context, speculation="baseline")
    elif backend == "maya":
        from test_maya import fixture_model, load_models
        from ai_toolbox_cockpit.backends.maya.server_runner import build_server_cmd
        model = fixture_model(load_models()[0], tmp_path)
        command = build_server_cmd(**common, platform_id="strix-halo", model=model, directory=tmp_path, context_size=context)
    elif backend == "gufo":
        from ai_toolbox_cockpit.backends.gufo.server_runner import build_server_cmd
        target = tmp_path / "model.gguf"
        target.touch()
        model = {"id": "fixture", "served_model_name": "fixture", "context_size": 8192}
        resolved = {"targets": {target.name: target}, "model": target, "sidecar": None}
        with patch("ai_toolbox_cockpit.backends.gufo.server_runner.get_model", return_value=model), patch("ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files", return_value=resolved):
            command = build_server_cmd(**common, platform_id="strix-halo", model_id="fixture", context_size=context)
    elif backend == "halogen":
        from test_halogen import small_bundle
        from ai_toolbox_cockpit.backends.halogen.runner import build_server_cmd
        bundle = small_bundle(tmp_path)
        with patch("ai_toolbox_cockpit.backends.halogen.runner.get_bundle", return_value=bundle):
            command = build_server_cmd(**common, platform_id="strix-halo", models_dir=tmp_path, bundle_id=bundle["id"], context_size=context, kv_pool_positions=context)
        assert f"HALOGEN_CTX={context}" in command
        return
    elif backend == "r9v":
        from test_r9v import fixture, options
        from ai_toolbox_cockpit.backends.r9v.runner import build_server_cmd
        package = fixture(tmp_path)
        with patch("ai_toolbox_cockpit.backends.r9v.runner.get_package", return_value=package):
            command = build_server_cmd(**options(tmp_path, package), values={"context": str(context)})
        assert f"R9V_MAX_MODEL_LEN={context}" in command
        return
    elif backend == "ds4":
        from test_ds4_commands import Ds4CommandTests
        command = Ds4CommandTests().build(str(tmp_path), ctx=context)
        flag = "--ctx"
    elif backend == "vllm":
        from test_vllm_commands import VllmCommandTests
        VllmCommandTests.setUpClass()
        command = VllmCommandTests().build("openai/gpt-oss-20b", max_model_len=str(context))
        flag = "--max-model-len"
    else:
        from test_llama_embeddings import PATH, ROOT
        from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd
        with patch("ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir", return_value=Path(ROOT)):
            command = build_server_cmd(**common, model_path=PATH if backend == "llama_embedding" else f"{ROOT}/chat.gguf", context_size=context, use_fa=True, use_no_mmap=True, custom_args="", batch_size=context, ubatch_size=context)
        flag = "-c"
    assert command[command.index(flag) + 1] == str(context)


@pytest.mark.parametrize("backend", ["radiance", "affinity", "hipfire"])
def test_catalogue_default_is_not_a_context_ceiling(backend):
    from ai_toolbox_cockpit.catalog import load_model_catalog
    from importlib import import_module
    module = import_module(f"ai_toolbox_cockpit.backends.{backend}.model_manager")
    cls = getattr(module, {"radiance": "RadianceModel", "affinity": "AffinityModel", "hipfire": "HipfireModel"}[backend])
    entry = load_model_catalog().backends[backend].entries[0]
    assert cls.from_entry(dict(entry, context_size=16777216)).context_size == 16777216


@pytest.mark.parametrize("saved_context", [None, "524288", "8192"])
def test_radiance_profile_default_and_user_override_survive_refresh(tmp_path, saved_context):
    from textual.app import App, ComposeResult
    from textual.widgets import Input
    from ai_toolbox_cockpit.catalog import load_toolbox_catalog
    from ai_toolbox_cockpit.backends.radiance.server import RadianceServerPanel
    from ai_toolbox_cockpit.widgets import SearchableSelect
    class TestApp(App):
        active_platform_id = "r9700"
        toolbox_catalog = load_toolbox_catalog()
        def compose(self) -> ComposeResult:
            yield RadianceServerPanel(id="panel")
    async def check():
        with patch.dict("os.environ", {"XDG_CONFIG_HOME": str(tmp_path)}), patch("ai_toolbox_cockpit.backends.radiance.server.detect_container_engines", return_value=[]):
            if saved_context:
                from ai_toolbox_cockpit.settings import save_backend_settings
                save_backend_settings("radiance", {"model": "radiance-qwen38-flash-iq4r-8267a673", "profile": "default", "context": saved_context})
            app = TestApp()
            async with app.run_test(size=(140, 65)) as pilot:
                select = app.query_one("#radiance-model", SearchableSelect)
                select.value = "radiance-qwen38-flash-iq4r-8267a673"
                await pilot.pause()
                field = app.query_one("#radiance-context", Input)
                assert field.value == (saved_context or "262144")
                field.value = "1000000"
                panel = app.query_one("#panel", RadianceServerPanel)
                panel.profile_changed()
                panel.refresh_model_inventory()
                await pilot.pause()
                assert field.value == "1000000"
                # Switching away and back must not discard an explicit edit.
                from ai_toolbox_cockpit.backends.radiance.model_manager import load_models
                select.value = load_models()[0].id
                await pilot.pause()
                select.value = "radiance-qwen38-flash-iq4r-8267a673"
                await pilot.pause()
                assert field.value == "1000000"
                panel.reset_context()
                assert field.value == "262144"
                app.save_screenshot("/tmp/radiance-context-override.svg")
    asyncio.run(check())
