"""Inkling downloads, launch settings and user overrides; no engine execution."""
import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import App
from textual.widgets import Input

from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.backends.llama_cpp.config import get_model_config, get_toolbox_defaults, get_inference_profiles, get_dspark_config
from ai_toolbox_cockpit.backends.llama_cpp.models import get_download_sources
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import get_download_cmd
from ai_toolbox_cockpit.backends.gufo.model_manager import get_model, get_download_commands
from ai_toolbox_cockpit.backends.gufo.server_runner import build_server_cmd
from ai_toolbox_cockpit.backends.gufo.server import GufoServerPanel
from ai_toolbox_cockpit.settings import save_backend_settings
from ai_toolbox_cockpit.widgets import SearchableSelect

REPO = 'unsloth/Inkling-2-Mini-GGUF'
REV = '8078d149286eec7193795413b9e0643da6d11fa1'
TARGET = 'Inkling-2-Mini-UD-Q4_K_XL.gguf'
Q8 = 'Inkling-2-Mini-UD-Q8_K_XL.gguf'
DRAFT = 'dspark-Inkling-2-Mini-Q8_0.gguf'
GUFO_ID = 'gufo-inkling-2-mini-ud-q4-k-xl'


def test_only_current_mainline_quants_are_offered_and_downloads_are_pinned(tmp_path):
    models = load_model_catalog().backends['llama_cpp'].entries
    entry = next(m for m in models if m['repo'] == REPO)
    source = next(s for s in get_download_sources(models) if s['repo'] == REPO)
    assert source['revision'] == REV
    assert source['allowed_patterns'] == [TARGET, Q8]
    assert DRAFT not in source['allowed_patterns']
    for name in (TARGET, Q8):
        command = get_download_cmd(REPO, name, REV)
        assert command[command.index('--revision') + 1] == REV
        assert name in command
        assert get_model_config(str(tmp_path / 'Inkling-2-Mini-GGUF' / name))['id'] == entry['id']
    defaults = get_toolbox_defaults(entry, 'strix-halo-llama-rocm-10-0')
    assert defaults['context_size'] in {262144, 1048576}
    assert (defaults['batch_size'], defaults['ubatch_size'], defaults['parallel_sequences']) == (2048, 512, 1)
    assert defaults['load_mode'] == 'none'
    args = shlex.split(get_inference_profiles(entry)['Thinking (Effort: High)']['args'])
    for flag, value in [('--temp', '1'), ('--top-p', '1'), ('--top-k', '0'), ('--min-p', '0')]:
        assert args[args.index(flag) + 1] == value
    assert get_dspark_config(entry)["default_draft_n"] == 3


def test_gufo_downloads_exact_target_and_matching_drafter(tmp_path):
    entry = get_model(GUFO_ID)
    commands = get_download_commands(entry, tmp_path)
    assert len(commands) == 1
    assert commands[0][commands[0].index('--revision') + 1] == REV
    assert TARGET in commands[0] and DRAFT in commands[0]
    assert Q8 not in commands[0]
    assert entry['files'][0]['size_bytes'] == 17629760736
    assert entry['speculation']['size_bytes'] == 503889952
    assert entry['context_size'] == 1048576


@pytest.mark.parametrize('mode', ['baseline', 'dspark'])
@pytest.mark.parametrize('context', [None, 16777216])
def test_gufo_launch_honors_method_unlimited_output_and_explicit_context(tmp_path, mode, context):
    entry = get_model(GUFO_ID)
    target = tmp_path / TARGET
    draft = tmp_path / DRAFT
    target.touch()
    draft.touch()
    files = {'targets': {TARGET: target}, 'model': target, 'sidecar': draft, 'vision': None}
    with patch('ai_toolbox_cockpit.backends.gufo.server_runner.resolved_files', return_value=files):
        cmd = build_server_cmd(engine='podman', image='example/gufo:test', engine_args=[], platform_id='strix-halo', model_id=GUFO_ID, speculation_mode=mode, context_size=context)
        capped = build_server_cmd(engine='podman', image='example/gufo:test', engine_args=[], platform_id='strix-halo', model_id=GUFO_ID, max_tokens=12345)
    assert cmd[cmd.index('--context') + 1] == str(context or 1048576)
    assert cmd[cmd.index('--max-tokens') + 1] == '-1'
    assert capped[capped.index('--max-tokens') + 1] == '12345'
    assert cmd[cmd.index('--reasoning-effort') + 1] == 'high'
    if mode == 'baseline':
        assert cmd[cmd.index('--speculative') + 1] == 'off'
        assert '--dspark-model' not in cmd
    else:
        assert cmd[cmd.index('--dspark-model') + 1].endswith('/' + DRAFT)
        assert '--draft-tokens' not in cmd  # Native policy is adaptive.


@pytest.mark.parametrize('saved_context', [None, '16777216'])
def test_gufo_ui_defaults_and_saved_overrides_survive_refresh(tmp_path, monkeypatch, saved_context):
    import asyncio
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    if saved_context:
        save_backend_settings('gufo', {'model_id': GUFO_ID, 'context': saved_context, 'max_tokens': '12345', 'speculation_mode': 'baseline'})
    class TestApp(App):
        active_platform_id = 'strix-halo'
        toolbox_catalog = load_toolbox_catalog()
        def compose(self):
            yield GufoServerPanel(id='gufo-test')
    async def check():
        with patch('ai_toolbox_cockpit.backends.gufo.server.detect_container_engines', return_value=[SimpleNamespace(value='podman')]), patch('ai_toolbox_cockpit.backends.gufo.server.resolved_files', return_value={'sidecar': tmp_path / DRAFT, 'vision': None}):
            app = TestApp()
            async with app.run_test(size=(120, 55)) as pilot:
                app.query_one('#gufo-model', SearchableSelect).value = GUFO_ID
                await pilot.pause()
                assert app.query_one('#gufo-context', Input).value == (saved_context or '1048576')
                assert app.query_one('#gufo-max-tokens', Input).value == ('12345' if saved_context else '-1')
                assert app.query_one('#gufo-speculation', SearchableSelect).value == ('baseline' if saved_context else 'dspark')
                assert not app.query_one('#gufo-draft-row').display
                if not saved_context and os.environ.get('COCKPIT_SCREENSHOTS_DIR'):
                    output = Path(os.environ['COCKPIT_SCREENSHOTS_DIR'])
                    output.mkdir(parents=True, exist_ok=True)
                    (output / 'gufo-inkling.svg').write_text(app.export_screenshot())
                app.query_one('#gufo-context', Input).value = '16777216'
                app.query_one('#gufo-max-tokens', Input).value = '4321'
                mode = 'dspark' if saved_context else 'baseline'
                app.query_one('#gufo-speculation', SearchableSelect).value = mode
                await pilot.pause()
                app.query_one('#gufo-test', GufoServerPanel).refresh_model_inventory()
                await pilot.pause()
                assert app.query_one('#gufo-context', Input).value == '16777216'
                assert app.query_one('#gufo-max-tokens', Input).value == '4321'
                assert app.query_one('#gufo-speculation', SearchableSelect).value == mode
    asyncio.run(check())
