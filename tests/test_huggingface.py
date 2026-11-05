import os
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.huggingface import (
    DownloadOptions,
    get_download_options,
    get_hf_token,
    huggingface_environment,
    huggingface_download_command,
    save_hf_token,
    save_download_options,
)


class HuggingFaceTokenTests(TestCase):
    def test_environment_token_takes_precedence_over_saved_token(self) -> None:
        with (
            patch.dict(os.environ, {"HF_TOKEN": " environment-token "}),
            patch(
                "ai_toolbox_cockpit.huggingface.get_setting",
                return_value="saved-token",
            ) as get_setting,
        ):
            self.assertEqual(get_hf_token(), "environment-token")

        get_setting.assert_not_called()

    def test_saved_token_is_used_when_environment_token_is_missing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ai_toolbox_cockpit.huggingface.get_setting",
                return_value=" saved-token ",
            ),
        ):
            self.assertEqual(get_hf_token(), "saved-token")

    def test_download_environment_uses_session_token_and_fast_xet(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            environment = huggingface_environment("session-token")

        self.assertEqual(environment["HF_TOKEN"], "session-token")
        self.assertEqual(environment["HF_XET_HIGH_PERFORMANCE"], "1")

    def test_remembered_token_is_saved_to_cockpit_settings(self) -> None:
        with patch(
            "ai_toolbox_cockpit.huggingface.set_setting",
            return_value=True,
        ) as set_setting:
            self.assertTrue(save_hf_token(" remembered-token "))

        set_setting.assert_called_once_with("hf_token", "remembered-token")


class DownloadOptionsTests(TestCase):
    def test_all_hf_command_builders_apply_the_saved_file_concurrency(self):
        directory = Path("/tmp/cockpit models")
        bundle = {"repo": "example/model", "revision": "pinned", "directory": "bundle",
                  "files": [{"path": "part-1.gguf", "size_bytes": 1}, {"path": "part-2.gguf", "size_bytes": 1}]}
        model = SimpleNamespace(repo=bundle["repo"], revision=bundle["revision"], directory="bundle",
                                filename="model.gguf", files=(("part-1.gguf", 1), ("part-2.gguf", 1)))
        managers = {name: import_module(f"ai_toolbox_cockpit.backends.{name}.model_manager")
                    for name in ("llama_cpp", "ds4", "gufo", "vllm", "halogen", "r9v", "radiance", "hipfire", "affinity", "maya", "strata")}
        dsv41 = import_module("ai_toolbox_cockpit.backends.strata.dsv41")
        builders = {
            "llama_cpp": lambda: [managers["llama_cpp"].get_download_cmd("example/model", "Q4", "pinned")],
            "ds4": lambda: [managers["ds4"].get_download_cmd("example/model", "model.gguf", "pinned")],
            "gufo": lambda: managers["gufo"].get_download_commands(bundle, directory),
            "vllm": lambda: [managers["vllm"].get_download_cmd(bundle | {"download": {"files": bundle["files"]}}, directory)],
            "halogen": lambda: [managers["halogen"].get_download_cmd(bundle, directory)],
            "r9v": lambda: [managers["r9v"].get_download_cmd(bundle, directory)],
            "radiance": lambda: [managers["radiance"].download_command(model, directory)],
            "hipfire": lambda: [managers["hipfire"].download_command(model, directory)],
            "affinity": lambda: [managers["affinity"].hf_command("example/model", "pinned", ["part-1.gguf", "part-2.gguf"], directory)],
            "maya": lambda: [managers["maya"].download_command(model, directory)],
            "strata": lambda: [managers["strata"].download_command(model, directory)],
            "strata_dsv41": lambda: [dsv41.hf_command("example/model", "pinned", model.files, directory)],
        }
        for backend, build in builders.items():
            with self.subTest(backend=backend):
                self.assertTrue(save_download_options(DownloadOptions()))
                normal = build()
                self.assertTrue(all("--max-workers" not in command for command in normal))
                self.assertTrue(save_download_options(DownloadOptions(single_file=True)))
                self.assertEqual(build(), [command + ["--max-workers", "1"] for command in normal])

    def test_single_file_is_opt_in_for_existing_saved_settings(self):
        with patch("ai_toolbox_cockpit.huggingface.get_setting", return_value={"disable_xet": True, "timeout_seconds": 300}):
            self.assertEqual(get_download_options(), DownloadOptions(True, 300))

    def test_serial_download_setting_round_trips_and_can_be_disabled(self):
        command = ["hf", "download", "example/model", "first.gguf", "second.gguf"]
        self.assertTrue(save_download_options(DownloadOptions(single_file=True)))
        self.assertTrue(get_download_options().single_file)
        self.assertEqual(huggingface_download_command(command), [*command, "--max-workers", "1"])
        self.assertNotIn("--max-workers", command)
        self.assertTrue(save_download_options(DownloadOptions()))
        self.assertEqual(huggingface_download_command(command), command)

    def test_unsaved_options_reflect_and_preserve_shell_settings(self):
        with patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "180"}):
            self.assertEqual(get_download_options(), DownloadOptions(True, 180))
            environment = huggingface_environment()
        self.assertEqual(environment["HF_HUB_DISABLE_XET"], "1")
        self.assertEqual(environment["HF_HUB_DOWNLOAD_TIMEOUT"], "180")

    def test_saved_options_override_shell_and_preserve_unrelated_environment(self):
        self.assertTrue(save_download_options(DownloadOptions(True, 300)))
        with patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0", "HF_HUB_DOWNLOAD_TIMEOUT": "10", "OTHER_SETTING": "keep"}):
            environment = huggingface_environment("session-token")
        self.assertEqual(environment["HF_HUB_DISABLE_XET"], "1")
        self.assertEqual(environment["HF_HUB_DOWNLOAD_TIMEOUT"], "300")
        self.assertEqual(environment["OTHER_SETTING"], "keep")
        self.assertEqual(environment["HF_TOKEN"], "session-token")

    def test_unchecking_restores_xet_and_library_timeout_even_with_shell_overrides(self):
        self.assertTrue(save_download_options(DownloadOptions()))
        with patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "300"}):
            environment = huggingface_environment()
        self.assertEqual(environment["HF_HUB_DISABLE_XET"], "0")
        self.assertNotIn("HF_HUB_DOWNLOAD_TIMEOUT", environment)
        self.assertEqual(environment["HF_XET_HIGH_PERFORMANCE"], "1")

    def test_invalid_saved_timeout_is_not_passed_to_hf(self):
        for invalid in (0, -10, "abc", True):
            with self.subTest(invalid=invalid), patch("ai_toolbox_cockpit.huggingface.get_setting", return_value={"timeout_seconds": invalid}):
                self.assertIsNone(get_download_options().timeout_seconds)

    def test_invalid_shell_timeout_does_not_crash_options_dialog(self):
        with patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0", "HF_HUB_DOWNLOAD_TIMEOUT": "bad"}):
            self.assertEqual(get_download_options(), DownloadOptions())
