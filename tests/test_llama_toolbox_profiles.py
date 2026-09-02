import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Input, Static, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.llama_cpp.config import (
    get_effective_mtp_config,
    get_model_config,
    get_mtp_server_args,
)
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
    get_download_cmd,
    get_hf_quants,
)
from ai_toolbox_cockpit.backends.llama_cpp.models import get_download_sources
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.widgets import SearchableSelect


TOOLBOX_ID = "strix-halo-llama-rocm-10-0-qwen-3-8-flash-next"
SIDECAR_REPO = "drluoto/Qwen3.8-Flash-Next-MTP-GGUF"
SIDECAR_FILE = "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"


class LlamaToolboxProfileTests(unittest.TestCase):
    def test_auxiliary_mtp_repository_is_a_download_source(self) -> None:
        entries = load_model_catalog().backends["llama_cpp"].entries
        sources = get_download_sources(entries)
        sidecar = next(source for source in sources if source["repo"] == SIDECAR_REPO)

        self.assertEqual(sidecar["role"], "mtp")
        self.assertEqual(sidecar["recommended_filename"], SIDECAR_FILE)

        files = [
            "mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf",
            SIDECAR_FILE,
            "mtp-Qwen3.8-Flash-Next-bf16.gguf",
        ]
        with patch(
            "ai_toolbox_cockpit.backends.llama_cpp.model_manager.HfApi.list_repo_files",
            return_value=files,
        ):
            self.assertEqual(get_hf_quants(SIDECAR_REPO), sorted(files))

    def test_sidecar_download_targets_its_own_repository_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
            return_value=Path(temporary),
        ):
            command = get_download_cmd(SIDECAR_REPO, SIDECAR_FILE)

        self.assertEqual(command[command.index("download") + 1], SIDECAR_REPO)
        self.assertEqual(command[-1], SIDECAR_FILE)
        self.assertEqual(
            command[command.index("--local-dir") + 1],
            str(Path(temporary) / "Qwen3.8-Flash-Next-MTP-GGUF"),
        )

    def test_flash_next_toolbox_overlays_structured_mtp_recipe(self) -> None:
        toolbox = load_toolbox_catalog().toolboxes[TOOLBOX_ID]
        model = get_model_config(
            "/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
            "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
        )
        mtp = get_effective_mtp_config(model, toolbox)

        self.assertEqual(mtp["draft_models"], [SIDECAR_FILE])
        self.assertEqual(mtp["default_draft_n"], 3)
        self.assertEqual(
            get_mtp_server_args(mtp, "3", "1"),
            "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 -np 1 "
            "--spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24",
        )

    def test_llama_runner_accepts_direct_io_load_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            model = models_dir / "model.gguf"
            sidecar = models_dir / SIDECAR_FILE
            model.touch()
            sidecar.touch()
            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                command = build_server_cmd(
                    engine="podman",
                    image="docker.io/example/flash-next:latest",
                    model_path=str(model),
                    context_size=262144,
                    use_fa=True,
                    use_no_mmap=True,
                    custom_args="",
                    engine_args=[],
                    supports_load_mode=True,
                    load_mode="dio",
                    mtp_draft_model_path=str(sidecar),
                )

        self.assertEqual(command[command.index("--load-mode") + 1], "dio")
        self.assertEqual(
            command[command.index("--model-draft") + 1],
            f"/models/{SIDECAR_FILE}",
        )
        self.assertNotIn("--no-mmap", command)


class LlamaToolboxProfileUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_flash_next_selection_applies_pairing_and_warns_on_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            model_dir = models_dir / "Qwen3.8-Flash-Next-GGUF" / "UD-IQ4_XS"
            sidecar_dir = models_dir / "Qwen3.8-Flash-Next-MTP-GGUF"
            model_dir.mkdir(parents=True)
            sidecar_dir.mkdir()
            recommended_model = (
                model_dir
                / "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
            )
            other_quant = (
                model_dir
                / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf"
            )
            unrelated_model = models_dir / "unrelated.gguf"
            sidecar = sidecar_dir / SIDECAR_FILE
            for path in (recommended_model, other_quant, unrelated_model, sidecar):
                path.touch()
            local_models = [
                {"name": unrelated_model.name, "path": str(unrelated_model)},
                {"name": other_quant.name, "path": str(other_quant)},
                {"name": recommended_model.name, "path": str(recommended_model)},
            ]

            with (
                patch(
                    "ai_toolbox_cockpit.views.toolboxes.ToolboxesView.refresh_installed",
                    return_value=None,
                ),
                patch(
                    "ai_toolbox_cockpit.app.AiToolboxCockpitApp.check_application_update",
                    return_value=None,
                ),
                patch("ai_toolbox_cockpit.app.available_update", return_value=None),
                patch("ai_toolbox_cockpit.app.load_active_platform", return_value="strix-halo"),
                patch("ai_toolbox_cockpit.app.save_active_platform"),
                patch(
                    "ai_toolbox_cockpit.backends.llama_cpp.models.scan_local_models",
                    return_value=[],
                ),
                patch(
                    "ai_toolbox_cockpit.backends.llama_cpp.server.scan_local_models",
                    return_value=local_models,
                ),
                patch(
                    "ai_toolbox_cockpit.backends.llama_cpp.server.get_local_mtp_models",
                    return_value=[sidecar],
                ),
            ):
                app = AiToolboxCockpitApp()
                async with app.run_test(size=(180, 60)) as pilot:
                    app.query_one(TabbedContent).active = "tab-servers"
                    await pilot.pause()

                    image = app.query_one("#llama-image", SearchableSelect)
                    image.value = TOOLBOX_ID
                    await pilot.pause()

                    self.assertEqual(
                        app.query_one("#llama-model", SearchableSelect).value,
                        str(recommended_model),
                    )
                    self.assertEqual(
                        app.query_one("#llama-load-mode", SearchableSelect).value,
                        "dio",
                    )
                    self.assertEqual(app.query_one("#llama-context", Input).value, "262144")
                    self.assertEqual(app.query_one("#llama-parallel", Input).value, "1")
                    self.assertTrue(app.query_one("#llama-fa", Checkbox).value)
                    self.assertEqual(
                        app.query_one("#llama-load-mode-row", Horizontal).styles.display,
                        "block",
                    )
                    self.assertEqual(
                        app.query_one("#llama-no-mmap", Checkbox).styles.display,
                        "none",
                    )
                    self.assertEqual(
                        app.query_one("#llama-mtp-model", SearchableSelect).value,
                        str(sidecar),
                    )
                    self.assertEqual(app.query_one("#llama-mtp-draft", Input).value, "3")
                    args = app.query_one("#llama-extra-args", Input).value
                    self.assertIn("--spec-type draft-mtp,ngram-mod", args)
                    self.assertIn("--spec-ngram-mod-n-max 64", args)
                    self.assertNotIn("--spec-draft-device", args)

                    guidance = app.query_one(
                        "#llama-toolbox-guidance-message", Static
                    )
                    self.assertIn("Recommended model and quant selected", str(guidance.render()))
                    self.assertEqual(
                        app.query_one("#llama-toolbox-guidance", Vertical).styles.display,
                        "block",
                    )

                    app.query_one("#llama-model", SearchableSelect).value = str(other_quant)
                    await pilot.pause()
                    self.assertIn("WARNING:", str(guidance.render()))
                    self.assertIn("tested quant", str(guidance.render()))


if __name__ == "__main__":
    unittest.main()
