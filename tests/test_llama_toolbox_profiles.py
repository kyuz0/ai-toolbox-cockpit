import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Input, Static, TabbedContent

from ai_toolbox_cockpit.app import AiToolboxCockpitApp
from ai_toolbox_cockpit.backends.llama_cpp.config import (
    get_effective_mtp_config,
    get_model_config,
    get_mtp_server_args,
    get_recommended_server_defaults,
)
from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
    get_download_cmd,
    get_hf_quants,
    get_hf_quants_with_sizes,
)
from ai_toolbox_cockpit.backends.llama_cpp.models import get_download_sources
from ai_toolbox_cockpit.backends.llama_cpp.server_runner import build_server_cmd
from ai_toolbox_cockpit.catalog import load_model_catalog, load_toolbox_catalog
from ai_toolbox_cockpit.widgets import SearchableSelect


TOOLBOX_ID = "strix-halo-llama-rocm-10-0-qwen-3-8-flash-next"
SIDECAR_REPO = "drluoto/Qwen3.8-Flash-Next-MTP-GGUF"
SIDECAR_FILE = "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
ENGRAM_TOOLBOX_ID = "strix-halo-llama-rocm-10-0-engramhalo"
ENGRAM_SIDECAR_REPO = "EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF"


class LlamaToolboxProfileTests(unittest.TestCase):
    def test_hf_quant_sizes_sum_shards_and_quant_directories(self) -> None:
        files = [
            SimpleNamespace(
                path="model-Q4-00001-of-00002.gguf",
                size=10_000_000_000,
            ),
            SimpleNamespace(
                path="model-Q4-00002-of-00002.gguf",
                size=12_000_000_000,
            ),
            SimpleNamespace(path="Q5/model-00001-of-00002.gguf", size=20),
            SimpleNamespace(path="Q5/model-00002-of-00002.gguf", size=30),
            SimpleNamespace(path="Q5/metadata.json", size=5),
            SimpleNamespace(path="README.md", size=100),
        ]
        with patch(
            "ai_toolbox_cockpit.backends.llama_cpp.model_manager.HfApi.list_repo_tree",
            return_value=files,
        ) as list_repo_tree:
            quants, sizes = get_hf_quants_with_sizes("example/model")

        self.assertEqual(quants, ["Q5", "model-Q4-*-of-*.gguf"])
        self.assertEqual(sizes["model-Q4-*-of-*.gguf"], 22_000_000_000)
        self.assertEqual(sizes["Q5"], 55)
        list_repo_tree.assert_called_once_with(
            repo_id="example/model",
            repo_type="model",
            recursive=True,
        )

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

    def test_engramhalo_sidecar_lookup_is_restricted_to_its_repository(self) -> None:
        from ai_toolbox_cockpit.backends.llama_cpp.model_manager import (
            get_local_mtp_models,
        )

        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            old_repo = models_dir / "Qwen3.8-Flash-Next-MTP-GGUF"
            engram_repo = models_dir / "Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF"
            old_repo.mkdir()
            engram_repo.mkdir()
            old_sidecar = old_repo / SIDECAR_FILE
            engram_sidecar = engram_repo / SIDECAR_FILE
            old_sidecar.touch()
            engram_sidecar.touch()

            with patch(
                "ai_toolbox_cockpit.backends.llama_cpp.model_manager.get_models_dir",
                return_value=models_dir,
            ):
                matches = get_local_mtp_models(
                    [SIDECAR_FILE], ENGRAM_SIDECAR_REPO
                )

        self.assertEqual(matches, [engram_sidecar])

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
            "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 "
            "--spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24",
        )

    def test_engramhalo_download_and_tested_ssd_mtp_profile(self) -> None:
        entries = load_model_catalog().backends["llama_cpp"].entries
        sources = get_download_sources(entries)
        sidecar = next(
            source for source in sources if source["repo"] == ENGRAM_SIDECAR_REPO
        )
        self.assertEqual(sidecar["recommended_filename"], SIDECAR_FILE)

        toolbox_catalog = load_toolbox_catalog()
        toolbox = toolbox_catalog.toolboxes[ENGRAM_TOOLBOX_ID]
        model = get_model_config(
            "/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/"
            "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
        )
        defaults = get_recommended_server_defaults(toolbox, model)
        mtp = get_effective_mtp_config(model, toolbox)

        self.assertEqual(toolbox.runtime_profile, "amd-rocm-hipblaslt")
        self.assertIn(
            "ROCBLAS_USE_HIPBLASLT=1",
            toolbox_catalog.runtime_profiles[toolbox.runtime_profile].engine_args,
        )
        self.assertEqual(defaults["context_size"], 163840)
        self.assertEqual(defaults["batch_size"], 8192)
        self.assertEqual(defaults["ubatch_size"], 2048)
        self.assertEqual(defaults["kv_cache_type"], "q8_0")
        self.assertEqual(defaults["load_mode"], "mmap")
        self.assertEqual(
            defaults["extra_args"], "--lazy-mode on -t 4 --no-webui"
        )
        self.assertEqual(mtp["draft_models"], [SIDECAR_FILE])
        self.assertEqual(mtp["sidecar_repo"], ENGRAM_SIDECAR_REPO)
        self.assertEqual(mtp["default_draft_n"], 4)
        self.assertEqual(
            get_mtp_server_args(mtp, "4", "1"),
            "--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 "
            "--spec-draft-p-min 0.75",
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

    async def test_engramhalo_selection_applies_tested_ssd_profile_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            model_dir = models_dir / "Qwen3.8-Flash-Next-GGUF" / "UD-IQ3_XXS"
            sidecar_dir = models_dir / "Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF"
            model_dir.mkdir(parents=True)
            sidecar_dir.mkdir()
            recommended_model = (
                model_dir
                / "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
            )
            sidecar = sidecar_dir / SIDECAR_FILE
            recommended_model.touch()
            sidecar.touch()

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
                    return_value=[
                        {"name": recommended_model.name, "path": str(recommended_model)}
                    ],
                ),
                patch(
                    "ai_toolbox_cockpit.backends.llama_cpp.server.get_local_mtp_models",
                    return_value=[sidecar],
                ),
            ):
                app = AiToolboxCockpitApp()
                async with app.run_test(size=(180, 65)) as pilot:
                    app.query_one(TabbedContent).active = "tab-servers"
                    await pilot.pause()

                    app.query_one("#llama-image", SearchableSelect).value = ENGRAM_TOOLBOX_ID
                    await pilot.pause()

                    self.assertEqual(
                        app.query_one("#llama-model", SearchableSelect).value,
                        str(recommended_model),
                    )
                    self.assertEqual(
                        app.query_one("#llama-load-mode", SearchableSelect).value,
                        "mmap",
                    )
                    self.assertEqual(app.query_one("#llama-context", Input).value, "163840")
                    self.assertEqual(app.query_one("#llama-batch", Input).value, "8192")
                    self.assertEqual(app.query_one("#llama-ubatch", Input).value, "2048")
                    self.assertEqual(app.query_one("#llama-parallel", Input).value, "1")
                    self.assertEqual(
                        app.query_one("#llama-kv-type", SearchableSelect).value,
                        "q8_0",
                    )
                    self.assertTrue(app.query_one("#llama-kv-enabled", Checkbox).value)
                    self.assertEqual(
                        app.query_one("#llama-mtp-model", SearchableSelect).value,
                        str(sidecar),
                    )
                    self.assertEqual(app.query_one("#llama-mtp-draft", Input).value, "4")
                    args = app.query_one("#llama-extra-args", Input).value
                    self.assertIn("--lazy-mode on", args)
                    self.assertNotIn("--tensor-read-lazy", args)
                    self.assertIn("-t 4", args)
                    self.assertIn("--spec-type draft-mtp,ngram-mod", args)
                    self.assertIn("--spec-draft-p-min 0.75", args)
                    self.assertNotIn("--spec-ngram-mod-n-max", args)

                    guidance = str(
                        app.query_one(
                            "#llama-toolbox-guidance-message", Static
                        ).render()
                    )
                    self.assertIn("Recommended model and quant selected", guidance)
                    self.assertIn("SSD mode requires mmap", guidance)
                    self.assertIn("MTP is validated for one slot only", guidance)

    async def test_engramhalo_keeps_mtp_off_until_its_sidecar_is_downloaded(self) -> None:
        model = {
            "name": "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
            "path": (
                "/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/"
                "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
            ),
        }
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
                return_value=[model],
            ),
            patch(
                "ai_toolbox_cockpit.backends.llama_cpp.server.get_local_mtp_models",
                return_value=[],
            ),
        ):
            app = AiToolboxCockpitApp()
            async with app.run_test(size=(180, 55)) as pilot:
                app.query_one(TabbedContent).active = "tab-servers"
                await pilot.pause()
                app.query_one("#llama-image", SearchableSelect).value = ENGRAM_TOOLBOX_ID
                await pilot.pause()

                self.assertFalse(app.query_one("#llama-mtp-enabled", Checkbox).value)
                self.assertNotIn(
                    "--spec-type draft-mtp",
                    app.query_one("#llama-extra-args", Input).value,
                )
                guidance = str(
                    app.query_one("#llama-toolbox-guidance-message", Static).render()
                )
                self.assertIn("MTP sidecar missing", guidance)


if __name__ == "__main__":
    unittest.main()
