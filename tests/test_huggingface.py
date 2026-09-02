import os
from unittest import TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.huggingface import (
    get_hf_token,
    huggingface_environment,
    save_hf_token,
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
