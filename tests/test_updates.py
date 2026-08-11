import unittest
from unittest.mock import patch

from ai_toolbox_cockpit.updates import TAGS_URL, available_update, latest_version, version_key


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class UpdateTests(unittest.TestCase):
    def test_repository_and_version_sorting(self) -> None:
        self.assertIn("kyuz0/ai-toolbox-cockpit", TAGS_URL)
        self.assertGreater(version_key("2026.10.1"), version_key("2026.9.20"))

    def test_latest_tag_is_selected_without_real_network(self) -> None:
        payload = b'[{"name":"v2026.8.11.1"},{"name":"2026.8.12.2"}]'
        with patch("ai_toolbox_cockpit.updates.urllib.request.urlopen", return_value=_Response(payload)):
            self.assertEqual(latest_version(), "2026.8.12.2")

    def test_update_only_reports_newer_version(self) -> None:
        with patch("ai_toolbox_cockpit.updates.latest_version", return_value="2026.8.12.2"):
            self.assertEqual(available_update("2026.8.11.1"), "2026.8.12.2")
            self.assertIsNone(available_update("2026.8.12.2"))


if __name__ == "__main__":
    unittest.main()
