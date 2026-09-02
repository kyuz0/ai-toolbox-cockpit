import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_toolbox_cockpit.storage import (
    DiskSpace,
    disk_space_for_path,
    disk_space_text,
    download_space_note,
    format_bytes,
)


class StorageTests(unittest.TestCase):
    def test_nonexistent_destination_uses_nearest_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "models" / "nested"
            usage = SimpleNamespace(total=1_000_000, used=600_000, free=400_000)
            with patch("ai_toolbox_cockpit.storage.shutil.disk_usage", return_value=usage) as disk_usage:
                self.assertEqual(
                    disk_space_for_path(destination),
                    DiskSpace(total=1_000_000, used=600_000, free=400_000),
                )

            disk_usage.assert_called_once_with(root)

    def test_capacity_readout_uses_decimal_units(self) -> None:
        with patch(
            "ai_toolbox_cockpit.storage.disk_space_for_path",
            return_value=DiskSpace(
                total=2_000_000_000_000,
                used=1_250_000_000_000,
                free=750_000_000_000,
            ),
        ):
            self.assertEqual(
                disk_space_text("/models"),
                "Available space: 750.0 GB free of 2.0 TB on the destination filesystem",
            )

        self.assertEqual(format_bytes(1_500_000_000), "1.5 GB")

    def test_download_note_warns_when_required_space_exceeds_free_space(self) -> None:
        note = download_space_note(120_000_000_000, 80_000_000_000)

        self.assertIn("Estimated download size: 120.0 GB", note)
        self.assertIn("available space: 80.0 GB", note)
        self.assertIn("WARNING", note)

    def test_download_note_has_no_warning_when_download_fits(self) -> None:
        note = download_space_note(20_000_000_000, 80_000_000_000)

        self.assertNotIn("WARNING", note)


if __name__ == "__main__":
    unittest.main()
