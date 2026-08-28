import sys
from unittest import TestCase
from unittest.mock import patch

from ai_toolbox_cockpit.main import cli_main
from ai_toolbox_cockpit.updates import RELAUNCH_AFTER_UPDATE


class MainTests(TestCase):
    def test_successful_application_update_relaunches_current_environment(self) -> None:
        with (
            patch("ai_toolbox_cockpit.main.AiToolboxCockpitApp") as app_class,
            patch("ai_toolbox_cockpit.main.os.execv") as execv,
        ):
            app_class.return_value.run.return_value = RELAUNCH_AFTER_UPDATE

            cli_main()

        execv.assert_called_once_with(
            sys.executable,
            [sys.executable, "-m", "ai_toolbox_cockpit.main"],
        )

    def test_normal_exit_does_not_relaunch(self) -> None:
        with (
            patch("ai_toolbox_cockpit.main.AiToolboxCockpitApp") as app_class,
            patch("ai_toolbox_cockpit.main.os.execv") as execv,
        ):
            app_class.return_value.run.return_value = None

            cli_main()

        execv.assert_not_called()
