import os
import sys

from .app import AiToolboxCockpitApp
from .updates import RELAUNCH_AFTER_UPDATE


def cli_main() -> None:
    result = AiToolboxCockpitApp().run()
    if result == RELAUNCH_AFTER_UPDATE:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "ai_toolbox_cockpit.main"],
        )


if __name__ == "__main__":
    cli_main()
