import os
import sys

from .cli import run_from_argv
from .updates import RELAUNCH_AFTER_UPDATE


def _load_app_class():
    from .app import AiToolboxCockpitApp

    return AiToolboxCockpitApp


def _run_tui() -> int:
    result = _load_app_class()().run()
    if result == RELAUNCH_AFTER_UPDATE:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "ai_toolbox_cockpit.main"],
        )
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    result = run_from_argv(argv)
    return _run_tui() if result is None else result


if __name__ == "__main__":
    raise SystemExit(cli_main())
