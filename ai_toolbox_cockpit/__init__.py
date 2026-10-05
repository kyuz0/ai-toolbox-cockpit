"""AI Toolbox Cockpit."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("ai-toolbox-cockpit")
except PackageNotFoundError:
    __version__ = "0.0.0+source"

