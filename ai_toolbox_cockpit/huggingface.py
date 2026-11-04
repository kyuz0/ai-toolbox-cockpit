"""Shared Hugging Face authentication and download environment helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from ai_toolbox_cockpit.settings import get_setting, set_setting


@dataclass(frozen=True)
class DownloadOptions:
    disable_xet: bool = False
    timeout_seconds: int | None = None


def get_download_options() -> DownloadOptions:
    """Load saved choices, falling back to shell settings until first saved."""
    saved = get_setting("hf_download_options")
    if isinstance(saved, dict):
        timeout = saved.get("timeout_seconds")
        return DownloadOptions(
            disable_xet=saved.get("disable_xet") is True,
            timeout_seconds=timeout if type(timeout) is int and timeout > 0 else None,
        )
    try:
        timeout = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", ""))
    except ValueError:
        timeout = 0
    return DownloadOptions(
        disable_xet=os.environ.get("HF_HUB_DISABLE_XET", "").upper() in {"1", "ON", "YES", "TRUE"},
        timeout_seconds=timeout if timeout > 0 else None,
    )


def save_download_options(options: DownloadOptions) -> bool:
    return set_setting("hf_download_options", asdict(options))


def get_hf_token() -> str:
    """Return the environment token first, then a token remembered by Cockpit."""
    environment_token = os.environ.get("HF_TOKEN", "").strip()
    if environment_token:
        return environment_token

    saved_token = get_setting("hf_token", "")
    return saved_token.strip() if isinstance(saved_token, str) else ""


def save_hf_token(token: str) -> bool:
    """Remember a Hugging Face token in the Cockpit configuration."""
    return set_setting("hf_token", token.strip())


def huggingface_environment(token: str = "") -> dict[str, str]:
    """Build an authenticated download environment with the saved network options."""
    environment = os.environ.copy()
    effective_token = token.strip() or get_hf_token()
    if effective_token:
        environment["HF_TOKEN"] = effective_token
    environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Without saved choices, preserve the existing shell-driven behaviour.
    if isinstance(get_setting("hf_download_options"), dict):
        options = get_download_options()
        environment["HF_HUB_DISABLE_XET"] = "1" if options.disable_xet else "0"
        if options.timeout_seconds is None:
            environment.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
        else:
            environment["HF_HUB_DOWNLOAD_TIMEOUT"] = str(options.timeout_seconds)
    return environment
