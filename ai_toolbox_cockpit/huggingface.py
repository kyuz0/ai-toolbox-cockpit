"""Shared Hugging Face authentication and download environment helpers."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from time import monotonic
from typing import Literal

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import HfHubHTTPError

from ai_toolbox_cockpit.settings import get_setting, set_setting


_last_token_check: tuple[tuple[str, bytes], float, bool] | None = None


@dataclass(frozen=True)
class DownloadOptions:
    disable_xet: bool = False
    timeout_seconds: int | None = None
    single_file: bool = False


def get_download_options() -> DownloadOptions:
    """Load saved choices, falling back to shell settings until first saved."""
    saved = get_setting("hf_download_options")
    if isinstance(saved, dict):
        timeout = saved.get("timeout_seconds")
        return DownloadOptions(
            disable_xet=saved.get("disable_xet") is True,
            timeout_seconds=timeout if type(timeout) is int and timeout > 0 else None,
            single_file=saved.get("single_file") is True,
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


def huggingface_download_command(command: list[str]) -> list[str]:
    """Apply the saved file concurrency preference to an hf download command."""
    if get_download_options().single_file:
        return [*command, "--max-workers", "1"]
    return command


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


def _hub_rejects_token(token: str) -> bool:
    global _last_token_check
    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT") or None)
    identity = (api.endpoint, sha256(token.encode()).digest())
    cached = _last_token_check
    if cached and cached[0] == identity and monotonic() < cached[1]:
        return cached[2]
    try:
        api.whoami(token=token)
    except HfHubHTTPError as error:
        if error.response is None or error.response.status_code != 401:
            return False
        rejected = True
    except Exception:
        # Offline or unavailable validation should not block ordinary Hub requests.
        return False
    else:
        rejected = False
    _last_token_check = (identity, monotonic() + 300, rejected)
    return rejected


def usable_hf_token(token: str | None = None, *, warn: bool = False) -> str | Literal[False] | None:
    """Return a credential, False for a rejected token, or None if none is configured."""
    credential = get_hf_token() if token is None else token
    if not credential and os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() not in {"1", "ON", "YES", "TRUE"}:
        credential = get_token()
    if not credential:
        return None
    if _hub_rejects_token(credential):
        if warn:
            print(
                "Hugging Face rejected the configured token. Public downloads can continue. "
                "Set a valid HF_TOKEN and restart Cockpit for private or gated models.",
                file=sys.stderr,
            )
        return False
    return credential


def huggingface_environment(token: str = "") -> dict[str, str]:
    """Build a download environment with the saved network options."""
    environment = os.environ.copy()
    configured_token = token.strip() or get_hf_token()
    effective_token = usable_hf_token(configured_token, warn=True)
    if effective_token and configured_token:
        environment["HF_TOKEN"] = effective_token
    elif effective_token is False:
        environment.pop("HF_TOKEN", None)
        environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
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
