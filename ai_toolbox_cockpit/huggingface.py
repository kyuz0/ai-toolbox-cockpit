"""Shared Hugging Face authentication and download environment helpers."""

from __future__ import annotations

import os

from ai_toolbox_cockpit.settings import get_setting, set_setting


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
    """Build an environment for an authenticated, high-performance HF download."""
    environment = os.environ.copy()
    effective_token = token.strip() or get_hf_token()
    if effective_token:
        environment["HF_TOKEN"] = effective_token
    environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    return environment
