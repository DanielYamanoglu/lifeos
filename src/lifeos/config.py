"""Environment-driven settings. Nothing here has a secret default."""
from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-5-5"


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default
