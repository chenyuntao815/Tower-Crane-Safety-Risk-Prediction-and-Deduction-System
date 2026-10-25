"""Read project-local .env settings without copying secrets into global state."""
import os
from pathlib import Path

from dotenv import dotenv_values

ENV_PATH = Path(__file__).resolve().parent / ".env"


def get_setting(name, default=None):
    # Explicit process environment wins, including an explicitly empty value.
    value = os.getenv(name)
    if value is None and ENV_PATH.is_file():
        value = dotenv_values(ENV_PATH, encoding="utf-8-sig").get(name)
    if value is None:
        return default
    return value
