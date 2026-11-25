import os
from pathlib import Path

import pytest

from pair_vision.config import Settings
from pair_vision.rules.config import load_rules
from pair_vision.session.frames import raw_to_key

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-key-1"
NEXT_KEY = "test-key-2"
SECRET = "callback-secret"


@pytest.fixture
def rules():
    return load_rules(ROOT / "configs" / "rules.default.yaml")


@pytest.fixture
def settings():
    return Settings(
        api_keys=(API_KEY, NEXT_KEY),
        callback_url="http://backend.test/internal/vision/events",
        callback_secret=SECRET,
        rules_path=str(ROOT / "configs" / "rules.default.yaml"),
        state_interval_s=0.2,
        max_sessions=2,
    )


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {API_KEY}"}


def new_key() -> str:
    return raw_to_key(os.urandom(16))
