from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from tests.fakes import FakeDetector


def make_image(size=(80, 60), fmt="PNG", **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, model_weights="fake.pt", warmup=False)


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def client(settings, detector):
    with TestClient(create_app(settings, detector=detector)) as c:
        yield c


@pytest.fixture
def png() -> bytes:
    return make_image()
