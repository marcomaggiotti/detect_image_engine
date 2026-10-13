from __future__ import annotations

import io

import pytest
from PIL import Image

from app.config import Settings
from app.detectors import BackendError, Detector, load_labels, resolve_backend
from app.imaging import InvalidImageError, decode_image
from tests.conftest import make_image
from tests.fakes import FakeDetector


def test_target_classes_from_env(monkeypatch):
    monkeypatch.setenv("DETECT_TARGET_CLASSES", "cat, dog ,person")
    assert Settings(_env_file=None).target_classes == ["cat", "dog", "person"]


def test_resolve_custom_backend():
    assert resolve_backend("tests.fakes:FakeDetector") is FakeDetector


@pytest.mark.parametrize("name", ["nope", "tests.fakes:NotADetector", "missing.module:Thing"])
def test_resolve_backend_errors(name):
    with pytest.raises(BackendError):
        resolve_backend(name)


def test_builtin_backends_are_detectors():
    for name in ("ultralytics", "torchvision"):
        try:
            assert issubclass(resolve_backend(name), Detector)
        except BackendError:
            pass  # optional dependency not installed


def test_load_labels(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("__background__\ncat\n\ndog\n")
    assert load_labels(path) == {0: "__background__", 1: "cat", 3: "dog"}


def test_decode_converts_to_rgb():
    buf = io.BytesIO()
    Image.new("L", (10, 20)).save(buf, format="PNG")
    image = decode_image(buf.getvalue(), max_pixels=1000)
    assert image.mode == "RGB" and image.size == (10, 20)


def test_decode_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    data = make_image(size=(40, 10), fmt="JPEG", exif=exif.tobytes())
    assert decode_image(data, max_pixels=10_000).size == (10, 40)


def test_decode_rejects_large_images():
    with pytest.raises(InvalidImageError, match="pixel limit"):
        decode_image(make_image(size=(100, 100)), max_pixels=9_999)


@pytest.mark.parametrize("data", [b"", b"garbage", make_image()[:40]])
def test_decode_rejects_bad_data(data):
    with pytest.raises(InvalidImageError):
        decode_image(data, max_pixels=10_000)
