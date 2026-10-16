"""training/: presets, threshold choice, dataset preparation (no ML framework needed)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from training import finetune, prepare
from training.presets import PRESETS, preset


def test_indoor_preset_never_flips_or_rotates_a_fixed_camera():
    p = preset("indoor-camera")
    assert p["flipud"] == 0 and p["degrees"] == 0 and p["perspective"] == 0
    assert p["hsv_v"] >= 0.5 and p["fliplr"] == 0.5 and p["close_mosaic"] > 0
    assert preset("default") == {}
    with pytest.raises(ValueError, match="unknown preset"):
        preset("underwater")


def test_finetune_passes_preset_and_patience_and_prints_a_threshold(monkeypatch, tmp_path, capsys):
    calls = {}

    class FakeYOLO:
        names = {0: "cat", 1: "person"}

        def __init__(self, weights):
            self.weights = weights
            self.trainer = SimpleNamespace(best=str(tmp_path / "best.pt"))

        def train(self, **kwargs):
            calls["train"] = kwargs

        def val(self, **kwargs):
            box = SimpleNamespace(
                map=0.5,
                map50=0.7,
                px=[0.1, 0.3, 0.5, 0.7],
                f1_curve=[[0.2, 0.6, 0.8, 0.4], [0.3, 0.7, 0.6, 0.2]],
            )
            return SimpleNamespace(box=box)

    fake = types.SimpleNamespace(YOLO=FakeYOLO, RTDETR=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    args = ["--data", "d.yaml", "--preset", "indoor-camera", "--patience", "7", "--name", "home-v1"]
    finetune.main(args)
    train = calls["train"]
    assert train["patience"] == 7 and train["flipud"] == 0.0 and train["hsv_v"] == 0.6
    out = capsys.readouterr().out
    assert "Best confidence threshold on the validation set (highest F1): 0.50" in out
    assert "DETECT_CONFIDENCE_THRESHOLD=0.50" in out and "DETECT_MODEL_VERSION=home-v1" in out


def test_best_confidence_without_curves():
    assert finetune.best_confidence(SimpleNamespace(box=SimpleNamespace())) is None


def test_split_is_by_whole_days_newest_for_validation():
    stems = [f"cam_202610{d:02d}T1{h}0000Z" for d in (5, 6, 7, 8, 9) for h in range(4)]
    train, val = prepare.split_by_day(stems, 0.2)
    assert {prepare.day_of(s) for s in val} == {"20261009"}
    assert not {prepare.day_of(s) for s in train} & {prepare.day_of(s) for s in val}
    one_day = [f"cam_20261005T1{h}0000Z" for h in range(10)]
    train, val = prepare.split_by_day(one_day, 0.2)
    assert val == one_day[-2:] and train == one_day[:-2]


def test_pseudo_labels_fill_in_only_what_is_missing():
    human = [(0, 0.5, 0.5, 0.2, 0.2)]  # a cat
    model = [
        (0, 0.51, 0.5, 0.2, 0.2),  # the same cat
        (1, 0.2, 0.2, 0.1, 0.3),
    ]  # a person the human did not label
    assert prepare.merge_pseudo(human, model) == human + [model[1]]


def make_source(root, days=(5, 6, 7), per_day=4, labelled=True):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("cat\nluna\n")
    for d in days:
        for h in range(per_day):
            stem = f"alphabot2_202610{d:02d}T1{h}0000Z"
            Image.new("RGB", (64, 48)).save(root / "images" / f"{stem}.jpg")
            text = "1 0.5 0.5 0.2 0.2\n" if labelled and h else ""  # h == 0: an empty frame
            (root / "labels" / f"{stem}.txt").write_text(text)


class FakeModel:
    names = {0: "person", 15: "cat"}

    def predict(self, source, conf, verbose):
        boxes = SimpleNamespace(cls=[0, 15], xywhn=[[0.2, 0.2, 0.1, 0.3], [0.7, 0.7, 0.2, 0.2]])
        return [SimpleNamespace(boxes=boxes)]


def test_prepare_writes_a_dataset_keeps_classes_and_warns(tmp_path):
    make_source(tmp_path / "src")
    keep = ["person", "cat"]
    report = prepare.prepare(tmp_path / "src", tmp_path / "out", 0.3, FakeModel(), keep)
    assert report["names"] == ["cat", "luna", "person"]
    assert report["train"] == 8 and report["val"] == 4  # the newest day
    yaml = (tmp_path / "out" / "dataset.yaml").read_text()
    assert "names:\n  0: cat\n  1: luna\n  2: person\n" in yaml and "train: images/train" in yaml
    labels = tmp_path / "out" / "labels" / "train"
    label = (labels / "alphabot2_20261005T110000Z.txt").read_text().splitlines()
    assert label[0].startswith("1 0.5")  # the human's luna
    assert label[1].startswith("2 0.2") and label[2].startswith("0 0.7")  # model: person, cat
    assert any("only 6 'luna' boxes in train" in w for w in report["warnings"])
    # The fake model sees someone in every frame: no empty frame is left.
    assert any("empty frames" in w for w in report["warnings"])
    assert len(list((tmp_path / "out" / "images" / "val").iterdir())) == 4


def test_prepare_without_auto_label_warns_about_missing_empty_frames(tmp_path):
    make_source(tmp_path / "src", days=(5,), per_day=5)
    for p in (tmp_path / "src" / "labels").iterdir():
        p.write_text("0 0.5 0.5 0.2 0.2\n")
    report = prepare.prepare(tmp_path / "src", tmp_path / "out")
    assert any("empty frames" in w for w in report["warnings"])
    assert any("one day" in w for w in report["warnings"])
    assert any("no 'luna' in validation" in w for w in report["warnings"])
    assert PRESETS  # imported for the CLI choices
