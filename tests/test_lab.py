"""The lab API: runtime settings, datasets, annotation, splits, robustness, jobs."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.lab.options import nest, options
from app.main import create_app
from tests.conftest import make_image
from tests.fakes import FakeDetector
from training.customize import LossCustomization, configure_criterion
from training.run_job import base_weights, train_arguments
from training.splits import build_split, repeat_factors


@pytest.fixture
def lab_client(tmp_path):
    settings = Settings(
        _env_file=None, model_weights="fake.pt", warmup=False, data_dir=str(tmp_path / "data")
    )
    with TestClient(create_app(settings, detector=FakeDetector())) as c:
        yield c


def make_dataset(client, name="ballet", classes=("person", "cat"), n=6):
    assert (
        client.post("/v1/lab/datasets", json={"name": name, "classes": list(classes)}).status_code
        == 201
    )
    files = [("files", (f"img{i}.png", make_image((80, 60)), "image/png")) for i in range(n)]
    saved = client.post(f"/v1/lab/datasets/{name}/images", files=files).json()["saved"]
    assert len(saved) == n
    return saved


def test_info_and_ui(lab_client):
    info = lab_client.get("/v1/info").json()
    assert info["name"] == "yolo" and "lab" in info["capabilities"]
    assert lab_client.get("/ui/").status_code == 200


def test_runtime_settings_change_live(lab_client):
    body = lab_client.put(
        "/v1/lab/runtime",
        json={
            "iou_threshold": 0.6,
            "image_size": 320,
            "confidence_threshold": 0.1,
            "default_classes": ["person", "dog"],
        },
    ).json()
    assert body["iou_threshold"] == 0.6 and body["image_size"] == 320
    assert body["default_classes"] == ["person", "dog"]
    detected = lab_client.post(
        "/v1/detect", files={"image": ("a.png", make_image(), "image/png")}
    ).json()
    assert detected["counts"] == {"person": 1, "dog": 1}
    assert lab_client.put("/v1/lab/runtime", json={"iou_threshold": 3}).status_code == 422
    assert (
        lab_client.put("/v1/lab/runtime", json={"default_classes": ["unicorn"]}).status_code == 400
    )


def test_dataset_annotation_and_stats(lab_client):
    saved = make_dataset(lab_client)
    lab_client.put(
        f"/v1/lab/datasets/ballet/labels/{saved[0]}",
        json={"boxes": [{"cls": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.4}]},
    )
    lab_client.put(f"/v1/lab/datasets/ballet/labels/{saved[1]}", json={"boxes": []})
    labels = lab_client.get(f"/v1/lab/datasets/ballet/labels/{saved[0]}").json()
    assert labels["annotated"] and labels["boxes"][0]["cls"] == 0
    assert not lab_client.get(f"/v1/lab/datasets/ballet/labels/{saved[2]}").json()["annotated"]
    bad = lab_client.put(
        f"/v1/lab/datasets/ballet/labels/{saved[0]}",
        json={"boxes": [{"cls": 9, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.4}]},
    )
    assert bad.status_code == 400

    stats = lab_client.get("/v1/lab/datasets/ballet/stats").json()
    assert stats["annotated"] == 2 and stats["background"] == 1 and stats["unannotated"] == 4
    assert stats["boxes"] == {"person": 1, "cat": 0}
    assert lab_client.get(f"/v1/lab/datasets/ballet/images/{saved[0]}?thumb=1").status_code == 200

    listing = lab_client.get("/v1/lab/datasets").json()
    assert listing[0]["images"] == 6 and listing[0]["annotated"] == 2


def test_autolabel_uses_the_active_model(lab_client):
    make_dataset(lab_client, n=2)
    result = lab_client.post("/v1/lab/datasets/ballet/autolabel", json={"confidence": 0.3}).json()
    assert result == {"labelled": 2, "classes": ["person", "cat"]}
    stats = lab_client.get("/v1/lab/datasets/ballet/stats").json()
    # FakeDetector: one person and two cats (one below 0.3 is dropped: 0.41 kept)
    assert stats["boxes"] == {"person": 2, "cat": 4}


def test_zip_round_trip(lab_client):
    make_dataset(lab_client, n=2)
    lab_client.post("/v1/lab/datasets/ballet/autolabel", json={})
    data = lab_client.get("/v1/lab/datasets/ballet/export").content
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert "classes.txt" in zf.namelist()
    lab_client.post("/v1/lab/datasets", json={"name": "copy", "classes": ["person", "cat"]})
    imported = lab_client.post(
        "/v1/lab/datasets/copy/import", files={"archive": ("d.zip", data, "application/zip")}
    ).json()
    assert imported == {"images": 2, "labelled": 2}


def test_split_and_repeat_factor_sampling(lab_client, tmp_path):
    saved = make_dataset(lab_client, n=10)
    for i, name in enumerate(saved):
        cls = 1 if i == 0 else 0  # one rare "cat" image
        lab_client.put(
            f"/v1/lab/datasets/ballet/labels/{name}",
            json={"boxes": [{"cls": cls, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}]},
        )
    dataset = tmp_path / "data" / "datasets" / "ballet"
    plain = build_split(dataset, tmp_path / "s1", val_fraction=0.2, seed=1)
    assert plain["train_images"] == 8 and plain["val_images"] == 2
    balanced = build_split(
        dataset,
        tmp_path / "s2",
        val_fraction=0.2,
        seed=1,
        rebalance="repeat-factor",
        rfs_threshold=0.5,
    )
    assert balanced["train_entries"] >= balanced["train_images"]
    assert "names:" in (tmp_path / "s2" / "dataset.yaml").read_text()
    assert repeat_factors([{0}, {0}, {0}, {1}], threshold=1.0, max_repeat=4) == [1, 1, 1, 2]


def test_robustness_and_benchmark(lab_client):
    res = lab_client.post(
        "/v1/lab/robustness",
        files={"image": ("a.png", make_image((120, 90)), "image/png")},
        data={
            "perturbations": "darken,hflip,rotation,motion_blur",
            "severities": "1,5",
            "classes": "cat,person",
        },
    ).json()
    assert len(res["results"]) == 8
    assert res["results"][0]["preview"].startswith("data:image/jpeg")
    # The fake detector ignores the image: unchanged boxes are kept on photometric changes.
    darken = next(r for r in res["results"] if r["perturbation"] == "darken")
    assert darken["kept"] == 1.0
    bench = lab_client.post(
        "/v1/lab/benchmark", data={"sizes": "320,640", "runs": 3, "half": "false"}
    ).json()
    assert [r["image_size"] for r in bench["results"]] == [320, 640]
    assert bench["results"][0]["fps"] > 0


def test_options_schema_and_nesting():
    opts = options()
    keys = {o["key"] for g in opts["groups"] for o in g["options"]}
    assert {"loss.box", "assigner.topk", "augment.mosaic", "imbalance.rebalance", "freeze"} <= keys
    assert nest({"loss.box": 1, "epochs": 3}) == {"loss": {"box": 1}, "epochs": 3}


def test_train_arguments_and_scratch_weights():
    cfg = {
        "epochs": 5,
        "freeze": 10,
        "loss": {"box": 9.0, "cls": 1.0, "focal_gamma": 2},
        "augment": {"preset": "ballet-studio", "mosaic": 0.2},
    }
    args = train_arguments(cfg)
    assert args["box"] == 9.0 and args["cls"] == 1.0 and "focal_gamma" not in args
    assert args["mosaic"] == 0.2 and args["flipud"] == 0.0 and args["freeze"] == 10
    assert base_weights({"base": "yolo11n.pt", "pretrained": False}) == "yolo11n.yaml"


def test_configure_criterion_sets_assigner_and_matcher():
    class Assigner:
        topk, alpha, beta = 10, 0.5, 6.0

    class Matcher:
        cost_gain = {"class": 2, "bbox": 5, "giou": 2}

    class Crit:
        assigner = Assigner()
        matcher = Matcher()

    crit = Crit()
    changed = configure_criterion(
        crit, LossCustomization(tal_topk=13, tal_beta=4.0, match_giou=3.0)
    )
    assert crit.assigner.topk == 13 and crit.assigner.beta == 4.0 and crit.assigner.alpha == 0.5
    assert crit.matcher.cost_gain["giou"] == 3.0
    assert len(changed) == 3


def test_jobs_run_in_a_subprocess_and_report_failure(lab_client, monkeypatch):
    make_dataset(lab_client, n=2)
    # "export" of a missing model fails fast in the child process (no GPU, maybe no ultralytics).
    job = lab_client.post(
        "/v1/lab/jobs", json={"kind": "export", "format": "onnx", "weights": "/nonexistent.pt"}
    )
    assert job.status_code == 400  # weights are resolved up front
    job = lab_client.post("/v1/lab/jobs", json={"kind": "train", "dataset": "ballet", "epochs": 1})
    assert job.status_code == 201
    job_id = job.json()["id"]
    for _ in range(200):
        status = lab_client.get(f"/v1/lab/jobs/{job_id}").json()
        if status["status"] not in {"running"}:
            break
        time.sleep(0.1)
    assert status["status"] == "failed"
    assert "error" in status["summary"]
    assert json.loads(json.dumps(lab_client.get("/v1/lab/jobs").json()))[0]["id"] == job_id
    assert lab_client.get(f"/v1/lab/jobs/{job_id}/log").status_code == 200
    assert lab_client.post("/v1/lab/jobs", json={"kind": "train"}).status_code == 422
