"""Background jobs (training, validation, export) as separate processes.

Each job is a folder ``<data_dir>/jobs/<id>/`` holding ``config.json``,
``log.txt``, ``progress.json`` and, when done, ``summary.json``. Running in a
child process keeps the API responsive and frees all GPU memory at the end.
One job runs at a time: they would compete for the same GPU.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import app

REPO_ROOT = Path(app.__file__).resolve().parent.parent


class JobError(RuntimeError):
    pass


class JobManager:
    def __init__(self, root: Path, weights_dir: Path) -> None:
        self.root = root
        self.weights_dir = weights_dir
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def _dir(self, job_id: str) -> Path:
        path = self.root / Path(job_id).name
        if not (path / "config.json").is_file():
            raise LookupError(f"job {job_id!r} not found")
        return path

    def running(self) -> str | None:
        with self._lock:
            for job_id, proc in self._procs.items():
                if proc.poll() is None:
                    return job_id
        return None

    def start(self, config: dict, name: str | None = None) -> dict:
        if (busy := self.running()) is not None:
            raise JobError(f"job {busy} is still running; wait or cancel it")
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        path = self.root / job_id
        path.mkdir(parents=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": job_id,
            "name": name or config.get("name") or job_id,
            "kind": config.get("kind", "train"),
            "created": time.time(),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        (path / "config.json").write_text(json.dumps(config, indent=2))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
        env.setdefault("YOLO_CONFIG_DIR", str(self.root.parent / "ultralytics"))
        log = (path / "log.txt").open("wb")
        # cwd = weights dir: pretrained weights named by Ultralytics are downloaded there once.
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "training.run_job", "--job-dir", str(path.resolve())],
            cwd=self.weights_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with self._lock:
            self._procs[job_id] = proc
        threading.Thread(target=self._wait, args=(job_id, proc, log), daemon=True).start()
        return self.get(job_id)

    def _wait(self, job_id: str, proc: subprocess.Popen, log) -> None:
        code = proc.wait()
        log.close()
        path = self.root / job_id
        status = path / "status.json"
        if not status.exists():
            status.write_text(json.dumps({"returncode": code, "finished": time.time()}))

    def cancel(self, job_id: str) -> dict:
        path = self._dir(job_id)
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is None or proc.poll() is not None:
            raise JobError("job is not running")
        (path / "status.json").write_text(
            json.dumps({"returncode": -15, "cancelled": True, "finished": time.time()})
        )
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        return self.get(job_id)

    def _status(self, job_id: str, path: Path) -> str:
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            return "running"
        status = _read(path / "status.json") or {}
        if status.get("cancelled"):
            return "cancelled"
        summary = _read(path / "summary.json")
        if summary is not None:
            return "failed" if "error" in summary else "completed"
        if status:
            return "failed"
        return "interrupted"  # the API restarted while it ran

    def get(self, job_id: str) -> dict:
        path = self._dir(job_id)
        meta = _read(path / "meta.json") or {"id": job_id}
        return {
            **meta,
            "status": self._status(job_id, path),
            "config": _read(path / "config.json"),
            "progress": _read(path / "progress.json"),
            "summary": _read(path / "summary.json"),
        }

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        jobs = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if (d / "config.json").is_file():
                job = self.get(d.name)
                job.pop("config", None)
                jobs.append(job)
        return jobs

    def log(self, job_id: str, tail: int = 200) -> str:
        path = self._dir(job_id) / "log.txt"
        if not path.exists():
            return ""
        # Progress bars rewrite the line with \r: keep the last state of each line.
        lines = [line.split("\r")[-1] for line in path.read_text(errors="replace").splitlines()]
        return "\n".join(lines[-tail:])

    def results(self, job_id: str) -> dict[str, list]:
        """Per-epoch curves from Ultralytics' results.csv (losses, mAP, lr)."""
        path = self._dir(job_id) / "train" / "results.csv"
        if not path.exists():
            return {}
        with path.open() as f:
            rows = list(csv.DictReader(f))
        columns: dict[str, list] = {}
        for row in rows:
            for k, v in row.items():
                key = (k or "").strip()
                try:
                    value = float(v)
                except (TypeError, ValueError):
                    value = None
                columns.setdefault(key, []).append(value)
        return columns

    def artifact(self, job_id: str, name: str) -> Path:
        """A plot or file written by the run (results.png, confusion_matrix.png...)."""
        path = self._dir(job_id)
        for folder in ("train", "val"):
            candidate = path / folder / Path(name).name
            if candidate.is_file():
                return candidate
        raise LookupError(name)

    def artifacts(self, job_id: str) -> list[str]:
        path = self._dir(job_id)
        names = set()
        for folder in ("train", "val"):
            if (path / folder).is_dir():
                names.update(
                    p.name for p in (path / folder).iterdir() if p.suffix in {".png", ".jpg"}
                )
        return sorted(names)


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
