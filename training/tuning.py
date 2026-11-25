"""Hyperparameter search: several short trainings (trials) that differ in a few
hyperparameters, compared on one validation metric.

    {
      "kind": "tune",
      "dataset_dir": "...", "base": "yolo11n-pose.pt", ...   # fixed options, as for "train"
      "search": {
        "strategy": "grid",          # every combination, or "random": "trials" draws
        "parameters": [
          {"key": "lr0", "values": [0.001, 0.01]},                    # grid: listed values
          {"key": "augment.mosaic", "min": 0.0, "max": 1.0},          # random: a range
          {"key": "lr0", "min": 0.0001, "max": 0.1, "log": true}      # log scale for rates
        ],
        "trials": 8,                 # random: how many; grid: at most this many
        "trial_epochs": 15,          # short trainings: the ranking matters, not the score
        "objective": "pose.map50_95" # or box.map50_95, box.map50, pose.map50
      }
    }

Each trial trains in ``trials/<n>/`` with the fixed options plus its own values. The
best trial's weights become the job's weights (``job:<id>`` loads them).
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

MAX_TRIALS = 30
OBJECTIVES = ("pose.map50_95", "pose.map50", "box.map50_95", "box.map50")


def nest(flat: dict) -> dict:
    """{"augment.mosaic": 0.5} -> {"augment": {"mosaic": 0.5}}"""
    out: dict = {}
    for key, value in flat.items():
        *parents, last = key.split(".")
        target = out
        for part in parents:
            target = target.setdefault(part, {})
        target[last] = value
    return out


def merge(base: dict, overrides: dict) -> dict:
    """``base`` with ``overrides`` applied, nested dictionaries merged key by key."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _draw(parameter: dict, rng: random.Random):
    if parameter.get("values"):
        return rng.choice(parameter["values"])
    low, high = float(parameter["min"]), float(parameter["max"])
    if parameter.get("log") and low > 0:
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if parameter.get("integer"):
        return int(round(value))
    return float(f"{value:.4g}")


def plan_trials(search: dict, seed: int = 0) -> list[dict]:
    """The values of each trial, as flat option keys."""
    parameters = search.get("parameters") or []
    if not parameters:
        raise ValueError("choose at least one hyperparameter to search")
    limit = min(int(search.get("trials") or MAX_TRIALS), MAX_TRIALS)
    if search.get("strategy", "grid") == "grid":
        for parameter in parameters:
            if not parameter.get("values"):
                raise ValueError(f"grid search: list the values of {parameter['key']}")
        keys = [parameter["key"] for parameter in parameters]
        combinations = itertools.product(*(parameter["values"] for parameter in parameters))
        plan = [dict(zip(keys, values, strict=True)) for values in combinations]
        if len(plan) > limit:
            raise ValueError(
                f"the grid has {len(plan)} combinations: at most {limit} "
                "(raise trials or remove values)"
            )
        return plan
    rng = random.Random(seed)
    return [
        {parameter["key"]: _draw(parameter, rng) for parameter in parameters} for _ in range(limit)
    ]


def objective_value(metrics: dict, objective: str) -> float | None:
    part, _, name = objective.partition(".")
    value = (metrics.get(part) or {}).get(name)
    if value is None and part == "pose":  # a detection model has no pose metrics
        value = (metrics.get("box") or {}).get(name)
    return value


TrainFunction = Callable[[dict, Path, Callable[[str], None]], dict]


def run_tune(
    config: dict, job_dir: Path, log: Callable[[str], None], train: TrainFunction, write_json
) -> dict:
    search = config["search"]
    objective = search.get("objective") or "box.map50_95"
    plan = plan_trials(search, seed=int(config.get("seed") or 0))
    fixed = {key: value for key, value in config.items() if key not in ("kind", "search", "export")}
    fixed["epochs"] = int(search.get("trial_epochs") or fixed.get("epochs") or 15)
    log(
        f"Hyperparameter search: {len(plan)} trials of {fixed['epochs']} epochs, "
        f"objective {objective}"
    )

    trials: list[dict] = []
    progress = {"stage": "tuning", "objective": objective, "total": len(plan), "trials": trials}
    started = time.time()
    for index, values in enumerate(plan, start=1):
        trial = {"index": index, "params": values, "status": "running", "objective": None}
        trials.append(trial)
        progress.update(current=index, elapsed_s=round(time.time() - started, 1))
        write_json(job_dir / "progress.json", progress)
        log(f"--- trial {index}/{len(plan)}: {values}")
        trial_dir = job_dir / "trials" / str(index)
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_started = time.time()
        try:
            summary = train(merge(fixed, nest(values)), trial_dir, log)
            trial.update(
                status="completed",
                metrics=summary.get("metrics"),
                objective=objective_value(summary.get("metrics") or {}, objective),
                weights=summary.get("weights"),
            )
        except Exception as error:  # noqa: BLE001 - one failing trial must not stop the search
            trial.update(status="failed", error=f"{type(error).__name__}: {error}")
            log(f"trial {index} failed: {error}")
        trial["seconds"] = round(time.time() - trial_started, 1)
        write_json(job_dir / "progress.json", progress)

    scored = [trial for trial in trials if trial.get("objective") is not None]
    if not scored:
        raise RuntimeError("no trial finished: see the log")
    best = max(scored, key=lambda trial: trial["objective"])
    log(f"Best trial {best['index']}: {objective} = {best['objective']} with {best['params']}")
    return {
        "weights": best["weights"],
        "objective": objective,
        "best": best,
        "trials": trials,
        "fixed": {key: value for key, value in fixed.items() if key != "dataset_dir"},
        "metrics": best.get("metrics"),
        "tune_seconds": round(time.time() - started, 1),
    }
