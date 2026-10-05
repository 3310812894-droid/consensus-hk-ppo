"""Portable result files and per-run provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Iterable

import numpy as np


def create_output_directory(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}. Choose a new directory.")
    path.mkdir(parents=True)
    return path


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty result table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.system()}
    for package in ("numpy", "gymnasium", "stable-baselines3", "torch",
                    "cloudpickle", "tensorboard", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    return versions


def is_success(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ("1", "true"):
        return True
    if str(value).lower() in ("0", "false"):
        return False
    raise ValueError(f"Invalid joint_success value: {value!r}")


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    keys = ("model", "N", "M", "epsilon", "distribution", "lambda_rev")
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in groups.items():
        successful = [row for row in group if is_success(row["joint_success"])]
        summary = dict(zip(keys, key))
        summary.update(episodes=len(group), joint_success_count=len(successful),
                       joint_success_rate=100.0 * len(successful) / len(group))
        for field in ("intervention_steps", "equivalent_rounds", "final_variance",
                      "gravity_shift", "final_oprr"):
            selected = successful if field in ("intervention_steps", "equivalent_rounds") else group
            values = np.asarray([float(row[field]) for row in selected])
            summary[f"{field}_mean"] = float(values.mean()) if values.size else ""
            summary[f"{field}_std"] = float(values.std(ddof=1)) if values.size > 1 else ""
            summary[f"{field}_min"] = float(values.min()) if values.size else ""
            summary[f"{field}_max"] = float(values.max()) if values.size else ""
        summaries.append(summary)
    return summaries

