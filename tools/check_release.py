"""Check release files, archive integrity, and exact manuscript scope."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".zip", ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".pickle", ".npy", ".npz", ".pem", ".key"}
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|/home/|/mnt/[a-z]/")
EXCLUDED_DIRS = {"runs", "outputs", ".venv", "venv", ".git", "__pycache__", ".pytest_cache"}
EXPECTED_ROWS = {
    "section_333_baselines.csv": 360, "section_35_ablation.csv": 60,
    "section_36_sensitivity.csv": 240, "section_37_scalability.csv": 30,
    "section_381_id_states.csv": 100, "section_382_ood_epsilon.csv": 40,
    "section_383_ood_distributions.csv": 480,
}


def included_files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in relative.parts):
            yield path


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    errors = []
    files = list(included_files())
    for path in files:
        relative = path.relative_to(ROOT)
        if path.name == ".env" or path.name.startswith(".env."):
            errors.append(f"Environment/secret file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Model or serialized file: {relative}")
        if path.suffix.lower() in {".py", ".json", ".csv", ".md", ".txt", ".toml"} and path != Path(__file__).resolve():
            text = path.read_text(encoding="utf-8-sig")
            if ABSOLUTE_PATH.search(text):
                errors.append(f"Local absolute path: {relative}")
            if path.suffix == ".py" and re.search(r"(?:from|import)\s+revision_code", text):
                errors.append(f"Original-project import: {relative}")
    provenance = json.loads((ROOT / "paper_results/provenance.json").read_text(encoding="utf-8"))
    for item in provenance["files"]:
        path = ROOT / "paper_results" / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["output_sha256"]:
            errors.append(f"Archived checksum mismatch: {item['file']}")
    for filename, count in EXPECTED_ROWS.items():
        if len(read_rows(ROOT / "paper_results" / filename)) != count:
            errors.append(f"Unexpected row count: {filename}")
    ablations = read_rows(ROOT / "paper_results/section_35_ablation.csv")
    if {row["model"] for row in ablations} != {"full_model", "no_mask", "no_gravity_penalty"}:
        errors.append("Ablation variants differ from the manuscript")
    baselines = read_rows(ROOT / "paper_results/section_333_baselines.csv")
    if {row["model"] for row in baselines} != {"full_model", "hk", "scod", "adaptive_confidence", "noise_hk", "inertial"}:
        errors.append("Baseline variants differ from the manuscript")
    distributions = read_rows(ROOT / "paper_results/section_383_ood_distributions.csv")
    if {row["distribution"] for row in distributions} != {"beta_center", "beta_extremes", "beta_high", "bimodal"}:
        errors.append("Distribution OOD variants differ from the manuscript")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Release check passed: {len(files)} files; no weights, absolute paths, or unrelated variants.")


if __name__ == "__main__":
    main()
