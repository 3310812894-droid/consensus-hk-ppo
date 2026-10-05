"""Select the paper's five expert examples by reversal quantiles and smallest ID."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

from .io import create_output_directory, read_csv, write_csv, write_json
from .metrics import strict_reversal_counts


def select_experts(rows: list[dict]) -> list[dict]:
    if len(rows) != 100:
        raise ValueError("The paper example contains exactly 100 experts")
    numbers = [int(re.search(r"\d+$", row["expert_id"]).group()) for row in rows]
    if set(numbers) != set(range(1, 101)):
        raise ValueError("Expert IDs must be unique and cover 1--100")
    initial = np.asarray([[float(row[f"initial_u{k}"]) for k in range(1, 6)] for row in rows])
    hk = np.asarray([[float(row[f"hk_final_u{k}"]) for k in range(1, 6)] for row in rows])
    ppo = np.asarray([[float(row[f"ppo_final_u{k}"]) for k in range(1, 6)] for row in rows])
    valid, hk_counts, hk_ties = strict_reversal_counts(initial, hk)
    _, ppo_counts, ppo_ties = strict_reversal_counts(initial, ppo)
    if np.any(valid != 10) or np.any(hk_counts != 0) or np.any(hk_ties) or np.any(ppo_ties):
        raise ValueError("The input does not meet the paper example's strict-order checks")
    if any(int(row["ppo_reversal_count"]) != int(count) for row, count in zip(rows, ppo_counts)):
        raise ValueError("Stored and recomputed reversal counts differ")
    quantiles = np.quantile(ppo_counts, [0, 0.25, 0.5, 0.75, 1.0])
    if np.any(quantiles != np.floor(quantiles)) or len(set(quantiles)) != 5:
        raise ValueError("Reversal quantiles are not five distinct observed integer counts")
    selected = []
    for basis, target in zip(("最小值", "Q1", "中位数", "Q3", "最大值"), quantiles):
        candidates = [(number, row) for number, row, count in zip(numbers, rows, ppo_counts)
                      if count == target]
        _, row = min(candidates, key=lambda item: item[0])
        selected.append(dict(selection_basis=basis, **row,
                             reversal_pairs_fraction=f"{int(target)}/10"))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("paper_results/section_342_experts.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = select_experts(read_csv(args.source))
    output = create_output_directory(args.output)
    write_csv(output / "typical_experts.csv", selected)
    write_json(output / "selection_rule.json", {
        "statistic": "Individual strict reversal count",
        "levels": ["minimum", "Q1", "median", "Q3", "maximum"],
        "tie_break": "Smallest expert number among exact count matches",
        "selected_experts": [row["expert_id"] for row in selected],
        "reversal_counts": [int(row["ppo_reversal_count"]) for row in selected],
    })
    print(output)


if __name__ == "__main__":
    main()
