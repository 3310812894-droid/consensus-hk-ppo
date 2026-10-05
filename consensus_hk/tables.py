"""Summarize episode CSVs with sample standard deviations and explicit units."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .io import create_output_directory, read_csv, summarize, write_csv


def mean_std(row, field, digits):
    mean, std = row[f"{field}_mean"], row[f"{field}_std"]
    if mean == "":
        return "--"
    def display(value):
        rounded = Decimal(f"{value:.12f}").quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
        return f"{rounded:.{digits}f}"
    if std == "":
        return display(mean)
    return f"{display(mean)} ± {display(std)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(read_csv(args.source))
    output = create_output_directory(args.output)
    write_csv(output / "summary.csv", rows)
    columns = ("模型", "N", "M", "ε", "λ_rev", "联合成功数", "成功折合轮次",
               "终态V", "终态G", "OPRR (%)")
    rendered = []
    for row in rows:
        rendered.append((str(row["model"]), str(row["N"]), str(row["M"]),
                         str(row["epsilon"]), str(row["lambda_rev"]),
                         f"{row['joint_success_count']}/{row['episodes']}",
                         mean_std(row, "equivalent_rounds", 2),
                         mean_std(row, "final_variance", 6),
                         mean_std(row, "gravity_shift", 6),
                         mean_std(row, "final_oprr", 2)))
    note = ("连续指标为均值±样本标准差（ddof=1）。折合轮次仅统计联合成功回合，"
            "终态V、G和OPRR统计全部回合。单个成功回合不计算标准差；OPRR标准差单位为百分点。")
    with (output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rendered:
            handle.write("| " + " | ".join(row) + " |\n")
        handle.write("\n" + note + "\n")
    def tex(value):
        return value.replace("_", r"\_").replace("%", r"\%").replace("ε", r"$\epsilon$").replace(
            "λ\\_rev", r"$\lambda_{\mathrm{rev}}$").replace("±", r"$\pm$")
    with (output / "summary_table.tex").open("x", encoding="utf-8") as handle:
        handle.write("% Tabular fragment; requires booktabs.\n\\begin{tabular}{lccccccccc}\n\\toprule\n")
        handle.write(" & ".join(map(tex, columns)) + " \\\\\n\\midrule\n")
        for row in rendered:
            handle.write(" & ".join(map(tex, row)) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n% " + note + "\n")
    print(output)


if __name__ == "__main__":
    main()
