"""Paper figure layouts from archived CSVs; no policy inference or training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from .io import create_output_directory, read_csv


EPSILONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
WEIGHTS = (0.5, 1.0, 1.5, 2.0)
STYLES = (
    dict(color="#587E9C", marker="o", linestyle="-"),
    dict(color="#B87573", marker="s", linestyle="--"),
    dict(color="#8B9270", marker="^", linestyle="-."),
)
CHINESE_FONT = None
WESTERN_FONT = None


def configure_fonts(font_file: Path | None = None):
    global CHINESE_FONT, WESTERN_FONT
    if font_file is not None:
        font_manager.fontManager.addfont(str(font_file))
        chinese = font_manager.FontProperties(fname=str(font_file)).get_name()
    else:
        available = {entry.name for entry in font_manager.fontManager.ttflist}
        candidates = ("SimSun", "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC",
                      "Noto Sans CJK SC", "Microsoft YaHei")
        chinese = next((name for name in candidates if name in available), None)
        if chinese is None:
            raise RuntimeError("Chinese font not found. Install a CJK font or pass --font /path/to/font.")
    western = "Times New Roman" if any(
        entry.name == "Times New Roman" for entry in font_manager.fontManager.ttflist
    ) else "DejaVu Serif"
    CHINESE_FONT, WESTERN_FONT = chinese, western
    plt.rcParams.update({
        "font.family": [western, chinese], "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "stix", "axes.linewidth": 0.65,
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "xtick.direction": "out", "ytick.direction": "out",
    })


def finalize_fonts(fig):
    """Keep CJK text out of the Western-only mathtext default font."""
    for text in fig.findobj(match=matplotlib.text.Text):
        family = CHINESE_FONT if any("\u4e00" <= char <= "\u9fff" for char in text.get_text()) else WESTERN_FONT
        text.set_fontfamily(family)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D2D6D9", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.6, length=3)


def plot_training(data: Path):
    with data.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    steps = np.cumsum([int(row["l"]) for row in records])
    groups = np.minimum((steps // 100_000).astype(int), 99)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85))
    fig.subplots_adjust(left=0.085, right=0.987, bottom=0.32, top=0.96, wspace=0.35)
    for ax, field, label, main_color, light_color, letter in zip(
        axes, ("r", "l"), ("累积回报", "干预步数"),
        ("#C97E7A", "#7199B6"), ("#E5BCB9", "#B5CDDE"), ("a", "b"),
    ):
        values = np.asarray([float(row[field]) for row in records])
        occupied = np.unique(groups)
        counts = np.asarray([np.sum(groups == group) for group in occupied])
        means = np.asarray([values[groups == group].mean() for group in occupied])
        trend = np.asarray([np.average(means[max(0, j - 2):j + 1],
                                      weights=counts[max(0, j - 2):j + 1])
                            for j in range(len(occupied))])
        x = (occupied + 0.5) * 0.1
        ax.plot(x, means, color=light_color, linewidth=0.9, label="分段均值")
        ax.plot(x, trend, color=main_color, linewidth=1.65, label="滑动均值")
        ax.set_xlabel("训练交互步数（百万步）")
        ax.set_ylabel(label)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.text(0.5, -0.40, f"（{letter}）训练回合{label}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
        style_axis(ax)
        ax.legend(fontsize=8)
    return fig


def smooth_within_sweeps(steps: np.ndarray, values: np.ndarray):
    """Display-only seven-point average, preserving sweep boundaries and endpoints."""
    smoothed = values.copy()
    for group in np.unique(steps // 100):
        index = np.flatnonzero(steps // 100 == group)
        for j in range(1, len(index) - 1):
            smoothed[index[j]] = values[index[max(0, j - 3):min(len(index), j + 4)]].mean()
    return smoothed


def plot_variance(data: Path):
    rows = read_csv(data)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for epsilon, color, marker in zip((0.05, 0.15, 0.25),
                                      ("#E64B35", "#4DBBD5", "#00A087"), ("o", "s", "^")):
        group = sorted([row for row in rows if row["model"] in ("PPO-HK", "full_model")
                        and float(row["epsilon"]) == epsilon], key=lambda row: int(row["micro_step"]))
        if not group:
            raise ValueError(f"Missing PPO variance trace for epsilon={epsilon}")
        steps = np.asarray([int(row["micro_step"]) for row in group])
        y = np.asarray([float(row["variance"]) for row in group])
        ax.plot(steps / 100, y, color=color, alpha=0.20, linewidth=0.65)
        ax.plot(steps / 100, smooth_within_sweeps(steps, y), color=color,
                marker=marker, markevery=sorted(set(np.linspace(0, len(y) - 1, 5, dtype=int))),
                linewidth=1.65, markersize=4.3, label=rf"$\epsilon={epsilon:.2f}$")
    ax.axhline(0.05, color="#7F8C8D", linewidth=1.15, linestyle="--", label="软共识方差阈值")
    ax.set_xlabel("折合演化轮次")
    ax.set_ylabel(r"群体总方差 $V(t)$")
    ax.set_xlim(0, max(13.2, max(int(row["micro_step"]) / 100 for row in rows
                              if row["model"] in ("PPO-HK", "full_model")) * 1.03))
    ax.set_ylim(0, 0.46)
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=9)
    finalize_fonts(fig)
    fig.tight_layout()
    return fig


def plot_trajectories(data: Path):
    rows = read_csv(data)
    fig, axes = plt.subplots(5, 2, figsize=(7.1, 5.9), sharex="col", sharey=True)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.925, hspace=0.14, wspace=0.12)
    cmap = LinearSegmentedColormap.from_list("muted_blue", ("#BDD5E3", "#7FAAC6", "#356C92"))
    for col, method in enumerate(("HK", "PPO-HK")):
        aliases = ("HK", "hk") if col == 0 else ("PPO-HK", "full_model")
        selected = [row for row in rows if row["method"] in aliases]
        times = sorted({float(row["time"]) for row in selected})
        indexed = {(float(row["time"]), int(row["expert"])): row for row in selected}
        experts = sorted({int(row["expert"]) for row in selected})
        if len(experts) != 100 or len(indexed) != len(times) * 100:
            raise ValueError("Incomplete five-dimensional trajectory data")
        for dimension in range(5):
            ax = axes[dimension, col]
            for expert in experts:
                values = [float(indexed[time, expert][f"u{dimension + 1}"]) for time in times]
                ax.plot(times, values, color=cmap(values[0]), linewidth=0.45, alpha=0.50)
            ax.set_xlim(0, 200 if col == 0 else max(14, times[-1]))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 50, 100, 150, 200] if col == 0 else [0, 4, 8, 12])
            ax.tick_params(labelleft=col == 0, labelsize=8)
            style_axis(ax)
    for dimension in range(5):
        axes[dimension, 0].text(-0.245, 0.5, f"方案{dimension + 1}",
                                transform=axes[dimension, 0].transAxes, ha="center", va="center", fontsize=8.2)
    axes[0, 0].set_title("（a）经典HK模型", pad=11)
    axes[0, 1].set_title("（b）共识收敛模型", pad=11)
    axes[-1, 0].set_xlabel("HK同步更新轮次")
    axes[-1, 1].set_xlabel("折合演化轮次")
    fig.text(0.024, 0.5525, "意见值", rotation=90, ha="center", va="center")
    color_axis = fig.add_axes([0.42, 0.082, 0.30, 0.009])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=color_axis, orientation="horizontal")
    bar.set_ticks([0, 0.5, 1])
    bar.ax.tick_params(labelsize=7.5, length=1.8, width=0.35, pad=2)
    bar.set_label("初始意见值", fontsize=7.8, labelpad=3)
    bar.outline.set_linewidth(0.3)
    return fig


def plot_sensitivity(data: Path):
    rows = read_csv(data)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    styles = (dict(color="#1F3B5C", marker="o", linestyle="-"),
              dict(color="#5A7D6A", marker="^", linestyle="--"),
              dict(color="#A06A4A", marker="D", linestyle="-."))
    for ax, metric, label, title in zip(
        axes, ("final_variance", "final_oprr", "gravity_shift"),
        (r"终态总方差 $V_T$", "OPRR (%)", r"归一化重心偏移 $G_T$"),
        ("(a) 终态总方差", "(b) OPRR", "(c) 群体意见重心偏移"),
    ):
        for epsilon, style in zip((0.05, 0.15, 0.25), styles):
            means = []
            for weight in WEIGHTS:
                group = [row for row in rows if float(row["epsilon"]) == epsilon
                         and float(row["lambda_rev"]) == weight]
                if len(group) != 20:
                    raise ValueError("Sensitivity plot requires 20 episodes per weight and epsilon")
                means.append(np.mean([float(row[metric]) for row in group]))
            ax.plot(WEIGHTS, means, linewidth=1.35, markersize=4.5, markerfacecolor="white",
                    label=rf"$\epsilon={epsilon:.2f}$", **style)
        ax.axvline(1.0, color="#666666", linestyle=(0, (4, 2)), linewidth=0.9)
        ax.set_xticks(WEIGHTS)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel(r"序数偏好反转惩罚权重 $\lambda_{\mathrm{rev}}$")
        ax.set_ylabel(label)
        ax.set_title(title)
        style_axis(ax)
    axes[0].axhline(0.05, color="#8A8A8A", linestyle=(0, (2, 2)), linewidth=0.9)
    axes[0].set_ylim(0, 0.054)
    axes[1].set_ylim(30, 47)
    axes[2].axhline(0.45, color="#8A8A8A", linestyle=(0, (2, 2)), linewidth=0.9)
    axes[2].set_ylim(0, 0.48)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="#666666", linestyle=(0, (4, 2))),
                Line2D([], [], color="#8A8A8A", linestyle=(0, (2, 2)))]
    labels += ["基准参数", "约束阈值"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.30, wspace=0.48)
    return fig


def plot_scalability(data: Path):
    rows = read_csv(data)
    indexed = {(int(row["N"]), int(row["M"]), float(row["epsilon"])): row for row in rows}
    expected = {(n, m, epsilon) for n, m in ((50, 5), (100, 5), (200, 5), (100, 3), (100, 10))
                for epsilon in EPSILONS}
    if len(rows) != 30 or set(indexed) != expected:
        raise ValueError("Scalability plot requires exactly five scales and six epsilon values")
    if any(int(row["seed"]) != 42 for row in rows):
        raise ValueError("Scalability plot uses only evaluation seed 42")
    fig, axes = plt.subplots(2, 2, figsize=(17 / 2.54, 11.6 / 2.54))
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.105, top=0.885, wspace=0.32, hspace=0.64)
    for row_index, scales in enumerate((((50, 5), (100, 5), (200, 5)), ((100, 3), (100, 5), (100, 10)))):
        variable = "N" if row_index == 0 else "M"
        for column, metric in enumerate(("equivalent_rounds", "final_oprr")):
            ax = axes[row_index, column]
            for scale, style in zip(scales, STYLES):
                values = [float(indexed[(*scale, epsilon)][metric]) for epsilon in EPSILONS]
                number = scale[0] if row_index == 0 else scale[1]
                ax.plot(EPSILONS, values, label=rf"${variable}={number}$", linewidth=1.5,
                        markersize=5, markeredgewidth=0.9, markerfacecolor="white", **style)
            ax.set_xlim(0.04, 0.31)
            ax.set_xticks(EPSILONS)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_xlabel(r"信任阈值 $\epsilon$")
            ax.set_ylabel("折合演化轮次" if column == 0 else "OPRR (%)")
            ax.set_ylim((0, 16) if column == 0 else (30, 55))
            ax.set_yticks([0, 4, 8, 12, 16] if column == 0 else [30, 35, 40, 45, 50, 55])
            ax.text(0.015, 1.025, f"({'abcd'[2 * row_index + column]})", transform=ax.transAxes,
                    ha="left", va="bottom")
            style_axis(ax)
        left, right = axes[row_index]
        center = (left.get_position().x0 + right.get_position().x1) / 2
        handles, labels = left.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="center",
                   bbox_to_anchor=(center, left.get_position().y1 + 0.063))
    return fig


def plot_stability(data: Path):
    rows = sorted(read_csv(data), key=lambda row: int(row["seed"]))
    if len(rows) != 100 or {int(row["seed"]) for row in rows} != set(range(10000, 10100)):
        raise ValueError("Stability plot requires seeds 10000--10099")
    if any(float(row["epsilon"]) != 0.05 for row in rows):
        raise ValueError("Stability plot uses only epsilon=0.05")
    fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 7.1 / 2.54))
    fig.subplots_adjust(left=0.088, right=0.98, bottom=0.115, top=0.865, wspace=0.49)
    specs = (("intervention_steps", "(a) 成功干预步数", "干预步数", "#A8DADC", "#1D3557", (1125, 1325)),
             ("final_variance", "(b) 终态总方差", r"终态总方差 $V_T$", "#F4A261", "#E76F51", (0.041, 0.0505)),
             ("final_oprr", "(c) OPRR", "OPRR (%)", "#E9C46A", "#D62828", (29, 42)))
    jitter = np.random.default_rng(381).uniform(-0.10, 0.10, size=100)
    for ax, (field, title, label, fill, point, limits) in zip(axes, specs):
        values = np.asarray([float(row[field]) for row in rows])
        if values.min() < limits[0] or values.max() > limits[1]:
            margin = max((values.max() - values.min()) * 0.10, 1e-5)
            limits = (min(limits[0], values.min() - margin), max(limits[1], values.max() + margin))
        ax.boxplot([values], positions=[0], widths=0.40, patch_artist=True,
                   showfliers=False, manage_ticks=False,
                   boxprops=dict(facecolor=fill, alpha=0.75, linewidth=0.85),
                   medianprops=dict(color="#5C5C5C", linewidth=1.05))
        ax.scatter(jitter, values, s=9, color=point, alpha=0.50, linewidths=0)
        ax.set_xlim(-0.50, 0.50)
        ax.set_xticks([])
        ax.set_ylim(limits)
        ax.set_title(title)
        ax.set_ylabel(label)
        style_axis(ax)
        if field == "final_variance":
            ax.axhline(0.05, color="#B45D5D", linewidth=0.9, linestyle="--", label="软共识方差阈值")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.legend(loc="lower left", bbox_to_anchor=(0.015, 0.20), fontsize=7)
    return fig


FIGURES = {
    "training": ("section_32_monitor.csv", "training_process", plot_training),
    "variance": ("section_33_variance_trajectories.csv", "variance_comparison", plot_variance),
    "trajectories": ("section_332_opinion_trajectories.csv", "five_alternatives", plot_trajectories),
    "sensitivity": ("section_36_sensitivity.csv", "ordinal_weight_sensitivity", plot_sensitivity),
    "scalability": ("section_37_scalability.csv", "scalability", plot_scalability),
    "stability": ("section_381_id_states.csv", "random_state_stability", plot_stability),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=(*FIGURES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("paper_results"))
    parser.add_argument("--source", type=Path, help="Override input CSV for a single section")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--font", type=Path, help="Optional Chinese font file")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    if args.source is not None and args.section == "all":
        parser.error("--source requires a single --section")
    configure_fonts(args.font)
    output = create_output_directory(args.output)
    for name in (FIGURES if args.section == "all" else (args.section,)):
        filename, stem, draw = FIGURES[name]
        fig = draw(args.source or args.data_dir / filename)
        finalize_fonts(fig)
        fig.savefig(output / f"{stem}.pdf")
        fig.savefig(output / f"{stem}.png", dpi=args.dpi)
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
