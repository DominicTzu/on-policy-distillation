"""Plot evaluation summaries.

Run from the project root after eval/evaluate.py:
python eval/plot_results.py
"""

from pathlib import Path

from eval_utils import ROOT_DIR, read_json


SUMMARY_FILE = ROOT_DIR / "outputs" / "eval_results" / "summary.json"
PLOTS_DIR = ROOT_DIR / "outputs" / "eval_results" / "plots"


def percent(values):
    return [100 * value for value in values]


def save_bar_chart(path, labels, values, title, ylabel, color="#4c78a8", ylim=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_grouped_bar_chart(path, groups, series, title, ylabel, ylim=None):
    import matplotlib.pyplot as plt

    labels = list(series.keys())
    width = 0.8 / max(len(labels), 1)
    x_positions = list(range(len(groups)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for series_idx, label in enumerate(labels):
        offset = (series_idx - (len(labels) - 1) / 2) * width
        values = series[label]
        xs = [x + offset for x in x_positions]
        ax.bar(xs, values, width=width, label=label)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_overall(summary):
    checkpoints = summary["checkpoints"]
    labels = [item["checkpoint_name"] for item in checkpoints]

    save_bar_chart(
        PLOTS_DIR / "accuracy.png",
        labels,
        percent([item["accuracy"] for item in checkpoints]),
        "Final-answer accuracy",
        "accuracy (%)",
        color="#4c78a8",
        ylim=(0, 100),
    )
    save_bar_chart(
        PLOTS_DIR / "parse_fail_rate.png",
        labels,
        percent([item["parse_fail_rate"] for item in checkpoints]),
        "Parse failure rate",
        "parse fail (%)",
        color="#f58518",
        ylim=(0, 100),
    )
    save_grouped_bar_chart(
        PLOTS_DIR / "generated_lengths.png",
        labels,
        {
            "response tokens": [
                item["avg_response_tokens"] for item in checkpoints
            ],
            "rationale tokens": [
                item["avg_rationale_tokens"] for item in checkpoints
            ],
        },
        "Generated length",
        "average tokens",
    )


def plot_difficulty(summary):
    checkpoints = [
        item for item in summary["checkpoints"] if item.get("difficulty")
    ]
    if not checkpoints:
        return

    groups = ["easy", "medium", "hard"]
    accuracy_series = {}
    length_series = {}
    for item in checkpoints:
        name = item["checkpoint_name"]
        accuracy_series[name] = [
            100 * item.get("difficulty", {}).get(level, {}).get("accuracy", 0.0)
            for level in groups
        ]
        length_series[name] = [
            item.get("difficulty", {}).get(level, {}).get(
                "avg_rationale_tokens", 0.0
            )
            for level in groups
        ]

    save_grouped_bar_chart(
        PLOTS_DIR / "difficulty_accuracy.png",
        groups,
        accuracy_series,
        "Accuracy by difficulty bucket",
        "accuracy (%)",
        ylim=(0, 100),
    )
    save_grouped_bar_chart(
        PLOTS_DIR / "difficulty_rationale_length.png",
        groups,
        length_series,
        "Rationale length by difficulty bucket",
        "average rationale tokens",
    )


def plot_error_recovery(summary):
    error_recovery = summary.get("error_recovery")
    if not error_recovery:
        return

    save_bar_chart(
        PLOTS_DIR / "error_recovery.png",
        ["recovered", "regressed"],
        [
            100 * error_recovery["error_recovery_rate"],
            100 * error_recovery["regression_rate_among_correct_before"],
        ],
        "Error recovery and regression",
        "rate (%)",
        color="#54a24b",
        ylim=(0, 100),
    )


def main():
    if not Path(SUMMARY_FILE).exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_FILE}")

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_json(SUMMARY_FILE)

    plot_overall(summary)
    plot_difficulty(summary)
    plot_error_recovery(summary)

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
