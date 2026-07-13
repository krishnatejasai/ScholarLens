import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

CLAIM_FILE = (
    EVALUATION_DIR
    / "claim_verification_evaluation.json"
)

RETRIEVAL_FIGURE = (
    FIGURES_DIR
    / "retrieval_metrics_comparison.png"
)

CONFUSION_FIGURE = (
    FIGURES_DIR
    / "claim_verification_test_confusion_matrix.png"
)

CLASS_F1_FIGURE = (
    FIGURES_DIR
    / "claim_verification_test_f1.png"
)

SPLIT_FIGURE = (
    FIGURES_DIR
    / "claim_verification_split_comparison.png"
)


RETRIEVAL_METRICS = [
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "MRR",
    "nDCG@5",
]

DENSE_RESULTS = [
    0.7500,
    0.9583,
    0.9833,
    0.9028,
    0.9183,
]

RERANKED_RESULTS = [
    0.7917,
    0.9500,
    0.9833,
    0.9292,
    0.9365,
]

CLAIM_LABELS = [
    "Supported",
    "Contradicted",
    "Insufficient\nEvidence",
]


def load_claim_evaluation() -> dict:
    if not CLAIM_FILE.exists():
        raise FileNotFoundError(
            f"Claim evaluation report not found: {CLAIM_FILE}"
        )

    with open(
        CLAIM_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def annotate_bars(
    axes,
    bars,
    digits: int = 3,
) -> None:
    for bar in bars:
        height = bar.get_height()

        axes.annotate(
            f"{height:.{digits}f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_retrieval_comparison() -> None:
    positions = np.arange(
        len(RETRIEVAL_METRICS)
    )

    width = 0.36

    figure, axes = plt.subplots(
        figsize=(10, 6)
    )

    dense_bars = axes.bar(
        positions - width / 2,
        DENSE_RESULTS,
        width,
        label="Dense Retrieval",
    )

    reranked_bars = axes.bar(
        positions + width / 2,
        RERANKED_RESULTS,
        width,
        label="Dense + Reranking",
    )

    axes.set_title(
        "ScholarLens Retrieval Benchmark"
    )

    axes.set_ylabel("Score")

    axes.set_xticks(positions)
    axes.set_xticklabels(RETRIEVAL_METRICS)

    axes.set_ylim(0.65, 1.03)

    axes.legend()

    axes.grid(
        axis="y",
        alpha=0.25,
    )

    annotate_bars(
        axes,
        dense_bars,
        digits=3,
    )

    annotate_bars(
        axes,
        reranked_bars,
        digits=3,
    )

    figure.tight_layout()

    figure.savefig(
        RETRIEVAL_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_test_confusion_matrix(
    evaluation: dict,
) -> None:
    matrix = np.array(
        evaluation["test"]["confusion_matrix"]
    )

    figure, axes = plt.subplots(
        figsize=(7, 6)
    )

    image = axes.imshow(matrix)

    axes.set_title(
        "Held-Out Claim Verification Confusion Matrix"
    )

    axes.set_xlabel("Predicted Label")
    axes.set_ylabel("True Label")

    axes.set_xticks(
        np.arange(len(CLAIM_LABELS))
    )

    axes.set_yticks(
        np.arange(len(CLAIM_LABELS))
    )

    axes.set_xticklabels(CLAIM_LABELS)
    axes.set_yticklabels(CLAIM_LABELS)

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            axes.text(
                column_index,
                row_index,
                str(matrix[
                    row_index,
                    column_index,
                ]),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axes,
        label="Number of Claims",
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_test_class_f1(
    evaluation: dict,
) -> None:
    report = evaluation[
        "test"
    ]["classification_report"]

    internal_labels = [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE",
    ]

    scores = [
        report[label]["f1-score"]
        for label in internal_labels
    ]

    figure, axes = plt.subplots(
        figsize=(8, 5)
    )

    bars = axes.bar(
        CLAIM_LABELS,
        scores,
    )

    axes.set_title(
        "Held-Out Claim Verification F1 by Class"
    )

    axes.set_ylabel("F1 Score")
    axes.set_ylim(0.0, 1.05)

    axes.grid(
        axis="y",
        alpha=0.25,
    )

    annotate_bars(
        axes,
        bars,
        digits=3,
    )

    figure.tight_layout()

    figure.savefig(
        CLASS_F1_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_split_comparison(
    evaluation: dict,
) -> None:
    split_names = [
        "Development",
        "Held-Out Test",
        "Overall",
    ]

    split_keys = [
        "development",
        "test",
        "overall",
    ]

    accuracy_values = [
        evaluation[key]["accuracy"]
        for key in split_keys
    ]

    macro_f1_values = [
        evaluation[key]["macro_f1"]
        for key in split_keys
    ]

    positions = np.arange(
        len(split_names)
    )

    width = 0.36

    figure, axes = plt.subplots(
        figsize=(8, 5)
    )

    accuracy_bars = axes.bar(
        positions - width / 2,
        accuracy_values,
        width,
        label="Accuracy",
    )

    f1_bars = axes.bar(
        positions + width / 2,
        macro_f1_values,
        width,
        label="Macro-F1",
    )

    axes.set_title(
        "Claim Verification Performance by Split"
    )

    axes.set_ylabel("Score")
    axes.set_ylim(0.0, 1.05)

    axes.set_xticks(positions)
    axes.set_xticklabels(split_names)

    axes.legend()

    axes.grid(
        axis="y",
        alpha=0.25,
    )

    annotate_bars(
        axes,
        accuracy_bars,
        digits=3,
    )

    annotate_bars(
        axes,
        f1_bars,
        digits=3,
    )

    figure.tight_layout()

    figure.savefig(
        SPLIT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation = load_claim_evaluation()

    plot_retrieval_comparison()
    plot_test_confusion_matrix(evaluation)
    plot_test_class_f1(evaluation)
    plot_split_comparison(evaluation)

    print(
        "\n========== SCHOLARLENS FIGURES =========="
    )

    print(f"Retrieval comparison: {RETRIEVAL_FIGURE}")
    print(f"Test confusion matrix: {CONFUSION_FIGURE}")
    print(f"Test class F1: {CLASS_F1_FIGURE}")
    print(f"Split comparison: {SPLIT_FIGURE}")

    print("\nFigure generation complete.")


if __name__ == "__main__":
    main()