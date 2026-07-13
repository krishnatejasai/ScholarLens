import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from verify_claims import ClaimVerifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "claim_evidence.json"
)

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "claim_verification_60.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

OUTPUT_JSON = (
    OUTPUT_DIR
    / "claim_verification_evaluation.json"
)

OUTPUT_MD = (
    OUTPUT_DIR
    / "claim_verification_evaluation.md"
)

LABELS = [
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def compute_metrics(
    results: list[dict],
) -> dict:
    y_true = [
        result["expected_label"]
        for result in results
    ]

    y_pred = [
        result["predicted_label"]
        for result in results
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    ).tolist()

    return {
        "total_claims": len(results),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def print_metric_block(
    title: str,
    metrics: dict,
) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Total claims: {metrics['total_claims']}"
    )
    print(
        f"Accuracy:     {metrics['accuracy']:.4f}"
    )
    print(
        f"Macro-F1:     {metrics['macro_f1']:.4f}"
    )

    print("\nPer-class metrics")

    report = metrics["classification_report"]

    for label in LABELS:
        values = report[label]

        print(
            f"{label}: "
            f"P={values['precision']:.3f}, "
            f"R={values['recall']:.3f}, "
            f"F1={values['f1-score']:.3f}, "
            f"Support={int(values['support'])}"
        )

    print("\nConfusion matrix")
    print("Rows=true, columns=predicted")
    print(LABELS)

    for row in metrics["confusion_matrix"]:
        print(row)


def build_markdown_report(
    output: dict,
) -> str:
    lines = [
        "# ScholarLens Claim Verification Evaluation",
        "",
        "## Evaluation Setup",
        "",
        f"- Total claims: {output['overall']['total_claims']}",
        f"- Development claims: {output['development']['total_claims']}",
        f"- Held-out test claims: {output['test']['total_claims']}",
        "- Balanced labels: SUPPORTED, CONTRADICTED, "
        "INSUFFICIENT_EVIDENCE",
        "- Evidence source: dense retrieval followed by "
        "cross-encoder reranking",
        "- Verification model: cross-encoder/nli-deberta-v3-base",
        "",
        "## Aggregate Results",
        "",
        "| Split | Claims | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
        (
            f"| Overall | {output['overall']['total_claims']} | "
            f"{output['overall']['accuracy']:.4f} | "
            f"{output['overall']['macro_f1']:.4f} |"
        ),
        (
            f"| Development | "
            f"{output['development']['total_claims']} | "
            f"{output['development']['accuracy']:.4f} | "
            f"{output['development']['macro_f1']:.4f} |"
        ),
        (
            f"| Held-out test | "
            f"{output['test']['total_claims']} | "
            f"{output['test']['accuracy']:.4f} | "
            f"{output['test']['macro_f1']:.4f} |"
        ),
    ]

    for split_key, split_title in [
        ("overall", "Overall"),
        ("development", "Development"),
        ("test", "Held-Out Test"),
    ]:
        metrics = output[split_key]
        report = metrics["classification_report"]

        lines.extend(
            [
                "",
                f"## {split_title} Per-Class Results",
                "",
                "| Label | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for label in LABELS:
            values = report[label]

            lines.append(
                f"| {label} | "
                f"{values['precision']:.4f} | "
                f"{values['recall']:.4f} | "
                f"{values['f1-score']:.4f} | "
                f"{int(values['support'])} |"
            )

        matrix = metrics["confusion_matrix"]

        lines.extend(
            [
                "",
                f"### {split_title} Confusion Matrix",
                "",
                "| True / Predicted | SUPPORTED | "
                "CONTRADICTED | INSUFFICIENT_EVIDENCE |",
                "|---|---:|---:|---:|",
            ]
        )

        for label, row in zip(LABELS, matrix):
            lines.append(
                f"| {label} | "
                f"{row[0]} | {row[1]} | {row[2]} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "ScholarLens performs three-way evidence-based claim "
                "verification after dense retrieval and cross-encoder "
                "reranking. Development results may be used for system "
                "analysis, while held-out test results represent the "
                "primary final benchmark."
            ),
            "",
            "## Limitations",
            "",
            (
                "- The benchmark contains manually authored claims "
                "derived from a focused 20-paper corpus."
            ),
            (
                "- Verification performance depends jointly on evidence "
                "retrieval, reranking, and NLI classification."
            ),
            (
                "- Full-passage NLI may be affected by irrelevant text, "
                "bibliographic content, or multiple claims within a chunk."
            ),
            (
                "- The results should not be interpreted as performance "
                "on general open-domain scientific fact verification."
            ),
        ]
    )

    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    prepared_data = load_json(EVIDENCE_FILE)
    benchmark = load_json(BENCHMARK_FILE)

    split_by_claim_id = {
        item["claim_id"]: item["split"]
        for item in benchmark
    }

    benchmark_label_by_id = {
        item["claim_id"]: item["label"]
        for item in benchmark
    }

    claims = prepared_data["claims"]

    print(
        "\n========== SCHOLARLENS CLAIM "
        "VERIFICATION EVALUATION =========="
    )
    print(f"Prepared claims: {len(claims)}")

    verifier = ClaimVerifier()

    results = []

    for index, item in enumerate(
        claims,
        start=1,
    ):
        claim_id = item["claim_id"]

        if claim_id not in split_by_claim_id:
            raise KeyError(
                f"Claim ID missing from benchmark: {claim_id}"
            )

        expected = benchmark_label_by_id[claim_id]
        split = split_by_claim_id[claim_id]

        verification = verifier.verify(
            claim=item["claim"],
            evidence_results=item["evidence_results"],
        )

        predicted = verification["verdict"]

        correct = expected == predicted

        results.append(
            {
                "claim_id": claim_id,
                "split": split,
                "claim": item["claim"],
                "expected_label": expected,
                "predicted_label": predicted,
                "correct": correct,
                "support_score": verification[
                    "support_score"
                ],
                "contradiction_score": verification[
                    "contradiction_score"
                ],
                "decisive_evidence": verification[
                    "decisive_evidence"
                ],
            }
        )

        symbol = "✓" if correct else "✗"

        print(
            f"[{index}/{len(claims)}] "
            f"{claim_id} | "
            f"Split={split} | "
            f"Expected={expected} | "
            f"Predicted={predicted} | "
            f"{symbol}"
        )

    development_results = [
        result
        for result in results
        if result["split"] == "dev"
    ]

    test_results = [
        result
        for result in results
        if result["split"] == "test"
    ]

    overall_metrics = compute_metrics(results)

    development_metrics = compute_metrics(
        development_results
    )

    test_metrics = compute_metrics(
        test_results
    )

    output = {
        "project": "ScholarLens AI",
        "evaluation_type": (
            "three-way scientific claim verification"
        ),
        "labels": LABELS,
        "overall": overall_metrics,
        "development": development_metrics,
        "test": test_metrics,
        "results": results,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    markdown = build_markdown_report(output)

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(markdown)

    print_metric_block(
        "OVERALL RESULTS",
        overall_metrics,
    )

    print_metric_block(
        "DEVELOPMENT RESULTS",
        development_metrics,
    )

    print_metric_block(
        "HELD-OUT TEST RESULTS",
        test_metrics,
    )

    incorrect_test_ids = [
        result["claim_id"]
        for result in test_results
        if not result["correct"]
    ]

    print("\nHeld-out test errors:")
    print(
        incorrect_test_ids
        if incorrect_test_ids
        else "None"
    )

    print(f"\nJSON report: {OUTPUT_JSON}")
    print(f"Markdown report: {OUTPUT_MD}")

    print(
        "\nClaim verification evaluation complete."
    )


if __name__ == "__main__":
    main()