import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "claim_verification_evaluation.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "claim_error_analysis.md"
)


def shorten(text: str, limit: int = 700) -> str:
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def main() -> None:
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}\n"
            "Run src/evaluate_claims.py first."
        )

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        evaluation = json.load(file)

    errors = [
        result
        for result in evaluation["results"]
        if not result["correct"]
    ]

    print(
        "\n========== SCHOLARLENS CLAIM ERROR ANALYSIS =========="
    )
    print(f"Total evaluated claims: {evaluation['total_claims']}")
    print(f"Incorrect predictions:  {len(errors)}")

    markdown_lines = [
        "# ScholarLens Claim Verification Error Analysis",
        "",
        f"- Total evaluated claims: {evaluation['total_claims']}",
        f"- Incorrect predictions: {len(errors)}",
        "",
    ]

    for index, error in enumerate(errors, start=1):
        evidence = error["decisive_evidence"]

        nli_scores = evidence.get(
            "nli_scores",
            {},
        )

        print("\n" + "=" * 78)
        print(f"Error {index}: {error['claim_id']}")
        print(f"Claim:      {error['claim']}")
        print(f"Expected:   {error['expected_label']}")
        print(f"Predicted:  {error['predicted_label']}")

        print("\nScores")
        print(
            f"Best support:       "
            f"{error['support_score']:.4f}"
        )
        print(
            f"Best contradiction: "
            f"{error['contradiction_score']:.4f}"
        )

        print("\nDecisive evidence")
        print(f"Paper ID:   {evidence['paper_id']}")
        print(f"Title:      {evidence['title']}")
        print(f"Page:       {evidence['page_number']}")
        print(f"Chunk ID:   {evidence['chunk_id']}")

        if "dense_rank" in evidence:
            print(f"Dense rank: {evidence['dense_rank']}")

        if "rank" in evidence:
            print(f"Final rank: {evidence['rank']}")

        if nli_scores:
            print("\nNLI probabilities")
            print(
                f"Entailment:    "
                f"{nli_scores.get('entailment', 0.0):.4f}"
            )
            print(
                f"Contradiction: "
                f"{nli_scores.get('contradiction', 0.0):.4f}"
            )
            print(
                f"Neutral:       "
                f"{nli_scores.get('neutral', 0.0):.4f}"
            )

        evidence_text = shorten(evidence["text"])

        print("\nEvidence:")
        print(evidence_text)

        markdown_lines.extend(
            [
                f"## {error['claim_id']}",
                "",
                f"**Claim:** {error['claim']}",
                "",
                f"- Expected: `{error['expected_label']}`",
                f"- Predicted: `{error['predicted_label']}`",
                f"- Support score: `{error['support_score']:.4f}`",
                (
                    "- Contradiction score: "
                    f"`{error['contradiction_score']:.4f}`"
                ),
                f"- Evidence paper: `{evidence['paper_id']}`",
                f"- Page: `{evidence['page_number']}`",
                f"- Chunk: `{evidence['chunk_id']}`",
                "",
                "**Decisive evidence:**",
                "",
                f"> {evidence_text}",
                "",
            ]
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(markdown_lines))

    print("\n" + "=" * 78)
    print(f"Error report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()