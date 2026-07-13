import json
from pathlib import Path

from retrieve import ScholarLensRetriever
from rerank import ScholarLensReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "claim_verification_60.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "claim_evidence.json"

RETRIEVE_K = 20
TOP_K = 5


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Claim benchmark not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        claims = json.load(file)

    print(
        "\n========== PREPARING CLAIM EVIDENCE =========="
    )
    print(f"Claims: {len(claims)}")

    retriever = ScholarLensRetriever()
    reranker = ScholarLensReranker()

    prepared_claims = []

    for index, item in enumerate(claims, start=1):
        claim = item["claim"]

        print(
            f"[{index}/{len(claims)}] "
            f"{item['claim_id']}: {claim}"
        )

        dense_results = retriever.search(
            query=claim,
            top_k=RETRIEVE_K,
        )

        reranked_results = reranker.rerank(
            query=claim,
            candidates=dense_results,
            top_k=TOP_K,
        )

        prepared_claims.append(
            {
                "claim_id": item["claim_id"],
                "claim": claim,
                "expected_label": item["label"],
                "retrieve_k": RETRIEVE_K,
                "top_k": TOP_K,
                "evidence_results": reranked_results,
            }
        )

        print(
            f"  Saved {len(reranked_results)} "
            "reranked evidence passages."
        )

    output = {
        "total_claims": len(prepared_claims),
        "retrieve_k": RETRIEVE_K,
        "top_k": TOP_K,
        "claims": prepared_claims,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nClaim evidence preparation complete.")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()