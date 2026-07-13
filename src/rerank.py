import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from retrieve import ScholarLensRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrieval"
OUTPUT_FILE = OUTPUT_DIR / "reranked_results.json"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RETRIEVE_K = 10
DEFAULT_TOP_K = 5


class ScholarLensReranker:
    def __init__(self) -> None:
        print("Loading ScholarLens reranker...")
        self.model = CrossEncoder(RERANKER_MODEL)
        print(f"Reranker model: {RERANKER_MODEL}")
        print("Reranker ready.\n")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        if not candidates:
            return []

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        rerank_scores = self.model.predict(pairs)

        reranked_results = []

        for candidate, rerank_score in zip(
            candidates,
            rerank_scores,
        ):
            result = candidate.copy()
            result["dense_rank"] = candidate["rank"]
            result["dense_score"] = candidate["score"]
            result["rerank_score"] = float(rerank_score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        top_k = min(top_k, len(reranked_results))

        for new_rank, result in enumerate(
            reranked_results[:top_k],
            start=1,
        ):
            result["rank"] = new_rank

        return reranked_results[:top_k]


def print_results(
    query: str,
    results: list[dict],
) -> None:
    print("\n========== SCHOLARLENS RERANKED RETRIEVAL ==========")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}")

    for result in results:
        print("\n" + "=" * 70)
        print(f"Final rank:   {result['rank']}")
        print(f"Dense rank:   {result['dense_rank']}")
        print(f"Dense score:  {result['dense_score']:.4f}")
        print(f"Rerank score: {result['rerank_score']:.4f}")
        print(f"Paper:        {result['title']}")
        print(f"Page:         {result['page_number']}")
        print(f"Chunk ID:     {result['chunk_id']}")

        print("\nEvidence:")
        print(result["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ScholarLens dense retrieval followed by "
            "cross-encoder reranking"
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help="Scientific question or evidence query",
    )

    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=DEFAULT_RETRIEVE_K,
        help="Number of dense-retrieval candidates",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of reranked results to return",
    )

    args = parser.parse_args()

    retriever = ScholarLensRetriever()
    reranker = ScholarLensReranker()

    dense_results = retriever.search(
        query=args.query,
        top_k=args.retrieve_k,
    )

    reranked_results = reranker.rerank(
        query=args.query,
        candidates=dense_results,
        top_k=args.top_k,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "query": args.query,
        "retrieved_candidates": args.retrieve_k,
        "returned_results": len(reranked_results),
        "reranker_model": RERANKER_MODEL,
        "results": reranked_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print_results(args.query, reranked_results)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()