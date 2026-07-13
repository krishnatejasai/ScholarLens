import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_DIR = PROJECT_ROOT / "outputs" / "index"
INDEX_FILE = INDEX_DIR / "scholarlens.faiss"
METADATA_FILE = INDEX_DIR / "chunk_metadata.json"
INDEX_INFO_FILE = INDEX_DIR / "index_info.json"

DEFAULT_TOP_K = 5


def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Run src/embed.py first."
        )

    return faiss.read_index(str(INDEX_FILE))


def load_metadata() -> list[dict]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_index_info() -> dict:
    if not INDEX_INFO_FILE.exists():
        raise FileNotFoundError(
            f"Index info file not found: {INDEX_INFO_FILE}"
        )

    with open(INDEX_INFO_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


class ScholarLensRetriever:
    def __init__(self):
        print("Loading ScholarLens retrieval system...")

        self.index = load_index()
        self.metadata = load_metadata()
        self.index_info = load_index_info()

        model_name = self.index_info["embedding_model"]

        self.model = SentenceTransformer(model_name)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS vector count does not match metadata count."
            )

        print(f"Embedding model: {model_name}")
        print(f"Indexed chunks: {self.index.ntotal}")
        print("Retriever ready.\n")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, self.index.ntotal)

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for rank, (score, index_position) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if index_position == -1:
                continue

            chunk = self.metadata[index_position]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "title": chunk["title"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                }
            )

        return results


def print_results(
    query: str,
    results: list[dict],
) -> None:

    print("\n========== SCHOLARLENS RETRIEVAL ==========")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}")

    for result in results:
        print("\n" + "=" * 70)

        print(f"Rank:       {result['rank']}")
        print(f"Score:      {result['score']:.4f}")
        print(f"Paper:      {result['title']}")
        print(f"Page:       {result['page_number']}")
        print(f"Chunk ID:   {result['chunk_id']}")

        print("\nEvidence:")
        print(result["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScholarLens semantic scientific evidence retrieval"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Scientific question or evidence query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of evidence chunks to retrieve",
    )

    args = parser.parse_args()

    retriever = ScholarLensRetriever()

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()