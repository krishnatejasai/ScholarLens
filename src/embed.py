import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

INDEX_DIR = PROJECT_ROOT / "outputs" / "index"
INDEX_FILE = INDEX_DIR / "scholarlens.faiss"
METADATA_FILE = INDEX_DIR / "chunk_metadata.json"
INDEX_INFO_FILE = INDEX_DIR / "index_info.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Run src/chunk.py first."
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print("\n========== SCHOLARLENS EMBEDDING INDEX ==========")

    chunks = load_chunks()

    if not chunks:
        raise ValueError("No chunks found in chunks.json.")

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    metadata = []

    for chunk in chunks:
        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "filename": chunk["filename"],
                "title": chunk["title"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }
        )

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    index_info = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "similarity_metric": "cosine_similarity_via_normalized_inner_product",
        "total_vectors": int(index.ntotal),
        "batch_size": BATCH_SIZE,
    }

    with open(INDEX_INFO_FILE, "w", encoding="utf-8") as file:
        json.dump(index_info, file, indent=2)

    print("\n========== INDEX RESULTS ==========")
    print(f"Vectors indexed:       {index.ntotal}")
    print(f"Embedding dimension:   {embedding_dimension}")
    print("Similarity metric:     Cosine similarity")
    print(f"FAISS index saved to:  {INDEX_FILE}")
    print(f"Metadata saved to:     {METADATA_FILE}")
    print(f"Index info saved to:   {INDEX_INFO_FILE}")

    print("\nEmbedding index creation complete.")


if __name__ == "__main__":
    main()