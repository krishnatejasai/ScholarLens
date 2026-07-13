import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "papers.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 120
MAX_SENTENCE_LENGTH = 600


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def hard_split_text(
    text: str,
    max_length: int = MAX_SENTENCE_LENGTH,
) -> list[str]:
    """Split abnormally long text while preferring word boundaries."""
    text = normalize_text(text)

    if len(text) <= max_length:
        return [text] if text else []

    pieces = []
    remaining = text

    while len(remaining) > max_length:
        split_position = remaining.rfind(" ", 0, max_length)

        if split_position <= 0:
            split_position = max_length

        piece = remaining[:split_position].strip()

        if piece:
            pieces.append(piece)

        remaining = remaining[split_position:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def split_into_sentences(text: str) -> list[str]:
    text = normalize_text(text)

    if not text:
        return []

    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z0-9])"
    raw_sentences = re.split(sentence_pattern, text)

    sentences = []

    for sentence in raw_sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentences.extend(hard_split_text(sentence))

    return sentences


def create_chunk_record(
    paper: dict[str, Any],
    page: dict[str, Any],
    chunk_index: int,
    text: str,
) -> dict[str, Any]:
    return {
        "chunk_id": (
            f"{paper['paper_id']}"
            f"_page_{page['page_number']}"
            f"_chunk_{chunk_index}"
        ),
        "paper_id": paper["paper_id"],
        "filename": paper["filename"],
        "title": paper["title"],
        "page_number": page["page_number"],
        "chunk_index": chunk_index,
        "text": text,
        "character_count": len(text),
    }


def get_overlap_sentences(
    sentences: list[str],
) -> list[str]:
    overlap_sentences = []
    overlap_length = 0

    for sentence in reversed(sentences):
        if overlap_length >= CHUNK_OVERLAP:
            break

        overlap_sentences.insert(0, sentence)
        overlap_length += len(sentence) + 1

    return overlap_sentences


def build_page_chunks(
    paper: dict[str, Any],
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    sentences = split_into_sentences(page["text"])

    if not sentences:
        return []

    chunks = []
    current_sentences: list[str] = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        additional_length = len(sentence) + (
            1 if current_sentences else 0
        )

        if (
            current_sentences
            and current_length + additional_length > CHUNK_SIZE
        ):
            chunk_text = " ".join(current_sentences).strip()

            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                chunks.append(
                    create_chunk_record(
                        paper,
                        page,
                        chunk_index,
                        chunk_text,
                    )
                )
                chunk_index += 1

            current_sentences = get_overlap_sentences(
                current_sentences
            )

# Prevent overlap itself from consuming too much
# of the next chunk's capacity.
            while (
                len(" ".join(current_sentences)) > CHUNK_OVERLAP
                and len(current_sentences) > 1
            ):
                current_sentences.pop(0)

            current_length = len(
                " ".join(current_sentences)
            )

        current_sentences.append(sentence)
        current_length = len(" ".join(current_sentences))

    final_text = " ".join(current_sentences).strip()

    if len(final_text) >= MIN_CHUNK_LENGTH:
        chunks.append(
            create_chunk_record(
                paper,
                page,
                chunk_index,
                final_text,
            )
        )

    return chunks


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed papers file not found: {INPUT_FILE}\n"
            "Run src/ingest.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        papers = json.load(file)

    all_chunks: list[dict[str, Any]] = []

    for paper in papers:
        paper_chunks = []

        for page in paper["pages"]:
            paper_chunks.extend(
                build_page_chunks(paper, page)
            )

        all_chunks.extend(paper_chunks)

        print(
            f"{paper['filename']}: "
            f"{len(paper_chunks)} chunks from "
            f"{paper['page_count']} pages"
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
        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    chunk_lengths = [
        chunk["character_count"]
        for chunk in all_chunks
    ]

    print("\nChunking complete.")
    print(f"Total papers: {len(papers)}")
    print(f"Total chunks: {len(all_chunks)}")

    if chunk_lengths:
        print(
            "Average chunk length: "
            f"{sum(chunk_lengths) / len(chunk_lengths):.1f}"
        )
        print(
            f"Minimum chunk length: "
            f"{min(chunk_lengths)}"
        )
        print(
            f"Maximum chunk length: "
            f"{max(chunk_lengths)}"
        )
        print(
            "Chunks over 1,100 characters: "
            f"{sum(length > 1100 for length in chunk_lengths)}"
        )

    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()