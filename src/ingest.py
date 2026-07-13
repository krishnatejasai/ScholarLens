import json
import re
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw_papers"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "papers.json"


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metadata(document: fitz.Document, pdf_path: Path) -> dict[str, Any]:
    metadata = document.metadata or {}

    return {
        "paper_id": pdf_path.stem,
        "filename": pdf_path.name,
        "title": clean_text(metadata.get("title", "")) or pdf_path.stem,
        "author": clean_text(metadata.get("author", "")),
        "subject": clean_text(metadata.get("subject", "")),
        "keywords": clean_text(metadata.get("keywords", "")),
        "page_count": document.page_count,
    }


def extract_pages(document: fitz.Document) -> list[dict[str, Any]]:
    pages = []

    for page_number, page in enumerate(document, start=1):
        raw_text = page.get_text("text")
        cleaned_text = clean_text(raw_text)

        pages.append(
            {
                "page_number": page_number,
                "text": cleaned_text,
                "character_count": len(cleaned_text),
            }
        )

    return pages


def process_pdf(pdf_path: Path) -> dict[str, Any]:
    with fitz.open(pdf_path) as document:
        metadata = extract_metadata(document, pdf_path)
        pages = extract_pages(document)

    full_text = "\n\n".join(
        page["text"] for page in pages if page["text"]
    )

    return {
        **metadata,
        "pages": pages,
        "full_text": full_text,
        "character_count": len(full_text),
    }


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in: {INPUT_DIR}")
        print("Add at least one research paper PDF and run the script again.")
        return

    papers = []

    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")

        try:
            paper = process_pdf(pdf_path)
            papers.append(paper)

            print(
                f"  Extracted {paper['page_count']} pages and "
                f"{paper['character_count']:,} characters."
            )
        except Exception as error:
            print(f"  Failed to process {pdf_path.name}: {error}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(papers, file, indent=2, ensure_ascii=False)

    print("\nPDF ingestion complete.")
    print(f"Papers processed: {len(papers)}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()