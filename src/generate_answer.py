import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retrieve import ScholarLensRetriever
from rerank import ScholarLensReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "answers"
OUTPUT_FILE = OUTPUT_DIR / "citation_grounded_answer.json"

GENERATOR_MODEL = "google/flan-t5-base"
GROUNDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_RETRIEVE_K = 12
DEFAULT_TOP_K = 5

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 180

# A sentence must reach this similarity to receive a citation.
GROUNDING_THRESHOLD = 0.42

# Attach a second citation when it is independently relevant and close
# to the strongest evidence score.
SECONDARY_CITATION_THRESHOLD = 0.48
SECONDARY_SCORE_MARGIN = 0.06


class CitationGroundedGenerator:
    def __init__(self) -> None:
        print("Loading ScholarLens answer generator...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            GENERATOR_MODEL
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            GENERATOR_MODEL
        )
        self.model.eval()

        print(f"Generator model: {GENERATOR_MODEL}")

        print("Loading deterministic citation-grounding model...")

        self.grounding_model = SentenceTransformer(
            GROUNDING_MODEL
        )

        print(f"Grounding model: {GROUNDING_MODEL}")
        print("Citation-grounded generator ready.\n")

    @staticmethod
    def build_evidence_context(
        evidence_results: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        evidence_blocks: list[str] = []
        citation_map: list[dict[str, Any]] = []

        for index, result in enumerate(
            evidence_results,
            start=1,
        ):
            citation_id = f"E{index}"

            evidence_blocks.append(
                f"Evidence {citation_id}\n"
                f"Paper: {result['title']}\n"
                f"Page: {result['page_number']}\n"
                f"Passage: {result['text']}"
            )

            citation_map.append(
                {
                    "citation_id": citation_id,
                    "paper_id": result["paper_id"],
                    "title": result["title"],
                    "filename": result["filename"],
                    "page_number": result["page_number"],
                    "chunk_id": result["chunk_id"],
                    "dense_rank": result["dense_rank"],
                    "reranked_rank": result["rank"],
                    "dense_score": result["dense_score"],
                    "rerank_score": result["rerank_score"],
                    "text": result["text"],
                }
            )

        return "\n\n".join(evidence_blocks), citation_map

    @staticmethod
    def build_prompt(
        question: str,
        evidence_context: str,
    ) -> str:
        return f"""
You are an evidence-grounded research assistant.

Answer the research question using only the supplied evidence passages.

Instructions:
1. Write exactly three concise factual sentences.
2. Do not add a title or heading.
3. Do not repeat the question.
4. Do not use outside knowledge.
5. Do not invent facts, statistics, names, or conclusions.
6. Do not include citation labels; citations will be attached separately.
7. If the evidence cannot answer the question, write exactly:
   The available evidence is insufficient to answer this question.

Research question:
{question}

Evidence passages:
{evidence_context}

Answer:
""".strip()

    @staticmethod
    def clean_generated_answer(answer: str) -> str:
        # Remove citation tags if the generator produces them despite
        # being instructed not to.
        answer = re.sub(r"\[(?:E\d+|UNSUPPORTED)\]", "", answer)

        # Normalize spacing while preserving sentence punctuation.
        answer = re.sub(r"\s+", " ", answer).strip()

        return answer

    @staticmethod
    def split_sentences(answer: str) -> list[str]:
        if not answer.strip():
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            answer.strip(),
        )

        cleaned_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Add final punctuation when generation omits it.
            if sentence[-1] not in ".!?":
                sentence += "."

            cleaned_sentences.append(sentence)

        return cleaned_sentences

    def generate_raw_answer(
        self,
        prompt: str,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=45,
                num_beams=5,
                length_penalty=1.1,
                do_sample=False,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return self.clean_generated_answer(answer)

    def align_sentences_with_evidence(
        self,
        sentences: list[str],
        citation_map: list[dict[str, Any]],
    ) -> tuple[
        str,
        list[dict[str, Any]],
        list[str],
        list[str],
    ]:
        if not sentences:
            return "", [], [], []

        if not citation_map:
            unsupported = [
                {
                    "sentence": sentence,
                    "status": "UNSUPPORTED",
                    "citations": [],
                    "best_similarity": 0.0,
                    "evidence_matches": [],
                }
                for sentence in sentences
            ]

            answer = " ".join(
                f"{sentence} [UNSUPPORTED]"
                for sentence in sentences
            )

            return answer, unsupported, [], sentences

        evidence_texts = [
            citation["text"]
            for citation in citation_map
        ]

        sentence_embeddings = self.grounding_model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        evidence_embeddings = self.grounding_model.encode(
            evidence_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sentence_embeddings = np.asarray(
            sentence_embeddings,
            dtype=np.float32,
        )

        evidence_embeddings = np.asarray(
            evidence_embeddings,
            dtype=np.float32,
        )

        similarity_matrix = (
            sentence_embeddings @ evidence_embeddings.T
        )

        aligned_sentences: list[str] = []
        grounding_details: list[dict[str, Any]] = []
        used_citation_ids: set[str] = set()
        unsupported_sentences: list[str] = []

        for sentence_index, sentence in enumerate(sentences):
            scores = similarity_matrix[sentence_index]

            ranked_indices = np.argsort(scores)[::-1]

            best_index = int(ranked_indices[0])
            best_score = float(scores[best_index])

            selected_indices: list[int] = []

            if best_score >= GROUNDING_THRESHOLD:
                selected_indices.append(best_index)

                if len(ranked_indices) > 1:
                    second_index = int(ranked_indices[1])
                    second_score = float(scores[second_index])

                    if (
                        second_score
                        >= SECONDARY_CITATION_THRESHOLD
                        and best_score - second_score
                        <= SECONDARY_SCORE_MARGIN
                    ):
                        selected_indices.append(second_index)

            if selected_indices:
                citations = [
                    citation_map[index]["citation_id"]
                    for index in selected_indices
                ]

                citation_text = "".join(
                    f"[{citation_id}]"
                    for citation_id in citations
                )

                aligned_sentences.append(
                    f"{sentence} {citation_text}"
                )

                used_citation_ids.update(citations)
                status = "GROUNDED"
            else:
                citations = []
                aligned_sentences.append(
                    f"{sentence} [UNSUPPORTED]"
                )
                unsupported_sentences.append(sentence)
                status = "UNSUPPORTED"

            top_matches = []

            for evidence_index in ranked_indices[:3]:
                evidence_index = int(evidence_index)

                top_matches.append(
                    {
                        "citation_id": citation_map[
                            evidence_index
                        ]["citation_id"],
                        "similarity": float(
                            scores[evidence_index]
                        ),
                        "page_number": citation_map[
                            evidence_index
                        ]["page_number"],
                        "chunk_id": citation_map[
                            evidence_index
                        ]["chunk_id"],
                    }
                )

            grounding_details.append(
                {
                    "sentence_number": sentence_index + 1,
                    "sentence": sentence,
                    "status": status,
                    "citations": citations,
                    "best_similarity": best_score,
                    "evidence_matches": top_matches,
                }
            )

        ordered_used_citations = sorted(
            used_citation_ids,
            key=lambda citation_id: int(
                citation_id[1:]
            ),
        )

        grounded_answer = " ".join(aligned_sentences)

        return (
            grounded_answer,
            grounding_details,
            ordered_used_citations,
            unsupported_sentences,
        )

    def generate(
        self,
        question: str,
        evidence_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence_context, citation_map = (
            self.build_evidence_context(evidence_results)
        )

        prompt = self.build_prompt(
            question=question,
            evidence_context=evidence_context,
        )

        raw_answer = self.generate_raw_answer(prompt)

        sentences = self.split_sentences(raw_answer)

        (
            grounded_answer,
            sentence_grounding,
            used_citations,
            unsupported_sentences,
        ) = self.align_sentences_with_evidence(
            sentences=sentences,
            citation_map=citation_map,
        )

        total_sentences = len(sentences)
        grounded_sentence_count = (
            total_sentences - len(unsupported_sentences)
        )

        citation_coverage = (
            grounded_sentence_count / total_sentences
            if total_sentences
            else 0.0
        )

        evidence_utilization_ratio = (
            len(used_citations) / len(citation_map)
            if citation_map
            else 0.0
        )

        grounding_scores = [
            item["best_similarity"]
            for item in sentence_grounding
            if item["status"] == "GROUNDED"
        ]

        mean_grounding_similarity = (
            sum(grounding_scores) / len(grounding_scores)
            if grounding_scores
            else 0.0
        )

        citation_validation_passed = (
            total_sentences > 0
            and grounded_sentence_count == total_sentences
            and len(used_citations) > 0
        )

        return {
            "question": question,
            "raw_answer": raw_answer,
            "answer": grounded_answer,
            "generator_model": GENERATOR_MODEL,
            "grounding_model": GROUNDING_MODEL,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "retrieved_evidence_count": len(
                evidence_results
            ),
            "answer_sentence_count": total_sentences,
            "grounded_sentence_count": (
                grounded_sentence_count
            ),
            "unsupported_sentence_count": len(
                unsupported_sentences
            ),
            "unsupported_sentences": unsupported_sentences,
            "used_citations": used_citations,
            "invalid_citations": [],
            "citation_validation_passed": (
                citation_validation_passed
            ),
            "citation_coverage": citation_coverage,
            "evidence_utilization_ratio": (
                evidence_utilization_ratio
            ),
            "mean_grounding_similarity": (
                mean_grounding_similarity
            ),
            "sentence_grounding": sentence_grounding,
            "citation_map": citation_map,
        }


def print_answer(result: dict[str, Any]) -> None:
    print(
        "\n========== SCHOLARLENS "
        "CITATION-GROUNDED SYNTHESIS =========="
    )

    print(f"Question: {result['question']}")

    print("\nRaw generated answer:")
    print(result["raw_answer"])

    print("\nGrounded answer:")
    print(result["answer"])

    print("\nCitation validation")
    print(
        f"Answer sentences:       "
        f"{result['answer_sentence_count']}"
    )
    print(
        f"Grounded sentences:     "
        f"{result['grounded_sentence_count']}"
    )
    print(
        f"Unsupported sentences:  "
        f"{result['unsupported_sentence_count']}"
    )
    print(
        f"Used citations:         "
        f"{result['used_citations']}"
    )
    print(
        f"Validation passed:      "
        f"{result['citation_validation_passed']}"
    )
    print(
        f"Citation coverage:      "
        f"{result['citation_coverage']:.3f}"
    )
    print(
        f"Evidence utilization:   "
        f"{result['evidence_utilization_ratio']:.3f}"
    )
    print(
        f"Mean grounding score:   "
        f"{result['mean_grounding_similarity']:.4f}"
    )

    print("\nSentence-level grounding")

    for item in result["sentence_grounding"]:
        citation_text = (
            ", ".join(item["citations"])
            if item["citations"]
            else "None"
        )

        print(
            f"\nSentence {item['sentence_number']}: "
            f"{item['status']}"
        )
        print(
            f"Best similarity: "
            f"{item['best_similarity']:.4f}"
        )
        print(f"Citations: {citation_text}")
        print(item["sentence"])

    print("\nEvidence references")

    used = set(result["used_citations"])

    for citation in result["citation_map"]:
        if citation["citation_id"] not in used:
            continue

        print(
            f"[{citation['citation_id']}] "
            f"{citation['title']}, "
            f"page {citation['page_number']}, "
            f"chunk {citation['chunk_id']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ScholarLens citation-grounded synthesis "
            "from retrieved scientific evidence"
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help="Research question to answer",
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
        help="Number of reranked evidence chunks",
    )

    args = parser.parse_args()

    retriever = ScholarLensRetriever()
    reranker = ScholarLensReranker()
    generator = CitationGroundedGenerator()

    dense_results = retriever.search(
        query=args.question,
        top_k=args.retrieve_k,
    )

    reranked_results = reranker.rerank(
        query=args.question,
        candidates=dense_results,
        top_k=args.top_k,
    )

    result = generator.generate(
        question=args.question,
        evidence_results=reranked_results,
    )

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
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_answer(result)

    print(
        f"\nAnswer artifact saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()