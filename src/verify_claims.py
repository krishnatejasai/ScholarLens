import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from retrieve import ScholarLensRetriever
from rerank import ScholarLensReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "verification"
OUTPUT_FILE = OUTPUT_DIR / "claim_verification.json"

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

DEFAULT_RETRIEVE_K = 12
DEFAULT_TOP_K = 5

SUPPORT_THRESHOLD = 0.70
CONTRADICTION_THRESHOLD = 0.70


class ClaimVerifier:
    def __init__(self) -> None:
        print("Loading ScholarLens claim verifier...")

        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL
        )
        self.model.eval()

        self.id2label = {
            int(key): value.lower()
            for key, value in self.model.config.id2label.items()
        }

        print(f"NLI model: {NLI_MODEL}")
        print(f"Labels: {self.id2label}")
        print("Claim verifier ready.\n")

    def score_evidence(
        self,
        claim: str,
        evidence: str,
    ) -> dict:
        encoded = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0].cpu().numpy()

        label_scores = {
            self.id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        return {
            "entailment": label_scores.get("entailment", 0.0),
            "contradiction": label_scores.get("contradiction", 0.0),
            "neutral": label_scores.get("neutral", 0.0),
        }

    def verify(
        self,
        claim: str,
        evidence_results: list[dict],
    ) -> dict:
        evaluated_evidence = []

        for result in evidence_results:
            nli_scores = self.score_evidence(
                claim=claim,
                evidence=result["text"],
            )

            evaluated_evidence.append(
                {
                    **result,
                    "nli_scores": nli_scores,
                }
            )

        best_support = max(
            evaluated_evidence,
            key=lambda item: item["nli_scores"]["entailment"],
        )

        best_contradiction = max(
            evaluated_evidence,
            key=lambda item: item["nli_scores"]["contradiction"],
        )

        support_score = best_support["nli_scores"]["entailment"]
        contradiction_score = best_contradiction[
            "nli_scores"
        ]["contradiction"]

        if (
            support_score >= SUPPORT_THRESHOLD
            and support_score > contradiction_score
        ):
            verdict = "SUPPORTED"
            decisive_evidence = best_support

        elif (
            contradiction_score >= CONTRADICTION_THRESHOLD
            and contradiction_score > support_score
        ):
            verdict = "CONTRADICTED"
            decisive_evidence = best_contradiction

        else:
            verdict = "INSUFFICIENT_EVIDENCE"

            decisive_evidence = max(
                evaluated_evidence,
                key=lambda item: max(
                    item["nli_scores"]["entailment"],
                    item["nli_scores"]["contradiction"],
                ),
            )

        return {
            "claim": claim,
            "verdict": verdict,
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "decisive_evidence": decisive_evidence,
            "all_evidence": evaluated_evidence,
        }


def print_verification(result: dict) -> None:
    evidence = result["decisive_evidence"]
    scores = evidence["nli_scores"]

    print("\n========== SCHOLARLENS CLAIM VERIFICATION ==========")
    print(f"Claim:   {result['claim']}")
    print(f"Verdict: {result['verdict']}")

    print("\nAggregate scores")
    print(f"Best support score:       {result['support_score']:.4f}")
    print(
        "Best contradiction score: "
        f"{result['contradiction_score']:.4f}"
    )

    print("\nMost decisive evidence")
    print(f"Paper:      {evidence['title']}")
    print(f"Page:       {evidence['page_number']}")
    print(f"Chunk ID:   {evidence['chunk_id']}")
    print(f"Dense rank: {evidence['dense_rank']}")
    print(f"Final rank: {evidence['rank']}")

    print("\nNLI scores")
    print(f"Entailment:    {scores['entailment']:.4f}")
    print(f"Contradiction: {scores['contradiction']:.4f}")
    print(f"Neutral:       {scores['neutral']:.4f}")

    print("\nEvidence:")
    print(evidence["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ScholarLens claim verification using retrieved "
            "and reranked scientific evidence"
        )
    )

    parser.add_argument(
        "claim",
        type=str,
        help="Scientific or policy claim to verify",
    )

    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=DEFAULT_RETRIEVE_K,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    retriever = ScholarLensRetriever()
    reranker = ScholarLensReranker()
    verifier = ClaimVerifier()

    dense_results = retriever.search(
        query=args.claim,
        top_k=args.retrieve_k,
    )

    reranked_results = reranker.rerank(
        query=args.claim,
        candidates=dense_results,
        top_k=args.top_k,
    )

    verification = verifier.verify(
        claim=args.claim,
        evidence_results=reranked_results,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            verification,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_verification(verification)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()