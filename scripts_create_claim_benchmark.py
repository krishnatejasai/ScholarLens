import json
from pathlib import Path


OUTPUT_FILE = Path("data/eval/claim_verification_60.json")


SUPPORTED = [
    "Dense Passage Retrieval uses separate learned encoders to represent questions and passages.",
    "ColBERT applies late interaction between contextualized query and document token embeddings.",
    "Contriever learns dense retrieval representations using contrastive learning without relevance labels.",
    "GTR studies whether larger dual encoders generalize across retrieval domains.",
    "REALM retrieves external documents during language-model pretraining.",
    "RAG combines parametric sequence-to-sequence memory with a non-parametric dense document index.",
    "Atlas applies retrieval-augmented language models to few-shot learning.",
    "Self-RAG uses reflection mechanisms to retrieve, generate, and critique outputs.",
    "CRAG evaluates retrieved documents and can trigger corrective retrieval actions.",
    "RAGAS provides reference-free metrics for evaluating retrieval-augmented generation pipelines.",
    "ALCE studies how language models can generate answers with verifiable citations.",
    "FActScore decomposes long-form generations into atomic facts for factual-precision evaluation.",
    "SciFact contains scientific claims with supporting or refuting evidence and rationales.",
    "SciFact-Open evaluates scientific claim verification over a substantially larger open-domain corpus.",
    "FEVER includes supported, refuted, and not-enough-information claim labels.",
    "SPECTER uses citation information to learn scientific document representations.",
    "SciNCL uses neighborhood-based contrastive learning for scientific paper embeddings.",
    "S2ORC provides structured scholarly text, metadata, citations, and bibliographic information.",
    "Longformer uses attention patterns whose computational cost scales linearly with sequence length.",
    "Lost in the Middle reports weaker performance when relevant information appears in the middle of long contexts.",
]


CONTRADICTED = [
    "Dense Passage Retrieval relies exclusively on BM25 and does not use neural encoders.",
    "ColBERT requires a full cross-encoder computation for every document before retrieval.",
    "Contriever requires manually labeled relevance judgments for all of its training examples.",
    "GTR concludes that increasing dual-encoder size consistently reduces cross-domain retrieval performance.",
    "REALM stores all factual knowledge only in model parameters and never retrieves external documents.",
    "RAG uses only parametric memory and does not access a document index.",
    "Atlas is designed exclusively for supervised learning with thousands of labeled examples.",
    "Self-RAG always retrieves documents and does not critique its generated responses.",
    "CRAG assumes all retrieved passages are reliable and performs no retrieval-quality assessment.",
    "RAGAS requires a gold reference answer for every metric it computes.",
    "ALCE removes citations from generated answers because citations reduce factual reliability.",
    "FActScore evaluates factuality without decomposing text into atomic facts.",
    "SciFact uses a single evidence label and does not distinguish support from refutation.",
    "SciFact-Open restricts verification to the same small fixed corpus used by the original SciFact benchmark.",
    "FEVER contains only supported claims and does not include refuted or unverifiable examples.",
    "SPECTER learns paper representations without using citation relationships.",
    "SciNCL selects positive and negative papers randomly without neighborhood information.",
    "S2ORC contains no citation links or structured bibliographic metadata.",
    "Longformer preserves the quadratic self-attention cost of standard Transformers for long sequences.",
    "Lost in the Middle finds that models perform best whenever evidence is placed in the middle of the context.",
]


INSUFFICIENT = [
    "Dense Passage Retrieval reduced global electricity consumption by 18 percent in 2021.",
    "ColBERT was deployed in every major commercial search engine by 2022.",
    "Contriever increased agricultural crop yields by 35 percent.",
    "GTR was trained using private medical records from ten hospitals.",
    "REALM was used to predict rainfall in South Asia with 97 percent accuracy.",
    "The original RAG model generated legal judgments for the United States Supreme Court.",
    "Atlas reduced the cost of satellite launches by 40 percent.",
    "Self-RAG was tested as an autonomous driving controller on public roads.",
    "CRAG improved global internet speeds by 25 percent.",
    "RAGAS was adopted as a legally required evaluation standard by the European Union.",
    "ALCE guarantees that every generated citation is factually correct in all possible domains.",
    "FActScore proves that every language model hallucination can be eliminated.",
    "SciFact reports that scientific misinformation declined globally by 60 percent after its release.",
    "SciFact-Open was evaluated using confidential pharmaceutical trial records.",
    "FEVER was created to verify live television broadcasts in real time.",
    "SPECTER predicts future Nobel Prize winners from citation networks.",
    "SciNCL determines the financial market value of research papers.",
    "S2ORC contains complete full text for every scientific paper ever published.",
    "Longformer was used to discover a new antibiotic with guaranteed clinical effectiveness.",
    "Lost in the Middle concludes that increasing context length always improves accuracy on every task.",
]


def build_items(claims, label, start_index):
    items = []

    for offset, claim in enumerate(claims):
        claim_number = start_index + offset

        # Odd-numbered examples enter development; even enter test.
        split = "dev" if claim_number % 2 == 1 else "test"

        items.append(
            {
                "claim_id": f"c{claim_number:02d}",
                "claim": claim,
                "label": label,
                "split": split,
            }
        )

    return items


def main():
    benchmark = []

    benchmark.extend(
        build_items(SUPPORTED, "SUPPORTED", 1)
    )
    benchmark.extend(
        build_items(CONTRADICTED, "CONTRADICTED", 21)
    )
    benchmark.extend(
        build_items(
            INSUFFICIENT,
            "INSUFFICIENT_EVIDENCE",
            41,
        )
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(benchmark, indent=2),
        encoding="utf-8",
    )

    label_counts = {}
    split_counts = {}

    for item in benchmark:
        label_counts[item["label"]] = (
            label_counts.get(item["label"], 0) + 1
        )
        split_counts[item["split"]] = (
            split_counts.get(item["split"], 0) + 1
        )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total claims: {len(benchmark)}")
    print(f"Labels: {label_counts}")
    print(f"Splits: {split_counts}")


if __name__ == "__main__":
    main()