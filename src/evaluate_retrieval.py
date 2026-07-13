import json
import math
from pathlib import Path
from statistics import mean

from retrieve import ScholarLensRetriever
from rerank import ScholarLensReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUERY_FILE = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "retrieval_queries.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)

OUTPUT_JSON = OUTPUT_DIR / "retrieval_evaluation.json"
OUTPUT_MD = OUTPUT_DIR / "retrieval_evaluation.md"


DENSE_RETRIEVE_K = 50
RERANK_CANDIDATE_K = 30
FINAL_PAPER_K = 5


def load_queries():
    if not QUERY_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation query file not found: {QUERY_FILE}"
        )

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def collapse_chunks_to_papers(
    results,
    score_key,
):
    """
    Convert ranked chunk results into a unique paper ranking.

    The first occurrence of a paper determines its paper-level rank.
    The highest-ranked chunk for each paper is retained.
    """

    seen_papers = set()
    paper_results = []

    for result in results:
        paper_id = result["paper_id"]

        if paper_id in seen_papers:
            continue

        seen_papers.add(paper_id)

        paper_results.append(
            {
                "paper_id": paper_id,
                "title": result["title"],
                "filename": result["filename"],
                "chunk_id": result["chunk_id"],
                "page_number": result["page_number"],
                "score": float(result[score_key]),
            }
        )

    return paper_results


def recall_at_k(
    ranked_papers,
    relevant_papers,
    k,
):
    """
    Binary Recall@K over all labeled relevant papers.
    Any relevance grade greater than zero counts as relevant.
    """

    relevant_ids = {
        paper_id
        for paper_id, grade in relevant_papers.items()
        if grade > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved_ids = {
        item["paper_id"]
        for item in ranked_papers[:k]
    }

    hits = len(
        relevant_ids.intersection(retrieved_ids)
    )

    return hits / len(relevant_ids)


def reciprocal_rank(
    ranked_papers,
    relevant_papers,
):
    """
    Reciprocal rank of the first relevant paper.
    """

    relevant_ids = {
        paper_id
        for paper_id, grade in relevant_papers.items()
        if grade > 0
    }

    for rank, item in enumerate(
        ranked_papers,
        start=1,
    ):
        if item["paper_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    ranked_papers,
    relevant_papers,
    k,
):
    """
    Graded DCG using gain = 2^relevance - 1.
    """

    dcg = 0.0

    for rank, item in enumerate(
        ranked_papers[:k],
        start=1,
    ):
        relevance = relevant_papers.get(
            item["paper_id"],
            0,
        )

        gain = (2 ** relevance) - 1

        discount = math.log2(rank + 1)

        dcg += gain / discount

    return dcg


def ideal_dcg_at_k(
    relevant_papers,
    k,
):
    ideal_grades = sorted(
        relevant_papers.values(),
        reverse=True,
    )[:k]

    idcg = 0.0

    for rank, relevance in enumerate(
        ideal_grades,
        start=1,
    ):
        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)

        idcg += gain / discount

    return idcg


def ndcg_at_k(
    ranked_papers,
    relevant_papers,
    k,
):
    dcg = dcg_at_k(
        ranked_papers,
        relevant_papers,
        k,
    )

    idcg = ideal_dcg_at_k(
        relevant_papers,
        k,
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_ranking(
    ranked_papers,
    relevant_papers,
):
    return {
        "recall_at_1": recall_at_k(
            ranked_papers,
            relevant_papers,
            1,
        ),
        "recall_at_3": recall_at_k(
            ranked_papers,
            relevant_papers,
            3,
        ),
        "recall_at_5": recall_at_k(
            ranked_papers,
            relevant_papers,
            5,
        ),
        "mrr": reciprocal_rank(
            ranked_papers,
            relevant_papers,
        ),
        "ndcg_at_5": ndcg_at_k(
            ranked_papers,
            relevant_papers,
            5,
        ),
    }


def average_metrics(query_results, method):
    metric_names = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
    ]

    return {
        metric: mean(
            result[method]["metrics"][metric]
            for result in query_results
        )
        for metric in metric_names
    }


def percentage_change(
    baseline,
    improved,
):
    if baseline == 0:
        return None

    return (
        (improved - baseline)
        / baseline
        * 100.0
    )


def build_markdown_report(report):
    dense = report["aggregate_metrics"]["dense"]
    reranked = report["aggregate_metrics"]["reranked"]
    improvements = report["relative_improvement_percent"]

    lines = [
        "# ScholarLens Retrieval Evaluation",
        "",
        "## Evaluation Setup",
        "",
        f"- Evaluation queries: {report['evaluation_queries']}",
        f"- Corpus papers: {report['corpus_papers']}",
        f"- Indexed chunks: {report['indexed_chunks']}",
        f"- Dense retrieval depth: {report['dense_retrieve_k']}",
        f"- Reranking candidate depth: {report['rerank_candidate_k']}",
        "- Evaluation level: unique paper ranking",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Dense Retrieval | Dense + Reranking | Relative Change |",
        "|---|---:|---:|---:|",
    ]

    metric_labels = {
        "recall_at_1": "Recall@1",
        "recall_at_3": "Recall@3",
        "recall_at_5": "Recall@5",
        "mrr": "MRR",
        "ndcg_at_5": "nDCG@5",
    }

    for metric, label in metric_labels.items():
        improvement = improvements[metric]

        if improvement is None:
            improvement_text = "N/A"
        else:
            improvement_text = f"{improvement:+.2f}%"

        lines.append(
            f"| {label} | "
            f"{dense[metric]:.4f} | "
            f"{reranked[metric]:.4f} | "
            f"{improvement_text} |"
        )

    lines.extend(
        [
            "",
            "## Per-Query Results",
            "",
            "| Query | Dense MRR | Reranked MRR | Dense nDCG@5 | Reranked nDCG@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for result in report["query_results"]:
        lines.append(
            f"| {result['query_id']} | "
            f"{result['dense']['metrics']['mrr']:.4f} | "
            f"{result['reranked']['metrics']['mrr']:.4f} | "
            f"{result['dense']['metrics']['ndcg_at_5']:.4f} | "
            f"{result['reranked']['metrics']['ndcg_at_5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The benchmark compares dense semantic retrieval against "
                "a two-stage pipeline that applies cross-encoder reranking "
                "to dense retrieval candidates."
            ),
            "",
            (
                "Recall measures whether labeled relevant papers are present "
                "within the top-ranked results. MRR measures how early the "
                "first relevant paper appears, while nDCG@5 rewards rankings "
                "that place highly relevant papers above partially relevant "
                "supporting papers."
            ),
            "",
            "## Evaluation Limitations",
            "",
            (
                "- Relevance judgments are manually defined at the paper level "
                "rather than the individual evidence-chunk level."
            ),
            (
                "- The benchmark contains a focused collection of research "
                "papers and should not be interpreted as a general web-scale "
                "retrieval benchmark."
            ),
            (
                "- Queries are designed around the technical themes represented "
                "in the evaluation corpus."
            ),
        ]
    )

    return "\n".join(lines)


def main():
    print(
        "\n========== SCHOLARLENS RETRIEVAL EVALUATION =========="
    )

    queries = load_queries()

    print(f"Evaluation queries: {len(queries)}")

    retriever = ScholarLensRetriever()
    reranker = ScholarLensReranker()

    query_results = []

    for index, item in enumerate(
        queries,
        start=1,
    ):
        query_id = item["query_id"]
        query = item["query"]
        relevant_papers = item["relevant_papers"]

        print(
            f"\n[{index}/{len(queries)}] "
            f"{query_id}: {query}"
        )

        dense_chunks = retriever.search(
            query=query,
            top_k=DENSE_RETRIEVE_K,
        )

        dense_papers = collapse_chunks_to_papers(
            results=dense_chunks,
            score_key="score",
        )

        rerank_candidates = dense_chunks[
            :RERANK_CANDIDATE_K
        ]

        reranked_chunks = reranker.rerank(
            query=query,
            candidates=rerank_candidates,
            top_k=RERANK_CANDIDATE_K,
        )

        reranked_papers = collapse_chunks_to_papers(
            results=reranked_chunks,
            score_key="rerank_score",
        )

        dense_metrics = evaluate_ranking(
            ranked_papers=dense_papers,
            relevant_papers=relevant_papers,
        )

        reranked_metrics = evaluate_ranking(
            ranked_papers=reranked_papers,
            relevant_papers=relevant_papers,
        )

        query_result = {
            "query_id": query_id,
            "query": query,
            "relevant_papers": relevant_papers,
            "dense": {
                "metrics": dense_metrics,
                "top_5_papers": dense_papers[
                    :FINAL_PAPER_K
                ],
            },
            "reranked": {
                "metrics": reranked_metrics,
                "top_5_papers": reranked_papers[
                    :FINAL_PAPER_K
                ],
            },
        }

        query_results.append(query_result)

        print(
            "  Dense:    "
            f"MRR={dense_metrics['mrr']:.3f}, "
            f"nDCG@5={dense_metrics['ndcg_at_5']:.3f}"
        )

        print(
            "  Reranked: "
            f"MRR={reranked_metrics['mrr']:.3f}, "
            f"nDCG@5={reranked_metrics['ndcg_at_5']:.3f}"
        )

    dense_average = average_metrics(
        query_results,
        "dense",
    )

    reranked_average = average_metrics(
        query_results,
        "reranked",
    )

    relative_improvements = {
        metric: percentage_change(
            dense_average[metric],
            reranked_average[metric],
        )
        for metric in dense_average
    }

    corpus_papers = len(
        {
            metadata["paper_id"]
            for metadata in retriever.metadata
        }
    )

    report = {
        "project": "ScholarLens AI",
        "evaluation_type": (
            "paper-level retrieval benchmark"
        ),
        "evaluation_queries": len(queries),
        "corpus_papers": corpus_papers,
        "indexed_chunks": len(retriever.metadata),
        "dense_retrieve_k": DENSE_RETRIEVE_K,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "aggregate_metrics": {
            "dense": dense_average,
            "reranked": reranked_average,
        },
        "relative_improvement_percent": (
            relative_improvements
        ),
        "query_results": query_results,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    markdown = build_markdown_report(report)

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(markdown)

    print(
        "\n========== AGGREGATE RESULTS =========="
    )

    print("\nDENSE RETRIEVAL")
    print(
        f"Recall@1: {dense_average['recall_at_1']:.4f}"
    )
    print(
        f"Recall@3: {dense_average['recall_at_3']:.4f}"
    )
    print(
        f"Recall@5: {dense_average['recall_at_5']:.4f}"
    )
    print(
        f"MRR:      {dense_average['mrr']:.4f}"
    )
    print(
        f"nDCG@5:   {dense_average['ndcg_at_5']:.4f}"
    )

    print("\nDENSE + RERANKING")
    print(
        f"Recall@1: {reranked_average['recall_at_1']:.4f}"
    )
    print(
        f"Recall@3: {reranked_average['recall_at_3']:.4f}"
    )
    print(
        f"Recall@5: {reranked_average['recall_at_5']:.4f}"
    )
    print(
        f"MRR:      {reranked_average['mrr']:.4f}"
    )
    print(
        f"nDCG@5:   {reranked_average['ndcg_at_5']:.4f}"
    )

    print("\nRELATIVE CHANGE")

    for metric, value in relative_improvements.items():
        if value is None:
            text = "N/A"
        else:
            text = f"{value:+.2f}%"

        print(f"{metric}: {text}")

    print(
        f"\nJSON report: {OUTPUT_JSON}"
    )
    print(
        f"Markdown report: {OUTPUT_MD}"
    )

    print(
        "\nRetrieval evaluation complete."
    )


if __name__ == "__main__":
    main()