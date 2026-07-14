import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


COMMANDS = {
    "ingest": {
        "script": "ingest.py",
        "help": "Extract and preprocess text from research papers.",
    },
    "chunk": {
        "script": "chunk.py",
        "help": "Split processed documents into retrieval chunks.",
    },
    "embed": {
        "script": "embed.py",
        "help": "Generate embeddings and build the FAISS index.",
    },
    "retrieve": {
        "script": "retrieve.py",
        "help": "Run dense scientific-document retrieval.",
    },
    "rerank": {
        "script": "rerank.py",
        "help": "Rerank retrieved passages with a cross-encoder.",
    },
    "verify": {
        "script": "verify_claims.py",
        "help": "Verify scientific claims against retrieved evidence.",
    },
    "answer": {
        "script": "generate_answer.py",
        "help": "Generate citation-grounded answers from retrieved evidence.",
    },
    "eval-retrieval": {
        "script": "evaluate_retrieval.py",
        "help": "Evaluate dense retrieval and reranking performance.",
    },
    "prepare-claims": {
        "script": "prepare_claim_evidence.py",
        "help": "Prepare evidence for the claim-verification benchmark.",
    },
    "eval-claims": {
        "script": "evaluate_claims.py",
        "help": "Evaluate scientific claim-verification performance.",
    },
    "figures": {
        "script": "generate_figures.py",
        "help": "Generate evaluation plots and figures.",
    },
    "report": {
        "script": "generate_report.py",
        "help": "Generate the consolidated ScholarLens evaluation report.",
    },
    "analyze-claim-errors": {
        "script": "analyze_claim_errors.py",
        "help": "Analyze claim-verification errors.",
    },
}


def run_script(script_name: str, extra_args: list[str]) -> None:
    script_path = PROJECT_ROOT / "src" / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        *extra_args,
    ]

    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Command-line interface for ScholarLens scientific "
            "literature retrieval, verification, and answer generation."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for command_name, command_config in COMMANDS.items():
        command_parser = subparsers.add_parser(
            command_name,
            help=command_config["help"],
            description=command_config["help"],
        )

        command_parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help=(
                "Additional arguments forwarded to the underlying script."
            ),
        )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_script(
        script_name=COMMANDS[args.command]["script"],
        extra_args=args.args,
    )


if __name__ == "__main__":
    main()