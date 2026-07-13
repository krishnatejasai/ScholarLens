import argparse
import subprocess
import sys

COMMANDS = {
    "ingest": "src/ingest.py",
    "chunk": "src/chunk.py",
    "embed": "src/embed.py",
    "retrieve": "src/retrieve.py",
    "rerank": "src/rerank.py",
    "verify": "src/verify_claims.py",
    "answer": "src/generate_answer.py",
    "eval-retrieval": "src/evaluate_retrieval.py",
    "prepare-claims": "src/prepare_claim_evidence.py",
    "eval-claims": "src/evaluate_claims.py",
    "figures": "src/generate_figures.py",
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
    )

    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
    )

    parsed = parser.parse_args()

    subprocess.run(
        [sys.executable, COMMANDS[parsed.command]] + parsed.args,
        check=True,
    )


if __name__ == "__main__":
    main()