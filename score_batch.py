from __future__ import annotations

import argparse
from pathlib import Path

from rtad.batch import score_batch_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the micro-batch comparison scorer.")
    parser.add_argument("--artifact-bundle", required=True, type=Path)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--experiment-id", default="batch-local")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scored = score_batch_file(
        bundle_path=args.artifact_bundle,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
        experiment_id=args.experiment_id,
    )
    print(f"Scored events: {scored}")


if __name__ == "__main__":
    main()
