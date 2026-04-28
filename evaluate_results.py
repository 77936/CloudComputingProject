from __future__ import annotations

import argparse
from pathlib import Path

from rtad.evaluation import summarize_result_file, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize stream and batch result files.")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--stream-results", required=True, type=Path)
    parser.add_argument("--batch-results", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stream_summary = summarize_result_file(args.stream_results)
    batch_summary = summarize_result_file(args.batch_results)
    comparison = {
        "stream": stream_summary,
        "batch": batch_summary,
        "latency_delta_ms": stream_summary["avg_latency_ms"] - batch_summary["avg_latency_ms"],
        "f1_delta": stream_summary["f1"] - batch_summary["f1"],
    }
    output_path = args.artifact_dir / "comparison_summary.json"
    write_json(output_path, comparison)
    print(f"Wrote comparison summary: {output_path}")


if __name__ == "__main__":
    main()
