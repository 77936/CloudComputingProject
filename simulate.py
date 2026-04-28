from __future__ import annotations

import argparse
from pathlib import Path

from rtad.simulation import simulate_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay events for local or cloud-style streaming tests.")
    parser.add_argument("--artifact-bundle", required=True, type=Path)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--destination", required=True, choices=["local", "file", "pubsub"])
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--send-rate", type=float, default=0.0)
    parser.add_argument("--burst-multiplier", type=float, default=1.0)
    parser.add_argument("--experiment-id", default="stream-local")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sent = simulate_events(
        bundle_path=args.artifact_bundle,
        input_jsonl=args.input_jsonl,
        destination=args.destination,
        output_jsonl=args.output_jsonl,
        send_rate=args.send_rate,
        burst_multiplier=args.burst_multiplier,
        experiment_id=args.experiment_id,
    )
    print(f"Sent events: {sent}")


if __name__ == "__main__":
    main()
