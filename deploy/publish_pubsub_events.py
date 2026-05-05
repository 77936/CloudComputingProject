from __future__ import annotations

import argparse
from pathlib import Path

from google.cloud import pubsub_v1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish JSONL events to a Pub/Sub topic.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--max-messages", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    publisher = pubsub_v1.PublisherClient()
    topic_path = args.topic if args.topic.startswith("projects/") else publisher.topic_path(args.project, args.topic)
    futures = []
    count = 0

    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            futures.append(publisher.publish(topic_path, line.rstrip("\n").encode("utf-8")))
            count += 1
            if count >= args.max_messages:
                break

    message_ids = [future.result(timeout=60) for future in futures]
    print(f"Published messages: {len(message_ids)}")
    if message_ids:
        print(f"First message ID: {message_ids[0]}")
        print(f"Last message ID: {message_ids[-1]}")


if __name__ == "__main__":
    main()
