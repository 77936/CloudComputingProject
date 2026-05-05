from __future__ import annotations

import argparse
from datetime import UTC, datetime
import time

from google.cloud import aiplatform, bigquery, pubsub_v1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small Pub/Sub -> Vertex -> BigQuery smoke runner.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--bigquery-table", required=True)
    parser.add_argument("--max-messages", type=int, default=10)
    parser.add_argument("--experiment-id", default="stream-smoke")
    parser.add_argument("--pull-batch-size", type=int, default=50)
    parser.add_argument("--idle-timeout-seconds", type=int, default=30)
    return parser


def normalize_table(table: str) -> str:
    if ":" not in table:
        return table
    project, rest = table.split(":", 1)
    return f"{project}.{rest}"


def main() -> None:
    args = build_parser().parse_args()
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = (
        args.subscription
        if args.subscription.startswith("projects/")
        else subscriber.subscription_path(args.project, args.subscription)
    )
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{args.project}/locations/{args.region}/endpoints/{args.endpoint_id}",
        project=args.project,
        location=args.region,
    )
    bq_client = bigquery.Client(project=args.project)
    table = normalize_table(args.bigquery_table)
    total_processed = 0
    idle_start = time.monotonic()

    while total_processed < args.max_messages:
        response = subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": min(args.pull_batch_size, args.max_messages - total_processed),
            },
            timeout=30,
        )
        if not response.received_messages:
            if time.monotonic() - idle_start >= args.idle_timeout_seconds:
                break
            time.sleep(2)
            continue

        idle_start = time.monotonic()
        rows = []
        ack_ids = []
        for received in response.received_messages:
            ack_ids.append(received.ack_id)
            payload = received.message.data.decode("utf-8")
            import json

            event = json.loads(payload)
            prediction = endpoint.predict(
                instances=[
                    {
                        "event_id": event["event_id"],
                        "event_timestamp": event["event_timestamp"],
                        "features": event["features"],
                        "label": event.get("label"),
                        "binary_label": event.get("binary_label"),
                    }
                ]
            ).predictions[0]
            prediction["pipeline_mode"] = "stream"
            prediction["experiment_id"] = args.experiment_id
            prediction.setdefault("inference_timestamp", datetime.now(UTC).isoformat())
            rows.append(prediction)

        errors = bq_client.insert_rows_json(table, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")
        subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": ack_ids})
        total_processed += len(rows)
        print(f"Processed batch: {len(rows)}; total processed: {total_processed}")

    print(f"Processed and acknowledged Pub/Sub messages: {total_processed}")
    print(f"Inserted rows into BigQuery table: {table}")


if __name__ == "__main__":
    main()
