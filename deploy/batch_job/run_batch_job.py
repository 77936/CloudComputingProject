from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from google.cloud import bigquery, storage

from rtad.batch import score_batch_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud Run Job wrapper for RTAD batch scoring.")
    parser.add_argument("--artifact-uri", default=os.environ.get("ARTIFACT_URI"), required=False)
    parser.add_argument("--input-uri", default=os.environ.get("INPUT_URI"), required=False)
    parser.add_argument("--output-uri", default=os.environ.get("OUTPUT_URI"), required=False)
    parser.add_argument("--bigquery-table", default=os.environ.get("BIGQUERY_TABLE"), required=False)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "256")))
    parser.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", "batch-cloudrun"))
    return parser


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri or not uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI like gs://bucket/path, got {uri!r}")
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def download_gcs(uri: str, destination: Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)


def upload_gcs(source: Path, uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(source)


def load_jsonl_to_bigquery(source: Path, table: str) -> None:
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    with source.open("rb") as handle:
        job = client.load_table_from_file(handle, table, job_config=job_config)
    job.result()


def normalize_bigquery_table(table: str) -> str:
    if ":" not in table:
        return table
    project, rest = table.split(":", 1)
    return f"{project}.{rest}"


def main() -> None:
    args = build_parser().parse_args()
    missing = [
        name
        for name, value in {
            "artifact-uri": args.artifact_uri,
            "input-uri": args.input_uri,
            "output-uri": args.output_uri,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required setting(s): {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        artifact_path = work / "bundle.joblib"
        input_path = work / "input.jsonl"
        output_path = work / "batch_results.jsonl"

        download_gcs(args.artifact_uri, artifact_path)
        download_gcs(args.input_uri, input_path)

        scored = score_batch_file(
            bundle_path=artifact_path,
            input_jsonl=input_path,
            output_jsonl=output_path,
            batch_size=args.batch_size,
            experiment_id=args.experiment_id,
        )

        upload_gcs(output_path, args.output_uri)
        if args.bigquery_table:
            load_jsonl_to_bigquery(output_path, normalize_bigquery_table(args.bigquery_table))

    print(f"Scored events: {scored}")
    print(f"Output uploaded to: {args.output_uri}")
    if args.bigquery_table:
        print(f"Output appended to BigQuery table: {args.bigquery_table}")


if __name__ == "__main__":
    main()
