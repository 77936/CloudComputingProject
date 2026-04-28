from __future__ import annotations

import argparse
import json
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataflow pipeline for RTAD streaming inference.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--runner", default="DirectRunner")
    parser.add_argument("--temp_location", required=True)
    parser.add_argument("--staging_location", required=True)
    parser.add_argument("--input_topic", required=True)
    parser.add_argument("--bigquery_table", required=True)
    parser.add_argument("--endpoint_id", required=True)
    parser.add_argument("--experiment_id", default="stream-gcp")
    parser.add_argument("--job_name", default="rtad-streaming")
    parser.add_argument("--requirements_file")
    return parser


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    known_args, pipeline_args = parser.parse_known_args(argv)

    try:
        import apache_beam as beam
        from apache_beam.options.pipeline_options import PipelineOptions
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "apache-beam is not installed. Install requirements-gcp.txt before running the Dataflow job."
        ) from exc

    class ParseMessage(beam.DoFn):
        def process(self, element: bytes) -> Iterable[dict]:
            payload = json.loads(element.decode("utf-8"))
            yield payload

    class MarkForVertex(beam.DoFn):
        def process(self, element: dict) -> Iterable[dict]:
            element["pipeline_mode"] = "stream"
            element["vertex_endpoint_id"] = known_args.endpoint_id
            element["experiment_id"] = known_args.experiment_id
            yield element

    class CallVertexEndpoint(beam.DoFn):
        def setup(self) -> None:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self.session = AuthorizedSession(credentials)
            self.url = (
                f"https://{known_args.region}-aiplatform.googleapis.com/v1/"
                f"projects/{known_args.project}/locations/{known_args.region}/"
                f"endpoints/{known_args.endpoint_id}:predict"
            )

        def process(self, element: dict) -> Iterable[dict]:
            request_payload = {
                "instances": [
                    {
                        "event_id": element["event_id"],
                        "event_timestamp": element["event_timestamp"],
                        "features": element["features"],
                        "label": element.get("label"),
                        "binary_label": element.get("binary_label"),
                    }
                ]
            }
            response = self.session.post(self.url, json=request_payload, timeout=60)
            response.raise_for_status()
            predictions = response.json().get("predictions", [])
            if not predictions:
                return
            prediction = predictions[0]
            prediction["experiment_id"] = element.get("experiment_id")
            prediction["pipeline_mode"] = "stream"
            yield prediction

    options = PipelineOptions(
        pipeline_args,
        runner=known_args.runner,
        project=known_args.project,
        region=known_args.region,
        temp_location=known_args.temp_location,
        staging_location=known_args.staging_location,
        job_name=known_args.job_name,
        save_main_session=True,
    )

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "ReadPubSub" >> beam.io.ReadFromPubSub(topic=known_args.input_topic)
            | "ParseJson" >> beam.ParDo(ParseMessage())
            | "MarkForVertex" >> beam.ParDo(MarkForVertex())
            | "CallVertexEndpoint" >> beam.ParDo(CallVertexEndpoint())
            | "WriteBigQueryResults"
            >> beam.io.WriteToBigQuery(
                known_args.bigquery_table,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
            )
        )


if __name__ == "__main__":
    run()
