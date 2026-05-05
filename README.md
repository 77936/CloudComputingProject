# Real-Time Anomaly Detection

This project trains an unsupervised anomaly detection model locally on CICIDS2017
and then deploys the finalized artifacts to GCP for streaming and micro-batch
evaluation.

## Local workflow

1. Create or activate the local virtual environment.
2. Install the core dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

3. Train and export artifacts:

```powershell
.venv\Scripts\python.exe train.py --dataset-path .\dataset --output-dir .\artifacts --sample-fraction 0.05 --max-benign-train 50000 --export-sample-size 500
```

4. Run a local streaming simulation:

```powershell
.venv\Scripts\python.exe simulate.py --artifact-bundle .\artifacts\<run-id>\bundle.joblib --input-jsonl .\artifacts\<run-id>\test_events.jsonl --destination local --output-jsonl .\artifacts\<run-id>\stream_results.jsonl
```

5. Run the micro-batch comparison:

```powershell
.venv\Scripts\python.exe score_batch.py --artifact-bundle .\artifacts\<run-id>\bundle.joblib --input-jsonl .\artifacts\<run-id>\test_events.jsonl --output-jsonl .\artifacts\<run-id>\batch_results.jsonl --batch-size 256
```

6. Compare results:

```powershell
.venv\Scripts\python.exe evaluate_results.py --artifact-dir .\artifacts\<run-id> --stream-results .\artifacts\<run-id>\stream_results.jsonl --batch-results .\artifacts\<run-id>\batch_results.jsonl
```

## Repository layout

- `rtad/`: local training, inference, simulation, and evaluation code
- `deploy/`: GCP setup, Vertex deployment, and Dataflow scaffolding
- `tests/`: unit and smoke tests for the local path
- `train.py`, `simulate.py`, `score_batch.py`, `evaluate_results.py`: CLI entrypoints

## GCP deployment notes

Train locally first. The uploaded artifact bundle is the source of truth for both
Vertex serving and the batch comparison path. The PowerShell scripts under
`deploy/` create or launch the required GCP resources, but they assume that:

- `gcloud` is installed and authenticated
- billing is enabled for the target project
- the artifact bundle already exists locally

### GCP deployment run order

Set these values first:

```powershell
$PROJECT_ID = "your-gcp-project-id"
$REGION = "us-central1"
$BUCKET = "your-unique-rtad-bucket"
$RUN_ID = "20260428-095957"
$BUNDLE = ".\artifacts\$RUN_ID\bundle.joblib"
$EVENTS = ".\artifacts\$RUN_ID\test_events.jsonl"
$BQ_RESULTS = "$PROJECT_ID`:rtad.prediction_results"
$BQ_RESULTS_SQL = "$PROJECT_ID.rtad.prediction_results"
$TOPIC = "projects/$PROJECT_ID/topics/rtad-events"
```

1. Create the shared cloud resources:

```powershell
.\deploy\setup_gcp.ps1 -ProjectId $PROJECT_ID -Region $REGION -BucketName $BUCKET
```

This enables the required GCP APIs and creates the bucket, Pub/Sub topic and
subscription, BigQuery dataset, and BigQuery tables. Run this first after
confirming billing is enabled.

2. Deploy the Vertex AI prediction endpoint:

```powershell
.\deploy\deploy_vertex.ps1 -ProjectId $PROJECT_ID -Region $REGION -BucketName $BUCKET -ArtifactBundle $BUNDLE
```

This uploads the model bundle, builds the predictor image, uploads a Vertex AI
model, creates or reuses the endpoint, and deploys the model. Save the printed
endpoint ID for the Dataflow command.

3. Launch the streaming Dataflow path:

```powershell
.\deploy\launch_dataflow.ps1 `
  -ProjectId $PROJECT_ID `
  -Region $REGION `
  -BucketName $BUCKET `
  -InputTopic $TOPIC `
  -BigQueryTable $BQ_RESULTS `
  -EndpointId "PASTE_ENDPOINT_ID"
```

This starts the Pub/Sub -> Dataflow -> Vertex AI -> BigQuery pipeline. It can
keep billing while running, so cancel it after the demo or smoke test.

4. Publish a small sample event:

```powershell
$sample = Get-Content .\artifacts\$RUN_ID\test_events.jsonl -TotalCount 1
gcloud pubsub topics publish rtad-events --message="$sample"
```

Then verify that a row appears in BigQuery:

```powershell
bq query --use_legacy_sql=false "SELECT * FROM ``$BQ_RESULTS_SQL`` ORDER BY inference_timestamp DESC LIMIT 5"
```

5. Create or run the Cloud Run batch job:

```powershell
.\deploy\schedule_batch_job.ps1 `
  -ProjectId $PROJECT_ID `
  -Region $REGION `
  -BucketName $BUCKET `
  -ArtifactBundle $BUNDLE `
  -InputJsonl $EVENTS `
  -BigQueryTable $BQ_RESULTS `
  -ExecuteNow
```

This builds the batch scorer container, uploads the bundle and input JSONL to
GCS, creates or updates a Cloud Run Job, and optionally executes it. The job
uploads batch output to `gs://$BUCKET/batch/results/` and appends compatible
prediction rows to BigQuery. Use `-CreateScheduler` only if you need a recurring
Cloud Scheduler trigger; manual execution is safer for the demo.

6. Capture checkpoint evidence:

- Vertex AI endpoint and deployed model
- Dataflow job graph/status
- Pub/Sub topic and published messages
- BigQuery prediction rows
- Cloud Run Job execution
- GCS artifact/input/output objects
- Billing report grouped by service/SKU and labels

7. Stop expensive resources:

```powershell
.\deploy\cleanup_gcp.ps1 -ProjectId $PROJECT_ID -Region $REGION
```

By default this cancels running Dataflow jobs only. Add explicit switches when
you are ready to delete more:

```powershell
.\deploy\cleanup_gcp.ps1 `
  -ProjectId $PROJECT_ID `
  -Region $REGION `
  -BucketName $BUCKET `
  -DeleteCloudRunJob `
  -DeleteVertexEndpoint `
  -DeletePubSub `
  -DeleteBucketContents
```

### Cost notes

- `setup_gcp.ps1` creates mostly low-cost resources, but BigQuery storage and
  GCS storage can accrue small charges.
- `deploy_vertex.ps1` uses Cloud Build, Artifact Registry, GCS, and Vertex AI.
  A deployed Vertex endpoint can continue charging until undeployed/deleted.
- `launch_dataflow.ps1` starts a streaming Dataflow job. Stop it after testing.
- `schedule_batch_job.ps1` uses Cloud Build, Artifact Registry, GCS, Cloud Run
  Jobs, and optionally BigQuery loads. Costs are easier to isolate because the
  job is finite and labeled.
- The scripts apply labels such as `project=rtad`, `environment=dev`, and
  `component=vertex|dataflow|batch` where supported so Billing reports or
  Billing export to BigQuery can separate the paths.

## Cloud architecture

- Streaming path: `Simulator -> Pub/Sub -> Dataflow -> Vertex AI -> BigQuery`
- Batch path: `GCS input -> Cloud Run Job batch scorer -> GCS output + BigQuery`

Both paths write compatible results so the comparison focuses on latency,
throughput, and operational cost rather than different model behavior.
