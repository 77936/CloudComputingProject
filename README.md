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

## Cloud architecture

- Streaming path: `Simulator -> Pub/Sub -> Dataflow -> Vertex AI -> BigQuery`
- Batch path: staged events -> scheduled scorer -> BigQuery

Both paths write compatible results so the comparison focuses on latency,
throughput, and operational cost rather than different model behavior.
