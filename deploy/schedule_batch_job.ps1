param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [Parameter(Mandatory = $true)][string]$ArtifactBundle,
    [Parameter(Mandatory = $true)][string]$InputJsonl,
    [string]$JobName = "rtad-batch-score"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "Use Cloud Scheduler or a Cloud Run Job to trigger the batch scorer with the same artifact bundle."
Write-Host "Recommended command:"
Write-Host "$Root\.venv\Scripts\python.exe $Root\score_batch.py --artifact-bundle $ArtifactBundle --input-jsonl $InputJsonl --output-jsonl .\batch_results.jsonl"
Write-Host "Upload the produced results to BigQuery or GCS for comparison."
