[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [Parameter(Mandatory = $true)][string]$InputTopic,
    [Parameter(Mandatory = $true)][string]$BigQueryTable,
    [Parameter(Mandatory = $true)][string]$EndpointId,
    [string]$Runner = "DataflowRunner",
    [string]$JobName = "rtad-streaming",
    [string]$Environment = "dev",
    [string]$ExperimentId = "stream-gcp",
    [int]$NumWorkers = 1,
    [int]$MaxNumWorkers = 1,
    [ValidateSet("NONE", "THROUGHPUT_BASED")][string]$AutoscalingAlgorithm = "NONE",
    [string]$WorkerZone
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

$Root = Split-Path -Parent $PSScriptRoot
$Requirements = Join-Path $Root "requirements-gcp.txt"
$Pipeline = Join-Path $PSScriptRoot "dataflow_job.py"
$Python = Assert-LocalPythonExists -Root $Root
$dataflowLabels = @{
    project = "rtad"
    environment = $Environment.ToLowerInvariant()
    component = "dataflow"
    experiment = $ExperimentId.ToLowerInvariant()
} | ConvertTo-Json -Compress

Assert-GcloudReady -ProjectId $ProjectId
Assert-FileExists -Path $Requirements -Description "GCP requirements file"
Assert-FileExists -Path $Pipeline -Description "Dataflow pipeline"

$pythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pythonVersion -notin @("3.10", "3.11", "3.12")) {
    throw "Dataflow/Apache Beam should be launched from Python 3.10, 3.11, or 3.12. Current .venv uses Python $pythonVersion. Recreate .venv with Python 3.12 before launching Dataflow."
}

if (-not (Test-GcsBucketExists -BucketName $BucketName)) {
    throw "Bucket gs://$BucketName does not exist. Run deploy\setup_gcp.ps1 first."
}

if ($InputTopic -notmatch "^projects/[^/]+/topics/.+$") {
    Write-Host "InputTopic is not a fully qualified Pub/Sub topic. Dataflow usually expects projects/<project>/topics/<topic>."
}

if ($BigQueryTable -notmatch "^[^:]+:[^.]+\..+$") {
    Write-Host "BigQueryTable is not in project:dataset.table form. Dataflow BigQuery writes usually expect that format."
}

$args = @(
    $Pipeline,
    "--runner=$Runner",
    "--project=$ProjectId",
    "--region=$Region",
    "--temp_location=gs://$BucketName/dataflow-temp",
    "--staging_location=gs://$BucketName/dataflow-staging",
    "--input_topic=$InputTopic",
    "--bigquery_table=$BigQueryTable",
    "--endpoint_id=$EndpointId",
    "--experiment_id=$ExperimentId",
    "--requirements_file=$Requirements",
    "--job_name=$JobName",
    "--labels=$dataflowLabels",
    "--num_workers=$NumWorkers",
    "--max_num_workers=$MaxNumWorkers",
    "--autoscaling_algorithm=$AutoscalingAlgorithm"
)

if (-not [string]::IsNullOrWhiteSpace($WorkerZone)) {
    $args += "--worker_zone=$WorkerZone"
}

if ($PSCmdlet.ShouldProcess($JobName, "Launch Dataflow streaming job")) {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Dataflow launch failed. Install requirements-gcp.txt and fix the error above before publishing more events."
    }
}

Write-Host "Dataflow launch submitted for job: $JobName"
Write-Host "Stop it after testing to avoid ongoing streaming charges:"
Write-Host "gcloud dataflow jobs list --region=$Region --filter=`"name=$JobName`""
Write-Host "gcloud dataflow jobs cancel <JOB_ID> --region=$Region"
