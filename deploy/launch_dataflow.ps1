param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [Parameter(Mandatory = $true)][string]$InputTopic,
    [Parameter(Mandatory = $true)][string]$BigQueryTable,
    [Parameter(Mandatory = $true)][string]$EndpointId,
    [string]$Runner = "DataflowRunner",
    [string]$JobName = "rtad-streaming"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Requirements = Join-Path $Root "requirements-gcp.txt"
$Pipeline = Join-Path $PSScriptRoot "dataflow_job.py"

& "$Root\.venv\Scripts\python.exe" $Pipeline `
  --runner=$Runner `
  --project=$ProjectId `
  --region=$Region `
  --temp_location="gs://$BucketName/dataflow-temp" `
  --staging_location="gs://$BucketName/dataflow-staging" `
  --input_topic=$InputTopic `
  --bigquery_table=$BigQueryTable `
  --endpoint_id=$EndpointId `
  --requirements_file=$Requirements `
  --job_name=$JobName
