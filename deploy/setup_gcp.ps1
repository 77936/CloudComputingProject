[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [string]$DatasetName = "rtad",
    [string]$TopicName = "rtad-events",
    [string]$SubscriptionName = "rtad-events-sub",
    [string]$ResultsTable = "prediction_results",
    [string]$StagingTable = "staged_events",
    [string]$Environment = "dev"
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Assert-GcloudReady -ProjectId $ProjectId
Assert-BqReady

$setupLabels = Get-LabelString -Component "setup" -Environment $Environment
$streamLabels = Get-LabelString -Component "stream" -Environment $Environment
$batchLabels = Get-LabelString -Component "batch" -Environment $Environment

$services = @(
    "pubsub.googleapis.com",
    "dataflow.googleapis.com",
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "cloudscheduler.googleapis.com"
)

if ($PSCmdlet.ShouldProcess($ProjectId, "Enable required GCP APIs")) {
    & gcloud services enable $services
}

if (Test-GcsBucketExists -BucketName $BucketName) {
    Write-Host "Bucket gs://$BucketName already exists."
    if ($PSCmdlet.ShouldProcess("gs://$BucketName", "Update bucket labels")) {
        & gcloud storage buckets update "gs://$BucketName" --update-labels=$setupLabels
    }
} elseif ($PSCmdlet.ShouldProcess("gs://$BucketName", "Create Cloud Storage bucket")) {
    & gcloud storage buckets create "gs://$BucketName" --location=$Region --uniform-bucket-level-access
    & gcloud storage buckets update "gs://$BucketName" --update-labels=$setupLabels
}

if (Test-PubSubTopicExists -TopicName $TopicName) {
    Write-Host "Pub/Sub topic $TopicName already exists."
} elseif ($PSCmdlet.ShouldProcess($TopicName, "Create Pub/Sub topic")) {
    & gcloud pubsub topics create $TopicName --labels=$streamLabels
}

if (Test-PubSubSubscriptionExists -SubscriptionName $SubscriptionName) {
    Write-Host "Pub/Sub subscription $SubscriptionName already exists."
} elseif ($PSCmdlet.ShouldProcess($SubscriptionName, "Create Pub/Sub subscription")) {
    & gcloud pubsub subscriptions create $SubscriptionName --topic=$TopicName --labels=$streamLabels
}

if (Test-BqDatasetExists -ProjectId $ProjectId -DatasetName $DatasetName) {
    Write-Host "BigQuery dataset $ProjectId`:$DatasetName already exists."
    if ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName", "Update dataset labels")) {
        $labelArgs = Get-BqSetLabelArgs -LabelString $setupLabels
        & bq update @labelArgs "$ProjectId`:$DatasetName"
    }
} elseif ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName", "Create BigQuery dataset")) {
    $labelArgs = Get-BqLabelArgs -LabelString $setupLabels
    & bq --location=$Region mk --dataset @labelArgs "$ProjectId`:$DatasetName"
}

$resultsSchema = "event_id:STRING,event_timestamp:TIMESTAMP,inference_timestamp:TIMESTAMP,anomaly_score:FLOAT,anomaly_flag:INTEGER,model_version:STRING,label:STRING,binary_label:INTEGER,pipeline_mode:STRING,experiment_id:STRING"
$stagingSchema = "event_id:STRING,event_timestamp:TIMESTAMP,features:STRING,label:STRING,binary_label:INTEGER,experiment_id:STRING"

if (Test-BqTableExists -ProjectId $ProjectId -DatasetName $DatasetName -TableName $ResultsTable) {
    Write-Host "BigQuery table $ProjectId`:$DatasetName.$ResultsTable already exists."
    if ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName.$ResultsTable", "Update table labels")) {
        $labelArgs = Get-BqSetLabelArgs -LabelString $streamLabels
        & bq update @labelArgs "$ProjectId`:$DatasetName.$ResultsTable"
    }
} elseif ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName.$ResultsTable", "Create BigQuery results table")) {
    $labelArgs = Get-BqLabelArgs -LabelString $streamLabels
    & bq mk --table @labelArgs "$ProjectId`:$DatasetName.$ResultsTable" $resultsSchema
}

if (Test-BqTableExists -ProjectId $ProjectId -DatasetName $DatasetName -TableName $StagingTable) {
    Write-Host "BigQuery table $ProjectId`:$DatasetName.$StagingTable already exists."
    if ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName.$StagingTable", "Update table labels")) {
        $labelArgs = Get-BqSetLabelArgs -LabelString $batchLabels
        & bq update @labelArgs "$ProjectId`:$DatasetName.$StagingTable"
    }
} elseif ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName.$StagingTable", "Create BigQuery staging table")) {
    $labelArgs = Get-BqLabelArgs -LabelString $batchLabels
    & bq mk --table @labelArgs "$ProjectId`:$DatasetName.$StagingTable" $stagingSchema
}

Write-Host "GCP setup complete."
Write-Host "Pub/Sub topic: projects/$ProjectId/topics/$TopicName"
Write-Host "BigQuery results table: $ProjectId`:$DatasetName.$ResultsTable"
