[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [string]$BucketName,
    [string]$DataflowJobName = "rtad-streaming",
    [string]$CloudRunJobName = "rtad-batch-score",
    [string]$EndpointDisplayName = "rtad-endpoint",
    [string]$TopicName = "rtad-events",
    [string]$SubscriptionName = "rtad-events-sub",
    [string]$DatasetName = "rtad",
    [switch]$DeleteCloudRunJob,
    [switch]$DeleteVertexEndpoint,
    [switch]$DeletePubSub,
    [switch]$DeleteBucketContents,
    [switch]$DeleteBigQueryTables
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

Assert-GcloudReady -ProjectId $ProjectId

$dataflowJobs = & gcloud dataflow jobs list `
    --region=$Region `
    --status=active `
    --filter="name=$DataflowJobName" `
    --format="value(id)"

foreach ($jobId in $dataflowJobs) {
    if (-not [string]::IsNullOrWhiteSpace($jobId) -and $PSCmdlet.ShouldProcess($jobId, "Cancel Dataflow job")) {
        & gcloud dataflow jobs cancel $jobId --region=$Region
    }
}

if ($DeleteCloudRunJob -and (Test-CloudRunJobExists -JobName $CloudRunJobName -Region $Region)) {
    if ($PSCmdlet.ShouldProcess($CloudRunJobName, "Delete Cloud Run Job")) {
        & gcloud run jobs delete $CloudRunJobName --region=$Region --quiet
    }
}

if ($DeleteVertexEndpoint) {
    $endpoint = & gcloud ai endpoints list `
        --region=$Region `
        --filter="displayName=$EndpointDisplayName" `
        --format="value(name)" `
        --limit=1

    if (-not [string]::IsNullOrWhiteSpace($endpoint)) {
        $deployedModels = & gcloud ai endpoints describe $endpoint `
            --region=$Region `
            --format="value(deployedModels.id)"

        foreach ($deployedModel in $deployedModels) {
            if (-not [string]::IsNullOrWhiteSpace($deployedModel) -and $PSCmdlet.ShouldProcess($endpoint, "Undeploy model $deployedModel")) {
                & gcloud ai endpoints undeploy-model $endpoint --region=$Region --deployed-model-id=$deployedModel --quiet
            }
        }

        if ($PSCmdlet.ShouldProcess($endpoint, "Delete Vertex endpoint")) {
            & gcloud ai endpoints delete $endpoint --region=$Region --quiet
        }
    }
}

if ($DeletePubSub) {
    if (Test-PubSubSubscriptionExists -SubscriptionName $SubscriptionName) {
        if ($PSCmdlet.ShouldProcess($SubscriptionName, "Delete Pub/Sub subscription")) {
            & gcloud pubsub subscriptions delete $SubscriptionName --quiet
        }
    }

    if (Test-PubSubTopicExists -TopicName $TopicName) {
        if ($PSCmdlet.ShouldProcess($TopicName, "Delete Pub/Sub topic")) {
            & gcloud pubsub topics delete $TopicName --quiet
        }
    }
}

if ($DeleteBucketContents) {
    if ([string]::IsNullOrWhiteSpace($BucketName)) {
        throw "BucketName is required when -DeleteBucketContents is used."
    }
    if (Test-GcsBucketExists -BucketName $BucketName) {
        if ($PSCmdlet.ShouldProcess("gs://$BucketName/artifacts gs://$BucketName/dataflow-* gs://$BucketName/batch", "Delete RTAD GCS objects")) {
            & gcloud storage rm --recursive "gs://$BucketName/artifacts/**" "gs://$BucketName/dataflow-temp/**" "gs://$BucketName/dataflow-staging/**" "gs://$BucketName/batch/**" 2>$null
        }
    }
}

if ($DeleteBigQueryTables) {
    Assert-BqReady
    foreach ($table in @("prediction_results", "staged_events")) {
        if (Test-BqTableExists -ProjectId $ProjectId -DatasetName $DatasetName -TableName $table) {
            if ($PSCmdlet.ShouldProcess("$ProjectId`:$DatasetName.$table", "Delete BigQuery table")) {
                & bq rm -f -t "$ProjectId`:$DatasetName.$table"
            }
        }
    }
}

Write-Host "Cleanup complete. By default this script cancels running Dataflow jobs only."
Write-Host "Use explicit delete switches for Cloud Run, Vertex, Pub/Sub, GCS objects, or BigQuery tables."
