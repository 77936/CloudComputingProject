param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [string]$DatasetName = "rtad",
    [string]$TopicName = "rtad-events",
    [string]$SubscriptionName = "rtad-events-sub",
    [string]$ResultsTable = "prediction_results",
    [string]$StagingTable = "staged_events"
)

$ErrorActionPreference = "Stop"

gcloud config set project $ProjectId
gcloud services enable pubsub.googleapis.com dataflow.googleapis.com aiplatform.googleapis.com bigquery.googleapis.com storage.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud storage buckets create "gs://$BucketName" --location=$Region
gcloud pubsub topics create $TopicName
gcloud pubsub subscriptions create $SubscriptionName --topic=$TopicName
bq --location=$Region mk --dataset "$ProjectId`:$DatasetName"
bq mk --table "$ProjectId`:$DatasetName.$ResultsTable" event_id:STRING,event_timestamp:TIMESTAMP,inference_timestamp:TIMESTAMP,anomaly_score:FLOAT,anomaly_flag:INTEGER,model_version:STRING,label:STRING,binary_label:INTEGER,pipeline_mode:STRING,experiment_id:STRING
bq mk --table "$ProjectId`:$DatasetName.$StagingTable" event_id:STRING,event_timestamp:TIMESTAMP,features:STRING,label:STRING,binary_label:INTEGER,experiment_id:STRING

Write-Host "GCP resources created."
