param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [Parameter(Mandatory = $true)][string]$ArtifactBundle,
    [string]$Repository = "rtad-repo",
    [string]$ImageName = "rtad-vertex-predictor",
    [string]$ModelDisplayName = "rtad-isolation-forest",
    [string]$EndpointDisplayName = "rtad-endpoint",
    [string]$MachineType = "n1-standard-2"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PredictorDir = Join-Path $PSScriptRoot "vertex_predictor"
$ImageUri = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ImageName:latest"

gcloud config set project $ProjectId
gcloud artifacts repositories create $Repository --repository-format=docker --location=$Region --description="RTAD images" 2>$null
gcloud storage cp $ArtifactBundle "gs://$BucketName/artifacts/bundle.joblib"
gcloud builds submit $PredictorDir --tag $ImageUri

$modelUpload = gcloud ai models upload `
  --region=$Region `
  --display-name=$ModelDisplayName `
  --container-image-uri=$ImageUri `
  --container-predict-route="/predict" `
  --container-health-route="/health" `
  --container-env-vars="ARTIFACT_URI=gs://$BucketName/artifacts/bundle.joblib" `
  --format="value(name)"

$endpoint = gcloud ai endpoints create `
  --region=$Region `
  --display-name=$EndpointDisplayName `
  --format="value(name)"

gcloud ai endpoints deploy-model $endpoint `
  --region=$Region `
  --model=$modelUpload `
  --display-name="$ModelDisplayName-deployed" `
  --machine-type=$MachineType `
  --traffic-split=0=100

Write-Host "Vertex model deployed."
