[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [Parameter(Mandatory = $true)][string]$ArtifactBundle,
    [string]$Repository = "rtad-repo",
    [string]$ImageName = "rtad-vertex-predictor",
    [string]$ModelDisplayName = "rtad-isolation-forest",
    [string]$EndpointDisplayName = "rtad-endpoint",
    [string]$MachineType = "n1-standard-2",
    [string]$Environment = "dev",
    [string]$ExperimentId = "vertex-demo"
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

$Root = Split-Path -Parent $PSScriptRoot
$PredictorDir = Join-Path $PSScriptRoot "vertex_predictor"
$ImageUri = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ImageName`:latest"
$ArtifactUri = "gs://$BucketName/artifacts/bundle.joblib"
$labels = Get-LabelString -Component "vertex" -Environment $Environment -ExtraLabels @{ experiment = $ExperimentId }

Assert-GcloudReady -ProjectId $ProjectId
Assert-CommandAvailable -Name "gcloud"
Assert-FileExists -Path $ArtifactBundle -Description "Artifact bundle"

if (-not (Test-Path -LiteralPath $PredictorDir -PathType Container)) {
    throw "Vertex predictor directory was not found at '$PredictorDir'."
}

if (-not (Test-GcsBucketExists -BucketName $BucketName)) {
    throw "Bucket gs://$BucketName does not exist. Run deploy\setup_gcp.ps1 first."
}

if (Test-ArtifactRepositoryExists -Repository $Repository -Region $Region) {
    Write-Host "Artifact Registry repository $Repository already exists."
} elseif ($PSCmdlet.ShouldProcess("$Repository in $Region", "Create Artifact Registry Docker repository")) {
    & gcloud artifacts repositories create $Repository `
        --repository-format=docker `
        --location=$Region `
        --description="RTAD images" `
        --labels=$labels
}

if ($PSCmdlet.ShouldProcess($ArtifactUri, "Upload artifact bundle")) {
    & gcloud storage cp $ArtifactBundle $ArtifactUri
}

if ($PSCmdlet.ShouldProcess($ImageUri, "Build and push Vertex predictor image")) {
    & gcloud builds submit $PredictorDir --tag $ImageUri
}

$modelName = $null
if ($PSCmdlet.ShouldProcess($ModelDisplayName, "Upload Vertex AI model")) {
    $modelName = & gcloud ai models upload `
        --region=$Region `
        --display-name=$ModelDisplayName `
        --container-image-uri=$ImageUri `
        --container-predict-route="/predict" `
        --container-health-route="/health" `
        --container-env-vars="ARTIFACT_URI=$ArtifactUri" `
        --labels=$labels `
        --format="value(name)"
}

if ([string]::IsNullOrWhiteSpace($modelName)) {
    Write-Host "WhatIf mode: Vertex model upload skipped."
    return
}

$endpoint = & gcloud ai endpoints list `
    --region=$Region `
    --filter="displayName=$EndpointDisplayName" `
    --format="value(name)" `
    --limit=1

if ([string]::IsNullOrWhiteSpace($endpoint)) {
    if ($PSCmdlet.ShouldProcess($EndpointDisplayName, "Create Vertex AI endpoint")) {
        $endpoint = & gcloud ai endpoints create `
            --region=$Region `
            --display-name=$EndpointDisplayName `
            --labels=$labels `
            --format="value(name)"
    }
} else {
    Write-Host "Vertex endpoint $EndpointDisplayName already exists: $endpoint"
}

if ([string]::IsNullOrWhiteSpace($endpoint)) {
    throw "Vertex endpoint was not created."
}

if ($PSCmdlet.ShouldProcess($endpoint, "Deploy model $modelName")) {
    & gcloud ai endpoints deploy-model $endpoint `
        --region=$Region `
        --model=$modelName `
        --display-name="$ModelDisplayName-deployed" `
        --machine-type=$MachineType `
        --traffic-split=0=100
}

$endpointId = ($endpoint -split "/")[-1]
Write-Host "Vertex model deployed."
Write-Host "Endpoint resource: $endpoint"
Write-Host "Endpoint ID for launch_dataflow.ps1: $endpointId"
