[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Region,
    [Parameter(Mandatory = $true)][string]$BucketName,
    [Parameter(Mandatory = $true)][string]$ArtifactBundle,
    [Parameter(Mandatory = $true)][string]$InputJsonl,
    [Parameter(Mandatory = $true)][string]$BigQueryTable,
    [string]$Repository = "rtad-repo",
    [string]$ImageName = "rtad-batch-scorer",
    [string]$JobName = "rtad-batch-score",
    [string]$Environment = "dev",
    [string]$ExperimentId = "batch-cloudrun",
    [int]$BatchSize = 256,
    [string]$Memory = "2Gi",
    [string]$Cpu = "1",
    [string]$TaskTimeout = "1800s",
    [switch]$ExecuteNow,
    [switch]$CreateScheduler,
    [string]$ScheduleName = "rtad-batch-score-daily",
    [string]$Schedule = "0 9 * * *",
    [string]$TimeZone = "America/Los_Angeles",
    [string]$SchedulerServiceAccount
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

$Root = Split-Path -Parent $PSScriptRoot
$BatchDir = Join-Path $PSScriptRoot "batch_job"
$CloudBuildConfig = Join-Path $BatchDir "cloudbuild.yaml"
$ImageUri = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ImageName`:latest"
$ArtifactUri = "gs://$BucketName/artifacts/batch_bundle.joblib"
$InputUri = "gs://$BucketName/batch/input.jsonl"
$OutputUri = "gs://$BucketName/batch/results/$ExperimentId.jsonl"
$labels = Get-LabelString -Component "batch" -Environment $Environment -ExtraLabels @{ experiment = $ExperimentId }

Assert-GcloudReady -ProjectId $ProjectId
Assert-FileExists -Path $ArtifactBundle -Description "Artifact bundle"
Assert-FileExists -Path $InputJsonl -Description "Input JSONL file"
Assert-FileExists -Path $CloudBuildConfig -Description "Cloud Build config"

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

if ($PSCmdlet.ShouldProcess($ArtifactUri, "Upload batch artifact bundle")) {
    & gcloud storage cp $ArtifactBundle $ArtifactUri
}

if ($PSCmdlet.ShouldProcess($InputUri, "Upload batch input JSONL")) {
    & gcloud storage cp $InputJsonl $InputUri
}

if ($PSCmdlet.ShouldProcess($ImageUri, "Build and push Cloud Run batch scorer image")) {
    & gcloud builds submit $Root --config=$CloudBuildConfig --substitutions="_IMAGE_URI=$ImageUri"
}

$envVars = "ARTIFACT_URI=$ArtifactUri,INPUT_URI=$InputUri,OUTPUT_URI=$OutputUri,BIGQUERY_TABLE=$BigQueryTable,BATCH_SIZE=$BatchSize,EXPERIMENT_ID=$ExperimentId"

if (Test-CloudRunJobExists -JobName $JobName -Region $Region) {
    if ($PSCmdlet.ShouldProcess($JobName, "Update Cloud Run Job")) {
        & gcloud run jobs update $JobName `
            --region=$Region `
            --image=$ImageUri `
            --set-env-vars=$envVars `
            --update-labels=$labels `
            --memory=$Memory `
            --cpu=$Cpu `
            --task-timeout=$TaskTimeout `
            --max-retries=1 `
            --tasks=1
    }
} elseif ($PSCmdlet.ShouldProcess($JobName, "Create Cloud Run Job")) {
    & gcloud run jobs create $JobName `
        --region=$Region `
        --image=$ImageUri `
        --set-env-vars=$envVars `
        --labels=$labels `
        --memory=$Memory `
        --cpu=$Cpu `
        --task-timeout=$TaskTimeout `
        --max-retries=1 `
        --tasks=1
}

if ($ExecuteNow -and $PSCmdlet.ShouldProcess($JobName, "Execute Cloud Run Job now")) {
    & gcloud run jobs execute $JobName --region=$Region --wait
}

if ($CreateScheduler) {
    if ([string]::IsNullOrWhiteSpace($SchedulerServiceAccount)) {
        throw "SchedulerServiceAccount is required when -CreateScheduler is used."
    }

    $runJobUri = "https://run.googleapis.com/v2/projects/$ProjectId/locations/$Region/jobs/$JobName`:run"
    if ($PSCmdlet.ShouldProcess($ScheduleName, "Create or update Cloud Scheduler trigger")) {
        $existingSchedule = & gcloud scheduler jobs describe $ScheduleName --location=$Region --format="value(name)" 2>$null
        if ([string]::IsNullOrWhiteSpace($existingSchedule)) {
            & gcloud scheduler jobs create http $ScheduleName `
                --location=$Region `
                --schedule=$Schedule `
                --time-zone=$TimeZone `
                --uri=$runJobUri `
                --http-method=POST `
                --oauth-service-account-email=$SchedulerServiceAccount `
                --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" `
                --description="Runs the RTAD batch scorer Cloud Run Job."
        } else {
            & gcloud scheduler jobs update http $ScheduleName `
                --location=$Region `
                --schedule=$Schedule `
                --time-zone=$TimeZone `
                --uri=$runJobUri `
                --http-method=POST `
                --oauth-service-account-email=$SchedulerServiceAccount `
                --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" `
                --description="Runs the RTAD batch scorer Cloud Run Job."
        }
    }
}

Write-Host "Cloud Run batch job is ready: $JobName"
Write-Host "Artifact URI: $ArtifactUri"
Write-Host "Input URI: $InputUri"
Write-Host "Output URI: $OutputUri"
Write-Host "Execute manually with:"
Write-Host "gcloud run jobs execute $JobName --region=$Region --wait"
