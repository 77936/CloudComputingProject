$ErrorActionPreference = "Stop"

function Assert-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH. Install it before running this script."
    }
}

function Assert-GcloudReady {
    param([Parameter(Mandatory = $true)][string]$ProjectId)

    Assert-CommandAvailable -Name "gcloud"
    $account = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($account)) {
        throw "gcloud is not authenticated. Run 'gcloud auth login' before using these scripts."
    }

    & gcloud config set project $ProjectId | Out-Null
}

function Assert-BqReady {
    Assert-CommandAvailable -Name "bq"
}

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found at '$Path'."
    }
}

function Assert-LocalPythonExists {
    param([Parameter(Mandatory = $true)][string]$Root)

    $python = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Local Python was not found at '$python'. Create the venv and install requirements before launching Dataflow locally."
    }
    return $python
}

function Get-LabelString {
    param(
        [Parameter(Mandatory = $true)][string]$Component,
        [string]$Environment = "dev",
        [hashtable]$ExtraLabels = @{}
    )

    $labels = [ordered]@{
        project = "rtad"
        environment = $Environment.ToLowerInvariant()
        component = $Component.ToLowerInvariant()
    }

    foreach ($key in $ExtraLabels.Keys) {
        if (-not [string]::IsNullOrWhiteSpace($ExtraLabels[$key])) {
            $labels[$key.ToLowerInvariant()] = "$($ExtraLabels[$key])".ToLowerInvariant()
        }
    }

    return (($labels.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ",")
}

function Get-BqLabelArgs {
    param([Parameter(Mandatory = $true)][string]$LabelString)

    return ($LabelString -split "," | ForEach-Object { "--label=$($_ -replace '=', ':')" })
}

function Get-BqSetLabelArgs {
    param([Parameter(Mandatory = $true)][string]$LabelString)

    return ($LabelString -split "," | ForEach-Object { "--set_label=$($_ -replace '=', ':')" })
}

function Test-GcsBucketExists {
    param([Parameter(Mandatory = $true)][string]$BucketName)
    & gcloud storage buckets describe "gs://$BucketName" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-PubSubTopicExists {
    param([Parameter(Mandatory = $true)][string]$TopicName)
    & gcloud pubsub topics describe $TopicName *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-PubSubSubscriptionExists {
    param([Parameter(Mandatory = $true)][string]$SubscriptionName)
    & gcloud pubsub subscriptions describe $SubscriptionName *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-BqDatasetExists {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectId,
        [Parameter(Mandatory = $true)][string]$DatasetName
    )
    & bq show --format=none "$ProjectId`:$DatasetName" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-BqTableExists {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectId,
        [Parameter(Mandatory = $true)][string]$DatasetName,
        [Parameter(Mandatory = $true)][string]$TableName
    )
    & bq show --format=none "$ProjectId`:$DatasetName.$TableName" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-ArtifactRepositoryExists {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Region
    )
    & gcloud artifacts repositories describe $Repository --location=$Region *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-CloudRunJobExists {
    param(
        [Parameter(Mandatory = $true)][string]$JobName,
        [Parameter(Mandatory = $true)][string]$Region
    )
    & gcloud run jobs describe $JobName --region=$Region *> $null
    return ($LASTEXITCODE -eq 0)
}

function Convert-GcsUriToHttps {
    param([Parameter(Mandatory = $true)][string]$GcsUri)

    if ($GcsUri -notmatch "^gs://([^/]+)/(.+)$") {
        throw "Expected a GCS URI like gs://bucket/path, got '$GcsUri'."
    }
    return "https://storage.googleapis.com/$($Matches[1])/$($Matches[2])"
}
