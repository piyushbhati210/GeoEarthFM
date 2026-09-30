# ============================================================
# GeoEarthFM - Project Folder Organization
# SAFE CLEANUP VERSION
# ============================================================

$Root = Get-Location

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "        GEOEARTHFM PROJECT ORGANIZATION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# 1. Create required folders
# ------------------------------------------------------------

$folders = @(
    "embeddings",
    "models",
    "training",
    "inference",
    "geospatial",
    "notebooks",
    "tasks",
    "tasks\landcover",
    "tasks\change_detection",
    "tasks\flood_detection",
    "tasks\crop_monitoring",
    "dashboard",
    "outputs",
    "outputs\all_test_pairs",
    "outputs\final_analysis",
    "archive"
)

foreach ($folder in $folders) {

    $path = Join-Path $Root $folder

    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "[CREATED] $folder" -ForegroundColor Green
    }
    else {
        Write-Host "[EXISTS ] $folder" -ForegroundColor DarkGray
    }
}

# ------------------------------------------------------------
# 2. Remove Python cache folders
#    Safe: Python automatically recreates them
# ------------------------------------------------------------

Write-Host ""
Write-Host "Cleaning Python cache files..." -ForegroundColor Yellow

Get-ChildItem -Path $Root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "__pycache__" } |
    ForEach-Object {

        Write-Host "[REMOVE ] $($_.FullName)" -ForegroundColor DarkYellow

        Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

# ------------------------------------------------------------
# 3. Move analysis scripts into analysis folder
# ------------------------------------------------------------

$analysisFolder = Join-Path $Root "analysis"

if (!(Test-Path $analysisFolder)) {
    New-Item -ItemType Directory -Path $analysisFolder -Force | Out-Null
    Write-Host "[CREATED] analysis" -ForegroundColor Green
}

$analysisScripts = @(
    "analyze_final_results.py",
    "inspect_data.py"
)

foreach ($file in $analysisScripts) {

    $source = Join-Path $Root $file
    $destination = Join-Path $analysisFolder $file

    if (Test-Path $source) {

        if (!(Test-Path $destination)) {
            Move-Item $source $destination
            Write-Host "[MOVED  ] $file -> analysis\" -ForegroundColor Green
        }
        else {
            Write-Host "[SKIP   ] $file already exists in analysis\" -ForegroundColor DarkGray
        }
    }
}

# ------------------------------------------------------------
# 4. Move evaluation scripts into evaluation folder
# ------------------------------------------------------------

$evaluationFolder = Join-Path $Root "evaluation"

if (!(Test-Path $evaluationFolder)) {
    New-Item -ItemType Directory -Path $evaluationFolder -Force | Out-Null
    Write-Host "[CREATED] evaluation" -ForegroundColor Green
}

$evaluationScripts = @(
    "evaluate_all_test_pairs.py",
    "evaluate_change.py",
    "evaluate_landcover.py"
)

foreach ($file in $evaluationScripts) {

    $source = Join-Path $Root $file
    $destination = Join-Path $evaluationFolder $file

    if (Test-Path $source) {

        if (!(Test-Path $destination)) {
            Move-Item $source $destination
            Write-Host "[MOVED  ] $file -> evaluation\" -ForegroundColor Green
        }
        else {
            Write-Host "[SKIP   ] $file already exists in evaluation\" -ForegroundColor DarkGray
        }
    }
}

# ------------------------------------------------------------
# 5. Move visualization scripts into visualization folder
# ------------------------------------------------------------

$visualFolder = Join-Path $Root "visualization"

if (!(Test-Path $visualFolder)) {
    New-Item -ItemType Directory -Path $visualFolder -Force | Out-Null
    Write-Host "[CREATED] visualization" -ForegroundColor Green
}

$visualScripts = @(
    "plot_confusion_matrix.py",
    "plot_final_results.py"
)

foreach ($file in $visualScripts) {

    $source = Join-Path $Root $file
    $destination = Join-Path $visualFolder $file

    if (Test-Path $source) {

        if (!(Test-Path $destination)) {
            Move-Item $source $destination
            Write-Host "[MOVED  ] $file -> visualization\" -ForegroundColor Green
        }
        else {
            Write-Host "[SKIP   ] $file already exists in visualization\" -ForegroundColor DarkGray
        }
    }
}

# ------------------------------------------------------------
# 6. Move prediction script into inference
# ------------------------------------------------------------

$predictionScript = Join-Path $Root "predict_change.py"
$predictionDestination = Join-Path $Root "inference\predict_change.py"

if (Test-Path $predictionScript) {

    if (!(Test-Path $predictionDestination)) {
        Move-Item $predictionScript $predictionDestination
        Write-Host "[MOVED  ] predict_change.py -> inference\" -ForegroundColor Green
    }
}

# ------------------------------------------------------------
# 7. Move notes into docs
# ------------------------------------------------------------

$docsFolder = Join-Path $Root "docs"

if (!(Test-Path $docsFolder)) {
    New-Item -ItemType Directory -Path $docsFolder -Force | Out-Null
    Write-Host "[CREATED] docs" -ForegroundColor Green
}

$notes = Join-Path $Root "geo notes.txt"

if (Test-Path $notes) {

    $destination = Join-Path $docsFolder "geo_notes.txt"

    if (!(Test-Path $destination)) {
        Move-Item $notes $destination
        Write-Host "[MOVED  ] geo notes.txt -> docs\" -ForegroundColor Green
    }
}

# ------------------------------------------------------------
# 8. Remove EMPTY duplicate datasets folder
# ------------------------------------------------------------

$emptyDatasets = Join-Path $Root "datasets"

if (Test-Path $emptyDatasets) {

    $items = Get-ChildItem $emptyDatasets -Force

    if ($items.Count -eq 0) {

        Remove-Item $emptyDatasets -Force

        Write-Host "[REMOVED] Empty datasets folder" -ForegroundColor Green
    }
    else {

        Write-Host "[KEPT   ] datasets is not empty" -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------
# 9. DO NOT automatically delete app/
#    It may contain code that is still needed.
# ------------------------------------------------------------

Write-Host ""
Write-Host "[INFO   ] app/ was NOT deleted." -ForegroundColor Yellow
Write-Host "[INFO   ] dashboard/ remains the main dashboard folder." -ForegroundColor Yellow

# ------------------------------------------------------------
# 10. Create .gitignore if missing
# ------------------------------------------------------------

$gitignore = Join-Path $Root ".gitignore"

if (!(Test-Path $gitignore)) {

@"
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Large ML files
*.pt
*.pth
*.ckpt
*.safetensors

# Generated outputs
outputs/

# Generated embeddings
embeddings/

# Temporary files
*.tmp
*.log
"@ | Set-Content $gitignore

    Write-Host "[CREATED] .gitignore" -ForegroundColor Green
}

# ------------------------------------------------------------
# 11. Final structure
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "             ORGANIZATION COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Important folders:" -ForegroundColor White
Write-Host ""
Write-Host "dataset/              -> STTORM dataset"
Write-Host "geoearthfm/           -> core package"
Write-Host "models/               -> model files"
Write-Host "training/             -> training code"
Write-Host "inference/            -> prediction code"
Write-Host "evaluation/           -> evaluation code"
Write-Host "analysis/             -> result analysis"
Write-Host "visualization/        -> plots"
Write-Host "geospatial/           -> geospatial processing"
Write-Host "tasks/                -> downstream tasks"
Write-Host "embeddings/           -> GeoAI embeddings"
Write-Host "dashboard/            -> Streamlit dashboard"
Write-Host "outputs/              -> experiment results"
Write-Host "docs/                 -> notes/documentation"
Write-Host "archive/              -> old/unused files"
Write-Host ""

Write-Host "STTORM dataset and outputs were NOT deleted." -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan