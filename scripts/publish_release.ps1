<# 
.SYNOPSIS
    Devign Scanner - Build & Publish Release Script (Windows)

.DESCRIPTION
    Packages the scanner and publishes to GitHub Releases.

.PARAMETER Version
    Release version (e.g., v1.0.0)

.EXAMPLE
    .\scripts\publish_release.ps1 -Version v1.0.0
#>

param(
    [string]$Version = "v1.0.0"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $RootDir "build"
$ReleaseDir = Join-Path $BuildDir "release"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Devign Scanner Release Builder" -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Clean previous build
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null
New-Item -ItemType Directory -Path "$ReleaseDir\devign_infer" -Force | Out-Null
New-Item -ItemType Directory -Path "$ReleaseDir\models" -Force | Out-Null

Write-Host ""
Write-Host "[1/5] Copying scanner files..." -ForegroundColor Yellow

# Copy inference library
Copy-Item -Path "$RootDir\devign_pipeline\devign_infer\*" -Destination "$ReleaseDir\devign_infer\" -Recurse

# Copy CLI scanner
Copy-Item -Path "$RootDir\devign_pipeline\devign_scan.py" -Destination "$ReleaseDir\"

# Copy requirements
Copy-Item -Path "$RootDir\requirements-inference.txt" -Destination "$ReleaseDir\requirements.txt"

Write-Host "[2/5] Copying model files..." -ForegroundColor Yellow

# Check for model files
$modelPath = "$RootDir\output\models\best_model.pt"
if (Test-Path $modelPath) {
    Copy-Item -Path $modelPath -Destination "$ReleaseDir\models\"
    Write-Host "  [OK] best_model.pt" -ForegroundColor Green
} else {
    Write-Host "  [!] Warning: best_model.pt not found" -ForegroundColor Red
}

# Check for vocab file
$vocabPaths = @(
    "$RootDir\output\vocab.json",
    "$RootDir\models\vocab.json",
    "$RootDir\devign_pipeline\vocab.json"
)
$vocabFound = $false
foreach ($vocabPath in $vocabPaths) {
    if (Test-Path $vocabPath) {
        Copy-Item -Path $vocabPath -Destination "$ReleaseDir\models\vocab.json"
        Write-Host "  [OK] vocab.json (from $vocabPath)" -ForegroundColor Green
        $vocabFound = $true
        break
    }
}
if (-not $vocabFound) {
    Write-Host "  [!] Warning: vocab.json not found" -ForegroundColor Red
}

# Check for config
$configPath = "$RootDir\output\config.json"
if (Test-Path $configPath) {
    Copy-Item -Path $configPath -Destination "$ReleaseDir\models\"
    Write-Host "  [OK] config.json" -ForegroundColor Green
}

Write-Host "[3/5] Creating release packages..." -ForegroundColor Yellow

# Create zip files
$codeZip = "$BuildDir\devign-scanner-code.zip"
$fullZip = "$BuildDir\devign-scanner-full.zip"
$modelsZip = "$BuildDir\devign-models.zip"

# Code only (without models)
$codeItems = @(
    "$ReleaseDir\devign_infer",
    "$ReleaseDir\devign_scan.py",
    "$ReleaseDir\requirements.txt"
)
Compress-Archive -Path $codeItems -DestinationPath $codeZip -Force
Write-Host "  [OK] devign-scanner-code.zip" -ForegroundColor Green

# Full package
Compress-Archive -Path "$ReleaseDir\*" -DestinationPath $fullZip -Force
Write-Host "  [OK] devign-scanner-full.zip" -ForegroundColor Green

# Models only
if (Test-Path "$ReleaseDir\models\best_model.pt") {
    Compress-Archive -Path "$ReleaseDir\models" -DestinationPath $modelsZip -Force
    Write-Host "  [OK] devign-models.zip" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4/5] Package sizes:" -ForegroundColor Yellow
Get-ChildItem "$BuildDir\*.zip" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  $($_.Name): $sizeMB MB"
}

Write-Host ""
Write-Host "[5/5] Creating GitHub Release..." -ForegroundColor Yellow

# Check if gh is installed
$ghInstalled = Get-Command gh -ErrorAction SilentlyContinue
if (-not $ghInstalled) {
    Write-Host ""
    Write-Host "[!] GitHub CLI (gh) not installed. Skipping release upload." -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual upload instructions:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://github.com/hoangduy0308/C-Vul-Devign/releases/new"
    Write-Host "  2. Create tag: $Version"
    Write-Host "  3. Upload files from: $BuildDir\"
    exit 0
}

# Create release notes
$releaseNotes = @"
## Devign Vulnerability Scanner $Version

### Downloads
- **devign-scanner-code.zip**: Scanner code only (lightweight)
- **devign-scanner-full.zip**: Complete package with trained models
- **devign-models.zip**: Model files only

### Quick Start
```bash
# Download and extract
curl -L https://github.com/hoangduy0308/C-Vul-Devign/releases/download/$Version/devign-scanner-full.zip -o scanner.zip
unzip scanner.zip

# Scan your code
pip install torch numpy tqdm
python release/devign_scan.py scan /path/to/code
```

### GitHub Actions
```yaml
- uses: hoangduy0308/C-Vul-Devign@$Version
  with:
    threshold: '0.5'
```
"@

$releaseNotesPath = "$BuildDir\RELEASE_NOTES.md"
$releaseNotes | Out-File -FilePath $releaseNotesPath -Encoding UTF8

Set-Location $RootDir

# Check if release exists
$releaseExists = $false
try {
    gh release view $Version 2>$null
    $releaseExists = $true
} catch {}

$assets = @($codeZip, $fullZip)
if (Test-Path $modelsZip) {
    $assets += $modelsZip
}

if ($releaseExists) {
    Write-Host "Release $Version already exists. Updating..." -ForegroundColor Yellow
    gh release upload $Version @assets --clobber
} else {
    Write-Host "Creating new release $Version..." -ForegroundColor Yellow
    gh release create $Version @assets --title "Devign Scanner $Version" --notes-file $releaseNotesPath
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "[OK] Release $Version published successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "View at: https://github.com/hoangduy0308/C-Vul-Devign/releases/tag/$Version" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Green
