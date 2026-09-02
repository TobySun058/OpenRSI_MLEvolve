param(
    [string]$Task = "denoising-dirty-documents",
    [string]$DatasetDir = "E:\mle-bench-data",
    [string]$PythonExe = "C:\Users\tobys\miniconda\envs\mlevolve-frontis\python.exe",
    [string]$MlebenchExe = "C:\Users\tobys\miniconda\envs\mlevolve-frontis\Scripts\mlebench.exe",
    [string]$BaseUrl = "http://127.0.0.1:11434/v1",
    [string]$ApiKey = "EMPTY",
    [string]$FrontisModel = "frontis-ma1-30b-chat:latest",
    [string]$QwenModel = "qwen3-30b-chat:latest",
    [int]$Steps = 6,
    [int]$InitialDrafts = 1,
    [int]$Seed = 42,
    [int]$TimeLimit = 3600,
    [int]$Timeout = 900
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $root "runs\comparison_launches\$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prepareLog = Join-Path $logDir "prepare.log"
$frontisLog = Join-Path $logDir "frontis.log"
$qwenLog = Join-Path $logDir "qwen.log"
$metaLog = Join-Path $logDir "launcher.txt"

"Started: $(Get-Date -Format o)" | Out-File -FilePath $metaLog -Encoding utf8
"Task: $Task" | Out-File -FilePath $metaLog -Encoding utf8 -Append
"DatasetDir: $DatasetDir" | Out-File -FilePath $metaLog -Encoding utf8 -Append
"BaseUrl: $BaseUrl" | Out-File -FilePath $metaLog -Encoding utf8 -Append
"FrontisModel: $FrontisModel" | Out-File -FilePath $metaLog -Encoding utf8 -Append
"QwenModel: $QwenModel" | Out-File -FilePath $metaLog -Encoding utf8 -Append

$env:MLEVOLVE_ALLOW_PROMPT_TOOL_FALLBACK = "1"

& $MlebenchExe prepare --competition-id $Task --data-dir $DatasetDir --skip-verification *>&1 |
    Tee-Object -FilePath $prepareLog
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonExe (Join-Path $root "scripts\run_native_real_task.py") `
    --task $Task `
    --dataset-dir $DatasetDir `
    --model $FrontisModel `
    --base-url $BaseUrl `
    --api-key $ApiKey `
    --steps $Steps `
    --initial-drafts $InitialDrafts `
    --seed $Seed `
    --time-limit $TimeLimit `
    --timeout $Timeout `
    --allow-prompt-tool-fallback *>&1 |
    Tee-Object -FilePath $frontisLog
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonExe (Join-Path $root "scripts\run_native_real_task.py") `
    --task $Task `
    --dataset-dir $DatasetDir `
    --model $QwenModel `
    --base-url $BaseUrl `
    --api-key $ApiKey `
    --steps $Steps `
    --initial-drafts $InitialDrafts `
    --seed $Seed `
    --time-limit $TimeLimit `
    --timeout $Timeout `
    --allow-prompt-tool-fallback *>&1 |
    Tee-Object -FilePath $qwenLog

exit $LASTEXITCODE
