$src = "C:\Users\SiriRamesh\Documents\Codex\2026-08-19\referenced-chatgpt-conversation-this-is-an\outputs\enterprise-architecture-rag-assistant\src\enterprise_rag"
$dst = "C:\Users\SiriRamesh\Desktop\AgenticAI\MasteringAgenticAIPRogram\Week2\enterprise-architecture-rag-assistant\src\enterprise_rag"

$files = @(
  "answer.py",
  "evaluator.py",
  "vector_store.py",
  "retrieval.py",
  "document_loader.py",
  "index_pipeline.py"
)

if (!(Test-Path $dst)) {
  Write-Host "Destination folder not found: $dst" -ForegroundColor Red
  exit 1
}

$backupRoot = Join-Path $dst "._codex_backup"
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path $backupRoot $stamp
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

foreach ($f in $files) {
  $s = Join-Path $src $f
  $d = Join-Path $dst $f

  if (!(Test-Path $s)) { Write-Host "Missing source: $s" -ForegroundColor Yellow; continue }

  Copy-Item -Path $d -Destination (Join-Path $backupDir $f) -Force
  Copy-Item -Path $s -Destination $d -Force
  Write-Host "Copied: $f" -ForegroundColor Green
}

Write-Host "Done. Backup created: $backupDir" -ForegroundColor Cyan
