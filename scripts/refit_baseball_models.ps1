<#
.SYNOPSIS
  Re-fit and enable the baseball serving artifacts on this machine.

.DESCRIPTION
  The trained blend (artifacts/baseball_blend.json) and Platt calibrator
  (artifacts/baseball_calibrator.json) are gitignored, so a fresh checkout has
  NEITHER. Until they are fit + enabled, the API silently serves the legacy
  heuristic (mlb-wpct-v1) into the betting-edge engine, i.e. the good model is
  NOT driving the edges. Run this once per machine (and after a season rollover).

  It POSTs the fit endpoints, enables each artifact, then reads back state and,
  if a closing-odds CSV is supplied, runs the CLV / model-vs-market gate.

.PARAMETER Season
  Season to fit on (default 2023, a full completed season).

.PARAMETER BaseUrl
  Sports API base (default http://127.0.0.1:8100, must be running).

.PARAMETER OddsCsv
  Optional path to a historical closing-odds CSV to run the CLV gate.
  See sports/baseball/analytics/odds_ingest.py for accepted formats.

.EXAMPLE
  ./scripts/refit_baseball_models.ps1 -Season 2023
  ./scripts/refit_baseball_models.ps1 -Season 2023 -OddsCsv C:\data\mlb_2023_close.csv
#>
param(
  [int]$Season = 2023,
  [string]$BaseUrl = "http://127.0.0.1:8100",
  [string]$OddsCsv = ""
)

$ErrorActionPreference = "Stop"
$api = "$BaseUrl/api/baseball"

function Invoke-Api($method, $path) {
  Write-Host ">> $method $path" -ForegroundColor DarkCyan
  return Invoke-RestMethod -Method $method -Uri "$api$path" -TimeoutSec 360
}

# 0. Is the API up?
try { Invoke-Api GET "/pipeline/health" | Out-Null }
catch { Write-Host "Sports API not reachable at $BaseUrl. Start it: python -m main (port 8100)." -ForegroundColor Yellow; exit 1 }

Write-Host "`n=== Re-fitting baseball serving artifacts (season $Season) ===" -ForegroundColor Green

# 1. Trained logistic blend (supersedes hand-tuned weights + Platt at serve).
$blend = Invoke-Api POST "/blend/fit?season=$Season&enable=true"
$bstate = if ($blend.state) { $blend.state } else { $blend }
Write-Host ("   blend: enabled={0} beats_hand_tuned={1}" -f $bstate.enabled, $bstate.holdout.blend_beats_hand_tuned) -ForegroundColor Gray

# 2. Platt calibrator (fallback when blend disabled; still a useful check).
$cal = Invoke-Api POST "/calibration/fit?season=$Season&enable=true"
$cstate = Invoke-Api GET "/calibration"
Write-Host ("   calibrator: enabled={0} a={1} b={2}" -f $cstate.enabled, $cstate.params.a, $cstate.params.b) -ForegroundColor Gray

# 3. Read back the serving state.
Write-Host "`n=== Serving state ===" -ForegroundColor Green
$health = Invoke-Api GET "/pipeline/health"
Write-Host ("   serving model: {0} ({1})" -f $health.model.version, $health.model.serving) -ForegroundColor Gray

# 4. The real go-live gate: does the model beat the closing line?
if ($OddsCsv -and (Test-Path $OddsCsv)) {
  Write-Host "`n=== CLV / model-vs-market gate (season $Season) ===" -ForegroundColor Green
  $enc = [uri]::EscapeDataString($OddsCsv)
  $clv = Invoke-Api GET "/clv?season=$Season&odds=$enc"
  if ($clv.available) {
    Write-Host ("   games={0} model_brier={1} market_brier={2} beats_market={3}" -f $clv.games_scored, $clv.model.brier, $clv.market.brier, $clv.model_beats_market) -ForegroundColor Gray
    Write-Host ("   bets={0} flat_roi={1} tradeable={2}" -f $clv.betting.bets, $clv.betting.flat_roi, $clv.gate.tradeable) -ForegroundColor Gray
    $color = if ($clv.gate.tradeable) { "Green" } else { "Yellow" }
    Write-Host ("   VERDICT: {0}" -f $clv.gate.verdict) -ForegroundColor $color
  } else {
    Write-Host ("   CLV not available: {0}" -f $clv.reason) -ForegroundColor Yellow
  }
} else {
  Write-Host "`n(!) No -OddsCsv supplied - skipping the CLV gate." -ForegroundColor Yellow
  Write-Host "    The blend/calibrator only prove the model beats the HOME BASE RATE," -ForegroundColor Yellow
  Write-Host "    NOT the market. Do not trade real money until the CLV gate passes." -ForegroundColor Yellow
}

Write-Host "`nDone." -ForegroundColor Green
