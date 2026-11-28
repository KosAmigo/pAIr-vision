# B: 가짜 모델 부하 시험 일괄 실행. 사용: powershell -File tools/capacity_sim/run_b.ps1 -Out <결과파일>
param([string]$Out = "$env:TEMP\run_b.txt", [int]$Duration = 60)
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONPATH = "src"; $env:VISION_CALLBACK_SECRET = "dev-callback-secret"
$py = ".\.venv\Scripts\python.exe"
function Get-Tree($r) { $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId; $ids = @($r); $q = @($r)
  while ($q.Count) { $p = $q[0]; $q = @($q | Select-Object -Skip 1); foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $p })) { $ids += $c.ProcessId; $q += $c.ProcessId } }; $ids }
function Invoke-Case($label, $workers, $streams, $extra) {
  $sink = Start-Process $py -ArgumentList "tools/replay/sink.py --port 9000" -PassThru -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\sink.out" -RedirectStandardError "$env:TEMP\sink.err"
  $env:VISION_API_KEY = "dev-key"; $env:VISION_DETECTOR = "fake"; $env:VISION_MAX_SESSIONS = "100"
  $env:VISION_CALLBACK_URL = "http://127.0.0.1:9000/internal/vision/events"; $env:VISION_WORKERS = "$workers"
  $srv = Start-Process $py -ArgumentList "-m pair_vision" -PassThru -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\srv.out" -RedirectStandardError "$env:TEMP\srv.err"
  Start-Sleep -Seconds (6 + 2 * $workers)
  $ids = (Get-Tree $srv.Id) -join ","
  $raw = & $py tools/capacity_sim/load_both.py --api-key dev-key --streams $streams --duration $Duration --cpu-pids $ids @extra | Out-String
  $j = $raw | ConvertFrom-Json
  $lines = @("== $label | workers=$workers streams=$streams procs=$(($ids -split ',').Count)",
    "   send=$($j.send_phase_s)s serverCPU(코어)=$($j.server_cpu_cores) callbackQ max=$($j.callback_queue_max) end=$($j.callback_queue_end) sent=$($j.stats.sent_webcam)+$($j.stats.sent_phone) errors=$(($j.stats.PSObject.Properties | Where-Object { $_.Name -like 'error_*' } | ForEach-Object { $_.Name + '=' + $_.Value }) -join ' ')")
  $lines += $j.metrics_delta.PSObject.Properties | Where-Object { $_.Name -match 'dropped|failures|errors|processed|received|events_total|latency_seconds_(count|sum)|callback' -or $_.Name -match 'le="(0.01|0.05|0.1|0.25|0.5)"' } | ForEach-Object { "   $($_.Name) = $($_.Value)" }
  $lines += "   sink 수신 건수: $((Get-Content "$env:TEMP\sink.out" | Measure-Object -Line).Lines)"
  $lines | Tee-Object -FilePath $Out -Append
  Get-Process -Id ($ids -split ",") -ErrorAction SilentlyContinue | Stop-Process -Force
  Stop-Process -Id $sink.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 3
}
Remove-Item $Out -ErrorAction SilentlyContinue
Invoke-Case "A 20명 위상차 / 응시자별 소켓" 1 20 @("--stagger")
Invoke-Case "B 20명 동시 버스트 / 응시자별 소켓" 1 20 @()
Invoke-Case "C 20명 위상차 / 소켓 1개" 1 20 @("--stagger", "--one-ws")
Invoke-Case "D 20명 위상차 / 워커 4개(풀)" 4 20 @("--stagger")
Invoke-Case "E 60명 위상차 / 워커 1개(여유 탐색)" 1 60 @("--stagger")
Invoke-Case "F 100명 위상차 / 워커 1개(한계 탐색)" 1 100 @("--stagger")
"DONE" | Add-Content $Out
