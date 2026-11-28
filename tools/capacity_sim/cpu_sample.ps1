# 서버 프로세스(와 자식 워커)의 CPU 사용 시간을 구간 동안 재서 코어 환산 사용률을 낸다.
# 사용: powershell -File cpu_sample.ps1 -ServerPid 1234 -Seconds 60
param([int]$ServerPid, [int]$Seconds = 60)
function Get-Tree($rootPid) {
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
  $ids = @($rootPid); $queue = @($rootPid)
  while ($queue.Count) {
    $p = $queue[0]; $queue = @($queue | Select-Object -Skip 1)
    foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $p })) { $ids += $c.ProcessId; $queue += $c.ProcessId }
  }
  $ids
}
function Get-Cpu($ids) {
  $t = 0.0
  foreach ($i in $ids) { $pr = Get-Process -Id $i -ErrorAction SilentlyContinue; if ($pr) { $t += $pr.TotalProcessorTime.TotalSeconds } }
  $t
}
$ids = Get-Tree $ServerPid
$c0 = Get-Cpu $ids; $t0 = Get-Date
Start-Sleep -Seconds $Seconds
$c1 = Get-Cpu $ids; $dt = ((Get-Date) - $t0).TotalSeconds
$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
"procs=$($ids.Count) cpu_sec=$([math]::Round($c1-$c0,2)) wall=$([math]::Round($dt,1)) cores_used=$([math]::Round(($c1-$c0)/$dt,3)) of_total_pct=$([math]::Round(100*($c1-$c0)/$dt/$cores,2)) logical_cores=$cores"
