Write-Host "Killing Node.js processes on ports 3000-3003..."
$ports = 3000, 3001, 3002, 3003
foreach ($port in $ports) {
    $p = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($p) {
        Write-Host "Kill process $($p.OwningProcess) on port $port"
        Stop-Process -Id $p.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "Done."
