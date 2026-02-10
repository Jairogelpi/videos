# Startup Script for Project Tohjo
Write-Host "1. STOPPING old Node processes..."
$ports = 3000, 3001
foreach ($port in $ports) {
    echo "Searching for processes on port $port..."
    $procId = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    if ($procId) {
        echo "Killing PID $procId on port $port..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}
# Brute force cleanup for Node
taskkill /f /im node.exe /t 2>$null
Write-Host "Cleanup complete."

Write-Host "2. STARTING Docker Containers..."
docker start tohjo-redis supabase_db_videos

Write-Host "3. LAUNCHING API (Port 3001)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd apps/api; pnpm run dev"

Write-Host "4. LAUNCHING Web (Port 3000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd apps/web; pnpm run dev"

Write-Host "5. LAUNCHING Audio Worker (Python)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd workers/audio-cpu; .\.venv\Scripts\activate; python main.py"

Write-Host "6. LAUNCHING Render Worker (Remotion)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd workers/render; pnpm run start"

Write-Host "---------------------------------------------------"
Write-Host "DONE! Four new windows should open."
Write-Host "If they close immediately, check for errors in them."
Write-Host "Ensure Docker Desktop is running if containers fail."
Write-Host "---------------------------------------------------"
