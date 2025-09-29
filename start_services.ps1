# Start Trading API and RSI Client
# This script starts both the Go trading API server and the Python RSI client

Write-Host "🚀 Starting Trading API and RSI Client..." -ForegroundColor Green

# Function to handle cleanup on exit
function Cleanup {
    Write-Host "🛑 Shutting down services..." -ForegroundColor Yellow
    if ($goProcess -and !$goProcess.HasExited) {
        $goProcess.Kill()
    }
    if ($pythonProcess -and !$pythonProcess.HasExited) {
        $pythonProcess.Kill()
    }
    exit 0
}

# Set up signal handlers
$null = Register-EngineEvent PowerShell.Exiting -Action { Cleanup }

# Start the Go trading API server
Write-Host "📡 Starting Go Trading API server..." -ForegroundColor Cyan
$goProcess = Start-Process -FilePath "./trading-api" -PassThru -WindowStyle Hidden

# Wait a moment for the server to start
Start-Sleep -Seconds 3

# Check if the server is running
if ($goProcess.HasExited) {
    Write-Host "❌ Failed to start Go trading API server" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Go Trading API server started (PID: $($goProcess.Id))" -ForegroundColor Green

# Start the Python RSI client
Write-Host "📊 Starting Python RSI client..." -ForegroundColor Cyan
$pythonProcess = Start-Process -FilePath "python" -ArgumentList "test_RSI.py" -PassThru -WindowStyle Hidden

# Check if the RSI client started successfully
Start-Sleep -Seconds 2
if ($pythonProcess.HasExited) {
    Write-Host "❌ Failed to start Python RSI client" -ForegroundColor Red
    if (!$goProcess.HasExited) {
        $goProcess.Kill()
    }
    exit 1
}

Write-Host "✅ Python RSI client started (PID: $($pythonProcess.Id))" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Services running:" -ForegroundColor Yellow
Write-Host "   - Go Trading API: http://localhost:8080" -ForegroundColor White
Write-Host "   - WebSocket: ws://localhost:8080/ws" -ForegroundColor White
Write-Host "   - Client Dashboard: http://localhost:8080/client.html" -ForegroundColor White
Write-Host "   - Python RSI Client: Running in background" -ForegroundColor White
Write-Host ""
Write-Host "💡 Press Ctrl+C to stop all services" -ForegroundColor Cyan

# Wait for both processes
try {
    while (!$goProcess.HasExited -and !$pythonProcess.HasExited) {
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "🛑 Interrupted by user" -ForegroundColor Yellow
} finally {
    Cleanup
} 