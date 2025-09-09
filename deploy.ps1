# Trading API Docker Deployment Script for Windows
# This script builds and runs the trading API in Docker

param(
    [switch]$Production
)

Write-Host "🚀 Starting Trading API Docker Deployment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Docker is installed
try {
    docker --version | Out-Null
} catch {
    Write-Error "Docker is not installed or not running. Please install Docker Desktop first."
    exit 1
}

# Check if Docker Compose is installed
try {
    docker-compose --version | Out-Null
} catch {
    Write-Error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
}

# Check if CSV file exists
if (-not (Test-Path "infy_2min_60days.csv")) {
    Write-Error "CSV file 'infy_2min_60days.csv' not found in current directory."
    exit 1
}

# Stop existing containers
Write-Status "Stopping existing containers..."
try {
    docker-compose down --remove-orphans
} catch {
    # Ignore errors if no containers are running
}

# Build the Docker image
Write-Status "Building Docker image..."
docker-compose build --no-cache

# Start the services
if ($Production) {
    Write-Status "Starting services in production mode..."
    docker-compose --profile production up -d
} else {
    Write-Status "Starting services..."
    docker-compose up -d
}

# Wait for the service to be ready
Write-Status "Waiting for service to be ready..."
Start-Sleep -Seconds 10

# Check if the service is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success "Trading API is running successfully!"
        Write-Host ""
        Write-Host "📊 Service Information:" -ForegroundColor Cyan
        Write-Host "  • API Server: http://localhost:8080" -ForegroundColor White
        Write-Host "  • WebSocket: ws://localhost:8080/ws" -ForegroundColor White
        Write-Host "  • Health Check: http://localhost:8080/health" -ForegroundColor White
        Write-Host "  • Web Client: http://localhost:8080/client.html" -ForegroundColor White
        Write-Host ""
        Write-Host "🔧 Management Commands:" -ForegroundColor Cyan
        Write-Host "  • View logs: docker-compose logs -f trading-api" -ForegroundColor White
        Write-Host "  • Stop services: docker-compose down" -ForegroundColor White
        Write-Host "  • Restart services: docker-compose restart" -ForegroundColor White
        Write-Host ""
        Write-Success "Deployment completed successfully!"
    } else {
        throw "Service returned status code $($response.StatusCode)"
    }
} catch {
    Write-Error "Service is not responding. Check logs with: docker-compose logs trading-api"
    exit 1
} 