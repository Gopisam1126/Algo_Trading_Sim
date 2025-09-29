#!/bin/bash

# Trading API Docker Deployment Script
# This script builds and runs the trading API in Docker

set -e

echo "🚀 Starting Trading API Docker Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if CSV file exists
if [ ! -f "infy_2min_60days.csv" ]; then
    print_error "CSV file 'infy_2min_60days.csv' not found in current directory."
    exit 1
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Build the Docker image
print_status "Building Docker image..."
docker-compose build --no-cache

# Start the services
print_status "Starting services..."
docker-compose up -d

# Wait for the service to be ready
print_status "Waiting for service to be ready..."
sleep 10

# Check if the service is running
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    print_success "Trading API is running successfully!"
    echo ""
    echo "📊 Service Information:"
    echo "  • API Server: http://localhost:8080"
    echo "  • WebSocket: ws://localhost:8080/ws"
    echo "  • Health Check: http://localhost:8080/health"
    echo "  • Web Client: http://localhost:8080/client.html"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose logs -f trading-api"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo ""
    print_success "Deployment completed successfully!"
else
    print_error "Service is not responding. Check logs with: docker-compose logs trading-api"
    exit 1
fi 