#!/bin/bash

# Start Trading API and RSI Client
# This script starts both the Go trading API server and the Python RSI client

echo "🚀 Starting Trading API and RSI Client..."

# Function to handle cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$GO_PID" ]; then
        kill $GO_PID 2>/dev/null
    fi
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the Go trading API server
echo "📡 Starting Go Trading API server..."
./trading-api &
GO_PID=$!

# Wait a moment for the server to start
sleep 3

# Check if the server is running
if ! kill -0 $GO_PID 2>/dev/null; then
    echo "❌ Failed to start Go trading API server"
    exit 1
fi

echo "✅ Go Trading API server started (PID: $GO_PID)"

# Start the Python RSI client
echo "📊 Starting Python RSI client..."
python3 test_RSI.py &
PYTHON_PID=$!

# Check if the RSI client started successfully
sleep 2
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Failed to start Python RSI client"
    kill $GO_PID 2>/dev/null
    exit 1
fi

echo "✅ Python RSI client started (PID: $PYTHON_PID)"
echo ""
echo "🎯 Services running:"
echo "   - Go Trading API: http://localhost:8080"
echo "   - WebSocket: ws://localhost:8080/ws"
echo "   - Client Dashboard: http://localhost:8080/client.html"
echo "   - Python RSI Client: Running in background"
echo ""
echo "💡 Press Ctrl+C to stop all services"

# Wait for both processes
wait $GO_PID $PYTHON_PID 