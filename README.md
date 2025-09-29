# Live Trading Data API with RSI Analysis

A high-performance Go API that provides live simulated trading data via WebSockets with real-time RSI (Relative Strength Index) calculation and analysis.

## Features

- **Real-time WebSocket streaming** of trading data
- **Real-time RSI calculation** with 14-period default (Wilder's smoothing method)
- **Real-time MACD calculation** with configurable parameters (12,26,9 default)
- **RSI trading signals** (Overbought >70, Oversold <30, Neutral 30-70)
- **MACD trading signals** (MACD > Signal = BUY, MACD < Signal = SELL)
- **Web dashboard** with live RSI and MACD display and color-coded signals
- **Python RSI client** for standalone RSI analysis
- **Python MACD client** for standalone MACD analysis
- **Minimal latency** optimized for high-frequency trading applications
- **REST API endpoints** for current data and health checks
- **CORS enabled** for cross-origin requests
- **Automatic data looping** - continuously streams historical data
- **Multiple client support** - handles multiple WebSocket connections simultaneously
- **Buffered channels** for high throughput and low latency

## Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Web Client    │ ◄──────────────► │   Go API        │
│   (HTML/JS)     │                 │   (Port 8080)   │
│   + RSI Calc    │                 │                 │
└─────────────────┘                 └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ CSV Data        │
                                    │ (infy_2min_60days.csv) │
                                    └─────────────────┘

┌─────────────────┐    WebSocket    ┌─────────────────┐
│ Python RSI      │ ◄──────────────► │   Go API        │
│   Client        │                 │   (Port 8080)   │
│   (test_RSI.py) │                 │                 │
└─────────────────┘                 └─────────────────┘
```

## Quick Start

### Prerequisites

- Go 1.21 or higher
- Python 3.8 or higher (for RSI client)
- CSV data file (`infy_2min_60days.csv`)

### Installation

1. **Clone or download the project files**

2. **Install Go dependencies:**
   ```bash
   go mod tidy
   ```

3. **Run the API server:**
   ```bash
   go run main.go
   ```

   **Or use the startup script (Windows):**
   ```powershell
   .\start_services.ps1
   ```

   **Or use the startup script (Linux/Mac):**
   ```bash
   chmod +x start_services.sh
   ./start_services.sh
   ```

4. **Open the client in your browser:**
   ```
   http://localhost:8080/client.html
   ```
   Or simply open `client.html` in your browser.

### RSI Analysis

The system includes real-time RSI calculation with the following features:

#### Web Dashboard RSI
- **Real-time RSI calculation** in the browser using JavaScript
- **Color-coded signals**: Red (Overbought), Green (Oversold), Blue (Neutral)
- **14-period RSI** using Wilder's smoothing method
- **Live updates** as new data arrives

#### Python RSI Client
Run the standalone Python RSI client for detailed analysis:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the RSI client
python test_RSI.py
```

The Python client provides:
- **Detailed RSI statistics** every 50 data points
- **Color-coded console output** for trading signals
- **Comprehensive logging** with timestamps
- **Graceful shutdown** with Ctrl+C handling

#### RSI Trading Signals
- **Overbought (>70)**: Potential sell signal (red)
- **Oversold (<30)**: Potential buy signal (green)  
- **Neutral (30-70)**: No clear signal (blue/white)

## API Endpoints

### WebSocket
- **URL:** `ws://localhost:8080/ws`
- **Protocol:** WebSocket
- **Data Format:** JSON
- **Description:** Real-time streaming of trading data

### REST API
- **Health Check:** `GET http://localhost:8080/health`
- **Current Data:** `GET http://localhost:8080/api/current`

## Data Format

Each WebSocket message contains a JSON object with the following structure:

```json
{
  "timestamp": "2025-06-23 03:45:00+00:00",
  "price": 1586.9000244140625,
  "close": 1598.699951171875,
  "high": 1598.699951171875,
  "low": 1582.300048828125,
  "open": 1594.9000244140625,
  "volume": 174769,
  "ticker": "INFY.NS"
}
```

## Configuration

### Data Simulation Speed

You can adjust the simulation speed by modifying the `speed` parameter in `main.go`:

```go
simulator, err = NewDataSimulator("infy_2min_60days.csv", 500*time.Millisecond)
```

- `500*time.Millisecond` = 2 data points per second
- `1*time.Second` = 1 data point per second
- `100*time.Millisecond` = 10 data points per second

### Server Port

Change the port by modifying the `port` variable in `main.go`:

```go
port := ":8080"  // Change to your preferred port
```

## Performance Optimizations

1. **Buffered Channels:** Uses buffered channels (1000 messages) for high throughput
2. **Goroutines:** Separate goroutines for reading and writing WebSocket messages
3. **Memory Pooling:** Efficient memory management for JSON marshaling
4. **Connection Limits:** Automatic cleanup of slow or unresponsive clients
5. **Ping/Pong:** WebSocket ping/pong for connection health monitoring

## Client Integration

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = function(event) {
    console.log('Connected to trading data stream');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    // Process trading data
};

ws.onclose = function(event) {
    console.log('Connection closed');
};
```

### Python WebSocket Client

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Received: {data}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connected to trading data stream")

websocket.enableTrace(True)
ws = websocket.WebSocketApp("ws://localhost:8080/ws",
                          on_open=on_open,
                          on_message=on_message,
                          on_error=on_error,
                          on_close=on_close)

ws.run_forever()
```

## Monitoring

The API provides real-time logging of:
- Client connections/disconnections
- Data transmission
- Error conditions
- Performance metrics

## Troubleshooting

### Common Issues

1. **Port already in use:**
   - Change the port in `main.go`
   - Or kill the process using the port

2. **CSV file not found:**
   - Ensure `infy_2min_60days.csv` is in the same directory as `main.go`

3. **WebSocket connection failed:**
   - Check if the server is running
   - Verify the WebSocket URL
   - Check browser console for errors

4. **High latency:**
   - Reduce the number of connected clients
   - Increase buffer sizes in the code
   - Check network conditions

### Logs

The server outputs detailed logs including:
- Server startup information
- Client connection events
- Data transmission statistics
- Error messages

## Development

### Project Structure

```
├── main.go              # Main application file
├── go.mod               # Go module file
├── go.sum               # Go dependencies checksum
├── client.html          # Web client with RSI display
├── test_RSI.py          # Python RSI client
├── requirements.txt     # Python dependencies
├── start_services.ps1   # Windows startup script
├── start_services.sh    # Linux/Mac startup script
├── infy_2min_60days.csv # Trading data file
└── README.md            # This file
```

### Building for Production

```bash
# Build the application
go build -o trading-api main.go

# Run the binary
./trading-api
```

## Docker Deployment

### Quick Start with Docker

1. **Build and run with Docker Compose:**
   ```bash
   # Linux/Mac
   chmod +x deploy.sh
   ./deploy.sh
   
   # Windows PowerShell
   .\deploy.ps1
   ```

   This will start:
   - **Trading API server** on port 8080
   - **Python RSI client** in a separate container
   - **Web dashboard** with RSI display

2. **Manual Docker commands:**
   ```bash
   # Build the image
   docker-compose build
   
   # Start the service
   docker-compose up -d
   
   # View logs
   docker-compose logs -f trading-api
   
   # Stop the service
   docker-compose down
   ```

### Production Deployment

For production deployment with Nginx reverse proxy:

```bash
# Start with production profile
docker-compose --profile production up -d
```

This will start:
- Trading API on port 8080
- Nginx reverse proxy on ports 80 and 443
- Load balancing and rate limiting
- SSL termination (when configured)

### Docker Configuration

#### Dockerfile Features:
- **Multi-stage build** for optimized image size
- **Non-root user** for security
- **Health checks** for monitoring
- **Alpine Linux** for minimal footprint
- **CGO disabled** for static binary

#### Docker Compose Features:
- **Health monitoring** with automatic restarts
- **Volume mounting** for easy data updates
- **Network isolation** for security
- **Production-ready** Nginx configuration

#### Environment Variables:
```yaml
environment:
  - GIN_MODE=release  # Production mode for Gin
```

### Container Management

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f trading-api

# Restart service
docker-compose restart trading-api

# Update and restart
docker-compose pull && docker-compose up -d

# Clean up
docker-compose down --volumes --remove-orphans
```

### Monitoring and Logs

```bash
# View real-time logs
docker-compose logs -f trading-api

# Check container health
docker-compose ps

# Monitor resource usage
docker stats trading-api

# Access container shell
docker-compose exec trading-api sh
```

### Scaling

To scale the application for high load:

```bash
# Scale to multiple instances
docker-compose up -d --scale trading-api=3

# Use with load balancer
docker-compose --profile production up -d
```

### SSL/HTTPS Setup

1. **Generate SSL certificates:**
   ```bash
   mkdir ssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout ssl/key.pem -out ssl/cert.pem
   ```

2. **Uncomment HTTPS section in nginx.conf**

3. **Start with production profile:**
   ```bash
   docker-compose --profile production up -d
   ```

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests! 