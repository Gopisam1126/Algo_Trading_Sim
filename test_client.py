#!/usr/bin/env python3
"""
Test client for the Go Trading API WebSocket endpoint
"""

import websocket
import json
import time
from datetime import datetime

def on_message(ws, message):
    """Handle incoming WebSocket messages"""
    try:
        data = json.loads(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Received: {data['ticker']} @ ₹{data['price']:.2f} "
              f"(O:{data['open']:.2f} H:{data['high']:.2f} L:{data['low']:.2f} "
              f"C:{data['close']:.2f} V:{data['volume']:,})")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except KeyError as e:
        print(f"Missing key in data: {e}")

def on_error(ws, error):
    """Handle WebSocket errors"""
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket connection close"""
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    """Handle WebSocket connection open"""
    print("✅ Connected to trading data stream")
    print("📊 Receiving live INFY.NS data...")
    print("-" * 80)

def main():
    """Main function to run the test client"""
    print("🚀 Starting Go Trading API Test Client")
    print("=" * 80)
    
    # WebSocket URL
    ws_url = "ws://localhost:8080/ws"
    
    print(f"🔗 Connecting to: {ws_url}")
    print("⏳ Waiting for connection...")
    
    # Create WebSocket connection
    websocket.enableTrace(False)  # Set to True for debug info
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    try:
        # Run the WebSocket connection
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        ws.close()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        ws.close()

if __name__ == "__main__":
    main() 