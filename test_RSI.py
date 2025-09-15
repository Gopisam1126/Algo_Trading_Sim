#!/usr/bin/env python3
"""
WebSocket RSI Calculator Client
Connects to the Go trading WebSocket server and calculates RSI in real-time
"""

import asyncio
import websockets
import json
import pandas as pd
from collections import deque
import numpy as np
from datetime import datetime
import logging
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class RSICalculator:
    """Real-time RSI calculator with 14-period default"""
    
    def __init__(self, period=14):
        self.period = period
        self.prices = deque(maxlen=period + 1)  # Keep one extra for calculation
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.avg_gain = None
        self.avg_loss = None
        self.rsi_values = deque(maxlen=100)  # Keep last 100 RSI values for analysis
        
    def add_price(self, price):
        """Add a new price and calculate RSI if we have enough data"""
        self.prices.append(price)
        
        if len(self.prices) < 2:
            return None
            
        # Calculate price change
        price_change = self.prices[-1] - self.prices[-2]
        
        # Separate gains and losses
        gain = max(price_change, 0)
        loss = abs(min(price_change, 0))
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        # Need at least 'period' changes to calculate RSI
        if len(self.gains) < self.period:
            return None
            
        # Calculate RSI
        if self.avg_gain is None or self.avg_loss is None:
            # First RSI calculation - use simple average
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            # Subsequent RSI calculations - use smoothed average (Wilder's method)
            self.avg_gain = ((self.avg_gain * (self.period - 1)) + gain) / self.period
            self.avg_loss = ((self.avg_loss * (self.period - 1)) + loss) / self.period
        
        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.rsi_values.append(rsi)
        return rsi
    
    def get_rsi_signal(self, rsi):
        """Get trading signal based on RSI value"""
        if rsi is None:
            return "INSUFFICIENT_DATA"
        elif rsi > 70:
            return "OVERBOUGHT"
        elif rsi < 30:
            return "OVERSOLD"
        else:
            return "NEUTRAL"
    
    def get_stats(self):
        """Get RSI statistics"""
        if not self.rsi_values:
            return None
            
        return {
            "current_rsi": self.rsi_values[-1] if self.rsi_values else None,
            "avg_rsi": np.mean(self.rsi_values),
            "min_rsi": np.min(self.rsi_values),
            "max_rsi": np.max(self.rsi_values),
            "data_points": len(self.rsi_values),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss
        }

class TradingDataClient:
    """WebSocket client for trading data with RSI calculation"""
    
    def __init__(self, websocket_url="ws://trading-api:8080/ws", rsi_period=14):
        self.websocket_url = websocket_url
        self.rsi_calculator = RSICalculator(period=rsi_period)
        self.data_count = 0
        self.running = True
        
    def format_rsi_output(self, data, rsi, signal):
        """Format the output for display"""
        timestamp = data.get('timestamp', 'N/A')
        price = data.get('price', 0)
        ticker = data.get('ticker', 'N/A')
        
        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
        
        return (
            f"[{timestamp}] {ticker} | "
            f"Price: ${price:.2f} | "
            f"RSI(14): {rsi_str} | "
            f"Signal: {signal}"
        )
    
    async def connect_and_stream(self):
        """Connect to WebSocket and stream data"""
        logger.info(f"Connecting to WebSocket: {self.websocket_url}")
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                logger.info("✅ Connected to trading data stream")
                logger.info("📊 Starting RSI calculation (14-period)...")
                logger.info("=" * 80)
                
                async for message in websocket:
                    if not self.running:
                        break
                        
                    try:
                        # Parse the JSON data
                        data = json.loads(message)
                        self.data_count += 1
                        
                        # Extract price (using 'close' price for RSI calculation)
                        price = data.get('close', data.get('price', 0))
                        
                        # Calculate RSI
                        rsi = self.rsi_calculator.add_price(price)
                        signal = self.rsi_calculator.get_rsi_signal(rsi)
                        
                        # Display the data
                        output = self.format_rsi_output(data, rsi, signal)
                        
                        # Color coding for signals
                        if signal == "OVERBOUGHT":
                            logger.warning(f"🔴 {output}")
                        elif signal == "OVERSOLD":
                            logger.warning(f"🟢 {output}")
                        else:
                            logger.info(f"⚪ {output}")
                        
                        # Print stats every 50 data points
                        if self.data_count % 50 == 0:
                            await self.print_stats()
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error processing data: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            logger.error("❌ WebSocket connection closed")
        except websockets.exceptions.InvalidURI:
            logger.error(f"❌ Invalid WebSocket URI: {self.websocket_url}")
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            
    async def print_stats(self):
        """Print RSI statistics"""
        stats = self.rsi_calculator.get_stats()
        if stats:
            logger.info("=" * 80)
            logger.info("📈 RSI STATISTICS:")
            logger.info(f"   Current RSI: {stats['current_rsi']:.2f}")
            logger.info(f"   Average RSI: {stats['avg_rsi']:.2f}")
            logger.info(f"   Min RSI: {stats['min_rsi']:.2f}")
            logger.info(f"   Max RSI: {stats['max_rsi']:.2f}")
            logger.info(f"   Data Points: {stats['data_points']}")
            logger.info(f"   Avg Gain: {stats['avg_gain']:.4f}")
            logger.info(f"   Avg Loss: {stats['avg_loss']:.4f}")
            logger.info("=" * 80)
    
    def stop(self):
        """Stop the client"""
        self.running = False
        logger.info("🛑 Stopping client...")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n🛑 Received interrupt signal. Shutting down...")
    sys.exit(0)

async def main():
    """Main function"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("🚀 Starting RSI WebSocket Client")
    logger.info("📡 Connecting to trading data stream...")
    logger.info("💡 RSI Signals: >70 = OVERBOUGHT, <30 = OVERSOLD")
    logger.info("⌨️  Press Ctrl+C to stop")
    logger.info("")
    
    # Create and start the client
    client = TradingDataClient(
        websocket_url="ws://trading-api:8080/ws",
        rsi_period=14
    )
    
    try:
        await client.connect_and_stream()
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        client.stop()
        logger.info("✅ Client stopped")

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import websockets
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Install required packages with:")
        print("   pip install websockets pandas numpy")
        sys.exit(1)
    
    # Run the client
    asyncio.run(main())