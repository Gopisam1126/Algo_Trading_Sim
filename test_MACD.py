#!/usr/bin/env python3
"""
WebSocket MACD Calculator Client
Connects to the Go trading WebSocket server and calculates MACD in real-time
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

class MACDCalculator:
    """Real-time MACD calculator with configurable parameters"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.prices = deque(maxlen=slow_period + signal_period)  # Keep enough data
        self.fast_ema = None
        self.slow_ema = None
        self.macd_line = None
        self.signal_line = None
        self.histogram = None
        self.macd_values = deque(maxlen=100)  # Keep last 100 MACD values for analysis
        
    def calculate_ema(self, prices, period, prev_ema=None):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
            
        if prev_ema is None:
            # First EMA calculation - use simple average
            return sum(prices[-period:]) / period
        else:
            # Subsequent EMA calculations
            multiplier = 2 / (period + 1)
            return (prices[-1] * multiplier) + (prev_ema * (1 - multiplier))
    
    def add_price(self, price):
        """Add a new price and calculate MACD if we have enough data"""
        self.prices.append(price)
        
        if len(self.prices) < self.slow_period:
            return None
            
        # Calculate fast EMA
        self.fast_ema = self.calculate_ema(self.prices, self.fast_period, self.fast_ema)
        
        # Calculate slow EMA
        self.slow_ema = self.calculate_ema(self.prices, self.slow_period, self.slow_ema)
        
        if self.fast_ema is None or self.slow_ema is None:
            return None
            
        # Calculate MACD line
        self.macd_line = self.fast_ema - self.slow_ema
        
        # Calculate signal line (EMA of MACD line)
        if self.macd_line is not None:
            # We need to track MACD line values for signal calculation
            self.macd_values.append(self.macd_line)
            
            if len(self.macd_values) >= self.signal_period:
                # Calculate signal line as EMA of MACD line
                if self.signal_line is None:
                    # First signal line calculation - use simple average
                    recent_values = list(self.macd_values)
                    if len(recent_values) >= self.signal_period:
                        recent_values = recent_values[-self.signal_period:]
                    self.signal_line = sum(recent_values) / len(recent_values)
                else:
                    # Subsequent signal line calculations
                    multiplier = 2 / (self.signal_period + 1)
                    self.signal_line = (self.macd_line * multiplier) + (self.signal_line * (1 - multiplier))
                
                # Calculate histogram
                self.histogram = self.macd_line - self.signal_line
                
                return {
                    'macd_line': self.macd_line,
                    'signal_line': self.signal_line,
                    'histogram': self.histogram
                }
        
        return None
    
    def get_macd_signal(self, macd_data):
        """Get trading signal based on MACD values"""
        if macd_data is None:
            return "INSUFFICIENT_DATA"
        
        macd_line = macd_data['macd_line']
        signal_line = macd_data['signal_line']
        histogram = macd_data['histogram']
        
        # MACD signal logic
        if macd_line > signal_line and histogram > 0:
            if histogram > 0.5:  # Strong bullish signal
                return "STRONG_BUY"
            else:
                return "BUY"
        elif macd_line < signal_line and histogram < 0:
            if histogram < -0.5:  # Strong bearish signal
                return "STRONG_SELL"
            else:
                return "SELL"
        else:
            return "NEUTRAL"
    
    def get_stats(self):
        """Get MACD statistics"""
        if not self.macd_values:
            return None
            
        return {
            "current_macd": self.macd_line,
            "current_signal": self.signal_line,
            "current_histogram": self.histogram,
            "avg_macd": np.mean(self.macd_values),
            "min_macd": np.min(self.macd_values),
            "max_macd": np.max(self.macd_values),
            "data_points": len(self.macd_values),
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema
        }

class TradingDataClient:
    """WebSocket client for trading data with MACD calculation"""
    
    def __init__(self, websocket_url="ws://trading-api:8080/ws", fast_period=12, slow_period=26, signal_period=9):
        self.websocket_url = websocket_url
        self.macd_calculator = MACDCalculator(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        self.data_count = 0
        self.running = True
        
    def format_macd_output(self, data, macd_data, signal):
        """Format the output for display"""
        timestamp = data.get('timestamp', 'N/A')
        price = data.get('price', 0)
        ticker = data.get('ticker', 'N/A')
        
        if macd_data:
            macd_str = f"{macd_data['macd_line']:.4f}"
            signal_str = f"{macd_data['signal_line']:.4f}"
            histogram_str = f"{macd_data['histogram']:.4f}"
        else:
            macd_str = signal_str = histogram_str = "N/A"
        
        return (
            f"[{timestamp}] {ticker} | "
            f"Price: ${price:.2f} | "
            f"MACD: {macd_str} | "
            f"Signal: {signal_str} | "
            f"Histogram: {histogram_str} | "
            f"Signal: {signal}"
        )
    
    async def connect_and_stream(self):
        """Connect to WebSocket and stream data"""
        logger.info(f"Connecting to WebSocket: {self.websocket_url}")
        logger.info(f"MACD Configuration: Fast={self.macd_calculator.fast_period}, Slow={self.macd_calculator.slow_period}, Signal={self.macd_calculator.signal_period}")
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                logger.info("✅ Connected to trading data stream")
                logger.info("📊 Starting MACD calculation...")
                logger.info("=" * 80)
                
                async for message in websocket:
                    if not self.running:
                        break
                        
                    try:
                        # Parse the JSON data
                        data = json.loads(message)
                        self.data_count += 1
                        
                        # Extract price (using 'close' price for MACD calculation)
                        price = data.get('close', data.get('price', 0))
                        
                        # Calculate MACD
                        macd_data = self.macd_calculator.add_price(price)
                        signal = self.macd_calculator.get_macd_signal(macd_data)
                        
                        # Display the data
                        output = self.format_macd_output(data, macd_data, signal)
                        
                        # Color coding for signals
                        if signal == "STRONG_BUY":
                            logger.warning(f"🟢 {output}")
                        elif signal == "BUY":
                            logger.info(f"🟢 {output}")
                        elif signal == "STRONG_SELL":
                            logger.warning(f"🔴 {output}")
                        elif signal == "SELL":
                            logger.info(f"🔴 {output}")
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
        """Print MACD statistics"""
        stats = self.macd_calculator.get_stats()
        if stats:
            logger.info("=" * 80)
            logger.info("📈 MACD STATISTICS:")
            logger.info(f"   Current MACD: {stats['current_macd']:.4f}")
            logger.info(f"   Current Signal: {stats['current_signal']:.4f}")
            logger.info(f"   Current Histogram: {stats['current_histogram']:.4f}")
            logger.info(f"   Average MACD: {stats['avg_macd']:.4f}")
            logger.info(f"   Min MACD: {stats['min_macd']:.4f}")
            logger.info(f"   Max MACD: {stats['max_macd']:.4f}")
            logger.info(f"   Data Points: {stats['data_points']}")
            logger.info(f"   Fast EMA: {stats['fast_ema']:.2f}")
            logger.info(f"   Slow EMA: {stats['slow_ema']:.2f}")
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
    
    logger.info("🚀 Starting MACD WebSocket Client")
    logger.info("📡 Connecting to trading data stream...")
    logger.info("💡 MACD Signals: MACD > Signal = BUY, MACD < Signal = SELL")
    logger.info("⌨️  Press Ctrl+C to stop")
    logger.info("")
    
    # Create and start the client with configurable parameters
    client = TradingDataClient(
        websocket_url="ws://trading-api:8080/ws",
        fast_period=12,    # Fast EMA period
        slow_period=26,    # Slow EMA period
        signal_period=9    # Signal line period
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