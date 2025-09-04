import requests
import time
from threading import Lock
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDataBatcher:
    """
    Simple batching for 10-minute intervals and sending to AI endpoint
    """
    
    def __init__(self, ai_endpoint="http://localhost:11434/api/chat"):
        self.ai_endpoint = ai_endpoint
        self.batch_duration = 600  # 10 minutes in seconds
        self.max_data_points = 1200  # Expected data points per 10-minute batch (500ms intervals)
        
        # Current batch storage
        self.current_batch = []
        self.current_batch_start_time = None
        self.batch_lock = Lock()
        
        logger.info(f"AIDataBatcher initialized: collecting data for 10-minute batches")
    
    def add_data_point(self, timestamp, price_data, indicators):
        """
        Add a new data point to the current batch
        """
        with self.batch_lock:
            current_time = datetime.now()
            
            # Initialize batch if needed
            if self.current_batch_start_time is None:
                self.current_batch_start_time = current_time
            
            # Create data point
            data_point = {
                'timestamp': timestamp,
                'price_data': price_data,
                'technical_indicators': indicators
            }
            
            self.current_batch.append(data_point)
            
            # Check if batch is complete (10 minutes elapsed)
            if current_time >= self.current_batch_start_time + timedelta(seconds=self.batch_duration):
                self._send_batch_to_ai()
    
    def _send_batch_to_ai(self):
        """
        Send the completed 10-minute batch to AI endpoint
        """
        if not self.current_batch:
            return
        
        try:
            # Send the array directly to AI endpoint
            response = requests.post(
                self.ai_endpoint,
                json=self.current_batch,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            logger.info(f"Sent batch of {len(self.current_batch)} data points to AI. Status: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send batch to AI: {e}")
        
        # Reset for next batch
        self.current_batch = []
        self.current_batch_start_time = datetime.now()

# Global AI batcher instance
ai_batcher = AIDataBatcher()

def integrate_ai_batcher():
    """
    Integration function - replace your custom_send_to_api with this
    """
    async def enhanced_send_to_api(data):
        global current_indicators, current_price_data
        
        # Original functionality
        current_price_data = data
        current_indicators = data.get('technical_indicators', {})
        logger.info(f"Updated global indicators data: {len(current_indicators)} indicators available")
        
        # Add to AI batcher
        ai_batcher.add_data_point(
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            price_data=data.get('price_data', {}),
            indicators=current_indicators
        )
    
    return enhanced_send_to_api