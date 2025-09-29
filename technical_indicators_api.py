#!/usr/bin/env python3
"""
REST API Server for Technical Indicators
Provides JSON endpoints for AI systems to access technical indicator data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
import logging
from technical_indicators import TechnicalIndicators, TradingDataClient
from typing import Dict, Any
from threading import Lock
# from dotenv import load_dotenv
# from pathlib import Path
# import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# env_path = Path(__file__).parent / ".env"
# load_dotenv(dotenv_path=env_path)

# openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

# Global variables to store current data
current_indicators = {}
current_price_data = {}
websocket_client = None
client_thread = None
running = False

class AIDataBatcher:
    """
    Simple batching for 10-minute intervals and sending to AI endpoint
    """
    
    def __init__(self):
        self.openai_client = OpenAI(
            
        )
        self.batch_duration = 600  # 10 minutes in seconds
        self.max_data_points = 1200  # Expected data points per 10-minute batch (500ms intervals)
        
        # Current batch storage
        self.current_batch = []
        self.current_batch_start_time = None
        self.batch_lock = Lock()

        self.ai_response_lock = Lock()
        self.latest_ai_response = None
        self.ai_response_history = []
        self.max_history_length = 50
        self.batch_count = 0
        self.last_batch_sent_time = None
        self.ai_communication_status = "waiting"
        
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
        Send the completed 10-minute batch to AI endpoint and store response
        """
        if not self.current_batch:
            return

        try:
            self.ai_communication_status = "sending"

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this 10-minute trading data batch with {len(self.current_batch)} data points. Provide insights on market trends, volatility, and trading signals: {json.dumps(self.current_batch)}"
                    }
                ]
            )

            ai_analysis = response.choices[0].message.content

            # Store AI response
            with self.ai_response_lock:
                response_data = {
                    'timestamp': datetime.now().isoformat(),
                    'batch_number': self.batch_count + 1,
                    'data_points_analyzed': len(self.current_batch),
                    'ai_analysis': ai_analysis,
                    'batch_start_time': self.current_batch_start_time.isoformat(),
                    'batch_end_time': datetime.now().isoformat()
                }

                self.latest_ai_response = response_data
                self.ai_response_history.append(response_data)

                # Keep only last N responses
                if len(self.ai_response_history) > self.max_history_length:
                    self.ai_response_history.pop(0)

            self.ai_communication_status = "success"
            self.last_batch_sent_time = datetime.now()
            self.batch_count += 1
            logger.info(f"AI analysis received for batch {self.batch_count}")

        except Exception as e:
            self.ai_communication_status = "error"
            logger.error(f"Failed to send batch to AI: {e}")

        # Reset for next batch
        self.current_batch = []
        self.current_batch_start_time = datetime.now()

# Global AI batcher instance
ai_batcher = AIDataBatcher()

class WebSocketThread(threading.Thread):
    """Thread to run WebSocket client in background"""
    
    def __init__(self, websocket_url: str, config: dict):
        super().__init__()
        self.websocket_url = websocket_url
        self.config = config
        self.client = None
        self.daemon = True
    
    def run(self):
        """Run the WebSocket client"""
        global current_indicators, current_price_data, running
        
        async def run_client():
            global current_indicators, current_price_data, running
            
            try:
                self.client = TradingDataClient(self.websocket_url, self.config)
                running = True
                
                # Override the send_to_api method to update global data
                async def custom_send_to_api(data):
                    global current_indicators, current_price_data
                    current_price_data = data
                    current_indicators = data.get('technical_indicators', {})
                    logger.info(f"Updated global indicators data: {len(current_indicators)} indicators available")

                    # Add to AI batcher
                    ai_batcher.add_data_point(
                        timestamp=data.get('timestamp', datetime.now().isoformat()),
                        price_data=data.get('price_data', {}),
                        indicators=current_indicators
                    )
                
                self.client.send_to_api = custom_send_to_api
                
                await self.client.connect_and_stream()
            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
            finally:
                running = False
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(run_client())
        except Exception as e:
            logger.error(f"Failed to start WebSocket client: {e}")
        finally:
            loop.close()
    
    def stop(self):
        """Stop the WebSocket client"""
        global running
        running = False
        if self.client:
            self.client.stop()

    async def send_to_api(self, data: Dict[str, Any]):
        """Send technical indicators data to API endpoint"""
        # Update global data for API access
        global current_indicators, current_price_data
        
        current_price_data = data
        current_indicators = data.get('technical_indicators', {})
        
        logger.info(f"Updated global indicators data: {len(current_indicators)} indicators available")

def update_current_data(timestamp, price_data, indicators):
    """Update current data with new values"""
    global current_indicators, current_price_data
    
    current_price_data = {
        'timestamp': timestamp,
        'price_data': price_data,
        'technical_indicators': indicators
    }
    current_indicators = indicators

@app.route('/api/indicators/current', methods=['GET'])
def get_current_indicators():
    """Get current technical indicators"""
    global current_indicators, current_price_data
    
    if not current_indicators:
        return jsonify({
            'error': 'No data available',
            'message': 'Technical indicators not yet calculated'
        }), 404
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'data': current_price_data
    })

@app.route('/api/indicators/config', methods=['GET', 'POST'])
def manage_configuration():
    """Get or update indicator configuration"""
    global websocket_client
    
    if request.method == 'GET':
        if websocket_client and websocket_client.indicators:
            return jsonify({
                'configuration': websocket_client.indicators.config
            })
        else:
            return jsonify({
                'error': 'No client available',
                'message': 'WebSocket client not initialized'
            }), 404
    
    elif request.method == 'POST':
        try:
            new_config = request.get_json()
            if not new_config:
                return jsonify({'error': 'No configuration provided'}), 400
            
            if websocket_client:
                websocket_client.update_configuration(new_config)
                return jsonify({
                    'message': 'Configuration updated successfully',
                    'configuration': websocket_client.indicators.config
                })
            else:
                return jsonify({
                    'error': 'No client available',
                    'message': 'WebSocket client not initialized'
                }), 404
                
        except Exception as e:
            return jsonify({'error': f'Failed to update configuration: {str(e)}'}), 500

@app.route('/api/indicators/status', methods=['GET'])
def get_status():
    """Get system status"""
    global running, websocket_client
    
    return jsonify({
        'status': 'running' if running else 'stopped',
        'websocket_connected': running,
        'timestamp': datetime.now().isoformat(),
        'has_data': bool(current_indicators)
    })

@app.route('/api/indicators/start', methods=['POST'])
def start_indicators():
    """Start the technical indicators calculation"""
    global websocket_client, client_thread, running
    
    if running:
        return jsonify({'message': 'Already running'}), 200
    
    try:
        config = request.get_json() or {}
        websocket_url = request.args.get('websocket_url', 'ws://trading-api:8080/ws')
        
        # Create and start WebSocket client thread
        client_thread = WebSocketThread(websocket_url, config)
        client_thread.start()
        
        # Wait a bit for connection
        time.sleep(2)
        
        return jsonify({
            'message': 'Technical indicators calculation started',
            'websocket_url': websocket_url,
            'configuration': config
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start: {str(e)}'}), 500

@app.route('/api/indicators/stop', methods=['POST'])
def stop_indicators():
    """Stop the technical indicators calculation"""
    global running, client_thread
    
    if not running:
        return jsonify({'message': 'Already stopped'}), 200
    
    try:
        running = False
        if client_thread:
            client_thread.stop()
            client_thread.join(timeout=5)
        
        return jsonify({'message': 'Technical indicators calculation stopped'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to stop: {str(e)}'}), 500

@app.route('/api/indicators/historical', methods=['POST'])
def calculate_historical_indicators():
    """Calculate indicators for historical data"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No historical data provided'}), 400
        
        config = data.get('config', {})
        historical_data = data['data']
        
        # Create indicators instance
        indicators = TechnicalIndicators(config)
        
        # Process historical data
        results = []
        for data_point in historical_data:
            indicators_result = indicators.add_data_point(data_point)
            if indicators_result:
                results.append({
                    'timestamp': data_point.get('timestamp', datetime.now().isoformat()),
                    'price_data': {
                        'price': data_point.get('price', data_point.get('close')),
                        'high': data_point.get('high'),
                        'low': data_point.get('low'),
                        'open': data_point.get('open'),
                        'volume': data_point.get('volume')
                    },
                    'technical_indicators': indicators_result
                })
        
        return jsonify({
            'message': f'Processed {len(results)} data points',
            'results': results,
            'configuration': config
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to calculate historical indicators: {str(e)}'}), 500

@app.route('/api/indicators/all', methods=['GET'])
def get_all_indicators():
    """Get all available indicators with their current values"""
    global current_indicators, websocket_client
    
    if not current_indicators:
        return jsonify({
            'error': 'No data available',
            'message': 'Technical indicators not yet calculated'
        }), 404
    
    # Get comprehensive indicator data
    if websocket_client and websocket_client.indicators:
        all_indicators = websocket_client.indicators.get_all_indicators()
    else:
        all_indicators = {
            'timestamp': datetime.now().isoformat(),
            'indicators': current_indicators,
            'configuration': {}
        }
    
    return jsonify(all_indicators)

@app.route('/api/indicators/specific/<indicator>', methods=['GET'])
def get_specific_indicator(indicator):
    """Get a specific indicator value"""
    global current_indicators
    
    if not current_indicators:
        return jsonify({
            'error': 'No data available',
            'message': 'Technical indicators not yet calculated'
        }), 404
    
    # Check if indicator exists
    if indicator in current_indicators:
        return jsonify({
            'indicator': indicator,
            'value': current_indicators[indicator],
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'error': f'Indicator {indicator} not found',
            'available_indicators': list(current_indicators.keys())
        }), 404

@app.route('/api/indicators/signals', methods=['GET'])
def get_trading_signals():
    """Get trading signals from all indicators with volatility analysis"""
    global current_indicators
    
    if not current_indicators:
        return jsonify({
            'error': 'No data available',
            'message': 'Technical indicators not yet calculated'
        }), 404
    
    signals = {}
    volatility_analysis = None
    
    # Extract signals from indicators
    for indicator_name, indicator_data in current_indicators.items():
        if isinstance(indicator_data, dict) and 'signal' in indicator_data:
            signals[indicator_name] = indicator_data['signal']
        elif indicator_name == 'rsi' and isinstance(indicator_data, dict) and 'value' in indicator_data:
            # RSI signal logic
            rsi_value = indicator_data['value']
            if rsi_value > 70:
                signals[indicator_name] = 'OVERBOUGHT'
            elif rsi_value < 30:
                signals[indicator_name] = 'OVERSOLD'
            else:
                signals[indicator_name] = 'NEUTRAL'
        elif indicator_name == 'macd' and isinstance(indicator_data, dict):
            # MACD signal logic
            if 'macd_line' in indicator_data and 'signal_line' in indicator_data:
                macd_line = indicator_data['macd_line']
                signal_line = indicator_data['signal_line']
                if macd_line > signal_line:
                    signals[indicator_name] = 'BUY'
                else:
                    signals[indicator_name] = 'SELL'
        elif indicator_name == 'stochastic' and isinstance(indicator_data, dict):
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        elif indicator_name == 'cci' and isinstance(indicator_data, dict):
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        elif indicator_name == 'roc' and isinstance(indicator_data, dict):
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        elif indicator_name == 'parabolic_sar' and isinstance(indicator_data, dict):
            # Parabolic SAR signal logic
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        elif indicator_name == 'supertrend' and isinstance(indicator_data, dict):
            # SuperTrend signal logic
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        elif indicator_name == 'momentum' and isinstance(indicator_data, dict):
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        elif indicator_name == 'atr' and isinstance(indicator_data, dict):
            # ATR volatility signals (enhanced with absolute levels)
            atr_signal = indicator_data.get('signal', 'NEUTRAL')
            signals[indicator_name] = atr_signal
            
            # Generate detailed volatility analysis
            volatility_analysis = {
                'atr_value': indicator_data.get('value'),
                'combined_signal': atr_signal,
                'trend_signal': None,
                'level_signal': None,
                'interpretation': {
                    'volatility_trend': None,
                    'volatility_level': None,
                    'trading_implications': []
                }
            }
            
            # Parse ATR signal components
            signal_components = atr_signal.split(' | ') if ' | ' in atr_signal else [atr_signal]
            volatility_analysis['trend_signal'] = signal_components[0] if signal_components else 'NEUTRAL'
            volatility_analysis['level_signal'] = signal_components[1] if len(signal_components) > 1 else None
            
            # Interpret trend signals
            trend_signal = volatility_analysis['trend_signal']
            if 'INCREASING' in trend_signal:
                volatility_analysis['interpretation']['volatility_trend'] = 'Rising volatility - expect larger price movements'
                if 'HIGH_VOLATILITY' in trend_signal:
                    volatility_analysis['interpretation']['trading_implications'].extend([
                        'Consider wider stop losses',
                        'Potential for larger profits and losses'
                    ])
            elif 'DECREASING' in trend_signal:
                volatility_analysis['interpretation']['volatility_trend'] = 'Falling volatility - expect smaller price movements'
                volatility_analysis['interpretation']['trading_implications'].extend([
                    'Consider tighter stop losses',
                    'Range-bound trading may be more suitable'
                ])
            else:
                volatility_analysis['interpretation']['volatility_trend'] = 'Stable volatility'
            
            # Interpret level signals
            level_signal = volatility_analysis['level_signal']
            if level_signal:
                if 'EXTREMELY_HIGH' in level_signal:
                    volatility_analysis['interpretation']['volatility_level'] = 'Historically extremely high volatility'
                    volatility_analysis['interpretation']['trading_implications'].extend([
                        'Exercise extreme caution',
                        'Consider reducing position sizes',
                        'Potential for significant price gaps'
                    ])
                elif 'HIGH_VOLATILITY' in level_signal:
                    volatility_analysis['interpretation']['volatility_level'] = 'Above normal volatility'
                    volatility_analysis['interpretation']['trading_implications'].extend([
                        'Increased risk and opportunity',
                        'Monitor positions closely'
                    ])
                elif 'EXTREMELY_LOW' in level_signal:
                    volatility_analysis['interpretation']['volatility_level'] = 'Historically extremely low volatility'
                    volatility_analysis['interpretation']['trading_implications'].extend([
                        'Potential volatility expansion ahead',
                        'Consider volatility breakout strategies'
                    ])
                elif 'LOW_VOLATILITY' in level_signal:
                    volatility_analysis['interpretation']['volatility_level'] = 'Below normal volatility'
                    volatility_analysis['interpretation']['trading_implications'].append('Range-bound market conditions likely')
                elif 'NORMAL' in level_signal:
                    volatility_analysis['interpretation']['volatility_level'] = 'Normal volatility levels'
                    
        elif indicator_name == 'adx' and isinstance(indicator_data, dict):
            # ADX trend strength indication
            adx_value = indicator_data.get('adx', 0)
            plus_di = indicator_data.get('plus_di', 0)
            minus_di = indicator_data.get('minus_di', 0)
            
            if adx_value > 25:  # Strong trend
                if plus_di > minus_di:
                    signals[indicator_name] = 'STRONG_UPTREND'
                else:
                    signals[indicator_name] = 'STRONG_DOWNTREND'
            else:
                signals[indicator_name] = 'WEAK_TREND'
        
        elif indicator_name == 'ichimoku' and isinstance(indicator_data, dict):
            # Ichimoku cloud signal (simplified)
            conversion_line = indicator_data.get('conversion_line')
            base_line = indicator_data.get('base_line')
            
            if conversion_line and base_line:
                if conversion_line > base_line:
                    signals[indicator_name] = 'BULLISH'
                else:
                    signals[indicator_name] = 'BEARISH'
            else:
                signals[indicator_name] = 'NEUTRAL'
        
        elif indicator_name == 'fibonacci' and isinstance(indicator_data, dict):
            # Fibonacci retracement signal logic
            signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
        
        elif indicator_name == 'vwap' and isinstance(indicator_data, dict):
            # Enhanced VWAP signal logic with detailed analysis
            vwap_signal = indicator_data.get('signal', 'NEUTRAL')
            vwap_value = indicator_data.get('value')
            signals[indicator_name] = vwap_signal

            # Add VWAP-specific analysis to volatility_analysis if it doesn't exist
            if not volatility_analysis:
                # Get current price for VWAP analysis
                current_price = current_price_data.get('price_data', {}).get('price') if current_price_data else None

                if vwap_value and current_price:
                    vwap_analysis = {
                        'vwap_value': vwap_value,
                        'current_price': current_price,
                        'price_vs_vwap_percentage': ((current_price - vwap_value) / vwap_value) * 100 if vwap_value != 0 else 0,
                        'signal': vwap_signal,
                        'interpretation': {
                            'price_position': None,
                            'strength': None,
                            'trading_implications': []
                        }
                    }

                    # Interpret VWAP signals
                    if 'STRONG_BULLISH' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'Significantly above VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Strong bullish momentum'
                        vwap_analysis['interpretation']['trading_implications'].extend([
                            'Strong buying pressure',
                            'Price trading well above institutional average',
                            'Momentum continuation likely'
                        ])
                    elif 'BULLISH_ABOVE_VWAP' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'Above VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Moderate bullish'
                        vwap_analysis['interpretation']['trading_implications'].extend([
                            'Price above institutional average',
                            'Bullish bias present'
                        ])
                    elif 'SLIGHTLY_ABOVE_VWAP' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'Slightly above VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Weak bullish'
                        vwap_analysis['interpretation']['trading_implications'].append('Near institutional average - watch for direction')
                    elif 'STRONG_BEARISH' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'Significantly below VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Strong bearish momentum'
                        vwap_analysis['interpretation']['trading_implications'].extend([
                            'Strong selling pressure',
                            'Price trading well below institutional average',
                            'Downward momentum likely to continue'
                        ])
                    elif 'BEARISH_BELOW_VWAP' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'Below VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Moderate bearish'
                        vwap_analysis['interpretation']['trading_implications'].extend([
                            'Price below institutional average',
                            'Bearish bias present'
                        ])
                    elif 'SLIGHTLY_BELOW_VWAP' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'Slightly below VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Weak bearish'
                        vwap_analysis['interpretation']['trading_implications'].append('Near institutional average - watch for direction')
                    elif 'AT_VWAP' in vwap_signal:
                        vwap_analysis['interpretation']['price_position'] = 'At VWAP'
                        vwap_analysis['interpretation']['strength'] = 'Neutral'
                        vwap_analysis['interpretation']['trading_implications'].extend([
                            'Price at institutional average',
                            'Decision point - watch for breakout direction'
                        ])
        
        elif indicator_name == 'pivot_points' and isinstance(indicator_data, dict):
            # Enhanced Pivot Points signal logic with detailed analysis
            pivot_signal = indicator_data.get('signal', 'NEUTRAL')
            pivot_type = indicator_data.get('type', 'standard')
            signals[indicator_name] = pivot_signal

            # Add pivot points analysis to volatility_analysis if it doesn't exist or extend existing one
            pivot_analysis = {
                'pivot_type': pivot_type,
                'pivot_point': indicator_data.get('pivot_point'),
                'support_levels': indicator_data.get('support_levels', {}),
                'resistance_levels': indicator_data.get('resistance_levels', {}),
                'current_price': indicator_data.get('current_price'),
                'signal': pivot_signal,
                'interpretation': {
                    'position_relative_to_pivot': None,
                    'nearest_level': None,
                    'trading_implications': []
                }
            }

            # Interpret pivot points signals
            if 'ABOVE_PIVOT' in pivot_signal:
                pivot_analysis['interpretation']['position_relative_to_pivot'] = 'Price trading above pivot point'
                if 'BULLISH' in pivot_signal:
                    pivot_analysis['interpretation']['trading_implications'].extend([
                        'Bullish bias - price above pivot',
                        'Look for long opportunities',
                        'Resistance levels become targets'
                    ])
            elif 'BELOW_PIVOT' in pivot_signal:
                pivot_analysis['interpretation']['position_relative_to_pivot'] = 'Price trading below pivot point'
                if 'BEARISH' in pivot_signal:
                    pivot_analysis['interpretation']['trading_implications'].extend([
                        'Bearish bias - price below pivot',
                        'Look for short opportunities',
                        'Support levels become targets'
                    ])
            elif 'AT_PIVOT_POINT' in pivot_signal:
                pivot_analysis['interpretation']['position_relative_to_pivot'] = 'Price at pivot point'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    'Decision point - watch for direction',
                    'Potential reversal or continuation zone'
                ])

            # Handle support and resistance level signals
            if 'AT_SUPPORT' in pivot_signal:
                level_match = [part for part in pivot_signal.split('_') if part.isdigit()]
                level_num = level_match[0] if level_match else '1'
                pivot_analysis['interpretation']['nearest_level'] = f'At Support Level {level_num}'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    f'Price testing S{level_num} support',
                    'Potential bounce zone',
                    'Watch for reversal signals'
                ])
            elif 'NEAR_SUPPORT' in pivot_signal:
                level_match = [part for part in pivot_signal.split('_') if part.isdigit()]
                level_num = level_match[0] if level_match else '1'
                pivot_analysis['interpretation']['nearest_level'] = f'Near Support Level {level_num}'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    f'Approaching S{level_num} support',
                    'Prepare for potential support test'
                ])
            elif 'AT_RESISTANCE' in pivot_signal:
                level_match = [part for part in pivot_signal.split('_') if part.isdigit()]
                level_num = level_match[0] if level_match else '1'
                pivot_analysis['interpretation']['nearest_level'] = f'At Resistance Level {level_num}'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    f'Price testing R{level_num} resistance',
                    'Potential rejection zone',
                    'Watch for reversal signals'
                ])
            elif 'NEAR_RESISTANCE' in pivot_signal:
                level_match = [part for part in pivot_signal.split('_') if part.isdigit()]
                level_num = level_match[0] if level_match else '1'
                pivot_analysis['interpretation']['nearest_level'] = f'Near Resistance Level {level_num}'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    f'Approaching R{level_num} resistance',
                    'Prepare for potential resistance test'
                ])
            elif 'BULLISH_BREAKOUT' in pivot_signal:
                pivot_analysis['interpretation']['nearest_level'] = 'Breaking above resistance'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    'Bullish breakout in progress',
                    'Higher levels likely targets',
                    'Momentum continuation expected'
                ])
            elif 'BEARISH_BREAKDOWN' in pivot_signal:
                pivot_analysis['interpretation']['nearest_level'] = 'Breaking below support'
                pivot_analysis['interpretation']['trading_implications'].extend([
                    'Bearish breakdown in progress',
                    'Lower levels likely targets',
                    'Downward momentum expected'
                ])
            # Add VWAP analysis to response (you can modify this based on your needs)
            if not volatility_analysis:
                volatility_analysis = vwap_analysis
            else:
                # If volatility_analysis exists, add VWAP data to it
                volatility_analysis['vwap_analysis'] = vwap_analysis

    # Prepare response data
    response_data = {
        'timestamp': datetime.now().isoformat(),
        'signals': signals,
        'summary': {
            'buy_signals': len([s for s in signals.values() if s in ['BUY', 'BULLISH_CROSSOVER', 'BULLISH']]),
            'sell_signals': len([s for s in signals.values() if s in ['SELL', 'BEARISH_CROSSOVER', 'BEARISH']]),
            'neutral_signals': len([s for s in signals.values() if s == 'NEUTRAL']),
            'overbought_signals': len([s for s in signals.values() if s == 'OVERBOUGHT']),
            'oversold_signals': len([s for s in signals.values() if s == 'OVERSOLD']),
            'strong_trend_signals': len([s for s in signals.values() if 'STRONG' in s]),
            'weak_trend_signals': len([s for s in signals.values() if 'WEAK' in s]),
            'high_volatility_signals': len([s for s in signals.values() if 'HIGH_VOLATILITY' in s or 'EXTREMELY_HIGH_VOLATILITY' in s]),
            'low_volatility_signals': len([s for s in signals.values() if 'LOW_VOLATILITY' in s or 'EXTREMELY_LOW_VOLATILITY' in s]),
            'increasing_volatility_signals': len([s for s in signals.values() if 'INCREASING' in s]),
            'decreasing_volatility_signals': len([s for s in signals.values() if 'DECREASING' in s]),
            'stable_volatility_signals': len([s for s in signals.values() if 'STABLE_VOLATILITY' in s]),
            'fibonacci_support_signals': len([s for s in signals.values() if 'SUPPORT' in s]),
            'fibonacci_resistance_signals': len([s for s in signals.values() if 'RESISTANCE' in s]),
            'fibonacci_retracement_signals': len([s for s in signals.values() if 'RETRACEMENT' in s]),
            'fibonacci_continuation_signals': len([s for s in signals.values() if 'CONTINUATION' in s]),
            'vwap_bullish_signals': len([s for s in signals.values() if 'BULLISH' in s and 'VWAP' in s]),
            'vwap_bearish_signals': len([s for s in signals.values() if 'BEARISH' in s and 'VWAP' in s]),
            'above_vwap_signals': len([s for s in signals.values() if 'ABOVE_VWAP' in s]),
            'below_vwap_signals': len([s for s in signals.values() if 'BELOW_VWAP' in s]),
            'at_vwap_signals': len([s for s in signals.values() if 'AT_VWAP' in s]),
            'pivot_bullish_signals': len([s for s in signals.values() if 'BULLISH' in s and ('PIVOT' in s or 'BREAKOUT' in s)]),
            'pivot_bearish_signals': len([s for s in signals.values() if 'BEARISH' in s and ('PIVOT' in s or 'BREAKDOWN' in s)]),
            'at_support_signals': len([s for s in signals.values() if 'AT_SUPPORT' in s]),
            'at_resistance_signals': len([s for s in signals.values() if 'AT_RESISTANCE' in s]),
            'near_support_signals': len([s for s in signals.values() if 'NEAR_SUPPORT' in s]),
            'near_resistance_signals': len([s for s in signals.values() if 'NEAR_RESISTANCE' in s]),
            'breakout_signals': len([s for s in signals.values() if 'BREAKOUT' in s]),
            'breakdown_signals': len([s for s in signals.values() if 'BREAKDOWN' in s]),
            'at_pivot_signals': len([s for s in signals.values() if 'AT_PIVOT_POINT' in s]),

        }
    }

    # Add volatility analysis if available
    if volatility_analysis:
        response_data['analysis'] = volatility_analysis
    elif 'pivot_analysis' not in volatility_analysis:
        volatility_analysis['pivot_analysis'] = pivot_analysis

    return jsonify(response_data)

@app.route('/api/ai/latest-analysis', methods=['GET'])
def get_latest_ai_analysis():
    """Get the latest AI analysis"""
    global ai_batcher
    
    with ai_batcher.ai_response_lock:
        if ai_batcher.latest_ai_response:
            return jsonify(ai_batcher.latest_ai_response)
        else:
            return jsonify({
                'error': 'No AI analysis available',
                'message': 'No batches have been processed yet'
            }), 404

@app.route('/api/ai/analysis-history', methods=['GET'])
def get_ai_analysis_history():
    """Get AI analysis history"""
    global ai_batcher
    
    limit = request.args.get('limit', 5, type=int)
    
    with ai_batcher.ai_response_lock:
        history = ai_batcher.ai_response_history[-limit:] if ai_batcher.ai_response_history else []
        
        return jsonify({
            'total_analyses': len(ai_batcher.ai_response_history),
            'returned_count': len(history),
            'analyses': history
        })

@app.route('/api/ai/status', methods=['GET'])
def get_ai_status():
    """Get AI communication status"""
    global ai_batcher
    
    try:
        current_batch_size = 0
        batch_start_time_iso = None
        
        with ai_batcher.batch_lock:
            current_batch_size = len(ai_batcher.current_batch)
            if ai_batcher.current_batch_start_time:
                batch_start_time_iso = ai_batcher.current_batch_start_time.isoformat()
        
        with ai_batcher.ai_response_lock:
            has_latest_analysis = ai_batcher.latest_ai_response is not None
            history_count = len(ai_batcher.ai_response_history)
        
        # Safe access to last_batch_sent_time
        last_batch_sent_iso = None
        if ai_batcher.last_batch_sent_time:
            last_batch_sent_iso = ai_batcher.last_batch_sent_time.isoformat()
        
        status_data = {
            'service': 'OpenAI API',
            'client_available': ai_batcher.openai_client is not None,
            'communication_status': ai_batcher.ai_communication_status,
            'batch_count': ai_batcher.batch_count,
            'last_batch_sent': last_batch_sent_iso,
            'current_batch_size': current_batch_size,
            'batch_start_time': batch_start_time_iso,
            'has_latest_analysis': has_latest_analysis,
            'history_count': history_count,
            'batch_duration_seconds': ai_batcher.batch_duration,
            'model': 'gpt-4',
            'max_data_points_per_batch': ai_batcher.max_data_points,
            'max_history_length': ai_batcher.max_history_length
        }
        
        return jsonify(status_data)
        
    except AttributeError as ae:
        logger.error(f"AttributeError in AI status endpoint: {ae}")
        return jsonify({
            'error': 'AI batcher not properly initialized',
            'message': f'Missing attribute: {str(ae)}',
            'service': 'OpenAI API',
            'status': 'initialization_error'
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error in AI status endpoint: {e}")
        return jsonify({
            'error': 'Failed to get AI status',
            'message': str(e),
            'service': 'OpenAI API',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Technical Indicators API'
    })

if __name__ == '__main__':
    # Default configuration
    default_config = {
        'sma_periods': [20, 50, 200],
        'ema_periods': [12, 26, 50],
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'adx_period': 14,
        'psar_acceleration': 0.02,
        'psar_maximum': 0.2,
        'ichimoku_conversion': 9,
        'ichimoku_base': 26,
        'ichimoku_span_b': 52,
        'ichimoku_displacement': 26,
        'supertrend_period': 10,
        'supertrend_multiplier': 3.0,
        'stochastic_k_period': 14,  
        'stochastic_d_period': 3,
        'cci_period': 20,
        'roc_period': 14,
        'momentum_period': 10,
        'momentum_threshold_multiplier': 0.02,
        'atr_period': 14,
        'fibonacci_period': 20,
        'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
        'vwap_period': 20,
        'pivot_points_type': 'standard',
    }
    
    try:
        # Start WebSocket client automatically
        websocket_client = TradingDataClient(config=default_config)
        client_thread = WebSocketThread('ws://trading-api:8080/ws', default_config)
        client_thread.start()
        
        logger.info("Technical Indicators API Server starting...")
        logger.info("WebSocket client thread started")
        
        # Wait a moment for the thread to start
        import time
        time.sleep(2)
        
    except Exception as e:
        logger.error(f"Failed to start WebSocket client: {e}")
        logger.info("API will start without WebSocket connection")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 