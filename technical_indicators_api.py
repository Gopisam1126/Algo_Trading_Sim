#!/usr/bin/env python3
"""
REST API Server for Technical Indicators
Provides JSON endpoints for AI systems to access technical indicator data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import asyncio
import websockets
import json
import threading
import time
from datetime import datetime
import logging
from technical_indicators import TechnicalIndicators, TradingDataClient
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables to store current data
current_indicators = {}
current_price_data = {}
websocket_client = None
client_thread = None
running = False

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

# @app.route('/api/indicators/signals', methods=['GET'])
# def get_trading_signals():
#     """Get trading signals from all indicators"""
#     global current_indicators
    
#     if not current_indicators:
#         return jsonify({
#             'error': 'No data available',
#             'message': 'Technical indicators not yet calculated'
#         }), 404
    
#     signals = {}
    
#     # Extract signals from indicators
#     for indicator_name, indicator_data in current_indicators.items():
#         if isinstance(indicator_data, dict) and 'signal' in indicator_data:
#             signals[indicator_name] = indicator_data['signal']
#         elif indicator_name == 'rsi' and isinstance(indicator_data, dict) and 'value' in indicator_data:
#             # RSI signal logic
#             rsi_value = indicator_data['value']
#             if rsi_value > 70:
#                 signals[indicator_name] = 'OVERBOUGHT'
#             elif rsi_value < 30:
#                 signals[indicator_name] = 'OVERSOLD'
#             else:
#                 signals[indicator_name] = 'NEUTRAL'
#         elif indicator_name == 'macd' and isinstance(indicator_data, dict):
#             # MACD signal logic
#             if 'macd_line' in indicator_data and 'signal_line' in indicator_data:
#                 macd_line = indicator_data['macd_line']
#                 signal_line = indicator_data['signal_line']
#                 if macd_line > signal_line:
#                     signals[indicator_name] = 'BUY'
#                 else:
#                     signals[indicator_name] = 'SELL'
#         elif indicator_name == 'stochastic' and isinstance(indicator_data, dict):
#             signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
#         elif indicator_name == 'cci' and isinstance(indicator_data, dict):
#             signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
#         elif indicator_name == 'roc' and isinstance(indicator_data, dict):
#             signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
#         elif indicator_name == 'parabolic_sar' and isinstance(indicator_data, dict):
#             # Parabolic SAR signal logic
#             signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
#         elif indicator_name == 'supertrend' and isinstance(indicator_data, dict):
#             # SuperTrend signal logic
#             signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
#         elif indicator_name == 'momentum' and isinstance(indicator_data, dict):
#             signals[indicator_name] = indicator_data.get('signal', 'NEUTRAL')
#         elif indicator_name == 'adx' and isinstance(indicator_data, dict):
#             # ADX trend strength indication
#             adx_value = indicator_data.get('adx', 0)
#             plus_di = indicator_data.get('plus_di', 0)
#             minus_di = indicator_data.get('minus_di', 0)
            
#             if adx_value > 25:  # Strong trend
#                 if plus_di > minus_di:
#                     signals[indicator_name] = 'STRONG_UPTREND'
#                 else:
#                     signals[indicator_name] = 'STRONG_DOWNTREND'
#             else:
#                 signals[indicator_name] = 'WEAK_TREND'
#         elif indicator_name == 'ichimoku' and isinstance(indicator_data, dict):
#             # Ichimoku cloud signal (simplified)
#             conversion_line = indicator_data.get('conversion_line')
#             base_line = indicator_data.get('base_line')
            
#             if conversion_line and base_line:
#                 if conversion_line > base_line:
#                     signals[indicator_name] = 'BULLISH'
#                 else:
#                     signals[indicator_name] = 'BEARISH'
#             else:
#                 signals[indicator_name] = 'NEUTRAL'
    
#     return jsonify({
#         'timestamp': datetime.now().isoformat(),
#         'signals': signals,
#         'summary': {
#             'buy_signals': len([s for s in signals.values() if s in ['BUY', 'BULLISH_CROSSOVER', 'BULLISH']]),
#             'sell_signals': len([s for s in signals.values() if s in ['SELL', 'BEARISH_CROSSOVER', 'BEARISH']]),
#             'neutral_signals': len([s for s in signals.values() if s == 'NEUTRAL']),
#             'overbought_signals': len([s for s in signals.values() if s == 'OVERBOUGHT']),
#             'oversold_signals': len([s for s in signals.values() if s == 'OVERSOLD']),
#             'strong_trend_signals': len([s for s in signals.values() if 'STRONG' in s]),
#             'weak_trend_signals': len([s for s in signals.values() if 'WEAK' in s])
#         }
#     })

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

        return jsonify({
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
                'stable_volatility_signals': len([s for s in signals.values() if 'STABLE_VOLATILITY' in s])
            }
        })

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