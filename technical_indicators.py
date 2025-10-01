#!/usr/bin/env python3
"""
Comprehensive Technical Indicators Calculator
All indicators are configurable and support both real-time and historical calculations
"""

import asyncio
import math
import websockets
import json
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
import logging
import signal
import sys
from typing import Dict, List, Optional, Tuple, Any
from itertools import islice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Comprehensive technical indicators calculator with configurable parameters"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default configuration
        self.config = {
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
            'fibonacci_period': 20,  # Period to look back for swing high/low
            'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],  # Standard Fibonacci levels
            'vwap_period': 20,
            'pivot_points_type': 'standard',

        }
        
        if config:
            self.config.update(config)
        
        # Initialize data structures
        self.initialize_data_structures()
        
    def initialize_data_structures(self):
        """Initialize all data structures for indicators"""
        max_period = max(
            self.config['sma_periods'] + 
            self.config['ema_periods'] + 
            [self.config['macd_slow'], self.config['rsi_period'], 
             self.config['adx_period'], self.config['ichimoku_span_b'], 
             self.config['cci_period'], self.config['roc_period'],
             self.config['momentum_period'], self.config['fibonacci_period'],
             self.config['vwap_period']]
        )
        
        # Price data
        self.prices = deque(maxlen=max_period + 100)
        self.highs = deque(maxlen=max_period + 100)
        self.lows = deque(maxlen=max_period + 100)
        self.opens = deque(maxlen=max_period + 100)
        self.volumes = deque(maxlen=max_period + 100)
        
        # Indicator values storage
        self.sma_values = {period: deque(maxlen=100) for period in self.config['sma_periods']}
        self.ema_values = {period: deque(maxlen=100) for period in self.config['ema_periods']}
        self.macd_values = deque(maxlen=100)
        self.rsi_values = deque(maxlen=100)
        self.adx_values = deque(maxlen=100)
        self.psar_values = deque(maxlen=100)
        self.ichimoku_values = deque(maxlen=100)
        self.supertrend_values = deque(maxlen=100)
        self.stochastic_values = deque(maxlen=100)
        self.cci_values = deque(maxlen=100)
        self.roc_values = deque(maxlen=100)
        self.momentum_values = deque(maxlen=100)
        self.atr_values = deque(maxlen=100)
        self.fibonacci_values = deque(maxlen=100)
        self.vwap_values = deque(maxlen=100)
        self.pivot_points_values = deque(maxlen=100)
        
        # Initialize indicator states
        self.initialize_indicator_states()
    
    def initialize_indicator_states(self):
        """Initialize state variables for indicators"""
        # MACD state
        self.fast_ema = None
        self.slow_ema = None
        self.macd_line = None
        self.signal_line = None
        self.histogram = None
        
        # RSI state
        self.rsi_gains = deque(maxlen=self.config['rsi_period'])
        self.rsi_losses = deque(maxlen=self.config['rsi_period'])
        self.avg_gain = None
        self.avg_loss = None
        
        # ADX state
        self.plus_dm = deque(maxlen=self.config['adx_period'])
        self.minus_dm = deque(maxlen=self.config['adx_period'])
        self.tr_values = deque(maxlen=self.config['adx_period'])
        self.plus_di = None
        self.minus_di = None
        self.adx = None
        
        # Parabolic SAR state
        self.psar = None
        self.psar_af = self.config['psar_acceleration']
        self.psar_ep = None
        self.psar_long = True
        
        # Ichimoku state
        self.ichimoku_conversion_line = None
        self.ichimoku_base_line = None
        self.ichimoku_span_a = None
        self.ichimoku_span_b = None
        
        # SuperTrend state
        self.supertrend_atr = None
        self.supertrend_upper = None
        self.supertrend_lower = None
        self.supertrend_direction = 1  # 1 for uptrend, -1 for downtrend

        # Stochastic oscillator state
        self.stochastic_k_values = deque(maxlen=self.config['stochastic_d_period'])
        self.stochastic_k = None
        self.stochastic_d = None

        # CCI state
        self.typical_prices = deque(maxlen=self.config['cci_period'])
        self.cci_sma = None

        # ROC state
        self.roc_values = deque(maxlen=100)

        # ATR state
        self.atr_tr_values = deque(maxlen=self.config['atr_period'])
        self.atr = None

        # Fibonacci Retracement
        self.fibonacci_swing_high = None
        self.fibonacci_swing_low = None
        self.fibonacci_levels_cache = {}
        self.fibonacci_trend_direction = None

        # VMAP
        self.vwap_cumulative_volume = 0
        self.vwap_cumulative_pv = 0  # Price * Volume
        self.vwap_period_volumes = deque(maxlen=self.config['vwap_period'])
        self.vwap_period_pv = deque(maxlen=self.config['vwap_period'])

        # Pivot Points state
        self.pivot_points_data = None
        self.previous_day_high = None
        self.previous_day_low = None
        self.previous_day_close = None
    
    def add_data_point(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new data point and calculate all indicators"""
        # Extract data
        price = float(data.get('close', data.get('price', 0)))
        high = float(data.get('high', price))
        low = float(data.get('low', price))
        open_price = float(data.get('open', price))
        volume = int(data.get('volume', 0))
        
        # Add to data structures
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.opens.append(open_price)
        self.volumes.append(volume)
        
        # Calculate all indicators
        indicators = {}
        
        # Moving Averages
        indicators.update(self.calculate_moving_averages())
        
        # MACD
        macd_data = self.calculate_macd()
        if macd_data:
            indicators['macd'] = macd_data
        
        # RSI
        rsi = self.calculate_rsi()
        if rsi is not None:
            indicators['rsi'] = {
                'value': rsi,
                'signal': self.get_rsi_signal(rsi)
            }
        
        # ADX
        adx_data = self.calculate_adx()
        if adx_data:
            indicators['adx'] = adx_data
        
        # Parabolic SAR
        psar = self.calculate_parabolic_sar()
        if psar is not None:
            indicators['parabolic_sar'] = {
                'value': psar,
                'signal': 'BUY' if price > psar else 'SELL'
            }
        
        # Ichimoku Cloud
        ichimoku_data = self.calculate_ichimoku()
        if ichimoku_data:
            indicators['ichimoku'] = ichimoku_data
        
        # SuperTrend
        supertrend_data = self.calculate_supertrend()
        if supertrend_data:
            indicators['supertrend'] = supertrend_data

        # stochastic oscillator
        stochastic_data = self.calculate_stochastic_oscillator()
        if stochastic_data:
            indicators['stochastic'] = stochastic_data

        # CCI
        cci = self.calculate_cci()
        if cci is not None:
            indicators['cci'] = {
                'value': cci,
                'signal': self.get_cci_signal(cci)
            }

        # ROC
        roc = self.calculate_roc()
        if roc is not None:
            indicators['roc'] = {
                'value': roc,
                'signal': self.get_roc_signal(roc)
            }

        # Momentum Indicator
        momentum = self.calculate_momentum()
        if momentum is not None:
            indicators['momentum'] = {
                'value': momentum,
                'signal': self.get_momentum_signal(momentum)
            }

        atr = self.calculate_atr()
        if atr is not None:
            indicators['atr'] = {
                'value': atr,
                'signal': self.get_atr_signal(atr)
            }

        # Fibonacci Retracement
        fibonacci_data = self.calculate_fibonacci_retracement()
        if fibonacci_data:
            indicators['fibonacci'] = fibonacci_data

        # VWAP
        vwap = self.calculate_vwap()
        if vwap is not None:
            indicators['vwap'] = {
                'value': vwap,
                'signal': self.get_vwap_signal(vwap, price)
            }

        # Pivot Points
        pivot_points_data = self.calculate_pivot_points()
        if pivot_points_data:
            indicators['pivot_points'] = pivot_points_data
        
        return indicators

    def calculate_moving_averages(self) -> Dict[str, Any]:
        """Calculate SMA and EMA for all configured periods"""
        if len(self.prices) < min(self.config['sma_periods'] + self.config['ema_periods']):
            return {}

        result = {}

        # Calculate SMA using efficient islice method
        for period in self.config['sma_periods']:
            if len(self.prices) >= period:
                sma = sum(islice(self.prices, max(0, len(self.prices) - period), len(self.prices))) / period
                self.sma_values[period].append(sma)
                result[f'sma_{period}'] = sma

        # Calculate EMA using the helper function for consistency
        for period in self.config['ema_periods']:
            if len(self.prices) >= period:
                prev_ema = self.ema_values[period][-1] if self.ema_values[period] else None
                ema = self.calculate_ema(self.prices, period, prev_ema)
                if ema is not None:
                    self.ema_values[period].append(ema)
                    result[f'ema_{period}'] = ema

        return result

    def calculate_ema(self, prices: deque, period: int, prev_ema: Optional[float] = None) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None

        if prev_ema is None:
            # First EMA calculation - use efficient islice method
            return sum(islice(prices, max(0, len(prices) - period), len(prices))) / period
        else:
            # Subsequent EMA calculations
            multiplier = 2 / (period + 1)
            return (prices[-1] * multiplier) + (prev_ema * (1 - multiplier))

    def calculate_macd(self) -> Optional[Dict[str, float]]:
        """Calculate MACD indicator"""
        if len(self.prices) < self.config['macd_slow']:
            return None

        # Get or calculate fast EMA using the same method as moving averages
        fast_period = self.config['macd_fast']
        if fast_period in self.ema_values and self.ema_values[fast_period]:
            # Use existing EMA if it's already calculated in moving averages
            self.fast_ema = self.ema_values[fast_period][-1]
        else:
            # Calculate independently if not available
            self.fast_ema = self.calculate_ema(self.prices, fast_period, self.fast_ema)

        # Get or calculate slow EMA using the same method as moving averages
        slow_period = self.config['macd_slow']
        if slow_period in self.ema_values and self.ema_values[slow_period]:
            # Use existing EMA if it's already calculated in moving averages
            self.slow_ema = self.ema_values[slow_period][-1]
        else:
            # Calculate independently if not available
            self.slow_ema = self.calculate_ema(self.prices, slow_period, self.slow_ema)

        if self.fast_ema is None or self.slow_ema is None:
            return None

        # Calculate MACD line
        self.macd_line = self.fast_ema - self.slow_ema

        # Calculate signal line (EMA of MACD line)
        if self.macd_line is not None:
            self.macd_values.append(self.macd_line)

            if len(self.macd_values) >= self.config['macd_signal']:
                if self.signal_line is None:
                    # First signal line calculation using efficient islice method
                    self.signal_line = sum(islice(self.macd_values, max(0, len(self.macd_values) - self.config['macd_signal']), len(self.macd_values))) / self.config['macd_signal']
                else:
                    # Subsequent signal line calculations
                    multiplier = 2 / (self.config['macd_signal'] + 1)
                    self.signal_line = (self.macd_line * multiplier) + (self.signal_line * (1 - multiplier))

                # Calculate histogram
                self.histogram = self.macd_line - self.signal_line

                return {
                    'macd_line': self.macd_line,
                    'signal_line': self.signal_line,
                    'histogram': self.histogram,
                    'fast_ema': self.fast_ema,
                    'slow_ema': self.slow_ema
                }

        return None

    def calculate_rsi(self) -> Optional[float]:
        if len(self.prices) < 2:
            return None
        
        last_price = self.prices[-1]
        prev_price = self.prices[-2]

        # Safeguard 1: skip if bad tick (NaN, inf, or non-finite)
        if not (math.isfinite(last_price) and math.isfinite(prev_price)):
            return None
        
        # Calculate price change
        price_change = last_price - prev_price

        # Separate gains and losses
        gain = max(price_change, 0.0)
        loss = abs(min(price_change, 0.0))

        self.rsi_gains.append(gain)
        self.rsi_losses.append(loss)

        rsi_period = self.config['rsi_period']
        if len(self.rsi_gains) < rsi_period:
            return None
        
        # Safeguard 2: Initial or resync every 1000 updates
        if self.avg_gain is None or self.avg_loss is None or self.update_counter % 1000 == 0:
            self.avg_gain = sum(self.rsi_gains) / rsi_period
            self.avg_loss = sum(self.rsi_losses) / rsi_period
        else:
            # Wilder's smoothing
            self.avg_gain = ((self.avg_gain * (rsi_period - 1)) + gain) / rsi_period
            self.avg_loss = ((self.avg_loss * (rsi_period - 1)) + loss) / rsi_period

        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))

        self.rsi_values.append(rsi)

        # Maintain bounded deque size (safety in case config mismatch)
        if len(self.rsi_values) > 100:
            self.rsi_values.popleft()
            
        # Increment update counter (needed for drift resync)
        self.update_counter = getattr(self, "update_counter", 0) + 1
        return rsi
    
    def get_rsi_signal(self, rsi: float) -> str:
        """Get trading signal based on RSI value"""
        if rsi > 70:
            return "OVERBOUGHT"
        elif rsi < 30:
            return "OVERSOLD"
        else:
            return "NEUTRAL"
    
    def calculate_adx(self) -> Optional[Dict[str, float]]:
        """Calculate ADX indicator with optimal performance"""
        if len(self.highs) < 2 or len(self.lows) < 2:
            return None

        # Get current and previous values (cache for performance)
        high = self.highs[-1]
        low = self.lows[-1]
        prev_high = self.highs[-2]
        prev_low = self.lows[-2]
        prev_close = self.prices[-2]

        # Calculate True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        # Calculate Directional Movement
        high_diff = high - prev_high
        low_diff = prev_low - low

        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0.0

        # Initialize tracking variables
        period = self.config['adx_period']
        if not hasattr(self, '_adx_initialized'):
            # Initialize counters and accumulators for efficient rolling calculations
            self._adx_initialized = True
            self._tr_count = 0
            self._tr_sum = 0.0
            self._plus_dm_sum = 0.0
            self._minus_dm_sum = 0.0
            self._dx_count = 0
            self._dx_sum = 0.0

            # Initialize smoothed values
            self.smoothed_tr = None
            self.smoothed_plus_dm = None
            self.smoothed_minus_dm = None
            self.adx = None

            # Keep minimal history for rolling calculations
            self.recent_tr = deque(maxlen=period)
            self.recent_plus_dm = deque(maxlen=period)
            self.recent_minus_dm = deque(maxlen=period)
            self.recent_dx = deque(maxlen=period)

        # Update rolling sums
        self.recent_tr.append(tr)
        self.recent_plus_dm.append(plus_dm)
        self.recent_minus_dm.append(minus_dm)

        # Need at least period values to start calculations
        if len(self.recent_tr) < period:
            return None

        # Wilder's smoothing factor
        alpha = 1.0 / period

        # Calculate smoothed values using Wilder's smoothing
        if self.smoothed_tr is None:
            # First calculation: use current accumulated values
            # Since deque is exactly 'period' length, sum all elements
            self.smoothed_tr = sum(self.recent_tr) / period
            self.smoothed_plus_dm = sum(self.recent_plus_dm) / period
            self.smoothed_minus_dm = sum(self.recent_minus_dm) / period
        else:
            # Subsequent calculations: Wilder's smoothing (EMA)
            self.smoothed_tr = alpha * tr + (1 - alpha) * self.smoothed_tr
            self.smoothed_plus_dm = alpha * plus_dm + (1 - alpha) * self.smoothed_plus_dm
            self.smoothed_minus_dm = alpha * minus_dm + (1 - alpha) * self.smoothed_minus_dm

        # Calculate +DI and -DI
        if self.smoothed_tr == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = (self.smoothed_plus_dm / self.smoothed_tr) * 100
            minus_di = (self.smoothed_minus_dm / self.smoothed_tr) * 100

        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0.0
        else:
            dx = abs(plus_di - minus_di) / di_sum * 100

        # Update DX tracking
        self.recent_dx.append(dx)

        # Calculate ADX using Wilder's smoothing
        if self.adx is None:
            if len(self.recent_dx) >= period:
                # Initial ADX: average of accumulated DX values
                self.adx = sum(self.recent_dx) / len(self.recent_dx)
            else:
                return None
        else:
            # Subsequent ADX: Wilder's smoothing
            self.adx = alpha * dx + (1 - alpha) * self.adx

        # Maintain backward compatibility with existing attributes
        self.plus_di = plus_di
        self.minus_di = minus_di

        # Add to existing lists for backward compatibility (if they exist)
        if hasattr(self, 'tr_values'):
            self.tr_values.append(tr)
        if hasattr(self, 'plus_dm'):
            self.plus_dm.append(plus_dm)
        if hasattr(self, 'minus_dm'):
            self.minus_dm.append(minus_dm)
        if hasattr(self, 'adx_values'):
            self.adx_values.append(self.adx)

        return {
            'adx': self.adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx
        }
    
    def calculate_parabolic_sar(self) -> Optional[float]:
        """Calculate Parabolic SAR indicator"""
        if len(self.highs) < 2 or len(self.lows) < 2:
            return None
        
        high = self.highs[-1]
        low = self.lows[-1]
        
        if self.psar is None:
            # Initialize SAR
            self.psar = low if self.psar_long else high
            self.psar_ep = high if self.psar_long else low
            return self.psar
        
        # Update SAR
        if self.psar_long:
            # Long position
            if low < self.psar:
                # Switch to short
                self.psar_long = False
                self.psar = self.psar_ep
                self.psar_ep = low
                self.psar_af = self.config['psar_acceleration']
            else:
                # Continue long
                if high > self.psar_ep:
                    self.psar_ep = high
                    self.psar_af = min(self.psar_af + self.config['psar_acceleration'], self.config['psar_maximum'])
                
                self.psar = self.psar + self.psar_af * (self.psar_ep - self.psar)
                self.psar = min(self.psar, self.lows[-2], self.lows[-3] if len(self.lows) > 2 else self.lows[-2])
        else:
            # Short position
            if high > self.psar:
                # Switch to long
                self.psar_long = True
                self.psar = self.psar_ep
                self.psar_ep = high
                self.psar_af = self.config['psar_acceleration']
            else:
                # Continue short
                if low < self.psar_ep:
                    self.psar_ep = low
                    self.psar_af = min(self.psar_af + self.config['psar_acceleration'], self.config['psar_maximum'])
                
                self.psar = self.psar + self.psar_af * (self.psar_ep - self.psar)
                self.psar = max(self.psar, self.highs[-2], self.highs[-3] if len(self.highs) > 2 else self.highs[-2])
        
        self.psar_values.append(self.psar)
        return self.psar
    
    """TODO : The calculate_ichimoku needs better implementation. Current one uses lists for the calcualtion."""

    def calculate_ichimoku(self) -> Optional[Dict[str, float]]:
        """Calculate Ichimoku Cloud indicators"""
        if len(self.highs) < self.config['ichimoku_span_b'] or len(self.lows) < self.config['ichimoku_span_b']:
            return None
        
        # Calculate Conversion Line (Tenkan-sen)
        high_period = min(self.config['ichimoku_conversion'], len(self.highs))
        low_period = min(self.config['ichimoku_conversion'], len(self.lows))
        
        if high_period > 0 and low_period > 0:
            high_max = max(list(self.highs)[-high_period:])
            low_min = min(list(self.lows)[-low_period:])
            self.ichimoku_conversion_line = (high_max + low_min) / 2
        
        # Calculate Base Line (Kijun-sen)
        high_period = min(self.config['ichimoku_base'], len(self.highs))
        low_period = min(self.config['ichimoku_base'], len(self.lows))
        
        if high_period > 0 and low_period > 0:
            high_max = max(list(self.highs)[-high_period:])
            low_min = min(list(self.lows)[-low_period:])
            self.ichimoku_base_line = (high_max + low_min) / 2
        
        # Calculate Leading Span A (Senkou Span A)
        if self.ichimoku_conversion_line is not None and self.ichimoku_base_line is not None:
            self.ichimoku_span_a = (self.ichimoku_conversion_line + self.ichimoku_base_line) / 2
        
        # Calculate Leading Span B (Senkou Span B)
        high_period = min(self.config['ichimoku_span_b'], len(self.highs))
        low_period = min(self.config['ichimoku_span_b'], len(self.lows))
        
        if high_period > 0 and low_period > 0:
            high_max = max(list(self.highs)[-high_period:])
            low_min = min(list(self.lows)[-low_period:])
            self.ichimoku_span_b = (high_max + low_min) / 2
        
        if all(v is not None for v in [self.ichimoku_conversion_line, self.ichimoku_base_line, 
                                      self.ichimoku_span_a, self.ichimoku_span_b]):
            return {
                'conversion_line': self.ichimoku_conversion_line,
                'base_line': self.ichimoku_base_line,
                'span_a': self.ichimoku_span_a,
                'span_b': self.ichimoku_span_b
            }
        
        return None
    
    def calculate_atr(self) -> Optional[float]:
        """Calculate Average True Range (ATR) indicator"""
        if len(self.highs) < 2 or len(self.lows) < 2 or len(self.prices) < 2:
            return None
        
        # Calculate True Range for current period
        high = self.highs[-1]
        low = self.lows[-1]
        prev_close = self.prices[-2]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        # Store TR value
        self.atr_tr_values.append(tr)

        # Need at least the ATR period number of TR values
        if len(self.atr_tr_values) < self.config['atr_period']:
            return None
        # Calculate ATR as Simple Moving Average of True Range values
        if self.atr is None:
            # First ATR calculation - use simple average
            self.atr = sum(self.atr_tr_values) / len(self.atr_tr_values)
        else:
            # Subsequent ATR calculations using Wilder's method (exponential smoothing)
            # ATR = (Previous ATR * (n-1) + Current TR) / n
            self.atr = ((self.atr * (self.config['atr_period'] - 1)) + tr) / self.config['atr_period']
        self.atr_values.append(self.atr)
        
        return self.atr
        
    # Updation to handle the advanced ATR Signal
    def get_atr_signal(self, atr: float) -> str:
        """Get trading signal based on ATR value"""
        if len(self.atr_values) < 2:
            return "NEUTRAL"

        # Compare current ATR with previous ATR to determine volatility trend
        prev_atr = list(self.atr_values)[-2]
        atr_change = ((atr - prev_atr) / prev_atr) * 100

        # Calculate absolute ATR level signals
        atr_signal_components = []

        # Relative change signals (existing logic)
        if atr_change > 10:
            atr_signal_components.append("HIGH_VOLATILITY_INCREASING")
        elif atr_change > 5:
            atr_signal_components.append("VOLATILITY_INCREASING")
        elif atr_change < -10:
            atr_signal_components.append("VOLATILITY_DECREASING_SIGNIFICANTLY")
        elif atr_change < -5:
            atr_signal_components.append("VOLATILITY_DECREASING")
        else:
            atr_signal_components.append("STABLE_VOLATILITY")

        # Absolute ATR level signals (new feature)
        if len(self.atr_values) >= 20:  # Need sufficient history for meaningful comparison
            # Calculate average ATR over longer period (20 periods)
            atr_history = list(self.atr_values)[-20:]
            avg_atr = sum(atr_history) / len(atr_history)

            # Calculate standard deviation of ATR values
            atr_variance = sum((x - avg_atr) ** 2 for x in atr_history) / len(atr_history)
            atr_std = atr_variance ** 0.5

            # Define volatility levels based on standard deviations from mean
            if atr > avg_atr + (2 * atr_std):
                atr_signal_components.append("EXTREMELY_HIGH_VOLATILITY")
            elif atr > avg_atr + atr_std:
                atr_signal_components.append("HIGH_VOLATILITY_LEVEL")
            elif atr > avg_atr + (0.5 * atr_std):
                atr_signal_components.append("ABOVE_AVERAGE_VOLATILITY")
            elif atr < avg_atr - (2 * atr_std):
                atr_signal_components.append("EXTREMELY_LOW_VOLATILITY")
            elif atr < avg_atr - atr_std:
                atr_signal_components.append("LOW_VOLATILITY_LEVEL")
            elif atr < avg_atr - (0.5 * atr_std):
                atr_signal_components.append("BELOW_AVERAGE_VOLATILITY")
            else:
                atr_signal_components.append("NORMAL_VOLATILITY_LEVEL")

        # Combine signals with separator
        return " | ".join(atr_signal_components)
    
    def calculate_supertrend(self) -> Optional[Dict[str, Any]]:
        """Calculate SuperTrend indicator"""
        if len(self.highs) < self.config['supertrend_period'] or len(self.lows) < self.config['supertrend_period']:
            return None
        
        # Use the already calculated ATR if available, otherwise calculate it
        if self.atr is not None:
            atr = self.atr
        else:
            # Calculate ATR for SuperTrend
            tr_values = []
            for i in range(1, min(self.config['supertrend_period'] + 1, len(self.highs))):
                high = self.highs[-i]
                low = self.lows[-i]
                prev_close = self.prices[-i-1] if len(self.prices) > i else self.prices[-i]
                
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_values.append(tr)
            
            atr = sum(tr_values) / len(tr_values)
        
        # Calculate basic upper and lower bands
        hl_avg = (self.highs[-1] + self.lows[-1]) / 2
        basic_upper = hl_avg + (atr * self.config['supertrend_multiplier'])
        basic_lower = hl_avg - (atr * self.config['supertrend_multiplier'])
        
        # Initialize or update bands
        if self.supertrend_upper is None or self.supertrend_lower is None:
            # First calculation - initialize bands
            self.supertrend_upper = basic_upper
            self.supertrend_lower = basic_lower
            self.supertrend_direction = 1  # Start with uptrend assumption
        else:
            # Update upper band
            if basic_upper < self.supertrend_upper or self.prices[-2] > self.supertrend_upper:
                self.supertrend_upper = basic_upper
            else: self.supertrend_upper
            
            # Update lower band
            if basic_lower > self.supertrend_lower or self.prices[-2] < self.supertrend_lower:
                self.supertrend_lower = basic_lower
            else: self.supertrend_lower
        
        # Determine trend direction based on current price
        current_price = self.prices[-1]
        
        if len(self.prices) > 1:
            prev_price = self.prices[-2]
            
            # Check for trend change
            if self.supertrend_direction == -1 and current_price > self.supertrend_upper:
                self.supertrend_direction = 1  # Switch to uptrend
            elif self.supertrend_direction == 1 and current_price < self.supertrend_lower:
                self.supertrend_direction = -1  # Switch to downtrend
        
        # Calculate SuperTrend value based on current trend
        supertrend_value = self.supertrend_lower if self.supertrend_direction == 1 else self.supertrend_upper
        
        self.supertrend_values.append(supertrend_value)
        
        return {
            'value': supertrend_value,
            'direction': self.supertrend_direction,
            'upper_band': self.supertrend_upper,
            'lower_band': self.supertrend_lower,
            'atr': atr,
            'signal': 'BUY' if self.supertrend_direction == 1 else 'SELL'
        }
    
    def calculate_stochastic_oscillator(self) -> Optional[Dict[str, float]]:
        """Calculate Stochastic Oscillator (%K and %D)"""
        if len(self.highs) < self.config['stochastic_k_period'] or len(self.lows) < self.config['stochastic_k_period']:
            return None
        
        k_period = self.config['stochastic_k_period']
        d_period = self.config['stochastic_d_period']
        
        if len(self.prices) < k_period:
            return None
        
        # Initialize sliding window structures if not already done
        if not hasattr(self, '_stoch_min_deque'):
            # Sliding min/max structures - store (value, index) tuples
            self._stoch_min_deque = deque()
            self._stoch_max_deque = deque()
            self._stoch_current_idx = -1
            
            # Rolling sum structures for %D
            self._stoch_k_rolling = deque(maxlen=d_period)
            self._stoch_k_sum = 0.0
            
            # Previous values for crossover detection
            self._prev_k = None
            self._prev_d = None
            
            # Fixed-size deques to prevent unbounded growth
            self.stochastic_k_values = deque(maxlen=d_period)
            self.stochastic_values = deque(maxlen=100)
            
            # Populate sliding window with historical data
            # Start from the point where we have enough data for a full window
            start_idx = max(0, len(self.highs) - k_period)
            
            for i in range(start_idx, len(self.highs)):
                self._stoch_current_idx += 1
                high_val = self.highs[i]
                low_val = self.lows[i]
                
                # Maintain min deque (monotonic increasing)
                while self._stoch_min_deque and self._stoch_min_deque[-1][0] >= low_val:
                    self._stoch_min_deque.pop()
                self._stoch_min_deque.append((low_val, self._stoch_current_idx))
                
                # Maintain max deque (monotonic decreasing)
                while self._stoch_max_deque and self._stoch_max_deque[-1][0] <= high_val:
                    self._stoch_max_deque.pop()
                self._stoch_max_deque.append((high_val, self._stoch_current_idx))
                
                # Remove elements outside the current window
                window_start = self._stoch_current_idx - k_period + 1
                while self._stoch_min_deque and self._stoch_min_deque[0][1] < window_start:
                    self._stoch_min_deque.popleft()
                while self._stoch_max_deque and self._stoch_max_deque[0][1] < window_start:
                    self._stoch_max_deque.popleft()
        else:
            # Add only the latest data point
            self._stoch_current_idx += 1
            high_val = self.highs[-1]
            low_val = self.lows[-1]
            
            # Maintain min deque (monotonic increasing)
            while self._stoch_min_deque and self._stoch_min_deque[-1][0] >= low_val:
                self._stoch_min_deque.pop()
            self._stoch_min_deque.append((low_val, self._stoch_current_idx))
            
            # Maintain max deque (monotonic decreasing)
            while self._stoch_max_deque and self._stoch_max_deque[-1][0] <= high_val:
                self._stoch_max_deque.pop()
            self._stoch_max_deque.append((high_val, self._stoch_current_idx))
            
            # Remove elements outside the current window
            window_start = self._stoch_current_idx - k_period + 1
            while self._stoch_min_deque and self._stoch_min_deque[0][1] < window_start:
                self._stoch_min_deque.popleft()
            while self._stoch_max_deque and self._stoch_max_deque[0][1] < window_start:
                self._stoch_max_deque.popleft()
        
        # Ensure we have a full window before calculating
        if self._stoch_current_idx < k_period - 1:
            return None
        
        # Get current closing price and period high/low from deques
        current_price = self.prices[-1]
        highest_high = self._stoch_max_deque[0][0]
        lowest_low = self._stoch_min_deque[0][0]
        
        # Calculate %K
        if highest_high == lowest_low:
            # Flat price - skip calculation
            return None
        
        stochastic_k = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Clamp %K to valid range [0, 100]
        stochastic_k = max(0.0, min(100.0, stochastic_k))
        
        self.stochastic_k = stochastic_k
        self.stochastic_k_values.append(stochastic_k)
        
        # Calculate %D using rolling sum
        if len(self._stoch_k_rolling) == d_period:
            self._stoch_k_sum -= self._stoch_k_rolling[0]
            
        self._stoch_k_rolling.append(stochastic_k)
        self._stoch_k_sum += stochastic_k
        
        if len(self._stoch_k_rolling) == d_period:
            stochastic_d = self._stoch_k_sum / d_period
            self.stochastic_d = stochastic_d
            
            # Get signal with crossover detection
            signal = self.get_stochastic_signal(
                stochastic_k, 
                stochastic_d, 
                self._prev_k, 
                self._prev_d
            )
            
            # Store the complete stochastic data
            stochastic_data = {
                'k': stochastic_k,
                'd': stochastic_d,
                'signal': signal
            }
            
            self.stochastic_values.append(stochastic_data)
            
            # Update previous values for next crossover detection
            self._prev_k = stochastic_k
            self._prev_d = stochastic_d
            
            return stochastic_data
        
        # Return only %K if we don't have enough data for %D yet
        signal = self.get_stochastic_signal(stochastic_k, None, None, None)
        
        return {
            'k': stochastic_k,
            'd': None,
            'signal': signal
        }


    def get_stochastic_signal(self, k_value: float, d_value: Optional[float], 
                            prev_k: Optional[float], prev_d: Optional[float]) -> str:
        """Get trading signal based on Stochastic values with proper crossover detection"""
        
        # Detect crossovers first (requires previous values)
        if d_value is not None and prev_k is not None and prev_d is not None:
            # Bullish crossover: K crosses above D
            if prev_k <= prev_d and k_value > d_value:
                return "BULLISH_CROSSOVER"
            # Bearish crossover: K crosses below D
            if prev_k >= prev_d and k_value < d_value:
                return "BEARISH_CROSSOVER"
        
        # Overbought/Oversold conditions
        if k_value > 80:
            return "OVERBOUGHT"
        if k_value < 20:
            return "OVERSOLD"
        
        return "NEUTRAL"
    
    def calculate_cci(self) -> Optional[float]:
        """
        Calculate Commodity Channel Index (CCI).
        
        CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
        where Typical Price = (High + Low + Close) / 3
        """
        # Validate config
        try:
            period = self.config['cci_period']
        except KeyError:
            raise KeyError("Config must contain 'cci_period' key")
        
        # Validate input data existence and alignment
        if not (len(self.highs) >= period and len(self.lows) >= period and len(self.prices) >= period):
            return None
        
        if len(self.highs) != len(self.lows) or len(self.highs) != len(self.prices):
            raise ValueError(
                f"Input arrays must have equal length. "
                f"Got highs={len(self.highs)}, lows={len(self.lows)}, prices={len(self.prices)}"
            )
        
        # Validate numeric values
        if not all(isinstance(x, (int, float)) and not (x != x or abs(x) == float('inf')) 
                for x in [self.highs[-1], self.lows[-1], self.prices[-1]]):
            raise ValueError("Input arrays contain invalid values (NaN or Inf)")
        
        # Initialize deques on first call (should ideally be in __init__)
        if not hasattr(self, '_tp_deque'):
            self._tp_deque = deque(maxlen=period)
        if not hasattr(self, '_cci_deque'):
            self._cci_deque = deque(maxlen=1000)  # Memory-limited history
        
        # Calculate Typical Price (TP)
        typical_price = (self.highs[-1] + self.lows[-1] + self.prices[-1]) / 3
        self._tp_deque.append(typical_price)
        
        # Need at least the period number of typical prices
        if len(self._tp_deque) < period:
            return None
        
        # Calculate SMA and Mean Deviation in a single pass
        sma_tp = 0.0
        mean_deviation = 0.0
        
        for tp in self._tp_deque:
            sma_tp += tp
        sma_tp /= period
        
        for tp in self._tp_deque:
            mean_deviation += abs(tp - sma_tp)
        mean_deviation /= period
        
        # Calculate CCI with Lambert's constant (0.015)
        # Using 1e-9 as epsilon for practical floating-point comparison
        if mean_deviation < 1e-9:
            cci = 0.0
        else:
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        # Store CCI with memory management
        self._cci_deque.append(cci)
        
        return cci


    def get_cci_signal(self, cci: float) -> str:
        """
        Get trading signal based on CCI value.
        """
        if cci > 100:
            return "OVERBOUGHT"
        elif cci < -100:
            return "OVERSOLD"
        elif cci > 1e-9:
            return "BULLISH"
        elif cci < -1e-9:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def calculate_roc(self) -> Optional[float]:
        """Calculate Rate of Change (ROC) indicator"""
        if len(self.prices) < self.config['roc_period'] + 1:
            return None

        # Get current price and price n periods ago
        current_price = self.prices[-1]
        past_price = self.prices[-(self.config['roc_period'] + 1)]

        # Calculate ROC as percentage change
        if past_price == 0:
            roc = 0
        else:
            roc = ((current_price - past_price) / past_price) * 100

        self.roc_values.append(roc)
        return roc

    # 5. Add the get_roc_signal method
    def get_roc_signal(self, roc: float) -> str:
        """Get trading signal based on ROC value"""
        if roc > 10:
            return "STRONG_BULLISH"
        elif roc > 5:
            return "BULLISH"
        elif roc > 0:
            return "WEAK_BULLISH"
        elif roc < -10:
            return "STRONG_BEARISH"
        elif roc < -5:
            return "BEARISH"
        elif roc < 0:
            return "WEAK_BEARISH"
        else:
            return "NEUTRAL"
        
    
    def calculate_momentum(self) -> Optional[float]:
        """Calculate Momentum indicator"""
        if len(self.prices) < self.config['momentum_period'] + 1:
            return None

        # Get current price and price n periods ago
        current_price = self.prices[-1]
        past_price = self.prices[-(self.config['momentum_period'] + 1)]

        # Calculate Momentum as the difference
        momentum = current_price - past_price

        self.momentum_values.append(momentum)
        return momentum

    def get_momentum_signal(self, momentum: float) -> str:
        """Get trading signal based on Momentum value"""
        if len(self.prices) == 0:
            return "NEUTRAL"

        # Calculate dynamic threshold based on current price
        current_price = self.prices[-1]
        threshold = current_price * self.config['momentum_threshold_multiplier']
        strong_threshold = threshold * 2 #double for striong signals

        if momentum > 0:
            if momentum > strong_threshold:
                return "STRONG_BULLISH"
            elif momentum > threshold:
                return "BULLISH"
            else:
                return "WEAK_BULLISH"
        elif momentum < 0:
            if momentum < -strong_threshold:
                return "STRONG_BEARISH"
            elif momentum < -threshold:
                return "BEARISH"
            else:
                return "WEAK_BEARISH"
        else:
            return "NEUTRAL"
        
    # Fibonacci Retracement

    def calculate_fibonacci_retracement(self) -> Optional[Dict[str, Any]]:
        """Calculate Fibonacci Retracement levels"""
        if len(self.highs) < self.config['fibonacci_period'] or len(self.lows) < self.config['fibonacci_period']:
            return None

        # Get recent price data for the specified period
        recent_highs = list(self.highs)[-self.config['fibonacci_period']:]
        recent_lows = list(self.lows)[-self.config['fibonacci_period']:]
        recent_prices = list(self.prices)[-self.config['fibonacci_period']:]

        # Find swing high and swing low
        swing_high = max(recent_highs)
        swing_low = min(recent_lows)

        # Determine trend direction
        current_price = self.prices[-1]
        price_range = swing_high - swing_low

        if price_range == 0:
            return None

        # Determine if we're in an uptrend or downtrend
        # Check if current price is closer to swing high (uptrend) or swing low (downtrend)
        mid_point = (swing_high + swing_low) / 2
        trend_direction = 'up' if current_price > mid_point else 'down'

        # Calculate Fibonacci levels
        fibonacci_levels = {}

        if trend_direction == 'up':
            # In uptrend, calculate retracement levels from swing high
            for level in self.config['fibonacci_levels']:
                retracement_price = swing_high - (price_range * level)
                fibonacci_levels[f'fib_{level:.3f}'] = retracement_price
        else:
            # In downtrend, calculate extension levels from swing low
            for level in self.config['fibonacci_levels']:
                extension_price = swing_low + (price_range * level)
                fibonacci_levels[f'fib_{level:.3f}'] = extension_price

        # Store current values
        self.fibonacci_swing_high = swing_high
        self.fibonacci_swing_low = swing_low
        self.fibonacci_levels_cache = fibonacci_levels
        self.fibonacci_trend_direction = trend_direction

        # Determine which Fibonacci level is closest to current price
        closest_level = self.find_closest_fibonacci_level(current_price, fibonacci_levels)

        # Generate trading signal
        signal = self.get_fibonacci_signal(current_price, fibonacci_levels, trend_direction)

        fibonacci_data = {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'price_range': price_range,
            'trend_direction': trend_direction,
            'levels': fibonacci_levels,
            'closest_level': closest_level,
            'signal': signal,
            'current_price': current_price
        }

        self.fibonacci_values.append(fibonacci_data)
        return fibonacci_data

    def find_closest_fibonacci_level(self, current_price: float, fibonacci_levels: Dict[str, float]) -> Dict[str, Any]:
        """Find the closest Fibonacci level to current price"""
        if not fibonacci_levels:
            return None

        closest_distance = float('inf')
        closest_level_name = None
        closest_level_price = None

        for level_name, level_price in fibonacci_levels.items():
            distance = abs(current_price - level_price)
            if distance < closest_distance:
                closest_distance = distance
                closest_level_name = level_name
                closest_level_price = level_price

        return {
            'level_name': closest_level_name,
            'level_price': closest_level_price,
            'distance': closest_distance,
            'distance_percentage': (closest_distance / current_price) * 100
        }

    def get_fibonacci_signal(self, current_price: float, fibonacci_levels: Dict[str, float], trend_direction: str) -> str:
        """Get trading signal based on Fibonacci levels"""
        if not fibonacci_levels:
            return "NEUTRAL"

        # Get the closest level info
        closest_level = self.find_closest_fibonacci_level(current_price, fibonacci_levels)

        if not closest_level:
            return "NEUTRAL"

        distance_percentage = closest_level['distance_percentage']
        level_name = closest_level['level_name']
        level_price = closest_level['level_price']

        # Define proximity threshold (when price is very close to a Fibonacci level)
        proximity_threshold = 0.5  # 0.5% distance threshold

        # Strong signal when price is very close to key Fibonacci levels
        if distance_percentage < proximity_threshold:
            # 61.8% (Golden Ratio) and 38.2% are the most significant levels
            if '0.618' in level_name or '0.382' in level_name:
                if trend_direction == 'up' and current_price < level_price:
                    return "STRONG_SUPPORT_BOUNCE"
                elif trend_direction == 'down' and current_price > level_price:
                    return "STRONG_RESISTANCE_REJECTION"

            # 50% level (psychological level)
            elif '0.500' in level_name:
                if trend_direction == 'up' and current_price < level_price:
                    return "SUPPORT_BOUNCE"
                elif trend_direction == 'down' and current_price > level_price:
                    return "RESISTANCE_REJECTION"

            # Other levels (23.6%, 78.6%)
            else:
                if trend_direction == 'up' and current_price < level_price:
                    return "WEAK_SUPPORT"
                elif trend_direction == 'down' and current_price > level_price:
                    return "WEAK_RESISTANCE"

        # Medium distance signals
        elif distance_percentage < 2.0:
            if '0.618' in level_name or '0.382' in level_name:
                if trend_direction == 'up':
                    return "APPROACHING_SUPPORT" if current_price > level_price else "APPROACHING_RESISTANCE"
                else:
                    return "APPROACHING_RESISTANCE" if current_price > level_price else "APPROACHING_SUPPORT"

        # Trend continuation signals based on position relative to key levels
        fib_618 = fibonacci_levels.get('fib_0.618')
        fib_382 = fibonacci_levels.get('fib_0.382')

        if fib_618 and fib_382:
            if trend_direction == 'up':
                if current_price > fib_382:
                    return "TREND_CONTINUATION_BULLISH"
                elif current_price < fib_618:
                    return "RETRACEMENT_DEEP"
            else:
                if current_price < fib_382:
                    return "TREND_CONTINUATION_BEARISH"
                elif current_price > fib_618:
                    return "RETRACEMENT_DEEP"

        return "NEUTRAL"
    
    def calculate_vwap(self) -> Optional[float]:
        """Calculate Volume Weighted Average Price (VWAP)"""
        if len(self.prices) < 1 or len(self.volumes) < 1:
            return None

        current_price = self.prices[-1]
        current_volume = self.volumes[-1]

        if current_volume == 0:
            # If no volume data, return None or current price
            return None

        # Calculate typical price for current period
        if len(self.highs) >= 1 and len(self.lows) >= 1:
            typical_price = (self.highs[-1] + self.lows[-1] + current_price) / 3
        else:
            typical_price = current_price

        # Calculate Price * Volume for current period
        pv = typical_price * current_volume

        # Store current period data
        self.vwap_period_volumes.append(current_volume)
        self.vwap_period_pv.append(pv)

        # Calculate VWAP over the specified period
        if len(self.vwap_period_volumes) >= min(self.config['vwap_period'], 1):
            # Use available data up to the specified period
            total_volume = sum(self.vwap_period_volumes)
            total_pv = sum(self.vwap_period_pv)

            if total_volume == 0:
                return None

            vwap = total_pv / total_volume
            self.vwap_values.append(vwap)
            return vwap

        return None

    def get_vwap_signal(self, vwap: float, current_price: float) -> str:
        """Get trading signal based on VWAP value"""
        if vwap is None or current_price is None:
            return "NEUTRAL"

        # Calculate percentage difference between current price and VWAP
        price_diff_percentage = ((current_price - vwap) / vwap) * 100

        # Basic VWAP signals
        if current_price > vwap:
            if price_diff_percentage > 2.0:
                return "STRONG_BULLISH_ABOVE_VWAP"
            elif price_diff_percentage > 0.5:
                return "BULLISH_ABOVE_VWAP"
            else:
                return "SLIGHTLY_ABOVE_VWAP"
        elif current_price < vwap:
            if price_diff_percentage < -2.0:
                return "STRONG_BEARISH_BELOW_VWAP"
            elif price_diff_percentage < -0.5:
                return "BEARISH_BELOW_VWAP"
            else:
                return "SLIGHTLY_BELOW_VWAP"
        else:
            return "AT_VWAP"
        
    def calculate_pivot_points(self) -> Optional[Dict[str, Any]]:
        """Calculate Pivot Points with multiple calculation methods"""
        if len(self.highs) < 1 or len(self.lows) < 1 or len(self.prices) < 1:
            return None

        # Use the most recent complete day's data
        # For simplicity, we'll use the last available high, low, close
        # In production, you'd want to use actual previous day's OHLC data
        high = self.highs[-1]
        low = self.lows[-1]
        close = self.prices[-1]

        # Store previous day data
        self.previous_day_high = high
        self.previous_day_low = low
        self.previous_day_close = close

        pivot_type = self.config.get('pivot_points_type', 'standard')
        current_price = self.prices[-1]

        if pivot_type == 'standard':
            pivot_data = self.calculate_standard_pivot_points(high, low, close)
        elif pivot_type == 'fibonacci':
            pivot_data = self.calculate_fibonacci_pivot_points(high, low, close)
        elif pivot_type == 'woodie':
            pivot_data = self.calculate_woodie_pivot_points(high, low, close)
        elif pivot_type == 'camarilla':
            pivot_data = self.calculate_camarilla_pivot_points(high, low, close)
        else:
            pivot_data = self.calculate_standard_pivot_points(high, low, close)

        # Add signal analysis
        signal = self.get_pivot_points_signal(current_price, pivot_data)
        pivot_data['signal'] = signal
        pivot_data['current_price'] = current_price
        pivot_data['type'] = pivot_type

        self.pivot_points_values.append(pivot_data)
        return pivot_data

    def calculate_standard_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Standard Pivot Points"""
        # Pivot Point
        pp = (high + low + close) / 3
    
        # Support levels
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
    
        # Resistance levels
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
    
        return {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'support_3': s3,
            'resistance_1': r1,
            'resistance_2': r2,
            'resistance_3': r3
        }
    
    def calculate_fibonacci_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Fibonacci Pivot Points"""
        # Pivot Point
        pp = (high + low + close) / 3
        range_val = high - low
    
        # Fibonacci levels (38.2%, 61.8%)
        s1 = pp - (0.382 * range_val)
        s2 = pp - (0.618 * range_val)
        s3 = pp - (1.000 * range_val)
    
        r1 = pp + (0.382 * range_val)
        r2 = pp + (0.618 * range_val)
        r3 = pp + (1.000 * range_val)
    
        return {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'support_3': s3,
            'resistance_1': r1,
            'resistance_2': r2,
            'resistance_3': r3
        }
    
    def calculate_woodie_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Woodie's Pivot Points"""
        # Woodie's Pivot Point (gives more weight to closing price)
        pp = (high + low + (2 * close)) / 4
    
        # Support and Resistance levels
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
    
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
    
        return {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'support_3': s3,
            'resistance_1': r1,
            'resistance_2': r2,
            'resistance_3': r3
        }
    
    def calculate_camarilla_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Camarilla Pivot Points"""
        # Camarilla Pivot Point
        pp = close
        range_val = high - low
    
        # Camarilla levels use specific multipliers
        s1 = close - (range_val * 1.1/12)
        s2 = close - (range_val * 1.1/6)
        s3 = close - (range_val * 1.1/4)
        s4 = close - (range_val * 1.1/2)
    
        r1 = close + (range_val * 1.1/12)
        r2 = close + (range_val * 1.1/6)
        r3 = close + (range_val * 1.1/4)
        r4 = close + (range_val * 1.1/2)
    
        return {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'support_3': s3,
            'support_4': s4,
            'resistance_1': r1,
            'resistance_2': r2,
            'resistance_3': r3,
            'resistance_4': r4
        }
    
    def get_pivot_points_signal(self, current_price: float, pivot_data: Dict[str, float]) -> str:
        """Get trading signal based on Pivot Points"""
        pp = pivot_data['pivot_point']
        
        # Get support and resistance levels
        supports = [pivot_data.get(f'support_{i}') for i in range(1, 5) if pivot_data.get(f'support_{i}') is not None]
        resistances = [pivot_data.get(f'resistance_{i}') for i in range(1, 5) if pivot_data.get(f'resistance_{i}') is not None]
        
        # Remove None values and sort
        supports = [s for s in supports if s is not None]
        resistances = [r for r in resistances if r is not None]
        supports.sort(reverse=True)  # Sort descending (closest support first)
        resistances.sort()  # Sort ascending (closest resistance first)
    
        # Determine position relative to pivot point
        if current_price > pp:
            # Price above pivot - bullish bias
            
            # Check if near resistance levels
            for i, resistance in enumerate(resistances):
                distance_pct = abs(current_price - resistance) / current_price * 100
                if distance_pct < 0.1:  # Very close to resistance
                    return f"AT_RESISTANCE_{i+1}"
                elif distance_pct < 0.5:  # Near resistance
                    return f"NEAR_RESISTANCE_{i+1}"
            
            # Check if breaking through resistance
            if resistances and current_price > resistances[0]:
                return "BULLISH_BREAKOUT"
            
            return "BULLISH_ABOVE_PIVOT"
        
        elif current_price < pp:
            # Price below pivot - bearish bias
            
            # Check if near support levels
            for i, support in enumerate(supports):
                distance_pct = abs(current_price - support) / current_price * 100
                if distance_pct < 0.1:  # Very close to support
                    return f"AT_SUPPORT_{i+1}"
                elif distance_pct < 0.5:  # Near support
                    return f"NEAR_SUPPORT_{i+1}"
            
            # Check if breaking through support
            if supports and current_price < supports[0]:
                return "BEARISH_BREAKDOWN"
            
            return "BEARISH_BELOW_PIVOT"
        
        else:
            # Price at pivot point
            return "AT_PIVOT_POINT"
    

    def get_all_indicators(self) -> Dict[str, Any]:
        """Get all calculated indicators in a structured format"""
        return {
            'timestamp': datetime.now().isoformat(),
            'indicators': {
                'moving_averages': {
                    'sma': {f'sma_{period}': list(self.sma_values[period])[-1] if self.sma_values[period] else None 
                           for period in self.config['sma_periods']},
                    'ema': {f'ema_{period}': list(self.ema_values[period])[-1] if self.ema_values[period] else None 
                           for period in self.config['ema_periods']}
                },
                'macd': {
                    'macd_line': self.macd_line,
                    'signal_line': self.signal_line,
                    'histogram': self.histogram,
                    'fast_ema': self.fast_ema,
                    'slow_ema': self.slow_ema
                } if self.macd_line is not None else None,
                'rsi': {
                    'value': list(self.rsi_values)[-1] if self.rsi_values else None,
                    'signal': self.get_rsi_signal(list(self.rsi_values)[-1]) if self.rsi_values else None
                },
                'adx': {
                    'adx': self.adx,
                    'plus_di': self.plus_di,
                    'minus_di': self.minus_di
                } if self.adx is not None else None,
                'parabolic_sar': {
                    'value': list(self.psar_values)[-1] if self.psar_values else None,
                    'signal': 'BUY' if self.prices and self.psar_values and self.prices[-1] > self.psar_values[-1] else 'SELL'
                } if self.psar_values else None,
                'ichimoku': {
                    'conversion_line': self.ichimoku_conversion_line,
                    'base_line': self.ichimoku_base_line,
                    'span_a': self.ichimoku_span_a,
                    'span_b': self.ichimoku_span_b
                } if self.ichimoku_conversion_line is not None else None,
                'supertrend': {
                    'value': list(self.supertrend_values)[-1] if self.supertrend_values else None,
                    'direction': self.supertrend_direction,
                    'signal': 'BUY' if self.supertrend_direction == 1 else 'SELL'
                } if self.supertrend_values else None,
                'stochastic': {
                    'k': list(self.stochastic_values)[-1]['k'] if self.stochastic_values else None,
                    'd': list(self.stochastic_values)[-1]['d'] if self.stochastic_values else None,
                    'signal': list(self.stochastic_values)[-1]['signal'] if self.stochastic_values else None
                } if self.stochastic_values else None,
                'cci': {
                    'value': list(self.cci_values)[-1] if self.cci_values else None,
                    'signal': self.get_cci_signal(list(self.cci_values)[-1]) if self.cci_values else None
                } if self.cci_values else None,
                'roc': {
                    'value': list(self.roc_values)[-1] if self.roc_values else None,
                    'signal': self.get_roc_signal(list(self.roc_values)[-1]) if self.roc_values else None
                } if self.roc_values else None,
                'momentum': {
                    'value': list(self.momentum_values)[-1] if self.momentum_values else None,
                    'signal': self.get_momentum_signal(list(self.momentum_values)[-1]) if self.momentum_values else None
                } if self.momentum_values else None,
                'atr': {
                    'value': list(self.atr_values)[-1] if self.atr_values else None,
                    'signal': self.get_atr_signal(list(self.atr_values)[-1]) if self.atr_values else None
                } if self.atr_values else None,
                'fibonacci': {
                    'swing_high': list(self.fibonacci_values)[-1]['swing_high'] if self.fibonacci_values else None,
                    'swing_low': list(self.fibonacci_values)[-1]['swing_low'] if self.fibonacci_values else None,
                    'trend_direction': list(self.fibonacci_values)[-1]['trend_direction'] if self.fibonacci_values else None,
                    'levels': list(self.fibonacci_values)[-1]['levels'] if self.fibonacci_values else None,
                    'closest_level': list(self.fibonacci_values)[-1]['closest_level'] if self.fibonacci_values else None,
                    'signal': list(self.fibonacci_values)[-1]['signal'] if self.fibonacci_values else None
                } if self.fibonacci_values else None,
                'vwap': {
                    'value': list(self.vwap_values)[-1] if self.vwap_values else None,
                    'signal': self.get_vwap_signal(
                        list(self.vwap_values)[-1] if self.vwap_values else None,
                        list(self.prices)[-1] if self.prices else None
                    ) if self.vwap_values and self.prices else None
                } if self.vwap_values else None,
                'pivot_points': {
                    'pivot_point': list(self.pivot_points_values)[-1]['pivot_point'] if self.pivot_points_values else None,
                    'support_levels': {
                        's1': list(self.pivot_points_values)[-1].get('support_1') if self.pivot_points_values else None,
                        's2': list(self.pivot_points_values)[-1].get('support_2') if self.pivot_points_values else None,
                        's3': list(self.pivot_points_values)[-1].get('support_3') if self.pivot_points_values else None,
                        's4': list(self.pivot_points_values)[-1].get('support_4') if self.pivot_points_values else None,
                    },
                    'resistance_levels': {
                        'r1': list(self.pivot_points_values)[-1].get('resistance_1') if self.pivot_points_values else None,
                        'r2': list(self.pivot_points_values)[-1].get('resistance_2') if self.pivot_points_values else None,
                        'r3': list(self.pivot_points_values)[-1].get('resistance_3') if self.pivot_points_values else None,
                        'r4': list(self.pivot_points_values)[-1].get('resistance_4') if self.pivot_points_values else None,
                    },
                    'type': list(self.pivot_points_values)[-1].get('type') if self.pivot_points_values else None,
                    'signal': list(self.pivot_points_values)[-1].get('signal') if self.pivot_points_values else None
                } if self.pivot_points_values else None,
            },
            'configuration': self.config
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update indicator configuration and reinitialize if needed"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Check if we need to reinitialize data structures
        need_reinit = False
        for key in ['sma_periods', 'ema_periods', 'macd_slow', 'rsi_period', 'adx_period', 'ichimoku_span_b', 'stochastic_k_period', 'stochastic_d_period', 'cci_period', 'roc_period', 'momentum_period', 'atr_period', 'fibonacci_period', 'vwap_period']:
            if key in new_config and new_config[key] != old_config.get(key):
                need_reinit = True
                break
        
        if need_reinit:
            self.initialize_data_structures()
            logger.info("Configuration updated and data structures reinitialized")

class TradingDataClient:
    """WebSocket client for real-time technical indicators calculation"""
    
    def __init__(self, websocket_url: str = "ws://trading-api:8080/ws", config: Dict[str, Any] = None):
        self.websocket_url = websocket_url
        self.indicators = TechnicalIndicators(config)
        self.websocket = None
        self.running = False
        
    async def connect_and_stream(self):
        """Connect to WebSocket and stream technical indicators"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.websocket = await websockets.connect(self.websocket_url)
                logger.info(f"Connected to {self.websocket_url}")
                
                self.running = True
                async for message in self.websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        indicators = self.indicators.add_data_point(data)
                        
                        if indicators:
                            # Create comprehensive output
                            output = {
                                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                                'price_data': {
                                    'price': data.get('price', data.get('close')),
                                    'high': data.get('high'),
                                    'low': data.get('low'),
                                    'open': data.get('open'),
                                    'volume': data.get('volume')
                                },
                                'technical_indicators': indicators
                            }
                            
                            # Log the output
                            logger.info(f"Technical Indicators: {json.dumps(output, indent=2)}")
                            
                            # Here you could send to another API endpoint or store in database
                            await self.send_to_api(output)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
                break
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URI: {self.websocket_url}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying connection in 5 seconds... (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(5)
                else:
                    logger.error("Max retries reached. Stopping WebSocket client.")
                    break
            finally:
                if self.websocket:
                    await self.websocket.close()
    
    async def send_to_api(self, data: Dict[str, Any]):
        """Send technical indicators data to API endpoint"""
        # This method can be implemented to send data to your AI system
        # To do : Actual implementation yet to be done.
        logger.info(f"Technical indicators data ready for AI consumption: {len(data.get('technical_indicators', {}))} indicators")
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update indicator configuration"""
        self.indicators.update_configuration(new_config)
        logger.info("Configuration updated")
    
    def get_current_indicators(self) -> Dict[str, Any]:
        """Get current indicator values"""
        return self.indicators.get_all_indicators()
    
    def stop(self):
        """Stop the client"""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)

async def main():
    """Main function to run the technical indicators client"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration example
    config = {
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
    
    client = TradingDataClient(config=config)
    
    try:
        await client.connect_and_stream()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.stop()

if __name__ == "__main__":
    asyncio.run(main()) 