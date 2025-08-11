#!/usr/bin/env python3
"""
Comprehensive Technical Indicators Calculator
All indicators are configurable and support both real-time and historical calculations
"""

import asyncio
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
import math

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
             self.config['momentum_period']]
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
        
        return indicators
    
    def calculate_moving_averages(self) -> Dict[str, Any]:
        """Calculate SMA and EMA for all configured periods"""
        if len(self.prices) < min(self.config['sma_periods'] + self.config['ema_periods']):
            return {}
        
        result = {}
        
        # Calculate SMA
        for period in self.config['sma_periods']:
            if len(self.prices) >= period:
                sma = sum(list(self.prices)[-period:]) / period
                self.sma_values[period].append(sma)
                result[f'sma_{period}'] = sma
        
        # Calculate EMA
        for period in self.config['ema_periods']:
            if len(self.prices) >= period:
                ema = self.calculate_ema(self.prices, period, 
                                       self.ema_values[period][-1] if self.ema_values[period] else None)
                self.ema_values[period].append(ema)
                result[f'ema_{period}'] = ema
        
        return result
    
    def calculate_ema(self, prices: deque, period: int, prev_ema: Optional[float] = None) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        if prev_ema is None:
            # First EMA calculation - use simple average
            return sum(list(prices)[-period:]) / period
        else:
            # Subsequent EMA calculations
            multiplier = 2 / (period + 1)
            return (prices[-1] * multiplier) + (prev_ema * (1 - multiplier))
    
    def calculate_macd(self) -> Optional[Dict[str, float]]:
        """Calculate MACD indicator"""
        if len(self.prices) < self.config['macd_slow']:
            return None
        
        # Calculate fast EMA
        self.fast_ema = self.calculate_ema(self.prices, self.config['macd_fast'], self.fast_ema)
        
        # Calculate slow EMA
        self.slow_ema = self.calculate_ema(self.prices, self.config['macd_slow'], self.slow_ema)
        
        if self.fast_ema is None or self.slow_ema is None:
            return None
        
        # Calculate MACD line
        self.macd_line = self.fast_ema - self.slow_ema
        
        # Calculate signal line (EMA of MACD line)
        if self.macd_line is not None:
            self.macd_values.append(self.macd_line)
            
            if len(self.macd_values) >= self.config['macd_signal']:
                if self.signal_line is None:
                    # First signal line calculation
                    recent_values = list(self.macd_values)[-self.config['macd_signal']:]
                    self.signal_line = sum(recent_values) / len(recent_values)
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
        """Calculate RSI indicator"""
        if len(self.prices) < 2:
            return None
        
        # Calculate price change
        price_change = self.prices[-1] - self.prices[-2]
        
        # Separate gains and losses
        gain = max(price_change, 0)
        loss = abs(min(price_change, 0))
        
        self.rsi_gains.append(gain)
        self.rsi_losses.append(loss)
        
        if len(self.rsi_gains) < self.config['rsi_period']:
            return None
        
        # Calculate RSI
        if self.avg_gain is None or self.avg_loss is None:
            # First RSI calculation
            self.avg_gain = sum(self.rsi_gains) / self.config['rsi_period']
            self.avg_loss = sum(self.rsi_losses) / self.config['rsi_period']
        else:
            # Subsequent RSI calculations (Wilder's method)
            self.avg_gain = ((self.avg_gain * (self.config['rsi_period'] - 1)) + gain) / self.config['rsi_period']
            self.avg_loss = ((self.avg_loss * (self.config['rsi_period'] - 1)) + loss) / self.config['rsi_period']
        
        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.rsi_values.append(rsi)
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
        """Calculate ADX indicator"""
        if len(self.highs) < 2 or len(self.lows) < 2:
            return None
        
        # Calculate True Range
        high = self.highs[-1]
        low = self.lows[-1]
        prev_close = self.prices[-2]
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self.tr_values.append(tr)
        
        # Calculate Directional Movement
        high_diff = high - self.highs[-2]
        low_diff = self.lows[-2] - low
        
        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        self.plus_dm.append(plus_dm)
        self.minus_dm.append(minus_dm)
        
        if len(self.tr_values) < self.config['adx_period']:
            return None
        
        # Calculate smoothed averages
        if self.plus_di is None or self.minus_di is None:
            # First calculation
            self.plus_di = (sum(self.plus_dm) / sum(self.tr_values)) * 100
            self.minus_di = (sum(self.minus_dm) / sum(self.tr_values)) * 100
        else:
            # Subsequent calculations
            tr_sum = sum(list(self.tr_values)[-self.config['adx_period']:])
            plus_dm_sum = sum(list(self.plus_dm)[-self.config['adx_period']:])
            minus_dm_sum = sum(list(self.minus_dm)[-self.config['adx_period']:])
            
            self.plus_di = (plus_dm_sum / tr_sum) * 100
            self.minus_di = (minus_dm_sum / tr_sum) * 100
        
        # Calculate ADX
        di_diff = abs(self.plus_di - self.minus_di)
        di_sum = self.plus_di + self.minus_di
        
        if di_sum == 0:
            dx = 0
        else:
            dx = (di_diff / di_sum) * 100
        
        if self.adx is None:
            self.adx = dx
        else:
            self.adx = ((self.adx * (self.config['adx_period'] - 1)) + dx) / self.config['adx_period']
        
        self.adx_values.append(self.adx)
        
        return {
            'adx': self.adx,
            'plus_di': self.plus_di,
            'minus_di': self.minus_di,
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
    
    def calculate_supertrend(self) -> Optional[Dict[str, Any]]:
        """Calculate SuperTrend indicator"""
        if len(self.highs) < self.config['supertrend_period'] or len(self.lows) < self.config['supertrend_period']:
            return None
        
        # Calculate ATR
        tr_values = []
        for i in range(1, min(self.config['supertrend_period'] + 1, len(self.highs))):
            high = self.highs[-i]
            low = self.lows[-i]
            prev_close = self.prices[-i-1] if len(self.prices) > i else self.prices[-i]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        
        atr = sum(tr_values) / len(tr_values)
        
        # Calculate SuperTrend
        if self.supertrend_upper is None or self.supertrend_lower is None:
            # Initialize
            self.supertrend_upper = self.highs[-1] - (atr * self.config['supertrend_multiplier'])
            self.supertrend_lower = self.lows[-1] + (atr * self.config['supertrend_multiplier'])
        else:
            # Update bands
            basic_upper = (self.highs[-1] + self.lows[-1]) / 2 + (atr * self.config['supertrend_multiplier'])
            basic_lower = (self.highs[-1] + self.lows[-1]) / 2 - (atr * self.config['supertrend_multiplier'])
            
            if basic_upper < self.supertrend_upper or self.prices[-2] > self.supertrend_upper:
                self.supertrend_upper = basic_upper
            else:
                self.supertrend_upper = self.supertrend_upper
            
            if basic_lower > self.supertrend_lower or self.prices[-2] < self.supertrend_lower:
                self.supertrend_lower = basic_lower
            else:
                self.supertrend_lower = self.supertrend_lower
        
        # Determine trend direction
        current_price = self.prices[-1]
        prev_price = self.prices[-2] if len(self.prices) > 1 else current_price
        
        if prev_price <= self.supertrend_upper and current_price > self.supertrend_upper:
            self.supertrend_direction = 1  # Uptrend
        elif prev_price >= self.supertrend_lower and current_price < self.supertrend_lower:
            self.supertrend_direction = -1  # Downtrend
        
        # Calculate SuperTrend value
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

        # Get the required periods
        k_period = self.config['stochastic_k_period']
        d_period = self.config['stochastic_d_period']

        # Get current price and the highest high and lowest low over the period
        current_price = self.prices[-1]
        period_highs = list(self.highs)[-k_period:]
        period_lows = list(self.lows)[-k_period:]

        highest_high = max(period_highs)
        lowest_low = min(period_lows)

        # Calculate %K
        if highest_high == lowest_low:
            self.stochastic_k = 50  # Avoid division by zero
        else:
            self.stochastic_k = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100

        # Store %K value for %D calculation
        self.stochastic_k_values.append(self.stochastic_k)

        # Calculate %D (SMA of %K values)
        if len(self.stochastic_k_values) >= d_period:
            recent_k_values = list(self.stochastic_k_values)[-d_period:]
            self.stochastic_d = sum(recent_k_values) / d_period

            # Store the complete stochastic data
            stochastic_data = {
                'k': self.stochastic_k,
                'd': self.stochastic_d,
                'signal': self.get_stochastic_signal(self.stochastic_k, self.stochastic_d)
            }

            self.stochastic_values.append(stochastic_data)
            return stochastic_data

        # Return only %K if we don't have enough data for %D yet
        return {
            'k': self.stochastic_k,
            'd': None,
            'signal': self.get_stochastic_signal(self.stochastic_k, None)
        }
    
    def get_stochastic_signal(self, k_value: float, d_value: Optional[float]) -> str:
        """Get trading signal based on Stochastic values"""
        if k_value > 80:
            return "OVERBOUGHT"
        elif k_value < 20:
            return "OVERSOLD"
        elif d_value is not None:
            if k_value > d_value and k_value < 50:
                return "BULLISH_CROSSOVER"
            elif k_value < d_value and k_value > 50:
                return "BEARISH_CROSSOVER"

        return "NEUTRAL"
    
    def calculate_cci(self) -> Optional[float]:
        """Calculate Commodity Channel Index (CCI)"""
        if len(self.highs) < 1 or len(self.lows) < 1 or len(self.prices) < 1:
            return None

        # Calculate Typical Price (TP)
        typical_price = (self.highs[-1] + self.lows[-1] + self.prices[-1]) / 3
        self.typical_prices.append(typical_price)

        # Need at least the period number of typical prices
        if len(self.typical_prices) < self.config['cci_period']:
            return None

        # Calculate Simple Moving Average of Typical Price
        recent_tp = list(self.typical_prices)[-self.config['cci_period']:]
        sma_tp = sum(recent_tp) / len(recent_tp)

        # Calculate Mean Deviation
        mean_deviation = sum(abs(tp - sma_tp) for tp in recent_tp) / len(recent_tp)

        # Calculate CCI
        if mean_deviation == 0:
            cci = 0
        else:
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        self.cci_values.append(cci)
        return cci

    def get_cci_signal(self, cci: float) -> str:
        """Get trading signal based on CCI value"""
        if cci > 100:
            return "OVERBOUGHT"
        elif cci < -100:
            return "OVERSOLD"
        elif cci > 0:
            return "BULLISH"
        elif cci < 0:
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
        
    def calculate_atr(self) -> Optional[float]:
        """Calculate Average True Range (ATR) indicator"""
        if len(self.highs) < 2 or len(self.lows) < 2 or len(self.prices) < 2:
            return None

        # Calculate True Range for current period
        high = self.highs[-1]
        low = self.lows[-1]
        prev_close = self.prices[-2]

        # True Range is the maximum of:
        # 1. Current High - Current Low
        # 2. Current High - Previous Close (absolute value)
        # 3. Current Low - Previous Close (absolute value)
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

    # def get_atr_signal(self, atr: float) -> str:
    #     """Get trading signal based on ATR value"""
    #     if len(self.atr_values) < 2:
    #         return "NEUTRAL"

    #     # Compare current ATR with previous ATR to determine volatility trend
    #     prev_atr = list(self.atr_values)[-2]
    #     atr_change = ((atr - prev_atr) / prev_atr) * 100

    #     if atr_change > 10:
    #         return "HIGH_VOLATILITY_INCREASING"
    #     elif atr_change > 5:
    #         return "VOLATILITY_INCREASING"
    #     elif atr_change < -10:
    #         return "VOLATILITY_DECREASING_SIGNIFICANTLY"
    #     elif atr_change < -5:
    #         return "VOLATILITY_DECREASING"
    #     else:
    #         # You can also add absolute ATR level signals if needed
    #         # This would require calculating average ATR over longer period
    #         return "STABLE_VOLATILITY"
        
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
            },
            'configuration': self.config
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update indicator configuration and reinitialize if needed"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Check if we need to reinitialize data structures
        need_reinit = False
        for key in ['sma_periods', 'ema_periods', 'macd_slow', 'rsi_period', 'adx_period', 'ichimoku_span_b', 'stochastic_k_period', 'stochastic_d_period', 'cci_period', 'roc_period', 'momentum_period', 'atr_period']:
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