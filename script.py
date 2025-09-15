from itertools import islice
from typing import Dict, Any, Optional
from collections import deque

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