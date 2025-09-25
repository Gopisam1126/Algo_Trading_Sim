from collections import deque
from typing import Optional, Dict

def calculate_adx(self) -> Optional[Dict[str, float]]:
    """Calculate ADX indicator with optimal performance"""
    # removed list conversions for better performance
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
    
    # Update rolling sums efficiently
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