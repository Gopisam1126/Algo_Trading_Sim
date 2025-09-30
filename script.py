from collections import deque
from typing import Optional, Dict


def calculate_stochastic_oscillator(self) -> Optional[Dict[str, float]]:
    """Calculate Stochastic Oscillator (%K and %D)"""
    # if len(self.highs) < self.config['stochastic_k_period'] or len(self.lows) < self.config['stochastic_k_period']:
    # Validate data synchronization
    if not (len(self.highs) == len(self.lows) == len(self.prices)):
        raise ValueError("Data arrays (highs, lows, prices) must be synchronized")
    
    k_period = self.config['stochastic_k_period']
    d_period = self.config['stochastic_d_period']
    
    if len(self.prices) < k_period:
        return None
    
    # Initialize sliding window structures if not already done
    if not hasattr(self, '_stoch_min_deque'):
        # Sliding min/max structures
        self._stoch_min_deque = deque()
        self._stoch_max_deque = deque()
        self._stoch_window = deque(maxlen=k_period)
        
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
        start_idx = max(0, len(self.highs) - k_period)
        for i in range(start_idx, len(self.highs)):
            high_val = self.highs[i]
            low_val = self.lows[i]
            
            # Remove old values from window
            if len(self._stoch_window) == k_period:
                old_high, old_low = self._stoch_window[0]
                if self._stoch_min_deque and self._stoch_min_deque[0] == old_low:
                    self._stoch_min_deque.popleft()
                if self._stoch_max_deque and self._stoch_max_deque[0] == old_high:
                    self._stoch_max_deque.popleft()
            
            self._stoch_window.append((high_val, low_val))
            
            # Maintain min deque
            while self._stoch_min_deque and self._stoch_min_deque[-1] > low_val:
                self._stoch_min_deque.pop()

            self._stoch_min_deque.append(low_val)
            
            # Maintain max deque
            while self._stoch_max_deque and self._stoch_max_deque[-1] < high_val:
                self._stoch_max_deque.pop()

            self._stoch_max_deque.append(high_val)
    else:
        # Add only the latest data point
        high_val = self.highs[-1]
        low_val = self.lows[-1]
        
        # Remove old values from window
        if len(self._stoch_window) == k_period:
            old_high, old_low = self._stoch_window[0]
            if self._stoch_min_deque and self._stoch_min_deque[0] == old_low:
                self._stoch_min_deque.popleft()
            if self._stoch_max_deque and self._stoch_max_deque[0] == old_high:
                self._stoch_max_deque.popleft()
        
        self._stoch_window.append((high_val, low_val))
        
        # Maintain min deque
        while self._stoch_min_deque and self._stoch_min_deque[-1] > low_val:
            self._stoch_min_deque.pop()

        self._stoch_min_deque.append(low_val)
        
        # Maintain max deque
        while self._stoch_max_deque and self._stoch_max_deque[-1] < high_val:
            self._stoch_max_deque.pop()

        self._stoch_max_deque.append(high_val)
    
    if len(self._stoch_window) < k_period:
        return None
    
    # Get current closing price and period high/low
    current_price = self.prices[-1]
    highest_high = self._stoch_max_deque[0]
    lowest_low = self._stoch_min_deque[0]
    
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
    """Calculate Commodity Channel Index (CCI)"""
    # Validate input data alignment
    if not (len(self.highs) >= 1 and len(self.lows) >= 1 and len(self.prices) >= 1):
        return None
    
    if len(self.highs) != len(self.lows) or len(self.highs) != len(self.prices):
        raise ValueError("Input arrays must have equal length")
    
    # Calculate Typical Price (TP)
    typical_price = (self.highs[-1] + self.lows[-1] + self.prices[-1]) / 3
    
    # Use deque with maxlen for automatic memory management
    if not hasattr(self, '_tp_deque'):
        self._tp_deque = deque(maxlen=self.config['cci_period'])
    
    self._tp_deque.append(typical_price)
    self.typical_prices.append(typical_price)  # Keep if needed elsewhere
    
    # Need at least the period number of typical prices
    if len(self._tp_deque) < self.config['cci_period']:
        return None
    
    # Calculate SMA and Mean Deviation in a single pass
    period = self.config['cci_period']
    sma_tp = sum(self._tp_deque) / period
    
    # Calculate Mean Deviation
    mean_deviation = sum(abs(tp - sma_tp) for tp in self._tp_deque) / period
    
    # Calculate CCI with epsilon check for floating point comparison
    EPSILON = 1e-10
    if mean_deviation < EPSILON:
        cci = 0.0
    else:
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    # Manage memory for cci_values
    if not hasattr(self, '_max_cci_history'):
        self._max_cci_history = 1000  # Configurable limit
    
    self.cci_values.append(cci)
    if len(self.cci_values) > self._max_cci_history:
        self.cci_values.pop(0)  # Or use deque with maxlen
    
    return cci

def get_cci_signal(self, cci: float) -> str:
    """Get trading signal based on CCI value - Improved version"""
    EPSILON = 1e-10
    
    if cci > 100:
        return "OVERBOUGHT"
    elif cci < -100:
        return "OVERSOLD"
    elif cci > EPSILON:
        return "BULLISH"
    elif cci < -EPSILON:
        return "BEARISH"
    else:
        return "NEUTRAL"