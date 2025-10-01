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
    
def calculate_stochastic_oscillator(self) -> Optional[Dict[str, float]]:
    """Calculate Stochastic Oscillator (%K and %D)"""
    k_period = self.config['stochastic_k_period']
    d_period = self.config['stochastic_d_period']
    
    # Validate we have enough data
    if (len(self.highs) < k_period or 
        len(self.lows) < k_period or 
        len(self.prices) < k_period):
        return None
    
    # Validate all arrays have the same length
    if not (len(self.highs) == len(self.lows) == len(self.prices)):
        return None
    
    # Initialize sliding window structures if not already done
    if not hasattr(self, '_stoch_min_deque'):
        # Sliding min/max structures - store (value, index) tuples
        self._stoch_min_deque = deque()
        self._stoch_max_deque = deque()
        
        # Rolling sum structures for %D
        self._stoch_k_rolling = deque(maxlen=d_period)
        self._stoch_k_sum = 0.0
        
        # Previous values for crossover detection
        self._prev_k = None
        self._prev_d = None
        
        # Fixed-size deques to prevent unbounded growth
        self.stochastic_k_values = deque(maxlen=d_period)
        self.stochastic_values = deque(maxlen=100)
        
        # Initialize with the first k_period data points
        # Process the first window
        for i in range(k_period):
            idx = i
            high_val = self.highs[i]
            low_val = self.lows[i]
            
            # Maintain min deque (monotonic increasing)
            while self._stoch_min_deque and self._stoch_min_deque[-1][0] >= low_val:
                self._stoch_min_deque.pop()
            self._stoch_min_deque.append((low_val, idx))
            
            # Maintain max deque (monotonic decreasing)
            while self._stoch_max_deque and self._stoch_max_deque[-1][0] <= high_val:
                self._stoch_max_deque.pop()
            self._stoch_max_deque.append((high_val, idx))
        
        # Now process any remaining historical points
        for i in range(k_period, len(self.highs)):
            idx = i
            high_val = self.highs[i]
            low_val = self.lows[i]
            
            # Maintain min deque (monotonic increasing)
            while self._stoch_min_deque and self._stoch_min_deque[-1][0] >= low_val:
                self._stoch_min_deque.pop()
            self._stoch_min_deque.append((low_val, idx))
            
            # Maintain max deque (monotonic decreasing)
            while self._stoch_max_deque and self._stoch_max_deque[-1][0] <= high_val:
                self._stoch_max_deque.pop()
            self._stoch_max_deque.append((high_val, idx))
            
            # Remove elements outside the current window
            window_start = idx - k_period + 1
            while self._stoch_min_deque and self._stoch_min_deque[0][1] < window_start:
                self._stoch_min_deque.popleft()
            while self._stoch_max_deque and self._stoch_max_deque[0][1] < window_start:
                self._stoch_max_deque.popleft()
            
            # Calculate %K for this historical point
            current_price = self.prices[i]
            highest_high = self._stoch_max_deque[0][0]
            lowest_low = self._stoch_min_deque[0][0]
            
            if highest_high != lowest_low:
                k_val = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
                k_val = max(0.0, min(100.0, k_val))
                
                self.stochastic_k_values.append(k_val)
                
                # Update rolling sum for %D
                if len(self._stoch_k_rolling) == d_period:
                    self._stoch_k_sum -= self._stoch_k_rolling[0]
                self._stoch_k_rolling.append(k_val)
                self._stoch_k_sum += k_val
                
                # Calculate %D if we have enough %K values
                if len(self._stoch_k_rolling) == d_period:
                    d_val = self._stoch_k_sum / d_period
                    self._prev_k = k_val
                    self._prev_d = d_val
        
        # Mark as initialized - we've processed all historical data
        self._stoch_initialized = True
    else:
        # Add only the latest data point
        idx = len(self.highs) - 1
        high_val = self.highs[-1]
        low_val = self.lows[-1]
        
        # Maintain min deque (monotonic increasing)
        while self._stoch_min_deque and self._stoch_min_deque[-1][0] >= low_val:
            self._stoch_min_deque.pop()
        self._stoch_min_deque.append((low_val, idx))
        
        # Maintain max deque (monotonic decreasing)
        while self._stoch_max_deque and self._stoch_max_deque[-1][0] <= high_val:
            self._stoch_max_deque.pop()
        self._stoch_max_deque.append((high_val, idx))
        
        # Remove elements outside the current window
        window_start = idx - k_period + 1
        while self._stoch_min_deque and self._stoch_min_deque[0][1] < window_start:
            self._stoch_min_deque.popleft()
        while self._stoch_max_deque and self._stoch_max_deque[0][1] < window_start:
            self._stoch_max_deque.popleft()
    
    # Get current closing price and period high/low from deques
    current_price = self.prices[-1]
    highest_high = self._stoch_max_deque[0][0]
    lowest_low = self._stoch_min_deque[0][0]
    
    # Calculate %K
    if highest_high == lowest_low:
        # Flat price - return previous values or None
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

class FibonacciCalculator:
    def __init__(self, config):
        self.config = config
        self.highs = deque(maxlen=config.get('fibonacci_period', 100))
        self.lows = deque(maxlen=config.get('fibonacci_period', 100))
        self.prices = deque(maxlen=config.get('fibonacci_period', 100))
        self.fibonacci_values = deque(maxlen=100)
        
        # Cache variables
        self.fibonacci_swing_high = None
        self.fibonacci_swing_low = None
        self.fibonacci_levels_cache = None
        self.fibonacci_trend_direction = None
        self._cached_swing_points = None
        
        # Pre-convert fibonacci levels to numeric for faster access
        self.fib_levels_numeric = np.array(config.get('fibonacci_levels', [0.236, 0.382, 0.5, 0.618, 0.786]))

    def calculate_fibonacci_retracement(self) -> Optional[Dict[str, Any]]:
        """Calculate Fibonacci Retracement levels"""
        period = self.config['fibonacci_period']
        
        # Input validation
        if len(self.highs) < period or len(self.lows) < period or len(self.prices) < period:
            return None
        
        if period <= 0:
            return None

        # Convert to numpy arrays for faster calculations
        recent_highs = np.array(self.highs)
        recent_lows = np.array(self.lows)
        recent_prices = np.array(self.prices)

        # Validate data
        if len(recent_highs) == 0 or len(recent_lows) == 0 or len(recent_prices) == 0:
            return None
        
        if np.any(np.isnan(recent_highs)) or np.any(np.isnan(recent_lows)) or np.any(np.isnan(recent_prices)):
            return None

        # Find swing high and swing low using proper swing detection
        swing_high_idx, swing_low_idx = self._detect_swing_points(recent_highs, recent_lows)
        swing_high = recent_highs[swing_high_idx]
        swing_low = recent_lows[swing_low_idx]

        # Validate price range
        price_range = swing_high - swing_low
        if price_range <= 0:
            return None

        # Get current price and validate
        current_price = float(self.prices[-1])
        if current_price <= 0 or np.isnan(current_price):
            return None

        # Determine trend direction using proper momentum analysis
        trend_direction = self._detect_trend_direction(recent_prices, swing_high_idx, swing_low_idx)

        # Check if swing points changed before recalculating levels
        swing_points_changed = (self._cached_swing_points != (swing_high, swing_low))
        
        if swing_points_changed or self.fibonacci_levels_cache is None:
            # Calculate Fibonacci levels (always as retracements from swing points)
            fibonacci_levels = {}
            
            if swing_high_idx > swing_low_idx:
                # Uptrend: measure retracement from high down to low
                for level in self.fib_levels_numeric:
                    retracement_price = swing_high - (price_range * level)
                    fibonacci_levels[level] = retracement_price
            else:
                # Downtrend: measure retracement from low up to high
                for level in self.fib_levels_numeric:
                    retracement_price = swing_low + (price_range * level)
                    fibonacci_levels[level] = retracement_price
            
            # Cache the levels and swing points
            self.fibonacci_levels_cache = fibonacci_levels
            self._cached_swing_points = (swing_high, swing_low)
        else:
            # Use cached levels
            fibonacci_levels = self.fibonacci_levels_cache

        # Store current values
        self.fibonacci_swing_high = swing_high
        self.fibonacci_swing_low = swing_low
        self.fibonacci_trend_direction = trend_direction

        # Find closest level and generate signal in one pass
        closest_level, signal = self._find_closest_and_signal(
            current_price, fibonacci_levels, trend_direction, swing_high_idx > swing_low_idx
        )

        fibonacci_data = {
            'swing_high': float(swing_high),
            'swing_low': float(swing_low),
            'price_range': float(price_range),
            'trend_direction': trend_direction,
            'levels': {f'fib_{k:.3f}': float(v) for k, v in fibonacci_levels.items()},
            'closest_level': closest_level,
            'signal': signal,
            'current_price': current_price
        }

        self.fibonacci_values.append(fibonacci_data)
        return fibonacci_data

    def _detect_swing_points(self, highs: np.ndarray, lows: np.ndarray, window: int = 5) -> tuple:
        """Detect swing high and swing low using local extrema"""
        # Validate window size
        window = min(window, len(highs) // 4)
        window = max(1, window)
        
        swing_high_idx = 0
        swing_low_idx = 0
        max_score = -np.inf
        min_score = np.inf
        
        # Find swing high: local maximum with lower highs around it
        for i in range(window, len(highs) - window):
            left_slice = highs[max(0, i-window):i]
            right_slice = highs[i+1:min(len(highs), i+window+1)]
            
            if len(left_slice) > 0 and len(right_slice) > 0:
                if highs[i] >= np.max(left_slice) and highs[i] >= np.max(right_slice):
                    # Score based on prominence
                    score = highs[i] + (highs[i] - np.mean(left_slice)) + (highs[i] - np.mean(right_slice))
                    if score > max_score:
                        max_score = score
                        swing_high_idx = i
        
        # Find swing low: local minimum with higher lows around it
        for i in range(window, len(lows) - window):
            left_slice = lows[max(0, i-window):i]
            right_slice = lows[i+1:min(len(lows), i+window+1)]
            
            if len(left_slice) > 0 and len(right_slice) > 0:
                if lows[i] <= np.min(left_slice) and lows[i] <= np.min(right_slice):
                    # Score based on prominence
                    score = -lows[i] - (np.mean(left_slice) - lows[i]) - (np.mean(right_slice) - lows[i])
                    if score < min_score:
                        min_score = score
                        swing_low_idx = i
        
        # Fallback to simple max/min if no swing points detected
        if max_score == -np.inf:
            swing_high_idx = np.argmax(highs)
        if min_score == np.inf:
            swing_low_idx = np.argmin(lows)
        
        return swing_high_idx, swing_low_idx

    def _detect_trend_direction(self, prices: np.ndarray, swing_high_idx: int, swing_low_idx: int) -> str:
        """Detect trend direction based on swing point positioning and price momentum"""
        # Primary: Check which swing came more recently
        if swing_high_idx > swing_low_idx:
            primary_trend = 'up'
        else:
            primary_trend = 'down'
        
        # Secondary: Confirm with recent price momentum
        if len(prices) >= 10:
            recent_momentum = prices[-1] - prices[-10]
            momentum_trend = 'up' if recent_momentum > 0 else 'down'
            
            # Confirm trend if both agree
            if primary_trend == momentum_trend:
                return primary_trend
        
        return primary_trend

    def _find_closest_and_signal(self, current_price: float, fibonacci_levels: Dict[float, float], 
                                  trend_direction: str, is_uptrend_swing: bool) -> tuple:
        """Find closest Fibonacci level and generate signal in single pass"""
        if not fibonacci_levels:
            return None, "NEUTRAL"

        # Find closest level using numpy for speed
        levels_array = np.array(list(fibonacci_levels.keys()))
        prices_array = np.array(list(fibonacci_levels.values()))
        
        distances = np.abs(prices_array - current_price)
        closest_idx = np.argmin(distances)
        
        closest_distance = distances[closest_idx]
        closest_level_key = levels_array[closest_idx]
        closest_level_price = prices_array[closest_idx]
        
        distance_percentage = (closest_distance / current_price) * 100 if current_price > 0 else float('inf')
        
        closest_level = {
            'level_name': f'fib_{closest_level_key:.3f}',
            'level_price': float(closest_level_price),
            'distance': float(closest_distance),
            'distance_percentage': float(distance_percentage)
        }

        # Generate signal based on numeric level comparison
        signal = self._generate_signal(
            current_price, closest_level_key, closest_level_price, 
            distance_percentage, fibonacci_levels, is_uptrend_swing
        )

        return closest_level, signal

    def _generate_signal(self, current_price: float, closest_level: float, closest_price: float,
                        distance_percentage: float, fibonacci_levels: Dict[float, float], 
                        is_uptrend_swing: bool) -> str:
        """Generate trading signal based on Fibonacci levels using numeric comparisons"""
        proximity_threshold = 0.5  # 0.5% distance threshold
        medium_threshold = 2.0

        # Strong signal when price is very close to key Fibonacci levels
        if distance_percentage < proximity_threshold:
            # 61.8% (Golden Ratio) and 38.2% are most significant
            if np.isclose(closest_level, 0.618, atol=0.001) or np.isclose(closest_level, 0.382, atol=0.001):
                if is_uptrend_swing:
                    return "STRONG_SUPPORT_BOUNCE" if current_price >= closest_price else "STRONG_RESISTANCE_REJECTION"
                else:
                    return "STRONG_RESISTANCE_REJECTION" if current_price >= closest_price else "STRONG_SUPPORT_BOUNCE"

            # 50% level (psychological level)
            elif np.isclose(closest_level, 0.500, atol=0.001):
                if is_uptrend_swing:
                    return "SUPPORT_BOUNCE" if current_price >= closest_price else "RESISTANCE_REJECTION"
                else:
                    return "RESISTANCE_REJECTION" if current_price >= closest_price else "SUPPORT_BOUNCE"

            # Other levels
            else:
                if is_uptrend_swing:
                    return "WEAK_SUPPORT" if current_price >= closest_price else "WEAK_RESISTANCE"
                else:
                    return "WEAK_RESISTANCE" if current_price >= closest_price else "WEAK_SUPPORT"

        # Medium distance signals
        elif distance_percentage < medium_threshold:
            if np.isclose(closest_level, 0.618, atol=0.001) or np.isclose(closest_level, 0.382, atol=0.001):
                if is_uptrend_swing:
                    return "APPROACHING_SUPPORT" if current_price > closest_price else "APPROACHING_RESISTANCE"
                else:
                    return "APPROACHING_RESISTANCE" if current_price > closest_price else "APPROACHING_SUPPORT"

        # Trend continuation signals based on position relative to key levels
        fib_618 = fibonacci_levels.get(0.618)
        fib_382 = fibonacci_levels.get(0.382)

        if fib_618 is not None and fib_382 is not None:
            if is_uptrend_swing:
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

    def find_closest_fibonacci_level(self, current_price: float, fibonacci_levels: Dict[str, float]) -> Dict[str, Any]:
        """Find the closest Fibonacci level to current price"""
        if not fibonacci_levels or current_price <= 0:
            return None

        # Convert string keys back to numeric for calculation
        numeric_levels = {}
        for k, v in fibonacci_levels.items():
            try:
                # Extract numeric value from 'fib_X.XXX' format
                numeric_key = float(k.replace('fib_', ''))
                numeric_levels[numeric_key] = v
            except (ValueError, AttributeError):
                continue
        
        if not numeric_levels:
            return None

        levels_array = np.array(list(numeric_levels.values()))
        distances = np.abs(levels_array - current_price)
        closest_idx = np.argmin(distances)
        
        closest_distance = distances[closest_idx]
        closest_level_price = levels_array[closest_idx]
        
        # Find the key for this price
        closest_level_name = None
        for k, v in fibonacci_levels.items():
            if np.isclose(v, closest_level_price, atol=1e-6):
                closest_level_name = k
                break

        return {
            'level_name': closest_level_name,
            'level_price': float(closest_level_price),
            'distance': float(closest_distance),
            'distance_percentage': (closest_distance / current_price) * 100
        }

    def get_fibonacci_signal(self, current_price: float, fibonacci_levels: Dict[str, float], trend_direction: str) -> str:
        """Get trading signal based on Fibonacci levels"""
        if not fibonacci_levels or current_price <= 0:
            return "NEUTRAL"

        # Convert string keys to numeric
        numeric_levels = {}
        for k, v in fibonacci_levels.items():
            try:
                numeric_key = float(k.replace('fib_', ''))
                numeric_levels[numeric_key] = v
            except (ValueError, AttributeError):
                continue

        if not numeric_levels:
            return "NEUTRAL"

        # Get the closest level info (reuse existing method)
        closest_level = self.find_closest_fibonacci_level(current_price, fibonacci_levels)

        if not closest_level:
            return "NEUTRAL"

        distance_percentage = closest_level['distance_percentage']
        level_name = closest_level['level_name']
        level_price = closest_level['level_price']

        # Extract numeric level
        try:
            numeric_level = float(level_name.replace('fib_', ''))
        except (ValueError, AttributeError):
            return "NEUTRAL"

        # Determine if uptrend swing based on trend direction and price position
        is_uptrend_swing = trend_direction == 'up'

        return self._generate_signal(
            current_price, numeric_level, level_price,
            distance_percentage, numeric_levels, is_uptrend_swing
        )