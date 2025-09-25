import math
from collections import deque
from typing import Optional, Dict

class ADXCalculator:
    def __init__(self, config):
        """Initialize ADX calculator with proper data structures"""
        self.config = config
        self.period = config.get('adx_period', 14)
        
        # Use bounded deques for memory safety (keep 3x period for calculations)
        max_size = self.period * 3
        
        # Price data - bounded deques
        self.highs = deque(maxlen=max_size)
        self.lows = deque(maxlen=max_size)  
        self.prices = deque(maxlen=max_size)  # closes
        
        # Intermediate calculations - bounded deques
        self.tr_values = deque(maxlen=max_size)
        self.plus_dm = deque(maxlen=max_size)
        self.minus_dm = deque(maxlen=max_size)
        
        # Rolling sums for efficiency (avoid recomputing)
        self.tr_sum = 0.0
        self.plus_dm_sum = 0.0
        self.minus_dm_sum = 0.0
        
        # ADX state
        self.plus_di = 0.0
        self.minus_di = 0.0
        self.adx = None
        self.adx_values = deque(maxlen=max_size)
        self.dx_values = deque(maxlen=self.period)  # For ADX smoothing initialization
        
        # Track initialization state
        self.is_initialized = False
        self.tick_count = 0
        
        # Constants
        self.MIN_VALUE = 1e-10  # Prevent division by zero
        self.RESYNC_INTERVAL = self.period * 20  # Resync every 20*period ticks
    
    def _is_valid_tick(self, high: float, low: float, close: float) -> bool:
        """Validate tick data for invalid values"""
        values = [high, low, close]
        
        # Check for finite values
        if not all(math.isfinite(val) for val in values):
            return False
            
        # Check for reasonable price relationships
        if not (low <= close <= high and low <= high):
            return False
            
        # Check for zero or negative prices
        if any(val <= 0 for val in values):
            return False
            
        # Check for extreme values (optional - adjust thresholds as needed)
        if any(abs(val) > 1e6 for val in values):
            return False
            
        return True
    
    def _update_rolling_sums(self, new_tr: float, new_plus_dm: float, new_minus_dm: float):
        """Efficiently update rolling sums without recomputing"""
        
        # Add new values
        self.tr_sum += new_tr
        self.plus_dm_sum += new_plus_dm
        self.minus_dm_sum += new_minus_dm
        
        # Remove old values if we exceed the period
        if len(self.tr_values) >= self.period:
            # Remove the oldest values from sums
            old_tr = self.tr_values[-(self.period)]
            old_plus_dm = self.plus_dm[-(self.period)]
            old_minus_dm = self.minus_dm[-(self.period)]
            
            self.tr_sum -= old_tr
            self.plus_dm_sum -= old_plus_dm  
            self.minus_dm_sum -= old_minus_dm
        
        # Prevent negative sums due to floating point errors
        self.tr_sum = max(0.0, self.tr_sum)
        self.plus_dm_sum = max(0.0, self.plus_dm_sum)
        self.minus_dm_sum = max(0.0, self.minus_dm_sum)
    
    def _calculate_initial_adx(self) -> float:
        """Calculate initial ADX using simple moving average of DX values"""
        if len(self.dx_values) < self.period:
            return 0.0
        
        return sum(self.dx_values) / len(self.dx_values)
    
    def _should_resync_adx(self) -> bool:
        """Determine if ADX needs resyncing to prevent drift"""
        return (self.tick_count % self.RESYNC_INTERVAL == 0 and 
                len(self.dx_values) >= self.period)
    
    def add_tick(self, high: float, low: float, close: float):
        """Add a new price tick to the calculator"""
        
        # Clean invalid ticks
        if not self._is_valid_tick(high, low, close):
            return  # Skip invalid tick
        
        self.highs.append(high)
        self.lows.append(low) 
        self.prices.append(close)
        self.tick_count += 1
    
    def calculate_adx(self) -> Optional[Dict[str, float]]:
        """Calculate ADX indicator with all issues addressed"""
        
        # Need at least 2 bars for calculation
        if len(self.prices) < 2:
            return None
        
        # Get current and previous values
        high = self.highs[-1]
        low = self.lows[-1]
        close = self.prices[-1]
        prev_close = self.prices[-2]
        
        # Calculate True Range
        tr = max(
            high - low,
            abs(high - prev_close), 
            abs(low - prev_close)
        )
        
        # Calculate Directional Movement
        if len(self.highs) >= 2:
            high_diff = high - self.highs[-2]
            low_diff = self.lows[-2] - low
            
            plus_dm = high_diff if (high_diff > low_diff and high_diff > 0) else 0.0
            minus_dm = low_diff if (low_diff > high_diff and low_diff > 0) else 0.0
        else:
            plus_dm = minus_dm = 0.0
        
        # Store values in bounded deques
        self.tr_values.append(tr)
        self.plus_dm.append(plus_dm)
        self.minus_dm.append(minus_dm)
        
        # Update rolling sums efficiently
        self._update_rolling_sums(tr, plus_dm, minus_dm)
        
        # Need enough data for calculations
        if len(self.tr_values) < self.period:
            return None
        
        # Calculate Directional Indicators
        if self.tr_sum < self.MIN_VALUE:
            return None  # Avoid division by zero
        
        self.plus_di = (self.plus_dm_sum / self.tr_sum) * 100
        self.minus_di = (self.minus_dm_sum / self.tr_sum) * 100
        
        # Calculate Directional Index (DX)
        di_sum = self.plus_di + self.minus_di
        if di_sum < self.MIN_VALUE:
            dx = 0.0
        else:
            di_diff = abs(self.plus_di - self.minus_di)
            dx = (di_diff / di_sum) * 100
        
        self.dx_values.append(dx)
        
        # Calculate ADX with proper initialization and drift prevention
        if self.adx is None or not self.is_initialized:
            # Initial ADX calculation
            if len(self.dx_values) >= self.period:
                self.adx = self._calculate_initial_adx()
                self.is_initialized = True
            else:
                self.adx = dx  # Use current DX until we have enough data
        else:
            # Check if we need to resync to prevent long-term drift
            if self._should_resync_adx():
                self.adx = self._calculate_initial_adx()
            else:
                # Wilder's smoothing: ADX = ((ADX * (n-1)) + DX) / n
                self.adx = ((self.adx * (self.period - 1)) + dx) / self.period
        
        self.adx_values.append(self.adx)
        
        return {
            'adx': self.adx,
            'plus_di': self.plus_di,
            'minus_di': self.minus_di,
            'dx': dx,
            'tr': tr,
            'plus_dm': plus_dm,
            'minus_dm': minus_dm
        }
    
    def get_state_info(self) -> Dict:
        """Get current state information for debugging"""
        return {
            'tick_count': self.tick_count,
            'is_initialized': self.is_initialized,
            'data_length': len(self.tr_values),
            'required_length': self.period,
            'tr_sum': self.tr_sum,
            'plus_dm_sum': self.plus_dm_sum,
            'minus_dm_sum': self.minus_dm_sum,
            'current_adx': self.adx,
            'memory_usage': {
                'highs': len(self.highs),
                'lows': len(self.lows),
                'prices': len(self.prices),
                'tr_values': len(self.tr_values),
                'max_allowed': self.period * 3
            }
        }

# Example usage:
"""
# Initialize
config = {'adx_period': 14}
adx_calc = ADXCalculator(config)

# Process ticks
for high, low, close in price_data:
    adx_calc.add_tick(high, low, close)
    result = adx_calc.calculate_adx()
    if result:
        print(f"ADX: {result['adx']:.2f}, +DI: {result['plus_di']:.2f}, -DI: {result['minus_di']:.2f}")

# Check state
state = adx_calc.get_state_info()
print(f"Calculator state: {state}")
"""

def calculate_parabolic_sar(self) -> Optional[float]:
    """Calculate Parabolic SAR indicator with improved robustness"""
    
    # Thread safety protection
    if hasattr(self, '_psar_lock'):
        with self._psar_lock:
            return self._calculate_parabolic_sar_internal()
    else:
        # Initialize lock if not present
        self._psar_lock = threading.RLock()
        with self._psar_lock:
            return self._calculate_parabolic_sar_internal()

def _calculate_parabolic_sar_internal(self) -> Optional[float]:
    """Internal SAR calculation with all safety checks"""
    
    # Require at least 3 bars for proper bounding logic
    if len(self.highs) < 3 or len(self.lows) < 3:
        return None
    
    # Get current bar data with NaN/infinite checks
    high = self.highs[-1]
    low = self.lows[-1]
    
    # Validate current bar data
    if not (math.isfinite(high) and math.isfinite(low)) or high < low:
        return None
    
    # Get closes for proper SAR computation (assuming self.closes exists)
    if hasattr(self, 'closes') and len(self.closes) >= 3:
        current_close = self.closes[-1]
        prev_close = self.closes[-2]
        if not (math.isfinite(current_close) and math.isfinite(prev_close)):
            return None
    else:
        # Fallback to midpoint if closes not available
        current_close = (high + low) / 2
        prev_close = (self.highs[-2] + self.lows[-2]) / 2
    
    if self.psar is None:
        # Improved initialization: use previous bar's data for more stable start
        prev_high = self.highs[-2]
        prev_low = self.lows[-2]
        
        # Validate previous bar data
        if not (math.isfinite(prev_high) and math.isfinite(prev_low)):
            return None
        
        # Determine initial trend based on price momentum
        if current_close > prev_close:
            self.psar_long = True
            self.psar = prev_low  # Start SAR below previous low
            self.psar_ep = high   # Extreme point is current high
        else:
            self.psar_long = False
            self.psar = prev_high  # Start SAR above previous high
            self.psar_ep = low     # Extreme point is current low
        
        # Initialize acceleration factor
        self.psar_af = self.config['psar_acceleration']
        
        # Validate initial values
        if not math.isfinite(self.psar):
            return None
            
        return self.psar
    
    # Validate existing SAR state
    if not (math.isfinite(self.psar) and math.isfinite(self.psar_ep) and math.isfinite(self.psar_af)):
        # Reset if corrupted state detected
        self.psar = None
        return self._calculate_parabolic_sar_internal()
    
    # Update SAR based on current trend
    if self.psar_long:
        # Long position
        if low <= self.psar:
            # Switch to short trend
            self.psar_long = False
            self.psar = self.psar_ep
            self.psar_ep = low
            self.psar_af = self.config['psar_acceleration']
        else:
            # Continue long trend
            if high > self.psar_ep:
                self.psar_ep = high
                self.psar_af = min(self.psar_af + self.config['psar_acceleration'], 
                                 self.config['psar_maximum'])
            
            # Calculate new SAR
            new_psar = self.psar + self.psar_af * (self.psar_ep - self.psar)
            
            # Apply 2-bar rule: SAR cannot be above the low of current or previous bar
            # With proper 3-bar requirement, we can safely access [-2] and [-3]
            max_bound = max(self.lows[-2], self.lows[-3])
            self.psar = min(new_psar, max_bound)
    else:
        # Short position
        if high >= self.psar:
            # Switch to long trend
            self.psar_long = True
            self.psar = self.psar_ep
            self.psar_ep = high
            self.psar_af = self.config['psar_acceleration']
        else:
            # Continue short trend
            if low < self.psar_ep:
                self.psar_ep = low
                self.psar_af = min(self.psar_af + self.config['psar_acceleration'], 
                                 self.config['psar_maximum'])
            
            # Calculate new SAR
            new_psar = self.psar + self.psar_af * (self.psar_ep - self.psar)
            
            # Apply 2-bar rule: SAR cannot be below the high of current or previous bar
            min_bound = min(self.highs[-2], self.highs[-3])
            self.psar = max(new_psar, min_bound)
    
    # Final validation of computed SAR
    if not math.isfinite(self.psar):
        return None
    
    # Floating-point drift mitigation: round to reasonable precision
    self.psar = round(self.psar, 8)
    
    # Store value if psar_values list exists
    if hasattr(self, 'psar_values'):
        self.psar_values.append(self.psar)
    
    return self.psar



