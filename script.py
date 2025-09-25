from collections import deque
from typing import Optional

def calculate_parabolic_sar(self) -> Optional[float]:
    """Calculate Parabolic SAR indicator with improved accuracy and performance"""
    if len(self.highs) < 2 or len(self.lows) < 2:
        return None
    
    # Cache frequently accessed values for performance
    current_high = self.highs[-1]
    current_low = self.lows[-1]
    prev_high = self.highs[-2] if len(self.highs) >= 2 else current_high
    prev_low = self.lows[-2] if len(self.lows) >= 2 else current_low
    
    # Initialize tracking structures for bounded memory usage
    if not hasattr(self, '_psar_initialized'):
        self._psar_initialized = True
        
        # Use deque for bounded memory (keep last N values if needed)
        if hasattr(self, 'psar_values'):
            # Convert existing list to deque if it exists, keep reasonable history
            max_history = getattr(self.config, 'max_psar_history', 1000)
            if isinstance(self.psar_values, list):
                recent_values = self.psar_values[-max_history:] if self.psar_values else []
                self.psar_values = deque(recent_values, maxlen=max_history)
            elif not isinstance(self.psar_values, deque):
                self.psar_values = deque(maxlen=max_history)
        else:
            self.psar_values = deque(maxlen=getattr(self.config, 'max_psar_history', 1000))
    
    # Get configuration values (cache for performance)
    psar_accel = self.config['psar_acceleration']
    psar_max = self.config['psar_maximum']
    
    # Initial SAR calculation
    if self.psar is None:
        # Better initial trend determination
        if not hasattr(self, 'psar_long'):
            # Determine initial trend based on price action
            self.psar_long = current_high > prev_high and current_low >= prev_low
        
        # Initialize SAR and EP
        if self.psar_long:
            self.psar = current_low
            self.psar_ep = current_high
        else:
            self.psar = current_high  
            self.psar_ep = current_low
            
        # Initialize acceleration factor
        self.psar_af = psar_accel
        return self.psar
    
    # Store previous SAR for bounds checking
    prev_psar = self.psar
    
    # Determine trend switch conditions first (early evaluation)
    trend_switch = False
    
    if self.psar_long:
        # Check for trend switch to short
        if current_low <= self.psar:
            trend_switch = True
            new_trend_long = False
            new_psar = self.psar_ep  # SAR becomes the previous EP
            new_ep = current_low
            new_af = psar_accel
    else:
        # Check for trend switch to long  
        if current_high >= self.psar:
            trend_switch = True
            new_trend_long = True
            new_psar = self.psar_ep  # SAR becomes the previous EP
            new_ep = current_high
            new_af = psar_accel
    
    if trend_switch:
        # Apply trend switch
        self.psar_long = new_trend_long
        self.psar = new_psar
        self.psar_ep = new_ep
        self.psar_af = new_af
    else:
        # Continue current trend
        if self.psar_long:
            # Update EP and AF for long trend
            if current_high > self.psar_ep:
                self.psar_ep = current_high
                self.psar_af = min(self.psar_af + psar_accel, psar_max)
            
            # Calculate new SAR
            self.psar = prev_psar + self.psar_af * (self.psar_ep - prev_psar)
            
            # Apply SAR constraints for long trend
            # SAR must not exceed the low of current or previous period
            max_constraint = min(current_low, prev_low)
            # Also check 2 periods back if available
            if len(self.lows) > 2:
                max_constraint = min(max_constraint, self.lows[-3])
                
            self.psar = min(self.psar, max_constraint)
            
        else:
            # Update EP and AF for short trend
            if current_low < self.psar_ep:
                self.psar_ep = current_low
                self.psar_af = min(self.psar_af + psar_accel, psar_max)
            
            # Calculate new SAR
            self.psar = prev_psar + self.psar_af * (self.psar_ep - prev_psar)
            
            # Apply SAR constraints for short trend
            # SAR must not be below the high of current or previous period
            min_constraint = max(current_high, prev_high)
            # Also check 2 periods back if available
            if len(self.highs) > 2:
                min_constraint = max(min_constraint, self.highs[-3])
                
            self.psar = max(self.psar, min_constraint)
    
    # Add to values collection (bounded memory)
    self.psar_values.append(self.psar)
    
    return self.psar