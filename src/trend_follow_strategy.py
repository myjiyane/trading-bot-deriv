"""
Trend-following strategy for momentum-based trading.

Implements a simple momentum system that enters long positions when short MA
crosses above long MA and exits when the short MA crosses back below.
"""

import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculate simple moving average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def get_ma_crossover(prices: List[float], short_period: int = 10, long_period: int = 50) -> Optional[str]:
    """
    Detect moving average crossover.
    
    Returns:
        'golden_cross' if short MA just crossed above long MA
        'death_cross' if short MA just crossed below long MA
        None if no crossover detected
    """
    if len(prices) < long_period + 1:
        return None
    
    # Current values
    current_short = calculate_sma(prices, short_period)
    current_long = calculate_sma(prices, long_period)
    
    # Previous values (1 period ago)
    prev_short = calculate_sma(prices[:-1], short_period) if len(prices[:-1]) >= short_period else None
    prev_long = calculate_sma(prices[:-1], long_period) if len(prices[:-1]) >= long_period else None
    
    if current_short is None or current_long is None or prev_short is None or prev_long is None:
        return None
    
    # Check for crossover
    was_below = prev_short <= prev_long
    is_above = current_short > current_long
    
    was_above = prev_short >= prev_long
    is_below = current_short < current_long
    
    if was_below and is_above:
        return 'golden_cross'
    elif was_above and is_below:
        return 'death_cross'
    
    return None


class TrendFollowingPositionManager:
    """Manages trend-following strategy positions."""
    
    def __init__(self):
        self.in_position = False
        self.entry_price = None
        self.position_size = 0.0
        self.entry_reason = None
        self.stop_loss = None
        self.take_profit = None
        self.risk_reward_info = None
        
    def enter_long(self, price: float, size: float, reason: str = "golden_cross", 
                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                   risk_reward_info: Optional[Dict] = None) -> bool:
        """
        Enter a long position.
        
        Args:
            price: Entry price
            size: Position size in USDC
            reason: Reason for entry
            stop_loss: Stop-loss price level
            take_profit: Take-profit price level
            risk_reward_info: Risk/reward metrics
        """
        if self.in_position:
            logger.warning("Already in position, cannot enter again")
            return False
        
        self.in_position = True
        self.entry_price = price
        self.position_size = size
        self.entry_reason = reason
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_reward_info = risk_reward_info
        
        log_msg = f"🟢 Trend strategy: ENTER LONG at {price:.4f} for {size:.0f} units ({reason})"
        if stop_loss and take_profit:
            risk_reward_ratio = risk_reward_info.get("risk_reward_ratio", 0) if risk_reward_info else 0
            log_msg += f" | SL={stop_loss:.4f}, TP={take_profit:.4f}, R:R={risk_reward_ratio:.2f}:1"
        logger.info(log_msg)
        return True
    
    def exit_position(self, price: float, reason: str = "death_cross") -> Optional[Dict]:
        """Exit the current position."""
        if not self.in_position:
            logger.warning("Not in position, cannot exit")
            return None
        
        pnl = (price - self.entry_price) * self.position_size if self.entry_price else 0
        pnl_pct = ((price / self.entry_price) - 1) * 100 if self.entry_price else 0
        
        log_msg = (
            f"🔴 Trend strategy: EXIT LONG at {price:.4f} "
            f"(entry={self.entry_price:.4f}, pnl=${pnl:.2f}, {pnl_pct:.2f}%) "
            f"({reason})"
        )
        if self.stop_loss:
            log_msg += f" [SL={self.stop_loss:.4f}]"
        logger.info(log_msg)
        
        self.in_position = False
        result = {
            "entry_price": self.entry_price,
            "exit_price": price,
            "position_size": self.position_size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }
        
        self.entry_price = None
        self.position_size = 0.0
        self.entry_reason = None
        self.stop_loss = None
        self.take_profit = None
        self.risk_reward_info = None
        
        return result


def run_trend_strategy(
    prices: List[float],
    position_manager: TrendFollowingPositionManager,
    risk_manager,
    current_balance: float = 0.0,
    short_period: int = 10,
    long_period: int = 50,
    order_size: Optional[float] = None,
) -> Optional[Dict]:
    """
    Execute trend-following strategy with risk management.
    
    Implements a simple momentum system:
    - Enter long when short MA crosses above long MA (golden cross)
    - Exit when short MA crosses below long MA (death cross)
    
    Args:
        prices: List of recent price data
        position_manager: Position manager instance
        risk_manager: Risk manager instance for position sizing and SL/TP
        current_balance: Current account balance for position sizing
        short_period: Short-term MA period (default 10)
        long_period: Long-term MA period (default 50)
        order_size: Fixed order size (if None, use risk manager sizing)
        
    Returns:
        Dictionary with trade details if trade executed, None otherwise
    """
    if len(prices) < long_period + 1:
        logger.debug(f"Insufficient data for trend strategy: need {long_period + 1}, got {len(prices)}")
        return None
    
    current_price = prices[-1]
    crossover = get_ma_crossover(prices, short_period, long_period)
    
    if crossover is None:
        logger.debug("No MA crossover detected")
        return None
    
    # Golden cross: potential entry signal
    if crossover == 'golden_cross':
        if position_manager.in_position:
            logger.debug("Already in position, skipping entry")
            return None
        
        # Check risk manager permissions
        if hasattr(risk_manager, 'is_trading_halted') and risk_manager.is_trading_halted():
            logger.warning("Risk manager blocked trade: daily loss limit reached")
            return None
        
        # Calculate position size
        if order_size is None and risk_manager and current_balance > 0:
            atr = risk_manager.calculate_atr(prices, risk_manager.limits.atr_period)
            position_size = risk_manager.get_position_size(
                current_balance,
                atr=atr,
                entry_price=current_price
            )
            
            # Calculate SL/TP
            if atr:
                sl = risk_manager.get_stop_loss(current_price, atr, "long")
                tp = risk_manager.get_take_profit(current_price, atr, "long")
                risk_reward = risk_manager.get_risk_reward_levels(current_price, atr, "long")
            else:
                sl, tp, risk_reward = None, None, None
        else:
            position_size = order_size or 50.0
            sl, tp, risk_reward = None, None, None
        
        # Check trading permissions
        can_trade, reason = risk_manager.can_trade(position_size, current_balance) if risk_manager else (True, None)
        if not can_trade:
            logger.warning(f"Risk manager blocked trade: {reason}")
            return None
        
        # Enter position
        position_manager.enter_long(current_price, position_size, reason="golden_cross",
                                   stop_loss=sl, take_profit=tp, risk_reward_info=risk_reward)
        
        return {
            "action": "entry",
            "signal": "golden_cross",
            "price": current_price,
            "size": position_size,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward": risk_reward,
            "timestamp": len(prices),
        }
    
    # Death cross: potential exit signal
    elif crossover == 'death_cross':
        if not position_manager.in_position:
            logger.debug("Not in position, no exit action")
            return None
        
        # Exit position
        exit_result = position_manager.exit_position(current_price, reason="death_cross")
        
        # Record trade result with risk manager
        if exit_result and risk_manager:
            pnl = exit_result.get("pnl", 0)
            risk_manager.record_trade_result(pnl)
        
        return {
            "action": "exit",
            "signal": "death_cross",
            "price": current_price,
            **exit_result,
        } if exit_result else None
    
    return None

def run_strategy(prices: List[float], timestamps: List[str], current_index: int, 
                 state: Dict, params: Dict, risk_manager=None) -> Tuple[Optional[str], Dict]:
    """
    Run trend-following strategy at current index.
    
    This is the unified interface for backtesting integration.
    
    Args:
        prices: List of all historical prices
        timestamps: List of all timestamps
        current_index: Current position in price history
        state: Mutable state dict (tracks open positions)
        params: Strategy parameters:
            - short_ma_period: Short MA period (default 10)
            - long_ma_period: Long MA period (default 50)
        risk_manager: Risk manager instance (optional)
        
    Returns:
        (signal, state_update) tuple:
            - signal: "enter" or "exit" or None
            - state_update: Dict with trade details or empty dict
    """
    short_period = params.get("short_ma_period", 10)
    long_period = params.get("long_ma_period", 50)
    
    # Need minimum history
    if current_index < long_period:
        return None, {}
    
    # Get price window up to current index
    price_window = prices[:current_index + 1]
    current_price = prices[current_index]
    
    # Check for crossover
    crossover = get_ma_crossover(price_window, short_period, long_period)
    
    in_position = state.get("in_position", False)
    
    if crossover == "golden_cross" and not in_position:
        # Enter long
        return "enter", {
            "signal_type": "trend_follow",
            "entry_price": current_price,
            "timestamp": timestamps[current_index],
            "reason": "golden_cross",
        }
    
    elif crossover == "death_cross" and in_position:
        # Exit long
        entry_price = state.get("entry_price", current_price)
        profit = (current_price - entry_price)
        profit_pct = (profit / entry_price * 100) if entry_price > 0 else 0
        
        return "exit", {
            "signal_type": "trend_follow",
            "exit_price": current_price,
            "profit": profit,
            "profit_pct": profit_pct,
            "timestamp": timestamps[current_index],
            "reason": "death_cross",
        }
    
    return None, {}