"""
Mean reversion strategy for range-bound trading.

Implements a Bollinger band-based mean reversion system that enters when price
dips below the lower band and exits when price returns to the mean.
"""

import logging
import math
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculate simple moving average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_std_dev(prices: List[float], period: int) -> Optional[float]:
    """Calculate standard deviation over the last `period` prices."""
    if len(prices) < period:
        return None
    
    recent = prices[-period:]
    mean = sum(recent) / period
    variance = sum((x - mean) ** 2 for x in recent) / period
    return math.sqrt(variance)


def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Optional[Dict[str, float]]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Dict with keys 'middle', 'upper', 'lower', or None if insufficient data
    """
    sma = calculate_sma(prices, period)
    std_dev = calculate_std_dev(prices, period)
    
    if sma is None or std_dev is None:
        return None
    
    return {
        'middle': sma,
        'upper': sma + (num_std * std_dev),
        'lower': sma - (num_std * std_dev),
        'std_dev': std_dev,
    }


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum on a scale of 0-100.
    RSI < 30 indicates oversold conditions (potential buy signal).
    RSI > 70 indicates overbought conditions (potential sell signal).
    
    Args:
        prices: List of price data
        period: Period for RSI calculation (default 14)
        
    Returns:
        RSI value (0-100), or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains[-period:]) / period if len(gains) >= period else sum(gains) / len(gains)
    avg_loss = sum(losses[-period:]) / period if len(losses) >= period else sum(losses) / len(losses)
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


class MeanReversionPositionManager:
    """Manages mean reversion strategy positions."""
    
    def __init__(self):
        self.in_position = False
        self.entry_price = None
        self.position_size = 0.0
        self.entry_reason = None
        self.stop_loss = None
        self.take_profit = None
        self.risk_reward_info = None
        
    def enter_long(self, price: float, size: float, reason: str = "oversold",
                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                   risk_reward_info: Optional[Dict] = None) -> bool:
        """
        Enter a long position on mean reversion signal.
        
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
        
        log_msg = f"🟢 Mean reversion strategy: ENTER LONG at {price:.4f} for {size:.0f} units ({reason})"
        if stop_loss and take_profit:
            risk_reward_ratio = risk_reward_info.get("risk_reward_ratio", 0) if risk_reward_info else 0
            log_msg += f" | SL={stop_loss:.4f}, TP={take_profit:.4f}, R:R={risk_reward_ratio:.2f}:1"
        logger.info(log_msg)
        return True
    
    def exit_position(self, price: float, reason: str = "mean_reversion") -> Optional[Dict]:
        """Exit the current position."""
        if not self.in_position:
            logger.warning("Not in position, cannot exit")
            return None
        
        pnl = (price - self.entry_price) * self.position_size if self.entry_price else 0
        pnl_pct = ((price / self.entry_price) - 1) * 100 if self.entry_price else 0
        
        log_msg = (
            f"🔴 Mean reversion strategy: EXIT LONG at {price:.4f} "
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


def run_mean_reversion_strategy(
    prices: List[float],
    position_manager: MeanReversionPositionManager,
    risk_manager,
    current_balance: float = 0.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    oversold_threshold: float = 30.0,
    overbought_threshold: float = 70.0,
    order_size: Optional[float] = None,
) -> Optional[Dict]:
    """
    Execute mean reversion strategy with risk management.
    
    Implements a Bollinger band and RSI-based system:
    - Enter long when price dips below lower band AND RSI < 30 (oversold)
    - Exit when price returns to middle band OR RSI > 70 (overbought)
    
    Args:
        prices: List of recent price data
        position_manager: Position manager instance
        risk_manager: Risk manager instance for position sizing and SL/TP
        current_balance: Current account balance for position sizing
        bb_period: Bollinger band period (default 20)
        bb_std: Number of standard deviations for bands (default 2.0)
        rsi_period: RSI calculation period (default 14)
        oversold_threshold: RSI threshold for entry (default 30)
        overbought_threshold: RSI threshold for exit (default 70)
        order_size: Fixed order size (if None, use risk manager sizing)
        
    Returns:
        Dictionary with trade details if trade executed, None otherwise
    """
    if len(prices) < max(bb_period, rsi_period) + 1:
        logger.debug(f"Insufficient data for mean reversion strategy")
        return None
    
    current_price = prices[-1]
    
    # Calculate technical indicators
    bb = calculate_bollinger_bands(prices, bb_period, bb_std)
    rsi = calculate_rsi(prices, rsi_period)
    
    if bb is None or rsi is None:
        logger.debug("Unable to calculate technical indicators")
        return None
    
    middle_band = bb['middle']
    lower_band = bb['lower']
    upper_band = bb['upper']
    
    logger.debug(
        f"Mean reversion indicators: price={current_price:.4f}, "
        f"bb_lower={lower_band:.4f}, bb_middle={middle_band:.4f}, bb_upper={upper_band:.4f}, "
        f"rsi={rsi:.2f}"
    )
    
    # Entry signal: price below lower band AND RSI oversold
    if not position_manager.in_position:
        if current_price < lower_band and rsi < oversold_threshold:
            # Check if trading is halted
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
            position_manager.enter_long(current_price, position_size, reason=f"oversold_rsi={rsi:.1f}",
                                       stop_loss=sl, take_profit=tp, risk_reward_info=risk_reward)
            
            return {
                "action": "entry",
                "signal": "oversold",
                "price": current_price,
                "size": position_size,
                "rsi": rsi,
                "bb_lower": lower_band,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward": risk_reward,
                "timestamp": len(prices),
            }
    
    # Exit signal: price back to middle band OR RSI overbought
    else:
        exit_reason = None
        
        if current_price >= middle_band:
            exit_reason = "mean_reversion"
        elif rsi > overbought_threshold:
            exit_reason = "overbought"
        
        if exit_reason:
            # Exit position
            exit_result = position_manager.exit_position(current_price, reason=exit_reason)
            
            # Record trade result with risk manager
            if exit_result and risk_manager:
                pnl = exit_result.get("pnl", 0)
                risk_manager.record_trade_result(pnl)
            
            return {
                "action": "exit",
                "signal": exit_reason,
                "price": current_price,
                "rsi": rsi,
                "bb_middle": middle_band,
                **exit_result,
            } if exit_result else None
    
    return None

def run_strategy(prices: List[float], timestamps: List[str], current_index: int,
                 state: Dict, params: Dict, risk_manager=None) -> tuple:
    """
    Run mean-reversion strategy at current index.
    
    This is the unified interface for backtesting integration.
    
    Args:
        prices: List of all historical prices
        timestamps: List of all timestamps
        current_index: Current position in price history
        state: Mutable state dict (tracks open positions)
        params: Strategy parameters:
            - bb_period: Bollinger Band period (default 20)
            - bb_std_dev: Bollinger Band std dev (default 2.0)
            - rsi_period: RSI period (default 14)
            - rsi_threshold: RSI oversold threshold (default 30)
        risk_manager: Risk manager instance (optional)
        
    Returns:
        (signal, state_update) tuple:
            - signal: "enter" or "exit" or None
            - state_update: Dict with trade details or empty dict
    """
    bb_period = params.get("bb_period", 20)
    bb_std_dev = params.get("bb_std_dev", 2.0)
    rsi_period = params.get("rsi_period", 14)
    rsi_threshold = params.get("rsi_threshold", 30)
    
    # Need minimum history
    if current_index < max(bb_period, rsi_period):
        return None, {}
    
    # Get price window up to current index
    price_window = prices[:current_index + 1]
    current_price = prices[current_index]
    
    # Calculate indicators
    bb = calculate_bollinger_bands(price_window, bb_period, bb_std_dev)
    rsi = calculate_rsi(price_window, rsi_period)
    
    if not bb or rsi is None:
        return None, {}
    
    in_position = state.get("in_position", False)
    
    # Entry: price below lower band AND RSI < threshold (oversold)
    if current_price < bb["lower"] and rsi < rsi_threshold and not in_position:
        return "enter", {
            "signal_type": "mean_reversion",
            "entry_price": current_price,
            "timestamp": timestamps[current_index],
            "reason": f"oversold (RSI={rsi:.1f})",
            "bb_lower": bb["lower"],
        }
    
    # Exit: price above middle band OR RSI > 70 (overbought)
    elif in_position:
        if current_price > bb["middle"] or rsi > 70:
            entry_price = state.get("entry_price", current_price)
            profit = (current_price - entry_price)
            profit_pct = (profit / entry_price * 100) if entry_price > 0 else 0
            
            exit_reason = "back to middle" if current_price > bb["middle"] else "overbought"
            
            return "exit", {
                "signal_type": "mean_reversion",
                "exit_price": current_price,
                "profit": profit,
                "profit_pct": profit_pct,
                "timestamp": timestamps[current_index],
                "reason": exit_reason,
                "rsi": rsi,
            }
    
    return None, {}