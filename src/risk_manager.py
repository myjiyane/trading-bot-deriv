"""
Risk management module for the arbitrage bot.

Provides features like:
- Dynamic position sizing based on volatility
- Stop-loss and take-profit calculation using ATR
- Daily drawdown limits with automatic trading halt
- Position size limits and trading restrictions
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits configuration."""
    max_daily_loss: Optional[float] = None  # Maximum loss per day in USDC
    max_position_size: Optional[float] = None  # Maximum position size in USDC
    max_trades_per_day: Optional[int] = None  # Maximum number of trades per day
    min_balance_required: float = 10.0  # Minimum balance required to continue trading
    max_balance_utilization: float = 0.8  # Maximum percentage of balance to use per trade
    
    # Dynamic position sizing
    max_risk_per_trade: float = 0.02  # Max risk as % of account equity (default 2%)
    min_position_size: float = 10.0  # Minimum trade size in USDC
    
    # Stop-loss and take-profit
    atr_period: int = 14  # Period for ATR calculation
    sl_atr_multiplier: float = 1.0  # Stop-loss = entry ± (ATR × multiplier)
    tp_atr_multiplier: float = 1.5  # Take-profit = entry ± (ATR × multiplier)
    
    # Daily limits
    daily_loss_limit: Optional[float] = None  # Daily loss limit (overrides max_daily_loss if set)


class RiskManager:
    """Manages risk limits, position sizing, and stop-loss/take-profit levels.
    
    Supports both single-market and multi-market tracking with isolated per-market stats.
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits
        self.daily_stats: dict = {
            "date": datetime.now().date().isoformat(),
            "trades_count": 0,
            "total_loss": 0.0,
            "total_profit": 0.0,
            "trading_halted": False,
        }
        # Per-market daily statistics (keyed by market_slug)
        self.market_daily_stats: Dict[str, dict] = {}
    
    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if a new day has started."""
        today = datetime.now().date().isoformat()
        if self.daily_stats["date"] != today:
            self.daily_stats = {
                "date": today,
                "trades_count": 0,
                "total_loss": 0.0,
                "total_profit": 0.0,
                "trading_halted": False,
            }
            logger.info("✅ Daily risk limits reset for new day")
    
    def _get_market_daily_stats(self, market_slug: str) -> dict:
        """Get or initialize daily stats for a specific market."""
        if market_slug not in self.market_daily_stats:
            self.market_daily_stats[market_slug] = {
                "date": datetime.now().date().isoformat(),
                "trades_count": 0,
                "total_loss": 0.0,
                "total_profit": 0.0,
                "trading_halted": False,
            }
        
        # Reset if new day
        today = datetime.now().date().isoformat()
        if self.market_daily_stats[market_slug]["date"] != today:
            self.market_daily_stats[market_slug] = {
                "date": today,
                "trades_count": 0,
                "total_loss": 0.0,
                "total_profit": 0.0,
                "trading_halted": False,
            }
        
        return self.market_daily_stats[market_slug]
    
    # ============================================================================
    # POSITION SIZING
    # ============================================================================
    
    def calculate_atr(self, prices: list, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        
        ATR measures volatility independent of direction.
        
        Args:
            prices: List of recent price data
            period: ATR calculation period (default 14)
            
        Returns:
            ATR value, or None if insufficient data
        """
        if len(prices) < period + 1:
            return None
        
        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            high_close = abs(prices[i] - prices[i-1])
            low_close = abs(prices[i-1] - prices[i])
            tr = max(high_low, high_close, low_close)
            true_ranges.append(tr)
        
        # Calculate ATR as simple moving average of TR
        recent_tr = true_ranges[-period:]
        atr = sum(recent_tr) / len(recent_tr) if recent_tr else None
        
        return atr
    
    def get_position_size(self, 
                         account_equity: float, 
                         volatility: Optional[float] = None,
                         atr: Optional[float] = None,
                         entry_price: Optional[float] = None) -> float:
        """
        Calculate dynamic position size based on account equity and volatility.
        
        Position sizing formula:
        position_size = (account_equity × max_risk_per_trade) / (price × volatility_factor)
        
        Where volatility_factor is based on ATR when available.
        
        Args:
            account_equity: Current account equity in USDC
            volatility: Optional volatility estimate (0.01 = 1%)
            atr: Optional ATR value for volatility normalization
            entry_price: Entry price (used with ATR to calculate risk distance)
            
        Returns:
            Position size in USDC, respecting min/max constraints
        """
        # Calculate base risk amount
        risk_amount = account_equity * self.limits.max_risk_per_trade
        
        # Apply volatility adjustment if ATR available
        if atr is not None and entry_price is not None and entry_price > 0:
            # Risk distance = ATR (distance to stop-loss)
            risk_distance = atr
            # Position size = risk_amount / risk_distance
            position_size = risk_amount / risk_distance if risk_distance > 0 else risk_amount
        else:
            # No volatility data, use fixed risk amount as position size
            position_size = risk_amount
        
        # Apply constraints
        position_size = max(position_size, self.limits.min_position_size)  # Minimum
        
        if self.limits.max_position_size:
            position_size = min(position_size, self.limits.max_position_size)  # Maximum
        
        # Check balance utilization
        max_from_balance = account_equity * self.limits.max_balance_utilization
        position_size = min(position_size, max_from_balance)
        
        logger.debug(
            f"Position sizing: equity=${account_equity:.2f}, risk={self.limits.max_risk_per_trade*100:.1f}%, "
            f"atr={atr:.6f if atr else 'N/A'}, position=${position_size:.2f}"
        )
        
        return position_size
    
    # ============================================================================
    # STOP-LOSS AND TAKE-PROFIT
    # ============================================================================
    
    def get_stop_loss(self, entry_price: float, atr: float, direction: str = "long") -> float:
        """
        Calculate stop-loss level based on ATR.
        
        Stop-loss = entry_price ± (ATR × sl_atr_multiplier)
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            direction: 'long' or 'short'
            
        Returns:
            Stop-loss price level
        """
        if direction == "long":
            sl = entry_price - (atr * self.limits.sl_atr_multiplier)
        else:  # short
            sl = entry_price + (atr * self.limits.sl_atr_multiplier)
        
        return max(0, sl)  # Ensure positive price
    
    def get_take_profit(self, entry_price: float, atr: float, direction: str = "long") -> float:
        """
        Calculate take-profit level based on ATR.
        
        Take-profit = entry_price ± (ATR × tp_atr_multiplier)
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            direction: 'long' or 'short'
            
        Returns:
            Take-profit price level
        """
        if direction == "long":
            tp = entry_price + (atr * self.limits.tp_atr_multiplier)
        else:  # short
            tp = entry_price - (atr * self.limits.tp_atr_multiplier)
        
        return tp
    
    def get_risk_reward_levels(self,
                               entry_price: float,
                               atr: float,
                               direction: str = "long") -> Dict[str, float]:
        """
        Get complete risk/reward structure for a trade.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            
        Returns:
            Dict with entry, stop_loss, and take_profit levels
        """
        sl = self.get_stop_loss(entry_price, atr, direction)
        tp = self.get_take_profit(entry_price, atr, direction)
        
        return {
            "entry": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_distance": abs(entry_price - sl),
            "reward_distance": abs(tp - entry_price),
            "risk_reward_ratio": abs(tp - entry_price) / abs(entry_price - sl) if entry_price != sl else 0,
        }
    
    # ============================================================================
    # TRADING PERMISSIONS AND LIMITS
    # ============================================================================
    
    def can_trade(self, trade_size: float, current_balance: float, market_slug: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed based on risk limits.
        
        Args:
            trade_size: Size of the trade in USDC
            current_balance: Current account balance in USDC
            market_slug: Optional market slug for per-market tracking
            
        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        self._reset_daily_stats_if_needed()
        
        # Use market-specific stats if provided, otherwise use global stats
        stats = self._get_market_daily_stats(market_slug) if market_slug else self.daily_stats
        
        # Check if trading is halted due to daily loss limit
        if stats["trading_halted"]:
            market_msg = f" on {market_slug}" if market_slug else ""
            return False, f"Trading halted{market_msg}: daily loss limit reached"
        
        # Check minimum balance
        if current_balance < self.limits.min_balance_required:
            return False, f"Balance ${current_balance:.2f} below minimum ${self.limits.min_balance_required:.2f}"
        
        # Check maximum position size
        if self.limits.max_position_size and trade_size > self.limits.max_position_size:
            return False, f"Trade size ${trade_size:.2f} exceeds maximum ${self.limits.max_position_size:.2f}"
        
        # Check balance utilization
        max_trade_size = current_balance * self.limits.max_balance_utilization
        if trade_size > max_trade_size:
            return False, f"Trade size ${trade_size:.2f} exceeds {self.limits.max_balance_utilization*100:.0f}% of balance"
        
        # Check daily trade count (market-specific if provided)
        if self.limits.max_trades_per_day:
            if stats["trades_count"] >= self.limits.max_trades_per_day:
                market_msg = f" on {market_slug}" if market_slug else ""
                return False, f"Daily trade limit ({self.limits.max_trades_per_day}){market_msg} reached"
        
        # Check daily loss limit (market-specific if provided)
        loss_limit = self.limits.daily_loss_limit or self.limits.max_daily_loss
        if loss_limit:
            net_loss = stats["total_loss"] - stats["total_profit"]
            if net_loss >= loss_limit:
                stats["trading_halted"] = True
                market_msg = f" on {market_slug}" if market_slug else ""
                return False, f"Daily loss limit (${loss_limit:.2f}){market_msg} reached - trading halted"
        
        return True, None
    
    def record_trade_result(self, profit: float, market_slug: str = None) -> None:
        """
        Record the result of a trade for risk tracking.
        
        Args:
            profit: Profit/loss from the trade (negative for losses)
            market_slug: Optional market slug for per-market tracking
        """
        self._reset_daily_stats_if_needed()
        
        # Update both global and market-specific stats
        stats = self._get_market_daily_stats(market_slug) if market_slug else self.daily_stats
        self.daily_stats["trades_count"] += 1
        stats["trades_count"] += 1
        
        if profit > 0:
            self.daily_stats["total_profit"] += profit
            stats["total_profit"] += profit
            market_msg = f" [{market_slug}]" if market_slug else ""
            logger.info(f"📈 Trade profit: ${profit:.2f}{market_msg}")
        else:
            self.daily_stats["total_loss"] += abs(profit)
            stats["total_loss"] += abs(profit)
            market_msg = f" [{market_slug}]" if market_slug else ""
            logger.warning(f"📉 Trade loss: ${abs(profit):.2f}{market_msg}")
        
        # Check if daily loss limit hit (global or market-specific)
        loss_limit = self.limits.daily_loss_limit or self.limits.max_daily_loss
        if loss_limit:
            net_loss = stats["total_loss"] - stats["total_profit"]
            if net_loss >= loss_limit:
                stats["trading_halted"] = True
                market_msg = f" on {market_slug}" if market_slug else ""
                logger.error(f"🛑 Daily loss limit reached{market_msg}: ${net_loss:.2f} >= ${loss_limit:.2f}")
    
    # ============================================================================
    # STATUS AND REPORTING
    # ============================================================================
    
    def get_daily_stats(self, market_slug: str = None) -> Dict:
        """
        Get current daily statistics.
        
        Args:
            market_slug: Optional market slug for market-specific stats
            
        Returns:
            Dictionary with daily statistics
        """
        self._reset_daily_stats_if_needed()
        stats = self._get_market_daily_stats(market_slug) if market_slug else self.daily_stats
        net_pnl = stats["total_profit"] - stats["total_loss"]
        return {
            **stats,
            "net_pnl": net_pnl,
            "trading_allowed": not stats["trading_halted"],
        }
    
    def is_daily_loss_limit_reached(self) -> bool:
        """Check if daily loss limit has been reached."""
        loss_limit = self.limits.daily_loss_limit or self.limits.max_daily_loss
        if not loss_limit:
            return False
        
        self._reset_daily_stats_if_needed()
        net_loss = self.daily_stats["total_loss"] - self.daily_stats["total_profit"]
        return net_loss >= loss_limit
    
    def is_trading_halted(self) -> bool:
        """Check if trading has been halted for the day."""
        self._reset_daily_stats_if_needed()
        return self.daily_stats["trading_halted"]
    
    def get_risk_status(self, current_balance: float) -> Dict:
        """
        Get comprehensive risk status report.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Dict with risk status information
        """
        self._reset_daily_stats_if_needed()
        
        loss_limit = self.limits.daily_loss_limit or self.limits.max_daily_loss
        net_loss = self.daily_stats["total_loss"] - self.daily_stats["total_profit"]
        
        return {
            "date": self.daily_stats["date"],
            "current_balance": current_balance,
            "trades_today": self.daily_stats["trades_count"],
            "total_profit": self.daily_stats["total_profit"],
            "total_loss": self.daily_stats["total_loss"],
            "net_pnl": self.daily_stats["total_profit"] - self.daily_stats["total_loss"],
            "daily_loss_limit": loss_limit,
            "daily_loss_used": net_loss,
            "daily_loss_remaining": max(0, loss_limit - net_loss) if loss_limit else None,
            "daily_loss_pct": (net_loss / loss_limit * 100) if loss_limit and loss_limit > 0 else 0,
            "trading_halted": self.daily_stats["trading_halted"],
            "trading_allowed": not self.daily_stats["trading_halted"],
        }

