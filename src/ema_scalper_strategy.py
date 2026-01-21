"""
EMA crossover scalping strategy with RSI filter.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    alpha = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_ema_crossover(prices: List[float], short_period: int, long_period: int) -> Optional[str]:
    if len(prices) < long_period + 1:
        return None
    current_short = calculate_ema(prices, short_period)
    current_long = calculate_ema(prices, long_period)
    prev_short = calculate_ema(prices[:-1], short_period)
    prev_long = calculate_ema(prices[:-1], long_period)
    if None in (current_short, current_long, prev_short, prev_long):
        return None
    was_below = prev_short <= prev_long
    is_above = current_short > current_long
    was_above = prev_short >= prev_long
    is_below = current_short < current_long
    if was_below and is_above:
        return "bull_cross"
    if was_above and is_below:
        return "bear_cross"
    return None


class EmaScalperPositionManager:
    def __init__(self):
        self.in_position = False
        self.entry_price = None
        self.position_size = 0.0

    def enter_long(self, price: float, size: float) -> bool:
        if self.in_position:
            return False
        self.in_position = True
        self.entry_price = price
        self.position_size = size
        logger.info(f"🟢 EMA scalper: ENTER LONG at {price:.4f} for {size:.0f} units")
        return True

    def exit_position(self, price: float, reason: str) -> Optional[Dict]:
        if not self.in_position:
            return None
        pnl = (price - self.entry_price) * self.position_size if self.entry_price else 0
        pnl_pct = ((price / self.entry_price) - 1) * 100 if self.entry_price else 0
        logger.info(
            f"🔴 EMA scalper: EXIT LONG at {price:.4f} (entry={self.entry_price:.4f}, "
            f"pnl=${pnl:.2f}, {pnl_pct:.2f}%) ({reason})"
        )
        self.in_position = False
        result = {
            "entry_price": self.entry_price,
            "exit_price": price,
            "position_size": self.position_size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
        }
        self.entry_price = None
        self.position_size = 0.0
        return result


def run_ema_scalper(
    prices: List[float],
    position_manager: EmaScalperPositionManager,
    current_balance: float,
    short_period: int,
    long_period: int,
    rsi_period: int,
    rsi_min: float,
    rsi_max: float,
    order_size: float,
) -> Optional[Dict]:
    if len(prices) < long_period + 1:
        return None

    current_price = prices[-1]
    rsi = calculate_rsi(prices, rsi_period)
    crossover = get_ema_crossover(prices, short_period, long_period)

    if not position_manager.in_position and crossover == "bull_cross" and rsi is not None:
        if rsi_min < rsi < rsi_max:
            position_manager.enter_long(current_price, order_size)
            return {
                "action": "entry",
                "signal": "ema_bull_cross",
                "price": current_price,
                "size": order_size,
                "rsi": rsi,
            }

    if position_manager.in_position and crossover == "bear_cross":
        exit_result = position_manager.exit_position(current_price, reason="ema_bear_cross")
        if exit_result:
            return {
                "action": "exit",
                "signal": "ema_bear_cross",
                "price": current_price,
                **exit_result,
            }

    return None


def run_strategy(
    prices: List[float],
    timestamps: List[str],
    current_index: int,
    state: Dict,
    params: Dict,
) -> Tuple[Optional[str], Dict]:
    short_period = params.get("ema_short", 12)
    long_period = params.get("ema_long", 26)
    rsi_period = params.get("rsi_period", 14)
    rsi_min = params.get("rsi_min", 30)
    rsi_max = params.get("rsi_max", 70)

    if current_index < long_period:
        return None, {}
    price_window = prices[:current_index + 1]
    current_price = prices[current_index]
    rsi = calculate_rsi(price_window, rsi_period)
    crossover = get_ema_crossover(price_window, short_period, long_period)
    in_position = state.get("in_position", False)

    if crossover == "bull_cross" and not in_position and rsi is not None:
        if rsi_min < rsi < rsi_max:
            return "enter", {
                "signal_type": "ema_scalper",
                "entry_price": current_price,
                "timestamp": timestamps[current_index],
                "rsi": rsi,
            }

    if crossover == "bear_cross" and in_position:
        entry_price = state.get("entry_price", current_price)
        profit = (current_price - entry_price)
        profit_pct = (profit / entry_price * 100) if entry_price > 0 else 0
        return "exit", {
            "signal_type": "ema_scalper",
            "exit_price": current_price,
            "profit": profit,
            "profit_pct": profit_pct,
            "timestamp": timestamps[current_index],
            "reason": "ema_bear_cross",
        }

    return None, {}
