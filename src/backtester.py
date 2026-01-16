"""
Backtesting framework for strategy optimization and performance analysis.

Allows you to:
1. Replay historical price data through strategies
2. Optimize parameters (MA periods, RSI thresholds, ATR multipliers, etc.)
3. Compare performance across different parameter sets
4. Generate detailed backtest reports
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable
import json
from pathlib import Path

from .trend_detector import is_trending
from .trend_follow_strategy import run_trend_strategy
from .mean_reversion_strategy import run_mean_reversion_strategy
from .risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)


def load_price_data(
    filepath: str = None,
    symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    interval: str = "15m",
    settings=None
) -> tuple:
    """
    Load historical price data from CSV file or external API.
    
    Can be called in two ways:
    
    1. From CSV file:
        prices, timestamps = load_price_data(filepath='prices.csv')
    
    2. From external API:
        prices, timestamps = load_price_data(
            symbol='BTC',
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='15m',
            settings=settings
        )
    
    Expected CSV format:
        timestamp,price
        2024-01-01T00:00:00Z,0.5050
        2024-01-01T00:15:00Z,0.5051
    
    Or with OHLCV:
        timestamp,open,high,low,close,volume
    
    Args:
        filepath: Path to CSV file (mutually exclusive with symbol/dates)
        symbol: Cryptocurrency symbol like 'BTC', 'ETH', 'MATIC' (requires start_date, end_date)
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        interval: Candle interval (1m, 5m, 15m, 1h, 1d)
        settings: Settings object with DATA_PROVIDER_* config
        
    Returns:
        (prices, timestamps) tuple of lists
        
    Raises:
        ValueError: If neither filepath nor symbol provided, or incomplete parameters
        Exception: From external API (network, auth, data parsing)
    """
    
    # Mode 1: Load from CSV file
    if filepath:
        return _load_price_data_csv(filepath)
    
    # Mode 2: Fetch from external API
    if symbol and start_date and end_date:
        try:
            provider = getattr(settings, "data_provider_name", "") if settings else ""
            if provider == "deriv":
                from .deriv_price_feed import fetch_historical_prices
                logger.info(f"Fetching {symbol} data from Deriv ({start_date} to {end_date})")
                return fetch_historical_prices(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    app_id=getattr(settings, "deriv_app_id", ""),
                    ws_url=getattr(settings, "deriv_ws_url", "wss://ws.deriv.com/websockets/v3"),
                )

            from .external_price_feed import fetch_historical_prices
            logger.info(f"Fetching {symbol} data from external API ({start_date} to {end_date})")
            return fetch_historical_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                settings=settings
            )
        except ImportError as e:
            logger.error(f"external_price_feed module not available: {e}")
            raise
    
    raise ValueError(
        "load_price_data requires either filepath OR (symbol, start_date, end_date)\n"
        "Usage 1: load_price_data(filepath='prices.csv')\n"
        "Usage 2: load_price_data(symbol='BTC', start_date='2024-01-01', end_date='2024-01-31')"
    )


def _load_price_data_csv(filepath: str) -> tuple:
    """Load price data from CSV file."""
    import csv
    
    prices = []
    timestamps = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'price' in row:
                price = float(row['price'])
            elif 'close' in row:
                price = float(row['close'])
            else:
                price = float(list(row.values())[1])
            
            timestamp = row.get('timestamp', '')
            prices.append(price)
            timestamps.append(timestamp)
    
    logger.info(f"Loaded {len(prices)} price bars from {filepath}")
    return prices, timestamps


def evaluate_performance(trades: List, initial_balance: float = 10000.0) -> Dict:
    """
    Evaluate trading performance metrics from trade list.
    
    Args:
        trades: List of BacktestTrade objects
        initial_balance: Starting account balance
        
    Returns:
        Dictionary with all performance metrics
    """
    if not trades:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate_pct": 0.0, "total_return": 0.0, "total_return_pct": 0.0,
            "average_win": 0.0, "average_loss": 0.0, "largest_win": 0.0,
            "largest_loss": 0.0, "profit_factor": 0.0,
            "average_trade_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
        }
    
    winning_trades = [t for t in trades if t.win]
    losing_trades = [t for t in trades if not t.win]
    
    total_profit = sum(t.profit for t in winning_trades)
    total_loss = abs(sum(t.profit for t in losing_trades))
    net_profit = total_profit - total_loss
    
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0.0
    avg_win = total_profit / len(winning_trades) if winning_trades else 0.0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0.0
    
    profit_factor = (total_profit / total_loss) if total_loss > 0 else (1.0 if total_profit > 0 else 0.0)
    avg_trade_return_pct = (net_profit / initial_balance * 100 / len(trades)) if trades else 0.0
    
    # Sharpe ratio
    returns = [t.profit_pct for t in trades]
    if returns:
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        sharpe = (mean_return / std_dev) if std_dev > 0 else 0.0
    else:
        sharpe = 0.0
    
    max_drawdown = 0.0
    peak_balance = initial_balance
    for trade in trades:
        peak_balance = max(peak_balance, peak_balance + trade.profit)
        drawdown = ((peak_balance - (peak_balance - trade.profit)) / peak_balance * 100) if peak_balance > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        "total_trades": len(trades), "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades), "win_rate_pct": win_rate,
        "total_return": net_profit, "total_return_pct": (net_profit / initial_balance * 100),
        "average_win": avg_win, "average_loss": avg_loss,
        "largest_win": max((t.profit for t in winning_trades), default=0.0),
        "largest_loss": min((t.profit for t in losing_trades), default=0.0),
        "profit_factor": profit_factor, "average_trade_return_pct": avg_trade_return_pct,
        "max_drawdown_pct": max_drawdown, "sharpe_ratio": sharpe,
    }


@dataclass
class PriceBar:
    """Single price bar (OHLCV data)."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@dataclass
class BacktestTrade:
    """Record of a trade executed during backtest."""
    timestamp: str
    signal_type: str  # "trend_follow" or "mean_reversion"
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    position_size: float = 1.0
    profit: float = 0.0
    profit_pct: float = 0.0
    win: bool = False


@dataclass
class BacktestResult:
    """Complete backtest result for a parameter set."""
    params: Dict  # Parameter set used
    trades: List[BacktestTrade]
    total_return_pct: float
    win_rate: float
    profit_factor: float  # Gross profit / Gross loss
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    avg_trade_return_pct: float
    largest_win: float
    largest_loss: float
    
    def summary_dict(self) -> Dict:
        """Get summary as dictionary."""
        return {
            "params": self.params,
            "total_return_pct": f"{self.total_return_pct:.2f}%",
            "win_rate": f"{self.win_rate:.1f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown_pct": f"{self.max_drawdown_pct:.2f}%",
            "num_trades": self.num_trades,
            "winning_trades": self.num_winning_trades,
            "losing_trades": self.num_losing_trades,
            "avg_trade_return": f"{self.avg_trade_return_pct:.2f}%",
            "largest_win": f"${self.largest_win:.2f}",
            "largest_loss": f"${self.largest_loss:.2f}",
        }


class Backtester:
    """Backtesting engine for strategy optimization."""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting account balance in USDC
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[str, float]] = []  # (timestamp, balance)
    
    def backtest_prices(
        self,
        prices: List[float],
        timestamps: List[str],
        parameters: Dict,
        position_size: float = 1.0,
    ) -> BacktestResult:
        """
        Run backtest on a series of prices.
        
        Args:
            prices: List of prices (closing prices or midpoint)
            timestamps: List of timestamps for each price
            parameters: Strategy parameters (MA periods, RSI threshold, etc.)
            position_size: Position size per trade
            
        Returns:
            BacktestResult with full statistics
        """
        self.trades = []
        self.equity_curve = []
        self.current_balance = self.initial_balance
        
        if len(prices) < 60:  # Need minimum history for indicators
            logger.warning(f"Insufficient price data: {len(prices)} bars (need 60+)")
            return self._create_empty_result(parameters)
        
        # Maintain price history for indicators
        price_history = []
        open_position = None
        
        for i, (price, timestamp) in enumerate(zip(prices, timestamps)):
            price_history.append(price)
            
            # Keep rolling window of prices
            if len(price_history) > 100:
                price_history.pop(0)
            
            # Need enough history for indicators
            if len(price_history) < 60:
                continue
            
            # Evaluate trends and strategies
            trending = is_trending(
                price_history,
                short_ma_period=parameters.get("ma_short_period", 10),
                long_ma_period=parameters.get("ma_long_period", 50),
                adx_threshold=parameters.get("adx_threshold", 25.0)
            )
            
            # Generate signal
            signal = self._evaluate_signal(
                price=price,
                price_history=price_history,
                trending=trending,
                parameters=parameters
            )
            
            # Execute trade if signal and no open position
            if signal and not open_position:
                open_position = {
                    "entry_price": price,
                    "entry_timestamp": timestamp,
                    "signal_type": signal,
                    "position_size": position_size,
                }
                logger.debug(f"[{timestamp}] ENTRY: {signal} @ ${price:.4f}")
            
            # Check exit condition if position open
            if open_position:
                should_exit = self._check_exit_signal(
                    price=price,
                    price_history=price_history,
                    open_position=open_position,
                    parameters=parameters
                )
                
                if should_exit:
                    # Close position
                    profit = (price - open_position["entry_price"]) * open_position["position_size"]
                    profit_pct = ((price - open_position["entry_price"]) / open_position["entry_price"]) * 100
                    
                    trade = BacktestTrade(
                        timestamp=open_position["entry_timestamp"],
                        signal_type=open_position["signal_type"],
                        entry_price=open_position["entry_price"],
                        exit_price=price,
                        exit_timestamp=timestamp,
                        position_size=open_position["position_size"],
                        profit=profit,
                        profit_pct=profit_pct,
                        win=profit > 0,
                    )
                    self.trades.append(trade)
                    self.current_balance += profit
                    self.equity_curve.append((timestamp, self.current_balance))
                    
                    logger.debug(f"[{timestamp}] EXIT @ ${price:.4f}: P&L=${profit:.2f} ({profit_pct:.2f}%)")
                    open_position = None
        
        # Close any open position at end of test
        if open_position and prices:
            price = prices[-1]
            timestamp = timestamps[-1]
            profit = (price - open_position["entry_price"]) * open_position["position_size"]
            profit_pct = ((price - open_position["entry_price"]) / open_position["entry_price"]) * 100
            
            trade = BacktestTrade(
                timestamp=open_position["entry_timestamp"],
                signal_type=open_position["signal_type"],
                entry_price=open_position["entry_price"],
                exit_price=price,
                exit_timestamp=timestamp,
                position_size=open_position["position_size"],
                profit=profit,
                profit_pct=profit_pct,
                win=profit > 0,
            )
            self.trades.append(trade)
            self.current_balance += profit
        
        # Calculate statistics
        return self._calculate_statistics(parameters)
    
    def _evaluate_signal(
        self,
        price: float,
        price_history: List[float],
        trending: bool,
        parameters: Dict
    ) -> Optional[str]:
        """
        Evaluate if a trade signal should be generated.
        
        Returns:
            Signal type ("trend_follow", "mean_reversion") or None
        """
        if len(price_history) < 60:
            return None
        
        if trending:
            # Trend-following: Golden cross (10-MA > 50-MA)
            ma_short = sum(price_history[-10:]) / 10
            ma_long = sum(price_history[-50:]) / 50
            
            # Simple golden cross detection
            if len(price_history) > 50:
                prev_ma_short = sum(price_history[-11:-1]) / 10
                prev_ma_long = sum(price_history[-51:-1]) / 50
                
                if prev_ma_short <= prev_ma_long and ma_short > ma_long:
                    return "trend_follow"
        else:
            # Mean-reversion: Price below lower Bollinger Band + RSI < 30
            period = parameters.get("bb_period", 20)
            std_dev = parameters.get("bb_std_dev", 2.0)
            rsi_threshold = parameters.get("rsi_threshold", 30)
            
            # Calculate Bollinger Bands
            recent_prices = price_history[-period:]
            if len(recent_prices) == period:
                sma = sum(recent_prices) / period
                variance = sum((p - sma) ** 2 for p in recent_prices) / period
                std = variance ** 0.5
                lower_band = sma - (std * std_dev)
                
                # Calculate RSI
                rsi = self._calculate_rsi(price_history, period=14)
                
                if price < lower_band and rsi < rsi_threshold:
                    return "mean_reversion"
        
        return None
    
    def _check_exit_signal(
        self,
        price: float,
        price_history: List[float],
        open_position: Dict,
        parameters: Dict
    ) -> bool:
        """Check if position should be closed."""
        
        # Simple exit: price crosses back above middle band or ATR-based exit
        entry_price = open_position["entry_price"]
        profit = price - entry_price
        
        # Exit on 2% profit or 1% loss (simple risk/reward)
        exit_profit_target = parameters.get("exit_profit_target", 0.02)
        exit_loss_stop = parameters.get("exit_loss_stop", 0.01)
        
        if profit > (entry_price * exit_profit_target):
            return True  # Hit profit target
        
        if profit < -(entry_price * exit_loss_stop):
            return True  # Hit stop loss
        
        return False
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI for the price series."""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        recent_prices = prices[-period-1:]
        deltas = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_statistics(self, parameters: Dict) -> BacktestResult:
        """Calculate performance statistics from trades."""
        
        if not self.trades:
            return self._create_empty_result(parameters)
        
        total_return = self.current_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        winning_trades = [t for t in self.trades if t.win]
        losing_trades = [t for t in self.trades if not t.win]
        
        win_rate = (len(winning_trades) / len(self.trades) * 100) if self.trades else 0
        
        # Profit factor
        gross_profit = sum(t.profit for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.profit for t in losing_trades)) if losing_trades else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (1.0 if gross_profit > 0 else 0)
        
        # Average trade return
        avg_return = total_return / len(self.trades) if self.trades else 0
        avg_return_pct = (avg_return / self.initial_balance) * 100
        
        # Largest win/loss
        largest_win = max((t.profit for t in winning_trades), default=0)
        largest_loss = min((t.profit for t in losing_trades), default=0)
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio (simplified - daily returns)
        sharpe = self._calculate_sharpe_ratio()
        
        return BacktestResult(
            params=parameters,
            trades=self.trades,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            num_trades=len(self.trades),
            num_winning_trades=len(winning_trades),
            num_losing_trades=len(losing_trades),
            avg_trade_return_pct=avg_return_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        balances = [bal for _, bal in self.equity_curve]
        max_balance = balances[0]
        max_dd = 0.0
        
        for bal in balances:
            if bal > max_balance:
                max_balance = bal
            
            dd = ((max_balance - bal) / max_balance) * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate simplified Sharpe ratio."""
        if not self.trades or len(self.trades) < 2:
            return 0.0
        
        returns = [t.profit_pct for t in self.trades]
        avg_return = sum(returns) / len(returns)
        
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        # Simplified Sharpe (not annualized)
        sharpe = avg_return / std_dev
        return sharpe
    
    def _create_empty_result(self, parameters: Dict) -> BacktestResult:
        """Create empty result for invalid backtests."""
        return BacktestResult(
            params=parameters,
            trades=[],
            total_return_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            num_trades=0,
            num_winning_trades=0,
            num_losing_trades=0,
            avg_trade_return_pct=0.0,
            largest_win=0.0,
            largest_loss=0.0,
        )


class ParameterOptimizer:
    """Optimize strategy parameters using grid search or random search."""
    
    def __init__(self, backtester: Backtester):
        """
        Initialize optimizer.
        
        Args:
            backtester: Backtester instance to use for evaluations
        """
        self.backtester = backtester
        self.results: List[BacktestResult] = []
    
    def grid_search(
        self,
        prices: List[float],
        timestamps: List[str],
        param_ranges: Dict,
        position_size: float = 1.0,
    ) -> List[BacktestResult]:
        """
        Grid search for optimal parameters.
        
        Args:
            prices: Historical prices
            timestamps: Timestamps for prices
            param_ranges: Dict of param_name -> [list of values to test]
                Example: {
                    "ma_short_period": [5, 10, 15],
                    "ma_long_period": [30, 50, 70],
                    "rsi_threshold": [25, 30, 35]
                }
            position_size: Position size per trade
            
        Returns:
            List of BacktestResult sorted by return (best first)
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        combinations = itertools.product(*param_values)
        total_combos = len(param_ranges[param_names[0]]) ** len(param_names)
        
        logger.info(f"Grid search: Testing {total_combos} parameter combinations...")
        
        self.results = []
        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))
            
            # Run backtest
            result = self.backtester.backtest_prices(
                prices=prices,
                timestamps=timestamps,
                parameters=params,
                position_size=position_size,
            )
            self.results.append(result)
            
            if (i + 1) % max(1, total_combos // 10) == 0:
                logger.info(f"  Completed {i+1}/{total_combos}")
        
        # Sort by total return (best first)
        self.results.sort(key=lambda r: r.total_return_pct, reverse=True)
        
        logger.info(f"Grid search complete. Best return: {self.results[0].total_return_pct:.2f}%")
        
        return self.results
    
    def get_top_results(self, n: int = 10) -> List[Dict]:
        """Get top N results as dictionaries."""
        return [r.summary_dict() for r in self.results[:n]]
    
    def save_results(self, filename: str):
        """Save optimization results to JSON file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "best_result": self.results[0].summary_dict() if self.results else None,
            "top_10_results": self.get_top_results(10),
            "all_results": [r.summary_dict() for r in self.results],
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
