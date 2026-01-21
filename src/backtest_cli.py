#!/usr/bin/env python3
"""
Enhanced backtesting CLI - Run strategy backtests with parameter optimization.

Usage:
    # Backtest trend-following strategy
    python src/backtest_cli.py --strategy trend --csv prices.csv
    
    # Backtest mean-reversion strategy
    python src/backtest_cli.py --strategy reversion --csv prices.csv
    
    # Grid search optimization
    python src/backtest_cli.py --strategy trend --csv prices.csv --grid-search \
      --params '{"short_ma_period": [8,10,12], "long_ma_period": [40,50,60]}'
    
    # Custom parameters
    python src/backtest_cli.py --strategy trend --csv prices.csv \
      --params '{"short_ma_period": 12, "long_ma_period": 55}'
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester import Backtester, ParameterOptimizer, load_price_data, evaluate_performance
from src.trend_follow_strategy import run_strategy as trend_strategy
from src.mean_reversion_strategy import run_strategy as reversion_strategy
from src.ema_scalper_strategy import run_strategy as ema_scalper_strategy
from src.trend_detector import is_trending

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Default parameters for each strategy
DEFAULT_PARAMS = {
    "trend": {
        "short_ma_period": 10,
        "long_ma_period": 50,
    },
    "reversion": {
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "rsi_threshold": 30,
        "rsi_period": 14,
    },
    "switch": {
        "short_ma_period": 10,
        "long_ma_period": 50,
        "adx_threshold": 25.0,
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "rsi_threshold": 30,
        "rsi_period": 14,
    },
    "ema_scalper": {
        "ema_short": 12,
        "ema_long": 26,
        "rsi_period": 14,
        "rsi_min": 30,
        "rsi_max": 70,
    },
}

# Grid search ranges for optimization
GRID_RANGES = {
    "trend": {
        "short_ma_period": [5, 8, 10, 12, 15],
        "long_ma_period": [30, 40, 50, 60, 70],
    },
    "reversion": {
        "bb_period": [15, 20, 25],
        "bb_std_dev": [1.5, 2.0, 2.5],
        "rsi_threshold": [25, 30, 35],
    },
    "switch": {
        "short_ma_period": [5, 8, 10, 12, 15],
        "long_ma_period": [30, 40, 50, 60, 70],
        "bb_period": [15, 20, 25],
        "bb_std_dev": [1.5, 2.0, 2.5],
        "rsi_threshold": [25, 30, 35],
        "adx_threshold": [25.0],
    },
    "ema_scalper": {
        "ema_short": [8, 12, 16],
        "ema_long": [20, 26, 32],
        "rsi_min": [30, 35],
        "rsi_max": [65, 70],
    },
}


def run_strategy_backtest(
    strategy_name: str,
    prices: List[float],
    timestamps: List[str],
    params: Dict,
    initial_balance: float = 10000.0,
    position_size: float = 1.0,
) -> Dict:
    """
    Run a single backtest with specified strategy and parameters.
    
    Args:
        strategy_name: "trend" or "reversion"
        prices: List of prices
        timestamps: List of timestamps
        params: Strategy parameters
        initial_balance: Starting balance
        
    Returns:
        Dictionary with backtest results
    """
    if strategy_name == "trend":
        strategy_fn = trend_strategy
    elif strategy_name == "reversion":
        strategy_fn = reversion_strategy
    else:
        strategy_fn = ema_scalper_strategy
    
    trades = []
    balance = initial_balance
    state = {}  # Track open positions
    
    for i in range(len(prices)):
        signal, update = strategy_fn(prices, timestamps, i, state, params)
        
        if signal == "enter":
            state["in_position"] = True
            state["entry_price"] = update.get("entry_price", prices[i])
            state["entry_index"] = i
            state["entry_timestamp"] = update.get("timestamp", timestamps[i])
            
        elif signal == "exit" and state.get("in_position"):
            state["in_position"] = False
            profit = update.get("profit", 0) * position_size
            balance += profit
            
            # Record trade
            from src.backtester import BacktestTrade
            trade = BacktestTrade(
                timestamp=state.get("entry_timestamp", timestamps[state.get("entry_index", 0)]),
                signal_type=update.get("signal_type", strategy_name),
                entry_price=state.get("entry_price", prices[i]),
                exit_price=update.get("exit_price", prices[i]),
                profit=profit,
                profit_pct=update.get("profit_pct", 0),
                win=profit > 0,
            )
            trades.append(trade)
    
    # Evaluate performance
    perf = evaluate_performance(trades, initial_balance)
    
    return {
        "strategy": strategy_name,
        "params": params,
        "initial_balance": initial_balance,
        "final_balance": balance,
        "performance": perf,
        "trades": trades,
    }


def run_switch_backtest(
    prices: List[float],
    timestamps: List[str],
    params: Dict,
    initial_balance: float = 10000.0,
    position_size: float = 1.0,
) -> Dict:
    """
    Backtest strategy switching: trend-follow when trending, mean-reversion otherwise.
    """
    trades = []
    balance = initial_balance
    state = {"in_position": False, "active_strategy": None}

    for i in range(len(prices)):
        if i < 60:
            continue
        price_window = prices[:i + 1]
        trending = is_trending(
            price_window,
            short_ma_period=params.get("short_ma_period", 10),
            long_ma_period=params.get("long_ma_period", 50),
            adx_threshold=params.get("adx_threshold", 25.0),
        )

        if not state.get("in_position"):
            if trending:
                signal, update = trend_strategy(prices, timestamps, i, state, params)
                if signal == "enter":
                    state["in_position"] = True
                    state["active_strategy"] = "trend"
                    state["entry_price"] = update.get("entry_price", prices[i])
                    state["entry_index"] = i
                    state["entry_timestamp"] = update.get("timestamp", timestamps[i])
            else:
                signal, update = reversion_strategy(prices, timestamps, i, state, params)
                if signal == "enter":
                    state["in_position"] = True
                    state["active_strategy"] = "reversion"
                    state["entry_price"] = update.get("entry_price", prices[i])
                    state["entry_index"] = i
                    state["entry_timestamp"] = update.get("timestamp", timestamps[i])
        else:
            active = state.get("active_strategy")
            strategy_fn = trend_strategy if active == "trend" else reversion_strategy
            signal, update = strategy_fn(prices, timestamps, i, state, params)
            if signal == "exit":
                state["in_position"] = False
                profit = update.get("profit", 0) * position_size
                balance += profit

                from src.backtester import BacktestTrade
                trade = BacktestTrade(
                    timestamp=state.get("entry_timestamp", timestamps[state.get("entry_index", 0)]),
                    signal_type=update.get("signal_type", active or "switch"),
                    entry_price=state.get("entry_price", prices[i]),
                    exit_price=update.get("exit_price", prices[i]),
                    profit=profit,
                    profit_pct=update.get("profit_pct", 0),
                    win=profit > 0,
                )
                trades.append(trade)

                state["active_strategy"] = None

    perf = evaluate_performance(trades, initial_balance)
    return {
        "strategy": "switch",
        "params": params,
        "initial_balance": initial_balance,
        "final_balance": balance,
        "performance": perf,
        "trades": trades,
    }


def grid_search_optimization(
    strategy_name: str,
    prices: List[float],
    timestamps: List[str],
    param_ranges: Dict,
    initial_balance: float = 10000.0,
    position_size: float = 1.0,
) -> List[Dict]:
    """
    Run grid search optimization over parameter ranges.
    
    Args:
        strategy_name: "trend" or "reversion"
        prices: List of prices
        timestamps: List of timestamps
        param_ranges: Dict mapping param names to lists of values
        initial_balance: Starting balance
        
    Returns:
        List of results sorted by total return (best first)
    """
    import itertools
    
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    
    combinations = list(itertools.product(*param_values))
    total = len(combinations)
    
    logger.info(f"Grid search: Testing {total} parameter combinations...")
    
    results = []
    for i, values in enumerate(combinations):
        params = dict(zip(param_names, values))

        if strategy_name == "switch":
            result = run_switch_backtest(
                prices=prices,
                timestamps=timestamps,
                params=params,
                initial_balance=initial_balance,
                position_size=position_size,
            )
        else:
            result = run_strategy_backtest(
                strategy_name=strategy_name,
                prices=prices,
                timestamps=timestamps,
                params=params,
                initial_balance=initial_balance,
                position_size=position_size,
            )
        results.append(result)
        
        if (i + 1) % max(1, total // 10) == 0:
            logger.info(f"  Completed {i+1}/{total}")
    
    # Sort by total return
    results.sort(key=lambda r: r["performance"]["total_return_pct"], reverse=True)
    
    logger.info(f"Grid search complete. Best return: {results[0]['performance']['total_return_pct']:.2f}%")
    
    return results


def print_backtest_results(result: Dict):
    """Pretty-print backtest results."""
    perf = result["performance"]
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"\nStrategy: {result['strategy'].upper()}")
    print(f"Initial Balance: ${result['initial_balance']:.2f}")
    print(f"Final Balance: ${result['final_balance']:.2f}")
    
    print(f"\n{'Metric':<40} {'Value':>25}")
    print("-" * 66)
    print(f"{'Total Return':<40} {perf['total_return_pct']:>24.2f}%")
    print(f"{'Win Rate':<40} {perf['win_rate_pct']:>24.1f}%")
    print(f"{'Profit Factor':<40} {perf['profit_factor']:>24.2f}")
    print(f"{'Sharpe Ratio':<40} {perf['sharpe_ratio']:>24.2f}")
    print(f"{'Max Drawdown':<40} {perf['max_drawdown_pct']:>24.2f}%")
    print(f"{'Total Trades':<40} {perf['total_trades']:>24.0f}")
    print(f"{'Winning Trades':<40} {perf['winning_trades']:>24.0f}")
    print(f"{'Losing Trades':<40} {perf['losing_trades']:>24.0f}")
    print(f"{'Average Win':<40} ${perf['average_win']:>23.2f}")
    print(f"{'Average Loss':<40} ${perf['average_loss']:>23.2f}")
    print(f"{'Largest Win':<40} ${perf['largest_win']:>23.2f}")
    print(f"{'Largest Loss':<40} ${perf['largest_loss']:>23.2f}")
    
    print(f"\n{'Parameters':-^70}")
    for key, value in result["params"].items():
        print(f"  {key:<36} {str(value):>30}")
    
    if result["trades"]:
        print(f"\n{'Trade History (first 20)':-^70}")
        print(f"{'Timestamp':<20} {'Type':<15} {'Entry':>12} {'Exit':>12} {'P&L %':>10}")
        for trade in result["trades"][:20]:
            entry = f"${trade.entry_price:.4f}"
            exit_p = f"${trade.exit_price:.4f}" if trade.exit_price else "N/A"
            print(f"{trade.timestamp[:10]:<20} {trade.signal_type:<15} {entry:>12} {exit_p:>12} {trade.profit_pct:>9.2f}%")
        
        if len(result["trades"]) > 20:
            print(f"... and {len(result['trades']) - 20} more trades")
    else:
        print("\nNo trades generated with these parameters.")
    
    print("="*70 + "\n")


def print_grid_search_results(results: List[Dict], top_n: int = 10):
    """Pretty-print grid search results."""
    print("\n" + "="*70)
    print(f"GRID SEARCH RESULTS - TOP {min(top_n, len(results))}")
    print("="*70)
    
    for rank, result in enumerate(results[:top_n], 1):
        perf = result["performance"]
        print(f"\n#{rank}")
        print(f"  Return: {perf['total_return_pct']:>8.2f}% | "
              f"WinRate: {perf['win_rate_pct']:>6.1f}% | "
              f"ProfitFactor: {perf['profit_factor']:>5.2f} | "
              f"Sharpe: {perf['sharpe_ratio']:>6.2f}")
        print(f"  Params: {result['params']}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced backtesting CLI for strategy optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest from CSV file
  python src/backtest_cli.py --strategy trend --csv prices.csv
  
  # Fetch data from external API and backtest
  python src/backtest_cli.py --strategy trend --symbol BTC --start 2024-01-01 --end 2024-01-31
  
  # Grid search optimization with external data
  python src/backtest_cli.py --strategy trend --symbol BTC --start 2024-01-01 --end 2024-01-31 --grid-search
  
  # Custom parameters
  python src/backtest_cli.py --strategy trend --csv prices.csv --params '{"short_ma_period": 12}'
        """
    )
    
    # Price data input (CSV or external API)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with price data"
    )
    
    data_group.add_argument(
        "--symbol",
        type=str,
        help="Crypto symbol (BTC, ETH, MATIC, etc.) to fetch from external API"
    )
    
    # Date range (required if using --symbol)
    parser.add_argument(
        "--start",
        type=str,
        help="Start date in ISO format (YYYY-MM-DD) - required with --symbol"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date in ISO format (YYYY-MM-DD) - required with --symbol"
    )
    
    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["trend", "reversion", "switch", "ema_scalper"],
        default="trend",
        help="Strategy to backtest (default: trend)"
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="Candle interval (1m, 5m, 15m, 1h, 1d) - default: 15m"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Data provider (coingecko, coinmarketcap, polygon, kraken, binance, deriv)"
    )

    parser.add_argument(
        "--deriv",
        action="store_true",
        help="Shortcut for --provider deriv"
    )
    
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Starting balance (default: 10000)"
    )

    parser.add_argument(
        "--position-size",
        type=float,
        default=1.0,
        help="Position size multiplier for P&L (default: 1.0)"
    )
    
    parser.add_argument(
        "--params",
        type=str,
        help="JSON with strategy parameters (e.g. '{\"short_ma_period\": 12}')"
    )
    
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search optimization"
    )
    
    parser.add_argument(
        "--grid-params",
        type=str,
        help="JSON with parameter ranges for grid search"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load price data - either from CSV or external API
    logger.info("Loading price data...")
    try:
        if args.csv:
            logger.info(f"Loading from CSV: {args.csv}")
            prices, timestamps = load_price_data(filepath=args.csv)
        elif args.symbol:
            if not args.start or not args.end:
                logger.error("--start and --end are required when using --symbol")
                return 1

            logger.info(f"Fetching {args.symbol} from external API ({args.start} to {args.end})")

            # Load settings with provider override if specified
            from src.config import load_settings
            settings = load_settings()
            if args.deriv:
                settings.data_provider_name = "deriv"
            elif args.provider:
                settings.data_provider_name = args.provider
                logger.info(f"Using provider: {args.provider}")

            prices, timestamps = load_price_data(
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                settings=settings
            )
        else:
            logger.error("Either --csv or --symbol must be provided")
            return 1
    except Exception as e:
        logger.error(f"Failed to load prices: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if not prices:
        logger.error("No prices loaded from CSV")
        return 1
    
    logger.info(f"Loaded {len(prices)} price bars")
    
    if args.grid_search:
        # Grid search mode
        if args.grid_params:
            param_ranges = json.loads(args.grid_params)
        else:
            param_ranges = GRID_RANGES[args.strategy]
        
        logger.info(f"Grid search parameter ranges: {param_ranges}")
        
        results = grid_search_optimization(
            strategy_name=args.strategy,
            prices=prices,
            timestamps=timestamps,
            param_ranges=param_ranges,
            initial_balance=args.initial_balance,
            position_size=args.position_size,
        )
        
        print_grid_search_results(results, top_n=10)
        
        if args.output:
            output_data = {
                "strategy": args.strategy,
                "num_prices": len(prices),
                "total_tests": len(results),
                "best_result": {
                    "params": results[0]["params"],
                    "performance": results[0]["performance"],
                },
                "top_10": [
                    {
                        "params": r["params"],
                        "performance": r["performance"],
                    }
                    for r in results[:10]
                ],
            }
            
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    else:
        # Single backtest mode
        if args.params:
            params = json.loads(args.params)
        else:
            params = DEFAULT_PARAMS[args.strategy]
        
        logger.info(f"Running {args.strategy} backtest with params: {params}")
        
        if args.strategy == "switch":
            result = run_switch_backtest(
                prices=prices,
                timestamps=timestamps,
                params=params,
                initial_balance=args.initial_balance,
                position_size=args.position_size,
            )
        elif args.strategy == "ema_scalper":
            result = run_strategy_backtest(
                strategy_name=args.strategy,
                prices=prices,
                timestamps=timestamps,
                params=params,
                initial_balance=args.initial_balance,
                position_size=args.position_size,
            )
        else:
            result = run_strategy_backtest(
                strategy_name=args.strategy,
                prices=prices,
                timestamps=timestamps,
                params=params,
                initial_balance=args.initial_balance,
                position_size=args.position_size,
            )
        
        print_backtest_results(result)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "strategy": result["strategy"],
                    "params": result["params"],
                    "performance": result["performance"],
                }, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
