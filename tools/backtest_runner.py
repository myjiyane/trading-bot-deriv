#!/usr/bin/env python3
"""
Backtest runner CLI - Test and optimize strategy parameters.

Usage:
    # Backtest with default parameters
    python -m tools.backtest_runner

    # Backtest specific parameters
    python -m tools.backtest_runner --ma-short 10 --ma-long 50 --rsi-threshold 30

    # Grid search optimization
    python -m tools.backtest_runner --grid-search --ma-short "5,10,15" --ma-long "30,50,70"

    # Load prices from file
    python -m tools.backtest_runner --prices prices.json --timestamps timestamps.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester import Backtester, ParameterOptimizer, load_price_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_prices(num_bars: int = 200) -> tuple:
    """Generate sample price data for testing."""
    from datetime import datetime, timedelta
    import random
    
    logger.info(f"Generating {num_bars} sample price bars...")
    
    prices = []
    timestamps = []
    
    # Start with a base price and random walk
    current_price = 0.5050
    
    for i in range(num_bars):
        # Random walk with slight upward bias
        change = (random.random() - 0.45) * 0.01
        current_price = max(0.01, current_price + change)
        prices.append(current_price)
        
        # Generate timestamp (15-minute intervals)
        ts = datetime.now() - timedelta(minutes=(num_bars - i - 1) * 15)
        timestamps.append(ts.isoformat())
    
    return prices, timestamps


def load_prices_from_file(filename: str) -> tuple:
    """Load prices from JSON file using CSV loader."""
    if filename.endswith('.csv'):
        # Use the new CSV loader
        return load_price_data(filename)
    else:
        # JSON fallback
        with open(filename, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                prices = data.get("prices", [])
                timestamps = data.get("timestamps", [])
            else:
                prices = data
                timestamps = [str(i) for i in range(len(prices))]
    
    logger.info(f"Loaded {len(prices)} price bars from {filename}")
    return prices, timestamps


def load_trades_from_json(filename: str = "trades.json") -> tuple:
    """Load price data from trades.json (existing trade history)."""
    if not Path(filename).exists():
        logger.warning(f"{filename} not found, generating sample data")
        return generate_sample_prices()
    
    with open(filename, "r") as f:
        trades = json.load(f)
    
    if not trades:
        logger.warning("trades.json is empty, generating sample data")
        return generate_sample_prices()
    
    # Extract prices from trade records
    prices = []
    timestamps = []
    
    for trade in trades:
        if isinstance(trade, dict):
            if "price_up" in trade and "price_down" in trade:
                # Use midpoint of UP/DOWN prices
                mid = (trade["price_up"] + trade["price_down"]) / 2
                prices.append(mid)
                timestamps.append(trade.get("timestamp", ""))
    
    if prices:
        logger.info(f"Loaded {len(prices)} prices from existing {filename}")
        return prices, timestamps
    
    logger.warning("Could not extract prices from trades.json, generating sample data")
    return generate_sample_prices()


def run_single_backtest(
    prices: List[float],
    timestamps: List[str],
    params: Dict,
    initial_balance: float = 10000.0,
    position_size: float = 1.0,
):
    """Run a single backtest with given parameters."""
    logger.info(f"Running backtest with parameters: {params}")
    
    backtester = Backtester(initial_balance=initial_balance)
    result = backtester.backtest_prices(
        prices=prices,
        timestamps=timestamps,
        parameters=params,
        position_size=position_size,
    )
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    summary = result.summary_dict()
    for key, value in summary.items():
        if key != "params":
            print(f"{key:.<40} {value:>15}")
    
    print(f"\n{'Parameters':-^60}")
    for key, value in result.params.items():
        print(f"  {key:.<40} {value:>15}")
    
    print(f"\n{'Trades':-^60}")
    print(f"{'Timestamp':.<20} {'Type':.<18} {'Entry':>8} {'Exit':>8} {'P&L %':>10}")
    for trade in result.trades[:20]:  # Show first 20 trades
        print(
            f"{trade.timestamp[:10]:.<20} {trade.signal_type:.<18} "
            f"${trade.entry_price:>7.4f} ${trade.exit_price or 0:>7.4f} {trade.profit_pct:>9.2f}%"
        )
    
    if len(result.trades) > 20:
        print(f"... and {len(result.trades) - 20} more trades")
    
    print("="*60 + "\n")
    
    return result


def run_grid_search(
    prices: List[float],
    timestamps: List[str],
    param_ranges: Dict,
    initial_balance: float = 10000.0,
    position_size: float = 1.0,
):
    """Run grid search optimization."""
    logger.info(f"Parameter ranges: {param_ranges}")
    
    backtester = Backtester(initial_balance=initial_balance)
    optimizer = ParameterOptimizer(backtester)
    
    results = optimizer.grid_search(
        prices=prices,
        timestamps=timestamps,
        param_ranges=param_ranges,
        position_size=position_size,
    )
    
    # Print top results
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS - TOP 10")
    print("="*60)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n#{i}")
        summary = result.summary_dict()
        for key, value in summary.items():
            if key != "params":
                print(f"  {key:.<38} {value:>15}")
        
        print(f"  {'Parameters':.-<40}")
        for key, value in result.params.items():
            print(f"    {key:.<36} {value:>15}")
    
    print("="*60 + "\n")
    
    # Save results
    output_file = "backtest_results.json"
    optimizer.save_results(output_file)
    logger.info(f"Detailed results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backtest and optimize trading strategy parameters"
    )
    
    # Data input
    parser.add_argument(
        "--prices",
        type=str,
        default=None,
        help="JSON file with prices (or use trades.json)",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="JSON file with timestamps",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Starting balance for backtest (default: 10000)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1.0,
        help="Position size multiplier (default: 1.0)",
    )
    
    # Single backtest parameters
    parser.add_argument("--ma-short", type=int, default=10, help="Short MA period")
    parser.add_argument("--ma-long", type=int, default=50, help="Long MA period")
    parser.add_argument("--adx-threshold", type=float, default=25.0, help="ADX threshold for trending")
    parser.add_argument("--bb-period", type=int, default=20, help="Bollinger Band period")
    parser.add_argument("--bb-std-dev", type=float, default=2.0, help="Bollinger Band std dev")
    parser.add_argument("--rsi-threshold", type=int, default=30, help="RSI oversold threshold")
    parser.add_argument(
        "--exit-profit-target",
        type=float,
        default=0.02,
        help="Exit profit target (0.02 for 2 percent)",
    )
    parser.add_argument(
        "--exit-loss-stop",
        type=float,
        default=0.01,
        help="Stop loss (0.01 for 1 percent)",
    )
    
    # Grid search
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search optimization",
    )
    parser.add_argument(
        "--ma-short-range",
        type=str,
        help="MA short range for grid search (e.g. 5,10,15)",
    )
    parser.add_argument(
        "--ma-long-range",
        type=str,
        help="MA long range for grid search (e.g. 30,50,70)",
    )
    parser.add_argument(
        "--rsi-threshold-range",
        type=str,
        help="RSI threshold range (e.g. 25,30,35)",
    )
    
    args = parser.parse_args()
    
    # Load prices
    if args.prices:
        prices, timestamps = load_prices_from_file(args.prices)
    else:
        # Try to use existing trades.json if available
        prices, timestamps = load_trades_from_json()
    
    if args.grid_search:
        # Grid search mode
        param_ranges = {}
        
        if args.ma_short_range:
            param_ranges["ma_short_period"] = [int(x) for x in args.ma_short_range.split(",")]
        else:
            param_ranges["ma_short_period"] = [5, 10, 15]
        
        if args.ma_long_range:
            param_ranges["ma_long_period"] = [int(x) for x in args.ma_long_range.split(",")]
        else:
            param_ranges["ma_long_period"] = [30, 50, 70]
        
        if args.rsi_threshold_range:
            param_ranges["rsi_threshold"] = [int(x) for x in args.rsi_threshold_range.split(",")]
        else:
            param_ranges["rsi_threshold"] = [25, 30, 35]
        
        # Always include these
        param_ranges["bb_period"] = [20]
        param_ranges["bb_std_dev"] = [2.0]
        param_ranges["adx_threshold"] = [25.0]
        param_ranges["exit_profit_target"] = [0.02]
        param_ranges["exit_loss_stop"] = [0.01]
        
        run_grid_search(
            prices=prices,
            timestamps=timestamps,
            param_ranges=param_ranges,
            initial_balance=args.initial_balance,
            position_size=args.position_size,
        )
    else:
        # Single backtest mode
        params = {
            "ma_short_period": args.ma_short,
            "ma_long_period": args.ma_long,
            "adx_threshold": args.adx_threshold,
            "bb_period": args.bb_period,
            "bb_std_dev": args.bb_std_dev,
            "rsi_threshold": args.rsi_threshold,
            "exit_profit_target": args.exit_profit_target,
            "exit_loss_stop": args.exit_loss_stop,
        }
        
        run_single_backtest(
            prices=prices,
            timestamps=timestamps,
            params=params,
            initial_balance=args.initial_balance,
            position_size=args.position_size,
        )


if __name__ == "__main__":
    main()
