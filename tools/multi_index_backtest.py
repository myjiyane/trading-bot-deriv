#!/usr/bin/env python3
"""
Multi-index backtest runner for Deriv symbols.

Aggregates trades across multiple symbols and summarizes overall performance.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtester import load_price_data, evaluate_performance
from src.backtest_cli import run_strategy_backtest, run_switch_backtest, GRID_RANGES
from src.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_symbols(raw: str) -> List[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def aggregate_results(results: List[Dict], initial_balance_total: float) -> Dict:
    trades = []
    total_return = 0.0
    for result in results:
        trades.extend(result["trades"])
        total_return += result["performance"]["total_return"]

    perf = evaluate_performance(trades, initial_balance_total)
    return {
        "total_return": total_return,
        "total_return_pct": (total_return / initial_balance_total * 100) if initial_balance_total else 0.0,
        "performance": perf,
        "total_trades": perf.get("total_trades", 0),
    }

def default_params_for(settings, strategy: str) -> Dict:
    if strategy == "trend":
        return {
            "short_ma_period": settings.deriv_trend_short_ma,
            "long_ma_period": settings.deriv_trend_long_ma,
        }
    if strategy == "reversion":
        return {
            "bb_period": settings.deriv_bb_period,
            "bb_std_dev": settings.deriv_bb_std_dev,
            "rsi_threshold": settings.deriv_rsi_threshold,
            "rsi_period": settings.deriv_rsi_period,
        }
    return {
        "short_ma_period": settings.deriv_trend_short_ma,
        "long_ma_period": settings.deriv_trend_long_ma,
        "adx_threshold": settings.deriv_trend_adx_threshold,
        "bb_period": settings.deriv_bb_period,
        "bb_std_dev": settings.deriv_bb_std_dev,
        "rsi_threshold": settings.deriv_rsi_threshold,
        "rsi_period": settings.deriv_rsi_period,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-index backtest runner")
    parser.add_argument("--symbols", required=True, help="Comma-separated Deriv symbols (e.g., R_10,R_25,R_100)")
    parser.add_argument("--strategy", choices=["trend", "reversion", "switch"], default="switch")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="15m", help="Candle interval (1m,5m,15m,1h,1d)")
    parser.add_argument("--deriv", action="store_true", help="Use Deriv price feed")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial balance per symbol")
    parser.add_argument("--position-size", type=float, default=1.0, help="Position size multiplier")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search per symbol")
    parser.add_argument("--grid-params", type=str, default=None, help="JSON with grid parameters")
    parser.add_argument("--filter-positive", action="store_true", help="Filter to symbols with positive returns")
    parser.add_argument("--output", type=str, default="backtest_results_multi.json")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    if not symbols:
        logger.error("No symbols provided")
        return 1

    settings = load_settings()
    if args.deriv:
        settings.data_provider_name = "deriv"
    base_params = default_params_for(settings, args.strategy)

    results = []
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol} ({args.start} to {args.end})")
        try:
            prices, timestamps = load_price_data(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                settings=settings,
            )
        except Exception as exc:
            logger.warning(f"Skipping {symbol} due to data fetch error: {exc}")
            continue

        if args.grid_search:
            param_ranges = json.loads(args.grid_params) if args.grid_params else GRID_RANGES[args.strategy]
            grid_results = []
            for params in _iter_grid(param_ranges):
                if args.strategy == "switch":
                    result = run_switch_backtest(
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
                grid_results.append(result)

            grid_results.sort(key=lambda r: r["performance"]["total_return_pct"], reverse=True)
            best = grid_results[0]
            logger.info(f"{symbol}: best return {best['performance']['total_return_pct']:.2f}% with {best['params']}")
            results.append({"symbol": symbol, "best": best})
        else:
            if args.strategy == "switch":
                result = run_switch_backtest(
                    prices=prices,
                    timestamps=timestamps,
                    params=base_params,
                    initial_balance=args.initial_balance,
                    position_size=args.position_size,
                )
            else:
                result = run_strategy_backtest(
                    strategy_name=args.strategy,
                    prices=prices,
                    timestamps=timestamps,
                    params=base_params,
                    initial_balance=args.initial_balance,
                    position_size=args.position_size,
                )
            result["symbol"] = symbol
            results.append(result)

    if args.grid_search:
        per_symbol = []
        for item in results:
            best = item["best"]
            per_symbol.append({
                "symbol": item["symbol"],
                "params": best["params"],
                "performance": best["performance"],
            })
        output = {
            "strategy": args.strategy,
            "symbols": symbols,
            "per_symbol": per_symbol,
        }
    else:
        initial_balance_total = args.initial_balance * len(symbols)
        combined = aggregate_results(results, initial_balance_total)
        filtered = None
        if args.filter_positive:
            filtered_symbols = [
                r for r in results if r["performance"]["total_return"] > 0
            ]
            if filtered_symbols:
                filtered_balance = args.initial_balance * len(filtered_symbols)
                filtered = {
                    "symbols": [r.get("symbol") for r in filtered_symbols],
                    "initial_balance_total": filtered_balance,
                    "combined": aggregate_results(filtered_symbols, filtered_balance),
                }
        output = {
            "strategy": args.strategy,
            "symbols": symbols,
            "initial_balance_total": initial_balance_total,
            "position_size": args.position_size,
            "per_symbol": [
                {
                    "symbol": r.get("symbol"),
                    "performance": r["performance"],
                }
                for r in results
            ],
            "combined": combined,
            "filtered": filtered,
        }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    return 0


def _iter_grid(param_ranges: Dict) -> List[Dict]:
    import itertools

    names = list(param_ranges.keys())
    values = [param_ranges[name] for name in names]
    for combo in itertools.product(*values):
        yield dict(zip(names, combo))


if __name__ == "__main__":
    raise SystemExit(main())
