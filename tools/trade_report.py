#!/usr/bin/env python3
"""
Summarize trade logs written by the Deriv bot.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def load_trades(path: Path) -> List[Dict]:
    trades = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trades.append(json.loads(line))
    return trades


def summarize(trades: List[Dict]) -> Dict:
    if not trades:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate_pct": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
        }

    pnls = [t.get("pnl", 0.0) or 0.0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0

    return {
        "total_trades": len(pnls),
        "total_pnl": total_pnl,
        "win_rate_pct": win_rate,
        "average_win": sum(wins) / len(wins) if wins else 0.0,
        "average_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def monthly_breakdown(trades: List[Dict]) -> Dict[str, Dict]:
    buckets = defaultdict(list)
    for trade in trades:
        ts = trade.get("exit_timestamp") or trade.get("entry_timestamp")
        if not ts:
            continue
        month = ts[:7]
        buckets[month].append(trade)

    return {month: summarize(items) for month, items in sorted(buckets.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Trade report from Deriv bot logs")
    parser.add_argument("--log", default="trades.json", help="Path to trade log file (JSONL)")
    default_output = f"reports/trade_report_{datetime.now(timezone.utc).date().isoformat()}.json"
    parser.add_argument("--output", default=default_output, help="Output report file")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 1

    trades = load_trades(log_path)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_path),
        "summary": summarize(trades),
        "monthly": monthly_breakdown(trades),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
