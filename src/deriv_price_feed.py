"""
Deriv historical candle fetcher for backtesting.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Tuple

from .deriv_api import DerivWebSocket

logger = logging.getLogger(__name__)


_INTERVAL_TO_GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "1d": 86400,
}


def _parse_datetime(value: str) -> int:
    """Parse ISO date/datetime string to UTC epoch seconds."""
    if "T" in value:
        dt = datetime.fromisoformat(value)
    else:
        dt = datetime.fromisoformat(f"{value}T00:00:00")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


async def _fetch_candles(
    *,
    symbol: str,
    start_epoch: int,
    end_epoch: int,
    granularity: int,
    app_id: str,
    ws_url: str,
) -> List[dict]:
    ws = DerivWebSocket(app_id=app_id, ws_url=ws_url)
    await ws.connect()
    try:
        all_candles: List[dict] = []
        current_start = start_epoch
        step_seconds = 30 * 86400  # 30 days

        while current_start < end_epoch:
            current_end = min(end_epoch, current_start + step_seconds)
            payload = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "start": current_start,
                "end": current_end,
                "style": "candles",
                "granularity": granularity,
            }
            resp = await ws.request(payload, timeout=20.0)
            if resp.get("error"):
                error = resp["error"]
                message = error.get("message", "Deriv API error")
                code = error.get("code")
                raise RuntimeError(f"Deriv API error ({code}): {message}")

            candles = resp.get("candles", []) or []
            all_candles.extend(candles)
            current_start = current_end + 1

        all_candles.sort(key=lambda c: int(c.get("epoch", 0)))
        return all_candles
    finally:
        await ws.close()


def fetch_historical_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    *,
    app_id: str,
    ws_url: str,
) -> Tuple[List[float], List[str]]:
    """
    Fetch historical candle closes from Deriv for backtesting.
    """
    granularity = _INTERVAL_TO_GRANULARITY.get(interval, 900)
    start_epoch = _parse_datetime(start_date)
    end_epoch = _parse_datetime(end_date)
    now_epoch = int(datetime.now(tz=timezone.utc).timestamp())
    if end_epoch > now_epoch:
        end_epoch = now_epoch
    if end_epoch <= start_epoch:
        raise ValueError("end_date must be after start_date")

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running and running.is_running():
        raise RuntimeError("fetch_historical_prices cannot be called from a running event loop")

    candles = asyncio.run(
        _fetch_candles(
            symbol=symbol,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            granularity=granularity,
            app_id=str(app_id),
            ws_url=ws_url,
        )
    )

    prices: List[float] = []
    timestamps: List[str] = []
    for candle in candles:
        epoch = int(candle.get("epoch", 0))
        close_price = float(candle.get("close"))
        ts = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        prices.append(close_price)
        timestamps.append(ts)

    return prices, timestamps
