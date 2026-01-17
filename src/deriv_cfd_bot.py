"""
Deriv CFD-style trading bot for synthetic indices (e.g., Volatility 100).
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import Settings
from .deriv_api import DerivWebSocket
from .logger import print_error, print_header, print_success, setup_logging
from .price_history_collector import MultiMarketPriceCollector
from .risk_manager import RiskLimits, RiskManager
from .trend_detector import is_trending
from .trend_follow_strategy import TrendFollowingPositionManager, run_trend_strategy
from .mean_reversion_strategy import MeanReversionPositionManager, run_mean_reversion_strategy
from .utils import GracefulShutdown
from .config_validator import ConfigValidator

logger = logging.getLogger(__name__)


class DerivCFDBot:
    """CFD-style bot using Deriv multipliers (open/close positions)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ws = DerivWebSocket(
            app_id=settings.deriv_app_id,
            token=settings.deriv_demo_token,
            ws_url=settings.deriv_ws_url,
        )
        self.symbol = settings.deriv_symbol
        self.granularity = settings.deriv_granularity
        self.currency = settings.deriv_currency
        self.multiplier = settings.deriv_multiplier
        self.stake = settings.deriv_stake
        self.long_contract_type = settings.deriv_long_contract_type
        self.short_contract_type = settings.deriv_short_contract_type
        self.trend_confirm_candles = settings.deriv_trend_confirm_candles
        self.strategy_switch_cooldown = settings.deriv_strategy_switch_cooldown
        self.trend_short_ma = settings.deriv_trend_short_ma
        self.trend_long_ma = settings.deriv_trend_long_ma
        self.trend_adx_threshold = settings.deriv_trend_adx_threshold
        self.bb_period = settings.deriv_bb_period
        self.bb_std_dev = settings.deriv_bb_std_dev
        self.rsi_period = settings.deriv_rsi_period
        self.rsi_threshold = settings.deriv_rsi_threshold

        self.price_history = []
        self.max_price_history = 200
        self.last_candle_epoch = None
        self._history_loaded = False
        self._shutdown_requested = False
        self._shutdown_event = asyncio.Event()
        self.current_balance = None
        self.sim_balance = settings.sim_balance if settings.sim_balance > 0 else 10000.0

        self.price_collector = MultiMarketPriceCollector(data_dir="price_history")

        self.trend_follow_pm = TrendFollowingPositionManager()
        self.mean_reversion_pm = MeanReversionPositionManager()
        self.current_strategy = None
        self.enable_strategy_switching = True

        self.open_contract_id = None
        self.open_direction = None
        self.open_trade = None
        self._last_execution_ts = 0.0
        self._last_strategy_switch_ts = 0.0
        self._trend_state = None
        self._trend_streak = 0

        self.risk_manager = None
        if settings.max_daily_loss > 0 or settings.max_position_size > 0 or settings.max_trades_per_day > 0:
            risk_limits = RiskLimits(
                max_daily_loss=settings.max_daily_loss if settings.max_daily_loss > 0 else None,
                max_position_size=settings.max_position_size if settings.max_position_size > 0 else None,
                max_trades_per_day=settings.max_trades_per_day if settings.max_trades_per_day > 0 else None,
                min_balance_required=settings.min_balance_required,
                max_balance_utilization=settings.max_balance_utilization,
                max_risk_per_trade=settings.max_risk_per_trade,
                min_position_size=settings.min_position_size,
                atr_period=settings.atr_period,
                sl_atr_multiplier=settings.sl_atr_multiplier,
                tp_atr_multiplier=settings.tp_atr_multiplier,
                daily_loss_limit=settings.daily_loss_limit if settings.daily_loss_limit > 0 else None,
            )
            self.risk_manager = RiskManager(risk_limits)

        self.trade_log_file = settings.trade_log_file

    async def connect(self) -> None:
        await self.ws.connect()
        if self.settings.deriv_demo_token:
            auth = await self.ws.authorize()
            if auth.get("error"):
                raise RuntimeError(auth["error"].get("message", "Deriv authorization failed"))

            # Subscribe to balance updates
            await self.ws.request({"balance": 1, "subscribe": 1})

        # Subscribe to candles
        await self.ws.request({
            "ticks_history": self.symbol,
            "adjust_start_time": 1,
            "count": 200,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": self.granularity,
            "subscribe": 1,
        })

    def _record_candle(self, candle: Dict) -> Tuple[bool, bool, Optional[float]]:
        try:
            epoch = int(candle.get("epoch") or candle.get("open_time") or 0)
            open_p = float(candle.get("open"))
            high_p = float(candle.get("high"))
            low_p = float(candle.get("low"))
            close_p = float(candle.get("close"))
        except Exception:
            return False, False, None

        is_closed = bool(candle.get("is_closed", False))
        new_candle = False

        if self.last_candle_epoch == epoch:
            if self.price_history:
                self.price_history[-1] = close_p
        else:
            self.last_candle_epoch = epoch
            self.price_history.append(close_p)
            new_candle = True
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]

        try:
            timestamp = datetime.utcfromtimestamp(epoch).isoformat() + "Z"
            self.price_collector.add_price(
                market_slug=self.symbol,
                timestamp=timestamp,
                open_price=open_p,
                close_price=close_p,
                high_price=high_p,
                low_price=low_p,
            )
        except Exception as exc:
            logger.debug(f"Failed to collect candle: {exc}")

        return new_candle, is_closed, close_p

    async def handle_message(self, msg: Dict) -> None:
        if self._shutdown_requested:
            return
        msg_type = msg.get("msg_type")
        if msg_type == "balance":
            balance = msg.get("balance", {}).get("balance")
            if balance is not None:
                self.current_balance = float(balance)
            return

        if msg_type == "candles":
            candles = msg.get("candles", []) or []
            if not self._history_loaded and len(candles) > 1:
                for candle in candles:
                    self._record_candle(candle)
                self._history_loaded = True
                return
            self._history_loaded = True
            for candle in candles:
                new_candle, is_closed, _ = self._record_candle(candle)
                if is_closed:
                    await self.evaluate_strategies()
            return

        if msg_type == "ohlc":
            candle = msg.get("ohlc", {})
            self._history_loaded = True
            new_candle, is_closed, _ = self._record_candle(candle)
            if is_closed:
                await self.evaluate_strategies()
            return

    async def evaluate_strategies(self) -> Optional[Dict]:
        if self._shutdown_requested:
            return None

        if not self.enable_strategy_switching or len(self.price_history) < 50:
            return None

        now = asyncio.get_event_loop().time()
        if self.settings.cooldown_seconds and (now - self._last_execution_ts) < float(self.settings.cooldown_seconds):
            return None

        current_balance = self.current_balance or self.sim_balance
        trending = is_trending(self.price_history)
        trending = is_trending(
            self.price_history,
            short_ma_period=self.trend_short_ma,
            long_ma_period=self.trend_long_ma,
            adx_threshold=self.trend_adx_threshold,
        )
        if self._trend_state is None:
            self._trend_state = trending
            self._trend_streak = 1
        elif trending == self._trend_state:
            self._trend_streak += 1
        else:
            self._trend_state = trending
            self._trend_streak = 1

        if self._trend_streak < self.trend_confirm_candles and self.current_strategy:
            return None

        desired_strategy = "trend" if self._trend_state else "mean_reversion"
        if self.current_strategy and desired_strategy != self.current_strategy:
            if (now - self._last_strategy_switch_ts) < float(self.strategy_switch_cooldown):
                return None
            self._last_strategy_switch_ts = now

        if desired_strategy == "trend":
            if self.current_strategy != "trend":
                logger.info("📈 Switching to TREND-FOLLOWING strategy")
                self.current_strategy = "trend"
            trade_result = run_trend_strategy(
                prices=self.price_history,
                position_manager=self.trend_follow_pm,
                risk_manager=self.risk_manager,
                current_balance=current_balance,
                short_period=self.trend_short_ma,
                long_period=self.trend_long_ma,
                order_size=float(self.stake),
            )
        else:
            if self.current_strategy != "mean_reversion":
                logger.info("📉 Switching to MEAN-REVERSION strategy")
                self.current_strategy = "mean_reversion"
            trade_result = run_mean_reversion_strategy(
                prices=self.price_history,
                position_manager=self.mean_reversion_pm,
                risk_manager=self.risk_manager,
                current_balance=current_balance,
                bb_period=self.bb_period,
                bb_std=self.bb_std_dev,
                rsi_period=self.rsi_period,
                oversold_threshold=self.rsi_threshold,
                order_size=float(self.stake),
            )

        if trade_result:
            self._last_execution_ts = now
            await self.execute_trade(trade_result)
        return trade_result

    async def execute_trade(self, trade_result: Dict) -> None:
        if self._shutdown_requested:
            return
        action = trade_result.get("action")
        if action == "entry":
            if self.open_contract_id:
                return
            await self.open_position(direction="long", trade_result=trade_result)
        elif action == "exit":
            if not self.open_contract_id:
                return
            await self.close_position()

    async def open_position(self, *, direction: str, trade_result: Dict) -> None:
        if self._shutdown_requested:
            return
        if self.settings.dry_run:
            self.open_contract_id = "SIMULATED"
            self.open_direction = direction
            self.open_trade = {
                "symbol": self.symbol,
                "direction": direction,
                "entry_price": trade_result.get("price"),
                "entry_timestamp": datetime.utcnow().isoformat() + "Z",
            }
            logger.info(f"🟢 DRY-RUN: Open {direction.upper()} position @ {trade_result.get('price')}")
            return

        contract_type = self.long_contract_type if direction == "long" else self.short_contract_type
        payload = {
            "buy": 1,
            "price": float(self.stake),
            "parameters": {
                "amount": float(self.stake),
                "basis": "stake",
                "contract_type": contract_type,
                "currency": self.currency,
                "symbol": self.symbol,
                "multiplier": self.multiplier,
            },
        }
        response = await self.ws.request(payload)
        if response.get("error"):
            logger.error(f"Deriv buy error: {response['error'].get('message')}")
            return

        buy = response.get("buy", {})
        self.open_contract_id = buy.get("contract_id")
        self.open_direction = direction
        self.open_trade = {
            "symbol": self.symbol,
            "direction": direction,
            "entry_price": trade_result.get("price"),
            "entry_timestamp": datetime.utcnow().isoformat() + "Z",
            "contract_id": self.open_contract_id,
        }
        logger.info(f"🟢 Opened {direction.upper()} position (contract_id={self.open_contract_id})")

    async def close_position(self) -> None:
        if self._shutdown_requested:
            return
        if self.settings.dry_run:
            logger.info("🔴 DRY-RUN: Close position")
            self._record_trade_exit()
            self.open_contract_id = None
            self.open_direction = None
            self.open_trade = None
            return

        if not self.open_contract_id:
            return
        response = await self.ws.request({"sell": self.open_contract_id, "price": 0})
        if response.get("error"):
            logger.error(f"Deriv sell error: {response['error'].get('message')}")
            return
        logger.info(f"🔴 Closed position (contract_id={self.open_contract_id})")
        self._record_trade_exit()
        self.open_contract_id = None
        self.open_direction = None
        self.open_trade = None

    def _record_trade_exit(self) -> None:
        if not self.open_trade or not self.trade_log_file:
            return
        exit_price = self.price_history[-1] if self.price_history else None
        entry_price = self.open_trade.get("entry_price")
        pnl = None
        pnl_pct = None
        if entry_price and exit_price:
            pnl = (exit_price - entry_price) * float(self.stake)
            pnl_pct = ((exit_price / entry_price) - 1) * 100

        record = {
            **self.open_trade,
            "exit_price": exit_price,
            "exit_timestamp": datetime.utcnow().isoformat() + "Z",
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "stake": float(self.stake),
        }

        try:
            with open(self.trade_log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write trade log: {exc}")

    async def run(self) -> None:
        try:
            await self.connect()
        except Exception as exc:
            logger.error(f"Failed to connect to Deriv WebSocket: {exc}")
            return

        try:
            while not self._shutdown_requested:
                msg_task = asyncio.create_task(self.ws.next_message())
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                done, pending = await asyncio.wait(
                    {msg_task, shutdown_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if shutdown_task in done:
                    break
                msg = msg_task.result()
                if msg.get("msg_type") == "_ws_error":
                    logger.warning("WebSocket disconnected; reconnecting...")
                    await self.ws.close()
                    await asyncio.sleep(1.0)
                    try:
                        await self.connect()
                    except Exception as exc:
                        logger.error(f"Reconnect failed: {exc}")
                        await asyncio.sleep(2.0)
                    continue
                await self.handle_message(msg)
        finally:
            await self.ws.close()

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        self._shutdown_event.set()


async def main() -> None:
    shutdown_handler = GracefulShutdown()
    settings = Settings()
    setup_logging(verbose=settings.verbose, use_rich=settings.use_rich_output)

    if not ConfigValidator.validate_and_print(settings):
        print_error("Configuration validation failed. Please fix the errors and try again.")
        return

    print_header("🚀 Deriv V100 CFD Bot (Demo)")
    print_success("Configuration loaded and validated")

    bot = DerivCFDBot(settings)

    def on_shutdown():
        logger.info("Shutting down Deriv bot...")
        bot.request_shutdown()

    shutdown_handler.register_callback(on_shutdown)
    run_task = asyncio.create_task(bot.run())
    try:
        while not run_task.done():
            if shutdown_handler.is_force_exit_requested():
                run_task.cancel()
                break
            await asyncio.sleep(0.2)
    finally:
        if not run_task.done():
            run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
