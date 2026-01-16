import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file from project root if present.
# Do NOT override existing environment variables (so CI/terminal env wins over .env).
load_dotenv(override=False)


@dataclass
class Settings:
    api_key: str = os.getenv("POLYMARKET_API_KEY", "")
    api_secret: str = os.getenv("POLYMARKET_API_SECRET", "")
    api_passphrase: str = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    signature_type: int = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
    funder: str = os.getenv("POLYMARKET_FUNDER", "")
    
    # Market configuration - supports both single market (backward compatible) and multiple markets
    # Single market mode: POLYMARKET_MARKET_SLUG or POLYMARKET_MARKET_SLUGS (one entry)
    # Multi-market mode: POLYMARKET_MARKET_SLUGS="slug1,slug2,slug3"
    _market_slugs_env: str = os.getenv("POLYMARKET_MARKET_SLUGS", os.getenv("POLYMARKET_MARKET_SLUG", ""))
    market_slugs: list[str] = None  # Set in __post_init__
    
    market_slug: str = os.getenv("POLYMARKET_MARKET_SLUG", "")  # For backward compatibility (single market)
    market_id: str = os.getenv("POLYMARKET_MARKET_ID", "")  # For backward compatibility (single market)
    yes_token_id: str = os.getenv("POLYMARKET_YES_TOKEN_ID", "")  # For backward compatibility (single market)
    no_token_id: str = os.getenv("POLYMARKET_NO_TOKEN_ID", "")  # For backward compatibility (single market)
    
    ws_url: str = os.getenv("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com")
    use_wss: bool = os.getenv("USE_WSS", "false").lower() == "true"
    target_pair_cost: float = float(os.getenv("TARGET_PAIR_COST", "0.99"))
    balance_slack: float = float(os.getenv("BALANCE_SLACK", "0.15"))
    order_size: float = float(os.getenv("ORDER_SIZE", "50"))
    order_type: str = os.getenv("ORDER_TYPE", "FOK").upper()
    yes_buy_threshold: float = float(os.getenv("YES_BUY_THRESHOLD", "0.45"))
    no_buy_threshold: float = float(os.getenv("NO_BUY_THRESHOLD", "0.45"))
    verbose: bool = os.getenv("VERBOSE", "false").lower() == "true"
    dry_run: bool = os.getenv("DRY_RUN", "false").lower() == "true"
    cooldown_seconds: float = float(os.getenv("COOLDOWN_SECONDS", "10"))
    sim_balance: float = float(os.getenv("SIM_BALANCE", "0"))
    
    # Risk management settings
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "0"))  # 0 = disabled
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0"))  # 0 = disabled
    max_trades_per_day: int = int(os.getenv("MAX_TRADES_PER_DAY", "0"))  # 0 = disabled
    min_balance_required: float = float(os.getenv("MIN_BALANCE_REQUIRED", "10.0"))
    max_balance_utilization: float = float(os.getenv("MAX_BALANCE_UTILIZATION", "0.8"))
    
    # Dynamic position sizing
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))  # 2% of equity
    min_position_size: float = float(os.getenv("MIN_POSITION_SIZE", "10.0"))  # Min trade size in USDC
    
    # Stop-loss and take-profit (ATR-based)
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))  # ATR calculation period
    sl_atr_multiplier: float = float(os.getenv("SL_ATR_MULTIPLIER", "1.0"))  # SL = entry ± ATR × multiplier
    tp_atr_multiplier: float = float(os.getenv("TP_ATR_MULTIPLIER", "1.5"))  # TP = entry ± ATR × multiplier
    
    # Daily loss limit
    daily_loss_limit: float = float(os.getenv("DAILY_LOSS_LIMIT", "0"))  # 0 = disabled, overrides MAX_DAILY_LOSS
    
    # Statistics and logging
    enable_stats: bool = os.getenv("ENABLE_STATS", "true").lower() == "true"
    trade_log_file: str = os.getenv("TRADE_LOG_FILE", "trades.json")
    use_rich_output: bool = os.getenv("USE_RICH_OUTPUT", "true").lower() == "true"
    
    # Multi-market settings
    enable_multi_market: bool = os.getenv("ENABLE_MULTI_MARKET", "true").lower() == "true"
    auto_discover_markets: bool = os.getenv("AUTO_DISCOVER_MARKETS", "false").lower() == "true"
    
    # External price feed settings (for backtesting with historical data)
    data_provider_name: str = os.getenv("DATA_PROVIDER_NAME", "coingecko")
    data_provider_url: str = os.getenv("DATA_PROVIDER_URL", "https://api.coingecko.com/api/v3")
    data_provider_api_key: str = os.getenv("DATA_PROVIDER_API_KEY", "")
    data_provider_api_secret: str = os.getenv("DATA_PROVIDER_API_SECRET", "")
    data_provider_timeout: int = int(os.getenv("DATA_PROVIDER_TIMEOUT", "30"))
    data_provider_rate_limit_delay: float = float(os.getenv("DATA_PROVIDER_RATE_LIMIT_DELAY", "0.5"))

    # ============================================================================
    # Deriv (CFD / Multipliers) settings
    # ============================================================================
    deriv_app_id: str = os.getenv("DERIV_APP_ID", "")
    deriv_demo_token: str = os.getenv("DERIV_DEMO_TOKEN", "")
    deriv_ws_url: str = os.getenv("DERIV_WS_URL", "wss://ws.deriv.com/websockets/v3")
    deriv_symbol: str = os.getenv("DERIV_SYMBOL", "R_100")
    deriv_granularity: int = int(os.getenv("DERIV_GRANULARITY", "300"))  # 300=5m, 900=15m
    deriv_currency: str = os.getenv("DERIV_CURRENCY", "USD")
    deriv_multiplier: int = int(os.getenv("DERIV_MULTIPLIER", "100"))
    deriv_stake: float = float(os.getenv("DERIV_STAKE", os.getenv("ORDER_SIZE", "10")))
    deriv_long_contract_type: str = os.getenv("DERIV_LONG_CONTRACT_TYPE", "MULTUP")
    deriv_short_contract_type: str = os.getenv("DERIV_SHORT_CONTRACT_TYPE", "MULTDOWN")
    deriv_trend_confirm_candles: int = int(os.getenv("DERIV_TREND_CONFIRM_CANDLES", "3"))
    deriv_strategy_switch_cooldown: float = float(os.getenv("DERIV_STRATEGY_SWITCH_COOLDOWN", "300"))
    deriv_trend_short_ma: int = int(os.getenv("DERIV_TREND_SHORT_MA", "8"))
    deriv_trend_long_ma: int = int(os.getenv("DERIV_TREND_LONG_MA", "40"))
    deriv_trend_adx_threshold: float = float(os.getenv("DERIV_TREND_ADX_THRESHOLD", "25.0"))
    deriv_bb_period: int = int(os.getenv("DERIV_BB_PERIOD", "20"))
    deriv_bb_std_dev: float = float(os.getenv("DERIV_BB_STD_DEV", "1.5"))
    deriv_rsi_period: int = int(os.getenv("DERIV_RSI_PERIOD", "14"))
    deriv_rsi_threshold: float = float(os.getenv("DERIV_RSI_THRESHOLD", "35"))
    
    def __post_init__(self):
        """Parse market slugs from environment variables."""
        # Parse market slugs from comma-separated string
        if self._market_slugs_env:
            self.market_slugs = [s.strip() for s in self._market_slugs_env.split(",") if s.strip()]
        else:
            self.market_slugs = []
        
        # If market_slug is set (backward compatibility), use it as the only market
        if self.market_slug and not self.market_slugs:
            self.market_slugs = [self.market_slug]


def load_settings() -> Settings:
    return Settings()
