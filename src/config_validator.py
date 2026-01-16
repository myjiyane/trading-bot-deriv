"""
Configuration validation module.

Validates bot configuration and provides helpful error messages.
"""

import logging
from typing import List, Tuple

from .config import Settings

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates bot configuration settings."""
    
    @staticmethod
    def validate(settings: Settings) -> Tuple[bool, List[str]]:
        """
        Validate configuration settings.
        
        Args:
            settings: Settings instance to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Deriv connection settings
        if not settings.deriv_app_id:
            errors.append("DERIV_APP_ID is required")
        else:
            if not str(settings.deriv_app_id).isdigit():
                errors.append("DERIV_APP_ID must be numeric")

        # Token required for live/demo trading; optional in dry-run mode
        if not settings.dry_run and not settings.deriv_demo_token:
            errors.append("DERIV_DEMO_TOKEN is required when DRY_RUN=false")

        # Trading parameters
        if settings.deriv_granularity not in (60, 120, 300, 900, 1800, 3600):
            errors.append("DERIV_GRANULARITY must be one of 60, 120, 300, 900, 1800, 3600")

        if settings.deriv_stake <= 0:
            errors.append("DERIV_STAKE must be > 0")

        if settings.deriv_multiplier <= 0:
            errors.append("DERIV_MULTIPLIER must be > 0")

        if settings.deriv_trend_confirm_candles < 1:
            errors.append("DERIV_TREND_CONFIRM_CANDLES must be >= 1")

        if settings.deriv_strategy_switch_cooldown < 0:
            errors.append("DERIV_STRATEGY_SWITCH_COOLDOWN must be >= 0")

        if settings.deriv_trend_short_ma < 1 or settings.deriv_trend_long_ma < 1:
            errors.append("DERIV_TREND_SHORT_MA and DERIV_TREND_LONG_MA must be >= 1")

        if settings.deriv_trend_adx_threshold < 0:
            errors.append("DERIV_TREND_ADX_THRESHOLD must be >= 0")

        if settings.deriv_bb_period < 2:
            errors.append("DERIV_BB_PERIOD must be >= 2")

        if settings.deriv_bb_std_dev <= 0:
            errors.append("DERIV_BB_STD_DEV must be > 0")

        if settings.deriv_rsi_period < 2:
            errors.append("DERIV_RSI_PERIOD must be >= 2")

        if settings.deriv_rsi_threshold <= 0 or settings.deriv_rsi_threshold >= 100:
            errors.append("DERIV_RSI_THRESHOLD must be between 0 and 100")

        if settings.cooldown_seconds < 0:
            errors.append("COOLDOWN_SECONDS must be >= 0")

        # Balance validation
        if settings.dry_run and settings.sim_balance < 0:
            errors.append("SIM_BALANCE must be >= 0 in simulation mode")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_and_print(settings: Settings) -> bool:
        """
        Validate configuration and print errors.
        
        Args:
            settings: Settings instance to validate
            
        Returns:
            True if valid, False otherwise
        """
        is_valid, errors = ConfigValidator.validate(settings)
        
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            logger.error("\nPlease fix the errors in your .env file and try again.")
        
        return is_valid
