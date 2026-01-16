"""
Market condition detection module for identifying trending vs mean-reverting markets.

Provides functionality to determine if the market is trending using technical indicators.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def calculate_sma(prices: List[float], period: int) -> float:
    """
    Calculate simple moving average over the last `period` prices.
    
    Args:
        prices: List of price values
        period: Number of periods for the moving average
        
    Returns:
        Simple moving average value, or None if insufficient data
    """
    if len(prices) < period:
        return None
    
    return sum(prices[-period:]) / period


def calculate_adx(prices: List[float], period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) indicator.
    
    ADX measures trend strength on a scale of 0-100.
    ADX > 25 typically indicates a strong trend.
    ADX < 20 typically indicates no trend (choppy/ranging market).
    
    Args:
        prices: List of price values
        period: Number of periods for ADX calculation (default 14)
        
    Returns:
        ADX value (0-100), or None if insufficient data
    """
    if len(prices) < period * 2:
        return None
    
    # Calculate True Range
    true_ranges = []
    for i in range(1, len(prices)):
        high_low = abs(prices[i] - prices[i-1])
        high_close = abs(prices[i] - prices[i-1])
        low_close = abs(prices[i-1] - prices[i])
        tr = max(high_low, high_close, low_close)
        true_ranges.append(tr)
    
    # Calculate Directional Movements
    plus_dms = []
    minus_dms = []
    for i in range(1, len(prices)):
        up_move = prices[i] - prices[i-1]
        down_move = prices[i-1] - prices[i]
        
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0
        
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)
    
    # Calculate Directional Indicators
    atr = sum(true_ranges[-period:]) / period if len(true_ranges) >= period else sum(true_ranges) / len(true_ranges)
    
    if atr == 0:
        return 0.0
    
    plus_di = 100 * (sum(plus_dms[-period:]) / period) / atr if len(plus_dms) >= period else 0
    minus_di = 100 * (sum(minus_dms[-period:]) / period) / atr if len(minus_dms) >= period else 0
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0
    
    dx = 100 * abs(plus_di - minus_di) / di_sum
    
    # Smooth DX with exponential moving average
    adx_values = [dx]
    for i in range(period):
        if len(adx_values) >= period:
            alpha = 1.0 / period
            adx_values.append(alpha * dx + (1 - alpha) * adx_values[-1])
    
    return adx_values[-1] if adx_values else 0.0


def is_trending(prices: List[float], short_ma_period: int = 10, long_ma_period: int = 50, adx_threshold: float = 25.0) -> bool:
    """
    Determine if the market is currently trending.
    
    Uses a dual approach:
    1. Compares short-term MA (10-period) vs long-term MA (50-period)
    2. Checks ADX indicator for trend strength
    
    A market is considered trending if:
    - Short MA is above long MA, AND
    - ADX is above the threshold (indicating trend strength)
    
    Args:
        prices: List of recent price data
        short_ma_period: Period for short-term moving average (default 10)
        long_ma_period: Period for long-term moving average (default 50)
        adx_threshold: ADX threshold for trend confirmation (default 25.0)
        
    Returns:
        True if market is trending, False otherwise
    """
    if len(prices) < long_ma_period:
        logger.warning(f"Insufficient price data for trend detection. Need {long_ma_period}, got {len(prices)}")
        return False
    
    # Calculate moving averages
    short_ma = calculate_sma(prices, short_ma_period)
    long_ma = calculate_sma(prices, long_ma_period)
    
    if short_ma is None or long_ma is None:
        logger.warning("Unable to calculate moving averages")
        return False
    
    # Check if short MA is above long MA (uptrend)
    ma_trending = short_ma > long_ma
    
    # Calculate ADX for trend strength confirmation
    adx = calculate_adx(prices, period=14)
    
    if adx is None:
        # If ADX calculation fails, fall back to MA comparison alone
        logger.debug(f"ADX calculation failed, using MA comparison: short={short_ma:.4f}, long={long_ma:.4f}")
        return ma_trending
    
    # Require both MA condition and ADX strength
    is_strong_trend = adx >= adx_threshold
    
    logger.debug(
        f"Trend detection: short_ma={short_ma:.4f}, long_ma={long_ma:.4f}, "
        f"adx={adx:.2f}, threshold={adx_threshold}, trending={ma_trending and is_strong_trend}"
    )
    
    return ma_trending and is_strong_trend
