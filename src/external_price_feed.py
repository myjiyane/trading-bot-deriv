"""
External price feed integration for fetching historical OHLCV data from third-party APIs.

Supports multiple data providers:
  - CoinGecko: Free tier, requires no API key, covers most cryptocurrencies
  - CoinMarketCap: Paid API, requires API key
  - Polygon.io: Stock/Crypto data, requires API key
  - Kraken: Direct API, free tier available
  - Binance: Largest crypto exchange, free tier available
  - Custom: User-defined API endpoint

Usage:
    from config import load_settings
    from external_price_feed import fetch_historical_prices
    
    settings = load_settings()
    prices, timestamps = fetch_historical_prices(
        symbol='BTC',
        start_date='2024-01-01',
        end_date='2024-01-31',
        interval='15m',
        settings=settings
    )
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class PriceFeedError(Exception):
    """Base exception for price feed errors."""
    pass


class PriceFeedNetworkError(PriceFeedError):
    """Network/connectivity error when fetching from API."""
    pass


class PriceFeedAuthError(PriceFeedError):
    """Authentication/API key error."""
    pass


class PriceFeedDataError(PriceFeedError):
    """Data parsing or format error."""
    pass


def fetch_historical_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "15m",
    settings=None
) -> Tuple[List[float], List[str]]:
    """
    Fetch historical OHLCV data from external price feed API.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH', 'MATIC')
        start_date: Start date in ISO format (YYYY-MM-DD) or ISO datetime
        end_date: End date in ISO format (YYYY-MM-DD) or ISO datetime
        interval: Time interval for candles (1m, 5m, 15m, 1h, 1d, etc.)
        settings: Settings object with DATA_PROVIDER_* config (or None to use defaults)
    
    Returns:
        Tuple of (prices, timestamps) lists:
        - prices: List of closing prices (floats)
        - timestamps: List of ISO datetime strings
        
    Raises:
        PriceFeedNetworkError: Network/connectivity issues
        PriceFeedAuthError: API authentication failures
        PriceFeedDataError: Data parsing errors
        PriceFeedError: Generic failures
    
    Examples:
        # Using CoinGecko (free, no API key needed)
        prices, timestamps = fetch_historical_prices(
            symbol='BTC',
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='15m'
        )
        
        # Using custom settings
        prices, timestamps = fetch_historical_prices(
            symbol='BTC',
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='15m',
            settings=my_settings
        )
    """
    
    if requests is None:
        logger.error("requests library not installed. Install with: pip install requests")
        raise PriceFeedError("requests library required for external price feeds")
    
    # Load default settings if not provided
    if settings is None:
        try:
            from .config import load_settings
            settings = load_settings()
        except Exception as e:
            logger.warning(f"Could not load settings: {e}. Using hardcoded defaults.")
            settings = None
    
    # Get provider configuration
    provider = (settings.data_provider_name if settings else "coingecko").lower()
    url = settings.data_provider_url if settings else "https://api.coingecko.com/api/v3"
    api_key = settings.data_provider_api_key if settings else ""
    timeout = settings.data_provider_timeout if settings else 30
    rate_limit_delay = settings.data_provider_rate_limit_delay if settings else 0.5
    
    logger.info(f"Fetching {symbol} prices from {provider} ({interval} candles)")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        if provider == "coingecko":
            return _fetch_coingecko(symbol, start_date, end_date, interval, url, timeout, rate_limit_delay)
        elif provider == "coinmarketcap":
            return _fetch_coinmarketcap(symbol, start_date, end_date, interval, url, api_key, timeout, rate_limit_delay)
        elif provider == "polygon":
            return _fetch_polygon(symbol, start_date, end_date, interval, url, api_key, timeout, rate_limit_delay)
        elif provider == "kraken":
            return _fetch_kraken(symbol, start_date, end_date, interval, url, timeout, rate_limit_delay)
        elif provider == "binance":
            return _fetch_binance(symbol, start_date, end_date, interval, timeout, rate_limit_delay)
        else:
            logger.warning(f"Unknown provider '{provider}'. Trying generic API call.")
            return _fetch_generic(symbol, start_date, end_date, interval, url, api_key, timeout)
            
    except (PriceFeedNetworkError, PriceFeedAuthError, PriceFeedDataError) as e:
        logger.error(f"Price feed error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching prices: {e}")
        raise PriceFeedError(f"Failed to fetch {symbol} prices: {str(e)}")


def _fetch_coingecko(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    url: str,
    timeout: int,
    rate_limit_delay: float
) -> Tuple[List[float], List[str]]:
    """
    Fetch data from CoinGecko API (free tier, no API key required).
    
    CoinGecko API: https://www.coingecko.com/api/documentation
    Note: Free tier supports 10-50 calls/minute. Returns daily data natively.
    For 15m data, we aggregate 1h or daily data.
    """
    
    # Map symbol to CoinGecko coin ID
    symbol_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'MATIC': 'matic-network',
        'SOL': 'solana',
        'ADA': 'cardano',
        'XRP': 'ripple',
        'USDC': 'usd-coin',
        'USDT': 'tether',
    }
    
    coin_id = symbol_map.get(symbol.upper(), symbol.lower())
    
    try:
        # CoinGecko returns daily OHLCV by default
        # For intraday, we'd need to use different interval approach
        # This example uses daily data; for 15m you'd need a different provider
        
        endpoint = urljoin(url, f"/coins/{coin_id}/market_chart")
        
        # Parse dates
        start_dt = _parse_iso_date(start_date)
        end_dt = _parse_iso_date(end_date)
        
        params = {
            'vs_currency': 'usd',
            'days': (end_dt - start_dt).days,
            'interval': 'daily'
        }
        
        logger.debug(f"CoinGecko request: {endpoint}, params={params}")
        
        response = requests.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        prices_data = data.get('prices', [])
        
        if not prices_data:
            logger.warning(f"No data returned from CoinGecko for {symbol}")
            return [], []
        
        prices = []
        timestamps = []
        
        for timestamp_ms, price in prices_data:
            # CoinGecko returns timestamps in milliseconds
            dt = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
            iso_timestamp = dt.isoformat() + 'Z'
            
            # Filter by date range
            if start_dt <= dt <= end_dt:
                prices.append(float(price))
                timestamps.append(iso_timestamp)
            
            time.sleep(rate_limit_delay)
        
        logger.info(f"Successfully fetched {len(prices)} price points from CoinGecko")
        return prices, timestamps
        
    except requests.exceptions.Timeout as e:
        raise PriceFeedNetworkError(f"CoinGecko request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise PriceFeedNetworkError(f"CoinGecko connection error: {e}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise PriceFeedAuthError(f"CoinGecko authentication error: {e}")
        raise PriceFeedNetworkError(f"CoinGecko HTTP error: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise PriceFeedDataError(f"Failed to parse CoinGecko response: {e}")


def _fetch_coinmarketcap(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    url: str,
    api_key: str,
    timeout: int,
    rate_limit_delay: float
) -> Tuple[List[float], List[str]]:
    """
    Fetch data from CoinMarketCap API (paid tier, requires API key).
    
    CoinMarketCap API: https://coinmarketcap.com/api/documentation/v1/
    """
    
    if not api_key:
        raise PriceFeedAuthError("CoinMarketCap API key required but not provided")
    
    try:
        endpoint = urljoin(url, "/cryptocurrency/ohlcv/historical")
        
        headers = {
            'X-CMC_PRO_API_KEY': api_key,
            'Accept': 'application/json',
        }
        
        params = {
            'symbol': symbol.upper(),
            'time_start': start_date,
            'time_end': end_date,
            'interval': interval,
            'convert': 'USD'
        }
        
        logger.debug(f"CoinMarketCap request: {endpoint}, params={params}")
        
        response = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status', {}).get('error_code') != 0:
            error_msg = data.get('status', {}).get('error_message', 'Unknown error')
            raise PriceFeedDataError(f"CoinMarketCap API error: {error_msg}")
        
        prices = []
        timestamps = []
        
        ohlcv_data = data.get('data', {}).get('quotes', [])
        
        for candle in ohlcv_data:
            timestamp = candle.get('timestamp', '')
            close = candle.get('quote', {}).get('USD', {}).get('close')
            
            if close is not None:
                prices.append(float(close))
                timestamps.append(timestamp)
            
            time.sleep(rate_limit_delay)
        
        logger.info(f"Successfully fetched {len(prices)} price points from CoinMarketCap")
        return prices, timestamps
        
    except requests.exceptions.Timeout as e:
        raise PriceFeedNetworkError(f"CoinMarketCap request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise PriceFeedNetworkError(f"CoinMarketCap connection error: {e}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise PriceFeedAuthError(f"CoinMarketCap authentication error: {e}")
        raise PriceFeedNetworkError(f"CoinMarketCap HTTP error: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise PriceFeedDataError(f"Failed to parse CoinMarketCap response: {e}")


def _fetch_polygon(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    url: str,
    api_key: str,
    timeout: int,
    rate_limit_delay: float
) -> Tuple[List[float], List[str]]:
    """
    Fetch data from Polygon.io API (requires API key).
    
    Polygon.io API: https://polygon.io/docs/crypto/getting-started
    """
    
    if not api_key:
        raise PriceFeedAuthError("Polygon.io API key required but not provided")
    
    try:
        # Convert interval to Polygon format (1m, 5m, 15m, 30m, 1h, 1d)
        poly_interval = _normalize_interval(interval, 'polygon')
        
        endpoint = urljoin(url, f"/v2/aggs/ticker/C:{symbol}USD/range/{poly_interval}/{start_date}/{end_date}")
        
        params = {
            'apiKey': api_key,
            'limit': 50000,
            'sort': 'asc'
        }
        
        logger.debug(f"Polygon request: {endpoint}")
        
        response = requests.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('status') == 'OK':
            error_msg = data.get('message', 'Unknown error')
            raise PriceFeedDataError(f"Polygon.io API error: {error_msg}")
        
        prices = []
        timestamps = []
        
        for result in data.get('results', []):
            timestamp_ms = result.get('t')
            close = result.get('c')
            
            if timestamp_ms and close is not None:
                dt = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
                iso_timestamp = dt.isoformat() + 'Z'
                
                prices.append(float(close))
                timestamps.append(iso_timestamp)
            
            time.sleep(rate_limit_delay)
        
        logger.info(f"Successfully fetched {len(prices)} price points from Polygon.io")
        return prices, timestamps
        
    except requests.exceptions.Timeout as e:
        raise PriceFeedNetworkError(f"Polygon.io request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise PriceFeedNetworkError(f"Polygon.io connection error: {e}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise PriceFeedAuthError(f"Polygon.io authentication error: {e}")
        raise PriceFeedNetworkError(f"Polygon.io HTTP error: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise PriceFeedDataError(f"Failed to parse Polygon.io response: {e}")


def _fetch_kraken(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    url: str,
    timeout: int,
    rate_limit_delay: float
) -> Tuple[List[float], List[str]]:
    """
    Fetch data from Kraken API (free tier available).
    
    Kraken API: https://docs.kraken.com/rest/
    """
    
    try:
        # Convert symbol to Kraken pair format (e.g., BTC -> XXBTZUSD)
        kraken_symbol = _symbol_to_kraken_pair(symbol)
        
        # Convert interval to Kraken format (minutes)
        kraken_interval = _normalize_interval(interval, 'kraken')
        
        endpoint = urljoin(url, "/0/public/OHLC")
        
        start_dt = _parse_iso_date(start_date)
        start_timestamp = int(start_dt.timestamp())
        
        params = {
            'pair': kraken_symbol,
            'interval': kraken_interval,
            'since': start_timestamp
        }
        
        logger.debug(f"Kraken request: {endpoint}, params={params}")
        
        response = requests.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('error'):
            raise PriceFeedDataError(f"Kraken API error: {data['error']}")
        
        prices = []
        timestamps = []
        
        # Kraken returns data keyed by pair name
        pair_data = data.get('result', {}).get(kraken_symbol, [])
        
        end_dt = _parse_iso_date(end_date)
        
        for candle in pair_data:
            timestamp = int(candle[0])
            close = float(candle[4])  # Close price is at index 4
            
            dt = datetime.utcfromtimestamp(timestamp)
            
            if dt <= end_dt:
                iso_timestamp = dt.isoformat() + 'Z'
                prices.append(close)
                timestamps.append(iso_timestamp)
            
            time.sleep(rate_limit_delay)
        
        logger.info(f"Successfully fetched {len(prices)} price points from Kraken")
        return prices, timestamps
        
    except requests.exceptions.Timeout as e:
        raise PriceFeedNetworkError(f"Kraken request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise PriceFeedNetworkError(f"Kraken connection error: {e}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise PriceFeedAuthError(f"Kraken authentication error: {e}")
        raise PriceFeedNetworkError(f"Kraken HTTP error: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise PriceFeedDataError(f"Failed to parse Kraken response: {e}")


def _fetch_binance(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    timeout: int,
    rate_limit_delay: float
) -> Tuple[List[float], List[str]]:
    """
    Fetch data from Binance API (free tier available).
    
    Binance API: https://binance-docs.github.io/apidocs/
    """
    
    try:
        # Convert interval to Binance format (1m, 5m, 15m, 30m, 1h, etc.)
        binance_interval = _normalize_interval(interval, 'binance')
        
        endpoint = "https://api.binance.com/api/v3/klines"
        
        symbol_pair = f"{symbol.upper()}USDT"
        
        start_dt = _parse_iso_date(start_date)
        end_dt = _parse_iso_date(end_date)
        
        params = {
            'symbol': symbol_pair,
            'interval': binance_interval,
            'startTime': int(start_dt.timestamp() * 1000),
            'endTime': int(end_dt.timestamp() * 1000),
            'limit': 1000
        }
        
        logger.debug(f"Binance request: {endpoint}, params={params}")
        
        prices = []
        timestamps = []
        
        # Binance limits 1000 candles per request, so we may need multiple calls
        while params['startTime'] < params['endTime']:
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                break
            
            for candle in data:
                timestamp_ms = candle[0]
                close = float(candle[4])  # Close price at index 4
                
                dt = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
                iso_timestamp = dt.isoformat() + 'Z'
                
                prices.append(close)
                timestamps.append(iso_timestamp)
            
            # Update start time for next batch
            params['startTime'] = int(data[-1][0]) + 1
            
            time.sleep(rate_limit_delay)
        
        logger.info(f"Successfully fetched {len(prices)} price points from Binance")
        return prices, timestamps
        
    except requests.exceptions.Timeout as e:
        raise PriceFeedNetworkError(f"Binance request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise PriceFeedNetworkError(f"Binance connection error: {e}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise PriceFeedAuthError(f"Binance authentication error: {e}")
        raise PriceFeedNetworkError(f"Binance HTTP error: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise PriceFeedDataError(f"Failed to parse Binance response: {e}")


def _fetch_generic(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    url: str,
    api_key: str,
    timeout: int
) -> Tuple[List[float], List[str]]:
    """
    Generic fallback for custom/user-defined API endpoints.
    
    Expected JSON response format:
    {
        "prices": [{"timestamp": "2024-01-01T00:00:00Z", "close": 50000.0}, ...],
        "data": [...],
        "results": [{"timestamp": "2024-01-01T00:00:00Z", "price": 50000.0}, ...]
    }
    """
    
    try:
        params = {
            'symbol': symbol,
            'start': start_date,
            'end': end_date,
            'interval': interval,
            'limit': 10000
        }
        
        if api_key:
            params['api_key'] = api_key
        
        logger.debug(f"Generic API request: {url}")
        
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        prices = []
        timestamps = []
        
        # Try to extract from common response formats
        candles = (
            data.get('prices') or
            data.get('data') or
            data.get('results') or
            data.get('candles') or
            []
        )
        
        for candle in candles:
            if isinstance(candle, dict):
                close = candle.get('close') or candle.get('price') or candle.get('c')
                timestamp = candle.get('timestamp') or candle.get('time') or candle.get('t')
            elif isinstance(candle, (list, tuple)) and len(candle) >= 2:
                timestamp = candle[0]
                close = candle[1]
            else:
                continue
            
            if close is not None and timestamp:
                prices.append(float(close))
                timestamps.append(str(timestamp))
        
        if not prices:
            raise PriceFeedDataError("Could not extract price data from response")
        
        logger.info(f"Successfully fetched {len(prices)} price points from custom API")
        return prices, timestamps
        
    except requests.exceptions.Timeout as e:
        raise PriceFeedNetworkError(f"Generic API request timeout: {e}")
    except requests.exceptions.ConnectionError as e:
        raise PriceFeedNetworkError(f"Generic API connection error: {e}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise PriceFeedAuthError(f"Generic API authentication error: {e}")
        raise PriceFeedNetworkError(f"Generic API HTTP error: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise PriceFeedDataError(f"Failed to parse generic API response: {e}")


def resample_prices(
    prices: List[float],
    timestamps: List[str],
    from_interval: str,
    to_interval: str
) -> Tuple[List[float], List[str]]:
    """
    Resample price data to a different time interval.
    
    This is useful when the API provides higher frequency data (e.g., 1m candles)
    but you need lower frequency data (e.g., 15m candles).
    
    Args:
        prices: List of closing prices
        timestamps: List of corresponding ISO datetime strings
        from_interval: Current interval (e.g., '1m')
        to_interval: Target interval (e.g., '15m')
    
    Returns:
        Tuple of (resampled_prices, resampled_timestamps)
    
    Example:
        prices_1m, ts_1m = fetch_historical_prices('BTC', '2024-01-01', '2024-01-31', '1m')
        prices_15m, ts_15m = resample_prices(prices_1m, ts_1m, '1m', '15m')
    """
    
    if not prices or not timestamps:
        return [], []
    
    if from_interval == to_interval:
        return prices, timestamps
    
    # Extract interval values in minutes
    from_mins = _interval_to_minutes(from_interval)
    to_mins = _interval_to_minutes(to_interval)
    
    if from_mins >= to_mins:
        logger.warning(f"Cannot resample from {from_interval} to {to_interval} (not a coarser interval)")
        return prices, timestamps
    
    multiplier = to_mins // from_mins
    
    resampled_prices = []
    resampled_timestamps = []
    
    for i in range(0, len(prices), multiplier):
        # Take the last (close) price in each bucket
        bucket = prices[i:i+multiplier]
        if bucket:
            resampled_prices.append(bucket[-1])
            resampled_timestamps.append(timestamps[i])
    
    logger.info(f"Resampled {len(prices)} candles from {from_interval} to {to_interval}: {len(resampled_prices)} candles")
    
    return resampled_prices, resampled_timestamps


# Helper functions

def _parse_iso_date(date_str: str) -> datetime:
    """Parse ISO datetime string to datetime object."""
    # Try various ISO formats
    formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.replace('Z', '+00:00').split('+')[0], fmt.replace('Z', ''))
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date: {date_str}")


def _normalize_interval(interval: str, provider: str) -> str:
    """
    Normalize interval format to provider-specific format.
    
    Common formats:
      - minutes: 1m, 5m, 15m, 30m
      - hours: 1h, 4h
      - daily: 1d, d, daily
    """
    
    interval_lower = interval.lower()
    
    if provider == 'binance':
        # Binance: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
        return interval_lower
    elif provider == 'kraken':
        # Kraken: interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 43200)
        minutes = _interval_to_minutes(interval)
        return str(minutes)
    elif provider == 'polygon':
        # Polygon: 1m, 5m, 15m, 30m, 1h, 1d
        return interval_lower
    else:
        return interval_lower


def _interval_to_minutes(interval: str) -> int:
    """Convert interval string to minutes."""
    interval_lower = interval.lower()
    
    if 'm' in interval_lower:
        return int(interval_lower.replace('m', ''))
    elif 'h' in interval_lower:
        return int(interval_lower.replace('h', '')) * 60
    elif 'd' in interval_lower:
        return int(interval_lower.replace('d', '')) * 24 * 60
    elif 'w' in interval_lower:
        return int(interval_lower.replace('w', '')) * 7 * 24 * 60
    else:
        return 1  # Default to 1 minute


def _symbol_to_kraken_pair(symbol: str) -> str:
    """Convert standard symbol to Kraken pair format."""
    symbol_map = {
        'BTC': 'XXBTZUSD',
        'ETH': 'XETHZUSD',
        'MATIC': 'MATICUSD',
        'SOL': 'SOLUSD',
        'ADA': 'ADAUSD',
        'XRP': 'XRPUSD',
        'USDC': 'USDCUSD',
        'USDT': 'USDTZUSD',
    }
    
    return symbol_map.get(symbol.upper(), f"X{symbol.upper()}ZUSD")
