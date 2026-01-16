"""
Price history collector - Saves market prices for backtesting.

Records price data from WebSocket feeds so you can backtest strategies
on the exact same data the live bot uses.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class PriceHistoryCollector:
    """Collects and persists price history for backtesting."""
    
    def __init__(self, market_slug: str, data_dir: str = "price_history"):
        """
        Initialize price collector.
        
        Args:
            market_slug: Market identifier (e.g., "BTC-updown-15m-20240101")
            data_dir: Directory to store price data
        """
        self.market_slug = market_slug
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.prices: List[Dict] = []
        self.filename = self.data_dir / f"{market_slug}.json"
        
        # Load existing data if available
        self._load_existing()
    
    def add_price(self, timestamp: str, open_price: float, close_price: float, 
                  high_price: float = None, low_price: float = None, volume: int = 0):
        """
        Add a price bar to history.
        
        Args:
            timestamp: ISO format timestamp
            open_price: Opening price
            close_price: Closing price
            high_price: High price (optional)
            low_price: Low price (optional)
            volume: Volume (optional)
        """
        high_price = high_price or max(open_price, close_price)
        low_price = low_price or min(open_price, close_price)
        
        bar = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        }
        
        # Check if this timestamp already exists
        existing = next((p for p in self.prices if p["timestamp"] == timestamp), None)
        if existing:
            logger.debug(f"Updating price for {timestamp}")
            existing.update(bar)
        else:
            self.prices.append(bar)
    
    def save(self):
        """Save price history to file."""
        with open(self.filename, "w") as f:
            json.dump(self.prices, f, indent=2)
        
        logger.debug(f"Saved {len(self.prices)} prices to {self.filename}")
    
    def _load_existing(self):
        """Load existing price data if file exists."""
        if self.filename.exists():
            with open(self.filename, "r") as f:
                self.prices = json.load(f)
            logger.info(f"Loaded {len(self.prices)} existing prices from {self.filename}")
    
    def get_prices(self) -> List[float]:
        """Get all closing prices as list."""
        return [p["close"] for p in self.prices]
    
    def get_timestamps(self) -> List[str]:
        """Get all timestamps as list."""
        return [p["timestamp"] for p in self.prices]
    
    def export_for_backtest(self) -> tuple[List[float], List[str]]:
        """
        Export prices and timestamps for backtester.
        
        Returns:
            (prices, timestamps) tuple
        """
        return self.get_prices(), self.get_timestamps()
    
    def get_summary(self) -> Dict:
        """Get summary statistics of collected data."""
        if not self.prices:
            return {"count": 0}
        
        prices = self.get_prices()
        return {
            "count": len(prices),
            "market_slug": self.market_slug,
            "first_timestamp": self.prices[0]["timestamp"],
            "last_timestamp": self.prices[-1]["timestamp"],
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": sum(prices) / len(prices),
            "latest_price": prices[-1],
        }


class MultiMarketPriceCollector:
    """Manages price collection for multiple markets."""
    
    def __init__(self, data_dir: str = "price_history"):
        """
        Initialize multi-market collector.
        
        Args:
            data_dir: Directory to store all market price data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collectors: Dict[str, PriceHistoryCollector] = {}
    
    def add_price(self, market_slug: str, timestamp: str, open_price: float, 
                  close_price: float, high_price: float = None, 
                  low_price: float = None, volume: int = 0):
        """Add price for a specific market."""
        if market_slug not in self.collectors:
            self.collectors[market_slug] = PriceHistoryCollector(
                market_slug=market_slug,
                data_dir=str(self.data_dir),
            )
        
        self.collectors[market_slug].add_price(
            timestamp=timestamp,
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
        )
    
    def save_all(self):
        """Save all market data."""
        for collector in self.collectors.values():
            collector.save()
        logger.info(f"Saved price data for {len(self.collectors)} markets")
    
    def get_collector(self, market_slug: str) -> PriceHistoryCollector:
        """Get collector for specific market."""
        if market_slug not in self.collectors:
            self.collectors[market_slug] = PriceHistoryCollector(
                market_slug=market_slug,
                data_dir=str(self.data_dir),
            )
        return self.collectors[market_slug]
    
    def export_for_backtest(self, market_slug: str) -> tuple[List[float], List[str]]:
        """Export prices and timestamps for specific market."""
        collector = self.get_collector(market_slug)
        return collector.export_for_backtest()
    
    def get_summary(self) -> Dict:
        """Get summary for all markets."""
        return {
            market_slug: collector.get_summary()
            for market_slug, collector in self.collectors.items()
        }


def load_price_data(filename: str) -> tuple[List[float], List[str]]:
    """
    Load price data from JSON file.
    
    Args:
        filename: Path to JSON file with price history
        
    Returns:
        (prices, timestamps) tuple
    """
    with open(filename, "r") as f:
        data = json.load(f)
    
    prices = [bar["close"] for bar in data]
    timestamps = [bar["timestamp"] for bar in data]
    
    return prices, timestamps
