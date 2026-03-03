"""Abstract base class for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from prediction_bot.models import (
    Market,
    MarketDetail,
    OrderBook,
    PriceBar,
    Trade,
)


class MarketProvider(ABC):
    """Abstract interface for prediction market data providers.

    All providers (Kalshi, Polymarket, etc.) implement this interface
    so strategies and the trading engine can work with any provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""

    @abstractmethod
    async def get_markets(
        self,
        category: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[Market]:
        """Get list of active markets with current prices and volumes."""

    @abstractmethod
    async def get_market_details(self, market_id: str) -> MarketDetail:
        """Get extended market info: orderbook, recent trades, expiry."""

    @abstractmethod
    async def get_orderbook(self, market_id: str) -> OrderBook:
        """Get current orderbook for a market."""

    @abstractmethod
    async def get_recent_trades(
        self, market_id: str, limit: int = 50
    ) -> list[Trade]:
        """Get recent trades for a market."""

    @abstractmethod
    async def get_historical_prices(
        self,
        market_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1h",
    ) -> list[PriceBar]:
        """Get OHLCV-style price history for backtesting."""

    async def close(self) -> None:
        """Clean up resources (HTTP clients, etc.)."""
