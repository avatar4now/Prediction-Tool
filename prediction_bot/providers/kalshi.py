"""Kalshi prediction market data provider.

API docs: https://trading-api.readme.io/reference/getting-started

PAPER TRADING ONLY — this provider only reads public market data.
No real orders are ever placed through this provider.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from prediction_bot.models import (
    Market,
    MarketDetail,
    OrderBook,
    OrderBookLevel,
    PriceBar,
    Provider,
    Side,
    Trade,
)
from prediction_bot.providers.base import MarketProvider

logger = logging.getLogger(__name__)

# Kalshi expresses prices in cents (1–99).  We normalize to 0.0–1.0.
_CENTS_TO_PROB = 0.01


class KalshiProvider(MarketProvider):
    """Read-only data provider for the Kalshi prediction market."""

    def __init__(
        self,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        api_key: str = "",
        rate_limit_per_sec: int = 10,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=30.0,
        )
        self._rate_limit = rate_limit_per_sec

    @property
    def name(self) -> str:
        return "kalshi"

    # ── Public market list ──────────────────────────────────────────

    async def get_markets(
        self,
        category: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[Market]:
        params: dict = {"limit": min(limit, 200), "status": status}
        if category:
            params["series_ticker"] = category

        resp = await self._client.get("/markets", params=params)
        resp.raise_for_status()
        data = resp.json()

        markets: list[Market] = []
        for m in data.get("markets", []):
            markets.append(self._parse_market(m))
        return markets[:limit]

    # ── Market details ──────────────────────────────────────────────

    async def get_market_details(self, market_id: str) -> MarketDetail:
        resp = await self._client.get(f"/markets/{market_id}")
        resp.raise_for_status()
        market_data = resp.json().get("market", resp.json())
        market = self._parse_market(market_data)

        orderbook = await self.get_orderbook(market_id)
        trades = await self.get_recent_trades(market_id)

        return MarketDetail(
            market=market,
            orderbook=orderbook,
            recent_trades=trades,
        )

    # ── Orderbook ───────────────────────────────────────────────────

    async def get_orderbook(self, market_id: str) -> OrderBook:
        resp = await self._client.get(f"/markets/{market_id}/orderbook")
        resp.raise_for_status()
        data = resp.json().get("orderbook", resp.json())

        bids = [
            OrderBookLevel(price=lvl[0] * _CENTS_TO_PROB, quantity=lvl[1])
            for lvl in data.get("yes", [])
        ]
        asks = [
            OrderBookLevel(price=lvl[0] * _CENTS_TO_PROB, quantity=lvl[1])
            for lvl in data.get("no", [])
        ]
        return OrderBook(market_id=market_id, bids=bids, asks=asks)

    # ── Recent trades ───────────────────────────────────────────────

    async def get_recent_trades(
        self, market_id: str, limit: int = 50
    ) -> list[Trade]:
        params = {"limit": min(limit, 100), "ticker": market_id}
        resp = await self._client.get("/markets/trades", params=params)
        resp.raise_for_status()
        data = resp.json()

        trades: list[Trade] = []
        for t in data.get("trades", []):
            ts = t.get("created_time") or t.get("ts", "")
            trades.append(
                Trade(
                    trade_id=str(t.get("id", "")),
                    market_id=market_id,
                    provider=Provider.KALSHI,
                    side=Side.BUY if t.get("taker_side", "").lower() != "no" else Side.SELL,
                    price=t.get("yes_price", t.get("price", 0)) * _CENTS_TO_PROB,
                    quantity=t.get("count", t.get("contracts", 0)),
                    timestamp=self._parse_ts(ts),
                )
            )
        return trades[:limit]

    # ── Historical prices ───────────────────────────────────────────

    async def get_historical_prices(
        self,
        market_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1h",
    ) -> list[PriceBar]:
        if end is None:
            end = datetime.now(tz=timezone.utc)
        if start is None:
            start = end - timedelta(days=30)

        params = {
            "ticker": market_id,
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": self._map_interval(interval),
        }
        resp = await self._client.get(f"/markets/{market_id}/candlesticks", params=params)
        resp.raise_for_status()
        data = resp.json()

        bars: list[PriceBar] = []
        for c in data.get("candlesticks", []):
            bars.append(
                PriceBar(
                    timestamp=self._parse_ts(c.get("end_period_ts", c.get("t", ""))),
                    open=c.get("open", c.get("price", 0)) * _CENTS_TO_PROB,
                    high=c.get("high", c.get("price", 0)) * _CENTS_TO_PROB,
                    low=c.get("low", c.get("price", 0)) * _CENTS_TO_PROB,
                    close=c.get("close", c.get("price", 0)) * _CENTS_TO_PROB,
                    volume=c.get("volume", 0),
                )
            )
        return sorted(bars, key=lambda b: b.timestamp)

    # ── Cleanup ─────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._client.aclose()

    # ── Internal helpers ────────────────────────────────────────────

    def _parse_market(self, m: dict) -> Market:
        yes_price = (m.get("yes_bid", 0) or 0) * _CENTS_TO_PROB
        no_price = (m.get("no_bid", 0) or 0) * _CENTS_TO_PROB

        # Fallback: use last_price or yes_sub_title pricing
        if yes_price == 0:
            yes_price = (m.get("last_price", 0) or 0) * _CENTS_TO_PROB

        expiry_str = m.get("expiration_time") or m.get("close_time")
        expiry = self._parse_ts(expiry_str) if expiry_str else None

        return Market(
            market_id=m.get("ticker", m.get("id", "")),
            provider=Provider.KALSHI,
            title=m.get("title", m.get("subtitle", "")),
            category=m.get("series_ticker", m.get("category", "")),
            yes_price=yes_price,
            no_price=no_price,
            volume=m.get("volume", 0),
            open_interest=m.get("open_interest", 0),
            expiry=expiry,
            status=m.get("status", "open"),
            url=f"https://kalshi.com/markets/{m.get('ticker', '')}",
        )

    @staticmethod
    def _parse_ts(ts) -> datetime:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            # Handle ISO format with or without timezone
            ts = ts.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return datetime.now(tz=timezone.utc)
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _map_interval(interval: str) -> int:
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping.get(interval, 60)
