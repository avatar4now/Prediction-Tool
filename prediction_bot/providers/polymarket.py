"""Polymarket prediction market data provider.

API docs: https://docs.polymarket.com/

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


class PolymarketProvider(MarketProvider):
    """Read-only data provider for Polymarket (CLOB + Gamma APIs)."""

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        gamma_url: str = "https://gamma-api.polymarket.com",
        rate_limit_per_sec: int = 10,
    ):
        self._clob_url = clob_url.rstrip("/")
        self._gamma_url = gamma_url.rstrip("/")
        self._clob = httpx.AsyncClient(
            base_url=self._clob_url,
            headers={"Accept": "application/json"},
            timeout=30.0,
        )
        self._gamma = httpx.AsyncClient(
            base_url=self._gamma_url,
            headers={"Accept": "application/json"},
            timeout=30.0,
        )

    @property
    def name(self) -> str:
        return "polymarket"

    # ── Market listing (via Gamma API) ──────────────────────────────

    async def get_markets(
        self,
        category: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[Market]:
        params: dict = {
            "limit": min(limit, 100),
            "active": "true" if status == "open" else "false",
            "closed": "false" if status == "open" else "true",
        }
        if category:
            params["tag"] = category

        resp = await self._gamma.get("/markets", params=params)
        resp.raise_for_status()
        items = resp.json()

        # Gamma returns a list of market objects (each may have outcome tokens)
        markets: list[Market] = []
        for item in items if isinstance(items, list) else []:
            markets.extend(self._parse_gamma_market(item))
        return markets[:limit]

    # ── Market details ──────────────────────────────────────────────

    async def get_market_details(self, market_id: str) -> MarketDetail:
        # Fetch from Gamma for metadata
        resp = await self._gamma.get(f"/markets/{market_id}")
        resp.raise_for_status()
        gamma_data = resp.json()

        parsed = self._parse_gamma_market(gamma_data)
        market = parsed[0] if parsed else self._fallback_market(market_id)

        orderbook = await self.get_orderbook(market_id)
        trades = await self.get_recent_trades(market_id)

        return MarketDetail(
            market=market,
            orderbook=orderbook,
            recent_trades=trades,
        )

    # ── Orderbook (CLOB API) ───────────────────────────────────────

    async def get_orderbook(self, market_id: str) -> OrderBook:
        # The CLOB uses token_id for orderbook queries.
        # market_id here may be the condition_id; try both.
        resp = await self._clob.get(f"/book", params={"token_id": market_id})
        resp.raise_for_status()
        data = resp.json()

        bids = [
            OrderBookLevel(price=float(lvl.get("price", 0)), quantity=float(lvl.get("size", 0)))
            for lvl in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(lvl.get("price", 0)), quantity=float(lvl.get("size", 0)))
            for lvl in data.get("asks", [])
        ]
        return OrderBook(market_id=market_id, bids=bids, asks=asks)

    # ── Recent trades ───────────────────────────────────────────────

    async def get_recent_trades(
        self, market_id: str, limit: int = 50
    ) -> list[Trade]:
        # CLOB trades endpoint
        resp = await self._clob.get(
            "/trades",
            params={"asset_id": market_id, "limit": min(limit, 100)},
        )
        if resp.status_code != 200:
            return []
        data = resp.json()

        trades: list[Trade] = []
        for t in data if isinstance(data, list) else data.get("trades", data.get("data", [])):
            ts = t.get("timestamp") or t.get("created_at", "")
            trades.append(
                Trade(
                    trade_id=str(t.get("id", "")),
                    market_id=market_id,
                    provider=Provider.POLYMARKET,
                    side=Side.BUY if t.get("side", "").lower() == "buy" else Side.SELL,
                    price=float(t.get("price", 0)),
                    quantity=int(float(t.get("size", t.get("amount", 0)))),
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

        # Gamma API provides price history
        params = {
            "market": market_id,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "fidelity": self._map_fidelity(interval),
        }
        resp = await self._gamma.get("/prices", params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()

        bars: list[PriceBar] = []
        history = data if isinstance(data, list) else data.get("history", [])
        for point in history:
            ts = point.get("t") or point.get("timestamp", 0)
            price = float(point.get("p", point.get("price", 0)))
            bars.append(
                PriceBar(
                    timestamp=self._parse_ts(ts),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=float(point.get("v", point.get("volume", 0))),
                )
            )
        return sorted(bars, key=lambda b: b.timestamp)

    # ── Cleanup ─────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._clob.aclose()
        await self._gamma.aclose()

    # ── Internal helpers ────────────────────────────────────────────

    def _parse_gamma_market(self, item: dict) -> list[Market]:
        """Parse a Gamma API market response into Market models.

        A single Gamma 'market' can represent multiple outcome tokens.
        """
        markets: list[Market] = []

        # Gamma markets have tokens array for binary outcomes
        tokens = item.get("tokens", []) or item.get("clobTokenIds", [])
        condition_id = item.get("conditionId", item.get("id", ""))
        question = item.get("question", item.get("title", ""))
        category = item.get("groupItemTitle", item.get("category", ""))

        expiry_str = item.get("endDate") or item.get("end_date_iso")
        expiry = self._parse_ts(expiry_str) if expiry_str else None

        # Use the outcomePrices field if available
        outcome_prices = item.get("outcomePrices", "")
        prices = []
        if outcome_prices:
            try:
                if isinstance(outcome_prices, str):
                    import json
                    prices = json.loads(outcome_prices)
                else:
                    prices = outcome_prices
            except (ValueError, TypeError):
                prices = []

        yes_price = float(prices[0]) if len(prices) > 0 else 0.5
        no_price = float(prices[1]) if len(prices) > 1 else (1.0 - yes_price)

        volume = float(item.get("volume", item.get("volumeNum", 0)) or 0)

        market_id = condition_id
        # If tokens are available, use the first token as market_id
        if isinstance(tokens, list) and tokens:
            if isinstance(tokens[0], dict):
                market_id = tokens[0].get("token_id", condition_id)
            elif isinstance(tokens[0], str):
                market_id = tokens[0]

        slug = item.get("slug", "")
        markets.append(
            Market(
                market_id=market_id,
                provider=Provider.POLYMARKET,
                title=question,
                category=category,
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                open_interest=float(item.get("liquidity", item.get("openInterest", 0)) or 0),
                expiry=expiry,
                status="open" if item.get("active") else "closed",
                url=f"https://polymarket.com/event/{slug}" if slug else "",
            )
        )
        return markets

    def _fallback_market(self, market_id: str) -> Market:
        return Market(
            market_id=market_id,
            provider=Provider.POLYMARKET,
            title="Unknown Market",
            category="",
            yes_price=0.5,
            no_price=0.5,
            volume=0,
            open_interest=0,
            expiry=None,
            status="open",
        )

    @staticmethod
    def _parse_ts(ts) -> datetime:
        if isinstance(ts, (int, float)):
            if ts > 1e12:  # milliseconds
                ts = ts / 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            ts = ts.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return datetime.now(tz=timezone.utc)
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _map_fidelity(interval: str) -> int:
        """Map interval strings to Gamma API fidelity (minutes)."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping.get(interval, 60)
