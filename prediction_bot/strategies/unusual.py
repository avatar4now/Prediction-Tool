"""Unusual activity scanner for prediction markets.

Detects volume spikes, price momentum, and large trades to generate
a composite U-Score — similar to Unusual Whales but for prediction markets.

Indicators:
  - Volume spike: current volume vs historical average
  - Momentum: rapid price movement over short window
  - Smart money: large individual trades (whale detection)
  - Book imbalance: bid/ask size asymmetry signaling directional pressure
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from prediction_bot.models import (
    Market,
    MarketDetail,
    OrderBook,
    PriceBar,
    Trade,
)

logger = logging.getLogger(__name__)


# ── Indicator tags ────────────────────────────────────────────────────

INDICATOR_VOLUME = "VOL"       # Volume spike
INDICATOR_MOMENTUM = "MOM"     # Price momentum
INDICATOR_SMART = "SMART"      # Large / whale trades
INDICATOR_BOOK = "BOOK"        # Order book imbalance


@dataclass
class UnusualMarket:
    """A market flagged for unusual activity."""

    market: Market
    u_score: float                           # Composite 0–100
    indicators: list[str] = field(default_factory=list)  # Active indicator tags
    volume_ratio: float = 0.0                # Current vs avg volume
    price_change_pct: float = 0.0            # Recent price change %
    momentum_score: float = 0.0              # 0–1 momentum component
    smart_money_score: float = 0.0           # 0–1 whale component
    book_imbalance: float = 0.0              # -1 (sell pressure) to +1 (buy pressure)
    whale_trades: int = 0                    # Count of large trades
    recent_trades_count: int = 0             # Total recent trades
    direction: str = ""                      # "BULLISH" / "BEARISH" / "NEUTRAL"


class UnusualActivityScanner:
    """Scans markets for unusual activity and computes U-Scores."""

    def __init__(
        self,
        volume_spike_threshold: float = 2.0,
        momentum_threshold: float = 0.05,
        whale_size_multiplier: float = 5.0,
        book_imbalance_threshold: float = 0.3,
        lookback_bars: int = 20,
    ):
        self.volume_spike_threshold = volume_spike_threshold
        self.momentum_threshold = momentum_threshold
        self.whale_size_multiplier = whale_size_multiplier
        self.book_imbalance_threshold = book_imbalance_threshold
        self.lookback_bars = lookback_bars

    def score_market(
        self,
        market: Market,
        history: list[PriceBar],
        recent_trades: list[Trade],
        orderbook: Optional[OrderBook] = None,
    ) -> Optional[UnusualMarket]:
        """Compute U-Score for a single market.

        Returns None if no unusual activity detected.
        """
        indicators: list[str] = []
        scores: dict[str, float] = {}

        # ── Volume spike ──────────────────────────────────────────
        vol_ratio, vol_score = self._calc_volume_spike(market, history)
        if vol_score > 0:
            indicators.append(INDICATOR_VOLUME)
        scores["volume"] = vol_score

        # ── Momentum ──────────────────────────────────────────────
        price_change, mom_score = self._calc_momentum(history)
        if mom_score > 0:
            indicators.append(INDICATOR_MOMENTUM)
        scores["momentum"] = mom_score

        # ── Smart money / whale detection ─────────────────────────
        whale_count, smart_score = self._calc_smart_money(recent_trades)
        if smart_score > 0:
            indicators.append(INDICATOR_SMART)
        scores["smart"] = smart_score

        # ── Order book imbalance ──────────────────────────────────
        imbalance, book_score = self._calc_book_imbalance(orderbook)
        if book_score > 0:
            indicators.append(INDICATOR_BOOK)
        scores["book"] = book_score

        # ── Composite U-Score (0–100) ─────────────────────────────
        # Weighted: volume=35%, momentum=25%, smart_money=25%, book=15%
        raw = (
            scores["volume"] * 0.35
            + scores["momentum"] * 0.25
            + scores["smart"] * 0.25
            + scores["book"] * 0.15
        )
        u_score = min(100.0, raw * 100.0)

        if not indicators:
            return None

        # ── Direction ─────────────────────────────────────────────
        direction = self._determine_direction(
            price_change, imbalance, recent_trades
        )

        return UnusualMarket(
            market=market,
            u_score=round(u_score, 1),
            indicators=indicators,
            volume_ratio=round(vol_ratio, 2),
            price_change_pct=round(price_change * 100, 2),
            momentum_score=round(mom_score, 3),
            smart_money_score=round(smart_score, 3),
            book_imbalance=round(imbalance, 3),
            whale_trades=whale_count,
            recent_trades_count=len(recent_trades),
            direction=direction,
        )

    def _calc_volume_spike(
        self, market: Market, history: list[PriceBar]
    ) -> tuple[float, float]:
        """Compare current volume to historical average.

        Returns (volume_ratio, score 0–1).
        """
        if not history:
            # Use market.volume as-is — no baseline available
            if market.volume > 0:
                return (1.0, 0.0)
            return (0.0, 0.0)

        # Historical average volume per bar
        historical_vols = [b.volume for b in history[:-1]] if len(history) > 1 else [b.volume for b in history]
        avg_vol = statistics.mean(historical_vols) if historical_vols else 0

        if avg_vol <= 0:
            # No historical volume to compare — check if current is non-zero
            current_vol = history[-1].volume if history else market.volume
            if current_vol > 0:
                return (float("inf"), 0.5)
            return (0.0, 0.0)

        current_vol = history[-1].volume if history else market.volume
        ratio = current_vol / avg_vol

        if ratio < self.volume_spike_threshold:
            return (ratio, 0.0)

        # Score scales from 0 at threshold to 1 at 5x threshold
        score = min(1.0, (ratio - self.volume_spike_threshold) / (self.volume_spike_threshold * 4))
        return (ratio, score)

    def _calc_momentum(
        self, history: list[PriceBar]
    ) -> tuple[float, float]:
        """Calculate recent price momentum.

        Returns (price_change_pct, score 0–1).
        """
        if len(history) < 2:
            return (0.0, 0.0)

        recent = history[-min(5, len(history)):]
        start_price = recent[0].open
        end_price = recent[-1].close

        if start_price <= 0:
            return (0.0, 0.0)

        change = (end_price - start_price) / start_price

        abs_change = abs(change)
        if abs_change < self.momentum_threshold:
            return (change, 0.0)

        # Score scales from 0 at threshold to 1 at 5x threshold
        score = min(1.0, (abs_change - self.momentum_threshold) / (self.momentum_threshold * 4))
        return (change, score)

    def _calc_smart_money(
        self, recent_trades: list[Trade]
    ) -> tuple[int, float]:
        """Detect whale trades (outlier-size trades).

        Returns (whale_count, score 0–1).
        """
        if len(recent_trades) < 3:
            return (0, 0.0)

        sizes = [t.quantity for t in recent_trades if t.quantity > 0]
        if not sizes:
            return (0, 0.0)

        avg_size = statistics.mean(sizes)
        if avg_size <= 0:
            return (0, 0.0)

        threshold = avg_size * self.whale_size_multiplier
        whales = [s for s in sizes if s >= threshold]
        whale_count = len(whales)

        if whale_count == 0:
            return (0, 0.0)

        # Score based on whale trade frequency relative to total
        score = min(1.0, whale_count / max(1, len(sizes) * 0.1))
        return (whale_count, score)

    def _calc_book_imbalance(
        self, orderbook: Optional[OrderBook]
    ) -> tuple[float, float]:
        """Calculate bid/ask volume imbalance.

        Returns (imbalance -1 to +1, score 0–1).
        Positive = buy pressure, Negative = sell pressure.
        """
        if orderbook is None:
            return (0.0, 0.0)

        bid_vol = sum(lvl.quantity for lvl in orderbook.bids)
        ask_vol = sum(lvl.quantity for lvl in orderbook.asks)
        total = bid_vol + ask_vol

        if total <= 0:
            return (0.0, 0.0)

        imbalance = (bid_vol - ask_vol) / total  # -1 to +1

        abs_imb = abs(imbalance)
        if abs_imb < self.book_imbalance_threshold:
            return (imbalance, 0.0)

        score = min(1.0, (abs_imb - self.book_imbalance_threshold) / (1.0 - self.book_imbalance_threshold))
        return (imbalance, score)

    def _determine_direction(
        self,
        price_change: float,
        book_imbalance: float,
        recent_trades: list[Trade],
    ) -> str:
        """Determine overall direction of unusual activity."""
        signals = 0.0

        # Price momentum direction
        if abs(price_change) > 0.01:
            signals += 1.0 if price_change > 0 else -1.0

        # Book imbalance direction
        if abs(book_imbalance) > 0.1:
            signals += 1.0 if book_imbalance > 0 else -1.0

        # Recent trade flow direction
        if recent_trades:
            from prediction_bot.models import Side
            buys = sum(1 for t in recent_trades if t.side == Side.BUY)
            sells = len(recent_trades) - buys
            if buys > sells * 1.3:
                signals += 1.0
            elif sells > buys * 1.3:
                signals -= 1.0

        if signals >= 1.5:
            return "BULLISH"
        elif signals <= -1.5:
            return "BEARISH"
        return "NEUTRAL"
