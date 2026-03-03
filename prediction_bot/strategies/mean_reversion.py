"""Mean reversion strategy.

Buy contracts that spike down on low volume (panic selling), sell on recovery.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from prediction_bot.models import Market, PriceBar, Side, Signal
from prediction_bot.strategies.base import Strategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy for prediction markets."""

    @property
    def name(self) -> str:
        return "mean_reversion"

    def generate_signals(
        self,
        markets: list[Market],
        history: dict[str, list[PriceBar]],
        params: dict[str, Any],
    ) -> list[Signal]:
        lookback = params.get("lookback_periods", 20)
        std_entry = params.get("std_dev_entry", 2.0)
        std_exit = params.get("std_dev_exit", 0.5)
        min_vol_ratio = params.get("min_volume_ratio", 0.3)
        default_qty = params.get("default_quantity", 10)

        signals: list[Signal] = []

        for market in markets:
            bars = history.get(market.market_id, [])
            if len(bars) < lookback:
                continue

            window = bars[-lookback:]
            signal = self._analyze(
                market, window, std_entry, std_exit, min_vol_ratio, default_qty
            )
            if signal is not None:
                signals.append(signal)

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals

    def _analyze(
        self,
        market: Market,
        bars: list[PriceBar],
        std_entry: float,
        std_exit: float,
        min_vol_ratio: float,
        default_qty: int,
    ) -> Signal | None:
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        mean_price = sum(closes) / len(closes)
        variance = sum((c - mean_price) ** 2 for c in closes) / len(closes)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001

        current_price = closes[-1]
        z_score = (current_price - mean_price) / std_dev

        # Volume analysis: is current volume low (panic selling)?
        mean_vol = sum(volumes) / len(volumes) if volumes else 1
        current_vol = volumes[-1] if volumes else 0
        vol_ratio = current_vol / mean_vol if mean_vol > 0 else 1.0

        # ── BUY signal: price spiked down on low volume ─────────────
        if z_score < -std_entry and vol_ratio < min_vol_ratio:
            strength = min(1.0, abs(z_score) / (std_entry * 2))
            reason = (
                f"mean_reversion_buy: z-score {z_score:.2f} "
                f"(< -{std_entry:.1f}σ), volume ratio {vol_ratio:.2f} "
                f"(< {min_vol_ratio:.1f} = low vol panic)"
            )
            return Signal(
                market_id=market.market_id,
                provider=market.provider,
                side=Side.BUY,
                strength=strength,
                price=market.yes_price,
                quantity=default_qty,
                reason=reason,
                strategy=self.name,
                market_title=market.title,
            )

        # ── SELL signal: price reverted back to mean ────────────────
        if abs(z_score) < std_exit:
            strength = min(1.0, (std_exit - abs(z_score)) / std_exit)
            reason = (
                f"mean_reversion_sell: z-score {z_score:.2f} "
                f"(within ±{std_exit:.1f}σ of mean = reverted)"
            )
            return Signal(
                market_id=market.market_id,
                provider=market.provider,
                side=Side.SELL,
                strength=strength,
                price=market.yes_price,
                quantity=default_qty,
                reason=reason,
                strategy=self.name,
                market_title=market.title,
            )

        return None
