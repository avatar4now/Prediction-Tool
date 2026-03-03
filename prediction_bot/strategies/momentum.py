"""Momentum / trend-following strategy.

Buy contracts trending up in volume + price, sell when momentum reverses.
"""

from __future__ import annotations

import logging
from typing import Any

from prediction_bot.models import Market, PriceBar, Provider, Side, Signal
from prediction_bot.strategies.base import Strategy

logger = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    """Momentum-based trading strategy for prediction markets."""

    @property
    def name(self) -> str:
        return "momentum"

    def generate_signals(
        self,
        markets: list[Market],
        history: dict[str, list[PriceBar]],
        params: dict[str, Any],
    ) -> list[Signal]:
        lookback = params.get("lookback_periods", 10)
        vol_threshold = params.get("volume_threshold", 1.5)
        price_threshold = params.get("price_change_threshold", 0.03)
        exit_reversal = params.get("exit_reversal_pct", 0.02)
        default_qty = params.get("default_quantity", 10)

        signals: list[Signal] = []

        for market in markets:
            bars = history.get(market.market_id, [])
            if len(bars) < lookback + 1:
                continue

            recent = bars[-lookback:]
            older = bars[-(lookback * 2):-lookback] if len(bars) >= lookback * 2 else bars[:lookback]

            signal = self._analyze_market(
                market, recent, older,
                vol_threshold, price_threshold, exit_reversal, default_qty,
            )
            if signal is not None:
                signals.append(signal)

        # Sort by signal strength (strongest first)
        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals

    def _analyze_market(
        self,
        market: Market,
        recent_bars: list[PriceBar],
        older_bars: list[PriceBar],
        vol_threshold: float,
        price_threshold: float,
        exit_reversal: float,
        default_qty: int,
    ) -> Signal | None:
        if not recent_bars or not older_bars:
            return None

        # ── Price momentum ──────────────────────────────────────────
        recent_close = recent_bars[-1].close
        start_close = recent_bars[0].close
        if start_close == 0:
            return None

        price_change = (recent_close - start_close) / start_close

        # ── Volume momentum ─────────────────────────────────────────
        recent_vol = sum(b.volume for b in recent_bars) / len(recent_bars)
        older_vol = sum(b.volume for b in older_bars) / len(older_bars) if older_bars else 1
        if older_vol == 0:
            older_vol = 1
        vol_ratio = recent_vol / older_vol

        # ── Peak detection for exit signal ──────────────────────────
        peak_price = max(b.high for b in recent_bars)
        reversal_from_peak = (peak_price - recent_close) / peak_price if peak_price > 0 else 0

        # ── Generate BUY signal (upward momentum) ───────────────────
        if price_change > price_threshold and vol_ratio > vol_threshold:
            strength = min(1.0, (price_change / price_threshold + vol_ratio / vol_threshold) / 4)
            reason = (
                f"momentum_buy: price +{price_change:.1%} (>{price_threshold:.1%}), "
                f"volume {vol_ratio:.1f}x (>{vol_threshold:.1f}x)"
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

        # ── Generate SELL signal (momentum reversal from peak) ──────
        if reversal_from_peak > exit_reversal and price_change < 0:
            strength = min(1.0, reversal_from_peak / exit_reversal / 2)
            reason = (
                f"momentum_sell: {reversal_from_peak:.1%} reversal from peak "
                f"(>{exit_reversal:.1%}), price {price_change:+.1%}"
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
