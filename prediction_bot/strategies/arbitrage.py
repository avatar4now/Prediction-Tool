"""Cross-platform arbitrage strategy.

Detects price discrepancies between Kalshi and Polymarket for the same
underlying event.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from prediction_bot.models import Market, PriceBar, Provider, Side, Signal
from prediction_bot.strategies.base import Strategy

logger = logging.getLogger(__name__)


class ArbitrageStrategy(Strategy):
    """Cross-platform arbitrage between Kalshi and Polymarket."""

    @property
    def name(self) -> str:
        return "arbitrage"

    def generate_signals(
        self,
        markets: list[Market],
        history: dict[str, list[PriceBar]],
        params: dict[str, Any],
    ) -> list[Signal]:
        min_spread = params.get("min_spread_pct", 0.02)
        match_confidence = params.get("match_confidence", 0.8)
        default_qty = params.get("default_quantity", 10)

        # Separate markets by provider
        kalshi_markets = [m for m in markets if m.provider == Provider.KALSHI]
        poly_markets = [m for m in markets if m.provider == Provider.POLYMARKET]

        if not kalshi_markets or not poly_markets:
            logger.info("Arbitrage requires markets from both Kalshi and Polymarket")
            return []

        signals: list[Signal] = []

        # Try to match markets across platforms
        for km in kalshi_markets:
            best_match = self._find_best_match(km, poly_markets, match_confidence)
            if best_match is None:
                continue

            pm, confidence = best_match
            spread = km.yes_price - pm.yes_price

            if abs(spread) < min_spread:
                continue

            # Buy the cheaper one, sell the more expensive one
            if spread > 0:
                # Kalshi more expensive — buy Polymarket, signal sell Kalshi
                signals.append(
                    Signal(
                        market_id=pm.market_id,
                        provider=pm.provider,
                        side=Side.BUY,
                        strength=min(1.0, abs(spread) / min_spread / 2),
                        price=pm.yes_price,
                        quantity=default_qty,
                        reason=(
                            f"arb_buy: {pm.provider.value} @ {pm.yes_price:.2f} vs "
                            f"{km.provider.value} @ {km.yes_price:.2f}, "
                            f"spread={spread:+.2f}, match={confidence:.0%}"
                        ),
                        strategy=self.name,
                        market_title=pm.title,
                    )
                )
                signals.append(
                    Signal(
                        market_id=km.market_id,
                        provider=km.provider,
                        side=Side.SELL,
                        strength=min(1.0, abs(spread) / min_spread / 2),
                        price=km.yes_price,
                        quantity=default_qty,
                        reason=(
                            f"arb_sell: {km.provider.value} @ {km.yes_price:.2f} vs "
                            f"{pm.provider.value} @ {pm.yes_price:.2f}, "
                            f"spread={spread:+.2f}, match={confidence:.0%}"
                        ),
                        strategy=self.name,
                        market_title=km.title,
                    )
                )
            else:
                # Polymarket more expensive — buy Kalshi, signal sell Polymarket
                signals.append(
                    Signal(
                        market_id=km.market_id,
                        provider=km.provider,
                        side=Side.BUY,
                        strength=min(1.0, abs(spread) / min_spread / 2),
                        price=km.yes_price,
                        quantity=default_qty,
                        reason=(
                            f"arb_buy: {km.provider.value} @ {km.yes_price:.2f} vs "
                            f"{pm.provider.value} @ {pm.yes_price:.2f}, "
                            f"spread={spread:+.2f}, match={confidence:.0%}"
                        ),
                        strategy=self.name,
                        market_title=km.title,
                    )
                )
                signals.append(
                    Signal(
                        market_id=pm.market_id,
                        provider=pm.provider,
                        side=Side.SELL,
                        strength=min(1.0, abs(spread) / min_spread / 2),
                        price=pm.yes_price,
                        quantity=default_qty,
                        reason=(
                            f"arb_sell: {pm.provider.value} @ {pm.yes_price:.2f} vs "
                            f"{km.provider.value} @ {km.yes_price:.2f}, "
                            f"spread={spread:+.2f}, match={confidence:.0%}"
                        ),
                        strategy=self.name,
                        market_title=pm.title,
                    )
                )

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals

    def _find_best_match(
        self, target: Market, candidates: list[Market], min_confidence: float
    ) -> tuple[Market, float] | None:
        """Find the best matching market from candidates using title similarity."""
        best: tuple[Market, float] | None = None

        target_tokens = self._tokenize(target.title)
        if not target_tokens:
            return None

        for candidate in candidates:
            cand_tokens = self._tokenize(candidate.title)
            if not cand_tokens:
                continue

            # Jaccard similarity on title tokens
            intersection = target_tokens & cand_tokens
            union = target_tokens | cand_tokens
            similarity = len(intersection) / len(union) if union else 0.0

            if similarity >= min_confidence:
                if best is None or similarity > best[1]:
                    best = (candidate, similarity)

        return best

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract meaningful tokens from market title."""
        words = re.findall(r'\b\w{3,}\b', text.lower())
        stop_words = {
            "the", "will", "that", "this", "what", "which", "and",
            "for", "are", "was", "were", "been", "has", "have",
            "yes", "contract", "market",
        }
        return set(words) - stop_words
