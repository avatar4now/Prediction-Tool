"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from prediction_bot.models import Market, PriceBar, Signal


class Strategy(ABC):
    """Base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""

    @abstractmethod
    def generate_signals(
        self,
        markets: list[Market],
        history: dict[str, list[PriceBar]],
        params: dict[str, Any],
    ) -> list[Signal]:
        """Analyze markets and generate trading signals.

        Args:
            markets: List of active markets with current prices.
            history: Map of market_id -> price history bars.
            params: Strategy-specific parameters from config.

        Returns:
            List of Signal objects, sorted by strength (strongest first).
        """
