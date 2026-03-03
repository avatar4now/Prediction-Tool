"""Risk management for paper trading.

Enforces position limits, exposure caps, stop-losses, and expiry awareness.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from prediction_bot.models import (
    Market,
    Portfolio,
    Position,
    PositionStatus,
    Signal,
    Side,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk rules before order placement."""

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_exposure_pct: float = 0.50,
        stop_loss_pct: float = 0.20,
        expiry_close_hours: int = 24,
    ):
        self.max_position_pct = max_position_pct
        self.max_exposure_pct = max_exposure_pct
        self.stop_loss_pct = stop_loss_pct
        self.expiry_close_hours = expiry_close_hours

    def check_signal(
        self, signal: Signal, portfolio: Portfolio
    ) -> tuple[bool, str]:
        """Check if a signal passes all risk rules.

        Returns (allowed, reason).
        """
        if signal.side == Side.SELL:
            # Sells (closing positions) are generally allowed
            return True, "sell_allowed"

        total_value = portfolio.total_value
        if total_value <= 0:
            return False, "portfolio_value_zero"

        # ── Max position size per market ────────────────────────────
        order_cost = signal.price * signal.quantity
        max_position = total_value * self.max_position_pct

        # Include existing exposure in this market
        existing_exposure = sum(
            p.current_price * p.quantity
            for p in portfolio.positions
            if p.market_id == signal.market_id
            and p.status == PositionStatus.OPEN
        )
        if existing_exposure + order_cost > max_position:
            return False, (
                f"position_limit: ${existing_exposure + order_cost:.2f} "
                f"would exceed {self.max_position_pct:.0%} cap (${max_position:.2f})"
            )

        # ── Max total exposure ──────────────────────────────────────
        max_exposure = total_value * self.max_exposure_pct
        current_exposure = portfolio.positions_value
        if current_exposure + order_cost > max_exposure:
            return False, (
                f"exposure_limit: ${current_exposure + order_cost:.2f} "
                f"would exceed {self.max_exposure_pct:.0%} cap (${max_exposure:.2f})"
            )

        # ── Cash check ──────────────────────────────────────────────
        if order_cost > portfolio.cash:
            return False, f"insufficient_cash: need ${order_cost:.2f}, have ${portfolio.cash:.2f}"

        return True, "approved"

    def check_stop_losses(
        self, positions: list[Position]
    ) -> list[tuple[Position, str]]:
        """Check positions against stop-loss threshold.

        Returns list of (position, reason) tuples for positions that should be closed.
        """
        to_close: list[tuple[Position, str]] = []
        for pos in positions:
            if pos.status != PositionStatus.OPEN:
                continue

            pnl_pct = pos.pnl_pct
            if pnl_pct < -self.stop_loss_pct:
                reason = (
                    f"stop_loss: {pnl_pct:.1%} loss exceeds "
                    f"-{self.stop_loss_pct:.0%} threshold"
                )
                to_close.append((pos, reason))
                logger.warning(
                    "STOP-LOSS triggered for %s: %.1f%% loss",
                    pos.market_id,
                    pnl_pct * 100,
                )

        return to_close

    def check_expiry(
        self,
        positions: list[Position],
        markets: dict[str, Market],
    ) -> list[tuple[Position, str]]:
        """Check positions for approaching market expiry.

        Returns list of (position, reason) tuples for positions that should be closed.
        """
        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=self.expiry_close_hours)
        to_close: list[tuple[Position, str]] = []

        for pos in positions:
            if pos.status != PositionStatus.OPEN:
                continue

            market = markets.get(pos.market_id)
            if market is None or market.expiry is None:
                continue

            expiry = market.expiry
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)

            if expiry <= cutoff:
                hours_left = max(0, (expiry - now).total_seconds() / 3600)
                reason = (
                    f"expiry_close: market expires in {hours_left:.1f}h "
                    f"(< {self.expiry_close_hours}h threshold)"
                )
                to_close.append((pos, reason))
                logger.info(
                    "EXPIRY close for %s: %.1fh until expiry",
                    pos.market_id,
                    hours_left,
                )

        return to_close

    def adjust_quantity(
        self, signal: Signal, portfolio: Portfolio
    ) -> int:
        """Reduce quantity to fit within risk limits if needed."""
        total_value = portfolio.total_value
        if total_value <= 0:
            return 0

        max_position = total_value * self.max_position_pct
        max_exposure = total_value * self.max_exposure_pct

        # Existing exposure in this market
        existing_in_market = sum(
            p.current_price * p.quantity
            for p in portfolio.positions
            if p.market_id == signal.market_id
            and p.status == PositionStatus.OPEN
        )
        available_for_market = max(0, max_position - existing_in_market)

        # Available from total exposure limit
        current_exposure = portfolio.positions_value
        available_from_exposure = max(0, max_exposure - current_exposure)

        # Available from cash
        available_cash = portfolio.cash

        # Most constraining limit
        max_cost = min(available_for_market, available_from_exposure, available_cash)
        if signal.price <= 0:
            return 0

        max_qty = int(max_cost / signal.price)
        return min(signal.quantity, max_qty)
