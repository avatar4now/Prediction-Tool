"""Paper trading portfolio manager.

Manages simulated portfolio: cash, positions, order execution, P&L tracking.
PAPER TRADING ONLY — no real money ever leaves this system.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from prediction_bot.db.store import Store
from prediction_bot.models import (
    Order,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionStatus,
    Provider,
    Side,
    Signal,
)

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Simulated paper trading portfolio."""

    def __init__(self, store: Store, starting_capital: float = 10000.0):
        self._store = store
        self._starting_capital = starting_capital
        self._init_cash()

    def _init_cash(self) -> None:
        """Initialize cash balance if not already set."""
        stored = self._store.get_state("cash")
        if stored is None:
            self._store.set_state("cash", str(self._starting_capital))
            self._store.set_state("total_trades", "0")
            self._store.set_state("winning_trades", "0")

    @property
    def cash(self) -> float:
        return float(self._store.get_state("cash", str(self._starting_capital)))

    @cash.setter
    def cash(self, value: float) -> None:
        self._store.set_state("cash", str(value))

    @property
    def total_trades(self) -> int:
        return int(self._store.get_state("total_trades", "0"))

    @total_trades.setter
    def total_trades(self, value: int) -> None:
        self._store.set_state("total_trades", str(value))

    @property
    def winning_trades(self) -> int:
        return int(self._store.get_state("winning_trades", "0"))

    @winning_trades.setter
    def winning_trades(self, value: int) -> None:
        self._store.set_state("winning_trades", str(value))

    # ── Portfolio snapshot ──────────────────────────────────────────

    def get_portfolio(self) -> Portfolio:
        positions = self._store.get_open_positions()
        return Portfolio(
            cash=self.cash,
            positions=positions,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
        )

    # ── Order placement (paper only) ───────────────────────────────

    def place_order(
        self,
        market_id: str,
        provider: Provider,
        side: Side,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        reason: str = "",
        market_title: str = "",
        slippage: float = 0.01,
    ) -> Order:
        """Place a paper trade order.

        For market orders, fills immediately with simulated slippage.
        For limit orders, creates a pending order to be checked later.

        Returns the Order (filled or pending).
        """
        order = Order(
            order_id=str(uuid.uuid4())[:12],
            market_id=market_id,
            provider=provider,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            reason=reason,
            market_title=market_title,
        )

        if order_type == OrderType.MARKET:
            # Immediate simulated fill with slippage
            fill_price = self._apply_slippage(price, side, slippage)
            order = self._execute_fill(order, fill_price, quantity)
        else:
            # Limit order — store as pending
            order.status = OrderStatus.PENDING
            self._store.save_order(order)
            logger.info(
                "Limit order placed: %s %s %d @ %.4f on %s (%s)",
                side.value, market_id, quantity, price, provider.value, reason,
            )

        return order

    def place_order_from_signal(
        self, signal: Signal, slippage: float = 0.01
    ) -> Order:
        """Convert a strategy signal into a paper trade order."""
        return self.place_order(
            market_id=signal.market_id,
            provider=signal.provider,
            side=signal.side,
            quantity=signal.quantity,
            price=signal.price,
            reason=signal.reason,
            market_title=signal.market_title,
            slippage=slippage,
        )

    # ── Limit order checking ───────────────────────────────────────

    def check_limit_orders(self, current_prices: dict[str, float]) -> list[Order]:
        """Check pending limit orders against current prices and fill if matched.

        current_prices: {market_id: current_yes_price}
        """
        pending = self._store.get_orders(status=OrderStatus.PENDING)
        filled: list[Order] = []

        for order in pending:
            price = current_prices.get(order.market_id)
            if price is None:
                continue

            should_fill = False
            if order.side == Side.BUY and price <= order.price:
                should_fill = True
            elif order.side == Side.SELL and price >= order.price:
                should_fill = True

            if should_fill:
                order = self._execute_fill(order, order.price, order.quantity)
                filled.append(order)

        return filled

    # ── Position closing ───────────────────────────────────────────

    def close_position(
        self, position_id: str, current_price: float, reason: str = ""
    ) -> Optional[Order]:
        """Close an open position at the current price."""
        pos = self._store.get_position(position_id)
        if pos is None or pos.status == PositionStatus.CLOSED:
            return None

        # Create a closing order (opposite side)
        close_side = Side.SELL if pos.side == Side.BUY else Side.BUY
        order = self.place_order(
            market_id=pos.market_id,
            provider=pos.provider,
            side=close_side,
            quantity=pos.quantity,
            price=current_price,
            reason=reason or "position_close",
            market_title=pos.market_title,
            slippage=0.005,
        )
        return order

    # ── Internal execution ──────────────────────────────────────────

    def _execute_fill(
        self, order: Order, fill_price: float, fill_qty: int
    ) -> Order:
        """Simulate order fill and update portfolio state."""
        cost = fill_price * fill_qty

        if order.side == Side.BUY:
            if cost > self.cash:
                logger.warning(
                    "Insufficient cash: need %.2f, have %.2f", cost, self.cash
                )
                order.status = OrderStatus.CANCELLED
                order.reason += " [insufficient cash]"
                self._store.save_order(order)
                return order

            self.cash -= cost
            self._update_or_create_position(order, fill_price, fill_qty)

        elif order.side == Side.SELL:
            # Check if we have a position to sell
            pos = self._store.get_position_by_market(
                order.market_id, order.provider
            )
            if pos and pos.status == PositionStatus.OPEN:
                # Closing an existing long position
                pnl = (fill_price - pos.avg_entry_price) * min(fill_qty, pos.quantity)
                self.cash += fill_price * min(fill_qty, pos.quantity)

                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1

                remaining = pos.quantity - fill_qty
                if remaining <= 0:
                    pos.status = PositionStatus.CLOSED
                    pos.closed_at = datetime.now(tz=timezone.utc)
                    pos.realized_pnl = pnl
                    pos.quantity = 0
                else:
                    pos.quantity = remaining

                pos.current_price = fill_price
                self._store.save_position(pos)
            else:
                # Short selling (opening a short position)
                self.cash += cost
                self._update_or_create_position(order, fill_price, fill_qty)

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = fill_qty
        order.filled_at = datetime.now(tz=timezone.utc)
        self._store.save_order(order)

        logger.info(
            "FILLED: %s %s %d @ %.4f on %s | Cash: $%.2f | Reason: %s",
            order.side.value,
            order.market_id,
            fill_qty,
            fill_price,
            order.provider.value,
            self.cash,
            order.reason,
        )
        return order

    def _update_or_create_position(
        self, order: Order, fill_price: float, fill_qty: int
    ) -> None:
        """Update existing position or create a new one."""
        existing = self._store.get_position_by_market(
            order.market_id, order.provider
        )

        if existing and existing.status == PositionStatus.OPEN and existing.side == order.side:
            # Average into existing position
            total_qty = existing.quantity + fill_qty
            existing.avg_entry_price = (
                (existing.avg_entry_price * existing.quantity)
                + (fill_price * fill_qty)
            ) / total_qty
            existing.quantity = total_qty
            existing.current_price = fill_price
            self._store.save_position(existing)
        else:
            # New position
            pos = Position(
                position_id=str(uuid.uuid4())[:12],
                market_id=order.market_id,
                provider=order.provider,
                side=order.side,
                quantity=fill_qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
                market_title=order.market_title,
            )
            self._store.save_position(pos)

    @staticmethod
    def _apply_slippage(price: float, side: Side, slippage: float) -> float:
        """Simulate market impact / slippage."""
        if side == Side.BUY:
            return min(price + slippage, 0.99)
        else:
            return max(price - slippage, 0.01)

    # ── Snapshot ───────────────────────────────────────────────────

    def take_snapshot(self) -> None:
        """Record current portfolio state for equity curve tracking."""
        portfolio = self.get_portfolio()
        self._store.save_snapshot(portfolio)

    # ── Reset ──────────────────────────────────────────────────────

    def reset(self, starting_capital: Optional[float] = None) -> None:
        """Reset paper portfolio to starting state."""
        capital = starting_capital or self._starting_capital
        self._store.set_state("cash", str(capital))
        self._store.set_state("total_trades", "0")
        self._store.set_state("winning_trades", "0")
        logger.info("Portfolio reset to $%.2f", capital)
