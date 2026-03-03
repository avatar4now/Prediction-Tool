"""Trading session runner.

Orchestrates the scan → signal → risk-check → execute loop.
PAPER TRADING ONLY.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from prediction_bot.config import get_nested
from prediction_bot.db.store import Store
from prediction_bot.engine.portfolio_manager import PortfolioManager
from prediction_bot.engine.risk import RiskManager
from prediction_bot.models import Market, PriceBar, Provider, Side, Signal
from prediction_bot.providers.base import MarketProvider
from prediction_bot.strategies.base import Strategy

logger = logging.getLogger(__name__)


class TradingRunner:
    """Runs a paper trading session: scan markets, generate signals, execute."""

    def __init__(
        self,
        providers: list[MarketProvider],
        strategy: Strategy,
        portfolio: PortfolioManager,
        risk: RiskManager,
        config: dict[str, Any],
    ):
        self.providers = {p.name: p for p in providers}
        self.strategy = strategy
        self.portfolio = portfolio
        self.risk = risk
        self.config = config

    async def scan_markets(self, limit: int = 100) -> list[Market]:
        """Fetch active markets from all configured providers."""
        all_markets: list[Market] = []
        for name, provider in self.providers.items():
            try:
                markets = await provider.get_markets(limit=limit)
                all_markets.extend(markets)
                logger.info("Fetched %d markets from %s", len(markets), name)
            except Exception as e:
                logger.error("Failed to fetch markets from %s: %s", name, e)
        return all_markets

    async def fetch_history(
        self, markets: list[Market]
    ) -> dict[str, list[PriceBar]]:
        """Fetch price history for a list of markets."""
        history: dict[str, list[PriceBar]] = {}

        async def _fetch_one(market: Market) -> None:
            provider = self.providers.get(market.provider.value)
            if provider is None:
                return
            try:
                bars = await provider.get_historical_prices(market.market_id)
                history[market.market_id] = bars
            except Exception as e:
                logger.debug(
                    "No history for %s: %s", market.market_id, e
                )

        # Fetch in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]
            await asyncio.gather(*[_fetch_one(m) for m in batch])

        return history

    async def run_once(self) -> list[dict]:
        """Run a single trading cycle: scan → signals → risk → execute.

        Returns a list of action dicts for reporting.
        """
        actions: list[dict] = []

        # 1. Scan markets
        markets = await self.scan_markets()
        if not markets:
            logger.warning("No markets found")
            return actions

        # 2. Fetch price history for signal generation
        history = await self.fetch_history(markets)

        # 3. Generate signals from strategy
        strategy_params = get_nested(
            self.config, "strategies", self.strategy.name, default={}
        )
        # Inject default_quantity from trading config
        strategy_params["default_quantity"] = get_nested(
            self.config, "trading", "default_quantity", default=10
        )
        signals = self.strategy.generate_signals(markets, history, strategy_params)
        logger.info(
            "Strategy '%s' generated %d signals", self.strategy.name, len(signals)
        )

        # 4. Check risk rules and execute
        max_orders = get_nested(self.config, "trading", "max_orders_per_scan", default=5)
        slippage = get_nested(self.config, "trading", "simulated_fill_slippage", default=0.01)
        portfolio = self.portfolio.get_portfolio()

        executed = 0
        for signal in signals:
            if executed >= max_orders:
                break

            # Risk check
            allowed, reason = self.risk.check_signal(signal, portfolio)
            if not allowed:
                logger.info("Signal rejected by risk: %s — %s", signal.market_id, reason)
                actions.append({
                    "type": "rejected",
                    "market": signal.market_id,
                    "reason": reason,
                    "signal": signal,
                })
                continue

            # Adjust quantity if needed
            adjusted_qty = self.risk.adjust_quantity(signal, portfolio)
            if adjusted_qty <= 0:
                logger.info("Signal quantity adjusted to 0: %s", signal.market_id)
                continue
            signal.quantity = adjusted_qty

            # Execute paper trade
            order = self.portfolio.place_order_from_signal(signal, slippage=slippage)
            actions.append({
                "type": "executed",
                "market": signal.market_id,
                "order": order,
                "signal": signal,
            })
            executed += 1

            # Refresh portfolio snapshot for next risk check
            portfolio = self.portfolio.get_portfolio()

        # 5. Check stop-losses on existing positions
        open_positions = self.portfolio._store.get_open_positions()
        stop_losses = self.risk.check_stop_losses(open_positions)
        for pos, reason in stop_losses:
            order = self.portfolio.close_position(
                pos.position_id, pos.current_price, reason=reason
            )
            if order:
                actions.append({
                    "type": "stop_loss",
                    "market": pos.market_id,
                    "order": order,
                    "reason": reason,
                })

        # 6. Check expiry warnings
        market_map = {m.market_id: m for m in markets}
        expiry_closes = self.risk.check_expiry(open_positions, market_map)
        for pos, reason in expiry_closes:
            order = self.portfolio.close_position(
                pos.position_id, pos.current_price, reason=reason
            )
            if order:
                actions.append({
                    "type": "expiry_close",
                    "market": pos.market_id,
                    "order": order,
                    "reason": reason,
                })

        # 7. Check pending limit orders
        current_prices = {m.market_id: m.yes_price for m in markets}
        filled_limits = self.portfolio.check_limit_orders(current_prices)
        for order in filled_limits:
            actions.append({
                "type": "limit_fill",
                "market": order.market_id,
                "order": order,
            })

        # 8. Take portfolio snapshot
        self.portfolio.take_snapshot()

        return actions

    async def run_session(self, cycles: int = 1, interval_sec: int = 60) -> None:
        """Run multiple trading cycles with delay between them."""
        scan_interval = get_nested(
            self.config, "trading", "scan_interval_sec", default=interval_sec
        )

        for i in range(cycles):
            logger.info("=== Trading cycle %d/%d ===", i + 1, cycles)
            try:
                actions = await self.run_once()
                self._log_cycle_summary(i + 1, actions)
            except Exception as e:
                logger.error("Cycle %d failed: %s", i + 1, e)

            if i < cycles - 1:
                logger.info("Waiting %ds before next cycle...", scan_interval)
                await asyncio.sleep(scan_interval)

    def _log_cycle_summary(self, cycle: int, actions: list[dict]) -> None:
        executed = [a for a in actions if a["type"] == "executed"]
        rejected = [a for a in actions if a["type"] == "rejected"]
        stops = [a for a in actions if a["type"] == "stop_loss"]
        portfolio = self.portfolio.get_portfolio()

        logger.info(
            "Cycle %d complete: %d executed, %d rejected, %d stop-losses | "
            "Cash: $%.2f | Portfolio: $%.2f | Exposure: %.1f%%",
            cycle,
            len(executed),
            len(rejected),
            len(stops),
            portfolio.cash,
            portfolio.total_value,
            portfolio.exposure_pct * 100,
        )
