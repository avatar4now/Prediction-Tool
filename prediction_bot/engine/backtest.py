"""Backtesting engine for prediction market strategies.

Replays historical price data through a strategy and simulated portfolio
to evaluate performance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from prediction_bot.engine.portfolio_manager import PortfolioManager
from prediction_bot.engine.risk import RiskManager
from prediction_bot.db.store import Store
from prediction_bot.models import (
    Market,
    OrderType,
    PriceBar,
    Provider,
    Side,
)
from prediction_bot.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy: str
    market_id: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    starting_capital: float = 10000.0
    ending_capital: float = 10000.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trades_log: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Backtest: {self.strategy} on {self.market_id}\n"
            f"Period: {self.start_date} → {self.end_date}\n"
            f"Return: {self.total_return_pct:+.2%}\n"
            f"Trades: {self.total_trades} (win rate: {self.win_rate:.1%})\n"
            f"Sharpe: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown_pct:.2%}\n"
            f"Final Capital: ${self.ending_capital:,.2f}"
        )


def run_backtest(
    strategy: Strategy,
    bars: list[PriceBar],
    market_id: str,
    provider: Provider = Provider.KALSHI,
    starting_capital: float = 10000.0,
    params: dict[str, Any] | None = None,
    risk_params: dict[str, Any] | None = None,
) -> BacktestResult:
    """Run a backtest of a strategy on historical price bars.

    Simulates bar-by-bar execution with a fresh paper portfolio.
    """
    if not bars:
        return BacktestResult(strategy=strategy.name, market_id=market_id)

    params = params or {}
    risk_params = risk_params or {}

    # Use in-memory SQLite for backtest
    store = Store(":memory:")
    portfolio = PortfolioManager(store, starting_capital=starting_capital)
    risk = RiskManager(**risk_params)

    lookback = max(params.get("lookback_periods", 10), 5)
    equity_curve: list[tuple[datetime, float]] = []
    trades_log: list[dict] = []

    # Walk forward through bars
    for i in range(lookback, len(bars)):
        window = bars[:i + 1]
        current_bar = bars[i]

        # Build a synthetic Market from the current bar
        market = Market(
            market_id=market_id,
            provider=provider,
            title=market_id,
            category="backtest",
            yes_price=current_bar.close,
            no_price=1.0 - current_bar.close,
            volume=current_bar.volume,
            open_interest=0,
            expiry=None,
            status="open",
        )

        # Generate signals
        history = {market_id: window}
        signals = strategy.generate_signals([market], history, params)

        # Process signals
        port = portfolio.get_portfolio()
        for signal in signals[:1]:  # One signal per bar max
            allowed, reason = risk.check_signal(signal, port)
            if not allowed:
                continue

            qty = risk.adjust_quantity(signal, port)
            if qty <= 0:
                continue

            signal.quantity = qty
            order = portfolio.place_order_from_signal(signal, slippage=0.005)
            trades_log.append({
                "timestamp": current_bar.timestamp,
                "side": order.side.value,
                "price": order.filled_price or order.price,
                "quantity": order.filled_quantity or order.quantity,
                "reason": order.reason,
            })

        # Update position prices
        for pos in store.get_open_positions():
            pos.current_price = current_bar.close
            store.save_position(pos)

        # Check stop-losses
        open_pos = store.get_open_positions()
        for pos, reason in risk.check_stop_losses(open_pos):
            order = portfolio.close_position(
                pos.position_id, current_bar.close, reason=reason
            )
            if order:
                trades_log.append({
                    "timestamp": current_bar.timestamp,
                    "side": order.side.value,
                    "price": order.filled_price or order.price,
                    "quantity": order.filled_quantity or order.quantity,
                    "reason": reason,
                })

        # Record equity
        port = portfolio.get_portfolio()
        equity_curve.append((current_bar.timestamp, port.total_value))

    # Calculate metrics
    final_port = portfolio.get_portfolio()
    result = BacktestResult(
        strategy=strategy.name,
        market_id=market_id,
        start_date=bars[0].timestamp if bars else None,
        end_date=bars[-1].timestamp if bars else None,
        starting_capital=starting_capital,
        ending_capital=final_port.total_value,
        total_return_pct=(final_port.total_value - starting_capital) / starting_capital,
        total_trades=final_port.total_trades,
        winning_trades=final_port.winning_trades,
        win_rate=final_port.win_rate,
        sharpe_ratio=_calc_sharpe(equity_curve),
        max_drawdown_pct=_calc_max_drawdown(equity_curve),
        equity_curve=equity_curve,
        trades_log=trades_log,
    )

    store.close()
    return result


def _calc_sharpe(equity_curve: list[tuple[datetime, float]]) -> float:
    """Calculate annualized Sharpe ratio from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    values = [v for _, v in equity_curve]
    returns = [(values[i] - values[i - 1]) / values[i - 1]
               for i in range(1, len(values)) if values[i - 1] != 0]

    if not returns:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
    std_ret = math.sqrt(variance) if variance > 0 else 0.001

    # Annualize (assuming hourly bars → ~8760 per year)
    annualization = math.sqrt(min(len(returns), 8760))
    return (mean_ret / std_ret) * annualization


def _calc_max_drawdown(equity_curve: list[tuple[datetime, float]]) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    values = [v for _, v in equity_curve]
    peak = values[0]
    max_dd = 0.0

    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return max_dd
