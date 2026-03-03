"""Tests for the paper trading engine core."""

import pytest
from datetime import datetime, timezone

from prediction_bot.db.store import Store
from prediction_bot.engine.portfolio_manager import PortfolioManager
from prediction_bot.engine.risk import RiskManager
from prediction_bot.engine.backtest import run_backtest, _calc_sharpe, _calc_max_drawdown
from prediction_bot.models import (
    Market,
    OrderStatus,
    OrderType,
    PriceBar,
    Portfolio,
    Position,
    PositionStatus,
    Provider,
    Side,
    Signal,
)
from prediction_bot.strategies.momentum import MomentumStrategy
from prediction_bot.strategies.mean_reversion import MeanReversionStrategy


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def portfolio(store):
    return PortfolioManager(store, starting_capital=10000.0)


@pytest.fixture
def risk():
    return RiskManager(
        max_position_pct=0.05,
        max_exposure_pct=0.50,
        stop_loss_pct=0.20,
        expiry_close_hours=24,
    )


class TestPortfolioManager:
    def test_initial_cash(self, portfolio):
        assert portfolio.cash == 10000.0

    def test_place_market_order_buy(self, portfolio):
        order = portfolio.place_order(
            market_id="TEST-MKT",
            provider=Provider.KALSHI,
            side=Side.BUY,
            quantity=10,
            price=0.50,
            reason="test buy",
            market_title="Test Market",
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10
        assert portfolio.cash < 10000.0

    def test_place_market_order_insufficient_cash(self, portfolio):
        order = portfolio.place_order(
            market_id="TEST-MKT",
            provider=Provider.KALSHI,
            side=Side.BUY,
            quantity=100000,
            price=0.90,
            reason="too expensive",
        )
        assert order.status == OrderStatus.CANCELLED

    def test_place_limit_order(self, portfolio):
        order = portfolio.place_order(
            market_id="TEST-MKT",
            provider=Provider.KALSHI,
            side=Side.BUY,
            quantity=10,
            price=0.40,
            order_type=OrderType.LIMIT,
            reason="test limit",
        )
        assert order.status == OrderStatus.PENDING

    def test_check_limit_orders_fills(self, portfolio):
        portfolio.place_order(
            market_id="TEST-MKT",
            provider=Provider.KALSHI,
            side=Side.BUY,
            quantity=10,
            price=0.40,
            order_type=OrderType.LIMIT,
        )
        # Price drops to 0.35 — should fill
        filled = portfolio.check_limit_orders({"TEST-MKT": 0.35})
        assert len(filled) == 1
        assert filled[0].status == OrderStatus.FILLED

    def test_position_created_on_buy(self, portfolio, store):
        portfolio.place_order(
            market_id="TEST-MKT",
            provider=Provider.KALSHI,
            side=Side.BUY,
            quantity=10,
            price=0.50,
            market_title="Test",
        )
        positions = store.get_open_positions()
        assert len(positions) == 1
        assert positions[0].side == Side.BUY
        assert positions[0].quantity == 10

    def test_position_closed_on_sell(self, portfolio, store):
        portfolio.place_order(
            market_id="TEST-MKT",
            provider=Provider.KALSHI,
            side=Side.BUY,
            quantity=10,
            price=0.50,
            market_title="Test",
        )
        positions = store.get_open_positions()
        portfolio.close_position(positions[0].position_id, 0.60)
        positions = store.get_open_positions()
        assert len(positions) == 0

    def test_get_portfolio(self, portfolio):
        p = portfolio.get_portfolio()
        assert p.cash == 10000.0
        assert p.total_value == 10000.0
        assert p.positions == []

    def test_snapshot(self, portfolio, store):
        portfolio.take_snapshot()
        snaps = store.get_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["total_value"] == 10000.0

    def test_reset(self, portfolio):
        portfolio.place_order(
            market_id="TEST", provider=Provider.KALSHI,
            side=Side.BUY, quantity=10, price=0.50,
        )
        portfolio.reset(starting_capital=5000.0)
        assert portfolio.cash == 5000.0
        assert portfolio.total_trades == 0


class TestRiskManager:
    def test_buy_approved_within_limits(self, risk):
        signal = Signal(
            market_id="MKT", provider=Provider.KALSHI,
            side=Side.BUY, strength=0.8, price=0.10,
            quantity=10, reason="test", strategy="momentum",
        )
        portfolio = Portfolio(cash=10000.0)
        allowed, reason = risk.check_signal(signal, portfolio)
        assert allowed
        assert reason == "approved"

    def test_buy_rejected_position_limit(self, risk):
        # 5% of 10000 = 500. Buying 600 contracts at $1 each exceeds limit.
        signal = Signal(
            market_id="MKT", provider=Provider.KALSHI,
            side=Side.BUY, strength=0.8, price=1.0,
            quantity=600, reason="test", strategy="momentum",
        )
        portfolio = Portfolio(cash=10000.0)
        allowed, reason = risk.check_signal(signal, portfolio)
        assert not allowed
        assert "position_limit" in reason

    def test_sell_always_allowed(self, risk):
        signal = Signal(
            market_id="MKT", provider=Provider.KALSHI,
            side=Side.SELL, strength=0.5, price=0.50,
            quantity=10, reason="test", strategy="momentum",
        )
        portfolio = Portfolio(cash=0.0)
        allowed, _ = risk.check_signal(signal, portfolio)
        assert allowed

    def test_stop_loss_triggered(self, risk):
        pos = Position(
            position_id="pos1", market_id="MKT",
            provider=Provider.KALSHI, side=Side.BUY,
            quantity=10, avg_entry_price=0.50,
            current_price=0.35,  # 30% loss > 20% threshold
        )
        to_close = risk.check_stop_losses([pos])
        assert len(to_close) == 1

    def test_stop_loss_not_triggered(self, risk):
        pos = Position(
            position_id="pos1", market_id="MKT",
            provider=Provider.KALSHI, side=Side.BUY,
            quantity=10, avg_entry_price=0.50,
            current_price=0.45,  # 10% loss < 20% threshold
        )
        to_close = risk.check_stop_losses([pos])
        assert len(to_close) == 0

    def test_expiry_close(self, risk):
        pos = Position(
            position_id="pos1", market_id="MKT",
            provider=Provider.KALSHI, side=Side.BUY,
            quantity=10, avg_entry_price=0.50,
            current_price=0.50,
        )
        from datetime import timedelta
        soon = datetime.now(tz=timezone.utc) + timedelta(hours=12)
        market = Market(
            market_id="MKT", provider=Provider.KALSHI,
            title="Test", category="test", yes_price=0.50,
            no_price=0.50, volume=100, open_interest=50,
            expiry=soon, status="open",
        )
        to_close = risk.check_expiry([pos], {"MKT": market})
        assert len(to_close) == 1

    def test_adjust_quantity(self, risk):
        signal = Signal(
            market_id="MKT", provider=Provider.KALSHI,
            side=Side.BUY, strength=0.8, price=0.50,
            quantity=10000, reason="test", strategy="momentum",
        )
        portfolio = Portfolio(cash=10000.0)
        qty = risk.adjust_quantity(signal, portfolio)
        # Max position = 5% of 10000 = 500, at $0.50 = 1000 contracts max
        assert qty == 1000


class TestMomentumStrategy:
    def _make_bars(self, prices, volumes=None):
        bars = []
        for i, p in enumerate(prices):
            bars.append(PriceBar(
                timestamp=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                open=p, high=p + 0.01, low=p - 0.01, close=p,
                volume=volumes[i] if volumes else 100,
            ))
        return bars

    def test_generates_buy_signal_on_uptrend(self):
        strat = MomentumStrategy()
        # Create upward trending prices with high volume
        prices = [0.40 + i * 0.005 for i in range(25)]
        volumes = [100] * 12 + [200] * 13  # Recent volume 2x older
        bars = self._make_bars(prices, volumes)

        market = Market(
            market_id="TREND-UP", provider=Provider.KALSHI,
            title="Trending Up", category="test",
            yes_price=prices[-1], no_price=1 - prices[-1],
            volume=200, open_interest=100, expiry=None, status="open",
        )
        signals = strat.generate_signals(
            [market], {"TREND-UP": bars},
            {"lookback_periods": 10, "volume_threshold": 1.5,
             "price_change_threshold": 0.03, "exit_reversal_pct": 0.02,
             "default_quantity": 10},
        )
        assert len(signals) > 0
        assert signals[0].side == Side.BUY

    def test_no_signal_on_flat_market(self):
        strat = MomentumStrategy()
        prices = [0.50] * 25
        bars = self._make_bars(prices)

        market = Market(
            market_id="FLAT", provider=Provider.KALSHI,
            title="Flat", category="test",
            yes_price=0.50, no_price=0.50,
            volume=100, open_interest=100, expiry=None, status="open",
        )
        signals = strat.generate_signals(
            [market], {"FLAT": bars},
            {"lookback_periods": 10, "volume_threshold": 1.5,
             "price_change_threshold": 0.03, "exit_reversal_pct": 0.02,
             "default_quantity": 10},
        )
        assert len(signals) == 0


class TestMeanReversionStrategy:
    def _make_bars(self, prices):
        return [
            PriceBar(
                timestamp=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                open=p, high=p + 0.01, low=p - 0.01, close=p,
                volume=50 if i == len(prices) - 1 else 200,
            )
            for i, p in enumerate(prices)
        ]

    def test_buy_signal_on_panic_drop(self):
        strat = MeanReversionStrategy()
        # Stable prices then a sharp drop on low volume
        prices = [0.50] * 19 + [0.30]
        bars = self._make_bars(prices)

        market = Market(
            market_id="PANIC", provider=Provider.KALSHI,
            title="Panic Drop", category="test",
            yes_price=0.30, no_price=0.70,
            volume=50, open_interest=100, expiry=None, status="open",
        )
        signals = strat.generate_signals(
            [market], {"PANIC": bars},
            {"lookback_periods": 20, "std_dev_entry": 2.0,
             "std_dev_exit": 0.5, "min_volume_ratio": 0.3,
             "default_quantity": 10},
        )
        assert len(signals) > 0
        assert signals[0].side == Side.BUY


class TestBacktestHelpers:
    def test_sharpe_ratio(self):
        # Steady growth — use timedelta for safe date generation
        from datetime import timedelta
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        curve = [
            (base + timedelta(hours=i), 10000 + i * 10)
            for i in range(100)
        ]
        sharpe = _calc_sharpe(curve)
        assert sharpe > 0

    def test_max_drawdown(self):
        # Up then down
        curve = [
            (datetime(2024, 1, 1, tzinfo=timezone.utc), 10000),
            (datetime(2024, 1, 2, tzinfo=timezone.utc), 11000),
            (datetime(2024, 1, 3, tzinfo=timezone.utc), 9000),
            (datetime(2024, 1, 4, tzinfo=timezone.utc), 10000),
        ]
        dd = _calc_max_drawdown(curve)
        # Max drawdown from 11000 to 9000 = 18.18%
        assert abs(dd - 2000 / 11000) < 0.01

    def test_backtest_runs_on_bars(self):
        from datetime import timedelta
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            PriceBar(
                timestamp=base + timedelta(hours=i),
                open=0.40 + (i % 5) * 0.02,
                high=0.40 + (i % 5) * 0.02 + 0.01,
                low=0.40 + (i % 5) * 0.02 - 0.01,
                close=0.40 + (i % 5) * 0.02,
                volume=100 + i * 5,
            )
            for i in range(50)
        ]
        result = run_backtest(
            strategy=MomentumStrategy(),
            bars=bars,
            market_id="TEST",
            starting_capital=10000.0,
            params={"lookback_periods": 10, "volume_threshold": 1.5,
                    "price_change_threshold": 0.03, "exit_reversal_pct": 0.02,
                    "default_quantity": 10},
        )
        assert result.starting_capital == 10000.0
        assert result.ending_capital > 0
        assert len(result.equity_curve) > 0


class TestStore:
    def test_order_roundtrip(self, store):
        from prediction_bot.models import Order
        order = Order(
            order_id="test-001", market_id="MKT",
            provider=Provider.KALSHI, side=Side.BUY,
            order_type=OrderType.MARKET, price=0.50,
            quantity=10, status=OrderStatus.FILLED,
            filled_price=0.51, filled_quantity=10,
            reason="test",
        )
        store.save_order(order)
        orders = store.get_all_orders()
        assert len(orders) == 1
        assert orders[0].order_id == "test-001"
        assert orders[0].filled_price == 0.51

    def test_position_roundtrip(self, store):
        pos = Position(
            position_id="pos-001", market_id="MKT",
            provider=Provider.KALSHI, side=Side.BUY,
            quantity=10, avg_entry_price=0.50,
            current_price=0.55,
        )
        store.save_position(pos)
        positions = store.get_open_positions()
        assert len(positions) == 1
        assert positions[0].position_id == "pos-001"

    def test_config_state(self, store):
        store.set_state("cash", "9500.00")
        assert store.get_state("cash") == "9500.00"
        assert store.get_state("missing", "default") == "default"
