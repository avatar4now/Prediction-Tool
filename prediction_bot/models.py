"""Core data models for the prediction market trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class Provider(str, Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


@dataclass
class Market:
    """A prediction market contract."""

    market_id: str
    provider: Provider
    title: str
    category: str
    yes_price: float  # 0.0–1.0 (probability)
    no_price: float
    volume: float
    open_interest: float
    expiry: Optional[datetime]
    status: str  # "open", "closed", "settled"
    url: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid_price(self) -> float:
        return (self.yes_price + (1.0 - self.no_price)) / 2.0


@dataclass
class OrderBookLevel:
    """Single level in an order book."""

    price: float
    quantity: float


@dataclass
class OrderBook:
    """Order book for a market."""

    market_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass
class Trade:
    """A historical or executed trade."""

    trade_id: str
    market_id: str
    provider: Provider
    side: Side
    price: float
    quantity: int
    timestamp: datetime
    market_title: str = ""


@dataclass
class PriceBar:
    """OHLCV price bar for historical data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketDetail:
    """Extended market info including orderbook and recent trades."""

    market: Market
    orderbook: OrderBook
    recent_trades: list[Trade] = field(default_factory=list)


@dataclass
class Order:
    """A paper trade order."""

    order_id: str
    market_id: str
    provider: Provider
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    reason: str = ""  # Strategy signal reason
    market_title: str = ""


@dataclass
class Position:
    """A paper trading position."""

    position_id: str
    market_id: str
    provider: Provider
    side: Side
    quantity: int
    avg_entry_price: float
    current_price: float
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    realized_pnl: float = 0.0
    market_title: str = ""

    @property
    def unrealized_pnl(self) -> float:
        if self.status == PositionStatus.CLOSED:
            return 0.0
        if self.side == Side.BUY:
            return (self.current_price - self.avg_entry_price) * self.quantity
        else:
            return (self.avg_entry_price - self.current_price) * self.quantity

    @property
    def cost_basis(self) -> float:
        return self.avg_entry_price * self.quantity

    @property
    def pnl_pct(self) -> float:
        basis = self.cost_basis
        if basis == 0:
            return 0.0
        if self.status == PositionStatus.CLOSED:
            return self.realized_pnl / basis
        return self.unrealized_pnl / basis


@dataclass
class Portfolio:
    """Paper trading portfolio snapshot."""

    cash: float
    positions: list[Position] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0

    @property
    def positions_value(self) -> float:
        return sum(p.current_price * p.quantity for p in self.positions if p.status == PositionStatus.OPEN)

    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions if p.status == PositionStatus.OPEN)

    @property
    def exposure_pct(self) -> float:
        total = self.total_value
        if total == 0:
            return 0.0
        return self.positions_value / total

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


@dataclass
class Signal:
    """A trading signal generated by a strategy."""

    market_id: str
    provider: Provider
    side: Side
    strength: float  # 0.0–1.0
    price: float
    quantity: int
    reason: str
    strategy: str
    market_title: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
