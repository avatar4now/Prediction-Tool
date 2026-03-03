"""SQLite storage for trades, positions, and portfolio state."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from prediction_bot.models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionStatus,
    Portfolio,
    Provider,
    Side,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    filled_price REAL,
    filled_quantity INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    filled_at TEXT,
    reason TEXT DEFAULT '',
    market_title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_entry_price REAL NOT NULL,
    current_price REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    realized_pnl REAL DEFAULT 0.0,
    market_title TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_value REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS config_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class Store:
    """SQLite-backed persistence for the paper trading bot."""

    def __init__(self, db_path: str | Path = "predbot.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Orders ──────────────────────────────────────────────────────

    def save_order(self, order: Order) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, market_id, provider, side, order_type, price,
                quantity, status, filled_price, filled_quantity,
                created_at, filled_at, reason, market_title)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order.order_id,
                order.market_id,
                order.provider.value,
                order.side.value,
                order.order_type.value,
                order.price,
                order.quantity,
                order.status.value,
                order.filled_price,
                order.filled_quantity,
                order.created_at.isoformat(),
                order.filled_at.isoformat() if order.filled_at else None,
                order.reason,
                order.market_title,
            ),
        )
        self._conn.commit()

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        market_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Order]:
        query = "SELECT * FROM orders WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if market_id:
            query += " AND market_id = ?"
            params.append(market_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get_all_orders(self, limit: int = 500) -> list[Order]:
        rows = self._conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def _row_to_order(self, row: sqlite3.Row) -> Order:
        return Order(
            order_id=row["order_id"],
            market_id=row["market_id"],
            provider=Provider(row["provider"]),
            side=Side(row["side"]),
            order_type=OrderType(row["order_type"]),
            price=row["price"],
            quantity=row["quantity"],
            status=OrderStatus(row["status"]),
            filled_price=row["filled_price"],
            filled_quantity=row["filled_quantity"],
            created_at=datetime.fromisoformat(row["created_at"]),
            filled_at=_parse_dt(row["filled_at"]),
            reason=row["reason"],
            market_title=row["market_title"],
        )

    # ── Positions ───────────────────────────────────────────────────

    def save_position(self, pos: Position) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO positions
               (position_id, market_id, provider, side, quantity,
                avg_entry_price, current_price, status, opened_at,
                closed_at, realized_pnl, market_title)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pos.position_id,
                pos.market_id,
                pos.provider.value,
                pos.side.value,
                pos.quantity,
                pos.avg_entry_price,
                pos.current_price,
                pos.status.value,
                pos.opened_at.isoformat(),
                pos.closed_at.isoformat() if pos.closed_at else None,
                pos.realized_pnl,
                pos.market_title,
            ),
        )
        self._conn.commit()

    def get_open_positions(self) -> list[Position]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
        ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_all_positions(self, limit: int = 200) -> list[Position]:
        rows = self._conn.execute(
            "SELECT * FROM positions ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_position(self, position_id: str) -> Optional[Position]:
        row = self._conn.execute(
            "SELECT * FROM positions WHERE position_id = ?", (position_id,)
        ).fetchone()
        return self._row_to_position(row) if row else None

    def get_position_by_market(
        self, market_id: str, provider: Provider, status: PositionStatus = PositionStatus.OPEN
    ) -> Optional[Position]:
        row = self._conn.execute(
            "SELECT * FROM positions WHERE market_id = ? AND provider = ? AND status = ?",
            (market_id, provider.value, status.value),
        ).fetchone()
        return self._row_to_position(row) if row else None

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        return Position(
            position_id=row["position_id"],
            market_id=row["market_id"],
            provider=Provider(row["provider"]),
            side=Side(row["side"]),
            quantity=row["quantity"],
            avg_entry_price=row["avg_entry_price"],
            current_price=row["current_price"],
            status=PositionStatus(row["status"]),
            opened_at=datetime.fromisoformat(row["opened_at"]),
            closed_at=_parse_dt(row["closed_at"]),
            realized_pnl=row["realized_pnl"],
            market_title=row["market_title"],
        )

    # ── Portfolio snapshots ─────────────────────────────────────────

    def save_snapshot(self, portfolio: Portfolio) -> None:
        self._conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, cash, positions_value, total_value,
                total_trades, winning_trades)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                portfolio.cash,
                portfolio.positions_value,
                portfolio.total_value,
                portfolio.total_trades,
                portfolio.winning_trades,
            ),
        )
        self._conn.commit()

    def get_snapshots(self, limit: int = 365) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        rows = self._conn.execute(
            "SELECT timestamp, total_value FROM portfolio_snapshots ORDER BY timestamp ASC"
        ).fetchall()
        return [(datetime.fromisoformat(r["timestamp"]), r["total_value"]) for r in rows]

    # ── Config state (cash balance, etc.) ───────────────────────────

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO config_state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM config_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default
