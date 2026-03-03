"""CSV export of trade history."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from prediction_bot.db.store import Store
from prediction_bot.models import OrderStatus


def export_trades_csv(
    store: Store,
    output_dir: str = "exports",
    filename: str | None = None,
) -> str:
    """Export all filled trades to CSV.

    Returns the path to the generated CSV file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        from datetime import datetime
        filename = f"trades_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    csv_path = os.path.join(output_dir, filename)

    orders = store.get_all_orders(limit=10000)
    filled = [o for o in orders if o.status == OrderStatus.FILLED]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "market_id",
            "market_title",
            "provider",
            "side",
            "quantity",
            "price",
            "filled_price",
            "reason",
            "order_id",
        ])

        for o in filled:
            writer.writerow([
                o.created_at.isoformat(),
                o.market_id,
                o.market_title,
                o.provider.value,
                o.side.value,
                o.filled_quantity,
                o.price,
                o.filled_price,
                o.reason,
                o.order_id,
            ])

    return csv_path
