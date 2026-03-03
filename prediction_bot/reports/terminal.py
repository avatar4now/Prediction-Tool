"""Terminal-based reporting with rich tables."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from prediction_bot.models import (
    Market,
    Order,
    OrderStatus,
    Portfolio,
    Position,
    PositionStatus,
)


console = Console()


def print_markets(markets: list[Market]) -> None:
    """Print a table of active markets."""
    table = Table(title="Active Markets", show_lines=True)
    table.add_column("ID", style="cyan", max_width=24)
    table.add_column("Provider", style="blue")
    table.add_column("Title", max_width=50)
    table.add_column("Yes", justify="right", style="green")
    table.add_column("No", justify="right", style="red")
    table.add_column("Volume", justify="right")
    table.add_column("Expiry")

    for m in markets:
        expiry_str = m.expiry.strftime("%Y-%m-%d %H:%M") if m.expiry else "—"
        table.add_row(
            m.market_id[:24],
            m.provider.value,
            m.title[:50],
            f"${m.yes_price:.2f}",
            f"${m.no_price:.2f}",
            f"{m.volume:,.0f}",
            expiry_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(markets)} markets[/dim]")


def print_portfolio(portfolio: Portfolio) -> None:
    """Print portfolio summary and open positions."""
    console.print("\n[bold]Portfolio Summary[/bold]")
    console.print(f"  Cash:           ${portfolio.cash:>12,.2f}")
    console.print(f"  Positions:      ${portfolio.positions_value:>12,.2f}")
    console.print(f"  Total Value:    ${portfolio.total_value:>12,.2f}")
    console.print(f"  Unrealized P&L: ${portfolio.unrealized_pnl:>12,.2f}")
    console.print(f"  Exposure:       {portfolio.exposure_pct:>12.1%}")
    console.print(f"  Trades:         {portfolio.total_trades:>12d}")
    console.print(f"  Win Rate:       {portfolio.win_rate:>12.1%}")

    if portfolio.positions:
        table = Table(title="\nOpen Positions", show_lines=True)
        table.add_column("ID", style="cyan", max_width=12)
        table.add_column("Market", max_width=30)
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for p in portfolio.positions:
            if p.status != PositionStatus.OPEN:
                continue
            pnl = p.unrealized_pnl
            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                p.position_id[:12],
                (p.market_title or p.market_id)[:30],
                p.side.value.upper(),
                str(p.quantity),
                f"${p.avg_entry_price:.4f}",
                f"${p.current_price:.4f}",
                f"[{pnl_style}]${pnl:+.2f}[/{pnl_style}]",
                f"[{pnl_style}]{p.pnl_pct:+.1%}[/{pnl_style}]",
            )

        console.print(table)
    else:
        console.print("\n  [dim]No open positions[/dim]")


def print_trade_history(orders: list[Order]) -> None:
    """Print trade history table."""
    table = Table(title="Trade History", show_lines=True)
    table.add_column("Time", style="dim")
    table.add_column("Market", max_width=30)
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Status")
    table.add_column("Reason", max_width=40)

    for o in orders:
        status_style = "green" if o.status == OrderStatus.FILLED else "yellow"
        side_style = "green" if o.side.value == "buy" else "red"
        table.add_row(
            o.created_at.strftime("%Y-%m-%d %H:%M"),
            (o.market_title or o.market_id)[:30],
            f"[{side_style}]{o.side.value.upper()}[/{side_style}]",
            str(o.filled_quantity or o.quantity),
            f"${(o.filled_price or o.price):.4f}",
            f"[{status_style}]{o.status.value}[/{status_style}]",
            o.reason[:40],
        )

    console.print(table)
    console.print(f"\n[dim]{len(orders)} orders[/dim]")


def print_cycle_actions(actions: list[dict]) -> None:
    """Print summary of actions from a trading cycle."""
    if not actions:
        console.print("[dim]No actions this cycle[/dim]")
        return

    for action in actions:
        action_type = action["type"]
        market = action.get("market", "?")

        if action_type == "executed":
            order = action["order"]
            console.print(
                f"  [green]EXEC[/green] {order.side.value.upper()} "
                f"{order.filled_quantity or order.quantity}x {market} "
                f"@ ${(order.filled_price or order.price):.4f} — {order.reason}"
            )
        elif action_type == "rejected":
            console.print(
                f"  [yellow]SKIP[/yellow] {market} — {action['reason']}"
            )
        elif action_type == "stop_loss":
            console.print(
                f"  [red]STOP[/red] {market} — {action['reason']}"
            )
        elif action_type == "expiry_close":
            console.print(
                f"  [magenta]EXPIRY[/magenta] {market} — {action['reason']}"
            )
        elif action_type == "limit_fill":
            order = action["order"]
            console.print(
                f"  [blue]LIMIT[/blue] {order.side.value.upper()} "
                f"{order.filled_quantity}x {market} @ ${order.filled_price:.4f}"
            )
