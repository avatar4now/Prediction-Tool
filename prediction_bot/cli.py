"""CLI interface for the prediction market trading bot.

Commands: scan, trade, portfolio, history, report, backtest
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from prediction_bot.config import get_nested, load_config


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_providers(cfg: dict) -> list:
    """Instantiate configured market data providers."""
    from prediction_bot.providers.kalshi import KalshiProvider
    from prediction_bot.providers.polymarket import PolymarketProvider

    providers = []

    kalshi_cfg = get_nested(cfg, "providers", "kalshi", default={})
    if kalshi_cfg:
        providers.append(
            KalshiProvider(
                base_url=kalshi_cfg.get("base_url", "https://api.elections.kalshi.com/trade-api/v2"),
                api_key=kalshi_cfg.get("api_key", ""),
                rate_limit_per_sec=kalshi_cfg.get("rate_limit_per_sec", 10),
            )
        )

    poly_cfg = get_nested(cfg, "providers", "polymarket", default={})
    if poly_cfg:
        providers.append(
            PolymarketProvider(
                clob_url=poly_cfg.get("base_url", "https://clob.polymarket.com"),
                gamma_url=poly_cfg.get("gamma_url", "https://gamma-api.polymarket.com"),
                rate_limit_per_sec=poly_cfg.get("rate_limit_per_sec", 10),
            )
        )

    return providers


def _get_strategy(cfg: dict):
    """Instantiate the active strategy from config."""
    from prediction_bot.strategies.momentum import MomentumStrategy
    from prediction_bot.strategies.mean_reversion import MeanReversionStrategy
    from prediction_bot.strategies.news import NewsStrategy
    from prediction_bot.strategies.arbitrage import ArbitrageStrategy

    name = get_nested(cfg, "strategies", "active", default="momentum")
    strategies = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "news": NewsStrategy,
        "arbitrage": ArbitrageStrategy,
    }
    cls = strategies.get(name)
    if cls is None:
        click.echo(f"Unknown strategy: {name}. Available: {', '.join(strategies)}")
        sys.exit(1)
    return cls()


def _get_store(cfg: dict):
    from prediction_bot.db.store import Store
    db_path = get_nested(cfg, "database", "path", default="predbot.db")
    return Store(db_path)


def _get_portfolio(store, cfg: dict):
    from prediction_bot.engine.portfolio_manager import PortfolioManager
    capital = get_nested(cfg, "portfolio", "starting_capital", default=10000.0)
    return PortfolioManager(store, starting_capital=capital)


def _get_risk(cfg: dict):
    from prediction_bot.engine.risk import RiskManager
    risk_cfg = get_nested(cfg, "risk", default={})
    return RiskManager(
        max_position_pct=risk_cfg.get("max_position_pct", 0.05),
        max_exposure_pct=risk_cfg.get("max_exposure_pct", 0.50),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", 0.20),
        expiry_close_hours=risk_cfg.get("expiry_close_hours", 24),
    )


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Prediction market paper trading bot.

    Paper trading only — no real money is ever at risk.
    """
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config)
    ctx.obj["verbose"] = verbose


# ── scan ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--provider", "-p", default=None, help="Filter by provider (kalshi/polymarket)")
@click.option("--limit", "-n", default=50, help="Max markets to show")
@click.pass_context
def scan(ctx, provider, limit):
    """Show active markets with current prices and volume."""
    from prediction_bot.reports.terminal import print_markets

    cfg = ctx.obj["cfg"]
    providers = _get_providers(cfg)

    if provider:
        providers = [p for p in providers if p.name == provider]
        if not providers:
            click.echo(f"No provider named '{provider}'")
            return

    async def _scan():
        all_markets = []
        for p in providers:
            try:
                markets = await p.get_markets(limit=limit)
                all_markets.extend(markets)
            except Exception as e:
                click.echo(f"Error fetching from {p.name}: {e}")
            finally:
                await p.close()
        return all_markets

    markets = asyncio.run(_scan())
    if markets:
        print_markets(markets[:limit])
    else:
        click.echo("No markets found. Check your config and network connection.")


# ── trade ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--cycles", "-n", default=1, help="Number of trading cycles to run")
@click.option("--strategy", "-s", default=None, help="Strategy override (momentum/mean_reversion/news/arbitrage)")
@click.pass_context
def trade(ctx, cycles, strategy):
    """Run the trading bot for a session (paper trades only)."""
    from prediction_bot.engine.runner import TradingRunner
    from prediction_bot.reports.terminal import print_cycle_actions, print_portfolio

    cfg = ctx.obj["cfg"]
    if strategy:
        cfg.setdefault("strategies", {})["active"] = strategy

    providers = _get_providers(cfg)
    strat = _get_strategy(cfg)
    store = _get_store(cfg)
    portfolio = _get_portfolio(store, cfg)
    risk = _get_risk(cfg)

    runner = TradingRunner(providers, strat, portfolio, risk, cfg)

    click.echo(f"Starting paper trading session: strategy={strat.name}, cycles={cycles}")
    click.echo("=" * 60)

    async def _trade():
        for i in range(cycles):
            click.echo(f"\n--- Cycle {i + 1}/{cycles} ---")
            try:
                actions = await runner.run_once()
                print_cycle_actions(actions)
            except Exception as e:
                click.echo(f"Cycle error: {e}")
                logging.getLogger(__name__).exception("Cycle failed")

            if i < cycles - 1:
                interval = get_nested(cfg, "trading", "scan_interval_sec", default=60)
                click.echo(f"Waiting {interval}s...")
                await asyncio.sleep(interval)

        # Close provider connections
        for p in providers:
            await p.close()

    asyncio.run(_trade())

    # Print final portfolio state
    click.echo("\n" + "=" * 60)
    from prediction_bot.reports.terminal import print_portfolio
    print_portfolio(portfolio.get_portfolio())
    store.close()


# ── portfolio ───────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def portfolio(ctx):
    """Show current paper positions and P&L."""
    from prediction_bot.reports.terminal import print_portfolio

    cfg = ctx.obj["cfg"]
    store = _get_store(cfg)
    pm = _get_portfolio(store, cfg)
    print_portfolio(pm.get_portfolio())
    store.close()


# ── history ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--limit", "-n", default=50, help="Number of orders to show")
@click.option("--export-csv", is_flag=True, help="Export to CSV file")
@click.pass_context
def history(ctx, limit, export_csv):
    """Show trade history."""
    from prediction_bot.reports.terminal import print_trade_history

    cfg = ctx.obj["cfg"]
    store = _get_store(cfg)

    orders = store.get_all_orders(limit=limit)
    print_trade_history(orders)

    if export_csv:
        from prediction_bot.reports.csv_export import export_trades_csv
        csv_dir = get_nested(cfg, "reporting", "csv_export_path", default="exports")
        path = export_trades_csv(store, output_dir=csv_dir)
        click.echo(f"\nCSV exported to: {path}")

    store.close()


# ── report ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--output-dir", "-o", default=None, help="Output directory for PDF")
@click.pass_context
def report(ctx, output_dir):
    """Generate PDF performance report."""
    from prediction_bot.reports.pdf_report import generate_pdf_report

    cfg = ctx.obj["cfg"]
    store = _get_store(cfg)

    if output_dir is None:
        output_dir = get_nested(cfg, "reporting", "pdf_export_path", default="reports")

    path = generate_pdf_report(store, output_dir=output_dir)
    click.echo(f"PDF report generated: {path}")
    store.close()


# ── backtest ────────────────────────────────────────────────────────

@cli.command()
@click.argument("market_id")
@click.option("--strategy", "-s", default=None, help="Strategy to backtest")
@click.option("--provider", "-p", default="kalshi", help="Provider for historical data")
@click.option("--capital", default=10000.0, help="Starting capital for backtest")
@click.option("--interval", default="1h", help="Price bar interval (1m/5m/15m/1h/4h/1d)")
@click.pass_context
def backtest(ctx, market_id, strategy, provider, capital, interval):
    """Backtest a strategy on historical market data.

    MARKET_ID is the ticker/id of the market to backtest against.
    """
    from prediction_bot.engine.backtest import run_backtest

    cfg = ctx.obj["cfg"]
    if strategy:
        cfg.setdefault("strategies", {})["active"] = strategy

    strat = _get_strategy(cfg)
    providers = _get_providers(cfg)

    # Find the matching provider
    target_provider = None
    for p in providers:
        if p.name == provider:
            target_provider = p
            break

    if target_provider is None:
        click.echo(f"Provider '{provider}' not configured")
        return

    click.echo(f"Backtesting {strat.name} on {market_id} ({provider})...")

    async def _fetch():
        try:
            bars = await target_provider.get_historical_prices(
                market_id, interval=interval
            )
            return bars
        finally:
            await target_provider.close()

    bars = asyncio.run(_fetch())
    if not bars:
        click.echo("No historical data available for this market.")
        return

    click.echo(f"Loaded {len(bars)} price bars")

    # Get strategy params from config
    strategy_params = get_nested(cfg, "strategies", strat.name, default={})
    strategy_params["default_quantity"] = get_nested(
        cfg, "trading", "default_quantity", default=10
    )
    risk_params = get_nested(cfg, "risk", default={})

    from prediction_bot.models import Provider as ProviderEnum
    prov_enum = ProviderEnum.KALSHI if provider == "kalshi" else ProviderEnum.POLYMARKET

    result = run_backtest(
        strategy=strat,
        bars=bars,
        market_id=market_id,
        provider=prov_enum,
        starting_capital=capital,
        params=strategy_params,
        risk_params=risk_params,
    )

    click.echo("\n" + "=" * 60)
    click.echo(result.summary())
    click.echo("=" * 60)

    # Plot equity curve if we have data
    if len(result.equity_curve) >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            dates = [d for d, _ in result.equity_curve]
            values = [v for _, v in result.equity_curve]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(dates, values, linewidth=1.5)
            ax.set_title(f"Backtest: {strat.name} on {market_id}")
            ax.set_ylabel("Portfolio Value ($)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            chart_path = f"backtest_{market_id}_{strat.name}.png"
            fig.savefig(chart_path, dpi=150)
            plt.close(fig)
            click.echo(f"Equity curve saved to: {chart_path}")
        except Exception as e:
            click.echo(f"Could not generate chart: {e}")


# ── reset ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--capital", default=None, type=float, help="New starting capital")
@click.confirmation_option(prompt="Reset paper portfolio? All trades/positions will remain in history.")
@click.pass_context
def reset(ctx, capital):
    """Reset paper portfolio to starting capital."""
    cfg = ctx.obj["cfg"]
    store = _get_store(cfg)
    pm = _get_portfolio(store, cfg)
    pm.reset(starting_capital=capital)
    click.echo(f"Portfolio reset. Cash: ${pm.cash:,.2f}")
    store.close()


if __name__ == "__main__":
    cli()
