"""PDF performance report generation.

fpdf2 and matplotlib are imported lazily so the rest of the bot works
even if the cryptography backend is unavailable in the current environment.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from prediction_bot.db.store import Store
from prediction_bot.models import OrderStatus


def _lazy_imports():
    """Lazy-import fpdf2 + matplotlib; raises ImportError with a helpful
    message if the dependencies are broken."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.dates as _mdates
        from fpdf import FPDF as _FPDF
    except (ImportError, Exception) as exc:
        raise ImportError(
            "PDF report generation requires fpdf2 and matplotlib with a "
            f"working cryptography backend. Original error: {exc}"
        ) from exc
    return _FPDF, _plt, _mdates


def generate_pdf_report(
    store: Store,
    output_dir: str = "reports",
    filename: str | None = None,
) -> str:
    """Generate a PDF performance report.

    Returns the path to the generated PDF.
    """
    _FPDF, plt, mdates = _lazy_imports()

    class _Report(_FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "Prediction Bot — Performance Report", ln=True, align="C")
            self.set_font("Helvetica", "", 9)
            self.cell(
                0, 6,
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                ln=True, align="C",
            )
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    pdf = _Report()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Portfolio summary ───────────────────────────────────────────
    snapshots = store.get_snapshots(limit=1)
    if snapshots:
        latest = snapshots[0]
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Portfolio Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)

        rows = [
            ("Cash", f"${latest['cash']:,.2f}"),
            ("Positions Value", f"${latest['positions_value']:,.2f}"),
            ("Total Value", f"${latest['total_value']:,.2f}"),
            ("Total Trades", str(latest["total_trades"])),
            ("Winning Trades", str(latest["winning_trades"])),
            ("Win Rate", f"{latest['winning_trades'] / max(latest['total_trades'], 1):.1%}"),
        ]
        for label, value in rows:
            pdf.cell(60, 6, label, border=1)
            pdf.cell(50, 6, value, border=1, ln=True)
        pdf.ln(6)

    # ── Equity curve chart ──────────────────────────────────────────
    equity_data = store.get_equity_curve()
    if len(equity_data) >= 2:
        chart_path = os.path.join(output_dir, "_equity_curve.png")
        _plot_equity_curve(equity_data, chart_path, plt, mdates)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Equity Curve", ln=True)
        pdf.image(chart_path, w=180)
        pdf.ln(4)
        try:
            os.remove(chart_path)
        except OSError:
            pass

    # ── Trade log ───────────────────────────────────────────────────
    orders = store.get_all_orders(limit=200)
    filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
    if filled_orders:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Trade Log", ln=True)
        pdf.set_font("Helvetica", "B", 8)

        col_widths = [30, 50, 15, 15, 20, 60]
        headers = ["Time", "Market", "Side", "Qty", "Price", "Reason"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 6, h, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        for o in filled_orders[:100]:
            pdf.cell(col_widths[0], 5, o.created_at.strftime("%Y-%m-%d %H:%M"), border=1)
            pdf.cell(col_widths[1], 5, (o.market_title or o.market_id)[:30], border=1)
            pdf.cell(col_widths[2], 5, o.side.value.upper(), border=1)
            pdf.cell(col_widths[3], 5, str(o.filled_quantity), border=1)
            pdf.cell(col_widths[4], 5, f"${o.filled_price:.4f}" if o.filled_price else "—", border=1)
            pdf.cell(col_widths[5], 5, o.reason[:35], border=1)
            pdf.ln()

    # ── Market breakdown ────────────────────────────────────────────
    if filled_orders:
        market_stats: dict[str, dict] = {}
        for o in filled_orders:
            key = o.market_title or o.market_id
            if key not in market_stats:
                market_stats[key] = {"trades": 0, "total_cost": 0.0}
            market_stats[key]["trades"] += 1
            if o.filled_price:
                market_stats[key]["total_cost"] += o.filled_price * o.filled_quantity

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Market Breakdown", ln=True)
        pdf.set_font("Helvetica", "B", 8)

        pdf.cell(80, 6, "Market", border=1)
        pdf.cell(30, 6, "Trades", border=1)
        pdf.cell(40, 6, "Total Volume", border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for market, stats in sorted(market_stats.items(), key=lambda x: -x[1]["trades"]):
            pdf.cell(80, 5, market[:45], border=1)
            pdf.cell(30, 5, str(stats["trades"]), border=1)
            pdf.cell(40, 5, f"${stats['total_cost']:,.2f}", border=1)
            pdf.ln()

    pdf.output(pdf_path)
    return pdf_path


def _plot_equity_curve(equity_data, output_path, plt, mdates):
    """Generate equity curve chart as PNG."""
    dates = [d for d, _ in equity_data]
    values = [v for _, v in equity_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, linewidth=1.5, color="#2196F3")
    ax.fill_between(dates, values, alpha=0.1, color="#2196F3")

    ax.set_title("Portfolio Equity Curve", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
