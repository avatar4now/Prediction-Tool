"""News-driven strategy.

Integrate a news API to detect events that could move contract prices
before the market prices them in.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from prediction_bot.models import Market, PriceBar, Side, Signal
from prediction_bot.strategies.base import Strategy

logger = logging.getLogger(__name__)


class NewsStrategy(Strategy):
    """News-driven trading strategy."""

    @property
    def name(self) -> str:
        return "news"

    def generate_signals(
        self,
        markets: list[Market],
        history: dict[str, list[PriceBar]],
        params: dict[str, Any],
    ) -> list[Signal]:
        api_key = params.get("news_api_key", "")
        if not api_key:
            logger.warning("No news_api_key configured — skipping news strategy")
            return []

        stale_hours = params.get("stale_hours", 4)
        sentiment_threshold = params.get("sentiment_threshold", 0.6)
        keywords = params.get("keywords", [])
        default_qty = params.get("default_quantity", 10)

        # Fetch headlines
        articles = self._fetch_headlines(api_key, keywords)
        if not articles:
            return []

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=stale_hours)

        signals: list[Signal] = []
        for market in markets:
            relevant = self._find_relevant_articles(market, articles, cutoff)
            if not relevant:
                continue

            sentiment = self._simple_sentiment(relevant)
            if abs(sentiment) < sentiment_threshold:
                continue

            side = Side.BUY if sentiment > 0 else Side.SELL
            strength = min(1.0, abs(sentiment))
            headline_preview = relevant[0].get("title", "")[:80]
            reason = (
                f"news_{side.value}: sentiment={sentiment:.2f}, "
                f"articles={len(relevant)}, headline=\"{headline_preview}\""
            )
            signals.append(
                Signal(
                    market_id=market.market_id,
                    provider=market.provider,
                    side=side,
                    strength=strength,
                    price=market.yes_price,
                    quantity=default_qty,
                    reason=reason,
                    strategy=self.name,
                    market_title=market.title,
                )
            )

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals

    def _fetch_headlines(
        self, api_key: str, extra_keywords: list[str]
    ) -> list[dict]:
        """Fetch recent headlines from NewsAPI."""
        try:
            resp = httpx.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "apiKey": api_key,
                    "language": "en",
                    "pageSize": 100,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json().get("articles", [])
        except Exception as e:
            logger.error("Failed to fetch news: %s", e)
            return []

    def _find_relevant_articles(
        self, market: Market, articles: list[dict], cutoff: datetime
    ) -> list[dict]:
        """Find articles relevant to a market based on title keyword matching."""
        # Extract keywords from market title
        title_words = set(
            w.lower()
            for w in re.findall(r'\b\w{4,}\b', market.title)
        )
        # Filter out very common words
        stop_words = {
            "will", "that", "this", "what", "which", "when", "where",
            "there", "their", "have", "been", "from", "they", "with",
            "about", "would", "could", "should", "before", "after",
        }
        title_words -= stop_words

        if not title_words:
            return []

        relevant: list[dict] = []
        for article in articles:
            # Check freshness
            pub = article.get("publishedAt", "")
            if pub:
                try:
                    pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if pub_dt < cutoff:
                        continue
                except ValueError:
                    pass

            headline = (article.get("title") or "").lower()
            desc = (article.get("description") or "").lower()
            text = headline + " " + desc

            # Check for keyword overlap
            matches = sum(1 for kw in title_words if kw in text)
            if matches >= 2:
                relevant.append(article)

        return relevant

    @staticmethod
    def _simple_sentiment(articles: list[dict]) -> float:
        """Basic keyword-based sentiment scoring (-1.0 to 1.0).

        Positive words push sentiment up, negative words push it down.
        """
        positive = {
            "win", "wins", "winning", "pass", "passed", "approve",
            "approved", "success", "gain", "gains", "rise", "rises",
            "increase", "surge", "surges", "likely", "favored",
            "confirmed", "agree", "agreement", "deal", "boost",
        }
        negative = {
            "lose", "loses", "losing", "fail", "failed", "reject",
            "rejected", "defeat", "drop", "drops", "fall", "falls",
            "decline", "crash", "unlikely", "oppose", "opposed",
            "block", "blocked", "cancel", "cancelled", "crisis",
        }

        pos_count = 0
        neg_count = 0
        for article in articles:
            text = (
                (article.get("title") or "") + " " + (article.get("description") or "")
            ).lower()
            words = set(re.findall(r'\b\w+\b', text))
            pos_count += len(words & positive)
            neg_count += len(words & negative)

        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
