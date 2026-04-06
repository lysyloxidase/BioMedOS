"""Temporal literature trend analysis."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from itertools import pairwise
from typing import cast

import numpy as np

from biomedos.data.pubmed import PubMedArticle
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import NodeType


class TrendDetector:
    """Detect literature growth and decline trends."""

    def __init__(
        self,
        *,
        knowledge_graph: KnowledgeGraph | None = None,
        articles: list[PubMedArticle] | None = None,
    ) -> None:
        """Initialize the trend detector."""

        self.knowledge_graph = knowledge_graph
        self.articles = articles or []

    def detect(self, topic: str) -> dict[str, object]:
        """Detect temporal trends for a biomedical topic."""

        records = self._collect_articles(topic)
        yearly_counts = Counter(article.year for article in records if article.year is not None)
        ordered_years = sorted(yearly_counts)
        publication_velocity = self._publication_velocity(ordered_years, yearly_counts)
        embedding_drift = self._lexical_drift(records)
        emerging_associations = self._emerging_associations(records)
        return {
            "topic": topic,
            "yearly_counts": {str(year): yearly_counts[year] for year in ordered_years},
            "publication_velocity": publication_velocity,
            "embedding_drift": embedding_drift,
            "emerging_associations": emerging_associations,
        }

    def _collect_articles(self, topic: str) -> list[PubMedArticle]:
        """Collect topic-matching articles from local sources."""

        lowered_topic = topic.lower()
        if self.articles:
            return [
                article
                for article in self.articles
                if lowered_topic in article.title.lower()
                or lowered_topic in article.abstract.lower()
            ]
        if self.knowledge_graph is None:
            return []

        records: list[PubMedArticle] = []
        for node_id, payload in self.knowledge_graph.graph.nodes(data=True):
            if payload.get("node_type") != NodeType.PUBLICATION:
                continue
            title = str(payload.get("title", payload.get("name", "")))
            abstract = str(payload.get("abstract", ""))
            if lowered_topic not in title.lower() and lowered_topic not in abstract.lower():
                continue
            records.append(
                PubMedArticle(
                    pmid=str(payload.get("pmid", node_id)),
                    title=title,
                    abstract=abstract,
                    authors=list(payload.get("authors", [])),
                    year=int(payload.get("year")) if payload.get("year") is not None else None,
                )
            )
        return records

    def _publication_velocity(
        self,
        ordered_years: list[int],
        yearly_counts: Counter[int],
    ) -> float:
        """Estimate trend slope from yearly publication counts."""

        if len(ordered_years) < 2:
            return 0.0
        x = np.asarray(ordered_years, dtype=np.float64)
        y = np.asarray([yearly_counts[year] for year in ordered_years], dtype=np.float64)
        slope, _ = np.polyfit(x, y, deg=1)
        return float(slope)

    def _lexical_drift(self, records: list[PubMedArticle]) -> float:
        """Estimate year-to-year lexical drift in abstracts."""

        by_year: dict[int, Counter[str]] = defaultdict(Counter)
        for article in records:
            if article.year is None:
                continue
            tokens = re.findall(r"[A-Za-z0-9_:-]+", article.abstract.lower())
            by_year[article.year].update(tokens)
        ordered_years = sorted(by_year)
        if len(ordered_years) < 2:
            return 0.0
        drifts: list[float] = []
        for left_year, right_year in pairwise(ordered_years):
            left = by_year[left_year]
            right = by_year[right_year]
            vocabulary = set(left) | set(right)
            if not vocabulary:
                continue
            numerator = sum(min(left[token], right[token]) for token in vocabulary)
            denominator = sum(max(left[token], right[token]) for token in vocabulary)
            similarity = numerator / denominator if denominator else 1.0
            drifts.append(1.0 - similarity)
        return float(sum(drifts) / len(drifts)) if drifts else 0.0

    def _emerging_associations(self, records: list[PubMedArticle]) -> list[dict[str, object]]:
        """Find entities with increasing recent mention frequency."""

        if not records:
            return []
        by_year: dict[int, Counter[str]] = defaultdict(Counter)
        for article in records:
            if article.year is None:
                continue
            terms = {
                token
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9_-]{2,}",
                    f"{article.title} {article.abstract}",
                )
                if not token.islower() or any(character.isdigit() for character in token)
            }
            by_year[article.year].update(term.lower() for term in terms)

        ordered_years = sorted(by_year)
        if len(ordered_years) < 2:
            return []
        recent_years = ordered_years[-2:]
        historical_years = ordered_years[:-2] or ordered_years[-2:-1]
        recent_counts: Counter[str] = Counter()
        historical_counts: Counter[str] = Counter()
        for year in recent_years:
            recent_counts.update(by_year[year])
        for year in historical_years:
            historical_counts.update(by_year[year])

        emerging: list[dict[str, int | float | str]] = []
        for term, count in recent_counts.items():
            baseline = historical_counts.get(term, 0)
            growth = count - baseline
            if growth <= 0:
                continue
            ratio = count / max(baseline, 1)
            score = growth * math.log1p(ratio)
            emerging.append({"term": term, "recent_mentions": count, "growth_score": score})
        emerging.sort(
            key=lambda item: cast(float, item["growth_score"]),
            reverse=True,
        )
        return cast(list[dict[str, object]], emerging[:10])
