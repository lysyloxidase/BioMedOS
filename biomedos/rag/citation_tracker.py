"""Citation tracking helpers for generated answers."""

from __future__ import annotations

import re
from collections import defaultdict

from biomedos.data.pubmed import PubMedArticle


class CitationTracker:
    """Track inline citations in generated text and format bibliography."""

    PMID_PATTERN = re.compile(r"\[(?:PMID:)?(\d+)\]")

    def format_inline(self, authors: list[str], year: int | None) -> str:
        """Format an inline citation label.

        Args:
            authors: Ordered author list.
            year: Publication year.

        Returns:
            Formatted author-year label without surrounding brackets.
        """

        surname = self._first_author_surname(authors)
        year_label = str(year) if year is not None else "n.d."
        if len(authors) <= 1:
            return f"{surname}, {year_label}"
        return f"{surname} et al., {year_label}"

    def format_citations(
        self,
        text: str,
        references: list[PubMedArticle],
    ) -> tuple[str, str]:
        """Convert PMID citations into author-year citations and bibliography.

        Args:
            text: Generated text containing PMID citations like ``[PMID:12345]``.
            references: Available PubMed references.

        Returns:
            Tuple of formatted text and bibliography markdown.
        """

        pmid_to_key, ordered_keys = self._build_reference_index(references)

        def replace(match: re.Match[str]) -> str:
            pmid = match.group(1)
            key = pmid_to_key.get(pmid, f"PMID:{pmid}")
            return f"[{key}]"

        cited_text = self.PMID_PATTERN.sub(replace, text)
        bibliography = self._format_bibliography(pmid_to_key, ordered_keys)
        return cited_text, bibliography

    def verify_citations(self, text: str, available_pmids: set[str]) -> list[str]:
        """Return PMID citations that do not map to available references.

        Args:
            text: Generated text containing PMID citations.
            available_pmids: Set of valid PMIDs.

        Returns:
            Invalid PMID citation identifiers.
        """

        invalid: list[str] = []
        for pmid in self.PMID_PATTERN.findall(text):
            if pmid not in available_pmids and pmid not in invalid:
                invalid.append(pmid)
        return invalid

    def extract_cited_pmids(self, text: str) -> list[str]:
        """Return unique PMIDs cited in a generated passage."""

        cited_pmids: list[str] = []
        for pmid in self.PMID_PATTERN.findall(text):
            if pmid not in cited_pmids:
                cited_pmids.append(pmid)
        return cited_pmids

    def _build_reference_index(
        self,
        references: list[PubMedArticle],
    ) -> tuple[dict[str, str], list[tuple[str, PubMedArticle]]]:
        """Build PMID-to-citation mapping and ordered bibliography entries."""

        base_counts: defaultdict[str, int] = defaultdict(int)
        pmid_to_key: dict[str, str] = {}
        ordered_keys: list[tuple[str, PubMedArticle]] = []

        for reference in references:
            base_key = self.format_inline(reference.authors, reference.year)
            base_counts[base_key] += 1
            suffix = "" if base_counts[base_key] == 1 else chr(ord("a") + base_counts[base_key] - 2)
            citation_key = f"{base_key}{suffix}"
            pmid_to_key[reference.pmid] = citation_key
            ordered_keys.append((citation_key, reference))

        return pmid_to_key, ordered_keys

    def _format_bibliography(
        self,
        pmid_to_key: dict[str, str],
        ordered_keys: list[tuple[str, PubMedArticle]],
    ) -> str:
        """Create a bibliography block in Markdown."""

        del pmid_to_key
        if not ordered_keys:
            return ""

        lines = ["## Bibliography"]
        for citation_key, reference in ordered_keys:
            author_text = ", ".join(reference.authors) if reference.authors else "Unknown author"
            journal_text = reference.journal or "Unknown journal"
            year_text = str(reference.year) if reference.year is not None else "n.d."
            doi_text = f" DOI: {reference.doi}." if reference.doi else ""
            lines.append(
                f"- [{citation_key}] {author_text} ({year_text}). {reference.title}. "
                f"*{journal_text}*. PMID: {reference.pmid}.{doi_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _first_author_surname(authors: list[str]) -> str:
        """Return the surname of the first listed author."""

        if not authors:
            return "Unknown"
        return authors[0].split()[0].strip(",") or "Unknown"
