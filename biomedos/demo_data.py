"""Bundled local demo data for offline-first BioMedOS workflows."""

from __future__ import annotations

from biomedos.data.pubmed import PubMedArticle


def demo_articles() -> list[PubMedArticle]:
    """Return local literature snippets for demo and fallback RAG workflows."""

    return [
        PubMedArticle(
            pmid="9001",
            title="EGFR signaling reshapes lung cancer therapy response",
            abstract=(
                "EGFR signaling drives tumor growth in lung cancer and remains "
                "a core target for gefitinib therapy."
            ),
            journal="BioMedOS Reports",
            authors=["Doe J", "Smith A"],
            year=2024,
            keywords=["EGFR", "lung cancer"],
        ),
        PubMedArticle(
            pmid="9002",
            title="LOXL2 and extracellular matrix remodeling in fibrosis",
            abstract=(
                "LOXL2 supports fibrotic remodeling, suggesting a mechanistic "
                "bridge between matrix stiffness and disease progression."
            ),
            journal="BioMedOS Reports",
            authors=["Chan K", "Patel R"],
            year=2025,
            keywords=["LOXL2", "fibrosis"],
        ),
        PubMedArticle(
            pmid="9003",
            title="DNA repair dependencies in BRCA-associated tumors",
            abstract=(
                "BRCA1 and BRCA2 loss create DNA repair vulnerabilities that "
                "support PARP inhibitor strategies such as olaparib."
            ),
            journal="BioMedOS Reports",
            authors=["Ng L", "Ortiz M"],
            year=2024,
            keywords=["BRCA1", "BRCA2", "olaparib"],
        ),
        PubMedArticle(
            pmid="9004",
            title="JAK-STAT rewiring in inflammatory fibrosis",
            abstract=(
                "Inflammatory signaling through JAK-STAT intersects with matrix "
                "remodeling genes during fibrotic progression."
            ),
            journal="BioMedOS Reports",
            authors=["Singh P", "Taylor E"],
            year=2023,
            keywords=["JAK-STAT", "fibrosis"],
        ),
        PubMedArticle(
            pmid="9005",
            title="ALK and EGFR crosstalk in thoracic malignancy",
            abstract=(
                "ALK and EGFR can create parallel signaling routes in thoracic "
                "malignancy, complicating targeted treatment selection."
            ),
            journal="BioMedOS Reports",
            authors=["Khan S", "Meyer T"],
            year=2025,
            keywords=["ALK", "EGFR"],
        ),
        PubMedArticle(
            pmid="9006",
            title="Metformin modulates mTOR-associated metabolic stress",
            abstract=(
                "Metformin alters mTOR-linked metabolic programs and may affect "
                "combination regimens in oncology settings."
            ),
            journal="BioMedOS Reports",
            authors=["Ivanov D", "Li Y"],
            year=2022,
            keywords=["metformin", "MTOR"],
        ),
    ]


__all__ = ["demo_articles"]
