"""Bundled local demo data for offline-first BioMedOS workflows."""

from __future__ import annotations

from collections.abc import Sequence

from biomedos.data.pubmed import PubMedArticle
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioEdge, DiseaseNode, DrugNode, EdgeType, GeneNode, PathwayNode


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


def build_demo_graph(genes: Sequence[str]) -> KnowledgeGraph:
    """Build a compact demo graph from a short gene list.

    Args:
        genes: Gene symbols used to seed the demo graph.

    Returns:
        A small but non-trivial biomedical knowledge graph suitable for local demos.
    """

    graph = KnowledgeGraph()
    diseases = [
        DiseaseNode(id="disease:lung_cancer", name="Lung Cancer", efo_id="EFO:0001071"),
        DiseaseNode(id="disease:fibrosis", name="Fibrosis", efo_id="EFO:0000400"),
        DiseaseNode(id="disease:breast_cancer", name="Breast Cancer", efo_id="EFO:0000305"),
        DiseaseNode(id="disease:glioblastoma", name="Glioblastoma", efo_id="EFO:0000519"),
    ]
    pathways = [
        PathwayNode(id="pathway:egfr", name="EGFR Signaling", reactome_id="R-HSA-177929"),
        PathwayNode(id="pathway:dna", name="DNA Repair", reactome_id="R-HSA-73894"),
        PathwayNode(id="pathway:ecm", name="ECM Remodeling", reactome_id="R-HSA-1474244"),
        PathwayNode(id="pathway:jak", name="JAK-STAT", reactome_id="R-HSA-6785807"),
    ]
    drugs = [
        DrugNode(id="drug:gefitinib", name="Gefitinib", chembl_id="CHEMBL939", max_phase=4),
        DrugNode(id="drug:olaparib", name="Olaparib", chembl_id="CHEMBL521686", max_phase=4),
        DrugNode(id="drug:pirfenidone", name="Pirfenidone", chembl_id="CHEMBL1744", max_phase=4),
        DrugNode(id="drug:trametinib", name="Trametinib", chembl_id="CHEMBL2103875", max_phase=4),
        DrugNode(id="drug:metformin", name="Metformin", chembl_id="CHEMBL1431", max_phase=4),
    ]
    for node in [*diseases, *pathways, *drugs]:
        graph.merge_node(node)

    disease_cycle = [d.id for d in diseases]
    pathway_cycle = [p.id for p in pathways]
    gene_list = list(genes)
    for index, symbol in enumerate(gene_list):
        gene_id = f"gene:{symbol.lower()}"
        graph.merge_node(
            GeneNode(
                id=gene_id,
                name=symbol,
                symbol=symbol,
                chromosome=str((index % 22) + 1),
                description=f"{symbol} demo biomarker",
                sources=["demo"],
            )
        )
        graph.merge_edge(
            BioEdge(
                source_id=gene_id,
                target_id=disease_cycle[index % len(disease_cycle)],
                edge_type=EdgeType.GENE_DISEASE,
                score=0.65 + (index % 4) * 0.08,
                sources=["demo"],
            )
        )
        graph.merge_edge(
            BioEdge(
                source_id=gene_id,
                target_id=pathway_cycle[index % len(pathway_cycle)],
                edge_type=EdgeType.GENE_PATHWAY,
                score=0.62 + (index % 3) * 0.1,
                sources=["demo"],
            )
        )
        if index < len(gene_list) - 1:
            graph.merge_edge(
                BioEdge(
                    source_id=gene_id,
                    target_id=f"gene:{gene_list[index + 1].lower()}",
                    edge_type=EdgeType.GENE_GENE,
                    score=0.58 + (index % 5) * 0.06,
                    sources=["demo"],
                )
            )

    targets = {
        "drug:gefitinib": ["gene:egfr", "gene:alk"],
        "drug:olaparib": ["gene:brca1", "gene:brca2", "gene:tp53"],
        "drug:pirfenidone": ["gene:loxl1", "gene:loxl2", "gene:loxl3"],
        "drug:trametinib": ["gene:braf", "gene:kras", "gene:egfr"],
        "drug:metformin": ["gene:mtor", "gene:pik3ca"],
    }
    treated = {
        "drug:gefitinib": "disease:lung_cancer",
        "drug:olaparib": "disease:breast_cancer",
        "drug:pirfenidone": "disease:fibrosis",
        "drug:trametinib": "disease:glioblastoma",
        "drug:metformin": "disease:glioblastoma",
    }
    for drug_id, gene_ids in targets.items():
        for gene_id in gene_ids:
            graph.merge_edge(
                BioEdge(
                    source_id=drug_id,
                    target_id=gene_id,
                    edge_type=EdgeType.DRUG_TARGET,
                    score=0.78,
                    sources=["demo"],
                )
            )
        graph.merge_edge(
            BioEdge(
                source_id=drug_id,
                target_id=treated[drug_id],
                edge_type=EdgeType.DRUG_DISEASE,
                score=0.84,
                sources=["demo"],
            )
        )
    return graph


__all__ = ["build_demo_graph", "demo_articles"]
