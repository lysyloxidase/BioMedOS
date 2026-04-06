"""Shared pytest fixtures for BioMedOS."""

from __future__ import annotations

from typing import Any

import pytest

from biomedos.data.pubmed import PubMedArticle
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import (
    BioEdge,
    CellTypeNode,
    ClinicalTrialNode,
    CompoundNode,
    DiseaseNode,
    DrugNode,
    EdgeType,
    GeneNode,
    PathwayNode,
    PhenotypeNode,
    ProteinNode,
    PublicationNode,
    SideEffectNode,
    TissueNode,
)


class MockOllama:
    """Minimal async mock that mimics the Ollama client interface."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize the mock client."""

        self.responses = responses or []
        self.calls: list[dict[str, object]] = []

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Return the next queued response."""

        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "system": system,
                "options": options,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return "{}"


@pytest.fixture
def mock_ollama() -> MockOllama:
    """Provide a configurable mock Ollama client."""

    return MockOllama()


@pytest.fixture
def sample_articles() -> list[PubMedArticle]:
    """Return five sample PubMed articles."""

    return [
        PubMedArticle(
            pmid=f"100{i}",
            title=f"Sample article {i}",
            abstract=f"Abstract {i} on EGFR and BRCA1 biology.",
            journal="BioMed Journal",
            authors=["Doe J", "Smith A"],
            year=2020 + i,
            mesh_terms=["EGFR", "BRCA1"],
            doi=f"10.1000/sample-{i}",
            keywords=["egfr", "brca1"],
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_kg() -> KnowledgeGraph:
    """Build a sample graph with 30 nodes and 50 edges across all node types."""

    knowledge_graph = KnowledgeGraph()
    nodes = [
        GeneNode(id="gene:tp53", name="TP53", symbol="TP53", chromosome="17"),
        GeneNode(id="gene:egfr", name="EGFR", symbol="EGFR", chromosome="7"),
        GeneNode(id="gene:brca1", name="BRCA1", symbol="BRCA1", chromosome="17"),
        GeneNode(id="gene:alk", name="ALK", symbol="ALK", chromosome="2"),
        GeneNode(id="gene:loxl2", name="LOXL2", symbol="LOXL2", chromosome="8"),
        ProteinNode(
            id="protein:tp53", name="p53 protein", uniprot_id="P04637", sequence_length=393
        ),
        ProteinNode(
            id="protein:egfr", name="EGFR protein", uniprot_id="P00533", sequence_length=1210
        ),
        DiseaseNode(id="disease:lung_cancer", name="Lung Cancer", efo_id="EFO:0001071"),
        DiseaseNode(id="disease:breast_cancer", name="Breast Cancer", efo_id="EFO:0000305"),
        DiseaseNode(id="disease:fibrosis", name="Fibrosis", efo_id="EFO:0000400"),
        DrugNode(id="drug:gefitinib", name="Gefitinib", chembl_id="CHEMBL939", max_phase=4),
        DrugNode(id="drug:olaparib", name="Olaparib", chembl_id="CHEMBL521686", max_phase=4),
        DrugNode(id="drug:pirfenidone", name="Pirfenidone", chembl_id="CHEMBL1744", max_phase=4),
        CompoundNode(id="compound:cmpd1", name="Compound One", smiles="CCO"),
        PathwayNode(id="pathway:egfr", name="EGFR Signaling", reactome_id="R-HSA-177929"),
        PathwayNode(id="pathway:dna", name="DNA Repair", reactome_id="R-HSA-73894"),
        PhenotypeNode(id="phenotype:cough", name="Chronic Cough", hpo_id="HP:0012735"),
        PhenotypeNode(id="phenotype:fibrotic_skin", name="Skin Fibrosis", hpo_id="HP:0001075"),
        CellTypeNode(id="cell:epithelial", name="Epithelial Cell"),
        TissueNode(id="tissue:lung", name="Lung Tissue"),
        SideEffectNode(id="side_effect:rash", name="Rash"),
        ClinicalTrialNode(id="trial:nct0001", name="NCT0001", nct_id="NCT0001", phase="Phase 2"),
        PublicationNode(id="publication:1001", name="EGFR in Lung Cancer", pmid="1001", year=2020),
        PublicationNode(
            id="publication:1002", name="BRCA1 and Breast Cancer", pmid="1002", year=2021
        ),
        PublicationNode(id="publication:1003", name="LOXL2 in Fibrosis", pmid="1003", year=2022),
        PublicationNode(id="publication:1004", name="TP53 review", pmid="1004", year=2023),
        PublicationNode(id="publication:1005", name="ALK signaling", pmid="1005", year=2023),
        PublicationNode(
            id="publication:1006", name="EGFR fibrosis crossover", pmid="1006", year=2024
        ),
        PublicationNode(id="publication:1007", name="BRCA1 in fibrosis", pmid="1007", year=2024),
        PublicationNode(id="publication:1008", name="TP53 in lung cancer", pmid="1008", year=2025),
    ]
    for node in nodes:
        knowledge_graph.add_node(node)

    edges = [
        BioEdge(
            source_id="gene:tp53",
            target_id="disease:breast_cancer",
            edge_type=EdgeType.GENE_DISEASE,
            score=0.82,
        ),
        BioEdge(
            source_id="gene:egfr",
            target_id="disease:lung_cancer",
            edge_type=EdgeType.GENE_DISEASE,
            score=0.97,
        ),
        BioEdge(
            source_id="gene:brca1",
            target_id="disease:breast_cancer",
            edge_type=EdgeType.GENE_DISEASE,
            score=0.99,
        ),
        BioEdge(
            source_id="gene:loxl2",
            target_id="disease:fibrosis",
            edge_type=EdgeType.GENE_DISEASE,
            score=0.91,
        ),
        BioEdge(
            source_id="gene:alk",
            target_id="disease:lung_cancer",
            edge_type=EdgeType.GENE_DISEASE,
            score=0.88,
        ),
        BioEdge(
            source_id="gene:egfr",
            target_id="pathway:egfr",
            edge_type=EdgeType.GENE_PATHWAY,
            score=0.95,
        ),
        BioEdge(
            source_id="gene:brca1",
            target_id="pathway:dna",
            edge_type=EdgeType.GENE_PATHWAY,
            score=0.96,
        ),
        BioEdge(
            source_id="gene:tp53", target_id="gene:brca1", edge_type=EdgeType.GENE_GENE, score=0.7
        ),
        BioEdge(
            source_id="gene:egfr", target_id="gene:alk", edge_type=EdgeType.GENE_GENE, score=0.74
        ),
        BioEdge(
            source_id="protein:tp53",
            target_id="protein:egfr",
            edge_type=EdgeType.PROTEIN_PROTEIN,
            score=0.63,
        ),
        BioEdge(
            source_id="drug:gefitinib",
            target_id="gene:egfr",
            edge_type=EdgeType.DRUG_TARGET,
            score=0.98,
        ),
        BioEdge(
            source_id="drug:olaparib",
            target_id="gene:brca1",
            edge_type=EdgeType.DRUG_TARGET,
            score=0.94,
        ),
        BioEdge(
            source_id="drug:pirfenidone",
            target_id="gene:loxl2",
            edge_type=EdgeType.DRUG_TARGET,
            score=0.85,
        ),
        BioEdge(
            source_id="drug:gefitinib",
            target_id="disease:lung_cancer",
            edge_type=EdgeType.DRUG_DISEASE,
            score=0.93,
        ),
        BioEdge(
            source_id="drug:olaparib",
            target_id="disease:breast_cancer",
            edge_type=EdgeType.DRUG_DISEASE,
            score=0.92,
        ),
        BioEdge(
            source_id="drug:gefitinib",
            target_id="drug:olaparib",
            edge_type=EdgeType.DRUG_DRUG,
            score=0.51,
        ),
        BioEdge(
            source_id="drug:gefitinib",
            target_id="side_effect:rash",
            edge_type=EdgeType.DRUG_SIDE_EFFECT,
            score=0.77,
        ),
        BioEdge(
            source_id="compound:cmpd1",
            target_id="gene:alk",
            edge_type=EdgeType.COMPOUND_TARGET,
            score=0.68,
        ),
        BioEdge(
            source_id="disease:lung_cancer",
            target_id="phenotype:cough",
            edge_type=EdgeType.DISEASE_PHENOTYPE,
            score=0.81,
        ),
        BioEdge(
            source_id="disease:fibrosis",
            target_id="phenotype:fibrotic_skin",
            edge_type=EdgeType.DISEASE_PHENOTYPE,
            score=0.86,
        ),
        BioEdge(
            source_id="disease:lung_cancer",
            target_id="pathway:egfr",
            edge_type=EdgeType.DISEASE_PATHWAY,
            score=0.89,
        ),
        BioEdge(
            source_id="disease:breast_cancer",
            target_id="pathway:dna",
            edge_type=EdgeType.DISEASE_PATHWAY,
            score=0.9,
        ),
        BioEdge(
            source_id="gene:egfr",
            target_id="tissue:lung",
            edge_type=EdgeType.GENE_TISSUE,
            score=0.8,
        ),
        BioEdge(
            source_id="pathway:egfr",
            target_id="pathway:dna",
            edge_type=EdgeType.PATHWAY_PATHWAY,
            score=0.61,
        ),
        BioEdge(
            source_id="publication:1001",
            target_id="gene:egfr",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1001",
            target_id="disease:lung_cancer",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1002",
            target_id="gene:brca1",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1002",
            target_id="disease:breast_cancer",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1003",
            target_id="gene:loxl2",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1003",
            target_id="disease:fibrosis",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="drug:gefitinib",
            target_id="disease:fibrosis",
            edge_type=EdgeType.DRUG_CONTRAINDICATION,
            score=0.55,
        ),
        BioEdge(
            source_id="gene:loxl2",
            target_id="phenotype:fibrotic_skin",
            edge_type=EdgeType.GENE_PHENOTYPE,
            score=0.83,
        ),
        BioEdge(
            source_id="trial:nct0001",
            target_id="drug:gefitinib",
            edge_type=EdgeType.TRIAL_DRUG,
            score=0.75,
        ),
        BioEdge(
            source_id="publication:1004",
            target_id="gene:tp53",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1004",
            target_id="disease:breast_cancer",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1005",
            target_id="gene:alk",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1005",
            target_id="disease:lung_cancer",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1006",
            target_id="gene:egfr",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1006",
            target_id="disease:fibrosis",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1007",
            target_id="gene:brca1",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1007",
            target_id="disease:fibrosis",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1008",
            target_id="gene:tp53",
            edge_type=EdgeType.PUBLICATION_GENE,
            score=0.7,
        ),
        BioEdge(
            source_id="publication:1008",
            target_id="disease:lung_cancer",
            edge_type=EdgeType.PUBLICATION_DISEASE,
            score=0.7,
        ),
        BioEdge(
            source_id="gene:tp53",
            target_id="pathway:dna",
            edge_type=EdgeType.GENE_PATHWAY,
            score=0.78,
        ),
        BioEdge(
            source_id="gene:brca1", target_id="gene:tp53", edge_type=EdgeType.GENE_GENE, score=0.71
        ),
        BioEdge(
            source_id="drug:olaparib",
            target_id="gene:alk",
            edge_type=EdgeType.DRUG_TARGET,
            score=0.62,
        ),
        BioEdge(
            source_id="disease:fibrosis",
            target_id="pathway:egfr",
            edge_type=EdgeType.DISEASE_PATHWAY,
            score=0.58,
        ),
        BioEdge(
            source_id="gene:alk",
            target_id="pathway:egfr",
            edge_type=EdgeType.GENE_PATHWAY,
            score=0.79,
        ),
        BioEdge(
            source_id="compound:cmpd1",
            target_id="gene:egfr",
            edge_type=EdgeType.COMPOUND_TARGET,
            score=0.59,
        ),
        BioEdge(
            source_id="drug:pirfenidone",
            target_id="disease:fibrosis",
            edge_type=EdgeType.DRUG_DISEASE,
            score=0.88,
        ),
    ]
    assert len(edges) == 50

    for edge in edges:
        knowledge_graph.add_edge(edge)

    return knowledge_graph
