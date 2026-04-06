"""Typed biomedical knowledge graph schema models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeType(StrEnum):
    """Supported biomedical node types."""

    GENE = "Gene"
    PROTEIN = "Protein"
    DISEASE = "Disease"
    DRUG = "Drug"
    COMPOUND = "Compound"
    PATHWAY = "Pathway"
    PHENOTYPE = "Phenotype"
    CELL_TYPE = "CellType"
    TISSUE = "Tissue"
    SIDE_EFFECT = "SideEffect"
    CLINICAL_TRIAL = "ClinicalTrial"
    PUBLICATION = "Publication"


class EdgeType(StrEnum):
    """Supported biomedical relationship types."""

    GENE_DISEASE = "gene_associated_with_disease"
    GENE_PATHWAY = "gene_in_pathway"
    GENE_GENE = "gene_interacts_with_gene"
    PROTEIN_PROTEIN = "protein_interacts_with_protein"
    DRUG_TARGET = "drug_targets_gene"
    DRUG_DISEASE = "drug_treats_disease"
    DRUG_DRUG = "drug_interacts_with_drug"
    DRUG_SIDE_EFFECT = "drug_causes_side_effect"
    COMPOUND_TARGET = "compound_binds_target"
    DISEASE_PHENOTYPE = "disease_has_phenotype"
    DISEASE_PATHWAY = "disease_involves_pathway"
    GENE_TISSUE = "gene_expressed_in_tissue"
    PATHWAY_PATHWAY = "pathway_crosstalks_with"
    PUBLICATION_GENE = "publication_mentions_gene"
    PUBLICATION_DISEASE = "publication_mentions_disease"
    DRUG_CONTRAINDICATION = "drug_contraindicated_for"
    GENE_PHENOTYPE = "gene_associated_with_phenotype"
    TRIAL_DRUG = "trial_investigates_drug"


class BioNode(BaseModel):
    """Base biomedical graph node."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    node_type: NodeType
    properties: dict[str, object] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class GeneNode(BioNode):
    """Gene node metadata."""

    node_type: NodeType = NodeType.GENE
    ensembl_id: str | None = None
    symbol: str | None = None
    chromosome: str | None = None
    description: str | None = None


class DrugNode(BioNode):
    """Drug node metadata."""

    node_type: NodeType = NodeType.DRUG
    chembl_id: str | None = None
    smiles: str | None = None
    max_phase: int | None = None
    mechanism: str | None = None


class DiseaseNode(BioNode):
    """Disease node metadata."""

    node_type: NodeType = NodeType.DISEASE
    efo_id: str | None = None
    therapeutic_area: str | None = None


class ProteinNode(BioNode):
    """Protein node metadata."""

    node_type: NodeType = NodeType.PROTEIN
    uniprot_id: str | None = None
    sequence_length: int | None = None
    function: str | None = None


class PathwayNode(BioNode):
    """Pathway node metadata."""

    node_type: NodeType = NodeType.PATHWAY
    reactome_id: str | None = None
    pathway_type: str | None = None


class PhenotypeNode(BioNode):
    """Phenotype node metadata."""

    node_type: NodeType = NodeType.PHENOTYPE
    hpo_id: str | None = None
    frequency: str | None = None


class PublicationNode(BioNode):
    """Publication node metadata."""

    node_type: NodeType = NodeType.PUBLICATION
    pmid: str | None = None
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)


class CompoundNode(BioNode):
    """Compound node metadata."""

    node_type: NodeType = NodeType.COMPOUND
    smiles: str | None = None


class CellTypeNode(BioNode):
    """Cell type node metadata."""

    node_type: NodeType = NodeType.CELL_TYPE


class TissueNode(BioNode):
    """Tissue node metadata."""

    node_type: NodeType = NodeType.TISSUE


class SideEffectNode(BioNode):
    """Side-effect node metadata."""

    node_type: NodeType = NodeType.SIDE_EFFECT


class ClinicalTrialNode(BioNode):
    """Clinical trial node metadata."""

    node_type: NodeType = NodeType.CLINICAL_TRIAL
    nct_id: str | None = None
    phase: str | None = None


NODE_MODEL_BY_TYPE: dict[NodeType, type[BioNode]] = {
    NodeType.GENE: GeneNode,
    NodeType.PROTEIN: ProteinNode,
    NodeType.DISEASE: DiseaseNode,
    NodeType.DRUG: DrugNode,
    NodeType.COMPOUND: CompoundNode,
    NodeType.PATHWAY: PathwayNode,
    NodeType.PHENOTYPE: PhenotypeNode,
    NodeType.CELL_TYPE: CellTypeNode,
    NodeType.TISSUE: TissueNode,
    NodeType.SIDE_EFFECT: SideEffectNode,
    NodeType.CLINICAL_TRIAL: ClinicalTrialNode,
    NodeType.PUBLICATION: PublicationNode,
}


class BioEdge(BaseModel):
    """Base biomedical graph edge."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    score: float = 1.0
    properties: dict[str, object] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


def create_node_model(payload: dict[str, Any]) -> BioNode:
    """Instantiate the correct node subclass from raw payload data."""

    node_type = payload.get("node_type", NodeType.GENE)
    if not isinstance(node_type, NodeType):
        node_type = NodeType(str(node_type))
    model_cls = NODE_MODEL_BY_TYPE.get(node_type, BioNode)
    return model_cls.model_validate({**payload, "node_type": node_type})


def create_edge_model(payload: dict[str, Any]) -> BioEdge:
    """Instantiate a validated biomedical edge from raw payload data."""

    edge_type = payload.get("edge_type", EdgeType.GENE_GENE)
    if not isinstance(edge_type, EdgeType):
        edge_type = EdgeType(str(edge_type))
    return BioEdge.model_validate({**payload, "edge_type": edge_type})
