"""Build a starter biomedical knowledge graph from local-first data sources."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from tqdm import tqdm  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomedos.config import get_settings, resolve_project_path
from biomedos.core.logging import configure_logging
from biomedos.data.open_targets import OpenTargetsClient
from biomedos.data.pubmed import PubMedClient
from biomedos.data.string_db import StringDBClient
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import (
    BioEdge,
    DiseaseNode,
    DrugNode,
    EdgeType,
    GeneNode,
    PublicationNode,
)

SUPPORTED_SOURCES = (
    "open_targets",
    "string_db",
    "pubmed",
    "chembl",
    "uniprot",
    "rxnorm",
    "reactome",
    "disgenet",
)


async def ingest_open_targets(knowledge_graph: KnowledgeGraph, genes: list[str]) -> None:
    """Ingest disease and drug evidence from Open Targets."""

    client = OpenTargetsClient()
    try:
        for gene_symbol in tqdm(genes, desc="Open Targets genes", leave=False):
            gene_id = f"gene:{gene_symbol.lower()}"
            knowledge_graph.merge_node(
                GeneNode(
                    id=gene_id,
                    name=gene_symbol,
                    symbol=gene_symbol,
                    sources=["user_seed", "open_targets"],
                )
            )
            associations = await client.get_disease_associations(gene_symbol, limit=5)
            for association in associations:
                disease_id = f"disease:{association.disease_id.lower()}"
                knowledge_graph.merge_node(
                    DiseaseNode(
                        id=disease_id,
                        name=association.disease_name,
                        efo_id=association.disease_id,
                        sources=["open_targets"],
                    )
                )
                knowledge_graph.merge_edge(
                    BioEdge(
                        source_id=gene_id,
                        target_id=disease_id,
                        edge_type=EdgeType.GENE_DISEASE,
                        score=association.score,
                        sources=["open_targets"],
                    )
                )
                for drug in await client.get_drugs_for_disease(association.disease_id, limit=3):
                    drug_id = f"drug:{drug.drug_id.lower()}"
                    knowledge_graph.merge_node(
                        DrugNode(
                            id=drug_id,
                            name=drug.drug_name,
                            max_phase=drug.phase,
                            sources=["open_targets"],
                        )
                    )
                    knowledge_graph.merge_edge(
                        BioEdge(
                            source_id=drug_id,
                            target_id=disease_id,
                            edge_type=EdgeType.DRUG_DISEASE,
                            score=0.8,
                            properties={
                                "status": drug.status or "",
                                "mechanism": drug.mechanism or "",
                            },
                            sources=["open_targets"],
                        )
                    )
    finally:
        await client.close()


async def ingest_string_db(knowledge_graph: KnowledgeGraph, genes: list[str]) -> None:
    """Ingest protein interaction edges from STRING DB."""

    client = StringDBClient()
    try:
        interactions = await client.get_network(genes)
        for interaction in tqdm(interactions, desc="STRING DB edges", leave=False):
            source_id = f"gene:{interaction.preferred_name_a.lower()}"
            target_id = f"gene:{interaction.preferred_name_b.lower()}"
            knowledge_graph.merge_node(
                GeneNode(
                    id=source_id,
                    name=interaction.preferred_name_a,
                    symbol=interaction.preferred_name_a,
                )
            )
            knowledge_graph.merge_node(
                GeneNode(
                    id=target_id,
                    name=interaction.preferred_name_b,
                    symbol=interaction.preferred_name_b,
                )
            )
            knowledge_graph.merge_edge(
                BioEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=EdgeType.GENE_GENE,
                    score=interaction.score,
                    properties={"evidence": interaction.evidence or ""},
                    sources=["string_db"],
                )
            )
    finally:
        await client.close()


async def ingest_pubmed(knowledge_graph: KnowledgeGraph, genes: list[str]) -> None:
    """Ingest lightweight publication evidence from PubMed."""

    client = PubMedClient()
    try:
        for gene_symbol in tqdm(genes, desc="PubMed genes", leave=False):
            pmids = await client.search(
                f"{gene_symbol} biomarker OR {gene_symbol} disease", max_results=3
            )
            articles = await client.fetch_abstracts(pmids)
            for article in articles:
                publication_id = f"publication:{article.pmid}"
                knowledge_graph.merge_node(
                    PublicationNode(
                        id=publication_id,
                        name=article.title or article.pmid,
                        pmid=article.pmid,
                        title=article.title,
                        abstract=article.abstract,
                        year=article.year,
                        authors=article.authors,
                        sources=["pubmed"],
                    )
                )
                knowledge_graph.merge_edge(
                    BioEdge(
                        source_id=publication_id,
                        target_id=f"gene:{gene_symbol.lower()}",
                        edge_type=EdgeType.PUBLICATION_GENE,
                        score=0.6,
                        sources=["pubmed"],
                    )
                )
    finally:
        await client.close()


async def build_graph(genes: list[str], sources: list[str], output: str) -> KnowledgeGraph:
    """Build and persist a knowledge graph.

    Args:
        genes: Seed gene list.
        sources: Data source names to ingest.
        output: Output graph path.

    Returns:
        Built knowledge graph.
    """

    logger = configure_logging()
    knowledge_graph = KnowledgeGraph()

    for gene_symbol in genes:
        knowledge_graph.merge_node(
            GeneNode(
                id=f"gene:{gene_symbol.lower()}",
                name=gene_symbol,
                symbol=gene_symbol,
                sources=["user_seed"],
            )
        )

    for source in tqdm(sources, desc="Sources"):
        try:
            if source == "open_targets":
                await ingest_open_targets(knowledge_graph, genes)
            elif source == "string_db":
                await ingest_string_db(knowledge_graph, genes)
            elif source == "pubmed":
                await ingest_pubmed(knowledge_graph, genes)
            else:
                logger.info("Skipping scaffolded source: %s", source)
        except Exception as exc:  # pragma: no cover - defensive CLI behavior
            logger.warning("Source %s failed: %s", source, exc)

    output_path = resolve_project_path(output)
    knowledge_graph.save(output_path)
    logger.info(
        "Saved graph with %s nodes and %s edges to %s",
        knowledge_graph.graph.number_of_nodes(),
        knowledge_graph.graph.number_of_edges(),
        output_path,
    )
    return knowledge_graph


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a BioMedOS knowledge graph.")
    parser.add_argument("--genes", nargs="*", default=settings.DEFAULT_GENES)
    parser.add_argument("--sources", nargs="*", default=list(SUPPORTED_SOURCES))
    parser.add_argument("--output", default=settings.GRAPH_PERSIST_PATH)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    asyncio.run(build_graph(args.genes, args.sources, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
