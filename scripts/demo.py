"""Interactive BioMedOS demo runner."""
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomedos.agents.drug_repurposer import DrugRepurposerAgent
from biomedos.agents.geneticist import GeneticistAgent
from biomedos.agents.graph_explorer import GraphExplorerAgent
from biomedos.agents.hypothesis_generator import HypothesisGeneratorAgent
from biomedos.agents.link_predictor import LinkPredictorAgent
from biomedos.agents.literature import LiteratureAgent
from biomedos.agents.pathway_analyst import PathwayAnalystAgent
from biomedos.config import Settings, get_settings, resolve_project_path
from biomedos.core.llm_client import OllamaClient
from biomedos.core.logging import configure_logging
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.data.pubmed import PubMedAnnotation, PubMedArticle, PubMedClient
from biomedos.demo_data import demo_articles
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioEdge, DiseaseNode, DrugNode, EdgeType, GeneNode, PathwayNode
from biomedos.ml.link_prediction import LinkPredictionPipeline
from biomedos.orchestration.state import Task, TaskType
from biomedos.rag.pubmed_indexer import PubMedIndexer


class StaticPubMedClient(PubMedClient):
    """Small in-memory PubMed client for deterministic demos."""

    def __init__(self, articles: Sequence[PubMedArticle]) -> None:
        """Initialize the static article index."""

        self._articles = {article.pmid: article for article in articles}

    async def search(self, query: str, *, max_results: int = 20) -> list[str]:
        """Rank local articles by token overlap."""

        query_terms = set(query.lower().split())
        scored: list[tuple[int, str]] = []
        for article in self._articles.values():
            haystack = f"{article.title} {article.abstract}".lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score > 0:
                scored.append((score, article.pmid))
        scored.sort(reverse=True)
        return [pmid for _, pmid in scored[:max_results]]

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Return local articles for the requested PMIDs."""

        return [self._articles[pmid] for pmid in pmids if pmid in self._articles]

    async def fetch_annotations(self, pmids: list[str]) -> dict[str, list[PubMedAnnotation]]:
        """Return an empty annotation payload for the demo."""

        return {pmid: [] for pmid in pmids}

    async def close(self) -> None:
        """No-op close method for compatibility."""


def build_demo_graph(settings: Settings) -> KnowledgeGraph:
    """Build a compact graph with enough structure for the full demo."""

    graph = KnowledgeGraph()
    genes = settings.DEFAULT_GENES[:20]
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
    for index, symbol in enumerate(genes):
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
        if index < len(genes) - 1:
            graph.merge_edge(
                BioEdge(
                    source_id=gene_id,
                    target_id=f"gene:{genes[index + 1].lower()}",
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

async def maybe_check_ollama(settings: Settings, console: Console) -> OllamaClient | None:
    """Return an Ollama client when available."""

    llm = OllamaClient(settings=settings)
    available = await llm.health_check()
    if not available:
        console.print(
            Panel("Ollama is offline. The demo will use deterministic fallbacks.", title="LLM")
        )
        await llm.close()
        return None
    models = ", ".join(await llm.list_models())
    console.print(
        Panel(f"Ollama is online.\nInstalled models: {models or 'none listed'}", title="LLM")
    )
    return llm


async def prepare_index(
    vector_store: ChromaVectorStore,
    console: Console,
) -> tuple[StaticPubMedClient, PubMedIndexer]:
    """Index local demo literature into the vector store."""

    client = StaticPubMedClient(demo_articles())
    indexer = PubMedIndexer(client, vector_store)
    indexed = indexer.index_articles(demo_articles(), source="demo")
    stats = indexer.get_stats()
    console.print(
        Panel(
            f"Indexed {indexed} local articles.\n"
            f"Sources: {stats['by_source']}\nYears: {stats['by_year']}",
            title="PubMed Index",
        )
    )
    return client, indexer


def render_graph_summary(graph: KnowledgeGraph, console: Console) -> None:
    """Print compact graph statistics."""

    stats = graph.stats()
    node_types = stats.get("node_types")
    edge_types = stats.get("edge_types")
    table = Table(title="Mini Graph")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Nodes", str(stats["nodes"]))
    table.add_row("Edges", str(stats["edges"]))
    table.add_row(
        "Node types",
        ", ".join(sorted(node_types.keys())) if isinstance(node_types, dict) else "n/a",
    )
    table.add_row(
        "Edge types",
        (
            ", ".join(sorted(edge_types.keys())[:6]) + " ..."
            if isinstance(edge_types, dict)
            else "n/a"
        ),
    )
    console.print(table)


def render_metrics(pipeline: LinkPredictionPipeline, console: Console) -> None:
    """Train GraphSAGE for 50 epochs and show metrics."""

    summary = pipeline.train(epochs=50)
    metrics = summary.metrics
    table = Table(title="GraphSAGE Metrics")
    for column in ("AUROC", "AUPRC", "Hits@10", "Hits@50", "MRR"):
        table.add_column(column)
    table.add_row(
        f"{metrics.auroc:.3f}",
        f"{metrics.auprc:.3f}",
        f"{metrics.hits_at_10:.3f}",
        f"{metrics.hits_at_50:.3f}",
        f"{metrics.mrr:.3f}",
    )
    console.print(table)


async def run_agent_showcase(
    graph: KnowledgeGraph,
    vector_store: ChromaVectorStore,
    settings: Settings,
    llm: OllamaClient | None,
    pubmed_client: StaticPubMedClient,
    indexer: PubMedIndexer,
    console: Console,
) -> None:
    """Run seven example tasks through different agents."""

    literature = LiteratureAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=settings,
        pubmed_client=pubmed_client,
        indexer=indexer,
    )
    agents = [
        (
            "Literature",
            literature,
            Task(
                id="demo-lit",
                type=TaskType.LITERATURE,
                description="Literature",
                payload={"query": "EGFR signaling in lung cancer"},
            ),
        ),
        (
            "Graph Explorer",
            GraphExplorerAgent(
                llm_client=llm, knowledge_graph=graph, vector_store=vector_store, settings=settings
            ),
            Task(
                id="demo-graph",
                type=TaskType.GRAPH_EXPLORER,
                description="EGFR to lung cancer",
                payload={"source_id": "gene:egfr", "target_id": "disease:lung_cancer"},
            ),
        ),
        (
            "Link Predictor",
            LinkPredictorAgent(
                llm_client=llm, knowledge_graph=graph, vector_store=vector_store, settings=settings
            ),
            Task(
                id="demo-link",
                type=TaskType.LINK_PREDICTOR,
                description="EGFR",
                payload={"source": "EGFR", "target_type": "Disease", "top_k": 5, "epochs": 50},
            ),
        ),
        (
            "Drug Repurposer",
            DrugRepurposerAgent(
                llm_client=llm, knowledge_graph=graph, vector_store=vector_store, settings=settings
            ),
            Task(
                id="demo-repurpose",
                type=TaskType.DRUG_REPURPOSER,
                description="Fibrosis",
                payload={"disease": "Fibrosis", "epochs": 40},
            ),
        ),
        (
            "Geneticist",
            GeneticistAgent(
                llm_client=llm, knowledge_graph=graph, vector_store=vector_store, settings=settings
            ),
            Task(
                id="demo-gene",
                type=TaskType.GENETICIST,
                description="BRCA1",
                payload={"gene": "BRCA1"},
            ),
        ),
        (
            "Pathway Analyst",
            PathwayAnalystAgent(
                llm_client=llm, knowledge_graph=graph, vector_store=vector_store, settings=settings
            ),
            Task(
                id="demo-pathway",
                type=TaskType.PATHWAY_ANALYST,
                description="EGFR ALK TP53 BRCA1",
                payload={"genes": ["EGFR", "ALK", "TP53", "BRCA1"]},
            ),
        ),
        (
            "Hypothesis Generator",
            HypothesisGeneratorAgent(
                llm_client=llm, knowledge_graph=graph, vector_store=vector_store, settings=settings
            ),
            Task(
                id="demo-hypothesis",
                type=TaskType.HYPOTHESIS_GENERATOR,
                description="LOXL2",
                payload={"source": "LOXL2", "target_type": "Disease", "top_k": 5, "epochs": 35},
            ),
        ),
    ]

    for label, agent, task in agents:
        result = await agent.run(task)
        console.print(
            Panel(result.summary, title=label, subtitle=f"confidence={result.confidence:.2f}")
        )


async def async_main() -> int:
    """Run the full BioMedOS demo."""

    settings = get_settings().model_copy(update={"FAST_LOCAL_MODE": True})
    console = Console()
    configure_logging()

    console.print(
        Panel(
            "BioMedOS interactive demo\n"
            "1. Check Ollama\n"
            "2. Build mini graph\n"
            "3. Index PubMed\n"
            "4. Train GraphSAGE\n"
            "5. Run seven agents",
            title="Demo",
        )
    )
    llm = None if settings.FAST_LOCAL_MODE else await maybe_check_ollama(settings, console)
    if settings.FAST_LOCAL_MODE:
        console.print(
            Panel(
                "Fast local mode is enabled. The demo will use deterministic agent fallbacks.",
                title="LLM",
            )
        )

    graph = build_demo_graph(settings)
    render_graph_summary(graph, console)
    graph_path = resolve_project_path("data/demo_knowledge_graph.gpickle")
    graph.save(graph_path)
    console.print(f"Saved demo graph to {graph_path}")

    vector_store = ChromaVectorStore(persist_dir=":memory:", collection_name="demo_pubmed")
    pubmed_client, indexer = await prepare_index(vector_store, console)

    pipeline = LinkPredictionPipeline(
        graph,
        model_name="graphsage",
        edge_type=EdgeType.GENE_DISEASE,
        settings=settings,
    )
    render_metrics(pipeline, console)
    await run_agent_showcase(graph, vector_store, settings, llm, pubmed_client, indexer, console)

    if llm is not None:
        await llm.close()
    await pubmed_client.close()
    return 0


def main() -> int:
    """Script entrypoint for the interactive demo."""

    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
