"""Gene set enrichment analysis."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field
from scipy.stats import fisher_exact

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType


class EnrichmentResult(BaseModel):
    """A single enrichment hit."""

    term_id: str
    term_name: str
    category: str
    overlap: int
    query_size: int
    term_size: int
    p_value: float
    fdr: float
    genes: list[str] = Field(default_factory=list)


class GeneSetEnrichment:
    """Perform gene set enrichment with multiple testing correction."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        """Initialize the enrichment analyzer."""

        self.knowledge_graph = knowledge_graph

    def pathway_enrichment(
        self,
        genes: list[str],
        background: list[str] | None = None,
    ) -> list[EnrichmentResult]:
        """Run pathway enrichment over gene-pathway memberships."""

        return self._run_enrichment(
            genes,
            background,
            category="pathway",
            edge_type=EdgeType.GENE_PATHWAY,
            term_node_type=NodeType.PATHWAY,
        )

    def disease_enrichment(
        self,
        genes: list[str],
        background: list[str] | None = None,
    ) -> list[EnrichmentResult]:
        """Run disease enrichment over gene-disease associations."""

        return self._run_enrichment(
            genes,
            background,
            category="disease",
            edge_type=EdgeType.GENE_DISEASE,
            term_node_type=NodeType.DISEASE,
        )

    def run(self, genes: list[str], background: list[str]) -> list[dict[str, object]]:
        """Run pathway enrichment with backward-compatible output."""

        return [
            result.model_dump(mode="python")
            for result in self.pathway_enrichment(genes, background)
        ]

    def _run_enrichment(
        self,
        genes: list[str],
        background: list[str] | None,
        *,
        category: str,
        edge_type: EdgeType,
        term_node_type: NodeType,
    ) -> list[EnrichmentResult]:
        """Run an enrichment analysis for a given relation family."""

        query_gene_ids = self._resolve_gene_ids(genes)
        if not query_gene_ids:
            return []
        background_ids = self._resolve_background(background)
        if not background_ids:
            background_ids = self._resolve_background(None)

        term_to_genes: dict[str, set[str]] = defaultdict(set)
        for edge in self.knowledge_graph.get_edges(edge_type=edge_type):
            if edge.source_id in background_ids:
                term_to_genes[edge.target_id].add(edge.source_id)

        results: list[EnrichmentResult] = []
        for term_id, term_genes in term_to_genes.items():
            overlap_ids = query_gene_ids & term_genes
            if not overlap_ids:
                continue
            a = len(overlap_ids)
            b = len(query_gene_ids - term_genes)
            c = len(term_genes - query_gene_ids)
            d = max(len(background_ids - (query_gene_ids | term_genes)), 0)
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            term_node = self.knowledge_graph.get_node(term_id)
            if term_node is None or term_node.node_type != term_node_type:
                continue
            overlap_names: list[str] = []
            for gene_id in overlap_ids:
                gene_node = self.knowledge_graph.get_node(gene_id)
                if gene_node is not None:
                    overlap_names.append(gene_node.name)
            results.append(
                EnrichmentResult(
                    term_id=term_id,
                    term_name=term_node.name,
                    category=category,
                    overlap=a,
                    query_size=len(query_gene_ids),
                    term_size=len(term_genes),
                    p_value=float(p_value),
                    fdr=1.0,
                    genes=sorted(overlap_names),
                )
            )

        fdr_values = self._benjamini_hochberg([result.p_value for result in results])
        for result, fdr in zip(results, fdr_values, strict=True):
            result.fdr = float(fdr)
        results.sort(key=lambda item: (item.fdr, item.p_value, -item.overlap))
        return results

    def _resolve_gene_ids(self, genes: list[str]) -> set[str]:
        """Resolve user-supplied gene names or identifiers."""

        resolved: set[str] = set()
        for gene in genes:
            node = self.knowledge_graph.get_node(gene)
            if node is not None and node.node_type == NodeType.GENE:
                resolved.add(node.id)
                continue
            matches = self.knowledge_graph.search_nodes(gene, node_type=NodeType.GENE, limit=1)
            if matches:
                resolved.add(matches[0].id)
        return resolved

    def _resolve_background(self, background: list[str] | None) -> set[str]:
        """Resolve a gene background universe."""

        if background is None:
            return {
                node_id
                for node_id, payload in self.knowledge_graph.graph.nodes(data=True)
                if payload.get("node_type") == NodeType.GENE
            }
        return self._resolve_gene_ids(background)

    def _benjamini_hochberg(self, p_values: list[float]) -> list[float]:
        """Apply Benjamini-Hochberg multiple testing correction."""

        if not p_values:
            return []
        indexed = sorted(enumerate(p_values), key=lambda item: item[1])
        adjusted = [1.0] * len(p_values)
        running_min = 1.0
        total = len(p_values)
        for rank, (index, p_value) in enumerate(reversed(indexed), start=1):
            adjusted_value = min(running_min, p_value * total / max(total - rank + 1, 1))
            running_min = adjusted_value
            adjusted[index] = adjusted_value
        return adjusted


EnrichmentAnalyzer = GeneSetEnrichment
