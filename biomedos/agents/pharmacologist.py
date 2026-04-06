"""Pharmacology specialist agent."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from biomedos.agents.base import BaseAgent
from biomedos.clinical.contraindication_checker import ContraindicationChecker
from biomedos.clinical.ddi_checker import DDIChecker, DDIResult
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.data.chembl import ChEMBLClient
from biomedos.data.openfda import OpenFDAClient
from biomedos.data.rxnorm import RxNormClient
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import NodeType
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class PharmacologistAgent(BaseAgent):
    """Assess drug interactions, PK, ADMET, and safety."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        rxnorm_client: RxNormClient | None = None,
        openfda_client: OpenFDAClient | None = None,
        chembl_client: ChEMBLClient | None = None,
        ddi_checker: DDIChecker | None = None,
        contraindication_checker: ContraindicationChecker | None = None,
    ) -> None:
        """Initialize the pharmacology agent."""

        super().__init__(
            name="pharmacologist",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.rxnorm_client = rxnorm_client or RxNormClient()
        self.openfda_client = openfda_client or OpenFDAClient()
        self.chembl_client = chembl_client or ChEMBLClient()
        self.ddi_checker = ddi_checker or DDIChecker(self.rxnorm_client, self.openfda_client)
        self.contraindication_checker = contraindication_checker or ContraindicationChecker(
            self.openfda_client
        )

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the pharmacology agent."""

        del state
        drugs = self._extract_drugs(task)
        conditions = self._extract_conditions(task)
        interactions = await self.ddi_checker.check(drugs) if len(drugs) > 1 else []

        drug_reports: list[dict[str, object]] = []
        for drug_name in drugs:
            normalized = await self.rxnorm_client.normalize_drug(drug_name)
            rxcui = str(normalized.get("rxcui", ""))
            ndc_codes = await self.rxnorm_client.get_ndc_codes(rxcui) if rxcui else []
            openfda_events = await self.openfda_client.adverse_events(drug_name)
            labels = await self.openfda_client.drug_labels(drug_name)
            recalls = await self.openfda_client.recalls(drug_name)
            contraindications = await self.contraindication_checker.check(drug_name, conditions)
            mechanisms = await self._mechanisms_for_drug(drug_name)
            pk_summary = self._extract_label_section(
                labels,
                ("pharmacokinetics", "clinical_pharmacology", "pharmacodynamics"),
            )
            metabolism = self._extract_label_section(
                labels,
                ("pharmacokinetics", "drug_interactions", "clinical_pharmacology"),
            )
            drug_reports.append(
                {
                    "drug": drug_name,
                    "normalized": normalized,
                    "side_effects": self._top_reactions(openfda_events),
                    "mechanisms": mechanisms,
                    "pk_summary": pk_summary,
                    "metabolism_pathway": metabolism,
                    "ndc_codes": ndc_codes[:5],
                    "recalls": recalls[:3],
                    "contraindications": [
                        item.model_dump(mode="python") for item in contraindications
                    ],
                    "label_summary": labels[:2],
                }
            )

        summary = self._format_summary(drugs, interactions, drug_reports)
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={
                "drugs": drugs,
                "interactions": [item.model_dump(mode="python") for item in interactions],
                "drug_reports": drug_reports,
            },
            confidence=0.8 if drug_reports else 0.2,
        )

    def _extract_drugs(self, task: Task) -> list[str]:
        """Extract one or more drug names from the task."""

        raw_drugs = task.payload.get("drugs")
        if isinstance(raw_drugs, list):
            return [str(drug).strip() for drug in raw_drugs if str(drug).strip()]
        text = str(task.payload.get("drug", task.payload.get("query", task.description)))
        parts = re.split(r",|;|\band\b", text)
        drugs = [part.strip() for part in parts if part.strip()]
        return drugs or [text]

    def _extract_conditions(self, task: Task) -> list[str]:
        """Extract patient conditions from the task payload."""

        raw_conditions = task.payload.get("conditions")
        if not isinstance(raw_conditions, list):
            return []
        return [str(condition).strip() for condition in raw_conditions if str(condition).strip()]

    async def _mechanisms_for_drug(self, drug_name: str) -> list[str]:
        """Collect mechanism data from the KG and ChEMBL when available."""

        drug_node = self.resolve_node(drug_name, node_type=NodeType.DRUG)
        if drug_node is None:
            return []
        fallback_mechanisms = self._kg_target_mechanisms(drug_node.id)
        chembl_id = getattr(drug_node, "chembl_id", None)
        if chembl_id is None:
            return fallback_mechanisms
        try:
            mechanisms = await self.chembl_client.get_mechanisms(str(chembl_id))
        except Exception:
            return fallback_mechanisms
        resolved = [
            mechanism.mechanism or mechanism.action_type or mechanism.target_chembl_id
            for mechanism in mechanisms
        ]
        return resolved or fallback_mechanisms

    def _kg_target_mechanisms(self, drug_id: str) -> list[str]:
        """Build a lightweight mechanism summary from local drug-target edges."""

        edge_type = self.infer_edge_type(NodeType.DRUG, NodeType.GENE)
        if edge_type is None:
            return []
        target_names: list[str] = []
        for edge in self.knowledge_graph.get_edges(
            source_id=drug_id,
            edge_type=edge_type,
        ):
            target = self.knowledge_graph.get_node(edge.target_id)
            if target is not None:
                target_names.append(target.name)
        return [f"Targets {', '.join(sorted(set(target_names)))}"] if target_names else []

    def _top_reactions(self, events: list[dict[str, object]]) -> list[str]:
        """Extract top adverse-event labels from OpenFDA records."""

        counts: Counter[str] = Counter()
        for event in events:
            patient = event.get("patient", {})
            if not isinstance(patient, dict):
                continue
            items = patient.get("reaction", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                term = item.get("reactionmeddrapt")
                if term is not None:
                    counts[str(term)] += 1
        return [term for term, _ in counts.most_common(5)]

    def _extract_label_section(
        self,
        labels: list[dict[str, object]],
        field_names: tuple[str, ...],
    ) -> str | None:
        """Extract a concise label section from OpenFDA results."""

        snippets: list[str] = []
        for label in labels:
            for field_name in field_names:
                value = label.get(field_name)
                if isinstance(value, list):
                    snippets.extend(str(item).strip() for item in value if str(item).strip())
                elif isinstance(value, str) and value.strip():
                    snippets.append(value.strip())
        if not snippets:
            return None
        return " ".join(snippets)[:280].strip()

    def _format_summary(
        self,
        drugs: list[str],
        interactions: Sequence[DDIResult],
        drug_reports: list[dict[str, object]],
    ) -> str:
        """Build a concise safety and PK summary."""

        lines = [f"Pharmacology review for: {', '.join(drugs)}"]
        if interactions:
            lines.append(
                "Interactions: "
                + ", ".join(
                    f"{item.drug_a}/{item.drug_b} ({item.severity.value})" for item in interactions
                )
            )
        else:
            lines.append("Interactions: none detected from the available sources.")
        for report in drug_reports:
            raw_side_effects = report.get("side_effects", [])
            side_effect_values = raw_side_effects if isinstance(raw_side_effects, list) else []
            side_effects = (
                ", ".join(str(value) for value in side_effect_values)
                if side_effect_values
                else "none reported"
            )
            raw_mechanisms = report.get("mechanisms", [])
            mechanism_values = raw_mechanisms if isinstance(raw_mechanisms, list) else []
            mechanisms = (
                ", ".join(str(value) for value in mechanism_values)
                if mechanism_values
                else "no mechanism available"
            )
            pk_summary = (
                str(report["pk_summary"]) if report["pk_summary"] else "no PK summary available"
            )
            metabolism = (
                str(report["metabolism_pathway"])
                if report["metabolism_pathway"]
                else "no metabolism pathway available"
            )
            lines.append(
                f"- {report['drug']}: side effects={side_effects}; "
                f"mechanisms={mechanisms}; PK={pk_summary}; metabolism={metabolism}."
            )
        return "\n".join(lines)
