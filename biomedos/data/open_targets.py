"""Open Targets GraphQL client."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from biomedos.data.base_client import AsyncAPIClient


class GeneInfo(BaseModel):
    """A concise Open Targets gene record."""

    ensembl_id: str
    symbol: str
    name: str | None = None
    biotype: str | None = None
    description: str | None = None


class DiseaseAssociation(BaseModel):
    """A gene-to-disease association from Open Targets."""

    disease_id: str
    disease_name: str
    score: float
    evidence_count: int = 0


class DrugAssociation(BaseModel):
    """A disease-associated drug record."""

    drug_id: str
    drug_name: str
    phase: int | None = None
    status: str | None = None
    mechanism: str | None = None


class TractabilityEntry(BaseModel):
    """Tractability evidence from Open Targets."""

    modality: str
    label: str
    value: bool


class OpenTargetsClient(AsyncAPIClient):
    """Client for the Open Targets GraphQL platform."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the Open Targets client."""

        super().__init__(
            "https://api.platform.opentargets.org/api/v4/graphql",
            requests_per_second=5.0,
            max_concurrency=3,
            client=client,
        )

    async def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Execute a GraphQL query."""

        payload = await self._request_json(
            "POST",
            "",
            json_body={"query": query, "variables": variables},
        )
        data = payload.get("data", {})
        if not isinstance(data, dict):
            return {}
        return data

    async def get_gene_info(self, ensembl_id: str) -> GeneInfo | None:
        """Fetch core metadata for a target gene."""

        query = """
        query Target($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            biotype
            functionDescriptions
          }
        }
        """
        data = await self._graphql(query, {"ensemblId": ensembl_id})
        target = data.get("target", {})
        if not isinstance(target, dict) or not target:
            return None
        descriptions = target.get("functionDescriptions", [])
        description = descriptions[0] if isinstance(descriptions, list) and descriptions else None
        return GeneInfo(
            ensembl_id=str(target.get("id", ensembl_id)),
            symbol=str(target.get("approvedSymbol", "")),
            name=str(target.get("approvedName"))
            if target.get("approvedName") is not None
            else None,
            biotype=str(target.get("biotype")) if target.get("biotype") is not None else None,
            description=str(description) if description is not None else None,
        )

    async def get_disease_associations(
        self,
        ensembl_id: str,
        *,
        limit: int = 10,
    ) -> list[DiseaseAssociation]:
        """Fetch disease associations for a target gene."""

        query = """
        query Associations($ensemblId: String!, $size: Int!) {
          target(ensemblId: $ensemblId) {
            associatedDiseases(page: {index: 0, size: $size}) {
              rows {
                disease {
                  id
                  name
                }
                score
                datasourceScores {
                  score
                }
              }
            }
          }
        }
        """
        data = await self._graphql(query, {"ensemblId": ensembl_id, "size": limit})
        target = data.get("target", {})
        if not isinstance(target, dict):
            return []
        associations = target.get("associatedDiseases", {})
        rows = associations.get("rows", []) if isinstance(associations, dict) else []
        results: list[DiseaseAssociation] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            disease = row.get("disease", {})
            datasource_scores = row.get("datasourceScores", [])
            evidence_count = len(datasource_scores) if isinstance(datasource_scores, list) else 0
            if not isinstance(disease, dict):
                continue
            results.append(
                DiseaseAssociation(
                    disease_id=str(disease.get("id", "")),
                    disease_name=str(disease.get("name", "")),
                    score=float(row.get("score", 0.0)),
                    evidence_count=evidence_count,
                )
            )
        return results

    async def get_drugs_for_disease(
        self, disease_id: str, *, limit: int = 10
    ) -> list[DrugAssociation]:
        """Fetch drugs associated with a disease."""

        query = """
        query DiseaseDrugs($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            knownDrugs(size: $size) {
              rows {
                drug {
                  id
                  name
                  maximumClinicalTrialPhase
                }
                status
                mechanismOfAction
              }
            }
          }
        }
        """
        data = await self._graphql(query, {"diseaseId": disease_id, "size": limit})
        disease = data.get("disease", {})
        if not isinstance(disease, dict):
            return []
        known_drugs = disease.get("knownDrugs", {})
        rows = known_drugs.get("rows", []) if isinstance(known_drugs, dict) else []
        drugs: list[DrugAssociation] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            drug = row.get("drug", {})
            if not isinstance(drug, dict):
                continue
            drugs.append(
                DrugAssociation(
                    drug_id=str(drug.get("id", "")),
                    drug_name=str(drug.get("name", "")),
                    phase=self._safe_int(drug.get("maximumClinicalTrialPhase")),
                    status=str(row.get("status")) if row.get("status") is not None else None,
                    mechanism=(
                        str(row.get("mechanismOfAction"))
                        if row.get("mechanismOfAction") is not None
                        else None
                    ),
                )
            )
        return drugs

    async def get_tractability(self, ensembl_id: str) -> list[TractabilityEntry]:
        """Fetch tractability evidence for a target gene."""

        query = """
        query Tractability($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            tractability {
              modality
              label
              value
            }
          }
        }
        """
        data = await self._graphql(query, {"ensemblId": ensembl_id})
        target = data.get("target", {})
        if not isinstance(target, dict):
            return []
        tractability = target.get("tractability", [])
        if not isinstance(tractability, list):
            return []
        return [
            TractabilityEntry(
                modality=str(item.get("modality", "")),
                label=str(item.get("label", "")),
                value=bool(item.get("value", False)),
            )
            for item in tractability
            if isinstance(item, dict)
        ]

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """Safely coerce a JSON scalar to an integer."""

        if value is None:
            return None
        return int(str(value))
