"""ChEMBL REST API client."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from biomedos.data.base_client import AsyncAPIClient


class ChEMBLDrug(BaseModel):
    """Structured drug metadata from ChEMBL."""

    chembl_id: str
    name: str | None = None
    max_phase: int | None = None
    molecule_type: str | None = None


class ChEMBLMechanism(BaseModel):
    """A mechanism of action record from ChEMBL."""

    target_chembl_id: str
    mechanism: str | None = None
    action_type: str | None = None


class ChEMBLBioactivity(BaseModel):
    """A target bioactivity measurement from ChEMBL."""

    assay_chembl_id: str
    molecule_chembl_id: str
    activity_type: str | None = None
    activity_value: float | None = None
    units: str | None = None


class ChEMBLClient(AsyncAPIClient):
    """Client for the ChEMBL REST API."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the ChEMBL client."""

        super().__init__(
            "https://www.ebi.ac.uk/chembl/api/data",
            requests_per_second=5.0,
            client=client,
        )

    async def get_drug_info(self, chembl_id: str) -> ChEMBLDrug | None:
        """Fetch drug information by ChEMBL identifier."""

        payload = await self._request_json("GET", f"/molecule/{chembl_id}.json")
        if not payload:
            return None
        return ChEMBLDrug(
            chembl_id=str(payload.get("molecule_chembl_id", chembl_id)),
            name=str(payload.get("pref_name")) if payload.get("pref_name") is not None else None,
            max_phase=self._safe_int(payload.get("max_phase")),
            molecule_type=(
                str(payload.get("molecule_type"))
                if payload.get("molecule_type") is not None
                else None
            ),
        )

    async def get_target_drugs(self, target_chembl_id: str, *, limit: int = 10) -> list[ChEMBLDrug]:
        """Fetch molecules associated with a ChEMBL target."""

        payload = await self._request_json(
            "GET",
            "/mechanism.json",
            params={"target_chembl_id": target_chembl_id, "limit": limit},
        )
        mechanisms = payload.get("mechanisms", [])
        if not isinstance(mechanisms, list):
            return []
        drugs: list[ChEMBLDrug] = []
        seen: set[str] = set()
        for mechanism in mechanisms:
            if not isinstance(mechanism, dict):
                continue
            molecule_chembl_id = str(mechanism.get("molecule_chembl_id", ""))
            if not molecule_chembl_id or molecule_chembl_id in seen:
                continue
            seen.add(molecule_chembl_id)
            drugs.append(
                ChEMBLDrug(
                    chembl_id=molecule_chembl_id,
                    name=(
                        str(mechanism.get("pref_name"))
                        if mechanism.get("pref_name") is not None
                        else None
                    ),
                    max_phase=self._safe_int(mechanism.get("max_phase")),
                )
            )
        return drugs

    async def get_mechanisms(self, chembl_id: str) -> list[ChEMBLMechanism]:
        """Fetch mechanisms of action for a ChEMBL molecule."""

        payload = await self._request_json(
            "GET",
            "/mechanism.json",
            params={"molecule_chembl_id": chembl_id},
        )
        mechanisms = payload.get("mechanisms", [])
        if not isinstance(mechanisms, list):
            return []
        return [
            ChEMBLMechanism(
                target_chembl_id=str(item.get("target_chembl_id", "")),
                mechanism=str(item.get("mechanism_of_action"))
                if item.get("mechanism_of_action") is not None
                else None,
                action_type=str(item.get("action_type"))
                if item.get("action_type") is not None
                else None,
            )
            for item in mechanisms
            if isinstance(item, dict)
        ]

    async def get_bioactivities(
        self,
        target_chembl_id: str,
        *,
        limit: int = 50,
    ) -> list[ChEMBLBioactivity]:
        """Fetch bioactivities associated with a target."""

        payload = await self._request_json(
            "GET",
            "/activity.json",
            params={"target_chembl_id": target_chembl_id, "limit": limit},
        )
        activities = payload.get("activities", [])
        if not isinstance(activities, list):
            return []
        records: list[ChEMBLBioactivity] = []
        for activity in activities:
            if not isinstance(activity, dict):
                continue
            value = activity.get("standard_value")
            records.append(
                ChEMBLBioactivity(
                    assay_chembl_id=str(activity.get("assay_chembl_id", "")),
                    molecule_chembl_id=str(activity.get("molecule_chembl_id", "")),
                    activity_type=(
                        str(activity.get("standard_type"))
                        if activity.get("standard_type") is not None
                        else None
                    ),
                    activity_value=float(value) if value is not None else None,
                    units=str(activity.get("standard_units"))
                    if activity.get("standard_units") is not None
                    else None,
                )
            )
        return records

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """Safely coerce a JSON scalar to an integer."""

        if value is None:
            return None
        return int(str(value))
