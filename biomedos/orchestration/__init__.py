"""Workflow orchestration primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from biomedos.orchestration.state import AgentResult, Task, TaskType, WorkflowState

if TYPE_CHECKING:
    from biomedos.orchestration.decomposer import QueryDecomposer
    from biomedos.orchestration.workflow import BiomedicalWorkflow

__all__ = [
    "AgentResult",
    "BiomedicalWorkflow",
    "QueryDecomposer",
    "Task",
    "TaskType",
    "WorkflowState",
]


def __getattr__(name: str) -> Any:
    """Lazily expose optional orchestration helpers without import cycles."""

    if name == "BiomedicalWorkflow":
        from biomedos.orchestration.workflow import BiomedicalWorkflow

        return BiomedicalWorkflow
    if name == "QueryDecomposer":
        from biomedos.orchestration.decomposer import QueryDecomposer

        return QueryDecomposer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
