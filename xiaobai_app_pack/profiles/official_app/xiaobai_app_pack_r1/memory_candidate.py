from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

_PACK_PARENT = Path(__file__).resolve().parents[4]
if str(_PACK_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACK_PARENT))

from xiaobai_app_pack.sidecars.memory_candidate_client import MemoryCandidateClientError, create_candidate


logger = logging.getLogger(__name__)


class MemoryCandidate(Tool):
    """Create a parent-reviewable memory candidate."""

    name = "memory_candidate"
    description = (
        "Create a parent-review memory candidate. Use this only when the user explicitly asks Xiaobai to remember "
        "a stable family, learning, preference, or project fact. Do not use it for ordinary chat."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The short observable fact to store as a pending candidate.",
            },
            "subject": {
                "type": "string",
                "description": "Who or what the fact is about, such as taotao, eden, dad, xiaobai, or project.",
            },
            "source_turn_id": {
                "type": "string",
                "description": "Optional host/runtime turn id if available.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "sensitivity": {
                "type": "string",
                "enum": ["normal", "private", "sensitive"],
            },
            "requested_by": {
                "type": "string",
                "description": "Speaker label if known; otherwise use unknown.",
            },
        },
        "required": ["fact"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        fact = kwargs.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            return {"error": "fact must be a non-empty string"}

        payload = {
            "fact": fact.strip(),
            "subject": str(kwargs.get("subject") or "unknown").strip() or "unknown",
            "source_turn_id": str(kwargs.get("source_turn_id") or "").strip(),
            "confidence": str(kwargs.get("confidence") or "medium").strip().lower(),
            "sensitivity": str(kwargs.get("sensitivity") or "normal").strip().lower(),
            "requested_by": str(kwargs.get("requested_by") or "unknown").strip() or "unknown",
        }
        try:
            result = create_candidate(payload)
        except MemoryCandidateClientError as exc:
            logger.warning("memory_candidate sidecar unavailable: %s", exc)
            return {
                "error": "memory candidate sidecar is unavailable; no memory was saved",
                "sidecar_unavailable": True,
            }

        candidate = result.get("candidate", {})
        logger.info("Tool call: memory_candidate fact=%s candidate_id=%s", fact[:120], candidate.get("id"))
        return {
            "status": "candidate_created",
            "candidate_id": candidate.get("id"),
            "approval_required": True,
            "candidate": candidate,
        }
