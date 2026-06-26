from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

_PACK_PARENT = Path(__file__).resolve().parents[4]
if str(_PACK_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACK_PARENT))

from xiaobai_app_pack.sidecars.memory_candidate_client import MemoryCandidateClientError, forget_approved


logger = logging.getLogger(__name__)


class Forget(Tool):
    """Forget one approved Xiaobai sidecar memory."""

    name = "forget"
    description = (
        "Remove an approved Xiaobai memory by a short matching phrase. Use this when the user asks Xiaobai to "
        "forget something that may have been approved from the memory candidate sidecar."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A short phrase contained in the approved memory fact to remove.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "query must be a non-empty string"}
        try:
            result = forget_approved(query.strip())
        except MemoryCandidateClientError as exc:
            logger.warning("forget sidecar unavailable: %s", exc)
            return {
                "error": "memory candidate sidecar is unavailable; no sidecar memory was removed",
                "sidecar_unavailable": True,
            }

        removed = result.get("removed")
        if not removed:
            return {"error": f'no approved Xiaobai memory matched "{query}"'}
        return {"removed": removed, "memory_id": removed.get("id")}
