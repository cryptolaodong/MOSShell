from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "http://127.0.0.1:8788"


class MemoryCandidateClientError(RuntimeError):
    """Raised when the memory candidate sidecar cannot satisfy a request."""


def sidecar_url() -> str:
    return os.getenv("XIAOBAI_MEMORY_SIDECAR_URL", DEFAULT_URL).rstrip("/")


def _timeout() -> float:
    raw = os.getenv("XIAOBAI_MEMORY_SIDECAR_TIMEOUT", "2.0")
    try:
        return max(0.2, float(raw))
    except ValueError:
        return 2.0


def request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        f"{sidecar_url()}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as response:
            body = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise MemoryCandidateClientError(str(exc)) from exc

    try:
        parsed = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise MemoryCandidateClientError(f"invalid JSON response: {body[:200]}") from exc
    if not isinstance(parsed, dict):
        raise MemoryCandidateClientError("sidecar response must be a JSON object")
    return parsed


def create_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    return request_json("POST", "/memory/candidates", payload)


def list_candidates() -> list[dict[str, Any]]:
    data = request_json("GET", "/memory/candidates")
    candidates = data.get("candidates", [])
    return candidates if isinstance(candidates, list) else []


def approve_candidate(candidate_id: str) -> dict[str, Any]:
    return request_json("POST", f"/memory/candidates/{candidate_id}/approve")


def reject_candidate(candidate_id: str) -> dict[str, Any]:
    return request_json("POST", f"/memory/candidates/{candidate_id}/reject")


def delete_candidate(candidate_id: str) -> dict[str, Any]:
    return request_json("DELETE", f"/memory/candidates/{candidate_id}")


def list_approved() -> list[dict[str, Any]]:
    data = request_json("GET", "/memory/approved")
    memories = data.get("memories", [])
    return memories if isinstance(memories, list) else []


def forget_approved(query: str) -> dict[str, Any]:
    return request_json("POST", "/memory/forget", {"query": query})


def delete_memory(memory_id: str) -> dict[str, Any]:
    return request_json("DELETE", f"/memory/{memory_id}")
