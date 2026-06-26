from __future__ import annotations

import argparse
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788


def default_data_dir() -> Path:
    return Path(os.getenv("XIAOBAI_MEMORY_DATA_DIR", "~/.local/share/xiaobai_app_pack/memory")).expanduser()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state() -> dict[str, list[dict[str, Any]]]:
    return {"candidates": [], "approved_memories": [], "events": []}


class MemoryStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "memory_candidates.v1.json"
        self._lock = threading.Lock()

    def _load_unlocked(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return _empty_state()
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return _empty_state()
        state = _empty_state()
        for key in state:
            value = data.get(key, [])
            state[key] = value if isinstance(value, list) else []
        return state

    def _save_unlocked(self, state: dict[str, list[dict[str, Any]]]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        tmp.replace(self.path)

    def create_candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        fact = str(payload.get("fact", "")).strip()
        if not fact:
            raise ValueError("fact is required")

        candidate = {
            "id": uuid.uuid4().hex,
            "subject": str(payload.get("subject") or "unknown").strip() or "unknown",
            "fact": fact,
            "source_turn_id": str(payload.get("source_turn_id") or "").strip(),
            "confidence": _enum_value(payload.get("confidence"), {"low", "medium", "high"}, "medium"),
            "sensitivity": _enum_value(payload.get("sensitivity"), {"normal", "private", "sensitive"}, "normal"),
            "requested_by": str(payload.get("requested_by") or "unknown").strip() or "unknown",
            "created_at": now_iso(),
            "approval_required": True,
            "status": "candidate",
        }
        with self._lock:
            state = self._load_unlocked()
            state["candidates"].append(candidate)
            state["events"].append({"type": "candidate_created", "id": candidate["id"], "at": now_iso()})
            self._save_unlocked(state)
        return candidate

    def list_candidates(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._load_unlocked()["candidates"])

    def list_approved(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._load_unlocked()["approved_memories"])

    def approve_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_unlocked()
            candidate = _find_by_id(state["candidates"], candidate_id)
            candidate["status"] = "approved"
            candidate["approved_at"] = now_iso()
            memory = {
                "id": candidate["id"],
                "subject": candidate["subject"],
                "fact": candidate["fact"],
                "source_candidate_id": candidate["id"],
                "approved_at": candidate["approved_at"],
                "sensitivity": candidate["sensitivity"],
            }
            state["approved_memories"] = [m for m in state["approved_memories"] if m.get("id") != memory["id"]]
            state["approved_memories"].append(memory)
            state["events"].append({"type": "candidate_approved", "id": candidate_id, "at": now_iso()})
            self._save_unlocked(state)
        return {"candidate": candidate, "memory": memory}

    def reject_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_unlocked()
            candidate = _find_by_id(state["candidates"], candidate_id)
            candidate["status"] = "rejected"
            candidate["rejected_at"] = now_iso()
            state["events"].append({"type": "candidate_rejected", "id": candidate_id, "at": now_iso()})
            self._save_unlocked(state)
        return candidate

    def delete_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_unlocked()
            candidate = _find_by_id(state["candidates"], candidate_id)
            candidate["status"] = "deleted"
            candidate["deleted_at"] = now_iso()
            state["approved_memories"] = [m for m in state["approved_memories"] if m.get("id") != candidate_id]
            state["events"].append({"type": "candidate_deleted", "id": candidate_id, "at": now_iso()})
            self._save_unlocked(state)
        return candidate

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_unlocked()
            memory = _find_by_id(state["approved_memories"], memory_id)
            state["approved_memories"] = [m for m in state["approved_memories"] if m.get("id") != memory_id]
            state["events"].append({"type": "memory_deleted", "id": memory_id, "at": now_iso()})
            self._save_unlocked(state)
        return memory

    def forget_by_query(self, query: str) -> dict[str, Any]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query is required")
        pattern = re.compile(re.escape(cleaned), re.IGNORECASE)
        with self._lock:
            state = self._load_unlocked()
            matches = [m for m in state["approved_memories"] if pattern.search(str(m.get("fact", "")))]
            if not matches:
                return {"removed": None, "matches": []}
            removed = matches[0]
            state["approved_memories"] = [m for m in state["approved_memories"] if m.get("id") != removed.get("id")]
            state["events"].append({"type": "memory_forgotten", "id": removed.get("id"), "query": cleaned, "at": now_iso()})
            self._save_unlocked(state)
        return {"removed": removed, "matches": matches}


def _enum_value(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or default).strip().lower()
    return text if text in allowed else default


def _find_by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise KeyError(item_id)


def make_handler(store: MemoryStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "XiaobaiMemoryCandidate/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body or "{}")
            if not isinstance(data, dict):
                raise ValueError("payload must be a JSON object")
            return data

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._json(200, {"ok": True})
                return
            if path == "/memory/candidates":
                self._json(200, {"candidates": store.list_candidates()})
                return
            if path == "/memory/approved":
                self._json(200, {"memories": store.list_approved()})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/memory/candidates":
                    self._json(201, {"candidate": store.create_candidate(self._payload())})
                    return
                match = re.fullmatch(r"/memory/candidates/([^/]+)/(approve|reject)", path)
                if match:
                    candidate_id, action = match.groups()
                    if action == "approve":
                        self._json(200, store.approve_candidate(candidate_id))
                    else:
                        self._json(200, {"candidate": store.reject_candidate(candidate_id)})
                    return
                if path == "/memory/forget":
                    self._json(200, store.forget_by_query(str(self._payload().get("query", ""))))
                    return
            except KeyError as exc:
                self._json(404, {"error": "not_found", "id": str(exc)})
                return
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "not_found"})

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path
            try:
                match = re.fullmatch(r"/memory/candidates/([^/]+)", path)
                if match:
                    self._json(200, {"candidate": store.delete_candidate(match.group(1))})
                    return
                match = re.fullmatch(r"/memory/([^/]+)", path)
                if match:
                    self._json(200, {"removed": store.delete_memory(match.group(1))})
                    return
            except KeyError as exc:
                self._json(404, {"error": "not_found", "id": str(exc)})
                return
            self._json(404, {"error": "not_found"})

    return Handler


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, data_dir: Path | None = None) -> None:
    store = MemoryStore(data_dir or default_data_dir())
    server = ThreadingHTTPServer((host, port), make_handler(store))
    print(f"Xiaobai memory candidate sidecar listening on http://{host}:{port}")
    print(f"Data file: {store.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Xiaobai memory candidate sidecar stopping")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser("Xiaobai memory candidate sidecar")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.host, args.port, args.data_dir)


if __name__ == "__main__":
    main()
