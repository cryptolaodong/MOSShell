from __future__ import annotations

import asyncio
import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from xiaobai_app_pack.sidecars.memory_candidate_client import request_json
from xiaobai_app_pack.sidecars.memory_candidate_cli import main as memory_cli_main
from xiaobai_app_pack.sidecars.memory_candidate_sidecar import MemoryStore, make_handler


PACK_ROOT = Path(__file__).resolve().parents[1] / "xiaobai_app_pack"
PROFILE_ROOT = PACK_ROOT / "profiles" / "official_app" / "xiaobai_app_pack_r1"


def _load_profile_tool(filename: str, class_name: str):
    path = PROFILE_ROOT / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _serve_store(tmp_path: Path):
    store = MemoryStore(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_profile_does_not_enable_direct_remember() -> None:
    tools = (PROFILE_ROOT / "tools.txt").read_text(encoding="utf-8").splitlines()
    assert "memory_candidate" in tools
    assert "forget" in tools
    assert "remember" not in tools
    assert "camera" not in tools


def test_memory_candidate_sidecar_lifecycle(tmp_path: Path, monkeypatch) -> None:
    server, url = _serve_store(tmp_path)
    monkeypatch.setenv("XIAOBAI_MEMORY_SIDECAR_URL", url)
    try:
        created = request_json(
            "POST",
            "/memory/candidates",
            {
                "fact": "陶陶喜欢恐龙英语单词",
                "subject": "taotao",
                "confidence": "high",
                "sensitivity": "normal",
            },
        )
        candidate = created["candidate"]
        assert candidate["status"] == "candidate"
        assert candidate["approval_required"] is True

        approved = request_json("POST", f"/memory/candidates/{candidate['id']}/approve")
        assert approved["candidate"]["status"] == "approved"
        assert approved["memory"]["fact"] == "陶陶喜欢恐龙英语单词"

        forgotten = request_json("POST", "/memory/forget", {"query": "恐龙英语"})
        assert forgotten["removed"]["id"] == candidate["id"]
        assert request_json("GET", "/memory/approved")["memories"] == []

        second = request_json("POST", "/memory/candidates", {"fact": "Eden likes space stories"})
        rejected = request_json("POST", f"/memory/candidates/{second['candidate']['id']}/reject")
        assert rejected["candidate"]["status"] == "rejected"

        third = request_json("POST", "/memory/candidates", {"fact": "Dad prefers short reports"})
        deleted = request_json("DELETE", f"/memory/candidates/{third['candidate']['id']}")
        assert deleted["candidate"]["status"] == "deleted"
    finally:
        server.shutdown()


def test_profile_tools_use_sidecar(tmp_path: Path, monkeypatch) -> None:
    server, url = _serve_store(tmp_path)
    monkeypatch.setenv("XIAOBAI_MEMORY_SIDECAR_URL", url)
    try:
        MemoryCandidate = _load_profile_tool("memory_candidate.py", "MemoryCandidate")
        Forget = _load_profile_tool("forget.py", "Forget")

        created = asyncio.run(
            MemoryCandidate()(None, fact="爸爸喜欢简短验收报告", subject="dad", confidence="high")
        )
        assert created["status"] == "candidate_created"
        candidate_id = created["candidate_id"]

        request_json("POST", f"/memory/candidates/{candidate_id}/approve")
        removed = asyncio.run(Forget()(None, query="简短验收"))
        assert removed["memory_id"] == candidate_id
        assert "爸爸喜欢简短验收报告" in removed["removed"]["fact"]
    finally:
        server.shutdown()


def test_memory_review_cli_lifecycle_and_redaction(tmp_path: Path, monkeypatch, capsys) -> None:
    server, url = _serve_store(tmp_path)
    monkeypatch.setenv("XIAOBAI_MEMORY_SIDECAR_URL", url)
    try:
        created = request_json(
            "POST",
            "/memory/candidates",
            {
                "fact": "爸爸的临时 token 是 sk-testsecret12345678",
                "subject": "dad",
                "sensitivity": "private",
            },
        )
        candidate_id = created["candidate"]["id"]

        assert memory_cli_main(["candidates"]) == 0
        out = capsys.readouterr().out
        assert candidate_id in out
        assert "sk-testsecret" not in out
        assert "[redacted]" in out

        assert memory_cli_main(["approve", candidate_id]) == 0
        assert "approved" in capsys.readouterr().out

        assert memory_cli_main(["approved"]) == 0
        out = capsys.readouterr().out
        assert candidate_id in out
        assert "sk-testsecret" not in out

        assert memory_cli_main(["forget", "爸爸"]) == 0
        assert "forgot" in capsys.readouterr().out

        second = request_json("POST", "/memory/candidates", {"fact": "Eden likes space stories"})
        second_id = second["candidate"]["id"]
        assert memory_cli_main(["reject", second_id]) == 0
        assert "rejected" in capsys.readouterr().out

        third = request_json("POST", "/memory/candidates", {"fact": "Dad prefers short reports"})
        third_id = third["candidate"]["id"]
        assert memory_cli_main(["delete-candidate", third_id]) == 0
        assert "deleted candidate" in capsys.readouterr().out
    finally:
        server.shutdown()
