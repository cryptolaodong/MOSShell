from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Sequence

from xiaobai_app_pack.sidecars.memory_candidate_client import (
    MemoryCandidateClientError,
    approve_candidate,
    delete_candidate,
    delete_memory,
    forget_approved,
    list_approved,
    list_candidates,
    reject_candidate,
)


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_\- ]?key|token|secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
)


def redact(value: Any) -> str:
    text = "" if value is None else str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def format_item(item: dict[str, Any]) -> str:
    item_id = redact(item.get("id", ""))
    status = redact(item.get("status", "approved"))
    subject = redact(item.get("subject", "unknown"))
    sensitivity = redact(item.get("sensitivity", "normal"))
    fact = redact(item.get("fact", ""))
    return f"{item_id} | {status} | {subject} | {sensitivity} | {fact}"


def print_items(title: str, items: list[dict[str, Any]]) -> None:
    print(title)
    if not items:
        print("(empty)")
        return
    for item in items:
        print(format_item(item))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("xiaobai-memory-review")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("candidates", help="List memory candidates")
    sub.add_parser("approved", help="List approved Xiaobai memories")

    approve = sub.add_parser("approve", help="Approve a memory candidate")
    approve.add_argument("candidate_id")

    reject = sub.add_parser("reject", help="Reject a memory candidate")
    reject.add_argument("candidate_id")

    delete = sub.add_parser("delete-candidate", help="Delete a candidate and its approved memory if any")
    delete.add_argument("candidate_id")

    forget = sub.add_parser("forget", help="Forget one approved memory by matching phrase")
    forget.add_argument("query")

    delete_memory_parser = sub.add_parser("delete-memory", help="Delete one approved memory by id")
    delete_memory_parser.add_argument("memory_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "candidates":
            print_items("Memory candidates:", list_candidates())
            return 0
        if args.command == "approved":
            print_items("Approved memories:", list_approved())
            return 0
        if args.command == "approve":
            result = approve_candidate(args.candidate_id)
            print("approved")
            print(format_item(result.get("memory", {})))
            return 0
        if args.command == "reject":
            result = reject_candidate(args.candidate_id)
            print("rejected")
            print(format_item(result.get("candidate", {})))
            return 0
        if args.command == "delete-candidate":
            result = delete_candidate(args.candidate_id)
            print("deleted candidate")
            print(format_item(result.get("candidate", {})))
            return 0
        if args.command == "forget":
            result = forget_approved(args.query)
            removed = result.get("removed")
            if not isinstance(removed, dict) or not removed:
                print("no matching approved memory")
                return 1
            print("forgot")
            print(format_item(removed))
            return 0
        if args.command == "delete-memory":
            result = delete_memory(args.memory_id)
            print("deleted memory")
            print(format_item(result.get("removed", {})))
            return 0
    except MemoryCandidateClientError as exc:
        print(f"memory sidecar unavailable: {redact(exc)}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
