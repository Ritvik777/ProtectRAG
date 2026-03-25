#!/usr/bin/env python3
"""
Run ProtectRAG on PR / CI / agent-instruction style samples.

From the repo root (editable install):

    PYTHONPATH=src python examples/scan_pr_agent_samples.py

After ``pip install protectrag``:

    python examples/scan_pr_agent_samples.py

Synthetic hostile strings only — for integration testing, not real URLs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from protectrag.ingest import IngestDecision, ingest_document
from protectrag.scanner import InjectionSeverity, scan_document_for_injection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Emit protectrag ingest JSON logs (default: table only)",
    )
    args = parser.parse_args()
    if not args.verbose:
        logging.getLogger("protectrag").setLevel(logging.CRITICAL + 1)

    root = Path(__file__).resolve().parent
    path = root / "pr_agent_surface_samples.json"
    rows = json.loads(path.read_text(encoding="utf-8"))

    print(f"Loaded {len(rows)} samples from {path.name}\n")
    print(f"{'id':<22} {'expect':<10} {'severity':<8} {'ingest':<18} {'match'}")
    print("-" * 88)

    wrong = 0
    for row in rows:
        sid = row["id"]
        expect = row["expect"]
        text = row["text"]
        scan = scan_document_for_injection(text, document_id=sid)
        ing = ingest_document(
            text,
            document_id=sid,
            block_on=InjectionSeverity.HIGH,
            warn_on=InjectionSeverity.MEDIUM,
        )
        rules = ",".join(scan.matched_rules[:4])
        if len(scan.matched_rules) > 4:
            rules += ",..."
        print(
            f"{sid:<22} {expect:<10} {scan.severity.name:<8} {ing.decision.value:<18} {rules or '-'}"
        )
        alerted = scan.should_alert
        if expect == "injection" and not alerted:
            print(f"  !! expected injection to alert, got {scan.severity.name}")
            wrong += 1
        if expect == "clean" and alerted:
            print(f"  !! expected clean, got alert severity={scan.severity.name}")
            wrong += 1
        if expect == "injection" and ing.decision == IngestDecision.ALLOW:
            print("  !! expected non-ALLOW ingest decision for labeled injection")
            wrong += 1

    print("-" * 88)
    if wrong:
        print(f"Finished with {wrong} expectation mismatch(es).")
        return 1
    print("All sample expectations met (heuristic + default ingest policy).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print("Import error:", e, file=sys.stderr)
        print("Try: pip install -e .   or   PYTHONPATH=src python ...", file=sys.stderr)
        sys.exit(2)
