"""HUMN-Eval v0 scorer: automatic scoring for reference-backed items.

Usage:
    # 1. generate model answers: one JSONL with {"id": ..., "answer": ...}
    # 2. score:
    python worker/eval_score.py --eval eval/eval.jsonl --answers answers.jsonl

`dialect_id` items score by exact label match. All other capabilities are
counted as needing human/LLM grading (listed in the report).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--answers", required=True)
    args = ap.parse_args()

    items = {}
    for line in open(args.eval, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            items[d["id"]] = d
    answers = {}
    for line in open(args.answers, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            answers[d["id"]] = d.get("answer", "")

    auto_ok = auto_total = 0
    pending: Counter = Counter()
    for iid, it in items.items():
        if iid not in answers:
            pending[it["capability"] + ":unanswered"] += 1
            continue
        if it["capability"] == "dialect_id" and it.get("reference"):
            auto_total += 1
            if norm(str(answers[iid])) == norm(str(it["reference"])):
                auto_ok += 1
        else:
            pending[it["capability"] + ":needs_grading"] += 1

    print("=== HUMN-Eval v0 report ===")
    print(f"items: {len(items)}, answered: {len(answers)}")
    if auto_total:
        print(f"dialect_id accuracy: {auto_ok}/{auto_total} = {auto_ok / auto_total:.1%}")
    else:
        print("dialect_id: no reference-backed answers to score")
    if pending:
        print("pending human/LLM grading:")
        for k, v in pending.most_common():
            print(f"  {k}: {v}")
    Path("eval_report.json").write_text(json.dumps(
        {"n_items": len(items), "n_answered": len(answers),
         "dialect_id": {"ok": auto_ok, "total": auto_total},
         "pending": dict(pending)}, indent=2))
    print("wrote eval_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
