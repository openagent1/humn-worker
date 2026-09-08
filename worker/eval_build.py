"""HUMN-Eval v0 builder: held-out evaluation slices from merged outputs.

Reads the FINAL merged dirs (pretrain_*/sft_chat_*) and samples a small,
fixed evaluation set per capability. Writes eval/eval.jsonl + eval/README.

Capabilities (keyword/behavior probes, model-graded or human-graded later):
  - dialect_id:      given a text, name the variety (ar_eg/ar_msa/...)
  - sarcasm:         is this sarcastic? (from ArSarcasm-labeled rows)
  - arabizi_read:    read Arabizi, respond in Arabic
  - code_switch:     handle mixed AR/EN input
  - emoji_slang:     interpret emoji/slang-heavy text
  - mult_turn:       continue a 2-turn conversation coherently

Usage:
    python worker/eval_build.py --final-dir <workdir/final> --out eval/ --n-per-cap 50 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_rows(final_dir: Path):
    rows = []
    for fp in sorted(final_dir.rglob("*.jsonl")):
        config = fp.parent.name
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d["_config"] = config
                rows.append(d)
    return rows


def build_eval(rows: list[dict], n_per_cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    eval_items: list[dict] = []

    def sample(pool, k):
        pool = list(pool)
        rng.shuffle(pool)
        return pool[:k]

    # 1. dialect_id: texts with confident lang labels
    dialect_pool = [r for r in rows if r.get("text") and r.get("lang") in
                    ("ar_eg", "ar_msa", "ar_levant", "ar_gulf", "ar_maghrebi", "en")]
    for r in sample(dialect_pool, n_per_cap):
        eval_items.append({
            "id": f"dialect-{r['id']}", "capability": "dialect_id",
            "prompt": f"What Arabic variety is this text written in? Text: {r['text'][:500]}",
            "reference": r["lang"], "source": r.get("source")})

    # 2. sarcasm: ArSarcasm-sourced rows (labels live upstream; probe generically)
    sarc_pool = [r for r in rows if r.get("source") == "iabufarha/ar_sarcasm" and r.get("text")]
    for r in sample(sarc_pool, n_per_cap):
        eval_items.append({
            "id": f"sarcasm-{r['id']}", "capability": "sarcasm",
            "prompt": f"Is the following Arabic text sarcastic? Answer yes or no, then explain briefly: {r['text'][:500]}",
            "reference": None, "source": r.get("source")})

    # 3. arabizi_read: latin-script rows from arabizi/mixed splits
    arabizi_pool = [r for r in rows if r.get("lang") in ("arabizi", "mixed") and r.get("text")]
    for r in sample(arabizi_pool, n_per_cap):
        eval_items.append({
            "id": f"arabizi-{r['id']}", "capability": "arabizi_read",
            "prompt": f"Read this Arabizi text and respond to it in Arabic: {r['text'][:500]}",
            "reference": None, "source": r.get("source")})

    # 4. emoji_slang: rows heavy with emoji / elongation
    def is_emoji_heavy(r):
        t = r.get("text", "")
        return sum(1 for c in t if 0x1F300 <= ord(c) or 0x2600 <= ord(c) <= 0x27BF) >= 2
    emoji_pool = [r for r in rows if r.get("text") and is_emoji_heavy(r)]
    for r in sample(emoji_pool, n_per_cap):
        eval_items.append({
            "id": f"emoji-{r['id']}", "capability": "emoji_slang",
            "prompt": f"What does this message mean? Explain like a friend: {r['text'][:500]}",
            "reference": None, "source": r.get("source")})

    # 5. multi_turn: conversations with 2+ turns -> predict continuation
    chat_rows = [r for r in rows if r.get("messages") and len(r["messages"]) >= 3]
    for r in sample(chat_rows, n_per_cap):
        msgs = r["messages"][:-1]
        ref = r["messages"][-1]
        eval_items.append({
            "id": f"multiturn-{r['id']}", "capability": "multi_turn",
            "prompt": "Continue this conversation naturally with the next message: " + json.dumps(
                msgs, ensure_ascii=False),
            "reference": ref, "source": r.get("source")})

    return eval_items


EVAL_README = """# HUMN-Eval v0

Held-out behavioral probes for human-like Arabic/internet-register models.
Built by `worker/eval_build.py` from the v1 merged corpus (items are
sampled, never trained on — keep this set OUT of training mixes).

## Capabilities
- `dialect_id` — name the Arabic variety (has reference labels)
- `sarcasm` — detect sarcasm (human-graded)
- `arabizi_read` — understand Arabizi (human-graded)
- `emoji_slang` — interpret emoji/slang (human-graded)
- `multi_turn` — continue a conversation (reference last message)

## Scoring
`worker/eval_score.py` scores `dialect_id` automatically (exact match on the
reference label) and packages the rest for human/LLM grading.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_rows(Path(args.final_dir))
    print(f"[eval] loaded {len(rows):,} merged rows")
    items = build_eval(rows, args.n_per_cap, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(EVAL_README, encoding="utf-8")
    from collections import Counter
    print(f"[eval] wrote {len(items)} items -> {out / 'eval.jsonl'}")
    print("[eval] by capability:", dict(Counter(i["capability"] for i in items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
