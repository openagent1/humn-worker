"""Worker entrypoint: python worker/run_worker.py --spec <job.yaml> [options].

Env:
  HF_TOKEN_RO / HF_TOKEN_RW  tokens (or --dev for no-token local run)
  WORKER_RUN                 run label for manifest bookkeeping
  SOFT_DEADLINE_MIN          minutes until clean checkpoint+exit (default 45)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml

from worker_core import run_job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path or URL to frozen job spec YAML")
    ap.add_argument("--workdir", default="/tmp/humn-work" if os.name != "nt" else os.environ.get("TEMP", "./work"))
    ap.add_argument("--dev", action="store_true", help="no uploads, small chunk cap, no tokens")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--no-artifacts", action="store_true", help="process but skip uploads")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if spec_path.exists():
        with open(spec_path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
    elif args.spec.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(args.spec, timeout=60) as r:
            spec = yaml.safe_load(r.read().decode())
    else:
        print(f"[error] spec not found: {args.spec}", file=sys.stderr)
        return 1

    workdir = Path(args.workdir) / spec["job_id"]
    workdir.mkdir(parents=True, exist_ok=True)

    token_ro = None if args.dev else (os.environ.get("HF_TOKEN_RO") or None)
    token_rw = None if (args.dev or args.no_artifacts) else (os.environ.get("HF_TOKEN_RW") or None)

    deadline_min = int(os.environ.get("SOFT_DEADLINE_MIN", "45"))
    soft_deadline = time.time() + deadline_min * 60 if not args.dev else None

    manifest = run_job(
        spec, workdir,
        progress=lambda m: print(f"[{spec['job_id']}] {m}", flush=True),
        token_ro=token_ro, token_rw=token_rw,
        max_chunks=args.max_chunks,
        soft_deadline_at=soft_deadline,
        write_artifacts=not (args.dev or args.no_artifacts),
    )
    stats = manifest["stats"]
    print(f"[done] {stats['chunks_done']}/{stats['chunks_total']} chunks, "
          f"{stats.get('rows_done', 0):,} rows")
    incomplete = stats["chunks_done"] < stats["chunks_total"]
    if incomplete:
        print("[INCOMPLETE] green must mean DONE. Re-run the workflow — "
              "it resumes automatically from the manifest.", file=sys.stderr)
        return 2
    if stats["chunks_failed"]:
        print(f"[warn] {stats['chunks_failed']} chunks failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
