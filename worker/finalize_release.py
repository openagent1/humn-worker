"""Assemble the final release from sharded merge legs (timeout-proof publishing).

Each merge leg uploads its split dirs under <LEG>/ in a work repo.
Finalize downloads them, concatenates per split dir (validating JSON),
merges the leg stats, renders the dataset card, and uploads everything
to the final repo. No single job runs long: legs are minutes each,
finalize is pure streaming copy.

Usage (CI):
    python worker/finalize_release.py --work-repo auto-work --legs chat,social,web \\
        --out-repo auto --dataset-version v1 --token $HF_TOKEN_RW

Offline test:
    python worker/finalize_release.py --parts-dir <dir-with-leg-subdirs> \\
        --out-repo test/repo --dataset-version v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from merge_dataset import ATTRIBUTIONS, CARD, ShardWriter
from worker_core import hf_upload_file, hf_whoami


def resolve(name: str, token: str | None, kind: str, version: str) -> str:
    if name == "auto":
        user = hf_whoami(token)
        return f"{user}/humn-social-register-v{version.lstrip('v')}"
    if name == "auto-work":
        user = hf_whoami(token)
        return f"{user}/humn-v1-work"
    return name


def download_leg(work_repo: str, leg: str, dest: Path, token: str) -> list[Path]:
    """Download every file under <leg>/ in the work repo. Returns local paths."""
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=work_repo, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print(f"[finalize] leg {leg}: cannot list {work_repo} ({e})", flush=True)
        return []
    got = []
    for f in sorted(files):
        if not f.startswith(leg + "/"):
            continue
        local = dest / leg / f[len(leg) + 1:]
        local.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=work_repo, filename=f, repo_type="dataset",
                        token=token, local_dir=dest / leg,
                        local_dir_use_symlinks=False)
        # hf_hub_download mirrors repo structure; move into place if nested
        src = dest / leg / f
        if src != local and src.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            src.replace(local)
        if local.exists():
            got.append(local)
            print(f"[finalize] {leg}: fetched {f}", flush=True)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-repo", default="",
                    help="work repo holding leg outputs (not needed with --parts-dir)")
    ap.add_argument("--legs", default="", help="comma-separated leg names")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--dataset-version", default="v1")
    ap.add_argument("--token", default=None)
    ap.add_argument("--workdir", default="finalize-work")
    ap.add_argument("--parts-dir", default="",
                    help="offline test: local dir containing <leg>/ subdirs, skip download")
    args = ap.parse_args()

    token = args.token or __import__("os").environ.get("HF_TOKEN_RW")
    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    if not legs:
        print("[error] --legs required (e.g. chat,social,web)", file=sys.stderr)
        return 1
    out_repo = resolve(args.out_repo, token, "out", args.dataset_version)
    print(f"[finalize] {legs} -> {out_repo}", flush=True)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    parts_root = workdir / "parts"

    if args.parts_dir:
        import shutil

        src = Path(args.parts_dir)
        if parts_root.exists():
            shutil.rmtree(parts_root)
        shutil.copytree(src, parts_root)
        print(f"[finalize] using local parts dir {src}", flush=True)
    else:
        if not token:
            print("[error] need a token to download legs", file=sys.stderr)
            return 1
        work_repo = resolve(args.work_repo, token, "work", args.dataset_version)
        for leg in legs:
            n = len(download_leg(work_repo, leg, parts_root, token))
            print(f"[finalize] leg {leg}: {n} files", flush=True)

    # concat per split dir (streaming, JSON-validated), merge stats
    final_dir = workdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, ShardWriter] = {}
    split_counts: Counter = Counter()
    lang_counter: Counter = Counter()
    flag_counter: Counter = Counter()
    used_sources: set[str] = set()
    n_dup = n_norm_dup = 0
    total_in = 0

    def writer_for(split: str) -> ShardWriter:
        if split not in writers:
            writers[split] = ShardWriter(final_dir / split)
        return writers[split]

    for leg in legs:
        leg_dir = parts_root / leg
        if not leg_dir.exists():
            print(f"[finalize] leg {leg}: MISSING locally, skipped", flush=True)
            continue
        sp = leg_dir / "stats.json"
        if sp.exists():
            st = json.loads(sp.read_text(encoding="utf-8"))
            for k, v in (st.get("split_counts") or {}).items():
                split_counts[k] += v
            for k, v in (st.get("lang_counter") or {}).items():
                lang_counter[k] += v
            for k, v in (st.get("flag_counter") or {}).items():
                flag_counter[k] += v
            n_dup += st.get("n_dup", 0)
            n_norm_dup += st.get("n_norm_dup", 0)
            used_sources.update(st.get("used_sources", []))
        for fp in sorted(leg_dir.rglob("*.jsonl")):
            rel = fp.relative_to(leg_dir)
            if len(rel.parts) < 2:
                continue  # only split-dir files, not strays
            split = rel.parts[0]
            w = writer_for(split)
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    w.write(json.loads(line))  # validates JSON
                    total_in += 1
        print(f"[finalize] leg {leg} concatenated", flush=True)

    for w in writers.values():
        w.close()
    total_rows = sum(split_counts.values()) or total_in
    print(f"[finalize] {total_rows:,} rows across {len(writers)} split dirs", flush=True)

    # card
    sources_txt = "\n".join(
        f"- {repo} — {lic} — {url}"
        for repo, (lic, url) in ATTRIBUTIONS.items() if repo in used_sources)
    unknown_src = sorted(used_sources - set(ATTRIBUTIONS))
    if unknown_src:
        sources_txt += "\n" + "\n".join(f"- {repo} — license: see source page" for repo in unknown_src)
    size_cat = "<1K" if total_rows < 1_000 else "1K<n<10K" if total_rows < 10_000 else \
               "10K<n<100K" if total_rows < 100_000 else "100K<n<1M" if total_rows < 1_000_000 else "1M<n<10M"
    langmix_txt = "\n".join(f"- {k}: {v:,}" for k, v in lang_counter.most_common(20)) or "- UNKNOWN"
    quality_txt = (
        f"- canonical rows: {total_rows:,}\n"
        f"- exact duplicates removed (per-leg): {n_dup:,}\n"
        f"- normalized near-duplicates removed (per-leg): {n_norm_dup:,}\n"
        + ("\n".join(f"- flagged {k}: {v:,}" for k, v in flag_counter.most_common())
           if flag_counter else "- flags: none")
        + "\n- assembly: sharded merge legs (chat/social/web), concatenated per split; "
          "small cross-leg residual duplication possible by design (overlapping "
          "sources were grouped in the same leg to minimize it)"
    )
    composition = [f"- {s}/train-*.jsonl: {split_counts[s]:,} rows" for s in sorted(split_counts)]
    composition.append(f"- assembled from legs: {', '.join(legs)} (see build-stats.json)")
    card = CARD.format(version="v" + args.dataset_version.lstrip("v"),
                       size_cat=size_cat, composition="\n".join(composition),
                       sources=sources_txt, langmix=langmix_txt, quality=quality_txt)
    card_p = final_dir / "README.md"
    card_p.write_text(card, encoding="utf-8")
    stats_out = {"split_counts": dict(split_counts), "lang_counter": dict(lang_counter),
                 "flag_counter": dict(flag_counter), "n_dup": n_dup, "n_norm_dup": n_norm_dup,
                 "total_rows": total_rows, "used_sources": sorted(used_sources),
                 "legs": legs}
    (final_dir / "build-stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.parts_dir and not token:
        print(f"[finalize] dev mode: files ready in {final_dir} (no upload without token)")
        return 0
    if not token:
        print("[error] no HF token for upload", file=sys.stderr)
        return 1
    for rel in sorted(final_dir.rglob("*")):
        if not rel.is_file() or rel.stat().st_size == 0:
            continue
        hf_upload_file(out_repo, rel, rel.relative_to(final_dir).as_posix(), token)
    print(f"\n[UPLOADED] https://huggingface.co/datasets/{out_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
