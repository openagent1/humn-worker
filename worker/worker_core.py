"""Worker core: executes a JobSpec chunk-by-chunk with checkpoint/resume.

Runs identically in GH Actions, Colab, Kaggle, or local dev mode. Depends on
optional libs (pyarrow/httpx/huggingface_hub) which the environment installs;
dev-mode falls back to plain HTTP when pyarrow is unavailable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

# ---- manifest (schemas/checkpoint_manifest.schema.yaml) ----


def new_manifest(job_id: str, code_version: str, chunk_ids: list[str]) -> dict:
    return {
        "job_id": job_id,
        "code_version": code_version,
        "updated_at": _now(),
        "worker_run": os.environ.get("WORKER_RUN", "dev"),
        "stats": {"chunks_total": len(chunk_ids), "chunks_done": 0, "chunks_pending": len(chunk_ids),
                  "chunks_failed": 0, "rows_done": 0},
        "chunks": [{"chunk_id": c, "status": "pending", "attempts": 0} for c in chunk_ids],
    }


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    """Atomic local write (tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def recalc_stats(manifest: dict) -> dict:
    chunks = manifest["chunks"]
    done = [c for c in chunks if c["status"] == "done"]
    manifest["stats"] = {
        "chunks_total": len(chunks),
        "chunks_done": len(done),
        "chunks_pending": sum(1 for c in chunks if c["status"] == "pending"),
        "chunks_failed": sum(1 for c in chunks if c["status"] == "failed"),
        "rows_done": sum(c.get("rows", 0) for c in done),
    }
    return manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---- HF HTTP API helpers (no SDK dependency; token optional for public repos) ----


def _clean_token(token: str | None) -> str | None:
    """Empty string tokens (unset GH secrets) are treated as anonymous."""
    if token is None or str(token).strip() == "":
        return None
    return token.strip()


_WHOAMI_CACHE: dict = {}


def hf_whoami(token: str) -> str:
    """Resolve the username that owns this token (cached)."""
    token = _clean_token(token)
    if not token:
        raise RuntimeError("no HF token provided (HF_TOKEN_RO / HF_TOKEN_RW)")
    if "user" in _WHOAMI_CACHE:
        return _WHOAMI_CACHE["user"]
    import urllib.request

    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "humn-worker/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    name = data.get("name")
    if not name:
        raise RuntimeError(f"whoami failed (token starts {token[:5]}...): {str(data)[:200]}")
    _WHOAMI_CACHE["user"] = name
    return name


def resolve_repo(repo: str, token: str | None) -> str:
    """Rewrite the placeholder namespace 'humn-internal/*' to the token owner's
    username: humn-internal/social-register-ckpt -> <you>/humn-social-register-ckpt.
    Any other repo name passes through unchanged."""
    if repo.startswith("humn-internal/"):
        user = hf_whoami(token)
        return f"{user}/humn-{repo[len('humn-internal/'):]}"
    return repo


def hf_list_parquet_files(repo: str, config: str, split: str = "train", token: str | None = None) -> list[str]:
    """List parquet shard filenames via the datasets-server /parquet endpoint.

    The endpoint returns all splits; filter by the URL's split path segment.
    """
    import urllib.request
    import urllib.parse

    url = (f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(repo, safe='')}"
           f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}")
    req = urllib.request.Request(url, headers={"User-Agent": "humn-worker/0.1"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    if "error" in data:
        raise RuntimeError(f"datasets-server error: {data['error']}")
    files = [f["url"] for f in data.get("parquet_files", [])]
    if split:
        files = [u for u in files if f"/{split}/" in u]
    return files


def hf_download(url: str, dest: Path, token: str | None = None) -> Path:
    """Stream download with progress; returns local path."""
    import urllib.request

    token = _clean_token(token)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "humn-worker/0.1"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"download failed {e.code} for {url} "
            f"(token: {'anon' if not token else token[:5] + '...'}) {body}"
        ) from e
    return dest


def hf_upload_file(repo: str, local_path: Path, path_in_repo: str, token: str,
                    repo_type: str = "dataset", create: bool = True) -> None:
    """Upload one file to an HF repo. Uses huggingface_hub (required dep).

    Raises RuntimeError with the actual HTTP error so CI logs show the cause.
    """
    token = _clean_token(token)
    if not token:
        raise RuntimeError(
            "no HF write token provided: set HF_TOKEN_RW secret (Settings > Secrets > Actions)")
    repo = resolve_repo(repo, token)  # humn-internal/* -> <your-username>/humn-*
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=token)
        if create:
            api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                        repo_id=repo, repo_type=repo_type)
        return
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed: pip install -r worker/requirements.txt") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"HF upload failed for {repo}/{path_in_repo} "
            f"(token starts: {token[:5]}..., error: {type(e).__name__}: {str(e)[:300]})"
        ) from e


# ---- chunk iteration ----


def iter_rows_parquet(file_path: Path, columns: list[str] | None = None):
    """Yield row dicts from a parquet file. Requires pyarrow.

    Handles one level of nesting: a schema whose columns are lists of structs
    (e.g. {version, description, entries: [...]}) is flattened so each list
    element is yielded as a row; scalar metadata columns are attached.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(file_path))
    top_names = pf.schema_arrow.names
    # detect list columns (e.g. "list<item: struct(...)>")
    nested_list_cols = [
        n for n in top_names
        if str(pf.schema_arrow.field(n).type).startswith("list<")
    ]
    if nested_list_cols:
        list_col = nested_list_cols[0]  # primary payload column
        meta_cols = [n for n in top_names if n != list_col and n not in (columns or [])]
        for batch in pf.iter_batches(batch_size=100):
            for rec in batch.to_pylist():
                entries = rec.get(list_col) or []
                for entry in entries:
                    if isinstance(entry, dict):
                        out = dict(entry)
                    else:
                        out = {"value": entry}
                    out[f"_{list_col}_len"] = len(entries)
                    out.update({m: rec.get(m) for m in meta_cols})
                    yield out
        return

    cols = None
    if columns:
        available = set(top_names)
        cols = [c for c in columns if c in available] or None
    for batch in pf.iter_batches(batch_size=1000, columns=cols):
        yield from batch.to_pylist()


def chunk_ids_from_files(files: list[str], chunk_rows: int) -> list[str]:
    """One chunk per parquet shard by default (file_shard strategy)."""
    return [f"{i:04d}" for i in range(len(files))]


# ---- runner ----

ProgressFn = Callable[[str], None]


def run_job(spec: dict, workdir: Path, progress: ProgressFn = print,
            token_ro: str | None = None, token_rw: str | None = None,
            max_chunks: int | None = None, soft_deadline_at: float | None = None,
            write_artifacts: bool = True, fetch_files_fn=None) -> dict:
    """Execute a job spec. Returns the final manifest.

    write_artifacts=False -> dev mode: everything stays in workdir (no upload).
    soft_deadline_at: unix time at which to stop and checkpoint cleanly.
    """
    job_id = spec["job_id"]
    task = spec["task"]
    ds = spec["data_source"]
    checkpoint_repo = spec["checkpoint"]["repo"]
    manifest_path = workdir / "manifest.json"

    files = (fetch_files_fn or hf_list_parquet_files)(ds["repo"], ds.get("config", ""), ds.get("split", "train"), token_ro)
    if not files:
        raise RuntimeError(f"no parquet files found for {ds['repo']}/{ds.get('config')}")
    chunk_ids = chunk_ids_from_files(files, spec["chunking"].get("chunk_rows", 50_000))
    progress(f"found {len(files)} parquet shards -> {len(chunk_ids)} chunks")

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        manifest["worker_run"] = os.environ.get("WORKER_RUN", "dev")
        # reset stale in-progress chunks
        for c in manifest["chunks"]:
            if c["status"] == "in_progress":
                c["status"] = "pending"
        progress(f"resuming: manifest has {manifest['stats']['chunks_done']} done chunks")
    else:
        manifest = new_manifest(job_id, spec["code_version"], chunk_ids)

    processed = 0
    for c, url in zip(manifest["chunks"], files):
        if max_chunks is not None and processed >= max_chunks:
            progress(f"chunk cap ({max_chunks}) reached; stopping cleanly")
            break
        if soft_deadline_at is not None and time.time() > soft_deadline_at:
            progress("soft deadline reached; stopping cleanly")
            break
        if c["status"] == "done":
            continue
        chunk_id = c["chunk_id"]
        c["status"] = "in_progress"
        c["attempts"] += 1
        c["last_worker"] = os.environ.get("WORKER_RUN", "dev")
        c["updated_at"] = _now()
        save_manifest(manifest_path, recalc_stats(manifest))

        try:
            out_path = workdir / job_id / f"{chunk_id}.jsonl"
            rows = _process_chunk(task, url, chunk_id, ds.get("columns"), workdir, out_path, token_ro)
            checksum = sha256_file(out_path)
            c.update(status="done", rows=rows, sha256=checksum,
                     output=str(out_path.name), metrics={"rows": rows}, updated_at=_now(), error=None)
            if write_artifacts and token_rw:
                hf_upload_file(checkpoint_repo, out_path, f"{job_id}/{chunk_id}.jsonl", token_rw)
            processed += 1
            progress(f"chunk {chunk_id}: done rows={rows}")
        except Exception as e:  # noqa: BLE001
            c["status"] = "failed" if c["attempts"] >= spec.get("retry", {}).get("max_attempts", 3) else "pending"
            c["error"] = str(e)[:500]
            c["updated_at"] = _now()
            progress(f"chunk {chunk_id}: {c['status']} ({e})")

        save_manifest(manifest_path, recalc_stats(manifest))

    manifest = recalc_stats(manifest)
    manifest["updated_at"] = _now()
    save_manifest(manifest_path, manifest)
    if write_artifacts and token_rw:
        try:
            hf_upload_file(checkpoint_repo, manifest_path, spec["checkpoint"].get("manifest_path", "manifest.json"), token_rw)
        except RuntimeError as e:
            progress(f"FINAL manifest upload failed: {e}")
            progress("chunks were processed but artifacts could not be saved — check HF_TOKEN_RW secret")
            return manifest
    progress(f"job {job_id}: {manifest['stats']['chunks_done']}/{manifest['stats']['chunks_total']} chunks done")
    return manifest


def _process_chunk(task: str, url: str, chunk_id: str, columns: list[str] | None,
                   workdir: Path, out_path: Path, token: str | None) -> int:
    """Download shard (cached), scan rows, emit task output jsonl. Local-light
    in dev mode: shard stays in workdir, never in the engine repo."""
    shard_dir = workdir / "shards"
    shard_path = shard_dir / url.rsplit("/", 1)[-1]
    if not shard_path.exists():
        hf_download(url, shard_path, token)

    rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for row in iter_rows_parquet(shard_path, columns):
            if task == "scan_parquet":
                # generic: total chars across string fields (schema-agnostic)
                n_chars = sum(len(v) for v in row.values() if isinstance(v, str))
                rec = {"chunk_id": chunk_id, "n_chars": n_chars, "n_fields": len(row)}
            elif task == "sample_extract":
                rec = {"chunk_id": chunk_id, "row": {k: str(v)[:2000] for k, v in row.items()}}
            else:
                raise ValueError(f"worker task not implemented yet: {task}")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1
    shard_path.unlink(missing_ok=True)  # delete shard after processing
    return rows
