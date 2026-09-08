# humn-worker (public)

Repo: https://github.com/openagent1/humn-worker

Free compute worker for the HUMN Data Engine. Runs on GitHub Actions
standard runners (free for public repos — verified 2026-09-06).

**Rules:**
- This repo is PUBLIC for free minutes → it must never contain data or secrets.
- Job specs only reference credential *names*; actual tokens come from
  GitHub encrypted secrets (`HF_TOKEN_RO`, `HF_TOKEN_RW`).
- Workers stream data remote→worker and upload artifacts to the HF
  checkpoint repo; only manifests/logs return as GH artifacts.

## One-time setup

1. Create a **public** repo on GitHub and push this directory.
2. In repo Settings → Secrets and variables → Actions, add:
   - `HF_TOKEN_RO` — Hugging Face read token (datasets:read)
   - `HF_TOKEN_RW` — Hugging Face write token scoped to `humn-internal/*` repos
3. Drop job specs into `specs/`, commit.

## Dispatch a job

From the Actions UI: *HUMN Data Worker* → *Run workflow* → pick spec.

Or from the local orchestrator:

```powershell
$env:GITHUB_TOKEN = "<PAT with repo scope>"
uv run humn-engine job trigger-gh <job_id>          # from humn-engine/
```

Or via API:

```bash
curl -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/<owner>/humn-worker/actions/workflows/worker.yml/dispatches \
  -d '{"ref":"main","inputs":{"spec":"specs/2026-09-06-dev-arsarcasm-scan-001.yaml"}}'
```

## Runtime behavior

- `SOFT_DEADLINE_MIN` (default 150): worker checkpoints and exits cleanly
  well before the job timeout; re-dispatch resumes from the manifest.
- Chunk outputs are idempotent: `(job_id, chunk_id, code_version)` naming;
  re-runs overwrite, never duplicate.
- Manifest after every chunk: `hf://<checkpoint_repo>/manifest.json`.
