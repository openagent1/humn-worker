# HUMN Data Engine — CHANGELOG

## Unreleased (v1.0 target)
- Canonical schemas: every row carries `id`, `source`, `lang`, `flags.*`, `dup_group`
- Per-language release layout: `pretrain_<lang>/`, `sft_chat_<lang>/` (uniform schema per config)
- `--exclude-sources`: compliance purge (LMSYS-derived rows never in public releases)
- `--flag-content`: lexicon quality flags (political/toxic/spam/low_info) — labels, never deletes
- Normalized near-dup removal (NFKC/lowercase/diacritic-stripped) in addition to exact dedup
- Full quality stats in every build (avg/median length, emoji/URL rates, flag rates, dup rates)
- `--curated`: human contribution path (`curated/TEMPLATE.jsonl`, strict validation)
- `worker/eval_build.py` + `worker/eval_score.py`: HUMN-Eval v0 (dialect_id auto-scored, rest queued for grading)
- Aggregate license corrected to ODC-By 1.0 (§4.2a) with per-source appendix + §4.3 model notice
- New sources: FineWeb2 arb/ary/ars/apc + OpenSubtitles (`opus_zip` worker source, `max_shards` cap)
- `ShardWriter`: all output files rolled at ~400MB (viewer-friendly)

## v0.3 (unreleased, superseded by v1 plan)
- Fixed viewer CastError (uniform schema per config, always-emit `lang`)
- Aya `inputs/targets` extraction, language-mix stats in card
- `--dataset-version` for repo naming + card title

## v0.2 (published, SUPERSEDED — delete after v1.0 verifies)
- 3.2M rows merged → 1.23M pretrain + 254K SFT conversations
- Known issues: mixed schemas in repo root (viewer CastError), LMSYS rows included
  in public release (license violation — purged from v1.0+), CC-BY-4.0 aggregate
  label incorrect (should be ODC-By)

## v0.1 (published, SUPERSEDED — delete after v1.0 verifies)
- First end-to-end pipeline proof (~10K rows)
