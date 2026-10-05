# Index of `results/`

This directory grew organically across many separate experiment campaigns,
so filenames follow several different naming schemes depending on when a
file was produced. Nothing here has been renamed or moved (file paths used
throughout NOTES_AND_ASSUMPTIONS.md stay valid) -- this index exists purely
to let you browse by research question instead of by flat filename. For the
full investigation narrative behind any of these, read
[`NOTES_AND_ASSUMPTIONS.md`](../NOTES_AND_ASSUMPTIONS.md) (sections
referenced below).

## Pipeline validation (not meaningful results)

- `ollama_smoke.json`, `smoke_test.json`, `pilot_log.txt`,
  `small_fast_log.txt` -- earliest pipeline sanity checks, 2026-08-18/26.
- `pilot_*.json` (22 files) -- 20-step smoke-test grid, all 5 frameworks x
  2 tiers x 2 metrics. Section (f). Job IDs: `jobs/pilot.txt`.

## The main 200-step grid (Llama 3.1 8B/70B)

- `results.json` -- first full grid, pre negotiation-bug-fixes. Section (g).
  Job IDs: `jobs/full.txt`.
- `results_postfix.json` -- same grid, rerun after all 3 negotiation bug
  fixes (sections h/i/i.2). Section (j). Job IDs: `jobs/full2.txt`.
- `full_<framework>_<tier>_<metric>.json` (20 files) -- per-shard outputs
  merged into `results_postfix.json` above.

## RQ1 -- cross-model generalization (Llama vs. Gemma vs. Qwen)

- `full_gemma_<framework>_small_<metric>.json` (10 files) -- Gemma2:9b,
  full small-tier grid. Sections (l), (l.2), (m item 1).
- `rq1_qwen257b_<framework>_small_<metric>.json` (3 files) -- Qwen2.5:7b,
  reduced small-tier scope (standalone/cost, negotiation_tool/both
  metrics). Section (n)/(o). **Qwen small-tier coverage is 3/10 shards by
  deliberate scope, not an oversight** -- see section (q)'s RQ1 audit.
- `rq1_gemma227b_<framework>_large_<metric>.json` (2 files) -- Gemma2:27b,
  reduced large-tier scope (standalone/cost, negotiation_tool/bullwhip).
  Section (o).
- **Qwen large-tier: 0 files, never attempted** (confirmed, section (q)).

## RQ2 -- memory/transcript-grounding ablation

- `rq2_noground_negotiation_tool_large_<metric>.json` (2 files) -- original,
  unpaired ablation (seed defaulted to 13). Section (n)/(o).
- `rq35_seed{13,17,23,29,31}_negotiation_tool_large_<metric>.json`
  (10 files) -- the **grounded** side of the paired comparison, 5 seeds.
  Section (o).
- `rq_paired_seed{17,23,29,31}_noground_negotiation_tool_large_<metric>.json`
  (8 files) -- the **ungrounded** side of the paired comparison, seeds
  17/23/29/31 (seed 13's ungrounded arm reuses `rq2_noground_*` above,
  verified as a valid pairing in section (o)). Section (p)'s 5-seed paired
  table and mean paired difference.
- `mech_seed{13,17,23,29,31}_noground_negotiation_tool_large_<metric>.json`
  (10 files, large -- these carry full transcripts) -- re-run of the
  ungrounded arm **with `--include-transcripts`**, used only for the
  mechanism investigation in section (r) (is "ungrounded beats grounded" a
  real memory effect, or does ungrounded degenerate into independent
  `round(own_eoq)` answers?). Summary cost/bullwhip numbers match the
  `rq_paired_seed*`/`rq2_noground_*` files above exactly -- these are the
  same runs, just with transcripts captured this time.

## RQ3 -- negotiation-style transcript taxonomy (Llama vs. Gemma vs. Qwen)

- `diagnostic/diag_negotiation_tool_*.json` (7 files) -- Llama transcripts
  across every fix-state (pre-fix through all-fixes-applied), n=2800
  pooled. Section (n) item 5.
- `rq3_transcripts_gemma29b_negotiation_tool_small_<metric>.json` (2 files)
  and `rq3_transcripts_qwen257b_negotiation_tool_small_<metric>.json`
  (2 files) -- Gemma2:9b and Qwen2.5:7b transcripts, same classifier.
  Section (p)'s three-way table.

## RQ4 -- effort vs. benefit

Uses `results_postfix.json` directly (elapsed_sec + cost per framework,
70B tier) -- no dedicated files, see section (n) item 6's table.

## RQ5 -- bullwhip metric variance/reliability

Uses the `rq35_seed*_negotiation_tool_large_bullwhip.json` files listed
under RQ2 above (same 5-seed grounded data serves both RQs) -- see
section (o)'s variance table.

## `diagnostic/` subdirectory

- `diag_info_sharing_tool_<metric>.json`, `diag_standalone_small_cost.json`
  -- early per-step diagnostic dumps, not tied to a specific RQ above.
- `diag_negotiation_tool_<metric>[_<fixstate>].json` -- see RQ3 above;
  also the source of section (m) item 3's full negotiation fix-state
  inventory table.
- `gemma_full_standalone_small_cost.json`, `gemma_smoke_standalone_small_
  cost.json` -- early Gemma pipeline-validation, precedes the full grid
  above.
