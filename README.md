# Supply Chain LLM Consensus — Replication

Reconstruction of Jannelli, Schoepf, Bickel, Netland & Brintrup (2025),
"Agentic LLMs in the supply chain: towards autonomous multi-agent
consensus-seeking" (IJPR, DOI 10.1080/00207543.2025.2604311 / arXiv:2411.10184).

**Read [NOTES_AND_ASSUMPTIONS.md](NOTES_AND_ASSUMPTIONS.md) first.** No
original code exists for this paper or its base environment (Liu et al.
2022); this is a from-scratch reconstruction from the paper's text, with an
explicit, user-approved model substitution (Llama 3.1 8B/70B instead of
Gemini 1.5 Flash/Pro) and one unconfirmed parameter (the exact Merton Jump
Diffusion demand parameters). Every judgment call is logged there, not
buried in code comments.

**Compute moved to an HPC cluster (2026-08-29/30).** Both model tiers now
run locally and for free via Ollama on Northeastern's Discovery cluster
("Explorer", SLURM-scheduled): 8B on AVX512 CPU nodes or GPU, 70B on GPU
only (CPU is ~300x too slow for 70B -- measured, not assumed). This
replaced the earlier OpenRouter-for-70B / local-Ollama-for-8B split.
Section (f) of NOTES_AND_ASSUMPTIONS.md has the full writeup: a real proxy
misconfiguration, a thread-oversubscription bug, an AVX512-vs-no-AVX512
30x throughput gap, and the resulting node-selection rules baked into
`llm_client.py` and `config.py`. **[EVIDENCE.md](EVIDENCE.md)** has the raw,
independently-checkable proof this actually ran on the cluster (SLURM
accounting records, GPU hardware UUIDs, model-loader confirmation of which
weights loaded) rather than just a narrative describing it.

## Layout

```
src/
  config.py              Fixed experimental parameters (Appendix 6, Table 2) + model IDs
  demand.py               Merton-Jump-Diffusion-style customer demand generator (flagged placeholder)
  environment.py           3-echelon sequential supply chain simulation (Section 3)
  tools.py                 Demand-forecast (linear regression) and EOQ tools (Section 4.4)
  baselines.py             (S,s) policy, tool-only baseline, Chen et al. (2000) baseline (Section 5.1)
  prompts.py               P1-P7 prompt construction (Appendix 2) + negotiation prompts (Section 6.2.1)
  llm_client.py            OpenRouter + Ollama chat-completions clients, retry-on-malformed-output
  mock_llm.py              Free deterministic stand-in for --mock pipeline validation
  metrics.py               Global cost, per-agent & aggregate bullwhip coefficient of variation
  results_table.py         Formats output to mirror the paper's Table 1 / Table 2 layout
  experiment_runner.py     Builds and runs the full 25-configuration grid
  frameworks/
    standalone.py          (a) Standalone LLM agents, (c) + tool usage -- no LangGraph needed
    info_sharing.py        (b) Info sharing, (d) + tool usage -- LangGraph, mirrors Figure 6
    negotiation.py         (e) Negotiation around tool output -- LangGraph, mirrors Figure 7
run_experiments.py         CLI entry point
submit_shard.sh            SLURM job submission helper (one shard = one job; see NOTES_AND_ASSUMPTIONS.md section (f))
pilot_job_ids.txt          Job-ID provenance for the 20-shard pilot grid (proof of what ran where)
results/                   Output JSON + printed tables land here
```

## Running it

```bash
pip install -r requirements.txt

# Free pipeline validation (mock LLM, no API calls, NOT meaningful results):
python run_experiments.py --mock --steps 20

# Real run, on the HPC cluster -- both tiers free/local via Ollama.
# See NOTES_AND_ASSUMPTIONS.md section (f) for the SLURM job-script
# patterns (proxy env vars, CPU thread pinning, AVX512 node constraints).
# Grid is shardable by framework/metric/model-tier so shards can run as
# separate parallel SLURM jobs, then get merged:
python run_experiments.py --only-framework negotiation_tool --only-metric cost \
    --only-model-tier small --out results/shard_neg_cost_small.json
python run_experiments.py --merge "results/shard_*.json" --out results/results.json

# Or the full grid sequentially in one process:
python run_experiments.py --steps 20   # pilot first, recommended
python run_experiments.py              # full 200-step x 25-config grid (paper's spec)
```

Output is a JSON dump of every configuration's result (`results/results.json`
by default) plus two printed tables -- one for the Global Cost grid (12
configs: 2 non-LLM baselines + 5 frameworks x 2 model tiers), one for the
Global Bullwhip grid (13 configs: 3 non-LLM baselines + 5 frameworks x 2
model tiers) -- formatted to line up against the paper's Table 1 (Section
6.1 / Appendix 4) and Table 2 (Section 6.2) respectively.

## What matches the paper exactly

- All fixed parameters in `config.py` (3 agents, lead time 2, 200 steps, 10-
  step agent memory, 30-step tool lookback, 3 negotiation rounds, max order
  100, temperature 0.1, max output tokens 90, unit costs) — Appendix 6,
  Table 2, verbatim.
- The five frameworks and their information-flow structure (Figure 3) and
  LangGraph orchestration shape (Figures 6 and 7).
- The two tools (linear-regression demand forecast; EOQ) and their exact
  formulas (Section 4.4, Section 3.2).
- The three non-LLM baselines and the prompt component breakdown (P1-P7,
  Appendix 2) and the Flash-vs-Pro prompt divergence (Appendix 3).

## What does NOT match the paper (by necessity, all flagged)

- Model family (Llama 3.1 8B/70B, not Gemini 1.5 Flash/Pro).
- Exact demand series (Merton Jump Diffusion parameters were never
  published; this uses a hand-tuned placeholder that visually approximates
  Figure 17 only).
- A handful of documented judgment calls where the paper's text was
  ambiguous (agent indexing convention, within-timestep simulation order,
  Chen et al. 2000 baseline's exact formula, per-pair EOQ computation,
  middle-agent decision reconciliation) — see NOTES_AND_ASSUMPTIONS.md
  section (d) for the full list and reasoning behind each.

## Current status (2026-08-31)

The full 20-shard pilot grid (5 frameworks x 2 model tiers x 2 metrics,
`--steps 20`, real local Ollama backends, no mocks) has been run to
completion on the HPC cluster — every shard produced valid output. This is
a pipeline-validation milestone, **not** a result: 20 steps is a smoke
test, the paper's spec is 200. Raw output is in `results/pilot_*.json`;
job-ID provenance (which SLURM job produced which file, including two
shards that needed a rerun after infrastructure bugs were found and fixed)
is in `pilot_job_ids.txt` and written up in full in
NOTES_AND_ASSUMPTIONS.md section (f), along with every measured throughput
number (tokens/sec by backend/CPU generation) that the node-selection
rules in `config.py`/`llm_client.py` are based on.

The pipeline has also been validated end-to-end with `--mock` (all 25
configurations run without error across both metrics and both model
tiers). The full 200-step x 25-config grid (the paper's actual spec) has
not been run yet.
