# Supply Chain LLM Consensus — Replication

Reconstruction of Jannelli, Schoepf, Bickel, Netland & Brintrup (2025),
"Agentic LLMs in the supply chain: towards autonomous multi-agent
consensus-seeking" (IJPR, DOI 10.1080/00207543.2025.2604311 / arXiv:2411.10184).

**Read [NOTES_AND_ASSUMPTIONS.md](NOTES_AND_ASSUMPTIONS.md) first.** No
original code exists for this paper or its base environment (Liu et al.
2022); this is a from-scratch reconstruction from the paper's text, with an
explicit, user-approved model substitution (Llama 3.1 8B/70B instead of
Gemini 1.5 Flash/Pro) and one unconfirmed parameter (the exact Merton Jump
Diffusion demand parameters). Every judgment call, every bug found and
fixed, and every research-question finding is logged there in full, dated,
evidence-backed detail (sections a-r as of 2026-10-05) -- this README is a
summary and entry point, not a replacement for it.

**Compute runs on an HPC cluster (migrated 2026-08-29/30).** Both model
tiers run locally and for free via Ollama on Northeastern's Discovery
cluster ("Explorer", SLURM-scheduled): small-tier models (8B-9B class) on
AVX512 CPU nodes or GPU, large-tier models (27B-70B class) on GPU only
(CPU is ~300x too slow for 70B -- measured, not assumed). Section (f) of
NOTES_AND_ASSUMPTIONS.md has the full infrastructure writeup.
**[EVIDENCE.md](EVIDENCE.md)** has the raw, independently-checkable proof
this actually ran on the cluster (SLURM accounting records, GPU hardware
UUIDs, model-loader confirmation of which weights loaded).

## The five research questions

Beyond straight replication, this project investigates five questions the
paper itself doesn't address. Status below is the short version -- the
full evidence, per-seed tables, and exact file references for each are in
NOTES_AND_ASSUMPTIONS.md section (q) (readiness audit) and section (r)
(RQ2's mechanism follow-up).

| RQ | Question | Status |
|---|---|---|
| **RQ1** | Does the paper's pattern hold across model families (Llama/Gemma/Qwen), not just the one substituted model? | Small tier: full Llama-vs-Gemma comparison done; Qwen covers 3/10 shards by deliberate reduced scope. Large tier: a single Gemma-vs-Llama contrast only; Qwen never attempted at large tier. |
| **RQ2** | Does grounding the negotiation's final answer in the actual conversation transcript (vs. not) change outcomes? | Done -- a proper 5-seed paired comparison. Grounding does not help; it measurably hurts cost (5/5 seeds) and one of two bullwhip-run outcomes. **Follow-up (section r):** for the cost-run, this is because the ungrounded condition degenerates into independent `round(own_eoq)` answers, not real negotiation -- a different claim than "memory hurts negotiation." The bullwhip-run's mechanism is genuinely ambiguous. |
| **RQ3** | Does negotiation *style* (convergence vs. breakdown) differ by model family? | Done -- three-way classifier results (Llama n=2,800, Gemma n=792, Qwen n=792). |
| **RQ4** | Is negotiation's extra LLM-call overhead justified by better outcomes? | Done -- negotiation is ~14x slower than standalone and still produces worse cost. Not justified. |
| **RQ5** | How reliable/reproducible is the bullwhip metric itself across repeated runs? | Done -- 5 seeded reruns show the bullwhip-run's bullwhip standard deviation (~3.6) exceeds its own mean (~3.0). The metric itself is unreliable for this framework; single-run bullwhip numbers (including the original paper's) should be read skeptically. |

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
demo_negotiation.py         Live demo: streams the real negotiation LangGraph turn-by-turn to the console
demo_negotiation_grounded.py  Separate demo-only prompt variant grounded in richer inventory state (NOT the paper-faithful prompts -- see its own docstring)
jobs/                      SLURM job-ID provenance logs (one file per campaign; see jobs/README.md)
results/                   Output JSON + printed tables land here (see results/INDEX.md to browse by research question)
docs/archive/              Superseded early-phase docs, kept for history only
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

# Model-family / seed / memory-ablation overrides (all off by default,
# existing behavior unaffected unless set -- see NOTES_AND_ASSUMPTIONS.md
# section (n) for exactly what each does):
OLLAMA_SMALL_MODEL=gemma2:9b python run_experiments.py --only-model-tier small ...
OLLAMA_LARGE_MODEL=gemma2:27b python run_experiments.py --only-model-tier large ...
EXPERIMENT_SEED=17 python run_experiments.py ...
DISABLE_TRANSCRIPT_GROUNDING=1 python run_experiments.py --only-framework negotiation_tool ...
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

- Model family (Llama 3.1 8B/70B as the primary substitution for Gemini 1.5
  Flash/Pro; Gemma 2 9B/27B and Qwen 2.5 7B added later as a cross-model
  generalization check -- RQ1).
- Exact demand series (Merton Jump Diffusion parameters were never
  published; this uses a hand-tuned placeholder that visually approximates
  Figure 17 only).
- A handful of documented judgment calls where the paper's text was
  ambiguous (agent indexing convention, within-timestep simulation order,
  Chen et al. 2000 baseline's exact formula, per-pair EOQ computation,
  middle-agent decision reconciliation) — see NOTES_AND_ASSUMPTIONS.md
  section (d) for the full list and reasoning behind each.

## Current status (2026-10-05)

**Bottom line up front: this replication does not reproduce the paper's
core structural claims** (that negotiation is the best framework on both
cost and bullwhip), even after finding and fixing three real negotiation
bugs. That's a legitimate, well-evidenced outcome for a from-scratch
replication with a substituted model family -- not a failure to document,
and not "successfully recreated" either. Full timeline:

- **2026-08-31 -- pipeline validation.** 20-shard pilot grid (`--steps 20`)
  completed on the cluster, every shard valid output. Not a result -- a
  smoke test. (Section f, `jobs/pilot.txt`.)
- **2026-09-02 -- first full 200-step grid.** All 20 LLM-driven shards
  completed, zero failures. Negotiation showed up as the *worst* framework
  on both metrics for 70B, contradicting the paper's claim it's the *best*.
  (Section g, `jobs/full.txt`.)
- **2026-09-05 through 09-13 -- three real negotiation bugs found, fixed,
  and verified** by inspecting actual transcripts, not guessed: (1) context
  loss in the final-answer step (stateless chat client meant the model
  answered blind), (2) decimal-point stripping instead of rounding, (3) a
  degenerate zero-EOQ startup case. Cost improved 4-18x after fixing all
  three. **Bullwhip did not recover**, and turned out to be highly
  run-to-run variable on its own (confirmed later and quantified in RQ5).
  Negotiation still didn't beat the paper's claimed pattern. (Sections
  h-j, `jobs/full2.txt`.)
- **2026-09-14/16 -- root cause of a separate 8B-tier cost catastrophe.**
  Not a bug: real per-step data showed unbounded inventory growth (10 ->
  10,077 units over 200 steps) because Llama 3.1 8B never used its own
  inventory level to stop ordering, despite it being in every prompt.
  Consistent with 8B likely not being a fair capability match for the
  undisclosed, heavily-tuned Gemini 1.5 Flash. (Section k.)
- **2026-09-16 through 10-05 -- cross-model generalization and four more
  research questions (RQ1-RQ5), see the table above.** Added Gemma 2
  (9B/27B) and Qwen 2.5 (7B) as additional model families, built a proper
  5-seed paired memory ablation, classified negotiation transcripts across
  all three model families, quantified negotiation's call-overhead vs.
  benefit, and quantified the bullwhip metric's own run-to-run variance.
  Also found and fixed real infrastructure bugs along the way (a disk-quota
  exhaustion, an exit-code-masking bug in job scripts, and a NUMA/thread-
  oversubscription bug that made CPU inference hang indefinitely) --
  documented as lessons in sections (n)-(r), not glossed over.
