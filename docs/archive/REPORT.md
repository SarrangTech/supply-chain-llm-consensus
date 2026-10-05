> **Superseded, 2026-10-05.** This is a day-1 snapshot (last updated
> 2026-08-18) and is now factually wrong about the project's state --
> e.g. it says the full real-data run never completed due to insufficient
> OpenRouter credit and incapable local hardware, both long since resolved
> by the move to the Explorer HPC cluster. Kept for historical interest
> only. **For the actual current state, read
> [`NOTES_AND_ASSUMPTIONS.md`](../../NOTES_AND_ASSUMPTIONS.md) and
> [`README.md`](../../README.md).**

# Replication Report: Agentic LLMs in the Supply Chain

Target paper: Jannelli, Schoepf, Bickel, Netland & Brintrup (2025), "Agentic
LLMs in the supply chain: towards autonomous multi-agent consensus-seeking,"
*International Journal of Production Research*, DOI 10.1080/00207543.2025.2604311
(arXiv:2411.10184).

**This is the short version.** For a full chronological, step-by-step
account of every action taken and the rationale behind it, see
[FULL_SESSION_LOG.md](FULL_SESSION_LOG.md). For the focused list of
paper-specific deviations and judgment calls (without the narrative), see
[NOTES_AND_ASSUMPTIONS.md](NOTES_AND_ASSUMPTIONS.md).

## 1. Executive summary

The full codebase replicating the paper's methodology is built, tested, and
validated end-to-end. It has NOT yet completed a full real-data run,
because it hit two independent real-world blockers on the way to one:
insufficient OpenRouter account credit, and this machine's hardware being
physically incapable of running the "large" model tier locally. Neither
blocker reflects a defect in the code; both are resourcing decisions this
report lays out options for.

## 2. What was done

### 2.1 Preliminary research (before writing any code)
- Searched for the paper's own source code (it states "we open-source our
  code") across arXiv, the IJPR page, and general web search. **No
  repository was found.**
- Searched for the base simulation environment this paper extends -- Liu,
  Hu, Peng & Yang (2022), "Multi-Agent Deep Reinforcement Learning for
  Multi-Echelon Inventory Management" (Rotman/SSRN working paper 4262186).
  **No repository was found for this either.**
- Conclusion: this is a from-scratch reconstruction from the paper's text,
  not an adaptation of existing code. Documented in `NOTES_AND_ASSUMPTIONS.md`.

### 2.2 Explicit, user-approved deviations from the paper
- **Models**: the paper uses Gemini 1.5 Flash / Gemini 1.5 Pro directly.
  Those exact models are now retired from Google's API entirely (confirmed
  via live documentation check). No Gemini credentials were available
  either way. Per your instruction, the codebase uses **Llama 3.1 8B
  Instruct** (Flash-analog) and **Llama 3.1 70B Instruct** (Pro-analog) via
  **OpenRouter** instead -- a same-family, small-vs-large substitution that
  preserves the paper's structural comparison, not its literal numbers.
- **Customer demand series**: the paper's exact Merton Jump Diffusion
  parameters ("run number 13") were never published and could not be
  recovered. `src/demand.py` implements a hand-tuned placeholder
  (OU-mean-reversion + jumps) that visually matches the paper's Figure 17,
  with every parameter named, exposed, and flagged as unconfirmed.

### 2.3 System built (all in `C:\Users\batha\supply-chain-llm-consensus\`)
- `src/environment.py` -- 3-echelon sequential supply chain simulation
  (Retailer/Distributor/Manufacturer), lead time 2, full cost accounting.
- `src/tools.py` -- linear-regression demand forecast tool; EOQ tool.
- `src/baselines.py` -- (S,s) restocking policy, tool-only baseline, Chen
  et al. (2000) centralised-demand baseline.
- `src/prompts.py` -- the paper's P1-P7 prompt structure (Appendix 2),
  including the Flash-vs-Pro tool-emphasis divergence (Appendix 3).
- `src/llm_client.py` -- OpenRouter chat-completions client with
  retry-on-malformed-output, matching the paper's stated retry rationale.
- `src/frameworks/` -- all five consensus-seeking frameworks (Figure 3):
  standalone, standalone+tool, info-sharing, info-sharing+tool, and
  negotiation+tool. Info-sharing and negotiation are implemented as
  **LangGraph** state graphs mirroring the paper's Figures 6 and 7
  specifically, per the paper's own stated tooling choice.
- `src/metrics.py`, `src/results_table.py`, `src/experiment_runner.py` --
  global cost, per-agent/aggregate bullwhip coefficient of variation, and a
  25-configuration experiment grid (12 cost configs + 13 bullwhip configs)
  reproducing the paper's Table 1 / Table 2 layout, with config-level
  filtering so independent shards can run as parallel processes.
- `run_experiments.py` -- CLI: `--mock` (free dry run), real runs, `--steps`
  overrides for pilots, `--only-*` filters for sharding, `--merge` for
  recombining shard outputs.
- Every judgment call made resolving an ambiguity in the paper's text (agent
  indexing convention, within-timestep simulation order, the Chen et al.
  2000 baseline's exact formula, per-pair EOQ computation, middle-agent
  decision reconciliation, etc.) is logged in `NOTES_AND_ASSUMPTIONS.md`
  rather than silently assumed.

### 2.4 Validation performed
1. **Mock dry run** (free, deterministic stub LLM): all 25 configurations
   ran end-to-end with no errors, across both metrics and both model tiers
   -- confirms the environment, tools, baselines, prompts, and both
   LangGraph frameworks are wired correctly.
2. **Real-API smoke test** (2 simulation steps, real OpenRouter calls):
   caught a genuine issue -- Llama 3.1 8B, unlike the paper's Gemini
   models, defaults to verbose step-by-step markdown reasoning and blew
   through the paper's fixed 90-output-token budget before ever reaching
   the required `[[N]]` answer, causing malformed-output failures.
   **Fixed** by adding a strict-format system message (an analogous,
   disclosed adaptation to the paper's own documented Flash-vs-Pro prompt
   tweak in Appendix 3), without changing the paper's stated 90-token limit.
   Re-ran clean after the fix, all 25 configs succeeded.
3. **Timing/cost analysis** from the smoke test: ~24,000 total LLM calls
   needed for the full 200-step x 25-config grid (negotiation's structured
   back-and-forth accounts for the bulk of that). At confirmed live
   OpenRouter pricing ($0.02/$0.04 per M input/output tokens for 8B;
   $0.40/$0.40 for 70B), estimated total cost is **roughly $3-5**. Estimated
   sequential wall-clock time is **roughly 12 hours**, driven almost
   entirely by negotiation's ~16 sequential API calls per simulation step
   per neighbour-pair -- not a cost problem, a call-chaining problem.
4. **Real 20-step pilot, partial run**: 2 non-LLM baselines and one full
   LLM configuration (`standalone`/8B/cost: cost=5,941, bullwhip=1.10,
   80.8s for 20 steps, consistent with the smoke-test timing model) all
   completed successfully. The run then failed on the very next
   configuration with `402 Client Error: Payment Required` from
   OpenRouter -- **an account/key credit exhaustion, not a code defect.**
5. **Hardware feasibility check** for local inference (in response to
   asking about a free/local alternative): this machine has an 8-core Intel
   Core Ultra 7 256V, **15.6GB total RAM**, ~43.5GB free disk, and an
   integrated Intel Arc 140V GPU (no NVIDIA GPU). Llama 3.1 8B (quantized)
   fits comfortably and would run locally via Ollama at no cost. **Llama
   3.1 70B needs ~40GB+ RAM even quantized and cannot be loaded on this
   machine at all** -- this is a hard capacity limit, not a speed one.

## 3. Where things stand right now

- Code: complete, validated, ready to run.
- Real data: only 1 of 25 configurations has a real, complete result so
  far (standalone/8B/cost, 20 steps). Everything else is blocked on one of
  the two issues below.
- **Blocker A**: OpenRouter account/key is out of credit (HTTP 402).
- **Blocker B**: this machine cannot run the 70B "large" tier locally under
  any circumstances (RAM ceiling, not a patience problem).

## 4. Options going forward, with rationale

| # | Option | Cost | Time to finish full grid | What you keep | What you give up |
|---|--------|------|---------------------------|----------------|-------------------|
| 1 | **Add ~$5-10 credit to OpenRouter** | ~$3-5 (already computed from real pricing) | Hours (esp. if shards run in parallel, already built) | Full fidelity: real small-vs-large model gap, exact 200-step spec | Nothing meaningful -- cheapest path to an actually-finished replication |
| 2 | **8B local (free) + 70B via paid API (hybrid)** | ~$2.50-3.50 (only the 70B share) | Similar to #1 | Same fidelity as #1, at lower cost | Slightly more moving parts (two execution paths instead of one) |
| 3 | **Both tiers local, smaller "large" substitute (~13-14B model)** | $0 | Hours, all local, no rate limits | Zero ongoing cost, no API dependency at all | Weakens the paper's actual point of comparison -- an 8B-vs-13B gap is much smaller than Gemini's real Flash-vs-Pro gap, so "does bigger model help?" becomes a much less meaningful question |
| 4 | **Skip the large tier, 8B only** | $0 | Fastest, half the grid | Simplicity, zero cost | Drops one of the paper's two research questions entirely (whether model size affects consensus quality) |
| 5 | **Free hosted-API tiers** (OpenRouter `:free` models or Google AI Studio free tier) | $0 | **~24+ days**, rate-limited to 20 req/min and 50-1,000 req/day against a ~24,000-call need | Zero cost | Turns a same-day task into a multi-week one; also, exact free-tier model availability shifts month to month (Llama 3.1 free listings were reportedly delisted in early August 2026) so model identity isn't even stable |
| 6 | **Get a better device** (rent a cloud GPU instance, or use/borrow a machine with 64GB+ RAM and ideally an NVIDIA GPU) | Variable -- cloud GPU rental (e.g. ~$0.50-2/hr depending on provider/GPU tier) or $0 if you already have access to suitable hardware | Fast once provisioned; provisioning/setup adds upfront time | True fully-local execution of both tiers, no ongoing per-call cost after the machine/instance is available, reusable for future work | Requires either spending on rented compute (which competes directly with option 1/2's ~$3-5) or having spare hardware; more setup complexity (driver/CUDA setup, model downloads) |
| 7 | **Reduce experimental scope** (fewer steps and/or fewer configs) | Scales down whichever cost/time axis you're optimizing | Scales down accordingly | Faster/cheaper by design | A further, explicitly-flagged deviation from the paper's 200-step / 25-config spec -- noisier bullwhip/cost estimates, especially for slower-converging frameworks |

**My assessment, for what it's worth**: options 1 and 2 dominate on
pure cost-effectiveness -- a cloud GPU rental (option 6) would need to run
for less than a couple of hours to already cost more than the entire
OpenRouter bill for the full grid, and free tiers (option 5) trade an
essentially negligible amount of money for multiple weeks of calendar
time. Option 2 (hybrid) is marginally cheaper than option 1 and reuses the
Ollama setup already scoped out, at the cost of a bit more operational
complexity. Options 3/4/7 are reasonable if the goal shifts from "replicate
the paper's comparison as faithfully as affordable" to "get *some* signal
for free/fast" -- that's a legitimate but different goal, worth being
explicit about if you choose it.

## 5. Everything currently unresolved / open

- Which option above to pursue (money vs. time vs. fidelity is your call).
- If option 1/2/6: OpenRouter needs credit added (blocker A).
- If option 2/3/6-with-existing-hardware: Ollama needs to be installed
  (`winget install Ollama.Ollama` is confirmed available on this machine).
- Live model availability for whichever path is chosen should be
  re-checked at execution time (both OpenRouter's paid catalog and any
  `:free` listings shift over time, as already observed).
