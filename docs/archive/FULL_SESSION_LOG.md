> **Superseded, 2026-10-05.** Last updated 2026-08-28, predates the full
> 200-step grid, every negotiation bug fix, the cross-model (Gemma/Qwen)
> ablation, and all five research questions' framing. Kept for historical
> interest only. **For the actual current state, read
> [`NOTES_AND_ASSUMPTIONS.md`](../../NOTES_AND_ASSUMPTIONS.md) and
> [`README.md`](../../README.md).**

# Full session log — every step taken, in order, with rationale

This is the exhaustive, chronological companion to `REPORT.md` (which stays
as the short executive summary + decision table). Read this if you want to
see exactly what was done, in what order, and why, rather than just the
end state.

---

## Phase 0 — Task intake

You gave a detailed specification to replicate Jannelli et al. (2025),
"Agentic LLMs in the supply chain," including: the environment, all five
consensus-seeking frameworks, both tools, all three baselines, the metrics,
every fixed experimental parameter, which models to use, the full 25-config
experiment grid, and four deliverables (code, run script, results table,
and a notes file flagging assumptions). The spec explicitly said: check for
the authors' original code first, and don't guess at ambiguous parameters
without flagging it.

---

## Phase 1 — Searching for existing code (Step 0 of your spec)

**What I did:**
- Ran four parallel web searches: the paper's title + "github", the arXiv
  ID (2411.10184) + "supply chain consensus LLM agents", the base paper
  (Liu, Hu, Peng & Yang 2022, "Multi-Agent Deep Reinforcement Learning for
  Multi-Echelon Inventory Management," Rotman/SSRN 4262186) + "github", and
  "Cambridge Supply Chain AI Lab SCAIL github."
- Fetched the arXiv abstract page (arxiv.org/abs/2411.10184) directly and
  asked it to list any code/repo links.
- Ran two more targeted searches: `site:github.com` for supply-chain
  LLM-consensus-negotiation-bullwhip-Gemini repos, and the author names
  combined with "consensus-seeking" + "github.com repository open source."
- One more search specifically for "HAPPO multi-echelon inventory
  management github" (HAPPO is the RL algorithm the Liu et al. 2022 paper
  uses), to try a different angle on the base-environment repo.

**Why:** the paper explicitly states "we open-source our code." If that
repo existed, it should be the ground truth for every ambiguous design
decision below, rather than my guessing. I searched from multiple angles
(paper title, arXiv ID, author names, lab name, base-paper name, algorithm
name) before concluding it doesn't exist publicly, rather than stopping
after one search.

**Result:** No repository found for either paper. One superficially
similar result (`zefang-liu/InvAgent`) turned out to belong to a
*different* paper (Quan & Liu 2024) that Jannelli et al. cite as related
work, not their own code. I reported this to you explicitly before writing
any code, per your Step 0 instruction, rather than silently proceeding.

---

## Phase 2 — Resolving blocking decisions before writing code

Three things in your spec couldn't be resolved by reading the paper or
guessing sensibly — they needed your input:

**2a. Merton Jump Diffusion demand parameters.** The paper names the demand
process and shows a plot (Figure 17) but never publishes drift, volatility,
jump intensity/mean/std, or seed. I checked for API keys already present in
the environment (`GEMINI_API_KEY`, `GOOGLE_API_KEY` — both absent) as part
of scoping this, then asked you: infer-and-flag, or wait for you to source
the real values separately. **You chose: infer and flag clearly.**

**2b. Model/API access.** No Gemini or Google API key existed in this
environment. I asked whether to build the harness only (no runs), or have
you supply a key immediately. **You answered with a third option I hadn't
offered: use open-source models instead of Gemini entirely.** This was a
bigger decision than my original question anticipated, so I followed your
actual answer rather than forcing it into one of my two original options.

**2c. Whether to pre-check Gemini model availability.** Moot once you chose
open-source models, but you'd said "check availability now" before that
became irrelevant.

**Given your "open source" answer, I then asked two more targeted
questions** (since "use open source" still leaves "which one" and "how do
we run it" unresolved):
- Checked the environment for a local model runtime first: no Ollama
  installed, no `nvidia-smi` (no NVIDIA GPU found at that point — this
  was revisited later in Phase 18 with a fuller hardware check), and no
  API keys for OpenAI/HuggingFace/Anthropic either. I did this check
  *before* asking you, so the question I asked would be grounded in what's
  actually possible on this machine rather than abstract.
- Asked which small/large model pair should stand in for Gemini
  Flash/Pro. **You chose Llama 3.1 8B (small) / 70B (large).**
- Asked how to actually run them, given no local runtime existed. **You
  chose a hosted API, with you providing the key.**
- Asked which hosted provider. **You chose OpenRouter.**

**Why this mattered:** the paper's own research questions include "does a
bigger model help," so preserving *some* meaningful small-vs-large gap
(not just two similarly-sized models) was important to get right before
committing to a specific provider and pricing model.

---

## Phase 3 — Building the codebase

Project created at `C:\Users\batha\supply-chain-llm-consensus\`. Built in
this order:

**3a. `src/config.py`** — every fixed parameter from the paper's Appendix 6
Table 2 (3 agents, 3 negotiation rounds, 10-step agent memory, max order
100, lead time 2, temperature 0.1, max output tokens 90, all unit costs = 1,
30-period tool lookback, 200 steps), the (S,s) policy parameters, and the
Llama-via-OpenRouter model mapping with an explicit comment block
documenting that this is a deviation, not an oversight.
*Rationale:* one file holding every number that came directly from the
paper's text, so nothing "fixed" is buried inside logic elsewhere.

**3b. `src/demand.py`** — the Merton Jump Diffusion demand generator.
- **First attempt:** a standard multiplicative geometric-Brownian-motion
  style Merton jump-diffusion (drift + volatility + compound Poisson
  jumps). I ran it (`python src/demand.py`) to sanity-check the shape
  against Figure 17 *before* wiring it into anything else.
- **Result:** it decayed to 0 within ~10 steps and stayed there — wrong
  shape. A pure multiplicative process can't decay from ~10 to a small
  positive baseline (~1-2) and then *stabilize* there; it either collapses
  to 0 or needs to be re-anchored, which isn't how GBM-style processes
  behave.
- **Fix:** redesigned as an Ornstein-Uhlenbeck mean-reversion process with
  added jumps (reverts toward a baseline_demand of ~1.6, with Gaussian
  noise and occasional upward jumps), keeping "Merton Jump Diffusion" as
  the closest available label per the paper, since a literal multiplicative
  Merton process can't reproduce the plotted behavior at all.
- Reran the generator and visually confirmed the new shape matched Figure
  17 closely enough (starts ~10, decays to ~1-2 by step ~40-50, then
  low-variance noise) to accept as a documented placeholder.
*Rationale for testing before integrating:* cheaper to catch a wrong demand
shape in isolation (one function, instant feedback) than after it's wired
into the full simulation and obscured by everything else.

**3c. `src/environment.py`** — the 3-echelon simulation.
- Decided the agent-indexing convention (agent 0 = Retailer/faces customer
  demand, agent 1 = Distributor, agent 2 = Manufacturer/unconstrained
  supply) by inference from Figures 9-11, where EOQ magnitude grows from
  agent 0 to agent 2 — matching the paper's statement that upstream agents
  amplify demand. The paper never states this as explicit indices, so this
  is flagged as a judgment call, not fact.
- Decided the within-timestep order of operations (receive shipments due
  this step → fulfill downstream demand, backlog-first → place new orders
  → compute costs → new orders arrive after `lead_time` steps). This is the
  standard multi-echelon/"beer game" simulation convention; the paper gives
  the cost formula and bullwhip formula but never a formal state-transition
  equation, so this is also flagged.
- `observe()` returns only the last 10 steps of history (matching the
  paper's stated 10-step agent memory), while the *full* history stays
  accessible on the `AgentState` object directly, for the tools that need a
  30-period lookback — kept these two windows deliberately separate since
  the paper distinguishes "observation memory" (10 steps) from what the
  demand-forecast/EOQ tools use (30 periods).

**3d. `src/tools.py`** — linear-regression demand forecast tool (`np.polyfit`
over the last 30 observations, falling back to the last observed order if
fewer than 30 exist yet, exactly as the paper specifies) and the EOQ tool
(`sqrt(2 * mean_demand * ordering_cost / holding_cost)`).

**3e. `src/baselines.py`** — the (S,s) policy (using raw on-hand inventory,
flagged as a judgment call since the paper doesn't say whether it means
raw inventory or inventory position), the tool-only baselines (the raw
tool output becomes the order, no LLM at all — this is explicitly what the
paper calls its "hard"/"strong" baseline), and a Chen et al. (2000)
centralised-demand, moving-average-forecast, order-up-to-policy baseline —
flagged as a simplified reconstruction since the paper describes this
baseline only at a high level, never publishing its exact formula.

**3f. `src/metrics.py`** — coefficient of variation per agent, and the
aggregate bullwhip metric as the *product* of each agent's CoV (per
Fransoo & Wouters 2000, as cited in the paper), plus total global cost.

**3g. Installed dependencies** (`langgraph`, `requests`, `numpy`) via pip,
and verified the imports worked before writing code that depended on them.
*Rationale:* confirm the tooling is actually available before building
architecture around it, especially LangGraph, since the paper is explicit
that it's the orchestration framework of choice.

**3h. `src/llm_client.py`** — a thin wrapper around OpenRouter's
OpenAI-compatible chat-completions endpoint via `requests`. Implements the
retry-on-malformed-output behavior the paper itself describes (temperature
0.1 specifically chosen so retries can occasionally produce a different,
hopefully well-formatted, answer), parses the required `[[N]]` format via
regex, and clips the parsed value into `[0, max_order]` as a hard-coded
backstop independent of whatever the prompt says — mirroring the paper's
own stated safety mechanism (Section 7.1).

**3i. `src/mock_llm.py`** — a free, deterministic stand-in for `LLMClient`,
used only to validate that the rest of the system (environment, prompts,
LangGraph graphs, metrics, experiment grid) is wired correctly without
spending any money. Explicitly documented as producing non-meaningful
numbers.

**3j. `src/prompts.py`** — assembles the P1-P7 prompt structure from
Appendix 2 (problem description, objective function, own observation,
10-step memory, recent demand/neighbor info, tool output with contextual
framing, final question + strict output format). Also implements the
Flash-vs-Pro prompt divergence from Appendix 3: the paper found Gemini Pro
needed an extra "use this as an upper bound" instruction that Gemini Flash
didn't need, so the same *style* of extra instruction is applied to the
"large" (70B) tier here, as a structural echo of that finding rather than a
guarantee it behaves identically on a different model family. Also
includes the negotiation-specific prompts (opening line, turn-taking,
final-answer question). Flagged clearly: the paper shows only one
fully-worked prompt example (Figure 13, cost + standalone + tool); every
other prompt variant here is built to match the paper's *described*
structure, not copied from an unseen original.

**3k. `src/frameworks/standalone.py`** — frameworks (a) and (c) (standalone,
and standalone+tool). No LangGraph here, since the paper only describes
using LangGraph for the two *communication* frameworks (Section 4.2) —
standalone agents don't talk to each other, so there's nothing to
orchestrate as a graph.

**3l. `src/frameworks/info_sharing.py`** — frameworks (b) and (d)
(information sharing, with and without tool usage). Built as a LangGraph
`StateGraph` with three nodes — `info_sharing → final_decision →
summarise` — directly mirroring the paper's Figure 6. Invoked once per
adjacent agent pair, dynamically, so it scales to a chain of any length
(matching the paper's own claim about its implementation). Documented a
judgment call here: when a middle agent (e.g. agent 1) takes part in *two*
pairs in the same step, the paper only says "the final decision stage
reconciles these negotiations" without giving the rule — I process pairs
strictly downstream-to-upstream and let an agent's *last*-computed
pairwise decision in the step win.

**3m. `src/frameworks/negotiation.py`** — framework (e). A LangGraph
`StateGraph` with the intro/loop/final-decision/summarise structure from
Figure 7, using a conditional edge to loop `agent_initiates ↔
agent_responds` exactly `num_iter = 3` times before finalizing — matching
the paper's stated round count. Documented a second judgment call here:
the paper's text says each agent should compute a *different* EOQ per
bilateral negotiation depending on which neighbor it's negotiating with,
but the paper's own worked example (Figures 9-11) is internally
inconsistent with that claim (the same agent's EOQ is shown identically in
both of its negotiations). Rather than silently pick a side, I documented
both readings and implemented the simpler, internally-consistent one
(each agent computes its tool output once per step from its own demand
history, reused across whichever pair it negotiates in).

**3n. Cleanup** — renamed an internal helper (`_p5_neighbor_info` →
`neighbor_info_text`) across `prompts.py` and `info_sharing.py` purely for
readability, no behavior change.

**3o. `src/experiment_runner.py`** — dispatches each config to the right
baseline or framework, builds the exact 12-config cost grid and 13-config
bullwhip grid your spec calls for, and runs them all against one shared,
deterministically-generated demand series so every config is compared
apples-to-apples.

**3p. `src/results_table.py`** — formats results into a table shaped like
the paper's Table 1 / Table 2, so numbers can be compared line-by-line
against the published ones (once real numbers exist).

**3q. `run_experiments.py`** — CLI entry point, initially with `--mock`,
`--steps`, and `--out` flags.

---

## Phase 4 — First validation pass (free, mock LLM)

Ran `python run_experiments.py --mock --steps 15 --out results/pilot_mock.json`.

**Result:** all 25 configurations completed with no errors — first proof
that the environment, both tools, all three baselines, all five frameworks
(including both LangGraph graphs), and the metrics/table code all fit
together correctly. Noticed the printed table's columns were misaligned
because the fixed 55-character label column was too narrow for labels like
`"Standalone LLM (meta-llama/llama-3.1-8b-instruct (Flash-analog))"`.

**Fix:** shortened the displayed model names (`Llama-3.1-8B (Flash-analog)`
instead of the full OpenRouter model ID), widened/adjusted the column
logic, and added line-wrapping for any label still too long. Reran at
`--steps 10` to confirm the fix — output was clean.

*Rationale for testing with mock first:* this validates 100% of the
plumbing (every framework, every baseline, every code path) at zero cost
and zero risk, before spending a cent or a minute on real API calls. Any
wiring bug found here is free to fix; the same bug found during a real run
would cost money and time to discover.

---

## Phase 5 — Documentation

Wrote `NOTES_AND_ASSUMPTIONS.md` (every deviation and every judgment call
made while resolving an ambiguity in the paper's text, each with the
reasoning behind it — not buried in code comments where it'd be easy to
miss) and `README.md` (project layout, how to run it, what does and
doesn't match the paper). Deleted the two mock dry-run output files
(`pilot_mock.json`, `pilot_mock2.json`) since they're wiring-test artifacts
with no real meaning, not results worth keeping around.

---

## Phase 6 — Mid-task interruption: boss email

You asked, mid-turn, for a status-update email to your boss. I paused the
technical build just long enough to draft one (summarizing the repo-search
finding, the model substitution, the frameworks in progress, and next
steps), then resumed the coding work in the same turn.
*Rationale:* your message came in while I had tool calls in flight; I
addressed it immediately in my next reply rather than deferring it to the
end, per instructions to treat mid-turn messages as needing a direct
response, not a queued one.

---

## Phase 7 — "Where are results and files saved"

Answered directly: everything lives under
`C:\Users\batha\supply-chain-llm-consensus\`, gave the full file tree, and
confirmed the `results\` folder was empty at that point (having just
deleted the mock test artifacts).

---

## Phase 8 — "Tell me how to do this"

Walked through the concrete steps to actually run the system for real: get
an OpenRouter key, set it as an environment variable (gave the PowerShell
syntax specifically, since PowerShell is this machine's primary shell),
check live model availability before spending money, run a small pilot
before the full grid, review pilot output, then run the full grid — and
asked whether you wanted to proceed solo or have me walk through the
OpenRouter signup screens with you.

---

## Phase 9 — "What would you want me to name api"

Answered: the key's name is just a dashboard label with no functional
effect (doesn't affect permissions or billing), suggested a couple of
descriptive options.

---

## Phase 10 — You provided a live API key; first real-API test

**What happened:** you pasted `sk-or-v1-...` directly into chat.

**My handling:**
- Flagged immediately that pasting a live secret into chat means it's now
  in this conversation's history — recommended treating it as exposed and
  considering rotation later, without being alarmist, since it's your key
  and your call.
- Committed to never writing it to any file or echoing it back in output.
- Chose to run a *minimal* 2-step smoke test across the full config grid
  first, rather than jumping straight to the previously-planned 20-step
  pilot — because the mock-LLM validation from Phase 4 proved the wiring
  works, but never proved that a *real* LLM's replies actually parse
  correctly against the strict `[[N]]` format. Testing that cheaply first,
  before committing more budget, was the point.
- Set the key via `export` inline in the same Bash call that ran the test,
  since this tool's shell state does not persist between separate Bash
  calls — the key had to be set and used in one shot.

**Result:** the smoke test crashed on the third configuration
(`standalone`/8B/cost) with a `MalformedOutputError`. The actual model
output showed why: Llama 3.1 8B, unlike the paper's Gemini models, defaults
to verbose step-by-step markdown reasoning ("## Step 1: Determine the
current state...") and burned through the paper's fixed 90-output-token
budget before ever reaching a `[[N]]`-formatted answer.

**Fix decision and rationale:** I deliberately did *not* raise
`max_output_tokens` above 90, because that's one of the paper's explicitly
stated fixed parameters (Appendix 6, Table 2) and silently changing it
would be exactly the kind of unflagged deviation your original spec told
me not to make. Instead, I added a system message
(`_STRICT_FORMAT_SYSTEM_MESSAGE`) instructing the model to skip all
reasoning and reply with only the bracketed number, and strengthened the
wording of the final-question prompts (P7 and the negotiation final
question) to say the same thing explicitly. I documented this as a new,
disclosed model-specific adaptation, analogous in *kind* to the paper's own
documented Gemini-Pro-specific prompt tweak (Appendix 3) — the paper had to
adapt its prompt per model family too; this is the same category of fix,
just addressing a different failure mode (verbosity) for a different model
family (Llama vs. Gemini).

**Implementation:** updated `llm_client.py` (system message applied by
default in `get_order_decision`; added an opt-in `strict_format` flag to
`chat()` so mid-negotiation turns can stay conversational while only the
final-answer turn enforces strict output), updated `mock_llm.py`'s
interface to match, and updated `negotiation.py`'s final-decision node to
request strict formatting specifically for that call.

**Retest:** reran the same 2-step smoke test — completed cleanly, all 25
configs produced results with real API calls. One configuration
(`info_sharing_tool`/large/cost) showed an anomalous 514-second runtime
against ~7-10 seconds for its closest siblings — flagged for follow-up
rather than ignored.

---

## Phase 11 — Initial (overly pessimistic) time/cost extrapolation

Computed per-config timings from the smoke-test JSON and linearly
extrapolated to the full 200-step, 25-config grid: total ≈ 934 seconds for
2 steps × 25 configs, extrapolating to ≈ 26 hours for the full grid,
including that 514-second outlier at face value.

---

## Phase 12 — Investigating the outlier before trusting the estimate

Rather than accept the pessimistic 26-hour number at face value, I isolated
just the anomalous configuration (`info_sharing_tool`/large/cost) and reran
it alone, in the background, using the exact same code path
(`run_single_experiment`) as the full grid.

**Result:** 9.7 seconds — consistent with its sibling configs, confirming
the earlier 514 seconds was a one-off network/API hiccup, not a systemic
bug. Recomputed the extrapolation with the corrected number: ≈ 430 seconds
for 2 steps × 25 configs, extrapolating to **≈ 12 hours** for the full
200-step grid, and ≈ 24,000 total LLM calls, with the negotiation
framework's inherently chatty structure (≈16 sequential calls per
timestep, matching the paper's own `num_iter = 3` spec) identified as the
dominant cost, not a flaw in the implementation.

*Rationale for checking before reporting:* reporting a 26-hour estimate to
you when a large chunk of it was a network fluke would have led to a worse
decision (e.g. cutting scope more than necessary) than the corrected,
more accurate 12-hour estimate.

---

## Phase 13 — First real decision point: how to handle ~12h / ~24k calls

Presented three options (run faithfully with parallelization, reduce
steps as a flagged scope cut, or check pricing first) via a structured
question rather than picking one myself, since this is fundamentally a
time/cost tradeoff only you can weigh. **You chose: check pricing first.**

---

## Phase 14 — Live pricing research and reframing the tradeoff

Fetched OpenRouter's live pricing pages for both models (`WebFetch`, no API
spend involved — this doesn't touch your key at all):
- Llama 3.1 8B: $0.02 / $0.04 per million input/output tokens
- Llama 3.1 70B: $0.40 / $0.40 per million input/output tokens

Estimated total cost for the full grid (~24,000 calls, ~500-600 input
tokens and ~40-60 output tokens per call on average, roughly split between
the two tiers): **~$3-5**, with an explicit caveat that this is an estimate
with real uncertainty, not a guarantee, and a rough safety margin noted
(up to ~$10 accounting for occasional malformed-output retries).

**Key reframing:** the real constraint isn't money (a few dollars is
negligible) — it's the ~12 hours of wall-clock time, which is entirely a
function of the negotiation framework's sequential call-chaining and can't
be sped up per-config, only parallelized *across* configs. I surfaced this
explicitly rather than letting a scary-sounding "12 hours" and "24,000
calls" imply a scary dollar cost that wasn't actually true.

---

## Phase 15 — Your decision: pilot first, then parallelize

You said: "Run the pilot first as planned. Once that's clean, parallelize
across configs for the full run." Two things happened concurrently:

**15a. Launched the actual 20-step pilot** (the one originally planned back
in Phase 8, distinct from the 2-step smoke tests) across all 25 configs,
with output tee'd to both the terminal and `results/pilot_log.txt`, run in
the background given the ~70-80 minute expected runtime (extrapolated from
smoke-test timings).

**15b. While that ran, built the parallelization infrastructure**, so it
would be ready the instant the pilot was confirmed clean, rather than
waiting idle:
- Added `build_full_grid()`, `filter_grid()` (supports filtering the
  25-config grid by metric / framework / model tier / whether to include
  baselines), and `merge_result_files()` to `experiment_runner.py`. This is
  safe to parallelize because every config builds its own environment and
  its own LLM clients from scratch — no shared mutable state between
  configs — and the demand series is regenerated identically in every
  process because it comes from a fixed random seed, not shared runtime
  state.
- Added matching CLI flags (`--only-metric`, `--only-framework`,
  `--only-model-tier`, `--skip-baselines`, `--merge`) to `run_experiments.py`.
- **Tested this new logic immediately**, using the free mock LLM (three
  quick runs: two differently-filtered shards, then a merge of both),
  confirming filtering isolated exactly the intended configs and that
  merging correctly recombined two shard files into one. Deleted the test
  artifacts afterward.
- While waiting on the pilot, peeked at `results/pilot_log.txt` and found
  it empty despite the process having run for a while. Diagnosed this as
  Python's default block-buffering when stdout is piped through `tee`
  (rather than being an interactive terminal) — combined with the fact
  that the pilot had been launched *before* I added `flush=True` to the
  print statements in `run_full_grid()` (that change came slightly later,
  while building the sharding code). Concluded this only affected my
  ability to peek at *interim* progress, not the correctness of the final
  output, and didn't try to intervene in the already-running process.

---

## Phase 16 — Pilot results: partial success, then a real blocker

The pilot completed (notified automatically). Reading its output:

- Both non-LLM baselines completed: Restocking Policy (cost=13,403,
  bullwhip=4.12), Demand Forecasting Tool (cost=803, bullwhip=0.15).
- One full LLM configuration completed end-to-end: `standalone`/8B/cost —
  cost=5,941, bullwhip=1.10, 80.8 seconds for 20 steps (≈4 seconds/step),
  which lined up with the smoke-test-derived timing model — a useful
  cross-check that the earlier extrapolation was sound.
- The **next** configuration (`info_sharing`/small/cost) failed with:
  `402 Client Error: Payment Required for url: https://openrouter.ai/api/v1/chat/completions`

**Diagnosis:** this is OpenRouter reporting the account/key is out of
credit — a billing state, not a code defect. (I also noticed, incidentally,
that piping the run through `tee` meant the overall shell pipeline reported
exit code 0 even though the Python process itself had raised an unhandled
exception — a shell-pipeline mechanics detail worth knowing, not a bug in
the experiment code.)

**What I did:** reported the exact error, what had and hadn't completed,
and explicitly said I would not retry any paid calls until you confirmed
credit was added — rather than silently retrying against a key that would
just fail again.

---

## Phase 17 — "Isn't there a free option"

Researched two candidate free paths before answering, rather than
speculating:

- **Google AI Studio / Gemini free tier:** confirmed genuinely free
  (no credit card required), but also confirmed **Gemini 1.5 Flash and
  1.5 Pro — the paper's actual models — are now fully retired** from
  Google's API, and that Google moved all Pro-tier models to paid-only
  this year, leaving only Flash-class models on the free tier. This
  matters because it means going back to Gemini wouldn't even restore a
  real small-vs-large comparison anymore — there's no free "Pro" to pair
  with a free "Flash."
- **OpenRouter's `:free` model tier:** confirmed real, but rate-limited
  rather than credit-limited (20 requests/minute, and only 50-1,000
  requests per day, with the 1,000/day tier requiring a prior one-time
  purchase). Also found that Llama 3.1's free-tier listings were reportedly
  delisted in early August in favor of Llama 3.3 variants — meaning even
  the model identity isn't stable on the free path.

**Calculation:** even at the best-case free-tier cap (1,000 requests/day),
the ~24,000-call full grid would take **~24 days**, versus hours and ~$3-5
on paid credits. I presented this explicitly as "free trades a small amount
of money for a large amount of calendar time, and even changes which model
you're using," rather than assuming "free" was self-evidently the better
choice, and asked how you wanted to proceed given that reframing.

---

## Phase 18 — Your idea: local Ollama, no external API calls

You asked whether local Ollama was viable. Rather than answer from the
earlier, incomplete GPU check (Phase 2 had only confirmed *no NVIDIA GPU*
via `nvidia-smi`, not full feasibility), I ran a proper hardware check:

- CPU: 8-core Intel Core Ultra 7 256V
- RAM: 15.6 GB total
- Free disk: ~43.5 GB
- GPU: integrated Intel Arc 140V (no NVIDIA)

**Conclusion:** Llama 3.1 8B (quantized) fits comfortably in 15.6GB RAM and
would run locally via Ollama for free. **Llama 3.1 70B needs roughly 40GB+
RAM even quantized and simply cannot be loaded into memory on this
machine** — this is a hard capacity ceiling, not something more time or
patience fixes.

Also checked whether Ollama could even be installed without needing
interactive GUI steps: confirmed `winget` (Windows's package manager) is
available and working (v1.29.290), so `winget install Ollama.Ollama` would
work non-interactively if you chose to proceed with local inference.

**What I did with this:** rather than pick a workaround myself (e.g.
silently swapping in a smaller "large" model), I presented the tradeoff —
hybrid (8B local free + 70B paid API, preserves the real size gap), both
tiers local with a smaller substitute (~13-14B model, free but a much
smaller and less meaningful size gap than the paper's actual Flash-vs-Pro
comparison), or dropping the large tier entirely (free, simplest, but
drops one of the paper's two research questions) — and asked which you
wanted.

---

## Phase 19 — Your request: comprehensive report + GitHub commit

You asked for a full report with professional options (e.g. better
hardware, OpenRouter payment) and rationale for each, plus a GitHub commit.
Before writing or committing anything:

- **Searched the entire project directory for the literal API key string**
  (`grep -rl "sk-or-v1" .`) to make absolutely sure it had never been
  accidentally written to a file (it hadn't — the key had only ever been
  passed inline via `export` in individual Bash commands, never persisted
  to disk). This was a deliberate safety check *before* touching git, since
  committing a secret to a repository — even a private one — is much
  harder to fully undo than not committing it in the first place.
- Checked git identity config (already set: `SarrangTech` /
  `SarrangTech@users.noreply.github.com`) and whether `gh` (GitHub CLI) was
  available (it wasn't) and whether a git repo already existed in the
  project folder (it didn't).
- Wrote `REPORT.md`: executive summary, what was researched, what was
  built (file by file), what validation was performed and its results,
  current state, and a 7-option decision table (add OpenRouter credit;
  hybrid local-8B/paid-70B; both-local with a smaller substitute; skip the
  large tier; free hosted tiers; rent/acquire better hardware; reduce
  experimental scope), each with cost, time-to-finish, what you keep, and
  what you give up — plus a closing personal assessment of which options
  are most cost-effective and why.
- Wrote `.gitignore` (excluding `__pycache__`, `.env`, `*.key` — belt and
  suspenders on top of the manual key-leak check already done).
- Ran `git init`, staged everything, and **checked `git status` before
  committing** to visually confirm nothing unexpected (like a stray file
  containing the key) had been staged.
- Committed with a descriptive message summarizing the whole build.
- Asked you two clarifying questions rather than assuming: how to handle
  the GitHub push (you create an empty repo and give me the URL, vs. I
  install `gh` and you authenticate it yourself interactively — I
  deliberately avoided reading your existing `.git-credentials` file
  myself to extract a token, since that's your stored secret to control,
  not mine to use without being asked), and public vs. private visibility.
  **You chose: you'd create the repo and share the URL; private.**
- You provided the URL
  (`https://github.com/SarrangTech/supply-chain-llm-consensus`); I added it
  as the `origin` remote, renamed the local branch to `main` to match
  GitHub's default, and pushed — succeeded, using your existing git
  credential helper, without me ever needing to see a GitHub token.
- Reported back a concise state summary: what's done vs. what's still
  blocked on your decision.

---

## Phase 20 — Writing this document

You said the report wasn't comprehensive enough and asked for every step
with rationale. This file (up to this point) was that full account, phase
by phase, in the order it actually happened. `REPORT.md` remains the
shorter executive-summary-plus-decision-table version for quick reference;
`NOTES_AND_ASSUMPTIONS.md` remains the focused list of paper-specific
deviations and judgment calls (agent indexing, simulation ordering, the
Chen et al. baseline's exact formula, etc.) without the full narrative.
This file is the connective narrative across all three.

---

## Phase 21 — Local Ollama for the small tier (2026-08-26)

You asked to run the small model locally via Ollama while waiting for the
university to provide more compute for the large tier. Steps taken:

- Rechecked free disk space (~25.8GB free by this point, down from ~43.5GB
  earlier in the session due to unrelated activity on the machine) —
  still comfortably enough for an ~5GB model.
- Installed Ollama non-interactively via `winget install --id Ollama.Ollama
  --silent`, run in the background since the download/install took longer
  than a few minutes.
- Verified the install by locating `ollama.exe` directly
  (`C:\Users\batha\AppData\Local\Programs\Ollama\`) after discovering the
  new PATH entry hadn't propagated to the current shell session yet —
  used the full binary path rather than waiting on a shell restart.
- Pulled `llama3.1:8b` (~4.9GB), run in the background.
- **Refactored `src/llm_client.py`** rather than writing a second,
  duplicate client: extracted the shared retry/malformed-output-parsing
  logic (which must behave identically regardless of backend) into a
  `BaseChatClient` base class, then made both `LLMClient` (OpenRouter) and
  a new `OllamaClient` (local, talks to `http://localhost:11434/api/chat`)
  thin subclasses that only implement `_call_raw()`. This avoids having
  two copies of the retry logic drift apart over time.
- **Updated `src/config.py`** to add a `BACKENDS` dict (`{"small":
  "ollama", "large": "openrouter"}`) as the single place controlling which
  tier uses which backend, plus `OLLAMA_MODELS` (Ollama's model-name
  strings differ from OpenRouter's) and `OLLAMA_BASE_URL`. Documented this
  as a further, dated deviation on top of the existing model-substitution
  note, rather than editing the original note in place and losing the
  history of *when* this changed.
- **Updated `experiment_runner.py`'s `make_client_factory`** to read
  `BACKENDS` and instantiate the right client per tier.
- **Fixed a real CLI bug this change surfaced**: `run_experiments.py` was
  unconditionally requiring `OPENROUTER_API_KEY` for any non-mock run, even
  though a small-tier-only shard now needs no API key at all (it's fully
  local). Reordered the CLI so the grid is built and filtered *first*,
  then the API-key check only fires if the resulting (possibly filtered)
  grid actually contains a config routed to OpenRouter.
- **Smoke-tested the new Ollama backend for real** (2 steps, all 10
  small-tier LLM configs plus baselines, no API key set) before trusting
  it — completed cleanly, exit code 0.
- **Found a real, reportable timing result from that smoke test**: four of
  the five frameworks extrapolate to a reasonable ~1-2.5 hours per 200-step
  config locally, but negotiation extrapolates to ~13.5 hours per config
  (~27 hours for both metrics combined) — a ~6-7x slowdown versus the
  hosted API's ~2.1 hours per config, because negotiation's heavy
  sequential-call structure has nowhere to go but through a single CPU
  without a GPU behind it. Reported this precisely (with the per-framework
  breakdown table) rather than a single blended "it's slow" estimate,
  and asked how you wanted to handle negotiation specifically given that
  gap. **You chose to run it locally anyway and accept ~27 hours in the
  background.**
- Before launching the long run, flagged a practical caveat: since this
  now runs on local CPU rather than a hosted API, the laptop needs to
  stay awake for the run to keep progressing (a hosted-API run never had
  this constraint).
- **Sequenced the run to surface results sooner**: rather than run the
  full small-tier grid in its default order (which interleaves a slow
  negotiation config in the middle before the last metric's fast
  frameworks even start), launched the four fast frameworks (both metrics)
  as one shard first (~15h), planning to run negotiation (both metrics)
  as a second shard afterward (~27h) — considered running them
  concurrently instead, but rejected that: unlike the hosted API (where
  OpenRouter has independent server capacity per request), local
  inference through one Ollama process on one CPU doesn't actually
  parallelize throughput across concurrent requests, so running shards
  "in parallel" locally would mostly just add contention rather than save
  wall-clock time.
- Updated `NOTES_AND_ASSUMPTIONS.md` with a dated addendum describing this
  change and the negotiation timing finding.
