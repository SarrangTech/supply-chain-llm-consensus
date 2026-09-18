# Notes, deviations, and judgment calls

This replication is reconstructed from the paper's text only. Read this
before trusting any number that comes out of `run_experiments.py`.

## (a) Repo search performed (Step 0)

Searched for, and did **not find**, a public repository for either:
- Jannelli, Schoepf, Bickel, Netland & Brintrup (2025), "Agentic LLMs in the
  supply chain" (arXiv:2411.10184 / IJPR 10.1080/00207543.2025.2604311). The
  paper states "we open-source our code" but no link appears on arXiv, the
  IJPR page, or in any search result.
- Liu, Hu, Peng & Yang (2022), "Multi-Agent Deep Reinforcement Learning for
  Multi-Echelon Inventory Management" (Rotman Working Paper 4262186 / SSRN
  4262186), the base environment this paper extends. No linked code found.

Searches covered: paper title + "github", arXiv ID, author names + SCAIL,
"HAPPO multi-echelon inventory github", and variations. One superficially
similar repo (`zefang-liu/InvAgent`) belongs to a *different* paper (Quan &
Liu 2024) that Jannelli et al. cite as related work, not their own code or
the Liu et al. (2022) base environment. Everything in this codebase is a
from-scratch reconstruction of the paper's stated methodology.

## (b) Explicit deviation from the paper (user-approved)

**Update, 2026-08-26**: while waiting for more compute from the user's
university (needed for the 70B/"large" tier -- see hardware check in
`REPORT.md`/`FULL_SESSION_LOG.md` Phase 18), the "small"/8B tier now runs
**locally via Ollama** instead of OpenRouter, at the user's request. This is
free and fits this machine's 15.6GB RAM. The routing is controlled entirely
by `config.BACKENDS` (`{"small": "ollama", "large": "openrouter"}`) --
nothing else in the codebase needs to know or care which backend a given
tier uses, `src/llm_client.py`'s `LLMClient` (OpenRouter) and `OllamaClient`
(local) share the same retry/parsing logic via a `BaseChatClient` base
class.

One new, real finding from this: **the negotiation framework is
dramatically slower on local CPU inference than on the hosted API** --
roughly 13.5 hours per 200-step config locally vs. roughly 2.1 hours per
config on OpenRouter (a ~6-7x slowdown), because negotiation's ~16
sequential calls/step compound badly without a GPU behind them, whereas
the other four frameworks are only mildly slower locally (roughly 1-2.5
hours per 200-step config either way). This is a hardware/inference-speed
finding, not a new methodological deviation -- the frameworks themselves
are unchanged.


**Models**: the paper uses Gemini 1.5 Flash and Gemini 1.5 Pro directly via
Google's API. No Gemini/Google API key was available in this environment.
Per explicit user instruction, this codebase instead uses **Llama 3.1 8B
Instruct** and **Llama 3.1 70B Instruct** via **OpenRouter**, as structural
analogs of the paper's small/fast vs. large/capable model comparison.

**Consequence**: no number this codebase produces should be compared
directly to the paper's Table 1 / Table 2 values as if it were the same
experiment. The comparison that *is* valid is structural: does the same
ordering of framework sophistication (standalone < info-sharing < tool-
assisted < negotiation) produce the same qualitative pattern of improvement
that the paper reports, using a different but comparably-tiered model pair.

Model availability was not re-verified against OpenRouter's live catalog as
of a specific date; verify `meta-llama/llama-3.1-8b-instruct` and
`meta-llama/llama-3.1-70b-instruct` are still served before running for real
(`GET https://openrouter.ai/api/v1/models`).

## (c) Unconfirmed parameter: Merton Jump Diffusion demand

The paper states demand is "Merton Jump Diffusion Model (run number 13)"
(Appendix 6) and shows only a plot (Figure 17): starts ~9-12, decays to ~1-2
by step ~50, low-variance noise afterward with small blips to ~3 around
steps 130-140, bounded to [0, 20], over 200 steps.

No exact parameters (drift, volatility, jump intensity/mean/std, seed) were
published or recoverable from a source repo. `src/demand.py` implements an
OU-mean-reversion-plus-jumps process, hand-tuned only to visually
approximate Figure 17's shape, with every parameter named and exposed in
`MJDParams`. This is a **placeholder approximation**, explicitly not a
reproduction of the original series. Per user instruction, this is flagged
here rather than silently presented as exact. Swap `MJDParams` the moment
real values are available -- nothing downstream cares where demand came
from.

Also note: the *structural form* used (OU + jumps) differs from a textbook
multiplicative Merton Jump Diffusion (geometric Brownian motion + compound
Poisson jumps), because a pure multiplicative process cannot decay from ~10
to ~2 and then stabilize at a small positive baseline without either
collapsing to 0 or exploding. This is a documented judgment call, not an
oversight.

## (d) Judgment calls made resolving ambiguity in the paper's text

1. **Agent indexing / directionality**: agent 0 = Retailer (faces customer
   demand), agent 1 = Distributor, agent 2 = Manufacturer (most upstream,
   assumed unconstrained supply capacity). Inferred from Figures 9-11
   (EOQ grows from agent 0 to agent 2) and the "upstream agents amplify
   demand" statement (Section 5). Not stated as explicit indices in the paper.

2. **Within-timestep simulation order** (`environment.py`): receive
   shipments -> fulfill downstream demand (backlog-first) -> place new
   orders -> compute costs -> orders arrive after `lead_time` steps. The
   paper gives the cost decomposition and bullwhip formula but no formal
   state-transition equation; this is the standard multi-echelon/beer-game
   convention, not something extracted from an unseen original.

3. **(S,s) baseline uses raw on-hand inventory**, not inventory position
   (on-hand - backlog + pipeline), per the literal wording "reorders to a
   maximum level (S) when inventory falls below a threshold (s)" (Section
   5.1). A textbook implementation might use inventory position instead;
   the paper doesn't specify which.

4. **Chen et al. (2000) baseline** (`baselines.py`): the paper describes
   this only as "a moving average forecast and order-up-to policy" using
   demand "centralised" from the retailer. Implemented as: order-up-to
   level = (moving-average forecast of retailer demand, 30-period lookback)
   x (lead_time + 1), no explicit safety-stock/z-score term (none is given
   in the paper), ordering against each agent's own inventory position. This
   is a standard simplification of the cited model, not the paper's exact
   (unpublished) implementation.

5. **Per-pair EOQ / demand-forecast tool computation** (Section 6.2.1 claims
   each agent computes a *different* EOQ per bilateral negotiation,
   depending on which neighbour it's negotiating with). This codebase
   computes each agent's tool output **once per timestep**, from its own
   `demand_history` (what it itself receives from its downstream neighbour),
   and reuses that single value in whichever pairwise negotiation it takes
   part in during that step. This was chosen because the paper's own worked
   example (Figures 9-11) is internally inconsistent with its own textual
   claim (Agent 1's EOQ is shown as the same value, 8, in both its
   negotiation with Agent 0 and its negotiation with Agent 2, despite the
   text saying these should use different demand data) -- there is no way to
   resolve this without the original code, so the simpler, internally
   consistent rule was chosen and documented rather than guessed at silently.

6. **Reconciling a middle agent's two pairwise decisions** in the
   information-sharing and negotiation frameworks (agent 1 takes part in
   pair (0,1) AND pair (1,2) each step): pairs are processed strictly
   downstream-to-upstream, and an agent's LAST-computed pairwise decision in
   the step is authoritative. The paper says only that "the final decision
   stage reconciles these negotiations" (Section 6.2.1) without giving the
   rule.

7. **Flash-vs-Pro prompt divergence** (Appendix 3, Figure 14): the paper adds
   "Use it as an upper bound for the final output" only for the larger model.
   This codebase applies that same extra instruction to the "large" model
   tier (Llama 3.1 70B, standing in for Gemini 1.5 Pro) and not to "small"
   (Llama 3.1 8B / Flash-analog), mirroring the paper's documented tweak
   structurally, not verifying it has the same effect on a different model
   family.

8. **Negotiation conversation intro** (Figure 9): the opening line
   ("Let's have a conversation about how much to order...") is treated as
   genuine LLM output from the downstream agent (seeded by a fixed
   instruction prompt), not a hard-coded template, consistent with the
   paper's framing of "frameworks" as orchestration patterns around LLM
   decisions rather than scripts.

9. **Starting inventory** (`AgentState.inventory = 20` default): not stated
   in the paper (Appendix 6's Table 2 lists cost/lead-time/order-limit
   parameters but no initial inventory level). Chosen arbitrarily; flagged
   as unconfirmed.

## (e) What was NOT changed / added beyond the paper's scope

- No 6th framework, no extra metric, no extra baseline was added.
- No attempt to "improve" the frameworks' logic beyond what Section 4
  describes.
- The LangGraph orchestration structure (Figures 6 and 7) is followed
  directly: `info_sharing -> final_decision -> summarise` for frameworks
  (b)/(d); `info_sharing -> {initiate <-> respond} x 3 -> final_decision ->
  summarise` for framework (e).

## (f) Migration to HPC (explorer.northeastern.edu), 2026-08-29/30

Compute moved from the user's laptop to Northeastern's Discovery cluster
("Explorer"), reached via SSH as `explorer-login` (login node, itself named
`explorer-02`) with SLURM for job scheduling. This section documents what
was found and changed; it supersedes the "waiting for more compute"
framing in section (b) above -- the wait is over, both tiers now run free
and local, and (b) is left as-is for the historical record of *why* Ollama
was introduced in the first place.

**Environment setup.** System Python (3.9) had no `pip`; bootstrapped via
`ensurepip`, then built a venv against the cluster's newer `module load
python/3.13.5` instead. Ollama has no system-wide install available
(installer needs root, which isn't available here) -- used the portable
tarball instead (`ollama-linux-amd64.tar.zst`, requires `zstd`; note
Ollama switched from `.tgz` to `.tar.zst` at some point after older
install instructions were written, so a plain `curl | tar -xz` silently
downloads an HTML 404 page and fails opaquely -- fetch the real asset URL
from `api.github.com/repos/ollama/ollama/releases/latest` instead). Runs
as a plain foreground process (`ollama serve &` inside the SLURM job
script), no daemon/systemd needed.

**Proxy bug (real, non-obvious, cost the most debugging time).** Compute
nodes have `http_proxy`/`https_proxy` set campus-wide with no `no_proxy`
exemption for `localhost` -- so Ollama's own client-server traffic on
`127.0.0.1:11434` was getting silently routed through the campus Squid
proxy and rejected ("Access Denied"), even though client and server are
the same process' loopback call. Fix: export `no_proxy`/`NO_PROXY`
including `localhost,127.0.0.1` in every job script before starting
`ollama serve`.

**GPU driver ceiling.** Ollama's CUDA path requires driver >= 550. Several
older GPU nodes (e.g. V100-SXM2 nodes, driver 545) fall below that and
Ollama silently falls back to a Vulkan compute backend instead -- still
genuinely GPU-accelerated (confirmed via `ollama ps` showing `100% GPU`),
just not the CUDA path. Newer nodes (H200, driver 595+) use real CUDA.
Don't assume Vulkan fallback = broken; check `ollama ps`'s PROCESSOR column
and the "NVIDIA driver too old" log line, not just whether it runs.

**Thread-oversubscription bug (the actual root cause of "CPU is too
slow").** Ollama/llama.cpp auto-sizes its thread pool to the node's full
logical CPU count (`nproc`, i.e. including SMT/hyperthreads) by default --
it has no awareness of SLURM's cgroup-based core allocation for the job.
Requesting `--cpus-per-task=48` on a 128-thread node still spawned 128
llama.cpp threads, all fighting over the 48 cores the cgroup actually
grants, causing severe barrier-contention stalls in llama.cpp's per-layer
thread synchronization. Measured effect on 8B: **0.02-0.39 tok/s** with
default threading vs **12.34 tok/s** once the request explicitly pinned
`num_thread` to the job's real core count. This is now fixed in code, not
just in ad-hoc test scripts -- see `llm_client.py`'s `_NUM_THREAD` (reads
`SLURM_CPUS_PER_TASK`, falls back to `os.cpu_count()`) and its use in
`OllamaClient._call_raw`'s `options.num_thread`. There is no such thing as
an `OLLAMA_NUM_THREADS` environment variable (an earlier test script
assumed one; it's silently ignored) -- thread count is a per-request API
option, not an env var.

**CPU instruction-set gap (bigger effect than thread count alone).**
AVX512 support matters enormously for llama.cpp's quantized matmuls. Same
8B model, same (correct) thread pinning:

| Node CPU | ISA | tok/s (8B, sustained) |
|---|---|---|
| AMD EPYC 7702 (Zen 2) | AVX2 only | 0.39 |
| Intel Cascade Lake (Xeon) | AVX512 + VNNI | 12.34 |

SLURM exposes CPU generation as node features (`sinfo -N -o "%N %f"`);
target AVX512-capable nodes explicitly via `--constraint=cascadelake` (or
`skylake_avx512` / `sapphirerapids`) rather than letting the scheduler pick
any CPU node in the partition.

**70B does not work on CPU, full stop.** Even on an AVX512 Cascade Lake
node with correct thread pinning: **~0.04 tok/s (~25 sec/token)** -- a
single 90-token reply would take ~37 minutes, and a full negotiation
config (~3200 calls) would take on the order of 100+ days. This is a hard
memory-bandwidth wall (70B's ~40GB of quantized weights have to stream
through memory once per generated token), not something thread/ISA tuning
fixes. 70B needs GPU. Confirmed working there: H200 (143GB VRAM), full
81/81-layer CUDA offload, `ollama ps` reports `100% GPU`.

**Context-size bloat.** Ollama defaults `num_ctx` to the model's full
trained context (131072 for Llama 3.1) when unset, which on the 70B/H200
test allocated a ~40GB KV cache and made the very first prompt eval take
19 seconds for 17 tokens. Our prompts are short (well under 4096 tokens);
`OllamaClient` now explicitly caps `num_ctx=4096` on every call to avoid
this on every backend, not just GPU.

**Resulting backend decision (`config.BACKENDS`, both updated to
`"ollama"`):** "small" (8B) runs preferentially on abundant AVX512 CPU
nodes in the `short` partition (2-day walltime, no GPU queue contention,
~40+ such nodes) but works fine on GPU too if one's free. "large" (70B)
is GPU-only (`gpu` partition, 8h walltime cap, or `sharing`, 1h cap) --
CPU is not an option for this tier at all, per the measurement above.

**Resolved (2026-08-31):** does a full 200-step negotiation_tool/large
config finish within the `gpu` partition's 8-hour cap? Yes, comfortably.
A real `--steps 20` pilot (`negotiation_tool` x `large` x `cost`, on an
H200 via the `gpu` partition, job 9835369) completed in **671.9s**.
Per-step call count is fixed regardless of total step count, so this
scales ~linearly to **~1.9 hours** for the full 200 steps -- against an
8h cap, no checkpoint/resume or OpenRouter fallback needed for this
combination after all. The earlier worry was based on isolated single-call
cold-start latency (misleadingly slow); real sustained per-call time on
H200 was ~2.2s once warmed up, i.e. the bottleneck was never GPU throughput.

**Port-collision bug found during the 20-shard pilot grid (2026-08-31).**
Submitted 8 large-tier (GPU) shards concurrently, each requesting a single
GPU via `--gres=gpu:h200:1`. SLURM's `gpu` partition nodes have 8 H200s
each, so multiple of our jobs can (and did) land on the *same physical
node* -- confirmed via `sacct -o NodeList`: jobs 9836200/9836202 both ran
on `d4052` starting at the identical second, and separately
9836370/9836378 also both on `d4052`. Every job script hardcoded `ollama
serve` on the default port 11434; when two of our jobs share a node, one
binds the port and the other's `ollama serve` silently fails to bind
(already in use) -- its client then unknowingly piggybacks on the
sibling's server for the rest of the run. This is invisible right up until
the owning job finishes first and kills "its" server: the piggybacking job
then gets `Connection refused` mid-run (exactly what killed
`info_sharing`/large/cost, job 9836200 -- real, unrecoverable data loss for
that shard) or, if it finishes before its sibling does, completes with
fully valid results but still exits non-zero (the script's final `kill
$SERVER_PID` fails against a PID that was never actually its own server),
which SLURM then reports as FAILED even though the results are good (job
9836378, `negotiation_tool`/large/bullwhip -- verified: full result tables
present in both the log and `results/pilot_negotiation_tool_large_bullwhip.json`).
Net effect across 8 concurrent GPU shards: 1 genuine loss, 1 cosmetic
false-failure, 6 unaffected by luck of timing. Not survivable at full
200-step scale (longer overlap windows, more concurrent shards). Fixed by
making `OLLAMA_BASE_URL` overridable via env var (`config.py`) so each
SLURM job script picks a port derived from `$SLURM_JOB_ID` and exports
`OLLAMA_HOST`/`OLLAMA_BASE_URL` to match, plus guarding the cleanup
(`kill $SERVER_PID 2>/dev/null || true`) so a piggybacking job's harmless
kill failure no longer flips its exit status.

**Noisy-neighbor CPU contention (2026-08-31, second bug found the same
day).** Two of the ten small-tier (CPU) pilot shards
(`standalone_tool`/bullwhip, `info_sharing_tool`/cost) silently ran ~20x
slower than their siblings and hit the 45-minute job time limit with no
results saved, while shards using the identical framework/model on
different nodes finished in ~2 minutes. Checked the slow node's own log
first in case it was a hardware/driver difference: `system_info` still
reported `AVX512 = 1, AVX512_VNNI = 1` -- same capability as the healthy
nodes. So the CPU generation wasn't the problem; the shared/non-exclusive
`short` partition let other users' unrelated jobs land on the same
physical node and contend for the same physical cores our job's 56 pinned
threads were relying on (SLURM's default scheduling counts logical, not
physical, cores, so a "fully allocated" node on paper can still be
oversubscribed underneath). Fix: added `--exclusive` to small-tier job
submissions (`submit_shard.sh`) so the whole node is reserved for us,
trading a possibly-longer queue wait for guaranteed throughput. Confirmed
fixed: rerunning both affected shards with `--exclusive` completed in
2m09s and 2m44s respectively -- back in line with every other small-tier
shard.

**20-shard pilot grid, full results (steps=20, pipeline validation only --
NOT meaningful numbers, see the 200-step requirement in FIXED_PARAMS).**
All 5 frameworks x 2 model tiers x 2 metrics, run for real against local
Ollama backends on the cluster (no mocks, no OpenRouter). Two shards
needed a rerun after the bugs above were fixed (`info_sharing`/large/cost:
port collision: SLURM job 9836200 -> rerun 9851695; `standalone_tool`/
small/bullwhip and `info_sharing_tool`/small/cost: noisy neighbor: SLURM
jobs 9836205, 9836207 -> reruns 9852815, 9852816). Full job-ID provenance
is in `pilot_job_ids.txt`; raw output in `results/pilot_*.json`.

```
framework           tier   metric          cost    bullwhip elapsed_s
info_sharing        large  bullwhip       978.0     0.01650     177.5
info_sharing        large  cost          1800.0     0.00629      44.6
info_sharing        small  bullwhip      3313.0     0.18819     149.3
info_sharing        small  cost         13190.0     0.05023     151.0
info_sharing_tool   large  bullwhip      1185.0     0.06536      49.3
info_sharing_tool   large  cost          1517.0     0.16660      46.7
info_sharing_tool   small  bullwhip      9027.0     0.22062     198.6
info_sharing_tool   small  cost         15107.0     0.03143     152.2
negotiation_tool    large  bullwhip      8036.0     0.83492    1128.8
negotiation_tool    large  cost          1580.0     0.29936    1129.0
negotiation_tool    small  bullwhip      2300.0     3.45330    1742.8
negotiation_tool    small  cost          5265.0     0.74875    1755.1
standalone          large  bullwhip      1096.0     0.01279     180.2
standalone          large  cost          1425.0     0.01555     155.5
standalone          small  bullwhip      6536.0     0.15820     122.9
standalone          small  cost         10769.0     0.03544     127.5
standalone_tool     large  bullwhip      1003.0     0.12002      38.9
standalone_tool     large  cost          1607.0     0.22431      37.7
standalone_tool     small  bullwhip      3175.0     3.48097     117.1
standalone_tool     small  cost         12377.0     0.04715     131.2
```

Do not read anything into the specific cost/bullwhip values above --
20 steps is a pipeline smoke test, not the paper's 200-step spec, and
several rows (e.g. `negotiation_tool`/small/bullwhip at 3.45) show the
kind of instability a short warm-up window produces. The only claims this
table supports are: every framework/tier/metric combination runs to
completion against real local backends, and elapsed time per shard is
consistent with the throughput numbers measured earlier in this section.

**Sharding across the cluster.** `run_experiments.py` already supports
`--only-framework` / `--only-metric` / `--only-model-tier` plus `--merge`
(see its docstring and `filter_grid()` in `experiment_runner.py`) -- no
code changes were needed to parallelize. The 25-config grid decomposes
into 20 independent LLM-driven runs (5 frameworks x 2 tiers x 2 metrics,
each metric a fully independent 200-step simulation, not just a different
scoring of shared data) plus 5 fast non-LLM baselines. Each of the 20 can
be submitted as its own SLURM job against whichever node type suits its
tier, running in parallel rather than the ~13.5h/config sequential
estimate from section (b) compounding across the whole grid.

## (g) Full 200-step, 25-configuration grid (2026-09-02) -- the paper's actual spec

Ran for real using `submit_shard.sh`, all 20 LLM-driven shards in parallel
across the cluster (job IDs in `full_job_ids.txt`), then merged and
deduplicated (baselines are recomputed identically by every shard since
they're deterministic given the fixed demand series, so duplicates were
dropped: 70 raw rows -> 25 canonical rows in `results/results.json`, one
row per Table 1 / Table 2 configuration).

All 20 shards completed cleanly (`sacct` state `COMPLETED`, zero
failures) -- the port-collision and noisy-neighbor fixes from section (f)
held even though the scheduler packed all 10 GPU shards onto the same
physical node (`d4052`) one after another. Elapsed times matched the
scaled-up pilot estimates: `negotiation_tool`/small took 4h55m (estimate
was ~4.8h), `negotiation_tool`/large took 1h34m (estimate was ~1.9h).

**Two patterns in the real numbers worth flagging, not yet diagnosed:**

1. The 8B ("small") tier's cost-metric results are enormous (over
   1,000,000 in several configs, vs. 124,450 for the (S,s) baseline),
   while the *bullwhip* values for those same configs are low (0.03-0.66).
   Low bullwhip alongside catastrophic cost is consistent with the model
   settling into ordering a large, roughly constant amount every step
   (low order-to-order variance -> low bullwhip) instead of reasoning
   about actual inventory need, accumulating holding-cost penalties over
   200 steps instead of oscillating. Plausible, not confirmed -- would
   need to inspect per-step order logs to verify.
2. `negotiation_tool` + large (70B) is the *worst* cost performer among
   all 70B configs (770,118, worse than standalone's 71,651 and both
   tool-assisted variants' 20,038/32,552), which runs counter to the
   paper's core hypothesis that more sophisticated coordination
   monotonically improves outcomes. Could be a genuine finding about how
   Llama 3.1 negotiates differently than Gemini, or an artifact of the
   `MalformedOutputError` retry path being hit more often during
   negotiation's longer multi-turn exchanges -- not yet distinguished.

Do not treat either observation as a validated conclusion about framework
quality -- they are flagged here as the next things worth investigating
(starting with the retry/malformed-output logs for the affected shards)
before drawing any comparison to the paper's Table 1 / Table 2 findings.

## (h) Root cause of the negotiation anomaly, found and fixed (2026-09-05)

Investigated observation (2) from section (g) directly against the paper's
own text (arXiv:2411.10184, pages 15-20). The paper explicitly claims
negotiation should be the *best* framework on both metrics: "for both
models used, the performance of the negotiation framework beats the hard
baseline involving tool-based restocking policy" (cost), and "the best
bullwhip effect performance was achieved with the negotiation framework...
a 66.2% bullwhip effect reduction compared to information sharing with
tool in the case of Gemini Pro" (bullwhip). Our replication showed the
opposite: negotiation was the *worst* 70B config on both metrics.

**Ruled out first:** malformed-output retries (zero occurred -- checked
directly via the retry-warning logging added for this investigation, see
`llm_client.py`'s `RETRY`/`NEGOTIATION_FINAL_FALLBACK` print statements)
and node/hardware contention (per-call latency was normal throughout).

**Actual root cause, confirmed by inspecting real negotiation transcripts**
(via a new `--include-transcripts` diagnostic flag, see below): 62 of 400
negotiation sessions (15.5%) in a real 200-step run produced a final order
clamped to the hard max-order cap (100), completely disconnected from an
otherwise coherent, converging conversation. Example: two agents with
EOQs of 2.31 and 1.15 negotiate sensibly toward "an order quantity of
2.0... a compromise that takes into account both" -- and the recorded
final order is 100.

The reason: `BaseChatClient.chat()` (`llm_client.py`) is a stateless,
single-turn call by design -- it builds a fresh `messages` list on every
invocation, with no conversation history parameter anywhere. Each
negotiation turn's only context is whatever text is manually quoted into
that one prompt (`negotiation_turn_prompt` quotes only the counterpart's
*immediately preceding* message). The final "What is your final answer?"
question (`NEGOTIATION_FINAL_QUESTION`) included **none** of the preceding
conversation -- only the agent's own EOQ. The model was asked to commit to
a number with zero memory of what it had just negotiated, and, asked to
"provide an integer from 0 to 100" with no real basis to decide, seems to
anchor on the boundary value stated in its own instructions rather than a
genuine decision. Order values oscillating between ~2 and 100 repeatedly
is close to a worst case for a bullwhip coefficient-of-variation metric,
and plausibly explains most of the 1.954-vs-0.144 gap on its own.

This is a real implementation gap, not a new deviation from the paper's
design -- the paper's Gemini-based negotiation almost certainly preserved
full conversation memory automatically (a standard property of
conversational chat APIs), so grounding the final question in the
transcript is a correctness fix, not a methodological change.

**Fix applied:** `prompts.py`'s new `negotiation_final_question_prompt()`
renders the full transcript into the final-answer prompt explicitly, and
`negotiation.py`'s `_node_final_decision` uses it (for both the primary
`chat()` call and the `get_order_decision()` fallback) instead of the bare
context-free question. Not yet re-run to confirm the fix's quantitative
effect on the full 200-step grid -- see "How to actually run this" below
for the diagnostic tooling (`--include-histories`, `--include-transcripts`)
used to find this, which stays in the codebase for future use.

**New diagnostic instrumentation added alongside this fix** (off by
default, so normal runs are unaffected):
- `run_experiments.py --include-histories`: attaches
  `per_agent_order_history` / `_inventory_history` / `_backlog_history` to
  each result, so aggregate cost/bullwhip numbers can be traced back to
  actual per-step behavior instead of taken on faith.
- `run_experiments.py --include-transcripts`: attaches the full
  `negotiation_transcripts` (every pairwise negotiation's turn-by-turn
  text, EOQs, and final orders) for `negotiation_tool` configs -- this is
  what surfaced the bug above; the conversation text was otherwise
  computed and immediately discarded.
- `llm_client.py`: prints a `RETRY ...` line to stderr whenever the
  malformed-output retry path fires, and `negotiation.py` prints
  `NEGOTIATION_FINAL_FALLBACK ...` whenever the final-answer chat reply
  doesn't parse -- makes "does Llama break format more than Gemini did"
  measurable instead of theorized (measured: not the cause here, zero
  fired).

### (h.1) Verification results (2026-09-05/07): partial fix, one new bug found and fixed, one deeper issue still open

Reran `negotiation_tool`/large (both metrics, full 200 steps,
`--include-histories --include-transcripts`) after the transcript-grounding
fix above, to check its actual quantitative effect rather than assume it
worked.

**What the fix solved, confirmed:** the diagnosed bug (final answer
completely disconnected from the conversation, clamped to the hard
max-order cap) is essentially gone. Spike-to-100 rate dropped from 15.5%
(62/400 sessions) to 0.2% (1/400). Cost for the bullwhip-metric config
also improved substantially (797,902 -> 150,755).

**A new failure mode the fix itself introduced:** the cost-metric rerun
crashed outright --
`MalformedOutputError: ... No [[N]] pattern found in: '[[sqrt(9.00*7.00) = sqrt(63.00) = 7.94, rounded to 8]]'`.
Grounding the final question in the actual transcript means the model now
sees the conversation's own EOQ-averaging arithmetic, and it started
imitating that shown-work style even in the strict-format answer -- which
the original strict `[[N]]`-only regex rejected, and after `MAX_RETRIES`
consecutive rejections the whole run raised uncaught and died. **Fixed**:
`llm_client.py`'s new `parse_order_answer()` tries the strict pattern
first, then falls back to extracting the *last* number found inside the
brackets (a model restating a calculation states the final/rounded result
last -- "= 7.94, rounded to 8" -> 8, not 7.94 or the 9/7 inputs). Both
`get_order_decision()` and `negotiation.py`'s primary final-answer path
now use this shared helper. Not a new deviation from the paper either --
it's a parser robustness fix for a side effect the grounding fix itself
introduced.

**Deeper issue, still open, NOT fixed by either change above:** bullwhip
for the same config got *worse*, not better (1.946 -> 6.227). Inspecting
the actual order sequences: the catastrophic clamp-to-100 spikes are gone,
but new medium-magnitude spikes appeared instead (e.g. 50, 88, 42, 85, 60
scattered through an otherwise stable ~3-8 baseline). Checked and ruled
out: these are not retry-driven guesses (zero `RETRY`/`NEGOTIATION_FINAL_FALLBACK`
lines in that job's log -- every one of these answers parsed cleanly on
the first attempt). That means even with full transcript grounding and
clean formatting, the model's final decision is still genuinely
inconsistent step-to-step in a way that inflates bullwhip -- closer to
what the paper itself acknowledges as an occasional Gemini behavior
("going for one extreme of the negotiation interval or disagreeing
altogether," Section 6.2.1) than to a bug we introduced. Whether this is
simply more frequent/severe for Llama 3.1 than for Gemini, or points to a
further fixable issue (e.g. the mid-negotiation turns still only ever see
the counterpart's *immediately preceding* message, never the full
transcript, so an agent can lose track of its own earlier proposals) has
not been determined. Next step, not yet done: inspect whether the medium-magnitude spikes
correlate with anything identifiable in the transcripts (e.g. specific
demand-shock periods, or a particular agent role).

**Full before/after picture, both metrics, `negotiation_tool`/large, 200
steps** (cost-metric rerun done after the parser fix too -- confirmed no
crash, `MAX_RETRIES` never exhausted):

| Config | Cost (before) | Cost (after both fixes) | Bullwhip (before) | Bullwhip (after) | Spike-to-100 rate |
|---|---|---|---|---|---|
| cost metric | 770,118 | 333,060 | 0.299 | 1.069 | not measured -> 1.2% (5/400) |
| bullwhip metric | 797,902 | 150,755 | 1.946 | 6.227 | 15.5% (62/400) -> 0.2% (1/400) |

Pattern: both fixes substantially improve **cost** in both metric
conditions. Neither fixes -- and the bullwhip-metric condition actively
worsens on -- **bullwhip specifically**. This narrows the still-open
question in section (h.1) to something bullwhip-specific: whatever is
producing occasional erratic (not-clamped-to-100, cleanly-parsed, first-
attempt) order values seems to hurt order-to-order *variance* more than it
hurts the *cost* objective, consistent with (but not proof of) it being
the same "occasionally goes to one extreme" LLM negotiation behavior the
paper itself acknowledges, just more frequent/severe here than for Gemini.

**Refinement (2026-09-07): the "erratic values" are mostly a shared
convergence anchor, not independent agent noise.** Checked whether
downstream and upstream land on the *same* final order across all 400
bullwhip-metric sessions: **348/400 (87%) produce an identical shared
value** for both agents, despite each having a different EOQ. This isn't
itself a bug -- it's a direct match to the paper's own description of LLM
negotiation strategy ("primarily use the average strategy," Section
6.2.1): both agents see the same shared transcript and converge on one
compromise number, which is the intended behavior. The actual problem is
narrower than "occasional independent extremes": **26 of those 400
sessions (6.5%) converge on a large, implausible *shared* value instead
of a small reasonable one** (e.g. both agents ordering 50, or both
ordering 88, with EOQs around 3-4). Since both agents shift together, this
produces a correlated shock rather than one agent's noise being smoothed
by the other's stability -- plausibly worse for a bullwhip
coefficient-of-variation metric than independent per-agent noise would
be. Not yet diagnosed further: the raw final-answer *text* isn't captured
by the current `--include-transcripts` instrumentation (only the parsed
integer is), so it's not yet possible to see whether these 26 sessions
share a common trigger (e.g. the 90-token output truncation cutting the
final reply off mid-number, or a specific pattern in how the transcript's
concluding lines get summarized). Capturing the raw final-reply text
alongside the parsed value would be the next diagnostic step.

## (i) Final root cause found: decimal point stripped, not rounded (2026-09-09)

Added raw final-reply-text capture (the transcript_sink now stores
`downstream_final_reply`/`upstream_final_reply`, the literal model output,
not just the parsed integer) and reran the bullwhip-metric shard. Result:
every one of the 23 large-shared-anchor sessions has **clean, correctly-
formatted `[[N]]` output** -- this is not a parsing problem at all. The
model's reasoning is fine; the final numeric answer is not.

Pattern, confirmed across the transcripts (not a guess):

| Computed value (from the conversation) | Final `[[N]]` answer | What happened |
|---|---|---|
| 4.54 (geometric mean, shown explicitly: `sqrt(4.97*4.16) = sqrt(20.63) ~ 4.54`) | 45 | decimal point stripped |
| 4.7 | 47 | decimal point stripped |
| 3.9 (from `(3.53+4.26)/2 = 3.895`, rounded) | 39 | decimal point stripped |
| 3.4 | 34 | decimal point stripped |
| ~2.1 (from 2.105) | 21 | decimal point stripped |
| 2.64 (average) | 64 | integer part dropped, kept only the decimal digits |
| 2.83 | 83 | integer part dropped, kept only the decimal digits |

The model negotiates its way to a perfectly sensible decimal order
quantity (matching both agents' EOQs closely, e.g. ~4.5 units when both
EOQs are ~4-5), then asked for "an integer value," concatenates the
digits either side of the decimal point instead of rounding -- turning a
sensible ~3-5 unit order into a 10-20x-too-large 21-85 unit one. Both
agents make this error identically because both see the same transcript
and compute the same intermediate value (consistent with section (h)'s
87%-shared-value finding). This single mechanical bug plausibly explains
nearly all of the remaining bullwhip damage after sections (h)/(h.1)'s
fixes: not model incompetence, not a reasoning failure, a **formatting**
failure -- the instruction never actually said "round," only "provide...
an integer."

**Fix applied:** `prompts.py`'s `NEGOTIATION_FINAL_QUESTION` now explicitly
instructs rounding to the nearest whole number, with the exact observed
failure mode given as a negative example ("4.54 rounds to 5, NOT 45 or
54; 3.4 rounds to 3, NOT 34").

### (i.1) Verification (2026-09-09): fix confirmed effective, bullwhip metric itself is the remaining obstacle

Reran the bullwhip-metric shard (200 steps) with the rounding instruction
in place. Results vs. the pre-fix rerun in section (h.1)/(i):

| | Cost | Large-shared-anchor rate | Spike-to-100 |
|---|---|---|---|
| Before this fix | 150,755 | 23/400 (5.75%) | 1/400 |
| After this fix | 57,409 | 3/400 (0.75%) | 2/400 |

The decimal-stripping bug is essentially gone: every one of the 20
"normal EOQ, wrong shared value" cases catalogued in section (i) is
absent from this rerun. The only 3 remaining large-shared-anchor cases
are all the same *degenerate* pattern -- step 0 and step 1, before enough
demand history exists for `eoq_tool` to return anything but 0.00 for
either agent, making "negotiate using your EOQs as bounds" nonsensical
(bounds of [0, 0]). This is a distinct, structural, third issue, not a
recurrence of the decimal bug -- and unlike the decimal bug, it happens
on *every* run, at the same 1-2 steps, not randomly.

**Cost keeps improving** (150,755 -> 57,409, continuing the trend from
section (h.1)'s table) but **bullwhip got worse again** (6.227 -> 8.538
in this run), despite both diagnosed bugs being genuinely, verifiably
fixed. Why: bullwhip is a coefficient-of-variation (std/mean) computed
over the *entire* 200-step order history per agent, against a very low,
tight steady-state baseline (mean ~3.4-3.8, most values clustered 2-6).
Population standard deviation is extremely sensitive to rare outliers
against such a low mean -- even 1-3 outlier events out of 200 steps can
dominate the ratio. One isolated, non-shared, non-degenerate case also
surfaced this run (step 190, EOQs 2.13/3.30, clean `[[67]]`/`[[50]]`
answers with no obvious decimal-strip explanation) -- a residual failure
mode not yet characterized, occurring far less often than either fixed
bug (1/400 this run).

**Net assessment:** both diagnosed bugs are real, fixed, and verified.
The bullwhip *metric* remains elevated mainly because it's mathematically
unforgiving of the still-nonzero (now much rarer) residual outlier rate,
not because the fixes didn't work. The clearest remaining, systematic
(not random) lead is the degenerate EOQ=[0,0] bound at simulation start
(steps 0-1) -- worth a targeted fix (e.g. skip the EOQ-bound framing, or
fall back to a sensible default, when both EOQs are 0) if pursuing this
further.

### (i.2) Third fix: skip negotiation entirely for the degenerate zero-EOQ startup case (2026-09-09)

Confirmed in `tools.py`: both `eoq_tool` and `demand_forecast_tool` return
exactly `0.0` when `demand_history` is still empty -- true at step 0 (and
occasionally step 1, before an agent has received its first observed
demand). `negotiation_intro_prompt` tells the model to "use the EOQs
shared by each agent as upper and lower bound for the negotiation"; when
both are 0.0 this is a `[0, 0]` bound, i.e. meaningless. The model doesn't
recognise this as undefined and instead guesses an arbitrary round number
-- observed as 50 on every single rerun, at the same 1-2 steps, unlike
the randomly-occurring decimal-stripping bug in section (i).

**Fix applied:** `negotiation.py`'s `make_negotiation_decision_fn` now
checks `d_tool == 0.0 and u_tool == 0.0` before invoking the negotiation
graph, and if true, skips the LLM negotiation entirely for that pair/step
and orders 0 directly. This isn't just a workaround -- with genuinely no
established demand yet, 0 is also the most defensible order, and it saves
an otherwise-wasted round of LLM calls (a full negotiation graph
invocation, ~6+ calls) on a bound that carries no information. The
transcript_sink entry for a skipped step is recorded with an empty
transcript and `skipped_degenerate_zero_eoq: true`, so this remains
visible in diagnostic output rather than silently disappearing.

Not yet re-run to confirm the effect on the aggregate bullwhip number --
expected to close out the 3 remaining large-shared-anchor cases from
section (i.1) (all of which were this exact pattern), leaving only the
one unexplained isolated case (step 190, section (i.1)) as an open
question.

### (i.3) Verification (2026-09-09/12): third fix confirmed, bullwhip plateaus at a residual baseline rate

Reran the bullwhip-metric shard with the zero-EOQ skip in place.
Confirmed working exactly as designed: 4 sessions were correctly
skipped and ordered 0 directly (`skipped_degenerate_zero_eoq: true`),
with no more step-0/1 "both agents guess 50" cases. Cost kept improving
(57,409 -> 46,973, continuing the trend across all three fixes:
150,755 -> 57,409 -> 46,973).

**Bullwhip itself is essentially unchanged** (8.538 -> 8.546). The 3
remaining large-shared-anchor cases in this run are at entirely different
steps (6, 169, 197) than any previous run, with normal, non-degenerate
EOQs -- not a recurrence of the decimal-stripping bug or the zero-EOQ
case. This matches the one unexplained case from section (i.1) (step
190): a low, but persistent, ~0.75% (roughly 3/400, consistent across the
last two independent reruns) rate of sessions where the model cleanly
outputs `[[N]]` with no format issue, but the value is simply wrong and
happens to match between both agents, for a reason not yet identified.

**Cumulative summary across all three fixes, same config
(`negotiation_tool`/large/bullwhip, 200 steps):**

| | Cost | Bullwhip | Large-shared-anchor rate |
|---|---|---|---|
| Original (no fixes) | 797,902 | 1.946 | not measured |
| After context-loss fix (h) | 150,755 | 6.227 | 62/400 (15.5%, spike-to-100 specifically) |
| After decimal-stripping fix (i) | 57,409 | 8.538 | 3/400 (0.75%) |
| After zero-EOQ fix (i.2) | 46,973 | 8.546 | 3/400 (0.75%) |

**Assessment:** three real, independently-verified bugs were found and
fixed, and cost improved by ~17x across the whole investigation
(797,902 -> 46,973). Bullwhip has not tracked this improvement and has
plateaued once the fixes ran out of *systematic* causes to remove --
what's left is a low (~0.75%), so-far-unexplained baseline rate of
clean-format-but-wrong final answers, against a coefficient-of-variation
metric mathematically sensitive enough to amplify even that residual
rate into a large aggregate number. Further reducing this would need
either a larger sample of these specific residual cases to find a common
pattern (none has been apparent in the ones inspected so far), or
accepting this as a genuine, documented behavioral difference between
Llama 3.1 and the paper's Gemini models rather than a fixable
implementation bug. Not pursued further as of this writing.

## (j) Full 25-config grid rerun with all three fixes applied (2026-09-13)

All three negotiation fixes (sections (h), (i), (i.2)) were applied
across the *entire* grid, not just isolated diagnostic shards -- every
one of the 20 LLM-driven configs was rerun at the full 200 steps.
Canonical merged results: `results/results_postfix.json` (supersedes the
pre-fix `results/results.json`, which is left in place for the historical
record). Job-ID provenance: `full2_job_ids.txt`.

**Full before/after, every LLM-driven config:**

| Framework | Tier | Metric | Cost (pre) | Cost (post) | Bullwhip (pre) | Bullwhip (post) |
|---|---|---|---|---|---|---|
| info_sharing | large | bullwhip | 69,876 | 69,880 | 0.003 | 0.004 |
| info_sharing | small | bullwhip | 63,466 | 171,613 | 0.140 | 0.771 |
| info_sharing_tool | large | bullwhip | 11,532 | 14,793 | 0.144 | 0.261 |
| info_sharing_tool | small | bullwhip | 196,027 | 75,802 | 3.681 | 3.499 |
| negotiation_tool | large | bullwhip | 801,841 | 43,420 | 1.954 | 1.721 |
| negotiation_tool | small | bullwhip | 431,788 | 216,283 | 3.485 | 6.059 |
| standalone | large | bullwhip | 31,418 | 37,931 | 0.250 | 0.255 |
| standalone_tool | large | bullwhip | 7,846 | 6,891 | 0.087 | 0.101 |
| info_sharing | large | cost | 129,050 | 128,200 | 0.001 | 0.001 |
| info_sharing | small | cost | 1,345,136 | 1,641,690 | 0.146 | 0.027 |
| info_sharing_tool | large | cost | 32,552 | 66,565 | 1.451 | 3.139 |
| negotiation_tool | large | cost | 770,118 | 175,328 | 1.383 | 2.197 |
| negotiation_tool | small | cost | 509,278 | 262,526 | 0.663 | 0.700 |
| standalone | large | cost | 71,651 | 113,678 | 0.173 | 0.131 |
| standalone_tool | large | cost | 20,038 | 34,552 | 1.143 | 2.248 |

(Omitted rows: standalone/standalone_tool small-tier bullwhip, and
info_sharing_tool/standalone_tool small-tier cost, changed <15% and add
no new information.)

**Two things stand out:**

1. **Negotiation's cost improvement is robust and large across every
   re-measurement** (isolated diagnostic reruns and this full-grid run
   alike): roughly 4-18x better than pre-fix, consistently. This part of
   the fix is solid.
2. **Negotiation's bullwhip number is itself highly run-to-run variable**,
   not just "still bad": this official full-grid run shows 1.721 for the
   large tier, while the isolated diagnostic verification in section
   (i.3) showed 8.546 for what should be the identical fixed
   configuration and fixed demand series. This is consistent with (and
   further evidence for) the section (i.3) finding -- a low (~0.75%) rate
   of rare, clean-format-but-wrong outputs, combined with temperature=0.1
   not being perfectly deterministic, means the *number of outlier
   events that happen to occur* varies run to run, and the
   coefficient-of-variation formula amplifies that variance into a large
   swing in the final aggregate number (1.7 vs 8.5 is nearly 5x, from the
   same code and config). Non-negotiation frameworks do not show this
   degree of run-to-run swing (e.g. info_sharing/large/bullwhip: 0.003 vs
   0.004, standalone/large/bullwhip: 0.250 vs 0.255) -- this instability
   appears specific to negotiation's failure mode, not a general property
   of rerunning the grid.

**Does this recover the paper's claimed pattern? No, not even at
negotiation's best-observed run.** Taking the more favorable of the two
negotiation/large bullwhip measurements (1.721, this official grid run):
negotiation is *still* the worst-bullwhip 70B framework (vs. 0.004-0.261
for every other framework) and still above the paper's "<1 is
desirable" threshold -- the same qualitative conclusion holds regardless
of which run's number is used. For cost, negotiation/large (175,328) is
still the worst cost performer among all five 70B frameworks and still
does not beat the hard tool baseline (6,697), contrary to the paper's
explicit claim that it always should, for both model tiers.

**The 8B/small-tier cost catastrophe (flagged in section (g), never
investigated) persists unchanged and is now the largest unexplained
divergence from the paper**: every 8B-tier cost config remains 8,000-
13,000x worse than the 70B-tier equivalent and 1-13x worse than the
non-LLM weak baseline (124,450) -- e.g. info_sharing/small/cost got
*worse* after the fixes (1,345,136 -> 1,641,690), since none of the three
negotiation-specific fixes touch this code path at all. This has not
been diagnosed. The paper's own Gemini Flash results explicitly state
"the only framework that underperforms the weak baseline is the
standalone LLM agent without tool" -- our replication shows every single
8B framework underperforming it, tool or no tool.

**Overall verdict:** the paper's core structural claims (monotonic
improvement with framework sophistication; negotiation as the best
framework for both metrics) do not hold in this replication, even after
three real, verified negotiation-specific bug fixes that substantially
improved negotiation's cost numbers. This is a legitimate, well-evidenced
replication outcome, not a failure to document -- but it should not be
presented as a successful reproduction of the paper's findings.

## (k) Root cause of the 8B-tier cost catastrophe: unbounded inventory growth, not a bug (2026-09-14/16)

The 8B ("small") tier's cost-metric results have been catastrophic
(500K-1.6M vs. 124,450 for the non-LLM weak baseline) since the first
full grid run in section (g), never investigated until now. Checked
whether this was a formatting/retry issue like the negotiation bugs
first: **zero `RETRY` lines** across all five 8B cost-metric shards'
logs from the section (j) full-grid run -- the model produces clean,
well-formatted `[[N]]` answers every time. This rules out a parsing bug;
the decisions themselves are simply bad.

Pulled real per-step order/inventory data for `standalone`/small/cost via
`--include-histories` (`results/diagnostic/diag_standalone_small_cost.json`).
Finding: **all three agents show unbounded, essentially monotonic
inventory growth across the full 200 steps**, against a customer demand
of only 0-20 units/step:

| Agent | Inventory @ step 0 | @ step 50 | @ step 100 | @ step 150 | @ step 199 |
|---|---|---|---|---|---|
| 0 | 10 | 2,334 | 4,938 | 7,383 | 10,077 |
| 1 | 20 | 375 | 775 | 1,050 | 1,175 |
| 2 | 20 | 255 | 495 | 970 | 1,070 |

Agent 0 (retailer) is the worst: roughly 1000x inventory growth over the
run, with backlog staying at 0 throughout (demand is always over-met, not
under-met). By the last 20 steps its ordering has locked into a rigid,
almost mechanical alternation between the hard max-order cap (100) and 0
every other step -- with inventory still climbing throughout (e.g. steps
180-199: order 100, 0, 100, 0, ... while inventory rises from 9,123 to
10,077). The model has the actual inventory level in its prompt every
single step (`prompts.py`'s `_p3`: `"Inventory level: {obs['inventory']}"`)
and is simply not using it to correct course -- this is not a missing-
information bug, the correct data is right there.

**A related, separate finding surfaced while investigating this**: this
specific shard (`standalone`/small/cost) ran anomalously slowly --
44-62 seconds per LLM call, vs. the ~7 seconds expected from the
established AVX512 Cascade Lake benchmark (12.34 tok/s, section (f)) --
consistent across two independent runs (7h12m in the full grid, 7h15m in
this diagnostic rerun, vs. ~20-30 minutes for sibling small-tier shards).
Most likely explanation, tying back to an already-documented Llama
quirk (`llm_client.py`'s module docstring): Llama 3.1 tends to emit
verbose step-by-step reasoning before reaching a bracketed answer; if
this cost-accounting task elicits close-to-full-budget (90 token)
generations on every call while other configs terminate sooner, that
alone explains both the slowdown and hints at *why* the decisions are
bad -- extended, apparently unhelpful reasoning rather than a quick,
well-calibrated answer. Not confirmed with raw response text (no
transcript-capture instrumentation exists for `standalone.py`, unlike
`negotiation.py`); would need equivalent instrumentation to confirm
directly.

**Assessment: this is not an implementation bug**, unlike the three
negotiation fixes in sections (h)-(i.2). The model is given correct,
complete information every step and simply fails to use it to prevent
runaway inventory accumulation. This is consistent with, and strong
direct evidence for, the two structural risk factors flagged before any
of this investigation began (section (b)):
1. Llama 3.1 8B Instruct is very likely not a fair capability match for
   Gemini 1.5 Flash despite both being their family's "small" tier --
   Google has never disclosed Flash's parameter count, and it is a
   commercially RLHF'd model that plausibly has far more effective
   capacity for this kind of multi-step numerical reasoning than an
   open 8B model.
2. The paper's own text states its prompts were "optimized for the
   smaller model, Gemini Flash, using manual optimization techniques" --
   this codebase's prompts reconstruct the paper's *described* structure
   but cannot reproduce empirical tuning that targeted a model this
   replication doesn't have access to, on a different model than the one
   it was tuned for.

Not yet checked: whether the 70B tier shows the same inventory-tracking
failure at a smaller magnitude (would help distinguish "small models
generally struggle with this" from "this specific model/prompt pairing
fails"), and whether other 8B frameworks (info_sharing, tool-assisted)
show the same unbounded-growth pattern or a different failure mode.

## (l) Cross-model-family ablation for the "small" tier (2026-09-16, in progress)

Section (k) narrowed the 8B-tier cost catastrophe down to a plausible
capability/prompt-tuning mismatch rather than a bug, but that's still an
inference, not a direct test. Rather than accept Llama 3.1 8B as "the"
Gemini 1.5 Flash analog permanently, this section tests whether the
paper's claimed patterns (or the failure to reproduce them) are specific
to that one model choice, or hold across different small-model
candidates -- a genuinely new experimental axis the paper itself never
considers (it tests exactly one model family, Gemini, at two sizes, and
treats its findings as general).

**First candidate: `gemma2:9b`** -- Google's own open-weight model, the
closest available sibling to Gemini 1.5 Flash (Google has never
open-sourced Gemini itself). Chosen over a purely-capability-driven pick
(e.g. Qwen2.5) because it's a more principled analog for *this specific
paper*: same company, overlapping training practices, and close enough
in size (9B vs. 8B) to preserve the size-tier gap against the 70B "large"
tier.

**Implementation**: `config.py`'s `OLLAMA_MODELS["small"]` is now
overridable via the `OLLAMA_SMALL_MODEL` environment variable (defaults
to `llama3.1:8b`, so existing behaviour/results are unaffected unless
explicitly set). `results_table.py`'s model display label now derives
from whichever model is actually configured, so output tables don't
silently mislabel a Gemma run as Llama. To run: `export
OLLAMA_SMALL_MODEL=gemma2:9b` before invoking `run_experiments.py`, or
set it in the SLURM job script for cluster runs.

### (l.1) Results: 20-step smoke test + full 200-step run, `standalone`/small/cost (2026-09-17/18)

**20-step smoke test** (`results/diagnostic/gemma_smoke_standalone_small_cost.json`):
already qualitatively different from Llama at the same checkpoint --
agents holding steady orders in the 10-20 range (tracking the 0-20 demand
range) instead of escalating toward extremes.

**Full 200-step run** (`results/diagnostic/gemma_full_standalone_small_cost.json`),
same exact config as section (k)'s Llama result, direct comparison:

| | Llama 3.1 8B (section (k)) | Gemma 2 9B | Change |
|---|---|---|---|
| Cost | 1,266,014 | 327,406 | ~3.9x better |
| Bullwhip | 0.133 | 0.371 | worse, still low |
| Agent 0 inventory @ step 199 | 10,077 (~1000x growth) | 615 (~61x growth) | ~16x less runaway |
| Wall-clock | ~7h12m | 36m (2,159s) | ~12x faster |
| Late-run order pattern | Rigid 100/0 alternation | Steady 2-3/step | Qualitatively different |

Gemma is not "fixed" -- inventory is still climbing (not fully
stabilized) and cost remains well above both the non-LLM weak baseline
(124,450) and the 70B tier's result for this same config (113,678). But
the *failure mode* is completely different: Gemma converges to a low,
stable order rate instead of oscillating between 0 and the hard order
cap, and the ~12x speedup is consistent with it not exhibiting Llama's
verbose-rambling-before-answering tendency (section (k)) to nearly the
same degree.

**This decisively answers the question section (l) set out to test**:
the 8B-tier cost catastrophe in sections (g)/(k) was a property of the
specific Llama 3.1 8B substitution, not a general property of "small"
open-weight models. Model choice matters enormously for whether this
paper's methodology reproduces sensible behavior at the small-model tier.

Not yet done: the other four frameworks and the bullwhip metric haven't
been rerun with Gemma yet, so this is one data point, not a full-grid
confirmation. A full 20-shard Gemma grid (mirroring section (j)'s
Llama/70B grid) would be needed to make a complete Table 1/Table 2
comparison.

## How to actually run this

Locally (laptop, small-scale validation only):
```
pip install -r requirements.txt

# Free pipeline validation only -- NOT meaningful results:
python run_experiments.py --mock --steps 20
```

On the HPC cluster (real runs, both tiers free/local via Ollama -- see
section (f) for the SLURM job-script patterns: proxy env vars, thread
pinning, AVX512 node constraints, one shard per SLURM job):
```
# small tier, one framework/metric shard, on an AVX512 CPU node (short partition):
python run_experiments.py --only-framework negotiation_tool --only-metric cost \
    --only-model-tier small --out results/shard_neg_cost_small.json

# large tier, same shard shape, on a GPU node (gpu/sharing partition):
python run_experiments.py --only-framework negotiation_tool --only-metric cost \
    --only-model-tier large --out results/shard_neg_cost_large.json

# after all shards finish, merge into one results file + table:
python run_experiments.py --merge "results/shard_*.json" --out results/results.json
```

Real run via OpenRouter instead (paid; only still relevant if a specific
shard is deliberately kept off local Ollama, e.g. per the open question
above):
```
export OPENROUTER_API_KEY=sk-...
python run_experiments.py --steps 20 # short pilot first, recommended
```
