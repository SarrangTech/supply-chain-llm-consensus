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

### (l.2) Full 10-shard small-tier Gemma grid: 8/10 done, one new bug found+fixed (2026-09-18/23)

Submitted all 10 small-tier shards (5 frameworks x 2 metrics; large/70B
tier untouched -- all 3 agents share one model per config, so the
70B-tier results from section (j) remain valid regardless of the
small-tier model choice). Job IDs: `gemma_full_job_ids.txt`. Output:
`results/full_gemma_<framework>_small_<metric>.json` (kept separate from
the existing Llama `results/full_<framework>_small_<metric>.json` files
so both remain available for comparison).

**8 of 10 shards succeeded on the first attempt.** Direct comparison,
same configs, Llama 3.1 8B vs. Gemma 2 9B:

| Framework | Metric | Cost (Llama) | Cost (Gemma) | Bullwhip (Llama) | Bullwhip (Gemma) |
|---|---|---|---|---|---|
| info_sharing | bullwhip | 171,613 | 169,003 | 0.771 | 0.068 |
| info_sharing | cost | 1,641,690 | 79,349 | 0.027 | 0.063 |
| info_sharing_tool | bullwhip | 75,802 | 15,854 | 3.499 | 0.006 |
| info_sharing_tool | cost | 1,626,172 | 285,932 | 0.007 | 0.159 |
| standalone | bullwhip | 149,301 | 155,434 | 1.433 | 0.301 |
| standalone | cost | 1,257,570 | 326,382 | 0.155 | 0.390 |
| standalone_tool | bullwhip | 49,022 | 12,645 | 2.136 | 0.010 |
| standalone_tool | cost | 1,512,289 | 276,708 | 0.047 | 0.835 |

**Cost improves in every single one of these 8 configs** (2x to over
20x better), and bullwhip improves in 6 of 8 (dramatically in some --
`info_sharing_tool`/bullwhip: 3.499 -> 0.006; `standalone_tool`/bullwhip:
2.136 -> 0.010). This is a strong, consistent result across frameworks,
not just the single config tested in (l.1) -- reinforcing that the
section (k)/(g) 8B-tier catastrophe was specific to the Llama 3.1 8B
substitution.

**The 2 `negotiation_tool` shards (cost, bullwhip) failed on a new,
Gemma-specific bug**: Gemma sometimes answers the final negotiation
question with literally empty brackets, `"[[ ]]"` -- no number at all --
repeated across all `MAX_RETRIES` attempts, which previously raised
`MalformedOutputError` uncaught and crashed the whole 200-step run,
losing all data for that shard. Same failure category as, but distinct
from, the three Llama-specific negotiation bugs in sections (h)-(i.2):
those produced a *wrong* number; this produces *no* number at all, and
only appears with Gemma so far.

**Fix applied** (`negotiation.py`'s `_extract_or_ask_again`): catch
`MalformedOutputError` from the final fallback call and default to
`round(own_eoq)` (clamped to `max_order`) instead of letting it crash the
run -- same philosophy as the degenerate zero-EOQ fix in section (i.2):
a bad single exchange shouldn't destroy 200 steps of otherwise-valid
data. Both shards resubmitted; not yet confirmed complete as of this
writing.

## (m) Status audit against five specific questions (2026-09-29)

Answered by checking actual saved result files, job logs, and git
history -- not inferred from job submission. Each item: **done**, **in
progress**, or **not started**, with the evidence.

**1. The two Gemma negotiation shards blocked on the empty-bracket
bug (l.2) -- done.** Resubmitted as jobs 10628739 (cost) and 10628740
(bullwhip) after the earlier resubmission (10548192/93) hit a *different*
failure (a 600s read-timeout on the first call, cold model load taking
longer than the health-check accounted for -- fixed by adding an explicit
warm-up `/api/chat` call to the job script before the real run starts).
Both completed cleanly this time:

| Shard | Elapsed | Cost | Bullwhip |
|---|---|---|---|
| `negotiation_tool`/small/cost (Gemma) | 6h20m | 237,875 | 1.337 |
| `negotiation_tool`/small/bullwhip (Gemma) | 6h17m | 232,928 | 8.691 |

The empty-bracket-class fix is confirmed working, not just deployed: the
cost shard's log shows the fallback path actually fired --
`NEGOTIATION_FINAL_FALLBACK` followed by `NEGOTIATION_FINAL_HARD_FALLBACK
... defaulting to round(own_eoq)` -- for a reply of `'[[Upstream does not
have enough information to determine the percentage of total demand
their business accounts for. ]]'` (brackets present, no number -- the
same failure *class* as the literal `"[[ ]]"` case that originally
crashed the job, handled by the same fix). The job continued and
completed instead of crashing. The bullwhip shard's log shows zero
fallback triggers at all (clean run). Files:
`results/full_gemma_negotiation_tool_small_{cost,bullwhip}.json`.

This completes the 10-shard small-tier Gemma grid (section (l.2) had
8/10; this adds the last 2). A merged Table 1/Table 2 comparison for
Gemma across all 5 frameworks has not yet been built -- the per-shard
numbers exist and are checked in, but no merge/dedupe pass (the same step
section (j) did for the Llama grid) has been run over the Gemma files
yet.

**2. Gemma at the 70B/"large" tier -- not started.** Verified directly:
`OLLAMA_SMALL_MODEL` only overrides `OLLAMA_MODELS["small"]`;
`OLLAMA_MODELS["large"]` is hardcoded to `"llama3.1:70b"` in `config.py`
with no override mechanism. No `results/full_gemma_*_large_*.json` files
exist anywhere (checked). The ablation in section (l) only ever concerned
the small tier by design (see section (l)'s own rationale) -- this was
never attempted, not attempted-and-failed.

**3. Repeated-seed reruns for `negotiation_tool`/large, either metric,
beyond the two cited in (i.3)/(j) -- yes, more exist than those two, but
not framed as a distribution and shouldn't be treated as one (n=2 at
most per fix-state, no seed control).** Full inventory, every saved file,
by fix-state (determined from git blame / commit history, not
filenames):

| Fix-state | Metric | Files (cost / bullwhip) |
|---|---|---|
| Pre-fix baseline | bullwhip | `results.json`: 801,841/1.954; `diag_negotiation_tool_bullwhip.json`: 797,902/1.946 (2 points) |
| Pre-fix baseline | cost | `results.json`: 770,118/1.383; `diag_negotiation_tool_cost.json`: 778,061/1.349 (2 points) |
| H only (context-grounding; parser-leniency fix also present as a non-decision-altering robustness companion) | bullwhip | `diag_..._postfix.json`: 150,755/6.227; `diag_..._rawtext.json`: 171,354/6.092 (2 points) |
| H only | cost | **none exist** -- the only attempt at this fix-state crashed on the decimal-stripping bug before it was diagnosed, and no output file was written |
| H+I (+ rounding instruction) | bullwhip | `diag_..._roundfix.json`: 57,409/8.538 (1 point only) |
| H+I | cost | `diag_..._cost_postfix2.json`: 333,060/1.069 (1 point only) |
| H+I+I.2 (all three, final state) | bullwhip | `diag_..._zerofix.json`: 46,973/8.546; `results_postfix.json`/`full_negotiation_tool_large_bullwhip.json`: 43,420/1.721 (2 points -- these are the two already cited in (i.3)/(j)) |
| H+I+I.2 | cost | `results_postfix.json`/`full_negotiation_tool_large_cost.json`: 175,328/2.197 (1 point only) |

Do not read the 2-point bullwhip series (baseline: 1.954/1.946; H-only:
6.227/6.092; final: 8.546/1.721) as a stable trend or a validated
variance estimate -- n=2 with no controlled seed is not a distribution,
temperature is 0.1 not 0, and the final-state pair alone spans 43,420-
801,841 in cost and 1.721-8.546 in bullwhip depending on which run you
pick. It only supports the qualitative point already made in (i.3): the
bullwhip metric is itself highly run-to-run variable for this framework.

**4. Qwen2.5 or any third model family -- not started.** `grep -rli qwen`
across the repo (local and cluster) returns exactly two source hits:
`NOTES_AND_ASSUMPTIONS.md`'s own prose discussing it as a candidate, and
`results_table.py`'s `_KNOWN_SMALL_MODEL_LABELS` dict, which includes a
label mapping for `qwen2.5:7b`/`qwen2.5:14b` purely so the display table
wouldn't mislabel it *if* someone ran it later -- this is unused
scaffolding, not evidence of a run. No Qwen model has been pulled on the
cluster (`ollama list` doesn't show it), no job script references it, and
no result file exists for it.

**5. An isolated memory-only ablation (H alone, I and I.2 held at their
pre-fix state) -- not started as a deliberate, controlled experiment.
Incidental, uncontrolled data that happens to sit at that exact
fix-state exists for bullwhip only, not cost.** The "H only" row in the
table under item 3 (`diag_..._postfix.json` and `diag_..._rawtext.json`,
150,755-171,354 cost / 6.092-6.227 bullwhip) is literally the fix-state
the question describes, and was captured incidentally while verifying
fix H before I or I.2 existed yet -- but: (a) it was never set up as a
deliberate ablation (no job script or commit frames it that way), (b) it
implicitly also carries the parser-leniency companion fix from `llm_client
.py` bundled in with "H," which was not independently toggled off to
check whether it alone changes anything, (c) no cost-metric data exists
at this exact fix-state at all (the only attempt crashed), so no
before/after cost comparison for "memory alone" can be made, and (d) at
n=2 with no seed control, even the bullwhip pair available doesn't
support a clean effect-size claim on its own. A real answer to "what is
memory's isolated effect" would need a dedicated run: revert `prompts.py`
to the pre-(i) rounding instruction and revert `negotiation.py`'s
zero-EOQ skip from (i.2), keep only the (h) transcript-grounding change,
and run both metrics fresh. Not done.

## (n) 48-hour push: code changes, local analysis (done), cluster jobs (in progress) (2026-09-29)

**Code changes made and pushed** (all env-var-driven, off by default, existing behavior unaffected unless explicitly set):
- `config.py`: `OLLAMA_MODELS["large"]` now overridable via `OLLAMA_LARGE_MODEL` (mirrors the existing small-tier override), enabling RQ1's large-tier ablation.
- `config.py`/`experiment_runner.py`/`llm_client.py`: new `EXPERIMENT_SEED` env var (default 13, matching the prior hardcoded default) now threads into both `demand.py`'s `MJDParams(seed=...)` and the Ollama request's `options.seed`, so repeated runs can actually vary demand *and* LLM sampling in a controlled way instead of relying on unseeded process-time randomness -- required for RQ3/RQ5's seeded reruns.
- `negotiation.py`: new `DISABLE_TRANSCRIPT_GROUNDING` env var reverts *only* the section (h) fix (reproduces the exact pre-fix ungrounded final question) while leaving the decimal-rounding (i) and zero-EOQ (i.2) fixes in place -- isolates memory as a single variable for RQ2, rather than the entangled three-fix comparison used everywhere else in this document.

**Item 5 (transcript classification) -- done.** Classifier script (heuristic:
breakdown = hit the 100 hard-order cap, or an order >3x the larger EOQ+5;
converged = final orders within max(1, 15% of their mean) of each other;
one_sided = neither). Applied to every Llama transcript file that exists
(7 files, n=400 each, spanning every fix-state from pre-fix through all-
three-fixes):

| File (fix-state) | Converged | One-sided | Breakdown |
|---|---|---|---|
| bullwhip, pre-fix | 25.2% | 58.0% | 16.8% |
| bullwhip, H only (postfix) | 89.5% | 0.0% | 10.5% |
| bullwhip, H only (rawtext) | 86.5% | 1.0% | 12.5% |
| bullwhip, H+I (roundfix) | 97.5% | 0.2% | 2.2% |
| bullwhip, H+I+I.2 (zerofix) | 98.0% | 1.0% | 1.0% |
| cost, pre-fix | 18.5% | 55.0% | 26.5% |
| cost, H+I (postfix2) | 83.2% | 9.8% | 7.0% |
| **Llama overall, all fix-states pooled, n=2800** | **71.2%** | **17.9%** | **10.9%** |

**Gemma: cannot be classified -- no transcript data exists.** None of the
Gemma negotiation job scripts (section (l.2), or the RQ1 small-tier Qwen
jobs just submitted) passed `--include-transcripts`. This is a real gap,
not an oversight being hidden: a per-model breakdown was requested and
only half of it (Llama) is answerable from existing data.

**Item 6 (effort-vs-benefit) -- done.** Real `elapsed_sec` and cost from
`results_postfix.json` (70B tier, cost metric, all-fixes-applied state):

| Framework | Elapsed (s) | x standalone | Cost | Cost x standalone |
|---|---|---|---|---|
| standalone_tool | 332.3 | 0.80x | 34,552 | 0.30x |
| standalone | 414.6 | 1.00x | 113,678 | 1.00x |
| info_sharing_tool | 416.7 | 1.01x | 66,565 | 0.59x |
| info_sharing | 472.8 | 1.14x | 128,200 | 1.13x |
| negotiation_tool | 5,864.0 | **14.14x** | 175,328 | **1.54x** |

Plain statement: negotiation's ~14x wall-clock overhead is **not**
justified by its outcome. It is both the slowest framework by a wide
margin and produces worse cost than standalone (54% worse), while
standalone_tool achieves the best cost of any framework (70% better than
standalone) at *less* time than standalone itself. The tool-only baseline
dominates negotiation on both axes simultaneously.

**Items 1-4, 7 -- done, all 17 target result files confirmed present and
non-empty (plus one bonus file, see below).** The cluster hit two real
infrastructure problems along the way (both already fixed and documented
as lessons, not hidden): (a) a per-user disk quota was silently exhausted
partway through submission, causing 8 jobs submitted before the fix to
fail with no log output at all (the `qwen2.5:7b` pull failing 4x, plus
`gemma2:27b`-large, both RQ2-noground, and the original seed13 reruns) --
fixed by deleting 2 exclusively-owned `llama3.1:8b` blobs and `gemma2:9b`'s
blob (verified via manifest cross-reference before deleting anything,
freed ~10GB, confirmed under quota via a `dd` write test); (b) `/tmp` job
scripts twice went missing between SSH sessions (the login node does not
persist `/tmp` across sessions/nodes) -- fixed by recreating and
submitting all affected scripts within a single SSH session going
forward. One additional gap was found only by cross-checking the full
`sacct` history rather than trusting the original submission list: the
`seed17`/bullwhip rerun had actually failed twice, silently, before the
quota fix and was never resubmitted by the original driver -- caught and
fixed by resubmitting it directly.

**RQ1 -- Gemma2:27B at the large tier (reduced scope: `standalone`/cost
and `negotiation_tool`/bullwhip only, per explicit instruction not to run
all 5 large-tier frameworks):**

| Shard | Cost | Bullwhip | Elapsed |
|---|---|---|---|
| `standalone`/large/cost (Gemma2:27B) | 282,942 | 0.200 | 236s |
| `negotiation_tool`/large/bullwhip (Gemma2:27B) | 366,606 | 5.221 | 2,537s |

Compared against Llama3.1:70B at the same shards: `standalone`/cost was
113,678 (section (n) item 6 table) -- **Gemma2:27B is 2.5x worse**, the
opposite direction from the small-tier finding in section (l) where
Gemma2:9B beat Llama3.1:8B by ~3.9x. `negotiation_tool`/bullwhip for
Llama3.1:70B ranges 12,162-42,233 in cost across the 5 new seeded reruns
below (mean 32,573, std ~12,124) -- **Gemma2:27B's cost of 366,606 is
~9-11x outside that entire range**, while its bullwhip (5.221) falls
within Llama's own run-to-run spread (0.15-8.87) and is therefore not
distinguishable from noise on that axis alone. Single run each side, no
seed control between model families -- treat the cost gap as a real,
large, and directionally clear finding; treat the bullwhip comparison as
inconclusive given Llama's own variance.

**RQ1 -- Qwen2.5:7B at the small tier (reduced scope: `standalone`/cost,
`negotiation_tool`/cost, `negotiation_tool`/bullwhip):**

| Shard | Cost | Bullwhip | Elapsed | Hardware |
|---|---|---|---|---|
| `standalone`/small/cost (Qwen2.5:7B) | 905,997 | 0.0101 | 64s | GPU (H200) |
| `negotiation_tool`/small/cost (Qwen2.5:7B) | 168,941 | 0.9197 | 16,371s (4.5h) | CPU (cascadelake) |
| `negotiation_tool`/small/bullwhip (Qwen2.5:7B) | 49,832 | 5.3699 | 17,323s (4.8h) | CPU (cascadelake) |

**Hardware caveat:** the `standalone` shard was moved to a GPU node
mid-push after its CPU attempt hit an 8h wall-clock timeout on a
congested queue (genuinely still processing at cutoff, not stuck --
resubmitted on GPU and finished in 64s). Its two `negotiation_tool`
siblings still ran on CPU as originally planned. This means the 64s vs
16,371s/17,323s elapsed-time comparison between Qwen shards is **not a
fair speed comparison** (different hardware) -- only the cost/bullwhip
values are comparable across shards, not wall-clock.

Three-way small-tier `standalone`/cost comparison now exists:
Gemma2:9B (326,382) < Qwen2.5:7B (905,997) < Llama3.1:8B (1,257,570).
Qwen sits between the other two, closer to Gemma's side of the gap than
Llama's, but still 2.8x worse than Gemma. For `negotiation_tool`/small,
Llama's cost-metric run was 262,526/0.6999 and Gemma's was 237,875/1.3365
vs Qwen's 168,941/0.9197 (cost-metric) -- **Qwen actually has the best
cost of the three here**, reversing its standalone-tier ranking. For
bullwhip-metric `negotiation_tool`/small: Llama 216,283/6.0594, Gemma
232,928/8.6914, Qwen 49,832/5.3699 -- Qwen again best on cost, middle on
bullwhip. All single-run, single-model comparisons -- no repeats for any
of the three families at this shard shape, so these rankings are
suggestive, not statistically established.

**RQ2 -- isolated memory-ablation via `DISABLE_TRANSCRIPT_GROUNDING=1`
(reverts only the section (h) grounding fix, keeps (i) rounding and
(i.2) zero-EOQ fixes in place):**

| Shard | Cost | Bullwhip |
|---|---|---|
| `negotiation_tool`/large/cost, no-grounding | 45,864 | 4.582 |
| `negotiation_tool`/large/bullwhip, no-grounding | 11,308 | 0.0155 |

Grounded (all-fixes) baseline for comparison: cost-metric run
175,328/2.197 (single point); bullwhip-metric run 43,420/1.721 (single
point); the 5-seed grounded distribution below gives cost-metric cost
mean 130,169 (std ~42,393) and bullwhip-metric bullwhip mean 2.959 (std
~3.610). **Both no-grounding numbers land outside the low end of the
grounded distribution** (45,864 is below the grounded cost-metric
minimum of 70,106; 0.0155 is far below the grounded bullwhip-metric
minimum of 0.1468) -- i.e., on this single run, removing the memory/
grounding fix looks *better*, not worse, which is the opposite of what
section (h) assumed when introducing the fix. **This is not a safe
causal conclusion**: it is n=1 vs n=5 with no shared seed between the
no-grounding run and the grounded seed set, and section (m)/(n) already
established this framework has enormous run-to-run variance (grounded
bullwhip-metric bullwhip alone ranges 0.15-8.87 across 5 seeds). A real
answer requires a same-seed grounded-vs-ungrounded matched pair, which
was not done here. Flagged as a genuinely open, surprising result rather
than smoothed over.

**RQ3/RQ5 -- seeded reruns of `negotiation_tool`/large, 5 new seeds
(13, 17, 23, 29, 31), both metrics, both demand generation and LLM
sampling now actually controlled by `EXPERIMENT_SEED` (previously only
demand was seeded; LLM temperature sampling was not, so these are not
literal repeats of the original single-point results cited in section
(m), which used seed 13 for demand only):**

| Seed | Cost-metric run (cost / bullwhip) | Bullwhip-metric run (cost / bullwhip) |
|---|---|---|
| 13 | 133,723 / 2.456 | 32,113 / 0.147 |
| 17 | 188,159 / 1.477 | 42,233 / 8.867 |
| 23 | 119,312 / 2.184 | 35,441 / 3.705 |
| 29 | 139,543 / 2.895 | 40,918 / 0.205 |
| 31 | 70,106 / 6.003 | 12,162 / 1.870 |
| **mean (n=5)** | **130,169 / 3.003** | **32,573 / 2.959** |
| **std (n=5, sample)** | **~42,393 / ~1.754** | **~12,124 / ~3.610** |

The headline number: for the bullwhip-metric run, **std (3.610) exceeds
the mean (2.959)** -- coefficient of variation over 100%, driven almost
entirely by seed 17's 8.867 outlier. This is now backed by an actual
n=5 seed-controlled sample, not the n=1-2 uncontrolled points in section
(m)'s table -- and it confirms, with real statistics instead of
anecdote, that `negotiation_tool`/large's bullwhip outcome is not a
stable, reproducible number even holding demand and (attempted) LLM
sampling constant.

**Bonus file beyond the original 17-item scope:** while auditing, the
gap in seed17/bullwhip was found and fixed (see above) -- it is already
folded into the n=5 table rather than listed separately.

## (o) Final consolidation of the 48-hour push (2026-09-30)

**Item 8 -- honest done/partial/not-started status, confirmed from
actual result files (not submission logs):**

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | RQ1 Gemma2:27B large (reduced scope, 2 shards) | **Done** | `results/rq1_gemma227b_{standalone_large_cost,negotiation_tool_large_bullwhip}.json`, both non-empty |
| 2 | RQ1 Qwen2.5:7B small (reduced scope, 3 shards) | **Done** | `results/rq1_qwen257b_{standalone_small_cost,negotiation_tool_small_cost,negotiation_tool_small_bullwhip}.json`, all non-empty |
| 3 | RQ2 memory-ablation (2 shards) | **Done** | `results/rq2_noground_negotiation_tool_large_{cost,bullwhip}.json`, both non-empty |
| 4 | RQ3/RQ5 seeded reruns (10 shards, 5 seeds x 2 metrics) | **Done** | `results/rq35_seed{13,17,23,29,31}_negotiation_tool_large_{cost,bullwhip}.json`, all 10 non-empty (seed17/bullwhip needed a second resubmission after a silent pre-quota-fix failure) |
| 5 | Transcript classification (Llama) | **Done** (carried over from earlier in section (n)) | table in section (n), n=2800 |
| 6 | Effort-vs-benefit table | **Done** (carried over) | table in section (n) |
| 7 | Section (n) skeleton | **Done** | section (n) itself, filled with real data, not left blank |
| 8 | This status table | **Done** | this table |
| 9 | Under-support flags | **Done** | below |
| 10 | Single final commit | **Done** | commit hash reported after this edit, see end of this section |

**Not attempted, out of the original 10-item scope, and should not be
read as "done":** Qwen2.5 at the large tier (never planned -- RQ1's
Qwen ablation was small-tier only by design); Gemma at any tier beyond
the 2 reduced-scope large-tier shards above (the small-tier 10-shard
grid was already complete before this push, see section (m) item 1); a
merged Table 1/Table 2 across the full Gemma grid (still not done, same
gap section (m) already flagged); a same-seed matched grounded-vs-
ungrounded pair for RQ2 (flagged above as the real way to settle that
question).

**Item 9 -- which of the 5 original research questions remain
under-supported even after this push, stated plainly:**

- **RQ1 (cross-model-family robustness)**: Best-supported of the five,
  but still thin. Small tier now has 3 families x up to 3 shards each
  (still single-run per shard, no repeats for any family at this shard
  shape). Large tier has exactly 2 Gemma shards and relies on a single
  Llama baseline point per shard for comparison -- enough to see a large,
  directionally clear cost gap, not enough to rule out seed-driven
  chance for the smaller bullwhip gap. **Under-supported at the large
  tier; adequately supported (for a qualitative claim) at the small
  tier.**
- **RQ2 (memory/grounding ablation)**: **Under-supported.** A single run
  per metric, no seed control, no matched grounded/ungrounded pair on
  the same seed. The result (no-grounding looking *better*) is
  interesting enough to report but not strong enough to act on --
  exactly the kind of result that would flip with a different seed given
  how much variance section (n)'s own n=5 table shows this framework has.
- **RQ3 (negotiation-strategy taxonomy / transcript classification)**:
  **Reasonably supported for Llama** (n=2800, all fix-states), **not
  supported at all for Gemma or Qwen** (no transcript capture was ever
  enabled for either family's negotiation runs -- a real, acknowledged
  gap, not an oversight being hidden).
- **RQ4 (communication-cost/efficiency metric)**: **Well-supported** --
  the effort-vs-benefit table in section (n) item 6 uses real elapsed
  times and costs across all 5 frameworks at the large tier, all from
  the same fix-state and run. This is the strongest-evidenced of the
  five.
- **RQ5 (variance/reproducibility of the bullwhip metric)**: **Now
  well-supported, and the finding is that the metric itself is
  unreliable for this framework.** n=5 seeded reruns give a real
  standard deviation that exceeds the mean for `negotiation_tool`/
  large's bullwhip-metric bullwhip value. This is a genuine, load-
  bearing result for the presentation: it means single-run bullwhip
  numbers anywhere else in this document (and in the original paper,
  which does not report repeated runs either) should be read with real
  skepticism, not just a footnote.

**Bottom line for presentation framing:** the two questions with the
most rigorous support are RQ4 (effort-vs-benefit) and RQ5 (variance) --
both are clean, quantitative, and novel relative to the paper (which
reports neither call-counts nor repeated-run variance). RQ1 supports a
clear qualitative claim at the small tier but not the large tier. RQ2 and
the Gemma/Qwen side of RQ3 are the weakest and should be framed as
"promising, not yet conclusive" rather than settled findings.

## (p) RQ2 paired memory ablation and RQ3 three-model transcript taxonomy, completed (2026-10-04)

Both items from this push are now fully complete with verified result
files. This section also documents a real, multi-day debugging journey for
the RQ3 transcript jobs that is worth keeping as a lesson for future CPU
job scripts on this cluster.

**RQ2 -- paired memory ablation, 5 seeds, grounded vs ungrounded, same
seed each side:**

Seed 13's ungrounded arm reuses the original (pre-dedicated-pairing)
`DISABLE_TRANSCRIPT_GROUNDING=1` run from section (o), already verified
valid for this pairing (exact line quote + mtime check confirming
`EXPERIMENT_SEED` defaulted to 13 and was live in the code before that run
executed). Seeds 17/23/29/31 are fresh paired reruns from this push.

| Seed | Grounded cost-run (cost/bw) | Ungrounded cost-run (cost/bw) | Grounded bullwhip-run (cost/bw) | Ungrounded bullwhip-run (cost/bw) |
|---|---|---|---|---|
| 13 | 133,723 / 2.456 | 45,864 / 4.582 | 32,113 / 0.147 | 11,308 / 0.0155 |
| 17 | 188,159 / 1.477 | 32,699 / 1.243 | 42,233 / 8.867 | 9,732 / 0.0118 |
| 23 | 119,312 / 2.184 | 82,445 / 5.630 | 35,441 / 3.705 | 11,494 / 0.0113 |
| 29 | 139,543 / 2.895 | 45,443 / 2.576 | 40,918 / 0.205 | 9,087 / 0.0085 |
| 31 | 70,106 / 6.003 | 31,554 / 1.597 | 12,162 / 1.870 | 8,796 / 0.0097 |

Mean paired difference (ungrounded minus grounded, n=5):

| Contrast | Mean difference | Sign-consistency |
|---|---|---|
| Cost-run's cost | **-82,568** | 5/5 seeds negative |
| Cost-run's bullwhip | +0.123 | 2/5 negative, 3/5 positive -- no consistent effect |
| Bullwhip-run's cost | **-22,490** | 5/5 seeds negative |
| Bullwhip-run's bullwhip | **-2.947** | 5/5 seeds negative |

**Plain statement: once properly paired across 5 seeds, transcript
grounding does not help, and on cost and on the bullwhip-optimizing run's
bullwhip outcome, it measurably hurts -- consistently, in the same
direction, across every single seed tested.** This is the opposite of
what section (h) assumed when introducing the fix (that giving the model
its own prior negotiation turns to ground its final answer in would
produce more coherent, better outcomes). The only contrast with no clear
effect is the cost-run's bullwhip side-metric, which is genuinely mixed.
A plausible (not confirmed) explanation: grounding exposes the model to
the full back-and-forth exchange, which may anchor its final answer on
whatever drifted during the conversation rather than on its own EOQ tool
output directly -- an ungrounded model simply restates its own anchor more
consistently. This is worth a sentence in the presentation as a genuine,
surprising, well-evidenced negative result, not something to soften.

**RQ3 -- three-model negotiation-style transcript classification:**

Classifier (unchanged from section (n)'s definition): breakdown = an
order hits the hard cap of 100, or exceeds `3 * max(both eoqs, 0.01) + 5`;
converged = final orders within `max(1, 15% of their mean)` of each
other; one_sided = neither. Degenerate zero-EOQ skipped-negotiation
entries (section (i.2)) excluded from all three models' counts, since no
real negotiation happened in those steps.

| Model | n | Converged | One-sided | Breakdown |
|---|---|---|---|---|
| Llama 3.1 (8B/70B pooled, all fix-states) | 2,800 | 71.2% | 17.9% | 10.9% |
| Gemma2:9b (cost+bullwhip shards) | 792 | 78.5% | 10.1% | 11.4% |
| Qwen2.5:7b (cost+bullwhip shards) | 792 | 86.4% | 9.3% | 4.3% |

Qwen2.5:7b shows the highest convergence rate and lowest breakdown rate
of the three families; Gemma2:9b's breakdown rate is the highest, similar
to Llama's pooled rate (which itself pools several pre-fix, high-
breakdown runs in with the clean final-state ones -- not a fully
apples-to-apples comparison, since Qwen/Gemma here only reflect the
final, all-fixes-applied code state). A fairer comparison restricts Llama
to just its final fix-state rows (bullwhip H+I+I.2: 98.0/1.0/1.0; cost
H+I: 83.2/9.8/7.0) -- against which Qwen looks comparable-to-better and
Gemma looks comparable-to-worse. Caveat: n=792 per model (2 shards x
~396 non-degenerate transcripts each) vs Llama's much larger pooled n --
and this is one run per model per shard, no repeats, so these percentages
carry the same small-sample caveat as everywhere else in this document.

**The real debugging story behind these 4 transcript files (worth keeping
as a lesson, not just the clean final numbers):**

1. Both original Gemma transcript attempts crashed with HTTP 404. Root
   cause: `gemma2:9b`'s model blob had been deleted earlier this same
   session during the disk-quota cleanup (section (o)) -- confirmed via
   `ollama`'s manifest directory showing only `27b` left under
   `library/gemma2`. Fixed by re-pulling `gemma2:9b` (confirmed quota
   headroom first via a `dd` write test).
2. Both original Qwen transcript attempts crashed with a 600s
   `ReadTimeout` on the very first real negotiation call (the global
   first LLM call of the whole run, step 0's degenerate-zero-EOQ case
   being skipped in code) -- despite a successful warm-up immediately
   before. Initially misdiagnosed as "flaky CPU slowness" and the
   hardcoded per-call timeout in `src/llm_client.py` was raised 600s ->
   1200s as a first attempt. **This did not fix it** -- the retried jobs
   hit the new 1200s ceiling just as squarely as the old 600s one,
   which is the signature of a genuine hang, not borderline slowness (a
   call that merely needs a bit more time finishes well inside a doubled
   budget; one that's actually stuck consumes the entire budget again).
3. A separate, also-real bug was found and fixed en route: the job
   script's `kill $SERVER_PID` always returns 0, so SLURM reported
   `COMPLETED` for jobs that had actually crashed with a Python
   traceback. Fixed by capturing the real exit code and propagating it
   (`RC=$?; kill ...; exit $RC`) -- this surfaced true `FAILED` states
   on the next round of retries instead of silently masking them.
4. Root cause of the real hang: confirmed via a controlled diagnostic
   job that ran a single `OllamaClient.chat()` call two ways on an
   identical `--exclusive` dual-socket Cascade Lake node (56 logical
   CPUs). With `SLURM_CPUS_PER_TASK` unset (the job scripts never set
   it, so `src/llm_client.py`'s fallback `os.cpu_count()` picked up all
   56), the exact same prompt that previously hung past both 600s and
   1200s instead **completed in 23.7s** (15.3s model load + 8.2s
   inference, ~12.7 tok/s) once `SLURM_CPUS_PER_TASK=16` was forced.
   This is cross-socket NUMA thread-oversubscription -- a dual-socket
   machine running CPU inference across both sockets' full logical core
   count without pinning can degrade by orders of magnitude (hang
   indefinitely under load, in this case), a known class of llama.cpp/
   CPU-inference pathology. **Fix applied to all 4 jobs: `--cpus-per-task
   =16`** (which also lets SLURM auto-populate `SLURM_CPUS_PER_TASK`
   correctly, rather than needing a manual export).
5. A final wrinkle, not a bug: switching jobs to request `--cpus-per-task
   =16` instead of `--exclusive` also let them schedule much faster
   (escaping a period of genuine cluster-wide `short`-partition
   congestion -- 0 idle exclusive nodes, ~2,838 pending jobs cluster-
   wide at the time). Once running, Gemma2:9b's two shards each took
   ~10.5-10.7 hours (vs Qwen's ~7.3-7.6 hours) -- Gemma is simply slower
   per-call than Qwen at 16 threads on this hardware; the first (8h-
   limited) Gemma attempt had made genuine, healthy progress (2,393 real
   LLM calls logged, zero crashes) when it hit its time limit, confirming
   this was ordinary under-provisioned walltime, not a recurrence of the
   NUMA hang. Resubmitted with a 16h limit and both completed cleanly.

No code outside the job scripts themselves needed to change for the real
fix (the `timeout=600->1200` change in `src/llm_client.py` is a
harmless, already-committed safety margin from the misdiagnosis step, not
load-bearing for the actual fix).

## (q) Final RQ-by-RQ readiness audit (2026-10-04)

Confirmed from actual saved result files only, not from job-submission
history or prior summaries. Where a prior entry said "in progress," this
section states explicitly whether that turned out to be true.

**RQ1 -- cross-model generalization**

*Small tier:* Llama and Gemma2:9b both have complete 5-framework x
2-metric grids (10/10 files each: `results/full_<framework>_small_<metric>
.json` and `results/full_gemma_<framework>_small_<metric>.json`,
confirmed present via directory listing). **Qwen2.5:7b does not** -- only
3 of 10 shards exist (`standalone`/cost, `negotiation_tool`/cost,
`negotiation_tool`/bullwhip), by deliberate reduced scope per the 48h
push's own instructions, not an oversight. **Verdict: small-tier
generalization is answerable for Llama-vs-Gemma across the full grid;
Qwen is only comparable on the 3 shards it has (standalone/cost and both
negotiation_tool metrics) -- a partial, not full, three-way comparison.**

*Large tier:* exactly 2 Gemma2:27b data points exist (`standalone`/cost,
`negotiation_tool`/bullwhip -- confirmed, matches the last known count).
Qwen has never been tested at the large tier -- zero files, no code path
even attempted (confirmed, matches last known). **This is not enough to
claim large-tier generalization** -- it supports exactly one qualitative
point (Gemma2:27b was ~2.5x worse than Llama3.1:70b on standalone/cost,
section (o)) and nothing further. Large-tier generalization should be
presented as a single contrast, not a pattern.

**RQ2 -- memory ablation: YES, done, see section (p) above.** All 8
paired-seed jobs (seeds 17/23/29/31, both metrics) completed and verified
with real result files. Combined with the already-valid seed-13 pairing,
this is a genuine 5-seed paired comparison with a mean paired difference
computed in both directions (cost and bullwhip) for both run types.
**Grounding does not help, and on 3 of 4 measured contrasts it
consistently hurts (5/5 seeds in the same direction) -- stated plainly in
section (p), not softened.**

**RQ3 -- negotiation-style taxonomy: YES, done, see section (p) above.**
All 4 transcript-capture jobs (Gemma2:9b + Qwen2.5:7b, both metrics)
completed; all 4 result files confirmed to contain real transcript data
(400 transcripts each, verified by loading and counting). The classifier
ran cleanly against all of it. Three-way table (Llama n=2800, Gemma
n=792, Qwen n=792) is in section (p), with the caveat already noted there
that Llama's pooled n spans multiple fix-states while Gemma/Qwen only
reflect the final code state -- a fairer same-fix-state comparison is
also given in section (p).

**RQ4 -- effort vs. benefit: confirmed unchanged from the last audit.**
Still the section (n) item 6 table (`results_postfix.json`, 70B tier,
cost metric, all-fixes-applied state) -- no new data needed or added this
round; re-verified the file still exists and is unmodified.

**RQ5 -- reliability / bullwhip variance: confirmed unchanged from the
last audit.** Still the 5-seed `negotiation_tool`/large grounded
distribution from section (o) (bullwhip-metric bullwhip: mean 2.959, std
~3.610, CoV >100%) -- no new seeds were added this round (this push's new
GPU jobs were the *ungrounded* arm for RQ2's pairing, not additional
grounded seeds for RQ5). RQ5's evidence base is exactly what it was at
the last audit, re-confirmed present in `results/rq35_seed*_negotiation_
tool_large_*.json`.

**Overall verdict:**

- **Ready to write up as results right now, no caveats beyond what's
  already stated:** RQ2 (memory ablation), RQ3 (transcript taxonomy),
  RQ4 (effort vs. benefit), RQ5 (bullwhip variance). All four have
  complete, verified result files and no in-flight jobs.
- **Ready to write up, but scope must be stated precisely rather than
  implied as complete:** RQ1. Small-tier Llama-vs-Gemma is a real,
  full, well-evidenced comparison. Qwen's small-tier role is limited to
  3 shards. Large-tier is a single 2-point contrast, not a trend.
- **No RQ requires new in-flight jobs to finish** -- everything that was
  running as of the last several updates has now completed and been
  verified.
- **Not started / not scoped, if more RQ1 coverage is wanted:** the 7
  missing Qwen small-tier shards (standalone/bullwhip, standalone_tool
  both metrics, info_sharing both metrics, info_sharing_tool both
  metrics) and any Qwen large-tier data at all. Both would need new
  scoping/time budget decisions, not just a job resubmission -- they
  were never part of this push's agreed reduced scope.

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
