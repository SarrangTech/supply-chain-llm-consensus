# Results summary, by research question

Single entry point for writing the paper's results/methodology sections.
Every number below was re-verified directly from its source file on
2026-10-06 (not pulled from conversation memory) -- see "Final number
re-verification" at the bottom of this file for the exact recomputation
and any discrepancies found.

For full methodology, per-seed tables, and raw evidence behind any line
here, see the NOTES_AND_ASSUMPTIONS.md section(s) listed.

---

## RQ1 -- Cross-model generalization

**Question:** Does the paper's pattern hold across model families (Llama,
Gemma, Qwen), or is it specific to one substituted model?

**Verdict:** Small tier: a full, well-evidenced 2-model comparison
(Llama vs. Gemma, all 5 frameworks x 2 metrics each) plus a 3-shard Qwen
subset. Gemma2:9b beats Llama3.1:8b by ~3.9x on standalone/cost
(326,382 vs. 1,257,570); the ranking reverses for `negotiation_tool`,
where Qwen2.5:7b has the best cost of the three. Large tier: only a
single Gemma-vs-Llama contrast exists (2 Gemma data points total); Qwen
was never attempted at the large tier at all. **The small-tier claim is
solid; the large-tier claim is a single data point, not a trend, and must
be presented as such.**

- **Result files:** `results/full_<framework>_small_<metric>.json` (10,
  Llama), `results/full_gemma_<framework>_small_<metric>.json` (10,
  Gemma), `results/rq1_qwen257b_{standalone_small_cost,negotiation_tool_
  small_cost,negotiation_tool_small_bullwhip}.json` (3, Qwen -- **3/10
  small-tier shards, by deliberate scope**), `results/rq1_gemma227b_
  {standalone_large_cost,negotiation_tool_large_bullwhip}.json` (2, Gemma
  large tier), **0 Qwen large-tier files (never attempted)**.
- **NOTES sections:** (l), (l.2), (m item 1), (o), (q).
- **Confidence:** solid with stated scope limits.

## RQ2 -- Memory/transcript-grounding ablation

**Question:** Does grounding the negotiation's final answer in the actual
conversation transcript change outcomes, compared to not grounding it?

**Verdict:** A proper 5-seed paired comparison (same seed, same demand,
same LLM sampling, only the grounding flag differs) shows grounding does
not help and, on 3 of 4 measured contrasts, consistently hurts (5/5
seeds in the same direction). **This splits by metric and the mechanism
differs, and neither half should be flattened into the other:**
- **Cost-run:** mean paired difference (ungrounded - grounded) is
  **-82,568 in cost** (consistent, 5/5 seeds). Follow-up mechanism check
  found this is **not actually a memory effect** -- the ungrounded
  condition's `round(own_eoq)` match rate is 97.3% (vs. 11.9% grounded)
  and two-agent agreement collapses to 30.9% (vs. 74.8% grounded). The
  ungrounded cost-run isn't negotiating without memory, it's not
  negotiating at all -- each agent independently reports its own EOQ.
  **Correct framing: "an undisguised EOQ decision beats actual negotiation
  on cost," not "memory hurts negotiation."**
- **Bullwhip-run:** mean paired difference is **-2.947 in bullwhip**
  (5/5 seeds) and **-22,490 in cost** (5/5 seeds). The same mechanism
  check here is genuinely ambiguous: the grounded baseline *already*
  shows 79.3% round(own_eoq) match and 93.2% agreement (vs. 99.2%/83.7%
  ungrounded) -- a difference of degree, not of kind. **No clean
  memory-vs-no-memory story can be told for the bullwhip-run; this is
  reported as an open question, not resolved.**

- **Result files:** `results/rq35_seed{13,17,23,29,31}_negotiation_tool_
  large_{cost,bullwhip}.json` (10, grounded), `results/rq2_noground_
  negotiation_tool_large_{cost,bullwhip}.json` (2, ungrounded seed 13),
  `results/rq_paired_seed{17,23,29,31}_noground_negotiation_tool_large_
  {cost,bullwhip}.json` (8, ungrounded seeds 17-31), `results/mech_seed
  {13,17,23,29,31}_noground_negotiation_tool_large_{cost,bullwhip}.json`
  (10, same ungrounded runs rerun with `--include-transcripts` for the
  mechanism check only).
- **NOTES sections:** (p) for the paired table, (r) for the mechanism
  investigation.
- **Confidence:** cost-run mechanism is **solid**; bullwhip-run mechanism
  is an **open question, reported honestly**. The outcome numbers
  themselves (the mean paired differences) are solid for both.

## RQ3 -- Negotiation-style transcript taxonomy

**Question:** Does negotiation *style* (converges on a shared number vs.
breaks down vs. ends one-sided) differ by model family?

**Verdict:** Classified across all three model families using one
consistent heuristic (breakdown = hits the order cap or >3x the larger
EOQ+5; converged = within 15% of the pair's mean; one-sided = neither).
Qwen2.5:7b shows the highest convergence (86.4%) and lowest breakdown
(4.3%) of the three; Gemma2:9b has the highest breakdown rate (11.4%,
comparable to Llama's pooled rate); Llama's number pools multiple
fix-states (not an apples-to-apples comparison with the single-fix-state
Gemma/Qwen numbers -- see caveat below).

| Model | n | Converged | One-sided | Breakdown |
|---|---|---|---|---|
| Llama 3.1 (8B/70B pooled, all fix-states) | 2,796 | 71.2% | 17.9% | 10.9% |
| Gemma2:9b | 792 | 78.5% | 10.1% | 11.4% |
| Qwen2.5:7b | 792 | 86.4% | 9.3% | 4.3% |

- **Result files:** `results/diagnostic/diag_negotiation_tool_*.json` (7,
  Llama), `results/rq3_transcripts_gemma29b_negotiation_tool_small_
  {cost,bullwhip}.json` (2), `results/rq3_transcripts_qwen257b_
  negotiation_tool_small_{cost,bullwhip}.json` (2).
- **NOTES sections:** (n item 5) for Llama, (p) for the three-way table.
- **Confidence:** solid with stated scope limits (Llama's n pools
  multiple fix-states; Gemma/Qwen are single-fix-state, single-run each).

## RQ4 -- Effort vs. benefit

**Question:** Is negotiation's extra LLM-call overhead justified by
better outcomes?

**Verdict:** No. Negotiation is far slower and still produces worse cost
than the cheapest alternative. Two distinct multipliers, not to be
conflated:
- **Wall-clock multiplier:** 14.14x slower than standalone (5,864s vs.
  415s, same 70B-tier cost-metric run).
- **Call-count multiplier:** previously never isolated as its own number
  -- only the wall-clock proxy existed. Computed directly from transcript
  data for this summary: negotiation_tool makes 8 real LLM calls per
  non-degenerate pair-session (1 opening + 5 turns + 2 final decisions,
  confirmed via actual transcript length across all 5 mechanism-check
  files) x ~395/400 non-degenerate sessions per 200-step run =~3,152
  calls, vs. standalone's 3 calls/step x 200 steps = 600 calls --
  **~5.25x more LLM calls**, meaningfully lower than the 14.14x wall-clock
  figure (the remaining gap is per-call latency: negotiation's calls
  carry longer, growing conversation context).
- **Cost multiplier:** 1.54x worse cost than standalone (175,328 vs.
  113,678) despite all that extra work. `standalone_tool` achieves 70%
  *better* cost than standalone at *less* elapsed time than standalone
  itself -- it dominates negotiation on every axis simultaneously.

- **Result file:** `results/results_postfix.json` (all 5 frameworks, 70B
  tier, cost metric, all-fixes-applied state).
- **NOTES section:** (n item 6).
- **Confidence:** solid.

## RQ5 -- Bullwhip metric reliability/variance

**Question:** How reproducible is the bullwhip metric itself across
repeated runs of the identical configuration?

**Verdict:** Not reproducible in any single-run sense. 5 seeded reruns of
`negotiation_tool`/large's bullwhip-optimizing run give bullwhip values
of 0.147, 8.867, 3.705, 0.205, 1.870 -- **mean 2.959, sample standard
deviation 3.610**. The standard deviation exceeds the mean (coefficient
of variation >100%), driven almost entirely by one seed's outlier
(seed 17 at 8.867). **Any single-run bullwhip number in this document, or
in the original paper (which reports no repeated runs either), should be
read with this variance in mind.**

- **Result files:** `results/rq35_seed{13,17,23,29,31}_negotiation_tool_
  large_bullwhip.json` (5, same files that supply RQ2's grounded side).
- **NOTES section:** (o).
- **Confidence:** solid.

---

## Final number re-verification (2026-10-06)

Every figure below was recomputed directly from its source file, not
quoted from earlier in the conversation.

| Figure | Recomputed value | Status |
|---|---|---|
| RQ2 cost-run mean paired diff (cost) | -82,567.6 | confirmed, unchanged |
| RQ2 cost-run mean paired diff (bullwhip) | +0.123 | confirmed, unchanged |
| RQ2 bullwhip-run mean paired diff (cost) | -22,490.0 | confirmed, unchanged |
| RQ2 bullwhip-run mean paired diff (bullwhip) | -2.948 | confirmed, unchanged (reported earlier as -2.947; rounding-level only) |
| RQ3 Llama converged/one-sided/breakdown | 71.2% / 17.9% / 10.9%, n=2,796 | **discrepancy found and resolved**: original n=2,800 (section n) silently included 4 degenerate zero-EOQ sessions as auto-"converged" (0==0); recomputed here excluding them, consistent with the convention Gemma/Qwen's n=792 already used. Percentages unchanged at 1-decimal precision; n corrected to 2,796. |
| RQ3 Gemma converged/one-sided/breakdown | 78.5% / 10.1% / 11.4%, n=792 | confirmed, unchanged |
| RQ3 Qwen converged/one-sided/breakdown | 86.4% / 9.3% / 4.3%, n=792 | confirmed, unchanged |
| RQ4 negotiation_tool wall-clock multiplier | 14.14x | confirmed, unchanged |
| RQ4 negotiation_tool cost multiplier | 1.54x | confirmed, unchanged |
| RQ4 negotiation_tool **call-count** multiplier | **~5.25x** | **newly computed, not previously isolated** -- prior reporting only ever gave the wall-clock proxy (14.14x) under the "effort" label; a literal call-count multiplier had never been separately verified until this re-check. |
| RQ5 bullwhip mean / stdev (5 seeds) | 2.959 / 3.610 | confirmed, unchanged |
| RQ1 Llama standalone/small/cost | 1,257,570 | confirmed, unchanged |
| RQ1 Gemma standalone/small/cost | 326,382 | confirmed, unchanged |
| RQ1 Qwen small-tier coverage | 3/10 files | confirmed, unchanged |
| RQ1 Qwen large-tier coverage | 0/10 (0 files exist) | confirmed, unchanged |
