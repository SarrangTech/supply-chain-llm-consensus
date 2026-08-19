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

## How to actually run this

```
pip install -r requirements.txt

# Free pipeline validation only -- NOT meaningful results:
python run_experiments.py --mock --steps 20

# Real run (costs money, calls OpenRouter):
export OPENROUTER_API_KEY=sk-...
python run_experiments.py            # full 200-step x 25-config grid
python run_experiments.py --steps 20 # short pilot first, recommended
```
