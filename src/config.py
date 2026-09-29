"""
Fixed experimental parameters, reproduced from Jannelli et al. (2025),
Appendix 6, Table 2, unless otherwise flagged.

Every value in FIXED_PARAMS is stated explicitly in the paper. Everything else
in this file (model IDs, MJD demand parameters) is a deviation or an inference
and is flagged inline and in NOTES_AND_ASSUMPTIONS.md.
"""
import os

# ---------------------------------------------------------------------------
# Appendix 6, Table 2 -- stated exactly in the paper.
# ---------------------------------------------------------------------------
FIXED_PARAMS = {
    "num_agents": 3,                 # Retailer (0), Distributor (1), Manufacturer (2)
    "num_negotiation_iters": 3,      # "rounds of communication" / num_iter
    "memory_window": 10,             # previous observations kept in agent memory
    "max_order_amount": 100,         # hard upper bound, enforced in code + prompt
    "lead_time": 2,                  # steps between order and delivery
    "temperature": 0.1,
    "max_output_tokens": 90,
    "inventory_cost": 1,
    "backlog_cost": 1,
    "ordering_cost": 1,              # variable, per-unit
    "fixed_ordering_cost": 1,        # fixed, charged only if order_qty > 0
    "demand_forecast_lookback": 30,  # observations used by the linear-regression tool
    "num_steps": 200,
}

# RQ3/RQ5 seed-controlled reruns (2026-09-29): overridable via EXPERIMENT_SEED
# so repeated runs can actually vary demand + LLM sampling instead of relying
# on unseeded process-time randomness (which isn't a controlled experiment).
# Default (13) matches demand.py's prior hardcoded default -- existing
# single-seed results are unaffected unless this is explicitly set.
EXPERIMENT_SEED = int(os.environ.get("EXPERIMENT_SEED", 13))

# (S, s) restocking policy baseline (Section 5.1): reorder up to S when
# inventory falls below s.
SS_POLICY = {"S": 100, "s": 60}

# ---------------------------------------------------------------------------
# DEVIATION FROM THE PAPER (explicit, user-approved):
# The paper uses Gemini 1.5 Flash / Gemini 1.5 Pro. No Gemini API key is
# available in this environment, and the user asked to substitute open-source
# models instead: Llama 3.1 8B Instruct standing in for the "small/fast"
# model, and Llama 3.1 70B Instruct standing in for the "large/capable"
# model. This is a like-for-like size-tier substitution within one model
# family, matching the paper's Flash-vs-Pro comparison *structurally*, but
# the numbers this produces are NOT the paper's numbers and are not claimed
# to be. See NOTES_AND_ASSUMPTIONS.md.
#
# FURTHER DEVIATION (2026-08-26, explicit, user-approved): the 70B/"large"
# tier needs ~40GB+ RAM even quantized, which this machine's 15.6GB doesn't
# have -- it's a hard capacity ceiling, not a speed problem (see
# NOTES_AND_ASSUMPTIONS.md and REPORT.md for the full hardware check). While
# waiting on more compute from the user's university, the "small" tier runs
# LOCALLY via Ollama (free, no rate limits, no API key needed), and the
# "large" tier stays routed to OpenRouter (paid, blocked on account credit)
# until bigger hardware is available. BACKENDS below is the single place
# that controls this routing -- flip a tier's entry here once circumstances
# change, nothing else needs to know.
# ---------------------------------------------------------------------------
MODELS = {
    "small": "meta-llama/llama-3.1-8b-instruct",   # analog of Gemini 1.5 Flash (OpenRouter id)
    "large": "meta-llama/llama-3.1-70b-instruct",  # analog of Gemini 1.5 Pro (OpenRouter id)
}

# Ollama uses different model-name strings than OpenRouter for the same
# underlying weights.
#
# ABLATION (2026-09-16, see NOTES_AND_ASSUMPTIONS.md section (l)): the "small"
# tier's model is overridable via OLLAMA_SMALL_MODEL so the same grid can be
# rerun against a different candidate model without touching this file --
# used to test whether the paper's claimed patterns are specific to the
# Llama 3.1 8B substitution or hold across model families (e.g. Gemma 2 9B,
# Google's own open model and the closest available sibling to Gemini 1.5
# Flash). Unset by default, so normal runs are unaffected.
OLLAMA_MODELS = {
    "small": os.environ.get("OLLAMA_SMALL_MODEL", "llama3.1:8b"),
    "large": os.environ.get("OLLAMA_LARGE_MODEL", "llama3.1:70b"),
}

# FURTHER DEVIATION (2026-08-30, explicit, user-approved): moved from the
# laptop above to the user's university HPC cluster (SLURM), which has both
# real GPUs (V100/A100/H200) and, importantly, CPU nodes with AVX512/VNNI
# (Cascade Lake / Skylake-AVX512) -- measured 8B throughput went from
# 0.39 tok/s (Zen2 CPU node, no AVX512) to 12.34 tok/s (Cascade Lake) to
# ~11.5 tok/s (V100 GPU, Vulkan-fallback since this cluster's driver is too
# old for Ollama's CUDA path). 70B was measured at ~0.04 tok/s on CPU (not
# viable -- a single 90-token reply would take ~37 minutes) vs. full CUDA
# GPU offload on H200 (fast, 100% GPU). Both tiers now route to Ollama:
# "small" preferentially to abundant AVX512 CPU nodes (no GPU queue wait,
# 2-day walltime), "large" to GPU nodes only (CPU is a non-starter for 70B).
BACKENDS = {
    "small": "ollama",
    "large": "ollama",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
# Overridable via env var so multiple SLURM jobs sharing one physical GPU
# node (each requesting a single GPU out of the node's several) can each run
# their own `ollama serve` on a distinct port instead of colliding on the
# default 11434 -- see NOTES_AND_ASSUMPTIONS.md section (f) for the port-
# collision incident this fixes (two co-located jobs silently shared one
# Ollama instance; one lost its results when the other job finished and
# killed the process it didn't know it depended on).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

METRICS = ("global_cost", "global_bullwhip")

FRAMEWORKS = (
    "standalone",
    "info_sharing",
    "standalone_tool",
    "info_sharing_tool",
    "negotiation_tool",
)
