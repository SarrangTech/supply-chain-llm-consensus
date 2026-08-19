"""
Fixed experimental parameters, reproduced from Jannelli et al. (2025),
Appendix 6, Table 2, unless otherwise flagged.

Every value in FIXED_PARAMS is stated explicitly in the paper. Everything else
in this file (model IDs, MJD demand parameters) is a deviation or an inference
and is flagged inline and in NOTES_AND_ASSUMPTIONS.md.
"""

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

# (S, s) restocking policy baseline (Section 5.1): reorder up to S when
# inventory falls below s.
SS_POLICY = {"S": 100, "s": 60}

# ---------------------------------------------------------------------------
# DEVIATION FROM THE PAPER (explicit, user-approved):
# The paper uses Gemini 1.5 Flash / Gemini 1.5 Pro. No Gemini API key is
# available in this environment, and the user asked to substitute open-source
# models served via OpenRouter instead: Llama 3.1 8B Instruct standing in for
# the "small/fast" model, and Llama 3.1 70B Instruct standing in for the
# "large/capable" model. This is a like-for-like size-tier substitution
# within one model family, matching the paper's Flash-vs-Pro comparison
# *structurally*, but the numbers this produces are NOT the paper's numbers
# and are not claimed to be. See NOTES_AND_ASSUMPTIONS.md.
# ---------------------------------------------------------------------------
MODELS = {
    "small": "meta-llama/llama-3.1-8b-instruct",   # analog of Gemini 1.5 Flash
    "large": "meta-llama/llama-3.1-70b-instruct",  # analog of Gemini 1.5 Pro
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

METRICS = ("global_cost", "global_bullwhip")

FRAMEWORKS = (
    "standalone",
    "info_sharing",
    "standalone_tool",
    "info_sharing_tool",
    "negotiation_tool",
)
